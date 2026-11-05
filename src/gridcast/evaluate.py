"""Turn saved predictions into calibrated scores, comparison tables and significance tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast import metrics as m
from gridcast.backtest import PREDICTIONS_DIR
from gridcast.config import Config
from gridcast.conformal import OnlineCQR, online_bias_offsets

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results")
ENSEMBLE_MEMBERS = ("lightgbm", "attention_net")
ENSEMBLE = "ensemble"
DISPLAY_ORDER = (
    "seasonal_naive", "ridge", "lightgbm_no_weather", "lightgbm", "attention_net", ENSEMBLE,
    "lightgbm_oracle",
)  # fmt: skip


def load_predictions(cfg: Config) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(PREDICTIONS_DIR.glob("*.parquet")):
        if path.stem != ENSEMBLE:
            tables[path.stem] = (
                pd.read_parquet(path).sort_values(["target_time"]).reset_index(drop=True)
            )
    if all(member in tables for member in ENSEMBLE_MEMBERS):
        # Averaging quantiles (Vincentisation) keeps the spread of its members, unlike a mixture.
        base = tables[ENSEMBLE_MEMBERS[0]].copy()
        for quantile in cfg.task.quantiles:
            column = f"q{quantile:.2f}"
            base[column] = np.mean([tables[k][column].to_numpy() for k in ENSEMBLE_MEMBERS], axis=0)
        tables[ENSEMBLE] = base
    return tables


def _matrix(table: pd.DataFrame, column: str, horizon: int) -> np.ndarray:
    return table[column].to_numpy(float).reshape(-1, horizon)


def conformalise(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Replay bias correction and online CQR in time order, adding calibrated interval bounds."""
    horizon = cfg.task.horizon_hours
    table = table.sort_values(["issue_time", "step"]).reset_index(drop=True)
    y = _matrix(table, "y", horizon)
    median = _matrix(table, "q0.50", horizon)
    offsets = online_bias_offsets(y, median, cfg.conformal.window_days)
    table["bias_offset"] = np.repeat(offsets, horizon)
    for quantile in cfg.task.quantiles:
        table[f"q{quantile:.2f}"] += table["bias_offset"]
    for level in cfg.conformal.levels:
        tail = (1.0 - level) / 2.0
        low = _matrix(table, f"q{tail:.2f}", horizon)
        high = _matrix(table, f"q{1.0 - tail:.2f}", horizon)
        calibrator = OnlineCQR(level, cfg.conformal.window_days, cfg.conformal.gamma, horizon)
        cal_low, cal_high = calibrator.run(y, low, high)
        table[f"cal_lo_{level:.2f}"] = cal_low.ravel()
        table[f"cal_hi_{level:.2f}"] = cal_high.ravel()
    return table


def score(table: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """All metrics for the observed test hours of one model."""
    test = table[(table["split"] == "test") & table["observed"]]
    y = test["y"].to_numpy()
    median = test["q0.50"].to_numpy()
    quantile_values = test[[f"q{q:.2f}" for q in cfg.task.quantiles]].to_numpy()
    out = {
        "mae": m.mae(y, median),
        "rmse": m.rmse(y, median),
        "mape": m.mape(y, median),
        "bias": float(np.mean(median - y)),
        "pinball": m.mean_pinball(y, quantile_values, cfg.task.quantiles),
    }
    for level in cfg.conformal.levels:
        tail = (1.0 - level) / 2.0
        low, high = test[f"q{tail:.2f}"].to_numpy(), test[f"q{1.0 - tail:.2f}"].to_numpy()
        cal_low = test[f"cal_lo_{level:.2f}"].to_numpy()
        cal_high = test[f"cal_hi_{level:.2f}"].to_numpy()
        tag = f"{int(level * 100)}"
        out[f"coverage_raw_{tag}"] = m.coverage(y, low, high)
        out[f"coverage_cal_{tag}"] = m.coverage(y, cal_low, cal_high)
        out[f"width_raw_{tag}"] = m.interval_width(low, high)
        out[f"width_cal_{tag}"] = m.interval_width(cal_low, cal_high)
        out[f"winkler_raw_{tag}"] = m.winkler(y, low, high, level)
        out[f"winkler_cal_{tag}"] = m.winkler(y, cal_low, cal_high, level)
    return out


def _daily_abs_error(test: pd.DataFrame, column: str) -> pd.Series:
    """Mean absolute error per target day: the unit used for all significance tests."""
    errors = (test["y"] - test[column]).abs()
    return errors.groupby(test["issue_time"]).mean()


def significance(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    """DM test and block-bootstrap CI for each model's MAE improvement over the TSO."""
    out: dict[str, dict[str, float]] = {}
    for name, table in tables.items():
        test = table[(table["split"] == "test") & table["observed"] & table["tso"].notna()]
        model_loss = _daily_abs_error(test, "q0.50")
        tso_loss = _daily_abs_error(test, "tso")
        result = m.diebold_mariano(model_loss.to_numpy(), tso_loss.to_numpy())
        low, high = m.block_bootstrap_ci((tso_loss - model_loss).to_numpy())
        out[name] = {**result, "improvement_ci_low": low, "improvement_ci_high": high}
    return out


def per_fold(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, table in tables.items():
        test = table[(table["split"] == "test") & table["observed"]]
        for fold, part in test.groupby("fold"):
            rows.append(
                {
                    "model": name,
                    "fold": int(fold),
                    "start": str(part["target_time"].min().date()),
                    "mae": m.mae(part["y"].to_numpy(), part["q0.50"].to_numpy()),
                    "tso_mae": m.mae(part["y"].to_numpy(), part["tso"].to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def select_deployment_model(tables: dict[str, pd.DataFrame], cfg: Config) -> dict[str, float]:
    """Pinball on the pre-test calibration span, so test data cannot influence the choice."""
    out = {}
    for name in ("lightgbm", "attention_net", ENSEMBLE):
        if name not in tables:
            continue
        table = tables[name]
        part = table[table["split"] == "calibration"]
        values = part[[f"q{q:.2f}" for q in cfg.task.quantiles]].to_numpy()
        out[name] = m.mean_pinball(part["y"].to_numpy(), values, cfg.task.quantiles)
    return out


def run_evaluation(cfg: Config) -> dict[str, object]:
    tables = {name: conformalise(t, cfg) for name, t in load_predictions(cfg).items()}
    scores = {name: score(table, cfg) for name, table in tables.items()}
    any_table = next(iter(tables.values()))
    test = any_table[
        (any_table["split"] == "test") & any_table["observed"] & any_table["tso"].notna()
    ]
    tso = {
        "mae": m.mae(test["y"].to_numpy(), test["tso"].to_numpy()),
        "rmse": m.rmse(test["y"].to_numpy(), test["tso"].to_numpy()),
        "mape": m.mape(test["y"].to_numpy(), test["tso"].to_numpy()),
        "bias": float((test["tso"] - test["y"]).mean()),
    }
    selection = select_deployment_model(tables, cfg)
    report: dict[str, object] = {
        "test_period": [str(test["target_time"].min()), str(test["target_time"].max())],
        "test_days": int(test["issue_time"].nunique()),
        "test_hours": len(test),
        "tso": tso,
        "models": scores,
        "significance_vs_tso": significance(
            {k: v for k, v in tables.items() if k != "lightgbm_oracle"}
        ),
        "per_fold": per_fold(tables).to_dict(orient="records"),
        "selection_pinball_pre_test": selection,
        "selected_model": min(selection, key=lambda key: selection[key]) if selection else None,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    (RESULTS_DIR / "model_comparison.md").write_text(render_table(report))
    return report


def render_table(report: dict) -> str:
    """Markdown table generated from metrics.json, so the README cannot drift from results."""
    tso, models = report["tso"], report["models"]
    lines = [
        "| Model | Weather at forecast time | MAE (MW) | RMSE (MW) | MAPE | Skill vs TSO | "
        "Pinball (MW) | 90% cov. (calibrated) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| TSO day-ahead forecast (SMARD) | n/a | {tso['mae']:,.0f} | {tso['rmse']:,.0f} | "
        f"{tso['mape']:.2f}% | reference | n/a | n/a |",
    ]
    weather = {
        "seasonal_naive": "none", "ridge": "archived NWP", "lightgbm_no_weather": "none",
        "lightgbm": "archived NWP", "attention_net": "archived NWP", ENSEMBLE: "archived NWP",
        "lightgbm_oracle": "hindsight ERA5 (upper bound)",
    }  # fmt: skip
    labels = {
        "seasonal_naive": "Seasonal naive", "ridge": "Ridge",
        "lightgbm_no_weather": "LightGBM, no weather", "lightgbm": "LightGBM",
        "attention_net": "Attention net (PyTorch)", ENSEMBLE: "Ensemble (LightGBM + attention)",
        "lightgbm_oracle": "LightGBM, oracle weather",
    }  # fmt: skip
    for name in DISPLAY_ORDER:
        if name not in models:
            continue
        s = models[name]
        skill = (1.0 - s["mae"] / tso["mae"]) * 100.0
        lines.append(
            f"| {labels[name]} | {weather[name]} | {s['mae']:,.0f} | {s['rmse']:,.0f} | "
            f"{s['mape']:.2f}% | {skill:+.1f}% | {s['pinball']:,.0f} | "
            f"{s['coverage_cal_90'] * 100:.1f}% |"
        )
    return "\n".join(lines) + "\n"
