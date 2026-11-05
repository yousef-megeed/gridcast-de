"""EDA, error analysis and interpretability figures for the README."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridcast import metrics as m
from gridcast.config import TIMEZONE, Config
from gridcast.data.pipeline import load_dataset
from gridcast.evaluate import RESULTS_DIR, conformalise, load_predictions
from gridcast.features import SeriesStore, calendar_frame, daily_issue_times

logger = logging.getLogger(__name__)
FIGURES = Path("reports/figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SEQUENTIAL = "Blues"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 150,
            "font.size": 10,
            "font.family": "sans-serif",
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.bottom": False,
            "ytick.left": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)


def eda_figures(frame: pd.DataFrame) -> dict[str, float]:
    local_index = frame.index.tz_convert(TIMEZONE)
    gw = frame["load"] / 1000.0

    fig, ax = plt.subplots(figsize=(9, 3.4))
    weekly = gw.resample("W").mean().iloc[1:-1]
    yearly = gw.resample("YS").mean().iloc[:-1]
    ax.plot(weekly.index, weekly, color=BLUE, alpha=0.45, linewidth=1.0, label="Weekly mean")
    ax.plot(
        yearly.index + pd.Timedelta(days=182), yearly, color=INK, marker="o", markersize=4,
        label="Annual mean",
    )  # fmt: skip
    ax.set_ylabel("Grid load (GW)")
    ax.set_title("German grid load, weekly and annual means")
    ax.legend(loc="upper right", ncols=2)
    _save(fig, "eda_load_history")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    table = gw.groupby([local_index.month, local_index.hour]).mean().unstack(0)
    image = ax.imshow(table, aspect="auto", origin="lower", cmap=SEQUENTIAL)
    ax.grid(False)
    ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(0, 24, 4))
    ax.set_xlabel("Month")
    ax.set_ylabel("Local hour")
    ax.set_title("Mean load by month and hour")
    fig.colorbar(image, ax=ax, label="GW", shrink=0.9)
    _save(fig, "eda_seasonality")

    calendar = calendar_frame(frame.index)
    workday = (calendar["is_weekend"] == 0) & (calendar["holiday_share"] < 0.5)
    daily = pd.DataFrame({"gw": gw[workday.to_numpy()], "temp": frame["temp"][workday.to_numpy()]})
    daily = daily.resample("D").mean().dropna()
    daily = daily[daily.index >= "2023-01-01"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.scatter(daily["temp"], daily["gw"], s=10, color=BLUE, alpha=0.55, linewidths=0)
    ax.set_xlabel("Demand-weighted temperature (°C)")
    ax.set_ylabel("Daily mean load (GW)")
    ax.set_title("Working days since 2023: load vs temperature")
    _save(fig, "eda_temperature")

    return {
        "mean_gw_2018": float(gw[gw.index.year == 2018].mean()),
        "mean_gw_2025": float(gw[gw.index.year == 2025].mean()),
        "temp_load_corr_workdays_2023on": float(daily["temp"].corr(daily["gw"])),
    }


def _test(table: pd.DataFrame) -> pd.DataFrame:
    return table[(table["split"] == "test") & table["observed"] & table["tso"].notna()].copy()


def comparison_figure(tables: dict[str, pd.DataFrame]) -> None:
    labels = {
        "seasonal_naive": "Seasonal naive", "ridge": "Ridge",
        "lightgbm_no_weather": "LightGBM, no weather", "lightgbm": "LightGBM",
        "attention_net": "Attention net", "ensemble": "Ensemble",
    }  # fmt: skip
    rows = [
        (
            "TSO forecast",
            *(lambda t: (m.mae(t["y"], t["tso"]), ORANGE))(_test(next(iter(tables.values())))),
        )
    ]
    for name, label in labels.items():
        if name in tables:
            test = _test(tables[name])
            rows.append((label, m.mae(test["y"], test["q0.50"]), BLUE))
    rows.sort(key=lambda row: row[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows],
                   height=0.62)  # fmt: skip
    ax.bar_label(bars, fmt="{:,.0f}", padding=4, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MAE on the test period (MW), lower is better")
    ax.set_title("Day-ahead point accuracy, walk-forward test")
    ax.margins(x=0.1)
    _save(fig, "model_comparison")


def error_profile_figures(best: pd.DataFrame, name: str) -> None:
    test = _test(best)
    test["hour"] = test["target_time"].dt.tz_convert(TIMEZONE).dt.hour
    test["model_ae"] = (test["y"] - test["q0.50"]).abs()
    test["tso_ae"] = (test["y"] - test["tso"]).abs()

    fig, ax = plt.subplots(figsize=(8, 3.3))
    by_hour = test.groupby("hour")[["model_ae", "tso_ae"]].mean()
    ax.plot(by_hour.index, by_hour["tso_ae"], color=ORANGE, label="TSO forecast")
    ax.plot(by_hour.index, by_hour["model_ae"], color=BLUE, label=f"GridCast ({name})")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Local hour of day")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Error by hour of day")
    ax.legend(loc="lower right", ncols=2)
    _save(fig, "error_by_hour")

    fig, ax = plt.subplots(figsize=(8, 3.3))
    local_time = test["target_time"].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    month = test.set_index(local_time)[["model_ae", "tso_ae"]].resample("MS").mean()
    ax.plot(month.index, month["tso_ae"], color=ORANGE, marker="o", markersize=4,
            label="TSO forecast")  # fmt: skip
    ax.plot(month.index, month["model_ae"], color=BLUE, marker="o", markersize=4,
            label=f"GridCast ({name})")  # fmt: skip
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Monthly MAE (MW)")
    ax.set_title("Monthly error across the test period")
    ax.legend(loc="lower right", ncols=2)
    _save(fig, "error_by_month")


def calibration_figure(best: pd.DataFrame, cfg: Config) -> dict[str, dict[str, float]]:
    test = best[(best["split"] == "test") & best["observed"]]
    y = test["y"].to_numpy()
    reliability = {
        f"{q:.2f}": float((y <= test[f"q{q:.2f}"].to_numpy()).mean()) for q in cfg.task.quantiles
    }
    levels = list(cfg.conformal.levels)
    raw, calibrated = [], []
    for level in levels:
        tail = (1.0 - level) / 2.0
        raw.append(m.coverage(y, test[f"q{tail:.2f}"].to_numpy(),
                              test[f"q{1 - tail:.2f}"].to_numpy()))  # fmt: skip
        calibrated.append(m.coverage(y, test[f"cal_lo_{level:.2f}"].to_numpy(),
                                     test[f"cal_hi_{level:.2f}"].to_numpy()))  # fmt: skip

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    span = [0.4, 1.0]
    ax.plot(span, span, color=INK_2, linestyle="--", linewidth=1.0, label="Perfect coverage")
    ax.plot(levels, raw, color=ORANGE, marker="o", markersize=6, label="Raw quantiles")
    ax.plot(levels, calibrated, color=BLUE, marker="s", markersize=6, label="Online CQR")
    ax.set_xlabel("Nominal central interval")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Interval calibration on the test period")
    ax.legend(loc="upper left")
    _save(fig, "calibration")
    return {"quantile_reliability": reliability}


def fan_figure(best: pd.DataFrame, start: str, days: int, filename: str, title: str) -> None:
    window_start = pd.Timestamp(start, tz=TIMEZONE)
    window_end = window_start + pd.Timedelta(days=days)
    part = best[(best["target_time"] >= window_start) & (best["target_time"] < window_end)]
    part = part.sort_values("target_time")
    time = part["target_time"].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(time, part["cal_lo_0.90"] / 1e3, part["cal_hi_0.90"] / 1e3, color=BLUE,
                    alpha=0.16, linewidth=0, label="90% interval")  # fmt: skip
    ax.fill_between(time, part["cal_lo_0.50"] / 1e3, part["cal_hi_0.50"] / 1e3, color=BLUE,
                    alpha=0.28, linewidth=0, label="50% interval")  # fmt: skip
    ax.plot(time, part["tso"] / 1e3, color=ORANGE, linewidth=1.2, label="TSO forecast")
    ax.plot(time, part["q0.50"] / 1e3, color=BLUE, label="GridCast median")
    ax.plot(time, part["y"] / 1e3, color=INK, linewidth=1.2, label="Actual load")
    ax.set_ylabel("GW")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=5)
    _save(fig, filename)


def segment_table(best: pd.DataFrame, frame: pd.DataFrame) -> list[dict[str, object]]:
    """Where the model wins and loses, including the effect of weather-forecast error."""
    test = _test(best)
    calendar = calendar_frame(pd.DatetimeIndex(test["target_time"]))
    test["segment"] = np.select(
        [
            calendar["holiday_share"].to_numpy() >= 0.5,
            calendar["bridge_day"].to_numpy() == 1,
            calendar["winter_break"].to_numpy() == 1,
            calendar["holiday_share"].to_numpy() > 0,
            calendar["is_weekend"].to_numpy() == 1,
        ],
        ["National holiday", "Bridge day", "Christmas period", "Regional holiday", "Weekend"],
        default="Normal working day",
    )
    rows = []

    def add(label: str, part: pd.DataFrame) -> None:
        rows.append(
            {
                "segment": label,
                "hours": len(part),
                "model_mae": m.mae(part["y"], part["q0.50"]),
                "tso_mae": m.mae(part["y"], part["tso"]),
                "model_bias": float((part["q0.50"] - part["y"]).mean()),
            }
        )

    for label, part in test.groupby("segment"):
        add(str(label), part)
    error = (frame["temp_nwp"] - frame["temp"]).abs().reindex(test["target_time"]).to_numpy()
    edges = np.nanquantile(error, [1 / 3, 2 / 3])
    for label, mask in (
        (f"Temp. forecast error < {edges[0]:.1f} °C", error < edges[0]),
        (f"Temp. forecast error ≥ {edges[1]:.1f} °C", error >= edges[1]),
    ):
        add(label, test[mask])
    return rows


def worst_days(best: pd.DataFrame, count: int = 10) -> list[dict]:
    test = _test(best)
    test["ae"] = (test["y"] - test["q0.50"]).abs()
    test["tso_ae"] = (test["y"] - test["tso"]).abs()
    test["signed"] = test["q0.50"] - test["y"]
    test["day"] = test["target_time"].dt.tz_convert(TIMEZONE).dt.date
    daily = test.groupby("day").agg(
        model_mae=("ae", "mean"), tso_mae=("tso_ae", "mean"), bias=("signed", "mean")
    )
    daily = daily.sort_values("model_mae", ascending=False).head(count).reset_index()
    daily["weekday"] = pd.to_datetime(daily["day"]).dt.day_name()
    daily["day"] = daily["day"].astype(str)
    return daily.round(0).to_dict(orient="records")


def shap_figure(cfg: Config, bundle: Path) -> list[dict]:
    from gridcast.models.gbm import GBMForecaster

    store = SeriesStore(load_dataset(cfg))
    model = GBMForecaster.load(bundle / "lightgbm", cfg.task, cfg.gbm)
    issues = daily_issue_times(cfg.backtest.fold_starts[-1], cfg.backtest.end, cfg.task)
    features, values = model.contributions(store, store.positions(issues), "nwp")
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[-12:]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.barh(np.array(features.columns)[order], importance[order] * 100, color=BLUE,
                   height=0.62)  # fmt: skip
    ax.bar_label(bars, fmt="{:.2f}", padding=4, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    ax.set_xlabel("Mean |SHAP| (% of the weekly load level)")
    ax.set_title("LightGBM median: top feature attributions")
    _save(fig, "shap_importance")
    output = [
        {"feature": str(features.columns[i]), "mean_abs_shap_pct": float(importance[i] * 100)}
        for i in order[::-1]
    ]
    (RESULTS_DIR / "shap_importance.json").write_text(json.dumps(output, indent=2))
    return output


def attention_figure(cfg: Config, bundle: Path) -> None:
    from gridcast.models.neural import NeuralForecaster

    store = SeriesStore(load_dataset(cfg))
    model = NeuralForecaster.load(bundle / "attention_net", cfg.task, cfg.nn)
    issues = daily_issue_times(cfg.backtest.fold_starts[-1], cfg.backtest.end, cfg.task)
    attention = model.attention(store, store.positions(issues), "nwp").mean(axis=0)
    context, patch = model.cfg.context_hours, model.cfg.patch_hours
    fig, ax = plt.subplots(figsize=(9, 3.2))
    image = ax.imshow(attention, aspect="auto", origin="lower", cmap=SEQUENTIAL)
    ax.grid(False)
    ticks = np.arange(0, attention.shape[1], max(1, 24 // patch))
    ax.set_xticks(ticks, [f"-{(context - t * patch) // 24}d" for t in ticks])
    ax.set_xlabel("Past patch, relative to the issue time")
    ax.set_ylabel("Target step (local hour)")
    ax.set_title("Cross-attention of the last decoder layer, averaged over the latest fold")
    fig.colorbar(image, ax=ax, label="Attention weight", shrink=0.9)
    _save(fig, "attention_map")


def run_analysis(cfg: Config, part: str = "all", bundle: Path = Path("artifacts/model")) -> None:
    _style()
    if part == "shap":
        # LightGBM may not be deployed, so it is trained separately for explanations.
        shap_figure(cfg, Path("artifacts/explain"))
        return
    if part == "attention":
        attention_figure(cfg, bundle)
        return

    frame = load_dataset(cfg)
    tables = {name: conformalise(table, cfg) for name, table in load_predictions(cfg).items()}
    report = json.loads((RESULTS_DIR / "metrics.json").read_text())
    selected = report["selected_model"]
    best = tables[selected]

    output = {
        "selected_model": selected,
        "eda": eda_figures(frame),
        "calibration": calibration_figure(best, cfg),
        "segments": segment_table(best, frame),
        "worst_days": worst_days(best),
    }
    comparison_figure(tables)
    error_profile_figures(best, selected)
    fan_figure(best, "2026-10-12", 7, "forecast_normal_week", "Normal October week, 2026")
    fan_figure(best, "2025-12-22", 7, "forecast_christmas_week", "Christmas week, 2025")
    (RESULTS_DIR / "error_analysis.json").write_text(json.dumps(output, indent=2, default=str))
