"""Package the deployment model as a self-describing, versioned directory."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast import __version__
from gridcast.backtest import MLFLOW_URI, VARIANTS, make_model
from gridcast.config import Config, config_hash
from gridcast.conformal import OnlineCQR, online_bias_offsets
from gridcast.data.pipeline import load_dataset
from gridcast.evaluate import ENSEMBLE, ENSEMBLE_MEMBERS, RESULTS_DIR, load_predictions
from gridcast.features import SeriesStore
from gridcast.integrity import sha256

logger = logging.getLogger(__name__)


def members_of(selected: str) -> tuple[str, ...]:
    return ENSEMBLE_MEMBERS if selected == ENSEMBLE else (selected,)


def fit_member(cfg: Config, member: str, output: str) -> None:
    """Train one member on all data. Runs in its own process via ``build_bundle``."""
    variant = VARIANTS[member]
    store = SeriesStore(load_dataset(cfg))
    end = pd.Timestamp(cfg.backtest.end, tz="UTC")
    train_end = end - pd.Timedelta(days=cfg.backtest.val_days)
    iterations = make_model(variant, cfg).fit(store, train_end, end).best_iterations
    model = make_model(variant, cfg).fit(store, end, None, iterations)
    model.save(Path(output) / member)
    logger.info("Saved %s after %s iterations", member, iterations)


def conformal_state(table: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Replay the online calibrator over all forecasts to get today's margins per step."""
    horizon = cfg.task.horizon_hours
    table = table.sort_values(["issue_time", "step"])
    y = table["y"].to_numpy(float).reshape(-1, horizon)
    median = table["q0.50"].to_numpy(float).reshape(-1, horizon)
    # delay_days=0 with one extra row: at packaging time every replayed day has resolved.
    padded_y = np.vstack([y, np.full((1, horizon), np.nan)])
    padded_median = np.vstack([median, np.full((1, horizon), np.nan)])
    offsets = online_bias_offsets(padded_y, padded_median, cfg.conformal.window_days, delay_days=1)
    state: dict[str, object] = {"bias_offset_mw": round(float(offsets[-1]), 2)}
    replay = online_bias_offsets(y, median, cfg.conformal.window_days)[:, None]
    for level in cfg.conformal.levels:
        tail = (1.0 - level) / 2.0
        low = table[f"q{tail:.2f}"].to_numpy(float).reshape(-1, horizon) + replay
        high = table[f"q{1.0 - tail:.2f}"].to_numpy(float).reshape(-1, horizon) + replay
        calibrator = OnlineCQR(level, cfg.conformal.window_days, cfg.conformal.gamma, horizon)
        calibrator.run(y, low, high)
        calibrator.delay_days = 0  # At packaging time, every replayed day has resolved.
        state[f"{level:.2f}"] = {
            "alpha": calibrator.alpha,
            "margins_mw": np.round(calibrator.margins(), 2).tolist(),
        }
    return state


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def build_bundle(cfg: Config, output: str, reuse: bool = False) -> Path:
    report = json.loads((RESULTS_DIR / "metrics.json").read_text())
    selected = report["selected_model"]
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    for member in members_of(selected):
        if reuse and (directory / member).is_dir():
            logger.info("Reusing trained member %s", member)
            continue
        subprocess.run(
            [sys.executable, "-m", "gridcast.cli", "bundle", "--member", member,
             "--output", output],
            check=True,
        )  # fmt: skip

    state = conformal_state(load_predictions(cfg)[selected], cfg)
    (directory / "conformal.json").write_text(json.dumps(state, indent=2))
    (directory / "config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=2))

    frame = load_dataset(cfg)
    snapshot = frame.loc[: pd.Timestamp(cfg.backtest.end, tz="UTC")].iloc[
        -(cfg.task.history_hours + 200) :
    ]
    snapshot.to_parquet(directory / "demo_snapshot.parquet")

    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "model_name": "gridcast-de",
        "package_version": __version__,
        "selected_model": selected,
        "members": list(members_of(selected)),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config_hash": config_hash(),
        "trained_until": cfg.backtest.end,
        "data_rows": len(frame),
        "backtest": {
            "test_period": report["test_period"],
            "mae_mw": report["models"][selected]["mae"],
            "tso_mae_mw": report["tso"]["mae"],
            "pinball_mw": report["models"][selected]["pinball"],
            "coverage_cal_90": report["models"][selected]["coverage_cal_90"],
        },
        "files": {str(p.relative_to(directory)): sha256(p) for p in files},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _log_to_mlflow(directory, manifest)
    return directory


def _log_to_mlflow(directory: Path, manifest: dict) -> None:
    """Log the bundle as a versioned artifact in the tracking store."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("gridcast-deployment")
    with mlflow.start_run(run_name=f"bundle-{manifest['selected_model']}"):
        mlflow.log_params(
            {
                "selected_model": manifest["selected_model"],
                "trained_until": manifest["trained_until"],
                "config_hash": manifest["config_hash"],
            }
        )
        mlflow.log_metrics(
            {key: value for key, value in manifest["backtest"].items() if key != "test_period"}
        )
        mlflow.log_artifacts(str(directory), artifact_path="bundle")
