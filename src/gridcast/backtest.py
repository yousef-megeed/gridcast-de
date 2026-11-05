"""Rolling-origin backtest with expanding training windows.

For each half-year fold:

1. Fit on targets before ``fold_start - val_days``, early-stopping on the validation span.
2. Refit on everything before ``fold_start``, with the iteration count from step 1, because
   the most recent months matter most under drift.
3. Forecast each test day once, using archived NWP that was available at the issue time.

Fold 0's early-stopped model also forecasts the validation span out of sample. Those
forecasts warm up the conformal calibrator before the first test day.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from gridcast.config import Config, config_hash
from gridcast.data.pipeline import load_dataset
from gridcast.features import SeriesStore, daily_issue_times, targets
from gridcast.metrics import mae, mean_pinball
from gridcast.models.base import Forecaster

logger = logging.getLogger(__name__)
PREDICTIONS_DIR = Path("results/predictions")
CALIBRATION_WARMUP_DAYS = 120
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "gridcast-backtest"


@dataclass(frozen=True)
class Variant:
    """A trained model plus the weather scenario used at forecast time."""

    name: str
    model: str
    train_weather: str
    source: str


VARIANTS = {
    "seasonal_naive": Variant("seasonal_naive", "seasonal_naive", "none", "none"),
    "ridge": Variant("ridge", "ridge", "era5", "nwp"),
    "lightgbm": Variant("lightgbm", "lightgbm", "era5", "nwp"),
    "lightgbm_no_weather": Variant("lightgbm_no_weather", "lightgbm", "none", "none"),
    "attention_net": Variant("attention_net", "attention_net", "era5", "nwp"),
}
# Same trained model as "lightgbm", scored with hindsight weather: an upper bound, not a result.
ORACLE_OF = {"lightgbm": "lightgbm_oracle"}


def make_model(variant: Variant, cfg: Config, seed: int | None = None) -> Forecaster:
    """Instantiate lazily: torch and LightGBM deadlock on macOS when training in one process."""
    seed = cfg.seed if seed is None else seed
    if variant.model == "seasonal_naive":
        from gridcast.models.baselines import SeasonalNaive

        return SeasonalNaive(cfg.task)
    if variant.model == "ridge":
        from gridcast.models.baselines import RidgeForecaster

        return RidgeForecaster(cfg.task, alpha=cfg.ridge_alpha)
    if variant.model == "lightgbm":
        from gridcast.models.gbm import GBMForecaster

        return GBMForecaster(cfg.task, cfg.gbm, weather=variant.train_weather, seed=seed)
    if variant.model == "attention_net":
        from gridcast.models.neural import NeuralForecaster

        return NeuralForecaster(cfg.task, cfg.nn, seed=seed)
    raise ValueError(variant.model)


def to_long(
    store: SeriesStore,
    issues: pd.DatetimeIndex,
    quantile_values: np.ndarray,
    cfg: Config,
    fold: int,
    split: str,
) -> pd.DataFrame:
    """Flatten ``(issues, horizon, quantiles)`` into a tidy table with actuals and the TSO."""
    positions = store.positions(issues)
    leads = np.arange(cfg.task.gap_hours, cfg.task.gap_hours + cfg.task.horizon_hours)
    target_pos = positions[:, None] + leads[None, :]
    frame = pd.DataFrame(
        {
            "issue_time": np.repeat(issues.to_numpy(), len(leads)),
            "target_time": store.index[target_pos.ravel()],
            "step": np.tile(np.arange(len(leads)), len(issues)),
            "fold": fold,
            "split": split,
            "y": store.load[target_pos].ravel(),
            "observed": store.frame["load_observed"].to_numpy()[target_pos].ravel(),
            "tso": store.frame["tso_forecast"].to_numpy()[target_pos].ravel(),
        }
    )
    for i, quantile in enumerate(cfg.task.quantiles):
        frame[f"q{quantile:.2f}"] = quantile_values[..., i].ravel()
    return frame


def fold_windows(cfg: Config) -> list[tuple[str, str]]:
    starts = list(cfg.backtest.fold_starts)
    return list(zip(starts, [*starts[1:], cfg.backtest.end], strict=True))


def run_backtest(cfg: Config, name: str, seeds: tuple[int, ...] | None = None) -> None:
    """Run one variant over all folds and write its predictions to parquet."""
    variant = VARIANTS[name]
    store = SeriesStore(load_dataset(cfg))
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    seeds = seeds or (cfg.seed,)
    collected: list[pd.DataFrame] = []
    oracle: list[pd.DataFrame] = []

    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "variant": name,
                "train_weather": variant.train_weather,
                "forecast_weather": variant.source,
                "config_hash": config_hash(),
                "seeds": ",".join(map(str, seeds)),
            }
        )
        mlflow.log_dict(cfg.model_dump(mode="json"), "config.json")
        for fold, (start, end) in enumerate(fold_windows(cfg)):
            start_ts = pd.Timestamp(start, tz="UTC")
            train_end = start_ts - pd.Timedelta(days=cfg.backtest.val_days)
            issues = daily_issue_times(start, end, cfg.task)
            positions = store.positions(issues)
            y = targets(store, positions, cfg.task)

            with mlflow.start_run(run_name=f"{name}-fold{fold}", nested=True):
                began = time.time()
                outputs, oracle_outputs, warmups, iterations = [], [], [], []
                for seed in seeds:
                    model = make_model(variant, cfg, seed).fit(store, train_end, start_ts)
                    iterations.append(model.best_iterations)
                    if fold == 0:
                        warm_start = start_ts - pd.Timedelta(days=CALIBRATION_WARMUP_DAYS)
                        warm_issues = daily_issue_times(str(warm_start.date()), start, cfg.task)
                        warmups.append(
                            model.predict(store, store.positions(warm_issues), variant.source)
                        )
                    model = make_model(variant, cfg, seed).fit(
                        store, start_ts, None, model.best_iterations
                    )
                    outputs.append(model.predict(store, positions, variant.source))
                    if name in ORACLE_OF:
                        oracle_outputs.append(model.predict(store, positions, "era5"))
                quantiles = np.mean(outputs, axis=0)
                collected.append(to_long(store, issues, quantiles, cfg, fold, "test"))
                if fold == 0:
                    collected.append(
                        to_long(
                            store, warm_issues, np.mean(warmups, axis=0), cfg, -1, "calibration"
                        )
                    )
                if oracle_outputs:
                    oracle.append(
                        to_long(store, issues, np.mean(oracle_outputs, axis=0), cfg, fold, "test")
                    )
                metrics = {
                    "mae": mae(y, quantiles[..., cfg.task.quantiles.index(0.5)]),
                    "pinball": mean_pinball(y, quantiles, cfg.task.quantiles),
                    "fit_predict_seconds": time.time() - began,
                }
                mlflow.log_params({"test_start": start, "test_end": end, "fold": fold})
                if iterations[0] is not None:
                    mlflow.log_param("best_iterations", json.dumps(iterations))
                mlflow.log_metrics(metrics)
                logger.info("%s fold %d %s", name, fold, metrics)

        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = PREDICTIONS_DIR / f"{name}.parquet"
        table = pd.concat(collected).sort_values("target_time")
        table.to_parquet(path)
        if oracle:
            pd.concat(oracle).to_parquet(PREDICTIONS_DIR / f"{ORACLE_OF[name]}.parquet")
        test = table[table["split"] == "test"]
        mlflow.log_metrics(
            {
                "mae": mae(test["y"].to_numpy(), test["q0.50"].to_numpy()),
                "tso_mae": mae(test["y"].to_numpy(), test["tso"].to_numpy()),
            }
        )
