"""Hyperparameter search, confined to data before the first test fold.

Train: targets before ``tune_train_end``. Validate: ``[tune_train_end, tune_val_end)``.
The search uses ERA5 weather because archived NWP is incomplete in the validation year.
That can only affect the choice of hyperparameters, never the reported test scores.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from gridcast.config import Config
from gridcast.data.pipeline import load_dataset
from gridcast.features import SeriesStore, daily_issue_times, targets
from gridcast.metrics import mean_pinball

logger = logging.getLogger(__name__)
TUNED_PATH = Path("configs/tuned.yaml")


def _merge_overlay(section: str, values: object) -> None:
    current = yaml.safe_load(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}
    current = current or {}
    current[section] = values
    TUNED_PATH.write_text(yaml.safe_dump(current, sort_keys=True))


def _setup(cfg: Config):
    store = SeriesStore(load_dataset(cfg))
    train_end = pd.Timestamp(cfg.backtest.tune_train_end, tz="UTC")
    val_end = pd.Timestamp(cfg.backtest.tune_val_end, tz="UTC")
    issues = daily_issue_times(cfg.backtest.tune_train_end, cfg.backtest.tune_val_end, cfg.task)
    positions = store.positions(issues)
    return store, train_end, val_end, positions, targets(store, positions, cfg.task)


def tune_ridge(cfg: Config) -> dict[str, float]:
    from gridcast.models.baselines import RidgeForecaster

    store, train_end, _, positions, y = _setup(cfg)
    scores = {}
    for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        model = RidgeForecaster(cfg.task, alpha=alpha).fit(store, train_end)
        scores[alpha] = mean_pinball(y, model.predict(store, positions, "era5"), cfg.task.quantiles)
    best = float(min(scores, key=lambda key: scores[key]))
    _merge_overlay("ridge_alpha", best)
    logger.info("ridge alpha scores %s -> %s", scores, best)
    return {"ridge_alpha": best, "val_pinball": float(scores[best])}


def tune_gbm(cfg: Config, n_trials: int = 30) -> dict[str, float]:
    import lightgbm as lgb

    from gridcast.features import tabular_features, training_positions

    store, train_end, _, positions, y = _setup(cfg)
    train_pos = training_positions(
        store, store.index[0], train_end, cfg.gbm.origin_stride_hours, cfg.task
    )
    x_train, level = tabular_features(store, train_pos, "era5", cfg.task)
    label = targets(store, train_pos, cfg.task).ravel() / level
    train_set = lgb.Dataset(
        x_train, label=label, free_raw_data=False, params={"feature_pre_filter": False}
    )
    x_val, val_level = tabular_features(store, positions, "era5", cfg.task)
    valid = lgb.Dataset(x_val, label=y.ravel() / val_level, reference=train_set)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
            "verbosity": -1,
            "seed": cfg.seed,
            "num_threads": 8,
            "bagging_freq": 1,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 30.0, log=True),
        }
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=cfg.gbm.num_boost_round,
            valid_sets=[valid],
            callbacks=[lgb.early_stopping(cfg.gbm.early_stopping_rounds, verbose=False)],
        )
        prediction = booster.predict(x_val) * val_level
        return float(np.mean(np.abs(y.ravel() - prediction)))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(objective, n_trials=n_trials)
    _merge_overlay("gbm", study.best_params)
    logger.info("gbm best MAE %.1f %s", study.best_value, study.best_params)
    return {"val_mae": study.best_value, **study.best_params}


def tune_nn(cfg: Config, n_trials: int = 15) -> dict[str, float]:
    from gridcast.models.neural import NeuralForecaster

    store, train_end, val_end, positions, y = _setup(cfg)

    def objective(trial: optuna.Trial) -> float:
        d_model = trial.suggest_categorical("d_model", [32, 64, 96])
        params = {
            "d_model": d_model,
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "patch_hours": trial.suggest_categorical("patch_hours", [6, 12, 24]),
            "context_hours": trial.suggest_categorical("context_hours", [168, 336]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 2e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "origin_stride_hours": 6,
            "max_epochs": 12,
            "patience": 3,
        }
        nn_cfg = cfg.nn.model_copy(update=params)
        model = NeuralForecaster(cfg.task, nn_cfg, seed=cfg.seed)
        model.fit(store, train_end, val_end)
        return mean_pinball(y, model.predict(store, positions, "era5"), cfg.task.quantiles)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(objective, n_trials=n_trials)
    _merge_overlay("nn", study.best_params)
    logger.info("nn best pinball %.2f %s", study.best_value, study.best_params)
    return {"val_pinball": study.best_value, **study.best_params}
