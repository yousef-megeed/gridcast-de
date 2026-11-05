"""Direct multi-horizon LightGBM, with one booster per quantile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from gridcast.config import GBMConfig, TaskConfig
from gridcast.features import SeriesStore, tabular_features, targets, training_positions
from gridcast.models.base import Forecaster, sort_quantiles


class GBMForecaster(Forecaster):
    """Predicts ``load / level``, because trees cannot extrapolate below their training range.

    German grid load has fallen by about 8 GW since 2018, so an unnormalised target would
    leave the model permanently biased high.
    """

    name = "lightgbm"

    def __init__(self, task: TaskConfig, cfg: GBMConfig, weather: str = "era5", seed: int = 42):
        super().__init__(task)
        self.cfg = cfg
        self.seed = seed
        self.train_weather = weather  # "none" trains without future-weather features
        self.boosters: dict[float, lgb.Booster] = {}
        self._best_iterations: int | None = None
        self.feature_names: list[str] = []

    @property
    def best_iterations(self) -> int | None:
        return self._best_iterations

    def _params(self, quantile: float) -> dict[str, Any]:
        return {
            "objective": "quantile",
            "alpha": quantile,
            "metric": "quantile",
            "seed": self.seed,
            "verbosity": -1,
            "num_threads": 8,
            **self.cfg.params,
        }

    def fit(self, store, train_end, val_end=None, fixed_iterations=None):
        positions = training_positions(
            store, store.index[0], train_end, self.cfg.origin_stride_hours, self.task
        )
        features, level = tabular_features(store, positions, self.train_weather, self.task)
        label = targets(store, positions, self.task).ravel() / level
        self.feature_names = list(features.columns)
        train_set = lgb.Dataset(features, label=label, free_raw_data=False)

        rounds = fixed_iterations or self.cfg.num_boost_round
        if val_end is not None:
            # Early stopping on the median alone: the other quantiles reuse its length,
            # because the tails are too noisy to stop on and this is seven times cheaper.
            val_positions = training_positions(store, train_end, val_end, 24, self.task)
            val_x, val_level = tabular_features(store, val_positions, self.train_weather, self.task)
            val_y = targets(store, val_positions, self.task).ravel() / val_level
            valid = lgb.Dataset(val_x, label=val_y, reference=train_set)
            median = lgb.train(
                self._params(0.5),
                train_set,
                num_boost_round=self.cfg.num_boost_round,
                valid_sets=[valid],
                callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)],
            )
            rounds = max(median.best_iteration, 50)
            self._best_iterations = rounds
        elif fixed_iterations is not None:
            self._best_iterations = fixed_iterations

        self.boosters = {
            float(q): lgb.train(self._params(float(q)), train_set, num_boost_round=rounds)
            for q in self.quantiles
        }
        return self

    def predict(self, store, positions, source):
        features, level = tabular_features(store, positions, source, self.task)
        features = features[self.feature_names]
        ratio = np.stack(
            [self.boosters[float(q)].predict(features) for q in self.quantiles], axis=-1
        )
        values = ratio * level[:, None]
        return sort_quantiles(values.reshape(-1, self.task.horizon_hours, self.quantiles.size))

    def contributions(self, store: SeriesStore, positions: np.ndarray, source: str):
        """TreeSHAP values of the median booster, on the scale of ``load / level``."""
        features, _ = tabular_features(store, positions, source, self.task)
        features = features[self.feature_names]
        values = np.asarray(self.boosters[0.5].predict(features, pred_contrib=True))
        return features, values[:, :-1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for quantile, booster in self.boosters.items():
            booster.save_model(str(directory / f"q{quantile:.2f}.txt"))
        (directory / "gbm.json").write_text(
            json.dumps({"features": self.feature_names, "weather": self.train_weather})
        )

    @classmethod
    def load(cls, directory: Path, task: TaskConfig, cfg: GBMConfig) -> GBMForecaster:
        meta = json.loads((directory / "gbm.json").read_text())
        model = cls(task, cfg, weather=meta["weather"])
        model.feature_names = meta["features"]
        model.boosters = {
            float(q): lgb.Booster(model_file=str(directory / f"q{float(q):.2f}.txt"))
            for q in task.quantiles
        }
        return model
