"""Interpretable reference models: if these are hard to beat, a complex model is unjustified."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gridcast.config import TaskConfig
from gridcast.features import SeriesStore, tabular_features, targets, training_positions
from gridcast.models.base import Forecaster


def _residual_quantiles(
    y: np.ndarray, pred: np.ndarray, hours: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Empirical residual quantiles per hour of day, turning a point model probabilistic."""
    table = np.zeros((24, quantiles.size))
    for hour in range(24):
        table[hour] = np.nanquantile((y - pred)[hours == hour], quantiles)
    return table


def _target_hours(store: SeriesStore, positions: np.ndarray, task: TaskConfig) -> np.ndarray:
    leads = np.arange(task.gap_hours, task.gap_hours + task.horizon_hours)
    return store.cal("hour")[positions[:, None] + leads[None, :]].astype(int)


class SeasonalNaive(Forecaster):
    """Same hour on the most recent available matching weekday."""

    name = "seasonal_naive"

    def __init__(self, task: TaskConfig, stride: int = 24):
        super().__init__(task)
        self.stride = stride
        self.table = np.zeros((24, self.quantiles.size))

    def _point(self, store: SeriesStore, positions: np.ndarray) -> np.ndarray:
        leads = np.arange(self.task.gap_hours, self.task.gap_hours + self.task.horizon_hours)
        return store.load[positions[:, None] + leads[None, :] - 168]

    def fit(self, store, train_end, val_end=None, fixed_iterations=None):
        # Residuals come from the last year of training data: older ones would overstate them.
        start = train_end - pd.Timedelta(days=365)
        positions = training_positions(store, start, train_end, self.stride, self.task)
        self.table = _residual_quantiles(
            targets(store, positions, self.task),
            self._point(store, positions),
            _target_hours(store, positions, self.task),
            self.quantiles,
        )
        return self

    def predict(self, store, positions, source):
        hours = _target_hours(store, positions, self.task)
        return self._point(store, positions)[..., None] + self.table[hours]


class RidgeForecaster(Forecaster):
    """Regularised linear model on level-normalised lags, with hour and weekday interactions."""

    name = "ridge"

    def __init__(self, task: TaskConfig, alpha: float = 10.0, stride: int = 12):
        super().__init__(task)
        self.alpha = alpha
        self.stride = stride
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.table = np.zeros((24, self.quantiles.size))
        self.columns: list[str] = []

    def _design(self, store: SeriesStore, positions: np.ndarray, source: str):
        frame, level = tabular_features(store, positions, source, self.task)
        hour = frame["hour"].astype(int)
        dow = frame["dow"].astype(int)
        design = {
            column: frame[column].to_numpy()
            for column in frame
            if column not in {"hour", "dow", "month", "doy", "lead"}
        }
        design = {key: np.nan_to_num(value) for key, value in design.items()}
        for h in range(24):
            design[f"hour_{h}"] = (hour == h).to_numpy(float)
        for d in range(7):
            design[f"dow_{d}"] = (dow == d).to_numpy(float)
        doy = frame["doy"].to_numpy()
        for k in (1, 2):
            design[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
            design[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
        return pd.DataFrame(design), level

    def fit(self, store, train_end, val_end=None, fixed_iterations=None):
        start = store.index[0]
        positions = training_positions(store, start, train_end, self.stride, self.task)
        design, level = self._design(store, positions, "era5")
        y = targets(store, positions, self.task)
        self.columns = list(design.columns)
        self.model.fit(design, y.ravel() / level)
        recent = training_positions(
            store, train_end - pd.Timedelta(days=365), train_end, self.stride, self.task
        )
        design, level = self._design(store, recent, "era5")
        point = (self.model.predict(design) * level).reshape(-1, self.task.horizon_hours)
        self.table = _residual_quantiles(
            targets(store, recent, self.task),
            point,
            _target_hours(store, recent, self.task),
            self.quantiles,
        )
        return self

    def predict(self, store, positions, source):
        design, level = self._design(store, positions, source)
        point = (self.model.predict(design[self.columns]) * level).reshape(
            -1, self.task.horizon_hours
        )
        return point[..., None] + self.table[_target_hours(store, positions, self.task)]
