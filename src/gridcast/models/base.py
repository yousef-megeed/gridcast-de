"""Common model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.config import TaskConfig
from gridcast.features import SeriesStore


class Forecaster(ABC):
    """Maps issue positions to quantile forecasts of shape ``(issues, horizon, quantiles)``."""

    name: str

    def __init__(self, task: TaskConfig):
        self.task = task
        self.quantiles = np.asarray(task.quantiles)

    @abstractmethod
    def fit(
        self,
        store: SeriesStore,
        train_end: pd.Timestamp,
        val_end: pd.Timestamp | None = None,
        fixed_iterations: int | None = None,
    ) -> Forecaster:
        """Train on targets before ``train_end``, early-stopping on ``[train_end, val_end)``.

        With ``val_end=None`` there is no validation data, and ``fixed_iterations`` (found
        by an earlier early-stopped fit) sets the training length.
        """

    @abstractmethod
    def predict(self, store: SeriesStore, positions: np.ndarray, source: str) -> np.ndarray:
        """Return MW quantiles. ``source`` selects the weather information scenario."""

    @property
    def best_iterations(self) -> int | None:
        return None

    def save(self, directory: Path) -> None:
        raise NotImplementedError(f"{self.name} is not a deployable model")


def sort_quantiles(values: np.ndarray) -> np.ndarray:
    """Rearrangement (Chernozhukov et al., 2010): never worsens pinball loss, removes crossing."""
    return np.sort(values, axis=-1)
