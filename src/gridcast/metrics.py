"""Point and probabilistic scores, plus significance testing."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

Array = Any  # numpy arrays or pandas Series: every score here is element-wise


def mae(y: Array, pred: Array) -> float:
    return float(np.nanmean(np.abs(y - pred)))


def rmse(y: Array, pred: Array) -> float:
    return float(np.sqrt(np.nanmean((y - pred) ** 2)))


def mape(y: Array, pred: Array) -> float:
    return float(np.nanmean(np.abs((y - pred) / y)) * 100.0)


def pinball(y: np.ndarray, pred: np.ndarray, quantile: float) -> np.ndarray:
    """Element-wise pinball loss: the proper scoring rule whose minimiser is the quantile."""
    diff = y - pred
    return np.maximum(quantile * diff, (quantile - 1.0) * diff)


def mean_pinball(y: np.ndarray, pred: np.ndarray, quantiles: tuple[float, ...]) -> float:
    """Average over quantiles; with a dense grid, ``2 * mean_pinball`` approximates CRPS."""
    return float(
        np.nanmean([np.nanmean(pinball(y, pred[..., i], q)) for i, q in enumerate(quantiles)])
    )


def coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.nanmean(((y >= low) & (y <= high))[np.isfinite(y)]))


def interval_width(low: np.ndarray, high: np.ndarray) -> float:
    return float(np.nanmean(high - low))


def winkler(y: np.ndarray, low: np.ndarray, high: np.ndarray, level: float) -> float:
    """Interval score: width plus a penalty of ``2 / alpha`` per MW outside the interval."""
    alpha = 1.0 - level
    penalty = (2.0 / alpha) * (np.maximum(low - y, 0.0) + np.maximum(y - high, 0.0))
    return float(np.nanmean(high - low + penalty))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, max_lag: int = 7) -> dict[str, float]:
    """Diebold-Mariano test on daily mean losses, with a Newey-West long-run variance.

    Daily losses are autocorrelated, so an ordinary paired t-test would overstate
    significance. Negative ``statistic`` means ``a`` is better than ``b``. Uses the
    Harvey-Leybourne-Newbold small-sample correction and a Student-t reference.
    """
    diff = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    centred = diff - diff.mean()
    variance = float(centred @ centred) / n
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)  # Bartlett kernel guarantees a positive variance
        variance += 2.0 * weight * float(centred[lag:] @ centred[:-lag]) / n
    horizon = 1
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = float(diff.mean() / np.sqrt(variance / n) * correction)
    return {
        "statistic": statistic,
        "p_value": float(2.0 * stats.t.sf(abs(statistic), df=n - 1)),
        "mean_loss_difference": float(diff.mean()),
        "n_days": float(n),
    }


def block_bootstrap_ci(
    values: np.ndarray, block: int = 7, draws: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Moving-block bootstrap CI of the mean, preserving weekly autocorrelation."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    n = values.size
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    index = (starts[..., None] + np.arange(block)).reshape(draws, -1)[:, :n]
    means = values[index].mean(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)
