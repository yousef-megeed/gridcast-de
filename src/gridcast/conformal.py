"""Online conformalised quantile regression with adaptive coverage control.

CQR (Romano et al., 2019) widens or narrows a model's raw interval by a data-driven margin.
Its guarantee requires exchangeable data, which a drifting load series is not. ACI (Gibbs &
Candes, 2021) restores long-run coverage by nudging the working miscoverage level:
``alpha_next = alpha + gamma * (target_alpha - observed_miss_rate)``.

The feedback delay is explicit. A forecast for day ``D`` is issued on ``D-1``, when the
latest fully resolved day is ``D-2``. With ``delay_days=2``, a score joins the window one
day after that, which errs on the side of safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def nonconformity(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """CQR score: signed distance outside the interval (negative when safely inside)."""
    return np.maximum(low - y, y - high)


def conformal_margin(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected ``(1 - alpha)`` empirical quantile of the scores."""
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return 0.0
    if alpha <= 0.0:
        return float(scores.max())
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float(scores.max())
    if rank < 1:
        return float(scores.min())
    return float(np.partition(scores, rank - 1)[rank - 1])


def online_bias_offsets(
    y: np.ndarray, median: np.ndarray, window_days: int, delay_days: int = 2
) -> np.ndarray:
    """Offset to add to each day's forecast: the recent median error, using resolved days only.

    Models train on reanalysis weather but forecast from NWP, whose distribution differs, and
    this shifts their output. One pooled offset per day is deliberately low-variance: a
    separate offset per hour would chase noise, given only ``window_days`` samples for each.
    """
    errors = y - median
    offsets = np.zeros(y.shape[0])
    for day in range(y.shape[0]):
        end = day - delay_days + 1  # days up to and including day - delay_days have resolved
        if end <= 0:
            continue
        recent = errors[max(0, end - window_days) : end]
        if np.isfinite(recent).any():
            offsets[day] = float(np.nanmedian(recent))
    return offsets


@dataclass
class OnlineCQR:
    """Calibrates one central interval level, with a separate margin for each lead."""

    level: float
    window_days: int = 90
    gamma: float = 0.02
    horizon: int = 24
    delay_days: int = 2
    alpha: float = field(init=False)
    _scores: list[np.ndarray] = field(init=False, default_factory=list)
    _misses: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.alpha = 1.0 - self.level

    def margins(self) -> np.ndarray:
        """Margins for the next forecast, using scores that have already resolved."""
        visible = self._scores[: -self.delay_days] if self.delay_days else self._scores
        recent = visible[-self.window_days :]
        if not recent:
            return np.zeros(self.horizon)
        stack = np.stack(recent)
        return np.array(
            [conformal_margin(stack[:, lead], self.alpha) for lead in range(self.horizon)]
        )

    def update(self, y: np.ndarray, low: np.ndarray, high: np.ndarray, margins: np.ndarray) -> None:
        """Record one day's outcome. Call in chronological order."""
        self._scores.append(nonconformity(y, low, high))
        valid = np.isfinite(y)
        missed = (y < low - margins) | (y > high + margins)
        self._misses.append(float(missed[valid].mean()) if valid.any() else np.nan)
        # Only misses visible by the next issue time may drive adaptation.
        if len(self._misses) > self.delay_days:
            observed = self._misses[-self.delay_days - 1]
            if np.isfinite(observed):
                target = 1.0 - self.level
                self.alpha = float(np.clip(self.alpha + self.gamma * (target - observed), 0.0, 1.0))

    def run(
        self, y: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Replay a chronological ``(days, horizon)`` sequence and return calibrated bounds."""
        out_low, out_high = low.copy(), high.copy()
        for day in range(y.shape[0]):
            margins = self.margins()
            out_low[day] = low[day] - margins
            out_high[day] = high[day] + margins
            self.update(y[day], low[day], high[day], margins)
        return out_low, out_high
