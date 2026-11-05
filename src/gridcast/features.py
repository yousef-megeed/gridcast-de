"""Feature engineering shared by training, backtesting and serving.

Keeping a single code path is the main defence against training/serving skew. Every
feature is computed from an *issue position* ``p`` (the last observed hour) and a *lead*
``l``, so the target is at ``p + l``. Load is only ever read at positions ``<= p``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import holidays as holidays_lib
import numpy as np
import pandas as pd

from gridcast.config import TIMEZONE, TaskConfig, local_midnight_utc

# 2023 population shares; a holiday in Bavaria alone must not look like a national one.
STATE_POPULATION = {
    "NW": 18.1, "BY": 13.4, "BW": 11.3, "NI": 8.1, "HE": 6.4, "RP": 4.2, "SN": 4.1, "BE": 3.8,
    "SH": 3.0, "BB": 2.6, "ST": 2.2, "TH": 2.1, "HH": 1.9, "MV": 1.6, "SL": 1.0, "HB": 0.7,
}  # fmt: skip
WEEK_HOURS = 168
LAGS = (48, 72, 96, 120, 144, 168, 336, 504, 672)
WEATHER_SOURCES = ("none", "era5", "nwp")


@lru_cache(maxsize=64)
def _holiday_share_for_year(year: int) -> dict[pd.Timestamp, float]:
    total = sum(STATE_POPULATION.values())
    shares: dict[pd.Timestamp, float] = {}
    for state, population in STATE_POPULATION.items():
        for day in holidays_lib.country_holidays("DE", subdiv=state, years=year):
            stamp = pd.Timestamp(day)
            shares[stamp] = shares.get(stamp, 0.0) + population / total
    return shares


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar covariates, all known in advance and computed in local German time."""
    local = index.tz_convert(TIMEZONE)
    days = pd.DatetimeIndex(local.tz_localize(None).normalize())
    span = pd.date_range(days.min() - pd.Timedelta(days=7), days.max() + pd.Timedelta(days=7))

    share_map: dict[pd.Timestamp, float] = {}
    for year in sorted(set(span.year)):
        share_map.update(_holiday_share_for_year(int(year)))
    share = pd.Series([share_map.get(day, 0.0) for day in span], index=span)
    # Sundays are already low, so a holiday only adds information on working days and Saturdays.
    major = (share >= 0.5).astype(float)
    weekday = pd.Series(span.dayofweek, index=span)
    before = major.shift(-1, fill_value=0.0)
    after = major.shift(1, fill_value=0.0)
    bridge = (((weekday == 4) & (after == 1.0)) | ((weekday == 0) & (before == 1.0))).astype(
        float
    ) * (1.0 - major)
    winter_break = pd.Series(
        ((span.month == 12) & (span.day >= 23)) | ((span.month == 1) & (span.day <= 6)),
        index=span,
    ).astype(float)

    return pd.DataFrame(
        {
            "hour": local.hour.to_numpy(float),
            "dow": local.dayofweek.to_numpy(float),
            "month": local.month.to_numpy(float),
            "doy": local.dayofyear.to_numpy(float),
            "is_weekend": (local.dayofweek >= 5).astype(float),
            "holiday_share": share.reindex(days).to_numpy(),
            "before_holiday": before.reindex(days).to_numpy(),
            "after_holiday": after.reindex(days).to_numpy(),
            "bridge_day": bridge.reindex(days).to_numpy(),
            "winter_break": winter_break.reindex(days).to_numpy(),
        },
        index=index,
    )


class SeriesStore:
    """Array view on the hourly table, indexed by integer position."""

    def __init__(self, frame: pd.DataFrame):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError("frame must have a timezone-aware DatetimeIndex")
        if len(frame) > 1 and not (np.diff(frame.index.asi8) == np.diff(frame.index.asi8)[0]).all():
            raise ValueError("frame index must be regular and hourly")
        self.index = frame.index
        self.frame = frame
        self.load = frame["load"].to_numpy(float)
        self.calendar = calendar_frame(frame.index)
        self._cal = {column: self.calendar[column].to_numpy() for column in self.calendar}

    def cal(self, name: str) -> np.ndarray:
        return self._cal[name]

    def weather(self, source: str, variable: str) -> np.ndarray:
        """Return target-time weather under a given information scenario."""
        if source == "none":
            return np.full(len(self.index), np.nan)
        if source == "era5":
            return self.frame[variable].to_numpy(float)
        if source == "nwp":
            return self.frame[f"{variable}_nwp"].to_numpy(float)
        raise ValueError(f"Unknown weather source {source!r}")

    def positions(self, times: pd.DatetimeIndex) -> np.ndarray:
        positions = self.index.get_indexer(times)
        if (positions < 0).any():
            raise KeyError("Some requested times are outside the store")
        return positions


def daily_issue_times(start: str, end: str, task: TaskConfig) -> pd.DatetimeIndex:
    """Issue instant for each target day in ``[start, end)``.

    The target begins at local midnight, and the last observed hour is ``gap_hours`` before
    that. On DST days the window is 24 steps from midnight rather than the 23/25 local hours.
    """
    days = pd.date_range(start, end, freq="D", inclusive="left")
    gap = pd.Timedelta(hours=task.gap_hours)
    return pd.DatetimeIndex([local_midnight_utc(day) - gap for day in days])


def training_positions(
    store: SeriesStore, start: pd.Timestamp, end: pd.Timestamp, stride: int, task: TaskConfig
) -> np.ndarray:
    """Issue positions whose *entire target window* lies inside ``[start, end)``.

    Anchoring on the target's end prevents a sample from leaking across the
    train/validation/test boundary.
    """
    last_lead = task.gap_hours + task.horizon_hours - 1
    start_pos = int(store.index.searchsorted(start))
    end_pos = int(store.index.searchsorted(end))
    low = max(task.history_hours, start_pos)
    high = end_pos - last_lead
    candidates = np.arange(low, max(low, high), stride)
    if candidates.size == 0:
        return candidates
    leads = np.arange(task.gap_hours, last_lead + 1)
    context = store.load[candidates[:, None] - np.arange(task.history_hours)[None, :]]
    target = store.load[candidates[:, None] + leads[None, :]]
    valid = np.isfinite(context).all(axis=1) & np.isfinite(target).all(axis=1)
    return candidates[valid]


@dataclass(frozen=True)
class Levels:
    """Per-issue load level used to normalise all load values."""

    week_mean: np.ndarray


def issue_levels(store: SeriesStore, positions: np.ndarray) -> np.ndarray:
    """Mean load over the trailing week: removes the multi-year downward drift in demand."""
    window = positions[:, None] - np.arange(WEEK_HOURS)[None, :]
    return store.load[window].mean(axis=1)


def tabular_features(
    store: SeriesStore, positions: np.ndarray, source: str, task: TaskConfig
) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per ``(issue, lead)``. Returns features and the repeated level for de-scaling."""
    leads = np.arange(task.gap_hours, task.gap_hours + task.horizon_hours)
    n_leads = len(leads)
    p = np.repeat(positions, n_leads)
    lead = np.tile(leads, len(positions))
    target = p + lead

    level = np.repeat(issue_levels(store, positions), n_leads)
    load = store.load
    columns: dict[str, np.ndarray] = {}
    for lag in LAGS:
        columns[f"lag_{lag}"] = load[target - lag] / level
    columns["same_hour_4w_mean"] = np.mean(
        [columns[f"lag_{lag}"] for lag in (168, 336, 504, 672)], axis=0
    )
    columns["last_obs"] = load[p] / level
    last_day = load[p[:, None] - np.arange(24)[None, :]].mean(axis=1)
    columns["last_24h_mean"] = last_day / level
    prior_week = load[p[:, None] - np.arange(WEEK_HOURS, 2 * WEEK_HOURS)[None, :]].mean(axis=1)
    columns["level_trend"] = level / prior_week
    columns["level_gw"] = level / 1000.0

    for name in store.calendar.columns:
        columns[name] = store.cal(name)[target]
    columns["lead"] = lead.astype(float)
    # Tells the model whether its strongest lag was itself distorted by a holiday.
    columns["holiday_share_lag168"] = store.cal("holiday_share")[target - WEEK_HOURS]

    past_temp = store.frame["temp"].to_numpy(float)
    temp_week = past_temp[p[:, None] - np.arange(WEEK_HOURS)[None, :]]
    columns["temp_past_24h"] = np.nanmean(temp_week[:, :24], axis=1)
    if source != "none":
        temp = store.weather(source, "temp")[target]
        columns["temp"] = temp
        columns["rad"] = store.weather(source, "rad")[target]
        columns["wind"] = store.weather(source, "wind")[target]
        # The level already embeds last week's weather, so what matters is the change from it.
        columns["temp_vs_week"] = temp - np.nanmean(temp_week, axis=1)
        columns["hdd"] = np.maximum(15.0 - temp, 0.0)
        columns["cdd"] = np.maximum(temp - 22.0, 0.0)
    return pd.DataFrame(columns), level


def targets(store: SeriesStore, positions: np.ndarray, task: TaskConfig) -> np.ndarray:
    """Observed load in MW, shape ``(n_issues, horizon)``."""
    leads = np.arange(task.gap_hours, task.gap_hours + task.horizon_hours)
    return store.load[positions[:, None] + leads[None, :]]
