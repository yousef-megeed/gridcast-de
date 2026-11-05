from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.config import local_midnight_utc
from gridcast.features import (
    SeriesStore,
    calendar_frame,
    daily_issue_times,
    issue_levels,
    tabular_features,
    targets,
    training_positions,
)


def _issue_position(store: SeriesStore, days: int = 120) -> np.ndarray:
    return np.array([24 * days + 5])


@pytest.mark.parametrize("source", ["none", "era5", "nwp"])
def test_features_never_read_load_after_the_issue_time(frame, cfg, source):
    """The central leakage guarantee: corrupting the future must not change any feature."""
    position = _issue_position(SeriesStore(frame))
    clean, _ = tabular_features(SeriesStore(frame), position, source, cfg.task)

    corrupted = frame.copy()
    corrupted.iloc[position[0] + 1 :, corrupted.columns.get_loc("load")] = 1e9
    dirty, _ = tabular_features(SeriesStore(corrupted), position, source, cfg.task)
    pd.testing.assert_frame_equal(clean, dirty)


def test_features_do_read_load_up_to_the_issue_time(frame, cfg):
    """Guards the guard: the leakage test would pass trivially if load were never used."""
    position = _issue_position(SeriesStore(frame))
    clean, _ = tabular_features(SeriesStore(frame), position, "none", cfg.task)
    changed = frame.copy()
    changed.iloc[position[0], changed.columns.get_loc("load")] += 5000
    dirty, _ = tabular_features(SeriesStore(changed), position, "none", cfg.task)
    assert not clean["last_obs"].equals(dirty["last_obs"])


def test_no_weather_scenario_has_no_future_weather_columns(frame, cfg):
    position = _issue_position(SeriesStore(frame))
    features, _ = tabular_features(SeriesStore(frame), position, "none", cfg.task)
    assert not {"temp", "rad", "wind", "hdd", "cdd", "temp_vs_week"} & set(features.columns)


def test_nwp_scenario_reads_only_nwp_columns_at_target_time(frame, cfg):
    store = SeriesStore(frame)
    position = _issue_position(store)
    features, _ = tabular_features(store, position, "nwp", cfg.task)
    target = position[0] + cfg.task.gap_hours
    assert features["temp"].iloc[0] == frame["temp_nwp"].iloc[target]


def test_shapes_and_level(frame, cfg):
    store = SeriesStore(frame)
    positions = np.array([24 * 100, 24 * 101])
    features, level = tabular_features(store, positions, "era5", cfg.task)
    assert len(features) == len(level) == 2 * cfg.task.horizon_hours
    assert targets(store, positions, cfg.task).shape == (2, cfg.task.horizon_hours)
    expected = store.load[positions[0] - 167 : positions[0] + 1].mean()
    assert issue_levels(store, positions)[0] == pytest.approx(expected)


def test_training_positions_do_not_cross_the_boundary(frame, cfg):
    store = SeriesStore(frame)
    end = store.index[24 * 120]
    positions = training_positions(store, store.index[0], end, 1, cfg.task)
    last_target = positions.max() + cfg.task.gap_hours + cfg.task.horizon_hours - 1
    assert store.index[last_target] < end
    assert positions.min() >= cfg.task.history_hours


def test_training_positions_skip_missing_load(frame, cfg):
    gappy = frame.copy()
    gap = 24 * 110
    gappy.iloc[gap, gappy.columns.get_loc("load")] = np.nan
    store = SeriesStore(gappy)
    positions = training_positions(store, store.index[0], store.index[-1], 1, cfg.task)
    in_target = (positions >= gap - 41) & (positions <= gap - 18)
    in_context = (positions >= gap) & (positions < gap + cfg.task.history_hours)
    assert not (in_target | in_context).any()
    # A gap inside the unused 18-hour lead-time window does not invalidate a sample.
    assert (positions == gap - 10).any()


@pytest.mark.parametrize("day", ["2025-03-30", "2025-10-26", "2025-07-01", "2025-01-15"])
def test_issue_time_is_six_local_on_previous_day(cfg, day):
    """Holds across both DST changes, because 02:00 falls after the target's midnight."""
    issue = daily_issue_times(day, str(pd.Timestamp(day) + pd.Timedelta(days=1)), cfg.task)[0]
    local = issue.tz_convert("Europe/Berlin")
    assert local.hour == 6
    assert local.date() == (pd.Timestamp(day) - pd.Timedelta(days=1)).date()
    assert issue + pd.Timedelta(hours=18) == local_midnight_utc(day)


def test_calendar_is_local_time_not_utc():
    index = pd.DatetimeIndex(["2025-06-30T22:00:00Z"])
    calendar = calendar_frame(index)
    assert calendar["hour"].iloc[0] == 0
    assert calendar["month"].iloc[0] == 7


def test_holiday_share_distinguishes_national_and_regional():
    index = pd.date_range("2025-01-01", "2025-12-31", freq="D", tz="Europe/Berlin").tz_convert(
        "UTC"
    )
    share = calendar_frame(index)["holiday_share"]
    local = index.tz_convert("Europe/Berlin").strftime("%m-%d")
    assert share[local == "12-25"].iloc[0] == pytest.approx(1.0)
    regional = share[local == "01-06"].iloc[0]  # Epiphany: BW, BY and ST only
    assert 0.2 < regional < 0.4
    assert share[local == "03-12"].iloc[0] == 0.0


def test_bridge_day_detected():
    # Ascension Day, Thursday 29 May 2025; the following Friday is a bridge day.
    index = pd.date_range("2025-05-28", "2025-05-31", freq="D", tz="Europe/Berlin").tz_convert(
        "UTC"
    )
    bridge = calendar_frame(index)["bridge_day"].tolist()
    assert bridge == [0.0, 0.0, 1.0, 0.0]


def test_store_rejects_naive_and_irregular_index(frame):
    with pytest.raises(ValueError, match="timezone-aware"):
        SeriesStore(frame.tz_localize(None))
    with pytest.raises(ValueError, match="regular"):
        SeriesStore(frame.drop(frame.index[5]))
