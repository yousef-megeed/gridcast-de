from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest

from gridcast.data import sources
from gridcast.data.pipeline import assemble


def _weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"temperature_2m": 10.0, "shortwave_radiation": 100.0, "wind_speed_10m": 5.0}, index=index
    )


def _assemble(load: pd.Series):
    index = pd.date_range(load.index.min(), load.index.max(), freq="h", tz="UTC")
    tso = pd.Series(50_000.0, index=index)
    return assemble(load, tso, _weather(index), _weather(index), 25_000, 95_000)


def _load(hours: int = 48) -> pd.Series:
    index = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    return pd.Series(50_000.0, index=index)


def test_assemble_reindexes_gaps_and_flags_them():
    load = _load().drop(_load().index[10:12])
    frame, report = _assemble(load)
    assert len(frame) == 48
    assert report.missing_load_hours == 2
    assert report.interpolated_load_hours == 2
    assert not frame["load_observed"].iloc[10]
    assert frame["load"].iloc[10] == pytest.approx(50_000)


def test_assemble_keeps_long_gaps_partly_missing():
    load = _load().drop(_load().index[10:20])
    frame, report = _assemble(load)
    assert report.missing_load_hours == 10
    assert frame["load"].isna().sum() == 7  # only MAX_INTERPOLATE_HOURS get filled


def test_assemble_removes_implausible_values():
    load = _load()
    load.iloc[5] = 0.0
    load.iloc[6] = 500_000.0
    frame, report = _assemble(load)
    assert report.out_of_range_load_hours == 2
    assert not frame["load_observed"].iloc[5:7].any()


def test_assemble_drops_duplicates():
    load = _load()
    doubled = pd.concat([load, load.iloc[:3]]).sort_index()
    frame, report = _assemble(doubled)
    assert report.duplicate_timestamps == 3
    assert frame.index.is_unique


def test_weighted_weather_applies_weights_and_tolerates_a_missing_site():
    times = ["2025-01-01T00:00", "2025-01-01T01:00"]
    payload = [
        {"hourly": {"time": times, "temperature_2m": [float(i), None],
                    "shortwave_radiation": [0.0, 0.0], "wind_speed_10m": [1.0, 1.0]}}
        for i in range(len(sources.SITES))
    ]  # fmt: skip
    frame = sources._weighted_weather(payload)
    weights = np.array([site.weight for site in sources.SITES])
    expected = (np.arange(len(weights)) * weights).sum() / weights.sum()
    assert frame["temperature_2m"].iloc[0] == pytest.approx(expected)
    assert np.isnan(frame["temperature_2m"].iloc[1])  # no site reported: not silently zero


def test_fetch_smard_parses_chunks_and_uses_cache(tmp_path):
    calls = []
    old = 1_500_000_000_000
    payload = {"series": [[old, 50_000.0], [old + 3_600_000, None], [old + 7_200_000, 51_000.0]]}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("index_hour.json"):
            return httpx.Response(200, json={"timestamps": [old]})
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    series = sources.fetch_smard_series(410, cache_dir=tmp_path, client=client)
    assert series.tolist() == [50_000.0, 51_000.0]  # nulls dropped
    assert str(series.index.tz) == "UTC"
    chunks_first = len(calls)
    sources.fetch_smard_series(410, cache_dir=tmp_path, client=client)
    assert len(calls) == chunks_first + 1  # second run fetches only the index


@pytest.mark.network
def test_live_smard_contract():
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
    series = sources.fetch_smard_series(sources.SMARD_LOAD, since=since)
    assert series.between(25_000, 95_000).all()
