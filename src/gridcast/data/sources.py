"""Clients for the two public data sources.

* SMARD (Bundesnetzagentur): realised grid load and the TSOs' day-ahead load forecast.
* Open-Meteo: ERA5 reanalysis for training, archived NWP runs for honest evaluation and
  the current NWP forecast for live inference.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SMARD_BASE = "https://www.smard.de/app/chart_data"
SMARD_LOAD = 410
SMARD_TSO_FORECAST = 411
ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARS = ("temperature_2m", "shortwave_radiation", "wind_speed_10m")
USER_AGENT = "gridcast-de/1.0 (research portfolio project)"


@dataclass(frozen=True)
class Site:
    name: str
    lat: float
    lon: float
    weight: float  # approximate metropolitan-region population in millions


# Weighted average approximating where German demand sits, without needing gridded data.
SITES: tuple[Site, ...] = (
    Site("Essen (Rhine-Ruhr)", 51.46, 7.01, 10.0),
    Site("Berlin", 52.52, 13.40, 6.2),
    Site("Munich", 48.14, 11.58, 6.1),
    Site("Frankfurt", 50.11, 8.68, 5.8),
    Site("Stuttgart", 48.78, 9.18, 5.4),
    Site("Hamburg", 53.55, 9.99, 5.4),
    Site("Hannover", 52.37, 9.73, 3.8),
    Site("Nuremberg", 49.45, 11.08, 3.6),
    Site("Leipzig", 51.34, 12.37, 2.4),
)


def _get_json(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> Any:
    """GET with exponential backoff; respects rate limits, which is what these APIs signal."""
    for attempt in range(6):
        try:
            response = client.get(url, params=params)
            if response.status_code == 429:
                wait = 65.0
                logger.warning("Rate limited; sleeping %.0fs", wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if attempt == 5:
                raise
            logger.warning("Request failed (%s), retrying", exc)
            time.sleep(2**attempt)
    raise RuntimeError(f"Exhausted retries for {url}")


def _client() -> httpx.Client:
    return httpx.Client(timeout=60, headers={"User-Agent": USER_AGENT}, follow_redirects=True)


def fetch_smard_series(
    filter_id: int,
    cache_dir: Path | None = None,
    since: pd.Timestamp | None = None,
    client: httpx.Client | None = None,
) -> pd.Series:
    """Download one hourly SMARD series from its weekly chunk files.

    Chunks older than three weeks are final and cached on disk, which makes re-runs
    incremental. ``since`` restricts the download for live inference.
    """
    owned = client is None
    client = client or _client()
    try:
        index = _get_json(client, f"{SMARD_BASE}/{filter_id}/DE/index_hour.json")["timestamps"]
        now_ms = pd.Timestamp.now(tz="UTC").timestamp() * 1000
        rows: list[list[Any]] = []
        for stamp in index:
            if since is not None and stamp < (since.timestamp() - 8 * 86400) * 1000:
                continue
            final = now_ms - stamp > 21 * 86400 * 1000
            path = cache_dir / str(filter_id) / f"{stamp}.json" if cache_dir else None
            if path is not None and final and path.exists():
                series = json.loads(path.read_text())
            else:
                url = f"{SMARD_BASE}/{filter_id}/DE/{filter_id}_DE_hour_{stamp}.json"
                series = _get_json(client, url)["series"]
                if path is not None and final:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(series))
            rows.extend(series)
    finally:
        if owned:
            client.close()

    frame = pd.DataFrame(rows, columns=["ms", "value"]).dropna().drop_duplicates("ms", keep="last")
    index_utc = pd.to_datetime(frame["ms"], unit="ms", utc=True)
    return pd.Series(
        frame["value"].to_numpy(float), index=index_utc, name=str(filter_id)
    ).sort_index()


def _weighted_weather(payload: list[dict[str, Any]], suffix: str = "") -> pd.DataFrame:
    """Collapse per-site hourly payloads into one demand-weighted national series."""
    weights = np.array([site.weight for site in SITES], dtype=float)
    time_index = pd.to_datetime(payload[0]["hourly"]["time"], utc=True)
    columns: dict[str, np.ndarray] = {}
    for variable in WEATHER_VARS:
        key = f"{variable}{suffix}"
        stack = np.array(
            [np.array(item["hourly"][key], dtype=float) for item in payload], dtype=float
        )
        mask = np.isfinite(stack)
        total = (np.where(mask, stack, 0.0) * weights[:, None]).sum(axis=0)
        norm = (mask * weights[:, None]).sum(axis=0)
        # Require most of the demand weight to be present; otherwise the mean is not national.
        enough = norm >= 0.7 * weights.sum()
        columns[variable] = np.where(enough, total / np.where(norm == 0, 1, norm), np.nan)
    return pd.DataFrame(columns, index=time_index)


def _site_params() -> dict[str, str]:
    return {
        "latitude": ",".join(str(site.lat) for site in SITES),
        "longitude": ",".join(str(site.lon) for site in SITES),
        "timezone": "UTC",
    }


def _fetch_weather_range(
    url: str, start: str, end: str, suffix: str, cache: Path | None, step_days: int
) -> pd.DataFrame:
    """Download weather in date chunks, caching each completed chunk."""
    hourly = ",".join(f"{variable}{suffix}" for variable in WEATHER_VARS)
    parts = []
    stamps = pd.date_range(start, end, freq=f"{step_days}D")
    with _client() as client:
        for chunk_start in stamps:
            chunk_end = min(chunk_start + pd.Timedelta(days=step_days - 1), pd.Timestamp(end))
            path = None
            if cache is not None:
                path = cache / f"{chunk_start.date()}_{chunk_end.date()}.parquet"
                if path.exists():
                    parts.append(pd.read_parquet(path))
                    continue
            params = {
                **_site_params(),
                "hourly": hourly,
                "start_date": str(chunk_start.date()),
                "end_date": str(chunk_end.date()),
            }
            frame = _weighted_weather(_get_json(client, url, params), suffix)
            parts.append(frame)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path)
            logger.info("weather %s..%s", chunk_start.date(), chunk_end.date())
            time.sleep(1.0)  # stay well within the free-tier limit
    return pd.concat(parts).sort_index()


def fetch_era5(start: str, end: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """ERA5-based reanalysis: hindsight weather, valid for training only."""
    cache = cache_dir / "era5" if cache_dir else None
    return _fetch_weather_range(ERA5_URL, start, end, "", cache, 366)


def fetch_archived_nwp(start: str, end: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Archived NWP values predicted 48 hours before their valid time.

    Target hours are at most 41 hours after the issue time, so a value predicted 48 hours
    earlier always existed at issue time. This is slightly conservative: operationally, a
    run with a shorter lead would be available.
    """
    cache = cache_dir / "nwp_day2" if cache_dir else None
    return _fetch_weather_range(PREVIOUS_RUNS_URL, start, end, "_previous_day2", cache, 92)


def fetch_live_nwp(past_days: int = 3, forecast_days: int = 3) -> pd.DataFrame:
    """Current forecast used for live inference."""
    params = {
        **_site_params(),
        "hourly": ",".join(WEATHER_VARS),
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    with _client() as client:
        return _weighted_weather(_get_json(client, FORECAST_URL, params))
