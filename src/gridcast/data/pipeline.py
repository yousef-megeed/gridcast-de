"""Assemble and validate the single hourly table every model trains on."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gridcast.config import Config
from gridcast.data import sources

logger = logging.getLogger(__name__)

WEATHER_COLUMNS = ("temp", "rad", "wind")
_RENAME = {"temperature_2m": "temp", "shortwave_radiation": "rad", "wind_speed_10m": "wind"}
# Short gaps are interpolated, longer ones stay missing and exclude the affected samples.
MAX_INTERPOLATE_HOURS = 3


@dataclass(frozen=True)
class QualityReport:
    rows: int
    start: str
    end: str
    missing_load_hours: int
    interpolated_load_hours: int
    out_of_range_load_hours: int
    duplicate_timestamps: int
    missing_tso_hours: int
    nwp_first_complete: str | None
    nwp_coverage_after_first: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assemble(
    load: pd.Series,
    tso: pd.Series,
    era5: pd.DataFrame,
    nwp: pd.DataFrame,
    load_min: float,
    load_max: float,
) -> tuple[pd.DataFrame, QualityReport]:
    """Align all sources on a gap-free UTC hourly index and record what was repaired."""
    duplicates = int(load.index.duplicated().sum())
    load = load[~load.index.duplicated(keep="last")]
    full_index = pd.date_range(load.index.min(), load.index.max(), freq="h", tz="UTC", name="time")

    frame = pd.DataFrame(index=full_index)
    frame["load"] = load.reindex(full_index)
    frame["tso_forecast"] = tso.reindex(full_index)
    # The tail of the series can be provisional; zeros and outliers are recording faults.
    implausible = (frame["load"] < load_min) | (frame["load"] > load_max)
    out_of_range = int(implausible.sum())
    frame.loc[implausible, "load"] = np.nan
    frame.loc[
        (frame["tso_forecast"] < load_min) | (frame["tso_forecast"] > load_max), "tso_forecast"
    ] = np.nan

    missing = int(frame["load"].isna().sum())
    filled = frame["load"].interpolate(limit=MAX_INTERPOLATE_HOURS, limit_area="inside")
    interpolated = int(frame["load"].isna().sum() - filled.isna().sum())
    frame["load_observed"] = frame["load"].notna()  # evaluation uses observed hours only
    frame["load"] = filled

    for raw, column in _RENAME.items():
        frame[column] = era5[raw].reindex(full_index).interpolate(limit=MAX_INTERPOLATE_HOURS)
        frame[f"{column}_nwp"] = nwp[raw].reindex(full_index)

    nwp_ok = frame[[f"{c}_nwp" for c in WEATHER_COLUMNS]].notna().all(axis=1)
    # First day after which at least 99% of the following days are complete.
    daily = nwp_ok.resample("D").mean()
    complete = daily.iloc[::-1].expanding().mean().iloc[::-1] >= 0.99
    first = complete.idxmax() if complete.any() else None
    coverage = float(nwp_ok[first:].mean()) if first is not None else 0.0

    report = QualityReport(
        rows=len(frame),
        start=str(full_index[0]),
        end=str(full_index[-1]),
        missing_load_hours=missing,
        interpolated_load_hours=interpolated,
        out_of_range_load_hours=out_of_range,
        duplicate_timestamps=duplicates,
        missing_tso_hours=int(frame["tso_forecast"].isna().sum()),
        nwp_first_complete=str(first.date()) if first is not None else None,
        nwp_coverage_after_first=round(coverage, 5),
    )
    return frame, report


def build_dataset(cfg: Config) -> tuple[pd.DataFrame, QualityReport]:
    raw = cfg.data.raw_dir
    logger.info("Fetching SMARD load and TSO forecast")
    load = sources.fetch_smard_series(sources.SMARD_LOAD, raw / "smard")
    tso = sources.fetch_smard_series(sources.SMARD_TSO_FORECAST, raw / "smard")
    load = load[load.index >= pd.Timestamp(cfg.data.start, tz="UTC")]

    # ERA5 lags real time by a few days, and closed chunks are cached, so stop at a stable date.
    end = (load.index.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    logger.info("Fetching ERA5 weather up to %s", end)
    era5 = sources.fetch_era5(cfg.data.start, end, raw / "weather")
    logger.info("Fetching archived NWP up to %s", end)
    nwp = sources.fetch_archived_nwp("2022-01-01", end, raw / "weather")

    frame, report = assemble(load, tso, era5, nwp, cfg.data.load_min_mw, cfg.data.load_max_mw)
    cfg.data.processed_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cfg.data.processed_path)
    return frame, report


def load_dataset(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.data.processed_path)
