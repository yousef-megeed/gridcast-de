"""Inference: load the bundle, build features with the training code, return forecasts."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

# torch and LightGBM ship separate OpenMP runtimes, which can deadlock in one process on
# macOS. Single-threaded inference is fast enough here and is safe everywhere.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from gridcast.config import Config
from gridcast.features import SeriesStore
from gridcast.integrity import verify_bundle
from gridcast.models.base import Forecaster, sort_quantiles
from gridcast.serve.schemas import ForecastPoint, ForecastRequest, ForecastResponse, Interval

logger = logging.getLogger(__name__)
_LIVE_TTL_SECONDS = 900
_WEATHER = ("temp", "rad", "wind")


class InputError(ValueError):
    """Input passed schema validation but cannot produce a forecast."""


def _load_members(directory: Path, members: list[str], cfg: Config) -> dict[str, Forecaster]:
    loaded: dict[str, Forecaster] = {}
    for member in members:
        if member == "lightgbm":
            from gridcast.models.gbm import GBMForecaster

            loaded[member] = GBMForecaster.load(directory / member, cfg.task, cfg.gbm)
        elif member == "attention_net":
            import torch

            from gridcast.models.neural import NeuralForecaster

            torch.set_num_threads(1)
            loaded[member] = NeuralForecaster.load(directory / member, cfg.task, cfg.nn)
        else:
            raise ValueError(f"Unsupported bundle member {member}")
    return loaded


class ForecastService:
    def __init__(self, directory: Path):
        verify_bundle(directory)
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.cfg = Config.model_validate(json.loads((directory / "config.json").read_text()))
        self.conformal = json.loads((directory / "conformal.json").read_text())
        self.members = _load_members(directory, self.manifest["members"], self.cfg)
        self.version = f"{self.manifest['package_version']}+{self.manifest['config_hash']}"
        self.snapshot = pd.read_parquet(directory / "demo_snapshot.parquet")
        self._live: tuple[float, ForecastResponse] | None = None
        self._lock = threading.Lock()
        # torch modules are not reentrant under a shared _past_weather attribute.
        self._predict_lock = threading.Lock()

    @property
    def min_history_hours(self) -> int:
        return self.cfg.task.history_hours + 1

    def predict_frame(
        self, frame: pd.DataFrame, issue: pd.Timestamp, source: str, label: str
    ) -> ForecastResponse:
        """Forecast from an hourly table with columns ``load``, weather and ``*_nwp`` columns."""
        task = self.cfg.task
        store = SeriesStore(frame)
        position = store.positions(pd.DatetimeIndex([issue]))
        if position[0] < task.history_hours:
            raise InputError(f"Need at least {self.min_history_hours} hours of load history")
        window = store.load[position[0] - task.history_hours : position[0] + 1]
        if not np.isfinite(window).all():
            raise InputError("Load history contains missing values")

        with self._predict_lock:
            values = np.mean(
                [member.predict(store, position, "nwp") for member in self.members.values()],
                axis=0,
            )[0]
        values = sort_quantiles(values) + self.conformal.get("bias_offset_mw", 0.0)

        steps = np.arange(task.gap_hours, task.gap_hours + task.horizon_hours)
        target_position = position[0] + steps
        tso = frame.get("tso_forecast")
        actual = frame["load"]
        points = []
        for i, pos in enumerate(target_position):
            quantiles = {f"{q:.2f}": float(values[i, j]) for j, q in enumerate(task.quantiles)}
            intervals = {}
            for level in self.cfg.conformal.levels:
                tail = (1.0 - level) / 2.0
                margin = self.conformal[f"{level:.2f}"]["margins_mw"][i]
                intervals[f"{level:.2f}"] = Interval(
                    lower_mw=min(quantiles[f"{tail:.2f}"] - margin, quantiles["0.50"]),
                    upper_mw=max(quantiles[f"{1 - tail:.2f}"] + margin, quantiles["0.50"]),
                )
            tso_value = float(tso.iloc[pos]) if tso is not None else float("nan")
            actual_value = float(actual.iloc[pos])
            points.append(
                ForecastPoint(
                    time=store.index[pos].to_pydatetime(),
                    median_mw=quantiles["0.50"],
                    intervals=intervals,
                    quantiles_mw=quantiles,
                    tso_forecast_mw=tso_value if np.isfinite(tso_value) else None,
                    actual_mw=actual_value if np.isfinite(actual_value) else None,
                )
            )
        return ForecastResponse(
            issue_time=issue.to_pydatetime(),
            model_name=self.manifest["model_name"],
            model_version=self.version,
            selected_model=self.manifest["selected_model"],
            source=label,
            warnings=self._warnings(frame, position[0], target_position),
            points=points,
        )

    def _warnings(self, frame: pd.DataFrame, position: int, targets: np.ndarray) -> list[str]:
        """Flag inputs far from the conditions the model was trained on."""
        out = []
        recent = frame["load"].iloc[position - 167 : position + 1].mean()
        if not 35_000 <= recent <= 75_000:
            out.append(f"Weekly mean load {recent:,.0f} MW is outside the historical range")
        temp = frame["temp_nwp"].iloc[targets]
        if temp.min() < -20 or temp.max() > 40:
            out.append("Forecast temperature is outside the historical national range")
        return out

    def predict_request(self, request: ForecastRequest) -> ForecastResponse:
        load = pd.Series(
            [p.load_mw for p in request.load],
            index=pd.DatetimeIndex([p.time for p in request.load]).tz_convert("UTC"),
        )
        weather = pd.DataFrame(
            {
                "temp": [p.temperature_c for p in request.weather],
                "rad": [p.shortwave_radiation_wm2 for p in request.weather],
                "wind": [p.wind_speed_kmh for p in request.weather],
            },
            index=pd.DatetimeIndex([p.time for p in request.weather]).tz_convert("UTC"),
        )
        issue = load.index[-1]
        if len(load) < self.min_history_hours:
            raise InputError(
                f"Need at least {self.min_history_hours} hours of load history, got {len(load)}"
            )
        context = self.cfg.nn.context_hours
        if weather.index[0] > issue - pd.Timedelta(hours=context - 1):
            raise InputError(f"Weather must start at least {context - 1} hours before the issue")
        return self.predict_frame(_build_frame(load, weather), issue, "nwp", "request")

    def predict_snapshot(self) -> ForecastResponse:
        """Offline demo: the bundled history, ending at the last day the model trained on."""
        frame = self.snapshot
        task = self.cfg.task
        # Use the operational schedule, so that the target is a full local day from midnight.
        latest_observed = frame.index[-1] - pd.Timedelta(
            hours=task.gap_hours + task.horizon_hours - 1
        )
        issue = latest_issue(latest_observed, task.gap_hours)
        return self.predict_frame(frame, issue, "nwp", "bundled snapshot")

    def predict_live(self) -> ForecastResponse:
        """Tomorrow's forecast from live SMARD and Open-Meteo data, cached for 15 minutes."""
        with self._lock:
            if self._live and time.monotonic() - self._live[0] < _LIVE_TTL_SECONDS:
                return self._live[1]
        response = self._fetch_and_predict_live()
        with self._lock:
            self._live = (time.monotonic(), response)
        return response

    def _fetch_and_predict_live(self) -> ForecastResponse:
        from gridcast.data import sources

        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=self.min_history_hours + 72)
        load = sources.fetch_smard_series(sources.SMARD_LOAD, since=since)
        tso = sources.fetch_smard_series(sources.SMARD_TSO_FORECAST, since=since)
        weather = sources.fetch_live_nwp(past_days=16, forecast_days=3)
        weather = weather.rename(
            columns={"temperature_2m": "temp", "shortwave_radiation": "rad",
                     "wind_speed_10m": "wind"}
        )  # fmt: skip
        issue = latest_issue(load.index[-1], self.cfg.task.gap_hours)
        frame = _build_frame(load[load.index <= issue], weather, load, tso)
        return self.predict_frame(frame, issue, "nwp", "live (SMARD, Open-Meteo)")


def latest_issue(last_observed: pd.Timestamp, gap_hours: int) -> pd.Timestamp:
    """Most recent operational issue instant (06:00 local) with load already published."""
    local = last_observed.tz_convert("Europe/Berlin")
    day = local.tz_localize(None).normalize() + pd.Timedelta(days=1)
    for _ in range(3):
        midnight = day.tz_localize("Europe/Berlin").tz_convert("UTC")
        issue = midnight - pd.Timedelta(hours=gap_hours)
        if issue <= last_observed:
            return issue
        day -= pd.Timedelta(days=1)
    raise InputError("No recent issue time is available in the published load series")


def _build_frame(
    history: pd.Series,
    weather: pd.DataFrame,
    actual: pd.Series | None = None,
    tso: pd.Series | None = None,
) -> pd.DataFrame:
    """Assemble an hourly frame in the schema used by training."""
    end = max(history.index[-1], weather.index[-1])
    index = pd.date_range(history.index[0], end, freq="h", tz="UTC", name="time")
    frame = pd.DataFrame(index=index)
    frame["load"] = (actual if actual is not None else history).reindex(index)
    # Load after the issue time can only reach model code through a bug, and a leakage test
    # guards that. It is kept here so that responses can show actuals for past target days.
    frame["load_observed"] = frame["load"].notna()
    frame["tso_forecast"] = tso.reindex(index) if tso is not None else np.nan
    for column in _WEATHER:
        frame[column] = weather[column].reindex(index)
        frame[f"{column}_nwp"] = frame[column]
    return frame
