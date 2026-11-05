"""Request and response contracts. Validation failures become HTTP 422."""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import pairwise

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator

MAX_POINTS = 2000


class LoadPoint(BaseModel):
    time: AwareDatetime
    load_mw: float = Field(gt=0, le=200_000)


class WeatherPoint(BaseModel):
    time: AwareDatetime
    temperature_c: float = Field(ge=-60, le=60)
    shortwave_radiation_wm2: float = Field(ge=0, le=1500)
    wind_speed_kmh: float = Field(ge=0, le=300)


def _check_hourly(times: list[datetime], label: str) -> None:
    for earlier, later in pairwise(times):
        if later - earlier != timedelta(hours=1):
            raise ValueError(
                f"{label} must be consecutive hours; found {earlier} followed by {later}"
            )


class ForecastRequest(BaseModel):
    """Everything the model needs, sent explicitly, so the endpoint stays stateless.

    The last load point is the issue time. The forecast covers the 24 hours that start
    18 hours after it. Weather must span the past context through the end of the target.
    """

    load: list[LoadPoint] = Field(min_length=1, max_length=MAX_POINTS)
    weather: list[WeatherPoint] = Field(min_length=1, max_length=MAX_POINTS)

    @field_validator("load")
    @classmethod
    def load_is_hourly(cls, points: list[LoadPoint]) -> list[LoadPoint]:
        _check_hourly([point.time for point in points], "load")
        return points

    @field_validator("weather")
    @classmethod
    def weather_is_hourly(cls, points: list[WeatherPoint]) -> list[WeatherPoint]:
        _check_hourly([point.time for point in points], "weather")
        return points

    @model_validator(mode="after")
    def weather_covers_target(self) -> ForecastRequest:
        needed = self.load[-1].time + timedelta(hours=41)
        if self.weather[-1].time < needed:
            raise ValueError(f"weather must extend to at least {needed.isoformat()}")
        return self


class Interval(BaseModel):
    lower_mw: float
    upper_mw: float


class ForecastPoint(BaseModel):
    time: datetime
    median_mw: float
    intervals: dict[str, Interval] = Field(
        description="Conformally calibrated central intervals, keyed by nominal level."
    )
    quantiles_mw: dict[str, float] = Field(description="Raw model quantiles before calibration.")
    tso_forecast_mw: float | None = None
    actual_mw: float | None = None


class ForecastResponse(BaseModel):
    issue_time: datetime
    model_name: str
    model_version: str
    selected_model: str
    source: str
    warnings: list[str]
    points: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    uptime_seconds: float
