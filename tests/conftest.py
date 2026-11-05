"""Synthetic fixtures: the test suite needs no network and no downloaded data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.config import Config, load_config


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config("configs/default.yaml", tuned=None)


def synthetic_frame(hours: int = 24 * 180, start: str = "2024-01-01", seed: int = 0):
    """Hourly table in the production schema, with daily, weekly and temperature structure."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC", name="time")
    t = np.arange(hours)
    temp = 10 + 10 * np.sin(2 * np.pi * t / (24 * 365)) + 3 * np.sin(2 * np.pi * t / 24)
    load = (
        55_000
        + 8_000 * np.sin(2 * np.pi * (t - 6) / 24)
        - 6_000 * (index.dayofweek >= 5)
        - 200 * (temp - 10)
        + rng.normal(0, 400, hours)
    )
    frame = pd.DataFrame(index=index)
    frame["load"] = load
    frame["load_observed"] = True
    frame["tso_forecast"] = load + rng.normal(0, 1500, hours)
    frame["temp"] = temp
    frame["rad"] = np.clip(400 * np.sin(2 * np.pi * (t - 6) / 24), 0, None)
    frame["wind"] = 15 + rng.normal(0, 2, hours)
    for column in ("temp", "rad", "wind"):
        frame[f"{column}_nwp"] = frame[column] + rng.normal(0, 0.5, hours)
    return frame


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return synthetic_frame()
