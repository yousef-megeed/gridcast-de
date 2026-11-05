"""Typed access to the YAML experiment configuration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

TIMEZONE = "Europe/Berlin"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskConfig(_Frozen):
    history_hours: int
    gap_hours: int
    horizon_hours: int
    quantiles: tuple[float, ...]


class DataConfig(_Frozen):
    raw_dir: Path
    processed_path: Path
    start: str
    load_min_mw: float
    load_max_mw: float


class BacktestConfig(_Frozen):
    tune_train_end: str
    tune_val_end: str
    fold_starts: tuple[str, ...]
    end: str
    val_days: int


class ConformalConfig(_Frozen):
    levels: tuple[float, ...]
    window_days: int
    gamma: float


class GBMConfig(_Frozen):
    origin_stride_hours: int
    params: dict[str, Any]
    num_boost_round: int
    early_stopping_rounds: int


class NNConfig(_Frozen):
    origin_stride_hours: int
    context_hours: int
    patch_hours: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    device: str


class Config(_Frozen):
    seed: int
    ridge_alpha: float
    task: TaskConfig
    data: DataConfig
    backtest: BacktestConfig
    conformal: ConformalConfig
    gbm: GBMConfig
    nn: NNConfig


def load_config(
    path: str | Path = "configs/default.yaml", tuned: str | Path | None = "configs/tuned.yaml"
) -> Config:
    """Load the base config and overlay hyperparameters found by ``gridcast tune``."""
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    if tuned is not None and Path(tuned).exists():
        with open(tuned) as handle:
            overlay = yaml.safe_load(handle) or {}
        if "ridge_alpha" in overlay:
            raw["ridge_alpha"] = overlay["ridge_alpha"]
        raw["gbm"]["params"].update(overlay.get("gbm", {}))
        raw["nn"].update(overlay.get("nn", {}))
    return Config.model_validate(raw)


def config_hash(
    path: str | Path = "configs/default.yaml", tuned: str | Path = "configs/tuned.yaml"
) -> str:
    digest = hashlib.sha256(Path(path).read_bytes())
    if Path(tuned).exists():
        digest.update(Path(tuned).read_bytes())
    return digest.hexdigest()[:12]


def local_midnight_utc(day: str | pd.Timestamp) -> pd.Timestamp:
    """Return the UTC instant at which local day ``day`` begins in Germany."""
    stamp = pd.Timestamp(day)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize().tz_localize(TIMEZONE).tz_convert("UTC")
