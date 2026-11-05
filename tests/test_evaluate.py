"""Tests for the code that produces the reported numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast import backtest
from gridcast.evaluate import conformalise, render_table, score, significance


def _table(cfg, days: int = 240, bias: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A prediction table in the backtest schema, with a controllable forecast bias."""
    rng = np.random.default_rng(seed)
    issue = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")
    horizon = cfg.task.horizon_hours
    y = 50_000 + rng.normal(0, 1_000, days * horizon)
    table = pd.DataFrame(
        {
            "issue_time": np.repeat(issue, horizon),
            "target_time": pd.date_range("2025-01-02", periods=days * horizon, freq="h", tz="UTC"),
            "step": np.tile(np.arange(horizon), days),
            "fold": 0,
            "split": np.where(np.arange(days * horizon) < 120 * horizon, "calibration", "test"),
            "y": y,
            "observed": True,
            "tso": y + rng.normal(0, 2_000, days * horizon),
        }
    )
    for q, offset in zip(cfg.task.quantiles, (-1200, -900, -500, 0, 500, 900, 1200), strict=True):
        table[f"q{q:.2f}"] = 50_000 + bias + offset
    return table


def test_a_day_s_bounds_do_not_depend_on_its_own_outcome(cfg):
    """No look-ahead: changing the last day's actuals must not change that day's forecast."""
    table = _table(cfg)
    clean = conformalise(table, cfg)
    changed = table.copy()
    last_day = changed["issue_time"] == changed["issue_time"].max()
    changed.loc[last_day, "y"] += 20_000
    dirty = conformalise(changed, cfg)
    columns = [c for c in clean if c.startswith(("cal_", "q0"))]
    pd.testing.assert_frame_equal(clean.loc[last_day, columns], dirty.loc[last_day, columns])


def test_bias_correction_removes_a_constant_forecast_bias(cfg):
    corrected = conformalise(_table(cfg, bias=-1_500), cfg)
    assert abs(score(corrected, cfg)["bias"]) < 150


def test_calibration_reaches_nominal_coverage(cfg):
    scores = score(conformalise(_table(cfg, bias=-1_000), cfg), cfg)
    assert scores["coverage_cal_90"] == pytest.approx(0.9, abs=0.03)
    assert scores["coverage_cal_50"] == pytest.approx(0.5, abs=0.05)
    assert scores["coverage_raw_90"] < 0.8


def test_score_ignores_calibration_rows_and_unobserved_hours(cfg):
    table = conformalise(_table(cfg), cfg)
    baseline = score(table, cfg)["mae"]
    corrupted = table.copy()
    corrupted.loc[corrupted["split"] == "calibration", "y"] = 1e9
    corrupted.loc[corrupted.index[-1], "observed"] = False
    corrupted.loc[corrupted.index[-1], "y"] = 1e9
    assert score(corrupted, cfg)["mae"] == pytest.approx(baseline, rel=1e-3)


def test_significance_reports_a_better_model(cfg):
    table = conformalise(_table(cfg), cfg)
    table["q0.50"] = table["y"] + 100.0
    result = significance({"model": table})["model"]
    assert result["statistic"] < 0 and result["p_value"] < 0.001
    assert result["improvement_ci_low"] > 0


def test_render_table_marks_skill_relative_to_tso():
    model = {"mae": 900.0, "rmse": 1200.0, "mape": 2.0, "pinball": 300.0, "coverage_cal_90": 0.9}
    report = {
        "tso": {"mae": 1000.0, "rmse": 1300.0, "mape": 3.0},
        "models": {"lightgbm": model},
    }
    assert "+10.0%" in render_table(report)


def test_backtest_end_to_end_keeps_test_days_out_of_training(cfg, frame, tmp_path, monkeypatch):
    """Runs the real engine on a small dataset, including MLflow tracking and parquet output."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backtest, "load_dataset", lambda _: frame)
    monkeypatch.setattr(backtest, "config_hash", lambda: "test")
    small = cfg.model_copy(
        update={
            "backtest": cfg.backtest.model_copy(
                update={
                    "fold_starts": ("2024-05-15", "2024-06-01"),
                    "end": "2024-06-15",
                    "val_days": 7,
                }
            )
        }
    )
    backtest.run_backtest(small, "seasonal_naive")
    table = pd.read_parquet(backtest.PREDICTIONS_DIR / "seasonal_naive.parquet")
    test = table[table["split"] == "test"]
    assert set(test["fold"]) == {0, 1}
    assert len(test) == 31 * cfg.task.horizon_hours
    assert (table["split"] == "calibration").any()
    assert table[table["split"] == "calibration"]["target_time"].max() < test["target_time"].min()
    assert test["issue_time"].is_monotonic_increasing
