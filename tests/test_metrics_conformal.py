from __future__ import annotations

import numpy as np
import pytest

from gridcast import metrics as m
from gridcast.conformal import OnlineCQR, conformal_margin, nonconformity, online_bias_offsets


def test_pinball_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 50_000)
    grid = np.linspace(-3, 3, 241)
    losses = [m.pinball(y, np.full_like(y, c), 0.9).mean() for c in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(1.2816, abs=0.05)


def test_median_pinball_is_half_mae():
    y, pred = np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 2.0])
    assert m.pinball(y, pred, 0.5).mean() == pytest.approx(0.5 * m.mae(y, pred))


def test_basic_scores():
    y, pred = np.array([100.0, 200.0]), np.array([110.0, 190.0])
    assert m.mae(y, pred) == 10
    assert m.rmse(y, pred) == 10
    assert m.mape(y, pred) == pytest.approx(7.5)
    assert m.coverage(y, np.array([90.0, 201.0]), np.array([110.0, 210.0])) == 0.5


def test_winkler_equals_width_when_inside_and_penalises_misses():
    y = np.array([5.0])
    assert m.winkler(y, np.array([0.0]), np.array([10.0]), 0.9) == 10.0
    # Width 1 + (2 / 0.1) * 4 MW outside.
    assert m.winkler(y, np.array([0.0]), np.array([1.0]), 0.9) == pytest.approx(81.0)


def test_diebold_mariano_detects_a_real_difference_and_not_a_null():
    rng = np.random.default_rng(1)
    base = rng.normal(100, 10, 500)
    better = m.diebold_mariano(base - 5, base + rng.normal(0, 3, 500))
    assert better["statistic"] < -5 and better["p_value"] < 1e-3
    null = m.diebold_mariano(base + rng.normal(0, 3, 500), base + rng.normal(0, 3, 500))
    assert null["p_value"] > 0.01


def test_diebold_mariano_widens_under_autocorrelation():
    """HAC variance must be less confident than an i.i.d. test on persistent differences."""
    rng = np.random.default_rng(2)
    diff = np.zeros(1000)
    for i in range(1, 1000):
        diff[i] = 0.9 * diff[i - 1] + rng.normal(0, 1)
    diff += 0.3
    hac = abs(m.diebold_mariano(diff, np.zeros(1000), max_lag=30)["statistic"])
    iid = abs(m.diebold_mariano(diff, np.zeros(1000), max_lag=0)["statistic"])
    assert hac < iid


def test_block_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(3).normal(10, 2, 400)
    low, high = m.block_bootstrap_ci(values)
    assert low < values.mean() < high
    assert high - low < 1.0


def test_nonconformity_sign():
    assert nonconformity(np.array(5.0), np.array(0.0), np.array(10.0)) == -5
    assert nonconformity(np.array(12.0), np.array(0.0), np.array(10.0)) == 2


def test_conformal_margin_uses_finite_sample_rank():
    scores = np.arange(1.0, 10.0)  # n=9, alpha=0.1 -> rank ceil(10 * 0.9) = 9
    assert conformal_margin(scores, 0.1) == 9.0
    assert conformal_margin(scores, 0.5) == 5.0
    assert conformal_margin(np.array([]), 0.1) == 0.0


def _simulate(mean_shift: float, sigma: float, days: int = 700, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, (days, 24))
    y[days // 2 :] = rng.normal(mean_shift, sigma, (days - days // 2, 24))
    low, high = np.full_like(y, -1.2816), np.full_like(y, 1.2816)  # nominal 80% before the shift
    return y, low, high


def test_cqr_repairs_an_overconfident_model():
    y = np.random.default_rng(4).normal(0, 2.0, (500, 24))
    low, high = np.full_like(y, -1.2816), np.full_like(y, 1.2816)
    assert m.coverage(y, low, high) < 0.55  # the raw interval is badly undercovering
    cal_low, cal_high = OnlineCQR(0.8, 90, 0.0).run(y, low, high)
    assert m.coverage(y[100:], cal_low[100:], cal_high[100:]) == pytest.approx(0.8, abs=0.03)


def test_cqr_narrows_an_underconfident_model():
    y = np.random.default_rng(5).normal(0, 0.5, (500, 24))
    low, high = np.full_like(y, -3.0), np.full_like(y, 3.0)
    cal_low, cal_high = OnlineCQR(0.8, 90, 0.0).run(y, low, high)
    assert (cal_high - cal_low)[100:].mean() < 2.0


def test_aci_recovers_coverage_after_a_distribution_shift():
    y, low, high = _simulate(mean_shift=1.0, sigma=1.5)
    cal_low, cal_high = OnlineCQR(0.8, 60, 0.02).run(y, low, high)
    tail = len(y) // 2 + 100
    assert m.coverage(y[tail:], cal_low[tail:], cal_high[tail:]) == pytest.approx(0.8, abs=0.04)


def test_delay_prevents_using_unresolved_days():
    calibrator = OnlineCQR(0.8, 90, 0.0, horizon=1, delay_days=2)
    y, low, high = np.array([100.0]), np.array([0.0]), np.array([1.0])
    assert calibrator.margins()[0] == 0.0
    calibrator.update(y, low, high, np.zeros(1))
    calibrator.update(y, low, high, np.zeros(1))
    assert calibrator.margins()[0] == 0.0  # both days are still within the feedback delay
    calibrator.update(y, low, high, np.zeros(1))
    assert calibrator.margins()[0] > 0.0


def test_bias_offsets_remove_a_constant_shift():
    rng = np.random.default_rng(6)
    y = rng.normal(0, 100, (300, 24))
    median = y - 400 + rng.normal(0, 100, (300, 24))
    offsets = online_bias_offsets(y, median, 90)
    assert offsets[:2].tolist() == [0.0, 0.0]  # nothing has resolved yet
    assert offsets[100:].mean() == pytest.approx(400, abs=15)


def test_bias_offsets_respect_the_feedback_delay():
    y, median = np.zeros((6, 1)), np.zeros((6, 1))
    y[3] = 1000.0  # a shock on day 3
    y[3:] = 1000.0
    offsets = online_bias_offsets(y, median, window_days=1, delay_days=2)
    assert offsets[4] == 0.0  # day 3 has not resolved when day 4 is issued
    assert offsets[5] == 1000.0
