"""Model tests run in small configurations. LightGBM and torch tests live in separate files,

because the two libraries' OpenMP runtimes can deadlock when training in one macOS process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gridcast.features import SeriesStore
from gridcast.models.baselines import RidgeForecaster, SeasonalNaive
from gridcast.models.neural import (
    MonotoneQuantileHead,
    NeuralForecaster,
    QuantileAttentionNet,
    pinball_loss,
)


@pytest.fixture(scope="module")
def store(frame) -> SeriesStore:
    return SeriesStore(frame)


def _quantiles(cfg) -> torch.Tensor:
    return torch.tensor(cfg.task.quantiles, dtype=torch.float32)


def test_head_cannot_produce_crossing_quantiles(cfg):
    head = MonotoneQuantileHead(16, _quantiles(cfg))
    for seed in range(5):
        torch.manual_seed(seed)
        output = head(torch.randn(8, 24, 16) * 50, torch.zeros(8, 24))
        assert (output.diff(dim=-1) > 0).all()


def test_head_median_tracks_anchor(cfg):
    head = MonotoneQuantileHead(16, _quantiles(cfg))
    features = torch.randn(2, 24, 16)
    shift = 3.0 * torch.ones(2, 24)
    gap = head(features, shift) - head(features, torch.zeros(2, 24))
    assert gap.detach().numpy() == pytest.approx(3.0, abs=1e-5)


def test_net_output_shape_and_gradients(cfg):
    net = QuantileAttentionNet(168, 12, 24, _quantiles(cfg), d_model=16, n_heads=2, n_layers=1)
    from gridcast.models.neural import FUTURE_CHANNELS, PAST_CHANNELS

    past, future = torch.randn(4, 168, PAST_CHANNELS), torch.randn(4, 24, FUTURE_CHANNELS)
    net.eval()  # attention dropout would otherwise stop the weights summing to one
    output, attention = net(past, future, torch.zeros(4, 24), return_attention=True)
    assert output.shape == (4, 24, 7)
    assert attention.shape == (4, 24, 14)
    assert attention.sum(-1).detach().numpy() == pytest.approx(1.0, abs=1e-4)
    pinball_loss(output, torch.zeros(4, 24), _quantiles(cfg)).backward()
    assert all(p.grad is not None for p in net.parameters())


def test_net_rejects_indivisible_patch_size(cfg):
    with pytest.raises(ValueError, match="divisible"):
        QuantileAttentionNet(100, 12, 24, _quantiles(cfg))


def test_pinball_loss_matches_numpy_metric(cfg):
    from gridcast.metrics import pinball

    pred = torch.tensor([[[1.0, 2.0]]])
    y = torch.tensor([[1.5]])
    q = torch.tensor([0.1, 0.9])
    expected = (pinball(np.array(1.5), np.array(1.0), 0.1)
                + pinball(np.array(1.5), np.array(2.0), 0.9)) / 2  # fmt: skip
    assert pinball_loss(pred, y, q).item() == pytest.approx(float(expected))


def _small_nn(cfg):
    return cfg.nn.model_copy(
        update={
            "d_model": 16, "n_heads": 2, "n_layers": 1, "context_hours": 168, "patch_hours": 24,
            "origin_stride_hours": 12, "max_epochs": 30, "batch_size": 64, "patience": 3,
            "lr": 0.003,
        }
    )  # fmt: skip


def test_neural_learns_and_round_trips(store, cfg, tmp_path):
    torch.manual_seed(0)
    end = store.index[24 * 150]
    val_end = store.index[24 * 170]
    model = NeuralForecaster(cfg.task, _small_nn(cfg), seed=0).fit(store, end, val_end)
    positions = np.array([24 * 171, 24 * 172])
    prediction = model.predict(store, positions, "nwp")
    assert prediction.shape == (2, 24, 7)
    assert (np.diff(prediction, axis=-1) > 0).all()

    from gridcast.features import targets

    y = targets(store, positions, cfg.task)
    # 240 training windows are enough to learn the level, not to compete with a full run.
    assert np.mean(np.abs(y - prediction[..., 3]) / y) < 0.06

    model.save(tmp_path)
    loaded = NeuralForecaster.load(tmp_path, cfg.task, _small_nn(cfg))
    np.testing.assert_allclose(loaded.predict(store, positions, "nwp"), prediction, rtol=1e-5)


def test_neural_prediction_ignores_future_load(frame, cfg):
    """Same leakage guarantee as the tabular features, proved end to end through the network."""
    store = SeriesStore(frame)
    model = NeuralForecaster(cfg.task, _small_nn(cfg).model_copy(update={"max_epochs": 1}), seed=0)
    model.fit(store, store.index[24 * 100])
    position = np.array([24 * 120])
    clean = model.predict(store, position, "nwp")
    corrupted = frame.copy()
    corrupted.iloc[position[0] + 1 :, corrupted.columns.get_loc("load")] = 1e9
    np.testing.assert_array_equal(clean, model.predict(SeriesStore(corrupted), position, "nwp"))


def test_seasonal_naive_median_is_close_to_last_week(store, cfg):
    model = SeasonalNaive(cfg.task).fit(store, store.index[24 * 150])
    position = np.array([24 * 160])
    prediction = model.predict(store, position, "none")
    target = position[0] + 18
    assert abs(prediction[0, 0, 3] - store.load[target - 168]) < 1000


def test_ridge_beats_naive_on_structured_data(store, cfg):
    from gridcast.features import targets

    end = store.index[24 * 150]
    positions = np.arange(24 * 151, 24 * 170, 24)
    y = targets(store, positions, cfg.task)
    ridge = RidgeForecaster(cfg.task, alpha=1.0).fit(store, end).predict(store, positions, "nwp")
    naive = SeasonalNaive(cfg.task).fit(store, end).predict(store, positions, "none")
    assert np.abs(y - ridge[..., 3]).mean() < np.abs(y - naive[..., 3]).mean()
    assert pd.notna(ridge).all()
