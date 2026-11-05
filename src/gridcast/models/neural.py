"""Cross-attention quantile network in PyTorch.

Past load and weather are cut into patches and encoded by a Transformer. Each of the 24
target hours becomes a query built from what is known about that hour in advance. The query
attends over the past patches, and the head predicts a *correction* to a seasonal anchor.

Three design decisions are worth defending:

* Residual learning: the network starts from "same hour in previous weeks", so it only has
  to learn departures from a strong seasonal baseline.
* Instance normalisation: inputs and outputs are relative to the trailing weekly level, which
  gives robustness to the long-term decline in demand.
* Monotone quantile head: quantiles are the median plus or minus cumulative softplus
  increments, so they cannot cross by construction.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gridcast.config import NNConfig, TaskConfig
from gridcast.features import WEEK_HOURS, SeriesStore, issue_levels, targets, training_positions
from gridcast.models.base import Forecaster

logger = logging.getLogger(__name__)

PAST_CHANNELS = 8
FUTURE_CHANNELS = 23


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    diff = y.unsqueeze(-1) - pred
    return torch.max(quantiles * diff, (quantiles - 1.0) * diff).mean()


class MonotoneQuantileHead(nn.Module):
    """Maps features to non-crossing quantiles around a central median."""

    def __init__(self, d_model: int, quantiles: torch.Tensor):
        super().__init__()
        self.register_buffer("quantiles", quantiles)
        self.median_index = int(torch.argmin(torch.abs(quantiles - 0.5)))
        self.linear = nn.Linear(d_model, quantiles.numel())

    def forward(self, features: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        raw = self.linear(features)
        k = self.median_index
        median = anchor + raw[..., k]
        steps = nn.functional.softplus(raw) + 1e-4
        upper = median.unsqueeze(-1) + torch.cumsum(steps[..., k + 1 :], dim=-1)
        lower = median.unsqueeze(-1) - torch.cumsum(steps[..., :k].flip(-1), dim=-1).flip(-1)
        return torch.cat([lower, median.unsqueeze(-1), upper], dim=-1)


class DecoderBlock(nn.Module):
    """Pre-norm self-attention (across target hours), cross-attention (to the past), MLP."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )  # fmt: skip

    def forward(self, query: torch.Tensor, memory: torch.Tensor):
        normed = self.norm_self(query)
        query = query + self.self_attn(normed, normed, normed, need_weights=False)[0]
        attended, weights = self.cross_attn(self.norm_cross(query), memory, memory)
        query = query + attended
        return query + self.mlp(self.norm_mlp(query)), weights


class QuantileAttentionNet(nn.Module):
    def __init__(
        self,
        context_hours: int,
        patch_hours: int,
        horizon: int,
        quantiles: torch.Tensor,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if context_hours % patch_hours:
            raise ValueError("context_hours must be divisible by patch_hours")
        self.patch_hours = patch_hours
        n_patches = context_hours // patch_hours
        self.patch = nn.Linear(patch_hours * PAST_CHANNELS, d_model)
        self.patch_position = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )  # fmt: skip
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.future = nn.Linear(FUTURE_CHANNELS, d_model)
        self.lead = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = MonotoneQuantileHead(d_model, quantiles)

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        anchor: torch.Tensor,
        return_attention: bool = False,
    ):
        batch, hours, _ = past.shape
        patches = past.reshape(batch, hours // self.patch_hours, -1)
        memory = self.encoder(self.patch(patches) + self.patch_position)
        query = self.future(future) + self.lead
        attention = None
        for block in self.blocks:
            query, attention = block(query, memory)
        output = self.head(self.norm(query), anchor)
        return (output, attention) if return_attention else output


class NeuralForecaster(Forecaster):
    name = "attention_net"

    def __init__(self, task: TaskConfig, cfg: NNConfig, seed: int = 42):
        super().__init__(task)
        self.cfg = cfg
        self.seed = seed
        self.device = torch.device(cfg.device)
        self.net: QuantileAttentionNet | None = None
        self.stats: dict[str, list[float]] = {}
        self._best_epochs: int | None = None

    @property
    def best_iterations(self) -> int | None:
        return self._best_epochs

    def _new_net(self) -> QuantileAttentionNet:
        return QuantileAttentionNet(
            self.cfg.context_hours,
            self.cfg.patch_hours,
            self.task.horizon_hours,
            torch.tensor(self.task.quantiles, dtype=torch.float32),
            self.cfg.d_model,
            self.cfg.n_heads,
            self.cfg.n_layers,
            self.cfg.dropout,
        ).to(self.device)

    def _fit_scaler(self, store: SeriesStore, end_pos: int) -> None:
        self.stats = {}
        for variable in ("temp", "rad", "wind"):
            values = store.frame[variable].to_numpy(float)[:end_pos]
            self.stats[variable] = [float(np.nanmean(values)), float(np.nanstd(values))]

    def _scaled(self, store: SeriesStore, variable: str, source: str) -> np.ndarray:
        mean, std = self.stats[variable]
        return np.nan_to_num((store.weather(source, variable) - mean) / std)

    def _tensors(self, store: SeriesStore, positions: np.ndarray, source: str):
        """Build ``past``, ``future``, ``anchor`` and ``level`` for a batch of issue positions."""
        context = np.arange(self.cfg.context_hours - 1, -1, -1)
        past_index = positions[:, None] - context[None, :]
        leads = np.arange(self.task.gap_hours, self.task.gap_hours + self.task.horizon_hours)
        target = positions[:, None] + leads[None, :]
        level = issue_levels(store, positions)
        scale = level[:, None]

        hour = store.cal("hour")
        dow = store.cal("dow")
        doy = store.cal("doy")
        share = store.cal("holiday_share")
        past = np.stack(
            [
                store.load[past_index] / scale - 1.0,
                self._past_weather[0][past_index],
                self._past_weather[1][past_index],
                share[past_index],
                np.sin(2 * np.pi * hour[past_index] / 24),
                np.cos(2 * np.pi * hour[past_index] / 24),
                np.sin(2 * np.pi * dow[past_index] / 7),
                np.cos(2 * np.pi * dow[past_index] / 7),
            ],
            axis=-1,
        )
        lag = {k: store.load[target - k] / scale - 1.0 for k in (168, 336, 504, 672)}
        anchor = (lag[168] + lag[336] + lag[504] + lag[672]) / 4.0
        temp = self._scaled(store, "temp", source)
        rad = self._scaled(store, "rad", source)
        wind = self._scaled(store, "wind", source)
        past_temp_week = self._past_weather[0][
            positions[:, None] - np.arange(WEEK_HOURS)[None, :]
        ].mean(axis=1, keepdims=True)
        future = np.stack(
            [
                temp[target],
                rad[target],
                wind[target],
                temp[target] - past_temp_week,
                np.sin(2 * np.pi * hour[target] / 24),
                np.cos(2 * np.pi * hour[target] / 24),
                np.sin(2 * np.pi * dow[target] / 7),
                np.cos(2 * np.pi * dow[target] / 7),
                np.sin(2 * np.pi * doy[target] / 365.25),
                np.cos(2 * np.pi * doy[target] / 365.25),
                store.cal("is_weekend")[target],
                share[target],
                store.cal("before_holiday")[target],
                store.cal("after_holiday")[target],
                store.cal("bridge_day")[target],
                store.cal("winter_break")[target],
                share[target - 168],
                lag[168],
                lag[336],
                lag[504],
                lag[672],
                np.repeat(store.load[positions][:, None] / scale - 1.0, target.shape[1], axis=1),
                np.repeat(
                    store.load[positions[:, None] - np.arange(24)[None, :]].mean(1, keepdims=True)
                    / scale
                    - 1.0,
                    target.shape[1],
                    axis=1,
                ),
            ],
            axis=-1,
        )
        as_tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        return as_tensor(past), as_tensor(future), as_tensor(anchor), level

    def _prepare(self, store: SeriesStore) -> None:
        self._past_weather = (
            np.nan_to_num(
                (store.frame["temp"].to_numpy(float) - self.stats["temp"][0])
                / self.stats["temp"][1]
            ),
            np.nan_to_num(
                (store.frame["rad"].to_numpy(float) - self.stats["rad"][0]) / self.stats["rad"][1]
            ),
        )

    def _epoch_loss(self, net, store, positions, labels) -> float:
        net.eval()
        total = 0.0
        q = torch.tensor(self.task.quantiles, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for start in range(0, len(positions), 1024):
                chunk = positions[start : start + 1024]
                past, future, anchor, level = self._tensors(store, chunk, "era5")
                y = torch.as_tensor(
                    labels[start : start + 1024] / level[:, None] - 1.0,
                    dtype=torch.float32, device=self.device,
                )  # fmt: skip
                total += pinball_loss(net(past, future, anchor), y, q).item() * len(chunk)
        return total / len(positions)

    def fit(self, store, train_end, val_end=None, fixed_iterations=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        positions = training_positions(
            store, store.index[0], train_end, self.cfg.origin_stride_hours, self.task
        )
        self._fit_scaler(store, int(store.index.searchsorted(train_end)))
        self._prepare(store)
        labels = targets(store, positions, self.task)

        val_positions = val_labels = None
        if val_end is not None:
            val_positions = training_positions(store, train_end, val_end, 1, self.task)
            val_labels = targets(store, val_positions, self.task)

        epochs = fixed_iterations or self.cfg.max_epochs
        net = self._new_net()
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        steps_per_epoch = int(np.ceil(len(positions) / self.cfg.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.cfg.lr, total_steps=epochs * steps_per_epoch, pct_start=0.1
        )
        q = torch.tensor(self.task.quantiles, dtype=torch.float32, device=self.device)

        best_loss, best_state, best_epoch = float("inf"), None, epochs
        since_best = 0
        for epoch in range(1, epochs + 1):
            net.train()
            order = rng.permutation(len(positions))
            running = 0.0
            for start in range(0, len(order), self.cfg.batch_size):
                batch = order[start : start + self.cfg.batch_size]
                past, future, anchor, level = self._tensors(store, positions[batch], "era5")
                y = torch.as_tensor(
                    labels[batch] / level[:, None] - 1.0, dtype=torch.float32, device=self.device
                )
                loss = pinball_loss(net(past, future, anchor), y, q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running += loss.item() * len(batch)
            if val_positions is not None:
                val_loss = self._epoch_loss(net, store, val_positions, val_labels)
                logger.info("epoch %d train %.5f val %.5f", epoch, running / len(order), val_loss)
                if val_loss < best_loss - 1e-6:
                    best_loss, best_epoch = val_loss, epoch
                    best_state = copy.deepcopy(net.state_dict())
                    since_best = 0
                else:
                    since_best += 1
                    if since_best >= self.cfg.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
            self._best_epochs = best_epoch
            self.best_val_loss = best_loss
        else:
            self._best_epochs = epochs
        self.net = net
        return self

    def predict(self, store, positions, source):
        if self.net is None:
            raise RuntimeError("Model is not fitted")
        self._prepare(store)
        self.net.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(positions), 1024):
                past, future, anchor, level = self._tensors(
                    store, positions[start : start + 1024], source
                )
                ratio = self.net(past, future, anchor).cpu().numpy()
                outputs.append((ratio + 1.0) * level[:, None, None])
        return np.concatenate(outputs)

    def attention(self, store: SeriesStore, positions: np.ndarray, source: str) -> np.ndarray:
        """Cross-attention of the last decoder layer, ``(issues, leads, patches)``."""
        if self.net is None:
            raise RuntimeError("Model is not fitted")
        self._prepare(store)
        self.net.eval()
        with torch.no_grad():
            past, future, anchor, _ = self._tensors(store, positions, source)
            return self.net(past, future, anchor, return_attention=True)[1].cpu().numpy()

    def save(self, directory: Path) -> None:
        if self.net is None:
            raise RuntimeError("Model is not fitted")
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / "attention_net.pt")
        (directory / "attention_net.json").write_text(json.dumps({"stats": self.stats}))

    @classmethod
    def load(cls, directory: Path, task: TaskConfig, cfg: NNConfig) -> NeuralForecaster:
        model = cls(task, cfg)
        model.stats = json.loads((directory / "attention_net.json").read_text())["stats"]
        model.net = model._new_net()
        state = torch.load(
            directory / "attention_net.pt", map_location=model.device, weights_only=True
        )
        model.net.load_state_dict(state)
        model.net.eval()
        return model
