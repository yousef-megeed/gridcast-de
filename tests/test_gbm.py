"""LightGBM tests. CI runs this file in its own process (see the Makefile and workflow)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.features import SeriesStore, targets


@pytest.fixture(scope="module")
def fitted(frame, cfg):
    from gridcast.models.gbm import GBMForecaster

    small = cfg.gbm.model_copy(update={"num_boost_round": 60, "early_stopping_rounds": 10})
    store = SeriesStore(frame)
    model = GBMForecaster(cfg.task, small).fit(store, store.index[24 * 140], store.index[24 * 165])
    return store, small, model


def test_quantiles_are_sorted_and_accurate(fitted, cfg):
    store, _, model = fitted
    positions = np.arange(24 * 166, 24 * 176, 24)
    prediction = model.predict(store, positions, "nwp")
    assert prediction.shape == (10, 24, 7)
    assert (np.diff(prediction, axis=-1) >= 0).all()
    mape = np.mean(np.abs(targets(store, positions, cfg.task) - prediction[..., 3])
                   / targets(store, positions, cfg.task))  # fmt: skip
    assert mape < 0.05


def test_round_trip(fitted, cfg, tmp_path):
    from gridcast.models.gbm import GBMForecaster

    store, small, model = fitted
    model.save(tmp_path)
    positions = np.array([24 * 170])
    loaded = GBMForecaster.load(tmp_path, cfg.task, small)
    np.testing.assert_allclose(
        loaded.predict(store, positions, "nwp"), model.predict(store, positions, "nwp")
    )


def test_shap_values_sum_to_prediction(fitted):
    store, _, model = fitted
    positions = np.array([24 * 170])
    features, values = model.contributions(store, positions, "nwp")
    raw = model.boosters[0.5].predict(features, pred_contrib=True)
    assert values.shape == (24, features.shape[1])
    np.testing.assert_allclose(raw.sum(axis=1), model.boosters[0.5].predict(features), rtol=1e-6)


def test_no_weather_model_predicts_without_weather_columns(frame, cfg):
    from gridcast.models.gbm import GBMForecaster

    small = cfg.gbm.model_copy(update={"num_boost_round": 20})
    store = SeriesStore(frame)
    model = GBMForecaster(cfg.task, small, weather="none").fit(store, store.index[24 * 140])
    assert "temp" not in model.feature_names
    assert model.predict(store, np.array([24 * 170]), "none").shape == (1, 24, 7)


def test_service_serves_a_lightgbm_bundle_from_disk(fitted, cfg, tmp_path):
    """Regression test: loading boosters from files once crashed the API on startup."""
    import hashlib
    import json

    from gridcast.serve.service import ForecastService

    store, small, model = fitted
    model.save(tmp_path / "lightgbm")
    config = cfg.model_copy(update={"gbm": small})
    (tmp_path / "config.json").write_text(json.dumps(config.model_dump(mode="json")))
    conformal = {
        f"{level:.2f}": {"alpha": 1 - level, "margins_mw": [0.0] * 24}
        for level in cfg.conformal.levels
    }
    (tmp_path / "conformal.json").write_text(json.dumps({"bias_offset_mw": 250.0, **conformal}))
    store.frame.iloc[-(cfg.task.history_hours + 200) :].to_parquet(
        tmp_path / "demo_snapshot.parquet"
    )
    files = {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    manifest = {
        "model_name": "gridcast-de",
        "package_version": "test",
        "config_hash": "abc",
        "selected_model": "lightgbm",
        "members": ["lightgbm"],
        "files": files,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    response = ForecastService(tmp_path).predict_snapshot()
    issue_position = store.positions(pd.DatetimeIndex([response.issue_time]))
    direct = model.predict(store, issue_position, "nwp")[0, 0, 3]
    assert len(response.points) == 24
    # The stored online bias correction must reach the served forecast.
    assert response.points[0].median_mw == pytest.approx(direct + 250.0)
