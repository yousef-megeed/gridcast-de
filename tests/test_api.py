"""API tests against a real, small bundle containing a trained neural member."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridcast.features import SeriesStore
from gridcast.models.neural import NeuralForecaster
from gridcast.serve.app import app
from tests.conftest import synthetic_frame


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cfg):
    directory = tmp_path_factory.mktemp("bundle")
    frame = synthetic_frame()
    nn_cfg = cfg.nn.model_copy(
        update={
            "d_model": 16, "n_heads": 2, "n_layers": 1, "context_hours": 168, "patch_hours": 24,
            "origin_stride_hours": 24, "max_epochs": 1,
        }
    )  # fmt: skip
    small = cfg.model_copy(update={"nn": nn_cfg})
    NeuralForecaster(cfg.task, nn_cfg).fit(SeriesStore(frame), frame.index[24 * 100]).save(
        directory / "attention_net"
    )
    (directory / "config.json").write_text(json.dumps(small.model_dump(mode="json")))
    margins = {
        f"{level:.2f}": {"alpha": 1 - level, "margins_mw": [100.0] * 24}
        for level in cfg.conformal.levels
    }
    (directory / "conformal.json").write_text(json.dumps(margins))
    frame.iloc[-(cfg.task.history_hours + 200) :].to_parquet(directory / "demo_snapshot.parquet")
    files = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }
    manifest = {
        "model_name": "gridcast-de",
        "package_version": "test",
        "config_hash": "abc",
        "selected_model": "attention_net",
        "members": ["attention_net"],
        "backtest": {},
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture(scope="module")
def client(bundle):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GRIDCAST_MODEL_DIR", str(bundle))
        with TestClient(app) as test_client:
            yield test_client


def _payload(frame: pd.DataFrame | None = None, history: int = 700) -> dict:
    frame = synthetic_frame() if frame is None else frame
    load = frame.iloc[100 : 100 + history]
    weather = frame.iloc[100 : 100 + history + 41]
    return {
        "load": [{"time": t.isoformat(), "load_mw": float(v)} for t, v in load["load"].items()],
        "weather": [
            {
                "time": t.isoformat(),
                "temperature_c": float(row.temp),
                "shortwave_radiation_wm2": float(row.rad),
                "wind_speed_kmh": float(row.wind),
            }
            for t, row in weather.iterrows()
        ],
    }


def test_health_and_model_card(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert client.get("/v1/model").json()["selected_model"] == "attention_net"


def test_forecast_contract(client):
    response = client.post("/v1/forecast", json=_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["points"]) == 24
    issue = pd.Timestamp(body["issue_time"])
    assert pd.Timestamp(body["points"][0]["time"]) == issue + pd.Timedelta(hours=18)
    for point in body["points"]:
        assert point["actual_mw"] is None  # a stateless request never carries the future
        inner, outer = point["intervals"]["0.50"], point["intervals"]["0.90"]
        assert outer["lower_mw"] <= inner["lower_mw"] <= point["median_mw"]
        assert point["median_mw"] <= inner["upper_mw"] <= outer["upper_mw"]


def test_conformal_margin_is_applied(client):
    point = client.post("/v1/forecast", json=_payload()).json()["points"][0]
    assert point["intervals"]["0.90"]["upper_mw"] == pytest.approx(
        point["quantiles_mw"]["0.95"] + 100.0
    )


def test_request_is_deterministic(client):
    payload = _payload()
    assert (
        client.post("/v1/forecast", json=payload).json()["points"]
        == client.post("/v1/forecast", json=payload).json()["points"]
    )


def test_short_history_is_rejected(client):
    response = client.post("/v1/forecast", json=_payload(history=100))
    assert response.status_code == 422
    assert "history" in response.json()["detail"]


def test_gap_is_rejected(client):
    payload = _payload()
    del payload["load"][50]
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 422
    assert "consecutive" in response.text


def test_naive_timestamp_is_rejected(client):
    payload = _payload()
    payload["load"][0]["time"] = "2024-01-01T00:00:00"
    assert client.post("/v1/forecast", json=payload).status_code == 422


def test_negative_load_is_rejected(client):
    payload = _payload()
    payload["load"][3]["load_mw"] = -5
    assert client.post("/v1/forecast", json=payload).status_code == 422


def test_weather_must_cover_target(client):
    payload = _payload()
    payload["weather"] = payload["weather"][:-5]
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 422
    assert "extend" in response.text


def test_implausible_level_warns(client):
    frame = synthetic_frame()
    frame["load"] = frame["load"] * 2
    assert any("outside" in w for w in client.post(
        "/v1/forecast", json=_payload(frame)).json()["warnings"])  # fmt: skip


def test_demo_endpoint_works_offline(client):
    response = client.get("/v1/forecast/demo")
    assert response.status_code == 200
    assert response.json()["points"][0]["actual_mw"] is not None
    # The demo follows the operational schedule: issued at 06:00 for a day starting at midnight.
    issue = pd.Timestamp(response.json()["issue_time"]).tz_convert("Europe/Berlin")
    first = pd.Timestamp(response.json()["points"][0]["time"]).tz_convert("Europe/Berlin")
    assert (issue.hour, first.hour) == (6, 0)


def test_live_failure_returns_503(client, monkeypatch):
    import httpx

    from gridcast.data import sources

    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(sources, "fetch_smard_series", fail)
    assert client.get("/v1/forecast/live").status_code == 503


def test_metrics_and_ui(client):
    client.get("/health")
    body = client.get("/metrics").text
    assert 'gridcast_requests_total{endpoint="/health",status="200"}' in body
    assert "GridCast DE" in client.get("/").text


def test_tampered_bundle_is_refused(bundle, tmp_path):
    import shutil

    from gridcast.integrity import verify_bundle

    copy = tmp_path / "copy"
    shutil.copytree(bundle, copy)
    (copy / "conformal.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(copy)


def test_missing_bundle_reports_unhealthy(monkeypatch, tmp_path):
    monkeypatch.setenv("GRIDCAST_MODEL_DIR", str(tmp_path / "nope"))
    with TestClient(app) as local:
        assert local.get("/health").status_code == 503
        assert local.post("/v1/forecast", json=_payload()).status_code == 503


def test_latest_issue_is_six_local_time():
    from gridcast.serve.service import latest_issue

    issue = latest_issue(pd.Timestamp("2026-11-05T15:00:00Z"), 18)
    assert issue == pd.Timestamp("2026-11-05T05:00:00Z")  # 06:00 CET
    early = latest_issue(pd.Timestamp("2026-11-05T03:00:00Z"), 18)
    assert early == pd.Timestamp("2026-11-04T05:00:00Z")
    assert np.isfinite(issue.timestamp())
