"""FastAPI application."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from gridcast import __version__
from gridcast.serve.schemas import ForecastRequest, ForecastResponse, HealthResponse

logger = logging.getLogger("gridcast.api")
STATIC = Path(__file__).parent / "static"
REQUESTS = Counter("gridcast_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("gridcast_request_seconds", "Request latency", ["endpoint"])
STARTED = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from gridcast.serve.service import ForecastService

    directory = Path(os.environ.get("GRIDCAST_MODEL_DIR", "artifacts/model"))
    try:
        app.state.service = ForecastService(directory)
        logger.info("Loaded model %s from %s", app.state.service.version, directory)
    except (FileNotFoundError, ValueError) as exc:
        # Keep running so /health can explain the failure; forecast routes answer 503.
        app.state.service = None
        logger.error("Could not load model from %s: %s", directory, exc)
    yield


app = FastAPI(
    title="GridCast DE",
    version=__version__,
    summary="Probabilistic day-ahead forecasts of German electricity grid load.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe(request: Request, call_next):
    route = request.url.path
    began = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - began
    # Label only known routes so that scanners cannot create unbounded metric series.
    label = route if route in {getattr(r, "path", None) for r in app.routes} else "other"
    REQUESTS.labels(label, str(response.status_code)).inc()
    LATENCY.labels(label).observe(duration)
    logger.info("%s %s %d %.1fms", request.method, route, response.status_code, duration * 1000)
    return response


def _service(request: Request):
    service = request.app.state.service
    if service is None:
        raise HTTPException(503, "Model bundle is not loaded")
    return service


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health(request: Request, response: Response) -> HealthResponse:
    """Liveness and readiness. Returns 503 when no model is loaded."""
    service = request.app.state.service
    if service is None:
        response.status_code = 503
    return HealthResponse(
        status="ok" if service else "model_unavailable",
        model_loaded=service is not None,
        model_version=service.version if service else None,
        uptime_seconds=round(time.time() - STARTED, 1),
    )


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/model", tags=["forecast"])
def model_info(request: Request) -> dict:
    """Model card: version, training cut-off, backtest scores and file checksums."""
    return _service(request).manifest


@app.post("/v1/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(payload: ForecastRequest, request: Request) -> ForecastResponse:
    """Stateless forecast from caller-supplied history and weather."""
    from gridcast.serve.service import InputError

    try:
        return _service(request).predict_request(payload)
    except InputError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/v1/forecast/demo", response_model=ForecastResponse, tags=["forecast"])
def forecast_demo(request: Request) -> ForecastResponse:
    """Forecast from the history bundled with the model. Works offline."""
    return _service(request).predict_snapshot()


@app.get("/v1/forecast/live", response_model=ForecastResponse, tags=["forecast"])
def forecast_live(request: Request) -> ForecastResponse:
    """Latest operational forecast from live public data. Cached for 15 minutes."""
    from gridcast.serve.service import InputError

    service = _service(request)
    try:
        return service.predict_live()
    except (httpx.HTTPError, KeyError, InputError) as exc:
        logger.warning("Live data unavailable: %s", exc)
        raise HTTPException(503, f"Live data unavailable: {exc}") from exc
