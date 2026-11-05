# Serving image. Installs CPU-only torch from pinned requirements, then runs as non-root.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install "uv==0.13.7"

WORKDIR /app
COPY requirements.lock pyproject.toml README.md ./
RUN uv pip install --system -r requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

COPY src ./src
RUN uv pip install --system --no-deps .

COPY artifacts/model ./artifacts/model

RUN useradd --create-home --uid 10001 gridcast
USER gridcast

ENV GRIDCAST_MODEL_DIR=/app/artifacts/model OMP_NUM_THREADS=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "gridcast.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
