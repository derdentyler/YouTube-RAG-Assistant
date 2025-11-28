FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s ~/.local/bin/poetry /usr/local/bin/poetry

ENV POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="" \
    FORCE_CUDA="0"

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

# Install CPU-only torch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
RUN poetry install --no-interaction --only main || \
    (echo "Poetry install failed, trying to fix torch..." && \
     pip install --no-cache-dir --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu && \
     poetry install --no-interaction --only main)

# Ensure torch remains CPU-only
RUN pip install --no-cache-dir --upgrade --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cpu || true

COPY src/ ./src/
COPY services/reranker_service/ ./services/reranker_service/
COPY services/contracts/ ./services/contracts/
COPY config/ ./config/
COPY models/reranker/ ./models/reranker/

ENV PYTHONPATH=/app

CMD ["uvicorn", "services.reranker_service.main:app", "--host", "0.0.0.0", "--port", "8003"]

