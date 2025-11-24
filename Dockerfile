FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    git \
    curl \
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

# Устанавливаем CPU-only версию torch ПЕРЕД poetry install
# Это предотвратит попытку poetry установить CUDA-версию с nvidia-cudnn-cu12
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Теперь устанавливаем зависимости через poetry
# Poetry увидит, что torch уже установлен, и пропустит его установку
RUN poetry install --no-interaction --only main || \
    (echo "Poetry install failed, trying to fix torch..." && \
     pip install --no-cache-dir --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu && \
     poetry install --no-interaction --only main)

# Убеждаемся, что torch остался CPU-only версией (на случай если poetry установил CUDA-версию)
RUN pip install --no-cache-dir --upgrade --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cpu || true

# Ставим llama-cpp-python вручную — pip сам подберёт правильный wheel для Linux
RUN pip install --no-cache-dir llama-cpp-python==0.2.89

COPY models/reranker/logreg_reranker.pkl /app/models/reranker/logreg_reranker.pkl

COPY . /app

ENV PYTHONPATH=/app/src

CMD ["poetry", "run", "start-api"]
