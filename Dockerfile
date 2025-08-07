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
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN poetry install --no-interaction --only main

# Ставим llama-cpp-python вручную — pip сам подберёт правильный wheel для Linux
RUN pip install --no-cache-dir llama-cpp-python==0.2.89

COPY models/reranker/logreg_reranker.pkl /app/models/reranker/logreg_reranker.pkl

COPY . /app

ENV PYTHONPATH=/app/src

CMD ["poetry", "run", "start-api"]
