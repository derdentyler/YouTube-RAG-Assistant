FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s ~/.local/bin/poetry /usr/local/bin/poetry

ENV POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry install --no-interaction --only main

COPY src/ ./src/
COPY services/orchestrator/ ./services/orchestrator/
COPY services/clients/ ./services/clients/
COPY services/contracts/ ./services/contracts/
COPY config/ ./config/
COPY src/prompts/ ./src/prompts/

ENV PYTHONPATH=/app

CMD ["uvicorn", "services.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]

