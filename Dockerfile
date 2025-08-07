# 1. Базовый образ
FROM python:3.12-slim

# 2. Системные зависимости для сборки C++ расширений (и psycopg2)
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

# 3. Устанавливаем Poetry
ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s ~/.local/bin/poetry /usr/local/bin/poetry

# 4. Отключаем создание venv внутри Poetry
ENV POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 5. Копируем только pyproject.toml и poetry.lock
COPY pyproject.toml poetry.lock* /app/

# 6. Устанавливаем только основные зависимости
RUN poetry install --no-interaction --only main

# 7. Устанавливаем llama-cpp-python вручную (опционально, только если нужен в образе)
RUN pip install --no-cache-dir llama-cpp-python==0.2.89

# 8. Копируем reranker
COPY models/reranker/logreg_reranker.pkl /app/models/reranker/logreg_reranker.pkl

# 9. Копируем весь код проекта
COPY . /app

# 10. Задаём PYTHONPATH
ENV PYTHONPATH=/app/src

# 11. Точка входа
CMD ["poetry", "run", "start-api"]
