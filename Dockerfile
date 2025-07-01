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

# 6. Устанавливаем все зависимости **кроме** llama-cpp-python
#    (по-прежнему через Poetry)
RUN poetry install --no-interaction --no-ansi --only main

# 7. Ставим llama-cpp-python вручную через pip
RUN pip install --no-cache-dir llama-cpp-python

# 8. Копируем весь код проекта
COPY . /app

# 9. Задаём PYTHONPATH (если нужно)
ENV PYTHONPATH=/app/src

# 10. Точка входа
CMD ["poetry", "run", "start-api"]
