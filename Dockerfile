FROM python:3.12-slim

# Устанавливаем Poetry для управления зависимостями
RUN pip install --no-cache-dir poetry

# Создаем и переходим в рабочую директорию
WORKDIR /app

# Копируем файлы с зависимостями
COPY pyproject.toml poetry.lock /app/

# Устанавливаем только основные зависимости проекта
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --only main

# Копируем весь проект в контейнер
COPY . /app

# Устанавливаем PYTHONPATH
ENV PYTHONPATH=/app/src

# Пробрасываем порт
EXPOSE 8000

# Запускаем API по умолчанию
CMD ["poetry", "run", "start-api"]
