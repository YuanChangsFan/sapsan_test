FROM python:3.11-slim

# Устанавливаем зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Переменная окружения для Mistral API Key
ENV MISTRAL_API_KEY=

# Запускаем приложение
CMD ["uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "8000"]