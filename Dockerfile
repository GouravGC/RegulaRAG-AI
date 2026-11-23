FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    TESSERACT_CMD=/usr/bin/tesseract

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app/config.py app/state.py ./app/
COPY app/api/ ./app/api/
COPY app/services/ ./app/services/

EXPOSE 10000

CMD ["sh", "-c", "exec uvicorn app.api.routes:app --host 0.0.0.0 --port ${PORT:-10000}"]