
FROM python:3.11-slim

LABEL org.opencontainers.image.title="RegulaRAG AI"
LABEL org.opencontainers.image.description="Source-grounded regulatory research assistant with hybrid RAG and OCR"
LABEL org.opencontainers.image.source="https://github.com/GouravGC/RegulaRAG-AI"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TESSERACT_CMD=/usr/bin/tesseract

WORKDIR /app

# System dependencies and OCR engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    fonts-dejavu-core \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application
COPY . .

RUN chmod +x /app/start.sh

EXPOSE 8501

CMD ["/app/start.sh"]
