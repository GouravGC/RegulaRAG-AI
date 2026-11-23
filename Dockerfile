
FROM python:3.11-slim

LABEL org.opencontainers.image.title="RegulaRAG AI"
LABEL org.opencontainers.image.description="Source-grounded regulatory research assistant with hybrid RAG and OCR"
LABEL org.opencontainers.image.source="https://github.com/GouravGC/RegulaRAG-AI"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TESSERACT_CMD=/usr/bin/tesseract

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    tesseract-ocr \
    tesseract-ocr-eng \
    fonts-dejavu-core \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Normalize the startup script inside the Linux image.
# Remove UTF-8 BOM and Windows line endings.
RUN python -c "from pathlib import Path; p=Path('/app/start.sh'); b=p.read_bytes().removeprefix(bytes.fromhex('efbbbf')); p.write_bytes(b.replace(b'\r\n',b'\n').replace(b'\r',b'\n'))" \
    && chmod 755 /app/start.sh \
    && test "$(head -c 2 /app/start.sh)" = '#!' \
    && bash -n /app/start.sh

EXPOSE 8501

# Run through Bash instead of executing the script directly.
CMD ["/bin/bash", "/app/start.sh"]
