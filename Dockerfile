
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

# Normalize start.sh, including UTF-16 and UTF-8 BOM encodings.
RUN python -c "from pathlib import Path; p=Path('/app/start.sh'); b=p.read_bytes(); s=b.decode('utf-16') if b.startswith((b'\xff\xfe',b'\xfe\xff')) else b.decode('utf-8-sig'); s=s.replace('\r\n','\n').replace('\r','\n'); s=s if s.startswith('#!') else '#!/usr/bin/env bash\n'+s; p.write_text(s,encoding='utf-8',newline='\n')" \
    && chmod 755 /app/start.sh \
    && bash -n /app/start.sh

EXPOSE 8501

CMD ["/bin/bash", "/app/start.sh"]
