# RegulaRAG AI

A resource-conscious regulatory RAG application built with Streamlit, FastAPI, OpenRouter, FAISS, BM25 and Reciprocal Rank Fusion.

## Features
- Upload PDF, TXT and Markdown files.
- Extract text and preserve PDF page metadata.
- Generate embeddings remotely through OpenRouter.
- Build session-scoped in-memory FAISS indexes.
- Combine dense retrieval and BM25 with RRF.
- Generate evidence-grounded answers with source markers.
- Clear session data explicitly.
- Docker, pytest and GitHub Actions baseline.

## Architecture
Streamlit UI -> FastAPI -> OpenRouter embeddings -> in-memory FAISS + BM25 -> RRF -> OpenRouter chat model.

## Local setup (Windows CMD)
```cmd
conda create -n regularag python=3.11.10 -y
conda activate regularag
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
notepad .env
python run.py
```
Open http://127.0.0.1:8501.

Set `OPENROUTER_API_KEY` in `.env`. Embeddings use the separate OpenRouter `/embeddings` endpoint. Model availability and free-tier limits can change; check the OpenRouter model catalogue if the configured embedding model is unavailable.

## Docker
```cmd
docker compose up --build
```
Open http://localhost:8501.

## Tests
```cmd
pytest -q
```

## Deployment notes
This project intentionally keeps vector indexes in process memory. Render Free has an ephemeral filesystem and limited RAM/CPU; after a restart or spin-down, users must re-upload documents. Keep upload limits conservative. A single free instance is suitable for a small demo, not concurrent production workloads. For larger PDFs, scanned PDFs, large corpora or multiple concurrent users, use persistent object storage and a managed database or paid instance.

## Security and limitations
- Never commit `.env`.
- Session IDs are isolation keys, not authentication. Add real authentication and rate limiting before public production use.
- Scanned PDFs require OCR, not included in this lightweight build.
- Citations identify source file and extracted PDF page. Verify regulatory interpretations against the original publication.
- This is an informational research tool, not legal advice.
