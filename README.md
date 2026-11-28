# RegulaRAG AI

**A resource-conscious AI-powered regulatory research assistant built with Streamlit, FastAPI, OpenRouter, FAISS, BM25, and Reciprocal Rank Fusion (RRF).**

RegulaRAG AI helps users explore regulatory documents, retrieve relevant provisions, and generate evidence-grounded answers with source references. It combines semantic and keyword-based retrieval to make regulatory research more accessible and traceable.

## Project Links

| Resource | Link |
|---|---|
| Frontend | https://regularag-ai.streamlit.app/ |
| Backend API | https://regularag-ai-latest.onrender.com/ |
| API Documentation | https://regularag-ai-latest.onrender.com/docs |
| GitHub Repository | https://github.com/GouravGC/RegulaRAG-AI |
| CI/CD Workflow | https://github.com/GouravGC/RegulaRAG-AI/actions/runs/45164443080/job/137657652604 |
| Project Evidence | https://github.com/GouravGC/RegulaRAG-AI/tree/main/evidences |
| Video Demonstration | https://youtu.be/CXAg9eyFtto |

## Features

- **Document Upload:** Upload PDF, TXT, and Markdown files.
- **Text Extraction:** Extract document content while preserving PDF page metadata.
- **OCR Support:** OCR fallback for scanned-document processing where configured.
- **Remote Embeddings:** Generate document embeddings through the OpenRouter API.
- **Vector Search:** Build session-scoped, in-memory FAISS indexes.
- **Keyword Search:** Use BM25 to retrieve relevant terms and provisions.
- **Hybrid Retrieval:** Combine dense and keyword search using Reciprocal Rank Fusion (RRF).
- **Evidence-Grounded Answers:** Generate responses using retrieved document context.
- **Source References:** Display source markers linked to document evidence and PDF pages.
- **Session Isolation:** Maintain document indexes within individual application sessions.
- **Session Cleanup:** Allow users to explicitly clear session data.
- **Containerization:** Docker-based backend deployment.
- **Automated Testing:** Pytest-based test suite.
- **CI/CD:** GitHub Actions workflow for automated project checks.

## Architecture

```text
                  User
                   |
                   v
          Streamlit Frontend
                   |
                   v
             FastAPI Backend
                   |
                   v
           Document Ingestion
                   |
          +--------+--------+
          |                 |
          v                 v
     Text Extraction   PDF Metadata
          |
          v
      Text Chunking
          |
          v
    OpenRouter Embeddings
          |
          v
    +-----+------+
    |            |
    v            v
   FAISS        BM25
 Dense Search  Keyword Search
    |            |
    +------+-----+
           |
           v
   Reciprocal Rank Fusion
           |
           v
    Retrieved Evidence
           |
           v
   OpenRouter Chat Model
           |
           v
   Grounded Answer + Sources
           |
           v
      Streamlit UI
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Frontend | Streamlit |
| Backend | FastAPI |
| API Server | Uvicorn |
| LLM and Embeddings | OpenRouter |
| Vector Search | FAISS |
| Keyword Retrieval | BM25 |
| Ranking | Reciprocal Rank Fusion |
| PDF Processing | PyMuPDF, pypdf |
| OCR | RapidOCR where configured |
| Containerization | Docker |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Frontend Hosting | Streamlit Community Cloud |
| Backend Hosting | Render |

## Current Demonstration

The application has successfully processed a regulatory PDF:

**Reserve Bank of India (Commercial Banks – Miscellaneous) Directions, 2025**

Latest successful indexing result:

| Metric | Result |
|---|---:|
| PDF pages | 22 |
| Indexed chunks | 80 |
| Embedding dimensions | 2,048 |
| Vector backend | FAISS |
| Indexing status | Successful |

The initial RAG test successfully retrieved relevant evidence from multiple PDF pages and generated an answer with source references.

The application has demonstrated its document ingestion, embedding, hybrid retrieval, and evidence-grounded answer generation workflow.

## RAG Evaluation Framework

The next evaluation stage focuses on three important dimensions of retrieval-augmented generation quality.

### 1. Retrieval Recall@5

Measures whether the correct statutory provision appears among the top five retrieved chunks.

### 2. Citation Accuracy

Checks whether generated answers reference the correct PDF pages and statutory provisions.

### 3. Answer Correctness

Evaluates whether generated answers accurately represent the original source, including mandatory requirements, advisory provisions, exceptions, and conditions.

**Proposed evaluation targets (not yet verified benchmark results):**

| Metric | Target |
|---|---:|
| Retrieval Recall@5 | ≥ 90% |
| Citation Accuracy | ≥ 95% |
| Answer Correctness | ≥ 90% |

Evaluation should use manually verified reference questions and source provisions. These targets must not be interpreted as measured project performance.

## Local Setup — Windows CMD

### 1. Clone the repository

```cmd
git clone https://github.com/GouravGC/RegulaRAG-AI.git
cd RegulaRAG-AI
```

### 2. Create the Conda environment

```cmd
conda create -n regularag python=3.11.10 -y
conda activate regularag
```

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

```cmd
copy .env.example .env
notepad .env
```

Add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Configure the chat and embedding models according to the environment variable names supported by the project.

Embeddings use OpenRouter's separate `/embeddings` endpoint. Model availability and free-tier limits may change, so verify the configured model in the OpenRouter model catalogue.

### 5. Run the application

```cmd
python run.py
```

Open:

http://127.0.0.1:8501

## Docker Setup

Build and start the application using Docker Compose:

```cmd
docker compose up --build
```

Open:

http://localhost:8501

## Run Tests

Execute the test suite:

```cmd
pytest -q
```

## Deployment

RegulaRAG AI uses a separate frontend and backend deployment.

- **Frontend:** Streamlit Community Cloud.
- **Backend:** FastAPI deployed on Render.
- **Containerization:** Docker.
- **CI/CD:** GitHub Actions.

The backend exposes API endpoints for health checks, document ingestion, question answering, and session management.

Live API documentation:

https://regularag-ai-latest.onrender.com/docs

## Deployment Considerations

This project intentionally uses in-process, session-scoped vector indexes to keep infrastructure requirements relatively low.

Important considerations:

- Render Free instances have limited RAM and CPU resources.
- The filesystem on free instances is ephemeral.
- In-memory indexes are not persistent across process restarts or spin-downs.
- Users may need to re-upload documents after a backend restart.
- Upload limits should remain conservative.
- Large PDFs and concurrent workloads may require additional resources.

For larger deployments, consider persistent object storage, a managed database, durable vector storage, background processing, and appropriately sized infrastructure.

## Security and Limitations

- Never commit `.env` files or API credentials.
- Session IDs are isolation keys, not authentication mechanisms.
- Authentication and rate limiting should be implemented before handling sensitive documents in production.
- OCR availability depends on the configured processing pipeline.
- Citations identify source files and extracted PDF pages; always verify important regulatory interpretations against the original publication.
- Model responses can contain errors or incomplete interpretations.
- This application is an informational research assistant, not a substitute for professional legal or regulatory advice.

## Future Improvements

- Build a manually verified regulatory evaluation dataset.
- Implement automated Recall@5 measurement.
- Add citation verification against source provisions.
- Evaluate mandatory versus advisory language.
- Introduce persistent document and vector storage.
- Improve multi-user session management.
- Expand automated regression and integration testing.

## Author

**Gourav Chhatwani**

GitHub: https://github.com/GouravGC

---

**RegulaRAG AI — Making regulatory document research more accessible, searchable, and evidence-driven.**