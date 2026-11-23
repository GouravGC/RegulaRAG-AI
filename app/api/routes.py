
import logging
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import MAX_UPLOAD_MB, TOP_K
from app.state import clear_session, get_session
from app.services.ingestion import index_document
from app.services.retrieval import retrieve
from app.services.generation import generate_answer

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RegulaRAG AI",
    description="Source-grounded regulatory intelligence API",
    version="1.0.0",
)


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=3, max_length=5000)
    top_k: int = Field(default=TOP_K, ge=1, le=20)


@app.get("/")
def root():
    return {
        "name": "RegulaRAG AI",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "service": "RegulaRAG AI"}


def validate_session_id(session_id: str) -> str:
    try:
        return str(UUID(session_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID. A valid UUID is required.",
        )


@app.post("/documents")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    session_id = validate_session_id(session_id)

    filename = file.filename or "uploaded_document"
    allowed_extensions = {".pdf", ".txt", ".md", ".markdown"}

    from pathlib import Path

    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Supported formats: PDF, TXT, MD, MARKDOWN.",
        )

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    try:
        content = await file.read(max_bytes + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.",
            )

        result = index_document(
            session_id=session_id,
            filename=filename,
            content=content,
        )

        return {
            "status": "success",
            **result,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(
            status_code=502,
            detail=f"Document indexing failed: {str(exc)[:300]}",
        )
    finally:
        await file.close()


@app.post("/ask")
def ask_question(request: AskRequest):
    session_id = validate_session_id(request.session_id)
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    session = get_session(session_id)

    if not session.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents are indexed in this session. Upload a document first.",
        )

    try:
        sources = retrieve(
            session_id=session_id,
            question=question,
            top_k=request.top_k,
        )

        answer = generate_answer(
            question=question,
            sources=sources,
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(sources),
            "session_id": session_id,
        }

    except Exception as exc:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=502,
            detail=f"Question answering failed: {str(exc)[:300]}",
        )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    session_id = validate_session_id(session_id)

    clear_session(session_id)

    return {
        "status": "success",
        "message": "Session cleared.",
        "session_id": session_id,
    }
