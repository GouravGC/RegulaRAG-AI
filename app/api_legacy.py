from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from app.config import APP_NAME, MAX_UPLOAD_MB, TOP_K
from app.state import clear_session, get_session
from app.services.ingestion import index_document
from app.services.retrieval import retrieve
from app.services.generation import generate_answer

app = FastAPI(title=APP_NAME, version="1.0.0")

class AskRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=128)
    question: str = Field(min_length=2, max_length=4000)
    top_k: int = Field(default=TOP_K, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME}

@app.post("/documents")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    try:
        result = index_document(session_id, file.filename or "upload", content)
        return {"status": "indexed", **result}
    except Exception as exc:
        raise HTTPException(400, str(exc))

@app.post("/ask")
def ask(request: AskRequest):
    session = get_session(request.session_id)
    if not session.chunks:
        raise HTTPException(400, "Upload a document before asking questions.")
    try:
        sources = retrieve(request.session_id, request.question, request.top_k)
        answer = generate_answer(request.question, sources)
        return {"answer": answer, "sources": sources}
    except Exception as exc:
        raise HTTPException(502, str(exc))

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"status": "cleared"}
