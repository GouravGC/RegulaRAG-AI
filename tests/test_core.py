from app.services.ingestion import chunk_pages
from app.state import clear_session, get_session

def test_chunk_pages_preserves_source_page():
    pages = [{"text": "Regulatory guidance " * 100, "page": 7}]
    chunks = chunk_pages("policy.pdf", pages)
    assert chunks
    assert all(item["source"] == "policy.pdf" for item in chunks)
    assert all(item["page"] == 7 for item in chunks)

def test_session_isolation():
    clear_session("session-alpha")
    clear_session("session-beta")
    a = get_session("session-alpha")
    b = get_session("session-beta")
    a.chunks.append({"text": "private alpha"})
    assert len(a.chunks) == 1
    assert len(b.chunks) == 0

def test_chunk_overlap_configuration_produces_chunks():
    pages = [{"text": "A" * 2600, "page": 1}]
    chunks = chunk_pages("long.txt", pages)
    assert len(chunks) >= 2
