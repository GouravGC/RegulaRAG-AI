from dataclasses import dataclass, field
from typing import Any
import threading

@dataclass
class SessionIndex:
    chunks: list[dict[str, Any]] = field(default_factory=list)
    vectors: Any = None
    faiss_index: Any = None
    bm25: Any = None
    dimension: int | None = None

SESSIONS: dict[str, SessionIndex] = {}
LOCK = threading.RLock()

def get_session(session_id: str) -> SessionIndex:
    with LOCK:
        return SESSIONS.setdefault(session_id, SessionIndex())

def clear_session(session_id: str) -> None:
    with LOCK:
        SESSIONS.pop(session_id, None)
