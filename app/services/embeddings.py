import requests
import numpy as np
from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

class EmbeddingError(RuntimeError):
    pass

def embed_texts(texts: list[str]) -> np.ndarray:
    if not OPENROUTER_API_KEY:
        raise EmbeddingError("OPENROUTER_API_KEY is missing. Add it to your .env file.")
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    all_vectors = []
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "RegulaRAG AI",
    }
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/embeddings",
            headers=headers,
            json={"model": EMBEDDING_MODEL, "input": batch, "encoding_format": "float"},
            timeout=90,
        )
        if not response.ok:
            raise EmbeddingError(f"Embedding API error {response.status_code}: {response.text[:500]}")
        payload = response.json()
        rows = sorted(payload["data"], key=lambda item: item.get("index", 0))
        all_vectors.extend(row["embedding"] for row in rows)

    vectors = np.asarray(all_vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise EmbeddingError("Embedding API returned an unexpected vector shape.")
    return vectors
