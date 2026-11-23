
import requests
import numpy as np

from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
)


class EmbeddingError(RuntimeError):
    pass


def embed_texts(texts: list[str]) -> np.ndarray:
    if not OPENROUTER_API_KEY:
        raise EmbeddingError(
            "OPENROUTER_API_KEY is missing from backend configuration."
        )

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    base_url = OPENROUTER_BASE_URL.rstrip("/")
    endpoint = f"{base_url}/embeddings"

    all_vectors = []

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/GouravGC/RegulaRAG-AI",
        "X-Title": "RegulaRAG AI",
    }

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": batch,
                    "encoding_format": "float",
                },
                timeout=90,
                allow_redirects=False,
            )

        except requests.RequestException as exc:
            raise EmbeddingError(
                f"Embedding connection failed: {str(exc)[:300]}"
            ) from exc

        if response.is_redirect:
            raise EmbeddingError(
                "Embedding endpoint returned an HTTP redirect. "
                f"Status: {response.status_code}. "
                f"Redirect target: {response.headers.get('Location', 'unknown')}. "
                "Check OPENROUTER_BASE_URL."
            )

        if not response.ok:
            raise EmbeddingError(
                f"Embedding API error {response.status_code}: "
                f"{response.text[:500]}. "
                f"Endpoint: {endpoint}. "
                f"Authorization header present: "
                f"{bool(response.request.headers.get('Authorization'))}"
            )

        try:
            payload = response.json()
            rows = sorted(
                payload["data"],
                key=lambda item: item.get("index", 0),
            )
            all_vectors.extend(
                row["embedding"] for row in rows
            )

        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError(
                "Embedding API returned an unexpected response format."
            ) from exc

    vectors = np.asarray(all_vectors, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise EmbeddingError(
            "Embedding API returned an unexpected vector shape."
        )

    return vectors
