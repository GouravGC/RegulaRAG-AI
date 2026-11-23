import numpy as np
from app.state import get_session
from app.services.embeddings import embed_texts

def retrieve(session_id: str, question: str, top_k: int = 5) -> list[dict]:
    session = get_session(session_id)
    if not session.chunks:
        return []
    qvec = embed_texts([question])[0].astype(np.float32)
    qnorm = qvec / max(float(np.linalg.norm(qvec)), 1e-12)

    if session.faiss_index is not None:
        dense_scores, dense_ids = session.faiss_index.search(qnorm.reshape(1, -1), min(len(session.chunks), top_k * 4))
        dense_rank = {int(idx): rank + 1 for rank, idx in enumerate(dense_ids[0]) if idx >= 0}
        dense_score = {int(idx): float(score) for idx, score in zip(dense_ids[0], dense_scores[0]) if idx >= 0}
    else:
        scores = session.vectors @ qnorm
        order = np.argsort(scores)[::-1][:top_k * 4]
        dense_rank = {int(idx): rank + 1 for rank, idx in enumerate(order)}
        dense_score = {int(idx): float(scores[idx]) for idx in order}

    bm_scores = session.bm25.get_scores(question.lower().split())
    bm_order = np.argsort(bm_scores)[::-1][:top_k * 4]
    bm_rank = {int(idx): rank + 1 for rank, idx in enumerate(bm_order)}
    bm_score = {int(idx): float(bm_scores[idx]) for idx in bm_order}

    # Reciprocal Rank Fusion: combine ranks, not incomparable raw scores.
    candidates = set(dense_rank) | set(bm_rank)
    fused = []
    for idx in candidates:
        rrf = 1 / (60 + dense_rank[idx]) if idx in dense_rank else 0
        rrf += 1 / (60 + bm_rank[idx]) if idx in bm_rank else 0
        item = dict(session.chunks[idx])
        item.update({
            "rrf_score": rrf,
            "dense_score": dense_score.get(idx, 0.0),
            "bm25_score": bm_score.get(idx, 0.0),
        })
        fused.append(item)
    return sorted(fused, key=lambda x: x["rrf_score"], reverse=True)[:top_k]
