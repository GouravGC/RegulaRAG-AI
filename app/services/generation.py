import requests
from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, CHAT_MODEL

def generate_answer(question: str, sources: list[dict]) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to .env.")
    if not sources:
        return "I couldn't find supporting evidence in the uploaded documents. Please upload a relevant source or rephrase your question."

    context = "\n\n".join(
        f"[S{i}] Source: {s['source']} | Page: {s['page']}\n{s['text']}"
        for i, s in enumerate(sources, start=1)
    )
    system = (
        "You are RegulaRAG, a regulatory research assistant. Answer only from the supplied evidence. "
        "Never invent laws, clauses, dates, thresholds, or obligations. If evidence is insufficient, say so. "
        "Cite every material claim using [S1], [S2] exactly as supplied. Distinguish direct requirements "
        "from interpretation. This is informational research, not legal advice."
    )
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
            ],
            "temperature": 0.1,
        },
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"Chat API error {response.status_code}: {response.text[:500]}")
    return response.json()["choices"][0]["message"]["content"]
