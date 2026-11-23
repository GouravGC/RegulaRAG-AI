
import uuid

import requests
import streamlit as st

from app.config import APP_NAME, API_BASE_URL, TOP_K


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []


# --------------------------------------------------
# API helpers
# --------------------------------------------------

def api_request(method, endpoint, **kwargs):
    url = f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=180,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        raise RuntimeError("The backend request timed out. Please try again.")

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to the RegulaRAG backend. "
            "Check whether the API service is running."
        )

    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)

        raise RuntimeError(f"Backend error: {detail}")


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📚 RegulaRAG AI")

st.caption(
    "Source-grounded regulatory intelligence | "
    "Hybrid RAG | FAISS + BM25 | OpenRouter"
)

st.divider()


# --------------------------------------------------
# Sidebar: Knowledge base
# --------------------------------------------------

with st.sidebar:
    st.title("Knowledge Base")

    st.caption("Upload regulatory documents to build your private session index.")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md", "markdown"],
        help="Supported formats: PDF, TXT and Markdown.",
    )

    if st.button(
        "Index Document",
        disabled=uploaded_file is None,
        use_container_width=True,
    ):
        try:
            with st.spinner("Extracting, chunking and embedding document..."):
                result = api_request(
                    "POST",
                    "/documents",
                    data={
                        "session_id": st.session_state.session_id,
                    },
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    },
                )

            st.session_state.documents.append(result)

            st.success(
                f"Successfully indexed {result['chunks_added']} chunks."
            )

            st.json(result)

        except Exception as exc:
            st.error(str(exc))

    st.divider()

    st.subheader("Session")

    st.code(st.session_state.session_id[:8])

    st.metric(
        "Indexed documents",
        len(st.session_state.documents),
    )

    if st.button(
        "Clear Session",
        use_container_width=True,
    ):
        try:
            api_request(
                "DELETE",
                f"/session/{st.session_state.session_id}",
            )

            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.documents = []

            st.success("Session cleared.")
            st.rerun()

        except Exception as exc:
            st.error(str(exc))

    st.divider()

    st.caption("RegulaRAG AI | Regulatory Research Assistant")


# --------------------------------------------------
# Main tabs
# --------------------------------------------------

tab_chat, tab_lab, tab_eval = st.tabs(
    [
        "💬 Compliance Chat",
        "🔬 Retrieval Laboratory",
        "📊 Evaluation",
    ]
)


# --------------------------------------------------
# Compliance chat
# --------------------------------------------------

with tab_chat:
    st.subheader("Ask your regulatory documents")

    st.write(
        "Ask questions about uploaded policies, regulatory frameworks, "
        "compliance requirements and supporting evidence."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View retrieved evidence"):
                    for index, source in enumerate(
                        message["sources"],
                        start=1,
                    ):
                        st.markdown(
                            f"**[S{index}] {source['source']} "
                            f"— Page {source['page']}**"
                        )

                        st.caption(source["text"][:1200])

    prompt = st.chat_input(
        "Ask a question about your uploaded regulations..."
    )

    if prompt:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(
                "Retrieving evidence and generating a grounded answer..."
            ):
                try:
                    payload = api_request(
                        "POST",
                        "/ask",
                        json={
                            "session_id": st.session_state.session_id,
                            "question": prompt,
                            "top_k": TOP_K,
                        },
                    )

                    answer = payload["answer"]
                    sources = payload.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("View retrieved evidence"):
                            for index, source in enumerate(
                                sources,
                                start=1,
                            ):
                                st.markdown(
                                    f"**[S{index}] {source['source']} "
                                    f"— Page {source['page']}**"
                                )

                                st.caption(source["text"][:1200])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as exc:
                    st.error(str(exc))


# --------------------------------------------------
# Retrieval laboratory
# --------------------------------------------------

with tab_lab:
    st.subheader("Hybrid Retrieval Laboratory")

    st.markdown(
        """
        RegulaRAG combines two retrieval strategies:

        **1. Semantic retrieval — FAISS**

        Finds passages based on semantic similarity using dense embeddings.

        **2. Keyword retrieval — BM25**

        Finds passages using lexical matching and important regulatory terms.

        **3. Reciprocal Rank Fusion**

        Combines the rankings from both retrieval systems to produce a
        unified evidence set for answer generation.
        """
    )

    st.info(
        "Ask a question in Compliance Chat to inspect the evidence "
        "retrieved for that query."
    )

    st.metric(
        "Current retrieval top-k",
        TOP_K,
    )


# --------------------------------------------------
# Evaluation dashboard
# --------------------------------------------------

with tab_eval:
    st.subheader("Evaluation Dashboard")

    st.write(
        "Monitor the current conversation and indexed knowledge base."
    )

    total_questions = sum(
        1
        for message in st.session_state.messages
        if message["role"] == "user"
    )

    total_answers = sum(
        1
        for message in st.session_state.messages
        if message["role"] == "assistant"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Indexed documents",
        len(st.session_state.documents),
    )

    col2.metric(
        "Questions asked",
        total_questions,
    )

    col3.metric(
        "Answers generated",
        total_answers,
    )

    st.divider()

    st.markdown("**Quality evaluation**")

    st.info(
        "Benchmark metrics such as Recall@K, MRR, faithfulness and "
        "answer relevance should only be reported after running a "
        "ground-truth evaluation dataset. No fabricated scores are shown."
    )

    st.code(
        "python -m pytest -q",
        language="bash",
    )
