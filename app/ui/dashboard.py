
import uuid

import requests
import streamlit as st


# --------------------------------------------------
# Backend configuration
# --------------------------------------------------

APP_NAME = "RegulaRAG AI"

API_BASE_URL = "https://regularag-ai-latest.onrender.com"

TOP_K = 5


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
        raise RuntimeError(
            "The backend request timed out. Please try again."
        )

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

    st.caption(
        "Upload regulatory documents or crawl websites "
        "to build your private session index."
    )

    # --------------------------------------------------
    # PDF / TXT / Markdown document ingestion
    # --------------------------------------------------

    st.subheader("📄 Document Loader")

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
            with st.spinner(
                "Extracting, chunking and embedding document..."
            ):
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

    # --------------------------------------------------
    # Website URL ingestion
    # --------------------------------------------------

    st.subheader("🌐 Website Knowledge Loader")

    website_url = st.text_input(
        "Website URL",
        placeholder="https://www.rbi.org.in/",
        help=(
            "Enter the starting URL of a regulatory website. "
            "Only accessible, permitted pages will be crawled."
        ),
        key="website_url",
    )

    max_pages = st.slider(
        "Maximum pages to crawl",
        min_value=1,
        max_value=40,
        value=40,
        key="website_max_pages",
    )

    if st.button(
        "🌐 Index Website",
        disabled=not website_url.strip(),
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Crawling website, extracting content and building index..."
            ):
                result = api_request(
                    "POST",
                    "/websites",
                    json={
                        "session_id": st.session_state.session_id,
                        "url": website_url.strip(),
                        "max_pages": max_pages,
                    },
                )

            st.session_state.documents.append(result)

            st.success(
                f"Website indexed successfully! "
                f"{result['pages_crawled']} pages crawled and "
                f"{result['chunks_added']} chunks added."
            )

            st.json(result)

        except Exception as exc:
            st.error(str(exc))

    st.divider()

    # --------------------------------------------------
    # Session management
    # --------------------------------------------------

    st.subheader("Session")

    st.code(st.session_state.session_id[:8])

    st.metric(
        "Indexed sources",
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

    st.subheader("Ask your regulatory knowledge base")

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

                        source_name = source.get("source", "Unknown source")
                        page_number = source.get("page")

                        if page_number is not None:
                            st.markdown(
                                f"**[S{index}] {source_name} "
                                f"— Page {page_number}**"
                            )
                        else:
                            st.markdown(
                                f"**[S{index}] {source_name}**"
                            )

                        if source.get("source_url"):
                            st.markdown(
                                f"[Open source website]({source['source_url']})"
                            )

                        st.caption(source.get("text", "")[:1200])

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

                                source_name = source.get(
                                    "source",
                                    "Unknown source",
                                )

                                page_number = source.get("page")

                                if page_number is not None:
                                    st.markdown(
                                        f"**[S{index}] {source_name} "
                                        f"— Page {page_number}**"
                                    )
                                else:
                                    st.markdown(
                                        f"**[S{index}] {source_name}**"
                                    )

                                if source.get("source_url"):
                                    st.markdown(
                                        f"[Open source website]"
                                        f"({source['source_url']})"
                                    )

                                st.caption(
                                    source.get("text", "")[:1200]
                                )

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
        "Indexed sources",
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
