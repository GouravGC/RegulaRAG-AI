import uuid
import requests
import streamlit as st
from app.config import APP_NAME

API_URL = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000")
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []

st.title("📚 RegulaRAG AI")
st.caption("Source-grounded regulatory intelligence • PDF / TXT / Markdown")

with st.sidebar:
    st.subheader("Knowledge base")
    uploaded = st.file_uploader("Upload a document", type=["pdf", "txt", "md", "markdown"])
    if st.button("Index document", disabled=uploaded is None, use_container_width=True):
        try:
            response = requests.post(
                f"{API_URL}/documents",
                data={"session_id": st.session_state.session_id},
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
            st.session_state.documents.append(result)
            st.success(f"Indexed {result['chunks_added']} chunks from {result['filename']}.")
            st.json(result)
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.write(f"**Session:** `{st.session_state.session_id[:8]}`")
    if st.button("Clear this session", use_container_width=True):
        try:
            requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=10)
        except requests.RequestException:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.documents = []
        st.rerun()

tab_chat, tab_lab, tab_eval = st.tabs(["💬 Compliance chat", "🔬 Retrieval laboratory", "📊 Evaluation"])

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Retrieved evidence"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**[S{i}] {source['source']} — page {source['page']}**")
                        st.caption(source["text"][:1200])
    prompt = st.chat_input("Ask a question about your uploaded regulations...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"session_id": st.session_state.session_id, "question": prompt, "top_k": 5},
                        timeout=180,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    st.markdown(payload["answer"])
                    with st.expander("Retrieved evidence"):
                        for i, source in enumerate(payload["sources"], 1):
                            st.markdown(f"**[S{i}] {source['source']} — page {source['page']}**")
                            st.caption(source["text"][:1200])
                    st.session_state.messages.append({"role": "assistant", "content": payload["answer"], "sources": payload["sources"]})
                except Exception as exc:
                    st.error(str(exc))

with tab_lab:
    st.subheader("Retrieval laboratory")
    st.write("The backend combines semantic FAISS retrieval and BM25 keyword retrieval using Reciprocal Rank Fusion.")
    st.info("Ask a question in the Compliance chat to inspect its retrieved evidence and scores.")

with tab_eval:
    st.subheader("Evaluation dashboard")
    st.write("Run the included evaluation tests locally with `pytest -q`.")
    st.metric("Indexed documents in this UI session", len(st.session_state.documents))
    st.metric("Conversation turns", sum(1 for m in st.session_state.messages if m["role"] == "user"))
    st.caption("Quality metrics are only reported after running a ground-truth evaluation set; no benchmark scores are fabricated.")
