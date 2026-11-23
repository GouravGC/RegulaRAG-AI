import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_setting(name: str, default: str) -> str:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    if value is None or str(value).strip() == "":
        value = os.getenv(name, default)

    return str(value)


APP_NAME = get_setting("APP_NAME", "RegulaRAG AI")
API_BASE_URL = get_setting(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

TOP_K = int(get_setting("TOP_K", "5"))