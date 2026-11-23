import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "RegulaRAG AI"

# Render FastAPI backend
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://regularag-ai-latest.onrender.com",
).rstrip("/")

TOP_K = int(os.getenv("TOP_K", "5"))