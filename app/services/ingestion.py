
from pathlib import Path
import io
import os
import shutil
import logging

import fitz
import numpy as np
import pytesseract
from PIL import Image
from rank_bm25 import BM25Okapi

try:
    import faiss
except ImportError:
    faiss = None

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.state import get_session
from app.services.embeddings import embed_texts


logger = logging.getLogger(__name__)

OCR_TIMEOUT = 25
OCR_MAX_DIMENSION = 1800
MIN_TEXT_LENGTH = 40


def configure_tesseract():
    configured_path = os.getenv("TESSERACT_CMD", "").strip()

    if configured_path:
        if not Path(configured_path).is_file():
            raise RuntimeError(
                f"TESSERACT_CMD points to a missing file: {configured_path}"
            )

        pytesseract.pytesseract.tesseract_cmd = configured_path
        return

    detected_path = shutil.which("tesseract")

    if detected_path:
        pytesseract.pytesseract.tesseract_cmd = detected_path
        return

    raise RuntimeError(
        "Tesseract OCR is unavailable. "
        "Run the application using the configured Docker image."
    )


def ocr_pdf_page(page) -> str:
    """Run OCR on a resized grayscale PDF page."""
    configure_tesseract()

    pix = page.get_pixmap(
        matrix=fitz.Matrix(1.0, 1.0),
        colorspace=fitz.csGRAY,
        alpha=False,
    )

    image = Image.open(
        io.BytesIO(pix.tobytes("png"))
    )

    try:
        image.thumbnail(
            (OCR_MAX_DIMENSION, OCR_MAX_DIMENSION),
            Image.Resampling.LANCZOS,
        )

        return pytesseract.image_to_string(
            image,
            lang="eng",
            config="--oem 1 --psm 11",
            timeout=OCR_TIMEOUT,
        ).strip()

    finally:
        image.close()


def extract_pages(filename: str, content: bytes) -> list[dict]:
    suffix = Path(filename).suffix.lower()
    pages = []
    skipped_ocr_pages = []

    if suffix == ".pdf":
        try:
            document = fitz.open(
                stream=content,
                filetype="pdf",
            )

            try:
                if document.needs_pass:
                    raise ValueError(
                        "This PDF is password-protected. "
                        "Upload an unlocked PDF."
                    )

                for page_number, page in enumerate(
                    document,
                    start=1,
                ):
                    text = page.get_text("text").strip()

                    if len(text) < MIN_TEXT_LENGTH:
                        try:
                            ocr_text = ocr_pdf_page(page)

                            if len(ocr_text) > len(text):
                                text = ocr_text

                        except Exception as exc:
                            skipped_ocr_pages.append(page_number)

                            logger.warning(
                                "OCR skipped for PDF page %s: %s",
                                page_number,
                                str(exc)[:200],
                            )

                    if text.strip():
                        pages.append({
                            "text": text,
                            "page": page_number,
                        })

            finally:
                document.close()

        except ValueError:
            raise

        except Exception as exc:
            raise RuntimeError(
                f"PDF extraction failed: {str(exc)[:300]}"
            ) from exc

    elif suffix in {".txt", ".md", ".markdown"}:
        text = content.decode(
            "utf-8-sig",
            errors="replace",
        )

        for page_number, block in enumerate(
            text.split("\f"),
            start=1,
        ):
            if block.strip():
                pages.append({
                    "text": block,
                    "page": page_number,
                })

    else:
        raise ValueError(
            "Supported formats: PDF, TXT, MD, MARKDOWN."
        )

    if not pages:
        raise ValueError(
            "No extractable text found. "
            "The PDF may be blank, unreadable, or image-only."
        )

    if skipped_ocr_pages:
        logger.warning(
            "Document %s indexed with OCR skipped on pages: %s",
            filename,
            skipped_ocr_pages,
        )

    return pages


def chunk_pages(filename: str, pages: list[dict]) -> list[dict]:
    chunks = []

    step = max(
        1,
        CHUNK_SIZE - CHUNK_OVERLAP,
    )

    for page in pages:
        text = " ".join(page["text"].split())

        for start in range(0, len(text), step):
            piece = text[start:start + CHUNK_SIZE].strip()

            if len(piece) >= 30:
                chunks.append({
                    "text": piece,
                    "source": filename,
                    "page": page["page"],
                    "chunk_id": len(chunks),
                })

    return chunks


def index_document(
    session_id: str,
    filename: str,
    content: bytes,
) -> dict:
    # 1. Extract document content.
    pages = extract_pages(filename, content)

    # 2. Create chunks.
    chunks = chunk_pages(filename, pages)

    if not chunks:
        raise ValueError(
            "The document did not contain enough text to index."
        )

    # 3. Generate embeddings through OpenRouter.
    vectors = embed_texts([
        chunk["text"] for chunk in chunks
    ])

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError(
            "Embedding service returned an invalid vector array."
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Embedding service returned invalid numerical values."
        )

    session = get_session(session_id)

    # 4. Combine with existing session documents.
    combined_chunks = session.chunks + chunks

    if session.vectors is not None:
        if session.vectors.shape[1] != vectors.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch. Clear the session "
                "and re-index all documents with the same model."
            )

        combined_vectors = np.vstack([
            session.vectors,
            vectors,
        ])
    else:
        combined_vectors = vectors

    # 5. Normalize vectors for cosine similarity.
    norms = np.linalg.norm(
        combined_vectors,
        axis=1,
        keepdims=True,
    )

    normalized = combined_vectors / np.maximum(
        norms,
        1e-12,
    )

    normalized = normalized.astype(np.float32)

    # 6. Build FAISS index.
    faiss_index = None

    if faiss is not None:
        faiss_index = faiss.IndexFlatIP(
            normalized.shape[1]
        )
        faiss_index.add(normalized)

    # 7. Build BM25 index.
    tokenized = [
        chunk["text"].lower().split()
        for chunk in combined_chunks
    ]

    bm25_index = BM25Okapi(tokenized)

    # 8. Update session after successful indexing.
    session.chunks = combined_chunks
    session.vectors = normalized
    session.faiss_index = faiss_index
    session.bm25 = bm25_index
    session.dimension = int(vectors.shape[1])

    return {
        "filename": filename,
        "pages": len(pages),
        "chunks_added": len(chunks),
        "total_chunks": len(combined_chunks),
        "embedding_dimension": session.dimension,
        "vector_backend": (
            "FAISS" if faiss_index is not None
            else "NumPy fallback"
        ),
        "ocr_enabled": True,
    }
