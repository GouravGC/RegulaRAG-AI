
from pathlib import Path
import io
import logging

import fitz
import numpy as np
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

MIN_TEXT_LENGTH = 40
OCR_MAX_DIMENSION = 2200
OCR_RENDER_SCALE = 1.5

_ocr_engine = None


def get_ocr_engine():
    """Initialize RapidOCR only when scanned-page OCR is required."""
    global _ocr_engine

    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _ocr_engine = RapidOCR()

        except ImportError as exc:
            raise RuntimeError(
                "RapidOCR is not installed. "
                "Install rapidocr-onnxruntime from requirements.txt."
            ) from exc

    return _ocr_engine


def ocr_pdf_page(page) -> str:
    """Extract text from a scanned PDF page using RapidOCR."""

    pix = page.get_pixmap(
        matrix=fitz.Matrix(
            OCR_RENDER_SCALE,
            OCR_RENDER_SCALE,
        ),
        colorspace=fitz.csRGB,
        alpha=False,
    )

    image = Image.open(
        io.BytesIO(pix.tobytes("png"))
    ).convert("RGB")

    try:
        image.thumbnail(
            (OCR_MAX_DIMENSION, OCR_MAX_DIMENSION),
            Image.Resampling.LANCZOS,
        )

        image_array = np.asarray(image)

        result, _ = get_ocr_engine()(image_array)

        if not result:
            return ""

        extracted_lines = []

        for item in result:
            if item and len(item) >= 2:
                text = str(item[1]).strip()

                if text:
                    extracted_lines.append(text)

        return "\n".join(extracted_lines).strip()

    finally:
        image.close()


def extract_pages(filename: str, content: bytes) -> list[dict]:
    """Extract selectable text and OCR scanned PDF pages."""
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
                                "RapidOCR skipped for PDF page %s: %s",
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
            "The document may be blank or unreadable."
        )

    if skipped_ocr_pages:
        logger.warning(
            "Document %s has OCR-skipped pages: %s",
            filename,
            skipped_ocr_pages,
        )

    return pages


def chunk_pages(filename: str, pages: list[dict]) -> list[dict]:
    """Split extracted pages into overlapping text chunks."""
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
    # 1. Extract PDF or text content.
    pages = extract_pages(filename, content)

    # 2. Create overlapping chunks.
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



from app.services.web.loader import crawl_website, WebsiteLoaderError


def index_website(
    session_id: str,
    url: str,
    max_pages: int = 40,
) -> dict:
    """Crawl a website and add its pages to the existing RAG session."""

    # 1. Extract website pages directly from HTML.
    pages = crawl_website(
        start_url=url,
        max_pages=max_pages,
    )

    # 2. Convert website pages into chunks, preserving source metadata.
    chunks = []

    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    for page in pages:
        text = " ".join(page["text"].split())

        for start in range(0, len(text), step):
            piece = text[start:start + CHUNK_SIZE].strip()

            if len(piece) >= 30:
                chunks.append({
                    "text": piece,
                    "source": page["title"],
                    "page": None,
                    "source_url": page["source_url"],
                    "source_type": "website",
                    "title": page["title"],
                    "chunk_id": len(chunks),
                })

    if not chunks:
        raise ValueError("Website did not contain enough text to index.")

    # 3. Generate embeddings using the existing OpenRouter integration.
    vectors = embed_texts([
        chunk["text"] for chunk in chunks
    ])

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Embedding service returned an invalid vector array.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding service returned invalid numerical values.")

    session = get_session(session_id)

    # 4. Combine with existing PDF, TXT and website chunks.
    combined_chunks = session.chunks + chunks

    if session.vectors is not None:
        if session.vectors.shape[1] != vectors.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch. Clear the session "
                "and re-index all documents using the same model."
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

    normalized = combined_vectors / np.maximum(norms, 1e-12)
    normalized = normalized.astype(np.float32)

    # 6. Rebuild FAISS index.
    faiss_index = None

    if faiss is not None:
        faiss_index = faiss.IndexFlatIP(normalized.shape[1])
        faiss_index.add(normalized)

    # 7. Rebuild BM25 index.
    tokenized = [
        chunk["text"].lower().split()
        for chunk in combined_chunks
    ]

    bm25_index = BM25Okapi(tokenized)

    # 8. Update the existing session.
    session.chunks = combined_chunks
    session.vectors = normalized
    session.faiss_index = faiss_index
    session.bm25 = bm25_index
    session.dimension = int(vectors.shape[1])

    return {
        "source_url": url,
        "pages_crawled": len(pages),
        "chunks_added": len(chunks),
        "total_chunks": len(combined_chunks),
        "embedding_dimension": session.dimension,
        "vector_backend": (
            "FAISS" if faiss_index is not None
            else "NumPy fallback"
        ),
        "source_type": "website",
    }
