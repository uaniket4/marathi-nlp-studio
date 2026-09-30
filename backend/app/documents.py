"""Extract plain text from uploaded documents (.txt / .pdf / .docx).

Kept deliberately small: no OCR, no layout analysis — just enough to pull the
readable text out so the search pipeline can run over it. Parsing errors are
turned into clean messages by the caller; no stack traces reach the client.
"""

from __future__ import annotations

import io
import os
from typing import Tuple


class UnsupportedDocument(Exception):
    """Raised for a file type we cannot extract text from."""


def _extract_txt(raw: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    # Last resort: decode lossily rather than fail outright.
    return raw.decode("utf-8", errors="replace")


def _extract_pdf(raw: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _extract_docx(raw: bytes) -> str:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(raw))
    return "\n".join(p.text for p in document.paragraphs)


def extract_text(filename: str, raw: bytes, max_chars: int) -> Tuple[str, bool]:
    """Return ``(text, truncated)`` extracted from ``raw`` based on the extension.

    Raises :class:`UnsupportedDocument` for unknown types. ``text`` is stripped
    and capped to ``max_chars``; ``truncated`` says whether the cap was applied.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in (".txt", ".text", ".md"):
        text = _extract_txt(raw)
    elif ext == ".pdf":
        text = _extract_pdf(raw)
    elif ext in (".docx",):
        text = _extract_docx(raw)
    else:
        raise UnsupportedDocument(
            f"Unsupported file type '{ext or 'unknown'}'. Upload a .txt, .pdf or .docx file."
        )

    text = (text or "").strip()
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return text, truncated
