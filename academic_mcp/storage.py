"""Filesystem storage for downloaded PDFs and converted Markdown.

Everything is keyed by a DOI-derived safe filename so that cache lookups are
pure string operations and independent of which resolver produced the file.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import settings

logger = logging.getLogger("academic_mcp.storage")

MIN_PDF_BYTES = 5000
_PDF_MAGIC = b"%PDF"

_UNSAFE = re.compile(r'[<>:"|?*\\/ ]')
_TRAILING_PUNCT = re.compile(r"[.,;)\]\}]+$")


def normalize_doi(doi: str) -> str:
    """Strip a ``https://doi.org/`` prefix and trailing punctuation.

    >>> normalize_doi("HTTPS://DOI.ORG/10.1038/foo")
    '10.1038/foo'
    >>> normalize_doi("10.1038/foo.")
    '10.1038/foo'
    """
    if not doi:
        return ""
    s = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    s = _TRAILING_PUNCT.sub("", s)
    if "/" in s:
        head, _, tail = s.partition("/")
        s = f"{head.lower()}/{tail}"
    else:
        s = s.lower()
    return s


def doi_to_key(doi: str) -> str:
    """Filesystem-safe key for a DOI (``10.1038/x`` → ``10.1038_x``)."""
    clean = normalize_doi(doi)
    if not clean:
        return ""
    return _UNSAFE.sub("_", clean)


def pdf_path(key: str) -> Path:
    return settings.pdf_dir / f"{key}.pdf"


def md_path(key: str) -> Path:
    return settings.md_dir / f"{key}.md"


def is_valid_pdf(path: Path) -> bool:
    """Magic bytes + size check.

    Guards against a poisoned cache: any process can drop a garbage ``.pdf``
    into the directory, and without this check every later download would
    silently return the corrupt file forever.
    """
    try:
        if not path.is_file() or path.stat().st_size < MIN_PDF_BYTES:
            return False
        with path.open("rb") as fh:
            return fh.read(4) == _PDF_MAGIC
    except OSError:
        return False


def looks_like_pdf(data: bytes) -> bool:
    return bool(data) and data[:4] == _PDF_MAGIC and len(data) > MIN_PDF_BYTES


def discard_invalid_pdf(path: Path) -> None:
    """Delete a corrupt cache entry so the next call re-downloads."""
    try:
        path.unlink()
        logger.warning("Discarded corrupt cached PDF: %s", path.name)
    except OSError:
        pass


def write_pdf(key: str, data: bytes) -> Path:
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_path(key)
    path.write_bytes(data)
    return path


def write_md(key: str, text: str) -> Path:
    settings.md_dir.mkdir(parents=True, exist_ok=True)
    path = md_path(key)
    path.write_text(text, encoding="utf-8")
    return path


def read_md(key: str) -> str | None:
    path = md_path(key)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None
