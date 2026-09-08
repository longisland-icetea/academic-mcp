"""Download → convert → cache orchestration.

This is the single entry point the MCP tools call.  It owns:

* cache lookup (Markdown first, then PDF, then the network),
* resolver ordering (:func:`academic_mcp.resolvers.plan`),
* PDF validity checks (a corrupt cache entry must never be served),
* Markdown conversion (:mod:`academic_mcp.mineru`, the only path).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from . import mineru, storage
from .config import settings
from .resolvers import Attempt, Paper, plan, run

logger = logging.getLogger("academic_mcp.pipeline")

# Per-paper locks, NOT one global lock.
#
# The first version used a single asyncio.Lock around the whole fetch. That
# serialised *unrelated* papers too, and because MinerU conversion (30-60 s)
# happened while holding it, ten concurrent readers queued for ~10 minutes.
# What we actually need is deduplication per paper: the browser resolver has
# its own semaphore, so different papers can safely proceed in parallel.
_key_locks: dict[str, asyncio.Lock] = {}
_key_locks_guard = asyncio.Lock()


async def _lock_for(key: str) -> asyncio.Lock:
    async with _key_locks_guard:
        lock = _key_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _key_locks[key] = lock
        return lock


@dataclass
class FetchOutcome:
    ok: bool
    key: str = ""
    doi: str = ""
    text: str = ""
    pdf_path: str = ""
    md_path: str = ""
    strategy: str = ""
    source: str = ""  # "cache" | "download" | "convert"
    attempts: list[Attempt] = field(default_factory=list)
    error: str = ""

    def succeed(self, text: str, strategy: str, source: str) -> FetchOutcome:
        self.ok = True
        self.text = text
        self.strategy = strategy
        self.source = source
        self.md_path = str(storage.md_path(self.key))
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "doi": self.doi,
            "key": self.key,
            "chars": len(self.text),
            "pdf_path": self.pdf_path,
            "md_path": self.md_path,
            "strategy": self.strategy,
            "source": self.source,
            "attempts": [a.as_dict() for a in self.attempts],
            "error": self.error,
        }


class PipelineError(RuntimeError):
    pass


def _resolve_key(doi: str, title: str) -> str:
    key = storage.doi_to_key(doi)
    if not key and title:
        key = storage.doi_to_key(title[:80]) or "_".join(title.split()[:8])[:80]
    return key


async def _resolve(paper: Paper, outcome: FetchOutcome) -> bytes | None:
    """Run the planned resolvers in order; return PDF bytes or None."""
    for resolver in plan(paper):
        data, attempt = await run(resolver, paper)
        outcome.attempts.append(attempt)
        if data:
            outcome.strategy = resolver.name
            return data
    return None


async def get_text(
    doi: str = "",
    title: str = "",
    first_author: str = "",
    *,
    force: bool = False,
) -> FetchOutcome:
    """Return the full text (Markdown) of a paper, downloading if needed."""
    if not doi and not title:
        return FetchOutcome(ok=False, error="either doi or title is required")

    settings.ensure_dirs()
    doi = storage.normalize_doi(doi)
    key = _resolve_key(doi, title)
    if not key:
        return FetchOutcome(ok=False, error="could not derive a storage key")

    outcome = FetchOutcome(ok=False, key=key, doi=doi)

    # 1. Markdown cache.
    if not force:
        cached = storage.read_md(key)
        if cached:
            outcome.succeed(cached, "cache", "cache")
            return outcome

    async with await _lock_for(key):
        # Re-check inside the lock: another caller may have finished while we waited.
        if not force:
            cached = storage.read_md(key)
            if cached:
                outcome.succeed(cached, "cache", "cache")
                return outcome

        # 2. PDF cache → convert.
        pdf = storage.pdf_path(key)
        if not force and storage.is_valid_pdf(pdf):
            outcome.pdf_path = str(pdf)
            try:
                text = await asyncio.to_thread(mineru.convert_paper_pdf, pdf, key)
                storage.write_md(key, text)
                outcome.succeed(text, f"cache-pdf+{settings.mineru_model}", "convert")
                return outcome
            except mineru.MineruError as exc:
                outcome.error = str(exc)
                return outcome
        elif pdf.exists():
            storage.discard_invalid_pdf(pdf)

        # 3. Network.
        paper = Paper(doi=doi, title=title, first_author=first_author, key=key)
        data: bytes | None = None
        try:
            data = await asyncio.wait_for(
                _resolve(paper, outcome), timeout=settings.download_budget
            )
        except TimeoutError:
            outcome.error = f"download exceeded {settings.download_budget:.0f}s budget"
            return outcome

        if not data:
            outcome.error = "all download strategies failed"
            return outcome

        if not storage.looks_like_pdf(data):
            outcome.error = f"downloaded {len(data)} bytes but it is not a PDF"
            return outcome

        outcome.pdf_path = str(storage.write_pdf(key, data))
        try:
            text = await asyncio.to_thread(mineru.convert_paper_pdf, outcome.pdf_path, key)
        except mineru.MineruError as exc:
            # The PDF is cached, so a later call only needs the (cheap)
            # conversion step — this is a recoverable failure.
            outcome.error = f"downloaded PDF but conversion failed: {exc}"
            return outcome

        storage.write_md(key, text)
        outcome.succeed(text, f"{outcome.strategy}+{settings.mineru_model}", "download")
        return outcome


async def get_pdf(doi: str = "", title: str = "", first_author: str = "") -> FetchOutcome:
    """Return the path to a downloaded PDF (no conversion)."""
    if not doi and not title:
        return FetchOutcome(ok=False, error="either doi or title is required")
    settings.ensure_dirs()
    doi = storage.normalize_doi(doi)
    key = _resolve_key(doi, title)
    outcome = FetchOutcome(ok=False, key=key, doi=doi)

    # Same per-key lock as get_text: two callers after the same paper would
    # otherwise both download it and both write the same file.
    async with await _lock_for(key):
        pdf = storage.pdf_path(key)
        if storage.is_valid_pdf(pdf):
            outcome.ok = True
            outcome.pdf_path = str(pdf)
            outcome.strategy = "cache"
            outcome.source = "cache"
            return outcome
        if pdf.exists():
            storage.discard_invalid_pdf(pdf)

        paper = Paper(doi=doi, title=title, first_author=first_author, key=key)
        try:
            data = await asyncio.wait_for(
                _resolve(paper, outcome), timeout=settings.download_budget
            )
        except TimeoutError:
            outcome.error = f"download exceeded {settings.download_budget:.0f}s budget"
            return outcome

        if data and storage.looks_like_pdf(data):
            outcome.ok = True
            outcome.pdf_path = str(storage.write_pdf(key, data))
            outcome.strategy = outcome.strategy or "download"
            outcome.source = "download"
            return outcome
        outcome.error = "all download strategies failed"
        return outcome
