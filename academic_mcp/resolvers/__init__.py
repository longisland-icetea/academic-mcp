"""Resolver registry: turns a paper into an ordered list of strategies.

Order matters.  Cheap-and-reliable first, expensive-and-fragile last:

    arxiv-direct → elsevier-api → direct-pdf → camoufox-browser
    → arxiv-title-search

``direct-pdf`` covers open-access publishers whose PDF endpoint needs no
session cookies (Nature, Springer, SciPost) — one GET instead of a browser.

The last entry is a *fallback for content*, not for access: if every
publisher path is paywalled, an arXiv preprint of the same paper is still
good enough to read, and vastly better than nothing.  It runs last so the
version-of-record is preferred when reachable.
"""

from __future__ import annotations

import logging

from ..config import settings
from .arxiv import ArxivDirectResolver, ArxivTitleResolver
from .base import STATS, Attempt, Paper, Resolver, run
from .publisher import DirectPdfResolver, ElsevierResolver

logger = logging.getLogger("academic_mcp.resolvers")

_CAMOUFOX: Resolver | None = None


def _camoufox() -> Resolver | None:
    """Import the browser resolver lazily — playwright is a heavy import."""
    global _CAMOUFOX
    if _CAMOUFOX is None and settings.camoufox_enabled:
        try:
            from .camoufox import CamoufoxResolver

            _CAMOUFOX = CamoufoxResolver()
        except ImportError as exc:
            logger.warning("camoufox resolver unavailable: %s", exc)
            _CAMOUFOX = False  # type: ignore[assignment]
    return _CAMOUFOX or None


def plan(paper: Paper) -> list[Resolver]:
    """Ordered resolvers that could plausibly obtain this paper."""
    candidates: list[Resolver] = [
        ArxivDirectResolver(),
        ElsevierResolver(),
        DirectPdfResolver(),
    ]
    browser = _camoufox()
    if browser is not None:
        candidates.append(browser)
    candidates.append(ArxivTitleResolver())
    return [r for r in candidates if r.applies(paper)]


__all__ = [
    "Paper",
    "Resolver",
    "Attempt",
    "STATS",
    "plan",
    "run",
    "DirectPdfResolver",
    "ElsevierResolver",
]
