"""Publisher-direct resolvers: plain HTTP, no browser.

Two kinds:

* :data:`DIRECT_PDF` — publishers whose PDF endpoint is session-free, so a
  single GET returns ``application/pdf``.  Every entry here was verified by
  actually fetching a DOI and checking the content type; do not add entries
  because a URL "looks right".  A wrong guess costs a wasted request and, worse,
  quietly masks the browser path that would have worked.
* :class:`ElsevierResolver` — ScienceDirect Article Retrieval API.

These run before :mod:`.camoufox`, which needs a real browser and is therefore
slower and more fragile.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from .. import httpclient
from ..config import settings
from .base import Paper

logger = logging.getLogger("academic_mcp.resolvers.publisher")

_NATURE_NEW = re.compile(r"10\.1038/(s\d{4,6}-[\d\w-]+)")
_NATURE_ANY = re.compile(r"10\.1038/(.+)")


def nature_id(doi: str) -> str | None:
    """Article slug for a Nature-family DOI.

    >>> nature_id("10.1038/s41467-020-20667-2")
    's41467-020-20667-2'
    >>> nature_id("10.1038/nature04732")
    'nature04732'
    """
    if not doi:
        return None
    m = _NATURE_NEW.search(doi) or _NATURE_ANY.search(doi)
    return m.group(1) if m else None


def is_elsevier(doi: str) -> bool:
    return doi.lower().startswith("10.1016/")


@dataclass(frozen=True)
class DirectSource:
    """A publisher PDF endpoint reachable with a plain GET."""

    label: str
    pdf: Callable[[str], str]
    referer: Callable[[str], str] | None = None
    verified: str = ""


def _nature_pdf(doi: str) -> str:
    slug = nature_id(doi) or ""
    return f"https://www.nature.com/articles/{slug}.pdf"


def _nature_referer(doi: str) -> str:
    # Without a Referer, nature.com answers with an HTML interstitial instead
    # of the PDF.
    slug = nature_id(doi) or ""
    return f"https://www.nature.com/articles/{slug}"


def _springer_pdf(doi: str) -> str:
    return f"https://link.springer.com/content/pdf/{doi}.pdf"


def _scipost_pdf(doi: str) -> str:
    suffix = doi.split("/", 1)[1]
    return f"https://scipost.org/{suffix}/pdf"


DIRECT_PDF: dict[str, DirectSource] = {
    "10.1038/": DirectSource(
        label="nature",
        pdf=_nature_pdf,
        referer=_nature_referer,
        verified="open-access Nature titles return application/pdf",
    ),
    "10.1007/": DirectSource(
        label="springer",
        pdf=_springer_pdf,
        verified="2026-09-05: 10.1007/BF01608499 -> 200 application/pdf 835 KB",
    ),
    # SciPost deliberately NOT here, even though a bare GET does return the
    # PDF.  It sits behind Anubis, an anti-scraper proof-of-work gate: plain
    # non-browser clients pass through untouched, but anything with a browser
    # User-Agent gets a JS proof-of-work challenge.  Routing around that by
    # lying about our User-Agent would be evading an anti-abuse control
    # someone installed on purpose — and Anubis's own text says individual
    # access is meant to be fine, it is bulk scraping they are throttling.
    # So SciPost goes through Camoufox, which runs real JS and answers the
    # challenge legitimately (~23 s instead of ~2 s; correctness first).
}


class DirectPdfResolver:
    """GET the PDF straight from the publisher for open-access titles."""

    name = "direct-pdf"

    def applies(self, paper: Paper) -> bool:
        return bool(self._source(paper.doi))

    @staticmethod
    def _source(doi: str) -> DirectSource | None:
        for prefix, src in DIRECT_PDF.items():
            if doi.startswith(prefix):
                return src
        return None

    async def fetch(self, paper: Paper) -> bytes | None:
        src = self._source(paper.doi)
        if not src:
            return None
        if src.label == "nature" and not nature_id(paper.doi):
            return None

        url = src.pdf(paper.doi)
        headers = {"User-Agent": httpclient.BROWSER_UA, "Accept": "application/pdf,*/*"}
        if src.referer:
            headers["Referer"] = src.referer(paper.doi)

        result = await httpclient.fetch(url, headers=headers, want_pdf=True, attempts=2)
        if result.ok:
            logger.debug("direct %s PDF ok for %s", src.label, paper.doi)
            return result.content
        logger.debug(
            "direct %s PDF failed for %s: %s", src.label, paper.doi, result.error
        )
        return None


class ElsevierResolver:
    """ScienceDirect Article Retrieval API (official, no browser needed)."""

    name = "elsevier-api"

    def applies(self, paper: Paper) -> bool:
        return is_elsevier(paper.doi) and bool(settings.elsevier_api_key)

    async def fetch(self, paper: Paper) -> bytes | None:
        url = f"https://api.elsevier.com/content/article/doi/{paper.doi}"
        headers = {
            "X-ELS-APIKey": settings.elsevier_api_key,
            "Accept": "application/pdf",
        }
        if settings.elsevier_insttoken:
            headers["X-ELS-Insttoken"] = settings.elsevier_insttoken

        result = await httpclient.fetch(url, headers=headers, want_pdf=True, attempts=2)
        if result.ok:
            return result.content

        # 404 usually means "your entitlement does not cover this journal",
        # which is not retryable — say so instead of reporting a bare status.
        if result.status in (401, 403, 404):
            logger.info(
                "Elsevier API: no entitlement for %s (HTTP %s)", paper.doi, result.status
            )
        return None
