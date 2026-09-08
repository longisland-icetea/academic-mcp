"""arXiv resolvers: direct download by ID, and preprint discovery by title.

arXiv is the highest-value source for a condensed-matter physicist: it is
free, always reachable, and covers essentially every paper that matters.
Both resolvers therefore run early — the direct one first, the title-search
one last (after paywalled publishers have had their chance at the
version-of-record).
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re

from .. import httpclient
from ..config import settings
from .base import Paper

logger = logging.getLogger("academic_mcp.resolvers.arxiv")

_NEW_ID = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$")
_OLD_ID = re.compile(r"^([a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")
_ARXIV_ANY = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")

# Search engines choke on Unicode punctuation (titles copy-pasted from PDFs
# are full of en-dashes and curly quotes).
_UNICODE_FIX = str.maketrans(
    {
        "–": "-", "—": "-", "―": "-", "−": "-",
        "‘": "'", "’": "'", "“": '"', "”": '"',
    }
)


def extract_arxiv_id(doi: str) -> str | None:
    """Return the bare arXiv ID for an arXiv DOI or raw ID, else None.

    >>> extract_arxiv_id("10.48550/arxiv.2303.09844")
    '2303.09844'
    >>> extract_arxiv_id("2303.09844v2")
    '2303.09844'
    """
    if not doi:
        return None
    prefix = "10.48550/"
    if doi.lower().startswith(prefix):
        rest = doi[len(prefix):]
        return rest[6:] if rest.lower().startswith("arxiv.") else rest
    m = _NEW_ID.match(doi.strip()) or _OLD_ID.match(doi.strip())
    return m.group(1) if m else None


def _candidate_urls(arxiv_id: str) -> list[str]:
    return [
        f"https://arxiv.org/pdf/{arxiv_id}",
        f"https://arxiv.org/pdf/{arxiv_id}v1",
    ]


class ArxivDirectResolver:
    """Download straight from arxiv.org when the DOI *is* an arXiv ID."""

    name = "arxiv-direct"

    def applies(self, paper: Paper) -> bool:
        return extract_arxiv_id(paper.doi) is not None

    async def fetch(self, paper: Paper) -> bytes | None:
        arxiv_id = extract_arxiv_id(paper.doi)
        if not arxiv_id:
            return None
        headers = {
            "User-Agent": httpclient.BROWSER_UA,
            "Accept": "application/pdf",
        }
        for url in _candidate_urls(arxiv_id):
            result = await httpclient.fetch(url, headers=headers, want_pdf=True)
            if result.ok:
                return result.content
            # arXiv is not paywalled, so a timeout is worth one proxy retry.
            if "timed out" in result.error.lower() or "connect" in result.error.lower():
                if settings.gfw_proxy:
                    proxied = await httpclient.fetch_via_proxy(
                        url, settings.gfw_proxy, headers=headers
                    )
                    if proxied.ok:
                        return proxied.content
        return None


class ArxivTitleResolver:
    """Find the arXiv preprint by title, then download it.

    Uses DDGS (multi-engine, ``site:arxiv.org``) and verifies each candidate
    against arxiv.org metadata before accepting: fetching the *wrong* paper
    is far worse than failing, because the reading subagent will write
    confident notes about it.
    """

    name = "arxiv-title-search"

    def applies(self, paper: Paper) -> bool:
        return bool(paper.title) and extract_arxiv_id(paper.doi) is None

    async def fetch(self, paper: Paper) -> bytes | None:
        if not settings.arxiv_search_enabled:
            return None
        arxiv_id = await self._find(paper)
        if not arxiv_id:
            return None
        result = await httpclient.fetch(
            f"https://arxiv.org/pdf/{arxiv_id}", want_pdf=True
        )
        if result.ok:
            logger.info("Title search matched %s → arXiv:%s", paper.title[:60], arxiv_id)
            return result.content
        if settings.gfw_proxy:
            proxied = await httpclient.fetch_via_proxy(
                f"https://arxiv.org/pdf/{arxiv_id}", settings.gfw_proxy
            )
            if proxied.ok:
                return proxied.content
        return None

    # ── internals ──────────────────────────────────────────────────────

    async def _find(self, paper: Paper) -> str | None:
        # The DDGS call is blocking and runs in a thread. Bounded, because an
        # unbounded stall here would hold the pipeline's per-key lock and stop
        # every other fetch behind it.
        try:
            ids = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None, self._search_sync, paper.title
                ),
                timeout=settings.arxiv_search_timeout,
            )
        except TimeoutError:
            logger.info(
                "arXiv title search timed out after %.0fs",
                settings.arxiv_search_timeout,
            )
            return None
        if not ids:
            return None
        client = await httpclient.get_client()
        return await self._verify(ids, paper, client)

    def _search_sync(self, title: str) -> list[str]:
        """Blocking DDGS search — run in a thread."""
        try:
            from ddgs import DDGS
        except ImportError:
            logger.info("DDGS not installed — arXiv title search unavailable")
            return []

        queries = [f'"{title[:200]}"']
        words = title.split()
        if len(words) > 6:
            queries.append(" ".join(words[:6]))

        found: list[str] = []
        for query in queries:
            try:
                with DDGS(
                    proxy=settings.gfw_proxy or settings.download_proxy
                ) as ddgs:
                    for hit in ddgs.text(
                        f"site:arxiv.org {query.translate(_UNICODE_FIX)}",
                        max_results=50,
                        backend="duckduckgo,brave,google",
                    ):
                        m = _ARXIV_ANY.search(hit.get("href", ""))
                        if m and m.group(1) not in found:
                            found.append(m.group(1))
            except Exception as exc:  # noqa: BLE001
                logger.debug("DDGS query failed: %s", exc)
            if found:
                break
        logger.debug("arXiv title search found %d candidate(s)", len(found))
        return found[:10]

    async def _verify(self, ids: list[str], paper: Paper, client) -> str | None:
        query = re.sub(r"[^\w\s-]", " ", paper.title).lower().strip()
        query_words = set(query.split())
        author_last = paper.first_author.strip().split()[-1].lower() if paper.first_author else ""

        for arxiv_id in ids:
            resp = await httpclient.fetch(
                f"https://arxiv.org/abs/{arxiv_id}", attempts=1
            )
            if not resp.ok:
                continue
            html = resp.content.decode("utf-8", errors="replace")

            meta = re.search(
                r'<meta\s+name="citation_title"\s+content="([^"]*)"', html
            )
            if meta:
                found_title = meta.group(1).strip()
            else:
                page_title = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
                if not page_title:
                    continue
                found_title = re.sub(
                    r"^\[\d+\.\d+(?:v\d+)?\]\s*", "", page_title.group(1).strip()
                )

            similarity = difflib.SequenceMatcher(None, query, found_title.lower()).ratio()
            overlap = (
                len(query_words & set(found_title.lower().split())) / len(query_words)
                if query_words
                else 0.0
            )

            author_match = False
            if author_last:
                authors = re.findall(
                    r'<meta\s+name="citation_author"\s+content="([^"]*)"', html
                )
                entry = authors[0].strip().lower() if authors else ""
                entry_last = entry.split(",")[0].strip() if entry else ""
                author_match = author_last == entry_last or author_last in entry

            # With author confirmation the title bar can be looser (arXiv
            # titles often drop subtitles); without it, require both metrics.
            matched = (
                similarity >= 0.55 if author_match else (similarity >= 0.75 and overlap >= 0.6)
            )
            logger.debug(
                "verify %s: sim=%.2f ovl=%.2f author=%s → %s",
                arxiv_id,
                similarity,
                overlap,
                author_match,
                "MATCH" if matched else "no",
            )
            if matched:
                return arxiv_id
        return None
