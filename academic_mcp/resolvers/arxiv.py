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
import xml.etree.ElementTree as ET

from .. import httpclient
from ..config import settings
from .base import Paper

logger = logging.getLogger("academic_mcp.resolvers.arxiv")

_NEW_ID = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$")
_OLD_ID = re.compile(r"^([a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")
_ARXIV_ANY = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")

# The arXiv API's own title search. Quoted, because an unquoted query is an
# OR over every word and returns a page of unrelated papers for any title.
_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"

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
        # Both search paths block on the network and run in a thread. Bounded,
        # because an unbounded stall here would hold the pipeline's per-key lock
        # and stop every other fetch behind it.
        try:
            ids = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None, self._search_sync, paper
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

    def _search_sync(self, paper: Paper) -> list[str]:
        """Blocking discovery — run in a thread. API first, web search second.

        The order is the whole point. This resolver exists to rescue a paper the
        publisher would not give us, so it has to work when the open web is
        hostile — and it is: measured on 2026-10-04, every backend DDGS was
        configured with was refusing us (Brave 429, Google 403, DuckDuckGo an
        empty 202), so 23 of 27 title searches failed and a pile of paywalled
        APS papers went undownloaded *while their preprints sat on arXiv*. The
        arXiv API answered both of the papers that had just failed, in under a
        second, with no search engine involved.

        DDGS therefore stays only as a rescue for the case the API cannot serve:
        a paper whose arXiv title differs too much from the published one to
        survive the API's phrase match.
        """
        ids = self._api_ids(paper.title)
        if ids:
            return ids
        return self._ddgs_ids(paper.title)

    def _api_ids(self, title: str) -> list[str]:
        """Ask arXiv's own API for the title. No search engine in the path.

        Quoted so the query is a phrase match. A title-style search rather than
        the default "all fields" one, because a quoted `all:` query over a full
        sentence tends to return nothing at all.
        """
        query = re.sub(r'["\\]', " ", title[:200].translate(_UNICODE_FIX)).strip()
        if not query:
            return []
        url = (
            f"{_ARXIV_API}?search_query=ti:%22{query.replace(' ', '+')}%22"
            "&start=0&max_results=10"
        )
        headers = {"User-Agent": httpclient.BROWSER_UA, "Accept": "application/atom+xml"}
        try:
            import httpx

            # `export.arxiv.org` is HTTP-only for the API; the 301 to HTTPS is
            # followed, and on https it answers 200 without a proxy from here.
            resp = httpx.get(url, headers=headers, timeout=20.0, follow_redirects=True)
            if resp.status_code != 200:
                logger.info("arXiv API answered HTTP %d", resp.status_code)
                return []
            body = resp.text
        except Exception as exc:  # noqa: BLE001 — discovery must not raise
            logger.info("arXiv API query failed: %s: %s", type(exc).__name__, exc)
            return []

        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            logger.info("arXiv API returned unparseable XML: %s", exc)
            return []

        ids: list[str] = []
        for entry in root.iter(f"{_ATOM_NS}entry"):
            raw = (entry.findtext(f"{_ATOM_NS}id") or "").strip()
            m = re.search(r"abs/([^/]+)$", raw)
            if not m:
                continue
            arxiv_id = re.sub(r"v\d+$", "", m.group(1))
            if arxiv_id and arxiv_id not in ids:
                ids.append(arxiv_id)
        logger.debug("arXiv API returned %d candidate(s) for %r", len(ids), title[:60])
        return ids[:10]

    def _ddgs_ids(self, title: str) -> list[str]:
        """Blocking DDGS search — run in a thread. The rescue, not the default."""
        try:
            from ddgs import DDGS
        except ImportError:
            logger.info("DDGS not installed — arXiv web-search fallback unavailable")
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
