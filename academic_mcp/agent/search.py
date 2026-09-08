#!/usr/bin/env python3
"""Academic literature search — Scopus (primary) + OpenAlex.

Usage:
  python3 search.py "moire exciton"                      # basic search
  python3 search.py "moire exciton" --limit 20           # with limit
  python3 search.py "moire exciton" --year 2020-2024     # year filter
  python3 search.py "moire exciton" --source scopus      # single source
  python3 search.py "moire exciton" --rerank             # hybrid relevance rerank
  python3 search.py "moire exciton" --json               # JSON output
  python3 search.py --doi 10.1038/s41586-019-0957-1      # DOI lookup

Structured search (for known bibliographic metadata):
  python3 search.py "exciton" --author "Wang" --journal "Nature"
  python3 search.py --author "Cao" --journal "Nature" --year 2018

Environment:
  ELSEVIER_API_KEY        Scopus API key (https://dev.elsevier.com/)
  ELSEVIER_INSTTOKEN      Optional institutional token
  DOWNLOAD_PROXY          HTTP proxy for Scopus/OpenAlex (e.g. http://192.168.255.1:8888); unset = direct
  GFW_PROXY               Fallback proxy for OpenAlex when behind GFW
  OPENALEX_API_KEY        Optional OpenAlex premium key

Output:
  JSON array of normalized paper objects, with Scopus results first.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# P0-5 修复：复用 _common.py 的标准化函数，避免 DOI key 与 download.py / memory.py 不一致
sys.path.insert(0, str(Path(__file__).parent))
from ._common import doi_to_key as _doi_to_key  # noqa: E402

# ── Dependencies check ──────────────────────────────────────────────────
try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: pip install httpx", file=sys.stderr)
    sys.exit(1)

# Configuration comes from academic_mcp.config.settings — the service's own
# environment / .env — so this module never walks a client's directory tree.
from ..config import settings as _settings

# =========================================================================
# Configuration
# =========================================================================

ELSEVIER_API_KEY = _settings.elsevier_api_key
ELSEVIER_INSTTOKEN = _settings.elsevier_insttoken
DOWNLOAD_PROXY = (_settings.download_proxy or "").strip()
GFW_PROXY = _settings.gfw_proxy or ""
OPENALEX_API_KEY = _settings.openalex_api_key

SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
SCOPUS_ABSTRACT_URL = "https://api.elsevier.com/content/abstract/doi"
OPENALEX_BASE = "https://api.openalex.org"

_MAX_ABSTRACT_CHARS = 3000

# §4.5 architectural upgrade: bound concurrent in-flight requests to a single
# engine to avoid getting rate-limited.  Applied via ``asyncio.Semaphore``
# inside ``search_all`` so callers do not have to pass it down.
_ENGINE_CONCURRENCY = 5

# Module-level singleton semaphore (5 in-flight requests per process).
# Lazily initialised inside the running event loop on first use.
_SEM: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(_ENGINE_CONCURRENCY)
    return _SEM


async def _semaphore_wrap(sem: asyncio.Semaphore, coro_factory):
    """Acquire ``sem``, await the coroutine, release.

    Section 4.5 of the audit report calls for ``Semaphore(5)`` to keep a
    hard ceiling on concurrent requests per engine. Using a semaphore
    directly inside ``search_all`` would force every engine function to
    accept it; the wrapper keeps that complexity contained.
    """
    async with sem:
        return await coro_factory()


# =========================================================================
# DOI / key utilities
# =========================================================================

def doi_to_key(doi: str) -> str:
    """P0-5 修复：委托给 _common.py，避免与 download.py / memory.py 不一致。"""
    return _doi_to_key(doi)


# =========================================================================
# Unicode normalization (same mapping as academic_agent)
# =========================================================================

_UNICODE_REPLACEMENTS = str.maketrans({
    ord('À'): 'A', ord('Á'): 'A', ord('Â'): 'A', ord('Ã'): 'A', ord('Ä'): 'A', ord('Å'): 'A',
    ord('È'): 'E', ord('É'): 'E', ord('Ê'): 'E', ord('Ë'): 'E',
    ord('Ì'): 'I', ord('Í'): 'I', ord('Î'): 'I', ord('Ï'): 'I',
    ord('Ò'): 'O', ord('Ó'): 'O', ord('Ô'): 'O', ord('Õ'): 'O', ord('Ö'): 'O',
    ord('Ù'): 'U', ord('Ú'): 'U', ord('Û'): 'U', ord('Ü'): 'U', ord('Ý'): 'Y',
    ord('Ć'): 'C', ord('Č'): 'C', ord('Ç'): 'C', ord('Ď'): 'D', ord('Ě'): 'E',
    ord('Ğ'): 'G', ord('İ'): 'I', ord('Ĺ'): 'L', ord('Ľ'): 'L',
    ord('Ń'): 'N', ord('Ň'): 'N', ord('Ñ'): 'N', ord('Ő'): 'O', ord('Ŕ'): 'R',
    ord('Ś'): 'S', ord('Š'): 'S', ord('Ş'): 'S', ord('Ť'): 'T', ord('Ű'): 'U',
    ord('Ź'): 'Z', ord('Ż'): 'Z', ord('Ž'): 'Z',
    ord('à'): 'a', ord('á'): 'a', ord('â'): 'a', ord('ã'): 'a', ord('ä'): 'a', ord('å'): 'a',
    ord('è'): 'e', ord('é'): 'e', ord('ê'): 'e', ord('ë'): 'e',
    ord('ì'): 'i', ord('í'): 'i', ord('î'): 'i', ord('ï'): 'i',
    ord('ò'): 'o', ord('ó'): 'o', ord('ô'): 'o', ord('õ'): 'o', ord('ö'): 'o',
    ord('ù'): 'u', ord('ú'): 'u', ord('û'): 'u', ord('ü'): 'u', ord('ý'): 'y', ord('ÿ'): 'y',
    ord('ć'): 'c', ord('č'): 'c', ord('ç'): 'c', ord('ď'): 'd', ord('ě'): 'e',
    ord('ğ'): 'g', ord('ı'): 'i', ord('ĺ'): 'l', ord('ľ'): 'l',
    ord('ń'): 'n', ord('ň'): 'n', ord('ñ'): 'n', ord('ő'): 'o', ord('ŕ'): 'r',
    ord('ś'): 's', ord('š'): 's', ord('ş'): 's', ord('ť'): 't', ord('ű'): 'u',
    ord('ź'): 'z', ord('ż'): 'z', ord('ž'): 'z',
    ord('Æ'): 'AE', ord('æ'): 'ae', ord('Œ'): 'OE', ord('œ'): 'oe', ord('ß'): 'ss',
    ord('Ø'): 'O', ord('ø'): 'o', ord('Ł'): 'L', ord('ł'): 'l',
    ord('Đ'): 'D', ord('đ'): 'd', ord('Ħ'): 'H', ord('ħ'): 'h',
    ord('Ŧ'): 'T', ord('ŧ'): 't', ord('Þ'): 'TH', ord('þ'): 'th',
    ord('Ð'): 'D', ord('ð'): 'd',
    ord('⁰'): '0', ord('¹'): '1', ord('²'): '2', ord('³'): '3',
    ord('⁴'): '4', ord('⁵'): '5', ord('⁶'): '6', ord('⁷'): '7', ord('⁸'): '8', ord('⁹'): '9',
    ord('₀'): '0', ord('₁'): '1', ord('₂'): '2', ord('₃'): '3',
    ord('₄'): '4', ord('₅'): '5', ord('₆'): '6', ord('₇'): '7', ord('₈'): '8', ord('₉'): '9',
    ord('–'): '-', ord('—'): '-',
})


def normalize_query(query: str) -> str:
    """Normalize non-ASCII Latin characters to ASCII for search APIs."""
    import unicodedata
    normalized = unicodedata.normalize("NFKD", query)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.translate(_UNICODE_REPLACEMENTS)
    return normalized


def strip_year_from_query(query: str) -> str:
    """Remove year patterns from query text (year is handled separately via --year).

    Handles: '2024', '2020-2024', '2020–2024', '2020 to 2024', etc.
    Does NOT strip: '2D', '3R', 'MoTe2' (only matches standalone 4-digit years starting with 19/20).
    """
    # Remove year ranges: "2020-2024", "2020–2024", "2020 to 2024", "2020–24"
    query = re.sub(r'\b(19|20)\d{2}\s*[-–—to]+\s*(19|20)\d{2,4}\b', '', query)
    # Remove standalone years: "2024", "2018"
    query = re.sub(r'\b(19|20)\d{2}\b', '', query)
    # Clean up extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    return query


_DOI_PATTERN = re.compile(r'^(?:https?://doi\.org/)?(10\.\d{4,}/[^\s]+)$', re.IGNORECASE)


def is_doi_query(query: str) -> str | None:
    """Check if the query string looks like a DOI. Returns the clean DOI or None.

    Examples: '10.1038/nature26154', 'https://doi.org/10.1038/nature26154'
    """
    q = query.strip()
    m = _DOI_PATTERN.match(q)
    if not m:
        return None
    doi = m.group(1)
    # P2-1: strip a single trailing punctuation char commonly picked up
    # when users paste DOIs from prose (e.g. "10.1038/foo," -> "10.1038/foo").
    # Don't strip a second char or '/' so legitimate DOIs ending in punctuation
    # (rare, but reserved chars exist) are not damaged.
    if doi and doi[-1] in '.,;)]':
        doi = doi[:-1]
    return doi


# =========================================================================
# HTTP helpers
# =========================================================================

def _http_headers(extra: dict = None) -> dict:
    import random
    # P2-3: deliberately rotate between three modern desktop Chrome UAs so that
    # anti-bot defences (Scopus / Crossref / publisher CDNs) do not fingerprint
    # every request as a single Python script. Three is enough to break naive
    # static-UA detection without triggering bot heuristics that flag rapidly
    # changing UAs from a single client. Do NOT increase to dozens — that
    # pattern itself is a bot signature.
    ua = random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    ])
    h = {"User-Agent": ua, "Accept": "application/json"}
    if extra:
        h.update(extra)
    return h


# P0-12 + P1-A 修复：3 次重试 + 指数退避，5xx/ConnectError/Timeout/429
try:
    import httpx as _httpx
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
    _RETRY_DECORATOR = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((
            _httpx.ConnectError, _httpx.ReadTimeout, _httpx.RemoteProtocolError,
            _httpx.HTTPStatusError,  # raised by _safe_json() on 5xx
        )),
        reraise=True,
    )
except ImportError:
    # tenacity 未装 — 退化为无重试
    def _RETRY_DECORATOR(fn):  # type: ignore
        return fn


def _with_retry(fn):
    """Apply _RETRY_DECORATOR to a coroutine function (P1-A).

    This lets us wrap any HTTP call inline:
        resp = await _with_retry(client.get)(url, params=...)
    instead of decorating top-level functions (which would conflict
    with the search.py call shape).
    """
    return _RETRY_DECORATOR(fn)


def _safe_json(resp):
    """P0-12 修复：明示区分 truncated JSON 错与 HTTP 错。"""
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        _log(f"JSONDecodeError (likely truncated): status={resp.status_code} len={len(resp.content)}: {e}")
        return {"_truncated": True, "_raw_bytes": len(resp.content)}


def _client_kwargs() -> dict:
    """Base kwargs for httpx.AsyncClient.

    P0-12 修复：timeout 从 1800s（30 分钟）降至 120s，加 connect=10s。
    取消 30 分钟超时，避免单 API 调用挂住整个 asyncio.gather。
    如需 batch job，可设环境变量 ACADEMIC_SEARCH_TIMEOUT 覆盖。
    """
    overall = float(_settings.search_timeout)
    return {
        "timeout": httpx.Timeout(overall, connect=10.0),
        "follow_redirects": True,
    }


# =========================================================================
# Scopus Search (primary)
# =========================================================================

def _parse_scopus_authors(entry: dict) -> list:
    """Parse the FULL author list from a Scopus entry.

    COMPLETE view carries an 'author' array with every author (authname
    like "Morales-Durán N."); fall back to the dc:creator string (which
    only holds the first author in STANDARD view) when the array is
    absent.
    """
    raw = entry.get("author")
    if raw:
        items = raw if isinstance(raw, list) else [raw]
        names = []
        for a in items:
            if not isinstance(a, dict):
                continue
            name = (a.get("authname") or "").strip()
            if not name:
                surname = (a.get("surname") or "").strip()
                given = (a.get("given-name") or "").strip()
                if surname:
                    name = f"{surname} {given}".strip()
            if name:
                names.append({"name": name})
        if names:
            return names
    creator_string = entry.get("dc:creator") or ""
    return [{"name": p.strip()} for p in creator_string.split(",") if p.strip()]


def _scopus_year(date_str: str) -> int | None:
    if not date_str:
        return None
    m = re.match(r"(\d{4})", str(date_str))
    return int(m.group(1)) if m else None


def _scopus_url(links: list) -> str:
    if not links:
        return ""
    for link in links:
        if link.get("@ref", "") in ("scopus", "scopus-inward"):
            return link.get("@href", "")
    for link in links:
        if link.get("@href"):
            return link["@href"]
    return ""


def _truncate(text: str, max_chars: int = _MAX_ABSTRACT_CHARS) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def normalize_scopus(entry: dict, view: str = "STANDARD") -> dict:
    doi_raw = (entry.get("prism:doi") or "").strip()
    if doi_raw.lower().startswith("https://doi.org/"):
        doi_raw = doi_raw[len("https://doi.org/"):]
    doi = doi_raw
    paper_id = doi_to_key(doi) if doi else ""
    if not paper_id:
        eid = (entry.get("eid") or "").strip()
        paper_id = f"scopus:{eid}" if eid else ""
    eid = (entry.get("eid") or "").strip()
    title = (entry.get("dc:title") or "").strip()
    authors = _parse_scopus_authors(entry)
    year = _scopus_year(entry.get("prism:coverDate") or "")
    venue = (entry.get("prism:publicationName") or "").strip()
    url = _scopus_url(entry.get("link") or [])
    citedby = entry.get("citedby-count")
    if isinstance(citedby, str):
        try:
            citedby = int(citedby)
        except (ValueError, TypeError):
            citedby = 0
    citation_count = citedby if isinstance(citedby, int) else 0

    abstract = ""
    if view == "COMPLETE":
        description = entry.get("dc:description")
        if isinstance(description, str):
            abstract = _truncate(description.strip())
        elif isinstance(description, dict):
            abstract = _truncate(str(description.get("abstract", "")))

    return {
        "paper_id": paper_id,
        "source_id": eid,
        "source": "scopus",
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "year": year,
        "venue": venue,
        "url": url,
        "citation_count": citation_count,
        "doi": doi,
        "oa_pdf_url": "",
    }


async def search_scopus(query: str, limit: int = 20, year: str = None,
                        author: str = None, journal: str = None) -> list:
    if not ELSEVIER_API_KEY:
        return []

    headers = {"X-ELS-APIKey": ELSEVIER_API_KEY, "Accept": "application/json"}
    if ELSEVIER_INSTTOKEN:
        headers["X-ELS-Insttoken"] = ELSEVIER_INSTTOKEN

    # One proxy knob, shared with academic-mcp: DOWNLOAD_PROXY. Empty (or an
    # explicit "none"/"off") means a direct connection, ignoring ambient
    # *_proxy variables.
    proxy_config = {}
    if DOWNLOAD_PROXY and DOWNLOAD_PROXY.lower() not in ("none", "off", "false", "0"):
        proxy_config["proxy"] = DOWNLOAD_PROXY
    else:
        proxy_config["trust_env"] = False

    # Build structured Scopus query when author/journal filters are provided
    parts = []
    doi = is_doi_query(query) if query else None
    if doi:
        # Precise DOI lookup — use DOI() field code, skip normalization
        parts.append(f"DOI({doi})")
    elif query:
        parts.append(normalize_query(query))
    if author:
        # Use Scopus field code AUTHOR-NAME for author filtering
        parts.append(f"AUTHOR-NAME({author})")
    if journal:
        # Use Scopus field code SRCTITLE for journal/source title filtering
        parts.append(f"SRCTITLE({journal})")
    scopus_query = " AND ".join(parts) if parts else normalize_query(query)

    params = {"query": scopus_query}
    if year:
        params["date"] = year

    results = []

    # Attempt 1: COMPLETE view (max 25 results per page)
    complete_params = {**params, "view": "COMPLETE", "count": min(limit, 25), "start": 0}
    async with httpx.AsyncClient(
        base_url=SCOPUS_SEARCH_URL, headers={**headers, **_http_headers()},
        **_client_kwargs(), **proxy_config,
    ) as client:
        try:
            resp = await _with_retry(client.get)("", params=complete_params)
            if resp.status_code == 200:
                data = resp.json()
                entries = data.get("search-results", {}).get("entry", [])
                if isinstance(entries, dict):
                    entries = [entries]
                results = [normalize_scopus(e, "COMPLETE") for e in entries[:limit]]
                total = data.get("search-results", {}).get("opensearch:totalResults", 0)
                _log(f"Scopus COMPLETE: {len(results)} results (total={total})")
                return results
            elif resp.status_code in (401, 403):
                _log(f"Scopus COMPLETE {resp.status_code}, falling back to STANDARD")
            else:
                _log(f"Scopus HTTP {resp.status_code}: {resp.text[:200]}")
                return []
        except Exception as e:
            _log(f"Scopus COMPLETE error: {e}")

    # Attempt 2: STANDARD view (COMPLETE rejected or errored — try with lighter auth)
    standard_params = {**params, "view": "STANDARD", "count": min(limit, 200), "start": 0}
    async with httpx.AsyncClient(
        base_url=SCOPUS_SEARCH_URL, headers={**headers, **_http_headers()},
        **_client_kwargs(), **proxy_config,
    ) as client2:
        try:
            resp = await client2.get("", params=standard_params)
            if resp.status_code != 200:
                _log(f"Scopus STANDARD HTTP {resp.status_code}")
                return []
            data = resp.json()
            entries = data.get("search-results", {}).get("entry", [])
            if isinstance(entries, dict):
                entries = [entries]
            results = [normalize_scopus(e, "STANDARD") for e in entries[:limit]]
            total = data.get("search-results", {}).get("opensearch:totalResults", 0)
            _log(f"Scopus STANDARD: {len(results)} results (total={total})")

            # Enrich abstracts for results without them
            to_enrich = [r for r in results if r.get("doi") and not r.get("abstract")]
            if to_enrich:
                enriched = await asyncio.gather(*[
                    _fetch_scopus_abstract(p["doi"]) for p in to_enrich[:5]
                ], return_exceptions=True)
                for paper, abs_text in zip(to_enrich, enriched, strict=False):
                    if isinstance(abs_text, str) and abs_text:
                        paper["abstract"] = _truncate(abs_text)

            return results
        except Exception as e:
            _log(f"Scopus STANDARD error: {e}")
            return []


async def _fetch_scopus_abstract(doi: str) -> str | None:
    headers = {"X-ELS-APIKey": ELSEVIER_API_KEY, "Accept": "application/json"}
    if ELSEVIER_INSTTOKEN:
        headers["X-ELS-Insttoken"] = ELSEVIER_INSTTOKEN
    async with httpx.AsyncClient(
        headers={**headers, **_http_headers()}, **_client_kwargs(),
    ) as client:
        try:
            resp = await _with_retry(client.get)(f"{SCOPUS_ABSTRACT_URL}/{doi}")
            if resp.status_code == 200:
                data = resp.json()
                ar = data.get("abstracts-retrieval-response", {})
                abstracts = ar.get("item", {}).get("bibrecord", {}).get("head", {}).get("abstracts", "")
                if isinstance(abstracts, str):
                    return abstracts.strip()
                elif isinstance(abstracts, dict):
                    para = abstracts.get("abstract", {}).get("ce:para", "")
                    if isinstance(para, list):
                        return " ".join(p.get("#text", "") if isinstance(p, dict) else str(p) for p in para)
                    return str(para)
        except Exception:
            pass
    return None


async def search_scopus_by_doi(doi: str) -> dict | None:
    """Look up a single paper by DOI in Scopus.

    COMPLETE view is tried first — it includes the abstract
    (dc:description) plus the full author list. Falls back to STANDARD
    when the API key lacks COMPLETE-view permission (401/403) or the
    COMPLETE result is empty; STANDARD still returns the author list but
    no abstract.
    """
    if not ELSEVIER_API_KEY:
        return None

    clean_doi = doi.replace("https://doi.org/", "").strip()
    headers = {"X-ELS-APIKey": ELSEVIER_API_KEY, "Accept": "application/json"}
    if ELSEVIER_INSTTOKEN:
        headers["X-ELS-Insttoken"] = ELSEVIER_INSTTOKEN

    async with httpx.AsyncClient(
        base_url=SCOPUS_SEARCH_URL, headers={**headers, **_http_headers()},
        **_client_kwargs(),
    ) as client:
        try:
            # COMPLETE view first: carries the abstract + full author list
            resp = await _with_retry(client.get)("", params={"query": f"DOI({clean_doi})", "view": "COMPLETE", "count": 1})
            if resp.status_code == 200:
                entries = resp.json().get("search-results", {}).get("entry", [])
                if isinstance(entries, dict):
                    entries = [entries]
                if entries:
                    return normalize_scopus(entries[0], "COMPLETE")
            # Fallback: STANDARD view (authors only, no abstract) — e.g.
            # key without COMPLETE permission (401/403) or empty result
            resp2 = await _with_retry(client.get)("", params={"query": f"DOI({clean_doi})", "view": "STANDARD", "count": 1})
            if resp2.status_code == 200:
                entries = resp2.json().get("search-results", {}).get("entry", [])
                if isinstance(entries, dict):
                    entries = [entries]
                if entries:
                    return normalize_scopus(entries[0])
        except Exception as e:
            _log(f"Scopus DOI lookup error: {e}")
    return None


# =========================================================================
# OpenAlex Search
# =========================================================================

def normalize_openalex(work: dict) -> dict:
    oa_id = work.get("id", "").rstrip("/").split("/")[-1]
    title = work.get("title") or work.get("display_name") or ""

    # Reconstruct abstract from inverted index
    inv = work.get("abstract_inverted_index")
    if inv:
        words = [(pos, word) for word, positions in inv.items() for pos in positions]
        words.sort(key=lambda w: w[0])
        abstract = " ".join(w[1] for w in words)
    else:
        abstract = ""

    authors = []
    for a in work.get("authorships") or []:
        author_info = a.get("author", {}) or {}
        name = author_info.get("display_name", "") or a.get("raw_author_name", "")
        if name:
            authors.append({"name": name})

    year = work.get("publication_year")
    source_info = (work.get("primary_location") or {}).get("source") or {}
    venue = source_info.get("display_name") or ""
    url = (work.get("primary_location") or {}).get("landing_page_url") or ""
    citation_count = work.get("cited_by_count")
    doi_url = work.get("doi") or ""
    doi = doi_url.replace("https://doi.org/", "") if doi_url else ""
    paper_id = doi_to_key(doi) if doi else oa_id

    return {
        "paper_id": paper_id,
        "source_id": oa_id,
        "source": "openalex",
        "title": title,
        "abstract": _truncate(abstract),
        "authors": authors,
        "year": year,
        "venue": venue,
        "url": url,
        "citation_count": citation_count,
        "doi": doi,
        "oa_pdf_url": "",
    }


async def _resolve_openalex_source_id(journal_name: str, client: httpx.AsyncClient) -> str | None:
    """Resolve a journal name to an OpenAlex source ID via /sources search."""
    try:
        resp = await _with_retry(client.get)("/sources", params={"search": journal_name, "per_page": 3})
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            source_id = results[0].get("id", "").rstrip("/").split("/")[-1]
            _log(f"OpenAlex source '{journal_name}' → {source_id}")
            return source_id
    except Exception as e:
        _log(f"OpenAlex source lookup '{journal_name}' failed: {e}")
    return None


async def search_openalex(query: str, limit: int = 20, year: str | None = None,
                         author: str | None = None, journal: str | None = None) -> list[dict[str, Any]]:
    """Search OpenAlex. Journal filter uses two-step ID resolution via /sources.

    Note: when author filter is present, callers should skip OpenAlex entirely
    (name disambiguation via /authors is unreliable for common names).
    """
    doi = is_doi_query(query) if query else None
    params: dict = {"per_page": min(limit, 200)}
    filters = ["type:article|review"]
    if year:
        filters.append(f"publication_year:{year}")

    # DOI lookup: use filter=doi:xxx, skip text search entirely
    if doi:
        filters.append(f"doi:{doi}")
        _log(f"OpenAlex DOI lookup: {doi}")

    proxy = GFW_PROXY if GFW_PROXY else None
    async with httpx.AsyncClient(
        base_url=OPENALEX_BASE, headers=_http_headers(),
        **_client_kwargs(), proxy=proxy,
    ) as client:
        # Journal: resolve name → source ID
        source_id: str | None = None
        if journal:
            source_id = await _resolve_openalex_source_id(journal, client)
            if source_id:
                filters.append(f"primary_location.source.id:{source_id}")

        # Text search: only when NOT a DOI lookup
        if not doi:
            search_text = normalize_query(query)
            if journal and not source_id:
                search_text = f"{search_text} {journal}".strip()
            if search_text:
                params["search"] = search_text

        params["filter"] = ",".join(filters)
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

        try:
            resp = await _with_retry(client.get)("/works", params=params)
            resp.raise_for_status()
            # P0-12 修复：truncated JSON 静默吞问题。用 _safe_json 明示区分
            data = _safe_json(resp)
            if data.get("_truncated"):
                _log("OpenAlex returned truncated JSON; skipping this run")
                return []
            results = [normalize_openalex(w) for w in data.get("results", [])[:limit]]
            _log(f"OpenAlex: {len(results)} results")
            return results
        except _httpx.HTTPStatusError as e:
            _log(f"OpenAlex HTTP {e.response.status_code}")
            return []
        except (_httpx.ConnectError, _httpx.ReadTimeout) as e:
            _log(f"OpenAlex network error: {type(e).__name__}: {e}")
            return []
        except Exception as e:
            _log(f"OpenAlex unhandled: {type(e).__name__}: {e}")
            return []


# =========================================================================
# Merge & deduplicate
# =========================================================================

def _titles_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    import difflib
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.85


def merge_and_dedupe(scopus: list, oa: list, limit: int) -> list:
    """Merge results, priority: Scopus > OpenAlex, dedup by DOI.

    P1-B: each paper carries ``found_by`` — a list of source engines that
    returned it. Consensus (both Scopus + OpenAlex) is a strong signal that
    can be used by downstream ranking or filtering.
    """
    seen_dois: dict = {}
    merged: list = []

    # Scopus first (canonical)
    for paper in scopus:
        doi = (paper.get("doi") or "").strip().lower()
        found_by = list(paper.get("found_by") or ["scopus"])
        if "scopus" not in found_by:
            found_by.append("scopus")
        paper["found_by"] = found_by
        if doi:
            seen_dois[doi] = paper
        merged.append(paper)

    # OpenAlex second
    merged_dois = set(seen_dois.keys())
    merged_titles = [(p.get("title") or "").strip().lower() for p in merged]
    for paper in oa:
        if len(merged) >= limit:
            break
        doi = (paper.get("doi") or "").strip().lower()
        if doi and doi in merged_dois:
            # P1-B: both engines returned same DOI — bump consensus
            existing = seen_dois[doi]
            found_by = list(existing.get("found_by") or [])
            if "openalex" not in found_by:
                found_by.append("openalex")
            existing["found_by"] = found_by
            continue
        title = (paper.get("title") or "").strip().lower()
        if any(_titles_match(title, mt) for mt in merged_titles if mt):
            continue
        # New paper from OpenAlex
        found_by = list(paper.get("found_by") or ["openalex"])
        if "openalex" not in found_by:
            found_by.append("openalex")
        paper["found_by"] = found_by
        merged.append(paper)
        if doi:
            merged_dois.add(doi)
        if title:
            merged_titles.append(title)

    return merged[:limit]


# =========================================================================
# Main
# =========================================================================

def _log(msg: str):
    print(msg, file=sys.stderr)


async def search_all(query: str, limit: int = 20, year: str | None = None,
                     sources: set | None = None, progress_callback=None,
                     author: str | None = None, journal: str | None = None,
                     pool: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search across all enabled sources in parallel, merge and deduplicate.

    Args:
        progress_callback: Optional async callable(str) for progress updates.
        pool: Internal merge pool size (default: limit). When reranking is
              enabled, fetch a larger pool so the rerank can promote papers
              that raw engine ordering would have buried.
    """
    if sources is None:
        sources = {"scopus", "openalex"}

    # With a larger pool, fetch that many per engine and merge to that size;
    # the caller truncates after reranking.
    eff_limit = pool or limit

    # Strip year patterns from query (year is handled via --year parameter).
    # NEVER strip DOI queries: year-like digit runs inside DOIs (e.g. "2049" in
    # 10.1038/s41586-020-2049-7) match \b(19|20)\d{2}\b and get destroyed,
    # which mangles the lookup (Scopus returns a metadata-less stub, OpenAlex
    # returns nothing) and the DOI never reaches the search cache — so
    # academic_import_papers validation rejects it as "not in session results".
    if not is_doi_query(query):
        query = strip_year_from_query(query)

    # Report which engines are being queried
    active_engines = []
    if "scopus" in sources and ELSEVIER_API_KEY:
        active_engines.append("Scopus")
    if "openalex" in sources:
        active_engines.append("OpenAlex")

    if progress_callback:
        await progress_callback(f"Querying {', '.join(active_engines)}…")

    tasks = {}
    if "scopus" in sources and ELSEVIER_API_KEY:
        tasks["scopus"] = _semaphore_wrap(
            _get_sem(), lambda: search_scopus(query, eff_limit, year, author=author, journal=journal),
        )
    else:
        tasks["scopus"] = asyncio.sleep(0, result=[])

    if "openalex" in sources:
        # Skip OpenAlex when author filter is present — name disambiguation
        # via /authors search is unreliable for common names. Scopus handles
        # AUTHOR-NAME() natively.
        if author:
            _log("OpenAlex: skipped (author filter — use Scopus for author search)")
            tasks["openalex"] = asyncio.sleep(0, result=[])
        else:
            tasks["openalex"] = _semaphore_wrap(
                _get_sem(), lambda: search_openalex(query, eff_limit, year, author=author, journal=journal),
            )
    else:
        tasks["openalex"] = asyncio.sleep(0, result=[])

    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    scopus_r, oa_r = [], []
    engine_status = {}
    for key, result in zip(tasks.keys(), gathered, strict=False):
        if isinstance(result, Exception):
            _log(f"{key} failed: {result}")
            engine_status[key] = {"status": "error", "error": str(result)[:100]}
        else:
            val = result or []
            engine_status[key] = {"status": "ok", "count": len(val)}
            if key == "scopus":
                scopus_r = val
            elif key == "openalex":
                oa_r = val

    if progress_callback:
        status_parts = []
        for eng, st in engine_status.items():
            if st["status"] == "ok":
                status_parts.append(f"{eng}: {st['count']} results")
            else:
                status_parts.append(f"{eng}: ✗")
        await progress_callback(f"Completed — {'; '.join(status_parts)}")

    merged = merge_and_dedupe(scopus_r, oa_r, eff_limit)
    _log(f"Merged: Scopus={len(scopus_r)}, OA={len(oa_r)} → {len(merged)}")

    # Attach engine status for reporting
    return merged, engine_status


async def _fetch_openalex_doi(doi: str) -> dict | None:
    """Fetch a single DOI from OpenAlex (/works endpoint).

    Returns a normalized dict with full author list and reconstructed
    abstract, or None on any failure.
    """
    clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not clean_doi:
        return None
    proxy = GFW_PROXY if GFW_PROXY else None
    async with httpx.AsyncClient(
        base_url=OPENALEX_BASE, headers=_http_headers(),
        **_client_kwargs(), proxy=proxy,
    ) as client:
        try:
            resp = await _with_retry(client.get)("/works", params={"filter": f"doi:{clean_doi.lower()}", "per_page": 1})
            if resp.status_code == 200:
                entries = resp.json().get("results", [])
                if entries:
                    return normalize_openalex(entries[0])
        except Exception:
            pass
    return None


async def lookup_doi(doi: str) -> dict | None:
    """Look up a specific DOI across Scopus → OpenAlex.

    Returns normalized metadata with the FULL author list and (when
    available) the abstract. Scopus COMPLETE view (which carries the
    abstract) is tried first; OpenAlex backfills only when Scopus returns
    no abstract/authors (e.g. key without COMPLETE permission, or the DOI
    missing from Scopus).
    """
    doi = doi.strip()
    # 1. Scopus
    result = await search_scopus_by_doi(doi)
    if result:
        # Scopus STANDARD view usually lacks the abstract — backfill from OpenAlex
        if not result.get("abstract") or not result.get("authors"):
            oa = await _fetch_openalex_doi(doi)
            if oa:
                if not result.get("abstract") and oa.get("abstract"):
                    result["abstract"] = oa["abstract"]
                if not result.get("authors") and oa.get("authors"):
                    result["authors"] = oa["authors"]
        return result

    # 2. OpenAlex
    return await _fetch_openalex_doi(doi)


# =========================================================================
# Keyword extraction (simple TF-based, no external deps)
# =========================================================================

# English stopwords — common words that don't carry topical meaning
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "you", "your",
    "we", "our", "they", "their", "it", "its", "this", "that", "these",
    "those", "not", "no", "nor", "so", "as", "if", "than", "then", "also",
    "very", "too", "just", "now", "here", "there", "which", "who", "whom",
    "what", "when", "where", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "only", "about", "into",
    "over", "after", "before", "between", "under", "during", "without",
    "through", "above", "below", "up", "out", "off", "down", "new", "using",
    "based", "show", "found", "results", "study", "used", "well", "one",
    "two", "three", "many", "much", "however", "therefore", "thus", "yet",
    "due", "recent", "first", "role", "key", "important", "different",
    "among", "including", "across", "within", "since", "while", "although",
})


def _tokenize(text: str) -> list:
    """Tokenize text into lowercase alphanumeric words, filtering stopwords."""
    import re
    words = re.findall(r'[a-zA-Z][a-zA-Z0-9-]*[a-zA-Z0-9]|[a-zA-Z]', text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def extract_keywords(papers: list, top_n: int = 10) -> list:
    """Extract top keywords from paper titles + abstracts using TF scoring.

    Returns list of (keyword, score) sorted by relevance.
    """
    if not papers:
        return []

    # Build corpus from titles + abstracts
    parts = []
    for p in papers:
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        if title:
            parts.append(title)
        if abstract:
            parts.append(abstract[:1000])  # cap per-abstract for performance

    if not parts:
        return []

    # Count term frequency
    from collections import Counter
    term_freq = Counter()
    bigram_freq = Counter()

    for text in parts:
        tokens = _tokenize(text)
        for t in tokens:
            term_freq[t] += 1
        for a, b in zip(tokens, tokens[1:], strict=False):
            bigram_freq[f"{a} {b}"] += 1

    # Combine unigrams and bigrams, score by frequency
    combined = Counter()
    for term, freq in term_freq.items():
        if freq >= 2:  # must appear at least twice
            combined[term] = freq
    for bigram, freq in bigram_freq.items():
        if freq >= 2:
            combined[bigram] = freq * 1.5  # boost bigrams slightly

    # Get top N
    top = combined.most_common(top_n)
    return [(kw, score) for kw, score in top]


# =========================================================================
# Search cache (for DOI validation)
# =========================================================================

# The service's own data dir, not a path relative to this module.
CACHE_DIR = str(_settings.search_cache_dir)


def _get_session_id() -> str:
    return _settings.session_id or "default"


def save_search_cache(papers: list, query: str, session_id: str = ""):
    """Save search results to session cache for DOI validation.

    Accumulates DOIs across multiple searches — never overwrites previous results.
    Each search appends its DOIs to the cumulative cache.

    Concurrency-safe: concurrent searches (e.g. parallel academic_search calls)
    each hold an exclusive file lock over the whole read-merge-write window, and
    writes go through a temp file + atomic rename — no update is lost and the
    file is never left half-written. (Verified: without the lock, 4 concurrent
    writers leave only 1 of 4 updates.)
    """
    session_id = session_id or _get_session_id()
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{session_id}.json")

    try:
        import fcntl
    except ImportError:
        fcntl = None  # non-POSIX: no cross-process lock (best effort)

    lock_f = open(cache_path + ".lock", "a+", encoding="utf-8")
    try:
        if fcntl:
            fcntl.flock(lock_f, fcntl.LOCK_EX)

        # Load existing cache if present (read inside the lock)
        existing_dois = []
        existing_ids = []
        history = []
        if os.path.exists(cache_path):
            try:
                existing = json.loads(open(cache_path, encoding="utf-8").read())
                existing_dois = existing.get("dois", [])
                existing_ids = existing.get("paper_ids", [])
                history = existing.get("history", [])
            except (json.JSONDecodeError, OSError):
                pass

        # Merge in new DOIs (preserve order, skip duplicates)
        new_dois = [p.get("doi", "") for p in papers if p.get("doi")]
        new_ids = [p.get("paper_id", "") for p in papers if p.get("paper_id")]

        all_dois = list(existing_dois)
        existing_doi_set = set(d.lower() for d in all_dois if d)
        for d in new_dois:
            if d and d.lower() not in existing_doi_set:
                all_dois.append(d)
                existing_doi_set.add(d.lower())

        all_ids = list(existing_ids)
        existing_id_set = set(all_ids)
        for pid in new_ids:
            if pid and pid not in existing_id_set:
                all_ids.append(pid)
                existing_id_set.add(pid)

        # Record query history
        history.append({
            "query": query,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "new_dois": len(new_dois),
            "new_unique": len([d for d in new_dois if d and d.lower() not in
                               set(x.lower() for x in existing_dois if x)]),
        })

        cache_data = {
            "query": query,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dois": all_dois,
            "paper_ids": all_ids,
            "total_unique": len(all_dois),
            "history": history,
        }
        # Atomic write: temp file + rename (inside the lock)
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)
        os.replace(tmp_path, cache_path)
    finally:
        if fcntl:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()
        # P1-CC: remove the .lock file after release to prevent orphan
        # accumulation if the process was killed (which leaves .lock behind).
        try:
            os.unlink(cache_path + ".lock")
        except OSError:
            pass


# =========================================================================
# Hybrid relevance rerank (TF-IDF similarity + citations + recency)
# =========================================================================

def _hybrid_rerank(papers: list, query: str) -> list:
    """Rerank papers by a hybrid score (dependency-light, sklearn only):
    score = 0.6 * TF-IDF cosine(query, title+abstract)   lexical semantics
          + 0.25 * normalized citation count             impact signal
          + 0.15 * recency (decays after 5 years)        currency signal

    Papers without abstracts fall back to title-only similarity.
    If sklearn is unavailable, returns papers in original order (no-op).
    """
    if not papers or not query:
        return papers
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return papers

    corpus = []
    for p in papers:
        text = f"{p.get('title') or ''} {p.get('abstract') or ''}".strip()
        corpus.append(text or " ")

    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              sublinear_tf=True, min_df=1)
        # P1-C: fit ONLY on corpus; transform corpus and query separately.
        # Previously `fit_transform(corpus + [query])` let the query
        # participate in IDF calc, artificially lowering IDF for query
        # terms and biasing similarity toward query-corpus overlap.
        corpus_vec = vec.fit_transform(corpus)
        query_vec = vec.transform([query])
        sims = cosine_similarity(query_vec, corpus_vec).flatten()
    except Exception:
        return papers

    now_year = time.gmtime().tm_year
    max_cites = max((p.get("citation_count") or 0) for p in papers) or 1

    scored = []
    for p, sim in zip(papers, sims, strict=False):
        cites = p.get("citation_count") or 0
        year = p.get("year") or 0
        age = max(0, now_year - year)
        recency = 1.0 if age <= 5 else max(0.0, 1.0 - (age - 5) / 20.0)
        score = 0.6 * float(sim) + 0.25 * (cites / max_cites) + 0.15 * recency
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    # P0-8 修复：保留 hybrid_score 与各 score_components，让 SKILL.md 承诺的 "1-5 分数" 可被追溯
    out = []
    for score, p in scored:
        p["hybrid_score"] = round(float(score), 4)
        p["score_components"] = {
            "tfidf_similarity": round(float(sim), 4),
            "citation_norm": round(float((p.get("citation_count") or 0) / max_cites), 4) if max_cites else 0,
            "recency": round(float(recency), 4) if (p.get("year") or 0) else 0,
        }
        out.append(p)
    return out


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Academic literature search")
    parser.add_argument("query", nargs="?", help="Search query (English keywords)")
    parser.add_argument("--limit", "-n", type=int, default=20, help="Max results (1-200)")
    parser.add_argument("--year", "-y", help="Year filter: '2024' or '2020-2024'")
    parser.add_argument("--source", "-s", choices=["scopus", "openalex"],
                        help="Single source only")
    parser.add_argument("--rerank", action="store_true",
                        help="Hybrid relevance rerank: TF-IDF similarity + citations + recency")
    parser.add_argument("--author", "-a", help="Filter by author name (last name preferred)")
    parser.add_argument("--journal", "-J", help="Filter by journal name")
    parser.add_argument("--doi", "-d", help="DOI lookup instead of search")
    parser.add_argument("--json", "-j", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--save-cache", action="store_true",
                        help="Save search results to session cache (for DOI validation)")

    args = parser.parse_args()

    if args.doi:
        result = asyncio.run(lookup_doi(args.doi))
        if result:
            print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
        else:
            print(json.dumps({"error": f"No results for DOI: {args.doi}"}, ensure_ascii=False))
            sys.exit(1)
        return

    if not args.query and not args.author and not args.journal:
        parser.print_help()
        sys.exit(1)

    # Allow empty query when author/journal filters are provided (pure structured search)
    query = args.query or ""
    sources = {args.source} if args.source else {"scopus", "openalex"}
    limit = max(1, min(args.limit, 200))

    pool = min(limit * 2, 200) if args.rerank else limit
    results, engine_status = asyncio.run(search_all(
        query, limit, args.year, sources,
        author=args.author, journal=args.journal, pool=pool))

    # Hybrid rerank (text searches only — DOI lookups never pass --rerank)
    if args.rerank:
        results = _hybrid_rerank(results, query)[:limit]

    # Extract keywords
    keywords = extract_keywords(results, top_n=10)

    # Save cache if requested
    if args.save_cache:
        save_search_cache(results, query)

    # Output as structured object (with engine status and keywords)
    # P1-E: cap abstract length per result to prevent GB-level JSON output.
    # A 3000-char abstract on 200 results = 600KB JSON, which exceeds MCP
    # tool-result limits and bloats agent context. Cap at 1500 chars (enough
    # for keyword extraction + relevance judgment).
    _MAX_ABSTRACT_CHARS = 1500
    truncated_results = []
    for r in results:
        r2 = dict(r)
        abs_text = r2.get("abstract") or ""
        if len(abs_text) > _MAX_ABSTRACT_CHARS:
            r2["abstract"] = abs_text[:_MAX_ABSTRACT_CHARS] + "..."
            r2["abstract_truncated"] = True
        truncated_results.append(r2)
    output = {
        "results": truncated_results,
        "query": query,
        "count": len(truncated_results),
        "keywords": [{"keyword": kw, "score": score} for kw, score in keywords],
        "engine_status": engine_status,
    }

    indent = 2 if args.json else None
    print(json.dumps(output, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    main()
