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
  DOWNLOAD_PROXY          Primary proxy for Scopus (e.g. http://192.168.255.1:8888); unset = direct
  DOWNLOAD_PROXY_MODE     primary|fallback|failover|none — failover retries a transport failure
  DOWNLOAD_PROXY          Rescue route for failover mode
  GFW_PROXY               Primary proxy for OpenAlex when behind GFW
  OPENALEX_API_KEY        Optional OpenAlex premium key

Output:
  JSON array of normalized paper objects, with Scopus results first.
"""

import argparse
import asyncio
import difflib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
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
from ..config import settings as _settings  # noqa: E402

# Shared HTTP plumbing: the route-health memory (order_routes / mark_route_*)
# is what makes failover fast as well as correct — a route that just failed
# while another succeeded is tried last on the next request.
from ..httpclient import mark_route_failed, mark_route_ok, order_routes

logger = logging.getLogger("academic_mcp.agent.search")

# =========================================================================
# Configuration
# =========================================================================

ELSEVIER_API_KEY = _settings.elsevier_api_key
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
            _httpx.ConnectError,
            # `ConnectTimeout` is NOT a subclass of `ConnectError` — both derive
            # from `TransportError`. The tuple used to name only `ConnectError`
            # and `ReadTimeout`, so a connect timeout was the one transport
            # failure that never retried. It is named explicitly (and
            # `TimeoutException`/`TransportError` cover the rest of the family)
            # rather than left to chance: a dropped packet on the first attempt
            # is exactly the case a retry exists for.
            _httpx.ConnectTimeout, _httpx.ReadTimeout, _httpx.TimeoutException,
            _httpx.RemoteProtocolError, _httpx.TransportError,
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


#: Connect timeout for the Elsevier host specifically.
#:
#: Scopus is the only engine that is probed on a path where a *transport* answer
#: is not needed to produce a result — OpenAlex is queried in parallel and the
#: merge is unaffected. Probing it with the same 10 s connect timeout cost a
#: measured 45 s per search when the host is unreachable (3 attempts × 10 s
#: connect + exponential backoff), which is what made "the search is slow" a
#: separate complaint from "Scopus is down". A down host answers a SYN with
#: nothing at all, so a shorter connect budget detects it sooner while a
#: REACHABLE but slow Scopus still gets the full read timeout for its data.
_SCOPUS_CONNECT_TIMEOUT = 4.0


def _scopus_client_kwargs() -> dict:
    kwargs = _client_kwargs()
    kwargs["timeout"] = httpx.Timeout(float(_settings.search_timeout), connect=_SCOPUS_CONNECT_TIMEOUT)
    return kwargs


# ── Proxy routing for search engines ──────────────────────────────────────
#
# Same policy as academic_mcp.httpclient.fetch, in the shape these engine calls
# use: a transport-level failure on the first route retries the request through
# the next route, which is DOWNLOAD_PROXY. An HTTP answer never fails over — the
# engine answered, and another exit cannot change that.

def _route_chain(primary: str | None) -> list[str | None]:
    """One engine's route chain: primary first, rescue hop last.

    ``primary=None`` means a direct connection. In ``failover`` mode the chain
    gains DOWNLOAD_PROXY as its rescue route; other modes leave the chain as it
    was made.

    The rescue route used to come from a separate ``DOWNLOAD_PROXY``
    key, so a deployment that set only DOWNLOAD_PROXY — the natural reading of
    "I have a proxy" — got a single-route chain and no failover at all. The
    rescue route is now the same setting it always should have been.
    """
    routes: list[str | None] = [primary]
    hop = _settings.download_proxy
    if _settings.download_proxy_mode == "failover" and hop and hop != primary:
        routes.append(hop)
    return routes


def _scopus_routes() -> list[str | None]:
    """Scopus route chain: DIRECT first, DOWNLOAD_PROXY as the rescue.

    Not ``_route_chain(DOWNLOAD_PROXY)``: passing the proxy as the primary put
    it at BOTH ends of the chain, so the dedup collapsed it to a single
    proxy-only route. That inverted the intent — the institution-IP direct
    connection is the fast, correct route to api.elsevier.com and the proxy is
    the backup, so routing every search through the proxy would make normal
    operation slower and depend on a hop that exists to be a fallback.
    """
    return _route_chain(None)


def _openalex_routes() -> list[str | None]:
    """OpenAlex route chain: GFW_PROXY when configured, else direct.

    OpenAlex really is reached THROUGH the GFW proxy (that is why the setting
    exists), so here the proxy is the primary route and DOWNLOAD_PROXY is not
    involved at all.
    """
    return _route_chain(GFW_PROXY or None)


class FailoverClient:
    """Small httpx wrapper: per-request retry, plus transport-level failover.

    ``get()`` applies the module's bounded-retry policy (``_with_retry``) on
    the first route and, when that fails at the *transport* level (connect
    timeout, reset, blackhole), retries once on each remaining route — so a
    blackholed primary can no longer take the whole engine down.

    A standalone class rather than a helper in ``academic_mcp.httpclient``
    because these engines use the tenacity retry shape and their own timeouts,
    and because this module is also usable as a script.
    """

    def __init__(self, routes: list[str | None], *, ns: str, **client_kwargs: Any) -> None:
        self._routes = list(routes) or [None]
        self._ns = ns
        # Deterministic routing: never inherit ambient *_proxy variables,
        # which would otherwise silently hijack every call.
        self._client_kwargs = {"trust_env": False, **client_kwargs}
        self._clients: dict[str | None, httpx.AsyncClient] = {}

    def _client_for(self, proxy: str | None) -> httpx.AsyncClient:
        client = self._clients.get(proxy)
        if client is None or client.is_closed:
            kwargs = dict(self._client_kwargs)
            if proxy:
                kwargs["proxy"] = proxy
            client = httpx.AsyncClient(**kwargs)
            self._clients[proxy] = client
        return client

    async def get(self, url: str, *, params: dict | None = None, **kwargs: Any) -> httpx.Response:
        routes = order_routes(self._routes, self._ns)
        failed: list[str | None] = []
        last_exc: Exception | None = None
        for idx, route in enumerate(routes):
            try:
                resp = await _with_retry(self._client_for(route).get)(
                    url, params=params, **kwargs
                )
                mark_route_ok(self._ns, route)
                for bad in failed:
                    mark_route_failed(self._ns, bad)
                return resp
            except (_httpx.TransportError, _httpx.TimeoutException) as exc:
                last_exc = exc
                failed.append(route)
                if idx + 1 < len(routes):
                    _log(
                        f"{self._ns}: {'direct' if route is None else route} "
                        f"unreachable ({type(exc).__name__}); trying "
                        f"{routes[idx + 1] or 'direct'}"
                    )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("no routes configured")

    async def aclose(self) -> None:
        for client in self._clients.values():
            if not client.is_closed:
                await client.aclose()
        self._clients.clear()

    async def __aenter__(self) -> "FailoverClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


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


# ── Scopus engine cooldown ────────────────────────────────────────────────
#
# When api.elsevier.com is unreachable from this host (no route, DNS dropped,
# proxy misconfigured — observed here as ConnectTimeout), EVERY search paid the
# connect timeout before falling back to OpenAlex, because each call re-tried
# from scratch. The user-visible effect is that all literature search is slow
# and the log fills with the same error, while the results are unaffected.
#
# So a TRANSPORT-level failure opens a short cooldown: for the next
# _SCOPUS_COOLDOWN seconds Scopus is skipped outright and OpenAlex answers
# alone. Only transport failures count — an HTTP 401/403/400 means the engine
# answered and is worth asking again, and a per-query miss says nothing about
# reachability. The first call after a cooldown expires always retries, so a
# network that comes back is picked up within one window without a restart.
_SCOPUS_COOLDOWN = 300.0
_scopus_down_until = 0.0


def _scopus_cooling() -> bool:
    return time.monotonic() < _scopus_down_until


def _scopus_transport_failed(exc: BaseException) -> None:
    """Record a transport-level Scopus failure and start the cooldown."""
    global _scopus_down_until
    if isinstance(exc, (_httpx.TransportError, _httpx.TimeoutException)):
        _scopus_down_until = time.monotonic() + _SCOPUS_COOLDOWN


async def search_scopus(query: str, limit: int = 20, year: str = None,
                    author: str = None, journal: str = None) -> list:
    if not ELSEVIER_API_KEY:
        return []
    if _scopus_cooling():
        _log("Scopus: skipped (unreachable recently; OpenAlex answers alone)")
        return []

    headers = {"X-ELS-APIKey": ELSEVIER_API_KEY, "Accept": "application/json"}

    # Route chain: DOWNLOAD_PROXY when configured, else direct; in failover
    # mode a transport failure retries through DOWNLOAD_PROXY. Ambient
    # *_proxy variables are never inherited — routing is explicit.
    routes = _scopus_routes()

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
    async with FailoverClient(
        routes, ns="scopus",
        base_url=SCOPUS_SEARCH_URL, headers={**headers, **_http_headers()},
        **_scopus_client_kwargs(),
    ) as client:
        try:
            resp = await client.get("", params=complete_params)
            if resp.status_code == 200:
                data = resp.json()
                entries = _scopus_real_entries(data)
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
            _log(f"Scopus COMPLETE error: {type(e).__name__}: {e!r}")
            _scopus_transport_failed(e)

    # Attempt 2: STANDARD view (COMPLETE rejected or errored — try with lighter auth)
    standard_params = {**params, "view": "STANDARD", "count": min(limit, 200), "start": 0}
    async with FailoverClient(
        routes, ns="scopus",
        base_url=SCOPUS_SEARCH_URL, headers={**headers, **_http_headers()},
        **_scopus_client_kwargs(),
    ) as client2:
        try:
            resp = await client2.get("", params=standard_params)
            if resp.status_code != 200:
                _log(f"Scopus STANDARD HTTP {resp.status_code}")
                return []
            data = resp.json()
            entries = _scopus_real_entries(data)
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
            _log(f"Scopus STANDARD error: {type(e).__name__}: {e!r}")
            _scopus_transport_failed(e)
            return []


async def _fetch_scopus_abstract(doi: str) -> str | None:
    headers = {"X-ELS-APIKey": ELSEVIER_API_KEY, "Accept": "application/json"}
    async with FailoverClient(
        _scopus_routes(), ns="scopus",
        headers={**headers, **_http_headers()}, **_client_kwargs(),
    ) as client:
        try:
            resp = await client.get(f"{SCOPUS_ABSTRACT_URL}/{doi}")
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


def _scopus_real_entries(data: dict) -> list:
    """Scopus search entries, with the "Result set was empty" stub removed.

    An empty result set is returned as HTTP 200 with a NON-EMPTY `entry` list::

        "entry": [{"@_fa": "true", "error": "Result set was empty"}]

    so code that only checks truthiness of `entry` normalizes the stub into a
    paper with a blank title, no DOI and no authors. That is how a DOI which
    does not exist came back from the service as `ok: true, count: 1`, and how a
    keyword search with no matches could return a phantom blank result.
    """
    entries = data.get("search-results", {}).get("entry", [])
    if isinstance(entries, dict):
        entries = [entries]
    return [e for e in entries if isinstance(e, dict) and "error" not in e]


async def search_scopus_by_doi(doi: str) -> dict | None:
    """Look up a single paper by DOI in Scopus.

    COMPLETE view is tried first — it includes the abstract
    (dc:description) plus the full author list. Falls back to STANDARD
    when the API key lacks COMPLETE-view permission (401/403) or the
    COMPLETE result is empty; STANDARD still returns the author list but
    no abstract.

    A "no such DOI" is reported by Scopus as **HTTP 200 with a NON-EMPTY
    `entry` list whose only element is an error stub**::

        "entry": [{"@_fa": "true", "error": "Result set was empty"}]

    so `if entries:` is always true and the stub was normalized into a "paper"
    with a blank title, no DOI and no authors. That is how a DOI which does not
    exist came back from the service as `ok: true, count: 1` with an empty
    record — and any caller that treats a hit as a real paper (import
    authorisation, for one) accepted it and only discovered the problem at
    download time. Dropping the stub is what makes a miss a miss.
    """
    if not ELSEVIER_API_KEY:
        return None
    if _scopus_cooling():
        _log("Scopus: DOI lookup skipped (unreachable recently; OpenAlex answers alone)")
        return None

    clean_doi = doi.replace("https://doi.org/", "").strip()
    headers = {"X-ELS-APIKey": ELSEVIER_API_KEY, "Accept": "application/json"}

    async with FailoverClient(
        _scopus_routes(), ns="scopus",
        base_url=SCOPUS_SEARCH_URL, headers={**headers, **_http_headers()},
        **_scopus_client_kwargs(),
    ) as client:
        try:
            # COMPLETE view first: carries the abstract + full author list
            resp = await client.get("", params={"query": f"DOI({clean_doi})", "view": "COMPLETE", "count": 1})
            if resp.status_code == 200:
                entries = _scopus_real_entries(resp.json())
                if entries:
                    return normalize_scopus(entries[0], "COMPLETE")
            # Fallback: STANDARD view (authors only, no abstract) — e.g.
            # key without COMPLETE permission (401/403) or empty result
            resp2 = await client.get("", params={"query": f"DOI({clean_doi})", "view": "STANDARD", "count": 1})
            if resp2.status_code == 200:
                entries = _scopus_real_entries(resp2.json())
                if entries:
                    return normalize_scopus(entries[0])
        except Exception as e:
            _log(f"Scopus DOI lookup error: {type(e).__name__}: {e!r}")
            _scopus_transport_failed(e)
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
        resp = await client.get("/sources", params={"search": journal_name, "per_page": 3})
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

    async with FailoverClient(
        _openalex_routes(), ns="openalex",
        base_url=OPENALEX_BASE, headers=_http_headers(),
        **_client_kwargs(),
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
            resp = await client.get("/works", params=params)
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
        except _httpx.TransportError as e:
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
    async with FailoverClient(
        _openalex_routes(), ns="openalex",
        base_url=OPENALEX_BASE, headers=_http_headers(),
        **_client_kwargs(),
    ) as client:
        try:
            resp = await client.get("/works", params={"filter": f"doi:{clean_doi.lower()}", "per_page": 1})
            if resp.status_code == 200:
                entries = resp.json().get("results", [])
                if entries:
                    return normalize_openalex(entries[0])
        except Exception:
            pass
    return None


# ── resolving a work from a partial citation ─────────────────────────────
#
# A user (or a draft's bibliography) usually names a paper the way a human
# would — "the Xu 2021 Continuous Mott transition paper, Nature" — not by DOI.
# Everything below turns that into either one confident DOI or a short list to
# choose from. It NEVER invents one: a wrong DOI is worse than a question,
# because every later step (download, citation, bibliography) treats the DOI as
# ground truth and nothing downstream can tell it was a guess.

#: Title-similarity floor for accepting a candidate as THE work.
#:
#: Two floors, because an author match is independent evidence. With one, a
#: near-miss title is enough: `SequenceMatcher` puts unrelated papers in the same
#: field at 0.85–0.9 surprisingly often ("...in twisted bilayer graphene" vs
#: "...in twisted bilayer WSe2"), and a title-only citation carries no second
#: signal to break the tie. Without one, the bar rises.
_TITLE_MATCH_STRICT = 0.92
_TITLE_MATCH_LOOSE = 0.97


def _title_similarity(a: str, b: str) -> float:
    """How close two titles are, in [0, 1]; case- and punctuation-insensitive.

    Edit-distance similarity, with containment left to :func:`_title_containment`.
    The two are deliberately separate because they fail in opposite directions:
    similarity is high for a title plus a short suffix (a small edit) AND high
    for two near-identical-but-different works, while containment is the only
    thing that can tell a FULL title apart from a FRAGMENT of one. Collapsing
    them into one number is what let a truncated query ("Continuous Mott
    transition") score as confidently as the whole title.
    """
    def norm(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()

    left, right = norm(a), norm(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()


#: Words that turn a title into a variant of itself rather than a different
#: work. A title plus one of these (and little else) is still the same paper.
_TITLE_MARKERS = (
    "reply", "corrigendum", "erratum", "retraction", "comment", "addendum",
    "publisher correction", "author correction", "editorial expression",
)
#: How much text a marker may carry with it. Enough for `: a reply to Xu et al`,
#: not enough for a second clause that narrows the work.
_TITLE_MARKER_MAX_EXTRA = 40

#: Lowest title similarity a hit must reach to be offered as a CANDIDATE.
#: `search_by_title`'s output is not a ranking, it is a question put to the
#: caller ("which of these is it?"), so a hit that is merely the engine's best
#: fuzzy match must not be in it. Measured on this corpus the separation is
#: wide: a correct title scores 1.000, and every unrelated hit — papers about
#: photocatalysts, memristors, heat transport — landed in 0.13-0.44. 0.60 sits
#: in that empty band. Below it, `resolve_work` reports an honest miss, which is
#: a better answer than a list nobody can choose from.
TITLE_CANDIDATE_MIN_SIMILARITY = 0.60


def _title_key(title: str) -> str:
    """Normalised title, for deciding whether two records are the SAME work.

    Lowercased, punctuation and whitespace collapsed. Used to dedup a candidate
    list, where the identity that matters is the work and not the identifier:
    one index may return a record with no DOI while another returns the same
    paper with one, and both must occupy a single slot.
    """
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _title_suffix_is_marker(query: str, candidate: str) -> bool:
    """Whether the candidate is the query plus a variant marker, not a superset.

    This is the check that tells a COMPLETE title from a FRAGMENT of one, which
    containment alone cannot: "Continuous Mott transition" and "Continuous Mott
    transition in semiconductor moiré superlattices: a reply to Xu" are both
    substrings of some published title, so both look "contained". What separates
    them is what is left over — a fragment leaves the identifying tail of the
    real title unaccounted for, while a variant leaves only a marker.
    """
    import re

    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())).strip()

    left, right = norm(query), norm(candidate)
    if not left or not right:
        return False
    if right == left:
        return True
    # Only the case where the candidate says MORE than the caller asked about.
    if left not in right:
        return False
    extra = right.replace(left, " ", 1).strip()
    if not extra or len(extra) > _TITLE_MARKER_MAX_EXTRA:
        return False
    return any(marker in extra for marker in _TITLE_MARKERS)


def _surname(name: str) -> str:
    """Best-effort surname from the spellings the engines actually return.

    Those are `Sun J.Y.`, `Kaur P.`, `Rahul` — a surname FIRST with initials
    after it, not the `First Last` order Western names are usually written in.
    Taking the last token therefore returned an initial for two of those three,
    so an author the caller named ("Xu") matched nobody and every candidate was
    rejected. The rule below reads the first token that is not an initial.
    """
    import re

    text = (name or "").strip()
    if not text:
        return ""
    if "," in text:  # `Last, First` — the committed answer is before the comma
        return text.split(",", 1)[0].strip().lower()
    for token in text.split():
        # An initial is a single letter, optionally followed by `.` and/or more
        # single letters (`J.Y.`). Anything longer is a name.
        if re.fullmatch(r"[A-Za-z](?:\.[A-Za-z])*\.?", token):
            continue
        return token.strip(".").lower()
    return ""


def _author_matches(paper: dict, author: str) -> bool:
    """Whether ``author`` names any author of ``paper`` (surname containment)."""
    want = _surname(author)
    if not want:
        return False
    for entry in paper.get("authors") or []:
        name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
        if want and want in _surname(name):
            return True
    return False


def _compact_candidate(paper: dict) -> dict:
    """The four fields a caller needs to show a paper to a human and pick one."""
    return {
        "doi": paper.get("doi") or "",
        "title": paper.get("title") or "",
        "authors": [
            a.get("name", "") if isinstance(a, dict) else str(a)
            for a in (paper.get("authors") or [])
        ],
        "year": paper.get("year"),
        "venue": paper.get("venue") or "",
    }


async def search_openalex_by_title(title: str, limit: int = 10) -> list[dict[str, Any]]:
    """OpenAlex works whose TITLE matches ``title``.

    `search_by_title` used to send a title through the keyword search, which
    searches title AND abstract. For a title that is the wrong field: measured
    here, the query "Continuous Mott transition" returned five papers about
    photocatalysts and memristors — the engine ranked topical relevance over the
    title — and the paper actually being sought did not appear at all, because
    the engine's top N is not the caller's ranking rule.

    ``filter=title.search:`` matches the title field alone, which is what a
    citation lookup means. The engine is still asked for more rows than the
    caller wants: matching is not ranking, and `resolve_work` re-scores by
    similarity anyway.
    """
    clean = str(title or "").strip()
    if not clean:
        return []
    params: dict = {
        "filter": f"title.search:{clean},type:article|review",
        "per_page": max(1, min(int(limit or 10), 50)),
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    proxy = GFW_PROXY if GFW_PROXY else None
    async with httpx.AsyncClient(
        base_url=OPENALEX_BASE, headers=_http_headers(),
        **_client_kwargs(), proxy=proxy,
    ) as client:
        try:
            resp = await _with_retry(client.get)("/works", params=params)
            resp.raise_for_status()
            data = _safe_json(resp)
            if data.get("_truncated"):
                _log("OpenAlex title search returned truncated JSON; skipping")
                return []
            return [normalize_openalex(w) for w in data.get("results", [])[:limit]]
        except Exception as e:
            _log(f"OpenAlex title search failed: {type(e).__name__}: {e!r}")
            return []


async def search_scopus_by_title(title: str, limit: int = 10,
                                 author: str | None = None,
                                 journal: str | None = None) -> list[dict[str, Any]]:
    """Scopus entries whose ``TITLE()`` matches ``title``.

    Same reasoning as :func:`search_openalex_by_title`: the relevance-ranked
    keyword query is the wrong instrument for "which paper IS this", and Scopus
    has a field code for the right one.
    """
    clean = str(title or "").strip()
    if not clean or not ELSEVIER_API_KEY:
        return []
    if _scopus_cooling():
        _log("Scopus: title search skipped (unreachable recently)")
        return []

    parts = [f"TITLE({clean})"]
    if author:
        parts.append(f"AUTHOR-NAME({author})")
    if journal:
        parts.append(f"SRCTITLE({journal})")
    headers = {"X-ELS-APIKey": ELSEVIER_API_KEY, "Accept": "application/json"}
    params = {
        "query": " AND ".join(parts),
        "view": "STANDARD",
        "count": max(1, min(int(limit or 10), 25)),
    }
    async with httpx.AsyncClient(
        base_url=SCOPUS_SEARCH_URL, headers={**headers, **_http_headers()},
        **_scopus_client_kwargs(),
    ) as client:
        try:
            resp = await _with_retry(client.get)("", params=params)
            if resp.status_code != 200:
                _log(f"Scopus title search HTTP {resp.status_code}")
                return []
            entries = _scopus_real_entries(resp.json())
            return [normalize_scopus(e, "STANDARD") for e in entries[:limit]]
        except Exception as e:
            _log(f"Scopus title search error: {type(e).__name__}: {e!r}")
            _scopus_transport_failed(e)
            return []


async def search_by_title(title: str, author: str = "", journal: str = "",
                          limit: int = 5) -> list[dict]:
    """Find papers by title (+ optional author/journal), best match first.

    Three sources, deliberately: each engine's TITLE() field match, which is
    precise but varies between indexes (Scopus abbreviates, OpenAlex stores the
    published form, and either may carry a subtitle the caller omitted), plus the
    keyword search as a recall net for a title spelled differently in both.

    It does NOT judge which hit is the right paper, and it does not return every
    hit: candidates below :data:`TITLE_CANDIDATE_MIN_SIMILARITY` are dropped
    here, because the caller turns this list into a question. The run that
    prompted this returned five photocatalysis papers scoring 0.13-0.20 as
    "candidates" for a moiré-physics title, and a caller shown that list can only
    choose a wrong answer.
    """
    query = str(title or "").strip()
    if not query:
        return []
    want = max(1, min(int(limit or 5), 20))

    # Ask every source for more than the caller wants: matching is not ranking.
    pool = max(want * 3, 10)
    batches = await asyncio.gather(
        search_scopus_by_title(query, pool, author=author or None, journal=journal or None),
        search_openalex_by_title(query, pool),
        search_openalex(query, pool, author=author or None, journal=journal or None),
        return_exceptions=True,
    )

    # Dedup by NORMALISED TITLE, not by DOI. Keying on the DOI made the same
    # work appear twice whenever one index returned the record without one —
    # observed here as Scopus' DOI-less "Continuous Mott transition in
    # semiconductor moire superlattices" sitting next to OpenAlex' DOI-bearing
    # record at similarity 0.984. Both entries are the same paper, and offering
    # them as two choices asks the caller to pick between identical things.
    by_title: dict[str, dict] = {}
    for batch in batches:
        if isinstance(batch, BaseException) or not batch:
            continue
        for paper in batch:
            title_key = _title_key(paper.get("title") or "")
            if not title_key:
                # No title to judge it by: keep a DOI'd record only.
                title_key = (paper.get("doi") or "").strip().lower()
                if not title_key:
                    continue
            existing = by_title.get(title_key)
            # Prefer the richer record: a DOI is what makes the paper
            # downloadable, so an entry that has one wins the slot.
            if existing is None or (
                not (existing.get("doi") or "").strip() and (paper.get("doi") or "").strip()
            ):
                by_title[title_key] = paper

    scored: list[tuple[float, dict]] = []
    for paper in by_title.values():
        score = _title_similarity(query, paper.get("title") or "")
        if score < TITLE_CANDIDATE_MIN_SIMILARITY:
            continue
        scored.append((score, paper))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [paper for _, paper in scored[:want]]


async def resolve_work(doi: str = "", title: str = "", author: str = "",
                       journal: str = "") -> dict:
    """Resolve whatever the caller has into ONE work, or a list to choose from.

    Returns ``{"ok": True, "resolved": True, "paper": {...}}`` when exactly one
    candidate is confidently the work, ``{"ok": True, "resolved": False,
    "candidates": [...]}`` when the answer is ambiguous, and ``{"ok": False,
    "error": ...}`` when nothing was found.

    A DOI short-circuits everything: it is an IDENTIFIER, not a search, so its
    metadata is authoritative and it is never compared against a title. Rejecting
    a DOI because its title disagrees with the caller's would refuse the one
    input that cannot be wrong.

    PURE resolver, like :func:`lookup_doi` — no cache write, no download. Every
    caller decides for itself what a resolution authorises.
    """
    clean_doi = str(doi or "").strip()
    if clean_doi:
        paper = await lookup_doi(clean_doi)
        if paper is None:
            return {
                "ok": False,
                "doi": clean_doi,
                "error": f"no metadata for DOI {clean_doi}",
                "hint": "The DOI may be mistyped, or the work may not be indexed "
                        "by Scopus/OpenAlex. Check it against the source you took it from.",
            }
        return {"ok": True, "resolved": True, "paper": _compact_candidate(paper),
                "matched_by": "doi"}

    clean_title = str(title or "").strip()
    if not clean_title:
        return {
            "ok": False,
            "error": "a DOI or a title is required",
            "hint": "Pass a DOI when it is known — it is the only form that cannot "
                    "be misread. A title alone works but may return candidates.",
        }

    candidates = await search_by_title(clean_title, author=author, journal=journal)
    # Rank by TITLE SIMILARITY, not by the engines' own relevance order. Theirs
    # answers "what is about this topic"; a citation lookup asks "which of these
    # IS this paper", and the two disagree exactly when the title is imprecise —
    # the case this resolver exists for.
    for paper in candidates:
        paper["_title_score"] = _title_similarity(clean_title, paper.get("title") or "")
        paper["_title_exact"] = _title_suffix_is_marker(clean_title, paper.get("title") or "")
        paper["_author_match"] = _author_matches(paper, author) if author else False
    candidates.sort(key=lambda p: (p["_title_score"], p["_author_match"]), reverse=True)
    if not candidates:
        return {
            "ok": False,
            "title": clean_title,
            "error": "no paper found for that title",
            "hint": "Retry with the exact published title, or add author/journal. "
                    "If it is still not found, search for the topic instead.",
        }

    # A confident single answer needs THREE things, and the third is the one
    # that stops a guess: the query must be the whole title (coverage), not a
    # fragment of one. Then, with an author given, that author must be on the
    # paper — a named author who matches nobody is evidence AGAINST the
    # candidate, not neutral, because the caller told us who wrote it.
    def confident(paper: dict) -> bool:
        score = paper.get("_title_score") or 0.0
        if not _titles_match(clean_title, paper.get("title") or ""):
            return False
        # The query must BE the title (or the title plus a variant marker). A
        # query that is only part of it leaves the identifying tail unread, and
        # every candidate sharing that opening would qualify.
        if not paper.get("_title_exact"):
            return False
        # The author is consulted BEFORE the score shortcut. Checking the score
        # first let an exact title with a mismatched author resolve anyway — the
        # caller had named who wrote the paper, so a candidate they are not on is
        # evidence against it, not a detail the title can overrule.
        if author and not paper.get("_author_match"):
            return False
        if score >= _TITLE_MATCH_LOOSE:
            return True
        if author:
            # With the author confirmed, a looser title match is enough: an
            # independent identifier agreeing with an imperfect title is
            # stronger evidence than a perfect title alone.
            return score >= _TITLE_MATCH_STRICT
        return False

    def is_variant(query: str, candidate_title: str) -> bool:
        """True when the candidate is the query PLUS a marker — i.e. a reply,
        corrigendum or erratum, which is a DIFFERENT work.

        Both the marker and the fact that the candidate says more are required.
        A marker alone would flag a paper whose title merely opens with the
        query's words; extra text alone would flag any superset. Together they
        name the one case the resolver must not answer with: the user asked for
        the paper and got its rebuttal.
        """
        import re

        def norm(text: str) -> str:
            return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()

        left, right = norm(query), norm(candidate_title)
        if not left or not right or left == right or left not in right:
            return False
        extra = right.replace(left, " ", 1).strip()
        return any(marker in extra for marker in _TITLE_MARKERS)

    top = candidates[0]
    # A reply/corrigendum is a different work, so answering the base title with
    # it would be wrong even though the title "contains" the query. Hand back
    # the choice instead of silently preferring one.
    if is_variant(clean_title, top.get("title") or ""):
        return {
            "ok": True,
            "resolved": False,
            "title": clean_title,
            "candidates": [_compact_candidate(p) for p in candidates[:5]],
        }
    if confident(top):
        return {
            "ok": True,
            "resolved": True,
            "paper": _compact_candidate(top),
            "matched_by": "title",
            "title_score": round(top.get("_title_score") or 0.0, 3),
        }

    # Not confident: hand back what was found and let a human choose. Picking the
    # top hit here is exactly the guess this function exists to refuse.
    return {
        "ok": True,
        "resolved": False,
        "title": clean_title,
        "candidates": [_compact_candidate(p) for p in candidates[:5]],
    }


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
# Word tokens for the built-in reranker. Letters/digits plus intra-word `-` and
# `'`, which is what paper titles actually contain ("MoTe2", "GW-BSE",
# "chalcogenide's").
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")

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
    # The cache path is `<dir>/<sid>.json`, so the id is a filename: it goes
    # through the same guard `validate.check` uses. Otherwise a malformed id is
    # WRITTEN under one name and READ under another, and the mismatch reaches
    # the caller as "this DOI was never searched" -- the opposite of the truth.
    from .memory import normalise_session_id

    session_id = normalise_session_id(session_id)
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
        # The `.lock` file is deliberately NOT unlinked. Removing it looks like
        # tidiness and is a correctness bug: a waiter that already opened the
        # inode blocks on a file nobody will ever unlock again, while a third
        # writer creates a NEW file at the same path and locks that — two
        # processes inside a critical section that is documented as exclusive.
        # What that costs is silent: the merge below reads, unions and rewrites
        # the DOI list, so an interleaved writer's DOIs are simply gone, and the
        # only symptom is a later `validate_doi` refusal for a paper the session
        # really did search for. An empty leftover file is the cheaper artefact.
        # (`memory.save_memory` keeps its lock file for the same reason.)


# =========================================================================
# Hybrid relevance rerank (TF-IDF similarity + citations + recency)
# =========================================================================

def _tokenize(text: str) -> list[str]:
    """Lowercased word and word-bigram tokens for the fallback scorer.

    Bigrams matter for the same reason the sklearn path sets
    `ngram_range=(1, 2)`: a title search is usually a phrase, and unigrams alone
    reward a paper that merely shares the common words.
    """
    words = [w for w in _WORD_RE.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


def _tfidf_cosines(corpus: list, query: str) -> list:
    """TF-IDF cosine similarity of `query` against each document.

    The pure-Python path, used when scikit-learn is not installed. It replaced a
    silent no-op: the previous code returned the input order with no
    `hybrid_score` at all, so `rerank=True` — the default — looked like it worked
    while the ranking was simply the engines'. This is a real implementation of
    the same measure (sublinear tf, smoothed idf, bigrams, cosine), not an
    approximation of one. IDF comes from the documents only; the query is
    transformed with those IDFs and never contributes to them, which is the rule
    the sklearn path documents.
    """
    doc_tokens = [_tokenize(doc) for doc in corpus]
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0] * len(corpus)

    document_frequency: Counter = Counter()
    for tokens in doc_tokens:
        document_frequency.update(set(tokens))
    total = max(1, len(doc_tokens))

    def weight(term: str, count: int) -> float:
        idf = math.log((1.0 + total) / (1.0 + document_frequency[term])) + 1.0
        return (1.0 + math.log(count)) * idf

    query_vector = {term: weight(term, n) for term, n in Counter(query_tokens).items()}
    query_norm = math.sqrt(sum(v * v for v in query_vector.values())) or 1.0

    scores: list = []
    for tokens in doc_tokens:
        vector = {term: weight(term, n) for term, n in Counter(tokens).items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        dot = sum(query_vector.get(term, 0.0) * v for term, v in vector.items())
        scores.append(dot / (query_norm * norm))
    return scores


def _hybrid_rerank(papers: list, query: str) -> list:
    """Rerank papers by a hybrid score:
    score = 0.6 * TF-IDF cosine(query, title+abstract)   lexical semantics
          + 0.25 * normalized citation count             impact signal
          + 0.15 * recency (decays after 5 years)        currency signal

    Papers without abstracts fall back to title-only similarity. scikit-learn is
    used when it is importable and the built-in scorer otherwise — never a silent
    pass-through, because a rerank that quietly does nothing is indistinguishable
    from one that agrees with the engines.
    """
    if not papers or not query:
        return papers

    corpus = []
    for p in papers:
        text = f"{p.get('title') or ''} {p.get('abstract') or ''}".strip()
        corpus.append(text or " ")

    sims: list | None = None
    engine = "tfidf-python"
    vectorizer = None
    cosine_similarity = None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer
    except ImportError:
        pass
    if vectorizer is not None and cosine_similarity is not None:
        try:
            vec = vectorizer(stop_words="english", ngram_range=(1, 2),
                             sublinear_tf=True, min_df=1)
            # P1-C: fit ONLY on corpus; transform corpus and query separately.
            # Previously `fit_transform(corpus + [query])` let the query
            # participate in IDF calc, artificially lowering IDF for query
            # terms and biasing similarity toward query-corpus overlap.
            corpus_vec = vec.fit_transform(corpus)
            query_vec = vec.transform([query])
            sims = [float(x) for x in cosine_similarity(query_vec, corpus_vec).flatten()]
            engine = "tfidf-sklearn"
        except Exception as exc:  # noqa: BLE001 - a bad corpus must not fail a search
            logger.warning("sklearn rerank failed (%s); using the built-in scorer", exc)
    if sims is None:
        sims = _tfidf_cosines(corpus, query)

    now_year = time.gmtime().tm_year
    max_cites = max((p.get("citation_count") or 0) for p in papers) or 1

    # Each component travels WITH its paper. The previous version kept only
    # `(score, paper)` and then re-read `sim`/`recency` from the loop variable in
    # the reporting loop, so every paper reported the LAST paper's similarity and
    # recency — a number the model is invited to reason about, silently wrong for
    # all but one row.
    scored = []
    for p, sim in zip(papers, sims, strict=False):
        cites = p.get("citation_count") or 0
        year = p.get("year") or 0
        age = max(0, now_year - year)
        recency = 1.0 if age <= 5 else max(0.0, 1.0 - (age - 5) / 20.0)
        score = 0.6 * float(sim) + 0.25 * (cites / max_cites) + 0.15 * recency
        scored.append((score, p, float(sim), recency))

    scored.sort(key=lambda x: x[0], reverse=True)
    # P0-8 修复：保留 hybrid_score 与各 score_components，让 SKILL.md 承诺的 "1-5 分数" 可被追溯
    out = []
    for score, p, sim, recency in scored:
        p["hybrid_score"] = round(float(score), 4)
        p["rerank_engine"] = engine
        p["score_components"] = {
            "tfidf_similarity": round(sim, 4),
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
