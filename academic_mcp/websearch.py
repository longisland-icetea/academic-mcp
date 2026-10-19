"""DeepSeek's Search API — the web engine behind the arXiv title rescue.

The resolver that hunts a preprint by title used to ask DDGS, which fans out to
Google, Brave and DuckDuckGo.  Measured on 2026-10-04 from this host all three
refused us (Brave 429, Google 403, DuckDuckGo an empty 202), so 23 of 27 title
searches failed while the preprints sat on arXiv.  DeepSeek's own Search API
answers the same question from one endpoint, in ~0.5 s, without scraping the
engines that were blocking us.

It is a plain JSON endpoint (``POST /v1/tools/search``), so it needs no SDK:

    {query, count, ori_question?, localization?, need_content?}
        -> {results: [{rerank_score, title, url, snippet, summary, content,
                       site_name, site_icon, date, authority}]}

Only ``url`` is used here: the arXiv ID in a result link is a *lead*, and every
lead is verified against arxiv.org metadata before a byte is downloaded.

Two properties of the API shape this module:

* billing is per successful request (5 CNY / 1000), not per result, and the
  account-wide ceiling is 5 QPS.  This path issues at most two queries per
  paper that reaches the last resolver, so the cost is noise, and a larger
  ``count`` is free — which is why it is raised rather than lowered;
* the caller is discovery code whose contract is "return no candidates rather
  than raise".  A missing key, a 401, a 429, a 5xx, malformed JSON and a dead
  network all return ``[]`` after at most ``attempts`` requests.

No proxy is used, on purpose: ``api.deepseek.com`` is the endpoint the DeepSeek
model API itself uses, and routing it through ``GFW_PROXY`` (the arXiv/web
route) would send mainland traffic abroad and make a reachable API look broken.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger("academic_mcp.websearch")

# The API accepts 1–128 Unicode characters per query; a truncated quoted phrase
# is still a phrase, a query over the limit is a 400.
MAX_QUERY_CHARS = 128
# Per-request ceiling.  The resolver puts a whole-discovery bound around the
# call, but a blocking request needs its own, or it outlives that bound.
MAX_REQUEST_SECONDS = 15.0
# Docs: back off and retry on 408/429/5xx; never retry the other 4xx.
RETRY_DELAY_SECONDS = 0.8


def clip_query(text: str, limit: int = MAX_QUERY_CHARS) -> str:
    """Trim ``text`` to ``limit`` characters without cutting a word in half.

    A partial trailing word inside a quoted phrase is enough to turn a good
    phrase match into no result, so the cut is walked back to a space.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip()


def build_queries(title: str, prefix: str = "site:arxiv.org") -> list[str]:
    """Search queries for one title, longest-first and within the API's limit.

    The quoted full title is the precise one; the leading words are the loose
    net for the case the published title differs from the arXiv one (an added
    subtitle, a changed spelling).  Unquoted on purpose — six quoted words that
    are not an exact phrase match nothing at all.
    """
    title = " ".join(title.split())
    if not title:
        return []
    room = max(0, MAX_QUERY_CHARS - len(prefix) - 3)  # space + two quotes
    phrase = clip_query(title, room)
    if not phrase:
        return []
    queries = [f'{prefix} "{phrase}"']
    words = title.split()
    if len(words) > 6:
        loose = clip_query(" ".join(words[:6]), MAX_QUERY_CHARS - len(prefix) - 1)
        queries.append(f"{prefix} {loose}")
    return queries


def search(
    query: str,
    *,
    count: int = 20,
    timeout: float | None = None,
    attempts: int = 2,
) -> list[dict[str, Any]]:
    """Run one Search request.  Blocks; returns ``results`` (possibly empty).

    Never raises for a network or API failure — see the module docstring.
    ``timeout`` defaults to the arXiv-search budget, capped at
    :data:`MAX_REQUEST_SECONDS`.
    """
    key = settings.deepseek_api_key
    if not key:
        logger.debug("DeepSeek search: no DEEPSEEK_API_KEY configured")
        return []
    if not query.strip():
        return []

    url = f"{settings.deepseek_search_base.rstrip('/')}/v1/tools/search"
    budget = settings.arxiv_search_timeout if timeout is None else timeout
    read = min(MAX_REQUEST_SECONDS, max(1.0, budget))
    request_timeout = httpx.Timeout(read, connect=min(5.0, read))
    payload = {"query": query, "count": max(1, min(50, int(count)))}
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(
                timeout=request_timeout,
                follow_redirects=True,
                trust_env=False,
            ) as client:
                response = client.post(url, json=payload, headers=headers)
        except Exception as exc:  # noqa: BLE001 — discovery must not raise
            logger.info(
                "DeepSeek search request failed: %s: %s", type(exc).__name__, exc
            )
            return []

        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError:
                logger.info("DeepSeek search returned unparseable JSON")
                return []
            results = body.get("results") if isinstance(body, dict) else None
            if not isinstance(results, list):
                logger.info("DeepSeek search response has no results array")
                return []
            return [item for item in results if isinstance(item, dict)]

        retryable = response.status_code in (408, 429) or response.status_code >= 500
        if retryable and attempt < attempts:
            logger.info(
                "DeepSeek search HTTP %d; retrying in %.1fs",
                response.status_code,
                RETRY_DELAY_SECONDS,
            )
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        logger.info(
            "DeepSeek search HTTP %d: %s", response.status_code, response.text[:300]
        )
        return []
    return []
