#!/usr/bin/env python3
"""Citation chaining (snowballing) via OpenAlex — forward & backward expansion.

Forward:  papers that CITE the seed paper(s)   (filter=cites:Wxxx)
Backward: papers CITED BY the seed paper(s)    (referenced_works field)

Usage:
  python3 snowball.py --seed 10.1038/nature26154 --direction forward --limit 20
  python3 snowball.py --seed 10.1038/xxx --direction backward --limit 20
  python3 snowball.py --seed DOI1 --seed DOI2 --direction both --limit 30

Output: same JSON shape as search.py (results + engine_status).
IMPORTANT: results are written to the session search cache (data/search_cache/),
so DOIs returned here pass academic_import_papers DOI validation — they can be
downloaded directly without re-searching.

Environment:
  GFW_PROXY          Proxy for OpenAlex when behind GFW (also accepted as --proxy)
  OPENALEX_API_KEY   Optional OpenAlex premium key

P2-12: proxy is env-configurable (GFW_PROXY) AND has a --proxy CLI flag.
Previously the audit noted "proxies hardcoded disable" — this was incorrect,
the env var already worked, but the lack of a CLI flag made it undiscoverable.
Now both work and an unset env does NOT default to "none" (preserves trust_env
behaviour so users behind GFW can simply export GFW_PROXY=http://host:port).
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: pip install httpx", file=sys.stderr)
    sys.exit(1)

# Reuse normalization + search-cache logic from search.py (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from .search import (  # noqa: E402
    _client_kwargs,
    _http_headers,
    _log,
    normalize_openalex,
    save_search_cache,
)

# Load .env (same pattern as search.py)
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_ENV_PATH):
    try:
        with open(_ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass

OPENALEX_BASE = "https://api.openalex.org"

# §4.5: bound concurrent in-flight OpenAlex requests via asyncio.Semaphore.
# Backward expansion can fan out to 50+ referenced_works per seed — without
# a cap we risk being throttled. Sem is created lazily inside the loop.
_SNOWBALL_SEM: asyncio.Semaphore | None = None
_SNOWBALL_CONCURRENCY = 5


def _get_sem() -> asyncio.Semaphore:
    global _SNOWBALL_SEM
    if _SNOWBALL_SEM is None:
        _SNOWBALL_SEM = asyncio.Semaphore(_SNOWBALL_CONCURRENCY)
    return _SNOWBALL_SEM
# P2-12: keep env var name GFW_PROXY for backward compat, but the value is now
# overridable per-invocation via --proxy. An unset / empty value leaves httpx
# to honour the system HTTP_PROXY / HTTPS_PROXY (trust_env defaults to True),
# which is the correct behaviour for users behind a GFW who already set the
# standard env vars. To explicitly disable, set GFW_PROXY="none" or pass
# --proxy=none.
from ..config import settings as _settings  # noqa: E402  (after the module docstring block)

GFW_PROXY = _settings.gfw_proxy or ""
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
MAILTO = os.getenv("OPENALEX_MAILTO", "pi-academic-search@localhost")


def _doi_match(work: dict, seed_doi: str) -> bool:
    """Check if OpenAlex work record matches the seed DOI.

    P1-X: seed_dois normalization — strip https://doi.org/ prefix
    and compare lowercase. Backward compat with unnormalized seeds.
    """
    clean = (seed_doi or "").strip().lower().replace("https://doi.org/", "")
    work_doi = ((work.get("doi") or "").replace("https://doi.org/", "")).lower()
    return bool(clean) and clean == work_doi


async def _resolve_work(client: httpx.AsyncClient, doi: str) -> dict[str, Any] | None:
    """Resolve a DOI to an OpenAlex work dict."""
    clean = doi.replace("https://doi.org/", "").strip()
    params = {"filter": f"doi:{clean}", "per_page": 1, "mailto": MAILTO}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    # §4.5: bound concurrent requests so a large backward expansion
    # doesn't get throttled by OpenAlex.
    async with _get_sem():
        try:
            resp = await client.get("/works", params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return results[0] if results else None
        except Exception as e:
            _log(f"snowball: resolve {doi} failed: {e}")
            return None


async def _fetch_works(client: httpx.AsyncClient, filter_str: str, limit: int) -> list[dict[str, Any]]:
    """Fetch works matching an OpenAlex filter string."""
    params = {"filter": filter_str, "per_page": min(limit, 200), "mailto": MAILTO}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    async with _get_sem():
        try:
            resp = await client.get("/works", params=params)
            resp.raise_for_status()
            return resp.json().get("results", [])[:limit]
        except Exception as e:
            _log(f"snowball: filter {filter_str} failed: {e}")
            return []


async def _fetch_work_by_id(client: httpx.AsyncClient, work_id: str) -> dict[str, Any] | None:
    """Fetch a single work by OpenAlex ID (used for backward expansion)."""
    params = {"mailto": MAILTO}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    async with _get_sem():
        try:
            resp = await client.get(f"/works/{work_id}", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None


async def snowball(seeds: list[str], direction: str, limit: int, proxy: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand from seed DOIs. Returns (normalized papers, engine_status).

    Args:
        seeds: Seed paper DOIs.
        direction: "forward" / "backward" / "both".
        limit: Max results.
        proxy: Optional proxy URL (overrides GFW_PROXY env). P2-12.
    """
    # httpx renamed `proxies=` → `proxy=` in 0.26 and REMOVED `proxies=` in
    # 0.28.  (An earlier "fix" inverted this and passed `proxies=`, which
    # raises TypeError on every modern httpx — citation chaining was dead.)
    # P2-12: prefer the explicit argument, fall back to GFW_PROXY env.
    effective_proxy = (proxy or GFW_PROXY or "").strip()
    proxy_arg = {}
    if effective_proxy and effective_proxy.lower() != "none":
        proxy_arg = {"proxy": effective_proxy}
    async with httpx.AsyncClient(
        base_url=OPENALEX_BASE, headers=_http_headers(),
        **_client_kwargs(), **proxy_arg,
    ) as client:
        # 1. Resolve seeds (P1-T: parallel via asyncio.gather)
        seed_works = await asyncio.gather(
            *(_resolve_work(client, doi) for doi in seeds),
            return_exceptions=False,
        )
        seed_works = [w for w in seed_works if w]
        for doi, w in zip(seeds, seed_works, strict=False):
            _log(f"snowball: seed {doi} -> {w.get('id', '?')}")
        for doi in seeds:
            if not any(_doi_match(w, doi) for w in seed_works):
                _log(f"snowball: seed {doi} NOT FOUND in OpenAlex")
        if not seed_works:
            return [], {"status": "error", "error": "no seeds resolved"}

        # 2. Expand (P1-T: parallel expand across seeds)
        expand_tasks = []
        if direction in ("forward", "both"):
            for w in seed_works:
                wid = w["id"].rstrip("/").split("/")[-1]
                expand_tasks.append(_fetch_works(client, f"cites:{wid}", limit))
        if direction in ("backward", "both"):
            for w in seed_works:
                for ref_id in (w.get("referenced_works") or [])[:limit]:
                    expand_tasks.append(_fetch_work_by_id(client, ref_id))
        if expand_tasks:
            expanded = await asyncio.gather(*expand_tasks, return_exceptions=False)
            raw: list[dict] = []
            for e in expanded:
                if isinstance(e, list):
                    raw += e
                elif e is not None:
                    raw.append(e)
        else:
            raw = []

        # 3. Normalize + dedupe (by DOI); drop seeds themselves
        seed_ids = {w["id"].rstrip("/").split("/")[-1] for w in seed_works}
        seed_dois = {s.strip().lower().replace("https://doi.org/", "") for s in seeds}
        seen: dict[str, dict] = {}
        for w in raw:
            wid = (w.get("id") or "").rstrip("/").split("/")[-1]
            if wid in seed_ids:
                continue
            p = normalize_openalex(w)
            if not p.get("doi"):
                continue
            if p["doi"].lower() in seed_dois:
                continue
            key = p["doi"].lower()
            if key not in seen:
                seen[key] = p
        results = list(seen.values())[:limit]

        # 4. Save to session search cache (enables academic_import_papers download)
        if results:
            save_search_cache(results, f"snowball:{direction}({','.join(seeds)})")

        status = {"status": "ok", "count": len(results),
                  "direction": direction, "seeds": len(seed_works)}
        return results, status


def main():
    parser = argparse.ArgumentParser(description="Citation chaining (snowballing) via OpenAlex")
    parser.add_argument("--seed", "-s", action="append", required=True,
                        help="Seed paper DOI (repeatable)")
    parser.add_argument("--direction", "-d", choices=["forward", "backward", "both"],
                        default="forward",
                        help="forward = papers citing the seed; backward = seed's references")
    parser.add_argument("--limit", "-n", type=int, default=20, help="Max results (1-200)")
    # P2-12: CLI proxy flag (overrides GFW_PROXY env). Empty / unset means
    # honour the system HTTP_PROXY via httpx's default trust_env=True.
    parser.add_argument("--proxy", "-p", default="",
                        help="Proxy URL (overrides GFW_PROXY env). Use 'none' to disable.")
    parser.add_argument("--json", "-j", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seed if s.strip()]
    limit = max(1, min(args.limit, 200))

    results, status = asyncio.run(snowball(seeds, args.direction, limit, proxy=args.proxy))

    output = {
        "results": results,
        "query": f"snowball:{args.direction}({','.join(seeds)})",
        "count": len(results),
        "engine_status": {"openalex": status},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2 if args.json else None))


if __name__ == "__main__":
    main()
