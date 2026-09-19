"""Async wrappers the MCP server exposes for the agent-side helpers.

`search.py`, `snowball.py` and `memory.py` were originally CLI scripts shipped
with the academic-search skill and driven by a client that had to own a Python
interpreter. They live here now, so the service is the only thing that needs
Python and clients talk JSON-RPC only.

Nothing in this module changes their behaviour: the same functions, the same
JSON shapes, the same on-disk formats.
"""

from __future__ import annotations

import asyncio
from typing import Any

from . import memory as memory_mod
from . import search as search_mod
from . import snowball as snowball_mod

# ── search ────────────────────────────────────────────────────────────────


async def search_papers(
    query: str = "",
    doi: str = "",
    limit: int = 20,
    year: str | None = None,
    source: str = "",
    author: str = "",
    journal: str = "",
    session_id: str = "",
    rerank: bool = True,
    save_cache: bool = True,
) -> dict[str, Any]:
    """Scopus (primary) + OpenAlex, merged, deduplicated and hybrid-reranked.

    Mirrors `search.py` exactly, including the two side effects the CLI flags
    carried: `--rerank` (TF-IDF + citations + recency) and `--save-cache`
    (records this session's DOIs so `validate_doi` can authorise a download).
    """
    limit = max(1, min(int(limit or 20), 200))
    year = year or None

    if doi:
        paper = await search_mod.lookup_doi(doi)
        if paper is None:
            return {"ok": False, "error": f"no metadata for DOI {doi}", "results": [], "count": 0}
        # Authorise what we just resolved, exactly as the keyword path does.
        #
        # This branch used to return BEFORE the `save_cache` call below, so a
        # DOI lookup resolved the paper and then declined to record it — and
        # since `validate_doi` authorises downloads by reading that same cache,
        # the documented remedy for a rejected DOI ("academic_search(query=
        # '<DOI>') 已知 DOI 的补票通道") could never work. Verified against the
        # live service: lookup succeeded with count=1 and the DOI was still
        # absent from the cache afterwards, so the following validate_doi
        # rejected it with code `reject`.
        #
        # That made the rejection message actively misleading — it told the
        # caller to take a path that provably could not succeed, which is worse
        # than saying nothing, because the caller retries instead of looking for
        # the real cause.
        if save_cache:
            search_mod.save_search_cache([paper], query or doi, session_id)
        return {"ok": True, "query": doi, "doi": doi, "count": 1, "results": [paper]}

    if not (query or author or journal):
        return {"ok": False, "error": "query, author or journal is required", "results": [], "count": 0}

    sources = {source} if source else {"scopus", "openalex"}
    pool = min(limit * 2, 200) if rerank else limit
    results, engine_status = await search_mod.search_all(
        query, limit, year, sources,
        author=author or None, journal=journal or None, pool=pool,
    )
    if rerank and query:
        results = search_mod._hybrid_rerank(results, query)[:limit]
    keywords = search_mod.extract_keywords(results, top_n=10)
    if save_cache:
        search_mod.save_search_cache(results, query, session_id)

    # Same context guard the CLI applied: a 3000-char abstract on 200 results
    # would exceed tool-result limits and bloat the caller's context.
    trimmed: list[dict[str, Any]] = []
    for paper in results:
        copy = dict(paper)
        abstract = copy.get("abstract") or ""
        if len(abstract) > 1500:
            copy["abstract"] = abstract[:1500] + "..."
            copy["abstract_truncated"] = True
        trimmed.append(copy)

    return {
        "ok": True,
        "query": query,
        "count": len(trimmed),
        "keywords": keywords,
        "engines": engine_status,
        "results": trimmed,
    }


async def citation_chain(
    seeds: list[str],
    direction: str = "both",
    limit: int = 20,
    proxy: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    """Forward/backward citation chain over DOIs (OpenAlex).

    `session_id` matters beyond bookkeeping: `snowball()` authorises the
    returned DOIs by writing them into the session's search cache, and that
    cache is exactly what `validate_doi` reads. Without the caller's session
    the DOIs land in the `default` cache, so `academic_import_papers` rejects
    them for a session that legitimately discovered them.
    """
    clean = [str(seed).strip() for seed in (seeds or []) if str(seed).strip()]
    if not clean:
        return {"ok": False, "error": "at least one seed DOI is required", "results": []}
    limit = max(1, min(int(limit or 20), 100))
    direction = direction if direction in ("forward", "backward", "both") else "both"
    results, stats = await snowball_mod.snowball(
        clean, direction, limit, proxy or "", session_id=session_id
    )
    return {"ok": True, "seeds": clean, "direction": direction, "count": len(results), "stats": stats, "results": results}


# ── memory ────────────────────────────────────────────────────────────────

_MEMORY_OPS = (
    "session-key", "working-memory", "update", "list", "get", "goal", "note",
    "finding", "unresolved", "progress", "resume", "delete-paper", "dump",
)


def _resolve_session(session_id: str) -> str:
    return memory_mod.get_session_id() if not session_id else session_id


async def memory(
    op: str,
    session_id: str = "",
    data: dict[str, Any] | None = None,
    ids: list[str] | None = None,
    topic: str = "",
    full_text: bool = False,
    text: str = "",
    tier: str = "summary",
    limit: int = 20,
    paper_id: str = "",
    doi: str = "",
    skill: str = "",
    cwd: str = "",
) -> dict[str, Any]:
    """One entry point for every research-memory operation.

    Writes go through `save_memory`, which keeps the same flock + tmp + .bak
    protocol the CLI used, so concurrent tool calls cannot corrupt a document.
    """
    op = str(op or "").strip().lower()
    if op not in _MEMORY_OPS:
        return {"ok": False, "error": f"unknown op {op!r}; expected one of {', '.join(_MEMORY_OPS)}"}

    # The one place the cwd → session_id rule lives. Clients call this instead
    # of re-deriving the key, so a key can never disagree between caller and
    # service (that would silently split one project's memory in two).
    if op == "session-key":
        return {"ok": True, "cwd": str(cwd or ""), "session_id": memory_mod.session_key_for_cwd(cwd)}

    sid = _resolve_session(session_id)
    # `working-memory` and `dump` are pure reads; everything else persists.
    mem = memory_mod.load_memory(sid)

    if op == "working-memory":
        return {"ok": True, "session_id": sid, "text": memory_mod.format_working_memory(mem, tier=tier or "summary")}
    if op == "dump":
        return {"ok": True, "session_id": sid, "memory": mem}
    if op == "list":
        return {"ok": True, "session_id": sid, "papers": memory_mod.cmd_list(mem)}
    if op == "get":
        result = memory_mod.cmd_get(mem, ids or [], topic or None, include_full_text=full_text)
        return {"ok": True, "session_id": sid, "result": result}
    if op == "telemetry":
        return {"ok": True, "session_id": sid, "result": memory_mod.cmd_telemetry(mem, limit=int(limit or 20))}

    payload = data or {}
    if op == "update":
        result = memory_mod.cmd_update(mem, payload)
    elif op == "goal":
        result = memory_mod.cmd_goal(mem, str(payload.get("goal") or text or ""))
    elif op == "note":
        result = memory_mod.cmd_note(mem, str(payload.get("text") or text or ""))
    elif op == "finding":
        result = memory_mod.cmd_finding(mem, payload)
    elif op == "unresolved":
        result = memory_mod.cmd_unresolved(mem, str(payload.get("question") or text or ""))
    elif op == "progress":
        result = memory_mod.cmd_progress(mem, payload)
    elif op == "resume":
        result = memory_mod.cmd_resume(mem, skill or "")
    elif op == "delete-paper":
        result = memory_mod.cmd_delete_paper(mem, paper_id or doi, doi)
        if result.get("status") != "ok":
            return {"ok": False, "session_id": sid, "error": result}
    else:  # pragma: no cover - _MEMORY_OPS is exhaustive
        return {"ok": False, "error": f"unhandled op {op!r}"}

    memory_mod.save_memory(sid, mem)
    return {"ok": True, "session_id": sid, "result": result}


async def telemetry(session_id: str = "", limit: int = 20) -> dict[str, Any]:
    """Read the per-session tool-call log written by clients."""
    sid = _resolve_session(session_id)
    mem = memory_mod.load_memory(sid)
    return {"ok": True, "session_id": sid, "result": memory_mod.cmd_telemetry(mem, limit=int(limit or 20))}


def register(server: Any) -> None:
    """Attach every tool above to an MCPServer instance."""

    @server.tool(
        name="search_papers",
        description=(
            "Search the literature: Scopus (needs ELSEVIER_API_KEY) plus OpenAlex, "
            "merged and deduplicated. Pass `doi` for a direct metadata lookup. "
            "Returns normalized paper objects (title, authors, year, doi, abstract, url)."
        ),
    )
    async def search_papers_tool(
        query: str = "",
        doi: str = "",
        limit: int = 20,
        year: str = "",
        source: str = "",
        author: str = "",
        journal: str = "",
        session_id: str = "",
        rerank: bool = True,
        save_cache: bool = True,
    ) -> str:
        result = await search_papers(
            query, doi, limit, year or None, source, author, journal,
            session_id, rerank, save_cache,
        )
        return _payload(result)

    @server.tool(
        name="citation_chain",
        description=(
            "Follow citations from seed DOIs through OpenAlex: `direction` is "
            "forward (who cites them), backward (what they cite) or both."
        ),
    )
    async def citation_chain_tool(
        seeds: list[str],
        direction: str = "both",
        limit: int = 20,
        proxy: str = "",
        session_id: str = "",
    ) -> str:
        return _payload(
            await citation_chain(seeds, direction, limit, proxy, session_id)
        )

    @server.tool(
        name="memory",
        description=(
            "Research memory for one session/project. ops: working-memory (formatted "
            "prompt block), session-key (cwd → session_id, the canonical key rule), "
            "update (papers/findings/topic_map), list, get, goal, note, finding, "
            "unresolved, progress, resume, delete-paper, dump."
        ),
    )
    async def memory_tool(
        op: str,
        session_id: str = "",
        data: dict | None = None,
        ids: list[str] | None = None,
        topic: str = "",
        full_text: bool = False,
        text: str = "",
        tier: str = "summary",
        limit: int = 20,
        paper_id: str = "",
        doi: str = "",
        skill: str = "",
        cwd: str = "",
    ) -> str:
        return _payload(await memory(
            op, session_id, data, ids, topic, full_text, text, tier, limit,
            paper_id, doi, skill, cwd,
        ))

    @server.tool(
        name="telemetry",
        description="Read the academic tool-call log for one session (data/telemetry/<session>.jsonl).",
    )
    async def telemetry_tool(session_id: str = "", limit: int = 20) -> str:
        return _payload(await telemetry(session_id, limit))


def _payload(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


__all__ = ["register", "search_papers", "citation_chain", "memory", "telemetry"]

# `asyncio` is re-exported so callers can drive these helpers from a sync
# context in tests without importing it separately.
_ = asyncio
