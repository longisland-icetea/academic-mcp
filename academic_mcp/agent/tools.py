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

from .. import pipeline
from .. import storage as storage_mod
from . import memory as memory_mod
from . import search as search_mod
from . import snowball as snowball_mod
from .search import resolve_work

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


def _library_note_state(session_id: str, paper_id: str, doi: str) -> tuple[bool, int]:
    """Whether this project's library already holds notes for a paper.

    The caller needs this to decide between reading a paper and re-reading it.
    `memory op=update` MERGES a paper note rather than replacing it, so a second
    reading pass does not refresh the notes — it appends a near-duplicate at the
    cost of a full model call. Read side-effect free: `load_memory` only reads.
    """
    try:
        library = memory_mod.load_memory(session_id).get("paper_library") or []
    except Exception:  # noqa: BLE001 — a state hint must never fail the fetch
        return False, 0
    index = memory_mod._find_paper_idx(library, paper_id or "", doi or "")
    if index is None:
        return False, 0
    entry = library[index]
    filled = [key for key in ("key_notes", "detailed_notes") if (entry.get(key) or "").strip()]
    # A record written by `distill: false` carries a placeholder, not a reading,
    # so its presence alone must not count as "already read" — that would make
    # the follow-up distillation pass skip the paper forever.
    placeholder = "[未精读]" in (entry.get("detailed_notes") or "")
    return (bool(filled) and not placeholder), len(filled)


async def get_paper(doi: str = "", session_id: str = "", mode: str = "read",
                    title: str = "", distill: bool = True) -> dict[str, Any]:
    """Everything one paper needs, in ONE call: metadata, local paths, and the text.

    This exists because the alternative was worse for everybody. To get a single
    known paper the caller had to run `search_papers(doi=…)` for metadata,
    `validate_doi` for permission, then `fetch_paper_text` for the text — three
    round trips, each with its own shape — so in practice agents skipped all
    three and curled Crossref for the metadata and the publisher for the PDF.
    Measured over one project's recent sessions: 145 hand-rolled HTTP fetches
    against 65 calls to the tools that exist precisely to do this.

    Identification:
      * ``doi`` — authoritative. A DOI is an identifier, so its metadata is
        taken as given and never second-guessed against ``title``.
      * ``title`` (with optional ``author``/``journal``) — resolved through
        :func:`search.resolve_work`. When the title matches more than one
        candidate this returns ``needs_choice`` with the shortlist instead of
        picking one: a wrong DOI would be treated as ground truth by every later
        step and nothing downstream could tell it was a guess.

    `mode`:
      * ``read`` (default) — ensure the paper is on disk. If the markdown is
        already cached this does NO network work at all and returns its path;
        otherwise it downloads and converts, then returns the path.
      * ``meta`` — metadata only, never downloads.

    `distill` is carried for the caller's benefit and changes nothing here: the
    service has no reader model. The caller decides whether to distil, and the
    value is echoed back so a batch report can say what each paper got.

    `_hadNotes` / `_noteCount` report whether this project already holds notes
    for the paper, which is what lets the caller skip a second reading pass: a
    note update MERGES, so re-reading appends a near-duplicate rather than
    refreshing anything.

    The full text is never returned inline — a 200-page paper would be pruned
    to its first 4 KB by the caller's tool-result budget. The return value
    carries `md_path` instead, and the caller reads the sections it needs.
    """
    from .. import validate as validate_mod

    sid = memory_mod.normalise_session_id(session_id or memory_mod.get_session_id())
    clean = str(doi or "").strip()
    wanted_title = str(title or "").strip()
    if not clean and not wanted_title:
        return {"ok": False, "error": "a DOI or a title is required"}

    # Metadata first: it is cheap, it is what `meta` wants, and the title is
    # what the arXiv-preprint fallback needs to confirm a match. A DOI goes
    # straight to `lookup_doi`; only a title needs the resolver, and routing a
    # DOI through it would put the resolver's confidence test in front of the
    # one input that cannot be wrong.
    if clean:
        paper = await search_mod.lookup_doi(clean)
    else:
        resolution = await search_mod.resolve_work(title=wanted_title)
        if not resolution.get("ok"):
            return {
                "ok": False,
                "title": wanted_title,
                "error": resolution.get("error") or "title did not resolve",
                **({"hint": resolution["hint"]} if resolution.get("hint") else {}),
            }
        if not resolution.get("resolved"):
            return {
                "ok": False,
                "needs_choice": True,
                "title": wanted_title,
                "candidates": resolution.get("candidates") or [],
                "hint": "More than one paper matches that title. Show these candidates "
                        "to the user and re-call with the chosen DOI — do not pick one "
                        "yourself, and do not send the title again unchanged.",
            }
        chosen = resolution["paper"]
        paper = await search_mod.lookup_doi(chosen["doi"])
        if paper is None:
            # The resolver found it through a search engine that the per-DOI
            # lookup cannot see (no DOI in the record, or a Scopus/OpenAlex
            # disagreement). The candidate is still the best answer available,
            # but it is not enough to download from, so say so rather than
            # pretending the resolution failed.
            return {
                "ok": False,
                "doi": chosen.get("doi") or "",
                "error": f"resolved '{chosen.get('title') or wanted_title}' but its DOI "
                         f"{chosen.get('doi') or '(missing)'} has no metadata",
                "hint": "Retry with the DOI, or search for the topic — the record found "
                        "for this title cannot be downloaded as it stands.",
            }

    if paper is None:
        return {
            "ok": False,
            "doi": clean,
            "error": f"no metadata for DOI {clean}",
            "hint": "The DOI may be mistyped, or the work may not be indexed by "
                    "Scopus/OpenAlex. Check the DOI against the source you took it from.",
        }

    meta = {
        "doi": paper.get("doi") or clean,
        "paper_id": paper.get("paper_id") or "",
        "title": paper.get("title") or "",
        "authors": [a.get("name") for a in (paper.get("authors") or []) if isinstance(a, dict) and a.get("name")],
        "first_author": paper.get("first_author") or "",
        "year": paper.get("year"),
        "venue": paper.get("venue") or "",
        "volume": paper.get("volume") or "",
        "pages": paper.get("pages") or "",
        "citation_count": paper.get("citation_count"),
    }
    had_notes, note_count = _library_note_state(sid, meta["paper_id"], meta["doi"])
    if str(mode).lower() == "meta":
        return {"ok": True, "mode": "meta", "session_id": sid, "distill": distill,
                "_hadNotes": had_notes, "_noteCount": note_count, **meta}

    # `read`: an already-cached markdown is the whole answer, and it must be
    # answered BEFORE the authorization check — re-reading a paper this project
    # already holds cannot be a download, so it must not be refused as one.
    key = storage_mod.doi_to_key(meta["doi"])
    cached = memory_mod._full_text_path(key) or memory_mod._full_text_path(meta["doi"]) \
        or memory_mod._full_text_path(meta["paper_id"])

    # Authorise the DOI before returning, on EVERY successful path. The cache
    # write used to sit only on the freshly-downloaded path, which split the
    # workflow it exists to serve: `get_paper` on a paper already on disk
    # returned its path without recording the DOI, and the follow-up
    # `import_papers` — which authorises through the same cache — then refused
    # that DOI as "never searched in this session". Resolving a DOI through the
    # service is what makes it citable, so both outcomes record it.
    #
    # It has to happen HERE and not in `lookup_doi`: that function is a pure
    # resolver — DOI in, metadata out, `None` on a miss — and it touches no
    # state, so every caller decides for itself what a resolution authorises.
    try:
        search_mod.save_search_cache([paper], meta["doi"], sid)
    except Exception:  # noqa: BLE001 — authorisation bookkeeping must not fail the read
        pass

    if cached is not None:
        stat = cached.stat()
        return {
            "ok": True, "mode": "read", "session_id": sid, "cached": True,
            "distill": distill, "_hadNotes": had_notes, "_noteCount": note_count,
            "md_path": str(cached), "chars": stat.st_size, **meta,
        }

    # Not on disk: this IS a download, so it needs the session's authorisation.
    try:
        validate_mod.check(meta["doi"], sid)
    except validate_mod.Denied as exc:
        return {"ok": False, "doi": meta["doi"], "error": exc.reason, "code": exc.code}

    try:
        outcome = await pipeline.get_text(
            doi=meta["doi"],
            title=meta["title"],
            first_author=meta["first_author"],
        )
    except Exception as exc:  # noqa: BLE001 — reported to the caller, not raised
        return {"ok": False, "doi": meta["doi"], "error": f"{type(exc).__name__}: {exc}"}

    if not outcome.ok:
        return {
            "ok": False,
            "doi": outcome.doi or meta["doi"],
            "error": outcome.error or "download failed",
            "attempts": [a.as_dict() for a in outcome.attempts],
            "hint": "Every route was tried (arXiv, Elsevier/ScienceDirect, publisher "
                    "PDF, headless browser, arXiv preprint search). A miss here is a "
                    "property of the paper (no open copy), not of the caller's "
                    "permissions — so do not retry the same DOI by other means.",
        }

    # The DOI was authorised before the download (see above); nothing more to
    # record here.
    return {
        "ok": True, "mode": "read", "session_id": sid, "cached": False,
        "distill": distill, "_hadNotes": had_notes, "_noteCount": note_count,
        "md_path": outcome.md_path, "pdf_path": outcome.pdf_path,
        "chars": len(outcome.text or ""),
        "strategy": outcome.strategy, "source": outcome.source,
        **meta,
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
    "session-key", "working-memory", "update", "edit", "list", "get", "stores",
    "goal", "note", "finding", "unresolved", "progress", "resume", "delete-paper", "dump",
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
    field: str = "",
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
    if op == "stores":
        # The project-level stores alone. `dump` was the only way to read these,
        # and it ships the whole document — 912 KB for a 77-paper project, of
        # which the four stores are ~18 KB. Reading them is what an edit is
        # prepared from, so this is on the hot path between "the model wants to
        # fix a note" and "the model has the exact bytes to match".
        result = memory_mod.cmd_stores(mem, field)
        # An unknown field is a caller error, not an empty project: reporting
        # `ok: true` with `status: error` inside would let a wrapper that checks
        # only the outer flag present a failure as a successful read.
        if result.get("status") != "ok":
            return {"ok": False, "session_id": sid, "error": result}
        return {"ok": True, "session_id": sid, "result": result}
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
    elif op == "edit":
        # A rejected edit must NOT persist. `cmd_edit` reports "not found" and
        # "ambiguous" as errors; saving on those would rewrite the document with
        # an unchanged body while still rotating the `.bak` — turning a no-op
        # into a lost rollback point.
        result = memory_mod.cmd_edit(mem, payload)
        if result.get("status") != "ok":
            return {"ok": False, "session_id": sid, "error": result}
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
        name="get_paper",
        description=(
            "Everything one paper needs, in ONE call: metadata, local paths and the "
            "paper's Markdown. Identify it by `doi` (authoritative) or by `title` "
            "(with optional author/journal) — a title that matches several papers "
            "returns `needs_choice` with a shortlist instead of guessing. "
            "mode='read' (default) returns the cached path when the paper is already "
            "local, otherwise downloads and converts it; mode='meta' returns metadata "
            "only and never downloads. `distill` is echoed back for the caller: the "
            "service has no reader model. Use this instead of requesting a publisher "
            "page or a metadata API directly — it applies this session's DOI "
            "authorisation and the service's resolver chain."
        ),
    )
    async def get_paper_tool(doi: str = "", session_id: str = "", mode: str = "read",
                             title: str = "", distill: bool = True) -> str:
        return _payload(await get_paper(doi, session_id=session_id, mode=mode,
                                        title=title, distill=distill))

    @server.tool(
        name="resolve_reference",
        description=(
            "Resolve an incomplete reference to a work: a DOI, or a title with "
            "optional author/journal. Returns one `paper` when a single candidate is "
            "confidently the work, or a `candidates` shortlist when it is ambiguous — "
            "it never guesses. Pure resolver: no download, no authorisation."
        ),
    )
    async def resolve_reference_tool(doi: str = "", title: str = "", author: str = "",
                                     journal: str = "") -> str:
        return _payload(await resolve_work(doi, title, author, journal))

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
            "update (papers/findings/topic_map), edit (literal old_string → new_string "
            "on stored notes/findings, edit-tool semantics), list, get, stores "
            "(project-level stores only), goal, note, finding, unresolved, progress, "
            "resume, delete-paper, dump."
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
        field: str = "",
    ) -> str:
        return _payload(await memory(
            op, session_id, data, ids, topic, full_text, text, tier, limit,
            paper_id, doi, skill, cwd, field,
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
