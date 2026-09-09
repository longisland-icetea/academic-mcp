"""MCP server exposing paper download / conversion as tools.

Transport: streamable HTTP (``/mcp``), stateless — so a plain JSON-RPC
``tools/call`` POST works without a handshake, which is how the pi extension
talks to it.  The same endpoint is registered with pi-mcp-adapter so other
MCP clients (and the main agent) can use it too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import __version__, httpclient, mineru, pipeline, storage
from .agent import tools as agent_tools
from .config import settings
from .resolvers import STATS
from .validate import Denied, check

logger = logging.getLogger("academic_mcp")


def configure_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if settings.log_file:
        try:
            os.makedirs(os.path.dirname(settings.log_file) or ".", exist_ok=True)
            handlers.append(logging.FileHandler(settings.log_file, encoding="utf-8"))
        except OSError as exc:
            print(f"[academic-mcp] cannot open log file: {exc}", file=sys.stderr)
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    # Playwright/Camoufox are extremely chatty at INFO.
    for noisy in ("playwright", "camoufox", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


mcp = MCPServer(
    name="academic-mcp",
    title="Academic Paper Fetcher",
    instructions=(
        "The academic backend: literature search (Scopus + OpenAlex), citation "
        "chains, research memory, paper download and PDF→Markdown conversion. "
        "Clients need no Python of their own — every capability is a tool here."
    ),
    version=__version__,
)

# Bumped whenever a tool's argument surface changes. Clients compare it against
# their own expectations (see the `contract` tool) instead of discovering a
# mismatch from a failed call.
#
# 2 — `citation_chain` gained `session_id`. Clients that pass it rely on the
#     returned DOIs being written to THAT session's search cache; an older
#     server accepts the call and files them under `default`, which makes
#     `validate_doi` reject them later.
CONTRACT_VERSION = 2


def _payload(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _fail(message: str, **extra: Any) -> str:
    body = {"ok": False, "error": message}
    body.update(extra)
    return _payload(body)


# ══════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════


@mcp.tool(
    name="fetch_paper_text",
    description=(
        "Fetch the full text of an academic paper as Markdown. Tries, in order: "
        "arXiv direct, the Elsevier/ScienceDirect API, a direct PDF from the "
        "publisher, then a headless browser, then an arXiv preprint search. "
        "Converts the PDF with MinerU and caches the result on disk. "
        "Set validate=false to bypass the session search-cache authorisation check."
    ),
)
async def fetch_paper_text(
    doi: str,
    title: str = "",
    first_author: str = "",
    session_id: str = "",
    validate: bool = True,
    force: bool = False,
) -> str:
    """Return the paper's Markdown text (or a structured error).

    Args:
        doi: The paper DOI.
        title: Paper title — enables the arXiv-preprint fallback.
        first_author: First author surname, used to confirm preprint matches.
        session_id: pi session id for the search-cache check (defaults to $PI_SESSION).
        validate: Check the DOI against this session's search cache.
        force: Ignore cached Markdown and re-download.
    """
    try:
        normalised = check(doi, session_id or None) if validate else storage.normalize_doi(doi)
    except Denied as exc:
        return _fail(exc.reason, code=exc.code)

    try:
        outcome = await pipeline.get_text(
            doi=normalised,
            title=title,
            first_author=first_author,
            force=force,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("fetch_paper_text failed for %s", normalised)
        return _fail(f"{type(exc).__name__}: {exc}")

    if not outcome.ok:
        return _fail(outcome.error or "download failed", attempts=[a.as_dict() for a in outcome.attempts])

    # The Markdown itself is the payload; metadata goes in a header block so
    # the reader subagent knows what it got and can cite it correctly.
    header = (
        f"# {title or outcome.doi or 'Paper'}\n\n"
        f"<!-- academic-mcp: doi={outcome.doi} key={outcome.key} "
        f"strategy={outcome.strategy} source={outcome.source} "
        f"pdf={outcome.pdf_path} md={outcome.md_path} -->\n\n"
    )
    return header + outcome.text


@mcp.tool(
    name="fetch_paper_pdf",
    description=(
        "Download a paper's PDF and return the local file path (no conversion). "
        "Use fetch_paper_text unless you specifically need the PDF binary."
    ),
)
async def fetch_paper_pdf(
    doi: str,
    title: str = "",
    first_author: str = "",
    session_id: str = "",
    validate: bool = True,
) -> str:
    """Return the path to the cached/downloaded PDF."""
    try:
        normalised = check(doi, session_id or None) if validate else storage.normalize_doi(doi)
    except Denied as exc:
        return _fail(exc.reason, code=exc.code)

    outcome = await pipeline.get_pdf(
        doi=normalised, title=title, first_author=first_author
    )
    if not outcome.ok:
        return _fail(outcome.error or "download failed", attempts=[a.as_dict() for a in outcome.attempts])
    return _payload(outcome.as_dict())


@mcp.tool(
    name="convert_document",
    description=(
        "Convert a document (local path or URL) to Markdown with the MinerU cloud "
        "API: PDF, Word, PPT, Excel, images, HTML. Caches the result and returns "
        "the markdown PATH, size, heading outline and page→line map — then read that "
        "path with the read tool (use offset/limit). Never rely on the short preview "
        "alone. For papers with a DOI prefer fetch_paper_text."
    ),
)
async def convert_document(
    source: str,
    pages: str = "",
    lang: str = "",
    model: str = "",
    name: str = "",
    ocr: bool | None = None,
    force: bool = False,
    outline: bool = True,
    pages_map: bool = False,
    preview_chars: int = 1200,
) -> str:
    """Convert any supported document to Markdown.

    Args:
        source: Local path or http(s) URL.
        pages: Page range, e.g. \"1-10\" or \"2,4-6\" (PDF only). Saves quota.
        lang: Document language (default from config, usually \"en\").
        model: \"vlm\" (best) or \"pipeline\" (faster, cheaper).
        name: Output basename; defaults to the source filename.
        ocr: Force OCR on/off (default: let MinerU decide). Scanned PDFs need it.
        force: Ignore the cache and re-convert.
        outline: Include the heading outline with line numbers.
        pages_map: Include page→start-line map (needs the content-list sidecar).
        preview_chars: How much of the document to inline.
    """
    try:
        result = await asyncio.to_thread(
            mineru.convert_document,
            source,
            name=name or None,
            pages=pages or None,
            lang=lang or None,
            model=model or None,
            ocr=ocr,
            force=force,
            outline=outline,
            pages_map=pages_map,
        )
    except mineru.MineruError as exc:
        return _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("convert_document failed for %s", source)
        return _fail(f"{type(exc).__name__}: {exc}")

    head = _payload({
        "ok": True,
        "source": source,
        "md_path": str(result.md_path),
        "assets": str(result.assets) if result.assets else "",
        "chars": result.chars,
        "lines": result.lines,
        "images": result.meta.get("images", 0),
        "cached": bool(result.meta.get("cached")),
        "model": result.meta.get("model"),
        "outline": result.outline[:60],
        "pages_map": result.pages_map[:400],
    })
    hint = (
        "文档较长：先用 read 带 offset/limit 读 outline 里的目标章节，别整篇读。"
        if result.lines > 400
        else "用 read 读取 md_path 获取全文。"
    )
    return f"{head}\n\n{hint}\n\n--- preview ---\n\n{result.text[:preview_chars]}"


@mcp.tool(
    name="validate_doi",
    description=(
        "Check whether a DOI may be downloaded: it must appear in this session's "
        "search cache and the session must have a research_goal. Cheap — no network."
    ),
)
async def validate_doi(doi: str, session_id: str = "") -> str:
    """Return {ok, doi} or {ok:false, reason, code}."""
    try:
        normalised = check(doi, session_id or None)
    except Denied as exc:
        return _payload({"ok": False, "doi": doi, "reason": exc.reason, "code": exc.code})
    return _payload({"ok": True, "doi": normalised})


@mcp.tool(
    name="health",
    description=(
        "Report server configuration and resolver statistics. Call this first when "
        "downloads fail, to see which backends are configured."
    ),
)
async def health() -> str:
    """Configuration + lifetime resolver stats (no secrets)."""
    import platform

    info: dict[str, Any] = {
        "ok": True,
        "server": "academic-mcp",
        "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "python": platform.python_version(),
        "config": settings.as_public_dict(),
        "cache": {
            "pdfs": _count(settings.pdf_dir, "*.pdf"),
            "markdown": _count(settings.md_dir, "*.md"),
        },
        "resolvers": {
            "calls": STATS.calls,
            "successes": STATS.successes,
            "total_seconds": round(STATS.total_seconds, 1),
            "last_error": STATS.last_error,
            "by_strategy": {
                name: {"calls": c, "successes": s}
                for name, (c, s) in STATS.by_strategy.items()
            },
        },
    }
    return _payload(info)


# Agent-side tools (search / citation chain / memory / telemetry). Registered
# here so the whole academic surface is one MCP endpoint.
agent_tools.register(mcp)


@mcp.tool(
    name="contract",
    description=(
        "Machine-readable tool contract: every tool with its required and "
        "optional arguments, plus a contract version. Clients validate against "
        "this instead of assuming argument names."
    ),
)
async def contract() -> str:
    """The tool surface as data, derived from the live registrations."""
    tools: dict[str, Any] = {}
    for tool in await mcp.list_tools():
        schema = getattr(tool, "input_schema", None) or getattr(tool, "parameters", None) or {}
        props = sorted((schema.get("properties") or {}).keys())
        required = sorted(schema.get("required") or [])
        tools[tool.name] = {
            "required": required,
            "optional": [name for name in props if name not in required],
        }
    return _payload({
        "ok": True,
        "server": "academic-mcp",
        "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "tools": tools,
    })


def _count(directory, pattern: str) -> int:
    try:
        return len(list(directory.glob(pattern)))
    except OSError:
        return 0


# ══════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════


async def _shutdown() -> None:
    # Best-effort: this runs on a fresh loop after the server loop has closed,
    # so anything loop-bound may already be torn down.
    try:
        await httpclient.aclose_client()
    except Exception as exc:  # noqa: BLE001
        logger.debug("shutdown cleanup failed: %s", exc)


def main() -> None:
    configure_logging()
    settings.ensure_dirs()
    mineru.migrate_legacy_cache()
    logger.info(
        "academic-mcp starting on http://%s:%s/mcp (data=%s)",
        settings.host,
        settings.port,
        settings.data_dir,
    )
    if not settings.mineru_token:
        logger.warning("MINERU_TOKEN is unset — PDF→Markdown conversion will fail")
    if not settings.elsevier_api_key:
        logger.info("ELSEVIER_API_KEY unset — ScienceDirect resolver disabled")

    try:
        asyncio.run(
            mcp.run_streamable_http_async(
                host=settings.host,
                port=settings.port,
                streamable_http_path="/mcp",
                stateless_http=True,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        asyncio.run(_shutdown())


if __name__ == "__main__":
    main()
