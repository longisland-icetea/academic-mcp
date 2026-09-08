"""MinerU cloud API client — the ONLY document→Markdown path in this system.

Two callers share it:

* the paper pipeline (:mod:`academic_mcp.pipeline`) wants text only, cached
  under ``data/texts/<doi_key>.md``;
* the ``convert_document`` MCP tool wants a fully-featured conversion of an
  arbitrary user document (images, heading outline, page→line map), cached
  under ``~/.cache/pi-doc-read``.

Everything else about MinerU lives here so there is exactly one
implementation of the upload/poll/extract dance.

MinerU API v4 flow:
    POST /file-urls/batch   → pre-signed upload URL      (local files)
    POST /extract/task/batch → submit URL directly        (remote files)
    PUT  <upload url>        → raw bytes
    GET  /extract-results/batch/{batch_id}  → poll until done/failed
    GET  <full_zip_url>      → result ZIP

Two API quirks that must survive any cleanup:
* the OSS ``PUT`` requires an **empty** ``Content-Type`` (the signature covers
  it) — httpx's default form type yields ``403 SignatureDoesNotMatch``;
* ``Content-Length`` must be explicit — OSS rejects chunked uploads.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import storage
from .config import settings
from .markdown import clean_markdown

logger = logging.getLogger("academic_mcp.mineru")

SUPPORTED_SUFFIX = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
    ".html", ".htm",
}

MAX_BYTES = 200 * 1024 * 1024

# The service's own cache: converted documents never belong to a client.
DOC_CACHE_DIR = Path(
    os.path.expanduser(os.environ.get("ACADEMIC_DOC_CACHE", "~/.cache/academic-mcp/doc-read"))
)


class MineruError(RuntimeError):
    """Conversion failed. There is deliberately no fallback converter."""


class _Transient(RuntimeError):
    """Worth one retry (429 / 5xx / connection hiccup)."""


# ══════════════════════════════════════════════════════════════════════
# Result type
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Conversion:
    """Outcome of a document conversion."""

    md_path: Path
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    assets: Path | None = None
    outline: list[dict[str, Any]] = field(default_factory=list)
    pages_map: list[dict[str, int]] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def lines(self) -> int:
        return self.text.count("\n") + 1


# ══════════════════════════════════════════════════════════════════════
# HTTP plumbing
# ══════════════════════════════════════════════════════════════════════


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.mineru_token}",
        "Content-Type": "application/json",
        "Accept": "*/*",
    }


@contextmanager
def _client() -> Iterator[httpx.Client]:
    """Authed MinerU client for one conversion attempt."""
    with httpx.Client(
        base_url=settings.mineru_api_base,
        headers=_headers(),
        timeout=120,
        trust_env=False,
    ) as client:
        yield client


def _check_token_failure(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise MineruError(
            f"MinerU rejected the token (HTTP {resp.status_code}). The token has a "
            "limited lifetime — get a fresh one at https://mineru.net/apiManage and "
            "update MINERU_TOKEN."
        )


def _create_task(
    client: httpx.Client,
    *,
    files: list[dict[str, Any]],
    model: str,
    lang: str,
    formula: bool,
    table: bool,
    ocr: bool | None,
    url: bool,
) -> tuple[str, list[str]]:
    """Create a batch. Returns (batch_id, upload_urls)."""
    payload: dict[str, Any] = {
        "files": files,
        "model_version": model,
        "language": lang,
        "enable_formula": formula,
        "enable_table": table,
    }
    if ocr is not None:
        payload["is_ocr"] = bool(ocr)

    endpoint = "/extract/task/batch" if url else "/file-urls/batch"
    resp = client.post(endpoint, json=payload)
    _check_token_failure(resp)
    if resp.status_code >= 500 or resp.status_code == 429:
        raise _Transient(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise MineruError(f"MinerU task creation failed: HTTP {resp.status_code} {resp.text[:200]}")
    data = (resp.json() or {}).get("data") or {}
    batch_id = data.get("batch_id")
    if not batch_id:
        raise MineruError("MinerU response missing batch_id")
    return batch_id, list(data.get("file_urls") or [])


def _put_file(upload_url: str, path: Path) -> None:
    data = path.read_bytes()
    with httpx.Client(timeout=300, trust_env=False) as client:
        resp = client.put(
            upload_url,
            content=data,
            headers={"Content-Type": "", "Content-Length": str(len(data))},
        )
    if resp.status_code >= 500 or resp.status_code == 429:
        raise _Transient(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise MineruError(f"MinerU upload failed: HTTP {resp.status_code} {resp.text[:200]}")


def _poll(client: httpx.Client, batch_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + settings.mineru_timeout
    interval = max(1.0, settings.mineru_poll_interval)
    last = ""
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            resp = client.get(f"/extract-results/batch/{batch_id}")
        except httpx.HTTPError as exc:
            last = f"poll error: {type(exc).__name__}"
            continue
        if resp.status_code >= 500 or resp.status_code == 429:
            last = f"HTTP {resp.status_code}"
            continue
        _check_token_failure(resp)
        if resp.status_code != 200:
            raise MineruError(f"MinerU status poll failed: HTTP {resp.status_code}")
        items = ((resp.json() or {}).get("data") or {}).get("extract_result") or []
        if not items:
            last = "no result yet"
            continue
        entry = items[0]
        state = entry.get("state", "")
        if state != last:
            logger.info("MinerU batch %s: %s", batch_id[:8], state)
            last = state
        if state == "done":
            return entry
        if state == "failed":
            raise MineruError(
                f"MinerU extraction failed: {entry.get('err_msg') or 'no reason given'}"
            )
    raise MineruError(
        f"MinerU timed out after {settings.mineru_timeout}s (last state: {last})"
    )


def _download(url: str) -> bytes:
    with httpx.Client(timeout=300, trust_env=False) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise MineruError(f"MinerU result download failed: HTTP {resp.status_code}")
    return resp.content


# ══════════════════════════════════════════════════════════════════════
# ZIP handling
# ══════════════════════════════════════════════════════════════════════

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_IMAGE_LINK = re.compile(r"!\[([^\]]*)\]\((images/[^)\s]+)\)")


def _normalise(text: str) -> str:
    """Flatten to comparable text: strip tags/markdown, collapse whitespace."""
    if not text:
        return ""
    s = _TAG.sub("", text)
    for ch in ("\\*", "*", "`", "$", "\\(", "\\)"):
        s = s.replace(ch, "")
    return _WS.sub(" ", s).strip()


def _block_text(block: Any) -> str:
    if not isinstance(block, dict):
        return ""
    if isinstance(block.get("text"), str):
        return block["text"]

    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key in ("bbox", "page_idx", "type", "level", "text_level"):
                    continue
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(block.get("content", block))
    return " ".join(out)


def _build_pages_map(md_text: str, content_list: list | None) -> list[dict[str, int]]:
    """page → starting line, so the agent can read one page without scanning."""
    if not content_list:
        return []
    norm_lines = [_normalise(x) for x in md_text.split("\n")]
    joined = "\n".join(norm_lines)

    per_page: dict[int, list[str]] = {}
    for block in content_list:
        if not isinstance(block, dict) or not isinstance(block.get("page_idx"), int):
            continue
        snippet = _normalise(_block_text(block))
        if len(snippet) < 8:
            continue
        bucket = per_page.setdefault(block["page_idx"], [])
        if len(bucket) < 6:
            bucket.append(snippet)

    result: list[dict[str, int]] = []
    cursor = 0
    for page in sorted(per_page):
        hit: int | None = None
        for snippet in per_page[page]:
            for probe in (snippet[:80], snippet[:40], snippet[:20]):
                if len(probe) < 8:
                    break
                idx = joined.find(probe, cursor)
                if idx >= 0:
                    hit = idx
                    break
            if hit is not None:
                break
        if hit is None:
            continue  # figure-only page — nothing to anchor on
        result.append({"page": page + 1, "line": joined.count("\n", 0, hit) + 1})
        cursor = hit
    return result


def _build_outline(md_text: str, max_items: int = 200) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    in_fence = False
    for i, line in enumerate(md_text.split("\n"), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*\S)", line)
        if m:
            out.append({"level": len(m.group(1)), "line": i, "title": m.group(2).strip()})
            if len(out) >= max_items:
                break
    return out


def _extract_zip(blob: bytes, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Unpack the result ZIP. Returns (md_path, assets_dir)."""
    assets = out_dir / f"{stem}_assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(assets)

    candidates = sorted(assets.glob("*.md"))
    if not candidates:
        raise MineruError("MinerU result ZIP contained no Markdown")
    src = next((p for p in candidates if p.name.lower() == "full.md"), candidates[0])
    text = src.read_text(encoding="utf-8", errors="replace")
    # Rewrite ![](images/x.jpg) to absolute paths so the agent can read them.
    text = _IMAGE_LINK.sub(lambda m: f"![{m.group(1)}]({assets.resolve()}/{m.group(2)})", text)

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(text, encoding="utf-8")
    return md_path, assets


def _load_content_list(assets: Path) -> list | None:
    matches = sorted(assets.glob("*content_list.json"))
    if not matches:
        return None
    try:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ══════════════════════════════════════════════════════════════════════
# Source resolution
# ══════════════════════════════════════════════════════════════════════


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(text: str) -> str:
    """Filesystem-safe output stem.

    Used as a path *component* (``<stem>.md``, ``<stem>_assets``, and by MinerU
    as ``<stem>/auto/<stem>.md``), so it must never resolve to ``.`` or ``..``
    or start with a dot -- any of those would let a caller-supplied name walk
    out of the cache directory.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip(".")[:120]
    if not cleaned or cleaned in (".", ".."):
        cleaned = "document"
    return cleaned


def _resolve(source: str, name: str | None) -> tuple[bool, str, str, Path | None]:
    """Return (is_url, source_id, source_hash, local_path)."""
    if source.startswith(("http://", "https://")):
        _stem = _safe_stem(name) if name else _safe_stem(  # kept for debugging
            source.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        )
        return True, source, hashlib.sha256(source.encode()).hexdigest(), None

    path = Path(os.path.expanduser(source))
    if not path.is_file():
        raise MineruError(f"file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIX:
        raise MineruError(
            f"unsupported file type: {path.suffix} "
            f"(supported: {', '.join(sorted(SUPPORTED_SUFFIX))})"
        )
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise MineruError(f"file is {size / 1e6:.0f} MB — MinerU rejects files over 200 MB")
    return False, str(path), _sha256_file(path), path


def _cache_key(source: str, src_hash: str, opts: dict[str, Any]) -> str:
    raw = json.dumps({"src": source, "hash": src_hash, **opts}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════


def convert_document(
    source: str,
    *,
    out_dir: Path | None = None,
    name: str | None = None,
    model: str | None = None,
    lang: str | None = None,
    pages: str | None = None,
    ocr: bool | None = None,
    formula: bool = True,
    table: bool = True,
    force: bool = False,
    outline: bool = False,
    pages_map: bool = False,
) -> Conversion:
    """Convert any supported document (local path or URL) to Markdown.

    Results are cached beside the Markdown in ``<stem>.meta.json``; the cache
    key covers the source hash *and* every conversion option, so changing
    ``--pages`` or the model produces a fresh conversion rather than silently
    reusing the wrong one.

    Raises:
        MineruError: on any failure — no fallback converter exists.
    """
    if not settings.mineru_token:
        raise MineruError(
            "MINERU_TOKEN is not configured. Get one at https://mineru.net/apiManage "
            "and export MINERU_TOKEN."
        )

    model = model or settings.mineru_model
    lang = lang or settings.mineru_language
    out_dir = Path(out_dir or DOC_CACHE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_url, source_id, src_hash, local_path = _resolve(source, name)
    opts = {
        "model": model, "lang": lang, "pages": pages,
        "ocr": ocr, "formula": formula, "table": table,
    }
    key = _cache_key(source_id, src_hash, opts)

    # `name` comes from the MCP caller and is interpolated straight into output
    # paths (`<stem>.md`, `<stem>_assets`), so it must be sanitised or it can
    # walk out of the cache directory.
    stem = _safe_stem(name) if name else _safe_stem(
        Path(source).stem if not is_url
        else source.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    )
    meta_path = out_dir / f"{stem}.meta.json"

    # Same name, different content → do not clobber; disambiguate by hash.
    if not name and meta_path.is_file():
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")).get("cache_key") != key:
                stem = f"{stem}-{src_hash[:6]}"
                meta_path = out_dir / f"{stem}.meta.json"
        except (json.JSONDecodeError, OSError):
            pass

    md_path = out_dir / f"{stem}.md"

    if not force and md_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
        if meta.get("cache_key") == key:
            text = md_path.read_text(encoding="utf-8", errors="replace")
            meta["cached"] = True
            logger.info("MinerU cache hit: %s", md_path)
            return Conversion(
                md_path=md_path,
                text=text,
                meta=meta,
                assets=Path(meta["assets"]) if meta.get("assets") else None,
                outline=_build_outline(text) if outline or meta.get("outline") else meta.get("outline", []),
                pages_map=meta.get("pages_map", []) if pages_map else [],
            )

    file_entry: dict[str, Any] = {"data_id": key}
    if is_url:
        file_entry["url"] = source_id
    else:
        file_entry["name"] = Path(source_id).name
    if pages:
        file_entry["page_ranges"] = pages

    last: Exception | None = None
    for attempt in (1, 2):
        try:
            with _client() as client:
                batch_id, upload_urls = _create_task(
                    client,
                    files=[file_entry],
                    model=model,
                    lang=lang,
                    formula=formula,
                    table=table,
                    ocr=ocr,
                    url=is_url,
                )
                if not is_url:
                    if not upload_urls:
                        raise MineruError("MinerU returned no upload URL")
                    _put_file(upload_urls[0], local_path)
                entry = _poll(client, batch_id)
                zip_url = entry.get("full_zip_url") or ""
                if not zip_url:
                    raise MineruError("MinerU reported done but returned no result URL")
            blob = _download(zip_url)
            break
        except _Transient as exc:
            last = exc
            logger.warning("MinerU transient failure (attempt %d): %s", attempt, exc)
            if attempt == 2:
                raise MineruError(f"MinerU conversion failed: {exc}") from exc
            time.sleep(3)
        except MineruError:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            logger.warning("MinerU error (attempt %d): %s", attempt, exc)
            if attempt == 2:
                raise MineruError(f"MinerU conversion failed: {exc}") from exc
            time.sleep(3)
    else:  # pragma: no cover - loop always breaks or raises
        raise MineruError(f"MinerU conversion failed: {last}")

    md_path, assets = _extract_zip(blob, out_dir, stem)
    text = md_path.read_text(encoding="utf-8", errors="replace")
    content_list = _load_content_list(assets)

    meta = {
        "source": source_id,
        "source_hash": src_hash,
        "cache_key": key,
        "model": model,
        "lang": lang,
        "pages": pages,
        "ocr": ocr,
        "formula": formula,
        "table": table,
        "md": str(md_path.resolve()),
        "assets": str(assets.resolve()),
        "lines": text.count("\n") + 1,
        "chars": len(text),
        "images": len(_IMAGE_LINK.findall(text)),
        "converted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cached": False,
    }
    if outline:
        meta["outline"] = _build_outline(text)
    if pages_map and content_list:
        meta["pages_map"] = _build_pages_map(text, content_list)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return Conversion(
        md_path=md_path,
        text=text,
        meta=meta,
        assets=assets,
        outline=meta.get("outline", []),
        pages_map=meta.get("pages_map", []),
    )


def convert_paper_pdf(pdf_path: str | Path, key: str, *, attempts: int = 2) -> str:
    """Convert a downloaded paper PDF to cleaned Markdown (text only).

    Used by the paper pipeline. The Markdown is cached under
    ``data/texts/<key>.md`` so repeated reads cost nothing.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise MineruError(f"PDF not found: {path}")

    cached = storage.read_md(key)
    if cached:
        return cached

    if not settings.mineru_token:
        raise MineruError(
            "MINERU_TOKEN is not configured — PDF→Markdown conversion is unavailable. "
            "Get a token at https://mineru.net/apiManage and export MINERU_TOKEN."
        )

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = convert_document(
                str(path),
                out_dir=settings.md_dir,
                name=key,
                lang=settings.mineru_language,
                force=True,  # md cache is checked above; never reuse a stale one
            )
            text = clean_markdown(result.text)
            storage.write_md(key, text)
            return text
        except MineruError:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            logger.warning("MinerU paper conversion failed (attempt %d): %s", attempt, exc)
            if attempt < attempts:
                time.sleep(3)
    raise MineruError(f"MinerU conversion failed: {last}")


def migrate_legacy_cache() -> None:
    """Adopt an older cache directory if one is present.

    Conversions cost MinerU quota, so throwing away an existing cache on a
    rename would be a real (if small) waste. Checked in order: the pi-era
    ``~/.cache/pi-doc-read`` and the even older ``~/.cache/pi-pdf-read``.
    """
    target = DOC_CACHE_DIR
    if target.exists():
        return
    for name in ("pi-doc-read", "pi-pdf-read"):
        legacy = Path(os.path.expanduser(f"~/.cache/{name}"))
        if legacy.is_dir():
            try:
                legacy.rename(target)
                logger.info("Migrated legacy cache %s → %s", legacy, target)
            except OSError as exc:
                logger.debug("Could not migrate legacy cache: %s", exc)
            return
