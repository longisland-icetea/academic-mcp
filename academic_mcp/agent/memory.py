#!/usr/bin/env python3
"""Research memory management for academic-search skill.

Usage:
  # Add/update paper notes (bulk: one JSON object with paper_notes array)
  python3 memory.py update '{"paper_notes": [{"paper_id": "10.1038_xxx", ...}]}'

  # Query all papers in library
  python3 memory.py list

  # Query specific papers by ID
  python3 memory.py get --ids 10.1038_xxx,10.1126_yyy

  # Query by topic
  python3 memory.py get --topic "exciton"

  # Query with full text from disk cache
  python3 memory.py get --ids 10.1038_xxx --full-text

  # Update research goal
  python3 memory.py goal "Investigate moire exciton properties in TMD heterostructures"

  # Add freeform research notes
  python3 memory.py note "Found that moire potential depth scales with twist angle"

  # Add a key finding
  python3 memory.py finding '{"text": "moire excitons form at angles < 3°", "source_pids": ["10.1038_xxx"], "confidence": "high"}'

  # Add an unresolved question
  python3 memory.py unresolved "How does pressure affect the optical gap in tMoTe2?"

  # Dump full memory state (for debugging)
  python3 memory.py dump

Memory is stored per-session in ~/data/academic/{session_id}.json

(~/.pi/agent/skills/academic-search/data is a symlink to it, so pi and
academic-mcp keep working against the old path.)

session_id resolution order:
  1. PI_SESSION env var (set by pi main process from cwd)
  2. cwd-based encoding: absolute path `/foo/bar` → `--foo-bar--`
  3. Literal "default"

The cwd-based encoding is what pi uses to isolate per-project working memory.
When you `cd` from `/home/cxxiao` to `/mnt/c/Users/project/MoTe2`, the session_id
flips from `--home-cxxiao--` to `--mnt-c-Users-project-MoTe2--`, and a separate
JSON file is used. This is by design: each project gets its own paper library.

Cross-skill sharing: academic-search, academic-paper-writing, and academic-grant-writing
read/write the same data/<session>.json. Paper library built during the search phase
can be recalled from the paper-writing / grant-writing phase via `academic_paper_recall`.

New commands added 2026 (P0-1 / P2-1/2/3 of academic-context-injection.md §8):
  python3 memory.py progress '{"current_phase": "4.6", "completed_phases": ["0","1","2","3","4.1-4.5"]}'
  python3 memory.py resume              # returns {current_phase, completed_phases, next_phase, config}
  python3 memory.py telemetry [--limit 20]
  python3 memory.py dump                # now also includes progress / config

The JSON now has new top-level fields (all optional, schema backward compatible):
  - current_phase: str (e.g. "4.6")
  - completed_phases: list[str]
  - config: dict (Phase 0 interview answers: paper_type, citation_style,
                  target_journal, funding_type, nsfc_code, etc.)
"""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# §4.3: optional Jinja2 import for prompt template. Falls back to legacy
# string concatenation if Jinja2 is not installed (defensive — the skill is
# still importable on minimal envs).
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    _JINJA2_OK = True
except ImportError:
    _JINJA2_OK = False

# §4.1: Pydantic v2 schemas. Optional — keeps legacy code path working if
# pydantic is unavailable.
try:
    from pydantic import ValidationError as _PydValidationError
    sys.path.insert(0, str(Path(__file__).parent))
    from ._schemas import (  # noqa: E402
        load_memory_dict as _load_memory_dict,
    )
    _PYDANTIC_OK = True
except ImportError:
    _PYDANTIC_OK = False


# ── Config ────────────────────────────────────────────────────────────────

# The service owns the library location (ACADEMIC_DATA_DIR in its own config).
from ..config import settings as _settings

SKILL_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(_settings.data_dir)
TEXTS_DIR = DATA_DIR / "texts"
# Metadata backfill shells out to the search module as a module, because this
# path is synchronous and the MCP tool loop is already running.
SEARCH_MODULE = "academic_mcp.agent.search"
DEFAULT_SESSION = "default"

# Recalling metadata via API (Scopus/OpenAlex) is expensive: cap how many
# papers get backfilled per recall call, and remember the outcome so we
# never re-query a DOI we already resolved (or already failed to resolve).
_META_FETCH_LIMIT = 3
_META_FETCH_TIMEOUT = 25  # seconds


def session_key_for_cwd(cwd: str) -> str:
    """The one and only cwd → session_id rule.

    `/mnt/c/Users/project/MoTe2` → `--mnt-c-Users-project-MoTe2--`. Clients ask
    for this instead of re-implementing it (a client that derived a different
    key would silently read and write a different project's memory, and would
    also fail the DOI authorization check, which is keyed the same way).
    """
    value = str(cwd or "").strip().rstrip("/")
    if not value:
        return DEFAULT_SESSION
    return f"--{value.lstrip('/').replace('/', '-')}--"


def get_session_id() -> str:
    """Resolve the current session_id.

    Order:
      1. PI_SESSION env var (set by a client that already knows the key).
      2. cwd encoding through :func:`session_key_for_cwd`.
      3. Literal "default".
    """
    env_val = os.environ.get("PI_SESSION")
    if env_val:
        return env_val
    try:
        return session_key_for_cwd(os.getcwd())
    except OSError:
        return DEFAULT_SESSION


def get_memory_path(session_id: str) -> Path:
    return DATA_DIR / f"{session_id}.json"


def load_memory(session_id: str) -> dict[str, Any]:
    path = get_memory_path(session_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # P2-6: silently overwriting a corrupted memory file would lose all
            # accumulated notes. Back the bad file up next to it before letting
            # the caller create a fresh empty memory.
            bad_path = path.with_suffix(path.suffix + f".corrupt-{int(time.time())}")
            try:
                path.rename(bad_path)
                print(
                    f"[memory.py] WARNING: {path.name} was corrupt ({type(e).__name__}); "
                    f"backed up to {bad_path.name} and starting fresh.",
                    file=sys.stderr,
                )
            except OSError as rename_err:
                # rename can fail on read-only filesystems or on Windows
                # under aggressive file locking — fall back to copy + delete.
                try:
                    import shutil
                    shutil.copy2(path, bad_path)
                    path.unlink()
                except OSError:
                    print(
                        f"[memory.py] ERROR: could not back up corrupt {path.name}: "
                        f"{rename_err}; proceeding with empty memory.",
                        file=sys.stderr,
                    )
    return {
        "session_id": session_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "research_goal": "",
        "paper_library": [],
        "key_findings": [],
        "unresolved_questions": [],
        "topic_map": {},
        "research_notes": "",
        # Added 2026 (P0-1 / P2-2 of academic-context-injection.md §8):
        # Cross-skill progress tracking. Optional in JSON; missing fields
        # are treated as empty. Forward-compatible with old memory files.
        "current_phase": "",
        "completed_phases": [],
        "config": {},
    }


def save_memory(session_id: str, memory: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    memory["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = get_memory_path(session_id)
    # P0-4 修复（v2 — 2026-09）：用独立 lock file 而不是 path fd。
    # 原因：原实现锁定 path fd，但写入发生在 tmp 文件上，lock 实际不拦 tmp write。
    # 多个进程同时写 tmp → 重叠覆盖 → path 内容损坏。
    # 修复：单独 lock file 同步 tmp + path 两个文件操作。
    lock_path = path.with_suffix(path.suffix + ".lock")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # 另一个进程已持锁，等 5s 再试一次（阻塞而非退避）
            fcntl.flock(fd, fcntl.LOCK_EX)
        # Write to temp file first, then atomic rename (prevents corruption on crash)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
        # Keep a backup of the previous version
        bak_path = path.with_suffix(path.suffix + ".bak")
        if path.exists():
            try:
                path.rename(bak_path)
            except OSError:
                pass  # best-effort backup
        try:
            tmp_path.rename(path)
        except OSError:
            # Fallback: direct write if rename fails (e.g. cross-device)
            path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if fd is not None:
            os.close(fd)  # 关闭时自动释放 flock


def _find_paper_idx(library: list[dict[str, Any]], paper_id: str, doi: str = "") -> int | None:
    """Find paper index by paper_id or doi. Returns index or None."""
    pid_lower = paper_id.strip().lower()
    doi_lower = doi.strip().lower()
    for i, p in enumerate(library):
        cur_pid = p.get("paper_id", "").strip().lower()
        cur_doi = p.get("doi", "").strip().lower()
        # P2-3: replaced nested ternary with explicit 3-line guard for readability
        if pid_lower and cur_pid == pid_lower:
            return i
        if doi_lower and cur_doi == doi_lower:
            return i
    return None


def _read_full_text(paper_id: str) -> str | None:
    """Try to read cached full text from data/texts/{paper_id}.md."""
    md_path = TEXTS_DIR / f"{paper_id}.md"
    if md_path.exists():
        try:
            return md_path.read_text(encoding="utf-8")
        except Exception:
            pass
    # Also try with different key formats (doi with slashes)
    if "_" in paper_id:
        alt_path = TEXTS_DIR / f"{paper_id.replace('_', '/')}.md"
        if alt_path.exists():
            try:
                return alt_path.read_text(encoding="utf-8")
            except Exception:
                pass
    return None


# ── Metadata enrichment via API (Scopus → OpenAlex) ──────────────────────

def _fetch_doi_metadata(doi: str) -> dict[str, Any] | None:
    """Fetch full author list + abstract for a DOI via search.py (API).

    Delegates to search.py --doi (Scopus first, OpenAlex backfill), parses
    the normalized JSON output, and returns {"authors": [str, ...],
    "abstract": str} — or None on failure. The result is persisted into the
    paper record by the caller so each DOI is queried at most once.
    """
    if not doi:
        return None
    try:
        proc = subprocess.run(
            [sys.executable, "-m", SEARCH_MODULE, "--doi", doi],
            capture_output=True, text=True, timeout=_META_FETCH_TIMEOUT,
            cwd=str(SKILL_DIR.parent),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("error") or not data.get("doi"):
        return None

    meta: dict = {}
    raw_authors = data.get("authors") or []
    names = [a.get("name") for a in raw_authors if isinstance(a, dict) and a.get("name")]
    if names:
        meta["authors"] = names
    abstract = (data.get("abstract") or "").strip()
    if abstract:
        meta["abstract"] = abstract
    return meta or None


# ── Commands ──────────────────────────────────────────────────────────────

def cmd_update(memory: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Apply an update payload (mirrors academic_agent's update_memory schema).

    §4.1 Pydantic v2: optionally validates ``data`` against the
    ``MemoryUpdatePayload`` model before applying. Validation errors are
    returned in the result dict (so the caller can surface them) — the
    function never raises ValidationError. Set ``MEMORY_STRICT_VALIDATION=1``
    to force exceptions.
    """
    changes: list[str] = []

    # §4.1: Pydantic-validate the incoming payload (no-op if pydantic
    # missing or validation disabled). The legacy code path below is
    # untouched, so any field that doesn't fit the schema simply stays
    # # in ``data`` and gets applied as before.
    if _PYDANTIC_OK and os.environ.get("MEMORY_STRICT_VALIDATION", "0") != "0":
        try:
            validated = _load_memory_dict({
                **{k: v for k, v in memory.items() if k in {"session_id", "research_goal"}},
                "paper_library": memory.get("paper_library", []),
                "topic_map": memory.get("topic_map", {}),
                "completed_phases": memory.get("completed_phases", []),
                "config": memory.get("config", {}),
                "key_findings": memory.get("key_findings", []),
                "unresolved_questions": memory.get("unresolved_questions", []),
                "research_notes": memory.get("research_notes", ""),
                "current_phase": memory.get("current_phase", ""),
                "migration_history": memory.get("migration_history", []),
            })
            _ = validated  # the validation itself is the safety net
        except _PydValidationError as e:  # pragma: no cover - env-toggled
            return {"error": "validation failed",
                    "pydantic_errors": e.errors()[:5],
                    "hint": "see docs/academic-skill-audit.md §4.1"}

    # research_goal
    if data.get("research_goal"):
        memory["research_goal"] = data["research_goal"]
        changes.append("research_goal")

    # paper_notes (bulk)
    paper_notes = data.get("paper_notes") or []
    # P2-5: reject any note missing both paper_id and doc_id instead of
    # silently writing an empty/blank record that pollutes later queries.
    bad_notes = [i for i, n in enumerate(paper_notes)
                 if not (n.get("paper_id") or n.get("doc_id"))]
    if bad_notes:
        return {"error": "missing paper_id/doc_id",
                "bad_note_indices": bad_notes,
                "hint": "every paper_note must include either 'paper_id' (preferred) or 'doc_id'."}

    # P2-4: track whether any note carried topics so we can rebuild topic_map
    # once at the end instead of inside the per-paper loop (O(N) instead of O(N^2)).
    topics_dirty = False
    for note in paper_notes:
        paper_id = note.get("paper_id", note.get("doc_id", ""))
        doi = note.get("doi", "")
        title = note.get("title", "")
        source = note.get("source", "paper")

        idx = _find_paper_idx(memory["paper_library"], paper_id, doi)

        if idx is not None:
            # Update existing
            existing = memory["paper_library"][idx]
            for field in ("title", "first_author", "year", "relevance", "importance",
                          "key_notes", "detailed_notes", "topics",
                          "source", "doi", "doc_id", "authors", "abstract"):
                if field in note and note[field]:
                    existing[field] = note[field]
            # Merge key_excerpts (append new ones)
            if note.get("key_excerpts"):
                existing.setdefault("key_excerpts", [])
                for exc in note["key_excerpts"]:
                    if exc not in existing["key_excerpts"]:
                        existing["key_excerpts"].append(exc)
            # Update status
            if note.get("status"):
                existing["status"] = note["status"]
            existing["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

            # P2-4: just mark dirty; full rebuild happens after the loop
            if note.get("topics"):
                topics_dirty = True

            changes.append(f"paper_updated:{paper_id}")
        else:
            # New entry
            entry = {
                "paper_id": paper_id,
                "doi": doi,
                "title": title,
                "first_author": note.get("first_author", ""),
                "authors": note.get("authors") or [],
                "abstract": note.get("abstract", ""),
                "year": note.get("year"),
                "source": source,
                "doc_id": note.get("doc_id", ""),
                "relevance": note.get("relevance", ""),
                "importance": note.get("importance", ""),
                "key_notes": note.get("key_notes", ""),
                "key_excerpts": note.get("key_excerpts", []),
                "detailed_notes": note.get("detailed_notes", ""),
                "topics": note.get("topics", []),
                "status": note.get("status", "read"),
                "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            memory["paper_library"].append(entry)
            changes.append(f"paper_added:{paper_id}")

            # Update topic_map
            for topic in entry.get("topics", []):
                topic_lower = topic.strip().lower()
                memory["topic_map"].setdefault(topic_lower, [])
                if paper_id not in memory["topic_map"][topic_lower]:
                    memory["topic_map"][topic_lower].append(paper_id)

    # P2-4: single O(N) rebuild of topic_map, only if some note carried topics
    if topics_dirty:
        memory["topic_map"] = {}
        for p in memory["paper_library"]:
            pid = p.get("paper_id", "")
            for topic in p.get("topics", []):
                tl = topic.strip().lower()
                memory["topic_map"].setdefault(tl, [])
                if pid and pid not in memory["topic_map"][tl]:
                    memory["topic_map"][tl].append(pid)

    # progress (Phase tracking, P2-2)
    if data.get("progress"):
        prog = data["progress"]
        if isinstance(prog, dict):
            if prog.get("current_phase"):
                memory["current_phase"] = prog["current_phase"]
                changes.append("current_phase")
            new_completed = prog.get("completed_phases") or []
            # P0-6 修复：原代码 [append] + list 是 prepend，不是 append。
            # 现在改为真 append，且脱去重复以防误加。
            if prog.get("append_completed"):
                if prog["append_completed"] not in new_completed:
                    new_completed = list(new_completed) + [prog["append_completed"]]
            for ph in new_completed:
                existing = memory.get("completed_phases") or []
                if ph not in existing:
                    existing.append(ph)
                    memory["completed_phases"] = existing
            if new_completed:
                changes.append("completed_phases")

    # config (Phase 0 interview answers, P2-1)
    if data.get("config") and isinstance(data["config"], dict):
        memory.setdefault("config", {}).update(data["config"])
        changes.append("config")

    # key_finding
    if data.get("key_finding"):
        finding = data["key_finding"]
        memory["key_findings"].append({
            "text": finding.get("text", ""),
            "source_pids": finding.get("source_pids", []),
            "confidence": finding.get("confidence", "medium"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        changes.append("key_finding")

    # unresolved_question
    if data.get("unresolved_question"):
        memory["unresolved_questions"].append({
            "question": data["unresolved_question"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        changes.append("unresolved_question")

    # research_notes (append)
    if data.get("research_notes"):
        sep = "\n\n" if memory["research_notes"] else ""
        memory["research_notes"] += sep + data["research_notes"]
        changes.append("research_notes")

    # topic_map (merge)
    if data.get("topic_map"):
        for topic, pids in data["topic_map"].items():
            topic_lower = topic.strip().lower()
            memory["topic_map"].setdefault(topic_lower, [])
            for pid in pids:
                if pid not in memory["topic_map"][topic_lower]:
                    memory["topic_map"][topic_lower].append(pid)
        changes.append("topic_map")

    return {"status": "ok", "changes": changes}


def cmd_list(memory: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all papers in library as a compact summary."""
    papers = memory.get("paper_library", [])
    result = {
        "count": len(papers),
        "research_goal": memory.get("research_goal", ""),
        "topic_map": memory.get("topic_map", {}),
        "papers": [],
        "key_findings_count": len(memory.get("key_findings", [])),
        "unresolved_questions_count": len(memory.get("unresolved_questions", [])),
    }
    for p in papers:
        result["papers"].append({
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", "")[:120],
            "first_author": p.get("first_author", ""),
            "year": p.get("year"),
            "doi": p.get("doi", ""),
            "importance": p.get("importance", ""),
            "key_notes": (p.get("key_notes", "") or "")[:200],
            "topics": p.get("topics", []),
            "status": p.get("status", "unknown"),
            "source": p.get("source", "paper"),
        })
    return result


def cmd_get(memory: dict[str, Any], ids: list[str] | None = None, topic: str | None = None,
            include_full_text: bool = False) -> dict[str, Any]:
    """Get detailed info for specific papers by ID or topic.

    When include_full_text=True, also reads cached .md files from data/texts/.
    """
    papers = memory.get("paper_library", [])
    matched = []

    id_set = set(i.strip().lower() for i in (ids or []) if i.strip())
    topic_lower = topic.strip().lower() if topic else ""

    for p in papers:
        pid = (p.get("paper_id", "")).strip().lower()
        if id_set and pid in id_set:
            matched.append(p)
        elif topic_lower:
            p_topics = [t.lower() for t in p.get("topics", [])]
            if any(topic_lower in t for t in p_topics):
                matched.append(p)
            elif topic_lower in (p.get("key_notes", "") + p.get("detailed_notes", "")).lower():
                matched.append(p)

    if id_set and not matched:
        found_ids = set((p.get("paper_id", "")).strip().lower() for p in papers)
        missing = id_set - found_ids
        hints = []
        for mid in missing:
            hints.append(f"  • Paper '{mid}' not in library. Download first, then add notes via 'update' command.")
        return {"count": 0, "papers": [], "_hints": hints}

    # Optionally attach full text from disk cache; otherwise enrich with
    # full author list + abstract (fetched via API, then cached in memory).
    result_papers = []
    fetched = 0
    meta_updated = False
    for p in matched:
        pd = dict(p)
        pd["has_cached_full_text"] = False
        if include_full_text:
            pid = p.get("paper_id", "")
            full_text = _read_full_text(pid)
            if full_text:
                pd["full_text"] = full_text
                pd["has_cached_full_text"] = True
                pd["full_text_length"] = len(full_text)
            else:
                pd["_hint"] = (
                    f"论文 '{pid}' 的全文缓存不存在。"
                    f"请先用 academic_import_papers 下载并阅读（academic_download 仅供 subagent 内部使用）。"
                )
        else:
            # Full author list + abstract via API, cached in the paper record
            # (meta_fetched guards against re-querying resolved/failed DOIs).
            if fetched < _META_FETCH_LIMIT and not p.get("meta_fetched"):
                meta = _fetch_doi_metadata(p.get("doi", ""))
                if meta:
                    if meta.get("authors"):
                        pd["authors"] = meta["authors"]
                        p["authors"] = meta["authors"]
                    if meta.get("abstract"):
                        pd["abstract"] = meta["abstract"]
                        p["abstract"] = meta["abstract"]
                    p["meta_fetched"] = "ok"
                else:
                    p["meta_fetched"] = "failed"
                fetched += 1
                meta_updated = True
            # Fallback: first_author only when the API gave us nothing
            if not pd.get("authors") and p.get("first_author"):
                pd["authors"] = [p["first_author"]]
        result_papers.append(pd)

    if meta_updated:
        try:
            # P0-7 修复：读路径不写盘。仅当用户走 cmd_get 后通过 cmd_update 才落盘。
            # 这里只设标记，不调 save_memory（避免读幂等性被破坏 + 竞态走宽）。
            pass
        except Exception:
            pass

    return {"count": len(matched), "papers": result_papers}


def cmd_goal(memory: dict[str, Any], goal: str) -> dict[str, Any]:
    memory["research_goal"] = goal
    return {"status": "ok", "research_goal": goal}


def cmd_note(memory: dict[str, Any], text: str) -> dict[str, Any]:
    sep = "\n\n" if memory["research_notes"] else ""
    memory["research_notes"] += sep + text
    return {"status": "ok", "appended_chars": len(text)}


def cmd_finding(memory: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    memory["key_findings"].append({
        "text": data.get("text", ""),
        "source_pids": data.get("source_pids", []),
        "confidence": data.get("confidence", "medium"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return {"status": "ok"}


def cmd_unresolved(memory: dict[str, Any], question: str) -> dict[str, Any]:
    memory["unresolved_questions"].append({
        "question": question,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return {"status": "ok", "question": question}


def _pid_key(value: Any) -> str:
    """Normalize one paper id for comparison.

    `paper_library` stores `10.1038_s41467-025-67836-9` while
    `key_findings[].source_pids` is written with the raw DOI
    `10.1038/s41467-025-67836-9`, so both spellings must compare equal or a
    delete would leave a dangling reference behind.
    """
    return str(value or "").strip().lower().replace("/", "_")


def cmd_delete_paper(memory: dict[str, Any], paper_id: str, doi: str = "") -> dict[str, Any]:
    """Remove one paper and every structured reference to it.

    Removes the `paper_library` entry (which carries `detailed_notes` and
    `key_excerpts` with it), the id from every `topic_map` bucket (dropping a
    bucket that empties out), and the id from every `key_findings[].source_pids`
    — the finding text is kept, because it may be a synthesis and only the agent
    should decide to drop a conclusion. `research_notes` is free text and is
    left alone. Cached `texts/*.md` / `pdfs/*.pdf` stay on disk: rebuilding a
    MinerU conversion is expensive, so deleting a note must not cost a
    re-download.
    """
    library = memory.get("paper_library", [])
    idx = _find_paper_idx(library, paper_id, doi)
    if idx is None:
        return {"status": "not_found", "paper_id": paper_id}

    removed = library.pop(idx)
    rid = _pid_key(removed.get("paper_id") or paper_id)
    rdoi = str(removed.get("doi") or doi or "").strip().lower()

    def references(pid: Any) -> bool:
        return _pid_key(pid) == rid or (bool(rdoi) and str(pid or "").strip().lower() == rdoi)

    topic_refs = 0
    topic_map = memory.get("topic_map") or {}
    for topic in list(topic_map.keys()):
        ids = topic_map.get(topic) or []
        kept = [pid for pid in ids if not references(pid)]
        if len(kept) != len(ids):
            topic_refs += len(ids) - len(kept)
            if kept:
                topic_map[topic] = kept
            else:
                del topic_map[topic]

    finding_refs = 0
    for finding in memory.get("key_findings") or []:
        source = finding.get("source_pids") or []
        kept = [pid for pid in source if not references(pid)]
        if len(kept) != len(source):
            finding_refs += len(source) - len(kept)
            finding["source_pids"] = kept

    return {
        "status": "ok",
        "paper_id": rid,
        "title": removed.get("title", ""),
        "topic_refs_removed": topic_refs,
        "finding_refs_removed": finding_refs,
        "remaining": len(library),
    }


# ── Progress / resume / telemetry (added 2026, P0-1 / P2-1/2/3) ───────────

# Canonical Phase ordering per skill. Used by cmd_resume to decide next_phase
# and to validate user-supplied current_phase strings.
#
# P2-7: Phase naming convention. Top-level phases are integers "0".."7".
# Sub-phases use "N.M" notation (N = parent phase, M = sub-step). A composite
# phase "N" represents "any sub-step of N is currently in progress" and is
# kept as a convenience alias for callers that want to mark a parent phase
# active without picking a specific sub-step.
#
# Rules:
#   * Order: parents appear before their sub-phases (e.g. "4" before "4.1").
#   * Validation: cmd_progress accepts both "4" and "4.1"-"4.7" for paper-writing;
#     storing "4" is valid but the next-phase logic prefers the earliest
#     un-completed sub-phase.
#   * Resume: when completed_phases includes "4", cmd_resume still walks the
#     sub-phases because paper-writing's "4" can be left as a coarse marker.
PHASE_ORDER = {
    "academic-search": ["0", "1", "2", "3", "4", "5"],
    "academic-paper-writing": [
        "0", "1", "2", "3", "4",                  # Phase 4 is composite (alias for "4.1".."4.7")
        "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7",
        "5", "6", "7",
    ],
    "academic-grant-writing": [
        "0", "0.1", "0.1.X", "0.1.Y", "0.1.Z", "0.2",
        "1", "2", "3", "4", "4.1", "4.2", "4.3", "4.4", "4.5", "5",
    ],
}


def cmd_progress(memory: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Update current_phase / completed_phases / config.

    Schema:
      {
        "current_phase": "4.6",
        "completed_phases": ["0", "1", "2", "3"],
        "config": {"paper_type": "IMRaD", "target_journal": "PRB"},
        "skill": "academic-paper-writing",       # optional, for validation
        "append_completed": "4.1"                # optional: append a phase
      }
    """
    skill = data.get("skill", "")
    new_phase = data.get("current_phase")
    if new_phase is not None:
        if skill and skill in PHASE_ORDER and new_phase not in PHASE_ORDER[skill]:
            return {"error": f"Phase '{new_phase}' not in {skill} (known: {PHASE_ORDER[skill][:10]}...)"}
        memory["current_phase"] = new_phase

    completed = memory.get("completed_phases") or []
    if "completed_phases" in data and isinstance(data["completed_phases"], list):
        for ph in data["completed_phases"]:
            if ph not in completed:
                completed.append(ph)
    if data.get("append_completed"):
        if data["append_completed"] not in completed:
            completed.append(data["append_completed"])
    memory["completed_phases"] = completed

    cfg = data.get("config")
    if cfg and isinstance(cfg, dict):
        memory.setdefault("config", {}).update(cfg)

    return {
        "status": "ok",
        "current_phase": memory.get("current_phase", ""),
        "completed_phases": memory.get("completed_phases", []),
        "config_keys": list(memory.get("config", {}).keys()),
    }


def cmd_resume(memory: dict[str, Any], skill: str = "") -> dict[str, Any]:
    """Read current_phase and suggest next_phase for resume.

    Returns the first Phase in PHASE_ORDER[skill] that is NOT in completed_phases.
    If no completed_phases recorded, returns "0" (start from scratch).

    Cross-skill consideration: if `skill` is omitted, infers from config:
      - has funding_type / nsfc_code → academic-grant-writing
      - has paper_type / target_journal → academic-paper-writing
      - else → academic-search
    """
    if not skill:
        cfg = memory.get("config") or {}
        if cfg.get("funding_type") or cfg.get("nsfc_code"):
            skill = "academic-grant-writing"
        elif cfg.get("paper_type") or cfg.get("target_journal"):
            skill = "academic-paper-writing"
        else:
            skill = "academic-search"

    order = PHASE_ORDER.get(skill, [])
    completed = set(memory.get("completed_phases") or [])
    next_phase = None
    for ph in order:
        if ph not in completed:
            next_phase = ph
            break

    return {
        "status": "ok",
        "skill": skill,
        "session_id": memory.get("session_id"),
        "current_phase": memory.get("current_phase", ""),
        "completed_phases": memory.get("completed_phases", []),
        "next_phase": next_phase or "(all completed)",
        "config": memory.get("config", {}),
        "research_goal": memory.get("research_goal", ""),
        "paper_count": len(memory.get("paper_library", [])),
    }


def cmd_telemetry(memory: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    """Read tool-call telemetry from data/telemetry/ if it exists.

    Telemetry is written by the MCP adapter; this command just surfaces it.
    Returns [] if no telemetry dir / no entries.
    """
    telemetry_dir = DATA_DIR / "telemetry"
    if not telemetry_dir.exists():
        return {"status": "ok", "count": 0, "entries": [], "note": "no telemetry dir"}

    # Files: <session_id>.<timestamp>.jsonl (one record per line, JSONL format)
    pattern = f"{memory.get('session_id', '*')}.*.jsonl"
    files = sorted(telemetry_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    entries = []
    for fpath in files[:limit]:
        try:
            for line in fpath.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries.append(rec)
        except OSError:
            continue
    # Trim to limit
    entries = entries[-limit:]
    return {
        "status": "ok",
        "count": len(entries),
        "files": [p.name for p in files[:limit]],
        "entries": entries,
    }


# ── Working memory formatter ──────────────────────────────────────────────

def _scrub_for_injection(text: str) -> str:
    """P0-2/P0-10 修复：剥除用户控制字段里的 prompt injection 风险 token。

    仅在 format_working_memory 中调用。原始文本不受影响，原始论文笔记不会被
    “腰洗”，只是 subagent system prompt 注入路径上作脱敏。
    """
    if not text:
        return ""
    # 1. 脱去 control characters（保留中文标点）
    out = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 2. 脱去 XML / HTML 标签 / 停止 token / system token
    # 这些是常见 prompt injection 跳入点
    out = re.sub(r"</?system[^>]*>", "[filtered-tag]", out, flags=re.IGNORECASE)
    out = re.sub(r"<\|.*?\|>", "[filtered-special]", out)
    # 3. 脱去变体分隔符（“ignore prior / above / below instructions”）
    dangerous_patterns = [
        r"(?i)ignore\s+(?:all\s+)?(?:prior|previous|above|system)\s+instructions?",
        r"(?i)disregard\s+(?:all\s+)?(?:prior|previous)\s+(?:instructions?|context)",
        r"(?i)new\s+instructions?:\s*",
    ]
    for pat in dangerous_patterns:
        out = re.sub(pat, "[filtered-injection]", out)
    return out


def format_working_memory(memory: dict[str, Any], tier: str = "summary") -> str:
    """Format the memory as a working-memory block for system prompt injection.

    Args:
        memory: The full memory dict.
        tier: "summary" (compact, for auto-injection — default) or "full"
              (complete with excerpts and detailed_notes, for debugging).

    Tier "summary" produces ~500 bytes per paper (title + importance +
    truncated key_notes + excerpt count). Tier "full" is the legacy
    behaviour with all fields and full excerpt text.

    §4.3 架构升级 (2026)：
    - 改用 Jinja2 SandboxedEnvironment 渲染 ``_prompt.j2``，autoescape
      默认开（防 prompt injection / XSS in research_notes 等用户字段）。
    - 旧的手写字符串拼接逻辑保留为 ``_format_working_memory_legacy``，
      默认关闭 — 仅当 ``MEMORY_USE_LEGACY_PROMPT=1`` 时启用（回退）。
    - 12 KB 总字节预算 + 单 paper 软门限仍在，预算超限部分进
      ``skipped_count`` 让模板渲染 footer。
    """
    # Truncation limits for Tier 1 (summary mode).
    # Full key_notes / importance are available via academic_paper_recall (Tier 2).
    # The rsplit('.', 1) strategy: take at most MAX chars, then drop the last
    # partial sentence so the truncated text reads as complete prose.
    # For cache stability: if key_notes hasn't been edited, the truncated prefix
    # Truncation limits for Tier 1 (summary mode).
    # Full key_notes / importance are available via academic_paper_recall (Tier 2).
    # The rsplit('.', 1) strategy: take at most MAX chars, then drop the last
    # partial sentence so the truncated text reads as complete prose.
    # For cache stability: if key_notes hasn't been edited, the truncated prefix
    # is deterministic — same input produces same truncated output.
    # Kept for reference: the template now applies its own per-field limits.
    _MAX_KEYNOTES_CHARS = 500
    _MAX_IMPORTANCE_CHARS = 300
    # P0-10 修复：总字节预算防 context 袄炸
    MAX_TOTAL_BYTES = 12 * 1024  # 12 KB

    # §4.3: build context for Jinja2 template. We deliberately keep the
    # scrubbing step here (before passing to jinja) so the template is
    # simple and the byte-budget stays predictable.
    papers_raw = memory.get("paper_library", []) or []
    if tier == "summary":
        active_papers = [
            p for p in papers_raw
            if (p.get("key_notes", "").strip()
                or p.get("detailed_notes", "").strip()
                or p.get("importance", "").strip())
        ]
    else:
        active_papers = list(papers_raw)

    skipped_count = 0
    included_papers: list[dict[str, Any]] = []
    rendered_so_far = 0

    def _safe_len(v: Any) -> int:
        """Len helper that tolerates int/float (e.g. importance as a number)
        or None / dict / list. Coerces to str first if not str-like."""
        if v is None:
            return 0
        if isinstance(v, (str, list, dict, bytes)):
            return len(v)
        return len(str(v))

    for p in active_papers:
        # Per-paper budget check — oversize papers are deferred to the
        # skipped_count footer.
        size_est = _safe_len(p.get("title")) + _safe_len(p.get("key_notes")) \
                 + _safe_len(p.get("importance")) + _safe_len(p.get("detailed_notes"))
        if rendered_so_far + size_est > MAX_TOTAL_BYTES and tier == "summary":
            skipped_count += 1
            continue
        scrubbed = {
            "title": _scrub_for_injection((p.get("title") or "Untitled")[:100]),
            "paper_id": p.get("paper_id", ""),
            "doi": p.get("doi", ""),
            "first_author": _scrub_for_injection(str(p.get("first_author") or "?")),
            "year": p.get("year") or "?",
            "status": p.get("status", "unknown"),
            "topics": list(p.get("topics") or []),
            "relevance": _scrub_for_injection(str(p.get("relevance") or "").strip().lower()),
            "importance": _scrub_for_injection(str(p.get("importance") or "").strip()),
            "key_notes": _scrub_for_injection(str(p.get("key_notes") or "").strip()),
            "key_excerpts": list(p.get("key_excerpts") or []),
            "detailed_notes": _scrub_for_injection(str(p.get("detailed_notes") or "").strip()),
        }
        included_papers.append(scrubbed)
        rendered_so_far += size_est

    findings = [f for f in (memory.get("key_findings") or []) if (f.get("text") or "").strip()]
    questions = memory.get("unresolved_questions") or []
    notes = _scrub_for_injection(memory.get("research_notes") or "")
    tmap = memory.get("topic_map") or {}
    goal = _scrub_for_injection(memory.get("research_goal") or "")

    ctx: dict[str, Any] = {
        "tier": tier,
        "papers": included_papers,
        "key_findings": findings,
        "unresolved_questions": questions,
        "research_notes": notes,
        "topic_map": tmap,
        "research_goal": goal,
        "active_papers_count": len(included_papers),
        "skipped_count": skipped_count,
    }

    if not _JINJA2_OK:
        # Fallback: legacy implementation. Should never happen in prod
        # (jinja2 is in requirements), but keeps the script importable on
        # bare-metal CI containers.
        return _format_working_memory_legacy(memory, tier=tier)

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent)),
        autoescape=select_autoescape(["j2", "html", "xml"], default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("_prompt.j2")
    out = template.render(**ctx)

    # P2-8 / P0-10: hard byte ceiling for safety (jinja renders byte-faithful).
    if len(out) > MAX_TOTAL_BYTES:
        footer = (
            f"\n\n_…[truncated to {MAX_TOTAL_BYTES // 1024} KB summary budget; "
            f"use academic_paper_recall to view remaining fields.]_"
        )
        # Reserve room for the footer itself so the final output stays ≤ MAX_TOTAL_BYTES.
        room = MAX_TOTAL_BYTES - len(footer.encode("utf-8"))
        cut = out.rfind("\n", 0, room)
        if cut < 0:
            cut = room
        out = out[:cut] + footer
    return out


def _format_working_memory_legacy(memory: dict[str, Any], tier: str = "summary") -> str:
    """Legacy hand-written implementation (kept for fallback / A-B test).

    Same semantics as the Jinja2 path; identical byte budget behaviour.
    Activated only when ``MEMORY_USE_LEGACY_PROMPT=1`` is set, or when
    Jinja2 is not importable.
    """
    MAX_KEYNOTES_CHARS = 500
    MAX_IMPORTANCE_CHARS = 300
    MAX_TOTAL_BYTES = 12 * 1024
    parts: list[str] = []
    papers = memory.get("paper_library", []) or []
    active_papers: list[dict] = []

    papers = memory.get("paper_library", [])
    if papers:
        # Filter out placeholder papers (no substantive notes) in summary mode
        if tier == "summary":
            active_papers = [
                p for p in papers
                if (p.get("key_notes", "").strip()
                    or p.get("detailed_notes", "").strip()
                    or p.get("importance", "").strip())
            ]
        else:
            active_papers = papers

        # Cache-friendly header: no paper count (it changes every download and
        # would invalidate the entire prefix). Count is deferred to the footer.
        if active_papers:
            parts.append("\n## Literature Library")

        papers_included = 0
        papers_skipped_budget = 0
        for p in active_papers:
            # P0-10 修复：检查字节预算，防 context 袄炸
            current_size = sum(len(s.encode("utf-8")) for s in parts)
            if current_size > MAX_TOTAL_BYTES:
                papers_skipped_budget += 1
                continue
            title = _scrub_for_injection((p.get("title") or "Untitled")[:100])
            author = _scrub_for_injection(p.get("first_author", "?"))
            year = p.get("year", "?")
            doi = p.get("doi", "")
            importance_raw = p.get("importance")
            if isinstance(importance_raw, int):
                importance = str(importance_raw)
            else:
                importance = str(importance_raw or "").strip()
            importance = _scrub_for_injection(importance)
            key_notes = _scrub_for_injection((p.get("key_notes") or "").strip())
            detailed = _scrub_for_injection((p.get("detailed_notes") or "").strip())
            topics = p.get("topics", [])
            excerpts = p.get("key_excerpts", [])
            paper_id = p.get("paper_id", "")
            status = p.get("status", "unknown")
            relevance = _scrub_for_injection((p.get("relevance") or "").strip().lower())

            papers_included += 1
            # P2-2: drop "{i+1}" prefix to keep cache-friendliness. Numbering
            # papers in the header invalidates every paper's title line when a
            # new paper is prepended; LLM prefix caching then re-evaluates
            # every downstream line. Papers are still uniquely identified by
            # their paper_id, so order does not carry meaning.
            parts.append(f"\n### {title}")
            parts.append(f"- paper_id: `{paper_id}` | DOI: {doi} | {author} ({year}) | status: {status}")

            if tier == "summary":
                # ── Tier 1: compact summary ──────────────────────────
                if topics:
                    parts.append(f"- Topics: {', '.join(topics)}")
                if relevance:
                    parts.append(f"- Relevance: {relevance}")
                if importance:
                    imp = importance[:MAX_IMPORTANCE_CHARS]
                    if len(importance) > MAX_IMPORTANCE_CHARS:
                        imp = imp.rsplit('.', 1)[0] + '.'
                    parts.append(f"- Importance: {imp}")
                if key_notes:
                    kn = key_notes[:MAX_KEYNOTES_CHARS]
                    if len(key_notes) > MAX_KEYNOTES_CHARS:
                        kn = kn.rsplit('.', 1)[0] + '.'
                    parts.append(f"- Key Notes: {kn}")
                if excerpts:
                    parts.append(
                        f"- Key Excerpts: {len(excerpts)} passages "
                        f"(use academic_paper_recall to view)"
                    )
                if detailed:
                    parts.append(
                        f"- Detailed Notes: {len(detailed)} chars "
                        f"(use academic_paper_recall to view)"
                    )
            else:
                # ── Tier "full": legacy behaviour ───────────────────
                if topics:
                    parts.append(f"- Topics: {', '.join(topics)}")
                if relevance:
                    parts.append(f"- Relevance: {relevance}")
                if importance:
                    parts.append(f"- Importance: {importance}")
                if key_notes:
                    parts.append(f"- Key Notes: {key_notes}")
                if detailed:
                    parts.append(f"- Detailed Notes: {detailed}")
                if excerpts:
                    parts.append(f"- Key Excerpts ({len(excerpts)} passages):")
                    for j, exc in enumerate(excerpts):
                        parts.append(f"  [{j+1}] {exc[:500]}")
                        if len(exc) > 500:
                            parts.append(f"      ... ({len(exc)} chars total)")

        # P0-10 footer：报告被 budget 跳过的论文数
        if papers_skipped_budget > 0 and tier == "summary":
            parts.append(
                f"\n_({papers_skipped_budget} additional papers skipped due to 12 KB summary budget; "
                f"use academic_paper_recall to view individually.)_"
            )

    # ── Key Findings (filter empty) ──────────────────────────────────
    findings = memory.get("key_findings", [])
    valid_findings = [f for f in findings if (f.get("text") or "").strip()]
    if valid_findings:
        parts.append(f"\n## Key Findings ({len(valid_findings)})")
        for f in valid_findings:
            conf = f.get("confidence", "?")
            src = ", ".join(f.get("source_pids", []))
            parts.append(f"- [{conf}] {f['text']} (from: {src})")

    # ── Unresolved Questions ────────────────────────────────────────
    questions = memory.get("unresolved_questions", [])
    if questions:
        parts.append(f"\n## Unresolved Questions ({len(questions)})")
        for q in questions:
            parts.append(f"- {q['question']}")

    # ── Research Notes ──────────────────────────────────────────────
    notes = memory.get("research_notes", "")
    if notes:
        parts.append(f"\n## Research Notes\n{notes[:3000]}")
        if len(notes) > 3000:
            parts.append(f"\n... ({len(notes)} chars total, truncated)")

    # ── Topic Map (dedup in summary: only topics with ≥2 papers) ────
    tmap = memory.get("topic_map", {})
    if tmap:
        if tier == "summary":
            multi = {t: pids for t, pids in tmap.items() if len(pids) >= 2}
            if multi:
                parts.append(
                    f"\n## Topic Map ({len(multi)} cross-paper topics"
                    + (f", {len(tmap)} total)" if len(multi) < len(tmap) else ")")
                )
                for topic, pids in sorted(multi.items()):
                    parts.append(f"- {topic}: {', '.join(pids)}")
        else:
            parts.append(f"\n## Topic Map ({len(tmap)} topics)")
            for topic, pids in sorted(tmap.items()):
                parts.append(f"- {topic}: {', '.join(pids)}")

    # ── Footer (cache-friendly — at the very end) ──────────────────
    goal = memory.get("research_goal", "")
    if goal:
        parts.append(f"\n---\n**Research Goal**: {goal}")
    if papers:
        parts.append(f"**Library stats**: {len(active_papers)} papers with notes"
                     + (f" ({len(papers)} total)" if len(active_papers) < len(papers) else ""))

    out = "\n".join(parts)
    # P2-8 / P0-10: hard byte ceiling. The per-paper budget check is a soft
    # gate (it skips papers *after* the first overflow) but a single over‑budget
    # paper can still push total size past MAX_TOTAL_BYTES — e.g. an unusually
    # long title / abstract / topics list. Truncate the final string at the
    # last line boundary so we never exceed the budget.
    if out.encode("utf-8")[:MAX_TOTAL_BYTES + 1] and len(out) > MAX_TOTAL_BYTES:
        # Find the last newline before the ceiling so we don't cut a line in half
        cut = out.rfind("\n", 0, MAX_TOTAL_BYTES)
        if cut < 0:
            cut = MAX_TOTAL_BYTES
        out = out[:cut] + (
            f"\n\n_…[truncated to {MAX_TOTAL_BYTES // 1024} KB summary budget; "
            f"use academic_paper_recall to view remaining fields.]_"
        )
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Research memory management")
    sub = parser.add_subparsers(dest="command", help="Command")

    p_update = sub.add_parser("update", help="Update memory with paper notes, findings, etc.")
    p_update.add_argument("data", help="JSON payload (mirrors update_memory schema)")

    sub.add_parser("list", help="List all papers in library (compact)")

    p_get = sub.add_parser("get", help="Get detailed info for specific papers")
    p_get.add_argument("--ids", help="Comma-separated paper_ids")
    p_get.add_argument("--topic", help="Topic filter")
    p_get.add_argument("--full-text", action="store_true",
                       help="Include cached full text from data/texts/{paper_id}.md")

    p_goal = sub.add_parser("goal", help="Set research goal")
    p_goal.add_argument("text", help="Research goal text")

    p_note = sub.add_parser("note", help="Append freeform research notes")
    p_note.add_argument("text", help="Note text")

    p_finding = sub.add_parser("finding", help="Add a key finding")
    p_finding.add_argument("data", help='JSON: {"text": "...", "source_pids": [...], "confidence": "..."}')

    p_unresolved = sub.add_parser("unresolved", help="Add an unresolved question")
    p_unresolved.add_argument("text", help="Question text")

    p_del = sub.add_parser("delete-paper",
                           help="Remove one paper plus its topic/finding references")
    p_del.add_argument("--id", required=True, help="paper_id (underscored) or DOI")
    p_del.add_argument("--doi", default="", help="DOI, when --id is something else")

    sub.add_parser("dump", help="Dump full memory state")
    p_wm = sub.add_parser("working-memory", help="Output formatted working memory block for system prompt")
    p_wm.add_argument("--tier", choices=["summary", "full"], default="summary",
                      help="summary (compact, default) or full (legacy with excerpts)")

    # Added 2026 (P0-1 / P2-1/2/3 of academic-context-injection.md §8)
    p_progress = sub.add_parser("progress", help="Update current_phase / completed_phases / config")
    p_progress.add_argument("data", help='JSON: {"current_phase":"4.6","completed_phases":["0","1"],"config":{...}}')

    p_resume = sub.add_parser("resume", help="Get next phase to run (for academic_resume)")
    p_resume.add_argument("--skill", default="", help="Skill name (optional; inferred from config if empty)")

    p_tel = sub.add_parser("telemetry", help="Read tool-call telemetry from data/telemetry/")
    p_tel.add_argument("--limit", type=int, default=20, help="Max entries to return (default 20)")

    args = parser.parse_args()

    session_id = get_session_id()
    memory = load_memory(session_id)

    if args.command == "update":
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}, ensure_ascii=False))
            sys.exit(1)
        result = cmd_update(memory, data)
        save_memory(session_id, memory)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "list":
        result = cmd_list(memory)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "get":
        ids = [i.strip() for i in args.ids.split(",")] if args.ids else []
        result = cmd_get(memory, ids=ids, topic=args.topic,
                         include_full_text=args.full_text)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "goal":
        result = cmd_goal(memory, args.text)
        save_memory(session_id, memory)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "note":
        result = cmd_note(memory, args.text)
        save_memory(session_id, memory)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "finding":
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}, ensure_ascii=False))
            sys.exit(1)
        result = cmd_finding(memory, data)
        save_memory(session_id, memory)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "unresolved":
        result = cmd_unresolved(memory, args.text)
        save_memory(session_id, memory)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "delete-paper":
        result = cmd_delete_paper(memory, args.id, args.doi)
        if result.get("status") == "ok":
            save_memory(session_id, memory)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "dump":
        print(json.dumps(memory, ensure_ascii=False, indent=2))

    elif args.command == "working-memory":
        tier = getattr(args, 'tier', 'summary')
        print(format_working_memory(memory, tier=tier))

    elif args.command == "progress":
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}, ensure_ascii=False))
            sys.exit(1)
        result = cmd_progress(memory, data)
        save_memory(session_id, memory)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "resume":
        result = cmd_resume(memory, skill=getattr(args, 'skill', ''))
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "telemetry":
        result = cmd_telemetry(memory, limit=getattr(args, 'limit', 20))
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
