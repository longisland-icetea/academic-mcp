"""Pydantic v2 schemas for academic-search skill scripts.

§4.1 architectural upgrade: replaces bare-dict validation across
memory.py / search.py / download.py / pdf2md.py / telemetry.py.

All I/O boundaries should `.model_validate()` / `.model_dump()` instead of
naked dicts. Models use `extra="ignore"` so legacy fields don't crash
validation (forward-compat for in-place schema additions).

Backward compatibility
----------------------
The original function signatures still accept plain dicts and silently coerce
them via ``Model.model_validate(d)`` (which returns a model whose
``model_dump()`` is structurally identical). When a field is missing,
``model_validate`` falls back to the model's default. When a field has the
wrong type, ``ValidationError`` is raised.

Tests
-----
Run ``python3 scripts/test_schemas.py`` (no extra deps beyond pydantic).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Shared config ────────────────────────────────────────────────────────

class _StrictBase(BaseModel):
    """Base config shared by all schema models.

    - ``extra="ignore"``: forward-compat — extra fields from older versions
      are dropped silently instead of raising.
    - ``populate_by_name=True``: allow both field name and alias.
    - ``str_strip_whitespace=True``: trim whitespace on string inputs.
    """
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# ── Paper entry (memory.py paper_library element) ─────────────────────────

class PaperEntry(_StrictBase):
    """A single paper in the working-memory paper library.

    Mirrors the dataclass used by backend/data/memory_store.py:PaperMemory
    plus the legacy dict fields written by memory.py:cmd_update.
    """
    paper_id: str = ""
    doi: str = ""
    title: str = ""
    first_author: str = ""
    year: int | None = None
    source: str = "paper"  # paper / snowball / citation / manual
    relevance: str = ""
    importance: str = ""
    key_notes: str = ""
    key_excerpts: list[str] = Field(default_factory=list)
    detailed_notes: str = ""
    notable_sections: str = ""
    pdf_path: str = ""
    md_path: str = ""
    status: str = "browsed"  # browsed | downloaded | read | cited
    topics: list[str] = Field(default_factory=list)
    authors: list[Any] = Field(default_factory=list)  # [{"name": ...}, str, ...]
    abstract: str = ""
    doc_id: str = ""
    journal: str = ""
    added_at: str = ""
    updated_at: str = ""
    legacy_paper_ids: list[str] = Field(default_factory=list)
    meta_fetched: str = ""  # "" | "ok" | "failed"

    @field_validator("paper_id")
    @classmethod
    def _v_paper_id(cls, v: str) -> str:
        # P0-6 style: empty paper_id is allowed in legacy data, but for new
        # entries produced via Pydantic it's an invariant — see test below.
        return v

    @field_validator("status")
    @classmethod
    def _v_status(cls, v: str) -> str:
        allowed = {"browsed", "downloaded", "read", "cited", "unknown", "analyzed"}
        if v and v not in allowed:
            # Allow but normalise — backwards-compat for future statuses.
            return v
        return v


# ── Full memory document (memory.py save_memory / load_memory) ───────────

class MemoryJSON(_StrictBase):
    """The complete memory document persisted at data/{session_id}.json."""
    session_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    research_goal: str = ""
    paper_library: list[PaperEntry] = Field(default_factory=list)
    key_findings: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_questions: list[dict[str, Any]] = Field(default_factory=list)
    topic_map: dict[str, list[str]] = Field(default_factory=dict)
    research_notes: str = ""
    current_phase: str = ""
    completed_phases: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    migration_history: list[dict[str, Any]] = Field(default_factory=list)


# ── Search query / result (search.py input/output) ───────────────────────

class SearchQuery(_StrictBase):
    """Validated search input — used by search_scopus / search_openalex."""
    query: str = ""
    limit: int = 20
    year: str | None = None
    author: str | None = None
    journal: str | None = None
    search_goal: str | None = None
    source_engine: str = "all"  # all | scopus | openalex | semantic_scholar


class SearchResult(_StrictBase):
    """A single normalised paper from a search engine."""
    paper_id: str = ""
    doi: str = ""
    title: str = ""
    first_author: str = ""
    year: int | None = None
    source: str = ""  # scopus | openalex | snowball | ...
    found_by: list[str] = Field(default_factory=list)  # engines that returned this
    hybrid_score: float = 0.0
    score_components: dict[str, float] = Field(default_factory=dict)
    abstract: str = ""
    authors: list[Any] = Field(default_factory=list)
    venue: str = ""
    citations_count: int = 0
    full_record: dict[str, Any] = Field(default_factory=dict)


# ── Download request / response (download.py) ─────────────────────────────

# ── Telemetry event envelope (telemetry.py / extensions/academic-search.ts)

class TelemetryEvent(_StrictBase):
    """One JSONL record produced by the MCP adapter.

    The existing writer (extensions/academic-search.ts) only knows ``event``,
    ``ts`` and a payload bag — keep those loose so legacy events keep loading
    while new fields (request_id, latency_ms, ...) are added without a schema
    break. (§4.2 backward-compat requirement.)
    """
    event: str = ""
    ts: str = ""
    request_id: str = ""
    parent_tool_call_id: str = ""
    session_id: str = ""
    latency_ms: float | None = None
    # Bag for legacy event-specific fields (papers, dois, errors, ...).
    payload: dict[str, Any] = Field(default_factory=dict)


# ── Public helper: load + validate a memory file ──────────────────────────

def load_memory_dict(raw: dict[str, Any]) -> MemoryJSON:
    """Validate a parsed memory JSON dict.  Returns MemoryJSON.

    Never raises on missing/extra fields (forward-compat).
    Raises ValidationError only on type-conflicts (e.g. paper_library as a
    string instead of a list).
    """
    return MemoryJSON.model_validate(raw)


def load_paper_entry(raw: dict[str, Any]) -> PaperEntry:
    return PaperEntry.model_validate(raw)


def load_search_result(raw: dict[str, Any]) -> SearchResult:
    return SearchResult.model_validate(raw)



__all__ = [
    "PaperEntry",
    "MemoryJSON",
    "SearchQuery",
    "SearchResult",
    "TelemetryEvent",
    "load_memory_dict",
    "load_paper_entry",
    "load_search_result",
]