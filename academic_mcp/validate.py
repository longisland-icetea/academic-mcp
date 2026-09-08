"""DOI authorisation check.

The pi extension refuses to download a DOI that this session never saw in a
search result.  That is a deliberate guard against the reading subagent
hallucinating a plausible-looking DOI (or a paper's reference list injecting
one) and having us fetch arbitrary content on the user's behalf.

The check used to live in ``scripts/download.py``; it belongs with the
service that performs the download, so it moved here.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import storage
from .config import settings

logger = logging.getLogger("academic_mcp.validate")

_SESSION_ENV = "PI_SESSION"


class Denied(Exception):
    """Raised with a message meant for the calling agent."""

    def __init__(self, reason: str, code: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code


# pi's session ids look like `--home-cxxiao--` / `--mnt-c-Users-project-BSE_hBN--`.
# Anything else is rejected rather than interpolated into a path: the value ends
# up in `data_dir / f"{sid}.json"`, so an unvalidated one is a traversal hole
# (e.g. "../../etc/passwd" would happily read outside the data directory).
_SESSION_ID_RE = re.compile(r"^--[A-Za-z0-9._-]+--$")


def _session_id(session_id: str | None) -> str:
    import os

    raw = session_id or os.environ.get(_SESSION_ENV) or "default"
    if raw != "default" and not _SESSION_ID_RE.match(raw):
        logger.warning("rejecting malformed session id %r; falling back to 'default'", raw)
        return "default"
    return raw


def _norm(doi: str) -> str:
    """Canonical form for cache membership: bare, lowercased DOI."""
    d = (doi or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/"):
        if d.lower().startswith(prefix):
            d = d[len(prefix):]
            break
    return d.lower()


def _session_memory_path(sid: str) -> Path:
    return settings.data_dir / f"{sid}.json"


def _research_goal(sid: str) -> str:
    path = _session_memory_path(sid)
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    return (data.get("research_goal") or "").strip()


def _cached_dois(sid: str) -> tuple[set[str], str]:
    """Return (dois, why). ``dois`` is empty whenever it cannot be used.

    The ``why`` string is diagnostic only -- every caller collapses the cases
    into a single rejection, so it is logged rather than surfaced.
    """
    path = settings.search_cache_dir / f"{sid}.json"
    if not path.is_file():
        return set(), "no_cache"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("search cache %s unreadable: %s", path, exc)
        return set(), "cache_corrupt"
    # Normalise both sides. Cached entries are lowercased AND stripped of any
    # https://doi.org/ prefix: search.py/snowball.py currently store bare DOIs,
    # but matching on raw string equality would silently reject if any future
    # writer stored a URL form, and the failure looks exactly like "you never
    # searched for this paper".
    dois = {_norm(d) for d in (data.get("dois") or []) if d and d.strip()}
    if not dois:
        return set(), "empty_cache"
    return dois, "not_in_cache"


# A DOI is `10.<registrant>/<suffix>`. Without this check, `normalize_doi`
# happily returns "not-a-doi" (it only strips a URL prefix and punctuation),
# so a hallucinated string fell through to `reject` -- which tells the agent
# to go search, when the real problem is that the DOI is not a DOI at all.
_DOI_SHAPE = re.compile(r"^10\.\d{4,9}/\S+$")


def check(doi: str, session_id: str | None = None, *, require_goal: bool = True) -> str:
    """Validate a DOI against the session's search cache.

    Returns the normalised DOI.  Raises :class:`Denied` with an
    agent-readable reason otherwise.
    """
    clean = storage.normalize_doi(doi)
    if not clean or not _DOI_SHAPE.match(clean):
        raise Denied(
            f"`{doi}` 不是合法的 DOI（应为 `10.<注册号>/<后缀>`，例如 "
            "`10.1038/s41467-020-20667-2`）。\n"
            "请核对来源；只有拿到真实 DOI 才能检索并下载。",
            "malformed",
        )

    sid = _session_id(session_id)

    if require_goal and not _research_goal(sid):
        raise Denied(
            "本次会话尚未设置 research_goal — 拒绝下载。\n"
            "请先调用 academic_memory_update 设置研究目标，"
            "让下载请求有明确的授权上下文。",
            "goal_missing",
        )

    dois, why = _cached_dois(sid)
    # Everything from here is the same outcome with the same remedy: this DOI
    # is not in the session's allowlist. The old code split it into four codes
    # (no_cache / cache_corrupt / empty_cache / doi_rejected), which read as
    # four different problems but all mean "search for it first". Collapsed to
    # one `reject`; the specific cause goes to the log for debugging, not to
    # the agent -- the agent can act on it either way.
    if not dois or clean.lower() not in dois:
        logger.info(
            "reject %s for session %s (%s; %d dois cached)", clean, sid, why, len(dois or ())
        )
        raise Denied(
            f"DOI `{doi}` 尚未在本会话内授权下载（reject）。\n"
            "要下载它，先让检索命中它，任选其一：\n"
            "  · academic_search(query=\"<主题关键词>\") —— 它出现在结果里即可\n"
            "  · academic_search(query=\"<该 DOI>\", limit=5) —— 已知 DOI 的补票通道\n"
            "  · academic_citation_chain(seed=[\"<已知 DOI>\"]) —— 沿引文网络碰到它\n"
            "注意：白名单按会话（cwd）隔离，换目录要重新检索。",
            "reject",
        )
    return clean
