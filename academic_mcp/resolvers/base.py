"""Resolver interface shared by every download strategy.

A *resolver* answers one question: "given this paper, can you produce PDF
bytes?"  Resolvers never touch the filesystem and never convert — the
pipeline in :mod:`academic_mcp.pipeline` owns caching, ordering and
Markdown conversion.  That separation is what makes the strategy list
trivially reorderable and each strategy unit-testable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger("academic_mcp.resolvers")


@dataclass
class Paper:
    """The identity of the paper we are trying to obtain."""

    doi: str = ""
    title: str = ""
    first_author: str = ""
    key: str = ""

    @property
    def label(self) -> str:
        return self.doi or self.title[:60] or self.key


@dataclass
class Attempt:
    """Structured record of one resolver's outcome (for tool output)."""

    strategy: str
    ok: bool
    detail: str = ""
    seconds: float = 0.0
    bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "ok": self.ok,
            "detail": self.detail,
            "seconds": round(self.seconds, 1),
            "bytes": self.bytes,
        }


@runtime_checkable
class Resolver(Protocol):
    """One download strategy."""

    name: str

    def applies(self, paper: Paper) -> bool:
        """Cheap pre-check — skip resolvers that cannot possibly work."""
        ...

    async def fetch(self, paper: Paper) -> bytes | None:
        """Return PDF bytes, or None to let the next resolver try."""
        ...


@dataclass
class ResolverStats:
    """Lifetime counters, exposed by the ``health`` tool."""

    calls: int = 0
    successes: int = 0
    last_error: str = ""
    total_seconds: float = 0.0
    by_strategy: dict[str, list[int]] = field(default_factory=dict)

    def record(self, attempt: Attempt) -> None:
        self.calls += 1
        self.total_seconds += attempt.seconds
        bucket = self.by_strategy.setdefault(attempt.strategy, [0, 0])
        bucket[0] += 1
        if attempt.ok:
            self.successes += 1
            bucket[1] += 1
        elif attempt.detail:
            self.last_error = attempt.detail


STATS = ResolverStats()


async def run(resolver: Resolver, paper: Paper) -> tuple[bytes | None, Attempt]:
    """Execute one resolver with timing and error isolation."""
    started = time.monotonic()
    try:
        data = await resolver.fetch(paper)
    except Exception as exc:  # noqa: BLE001 - a broken resolver must not kill the chain
        attempt = Attempt(
            strategy=resolver.name,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}"[:300],
            seconds=time.monotonic() - started,
        )
        STATS.record(attempt)
        logger.warning("[%s] raised for %s: %s", resolver.name, paper.label, attempt.detail)
        return None, attempt

    seconds = time.monotonic() - started
    if data:
        attempt = Attempt(
            strategy=resolver.name, ok=True, seconds=seconds, bytes=len(data)
        )
        STATS.record(attempt)
        logger.info("[%s] SUCCESS %d bytes in %.1fs for %s", resolver.name, len(data), seconds, paper.label)
        return data, attempt

    attempt = Attempt(strategy=resolver.name, ok=False, detail="no PDF", seconds=seconds)
    STATS.record(attempt)
    logger.info("[%s] failed in %.1fs for %s", resolver.name, seconds, paper.label)
    return None, attempt
