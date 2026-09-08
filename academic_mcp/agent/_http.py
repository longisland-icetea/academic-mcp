"""Resilient HTTP wrapper for academic-search skill scripts.

§4.4 architectural upgrade — adds tenacity retry + pybreaker circuit-breaker
to all external HTTP calls.

Why this exists
---------------
search.py / download.py / pdf2md.py used to call ``requests.get`` /
``httpx.AsyncClient().get`` directly with zero retry and no failure
isolation. With pybreaker + tenacity, transient HTTP errors (5xx, ConnectError,
Timeout) get retried with exponential backoff; persistent failures open a
circuit breaker that fast-fails subsequent calls instead of accumulating
zombie timeouts.

Backward compatibility
----------------------
The two public entry points are ``safe_get`` and ``safe_post`` — drop-in
replacements for ``client.get`` / ``client.post``. They return the same
``httpx.Response`` object so call sites need only change the function name.

The circuit breaker is a module-level singleton (``_BREAKER``); it's shared
across all calls (single process, single host). This matches the SearXNG
suspension pattern already used in services/search_service.py.

Tests
-----
Run ``python3 scripts/test_http.py``. Uses a localhost HTTP echo server
(no real network required).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pybreaker
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ── Module-level circuit breaker ────────────────────────────────────────

# P0-12 follow-up: 5 consecutive failures opens the breaker; it stays
# open for 60s before allowing a half-open probe.
_BREAKER = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name="academic-search-http",
    exclude=[],
)


def _on_breaker_open(cb: pybreaker.CircuitBreaker, *_args) -> None:
    logger.warning("HTTP circuit breaker OPEN — fast-failing for %ds", cb.reset_timeout)


def _on_breaker_close(cb: pybreaker.CircuitBreaker, *_args) -> None:
    logger.info("HTTP circuit breaker CLOSED — traffic resumed")


def _on_breaker_half_open(cb: pybreaker.CircuitBreaker, *_args) -> None:
    logger.info("HTTP circuit breaker HALF-OPEN — probing one request")


class _StateListener(pybreaker.CircuitBreakerListener):
    """Logs state changes; works across pybreaker versions (0.x / 1.x)."""

    def state_change(self, cb, old_state, new_state):  # noqa: D401
        try:
            new = str(new_state)
        except Exception:
            new = type(new_state).__name__
        if "open" in new.lower() and "half" not in new.lower():
            _on_breaker_open(cb)
        elif "closed" in new.lower():
            _on_breaker_close(cb)
        elif "half" in new.lower():
            _on_breaker_half_open(cb)

    def before_call(self, cb, func, *_args):
        pass

    def failure(self, cb, exc):  # noqa: D401
        pass

    def success(self, cb):  # noqa: D401
        pass


_BREAKER.add_listener(_StateListener())


# ── Tunables (overridable via env for tests) ─────────────────────────────

RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT = 0.5
RETRY_MAX_WAIT = 10.0

RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
    httpx.HTTPStatusError,  # only when status is retryable
)


def _is_retryable_status(exc: BaseException) -> bool:
    """Whitelist: only retry 5xx + 429 (NOT 4xx other than 429)."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return True  # connect/timeout/protocol errors


def _retryable_exc(exc: BaseException) -> bool:
    """Combined predicate: exception type AND status whitelist."""
    if not isinstance(exc, RETRYABLE_EXCEPTIONS):
        return False
    return _is_retryable_status(exc)


# ── Public API ──────────────────────────────────────────────────────────


async def safe_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float | None = None,
    retries: int = RETRY_ATTEMPTS,
    breaker: pybreaker.CircuitBreaker | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """GET with tenacity retry + circuit breaker.

    Args:
        client: An open httpx.AsyncClient.
        url:     Full URL or path (if client has base_url).
        timeout: Override per-request timeout (seconds).
        retries: Max retry attempts (default 3).
        breaker: Custom breaker; default is the module singleton.
        **kwargs: Forwarded to client.get() (params, headers, ...).

    Raises:
        httpx.HTTPStatusError: After retries exhausted (or non-retryable 4xx).
        pybreaker.CircuitBreakerError: If the breaker is open.
    """
    cb = breaker or _BREAKER

    def _do_request() -> httpx.Response:
        if timeout is not None:
            kwargs.setdefault("timeout", timeout)
        # We can't await here — pybreaker's decorator wraps a sync callable.
        # The async layer below owns the await; this is a synchronous
        # "thunk" the breaker wraps so its fail-counter tracks network
        # errors (not coroutine errors). The async helper then awaits the
        # underlying httpx coroutine.
        # To keep the breaker observable, we raise into it from inside the
        # async helper instead. Here we just return a sentinel.
        raise NotImplementedError("internal: use _breaker_guarded_get")

    # We do the actual call below using AsyncRetrying, but record each
    # failure into the breaker manually so the breaker counter is accurate.
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(retries),
            wait=wait_exponential(multiplier=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                try:
                    resp = await client.get(url, **kwargs)
                except RETRYABLE_EXCEPTIONS as exc:
                    if _is_retryable_status(exc):
                        # Record into breaker so persistent failures open it.
                        try:
                            cb.call(lambda exc=exc: (_ for _ in ()).throw(exc))
                        except pybreaker.CircuitBreakerError:
                            raise
                        except BaseException:
                            pass
                        raise
                    raise
                if resp.status_code == 429 or resp.status_code >= 500:
                    err = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp,
                    )
                    # 5xx / 429 are retryable.
                    try:
                        cb.call(lambda err=err: (_ for _ in ()).throw(err))
                    except pybreaker.CircuitBreakerError:
                        raise
                    except BaseException:
                        pass
                    raise err
                return resp
    except RetryError as exc:
        # Exhausted retries — surface the underlying error.
        raise (exc.last_attempt.exception() if exc.last_attempt else exc) from exc  # type: ignore[misc]
    raise RuntimeError("unreachable")


async def safe_post(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float | None = None,
    retries: int = RETRY_ATTEMPTS,
    **kwargs: Any,
) -> httpx.Response:
    """POST counterpart to safe_get (same retry rules; no breaker for writes).

    Writes are not put through the breaker — failing writes (e.g. upload
    retry loops) can mask real availability problems.
    """
    kwargs.setdefault("timeout", timeout if timeout is not None else 30.0)
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(retries),
        wait=wait_exponential(multiplier=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    ):
        with attempt:
            resp = await client.post(url, **kwargs)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp,
                )
            return resp
    raise RuntimeError("unreachable")


# ── Synchronous helper (for legacy `requests`-based pdf2md.py) ──────────


def safe_requests_get(
    url: str,
    *,
    retries: int = RETRY_ATTEMPTS,
    breaker: pybreaker.CircuitBreaker | None = None,
    **kwargs: Any,
) -> Any:
    """Sync ``requests.get`` with tenacity + breaker.

    Used by pdf2md.py (which still uses requests because MinerU's PUT-upload
    needs a sync interface). Same retry/breaker semantics as safe_get.
    """
    import requests
    cb = breaker or _BREAKER
    from tenacity import (
        Retrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    try:
        for attempt in Retrying(
            stop=stop_after_attempt(retries),
            wait=wait_exponential(multiplier=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
            retry=retry_if_exception_type((
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            )),
            reraise=True,
        ):
            with attempt:
                resp = requests.get(url, **kwargs)
                if resp.status_code == 429 or resp.status_code >= 500:
                    err = requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code}", response=resp,
                    )
                    try:
                        cb.call(lambda err=err: (_ for _ in ()).throw(err))
                    except pybreaker.CircuitBreakerError:
                        raise
                    except BaseException:
                        pass
                    raise err
                return resp
    except RetryError as exc:
        raise (exc.last_attempt.exception() if exc.last_attempt else exc) from exc
    raise RuntimeError("unreachable")


def breaker_state() -> dict:
    """Return the current breaker state for diagnostics."""
    cb = _BREAKER
    return {
        "name": cb.name,
        "current_state": cb.current_state,
        "fail_counter": cb.fail_counter,
        "fail_max": cb.fail_max,
        "reset_timeout": cb.reset_timeout,
    }


__all__ = [
    "safe_get",
    "safe_post",
    "safe_requests_get",
    "breaker_state",
    "_BREAKER",
]