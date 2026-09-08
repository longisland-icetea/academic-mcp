"""Shared HTTP plumbing: one pooled client, bounded retries, proxy policy.

The original backend spread three different retry wrappers across four modules
(``tenacity`` in ``http_client.py``, a hand-rolled one in ``pdf_service.py``,
a no-op in ``url_builders.py``).  Here there is one: :func:`fetch`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger("academic_mcp.http")

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_RETRYABLE = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)

_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


def _proxy_arg() -> str | None:
    """Proxy for the shared client, honouring DOWNLOAD_PROXY_MODE."""
    if not settings.download_proxy:
        return None
    if settings.download_proxy_mode == "none":
        return None
    return settings.download_proxy


async def get_client(**overrides: Any) -> httpx.AsyncClient:
    """Return the process-wide pooled client (created on first use).

    Passing ``overrides`` is discouraged: such a client is neither pooled nor
    closed by :func:`aclose_client`, so callers must own its lifetime. It is
    kept only for one-off proxy hops.
    """
    global _client
    if overrides:
        logger.debug("creating unpooled httpx client with %s", sorted(overrides))
        kwargs: dict[str, Any] = dict(
            timeout=httpx.Timeout(
                settings.http_timeout, connect=settings.http_connect_timeout
            ),
            follow_redirects=True,
            trust_env=False,
            headers={"User-Agent": BROWSER_UA, "Accept": "*/*"},
        )
        proxy = _proxy_arg()
        if proxy and settings.download_proxy_mode == "primary":
            kwargs["proxy"] = proxy
        kwargs.update(overrides)
        return httpx.AsyncClient(**kwargs)

    if _client is None or _client.is_closed:
        async with _client_lock:
            if _client is None or _client.is_closed:
                kwargs = dict(
                    timeout=httpx.Timeout(
                        settings.http_timeout, connect=settings.http_connect_timeout
                    ),
                    follow_redirects=True,
                    # Never inherit http_proxy from the shell: publisher access
                    # must use the campus/institution IP, not the GFW proxy.
                    trust_env=False,
                    headers={"User-Agent": BROWSER_UA, "Accept": "*/*"},
                )
                proxy = _proxy_arg()
                if proxy and settings.download_proxy_mode == "primary":
                    kwargs["proxy"] = proxy
                    logger.info("Using download proxy as primary")
                _client = httpx.AsyncClient(**kwargs)
    return _client


async def aclose_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


class FetchResult:
    """Outcome of a single download attempt."""

    __slots__ = ("ok", "content", "status", "content_type", "error")

    def __init__(
        self,
        ok: bool,
        content: bytes = b"",
        status: int = 0,
        content_type: str = "",
        error: str = "",
    ) -> None:
        self.ok = ok
        self.content = content
        self.status = status
        self.content_type = content_type
        self.error = error


async def fetch(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    want_pdf: bool = False,
    timeout: float | None = None,
    attempts: int = 2,
    client: httpx.AsyncClient | None = None,
    proxy: str | None = None,
) -> FetchResult:
    """GET a URL with bounded retries.

    Args:
        url: Target URL.
        headers: Extra request headers.
        want_pdf: Reject responses that are not a plausible PDF
            (magic bytes or ``application/pdf`` content type).
        timeout: Override the default read timeout.
        attempts: Total attempts (1 = no retry).
        client: Use this client instead of the shared one.
        proxy: Force a specific proxy for this request.

    Returns a :class:`FetchResult`; never raises for network failures.
    """
    kwargs: dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if proxy:
        kwargs["proxy"] = proxy

    last_error = ""
    for attempt in range(1, max(1, attempts) + 1):
        active = client or await get_client(**({"proxy": proxy} if proxy else {}))
        try:
            resp = await active.get(url, headers=headers, **kwargs)
        except _RETRYABLE as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.debug("fetch %s attempt %d failed: %s", url[:100], attempt, last_error)
            if attempt < attempts:
                await asyncio.sleep(0.5 * attempt)
            continue
        except Exception as exc:  # noqa: BLE001 - never propagate to the caller
            return FetchResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        status = resp.status_code
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip()

        if status == 200:
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > settings.http_max_bytes:
                return FetchResult(
                    ok=False,
                    status=status,
                    error=f"response too large ({declared} bytes, cap {settings.http_max_bytes})",
                )
            if len(resp.content) > settings.http_max_bytes:
                return FetchResult(
                    ok=False,
                    status=status,
                    error=(
                        f"response too large ({len(resp.content)} bytes, "
                        f"cap {settings.http_max_bytes})"
                    ),
                )
            if want_pdf:
                magic_ok = resp.content[:4] == b"%PDF"
                if not (magic_ok or "pdf" in ctype.lower()):
                    return FetchResult(
                        ok=False,
                        status=status,
                        content_type=ctype,
                        error=f"not a PDF (content-type={ctype!r})",
                    )
                if len(resp.content) < 5000:
                    return FetchResult(
                        ok=False,
                        status=status,
                        content_type=ctype,
                        error=f"response too small ({len(resp.content)} bytes)",
                    )
            return FetchResult(
                ok=True, content=resp.content, status=status, content_type=ctype
            )

        if status in (401, 403):
            return FetchResult(ok=False, status=status, error="access denied (paywalled)")
        if status == 404:
            return FetchResult(ok=False, status=status, error="not found")
        if status == 429 or status >= 500:
            last_error = f"HTTP {status}"
            if attempt < attempts:
                await asyncio.sleep(1.0 * attempt)
                continue
        return FetchResult(
            ok=False, status=status, content_type=ctype, error=f"HTTP {status}"
        )

    return FetchResult(ok=False, error=last_error or "unknown error")


async def fetch_via_proxy(
    url: str,
    proxy: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 45.0,
) -> FetchResult:
    """One-shot GET through an explicit proxy (used as a fallback hop)."""
    try:
        async with httpx.AsyncClient(
            proxy=proxy, follow_redirects=True, timeout=timeout, trust_env=False
        ) as client:
            resp = await client.get(url, headers=headers)
    except Exception as exc:  # noqa: BLE001
        return FetchResult(ok=False, error=f"{type(exc).__name__}: {exc}")
    if resp.status_code == 200 and resp.content[:4] == b"%PDF" and len(resp.content) > 5000:
        return FetchResult(ok=True, content=resp.content, status=200)
    return FetchResult(ok=False, status=resp.status_code, error=f"HTTP {resp.status_code}")
