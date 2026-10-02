"""Shared HTTP plumbing: one pooled client, bounded retries, proxy failover.

The original backend spread three different retry wrappers across four modules
(``tenacity`` in ``http_client.py``, a hand-rolled one in ``pdf_service.py``,
a no-op in ``url_builders.py``).  Here there is one: :func:`fetch`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger("academic_mcp.http")

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ``ConnectTimeout`` is named explicitly on purpose: it derives from
# ``TimeoutException``, NOT from ``ConnectError``, so it used to be the one
# transport failure fetch() never retried — and a dropped SYN is exactly the
# blackhole pattern the failover hop exists for.
_RETRYABLE = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)

_client: httpx.AsyncClient | None = None
# One pooled client per distinct rescue proxy (``failover``/``fallback`` hops).
# Keeping them alive reuses the connection/TLS session across failovers instead
# of paying a fresh handshake on every rescue attempt.
_extra_clients: dict[str, httpx.AsyncClient] = {}
_client_lock = asyncio.Lock()


def _primary_proxy() -> str | None:
    """Proxy of the first route, honouring DOWNLOAD_PROXY_MODE.

    * ``primary`` — DOWNLOAD_PROXY, always;
    * ``fallback`` / ``failover`` — never on the first route: the direct
      connection is tried first and DOWNLOAD_PROXY is the rescue hop;
    * ``none`` / unknown — never.

    ``failover`` used to put DOWNLOAD_PROXY on the FIRST route, back when its
    rescue hop came from a separate DOWNLOAD_PROXY_FALLBACK key. Now that the
    rescue hop IS DOWNLOAD_PROXY, leaving it first would put the same address at
    both ends of the chain — the hop could never fire, and the mode would be
    indistinguishable from ``primary``. Direct-first is what makes it a failover.
    """
    if not settings.download_proxy:
        return None
    if settings.download_proxy_mode == "primary":
        return settings.download_proxy
    return None


def _fallback_proxy() -> str | None:
    """Rescue hop of ``fallback`` mode: direct first, then DOWNLOAD_PROXY."""
    if settings.download_proxy_mode != "fallback":
        return None
    return settings.download_proxy


def _failover_proxy() -> str | None:
    """Rescue hop of ``failover`` mode: DOWNLOAD_PROXY.

    This used to read a third key, ``DOWNLOAD_PROXY_FALLBACK``, which made
    ``failover`` the one mode whose rescue route DOWNLOAD_PROXY could not
    configure — so a deployment that set only DOWNLOAD_PROXY got no hop at all,
    exactly when it had declared a proxy to hop to. The routing vocabulary is
    now two keys wide:

      * ``GFW_PROXY``     — arXiv / search-engine / OpenAlex access;
      * ``DOWNLOAD_PROXY`` — publisher and API access.

    ``failover`` and ``fallback`` therefore describe the same two routes
    (primary, then DOWNLOAD_PROXY); they are kept as separate names only so an
    existing ``DOWNLOAD_PROXY_MODE`` value keeps its meaning.
    """
    if settings.download_proxy_mode != "failover":
        return None
    hop = settings.download_proxy
    if not hop or hop == _primary_proxy():
        return None
    return hop


def _routes() -> list[str | None]:
    """Ordered routes for one request: primary first, rescue hop last.

    Most modes yield a single route (``None`` = direct). ``fallback`` and
    ``failover`` add their rescue proxy, which :func:`fetch` only reaches
    after the primary route fails at the *transport* level.
    """
    routes: list[str | None] = [_primary_proxy()]
    for hop in (_fallback_proxy(), _failover_proxy()):
        if hop and hop not in routes:
            routes.append(hop)
    return routes


# ── Route-health memory ────────────────────────────────────────────────────
#
# A route that just failed at the transport level while a later route worked is
# skipped to the back of the chain for _ROUTE_TTL seconds. Without this a
# blackholed primary taxes EVERY request with its connect timeout (15 s per
# attempt for downloads, 4 s for Scopus) even though the rescue hop is a second
# away — a failover slower than the outage it rescues is not a failover. Keyed
# by (namespace, route) so a dead Scopus route never diverts publisher
# downloads, and vice versa.
_ROUTE_TTL = 300.0
_route_dead_until: dict[tuple[str, str | None], float] = {}


def order_routes(routes: list[str | None], ns: str) -> list[str | None]:
    """The same chain, with recently-failed routes moved to the back."""
    now = time.monotonic()
    live = [r for r in routes if _route_dead_until.get((ns, r), 0.0) <= now]
    dead = [r for r in routes if _route_dead_until.get((ns, r), 0.0) > now]
    return live + dead


def mark_route_failed(ns: str, route: str | None) -> None:
    """Record a transport failure: skip this route for the next _ROUTE_TTL."""
    _route_dead_until[(ns, route)] = time.monotonic() + _ROUTE_TTL


def mark_route_ok(ns: str, route: str | None) -> None:
    """A route just carried a request; clear any stale failure record."""
    _route_dead_until.pop((ns, route), None)


async def get_client(**overrides: Any) -> httpx.AsyncClient:
    """Return the process-wide pooled client (created on first use).

    ``overrides`` is discouraged: that path returns a fresh client the caller
    owns and must close. Only :func:`_client_for` uses it, for rescue hops,
    whose lifetime it manages and :func:`aclose_client` closes.
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
        proxy = _primary_proxy()
        if proxy:
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
                proxy = _primary_proxy()
                if proxy:
                    kwargs["proxy"] = proxy
                    logger.info("Using download proxy as primary")
                _client = httpx.AsyncClient(**kwargs)
    return _client


async def aclose_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
    for extra in _extra_clients.values():
        if not extra.is_closed:
            await extra.aclose()
    _extra_clients.clear()


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


async def _client_for(proxy: str | None) -> httpx.AsyncClient:
    """Client for one route: each distinct proxy gets its own pooled client.

    The primary route uses the main pooled client; a rescue proxy keeps a
    client of its own, alive across requests and closed together with the main
    one (:func:`aclose_client`), so repeated failovers reuse connections and
    TLS sessions instead of rebuilding them.
    """
    if proxy == _primary_proxy():
        return await get_client()
    key = proxy or ""
    client = _extra_clients.get(key)
    if client is None or client.is_closed:
        client = await get_client(**({"proxy": proxy} if proxy else {}))
        _extra_clients[key] = client
    return client


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
    """GET a URL with bounded retries and proxy failover.

    Args:
        url: Target URL.
        headers: Extra request headers.
        want_pdf: Reject responses that are not a plausible PDF
            (magic bytes or ``application/pdf`` content type).
        timeout: Override the default read timeout.
        attempts: Total attempts *per route* (1 = no retry).
        client: Use this client instead of the shared one. The caller then
            owns routing — no failover is attempted.
        proxy: Force this proxy for the request; also disables failover.

    Returns a :class:`FetchResult`; never raises for network failures.

    Route policy (no explicit ``client``/``proxy``): the request starts on the
    normal route — direct, or ``DOWNLOAD_PROXY`` when the mode puts it first.
    With ``DOWNLOAD_PROXY_MODE=failover`` a transport-level failure (connect
    error, timeout, protocol error) is retried once through
    ``DOWNLOAD_PROXY``; ``fallback`` mode reaches the same ``DOWNLOAD_PROXY``
    the same way.  An HTTP answer (401/403/429/5xx...) never triggers the
    extra hop: the far end answered, so another exit cannot help.
    A route that failed this way while a later one succeeded is skipped to
    the back of the chain for a while, so one blackholed primary does not
    cost every following download its connect timeout.
    """
    if client is not None:
        result, _ = await _fetch_route(
            url,
            headers=headers,
            want_pdf=want_pdf,
            timeout=timeout,
            attempts=attempts,
            client=client,
        )
        return result

    routes: list[str | None] = order_routes(
        [proxy] if proxy else _routes(), ns="download"
    )
    result = FetchResult(ok=False, error="no route attempted")
    failed: list[str | None] = []
    for idx, route in enumerate(routes):
        result, transport_failed = await _fetch_route(
            url,
            headers=headers,
            want_pdf=want_pdf,
            timeout=timeout,
            attempts=attempts,
            proxy=route,
        )
        if result.ok:
            mark_route_ok("download", route)
            for bad in failed:
                mark_route_failed("download", bad)
            return result
        if not transport_failed or idx + 1 >= len(routes):
            return result
        failed.append(route)
        logger.info(
            "fetch %s: %s unreachable (%s); retrying via %s",
            url[:100],
            "direct route" if route is None else route,
            result.error,
            routes[idx + 1],
        )
    return result


async def _fetch_route(
    url: str,
    *,
    headers: dict[str, str] | None,
    want_pdf: bool,
    timeout: float | None,
    attempts: int,
    client: httpx.AsyncClient | None = None,
    proxy: str | None = None,
) -> tuple[FetchResult, bool]:
    """One route's bounded retry loop.

    The boolean is True only when every attempt failed with a network error
    (never for an HTTP answer) — that is what tells :func:`fetch` whether the
    failover hop is worth trying.
    """
    kwargs: dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = timeout

    # httpx routes per client, not per request: the route's client is chosen
    # once, outside the retry loop (``_client_for`` pools it). Passing
    # ``proxy=`` to ``.get()`` would raise TypeError on httpx 0.28.
    active = client or await _client_for(proxy)

    last_error = ""
    transport_failed = False
    for attempt in range(1, max(1, attempts) + 1):
        try:
            resp = await active.get(url, headers=headers, **kwargs)
        except _RETRYABLE as exc:
            transport_failed = True
            last_error = f"{type(exc).__name__}: {exc}"
            logger.debug("fetch %s attempt %d failed: %s", url[:100], attempt, last_error)
            if attempt < attempts:
                await asyncio.sleep(0.5 * attempt)
            continue
        except Exception as exc:  # noqa: BLE001 - never propagate to the caller
            return (
                FetchResult(ok=False, error=f"{type(exc).__name__}: {exc}"),
                isinstance(exc, httpx.TransportError),
            )

        status = resp.status_code
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip()

        if status == 200:
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > settings.http_max_bytes:
                return (
                    FetchResult(
                        ok=False,
                        status=status,
                        error=f"response too large ({declared} bytes, cap {settings.http_max_bytes})",
                    ),
                    False,
                )
            if len(resp.content) > settings.http_max_bytes:
                return (
                    FetchResult(
                        ok=False,
                        status=status,
                        error=(
                            f"response too large ({len(resp.content)} bytes, "
                            f"cap {settings.http_max_bytes})"
                        ),
                    ),
                    False,
                )
            if want_pdf:
                magic_ok = resp.content[:4] == b"%PDF"
                if not (magic_ok or "pdf" in ctype.lower()):
                    return (
                        FetchResult(
                            ok=False,
                            status=status,
                            content_type=ctype,
                            error=f"not a PDF (content-type={ctype!r})",
                        ),
                        False,
                    )
                if len(resp.content) < 5000:
                    return (
                        FetchResult(
                            ok=False,
                            status=status,
                            content_type=ctype,
                            error=f"response too small ({len(resp.content)} bytes)",
                        ),
                        False,
                    )
            return (
                FetchResult(
                    ok=True, content=resp.content, status=status, content_type=ctype
                ),
                False,
            )

        if status in (401, 403):
            return (
                FetchResult(ok=False, status=status, error="access denied (paywalled)"),
                False,
            )
        if status == 404:
            return (FetchResult(ok=False, status=status, error="not found"), False)
        if status == 429 or status >= 500:
            last_error = f"HTTP {status}"
            if attempt < attempts:
                await asyncio.sleep(1.0 * attempt)
                continue
        return (
            FetchResult(
                ok=False, status=status, content_type=ctype, error=f"HTTP {status}"
            ),
            False,
        )

    return FetchResult(ok=False, error=last_error or "unknown error"), transport_failed


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
