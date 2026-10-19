"""MinerU token lifecycle: expiry, persistence and automatic renewal.

``MINERU_TOKEN`` is not a permanent API key. It is the OpenXLab SSO token
(``uaa-token`` cookie) that mineru.net hands out when you log in, and it is a
JWT with a short ``exp`` — 14 days for the login measured on 2026-10-19 (the
previous one lasted 90 days, so the number is MinerU's to change; this code reads
``exp`` rather than assuming one). When it lapses every MinerU call answers::

    HTTP 401  {"msgCode": "A0202", "msg": "user authenticate failed"}

Downloads are unaffected — they never touch MinerU — so the symptom presents as
"only the conversion is broken", which is exactly how the outage of 2026-10-11
went unnoticed for eight days. That is the failure this module exists to end.

Renewal happens on the real login page, not through an API
---------------------------------------------------------
``sso.openxlab.org.cn`` encrypts the password with RSA and, before it will
accept a login, demands an Aliyun captcha (``captchaVerifyParam``). A plain
HTTP login is therefore rejected with ``A0101 request parameter validation
error`` — measured, not assumed. So renewal drives a browser: fill the form,
click 登录, click the Aliyun "确认您不是机器人" checkbox, and read the
``uaa-token`` cookie the SSO sets on success.

Measured 2026-10-19 on this deployment:

* ``https://mineru.net/apiManage/token`` redirects to the SSO form, which is
  already visible on load — no "click 登录 in the header" step is needed;
* submitting without a captcha opens ``#aliyunCaptcha-checkbox-icon`` and the
  login request is withheld until it passes; clicking that element with
  Camoufox passes it and the request fires;
* the login request is ``POST /gw/uaa-be/api/v1/login/byAccount`` with
  ``{"account", "password": <RSA(account||password||epoch)>, "marketing"}``.

Where a renewed token is stored
-------------------------------
Not in ``.env``: the systemd unit runs under ``ProtectSystem=strict`` and only
``~/.local/state/academic-mcp``, the data dir and the cache are writable, so a
service-side ``.env`` edit would fail with ``EROFS``. The token is written to
``MINERU_TOKEN_FILE`` (default ``~/.local/state/academic-mcp/mineru_token.json``,
mode 0600) and ``current()`` prefers whichever of that file and ``.env`` expires
LATER — so a hand-pasted fresh token in ``.env`` still wins, and a stale value in
either place can never shadow a newer one.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .config import settings

logger = logging.getLogger("academic_mcp.mineru_auth")


class MineruAuthError(RuntimeError):
    """A token could not be obtained (no credentials, login failed, …)."""


# ══════════════════════════════════════════════════════════════════════
# Token inspection
# ══════════════════════════════════════════════════════════════════════


def claims(token: str) -> dict:
    """The JWT payload, without verifying the signature.

    The signature is MinerU's business — this only answers "when does it
    expire", and reading a claim you cannot forge is not a security decision.
    """
    parts = (token or "").split(".")
    if len(parts) != 3:
        return {}
    segment = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        decoded = json.loads(base64.urlsafe_b64decode(segment))
    except (ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def expiry_of(token: str) -> datetime | None:
    """``exp`` as an aware datetime, or None for an opaque/unsigned token."""
    raw = claims(token).get("exp")
    if not isinstance(raw, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(raw, UTC)
    except (OverflowError, OSError, ValueError):
        return None


@dataclass(frozen=True)
class TokenInfo:
    """One token and what is known about it."""

    token: str
    exp: datetime | None
    source: str  # "env" | "state"
    account: str = ""

    @property
    def days_left(self) -> float | None:
        if self.exp is None:
            return None
        return (self.exp - datetime.now(UTC)).total_seconds() / 86400

    @property
    def expired(self) -> bool:
        left = self.days_left
        return left is not None and left <= 0

    def needs_renewal(self, within_days: int) -> bool:
        """True when this token is expired or about to be.

        A token with no readable ``exp`` is left alone: without a deadline
        there is nothing to schedule against, and renewing blindly on every
        call would be a login per request.
        """
        left = self.days_left
        return left is not None and left <= within_days

    def label(self) -> str:
        if self.exp is None:
            return f"token from {self.source} (no exp claim)"
        stamp = self.exp.strftime("%Y-%m-%d %H:%M UTC")
        left = self.days_left or 0.0
        state = "EXPIRED" if left <= 0 else f"{left:.1f} days left"
        return f"token from {self.source} expires {stamp} ({state})"


# ══════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════

_CACHE: tuple[float, TokenInfo | None] | None = None
_CACHE_TTL = 30.0
_LOCK = threading.Lock()  # one browser login at a time


def state_path() -> Path:
    return Path(settings.mineru_token_file).expanduser()


def read_state() -> TokenInfo | None:
    path = state_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    token = payload.get("token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        return None
    exp = expiry_of(token)
    if exp is None and isinstance(payload.get("exp"), (int, float)):
        exp = datetime.fromtimestamp(payload["exp"], UTC)
    return TokenInfo(
        token=token,
        exp=exp,
        source="state",
        account=str(payload.get("account") or ""),
    )


def write_state(info: TokenInfo) -> None:
    """Persist atomically, mode 0600, in a 0700 directory.

    The token is a bearer credential for the account's MinerU quota: it is a
    secret, so it is created private rather than chmod-ed afterwards.
    """
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    payload = {
        "token": info.token,
        "exp": int(info.exp.timestamp()) if info.exp else None,
        "account": info.account,
        "obtained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "renewed_by": "academic-mcp",
    }
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".mineru-token-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    global _CACHE
    _CACHE = None


def _pick(env_info: TokenInfo | None, state_info: TokenInfo | None) -> TokenInfo | None:
    """Whichever token expires later.

    Comparing expiry rather than preference is what makes a manual ``.env``
    refresh take effect immediately (later ``exp`` wins), and what stops a
    stale ``.env`` value from shadowing the token renewal just wrote (the file
    one is later). Equal expiry — the same token in both places — reports the
    configured source, because that is the one an operator can see.
    """
    if env_info is None:
        return state_info
    if state_info is None:
        return env_info
    env_exp = env_info.exp or datetime.min.replace(tzinfo=UTC)
    state_exp = state_info.exp or datetime.min.replace(tzinfo=UTC)
    return state_info if state_exp > env_exp else env_info


def current(refresh: bool = False) -> TokenInfo | None:
    """The token the MinerU client should use right now."""
    global _CACHE
    if not refresh and _CACHE is not None:
        stamped, cached = _CACHE
        if time.monotonic() - stamped < _CACHE_TTL:
            return cached
    env_token = settings.mineru_token
    env_info = (
        TokenInfo(token=env_token, exp=expiry_of(env_token), source="env",
                  account=settings.mineru_account)
        if env_token else None
    )
    info = _pick(env_info, read_state())
    _CACHE = (time.monotonic(), info)
    return info


def current_token() -> str:
    info = current()
    if info is None or not info.token:
        raise MineruAuthError(
            "MINERU_TOKEN is not configured and no renewed token is stored. Set "
            "MINERU_TOKEN in .env, or set MINERU_ACCOUNT + MINERU_PASSWORD and run "
            "`python -m academic_mcp.mineru_auth renew`."
        )
    return info.token


# ══════════════════════════════════════════════════════════════════════
# Validity probe
# ══════════════════════════════════════════════════════════════════════


def probe(token: str) -> bool:
    """Is this token accepted by the MinerU API?

    Queries a batch id that cannot exist: a valid token gets "no such batch",
    which is an authenticated answer; an invalid one gets 401/403. Costs no
    quota, unlike creating a real task to test with.
    """
    if not token:
        return False
    url = f"{settings.mineru_api_base}/extract-results/batch/{'0' * 32}"
    try:
        with httpx.Client(timeout=20, trust_env=False) as client:
            response = client.get(url, headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        logger.warning("MinerU token probe failed to reach the API: %s", exc)
        return False
    return response.status_code not in (401, 403)


# ══════════════════════════════════════════════════════════════════════
# Browser login
# ══════════════════════════════════════════════════════════════════════

# The SSO form is a plain login page: account field, password field, one 登录
# button. Selectors are placeholders/roles, not the CSS-module class names the
# reference implementation used (those carry a build hash and break on deploy).
ACCOUNT_INPUT = 'input[placeholder="邮箱/手机号/用户名"]'
PASSWORD_INPUT = 'input[placeholder="密码"]'
LOGIN_BUTTON = "登录"
# Aliyun captcha. The checkbox variant is what this deployment gets; the slider
# selectors are listed so a challenge-type change is a logged symptom, not a
# mystery timeout.
CAPTCHA_CHECKBOXES = (
    "#aliyunCaptcha-checkbox-icon",
    ".aliyunCaptcha-checkbox-icon",
    "#aliyunCaptcha-checkbox-left",
)
CAPTCHA_SLIDERS = (
    "#aliyunCaptcha-sliding-slider",
    ".aliyunCaptcha-sliding-slider",
)
CAPTCHA_POPUP = "#aliyunCaptcha-window-popup"
COOKIE_NAME = "uaa-token"


def _visible_display() -> bool:
    """Put the browser on the desktop the human is looking at.

    ``--visible`` is the escape hatch for a captcha the automation cannot
    solve: ``camoufox._setup_display()`` prefers the off-screen Xvfb, which
    nobody can interact with.
    """
    from .resolvers import camoufox as cf

    if os.environ.get("DISPLAY"):
        cf._force_x11()
        return True
    if cf._display_reachable(0):
        os.environ["DISPLAY"] = ":0"
        cf._force_x11()
        return True
    logger.warning(
        "--visible was requested but no desktop display is reachable (DISPLAY is unset "
        "and :0 is not listening); the browser will run headless, so nobody can solve "
        "the captcha by hand"
    )
    return False


async def _browser_token(page) -> str | None:
    """The SSO cookie, if the session already carries one."""
    try:
        cookies = await page.context.cookies()
    except Exception:  # noqa: BLE001 — a torn-down context is not an answer
        return None
    for cookie in cookies:
        if cookie.get("name") == COOKIE_NAME and cookie.get("value"):
            return str(cookie["value"])
    return None


async def _click_captcha(page) -> bool:
    """Click the Aliyun checkbox. Returns True when something was clicked."""
    for selector in CAPTCHA_CHECKBOXES:
        try:
            locator = page.locator(selector)
            if await locator.count() == 0:
                continue
            if not await locator.first.is_visible():
                continue
            await locator.first.click(timeout=3000)
            logger.info("clicked the Aliyun captcha checkbox (%s)", selector)
            return True
        except Exception as exc:  # noqa: BLE001 — try the next selector
            logger.debug("captcha click failed for %s: %s", selector, exc)
    return False


async def _captcha_kind(page) -> str:
    """'checkbox' | 'slider' | 'popup' | '' — for diagnostics and waiting."""
    for selector in CAPTCHA_CHECKBOXES:
        try:
            if await page.locator(selector).count() and await page.locator(selector).first.is_visible():
                return "checkbox"
        except Exception:  # noqa: BLE001
            pass
    for selector in CAPTCHA_SLIDERS:
        try:
            if await page.locator(selector).count() and await page.locator(selector).first.is_visible():
                return "slider"
        except Exception:  # noqa: BLE001
            pass
    try:
        if await page.locator(CAPTCHA_POPUP).count() and await page.locator(CAPTCHA_POPUP).first.is_visible():
            return "popup"
    except Exception:  # noqa: BLE001
        pass
    return ""


async def _login_async(account: str, password: str, *, visible: bool, timeout: float) -> str:
    from camoufox import AsyncCamoufox

    from .resolvers import camoufox as cf

    if importlib.util.find_spec("camoufox") is None:
        raise MineruAuthError(
            "camoufox is not installed, so the login page cannot be driven. "
            "Install it (pip install 'camoufox[geoip]==0.5.5' && camoufox fetch) or "
            "renew the token by hand and paste it into MINERU_TOKEN."
        )

    cf._clear_proxy_env()
    headful = _visible_display() if visible else cf._setup_display()
    deadline = time.monotonic() + timeout
    logger.info(
        "renewing the MinerU token: logging in as %s (headless=%s)", account, not headful
    )

    async with AsyncCamoufox(**cf._browser_config(not headful, None)) as browser:
        page = await browser.new_page()
        await page.goto(settings.mineru_login_url, wait_until="domcontentloaded", timeout=60000)

        # A session that is already authenticated lands on the token console.
        for _ in range(5):
            token = await _browser_token(page)
            if token and probe(token):
                return token
            await asyncio.sleep(0.6)

        try:
            await page.locator(ACCOUNT_INPUT).first.wait_for(state="visible", timeout=30000)
        except Exception as exc:  # noqa: BLE001
            raise MineruAuthError(
                "the MinerU login form did not appear. The page returned "
                f"{page.url!r} — check MINERU_LOGIN_URL, or that mineru.net is reachable."
            ) from exc

        # The login answer itself is the only reliable signal. Scraping the DOM
        # for an error missed a rejected password completely (the toast had
        # already faded by the time the captcha closed), and the run sat there
        # until its deadline. The response is unambiguous: msgCode "10000" is
        # success, anything else carries the reason in `msg`.
        outcome: dict[str, str] = {}

        async def _record_login(response) -> None:
            if "/api/v1/login/byAccount" not in (response.url or ""):
                return
            try:
                body = await response.json()
            except Exception:  # noqa: BLE001 — not JSON is not an answer
                return
            outcome["msgCode"] = str((body or {}).get("msgCode") or "")
            outcome["msg"] = str((body or {}).get("msg") or "")

        page.on(
            "response",
            lambda response: asyncio.ensure_future(_record_login(response)),
        )

        await page.locator(ACCOUNT_INPUT).first.fill(account)
        await page.locator(PASSWORD_INPUT).first.fill(password)
        await page.get_by_role("button", name=LOGIN_BUTTON).first.click()
        logger.info("submitted the login form; waiting for the SSO token")

        last_captcha, last_click, clicks = "", 0.0, 0
        rejected = 0
        while time.monotonic() < deadline:
            if outcome.get("msgCode") and outcome["msgCode"] != "10000":
                raise MineruAuthError(
                    f"MinerU rejected the login: {outcome.get('msg') or outcome['msgCode']}"
                )
            token = await _browser_token(page)
            if token:
                # A stale uaa-token cookie can outlive the session that set it,
                # and a fresh one can be set a moment before it becomes valid:
                # never return a token the API rejects, but do not give up on
                # the first miss either.
                if probe(token):
                    logger.info("MinerU login succeeded; a fresh token is in hand")
                    return token
                rejected += 1
                if rejected >= 3:
                    raise MineruAuthError(
                        "the SSO returned a uaa-token but the MinerU API rejects it "
                        "(HTTP 401). The account may not be entitled to the parse API, "
                        "or the token is not the one the API expects; check that this "
                        "account can use https://mineru.net/apiManage in a normal browser."
                    )
            kind = await _captcha_kind(page)
            if kind and kind != last_captcha:
                logger.warning("Aliyun captcha appeared (%s) — solving it", kind)
                last_captcha = kind
            # Capped on purpose: every click submits a login, so a captcha that
            # reopens after a rejected password would otherwise burn attempt
            # after attempt against the account. Three is enough to tell "not
            # yet rendered" from "will not pass".
            if (
                kind == "checkbox"
                and clicks < 3
                and time.monotonic() - last_click > 5
            ):
                last_click = time.monotonic()
                clicks += 1
                await _click_captcha(page)
            elif kind == "slider":
                raise MineruAuthError(
                    "the Aliyun captcha switched to a slider, which this automation cannot "
                    "solve. Re-run with `python -m academic_mcp.mineru_auth renew --visible` "
                    "and drag the slider yourself, or paste a fresh MINERU_TOKEN into .env."
                )

            await asyncio.sleep(1.0)

        if outcome.get("msgCode") == "10000":
            raise MineruAuthError(
                "the SSO accepted the login but no uaa-token cookie appeared. "
                "Re-run with `--visible` to see what the page does; if the site "
                "changed its login response, paste a fresh MINERU_TOKEN into .env."
            )
        raise MineruAuthError(
            f"the MinerU login did not finish within {int(timeout)}s "
            f"(captcha clicks: {clicks}, last captcha seen: {last_captcha or 'none'}). "
            "If the captcha was never dismissed, retry with "
            "`python -m academic_mcp.mineru_auth renew --visible` and solve it by hand."
        )


def _run_sync(coro):
    """Run a coroutine from sync code, even inside a thread that has a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, object] = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised in the caller
            box["error"] = exc

    thread = threading.Thread(target=runner, name="mineru-login", daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


def renew(
    *,
    force: bool = False,
    stale: str | None = None,
    visible: bool = False,
    timeout: float | None = None,
) -> TokenInfo:
    """Obtain a fresh token, persisting it. Raises :class:`MineruAuthError`.

    ``stale`` is the token that just got rejected. When another thread has
    already replaced it while this call waited for the lock, that newer token
    is returned instead of logging in a second time — a burst of 401s from
    parallel conversions must cause exactly one browser login.
    """
    if not settings.can_renew:
        missing = "MINERU_ACCOUNT" if not settings.mineru_account else "MINERU_PASSWORD"
        raise MineruAuthError(
            f"automatic renewal is not configured ({missing} is empty, or "
            "MINERU_TOKEN_RENEW=false). Set both in .env, or paste a fresh "
            "MINERU_TOKEN from https://mineru.net/apiManage."
        )

    with _LOCK:
        info = current(refresh=True)
        if stale and info is not None and info.token != stale:
            logger.info("another caller already renewed the MinerU token")
            return info
        if (
            not force
            and info is not None
            and not info.needs_renewal(settings.mineru_token_renew_days)
        ):
            return info

        token = _run_sync(
            asyncio.wait_for(
                _login_async(
                    settings.mineru_account,
                    settings.mineru_password,
                    visible=visible,
                    timeout=float(timeout or settings.mineru_login_timeout),
                ),
                timeout=float(timeout or settings.mineru_login_timeout) + 30,
            )
        )
        if not token:
            raise MineruAuthError("the login finished but no uaa-token was captured")
        fresh = TokenInfo(
            token=token,
            exp=expiry_of(token),
            source="state",
            account=settings.mineru_account,
        )
        write_state(fresh)
        logger.info("MinerU token renewed: %s", fresh.label())
        return fresh


def ensure() -> TokenInfo | None:
    """Renew when configured and needed; never raise.

    Used by the startup path, where a failed login must degrade to a warning
    rather than keep the MCP server from serving.
    """
    try:
        info = current(refresh=True)
        if not settings.can_renew:
            return info
        if info is not None and not info.needs_renewal(settings.mineru_token_renew_days):
            return info
        return renew(force=True)
    except MineruAuthError as exc:
        logger.warning("MinerU token renewal skipped: %s", exc)
        return current(refresh=True)
    except Exception as exc:  # noqa: BLE001 — a browser crash must not kill startup
        logger.warning("MinerU token renewal failed: %s: %s", type(exc).__name__, exc)
        return current(refresh=True)


def startup_check() -> None:
    """Log the token's state, and renew it off the critical path."""
    info = current(refresh=True)
    if info is None:
        logger.warning(
            "MINERU_TOKEN is unset — PDF→Markdown conversion will fail. Set it, or "
            "set MINERU_ACCOUNT + MINERU_PASSWORD for automatic renewal."
        )
        # Credentials are enough on their own: a deployment that only fills in
        # the account and password should get a token at first start rather
        # than an error telling it to also paste one.
        if settings.can_renew:
            threading.Thread(target=ensure, name="mineru-renew", daemon=True).start()
        return
    if info.expired:
        logger.error(
            "MinerU %s — every conversion will fail with HTTP 401 until it is renewed",
            info.label(),
        )
        if settings.can_renew:
            threading.Thread(
                target=ensure, name="mineru-renew", daemon=True
            ).start()
        return
    logger.info("MinerU %s", info.label())
    if info.needs_renewal(settings.mineru_token_renew_days) and settings.can_renew:
        logger.info(
            "renewing the MinerU token in the background (threshold %d days)",
            settings.mineru_token_renew_days,
        )
        threading.Thread(target=ensure, name="mineru-renew", daemon=True).start()


def status() -> dict:
    """Machine-readable token state for the `health` tool and the CLI."""
    info = current(refresh=True)
    return {
        "configured": info is not None,
        "source": info.source if info else "",
        "expires_at": info.exp.strftime("%Y-%m-%dT%H:%M:%SZ") if info and info.exp else "",
        "days_left": round(info.days_left, 2) if info and info.days_left is not None else None,
        "expired": bool(info and info.expired),
        "needs_renewal": bool(info and info.needs_renewal(settings.mineru_token_renew_days)),
        "renew_enabled": bool(settings.can_renew),
        "account": settings.mineru_account,
        "state_file": str(state_path()),
    }


# ══════════════════════════════════════════════════════════════════════
# CLI — `python -m academic_mcp.mineru_auth {status,renew,probe}`
# ══════════════════════════════════════════════════════════════════════


def _fingerprint(token: str) -> str:
    """Enough of the token to correlate two reports, not enough to use it."""
    return f"{token[:10]}…{token[-4:]}" if len(token) > 18 else "***"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m academic_mcp.mineru_auth",
        description=(
            "Inspect and renew the MinerU token. The token is an OpenXLab SSO JWT "
            "with a short lifetime (14 days when measured on 2026-10-19); renewal drives the real login page in a browser "
            "because the Aliyun captcha cannot be bypassed over plain HTTP."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="expiry, source and whether renewal is configured")
    sub.add_parser("probe", help="check the stored token against the MinerU API")
    renew_parser = sub.add_parser("renew", help="log in and store a fresh token")
    renew_parser.add_argument(
        "--visible",
        action="store_true",
        help="put the browser on the desktop so a human can finish the captcha",
    )
    renew_parser.add_argument(
        "--timeout", type=float, default=None, help="login ceiling in seconds"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.command == "status":
        state = status()
        print(json.dumps(state, ensure_ascii=False, indent=2))
        info = current(refresh=True)
        if info is not None:
            print(f"\n{info.label()}")
            print(f"token: {_fingerprint(info.token)}")
        if not state["renew_enabled"]:
            print(
                "\nautomatic renewal is OFF: set MINERU_ACCOUNT and MINERU_PASSWORD in .env "
                "(and MINERU_TOKEN_RENEW=true)."
            )
        return 0

    if args.command == "probe":
        info = current(refresh=True)
        if info is None:
            print("no token configured")
            return 2
        ok = probe(info.token)
        print(f"{'accepted' if ok else 'REJECTED'} by the MinerU API — {info.label()}")
        return 0 if ok else 1

    try:
        info = renew(force=True, visible=args.visible, timeout=args.timeout)
    except MineruAuthError as exc:
        print(f"renewal failed: {exc}", file=sys.stderr)
        return 1
    print(f"renewed: {info.label()}")
    print(f"token: {_fingerprint(info.token)}")
    print(f"stored: {state_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
