"""Camoufox/Playwright browser resolver — the paywall fallback.

Everything above this resolver is plain HTTPS and takes seconds.  This one
launches a real (fingerprint-spoofed) Firefox, walks the publisher's article
page, solves CloudFlare Turnstile if challenged, and extracts the PDF by
click / network interception / in-page fetch.

Two invariants that the original 2171-line module got right and that must
survive any rewrite:

1. **Non-headless, on a virtual display.**  Headless Firefox is detected and
   challenged harder; the Turnstile checkbox also has to be clickable, which
   headless mode makes unreliable.  We therefore run ``headless=False`` and
   point DISPLAY at Xvfb (preferred) or WSLg.
2. **One browser at a time, with a deadline.**  Firefox is heavy; parallel
   readers would otherwise launch a dozen instances and thrash.  A global
   semaphore serialises launches and every session has a hard budget.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from ..config import settings
from .base import Paper

logger = logging.getLogger("academic_mcp.camoufox")

# Serialise browser launches across the whole server process.
_browser_lock = asyncio.Semaphore(1)

_MIN_PDF = 5000
_CF_HOSTS = ("challenges.cloudflare.com", "cloudflare.com")


# ══════════════════════════════════════════════════════════════════════
# Publisher detection & URLs
# ══════════════════════════════════════════════════════════════════════

def detect_publisher(doi: str) -> str | None:
    if not doi:
        return None
    table = (
        ("10.1021/", "acs"),
        ("10.1126/", "science"),
        ("10.34133/", "science"),
        ("10.1002/", "wiley"),
        ("10.1088/", "iop"),
        ("10.1016/", "elsevier"),
        ("10.1007/", "springer"),
        ("10.3390/", "mdpi"),
        ("10.1103/", "aps"),
        ("10.1146/", "annualreviews"),
        ("10.1038/", "nature"),
        ("10.1371/", "plos"),
        ("10.1145/", "acm"),
        ("10.1063/", "aip"),
        ("10.1093/", "oup"),
        ("10.1109/", "ieee"),
        ("10.1073/", "pnas"),
        ("10.2197/", "ipsj"),
        ("10.18653/", "acl"),
        ("10.1201/", "taylorfrancis"),
        ("10.1609/", "aaai"),
        ("10.21468/", "scipost"),
        ("10.1080/", "taylorfrancis"),
        ("10.1051/", "edp"),
        ("10.1186/", "bmc"),
    )
    for prefix, name in table:
        if doi.startswith(prefix):
            return name
    if doi.startswith("10.1039/") and "s4" not in doi:
        return "rsc"
    return None


def _rsc_urls(doi: str) -> tuple[str, str]:
    """RSC: doi.org for the session, ``/en/content/articlepdf/...`` for the PDF.

    This is the pattern the pre-rewrite code used, and after measuring both
    candidates on 2026-09-05 it is the correct one for fetching:

      * ``/en/content/articlepdf/{year}/{jcode}/{suffix}`` -> the PDF
        (verified: 5.4 MB, 22 pages, correct title, ``%%EOF`` present);
      * ``/cp/article/27/47/25232/887420/pdf`` (the modern landing path)
        -> renders the article HTML, no PDF response to capture.

    The DOI *landing* URL has changed to a volume/issue/page/article-id form
    that cannot be derived from the DOI, so the article page is reached via
    doi.org. That page is used only to establish the session — it exposes no
    PDF link, which is why an earlier version concluded RSC was paywalled.
    """
    suffix = doi.replace("10.1039/", "")
    year = "2024"
    if len(suffix) > 1 and suffix[0] == "d" and suffix[1].isdigit():
        year = f"202{suffix[1]}"
    m = re.match(r"^[a-z]+", suffix[2:] if len(suffix) > 2 else suffix)
    jcode = m.group(0) if m else ""
    if not jcode:
        return f"https://doi.org/{doi}", ""
    return (
        f"https://doi.org/{doi}",
        f"https://pubs.rsc.org/en/content/articlepdf/{year}/{jcode}/{suffix}",
    )


_MDPI_ISSN = {
    "nano": "2079-4991", "ma": "1996-1944", "ijms": "1422-0067",
    "s": "1424-8220", "rs": "2072-4292", "c": "2073-8995",
    "ijerph": "1660-4601", "applsci": "2076-3417",
    "sustainability": "2071-1050", "molecules": "1420-3049",
    "polymers": "2073-4360", "cells": "2073-4409", "cancers": "2072-6694",
    "biology": "2079-7737", "pharmaceutics": "1999-4923",
    "energies": "1996-1073", "water": "2073-4441", "nutrients": "2072-6643",
    "fi": "1999-5903",
}


def _mdpi_urls(doi: str) -> tuple[str, str]:
    suffix = doi.split("10.3390/", 1)[-1] if "10.3390/" in doi else doi
    m = re.match(r"^([a-zA-Z]+)(\d{2})(\d{2})(\d+)$", suffix)
    if not m:
        return f"https://www.mdpi.com/{doi}", f"https://www.mdpi.com/{doi}/pdf"
    issn = _MDPI_ISSN.get(m.group(1).lower())
    if not issn:
        return f"https://www.mdpi.com/{doi}", f"https://www.mdpi.com/{doi}/pdf"
    vol, issue, article = str(int(m.group(2))), str(int(m.group(3))), str(int(m.group(4)))
    return (
        f"https://www.mdpi.com/{issn}/{vol}/{issue}/{article}",
        f"https://www.mdpi.com/{issn}/{vol}/{issue}/{article}/pdf",
    )


def build_urls(doi: str, publisher: str | None) -> tuple[str, str]:
    """(article_url, pdf_url) for a publisher. Empty pdf_url = extract later."""
    if publisher == "acs":
        return f"https://pubs.acs.org/doi/{doi}", f"https://pubs.acs.org/doi/pdf/{doi}"
    if publisher == "science":
        return (
            f"https://www.science.org/doi/reader/{doi}",
            f"https://www.science.org/doi/pdf/{doi}?download=true",
        )
    if publisher == "wiley":
        # /doi/pdf/ is an HTML viewer; the real PDF is /doi/pdfdirect/.
        return (
            f"https://onlinelibrary.wiley.com/doi/{doi}",
            f"https://advanced.onlinelibrary.wiley.com/doi/pdfdirect/{doi}",
        )
    if publisher == "iop":
        return (
            f"https://iopscience.iop.org/article/{doi}",
            f"https://iopscience.iop.org/article/{doi}/pdf",
        )
    if publisher == "rsc":
        return _rsc_urls(doi)
    if publisher == "springer":
        return (
            f"https://link.springer.com/article/{doi}",
            f"https://link.springer.com/content/pdf/{doi}.pdf",
        )
    if publisher == "mdpi":
        return _mdpi_urls(doi)
    if publisher == "nature":
        suffix = doi.rsplit("/", 1)[-1]
        return (
            f"https://www.nature.com/articles/{suffix}",
            f"https://www.nature.com/articles/{suffix}.pdf",
        )
    if publisher == "plos":
        return (
            f"https://journals.plos.org/plosone/article?id={doi}",
            f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable",
        )
    if publisher == "acm":
        return f"https://doi.org/{doi}", f"https://dl.acm.org/doi/pdf/{doi}"
    if publisher == "pnas":
        return (
            f"https://www.pnas.org/doi/{doi}",
            f"https://www.pnas.org/doi/pdf/{doi}?download=true",
        )
    if publisher == "acl":
        suffix = (
            doi.split("10.18653/v1/", 1)[-1]
            if "10.18653/v1/" in doi
            else doi.rsplit("/", 1)[-1]
        )
        return f"https://aclanthology.org/{suffix}/", f"https://aclanthology.org/{suffix}.pdf"
    if publisher == "annualreviews":
        return f"https://doi.org/{doi}", f"https://www.annualreviews.org/doi/pdf/{doi}"
    if publisher == "scipost":
        # Verified 2026-09-05: both endpoints return 200, /pdf is a real
        # application/pdf. SciPost is fully open access, so this skips the
        # browser entirely.
        suffix = doi.split("/", 1)[1]
        return f"https://scipost.org/{suffix}", f"https://scipost.org/{suffix}/pdf"
    # elsevier / aps / aip / oup / ieee / ipsj / taylorfrancis / aaai / edp /
    # bmc / unknown: the PDF URL is discovered from the landing page after
    # navigation, because these either sit behind a session-dependent URL or
    # have journal-specific paths that cannot be derived from the DOI.
    return f"https://doi.org/{doi}", ""


# ══════════════════════════════════════════════════════════════════════
# Display setup
# ══════════════════════════════════════════════════════════════════════

def _display_reachable(display_number: int) -> bool:
    """Is an X server actually listening on :N?

    Checking only ``/tmp/.X<n>-lock`` is not enough: X servers also listen on
    the Linux **abstract** socket namespace (``@/tmp/.X11-unix/X99``), which
    ``ls`` cannot see.  On wsl-zz-desktop another service's Xvfb :99 is only
    reachable that way, so a lock-file-only check silently falls through to
    WSLg instead of the virtual display.
    """
    n = str(display_number)
    if os.path.exists(f"/tmp/.X{n}-lock"):
        return True
    if os.path.exists(f"/tmp/.X11-unix/X{n}"):
        return True
    try:
        import socket

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.5)
            # A leading NUL selects the abstract namespace.
            sock.connect(f"\0/tmp/.X11-unix/X{n}")
            return True
        except OSError:
            return False
        finally:
            sock.close()
    except Exception:  # noqa: BLE001
        return False


def _setup_display() -> bool:
    """Make sure a usable X display exists. Returns True when headful is possible.

    Preference order:

    1. A reachable Xvfb on ``:99``–``:96`` — headful and off-screen, ideal.
    2. The ambient ``DISPLAY`` (WSLg ``:0``) — headful, but windows appear on
       the Windows desktop.  Acceptable: losing the browser resolver entirely
       is worse than a transient Firefox window.
    3. Start our own Xvfb on ``:99`` if the binary is installed.
    4. Nothing available → caller falls back to headless.
    """
    if sys.platform != "linux":
        return True

    for n in (99, 98, 97, 96):
        if _display_reachable(n):
            disp = f":{n}"
            if os.environ.get("DISPLAY") != disp:
                logger.info("DISPLAY: %s → %s", os.environ.get("DISPLAY") or "(unset)", disp)
            os.environ["DISPLAY"] = disp
            _force_x11()
            return True

    if os.environ.get("DISPLAY"):
        logger.debug(
            "No Xvfb reachable; using ambient DISPLAY=%s (browser windows may appear)",
            os.environ["DISPLAY"],
        )
        _force_x11()
        return True

    if shutil.which("Xvfb"):
        try:
            proc = subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1920x1080x24", "-ac",
                 "+extension", "GLX", "+render"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            for _ in range(24):
                if _display_reachable(99):
                    break
                time.sleep(0.25)
            if _display_reachable(99):
                os.environ["DISPLAY"] = ":99"
                _force_x11()
                logger.info("Started Xvfb :99 (pid %d)", proc.pid)
                return True
            proc.terminate()
        except OSError as exc:
            logger.debug("Xvfb start failed: %s", exc)

    logger.warning("No X display available — falling back to a headless browser")
    return False


def _force_x11() -> None:
    """Firefox prefers Wayland when both are present; pin it to X11."""
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["MOZ_ENABLE_WAYLAND"] = "0"
    os.environ["GDK_BACKEND"] = "x11"


_PROXY_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "DOWNLOAD_PROXY",
)


def _clear_proxy_env() -> None:
    """Browser traffic uses its own proxy setting; env proxies would leak."""
    for var in _PROXY_VARS:
        os.environ.pop(var, None)


def _browser_config(headless: bool) -> dict:
    from camoufox import DefaultAddons  # heavy import; keep it out of module load

    config: dict = {
        "headless": headless,
        # Camoufox bundles uBlock Origin and downloads it on first launch. That
        # download currently answers 451, which leaves a half-extracted addon
        # behind — and every later launch then dies with
        # `InvalidAddonPath: manifest.json is missing`. We capture PDFs with
        # Playwright route interception, so the blocker buys us nothing.
        "exclude_addons": [DefaultAddons.UBO],
        "os": ["windows", "macos", "linux"],
        "humanize": {"maxTime": 2.0},
        "geoip": True,
        "disable_coop": True,
        "enable_cache": True,
        "block_webrtc": True,
        "i_know_what_im_doing": True,
        "firefox_user_prefs": {
            "media.peerconnection.enabled": False,
            "dom.webdriver.enabled": False,
            "dom.webnotifications.enabled": False,
        },
    }
    proxy = settings.download_proxy
    if proxy and settings.download_proxy_mode != "none":
        config["proxy"] = {"server": proxy}
    else:
        config["proxy"] = None
    return config


# ══════════════════════════════════════════════════════════════════════# ══════════════════════════════════════════════════════════════════════
# CloudFlare handling
#
# Rewritten 2026-09-05 after instrumenting APS + Wiley. Three things the old
# flow got wrong, all verified by measurement rather than intuition:
#
# 1. It assumed every CF page shows a Turnstile checkbox and clicked at
#    coordinates from a screenshot analysis. On APS the challenge is the
#    text-only "Performing security verification" interstitial — there is no
#    checkbox (`frames: []`), and the detector locked onto the heading text
#    instead. Measured: APS resolves in ~2 s with ZERO clicks. The clicks were
#    noise.
#
# 2. It always fired a second hard-coded click at 33%/44% of the viewport, so
#    "which click worked" was unknowable — and `page.viewport_size` is None
#    under Camoufox (fingerprinting), so that percentage was computed against
#    a 1280x720 default while the real viewport was 1680x951. The "fallback"
#    was landing at 25%/33% of the actual page.
#
# 3. It treated the mere presence of a `/cdn-cgi/` SCRIPT as "still blocked".
#    Publishers keep that script loaded after the challenge passes, so it is a
#    permanent false positive: on a fully-resolved Wiley page (HTTP 200, real
#    title, 103 KB of body) that was the only signal still firing, and the
#    loop burned its whole budget waiting for a challenge that was long gone.
#
# The replacement classifies first, then acts: wait for auto-resolving
# challenges, click only a real visible Turnstile widget (using its DOM
# bounding box, not pixels), and give up immediately on interactive captchas.
# ══════════════════════════════════════════════════════════════════════

# Signals that reliably mean "Cloudflare is standing in front of the page".
# Deliberately absent: presence of a /cdn-cgi/ <script> (see note 3 above) and
# the mere existence of an iframe (must be visible with a real box).
_PROBE_JS = """() => {
    const out = {};
    const href = location.href.toLowerCase();
    out.url_chl = href.includes('__cf_chl') || href.includes('/cdn-cgi/');
    const nav = performance.getEntriesByType('navigation');
    out.status = nav.length ? nav[0].responseStatus : 0;
    const t = (document.title || '').toLowerCase();
    out.title_cf = ['just a moment', 'attention required', '请稍候', 'cloudflare',
                    'checking'].some(k => t.includes(k));
    const b = (document.body ? document.body.innerText : '').toLowerCase();
    out.text_cf = ['checking your browser', 'ray id', 'performing security verification',
                   'enable javascript and cookies', '请稍候', '安全验证', '自动程序',
                   'ddos protection by cloudflare']
                  .some(k => b.includes(k));
    // Interactive captchas we cannot solve — fail fast instead of burning time.
    out.interactive = !!document.querySelector(
        "input[name='cf_captcha_kind'], .h-captcha, .g-recaptcha, #cf-captcha-container");
    // Turnstile widget: only counts if it is actually rendered on screen.
    out.turnstile = null;
    for (const f of document.querySelectorAll('iframe')) {
        const src = f.src || '';
        if (!/challenges\\.cloudflare\\.com|turnstile/.test(src)) continue;
        const r = f.getBoundingClientRect();
        const cs = getComputedStyle(f);
        const shown = cs.display !== 'none' && cs.visibility !== 'hidden'
                      && r.width > 4 && r.height > 4;
        if (shown) {
            out.turnstile = {x: r.x, y: r.y, w: r.width, h: r.height};
            break;
        }
    }
    if (!out.turnstile) {
        const d = document.querySelector('.cf-turnstile');
        if (d) {
            const r = d.getBoundingClientRect();
            if (r.width > 4 && r.height > 4) out.turnstile = {x: r.x, y: r.y, w: r.width, h: r.height};
        }
    }
    out.viewport = {w: window.innerWidth, h: window.innerHeight};
    return out;
}"""

# none | auto | turnstile | interactive
class Challenge:
    __slots__ = ("kind", "reason", "box", "viewport")

    def __init__(self, kind: str, reason: str = "", box: dict | None = None,
                 viewport: dict | None = None) -> None:
        self.kind = kind
        self.reason = reason
        self.box = box
        self.viewport = viewport

    @property
    def blocked(self) -> bool:
        return self.kind != "none"

    def __repr__(self) -> str:
        return f"Challenge({self.kind}, {self.reason!r})"


async def _probe_challenge(page) -> Challenge:
    """Classify the current page's Cloudflare state in one DOM round-trip."""
    try:
        s = await page.evaluate(_PROBE_JS)
    except Exception as exc:  # noqa: BLE001
        logger.debug("CF probe failed: %s", exc)
        return Challenge("none", "probe-failed")

    vp = s.get("viewport") or {}
    box = s.get("turnstile")

    if s.get("interactive"):
        return Challenge("interactive", "interactive-captcha", box, vp)
    if s.get("url_chl"):
        return Challenge("turnstile" if box else "auto", "url:__cf_chl", box, vp)
    if s.get("status") in (403, 503):
        return Challenge("turnstile" if box else "auto", f"http:{s['status']}", box, vp)
    if s.get("title_cf"):
        return Challenge("turnstile" if box else "auto", "title", box, vp)
    if s.get("text_cf"):
        return Challenge("turnstile" if box else "auto", "text", box, vp)
    if box:
        return Challenge("turnstile", "widget-only", box, vp)
    return Challenge("none", "", None, vp)


async def _click_turnstile_widget(page, box: dict, prefix: str = "") -> None:
    """Click the Turnstile checkbox using its measured bounding box.

    The widget is ~300x65 px with a ~28 px checkbox at its left edge, so the
    target is (x + 28, vertical centre). One click — the old code fired two
    (detected + fixed fallback) which made success unattributable.
    """
    w = box.get("w", 0) or 0
    h = box.get("h", 0) or 0
    cx = int(box["x"] + min(28, max(8, w * 0.09)))
    cy = int(box["y"] + h / 2)
    # Keep the click inside the widget even if the box is oddly shaped.
    cx = min(max(int(box["x"]) + 4, cx), int(box["x"] + w) - 4)
    cy = min(max(int(box["y"]) + 4, cy), int(box["y"] + h) - 4)
    logger.info(
        "%sclicking Turnstile checkbox at (%d,%d) inside widget box "
        "x=%.0f y=%.0f w=%.0f h=%.0f",
        prefix, cx, cy, box["x"], box["y"], w, h,
    )
    await page.mouse.move(cx, cy)
    await asyncio.sleep(0.15 + random.random() * 0.2)
    await page.mouse.click(cx, cy)


async def _resolve_challenge(
    page,
    *,
    prefix: str = "",
    budget: float = 60.0,
    click_interval: float = 5.0,
) -> tuple[bool, str]:
    """Drive the page past Cloudflare. Returns (resolved, how).

    Strategy by kind:
      auto       — wait only. These JS challenges solve themselves; clicks are
                   noise (measured: APS clears in ~2 s untouched).
      turnstile  — click the checkbox by bounding box, re-click at most every
                   `click_interval`, keep waiting in between.
      interactive — unsolvable. Return immediately so callers can move on
                   instead of burning the session budget.
    """
    started = time.monotonic()
    last_click = 0.0
    seen_kinds: set[str] = set()

    while True:
        elapsed = time.monotonic() - started
        if elapsed > budget:
            logger.info(
                "%sCF unresolved after %.0fs (kinds seen: %s)",
                prefix, elapsed, ",".join(sorted(seen_kinds)) or "none",
            )
            return False, "timeout"

        ch = await _probe_challenge(page)
        if not ch.blocked:
            if seen_kinds:
                logger.info(
                    "%sCF cleared after %.1fs (kinds: %s)",
                    prefix, elapsed, ",".join(sorted(seen_kinds)),
                )
            return True, "none"

        seen_kinds.add(ch.kind)

        if ch.kind == "interactive":
            logger.warning(
                "%sinteractive captcha (image selection) — cannot solve, giving up",
                prefix,
            )
            return False, "interactive"

        if ch.kind == "turnstile" and ch.box:
            if elapsed - last_click >= click_interval or last_click == 0.0:
                await _click_turnstile_widget(page, ch.box, prefix=prefix)
                last_click = elapsed
                # Give the widget a moment before re-probing.
                await asyncio.sleep(1.5)
                continue

        # "auto" challenges (and turnstile between clicks): just wait.
        await asyncio.sleep(1.5)


async def _with_cf_bypass(
    page,
    doi: str = "",
    *,
    goto_url: str | None = None,
    prefix: str = "",
    budget: float = 60.0,
) -> bool:
    """Navigate (optional), then wait out whatever Cloudflare puts in front.

    `doi`, when given, is used as an early exit signal: once the article text
    is on the page we can stop probing regardless of what the challenge state
    claims — that check is what makes the "script still loaded" false positive
    harmless.
    """
    if goto_url:
        try:
            resp = await asyncio.wait_for(
                page.goto(goto_url, wait_until="commit", timeout=20000), timeout=25
            )
            if resp is not None and resp.status in (403, 503):
                logger.info("%sHTTP %d — challenge expected", prefix, resp.status)
        except Exception:
            logger.debug("%sgoto did not complete; continuing", prefix)

    started = time.monotonic()
    resolved = False
    how = "none"

    while time.monotonic() - started < budget:
        # Early exit: if the DOI we are after is on the page, we are through.
        if doi:
            try:
                body = (await page.inner_text("body") or "")[:12000].lower()
                if doi.lower() in body or f"doi.org/{doi.lower()}" in body:
                    if len(body) > 400:
                        logger.info("%scontent loaded", prefix)
                        resolved, how = True, "content"
                        break
            except Exception:
                pass

        ch = await _probe_challenge(page)
        if not ch.blocked:
            resolved, how = True, "none"
            break

        resolved, how = await _resolve_challenge(
            page, prefix=prefix, budget=max(5.0, budget - (time.monotonic() - started))
        )
        break

    try:
        await page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
    await asyncio.sleep(0.5)
    logger.debug("%sbypass finished: resolved=%s via %s", prefix, resolved, how)
    return resolved

async def _meta_pdf_url(page) -> str:
    """Read the PDF location from the page's own metadata.

    ``<meta name="citation_pdf_url">`` is the Highwire/Google-Scholar standard
    and is emitted by AIP, APS, OUP, IEEE, Wiley, Springer and others — often as
    the *only* place the PDF URL appears, because the visible download control
    is rendered by JS or hidden behind a login.

    We previously looked only at ``<a href>`` elements, which is why AIP failed
    with "no PDF link matched" even though the URL was sitting in the page head
    the whole time.
    """
    try:
        return await page.evaluate(
            """() => {
                const names = ['citation_pdf_url', 'citation_pdf', 'dc.identifier.pdf'];
                for (const n of names) {
                    const m = document.querySelector(`meta[name="${n}"]`);
                    if (m && m.content) return m.content;
                    const p = document.querySelector(`meta[property="${n}"]`);
                    if (p && p.content) return p.content;
                }
                return '';
            }"""
        ) or ""
    except Exception:
        return ""


async def _wait_rendered(page, timeout: float = 20.0) -> bool:
    """Wait until the page has a real DOM.

    After a Cloudflare challenge clears, some publishers (Science among them)
    reload or client-side redirect, so whatever we query immediately after the
    bypass may be a transient half-built document — or an empty one. Polling
    for actual content is cheap and removes a whole class of "no PDF button
    found" false negatives.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            n = await page.evaluate(
                "() => ({a: document.querySelectorAll('a').length, "
                "t: document.body ? document.body.innerText.length : 0})"
            )
            if n and n.get("a", 0) >= 20 and n.get("t", 0) >= 500:
                return True
        except Exception:
            pass  # execution context swapped by a navigation; try again
        await asyncio.sleep(1.0)
    return False


# A PDF-viewer page (Science's /doi/epdf/) renders the document in a canvas and
# has no body text *by design*, so an empty body there is not evidence of a
# block. We still do the reload: on Science that round-trip is what provokes the
# viewer into fetching the PDF we then capture.
_VIEWER_RE = re.compile(r"/(epdf|pdf)(/|\?|$)|/viewer|pdf\.html", re.I)

# A URL that resolves to the PDF itself rather than a page offering one. Covers
# both real .pdf files and the path-style endpoints publishers use
# (J-STAGE ends its PDF URLs with "/_pdf", Science with "/epdf").
_PDF_URL_RE = re.compile(r"(\.pdf|/_pdf|/epdf)(\?|$)", re.I)


async def _await_pdf_response(page, timeout: float = 45.0) -> bool:
    """Poll until a complete PDF response has been captured."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await _collect_responses(page):
            return True
        await asyncio.sleep(1.5)
    return False


async def _page_has_content(page) -> bool:
    """Does the page have real content, or is it a title-only shell?

    Some publishers (Science especially) intermittently answer a perfectly
    valid request with a document that has a correct ``<title>`` but an empty
    ``<body>`` — a soft bot block. Retrying the navigation once is what a
    human would do by hitting reload; it is not an attempt to evade anything.
    """
    try:
        n = await page.evaluate(
            "() => ({t: document.body ? document.body.innerText.length : 0, "
            "a: document.querySelectorAll('a').length})"
        )
    except Exception:
        return False
    return bool(n and n.get("t", 0) >= 500 and n.get("a", 0) >= 20)


async def _ensure_content(
    page, article_url: str | None, doi: str, prefix: str = ""
) -> bool:
    """Wait for the page to render; if it is a stub, reload once and re-check."""
    if await _wait_rendered(page, timeout=20.0):
        return True
    # A PDF viewer legitimately has no body text — Science's /doi/epdf/ page
    # renders the PDF in a canvas and fetches it as a network response. Treat
    # "a PDF already arrived" as "we have what we came for" instead of forcing
    # a reload that can only waste ~20 s.
    if await _collect_responses(page):
        logger.info("%sPDF already captured as a network response", prefix)
        return True
    hint = " (viewer page — an empty body can be normal here)" \
        if _VIEWER_RE.search(page.url or "") else ""
    logger.info(
        "%spage is a stub%s (%s) — reloading once",
        prefix, hint, await _describe_page(page),
    )
    try:
        if article_url:
            await asyncio.wait_for(
                page.goto(article_url, wait_until="commit", timeout=20000), timeout=25
            )
        else:
            await asyncio.wait_for(page.reload(), timeout=25)
    except Exception as exc:
        logger.debug("%sreload failed: %s", prefix, type(exc).__name__)
        return False
    if await _wait_rendered(page, timeout=20.0):
        logger.info("%sreload produced content", prefix)
        return True
    logger.info(
        "%sstill a stub after reload (%s)", prefix, await _describe_page(page)
    )
    return False


async def _describe_page(page) -> str:
    """One-line description of where we ended up, for failure logs.

    Distinguishes the three ways a download can fail, which otherwise all look
    like "no PDF": blocked (challenge), paywalled (stub page, no PDF link), and
    broken (page fine but our selectors missed the link).
    """
    try:
        info = await page.evaluate(
            """() => {
                const txt = document.body ? document.body.innerText : '';
                let pdfish = 0;
                for (const a of document.querySelectorAll('a')) {
                    const h = a.getAttribute('href') || '';
                    if (/pdf|download/i.test(h)) pdfish++;
                }
                return {url: location.href, title: (document.title||'').slice(0,70),
                        body: txt.length, links: document.querySelectorAll('a').length,
                        pdfish: pdfish};
            }"""
        )
    except Exception as exc:
        return f"<page unreadable: {type(exc).__name__}>"
    if info["body"] < 500:
        kind = "stub/empty page (paywall or bot block)"
    elif info["pdfish"] == 0:
        kind = "page rendered but no PDF link matched"
    else:
        kind = "PDF link present but not captured"
    return (
        f"{kind}: url={info['url'][:90]} title={info['title']!r} "
        f"body={info['body']} links={info['links']} pdfish={info['pdfish']}"
    )


async def _dismiss_popups(page) -> None:
    for text in ("Accept all", "Accept All", "Accept", "同意", "Close", "關閉"):
        try:
            btn = await page.query_selector(f'button:has-text("{text}")')
            if btn and await btn.is_visible():
                await asyncio.wait_for(btn.click(timeout=0), timeout=3)
                await asyncio.sleep(0.3)
                return
        except Exception:
            continue


# ══════════════════════════════════════════════════════════════════════
# PDF extraction strategies
# ══════════════════════════════════════════════════════════════════════

_FETCH_JS = """async ([url, timeoutMs]) => {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
        const r = await fetch(url, {
            credentials: 'include',
            redirect: 'follow',
            signal: ctrl.signal,
        });
        if (!r.ok) return 'HTTP-' + r.status;
        const buf = new Uint8Array(await r.arrayBuffer());
        let bin = '';
        const chunk = 0x8000;
        for (let i = 0; i < buf.length; i += chunk) {
            bin += String.fromCharCode.apply(null, buf.subarray(i, i + chunk));
        }
        return btoa(bin);
    } catch (e) {
        // Include the message, not just the name: "ERR:TypeError" told us
        // nothing, while the message distinguishes CORS, network and redirect
        // failures.
        return 'ERR:' + ((e && e.name) || 'Error') + ': ' + ((e && e.message) || String(e));
    } finally {
        clearTimeout(timer);
    }
}"""


def _valid(data: bytes | None) -> bool:
    """Is this a complete, usable PDF?

    The end-of-file marker matters: reading a Playwright response body before
    the response has finished yields a *truncated* file that still starts with
    ``%PDF`` and passes a size check. We shipped one of those (a 262144-byte
    fragment of a 1.1 MB paper — exactly 256 KiB, no ``%%EOF``) before this
    check existed.
    """
    if not data or len(data) <= _MIN_PDF:
        return False
    if data[:4] != b"%PDF":
        return False
    return b"%%EOF" in data[-4096:]


async def _browser_fetch(page, url: str, timeout: int = 20) -> bytes | None:
    """Fetch a URL with the browser's session cookies, in-page."""
    try:
        result = await asyncio.wait_for(
            page.evaluate(_FETCH_JS, [url, timeout * 1000]), timeout=timeout + 8
        )
    except Exception as exc:
        # Logged at INFO on purpose: a silent None here is what made the
        # Science failure look like "no PDF button" instead of "PDF fetch
        # was refused".
        logger.info("  browser fetch %s failed: %s", url[:80], type(exc).__name__)
        return None
    if isinstance(result, str) and result.startswith("HTTP-"):
        logger.info("  browser fetch %s -> %s", url[:80], result)
        return None
    if isinstance(result, str) and result.startswith("ERR:"):
        logger.info("  browser fetch %s -> %s", url[:80], result)
        return None
    if not result or not isinstance(result, str):
        logger.info("  browser fetch %s -> empty", url[:80])
        return None
    try:
        import base64

        data = base64.b64decode(result)
    except Exception:
        logger.info("  browser fetch %s -> undecodable", url[:80])
        return None
    if not _valid(data):
        logger.info(
            "  browser fetch %s -> %d bytes, not a PDF", url[:80], len(data)
        )
        return None
    return data


def _pdf_candidates(url: str) -> list[str]:
    """URLs to try when the browser is sitting on a PDF-ish endpoint.

    J-STAGE is why this exists: its ``.../_pdf`` URL answers slowly enough that
    ``api_get`` times out, while ``.../_pdf/-char/en`` is a direct
    ``application/pdf`` response. Try both, cheapest-looking first.
    """
    out = [url]
    if re.search(r"/_pdf(\?|$)", url, re.I):
        out.append(url + "/-char/en")
    return out


async def _api_get(page, url: str, timeout: float = 45.0) -> bytes | None:
    """Fetch a URL with the browser context's cookies via Playwright's request.

    This is the sweet spot for PDF capture:
      * it shares the browser's cookie jar, so session-gated PDFs work;
      * it is **not** subject to CORS, unlike an in-page ``fetch()`` (publisher
        PDF URLs very often 302 to a CDN, which fails as "NetworkError when
        attempting to fetch resource");
      * it does not depend on intercepting a network event, so it also works
        when the browser has already navigated to the PDF and rendered it in its
        built-in viewer — where there is no response left to capture (that is
        exactly how 10.12693/aphyspola.85.245 was slipping through).
    """
    try:
        resp = await asyncio.wait_for(
            page.request.get(url, timeout=timeout * 1000), timeout=timeout + 5
        )
    except Exception as exc:
        # INFO, not DEBUG: a timeout here is the difference between "we got the
        # PDF" and "no PDF", and it is invisible otherwise.
        logger.info("  api_get %s -> failed: %s: %s",
                    url[:80], type(exc).__name__, str(exc)[:60])
        return None
    try:
        if resp.status >= 400:
            logger.info("  api_get %s -> HTTP %d", url[:80], resp.status)
            return None
        body = await resp.body()
    except Exception as exc:
        logger.info("  api_get %s -> body failed: %s", url[:80], type(exc).__name__)
        return None
    if not _valid(body):
        logger.info("  api_get %s -> %d bytes, not a valid PDF", url[:80], len(body))
        return None
    return body


async def _capture_pdf_route(page, url: str, doi: str = "", prefix: str = "") -> bytes | None:
    """Navigate to *url* with a route interceptor that grabs the PDF body.

    ``page.route`` is used instead of ``page.on("response")`` because the
    browser often consumes the response body before ``response.body()`` is
    called ("body evicted" errors).
    """
    from playwright.async_api import Route

    captured: list[bytes] = []

    async def _intercept(route: Route) -> None:
        host = (route.request.url or "").lower()
        if any(h in host for h in _CF_HOSTS):
            await route.continue_()
            return
        try:
            resp = await route.fetch()
            if "application/pdf" in (resp.headers.get("content-type") or "").lower():
                try:
                    captured.append(await resp.body())
                except Exception:
                    pass
            await route.fulfill(response=resp)
        except Exception:
            try:
                await route.continue_()
            except Exception:
                pass

    await page.route("**/*", _intercept)
    try:
        await _with_cf_bypass(page, doi, goto_url=url, prefix=prefix)
        await asyncio.sleep(2)
    finally:
        try:
            await page.unroute("**/*")
        except Exception:
            pass

    for blob in captured:
        if _valid(blob):
            return blob
    return None


# Cap the response bucket. Every PDF-ish response is recorded, and the candidate
# filter is deliberately broad (octet-stream too), so an unbounded list would
# grow for the life of the browser session -- and each poll re-awaits
# `finished()` on all of them, so it also degrades quadratically.
_MAX_TRACKED_RESPONSES = 40


async def _collect_responses(page) -> list[bytes]:
    """Drain the network-response bucket into validated PDF bytes.

    ``await resp.finished()`` is what makes this safe: without it we can read
    a half-streamed body and cache a truncated PDF that looks fine to every
    other check.

    Responses are marked once awaited, so repeated polls (``_await_pdf_response``
    polls every 1.5 s) do not re-block on ones that will never finish.
    """
    bucket = getattr(page, "_pdf_responses", None)
    if not bucket:
        return []
    seen = getattr(page, "_pdf_responses_checked", None)
    if seen is None:
        seen = set()
        page._pdf_responses_checked = seen

    out: list[bytes] = []
    for idx, resp in enumerate(list(bucket)):
        if idx in seen:
            continue
        try:
            if resp.status >= 400:
                seen.add(idx)
                continue
            await asyncio.wait_for(resp.finished(), timeout=15)
        except Exception:
            # Not finished *yet* -- leave it unmarked and try again next poll.
            continue
        seen.add(idx)
        try:
            body = await resp.body()
        except Exception:
            continue
        if _valid(body):
            out.append(body)
    return out


_PUBLISHER_SELECTORS: dict[str, list[str]] = {
    "scipost": ['a[href$="/pdf"]', 'a[href*="/pdf"]', 'a[title*="PDF" i]'],
    "acs": ['a[href*="/doi/pdf/"]', "a.article-pdfLink", 'a[title*="PDF" i]', "a.btn-pdf"],
    "wiley": ['a[href*="/doi/pdfdirect/"]', 'a[href*="/doi/pdf/"]', "a.show-pdf-link"],
    "iop": ['a[href*="/article/"][href$="/pdf"]', "a#pdf-link-0", "a.btn-pdf"],
    "rsc": ['a[href*="/articlepdf/"]', 'a[title*="PDF" i]', "a.btn-pdf"],
    "science": ['a[href*="/doi/pdf/"]', 'a[title*="PDF" i]', "a.download-pdf"],
    "acm": ['a[href*="/doi/pdf/"]', 'a[title*="PDF" i]', "a.download-pdf"],
    "aip": ['a[href*="/article-pdf/"]', 'a[href*="/pdf/"]', 'a[title*="PDF" i]'],
    "springer": ['a[href*=".pdf"]', 'a[title*="PDF" i]'],
    "elsevier": ['a[href*="pdfft"]', 'a[href*="/pdf/"]', "a.download-pdf-link"],
    "nature": ['a[href*=".pdf"]', 'a[data-track-action="download pdf"]'],
    "aps": [
        'a[href*="/pdf/"]:not([href*="#"])',
        '.tabs a[href*="/pdf/"]',
        '.article-tools a[href*="/pdf/"]',
        'a[title*="PDF" i]',
    ],
    "oup": ['a[href*="/article-pdf/"]', '.article-tools a[href*="pdf"]', 'a[title*="PDF" i]'],
    "ieee": ['a[href*="stamp.jsp"]', 'a[href*="getPDF"]', 'a[title*="PDF" i]'],
    "pnas": ['a[href*="/doi/pdf/"]', "a.download-pdf"],
    "annualreviews": ['a[href*="/doi/pdf/"]', "a.download-pdf"],
}

_GENERIC_SELECTORS = (
    "a[href*='/pdf/']:has-text('PDF')",
    "a:has-text('PDF'):not(:has-text('Export')):not(:has-text('Supplemental'))",
    "a[data-type='pdf'], a.download-pdf",
)


def _maybe_pdf_response(resp) -> bool:
    """True when a Playwright response looks like a PDF.

    Playwright response headers are a plain dict, and the content type is the
    only reliable signal before the body is read. Kept deliberately small: it
    runs inside a response hook on every request of a page.
    """
    try:
        headers = getattr(resp, "headers", None) or {}
        content_type = str(headers.get("content-type") or "").lower()
        if "application/pdf" in content_type:
            return True
        url = str(getattr(resp, "url", "") or "").lower()
        return url.endswith(".pdf") or "/article-pdf/" in url
    except Exception:
        return False


def _root_domain(host: str) -> str:
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


async def _click_pdf_button(page, publisher: str, prefix: str = "") -> bytes | None:
    """Find a PDF link on the page; navigate to it (handling CF) or click it."""
    btn = None
    for sel in _PUBLISHER_SELECTORS.get(publisher, []):
        try:
            btn = await page.query_selector(sel)
            if btn:
                logger.info("%sbutton via %s", prefix, sel[:50])
                break
        except Exception:
            continue
    if not btn:
        for sel in _GENERIC_SELECTORS:
            try:
                btn = await page.query_selector(sel)
                if btn:
                    logger.info("%sbutton via generic %s", prefix, sel[:50])
                    break
            except Exception:
                continue
    if not btn:
        logger.info("%sno PDF button found", prefix)
        return None

    try:
        href = await btn.get_attribute("href") or ""
        target = (await btn.get_attribute("target") or "").lower()
    except Exception:
        return None

    # target=_blank opens a new tab: the download event never fires in this
    # page context, so navigate here instead and let the route grab the PDF.
    if href:
        absolute = href if href.startswith("http") else urljoin(page.url, href)
        if target == "_blank":
            logger.info("%starget=_blank → navigating to %s", prefix, absolute[:100])
            return await _capture_pdf_route(page, absolute, prefix=f"{prefix}blank: ")
        if href.startswith("http") and _root_domain(
            urlparse(absolute).netloc.lower()
        ) != _root_domain(urlparse(page.url).netloc.lower()):
            # Off-domain "PDF" links are usually cited references, not the article.
            logger.info("%srejected off-domain link %s", prefix, absolute[:80])
            return None
        data = await _browser_fetch(page, absolute)
        if data:
            logger.info("%sfetched href: %d bytes", prefix, len(data))
            return data

    future: asyncio.Future = asyncio.Future()

    async def _on_dl(dl) -> None:
        if not future.done():
            future.set_result(dl)

    page.on("download", _on_dl)
    try:
        await asyncio.wait_for(btn.click(timeout=8000), timeout=10)
        await asyncio.sleep(2)
        try:
            dl = await asyncio.wait_for(future, timeout=6)
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            await dl.save_as(str(tmp_path))
            data = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)
            if _valid(data):
                logger.info("%sdownload event: %d bytes", prefix, len(data))
                return data
        except TimeoutError:
            pass

        await _with_cf_bypass(page, "", prefix=f"{prefix}post-click: ")
        for body in await _collect_responses(page):
            logger.info("%snetwork capture: %d bytes", prefix, len(body))
            return body
        data = await _browser_fetch(page, page.url)
        if data:
            logger.info("%sfetched current URL: %d bytes", prefix, len(data))
            return data
    except Exception as exc:
        logger.debug("%sclick error: %s", prefix, exc)
    finally:
        try:
            page.remove_listener("download", _on_dl)
        except Exception:
            pass
    return None


async def _elsevier_pdfft(page, prefix: str = "") -> bytes | None:
    """ScienceDirect: build the pdfft URL from PII + md5 on the article page."""
    pii_match = re.search(r"/pii/([A-Z]\d+)", page.url)
    if not pii_match:
        return None
    pii = pii_match.group(1)
    try:
        html = await page.content()
    except Exception:
        return None
    md5_match = re.search(r'md5["\'\s=:]+([a-f0-9]{32})', html, re.IGNORECASE)
    if not md5_match:
        logger.info("%sno md5 token on page", prefix)
        return None
    url = (
        f"https://www.sciencedirect.com/science/article/pii/{pii}"
        f"/pdfft?md5={md5_match.group(1)}&pid=1-s2.0-{pii}-main.pdf"
    )
    return await _capture_pdf_route(page, url, prefix=prefix)


async def _aps_pdf(page, doi: str, prefix: str = "") -> bytes | None:
    """APS: PDF links live on the abstract page behind a second CF layer."""
    for body in await _collect_responses(page):
        logger.info("%sarticle-page capture: %d bytes", prefix, len(body))
        return body

    try:
        html = await page.content()
    except Exception:
        html = ""
    pdf_url = None
    for pattern in (
        r'https?://journals\.aps\.org/[a-z]+/pdf/10\.1103/[^\s"\']+',
        r'href="(/[a-z]+/pdf/10\.1103/[^\s"\']+)"',
        r"/[a-z]+/pdf/10\.1103/[^\s\"'<>]+",
    ):
        m = re.search(pattern, html)
        if m:
            pdf_url = m.group(0)
            if pdf_url.startswith('href="'):
                pdf_url = pdf_url[6:-1]
            if pdf_url.startswith("/"):
                pdf_url = f"https://journals.aps.org{pdf_url}"
            break
    if not pdf_url and "/abstract/" in page.url:
        pdf_url = page.url.replace("/abstract/", "/pdf/")
    if not pdf_url:
        logger.info("%scould not derive PDF URL", prefix)
        return None

    data = await _click_pdf_button(page, "aps", prefix=f"{prefix}s0: ")
    if data:
        return data
    data = await _capture_pdf_route(page, pdf_url, doi, prefix=f"{prefix}pdf: ")
    if data:
        return data
    data = await _browser_fetch(page, pdf_url, timeout=25)
    if data:
        return data
    try:
        printed = await asyncio.wait_for(page.pdf(), timeout=15)
        if _valid(printed):
            logger.info("%sprinted page: %d bytes", prefix, len(printed))
            return printed
    except Exception:
        pass
    return None


async def _aip_pdf(page, prefix: str = "") -> bytes | None:
    """AIP: the PDF link is target=_blank, so extract the href and navigate."""
    href = ""
    try:
        href = await page.evaluate(
            """() => {
                for (const sel of ['a[href*="/article-pdf/"]', 'a[href$=".pdf"]',
                                   'a[href*="/pdf/"]']) {
                    for (const a of document.querySelectorAll(sel)) {
                        const h = a.getAttribute('href') || '';
                        if (h) return h;
                    }
                }
                return '';
            }"""
        )
    except Exception:
        pass
    if not href:
        try:
            html = await page.content()
            m = re.search(r'href="((?:/[a-zA-Z]+)+/article-pdf/[^"\s<>]+\.pdf)"', html)
            href = m.group(1) if m else ""
        except Exception:
            href = ""
    if not href:
        return None
    url = urljoin(page.url, href)
    if not url.startswith("http"):
        return None
    # AIP's article-pdf endpoint carries its own CF challenge.
    captured: list = []

    def _on_resp(resp) -> None:
        try:
            if resp.status < 400 and _maybe_pdf_response(resp):
                captured.append(resp)
        except Exception:
            pass

    page.on("response", _on_resp)
    try:
        await _with_cf_bypass(page, "", goto_url=url, prefix=prefix)
        await asyncio.sleep(2)
        for resp in captured:
            try:
                await asyncio.wait_for(resp.finished(), timeout=30)
                body = await resp.body()
            except Exception:
                continue
            if _valid(body):
                return body
        printed = await asyncio.wait_for(page.pdf(), timeout=15)
        return printed if _valid(printed) else None
    finally:
        try:
            page.remove_listener("response", _on_resp)
        except Exception:
            pass


async def _mdpi_two_level(page, prefix: str = "") -> bytes | None:
    """MDPI hides "Download PDF" behind a dropdown that needs a JS click."""
    trigger = await page.query_selector(
        'a:has-text("Download"):not(:has-text("PDF")), '
        'button:has-text("Download"):not(:has-text("PDF"))'
    )
    if trigger:
        try:
            await trigger.click()
            await asyncio.sleep(3)
        except Exception:
            pass

    future: asyncio.Future = asyncio.Future()

    async def _on_dl(dl) -> None:
        if not future.done():
            future.set_result(dl)

    page.on("download", _on_dl)
    try:
        clicked = await page.evaluate(
            """() => {
                for (const a of document.querySelectorAll('a')) {
                    if (a.textContent.trim() === 'Download PDF' && a.href) {
                        a.click();
                        return a.href;
                    }
                }
                return null;
            }"""
        )
        if not clicked:
            logger.info("%s'Download PDF' link not found", prefix)
            return None
        try:
            dl = await asyncio.wait_for(future, timeout=15)
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            await dl.save_as(str(tmp_path))
            data = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)
            return data if _valid(data) else None
        except TimeoutError:
            return None
    finally:
        try:
            page.remove_listener("download", _on_dl)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# Resolver
# ══════════════════════════════════════════════════════════════════════

_assets_lock = asyncio.Lock()
_assets_ready = False


async def ensure_assets() -> None:
    """Thread-offloaded, run-once version of :meth:`CamoufoxResolver._ensure_assets`."""
    global _assets_ready
    if _assets_ready:
        return
    async with _assets_lock:
        if _assets_ready:
            return
        await asyncio.to_thread(CamoufoxResolver._ensure_assets)
        _assets_ready = True


class CamoufoxResolver:
    """Drive a real browser to fetch a paywalled PDF."""

    name = "camoufox-browser"

    def applies(self, paper: Paper) -> bool:
        return settings.camoufox_enabled and bool(paper.doi)

    async def fetch(self, paper: Paper) -> bytes | None:
        try:
            from camoufox import AsyncCamoufox
        except ImportError:
            logger.warning("camoufox not installed — browser resolver disabled")
            return None

        _clear_proxy_env()
        await ensure_assets()
        headful = _setup_display()
        headless = settings.camoufox_headless
        if headless is None:  # "auto"
            headless = not headful

        publisher = detect_publisher(paper.doi)
        article_url, pdf_url = build_urls(paper.doi, publisher)
        logger.info("publisher=%s article=%s", publisher, article_url[:90])

        async with _browser_lock:
            logger.info("browser lock acquired (headless=%s)", headless)
            try:
                async with AsyncCamoufox(**_browser_config(headless)) as browser:
                    page = await browser.new_page()
                    page.on("crash", lambda: None)
                    page._pdf_responses = []
                    page._pdf_responses_checked = set()

                    def _on_resp(resp) -> None:
                        try:
                            if _maybe_pdf_response(resp) or "__cf_chl" in (resp.url or ""):
                                page._pdf_responses.append(resp)
                                # Bounded: without this the bucket (and the work
                                # per poll) grows for the whole session.
                                if len(page._pdf_responses) > _MAX_TRACKED_RESPONSES:
                                    del page._pdf_responses[0]
                                    page._pdf_responses_checked.clear()
                        except Exception:
                            pass

                    page.on("response", _on_resp)
                    try:
                        return await asyncio.wait_for(
                            self._run(page, paper, publisher, article_url, pdf_url),
                            timeout=settings.camoufox_budget,
                        )
                    except TimeoutError:
                        logger.warning(
                            "browser session exceeded %ss for %s",
                            settings.camoufox_budget,
                            paper.label,
                        )
                        return None
            except Exception as exc:
                logger.warning("browser session failed: %s: %s", type(exc).__name__, exc)
                return None

    # ── orchestration inside one browser session ──────────────────────

    async def _run(
        self,
        page,
        paper: Paper,
        publisher: str | None,
        article_url: str,
        pdf_url: str,
    ) -> bytes | None:
        p = "  "

        # Fast path: publishers whose PDF endpoint needs no session cookies.
        if pdf_url and publisher in ("nature", "springer", "iop"):
            data = await _capture_pdf_route(page, pdf_url, paper.doi, prefix=f"{p}direct: ")
            if data:
                return data

        await _with_cf_bypass(page, paper.doi, goto_url=article_url, prefix=p)
        logger.info("%sarticle page: %s", p, page.url[:110])
        # The DOM can still be in flux right after a challenge clears, and
        # some publishers intermittently hand back a title-only stub.
        await _ensure_content(page, article_url, paper.doi, prefix=p)
        try:
            await asyncio.wait_for(_dismiss_popups(page), timeout=5)
        except Exception:
            pass

        # A viewer page (Science's /doi/epdf/) may already have pulled the PDF
        # down while we were waiting. Check before hunting for buttons.
        for body in await _collect_responses(page):
            logger.info("%scaptured during navigation: %d bytes", p, len(body))
            return body

        # No URL from the publisher table? Ask the page itself. AIP in
        # particular publishes `citation_pdf_url` but renders no <a> download
        # link at all when the session is not signed in, so a link-only hunt
        # reports "no PDF" while the URL is right there in <head>.
        if not pdf_url:
            meta_url = await _meta_pdf_url(page)
            if meta_url:
                pdf_url = meta_url
                logger.info("%spdf url from page metadata: %s", p, meta_url[:110])
            else:
                logger.debug("%sno citation_pdf_url meta on this page", p)

        # ACS / Wiley: the PDF endpoint needs cookies from the article page,
        # and is separately CF-protected.
        if pdf_url and publisher in ("acs", "wiley"):
            data = await _capture_pdf_route(
                page, pdf_url, paper.doi, prefix=f"{p}post-login: "
            )
            if data:
                return data
            for body in await _collect_responses(page):
                return body

        if publisher == "ieee" and not pdf_url:
            m = re.search(r"/document/(\d+)", page.url)
            if m:
                pdf_url = (
                    "https://ieeexplore.ieee.org/stampPDF/"
                    f"getPDF.jsp?tp=&arnumber={m.group(1)}"
                )

        # Publisher-specific tricks first, but NEVER `return` from them: a
        # failed trick used to abort the whole run, so we never fell through to
        # the generic paths (direct PDF url, metadata, click, navigation). MDPI
        # and APS both lost downloads that way.
        if publisher == "aps":
            await asyncio.sleep(2)
            data = await _aps_pdf(page, paper.doi, prefix=p)
            if data:
                return data

        if publisher == "mdpi":
            data = await _mdpi_two_level(page, prefix=f"{p}s5: ")
            if data:
                return data

        if pdf_url:
            if publisher == "aip":
                data = await _aip_pdf(page, prefix=f"{p}s1a: ")
                if data:
                    return data
            data = await _browser_fetch(page, pdf_url)
            if data:
                logger.info("%sfetched pdf_url: %d bytes", p, len(data))
                return data
            # In-page fetch can be blocked by CORS; fall back to the browser
            # context's own request API (same cookies, no CORS), then to a
            # navigation with route() interception.
            data = await _api_get(page, pdf_url)
            if data:
                logger.info("%sapi_get pdf_url: %d bytes", p, len(data))
                return data
            data = await _capture_pdf_route(page, pdf_url, paper.doi, prefix=f"{p}nav: ")
            if data:
                logger.info("%scaptured via navigation: %d bytes", p, len(data))
                return data

        if publisher != "aps":
            data = await _click_pdf_button(page, publisher or "", prefix=f"{p}s0: ")
            if data:
                return data

        if publisher == "elsevier":
            return await _elsevier_pdfft(page, prefix=f"{p}s0e: ")

        for body in await _collect_responses(page):
            logger.info("%slate network capture: %d bytes", p, len(body))
            return body

        # The page may already BE the PDF: several publishers (Acta Physica
        # Polonica, J-STAGE) navigate straight to the PDF URL and let Firefox
        # render it in its built-in viewer. There is no response left to
        # intercept in that state, so re-request the current URL with the
        # browser context's cookies.
        current = page.url or ""
        if _PDF_URL_RE.search(current):
            for cand in _pdf_candidates(current):
                data = await _api_get(page, cand, timeout=30.0)
                if data:
                    logger.info("%spage is the PDF; api_get %s: %d bytes",
                                p, cand.rsplit("/", 1)[-1][:24], len(data))
                    return data

        # Last chance: a viewer may only finish fetching the PDF once we stop
        # interacting with the page. Cheap to wait, and it recovers cases that
        # would otherwise look like a hard block.
        if await _await_pdf_response(page, timeout=20.0):
            for body in await _collect_responses(page):
                logger.info("%slate PDF response: %d bytes", p, len(body))
                return body

        # Nothing worked. Say *why*, in one line, so the next person does not
        # have to run a diagnostic browser session to find out.
        logger.info("%sno PDF obtained — %s", p, await _describe_page(page))
        return None

    @staticmethod
    def _ensure_assets() -> None:
        """Make sure the browser binary and GeoIP DB are present.

        Blocking (it may download), so callers must use :func:`ensure_assets`,
        which runs it in a thread -- otherwise a cold cache would stall the
        whole event loop and with it every other request.
        """
        try:
            from camoufox.pkgman import camoufox_path

            camoufox_path()
        except Exception as exc:
            logger.debug("camoufox_path() failed: %s", exc)
        try:
            from camoufox.locale import MMDB_FILE, download_mmdb

            if not MMDB_FILE.exists():
                download_mmdb()
        except Exception as exc:
            logger.debug("mmdb check/download failed: %s", exc)
