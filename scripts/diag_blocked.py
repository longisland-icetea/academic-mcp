#!/usr/bin/env python3
"""Why is this DOI blocked?  Capture screenshot + full evidence.

For a DOI we cannot download, dump everything needed to tell "wrong URL" from
"bot detection" apart:

  * screenshot  -> /tmp/blocked-diag/<tag>.png   (open it and look)
  * final URL, HTTP status, response headers     (bot-protection vendors
    announce themselves here: cf-ray, _abck, x-datadome, server: Akamai...)
  * full body text (not truncated)
  * cookies, and any known challenge markers

Usage:
    .venv/bin/python tools/diag_blocked.py 10.1063/5.0013092 [more dois...]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from academic_mcp.resolvers import camoufox as cf  # noqa: E402

# Match the systemd unit: Camoufox puts its browser under
# user_cache_dir("camoufox"), which honours XDG_CACHE_HOME. Without this,
# running a tool by hand uses ~/.cache/camoufox (empty here) and Camoufox
# re-downloads the addon — which currently fails with HTTP 451 — leaving a
# half-extracted addon that then breaks every launch.
_CACHE_HOME = Path.home() / ".local/state/academic-mcp/cache"
if _CACHE_HOME.is_dir():
    os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_HOME))


OUT = Path("/tmp/blocked-diag")
OUT.mkdir(parents=True, exist_ok=True)

PROBE = """() => {
    const out = {};
    out.url = location.href;
    out.title = document.title || '';
    out.html_len = (document.documentElement.outerHTML || '').length;
    out.body_html_len = document.body ? document.body.innerHTML.length : -1;
    out.body_text_len = document.body ? document.body.innerText.length : -1;
    out.body_text = document.body
        ? document.body.innerText.replace(/\\s+/g, ' ').slice(0, 2500) : '';
    out.n_a = document.querySelectorAll('a').length;
    out.noscript = document.querySelector('noscript')
        ? (document.querySelector('noscript').textContent || '').slice(0, 200) : '';
    // challenge / protection markers
    const html = (document.documentElement.outerHTML || '').toLowerCase();
    out.markers = {
      cf_chl: html.includes('__cf_chl'),
      cf_turnstile: html.includes('cf-turnstile') || html.includes('challenges.cloudflare'),
      cf_ray: /cf-ray|cloudflare/i.test(html),
      akamai: html.includes('_abck') || html.includes('akam'),
      datadome: html.includes('datadome'),
      imperva: html.includes('imperva') || html.includes('incapsula'),
      captcha: html.includes('captcha'),
      recaptcha: html.includes('recaptcha'),
      access_denied: /access denied|permission denied|not authorized/i.test(out.body_text),
      login: /log in|sign in|login required|institutional access/i.test(out.body_text),
      purchase: /purchase|buy this article|rent this article|paywall/i.test(out.body_text),
      bot: /are you a robot|bot detect|unusual traffic|verify you are (a )?human/i.test(out.body_text),
    };
    out.cookies = document.cookie ? document.cookie.slice(0, 600) : '';
    // Show the markup around whatever is deciding access, so we can tell a
    // paywall skeleton from a login wall from a JS-rendered page.
    out.snippets = [];
    for (const kw of ['sign in', 'access', 'pdf', 'purchase', 'institution']) {
      const re = new RegExp(kw, 'gi');
      let m;
      let n = 0;
      while ((m = re.exec(html)) !== null && n < 2) {
        const st = Math.max(0, m.index - 120);
        out.snippets.push({kw, ctx: html.slice(st, m.index + 220)});
        n++;
      }
    }
    out.jsonld = [];
    for (const sc of document.querySelectorAll('script[type="application/ld+json"]')) {
      out.jsonld.push((sc.textContent || '').slice(0, 300));
    }
    out.pdf_links = [];
    for (const a of document.querySelectorAll('a')) {
      const h = (a.getAttribute('href') || '');
      if (/pdf|download/i.test(h)) out.pdf_links.push(h.slice(0, 120));
      if (out.pdf_links.length >= 10) break;
    }
    return out;
}"""


async def _shoot(page, path: Path) -> str:
    """Screenshot a page that never finishes loading its webfonts.

    Playwright's screenshot waits for font loading, and on these blocked
    publisher pages that wait never completes (30 s timeout, every time).
    Aborting font requests makes the wait terminate; CDP is the fallback
    because it has no font wait at all.
    """
    try:
        async def _kill_fonts(route):
            if route.request.resource_type in ("font", "stylesheet"):
                await route.abort()
            else:
                await route.continue_()
        await page.route("**/*", _kill_fonts)
    except Exception:
        pass
    try:
        blob = await page.screenshot(type="png", animations="disabled", timeout=15000)
        path.write_bytes(blob)
        return f"{path} ({len(blob)} bytes)"
    except Exception as exc:
        first = f"{type(exc).__name__}: {str(exc)[:80]}"
    try:
        await page.unroute("**/*")
    except Exception:
        pass
    try:
        cdp = await page.context.new_cdp_session(page)
        res = await cdp.send("Page.captureScreenshot", {"format": "png"})
        import base64
        blob = base64.b64decode(res["data"])
        path.write_bytes(blob)
        return f"{path} ({len(blob)} bytes, via cdp; playwright said {first})"
    except Exception as exc:
        return f"failed: {first} | cdp: {type(exc).__name__}: {str(exc)[:80]}"


async def probe(page, doi: str, warm: bool) -> dict:
    tag = doi.replace("/", "_") + ("-warm" if warm else "-cold")
    url = f"https://doi.org/{doi}"
    if warm:
        # Establish a session on the publisher's own domain first: some sites
        # only serve content to requests that carry a referer from themselves.
        host = "https://pubs.aip.org" if doi.startswith("10.1063") else "https://pubs.rsc.org"
        try:
            await cf._with_cf_bypass(page, goto_url=host, prefix="  warm: ")
            await asyncio.sleep(3)
        except Exception as exc:
            print(f"  warm-up nav failed: {type(exc).__name__}")
    try:
        resp = await page.goto(url, wait_until="commit", timeout=30000)
        status = resp.status if resp else None
        headers = dict(resp.headers) if resp else {}
    except Exception as exc:
        status, headers = None, {"error": f"{type(exc).__name__}: {exc}"}

    await cf._with_cf_bypass(page, prefix="  ")
    await asyncio.sleep(6)
    info = await page.evaluate(PROBE)
    info["http_status"] = status
    info["headers"] = {
        k: v for k, v in headers.items()
        if k.lower() in ("server", "cf-ray", "cf-mitigated", "x-datadome",
                        "x-cache", "content-type", "set-cookie", "error")
    }
    try:
        info["screenshot"] = await _shoot(page, OUT / f"{tag}.png")
    except Exception as exc:
        info["screenshot"] = f"failed: {type(exc).__name__}: {exc}"
    # Also record what the page says the PDF url is, when it publishes one.
    try:
        info["citation_pdf_url"] = await page.evaluate(
            "() => (document.querySelector('meta[name=\"citation_pdf_url\"]') || {}).content || ''"
        )
    except Exception:
        info["citation_pdf_url"] = ""
    return info


async def main() -> None:
    from camoufox import AsyncCamoufox

    dois = sys.argv[1:] or ["10.1063/5.0013092", "10.1039/d5cp03472h"]
    cf._setup_display()
    cf._clear_proxy_env()
    cf.CamoufoxResolver._ensure_assets()  # sync; fine in a CLI

    report = []
    async with AsyncCamoufox(**cf._browser_config(headless=False)) as browser:
        for doi in dois:
            for warm in (False, True):
                page = await browser.new_page()
                print(f"\n{'='*70}\n{doi}  warm={warm}\n{'='*70}")
                try:
                    info = await probe(page, doi, warm)
                except Exception as exc:
                    info = {"doi": doi, "warm": warm, "error": f"{type(exc).__name__}: {exc}"}
                info["doi"] = doi
                info["warm"] = warm
                report.append(info)
                for k in ("url", "http_status", "title", "body_text_len",
                          "body_html_len", "n_a", "citation_pdf_url", "screenshot"):
                    print(f"  {k}: {info.get(k)}")
                print(f"  headers: {json.dumps(info.get('headers', {}))[:300]}")
                mk = {k: v for k, v in (info.get("markers") or {}).items() if v}
                print(f"  markers: {mk or '(none)'}")
                print(f"  pdf_links: {info.get('pdf_links')}")
                print(f"  text: {info.get('body_text', '')[:600]!r}")
                await page.close()

    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nreport + screenshots: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
