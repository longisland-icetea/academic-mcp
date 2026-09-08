#!/usr/bin/env python3
"""Live cross-publisher smoke test of the download resolvers.

Uses fetch_paper_pdf (no MinerU conversion) so the test exercises download +
Cloudflare handling without burning conversion quota.

    .venv/bin/python tools/test_publishers.py dois.txt [out.json]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging  # noqa: E402

from academic_mcp import pipeline  # noqa: E402
from academic_mcp.resolvers.camoufox import build_urls, detect_publisher  # noqa: E402


async def one(doi: str) -> dict:
    t0 = time.monotonic()
    try:
        out = await pipeline.get_pdf(doi=doi)
    except Exception as exc:  # noqa: BLE001
        return {"doi": doi, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    rec = out.as_dict()
    rec["seconds"] = round(time.monotonic() - t0, 1)
    rec.pop("text", None)
    return rec


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("asyncio", "httpx", "httpcore", "primp", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/pub-test.json")
    dois = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    results = []
    for doi in dois:
        pub = detect_publisher(doi)
        article, pdf = build_urls(doi, pub) if pub else (f"https://doi.org/{doi}", "")
        r = await one(doi)
        r["publisher"] = pub or "(none)"
        r["article_url"] = article
        r["pdf_url"] = pdf
        results.append(r)
        status = "OK  " if r.get("ok") else "FAIL"
        print(
            f"{status} {doi:<38} pub={r['publisher']:<13} "
            f"strategy={r.get('strategy', '-'):<18} {r.get('seconds')}s",
            flush=True,
        )
        if not r.get("ok"):
            print(f"     attempts={json.dumps(r.get('attempts'), ensure_ascii=False)[:300]}",
                  flush=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{ok}/{len(results)} succeeded -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
