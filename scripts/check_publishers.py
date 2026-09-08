#!/usr/bin/env python3
"""Which publisher prefixes actually appear in the corpus, and does
detect_publisher() recognise them?

Feeds a list of DOIs (one per line) through detect_publisher() / build_urls()
and reports coverage, so gaps in the table are visible instead of silently
falling through to the generic doi.org path.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from academic_mcp.resolvers.camoufox import build_urls, detect_publisher  # noqa: E402

KNOWN = {
    "10.1038": "Nature family",
    "10.1103": "APS (Phys. Rev.)",
    "10.1021": "ACS",
    "10.1016": "Elsevier",
    "10.1088": "IOP",
    "10.1126": "Science / AAAS",
    "10.21468": "SciPost",
    "10.1002": "Wiley",
    "10.1063": "AIP",
    "10.1007": "Springer",
    "10.3390": "MDPI",
    "10.1073": "PNAS",
    "10.1039": "RSC",
    "10.1146": "Annual Reviews",
    "10.1109": "IEEE",
    "10.1093": "OUP",
    "10.1080": "Taylor & Francis",
    "10.1051": "EDP Sciences",
    "10.1143": "JPSJ (old)",
    "10.7566": "JPSJ (current)",
    "10.12693": "από APH N.S.",
    "10.1590": "SciELO",
    "10.1609": "AAAI",
    "10.48550": "arXiv",
    "10.1186": "BMC / Springer",
    "10.1515": "De Gruyter",
    "10.3379": "Jpn. J. Appl. Phys. (old)",
    "10.4230": "Dagstuhl (LIPIcs)",
}


def main() -> None:
    dois = [
        line.strip()
        for line in Path(sys.argv[1] if len(sys.argv) > 1 else "-").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip()
    ]
    # Keep only things that look like DOIs (the corpus has Scopus "W..." ids too).
    dois = [d for d in dois if d.startswith("10.") and "/" in d]

    prefixes = Counter(d.split("/", 1)[0] for d in dois)

    covered: list[tuple[str, int, str, str]] = []
    uncovered: list[tuple[str, int, str]] = []
    for prefix, n in prefixes.most_common():
        name = KNOWN.get(prefix, "?")
        sample = next(d for d in dois if d.startswith(prefix + "/"))
        pub = detect_publisher(sample)
        if pub:
            article, pdf = build_urls(sample, pub)
            covered.append((prefix, n, name, pub))
        else:
            uncovered.append((prefix, n, name))

    print(f"{len(dois)} DOIs, {len(prefixes)} prefixes\n")
    print("== RECOGNISED ==")
    for prefix, n, name, pub in covered:
        print(f"  {prefix:<12} n={n:<4} {name:<22} -> {pub}")
    print("\n== NOT RECOGNISED (falls through to doi.org generic path) ==")
    for prefix, n, name in uncovered:
        print(f"  {prefix:<12} n={n:<4} {name}")
    n_cov = sum(n for _, n, _, _ in covered)
    n_unc = sum(n for _, n, _ in uncovered)
    print(f"\ncoverage: {len(covered)}/{len(prefixes)} prefixes, {n_cov}/{len(dois)} DOIs")
    if uncovered:
        print(f"untouched DOIs: {n_unc} ({n_unc / len(dois) * 100:.0f}%)")


if __name__ == "__main__":
    main()
