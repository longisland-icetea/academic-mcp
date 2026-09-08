"""Post-processing for converted Markdown.

MinerU output is already clean; what remains is stripping residual image
placeholders (from converter versions that emit them) and collapsing blank
runs so long papers do not burn context on whitespace.
"""

from __future__ import annotations

import re

_BLANK_RUN = re.compile(r"\n{3,}")


def clean_markdown(text: str) -> str:
    lines: list[str] = []
    in_picture = False
    picture_lines = 0

    for line in text.split("\n"):
        lowered = line.strip().lower()

        # Single-line placeholder: **==> picture [...] intentionally omitted <==**
        if "==> picture" in lowered and "intentionally omitted" in lowered and "<==" in lowered:
            continue

        # Multi-line picture region
        if "picture omitted" in lowered or "start of picture" in lowered:
            in_picture = True
            picture_lines = 0
            continue
        if in_picture and "end of picture" in lowered:
            in_picture = False
            continue
        if in_picture:
            picture_lines += 1
            if picture_lines <= 3:
                continue
            in_picture = False  # keep content beyond a short block

        lines.append(line)

    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip() + "\n"
