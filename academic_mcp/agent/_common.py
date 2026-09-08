"""Common utilities shared across academic-search scripts.

Centralizes functions that were previously duplicated across
memory.py / search.py / download.py, causing key-format drift
(see audit report P0-5).

Import as: `sys.path.insert(0, str(Path(__file__).parent)); import _common`
or: `from _common import doi_to_key` after sys.path setup.
"""
import re
from pathlib import Path


def normalize_doi(doi: str) -> str:
    """Strip URL prefix + trailing punctuation; lowercase the prefix.

    >>> normalize_doi("HTTPS://DOI.ORG/10.1038/foo")
    '10.1038/foo'
    >>> normalize_doi("10.1038/foo.")
    '10.1038/foo'
    """
    if not doi:
        return ""
    s = doi.strip()
    lower = s.lower()
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if lower.startswith(prefix):
            s = s[len(prefix):]
            lower = s.lower()
            break
    # Strip a single trailing punctuation char
    s = re.sub(r'[.,;)\]\}]+$', '', s)
    # Lowercase only the prefix (10.XXXX/)
    if "/" in s:
        prefix, suffix = s.split("/", 1)
        s = prefix.lower() + "/" + suffix
    else:
        s = s.lower()
    return s


def doi_to_key(doi: str) -> str:
    """Convert DOI to filesystem-safe key.

    >>> doi_to_key("10.1038/s42005-026-02497-8")
    '10.1038_s42005-026-02497-8'
    >>> doi_to_key("10.1038/foo<bar>")
    '10.1038_foo_bar_'
    """
    clean = normalize_doi(doi)
    if not clean:
        return ""
    safe = clean.replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe = re.sub(r'[<>:"|?*]', '_', safe)
    return safe


def legacy_doi_to_key(doi: str) -> str:
    """The pre-2026-09-01 key format used in old paper_library entries.

    Differs from doi_to_key() in that it does NOT substitute `<>:"|?*`,
    so old entries written before this migration may have keys with these
    characters. Use doi_to_key() for new keys; use this for backward-lookup.
    """
    clean = normalize_doi(doi)
    if not clean:
        return ""
    return clean.replace("/", "_").replace("\\", "_").replace(" ", "_")


def make_legacy_keys(doi: str) -> list:
    """Return both old-style and new-style keys for backward lookup."""
    if not doi:
        return []
    return [k for k in {legacy_doi_to_key(doi), doi_to_key(doi)} if k]


def short_dedup_key(title: str, year) -> str:
    """Fallback key when DOI is missing: <first title word>_<year>."""
    if not title:
        return ""
    first = re.sub(r'[^A-Za-z0-9]+', '', title.split()[0])[:20] if title.split() else "untitled"
    return f"{first}_{year}" if year else first

# ── .env parsing (P1-Z: shared helper) ─────────────────────────────────

def read_env_key(paths: list[Path], key: str) -> str | None:
    """Read a single KEY from the first existing .env file in ``paths``.

    Strips whitespace, surrounding quotes, and trailing inline comments.
    Returns None if no file found or KEY absent.

    Replaces 3 copies of the same parse logic across search.py /
    download.py / memory.py / snowball.py.
    """
    import re as _re
    for p in paths:
        if not p.exists():
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() != key:
                continue
            v = v.strip()
            # Strip surrounding quotes
            if (v.startswith('"') and v.endswith('"')) or (
                v.startswith("'") and v.endswith("'")
            ):
                v = v[1:-1]
            # Strip inline comment
            v = _re.sub(r"\s+#.*$", "", v)
            return v.strip() or None
    return None
