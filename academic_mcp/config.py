"""Configuration for the academic-mcp server.

All settings come from environment variables, with an optional fallback to
``<MCP_HOME>/.env`` — this server's own directory — and then ``~/.shellrc``.

Only settings that matter for *downloading and converting papers* live here —
the FastAPI monolith on wsl-zz-desktop carried ~60 unrelated settings (LLM
tiers, session DB, CORS, JWT, ...) which are deliberately not ported.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# This server's own directory — where its configuration file lives. The
# service no longer reads anything from a client's tree (it used to read the
# academic-search skill's .env, which coupled it to one particular agent).
MCP_HOME = Path(
    os.environ.get(
        "ACADEMIC_MCP_HOME",
        Path(__file__).resolve().parents[1],
    )
).expanduser()

# Files consulted for missing keys, in order.  Values already present in the
# process environment always win.
ENV_FILES = (
    MCP_HOME / ".env",
    Path.home() / ".shellrc",
)


def _load_env_files() -> dict[str, str]:
    """Return KEY=value pairs from the first file that defines each key."""
    out: dict[str, str] = {}
    for path in ENV_FILES:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            # .shellrc lines are `export KEY="value"` — strip the export.
            key = key.removeprefix("export ").strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            val = re.sub(r"\s+#.*$", "", val).strip()
            if val and key not in out:
                out[key] = val
    return out


_FILE_ENV = _load_env_files()


def env(name: str, default: str = "") -> str:
    return os.environ.get(name) or _FILE_ENV.get(name) or default


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        raw = _FILE_ENV.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    raw = env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _normalise_proxy(raw: str) -> str | None:
    """Treat empty / 'none' / 'off' as 'no proxy'."""
    raw = (raw or "").strip()
    if not raw or raw.lower() in ("none", "off", "false", "0"):
        return None
    return raw


@dataclass
class Settings:
    # ── Storage ────────────────────────────────────────────────────────
    # Reuses the academic-search skill's data dir so the existing cache of
    # ~235 converted papers stays valid after the migration.
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.path.expanduser(env("ACADEMIC_DATA_DIR", str(MCP_HOME / "data")))
        )
    )

    # ── HTTP ───────────────────────────────────────────────────────────
    http_timeout: float = field(default_factory=lambda: env_float("ACADEMIC_HTTP_TIMEOUT", 45.0))
    http_connect_timeout: float = field(
        default_factory=lambda: env_float("ACADEMIC_HTTP_CONNECT_TIMEOUT", 15.0)
    )
    # Hard cap on a single response body. A 200 response is buffered whole, so
    # an unexpectedly large (or endless) body would otherwise sit in memory --
    # and a dozen concurrent fetches is all it takes to exhaust it.
    http_max_bytes: int = field(
        default_factory=lambda: env_int("ACADEMIC_HTTP_MAX_BYTES", 200 * 1024 * 1024)
    )

    # -- Deadlines -------------------------------------------------------
    # Total budget for "resolve + download one paper". Without this a stalled
    # resolver holds its per-key lock indefinitely.
    download_budget: float = field(
        default_factory=lambda: env_float("ACADEMIC_DOWNLOAD_BUDGET", 420.0)
    )

    # ── MinerU cloud API (the ONLY PDF→Markdown path) ──────────────────
    mineru_token: str = field(default_factory=lambda: env("MINERU_TOKEN") or env("MINERU_API_KEY"))
    mineru_api_base: str = field(
        default_factory=lambda: env("MINERU_API_BASE", "https://mineru.net/api/v4").rstrip("/")
    )
    # vlm = best quality (formulas/tables/layout); pipeline = faster, cheaper.
    mineru_model: str = field(default_factory=lambda: env("MINERU_MODEL", "vlm"))
    mineru_language: str = field(default_factory=lambda: env("MINERU_LANGUAGE", "en"))
    mineru_timeout: int = field(default_factory=lambda: env_int("MINERU_TIMEOUT", 900))
    mineru_poll_interval: float = field(default_factory=lambda: env_float("MINERU_POLL_INTERVAL", 4.0))

    # ── Elsevier / ScienceDirect API ───────────────────────────────────
    elsevier_api_key: str = field(default_factory=lambda: env("ELSEVIER_API_KEY"))
    elsevier_insttoken: str = field(default_factory=lambda: env("ELSEVIER_INSTTOKEN"))

    # ── Literature search (Scopus + OpenAlex) ──────────────────────────
    openalex_api_key: str = field(default_factory=lambda: env("OPENALEX_API_KEY"))
    # Budget for one search request across every engine.
    search_timeout: float = field(default_factory=lambda: env_float("ACADEMIC_SEARCH_TIMEOUT", 120.0))

    # ── Proxies ────────────────────────────────────────────────────────
    # download_proxy: used for plain-HTTP publisher downloads.
    download_proxy: str | None = field(
        default_factory=lambda: _normalise_proxy(env("DOWNLOAD_PROXY"))
    )
    # "primary"  – always go through the proxy
    # "fallback" – direct first, proxy on timeout/network error
    # "none"     – never (except arXiv, which is not paywalled)
    download_proxy_mode: str = field(
        default_factory=lambda: (env("DOWNLOAD_PROXY_MODE", "fallback").lower())
    )
    # gfw_proxy: used for arXiv / search-engine fallbacks.
    gfw_proxy: str | None = field(default_factory=lambda: _normalise_proxy(env("GFW_PROXY")))

    # ── Camoufox browser automation ────────────────────────────────────
    camoufox_enabled: bool = field(default_factory=lambda: env_bool("CAMOUFOX_ENABLED", True))
    camoufox_timeout: int = field(default_factory=lambda: env_int("CAMOUFOX_TIMEOUT", 90))
    # None = auto: run headful when a display is available (Xvfb or WSLg),
    # headless otherwise.  Headful matters — headless Firefox is challenged
    # harder by CloudFlare and the Turnstile checkbox is unreliable headless.
    camoufox_headless: bool | None = field(
        default_factory=lambda: (
            None if env("CAMOUFOX_HEADLESS", "auto").lower() == "auto"
            else env_bool("CAMOUFOX_HEADLESS", False)
        )
    )
    # Hard ceiling for one browser session — protects the global browser lock.
    camoufox_budget: int = field(default_factory=lambda: env_int("CAMOUFOX_BUDGET", 240))

    # ── arXiv title search (DDGS) ──────────────────────────────────────
    arxiv_search_enabled: bool = field(default_factory=lambda: env_bool("ARXIV_SEARCH_ENABLED", True))
    # Enforced with asyncio.wait_for around the blocking DDGS call. DDGS runs
    # in a thread and can stall on the network; unwrapped, that would hold the
    # caller's lock forever and stall every other fetch behind it.
    arxiv_search_timeout: float = field(
        default_factory=lambda: env_float("ARXIV_SEARCH_TIMEOUT", 35.0)
    )

    # ── Session identity ───────────────────────────────────────────────
    # Fallback only: callers pass `session_id` explicitly on every tool call.
    session_id: str = field(default_factory=lambda: env("PI_SESSION"))

    # ── Server ─────────────────────────────────────────────────────────
    host: str = field(default_factory=lambda: env("ACADEMIC_MCP_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: env_int("ACADEMIC_MCP_PORT", 8790))
    log_level: str = field(default_factory=lambda: env("ACADEMIC_MCP_LOG_LEVEL", "INFO").upper())
    log_file: str | None = field(default_factory=lambda: env("ACADEMIC_MCP_LOG_FILE") or None)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def md_dir(self) -> Path:
        return self.data_dir / "texts"

    @property
    def search_cache_dir(self) -> Path:
        return self.data_dir / "search_cache"

    def ensure_dirs(self) -> None:
        for d in (self.pdf_dir, self.md_dir):
            d.mkdir(parents=True, exist_ok=True)

    def as_public_dict(self) -> dict:
        """Settings safe to expose over the wire (no secrets)."""

        def _mask(value: str | None) -> str:
            if not value:
                return ""
            return f"***{value[-4:]}" if len(value) > 8 else "***"

        return {
            "data_dir": str(self.data_dir),
            "pdf_dir": str(self.pdf_dir),
            "md_dir": str(self.md_dir),
            "mineru_configured": bool(self.mineru_token),
            "mineru_api_base": self.mineru_api_base,
            "mineru_model": self.mineru_model,
            "elsevier_configured": bool(self.elsevier_api_key),
            "openalex_configured": bool(self.openalex_api_key),
            "download_proxy": _mask(self.download_proxy) if self.download_proxy else None,
            "download_proxy_mode": self.download_proxy_mode,
            "gfw_proxy": _mask(self.gfw_proxy) if self.gfw_proxy else None,
            "camoufox_enabled": self.camoufox_enabled,
            "camoufox_timeout": self.camoufox_timeout,
            "arxiv_search_enabled": self.arxiv_search_enabled,
        }


settings = Settings()
