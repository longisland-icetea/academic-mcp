"""Token lifecycle tests: expiry, precedence, storage, and the 401 retry.

Offline by design — no browser, no network, no MinerU quota. The browser login
itself cannot be tested here (it needs credentials and a captcha), so what is
pinned is everything around it: what counts as expired, which token wins, how a
401 triggers exactly one renewal, and that a renewed token is never shadowed by
an older one.
"""

from __future__ import annotations

import base64
import json
import stat
import time

import httpx
import pytest

from academic_mcp import mineru, mineru_auth
from academic_mcp.config import settings


def make_token(exp_offset_days: float = 30, **extra) -> str:
    """A structurally valid JWT whose exp is relative to now."""
    header = base64.urlsafe_b64encode(b'{"alg":"HS512","typ":"JWT"}').rstrip(b"=")
    payload = {"exp": int(time.time() + exp_offset_days * 86400), "jti": "test", **extra}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    return f"{header.decode()}.{body.decode()}.signature"


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    """Keep every test off the real token file and the module-level cache."""
    monkeypatch.setattr(settings, "mineru_token_file", tmp_path / "state" / "mineru_token.json")
    mineru_auth._CACHE = None
    yield
    mineru_auth._CACHE = None


# ══════════════════════════════════════════════════════════════════════
# Expiry
# ══════════════════════════════════════════════════════════════════════


def test_expiry_is_read_from_the_jwt():
    exp = mineru_auth.expiry_of(make_token(30))
    assert exp is not None
    assert 29 < (exp.timestamp() - time.time()) / 86400 < 31


def test_opaque_or_broken_tokens_have_no_expiry():
    assert mineru_auth.expiry_of("not-a-jwt") is None
    assert mineru_auth.expiry_of("a.b.c") is None
    assert mineru_auth.expiry_of("") is None


def test_expired_and_soon_to_expire_are_distinguished():
    expired = mineru_auth.TokenInfo(make_token(-1), mineru_auth.expiry_of(make_token(-1)), "env")
    soon = mineru_auth.TokenInfo(make_token(2), mineru_auth.expiry_of(make_token(2)), "env")
    healthy = mineru_auth.TokenInfo(make_token(60), mineru_auth.expiry_of(make_token(60)), "env")

    assert expired.expired and expired.needs_renewal(7)
    assert not soon.expired and soon.needs_renewal(7)
    assert not healthy.needs_renewal(7)


def test_token_without_expiry_is_never_auto_renewed():
    unknown = mineru_auth.TokenInfo("opaque", None, "env")
    assert unknown.days_left is None
    assert not unknown.expired
    assert not unknown.needs_renewal(7)


# ══════════════════════════════════════════════════════════════════════
# Which token wins
# ══════════════════════════════════════════════════════════════════════


def _info(days: float, source: str) -> mineru_auth.TokenInfo:
    token = make_token(days)
    return mineru_auth.TokenInfo(token, mineru_auth.expiry_of(token), source)


def test_later_expiry_wins_between_env_and_state():
    # A hand-pasted fresh token in .env must beat a stale stored one …
    assert mineru_auth._pick(_info(60, "env"), _info(1, "state")).source == "env"
    # … and a renewal written to the state file must beat an expired .env value.
    assert mineru_auth._pick(_info(-3, "env"), _info(89, "state")).source == "state"


def test_missing_side_is_used_as_is():
    only_env = _info(10, "env")
    only_state = _info(10, "state")
    assert mineru_auth._pick(only_env, None) is only_env
    assert mineru_auth._pick(None, only_state) is only_state
    assert mineru_auth._pick(None, None) is None


def test_current_prefers_a_fresh_stored_token(monkeypatch):
    stale = make_token(-5)
    monkeypatch.setattr(settings, "mineru_token", stale)
    fresh = make_token(80)
    mineru_auth.write_state(
        mineru_auth.TokenInfo(fresh, mineru_auth.expiry_of(fresh), "state")
    )

    info = mineru_auth.current(refresh=True)
    assert info is not None
    assert info.token == fresh
    assert info.source == "state"


def test_current_prefers_a_manually_pasted_token(monkeypatch):
    """The failure mode that started this: a fresh .env token must win."""
    monkeypatch.setattr(settings, "mineru_token", make_token(80))
    stale = make_token(-5)
    mineru_auth.write_state(
        mineru_auth.TokenInfo(stale, mineru_auth.expiry_of(stale), "state")
    )

    info = mineru_auth.current(refresh=True)
    assert info is not None
    assert info.source == "env"
    assert info.token == settings.mineru_token


# ══════════════════════════════════════════════════════════════════════
# Storage
# ══════════════════════════════════════════════════════════════════════


def test_state_round_trip_is_private():
    token = make_token(45, account="a@b.c")
    written = mineru_auth.TokenInfo(token, mineru_auth.expiry_of(token), "state", "a@b.c")
    mineru_auth.write_state(written)

    read = mineru_auth.read_state()
    assert read is not None
    assert read.token == token
    assert read.account == "a@b.c"
    assert read.exp == written.exp

    path = mineru_auth.state_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_corrupt_state_is_ignored_not_fatal():
    path = mineru_auth.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert mineru_auth.read_state() is None


# ══════════════════════════════════════════════════════════════════════
# 401 handling
# ══════════════════════════════════════════════════════════════════════


def _response(status: int, token: str) -> httpx.Response:
    request = httpx.Request(
        "POST",
        "https://mineru.net/api/v4/file-urls/batch",
        headers={"Authorization": f"Bearer {token}"},
    )
    return httpx.Response(status, request=request)


def test_401_becomes_authexpired_carrying_the_failed_token():
    with pytest.raises(mineru.AuthExpired) as caught:
        mineru._check_token_failure(_response(401, "the-stale-token"))
    assert caught.value.token == "the-stale-token"
    assert caught.value.status == 401
    assert isinstance(caught.value, mineru.MineruError)


def test_200_is_not_a_token_failure():
    mineru._check_token_failure(_response(200, "whatever"))


def _stub_conversion(monkeypatch, tmp_path, failures: int):
    """Make convert_document run offline, failing the first `failures` calls."""
    monkeypatch.setattr(mineru, "DOC_CACHE_DIR", tmp_path / "docs")
    monkeypatch.setattr(settings, "mineru_token", "stale-token")
    mineru_auth._CACHE = None
    calls = {"create": 0, "renew": 0}

    def fake_create(client, **kwargs):
        calls["create"] += 1
        if calls["create"] <= failures:
            raise mineru.AuthExpired("rejected", token="stale-token", status=401)
        return "batch-1", []

    def fake_extract(blob, out_dir, stem):
        md = out_dir / f"{stem}.md"
        md.write_text("# converted", encoding="utf-8")
        assets = out_dir / f"{stem}_assets"
        assets.mkdir(parents=True, exist_ok=True)
        return md, assets

    def fake_renew(**kwargs):
        calls["renew"] += 1
        assert kwargs.get("stale") == "stale-token"
        return mineru_auth.TokenInfo("fresh-token", None, "state")

    monkeypatch.setattr(mineru, "_create_task", fake_create)
    monkeypatch.setattr(mineru, "_poll", lambda client, batch: {"full_zip_url": "https://x/y.zip"})
    monkeypatch.setattr(mineru, "_download", lambda url: b"zip-bytes")
    monkeypatch.setattr(mineru, "_extract_zip", fake_extract)
    monkeypatch.setattr(mineru_auth, "renew", fake_renew)
    return calls


async def test_a_401_renews_once_and_the_conversion_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "mineru_account", "someone@example.com")
    monkeypatch.setattr(settings, "mineru_password", "secret")
    calls = _stub_conversion(monkeypatch, tmp_path, failures=1)

    result = mineru.convert_document("https://example.com/paper.pdf", name="probe")

    assert calls == {"create": 2, "renew": 1}
    assert result.text.startswith("# converted")


async def test_renewal_is_not_attempted_twice(monkeypatch, tmp_path):
    """A second 401 after a successful renewal must surface, not loop."""
    monkeypatch.setattr(settings, "mineru_account", "someone@example.com")
    monkeypatch.setattr(settings, "mineru_password", "secret")
    calls = _stub_conversion(monkeypatch, tmp_path, failures=2)

    with pytest.raises(mineru.AuthExpired):
        mineru.convert_document("https://example.com/paper.pdf", name="probe")
    assert calls == {"create": 2, "renew": 1}


async def test_no_credentials_means_no_renewal_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "mineru_account", "")
    monkeypatch.setattr(settings, "mineru_password", "")
    calls = _stub_conversion(monkeypatch, tmp_path, failures=1)

    with pytest.raises(mineru.AuthExpired):
        mineru.convert_document("https://example.com/paper.pdf", name="probe")
    assert calls == {"create": 1, "renew": 0}


def test_renew_without_credentials_says_what_is_missing(monkeypatch):
    monkeypatch.setattr(settings, "mineru_account", "")
    monkeypatch.setattr(settings, "mineru_password", "")
    with pytest.raises(mineru_auth.MineruAuthError) as caught:
        mineru_auth.renew(force=True)
    assert "MINERU_ACCOUNT" in str(caught.value)


# ══════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════


def test_status_never_contains_the_token(monkeypatch):
    token = make_token(5)
    monkeypatch.setattr(settings, "mineru_token", token)
    mineru_auth._CACHE = None

    payload = mineru_auth.status()
    assert token not in json.dumps(payload)
    assert payload["days_left"] is not None
    assert payload["expires_at"]


async def test_health_exposes_token_state(monkeypatch):
    monkeypatch.setattr(settings, "mineru_token", make_token(3))
    mineru_auth._CACHE = None
    from academic_mcp import server

    payload = json.loads(await server.health())
    assert "mineru_token" in payload
    assert payload["mineru_token"]["needs_renewal"] is True


def test_ensure_renews_when_no_token_exists_at_all(monkeypatch):
    """Credentials alone must be enough — no MINERU_TOKEN required."""
    monkeypatch.setattr(settings, "mineru_token", "")
    monkeypatch.setattr(settings, "mineru_account", "someone@example.com")
    monkeypatch.setattr(settings, "mineru_password", "secret")
    mineru_auth._CACHE = None

    calls: list[dict] = []

    def fake_renew(**kwargs):
        calls.append(kwargs)
        return mineru_auth.TokenInfo("fresh", None, "state")

    monkeypatch.setattr(mineru_auth, "renew", fake_renew)
    mineru_auth.ensure()
    assert calls and calls[0].get("force") is True


# ══════════════════════════════════════════════════════════════════════
# Upload signature
# ══════════════════════════════════════════════════════════════════════


def test_upload_content_type_is_the_files_own_mime_type():
    """The presigned URL's signature covers this value exactly."""
    from pathlib import Path

    assert mineru._upload_content_type(Path("paper.pdf")) == "application/pdf"
    assert mineru._upload_content_type(Path("figure.png")) == "image/png"
    assert mineru._upload_content_type(Path("paper.html")) == "text/html"
    assert mineru._upload_content_type(Path("deck.pptx")).startswith(
        "application/vnd.openxmlformats-officedocument.presentationml"
    )
    # An unknown suffix has to send something rather than crash; whether OSS
    # accepts it is MinerU's call, and a 403 will say so.
    assert mineru._upload_content_type(Path("thing.unknownext")) == "application/octet-stream"
