"""Offline smoke tests: no network, no browser, no MinerU quota.

They cover the parts a client depends on as a contract — the tool surface, the
session-key rule and the memory document round-trip — so a refactor that breaks
any of them fails here instead of in someone's literature review.
"""

from __future__ import annotations

import json

import pytest

from academic_mcp import __version__
from academic_mcp.agent import memory as memory_mod
from academic_mcp.agent import tools as agent_tools
from academic_mcp.server import CONTRACT_VERSION, contract, health, mcp

EXPECTED_TOOLS = {
    "fetch_paper_text",
    "fetch_paper_pdf",
    "convert_document",
    "validate_doi",
    "get_paper",
    "health",
    "contract",
    "search_papers",
    "citation_chain",
    "memory",
    "telemetry",
}


async def test_tool_surface_is_complete():
    names = {tool.name for tool in await mcp.list_tools()}
    assert EXPECTED_TOOLS <= names, f"missing: {sorted(EXPECTED_TOOLS - names)}"


async def test_contract_reports_argument_names():
    payload = json.loads(await contract())
    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["version"] == __version__
    memory_spec = payload["tools"]["memory"]
    assert "op" in memory_spec["required"]
    assert {"session_id", "cwd", "data"} <= set(memory_spec["optional"])
    assert "doi" in payload["tools"]["validate_doi"]["required"]


async def test_health_hides_secrets():
    payload = json.loads(await health())
    config = payload["config"]
    assert config["data_dir"]
    # Only booleans and masked strings may cross the wire.
    assert isinstance(config["mineru_configured"], bool)
    assert all(not str(config.get(key, "")).startswith("sk-") for key in config)


@pytest.mark.parametrize(
    ("cwd", "expected"),
    [
        ("/mnt/c/Users/project/MoTe2", "--mnt-c-Users-project-MoTe2--"),
        ("/home/alice/", "--home-alice--"),
        ("", "default"),
    ],
)
def test_session_key_rule(cwd, expected):
    assert memory_mod.session_key_for_cwd(cwd) == expected


def test_memory_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(memory_mod, "TEXTS_DIR", tmp_path / "texts")
    session_id = memory_mod.session_key_for_cwd("/work/paper")

    mem = memory_mod.load_memory(session_id)
    memory_mod.cmd_update(mem, {
        "research_goal": "Fractional Chern insulators",
        "paper_notes": [{
            "paper_id": "10.1_x",
            "doi": "10.1/x",
            "title": "A paper",
            "first_author": "Doe J.",
            "year": 2026,
            "importance": "key result",
            "topics": ["moiré"],
        }],
        "key_finding": {"text": "the gap closes", "source_pids": ["10.1/x"]},
    })
    memory_mod.save_memory(session_id, mem)

    reloaded = memory_mod.load_memory(session_id)
    assert reloaded["research_goal"] == "Fractional Chern insulators"
    assert [p["paper_id"] for p in reloaded["paper_library"]] == ["10.1_x"]
    assert reloaded["topic_map"]["moiré"] == ["10.1_x"]
    assert (tmp_path / f"{session_id}.json").is_file()


def test_delete_cleans_every_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod, "DATA_DIR", tmp_path)
    mem = memory_mod.load_memory("default")
    memory_mod.cmd_update(mem, {
        "paper_notes": [{"paper_id": "10.1_x", "doi": "10.1/x", "title": "A", "topics": ["t"]}],
        # the finding carries the raw DOI spelling, the library the underscored one
        "key_finding": {"text": "f", "source_pids": ["10.1/x"]},
    })

    result = memory_mod.cmd_delete_paper(mem, "10.1_x")
    assert result["status"] == "ok"
    assert result["topic_refs_removed"] == 1
    assert result["finding_refs_removed"] == 1
    assert mem["paper_library"] == []
    assert mem["topic_map"] == {}
    assert mem["key_findings"][0]["source_pids"] == []


async def test_memory_tool_working_memory_uses_template(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod, "DATA_DIR", tmp_path)
    mem = memory_mod.load_memory("default")
    memory_mod.cmd_update(mem, {"research_goal": "goal text"})
    memory_mod.save_memory("default", mem)

    payload = await agent_tools.memory("working-memory", session_id="default")
    assert payload["ok"] is True
    assert "goal text" in payload["text"]


async def test_memory_tool_session_key_op():
    payload = await agent_tools.memory("session-key", cwd="/a/b")
    assert payload == {"ok": True, "cwd": "/a/b", "session_id": "--a-b--"}


async def test_memory_tool_rejects_unknown_op():
    payload = await agent_tools.memory("nope")
    assert payload["ok"] is False
    assert "unknown op" in payload["error"]


async def test_memory_tool_normalises_a_caller_supplied_session_id(tmp_path, monkeypatch):
    """A `session_id` is a path component, so it must go through the one guard.

    The regression: `_resolve_session` returned the caller's string verbatim, so
    `memory` and `telemetry` were the only two entry points that skipped
    `normalise_session_id`. A traversal id then resolved to a real file path
    (`DATA_DIR / "../../…" / f"{sid}.json"`) instead of falling back to
    `default`, which made every `*.json` the service can reach readable — and,
    through `save_memory`'s tmp + `.bak` + rename, writable.
    """
    monkeypatch.setattr(memory_mod, "DATA_DIR", tmp_path)

    payload = await agent_tools.memory("dump", session_id="../../../tmp/evil")
    assert payload["session_id"] == "default", payload

    # The write path is the dangerous one: prove nothing landed outside the root.
    written = await agent_tools.memory("goal", data={"goal": "x"}, session_id="../../escaped")
    assert written["session_id"] == "default"
    assert not (tmp_path.parent / "escaped.json").exists()

    # A real key still round-trips unchanged, CJK included.
    key = memory_mod.session_key_for_cwd("/mnt/c/Users/Sync/郑州大学/8.重点研发")
    ok = await agent_tools.memory("dump", session_id=key)
    assert ok["session_id"] == key


async def test_telemetry_reads_only_the_session_it_was_asked_for(tmp_path, monkeypatch):
    """Telemetry is per session, and the id is not a glob pattern.

    The regression: `cmd_telemetry` read the id off `memory["session_id"]`, a key
    the stored document never carries, so it silently fell back to `"*"` and the
    default read globbed every session's log. `is_session_key` allows `*` inside
    a key body, so the id also reached `Path.glob` as a pattern.
    """
    monkeypatch.setattr(memory_mod, "DATA_DIR", tmp_path)
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir(parents=True)
    (telemetry_dir / "--home-mine--.jsonl").write_text(
        json.dumps({"tool": "mine"}) + "\n", encoding="utf-8"
    )
    (telemetry_dir / "--home-theirs--.jsonl").write_text(
        json.dumps({"tool": "theirs"}) + "\n", encoding="utf-8"
    )

    payload = await agent_tools.telemetry(session_id="--home-mine--")
    tools = [e["tool"] for e in payload["result"]["entries"]]
    assert tools == ["mine"], payload["result"]

    # An accidentally wildcard-shaped key must stay a literal, not a pattern.
    star = await agent_tools.memory("telemetry", session_id="--a*b--")
    assert star["result"]["entries"] == []


async def test_memory_update_refusal_is_not_reported_as_a_saved_note(tmp_path, monkeypatch):
    """A refused write must fail loudly, because the caller has already paid.

    The regression: `cmd_update` reports "missing paper_id/doc_id" as a bare
    `{"error": …}`, and the `update` branch wrapped it in `{"ok": True,
    "result": …}` — the only op that did. A client checking `ok` (the natural
    reading, and what `edit`/`delete-paper`/`stores` teach it to do) then printed
    "notes auto-saved" over notes that were never written, and the document was
    not saved at all.
    """
    monkeypatch.setattr(memory_mod, "DATA_DIR", tmp_path)

    refused = await agent_tools.memory(
        "update",
        session_id="--home-alice--",
        data={"paper_notes": [{"title": "no identity at all"}]},
    )
    assert refused["ok"] is False, refused
    assert refused["error"]["error"] == "missing paper_id/doc_id"
    # Nothing was persisted, so the failure cannot resurface as a half-written doc.
    assert memory_mod.load_memory("--home-alice--")["paper_library"] == []

    # The success path is untouched.
    saved = await agent_tools.memory(
        "update",
        session_id="--home-alice--",
        data={"paper_notes": [{"paper_id": "10.1_x", "doi": "10.1/x", "title": "t"}]},
    )
    assert saved["ok"] is True, saved
    assert saved["result"]["status"] == "ok"
    assert len(memory_mod.load_memory("--home-alice--")["paper_library"]) == 1


async def test_citation_chain_accepts_session_id():
    """`session_id` must reach the tool that writes the DOI allowlist.
    It was accepted by `citation_chain()` but dropped by the registered wrapper,
    so every citation-chain result was filed under `default` and then refused by
    `validate_doi` for the session that discovered it.
    """
    tool = next(t for t in await mcp.list_tools() if t.name == "citation_chain")
    assert "session_id" in tool.input_schema["properties"]
    spec = json.loads(await contract())["tools"]["citation_chain"]
    assert "session_id" in spec["optional"]


async def test_citation_chain_forwards_session_id_to_snowball(monkeypatch):
    """`session_id` must reach `snowball()`, which owns the DOI cache write.

    The parameter was accepted by `citation_chain()` but dropped before the
    call, so every citation-chain hit was filed under `default.json` and
    `validate_doi` then refused it for the session that discovered it.
    """
    from academic_mcp.agent import snowball as snowball_mod

    seen = {}

    async def fake_snowball(seeds, direction, limit, proxy="", session_id=""):
        seen.update(seeds=seeds, direction=direction, limit=limit, session_id=session_id)
        return [], {"status": "ok", "count": 0}

    monkeypatch.setattr(snowball_mod, "snowball", fake_snowball)
    payload = await agent_tools.citation_chain(
        ["10.1/x"], "forward", 5, session_id="--home-alice--"
    )
    assert payload["ok"] is True
    assert seen["session_id"] == "--home-alice--"
    assert seen["seeds"] == ["10.1/x"]


# ── session id: one key, one file ────────────────────────────────────────────
#
# A session id is a filename in both `data/` and `data/search_cache/`. The
# regression these cover: `validate._session_id` allowed only `[A-Za-z0-9._-]`,
# so a CJK project key was dropped to "default" on the READ side while
# `save_search_cache` wrote it under its real name. `validate_doi` then answered
# "not searched in this session" for a DOI sitting in the session's own cache.


@pytest.mark.parametrize(
    "cwd",
    [
        "/mnt/c/Users/project/MoTe2",
        "/mnt/c/Users/Sync/郑州大学/8.重点研发",
        "/mnt/c/Users/Sync/课件",
        "/mnt/c/Users/Sync/blender files",
        "/home/alice",
    ],
)
def test_every_cwd_encodes_to_a_key_the_service_accepts(cwd):
    """The encoder and the guard must agree on every real project path.

    This is the invariant the CJK bug broke: `session_key_for_cwd` produced a
    key, and the DOI check refused to use it. Any future tightening of
    `is_session_key` that rejects a key the encoder can produce fails here.
    """
    key = memory_mod.session_key_for_cwd(cwd)
    assert memory_mod.is_session_key(key), f"{cwd!r} encodes to unusable {key!r}"
    assert memory_mod.normalise_session_id(key) == key


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", "default"),
        (None, "default"),
        ("default", "default"),
        ("--home-alice--", "--home-alice--"),
        # The case that broke: CJK must survive verbatim.
        ("--mnt-c-Users-Sync-郑州大学-8.重点研发--", "--mnt-c-Users-Sync-郑州大学-8.重点研发--"),
        # Path traversal must not survive in any spelling.
        ("../../etc/passwd", "default"),
        ("--../../etc/passwd--", "default"),
        ("--a/b--", "default"),
        ("--a\\b--", "default"),
        ("--a/../b--", "default"),
        ("--nul\x00--", "default"),
        ("--new\nline--", "default"),
        # Not a key at all.
        ("dsh-e2e-test-probe", "default"),
        ("--x--" + "y" * 300 + "--", "default"),
    ],
)
def test_session_id_guard_accepts_real_keys_and_refuses_paths(raw, expected):
    from academic_mcp.validate import _session_id

    assert _session_id(raw) == expected


async def test_search_cache_and_doi_check_use_the_same_file(tmp_path, monkeypatch):
    """The end-to-end invariant: what search writes, validate reads.

    Without this, the two halves can drift again — each side passing its own
    unit test while the pair disagrees.
    """
    from academic_mcp import validate as validate_mod
    from academic_mcp.agent import search as search_mod

    cwd = "/mnt/c/Users/Sync/郑州大学/8.重点研发"
    key = memory_mod.session_key_for_cwd(cwd)
    doi = "10.1103/q2mk-7b6s"

    # The session has a research goal, so the goal gate is not what is tested.
    # `validate` and `search` each hold their own view of the data directory
    # (a `settings.data_dir` property off one, a module constant off the other),
    # so both are redirected here; a test that moved only one of them would
    # stop exercising the very agreement it exists to check.
    (tmp_path / "search_cache").mkdir(exist_ok=True)
    # `search_cache_dir` is a property off `data_dir`, so redirecting that one
    # field moves the allowlist for `validate`; `CACHE_DIR` is search's own copy.
    monkeypatch.setattr(validate_mod.settings, "data_dir", tmp_path)
    monkeypatch.setattr(memory_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(search_mod, "CACHE_DIR", str(tmp_path / "search_cache"))
    mem = memory_mod.load_memory(key)
    memory_mod.cmd_update(mem, {"research_goal": "moiré flat bands"})
    memory_mod.save_memory(key, mem)

    search_mod.save_search_cache(
        [{"doi": doi, "paper_id": doi.replace("/", "_")}], "some query", key
    )
    assert (tmp_path / f"{key}.json").is_file()

    assert validate_mod.check(doi, key) == doi


# ── get_paper: one call, no hand-rolling ─────────────────────────────────────
#
# The tool this replaces a workaround for: over one project's recent sessions
# there were 145 hand-rolled HTTP fetches (curl/urllib against Crossref, Europe
# PMC and publisher pages) against 65 calls to the tools that exist to do this.
# The reason was that a single known paper took three round trips in three
# different shapes, so agents skipped them. These tests pin the ONE call.


async def test_get_paper_requires_an_identifier():
    """Either a DOI or a title identifies the paper; neither is not a call."""
    from academic_mcp.agent import tools as t

    payload = await t.get_paper("")
    assert payload["ok"] is False
    assert "DOI" in payload["error"] and "title" in payload["error"]


async def test_get_paper_returns_cached_text_without_downloading(tmp_path, monkeypatch):
    """The read path must serve a paper that is already local.

    This is the case a caller hits on every re-read, and it exercises the whole
    contract without a network: the metadata comes from `lookup_doi`, which is
    stubbed, and the markdown is a real file on disk.
    """
    from academic_mcp.agent import memory as mem
    from academic_mcp.agent import search as search_mod
    from academic_mcp.agent import tools as t

    key = "10.1234_abc"
    monkeypatch.setattr(mem, "TEXTS_DIR", tmp_path)
    (tmp_path / f"{key}.md").write_text("# Title\n\nbody text\n", encoding="utf-8")

    async def fake_lookup(doi):
        return {
            "doi": "10.1234/abc", "paper_id": key, "title": "A Title",
            "authors": [{"name": "Doe J."}], "first_author": "Doe J.",
            "year": 2026, "venue": "Nature", "volume": "1", "pages": "1-2",
        }

    monkeypatch.setattr(search_mod, "lookup_doi", fake_lookup)

    payload = await t.get_paper("10.1234/abc", session_id="--home-alice--")
    assert payload["ok"] is True
    assert payload["cached"] is True
    assert payload["md_path"] == str(tmp_path / f"{key}.md")
    assert payload["title"] == "A Title"
    # The full text must NOT be inlined: the caller's pruner would shred it.
    assert "body text" not in json.dumps(payload)


async def test_get_paper_meta_mode_never_touches_the_disk(tmp_path, monkeypatch):
    from academic_mcp.agent import memory as mem
    from academic_mcp.agent import search as search_mod
    from academic_mcp.agent import tools as t

    monkeypatch.setattr(mem, "TEXTS_DIR", tmp_path)

    async def fake_lookup(doi):
        return {"doi": doi, "paper_id": "10.1_x", "title": "T", "year": 2026}

    monkeypatch.setattr(search_mod, "lookup_doi", fake_lookup)
    payload = await t.get_paper("10.1/x", session_id="--home-alice--", mode="meta")
    assert payload["ok"] is True
    assert payload["mode"] == "meta"
    assert "md_path" not in payload
    assert not list(tmp_path.iterdir())


async def test_get_paper_meta_mode_reports_an_unknown_doi():
    from academic_mcp.agent import search as search_mod
    from academic_mcp.agent import tools as t

    async def fake_lookup(doi):
        return None

    original = search_mod.lookup_doi
    search_mod.lookup_doi = fake_lookup
    try:
        payload = await t.get_paper("10.9999/nope", session_id="--home-alice--", mode="meta")
    finally:
        search_mod.lookup_doi = original
    assert payload["ok"] is False
    assert "no metadata" in payload["error"]
    assert "hint" in payload


# ── Scopus cooldown ─────────────────────────────────────────────────────────
#
# api.elsevier.com is unreachable from this host. Before the cooldown every
# search paid the connect timeout before falling back to OpenAlex (measured:
# 45 s per search with a 10 s connect timeout, 19 s with the Scopus-specific
# 4 s one, versus 1.5 s once Scopus is skipped). The results were never wrong —
# only slow — which is why this went unnoticed for as long as it did.


def test_a_transport_failure_opens_the_scopus_cooldown():
    import httpx

    from academic_mcp.agent import search as s

    s._scopus_down_until = 0.0
    assert s._scopus_cooling() is False
    s._scopus_transport_failed(httpx.ConnectTimeout("timed out"))
    assert s._scopus_cooling() is True, "a connect timeout must open the cooldown"
    s._scopus_down_until = 0.0


def test_an_http_answer_does_not_open_the_scopus_cooldown():
    """A 401/403 means Scopus ANSWERED. Cooling down on it would hide a real
    configuration problem (a key that lacks COMPLETE-view permission) behind a
    five-minute silence, and would penalise an engine that is working."""
    import httpx

    from academic_mcp.agent import search as s

    s._scopus_down_until = 0.0
    request = httpx.Request("GET", "https://api.elsevier.com/x")
    response = httpx.Response(403, request=request)
    s._scopus_transport_failed(httpx.HTTPStatusError("403", request=request, response=response))
    assert s._scopus_cooling() is False
    s._scopus_down_until = 0.0


async def test_search_scopus_skips_the_network_while_cooling(monkeypatch):
    """While cooling, Scopus must return without opening a client at all —
    that is the whole point, so assert on the client, not on the result."""
    from academic_mcp.agent import search as s

    opened = []

    class ExplodingClient:
        def __init__(self, *a, **kw):
            opened.append(kw)
            raise AssertionError("Scopus must not open a client while cooling")

    monkeypatch.setattr(s, "ELSEVIER_API_KEY", "test-key")
    monkeypatch.setattr(s.httpx, "AsyncClient", ExplodingClient)
    s._scopus_down_until = s.time.monotonic() + 60
    try:
        assert await s.search_scopus("anything") == []
        assert await s.search_scopus_by_doi("10.1/x") is None
        assert opened == []
    finally:
        s._scopus_down_until = 0.0


def test_the_scopus_probe_uses_a_shorter_connect_budget():
    """The probe's connect timeout is the cost of a down engine on every search,
    so it is deliberately tighter than the shared default. Pinned here because
    raising it back to the default silently restores the slowness."""
    from academic_mcp.agent import search as s

    scopus = s._scopus_client_kwargs()["timeout"]
    shared = s._client_kwargs()["timeout"]
    assert scopus.connect == s._SCOPUS_CONNECT_TIMEOUT
    assert scopus.connect < shared.connect, "the probe must be cheaper than the default"
    assert scopus.read == shared.read, "a slow-but-alive Scopus still gets the full read budget"


async def test_get_paper_authorises_the_doi_on_the_cached_path_too(tmp_path, monkeypatch):
    """A resolved DOI must be importable afterwards, whichever path served it.

    The cache write used to sit only on the freshly-downloaded branch, so
    `get_paper` on a paper already on disk returned its path and recorded
    nothing — and the follow-up `import_papers`, which authorises through that
    same cache, refused the DOI as "never searched in this session". One
    workflow, two halves, disagreeing about whether the paper existed.
    """
    from academic_mcp import validate as validate_mod
    from academic_mcp.agent import memory as mem
    from academic_mcp.agent import search as search_mod
    from academic_mcp.agent import tools as t

    key = "10.1234_abc"
    monkeypatch.setattr(mem, "TEXTS_DIR", tmp_path)
    monkeypatch.setattr(mem, "DATA_DIR", tmp_path)
    monkeypatch.setattr(search_mod, "CACHE_DIR", str(tmp_path / "search_cache"))
    monkeypatch.setattr(validate_mod.settings, "data_dir", tmp_path)
    (tmp_path / "search_cache").mkdir(exist_ok=True)
    (tmp_path / f"{key}.md").write_text("# T\n\nbody\n", encoding="utf-8")

    async def fake_lookup(doi):
        return {"doi": "10.1234/abc", "paper_id": key, "title": "T",
                "authors": [{"name": "Doe J."}], "first_author": "Doe J.", "year": 2026}

    monkeypatch.setattr(search_mod, "lookup_doi", fake_lookup)

    sid = "--home-alice--"
    # The goal gate is satisfied, so a refusal can only come from the allowlist.
    goal = mem.load_memory(sid)
    mem.cmd_update(goal, {"research_goal": "something"})
    mem.save_memory(sid, goal)

    payload = await t.get_paper("10.1234/abc", session_id=sid)
    assert payload["cached"] is True
    # The DOI is now authorised: validate agrees, so import_papers would not
    # turn it away.
    assert validate_mod.check("10.1234/abc", sid) == "10.1234/abc"


# ── title/author resolution ──────────────────────────────────────────────────
#
# A user names a paper the way a human would — "the Xu 2021 Continuous Mott
# transition paper" — not by DOI. The resolver turns that into either one
# confident DOI or a shortlist. These pin the part that matters most: it must
# never manufacture a DOI, because every later step treats a DOI as ground truth.


def _paper(doi, title, author="Xu Y.", year=2021, venue="Nature"):
    return {
        "paper_id": doi.replace("/", "_"), "doi": doi, "title": title,
        "authors": [{"name": author}], "first_author": author,
        "year": year, "venue": venue,
    }


async def test_resolve_work_accepts_a_confident_title(monkeypatch):
    from academic_mcp.agent import search as s

    async def fake_search(title, author="", journal="", limit=5):
        return [
            _paper("10.1038/s41586-021-03835-2", "Continuous Mott transition in semiconductor moiré superlattices"),
            _paper("10.1038/other", "Unrelated work on superconductivity"),
        ]

    monkeypatch.setattr(s, "search_by_title", fake_search)
    out = await s.resolve_work(title="Continuous Mott transition in semiconductor moiré superlattices")
    assert out["ok"] is True and out["resolved"] is True
    assert out["paper"]["doi"] == "10.1038/s41586-021-03835-2"


async def test_resolve_work_returns_candidates_instead_of_guessing(monkeypatch):
    """Two plausible titles, neither confident: the answer is a list."""
    from academic_mcp.agent import search as s

    async def fake_search(title, author="", journal="", limit=5):
        return [
            _paper("10.1/a", "Continuous Mott transition in semiconductor moiré superlattices", year=2021),
            _paper("10.1/b", "Continuous Mott transition in semiconductor heterostructures", year=2019),
        ]

    monkeypatch.setattr(s, "search_by_title", fake_search)
    out = await s.resolve_work(title="Continuous Mott transition")
    assert out["ok"] is True
    assert out["resolved"] is False
    assert "paper" not in out, "an ambiguous title must not produce a single paper"
    assert 2 <= len(out["candidates"]) <= 5
    assert {c["doi"] for c in out["candidates"]} == {"10.1/a", "10.1/b"}


async def test_resolve_work_reports_nothing_found(monkeypatch):
    from academic_mcp.agent import search as s

    async def fake_search(title, author="", journal="", limit=5):
        return []

    monkeypatch.setattr(s, "search_by_title", fake_search)
    out = await s.resolve_work(title="a paper that does not exist anywhere")
    assert out["ok"] is False
    assert "error" in out and "hint" in out


async def test_resolve_work_treats_a_doi_as_authoritative(monkeypatch):
    """A DOI is an identifier, not a search: it must not be second-guessed.

    Rejecting a DOI because its published title disagrees with the caller's
    spelling would refuse the one input that cannot be wrong.
    """
    from academic_mcp.agent import search as s

    async def fake_lookup(doi):
        return _paper("10.1038/real", "The actual published title")

    async def exploding_search(*a, **kw):
        raise AssertionError("a DOI must not go through title search")

    monkeypatch.setattr(s, "lookup_doi", fake_lookup)
    monkeypatch.setattr(s, "search_by_title", exploding_search)
    out = await s.resolve_work(doi="10.1038/real", title="a completely different title")
    assert out["ok"] is True and out["resolved"] is True
    assert out["matched_by"] == "doi"


async def test_resolve_work_requires_something_to_look_up():
    from academic_mcp.agent import search as s

    out = await s.resolve_work()
    assert out["ok"] is False
    assert "title" in out["error"]


async def test_get_paper_reports_needs_choice_for_an_ambiguous_title(monkeypatch):
    from academic_mcp.agent import search as s
    from academic_mcp.agent import tools as t

    async def fake_resolve(doi="", title="", author="", journal=""):
        return {
            "ok": True, "resolved": False,
            "candidates": [{"doi": "10.1/a", "title": "A"}, {"doi": "10.1/b", "title": "B"}],
        }

    monkeypatch.setattr(s, "resolve_work", fake_resolve)
    monkeypatch.setattr(t, "resolve_work", fake_resolve)
    payload = await t.get_paper(title="an ambiguous title", session_id="--home-alice--")
    assert payload["ok"] is False
    assert payload["needs_choice"] is True
    assert len(payload["candidates"]) == 2
    assert "hint" in payload


async def test_a_variant_title_returns_candidates_not_the_variant(monkeypatch):
    """A reply/corrigendum is a DIFFERENT work, so it is not the answer.

    Its title contains the caller's title verbatim, which is why a containment
    rule called it a match. Resolving the base title to its rebuttal is the same
    over-confidence as guessing, one spelling further on — so the resolver hands
    the choice back instead.
    """
    from academic_mcp.agent import search as s

    base = "Continuous Mott transition in semiconductor moiré superlattices"

    async def fake_search(title, author="", journal="", limit=5):
        return [_paper("10.1/reply", base + ": a reply to Xu et al")]

    monkeypatch.setattr(s, "search_by_title", fake_search)
    out = await s.resolve_work(title=base)
    assert out["ok"] is True
    assert out["resolved"] is False, "a reply must not be returned as the paper"
    assert out["candidates"][0]["doi"] == "10.1/reply"


async def test_an_exact_title_still_resolves_when_a_superset_also_matches(monkeypatch):
    """The counterpart: a real exact hit wins over a longer variant of it."""
    from academic_mcp.agent import search as s

    base = "Continuous Mott transition in semiconductor moiré superlattices"

    async def fake_search(title, author="", journal="", limit=5):
        return [
            _paper("10.1/reply", base + ": a reply to Xu et al"),
            _paper("10.1/base", base),
        ]

    monkeypatch.setattr(s, "search_by_title", fake_search)
    out = await s.resolve_work(title=base)
    assert out["ok"] is True and out["resolved"] is True
    assert out["paper"]["doi"] == "10.1/base"


async def test_a_title_fragment_is_not_a_confident_match(monkeypatch):
    """The fragment case, at the resolver level.

    "Continuous Mott transition" is a substring of the published title, so a
    containment test called it a match and the resolver returned whichever
    candidate ranked first — a guess. What disqualifies it is the uncovered tail.
    """
    from academic_mcp.agent import search as s

    async def fake_search(title, author="", journal="", limit=5):
        return [_paper("10.1/full", "Continuous Mott transition in semiconductor moiré superlattices")]

    monkeypatch.setattr(s, "search_by_title", fake_search)
    out = await s.resolve_work(title="Continuous Mott transition")
    assert out["ok"] is True
    assert out["resolved"] is False, "a fragment must not be resolved to a single paper"
    assert out["candidates"][0]["doi"] == "10.1/full"


async def test_an_author_ruling_out_every_candidate_yields_candidates(monkeypatch):
    """A named author who is on none of the hits is evidence AGAINST them.

    The title matches, so a title-only rule would resolve it; the author the
    caller supplied says otherwise, and guessing here is how a bibliography ends
    up citing the wrong paper.
    """
    from academic_mcp.agent import search as s

    async def fake_search(title, author="", journal="", limit=5):
        return [_paper("10.1/x", "Continuous Mott transition in moiré superlattices",
                       author="Someone Else")]

    monkeypatch.setattr(s, "search_by_title", fake_search)
    out = await s.resolve_work(title="Continuous Mott transition in moiré superlattices",
                               author="Xu")
    assert out["ok"] is True
    assert out["resolved"] is False
    assert out["candidates"], "the caller still needs to see what was found"


async def test_an_author_on_the_paper_still_resolves(monkeypatch):
    from academic_mcp.agent import search as s

    async def fake_search(title, author="", journal="", limit=5):
        return [_paper("10.1/x", "Continuous Mott transition in semiconductor moiré superlattices",
                       author="Xu Y.")]

    monkeypatch.setattr(s, "search_by_title", fake_search)
    out = await s.resolve_work(
        title="Continuous Mott transition in semiconductor moiré superlattices", author="Xu")
    assert out["resolved"] is True
    assert out["paper"]["doi"] == "10.1/x"


async def test_get_paper_reports_whether_notes_already_exist(tmp_path, monkeypatch):
    """The caller needs to know a paper was already read, to not read it again.

    A note update MERGES, so a second reading pass appends a near-duplicate at
    the cost of a whole model call — the field this test pins is what makes the
    caller able to skip it.
    """
    from academic_mcp.agent import memory as mem
    from academic_mcp.agent import search as search_mod
    from academic_mcp.agent import tools as t

    key = "10.1234_abc"
    sid = "--home-alice--"
    monkeypatch.setattr(mem, "TEXTS_DIR", tmp_path)
    monkeypatch.setattr(mem, "DATA_DIR", tmp_path)
    (tmp_path / f"{key}.md").write_text("# T\n\nbody\n", encoding="utf-8")

    async def fake_lookup(doi):
        return {"doi": "10.1234/abc", "paper_id": key, "title": "T",
                "authors": [{"name": "Doe J."}], "first_author": "Doe J.", "year": 2026}

    monkeypatch.setattr(search_mod, "lookup_doi", fake_lookup)

    # Not read yet.
    payload = await t.get_paper("10.1234/abc", session_id=sid)
    assert payload["_hadNotes"] is False

    # Now with notes on the record.
    record = mem.load_memory(sid)
    mem.cmd_update(record, {"paper_notes": [{"paper_id": key, "doi": "10.1234/abc",
                                             "key_notes": "a reading"}]})
    mem.save_memory(sid, record)
    payload = await t.get_paper("10.1234/abc", session_id=sid)
    assert payload["_hadNotes"] is True
    assert payload["_noteCount"] == 1


async def test_a_distill_false_placeholder_does_not_count_as_notes(tmp_path, monkeypatch):
    """`distill: false` files a record with a placeholder, not a reading.

    Counting that as "already read" would make the follow-up distillation pass
    skip the paper forever — the notes would never be written by anyone.
    """
    from academic_mcp.agent import memory as mem
    from academic_mcp.agent import search as search_mod
    from academic_mcp.agent import tools as t

    key = "10.1234_abc"
    sid = "--home-alice--"
    monkeypatch.setattr(mem, "TEXTS_DIR", tmp_path)
    monkeypatch.setattr(mem, "DATA_DIR", tmp_path)
    (tmp_path / f"{key}.md").write_text("# T\n\nbody\n", encoding="utf-8")

    async def fake_lookup(doi):
        return {"doi": "10.1234/abc", "paper_id": key, "title": "T", "year": 2026}

    monkeypatch.setattr(search_mod, "lookup_doi", fake_lookup)

    record = mem.load_memory(sid)
    mem.cmd_update(record, {"paper_notes": [{
        "paper_id": key, "doi": "10.1234/abc", "status": "read",
        "detailed_notes": "[未精读] 论文已下载并入库，尚未生成笔记（distill: false）。",
    }]})
    mem.save_memory(sid, record)

    payload = await t.get_paper("10.1234/abc", session_id=sid)
    assert payload["_hadNotes"] is False, "a placeholder must not pass as notes"


# ── proxy failover ───────────────────────────────────────────────────────────
#
# api.elsevier.com is intermittently blackholed from this host, so "the proxy"
# cannot be a single route: the download path (httpclient.fetch) and the search
# engines (FailoverClient) both walk a chain — primary first, DOWNLOAD_PROXY as
# the rescue hop last — and only a transport-level failure moves to the next
# route. These pin the chain composition, the failover itself, and the two rules
# that keep it honest: an HTTP answer never fails over, and a route that just
# failed is tried last for a while instead of costing every request its connect
# timeout.
#
# The rescue hop used to be a separate DOWNLOAD_PROXY_FALLBACK setting, so these
# all had to patch it. It is DOWNLOAD_PROXY now, which is why the routes below
# are DERIVED from the configured proxy rather than written as literals: that is
# the property worth pinning — whatever DOWNLOAD_PROXY is, it is the hop.

_FAILOVER_HOP = "http://rescue.example:8119"


def test_download_routes_follow_the_mode(monkeypatch):
    from academic_mcp import httpclient as hc

    monkeypatch.setattr(hc.settings, "download_proxy", _FAILOVER_HOP)

    # `failover` and `fallback` now describe the SAME two routes: the rescue hop
    # is DOWNLOAD_PROXY itself, not a separate DOWNLOAD_PROXY_FALLBACK setting.
    monkeypatch.setattr(hc.settings, "download_proxy_mode", "failover")
    assert hc._routes() == [None, _FAILOVER_HOP], "direct first, then the configured proxy"

    monkeypatch.setattr(hc.settings, "download_proxy_mode", "fallback")
    assert hc._routes() == [None, _FAILOVER_HOP]

    monkeypatch.setattr(hc.settings, "download_proxy_mode", "primary")
    assert hc._routes() == [_FAILOVER_HOP], "primary only, no hop"

    monkeypatch.setattr(hc.settings, "download_proxy_mode", "none")
    assert hc._routes() == [None], "none ignores the proxy entirely"

    # A proxy that IS the primary route must not also be appended as its own
    # rescue hop: the same address twice retries a dead exit instead of moving on.
    monkeypatch.setattr(hc.settings, "download_proxy_mode", "failover")
    monkeypatch.setattr(hc, "_primary_proxy", lambda: _FAILOVER_HOP)
    assert hc._routes() == [_FAILOVER_HOP], "no duplicate route"


# ── the browser's exit is a per-request decision, not a deployment setting ──
#
# `_browser_config` used to read the proxy off settings at session start, so
# EVERY browser request left through DOWNLOAD_PROXY whenever it was configured.
# Measured 2026-10-04 on one and the same PDF:
#
#     direct (Shanghai, CERNET)       -> 200, 808853 bytes, a real PDF
#     DOWNLOAD_PROXY (Hong Kong, HKU) -> 401, "Authorization Required"
#
# APS entitlements are per-IP, so a fixed exit turns a working download into an
# authorization wall. The browser now walks the same routes as `httpclient`,
# which is what lets the second one rescue the first.


def test_browser_routes_mirror_the_download_routes(monkeypatch):
    """Direct first, the proxy as the rescue hop — the httpclient order."""
    from academic_mcp.resolvers import camoufox as cf

    monkeypatch.setattr(cf.settings, "download_proxy", _FAILOVER_HOP)
    monkeypatch.setattr(cf.settings, "download_proxy_mode", "failover")
    assert cf._browser_routes() == [None, _FAILOVER_HOP]
    monkeypatch.setattr(cf.settings, "download_proxy_mode", "fallback")
    assert cf._browser_routes() == [None, _FAILOVER_HOP]

    # `primary` keeps the old behaviour available for a publisher that is only
    # reachable from the proxy.
    monkeypatch.setattr(cf.settings, "download_proxy_mode", "primary")
    assert cf._browser_routes() == [_FAILOVER_HOP]

    monkeypatch.setattr(cf.settings, "download_proxy_mode", "none")
    assert cf._browser_routes() == [None], "none ignores the configured proxy"

    # No proxy configured at all: one route, and it is direct.
    monkeypatch.setattr(cf.settings, "download_proxy", None)
    monkeypatch.setattr(cf.settings, "download_proxy_mode", "failover")
    assert cf._browser_routes() == [None]


def test_the_browser_config_takes_its_exit_from_the_caller(monkeypatch):
    """A session's exit must be settable per session, or the retry is impossible.

    The regression this pins: `_browser_config(headless)` read
    `settings.download_proxy` itself, so a second session could not be sent out
    of a different exit — the authorization wall was terminal.
    """
    from academic_mcp.resolvers import camoufox as cf

    monkeypatch.setattr(cf.settings, "download_proxy", _FAILOVER_HOP)

    direct = cf._browser_config(True, None)
    assert direct["proxy"] is None, "route None must mean direct, not 'read settings'"

    proxied = cf._browser_config(True, _FAILOVER_HOP)
    assert proxied["proxy"] == {"server": _FAILOVER_HOP}


# ── the arXiv fallback must not depend on a search engine ────────────────────
#
# The resolver whose whole job is to rescue a paper the publisher would not give
# us was asking Google, Brave and DuckDuckGo to find the preprint. Measured
# 2026-10-04 all three were refusing us (Brave 429, Google 403, DuckDuckGo an
# empty 202), so 23 of 27 title searches failed and paywalled APS papers went
# undownloaded *while their preprints sat on arXiv*. The API needed no search
# engine and answered both papers that had just failed, in under a second.

_ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>arXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2311.04560v2</id>
    <title>Fast Generation of GHZ-like States Using Collective-Spin XYZ Model</title>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2311.04560v1</id>
    <title>the same paper, first version</title>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2409.08524v1</id>
    <title>A different paper</title>
  </entry>
</feed>
"""


def _arxiv_api(monkeypatch, body=_ATOM_SAMPLE, status=200, recorder=None):
    """Stub `httpx.get`, which is how `_api_ids` reaches the network."""
    import httpx

    class FakeResponse:
        status_code = status
        text = body

    def fake_get(url, **kwargs):
        if recorder is not None:
            recorder.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(httpx, "get", fake_get)


def test_arxiv_api_search_returns_ids_and_drops_versions(monkeypatch):
    from academic_mcp.resolvers.arxiv import ArxivTitleResolver

    _arxiv_api(monkeypatch)
    ids = ArxivTitleResolver()._api_ids("Fast Generation of GHZ-like States")
    # v2 and v1 are the same preprint: one candidate, and the version suffix is
    # stripped because `arxiv.org/pdf/<id>` is asked for bare.
    assert ids == ["2311.04560", "2409.08524"], ids


def test_arxiv_api_search_asks_for_the_title_field(monkeypatch):
    """A quoted `ti:` phrase, not `all:`.

    An unquoted query is an OR over every word and answers with a page of
    unrelated papers; a quoted `all:` over a full title usually answers with
    nothing at all. The verification step downstream decides which candidate is
    right, so this only has to be a good query — but it has to be a good one.
    """
    from academic_mcp.resolvers.arxiv import ArxivTitleResolver

    seen: list[tuple] = []
    _arxiv_api(monkeypatch, recorder=seen)
    ArxivTitleResolver()._api_ids('A Title: with "quotes" & symbols')

    url, kwargs = seen[0]
    assert "search_query=ti:" in url, url
    assert "max_results=" in url
    # The title's own quotes and backslashes must not break the phrase we build.
    assert url.count("%22") == 2, url
    assert '"' not in url.split("search_query=")[1].split("&")[0]
    assert kwargs.get("follow_redirects") is True


def test_arxiv_api_failure_is_not_an_exception(monkeypatch):
    """Discovery must never raise: it is the last resolver in the chain."""
    import httpx

    from academic_mcp.resolvers.arxiv import ArxivTitleResolver

    def boom(url, **kwargs):
        raise httpx.ConnectTimeout("blackholed")

    monkeypatch.setattr(httpx, "get", boom)
    assert ArxivTitleResolver()._api_ids("Anything") == []

    _arxiv_api(monkeypatch, body="<html>not xml", status=200)
    assert ArxivTitleResolver()._api_ids("Anything") == []

    _arxiv_api(monkeypatch, status=503)
    assert ArxivTitleResolver()._api_ids("Anything") == []


def test_arxiv_search_prefers_the_api_and_keeps_ddgs_as_the_rescue(monkeypatch):
    """The API answers first; DDGS is only reached when it has nothing.

    Pinned as an ORDER because the reverse silently restores the failure: a
    dead search engine would then be consulted before the source that works.
    """
    from academic_mcp.resolvers.arxiv import ArxivTitleResolver
    from academic_mcp.resolvers.base import Paper

    resolver = ArxivTitleResolver()
    calls: list[str] = []

    monkeypatch.setattr(
        resolver, "_api_ids",
        lambda title: (calls.append("api"), ["2311.04560"])[1],
    )
    monkeypatch.setattr(
        resolver, "_ddgs_ids",
        lambda title: (calls.append("ddgs"), ["9999.99999"])[1],
    )

    paper = Paper(doi="10.1103/x", title="Fast Generation of GHZ-like States")
    assert resolver._search_sync(paper) == ["2311.04560"]
    assert calls == ["api"], "a hit from the API must not also pay for DDGS"

    # API empty -> the web fallback is still tried.
    calls.clear()
    monkeypatch.setattr(resolver, "_api_ids", lambda title: (calls.append("api"), [])[1])
    assert resolver._search_sync(paper) == ["9999.99999"]
    assert calls == ["api", "ddgs"]


async def test_arxiv_api_hit_is_verified_before_it_is_downloaded(monkeypatch):
    """A candidate from the API is still only a candidate.

    The API's phrase match can land on a different paper sharing a prefix, and
    fetching the wrong paper is worse than failing: the reading pass writes
    confident notes about it. `_find` must route the ids through `_verify`.
    """
    from academic_mcp.resolvers.arxiv import ArxivTitleResolver
    from academic_mcp.resolvers.base import Paper

    resolver = ArxivTitleResolver()
    monkeypatch.setattr(resolver, "_search_sync", lambda paper: ["2311.04560"])
    verified: list[list[str]] = []

    async def fake_verify(ids, paper, client):
        verified.append(ids)
        return None

    async def fake_client(**overrides):
        return object()

    monkeypatch.setattr(resolver, "_verify", fake_verify)
    monkeypatch.setattr(
        "academic_mcp.resolvers.arxiv.httpclient.get_client", fake_client
    )

    paper = Paper(doi="10.1103/x", title="Some Title")
    assert await resolver._find(paper) is None
    assert verified == [["2311.04560"]], "the API's candidates must be verified"


def test_search_route_chain_gains_the_failover_hop(monkeypatch):
    from academic_mcp.agent import search as s

    monkeypatch.setattr(s._settings, "download_proxy", _FAILOVER_HOP)
    monkeypatch.setattr(s._settings, "download_proxy_mode", "failover")
    assert s._route_chain(None) == [None, _FAILOVER_HOP]
    assert s._route_chain(_FAILOVER_HOP) == [_FAILOVER_HOP], "no duplicate hop"

    monkeypatch.setattr(s._settings, "download_proxy_mode", "fallback")
    assert s._route_chain(None) == [None], "only failover mode adds the hop"


def test_route_health_memory_reorders_and_clears(monkeypatch):
    from academic_mcp import httpclient as hc

    monkeypatch.setattr(hc, "_route_dead_until", {})
    routes = [None, _FAILOVER_HOP]
    assert hc.order_routes(routes, "ns-test") == routes

    hc.mark_route_failed("ns-test", None)
    assert hc.order_routes(routes, "ns-test") == [_FAILOVER_HOP, None]
    assert hc.order_routes(routes, "other-ns") == routes, "namespaces must not leak"

    hc.mark_route_ok("ns-test", None)
    assert hc.order_routes(routes, "ns-test") == routes


async def test_fetch_fails_over_on_a_transport_failure(monkeypatch):
    from academic_mcp import httpclient as hc

    # The proxy IS the rescue hop, so a failover test configures it: a direct
    # primary then DOWNLOAD_PROXY. (Previously this patched the removed
    # DOWNLOAD_PROXY_FALLBACK field and left download_proxy unset.)
    monkeypatch.setattr(hc.settings, "download_proxy", _FAILOVER_HOP)
    monkeypatch.setattr(hc.settings, "download_proxy_mode", "failover")
    monkeypatch.setattr(hc, "_route_dead_until", {})

    attempted: list[str | None] = []

    async def fake_fetch_route(url, *, headers=None, want_pdf=False, timeout=None,
                               attempts=2, client=None, proxy=None):
        attempted.append(proxy)
        if proxy is None:
            return hc.FetchResult(ok=False, error="ConnectTimeout: blackholed"), True
        return hc.FetchResult(ok=True, content=b"%PDF-1.4 filler", status=200), False

    monkeypatch.setattr(hc, "_fetch_route", fake_fetch_route)
    result = await hc.fetch("https://api.elsevier.com/content/article", attempts=1)

    assert result.ok is True
    assert attempted == [None, _FAILOVER_HOP], "the rescue hop must be tried second"
    # The blackholed route is remembered: the next request tries the hop first.
    assert hc.order_routes([None, _FAILOVER_HOP], "download") == [_FAILOVER_HOP, None]


async def test_fetch_does_not_fail_over_on_an_http_answer(monkeypatch):
    """Another exit cannot change an answer the far end already gave."""
    from academic_mcp import httpclient as hc

    # The proxy IS the rescue hop, so a failover test configures it: a direct
    # primary then DOWNLOAD_PROXY. (Previously this patched the removed
    # DOWNLOAD_PROXY_FALLBACK field and left download_proxy unset.)
    monkeypatch.setattr(hc.settings, "download_proxy", _FAILOVER_HOP)
    monkeypatch.setattr(hc.settings, "download_proxy_mode", "failover")
    monkeypatch.setattr(hc, "_route_dead_until", {})

    attempted: list[str | None] = []

    async def fake_fetch_route(url, *, headers=None, want_pdf=False, timeout=None,
                               attempts=2, client=None, proxy=None):
        attempted.append(proxy)
        return hc.FetchResult(ok=False, status=403, error="access denied (paywalled)"), False

    monkeypatch.setattr(hc, "_fetch_route", fake_fetch_route)
    result = await hc.fetch("https://api.elsevier.com/content/article", attempts=1)

    assert result.status == 403
    assert attempted == [None], "an HTTP answer must not trigger the hop"


async def test_search_failover_client_rescues_a_blackholed_route(monkeypatch):
    import httpx

    from academic_mcp import httpclient as hc
    from academic_mcp.agent import search as s

    monkeypatch.setattr(hc, "_route_dead_until", {})
    # The real retry policy would sleep seconds between attempts; the route
    # walk is what this test pins.
    monkeypatch.setattr(s, "_with_retry", lambda fn: fn)

    request = httpx.Request("GET", "https://api.elsevier.com/content/search/scopus")
    seen: list[str] = []

    class FakeClient:
        def __init__(self, alive: bool):
            self._alive = alive

        async def get(self, url, **kwargs):
            seen.append("alive" if self._alive else "dead")
            if not self._alive:
                raise httpx.ConnectTimeout("connect timed out", request=request)
            return httpx.Response(200, request=request)

        async def aclose(self):
            pass

    clients = {None: FakeClient(False), _FAILOVER_HOP: FakeClient(True)}
    monkeypatch.setattr(s.FailoverClient, "_client_for", lambda self, proxy: clients[proxy])

    async with s.FailoverClient([None, _FAILOVER_HOP], ns="test-failover") as client:
        resp = await client.get("", params={"query": "x"})

    assert resp.status_code == 200
    assert seen == ["dead", "alive"], "a transport failure must move to the rescue route"
    assert hc.order_routes([None, _FAILOVER_HOP], "test-failover") == [_FAILOVER_HOP, None]


# ── title resolution: the candidate list is a QUESTION, not a ranking ───────
#
# `search_by_title` first sent the title through the keyword search, which
# searches title AND abstract. Measured on this corpus, "Continuous Mott
# transition" then returned five papers about photocatalysts and memristors
# (similarity 0.13-0.20) and the paper actually being sought was absent
# entirely — the engine's top-N is topical relevance, not "which paper IS this".
# Two things have to hold: the engines are asked for the TITLE field, and hits
# too dissimilar to be the work are not offered as choices.


async def test_search_by_title_queries_the_title_field(monkeypatch):
    """Both engines must be asked for a title match, not a topic match."""
    from academic_mcp.agent import search as s

    seen = {}

    async def fake_scopus(title, limit=10, author=None, journal=None):
        seen["scopus"] = title
        return [{"doi": "10.1/a", "title": "A Title"}]

    async def fake_openalex_title(title, limit=10):
        seen["openalex_title"] = title
        return [{"doi": "10.1/b", "title": "A Title"}]

    async def fake_openalex(query, limit=20, year=None, author=None, journal=None):
        seen["keyword"] = query
        return []

    monkeypatch.setattr(s, "search_scopus_by_title", fake_scopus)
    monkeypatch.setattr(s, "search_openalex_by_title", fake_openalex_title)
    monkeypatch.setattr(s, "search_openalex", fake_openalex)

    out = await s.search_by_title("A Title")
    assert seen.get("scopus") == "A Title", "Scopus must get a TITLE() query"
    assert seen.get("openalex_title") == "A Title", "OpenAlex must get a title.search query"
    # The keyword search stays as a recall net, which is the point of the union.
    assert seen.get("keyword") == "A Title"
    # Two indexes returned the same work; it must occupy ONE slot.
    assert len(out) == 1, out


async def test_implausible_hits_are_not_offered_as_candidates(monkeypatch):
    """A hit below the floor must be dropped, not shown as a choice.

    The numbers are the measured ones: 0.199 for an unrelated photocatalysis
    paper against a moiré-physics title, versus 1.0 for the right one.
    """
    from academic_mcp.agent import search as s

    target = "Continuous Mott transition in semiconductor moiré superlattices"
    junk = "Atomistic origins of the sluggish cathodic reduction of alpha-FeOOH"

    async def fake_scopus(title, limit=10, author=None, journal=None):
        return []

    async def fake_openalex_title(title, limit=10):
        return [
            {"doi": "10.1/right", "title": target},
            {"doi": "10.1/junk", "title": junk},
        ]

    async def fake_openalex(query, limit=20, year=None, author=None, journal=None):
        return []

    monkeypatch.setattr(s, "search_scopus_by_title", fake_scopus)
    monkeypatch.setattr(s, "search_openalex_by_title", fake_openalex_title)
    monkeypatch.setattr(s, "search_openalex", fake_openalex)

    out = await s.search_by_title(target)
    assert [p["doi"] for p in out] == ["10.1/right"], out
    # And the floor is what removed it, not the title keying.
    assert s._title_similarity(target, junk) < s.TITLE_CANDIDATE_MIN_SIMILARITY


async def test_a_doi_record_without_a_doi_does_not_duplicate_the_work(monkeypatch):
    """One index returning a DOI-less record must not create a second choice."""
    from academic_mcp.agent import search as s

    title = "Continuous Mott transition in semiconductor moiré superlattices"

    async def fake_scopus(title, limit=10, author=None, journal=None):
        return [{"doi": "", "title": title}]          # Scopus, no DOI

    async def fake_openalex_title(title, limit=10):
        return [{"doi": "10.1038/s41586-021-03853-0", "title": title}]

    async def fake_openalex(query, limit=20, year=None, author=None, journal=None):
        return []

    monkeypatch.setattr(s, "search_scopus_by_title", fake_scopus)
    monkeypatch.setattr(s, "search_openalex_by_title", fake_openalex_title)
    monkeypatch.setattr(s, "search_openalex", fake_openalex)

    out = await s.search_by_title(title)
    assert len(out) == 1, [p.get("doi") for p in out]
    # The DOI-bearing record wins the slot: it is the one that can be downloaded.
    assert out[0]["doi"] == "10.1038/s41586-021-03853-0"


# ── local document conversion: allow-list, force, rerank, locking ──────────
#
# Each of these covers a defect that was silent: a path check that did not exist
# (any local file could be uploaded to MinerU), a `force` that re-downloaded and
# then served stale Markdown, a rerank that quietly did nothing when sklearn was
# absent, per-paper score components read from the wrong row, and a cache lock
# whose file was deleted on release.


def test_doc_roots_allow_list(tmp_path, monkeypatch):
    """Only files under a configured root may be uploaded for conversion.

    The regression: `convert_document` accepted ANY readable path, and a local
    file is uploaded to MinerU's cloud, so `~/.ssh/x.pdf` or a contract `.docx`
    left the machine on request. `ACADEMIC_DOC_ROOTS` is the boundary.
    """
    from academic_mcp.config import Settings

    root = tmp_path / "library"
    root.mkdir()
    inside = root / "paper.pdf"
    inside.write_bytes(b"%PDF-1.4\n")
    nested = root / "sub" / "deep.docx"
    nested.parent.mkdir()
    nested.write_bytes(b"x")

    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF-1.4\n")

    hidden = root / ".ssh" / "key.pdf"
    hidden.parent.mkdir()
    hidden.write_bytes(b"%PDF-1.4\n")

    cfg = Settings(doc_roots_raw=str(root))
    assert cfg.permits_doc_path(inside) is True
    assert cfg.permits_doc_path(nested) is True
    assert cfg.permits_doc_path(outside) is False, "a path outside every root must be refused"
    assert cfg.permits_doc_path(hidden) is False, "dot-directories are never permitted"

    # The error the caller sees must name the roots, so the fix is one step.
    from academic_mcp import mineru

    monkeypatch.setattr(mineru.settings, "doc_roots_raw", str(root))
    with pytest.raises(mineru.MineruError) as excinfo:
        mineru._resolve(str(outside), None)
    assert "permitted document roots" in str(excinfo.value)
    assert str(root) in str(excinfo.value)
    # And a permitted path still resolves.
    is_url, source_id, _hash, path = mineru._resolve(str(inside), None)
    assert (is_url, path) == (False, inside)


def test_permits_doc_path_resolves_symlinks(tmp_path, monkeypatch):
    """A symlink inside an allowed root must not reach outside it."""
    from academic_mcp.config import Settings

    root = tmp_path / "library"
    root.mkdir()
    target = tmp_path / "outside.pdf"
    target.write_bytes(b"%PDF-1.4\n")
    link = root / "link.pdf"
    try:
        link.symlink_to(target)
    except OSError:  # pragma: no cover - filesystem without symlink support
        pytest.skip("symlinks unavailable")

    cfg = Settings(doc_roots_raw=str(root))
    assert cfg.permits_doc_path(link) is False


def _seed_cache(cache, doc, name: str, body: str) -> None:
    """Write a plausible cache entry for `doc` using the service's own key rule.

    Built through `mineru`'s own helpers rather than by hand: a hand-written key
    that happens to disagree is a test that passes for the wrong reason.
    """
    from academic_mcp import mineru

    is_url, source_id, src_hash, local = mineru._resolve(str(doc), None)
    assert is_url is False
    model = mineru.model_for_source(str(doc), local, mineru.settings.mineru_model)
    key = mineru._cache_key(source_id, src_hash, {
        "model": model, "lang": mineru.settings.mineru_language,
        "pages": None, "ocr": None, "formula": True, "table": True,
    })
    (cache / f"{name}.md").write_text(body, encoding="utf-8")
    (cache / f"{name}.meta.json").write_text(
        json.dumps({"cache_key": key, "assets": str(cache / f"{name}_assets"), "outline": []}),
        encoding="utf-8",
    )


def test_convert_document_force_skips_both_caches(tmp_path, monkeypatch):
    """`force=True` must re-convert, and must not be defeated by a cached file.

    The regression was on the paper path: the pipeline re-downloaded the PDF and
    then asked `convert_paper_pdf`, whose first act was `read_md(key)` — so a
    "force" re-fetch returned Markdown produced from the PREVIOUS PDF, at full
    download cost. This pins the document-level half: with `force`, the stored
    `.md` is not what comes back.
    """
    from academic_mcp import mineru

    monkeypatch.setattr(mineru.settings, "doc_roots_raw", str(tmp_path))
    cache = tmp_path / "cache"
    cache.mkdir()
    doc = tmp_path / "doc.pdf"
    doc.write_bytes(b"%PDF-1.4\n")

    calls = {"n": 0}

    def fake_extract(_blob, out_dir, stem):
        calls["n"] += 1
        assets = out_dir / f"{stem}_assets"
        assets.mkdir(parents=True, exist_ok=True)
        md = out_dir / f"{stem}.md"
        text = f"# fresh body {calls['n']}\n"
        md.write_text(text, encoding="utf-8")
        return md, assets

    monkeypatch.setattr(mineru, "_read_capped", lambda *a, **k: (b"zip", False), raising=False)
    monkeypatch.setattr(mineru, "_download", lambda url: b"zip")
    monkeypatch.setattr(mineru, "_extract_zip", fake_extract)
    monkeypatch.setattr(mineru, "_create_task", lambda *a, **k: ("batch", ["http://upload"]))
    monkeypatch.setattr(mineru, "_put_file", lambda *a, **k: None)
    monkeypatch.setattr(mineru, "_poll", lambda *a, **k: {"state": "done", "full_zip_url": "http://zip"})
    monkeypatch.setattr(mineru.settings, "mineru_token", "test-token")

    _seed_cache(cache, doc, "doc", "# stale body\n")

    hit = mineru.convert_document(str(doc), out_dir=cache, name="doc")
    assert "stale body" in hit.text, "without force the cache is served"
    assert calls["n"] == 0, "a cache hit must not convert"

    forced = mineru.convert_document(str(doc), out_dir=cache, name="doc", force=True)
    assert "stale body" not in forced.text, "force must not be defeated by the document cache"
    assert "fresh body" in forced.text
    assert calls["n"] == 1


def test_pages_map_is_rebuilt_on_a_cache_hit(tmp_path, monkeypatch):
    """A cached conversion must still be able to answer `pages_map=true`.

    The regression: `pages_map` was absent from the cache key AND only stored
    when it had been requested, so the first caller who asked for it on an
    already-cached document got `[]` — and because the key matched, they got `[]`
    forever. Rebuilding from the stored sidecar is the fix; this pins it.
    """
    from academic_mcp import mineru

    monkeypatch.setattr(mineru.settings, "doc_roots_raw", str(tmp_path))
    cache = tmp_path / "cache"
    cache.mkdir()
    assets = cache / "doc_assets"
    assets.mkdir()
    (assets / "doc_content_list.json").write_text(
        json.dumps([
            {"type": "text", "page_idx": 0, "text": "first page words"},
            {"type": "text", "page_idx": 1, "text": "second page words"},
        ]),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.pdf"
    doc.write_bytes(b"%PDF-1.4\n")
    md = cache / "doc.md"
    md.write_text("first page words\n\nsecond page words\n", encoding="utf-8")
    (cache / "doc.meta.json").write_text(
        json.dumps({
            "cache_key": mineru._cache_key(str(doc), mineru._sha256_file(doc), {
                "model": mineru.settings.mineru_model, "lang": mineru.settings.mineru_language,
                "pages": None, "ocr": None, "formula": True, "table": True,
            }),
            "assets": str(assets),
            "outline": [],
        }),
        encoding="utf-8",
    )

    out = mineru.convert_document(str(doc), out_dir=cache, name="doc", pages_map=True)
    assert out.meta.get("cached") is True, "this must be a cache hit"
    assert out.pages_map, "pages_map must be rebuilt, not silently empty"
    # And it is persisted, so the next caller does not repeat the rebuild.
    stored = json.loads((cache / "doc.meta.json").read_text(encoding="utf-8"))
    assert stored.get("pages_map")


def test_hybrid_rerank_scores_each_paper_on_its_own_components():
    """`score_components` must belong to the row it is printed on.

    The regression: the reporting loop re-read `sim` and `recency` from the
    scoring loop's variable, so every paper reported the LAST paper's values —
    a number the model is explicitly invited to reason about.
    """
    from academic_mcp.agent import search as s

    papers = [
        {"title": "Exciton exchange coupling in MoTe2 monolayers",
         "abstract": "exchange coupling of excitons", "citation_count": 50, "year": 2021},
        {"title": "Unrelated photocatalysis study",
         "abstract": "water splitting on TiO2", "citation_count": 500, "year": 2019},
    ]
    out = s._hybrid_rerank(papers, "exciton exchange coupling MoTe2")
    assert out[0]["title"].startswith("Exciton"), [p["title"] for p in out]
    top, bottom = out[0]["score_components"], out[1]["score_components"]
    assert top["tfidf_similarity"] > bottom["tfidf_similarity"], (top, bottom)
    assert bottom["tfidf_similarity"] == 0.0, bottom
    # The engine is named, so "which scorer produced this order" is answerable.
    assert out[0]["rerank_engine"] in ("tfidf-sklearn", "tfidf-python")
    assert all(p["score_components"]["citation_norm"] > 0 for p in out)


def test_rerank_never_silently_returns_the_input_order(monkeypatch):
    """Without sklearn the rerank must still reorder by the hybrid score.

    The regression: `except ImportError: return papers` — `rerank=True` is the
    default, so on this deployment (sklearn is not installed and was never a
    declared dependency) the "hybrid rerank" the README advertises was a no-op,
    with no `hybrid_score` on any row to notice it by.
    """
    import builtins

    from academic_mcp.agent import search as s

    real_import = builtins.__import__

    def no_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("sklearn disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sklearn)
    papers = [
        {"title": "Unrelated photocatalysis study", "abstract": "water splitting", "citation_count": 1, "year": 2024},
        {"title": "Exciton exchange coupling in MoTe2", "abstract": "exchange coupling", "citation_count": 1, "year": 2024},
    ]
    out = s._hybrid_rerank(papers, "exciton exchange coupling")
    assert [p["title"] for p in out][0].startswith("Exciton"), "the fallback must actually rank"
    assert out[0]["rerank_engine"] == "tfidf-python"
    assert all("hybrid_score" in p for p in out)


def test_search_cache_lock_file_survives_release(tmp_path, monkeypatch):
    """The `.lock` file must NOT be unlinked on release.

    Unlinking it re-opens the race the lock exists to close: a waiter blocked on
    the old inode and a new writer creating a fresh file both believe they hold
    it, and the merge below loses whichever writer finishes second — visible only
    later, as `validate_doi` refusing a DOI the session really did search for.
    """
    from academic_mcp.agent import search as s

    cache_dir = tmp_path / "search_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(s, "CACHE_DIR", str(cache_dir))

    s.save_search_cache([{"doi": "10.1/x", "title": "t"}], "a query", "--home-alice--")
    lock = cache_dir / "--home-alice--.json.lock"
    assert lock.exists(), "the lock file is left in place on purpose"

    # A second writer must merge, not clobber.
    s.save_search_cache([{"doi": "10.2/y", "title": "u"}], "another query", "--home-alice--")
    data = json.loads((cache_dir / "--home-alice--.json").read_text(encoding="utf-8"))
    assert {"10.1/x", "10.2/y"} <= {d.lower() for d in data["dois"]}, data["dois"]
