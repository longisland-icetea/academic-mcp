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
