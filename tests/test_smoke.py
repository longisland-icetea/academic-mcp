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
