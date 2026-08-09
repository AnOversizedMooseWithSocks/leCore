"""Tests for the product-facing LocalAgentCore facade."""

import pytest

from holographic_product import LocalAgentCore, demo


def test_local_agent_core_remembers_and_recalls():
    core = LocalAgentCore(dim=256, seed=0)
    core.remember("render scenes with global illumination and light caches", label="render")
    core.remember("local agents need deterministic durable memory", label="memory")

    hits = core.recall("deterministic local memory", k=2)
    assert hits[0]["label"] == "memory"
    assert hits[0]["score"] >= hits[1]["score"]


def test_recall_is_query_safe_and_deterministic():
    core = demo()
    before = core.to_state()
    a = core.recall("deterministic local memory")
    b = core.recall("deterministic local memory")
    after = core.to_state()

    assert a == b
    assert before == after


def test_recall_rejects_invalid_k_and_abstains_on_empty_queries():
    core = demo()

    with pytest.raises(ValueError, match="positive integer"):
        core.recall("memory", k=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        core.recall("memory", abstain=2)
    assert core.recall("") == []


def test_memory_crud_is_ordered_paginated_and_idempotent():
    core = LocalAgentCore(dim=128, seed=0)
    first = core.remember("first memory", label="first")
    second = core.remember("second memory", label="second")
    third = core.remember("third memory", label="third")

    page_one = core.list_memories(limit=2)
    page_two = core.list_memories(limit=2, cursor=page_one["next_cursor"])

    assert [row["id"] for row in page_one["items"]] == [first["id"], second["id"]]
    assert page_one["next_cursor"] == second["id"]
    assert [row["id"] for row in page_two["items"]] == [third["id"]]
    assert page_two["next_cursor"] is None
    assert core.get_memory(second["id"]) == second
    assert core.get_memory("missing") is None
    updated = core.update_memory(
        second["id"],
        text="updated second memory",
        label=None,
        metadata={"revision": 2},
    )
    assert updated == {
        "id": second["id"],
        "text": "updated second memory",
        "label": None,
        "metadata": {"revision": 2},
    }
    assert core.update_memory("missing", text="ignored") is None
    assert core.recall("updated second", k=1)[0]["id"] == second["id"]
    assert core.forget(second["id"]) == updated
    assert core.forget(second["id"]) is None
    assert [entry.id for entry in core.entries] == [first["id"], third["id"]]
    with pytest.raises(ValueError, match="cursor"):
        core.list_memories(cursor="missing")


def test_route_uses_existing_skill_catalog():
    core = LocalAgentCore(dim=128, seed=0)
    routed = core.route("start pause resume cancel a job")

    assert routed["task"] == "start pause resume cancel a job"
    assert routed["decision"] == "act"
    assert "call" in routed["skill"]


def test_dashboard_reports_product_evidence():
    core = demo()
    data = core.dashboard()
    page = core.dashboard(html=True)

    assert data["name"] == "leCore LocalAgentCore"
    assert data["memory"]["entries"] == 3
    assert data["checks"]["deterministic_encoding"] is True
    assert "c_kernel" in data
    assert "leCore LocalAgentCore" in page
    assert "No Model Weights" in page


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "agent-core.json"
    core = demo()
    core.save(path)

    loaded = LocalAgentCore.load(path)

    assert loaded.to_state() == core.to_state()
    assert loaded.recall("audited c kernel hot path")[0]["label"] == "c-kernel"


def test_lecore_exports_product_area():
    import lecore

    assert "product" in lecore.areas()
    core = lecore.product.LocalAgentCore(dim=128, seed=1)
    core.remember("agent memory product facade", label="product")
    assert core.recall("agent memory")[0]["label"] == "product"
