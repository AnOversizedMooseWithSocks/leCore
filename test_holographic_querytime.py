"""Tests for holographic_querytime -- the versioned-history promote layer (P7-P12)."""
import numpy as np
from holographic_query import Database, update
from holographic_querytime import (TableHistory, select_as_of, history_of, diff_versions, revert_to,
                                   branch, compare, discard, prove, find_tampering)


def _fixture():
    db = Database(); db.add_namespace("user")
    db.create_table("user.acct", ["id", "balance", "status"], dim=1024, seed=0)
    t = db.namespaces["user"]["tables"]["acct"]; t.set_primary_key("id")
    for r in [{"id": 1, "balance": 100, "status": "open"}, {"id": 2, "balance": 50, "status": "open"}]:
        t.insert(r)
    h = TableHistory(t); v0 = h.commit(t, note="open")
    update(t, "id = 1", {"balance": 250}); v1 = h.commit(t, note="deposit")
    t.insert({"id": 3, "balance": 0, "status": "open"}); update(t, "id = 2", {"status": "closed"})
    v2 = h.commit(t, note="add 3, close 2")
    return h, (v0, v1, v2)


def test_p7_time_travel():
    h, (v0, v1, v2) = _fixture()
    assert select_as_of(h, v0, "SELECT balance FROM acct WHERE id = 1")[0]["balance"] == 100
    assert select_as_of(h, v2, "SELECT balance FROM acct WHERE id = 1")[0]["balance"] == 250


def test_p8_blame():
    h, (v0, v1, v2) = _fixture()
    blame = history_of(h, "id", 1)
    assert [b["row"]["balance"] for b in blame] == [100, 250, 250]
    assert history_of(h, "id", 3)[0]["row"] is None and history_of(h, "id", 3)[2]["row"]["id"] == 3


def test_p9_diff():
    h, (v0, v1, v2) = _fixture()
    d = diff_versions(h, v0, v2)
    assert d["n_added"] == 1 and d["added"][0]["id"] == 3
    assert {c["key"] for c in d["changed"]} == {1, 2}
    # field-level detail: id 2's status flipped
    id2 = next(c for c in d["changed"] if c["key"] == 2)
    assert id2["fields"]["status"] == ("open", "closed")


def test_p10_revert():
    h, (v0, v1, v2) = _fixture()
    reverted = revert_to(h, v0)
    assert next(r for r in reverted.rows if r["id"] == 1)["balance"] == 100
    assert len(h._versions) == 3                                    # revert doesn't rewrite history


def test_p11_branch_compare_discard():
    h, (v0, v1, v2) = _fixture()
    b = branch(h, at_version=v2)
    bt = b.checkout(-1); update(bt, "id = 1", {"balance": 999}); b.commit(bt, note="what-if")
    cmp = compare(b)
    assert any(c["key"] == 1 for c in cmp["changed"])
    assert len(h._versions) == 3                                    # compare must not mutate the parent
    assert discard(b) is True and b._versions == []


def test_p12_prove_and_locate_tamper():
    h, (v0, v1, v2) = _fixture()
    root = prove(h, v2)
    assert isinstance(root, str) and root != "empty"
    suspect = h._versions[v2]["records"].copy(); suspect[1] += 0.01
    assert find_tampering(h, v2, suspect) == 1                      # locates the altered row
    assert find_tampering(h, v2, h._versions[v2]["records"]) is None  # untampered


def test_checkout_is_queryable_and_deterministic():
    h, (v0, v1, v2) = _fixture()
    a = prove(h, v0); b = prove(h, v0)
    assert a == b                                                   # deterministic commitment
