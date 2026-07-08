"""Query PROMOTE P7-P12: the versioned table -- time-travel, blame, diff, revert, branch, provable audit."""
import numpy as np
from holographic.agents_and_reasoning.holographic_query import UserTable
from holographic.agents_and_reasoning.holographic_query_history import VersionedTable, _row_hash
from holographic.agents_and_reasoning.holographic_deltachain import merkle_root


def _vt():
    t = UserTable("accounts", ["id", "name", "balance"], dim=1024, seed=0)
    t.insert({"id": 1, "name": "alice", "balance": 100})
    t.insert({"id": 2, "name": "bob", "balance": 50})
    vt = VersionedTable(t)
    v0 = vt.commit("initial")
    t.rows[1]["balance"] = 75
    t.insert({"id": 3, "name": "carol", "balance": 200})
    v1 = vt.commit("bob paid, carol joined")
    return vt, v0, v1


def test_p7_time_travel():
    vt, v0, v1 = _vt()
    assert len(vt.select_as_of("SELECT id FROM accounts", v0)) == 2
    assert len(vt.select_as_of("SELECT id FROM accounts", v1)) == 3


def test_p7_checkout_rebuilds_exact():
    vt, v0, v1 = _vt()
    past = vt.checkout(v0)
    assert len(past) == 2 and {r["name"] for r in past.rows} == {"alice", "bob"}


def test_p9_diff_by_key():
    vt, v0, v1 = _vt()
    d = vt.diff(v0, v1, key="id")
    assert [r["name"] for r in d["added"]] == ["carol"]
    assert d["changed"][0]["key"] == 2 and d["changed"][0]["to"]["balance"] == 75
    assert d["removed"] == []


def test_p9_keyless_diff_added_removed():
    vt, v0, v1 = _vt()
    d = vt.diff(v0, v1)                                       # no key -> multiset content diff
    assert any(r["name"] == "carol" for r in d["added"])
    assert any(r.get("balance") == 50 for r in d["removed"])  # bob's old version shows as removed


def test_p8_blame():
    vt, v0, v1 = _vt()
    tl = vt.history_of(2, key="id")
    assert tl[0][1]["balance"] == 50 and tl[1][1]["balance"] == 75


def test_p10_revert_recorded_as_new_version():
    vt, v0, v1 = _vt()
    new_head = vt.revert(v0)
    assert len(vt.table) == 2 and new_head == 2               # back to v0's rows, but as a NEW version


def test_p11_branch_diverges_without_touching_main():
    vt, v0, v1 = _vt()
    br = vt.branch()
    br.table.insert({"id": 4, "name": "dave", "balance": 10})
    br.commit("dave")
    assert len(br.diff(vt.head(), br.head(), key="id")["added"]) == 1
    assert len(vt.table) == 3                                 # main unaffected


def test_p12_prove_and_locate_tampering():
    vt, v0, v1 = _vt()
    tampered = [dict(r) for r in vt._dict_snaps[v1]]
    tampered[0]["balance"] = 999999
    assert vt.find_tampering(tampered, v1) == [0]
    assert vt.prove(v1) != merkle_root([_row_hash(r) for r in tampered])
    assert vt.find_tampering([dict(r) for r in vt._dict_snaps[v1]], v1) == []   # untampered -> clean


def test_deterministic_proof():
    vt, v0, v1 = _vt()
    assert vt.prove(v1) == vt.prove(v1)
