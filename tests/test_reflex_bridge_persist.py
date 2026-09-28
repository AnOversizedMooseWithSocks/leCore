"""tests/test_reflex_bridge_persist.py -- sweep 177: the reflex arc connected to the learning stuff.

Two measured gaps, each pinned here by the probe that exposed it:

  1. PERSISTENCE. experience_save/experience_load (and the partition container that learning_save /
     boot use) carried the trace but not the reflex bridge's label book and seen gate, so a second mind
     -- or the same mind after a restart -- answered every reported repeat with "no experience yet".
     Probe before the fix: A fires via='reflex'; B loads A's 91,561-byte save; B -> "no experience yet".

  2. THE JUDGE NEVER TAUGHT. swarm_step ran its verification and wrote the step to the ledger and the
     bus, but never called reflex_learn -- the reflex only learned from caller-reported outcomes.

Every assert below is a behaviour, not a rate: the rates live in tools/bench_swarm_reflex.py.
"""
import os

import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")

QUERY = "smooth a bumpy mesh"


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def _teach_one_route(m):
    """Route QUERY, report the answer as correct, and return the answer the reflex should give back."""
    r = m.route_tiered(QUERY, reflex=True)
    assert r["tier"] == "answer"
    m.decision_outcome(r["id"], r["answer"])
    again = m.route_tiered(QUERY, reflex=True)
    assert again.get("via") == "reflex"                  # sanity: A itself answers from experience
    return r["answer"]


# ---------------------------------------------------------------- 1. persistence ----------------

def test_second_mind_fires_from_first_minds_experience_file(tmp_path):
    """The exact probe that failed before sweep 177, now required to pass."""
    a = _mind()
    answer = _teach_one_route(a)
    path = str(tmp_path / "exp.json")
    a.experience_save(path)

    b = _mind()
    assert b.reflex_decide(QUERY)["value"] is None     # a blank mind knows nothing (control)
    rep = b.experience_load(path)
    assert rep["reflex"]["seen"] >= 1 and rep["reflex"]["labels"] >= 1
    d = b.reflex_decide(QUERY)
    assert d["value"] == answer and d["via"] == "reflex"


def test_unrelated_query_still_refused_after_load(tmp_path):
    """Carrying the seen gate must not widen it: a question nobody reported stays refused."""
    a = _mind()
    _teach_one_route(a)
    path = str(tmp_path / "exp.json")
    a.experience_save(path)
    b = _mind()
    b.experience_load(path)
    assert b.reflex_decide("render a turntable of a crystal cluster")["value"] is None


def test_old_experience_file_without_bridge_loads_as_before(tmp_path):
    """Backward compatibility: a pre-177 file (no 'reflex' key) loads, and simply has no bridge."""
    import json
    a = _mind()
    _teach_one_route(a)
    path = str(tmp_path / "exp.json")
    a.experience_save(path)
    st = json.load(open(path))
    st.pop("reflex")
    json.dump(st, open(path, "w"))
    b = _mind()
    rep = b.experience_load(path)
    assert "reflex" not in rep and b.reflex_decide(QUERY)["value"] is None


def test_partition_container_round_trip(tmp_path):
    """learning_save -> learning_load (what boot does) carries the bridge section."""
    a = _mind()
    answer = _teach_one_route(a)
    root = str(tmp_path / "part")
    os.makedirs(os.path.join(root, "learning"), exist_ok=True)
    a.learning_save(root)
    b = _mind()
    b.learning_load(root)
    d = b.reflex_decide(QUERY)
    assert d["value"] == answer


def test_merge_import_is_idempotent():
    """Importing another agent's bridge twice adds nothing the second time."""
    a = _mind()
    _teach_one_route(a)
    st = a.reflex_bridge_state()
    b = _mind()
    first = b.reflex_bridge_restore(st, merge=True)
    second = b.reflex_bridge_restore(st, merge=True)
    assert first["added"] == len(st["kinds"]) and second["added"] == 0


# ---------------------------------------------------------------- 2. the judge teaches ----------

def _verified_step(m, state, n=500):
    return m.swarm_step(state, "catalog_families", done_when="%d resolve" % n, evidence={"claimed": n},
                        verify={"verb": "catalog_families", "args": {}},
                        expect=lambda f: sum(1 for v in f.values() if v[0]) >= n)


def test_verified_swarm_step_teaches_the_reflex():
    m = _mind()
    m.set_file_root(".")
    task = "count the catalog families that resolve"
    s = _verified_step(m, task)
    assert s["reflex"] and s["reflex"]["learned"] is True and s["outcome"] == "catalog_families"
    d = m.reflex_decide(task, key="ngram")
    assert d["value"] == "catalog_families"


def test_unverified_swarm_step_does_not_teach():
    """Self-reported steps are exactly what the fix excludes."""
    m = _mind()
    s = m.swarm_step("an unjudged claim", "catalog_families", done_when="trust me", evidence={"claimed": 1})
    assert s["reflex"] is None
    assert m.reflex_decide("an unjudged claim", key="ngram")["value"] is None


def test_refused_swarm_step_teaches_nothing():
    m = _mind()
    m.set_file_root(".")
    with pytest.raises(ValueError):
        _verified_step(m, "claim an impossible count", n=10 ** 6)
    assert m.reflex_decide("claim an impossible count", key="ngram")["value"] is None
    assert not (getattr(m, "_reflex_seen", None) or [])
