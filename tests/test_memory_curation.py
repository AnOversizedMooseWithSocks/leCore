"""MEMORY CURATION, DECAY AND REFLECTION -- the traps for NOOA section 6 item 5.

WHY THIS FILE EXISTS, and what it is guarding against. `docs/COMPETITIVE_NOOA.md` rates leCore
PARTIAL here -- "recall exists; the curation/decay/reflection subsystem does not" -- and the
obvious way to close that gap is the way this engine has already been burned by. sweep 129's
`unicron_turn_memory` records it in capitals: "ACT-R eviction made that survivable by overwriting
the least active slot, BUT EVICTION IS A LOSS, AND A FLAT FILE IS THE REASON IT WAS NEEDED." The
eviction turned out to be an artefact of a flat layout, and the fix was a better layout that
forgot nothing.

So the load-bearing property here is NOT that curation ranks well. It is that **curation cannot
destroy a taught fact**, and `test_nothing_is_ever_destroyed` asserts it as a conservation law:
rows_after + archived == rows_before, always. A curation pass that silently drops something a user
taught is a data-loss bug wearing a feature's clothes.

The second thing this file guards is the HONESTY of the measurement. There is no widely adopted
public benchmark that scores forgetting, decay or consolidation, so the numbers are self-measured
against baselines constructed in the module -- and `test_activation_LOSES_to_a_recency_window_
under_a_full_topic_switch` pins the loss as hard as the wins, because that is the result this
module is most tempted to quietly stop reporting.
"""
import json
import subprocess
import sys

import pytest

import lecore
from holographic.caching_and_storage.holographic_memcurate import (
    MemoryCurator, curation_benchmark, drift_sweep, fact_id, normalize_question, rows_from_mind)

ROWS = [("what is the capital of france", "paris"),
        ("who wrote dune", "frank herbert"),
        ("what is the boiling point of water", "100 C"),
        ("what is the capital of france", "paris, france")]      # a RE-TEACH


@pytest.fixture
def curator():
    c = MemoryCurator(rows=ROWS)
    c.observe("who wrote dune")
    c.observe("who wrote dune")
    c.observe("what is the capital of france")
    return c


# --------------------------------------------------------------------------------------
# 1. THE PROPERTY THAT MATTERS: nothing is ever destroyed.
# --------------------------------------------------------------------------------------

def test_nothing_is_ever_destroyed(curator):
    # A CONSERVATION LAW, asserted as one. "eviction is a loss" is on record in this repo; the
    # answer is that this module has no delete path at all, only demotion.
    before = len(curator.rows)
    curator.apply(keep=1)
    assert len(curator.rows) + len(curator.archive) == before
    assert len(curator.archive) == 3 and len(curator.rows) == 1


def test_restore_returns_the_store_byte_for_byte(curator):
    before = [dict(r) for r in curator.rows]
    curator.apply(keep=1)
    out = curator.restore()
    assert out["restored"] == 3 and out["archived"] == 0
    assert [dict(r) for r in curator.rows] == before


def test_the_module_has_no_delete_path_at_all():
    # The guard against a future sweep adding one "just for the archive". If a delete lands, it
    # must land deliberately, with this test changed and the reason written down.
    import inspect

    import holographic.caching_and_storage.holographic_memcurate as mod
    src = inspect.getsource(mod)
    assert "def delete" not in src and "def evict" not in src and "def drop" not in src
    assert "POLICIES = (\"archive\", \"plan\")" in src


def test_every_demotion_carries_a_reason(curator):
    j = curator.apply(keep=1)["journal"]
    assert len(j) == 3
    assert all(r["reason"] for r in j), "a silent demotion is worse than a loud failure"
    assert {r["action"] for r in j} == {"archived", "superseded"}
    assert all(r["restorable_as"] for r in j)


def test_write_through_round_trips_the_real_partition(tmp_path):
    m = lecore.UnifiedMind(dim=256, seed=0)
    for q, a in [("q one", "a1"), ("q two", "a2"), ("q three", "a3")]:
        m.teach(q, a)
    lad = m.zoo["ladder"]
    before = [list(r) for r in lad.taught_log]
    c = MemoryCurator(m)
    c.observe("q one")
    r = c.apply(keep=1, write_through=True)
    assert r["wrote_through"] == 1 and len(lad.taught_log) == 1
    c.restore(write_through=True)
    assert [list(x) for x in lad.taught_log] == before, "the partition was not restored exactly"


# --------------------------------------------------------------------------------------
# 2. DEFAULT-OFF: an existing partition behaves identically until curation is asked for.
# --------------------------------------------------------------------------------------

def test_building_a_curator_does_not_touch_the_partition():
    m = lecore.UnifiedMind(dim=256, seed=0)
    m.teach("a question", "an answer")
    lad = m.zoo["ladder"]
    before = [list(r) for r in lad.taught_log]
    c = m.memory_curate()
    c.plan(keep=0)
    c.reflect()
    c.ranked()
    assert [list(r) for r in lad.taught_log] == before, "a read-only call mutated the store"


def test_apply_without_write_through_leaves_the_partition_alone():
    m = lecore.UnifiedMind(dim=256, seed=0)
    for q in ("one", "two", "three"):
        m.teach(q, q.upper())
    before = [list(r) for r in m.zoo["ladder"].taught_log]
    c = m.memory_curate()
    c.apply(keep=1)                                   # write_through defaults to False
    assert [list(r) for r in m.zoo["ladder"].taught_log] == before


def test_plan_is_pure(curator):
    snap = [dict(r) for r in curator.rows]
    for _ in range(3):
        curator.plan(keep=1, threshold=-10.0)
    assert [dict(r) for r in curator.rows] == snap and curator.archive == {}


# --------------------------------------------------------------------------------------
# 3. DETERMINISM, and time that is logical rather than wall-clock.
# --------------------------------------------------------------------------------------

def test_no_wall_clock_anywhere_in_the_decision_path():
    # A decay that reads the system clock gives a different answer on Tuesday, which makes every
    # downstream curation decision irreproducible. Asserted at the SOURCE level because that is
    # the only form of the check a future edit cannot slip past.
    import inspect

    import holographic.caching_and_storage.holographic_memcurate as mod
    src = inspect.getsource(mod)
    for banned in ("import time", "import datetime", "time.time(", "datetime.now",
                   "time.monotonic", "perf_counter"):
        assert banned not in src, "wall-clock reference %r in the decision path" % banned


def test_the_same_partition_curates_identically_twice(curator):
    a = json.dumps(curator.plan(keep=2), default=str)
    b = json.dumps(MemoryCurator(rows=ROWS), default=str) and None
    c2 = MemoryCurator(rows=ROWS)
    for q in ("who wrote dune", "who wrote dune", "what is the capital of france"):
        c2.observe(q)
    assert json.dumps(c2.plan(keep=2), default=str) == a


def test_fact_ids_are_hashlib_stable_across_processes():
    # hashlib, never hash(): an audit journal has to be comparable with the run that produced it.
    code = ("import sys; sys.path.insert(0, %r); "
            "from holographic.caching_and_storage.holographic_memcurate import fact_id; "
            "print(fact_id('who wrote dune'))" % _repo())
    here = fact_id("who wrote dune")
    for seed in ("0", "1", "99999"):
        env = _env(seed)
        got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=env, check=True).stdout.strip()
        assert got == here, "fact id moved with PYTHONHASHSEED=%s" % seed
    assert fact_id("a") == "ca978112ca1bbdca"          # sha256('a')[:16], fixed forever


def test_identity_survives_retyping():
    assert normalize_question("  What IS   the  Capital ") == "what is the capital"
    assert fact_id("Who Wrote  DUNE") == fact_id("who wrote dune")


def _repo():
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(seed):
    import os
    e = dict(os.environ)
    e["PYTHONHASHSEED"] = seed
    e["PYTHONPATH"] = _repo() + os.pathsep + e.get("PYTHONPATH", "")
    return e


# --------------------------------------------------------------------------------------
# 4. THE THREE LEGS.
# --------------------------------------------------------------------------------------

def test_reflection_makes_a_re_taught_question_one_fact_with_a_history(curator):
    p = curator.plan(keep=2)
    assert p["counts"]["rows"] == 4 and p["counts"]["distinct_facts"] == 3
    sup = p["superseded"]
    assert len(sup) == 1
    assert sup[0]["row"]["born"] == 0 and sup[0]["superseded_by"] == 3
    assert "re-taught" in sup[0]["reason"]
    # the CURRENT answer is the newest one, and the older text is archived, not gone
    hot = {r["question"]: r["answer"] for r in p["hot"]}
    assert hot["what is the capital of france"] == "paris, france"
    curator.apply(p)
    assert any(v["answer"] == "paris" for v in curator.archive.values())


def test_decay_ranks_recency_and_frequency_together(curator):
    ranked = curator.ranked()
    assert [n for n, _a in ranked][:2] == [fact_id("what is the capital of france"),
                                           fact_id("who wrote dune")]
    # the untouched fact ranks last
    assert ranked[-1][0] == fact_id("what is the boiling point of water")


def test_a_retrieval_threshold_abstains_rather_than_returning_the_least_bad(curator):
    hi = curator.plan(threshold=10.0)          # nothing can clear this
    assert hi["counts"]["hot"] == 0 and hi["counts"]["archive"] == 3
    assert all("below threshold" in e["reason"] for e in hi["archive"])
    lo = curator.plan(threshold=-1e9)
    assert lo["counts"]["archive"] == 0


def test_curation_bounds_the_hot_set(curator):
    for k in (0, 1, 2, 3):
        p = curator.plan(keep=k)
        assert p["counts"]["hot"] == min(k, 3)


# --------------------------------------------------------------------------------------
# 5. THE MEASUREMENT -- self-measured, and the loss pinned as hard as the wins.
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep():
    return {r["drift"]: r for r in drift_sweep()["rows"]}


def test_keep_everything_is_the_unbounded_upper_bound(sweep):
    for row in sweep.values():
        assert row["keep_all"] == 1.0
        assert row["random"] == 0.2, "40 hot of 200 facts is a 0.2 no-information baseline"


def test_every_history_informed_policy_beats_chance(sweep):
    # THE DIAGNOSTIC THAT CAUGHT THE HARNESS'S OWN BUG. The first stream made drift a fraction of
    # RANK slots; under Zipf(1.1) the top 25 of 200 ranks carry 71% of the traffic, so history
    # became anti-predictive and all three policies scored 0.086-0.13 against a 0.20 random
    # baseline. Three policies losing to chance is a broken generator, not a finding.
    for d_, row in sweep.items():
        for pol in ("recency", "frequency", "actr"):
            assert row[pol] > 0.55, "%s at drift %s scored %.4f" % (pol, d_, row[pol])


def test_activation_beats_a_recency_window_where_both_signals_exist(sweep):
    for d_ in (0.0, 0.25, 0.5, 0.75):
        row = sweep[d_]
        assert row["beats_recency"] is True, (d_, row["ci_recency"])
        assert row["ci_recency"][0] > 0.0, "the interval must clear zero to be a win"


def test_activation_LOSES_to_a_recency_window_under_a_full_topic_switch(sweep):
    # THE NEGATIVE, PINNED. Under drift=1.0 the history's frequency signal is worthless and a
    # plain recency window wins by a margin whose whole interval is below zero. If a future
    # change makes this test pass by making the loss disappear, the harness has been bent.
    row = sweep[1.0]
    assert row["beats_recency"] is False
    assert row["actr_minus_recency"] < -0.02
    assert row["ci_recency"][1] < 0.0, "the loss is significant, not a wobble"
    assert row["winner"] == "recency"


def test_activation_only_ties_frequency_on_a_stationary_process(sweep):
    for d_ in (0.0, 0.25):
        row = sweep[d_]
        assert row["beats_frequency"] is False
        assert row["ci_frequency"][0] < 0.0 < row["ci_frequency"][1], "a tie, not a loss"
    for d_ in (0.5, 0.75, 1.0):
        assert sweep[d_]["beats_frequency"] is True


def test_the_result_is_labelled_self_measured_everywhere():
    # The label has to travel with the number or it stops travelling.
    b = curation_benchmark()
    s = drift_sweep(seeds=(0, 1))
    assert b["self_measured"] is True and s["self_measured"] is True
    for note in (b["note"], s["note"]):
        assert "LoCoMo" in note and "not comparable" in note
    import holographic.caching_and_storage.holographic_memcurate as mod
    # phrases chosen to survive the docstring's own line wrapping -- the first version of this
    # assertion matched across a newline and failed on prose that was in fact correct
    assert "SELF-MEASURED" in mod.__doc__
    assert "widely adopted public benchmark" in mod.__doc__ and "LoCoMo" in mod.__doc__


# --------------------------------------------------------------------------------------
# 6. WIRING.
# --------------------------------------------------------------------------------------

def test_the_faculty_is_wired_to_the_mind_and_documented():
    m = lecore.UnifiedMind(dim=256, seed=0)
    assert callable(getattr(m, "memory_curate", None))
    doc = m.memory_curate.__doc__ or ""
    assert doc.strip(), "an undocumented verb is undiscoverable"
    assert "eviction is a loss" in doc, "the kept negative must travel with the door"
    assert "SELF-MEASURED" in doc
    m.teach("x", "y")
    c = m.memory_curate()
    assert isinstance(c, MemoryCurator) and len(c.rows) == 1


def test_rows_from_mind_reads_an_empty_partition_without_complaint():
    m = lecore.UnifiedMind(dim=256, seed=0)
    assert rows_from_mind(m) == [] or all("question" in r for r in rows_from_mind(m))
    c = m.memory_curate()
    assert c.plan()["counts"]["rows"] == len(c.rows)
