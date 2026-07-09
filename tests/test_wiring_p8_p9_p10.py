"""P8 / P9 / P10 -- the silo-elimination round: capability that existed, worked, had tests, and was reachable by
nothing. Each test here exercises the DOOR that was added, not the module behind it (which had its own tests all
along). If a door is removed, this fails.
"""
import os
import tempfile

import numpy as np
import pytest


# ----------------------------------------------------------------------------------------------------------
# P9 -- `hardening` and `ablate` were "catalog-only": listed as capabilities, called by no engine file.
# ----------------------------------------------------------------------------------------------------------
def test_refine_retries_transient_failures_via_hardening():
    from holographic.misc.holographic_refine import refine
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return 1.0

    log = refine(produce=flaky, critique=lambda r: 1.0, adjust=lambda r, s: r,
                 accept=0.9, budget=2, attempts=5, backoff=0.0)
    assert log["accepted"] and calls["n"] == 3          # survived two transient failures


def test_refine_default_is_unchanged_and_still_raises():
    """attempts=1 (the default) must behave exactly as before -- no retry, the error surfaces."""
    from holographic.misc.holographic_refine import refine
    with pytest.raises(RuntimeError):
        refine(produce=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
               critique=lambda r: 1.0, adjust=lambda r, s: r)


def test_measure_reaches_the_fdr_gate():
    from holographic.misc.holographic_measure import fdr_gate
    from holographic.misc.holographic_ablate import verdict
    rng = np.random.default_rng(0)
    strong = {"mean": 0.9, "ci": (0.85, 0.95), "scores": list(0.9 + 0.01 * rng.standard_normal(8))}
    weak = {"mean": 0.5, "ci": (0.45, 0.55), "scores": list(0.5 + 0.01 * rng.standard_normal(8))}
    rows = [("strong", strong, weak, "base", verdict(strong, weak)),
            ("null", weak, weak, "base", verdict(weak, weak))]
    aug, n_load_bearing, n_survive = fdr_gate(rows, alpha=0.1)
    assert n_load_bearing == 1 and n_survive == 1
    assert aug[0][5] < 0.05 and aug[1][5] > 0.5        # the real effect survives; the null does not


def test_coordinator_exposes_the_hardened_path():
    from holographic.scene_and_pipeline.holographic_coordinator import hardened, InProcessBackend, _sum_bucket
    from holographic.scene_and_pipeline.holographic_distribute import reduce_sum
    got = hardened(InProcessBackend(), redundancy=3, attempts=2).run([[1, 2], [3, 4]], _sum_bucket, None, reduce_sum)
    assert got == 10.0                                  # redundant runs agree -> the voted answer


def test_mind_farm_takes_redundancy_and_attempts():
    import inspect
    from holographic.misc.holographic_unified import UnifiedMind
    params = inspect.signature(UnifiedMind.farm).parameters
    assert "redundancy" in params and "attempts" in params
    assert params["redundancy"].default == 1 and params["attempts"].default == 1   # trusted pool pays nothing


# ----------------------------------------------------------------------------------------------------------
# P8 -- the nine query_* satellites: durability, locking, concurrency, history, graph. Imported by nothing.
# ----------------------------------------------------------------------------------------------------------
def _db():
    from holographic.agents_and_reasoning.holographic_query import Database
    db = Database()
    db.add_namespace("shop")
    db.create_table("shop.items", ["name", "qty"], dim=128, seed=0)
    db.resolve("shop.items").insert({"name": "nail", "qty": "10"})
    return db


def test_database_durability_layer_round_trips():
    from holographic.agents_and_reasoning.holographic_query import Database, run_sql
    db = _db()
    path = os.path.join(tempfile.mkdtemp(), "snap.json")
    db.snapshot(path)
    assert os.path.exists(path)
    restored = Database.restore(path)                   # a CONSTRUCTOR, not an in-place load
    assert [r["name"] for r in run_sql("SELECT name FROM items", restored.resolve("shop.items"))] == ["nail"]


def test_database_concurrency_and_history_layers():
    db = _db()
    assert type(db.writer_lock()).__name__ == "SingleWriterLock"
    snap = db.snapshot_reader("shop.items")
    db.resolve("shop.items").insert({"name": "screw", "qty": "5"})
    assert len(snap.rows) == 1                          # the point-in-time read never sees the later write
    assert type(db.versioned("shop.items")).__name__ == "VersionedTable"
    assert type(db.journal(os.path.join(tempfile.mkdtemp(), "j.log"))).__name__ == "Journal"


def test_database_graph_layer():
    db = _db()
    db.add_namespace("g")
    db.create_table("g.edges", ["src", "dst"], dim=64, seed=0)
    e = db.resolve("g.edges")
    e.insert({"src": "a", "dst": "b"})
    e.insert({"src": "b", "dst": "c"})
    assert db.adjacency("g.edges", "src", "dst") == {"a": ["b"], "b": ["c"]}


# ----------------------------------------------------------------------------------------------------------
# P10 -- `compose` and `diffusion` were DARK: zero engine references. Real capability, no door.
# ----------------------------------------------------------------------------------------------------------
def test_compose_forward_generation_through_the_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=256, seed=0)
    specs = m.novel_object_specs()
    assert len(specs) > 10 and {"colour", "shape", "texture"} <= set(specs[0])
    one = m.compose_from_tags(specs[0])
    scene = m.compose_from_tags(specs[:3])
    assert one.shape == scene.shape and one.ndim == 1
    assert not np.allclose(one, scene)                  # a scene of 3 is not the same vector as one object


def test_regime_detector_fires_on_a_step_change_not_on_noise():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(0)

    quiet = m.regime_detector()
    flags = [quiet.observe([0.0 + 0.01 * rng.standard_normal()])[2] for _ in range(12)]
    assert not any(flags)                               # a wobble is not a regime change

    stepped = m.regime_detector()
    flags = [stepped.observe([0.0])[2] for _ in range(3)] + [stepped.observe([5.0])[2] for _ in range(3)]
    assert any(flags)                                   # a sustained step is


def test_the_two_dark_modules_now_have_engine_callers():
    """The point of P10: not that the code works (it always did) but that something can REACH it."""
    import subprocess
    for module in ("holographic_compose", "holographic_diffusion"):
        out = subprocess.run(["grep", "-rl", module, "holographic/"], capture_output=True, text=True).stdout
        callers = [f for f in out.split() if not f.endswith(module + ".py")]
        assert callers, "%s is dark again -- nothing in the engine references it" % module


# ======================================================================================================
# The audit tool's own regression trap: an audit that degrades SILENTLY is worse than no audit.
# ======================================================================================================
def test_a_parse_failure_is_recorded_loudly_and_not_swallowed():
    """`_imports` used to return an empty set on SyntaxError. An unparseable file therefore contributed ZERO
    references, and every module it alone reached was falsely reported DARK -- with nothing printed. Since
    `holographic_unified.py` alone references 342 modules, one broken edit could have flooded the report while the
    report claimed to be fine. Failures are now collected, printed, and fail `--check`."""
    import os
    import sys
    import tempfile

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import wiring_report as wr

    before = len(wr.PARSE_FAILURES)
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "good.py")
        bad = os.path.join(d, "bad.py")
        open(good, "w").write("from holographic.misc.holographic_pack import pack\n")
        open(bad, "w").write("from holographic.misc.holographic_pack import (\n")

        assert wr._imports(good) == {"holographic_pack"}
        assert len(wr.PARSE_FAILURES) == before          # a healthy file records nothing

        assert wr._imports(bad) == set()                 # still returns empty -- but no longer silently
        assert len(wr.PARSE_FAILURES) == before + 1
        path, why = wr.PARSE_FAILURES[-1]
        assert path == bad and why                        # the reason, with a line number, is captured

    del wr.PARSE_FAILURES[before:]                        # leave the module-level list as we found it


def test_the_live_tree_parses_so_the_dark_report_can_be_trusted():
    """The condition under which every other assertion in this file means anything."""
    import os
    import sys

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import wiring_report as wr

    before = len(wr.PARSE_FAILURES)
    wr.analyse(REPO)
    new = wr.PARSE_FAILURES[before:]
    assert not new, "these modules do not parse, so the import graph is incomplete: %s" % new
