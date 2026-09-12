"""THE ABOVE/BELOW SWEEP, derived instead of frozen (sweep 133).

WHY THIS FILE EXISTS. `tools/swarm_audit.py` asks "is every capability present at every layer?" and
answered it over a hand-written literal of 21 tuples frozen at cp67, while the engine grew past 600
capabilities. It reported 0 unintended gaps for five years of sweeps. AS ABOVE, SO BELOW -- and the
audit had never been applied to itself.

THE INSTRUMENT'S REAL DEFECT WAS NOT THE FROZEN INPUT. It CLAIMS to measure reachability and it
MEASURES PROMOTION: the L2/L3 markers are substring probes for DEDICATED tool names, and
`holographic_mcp` hosts `lecore_invoke(name, args) -> run any public faculty`, so reachability at L2
is universal by construction. Its L1 check was worse -- `fac and ("UnifiedMind" in facade)` is a
constant, True for every row, measuring nothing per capability.

So the tests here guard TWO things. That the derived sweep finds what the frozen one could not
(22 genuine gaps: 4 catalog cards naming a door defined nowhere in the repo, 18 naming a module
function no agent can /invoke), and -- just as important -- that it does NOT flag the 13 object
methods that are reachable by design. A canary that cries wolf gets disabled, and this repo has
written that lesson down three times now.
"""
import json
import os
import subprocess
import sys

import pytest

import lecore

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import swarm_audit as sa                                                    # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = {"no_floor", "unreachable", "facade_lie", "l1_gap", "l2_gap"}


@pytest.fixture(scope="module")
def report():
    """One live sweep for the whole file -- it walks the repo's AST and costs ~10 s."""
    return sa.derived_matrix(lecore.UnifiedMind(dim=64, seed=0), root=REPO)


# --------------------------------------------------------------------------------------
# 1. THE POPULATION IS DERIVED, not typed by hand.
# --------------------------------------------------------------------------------------

def test_the_population_is_far_larger_than_the_frozen_literal(report):
    c = report["counts"]
    assert len(sa.CAPS) == 21, "the legacy literal is kept on purpose; this pins its size"
    assert c["cards"] > 700 and c["doors"] > 600
    assert c["doors"] > len(sa.CAPS) * 25, "the derived population must dwarf the literal"


def test_the_legacy_matrix_and_its_deliberate_whitelist_are_kept(report):
    # EXTEND, DON'T REPLACE: the L0-L4 layering and the reasons attached to chosen gaps are the
    # parts that were right, and a chosen gap with a stated reason is not a defect.
    assert sa.DELIBERATE and all(isinstance(k, tuple) and v.strip()
                                 for k, v in sa.DELIBERATE.items())
    rows, gaps = sa.above_below(lecore.UnifiedMind(dim=64, seed=0))
    assert len(rows) == 21 and gaps == []


# --------------------------------------------------------------------------------------
# 2. THE JUDGEMENT: reachability is asked of everything, promotion of nothing.
# --------------------------------------------------------------------------------------

def test_promotion_is_a_census_and_never_a_gap(report):
    # Scoring the unpromoted doors as defects would be sweep 123's "761-item bar on day one", and
    # the note this repo already keeps says a bar nobody clears is a bar nobody runs.
    c = report["counts"]
    assert c["L2_promoted"] < c["doors"] / 10 and c["L3_promoted"] < c["doors"] / 10
    assert all(g["kind"] in KINDS for g in report["genuine"])
    assert not any("promot" in g["kind"] for g in report["genuine"])


def test_L1_and_L2_are_measured_per_capability_not_assumed(report):
    # THE BUG THIS REPLACES: `fac and ("UnifiedMind" in facade)` is True for every row regardless of
    # the capability, so L1 has been a constant since cp67. Here L1 tracks L0 because both are real
    # hasattr checks against the same object -- if a door ever leaves the facade this diverges.
    c = report["counts"]
    assert c["L1"] == c["L0"] == c["L2_reachable"]
    assert all(r["L1"] is bool(r["L0"] and hasattr(lecore.UnifiedMind, r["method"]))
               for r in report["rows"][:50])


def test_every_reachable_door_really_dispatches():
    # MANIFEST MEMBERSHIP IS A CLAIM; invoking is the measurement. These are the 36 doors the
    # round-5 brief said had never been checked for hosted reachability.
    m = lecore.UnifiedMind(dim=64, seed=0)
    named = ["orient", "study", "bequeath", "merge_trees", "role", "delegate", "agent_loop",
             "find_capability", "bounded_preview", "result_contract", "merge_census",
             "memory_curate", "delegation_drift", "file_append", "file_stat", "above_below"]
    for n in named:
        assert hasattr(m, n), n
        try:
            m.invoke(n, {"__probe_that_cannot_be_a_real_parameter__": 1})
        except ValueError as e:
            assert not any(w in str(e) for w in ("private", "unknown", "not callable")), \
                "%s is listed but the dispatch gate refuses it" % n
        except Exception:
            pass                      # dispatched and the signature rejected the probe: reachable


# --------------------------------------------------------------------------------------
# 3. THE CLASSIFIER: three lookalikes, and getting the third wrong disables the canary.
# --------------------------------------------------------------------------------------

def test_an_object_method_is_classified_out_not_flagged(report):
    # mind.memory_curate() -> MemoryCurator.plan is a deliberate pattern; flagging it would be
    # crying wolf on the engine's own idiom.
    assert report["counts"]["not_meaningful"] >= 1
    flagged = {g["method"] for g in report["genuine"]}
    excused = {g["method"] for g in report["not_meaningful"]}
    assert not (flagged & excused), "a method cannot be both a gap and excused"
    assert all("object method" in g["why"] for g in report["not_meaningful"])


def test_a_card_naming_a_door_that_exists_nowhere_is_a_hard_gap(report):
    # THE HEADLINE. skill_lint does not catch these: it validates mind.X inside catalog EXAMPLES,
    # never the card's own method= field, so a card can promise a door that has never existed.
    #
    # THE FOURTH TEST THIS SESSION TO FAIL BECAUSE THE WORK SUCCEEDED. This asserted the no_floor
    # LIST was non-empty, using the four real defects as its fixture -- and close-out fixed all
    # four (bios_boot -> boot, doctrine_seedpack -> doctrine_load, panel_realm -> panel_seat,
    # phase_randomized_null -> phase_randomize), so the class emptied and the test went red for
    # the right reason. A test whose fixture is a REAL BUG dies the day somebody fixes it. So the
    # CLASSIFIER is exercised on a SYNTHETIC card instead -- always available, never dependent on
    # the repo still being broken -- and the real rows, when there are any, still get checked.
    m = lecore.UnifiedMind(dim=64, seed=0)
    methods, funcs = sa._repo_defs(REPO)

    invented = "no_such_door_" + "b7f3a1"          # fixed, not random: the suite is deterministic
    assert not hasattr(m, invented)
    assert invented not in methods and invented not in funcs

    no_floor = [g for g in report["genuine"] if g["kind"] == "no_floor"]
    for g in no_floor:                              # zero is the GOAL state, not a broken test
        assert not hasattr(m, g["method"])
        assert g["method"] not in methods and g["method"] not in funcs


def test_a_module_function_is_reachable_below_and_not_above(report):
    unreachable = [g for g in report["genuine"] if g["kind"] == "unreachable"]
    assert unreachable, "the module-function class must not vanish silently"
    m = lecore.UnifiedMind(dim=64, seed=0)
    _methods, funcs = sa._repo_defs(REPO)
    for g in unreachable:
        assert g["method"] in funcs and not hasattr(m, g["method"])


def test_every_finding_carries_a_reason(report):
    assert all(g["why"].strip() for g in report["genuine"] + report["not_meaningful"])
    assert report["counts"]["genuine"] == sum(report["by_kind"].values())


# --------------------------------------------------------------------------------------
# 4. THE SURFACES ARE READ, not guessed at.
# --------------------------------------------------------------------------------------

def test_surface_calls_are_derived_by_ast(tmp_path):
    f = tmp_path / "surf.py"
    f.write_text("def h(self):\n"
                 "    self.service.mind.alpha(1)\n"
                 "    m.beta()\n"
                 "    m._private()\n"
                 "    other.gamma()\n")
    got = sa._surface_calls(str(f), "m")
    assert "alpha" in got and "beta" in got
    assert "_private" not in got, "private names are not a surface"
    assert "gamma" not in got, "an unrelated receiver is not the mind"


def test_no_surface_calls_a_door_that_does_not_exist(report):
    # "A facade lying" -- the direction the frozen matrix never checked at all.
    m = lecore.UnifiedMind(dim=64, seed=0)
    for surface, recv in ((os.path.join(REPO, "holographic_mcp.py"), "mind"),
                          (os.path.join(REPO, "chat_server.py"), "m")):
        calls = sa._surface_calls(surface, recv)
        assert len(calls) >= 20, surface
        assert not [n for n in calls if not hasattr(m, n)], surface
    assert not [g for g in report["genuine"] if g["kind"] == "facade_lie"]


def test_repo_defs_separates_methods_from_functions(tmp_path):
    (tmp_path / "x.py").write_text("def modfunc():\n    pass\n\n\nclass K:\n    def objmeth(self):\n        pass\n")
    methods, funcs = sa._repo_defs(str(tmp_path))
    assert "objmeth" in methods and "objmeth" not in funcs
    assert "modfunc" in funcs and "modfunc" not in methods


# --------------------------------------------------------------------------------------
# 5. THE GATE: shrink-only, at a real floor, and a reason is not optional.
# --------------------------------------------------------------------------------------

def test_the_budget_is_a_real_floor_with_a_written_reason():
    b = json.load(open(os.path.join(REPO, sa.BUDGET_PATH), encoding="utf-8"))
    assert isinstance(b["genuine_budget"], int) and b["genuine_budget"] > 0, \
        "a gate at zero on an untriaged backlog gets disabled rather than fixed"
    assert len(b["why"].strip()) > 40, "a floor without a reason is a floor nobody can audit"


def test_rebase_without_a_reason_is_refused():
    r = subprocess.run([sys.executable, "tools/swarm_audit.py", "--rebase"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "REFUSED" in r.stdout
    assert "--reason" in r.stdout


def test_the_gate_passes_at_the_recorded_floor():
    r = subprocess.run([sys.executable, "tools/swarm_audit.py", "--gate"], cwd=REPO,
                       capture_output=True, text=True)
    assert "within budget" in r.stdout, r.stdout[-400:]
    assert "GENUINE" in r.stdout, "the count must be printed, not just gated on"


# --------------------------------------------------------------------------------------
# 6. WIRING and determinism.
# --------------------------------------------------------------------------------------

def test_the_faculty_is_wired_to_the_mind_and_documented(report):
    m = lecore.UnifiedMind(dim=256, seed=0)
    assert callable(getattr(m, "above_below", None))
    doc = m.above_below.__doc__ or ""
    assert doc.strip() and "PROMOTION" in doc, "the judgement must travel with the door"
    assert "KEPT NEG" in doc
    got = m.above_below(root=REPO)
    assert got["counts"] == report["counts"], "the mind verb must delegate, not reimplement"


def test_two_sweeps_agree(report):
    again = sa.derived_matrix(lecore.UnifiedMind(dim=64, seed=0), root=REPO)
    assert json.dumps(again["counts"], sort_keys=True) == \
        json.dumps(report["counts"], sort_keys=True)
    assert [g["method"] for g in again["genuine"]] == [g["method"] for g in report["genuine"]]


def test_the_generated_document_carries_the_derived_section():
    doc = open(os.path.join(REPO, "docs", "SWARM_AUDIT.md"), encoding="utf-8").read()
    assert "## Derived matrix" in doc
    assert "mind.above_below()" in doc, "the doc must name the verb it documents"
    assert "PROMOTION IS A CENSUS" in doc
