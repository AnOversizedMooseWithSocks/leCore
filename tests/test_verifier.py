"""tests/test_verifier.py -- E5.2 verifier prototypes (CLM backlog, wave 2).

swarm_step's verified results now train per-tool SUCCESS / FAILURE prototypes (a two-option ProtoStore
'verify:<tool>'): a PASSED verify is a success example of its state and -- the change -- a FAILED verify is a labelled
FAILURE example (before, a refused step taught nothing). verify_precheck(state, actions) orders (or, with min_p,
pre-filters) candidate actions by them. THE RULE PINNED HERE ABOVE ALL: the pre-check NEVER replaces the verify --
swarm_step runs the verify command on every step that carries one, whatever the prototypes predict.
Hand-written data; the measured AUROC against the reflex is tools/bench_verifier.py.
"""
import os

import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")

TRUTH = {"smooth a bumpy mesh surface": "mesh_smooth", "compress a noisy float series": "series_compress",
         "grow crystals on a surface": "crystal_grow", "find the shortest path in a maze": "maze_solve"}
TOOLS = ["mesh_smooth", "series_compress", "crystal_grow", "maze_solve"]


def _mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    m.judge_calls = 0

    def judge(task, tool):
        m.judge_calls += 1
        return TRUTH.get(task) == tool
    m.judge = judge
    return m


def _step(m, task, tool, **kw):
    return m.swarm_step(task, tool, done_when="judge passes", evidence={"tool": tool}, worker="w",
                        verify={"verb": "judge", "args": {"task": task, "tool": tool}}, expect=True, **kw)


def _try_all(m, task, order):
    """What an agent does: submit candidates in order until the judge passes one. Returns judge calls used."""
    before = m.judge_calls
    for tool in order:
        try:
            _step(m, task, tool)
            break
        except ValueError:
            continue
    return m.judge_calls - before


def test_a_failed_verify_is_refused_and_becomes_a_labelled_failure_example():
    m = _mind()
    task = "smooth a bumpy mesh surface"
    with pytest.raises(ValueError, match="refused"):
        _step(m, task, "maze_solve")
    st = m.protostore("verify:maze_solve")
    assert st.labels == ["success", "failure"] and st.count[st.index("failure")] == 1
    pre = m.verify_precheck(task, ["maze_solve", "mesh_smooth"])
    assert pre["scores"]["maze_solve"] < 0.0 and pre["scores"]["mesh_smooth"] == 0.0     # unseen tool: neutral
    assert pre["order"] == ["mesh_smooth", "maze_solve"] and pre["verify_still_runs"] is True
    assert m.decision_records()["stats"]["records"] == 0                                 # a refused step: no record


def test_the_precheck_orders_the_known_good_tool_first_for_a_reworded_task():
    m = _mind()
    for task in TRUTH:
        _try_all(m, task, TOOLS)                                   # learn: failures AND the passing tool
    reworded = "smooth this bumpy mesh surface please"
    pre = m.verify_precheck(reworded, TOOLS)
    assert pre["order"][0] == "mesh_smooth"
    assert _try_all(m, reworded if reworded in TRUTH else "smooth a bumpy mesh surface", pre["order"]) == 1


def test_the_precheck_never_skips_a_real_verify():
    """Train the verifier until it is SURE 'mesh_smooth' succeeds on a task, then make the judge disagree: the verify
    still runs (the judge is called) and the step is REFUSED. Nothing in swarm_step consults the pre-check to accept."""
    m = _mind()
    task = "smooth a bumpy mesh surface"
    for _ in range(5):
        _step(m, task, "mesh_smooth")
    pre = m.verify_precheck(task, ["mesh_smooth"])
    assert pre["scores"]["mesh_smooth"] > 0.5
    TRUTH_BACKUP = TRUTH[task]
    try:
        TRUTH[task] = "something_else"                             # the world changed: the tool now fails
        calls = m.judge_calls
        with pytest.raises(ValueError, match="refused"):
            _step(m, task, "mesh_smooth")
        assert m.judge_calls == calls + 1                          # the verify RAN despite the confident pre-check
    finally:
        TRUTH[task] = TRUTH_BACKUP
    # the refusal taught a failure example: the success margin shrank
    assert m.verify_precheck(task, ["mesh_smooth"])["scores"]["mesh_smooth"] < pre["scores"]["mesh_smooth"]
    # an accepted step still records what the pre-check predicted, next to the verify that decided
    rec = _step(m, task, "mesh_smooth")
    assert rec["meta"]["verified"]["passed"] is True and "score" in rec["meta"]["precheck"]


def test_the_verify_door_calibrates_prequentially_and_min_p_never_empties_the_list():
    m = _mind()
    for _ in range(3):
        for task in TRUTH:
            _try_all(m, task, TOOLS)
    cal = m.door_calibration_report()["verify"]
    assert cal["labels"] >= 8 and 0 < cal["correct"] < cal["labels"]
    pre = m.verify_precheck("smooth a bumpy mesh surface", TOOLS, min_p=0.5)
    assert pre["calibrated"] is True and pre["order"] and pre["order"][0] == "mesh_smooth"
    assert set(pre["order"]) | set(pre["dropped"]) == set(TOOLS)
    everything = m.verify_precheck("smooth a bumpy mesh surface", TOOLS, min_p=1.01)
    assert everything["order"] == ["mesh_smooth"]                  # never empties: the best one stays


def test_a_state_carrying_a_secret_never_teaches_the_verifier():
    m = _mind()
    task = "use password: Hunter2-FAKE-9c1d to smooth a bumpy mesh surface"
    TRUTH[task] = "mesh_smooth"
    try:
        _step(m, task, "mesh_smooth")
    finally:
        del TRUTH[task]
    assert "verify:mesh_smooth" not in m.__dict__.get("_protostores", {})


def test_the_verifier_survives_a_restart(tmp_path):
    m = _mind()
    for task in TRUTH:
        _try_all(m, task, TOOLS)
    before = m.verify_precheck("smooth this bumpy mesh surface", TOOLS)
    root = str(tmp_path / "partition")
    m.learning_save(root)
    m2 = _mind()
    m2.learning_load(root)
    after = m2.verify_precheck("smooth this bumpy mesh surface", TOOLS)
    assert after["order"] == before["order"] and after["scores"] == before["scores"]
