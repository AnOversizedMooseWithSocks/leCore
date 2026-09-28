"""tests/test_distill_release.py -- the shipped memory bundle is a DISTILLATE, never a copy of session memory.

Found 2026-09-27: release_bundle/ had grown from the 62 distilled rows of 2026-08-23 to 497 rows copied from session
memory (session-salted probes, build history, deployment notes); by the distiller's own rules only 75 qualified.
These pins keep the rule honest and the shipped bundle passing it.
2026-09-27 (the core-memory extension): the bundle also carries confirmed wordings, methods, api specs, tool reflexes,
shareable ProtoStores and SOPs under their own rules -- tests/test_core_memory.py pins those; here, that the SHIPPED
report is audited (every artifact class and every exclusion) and that the release rule is the strict one."""
import os

import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dr():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import distill_release
    return distill_release


class _Ladder:
    """The one attribute the rule reads: taught_log rows [q, a, scope, provenance], plus the veto set."""

    def __init__(self, rows, vetoed=()):
        self.taught_log = [list(r) for r in rows]
        self._vetoed_qs = set(vetoed)


GOOD = "the reflex answers near-exact repeats at T0 and a lecore mind escalates what it cannot answer honestly"


def test_the_rule_keeps_generic_engine_knowledge_and_drops_everything_personal():
    d = _dr()
    lad = _Ladder([
        ("how does the reflex answer repeats", "OLD stale answer about the reflex in lecore, long enough to ship", "shared", "taught"),
        ("how does the reflex answer repeats", GOOD, "shared", "taught"),                    # newest wins
        ("[s:full-stack] what is the calibration constant", GOOD, "shared", "taught"),       # session-salted
        ("what is the calibration constant of sensor array 3", GOOD, "shared", "taught"),    # vetoed probe
        ("what did cp110 find", GOOD, "shared", "taught"),                                   # build history
        ("what did the audit find in the reflex", GOOD, "shared", "taught"),                 # history question
        ("how is the rollover done", GOOD + " [doctrine cp15]", "shared", "taught"),         # doctrine ships itself
        ("how do I paint a cel", "gouache on the back of the cel, lines on the front; recall the colour key",
         "shared", "taught"),                                                                # no engine anchor
        ("what is the reflex budget", GOOD, "weight-install", "taught"),                     # a work scope
        ("what did the model say", GOOD, "shared", "model-cached"),                          # provisional
        ("my password is Hunter2-FAKE-9c1d what now", GOOD, "shared", "taught"),             # FAKE secret
    ], vetoed={"what is the calibration constant of sensor array 3"})
    kept, excluded, n = d._select(lad)
    assert [q for q, _, _ in kept] == ["how does the reflex answer repeats"]
    assert kept[0][1] == GOOD, "the NEWEST answer must ship (replay is last-wins)"
    for reason in ("session-specific", "vetoed", "build-history", "history", "doctrine (the seedpack ships it)",
                   "work-scope", "provisional-provenance", "sensitive"):
        assert excluded.get(reason), (reason, excluded)
    assert excluded.get("no-engine-anchor", 0) + excluded.get("no-engine-content", 0) + \
        excluded.get("task-artifact", 0) >= 1


def test_the_shipped_bundle_passes_its_own_rule():
    """A bundle written by anything but the distiller fails here (the 2026-08-26 bundle: 75 of 497 passing)."""
    d = _dr()
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        rep = d.check("release_bundle")
    finally:
        os.chdir(cwd)
    assert rep["ok"], rep
    assert 50 <= rep["rows"] <= 400, rep


def test_the_seedpack_and_the_bundle_do_not_overlap():
    """Doctrine ships in the seedpack; a bundle copy of an older doctrine text would shadow the newer one."""
    import json
    from holographic.agents_and_reasoning.holographic_seedpack import DOCTRINE
    with open(os.path.join(ROOT, "release_bundle", "distill_report.json"), encoding="utf-8") as f:
        shipped = set(json.load(f)["questions"])
    assert not shipped & {q for q, _, _ in DOCTRINE}


def test_the_shipped_report_audits_every_artifact_class_under_the_strict_rule():
    """The release bundle is built by the EXTENDED distiller with the release rule (engine anchor on, no private
    hosts, no calibration carry): its report lists every artifact class (carried + excluded by reason) and what is
    deliberately not distilled."""
    import json
    with open(os.path.join(ROOT, "release_bundle", "distill_report.json"), encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["rule"] == {"anchor": True, "rows_only": False, "allow_private_hosts": False,
                           "carry_calibration": False}, rep["rule"]
    for cls in ("wordings", "methods", "negatives", "direction", "apis", "api_cards", "reflexes", "protostores",
                "sops", "workflows"):
        assert cls in rep["artifacts"] and {"carried", "excluded", "items"} <= set(rep["artifacts"][cls]), cls
    assert "reflex bridge experience" in rep["not_distilled"]


def test_the_wheel_carries_the_same_core_memory_as_the_repo():
    """The core memory rides in the wheel as lecore_data/release_bundle (owner-approved 2026-09-27); it must be the
    repo bundle byte for byte, pass the same audit, and be listed in setup.py's package_data."""
    import filecmp
    a, b = os.path.join(ROOT, "release_bundle"), os.path.join(ROOT, "lecore_data", "release_bundle")
    for rel in ("distill_report.json", "knowledge.lecore", os.path.join("learning", "state.lecore")):
        assert filecmp.cmp(os.path.join(a, rel), os.path.join(b, rel), shallow=False), rel
    src = open(os.path.join(ROOT, "setup.py"), encoding="utf-8").read()
    assert '"release_bundle/learning/*.lecore"' in src


def test_a_pip_style_install_copies_the_core_memory_into_a_user_partition(tmp_path, monkeypatch):
    """No repo next to the package: the wheel's core memory is COPIED once into $LECORE_HOME/memory and that copy is
    what boots -- never the package directory. A second call leaves the (now learning) copy alone."""
    import warnings
    import lecore
    monkeypatch.setenv("LECORE_HOME", str(tmp_path / "home"))
    home = lecore._core_memory_home()
    assert home == str(tmp_path / "home" / "memory")
    assert os.path.exists(os.path.join(home, "learning", "state.lecore"))
    marker = os.path.join(home, "mine.txt")
    open(marker, "w").write("user data")
    assert lecore._core_memory_home() == home and os.path.exists(marker)      # not re-copied over the user's copy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lecore.autoboot(partition=home, llm=None)
    assert m.ask("how do I read the confidence of a lecore answer")["tier"] == "T0"


def test_a_learned_wording_of_a_doctrine_row_reaches_the_core_memory(tmp_path):
    """Maple swarm run (2026-09-27): a `same` verdict linked "how should a swarm share memory" to the seedpack's swarm
    doctrine row, and the distillation dropped it with the row (the seedpack ships the row) -- so every fresh install
    escalated that wording again. The LINK now ships; the doctrine TEXT still does not (the seedpack teaches it)."""
    import warnings
    import lecore
    from holographic.agents_and_reasoning.holographic_seedpack import DOCTRINE
    dr = _dr()
    swarm_q = next(q for q, *_ in DOCTRINE if q.startswith("how should a swarm of agents share"))
    src = lecore.UnifiedMind(dim=256, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        src.boot(partition=str(tmp_path / "src"), doctrine=True, llm=lambda p: "")
    rid = next(r for r, row in src.meaning.rows.items() if row["canonical"] == swarm_q)
    wording = "how should a swarm share memory"
    assert src.meaning_resolve(wording, {"verdict": "same", "row": rid}, by="w1").get("learned")
    out = str(tmp_path / "bundle")
    rep = dr.distill(None, out, src=src)
    assert rep["artifacts"]["wordings"]["carried"] >= 1
    audit = dr.check(out)
    assert audit["ok"], audit["failing"]                            # the audit accepts a doctrine row's wording
    # the bundle carries no doctrine TEXT ...
    from holographic.io_and_interop.holographic_container import load_container
    blob = open(os.path.join(out, "learning", "state.lecore"), "rb").read()
    taught = [s for s in load_container(blob)["sections"] if s["kind"] == "lecore.learning.taught"][0]["meta"]
    assert not any("[doctrine" in str(p[1]) for p in taught.get("pairs", []))
    # ... and a fresh install booted on it serves the learned wording from memory, with the seedpack's text
    home = str(tmp_path / "home")
    import shutil
    shutil.copytree(out, home)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lecore.autoboot(partition=home, llm=None)
    got = m.serve(wording)
    assert got["served"] is True
    assert got["answer"].startswith(next(a for q, a, *_ in DOCTRINE if q == swarm_q))   # the seedpack text (+ its tag)
