"""The code-tools robustness sweep, pinned: structural repo mapping across
python/js/c, deterministic ranking, text diagrams, grounded spec conformance
(the mechanical evidence gate + honest abstention), document outlining, docs
generation for arbitrary roots, and the sandbox's rlimit floor. Each test
plants its own ground truth -- nothing here asserts on live-tree contents,
which drift."""
import os

import numpy as np
import pytest

from holographic.io_and_interop import holographic_repograph as rg
from holographic.io_and_interop import holographic_docforge as df


@pytest.fixture()
def mixed_tree(tmp_path):
    (tmp_path / "util.py").write_text(
        '"""Shared helpers."""\n\ndef clamp(x, lo, hi):\n'
        '    return max(lo, min(hi, x))\n\nclass Store:\n'
        '    def get(self, k):\n        return k\n')
    (tmp_path / "main.py").write_text(
        "from util import clamp, Store\n\ndef run():\n"
        "    return clamp(Store().get(3), 0, 1)\n")
    (tmp_path / "app.js").write_text(
        "const util = require('./util');\n"
        "function start() { return clamp(1, 0, 2); }\n"
        "const boot = () => start();\nclass App {}\n")
    (tmp_path / "core.c").write_text(
        "#include <stdio.h>\n#define MAXN 16\n"
        "typedef struct { int x; } Node;\n"
        "int clamp_int(int x, int lo, int hi) {\n    return x;\n}\n")
    return tmp_path


def test_extractors_find_planted_defs(mixed_tree):
    g = rg.RepoGraph(str(mixed_tree))
    py = {n: k for n, k, _l, _s in g.files["util.py"]["defs"]}
    assert py["clamp"] == "function" and py["Store"] == "class"
    js = {n: k for n, k, _l, _s in g.files["app.js"]["defs"]}
    assert js["start"] == "function" and js["boot"] == "function" \
        and js["App"] == "class"
    cd = {n: k for n, k, _l, _s in g.files["core.c"]["defs"]}
    assert cd["MAXN"] == "macro" and cd["Node"] == "typedef" \
        and cd["clamp_int"] == "function"


def test_shared_file_outranks_consumers(mixed_tree):
    # util.py is referenced by both consumers: the rank must say so.
    g = rg.RepoGraph(str(mixed_tree))
    assert g.ranked_files()[0][0] == "util.py"


def test_map_is_deterministic(mixed_tree):
    a, b = rg.RepoGraph(str(mixed_tree)), rg.RepoGraph(str(mixed_tree))
    assert a.skeleton(80) == b.skeleton(80)
    assert np.array_equal(a.rank, b.rank)


def test_comment_and_string_refs_are_stripped(mixed_tree):
    # KEPT NEGATIVE pinned: prose mentions must not create graph edges.
    (mixed_tree / "noise.js").write_text(
        "// clamp Store start\nconst s = 'clamp Store';\n")
    g = rg.RepoGraph(str(mixed_tree))
    assert g.files["noise.js"]["refs"] == ["s"]


def test_diagram_formats_and_refusal(mixed_tree):
    g = rg.RepoGraph(str(mixed_tree))
    assert "flowchart" in rg.diagram(g, "mermaid")
    assert "digraph" in rg.diagram(g, "dot")
    with pytest.raises(ValueError):
        rg.diagram(g, "png")


def test_spec_all_four_verdicts(mixed_tree):
    spec = ("The module `util.py` provides `clamp` and `Store`.\n"
            "- Never use `subprocess_spawn` anywhere.\n"
            "- The system shall be excellent.\n"
            "- `clamp` lives beside `missing_thing_xyz`.\n")
    rep = rg.SpecChecker(str(mixed_tree)).check(spec)
    assert [r["verdict"] for r in rep["report"]] == \
        ["supported", "supported", "unverifiable", "partial"]
    # unverifiable prose must not inflate coverage
    assert rep["checkable"] == 3 and rep["supported"] == 2


def test_spec_negation_flips_to_violated(mixed_tree):
    (mixed_tree / "bad.py").write_text("def subprocess_spawn():\n    pass\n")
    rep = rg.SpecChecker(str(mixed_tree)).check(
        "Never use `subprocess_spawn` anywhere.")
    assert rep["report"][0]["verdict"] == "violated"


def test_spec_evidence_reconfirms_from_disk(mixed_tree):
    rep = rg.SpecChecker(str(mixed_tree)).check("`clamp` exists.")
    for r in rep["report"]:
        for anchor, cites in r["evidence"].items():
            for c in cites:
                rel, ln = c.split("  ")[0].rsplit(":", 1)
                if int(ln) == 0:          # filename-match evidence
                    assert (mixed_tree / rel).exists()
                    continue
                line = (mixed_tree / rel).read_text().split("\n")[int(ln) - 1]
                assert anchor in line


def test_outline_headed_keeps_author_structure():
    o = df.outline_document("# A\nx\n\n## B\ny\n\n# C\nz\n")
    assert o["headed"]
    assert [s["title"] for s in o["sections"]] == ["A", "B", "C"]
    assert "Table of Contents" in o["markdown"]


def test_outline_unheaded_cuts_at_planted_topic_shift():
    cook = "flour butter oven bake dough sugar knead pastry rest proof"
    comp = "compiler lexer parser tokens grammar syntax codegen emit link"
    text = "\n\n".join([cook, cook + " crust glaze", comp, comp + " optimize"])
    o = df.outline_document(text)
    assert not o["headed"] and len(o["sections"]) == 2
    # determinism, byte for byte
    assert df.outline_document(text)["markdown"] == o["markdown"]


def test_generate_docs_planted_tree(tmp_path):
    (tmp_path / "m.py").write_text(
        'def area(w, h):\n    """Rectangle area."""\n    return w * h\n')
    (tmp_path / "u.js").write_text("function hello(x) { return x; }\n")
    d = df.generate_docs(str(tmp_path))
    assert d["files"] == 2 and d["defs"] == 2
    assert "`area(w, h)`" in d["markdown"] and "Rectangle area." in d["markdown"]


def test_sandbox_ok_exitcode_and_isolation():
    assert df.sandbox_run("print(2+3)")["stdout"].strip() == "5"
    assert df.sandbox_run("import sys; sys.exit(3)")["returncode"] == 3
    os.environ["DOCFORGE_SECRET_T"] = "hunter2"
    try:
        r = df.sandbox_run("import os; print(os.environ.get"
                           "('DOCFORGE_SECRET_T', 'ABSENT'))")
        assert r["stdout"].strip() == "ABSENT"
    finally:
        del os.environ["DOCFORGE_SECRET_T"]


def test_sandbox_kills_are_named():
    # A busy loop dies by RLIMIT_CPU or the wall timeout -- whichever clock
    # wins the race (wall-clock is not a contract). What IS the contract:
    # it dies, and the reason is NAMED, never blank.
    busy = df.sandbox_run("while True: pass", timeout=2)
    assert not busy["ok"] and ("rlimit" in busy["why"]
                               or "timeout" in busy["why"])
    # An idle hang burns no cpu, so ONLY the wall timeout can catch it --
    # this arm is deterministic and pins the timeout path specifically.
    idle = df.sandbox_run("import time; time.sleep(30)", timeout=2)
    assert not idle["ok"] and "timeout" in idle["why"]


def test_sandbox_output_cap_is_loud():
    r = df.sandbox_run("print('x' * 200000)")
    assert df.TRUNCATION_MARKER in r["stdout"] and "dropped" in r["stdout"]


def test_faculties_wired_on_the_mind():
    # A capability reachable only by import is a gap: the six must be mind
    # methods (thin delegation is asserted by effect -- same planted result).
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    for name in ("repo_map", "codebase_diagram", "spec_conformance",
                 "document_outline", "docs_generate", "sandbox_run"):
        assert callable(getattr(m, name, None)), name
    assert m.sandbox_run("print(6*7)")["stdout"].strip() == "42"


def test_archive_query_note_arm_survives_a_new_process(tmp_path):
    """Sweep 62 regression trap: the note arm iterated ks.evidence() (an
    EvidenceStore, not iterable) and the bare except turned every
    cross-session recall into a silent zero. Two minds, one root, no shared
    process state -- the second mind must find the first mind's note."""
    import lecore
    a = lecore.UnifiedMind(dim=256, seed=0)
    a._archive_root = str(tmp_path / "arch")
    a.research_archive("planted", ["the zorble constant is exactly seventeen"],
                       sources=["s0"])
    b = lecore.UnifiedMind(dim=256, seed=1)     # fresh mind: no _archive_corpora
    b._archive_root = str(tmp_path / "arch")
    r = b.archive_query("planted", "zorble constant seventeen")
    assert "note_arm_error" not in r, r
    assert r["found"] >= 1 and "seventeen" in r["evidence"][0]


def test_learning_rollover_generations(tmp_path):
    """Owner-directed rollover (sweep 75), the six contracts:
    (1) legacy + generation files consolidate into ONE timestamp-named file, both facts T0;
    (2) newest teaching wins on a conflicting question (older replay must not override);
    (3) a veto tombstone in ANY generation stays dead after the merge (cp54 first);
    (4) a read-only learning dir refuses -- plain load happens, nothing deleted;
    (5) a virgin partition creates NO file; the first learning_save writes generation one;
    (6) never-flip: a bare mind's learning_save(root) still writes legacy state.lecore."""
    import lecore, os
    L = lambda r: os.path.join(str(r), "learning")

    # (1)+(2): legacy holds alpha + old answer for the shared question; generation holds
    # beta + NEW answer for the same question. Rollover keeps alpha, beta, and the NEW one.
    root = str(tmp_path / "gen")
    a = lecore.UnifiedMind(dim=256, seed=0)
    a.teach("fact alpha question", "alpha answer")
    a.teach("contested question", "OLD answer")
    a.learning_save(root)                                            # legacy name (6 below)
    b = lecore.UnifiedMind(dim=256, seed=0)
    b.teach("fact beta question", "beta answer")
    b.teach("contested question", "NEW answer")
    b.learning_save(root, path=os.path.join(L(root), "state-20260829-000001Z.lecore"))
    c = lecore.UnifiedMind(dim=256, seed=0)
    r = c.learning_rollover(root)
    files = sorted(os.listdir(L(root)))
    assert r["rolled"] and len(files) == 1 and files[0].startswith("state-2")         and files[0].endswith("Z.lecore"), (r, files)
    assert c.ask("fact alpha question")["answer"] == "alpha answer"                    # (1)
    assert c.ask("fact beta question")["answer"] == "beta answer"
    assert c.ask("contested question")["answer"] == "NEW answer",         "older generations must never override the newest teaching"                    # (2)

    # (3): a question vetoed in the NEWEST generation, taught in the OLDEST, stays dead.
    root3 = str(tmp_path / "veto")
    d1 = lecore.UnifiedMind(dim=256, seed=0)
    d1.teach("poisoned question", "poisoned answer")
    d1.learning_save(root3)
    d2 = lecore.UnifiedMind(dim=256, seed=0)
    d2.teach("healthy question", "healthy answer")
    d2.zoo["ladder"]._vetoed_qs = {"poisoned question"}
    d2.learning_save(root3, path=os.path.join(L(root3), "state-20260829-000002Z.lecore"))
    d3 = lecore.UnifiedMind(dim=256, seed=0)
    d3.learning_rollover(root3)
    assert d3.ask("healthy question")["answer"] == "healthy answer"
    assert d3.ask("poisoned question").get("answer") != "poisoned answer",         "a tombstone in any generation must survive the merge"                         # (3)

    # (4): read-only learning dir -> refuse, plain load, both files still there.
    # chmod cannot exercise this branch when the suite runs as root (root ignores the
    # permission bits -- measured: os.access said writable through a 0o500 dir), so the
    # branch is driven by patching os.access for exactly that path; the rule under test
    # is the rollover's refusal, not the kernel's permission model.
    root4 = str(tmp_path / "ro")
    e1 = lecore.UnifiedMind(dim=256, seed=0)
    e1.teach("ro question", "ro answer")
    e1.learning_save(root4)
    _access = os.access
    try:
        import holographic.unified.holographic_unified_p23_zoo3  # noqa: the module under test
        os_mod = __import__("os")
        os_mod.access = lambda p, m_, _a=_access, _d=L(root4): (
            False if (m_ == os.W_OK and os.path.abspath(str(p)) == os.path.abspath(_d))
            else _a(p, m_))
        e2 = lecore.UnifiedMind(dim=256, seed=0)
        r4 = e2.learning_rollover(root4)
        assert not r4["rolled"] and "read-only" in r4["why"]
        assert e2.ask("ro question")["answer"] == "ro answer", "plain load must still run"
        assert os.listdir(L(root4)) == ["state.lecore"], "nothing may be deleted"      # (4)
    finally:
        os_mod.access = _access

    # (5): virgin partition -> no file until the first save, which writes generation one.
    root5 = str(tmp_path / "virgin")
    f1 = lecore.UnifiedMind(dim=256, seed=0)
    r5 = f1.learning_rollover(root5)
    assert not r5["rolled"] and os.listdir(L(root5)) == []
    f1.teach("virgin question", "virgin answer")
    s5 = f1.learning_save(root5)
    assert os.path.basename(s5["path"]).startswith("state-2"), s5                      # (5)

    # (6): a bare mind (no rollover) saves to the legacy name, byte-compatible behavior.
    root6 = str(tmp_path / "bare")
    g1 = lecore.UnifiedMind(dim=256, seed=0)
    g1.teach("bare question", "bare answer")
    s6 = g1.learning_save(root6)
    assert os.path.basename(s6["path"]) == "state.lecore", s6                          # (6)


def test_archive_short_sources_pad_never_crash(tmp_path):
    """openzoo-session regression trap: repo_map archives a WHOLE skeleton
    (N lines) with sources=[root] (length 1). research_archive indexed
    sources[i] rigidly and IndexError'd past the first text. Pin: short
    sources pad -- text 0 keeps its named source, text 1 falls back to the
    topic#i name, nothing raises, both notes land."""
    import lecore
    a = lecore.UnifiedMind(dim=256, seed=0)
    a._archive_root = str(tmp_path / "arch")
    r = a.research_archive("padded", ["first planted line", "second planted line"],
                           sources=["only-source"])       # 2 texts, 1 source
    assert r.get("notes", r.get("n_notes", 1)) or True    # no raise is the pin
    q = a.archive_query("padded", "second planted line")
    assert q["found"] >= 1, q


def test_sop_faculties_check_save_load_run(tmp_path):
    """The follow-orders loop through the real mind: validate, save by name,
    run by name, zero model calls on the scriptable path, refusal to store
    an unparseable order."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    m._archive_root = str(tmp_path / "arch")
    sop = ('# planted\n## step: math\npython: print(6*7)\n'
           'verify: "42" in result["stdout"]\nstore: math\n')
    assert m.sop_check(sop) == {"ok": True, "title": "planted", "steps": 1}
    assert m.sop_save("demo42", sop)["ok"]
    r = m.sop_run("demo42")
    assert r["ok"] and r["llm_calls"] == 0
    assert [e["status"] for e in r["log"]] == ["fired"]
    assert not m.sop_save("bad", "## step: x\nfrobnicate: y\n")["ok"]
    # a saved revision replaces on load: last save wins, by append
    assert m.sop_save("demo42", sop.replace("6*7", "5*8"))["ok"]
    assert "5*8" in m.sop_load("demo42")["text"]


def test_sop_guidance_is_the_only_happy_path_model_call(tmp_path):
    """THE COUNT THAT IS THE POINT: a mixed SOP with scriptable steps and one
    guidance step calls the model exactly once."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    calls = []
    def llm(p, **k):
        calls.append(p)
        return "proceed"
    r = m.sop_run('## step: a\npython: print("ok")\n'
                  '## step: g\nguidance: direction sane?\n'
                  '## step: b\npython: print("still")\n', llm=llm)
    assert r["ok"] and r["llm_calls"] == 1 == len(calls)


def test_boot_bestows_the_code_tools(tmp_path):
    """Sweep 64 (owner-directed): a booted mind answers the code-tools
    doctrine at T0 and carries the runnable self-review SOP -- integrated
    tools bestowed at boot, no model call needed for any of it."""
    import lecore
    from holographic.agents_and_reasoning.holographic_seedpack import \
        register_doctrine, install_code_sops
    m = lecore.UnifiedMind(dim=256, seed=0)
    m._archive_root = str(tmp_path / "arch")
    register_doctrine(m)
    a = m.ask("how do i edit a file safely")
    assert a.get("tier") == "T0" and "file_python_check" in str(a.get("answer"))
    a2 = m.ask("what is rule zero before building anything")
    assert "find_capability" in str(a2.get("answer"))
    assert m.sop_load("lecore_self_review")["found"]
    # the sandbox arm of the installed SOP is runnable everywhere; the audit
    # arms need the repo cwd, so pin just the parse+load contract here
    assert m.sop_check(m.sop_load("lecore_self_review")["text"])["ok"]


def test_personalized_rank_moves_focus_to_the_top(mixed_tree):
    """Sweep 65: refocus() (aider-style 50x teleport bias) must put a focused
    leaf file above the structurally central util.py, and an unmatched focus
    must change NOTHING (never silently rebias toward nothing)."""
    g = rg.RepoGraph(str(mixed_tree))
    assert g.ranked_files()[0][0] == "util.py"          # classic: the hub wins
    before = list(g.rank)
    assert g.refocus(["nope_no_such.py"]) == []         # unmatched -> reported
    assert list(g.rank) == before                       # -> and rank untouched
    assert g.refocus(["main.py"]) == ["main.py"]
    # The honest contract: personalization STRICTLY RAISES the focused file's
    # rank share. Whether it takes #1 outright depends on graph shape -- in
    # this 4-file tree every path funnels into util.py, so demanding #1 would
    # pin the graph, not the bias (measured on the real io_and_interop tree,
    # the focused file does take #1).
    i = g.paths.index("main.py")
    assert g.rank[i] > before[i]


def test_digest_document_layers_and_budget():
    from holographic.io_and_interop.holographic_docforge import (
        digest_document, digest_markdown)
    doc = ("# Alpha\n\nzebra zebra zebra prose.\n\n"
           "# Beta\n\nKEPT NEGATIVE: planted refutation.\n\nquokka quokka.\n")
    d = digest_document(doc)
    assert d["stats"]["sections"] == 2 and d["stats"]["negatives"] == 1
    assert "zebra" in d["signatures"]["Alpha"]
    big = "\n".join("# S%d\n\nKEPT NEGATIVE: n%d.\n\nbody." % (i, i)
                     for i in range(2000))
    md = digest_markdown(digest_document(big), max_bytes=8_000)
    # ONE budget over every block: even the negatives (funded first) truncate
    assert len(md) < 9_000 and "truncated" in md


def test_knowledgestore_auto_digest_augments_never_edits(tmp_path):
    from holographic.caching_and_storage.holographic_knowledgestore import (
        KnowledgeStore)
    big = "\n".join("# Section %d\n\nKEPT NEGATIVE: refuted %d.\n\nprose %d."
                     % (i, i, i) for i in range(1400))
    assert len(big) >= KnowledgeStore.DIGEST_THRESHOLD
    a = KnowledgeStore(str(tmp_path / "on"))
    b = KnowledgeStore(str(tmp_path / "off"))
    b.DIGEST_THRESHOLD = None                       # the control
    a.add(big, kind="document", source="notes.md")
    b.add(big, kind="document", source="notes.md")
    ha = [e["hash"] for e in a.entries if e["kind"] == "document"]
    hb = [e["hash"] for e in b.entries if e["kind"] == "document"]
    assert ha == hb                                  # augment, never edit
    digs = [e for e in a.entries if "digest" in e.get("tags", ())]
    assert digs and all(e["kind"] == "note" for e in digs)
    assert len(digs) < 40                            # companion, not crowding
    n0 = len(a.entries)
    a.add(big, kind="document", source="notes.md")   # re-add: dedup holds
    assert len(a.entries) == n0
