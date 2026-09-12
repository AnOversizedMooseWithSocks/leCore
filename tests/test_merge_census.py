"""THE POST-MERGE CENSUS: did the merge lose a definition, a parameter, or a file's content?

WHY THIS FILE EXISTS. `docs/NOTES_concepts.md` states the rule in capitals across three sweeps --
after ANY merge, census DEFINITIONS, SIGNATURES and LINE COUNTS, and before restoring a shrunk
file check whether the content MOVED -- and until sweep 129 nothing enforced any of it. Sweep 120
is the reason the rule exists: a CLEAN three-way merge silently dropped ten definitions the engine
depended on, because diff3 honoured the other side's deletions cleanly. A name-level census caught
those; a signature-level census then caught TWO MORE it had passed (a lost `record_every`
passthrough). Sweep 125 added the third leg and immediately corrected its own reflex: the catalog's
missing 970 lines had MOVED into holographic_catalog_aliases.py, so restoring the file wholesale
would have reverted the other line's improvement.

THE NUMBER THAT DECIDES WHETHER ANYONE RUNS THIS TWICE IS THE FALSE-POSITIVE COUNT, and it is
asserted at 0. Run on the real sweep-122 merge, a def-only census reported two LOST DEFINITIONS
where upstream had promoted a triplicated `_f1` into `holographic_occlusion.recall_f1` and left
`from ... import recall_f1 as _f1` behind. Nothing was removed. An import alias IS a definition of
that name, and `test_an_import_alias_is_not_a_lost_definition` is the trap that keeps it so.

Every count here comes from ONE fixture with eight deliberately injected faults
(`_census_fixture`), shared with the module selftest so the two never drift apart.
"""
import ast
import json
import subprocess

import pytest

import lecore
from holographic.io_and_interop.holographic_codestructure import (
    CENSUS_FAULTS, _census_fixture, def_index, merge_census, signature_of)


@pytest.fixture(scope="module")
def damaged(tmp_path_factory):
    """The two trees with known damage, censused once -- every test below reads the same report."""
    root = tmp_path_factory.mktemp("census")
    b, n = str(root / "base"), str(root / "new")
    _census_fixture(b, n)
    return merge_census(b, n, base_is_ref=False), b, n


# --------------------------------------------------------------------------------------
# 1. THE FALSE-POSITIVE COUNT. Everything else is worth nothing if this is not 0.
# --------------------------------------------------------------------------------------

def test_an_import_alias_is_not_a_lost_definition(damaged):
    # THE KEPT NEGATIVE, from a real merge. `def _f1(...)` became
    # `from pkg.occ import recall_f1 as _f1`: the name is still bound, nothing was removed, and a
    # def-only census called it a HARD ERROR. Twice.
    rep = damaged[0]
    assert "_f1" not in [r["name"] for r in rep["lost"]]
    # nor is it silently ignored -- it surfaces as a REVIEW row, which is what it is.
    sig = {r["name"]: r for r in rep["signature_changed"]}
    assert sig["_f1"]["was"].startswith("def _f1(") and sig["_f1"]["now"] == "import _f1"


def test_no_untouched_name_is_ever_reported(damaged):
    rep = damaged[0]
    named = {r["name"] for r in rep["lost"]} | {r["name"] for r in rep["signature_changed"]}
    assert named & {"kept", "CONST", "os", "already"} == set(), named


def test_a_syntax_error_is_its_own_bucket_not_a_storm_of_lost_defs(damaged):
    # The prototype let a SyntaxError in the new copy report EVERY definition in that file as
    # lost: one broken file, a hundred false hard errors.
    rep = damaged[0]
    assert rep["counts"]["unparseable_newly"] == 1
    assert rep["unparseable"]["newly"] == ["broken.py"]
    assert "fine" not in [r["name"] for r in rep["lost"]]


def test_a_file_unparseable_in_BOTH_trees_is_not_a_merge_finding(tmp_path):
    # Found by running the census on this very repo: tools/tour.py carries an f-string this
    # interpreter rejects and fails in both trees. Counting it pinned the verdict at REVIEW
    # forever, and a permanent yellow light is a light nobody reads.
    b, n = tmp_path / "b", tmp_path / "n"
    for d in (b, n):
        d.mkdir()
        (d / "broken.py").write_text("def f(:\n    pass\n")
        (d / "ok.py").write_text("def g():\n    pass\n")
    rep = merge_census(str(b), str(n), base_is_ref=False)
    assert rep["counts"]["unparseable_base"] == 1 and rep["counts"]["unparseable_new"] == 1
    assert rep["counts"]["unparseable_newly"] == 0
    assert rep["verdict"] == "CLEAN"


# --------------------------------------------------------------------------------------
# 2. DETECTION, one number per leg, on eight injected faults.
# --------------------------------------------------------------------------------------

def test_the_fixture_injects_eight_named_faults():
    assert len(CENSUS_FAULTS) == 8
    assert {f[1] for f in CENSUS_FAULTS} == {"definitions", "signatures", "line_counts",
                                             "unparseable", "none"}


def test_definition_leg_catches_the_deletion_and_the_deleted_file(damaged):
    c = damaged[0]["counts"]
    assert c["lost"] == 2 and c["lost_unexplained"] == 1 and c["lost_moved"] == 1
    assert [r["name"] for r in damaged[0]["lost"] if not r["moved_to"]] == ["deleted_def"]
    assert c["files_deleted"] == 1
    assert damaged[0]["files_deleted"][0]["file"] == "gone.py"
    assert damaged[0]["files_deleted"][0]["moved"] == {}, "orphan is findable nowhere"


def test_a_definition_that_moved_module_is_not_counted_as_damage(damaged):
    moved = [r for r in damaged[0]["lost"] if r["moved_to"]]
    assert [r["name"] for r in moved] == ["travels"]
    assert moved[0]["moved_to"] == ["mod_b.py"]
    assert damaged[0]["counts"]["lost_unexplained"] == 1, "a move was counted as a loss"


def test_signature_leg_catches_the_dropped_passthrough(damaged):
    # Sweep 120's SECOND casualty, which the name-level census passed.
    sig = {r["name"]: r for r in damaged[0]["signature_changed"]}
    assert sig["sim"]["was"] == "def sim(a, b, record_every=)"
    assert sig["sim"]["now"] == "def sim(a, b)"
    assert damaged[0]["counts"]["signature_changed"] == 3


def test_a_keyword_only_default_that_moved_is_caught_where_a_counter_would_miss_it():
    # THE REASON DEFAULTS ARE RECORDED PER ARGUMENT AND NOT COUNTED. Keyword-only parameters may
    # carry defaults in ANY order, so these two have the same number of defaults and different
    # meanings; a defaults-counting census calls them identical.
    a = ast.parse("def h(*, a=1, b): pass").body[0]
    b = ast.parse("def h(*, a, b=1): pass").body[0]
    n_a = sum(1 for d in a.args.kw_defaults if d is not None)
    n_b = sum(1 for d in b.args.kw_defaults if d is not None)
    assert n_a == n_b == 1, "the counter's whole view of these two signatures"
    assert signature_of(a) != signature_of(b), "per-argument defaults are what catch this"
    assert signature_of(a) == "def h(*, a=, b)" and signature_of(b) == "def h(*, a, b=)"


def test_line_count_leg_finds_the_shrink_and_the_join_says_it_moved(damaged):
    # Sweep 125's case, resolved rather than merely reported: 90 of 100 lines are gone from
    # data.txt and every one of them is in data_extra.txt. Do NOT restore.
    c = damaged[0]["counts"]
    assert c["shrunk"] == 1 and c["shrunk_moved"] == 1 and c["shrunk_unexplained"] == 0
    s = damaged[0]["shrunk"][0]
    assert (s["file"], s["base_lines"], s["new_lines"]) == ("data.txt", 100, 10)
    assert s["verdict"] == "moved"
    assert s["moved_into"][0] == {"file": "data_extra.txt", "lines": 90, "fraction": 1.0}


def test_a_shrink_with_nowhere_to_have_gone_stays_unexplained(tmp_path):
    # The other half of the leg: without a destination, a shrink is damage and says so.
    b, n = tmp_path / "b", tmp_path / "n"
    for d in (b, n):
        d.mkdir()
    (b / "d.txt").write_text("\n".join("row %d" % i for i in range(100)) + "\n")
    (n / "d.txt").write_text("row 0\n")
    rep = merge_census(str(b), str(n), base_is_ref=False)
    assert rep["counts"]["shrunk_unexplained"] == 1
    assert rep["shrunk"][0]["verdict"] == "unexplained" and rep["shrunk"][0]["moved_into"] == []
    assert rep["verdict"] == "REVIEW"


def test_the_verdict_is_losses_found_when_something_is_actually_gone(damaged):
    assert damaged[0]["verdict"] == "LOSSES FOUND"


# --------------------------------------------------------------------------------------
# 3. THE HOUSE RULES: deterministic, and ambiguity refused rather than guessed.
# --------------------------------------------------------------------------------------

def test_the_same_pair_gives_the_same_report(damaged):
    rep, b, n = damaged
    again = merge_census(b, n, base_is_ref=False)
    assert json.dumps(again, sort_keys=True) == json.dumps(rep, sort_keys=True)


def test_a_base_that_is_neither_a_directory_nor_a_ref_is_refused(damaged):
    with pytest.raises(ValueError, match="neither"):
        merge_census("definitely-not-a-ref-or-a-directory", damaged[2])


def test_a_base_that_is_both_a_directory_and_a_ref_is_refused(tmp_path, monkeypatch):
    # merge_trees refuses ambiguity loudly rather than auto-deciding; so does its partner. The
    # realistic shape of this: you stand in a repo that has a branch `main2` AND a directory
    # ./main2, and `merge_census("main2", ".")` could mean either. Guessing here would silently
    # census the wrong pair and report a clean bill of health on the wrong trees.
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "m.py").write_text("def f():\n    pass\n")
    for cmd in (["init", "-q"], ["add", "-A"],
                ["-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "one"]):
        subprocess.run(["git"] + cmd, cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "branch", "main2"], cwd=repo, check=True, capture_output=True)
    (repo / "main2").mkdir()
    monkeypatch.chdir(repo)                      # a directory path resolves against the CWD
    with pytest.raises(ValueError, match="ambiguous"):
        merge_census("main2", str(repo))
    # ...and the caller can settle it either way.
    assert merge_census("main2", str(repo), base_is_ref=True)["base_is_ref"] is True
    assert merge_census("main2", str(repo), base_is_ref=False)["base_is_ref"] is False


def test_a_git_ref_base_is_read_without_checking_anything_out(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "m.py").write_text("def kept():\n    pass\n\n\ndef gone():\n    pass\n")
    for cmd in (["init", "-q"], ["add", "-A"],
                ["-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "one"]):
        subprocess.run(["git"] + cmd, cwd=repo, check=True, capture_output=True)
    (repo / "m.py").write_text("def kept():\n    pass\n")
    rep = merge_census("HEAD", str(repo))
    assert rep["base_is_ref"] is True, "a bare ref must be detected without base_is_ref="
    assert rep["counts"]["lost_unexplained"] == 1
    assert [r["name"] for r in rep["lost"]] == ["gone"]
    assert rep["verdict"] == "LOSSES FOUND"


# --------------------------------------------------------------------------------------
# 4. THE INDEX ITSELF, and the scope decisions in it.
# --------------------------------------------------------------------------------------

def test_def_index_binds_imports_assignments_classes_and_methods():
    idx = def_index("import os\nfrom a import b as c\nX = 1\n"
                    "class K:\n    Y = 2\n    def m(self, x=1):\n        pass\n")
    assert idx["os"][0] == "import" and idx["c"][0] == "import"
    assert idx["X"][0] == "assign" and idx["K.Y"][0] == "assign"
    assert idx["K"][0] == "class" and idx["K.m"] == ("def", "def m(self, x=)")


def test_a_def_nested_inside_a_function_is_deliberately_not_indexed():
    # Indexing local helpers would report every refactor of one as a lost definition -- the
    # cry-wolf failure in a second costume.
    idx = def_index("def outer():\n    def inner():\n        pass\n    return inner\n")
    assert set(idx) == {"outer"}


def test_a_decorator_change_is_a_call_shape_change():
    a = ast.parse("@staticmethod\ndef f(x): pass").body[0]
    b = ast.parse("def f(x): pass").body[0]
    assert signature_of(a) != signature_of(b)
    # ...but a decorator's ARGUMENTS are tuning, not shape.
    c = ast.parse("@cache(128)\ndef f(x): pass").body[0]
    d = ast.parse("@cache(256)\ndef f(x): pass").body[0]
    assert signature_of(c) == signature_of(d)


# --------------------------------------------------------------------------------------
# 5. WIRING, and the promise that merge_trees did not change.
# --------------------------------------------------------------------------------------

def test_the_faculty_is_wired_to_the_mind_and_documented(damaged):
    m = lecore.UnifiedMind(dim=256, seed=0)
    assert callable(getattr(m, "merge_census", None))
    assert (m.merge_census.__doc__ or "").strip(), "an undocumented verb is undiscoverable"
    rep, b, n = damaged
    assert json.dumps(m.merge_census(b, n, base_is_ref=False), sort_keys=True) == \
        json.dumps(rep, sort_keys=True), "the mind method must delegate, not reimplement"


def test_merge_trees_still_behaves_exactly_as_documented(tmp_path):
    # The census is a PARTNER, not an edit: merge_trees' code was not touched this sweep and this
    # is the trap that keeps it that way.
    m = lecore.UnifiedMind(dim=256, seed=0)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "same.py").write_text("x = 1\n")
    (b / "same.py").write_text("x = 1\n")
    (a / "ext.py").write_text("x = 1\n")
    (b / "ext.py").write_text("x = 1\ny = 2\n")
    (a / "coll.py").write_text("p = 1\n")
    (b / "coll.py").write_text("q = 2\n")
    (b / "onlytheirs.py").write_text("z = 3\n")
    r = m.merge_trees(str(a), str(b))
    assert r["identical"] == 1 and r["only_theirs"] == ["onlytheirs.py"] and r["only_ours"] == []
    v = {row["file"]: row["verdict"] for row in r["differ"]}
    # CHARACTERISATION, not aspiration: the unique-line tests are ordered BEFORE the prefix test,
    # so a pure append reads as "ours_is_base" (ours has zero unique lines) and never reaches the
    # append_extension branch. Writing this test I expected append_extension_theirs and was wrong
    # -- which is exactly what a characterisation test is for. The outcome is the same either way
    # (theirs wins); only the label differs.
    assert v == {"ext.py": "ours_is_base", "coll.py": "both_changed"}
    assert r["differ"][1]["unique_lines"] == {"ours": 0, "theirs": 1}     # the ext.py row
    assert r["n_both_changed"] == 1 and r["applied"] == [] and r["refused"] == []


# ~8 s on this repo (1,959 + 1,968 files), and it depends on the PRE_MERGE_122 tag existing, so it
# is marked slow rather than risking the 15 s per-test budget on a loaded box. It is the real
# regression trap: the merge that shipped sweep 122 lost nothing, and it must stay that way.
@pytest.mark.slow
def test_the_real_sweep_122_merge_lost_nothing():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    ok = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "PRE_MERGE_122^{tree}"],
                        cwd=root, capture_output=True)
    if ok.returncode != 0:
        pytest.skip("PRE_MERGE_122 is not in this checkout")
    rep = merge_census("PRE_MERGE_122", str(root))
    assert rep["counts"]["lost_unexplained"] == 0, rep["lost"]
    assert rep["counts"]["files_deleted"] == 0
    assert rep["counts"]["shrunk_unexplained"] == 0, rep["shrunk"]
