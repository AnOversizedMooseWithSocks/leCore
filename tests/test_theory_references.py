"""BLD-3: the theory-and-guarantees doc (THEORY.md) must stay backed -- every test it cites must exist, so the
document can never rot into asserting a claim with no test behind it. This is the doc's own guarantee."""
import re
import os


def test_theory_doc_references_are_live():
    here = os.path.dirname(os.path.abspath(__file__))                 # tests/ -- where cited test files live
    repo_root = os.path.dirname(here)                                  # repo root -- where THEORY.md lives
    # THEORY.md lives in docs/ since the repo reorganisation; fall back to the old root location so this test
    # works in either layout.
    theory = os.path.join(repo_root, "docs", "THEORY.md")
    if not os.path.exists(theory):
        theory = os.path.join(repo_root, "THEORY.md")
    text = open(theory).read()

    # explicit `test_file.py::test_function` citations must resolve to a real function in that real file
    refs = re.findall(r"(test_[a-z0-9_]+\.py)::([A-Za-z0-9_]+)", text)
    assert refs, "THEORY.md should cite specific tests by name"
    missing = []
    for fname, func in refs:
        path = os.path.join(here, fname)
        if not os.path.exists(path) or func not in open(path).read():
            missing.append(f"{fname}::{func}")
    assert not missing, f"THEORY.md cites tests that do not exist: {missing}"

    # every bare `test_file.py` mention must at least be a real file
    files = set(re.findall(r"`?(test_[a-z0-9_]+\.py)`?", text))
    missing_files = [f for f in sorted(files) if not os.path.exists(os.path.join(here, f))]
    assert not missing_files, f"THEORY.md cites test files that do not exist: {missing_files}"
