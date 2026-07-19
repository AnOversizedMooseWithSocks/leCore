"""C13: prove the engine is deterministic WITHOUT PYTHONHASHSEED=0 -- and pin the one violation that wasn't.

The audit's reasoning was sound: determinism that depends on an env var is a footgun, because PYTHONHASHSEED
cannot be set after the interpreter starts, so an embedder (a node pack, a desktop app) physically cannot fix it
from inside their own process. They grepped, found no `hash()` calls, and concluded the requirement was probably
vestigial. My first grep agreed with them.

BOTH GREPS WERE WRONG. holographic_sequence seeded its atom vectors with `abs(hash((seed, sym)))`. Python salts
hash() per process for str, so every atom -- and the z-score the module returns -- moved run to run:
discover_sequential() measured 1.64 / 1.61 / 2.17 on identical input under PYTHONHASHSEED=random. It survived
greps because `hash((seed, sym))` looks nothing like a content hash, and it survived every TEST because CI pins
PYTHONHASHSEED=0, which papers over exactly this. The env var was not vestigial; it was load-bearing, and it was
hiding a bug rather than preventing one.

Fixed with hashlib (the engine's own rule: hashlib for content hashes, never hash()). The z VALUE moved
2.8379187419114142 -> 2.2324732585958955, because the atom vectors are differently seeded; the DECISION did not
flip ('executable' both sides), which is the contract that matters -- an atom's value is arbitrary, only its
consistency is load-bearing.

These tests run the check in SUBPROCESSES with PYTHONHASHSEED unset/random, because an in-process assertion
cannot see a salt it inherited. That is the whole reason the bug lived this long.
"""
import os
import subprocess
import sys

import pytest

_SNIPPET = (
    "import lecore;"
    "m = lecore.UnifiedMind(dim=256, seed=0);"
    "[m.absorb([(['a','b','c','d'],'seq')]) for _ in range(3)];"
    "print(repr(m.discover_sequential()))"
)


def _run_with_salt(snippet, salt):
    """Run a snippet in a FRESH interpreter with a given PYTHONHASHSEED. In-process assertions are blind to the
    salt they were started with -- which is precisely how a salted hash() hid here for so long."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = salt
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip()


@pytest.mark.parametrize("salts", [("1", "2", "12345")])
def test_discover_sequential_is_salt_independent(salts):
    """THE regression. Under the old salted hash() this returned a different z per process."""
    outs = {_run_with_salt(_SNIPPET, s) for s in salts}
    assert len(outs) == 1, ("discover_sequential is process-dependent again -- a salted hash() is back", outs)


def test_salted_runs_match_the_pinned_seed_zero_result():
    """A random salt must agree with PYTHONHASHSEED=0, or the env var is still load-bearing somewhere."""
    assert _run_with_salt(_SNIPPET, "99999") == _run_with_salt(_SNIPPET, "0")


def test_no_module_seeds_an_rng_from_pythons_salted_hash():
    """The static half, so the next one is caught at review rather than by a z-score wobbling in production.
    hash() is fine on ints/tuples-of-ints and for in-memory dict keys; it is NOT fine as a SEED, because the
    value crosses a process boundary the moment anyone compares two runs."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "holographic"
    bad = []
    for f in root.rglob("holographic_*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#")[0]
            if re.search(r"(default_rng|RandomState|seed)\s*\([^)]*[^a-z_.]hash\s*\(", code):
                bad.append("%s:%d" % (f.name, i))
    assert not bad, ("an RNG is seeded from Python's SALTED hash() -- use hashlib (see holographic_sequence's "
                     "WHY-comment for the measured failure)", bad)


def test_the_core_paths_were_already_salt_independent():
    """The audit's instinct was RIGHT about the rest of the engine: hashing, routing, tagging and rendering never
    used hash(). Worth pinning -- it is the evidence that PYTHONHASHSEED=0 is now belt-and-braces rather than
    load-bearing, which is what lets the requirement be relaxed."""
    snippet = (
        "import lecore, hashlib, numpy as np;"
        "m = lecore.UnifiedMind(dim=128, seed=0);"
        "img = m.render_mesh(m.mesh_box(), m.camera(eye=(2,2,2), target=(0,0,0)), width=16, height=16);"
        "print(hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()[:16],"
        " m.infer_semantic_tag('render_scene'),"
        " m.find_capability('search a big pile of vectors')[0].name)"
    )
    outs = {_run_with_salt(snippet, s) for s in ("0", "7", "424242")}
    assert len(outs) == 1, outs
