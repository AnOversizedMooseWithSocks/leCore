"""B3 -- batched bind/unbind at the sites where it is actually possible.

Unbinding one composite against K keys, or binding a fixed operand against K vectors, is ONE batched rfft -- not K
of them inside a Python loop. Bit-identical (`np.array_equal`), measured:

    recover_all   3.8x at (D=1024, K=8)    4.3x at K=16    1.7x by K=64
    bind_batch    2.8x at (D=1024, K=8)

THE CORRECTION, and it shrank the item by 25x. A first AST scan reported **77** candidate loop sites across 30
modules. It was counting LOOP-CARRIED ACCUMULATORS -- `acc = bind(acc, d)` in the VSA machine's interpreter, `x`
redrawn every trial in `flatness`. Those have a data dependency: each bind consumes the previous one's output, so
there is no set of K independent binds to batch. A strict scan (bind/unbind inside a comprehension, one operand
loop-invariant, the other the loop variable) finds exactly **three**, and all three are now batched.
"""

import ast
import pathlib

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import (bind, bind_batch, bind_fixed, involution_batch, unbind)
from holographic.agents_and_reasoning.holographic_hopfield import _decode_combo, _structure_project, dense_cleanup
from holographic.misc.holographic_determinism import argmax_tiebreak
from holographic.misc.holographic_superposed import recover_all


def _world(D=256, K=6, V=10, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=D), r.normal(size=(K, D)), r.normal(size=(V, D))


# ---------------------------------------------------------------------------------------------------------
# the primitives are bit-identical to the loops they replace
# ---------------------------------------------------------------------------------------------------------

def test_recover_all_is_bit_identical_to_a_loop_of_unbinds():
    for D, K in ((256, 4), (512, 8), (1024, 16)):
        r = np.random.default_rng(D + K)
        z, keys = r.normal(size=D), r.normal(size=(K, D))
        assert np.array_equal(np.stack([unbind(z, k) for k in keys]), recover_all(z, keys))


def test_bind_batch_is_bit_identical_to_a_loop_of_binds():
    for D, K in ((256, 4), (1024, 8)):
        r = np.random.default_rng(D)
        A, B = r.normal(size=(K, D)), r.normal(size=(K, D))
        assert np.array_equal(np.stack([bind(A[i], B[i]) for i in range(K)]), bind_batch(A, B))


def test_bind_fixed_with_the_involution_stack_is_recover_all():
    D, K = 512, 8
    r = np.random.default_rng(1)
    trace, keys = r.normal(size=D), r.normal(size=(K, D))
    assert np.array_equal(bind_fixed(trace, involution_batch(keys)), recover_all(trace, keys))


# ---------------------------------------------------------------------------------------------------------
# the three rewritten sites, each against the code it replaced
# ---------------------------------------------------------------------------------------------------------

def test_structure_project_is_bit_identical_to_the_loop_it_replaced():
    z, roles, fillers = _world()
    parts = [bind(r, dense_cleanup(unbind(z, r), fillers, 25.0, 1, readout="softmax")) for r in roles]
    ref = np.sum(parts, axis=0)
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    assert np.array_equal(ref, _structure_project(z, roles, fillers, 25.0, 1))


def test_decode_combo_is_identical_to_the_loop_it_replaced():
    z, roles, fillers = _world(seed=2)
    ref = tuple(argmax_tiebreak(fillers @ unbind(z, r)) for r in roles)
    assert _decode_combo(z, roles, fillers) == ref


def test_the_ai_recall_path_is_bit_identical():
    D, K = 512, 8
    r = np.random.default_rng(3)
    trace, keys = r.normal(size=D), r.normal(size=(K, D))
    assert np.array_equal(np.stack([unbind(trace, k) for k in keys]),
                          bind_fixed(trace, involution_batch(keys)))


def test_structure_project_still_denoises_toward_the_vocabulary():
    # Not just "the same numbers" -- the faculty must still work. A noised composite must decode back.
    D, K, V = 256, 4, 8
    r = np.random.default_rng(7)
    roles = r.normal(size=(K, D))
    fillers = r.normal(size=(V, D))
    fillers /= np.linalg.norm(fillers, axis=1, keepdims=True)
    truth = tuple(int(i) for i in (0, 3, 5, 1))
    z = np.sum(bind_batch(roles, fillers[list(truth)]), axis=0)
    noisy = z + 0.05 * r.normal(size=D)
    cleaned = _structure_project(noisy, roles, fillers, beta=40.0, steps=2)
    assert _decode_combo(cleaned, roles, fillers) == truth


# ---------------------------------------------------------------------------------------------------------
# THE RETRACTION: the other sites cannot batch, and the registry says so
# ---------------------------------------------------------------------------------------------------------

def test_a_loop_carried_accumulator_cannot_be_batched_and_the_scan_now_knows_it():
    # `acc = bind(acc, d)` -- each bind consumes the previous one's OUTPUT. The sequence is the semantics.
    # Batching would require the inputs to be independent, and they are not. This is a data dependency, closed
    # by mathematics, not a measurement that came out badly.
    D = 128
    r = np.random.default_rng(4)
    atoms = r.normal(size=(5, D))
    acc = atoms[0]
    for d in atoms[1:]:
        acc = bind(acc, d)
    # a "batched" version binds atoms[0] against each of the rest independently -- a DIFFERENT object
    naive = np.sum(bind_fixed(atoms[0], atoms[1:]), axis=0)
    assert not np.allclose(acc, naive)


def test_the_strict_scan_finds_three_sites_not_seventy_seven():
    # The scan that produced the corrected number, run as a test so the claim cannot rot. A batchable site is a
    # bind/unbind inside a COMPREHENSION with one operand loop-invariant and the other the loop variable.
    found = []
    for f in pathlib.Path("holographic").rglob("*.py"):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
                continue
            targets = {t.id for g in node.generators for t in ast.walk(g.target) if isinstance(t, ast.Name)}
            for call in ast.walk(node.elt):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id in ("bind", "unbind")):
                    names = [a.id for a in call.args if isinstance(a, ast.Name)]
                    if len(names) == 2:
                        inv = [n for n in names if n not in targets]
                        var = [n for n in names if n in targets]
                        if len(inv) == 1 and len(var) == 1:
                            found.append((f.stem, call.lineno))
    # all three were rewritten, so the strict scan should now find NONE left in the batchable form
    assert len(found) == 0, "un-batched sites remain: %s" % found


def test_the_registry_records_the_right_clients_and_the_retraction():
    import os
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "tools"))
    from unifiers import NOT_APPLICABLE, PENDING, REGISTRY, cites

    key = "superposed width (bind_fixed / recover_all / pack)"
    assert REGISTRY[key]["clients"] == ["holographic_hopfield", "holographic_ai"]
    assert cites("holographic_hopfield", key, repo) and cites("holographic_ai", key, repo)
    assert not any(u == key for u, _c in PENDING)                     # 2/2 wired

    # The five modules the first scan wrongly named are retired with the REASON, not silently dropped. They share
    # ONE entry under a slash-joined key -- `unaccounted()` splits on "/" -- because they share one reason.
    joined = next(k for k in NOT_APPLICABLE if k[0] == key and "holographic_machine" in k[1])
    assert set(joined[1].split("/")) == {"holographic_machine", "holographic_flatness", "holographic_query",
                                         "holographic_reasoning", "holographic_sequence"}
    assert "loop-carried" in NOT_APPLICABLE[joined].lower()
    assert "77" in NOT_APPLICABLE[joined]
