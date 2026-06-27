"""The ISA conformance suite (ISA-2): the contract's teeth. Every production base instruction must match its
definitional reference (TOL on continuous outputs, EXACT on decisions/reindexes), the identity-based golden
vectors must hold, and -- the centerpiece -- the bind_batch class (a value-conformant change that flips a
decision) must be caught by construction."""

import numpy as np

from holographic_ai import bind, unbind, bundle, cosine, involution, permute, random_vector, bind_batch
from holographic_determinism import argmax_tiebreak
from holographic_reference import (
    ref_bind, ref_involution, ref_unbind, ref_permute, ref_bundle, ref_cosine,
    value_conformant, exact_conformant, decision_conformant, run_conformance, TOL, _selftest,
)


def test_module_selftest():
    _selftest()


def test_all_base_ops_conform_to_their_reference():
    report = run_conformance(dim=64, seed=0)
    for op, r in report.items():
        assert r["passed"], f"{op} failed conformance: {r}"
    # the continuous ops match to ~machine epsilon, far inside TOL; the exact ops match bit-for-bit
    assert report["bind"]["class"] == "TOL" and report["bind"]["max_diff"] < 1e-9
    assert report["involution"]["class"] == "EXACT" and report["involution"]["max_diff"] == 0.0
    assert report["permute"]["class"] == "EXACT"


def test_golden_vectors_are_the_convolution_identities():
    # Golden vectors that are hand-verifiable FACTS about circular convolution -- they pin that bind really is
    # convolution, and they cannot rot the way frozen float arrays would.
    rng = np.random.default_rng(0)
    D = 256
    a = random_vector(D, rng)
    b = random_vector(D, rng)
    d0 = np.zeros(D); d0[0] = 1.0                      # the convolution identity
    dk = np.zeros(D); dk[7] = 1.0                      # a shifted impulse
    assert value_conformant(bind(a, d0), a)            # bind with the impulse is the identity (exact)
    assert value_conformant(bind(a, dk), np.roll(a, 7))  # bind with a shifted impulse is a cyclic shift (exact)
    assert value_conformant(bind(a, b), bind(b, a))    # bind is commutative (exact)
    # round-trip: unbind recovers b only APPROXIMATELY (involution is an exact inverse for unitary vectors, not
    # random ones), so the contract's guarantee is that b is the nearest atom -- the cleanup DECISION recovers it.
    cand = np.stack([random_vector(D, rng) for _ in range(8)] + [b])
    rec = unbind(bind(a, b), a)
    sims = cand @ rec / (np.linalg.norm(cand, axis=1) * (np.linalg.norm(rec) + 1e-12))
    assert argmax_tiebreak(sims) == len(cand) - 1      # b (placed last) is recovered as the cleanup winner


def test_exact_ops_are_bit_for_bit():
    rng = np.random.default_rng(1)
    a = random_vector(48, rng)
    assert exact_conformant(involution(a), ref_involution(a))
    assert exact_conformant(involution(involution(a)), a)   # exactly self-inverse
    assert exact_conformant(permute(a, 5), ref_permute(a, 5))
    assert exact_conformant(permute(permute(a, 5), -5), a)  # exactly invertible


def test_zero_vector_edges_are_pinned():
    # The EXACT edge cases the contract names: a zero-sum bundle is the zero vector, a zero-norm cosine is 0.0.
    a = random_vector(64, np.random.default_rng(2))
    assert np.array_equal(bundle([a, -a]), np.zeros(64))    # zero-sum -> zero vector, not a divide-by-zero
    assert cosine(a, np.zeros(64)) == 0.0


def test_bind_batch_is_value_conformant():
    # The production bind_batch matches the looped reference within tolerance -- it is a legitimate
    # microarchitecture variant on VALUE. (Its decision-safety is the separate concern below.)
    rng = np.random.default_rng(3)
    A = np.stack([random_vector(64, rng) for _ in range(5)])
    B = np.stack([random_vector(64, rng) for _ in range(5)])
    batched = bind_batch(A, B)
    for i in range(5):
        assert value_conformant(batched[i], ref_bind(A[i], B[i]))


def test_bind_batch_class_a_value_conformant_change_can_flip_a_decision():
    # THE REGRESSION the suite exists for. A change can stay within numeric tolerance yet flip the observable
    # cleanup decision -- exactly the bind_batch bug (bit-exact to 1e-12, but it flipped a trajectory). A
    # value-only check would PASS it; the contract's separate, exact DECISION check catches it.
    sims = np.array([0.5, 0.5, 0.3])                  # a tie at 0/1; the contract picks the lower index, 0
    flipped = sims + np.array([0.0, 1e-12, 0.0])      # a sub-tolerance bump flips the winner to 1
    assert value_conformant(sims, flipped)            # within TOL -> a value-only suite would accept it
    assert not decision_conformant(sims, flipped)     # but the DECISION moved -> the suite REJECTS it


def test_summation_order_changes_a_reduction_so_decisions_must_be_pinned():
    # The literal mechanism behind the bind_batch class: the SAME numbers summed in two orders differ, and on a
    # near-tie that flips an argmax. Shown via cancellation here; in the real bug it was ULPs.
    x = np.array([1e16, 1.0, -1e16, -1.0])            # sums to 0 in exact arithmetic; float order changes it
    fwd = 0.0
    for v in x:
        fwd += v
    rev = 0.0
    for v in x[::-1]:
        rev += v
    assert fwd != rev                                 # summation order changed the result
    # if fwd and rev were two candidates' scores against a third midway between them, the winner flips with
    # order -- whichever of fwd/rev is larger beats the midpoint, the other loses to it:
    mid = (fwd + rev) / 2
    assert argmax_tiebreak([fwd, mid]) != argmax_tiebreak([rev, mid])
