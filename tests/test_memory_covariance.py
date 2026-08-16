"""S6 -- the shufflebrain symmetry-class findings as REGRESSION TRAPS. The vision-tower
renumbering bug corrupted a multi-tower model because a re-basis was applied incoherently; these
traps pin the theorems that make coherent re-basis SAFE (and incoherent re-basis detectable):
the GDN outer-product memory is exactly covariant under the full orthogonal group, HRR under the
cyclic group ONLY. If either ever drifts, a rebasing 'optimization' somewhere has gone wrong."""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind


def _unit(v):
    return v / np.linalg.norm(v)


def test_gdn_memory_is_exactly_orthogonal_covariant():
    rng = np.random.default_rng(0)
    dk = 96
    Ks = [_unit(rng.standard_normal(dk)) for _ in range(15)]
    Vs = [_unit(rng.standard_normal(dk)) for _ in range(15)]
    S = np.zeros((dk, dk))
    for k, v in zip(Ks, Vs):
        S = 0.98 * S + np.outer(k, v)
    P = np.eye(dk)[rng.permutation(dk)]
    for k, v in zip(Ks, Vs):
        a = (P @ S).T @ (P @ k)
        b = S.T @ k
        assert np.max(np.abs(a - b)) < 1e-12, "coherent re-basis must be invisible"


def test_hrr_is_cyclic_covariant_and_NOT_permutation_covariant():
    rng = np.random.default_rng(1)
    D = 512
    ks = [_unit(rng.standard_normal(D)) for _ in range(8)]
    vs = [_unit(rng.standard_normal(D)) for _ in range(8)]
    T = np.sum([bind(k, v) for k, v in zip(ks, vs)], axis=0)
    base = np.mean([float(_unit(unbind(T, k)) @ v) for k, v in zip(ks, vs)])
    # cyclic, coherent: shifting trace AND cue cancels -- originals return exactly
    csh = np.mean([float(_unit(unbind(np.roll(T, 37), np.roll(k, 37))) @ v)
                   for k, v in zip(ks, vs)])
    assert abs(csh - base) < 1e-9, "the cyclic symmetry is HRR's contract"
    # arbitrary coherent permutation: NOT covariant -- the kept negative, pinned so nobody
    # 'generalizes' HRR re-basis and silently destroys every stored memory
    P = rng.permutation(D)
    perm = np.mean([float(_unit(unbind(T[P], k[P])) @ v[P]) for k, v in zip(ks, vs)])
    assert perm < 0.2 * base, "if this ever passes, an impossible covariance appeared -- suspect the probe"
