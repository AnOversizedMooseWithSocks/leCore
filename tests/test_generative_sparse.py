"""Sparse readout in the generative attractor (the session's finding applied to generation).

Pins: (1) softmax default unchanged; (2) sparse generation stays VALID (reencodes its decoded combo at
cosine ~1); (3) the MEASURED win -- sparse CURES generative mode collapse (softmax funnels seeds into the
same few structures; sparse stays diverse); (4) KEPT NEGATIVE -- over a continuous codebook generate() is
unaffected by the readout (both snap to a stored atom); (5) ABOVE/BELOW -- the mind delegates and threads.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import unitary_vector, random_vector, bind, unbind
from holographic.agents_and_reasoning.holographic_hopfield import generate, generate_structure, _unit_rows
from holographic.misc.holographic_unified import UnifiedMind


def _u(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _roles_fillers(dim, S, V, seed=123):
    r = np.random.default_rng(seed)
    roles = np.array([unitary_vector(dim, r) for _ in range(S)])
    fillers = np.array([random_vector(dim, r) for _ in range(V)])
    return roles, fillers


def _decode(z, roles, filu):
    return tuple(int((filu @ _u(unbind(z, roles[i]))).argmax()) for i in range(len(roles)))


def _validity_diversity(roles, fillers, readout, seeds=20, beta1=60.0):
    filu = _unit_rows(fillers); recon = []; combos = set()
    for s in range(seeds):
        z = generate_structure(roles, fillers, steps=16, beta1=beta1, seed=s, readout=readout)
        combo = _decode(z, roles, filu)
        clean = _u(np.sum([bind(roles[i], filu[combo[i]]) for i in range(len(roles))], axis=0))
        recon.append(float(z @ clean)); combos.add(combo)
    return float(np.mean(recon)), len(combos) / seeds


def test_generate_softmax_default_unchanged():
    # default readout is softmax: identical output to the explicit softmax path (backward compatible)
    roles, fillers = _roles_fillers(512, 3, 8)
    a = generate_structure(roles, fillers, seed=4)
    b = generate_structure(roles, fillers, seed=4, readout="softmax")
    assert np.allclose(a, b)
    cb = _unit_rows(np.array([random_vector(512, np.random.default_rng(i)) for i in range(6)]))
    assert np.allclose(generate(cb, seed=1), generate(cb, seed=1, readout="softmax"))


def test_sparse_structures_are_valid():
    # sparse generation lands on the valid-structure manifold (reencodes its decoded combo at cosine ~1)
    roles, fillers = _roles_fillers(1024, 3, 12)
    rec_soft, _ = _validity_diversity(roles, fillers, "softmax")
    rec_sparse, _ = _validity_diversity(roles, fillers, "sparsemax")
    assert rec_soft > 0.999 and rec_sparse > 0.999          # both perfectly valid; sparse pays no validity cost


def test_sparse_cures_mode_collapse():
    # the measured win: softmax funnels seeds into the same few structures; sparse stays diverse, same validity
    roles, fillers = _roles_fillers(1024, 3, 12)
    _, div_soft = _validity_diversity(roles, fillers, "softmax")
    _, div_sparse = _validity_diversity(roles, fillers, "sparsemax")
    assert div_sparse >= div_soft + 0.2                     # clearly more distinct structures
    assert div_sparse >= 0.8                                # nearly every seed a distinct valid structure


def test_continuous_generate_unaffected_kept_negative():
    # KEPT NEGATIVE: over a continuous codebook the readout does NOT change generation -- both readouts
    # land ON the manifold (and snap to a stored atom). The sparse win is for composed structures, not this.
    r = np.random.default_rng(7); a = unitary_vector(1024, r); b = unitary_vector(1024, r)
    O = np.arccos(np.clip(_u(a) @ _u(b), -1, 1))
    slerp = lambda t: (np.sin((1 - t) * O) * _u(a) + np.sin(t * O) * _u(b)) / np.sin(O)
    coarse = np.array([slerp(t) for t in np.linspace(0, 1, 10)])
    fineu = _unit_rows(np.array([slerp(t) for t in np.linspace(0, 1, 400)]))
    for ro in ("softmax", "sparsemax"):
        v = np.mean([float((fineu @ generate(coarse, steps=12, seed=s, readout=ro)).max()) for s in range(12)])
        assert v > 0.99                                     # both stay on the manifold; readout doesn't break it


def test_mind_generate_structure_delegates_and_threads():
    # ABOVE/BELOW: the mind's generate_structure IS the kernel generator with the readout passed through
    roles, fillers = _roles_fillers(512, 3, 10)
    mind = UnifiedMind(dim=256, seed=5)
    for ro in ("softmax", "sparsemax"):
        m = mind.generate_structure(roles, fillers, seed=2, readout=ro)
        k = generate_structure(roles, fillers, seed=2, readout=ro)
        assert np.allclose(m, k)                            # delegation, not reimplementation
