"""Extension conformance (ISA-3): each opt-in bind-mode extension must deliver the measured regime win that
justifies it over base `bind`, and the base kernel must stay minimal and unchanged. See ISA_EXTENSIONS.md."""

import math

import numpy as np

from holographic_ai import random_vector, bind, unbind, bundle, cosine
from holographic_reference import run_conformance


def test_clifford_rotation_composition_is_exact():
    # Regime: 3-D rotations. Win: the geometric product of two rotors IS the composed rotor, so composing then
    # applying equals applying sequentially -- exact (one product), which base convolution cannot do.
    from holographic_clifford import CliffordAlgebra
    cl = CliffordAlgebra()
    rng = np.random.default_rng(0)
    v = rng.standard_normal(3)
    R1 = cl.rotor(np.array([0, 0, 1.0]), math.pi / 3)
    R2 = cl.rotor(np.array([1.0, 0, 0]), math.pi / 4)
    seq = cl.rotate(R2, cl.rotate(R1, v))              # apply R1 then R2
    comp = cl.rotate(cl.compose(R2, R1), v)            # compose (R2 left-applied last), apply once
    assert np.max(np.abs(seq - comp)) < 1e-12          # composition is EXACT
    # a rotor application is length-preserving and exactly invertible
    assert abs(np.linalg.norm(cl.rotate(R1, v)) - np.linalg.norm(v)) < 1e-12
    back = cl.rotate(cl.rotor(np.array([0, 0, 1.0]), -math.pi / 3), cl.rotate(R1, v))
    assert np.max(np.abs(back - v)) < 1e-12


def test_fpe_designed_kernel_is_continuous_and_beats_random_atoms():
    # Regime: continuous values. Win: a DESIGNED kernel makes nearby values similar (smooth monotone falloff),
    # where independent random atoms have no continuity.
    from holographic_fpe import VectorFunctionEncoder
    fpe = VectorFunctionEncoder(1, dim=1024, bounds=[(0, 3)], kernel="rbf", seed=0)
    xs = np.linspace(0, 3, 7)
    base = fpe.encode([0.0])
    sims = [float(cosine(base, fpe.encode([x]))) for x in xs]
    assert sims[0] > 0.99                              # peak at the encoded value
    assert all(sims[i] >= sims[i + 1] - 1e-9 for i in range(len(sims) - 1))  # monotone falloff
    # a unit-offset FPE code is still clearly similar, where a random atom for the same offset is ~orthogonal
    rand = cosine(random_vector(1024, np.random.default_rng(1)), random_vector(1024, np.random.default_rng(2)))
    assert sims[2] > 0.4 > abs(rand)                   # offset ~1.0 stays similar; random has no continuity


def test_tensor_bind_higher_recall_at_overloading_load():
    # Regime: high capacity at the cost of D^2 storage. Win: at a load that overloads HRR's convolution, the
    # tensor-product memory still recalls cleanly.
    from holographic_tensor import TensorBindMemory
    rng = np.random.default_rng(0)
    D, npairs = 32, 12
    keys = [random_vector(D, rng) for _ in range(npairs)]
    vals = [random_vector(D, rng) for _ in range(npairs)]
    hrr = bundle([bind(k, v) for k, v in zip(keys, vals)])
    hrr_rec = float(np.mean([cosine(unbind(hrr, keys[i]), vals[i]) for i in range(npairs)]))
    tm = TensorBindMemory(keys, vals)
    tens_rec = float(np.mean([cosine(tm.recall(keys[i]), vals[i]) for i in range(npairs)]))
    assert tens_rec > hrr_rec + 0.3                    # the capacity win at this overloading load
    assert tens_rec > 0.7                              # tensor recall is clean where HRR has collapsed


def test_base_kernel_stays_minimal_and_unchanged_by_extensions():
    # The standing rule: extensions are opt-in and the base kernel is untouched. Importing/using the extensions
    # above does not change the base ops -- they still conform to their definitional references.
    import holographic_clifford, holographic_fpe, holographic_tensor  # noqa: F401  (import the extensions)
    report = run_conformance(dim=64, seed=0)
    for op, r in report.items():
        assert r["passed"], f"{op} base conformance broke: {r}"
