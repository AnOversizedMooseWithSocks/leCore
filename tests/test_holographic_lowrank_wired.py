"""B2 -- LowRankField wired to its real client, behind an ERROR budget rather than an energy criterion.

EVERY FIELD IN THIS SUITE IS REAL. No synthetic `np.outer(a, b)`, which is rank 1 by construction and would make
`worth_factoring` pass for the wrong reason. The fields are SDF slices from the shipped DSL, an analytic 1/r^2
irradiance field, fbm noise from the shipped pattern module, and white noise. Where a ground truth exists, the tests
compare against the ANALYTIC function, not against a resampled grid.

THE DEFECT THIS REPAIRS. `rank_gate` chooses a rank by ENERGY -- the fewest singular values carrying 99% of the
variance. **99% of the energy is not a small error.** Measured on real 128x128 fields:

    field                 gate rank (99% energy)   max|err| there   rank for 1% max-abs error
    sphere SDF slice              2                    7.45%                    4
    box SDF slice                 2                   18.19%                   12
    fbm noise (4 octaves)         5                   28.54%                   50
    white noise                  99                       --                  124

An SDF wrong by 7% of its amplitude does not sphere-trace. So `rank_for_error` sizes on max-abs error, and
`worth_factoring(X, max_error=...)` is the gate a consumer of the RECONSTRUCTION must pass.
"""

import numpy as np
import pytest

from holographic.caching_and_storage.holographic_tucker import LowRankField, rank_for_error, rank_gate
from holographic.mesh_and_geometry.holographic_sdf import parse_dsl, to_callable
from holographic.misc.holographic_fieldhome import Field, field_backends
from holographic.misc.holographic_pattern import fbm

N = 128
_G = np.linspace(-2.0, 2.0, N)
_XX, _YY = np.meshgrid(_G, _G, indexing="ij")
_P3 = np.stack([_XX.ravel(), _YY.ravel(), np.zeros(N * N)], axis=1)
LO, HI = np.array([-2.0, -2.0]), np.array([2.0, 2.0])


def _sdf_slice(expr):
    return to_callable(parse_dsl(expr))(_P3).reshape(N, N)


def sphere_sdf():
    """A REAL field: the z=0 slice of the shipped `(sphere 1.0)` SDF. Smooth, radially symmetric, NOT separable."""
    return _sdf_slice("(sphere 1.0)")


def box_sdf():
    """A REAL field with creases: the box SDF's corners are where low-rank structure goes to die."""
    return _sdf_slice("(box 1 1 1)")


def irradiance():
    """A REAL smooth field: 1/r^2 falloff from two point lights. What a light cache actually stores."""
    return sum(1.0 / (0.5 + ((_XX - l[0]) ** 2 + (_YY - l[1]) ** 2))
               for l in (np.array([1.5, 1.5]), np.array([-1.0, 0.5])))


def fbm_noise():
    """A REAL not-low-rank field from the shipped pattern module. Structured enough to fool an energy gate."""
    return np.asarray(fbm(scale=2.0, octaves=4, seed=0)(_P3)).reshape(N, N)


def white_noise():
    return np.random.default_rng(0).normal(size=(N, N))


def _amp(X):
    return float(np.abs(X).max())


def _max_err(X, r):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return float(np.abs((U[:, :r] * s[:r]) @ Vt[:r] - X).max())


# ---------------------------------------------------------------------------------------------------------
# the data is real -- assert that before trusting anything measured on it
# ---------------------------------------------------------------------------------------------------------

def test_the_test_fields_are_genuinely_two_dimensional_and_not_separable():
    # A rank-1 outer product would make every gate pass for the wrong reason. Guard against writing one by accident.
    for name, X in (("sphere", sphere_sdf()), ("box", box_sdf()), ("irradiance", irradiance()), ("fbm", fbm_noise())):
        assert not np.allclose(X, X[:, :1]), name          # not constant in y
        assert not np.allclose(X, X[:1, :]), name          # not constant in x
        s = np.linalg.svd(X, compute_uv=False)
        assert s[1] / s[0] > 1e-6, (name, "rank-1: this field is a synthetic outer product, not real data")

    # The SDF at a grid node must equal the ANALYTIC value AT THAT NODE. NB `linspace(-2, 2, 128)` has no sample
    # at exactly 0 -- the nearest node sits 0.0157 off-centre -- so asserting sdf == -1.0 at [N//2, N//2] fails
    # by 0.022 for a field that is perfectly correct. Bad coordinates make a good test fail: use the real node.
    i = j = N // 2
    node = np.array([_G[i], _G[j]])
    analytic = float(np.linalg.norm(node) - 1.0)
    assert abs(sphere_sdf()[i, j] - analytic) < 1e-12
    assert analytic < -0.97                                   # ... and it really is the centre of the sphere

    assert fbm_noise().std() > 0.05                          # the noise actually varies


# ---------------------------------------------------------------------------------------------------------
# THE DEFECT: energy is not error
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_the_energy_gate_leaves_a_large_residual_on_real_fields():
    for name, X, min_rel_err in (("sphere", sphere_sdf(), 0.05), ("box", box_sdf(), 0.15)):
        r = int(max(rank_gate(X)[0]))
        rel = _max_err(X, r) / _amp(X)
        assert rel > min_rel_err, (name, r, rel)            # 99% energy, and still 7-18% wrong


def test_kept_negative_the_energy_gate_blesses_fbm_noise_that_the_error_gate_refuses():
    X = fbm_noise()
    amp = _amp(X)
    energy_ok, _fb, _db = LowRankField.worth_factoring(X)
    assert energy_ok is True                                # the energy gate says "compress it"
    r = int(max(rank_gate(X)[0]))
    assert _max_err(X, r) / amp > 0.2                       # ... at a 20%+ reconstruction error

    # the error gate needs 10x the rank, at which point the saving has nearly evaporated
    r_err = rank_for_error(X, 0.01 * amp)
    assert r_err > 8 * r
    _ok, fb, db = LowRankField.worth_factoring(X, max_error=0.01 * amp)
    assert fb > 0.7 * db                                    # 1.27x: marginal, and not worth an SVD


def test_rank_for_error_is_the_smallest_rank_meeting_the_budget():
    X = sphere_sdf()
    amp = _amp(X)
    r = rank_for_error(X, 0.01 * amp)
    assert r == 4                                           # measured on this field
    assert _max_err(X, r) <= 0.01 * amp
    assert _max_err(X, r - 1) > 0.01 * amp                  # ... and it is the SMALLEST such rank


def test_rank_for_error_refuses_a_non_2d_field():
    with pytest.raises(ValueError):
        rank_for_error(np.zeros((4, 4, 4)), 0.1)


def test_the_error_gate_pays_on_smooth_fields_and_refuses_noise():
    for name, X, pays in (("sphere", sphere_sdf(), True), ("box", box_sdf(), True),
                          ("irradiance", irradiance(), True), ("white", white_noise(), False)):
        ok, fb, db = LowRankField.worth_factoring(X, max_error=0.01 * _amp(X))
        assert ok is pays, (name, fb, db)

    # and the smooth ones pay HANDSOMELY: 16x on a sphere SDF
    _ok, fb, db = LowRankField.worth_factoring(sphere_sdf(), max_error=0.01 * _amp(sphere_sdf()))
    assert db / fb > 10


def test_from_dense_with_max_error_meets_the_budget_it_was_given():
    for X in (sphere_sdf(), box_sdf(), irradiance()):
        amp = _amp(X)
        f = LowRankField.from_dense(X, max_error=0.01 * amp)
        assert float(np.abs(f.to_dense() - X).max()) <= 0.01 * amp
        assert f.rank == rank_for_error(X, 0.01 * amp)


# ---------------------------------------------------------------------------------------------------------
# the wired client: Field.low_rank
# ---------------------------------------------------------------------------------------------------------

def test_low_rank_is_a_registered_field_backend():
    assert "low_rank" in field_backends()


def test_the_factored_field_samples_exactly_at_grid_nodes():
    X = sphere_sdf()
    amp = _amp(X)
    f = Field.low_rank(X, LO, HI, max_error=0.01 * amp)
    assert f.kind == "low_rank" and f.backend.rank == 4

    ii, jj = np.meshgrid(np.arange(0, N, 7), np.arange(0, N, 11), indexing="ij")
    pts = np.stack([LO[0] + ii.ravel() / N * (HI[0] - LO[0]),
                    LO[1] + jj.ravel() / N * (HI[1] - LO[1])], axis=1)
    recon = f.backend.to_dense()[ii.ravel(), jj.ravel()]
    assert np.abs(f.sample(pts) - recon).max() < 1e-12       # the sampler reproduces the factorization exactly


def test_the_reconstruction_tracks_the_analytic_truth_not_a_resampled_grid():
    # GROUND TRUTH: the analytic sphere SDF. Comparing against a resampled grid would hide the grid's own error.
    X = sphere_sdf()
    amp = _amp(X)
    f = Field.low_rank(X, LO, HI, max_error=0.01 * amp)

    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.9, 1.9, size=(2000, 2))
    truth = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) - 1.0
    rel = float(np.abs(f.sample(pts) - truth).max()) / amp
    assert rel < 0.05                                        # measured 3.39%, dominated by the 0.031 grid spacing

    # the DENSE grid, sampled bilinearly, is not much better -- the gap is the SAMPLER, not the factorization
    rel_dense = float(np.abs(Field.grid(X, LO, HI).sample(pts) - truth).max()) / amp
    assert rel_dense < rel < rel_dense + 0.02


def test_low_rank_refuses_a_field_that_is_not_low_rank():
    with pytest.raises(ValueError, match="not low rank"):
        Field.low_rank(white_noise(), LO, HI, max_error=0.01 * _amp(white_noise()))


def test_low_rank_refuses_a_3d_grid():
    with pytest.raises(ValueError):
        Field.low_rank(np.zeros((8, 8, 8)), np.zeros(3), np.ones(3), max_error=1.0)


def test_the_factored_field_stores_far_fewer_bytes():
    X = sphere_sdf()
    f = Field.low_rank(X, LO, HI, max_error=0.01 * _amp(X))
    assert f.backend.nbytes() * 10 < X.nbytes                # measured 8,224 vs 131,072 -- 16x


# ---------------------------------------------------------------------------------------------------------
# the registry records both measured verdicts
# ---------------------------------------------------------------------------------------------------------

def test_the_registry_wires_fieldhome_and_defers_postfx_with_its_measurement():
    import os
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "tools"))
    from unifiers import DEFERRED, NOT_APPLICABLE, PENDING, REGISTRY, cites

    key = "tucker.LowRankField (compressed-domain compute)"
    assert REGISTRY[key]["clients"] == ["holographic_fieldhome"]
    assert cites("holographic_fieldhome", key, repo)
    assert not any(u == key for u, _c in PENDING)            # 1/1 wired

    # postfx is DEFERRED (the construction exists, it does not pay), NOT "impossible"
    assert (key, "holographic_postfx") in DEFERRED
    assert (key, "holographic_postfx") not in NOT_APPLICABLE
    assert "53.7x" in DEFERRED[(key, "holographic_postfx")]  # a verdict must carry its evidence
