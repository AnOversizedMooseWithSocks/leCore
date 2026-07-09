"""A2 (Walk on Stars) + A4/D1 (stateless coordinate-keyed randomness).

The two are one story: WoS is trivially parallel ONLY if its randomness needs no seed coordination, and that is what
`hash_unit` provides -- the random value is a pure function of WHERE you are and WHICH walk you're on, so any node
computes any walk in any order and gets the same answer. A stateful `default_rng` cannot do that.
"""
import numpy as np
import pytest

from holographic.misc.holographic_determinism import hash_unit, hash_direction, hash_u64
from holographic.simulation_and_physics.holographic_wost import solve_laplace


# --------------------------------------------------------------------------------------------------------------
# A4/D1 -- the stateless hash
# --------------------------------------------------------------------------------------------------------------
def test_hash_unit_is_a_pure_function():
    assert hash_unit(1, 2, 3) == hash_unit(1, 2, 3)
    assert hash_unit(1, 2) != hash_unit(2, 1)              # order matters (not a plain xor)
    assert hash_unit("a", 1) != hash_unit("b", 1)          # string domain separators work


def test_hash_unit_keys_floats_by_bit_pattern():
    assert hash_unit(0.1) != hash_unit(0.100000000000001)


def test_hash_unit_is_uniform_and_decorrelated():
    v = hash_unit(np.arange(100_000), 7)
    assert 0.495 < v.mean() < 0.505
    assert abs(v.std() - 1 / np.sqrt(12)) < 0.005
    assert 0.0 <= v.min() and v.max() < 1.0
    a, b = hash_unit(np.arange(50_000), 0), hash_unit(np.arange(50_000) + 1, 0)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.01            # neighbouring keys decorrelate


def test_hash_direction_is_uniform_on_the_sphere():
    d = hash_direction(np.arange(40_000), 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.linalg.norm(d.mean(0)) < 0.02              # no preferred direction
    assert abs(d[:, 2].std() - 1 / np.sqrt(3)) < 0.02    # z uniform in [-1,1] (equal area, not pole-clustered)
    c = hash_direction(np.arange(20_000), dim=2)
    assert np.allclose(np.linalg.norm(c, axis=1), 1.0)


def test_hash_rejects_unsupported_dim():
    with pytest.raises(ValueError):
        hash_direction(1, dim=5)


def test_hash_u64_is_deterministic_across_processes():
    """The point of the whole exercise: no PYTHONHASHSEED dependence, so two nodes agree. (Python's salted hash()
    would differ per process; this is pure integer arithmetic.)"""
    import subprocess
    import sys
    code = "from holographic.misc.holographic_determinism import hash_unit; print(repr(hash_unit('walk', 3, 7, 0.5)))"
    outs = set()
    for seed in ("0", "12345"):
        env = {"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}
        import os
        env["PYTHONPATH"] = os.getcwd()
        outs.add(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout.strip())
    assert len(outs) == 1, outs


# --------------------------------------------------------------------------------------------------------------
# A2 -- Walk on Spheres / Stars
# --------------------------------------------------------------------------------------------------------------
def _disk_sdf(P):
    return np.linalg.norm(np.atleast_2d(P), axis=1) - 1.0


def _u_is_x(P):
    return np.atleast_2d(P)[:, 0]


def _interior_points(n=40, r=0.8, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    rad = r * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([rad * np.cos(a), rad * np.sin(a)], 1)


def test_recovers_a_harmonic_function_and_converges_as_one_over_sqrt_n():
    pts = _interior_points()
    truth = pts[:, 0]
    e64 = np.sqrt(np.mean((solve_laplace(_disk_sdf, pts, _u_is_x, walks=64, seed=0, dim=2) - truth) ** 2))
    e1024 = np.sqrt(np.mean((solve_laplace(_disk_sdf, pts, _u_is_x, walks=1024, seed=0, dim=2) - truth) ** 2))
    assert e1024 < 0.03
    assert e1024 < 0.55 * e64                            # 16x walks -> ~4x less error


def test_result_is_bit_identical_and_independent_of_point_order():
    """This is what stateless hashing buys: farm-parallel with no seed coordination. A stateful rng fails this."""
    pts = _interior_points(20)
    a = solve_laplace(_disk_sdf, pts, _u_is_x, walks=32, seed=0, dim=2)
    assert np.array_equal(a, solve_laplace(_disk_sdf, pts, _u_is_x, walks=32, seed=0, dim=2))
    perm = np.random.default_rng(1).permutation(len(pts))
    assert np.allclose(solve_laplace(_disk_sdf, pts[perm], _u_is_x, walks=32, seed=0, dim=2), a[perm])


def test_poisson_source_term():
    """laplacian(u) = -1 with u = -|x|^2/4 on the boundary -> u(0) = 0."""
    u = solve_laplace(_disk_sdf, np.array([[0.0, 0.0]]),
                      lambda P: -(np.linalg.norm(np.atleast_2d(P), axis=1) ** 2) / 4.0,
                      walks=2048, seed=1, dim=2, source=lambda P: np.ones(len(np.atleast_2d(P))))
    assert abs(float(u[0])) < 0.06


def test_neumann_wall_is_reflected_not_read():
    """The discriminating test. Upper half-disk: u(x,y)=x is harmonic AND has zero flux across the flat edge, so the
    exact answer is u=x. POISON the flat edge with u=99. A correct reflecting walk never reads it; a merely
    absorbing one does, and is wrong by ~50. Without this poison the test would pass either way."""
    def sdf_half(P):
        P = np.atleast_2d(P)
        return np.maximum(np.linalg.norm(P, axis=1) - 1.0, -P[:, 1])

    def arc_only(P):                                     # the ABSORBING part of the boundary
        return np.linalg.norm(np.atleast_2d(P), axis=1) - 1.0

    def poisoned(P):
        P = np.atleast_2d(P)
        v = P[:, 0].copy()
        flat = (np.abs(P[:, 1]) < 5e-3) & (np.linalg.norm(P, axis=1) < 0.97)
        v[flat] = 99.0
        return v

    rng = np.random.default_rng(0)
    a = rng.uniform(0.2, np.pi - 0.2, 30)
    rad = 0.75 * np.sqrt(rng.uniform(0.05, 1, 30))
    pts = np.stack([rad * np.cos(a), rad * np.sin(a)], 1)
    truth = pts[:, 0]

    wost = solve_laplace(sdf_half, pts, poisoned, walks=1024, seed=0, dim=2, dirichlet_sdf=arc_only, max_steps=128)
    vanilla = solve_laplace(sdf_half, pts, poisoned, walks=1024, seed=0, dim=2)
    assert np.sqrt(np.mean((wost - truth) ** 2)) < 0.05
    assert np.sqrt(np.mean((vanilla - truth) ** 2)) > 10.0


def test_solves_in_3d_on_a_real_sdf_through_the_mind():
    """The whole point: leCore is SDF-native, so the solver needs no mesh. u(x,y,z)=x is harmonic in the ball."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    ball = sphere(1.0)
    pts = np.array([[0.0, 0.0, 0.0], [0.3, 0.1, -0.2], [-0.4, 0.2, 0.3]])
    u = m.solve_laplace(ball.eval, pts, lambda P: np.atleast_2d(P)[:, 0], walks=512, seed=0, dim=3)
    assert np.max(np.abs(u - pts[:, 0])) < 0.12          # Monte-Carlo error at 512 walks


# --------------------------------------------------------------------------------------------------------------
# A4/D1 remainder -- brdf.sample_ggx can draw its uniforms statelessly, which is what makes a path trace
# farm-parallel. (mis/traverse were probed and are NOT clients: their only default_rng use is inside _selftest.)
# --------------------------------------------------------------------------------------------------------------
def _ggx_setup(n=8000):
    N = np.tile([0.0, 0.0, 1.0], (n, 1))
    V = np.tile([0.0, 0.5, 0.866], (n, 1))
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return N, V, np.arange(n)


def test_sample_ggx_stateless_matches_stateful_statistically():
    from holographic.rendering.holographic_brdf import sample_ggx
    N, V, idx = _ggx_setup()
    Lh, ph = sample_ggx(N, V, 0.4, keys=(idx, 0, 7))
    Lr, pr = sample_ggx(N, V, 0.4, rng=np.random.default_rng(0))
    assert abs(Lh[:, 2].mean() - Lr[:, 2].mean()) < 0.02      # same estimator, different randomness source
    assert abs(ph.mean() - pr.mean()) < 0.05
    assert np.allclose(np.linalg.norm(Lh, axis=1), 1.0, atol=1e-6)


def test_sample_ggx_stateless_is_order_invariant_and_stateful_is_not():
    """The property the farm needs: trace pixels in any order, get the same image."""
    from holographic.rendering.holographic_brdf import sample_ggx
    N, V, idx = _ggx_setup(2000)
    perm = np.random.default_rng(1).permutation(len(idx))
    Lh, _ = sample_ggx(N, V, 0.4, keys=(idx, 0, 7))
    Lh_perm, _ = sample_ggx(N[perm], V[perm], 0.4, keys=(idx[perm], 0, 7))
    assert np.allclose(Lh_perm, Lh[perm])                     # stateless: order cannot matter

    Lr, _ = sample_ggx(N, V, 0.4, rng=np.random.default_rng(0))
    Lr_perm, _ = sample_ggx(N[perm], V[perm], 0.4, rng=np.random.default_rng(0))
    assert not np.allclose(Lr_perm, Lr[perm])                 # stateful: order changes the samples


def test_sample_ggx_requires_exactly_one_randomness_source():
    from holographic.rendering.holographic_brdf import sample_ggx
    N, V, idx = _ggx_setup(4)
    with pytest.raises(ValueError):
        sample_ggx(N, V, 0.4)
    with pytest.raises(ValueError):
        sample_ggx(N, V, 0.4, rng=np.random.default_rng(0), keys=(idx,))
