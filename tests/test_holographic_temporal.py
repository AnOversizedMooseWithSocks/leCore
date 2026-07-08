"""Sweep 3 item 5: the temporal-reuse loop re-solves only the dirty region and matches a full re-solve."""
import numpy as np
from holographic.simulation_and_physics.holographic_temporal import TemporalReuse
from holographic.misc.holographic_backwardwarp import backward_gather


def test_reuse_dirty_only_matches_full():
    n = 200; scene = np.sin(np.linspace(0, 6, n))
    solver = lambda sc: (lambda i: float(sc[i] * sc[i] + 0.5 * sc[i]))
    tr = TemporalReuse()
    _, c0 = tr.solve(solver(scene), n)
    assert c0 == n
    scene2 = scene.copy(); dirty = [10, 11, 99, 150]; scene2[dirty] += 0.3
    frame1, c1 = tr.solve(solver(scene2), n, dirty=dirty)
    assert c1 == len(dirty) and c1 < n // 10
    full = np.array([solver(scene2)(i) for i in range(n)])
    assert np.allclose(frame1, full, atol=1e-12)


def test_reproject_hole_free():
    n = 100; pos = np.linspace(0, 1, n); scene = np.cos(np.linspace(0, 5, n))
    solver = lambda i: float(scene[i])
    tr = TemporalReuse(); tr.solve(solver, n)
    reproj = lambda f: backward_gather(f, pos, np.clip(pos + 0.02, 0, 1))
    out, _ = tr.solve(solver, n, dirty=[0, 1], reproject=reproj)
    assert out.shape == (n,) and not np.isnan(out).any()


def test_accumulation_converges():
    rng = np.random.default_rng(0); tr = TemporalReuse()
    tr.solve(lambda i: 2.0 + rng.standard_normal(), 1)
    for _ in range(300):
        tr.solve(lambda i: 2.0 + rng.standard_normal(), 1, dirty=[0], accumulate=True)
    assert abs(tr.frame[0] - 2.0) < 0.2
