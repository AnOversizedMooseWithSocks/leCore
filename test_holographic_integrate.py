"""Phase 0 / G2: the shared time-integrator + uniform SimStep interface."""
import numpy as np
from holographic_integrate import (semi_implicit_euler, explicit_euler, verlet,
                                    SimStep, SolverAdapter, ParticleSim)


def _energy(p, v):
    return 0.5 * float(np.sum(v * v)) + 0.5 * float(np.sum(p * p))


def test_symplectic_beats_explicit_on_orbit():
    central = lambda pos, vel: -1.0 * pos
    p0 = np.array([[1.0, 0.0, 0.0]]); v0 = np.array([[0.0, 1.0, 0.0]])
    dt = 0.05
    sym = ParticleSim(p0.copy(), v0.copy(), central, integrator="symplectic")
    exp = ParticleSim(p0.copy(), v0.copy(), central, integrator="explicit")
    e0 = _energy(sym.pos, sym.vel)
    for _ in range(2000):
        sym.advance(dt); exp.advance(dt)
    drift_sym = abs(_energy(sym.pos, sym.vel) - e0) / e0
    drift_exp = abs(_energy(exp.pos, exp.vel) - e0) / e0
    assert drift_sym < drift_exp and drift_sym < 0.05


def test_verlet_free_fall_matches_analytic():
    g = np.array([0.0, -9.8, 0.0]); pos = np.array([0.0, 0.0, 0.0]); prev = pos.copy(); dt = 0.001
    for _ in range(1000):
        pos, prev = verlet(pos, prev, g, dt)
    assert abs(pos[1] - 0.5 * (-9.8)) < 0.05


def test_semi_implicit_vs_explicit_one_step():
    p = np.array([0.0]); v = np.array([1.0]); f = np.array([2.0]); dt = 0.5
    ps, vs = semi_implicit_euler(p, v, f, dt)      # v first: v=1+1=2, p=0+0.5*2=1.0
    pe, ve = explicit_euler(p, v, f, dt)           # p from old v: p=0+0.5*1=0.5
    assert abs(vs[0] - 2.0) < 1e-9 and abs(ps[0] - 1.0) < 1e-9
    assert abs(pe[0] - 0.5) < 1e-9


def test_solver_adapter_wraps_step():
    class _Fake:
        def __init__(self): self.t = 0.0
        def weird(self, a): self.t += a; return self.t
    f = _Fake()
    ad = SolverAdapter(f, lambda solver, dt, ctx: solver.weird(dt) or ctx)
    ad.advance(0.5, None); ad.advance(0.5, None)
    assert abs(f.t - 1.0) < 1e-9
    assert isinstance(ad, SimStep)


def test_deterministic():
    central = lambda pos, vel: -pos
    a = ParticleSim(np.array([[1.0, 0, 0]]), np.array([[0.0, 1, 0]]), central)
    b = ParticleSim(np.array([[1.0, 0, 0]]), np.array([[0.0, 1, 0]]), central)
    for _ in range(50):
        a.advance(0.05); b.advance(0.05)
    assert np.array_equal(a.pos, b.pos)
