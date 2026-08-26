"""C2: settle-gate soundness, MEASURED -- including the boundary where it is unsound.

The documented failure mode (DAAREM et al.): a solver can STAGNATE for many iterations with
little-to-no change before converging rapidly later, so a gate watching "has it stopped
moving" fires on the plateau. leCore's run_until_settled already makes the SOTA-correct
choice -- it watches a RESIDUAL stream and gates it through convergence_guard's i.i.d. check
-- and that buys real robustness. But no finite window can survive an arbitrarily long
plateau, and this pins where the boundary actually is.
"""
import numpy as np


def _plateau_sim(plateau_len):
    """decay -> flat plateau -> decay resumes. A gate that settles during the plateau is
    WRONG, because the system demonstrably moves again afterwards."""
    def step(s):
        x, t = float(s[0]), float(s[1])
        v = -0.05 * x if t < 40 else (0.0 if t < 40 + plateau_len else -0.08 * x)
        return np.array([x + v, t + 1.0])
    return np.array([1.0, 0.0]), step


def _residual(a, b):
    return float(abs(np.asarray(b)[0] - np.asarray(a)[0]))


def test_gate_survives_plateaus_up_to_its_window():
    """MEASURED: a plateau at or below the window does NOT trigger a false settle -- the
    guard reports 'never settled: every frame honestly simulated' rather than serving frames
    from a fake equilibrium."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    for plateau in (60, 96):
        st, step = _plateau_sim(plateau)
        out = m.run_until_settled(step, st, steps=500, residual=_residual, window=96)
        assert out["settle_step"] is None, (plateau, out["why"])


def test_a_longer_plateau_defeats_it_and_a_bigger_window_pushes_the_boundary():
    """The KEPT NEGATIVE, pinned so it is read as a property rather than rediscovered as a
    bug: no finite window survives an arbitrarily long stagnation. Measured -- window 96 is
    defeated at plateau 128; window 192 survives 160 and is defeated at 220. The window IS
    the trap length, and that is the honest way to describe the gate's guarantee."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    st, step = _plateau_sim(128)
    out = m.run_until_settled(step, st, steps=500, residual=_residual, window=96)
    assert out["settle_step"] is not None and out["settle_step"] < 40 + 128
    st, step = _plateau_sim(160)
    wide = m.run_until_settled(step, st, steps=500, residual=_residual, window=192)
    assert wide["settle_step"] is None      # the same plateau is survived by a wider window


def test_lyapunov_witness_certifies_only_genuine_gradient_flows():
    """C2's fix, and it is a THEOREM rather than a bigger window: for a true gradient flow a
    state plateau means grad E ~ 0 -- a critical point, which cannot resume -- so the
    stagnation trap is impossible. The certificate therefore checks the PRECONDITION.
    Pinned across four families: a real leCore relax() run certifies; the driven plateau that
    defeated the window gate does not; a still-falling run does not; a RISING witness does
    not (that is not a descent flow at all)."""
    import numpy as np
    from lecore import UnifiedMind
    from holographic.simulation_and_physics.holographic_morphogen import relax
    m = UnifiedMind(dim=64, seed=0)
    X = np.random.default_rng(0).normal(scale=1.5, size=(30, 3))
    _, hist = relax(X, np.full(30, 0.5), steps=300)
    res = [abs(hist[i + 1] - hist[i]) for i in range(len(hist) - 1)]
    c = m.lyapunov_certify(hist, res)
    assert c["certified"] and c["monotone"] and c["settled"]
    t = np.arange(400)
    driven = np.where(t < 40, 1.0 - 0.02 * t,
                      np.where(t < 200, 0.2, 0.2 - 0.005 * (t - 200)))
    assert not m.lyapunov_certify(driven)["certified"]
    assert not m.lyapunov_certify(np.exp(-np.arange(200) * 0.002))["certified"]
    rising = m.lyapunov_certify(np.arange(50, dtype=float))
    assert not rising["certified"] and not rising["monotone"]
