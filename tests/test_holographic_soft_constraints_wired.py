"""C3 -- `soft_relaxation` plumbed into its two callers, behind the gate that lets it be safe.

THE GATE, and every caller must pass it before the dial is plumbed through: `stiffness=(inf, zeta)` must be
**bit-identical** to the rigid `omega=1.0` default. Nothing changes unless you ask for it.

WHY IT IS NOT `omega`. `omega` is a per-sweep number, so the same dial means different physics at different
iteration counts. `stiffness=(hertz, zeta)` is Catto's Soft Step parameterization in physical units, substep-
invariant to first order. Measured at the IK call site, at a fixed physical horizon:

    iters   omega=0.30 reach error   stiffness=(8 Hz, 1.0) reach error
        5           0.4253                     0.0033
       20           0.0314                     0.0002
       80           0.0000                     0.0001

The stiffness dial holds its meaning; omega does not.
"""

import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_meshik import solve_ik
from holographic.rendering.holographic_denoise import soft_relaxation
from holographic.simulation_and_physics.holographic_softbody import RigidBody, SoftBody


CHAIN = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
TARGET = np.array([1.5, 1.5, 0.0])


def _stretched():
    """Bones at 2.5 apart with a rest length of 1.0, root pinned: the projections have real work to do."""
    sb = SoftBody(np.array([[0.0, 0, 0], [2.5, 0, 0], [5.0, 0, 0]]))
    sb.add_distance(0, 1, 1.0)
    sb.add_distance(1, 2, 1.0)
    sb.w[0] = 0.0
    return sb


def _bones(sb):
    return [float(np.linalg.norm(sb.x[i + 1] - sb.x[i])) for i in range(len(sb.x) - 1)]


# ---------------------------------------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------------------------------------

def test_the_gate_infinite_stiffness_is_bit_identical_at_the_ik_site():
    rigid, n1 = solve_ik(CHAIN, TARGET)
    gated, n2 = solve_ik(CHAIN, TARGET, stiffness=(np.inf, 1.0), dt=1 / 60.0)
    assert np.array_equal(rigid, gated) and n1 == n2      # bit-for-bit, not "to 1e-12"


def test_the_gate_infinite_stiffness_is_bit_identical_at_the_softbody_pbd_site():
    a, b = _stretched(), _stretched()
    a.step(dt=1 / 60.0, gravity=np.zeros(3), solver="pbd", iterations=20)
    b.step(dt=1 / 60.0, gravity=np.zeros(3), solver="pbd", iterations=20, stiffness=(np.inf, 1.0))
    assert np.array_equal(a.x, b.x) and np.array_equal(a.v, b.v)


def test_soft_relaxation_itself_still_pins_its_limits():
    assert soft_relaxation(np.inf, 1.0, 1 / 60.0) == 1.0   # the hard projection, exactly
    assert soft_relaxation(0.0, 1.0, 1 / 60.0) == 0.0      # zero stiffness: inert
    assert soft_relaxation(5.0, 0.0, 1 / 60.0) == 1.0      # zeta=0 degenerates to hard, never to ringing


# ---------------------------------------------------------------------------------------------------------
# the dial is a dial
# ---------------------------------------------------------------------------------------------------------

def test_a_soft_ik_chain_lags_its_target_and_stiffer_lags_less():
    reach = []
    for hz in (2.0, 8.0, 40.0):
        j, _ = solve_ik(CHAIN, TARGET, stiffness=(hz, 1.0), dt=1 / 60.0)
        reach.append(float(np.linalg.norm(j[-1] - TARGET)))
    assert reach == sorted(reach, reverse=True)            # stiffer => closer
    assert reach[0] > 0.1 and reach[-1] < 1e-3

    rigid, _ = solve_ik(CHAIN, TARGET)
    assert float(np.linalg.norm(rigid[-1] - TARGET)) < 1e-9   # rigid reaches exactly


def test_a_soft_pbd_bone_stretches_and_stiffer_stretches_less():
    lengths = {}
    for hz in (2.0, 20.0, np.inf):
        sb = _stretched()
        sb.step(dt=1 / 60.0, gravity=np.zeros(3), solver="pbd", iterations=20, stiffness=(hz, 1.0))
        lengths[hz] = _bones(sb)[0]
    assert lengths[2.0] > lengths[20.0] > lengths[np.inf]
    # PBD is ITERATIVE, not exact: 20 sweeps from a 2.5x stretch leave a 2.9e-6 residual. Asserting 1e-6 here
    # would be asserting the iteration count, not the physics.
    assert lengths[np.inf] == pytest.approx(1.0, abs=1e-4)  # rigid: rest length met, to the solver's tolerance
    assert lengths[2.0] > 1.5                               # soft: still stretched, as a spring should be


def test_the_dial_holds_its_meaning_where_omega_does_not():
    # THE POINT of the whole parameterization. At a FIXED physical horizon, vary the iteration count.
    from holographic.mesh_and_geometry.holographic_meshik import _chain_projections
    from holographic.rendering.holographic_denoise import project_onto_constraints

    lengths = [1.0, 1.0, 1.0]
    projs = _chain_projections(lengths, CHAIN[0].copy(), TARGET)
    omega_reach, stiff_reach = [], []
    for iters in (5, 20, 80):
        a, _, _ = project_onto_constraints(CHAIN.ravel(), projs, iters=iters, omega=0.30)
        b, _, _ = project_onto_constraints(CHAIN.ravel(), projs, iters=iters,
                                           stiffness=(8.0, 1.0), dt=1.0 / iters)
        omega_reach.append(float(np.linalg.norm(a.reshape(-1, 3)[-1] - TARGET)))
        stiff_reach.append(float(np.linalg.norm(b.reshape(-1, 3)[-1] - TARGET)))

    assert omega_reach[0] - omega_reach[-1] > 0.4          # omega drifts wildly with the iteration count
    assert max(stiff_reach) - min(stiff_reach) < 0.01      # the stiffness dial barely moves


# ---------------------------------------------------------------------------------------------------------
# scope + the name collision that once fooled the lint
# ---------------------------------------------------------------------------------------------------------

def test_the_xpbd_path_ignores_stiffness_because_compliance_already_is_it():
    # XPBD carries physical compliance per constraint -- the same idea, already there. Passing `stiffness` to it
    # must not silently change anything.
    a, b = _stretched(), _stretched()
    a.step(dt=1 / 60.0, gravity=np.zeros(3), solver="xpbd", iterations=20)
    b.step(dt=1 / 60.0, gravity=np.zeros(3), solver="xpbd", iterations=20, stiffness=(2.0, 1.0))
    assert np.array_equal(a.x, b.x)


def test_stiffness_is_default_off_everywhere():
    a, b = _stretched(), _stretched()
    a.step(dt=1 / 60.0, gravity=np.array([0.0, -9.81, 0.0]), solver="pbd")
    b.step(dt=1 / 60.0, gravity=np.array([0.0, -9.81, 0.0]), solver="pbd", stiffness=None)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(solve_ik(CHAIN, TARGET)[0], solve_ik(CHAIN, TARGET, stiffness=None)[0])


def test_rigidbody_stiffness_is_an_unrelated_scalar():
    # The name collision that once fooled the adoption lint into reporting softbody as WIRED. Different class,
    # different meaning: a scalar spring constant, not (hertz, zeta).
    rb = RigidBody(np.array([[0.0, 1.0, 0.0]]))
    rb.step(dt=1 / 60.0, gravity=np.array([0.0, -9.81, 0.0]), stiffness=1.0)   # a float, not a tuple
    assert np.isfinite(rb.x).all()


def test_the_lint_records_both_clients_as_wired_on_a_precise_symbol():
    import os
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "tools"))
    from unifiers import PENDING, REGISTRY, cites

    key = "denoise.soft_relaxation (stiffness in physical units)"
    assert "stiffness=stiffness" in REGISTRY[key]["symbols"]      # the pass-through, not the bare parameter name
    assert "stiffness=" not in REGISTRY[key]["symbols"]           # ... which would match RigidBody, and lie
    assert cites("holographic_meshik", key, repo)
    assert cites("holographic_softbody", key, repo)
    assert not any(u == key for u, _c in PENDING)                 # 2/2 wired
