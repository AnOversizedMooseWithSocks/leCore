"""X4 -- speculative margins + conservative advancement (Box3D lesson B4), and two bugs the work surfaced.

The backlog's bar: "tunnelling test suite passes without a dedicated CCD pass." It does, because conservative
advancement's core query (how far can I move without hitting anything?) IS the SDF value -- so CCD is sphere
tracing, and it delegates to the renderer's `raymarch.sphere_trace`.

Also pinned here:
  * KEPT NEGATIVE: a speculative margin DETECTS proximity, it does not PREVENT tunnelling (wrong side).
  * BUG FIX: `resolve_sdf_collision` silently left medial-axis points inside the collider (zero gradient).
  * BUG FIX: `sdf_normal` / `sphere_trace` crashed on a bare callable; the engine has two SDF conventions.
"""

import numpy as np
import pytest

from holographic.simulation_and_physics.holographic_collide import (
    resolve_sdf_collision, sdf_collision_projection, sdf_offset, time_of_impact, advance_ccd, _escape_direction)
from holographic.mesh_and_geometry.holographic_sdf import as_eval, sdf_normal
from holographic.rendering.holographic_raymarch import sphere_trace


WALL = lambda P: np.abs(np.asarray(P, float)[:, 0]) - 0.05      # a 0.1 m thick slab at x = 0
SPHERE = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0


def test_selftest_runs():
    from holographic.simulation_and_physics import holographic_collide as mod
    mod._selftest()


# -- the premise: discrete resolution really does tunnel -----------------------------------------------------

def test_discrete_resolution_tunnels_through_a_thin_wall():
    # 30 m/s at dt=1/60 => 0.5 m per step, across a 0.1 m wall. Chosen so NEITHER endpoint is inside:
    # sdf(-0.30) = +0.25 and sdf(+0.20) = +0.15. A point sample can never see the interior.
    X = np.array([[-0.30, 0.0, 0.0]])
    V = np.array([[30.0, 0.0, 0.0]])
    dt = 1 / 60.0
    landed = X + V * dt
    assert float(WALL(X)[0]) > 0 and float(WALL(landed)[0]) > 0     # both endpoints outside
    assert np.allclose(resolve_sdf_collision(landed, WALL), landed)  # nothing to resolve: it tunnelled


# -- the speculative margin: free, but not a tunnelling fix --------------------------------------------------

def test_speculative_margin_is_one_subtraction():
    off = sdf_offset(WALL, 0.2)
    P = np.array([[1.0, 0.0, 0.0]])
    assert abs(float(off(P)[0]) - (float(WALL(P)[0]) - 0.2)) < 1e-15
    shrunk = sdf_offset(WALL, -0.1)                                  # negative margin shrinks the collider
    assert abs(float(shrunk(P)[0]) - (float(WALL(P)[0]) + 0.1)) < 1e-15


def test_kept_negative_a_margin_detects_but_does_not_prevent_tunnelling():
    # The body came from x = -0.30 and landed at x = +0.20, already through. A 0.2 margin DOES fire (0.15 < 0.2)
    # ... and pushes it to +0.25: further along its direction of travel, out the WRONG side of the wall.
    landed = np.array([[0.20, 0.0, 0.0]])
    pushed = resolve_sdf_collision(landed, WALL, radius=0.2)
    assert pushed[0, 0] > landed[0, 0]                               # wrong side, not back where it came from
    assert float(WALL(pushed)[0]) >= 0.2 - 1e-9                      # it IS resolved, just to the wrong place


# -- conservative advancement: the actual fix, and it is sphere tracing ---------------------------------------

def test_time_of_impact_is_exact_and_needs_no_margin():
    X = np.array([[-0.30, 0.0, 0.0]])
    V = np.array([[30.0, 0.0, 0.0]])
    dt = 1 / 60.0
    hit, toi, contact = time_of_impact(X, V, dt, WALL)
    assert hit[0]
    assert abs(float(contact[0, 0]) + 0.05) < 1e-3                   # stopped on the NEAR face
    assert abs(float(toi[0]) - 0.25 / 30.0) < 1e-6                   # exact time of first contact
    assert 0.0 <= float(toi[0]) <= dt


def test_time_of_impact_misses_report_the_full_step():
    X = np.array([[-5.0, 0.0, 0.0]])                                 # far away; one step gets nowhere near
    V = np.array([[1.0, 0.0, 0.0]])
    dt = 1 / 60.0
    hit, toi, contact = time_of_impact(X, V, dt, WALL)
    assert not hit[0]
    assert abs(float(toi[0]) - dt) < 1e-15
    assert np.allclose(contact, X + V * dt)                          # the unobstructed landing point


def test_time_of_impact_ignores_stationary_points():
    X = np.array([[-0.30, 0.0, 0.0], [0.5, 0.0, 0.0]])
    V = np.array([[30.0, 0.0, 0.0], [0.0, 0.0, 0.0]])                # the second point never moves
    hit, toi, contact = time_of_impact(X, V, 1 / 60.0, WALL)
    assert hit[0] and not hit[1]
    assert np.allclose(contact[1], X[1])


def test_a_hit_beyond_this_points_sweep_is_not_a_hit_this_step():
    # Two points share one scalar max_dist inside sphere_trace; the slow one must not inherit the fast one's reach.
    X = np.array([[-0.30, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    V = np.array([[30.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    hit, toi, _ = time_of_impact(X, V, 1 / 60.0, WALL)
    assert hit[0] and not hit[1]                                     # the far point sweeps 0.5 m, wall is 2.95 m away
    assert abs(float(toi[1]) - 1 / 60.0) < 1e-15


def test_advance_ccd_stops_exactly_on_the_surface():
    X = np.array([[-0.30, 0.0, 0.0]])
    V = np.array([[30.0, 0.0, 0.0]])
    Xn, Vn, hit = advance_ccd(X, V, 1 / 60.0, WALL)
    assert hit[0]
    assert float(WALL(Xn)[0]) >= -1e-3                               # on the surface, not through it
    assert abs(float(Vn[0, 0])) < 1e-6                               # into-surface velocity cancelled


def test_advance_ccd_bounces_with_restitution():
    X = np.array([[-0.30, 0.0, 0.0]])
    V = np.array([[30.0, 0.0, 0.0]])
    _, Vn, hit = advance_ccd(X, V, 1 / 60.0, WALL, restitution=1.0)
    assert hit[0]
    assert float(Vn[0, 0]) < -1.0                                    # perfectly elastic: it reverses
    assert abs(abs(float(Vn[0, 0])) - 30.0) < 1e-3                   # ... and keeps its speed


def test_advance_ccd_leaves_free_points_alone():
    X = np.array([[-5.0, 0.0, 0.0]])
    V = np.array([[1.0, 0.0, 0.0]])
    Xn, Vn, hit = advance_ccd(X, V, 1 / 60.0, WALL)
    assert not hit[0]
    assert np.allclose(Xn, X + V / 60.0) and np.allclose(Vn, V)      # full step, untouched velocity


def test_ccd_survives_a_sweep_of_speeds_the_discrete_resolve_fails():
    # THE BAR: no tunnelling at any speed, with no dedicated CCD pass and no margin tuning.
    dt = 1 / 60.0
    for speed in (5.0, 30.0, 200.0, 5000.0):
        X = np.array([[-0.30, 0.0, 0.0]])
        V = np.array([[speed, 0.0, 0.0]])
        Xn, _, hit = advance_ccd(X, V, dt, WALL)
        if speed * dt > 0.25:                                        # fast enough to reach the wall this step
            assert hit[0], speed
            assert float(WALL(Xn)[0]) >= -1e-3, (speed, Xn)


# -- BUG FIX: the medial axis ---------------------------------------------------------------------------------

def test_medial_axis_points_are_no_longer_left_inside():
    # On the medial axis the SDF gradient is EXACTLY zero (central differences cancel), so the old code moved the
    # point by (radius - d) * 0 == 0 and reported success while leaving it inside the collider.
    mid = np.array([[0.0, 0.0, 0.0]])
    assert float(WALL(mid)[0]) < 0                                   # it really is inside
    assert np.allclose(sdf_normal(WALL, mid), 0.0)                   # ... and there really is no normal
    out = resolve_sdf_collision(mid, WALL)
    assert float(WALL(out)[0]) >= -1e-9, "a medial-axis point must not be left inside"


def test_the_medial_axis_escape_is_deterministic():
    # Both +x and -x are equally correct at the slab centre. The tie-break is lowest axis, then positive sign --
    # arbitrary, but the SAME every run, which is the engine's determinism rule.
    mid = np.array([[0.0, 0.0, 0.0]])
    outs = [resolve_sdf_collision(mid, WALL) for _ in range(5)]
    for o in outs:
        assert np.array_equal(o, outs[0])
    assert abs(float(outs[0][0, 0]) - 0.05) < 1e-9                   # +x wins the tie

    d = _escape_direction(WALL, mid, np.array([0.05]))
    assert np.array_equal(d[0], np.array([1.0, 0.0, 0.0]))


def test_well_defined_normals_are_bit_identical_after_the_fix():
    # The medial-axis branch must not perturb the ordinary path at all.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3)) * 0.4                               # a cloud inside the unit sphere
    out = resolve_sdf_collision(X, SPHERE, radius=0.0)
    assert (SPHERE(out) >= -1e-6).all()
    # recompute the ordinary formula by hand and demand bit-equality
    from holographic.simulation_and_physics.holographic_emitter import _sdf_normal
    d = SPHERE(X)
    inside = d < 0.0
    manual = X.copy()
    manual[inside] = X[inside] + (0.0 - d[inside])[:, None] * _sdf_normal(SPHERE, X[inside], 1e-3)
    assert np.array_equal(out, manual)


def test_the_projection_wrapper_still_satisfies_link_plus_collision():
    # The X2 soft-constraint engine still drives collision as one more projection -- unchanged.
    from holographic.rendering.holographic_denoise import project_onto_constraints
    N, D = 2, 3
    x0 = np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]).ravel()
    coll = sdf_collision_projection(SPHERE, N, D, radius=0.0)
    out, _, _ = project_onto_constraints(x0, [coll], iters=30)
    assert (SPHERE(out.reshape(N, D)) >= -0.02).all()


# -- BUG FIX: the engine's two SDF conventions ----------------------------------------------------------------

def test_as_eval_adapts_both_sdf_conventions():
    class _Node:
        eval = staticmethod(lambda P: np.linalg.norm(P, axis=1) - 1.0)

    P = np.array([[2.0, 0.0, 0.0]])
    assert abs(float(as_eval(_Node())(P)[0]) - 1.0) < 1e-12          # node object with .eval
    assert abs(float(as_eval(SPHERE)(P)[0]) - 1.0) < 1e-12           # bare callable


def test_sdf_normal_and_sphere_trace_accept_a_bare_callable():
    # Both used to require `.eval` and raised AttributeError on a lambda -- which is why holographic_sdf_render
    # carried a throwaway `_Obj()` wrapper. One adapter, no shims.
    assert np.allclose(sdf_normal(WALL, np.array([[0.5, 0.0, 0.0]])), [[1.0, 0.0, 0.0]])
    hit, t, _ = sphere_trace(WALL, np.array([[-1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), max_dist=5.0)
    assert hit[0] and abs(float(t[0]) - 0.95) < 1e-3


def test_ccd_is_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    X = np.array([[-0.30, 0.0, 0.0]])
    V = np.array([[30.0, 0.0, 0.0]])
    hit, toi, contact = m.time_of_impact(X, V, 1 / 60.0, WALL)
    assert hit[0] and abs(float(contact[0, 0]) + 0.05) < 1e-3
    Xn, _, h = m.advance_ccd(X, V, 1 / 60.0, WALL)
    assert h[0] and float(WALL(Xn)[0]) >= -1e-3
    off = m.sdf_offset(WALL, 0.1)
    assert abs(float(off(np.array([[1.0, 0, 0]]))[0]) - 0.85) < 1e-12
    assert "Tunnelling" in str(m.find_capability("stop a fast bullet going through a thin wall")[:3])


def test_ccd_and_the_speculative_margin_compose():
    # radius= gives the swept point a thickness: it stops `radius` short of the surface. That is the speculative
    # margin doing its real job -- a contact band -- on top of CCD doing the tunnelling job.
    X = np.array([[-0.30, 0.0, 0.0]])
    V = np.array([[30.0, 0.0, 0.0]])
    _, _, c0 = time_of_impact(X, V, 1 / 60.0, WALL, radius=0.0)
    _, _, c1 = time_of_impact(X, V, 1 / 60.0, WALL, radius=0.1)
    assert c1[0, 0] < c0[0, 0] - 0.05                                # stopped earlier, by roughly the margin
    assert abs(float(WALL(c1)[0]) - 0.1) < 1e-2


def test_as_eval_accepts_a_dsl_string_so_an_agent_can_call_ccd():
    # A callable cannot cross a JSON boundary -- the live-handle lesson. The s-expression can, and `parse_dsl`
    # already ships, so every SDF consumer becomes agent-invokable for free.
    f = as_eval("(sphere 1.0)")
    assert abs(float(f(np.array([[2.0, 0.0, 0.0]]))[0]) - 1.0) < 1e-12

    X = np.array([[-3.0, 0.0, 0.0]])
    V = np.array([[120.0, 0.0, 0.0]])
    hit, toi, contact = time_of_impact(X, V, 1 / 60.0, "(sphere 1.0)")
    assert hit[0] and np.allclose(contact[0], [-1.0, 0.0, 0.0], atol=1e-3)

    Xn, _, h = advance_ccd(X, V, 1 / 60.0, "(sphere 1.0)")
    assert h[0] and np.allclose(Xn[0], [-1.0, 0.0, 0.0], atol=1e-3)

    # and the discrete resolve takes the string form too
    out = resolve_sdf_collision(np.array([[0.2, 0.0, 0.0]]), "(sphere 1.0)")
    assert np.linalg.norm(out[0]) >= 1.0 - 1e-6


def test_the_string_callable_and_node_forms_agree_exactly():
    from holographic.mesh_and_geometry.holographic_sdf import sphere as sdf_sphere
    P = np.array([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    a = as_eval("(sphere 1.0)")(P)
    b = as_eval(sdf_sphere(1.0))(P)
    c = as_eval(SPHERE)(P)
    assert np.allclose(a, b) and np.allclose(b, c)
