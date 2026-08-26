"""PBD/XPBD softbody + shape-matching rigid body (holographic_softbody). The constraint sweep is the same
iterate-a-projection engine the resonator/denoiser/IK use; this module adds the dynamics (momentum, mass,
gravity, time-step-independent XPBD stiffness, collision). Tests pin the physical invariants."""

import numpy as np

from holographic.simulation_and_physics.holographic_softbody import SoftBody, RigidBody


def test_distance_constraint_converges():
    b = SoftBody(np.array([[0.0, 0.0], [3.0, 0.0]]))         # gap 3, rest 1
    b.add_distance(0, 1, rest=1.0); b.pin(0)
    b.step(dt=1 / 60, gravity=(0.0, 0.0), iterations=30)
    assert b.constraint_residual() < 1e-6


def test_xpbd_stiffness_is_timestep_independent():
    # the XPBD headline: static stretch = compliance*weight, the same for 1 vs 6 substeps once settled
    def stretch(substeps):
        s = SoftBody(np.array([[0.0, 0.0], [0.0, -1.0]]))
        s.add_distance(0, 1, rest=1.0, compliance=0.01); s.pin(0)
        for _ in range(500):
            s.step(dt=1 / 60, gravity=(0.0, -9.8), iterations=20, substeps=substeps, damping=0.02)
        return float(np.linalg.norm(s.x[0] - s.x[1]) - 1.0)
    st1, st6 = stretch(1), stretch(6)
    assert abs(st1 - 0.098) < 0.005 and abs(st1 - st6) < 0.005


def test_pin_is_immovable():
    s = SoftBody(np.array([[0.0, 0.0], [0.0, -1.0]])); s.pin(0)
    top0 = s.x[0].copy()
    for _ in range(50):
        s.step(dt=1 / 60, gravity=(0.0, -9.8), iterations=10)
    assert np.allclose(s.x[0], top0)                         # pinned particle never moves


def test_cloth_reaches_equilibrium():
    cloth = SoftBody.cloth(5, 5, spacing=1.0, compliance=0.0)
    for _ in range(150):
        cloth.step(dt=1 / 60, gravity=(0.0, -9.8), iterations=25)
    assert cloth.constraint_residual() < 0.05


def test_stable_at_large_dt():
    big = SoftBody.cloth(5, 5, spacing=1.0, compliance=0.0)
    for _ in range(60):
        big.step(dt=0.1, gravity=(0.0, -9.8), iterations=15)   # dt that explodes an explicit spring
    assert np.isfinite(big.x).all() and np.abs(big.x).max() < 1e3


def test_pbd_via_shipped_sweeper():
    # the PBD path delegates the constraint sweep to holographic_denoise.project_onto_constraints
    p = SoftBody(np.array([[0.0, 0.0], [3.0, 0.0]]))
    p.add_distance(0, 1, rest=1.0); p.pin(0)
    p.step(dt=1 / 60, gravity=(0.0, 0.0), iterations=40, solver="pbd")
    assert p.constraint_residual() < 1e-3


def test_rigid_body_stays_rigid_and_falls():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rb = RigidBody(square)
    for _ in range(120):
        rb.step(dt=1 / 60, gravity=(0.0, -9.8))
    assert rb.max_distance_drift() < 1e-6                    # never deforms
    assert rb.x[:, 1].mean() < 0.0                           # falls under gravity


def test_softbody_pushed_by_a_field_force():
    # VSA coupling: an attractor force from the fields layer pulls a free particle toward a point
    from holographic.misc.holographic_fields import attractor_force
    s = SoftBody(np.array([[10.0, 0.0], [11.0, 0.0]]))
    s.add_distance(0, 1, rest=1.0)                           # a little 2-particle body, nothing pinned
    c0 = s.x.mean(axis=0).copy()
    for _ in range(60):
        f = attractor_force(s.x, (0.0, 0.0), strength=20.0)
        s.step(dt=1 / 60, gravity=(0.0, 0.0), external_force=f, iterations=10, damping=0.02)
    c1 = s.x.mean(axis=0)
    assert np.linalg.norm(c1) < np.linalg.norm(c0)           # the body moved toward the attractor
    assert s.constraint_residual() < 0.05                    # and stayed intact


def test_bending_resists_folding():
    import math
    def fold(with_bending):
        s = SoftBody(np.array([[-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]]))
        s.add_distance(0, 1, 1.0); s.add_distance(1, 2, 1.0); s.pin(1)
        if with_bending:
            s.add_bending(0, 2)
        th = 0.7
        s.x[0] = [-math.cos(th), math.sin(th), 0]; s.x[2] = [math.cos(th), math.sin(th), 0]
        for _ in range(60):
            s.step(dt=1 / 60, gravity=(0, 0, 0), iterations=20)
        return float(np.linalg.norm(s.x[0] - s.x[2]))
    assert fold(False) < 1.6          # distance constraints alone leave the fold
    assert fold(True) > 1.95          # the bend spring flattens it back toward 2.0


def test_volume_constraint_restores_squashed_tet():
    tet = SoftBody(np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    tet.add_volume(0, 1, 2, 3); v0 = tet.total_volume()
    tet.x[3, 2] = 0.3                 # squash the apex
    tet.step(dt=1 / 60, gravity=(0, 0, 0), iterations=40)
    assert abs(tet.total_volume() - v0) < 0.02


def test_soft_box_preserves_volume_under_compression():
    box = SoftBody.soft_box(3, 3, 3, spacing=1.0, compliance=0.0, volume_compliance=0.0)
    v0 = box.total_volume()
    box.x[box.x[:, 1] > 1.5, 1] -= 0.4   # squash the top layer down
    box.step(dt=1 / 60, gravity=(0, 0, 0), iterations=40)
    assert abs(box.total_volume() - v0) < 0.05 * v0


def test_softbody_couples_to_3d_fluid_via_drag():
    """A softbody drifts with a 3-D flow when driven by drag_force_3d as external_force -- full parity with 2-D."""
    import numpy as np
    from holographic.simulation_and_physics.holographic_softbody import SoftBody
    from holographic.misc.holographic_fields import drag_force_3d
    N = 20; flow = np.ones((N, N, N)) * 1.5; zero = np.zeros((N, N, N))
    body = SoftBody(np.array([[5., 12., 12.], [6., 12., 12.], [7., 12., 12.]]))
    body.add_distance(0, 1); body.add_distance(1, 2)
    x0 = body.x[:, 0].mean()
    for _ in range(40):
        drag = drag_force_3d(body.x, body.v, flow, zero, zero, k=2.0)
        body.step(dt=1 / 60, gravity=(0, 0, 0), external_force=drag)
    assert body.x[:, 0].mean() > x0 + 0.2                      # drifted downstream
    assert body.constraint_residual() < 1e-6                   # strip stayed intact


def test_self_collision_rests_at_radius_no_coast():
    """Two overlapping non-bonded nodes separate to exactly the collision radius and stay there (the positional
    contact resolve injects no coasting velocity)."""
    import numpy as np
    from holographic.simulation_and_physics.holographic_softbody import SoftBody
    body = SoftBody(np.array([[0., 0., 0.], [0.3, 0., 0.]]))
    body.add_self_collision(radius=1.0)
    for _ in range(10):
        body.step(dt=1 / 60, gravity=(0, 0, 0))
    sep = float(np.linalg.norm(body.x[0] - body.x[1]))
    assert abs(sep - 1.0) < 0.05 and np.abs(body.v).max() < 1e-9


def test_self_collision_spreads_overlapping_nodes_and_excludes_bonds():
    """Self-collision pushes a clump of non-bonded nodes apart to the radius; bonded nodes are excluded so the
    structure isn't fought."""
    import numpy as np
    from holographic.simulation_and_physics.holographic_softbody import SoftBody
    pts = np.array([[0., 0., 0.], [0.1, 0., 0.], [0., 0.1, 0.], [0., 0., 0.1], [0.1, 0.1, 0.]])
    def min_gap(collide):
        b = SoftBody(pts.copy())
        if collide:
            b.add_self_collision(radius=1.0)
        for _ in range(40):
            b.step(dt=1 / 60, gravity=(0, 0, 0))
        return min(float(np.linalg.norm(b.x[i] - b.x[j])) for i in range(5) for j in range(i + 1, 5))
    assert min_gap(False) < 0.5 and min_gap(True) > 0.9        # collision separates the clump

    bonded = SoftBody(np.array([[0., 0., 0.], [0.3, 0., 0.]]))
    bonded.add_distance(0, 1); bonded.add_self_collision(radius=1.0)
    for _ in range(10):
        bonded.step(dt=1 / 60, gravity=(0, 0, 0))
    assert abs(float(np.linalg.norm(bonded.x[0] - bonded.x[1])) - 0.3) < 0.05   # bond preserved


def test_softbody_from_mesh_simulates_a_projected_mesh():
    """A projected mesh becomes a SoftBody (verts->particles, edges->constraints) and is driven by the physics
    -- here fluid drag carries it while it stays connected -- so the mesh pipeline uses what the physics added."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.simulation_and_physics.holographic_softbody import SoftBody
    import holographic.misc.holographic_fields as F
    m = box()
    body = SoftBody.from_mesh(m)
    assert body.N == m.n_vertices and len(body.constraints) == m.n_edges
    flow = np.ones((16, 16, 16)) * 1.0; z = np.zeros((16, 16, 16))
    body.x += 8.0                                              # move into the grid
    x0 = float(body.x[:, 0].mean())
    for _ in range(20):
        body.step(dt=1 / 60, gravity=(0, 0, 0), external_force=F.drag_force_3d(body.x, body.v, flow, z, z, k=2.0))
    assert float(body.x[:, 0].mean()) > x0                     # the projected mesh rode the fluid


def test_softbody_and_rigidbody_export_to_mesh():
    """A soft/rigid body built from a mesh retains its faces and re-exports its current (deformed/moved) state."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.simulation_and_physics.holographic_softbody import SoftBody, RigidBody
    sb = SoftBody.from_mesh(box())
    sb.x[:, 2] += 0.5
    msb = sb.to_mesh()
    assert msb.n_vertices == box().n_vertices and msb.n_faces == box().n_faces
    rb = RigidBody.from_mesh(box())
    rb.x += np.array([2.0, 0.0, 0.0])
    mrb = rb.to_mesh()
    assert np.allclose(mrb.vertices.mean(0)[0], box().vertices.mean(0)[0] + 2.0)
