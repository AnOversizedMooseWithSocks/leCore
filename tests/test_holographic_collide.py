"""Environment (SDF) collision as a projection: keep particles/cloth outside scene geometry, via the unified sweep."""
import numpy as np
from holographic.simulation_and_physics.holographic_collide import resolve_sdf_collision, sdf_collision_projection


def test_points_pushed_out_of_collider():
    R = 1.0
    sphere = lambda P: np.linalg.norm(P, axis=1) - R
    X = np.array([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [2.0, 0.0, 0.0]])     # 2 inside, 1 outside
    Xc = resolve_sdf_collision(X, sphere, radius=0.0)
    assert (sphere(Xc) >= -1e-6).all()                                    # nobody inside
    assert np.allclose(Xc[2], X[2])                                       # outside point untouched


def test_collision_in_unified_projection_sweep():
    """The collision projection co-satisfies with a distance link inside the SAME project_onto_constraints engine."""
    from holographic.rendering.holographic_denoise import project_onto_constraints
    R = 1.0; sphere = lambda P: np.linalg.norm(P, axis=1) - R
    N, D = 2, 3
    x0 = np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]).ravel(); dist = 2.4
    def link(flat):
        Xr = flat.reshape(N, D); n = Xr[0] - Xr[1]; d = np.linalg.norm(n)
        if d < 1e-9:
            return flat
        n = n / d; c = d - dist; Xn = Xr.copy(); Xn[0] = Xr[0] - 0.5 * c * n; Xn[1] = Xr[1] + 0.5 * c * n
        return Xn.ravel()
    out, _, _ = project_onto_constraints(x0, [link, sdf_collision_projection(sphere, N, D)], iters=60)
    Xf = out.reshape(N, D)
    assert (sphere(Xf) >= -0.02).all()
    assert abs(np.linalg.norm(Xf[0] - Xf[1]) - dist) < 0.15


def test_cloth_drapes_over_sphere_no_penetration():
    from holographic.simulation_and_physics.holographic_softbody import SoftBody
    rows = cols = 14; sp = 0.16
    cloth = SoftBody.cloth3d(rows=rows, cols=cols, spacing=sp, compliance=1e-6)
    cloth.x[:, 0] -= cols * sp / 2; cloth.x[:, 2] -= rows * sp / 2; cloth.x[:, 1] += 1.1
    for c in [0, cols - 1, (rows - 1) * cols, rows * cols - 1]:
        cloth.pin(c)
    R = 0.8; sphere = lambda P: np.linalg.norm(P, axis=1) - R
    for _ in range(70):
        cloth.step(dt=1 / 60.0, gravity=(0, -9.8, 0), iterations=18, collider=sphere, collide_radius=0.02)
    d = sphere(cloth.x)
    assert d.min() >= -0.02                                               # no penetration into the sphere
    assert d.min() < 0.15                                                 # the cloth actually reached and rests on it
