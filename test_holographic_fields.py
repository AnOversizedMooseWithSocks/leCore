"""Grid fields + particle simulation exposed to VSA (holographic_fields). The fluid steps run on the same
FFT-on-a-torus the bind operator uses: diffusion is a Gaussian bind, the pressure projection is an FFT
Helmholtz solve. Tests pin the physical invariants -- mass conservation, divergence-free projection, transport
-- and the particle/force behaviour."""

import numpy as np

from holographic_fields import (diffuse, divergence, curl, project_divergence_free, advect, fluid_step,
                                ParticleSystem, attractor_force, sample_field)

H = W = 32


def _blob(cx, cy, s=4.0):
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2)))


def test_diffuse_conserves_mass_and_smooths():
    f = _blob(16, 16)
    d = diffuse(f, amount=3.0)
    assert abs(f.mean() - d.mean()) < 1e-9          # DC preserved -> mass conserved
    assert d.var() < f.var()                        # high frequencies damped -> smoother


def test_projection_removes_divergence():
    rng = np.random.default_rng(0)
    vx = rng.normal(size=(H, W)); vy = rng.normal(size=(H, W))
    before = np.abs(divergence(vx, vy)).max()
    px, py = project_divergence_free(vx, vy)
    after = np.abs(divergence(px, py)).max()
    assert before > 1.0
    assert after < 1e-9                             # incompressible to machine precision


def test_projection_is_idempotent():
    rng = np.random.default_rng(1)
    vx = rng.normal(size=(H, W)); vy = rng.normal(size=(H, W))
    px, py = project_divergence_free(vx, vy)
    qx, qy = project_divergence_free(px, py)        # projecting an already-divergence-free field is a no-op
    assert np.max(np.abs(px - qx)) < 1e-9 and np.max(np.abs(py - qy)) < 1e-9


def test_curl_of_gradient_is_zero():
    # a pure gradient flow (vx = d/dx phi, vy = d/dy phi) has zero curl
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    phi = np.sin(2 * np.pi * X / W) * np.cos(2 * np.pi * Y / H)
    vx = np.gradient(phi, axis=1); vy = np.gradient(phi, axis=0)
    assert np.abs(curl(vx, vy)).max() < 0.5         # small (finite-difference gradient, not exact spectral)


def test_advect_moves_blob_by_v_dt():
    dens = _blob(8, 16)
    U = np.full((H, W), 3.0); V = np.zeros((H, W))
    moved = advect(dens, U, V, dt=2.0)              # expect a shift of +6 in x
    cx0 = (np.arange(W)[None, :] * dens).sum() / dens.sum()
    cx1 = (np.arange(W)[None, :] * moved).sum() / moved.sum()
    assert 5.0 < (cx1 - cx0) < 7.0


def test_fluid_step_stays_incompressible():
    vx = np.zeros((H, W)); vy = np.zeros((H, W)); density = _blob(16, 16)
    fx = _blob(8, 16) * 5.0
    for _ in range(5):
        vx, vy, density = fluid_step(vx, vy, density, dt=0.2, viscosity=0.05, fx=fx)
    assert np.abs(divergence(vx, vy)).max() < 1e-3  # the projection keeps the flow incompressible each step


def test_attractor_pulls_particles_inward():
    rng = np.random.default_rng(2)
    ps = ParticleSystem(rng.uniform(6, 26, size=(150, 2)))
    d0 = np.linalg.norm(ps.pos - np.array([16, 16]), axis=1).mean()
    for _ in range(30):
        ps.step(force=attractor_force(ps.pos, (16, 16), strength=6.0), dt=0.1, damping=0.05)
    d1 = np.linalg.norm(ps.pos - np.array([16, 16]), axis=1).mean()
    assert d1 < d0


def test_particles_ride_velocity_field():
    ps = ParticleSystem(np.array([[8.0, 16.0]]))
    ps.advect_by(np.full((H, W), 4.0), np.zeros((H, W)), dt=1.0)
    assert ps.pos[0, 0] > 11.0                      # carried downstream by the flow


def test_sample_field_bilinear():
    f = _blob(16, 16, s=5.0)
    # sampling at the peak returns ~1.0; sampling far away returns ~0
    assert sample_field(f, np.array([[16.0, 16.0]]))[0] > 0.95
    assert sample_field(f, np.array([[2.0, 2.0]]))[0] < 0.2


def test_scatter_is_adjoint_of_sample():
    from holographic_fields import scatter_to_field, sample_field
    f = scatter_to_field((H, W), np.array([[16.0, 16.0]]), np.array([1.0]))
    assert abs(sample_field(f, np.array([[16.0, 16.0]]))[0] - 1.0) < 1e-9   # exact at a cell center
    assert abs(f.sum() - 1.0) < 1e-9                                         # mass-preserving scatter


def test_drag_force_pulls_particles_toward_the_flow():
    from holographic_fields import drag_force
    vx = np.full((H, W), 4.0); vy = np.zeros((H, W))
    pos = np.array([[10.0, 10.0], [12.0, 14.0]]); vel = np.zeros((2, 2))
    for _ in range(20):
        f = drag_force(pos, vel, vx, vy, k=0.5)
        vel = vel + (1 / 60) * f; pos = pos + (1 / 60) * vel
    assert vel[:, 0].mean() > 0.3 and abs(vel[:, 1].mean()) < 0.1            # accelerates along +x only


def _rows():
    return np.arange(H)[:, None]


def test_buoyancy_makes_hot_smoke_rise():
    from holographic_fields import smoke_step
    vx = np.zeros((H, W)); vy = np.zeros((H, W)); dens = _blob(16, 6); temp = _blob(16, 6)
    c0 = float((_rows() * dens).sum() / dens.sum())
    for _ in range(40):
        vx, vy, dens, temp = smoke_step(vx, vy, dens, temp, dt=0.2, viscosity=0.02, buoyancy=3.0)
    c1 = float((_rows() * dens).sum() / dens.sum())
    assert c1 > c0 + 2.0                                   # hot smoke rose (centroid moved up = +row)


def test_buoyancy_force_direction():
    from holographic_fields import buoyancy_force
    temp = np.ones((H, W)) * 2.0
    fx, fy = buoyancy_force(temp, beta=1.0, ambient=0.0)
    assert np.allclose(fx, 0.0) and np.all(fy > 0)         # hotter than ambient -> upward, no horizontal


def test_vorticity_confinement_preserves_curl():
    from holographic_fields import smoke_step, curl
    def run(conf):
        vx = np.zeros((H, W)); vy = np.zeros((H, W)); dens = _blob(16, 6); temp = _blob(16, 6)
        for _ in range(30):
            vx, vy, dens, temp = smoke_step(vx, vy, dens, temp, dt=0.2, viscosity=0.02,
                                            buoyancy=3.0, confinement=conf)
        return float(np.abs(curl(vx, vy)).sum())
    assert run(2.0) > run(0.0)                              # confinement keeps more vorticity


def test_smoke_source_builds_a_rising_plume():
    from holographic_fields import smoke_step
    vx = np.zeros((H, W)); vy = np.zeros((H, W)); dens = np.zeros((H, W)); temp = np.zeros((H, W))
    src = _blob(16, 4, s=2.5)
    for _ in range(50):
        vx, vy, dens, temp = smoke_step(vx, vy, dens, temp, dt=0.2, viscosity=0.02,
                                        buoyancy=4.0, confinement=1.0, dens_source=src, temp_source=src)
    centroid = float((_rows() * dens).sum() / dens.sum())
    assert dens.sum() > 0 and centroid > 6.0               # smoke accumulated and rose above the row-4 source


def test_enforce_solid_zeros_velocity_in_the_mask():
    from holographic_fields import enforce_solid, disc_mask
    rng = np.random.default_rng(0)
    vx = rng.normal(size=(H, W)); vy = rng.normal(size=(H, W))
    solid = disc_mask((H, W), center=(16, 16), radius=5)
    vx, vy = enforce_solid(vx, vy, solid, iters=4)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    assert speed[solid > 0].mean() < 0.25 * speed[solid == 0].mean()   # the solid is (nearly) still


def test_obstacle_blocks_and_diverts_flow():
    from holographic_fields import fluid_step, disc_mask
    solid = disc_mask((H, W), center=(16, 16), radius=5)
    vx = np.zeros((H, W)); vy = np.zeros((H, W)); dens = np.zeros((H, W))
    fx = np.ones((H, W)) * 2.0
    for _ in range(60):
        vx, vy, dens = fluid_step(vx, vy, dens, dt=0.15, viscosity=0.05, fx=fx, solid=solid)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ambient = speed[np.sqrt((X - 16) ** 2 + (Y - 16) ** 2) > 11].mean()
    assert speed[solid > 0].mean() < 0.2 * ambient                    # flow blocked inside the obstacle


def test_smoke_flows_around_an_obstacle():
    from holographic_fields import smoke_step, disc_mask
    obstacle = disc_mask((H, W), center=(16, 16), radius=4)
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    src = np.exp(-(((X - 16) ** 2 + (Y - 4) ** 2) / (2 * 2.5 ** 2)))
    vx = np.zeros((H, W)); vy = np.zeros((H, W)); d = np.zeros((H, W)); t = np.zeros((H, W))
    for _ in range(50):
        vx, vy, d, t = smoke_step(vx, vy, d, t, dt=0.2, viscosity=0.02, buoyancy=4.0,
                                  confinement=1.0, dens_source=src, temp_source=src, solid=obstacle)
    assert d[obstacle > 0].sum() < 1e-6 and d.sum() > 0               # no smoke in the solid; smoke went around


# --- 3-D fluid / smoke (the same operators on a 3-D periodic grid via the n-D real FFT) ---
N3 = 16


def _blob3(cx, cy, cz, s=3.0):
    X, Y, Z = np.meshgrid(np.arange(N3), np.arange(N3), np.arange(N3), indexing="ij")
    return np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * s ** 2)))


def test_diffuse_3d_conserves_mass_and_smooths():
    from holographic_fields import diffuse_3d
    f = _blob3(8, 8, 8); d = diffuse_3d(f, 2.0)
    assert abs(f.mean() - d.mean()) < 1e-9 and d.var() < f.var()


def test_projection_3d_removes_divergence():
    from holographic_fields import divergence_3d, project_divergence_free_3d
    rng = np.random.default_rng(0)
    vx = rng.normal(size=(N3, N3, N3)); vy = rng.normal(size=(N3, N3, N3)); vz = rng.normal(size=(N3, N3, N3))
    assert np.abs(divergence_3d(vx, vy, vz)).max() > 1.0
    px, py, pz = project_divergence_free_3d(vx, vy, vz)
    assert np.abs(divergence_3d(px, py, pz)).max() < 1e-9


def test_advect_3d_moves_blob_by_v_dt():
    from holographic_fields import advect_3d
    dens = _blob3(4, 8, 8)
    U = np.full((N3, N3, N3), 3.0); Z = np.zeros((N3, N3, N3))
    moved = advect_3d(dens, U, Z, Z, dt=2.0)
    ax = np.arange(N3)[:, None, None]
    cx0 = (ax * dens).sum() / dens.sum(); cx1 = (ax * moved).sum() / moved.sum()
    assert 5.0 < (cx1 - cx0) < 7.0


def test_fluid_step_3d_stays_incompressible():
    from holographic_fields import fluid_step_3d, divergence_3d
    vx = np.zeros((N3, N3, N3)); vy = np.zeros((N3, N3, N3)); vz = np.zeros((N3, N3, N3))
    density = _blob3(8, 8, 8); fx = _blob3(4, 8, 8) * 5.0
    for _ in range(4):
        vx, vy, vz, density = fluid_step_3d(vx, vy, vz, density, dt=0.2, viscosity=0.05, fx=fx)
    assert np.abs(divergence_3d(vx, vy, vz)).max() < 1e-3


def test_smoke_3d_rises():
    from holographic_fields import smoke_step_3d
    vx = np.zeros((N3, N3, N3)); vy = np.zeros((N3, N3, N3)); vz = np.zeros((N3, N3, N3))
    d = _blob3(8, 3, 8); t = _blob3(8, 3, 8)
    ay = np.arange(N3)[None, :, None]
    y0 = (ay * d).sum() / d.sum()
    for _ in range(30):
        vx, vy, vz, d, t = smoke_step_3d(vx, vy, vz, d, t, dt=0.2, viscosity=0.02, buoyancy=3.0)
    y1 = (ay * d).sum() / d.sum()
    assert y1 > y0 + 1.0                               # hot smoke rose in +y


# --- seamless fractal volume synthesis (the seamless source the 3-D torus gives tiling) ---

def test_spectral_field_tiles_seamlessly():
    from holographic_fields import spectral_field, seam_continuity
    vol = spectral_field((32, 32, 32), beta=2.5, seed=0)
    ramp = np.linspace(0, 1, 32)[:, None, None] * np.ones((32, 32, 32))
    assert seam_continuity(vol) < 2.0                  # periodic by construction -> no seam
    assert seam_continuity(ramp) > 5.0                 # a non-periodic field has a big seam


def test_spectral_field_beta_controls_roughness():
    from holographic_fields import spectral_field
    def rough(b):
        f = spectral_field((48, 48), beta=b, seed=1); gx, gy = np.gradient(f)
        return np.sqrt(gx ** 2 + gy ** 2).mean()
    assert rough(0.5) > rough(3.0)                      # higher beta = smoother (more low-frequency)


def test_spectral_field_is_deterministic_and_normalized():
    from holographic_fields import spectral_field
    a = spectral_field((16, 16, 16), beta=2.0, seed=7)
    b = spectral_field((16, 16, 16), beta=2.0, seed=7)
    assert np.array_equal(a, b)                         # a whole volume reproducible from (shape, beta, seed)
    assert abs(a.mean()) < 1e-9 and abs(a.std() - 1.0) < 1e-6


def test_spatial_hash_pairs_matches_brute_force():
    """The cull primitive returns exactly the close pairs a full O(N^2) scan would, in any dimension."""
    import numpy as np
    from holographic_fields import spatial_hash_pairs
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, size=(200, 3)); r = 1.0
    got = set(map(tuple, spatial_hash_pairs(pts, r).tolist()))
    want = {(i, j) for i in range(len(pts)) for j in range(i + 1, len(pts))
            if ((pts[i] - pts[j]) ** 2).sum() <= r * r}
    assert got == want


def test_sphere_mask_obstacle_blocks_3d_flow():
    """3-D immersed boundary: a ball forces the flow around it and density cannot enter (disc_mask lifted)."""
    import numpy as np
    from holographic_fields import sphere_mask, fluid_step_3d
    N = 20
    vx = np.ones((N, N, N)); vy = np.zeros((N, N, N)); vz = np.zeros((N, N, N)); dens = np.zeros((N, N, N))
    ball = sphere_mask((N, N, N), (10, 10, 10), 4)
    dens[2:5, 8:12, 8:12] = 1.0
    for _ in range(5):
        vx, vy, vz, dens = fluid_step_3d(vx, vy, vz, dens, dt=0.2, solid=ball)
    inside = float(np.abs(vx[ball > 0]).mean()); ambient = float(np.abs(vx[ball == 0]).mean())
    assert inside < 0.1 * ambient                              # flow diverted around the ball
    assert float(dens[ball > 0].sum()) < 1e-9                  # density blocked from the solid


def test_sample_scatter_3d_are_adjoint():
    """scatter_to_field_3d is the exact adjoint of sample_field_3d: <scatter(v), f> == <v, sample(f)>."""
    import numpy as np
    from holographic_fields import sample_field_3d, scatter_to_field_3d
    N = 16; rng = np.random.default_rng(1)
    field = rng.standard_normal((N, N, N)); pos = rng.uniform(1, N - 2, size=(25, 3)); vals = rng.standard_normal(25)
    lhs = float((scatter_to_field_3d((N, N, N), pos, vals) * field).sum())
    rhs = float((vals * sample_field_3d(field, pos)).sum())
    assert np.isclose(lhs, rhs)


def test_drag_force_3d_pushes_toward_flow():
    """drag_force_3d = k*(v_fluid - v_node) sampled trilinearly -- the fluid->body coupling half, in 3-D."""
    import numpy as np
    from holographic_fields import drag_force_3d
    N = 16; flow = np.ones((N, N, N)) * 2.0; zero = np.zeros((N, N, N))
    nodes = np.array([[5., 5., 5.], [6., 6., 6.]]); nvel = np.zeros((2, 3))
    f = drag_force_3d(nodes, nvel, flow, zero, zero, k=1.0)
    assert np.allclose(f[:, 0], 2.0) and np.allclose(f[:, 1:], 0.0)


def test_pairwise_repulsion_matches_brute_force():
    """The culled short-range force equals the O(N^2) all-pairs sum exactly -- the spatial hash changes the
    cost, not the answer."""
    import numpy as np
    from holographic_fields import pairwise_repulsion
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, size=(300, 2)); r = 0.8
    fast = pairwise_repulsion(pts, r, strength=2.0)
    brute = np.zeros_like(pts)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = pts[i] - pts[j]; dist = float(np.linalg.norm(d))
            if 1e-12 < dist < r:
                f = 2.0 * (1 - dist / r) * d / dist; brute[i] += f; brute[j] -= f
    assert np.allclose(fast, brute)


def test_pairwise_repulsion_disperses_a_clump():
    """Short-range repulsion drives a tight clump apart -- the min pairwise distance grows. (Directional check,
    not a brittle absolute.)"""
    import numpy as np
    from holographic_fields import pairwise_repulsion, ParticleSystem
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, size=(20, 2))                      # a tight clump
    ps = ParticleSystem(pts.copy())
    def min_gap(P):
        return min(float(np.linalg.norm(P[i] - P[j])) for i in range(len(P)) for j in range(i + 1, len(P)))
    g0 = min_gap(ps.pos)
    for _ in range(30):
        ps.step(force=pairwise_repulsion(ps.pos, radius=1.5, strength=1.0), dt=0.1, damping=0.3)
    assert min_gap(ps.pos) > g0                                # the clump spread out


def test_spatial_hash_pairs_vectorized_deterministic_and_sorted():
    """The vectorized cell-list returns the SAME pairs as brute force (2-D and 3-D), as a deterministic
    (i<j)-sorted array -- the contract the scatter-based collision/repulsion rely on."""
    import numpy as np
    from holographic_fields import spatial_hash_pairs
    for D in (2, 3):
        pts = np.random.default_rng(D).uniform(0, 8, size=(220, D)); r = 1.0
        got = spatial_hash_pairs(pts, r)
        want = {(i, j) for i in range(len(pts)) for j in range(i + 1, len(pts))
                if ((pts[i] - pts[j]) ** 2).sum() <= r * r}
        assert set(map(tuple, got.tolist())) == want
        assert np.array_equal(got, got[np.lexsort((got[:, 1], got[:, 0]))])   # (i,j)-sorted
        assert np.array_equal(got, spatial_hash_pairs(pts, r))                # deterministic
        assert np.all(got[:, 0] < got[:, 1])                                  # i<j canonical
