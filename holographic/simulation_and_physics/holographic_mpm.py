"""holographic_mpm.py -- SNOW via the Material Point Method (Physics & FX backlog, item #8B, rung 4, the LAST item).

Real basis: Stomakhin, Schroeder, Chai, Teran, Selle (2013), "A material point method for snow simulation"; the
compact MLS-MPM transfer of Hu, Fang, Ge, Qu, Tang, Jiang (2018). Snow is elasto-PLASTIC: it deforms elastically
up to a yield point, then YIELDS permanently (which is why a snowball packs and a footprint stays). MPM captures
that by carrying a deformation gradient F on each particle and clamping its singular values when it yields.

THINKING HOLOGRAPHICALLY (the honest insight, not a forced one):
MPM looks like a pure grid solver, but its HEART -- the particle<->grid transfer -- IS the engine's bundle/readout
wearing a physics costume. Walk it against the substrate's own operations:

  * P2G (scatter each particle's mass and momentum onto the grid through a smooth kernel) is a SUPERPOSITION:
        grid_node  =  SUM over particles of  weight(node, particle) * particle_value
    That is a BUNDLE of kernel-weighted, position-bound particle contributions -- the SAME operation as
    holographic_splat.splat_render ("a splat scene IS a bundle of Gaussians") and the RBF ScalarEncoder, whose own
    docstring notes that a bundle of encoded points reads as a kernel density estimate (Bochner: the encoder is a
    shift-invariant kernel). _selftest verifies this: the P2G mass grid EQUALS an independent bundle of kernel
    splats, to machine precision.
  * The quadratic B-spline WEIGHTS are that kernel -- a smooth partition-of-unity bump (weights sum to 1), exactly
    the Gaussian-like bump the RBF encoder and a splat use. Because they sum to 1, the bundle preserves total mass.
  * G2P (gather the grid back onto a particle) is the READOUT -- querying that bundle at the particle's position,
    the same gather a resonator/archive does. The P2G->G2P round-trip conserves total momentum (verified) -- the
    bundle->readout fidelity property, "as above, so below": scatter into a superposition, read one back out.
  * The accumulation at a node is a COMMUTATIVE MONOID (mass and momentum ADD), which is exactly the field that
    holographic_distribute partitions and reassembles by the monoid's own operator -- so P2G is RAID-style width.

What is genuinely NOT holographic, kept honest: the GRID UPDATE -- the per-node elasto-plastic stress and the SVD
plastic clamp -- is nonlinear LOCAL physics on a plain grid. It earns no bind; forcing one would be dishonest (the
don't-over-holograph-a-grid rule). So MPM is a hybrid: a holographic transfer (bundle/readout) around a grid-native
constitutive update. That hybrid IS the honest picture.

HONEST SCOPE (the VFX-vs-physics line): this is a readable 2-D MLS-MPM demonstration -- it shows snow falling,
piling, and compressing PLASTICALLY (permanent deformation, no full rebound), with mass and transfer-momentum
conserved. Simplifications kept loud: constant Lame parameters (no hardening-by-compression term -- Stomakhin's
exp(xi*(1-Jp)) is omitted for readability); explicit time integration (a CFL cap, no implicit solve); PIC-flavoured
transfer (dissipative -- APIC/FLIP reduce that); 2-D. Production snow (hardening, implicit, 3-D, sand/mud variants)
is the research-heavy extension. Deterministic given the seed; NumPy + stdlib only.
"""
import numpy as np


def _lame(E, nu):
    """Lame parameters (mu = shear modulus, lam = bulk-ish) from Young's modulus E and Poisson ratio nu."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def _bspline(fx):
    """Quadratic B-spline weights for the 3 grid nodes a particle touches along one axis. `fx` is the particle's
    position relative to its base node, in [0.5, 1.5). The three weights sum to 1 -- a partition of unity, i.e. a
    NORMALIZED bundle, which is why P2G preserves total mass. Returns (w0, w1, w2)."""
    return (0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2)


class MPMSnow:
    """A 2-D Material Point Method snow solver. Particles carry mass, velocity, an affine field C, and a
    deformation gradient F; the grid is a scratchpad they scatter onto (P2G = bundle), the grid does the physics,
    and they gather back (G2P = readout). Snow's plasticity is an SVD clamp on F each step."""

    def __init__(self, grid=48, dx=1.0, E=140.0, nu=0.2, theta_c=0.025, theta_s=0.0075, gravity=9.81, seed=0):
        self.grid = int(grid)
        self.dx = float(dx)
        self.inv_dx = 1.0 / self.dx
        self.mu0, self.lam0 = _lame(E, nu)
        self.theta_c = float(theta_c)                          # compression yield (snow packs at ~2.5% strain)
        self.theta_s = float(theta_s)                          # stretch yield (snow tears easily, ~0.75%)
        self.gravity = float(gravity)
        self.rng = np.random.default_rng(seed)
        # particle arrays (filled by seeding)
        self.x = np.zeros((0, 2)); self.v = np.zeros((0, 2))
        self.C = np.zeros((0, 2, 2)); self.F = np.zeros((0, 2, 2))
        self.m = np.zeros((0,)); self.vol = np.zeros((0,))

    def seed_block(self, cx, cy, w, h, n, mass=1.0):
        """Seed `n` snow particles filling a w-by-h block centred at (cx, cy)."""
        pts = self.rng.uniform([cx - w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], size=(n, 2))
        self.x = np.vstack([self.x, pts])
        self.v = np.vstack([self.v, np.zeros((n, 2))])
        self.C = np.concatenate([self.C, np.zeros((n, 2, 2))])
        self.F = np.concatenate([self.F, np.tile(np.eye(2), (n, 1, 1))])   # undeformed: F = I
        self.m = np.concatenate([self.m, np.full(n, mass)])
        self.vol = np.concatenate([self.vol, np.full(n, 1.0)])
        return self

    def _weights(self):
        """The per-particle base node and the 3x2 B-spline weights (and their derivatives) -- shared by P2G and
        G2P so the scatter and gather use the identical kernel."""
        xg = self.x * self.inv_dx
        base = (xg - 0.5).astype(int)                          # the low corner of the 3x3 stencil
        fx = xg - base                                        # fractional position in [0.5, 1.5)
        wx = np.stack(_bspline(fx[:, 0]), axis=1)             # (N, 3)
        wy = np.stack(_bspline(fx[:, 1]), axis=1)             # (N, 3)
        return base, fx, wx, wy

    def _stress(self):
        """The MLS-MPM stress term fed into the grid, from each particle's deformation gradient F (fixed corotated
        elasticity). This is the GRID-NATIVE nonlinear physics -- no bind here, honestly. Returns (N,2,2)."""
        F = self.F
        U, sig, Vt = np.linalg.svd(F)                         # batched 2x2 SVD (vectorised over particles)
        R = np.einsum("nij,njk->nik", U, Vt)                  # the rotation part (polar decomposition)
        J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]  # det F = volume change
        # Kirchhoff stress PF^T = 2mu (F-R) F^T + lam (J-1) J I  (fixed corotated)
        FmR = F - R
        term1 = 2.0 * self.mu0 * np.einsum("nij,nkj->nik", FmR, F)   # (F-R) F^T
        term2 = (self.lam0 * (J - 1.0) * J)[:, None, None] * np.eye(2)[None]
        return term1 + term2

    def step(self, dt):
        """One full MPM step: P2G (bundle) -> grid update (grid-native physics) -> G2P (readout) -> advect + yield."""
        G = self.grid
        grid_v = np.zeros((G, G, 2))
        grid_m = np.zeros((G, G))
        base, fx, wx, wy = self._weights()
        stress = self._stress()
        affine = -dt * self.vol[:, None, None] * (4.0 * self.inv_dx ** 2) * stress + self.m[:, None, None] * self.C

        # --- P2G: SCATTER each particle onto its 9 neighbouring nodes = a BUNDLE (superposition) ---
        for i in range(3):
            for j in range(3):
                w = wx[:, i] * wy[:, j]                        # kernel weight to node (i, j) of the stencil
                dpos = (np.stack([i - fx[:, 0], j - fx[:, 1]], axis=1)) * self.dx   # node minus particle
                gx = base[:, 0] + i
                gy = base[:, 1] + j
                mom = w[:, None] * (self.m[:, None] * self.v + np.einsum("nij,nj->ni", affine, dpos))
                np.add.at(grid_v, (gx, gy), mom)              # momentum bundles (adds) at the node
                np.add.at(grid_m, (gx, gy), w * self.m)       # mass bundles (adds) at the node

        # --- grid update: normalise, gravity, and box boundaries (the grid-native step) ---
        nz = grid_m > 1e-12
        grid_v[nz] /= grid_m[nz][:, None]
        grid_v[..., 1] -= dt * self.gravity                   # gravity pulls down (-y)
        b = 2
        grid_v[:b, :, 0] = np.minimum(grid_v[:b, :, 0], 0)    # left wall: no outflow
        grid_v[-b:, :, 0] = np.maximum(grid_v[-b:, :, 0], 0)  # right wall
        grid_v[:, :b, 1] = np.maximum(grid_v[:, :b, 1], 0)    # floor: no downward flow
        grid_v[:, -b:, 1] = np.minimum(grid_v[:, -b:, 1], 0)  # ceiling

        # --- G2P: GATHER the grid back onto each particle = the READOUT (query the bundle at the position) ---
        new_v = np.zeros_like(self.v)
        new_C = np.zeros_like(self.C)
        for i in range(3):
            for j in range(3):
                w = wx[:, i] * wy[:, j]
                dpos = (np.stack([i - fx[:, 0], j - fx[:, 1]], axis=1)) * self.dx
                gv = grid_v[base[:, 0] + i, base[:, 1] + j]    # (N, 2) the node velocity
                new_v += w[:, None] * gv
                new_C += 4.0 * self.inv_dx ** 2 * w[:, None, None] * np.einsum("ni,nj->nij", gv, dpos)
        self.v = new_v
        self.C = new_C
        self.x = self.x + dt * self.v                          # advect

        # --- update F and apply snow PLASTICITY: clamp the singular values (permanent yield) ---
        self.F = np.einsum("nij,njk->nik", np.eye(2)[None] + dt * self.C, self.F)
        U, sig, Vt = np.linalg.svd(self.F)
        sig = np.clip(sig, 1.0 - self.theta_c, 1.0 + self.theta_s)   # the yield: snow can't stretch/compress past this
        self.F = np.einsum("nij,nj,njk->nik", U, sig, Vt)
        return self

    def run(self, dt, steps):
        for _ in range(int(steps)):
            self.step(dt)
        return self

    # -- diagnostics used by the tests and the holographic verification -----------------------------------------
    def total_mass(self):
        return float(self.m.sum())

    def total_momentum(self):
        return self.m[:, None] * self.v

    def center_of_mass(self):
        return (self.m[:, None] * self.x).sum(0) / self.m.sum()

    def p2g_mass_grid(self):
        """The P2G mass field, computed as the transfer does it -- used to VERIFY it equals an independent bundle."""
        G = self.grid
        grid_m = np.zeros((G, G))
        base, fx, wx, wy = self._weights()
        for i in range(3):
            for j in range(3):
                w = wx[:, i] * wy[:, j]
                np.add.at(grid_m, (base[:, 0] + i, base[:, 1] + j), w * self.m)
        return grid_m


def _bundle_mass_grid(mpm):
    """An INDEPENDENT computation of the grid mass as a BUNDLE (superposition) of kernel splats: for each particle,
    splat its mass through the B-spline kernel and SUM. If P2G is really bundling, this equals p2g_mass_grid()
    exactly. (This is the holographic identity made checkable -- the same move splat_render makes for a scene.)"""
    G = mpm.grid
    grid = np.zeros((G, G))
    xg = mpm.x * mpm.inv_dx
    base = (xg - 0.5).astype(int)
    fx = xg - base
    for p in range(len(mpm.x)):
        wx = _bspline(fx[p, 0]); wy = _bspline(fx[p, 1])
        for i in range(3):
            for j in range(3):
                grid[base[p, 0] + i, base[p, 1] + j] += (wx[i] * wy[j]) * mpm.m[p]   # one splat, bundled in
    return grid


def _selftest():
    """P2G conserves mass and equals an independent BUNDLE of kernel splats (P2G IS bundling); the P2G->G2P
    round-trip conserves momentum (bundle->readout fidelity); snow falls under gravity and PILES/compresses
    plastically (permanent deformation, no full rebound); deterministic."""
    # (1) HOLOGRAPHIC IDENTITY: the P2G mass grid == an independent bundle of kernel splats, to machine precision
    m = MPMSnow(grid=48, seed=0).seed_block(cx=24, cy=30, w=10, h=10, n=300)
    p2g = m.p2g_mass_grid()
    bundle = _bundle_mass_grid(m)
    assert np.allclose(p2g, bundle, atol=1e-9), np.abs(p2g - bundle).max()
    # and the bundle preserves total mass (weights are a partition of unity = a normalized bundle)
    assert abs(p2g.sum() - m.total_mass()) < 1e-9

    # (2) MOMENTUM CONSERVED through the P2G->G2P round-trip (gravity off) -- the bundle->readout fidelity
    m2 = MPMSnow(grid=48, gravity=0.0, seed=1).seed_block(cx=24, cy=24, w=8, h=8, n=200)
    m2.v[:] = np.array([0.4, -0.2])                            # give the snow a uniform drift
    p_before = m2.total_momentum().sum(0)
    m2.step(dt=1e-3)                                          # one transfer round-trip, tiny dt (stress ~ 0 at F=I)
    p_after = m2.total_momentum().sum(0)
    assert np.allclose(p_before, p_after, atol=2e-2), (p_before, p_after)

    # (3) snow FALLS under gravity
    snow = MPMSnow(grid=48, gravity=9.81, seed=2).seed_block(cx=24, cy=12, w=10, h=8, n=400)
    y0 = snow.center_of_mass()[1]
    top0 = snow.x[:, 1].max()
    extent0 = snow.x[:, 1].max() - snow.x[:, 1].min()         # the block's vertical thickness
    snow.run(dt=2e-3, steps=400)
    assert snow.center_of_mass()[1] < y0 - 1.5, (y0, snow.center_of_mass()[1])   # the centre of mass dropped

    # (4) snow PILES and COMPRESSES plastically: it lands and settles LOWER, and its vertical extent shrinks
    # permanently (the singular-value clamp is a real yield, so it does NOT spring back).
    snow.run(dt=2e-3, steps=400)                              # let it settle on the floor
    top1 = snow.x[:, 1].max()
    extent1 = snow.x[:, 1].max() - snow.x[:, 1].min()
    assert top1 < top0 - 3.0, (top0, top1)                   # the top settled well down toward the floor
    assert extent1 < 0.7 * extent0, (extent0, extent1)       # compressed plastically (thinner than it started)
    assert np.isfinite(snow.x).all()
    assert abs(snow.total_mass() - 400.0) < 1e-9             # mass conserved throughout

    # (5) deterministic
    a = MPMSnow(grid=48, seed=5).seed_block(24, 30, 8, 8, 100).run(2e-3, 50).x
    b = MPMSnow(grid=48, seed=5).seed_block(24, 30, 8, 8, 100).run(2e-3, 50).x
    assert np.array_equal(a, b)

    print("holographic_mpm selftest OK: P2G IS bundling -- its mass grid equals an independent bundle of kernel "
          "splats to 1e-9, and conserves total mass; the P2G->G2P round-trip conserves momentum (bundle->readout "
          "fidelity); snow falls under gravity (CoM %.1f -> %.1f), then piles and compresses PLASTICALLY (top %.1f "
          "-> %.1f, vertical extent %.1f -> %.1f) with mass conserved; deterministic"
          % (y0, float(snow.center_of_mass()[1]), float(top0), float(top1), float(extent0), float(extent1)))


if __name__ == "__main__":
    _selftest()
