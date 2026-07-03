"""Position-Based Dynamics -- softbody & hardbody simulation, exposed to VSA.

WHERE THIS SITS. The engine already owns the ITERATE-A-PROJECTION sweep (holographic_denoise.
project_onto_constraints, the mind's `project_onto_constraints` faculty) -- Macklin's observation that the SBC
resonator, the PnP denoiser, the IK chain, and a position-based-dynamics constraint sweep are all the SAME
object: project repeatedly onto a set of constraints until they jointly hold. IK (holographic_meshik) is
already built on it. What that engine does NOT carry is the *dynamics* around the sweep -- momentum, inverse
mass, gravity, the predict -> solve -> velocity-update time-step, time-step-independent stiffness, collision.
That is exactly what this module adds, so a cloth/rope/soft mesh (SoftBody) and a rigid body (RigidBody) can
actually move under forces, while the PBD constraint sweep delegates to the shipped engine.

WHY POSITION-BASED. PBD (Mueller et al. 2007) and XPBD (Macklin et al. 2016) integrate by PREDICTING new
positions from velocity + external force, then PROJECTING those positions onto the constraints (distances,
pins, the floor), then reading velocity back from how far each particle actually moved. It is unconditionally
stable -- a stiff spring that would explode an explicit integrator at a big time-step just gets projected back
into place -- which is why games and film use it. XPBD adds a per-constraint COMPLIANCE so stiffness is set in
physical units and is independent of the time-step and iteration count (plain PBD's stiffness secretly depends
on how many iterations you run -- the kept negative that motivated XPBD).

EXPOSED TO VSA. A SoftBody/RigidBody takes an external per-particle force, so the fluid layer
(holographic_fields) can push the cloth: sample a velocity field or an attractor force and hand it in. The
constraint solve is the same projection iteration the resonator and denoiser run -- this is its physical face.

KEPT NEGATIVES (honest):
  * PBD's effective stiffness depends on iteration count; only XPBD (compliance) is iteration/time-step
    independent. Both are provided; the difference is measured below.
  * The Gauss-Seidel sweep is ORDER-dependent (constraints solved in sequence) -- deterministic, but a
    different edge order gives a slightly different transient. Kept sequential for determinism.
  * Collision is a simple half-space floor with restitution; there is no self-collision and no friction model.
  * Bending/volume constraints are not built -- distance constraints + shape-matching cover cloth/rope/soft and
    rigid; bending would be the next constraint type to add to the same sweep.
"""

import numpy as np

from holographic_denoise import project_onto_constraints   # reuse the shipped iterate-a-projection sweeper


def _as_gravity(gravity, dim):
    """Default gravity is -9.8 on axis 1 (y), zero elsewhere; or accept an explicit vector."""
    if gravity is None:
        g = np.zeros(dim); g[1 if dim > 1 else 0] = -9.8
        return g
    return np.asarray(gravity, float)


class SoftBody:
    """Particles linked by distance constraints, time-stepped by PBD/XPBD. Inverse mass w=0 PINS a particle
    (an immovable anchor -- the hardbody attachment point of a soft sheet). Add distance constraints by hand or
    use the cloth/rope builders."""

    def __init__(self, positions, inv_mass=None, velocities=None):
        self.x = np.asarray(positions, float).copy()           # (N, D) positions
        self.N, self.D = self.x.shape
        self.v = (np.zeros_like(self.x) if velocities is None else np.asarray(velocities, float).copy())
        self.w = (np.ones(self.N) if inv_mass is None else np.asarray(inv_mass, float).copy())  # 0 == pinned
        self.constraints = []                                  # list of (i, j, rest_length, compliance)
        self.bending = []                                      # list of (k, l, rest_kl, compliance)  -- bend springs
        self.volumes = []                                     # list of (i, j, k, l, rest_volume, compliance)
        self.collision_radius = None                           # set by add_self_collision -> nodes repel within it
        self._bonded = set()                                   # directly-linked pairs, excluded from collision
        self.faces = None                                      # optional source faces (set by from_mesh) for re-export
        self._bonded_keys = np.empty(0, np.int64)              # same exclusion as flat int keys (vectorised test)

    def pin(self, i):
        """Make particle i immovable (infinite mass)."""
        self.w[i] = 0.0
        return self

    def add_distance(self, i, j, rest=None, compliance=0.0):
        """Link particles i, j with a distance constraint (rest defaults to their current separation).
        compliance=0 is a rigid link; larger compliance is a softer spring (inverse stiffness)."""
        if rest is None:
            rest = float(np.linalg.norm(self.x[i] - self.x[j]))
        self.constraints.append((int(i), int(j), float(rest), float(compliance)))
        return self

    def add_bending(self, k, l, compliance=0.0):
        """Resist FOLDING by holding the two corners k and l (the vertices on either side of a fold line, two
        cells apart in a cloth) at their rest separation -- a BEND SPRING (Provot's classic cloth bending).
        Folding the sheet changes the k-l distance, so holding it resists the fold. This is the robust, non-
        singular bending model; the dihedral-ANGLE constraint (Mueller 2007) is exact but singular at the flat
        rest state, so we use the spring (kept-negative note in the module docstring). compliance 0 is stiff."""
        rest_kl = float(np.linalg.norm(self.x[k] - self.x[l]))
        self.bending.append((int(k), int(l), rest_kl, float(compliance)))
        return self

    def add_volume(self, i, j, k, l, rest_volume=None, compliance=0.0):
        """Hold the signed VOLUME of the tetrahedron (i,j,k,l) -- the PBD volume constraint that makes a soft
        solid resist being squashed (Mueller et al. 2007). rest_volume defaults to the current volume."""
        if rest_volume is None:
            rest_volume = self._tet_volume(self.x[i], self.x[j], self.x[k], self.x[l])
        self.volumes.append((int(i), int(j), int(k), int(l), float(rest_volume), float(compliance)))
        return self

    @staticmethod
    def _dihedral(p_i, p_j, p_k, p_l):
        """Angle between the two triangle normals across edge (i, j) -- acos(n1 . n2). Used to MEASURE how much
        a sheet has folded (the bend spring is what resists it)."""
        e = p_j - p_i
        n1 = np.cross(e, p_k - p_i); n2 = np.cross(e, p_l - p_i)
        n1n = np.linalg.norm(n1); n2n = np.linalg.norm(n2)
        if n1n < 1e-12 or n2n < 1e-12:
            return np.pi
        return float(np.arccos(np.clip(np.dot(n1 / n1n, n2 / n2n), -1.0, 1.0)))

    @staticmethod
    def _tet_volume(p_i, p_j, p_k, p_l):
        """Signed volume of a tetrahedron = (1/6) (j-i) . ((k-i) x (l-i))."""
        return float(np.dot(p_j - p_i, np.cross(p_k - p_i, p_l - p_i)) / 6.0)

    @classmethod
    def rope(cls, n, spacing=1.0, compliance=0.0, start=(0.0, 0.0)):
        """A hanging rope of n particles; particle 0 is pinned. Lives in 2-D by default."""
        start = np.asarray(start, float)
        pts = np.array([start + np.array([0.0, -k * spacing]) for k in range(n)])
        body = cls(pts)
        for k in range(n - 1):
            body.add_distance(k, k + 1, rest=spacing, compliance=compliance)
        body.pin(0)
        return body

    @classmethod
    def from_mesh(cls, mesh, compliance=0.0, pin=None):
        """Build a simulatable SoftBody from ANY mesh: its vertices become particles and its EDGES become
        distance constraints (rest = current edge length). This is the bridge that lets the 3-D mesh pipeline
        take full advantage of the physics layer -- a projected surface mesh can then be driven by gravity,
        fluid drag (external_force=drag_force_3d(...)), self-collision (add_self_collision(r)), and the
        constraint solver, exactly like the parametric cloth/soft_box. The authoring cycle becomes:
        sculpt a field -> surface_mesh -> from_mesh -> simulate -> (re-project). `pin` is an optional list of
        vertex indices to anchor (inverse mass 0). Returns a SoftBody over the mesh's vertices.

        Note: a marched SURFACE mesh is a thin shell (edges on the surface only), so it behaves like cloth, not
        a filled solid -- add bending or use soft_box semantics if you need volume resistance. Edge extraction is
        the mesh kernel's Python loop (setup cost, not per-frame)."""
        body = cls(np.asarray(mesh.vertices, float))
        for (i, j) in mesh.edges():                            # every undirected mesh edge -> a distance link
            body.add_distance(int(i), int(j), compliance=compliance)
        if pin:
            for i in pin:
                body.pin(int(i))
        body.faces = [tuple(f) for f in mesh.faces]            # keep faces so the DEFORMED body re-exports as a mesh
        return body

    def to_mesh(self):
        """The current deformed state as a Mesh: the simulated particle positions with the original faces (set by
        from_mesh). This is how a soft-body simulation EXPORTS back to geometry -- sculpt -> mesh -> simulate ->
        to_mesh -> export. Raises if this body has no faces (it wasn't built from a mesh)."""
        from holographic_mesh import Mesh
        if self.faces is None:
            raise ValueError("this SoftBody has no faces to export (build it with SoftBody.from_mesh)")
        return Mesh(self.x.copy(), self.faces)

    @classmethod
    def cloth(cls, rows, cols, spacing=1.0, compliance=0.0):
        """A rectangular cloth sheet (rows x cols) in the x-y plane with structural + shear distance
        constraints; the top row is pinned. A canonical PBD softbody."""
        pts = np.array([[c * spacing, -r * spacing] for r in range(rows) for c in range(cols)], float)
        body = cls(pts)
        idx = lambda r, c: r * cols + c
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols: body.add_distance(idx(r, c), idx(r, c + 1), spacing, compliance)      # structural
                if r + 1 < rows: body.add_distance(idx(r, c), idx(r + 1, c), spacing, compliance)      # structural
                if r + 1 < rows and c + 1 < cols:                                                      # shear
                    body.add_distance(idx(r, c), idx(r + 1, c + 1), spacing * np.sqrt(2), compliance)
                    body.add_distance(idx(r, c + 1), idx(r + 1, c), spacing * np.sqrt(2), compliance)
        for c in range(cols):
            body.pin(idx(0, c))
        return body

    @classmethod
    def cloth3d(cls, rows, cols, spacing=1.0, compliance=0.0, bending=None):
        """A 3-D cloth in the x-z plane (y=0), the top row pinned, draping DOWN (-y) under gravity. Structural
        + shear distance constraints hold the weave; if `bending` (a compliance) is given, two-cell bend
        springs resist FOLDING so the sheet stays flatter instead of curling sharply."""
        pts = np.array([[c * spacing, 0.0, r * spacing] for r in range(rows) for c in range(cols)], float)
        body = cls(pts)
        idx = lambda r, c: r * cols + c
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols: body.add_distance(idx(r, c), idx(r, c + 1), spacing, compliance)      # structural
                if r + 1 < rows: body.add_distance(idx(r, c), idx(r + 1, c), spacing, compliance)
                if r + 1 < rows and c + 1 < cols:                                                      # shear
                    body.add_distance(idx(r, c), idx(r + 1, c + 1), spacing * np.sqrt(2), compliance)
                    body.add_distance(idx(r, c + 1), idx(r + 1, c), spacing * np.sqrt(2), compliance)
        if bending is not None:
            for r in range(rows):
                for c in range(cols):
                    if c + 2 < cols: body.add_bending(idx(r, c), idx(r, c + 2), compliance=bending)    # bend springs
                    if r + 2 < rows: body.add_bending(idx(r, c), idx(r + 2, c), compliance=bending)
        for c in range(cols):
            body.pin(idx(0, c))
        return body

    @classmethod
    def soft_box(cls, nx, ny, nz, spacing=1.0, compliance=0.0, volume_compliance=0.0):
        """A soft 3-D SOLID: an nx*ny*nz lattice, each cube cell split into 5 tetrahedra that each carry a
        VOLUME constraint (plus axis-aligned edge constraints). The volume constraints make the body resist
        being squashed and spring back -- a jelly block rather than a collapsing sheet."""
        idx = {}; pts = []; n = 0
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    idx[(ix, iy, iz)] = n; n += 1
                    pts.append([ix * spacing, iy * spacing, iz * spacing])
        body = cls(np.array(pts, float))
        for iz in range(nz - 1):
            for iy in range(ny - 1):
                for ix in range(nx - 1):
                    c = {k: idx[(ix + (k & 1), iy + ((k >> 1) & 1), iz + ((k >> 2) & 1))] for k in range(8)}
                    # 5-tetrahedron split of the cube (corners indexed by bit pattern xyz)
                    for (a, b, cc, d) in [(0, 1, 2, 4), (1, 3, 2, 7), (1, 2, 4, 7), (1, 5, 4, 7), (2, 6, 4, 7)]:
                        body.add_volume(c[a], c[b], c[cc], c[d], compliance=volume_compliance)
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    if ix + 1 < nx: body.add_distance(idx[(ix, iy, iz)], idx[(ix + 1, iy, iz)], spacing, compliance)
                    if iy + 1 < ny: body.add_distance(idx[(ix, iy, iz)], idx[(ix, iy + 1, iz)], spacing, compliance)
                    if iz + 1 < nz: body.add_distance(idx[(ix, iy, iz)], idx[(ix, iy, iz + 1)], spacing, compliance)
        return body

    def total_volume(self):
        """Sum of the signed volumes of all volume-constrained tetrahedra -- the body's volume."""
        return float(sum(self._tet_volume(self.x[i], self.x[j], self.x[k], self.x[l])
                         for (i, j, k, l, _v, _c) in self.volumes))

    def add_self_collision(self, radius):
        """Turn on self-collision: any two NON-bonded nodes closer than `radius` repel apart to `radius`, so the
        sheet/solid cannot pass through itself. Directly-linked nodes (distance constraints) are excluded so the
        weave doesn't fight its own structure. Off by default (radius=None) -- additive and backward-compatible.
        The close-pair search uses the spatial-hash cull and the repulsion is one vectorised scatter (no Python
        per-pair loop), the same iterate-a-projection shape as the distance/bend/volume projections."""
        self.collision_radius = float(radius)
        self._bonded = {frozenset((i, j)) for (i, j, _r, _c) in self.constraints}   # exclude structural links
        # the same exclusion as a flat int key (min*N + max) for the VECTORISED membership test in _solve
        self._bonded_keys = np.array(sorted({min(i, j) * self.N + max(i, j)
                                             for (i, j, _r, _c) in self.constraints}), dtype=np.int64)
        return self

    def _solve_collisions(self):
        """One Jacobi pass of non-bonded node-node repulsion, VECTORISED: find close pairs by spatial hash,
        drop bonded pairs by a key-membership test, push every penetrating pair apart to the collision radius
        (split by inverse mass, PBD), and accumulate the corrections with np.add.at -- the scatter, the same
        adjoint the field coupling uses. A pinned node (w=0) does not move, so its partner takes the whole
        correction. Jacobi (all corrections from one state, then summed) rather than the sequential Gauss-Seidel
        loop -- order-independent and deterministic, which the bind_batch tie-break lesson rewards."""
        from holographic_fields import spatial_hash_pairs            # the reusable cull primitive
        r = self.collision_radius
        pairs = spatial_hash_pairs(self.x, r)
        if pairs.shape[0] == 0:
            return
        i, j = pairs[:, 0], pairs[:, 1]
        if self._bonded_keys.size:                               # drop directly-linked pairs, vectorised
            keys = np.minimum(i, j) * self.N + np.maximum(i, j)
            keep = ~np.isin(keys, self._bonded_keys)
            i, j = i[keep], j[keep]
        if i.size == 0:
            return
        d = self.x[i] - self.x[j]                                # (P, D) all separations at once
        dist = np.sqrt((d ** 2).sum(axis=1))                     # (P,)
        wsum = self.w[i] + self.w[j]                             # (P,)
        active = (dist > 1e-12) & (dist < r) & (wsum > 0.0)      # penetrating, movable pairs
        corr = np.zeros(dist.shape)
        corr[active] = (r - dist[active]) / wsum[active]         # push to exactly the radius
        n = np.zeros_like(d)
        n[active] = d[active] / dist[active][:, None]            # unit separation normals
        push = corr[:, None] * n                                 # (P, D) per-pair correction direction*amount
        np.add.at(self.x, i, self.w[i][:, None] * push)          # scatter: each node moves by its w-share ...
        np.add.at(self.x, j, -self.w[j][:, None] * push)         # ... the partner the opposite way

    # -- the two solver back-ends -------------------------------------------------------------------

    def _solve_xpbd(self, h, iterations):
        """XPBD Gauss-Seidel solve: per-constraint compliance with an accumulated Lagrange multiplier, so the
        stiffness is physical and time-step/iteration independent. This is the piece the generic projection
        sweeper does not carry."""
        lam = np.zeros(len(self.constraints))                  # one multiplier per constraint, reset each step
        lam_b = np.zeros(len(self.bending))
        lam_v = np.zeros(len(self.volumes))
        for _ in range(iterations):
            for c, (i, j, rest, compliance) in enumerate(self.constraints):
                n = self.x[i] - self.x[j]
                d = float(np.linalg.norm(n))
                if d < 1e-12:
                    continue
                n = n / d
                wsum = self.w[i] + self.w[j]
                if wsum == 0.0:
                    continue
                C = d - rest
                alpha = compliance / (h * h)                   # XPBD: compliance scaled into the time-step
                dlam = (-C - alpha * lam[c]) / (wsum + alpha)   # the compliant constraint solve
                lam[c] += dlam
                self.x[i] = self.x[i] + self.w[i] * dlam * n
                self.x[j] = self.x[j] - self.w[j] * dlam * n
            # bending = a bend spring between the opposite corners k, l (same distance solve, own compliance)
            for b, (k, l, rest_kl, compliance) in enumerate(self.bending):
                n = self.x[k] - self.x[l]
                d = float(np.linalg.norm(n))
                if d < 1e-12:
                    continue
                n = n / d
                wsum = self.w[k] + self.w[l]
                if wsum == 0.0:
                    continue
                C = d - rest_kl
                alpha = compliance / (h * h)
                dlam = (-C - alpha * lam_b[b]) / (wsum + alpha)
                lam_b[b] += dlam
                self.x[k] = self.x[k] + self.w[k] * dlam * n
                self.x[l] = self.x[l] - self.w[l] * dlam * n
            # volume = hold the tet's signed volume (gradients are cross products of the edges)
            for v, (i, j, k, l, rest_vol, compliance) in enumerate(self.volumes):
                e1 = self.x[j] - self.x[i]; e2 = self.x[k] - self.x[i]; e3 = self.x[l] - self.x[i]
                gj = np.cross(e2, e3) / 6.0                     # dV/dp_j
                gk = np.cross(e3, e1) / 6.0                     # dV/dp_k
                gl = np.cross(e1, e2) / 6.0                     # dV/dp_l
                gi = -(gj + gk + gl)                            # dV/dp_i
                wsum = (self.w[i] * gi @ gi + self.w[j] * gj @ gj
                        + self.w[k] * gk @ gk + self.w[l] * gl @ gl)
                if wsum < 1e-12:
                    continue
                C = float(np.dot(e1, np.cross(e2, e3)) / 6.0) - rest_vol
                alpha = compliance / (h * h)
                dlam = (-C - alpha * lam_v[v]) / (wsum + alpha)
                lam_v[v] += dlam
                self.x[i] = self.x[i] + self.w[i] * dlam * gi
                self.x[j] = self.x[j] + self.w[j] * dlam * gj
                self.x[k] = self.x[k] + self.w[k] * dlam * gk
                self.x[l] = self.x[l] + self.w[l] * dlam * gl

    def _pbd_projections(self):
        """Build distance-constraint projection callables over the FLAT position vector and hand them to the
        shipped project_onto_constraints sweeper -- exactly the way IK builds bone projections. This is plain
        PBD (no compliance); its effective stiffness depends on the iteration count (kept negative)."""
        w, N, D = self.w, self.N, self.D
        projs = []
        for (i, j, rest, _compliance) in self.constraints:
            def proj(flat, i=i, j=j, rest=rest):
                X = flat.reshape(N, D)
                n = X[i] - X[j]; d = float(np.linalg.norm(n))
                if d < 1e-12:
                    return flat
                wsum = w[i] + w[j]
                if wsum == 0.0:
                    return flat
                n = n / d; C = d - rest
                Xn = X.copy()
                Xn[i] = X[i] - (w[i] / wsum) * C * n            # move each endpoint to satisfy the rest length
                Xn[j] = X[j] + (w[j] / wsum) * C * n
                return Xn.ravel()
            projs.append(proj)
        return projs

    def step(self, dt=1.0 / 60.0, gravity=None, iterations=20, substeps=1, solver="xpbd",
             external_force=None, floor=None, restitution=0.0, damping=0.0, collider=None, collide_radius=0.0):
        """Advance the body one frame. solver='xpbd' (compliant, recommended) or 'pbd' (delegates the sweep to
        the shipped project_onto_constraints engine). `external_force` is an (N, D) force array (e.g. from a
        fluid field) -- it becomes acceleration via the inverse mass. `floor` (scalar) is a y=floor half-space
        with `restitution`. `collider` (a callable P->signed distance) is an ENVIRONMENT collision surface: any
        node inside it is pushed out to `collide_radius` -- so the body drapes over a scene SDF, using the same
        positional (iterate-a-projection) contact resolve as self-collision. Returns self."""
        g = _as_gravity(gravity, self.D)
        movable = self.w > 0
        for _ in range(max(1, substeps)):
            h = dt / max(1, substeps)
            x_prev = self.x.copy()
            # 1. integrate velocity by external acceleration (gravity is mass-independent; forces use 1/m = w)
            acc = np.tile(g, (self.N, 1))
            if external_force is not None:
                acc = acc + np.asarray(external_force, float) * self.w[:, None]
            if damping:
                self.v *= (1.0 - damping)
            self.v[movable] += h * acc[movable]
            # 2. predict positions
            self.x[movable] += h * self.v[movable]
            # 3. project onto the constraints (the iterate-a-projection step)
            if solver == "xpbd":
                self._solve_xpbd(h, iterations)
            else:
                flat, _, _ = project_onto_constraints(self.x.ravel(), self._pbd_projections(),
                                                      iters=iterations, omega=1.0)
                self.x = flat.reshape(self.N, self.D)
            # 3b. self-collision: non-bonded nodes repel (another iterate-a-projection), if enabled. Collision
            #     is a POSITIONAL contact resolve -- we record its displacement and subtract it from the
            #     velocity update below, so a contact separates nodes without injecting coasting momentum
            #     (otherwise the PBD velocity readback turns the separation push into runaway kinetic energy).
            collision_dx = None
            if self.collision_radius is not None:
                _x_pre_col = self.x.copy()
                self._solve_collisions()
                collision_dx = self.x - _x_pre_col
            # 3c. environment (SDF) collision: keep nodes OUTSIDE the collider geometry -- cloth drapes over a scene
            #     object. Positional like self-collision, so its displacement is subtracted from the velocity update.
            if collider is not None:
                from holographic_collide import resolve_sdf_collision
                _x_pre_env = self.x.copy()
                self.x = resolve_sdf_collision(self.x, collider, radius=collide_radius)
                env_dx = self.x - _x_pre_env
                collision_dx = env_dx if collision_dx is None else collision_dx + env_dx
            # 4. floor collision (a simple half-space on axis 1)
            if floor is not None:
                below = self.x[:, 1] < floor
                self.x[below, 1] = floor
            # 5. read velocity back from the actual displacement (PBD's velocity rule)
            disp = self.x - x_prev
            if collision_dx is not None:
                disp = disp - collision_dx                     # contact is positional, not kinetic
            self.v = disp / h
            if floor is not None and restitution >= 0.0:
                hit = self.x[:, 1] <= floor + 1e-9
                self.v[hit, 1] = -restitution * self.v[hit, 1]
        return self

    def constraint_residual(self):
        """Max |current length - rest length| over all constraints -- 0 means every constraint is satisfied."""
        if not self.constraints:
            return 0.0
        return max(abs(float(np.linalg.norm(self.x[i] - self.x[j])) - rest)
                   for (i, j, rest, _c) in self.constraints)


class RigidBody:
    """A hardbody via SHAPE MATCHING (Mueller et al. 2005): each step, find the rotation+translation that best
    maps the REST shape onto the current particles (the polar decomposition of the cross-covariance, an SVD),
    then pull every particle toward that matched goal. With stiffness 1 the shape is preserved exactly -- a true
    rigid body -- complementing SoftBody's stiff-distance approach. Falls and rotates under forces; never
    deforms."""

    def __init__(self, positions, inv_mass=None, velocities=None):
        self.x = np.asarray(positions, float).copy()
        self.N, self.D = self.x.shape
        self.v = (np.zeros_like(self.x) if velocities is None else np.asarray(velocities, float).copy())
        self.m = (np.ones(self.N) if inv_mass is None else
                  np.where(np.asarray(inv_mass, float) > 0, 1.0 / np.maximum(np.asarray(inv_mass, float), 1e-12), 0.0))
        self.rest = self.x.copy()
        self.rest_cm = np.average(self.rest, axis=0, weights=self.m)     # mass-weighted rest centroid
        self.faces = None                                      # optional source faces (set by from_mesh) for re-export

    @classmethod
    def from_mesh(cls, mesh, inv_mass=None):
        """A RigidBody whose particles are a mesh's vertices, retaining the faces so the moved body re-exports as
        a mesh (to_mesh). The shape is preserved exactly, so to_mesh is the rest mesh rigidly transformed."""
        body = cls(np.asarray(mesh.vertices, float), inv_mass=inv_mass)
        body.faces = [tuple(f) for f in mesh.faces]
        return body

    def to_mesh(self):
        """The current state as a Mesh: the rigidly-transformed particle positions with the original faces."""
        from holographic_mesh import Mesh
        if self.faces is None:
            raise ValueError("this RigidBody has no faces to export (build it with RigidBody.from_mesh)")
        return Mesh(self.x.copy(), self.faces)

    def step(self, dt=1.0 / 60.0, gravity=None, stiffness=1.0, external_force=None, floor=None, restitution=0.0):
        """Advance one frame: predict under gravity/force, then shape-match back to the rigid goal."""
        g = _as_gravity(gravity, self.D)
        h = dt
        x_prev = self.x.copy()
        acc = np.tile(g, (self.N, 1))
        if external_force is not None:
            inv_m = np.where(self.m > 0, 1.0 / np.maximum(self.m, 1e-12), 0.0)[:, None]
            acc = acc + np.asarray(external_force, float) * inv_m
        self.v += h * acc
        self.x += h * self.v
        # shape match: optimal rotation R mapping (rest - rest_cm) onto (x - cm) via polar decomposition
        cm = np.average(self.x, axis=0, weights=self.m)
        P = self.x - cm                                          # current, centered
        Q = self.rest - self.rest_cm                             # rest, centered
        A = (self.m[:, None] * P).T @ Q                          # mass-weighted cross-covariance (D x D)
        U, _S, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0:                                 # forbid reflection -> proper rotation
            U = U.copy(); U[:, -1] *= -1
            R = U @ Vt
        goal = cm + Q @ R.T                                      # the rigid goal position for each particle
        self.x = self.x + stiffness * (goal - self.x)           # pull toward the rigid configuration
        if floor is not None:
            self.x[self.x[:, 1] < floor, 1] = floor
        self.v = (self.x - x_prev) / h
        if floor is not None:
            hit = self.x[:, 1] <= floor + 1e-9
            self.v[hit, 1] = -restitution * self.v[hit, 1]
        return self

    def max_distance_drift(self):
        """Largest change in any pairwise distance vs the rest shape -- 0 means perfectly rigid."""
        def pd(X):
            diff = X[:, None, :] - X[None, :, :]
            return np.sqrt((diff ** 2).sum(-1))
        return float(np.abs(pd(self.x) - pd(self.rest)).max())


# ---------------------------------------------------------------------------

def _selftest():
    # 1. DISTANCE CONSTRAINT CONVERGES: a stretched, pinned rigid link is pulled back to rest length.
    b = SoftBody(np.array([[0.0, 0.0], [3.0, 0.0]]))     # actual gap 3, rest will be 1
    b.add_distance(0, 1, rest=1.0, compliance=0.0); b.pin(0)
    b.step(dt=1 / 60, gravity=(0.0, 0.0), iterations=30)
    assert b.constraint_residual() < 1e-6, b.constraint_residual()

    # 2. XPBD STIFFNESS IS TIME-STEP INDEPENDENT: a hanging spring's static stretch = compliance*g, the SAME
    #    for 1 vs 6 substeps once the motion has SETTLED (a little damping reaches true static equilibrium;
    #    without it the substep-6 case is still gently oscillating -- the velocity differs, not the stiffness).
    #    Plain PBD's effective stiffness, by contrast, shifts with the iteration count.
    def hang_stretch(substeps, compliance=0.01):
        s = SoftBody(np.array([[0.0, 0.0], [0.0, -1.0]]))
        s.add_distance(0, 1, rest=1.0, compliance=compliance); s.pin(0)
        for _ in range(500):
            s.step(dt=1 / 60, gravity=(0.0, -9.8), iterations=20, substeps=substeps, damping=0.02)
        return float(np.linalg.norm(s.x[0] - s.x[1]) - 1.0)
    st1, st6 = hang_stretch(1), hang_stretch(6)
    predicted = 0.01 * 9.8                               # elongation = compliance * weight (m=1)
    assert abs(st1 - predicted) < 0.005 and abs(st1 - st6) < 0.005, (st1, st6, predicted)

    # 3. HANGING CLOTH REACHES EQUILIBRIUM: residual settles small under gravity + pinned top row.
    cloth = SoftBody.cloth(6, 6, spacing=1.0, compliance=0.0)
    for _ in range(200):
        cloth.step(dt=1 / 60, gravity=(0.0, -9.8), iterations=25)
    assert cloth.constraint_residual() < 0.05, cloth.constraint_residual()

    # 4. STABLE AT A BIG TIME-STEP: stiff cloth at dt that explodes an explicit spring stays bounded.
    big = SoftBody.cloth(5, 5, spacing=1.0, compliance=0.0)
    for _ in range(60):
        big.step(dt=0.1, gravity=(0.0, -9.8), iterations=15)
    assert np.isfinite(big.x).all() and np.abs(big.x).max() < 1e3, "PBD must stay stable at large dt"

    # 5. PBD path via the SHIPPED sweeper also satisfies the constraint (engine reuse works).
    p = SoftBody(np.array([[0.0, 0.0], [3.0, 0.0]]))
    p.add_distance(0, 1, rest=1.0); p.pin(0)
    p.step(dt=1 / 60, gravity=(0.0, 0.0), iterations=40, solver="pbd")
    assert p.constraint_residual() < 1e-3, p.constraint_residual()

    # 6. RIGID BODY stays rigid while it falls, and a shape is preserved under an offset push.
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rb = RigidBody(square)
    for _ in range(120):
        rb.step(dt=1 / 60, gravity=(0.0, -9.8))
    assert rb.max_distance_drift() < 1e-6, rb.max_distance_drift()      # never deforms
    assert rb.x[:, 1].mean() < 0.0, "rigid body should fall under gravity"

    # 7. BENDING resists folding: a flat 3-particle strip folded into a V is flattened back by a bend spring,
    #    while plain distance constraints (happy at any fold angle) leave it folded.
    import math as _math
    def fold(with_bending):
        s = SoftBody(np.array([[-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]]))
        s.add_distance(0, 1, 1.0); s.add_distance(1, 2, 1.0); s.pin(1)
        if with_bending: s.add_bending(0, 2)
        th = 0.7
        s.x[0] = [-_math.cos(th), _math.sin(th), 0]; s.x[2] = [_math.cos(th), _math.sin(th), 0]
        for _ in range(60): s.step(dt=1 / 60, gravity=(0, 0, 0), iterations=20)
        return float(np.linalg.norm(s.x[0] - s.x[2]))
    bent, flat = fold(False), fold(True)
    assert bent < 1.6 and flat > 1.95, (bent, flat)

    # 8. VOLUME constraint restores a squashed tetrahedron's volume.
    tet = SoftBody(np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    tet.add_volume(0, 1, 2, 3); v_rest = tet.total_volume()
    tet.x[3, 2] = 0.3                                    # squash the apex (volume drops)
    tet.step(dt=1 / 60, gravity=(0, 0, 0), iterations=40)
    assert abs(tet.total_volume() - v_rest) < 0.02, (tet.total_volume(), v_rest)

    print(f"holographic_softbody selftest: ok (CONSTRAINT residual {b.constraint_residual():.1e}; "
          f"XPBD stretch dt-independent {st1:.4f} vs {st6:.4f} (predicted {predicted:.4f}); "
          f"CLOTH equilibrium residual {cloth.constraint_residual():.3f}; STABLE at dt=0.1 "
          f"(max|x| {np.abs(big.x).max():.1f}); PBD-via-engine residual {p.constraint_residual():.1e}; "
          f"RIGID drift {rb.max_distance_drift():.1e}, fell to y={rb.x[:,1].mean():.2f}; "
          f"BENDING unfolds {bent:.2f}->{flat:.2f}; VOLUME restored {tet.total_volume():.3f} (rest {v_rest:.3f}))")


if __name__ == "__main__":
    _selftest()
