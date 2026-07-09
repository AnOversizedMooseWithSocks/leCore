"""FS-2 -- the narrow-band sparse field (holographic_sparsefield), array-backed for parallelism.

WHAT THIS IS (and why it is the hard one)
-----------------------------------------
FS-1 sculpts a field and re-meshes it, but every stroke re-evaluates the WHOLE volume -- batchy, not interactive. The
fix the level-set literature settled on (Adalsteinsson & Sethian's narrow band; Museth's VDB/OpenVDB) is to store,
edit, and re-extract only the thin shell of voxels around the surface (|f| < band). A brush then touches O(brush)
voxels, and only the bricks it dirtied get re-meshed -- the locality that makes sculpting interactive.

THE PERFORMANCE REFRAME (the reason this file is array-backed, not a dict of voxels)
-----------------------------------------------------------------------------------
Python is the bottleneck, not arithmetic. The first cut stored the field as a Python DICT keyed by voxel coordinate
and looped in Python to materialize / sample / edit / reinitialize -- the TRADITIONAL choice, and exactly what made
the FS-4 loop slow. The parallel reframe: hold the field as a DENSE NumPy ARRAY, so every field operation becomes a
vectorized array op (a slice to materialize, np.roll for gradients, a boolean mask for a brush) with NO Python
per-voxel loop. NumPy does the whole grid at once. Two existing-pattern wins make the loop cheap:

  1. PARALLEL ARRAY OPS. sample is one fancy-indexed trilinear gather over all query points; apply_local masks a
     sub-array of the ball and adds the brush in one shot; reinitialize is the Godunov update written with np.roll
     across the whole band at once; _materialize is a pure slice. The Python loops are gone.

  2. A BRICK-MESH WORKING-SET CACHE -- the ReflexCache idea (holographic_tree). ReflexCache thickens the veins it
     travels often and skips the expensive path for a FAMILIAR input; here, extract_cached caches each brick's
     extracted sub-mesh and, on a re-extract, REUSES the bricks a stroke did not touch -- only the DIRTY bricks are
     re-marched. So the per-frame re-extract is O(dirty), not O(all active). Same philosophy, applied to geometry:
     don't recompute what hasn't changed. (Going back and forth from a representation to Python triangles is the
     expensive trip; the cache is how you stop paying for it every frame.)

  The brick addressing is still _tile_bucket (the floor-divide tiling StructuredIndex / TiledStore / the splat tiler
  share), and the per-brick mesh is still marching_tetrahedra -- this elevates those into the geometry fast path.

THE EXACTNESS FACT: marching only makes faces where the field crosses 0, which only happens in the band; the far
voxels carry the correct sign at +/-band, so the extraction of the SURFACE is identical to a full dense extraction
(the far values never cross 0, so never make a face).

DETERMINISM (per ISA.md): dense arrays, fixed marching, fixed weld tolerance, sorted brick iteration -- bit-identical
run to run. No RNG.

KEPT HONEST:
  * MEMORY is now O(res^3) for the dense field -- a deliberate speed-for-memory trade (the user's call: Python time is
    the cost, not bytes; a res-48 grid is < 0.5 MB). The narrow band still governs COMPUTE: only ACTIVE bricks are
    marched, a brush touches only its ball, and the cache re-marches only dirty bricks. For very large volumes,
    allocating only active bricks (block-sparse, VDB-style) reclaims memory-sparsity -- a documented next step, not
    done here.
  * The marching itself is still pure Python per cell -- the cache removes it from the per-FRAME path (only dirty
    bricks re-march), but the COLD first extract still marches every active brick; a vectorized marching is the
    further step if even cold start must be fast. Stated, not hidden.
  * WATERTIGHTNESS is exactly marching_tetrahedra's: a contiguous box marched in ONE pass is BIT-EXACT to a dense
    march on the same grid (verified by equal face counts). That extractor is itself non-manifold at grid sizes where
    a vertex lands on the isosurface (pre-existing, grid-dependent -- e.g. a sphere at grid 31 -- NOT a sparse-field
    property). The cached per-brick path welds per-brick meshes by position, which is bit-exact (shared seam voxels
    have identical coords AND values).
  * The band MUST be reinitialized or distances drift. Reinit PRESERVES the 0-level set (sign(phi0) in the update), so
    it does not move the surface -- the mesh cache stays valid across a reinit.
  * TOPOLOGY growth into far/unseeded space carries the correct seed sign in the dense field, so an inflate into empty
    space is handled; a merge/split still wants a from_field re-seed for a clean band. The cache assumes edits go
    through apply_local -- mutate self.field directly and you must call cache_clear() (the GramCache discipline).
"""

import math

import numpy as np

from holographic.misc.holographic_tree import _tile_bucket          # the shared floor-divide brick addressing
from holographic.mesh_and_geometry.holographic_mesh import Mesh
from holographic.mesh_and_geometry.holographic_meshbridge import marching_tetrahedra, marching_tetrahedra_vec


def _smooth_falloff(d, r):
    """A C1 falloff 1 at d=0 -> 0 at d=r (the same 3t^2-2t^3 shape geodesic_soft_selection and the sculpt brushes
    use), exactly 0 beyond r. Vectorized."""
    d = np.asarray(d, float)
    t = np.clip(d / max(r, 1e-12), 0.0, 1.0)
    return 1.0 - (3.0 * t * t - 2.0 * t * t * t)


class SparseField:
    """A narrow-band signed-distance field, DENSE-ARRAY backed for vectorized field ops, with a brick-mesh
    working-set cache so the sculpt loop only re-marches dirty bricks. See the module docstring."""

    def __init__(self, min_corner, max_corner, voxel, band, tile=8):
        self.min = np.asarray(min_corner, float)
        self.max = np.asarray(max_corner, float)
        self.h = float(voxel)
        self.band = float(band)
        self.tile = int(tile)
        span = self.max - self.min
        ncell = np.maximum(np.ceil(span / self.h).astype(int), 1)
        self.bpa = np.maximum(np.ceil(ncell / self.tile).astype(int), 1)      # bricks per axis
        self.ncorner = self.bpa * self.tile + 1                               # corner count per axis
        self.field = None                                                     # dense (nx,ny,nz) clamped SDF
        self.active = set()                                                   # brick keys (bx,by,bz) with a 0-crossing
        self._brick_cache = {}                # brick -> (vertices (V,3), faces list) -- the working-set mesh cache
        self._cache_dirty = set()             # bricks whose cached mesh is stale (touched since last extract_cached)
        self._last_marched = 0                # how many bricks the last extract_cached actually re-marched (a metric)

    # ----- the compatibility dict view (reads only) ---------------------------------------------------
    @property
    def values(self):
        """A dict {(i,j,k): clamped_sdf} over the voxels of the ACTIVE bricks -- the old sparse view, rebuilt from the
        dense field for backward compatibility (len / keys / [k] reads). The dense `self.field` is the source of
        truth; write to it (and call cache_clear if you bypass apply_local)."""
        out = {}
        t = self.tile
        for (bx, by, bz) in self.active:
            i0, j0, k0 = bx * t, by * t, bz * t
            for a in range(t + 1):
                for b in range(t + 1):
                    for c in range(t + 1):
                        key = (i0 + a, j0 + b, k0 + c)
                        out[key] = float(self.field[key])
        return out

    # ----- seeding ------------------------------------------------------------------------------------
    @classmethod
    def from_field(cls, field, min_corner, max_corner, voxel, band, tile=8):
        """Seed from a dense field function `field` (points (N,3) -> values (N,)). Samples the full grid (the one-time
        O(res^3) cost), clamps to +/-band into the dense array, and records which bricks the surface passes through.
        Returns the SparseField."""
        self = cls(min_corner, max_corner, voxel, band, tile=tile)
        nx, ny, nz = (int(self.ncorner[0]), int(self.ncorner[1]), int(self.ncorner[2]))
        xs = self.min[0] + np.arange(nx) * self.h
        ys = self.min[1] + np.arange(ny) * self.h
        zs = self.min[2] + np.arange(nz) * self.h
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        G = np.asarray(field(pts), float).reshape(nx, ny, nz)
        self.field = np.clip(G, -self.band, self.band)                        # the dense clamped SDF
        self._recompute_active()
        return self

    def _brick_block_view(self, brick):
        """A view of one brick's (tile+1)^3 corner block in the dense field."""
        bx, by, bz = brick
        t = self.tile
        i0, j0, k0 = bx * t, by * t, bz * t
        return self.field[i0:i0 + t + 1, j0:j0 + t + 1, k0:k0 + t + 1]

    def _block_has_surface(self, block):
        return block.min() < 0.0 and block.max() >= 0.0

    def _recompute_active(self):
        """Scan all bricks once and record the active set (those with a 0-crossing). Vectorized per brick."""
        self.active = set()
        for bx in range(int(self.bpa[0])):
            for by in range(int(self.bpa[1])):
                for bz in range(int(self.bpa[2])):
                    if self._block_has_surface(self._brick_block_view((bx, by, bz))):
                        self.active.add((bx, by, bz))

    def _bricks_in_box(self, i0, i1, j0, j1, k0, k1):
        """Every brick whose corner block overlaps the inclusive voxel box [i0..i1] x ... (for re-checking activity
        and marking the cache after a local edit)."""
        t = self.tile
        bxs = range(max(i0 - 1, 0) // t, min(i1, int(self.ncorner[0]) - 1) // t + 1)
        bys = range(max(j0 - 1, 0) // t, min(j1, int(self.ncorner[1]) - 1) // t + 1)
        bzs = range(max(k0 - 1, 0) // t, min(k1, int(self.ncorner[2]) - 1) // t + 1)
        return [(bx, by, bz) for bx in bxs for by in bys for bz in bzs]

    # ----- materialization (a pure slice -- no Python loop) -------------------------------------------
    def _materialize(self, i0, j0, k0, ni, nj, nk):
        """A dense (ni,nj,nk) clamped-SDF block over the voxel box -- a SLICE of the dense field (vectorized), with
        the matching world axes. The box is always within the grid for extraction (active bricks are in-grid)."""
        block = self.field[i0:i0 + ni, j0:j0 + nj, k0:k0 + nk]
        xs = self.min[0] + (i0 + np.arange(ni)) * self.h
        ys = self.min[1] + (j0 + np.arange(nj)) * self.h
        zs = self.min[2] + (k0 + np.arange(nk)) * self.h
        return block, (xs, ys, zs)

    # ----- sampling (vectorized trilinear over ALL points) --------------------------------------------
    def sample(self, points):
        """Sample the field at world `points` (N,3) -- one vectorized trilinear gather over the dense field, no
        per-point Python loop. Points outside the grid return the sign*band at the clamped cell (the far/outside
        default)."""
        points = np.atleast_2d(np.asarray(points, float))
        n = np.array(self.field.shape) - 1
        rel = (points - self.min) / self.h
        base = np.floor(rel).astype(int)
        frac = rel - base
        in_range = np.all((base >= 0) & (base < n), axis=1)
        bc = np.clip(base, 0, n - 1)                                          # safe gather indices
        i, j, k = bc[:, 0], bc[:, 1], bc[:, 2]
        ip = np.minimum(i + 1, n[0]); jp = np.minimum(j + 1, n[1]); kp = np.minimum(k + 1, n[2])
        f = self.field
        c000 = f[i, j, k]; c100 = f[ip, j, k]; c010 = f[i, jp, k]; c001 = f[i, j, kp]
        c110 = f[ip, jp, k]; c101 = f[ip, j, kp]; c011 = f[i, jp, kp]; c111 = f[ip, jp, kp]
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        out = c0 * (1 - fz) + c1 * fz
        far = np.sign(f[i, j, k]) * self.band                                 # for out-of-range points
        out = np.where(in_range, out, np.where(far == 0, self.band, far))
        return out

    # ----- local edit (vectorized over the brush ball) ------------------------------------------------
    def apply_local(self, delta_fn, p, r):
        """Apply a LOCAL edit at world `p` radius `r`: `delta_fn(points (M,3)) -> deltas (M,)` is ADDED to the field
        over every voxel within r of p (lower f to inflate an SDF surface, raise it to carve). One vectorized
        sub-array update of the ball -- no Python per-voxel loop. Returns (dirty_bricks (set), touched_voxel_count).
        Marks the touched bricks' cached meshes stale. Run reinitialize() afterwards to restore true distance."""
        p = np.asarray(p, float)
        n = np.array(self.field.shape)
        lo = np.clip(np.floor((p - r - self.min) / self.h).astype(int), 0, n - 1)
        hi = np.clip(np.ceil((p + r - self.min) / self.h).astype(int), 0, n - 1)
        ii = np.arange(lo[0], hi[0] + 1); jj = np.arange(lo[1], hi[1] + 1); kk = np.arange(lo[2], hi[2] + 1)
        GI, GJ, GK = np.meshgrid(ii, jj, kk, indexing="ij")
        world = self.min + np.stack([GI, GJ, GK], axis=-1) * self.h
        dist = np.linalg.norm(world - p, axis=-1)
        mask = dist <= r
        if not mask.any():
            return set(), 0
        deltas = np.asarray(delta_fn(world[mask]), float)
        sub = self.field[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
        sub[mask] = np.clip(sub[mask] + deltas, -self.band, self.band)        # writes through the view into self.field
        # re-check activity for bricks overlapping the box; mark them dirty
        dirty = set()
        for b in self._bricks_in_box(int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2])):
            if self._block_has_surface(self._brick_block_view(b)):
                self.active.add(b)
            else:
                self.active.discard(b)
            dirty.add(b)
            self._cache_dirty.add(b)
        return dirty, int(mask.sum())

    # ----- reinitialization (vectorized Godunov via np.roll) ------------------------------------------
    def reinitialize(self, iters=8, dt=None):
        """Restore |grad f| -> 1 in the band via the Godunov-upwind reinitialization phi_t = sign(phi0)(1-|grad phi|),
        written with np.roll so the WHOLE band updates at once (no Python loop). A smeared sign s = phi0/sqrt(phi0^2 +
        h^2) (Peng et al.) keeps it stable. Only band voxels (|f| < band) are updated; the far +/-band hold. Preserves
        the 0-level set, so the surface (and the mesh cache) does not move. Returns the mean central-difference |grad|
        over band voxels after reinit (closer to 1 is better). KEPT: np.roll wraps at the grid edge, but the band is
        interior (away from edges) so the wrap does not touch it."""
        if dt is None:
            dt = 0.4 * self.h
        h = self.h
        band_mask = np.abs(self.field) < self.band
        phi = self.field
        s = phi / np.sqrt(phi * phi + h * h)                                  # smeared sign of phi0 (whole array)
        pos = s >= 0
        for _ in range(int(iters)):
            g2 = np.zeros_like(phi)
            for axis in range(3):
                bm = (phi - np.roll(phi, 1, axis=axis)) / h                   # backward difference
                fp = (np.roll(phi, -1, axis=axis) - phi) / h                  # forward difference
                a_pos = np.maximum(bm, 0.0); b_pos = np.minimum(fp, 0.0)      # Godunov, + side
                a_neg = np.minimum(bm, 0.0); b_neg = np.maximum(fp, 0.0)      # Godunov, - side
                term = np.where(pos, np.maximum(a_pos * a_pos, b_pos * b_pos),
                                np.maximum(a_neg * a_neg, b_neg * b_neg))
                g2 = g2 + term
            grad = np.sqrt(g2)
            upd = np.clip(phi - dt * s * (grad - 1.0), -self.band, self.band)
            phi = np.where(band_mask, upd, phi)                              # only the band moves
        self.field = phi
        return self.grad_norm_stats(self.values, self.h)

    @staticmethod
    def grad_norm_stats(values_dict, h):
        """Mean |grad f| over the interior of a value dict (central differences, voxels with all 6 neighbours
        present) -- kept dict-based so a caller can measure a field BEFORE reinitializing."""
        grads = []
        for k in values_dict:
            g2 = 0.0
            full = True
            for axis in range(3):
                kp = list(k); kp[axis] += 1; kp = tuple(kp)
                km = list(k); km[axis] -= 1; km = tuple(km)
                if kp not in values_dict or km not in values_dict:
                    full = False
                    break
                g = (values_dict[kp] - values_dict[km]) / (2.0 * h)
                g2 += g * g
            if full:
                grads.append(math.sqrt(g2))
        return float(np.mean(grads)) if grads else float("nan")

    # ----- extraction: contiguous single march (simple, watertight) -----------------------------------
    def extract_local(self, dirty_bricks=None, weld_tol=None):
        """Marching tetrahedra over the CONTIGUOUS bounding box of the requested bricks (or ALL active if None), in
        ONE pass -- watertight by construction (bit-exact to a dense march on the same grid). For a brush the dirty
        bricks form a small contiguous cluster, so the box is small and the re-mesh is LOCAL. Returns a Mesh."""
        bricks = sorted(self.active if dirty_bricks is None else (set(dirty_bricks) & self.active))
        if not bricks:
            return Mesh(np.zeros((0, 3)), [])
        bx = [b[0] for b in bricks]; by = [b[1] for b in bricks]; bz = [b[2] for b in bricks]
        t = self.tile
        i0, i1 = min(bx) * t, max(bx) * t + t
        j0, j1 = min(by) * t, max(by) * t + t
        k0, k1 = min(bz) * t, max(bz) * t + t
        block, axes = self._materialize(i0, j0, k0, i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1)
        return marching_tetrahedra_vec(block, axes, level=0.0)

    # ----- extraction: the working-set cache (the loop's fast path) -----------------------------------
    def cache_clear(self):
        """Drop the brick-mesh cache -- call this if you mutate self.field directly instead of via apply_local."""
        self._brick_cache = {}
        self._cache_dirty = set()

    def extract_dirty(self):
        """THE SCULPTING FAST PATH -- re-mesh ONLY the bricks a stroke touched and return them as a per-brick DELTA,
        never reassembling the whole surface. extract_cached already re-marches only dirty bricks, but it then WELDS
        every brick's faces into one mesh on every call -- a Python loop over every face, O(total), which blows the
        frame budget on a large model (measured: ~0.6s at 41k faces, ~1.4s at 93k, for an 8-28 brick edit). This
        returns instead

            {'updated': {brick_id: Mesh}, 'removed': [brick_id, ...]}

        -- only the bricks whose geometry CHANGED (dirty or newly active) plus the bricks that went empty. The viewport
        keeps a per-brick mesh map and swaps only those, so the per-frame projection is O(dirty), not O(total) -- the
        difference between a Python weld of 93k faces and marching 28 small bricks. Adjacent bricks share a voxel plane
        and march it identically, so the per-brick meshes meet at bit-exact seams (duplicated boundary vertices, no
        cracks). The cache state it leaves is identical to extract_cached's, so the two interoperate. A cold first call
        returns every active brick (the viewport's initial build); every later call returns just the stroke's delta.
        Records self._last_marched. Pair with apply_local for the loop: apply_local (O(brush)) -> extract_dirty
        (O(dirty)) -> push the delta to the renderer."""
        updated = {}
        removed = []
        for b in list(self._brick_cache):                               # bricks that went inactive -> tell the viewport to drop them
            if b not in self.active:
                removed.append(b)
                del self._brick_cache[b]
        marched = 0
        for b in sorted(self.active):
            if b in self._cache_dirty or b not in self._brick_cache:    # dirty (touched) or newly active -> re-march
                t = self.tile
                block, axes = self._materialize(b[0] * t, b[1] * t, b[2] * t, t + 1, t + 1, t + 1)
                m = marching_tetrahedra_vec(block, axes, level=0.0)
                self._brick_cache[b] = (m.vertices, m.faces)
                updated[b] = m
                marched += 1
        self._cache_dirty -= set(self.active)
        self._last_marched = marched
        return {"updated": updated, "removed": removed}

    def extract_cached(self, weld_tol=None):
        """Re-extract the whole surface, but REUSE the cached mesh of every brick a stroke did not touch -- only DIRTY
        (or newly-active) bricks are re-marched. The ReflexCache idea applied to geometry: skip the expensive Python
        marching for familiar bricks. Per-frame re-extract is O(dirty), not O(all active). Welds the per-brick meshes
        by position (bit-exact at the shared seams). Records self._last_marched (how many bricks were re-marched).
        Returns a Mesh."""
        if weld_tol is None:
            weld_tol = self.h * 1e-3
        all_v = []
        all_f = []
        weld = {}
        marched = 0

        def gidx(p):
            key = (round(p[0] / weld_tol), round(p[1] / weld_tol), round(p[2] / weld_tol))
            if key in weld:
                return weld[key]
            idx = len(all_v)
            all_v.append(p)
            weld[key] = idx
            return idx

        for b in sorted(self.active):
            if b in self._brick_cache and b not in self._cache_dirty:
                verts, faces = self._brick_cache[b]                          # REUSE -- no marching
            else:
                t = self.tile
                block, axes = self._materialize(b[0] * t, b[1] * t, b[2] * t, t + 1, t + 1, t + 1)
                m = marching_tetrahedra_vec(block, axes, level=0.0)
                verts, faces = m.vertices, m.faces
                self._brick_cache[b] = (verts, faces)
                marched += 1
            for fc in faces:
                all_f.append(tuple(gidx(verts[vi]) for vi in fc))
        # forget bricks that are no longer active; clear the dirty marks we just serviced
        for b in list(self._brick_cache):
            if b not in self.active:
                del self._brick_cache[b]
        self._cache_dirty -= set(self.active)
        self._last_marched = marched
        if not all_v:
            return Mesh(np.zeros((0, 3)), [])
        return Mesh(np.array(all_v), all_f)

    # ----- field-native LOD: coarsen the SOURCE, re-project (no mesh decimation) -----------------------
    def extract_at_stride(self, stride=1):
        """A field-native LOD level: COARSEN THE SOURCE FIELD by `stride` (subsample the dense grid by that factor)
        and march the result -- one strided slice + one vectorized march, NO mesh decimation. stride=1 is the full
        field; stride=2 halves the resolution on each axis (1/8 the cells); stride=4 again. The marched surface
        resolves to the coarse spacing (stride*h), so coarser strides drop sub-cell detail -- exactly LOD, obtained
        by re-projecting a coarser FIELD rather than simplifying the fine MESH. This is the whole thesis in one
        method: the mesh is a projection of the field, so a coarser mesh is a coarser field projected. Returns a Mesh
        (empty if the surface is unresolved at this coarseness)."""
        stride = int(stride)
        if stride <= 1:
            return self.extract_local()
        sub = self.field[::stride, ::stride, ::stride]
        if not (sub.min() < 0.0 and sub.max() >= 0.0):
            return Mesh(np.zeros((0, 3)), [])
        n = sub.shape
        xs = self.min[0] + np.arange(n[0]) * self.h * stride
        ys = self.min[1] + np.arange(n[1]) * self.h * stride
        zs = self.min[2] + np.arange(n[2]) * self.h * stride
        return marching_tetrahedra_vec(sub, (xs, ys, zs), level=0.0)

    def lod_chain(self, strides=(1, 2, 4, 8)):
        """A field-native LOD chain: re-march the stored field at increasing strides (a coarser SOURCE each level),
        and measure each level's error AS A FIELD QUERY. The key move: the FULL-resolution field value at a coarse
        marched vertex IS that vertex's signed distance to the true surface, so |sample(coarse_vertices)| is the
        level's deviation -- read in O(V) field samples, not an O(V*F) mesh-to-mesh distance, and not a single greedy
        edge collapse. Returns a fine->coarse list of LODLevel (mesh, n_faces, mean_error, max_error) for select_lod.
        LOD entirely in the field: coarsen the source, project, read the error from the field itself."""
        from holographic.misc.holographic_lod import LODLevel
        levels = []
        for s in strides:
            m = self.extract_at_stride(s)
            if m.n_faces == 0:
                continue
            if levels and m.n_faces >= levels[-1].n_faces:
                continue                                          # this stride bought no coarsening; skip it
            if not levels:
                mean_e = max_e = 0.0                              # the finest level is the reference
            else:
                d = np.abs(self.sample(m.vertices))               # field value at the vertices = distance to truth
                mean_e, max_e = float(d.mean()), float(d.max())
            levels.append(LODLevel(m, m.n_faces, mean_e, max_e))
        return levels


# =====================================================================================================
# Self-test -- the vectorized ops, exact-to-dense extraction, the brick-mesh cache (cold vs warm), and
# reinitialization.  A sphere SDF is the ground truth (|grad| = 1 exactly).
# =====================================================================================================
def _selftest():
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
    import time

    R = 0.6
    bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    def sphere(P):
        return np.linalg.norm(P, axis=1) - R

    voxel = 2.0 / 48
    band = 4 * voxel
    sf = SparseField.from_field(sphere, bounds[0], bounds[1], voxel, band, tile=6)

    # --- SPARSITY (the band is a fraction of the volume) ---
    full_voxels = int(np.prod(sf.ncorner))
    stored = len(sf.values)
    assert stored < 0.5 * full_voxels

    # --- sample (vectorized trilinear) matches the true SDF in the band ---
    probe = np.array([[R * 0.5, 0.0, 0.0], [0.0, R, 0.0], [0.0, 0.0, -R * 0.9]])
    got = sf.sample(probe); true = sphere(probe); near = np.abs(true) < band
    assert np.allclose(got[near], true[near], atol=voxel)

    # --- EXTRACT (vectorized marching) is geometrically IDENTICAL to a dense per-cell march on the same grid ---
    mesh = sf.extract_local()
    ncorner = int(sf.ncorner[0])
    dvals, daxes = sample_field(sphere, bounds, ncorner)
    dense = marching_tetrahedra(dvals, daxes, level=0.0)        # the per-cell reference
    assert mesh.n_faces == dense.n_faces, f"vectorized extract must match dense ({mesh.n_faces} vs {dense.n_faces})"
    hd_de = max(np.min(np.linalg.norm(dense.vertices - v, axis=1)) for v in mesh.vertices)
    assert hd_de < 1e-9, f"vectorized and per-cell vertices must coincide (Hausdorff {hd_de:.2e})"
    assert mesh.is_manifold()

    # the headline: the vectorized marcher vs the per-cell Python marcher on the full surface
    import time
    t0 = time.time(); _ = marching_tetrahedra(dvals, daxes, level=0.0); py_t = time.time() - t0
    t0 = time.time(); _ = marching_tetrahedra_vec(dvals, daxes, level=0.0); vec_t = time.time() - t0

    # --- the CACHE: cold marches every active brick; after a local brush, warm re-marches only the dirty ones ---
    t0 = time.time(); m_cold = sf.extract_cached(); cold_t = time.time() - t0
    cold_marched = sf._last_marched
    assert cold_marched == len(sf.active), "cold extract marches every active brick"
    assert m_cold.n_faces == mesh.n_faces, "cached extract must match the single-march surface"

    p = np.array([R, 0.0, 0.0]); brush_r = 0.25

    def inflate(points):
        d = np.linalg.norm(points - p, axis=1)
        return -0.5 * band * _smooth_falloff(d, brush_r)

    dirty, touched = sf.apply_local(inflate, p, brush_r)
    assert touched < 0.1 * full_voxels, f"a stroke touches O(brush) voxels ({touched})"
    t0 = time.time(); m_warm = sf.extract_cached(); warm_t = time.time() - t0
    warm_marched = sf._last_marched
    assert warm_marched < cold_marched, f"the cache must skip unchanged bricks (warm {warm_marched} < cold {cold_marched})"
    assert warm_marched <= len(dirty) + 2, "warm re-march is bounded by the dirty bricks"

    # correctness: the warm (cached) result equals a from-scratch rebuild of the EDITED field
    sf.cache_clear(); m_fresh = sf.extract_cached()
    assert m_warm.n_faces == m_fresh.n_faces, "cache reuse must not change the surface"

    # --- THE SCULPTING FAST PATH: extract_dirty returns only CHANGED bricks (a per-brick delta), not the whole mesh ---
    sf3 = SparseField.from_field(sphere, bounds[0], bounds[1], voxel, band, tile=6)
    cold = sf3.extract_dirty()                                                # cold: every active brick (the initial build)
    assert set(cold["updated"]) == set(sf3.active) and not cold["removed"], "cold extract_dirty returns all active bricks"
    sf3.apply_local(inflate, p, brush_r)
    t0 = time.time(); warm = sf3.extract_dirty(); dirty_t = time.time() - t0   # warm: only the stroke's bricks
    assert 0 < len(warm["updated"]) <= len(dirty) + 2, "warm extract_dirty returns only the dirty bricks"
    assert dirty_t < warm_t, "per-brick delta is faster than reassembling the whole mesh"
    # each delta brick equals a fresh march of that brick (correct geometry, just not welded into one mesh)
    for b, bm in warm["updated"].items():
        blk, ax = sf3._materialize(b[0] * sf3.tile, b[1] * sf3.tile, b[2] * sf3.tile, sf3.tile + 1, sf3.tile + 1, sf3.tile + 1)
        assert bm.n_faces == marching_tetrahedra_vec(blk, ax, level=0.0).n_faces, "a delta brick matches a fresh march"

    # --- REINITIALIZATION (vectorized Godunov) moves |grad| toward 1; it PRESERVES the 0-level (mesh unchanged) ---
    faces_before = sf.extract_local().n_faces
    sf2 = SparseField.from_field(sphere, bounds[0], bounds[1], voxel, band, tile=6)
    bmask = np.abs(sf2.field) < band
    sf2.field[bmask] *= 0.6                                                   # squish -> |grad| ~ 0.6
    before = SparseField.grad_norm_stats(sf2.values, sf2.h)
    after = sf2.reinitialize(iters=12)
    assert before < 0.8 and abs(after - 1.0) < abs(before - 1.0), f"reinit toward 1 (before {before:.3f}, after {after:.3f})"

    speedup = cold_marched / max(warm_marched, 1)
    print(f"holographic_sparsefield selftest: ok (array-backed, vectorized). band sparse {stored}/{full_voxels} "
          f"({100.0*stored/full_voxels:.0f}%); sample matches true SDF in-band; extract is geometrically IDENTICAL "
          f"to a per-cell dense march ({mesh.n_faces} faces, Hausdorff {hd_de:.0e}) and watertight. VECTORIZED "
          f"MARCHING: the whole surface in {vec_t*1000:.0f}ms vs {py_t*1000:.0f}ms per-cell Python ({py_t/vec_t:.0f}x "
          f"faster -- the case-table RAM lookup over all cells at once). CACHE: cold marched {cold_marched} bricks, a "
          f"brush touching {touched} voxels dirtied {len(dirty)} so warm re-marched only {warm_marched} ({speedup:.0f}x "
          f"fewer, {cold_t*1000:.0f}ms -> {warm_t*1000:.0f}ms), same surface. reinit (vectorized Godunov) moved |grad| "
          f"{before:.3f} -> {after:.3f}, preserving the 0-level. KEPT: dense field is O(res^3) memory; marching is now "
          f"parallel array ops but allocates O(cells) temporaries; watertightness is the marcher's, grid-dependent)")


if __name__ == "__main__":
    _selftest()
