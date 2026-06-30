"""Bidirectional ray<->object index (RAYIDX): record which objects each camera ray TOUCHED along its path, so an edit
re-shades only the rays that can possibly change -- including through glass / in reflections.

THE IDEA
--------
A renderer already discovers, per ray, every object the ray touches on its way to the eye -- the primary hit, and the
objects seen through glass or in a reflection on the secondary rays. Traditionally that information is thrown away at
the end of the frame and re-gathered from scratch next frame. But it is exactly the information a DELTA update needs:
if object X changes (its colour, its material, its lighting), the only pixels that can change are the ones whose ray
TOUCHED X somewhere along its path. So record it once as a bidirectional index --

    touched[pixel, object] = True   iff   that ray hit that object on primary OR secondary (through-glass) segments

-- and on any edit, `pixels_touching(changed_objects)` returns the exact set of pixels to re-shade. Everything else is
reused. This closes the gap the incremental SceneRenderer admits it has: "through glass or in reflections is not
incrementally refreshed". The same index generalises to specular reflection bounces (record the reflected-ray hit) and
to spatial bricks (record bricks instead of object ids) for region edits.

WHAT IT BUYS
------------
For a localized edit, re-shade O(touched pixels) instead of the whole frame -- and crucially the touched set INCLUDES
the indirect pixels (an object seen through glass updates when it changes, even though it is not the primary hit there).
Because the engine's `_shade_rays` is deterministic, the delta re-shade is BIT-EXACT against a full re-render on the
affected pixels (verified in the tests / tour).

HONEST SCOPE (kept negatives)
-----------------------------
  * A MATERIAL/colour/light edit leaves geometry unchanged, so the index stays valid and the delta is exact. A MOVE
    changes geometry, so the index must be rebuilt for the affected region first (the engine's dirty-region story);
    the index makes the RE-SHADE cheap, not the geometry re-trace free.
  * Built here for the primary + glass-secondary segments that `render_scene` actually traces (its only object-to-
    object secondary ray). render_scene's mirror term reflects the SKY, not objects, so a mirror-object-reflection
    demo would need object reflection added to the shader -- the index pattern is identical (record the reflected hit).
NumPy/stdlib only, deterministic.
"""
import numpy as np


class RayPathIndex:
    """Per-pixel set of object ids the ray touched (primary + secondary). `pixels_touching(ids)` is the reverse map:
    object -> the pixels to re-shade when it changes."""

    def __init__(self, touched, primary, width, height):
        self.touched = touched                 # (npix, n_obj) bool: ray r touched object o anywhere on its path
        self.primary = primary                 # (npix,) int primary-hit id (-1 = background)
        self.width = int(width); self.height = int(height)

    def pixels_touching(self, ids):
        """Boolean (npix,) mask of pixels whose ray touched ANY of `ids` (the edit's affected pixels)."""
        ids = np.atleast_1d(np.asarray(ids, int))
        ids = ids[(ids >= 0) & (ids < self.touched.shape[1])]   # ignore ids no object uses
        if self.touched.shape[1] == 0 or len(ids) == 0:
            return np.zeros(self.touched.shape[0], bool)
        return self.touched[:, ids].any(axis=1)

    def indirect_pixels(self, obj_id):
        """Pixels that touch `obj_id` only on a SECONDARY ray (seen through glass / in a reflection), not as the
        primary hit -- the pixels a primary-id-only incremental renderer would MISS."""
        touch = self.pixels_touching(obj_id)
        return touch & (self.primary != obj_id)


def build_ray_index(ctx, camera, width, height):
    """Trace the scene's geometry (primary + glass-secondary rays) and record, per pixel, every object id the ray
    touched. Reuses the same ctx (`_scene_setup`) and the same traces the deterministic renderer uses, so the recorded
    paths match what gets shaded. Returns a RayPathIndex."""
    from holographic_raymarch import sphere_trace, sdf_normal
    from holographic_semantic import _UnionSDF
    union = ctx["union"]; sdfs = ctx["sdfs"]; refl = ctx["refl"]
    see_through = ctx["is_glass"] | ctx.get("is_translucent", np.zeros(len(sdfs), bool))   # glass OR frosted
    eye, dirs = camera.ray_dirs(width, height)
    D = dirs.reshape(-1, 3)
    O = np.broadcast_to(eye, D.shape).astype(float).copy()
    npix = len(D); nobj = len(sdfs)
    touched = np.zeros((npix, nobj), bool)
    primary = -np.ones(npix, int)
    hit, t, P = sphere_trace(union, O, D)
    if np.any(hit):
        idx_hit = np.where(hit)[0]
        ids = union.ids(P[hit]); primary[idx_hit] = ids
        touched[idx_hit, ids] = True                            # primary segment
        Nh = sdf_normal(union, P[hit])
        st_here = see_through[ids]
        if np.any(st_here) and not np.all(see_through):         # see-through secondary: object behind glass/frosted
            ng_i = [i for i in range(nobj) if not see_through[i]]
            un2 = _UnionSDF([sdfs[i] for i in ng_i])
            gidx = idx_hit[st_here]
            Pg = P[gidx]; Dg = D[gidx]
            h2, t2, P2 = sphere_trace(un2, Pg + Dg * 6e-3, Dg)
            if np.any(h2):
                behind = np.array(ng_i)[un2.ids(P2[h2])]        # local opaque id -> global object id
                touched[gidx[h2], behind] = True
        mirror = refl[ids] > 0.05                               # reflection segment: object seen IN a reflection
        if np.any(mirror):
            midx = idx_hit[mirror]
            Dm = D[midx] - 2.0 * (D[midx] * Nh[mirror]).sum(1)[:, None] * Nh[mirror]
            Pm = P[hit][mirror] + Nh[mirror] * 3e-3
            hm, tm, Pmh = sphere_trace(union, Pm, Dm)
            if np.any(hm):
                touched[midx[hm], union.ids(Pmh[hm])] = True
    return RayPathIndex(touched, primary, width, height)


def delta_reshade(ctx_new, index, changed_ids, base_frame, camera):
    """Re-shade ONLY the pixels whose ray touched a changed object, using the new scene state, and composite into the
    cached `base_frame`. Returns (updated_frame, mask). Deterministic -> the updated pixels are bit-exact vs a full
    re-render. `ctx_new` is `_scene_setup` of the edited scene (same geometry for a material/colour/light edit)."""
    from holographic_semantic import _shade_rays
    H, W = base_frame.shape[:2]
    mask = index.pixels_touching(changed_ids)
    out = base_frame.copy().reshape(-1, 3)
    if np.any(mask):
        eye, dirs = camera.ray_dirs(W, H)
        D = dirs.reshape(-1, 3)[mask]
        O = np.broadcast_to(eye, (mask.sum(), 3)).astype(float).copy()
        col, _, _, _ = _shade_rays(ctx_new, O, D)
        out[mask] = col
    return out.reshape(H, W, 3), mask.reshape(H, W)


def _object_aabb(sdf, pad=0.0):
    """Axis-aligned bounding box (lo, hi) of a realized primitive: sphere center +/- radius, box center +/- half."""
    c = np.asarray(getattr(sdf, "c", (0.0, 0.0, 0.0)), float)
    if hasattr(sdf, "r"):
        ext = np.full(3, float(sdf.r))
    elif hasattr(sdf, "h"):
        ext = np.asarray(sdf.h, float)
    else:
        ext = np.full(3, 0.5)
    return c - ext - pad, c + ext + pad


def translate_object(ctx, obj_id, delta):
    """Return a NEW ctx with object `obj_id` moved by `delta` (a geometry edit). Copies the sdf with a shifted centre,
    rebuilds the union; colours/materials are unchanged. Used to apply a MOVE and re-shade only the affected pixels."""
    import copy
    from holographic_semantic import _UnionSDF
    new = dict(ctx)
    sdfs = list(ctx["sdfs"])
    s = copy.copy(sdfs[obj_id]); s.c = np.asarray(s.c, float) + np.asarray(delta, float)
    sdfs[obj_id] = s
    new["sdfs"] = sdfs; new["union"] = _UnionSDF(sdfs)
    return new


class BrickRayIndex:
    """A REGION-keyed ray index: per ray it stores the segment (origin, direction, hit-distance) AND the surface hit
    point + sun direction, so the pixels affected by a change inside a spatial region are found by EXACT vectorised
    ray-box tests -- BOTH the camera rays that reach the region (occlusion) and the pixels whose SHADOW ray crosses it
    (cast shadow). Where the object index keys edits by object id, this keys them by REGION, so a MOVE (geometry
    change) gets the same bounded delta: the rays that reach the bricks the object vacated/now occupies, plus the
    pixels whose shadow it moved. Conservative by construction (no sampling)."""

    def __init__(self, O, D, tmax, P, hit, sun_dir, lo, hi, grid, width, height,
                 sec_pix=None, sec_O=None, sec_D=None, sec_tmax=None, sec_id=None):
        self.O = O; self.D = D; self.tmax = tmax               # per-ray segment [eye, eye + tmax*D]
        self.P = P; self.hit = hit; self.sun_dir = np.asarray(sun_dir, float)   # hit points + sun (for shadow rays)
        # SECONDARY segments (reflection ray off a mirror, see-through ray inside glass), the pixel each belongs to, and
        # the object each currently hits (-1 = none) -- so a MOVE updates pixels that see the moved object IN a
        # reflection or THROUGH glass (currently show it) or that it moves INTO, not just direct/shadow pixels.
        self.sec_pix = np.zeros(0, int) if sec_pix is None else sec_pix
        self.sec_O = np.zeros((0, 3)) if sec_O is None else sec_O
        self.sec_D = np.zeros((0, 3)) if sec_D is None else sec_D
        self.sec_tmax = np.zeros(0) if sec_tmax is None else sec_tmax
        self.sec_id = np.zeros(0, int) if sec_id is None else sec_id
        self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float); self.grid = int(grid)
        self.cell = np.where((self.hi - self.lo) > 0, (self.hi - self.lo) / self.grid, 1.0)
        self.width = int(width); self.height = int(height)

    @staticmethod
    def _seg_hits_box(O, Dn, tmax, lo_b, hi_b):
        """Vectorised slab test: which segments O + [0,tmax]*Dn intersect the box [lo_b, hi_b]. Exact."""
        inv = 1.0 / np.where(np.abs(Dn) < 1e-12, 1e-12, Dn)
        t0 = (lo_b - O) * inv; t1 = (hi_b - O) * inv
        enter = np.minimum(t0, t1).max(axis=1)
        exit_ = np.maximum(t0, t1).min(axis=1)
        return (exit_ >= np.maximum(enter, 0.0)) & (enter <= tmax)

    def _rays_hit_box(self, lo_b, hi_b):
        return self._seg_hits_box(self.O, self.D, self.tmax, np.asarray(lo_b, float), np.asarray(hi_b, float))

    def _shadow_hit_box(self, lo_b, hi_b, sun_reach=30.0):
        """Pixels whose SHADOW ray (surface hit point toward the sun) crosses the box -> their cast shadow changes
        when an occluder enters/leaves the box. This is what moves the shadow of a moved object."""
        m = np.zeros(len(self.D), bool)
        if not np.any(self.hit):
            return m
        idx = np.where(self.hit)[0]
        Ps = self.P[idx] + self.sun_dir * 3e-3
        Dn = np.broadcast_to(self.sun_dir, Ps.shape)
        m[idx] = self._seg_hits_box(Ps, Dn, np.full(len(idx), sun_reach), np.asarray(lo_b, float), np.asarray(hi_b, float))
        return m

    def snap_to_bricks(self, lo_b, hi_b):
        """Expand an AABB to the brick lattice (so a region edit covers whole bricks). The lattice is unbounded -- we
        do NOT clamp to the original scene bounds, so an object that MOVES outside them is still covered fully."""
        c0 = np.floor((np.asarray(lo_b) - self.lo) / self.cell)
        c1 = np.floor((np.asarray(hi_b) - self.lo) / self.cell) + 1
        return self.lo + c0 * self.cell, self.lo + c1 * self.cell

    def _secondary_hits_box(self, lo_b, hi_b):
        """Pixels whose SECONDARY ray (reflection off a mirror, or see-through inside glass) crosses the box -> the
        moved object is now visible (or no longer visible) in that reflection / through that glass."""
        m = np.zeros(len(self.D), bool)
        if len(self.sec_pix) == 0:
            return m
        hits = self._seg_hits_box(self.sec_O, self.sec_D, self.sec_tmax,
                                  np.asarray(lo_b, float), np.asarray(hi_b, float))
        m[self.sec_pix[hits]] = True
        return m

    def _secondary_move_mask(self, obj_id, new_lo, new_hi, max_reach=30.0):
        """Pixels whose reflection/see-through CHANGES when object `obj_id` moves: those that CURRENTLY show it in
        their secondary ray (sec_id == obj_id -> it left), plus those whose secondary ray now REACHES the object's new
        region within `max_reach` (it moved into view). Precise old/new split -> conservative AND tight at the fringe."""
        m = np.zeros(len(self.D), bool)
        if len(self.sec_pix) == 0:
            return m
        shows = self.sec_id == obj_id                          # currently shows the object in reflection/glass
        reaches = self._seg_hits_box(self.sec_O, self.sec_D, np.full(len(self.sec_D), max_reach),
                                     np.asarray(new_lo, float), np.asarray(new_hi, float))
        m[self.sec_pix[shows | reaches]] = True
        return m

    def pixels_for_move(self, obj_id, old_aabb, new_aabb):
        """The bounded set of pixels a MOVE of `obj_id` can change: occlusion (camera ray reaches old/new region),
        cast shadow (shadow ray crosses old/new region), and reflection/see-through (secondary ray currently shows it
        or it moves into the secondary ray). All exact ray-box tests -> conservative, bit-exact when re-shaded."""
        old_lb, old_hb = self.snap_to_bricks(*old_aabb)
        new_lb, new_hb = self.snap_to_bricks(*new_aabb)
        mask = self._rays_hit_box(old_lb, old_hb) | self._rays_hit_box(new_lb, new_hb)
        mask |= self._shadow_hit_box(old_lb, old_hb) | self._shadow_hit_box(new_lb, new_hb)
        mask |= self._secondary_move_mask(obj_id, new_lb, new_hb)
        return mask

    def pixels_through_region(self, aabbs, shadows=True, secondary=True):
        """Boolean (npix,) mask of pixels affected by a change inside any (lo, hi) box (snapped to bricks): rays that
        REACH the box (occlusion), plus -- if enabled -- pixels whose SHADOW ray crosses it (cast shadow) and pixels
        whose SECONDARY ray crosses it (the object seen in a reflection or through glass)."""
        mask = np.zeros(len(self.D), bool)
        for lo_b, hi_b in aabbs:
            lb, hb = self.snap_to_bricks(lo_b, hi_b)
            mask |= self._rays_hit_box(lb, hb)
            if shadows:
                mask |= self._shadow_hit_box(lb, hb)
            if secondary:
                mask |= self._secondary_hits_box(lb, hb)
        return mask


def build_brick_index(ctx, camera, width, height, grid=10, samples=12, max_dist=14.0, pad=0.5):
    """Record each ray's segment (origin, direction, hit-distance) and its surface hit point + the sun direction, AND
    the SECONDARY segments (reflection ray off a mirror, see-through ray inside glass), so a MOVE becomes a bounded
    delta via exact ray-box tests for occlusion, the cast SHADOW, and the object's image in a reflection / through
    glass. (`samples` kept for back-compat, unused -- the query is exact, not sampled.)"""
    from holographic_raymarch import sphere_trace, sdf_normal
    from holographic_semantic import _UnionSDF
    union = ctx["union"]; sdfs = ctx["sdfs"]; is_glass = ctx["is_glass"]; refl = ctx["refl"]
    see_through = is_glass | ctx.get("is_translucent", np.zeros(len(sdfs), bool))   # glass OR frosted see-through
    nobj = len(sdfs)
    boxes = [_object_aabb(s) for s in sdfs if hasattr(s, "c")]
    if boxes:
        lo = np.min([b[0] for b in boxes], axis=0) - pad; hi = np.max([b[1] for b in boxes], axis=0) + pad
    else:
        lo = np.full(3, -5.0); hi = np.full(3, 5.0)
    eye, dirs = camera.ray_dirs(width, height)
    D = dirs.reshape(-1, 3); O = np.broadcast_to(eye, D.shape).astype(float).copy()
    hit, t, P = sphere_trace(union, O, D)
    tmax = np.where(hit, t, max_dist)
    Pfull = O + tmax[:, None] * D
    # SECONDARY segments -- the same reflection / glass-see-through rays the shader casts, so a moved object that is
    # visible IN a reflection or THROUGH glass also gets its pixels flagged (the reflection/GI-of-a-moved-object case).
    sec_pix = []; sec_O = []; sec_D = []; sec_tmax = []; sec_id = []
    if np.any(hit):
        idx_hit = np.where(hit)[0]
        ids = union.ids(P[hit])
        Nh = sdf_normal(union, P[hit])
        mirror = refl[ids] > 0.05                               # reflection ray off a mirror/metal surface
        if np.any(mirror):
            midx = idx_hit[mirror]
            Dm = D[midx] - 2.0 * (D[midx] * Nh[mirror]).sum(1)[:, None] * Nh[mirror]
            Pm = P[hit][mirror] + Nh[mirror] * 3e-3
            hm, tm, Pmh = sphere_trace(union, Pm, Dm)
            rid = -np.ones(len(midx), int)
            if np.any(hm):
                rid[hm] = union.ids(Pmh[hm])
            sec_pix.append(midx); sec_O.append(Pm); sec_D.append(Dm)
            sec_tmax.append(np.where(hm, tm, max_dist)); sec_id.append(rid)
        glass_here = see_through[ids]                           # see-through ray inside glass / frosted
        if np.any(glass_here) and not np.all(see_through):
            gidx = idx_hit[glass_here]
            Pg = P[hit][glass_here] + D[gidx] * 6e-3; Dg = D[gidx]
            ng_i = [i for i in range(nobj) if not see_through[i]]
            un2 = _UnionSDF([sdfs[i] for i in ng_i])
            hg, tg, Pgh = sphere_trace(un2, Pg, Dg)
            gid = -np.ones(len(gidx), int)
            if np.any(hg):
                gid[hg] = np.array(ng_i)[un2.ids(Pgh[hg])]      # local ng id -> global object id
            sec_pix.append(gidx); sec_O.append(Pg); sec_D.append(Dg)
            sec_tmax.append(np.where(hg, tg, max_dist)); sec_id.append(gid)
    if sec_pix:
        sec_pix = np.concatenate(sec_pix); sec_O = np.concatenate(sec_O)
        sec_D = np.concatenate(sec_D); sec_tmax = np.concatenate(sec_tmax); sec_id = np.concatenate(sec_id)
    else:
        sec_pix = sec_O = sec_D = sec_tmax = sec_id = None
    return BrickRayIndex(O, D, tmax, Pfull, hit, ctx["sun_dir"], lo, hi, grid, width, height,
                         sec_pix=sec_pix, sec_O=sec_O, sec_D=sec_D, sec_tmax=sec_tmax, sec_id=sec_id)


def delta_reshade_move(ctx_old, obj_id, delta, brick_index, base_frame, camera, ao_pad=0.6):
    """Apply a MOVE of `obj_id` by `delta` as a bounded delta: the affected pixels are the rays that traversed the
    bricks the object VACATED (old AABB) or now OCCUPIES (new AABB); re-shade only those against the moved scene and
    composite into `base_frame`. Returns (updated_frame, mask, ctx_new). Bit-exact vs a full re-render of the move."""
    from holographic_semantic import _shade_rays
    H, W = base_frame.shape[:2]
    old_aabb = _object_aabb(ctx_old["sdfs"][obj_id], pad=ao_pad)
    ctx_new = translate_object(ctx_old, obj_id, delta)
    new_aabb = _object_aabb(ctx_new["sdfs"][obj_id], pad=ao_pad)
    mask = brick_index.pixels_for_move(obj_id, old_aabb, new_aabb)
    out = base_frame.copy().reshape(-1, 3)
    if np.any(mask):
        eye, dirs = camera.ray_dirs(W, H)
        D = dirs.reshape(-1, 3)[mask]
        O = np.broadcast_to(eye, (mask.sum(), 3)).astype(float).copy()
        out[mask] = _shade_rays(ctx_new, O, D)[0]
    return out.reshape(H, W, 3), mask.reshape(H, W), ctx_new


def _selftest():
    """The index must flag through-glass pixels, and a delta re-shade must match a full re-render exactly."""
    from holographic_semantic import parse_description, render_scene, _scene_setup
    from holographic_render import Camera
    objs = parse_description("a glass ball beside a red ball")["objects"]
    cam = Camera(eye=(0.2, 1.4, 5.0), target=(0, 0.1, 0), fov_deg=48.0)
    W = H = 80
    ctx = _scene_setup(objs, True, "clear", "bright", (0.75, 0.9, 0.85))
    base = render_scene(objs, cam, width=W, height=H, ss=1, dither=0.0)
    index = build_ray_index(ctx, cam, W, H)
    # recolour the red ball (object id 1); re-shade only the pixels touching it
    objs2 = [dict(o) for o in objs]; objs2[1] = dict(objs2[1]); objs2[1]["color"] = "green"
    ctx2 = _scene_setup(objs2, True, "clear", "bright", (0.75, 0.9, 0.85))
    updated, mask = delta_reshade(ctx2, index, [1], base, cam)
    full = render_scene(objs2, cam, width=W, height=H, ss=1, dither=0.0)
    err = np.abs(updated - full).max()
    frac = mask.mean()
    assert err < 1e-9, err                                      # delta re-shade is bit-exact
    assert frac < 0.9                                           # and it did NOT re-shade the whole frame
    print("rayindex selftest ok: delta re-shade bit-exact (max err %.2e), re-shaded %.1f%% of pixels" % (err, 100 * frac))


if __name__ == "__main__":
    _selftest()
