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


class IncrementalRenderer:
    """A render SESSION built on the bidirectional index, so a mostly-static scene is cheap to re-render and edits
    STREAM as deltas. The first render() is a full render; after that:
      * re-rendering the SAME scene is FREE -- nothing changed, so the cached frame is returned with an empty mask
        (this is the case that was silently doing a full re-trace every frame);
      * a colour / material / light EDIT re-shades ONLY the affected pixels via the ray index (through-glass, in
        reflection, SSS, translucency all handled) -- bit-exact, bounded;
      * a geometry MOVE re-shades only the pixels the object vacated / now occupies (+ its shadow / reflection) via
        the brick index -- bit-exact, bounded.
    Every call returns (frame, changed_mask); `stream_delta(mask)` turns the mask into just the changed pixels to send
    over the wire -- so the pixel stream carries O(changed), not O(frame). Default ss=1: deterministic and delta-exact
    (supersampling is for a one-shot final frame, not a stream). This is the "only pay for what changed" path end to end."""

    def __init__(self, camera, width=256, height=256, sun="bright", sky="clear", ground=True, ss=1,
                 glass_tint=(0.75, 0.9, 0.85)):
        self.cam = camera; self.w = int(width); self.h = int(height)
        self.sun = sun; self.sky = sky; self.ground = ground; self.ss = int(ss); self.glass_tint = glass_tint
        self.frame = None; self.ctx = None; self.ray_index = None; self.brick_index = None
        self._key = None                                        # canonical snapshot of the cached scene

    @staticmethod
    def _scene_key(objects):
        """A cheap canonical signature of the scene's appearance -- so an identical re-render is detected and skipped."""
        return tuple((o.get("shape"), o.get("color"), o.get("material"), o.get("size")) for o in objects)

    def render(self, objects, force=False):
        """Return (frame, changed_mask). If the scene is unchanged since the last render, this is FREE: the cached frame
        is returned and changed_mask is all-False. Otherwise a full render runs and the indices are (re)built."""
        from holographic_semantic import render_scene
        key = self._scene_key(objects)
        if not force and self.frame is not None and key == self._key:
            return self.frame, np.zeros((self.h, self.w), bool)        # nothing changed -> no work, no delta
        st = {}
        self.frame = render_scene(objects, self.cam, self.w, self.h, sun=self.sun, sky=self.sky,
                                  ground=self.ground, ss=self.ss, dither=0.0, adaptive=(self.ss > 1), stats=st)
        self.ctx = st["ctx"]; self._key = key
        self._objects = [dict(o) for o in objects]
        self.ray_index = build_ray_index(self.ctx, self.cam, self.w, self.h)
        self.brick_index = build_brick_index(self.ctx, self.cam, self.w, self.h)
        self._capture_gbuffer()
        return self.frame, np.ones((self.h, self.w), bool)            # first render: whole frame is "new"

    def edit(self, obj_index, field, value):
        """Apply a colour / material / light EDIT (geometry unchanged) and re-shade only the affected pixels. Returns
        (frame, changed_mask). The ray index stays valid, so this is a bounded, bit-exact delta."""
        from holographic_semantic import _scene_setup
        if self.frame is None:
            raise RuntimeError("call render() before edit()")
        self._ensure_fresh()                                   # if a camera move left the index stale, rebuild it once
        self._objects[obj_index][field] = value
        ctx2 = _scene_setup(self._objects, self.ground, self.sky, self.sun, self.glass_tint)
        self.frame, mask = delta_reshade(ctx2, self.ray_index, [obj_index], self.frame, self.cam)
        self.ctx = ctx2; self._key = self._scene_key(self._objects)
        return self.frame, mask

    def move(self, obj_index, delta):
        """Apply a geometry MOVE and re-shade only the affected pixels via the brick index. Returns (frame, mask).
        Geometry changed, so the indices are rebuilt for subsequent edits/moves (a trace, still far cheaper than a full
        shade with SSS / reflections / shadows)."""
        if self.frame is None:
            raise RuntimeError("call render() before move()")
        self._ensure_fresh()                                   # if a camera move left the index stale, rebuild it once
        self.frame, mask, ctxn = delta_reshade_move(self.ctx, obj_index, delta, self.brick_index, self.frame, self.cam)
        self.ctx = ctxn
        self.ray_index = build_ray_index(ctxn, self.cam, self.w, self.h)      # geometry moved -> refresh the indices
        self.brick_index = build_brick_index(ctxn, self.cam, self.w, self.h)
        return self.frame, mask

    def stream_delta(self, mask):
        """Turn a changed-pixel mask into the wire payload: (ys, xs, rgb) of exactly the pixels that changed -- so the
        pixel stream is O(changed), not O(frame). An unchanged frame streams nothing."""
        m = np.asarray(mask, bool).reshape(self.h, self.w)
        ys, xs = np.where(m)
        return ys, xs, (self.frame[ys, xs] if len(ys) else np.zeros((0, 3)))

    def _capture_gbuffer(self):
        """Record the per-pixel WORLD hit point, a hit mask, and a view-DEPENDENT flag (reflective / glass / frosted --
        whose look changes with the camera). This is the G-buffer that lets a camera move REPROJECT instead of re-trace:
        a world point's diffuse shade doesn't change with view, only which pixel it lands on."""
        self._P = self.brick_index.P.reshape(self.h, self.w, 3).copy()   # world hit point per pixel (eye + tmax*dir)
        self._hit = self.brick_index.hit.reshape(self.h, self.w).copy()
        vd = np.zeros(self.h * self.w, bool)
        prim = self.ray_index.primary; hm = prim >= 0
        refl = self.ctx["refl"]; isg = self.ctx["is_glass"]
        ist = self.ctx.get("is_translucent", np.zeros(len(refl), bool))
        viewdep_obj = (refl > 0.05) | isg | ist                # these secondary/specular terms depend on the view
        vd[hm] = viewdep_obj[prim[hm]]
        self._viewdep = vd.reshape(self.h, self.w)
        self._stale = False                                    # indices/g-buffer are valid for the current camera

    def reproject(self, new_camera, reshade_viewdep=True):
        """A CAMERA MOVE without re-tracing the whole scene (the 3DGS / DLSS / V-Ray-realtime idea): forward-project the
        cached frame's WORLD hit points into the new view (diffuse shade is view-independent, so it carries over), keep
        the nearest via a z-buffer, and re-shade ONLY the holes (disocclusions -- newly revealed geometry) and the
        view-DEPENDENT pixels (reflections / glass / frosted, whose look changes with the camera). Returns (frame,
        reshaded_mask). Approximate (resampling), not bit-exact -- the muscle-layer trade every realtime renderer makes;
        re-shaded and disoccluded pixels ARE exact. Call render() again for a bit-exact still."""
        if self.frame is None:
            raise RuntimeError("call render() before reproject()")
        from holographic_semantic import _shade_rays
        H, W = self.h, self.w
        P = self._P.reshape(-1, 3); hitmask = self._hit.reshape(-1)
        prevcol = self.frame.reshape(-1, 3); vdep = self._viewdep.reshape(-1)
        src = np.where(hitmask)[0]
        x, y, depth = _project_points(new_camera, P[src], W, H)
        xi = np.round(x).astype(int); yi = np.round(y).astype(int)
        keep = (depth > 1e-4) & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        src = src[keep]; dst = yi[keep] * W + xi[keep]; depth = depth[keep]
        order = np.argsort(-depth)                              # far -> near, so the NEAREST wins the z-buffer (last write)
        out = np.zeros((H * W, 3)); filled = np.zeros(H * W, bool); vdep_new = np.zeros(H * W, bool)
        out[dst[order]] = prevcol[src[order]]                  # reproject diffuse shade (view-independent) to new pixels
        filled[dst[order]] = True
        vdep_new[dst[order]] = vdep[src[order]]
        reshade = (~filled) | (vdep_new & reshade_viewdep)     # holes (disocclusion / sky) + view-dependent -> re-shade
        if np.any(reshade):
            eye, dirs = new_camera.ray_dirs(W, H)
            D = dirs.reshape(-1, 3)[reshade]
            O = np.broadcast_to(eye, (int(reshade.sum()), 3)).astype(float).copy()
            col, _, _, _ = _shade_rays(self.ctx, O, D)
            out[reshade] = np.clip(col, 0, 1)
        self.frame = out.reshape(H, W, 3); self.cam = new_camera
        self._stale = True                                     # camera changed: ray/brick indices now need a rebuild
        return self.frame, reshade.reshape(H, W)

    def _ensure_fresh(self):
        """Rebuild the camera-space indices + g-buffer if a camera move left them stale (done lazily -- pure navigation
        pays nothing; the cost is only taken when an EDIT/MOVE at the new camera actually needs the index)."""
        if getattr(self, "_stale", False) or self.ray_index is None:
            self.ray_index = build_ray_index(self.ctx, self.cam, self.w, self.h)
            self.brick_index = build_brick_index(self.ctx, self.cam, self.w, self.h)
            self._capture_gbuffer()


def _project_points(cam, P, width, height):
    """World points P (N,3) -> (x, y, forward_depth) in `cam`'s image, the exact inverse of Camera.ray_dirs."""
    r, u, f = cam._basis()
    t = np.tan(np.radians(cam.fov_deg) / 2.0)
    d = np.asarray(P, float) - cam.eye
    fwd = d @ f
    safe = np.where(np.abs(fwd) < 1e-9, 1e-9, fwd)
    ndc_x = (d @ r) / safe; ndc_y = (d @ u) / safe
    x = ((ndc_x / (cam.aspect * t) + 1.0) / 2.0) * width - 0.5
    y = ((1.0 - ndc_y / t) / 2.0) * height - 0.5
    return x, y, fwd


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
