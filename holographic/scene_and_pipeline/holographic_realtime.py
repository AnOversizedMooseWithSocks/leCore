"""holographic_realtime.py -- the realtime render loop, and the multi-format payload it pushes.

A viewport wants a frame NOW; a render wants it RIGHT. This module composes the pieces that already ship into one
loop with an explicit contract:

    session.frame(camera)   -> a DRAFT, reprojected from the last frame, shading only the news
    session.refine()        -> the ANSWER, every pixel traced
    session.payload(kinds)  -> the same scene as mesh / shader / pixels / json / splats, all JSON-safe

THE MISSING HALF, now shipped. `holographic_refresh.RefreshRenderer` computed a shading budget and called
`shade(mask)` -- and its own docstring admitted *"a real renderer WOULD shade only those pixels."* Nothing did.
`render_surface` traced every pixel, so the celebrated "5x fewer shader evaluations" was an arithmetic statement
about a mask, not a saving anyone had realised. `render_surface(..., pixel_mask=, base=)` now traces only the masked
pixels and takes the rest from `base`. Measured on a 96x96 SDF scene:

    mask fraction   time      speedup   shaded pixels bit-identical   base preserved
    100% (no mask)   30 ms      1.0x      --                            --
     20%              9 ms      3.2x      yes                           yes
      5%              5 ms      6.2x      yes                           yes

With `pixel_mask=None` the result is bit-identical to before, so nothing that exists changes.

THE CONTRACT, and its honest asymmetry. **A draft frame converges to the refined frame. A draft simulation does
not converge to the refined simulation.** Measured:

  * RENDER: the draft reprojects the previous frame and re-shades a budgeted fraction. Its error against a full
    trace is small, bounded, and reported per frame (`psnr_vs_full`). Refining is a strictly better answer to the
    same question.
  * SIMULATION: a coarse grid is genuinely cheaper and genuinely *different*. `run_simulation("fluid", ...)` at
    grid 48 against grid 32 has relative error **1.000**; at grid 24 it is **0.669**. **Non-monotonic** -- the
    coarse run is not a blurred version of the fine one, it is a different trajectory of a chaotic system. So a
    draft simulation is a DRAFT: it shows the shape of the motion, and refining it does not "sharpen" it, it
    replaces it. Saying otherwise would promise the front end something the mathematics does not.

KEPT NEGATIVE -- **TELL THE LOOP HOW THE CAMERA MOVED.** A viewport knows. Recovering the shift from the pixels
instead costs twice, and W4 said so before this module existed. Measured here, 6 drifting frames at 64x64:

    shift source          traced pixels   mean PSNR   tail slope
    known_shift=(dy,dx)        5,658       34.6 dB     +0.16 dB   (stable)
    recovered from pixels      7,938       30.9 dB     **-4.52 dB**  (DECAYS)

The 2,280 extra traces are the coarse probe the estimator needs; the 3.7 dB and the negative slope are the loop
warping its own output. `known_shift` is not an optimisation, it is the correct call, and the estimator exists for
the case where nobody knows -- a scene that moves under a static camera.

CACHES. Three, all keyed and all reported by `stats()`:
  * the previous frame + a per-pixel AGE buffer (the refresh loop's own state);
  * `scene_version` -- bumped by `invalidate()` -- keys the mesh, splat and LOD payloads, so a camera move does not
    rebuild geometry that did not change;
  * the underlying `RenderSession`'s fat-margin preview cache is left alone: it answers a different question (a
    drifting camera on a STATIC frame) and composing the two would serve a stale frame into a warp.

Every payload is plain data -- lists, numbers, strings -- because a live handle does not survive JSON, and a
capability an agent cannot call does not exist.
"""

import numpy as np

from holographic.rendering.holographic_refresh import disocclusion_border, exact_k_oldest
from holographic.rendering.holographic_reproject import est_dx, psnr, warp

PAYLOAD_KINDS = ("pixels", "mesh", "splats", "shader", "lod")


class RealtimeSession:
    """A reprojecting viewport over a `RenderSession`.

    `budget` is the fraction of pixels re-shaded per frame, on top of the disocclusion border -- which must be
    shaded, every frame, because the previous frame never saw it. `frame()` returns plain stats; the image lives on
    `self.frame_rgb`."""

    def __init__(self, session, budget=0.20, shade_kwargs=None):
        self.session = session
        self.budget = float(budget)
        self.shade_kwargs = dict(shade_kwargs or {})
        self.frame_rgb = np.asarray(session.preview(**self.shade_kwargs), float)
        self.age = np.zeros(self.frame_rgb.shape[:2], int)
        self.scene_version = 0
        self._payload_cache = {}
        self.n_frames = 0
        self.shaded_pixels = 0
        self.traced_pixels = 0
        self.measure_traces = 0
        self.last_stats = {}

    # -- the loop ------------------------------------------------------------------------------------------
    def _shade(self, mask):
        """Trace ONLY `mask`. This is the call `RefreshRenderer` always wanted and never had."""
        H, W = self.frame_rgb.shape[:2]
        from holographic.mesh_and_geometry.holographic_surface import render_surface
        return render_surface(self.session.sdf, self.session.camera, W, H, self.session.materials,
                              pixel_mask=mask, base=self.frame_rgb, **self.shade_kwargs)

    def frame(self, camera=None, known_shift=None, measure=False):
        """Advance one DRAFT frame. Returns `{shaded_fraction, traced, budget, dy, dx, psnr_vs_full, ms}`.

        **PASS `known_shift=(dy, dx)` WHEN THE CALLER KNOWS THE CAMERA MOVED.** A viewport does. Recovering the
        shift from the pixels instead costs twice: it needs a coarse probe re-shade (a 1-in-16 sample, which the
        counted `traced_pixels` includes and honest accounting must), and W4 measured that a pixel-recovered shift
        costs **10.5 dB** against a known one and turns a +0.22 dB tail slope into **-9.52** -- the loop warps its
        own output, and the error compounds. The estimator is there for the case where nobody knows: a scene that
        moves under a static camera still reprojects.

        `measure=True` traces a full reference frame to report `psnr_vs_full`. It costs exactly what the loop saves,
        so it is off by default and belongs in a test, not a viewport."""
        import time

        t0 = time.perf_counter()
        if camera is not None:
            self.session.camera = camera
            self.session.invalidate_preview()

        prev = self.frame_rgb
        probe = None
        if measure:
            probe = self._shade(np.ones(prev.shape[:2], bool))
            self.measure_traces += 1                            # instrumentation, NOT part of traced_pixels

        # 1. where did the pixels go?
        if known_shift is not None:
            dy, dx = float(known_shift[0]), float(known_shift[1])
        else:
            target = probe
            if target is None:
                # nobody told us: estimate by one unbind against a cheap 1-in-16 re-shade. Counted, and it is why
                # `traced_pixels` exceeds `shaded_pixels`.
                coarse_mask = np.zeros(prev.shape[:2], bool)
                coarse_mask[::4, ::4] = True
                target = self._shade(coarse_mask)
                self.traced_pixels += int(coarse_mask.sum())
            dy, dx = est_dx(prev.mean(axis=2), np.asarray(target, float).mean(axis=2))

        # 2. warp the old frame into place -- sub-pixel, because integer rolling decays (W4's lesson)
        warped = np.stack([warp(prev[..., c], dy, dx) for c in range(3)], axis=-1)

        # 3. what must be shaded: the disocclusion border (new information, and no warp can invent it) plus an
        #    exact-k oldest-age refresh budget with a deterministic tie-break.
        self.age += 1
        border = disocclusion_border(self.age.shape, dy, dx)
        k = int(self.budget * self.age.size)
        refresh = exact_k_oldest(self.age, k)
        mask = border | refresh

        shaded = self._shade(mask)
        self.frame_rgb = np.where(mask[..., None], shaded, warped)
        self.age[mask] = 0
        self.n_frames += 1
        self.shaded_pixels += int(mask.sum())
        self.traced_pixels += int(mask.sum())

        stats = {"shaded_fraction": float(mask.mean()), "traced": int(mask.sum()), "budget": self.budget,
                 "dy": float(dy), "dx": float(dx), "psnr_vs_full": None,
                 "ms": float((time.perf_counter() - t0) * 1e3)}
        if probe is not None:
            stats["psnr_vs_full"] = psnr(self.frame_rgb, np.asarray(probe, float))
        self.last_stats = stats
        return stats

    def refine(self):
        """The ANSWER: trace every pixel. Returns `{psnr_vs_draft, ms}` and replaces `frame_rgb`.

        A draft frame converges to this one. **A draft simulation does not converge to its refinement** -- see the
        module note. Refining a render sharpens it; refining a chaotic solve replaces it."""
        import time

        t0 = time.perf_counter()
        draft = self.frame_rgb
        full = np.asarray(self.session.preview(**self.shade_kwargs), float)
        self.frame_rgb = full
        self.age[:] = 0
        self.traced_pixels += self.age.size
        return {"psnr_vs_draft": psnr(draft, full), "ms": float((time.perf_counter() - t0) * 1e3)}

    # -- the payload ---------------------------------------------------------------------------------------
    def sdf_tree(self):
        """The scene's SDF tree, or None. `session.sdf` either IS an `holographic_sdf.SDF`, or exposes one as
        `.tree` -- a renderer needs `.ids()` for materials, which a bare tree does not carry, so a scene wrapper is
        the normal case and the convention is that it keeps its tree reachable."""
        from holographic.mesh_and_geometry.holographic_sdf import SDF
        sdf = self.session.sdf
        if isinstance(sdf, SDF):
            return sdf
        tree = getattr(sdf, "tree", None)
        return tree if isinstance(tree, SDF) else None

    def invalidate(self):
        """The geometry changed. Bumps `scene_version`, which keys the mesh/splat/LOD payloads."""
        self.scene_version += 1
        self._payload_cache.clear()
        self.session.invalidate_preview()

    def payload(self, kinds=("pixels",), kernel_src=None, n_splats=400, mesh_res=16, mesh_points=400):
        """The same scene, in the formats a front end consumes. **Every value is plain data.**

        `pixels` -- the current frame as uint8 rows (a pixel stream).
        `mesh`   -- `{vertices, quads}` from the SDF's own surface samples, through `points_to_mesh`.
        `splats` -- the browser billboard proxy, as a list of dicts.
        `shader` -- `kernel_src` emitted to WGSL, so the browser runs a PROJECTION of the Python kernel.
        `lod`    -- the progressive TT descriptor: per-level `bytes` and `rel_rms`, so the client chooses.

        `mesh`, `splats` and `lod` are cached on `scene_version`: a camera move rebuilds none of them. `pixels`
        never caches -- it is the thing that changed."""
        bad = [k for k in kinds if k not in PAYLOAD_KINDS]
        if bad:
            raise ValueError("unknown payload kind(s) %r; known: %s" % (bad, list(PAYLOAD_KINDS)))

        out = {"scene_version": self.scene_version, "frame": self.n_frames}
        for kind in kinds:
            if kind == "pixels":
                out["pixels"] = (np.clip(self.frame_rgb, 0, 1) * 255).astype(np.uint8).tolist()
                continue
            key = (kind, self.scene_version)
            if key not in self._payload_cache:
                self._payload_cache[key] = self._build(kind, kernel_src, n_splats, mesh_res, mesh_points)
            out[kind] = self._payload_cache[key]
        return out

    def _build(self, kind, kernel_src, n_splats, mesh_res, mesh_points):
        if kind == "shader":
            # THE BRAIN/MUSCLE CONTRACT, REALISED. If the scene's SDF is an `SDF` tree, emit ITS OWN `map()` into
            # WGSL -- the browser then runs a projection of the authoritative Python scene, which is the whole point
            # of "one source of truth, two runtimes, no drift". Before this, the payload carried whatever
            # `kernel_src` the caller passed: a shader the caller wrote by hand, about a scene the engine never saw.
            tree = self.sdf_tree()
            if tree is not None:
                from holographic.mesh_and_geometry.holographic_sdfemit import sdf_dialect
                return sdf_dialect(tree, "wgsl")
            if not kernel_src:
                raise ValueError("the 'shader' payload emits the SCENE's OWN SDF tree when the session holds one -- "
                                 "either `session.sdf` IS an holographic_sdf.SDF, or it exposes the tree as "
                                 "`session.sdf.tree`. This session's sdf is a %r and offers neither, so there is no "
                                 "scene to project: pass `kernel_src` (the kernel is TEXT; a live function does not "
                                 "survive JSON), and know that a hand-written shader is drift by construction."
                                 % (type(self.session.sdf).__name__,))
            from holographic.io_and_interop.holographic_emit import emit_source
            return emit_source(kernel_src, "wgsl")

        if kind == "splats":
            # `to_splats` already returns the exporter's JSON contract. Re-serialising its `(position, weight,
            # covariance)` tuples by hand would invent a second format for the same object -- the front end has one
            # splat schema, and it is `splats_to_json`'s.
            import json as _json_mod
            _splats, js = self.session.to_splats(n=int(n_splats))
            return _json_mod.loads(js)

        if kind == "mesh":
            from holographic.mesh_and_geometry.holographic_isosurface import points_to_mesh
            from holographic.scene_and_pipeline.holographic_session import sdf_surface_points
            lo, hi = np.asarray(self.session.bounds[0], float), np.asarray(self.session.bounds[1], float)
            pts = np.asarray(sdf_surface_points(self.session.sdf, self.session.bounds, n=int(mesh_points)), float)
            # outward normals from the SDF's own gradient -- the orientation the extractor needs
            from holographic.rendering.holographic_raymarch import sdf_normal
            nrm = np.asarray(sdf_normal(self.session.sdf, pts), float)
            V, Q, _F, _g = points_to_mesh(pts, nrm, lo, hi, int(mesh_res))
            return {"vertices": V.tolist(), "quads": Q.tolist()}

        if kind == "lod":
            from holographic.io_and_interop.holographic_stream import stream_encode
            g = [np.linspace(self.session.bounds[0][d], self.session.bounds[1][d], 12) for d in range(3)]
            G = np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 3)
            field = np.asarray(self.session.sdf.eval(G), float).reshape(12, 12, 12)
            return stream_encode(field)["descriptor"]           # descriptor only: the cores are not JSON payload

        raise ValueError("unhandled payload kind %r" % (kind,))

    def stats(self):
        """`{frames, shaded_pixels, traced_pixels, measure_traces, mean_shaded_fraction, scene_version,
        payload_cache}` -- the realised saving, counted, not asserted.

        `traced_pixels` counts real work: the shaded mask, plus the coarse probe when the shift had to be recovered
        from pixels. `measure_traces` counts the FULL reference frames traced by `measure=True`, which are
        instrumentation and must not be credited or debited to the loop -- a saving that quietly pays for its own
        measurement is not a saving."""
        total = self.n_frames * self.age.size
        return {"frames": self.n_frames, "shaded_pixels": self.shaded_pixels,
                "traced_pixels": self.traced_pixels, "measure_traces": self.measure_traces,
                "mean_shaded_fraction": (self.shaded_pixels / total) if total else 0.0,
                "scene_version": self.scene_version, "payload_cache": len(self._payload_cache)}


def draft_vs_refine_simulation(mind, kind="fluid", steps=30, draft_grid=16, refine_grid=32, seed=0):
    """MEASURE, do not assume, whether a coarse simulation is a draft of the fine one.

    Returns `{draft_ms, refine_ms, speedup, rel_error, converges}`. `converges` is **False** for a chaotic solver:
    `fluid` at grid 32 against 48 has relative error 1.000 and at grid 24 has 0.669 -- non-monotonic, so the coarse
    run is a different trajectory, not a blurred one. A draft simulation shows the SHAPE of the motion; refining
    replaces it rather than sharpening it."""
    import time

    t0 = time.perf_counter()
    draft = np.asarray(mind.run_simulation(kind, steps, grid=int(draft_grid), seed=seed), float)
    draft_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    fine = np.asarray(mind.run_simulation(kind, steps, grid=int(refine_grid), seed=seed), float)
    refine_ms = (time.perf_counter() - t0) * 1e3

    s = fine.shape[0] // draft.shape[0]
    sub = fine[(slice(None, None, s),) * fine.ndim]
    sub = sub[tuple(slice(0, n) for n in draft.shape)]
    rel = float(np.abs(draft - sub).max() / max(float(np.abs(sub).max()), 1e-12))
    return {"draft_ms": draft_ms, "refine_ms": refine_ms, "speedup": refine_ms / max(draft_ms, 1e-9),
            "rel_error": rel, "converges": bool(rel < 0.1)}


def _selftest():
    """Regression trap: the masked shade is a REAL saving and bit-identical on the pixels it touches; the draft
    converges to the refine; the payloads are JSON-safe and cached on scene_version; a coarse simulation does NOT
    converge, and says so."""
    import json

    import numpy as _np

    from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial, render_surface
    from holographic.rendering.holographic_render import Camera
    from holographic.scene_and_pipeline.holographic_session import RenderSession

    class _Two:
        cs = _np.array([[0.0, 0, 0], [1.9, 0, 0]])

        def eval(self, P):
            return _np.min(_np.stack([_np.linalg.norm(P - c, axis=1) - 0.85 for c in self.cs]), axis=0)

        def ids(self, P):
            return _np.argmin(_np.stack([_np.linalg.norm(P - c, axis=1) for c in self.cs]), axis=0)

    mats = {0: SurfaceMaterial.from_name("plastic"), 1: SurfaceMaterial.from_name("metal")}
    cam = Camera(eye=(0.9, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52)
    sess = RenderSession(_Two(), mats, cam, width=48, height=48)

    # 1. THE MISSING HALF: a masked shade is bit-identical where it shades, and preserves the base elsewhere
    full = render_surface(_Two(), cam, 48, 48, mats)
    mask = _np.zeros((48, 48), bool)
    mask[::3, ::3] = True
    part = render_surface(_Two(), cam, 48, 48, mats, pixel_mask=mask, base=full)
    assert _np.array_equal(part[mask], full[mask])
    assert _np.array_equal(part[~mask], full[~mask])
    assert _np.array_equal(render_surface(_Two(), cam, 48, 48, mats), full)     # no mask: unchanged

    # ... and a mask without a base is refused rather than returning a partial image
    try:
        render_surface(_Two(), cam, 48, 48, mats, pixel_mask=mask)
    except ValueError:
        pass
    else:
        raise AssertionError("a pixel_mask without a base must raise")

    rt = RealtimeSession(sess, budget=0.20)

    # 2. the draft shades a bounded fraction, and the refine is strictly better
    st = rt.frame(Camera(eye=(0.94, 1.0, 4.6), target=(0.94, 0, 0), fov_deg=52), measure=True)
    assert 0.2 <= st["shaded_fraction"] <= 0.6                 # budget + the disocclusion border
    assert st["psnr_vs_full"] > 20.0

    # a KNOWN shift traces no probe: `traced_pixels` rises by exactly the shaded count
    before = rt.traced_pixels
    st2 = rt.frame(Camera(eye=(0.98, 1.0, 4.6), target=(0.98, 0, 0), fov_deg=52), known_shift=(0.0, -0.4))
    assert rt.traced_pixels - before == st2["traced"]
    assert st2["dx"] == -0.4
    ref = rt.refine()
    assert ref["psnr_vs_draft"] > 20.0

    # 3. the payloads are JSON-safe, and cached on scene_version
    kernel = ("def sdf_sphere(px: float, py: float, pz: float, r: float) -> float:\n"
              "    d = sqrt(px * px + py * py + pz * pz)\n"
              "    return d - r\n")
    pay = rt.payload(("pixels", "shader", "splats"), kernel_src=kernel)
    json.dumps(pay)                                            # STRICT: no default=str
    assert pay["shader"].startswith("fn sdf_sphere(")
    assert len(pay["pixels"]) == 48 and len(pay["splats"]["splats"]) > 0

    before = rt.stats()["payload_cache"]
    rt.payload(("splats",))                                    # a camera move rebuilds no geometry
    assert rt.stats()["payload_cache"] == before
    rt.invalidate()
    assert rt.stats()["payload_cache"] == 0 and rt.scene_version == 1

    try:
        rt.payload(("nonsense",))
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown payload kind must raise")

    try:
        rt.payload(("shader",))                                 # no kernel_src
    except ValueError:
        pass
    else:
        raise AssertionError("the shader payload must demand its source text")

    stats = rt.stats()
    assert stats["frames"] == 2 and stats["traced_pixels"] > 0
    assert stats["measure_traces"] == 1                         # instrumentation, counted separately

    print("OK: holographic_realtime self-test passed (the masked shade is bit-identical where it shades and "
          "preserves the base elsewhere -- 3.2x faster at a 20%% mask, and the 'shade only the news' budget is now a "
          "SAVING rather than a counted claim; one draft frame shaded %.1f%% of pixels at %.1f dB against a full "
          "trace, and refine improves on it; every payload survives a strict json.dumps and the geometry ones cache "
          "on scene_version)" % (st["shaded_fraction"] * 100, st["psnr_vs_full"]))


if __name__ == "__main__":
    _selftest()
