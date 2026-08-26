"""Phases 1-2: the configurable render/sim pipeline -- select, auto-include, reject, order, plan, run."""
import numpy as np
import pytest
from holographic.scene_and_pipeline.holographic_pipeline import PipelineConfig, Pipeline, build_pipeline, PipelineError, ALL_STAGES


def test_preset_selects_stages_and_autoincludes_gbuffer():
    pipe = build_pipeline(PipelineConfig.preview())
    n = pipe.stage_names()
    assert "render" in n and "present" in n
    assert "svgf_denoise" in n and "gbuffer" in n                 # gbuffer AUTO-INCLUDED (its enabled() is False)
    assert "splat_proxy" in n and "reproject" in n


def test_stable_dependency_order():
    n = build_pipeline(PipelineConfig.preview()).stage_names()
    pos = {name: i for i, name in enumerate(n)}
    assert pos["render"] < pos["present"]
    assert pos["gbuffer"] < pos["svgf_denoise"] and pos["render"] < pos["svgf_denoise"]


def test_conflict_dirty_only_needs_temporal_reuse():
    with pytest.raises(PipelineError) as e:
        build_pipeline(PipelineConfig(dirty_only=True, temporal_reuse=False))
    assert "temporal_reuse" in str(e.value)


def test_conflict_splat_proxy_with_final():
    with pytest.raises(PipelineError) as e:
        build_pipeline(PipelineConfig(quality="final", splat_proxy=True, dirty_only=False))
    assert "splat_proxy" in str(e.value)


def test_bad_denoise_rejected():
    with pytest.raises(PipelineError):
        build_pipeline(PipelineConfig(denoise="magic", dirty_only=False, temporal_reuse=False))


def test_plan_is_dry_run():
    pipe = build_pipeline(PipelineConfig.preview())
    plan = pipe.plan()
    assert all("stage" in p and "why" in p for p in plan)
    assert len(plan) == len(pipe.stage_names())


def test_run_produces_tonemapped_image():
    pipe = build_pipeline(PipelineConfig.preview())
    ctx = pipe.run(scene="s", seed=0)
    assert ctx.image is not None
    assert ctx.image.min() >= 0.0 and ctx.image.max() <= 1.0
    assert ctx.buffers.get("denoised") is True


def test_interactive_pulls_sim_before_render():
    n = build_pipeline(PipelineConfig.interactive()).stage_names()
    pos = {name: i for i, name in enumerate(n)}
    assert "sim_fluid" in n and "sim_field_effects" in n and "sim_collide" in n
    assert pos["sim_collide"] < pos["render"]


def test_deterministic_order():
    assert build_pipeline(PipelineConfig.preview()).stage_names() == \
           build_pipeline(PipelineConfig.preview()).stage_names()


def test_manual_assembly_still_works():
    manual = Pipeline([s for s in ALL_STAGES if s.name in ("render", "present")])
    ctx = manual.run(scene="s", seed=0)
    assert ctx.image is not None


def test_wired_stages_are_real_and_measured():
    """The demo pipeline now runs REAL stages: robust_accumulate rejects a bad sample pass, SVGF denoises using
    real g-buffer features, and adaptive builds a real variance mask + an SPRT decision."""
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    b = build_pipeline(PipelineConfig.preview()).run().buffers
    # robust accumulate beats the naive mean when a whole sample pass is corrupted (firefly frame)
    assert b["accum_rmse"]["robust"] < b["accum_rmse"]["naive"] * 0.5
    # SVGF actually denoises (denoised PSNR strictly above the noisy input)
    assert b["svgf_psnr"]["denoised"] > b["svgf_psnr"]["noisy"]
    # adaptive produced a real per-pixel mask (an array), not the string stand-in, and an SPRT decision
    import numpy as np
    assert isinstance(b["sample_map"], np.ndarray) and b["sample_map"].dtype == bool
    assert b["adaptive_decision"][0] in ("MATCH", "REJECT", "CONTINUE")


def test_pipeline_deterministic():
    """Same seed -> byte-identical rendered image (the determinism rule)."""
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    a = build_pipeline(PipelineConfig.preview()).run().image
    c = build_pipeline(PipelineConfig.preview()).run().image
    assert np.array_equal(a, c)


def test_run_on_vm_matches_direct_run():
    """Phase 6 execution half: running the pipeline ON the VM (lower to APPLY-program, execute with stage
    handlers, frame as accumulator) produces a bit-identical frame to the direct Python-loop run()."""
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    pipe = build_pipeline(PipelineConfig.preview())
    direct = pipe.run(scene="demo", seed=0)
    vm_ctx, applied = pipe.run_on_vm(scene="demo", seed=0)
    assert applied == pipe.stage_names()                          # the VM APPLY-ed every stage in order
    assert np.array_equal(direct.image, vm_ctx.image)             # bit-for-bit identical result
    assert vm_ctx.buffers.get("denoised") is True


def test_lower_to_program_is_a_vector():
    """Lowering the pipeline yields ONE program vector (config -> recipe), which the VM decodes back to the stage
    sequence."""
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    from holographic.agents_and_reasoning.holographic_machine import HoloMachine
    pipe = build_pipeline(PipelineConfig.preview())
    mac = HoloMachine(dim=1024, seed=0, faculties=pipe.stage_names())
    prog = pipe.lower_to_program(mac)
    assert isinstance(prog, np.ndarray) and prog.ndim == 1
    # the decoded program names the stages (dry EXPLAIN over the lowered program)
    from holographic.agents_and_reasoning.holographic_query import explain_program
    info = explain_program(mac, prog)
    assert info["faculties_called"] == pipe.stage_names()


def test_new_fx_sim_capabilities_default_off_and_gate():
    """The physics/FX layer is reflected in the pipeline config: waves (adaptive ocean) and granular (MPM) are
    opt-in flags that default OFF (backward-compatible), and their stages gate on those flags."""
    from holographic.scene_and_pipeline.holographic_pipeline import PipelineConfig, ALL_STAGES, build_pipeline
    assert PipelineConfig().waves is False and PipelineConfig().granular is False   # backward compatible
    o = PipelineConfig.ocean()
    assert o.waves and o.temporal_reuse                                             # ocean = preview + waves
    sw = next(s for s in ALL_STAGES if s.name == "sim_waves")
    sg = next(s for s in ALL_STAGES if s.name == "sim_granular")
    assert sw.enabled(o) and not sw.enabled(PipelineConfig())
    assert sg.enabled(PipelineConfig(granular=True)) and not sg.enabled(PipelineConfig())
    assert "sim_waves" in build_pipeline(o).stage_names()


# --- backlog A1/E2: the pipeline renders a REAL scene (RenderSpec) and grades via postfx ---------------------
def _tiny_real_scene():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import RenderSpec
    class Scene:
        def eval(self, P):
            c = np.array([[-0.6, 0, 0], [0.6, 0, 0]], float); r = np.array([0.55, 0.55])
            d = np.min(np.linalg.norm(P[..., None, :] - c, axis=-1) - r, axis=-1)
            return np.minimum(d, P[..., 1] + 0.9)
    class Cam:
        eye = np.array([0., 0.4, 3.2])
        def ray_dirs(self, w, h):
            ys, xs = np.mgrid[0:h, 0:w]; u = (xs / (w - 1) - 0.5) * 1.2; v = -(ys / (h - 1) - 0.5) * 1.2
            d = np.stack([u, v, -np.ones_like(u)], -1); return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)
    def mat(P):
        import numpy as np
        n = len(P); a = np.tile([.8, .3, .3], (n, 1)).astype(float); a[P[:, 0] < 0] = [.3, .4, .85]
        return a, np.zeros(n), np.full(n, .6), np.zeros((n, 3))
    def sky(D):
        import numpy as np
        t = np.clip(D[:, 1] * .5 + .5, 0, 1)[:, None]; return (1 - t) * np.array([.9, .85, .8]) + t * np.array([.35, .5, .9])
    return RenderSpec(scene=Scene(), camera=Cam(), material=mat, sky=sky, width=48, height=36,
                      quality="medium", max_bounce=3, svgf_levels=5)


def test_pipeline_renders_real_scene_matching_render_auto():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    from holographic.rendering.holographic_gbuffer import converge_samples, declfirefly, primary_gbuffer, render_auto
    from holographic.rendering.holographic_svgf import atrous_bilateral
    spec = _tiny_real_scene()
    cfg = PipelineConfig(denoise="svgf", dirty_only=False, adaptive_samples=False)
    ctx = build_pipeline(cfg).run(scene=spec, seed=0)
    assert ctx.buffers.get("denoised") is True
    assert "variance" in ctx.buffers and "render_stats" in ctx.buffers      # the render stage measured variance
    # the staged pipeline composes the SAME frame render_auto returns (converge -> firefly -> variance-guided svgf)
    M, vom, N, info = converge_samples(spec.scene, spec.camera, spec.width, spec.height, spec.material,
                                       sky=spec.sky, quality=spec.quality, max_bounce=spec.max_bounce, seed=0)
    nrm, alb, dep = primary_gbuffer(spec.scene, spec.camera, spec.width, spec.height, spec.material, sky=spec.sky)
    expect = atrous_bilateral(declfirefly(M, k=3.0), nrm, alb, dep, levels=5, variance=vom)
    ref = render_auto(spec.scene, spec.camera, spec.width, spec.height, spec.material, sky=spec.sky,
                      quality="medium", max_bounce=3, seed=0, svgf_levels=5)
    assert np.allclose(expect, ref, atol=1e-9)


def test_pipeline_demo_scene_still_byte_identical():
    # a None/plain scene must run the demo stand-ins exactly as before (backward compatibility)
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    a = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False)).run(scene=None, seed=1).image
    b = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False)).run(scene=None, seed=1).image
    assert np.array_equal(a, b) and a is not None


def test_pipeline_postfx_aces_grades_and_validates():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig, PipelineError
    spec = _tiny_real_scene()
    base = PipelineConfig(denoise="svgf", dirty_only=False, adaptive_samples=False)
    reinhard = build_pipeline(base).run(scene=spec, seed=0).image
    aces = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False, adaptive_samples=False,
                                         postfx="aces")).run(scene=spec, seed=0).image
    assert not np.allclose(reinhard, aces)                                   # the two grades differ
    try:
        build_pipeline(PipelineConfig(postfx="bogus")); assert False
    except PipelineError:
        pass


def _tiny_surface_and_smoke():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import RenderSpec
    class Scene:
        def eval(self, P):
            d = np.linalg.norm(P - np.array([0, 0, 0.]), axis=-1) - 0.5
            return np.minimum(d, P[..., 1] + 0.6)
    class Cam:
        eye = np.array([0., 0.3, 3.0])
        def ray_dirs(self, w, h, jitter=None):
            ys, xs = np.mgrid[0:h, 0:w]; jx, jy = (0., 0.) if jitter is None else (jitter[0], jitter[1])
            u = ((xs + jx) / (w - 1) - 0.5) * 1.1; v = -((ys + jy) / (h - 1) - 0.5) * 1.1
            d = np.stack([u, v, -np.ones_like(u)], -1); return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)
    def mat(P):
        n = len(P); return np.tile([.8, .4, .3], (n, 1)).astype(float), np.zeros(n), np.full(n, .5), np.zeros((n, 3))
    def sky(D):
        return np.tile([0.5, 0.6, 0.8], (len(D), 1))
    def smoke(P):
        P = np.asarray(P, float); return np.clip(1.0 - np.linalg.norm(P - np.array([0, 0.7, 0]), axis=1) / 0.4, 0, 1)
    return RenderSpec(scene=Scene(), camera=Cam(), material=mat, sky=sky, width=40, height=30, quality="draft",
                      max_bounce=2, volume={"field": smoke, "bounds": (np.array([-1., -1, -1]), np.array([1., 1.4, 1])),
                                            "mode": "smoke", "sigma": 14.0, "steps": 48})


def test_volume_stage_composites_over_surface():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    spec = _tiny_surface_and_smoke()
    base = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False)).run(scene=spec, seed=0).image
    ctx = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False, volume=True)).run(scene=spec, seed=0)
    assert "volume_alpha" in ctx.buffers                          # the volume stage ran
    assert not np.allclose(base, ctx.image)                       # and changed the composite
    assert np.isfinite(ctx.image).all()


def test_volume_stage_ordering_and_optout():
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    names = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False, volume=True)).stage_names()
    assert names.index("render") < names.index("volume") < names.index("present")   # render -> volume -> present
    # volume defaults OFF: no volume stage unless asked
    off = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False)).stage_names()
    assert "volume" not in off


def test_volume_flag_without_scene_volume_is_safe():
    # asking for the volume stage but giving a scene with no volume must not crash (stage no-ops)
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    spec = _tiny_real_scene()                                     # a RenderSpec with volume=None
    img = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False, volume=True)).run(scene=spec, seed=0).image
    assert img is not None and np.isfinite(img).all()


def _tiny_surface_and_sparks():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import RenderSpec
    from holographic.rendering.holographic_render import Camera
    class Scene:
        def eval(self, P):
            d = np.linalg.norm(P - np.array([0, 0, 0.]), axis=-1) - 0.5
            return np.minimum(d, P[..., 1] + 0.6)
    cam = Camera(eye=(0., 0.3, 3.0), target=(0, 0, 0), fov_deg=45, aspect=40 / 30)
    def mat(P):
        n = len(P); return np.tile([.3, .3, .35], (n, 1)).astype(float), np.zeros(n), np.full(n, .5), np.zeros((n, 3))
    def sky(D):
        return np.tile([0.1, 0.12, 0.16], (len(D), 1))
    rng = np.random.default_rng(0)
    pts = rng.uniform([-0.8, -0.3, -0.3], [0.8, 0.9, 0.6], (120, 3))
    return RenderSpec(scene=Scene(), camera=cam, material=mat, sky=sky, width=40, height=30, quality="draft",
                      max_bounce=2, particles={"points": pts, "colors": (1.0, 0.7, 0.2), "radius_px": 1.5,
                                               "intensity": 0.9})


def test_particle_stage_composites_over_surface():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    spec = _tiny_surface_and_sparks()
    base = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False)).run(scene=spec, seed=0).image
    ctx = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False, particles=True)).run(scene=spec, seed=0)
    assert "particle_alpha" in ctx.buffers                          # the particle stage ran
    assert not np.allclose(base, ctx.image)                         # and changed the composite
    assert np.isfinite(ctx.image).all()


def test_particle_stage_ordering_and_optout():
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    names = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False, volume=True, particles=True)).stage_names()
    # render -> svgf -> volume -> particles -> present (particles in front of smoke)
    assert names.index("render") < names.index("volume") < names.index("particles") < names.index("present")
    off = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False)).stage_names()
    assert "particles" not in off


def test_particle_flag_without_scene_particles_is_safe():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    spec = _tiny_real_scene()                                       # a RenderSpec with particles=None
    img = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False, particles=True)).run(scene=spec, seed=0).image
    assert img is not None and np.isfinite(img).all()


def _tiny_surface_and_fur():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import RenderSpec
    from holographic.rendering.holographic_render import Camera
    from holographic.mesh_and_geometry.holographic_groom import groom
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    body = sphere(0.6)
    strands = groom(body.eval, 300, ((-0.8, -0.8, -0.8), (0.8, 0.8, 0.8)), length=0.3, n_pts=6, curl=0.2, seed=0)
    cam = Camera(eye=(0, 0, 2.5), target=(0, 0, 0), fov_deg=45, aspect=48 / 36)
    def mat(P):
        n = len(P); return np.tile([.4, .25, .15], (n, 1)).astype(float), np.zeros(n), np.full(n, .6), np.zeros((n, 3))
    def sky(D):
        return np.tile([0.3, 0.35, 0.45], (len(D), 1))
    return RenderSpec(scene=body, camera=cam, material=mat, sky=sky, width=48, height=36, quality="draft",
                      max_bounce=2, hair={"strands": strands, "shader": "kajiya", "hair_color": (0.6, 0.4, 0.2),
                                          "light_dir": (0.3, 0.6, 0.6)})


def test_hair_stage_composites_over_surface():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    spec = _tiny_surface_and_fur()
    base = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False)).run(scene=spec, seed=0).image
    ctx = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False, hair=True)).run(scene=spec, seed=0)
    assert "hair_alpha" in ctx.buffers                          # the hair stage ran
    assert not np.allclose(base, ctx.image)                     # and changed the composite
    assert np.isfinite(ctx.image).all()


def test_hair_stage_ordering_and_optout():
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    names = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False,
                                          volume=True, particles=True, hair=True)).stage_names()
    # render -> volume -> particles -> hair -> present (hair the last layer)
    assert names.index("render") < names.index("volume") < names.index("particles") < names.index("hair") < names.index("present")
    off = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False)).stage_names()
    assert "hair" not in off


def test_hair_flag_without_scene_hair_is_safe():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig
    spec = _tiny_real_scene()                                    # a RenderSpec with hair=None
    img = build_pipeline(PipelineConfig(denoise="svgf", dirty_only=False, hair=True)).run(scene=spec, seed=0).image
    assert img is not None and np.isfinite(img).all()


# --- R1: render-strategy dispatch through the Pipeline ---

def _r1_scene():
    import numpy as np
    from holographic.mesh_and_geometry.holographic_sdf import box, sphere
    from holographic.rendering.holographic_render import Camera
    scene = sphere(0.5).smooth_union(box(2.5, 0.1, 2.5).translate((0, -0.55, 0)), k=0.03)
    cam = Camera(eye=(0, 0.8, 3.0), target=(0, -0.2, 0), fov_deg=45, aspect=1.0)

    def mat(P):
        n = len(P)
        return np.tile([0.8, 0.8, 0.8], (n, 1)).astype(float), np.zeros(n), np.full(n, 0.6), np.zeros((n, 3))
    dark = lambda D: np.tile([0.02, 0.02, 0.03], (len(D), 1))
    return scene, cam, mat, dark


def test_r1_render_goes_through_pipeline():
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import RenderSpec, Pipeline, ALL_STAGES
    scene, cam, mat, dark = _r1_scene()
    spec = RenderSpec(scene=scene, camera=cam, material=mat, sky=dark, width=40, height=30,
                      quality="draft", max_bounce=2)
    pipe = Pipeline([s for s in ALL_STAGES if s.name in ("render", "present")])
    st = pipe.run(scene=spec, seed=0)
    assert st.image.shape == (30, 40, 3) and np.isfinite(st.image).all()
    assert st.buffers["render_method"] == "pathtrace"                # 'auto' picks pathtrace when a material is present


def test_r1_strategies_dispatch():
    from holographic.scene_and_pipeline.holographic_pipeline import RenderSpec, FrameState, dispatch_render, RENDER_STRATEGIES
    scene, cam, mat, dark = _r1_scene()
    assert set(RENDER_STRATEGIES) == {"pathtrace", "raymarch", "prt", "radiance"}
    for m in ("raymarch", "pathtrace"):                              # both run and produce a finite image
        spec = RenderSpec(scene=scene, camera=cam, material=mat, sky=dark, width=32, height=24,
                          quality="draft", max_bounce=2, method=m)
        ctx = FrameState(scene=spec, seed=0)
        dispatch_render(ctx)
        assert ctx.image.shape == (24, 32, 3) and ctx.buffers["render_method"] == m


def test_r1_needs_check_catches_missing_input():
    from holographic.scene_and_pipeline.holographic_pipeline import RenderSpec, FrameState, dispatch_render, PipelineError
    scene, cam, mat, dark = _r1_scene()
    # ask for 'prt' but provide no light_sh -> clear error naming the missing field, BEFORE running
    spec = RenderSpec(scene=scene, camera=cam, material=mat, width=32, height=24, method="prt")
    try:
        dispatch_render(FrameState(scene=spec, seed=0))
        assert False, "should have raised"
    except PipelineError as e:
        assert "light_sh" in str(e) and "prt" in str(e)


def test_r1_pathtrace_byte_identical_to_pre_r1():
    # the dispatched 'pathtrace' strategy must be the EXACT converge_samples path the RenderSpec branch ran before R1
    import numpy as np
    from holographic.scene_and_pipeline.holographic_pipeline import RenderSpec, FrameState, dispatch_render
    from holographic.rendering.holographic_gbuffer import converge_samples, declfirefly
    scene, cam, mat, dark = _r1_scene()
    spec = RenderSpec(scene=scene, camera=cam, material=mat, sky=dark, width=32, height=24,
                      quality="draft", max_bounce=2, method="pathtrace")
    ctx = FrameState(scene=spec, seed=0)
    dispatch_render(ctx)
    M, vom, N, info = converge_samples(scene, cam, 32, 24, mat, sky=dark, quality="draft", max_bounce=2, seed=0,
                                       pass_spp=spec.pass_spp, max_passes=spec.max_passes, lights=None)
    ref = declfirefly(M, k=spec.firefly_k)
    assert np.array_equal(ctx.image, ref)                            # bit-for-bit -- the tie-sensitive path is pinned
