"""Phases 1-2: the configurable render/sim pipeline -- select, auto-include, reject, order, plan, run."""
import numpy as np
import pytest
from holographic_pipeline import (PipelineConfig, Pipeline, build_pipeline, PipelineError, ALL_STAGES)


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
    from holographic_pipeline import build_pipeline, PipelineConfig
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
    from holographic_pipeline import build_pipeline, PipelineConfig
    a = build_pipeline(PipelineConfig.preview()).run().image
    c = build_pipeline(PipelineConfig.preview()).run().image
    assert np.array_equal(a, c)


def test_run_on_vm_matches_direct_run():
    """Phase 6 execution half: running the pipeline ON the VM (lower to APPLY-program, execute with stage
    handlers, frame as accumulator) produces a bit-identical frame to the direct Python-loop run()."""
    import numpy as np
    from holographic_pipeline import build_pipeline, PipelineConfig
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
    from holographic_pipeline import build_pipeline, PipelineConfig
    from holographic_machine import HoloMachine
    pipe = build_pipeline(PipelineConfig.preview())
    mac = HoloMachine(dim=1024, seed=0, faculties=pipe.stage_names())
    prog = pipe.lower_to_program(mac)
    assert isinstance(prog, np.ndarray) and prog.ndim == 1
    # the decoded program names the stages (dry EXPLAIN over the lowered program)
    from holographic_query import explain_program
    info = explain_program(mac, prog)
    assert info["faculties_called"] == pipe.stage_names()


def test_new_fx_sim_capabilities_default_off_and_gate():
    """The physics/FX layer is reflected in the pipeline config: waves (adaptive ocean) and granular (MPM) are
    opt-in flags that default OFF (backward-compatible), and their stages gate on those flags."""
    from holographic_pipeline import PipelineConfig, ALL_STAGES, build_pipeline
    assert PipelineConfig().waves is False and PipelineConfig().granular is False   # backward compatible
    o = PipelineConfig.ocean()
    assert o.waves and o.temporal_reuse                                             # ocean = preview + waves
    sw = next(s for s in ALL_STAGES if s.name == "sim_waves")
    sg = next(s for s in ALL_STAGES if s.name == "sim_granular")
    assert sw.enabled(o) and not sw.enabled(PipelineConfig())
    assert sg.enabled(PipelineConfig(granular=True)) and not sg.enabled(PipelineConfig())
    assert "sim_waves" in build_pipeline(o).stage_names()
