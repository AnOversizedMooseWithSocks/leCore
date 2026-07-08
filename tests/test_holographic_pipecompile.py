"""Performance PW1/PW2: compile the pipeline plan once per config; per-stage bake hook."""
from holographic.scene_and_pipeline.holographic_pipeline import PipelineConfig, build_pipeline
from holographic.scene_and_pipeline.holographic_compile import CompileCache
from holographic.scene_and_pipeline.holographic_pipecompile import compiled_pipeline, run_compiled, bake_pipeline, config_spec


def test_compiled_matches_build():
    cfg = PipelineConfig.preview()
    assert compiled_pipeline(cfg, cache=CompileCache()).stage_names() == build_pipeline(cfg).stage_names()


def test_plan_compiled_once_across_frames():
    cache = CompileCache(); cfg = PipelineConfig.preview()
    for _ in range(10):
        run_compiled(cfg, scene={"objs": 3}, cache=cache)
    assert cache.stats["compiles"] == 1 and cache.stats["hits"] == 9   # one plan build, nine reuses


def test_different_config_recompiles():
    cache = CompileCache()
    compiled_pipeline(PipelineConfig.preview(), cache=cache)
    compiled_pipeline(PipelineConfig.final(), cache=cache)
    assert cache.stats["compiles"] == 2


def test_same_config_value_shares_plan():
    # two separately-constructed but identical configs share one compiled plan (content-addressed, not identity)
    cache = CompileCache()
    compiled_pipeline(PipelineConfig.preview(), cache=cache)
    compiled_pipeline(PipelineConfig.preview(), cache=cache)
    assert cache.stats["compiles"] == 1 and cache.stats["hits"] == 1


def test_options_change_key():
    cache = CompileCache()
    cfg = PipelineConfig.preview()
    compiled_pipeline(cfg, options={"bake_res": 64}, cache=cache)
    compiled_pipeline(cfg, options={"bake_res": 128}, cache=cache)
    assert cache.stats["compiles"] == 2                              # different options -> different plan key


def test_config_spec_deterministic():
    cfg = PipelineConfig.preview()
    assert config_spec(cfg) == config_spec(cfg)


def test_bake_hook_runs_once():
    pipe = compiled_pipeline(PipelineConfig.preview(), cache=CompileCache())
    calls = {"n": 0}
    pipe.stages[0].bake = lambda scene, seed: calls.__setitem__("n", calls["n"] + 1) or {"grid": 7}
    baked = bake_pipeline(pipe, scene={"objs": 3})
    assert calls["n"] == 1 and baked[pipe.stages[0].name] == {"grid": 7}


def test_stages_without_bake_skipped():
    pipe = compiled_pipeline(PipelineConfig.final(), cache=CompileCache())
    assert bake_pipeline(pipe, scene={}) == {}                       # no stage declares a bake -> empty
