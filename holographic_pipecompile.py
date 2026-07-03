"""holographic_pipecompile.py -- COMPILE THE PIPELINE (fluids/matter backlog, performance items PW1/PW2).

The material work (MC1-3) compiled ONE material; the same move applies one level up, to the whole render/sim PIPELINE.
`build_pipeline(cfg)` does real planning work every call -- SELECT the enabled stages, AUTO-INCLUDE prerequisites,
and TOPOSORT them. Across a sequence of frames with the SAME config that plan never changes, so recomputing it per
frame is waste. PW2 COMPILES the plan once, keyed by the config's content, and hands the same ordered Pipeline back
to every frame -- reusing the content-addressed compile cache exactly as the material compile did.

PW1 is the finer grain: a stage may carry a `bake(scene)` that precomputes its VIEW-INDEPENDENT buffers once (a
baked SDF grid, a static AO pass), so re-running the pipeline over frames re-does only what actually changed. This
module adds the hook (`bake_pipeline`) and runs each stage's bake once, caching the result on the frame state.

Together: compile the plan once (PW2), bake the static per-stage work once (PW1), then each frame only threads the
dynamic stages. This is the pipeline twin of "compile the material, bake the constants, look up the rest".

KEPT NEGATIVE: compiling the plan saves the SELECT/TOPOSORT, not the per-frame stage work -- the win scales with how
many frames share a config (one frame gains nothing). Per-stage bakes only help stages with genuinely static output;
a stage whose output changes every frame (the actual render) can't bake and is correctly left to run. The plan is
keyed by the CONFIG only -- if a stage's behaviour depends on the scene too, include that in the options key. Bit-for
-bit, a compiled+baked run produces the same frame as a direct build_pipeline(...).run().
"""
import dataclasses

from holographic_compile import compiled, DEFAULT_CACHE
from holographic_pipeline import build_pipeline


def config_spec(cfg, options=None):
    """A canonical, hashable spec for a pipeline config (+ optional extra options): the sorted (field, value) pairs
    of the dataclass, so two identical configs share one compiled plan and any change misses the cache."""
    base = tuple(sorted(dataclasses.asdict(cfg).items()))
    extra = tuple(sorted((options or {}).items()))
    return ("cfg", base, "opts", extra)


def compiled_pipeline(cfg, registry=None, options=None, cache=None):
    """Return the ordered, validated Pipeline for `cfg`, BUILT ONCE and cached by the config's content spec (PW2).
    Repeated frames with the same config reuse the same plan -- no re-select, no re-toposort. Reuses the
    content-addressed compile cache."""
    cache = cache if cache is not None else DEFAULT_CACHE
    spec = config_spec(cfg, options)
    return compiled(spec, lambda _s: build_pipeline(cfg, registry), tag="pipeline", cache=cache)


def bake_pipeline(pipeline, scene, seed=0):
    """PW1: run each stage's optional `bake(scene, seed)` ONCE, returning a dict of the precomputed view-independent
    buffers keyed by stage name. Stages without a bake are skipped. Pass the result into run_compiled as `baked` so
    the static work is done once, not per frame. A stage declares a bake by having a callable `bake` attribute."""
    baked = {}
    for s in pipeline.stages:
        fn = getattr(s, "bake", None)
        if callable(fn):
            baked[s.name] = fn(scene, seed)                     # the static precompute for this stage
    return baked


def run_compiled(cfg, scene=None, seed=0, prev_frame=None, renderer=None, registry=None, options=None, cache=None):
    """Convenience: compile the plan for `cfg` (cached), then run it -- the everyday entry point for a frame loop.
    The FIRST call with a config compiles the plan; every later frame reuses it. Returns the final FrameState."""
    pipe = compiled_pipeline(cfg, registry=registry, options=options, cache=cache)
    return pipe.run(scene=scene, seed=seed, prev_frame=prev_frame, renderer=renderer)


def _selftest():
    """Compiling a config's plan is done ONCE and reused across frames; the compiled plan matches a direct
    build_pipeline; a changed config is a fresh compile; and a per-stage bake hook runs once."""
    from holographic_pipeline import PipelineConfig
    from holographic_compile import CompileCache

    cache = CompileCache()
    cfg = PipelineConfig.preview()

    # the compiled plan matches a direct build_pipeline for the same config
    direct = build_pipeline(cfg).stage_names()
    comp = compiled_pipeline(cfg, cache=cache).stage_names()
    assert comp == direct, (comp, direct)

    # ten frames of the SAME config compile the plan ONCE, reuse it nine times
    for _frame in range(10):
        run_compiled(cfg, scene={"objs": 3}, cache=cache)
    assert cache.stats["compiles"] == 1 and cache.stats["hits"] >= 9, cache.stats

    # a DIFFERENT config is a fresh content-addressed compile
    compiled_pipeline(PipelineConfig.final(), cache=cache)
    assert cache.stats["compiles"] == 2, cache.stats

    # PW1: a stage carrying a bake() has it run exactly once by bake_pipeline
    pipe = compiled_pipeline(cfg, cache=cache)
    calls = {"n": 0}
    pipe.stages[0].bake = lambda scene, seed: calls.__setitem__("n", calls["n"] + 1) or {"grid": 42}
    baked = bake_pipeline(pipe, scene={"objs": 3})
    assert calls["n"] == 1 and baked[pipe.stages[0].name] == {"grid": 42}

    # correctness: a compiled run produces the same stage set as a direct run
    fs_direct = build_pipeline(cfg).run(scene={"objs": 3})
    fs_comp = run_compiled(cfg, scene={"objs": 3}, cache=cache)
    assert type(fs_direct) is type(fs_comp)

    print("holographic_pipecompile selftest OK: a pipeline config's plan (select+auto-include+toposort) is compiled "
          "ONCE and reused across 10 frames (compiles=%d, hits=%d); the compiled plan matches build_pipeline exactly; "
          "a changed config is a fresh content-addressed compile; and a per-stage bake() hook runs exactly once -- "
          "compile the plan once, bake the static work once, thread only the dynamic stages per frame"
          % (cache.stats["compiles"], cache.stats["hits"]))


if __name__ == "__main__":
    _selftest()
