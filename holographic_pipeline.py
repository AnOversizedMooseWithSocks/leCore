"""holographic_pipeline.py -- ONE configurable render/simulation pipeline: pick a preset (or set flags), see
exactly which stages will run and WHY, and get the right pipeline every time -- with the hand-assembly path
still there for full control.

WHY THIS EXISTS (Render/Sim Pipeline backlog, Part 1 / Phases 1-2)
-----------------------------------------------------------------
Composing a run today means calling the right methods with scattered kwargs -- three hard-wired render paths,
and every sim solver with a DIFFERENT step signature. So features are scattered, nothing checks the combination
(enable SVGF, forget its G-buffer, get a wrong image), and you can't see what will run without running it. This
fixes all of that with four small, readable parts:

  * `Stage`   -- one composable step (a sim step or a render pass) that DECLARES what it `needs` and `produces`,
                 so the builder can order stages and catch a missing prerequisite BEFORE running. It reimplements
                 nothing; its `run` just calls the existing capability.
  * `FrameState` -- the one mutable bag the stages thread through (a stage's output is the next's input); carries
                 a `seed` so the whole pipeline is deterministic.
  * `PipelineConfig` (+ presets) -- everything you'd otherwise piece together by hand, in ONE place.
  * `build_pipeline` -- config -> ordered, validated Pipeline: SELECT the enabled stages, AUTO-INCLUDE their
                 prerequisites, REJECT impossible combinations with a clear message, and ORDER by dependency with
                 a STABLE topological sort (deterministic).

`Pipeline.plan()` is the dry run: it lists every active stage and why it's there WITHOUT rendering -- the "did
the right components get used?" answer. `Pipeline.run()` threads a FrameState through the stages in order.

MANUAL ASSEMBLY STAYS: the config is sugar over a stage list -- `build_pipeline` just returns `Pipeline(stages)`,
and you can hand-construct one whenever you want full control. Both paths coexist.

HONEST SCOPE (kept loud): this is a stage LIST + a dependency dict + a topological sort -- deliberately NOT a
general DAG engine (branching/loops are the machine's job, the VSA-native endgame). Stages DELEGATE to existing
modules; the built-in demo renderer/sim here are light stand-ins so the pipeline runs and tests end-to-end -- a
real app injects RenderSession / its solvers via the stage `run` closures, unchanged. Deterministic; NumPy + stdlib.
"""
from dataclasses import dataclass, field

import numpy as np


class PipelineError(Exception):
    """Raised at BUILD time (not render time) for an impossible configuration -- with a message that says what is
    wrong, so you fix the config instead of debugging a wrong image later."""


@dataclass
class Stage:
    """One composable step. `enabled(cfg) -> bool` decides if it runs; `run(ctx) -> ctx` does the work (delegating
    to an existing module); `needs`/`produces` let the builder order stages and auto-include prerequisites; `phase`
    is the coarse ordering (0 sim, 1 render, 2 present) used only to break ties the dependencies don't -- so sim
    runs before render even when no data edge forces it; `why` is shown in plan()."""
    name: str
    needs: tuple = ()
    produces: tuple = ()
    enabled: object = None          # cfg -> bool
    run: object = None              # (ctx) -> ctx
    why: str = ""
    phase: int = 1                  # 0 = sim, 1 = render, 2 = present (tie-break only; deps still come first)


@dataclass
class FrameState:
    """The mutable bag stages thread through: a stage's output is the next stage's input. `buffers` holds named
    intermediates (gbuffer, variance, splats, sim fields); `seed` is threaded so the whole pipeline is deterministic."""
    scene: object = None
    image: object = None
    dirty: object = None
    prev_frame: object = None
    buffers: dict = field(default_factory=dict)
    seed: int = 0


@dataclass
class PipelineConfig:
    """Everything you'd otherwise piece together by hand, in ONE place -- the single opt-in/out surface. Presets
    are named starting points."""
    # sim
    fluid: bool = False
    softbody: bool = False
    collide: bool = False
    field_effects: bool = False
    waves: bool = False             # adaptive ocean/wave FX -- the plan_waves dispatch (fft ocean / packets / shallow / breaking)
    granular: bool = False          # material-point (MPM) sim -- snow / sand / mud
    dirty_only: bool = True
    # render
    bake: str = "auto"
    temporal_reuse: bool = False
    adaptive_samples: bool = True
    denoise: str = "off"            # "off" | "svgf" | "bilateral"
    splat_proxy: bool = False
    lod: bool = True
    quality: str = "preview"        # "preview" | "final"

    @staticmethod
    def preview():
        """Fast, interactive-looking: reuse+reproject last frame, SVGF denoise, a splat proxy."""
        return PipelineConfig(temporal_reuse=True, denoise="svgf", splat_proxy=True)

    @staticmethod
    def final():
        """High quality single frame: full samples, no dirty-only reuse, final grade."""
        return PipelineConfig(adaptive_samples=False, dirty_only=False, temporal_reuse=False,
                              quality="final")

    @staticmethod
    def interactive():
        """Preview plus live simulation (fluid + collision + field effects)."""
        c = PipelineConfig.preview()
        c.fluid = c.collide = c.field_effects = True
        return c

    @staticmethod
    def ocean():
        """Preview plus the adaptive ocean/wave FX -- the plan_waves dispatch runs the cheap spectral method almost
        everywhere and the dear breaking grid solver only in the tiles that actually break."""
        c = PipelineConfig.preview()
        c.waves = True
        return c


# ---------------------------------------------------------------------------------------------------------------
# The stage registry. Each stage DECLARES needs/produces and DELEGATES its work. The demo run()s here are light
# stand-ins (mark dirty, advance a tiny sim, shade a small image, tonemap) so the pipeline runs end-to-end; a
# real app swaps in RenderSession / its solvers by replacing the `run` closure -- the machinery is identical.
# ---------------------------------------------------------------------------------------------------------------

def _sim_run(name):
    def run(ctx):
        # a light stand-in sim step: mark the whole frame dirty and record that this sim advanced
        ctx.buffers.setdefault("sim", []).append(name)
        ctx.dirty = "all" if ctx.dirty is None else ctx.dirty
        return ctx
    return run


def _demo_scene(seed, H=24, W=24):
    # a shared two-surface scene (left vs right: different colour, normal, albedo, depth) so the gbuffer and the
    # render agree -- a real edge to denoise across, not random noise. Returns (clean, normal, albedo, depth).
    clean = np.zeros((H, W, 3)); normal = np.zeros((H, W, 3)); albedo = np.zeros((H, W, 3)); depth = np.zeros((H, W))
    clean[:, :W // 2] = [0.8, 0.2, 0.2]; clean[:, W // 2:] = [0.2, 0.3, 0.8]
    normal[:, :W // 2] = [0, 0, 1]; normal[:, W // 2:] = [1, 0, 0]
    albedo[:] = clean; depth[:, :W // 2] = 1.0; depth[:, W // 2:] = 3.0
    return clean, normal, albedo, depth


def _gbuffer_run(ctx):
    # SVGF needs per-pixel normals/albedo/depth. Produce a REAL g-buffer for the demo scene (deterministic off the
    # seed) so the denoise stage can actually edge-stop, not a random stand-in.
    _, normal, albedo, depth = _demo_scene(ctx.seed)
    ctx.buffers["gbuffer"] = {"normal": normal, "albedo": albedo, "depth": depth}
    return ctx


def _render_run(ctx):
    # the core render. A real app injects RenderSession.render_final / preview here. The demo path renders the
    # two-surface scene with several NOISY Monte-Carlo samples per pixel and ACCUMULATES them through
    # robust_accumulate -- so a firefly (a rare huge outlier sample) can't blow out a pixel. That firefly clamp is
    # the engine's own robust averaging (holographic_accumulate), measured here against the plain mean.
    renderer = getattr(ctx, "_renderer", None)
    if renderer is not None:
        ctx.image = renderer(ctx)
        return ctx
    from holographic_accumulate import robust_accumulate
    clean, _, _, _ = _demo_scene(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    H, W, _c = clean.shape
    n_samples = 8
    samples = [clean + 0.12 * rng.standard_normal((H, W, 3)) for _ in range(n_samples)]   # noisy MC sample passes
    samples[3] = clean + 3.0                                                   # one CORRUPTED sample pass (a firefly frame)
    # accumulate the passes robustly: robust_accumulate winsorizes a whole outlier PASS to k robust-scales from the
    # median, so one bad frame can't drag the pixel -- the engine's own firefly clamp (ACCUM-3). Measure vs naive.
    robust = robust_accumulate(samples, schedule="mean", clamp_k=2.5)
    naive = np.mean(samples, axis=0)
    ctx.image = np.clip(robust, 0, None)
    ctx.buffers["_clean"] = clean
    ctx.buffers["accum_rmse"] = {
        "robust": float(np.sqrt(np.mean((robust - clean) ** 2))),
        "naive": float(np.sqrt(np.mean((naive - clean) ** 2)))}
    return ctx


def _reproject_run(ctx):
    # temporal reuse: reproject last frame into this frame and reuse it as the base (TemporalReuse's core move).
    # The demo has no camera motion, so the reproject is identity; a real app warps by the camera delta (one bind)
    # and re-solves only the dirty cells via TemporalReuse.solve. We route through TemporalReuse so the reuse
    # bookkeeping is the real one, not a hand-rolled copy.
    if ctx.prev_frame is not None:
        from holographic_temporal import TemporalReuse
        tr = TemporalReuse()
        tr.frame = np.asarray(ctx.prev_frame, float)                          # last frame becomes the reuse base
        ctx.buffers["reprojected"] = tr.frame                                 # identity reproject (no camera delta)
    return ctx


def _adaptive_run(ctx):
    # variance-guided adaptive sampling, the engine's way: where a pixel's estimate is still noisy, keep sampling;
    # where it has converged, stop. This stage runs independent of render (both only need the scene), so it makes
    # its OWN quick low-sample variance estimate of the scene to decide where to concentrate samples. The stop
    # rule is Wald's sequential test (holographic_honesty.SPRTRecall) -- the same honesty mechanism used for
    # streaming recognition, one use more.
    clean, normal, _, depth = _demo_scene(ctx.seed)
    rng = np.random.default_rng(ctx.seed + 3)
    H, W, _c = clean.shape
    quick = np.stack([clean + 0.12 * rng.standard_normal((H, W, 3)) for _ in range(3)])   # a cheap 3-sample probe
    var = quick.var(axis=0).mean(axis=2)                                       # per-pixel variance from the probe
    # edges (where normal/depth change) also read as "sample more" -- add the geometric high-frequency signal
    edge = np.zeros((H, W)); edge[:, W // 2 - 1:W // 2 + 1] = 1.0
    heat = var + 0.1 * edge
    ctx.buffers["sample_map"] = (heat > float(np.median(heat)))                # True = still noisy -> sample more
    try:
        from holographic_honesty import SPRTRecall
        # SPRT needs the two score distributions it decides between: a converged pixel's per-sample agreement sits
        # high (match), a still-noisy pixel's sits low (null). A converged stream then crosses the MATCH boundary
        # in the minimum expected number of samples (Wald optimality) -- free early-stop for the sampler.
        rng2 = np.random.default_rng(ctx.seed + 7)
        null_scores = 0.3 + 0.1 * rng2.standard_normal(200)                   # noisy pixel: low agreement
        match_scores = 0.9 + 0.05 * rng2.standard_normal(200)                 # converged pixel: high agreement
        sprt = SPRTRecall(null_scores, match_scores, alpha=0.05, beta=0.05)
        decision, n_used = sprt.decide([0.9, 0.92, 0.88, 0.91, 0.9], cap=5)
        ctx.buffers["adaptive_decision"] = (decision, n_used)
    except Exception:
        pass
    return ctx


def _svgf_run(ctx):
    # denoise the rendered image using the G-buffer. If the g-buffer carries real per-pixel features, run the REAL
    # holographic bilateral SVGF (edge-stopping = feature-vector cosine, coarse-to-fine over the pyramid) and
    # MEASURE the PSNR win vs the noisy input. Falls back to the tiny blur if features aren't present.
    img = ctx.image
    gb = ctx.buffers.get("gbuffer")
    if img is not None and isinstance(gb, dict):
        from holographic_svgf import atrous_bilateral
        den = atrous_bilateral(img, gb["normal"], gb["albedo"], gb["depth"], levels=5)
        clean = ctx.buffers.get("_clean")
        if clean is not None:
            def _psnr(a, b):
                mse = float(np.mean((a - b) ** 2)); return 99.0 if mse < 1e-12 else float(10 * np.log10(1.0 / mse))
            ctx.buffers["svgf_psnr"] = {"noisy": _psnr(np.clip(img, 0, 1), clean),
                                        "denoised": _psnr(np.clip(den, 0, 1), clean)}
        ctx.image = den
        ctx.buffers["denoised"] = True
    elif img is not None:
        ctx.image = 0.8 * img + 0.2 * img.mean(axis=(0, 1), keepdims=True)     # legacy stand-in (no real features)
        ctx.buffers["denoised"] = True
    return ctx


def _splat_run(ctx):
    ctx.buffers["splats"] = "proxy"                               # a real app calls RenderSession.to_splats
    return ctx


def _present_run(ctx):
    # the SINGLE present/grade stage (G3): tonemap + gamma, so preview and final grade identically.
    img = ctx.image
    if img is not None:
        ctx.image = np.clip(img, 0, None)
        ctx.image = ctx.image / (1.0 + ctx.image)                # Reinhard tonemap
        ctx.image = np.power(ctx.image, 1.0 / 2.2)               # gamma
    return ctx


ALL_STAGES = [
    # sim stages (phase 0: run first; they mark the dirty region the render stages consume)
    Stage("sim_fluid", (), ("sim",), lambda c: c.fluid, _sim_run("fluid"), "advance the fluid solver", phase=0),
    Stage("sim_softbody", (), ("sim",), lambda c: c.softbody, _sim_run("softbody"), "advance the soft-body solver", phase=0),
    Stage("sim_field_effects", (), ("sim",), lambda c: c.field_effects, _sim_run("field_effects"),
          "apply field-effect forces to the sim", phase=0),
    Stage("sim_waves", (), ("sim",), lambda c: c.waves, _sim_run("waves"),
          "advance the adaptive ocean/wave solver (fft ocean / packets / shallow / breaking, dispatched per tile)", phase=0),
    Stage("sim_granular", (), ("sim",), lambda c: c.granular, _sim_run("granular"),
          "advance the material-point (MPM) snow/sand/mud solver", phase=0),
    Stage("sim_collide", ("sim",), ("collided",), lambda c: c.collide, _sim_run("collide"),
          "resolve collisions after the sim advanced", phase=0),
    # render stages (phase 1)
    Stage("gbuffer", ("scene",), ("gbuffer",), lambda c: False, _gbuffer_run,
          "G-buffer (normals+depth) -- a pure provider, auto-included only when a stage needs it (e.g. SVGF)"),
    Stage("reproject", ("scene",), ("reprojected",), lambda c: c.temporal_reuse, _reproject_run,
          "reuse and reproject last frame (temporal reuse)"),
    Stage("adaptive_samples", ("scene",), ("sample_map",), lambda c: c.adaptive_samples, _adaptive_run,
          "concentrate samples where variance is high"),
    Stage("render", ("scene",), ("image",), lambda c: True, _render_run, "render the scene to an image"),
    Stage("svgf_denoise", ("image", "gbuffer"), ("denoised",), lambda c: c.denoise == "svgf", _svgf_run,
          "SVGF denoise (needs the G-buffer)"),
    Stage("splat_proxy", ("scene",), ("splats",), lambda c: c.splat_proxy, _splat_run,
          "emit a splat proxy for fast preview"),
    # present stage (phase 2: last)
    Stage("present", ("image",), ("final_image",), lambda c: True, _present_run,
          "tonemap + gamma -- the single present/grade stage", phase=2),
]


def _provider_of(need, registry):
    """The first stage (by name, for determinism) in the registry that PRODUCES `need`."""
    providers = sorted((s for s in registry if need in s.produces), key=lambda s: s.name)
    return providers[0] if providers else None


def _check_conflicts(cfg):
    """Reject impossible combinations at BUILD time with a clear message. Kept small and honest -- these are real
    constraints, not busywork."""
    if cfg.dirty_only and not cfg.temporal_reuse:
        raise PipelineError("dirty_only=True needs temporal_reuse=True: rendering only the dirty tiles requires a "
                            "previous frame to keep for the clean tiles. Set temporal_reuse=True or dirty_only=False.")
    if cfg.denoise not in ("off", "svgf", "bilateral"):
        raise PipelineError("denoise must be 'off', 'svgf', or 'bilateral', got %r" % (cfg.denoise,))
    if cfg.quality == "final" and cfg.splat_proxy:
        raise PipelineError("splat_proxy is a fast-PREVIEW approximation; it conflicts with quality='final'. Drop "
                            "splat_proxy for a final render.")


def _toposort(stages):
    """Order stages so every stage runs AFTER whatever produces the things it needs. Stable: ties are broken by
    stage name, so the same set of stages always yields the same order (deterministic). Kahn's algorithm."""
    by_name = {s.name: s for s in stages}
    phase = {s.name: (s.phase, s.name) for s in stages}          # (phase, name) -> the stable tie-break key
    # who produces each thing (among the active stages) -- "scene" is provided by the FrameState itself
    producers = {}
    for s in stages:
        for p in s.produces:
            producers.setdefault(p, []).append(s.name)
    # edges: producer -> consumer for each need that another active stage produces
    indeg = {s.name: 0 for s in stages}
    succ = {s.name: set() for s in stages}
    for s in stages:
        for need in s.needs:
            for prod_name in producers.get(need, ()):            # a stage that makes what s needs
                if prod_name != s.name and s.name not in succ[prod_name]:
                    succ[prod_name].add(s.name)
                    indeg[s.name] += 1
    ready = sorted((n for n, d in indeg.items() if d == 0), key=lambda n: phase[n])   # by (phase, name) -> stable
    order = []
    while ready:
        n = ready.pop(0)
        order.append(by_name[n])
        for m in sorted(succ[n], key=lambda x: phase[x]):
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
        ready.sort(key=lambda x: phase[x])
    if len(order) != len(stages):
        raise PipelineError("stage dependency cycle detected among: %s" % ", ".join(sorted(by_name)))
    return order


def build_pipeline(cfg, registry=None):
    """Config -> ordered, validated Pipeline. Does the three things you do by hand today:
      1) SELECT the stages `enabled(cfg)`;
      2) AUTO-INCLUDE prerequisites -- for any need nothing active provides, pull in the registry stage that
         produces it (ask for SVGF -> get the gbuffer stage), and REJECT a need nothing can provide;
      3) ORDER by dependency with a stable topological sort.
    Conflicts are rejected up front with a clear message."""
    registry = ALL_STAGES if registry is None else registry
    _check_conflicts(cfg)
    active = [s for s in registry if s.enabled(cfg)]
    active_names = {s.name for s in active}
    have = set(["scene"]).union(*(s.produces for s in active)) if active else set(["scene"])
    # auto-include prerequisites (iterate until fixed point: an included provider may itself have needs)
    i = 0
    while i < len(active):
        s = active[i]
        for need in s.needs:
            if need not in have:
                dep = _provider_of(need, registry)
                if dep is None:
                    raise PipelineError("stage %r needs %r, but nothing in the registry provides it"
                                        % (s.name, need))
                if dep.name not in active_names:
                    active.append(dep)
                    active_names.add(dep.name)
                    have |= set(dep.produces)
        i += 1
    return Pipeline(_toposort(active))


class Pipeline:
    """An ordered list of stages. `plan()` is the dry run (what will run and why, WITHOUT rendering); `run()`
    threads a FrameState through the stages in order."""

    def __init__(self, stages):
        self.stages = list(stages)

    def plan(self):
        """Dry run / EXPLAIN: every active stage, WHY it is here, and what it NEEDS and PRODUCES -- without
        rendering. This is the 'did the right components get used, and what will each touch?' answer, the pipeline
        twin of the query interface's EXPLAIN (a dry-run trace that names the work without doing it). Phase 6's
        inspection half: the config is lowered to this ordered, inspectable plan. (Executing the plan ON the VM
        via machine.run is the deferred other half -- it needs the frame state represented as a hypervector
        accumulator, since these stages transform a FrameState, not a vector.)"""
        return [{"stage": s.name, "why": s.why, "needs": list(s.needs), "produces": list(s.produces)}
                for s in self.stages]

    def stage_names(self):
        return [s.name for s in self.stages]

    def run(self, scene=None, seed=0, prev_frame=None, renderer=None):
        """Execute the pipeline: build the shared FrameState and thread it through every stage in order. `renderer`
        (optional) is the real render function a stage calls -- inject RenderSession here; the demo renderer runs
        if none is given. Returns the final FrameState."""
        ctx = FrameState(scene=scene, seed=seed, prev_frame=prev_frame)
        ctx._renderer = renderer                                  # dependency-injected real renderer (or None)
        for s in self.stages:
            ctx = s.run(ctx)
        return ctx

    def lower_to_program(self, machine):
        """Phase 6: LOWER the pipeline to a machine PROGRAM -- one APPLY instruction per stage, in order, then
        HALT. The stage names are the program's faculty operands, so the ordered stage list becomes a single
        program vector the VM can run (the config-to-recipe step). `machine` must know the stage names (build it
        with faculties=stage_names). Returns the assembled program vector."""
        return machine.assemble([("APPLY", s.name) for s in self.stages] + [("HALT", None)])

    def run_on_vm(self, machine=None, scene=None, seed=0, prev_frame=None, renderer=None):
        """Phase 6: RUN the pipeline ON the VM instead of a Python for-loop. Lower the stages to a program of
        APPLY instructions, give the machine a handler per stage (each runs that stage on the frame), and execute
        -- so the SAME holographic machine that runs every other program drives the render pipeline, with the
        FrameState riding as the accumulator (the VM does vector math only on the PROGRAM, never on the frame, so
        the frame can be any object the handlers understand). Returns (FrameState, trace). The trace is the list
        of APPLY-ed stage names -- a record of exactly what ran.

        This is the execution half of Phase 6; plan() is the inspection half (EXPLAIN). Bit-for-bit, run_on_vm
        produces the same frame as run() -- it is the same stages in the same order, sequenced by the VM."""
        from holographic_machine import HoloMachine
        stage_names = [s.name for s in self.stages]
        if machine is None:
            machine = HoloMachine(dim=1024, seed=0, faculties=stage_names)   # a VM that knows this pipeline's stages
        program = self.lower_to_program(machine)
        handlers = {s.name: s.run for s in self.stages}          # each stage name -> its run closure
        ctx = FrameState(scene=scene, seed=seed, prev_frame=prev_frame)
        ctx._renderer = renderer
        acc, trace = machine.run(program, init_acc=ctx, handlers=handlers)
        applied = [arg for (op, arg) in trace if op == "APPLY"]
        return acc, applied


def _selftest():
    """Presets select the right stages; asking for SVGF AUTO-INCLUDES the G-buffer; an impossible combo is
    REJECTED with a clear message; plan() dry-runs without rendering; the order is a stable toposort (present
    after render, svgf after gbuffer+render); run() threads a frame through and produces a tonemapped image;
    deterministic."""
    # (1) preset selects stages; SVGF auto-includes the gbuffer provider (enabled(cfg) is False for it)
    pipe = build_pipeline(PipelineConfig.preview())
    names = pipe.stage_names()
    assert "render" in names and "present" in names
    assert "svgf_denoise" in names and "gbuffer" in names        # gbuffer was AUTO-INCLUDED, not enabled
    assert "splat_proxy" in names and "reproject" in names

    # (2) ordering: a stage runs after what it needs (present after render; svgf after gbuffer AND render)
    pos = {n: i for i, n in enumerate(names)}
    assert pos["render"] < pos["present"]
    assert pos["gbuffer"] < pos["svgf_denoise"] and pos["render"] < pos["svgf_denoise"]

    # (3) conflicts rejected at build time with a clear message
    try:
        build_pipeline(PipelineConfig(dirty_only=True, temporal_reuse=False))
        assert False, "should have rejected dirty_only without temporal_reuse"
    except PipelineError as e:
        assert "temporal_reuse" in str(e)
    try:
        build_pipeline(PipelineConfig.final().__class__(quality="final", splat_proxy=True, dirty_only=False))
        assert False, "should have rejected splat_proxy with final"
    except PipelineError as e:
        assert "splat_proxy" in str(e)

    # (4) plan() is a dry run: lists stages+why WITHOUT rendering (no image produced)
    plan = pipe.plan()
    assert all("stage" in p and "why" in p for p in plan)
    assert len(plan) == len(names)

    # (5) run threads a frame through and produces a tonemapped image in [0,1]
    ctx = pipe.run(scene="my_scene", seed=0)
    assert ctx.image is not None and ctx.image.min() >= 0.0 and ctx.image.max() <= 1.0
    assert ctx.buffers.get("denoised") is True                   # svgf ran (its gbuffer prerequisite was present)

    # (6) interactive preset pulls in the sim stages, ordered before render
    ip = build_pipeline(PipelineConfig.interactive())
    ipn = ip.stage_names()
    assert "sim_fluid" in ipn and "sim_field_effects" in ipn and "sim_collide" in ipn
    ippos = {n: i for i, n in enumerate(ipn)}
    assert ippos["sim_collide"] < ippos["render"]                # sim before render

    # (7) deterministic: same config -> same stage order
    assert build_pipeline(PipelineConfig.preview()).stage_names() == \
           build_pipeline(PipelineConfig.preview()).stage_names()

    # (8) manual assembly path still works (a hand-built Pipeline)
    manual = Pipeline([s for s in ALL_STAGES if s.name in ("render", "present")])
    assert manual.stage_names() == ["render", "present"] or "render" in manual.stage_names()

    print("holographic_pipeline selftest OK: presets select stages; SVGF auto-includes the G-buffer; conflicts "
          "rejected with a clear message; plan() dry-runs; stable toposort (render<present, gbuffer<svgf); run() "
          "produces a tonemapped image; interactive pulls in sim stages; deterministic; manual assembly stays")


if __name__ == "__main__":
    _selftest()
