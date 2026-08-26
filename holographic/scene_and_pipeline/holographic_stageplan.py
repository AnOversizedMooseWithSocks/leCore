"""holographic_stageplan.py -- BAKE-VS-COMPUTE PER STAGE (fluids/matter backlog, performance item PW3).

adaptive.plan_render already decides bake-vs-analytic for the WHOLE render, from the scene and workload (bake once you
have enough primitives or enough frames to amortise it). PW3 pushes that same break-even DOWN to the individual
pipeline stage: for each stage, decide whether to BAKE its output once (precompute a grid / a static pass) or COMPUTE
it every frame -- because within one pipeline some stages are static (a baked SDF, an ambient-occlusion pass, an
irradiance cache) and some are dynamic (the actual render, a moving sim), and they deserve different answers.

The rule, reusing plan_render's own break-even (_BAKE_MIN_FRAMES):
  * a STATIC stage (output depends only on the scene, not the frame/view/time) BAKES once it is reused across enough
    frames -- the bake amortises, exactly plan_render's logic;
  * a static stage used for too FEW frames COMPUTES -- the bake wouldn't pay back;
  * a DYNAMIC stage always COMPUTES -- there is nothing stable to bake.

This is the planner that tells PW1's bake_pipeline WHICH stages to bake, and PW2's compiled pipeline WHICH stages to
leave marching -- the decision layer over the two mechanisms. Every choice carries a human-readable reason, like
plan_render's `reasons`.

KEPT NEGATIVE: the planner needs to know each stage's static-ness -- it reads a stage's `static` flag, and only
falls back to a conservative heuristic (assume DYNAMIC) when none is given, because baking a truly-dynamic stage
would be a correctness bug, not just a slow choice. So an unannotated stage is never baked; the safe default costs
some speed, never correctness. Deterministic.
"""
from holographic.misc.holographic_adaptive import _BAKE_MIN_FRAMES


# stages whose output is, by their nature, static (scene-only) vs dynamic (per-frame) -- a small, honest heuristic
# used ONLY when a stage does not declare its own `static` flag. Conservative: anything unknown is treated dynamic.
_STATIC_HINTS = {"gbuffer", "sdf_bake", "bake", "ao", "irradiance", "prt", "lightmap", "atlas"}
_DYNAMIC_HINTS = {"render", "reproject", "svgf_denoise", "present", "sim", "adaptive_samples", "splat_proxy"}


def stage_is_static(stage):
    """Best-effort static-ness of a stage: an explicit `static` attribute wins; otherwise a name heuristic; otherwise
    the CONSERVATIVE default (dynamic -> never baked), because baking a dynamic stage would be a correctness bug."""
    flag = getattr(stage, "static", None)
    if flag is not None:
        return bool(flag)
    name = (getattr(stage, "name", None) or str(stage)).lower()
    if any(h in name for h in _STATIC_HINTS):
        return True
    return False                                                 # unknown -> dynamic -> compute (safe)


def plan_stage(name, static, frames, bake_min_frames=_BAKE_MIN_FRAMES):
    """Decide bake vs compute for ONE stage. Returns (choice, reason)."""
    if not static:
        return "compute", "dynamic output (changes per frame) -- nothing stable to bake"
    if frames >= bake_min_frames:
        return "bake", ("static output reused across %d frames (>= %d break-even) -- bake amortises"
                        % (frames, bake_min_frames))
    return "compute", ("static but only %d frame(s) (< %d break-even) -- a bake wouldn't pay back" % (frames, bake_min_frames))


def plan_stages(stages, frames, bake_min_frames=_BAKE_MIN_FRAMES):
    """Per-stage bake-vs-compute plan for a whole pipeline over `frames` frames. Returns a list of dicts
    {stage, choice, static, reason} -- the pipeline twin of plan_render's per-object `methods`+`reasons`."""
    plan = []
    for s in stages:
        name = getattr(s, "name", None) or str(s)
        static = stage_is_static(s)
        choice, reason = plan_stage(name, static, frames, bake_min_frames)
        plan.append({"stage": name, "choice": choice, "static": static, "reason": reason})
    return plan


def stages_to_bake(stages, frames, bake_min_frames=_BAKE_MIN_FRAMES):
    """Just the names the planner says to BAKE -- feed straight to PW1's bake_pipeline."""
    return [p["stage"] for p in plan_stages(stages, frames, bake_min_frames) if p["choice"] == "bake"]


def _selftest():
    """The per-stage planner bakes a static stage once it is reused enough, computes it when reused too few times,
    always computes a dynamic stage, and reads an explicit `static` flag over the heuristic."""
    class S:                                                     # a minimal stand-in for a pipeline Stage
        def __init__(self, name, static=None):
            self.name = name
            if static is not None:
                self.static = static

    # a mix: a baked SDF (static by name), a render (dynamic by name), an explicitly-static AO, an unknown stage
    stages = [S("sdf_bake"), S("render"), S("ambient", static=True), S("mystery")]

    # over MANY frames: static stages bake, dynamic/unknown compute
    plan = {p["stage"]: p["choice"] for p in plan_stages(stages, frames=30)}
    assert plan["sdf_bake"] == "bake"                            # static by name, many frames -> bake
    assert plan["ambient"] == "bake"                            # explicit static flag -> bake
    assert plan["render"] == "compute"                          # dynamic -> compute
    assert plan["mystery"] == "compute"                         # unknown -> conservative compute (never bake blindly)

    # over ONE frame: even a static stage computes (a bake wouldn't amortise)
    plan1 = {p["stage"]: p["choice"] for p in plan_stages(stages, frames=1)}
    assert plan1["sdf_bake"] == "compute" and plan1["ambient"] == "compute"

    # explicit static=False overrides the name heuristic (a "bake"-named stage that is actually dynamic)
    forced = S("sdf_bake", static=False)
    assert plan_stages([forced], frames=30)[0]["choice"] == "compute"

    # stages_to_bake feeds PW1
    assert set(stages_to_bake(stages, frames=30)) == {"sdf_bake", "ambient"}

    # reasons are human-readable and mention the break-even
    r = plan_stages([S("sdf_bake")], frames=30)[0]["reason"]
    assert "amortise" in r and "break-even" in r

    print("holographic_stageplan selftest OK: per-stage bake-vs-compute extends plan_render's break-even to the "
          "pipeline -- static stages (sdf_bake, explicit AO) BAKE over 30 frames, dynamic (render) and UNKNOWN "
          "(mystery) COMPUTE (unknown is conservatively never baked -- correctness over speed); at 1 frame even "
          "static stages compute (no amortisation); an explicit static=False overrides the name heuristic; "
          "stages_to_bake feeds PW1's bake_pipeline, each choice carrying a reason.")


if __name__ == "__main__":
    _selftest()
