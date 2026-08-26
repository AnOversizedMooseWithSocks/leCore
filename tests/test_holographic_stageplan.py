"""Performance PW3: per-stage bake-vs-compute planner (extends adaptive.plan_render to the pipeline)."""
from holographic.scene_and_pipeline.holographic_stageplan import plan_stage, plan_stages, stages_to_bake, stage_is_static
from holographic.misc.holographic_adaptive import _BAKE_MIN_FRAMES


class S:
    def __init__(self, name, static=None):
        self.name = name
        if static is not None:
            self.static = static


def test_static_stage_bakes_over_many_frames():
    assert plan_stage("sdf_bake", static=True, frames=30)[0] == "bake"


def test_static_stage_computes_when_few_frames():
    assert plan_stage("sdf_bake", static=True, frames=1)[0] == "compute"


def test_dynamic_stage_always_computes():
    assert plan_stage("render", static=False, frames=1000)[0] == "compute"


def test_static_ness_from_name_heuristic():
    assert stage_is_static(S("sdf_bake")) is True
    assert stage_is_static(S("render")) is False
    assert stage_is_static(S("mystery")) is False              # unknown -> conservative dynamic


def test_explicit_flag_overrides_heuristic():
    assert stage_is_static(S("sdf_bake", static=False)) is False   # a bake-named but declared-dynamic stage


def test_plan_stages_mixed():
    stages = [S("sdf_bake"), S("render"), S("ambient", static=True), S("mystery")]
    plan = {p["stage"]: p["choice"] for p in plan_stages(stages, frames=30)}
    assert plan == {"sdf_bake": "bake", "render": "compute", "ambient": "bake", "mystery": "compute"}


def test_stages_to_bake_feeds_pw1():
    stages = [S("sdf_bake"), S("render"), S("ambient", static=True)]
    assert set(stages_to_bake(stages, frames=30)) == {"sdf_bake", "ambient"}


def test_break_even_uses_plan_render_constant():
    # exactly at the break-even a static stage bakes; just below it computes
    assert plan_stage("bake", static=True, frames=_BAKE_MIN_FRAMES)[0] == "bake"
    assert plan_stage("bake", static=True, frames=_BAKE_MIN_FRAMES - 1)[0] == "compute"


def test_reason_is_human_readable():
    r = plan_stages([S("sdf_bake")], frames=30)[0]["reason"]
    assert "amortise" in r and "break-even" in r
