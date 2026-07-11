"""Make CI run every module's own _selftest -- the hole that let holographic_skills (and then holographic_lights)
sit RED and silent, because CI ran pytest and pytest never executed the module selftests.

Two tests, two independent judgements:
  * test_every_module_selftest_is_green -- the walk itself. Fails if ANY module's _selftest raises or times out.
  * test_no_selftest_budget_only_shrinks -- a budget of modules that have a __main__ but no real _selftest. It MAY
    SHRINK AND MUST NEVER GROW (same discipline as the duplicate-shape budget): writing a new module without a
    selftest, or deleting an existing one, fails here with the offending name. Backfilling a selftest onto a
    budgeted module is the ONLY correct way to change this file -- remove its line.

WHY this is not just `-m` in a shell loop: as a pytest test it inherits CI's collection, parallelism, and the
BLAS-thread pinning the walker applies per child. The walk is ~75s at 12 jobs on this box (measured); if your CI
runner is smaller, raise the timeout or the job count -- do not sample a subset, or the silence comes back.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
from run_selftests import walk, discover     # the tool owns the logic; the test owns the budget


_NO_SELFTEST_BUDGET = {
    'holographic.agents_and_reasoning.holographic_answer',
    'holographic.agents_and_reasoning.holographic_deliberate',
    'holographic.agents_and_reasoning.holographic_fountain',
    'holographic.agents_and_reasoning.holographic_honesty',
    'holographic.agents_and_reasoning.holographic_intent',
    'holographic.agents_and_reasoning.holographic_kan',
    'holographic.agents_and_reasoning.holographic_lexicon',
    'holographic.agents_and_reasoning.holographic_machine',
    'holographic.agents_and_reasoning.holographic_meaning_predict',
    'holographic.agents_and_reasoning.holographic_mind',
    'holographic.agents_and_reasoning.holographic_moe',
    'holographic.agents_and_reasoning.holographic_navigator',
    'holographic.agents_and_reasoning.holographic_predictive',
    'holographic.agents_and_reasoning.holographic_reasoning',
    'holographic.agents_and_reasoning.holographic_recurrent',
    'holographic.agents_and_reasoning.holographic_respond',
    'holographic.agents_and_reasoning.holographic_symbolic',
    'holographic.caching_and_storage.holographic_history',
    'holographic.caching_and_storage.holographic_sync',
    'holographic.io_and_interop.holographic_image',
    'holographic.io_and_interop.holographic_toolclient',
    'holographic.io_and_interop.holographic_uri',
    'holographic.io_and_interop.holographic_video',
    'holographic.mesh_and_geometry.holographic_manifold',
    'holographic.mesh_and_geometry.holographic_mobius',
    'holographic.misc.holographic_ablate',
    'holographic.misc.holographic_archive',
    'holographic.misc.holographic_codec',
    'holographic.misc.holographic_compose',
    'holographic.misc.holographic_compress',
    'holographic.misc.holographic_diffusion',
    'holographic.misc.holographic_extras',
    'holographic.misc.holographic_flow',
    'holographic.misc.holographic_fractal',
    'holographic.misc.holographic_generate',
    'holographic.misc.holographic_generation',
    'holographic.misc.holographic_market',
    'holographic.misc.holographic_measure',
    'holographic.misc.holographic_pack',
    'holographic.misc.holographic_partition',
    'holographic.misc.holographic_ratedistortion',
    'holographic.misc.holographic_recipe',
    'holographic.misc.holographic_relations',
    'holographic.misc.holographic_resolution',
    'holographic.misc.holographic_sbc',
    'holographic.misc.holographic_segment',
    'holographic.misc.holographic_sequence',
    'holographic.misc.holographic_signal_structure',
    'holographic.misc.holographic_structure',
    'holographic.misc.holographic_typed',
    'holographic.rendering.holographic_peel',
    'holographic.rendering.holographic_photos',
    'holographic.rendering.holographic_splat_archive',
    'holographic.sampling_and_signal.holographic_fhrr',
    'holographic.sampling_and_signal.holographic_tensor',
    'holographic.scene_and_pipeline.holographic_orchestrator',
    'holographic.scene_and_pipeline.holographic_organizer',
    'holographic.scene_and_pipeline.holographic_scene',
    'holographic.simulation_and_physics.holographic_assembly',
    'holographic.simulation_and_physics.holographic_dynamics',
    'holographic.simulation_and_physics.holographic_emergence',
    'holographic.simulation_and_physics.holographic_graph_memory',
    'holographic.simulation_and_physics.holographic_physics',
    'holographic.simulation_and_physics.holographic_slime',
}


def test_every_module_selftest_is_green():
    """Run all runnable selftests; fail loudly with the offending module(s) if any is red or hangs."""
    results, _ = walk(jobs=8, timeout=180)
    bad = [(mod, verdict, tail) for mod, verdict, _dt, tail in results if verdict != "OK"]
    assert not bad, "module selftest(s) not green:\n" + "\n".join(
        "  %-10s %s\n             %s" % (v, m, t) for m, v, t in bad)


def test_no_selftest_budget_only_shrinks():
    """The set of modules with a __main__ but no real _selftest may shrink, never grow. A new name here means
    someone shipped a module without a selftest; a missing name means one was backfilled -- delete its line."""
    _, missing = discover()
    missing = set(missing)
    grew = missing - _NO_SELFTEST_BUDGET
    assert not grew, ("module(s) shipped WITHOUT a selftest (add a `def _selftest` + `__main__`, or, if truly "
                      "not applicable, add to _NO_SELFTEST_BUDGET WITH A REASON):\n  " + "\n  ".join(sorted(grew)))
    # informational: if the budget shrank, that's good -- but the literal is now stale and should be trimmed.
    shrank = _NO_SELFTEST_BUDGET - missing
    if shrank:
        print("selftest backfilled onto %d module(s); remove from _NO_SELFTEST_BUDGET: %s"
              % (len(shrank), sorted(shrank)))
