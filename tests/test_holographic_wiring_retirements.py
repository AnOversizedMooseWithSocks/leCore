"""The tail of the superpowers wiring backlog: five clients that were named wrong, retired with evidence.

THE PATTERN, and it is the single most useful thing the audit produced: **the first wiring audit repeatedly named
the module where the SYMPTOM was visible rather than the module that OWNS the mechanism.**

    named            actual owner        why the name was wrong
    lightcache       session             the drifting query is a CAMERA, held one layer up
    postfx           fieldhome           the field is streamed, not stored
    heat             laplacian           the closed form lives one layer down
    farm             coordinator         farm is a TRANSPORT backend; it has no reduce
    physics          softbody            `physics` is VSA kinematics; it has no islands
    emitter          softbody            `emitter.advance` takes no collider

No scan could have found this. Reading each module did.

This suite pins the two claims those retirements rest on: that the exactness of `run_exact` is a property of the
REDUCE and not the transport (so `farm` has nothing to wire), and that the retired modules genuinely lack the
structure their unifier needs.
"""

import os
import sys

import numpy as np
import pytest

from holographic.scene_and_pipeline.holographic_coordinator import Coordinator, InProcessBackend, LocalPool

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "tools"))
from unifiers import (DEFERRED, DESIGN_CLIENTS, NOT_APPLICABLE, PENDING,  # noqa: E402
                      REGISTRY, status, unaccounted)


def _wide(n=400, d=4, seed=0):
    """Contributions spanning 16 orders of magnitude. Float non-associativity bites on DYNAMIC RANGE; without this
    spread the float baseline agrees and the test proves nothing."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) * (10.0 ** rng.integers(-8, 8, size=(n, 1)))


def _split(a, k):
    return [a[i] for i in np.array_split(np.arange(len(a)), k)]


# ---------------------------------------------------------------------------------------------------------
# the claim `farm`'s retirement rests on: exactness lives in the REDUCE, not the transport
# ---------------------------------------------------------------------------------------------------------

def test_run_exact_is_bit_identical_across_backends_and_bucket_counts():
    # A REAL process pool, not just the in-process backend -- if exactness depended on the transport, this is where
    # it would break. (The worker must be importable, not a lambda: a lambda cannot cross a process boundary.)
    from tests.helpers_exact_worker import contribs

    d = _wide()
    results = {}
    for name, backend in (("inprocess", InProcessBackend()), ("localpool", LocalPool(2))):
        c = Coordinator(backend)
        try:
            for k in (4, 7, 13):
                total, _info = c.run_exact(_split(d, k), contribs)
                results[(name, k)] = total
        finally:
            c.close()

    base = results[("inprocess", 4)]
    for key, val in results.items():
        assert np.array_equal(val, base), key


def test_the_float_path_still_drifts_which_is_why_run_exact_exists():
    from tests.helpers_exact_worker import bucket_sum

    d = _wide()
    c = Coordinator()
    try:
        f4 = c.run(_split(d, 4), bucket_sum)
        f7 = c.run(_split(d, 7), bucket_sum)
    finally:
        c.close()
    assert not np.array_equal(f4, f7)
    assert np.abs(f4 - f7).max() > 1e-9


def test_farm_performs_no_reduction_of_its_own():
    # The structural claim: NetworkFarm submits work and returns results. It has no opinion about arithmetic.
    import inspect

    from holographic.misc import holographic_farm as farm

    assert hasattr(farm, "NetworkFarm")
    src = inspect.getsource(farm)
    # `reduce` appears only in prose, never as a call or a definition
    assert "def reduce" not in src and "reduce(" not in src


# ---------------------------------------------------------------------------------------------------------
# the structural claims the other four retirements rest on
# ---------------------------------------------------------------------------------------------------------

def test_emitter_advance_takes_no_collider_so_there_is_nothing_to_sweep():
    import inspect

    from holographic.simulation_and_physics.holographic_emitter import advance

    params = set(inspect.signature(advance).parameters)
    assert "collider" not in params and "sdf" not in params
    assert params == {"pos", "vel", "force", "dt", "damping", "wrap_to"}


def test_physics_has_no_islands_to_put_to_sleep():
    from holographic.simulation_and_physics import holographic_physics as physics

    assert hasattr(physics, "Kinematics")
    for absent in ("constraints", "islands", "step_islands", "SleepTracker"):
        assert not hasattr(physics, absent), absent


def test_render_has_no_linear_shift_invariant_pass_to_compose():
    # Its 31 matches for "filter" are PNG scanline filters in the image encoder, not convolutions.
    import inspect

    from holographic.rendering import holographic_render as render

    src = inspect.getsource(render)
    assert "np.fft" not in src and "np.convolve" not in src
    assert "scanline" in src.lower()                       # ... and this is what "filter" means in there


def test_texturerender_has_no_transfer_to_compose():
    import inspect

    from holographic.rendering import holographic_texturerender as tr

    src = inspect.getsource(tr)
    assert "np.fft" not in src and "np.convolve" not in src
    assert "BakedGrid" not in src


# ---------------------------------------------------------------------------------------------------------
# the registry's own integrity
# ---------------------------------------------------------------------------------------------------------

def test_every_declared_client_is_wired():
    rows = status(_REPO)
    unwired = [(u, c) for u, c, w in rows if not w]
    assert not unwired, unwired
    assert len(PENDING) == 0
    assert isinstance(PENDING, set)                        # `PENDING = {}` would be a DICT, and the set math dies


def test_narrowing_the_registry_cannot_make_the_lint_green():
    # DESIGN_CLIENTS records every client the FIRST audit named. Each must end up wired, or retired with a reason.
    assert unaccounted(_REPO) == []

    # the five mis-named clients are all retired, and each carries its evidence
    retired = {}
    for (u, c), why in list(NOT_APPLICABLE.items()) + list(DEFERRED.items()):
        for name in c.split("/"):
            retired[(u, name)] = why

    for key in (("shader algebra (bake once, compose passes)", "holographic_render"),
                ("shader algebra (bake once, compose passes)", "holographic_texturerender"),
                ("shader algebra (bake once, compose passes)", "holographic_heat"),
                ("distribute.reduce_sum_exact_partitioned", "holographic_farm"),
                ("collide.advance_ccd / time_of_impact", "holographic_emitter"),
                ("island.SleepTracker (solve only what moves)", "holographic_physics")):
        assert key in retired, key
        assert len(retired[key]) > 100, key                # a verdict must carry its evidence, not just a verdict


def test_the_2026_cohort_is_recorded_in_design_clients():
    for unifier in ("superposed width (bind_fixed / recover_all / pack)",
                    "shader algebra (bake once, compose passes)",
                    "distribute.reduce_sum_exact_partitioned",
                    "collide.advance_ccd / time_of_impact",
                    "island.SleepTracker (solve only what moves)",
                    "tucker.LowRankField (compressed-domain compute)",
                    "cachehome.MarginCache (fat margin for a drifting query)"):
        assert unifier in DESIGN_CLIENTS, unifier
        # the design list must be a SUPERSET of the surviving clients -- you may retire, never invent
        assert set(REGISTRY[unifier]["clients"]) <= set(DESIGN_CLIENTS[unifier]), unifier


def test_impossible_and_merely_deferred_stay_distinct():
    assert not (set(NOT_APPLICABLE) & set(DEFERRED))

    # postfx once appeared in DEFERRED under TWO unifiers, for two different measured reasons. One has been PAID:
    # G8 gave `Pipeline` a half-spectrum mode, delegation became bit-identical at the same speed, and postfx is now
    # a wired client of the shader algebra. The other stands -- streaming a frame cannot amortize an SVD (53.7x).
    assert ("shader algebra (bake once, compose passes)", "holographic_postfx") not in DEFERRED
    assert "holographic_postfx" in REGISTRY["shader algebra (bake once, compose passes)"]["clients"]

    lr = ("tucker.LowRankField (compressed-domain compute)", "holographic_postfx")
    assert lr in DEFERRED and "53.7x" in DEFERRED[lr]

    # A DEFERRED entry is a debt with a stated interest rate. The lint's own contract is 60 characters of evidence
    # (`test_impossible_and_merely_deferred_are_not_confused`); I first wrote 100 here and it red-flagged a terse
    # but perfectly good 95-char entry. Match the existing contract rather than pad a reason to satisfy a test.
    for (_u, _c), why in DEFERRED.items():
        assert len(why) > 60
