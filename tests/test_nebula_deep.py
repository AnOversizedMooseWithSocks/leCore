"""Deep nebula property checks at production resolution -- marked `slow` because nebula_volume is O(res^3 * octaves)
and res=32 alone is ~22 s (res=48 ~72 s), which is over the 15 s per-test budget. The module's own _selftest keeps
the FAST contract (res=16, one volume) so every push is covered; this file adds the checks that only mean something
at higher resolution and the ridged-vs-smooth comparison the fast test drops, so no coverage is lost -- it just moves
behind --run-slow / LECORE_RUN_SLOW=1.
"""
import numpy as np
import pytest

from holographic.scene_and_pipeline.holographic_nebula import nebula_volume, nebula_column


@pytest.mark.slow  # res=32/48 volumes at ~22/72 s each; the fast contract lives in the module _selftest at res=16.
def test_nebula_structure_at_production_res():
    # at res=32 the void/filament separation is the real, shippable look; assert it holds where it matters.
    v = nebula_volume(res=32, seed=0)
    assert v.shape == (32, 32, 32) and 0.0 <= v.min() and v.max() <= 1.0
    assert np.array_equal(v, nebula_volume(res=32, seed=0)), "deterministic at production res"
    assert np.mean(v < 0.02) > 0.2, "dark voids present at res=32"
    assert v.max() > 0.5 and np.var(v) > 1e-3, "bright filaments present at res=32"

    # a centred star carves a measurable cavity at production res
    core = (slice(12, 20), slice(12, 20), slice(12, 20))
    v_star = nebula_volume(res=32, seed=0, star_positions=[(0.5, 0.5, 0.5)], cavity_radius=0.25)
    assert np.mean(v_star[core]) < np.mean(v[core]), "star carves a cavity at res=32"

    # column projection at production res is a sane image
    col = nebula_column(v)
    assert col.shape == (32, 32) and np.all(col >= 0.0)


@pytest.mark.slow  # two res=16 volumes; grouped here with the other deep checks rather than taxing the fast selftest.
def test_nebula_ridged_vs_smooth_differ():
    # the ridged (filament) transform must actually change the field -- a real, separable knob, not a no-op.
    ridged = nebula_volume(res=16, seed=2, ridged=True)
    smooth = nebula_volume(res=16, seed=2, ridged=False)
    assert not np.array_equal(ridged, smooth), "ridged transform does nothing -- filaments are fake"
