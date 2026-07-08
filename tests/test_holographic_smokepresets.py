"""Fluids/matter item 1: smoke presets -- six named looks over the wired smoke solver."""
import numpy as np
from holographic.misc.holographic_smokepresets import SMOKE_PRESETS, preset_names, simulate, plume_center_of_mass, render, _buoyant_vs_heavy


def test_six_presets_exist():
    assert set(preset_names()) == {"rising", "wispy", "billow", "heavy", "still_room", "stratified"}


def test_presets_are_distinct_looks():
    coms = {n: plume_center_of_mass(simulate(n, nx=40, ny=40, steps=45, seed=0)["density"]) for n in SMOKE_PRESETS}
    assert len({round(c, 2) for c in coms.values()}) >= 4          # genuinely different fields


def test_buoyancy_rises_gravity_sinks():
    up, down = _buoyant_vs_heavy()
    assert up > 0.5 > down                                          # the dial does what it says


def test_still_room_hangs_mid():
    com = plume_center_of_mass(simulate("still_room", nx=40, ny=40, steps=45, seed=0)["density"])
    assert 0.3 < com < 0.7


def test_all_presets_put_smoke_in_domain():
    for n in SMOKE_PRESETS:
        assert simulate(n, nx=32, ny=32, steps=12, seed=1)["density"].sum() > 0.0


def test_deterministic():
    a = simulate("rising", nx=32, ny=32, steps=15, seed=3)["density"]
    b = simulate("rising", nx=32, ny=32, steps=15, seed=3)["density"]
    assert np.array_equal(a, b)


def test_render_returns_normalized_image():
    img = render("billow", nx=32, ny=32, steps=15, seed=0)
    assert img.shape == (32, 32) and 0.0 <= img.min() and img.max() <= 1.0 + 1e-9


def test_unknown_preset_errors():
    import pytest
    with pytest.raises(ValueError):
        simulate("nope")
