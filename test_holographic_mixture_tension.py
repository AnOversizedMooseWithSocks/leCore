"""Fluids/matter item 4: double_well tension -- oil & water (the miscible<->immiscible dial)."""
import numpy as np
from holographic_mixture import Mixture, matter_step, _double_well, _blob


def _committed(phi):
    """Fraction of cells committed to a phase (near 0 or near 1) vs intermediate -- a sharpness measure."""
    return float(((phi < 0.15) | (phi > 0.85)).mean())


def _graded(shape):
    xs = np.mgrid[0:shape[0], 0:shape[1]][1]
    return np.clip((xs - 12) / 24.0, 0, 1)                    # a smooth 0->1 ramp (all intermediate)


def test_double_well_pushes_toward_phases():
    # W'(phi) drives phi<0.5 down toward 0 and phi>0.5 up toward 1
    assert _double_well(0.3) > 0                              # positive -> subtracted -> phi decreases
    assert _double_well(0.7) < 0                              # negative -> phi increases
    assert abs(_double_well(0.0)) < 1e-9 and abs(_double_well(1.0)) < 1e-9   # the two wells are fixed points


def test_tension_sharpens_interface():
    shape = (48, 48)
    mix = Mixture(shape, buoyancy=0.0, tension=2.0)
    mix.add("oil", _graded(shape), density=0.9, diffusivity=0.0)
    vx = np.zeros(shape); vy = np.zeros(shape)
    b0 = _committed(mix.channels["oil"])
    for _ in range(40):
        vx, vy = matter_step(mix, vx, vy, dt=0.1)
    assert _committed(mix.channels["oil"]) > b0 + 0.1         # the blended interface sharpened into two phases


def test_no_tension_stays_blended():
    shape = (48, 48)
    mix = Mixture(shape, buoyancy=0.0, tension=0.0)
    mix.add("dye", _graded(shape), density=1.0, diffusivity=0.0)
    vx = np.zeros(shape); vy = np.zeros(shape)
    b0 = _committed(mix.channels["dye"])
    for _ in range(40):
        vx, vy = matter_step(mix, vx, vy, dt=0.1)
    assert abs(_committed(mix.channels["dye"]) - b0) < 0.05   # miscible: stays graded


def test_more_tension_sharpens_more():
    shape = (40, 40)
    def sharp(tension):
        mix = Mixture(shape, buoyancy=0.0, tension=tension)
        mix.add("p", _graded(shape), density=1.0, diffusivity=0.0)
        vx = np.zeros(shape); vy = np.zeros(shape)
        for _ in range(30):
            vx, vy = matter_step(mix, vx, vy, dt=0.1)
        return _committed(mix.channels["p"])
    assert sharp(3.0) >= sharp(0.5)                           # stronger tension -> sharper


def test_oil_and_water_separate():
    # oil (light) in water with tension + drift: oil rises AND forms a sharp interface (immiscible separation)
    shape = (48, 48)
    mix = Mixture(shape, solvent_density=1.0, buoyancy=0.0, tension=2.0)
    mix.add("oil", _blob(shape, 24, 24, 8.0) * 0.6 + 0.2, density=0.5, diffusivity=0.0)
    vx = np.zeros(shape); vy = np.zeros(shape)
    for _ in range(40):
        vx, vy = matter_step(mix, vx, vy, dt=0.1, drift_strength=0.3)
    # after separation most cells are committed (phase-separated), not a uniform blend
    assert _committed(mix.channels["oil"]) > 0.7
    for phi in mix.channels.values():
        assert np.isfinite(phi).all()


def test_interface_stays_finite_width_kept_negative():
    """KEPT NEGATIVE: the immiscible interface is a DIFFUSE-INTERFACE model -- it is a few cells wide, never a
    perfectly sharp step. This asserts the interface exists (some intermediate cells remain), documenting the trade."""
    shape = (48, 48)
    mix = Mixture(shape, buoyancy=0.0, tension=2.0)
    mix.add("oil", _graded(shape), density=0.9, diffusivity=0.0)
    vx = np.zeros(shape); vy = np.zeros(shape)
    for _ in range(40):
        vx, vy = matter_step(mix, vx, vy, dt=0.1)
    intermediate = ((mix.channels["oil"] >= 0.15) & (mix.channels["oil"] <= 0.85)).sum()
    assert intermediate > 0                                   # a finite-width interface band remains (not a step)


def test_deterministic():
    shape = (32, 32)
    def run():
        mix = Mixture(shape, buoyancy=0.0, tension=1.5)
        mix.add("p", _graded(shape), density=1.0)
        vx = np.zeros(shape); vy = np.zeros(shape)
        for _ in range(10):
            vx, vy = matter_step(mix, vx, vy, dt=0.1)
        return mix.channels["p"]
    assert np.array_equal(run(), run())
