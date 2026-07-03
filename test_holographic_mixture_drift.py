"""Fluids/matter item 3: drift -- settling/separation (heavy sinks, light rises), the salt-fingering driver."""
import numpy as np
from holographic_mixture import Mixture, matter_step, _blob


def _com_y(field, shape):
    ys = np.mgrid[0:shape[0], 0:shape[1]][0]
    f = np.clip(field, 0, None)
    return float((ys * f).sum() / (f.sum() + 1e-12))


def _heavy_blob(drift, density=3.0, steps=25):
    shape = (48, 48)
    mix = Mixture(shape, solvent_density=1.0, buoyancy=0.0)
    mix.add("c", _blob(shape, 24, 24, 4.0), density=density, diffusivity=0.001)
    vx = np.zeros(shape); vy = np.zeros(shape)
    y0 = _com_y(mix.channels["c"], shape)
    for _ in range(steps):
        vx, vy = matter_step(mix, vx, vy, dt=0.1, drift_strength=drift)
    return y0, _com_y(mix.channels["c"], shape)


def test_drift_settles_heavy_channel():
    y0, y1 = _heavy_blob(drift=0.5)
    assert y1 < y0 - 0.5                                       # heavy channel sinks (row decreases = down)


def test_no_drift_no_settling():
    y0, y1 = _heavy_blob(drift=0.0)
    assert abs(y1 - y0) < 1e-6                                 # the baseline: without drift it stays put


def test_light_channel_rises():
    shape = (48, 48)
    mix = Mixture(shape, solvent_density=1.0, buoyancy=0.0)
    mix.add("light", _blob(shape, 24, 24, 4.0), density=0.2, diffusivity=0.001)
    vx = np.zeros(shape); vy = np.zeros(shape)
    y0 = _com_y(mix.channels["light"], shape)
    for _ in range(25):
        vx, vy = matter_step(mix, vx, vy, dt=0.1, drift_strength=0.5)
    assert _com_y(mix.channels["light"], shape) > y0 + 0.3     # lighter-than-solvent rises


def test_stronger_drift_settles_more():
    _, weak = _heavy_blob(drift=0.2)
    _, strong = _heavy_blob(drift=0.8)
    assert strong < weak                                       # more drift -> more settling


def test_two_channels_separate():
    # a heavy and a light channel co-located separate vertically (heavy down, light up) -- the immiscible precursor
    shape = (48, 48)
    mix = Mixture(shape, solvent_density=1.0, buoyancy=0.0)
    mix.add("heavy", _blob(shape, 24, 24, 4.0), density=3.0, diffusivity=0.002)
    mix.add("light", _blob(shape, 24, 24, 4.0), density=0.2, diffusivity=0.002)
    vx = np.zeros(shape); vy = np.zeros(shape)
    for _ in range(25):
        vx, vy = matter_step(mix, vx, vy, dt=0.1, drift_strength=0.5)
    assert _com_y(mix.channels["heavy"], shape) < _com_y(mix.channels["light"], shape)   # heavy below light


def test_density_now_drives_buoyancy():
    # the alpha fix: a dense region creates flow (vertical KE > 0) even with zero temperature
    shape = (40, 40)
    salt = np.zeros(shape); salt[26:, :] = 0.7                 # a heavy band up high
    mix = Mixture(shape, solvent_density=1.0, buoyancy=1.5)
    mix.add("salt", salt, density=2.5, diffusivity=0.002)
    vx = np.zeros(shape); vy = np.zeros(shape)
    for _ in range(30):
        vx, vy = matter_step(mix, vx, vy, dt=0.1, drift_strength=0.4)
    assert np.mean(vy ** 2) > 0.0                              # density buoyancy convects (was 0 before the fix)


def test_fingering_mechanism_present_kept_negative():
    """KEPT NEGATIVE: the salt-fingering INGREDIENTS (differential diffusion + drift + density buoyancy) all run,
    but resolving DISTINCT fingers cleanly is dominated by bulk overturning at this grid/time -- not claimed as a
    win. This test only asserts the mechanism runs and stays finite, documenting the honest boundary."""
    shape = (48, 48)
    mix = Mixture(shape, solvent_density=1.0, buoyancy=1.2)
    mix.add("fast", _blob(shape, 30, 24, 5.0), density=1.5, diffusivity=0.2)    # fast diffuser
    mix.add("slow", _blob(shape, 30, 24, 5.0), density=2.0, diffusivity=0.002)  # slow, heavy -> the driver
    vx = np.zeros(shape); vy = np.zeros(shape)
    for _ in range(30):
        vx, vy = matter_step(mix, vx, vy, dt=0.1, drift_strength=0.4)
    assert np.isfinite(mix.channels["slow"]).all() and np.isfinite(vy).all()
