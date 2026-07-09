"""Cymatics (A4): plate eigenmodes, sand settles on nodes, square vs circle differ, deterministic."""
import numpy as np
from holographic.simulation_and_physics.holographic_cymatics import ChladniPlate


def test_sand_settles_on_nodes():
    p = ChladniPlate("square", grid=36, n_modes=30, n_grains=5000, seed=0)
    assert p.modes.shape[1] >= 20 and (p.eigvals[1:] > 0).all()
    p.drive_mode(5); p.settle(steps=80, dt=0.1, strength=8.0)
    sand_u, plate_u = p.nodal_fraction_on_sand()
    assert sand_u < 0.5 * plate_u                                  # sand on the low-|u| nodal set


def test_drive_by_sound_excites_modes():
    p = ChladniPlate("square", grid=32, n_modes=24, n_grains=2000, seed=0)
    u = p.drive([p.mode_hz[3]], [1.0])
    assert np.abs(u).max() > 0.5 and np.isfinite(u).all()


def test_square_vs_circle_and_determinism():
    c = ChladniPlate("circle", grid=36, n_modes=30, n_grains=5000, seed=0)
    p = ChladniPlate("square", grid=36, n_modes=30, n_grains=5000, seed=0)
    assert c.mask.sum() < p.mask.sum()                             # disk smaller than square
    c.drive_mode(5); c.settle(steps=60)
    su, pu = c.nodal_fraction_on_sand()
    assert su < 0.6 * pu
    a = ChladniPlate("square", grid=28, n_modes=20, n_grains=2000, seed=1); a.drive_mode(4); a.settle(30)
    b = ChladniPlate("square", grid=28, n_modes=20, n_grains=2000, seed=1); b.drive_mode(4); b.settle(30)
    assert np.array_equal(a.sand_density(), b.sand_density())


def test_water_stands_at_antinodes():
    """A5 water (Faraday): the standing surface forms at the ANTINODES (correlates with |u|), opposite of sand."""
    import numpy as np
    w = ChladniPlate("square", grid=40, medium="water", n_modes=30, seed=0)
    w.drive_mode(8); w.settle(steps=40)
    au = np.abs(w.u)[w.mask]; surf = w.surface[w.mask]
    assert np.corrcoef(au, surf)[0, 1] > 0.8                       # water crests where the plate moves most


def test_water_cell_size_tracks_frequency():
    """A5 water: a higher drive frequency (higher mode) makes a FINER pattern (more antinode cells)."""
    import numpy as np
    def peaks(m):
        p = ChladniPlate("square", grid=48, medium="water", n_modes=40, seed=0); p.drive_mode(m); p.settle(30)
        s = p.surface; c = s[1:-1, 1:-1]
        ismax = (c >= s[:-2, 1:-1]) & (c >= s[2:, 1:-1]) & (c >= s[1:-1, :-2]) & (c >= s[1:-1, 2:]) & (c > 0.35 * s.max())
        return int(ismax.sum())
    assert peaks(22) > peaks(4)                                    # finer cells at higher drive frequency


def test_cornstarch_shear_thickens_under_fast_drive():
    """A5 cornstarch: holds peaks under FAST drive, relaxes under slow (the shear-thickening signature)."""
    fast = ChladniPlate("square", grid=40, medium="cornstarch", n_modes=30, base_hz=1400.0, seed=0)
    fast.drive_mode(10); fast.settle(40)
    slow = ChladniPlate("square", grid=40, medium="cornstarch", n_modes=30, base_hz=40.0, seed=0)
    slow.drive_mode(10); slow.settle(40)
    assert fast.peaks.max() > slow.peaks.max() * 2                 # stands up fast, slumps slow
