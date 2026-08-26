"""Physics backlog #8 (rung 4): the overturning free surface -- the barrel wave a height field can't represent."""
import numpy as np
from holographic.mesh_and_geometry.holographic_freesurface import FreeSurface, seed_breaking_crest, free_surface_step


def test_breaking_crest_overturns():
    fs = FreeSurface(g=9.81, ground=0.0)
    seed_breaking_crest(fs, length=10.0, n=40, crest_speed=8.0, phase_speed=3.0, height=4.0)
    assert not fs.is_overturning()                            # single-valued at t=0
    fs.advance(0.05, steps=20)
    assert fs.is_overturning()                                # the crest folded over
    assert fs.is_multivalued()                                # two sheets at one x -- no height field can hold it


def test_gentle_wave_does_not_break():
    calm = FreeSurface(g=9.81, ground=0.0)
    seed_breaking_crest(calm, length=10.0, n=40, crest_speed=3.2, phase_speed=3.0, height=1.0)
    calm.advance(0.05, steps=20)
    assert not calm.is_overturning()                         # stays single-valued


def test_settle_collapses_to_height():
    fs = FreeSurface()
    seed_breaking_crest(fs, crest_speed=8.0)
    fs.advance(0.05, steps=20)
    h = fs.settle_height(np.linspace(0, 20, 41))
    assert h.shape == (40,) and h.max() > 0 and np.all(np.isfinite(h))


def test_ballistic_gravity_falls():
    fs = FreeSurface(g=9.81, ground=-100.0)                   # ground far below so nothing rests
    fs.seed([[0.0, 10.0]], [[0.0, 0.0]])
    y0 = fs.pos[0, 1]
    fs.advance(0.1, steps=5)
    assert fs.pos[0, 1] < y0                                  # gravity pulled it down


def test_free_surface_step_on_tile():
    tile = np.zeros((8, 8)); tile[:, 4:] = np.linspace(0, 6, 4)[None, :]
    out = free_surface_step(tile, dt=0.1)
    assert out.shape == (8, 8) and np.isfinite(out).all()


def test_deterministic():
    a = FreeSurface(); seed_breaking_crest(a, crest_speed=8.0); a.advance(0.05, steps=20)
    b = FreeSurface(); seed_breaking_crest(b, crest_speed=8.0); b.advance(0.05, steps=20)
    assert np.array_equal(a.pos, b.pos)
