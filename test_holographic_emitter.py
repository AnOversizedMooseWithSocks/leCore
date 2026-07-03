"""Emit particles from a surface, with Param-driven speed and weight."""
import numpy as np
from holographic_emitter import emit_from_surface, advance
from holographic_param import Param


def test_particles_land_on_surface_with_radial_normals():
    R = 1.5
    sphere = lambda P: np.linalg.norm(P, axis=1) - R
    bounds = (np.full(3, -2.2), np.full(3, 2.2))
    pos, nrm, vel = emit_from_surface(sphere, 200, bounds, speed=2.0, seed=0)
    assert len(pos) > 50
    assert np.abs(np.linalg.norm(pos, axis=1) - R).max() < 0.05           # on the surface
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    assert (np.abs((nrm * radial).sum(1) - 1.0) < 0.02).all()             # outward normals
    assert np.allclose(np.linalg.norm(vel, axis=1), 2.0, atol=1e-6)       # speed along normal


def test_weight_map_biases_emission():
    R = 1.5
    sphere = lambda P: np.linalg.norm(P, axis=1) - R
    bounds = (np.full(3, -2.2), np.full(3, 2.2))
    top = Param(field=lambda P: (P[:, 2] > 0).astype(float))              # emit only from the top
    pos, _, _ = emit_from_surface(sphere, 200, bounds, weight=top, seed=1)
    assert len(pos) > 20 and (pos[:, 2] >= -0.1).all()


def test_speed_field_and_advance():
    R = 1.0
    sphere = lambda P: np.linalg.norm(P, axis=1) - R
    bounds = (np.full(3, -1.6), np.full(3, 1.6))
    fast = Param(field=lambda P: 1.0 + 2.0 * np.clip(P[:, 2], 0, None))   # faster higher up
    pos, nrm, vel = emit_from_surface(sphere, 200, bounds, speed=fast, seed=2)
    assert np.corrcoef(pos[:, 2], np.linalg.norm(vel, axis=1))[0, 1] > 0.5
    p2, v2 = advance(pos, vel, force=np.broadcast_to([0, 0, -3.0], pos.shape), dt=0.1)
    assert p2.shape == pos.shape                                          # one integration step runs
