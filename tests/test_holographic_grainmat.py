"""Wood grain (M1): volumetric rings, valid sockets, deterministic. (Module existed unwired/untested; this pins it.)"""
import numpy as np
from holographic.materials_and_texture.holographic_grainmat import wood_grain, wood_albedo, substrate_layers


def test_grain_scalar_in_range_and_deterministic():
    g = wood_grain(axis=(0, 1, 0), ring_scale=6.0, seed=1)
    P = np.random.default_rng(0).uniform(-2, 2, (1500, 3))
    v = np.asarray(g(P), float)
    assert v.shape == (1500,) and v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9
    assert np.array_equal(np.asarray(g(P)), np.asarray(g(P)))


def test_grain_is_volumetric_rings_follow_axis():
    """Moving ALONG the grain axis barely changes the ring value (rings are concentric about the axis); moving
    ACROSS it (changing distance from the axis) sweeps through rings. That is the 'cut the board, rings continue'
    property."""
    g = wood_grain(axis=(0, 1, 0), ring_scale=8.0, fibre=0.0, warp=0.0, seed=0)   # pure rings, no fibre/warp
    along = np.array([[0.5, y, 0.0] for y in np.linspace(-1, 1, 40)])             # vary height along axis
    across = np.array([[r, 0.0, 0.0] for r in np.linspace(0.0, 1.5, 40)])          # vary radius from axis
    assert np.ptp(g(along)) < np.ptp(g(across))                                    # rings vary across, not along


def test_wood_albedo_and_substrate_are_valid_rgb():
    alb = wood_albedo(light=(0.7, 0.5, 0.3), dark=(0.4, 0.25, 0.12))
    sub = substrate_layers(axis=(0, 1, 0))
    P = np.random.default_rng(2).uniform(-1, 1, (400, 3))
    for sock in (alb, sub):
        c = np.asarray(sock(P), float)
        assert c.shape == (400, 3) and c.min() >= -1e-9 and c.max() <= 1.0 + 1e-9
