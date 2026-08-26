"""Inverse-rendering IR1: auto-bump -- image -> height -> normal map -> material channel, with an abstain gate."""
import numpy as np
from holographic.mesh_and_geometry.holographic_autobump import image_to_height, normal_from_height, bump_confidence, auto_bump, flat_normal_map, pack_normal_rgb, quantize_normals, add_height_channel

N = 48
def _ramp():
    r = np.tile(np.linspace(0.2, 0.8, N), (N, 1)); return np.stack([r, r, r], axis=-1)
def _bump():
    u = np.linspace(0, 6 * np.pi, N); b = 0.5 + 0.4 * np.outer(np.sin(u), np.cos(u)); return np.stack([b, b, b], axis=-1)


def test_ramp_removed_by_highpass():
    h = image_to_height(_ramp(), sigma=4.0)
    m = int(0.1 * N)
    assert np.std(h[m:-m, m:-m]) < 0.01                     # a slow ramp does NOT become a slope


def test_ramp_abstains():
    assert auto_bump(_ramp())["abstained"]                 # nothing to bump -> flat


def test_bump_gives_varied_unit_normals():
    res = auto_bump(_bump(), strength=2.0)
    assert not res["abstained"] and res["confidence"] > 0.01
    n = res["normal"]
    assert n.shape == (N, N, 3)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-6)   # unit
    assert np.all(n[..., 2] > 0)                            # all point out (+z)
    assert np.std(n[..., 0]) > 0.05                         # varies -> relief


def test_flat_height_flat_normals():
    n = normal_from_height(np.zeros((N, N)))
    assert np.allclose(n[..., 2], 1.0) and np.allclose(n[..., :2], 0.0)


def test_pack_rgb_range():
    n = auto_bump(_bump())["normal"]
    rgb = pack_normal_rgb(n)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0            # packed into [0,1]


def test_octnormal_roundtrip():
    from holographic.mesh_and_geometry.holographic_octnormal import oct_decode
    q = quantize_normals(auto_bump(_bump())["normal"], bits=8)
    back = oct_decode(q.reshape(-1, q.shape[-1]))
    assert np.allclose(np.linalg.norm(back, axis=-1), 1.0, atol=1e-6)


def test_height_wires_into_material():
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    from holographic.materials_and_texture.holographic_material import Material, texture_field
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=1)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 9) for v in np.linspace(0.05, 0.95, 9)]
    mat = Material(enc, {"albedo": texture_field(enc, grid, [0.5] * len(grid))})
    add_height_channel(mat, enc, grid, _bump())
    assert "height" in mat.channels and np.isfinite(mat.sample("height", [0.5, 0.5]))


def test_deterministic():
    assert np.array_equal(image_to_height(_bump()), image_to_height(_bump()))
