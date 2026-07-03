"""Inverse-rendering IR14: render channels (AOVs) -- a channel is an unbind; beauty default unchanged."""
import numpy as np
from holographic_render import Camera
from holographic_sdf import box, sphere
from holographic_raymarch import render_sdf
from holographic_renderchannels import render_channels, material_channels, composites_to_beauty

CAM = Camera(eye=(2.5, 1.8, 2.5), target=(0, 0, 0), fov_deg=50.0)
RKW = dict(width=40, height=40, ao=False, shadows=False, reflect=0.0)


def test_default_is_beauty_only_bit_identical():
    only = render_channels(box(1, 0.7, 0.5), CAM, **RKW)
    assert set(only) == {"beauty"}
    assert np.array_equal(only["beauty"], render_sdf(box(1, 0.7, 0.5), CAM, **RKW))


def test_gbuffer_passes_valid():
    ch = render_channels(box(1, 0.7, 0.5), CAM, want=["depth", "normal", "position", "mask"], **RKW)
    hit = ch["mask"] > 0.5
    assert np.all(ch["depth"][hit] > 0)
    assert np.allclose(np.linalg.norm(ch["normal"][hit], axis=-1), 1.0, atol=1e-4)
    assert ch["position"].shape == (40, 40, 3)


def test_object_mattes_composite_to_coverage():
    objs = [box(0.6, 0.6, 0.6).translate((-0.9, 0, 0)), sphere(0.7).translate((0.9, 0, 0))]
    union = objs[0].union(objs[1])
    ch = render_channels(union, CAM, want=["mask"], objects=objs, **RKW)
    assert composites_to_beauty(ch) == 0.0                 # cover the frame, no gaps
    assert np.all(ch["object:0"] * ch["object:1"] == 0.0)  # disjoint, no double-count


def test_material_channels_are_unbinds():
    from holographic_fpe import VectorFunctionEncoder
    from holographic_material import Material, texture_field
    from holographic_ai import cosine
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=1)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 9) for v in np.linspace(0.05, 0.95, 9)]
    mat = Material(enc, {"albedo": texture_field(enc, grid, [0.6] * len(grid)),
                         "roughness": texture_field(enc, grid, [0.3] * len(grid))})
    mch = material_channels(mat)
    assert set(mch) == {"albedo", "roughness"}
    assert cosine(mch["albedo"], Material._unit(mat.channels["albedo"])) > 0.4


def test_selecting_subset():
    ch = render_channels(box(1, 0.7, 0.5), CAM, want=["depth"], **RKW)
    assert set(ch) == {"beauty", "depth"}                  # opt-in per channel (memory: only what you ask for)


def test_deterministic():
    a = render_channels(box(1, 0.7, 0.5), CAM, want=["normal"], **RKW)["normal"]
    b = render_channels(box(1, 0.7, 0.5), CAM, want=["normal"], **RKW)["normal"]
    assert np.array_equal(a, b)
