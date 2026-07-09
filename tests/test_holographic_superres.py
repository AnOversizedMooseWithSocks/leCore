"""Inverse-rendering ST3: guided (joint-bilateral) upsampling steered by the full-res G-buffer."""
import numpy as np
from holographic.rendering.holographic_render import Camera
from holographic.mesh_and_geometry.holographic_sdf import box
from holographic.rendering.holographic_raymarch import render_sdf
from holographic.rendering.holographic_renderchannels import render_channels
from holographic.rendering.holographic_fsr import easu_upscale, _box_downscale
from holographic.rendering.holographic_superres import guided_upsample, _psnr

CAM = Camera(eye=(2.5, 1.8, 2.5), target=(0, 0, 0), fov_deg=50.0)
RKW = dict(ao=False, shadows=False, reflect=0.0)


def _scene():
    sdf = box(1.0, 0.7, 0.5)
    native = render_sdf(sdf, CAM, width=64, height=64, **RKW)
    gb = render_channels(sdf, CAM, want=["normal", "depth"], width=64, height=64, **RKW)
    low = _box_downscale(native, 2)
    return native, gb, low


def test_guided_beats_plain_upscale():
    native, gb, low = _scene()
    guided = guided_upsample(low, gb["normal"], guide_depth=gb["depth"])[:64, :64]
    plain = easu_upscale(low, 2.0)[:64, :64]
    assert _psnr(guided, native) > _psnr(plain, native)      # the guide snaps colour edges to the geometry


def test_guided_shape_and_range():
    native, gb, low = _scene()
    g = guided_upsample(low, gb["normal"], guide_depth=gb["depth"])[:64, :64]
    assert g.shape == (64, 64, 3) and g.min() >= 0.0 and g.max() <= 1.0


def test_normal_only_guide_works():
    native, gb, low = _scene()
    g = guided_upsample(low, gb["normal"])                   # albedo/depth default to the normal guide
    assert g.shape[:2] == gb["normal"].shape[:2] and np.isfinite(g).all()


def test_deterministic():
    native, gb, low = _scene()
    a = guided_upsample(low, gb["normal"], guide_depth=gb["depth"])
    b = guided_upsample(low, gb["normal"], guide_depth=gb["depth"])
    assert np.array_equal(a, b)
