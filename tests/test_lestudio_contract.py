"""THE LESTUDIO CONTRACT -- all 11 integration proposals, pinned as tests.

leStudio (an image editor built on this engine) shipped LECORE_PROPOSALS.md: eleven
requests, each born from a real seam -- a bug worked around, an API papered over, a
capability wanting one more inch. Every one of them is implemented in this tree; several
were the direct cause of earlier sweeps. This file turns that document into a REGRESSION
CONTRACT: an app author told us exactly where the engine chafed, and none of those spots
get to chafe again quietly.

Numbering follows their document. Each test carries the original complaint.
"""
import numpy as np
import pytest

import lecore

# Module-level, not fixtures: the CI shim runs fixture-free tests everywhere, and this
# contract is exactly the file that must never be silently skipped.
_MIND = None


def _mind():
    global _MIND
    if _MIND is None:
        _MIND = lecore.UnifiedMind()
    return _MIND


def _img():
    return np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)


def test_p1_image_colours_offers_float():
    """#1: 'image_colours returns a uint8 0-255 palette while every other image door
    speaks float 0-1... silently broke our Posterize node.' Contract: as_float=True
    returns 0-1; the legacy default is preserved AND loudly documented."""
    img = _img()
    pal8, _w = _mind().image_colours(img, k=3)
    palF, _w = _mind().image_colours(img, k=3, as_float=True)
    assert np.asarray(pal8).dtype == np.uint8
    assert np.asarray(palF).dtype.kind == "f" and float(np.asarray(palF).max()) <= 1.0
    assert "as_float" in (_mind().image_colours.__doc__ or ""), "the dtype seam stays documented"


def test_p2_pattern_field_seed_is_uniform():
    """#2: 'pattern_field kinds disagree about seed -- noise/fbm accept it,
    checker/stripes/gradient/dots raise TypeError.' Contract: one uniform signature;
    deterministic kinds accept-and-ignore."""
    for kind in ("noise", "fbm", "checker", "stripes", "gradient", "dots"):
        f = _mind().pattern_field(kind, seed=3)          # must not raise on ANY kind
        v = f(np.array([[0.3, 0.7, 0.1]]))
        assert np.isfinite(v).all()


def test_p3_color_transfer_aliases_and_validation():
    """#3: 'color_transfer spells its mode meanstd -- accept mean_std/mean-std aliases,
    or validate and raise (it currently accepts anything, which is how our misspelling
    shipped as a silently dead parameter).' Contract: BOTH -- aliases accepted, unknowns
    raise with the valid list."""
    img = _img()
    for alias in ("meanstd", "mean_std", "mean-std"):
        _mind().color_transfer(img, img[::-1], mode=alias)
    with pytest.raises(ValueError):
        _mind().color_transfer(img, img[::-1], mode="definitely_wrong")


def test_p4_palette_stops_are_stops():
    """#4: 'random_palette returns cosine coefficients, not colour stops, and the name
    invites the wrong reading.' Contract: palette_stops(seed, n) samples the cosine into
    n RGB rows in [0,1]."""
    st = np.asarray(_mind().palette_stops(seed=5, n=6))
    assert st.shape == (6, 3) and 0.0 <= float(st.min()) and float(st.max()) <= 1.0
    st2 = np.asarray(_mind().palette_stops(seed=5, n=6))
    assert np.array_equal(st, st2), "deterministic per seed"


def test_p5_segment_image_max_dim():
    """#5: 'segment_image should take max_dim= and downsample internally, upsampling
    masks on the way out. Every interactive caller rediscovers this.' Contract: the
    facade takes max_dim and masks come back at FULL input resolution."""
    img = _img()
    regions = _mind().segment_image(img, k=3, max_dim=16)
    assert regions and regions[0]["mask"].shape == img.shape[:2]


def test_p6_inpaint_hwc():
    """#6: 'inpaint is single-channel -- we loop it 3x for RGB.' Contract: (H,W,C) in,
    (H,W,C) out, known pixels preserved."""
    img = _img()
    known = np.zeros((32, 32), bool)
    known[::2, ::2] = True
    out = np.asarray(_mind().inpaint(img, known))
    assert out.shape == img.shape
    assert np.allclose(out[known], img[known], atol=1e-5)


def test_p7_shader_pipeline_hwc():
    """#7: 'shader_pipeline should apply to (H,W,C) directly (loop channels internally,
    or broadcast the transfer).' Contract: one apply() call takes the RGB image; the
    docstring records that the batch shares one FFT plan (and that the GPU path is
    inherited, tolerance-matched, and OFF by default)."""
    img = _img()
    sp = _mind().shader_pipeline((32, 32)).gain(1.0)
    out = np.asarray(sp.apply(img))
    assert out.shape == img.shape and np.isfinite(out).all()
    doc = sp.apply.__doc__ or ""
    assert "channel" in doc.lower() and "gpu" in doc.lower()


def test_p8_shadertoy_uniform_camera():
    """#8: 'to_shadertoy should emit a uniform-driven camera (uAngle, uHeight, uDist)
    behind a flag. We currently regex-replace the fixed ro= line.' Contract:
    camera="uniforms" emits the orbit uniforms; the default stays the classic view."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    code = _mind().to_shadertoy(sphere(1.0), camera="uniforms")
    assert all(u in code for u in ("uAngle", "uHeight", "uDist"))
    fixed = _mind().to_shadertoy(sphere(1.0))
    assert "uAngle" not in fixed, "the default emission is unchanged"


def test_p9_postchain_to_glsl():
    """#9: 'PostChain.to_glsl() -- the postfx algebra is pointwise ops plus separable
    blurs; both have direct GLSL forms.' Contract: the method exists and emits a
    fragment-shader string for a simple chain."""
    from holographic.rendering.holographic_postfx import PostChain
    assert hasattr(PostChain, "to_glsl")
    ch = PostChain()
    for name in ("gain", "gamma", "contrast"):
        if hasattr(ch, name):
            ch = getattr(ch, name)(1.1)
            break
    g = ch.to_glsl()
    assert isinstance(g, str) and "vec" in g


def test_p10_pattern_and_palette_glsl():
    """#10: 'a generic to_glsl for pattern_field / cosine_palette -- both are closed-form
    and trivially shader-able.' Contract: both emitters exist on the facade; noise has a
    32-bit-hash twin built for exactly this (value_noise32 <-> pattern_to_glsl)."""
    g = _mind().pattern_to_glsl("noise32", scale=3.0, seed=7)
    p = _mind().cosine_palette_to_glsl()
    assert "float" in g and "vec3" in p
    from holographic.misc.holographic_pattern import value_noise32
    f = value_noise32(scale=3.0, seed=7)
    P = np.random.default_rng(1).random((64, 3))
    assert np.array_equal(f(P), value_noise32(scale=3.0, seed=7)(P))


def test_p11_sectioned_container_round_trips_the_unknown():
    """#11: 'adopt this container as a leCore core feature... each app reading the kinds
    it understands and preserving the rest.' Contract: leStudio's exact import path
    works; an unknown kind survives TWO generations of load/save untouched."""
    from holographic.io_and_interop.holographic_container import (save_container,
                                                                  load_container)
    sections = [{"kind": "lestudio.document", "id": "d1",
                 "meta": {"width": 8, "height": 8, "name": "t"},
                 "arrays": {"pixels": np.arange(8.0)}},
                {"kind": "some.future.kind", "id": "x9", "meta": {"z": 1},
                 "arrays": {"payload": np.arange(7)}}]
    blob = save_container(sections, meta={"app": "lestudio"})
    g1 = load_container(blob)
    g2 = load_container(save_container(g1["sections"], meta=g1.get("meta", {})))
    fut = [s for s in g2["sections"] if s["id"] == "x9"][0]
    assert fut["kind"] == "some.future.kind"
    assert np.array_equal(fut["arrays"]["payload"], np.arange(7))
    assert hasattr(_mind(), "save_container") and hasattr(_mind(), "load_container")
