"""Tests for holographic_proctex through the mind: the standard texture menu (2D+3D) and mask refraction."""
import numpy as np


def test_texture_menu_and_samplers_through_the_mind():
    """One named field serves image, volume, and raw points; deterministic; unknown names refused."""
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)

    img = m.texture_image("voronoi", size=48, kind="f2f1", scale=5, seed=1)
    assert img.shape == (48, 48) and 0.0 <= img.min() and img.max() <= 1.0
    assert np.array_equal(img, m.texture_image("voronoi", size=48, kind="f2f1", scale=5, seed=1))

    vol = m.texture_volume("musgrave", res=16, kind="ridged", seed=2)
    assert vol.shape == (16, 16, 16) and np.isfinite(vol).all()

    f = m.proc_texture("marble", seed=0)
    pts = np.random.default_rng(0).uniform(0, 1, (9, 3))
    assert np.asarray(f(pts)).shape == (9,)

    for bad in ("granite2", "perlinx"):
        try:
            m.proc_texture(bad)
            assert False, "unknown texture must raise"
        except ValueError:
            pass


def test_mask_refraction_edge_concentration_through_the_mind():
    """The requested behaviour, pinned: distortion strongest near the mask edge, zero outside; identity at
    ior=1; chromatic separates channels; deterministic ripple."""
    import lecore
    from holographic.misc.holographic_jit import distance_transform
    m = lecore.UnifiedMind(dim=64, seed=0)

    yy, xx = np.mgrid[0:80, 0:80]
    bg = np.stack([np.mod(xx // 6 + yy // 6, 2).astype(float)] * 3, axis=-1)
    mask = (xx - 40) ** 2 + (yy - 40) ** 2 < 26 ** 2

    assert np.array_equal(m.mask_refraction(bg, mask, strength=9.0, ior=1.0), bg)

    ref = m.mask_refraction(bg, mask, strength=9.0, ior=1.5)
    assert np.array_equal(ref[~mask], bg[~mask]), "outside the mask must be untouched"
    d = np.where(mask, distance_transform(~mask), 0.0)
    rim = mask & (d < 0.35 * d.max())
    plateau = mask & (d > 0.7 * d.max())
    delta = np.abs(ref - bg).mean(axis=-1)
    assert delta[rim].mean() > 3.0 * max(delta[plateau].mean(), 1e-9), \
        "distortion must concentrate at the edge (the lens meniscus)"

    chrom = m.mask_refraction(bg, mask, strength=9.0, ior=1.5, chromatic=0.3)
    assert not np.array_equal(chrom[..., 0], chrom[..., 2])

    r1 = m.mask_refraction(bg, mask, strength=9.0, ripple=(2.0, 5.0), seed=3)
    r2 = m.mask_refraction(bg, mask, strength=9.0, ripple=(2.0, 5.0), seed=3)
    assert np.array_equal(r1, r2)


def test_texture_driven_clouds_additive():
    """cloud_scene(texture=...) shapes the cloud from the procedural texture menu; the default (no texture)
    path stays EXACTLY make_cloud's -- the additive contract, pinned."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)

    a = m.cloud_scene(preset="cumulus", quality="fast", seed=0, width=48, height=48, grid=12, steps=16)
    b = m.make_cloud(seed=0, width=48, height=48, grid=12, steps=16, density=6.0)
    assert np.array_equal(np.asarray(a), np.asarray(b)), "no-texture cloud_scene must BE make_cloud"

    tm = m.cloud_scene(texture="musgrave", texture_params={"kind": "ridged", "scale": 2.5},
                       quality="fast", seed=1, width=48, height=48, steps=16)
    assert np.asarray(tm).shape == (48, 48, 3)
    assert not np.array_equal(np.asarray(tm), np.asarray(a)), "a texture cloud must differ from cumulus"
    tm2 = m.cloud_scene(texture="musgrave", texture_params={"kind": "ridged", "scale": 2.5},
                        quality="fast", seed=1, width=48, height=48, steps=16)
    assert np.array_equal(np.asarray(tm), np.asarray(tm2)), "deterministic in seed"


def test_style_transfer_promoted_and_composable():
    """Style transfer, promoted: the postfx 'style_transfer' step equals the direct color_transfer faculty,
    composes with other effects, refuses a missing reference, and the previously-dead user phrasings now
    surface the capability (the discoverability pin for this promotion)."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(0)
    frame = rng.uniform(0.1, 0.9, (32, 40, 3))
    ref = np.clip(rng.normal([0.7, 0.45, 0.25], 0.1, (24, 24, 3)), 0, 1)

    chain = m.postfx_chain(("style_transfer", {"reference": ref, "strength": 0.85}))
    assert np.allclose(chain.apply(frame), m.color_transfer(frame, ref, strength=0.85)), \
        "the chain step must BE the faculty"
    both = m.postfx_chain(("style_transfer", {"reference": ref}), ("film_grain", {"amount": 0.02, "seed": 1}))
    assert both.apply(frame).shape == frame.shape

    try:
        m.postfx_chain(("style_transfer", {})).apply(frame)
        assert False, "a missing reference must raise"
    except ValueError:
        pass

    for q in ("style transfer", "stylize an image", "post process with a style"):
        top = m.find_capability(q)[:1]
        assert top and "Style transfer" in top[0].name, ("promotion pin", q, top and top[0].name)


def test_sampler_and_ramps_through_the_mind():
    """Textures as numbers, numbers as textures: the exact assign->sample roundtrip, ColorRamp stop
    exactness, and the composition that motivated the layer -- a painted map (image_field) driving
    cloud_scene's density like any analytic texture."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)

    # the closing contract
    vals = np.random.default_rng(0).uniform(0, 1, (3, 5))
    tex = m.values_to_texture(vals)
    H, W = tex.shape
    uv = np.array([[(x + 0.5) / W, (y + 0.5) / H] for y in range(H) for x in range(W)])
    assert np.array_equal(m.sample_image(tex, uv).reshape(H, W), vals)

    # ColorRamp semantics
    for interp in ("linear", "constant", "smooth"):
        assert np.allclose(m.ramp([0.0, 0.5, 1.0], [0.0, 1.0, 0.2], interp=interp)([0.0, 0.5, 1.0]),
                           [0.0, 1.0, 0.2])
    assert np.allclose(m.ramp_texture([0.0, 1.0], [0.0, 1.0], size=8), (np.arange(8) + 0.5) / 8)

    # a PAINTED map drives a cloud density through the same field seam analytic textures use
    painted = np.random.default_rng(1).uniform(0, 1, (16, 16))
    f = m.image_field(painted, wrap="repeat")
    img = m.make_cloud(seed=0, width=40, height=40, grid=10, steps=12, field=lambda P: 4.0 * f(P))
    assert np.asarray(img).shape == (40, 40, 3)

    # refusals loud
    for bad in (dict(mode="cubic"), dict(wrap="mirror")):
        try:
            m.sample_image(tex, [[0.5, 0.5]], **bad)
            assert False
        except ValueError:
            pass


def test_http_boundary_mesh_symmetry_and_render_water():
    """The wiring-sweep fixes, pinned: (a) _jsonable emits a Mesh as EXACTLY the dict shape as_mesh accepts,
    so what /invoke returns can be posted straight back (sculpt_prepare's mesh is no longer a repr stub);
    (b) render_water is the one-shot JSON path to water pixels (a WaterBody object cannot cross HTTP)."""
    import numpy as np
    import lecore
    from holographic_service import _jsonable
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    m = lecore.UnifiedMind(dim=64, seed=0)

    mesh = Mesh(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], float), [(0,1,2),(0,1,3),(0,2,3),(1,2,3)])
    j = _jsonable(mesh)
    assert set(j) >= {"vertices", "faces"} and isinstance(j["vertices"], list)
    again = m._as_mesh({"vertices": j["vertices"], "faces": j["faces"]})       # the symmetry
    assert len(again.vertices) == 4 and len(again.faces) == 4

    r = m.sculpt_prepare(mesh, resolution=12, silhouette=None)
    jr = _jsonable(r)
    assert "vertices" in jr["mesh"], "sculpt_prepare's mesh must cross the boundary as data, not a stub"

    img = m.render_water(extent=20.0, seed=0, res=24, width=40, height=30)
    assert np.asarray(img).shape == (30, 40, 3)
    assert np.array_equal(img, m.render_water(extent=20.0, seed=0, res=24, width=40, height=30))
