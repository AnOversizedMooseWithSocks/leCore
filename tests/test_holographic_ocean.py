"""Tests for holographic_ocean -- the one-call Gerstner water preset, through the mind."""
import numpy as np


def test_make_water_through_the_mind():
    """make_water: one call -> height/positions/normals/bank; deterministic; animates; presets distinct;
    quick_material: plain numbers -> a material ball, rough vs polished visibly different."""
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)

    w = m.make_water(res=48, extent=30.0, seed=5, preset="ocean", shaded=True)
    assert w["height"].shape == (48, 48) and w["positions"].shape == (48, 48, 3)
    assert w["normals"].shape == (48, 48, 3) and w["image"].shape == (48, 48, 3)
    # unit normals
    assert np.allclose(np.linalg.norm(w["normals"], axis=-1), 1.0, atol=1e-9)
    # deterministic
    w2 = m.make_water(res=48, extent=30.0, seed=5, preset="ocean", shaded=True)
    assert np.array_equal(w["height"], w2["height"]) and np.array_equal(w["image"], w2["image"])
    # animation moves the surface, same bank
    w_t = m.make_water(res=48, extent=30.0, seed=5, preset="ocean", t=4.0)
    assert np.abs(w_t["height"] - w["height"]).mean() > 1e-3
    assert np.array_equal(w_t["bank"]["k"], w["bank"]["k"])
    # unknown preset refused
    try:
        m.make_water(preset="lava")
        assert False, "unknown preset must raise"
    except ValueError:
        pass

    ball_metal = m.quick_material(color=(1.0, 0.3, 0.1), roughness=0.1, metallic=1.0, res=48)
    ball_rough = m.quick_material(color=(1.0, 0.3, 0.1), roughness=0.95, metallic=0.0, res=48)
    a, b = np.asarray(ball_metal), np.asarray(ball_rough)
    assert a.shape == (48, 48, 3) and 0.0 <= a.min() and a.max() <= 1.0
    assert float(np.abs(a - b).mean()) > 0.01, "roughness/metallic must visibly change the ball"


def test_water_body_and_cloud_scene_through_the_mind():
    """The bundled tools: water_body (container-first, library IORs, pre-balanced lighting) and cloud_scene
    (presets x measured quality tiers). Pins the contracts a user relies on: lit output, correct refraction
    indices from the material library, coherent animation, loud refusals."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)

    # open water renders LIT and deterministic
    wb = m.water_body(extent=50.0, seed=1, res=80)
    img = wb.render("fast", width=120, height=90)
    assert img.shape == (90, 120, 3)
    assert float(np.percentile(img.mean(axis=2), 50)) > 0.25, "the bundled lighting must come out lit"
    img2 = m.water_body(extent=50.0, seed=1, res=80).render("fast", width=120, height=90)
    assert np.array_equal(img, img2), "deterministic in seed"

    # contained water: builds for every stock vessel; ripples animate; library IOR reaches the callback
    gb = m.water_body(container="glass", level=0.7, ripple=0.4, seed=1)
    gb.at_time(2.0); a = gb.water_sdf(np.array([[0.1, 0.85, 0.05]]))[0]
    gb.at_time(0.0); b = gb.water_sdf(np.array([[0.1, 0.85, 0.05]]))[0]
    assert abs(a - b) > 1e-9
    assert abs(m.water_body(container="glass", material="oil").ior - 1.47) < 1e-9, \
        "oil must refract at oil's index (the library), not water's"
    for c in ("pool", "bowl"):
        m.water_body(container=c, level=0.5, seed=0)

    # cloud_scene: preset/tier gates refuse loudly; overrides pass through (proven by a tiny fast render)
    for bad in (dict(preset="cirrus"), dict(quality="ultra")):
        try:
            m.cloud_scene(**bad)
            assert False, "must refuse %r" % bad
        except ValueError:
            pass
    img = m.cloud_scene(preset="cumulus", quality="fast", seed=0, width=64, height=64, grid=16, steps=24)
    assert np.asarray(img).shape == (64, 64, 3) and 0.0 <= np.min(img) and np.max(img) <= 1.0
