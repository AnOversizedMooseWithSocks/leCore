"""O6: physically-based specular in the mesh rasteriser.

The rasteriser was Lambert-only, so the best SHAPE and the best MATERIAL came from different
renderers -- the salamander's silhouette was right in the mesh path while its wet sheen only
existed in the SDF raymarcher. This closes that split.

SOTA is settled and was NOT reimplemented: Cook-Torrance with GGX/Trowbridge-Reitz D, SMITH
G (Heitz showed Smith is the correct microsurface profile over V-cavity), Schlick F, and
F0 = m*albedo + (1-m)*0.04. holographic_brdf.cook_torrance already ships exactly that, so
this path REUSES it -- one shared implementation of any algorithm, never two.
"""
import numpy as np
from holographic.rendering.holographic_render import Light


def _scene(m, res=32):
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    t = m.mesh_from_sdf(sph, ((-1.3,) * 3, (1.3,) * 3), res=res, vectorized=True)
    cd = m.fit_camera(t, direction=(0.4, 0.5, 1.0), fov_deg=40, width=160, height=160,
                      margin=1.1)
    cam = m.camera(eye=tuple(cd["eye"]), target=tuple(cd["target"]), up=tuple(cd["up"]),
                   fov_deg=cd["fov_deg"], aspect=1.0)
    lights = [Light("directional", direction=(0.4, -0.7, -0.5), intensity=2.0),
              Light("ambient", intensity=0.2)]
    return t, cam, lights


def test_pbr_produces_a_tight_bright_highlight():
    """The microfacet signature: a small very bright lobe over a DARKER body, versus
    Lambert's broad flat falloff. Peak rises while the 99th percentile drops."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    t, cam, lights = _scene(m)
    kw = dict(width=160, height=160, lights=lights, base_color=(0.3, 0.2, 0.15),
              ambient=0.0, smooth=True)
    lam = np.asarray(m.render_mesh(t, cam, **kw), float)
    pbr = np.asarray(m.render_mesh(t, cam, pbr=(0.0, 0.15), **kw), float)
    assert pbr.max() > lam.max()                       # a brighter specular peak
    assert np.percentile(pbr, 99) < np.percentile(lam, 99)   # over a darker body


def test_default_is_bit_identical_to_the_lambert_path():
    """House rule: existing decisions never flip. pbr=None must reproduce every shipped
    render exactly, so this is checked as bitwise equality, not 'close'."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    t, cam, lights = _scene(m)
    kw = dict(width=160, height=160, lights=lights, base_color=(0.3, 0.2, 0.15),
              ambient=0.0, smooth=True)
    a = np.asarray(m.render_mesh(t, cam, **kw), float)
    b = np.asarray(m.render_mesh(t, cam, pbr=None, **kw), float)
    assert np.array_equal(a, b)


def test_roughness_dims_the_peak_monotonically():
    """The physical control actually controls. MEASURED peak vs roughness: 0.10 -> 1.000,
    0.25 -> 0.673, 0.55 -> 0.246. If this ever inverts, D and G have been swapped or alpha
    is no longer roughness^2.

    A first version of this test asserted that a ROUGH highlight covers more pixels above a
    fixed 0.25 threshold. That measured nothing: by roughness 0.55 the entire highlight sits
    BELOW 0.25, so both sides read 0.000%. The threshold was wrong, not the shader -- the
    bright AREA peaks at intermediate roughness and then falls with the peak, so a fixed
    brightness cut cannot express 'wider'. Monotone peak falloff is the robust claim."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    t, cam, lights = _scene(m)
    kw = dict(width=160, height=160, lights=lights, base_color=(0.3, 0.2, 0.15),
              ambient=0.0, smooth=True)
    peaks = [np.asarray(m.render_mesh(t, cam, pbr=(0.0, r), **kw), float).max()
             for r in (0.10, 0.25, 0.55)]
    assert peaks == sorted(peaks, reverse=True), peaks
    assert peaks[0] > 1.5 * peaks[-1]                  # and the falloff is substantial
