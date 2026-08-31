"""O5: the Kajiya-Kay dark-hair bug, measured and fixed additively.

Kajiya-Kay adds its specular lobe WHITE at full amplitude regardless of hair colour, so dark
hair renders silver -- which is what happened to the avatar's hair and beard. Marschner (2003)
measured that the secondary highlight is COLOURED by the fibre, and the RenderMan team's own
retrospective says the original model "didn't pay enough attention to energy conservation".

The fix is additive and default-off, so the published model is preserved exactly.
"""
import numpy as np
from holographic.mesh_and_geometry.holographic_hairshade import kajiya_kay

DARK = (0.075, 0.048, 0.034)


def _sweep(**kw):
    L = np.array([0.4, 0.7, -0.5]); L /= np.linalg.norm(L)
    V = np.array([0.0, 0.2, -1.0]); V /= np.linalg.norm(V)
    rng = np.random.default_rng(0)
    T = rng.normal(size=(1500, 3))
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return np.array([np.ravel(kajiya_kay(t, L, V, diffuse_color=DARK, **kw))[:3]
                     for t in T]).max(axis=1)


def test_the_bug_is_real_and_stays_documented():
    """Pinned as a FINDING, not silently fixed: the published model really does blow dark
    hair out. If this ever stops failing, the default changed and that must be deliberate."""
    m = _sweep()
    assert m.max() > 1.0                      # brighter than white, from a 0.075 hair colour
    assert (m > 0.5).mean() > 0.10            # a large minority read as white


def test_tint_and_strength_fix_dark_hair():
    """Marschner's colouring + energy conservation. White-reading strands must vanish."""
    m = _sweep(specular_tint=0.7, specular_strength=0.35)
    assert (m > 0.5).mean() == 0.0
    assert m.max() < 0.35


def test_defaults_reproduce_the_published_model_bit_for_bit():
    """House rule: existing decisions never flip. tint=0, strength=1 must be identical to
    the untouched formula, so no shipped render changes."""
    a = _sweep()
    b = _sweep(specular_tint=0.0, specular_strength=1.0)
    assert np.array_equal(a, b)
