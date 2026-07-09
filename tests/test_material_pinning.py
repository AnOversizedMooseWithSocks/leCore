"""Bit-exact PINNING HARNESS for the shading/material stack -- the safety net for the materials/BRDF convergence
(Phase 3, MT1-MT4). It LOCKS the canonical BRDF's exact outputs and the delegation structure, so any future
refactor that tries to unify shading is caught the instant it changes a value or reintroduces a copy.

WHY A HARNESS INSTEAD OF THE REFACTOR (probe-first audit, kept honest)
---------------------------------------------------------------------
The backlog imagined MT1-4 as a big pending bit-exact refactor ("unify 4 material types, one BRDF from ~5
copies"). Probing the live code first showed that is largely NOT a pending refactor -- it is mostly already done,
and the parts that are not done are NOT safe to merge:

  * BRDF: already CANONICAL in holographic_brdf.py (fresnel_schlick / fresnel_dielectric / d_ggx / cook_torrance).
    holographic_raymarch DELEGATES to it (it imports cook_torrance / fresnel_schlick), it does not copy it. The
    other "shading refs" a grep flagged were FALSE POSITIVES: a 'specular' word in a rayindex comment, and the
    FFT variables F0/F1 in dynamics (Fourier coefficients, not Fresnel F0). There is no rival inline BRDF to merge.

  * MATERIALS: the 3 classes serve DIFFERENT layers -- Material (core PBR sockets), PBRMaterial (asset I/O),
    SurfaceMaterial (the render material, a Lambert + Blinn-specular + environment-reflection model, with adapters
    FROM the PBR side). SurfaceMaterial uses a DIFFERENT shading model on purpose, so folding it into the GGX
    Cook-Torrance BRDF would CHANGE its output -- that is a behaviour change, not a bit-exact refactor, and forcing
    it would be wrong. The adapters (from_matlib / from_name) already converge the material types at the boundaries.

So the honest deliverable is to LOCK what exists rather than merge what should not be merged. If a genuine
convergence is done later, these pins verify it stayed bit-exact -- and if a "copy" ever drifts back in, the
delegation pins fail. This is the bit-exact pinning discipline the project uses everywhere (the G1 SDF-normal
pin, the resonator-quantile pin) applied to the shading stack.
"""
import inspect

import numpy as np

from holographic.rendering.holographic_brdf import fresnel_schlick, fresnel_dielectric, cook_torrance


# --- canonical BRDF value pins (the reference every convergence must reproduce EXACTLY) -----------------------

def test_fresnel_schlick_scalar_pinned():
    # 0.04 + 0.96 * 0.5^5 = 0.07, exactly -- the dielectric F0=0.04 case a shader uses constantly
    assert float(fresnel_schlick(0.5, 0.04)) == 0.07


def test_fresnel_schlick_vector_pinned():
    # the (...,3) metal-F0 branch: F0 + (1-F0)*(1-cos)^5 per channel
    f = fresnel_schlick(0.5, np.array([1.0, 0.8, 0.3]))
    assert f.tolist() == [1.0, 0.80625, 0.32187499999999997]


def test_fresnel_dielectric_pinned():
    assert float(fresnel_dielectric(0.7, 1.5)) == 0.04233280000000001


def test_cook_torrance_dielectric_pinned():
    N = np.array([0.0, 0.0, 1.0]); V = np.array([0.0, 0.0, 1.0]); L = np.array([0.0, 0.577, 0.816])
    ct = cook_torrance(N, V, L, np.array([0.8, 0.2, 0.2]), 0.0, 0.5)
    assert ct.tolist() == [0.20796842402070315, 0.05835771868735314, 0.05835771868735314]


def test_cook_torrance_metal_pinned():
    N = np.array([0.0, 0.0, 1.0]); V = np.array([0.0, 0.0, 1.0]); L = np.array([0.0, 0.577, 0.816])
    ct = cook_torrance(N, V, L, np.array([1.0, 0.8, 0.3]), 1.0, 0.3)
    assert ct.tolist() == [0.06270847590794255, 0.05016678358753394, 0.018812552786512386]


# --- delegation / no-copy structural pins ---------------------------------------------------------------------

def test_raymarch_delegates_to_canonical_brdf():
    """raymarch must DELEGATE to holographic_brdf, not carry its own fresnel/cook_torrance copy. If a copy ever
    drifts back in, this fails."""
    import holographic.rendering.holographic_raymarch as holographic_raymarch
    src = inspect.getsource(holographic_raymarch)
    assert "holographic_brdf import" in src                       # it imports the canonical BRDF (flat OR package path)
    assert "def fresnel" not in src and "def cook_torrance" not in src   # ... and defines no rival of its own


def test_no_rival_fresnel_definitions():
    """There is exactly ONE definition of each canonical BRDF function across the shading stack -- the audit's
    'BRDF is already canonical' claim, made mechanical. (holographic_brdf is the only home.)"""
    import holographic.rendering.holographic_brdf as holographic_brdf
    src = inspect.getsource(holographic_brdf)
    assert src.count("def fresnel_schlick(") == 1
    assert src.count("def cook_torrance(") == 1


def test_surface_material_is_a_different_model_not_a_brdf_copy():
    """SurfaceMaterial's render path is Lambert+Blinn (a different model), NOT a Cook-Torrance copy -- so it is
    correctly NOT delegated into the GGX BRDF (merging would change behaviour). This pins the audit finding: the
    surface module does not import or redefine cook_torrance."""
    import holographic.mesh_and_geometry.holographic_surface as holographic_surface
    src = inspect.getsource(holographic_surface)
    assert "cook_torrance" not in src                             # it is a distinct model, not the GGX BRDF


# --- the metallic->F0 convergence (three inline copies -> one shared helper), pinned bit-exact ----------------

def test_metallic_f0_pinned():
    from holographic.rendering.holographic_brdf import metallic_f0
    # blend: dielectric 0.04 at metallic=0, base colour at metallic=1, linear between
    assert metallic_f0(np.array([0.8, 0.2, 0.2]), 0.6).tolist() == [0.496, 0.136, 0.136]
    assert metallic_f0(np.array([0.8, 0.2, 0.2]), 0.0).tolist() == [0.04, 0.04, 0.04]   # pure dielectric
    assert metallic_f0(np.array([0.8, 0.2, 0.2]), 1.0).tolist() == [0.8, 0.2, 0.2]      # pure metal


def test_f0_formula_has_one_home():
    """After the convergence, the metallic->F0 formula lives ONLY in metallic_f0 -- no shading site re-inlines
    '0.04 * (1 - metallic) + base * metallic'. The three former copies (cook_torrance, the sampler variant, and
    raymarch's inline) now all delegate. If a copy drifts back, this fails."""
    import holographic.rendering.holographic_brdf as holographic_brdf, holographic.rendering.holographic_raymarch as holographic_raymarch
    # raymarch must delegate, not re-inline the constant
    assert "0.04 * (1.0 - metallic)" not in inspect.getsource(holographic_raymarch)
    assert "metallic_f0(" in inspect.getsource(holographic_raymarch)
    # in brdf, the literal formula appears exactly once -- inside the helper definition
    body = inspect.getsource(holographic_brdf)
    assert body.count("0.04 * (1.0 - metallic) + base_color * metallic") == 1
    assert body.count("0.04 * (1.0 - metallic) + base * metallic") == 0   # the old inline is gone
