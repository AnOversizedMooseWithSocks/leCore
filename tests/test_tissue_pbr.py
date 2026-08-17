"""Physically-based tissue materials: organs, bone, fat and skin that are not flat.

Christensen-Burley (Pixar TM 15-04) is the production parameterisation and needs "per-channel
single scattering albedo and scattering distance parameters". The ORDERING of those distances
is grounded in measured SDOCT scattering coefficients rather than art direction: bone and skin
1.947-2.134 /mm, liver and brain 1.303-1.461, testis and spleen 0.523-0.634 -- "the scattering
coefficient is tissue specific".
"""
from holographic.materials_and_texture import holographic_creaturematerial as CM


def test_every_tissue_has_a_complete_pbr_description():
    t = CM.tissue_pbr_table()
    for name, v in t.items():
        assert len(v["base_color"]) == 3, name
        assert 0.0 <= v["roughness"] <= 1.0, name
        assert 0.0 <= v["sss_weight"] <= 1.0, name
        assert len(v["sss_radius"]) == 3, name
        assert all(x > 0 for x in v["sss_radius"]), name


def test_red_scatters_deepest_in_every_soft_tissue():
    """The signature of flesh. A scalar SSS radius cannot produce the warm silhouette that
    makes skin read as skin rather than as red plastic."""
    t = CM.tissue_pbr_table()
    for soft in ("skin", "fat", "muscle", "organ", "liver", "spleen"):
        r, g, b = t[soft]["sss_radius"]
        assert r > g > b, (soft, t[soft]["sss_radius"])


def test_scattering_distance_follows_the_measured_coefficients():
    """Distance is the reciprocal of the measured coefficient, so viscera (low coefficient)
    must scatter FURTHEST and bone (high coefficient) least. If this inverts, the table has
    drifted from its source."""
    t = CM.tissue_pbr_table()
    order = ["spleen", "organ", "muscle", "skin", "bone"]
    radii = [t[k]["sss_radius"][0] for k in order]
    assert radii == sorted(radii, reverse=True), list(zip(order, radii))
    # and the recorded coefficients run the opposite way, which is the consistency check
    coeffs = [t[k]["scatter_mm_inv"] for k in order]
    assert coeffs == sorted(coeffs), list(zip(order, coeffs))


def test_hard_tissues_are_not_translucent_like_organs():
    """Chitin and bone must not glow like a spleen -- a shell that scatters like viscera is
    the classic giveaway of a table filled in by feel."""
    t = CM.tissue_pbr_table()
    assert t["chitin"]["sss_weight"] < 0.2
    assert t["bone"]["sss_weight"] < t["organ"]["sss_weight"] / 2
    assert t["chitin"]["roughness"] < t["fat"]["roughness"]     # shell is glossier than fat


def test_unknown_tissue_raises_rather_than_guessing():
    try:
        CM.tissue_pbr("unobtainium")
        assert False, "unknown tissue must raise, not silently return a default"
    except ValueError:
        pass


def test_faculty_is_wired():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    v = m.tissue_pbr("skin")
    assert v["sss_radius"][0] > v["sss_radius"][2]
    assert len(m.tissue_pbr_table()) >= 10
