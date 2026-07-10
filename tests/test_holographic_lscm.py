"""F1 -- LSCM, and the two metrics that were hiding a folded map.

The backlog: *"UV = one linear solve + seam/atlas packing. Bar: unwrap a benchmark mesh, compare conformal
distortion vs xatlas."* The solve is here, and the bar turned out to need a metric the module did not have.

THE MEASURED TABLE (mean quasi-conformal ratio sigma1/sigma2; 1.0 is conformal):

    surface          lscm      isomap    planar
    flat patch    1.00000     1.10866   1.00000    <- LSCM is EXACT on a developable surface
    hemisphere    1.08600     1.87800   4.39000

THREE THINGS THE MEASUREMENT TAUGHT:

  1. **LSCM buys angles with AREA** -- 0.4420 area spread on a hemisphere cap against isomap's 0.2957. That is the
     definition of a conformal map, not a defect. Compare charts on the functional they optimise.
  2. **The mean quasi-conformal ratio is unbounded.** One near-degenerate face sends sigma2 to zero. On a cap
     stretched 6x in z, LSCM's mean is 398.0 and its median 4.8. Report the median.
  3. **Neither the stretch metric nor the mean ratio can see a FOLD.** On that stretched cap, isomap has a BETTER
     mean (2.573 vs 398.038) while folding 128 of 256 faces against LSCM's 72.

And a fold is a **minority orientation**, not a negative determinant. My first `flipped` counted `det J < 0` and
reported the planar chart as flipping 256 of 256 faces on a hemisphere cap. It had folded none of them: the chart
was globally mirrored.
"""

import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_mesh import Mesh
from holographic.mesh_and_geometry.holographic_meshuv import (
    flat_grid_mesh, hemisphere_cap, lscm, uv_angle_distortion, uv_area_distortion, uv_distortion, uv_report,
    uv_unwrap)


def _stretched_cap(zs, subdiv=3):
    """A cap with its z scaled: the curvature knob. At zs=6 every chart here folds, which is the interesting case."""
    base = hemisphere_cap(subdiv)
    V = np.asarray(base.vertices, float).copy()
    V[:, 2] *= zs
    return Mesh(V, np.asarray(base.faces, int))


def test_selftest_runs():
    from holographic.mesh_and_geometry import holographic_meshuv as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# LSCM is a conformal map
# ---------------------------------------------------------------------------------------------------------

def test_lscm_is_exactly_conformal_on_a_developable_surface():
    m = flat_grid_mesh(6)
    d = uv_angle_distortion(m, lscm(m))
    assert abs(d["median"] - 1.0) < 1e-9
    assert abs(d["max"] - 1.0) < 1e-9                 # EVERY face, not just on average
    assert d["flipped"] == 0
    assert uv_area_distortion(m, lscm(m)) < 1e-9      # a plane is conformal AND isometric to itself


def test_lscm_beats_the_other_charts_on_angle():
    for mesh in (flat_grid_mesh(6), hemisphere_cap(2)):
        rep = uv_report(mesh)
        assert rep["lscm"]["angle"]["median"] <= rep["isomap"]["angle"]["median"] + 1e-9
        assert rep["lscm"]["angle"]["median"] <= rep["planar"]["angle"]["median"] + 1e-9


def test_lscm_is_deterministic_and_pins_are_honoured():
    m = hemisphere_cap(2)
    assert np.array_equal(lscm(m), lscm(m))
    uv = lscm(m, pins=[(0, 0.0, 0.0), (5, 1.0, 0.0)])
    assert np.allclose(uv[0], (0.0, 0.0)) and np.allclose(uv[5], (1.0, 0.0))

    # ... and the pin choice CHANGES the map, which is why the default is stated rather than arbitrary
    assert not np.allclose(uv, lscm(m, pins=[(1, 0.0, 0.0), (7, 1.0, 0.0)]))


def test_lscm_routes_through_uv_unwrap_and_packs_to_the_unit_square():
    m = flat_grid_mesh(5)
    uv = uv_unwrap(m, method="lscm")
    assert uv.shape == (len(m.vertices), 2)
    assert uv.min() >= -1e-9 and uv.max() <= 1.0 + 1e-9


def test_lscm_rejects_degenerate_pin_choices():
    m = flat_grid_mesh(4)
    with pytest.raises(ValueError):
        lscm(m, pins=[(0, 0.0, 0.0), (0, 1.0, 0.0)])            # the same vertex twice
    with pytest.raises(ValueError):
        lscm(m, pins=[(0, 0.0, 0.0)])                            # one pin leaves a similarity free


# ---------------------------------------------------------------------------------------------------------
# KEPT NEGATIVE 1: a conformal map pays in area
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_lscm_wins_on_angle_and_loses_on_area():
    m = hemisphere_cap(3)
    rep = uv_report(m, methods=("lscm", "isomap"))
    assert rep["lscm"]["angle"]["median"] < rep["isomap"]["angle"]["median"]
    assert rep["lscm"]["area"] > rep["isomap"]["area"]           # measured 0.4420 vs 0.2957


def test_the_report_carries_all_three_metrics_so_nobody_picks_one_blind():
    rep = uv_report(flat_grid_mesh(5))
    for meth in ("lscm", "isomap", "planar"):
        assert set(rep[meth]) == {"angle", "area", "stretch"}
        assert set(rep[meth]["angle"]) == {"mean", "median", "max", "flipped", "n_faces"}


# ---------------------------------------------------------------------------------------------------------
# KEPT NEGATIVE 2: the mean is unbounded
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_the_mean_quasi_conformal_ratio_is_unbounded():
    m = _stretched_cap(6.0)
    d = uv_angle_distortion(m, lscm(m))
    assert d["mean"] > 50.0                                       # measured 398.0
    assert d["median"] < 10.0                                     # measured 4.8
    assert d["max"] > 100.0 * d["median"]                         # one face dominates the mean entirely


# ---------------------------------------------------------------------------------------------------------
# KEPT NEGATIVE 3: no scalar summary can see a fold
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_a_better_mean_can_hide_a_worse_fold():
    # THE FINDING. isomap's mean ratio is 150x better than LSCM's here, and it folds nearly twice as many faces.
    m = _stretched_cap(6.0)
    rep = uv_report(m, methods=("lscm", "isomap"))
    assert rep["isomap"]["angle"]["mean"] < rep["lscm"]["angle"]["mean"]
    assert rep["isomap"]["angle"]["flipped"] > rep["lscm"]["angle"]["flipped"]
    assert rep["isomap"]["stretch"] < rep["lscm"]["stretch"]      # the stretch metric agrees with the mean ...
    assert rep["isomap"]["angle"]["flipped"] > 100                # ... and both are wrong about the map


def test_a_fold_is_a_minority_orientation_not_a_negative_determinant():
    # A globally MIRRORED chart has det J < 0 on every face and folds nothing. Counting the sign reported the
    # planar chart as 256/256 flipped on a hemisphere cap. It had folded none of them.
    m = hemisphere_cap(3)
    for meth in ("lscm", "isomap", "planar"):
        uv = lscm(m) if meth == "lscm" else uv_unwrap(m, method=meth)
        assert uv_angle_distortion(m, uv)["flipped"] == 0, meth

    # mirroring a fold-free chart must not create folds
    uv = lscm(m)
    mirrored = uv.copy()
    mirrored[:, 0] *= -1.0
    assert uv_angle_distortion(m, mirrored)["flipped"] == 0


def test_folds_appear_as_curvature_rises_and_are_counted():
    counts = [uv_angle_distortion(_stretched_cap(z), lscm(_stretched_cap(z)))["flipped"] for z in (1.0, 3.0, 6.0)]
    assert counts[0] == 0                                          # a mild cap unwraps cleanly
    assert counts == sorted(counts)                                # folds only accumulate with curvature
    assert counts[-1] > 0                                          # LSCM does NOT guarantee a fold-free map


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    mesh = flat_grid_mesh(6)
    uv = m.mesh_lscm(mesh)
    d = m.mesh_uv_angle_distortion(mesh, uv)
    assert abs(d["median"] - 1.0) < 1e-9 and d["flipped"] == 0
    assert m.mesh_uv_area_distortion(mesh, uv) < 1e-9

    rep = m.mesh_uv_report(hemisphere_cap(2), methods=("lscm", "isomap"))
    assert rep["lscm"]["angle"]["median"] < rep["isomap"]["angle"]["median"]

    assert np.array_equal(m.mesh_uv_unwrap(mesh, method="lscm"), uv_unwrap(mesh, method="lscm"))

    for phrase in ("least squares conformal maps", "does my uv map fold", "conformal map"):
        assert "Conformal UV" in str(m.find_capability(phrase)[:3]), phrase
