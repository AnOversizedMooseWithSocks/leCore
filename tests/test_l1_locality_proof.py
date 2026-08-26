"""L1: the locality theorem, PROVED in Lean rather than sampled.

holographic_blendbasis MEASURED zero overreach on one mesh with a few correctives. That is
evidence, not a guarantee -- STAR needed a scan dataset to obtain the same property, so
claiming it by construction deserves a proof rather than a sample. lean/LeCoreLocality.lean
proves it for ALL distances, radii and amplitudes, including the composition over a stack.

These tests do two things a proof cannot: check the proof still typechecks (Tier 1), and
check that the PYTHON implementation actually computes what the Lean file models -- a proof
about a different function would be worthless.
"""
import os
import shutil
import subprocess
import numpy as np
import pytest

LEAN_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "lean", "LeCoreLocality.lean")


def test_proof_file_exists_and_has_no_sorry():
    """Tier 0: no Lean binary needed. A file full of `sorry` typechecks and proves nothing,
    so the absence of admitted goals is itself part of the claim."""
    assert os.path.exists(LEAN_FILE)
    src = open(LEAN_FILE, encoding="utf-8").read()
    assert "sorry" not in src
    for thm in ("weight_zero_outside", "disp_zero_outside", "stack_zero_outside",
                "clip_bounds"):
        assert thm in src, thm


@pytest.mark.skipif(shutil.which("lean") is None, reason="Lean not installed (Tier 0)")
def test_lean_typechecks_the_locality_theorems():
    """Tier 1: hand the file to the external authority and require silence."""
    r = subprocess.run(["lean", LEAN_FILE], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not r.stdout.strip(), r.stdout


def test_python_implements_what_lean_proves():
    """The bridge, and the part that makes the proof MEAN anything here: the shipped NumPy
    falloff must agree with the Lean model. Both are clip -> smoothstep, so both must be
    exactly zero at and beyond the radius and exactly one at the anchor."""
    from lecore import UnifiedMind
    from holographic.mesh_and_geometry.holographic_blendbasis import support_weights
    m = UnifiedMind(dim=64, seed=0)
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    mesh = m.mesh_from_sdf(sph, ((-1.3,) * 3, (1.3,) * 3), res=20, vectorized=True)
    V = np.asarray(mesh.vertices, float)
    src = int(np.argmax(V[:, 1]))
    r = 0.7
    w = support_weights(mesh, src, r, m)
    d = np.asarray(m.mesh_geodesic(mesh, src), float)
    # L1 (weight_zero_outside): EXACTLY zero at and beyond the radius -- not merely small
    assert np.all(w[d >= r] == 0.0)
    # clip_bounds: never inverts, never overshoots
    assert w.min() >= 0.0 and w.max() <= 1.0
    # full weight at the anchor
    assert abs(w[src] - 1.0) < 1e-12


def test_composition_matches_the_stack_theorem():
    """stack_zero_outside: a whole stack vanishes where every corrective is out of support.
    Verified on the real basis, since a rig runs dozens of correctives at once."""
    from lecore import UnifiedMind
    from holographic.mesh_and_geometry.holographic_blendbasis import make_corrective
    m = UnifiedMind(dim=64, seed=0)
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    mesh = m.mesh_from_sdf(sph, ((-1.3,) * 3, (1.3,) * 3), res=20, vectorized=True)
    V = np.asarray(mesh.vertices, float)
    top, bot = int(np.argmax(V[:, 1])), int(np.argmin(V[:, 1]))
    ts = [make_corrective(mesh, s, 0.5, "normal", 0.3, m) for s in (top, bot)]
    # EXTRAPOLATED weights, which is where a weaker guarantee would break
    mixed = np.asarray(m.blend_shapes(V, ts, [1.9, -0.8]), float)
    d_top = np.asarray(m.mesh_geodesic(mesh, top), float)
    d_bot = np.asarray(m.mesh_geodesic(mesh, bot), float)
    outside = (d_top >= 0.5) & (d_bot >= 0.5)
    assert outside.any(), "test vacuous: every vertex was inside some support"
    assert np.allclose(mixed[outside], V[outside], atol=0.0, rtol=0.0)
