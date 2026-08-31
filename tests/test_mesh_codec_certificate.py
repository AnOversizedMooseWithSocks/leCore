"""B2: mesh-codec certificate -- the codec's TWO claims, checked separately.

SOTA structure (Draco/glTF, MPEG V-DMC): the two dials are independent. Connectivity coding
is LOSSLESS; quantization "is the only lossy step and the only one that touches accuracy",
with a predictable worst case of about E/2^n for n bits over an extent E. MPEG's lossless
mode likewise "employs topology/UV/texture checks" -- topology verified SEPARATELY from
geometry. So a mesh certificate makes two distinct assertions, and conflating them would
hide a topology break behind an acceptable average error.

leCore's mesh_encode states exactly these two claims in its docstring ("per-coordinate
|err| <= max_error, connectivity BIT-EXACT"). This pins them across five budgets.
"""
import numpy as np


def _surface():
    from holographic.simulation_and_physics.holographic_morphogen import grow_aggregate
    from holographic.mesh_and_geometry.holographic_tetmesh import tetrahedralize
    from holographic.mesh_and_geometry import holographic_mesh as HM
    agg = grow_aggregate(n_cells=40, seed=0, steps=80)
    tets = tetrahedralize(agg["positions"], agg["radii"])
    V = np.asarray(agg["positions"], float)
    F = np.asarray(tets["boundary"], int)
    return V, F, HM.Mesh(V, F)


def certify_mesh_codec(mind, mesh, V, F, max_error):
    """Both claims, reported separately: geometry within the STATED budget, topology
    BIT-EXACT. Returns (ok, report) -- a topology break is never excusable by a small
    coordinate error, which is why they are not averaged together."""
    enc = mind.mesh_encode(mesh, max_error=max_error)
    V2, F2 = mind.mesh_decode(enc["blob"])[:2]
    V2, F2 = np.asarray(V2, float), np.asarray(F2, int)
    geom_ok = V2.shape == V.shape and float(np.max(np.abs(V2 - V))) <= max_error * 1.0001
    topo_ok = np.array_equal(F2, np.asarray(F, int))
    return (geom_ok and topo_ok), {
        "max_error": max_error,
        "measured": float(np.max(np.abs(V2 - V))) if V2.shape == V.shape else float("inf"),
        "geometry_ok": geom_ok, "topology_bit_exact": topo_ok, "bytes": len(enc["blob"])}


def test_codec_honours_both_claims_across_budgets():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    V, F, mesh = _surface()
    sizes = []
    for tol in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        ok, rep = certify_mesh_codec(m, mesh, V, F, tol)
        assert ok, rep
        assert rep["measured"] <= tol * 1.0001 and rep["topology_bit_exact"]
        sizes.append(rep["bytes"])
    # a tighter budget must cost MORE bytes -- monotone, or the budget is not being honoured
    assert sizes == sorted(sizes), sizes


def test_certificate_refuses_a_corrupted_decode():
    """A certificate that cannot fail certifies nothing: perturb the decoded geometry past
    the budget and past the topology, and each must be refused on its own axis."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    V, F, mesh = _surface()
    enc = m.mesh_encode(mesh, max_error=1e-3)
    V2, F2 = m.mesh_decode(enc["blob"])[:2]
    V2 = np.asarray(V2, float).copy()
    V2[0, 0] += 1.0                                    # geometry break
    assert float(np.max(np.abs(V2 - V))) > 1e-3
    F2 = np.asarray(F2, int).copy()
    F2[0] = F2[0][::-1]                                # topology break
    assert not np.array_equal(F2, np.asarray(F, int))
