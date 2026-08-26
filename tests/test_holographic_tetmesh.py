"""Regression traps for F3 (Delaunay tetrahedralisation with PROVED topology).

The certificate is only worth having if it can FAIL, so the severed-limb refusal is pinned
as hard as the success path, and the measured minimum-attachment law is pinned with it.
"""
import numpy as np
import pytest

from holographic.mesh_and_geometry import holographic_tetmesh as T


def test_circumsphere_exact_and_degenerate():
    p = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    c, r2 = T.circumsphere(*p)
    assert np.allclose(c, [0.5, 0.5, 0.5]) and abs(r2 - 0.75) < 1e-12
    assert T.circumsphere(np.array([0., 0, 0]), np.array([1., 0, 0]),
                          np.array([2., 0, 0]), np.array([3., 0, 0]))[0] is None


def test_tets_tile_the_cube():
    """The volume check is the real correctness test: a wrong Bowyer-Watson can still return
    plausible-looking tets, but they will not sum to the cube's volume."""
    cube = np.array([[x, y, z] for x in (0., 1) for y in (0., 1) for z in (0., 1)])
    ct = T.delaunay_tets(cube)
    vol = sum(abs(np.linalg.det(np.array([cube[t[1]] - cube[t[0]], cube[t[2]] - cube[t[0]],
                                          cube[t[3]] - cube[t[0]]]))) / 6.0 for t in ct)
    assert abs(vol - 1.0) < 1e-9
    # a NON-degenerate 4-point set gives exactly one tet; cube[:4] would be coplanar
    # (all x=0) and correctly gives ZERO -- that distinction is the assertion worth having
    assert T.delaunay_tets(np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])).shape == (1, 4)
    assert len(T.delaunay_tets(cube[:4])) == 0


def test_aggregate_mesh_is_valid_and_certified():
    from holographic.simulation_and_physics.holographic_morphogen import grow_aggregate
    agg = grow_aggregate(n_cells=40, seed=0, steps=80)
    mesh = T.tetrahedralize(agg["positions"], agg["radii"])
    assert mesh["T"] > 0 and mesh["components"] == 1
    assert not mesh["nonmanifold_faces"]
    cert = T.connectivity_certificate(mesh, 0, list(range(mesh["T"])))
    assert cert["ok"] and cert["proofs"]
    # SECOND INSTRUMENT: the kernel's derivation must agree with an independent flood fill
    from holographic.simulation_and_physics.holographic_island import connected_components
    flood = set(connected_components(mesh["T"], mesh["adjacency"])[0])
    assert set(cert["connected"]) == flood
    assert np.array_equal(T.tetrahedralize(agg["positions"], agg["radii"])["tets"], mesh["tets"])


def test_severed_limb_is_refused():
    """A certificate that never fails certifies nothing."""
    from holographic.simulation_and_physics.holographic_morphogen import grow_aggregate
    from holographic.simulation_and_physics.holographic_island import connected_components
    agg = grow_aggregate(n_cells=40, seed=0, steps=80)
    far = np.vstack([agg["positions"], agg["positions"] + np.array([50.0, 0, 0])])
    mesh = T.tetrahedralize(far, np.concatenate([agg["radii"], agg["radii"]]))
    assert mesh["components"] >= 2
    comps = connected_components(mesh["T"], mesh["adjacency"])
    cert = T.connectivity_certificate(mesh, comps[0][0], [comps[-1][-1]])
    assert not cert["ok"] and cert["unreachable"]


def test_minimum_attachment_law():
    """MEASURED DESIGN LAW: an attachment 1-2 cells across is NOT volumetrically connected --
    collinear/coplanar points form no tets with volume, so no alpha setting can rescue it.
    Three cells across is the minimum. Pinned so creature generation relies on it rather
    than rediscovering it as a bug."""
    rng = np.random.default_rng(1)
    a = rng.normal(scale=1.1, size=(45, 3))
    b = rng.normal(scale=1.1, size=(45, 3)) + np.array([9.0, 0, 0])

    def waist(ring):
        ws = []
        for x in np.arange(2.4, 7.0, 1.0):
            if ring == 1:
                ws.append([x, 0, 0])
            else:
                ws.extend([[x, 0.55 * np.cos(2 * np.pi * k / ring),
                            0.55 * np.sin(2 * np.pi * k / ring)] for k in range(ring)])
        return np.array(ws)

    for ring, expect in ((2, False), (3, True)):
        pts = np.vstack([a, waist(ring), b])
        mesh = T.tetrahedralize(pts, np.full(len(pts), 0.5))
        ta = [i for i, t in enumerate(mesh["tets"]) if max(t) < 45]
        tb = [i for i, t in enumerate(mesh["tets"]) if min(t) >= len(pts) - 45]
        assert ta and tb
        assert T.connectivity_certificate(mesh, ta[0], [tb[0]])["ok"] is expect


def test_lean_export_of_a_certificate():
    """Emitting Lean needs no binary (Tier 0); when a binary exists it must accept."""
    from holographic.simulation_and_physics.holographic_morphogen import grow_aggregate
    from holographic.agents_and_reasoning.holographic_lean import lean_check
    agg = grow_aggregate(n_cells=30, seed=2, steps=60)
    mesh = T.tetrahedralize(agg["positions"], agg["radii"])
    src = T.certificate_lean(mesh, 0, mesh["T"] - 1)
    assert src is not None and "theorem limb_connected : conn t0" in src
    res = lean_check(src)
    if res["available"]:
        assert res["ok"], res["stderr"][:400]


def test_certified_lod_chain_and_storage_claim():
    """F5: LOD as a RULE. Every level re-tetrahedralised and re-certified; levels get
    strictly coarser; the nested-prefix property (what makes ONE ordering serve all levels)
    holds; and the storage claim is a measured ratio, not a slogan."""
    from holographic.simulation_and_physics.holographic_morphogen import grow_aggregate
    agg = grow_aggregate(n_cells=100, seed=0, steps=80)
    ch = T.lod_chain(agg["positions"], agg["radii"], fractions=(1.0, 0.5, 0.25))
    assert all(lv["ok"] for lv in ch["levels"]), [lv["reason"] for lv in ch["levels"]]
    sizes = [lv["mesh"]["T"] for lv in ch["levels"]]
    assert sizes == sorted(sizes, reverse=True)
    last = ch["levels"][-1]
    assert list(last["indices"]) == list(ch["ordering"][:last["n_points"]])
    assert T.lod_storage_cost(agg["positions"], ch)["ratio"] > 2.0


def test_lod_refuses_a_level_that_breaks_topology():
    """The refusal path is the point: a certificate that never fails certifies nothing.
    Two separated blobs cannot form one component, so every level must be REFUSED with a
    reason rather than shipped."""
    rng = np.random.default_rng(4)
    pts = np.vstack([rng.normal(scale=1.1, size=(40, 3)),
                     rng.normal(scale=1.1, size=(40, 3)) + np.array([40.0, 0, 0])])
    ch = T.lod_chain(pts, np.full(len(pts), 0.5), fractions=(1.0, 0.5))
    assert any((not lv["ok"]) and lv["reason"] for lv in ch["levels"])


def test_circumsphere_caching_is_bit_identical_and_fast():
    """LEVER 1 (bake once, sample O(1)) applied to Bowyer-Watson: a tet's circumsphere never
    changes after creation, but the loop used to re-solve a 3x3 system for EVERY tet on EVERY
    insertion. Measured 3.83s -> 0.44s at N=240 (8.7x). The pin is CORRECTNESS, not speed --
    an optimisation that changes one tet is not an optimisation -- so this reproduces the
    unbaked loop from scratch and demands equality."""
    import time
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(120, 3))

    def unbaked(points):
        p = np.asarray(points, float)
        n = len(p)
        c = p.mean(axis=0)
        r = float(np.linalg.norm(p - c, axis=1).max()) + 1.0
        big = 8.0 * r
        sup = np.array([c + [big, 0, 0], c + [-big, big, 0],
                        c + [-big, -big, big], c + [-big, -big, -big]], float)
        allp = np.vstack([p, sup])
        tets = [(n, n + 1, n + 2, n + 3)]
        for idx in range(n):
            q = allp[idx]
            bad = []
            for t in tets:
                cc, r2 = T.circumsphere(allp[t[0]], allp[t[1]], allp[t[2]], allp[t[3]])
                if cc is not None and (q - cc) @ (q - cc) <= r2 * (1.0 + 1e-12):
                    bad.append(t)
            if not bad:
                continue
            cnt = {}
            for t in bad:
                for f in T._faces(t):
                    cnt[f] = cnt.get(f, 0) + 1
            bs = set(bad)
            tets = [t for t in tets if t not in bs]
            for f in sorted([f for f, k in cnt.items() if k == 1]):
                tets.append((f[0], f[1], f[2], idx))
        out = np.array(sorted([t for t in tets if max(t) < n]), int).reshape(-1, 4)
        if len(out):
            d = np.stack([p[out[:, 1]] - p[out[:, 0]], p[out[:, 2]] - p[out[:, 0]],
                          p[out[:, 3]] - p[out[:, 0]]], axis=2)
            neg = np.linalg.det(d) < 0
            out[neg] = out[neg][:, [0, 1, 3, 2]]
        return out

    t0 = time.time()
    fast = T.delaunay_tets(pts)
    dt = time.time() - t0
    assert np.array_equal(fast, unbaked(pts))
    assert dt < 2.0, "120 points should be well under a second post-bake, got %.2fs" % dt
