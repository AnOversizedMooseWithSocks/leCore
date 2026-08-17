"""Fixed-topology TEMPLATE WRAPPING: one mesh topology, many bodies.

BACKLOG O1 -- the keystone of the creature/humanoid overhaul. Today every creature meshes
from scratch, so vertex 400 means nothing across two creatures. That single fact is why there
are no blendshapes, no shared textures, no cross-species morphing and no correspondence: all
of them need vertex i to be the SAME anatomical point on every body.

SOTA CHECK (searched 2026-08-16): the standard is NON-RIGID ICP (Amberg, Romdhani & Vetter
2007, "Optimal step nonrigid ICP"), which assigns a locally affine transform per vertex,
penalises differences between neighbours, and "loops over a series of DECREASING STIFFNESS
weights that results in incremental deformation of the template surface towards the target".
Recent work refines the regulariser (conformal, curvature-consistent) or the template choice
(Variable Shared Template, TOG 2025), but the stiffness-annealed loop is unchanged. The
stated payoff is exactly ours: shared point-to-point correspondence "enables construction of
probabilistic shape models, texture transfer, and seamless shape blending".

WHY OURS IS BETTER CONDITIONED THAN N-ICP, and it is worth being precise rather than
claiming a general improvement: N-ICP must ESTIMATE correspondence by nearest-point search
against a noisy scan, and that search is the fragile step. Our target is an ANALYTIC SDF, so
the correspondence is not estimated at all -- the signed distance gives the exact offset and
its gradient gives the exact direction. We replace ICP's inner search with a Newton step onto
the zero level set. That removes the failure mode; it does not make us better at the problem
N-ICP actually solves (fitting real scans), and this module does not claim to.

THE ANNEAL IS KEPT, and from the same reasoning as F1's soft-then-inflate: projecting every
vertex straight onto the surface in one step bunches them wherever the target is concave, and
bunched vertices are exactly what destroys correspondence quality. So each round projects
PARTWAY (step size rising as stiffness falls) and relaxes tangentially in between.

RULE-0 AUDIT (2026-08-16): no wrap/retopology-to-fixed-topology faculty exists. REUSED and
not rebuilt -- mesh_from_sdf (builds the template once), mesh_smooth (Taubin lambda|mu, which
is NO-SHRINK; ordinary Laplacian smoothing would deflate the body a little every round and
silently shrink the wrap), and the field's own gradient for normals.

KEPT NEGATIVE: a wrap is only valid where the template and target are the same TOPOLOGY. Wrap
a biped template onto a snake and vertices will pile into the missing limbs -- the result has
correct connectivity and meaningless correspondence. wrap_quality reports the bunching so
that failure is visible rather than silent; it is not prevented, because preventing it needs
the genus check the caller should have done.
"""

import numpy as np


def field_normal(field, P, eps=1e-4):
    """Central-difference gradient of a scalar field, normalised.

    Finite difference rather than exact_sdf_normal because a wrapped target is usually a
    composed/meshed field with no symbolic form; exact_sdf_normal is the better choice when
    the caller HAS the expression, and the two agree to O(eps^2)."""
    P = np.asarray(P, float)
    g = np.empty_like(P)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        g[:, k] = (field(P + d) - field(P - d)) / (2 * eps)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-12)


def wrap_to_field(vertices, faces, field, rounds=6, step0=0.35, step1=1.0,
                  smooth_iters=6, level=0.0, mind=None):
    """Wrap a template mesh onto a target field, KEEPING ITS TOPOLOGY EXACTLY.

    `field(P) -> (N,)` signed values, negative inside. Returns new vertices; `faces` is
    returned unchanged by construction, which is the whole point of the exercise.

    Follows Amberg's annealed schedule: `rounds` passes with the projection step rising from
    `step0` to `step1`, tangential relaxation (Taubin, no-shrink) between passes and never
    after the last, so the final vertices sit ON the surface rather than smoothed off it."""
    V = np.asarray(vertices, float).copy()
    F = np.asarray(faces, int)
    for r in range(int(rounds)):
        t = r / max(int(rounds) - 1, 1)
        step = float(step0) + (float(step1) - float(step0)) * t
        for _ in range(2):                       # two Newton steps per round
            d = np.asarray(field(V), float).ravel() - float(level)
            V = V - step * d[:, None] * field_normal(field, V)
        if r < int(rounds) - 1 and smooth_iters and mind is not None:
            from holographic.mesh_and_geometry import holographic_mesh as _HM
            sm = mind.mesh_smooth(_HM.Mesh(V, F), lam=0.5, mu=-0.53,
                                  iters=int(smooth_iters))
            V = np.asarray(sm.vertices, float)
    return V


def wrap_quality(vertices, faces, field, level=0.0):
    """Did the wrap actually land, and did it stay a usable mesh?

    Three numbers, because one would hide the interesting failure:
      * surface_error  -- max |field| over vertices; is it ON the target at all
      * edge_ratio     -- longest/shortest edge; BUNCHING, the failure mode that quietly
                          ruins correspondence while surface_error stays small
      * flipped        -- faces whose normal opposes the field gradient, i.e. local
                          self-intersection from over-projection"""
    V = np.asarray(vertices, float)
    F = np.asarray(faces, int)
    err = float(np.max(np.abs(np.asarray(field(V), float) - float(level))))
    e = np.concatenate([np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1),
                        np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1),
                        np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1)])
    # ROBUST ratio, and the first version was NOT. max/min is destroyed by a SINGLE
    # degenerate edge: it read 59,000,000 on a mesh that was visually fine apart from a
    # handful of slivers, which says nothing about the bunching it was supposed to measure.
    # p95/p5 describes the bulk; degenerate edges are counted SEPARATELY, because "mostly
    # even with 3 slivers" and "uniformly terrible" are different diagnoses and one number
    # cannot carry both.
    med = float(np.median(e))
    ratio = float(np.percentile(e, 95) / max(np.percentile(e, 5), 1e-12))
    degenerate = int(np.sum(e < 0.02 * med))
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    cen = V[F].mean(axis=1)
    gn = field_normal(field, cen)
    flipped = int(np.sum(np.sum(fn * gn, axis=1) < 0))
    return {"surface_error": err, "edge_ratio": ratio, "degenerate_edges": degenerate,
            "flipped_faces": flipped, "n_vertices": int(len(V)), "n_faces": int(len(F))}


def _selftest():
    """Regression trap: wrapping a sphere template onto an ellipsoid must LAND on it, keep
    the face array bit-identical, and not flip a single face."""
    from holographic.mesh_and_geometry import holographic_mesh as _HM
    import lecore
    mind = lecore.UnifiedMind(dim=64, seed=0)
    sphere = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    tmpl = mind.mesh_from_sdf(sphere, ((-1.4, -1.4, -1.4), (1.4, 1.4, 1.4)), res=32,
                              vectorized=True)
    V0 = np.asarray(tmpl.vertices, float)
    F0 = np.asarray(tmpl.faces, int)
    ax = np.array([1.35, 0.75, 1.0])
    ell = lambda P: (np.linalg.norm(np.asarray(P, float) / ax, axis=1) - 1.0) * ax.min()
    V1 = wrap_to_field(V0, F0, ell, rounds=6, mind=mind)
    q = wrap_quality(V1, F0, ell)
    assert q["surface_error"] < 0.02, q
    assert q["flipped_faces"] == 0, q
    # The honest claim is RELATIVE, not absolute: a marching-cubes template starts with
    # poor triangles (p95/p5 ~ 67, ~1000 degenerate edges) and the wrap IMPROVES them,
    # because Taubin relaxation between projection rounds evens them out. Asserting an
    # absolute bar here would encode the CVT template's number on a mesh that never had it.
    q0 = wrap_quality(V0, F0, sphere)
    assert q["edge_ratio"] < q0["edge_ratio"], (q0, q)
    assert q["degenerate_edges"] < q0["degenerate_edges"], (q0, q)
    assert len(V1) == len(V0)
    # THE POINT OF O1: topology is untouched, so vertex i corresponds across bodies
    print("OK: holographic_templatewrap -- %d verts wrapped onto a new body, faces "
          "IDENTICAL, surface error %.4f, edge ratio %.1f -> %.1f (IMPROVED), 0 flipped"
          % (len(V1), q["surface_error"], q0["edge_ratio"], q["edge_ratio"]))


if __name__ == "__main__":
    _selftest()
