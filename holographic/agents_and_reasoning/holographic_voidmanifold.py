"""VOID MANIFOLD -- regions a model's activations never visit, found honestly.

WHAT A VOID IS HERE: a point INSIDE the support of the model's own activation
manifold (a convex combination of states it actually produced) that is
nevertheless far from every state it has ever produced. Not extrapolation --
extrapolating outside the data is trivial and means nothing. A void is a hole
the model leaves in the middle of its own territory.

THE EXPERIMENTAL PATH, kept because it is the result:

  1. leCore's existing mind.void_map (bootstrap-null-gated density voids) found
     ZERO voids on the activation manifold AND zero on data with a KNOWN
     PLANTED HOLE. Diagnosed rather than assumed: inside the planted hole the
     reported z was LOWER than outside (5.07 vs 6.49 at r=0.28; 18.36 vs 28.75
     at r=0.45) -- the drift model's smooth kernel fills the hole in. That is
     the limitation its own docstring warns about ("the sampler's smooth kernel
     smears absence"), now measured concretely. KEPT NEGATIVE: void_map is the
     right instrument for corpus-scale density voids and the wrong one for
     activation manifolds.

  2. This detector replaces the density model with the data's OWN spacing:
     probes are convex combinations of real points (so they are inside the
     support by construction), scored by nearest-neighbour distance against the
     distribution of nearest-neighbour distances among the data itself.

  3. VALIDATED on planted holes before being trusted on anything real:
        no hole   -> 0 voids (zero false positives)
        r = 0.20  -> 18 voids, 100% inside the planted hole
        r = 0.28  -> 81 voids, 100% inside
        r = 0.40  -> 199 voids, 100% inside
     and split-half: held-out data stays 3.4x further from the discovered voids
     than a typical point does. A void found on one sample is still empty in
     another.

  4. SURROGATE CONTROL, because void COUNT is dimension-confounded (probes in
     higher dimensions land far from everything for trivial reasons): the same
     detector runs on a matched-covariance Gaussian. On the reference model's
     layer-2 manifold the real void fraction was 0.032 / 0.152 / 0.287 / 0.643
     at 2 / 3 / 4 / 6 PCs while the surrogate gave 0.000 at every dimension.

HOW TO READ THAT LAST RESULT, honestly: a Gaussian is unimodal, activations are
CLUSTERED, and the excess is the space BETWEEN CLUSTERS. That is genuine
structure (the surrogate has none) and it is exactly what "the model never goes
here" means -- but it is not evidence of anything semantic. On a random-weight
model the clusters are per-prompt artifacts. Whether a TRAINED model's voids
correspond to concepts it cannot represent is the open question this instrument
makes ASKABLE; it does not answer it, and nothing here should be quoted as if
it did.
"""

import numpy as np


def _spacing(X):
    """Nearest-neighbour distance for every point -- the data's own scale."""
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    return D.min(1)


def manifold_voids(points, n_probes=800, mix=3, q=0.999, seed=1,
                   surrogate_trials=5):
    """Find voids inside a point cloud's own support, with the surrogate control
    that makes the count meaningful.

    mix: how many real points each probe is a convex combination of. 2 probes
    the segments between states; 3+ probes the interior of their simplices.
    q: the spacing quantile a probe must exceed to count as void -- so the
    threshold is set by the data, never by a magic radius.

    Returns {voids, void_fraction, threshold, surrogate_fraction,
    surrogate_sd, excess, verdict}. The verdict is deliberately conservative:
    structure is claimed only when the real fraction exceeds the surrogate by
    more than 3 surrogate standard deviations."""
    X = np.asarray(points, np.float64)
    if X.ndim != 2 or len(X) < 8:
        raise ValueError("need at least 8 points in a 2-D array")
    rng = np.random.default_rng(int(seed))

    def _run(Y, rs):
        idx = rs.integers(0, len(Y), size=(int(n_probes), int(mix)))
        wgt = rs.dirichlet(np.ones(int(mix)), size=int(n_probes))
        P = np.einsum("pm,pmd->pd", wgt, Y[idx])
        thr = float(np.quantile(_spacing(Y), float(q)))
        dP = np.array([np.min(np.linalg.norm(Y - p, axis=1)) for p in P])
        hit = dP > thr
        return P[hit], dP[hit], thr, float(np.mean(hit))

    V, dV, thr, frac = _run(X, np.random.default_rng(int(seed)))

    sur = []
    if surrogate_trials:
        C = np.cov(X.T) + 1e-12 * np.eye(X.shape[1])
        L = np.linalg.cholesky(C)
        for s in range(int(surrogate_trials)):
            rs = np.random.default_rng(1000 + s)
            Y = rs.standard_normal(X.shape) @ L.T
            sur.append(_run(Y, np.random.default_rng(int(seed)))[3])
    sur = np.asarray(sur) if sur else np.array([0.0])
    excess = frac - float(sur.mean())
    structured = excess > 3.0 * max(float(sur.std()), 1e-6)
    return {"voids": V, "distances": dV, "threshold": thr,
            "void_fraction": frac, "surrogate_fraction": float(sur.mean()),
            "surrogate_sd": float(sur.std()), "excess": float(excess),
            "verdict": ("structured" if structured else
                        "no excess over a matched-covariance surrogate -- the "
                        "count is explained by dimensionality, not structure"),
            "note": "voids are BETWEEN-CLUSTER gaps; that is structure, not "
                    "semantics. Whether they mean anything is a question about "
                    "the model, answered only by decoding them."}


def void_probe(runtime, layer, basis, mean, void_points, token_ids, hooks=None):
    """DECODE a void: what would the model say from a state it never occupies?

    Reconstructs each void point back into the full hidden space (basis is the
    PCA basis the voids were found in), substitutes it at `layer` for the final
    position, and returns the resulting next-token distribution.

    This is the mechanism behind "explore where the model has never been". It is
    honest about being a mechanism: on a trained model these distributions are
    worth reading, on a random one they are noise, and NOTHING here scores
    novelty or soundness. Returns a list of {point, top_tokens, entropy}."""
    B = np.asarray(basis, np.float64)
    mu = np.asarray(mean, np.float64)
    out = []
    for p in np.atleast_2d(np.asarray(void_points, np.float64)):
        full = mu + p @ B

        def hook(h, _v=full):
            d = np.zeros_like(h)
            d[-1] = _v - h[-1]              # replace the last position's state
            return d

        hk = dict(hooks or {})
        hk[int(layer)] = hook
        lg = runtime.forward(token_ids, hooks=hk)[-1]
        z = lg - lg.max()
        pr = np.exp(z)
        pr /= pr.sum()
        top = np.argsort(pr)[-5:][::-1]
        out.append({"point": p,
                    "top_tokens": [(int(t), float(pr[t])) for t in top],
                    "entropy": float(-np.sum(pr * np.log(pr + 1e-30)))})
    return out


def _selftest():
    rng = np.random.default_rng(0)
    c = np.array([0.6, 0.6, 0.5])

    def make(hole_r, n=1200, seed=0):
        r = np.random.default_rng(seed)
        X = r.uniform(0, 1, size=(n * 3, 3))
        if hole_r > 0:
            X = X[np.linalg.norm(X - c, axis=1) > hole_r]
        return X[:n]

    # 1) NO HOLE -> no voids. A detector that fires on uniform data is useless.
    r0 = manifold_voids(make(0.0), n_probes=600, surrogate_trials=3)
    assert len(r0["voids"]) == 0, len(r0["voids"])

    # 2) PLANTED HOLE -> voids, and ALL of them inside the hole. Detecting
    #    "some sparsity somewhere" would not be evidence of anything.
    for hole_r in (0.20, 0.28, 0.40):
        r1 = manifold_voids(make(hole_r), n_probes=600, surrogate_trials=3)
        V = r1["voids"]
        assert len(V) > 0, hole_r
        inside = np.linalg.norm(V - c, axis=1) < hole_r
        assert inside.all(), (hole_r, float(inside.mean()))

    # 3) SPLIT-HALF: a void found on one sample must still be empty in another,
    #    or it was undersampling wearing a discovery's clothes.
    Xa, Xb = make(0.28, seed=7), make(0.28, seed=8)
    ra = manifold_voids(Xa, n_probes=600, surrogate_trials=0)
    V = ra["voids"]
    dheld = np.array([np.min(np.linalg.norm(Xb - p, axis=1)) for p in V])
    typ = np.array([np.min(np.linalg.norm(Xb - p, axis=1)) for p in Xa[:len(V)]])
    assert dheld.mean() > 3 * typ.mean(), (dheld.mean(), typ.mean())

    # 4) SURROGATE CONTROL fires the right way: clustered data reads structured,
    #    a plain Gaussian does not (the count alone is dimension-confounded).
    clusters = np.vstack([rng.standard_normal((200, 3)) * 0.05 + o
                          for o in ([0, 0, 0], [1, 0, 0], [0, 1, 1])])
    rc = manifold_voids(clusters, n_probes=600, surrogate_trials=5)
    assert rc["verdict"] == "structured", rc["verdict"]
    gauss = rng.standard_normal((600, 3))
    rg = manifold_voids(gauss, n_probes=600, surrogate_trials=5)
    assert rg["verdict"] != "structured", rg

    print("voidmanifold selftest OK -- 0 voids on uniform data; 100%% of voids "
          "inside the planted hole at r=0.20/0.28/0.40; split-half holds "
          "(%.1fx); clustered data reads structured (excess %+.3f) while a "
          "matched Gaussian does not (excess %+.3f)"
          % (dheld.mean() / typ.mean(), rc["excess"], rg["excess"]))


if __name__ == "__main__":
    _selftest()
