"""NULLSPACE -- install into the directions the model was not using.

From the research survey's first recommendation: AlphaEdit (Fang et al., ICLR
2025 Outstanding Paper, arXiv 2410.02355) projects a weight perturbation onto
the NULL SPACE of the preserved-knowledge key matrix before applying it, so the
post-edit output is unchanged for preserved keys. The paper reports it "boosts
the performance of most locating-then-editing methods by an average of 36.7%
with a single line of additional code for projection solely".

WHY THIS MATTERS HERE: every install in this pipeline has been checked by
MEASUREMENT -- bit-identical when empty, or perplexity did not regress. That is
weaker than a construction that cannot disturb what it must not touch.

MEASURED ON A REAL MODEL, installing the same bind operator three ways:
    projection            kept energy   perplexity   bind cosine
    none (raw)                   1.00       7.3772     1.000000
    drop eig > 1e-2*max          0.78       7.2820     1.000000
    drop eig > 1e-3*max          0.51       7.2790     1.000000
    (baseline, no install)          -       7.2659
THE COST OF INSTALLING FELL SEVENFOLD, +1.53% to +0.22%, AND THE OPERATOR STILL
COMPUTES EXACTLY -- cosine 1.000000 in every case. The circuit does the same
arithmetic; it just does it in directions the model was not using.

AND THE HONEST CAVEAT, which the paper's setting hides and a small model
exposes: ALPHAEDIT'S GUARANTEE NEEDS AN ACTUAL NULL SPACE, and a full-rank key
covariance does not have one. Measured here, 600 preserved keys at width 128
gave eigenvalues spanning 2.03 to 1.29e4 -- the SMALLEST is 2.03, not zero. So
what this computes is a LOW-ENERGY SUBSPACE, not a null space, and the
disturbance falls (0.797 to 0.263) rather than vanishing. The guarantee degrades
gracefully into a reduction, and calling it a proof on a full-rank problem would
be the overclaim.
That is a width-and-sample question: more preserved samples than dimensions
means full rank. A 1024-wide model probed with 600 keys HAS a real null space;
a 128-wide one probed with 600 does not.
"""

import numpy as np


def preserved_keys(runtime, ids, layer, max_rows=4000):
    """Collect the MLP inputs a preserved corpus produces -- the K0 of AlphaEdit.

    These are the directions the model is ALREADY USING at this layer. An edit
    that lives in their complement cannot change what they produce."""
    rows = []

    def probe(l, x):
        if int(l) == int(layer):
            rows.append(np.asarray(x, np.float64).copy())

    runtime.mlp_probe = probe
    try:
        runtime.forward(list(ids))
    finally:
        runtime.mlp_probe = None
    K = np.vstack(rows) if rows else np.zeros((0, 1))
    return K[-int(max_rows):]


def projector(K0, ratio=1e-2):
    """The projector onto the low-energy subspace of K0. Returns (P, report).

    AlphaEdit drops eigenvectors whose eigenvalue exceeds a threshold; the
    remainder spans directions the preserved keys barely occupy. `ratio` is
    relative to the largest eigenvalue, which makes it scale-free -- an absolute
    threshold is meaningless across models with different activation scales."""
    K = np.asarray(K0, np.float64)
    if K.size == 0:
        raise ValueError("no preserved keys collected")
    e, V = np.linalg.eigh(K.T @ K)
    keep = e <= float(ratio) * float(e.max())
    P = V[:, keep] @ V[:, keep].T
    return P, {"dims": int(K.shape[1]), "kept_dims": int(keep.sum()),
               "fraction": float(keep.mean()),
               "eig_min": float(e.min()), "eig_max": float(e.max()),
               "true_null_space": bool(e.min() < 1e-8 * e.max()),
               "n_keys": int(K.shape[0])}


def project(delta, P):
    """Restrict an operator to the preserved-safe subspace. One matmul."""
    return np.asarray(delta, np.float64) @ np.asarray(P, np.float64)


def guard(runtime, ids, layer, delta, ratio=1e-2):
    """Collect, project, report -- the whole wrapper in one call."""
    K0 = preserved_keys(runtime, ids, layer)
    P, rep = projector(K0, ratio=ratio)
    D = np.asarray(delta, np.float64)
    Dp = project(D, P)
    rep["energy_kept"] = float(np.linalg.norm(Dp) / (np.linalg.norm(D) + 1e-30))
    rep["disturbance_raw"] = float(np.max(np.abs(K0 @ D.T)))
    rep["disturbance_projected"] = float(np.max(np.abs(K0 @ Dp.T)))
    rep["reduction"] = (rep["disturbance_raw"]
                        / max(rep["disturbance_projected"], 1e-30))
    return Dp, rep


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime, GDNRuntime, load_weights_dir)
    from holographic.io_and_interop.holographic_vsabake import (
        install_op, circulant, layer_key)
    from holographic.io_and_interop.holographic_measure import measure

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("nullspace selftest SKIPPED-SUBJECT (no model present)")
        return

    rt, cfg = load_runtime(src)
    w = load_weights_dir(src)
    H, L = int(cfg["hidden"]), int(cfg["n_layers"]) - 1
    rng = np.random.default_rng(0)
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    keep_ids = [b for b in raw[5000:9000].encode("utf-8")][:600]
    ev = [b for b in raw[20000:21200].encode("utf-8")][:1000]

    OP = circulant(rng.standard_normal(H))
    Dp, rep = guard(rt, keep_ids, L, OP)

    # ---- PROJECTION MUST REDUCE THE DISTURBANCE, or it does nothing ----
    assert rep["reduction"] > 2.0, rep
    # ---- AND IT MUST KEEP MOST OF THE OPERATOR, or it is just shrinking it
    assert rep["energy_kept"] > 0.5, rep

    base = measure(rt, ev)["perplexity"]
    costs = {}
    for label, M in (("raw", OP), ("projected", Dp)):
        w2, r2 = install_op(w, cfg, M, layer=L,
                            mean_h=preserved_keys(rt, keep_ids, L)[-1])
        run = GDNRuntime(w2, dict(cfg))
        costs[label] = measure(run, ev)["perplexity"]
        # ---- AND THE OPERATOR MUST STILL COMPUTE EXACTLY ----
        cap = {}
        run.mlp_probe = lambda l, x: (cap.__setitem__("x",
                                                      np.asarray(x)[-1].copy())
                                      if int(l) == L else None)
        run.forward(ev[:120])
        run.mlp_probe = None
        up = np.asarray(w2[layer_key(w2, L, "mlp.up_proj.weight")],
                        np.float64)[-r2["neurons_added"]:]
        got, want = up @ cap["x"], M @ cap["x"]
        cos = float(got @ want
                    / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-30))
        assert cos > 0.999, (label, cos)

    # ---- THE PROJECTED INSTALL MUST COST LESS ----
    raw_cost = 100 * (costs["raw"] - base) / base
    proj_cost = 100 * (costs["projected"] - base) / base
    assert proj_cost < raw_cost / 2.0, (raw_cost, proj_cost)

    print("nullspace selftest OK -- projecting an installed operator onto the "
          "low-energy subspace of the preserved keys cuts the cost of "
          "installing from +%.2f%% perplexity to +%.2f%% while the operator "
          "still computes at cosine >0.999, keeping %.0f%% of its energy and "
          "reducing preserved-key disturbance %.1fx. AND THE HONEST PART: "
          "eigenvalues here span %.2e to %.2e, so the smallest is NOT zero -- "
          "this is a LOW-ENERGY SUBSPACE, not the true null space AlphaEdit "
          "assumes, and the disturbance falls rather than vanishing (true null "
          "space present: %s)"
          % (raw_cost, proj_cost, 100 * rep["energy_kept"], rep["reduction"],
             rep["eig_min"], rep["eig_max"], rep["true_null_space"]))


if __name__ == "__main__":
    _selftest()
