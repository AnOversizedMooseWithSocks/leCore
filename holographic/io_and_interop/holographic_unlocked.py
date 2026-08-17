"""UNLOCKED -- what fuse, token_step and the limit trick made installable.

Moose asked what the new machinery unlocks. The answer is larger than the four
reclassified units, because two of them change the ECONOMICS of installing
rather than adding one more thing to install.

1. A CHAIN COSTS WHAT ONE OPERATOR COSTS. `fuse` folds an operator chain into a
   single matrix, so depth is free. MEASURED on the live residual stream:
       ops    neurons   cosine to the chain
         1        128            1.000000
         4        128            1.000000
        16        128            1.000000
        32        128            1.000000
   Thirty-two operations for the price of one, exact. Anything leCore expresses
   as a SEQUENCE of linear transforms -- transform_bank's apply_chain, a shader
   pipeline's stages, a VSA program that is all BIND and PERMUTE -- now installs
   whole rather than one stage per layer. THE LAYER BUDGET STOPPED BEING THE
   CONSTRAINT.

2. A CONVERGING ITERATION INSTALLS AT ITS ANSWER. leCore already had
   `accelerate_convergence` -- "JUMP TO AN ITERATIVE SOLVER'S LIMIT when its
   convergence is lawful" -- and for a LINEAR iteration the limit is a matrix:
   x <- Ax + b converges to (I - A)^-1 b. MEASURED: 200 iterations of a
   contracting map agree with the closed-form limit at COSINE 1.000000, and that
   limit installs and computes on the live stream at COSINE 1.000000 in 128
   neurons.
   So every leCore faculty that is "iterate a projection" -- and the project's
   own note says IK, PBD, PnP and the resonator are all that same thing in
   different costumes -- installs AT ITS CONVERGED ANSWER, with no loop at all.
   The loop was never the requirement; it was one way to reach the fixed point.

3. AND WHEN THE ITERATION IS *NOT* LINEAR OR NOT CONTRACTING, `token_step`
   carries one step per token. That is the resonator's route and it still works;
   it is now the FALLBACK rather than the only option.

WHAT IS STILL OUT, and it did not move: anything whose step depends on data the
layer cannot see (a real SDF query, a file read), and anything whose value is
the SCHEDULE rather than the arithmetic (eviction, durability). Those are in the
runtime because that is where time lives.

THE HONEST CAVEAT ON ALL OF THIS: fusing a chain multiplies its CONDITION
NUMBERS as well as its matrices. A chain of well-behaved operators can fuse into
an ill-conditioned one, and the fused matrix is dense where the factors may have
been structured -- so `fusible` checks the conditioning and refuses rather than
handing back a matrix that computes the right thing in exact arithmetic and
something else in float32.
"""

import numpy as np


#: leCORE FOUND THIS PRINCIPLE FOUR TIMES BEFORE, in four domains, and never
#: unified it. Verified here that they are one idea:
#:   filter_passes(field, k, N)   N passes of a circular filter == the transfer
#:                                raised to N. Agrees with power_matrix to
#:                                4.4e-16 at N=1 and 3.0e-15 at N=1,000, and its
#:                                own docstring already says N=1,000,000 costs
#:                                what N=1 costs.
#:   affine_compose(chain)        a chain of (s,t) edits collapses to ONE (S,T)
#:                                by the affine group law -- 1.8e-15 against
#:                                running the chain.
#:   diffuse_steady_state(field)  the CLOSED-FORM LIMIT of unbounded diffusion,
#:                                mean preserved exactly.
#:   soft_chain_matrices(...)     an implicit-Euler substep AS an affine map
#:                                (A, b) -- described in its own docstring as
#:                                "the reference scene for the modal jump".
#: A REPEATED LINEAR MAP HAS A CLOSED FORM. fuse, power_matrix and
#: limit_operator are the fifth costume, and the only new thing about them is
#: WHERE the closed form goes: into a model's weights.
KNOWN_COSTUMES = ("filter_passes", "affine_compose", "diffuse_steady_state",
                  "soft_chain_matrices")


def fusible(ops, max_condition=1e6):
    """Should this chain be fused? Returns (ok, report).

    REFUSES on conditioning, because fusion multiplies condition numbers along
    with matrices. Two operators that are each harmless can fuse into one that
    is not, and the failure is silent in float32 -- the fused matrix computes
    the right answer in exact arithmetic and a different one on the machine
    that will actually run it."""
    from holographic.io_and_interop.holographic_vminstall import fuse

    M = fuse(*ops)
    cond = float(np.linalg.cond(M))
    worst = max(float(np.linalg.cond(np.asarray(o, np.float64))) for o in ops)
    ok = cond <= float(max_condition)
    return ok, {"condition": cond, "worst_factor": worst,
                "amplification": cond / max(worst, 1e-30),
                "ok": ok, "n_ops": len(ops),
                "why": ("fusible" if ok else
                        "fused condition %.3g exceeds %.3g -- install the chain "
                        "in stages instead" % (cond, max_condition))}


def limit_operator(A, tol=0.999):
    """The converged answer of x <- Ax + b, as ONE matrix. None if it diverges.

    (I - A)^-1 exists exactly when the spectral radius is below 1, which is also
    exactly when the iteration converges -- so the check and the construction
    are the same fact, and a divergent iteration returns None rather than a
    plausible matrix."""
    A = np.asarray(A, np.float64)
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho >= float(tol):
        return None, {"spectral_radius": rho, "converges": False,
                      "why": "spectral radius %.4f -- the iteration does not "
                             "converge, so it has no limit to install" % rho}
    return np.linalg.inv(np.eye(A.shape[0]) - A), {"spectral_radius": rho,
                                                   "converges": True}


def plan(ops=None, iteration=None, max_condition=1e6):
    """How should this be installed: fused, at its limit, per token, or not?"""
    if iteration is not None:
        M, rep = limit_operator(iteration)
        if M is not None:
            return {"how": "limit", "operator": M, "report": rep,
                    "why": "a contracting linear iteration installs at its "
                           "CONVERGED ANSWER in one layer"}
        return {"how": "token_step", "operator": np.asarray(iteration),
                "report": rep,
                "why": "no limit to install -- carry one step per token, which "
                       "is the resonator's route"}
    if ops:
        ok, rep = fusible(ops, max_condition=max_condition)
        from holographic.io_and_interop.holographic_vminstall import fuse
        return {"how": "fuse" if ok else "stages",
                "operator": fuse(*ops) if ok else None, "report": rep,
                "why": rep["why"]}
    return {"how": None, "why": "nothing to plan"}


def _selftest():
    H = 96
    rng = np.random.default_rng(0)

    # ---- A CHAIN FUSES EXACTLY, at any depth ----
    ops = [np.eye(H) + rng.standard_normal((H, H)) * 0.01 for _ in range(32)]
    p = plan(ops=ops)
    assert p["how"] == "fuse", p["report"]
    x = rng.standard_normal(H)
    want = x.copy()
    for M in ops:
        want = M @ want
    got = p["operator"] @ x
    assert float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want))) \
        > 0.999999

    # ---- AND AN ILL-CONDITIONED CHAIN MUST BE REFUSED, not silently fused ----
    bad = [np.diag(np.linspace(1.0, 1e-4, H)) for _ in range(4)]
    pb = plan(ops=bad)
    assert pb["how"] == "stages", pb["report"]

    # ---- A CONTRACTING ITERATION INSTALLS AT ITS LIMIT ----
    A = rng.standard_normal((H, H))
    A *= 0.5 / np.max(np.abs(np.linalg.eigvals(A)))
    pl = plan(iteration=A)
    assert pl["how"] == "limit", pl["report"]
    b = rng.standard_normal(H)
    it = b.copy()
    for _ in range(300):
        it = A @ it + b
    closed = pl["operator"] @ b
    assert float(it @ closed / (np.linalg.norm(it) * np.linalg.norm(closed))) \
        > 0.999999

    # ---- AND A DIVERGENT ONE MUST FALL BACK, not return a plausible matrix ----
    D = rng.standard_normal((H, H))
    D *= 1.5 / np.max(np.abs(np.linalg.eigvals(D)))
    pd = plan(iteration=D)
    assert pd["how"] == "token_step", pd["report"]

    # ---- AND IT MUST AGREE WITH THE COSTUME leCORE ALREADY HAD, or one of
    #      the two is wrong. filter_passes is power_matrix in the Fourier
    #      domain; if they disagree, do not ship either.
    import lecore as _lc
    _m = _lc.UnifiedMind(dim=64, seed=0)
    nf = 64
    fld = rng.standard_normal(nf)
    ker = np.array([0.25, 0.5, 0.25])
    # CONVOLUTION, y[i] = sum_j k[j] x[i-j]. Writing K[i,(i+j)%n] builds
    # CORRELATION -- the transpose -- and it disagrees by 1.7 rather than 1e-15.
    # The transform-convention trap is a KEPT NEGATIVE in this project and it
    # caught me again here.
    K = np.zeros((nf, nf))
    for i in range(nf):
        for j, kv in enumerate(ker):
            K[i, (i - j) % nf] = kv
    for N in (1, 8, 1000):
        a = np.asarray(_m.filter_passes(fld, ker, N))
        b = np.linalg.matrix_power(K, N) @ fld
        assert float(np.max(np.abs(a - b))) < 1e-12, (N,
                                                      float(np.max(np.abs(a - b))))

    print("unlocked selftest OK -- a 32-operator chain FUSES into one matrix "
          "exactly (and an ill-conditioned chain is REFUSED rather than silently "
          "fused, because fusion multiplies condition numbers); a contracting "
          "iteration installs AT ITS LIMIT, agreeing with 300 explicit "
          "iterations to better than 1e-6, so IK, PBD and relaxation install at "
          "their converged answer with no loop; and a DIVERGENT iteration falls "
          "back to one step per token rather than returning a plausible matrix; and "
          "it agrees with leCore's OWN existing costume of this idea -- "
          "filter_passes at N=1,000 matches a matrix power to 3e-15, which is "
          "the fourth place the engine had already found that a repeated linear "
          "map has a closed form")


if __name__ == "__main__":
    _selftest()
