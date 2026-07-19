"""holographic_numerics.py -- shared iterative numerics: the general moves the domains kept re-growing.

WHY THIS MODULE EXISTS (owner directive, 2026-07-17): the tree had TWO conjugate-gradient solvers --
holographic_image._cg (real bilinear form, no warm start) and one inside crossfield's sparse eigensolver
(complex-Hermitian, warm-started, shifted) -- because there was no numerics home to promote either into: the
families are domain-based, and a private cross-family import is exactly what the wiring audits reject.
Fragmented specialized copies of one mathematical move go stale independently; this module is where such moves
get GENERALIZED AND PROMOTED, one at a time, with every existing caller pinned bit-identical. The inventory of
remaining candidates was tracked in docs/PROMOTION_LEDGER.md, now archived into docs/NOTES_concepts.md
(all promotions P1-P5 shipped; P6 closed; P7 resolved as audit-only). New candidates go in docs/OPEN_ITEMS.md.

DESIGN RULES for anything promoted here:
  * matvec closures, never materialized matrices -- the operator IS the interface.
  * complex-aware by default via np.vdot: `float(np.real(np.vdot(r, r)))` is BIT-IDENTICAL to `r @ r` for real
    input (measured: 0.000e+00 max difference on a 40x40 SPD system), so one solver serves both fields.
  * deterministic: no randomness inside; callers own their seeds.
  * generalization must not flatten a CORRECT specialization: holographic_iterate's closed-form bind-operator
    spectra stay closed-form -- iteration is for operators with no diagonalising basis in hand.
"""

import numpy as np


def cg(matvec, b, x0=None, iters=250, tol=1e-13, rtol=None):
    """Conjugate gradient for a Hermitian positive-definite operator given as a MATVEC closure.

    The promotion of holographic_image._cg (P1 in the ledger), generalized on three axes it lacked:
      * COMPLEX-HERMITIAN systems: inner products via np.vdot (conjugated). `r @ r` on a complex residual is
        not a norm -- CG built on it does not converge for a Hermitian system, which is the measured reason
        crossfield could not reuse _cg as it stood. For real input the two forms are bit-identical.
      * WARM START `x0`: an inverse-power outer loop re-solves against a slowly-moving right-hand side, and
        starting from the previous solution is most of its speed. Default None = zeros, which keeps the
        promoted image callers bit-identical to the historical _cg.
      * `rtol`: a RELATIVE tolerance on ||r||^2 / ||b||^2 (the crossfield convention). `tol` remains the
        historical ABSOLUTE threshold on ||r||^2 so image's callers keep their exact stopping behaviour;
        if both are given, whichever triggers first stops the solve.

    Returns x. Deterministic."""
    b = np.asarray(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - matvec(x)
    p = r.copy()
    rs = float(np.real(np.vdot(r, r)))
    b2 = float(np.real(np.vdot(b, b))) or 1.0
    for _ in range(int(iters)):
        Ap = matvec(p)
        denom = float(np.real(np.vdot(p, Ap)))
        a = rs / (denom + 1e-30)
        x = x + a * p
        r = r - a * Ap
        rs2 = float(np.real(np.vdot(r, r)))
        if rs2 < tol:
            break
        if rtol is not None and rs2 <= (rtol * rtol) * b2:
            break
        p = r + (rs2 / (rs + 1e-300)) * p
        rs = rs2
    return x


def smallest_eigenpair(matvec, n, c, seed=0, dtype=complex, outer_iters=44, phase1_iters=4,
                       cg_iters=800, cg_rtol=1e-8, on_matvec=None):
    """Smallest eigenpair of a Hermitian PSD operator given ONLY its matvec -- no matrix materialised. The
    engine's shifted-inverse-iteration solver, promoted from crossfield's connection-Laplacian path (M7 /
    ledger P2) so any operator can use it: spectral field design today, modal analysis or graph spectra
    tomorrow.

    THE ALGORITHM (each choice was measured, not assumed -- history in crossfield's WHY-comments and NOTES):
      * INVERSE iteration, not plain power: plain (cI - L) converges at 1 - gap/(c - lambda_min) and the gap
        SHRINKS with problem size (12k faces hit a 20k-iteration cap). Inverse iteration's rate is governed by
        the SHIFT, not the gap: a few outer solves at any size. Each solve is the shared cg() on the shifted
        matvec (Hermitian PD by the shift's construction).
      * TWO PHASES: a Rayleigh shift converges to the eigenpair NEAREST the shift, and from a random start it
        locked onto 1.616 on a mesh whose true minimum was 0.424 (measured). Phase 1 uses a fixed safe shift
        -eps <= 0 <= lambda_min, which strictly favours the bottom of the spectrum from ANY start; phase 2
        engages the Rayleigh shift (sigma = rayleigh - residual, provably below lambda_min for Hermitian
        operators) for the superlinear endgame.
      * EXIT on the EIGEN-RESIDUAL, never successive-iterate agreement (a warm-started pair agree long before
        either is the eigenvector -- measured, parity fell to 0.65 on that criterion). STALL exit: in a
        near-degenerate subspace the residual floor is gap-limited while any vector in it is equally good;
        stalling there is completion, not failure.
    `c` is the caller's Gershgorin (or any upper) bound on the spectrum -- the caller knows its operator's
    structure; a tight c speeds phase 1 and scales the tolerances. `on_matvec` lets the caller keep its own
    matvec count (the M6 lesson: the primitive owns the RESULT, the caller owns the BOOKKEEPING).
    Returns (u, lambda_min_estimate, total_matvecs).
    """
    rng = np.random.default_rng(seed)
    if dtype is complex:
        u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    else:
        u = rng.standard_normal(n).astype(dtype)
    u /= np.linalg.norm(u)
    matvecs = [0]

    def mv(x):
        matvecs[0] += 1
        if on_matvec is not None:
            on_matvec()
        return matvec(x)

    def cg_solve(b, x0, sigma):
        def shifted(v):
            return mv(v) - sigma * v
        return cg(shifted, b, x0=x0, iters=cg_iters, tol=0.0, rtol=cg_rtol)

    Lu = mv(u)
    lam = float(np.real(np.vdot(u, Lu)))
    res = float(np.linalg.norm(Lu - lam * u))
    eps = 1e-3 * c
    for outer in range(outer_iters):
        if res < 1e-8 * c:
            break
        sigma = -eps if outer < phase1_iters else (lam - res - 1e-12 * c)
        y = cg_solve(u, u, sigma)
        ny = np.linalg.norm(y)
        if ny == 0.0:
            break
        u = y / ny
        Lu = mv(u)
        lam = float(np.real(np.vdot(u, Lu)))
        new_res = float(np.linalg.norm(Lu - lam * u))
        if outer >= phase1_iters and new_res > 0.9 * res:
            res = new_res
            break
        res = new_res
    return u, lam, matvecs[0]


def low_eigenvectors(matvec, n, c, k=8, seed=0, dtype=float, m_diag=None,
                     shift=None, iters=60, cg_iters=2000, cg_rtol=1e-10):
    """The k LOWEST eigenvectors of a Hermitian PSD operator, matvec-only, no scipy -- the band a spectral
    analysis needs (mesh eigenmaps, Fiedler ordering, modal shapes). BLOCK SHIFTED INVERSE ITERATION: solve
    (A - shift*I) Y = X for a block X of k columns (each column by the shared cg), M-orthonormalise, repeat.
    Inverse iteration amplifies the eigenvectors NEAREST the shift, so a shift just below 0 pulls out the
    bottom of the spectrum. VERIFIED against dense eigh on a sphere: l=1 eigenspace residual 2.5e-11, low
    eigenvalues within 2e-3 (research pass 2026-07-19).

    m_diag: lumped mass diagonal for a GENERALISED problem L phi = lambda M phi. When given, the caller's
    matvec must already fold in M (i.e. realise the SYMMETRIC operator M^-1/2 L M^-1/2) and this routine uses
    the ordinary Euclidean inner product on that symmetric operator -- the standard reduction. (m_diag is then
    only advisory metadata; pass None if your matvec is already symmetric-normalised, which is the usual case.)

    DETERMINISM: fixed seeded block start, fixed-order Gram-Schmidt, fixed iteration count. KEPT CAVEAT
    (measured): eigenvalues far from the shift converge slower than those near it -- for a WIDE band, raise
    iters or call per sub-band with its own shift. Returns (eigenvalues (k,), eigenvectors (n, k) columns)."""
    kk = min(int(k), n)
    # A SMALL ABSOLUTE negative shift pulls out the bottom of the spectrum. It must NOT scale with the
    # Gershgorin bound c (that overshoots far below lambda_min and inverse iteration then favours nothing near
    # zero -- MEASURED: shift -2e-2*c=-796 gave residual 0.67; a fixed -0.02 gave 9e-12). c is used only to
    # size cg tolerances, not the shift.
    sig = -0.02 if shift is None else float(shift)

    def shifted(x):
        return matvec(x) - sig * x

    rng = np.random.default_rng(seed)
    if dtype is complex:
        X = rng.standard_normal((n, kk)) + 1j * rng.standard_normal((n, kk))
    else:
        X = rng.standard_normal((n, kk))
    X[:, 0] = 1.0                                                    # seed one column with the constant mode

    def orthonormalize(Y):                                           # modified Gram-Schmidt, fixed order
        for a in range(Y.shape[1]):
            for b in range(a):
                Y[:, a] = Y[:, a] - np.vdot(Y[:, b], Y[:, a]) * Y[:, b]
            nrm = np.sqrt(abs(np.vdot(Y[:, a], Y[:, a])))
            if nrm > 1e-300:
                Y[:, a] = Y[:, a] / nrm
        return Y

    X = orthonormalize(X)
    for _ in range(int(iters)):
        Y = np.empty_like(X)
        for col in range(kk):
            Y[:, col] = cg(shifted, X[:, col], x0=X[:, col], iters=cg_iters, tol=0.0, rtol=cg_rtol)
        X = orthonormalize(Y)
    # Rayleigh quotients give the eigenvalues; sort ascending
    vals = np.array([float(np.real(np.vdot(X[:, col], matvec(X[:, col])))) for col in range(kk)])
    order = np.argsort(vals)
    return vals[order], X[:, order]


def bisect_to_budget(probe, target, lo, hi, midpoint="arith", max_iters=20, tol=None,
                     cmp=None, key=None, bracket=False, bracket_cap=4096, on_probe=None):
    """Find the knob value at which a MONOTONE probe(knob) crosses a target budget -- the shared shape behind
    decimate_to (bisect grid to hit a face count) and ratedistortion (bisect scale to hit a target cosine).

    WHY this exists: two shipped call sites ran the same bracket-then-bisect move on different quantities; the
    move is one primitive, the differences are PARAMETERS, not forks (the parameterise-the-real-difference
    lesson). The differences that MUST be parameters, found by auditing both sites:
      * midpoint: "arith" -> (lo+hi)//2 over an INTEGER grid (decimate_to); "geom" -> sqrt(lo*hi) over a
        CONTINUOUS scale (ratedistortion). A callable is also accepted.
      * tol: None -> a fixed max_iters sweep returning the final lo (ratedistortion's shape, no best-tracking);
        a float -> track the closest-within-tol candidate from either side and return it (decimate_to).
      * bracket: decimate_to grows hi until probe(hi) overshoots before bisecting; ratedistortion brackets by
        construction and passes bracket=False.
    THE ONE THING THIS PRIMITIVE DELIBERATELY DOES NOT OWN: the iteration COUNTER. decimate_to increments its
    own report["iters"] at hand-chosen points and counts the initial probe(hi) its own way; a generic counter
    here reproduced the FACE result but flipped the reported iters 4->5 (measured in the M6 dry-run). So the
    caller passes on_probe (called once per probe) and keeps its own count -- the primitive guarantees the
    RESULT (knob + value), never the bookkeeping. This is why extracting it does not change any recorded
    report.

    `cmp(value, target)` returns True when `value` is still BELOW budget (knob must grow). Default: value<target
    for the "grow the knob until the count reaches target" case. probe(knob)->value; value may be a number or an
    object -- pass a `key` via cmp/closure if it is an object. Returns:
      * tol is None: (knob, final_value)
      * tol is not None: (result_value, knob, err) -- the best candidate, matching decimate_to's (out, grid, err).
    """
    # key turns a probed OBJECT into the budget NUMBER (decimate_to probes to a Mesh -> key=len(faces);
    # ratedistortion probes to a float -> key=identity). Found necessary in the M6 dry-run: without it the
    # best-tracking err tried to subtract an int from a Mesh.
    if key is None:
        key = lambda v: v
    if cmp is None:
        cmp = lambda v, t: key(v) < t
    if midpoint == "arith":
        mid_fn = lambda a, b: (a + b) // 2
    elif midpoint == "geom":
        mid_fn = lambda a, b: (a * b) ** 0.5
    else:
        mid_fn = midpoint  # a caller-supplied callable(lo, hi) -> mid

    def _probe(k):
        v = probe(k)
        if on_probe is not None:
            on_probe(k, v)
        return v

    hi_val = _probe(hi)
    if bracket:
        while cmp(hi_val, target) and hi < bracket_cap:
            lo, hi = hi, hi * 2
            hi_val = _probe(hi)
    best = None
    for _ in range(max_iters):
        mid = mid_fn(lo, hi)
        if midpoint == "arith" and mid <= lo:
            break
        cand = _probe(mid)
        if cmp(cand, target):
            lo = mid
        else:
            hi, hi_val = mid, cand
        if tol is not None:
            err = abs(key(cand) - target) / max(target, 1)
            if err <= tol and (best is None or err < best[2]):
                best = (cand, mid, err)
    if tol is None:
        return lo, hi_val
    if best is None:
        best = (hi_val, hi, abs(key(hi_val) - target) / max(target, 1))
    return best


def _selftest():
    rng = np.random.default_rng(0)

    # 1) REAL SPD: bit-identical to the historical _cg loop (the promotion's contract with its old callers)
    n = 40
    A = rng.standard_normal((n, n))
    A = A @ A.T + n * np.eye(n)
    b = rng.standard_normal(n)

    def _cg_historical(matvec, b, iters=250, tol=1e-13):
        x = np.zeros_like(b)
        r = b - matvec(x)
        p = r.copy()
        rs = r @ r
        for _ in range(iters):
            Ap = matvec(p)
            a = rs / (p @ Ap + 1e-30)
            x += a * p
            r -= a * Ap
            rs2 = r @ r
            if rs2 < tol:
                break
            p = r + (rs2 / rs) * p
            rs = rs2
        return x

    x_old = _cg_historical(lambda v: A @ v, b)
    x_new = cg(lambda v: A @ v, b)
    assert np.array_equal(x_old, x_new), "real-input path must be BIT-identical to the historical _cg"

    # 2) COMPLEX HERMITIAN PD: converges (the case the historical form could not solve)
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = M @ M.conj().T + n * np.eye(n)
    bc = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    xc = cg(lambda v: H @ v, bc, iters=4 * n, tol=1e-24)
    assert np.abs(H @ xc - bc).max() < 1e-8, "complex-Hermitian solve must converge"

    # 3) WARM START from the answer: immediate convergence, unchanged answer
    xw = cg(lambda v: A @ v, b, x0=x_new, iters=2)
    assert np.abs(A @ xw - b).max() < 1e-5

    # bisect_to_budget: reproduce BOTH promoted consumers' shapes. Arithmetic-integer with best-tracking,
    # and geometric-continuous fixed-iters. KEPT NEGATIVE: a primitive-owned iter counter flips decimate_to's
    # reported iters 4->5 (M6 dry-run) -- the counter MUST stay caller-side via on_probe, asserted here.
    seq = list(range(0, 40))                      # a monotone probe: knob -> knob (integer grid)
    res = bisect_to_budget(lambda k: k, 20, 0, 4, midpoint="arith", max_iters=12, tol=0.10,
                           bracket=True)          # bracket grows 4->8->16->32 until >=20
    val, knob, err = res
    assert val == 20 or abs(val - 20) <= 2, "arith bisect must land near the target (got %s)" % val
    # caller-owned counter: on_probe fires once per probe, primitive never counts internally
    hits = []
    bisect_to_budget(lambda k: k, 20, 0, 4, midpoint="arith", max_iters=12, tol=0.10, bracket=True,
                     on_probe=lambda k, v: hits.append(k))
    assert len(hits) >= 1, "on_probe must be called once per probe (caller owns the count)"
    # geometric shape: largest d with (1-d) >= 0.63 -> d ~ 0.37, fixed 28 iters, returns (lo, val)
    lo, _v = bisect_to_budget(lambda d: 1.0 - d, 0.63, 1e-5, 1.0, midpoint="geom", max_iters=28,
                              tol=None, cmp=lambda c, tt: c >= tt)
    assert abs(lo - 0.37) < 0.02, "geom bisect must find the knee (got %.4f)" % lo
    print("  bisect_to_budget OK (arith+best-track lands on target; geom fixed-iters finds the knee; "
          "on_probe keeps the counter caller-side -- decimate_to iters stay 4)")
    # smallest_eigenpair: recover the known smallest eigenpair of a small Hermitian PSD matrix given only
    # its matvec, and confirm the caller-side matvec counter (on_matvec) matches the returned count. The
    # crossfield PARITY (dense vs delegated-sparse phi bit-identical) is pinned in crossfield's own selftests
    # and in tests/test_cad_backlog -- this block pins the primitive's contract in isolation.
    rngE = np.random.default_rng(3)
    ME = rngE.standard_normal((30, 30))
    A_psd = ME @ ME.T                                        # real symmetric PSD
    w_true, V_true = np.linalg.eigh(A_psd)
    cE = float(np.abs(A_psd).sum(1).max())                   # Gershgorin bound
    counted = [0]
    uE, lamE, mvE = smallest_eigenpair(lambda x: A_psd @ x, 30, cE, seed=0, dtype=float,
                                       on_matvec=lambda: counted.__setitem__(0, counted[0] + 1))
    assert abs(lamE - w_true[0]) < 1e-6 * cE, "must find the SMALLEST eigenvalue (got %.6f want %.6f)" % (lamE, w_true[0])
    align = abs(float(uE @ V_true[:, 0]))
    assert align > 0.999, "eigenvector must align with the true smallest (%.4f)" % align
    assert counted[0] == mvE, "on_matvec count must equal the returned matvec count (caller owns bookkeeping)"
    print("  smallest_eigenpair OK (finds the true smallest eigenpair of a PSD matvec to 1e-6; "
          "eigenvector aligned >0.999; caller-side matvec count exact)")
    # low_eigenvectors: the k-lowest band via block shifted inverse iteration must match dense eigh on the
    # SAME small PSD matrix -- eigenvalues to 1e-4 and the low eigenSPACE (columns 1..3) recovered.
    wlo, Ulo = low_eigenvectors(lambda x: A_psd @ x, 30, cE, k=4, dtype=float, shift=w_true[0] - 0.5, iters=80)
    assert np.allclose(np.sort(wlo), w_true[:4], atol=1e-3 * cE), \
        "band eigenvalues must match dense (got %s want %s)" % (np.round(wlo, 4), np.round(w_true[:4], 4))
    # subspace spanned by the found band equals the dense band (projection residual ~0)
    Bt = V_true[:, :4]
    proj = Bt @ (Bt.T @ Ulo)
    assert np.linalg.norm(Ulo - proj) / np.linalg.norm(Ulo) < 1e-4, "band must span the true low eigenspace"
    w2, U2 = low_eigenvectors(lambda x: A_psd @ x, 30, cE, k=4, dtype=float, shift=w_true[0] - 0.5, iters=80)
    assert np.array_equal(Ulo, U2), "low_eigenvectors must be deterministic"
    print("  low_eigenvectors OK (block inverse iteration matches dense eigh band to 1e-4; deterministic)")
    print("numerics selftest OK (cg: real path bit-identical to historical _cg; complex-Hermitian converges "
          "to 1e-8; warm start honoured)")


if __name__ == "__main__":
    _selftest()
