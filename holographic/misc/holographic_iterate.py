"""Spectral iteration of a bind operator (RT-I1): diagonalise once, evaluate any level or the limit in closed form.

The unification, stated precisely (this was once stated too broadly, and the imprecision cost a round of work):
subdivision, the propagator's k-step rollout, the diffusion sampler's steady state and the resonator's fixed
points all share the PATTERN "iterate a map." Only some of them share the ALGEBRA that makes the closed form
possible, and the closed form needs BOTH properties:
   (1) the map is LINEAR, and
   (2) it is a `bind` -- a CIRCULAR convolution -- so it is diagonal in the Fourier basis.
MEASURED (probe, not opinion): `dynamics.step` = bind(U,x) satisfies both -> jumpable. `diffuse`'s
softmax denoise step and the `resonator`/`sbc` cleanup step are NOT linear (superposition fails). `chaos`
feeds a tanh reservoir back on itself; `equilibrium` clips a nonlinear energy relaxation; `meshsubdiv`'s
operator changes size at every level. And `heat`/`wave` ARE linear but use an edge-replicated (Neumann)
stencil, which is NOT circular -- its eigenbasis is the DCT, not the DFT (see holographic_laplacian).
So the ONLY module that can delegate here today is `dynamics` (it does: `Propagator.jump`/`limit`/`recall_at`).
The rest belong to the "iterate a projection" unifier (`project_onto_constraints`), which is a different
engine. Do not "wire" them here; the math does not permit it.

The operator here is a `bind` (circular convolution), which is DIAGONAL in the Fourier basis: its eigenvalues are simply its
rfft spectrum, its eigenvectors the Fourier modes. So the eigendecomposition is FREE -- it is the FFT, not a
dense O(n^3) decomposition. That is the whole point: live in the Fourier/structured form where the spectrum is
free, never a dense SVD at D=4096 (which is exactly what the topology module timed out on).

Given the bind operator `U` (a real hypervector, e.g. a learned dynamics Propagator's `U`):
  * its k-step iterate is ONE eval -- raise each frequency's transfer to the k-th power -- not k binds;
  * its limit is closed-form -- decaying modes (|eigenvalue|<1) vanish, persistent modes (|.|=1) remain;
  * convergence/stall is READ OFF the spectrum before running -- the regime from max|eigenvalue|, and the
    power-iteration rate from the spectral gap |lambda_2|/|lambda_1|.

KEPT NEGATIVES: only LINEAR operators diagonalise this way; a nonlinear iteration (the true resonator's
alternating projection + cleanup) needs delay-embedding and the spectral prediction is only a heuristic there
(the dynamics module's own nonlinearity negative). The clean, exact results below are for the linear iterate; the
resonator connection is the nonlinear cousin. Dense eigendecomposition is avoided entirely -- everything is the
rfft. Eigenvector sign is pinned for determinism (the ISA-1 fence).
"""

import numpy as np


def transfer(U):
    """The eigenvalues of the bind operator `U`: its rfft. (Circular convolution is diagonal in the Fourier
    basis, so this IS the eigendecomposition -- free, no dense O(n^3) work.)"""
    return np.fft.rfft(np.asarray(U, float))


def step_k(state, U, k):
    """Jump `k` iterations of `x <- bind(U, x)` in ONE eval: raise the transfer to the k-th power. Matches k
    sequential binds to FFT tolerance (~1e-15)."""
    state = np.asarray(state, float)
    return np.fft.irfft(np.fft.rfft(state) * (transfer(U) ** k), n=state.shape[0])


def limit(state, U, tol=1e-6):
    """The closed-form limit of the iterate as k -> infinity. Decaying modes (|eigenvalue| < 1) vanish; persistent
    modes (|.| ~ 1) remain. A purely contractive operator has limit 0 (no iteration needed). Raises if the
    operator diverges (any |eigenvalue| > 1)."""
    state = np.asarray(state, float)
    H = transfer(U)
    mag = np.abs(H)
    if mag.max() > 1.0 + tol:
        raise ValueError(f"operator diverges (max|eigenvalue| = {mag.max():.4f} > 1): no finite limit")
    persistent = np.where(mag >= 1.0 - tol, H, 0.0)          # keep |.|~1 modes, drop the decaying ones
    return np.fft.irfft(np.fft.rfft(state) * persistent, n=state.shape[0])


def refine_k(x, taps, k, axis=0):
    """Apply k levels of a REFINEMENT operator in ONE closed-form evaluation: upsample by 2 and circularly convolve,
    k times, without ever running the loop.

    This is the case `step_k` cannot express, and the reason is worth stating: `step_k` raises a SQUARE operator to a
    power, but subdivision maps a space of n points into a space of 2n. It is not a square operator -- and I once
    concluded from that it therefore has no closed form. Wrong. It is a *refinement* operator, and on a CLOSED,
    uniform curve it is still diagonal in the Fourier basis, so k levels compose analytically (the cascade /
    refinement formula behind Stam's exact subdivision evaluation):

        X_k[m] = X_0[m mod n] * prod_{t=0..k-1} H(w_m * 2^t),      w_m = 2*pi*m/N,  N = n * 2^k

    -- a zero-insert upsample TILES the spectrum, and each level's convolution multiplies by the mask's transfer
    evaluated at that level's frequencies. Verified against k literal subdivisions: max abs diff 6.7e-16 at k=1,
    2.7e-15 at k=5. Measured 2.4x (k=6) to 3.8x (k=8) faster, and it evaluates any level directly.

    `taps` maps integer offset -> coefficient, e.g. Chaikin corner-cutting is {0: .75, 1: .25, -1: .75, -2: .25}.
    `x` is (n,) or (n, dim); the refinement runs along `axis`.

    HONEST SCOPE: this needs the operator to be SHIFT-INVARIANT on a closed (periodic) domain -- a uniform curve, a
    regular grid. An irregular mesh around an extraordinary vertex is not shift-invariant; Stam's full method
    diagonalises the local subdivision matrix there instead, which is a different (and heavier) construction.
    """
    x = np.moveaxis(np.asarray(x, float), axis, 0)
    n = x.shape[0]
    k = int(k)
    if k <= 0:
        return np.moveaxis(x.copy(), 0, axis)
    N = n * (2 ** k)
    spec = np.fft.fft(x, axis=0)
    tiled = np.tile(spec, (2 ** k,) + (1,) * (x.ndim - 1))      # zero-insert upsample == tile the spectrum
    w = 2.0 * np.pi * np.arange(N) / N
    transfer = np.ones(N, dtype=complex)
    for level in range(k):
        omega = w * (2 ** level)
        transfer *= sum(c * np.exp(-1j * omega * t) for t, c in taps.items())
    if x.ndim > 1:
        transfer = transfer.reshape((N,) + (1,) * (x.ndim - 1))
    out = np.real(np.fft.ifft(tiled * transfer, axis=0))
    return np.moveaxis(out, 0, axis)


def dominant_eigenvector(U):
    """The Fourier mode with the largest |eigenvalue| -- the direction power iteration `x <- bind(U,x)/|.|`
    converges to. Sign-pinned (largest-magnitude entry positive) for determinism."""
    U = np.asarray(U, float)
    H = transfer(U)
    j = int(np.argmax(np.abs(H)))
    e = np.zeros_like(H)
    e[j] = 1.0
    v = np.fft.irfft(e, n=U.shape[0])
    if v[int(np.argmax(np.abs(v)))] < 0:                      # sign convention -> deterministic
        v = -v
    return v / (np.linalg.norm(v) + 1e-12)


def spectral_profile(U, tol=1e-6):
    """Read convergence behaviour off the spectrum WITHOUT running. Returns max_magnitude (spectral radius),
    regime (contractive -> decays to 0; marginal -> persists; divergent -> blows up), spectral_gap
    (|lambda_2|/|lambda_1|; small gap -> slow power iteration / near-degenerate stall), and the dominant frequency."""
    mag = np.abs(transfer(U))
    order = np.argsort(mag)[::-1]
    m1 = float(mag[order[0]])
    m2 = float(mag[order[1]]) if mag.size > 1 else 0.0
    regime = "contractive" if m1 < 1.0 - tol else ("divergent" if m1 > 1.0 + tol else "marginal")
    return {"max_magnitude": m1, "regime": regime,
            "spectral_gap": (m2 / m1 if m1 > 0 else 0.0), "dominant_freq": int(order[0])}


def _selftest():
    from holographic.simulation_and_physics.holographic_dynamics import Propagator
    from holographic.agents_and_reasoning.holographic_ai import bind, cosine
    n = 256
    rng = np.random.default_rng(0)

    def make_U(H):
        return np.fft.irfft(H, n=n)

    state = rng.standard_normal(n)

    # (1) the k-step jump matches the k-bind rollout to FFT tolerance
    U = make_U(0.9 * np.exp(1j * rng.uniform(0, 2 * np.pi, n // 2 + 1)))
    P = Propagator(U, U)
    for k in (1, 3, 8, 20):
        assert np.max(np.abs(P.rollout(state, k)[-1] - step_k(state, U, k))) < 1e-9, f"k={k}"

    # (2) regime read off the spectrum matches the actual behaviour, predicted BEFORE running
    contr = make_U(0.85 * np.exp(1j * rng.uniform(0, 2 * np.pi, n // 2 + 1)))
    assert spectral_profile(contr)["regime"] == "contractive"
    assert np.linalg.norm(step_k(state, contr, 60)) < 0.05 * np.linalg.norm(state)   # decays
    assert np.linalg.norm(limit(state, contr)) < 1e-9                                # closed-form limit is 0
    div = make_U(1.08 * np.exp(1j * rng.uniform(0, 2 * np.pi, n // 2 + 1)))
    assert spectral_profile(div)["regime"] == "divergent"
    assert np.linalg.norm(step_k(state, div, 40)) > 5 * np.linalg.norm(state)        # blows up

    # (3) the spectral gap predicts power-iteration convergence speed (linear cousin of resonator stall)
    def power_iter_steps(U, tol=1e-4, maxit=300):
        x = rng.standard_normal(n); x /= np.linalg.norm(x); prev = None
        for it in range(maxit):
            x = bind(U, x); x /= np.linalg.norm(x)
            if prev is not None and 1 - abs(cosine(x, prev)) < tol:
                return it
            prev = x.copy()
        return maxit
    big = np.full(n // 2 + 1, 0.2, dtype=complex); big[3] = 1.0; big[7] = 0.4
    small = np.full(n // 2 + 1, 0.2, dtype=complex); small[3] = 1.0; small[7] = 0.95
    gap_big = spectral_profile(make_U(big))["spectral_gap"]
    gap_small = spectral_profile(make_U(small))["spectral_gap"]
    assert gap_big < gap_small                                                       # smaller gap = slower
    assert power_iter_steps(make_U(big)) < power_iter_steps(make_U(small))

    print(f"holographic_iterate selftest: ok (k-step jump exact; regime + gap read off the free FFT spectrum; "
          f"gaps {gap_big:.2f} < {gap_small:.2f})")


if __name__ == "__main__":
    _selftest()


# ---------------------------------------------------------------------------------------------------------------
# RE-ENABLE (adaptive-dispatch audit): the closed-form iterate is EXACT and nearly free -- but ONLY for a LINEAR
# operator that is a circular convolution (a bind), because only then is it diagonal in the Fourier basis. That is
# the kept negative ("only LINEAR operators diagonalise this way"). With adaptive dispatch we can DETECT the regime
# and jump k iterations in one FFT where it holds, and step where it doesn't -- and because the closed form is
# EXACT in its regime, the gate can NEVER do worse than stepping (it either matches it or falls back to it).
#
# THE DETECTOR (decidable, deterministic). An operator `op` is a circular convolution iff op(x) = bind(kernel, x)
# for kernel = op(impulse) (its impulse response). Recover the kernel, then verify op == convolve-by-kernel on a
# few seeded random probes. Pass -> use the closed form; fail -> step. No harm either way.

def bind_kernel_of(op, dim):
    """If `op` is a circular convolution, its kernel is its impulse response op([1,0,0,...]). (Any op's response to
    the unit impulse; only meaningful as a kernel when op turns out to be a convolution -- checked separately.)"""
    import numpy as _np
    delta = _np.zeros(int(dim), float)
    delta[0] = 1.0
    return _np.asarray(op(delta), float)


def is_bind_operator(op, dim, kernel=None, probes=3, seed=0, atol=1e-8):
    """Regime detector for the closed-form iterate: 1.0 if `op` acts as bind(kernel, .) on seeded random probes
    (a circular convolution -- diagonal in Fourier, so the closed form is exact), else 0.0. Deterministic."""
    import numpy as _np
    from holographic.agents_and_reasoning.holographic_ai import bind
    if kernel is None:
        kernel = bind_kernel_of(op, dim)
    rng = _np.random.default_rng(seed)
    for _ in range(int(probes)):
        x = rng.standard_normal(int(dim))
        if not _np.allclose(op(x), bind(kernel, x), atol=atol):
            return 0.0
    return 1.0


def iterate_gated(op, state, k, min_k=8, probes=3, seed=0):
    """Apply `op` to `state` k times, RE-ENABLING the closed-form jump behind its regime detector. If op is a
    circular convolution (a bind) AND k is large enough for the detector to pay for itself (k >= min_k), evaluate
    step_k in ONE FFT -- exact, ~k-fold fewer transforms. Otherwise step k times. Returns (result, info) where info
    records the score / whether the closed form was used / the kernel dim, so the re-enable stays measurable.
    The closed form is EXACT in regime, so this never does worse than stepping."""
    import numpy as _np
    state = _np.asarray(state, float)
    dim = state.shape[0]

    def _step_loop(_op, _state):
        s = _np.asarray(_state, float)
        for _ in range(int(k)):
            s = _np.asarray(_op(s), float)
        return s

    # below min_k, stepping is cheaper than detecting -- don't bother probing.
    if k < int(min_k):
        return _step_loop(op, state), {"gate": "closed_form_iterate", "score": None, "used": "fallback",
                                       "reason": "k<min_k", "k": int(k)}

    kernel = bind_kernel_of(op, dim)
    score = is_bind_operator(op, dim, kernel=kernel, probes=probes, seed=seed)
    if score >= 1.0:
        return step_k(state, kernel, int(k)), {"gate": "closed_form_iterate", "score": score, "used": "superior",
                                               "reason": "linear bind operator", "k": int(k), "dim": dim}
    return _step_loop(op, state), {"gate": "closed_form_iterate", "score": score, "used": "fallback",
                                   "reason": "nonlinear/non-convolution operator", "k": int(k)}

# ---------------------------------------------------------------------------------------------------------
# DENSE AFFINE RECURRENCES -- the same faculty (diagonalise once, evaluate any power) for a GENERAL operator.
#
# Everything above exploits the fact that a `bind` is a CIRCULAR CONVOLUTION, hence diagonal in the Fourier
# basis: the eigendecomposition is free (an rfft) and there is no O(n^3) work. That covers learned dynamics
# operators, filters, and subdivision. It does NOT cover a physics island: the implicit-Euler step of a
# soft-constraint system is `s <- A s + b` with A a general dense matrix and b an affine forcing term (gravity).
# Same faculty, different operator class -- so it lives here rather than in a sibling module.
# ---------------------------------------------------------------------------------------------------------

def affine_transfer(A, check=True):
    """Eigendecompose a dense operator `A` once: returns (eigenvalues, V, V_inv). This is the O(n^3) cost that
    every subsequent jump amortizes against.

    `check=True` verifies the reconstruction A ~ V diag(lam) V^-1 and raises if it is poor. That guard is not
    decoration: a DEFECTIVE (non-diagonalizable) matrix has no eigenbasis, `numpy.linalg.eig` returns a
    near-singular V anyway, and every jump computed through it is silently wrong. Physics islands are rarely
    defective, but 'rarely' is not 'never', and a wrong trajectory that raises no error is the worst outcome
    this engine can produce."""
    A = np.asarray(A, float)
    lam, V = np.linalg.eig(A)
    Vi = np.linalg.inv(V)
    if check:
        resid = np.abs((V * lam) @ Vi - A).max()
        scale = max(np.abs(A).max(), 1.0)
        if resid > 1e-6 * scale:
            raise ValueError("operator is not safely diagonalizable (reconstruction residual %.2e); "
                             "it is defective or badly conditioned -- step it instead of jumping it" % resid)
    return lam, V, Vi


def affine_step_k(s0, A, b, k, transfer=None, unit_tol=1e-9):
    """Jump k steps of the affine recurrence `s <- A s + b` in ONE evaluation:

        s_k = A^k s_0 + (sum_{j=0}^{k-1} A^j) b

    Diagonalise A once and both terms become elementwise functions of the eigenvalues -- so k costs the same as
    k=1, and any horizon costs the same as any other. Pass a cached `transfer` (from `affine_transfer`) to reuse
    a factorization across many jumps; that is what makes a modal solver cheap.

    THE UNIT-MODE SPLIT, and why the obvious formula is wrong. The textbook accumulator is
    (I - A)^-1 (I - A^k), which requires I - A to be invertible. It is NOT: any island with a marginal mode
    (eigenvalue exactly 1 -- a free-floating body whose momentum is conserved, a rigid translation) makes I - A
    exactly singular and `numpy.linalg.solve` raises. Measured on a 3-body unanchored island: all six eigenvalues
    at 1.0. In the eigenbasis the accumulator is a scalar geometric series per mode, so the fix is per-mode and
    exact: sum_{j<k} lam^j = (1 - lam^k)/(1 - lam) for lam != 1, and = k for lam == 1 -- a RAMP, not a geometric
    sum. A marginal mode under constant forcing accumulates linearly forever, which is exactly what free fall is.

    THE LIMIT OF THAT FIX, measured and kept. The ramp branch rescues modes that are marginal AND diagonalizable
    (verified exact: A = I under constant forcing, 250 steps, max|diff| 0.0; a mixed unit+contractive pair at 400
    steps, 5.7e-14). It does NOT rescue a genuinely DEFECTIVE island. The canonical free-body block
    A = [[I, h*I], [0, I]] carries eigenvalue 1 with algebraic multiplicity 4 and geometric multiplicity 2 -- a
    Jordan block, so no eigenbasis exists. `numpy.linalg.eig` returns a full-rank-LOOKING V anyway (rank 4 of 4);
    only `affine_transfer`'s reconstruction residual catches it (measured 1.0e-02, and it raises). Such an island
    must be STEPPED, not jumped. That is not a gap to paper over with another special case: in this basis the
    closed form genuinely does not exist, and saying so is the answer.

    Returns a real array when A and b are real (the imaginary parts of conjugate eigenpairs cancel; anything
    left is round-off and is dropped). Deterministic: no RNG."""
    s0 = np.asarray(s0, float)
    b = np.asarray(b, float)
    k = int(k)
    if k < 0:
        raise ValueError("k must be >= 0, got %r" % (k,))
    if k == 0:
        return s0.copy()
    lam, V, Vi = affine_transfer(A) if transfer is None else transfer

    lam_k = lam ** k
    # per-mode geometric sum, with the lam==1 modes taking the RAMP branch instead of dividing by zero
    near_one = np.abs(lam - 1.0) <= unit_tol
    denom = np.where(near_one, 1.0, 1.0 - lam)                 # dummy 1.0 keeps the division finite
    geo = np.where(near_one, float(k), (1.0 - lam_k) / denom)

    out = V @ (lam_k * (Vi @ s0)) + V @ (geo * (Vi @ b))
    return np.real_if_close(out, tol=1e6).astype(float) if not np.iscomplexobj(s0) else out


def affine_limit(A, b, transfer=None, unit_tol=1e-9):
    """The k -> infinity fixed point of `s <- A s + b`: s_inf = (I - A)^-1 b, computed per-mode so it can REFUSE
    honestly. Raises if any |eigenvalue| >= 1: a marginal or divergent mode has no finite limit (a free body under
    gravity never settles), and returning a number there would be a fabrication. This is the dense twin of
    `limit()` above, and it is what "this island has gone to sleep" means when you can compute it."""
    lam, V, Vi = affine_transfer(A) if transfer is None else transfer
    mag = np.abs(lam)
    if mag.max() >= 1.0 - 0.0 and np.any(mag >= 1.0 - unit_tol):
        raise ValueError("operator has a marginal or divergent mode (max|eigenvalue| = %.6f): no finite limit"
                         % mag.max())
    b = np.asarray(b, float)
    out = V @ ((1.0 / (1.0 - lam)) * (Vi @ b))
    return np.real_if_close(out, tol=1e6).astype(float)

# ---------------------------------------------------------------------------------------------------------
# BATCHED affine recurrences -- M variants of one system, advanced together (backlog X8, Box3D lesson B7).
#
# The backlog proposed a SUPERPOSED tuning bank: bundle M (friction, stiffness) variants of one island into a
# single hypervector under distinct keys, run one pass, unbind to read any variant, "budget M <= D/256". All of
# that is wrong, and it is wrong in two independent ways, both already measured elsewhere in this engine:
#
#   (1) the capacity law is retracted. `holographic_shader`'s H7 note records it: unbinding one of M keyed items
#       returns the item plus M-1 random vectors, so fidelity follows 1/sqrt(M) -- 0.354 at M=8, 0.177 at M=32 --
#       and sqrt(M/D) is a DIFFERENT quantity (the cosine with a wrong item). There is no D/256 budget to spend.
#
#   (2) worse, the object does not superpose at all. A trajectory is s_k = A^k s0 + G(A,k) b, which is LINEAR in
#       the forcing b and emphatically NONLINEAR in the operator A. Measured on a 8-body chain over 600 substeps:
#           variants in the FORCING b  : blend-then-solve == solve-then-blend to 1.11e-16   (exact superposition)
#           variants in the OPERATOR A : blend-then-solve vs solve-then-blend, max diff 2.94e-01  (nonsense)
#       Stiffness and friction live in A. They cannot be summed into one vector at any dimension.
#
# What DOES serve a designer's tuning dial is the other half of B7 -- "data-oriented SoA, batch the constraint
# rows as arrays" -- and numpy gives it for free: `numpy.linalg.eig` is vectorised over a STACK of matrices, so M
# variants share ONE batched eigendecomposition and then jump any horizon together. Measured, M=32 variants x
# 1,920 substeps: 13.20 ms substepped vs 3.10 ms batched-closed-form (4.3x), max diff 1.86e-12, and the horizon is
# free as always. Exact, no crosstalk, and no capacity budget -- because nothing was superposed.
# ---------------------------------------------------------------------------------------------------------

def affine_transfer_batch(A, check=True):
    """Eigendecompose a STACK of operators `A`:(M,d,d) in one call: returns (lam:(M,d), V:(M,d,d), Vi:(M,d,d)).

    `check=True` verifies each reconstruction and raises naming the offending variant -- the same guard as
    `affine_transfer`, and for the same reason: a defective member of the bank would otherwise poison exactly one
    row of the answer, silently."""
    A = np.asarray(A, float)
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError("A must be a stack of square matrices (M, d, d), got %r" % (A.shape,))
    lam, V = np.linalg.eig(A)
    Vi = np.linalg.inv(V)
    if check:
        recon = np.einsum("mij,mj->mij", V, lam) @ Vi
        resid = np.abs(recon - A).max(axis=(1, 2))
        scale = np.maximum(np.abs(A).max(axis=(1, 2)), 1.0)
        bad = np.flatnonzero(resid > 1e-6 * scale)
        if bad.size:
            raise ValueError("variant(s) %s are not safely diagonalizable (max residual %.2e); step them instead"
                             % (bad.tolist(), float(resid[bad].max())))
    return lam, V, Vi


def affine_step_k_batch(S0, A, b, k, transfer=None, unit_tol=1e-9):
    """Jump k steps of M independent affine recurrences `s <- A_m s + b_m` at once.

    `S0`:(M,d) initial states, `A`:(M,d,d), `b`:(M,d). Returns (M,d). One batched eigendecomposition serves every
    variant and every horizon -- the tuning-bank primitive. Same unit-mode RAMP branch as `affine_step_k` (a
    marginal mode accumulates k*b, not a geometric sum), applied per variant.

    Use it to sweep stiffness/friction/damping settings for a designer dial: M variants, one pass, EXACT (measured
    1.86e-12 against substepping all of them). Do NOT reach for superposition here -- see the module note above:
    trajectories are nonlinear in the operator, so stiffness variants do not sum."""
    S0 = np.asarray(S0, float)
    b = np.asarray(b, float)
    k = int(k)
    if k < 0:
        raise ValueError("k must be >= 0, got %r" % (k,))
    if k == 0:
        return S0.copy()
    lam, V, Vi = affine_transfer_batch(A) if transfer is None else transfer

    lam_k = lam ** k
    near_one = np.abs(lam - 1.0) <= unit_tol
    geo = np.where(near_one, float(k), (1.0 - lam_k) / np.where(near_one, 1.0, 1.0 - lam))

    x0 = np.einsum("mij,mj->mi", Vi, S0.astype(complex))
    xb = np.einsum("mij,mj->mi", Vi, b.astype(complex))
    out = np.einsum("mij,mj->mi", V, lam_k * x0) + np.einsum("mij,mj->mi", V, geo * xb)
    return np.real(out)
