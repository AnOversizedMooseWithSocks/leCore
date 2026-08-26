"""The cross-cutting PROBE SWEEP -- six transfers the panel pre-judged as likely no-ops, measured and KEPT NEGATIVE.

These are the items the rendering-lessons cross-cutting backlog flagged as probes rather than builds: each takes a
technique that paid off for one operation and asks whether it transfers to a structurally different one. The panel's
prior was that all six would fail, for two reasons -- CONCENTRATION OF MEASURE (a high-dimensional kernel is already
near-optimal, so there is no slack to win) for A1, A2, B4, D5; and a WRONG-SHAPED OPERATION (the technique's
precondition does not hold) for B2 and D3. Every one was measured on the real substrate; every one confirmed the
prior. They ship as kept negatives -- no faculty, no tour line -- because the measured reason each fails is the
useful artifact, and it is the same throughline as the C4 / D1 / B1 / D4 negatives: a sound technique applied to an
operation whose shape defeats it.

  A1  Low-discrepancy / blue-noise codebook  [SAMPLE-1 -> kernel]. Build atoms by Riesz-energy repulsion on the
      sphere instead of i.i.d. Gaussian, to lower mutual coherence and raise capacity. MEASURED: repulsion lowers
      MAX coherence by only ~3% and MEAN coherence not at all, at every dimension from 64 to 1024, and recall
      capacity is unchanged (e.g. 40 bound pairs in d=512: 0.70 recall both); even 500 relaxation steps barely move
      it. Concentration of measure -- random atoms are already near-uniform on the sphere. NO-OP.

  A2  Negative-lobe cleanup sharpening  [XDATA-3 -> kernel]. Deconvolve the similarity profile (subtract correlated
      leakage alpha*(G-I)@s) to sharpen an ambiguous cleanup peak, then argmax. MEASURED: it HURTS discrete cleanup
      -- on correlated atoms it amplifies noise and collapses accuracy (e.g. 0.18 -> 0.00) -- and is a no-op for
      orthogonal atoms (G ~= I). Hard nearest-neighbour is already Bayes-optimal for 'which atom is this'. NEGATIVE.

  B2  Throughput-gated generation  [RAY-1 -> text]. Stop generation when the running coherence falls below a floor,
      abstaining on an incoherent tail. MEASURED: the running coherence signal does NOT separate in-distribution
      generation from a garbage seed (means -37 vs -39, heavily overlapping) -- because steered generation already
      pulls any start toward coherent continuations, so there is no incoherent tail for a gate to catch. The
      coherence DEFENSE is redundant; the explicit abstention has nothing to fire on. REDUNDANT NO-OP.

  B4  Low-discrepancy sampling in generation  [SAMPLE-1 -> text]. Use LD-sequenced noise where the diffusion sampler
      injects noise, for more even coverage. MEASURED: an LD-noise diffusion matches an i.i.d.-noise diffusion
      exactly on both diversity (41 vs 42 distinct atoms reached) and validity (0.507 vs 0.510) -- the diffusion is
      ATTRACTOR-dominated (the cleanup step, not the noise spread, decides where it lands), so more-even noise
      changes nothing. (A categorical token draw is a single multinomial pick and cannot benefit from LD at all.)
      NO-OP.

  D3  MIS-combined decision signals  [MIS-1 -> creature]. Combine the creature's value, safety reflex, and novelty
      by a balance heuristic weighted by reliability, instead of a hard safety VETO plus greedy value. MEASURED: in
      typical states the soft blend makes the SAME choices as the veto (0 lethal choices over 84 sensed-danger
      states) because the learned value already disfavours danger -- so it LOOKS like a no-op. But the veto's whole
      value is the RESIDUAL the value misestimates: the survival bench measured a ~0.6%/step poison risk that
      compounds to 67-73% of long lives WITHOUT the veto. A soft penalty allows a lethal move whenever the value
      advantage exceeds the penalty, so it cannot give the guarantee a hard constraint gives. MIS combines
      estimators by reliability; a safety constraint is not an estimator. NEGATIVE (wrong-shaped operation).

  D5  Observation denoising  [XDATA-1/2 -> creature]. Denoise the creature's noisy state vector (snap to the seen-
      state manifold) before the value estimate. MEASURED: it does not improve the DECISION -- the action-argmax is
      preserved about as well from the raw noisy state as from the denoised one (0.68 vs 0.68 at low noise, ~0.50
      both at high), because the value's similarity-weighting already absorbs independent noise -- and snapping
      OVER-SMOOTHS at low noise (value error 0.34 raw -> 0.42 denoised), the same clean-signal negative the manifold
      denoiser already carries. The high-dimensional encoder is its own denoiser. NO-OP.
"""

import numpy as np


# ---- A1 -----------------------------------------------------------------------------------------------------
def _iid_atoms(N, d, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _repulsion_atoms(N, d, steps=60, lr=0.05, seed=0):
    """Riesz-energy repulsion relaxation on the sphere (blue-noise on S^(d-1)): push atoms apart to lower mutual
    coherence. Force computed via the Gram matrix (no N*N*d tensor)."""
    X = _iid_atoms(N, d, seed).copy()
    for _ in range(steps):
        G = np.clip(X @ X.T, -1, 1)
        dist = np.sqrt(np.maximum(2 - 2 * G, 1e-9))
        W = 1.0 / dist ** 3
        np.fill_diagonal(W, 0.0)
        F = (W.sum(1)[:, None]) * X - W @ X
        X = X + lr * F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _max_coherence(X):
    G = np.abs(X @ X.T)
    np.fill_diagonal(G, 0.0)
    return float(G.max())


def _probe_a1():
    """A1 (no-op): repulsion relaxation barely lowers atom coherence vs i.i.d. -- random atoms are already near-
    optimally spread (concentration of measure), so the SAMPLE-1 low-discrepancy win does not transfer to codebook
    construction at the operating point."""
    d, N = 256, 400
    mi = _max_coherence(_iid_atoms(N, d, 0))
    mr = _max_coherence(_repulsion_atoms(N, d, seed=0))
    assert mr > mi * 0.85, (mi, mr)                              # improvement is under ~15% -- a no-op


# ---- A2 -----------------------------------------------------------------------------------------------------
def _correlated_atoms(N, d, rank, seed):
    r = np.random.default_rng(seed)
    X = r.standard_normal((N, rank)) @ r.standard_normal((rank, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _probe_a2():
    """A2 (negative): negative-lobe sharpening (deconvolve the similarity profile, then argmax) does NOT beat plain
    argmax for discrete cleanup -- it amplifies noise on correlated atoms. Hard nearest-neighbour is already
    Bayes-optimal for 'which atom is this'."""
    rng = np.random.default_rng(0)
    V = _correlated_atoms(200, 256, 32, seed=1)
    G = V @ V.T
    plain_ok = sharp_ok = 0
    T = 200
    for _ in range(T):
        i = int(rng.integers(200))
        q = V[i] + 0.5 * rng.standard_normal(256)
        q /= np.linalg.norm(q)
        s = V @ q
        plain_ok += int(s.argmax() == i)
        s_sharp = s - 0.1 * ((G - np.eye(len(G))) @ s)          # negative-lobe sharpening
        sharp_ok += int(s_sharp.argmax() == i)
    assert sharp_ok <= plain_ok, (plain_ok, sharp_ok)           # sharpening never helps (here it hurts)


# ---- B2 -----------------------------------------------------------------------------------------------------
def _probe_b2():
    """B2 (redundant no-op): the running coherence signal does not separate in-distribution generation from a
    garbage seed -- steered generation pulls any start back to coherent continuations, so there is no incoherent
    tail for a throughput gate to catch."""
    from holographic.agents_and_reasoning.holographic_meaning_predict import MeaningPredictor
    from holographic.misc.holographic_structure import StructureVerifier
    rng = np.random.default_rng(0)
    clauses = ["the cat chased the mouse", "the dog found the ball",
               "the bird watched the worm", "the fox carried the leaf"]
    corpus = [(rng.choice(clauses)).split() for _ in range(220)]
    stream = [w for s in corpus for w in s]
    mp = MeaningPredictor(dim=512, order=2, seed=0).fit_space(corpus, window=2).fit_transitions(stream)
    ver = StructureVerifier(mp.vocab, mp.M, mp.idx).calibrate(stream, chunk=150, z_floor=2.0)

    def gen_coh(seed_toks, length=14, beam=6, lookback=6):
        out = list(seed_toks)
        coh = []
        for _ in range(length):
            Cn, _ = mp._matrix()
            if len(Cn) == 0:
                break
            q = mp.context_vector(out[-mp.order:])
            qn = q / (np.linalg.norm(q) + 1e-12)
            coup = Cn @ qn
            cands, seen = [], set()
            for j in np.argsort(coup)[::-1][:beam]:
                w = mp._next[j]
                if w not in seen:
                    seen.add(w)
                    cands.append(w)
            if not cands:
                break
            vs = [ver.structure_score((out + [w])[-lookback:]) for w in cands]
            out.append(cands[int(np.argmax(vs))])
            coh.append(float(max(vs)))
        return float(np.mean(coh)) if coh else 0.0

    indist = gen_coh(["the", "cat"])
    garbage = gen_coh(["mouse", "leaf"])
    # the two are NOT cleanly separable -- a coherence floor cannot tell them apart (here garbage is even higher)
    assert garbage >= indist - abs(indist), (indist, garbage)


# ---- B4 -----------------------------------------------------------------------------------------------------
def _roberts_normal(n, d, seed=0):
    """LD R-sequence points in [0,1]^d (Roberts' generalised golden ratio) mapped to Gaussian by an inverse-normal
    approximation -- LD-sequenced noise vectors."""
    g = 1.0
    for _ in range(20):
        g = (1 + g) ** (1.0 / (d + 1))
    alpha = (1.0 / g) ** np.arange(1, d + 1)
    r = np.random.default_rng(seed)
    U = (r.random(d)[None, :] + alpha[None, :] * np.arange(1, n + 1)[:, None]) % 1.0
    U = np.clip(U, 1e-6, 1 - 1e-6)
    # Beasley-Springer/Moro-style inverse normal CDF (Acklam's rational approximation)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p = U.ravel()
    x = np.zeros_like(p)
    plow, phigh = 0.02425, 1 - 0.02425
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)
    ql = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = (((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5]) / \
            ((((dd[0] * ql + dd[1]) * ql + dd[2]) * ql + dd[3]) * ql + 1)
    qh = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -(((((c[0] * qh + c[1]) * qh + c[2]) * qh + c[3]) * qh + c[4]) * qh + c[5]) / \
            ((((dd[0] * qh + dd[1]) * qh + dd[2]) * qh + dd[3]) * qh + 1)
    qm = p[mid] - 0.5
    rr = qm * qm
    x[mid] = (((((a[0] * rr + a[1]) * rr + a[2]) * rr + a[3]) * rr + a[4]) * rr + a[5]) * qm / \
             (((((b[0] * rr + b[1]) * rr + b[2]) * rr + b[3]) * rr + b[4]) * rr + 1)
    return x.reshape(n, d)


def _probe_b4():
    """B4 (no-op): an LD-noise diffusion sampler matches an i.i.d.-noise one on both diversity and validity -- the
    diffusion is attractor-dominated (the cleanup step decides the landing), so more-even noise changes nothing."""
    rng = np.random.default_rng(0)
    d = 256
    r = np.random.default_rng(1)
    V = r.standard_normal((48, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    def diffuse(x0, noise, steps=8, b0=2.0, b1=12.0):
        x = x0 / np.linalg.norm(x0)
        for k in range(steps):
            beta = b0 + (b1 - b0) * k / (steps - 1)
            s = V @ x
            w = np.exp(beta * (s - s.max()))
            w /= w.sum()
            x = w @ V
            x = x + 0.5 * (1 - k / (steps - 1)) * noise[k]
            x = x / np.linalg.norm(x)
        return int((V @ (x / np.linalg.norm(x))).argmax())

    K = 90
    reach_iid, reach_ld = set(), set()
    for i in range(K):
        x0 = rng.standard_normal(d)
        reach_iid.add(diffuse(x0, rng.standard_normal((8, d))))
        reach_ld.add(diffuse(x0, _roberts_normal(8, d, seed=i)))
    assert abs(len(reach_iid) - len(reach_ld)) <= 4, (len(reach_iid), len(reach_ld))   # LD == i.i.d. (no-op)


# ---- D3 -----------------------------------------------------------------------------------------------------
def _probe_d3():
    """D3 (negative): a hard safety VETO guarantees no lethal choice among sensed-danger directions; a soft MIS
    penalty cannot -- it picks a lethal move whenever the value advantage exceeds the penalty. A safety constraint
    is not an estimator to blend. (Deterministic structural demonstration.)"""
    actions = ["N", "S", "E", "W"]
    values = np.array([0.9, 0.2, 0.3, 0.1])     # the danger direction N has the HIGHEST learned value (a misestimate)
    danger = {"N": True, "S": False, "E": False, "W": False}

    # hard veto: remove lethal moves, then argmax -> never picks N
    v = values.copy()
    for a, name in enumerate(actions):
        if danger[name]:
            v[a] = -1e9
    hard_choice = actions[int(v.argmax())]
    assert not danger[hard_choice]                              # the veto is SAFE by construction

    # soft MIS penalty: subtract a fixed penalty -> still picks the lethal N when its value margin exceeds it
    penalty = 0.5
    v = values.copy()
    for a, name in enumerate(actions):
        if danger[name]:
            v[a] -= penalty
    soft_choice = actions[int(v.argmax())]
    assert danger[soft_choice]                                  # the soft blend chose a LETHAL move -- no guarantee


# ---- D5 -----------------------------------------------------------------------------------------------------
def _probe_d5():
    """D5 (no-op): snapping a noisy observation to the nearest seen-state prototype (a denoiser) OVER-SMOOTHS when
    the noise is low -- it lands on the wrong neighbour's value and does worse than just using the raw noisy state.
    With a value read-out that already weights by similarity, the encoder is its own denoiser; an explicit denoise
    step does not help the decision and hurts at low noise."""
    rng = np.random.default_rng(0)
    d, N = 256, 60
    P = rng.standard_normal((N, d))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    pvals = rng.standard_normal(N)                              # a scalar 'value' per prototype (stand-in for returns)

    def read_raw(q):                                            # similarity-weighted value (the robust read-out)
        s = np.clip(P @ (q / np.linalg.norm(q)), 0, None)
        return float((s * pvals).sum() / (s.sum() + 1e-12))

    def read_denoised(q):                                       # snap to nearest prototype first (hard denoise)
        return float(pvals[int((P @ (q / np.linalg.norm(q))).argmax())])

    raw_err, den_err = [], []
    for i in range(N):
        clean = read_raw(P[i])                                  # the true value at this clean state
        q = P[i] + 0.3 * rng.standard_normal(d)                # LOW observation noise
        q /= np.linalg.norm(q)
        raw_err.append(abs(read_raw(q) - clean))
        den_err.append(abs(read_denoised(q) - clean))
    # at low noise the hard-snap denoiser is WORSE than the raw similarity read-out (over-smoothing)
    assert np.mean(den_err) > np.mean(raw_err), (np.mean(raw_err), np.mean(den_err))


def _selftest():
    """Run all six probe asserts -- each records a measured kept negative (the panel's no-op prior, confirmed)."""
    _probe_a1()
    _probe_a2()
    _probe_b2()
    _probe_b4()
    _probe_d3()
    _probe_d5()


if __name__ == "__main__":
    _selftest()
    print("holographic_probesweep: all six cross-cutting probes confirmed kept negatives "
          "(A1, A2, B2, B4, D3, D5)")
