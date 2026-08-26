"""holographic_projector.py -- PROJECT the codebase into VSA/installed form by MEASUREMENT (F34).

Moose's objection to hand-rolling every faculty as an "installable version" was correct, and the
tree already contained the answer: vminstall never translated the gather unit -- it MEASURED it
(T @ r, cosine 1.000000 on the live stream). Probing IS projection. This module generalizes that
into an automatic projector with honest refusal, three tiers cheapest-first:

  T1 PROBE (automatic, exact): a fixed-shape linear/affine core is measured into its matrix --
     columns = f(basis vectors) minus the affine offset -- and CERTIFIED on held-out random inputs.
     The certificate is the verdict: residual < tol -> "installs as one matvec"; else REFUSED.
     VERIFIED premises (prep session): bind(key,.) 4.7e-16, unbind 4.5e-16, permute exactly 0.0
     (and the extracted bind operator IS the circulant the ISA says -- columns are rolls of column
     0); unit-normalize 1.6e+01 REFUSED, abs 1.5e+00 REFUSED. The refusals ARE the core/shell
     boundary, discovered by measurement instead of declared by docstring (F33 inverts direction:
     the probe's verdict is ground truth, the docstring records it).
     STRUCTURE DETECTION (Quilez: store the rule, not the D^2 bytes): if the extracted matrix is a
     CIRCULANT (columns are rolls of column 0) the projector returns kind='circulant' with just the
     first column -- D floats instead of D^2, and the ISA's bind form recovered by measurement.
     Likewise 'permutation' (a 0/1 matrix with one 1 per row/col) stores just the index map.
  T2 FOLD (mechanical, given the F33 step shape): compiles step(state, x) faculties to REPEAT
     programs -- lives with the F27/F28 compiled-program milestone, NOT here (declared, not built:
     building it apart from the conformance program it exists to serve would be scaffolding).
  T3 APPLY (universal fallback): anything refused wraps as an APPLY step -- callable FROM a VSA
     program, honestly NOT installed; control stays runtime-side.

Probing cost is priced in the certificate: D calls + D^2 transient (collapsed to D when structure
is found). Dense probing a slow faculty is real money -- the certificate reports probe seconds so
the caller can decide, which is the machine-model setup-vs-marginal question yet again.
"""
import numpy as np
import time


def _try_rmsnorm(f, dim, n_check, tol, seed, worst, secs, scale=1.0):
    """G7 -- the first HOST-VOCABULARY target. Linear probing refused, but a transformer host OWNS
    normalization layers: fit y = g * x / rms(x) (gain from probe medians), certify on HELD-OUT
    inputs like every other kind. normalize (x/||x||) IS rmsnorm with constant gain -- the tree's
    most-refused function becomes installable the moment the target vocabulary matches the host's.
    Tracr (Lindner et al. 2023), the constructive prior art, compiles WITHOUT layer norm;
    certifying INTO normalization layers is precisely the lane it leaves open."""
    # scale-aware like the linear probe (the threshold op certified 'rmsnorm' on the ZERO
    # function at unit scale -- the same instrument lie, one fallback deeper)
    rng2 = np.random.default_rng(seed + 1)
    G = []
    for x in scale * rng2.standard_normal((12, dim)):
        y = np.asarray(f(x), float).reshape(-1)
        rms = np.sqrt(np.mean(x * x))
        with np.errstate(divide="ignore", invalid="ignore"):
            G.append(y * rms / x)
    g = np.median(np.stack(G), axis=0)
    if not np.all(np.isfinite(g)):
        return None
    worst_n = 0.0
    for _ in range(n_check):
        x = scale * rng2.standard_normal(dim)
        y = np.asarray(f(x), float).reshape(-1)
        yh = g * x / np.sqrt(np.mean(x * x))
        worst_n = max(worst_n, float(np.linalg.norm(y - yh) / (np.linalg.norm(y) + 1e-12)))
    if worst_n < max(tol, 1e-6):
        return {"kind": "rmsnorm", "gain": g, "residual": worst_n, "seconds": secs}
    return None


def _try_gated_elementwise(f, dim, n_check, tol, seed, scale=1.0):
    """H2 -- the SwiGLU-family HOST-VOCABULARY target (the customer is NAMED: Qwen3.5's blocks
    are 'SwiGLU activations, RMSNorm' -- the host owns this shape). Certify y_i = a_i * x_i *
    sigmoid(b_i * x_i): per-channel silu-with-gain/slope, the activation inside every SwiGLU
    block. Two gates before fitting: (1) ELEMENTWISE-ness -- perturbing channel j must move only
    output j (a cheap structural test that rejects mixing maps immediately); (2) the fit itself
    must certify on HELD-OUT inputs at the caller's scale, like every kind. Fit: K scaled probes
    give (x_i, y_i) samples per channel; b_i by log-grid + Newton polish, a_i analytic given b_i.
    SCOPE, honest: this certifies the ACTIVATION. The full SwiGLU block silu(Wg x) * (Wu x) is a
    TWO-BRANCH product -- the branches' linears certify, the product is host structure."""
    rng = np.random.default_rng(seed + 7)
    s = float(scale)
    # gate 1: elementwise-ness
    x0 = s * rng.standard_normal(dim)
    y0 = np.asarray(f(x0), float).reshape(-1)
    if y0.shape[0] != dim:
        return None
    for j in rng.choice(dim, size=min(4, dim), replace=False):
        x1 = x0.copy(); x1[j] += 0.37 * s
        d = np.abs(np.asarray(f(x1), float).reshape(-1) - y0)
        others = np.delete(d, j)
        if others.size and others.max() > 1e-10 * max(1.0, np.abs(d[j])):
            return None
    # samples per channel from K probes
    K = 24    # thick sample: thin per-channel draws left slope unidentifiable on unlucky
              # channels (dim=48 refused while dim=8 fit exactly -- one bad channel poisons worst)
    X = s * rng.standard_normal((K, dim))
    Y = np.stack([np.asarray(f(x), float).reshape(-1) for x in X])
    def sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    a = np.zeros(dim); b = np.zeros(dim)
    grid = np.concatenate([[0.0], np.logspace(-3, 2, 26), -np.logspace(-3, 2, 26)])
    for i in range(dim):
        xi, yi = X[:, i], Y[:, i]
        best = (np.inf, 0.0, 0.0)
        for bb in grid:
            g = xi * sig(bb * xi)
            den = float(g @ g)
            aa = float(yi @ g) / den if den > 1e-30 else 0.0
            e = float(np.sum((yi - aa * g) ** 2))
            if e < best[0]:
                best = (e, aa, bb)
        _, aa, bb = best
        def _err(bb2):
            g2 = xi * sig(bb2 * xi)
            den2 = float(g2 @ g2)
            aa2 = float(yi @ g2) / den2 if den2 > 1e-30 else 0.0
            return float(np.sum((yi - aa2 * g2) ** 2)), aa2
        e_cur, aa = _err(bb)
        for _ in range(40):                       # guarded Newton: a step must DECREASE the error
            g = xi * sig(bb * xi)                 # or it is rejected with halving -- an unguarded
            dg = xi * xi * sig(bb * xi) * (1 - sig(bb * xi))   # polish WORSENED unlucky channels
            r = yi - aa * g
            j1 = float(-aa * (r @ dg)); j2 = float(aa * aa * (dg @ dg))
            if j2 < 1e-30:
                break
            step = j1 / j2
            for _h in range(8):
                e_new, aa_new = _err(bb - step)
                if e_new < e_cur:
                    bb -= step; e_cur, aa = e_new, aa_new
                    break
                step *= 0.5
            else:
                break
        a[i], b[i] = aa, bb
    worst = 0.0
    for _ in range(n_check):
        x = s * rng.standard_normal(dim)
        y = np.asarray(f(x), float).reshape(-1)
        yh = a * x * sig(b * x)
        worst = max(worst, float(np.linalg.norm(y - yh) / (np.linalg.norm(y) + 1e-12)))
    if worst < max(tol, 1e-6):
        return {"kind": "gated", "gain": a, "slope": b, "residual": worst, "seconds": 0.0}
    return None


def _try_powerlaw(f, dim, n_check, tol, seed, scale=1.0):
    """ENGINE-SIDE vocabulary growth (the H2 burn-down's honest ending): render tone maps --
    gamma, sqrt-tone -- are elementwise ODD power laws y_i = s_i * sign(x_i) * |x_i|^g_i,
    diagonal in log-magnitude space, so two probe magnitudes per channel recover (g_i, s_i)
    exactly and held-out inputs certify it. Engine-native like circulant (apply computes the
    power directly); NOT claimed as a transformer-host op -- that would be dishonest, and the
    manifest kind says which side runs it. gated/silu stays with _try_gated: a power fit on a
    sigmoid fails held-out (pinned)."""
    rng = np.random.default_rng(seed + 5)
    x1 = np.full(dim, 0.5 * scale)
    x2 = np.full(dim, 2.0 * scale)
    try:
        y1 = np.asarray(f(x1), float).reshape(-1)
        y2 = np.asarray(f(x2), float).reshape(-1)
    except Exception:
        return None
    if y1.shape[0] != dim or np.any(y1 <= 0) is None:
        pass
    if y1.shape[0] != dim:
        return None
    with np.errstate(all="ignore"):
        g = np.log(np.abs(y2) / np.maximum(np.abs(y1), 1e-300)) / np.log(4.0)
        s = np.abs(y1) / np.maximum((0.5 * scale) ** g, 1e-300)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(s))):
        return None
    # odd-symmetry gate: a power law with sign carry must map -x to -y
    ym = np.asarray(f(-x1), float).reshape(-1)
    if not np.allclose(ym, -y1, rtol=1e-6, atol=1e-9):
        return None
    worst = 0.0
    for _ in range(n_check):
        # Cover several orders of magnitude, including the near-zero region.
        # The old 0.2..3.0-only probe mis-certified clip(x, -.1, .1) as the
        # exponent-zero law sign(x)*.1 because it never crossed the linear
        # center of the clip. A certificate is only as honest as its domain.
        mag = 10.0 ** rng.uniform(-3.0, np.log10(3.0), dim)
        x = mag * scale * rng.choice((-1.0, 1.0), dim)
        y = np.asarray(f(x), float).reshape(-1)
        yhat = s * np.sign(x) * np.abs(x) ** g
        worst = max(worst, float(np.linalg.norm(y - yhat) / (np.linalg.norm(y) + 1e-12)))
    if worst < tol:
        return {"kind": "powerlaw", "exponent": g, "gain": s, "residual": worst}
    return None


def probe_project(f, dim, n_check=24, tol=1e-8, seed=0, scale=1.0):
    """Measure callable `f: R^dim -> R^dim` into installed form, or refuse.

    Returns a dict: {'kind': 'circulant'|'permutation'|'dense'|'refused', 'residual': float,
    'seconds': float, and the payload -- 'column' (circulant), 'perm' (permutation),
    'matrix'+'offset' (dense affine), or nothing (refused)}. The residual is the max relative
    error over n_check held-out random inputs -- the certificate, not a hope."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    # scale: certify over the CALLER'S input range, not the probe's convenience. The image-op
    # sweep found the failure: (v>100)*255 certified 'circulant 0.0' because unit-scale probes
    # never crossed the threshold -- a perfect score on a nonlinear op was the INSTRUMENT lying
    # (probe range mismatch): the perfect-scores-are-instrument-hypotheses rule, again. Basis
    # probes and held-out checks both run at `scale`; payloads are normalized back so certified
    # operators stay scale-free. Certification is a claim ABOUT A DOMAIN; scale names the domain.
    s = float(scale)
    off = np.asarray(f(np.zeros(dim)), float).reshape(-1)
    out_dim = off.shape[0]          # RECTANGULAR maps certify too (G12 found the assumption:
    M = np.empty((out_dim, dim))    # image formation is 3 lights -> 64 pixels; a square-only
    I = np.eye(dim)                 # probe refused honest linear maps for a shape reason)
    for i in range(dim):
        M[:, i] = (np.asarray(f(s * I[i]), float).reshape(-1) - off) / s
    if not (np.all(np.isfinite(M)) and np.all(np.isfinite(off))):
        # NaN GATE (edge sweep finding): a NaN-producing map certified 'dense at 0.0 residual'
        # because Python's max(0.0, nan) keeps 0.0 -- NaN comparisons are False, so the refusal
        # threshold never fired. Non-finite probes are an unconditional refusal BEFORE any
        # threshold arithmetic: you cannot out-compare a NaN, you can only gate it.
        return {"kind": "refused", "residual": float("inf"), "seconds": time.perf_counter() - t0}
    worst = 0.0
    for _ in range(n_check):
        x = s * rng.standard_normal(dim)
        y = np.asarray(f(x), float).reshape(-1)
        yhat = M @ x + off
        worst = max(worst, float(np.linalg.norm(y - yhat) / (np.linalg.norm(y) + 1e-12)))
    secs = time.perf_counter() - t0
    if worst >= tol:
        r = _try_rmsnorm(f, dim, n_check, tol, seed, worst, secs, scale=s)
        if r is not None:
            return r
        r = _try_gated_elementwise(f, dim, n_check, tol, seed, scale=s)
        if r is not None:
            r["seconds"] = time.perf_counter() - t0
            return r
        r = _try_powerlaw(f, dim, n_check, tol, seed, scale=s)
        if r is not None:
            r["seconds"] = time.perf_counter() - t0
            return r
        return {"kind": "refused", "residual": worst, "seconds": secs}
    # structure detection, MOST SPECIFIC rule first -- caught by the selftest's own first run:
    # a cyclic shift is BOTH (roll matrices ARE circulants, column0 = a delta), so checking
    # circulant first swallowed the cheaper permutation form. Taxonomy rule: permutation (an
    # index map, D ints) before circulant (D floats) before dense (D^2).
    if out_dim == dim and np.allclose(off, 0.0, atol=1e-12):
        is_perm = (np.isin(M, (0.0, 1.0)).all()
                   and (M.sum(axis=0) == 1).all() and (M.sum(axis=1) == 1).all())
        if is_perm:
            return {"kind": "permutation", "perm": np.argmax(M, axis=0), "residual": worst, "seconds": secs}
        col0 = M[:, 0]
        if all(np.allclose(np.roll(col0, i), M[:, i], atol=1e-10) for i in range(1, dim, max(1, dim // 16))) \
           and np.allclose(np.stack([np.roll(col0, i) for i in range(dim)], axis=1), M, atol=1e-10):
            return {"kind": "circulant", "column": col0, "residual": worst, "seconds": secs}
    # G3 -- BLOCK-DIAGONAL (store the rule at the layer level): per-vertex transforms are the
    # SAME small block repeated down the diagonal (rigid transform of V vertices = one 3x3,
    # V times). Detect by scanning small divisors of dim; store ONE block: k*k params, not
    # dim^2 (measured below: a 40-vertex rigid transform certifies with 9+3 params, was
    # 14,400). Checked BEFORE dense, OUTSIDE the zero-offset guard
    # (a rigid transform's TRANSLATION is an offset -- the first pin run certified 'dense' and
    # taught exactly this; perm/circulant legitimately require zero offset, blockdiag does not).
    for k in (2, 3, 4, 6, 8):
        if out_dim == dim and dim % k == 0:
            blk = M[:k, :k]
            # off-diagonal blocks must be zero and every diagonal block equal to the first
            if np.allclose(M, np.kron(np.eye(dim // k), blk), atol=1e-10):
                return {"kind": "blockdiag", "block": blk, "offset": off,
                        "residual": worst, "seconds": secs}
    return _probe_return_dense(M, off, worst, secs)


def _probe_return_dense(M, off, worst, secs):
    return {"kind": "dense", "matrix": M, "offset": off, "residual": worst, "seconds": secs}


def cleanup_as_attention(codebook, beta=64.0):
    """G8 -- CLEANUP AS AN ATTENTION READ (the host's own mechanism): exact cleanup is
    argmax-then-fetch; a transformer expresses the same read as y = A^T softmax(beta * A x) --
    one attention head with the codebook as both keys and values. As beta grows the softmax
    sharpens toward the argmax winner. PRE-REGISTERED NEGATIVE (priced before shipping, not
    discovered after): softmax CANNOT express the lowest-index tie rule -- exactly tied scores
    average their rows at every finite beta, so ties are the permanent gap between the exact
    contract and the host's mechanism. The certificate below measures the agreement rate; ties
    are the residual's floor, by theorem not by bug."""
    A = np.asarray(codebook, float)
    def read(x, A=A, b=float(beta)):
        s = A @ x
        w = np.exp(b * (s - s.max()))
        w /= w.sum()
        return A.T @ w
    return read


def attention_read_certificate(codebook, queries, beta=64.0):
    """Measure the G8 read against EXACT cleanup on the caller's own queries: fraction whose
    attention output's nearest atom equals the exact argmax winner, at this beta. The honesty
    label for installing cleanup as a head -- same contract shape as measure_forest_recall."""
    A = np.asarray(codebook, float)
    read = cleanup_as_attention(A, beta)
    hits = 0
    for q in np.atleast_2d(np.asarray(queries, float)):
        exact = int(np.argmax(A @ q))
        y = read(q)
        hits += int(int(np.argmax(A @ y)) == exact)
    n = len(np.atleast_2d(queries))
    return {"agreement": hits / n, "beta": float(beta), "n": n}


def apply_projected(proj, x):
    """Run an installed form -- the matvec a layer (or the VM's opcode path) would perform.
    Circulant applies as an FFT-domain product (the SAME arithmetic as the ISA's bind)."""
    x = np.asarray(x, float).reshape(-1)
    k = proj["kind"]
    if k == "circulant":
        return np.fft.irfft(np.fft.rfft(proj["column"]) * np.fft.rfft(x), n=len(x))
    if k == "permutation":
        out = np.empty_like(x); out[proj["perm"]] = x  # column-index map: y[perm[i]] = x[i]
        return out
    if k == "blockdiag":
        b = proj["block"]; kk = b.shape[0]
        return (x.reshape(-1, kk) @ b.T).reshape(-1) + proj["offset"]
    if k == "rmsnorm":
        return proj["gain"] * x / np.sqrt(np.mean(x * x))
    if k == "powerlaw":
        return proj["gain"] * np.sign(x) * np.abs(x) ** proj["exponent"]
    if k == "gated":
        z = np.clip(proj["slope"] * x, -60, 60)
        return proj["gain"] * x / (1.0 + np.exp(-z))
    if k == "dense":
        return proj["matrix"] @ x + proj["offset"]
    raise ValueError("cannot apply a refused projection -- wrap it as an APPLY step (T3)")


def _selftest():
    from holographic.agents_and_reasoning.holographic_ai import bind, unbind
    rng = np.random.default_rng(3434)
    D = 128
    key = rng.standard_normal(D)

    # planted truth A: bind(key, .) projects as a CIRCULANT (D floats, not D^2), certified tiny
    p1 = probe_project(lambda x: bind(key, x), D)
    assert p1["kind"] == "circulant" and p1["residual"] < 1e-10, p1["kind"]
    x = rng.standard_normal(D)
    assert np.allclose(apply_projected(p1, x), bind(key, x), atol=1e-8), "installed bind must match live"

    # planted truth B: a roll projects as a PERMUTATION (an index map, not a matrix)
    p2 = probe_project(lambda v: np.roll(v, 7), D)
    assert p2["kind"] == "permutation" and np.allclose(apply_projected(p2, x), np.roll(x, 7)), p2["kind"]

    # planted truth C: a generic affine map stays DENSE and round-trips
    A = rng.standard_normal((D, D)) / np.sqrt(D); b = rng.standard_normal(D)
    p3 = probe_project(lambda v: A @ v + b, D)
    assert p3["kind"] == "dense" and np.allclose(apply_projected(p3, x), A @ x + b, atol=1e-8)

    # planted truth D (the boundary, discovered not declared -- AND MOVED BY DESIGN once): under
    # the matvec-only vocabulary, normalize was the canonical refusal; G7's host-vocabulary
    # extension certifies it as RMSNORM (asserted below), so the refusal exemplars here are maps
    # with NO host form in the vocabulary. abs stays refused; a refusal stays unusable.
    p5 = probe_project(np.abs, D)
    assert p5["kind"] == "refused", p5
    try:
        apply_projected(p5, x); raise AssertionError("applying a refusal must raise")
    except ValueError:
        pass

    # G3 pin: a 40-vertex rigid transform certifies BLOCKDIAG -- 9 block params + offset, not
    # 14,400 dense floats -- and the installed apply matches live math
    th = 0.37
    Rm = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    tv = np.array([0.5, -0.2, 1.0])
    def rigid_all(flat):
        V = flat.reshape(-1, 3)
        return ((V @ Rm.T) + tv).reshape(-1)
    pb = probe_project(rigid_all, 120)
    assert pb["kind"] == "blockdiag" and pb["block"].shape == (3, 3), pb["kind"]
    xf = rng.standard_normal(120)
    assert np.allclose(apply_projected(pb, xf), rigid_all(xf), atol=1e-10)

    # G7 pin: normalize -- the tree's most-refused function -- certifies RMSNORM against the
    # host vocabulary, tight, and the installed apply matches; clamp still refuses (no host form
    # fits it here -- the boundary moves only where measurement says it moves)
    pn = probe_project(lambda v: v / (np.linalg.norm(v) + 1e-12), D)
    assert pn["kind"] == "rmsnorm" and pn["residual"] < 1e-6, (pn["kind"], pn.get("residual"))
    # POWERLAW pins (H2's honest ending -- render tone maps certify engine-side): gamma 0.8 and
    # per-channel exponents certify at machine epsilon; silu must STILL route to gated (a power
    # fit on a sigmoid fails held-out); even-symmetric x^2 fails the odd gate and refuses --
    # the vocabulary grows without the taxonomy leaking.
    pg = probe_project(lambda v: np.sign(v) * np.abs(v) ** 0.8, 24)
    assert pg["kind"] == "powerlaw" and pg["residual"] < 1e-10
    xs = np.random.default_rng(3).standard_normal(24)
    assert np.max(np.abs(apply_projected(pg, xs) - np.sign(xs) * np.abs(xs) ** 0.8)) < 1e-12
    ps = probe_project(lambda v: v / (1 + np.exp(-v)), 24)
    assert ps["kind"] == "gated", ps["kind"]
    pe = probe_project(lambda v: v * v, 24)
    assert pe["kind"] == "refused", pe["kind"]
    xg = rng.standard_normal(D)
    live = xg / (np.linalg.norm(xg) + 1e-12)
    assert np.allclose(apply_projected(pn, xg), live, atol=1e-6)
    assert probe_project(lambda v: np.clip(v, -1, 1), D)["kind"] == "refused"

    # H2 PINS -- the GATED target (the NAMED customer: Qwen3.5's SwiGLU activation is silu):
    # the silu family certifies (plain, gained, sloped -- slope recovered to 2.0 exactly) and the
    # installed apply matches live; gelu_tanh REFUSES (genuinely outside a*x*sigmoid(b*x) --
    # kept negative: gelu is a DIFFERENT family, refuse rather than approximate silently); clamp
    # refuses; the elementwise GATE rejects mixing maps before any fitting happens.
    pg1 = probe_project(lambda v: v / (1.0 + np.exp(-v)), D)
    assert pg1["kind"] == "gated" and pg1["residual"] < 1e-10, (pg1["kind"], pg1.get("residual"))
    xg2 = rng.standard_normal(D)
    assert np.allclose(apply_projected(pg1, xg2), xg2 / (1.0 + np.exp(-xg2)), atol=1e-8)
    pg2 = probe_project(lambda v: v * (1 / (1 + np.exp(-2 * v))), D)
    assert pg2["kind"] == "gated" and abs(float(np.median(pg2["slope"])) - 2.0) < 1e-3
    assert probe_project(lambda v: 0.5 * v * (1 + np.tanh(0.79788456 * (v + 0.044715 * v ** 3))),
                         D)["kind"] == "refused", "gelu must refuse -- different family"
    assert probe_project(lambda v: v @ np.ones((D, D)) / D + v, D)["kind"] != "gated",         "mixing maps must not pass the elementwise gate"

    # conservation of meaning: installed unbind reproduces a recall round-trip at cosine ~1
    pu = probe_project(lambda v: unbind(v, key), D)
    tr = bind(key, x)
    r_live, r_inst = unbind(tr, key), apply_projected(pu, tr)
    cos = float(r_live @ r_inst / (np.linalg.norm(r_live) * np.linalg.norm(r_inst) + 1e-12))
    assert cos > 0.999999, cos

    print("OK: holographic_projector self-test passed (bind->circulant D floats, roll->permutation, "
          "affine->dense, nonlinear REFUSED with the refusal unusable, installed unbind cosine ~1)")


if __name__ == "__main__":
    _selftest()
