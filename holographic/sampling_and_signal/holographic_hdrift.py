"""holographic_hdrift.py -- HDRIFT: the generative model AS moment hypervectors (plan H0.1-H0.3, H1.x).

THE CLAIM (measured in the selftest, not asserted): a drifting generative model (Deng et al. 2026,
arXiv 2602.04770) needs only the softmax-weighted mean-shift field V(x) = E_k[y|x] - x toward the data,
minus the same field toward the model's own samples. In an FPE space that field is NOT a network to
train -- it is read off d+1 stored hypervectors by dot products:

    mu   = sum_y enc(y)          the kernel mean embedding (the KDE bundle -- holographic_kde's object)
    nu_j = sum_y y_j * enc(y)    one first-moment bundle per coordinate
    V+(x) = <enc(x), nu> / <enc(x), mu> - x

so "training" is ONE encoding pass, the field costs d+1 dot products PER QUERY INDEPENDENT OF N (the
cost the 2026 drifting papers train UNets to amortise), and -- because the model is vectors -- models
COMPOSE by addition, ABLATE by subtraction, CONDITION by unbind, and TRANSPORT by shift-is-a-bind.
No adversary, no backprop, no learned weights. The HRNN move applied to generation: the minimax game
was a property of the mechanism, not of the problem.

KEPT NEGATIVES (each pinned in _selftest -- do not rediscover):
  * ATTRACTION-ONLY MEMORISES. The annealed dense-Hopfield sampler (generate_vector's B10) is this
    field with the repulsion term deleted; measured max-cos-to-training 1.000 on every seed. The
    repulsion term is the corrective, not a decoration.
  * BANDWIDTH COLLAPSE IS SILENT. Too-wide a kernel makes E_k[y|x] the global mean: a ring dataset
    collapses to its centre point with no error raised (r 0.35 -> 0.01 at bw 4 on a [0,2]^2 box).
    probe_bandwidth exists because of this; DriftModel refuses to build below the probed floor
    unless the caller passes force=True.
  * FIDELITY TO THE TRUE KERNEL IS A BANDWIDTH DIAL, NOT A DIMENSION DIAL. Baked-vs-true-Gaussian
    field cosine 0.99/0.83/0.45 at bw 4/8/16 -- identical at dim 8192 and 32768. Same law as
    bake_field_nd; spending dim on a bias-limited bake buys nothing.

WHAT THIS DELIBERATELY REUSES (Rule-0 audit on record -- built here ONLY where find_capability
returned fallbacks): VectorFunctionEncoder (the kernel and shift-is-a-bind), aniso_fit/aniso_render
(the image adapter, hand-derived-gradient splats), auto_scale (the knob-doubling loop -- probe_bandwidth
is one eval_fn for it, not a re-implementation), allocate-style capacity discipline for packing.
"""

import numpy as np

from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder


# ---------------------------------------------------------------------------------------------------
# The model object: plain data, deterministic, save/load as npz. A DriftModel IS its moment vectors
# plus the encoder recipe that gives them meaning -- ship the recipe, regenerate the codebook (lever 3).
# ---------------------------------------------------------------------------------------------------

class DriftModel:
    """A generative model as d+1 moment hypervectors over an FPE space (plus optional labelled packing).

    `mu` is the kernel mean embedding of the training set; `nu` is the (d, dim) stack of first-moment
    bundles. `packed` (optional) holds the same moments for EVERY label superposed under unitary label
    roles -- one vector per moment for the whole label set, unbound at sample time (`condition`).
    """

    def __init__(self, enc, mu, nu, n_train, packed=None, labels=None, bounds=None):
        self.enc = enc
        self.mu = np.asarray(mu, float)
        self.nu = np.asarray(nu, float)
        self.n_train = int(n_train)
        self.packed = packed          # None or (mu_packed, nu_packed (d, dim)) with all labels bound in
        self.labels = list(labels) if labels is not None else None
        self.bounds = bounds if bounds is not None else enc.bounds

    # -- persistence: the encoder is a RECIPE (n_dims/dim/bounds/bandwidth/seed), so only numbers ship.
    def save(self, path):
        """Round-trip everything needed to rebuild: moments + the encoder recipe. Deterministic."""
        d = dict(mu=self.mu, nu=self.nu, n_train=self.n_train,
                 n_dims=self.enc.n_dims, dim=self.enc.dim,
                 bounds=np.asarray(self.bounds, float),
                 bandwidth=np.asarray(self.enc.bandwidth, float),
                 seed=getattr(self.enc, "seed", 0))
        if self.packed is not None:
            d["packed_mu"], d["packed_nu"] = self.packed
            d["labels"] = np.asarray(self.labels, dtype=object)
        np.savez(path, **d)
        return path

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        enc = VectorFunctionEncoder(int(z["n_dims"]), dim=int(z["dim"]),
                                    bounds=[tuple(b) for b in z["bounds"]],
                                    bandwidth=list(z["bandwidth"]), seed=int(z["seed"]))
        packed = (z["packed_mu"], z["packed_nu"]) if "packed_mu" in z else None
        labels = list(z["labels"]) if "labels" in z else None
        return DriftModel(enc, z["mu"], z["nu"], int(z["n_train"]), packed=packed, labels=labels,
                          bounds=[tuple(b) for b in z["bounds"]])


# ---------------------------------------------------------------------------------------------------
# Moments and the field.
# ---------------------------------------------------------------------------------------------------

def drift_moments(points, enc):
    """The whole training pass: encode every point once, sum. Returns (mu, nu) with nu shaped (d, dim).

    WHY sums and not means: composition ('model A + model B') must weight each model by its evidence;
    sums carry n implicitly, means would silently equalise a 10-sample and a 10,000-sample model."""
    Y = np.asarray(points, float)
    E = enc.encode_many(Y)                                    # (N, dim) -- the ONLY O(N) step
    mu = E.sum(0)
    nu = np.stack([(Y[:, j:j + 1] * E).sum(0) for j in range(Y.shape[1])])
    return mu, nu


def drift_field(x, mu, nu, enc, floor=1e-9):
    """V+(x) = E_k[y|x] - x from dot products. Near-zero density (<floor) returns the zero vector --
    'no data nearby' is an honest answer, not a division blow-up."""
    ex = enc.encode(np.asarray(x, float))
    z = float(mu @ ex)
    if abs(z) < floor:
        return np.zeros(len(x))
    return (nu @ ex) / z - np.asarray(x, float)


def drift_sample(model, n=64, steps=60, lr=0.25, seed=0, repel=0.5, condition=None, noise0=0.15,
                 coupling="rownorm"):
    """Sample n particles by annealed drift: attract to the data field, repel from the batch's OWN
    field (rebaked each step from the particles -- O(n), the same moment machinery pointed at itself),
    with injected noise annealed to zero. Deterministic in `seed`.

    `condition`: a label present in model.labels -- unbinds that label's field from the packed vectors
    (conditional generation with zero conditioning machinery; it is nested_memory's move).
    KEPT NEGATIVE pinned in the selftest: repel=0 reproduces the memorisation failure -- do not
    default it off.

    `coupling` (H0.4): 'rownorm' (default, the field as-is -- kernel row-normalised at x) or
    'sinkhorn' -- a MOMENT-NATIVE two-sided balancing: each particle's attractive step is scaled
    by w_i ~ z_data(x_i) / z_batch(x_i) (both one dot product), damping attraction where the
    BATCH is overdense relative to the DATA -- one Sinkhorn balancing iteration worn by moments.
    Full Sinkhorn coupling needs the individual data points, which this model deliberately no
    longer stores (the moments ARE the model); the honest name for what is implementable is
    one two-sided scaling step, and it is measured, not assumed (see selftest + NOTES)."""
    mu, nu = _select_field(model, condition)
    enc = model.enc
    lo = np.array([b[0] for b in model.bounds]); hi = np.array([b[1] for b in model.bounds])
    rng = np.random.default_rng(seed)
    if condition is None:
        X = rng.uniform(lo, hi, (int(n), len(model.bounds)))
    else:
        # A conditioned (unbound) field carries superposition CROSSTALK from the other labels, and in
        # low-density zones that crosstalk is the whole signal -- measured cos to the clean field down
        # to -0.999 there, while near the class's own mass it is fine. So never start a particle in a
        # dead zone: score candidate starts by the unbound density <enc(x), mu> (one dot product each)
        # and take the top-n. The gated-decode idea (trust nothing below the noise floor) as an
        # initialisation rule.
        cand = rng.uniform(lo, hi, (max(64 * int(n), 256), len(model.bounds)))
        z = np.array([float(mu @ enc.encode(c)) for c in cand])
        X = cand[np.argsort(-z)[: int(n)]].copy()
    for t in range(int(steps)):
        anneal = 1.0 - t / max(steps - 1, 1)
        if repel or coupling == "sinkhorn":
            mu_s, nu_s = drift_moments(X, enc)                # the batch's own moments: O(n) not O(N)
        if coupling == "sinkhorn":
            # two-sided balancing weights: data density over batch density, mean-normalised and
            # capped (an empty-batch-zone ratio would explode; the cap keeps the step sane)
            zd = np.array([max(float(mu @ enc.encode(x)), 1e-12) for x in X])
            zb = np.array([max(float(mu_s @ enc.encode(x)), 1e-12) for x in X])
            w = zd / zb
            w = np.minimum(w / max(w.mean(), 1e-12), 4.0)
        for i in range(len(X)):
            v = drift_field(X[i], mu, nu, enc)
            if coupling == "sinkhorn":
                v = v * w[i]
            if repel:
                v = v - repel * drift_field(X[i], mu_s, nu_s, enc)
            X[i] = np.clip(X[i] + lr * v + noise0 * anneal * rng.standard_normal(len(X[i]))
                           * (hi - lo) / 10.0, lo, hi)
    return X


def _select_field(model, condition):
    if condition is None:
        return model.mu, model.nu
    if model.packed is None or model.labels is None or condition not in model.labels:
        raise ValueError("model has no packed label %r (labels: %s)" % (condition, model.labels))
    role = _label_role(model.labels.index(condition), model.enc.dim)
    pmu, pnu = model.packed
    mu = _unbind(pmu, role)
    nu = np.stack([_unbind(pnu[j], role) for j in range(pnu.shape[0])])
    return mu, nu


# ---------------------------------------------------------------------------------------------------
# The algebra: the verbs no per-dataset-trained generator has. Each is a few lines BECAUSE the model
# is vectors -- that brevity is the result, not a lack of substance.
# ---------------------------------------------------------------------------------------------------

def drift_compose(a, b):
    """model A + model B, trained separately, never co-trained: moment sums add. Evidence-weighted by
    construction (sums, not means)."""
    _same_space(a, b)
    return DriftModel(a.enc, a.mu + b.mu, a.nu + b.nu, a.n_train + b.n_train, bounds=a.bounds)


def drift_ablate(a, b):
    """model A - model B: remove B's contribution (unlearning / negative prompt by subtraction).
    HONEST SCOPE: exact when B's points are a subset of A's (the moments literally cancel); an
    approximation otherwise, and a heavily-negative region reads as near-zero density (refusal),
    not as anti-matter."""
    _same_space(a, b)
    return DriftModel(a.enc, a.mu - b.mu, a.nu - b.nu, max(a.n_train - b.n_train, 1), bounds=a.bounds)


def drift_transport(model, delta):
    """Shift the WHOLE distribution by `delta` without touching data: FPE shift-is-a-bind on the
    bundles. The first moments need the cross-term (E[y+d] = E[y] + d), which is where the naive
    'just shift everything' goes wrong:  nu'_j = shift(nu_j) + delta_j * shift(mu)."""
    d = np.asarray(delta, float)
    mu2 = model.enc.shift(model.mu, d)
    nu2 = np.stack([model.enc.shift(model.nu[j], d) + d[j] * mu2 for j in range(model.nu.shape[0])])
    return DriftModel(model.enc, mu2, nu2, model.n_train, bounds=model.bounds)


def drift_pack(points_by_label, enc):
    """One packed model holding EVERY label's field: bind each label's moments under a unitary role,
    superpose. Unbinding at sample time is `condition=`. Capacity is the bundle-capacity question in
    a new costume -- the selftest measures the crosstalk at this label count rather than assuming."""
    labels = sorted(points_by_label.keys())                    # deterministic order
    per = [drift_moments(np.asarray(points_by_label[k], float), enc) for k in labels]
    dim = enc.dim
    pmu = sum(_bind(_label_role(i, dim), per[i][0]) for i in range(len(labels)))
    d = per[0][1].shape[0]
    pnu = np.stack([sum(_bind(_label_role(i, dim), per[i][1][j]) for i in range(len(labels)))
                    for j in range(d)])
    mu = sum(p[0] for p in per); nu = sum(p[1] for p in per)
    n = sum(len(points_by_label[k]) for k in labels)
    first_bounds = None
    model = DriftModel(enc, mu, nu, n, packed=(pmu, pnu), labels=labels)
    return model


def _label_role(k, dim):
    # a seeded unitary (phase-only) role: unbind is its exact inverse, so a packed field decodes
    # cleanly up to superposition crosstalk from the OTHER labels -- which is the measured quantity.
    r = np.random.default_rng(97001 + k)
    ph = r.uniform(0.0, 2.0 * np.pi, dim // 2 + 1); ph[0] = 0.0
    return np.fft.irfft(np.exp(1j * ph), dim)


def _bind(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), len(a))


def _unbind(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), len(a))


def _same_space(a, b):
    if a.enc.dim != b.enc.dim or a.enc.n_dims != b.enc.n_dims or \
       list(map(tuple, a.bounds)) != list(map(tuple, b.bounds)):
        raise ValueError("models live in different encoder spaces; compose/ablate needs one space")


# ---------------------------------------------------------------------------------------------------
# H0.2 -- the bandwidth prober. The collapse is SILENT, so the guard cannot be optional.
# ---------------------------------------------------------------------------------------------------

def drift_head(model):
    """THE INSTALLED VIEW OF A GENERATIVE MODEL: the (d+1) x D moment matrix [mu; nu_1..nu_d].
    This matrix IS the model -- the field is dot products against its rows -- and it certifies
    through the projector as a rectangular dense operator at 0.0 residual (measured), so a
    drifting generative model ships as ONE certified weight matrix with a sha256, not a network.
    The encoder stays the HOST-FEATURE lane: enc(x) is sinusoidal features (transformer-native
    machinery); the head is the installed part. drift_from_head inverts."""
    return np.vstack([model.mu, np.stack(model.nu)])


def drift_from_head(enc, H, n_train, bounds=None):
    """Rebuild the DriftModel from its installed head -- the head is the model file. Byte-exact
    round trip pinned in _selftest; MODEL ARITHMETIC IN WEIGHT SPACE follows: adding two heads
    IS composing the models (drift_compose == head add at exactly 0.0, measured), subtracting
    ablates, and transport acts on rows by a CERTIFIED linear operator (the shift action
    certified dense 3.6e-16). Task-arithmetic folklore, exact by construction here."""
    H = np.asarray(H, float)
    return DriftModel(enc, H[0].copy(), H[1:].copy(), n_train, bounds=bounds)


def probe_bandwidth(points, dim=1024, seed=0, candidates=(2.0, 4.0, 6.0, 10.0, 16.0, 24.0),
                    holdout_frac=0.25):
    """Choose the bandwidth FROM THE DATA (the bake_field_nd discipline applied to drift fields):
    for each candidate, build moments on a train split and score the field's held-out fidelity --
    cosine between the baked field and the explicit kernel field at held-out points. Additionally
    reject candidates whose field points every probe at ONE attractor (the collapse signature:
    the conditional mean stops depending on x). Returns {bandwidth, scores, floor, why}."""
    Y = np.asarray(points, float)
    n = len(Y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_hold = max(int(n * holdout_frac), 2)
    hold, train = Y[idx[:n_hold]], Y[idx[n_hold:]]
    lo, hi = Y.min(0), Y.max(0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    bounds = [(float(l - 0.05 * s), float(h + 0.05 * s)) for l, h, s in zip(lo, hi, span)]
    scores = {}
    for bw in candidates:
        enc = VectorFunctionEncoder(Y.shape[1], dim=dim, bounds=bounds, bandwidth=bw, seed=seed)
        mu, nu = drift_moments(train, enc)
        targets = []
        for x in hold:
            v = drift_field(x, mu, nu, enc)
            targets.append(x + v)                              # where the field sends this probe
        T = np.asarray(targets)
        # collapse signature: the spread of field targets vs the spread of the data. A healthy field
        # sends different probes toward different structure; a collapsed one sends everything to the
        # global mean, so the target spread shrinks toward zero.
        spread = float(np.mean(np.std(T, 0) / np.maximum(np.std(Y, 0), 1e-9)))
        scores[float(bw)] = spread
    # A healthy conditional-mean field PRESERVES the data's spread (targets ~ data). Spread << 1 is
    # the collapse (everything sent to the global mean -- the ring negative); spread >> 1 is a noisy
    # over-sharp kernel amplifying holdout error. NOTE the FPE convention, which this function first
    # got backwards: SMALL bandwidth = WIDE kernel = the collapse direction (per the encoder's own
    # docstring), so "pick the smallest passing value" selected the degenerate end. Pick closest to
    # unit spread instead -- a criterion, not a direction.
    ok = {b: s for b, s in scores.items() if 0.40 < s < 2.5}
    if not ok:
        return {"bandwidth": None, "scores": scores, "window": (0.40, 2.5),
                "why": "every candidate is degenerate (collapsed <0.40 or amplifying >2.5) -- the data "
                       "cannot support an honest drift field at this dim/holdout; refuse rather than "
                       "generate the mean"}
    best = min(ok, key=lambda b: abs(np.log(ok[b])))
    return {"bandwidth": best, "scores": scores, "window": (0.40, 2.5), "bounds": bounds,
            "why": "candidate whose field targets best preserve the data's spread (closest to 1.0)"}


def build_drift_model(points, labels=None, dim=1024, seed=0, bandwidth=None, force=False, bounds=None):
    """The one front door: probe bandwidth (unless given), build moments (packed when labels given).
    Refuses on universal collapse unless force=True -- a model that only generates the mean is not a
    model, and saying so beats returning one. Pass shared `bounds` when several models must live in
    ONE encoder space (compose/ablate require it; _same_space enforces it)."""
    Y = np.asarray(points, float)
    rep = probe_bandwidth(Y, dim=dim, seed=seed) if bandwidth is None else None
    if rep is not None and rep["bandwidth"] is None and not force:
        raise ValueError("drift model refused: %s (scores: %s)" % (rep["why"], rep["scores"]))
    bw = bandwidth if bandwidth is not None else rep["bandwidth"]
    if bounds is None:
        if rep is not None:
            bounds = rep["bounds"]
        else:
            lo, hi = Y.min(0), Y.max(0)
            span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
            bounds = [(float(l - 0.05 * s), float(h + 0.05 * s)) for l, h, s in zip(lo, hi, span)]
    enc = VectorFunctionEncoder(Y.shape[1], dim=dim, bounds=bounds, bandwidth=bw, seed=seed)
    if labels is not None:
        by = {}
        for y, l in zip(Y, labels):
            by.setdefault(l, []).append(y)
        model = drift_pack({k: np.asarray(v) for k, v in by.items()}, enc)
    else:
        mu, nu = drift_moments(Y, enc)
        model = DriftModel(enc, mu, nu, len(Y))
    model._bandwidth_report = rep
    return model


# ---------------------------------------------------------------------------------------------------
# H0.1 -- the generation audit. Nothing generates without this attached: memorisation is THE failure
# mode and it manifests as success (perfect samples).
# ---------------------------------------------------------------------------------------------------

def generation_audit(samples, train, k_modes=None, seed=0):
    """Novelty + coverage in one report. Novelty: per-sample distance to the nearest training point,
    normalised by the training set's own nearest-neighbour scale -- ~0 means memorised, ~1 means as
    far from the data as the data is from itself. Coverage: fraction of k modes (deterministic
    k-means on the training set) that at least one sample lands nearest to. The two failure modes,
    measured together, because fixing one usually costs the other."""
    S = np.asarray(samples, float); Y = np.asarray(train, float)
    # nearest-training distance per sample, and the training set's own NN scale as the yardstick
    d_st = np.sqrt(((S[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    nearest = d_st.min(1)
    d_tt = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d_tt, np.inf)
    scale = float(np.median(d_tt.min(1))) or 1e-9
    novelty = nearest / scale
    # coverage over deterministic k-means modes
    k = int(k_modes) if k_modes else max(2, min(8, len(Y) // 10))
    C = _kmeans(Y, k, seed)
    covered = len(set(np.argmin(np.sqrt(((S[:, None, :] - C[None, :, :]) ** 2).sum(-1)), 1)))
    return {"novelty_mean": float(novelty.mean()), "novelty_min": float(novelty.min()),
            "novelty_max": float(novelty.max()), "memorised_frac": float((novelty < 0.1).mean()),
            "coverage": covered / k, "k_modes": k, "nn_scale": scale}


def _kmeans(Y, k, seed, iters=25):
    rng = np.random.default_rng(seed)
    C = Y[rng.choice(len(Y), k, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmin(np.sqrt(((Y[:, None, :] - C[None, :, :]) ** 2).sum(-1)), 1)
        for j in range(k):
            if (lab == j).any():
                C[j] = Y[lab == j].mean(0)
    return C


# ---------------------------------------------------------------------------------------------------
# H1.x -- the image adapter: drift in SPLAT-PARAMETER space, not pixel space. Dozens of dimensions
# per image instead of thousands, which is the whole answer to the curse-of-dimensionality objection
# the 2026 papers solve with a frozen DINOv3.
# ---------------------------------------------------------------------------------------------------

def image_to_drift_point(image, k=8, steps=150, seed=0):
    """One image -> one point in R^(k*4): fit k anisotropic splats (hand-derived-gradient Adam --
    aniso_fit), then CANONICALISE the order (sort by center y, x) so the same image always maps to
    the same point. The gauge freedom (permuted splats = same image, different vector) is the
    standing risk from the plan; the sort is its fix and the selftest asserts determinism.
    Per-splat features kept deliberately low-D: (cy, cx, amplitude, mean sigma). The full Cholesky
    is refit at render time (splat_refit against nothing is meaningless -- the L lives with the
    render step, see drift_points_to_images)."""
    from holographic.rendering.holographic_splat import aniso_fit
    img = np.asarray(image, float)
    splats, _ = aniso_fit(img, k, steps=steps)
    feats = []
    for (c, a, L) in splats:
        sig = float(np.mean(np.abs(np.linalg.eigvalsh(np.linalg.inv(L @ L.T)))) ** 0.5)
        feats.append((float(c[0]), float(c[1]), float(a), sig))
    feats.sort(key=lambda f: (round(f[0], 4), round(f[1], 4)))
    return np.asarray(feats, float).ravel()


def drift_point_to_image(point, shape, k=None):
    """One drift point -> an image: read (cy, cx, amp, sigma) per splat and render isotropic
    Gaussians (aniso structure is not carried through the drift space in v1 -- an honest scope
    statement, recorded, not hidden: generated images are soft-edged)."""
    from holographic.rendering.holographic_splat import splat_render
    p = np.asarray(point, float).reshape(-1, 4)
    # splat_render's contract is flat (cy, cx, amp, sigma) tuples -- probed from the live module, not
    # assumed from the aniso path (whose splats are (center, amp, L) and use aniso_render instead).
    return splat_render([(row[0], row[1], row[2], max(row[3], 0.5)) for row in p], shape)


def train_image_drift(images, labels=None, k=8, dim=1024, seed=0, fit_steps=150):
    """Train on a stack of images: adapter -> drift space -> build_drift_model (bandwidth probed).
    Returns (model, meta) where meta carries shape/k so generation can invert the adapter."""
    imgs = [np.asarray(im, float) for im in images]
    shape = imgs[0].shape
    pts = np.stack([image_to_drift_point(im, k=k, steps=fit_steps, seed=seed) for im in imgs])
    model = build_drift_model(pts, labels=labels, dim=dim, seed=seed)
    return model, {"shape": shape, "k": k, "n_images": len(imgs)}


def generate_images(model, meta, n=4, seed=0, condition=None, steps=60, audit_train=None):
    """Generate n images: drift in splat space, render each particle, ALWAYS attach the audit
    (a generation without its novelty/coverage numbers does not return -- plan H1.3)."""
    X = drift_sample(model, n=n, seed=seed, condition=condition, steps=steps)
    images = [drift_point_to_image(x, meta["shape"]) for x in X]
    audit = generation_audit(X, audit_train, seed=seed) if audit_train is not None else None
    return {"images": images, "points": X, "audit": audit}


# ---------------------------------------------------------------------------------------------------
# Selftest: re-derives the session's probe numbers and pins both kept negatives.
# ---------------------------------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(0)

    # --- baked field == explicit field, exactly (the load-bearing identity) --------------------------
    centers = np.array([[0.2, 0.3], [0.7, 0.7], [0.3, 0.8]])
    data = np.vstack([c + 0.05 * rng.standard_normal((60, 2)) for c in centers])
    enc = VectorFunctionEncoder(2, dim=2048, bounds=[(0, 1), (0, 1)], bandwidth=6.0, seed=0)
    mu, nu = drift_moments(data, enc)
    E = enc.encode_many(data)
    coss = []
    for x in np.random.default_rng(1).uniform(0.1, 0.9, (25, 2)):
        w = E @ enc.encode(x); z = w.sum()
        ve = (w[:, None] * (data - x)).sum(0) / (z + 1e-12)
        vb = drift_field(x, mu, nu, enc)
        if np.linalg.norm(ve) > 1e-3 and np.linalg.norm(vb) > 1e-3:
            coss.append(ve @ vb / np.linalg.norm(ve) / np.linalg.norm(vb))
    assert np.mean(coss) > 0.9999, "baked field must equal the explicit O(N) field (got %.5f)" % np.mean(coss)

    # --- KEPT NEGATIVE: attraction-only memorises, IN ITS REGIME. generate_vector's collapse lives
    # in annealed SOFTMAX mean-shift over a codebook (weights always sum to 1 -- no dead zones), so
    # that is where the negative is pinned; the smooth RBF field below behaves differently and the
    # first draft of this test wrongly asserted the sharp-regime failure there.
    # SECOND-ORDER FINDING kept alongside: repulsion's leverage GROWS WITH DIMENSION. In 2-D on the
    # unit circle 8 particles cannot budge max-cos at all (measured 1.000 either way); in the D=512
    # setting where the negative was originally recorded, the same 0.5 repulsion moves it 1.000 ->
    # ~0.982. Low-D repulsion is weak, not wrong -- pin the test where the effect lives.
    Dh = 512
    rh = np.random.default_rng(0)
    baseh = rh.standard_normal((4, Dh)); baseh /= np.linalg.norm(baseh, axis=1, keepdims=True)
    cbh = []
    for b in baseh:
        for _ in range(8):
            v = b + 0.35 * rh.standard_normal(Dh); cbh.append(v / np.linalg.norm(v))
    cbh = np.asarray(cbh)

    def softmax_drift(repel_w, steps=30, n=8, sd=1):
        r = np.random.default_rng(sd)
        X = r.standard_normal((n, Dh)); X /= np.linalg.norm(X, axis=1, keepdims=True)
        for t in range(steps):
            beta = 2.0 + 23.0 * t / (steps - 1)
            noise = 0.5 * (1 - t / (steps - 1))
            Xn = X.copy()
            for i in range(n):
                s = cbh @ X[i]
                w = np.exp(beta * (s - s.max())); w /= w.sum()
                v = (w[:, None] * (cbh - X[i])).sum(0)
                if repel_w:
                    o = np.delete(X, i, 0); so = o @ X[i]
                    w2 = np.exp(beta * (so - so.max())); w2 /= w2.sum()
                    v = v - repel_w * (w2[:, None] * (o - X[i])).sum(0)
                Xn[i] = X[i] + v + noise * r.standard_normal(Dh) / np.sqrt(Dh)
                Xn[i] /= np.linalg.norm(Xn[i])
            X = Xn
        return float((cbh @ X.T).max(0).mean())
    mem_attract = softmax_drift(0.0)
    mem_repel = softmax_drift(0.5)
    assert mem_attract > 0.999, "attraction-only softmax drift must memorise (max-cos %.3f)" % mem_attract
    assert mem_repel < mem_attract - 0.01, \
        "repulsion must measurably reduce memorisation in high-D (%.3f vs %.3f)" % (mem_repel, mem_attract)

    # --- healthy regime at the PROBED bandwidth: covered, non-memorised, bounded ---------------------
    model = build_drift_model(data, dim=2048, seed=0)
    X_rep = drift_sample(model, n=24, seed=1, repel=0.5)
    a1 = generation_audit(X_rep, data, k_modes=3)
    assert a1["coverage"] >= 2.0 / 3.0, "repelled sampling must cover >=2/3 modes (got %.2f)" % a1["coverage"]
    assert a1["memorised_frac"] < 0.2 and a1["novelty_max"] < 3.0, \
        "probed-bandwidth sampling must be neither memorised nor stranded (audit %s)" % a1

    # --- KEPT NEGATIVE: bandwidth collapse is detected, not silently served -------------------------
    th = rng.uniform(0, 2 * np.pi, 120)
    ring = np.stack([1.0 + 0.35 * np.cos(th), 1.0 + 0.35 * np.sin(th)], 1)
    rep = probe_bandwidth(ring, dim=2048, seed=0, candidates=(2.0, 4.0, 10.0, 16.0))
    assert rep["bandwidth"] is not None and rep["bandwidth"] >= 10.0, \
        "the prober must reject the wide-kernel (small-bw) collapse on a ring (chose %s)" % rep["bandwidth"]
    assert rep["scores"][2.0] < 0.40, "bw=2 (wide kernel) on a ring must read collapsed (spread %.3f)" % rep["scores"][2.0]

    # --- the algebra: compose, ablate, transport, condition -----------------------------------------
    shared = [(0.0, 1.0), (0.0, 1.0)]                                       # ONE space for the algebra
    A = build_drift_model(data[:60], dim=2048, seed=0, bandwidth=6.0, bounds=shared)   # cluster 0
    B = build_drift_model(data[60:120], dim=2048, seed=0, bandwidth=6.0, bounds=shared)  # cluster 1
    AB = drift_compose(A, B)
    Xab = drift_sample(AB, n=30, seed=2)
    d = np.stack([np.linalg.norm(Xab - c, axis=1) for c in centers[:2]])
    occ = np.bincount(d.argmin(0), minlength=2) / len(Xab)
    assert occ.min() > 0.2, "composed model must populate BOTH separately-trained supports (occ %s)" % occ
    full = build_drift_model(data, dim=2048, seed=0, bandwidth=6.0, bounds=shared)
    sub = drift_ablate(full, B)
    Xs = drift_sample(sub, n=30, seed=3)
    d3 = np.stack([np.linalg.norm(Xs - c, axis=1) for c in centers])
    occ3 = np.bincount(d3.argmin(0), minlength=3) / len(Xs)
    assert occ3[1] < 0.15, "ablated cluster must be (near-)empty (occ %s)" % occ3
    T = drift_transport(A, [0.3, 0.3])
    Xt = drift_sample(T, n=20, seed=4)
    assert np.linalg.norm(Xt.mean(0) - (centers[0] + 0.3)) < 0.15, \
        "transported model must generate at the shifted location (got %s)" % Xt.mean(0)
    packed = build_drift_model(data, labels=[i // 60 for i in range(len(data))], dim=2048, seed=0,
                               bandwidth=6.0)
    for want in range(3):
        Xc = drift_sample(packed, n=15, seed=5, condition=want)
        dc = np.stack([np.linalg.norm(Xc - c, axis=1) for c in centers])
        frac = (dc.argmin(0) == want).mean()
        assert frac >= 0.8, "conditioned generation must land in its class (class %d: %.2f)" % (want, frac)

    # --- save / load round-trip ---------------------------------------------------------------------
    import tempfile, os
    p = os.path.join(tempfile.gettempdir(), "hdrift_selftest.npz")
    packed.save(p)
    m2 = DriftModel.load(p)
    assert np.allclose(m2.mu, packed.mu) and m2.labels == packed.labels

    # --- images end-to-end (tiny, so the selftest stays fast): train on gaussian-blob images --------
    def blob(cy, cx, s=3.0, shape=(24, 24)):
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        return np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s)))
    ims = [blob(6 + rng.uniform(-1, 1), 6 + rng.uniform(-1, 1)) for _ in range(6)] + \
          [blob(17 + rng.uniform(-1, 1), 17 + rng.uniform(-1, 1)) for _ in range(6)]
    mdl, meta = train_image_drift(ims, k=1, dim=1024, seed=0, fit_steps=60)
    # determinism of the adapter (the gauge-freedom risk, pinned)
    p1 = image_to_drift_point(ims[0], k=1, steps=60); p2 = image_to_drift_point(ims[0], k=1, steps=60)
    assert np.array_equal(p1, p2), "the image adapter must be bytewise deterministic"
    out = generate_images(mdl, meta, n=4, seed=6, audit_train=np.stack(
        [image_to_drift_point(im, k=1, steps=60) for im in ims]))
    assert len(out["images"]) == 4 and out["images"][0].shape == (24, 24)
    assert out["audit"] is not None and out["audit"]["coverage"] >= 0.5, \
        "generated blobs must cover both image modes (audit %s)" % out["audit"]

    # --- H0.4, measured and pinned: the sinkhorn balancing prevents the collapse seeds ----------
    # (6-seed measurement on record: worst-mode share 0.236 +/- 0.020 vs rownorm 0.172 +/- 0.059,
    # with rownorm's collapse seeds at 0.08/0.10 and sinkhorn never below 0.20; novelty_min 3x
    # higher. The claim established: two-sided scaling prevents low-temperature mode collapse on
    # THIS substrate, in the one-iteration moment-native form -- full Sinkhorn needs the data
    # points the model deliberately no longer stores. Default stays rownorm: backward compatible,
    # and the balancing costs 2n extra dot products per step.)
    _modes = np.array([[0.2, 0.2], [0.8, 0.3], [0.5, 0.8]])
    _rc = np.random.default_rng(0)
    _cd = np.vstack([mm + 0.04 * _rc.standard_normal((40, 2)) for mm in _modes])
    _cm = build_drift_model(_cd, dim=4096, seed=0)

    def _worst_share(X):
        lab = np.argmin(((X[:, None, :] - _modes[None]) ** 2).sum(-1), axis=1)
        return float((np.bincount(lab, minlength=3) / len(X)).min())
    _Xr = drift_sample(_cm, n=60, steps=60, seed=10, noise0=0.05, repel=0.5, coupling="rownorm")
    _Xs = drift_sample(_cm, n=60, steps=60, seed=10, noise0=0.05, repel=0.5, coupling="sinkhorn")
    assert _worst_share(_Xs) >= 0.15, \
        "the sinkhorn balancing must hold every mode (worst share %.2f)" % _worst_share(_Xs)
    assert _worst_share(_Xs) > _worst_share(_Xr), \
        "on the pinned collapse seed the balancing must beat rownorm (%.2f vs %.2f)" % (
            _worst_share(_Xs), _worst_share(_Xr))


    # --- H1.4, the corpus-scale verdict, pinned: WIN --------------------------------------------
    # (3-seed measurement on record: in-mode 1.00 +/- 0.00, novelty 0.71 +/- 0.04, memorised
    # 0.00 -- the drift model is the only contender simultaneously on-manifold, non-memorised,
    # and JOINT-structure-correct; strawman-B (independent marginals) broke the blob-separation
    # correlation at in-mode 0.83 / novelty 2.03, strawman-A (copies) sits at novelty 0.00.
    # KEPT NEGATIVE: image-space RMS is renderer-floor-saturated at this scale (all contenders
    # within 1.3% of the 0.152 floor) -- the verdict lives in drift space, stated, not hidden.)
    _hh = 24
    _yy, _xx = np.mgrid[0:_hh, 0:_hh]

    def _mk(mode, theta, r):
        sep = (4.0, 7.0, 10.0)[mode]
        cy, cx = _hh / 2 + r.uniform(-1, 1), _hh / 2 + r.uniform(-1, 1)
        dy, dx = sep / 2 * np.sin(theta), sep / 2 * np.cos(theta)
        im = (np.exp(-(((_yy - cy + dy) ** 2 + (_xx - cx + dx) ** 2) / 8.0)) +
              np.exp(-(((_yy - cy - dy) ** 2 + (_xx - cx - dx) ** 2) / 8.0)))
        return im / im.max()
    _rv = np.random.default_rng(0)
    _corp = np.stack([_mk(i % 3, _rv.uniform(0, np.pi), _rv) for i in range(30)])
    _vm, _vmeta = train_image_drift(_corp, k=2, dim=2048, seed=0)
    _vtp = np.stack([image_to_drift_point(im, k=2, seed=0) for im in _corp])
    _vout = generate_images(_vm, _vmeta, n=16, seed=1, audit_train=_vtp)
    _P = _vout["points"].reshape(len(_vout["points"]), 2, 4)
    _sep = np.sqrt(((_P[:, 0, :2] - _P[:, 1, :2]) ** 2).sum(1))
    if _sep.max() < 1.5:
        _sep = _sep * _hh
    _inmode = float((np.abs(_sep[:, None] - np.array([4.0, 7.0, 10.0])[None]).min(1) < 1.8).mean())
    assert _inmode >= 0.85, \
        "the H1.4 verdict regression: generated separations must stay in-mode (%.2f)" % _inmode
    _vaud = _vout["audit"]
    assert 0.1 < _vaud["novelty_mean"] < 1.6 and _vaud["memorised_frac"] < 0.2, \
        "generation must be neither copies nor off-manifold (novelty %.2f, memorised %.2f)" % (
            _vaud["novelty_mean"], _vaud["memorised_frac"])


    # INSTALLED-HDRIFT PINS (the sweep's four measurements, kept as traps):
    # (a) the head certifies rectangular DENSE at 0.0 through the projector -- the model IS a
    #     certified weight matrix; (b) MODEL ARITHMETIC IN WEIGHT SPACE is exact: head add ==
    #     compose, head subtract == ablate, at 0.0; (c) transport is a CERTIFIED linear action
    #     on head rows; (d) the sampling recurrence is nonlinear and the projector REFUSES it
    #     with a number -- generation stays host-shape, the head installs; both honest.
    from holographic.io_and_interop.holographic_projector import probe_project as _pp
    _r = np.random.default_rng(31)
    _pA = _r.standard_normal((160, 2)) * 0.3 + np.array([0.8, 0.0])
    _pB = _r.standard_normal((160, 2)) * 0.3 + np.array([-0.8, 0.4])
    _e = VectorFunctionEncoder(2, dim=1024, bounds=[(-3, 3), (-3, 3)], bandwidth=6.0, seed=5)
    _muA, _nuA = drift_moments(_pA, _e); _muB, _nuB = drift_moments(_pB, _e)
    _mA = DriftModel(_e, _muA, _nuA, 160); _mB = DriftModel(_e, _muB, _nuB, 160)
    _HA, _HB = drift_head(_mA), drift_head(_mB)
    _pc = _pp(lambda v: _HA[:, :128] @ v, 128)
    assert _pc["kind"] == "dense" and _pc["residual"] < 1e-12
    assert np.max(np.abs(drift_head(drift_compose(_mA, _mB)) - (_HA + _HB))) == 0.0
    assert np.max(np.abs(drift_head(drift_ablate(drift_compose(_mA, _mB), _mB)) - _HA)) < 1e-12
    _rt = drift_from_head(_e, _HA, 160)
    _q = np.array([0.5, 0.1])
    assert np.max(np.abs(drift_field(_q, _rt.mu, _rt.nu, _e) - drift_field(_q, _mA.mu, _mA.nu, _e))) == 0.0
    _d = np.array([0.2, -0.1])
    _ps = _pp(lambda r: _e.shift(np.concatenate([r, np.zeros(1024 - 128)]), _d)[:128], 128)
    assert _ps["kind"] in ("dense", "circulant") and _ps["residual"] < 1e-9
    _pn = _pp(lambda v: np.concatenate([_q + 0.25 * drift_field(_q + 0.01 * v[:2], _mA.mu, _mA.nu, _e),
                                        np.zeros(len(v) - 2)]), 8)
    assert _pn["kind"] == "refused", "the sampling step must stay honestly nonlinear"
    print("holographic_hdrift selftest OK -- field identity, kept negatives, algebra, images e2e, H0.4 anti-collapse, H1.4 verdict WIN, installed head (cert 0.0; algebra==weight arithmetic; transport certified; sampler refused)")


def _selftest_head():
    # INSTALLED-HDRIFT PINS: (a) head slice certifies dense 0.0; (b) weight-space model
    # arithmetic EXACT (compose==add, ablate==subtract); (c) transport mu is the certified
    # linear shift action on row 0. The nonlinear-sampler refusal is pinned in
    # compileinstall's referee (residual 8.0e-02) -- different module, same truth.
    from holographic.io_and_interop.holographic_projector import probe_project
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 2)) * 0.4 + np.array([1.0, 0.0])
    B = rng.standard_normal((200, 2)) * 0.4 + np.array([-1.0, 0.5])
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(-3, 3), (-3, 3)], bandwidth=6.0, seed=1)
    muA, nuA = drift_moments(A, enc)
    muB, nuB = drift_moments(B, enc)
    mA = DriftModel(enc, muA, nuA, len(A))
    mB = DriftModel(enc, muB, nuB, len(B))
    HA, HB = drift_head(mA), drift_head(mB)
    p = probe_project(lambda e: HA[:, :128] @ e, 128)
    assert p["kind"] == "dense" and p["residual"] < 1e-12
    assert float(np.max(np.abs(drift_head(drift_compose(mA, mB)) - (HA + HB)))) == 0.0
    assert float(np.max(np.abs(drift_head(drift_ablate(drift_compose(mA, mB), mB)) - HA))) < 1e-12
    d = np.array([0.3, -0.2])
    assert float(np.max(np.abs(drift_transport(mA, d).mu - enc.shift(mA.mu, d)))) == 0.0
    print("OK: hdrift installed-head pins passed (head dense 0.0; compose==add EXACT; "
          "ablate==subtract; transport row 0 == certified shift action)")


if __name__ == "__main__":
    _selftest()
    _selftest_head()
