"""Axis-role analysis: which axis is the INDEX (carrier) and which is the PAYLOAD (content)?

WHY THIS MODULE EXISTS
----------------------
A multi-dimensional dataset (audio [time x sample], video [time x H x W x C],
market data [time x asset x field], an alien/foreign message [row x col]) has
several axes, and each axis can play one of two roles when we turn it into a
holographic representation:

  * INDEX / carrier  -- the axis is the *enumeration* order. We keep one content
    vector per slice along it and lay the slices out in sequence (or address them
    by a similarity-preserving position code). The axis itself is NOT folded into
    the content vector.

  * PAYLOAD / content -- the axis's value IS part of what each item means, so we
    BIND it into the content vector (multiplicatively: circular convolution / the
    Hadamard product), making the axis-value part of the vector's identity.

The design decision is: *for each axis, index it or bind it?* Getting it wrong is
expensive, and the cost is asymmetric:

  * Binding is SIMILARITY-DESTROYING. bind(a, b) is ~orthogonal to a and to b
    (Plate 1995; Kleyko et al. survey 2022, arXiv:2111.06077). So binding a
    regular, low-information axis like time into every frame rotates each frame
    into a *private subspace* -- two adjacent frames that are 99% identical become
    ~orthogonal vectors, and cross-frame comparison (the whole point of a video
    representation) is destroyed. Rachkovskij & Kleyko (arXiv:2112.15475) measured
    exactly this for position: naive multiplicative position binding "does not
    preserve the similarity of symbol hypervectors at nearby positions."

  * The BENEFIT of binding an axis is that it lets a single superposition hold
    many (axis-value, content) pairs and still recover which content went with
    which value -- i.e. binding earns its keep only when the *combination* axis x
    content is the meaningful unit (the neuroscience "conjunctive code" case).

THE CRITERION (the contribution)
--------------------------------
The lower the information content of an axis, the more "binding it in" is pure
cost with little benefit. So:

  index an axis when it is BORING -- low marginal information (few distinct
  values, or a near-constant delta) AND low interaction with the content (knowing
  the axis value tells you little about the content beyond ordering);

  bind an axis when it is INFORMATIVE and its conjunction with content matters.

This module measures both quantities from data, per axis, and returns a
recommendation with the numbers that drive it. It does not *decide* for you on
borderline cases -- it reports the measurement and a threshold-based suggestion,
loud about the margin, so you can override in the conjunctive-coding regime.

WHAT WE MEASURE PER AXIS
------------------------
1. marginal information: the normalized entropy of the axis's *own* variation.
   For an ordered axis (time) we use the entropy RATE of its successive deltas --
   a constant delta (dt = const) has ~zero rate and is maximally "boring"; an
   irregular index carries information in its spacing. For an unordered/categorical
   axis we use label entropy over its distinct values, normalized by log(count).

2. content coupling: how much the content actually *changes* as a function of the
   axis, measured as 1 - (mean cross-slice cosine similarity along that axis).
   A time axis of a slowly-panning video has HIGH coupling (successive frames
   differ) but that is CONTINUITY, captured better by indexing; a truly random
   labelling axis has coupling too. Coupling alone does not say "bind" -- it says
   "this axis organizes the content," which is what an index is for. Binding pays
   only when coupling is high AND the axis is informative on its own (so the
   axis-value is worth recovering), which is the conjunctive case.

3. the recommendation combines them: an axis with LOW marginal information is an
   INDEX regardless of coupling (the carrier case -- time, scanline order). An
   axis with HIGH marginal information AND HIGH coupling is a BIND candidate
   (its value matters and it changes the content). The rest are indexed by
   default, because index is the cheap, comparability-preserving choice and the
   burden of proof is on binding.

KEPT NEGATIVES (loud)
---------------------
  * This is a HEURISTIC recommender, not a theorem. The threshold (LOW_INFO_FRAC)
    is a tunable, defaulted conservatively so we prefer INDEX (the safe choice)
    unless binding clearly pays. On a borderline axis it says so (small margin).
  * "Coupling" via cosine is a first-order probe. A high-frequency axis whose
    slices alias can read low coupling spuriously; pair with spectral_bandwidth
    for such data. Documented, not hidden.
  * The conjunctive-coding exception is REAL: if the downstream task needs the
    axis x content product as one unit (e.g. "the color-at-position", a resonator
    factor), you should BIND even a low-information axis. The recommender flags
    this by returning the coupling number so you can override; it cannot read your
    task's intent.
  * We do not invent structure. If every axis reads boring (a pure i.i.d. noise
    cube), the honest answer is "index all; nothing here rewards binding," and we
    say that rather than forcing a payload axis.
  * IRREGULAR-INDEX BORDERLINE (kept, measured): an axis with irregular spacing
    reads high marginal info (its timing carries bits), but that does NOT by
    itself mean binding pays -- binding pays only if the timing correlates WITH
    the content. Our coupling probe (adjacent-slice cosine) cannot separate
    "timing informs content" from "timing is just noisily irregular," so an
    irregular index can tip to BIND with a razor-thin margin (measured: marginal
    0.73, coupling 0.06, margin 0.011). The small margin is the honest signal --
    treat a sub-0.05-margin BIND as "inspect by hand," not a confident call. The
    clean fix (mutual information between the axis's local timing and the content
    change) is deferred, not claimed.

Only NumPy + stdlib. Deterministic (no RNG here; measurements are of the data).
"""

import math
import numpy as np

# Prefer INDEX unless an axis's own information exceeds this fraction of its max.
# Conservative on purpose: index is the comparability-preserving default, so we
# make binding earn its keep. Measured against the max entropy for the axis type,
# so it is scale-free across axis lengths.
LOW_INFO_FRAC = 0.35

# An axis whose content changes this little across slices is "flat" along that
# axis -- the content does not depend on it, so it is a pure index (or droppable).
FLAT_COUPLING = 0.05


def _entropy_bits(counts):
    """Shannon entropy (bits) of a count vector. Zero for a single populated bin.

    WHY: the raw information measure behind every axis score. A near-constant axis
    (all mass in one bin, e.g. a fixed dt) has entropy ~0 -- maximally boring.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _delta_entropy_rate(coords, bins=16):
    """Normalized entropy of the SUCCESSIVE DELTAS of an ordered coordinate axis.

    WHY deltas, not values: an ordered index (time stamps 0,1,2,3,...) has high
    entropy in its VALUES (they are all distinct) but that is not information --
    it is just counting. The information in an ordered axis lives in the
    IRREGULARITY of its spacing. A constant delta (uniform sampling) has zero
    delta-entropy = maximally boring = the ideal carrier. Irregular timestamps
    carry information in *when* samples land, and that raises the rate.

    Returns a value in [0, 1]: 0 = perfectly regular (bind nothing, pure index),
    toward 1 = highly irregular spacing (the axis positions themselves inform).
    """
    coords = np.asarray(coords, dtype=float).ravel()
    if coords.size < 3:
        return 0.0
    d = np.diff(coords)
    # Constant (or near-constant to float noise) delta => zero information.
    spread = np.ptp(d)
    if spread <= 1e-12 * (abs(np.mean(d)) + 1e-12):
        return 0.0
    lo, hi = float(d.min()), float(d.max())
    idx = np.floor((d - lo) / (hi - lo + 1e-12) * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    counts = np.bincount(idx, minlength=bins)
    h = _entropy_bits(counts)
    hmax = math.log2(bins)
    return float(h / hmax) if hmax > 0 else 0.0


def _label_entropy(values):
    """Normalized entropy of a categorical/unordered axis's distinct values.

    WHY: for an axis that is a *set* of labels (asset id, channel, symbol), the
    information is how many distinct, how evenly used. Normalized by log(#labels)
    so it is comparable across axes of different cardinality: 0 = one label
    (constant, boring), 1 = all labels equally likely (maximally informative).
    """
    values = np.asarray(values).ravel()
    if values.size == 0:
        return 0.0
    # Hash to stable integer bins (works for any dtype); never Python hash().
    uniq, counts = np.unique(values, return_counts=True)
    if uniq.size <= 1:
        return 0.0
    h = _entropy_bits(counts)
    hmax = math.log2(uniq.size)
    return float(h / hmax) if hmax > 0 else 0.0


def _slices_along(data, axis):
    """Yield the content slices along `axis` as flattened vectors.

    WHY flatten: coupling is "how different is slice i from slice i+1 as a whole,"
    so we compare each slice as one vector regardless of its internal shape.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[axis]
    moved = np.moveaxis(data, axis, 0)
    return moved.reshape(n, -1)


def _mean_adjacent_cosine(slabs):
    """Mean cosine similarity between ADJACENT slices (the continuity probe).

    WHY adjacent, not all-pairs: an INDEX axis (time in a smooth video) has high
    adjacent similarity -- neighbours are alike, which is exactly the structure an
    index+sequence preserves and binding would destroy. Low adjacent similarity
    means the axis reorders unrelated content (still an index, just a shuffled one).
    Returns mean cosine in [-1, 1]; we turn it into coupling = 1 - mean below.
    """
    slabs = np.asarray(slabs, dtype=float)
    if slabs.shape[0] < 2:
        return 1.0
    a = slabs[:-1]
    b = slabs[1:]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    if not np.any(ok):
        return 1.0
    cos = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return float(np.mean(cos))


def axis_report(data, axis, coords=None, categorical=False, bins=16):
    """Measure one axis's role signals: marginal information and content coupling.

    Parameters
    ----------
    data : ndarray
        The full multi-axis array.
    axis : int
        Which axis to characterize.
    coords : 1-D array, optional
        The axis's coordinate values (e.g. timestamps). If given and the axis is
        ordered, marginal info = delta-entropy-rate of these. If None, an ordered
        axis is assumed uniformly sampled -> marginal info 0 (the pure-carrier
        default, the common streaming case).
    categorical : bool
        If True, treat the axis as unordered labels (marginal info = label
        entropy over `coords` or over the slice index if coords is None).

    Returns
    -------
    dict with:
      marginal_info : float in [0,1]  -- how informative the axis is on its own.
      coupling      : float in [0,1]  -- 1 - mean adjacent cosine; how much the
                                         content varies along the axis.
      length        : int             -- number of slices.

    WHY both numbers: neither alone decides. Marginal info says "is the axis value
    worth recovering?" Coupling says "does the content depend on the axis?" Binding
    pays only when BOTH are high (the conjunctive regime). Index is right whenever
    marginal info is low (the carrier regime) -- that is the common, safe case.
    """
    data = np.asarray(data, dtype=float)
    length = data.shape[axis]

    if categorical:
        labels = coords if coords is not None else np.arange(length)
        marginal = _label_entropy(labels)
    else:
        if coords is None:
            # Uniformly-sampled ordered axis: constant delta => zero information.
            # This is the streaming default (audio/video/regular market bars).
            marginal = 0.0
        else:
            marginal = _delta_entropy_rate(coords, bins=bins)

    slabs = _slices_along(data, axis)
    coupling = 1.0 - _mean_adjacent_cosine(slabs)
    # Numerical guard: cosine can nudge slightly above 1.
    coupling = float(max(0.0, min(1.0, coupling)))

    return {"axis": int(axis), "marginal_info": float(marginal),
            "coupling": coupling, "length": int(length)}


def recommend_axis_role(report, low_info_frac=LOW_INFO_FRAC,
                        flat_coupling=FLAT_COUPLING):
    """Turn one axis_report into an INDEX / BIND / DROP recommendation + reason.

    The rule (burden of proof on binding):
      * marginal_info < low_info_frac  -> INDEX. The axis is a boring carrier;
        binding it in would rotate content into private subspaces for no benefit.
        This is the common, safe case (time, scanline order, regular bars).
      * marginal_info >= low_info_frac AND coupling >= flat_coupling -> BIND.
        The axis value is informative AND the content depends on it -- the
        conjunctive case where recovering (axis, content) as a unit is the point.
      * coupling < flat_coupling -> INDEX (or DROP): content does not vary along
        this axis, so it carries no payload; keep it as a cheap index or, if it is
        also low-info, note it is droppable (a constant axis).

    Returns a dict adding: role in {"index","bind"}, droppable (bool),
    margin (distance from the deciding threshold, so borderline calls are loud),
    reason (str).
    """
    mi = report["marginal_info"]
    cp = report["coupling"]

    droppable = cp < flat_coupling and mi < low_info_frac

    if mi < low_info_frac:
        role = "index"
        margin = low_info_frac - mi
        if cp < flat_coupling:
            reason = ("low marginal info and flat content along this axis: a "
                      "constant-carrier index (content does not depend on it)")
        else:
            reason = ("low marginal info: a boring carrier -- index it to keep "
                      "cross-slice comparability; binding would destroy it")
    else:
        if cp >= flat_coupling:
            role = "bind"
            margin = min(mi - low_info_frac, cp - flat_coupling)
            reason = ("high marginal info AND content varies along it: the "
                      "conjunctive case -- the (axis, content) pair is the unit, "
                      "so bind the axis value into content")
        else:
            role = "index"
            margin = flat_coupling - cp  # how flat it is
            reason = ("informative axis but content is flat along it: index it; "
                      "there is no content variation to bind against")

    out = dict(report)
    out.update({"role": role, "droppable": bool(droppable),
                "margin": float(margin), "reason": reason})
    return out


def analyze_axes(data, coords=None, categorical=None, bins=16,
                 low_info_frac=LOW_INFO_FRAC, flat_coupling=FLAT_COUPLING):
    """Full per-axis analysis of a tensor: measure + recommend a role for each axis.

    Parameters
    ----------
    data : ndarray
        The multi-axis dataset (e.g. video [T,H,W], market [T,asset,field]).
    coords : dict{axis:1-D array}, optional
        Coordinate values for specific axes (e.g. {0: timestamps}). Axes not
        listed are treated as uniformly-sampled ordered axes (marginal info 0).
    categorical : set/list of int, optional
        Axes to treat as unordered labels.

    Returns
    -------
    dict with:
      per_axis : list of recommendation dicts (one per axis).
      index_axes : list of axis indices recommended as INDEX.
      bind_axes  : list of axis indices recommended as BIND.
      summary    : one-line human string.

    WHY a whole-tensor call: the common real question is not "what is axis 3?" but
    "hand me a cube, tell me its schema" -- which axes carry the payload and which
    are the boring scaffolding. That is the auto-schema / auto-decomposition use.
    """
    data = np.asarray(data, dtype=float)
    coords = coords or {}
    cat = set(categorical or ())

    per_axis = []
    for ax in range(data.ndim):
        rep = axis_report(data, ax, coords=coords.get(ax),
                          categorical=(ax in cat), bins=bins)
        rec = recommend_axis_role(rep, low_info_frac=low_info_frac,
                                  flat_coupling=flat_coupling)
        per_axis.append(rec)

    index_axes = [r["axis"] for r in per_axis if r["role"] == "index"]
    bind_axes = [r["axis"] for r in per_axis if r["role"] == "bind"]

    if bind_axes:
        summary = ("index axes %s (carriers); bind axes %s (payload)"
                   % (index_axes, bind_axes))
    else:
        summary = ("index all axes %s -- nothing here rewards binding "
                   "(no axis is both informative and content-coupled)" % index_axes)

    return {"per_axis": per_axis, "index_axes": index_axes,
            "bind_axes": bind_axes, "summary": summary}


def comparability_cost(data, axis, dim=256, seed=0):
    """MEASURE the price of the wrong choice: bind a boring axis, watch comparability die.

    This is the empirical backbone of the whole criterion. We take the content
    slices along `axis` and compute the mean adjacent cosine two ways:

      INDEXED  : compare the raw slices (what indexing preserves).
      BOUND    : bind each slice with a distinct random per-slice key (what folding
                 the axis into content does), then compare adjacent bound vectors.

    For a boring axis the indexed similarity is high (neighbours are alike) and the
    bound similarity collapses toward 0 (each slice rotated into a private
    subspace). The ratio is the concrete, measured cost of binding the carrier.

    Returns dict{indexed_sim, bound_sim, collapse}: collapse = indexed - bound,
    the similarity destroyed by the wrong role choice. Uses a seeded RNG only to
    manufacture the per-slice binding keys (the thing binding-in-an-index does).

    WHY this belongs here: the recommender says "index this"; this function proves
    *why* on the user's own data, in one number, with the strongest honest baseline
    (the raw indexed similarity in the original space).
    """
    from holographic.agents_and_reasoning.holographic_ai import bind, random_vector

    slabs = _slices_along(data, axis)
    n, flat = slabs.shape
    indexed_sim = _mean_adjacent_cosine(slabs)

    # Project slices to `dim` for a fair bind (bind needs matched length); a fixed
    # seeded Gaussian projection is a similarity-preserving map (Johnson-
    # Lindenstrauss), so it does not itself distort the comparison.
    rng = np.random.default_rng(seed)
    if flat != dim:
        proj = rng.standard_normal((flat, dim)) / math.sqrt(flat)
        content = slabs @ proj
    else:
        content = slabs.copy()

    # Bind each slice with its own key -- exactly "fold the axis value into content."
    # Each key gets its own seeded rng so the per-slice keys are distinct and the
    # whole run is reproducible (the engine's determinism rule).
    keys = np.stack([random_vector(dim, np.random.default_rng(seed + 1 + i))
                     for i in range(n)])
    bound = np.stack([bind(content[i], keys[i]) for i in range(n)])
    bound_sim = _mean_adjacent_cosine(bound)

    collapse = float(indexed_sim - bound_sim)
    return {"indexed_sim": float(indexed_sim), "bound_sim": float(bound_sim),
            "collapse": collapse}


def rectify_carrier(coords, content, n_out=None):
    """REPAIR a nearly-boring carrier axis so it can serve as a clean uniform index.

    The axis-role criterion wants a carrier with ~zero marginal information (a
    constant delta). Real data offers two common defects, each with a standard fix,
    both applied here in order:

    1. NON-MONOTONE (the axis occasionally goes backwards / its delta dips
       negative): reparametrize by CUMULATIVE ARC LENGTH, s_i = sum |delta_j|.
       s is strictly non-decreasing BY CONSTRUCTION -- the odometer/covering-lift
       move from holographic_analytic (a monotone phase is an unwrappable phase):
       small reversals are absorbed into forward progress along the path, turning
       a wobbling axis back into a one-way clock. This is exactly "describe the
       sign by rotation, then keep only the winding."

    2. IRREGULAR (monotone but unevenly spaced): RESAMPLE the content onto a
       uniform grid in the (rectified) coordinate by linear interpolation -- the
       classical non-uniform-to-uniform resampling step, after which the carrier's
       delta is constant and its delta-entropy-rate is exactly 0 (maximally boring
       = ideal index).

    Parameters
    ----------
    coords : (N,) array -- the raw carrier values (may be irregular, may wobble).
    content : (N,) or (N, ...) array -- the payload sampled at those coords; every
        trailing-dim channel is resampled identically.
    n_out : int, optional -- output grid size (default N).

    Returns dict:
      coords    : the uniform output coordinates (in rectified units).
      content   : the resampled content.
      marginal_info_before / marginal_info_after : the axis-role boredom score,
        measured before and after -- 'after' is 0.0 by construction, and returning
        both makes the repair auditable rather than asserted.
      monotone_fraction : fraction of raw steps that were already forward -- how
        much rectification the axis needed (1.0 = only resampling was needed).

    KEPT NEGATIVES (loud):
      * Interpolation INVENTS values between samples under a smoothness
        assumption; content with structure finer than the local sample spacing is
        aliased, not recovered. The fix changes the INDEX, it cannot add payload
        information.
      * Arc-length reparametrization treats the series as a PATH: if the axis
        LARGELY reverses (revisits the same coordinate ranges repeatedly), the
        content stops being a function of the axis at all, and ordering it by
        path length is a modelling CHOICE (a trajectory parametrization), not a
        recovery of "the true axis." monotone_fraction is returned so a caller
        can see when they are in that regime (well below ~0.9 = inspect by hand).
      * A constant axis (zero total variation) has no direction to lift;
        rectification refuses rather than dividing by zero.
    """
    coords = np.asarray(coords, dtype=float).ravel()
    content = np.asarray(content, dtype=float)
    if coords.size != content.shape[0]:
        raise ValueError("coords and content must share their first dimension")
    if coords.size < 3:
        raise ValueError("need at least 3 samples to rectify")

    before = _delta_entropy_rate(coords)

    d = np.diff(coords)
    total_var = float(np.sum(np.abs(d)))
    if total_var <= 1e-15:
        raise ValueError("carrier has zero variation: nothing to index by")
    monotone_fraction = float(np.mean(d >= 0.0))

    # (1) the monotone lift: cumulative arc length. For an already-monotone axis
    # this is just coords shifted to start at 0 -- the lift is the identity there,
    # so applying it unconditionally is safe and keeps one code path.
    s = np.concatenate([[0.0], np.cumsum(np.abs(d))])

    # Arc length can stall (repeated coords -> zero step); np.interp needs strictly
    # increasing sample points, so nudge exact ties by the smallest representable
    # amount. WHY not drop ties: dropping would silently discard payload samples.
    ties = np.where(np.diff(s) <= 0)[0]
    if ties.size:
        eps = max(total_var, 1.0) * 1e-12
        for i in ties:
            s[i + 1] = s[i] + eps

    # (2) uniform resampling in s, every channel identically.
    n = int(n_out) if n_out else coords.size
    u = np.linspace(0.0, s[-1], n)
    flat = content.reshape(content.shape[0], -1)
    out = np.stack([np.interp(u, s, flat[:, c]) for c in range(flat.shape[1])],
                   axis=1).reshape((n,) + content.shape[1:])

    after = _delta_entropy_rate(u)   # 0.0 by construction; measured, not asserted

    return {"coords": u, "content": out,
            "marginal_info_before": float(before),
            "marginal_info_after": float(after),
            "monotone_fraction": monotone_fraction}


def _selftest():
    """Assert the exact behavioural contract, failing loudly on the core claims.

    The contracts (numeric, not just 'no exception'):
      1. A regular time axis over a smoothly-varying content reads LOW marginal
         info and is recommended INDEX.
      2. A high-entropy categorical payload axis whose content depends on it is
         recommended BIND.
      3. comparability_cost shows a POSITIVE collapse on the boring axis (binding
         the carrier destroys adjacent similarity) -- the whole thesis in a number.
      4. Determinism: the same array gives the same report twice.
    """
    rng = np.random.default_rng(0)

    # A video-like cube [T=20, H=8, W=8]: content is a smooth drift over time, so
    # the TIME axis is a boring regular carrier (uniform sampling, high adjacent
    # similarity). Build it as a low-frequency ramp so neighbours are alike.
    T, H, W = 20, 8, 8
    base = rng.standard_normal((H, W))
    drift = rng.standard_normal((H, W)) * 0.05
    video = np.stack([base + drift * t for t in range(T)], axis=0)

    res = analyze_axes(video)  # time axis uniform => marginal 0
    time_rec = res["per_axis"][0]
    assert time_rec["role"] == "index", ("time should index, got %r (%s)"
                                         % (time_rec["role"], time_rec["reason"]))
    assert time_rec["marginal_info"] < LOW_INFO_FRAC, time_rec["marginal_info"]

    # (2) A payload axis: build a cube whose content is a distinct random pattern
    # PER label along axis 1, and give axis 1 high-entropy labels. Its value is
    # informative AND the content depends on it -> BIND.
    L = 12
    patterns = rng.standard_normal((L, H * W))
    cube = np.stack([patterns[l].reshape(H, W) for l in range(L)], axis=0)  # [L,H,W]
    # Treat axis 0 here as categorical with all-distinct labels (max entropy).
    rep = axis_report(cube, 0, coords=np.arange(L), categorical=True)
    rec = recommend_axis_role(rep)
    assert rec["marginal_info"] > 0.9, rec["marginal_info"]  # all-distinct labels
    assert rec["role"] == "bind", ("distinct-label content axis should bind, got %r"
                                    % rec["role"])

    # (3) The measured cost: binding the boring time axis collapses comparability.
    cost = comparability_cost(video, 0, dim=128, seed=0)
    assert cost["indexed_sim"] > 0.9, cost  # neighbours genuinely alike
    assert cost["bound_sim"] < 0.2, cost    # rotated into private subspaces
    assert cost["collapse"] > 0.7, cost     # the thesis, in one number

    # (4) Determinism.
    a = analyze_axes(video)
    b = analyze_axes(video)
    assert a["summary"] == b["summary"]
    for ra, rb in zip(a["per_axis"], b["per_axis"]):
        assert abs(ra["marginal_info"] - rb["marginal_info"]) < 1e-15
        assert abs(ra["coupling"] - rb["coupling"]) < 1e-15

    # (5) Carrier rectification: an irregular axis becomes exactly boring
    # (marginal 0), and an occasionally-negative axis lifts to strictly monotone.
    t_irr = np.cumsum(rng.exponential(1.0, size=200))
    rect = rectify_carrier(t_irr, np.sin(0.1 * t_irr))
    assert rect["marginal_info_before"] > 0.3          # genuinely irregular
    assert rect["marginal_info_after"] == 0.0          # ideal carrier restored
    steps = np.full(200, 1.0); steps[::29] = -0.25     # rare small back-steps
    rect2 = rectify_carrier(np.cumsum(steps), np.cos(0.05 * np.arange(200)))
    assert np.all(np.diff(rect2["coords"]) > 0)        # strictly monotone lift
    assert rect2["monotone_fraction"] > 0.9            # and it says how much repair

    print("holographic_axisrole selftest OK "
          "(collapse on boring axis: %.3f -> %.3f, destroyed %.3f | "
          "rectify: %.3f -> %.3f)"
          % (cost["indexed_sim"], cost["bound_sim"], cost["collapse"],
             rect["marginal_info_before"], rect["marginal_info_after"]))


if __name__ == "__main__":
    _selftest()
