"""Demux: one stream, many sources -- separate the channels, group the objects.

WHY THIS MODULE EXISTS
----------------------
explore_series assumes the series describes ONE thing per channel. Real streams
often carry SEVERAL sources at once, in two standard disguises:

  INTERLEAVED (time-division multiplexing): one 1-D stream where sample i belongs
    to channel (i mod K) -- the Contact move: the layered signal separates into
    channels, each decoded on its own. The tell is CONTINUITY: at the TRUE stride
    K, each strided sub-stream is smooth (its samples are consecutive readings of
    one source); at a wrong stride the sub-streams jump between unrelated sources
    and continuity collapses. So the stride is FOUND, not assumed, by scoring
    every candidate K -- the same adjacent-similarity instrument the scaffold
    score uses, pointed at the demux question. (This is also the Arecibo
    semiprime move in 1-D: the right factorization is the one that makes
    neighbours cohere.)

  GROUPED (multi-object): a multi-channel series where several channels belong to
    each object -- a scene of animated meshes serialized as per-vertex/per-axis
    delta channels: all channels of one mesh share its motion, channels of
    different meshes do not. Objects are recovered by clustering channels on the
    |correlation| of their trajectories (absolute, because a mirrored coordinate
    of the same rigid motion anti-correlates and still belongs to the object).

demux_series runs both: detect an interleave stride on a 1-D stream (splitting it
into channels), then group the channels into objects. Each group can then be
handed to explore_series independently -- decode each channel separately, exactly
the Contact protocol.

KEPT NEGATIVES (loud)
---------------------
  * Stride detection assumes CYCLIC interleaving (round-robin i mod K). Packetized
    or variable-length multiplexing (headers, bursts) will not score a clean K;
    the score table travels so a mushy maximum is visible as one.
  * The stride score compares K against K=1; a source that is ITSELF white-noise-
    like has no continuity to gain at any stride, so a noise channel inside the
    interleave neither helps nor hurts detection -- but a stream of ONLY noise
    reads K=1 (honest: there is nothing to separate).
  * Harmonic ambiguity: if the true stride is K, every multiple m*K also yields
    smooth sub-streams (each sub-stream is a further downsample of one source).
    We prefer the SMALLEST K within tolerance of the best score -- Occam on the
    channel count -- and report the full table so the harmonic ladder is visible.
  * Grouping clusters on |Pearson correlation| with a threshold: nonlinear
    relationships between channels of one object (a rotation mixing axes
    time-varyingly) can fall below it and split an object; the threshold and the
    correlation matrix travel with the result. Greedy agglomeration is
    order-stable (deterministic) but not globally optimal clustering.

Only NumPy + stdlib. Deterministic (correlations + arithmetic; no RNG).
"""

import numpy as np

# A candidate stride must beat the unstrided baseline's continuity by at least
# this margin to count as an interleave at all (otherwise K=1: nothing to split).
STRIDE_MARGIN = 0.15

# Smallest-K preference: any K whose score is within this tolerance of the best
# wins over a larger K -- the Occam guard against the m*K harmonic ladder.
STRIDE_TOL = 0.05

# Channels correlate at least this much (absolute) to share an object.
GROUP_THRESHOLD = 0.6


def _continuity_1d(x):
    """Mean adjacent cosine of successive DELTAS of a 1-D stream -- the smoothness
    instrument. WHY deltas: raw lag-1 correlation of a trending series is high even
    when interleaved (the trend dominates); delta continuity asks 'do consecutive
    steps look like steps of one source?', which is the demux-relevant question."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3:
        return 0.0
    d = np.diff(x)
    a, b = d[:-1], d[1:]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def detect_interleave(x, max_k=12):
    """Find the interleave stride K of a 1-D stream by continuity, or K=1 honestly.

    For each candidate K, split the stream into K strided sub-streams x[k::K] and
    score their MEAN delta-continuity. The true stride makes every sub-stream a
    smooth single-source series; wrong strides mix sources and the score drops.
    Smallest K within STRIDE_TOL of the best wins (the harmonic guard).

    Returns dict: k (the stride; 1 = not interleaved), score, baseline (K=1
    score), table (every K's score -- the evidence), channels (the K sub-streams).
    """
    x = np.asarray(x, dtype=float).ravel()
    max_k = int(min(max_k, max(1, x.size // 8)))   # each sub-stream needs samples
    table = []
    for k in range(1, max_k + 1):
        subs = [x[i::k] for i in range(k)]
        score = float(np.mean([_continuity_1d(s) for s in subs]))
        table.append({"k": k, "score": score})
    baseline = table[0]["score"]
    best_score = max(t["score"] for t in table)
    if best_score < baseline + STRIDE_MARGIN:
        k = 1                                       # nothing to separate: honest
    else:
        # smallest K within tolerance of the best -- Occam over the harmonics
        k = min(t["k"] for t in table if t["score"] >= best_score - STRIDE_TOL)
    return {"k": k,
            "score": next(t["score"] for t in table if t["k"] == k),
            "baseline": baseline, "table": table,
            "channels": [x[i::k] for i in range(k)]}


def group_channels(series, threshold=GROUP_THRESHOLD):
    """Group the channels of a (N, C) series into OBJECTS by |correlation|.

    Channels of one object share its motion (up to sign -- a mirrored axis of the
    same rigid transform anti-correlates, hence the absolute value). Greedy
    agglomeration: seed a group with the lowest-index ungrouped channel, admit
    every channel whose |corr| with the seed clears the threshold, repeat.
    Deterministic by construction (index order breaks ties -- the engine's rule).

    Returns dict: groups (list of channel-index lists), corr (the |correlation|
    matrix -- the evidence), threshold.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 2:
        raise ValueError("series must be (N, C)")
    C = series.shape[1]
    X = series - series.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0)
    norms[norms < 1e-12] = 1.0
    Xn = X / norms
    corr = np.abs(Xn.T @ Xn)

    ungrouped = list(range(C))
    groups = []
    while ungrouped:
        seed = ungrouped[0]
        member_mask = corr[seed] >= threshold
        group = [c for c in ungrouped if member_mask[c]]
        if seed not in group:
            group = [seed] + group
        groups.append(sorted(group))
        ungrouped = [c for c in ungrouped if c not in group]
    return {"groups": groups, "corr": corr, "threshold": float(threshold)}


def demux_series(x, max_k=12, group_threshold=GROUP_THRESHOLD):
    """The full demux: split an interleaved stream, then group channels into objects.

    1-D input: detect_interleave finds the stride and splits; the recovered
    channels (trimmed to equal length) are then grouped. 2-D input (already
    multi-channel): grouping only. Each returned object is a (N, c) sub-series
    ready for explore_series -- decode each channel separately, then each object.

    Returns dict: stride (the detected K; None for 2-D input), stride_table,
    groups (channel indices per object), objects (list of (N, c) arrays),
    corr, n_objects.
    """
    x = np.asarray(x, dtype=float)
    stride, stride_table = None, None
    if x.ndim == 1:
        det = detect_interleave(x, max_k=max_k)
        stride = det["k"]
        stride_table = det["table"]
        n = min(len(c) for c in det["channels"])
        series = np.stack([c[:n] for c in det["channels"]], axis=1)
    else:
        series = x

    g = group_channels(series, threshold=group_threshold)
    objects = [series[:, idx] for idx in g["groups"]]
    return {"stride": stride, "stride_table": stride_table,
            "groups": g["groups"], "objects": objects,
            "corr": g["corr"], "n_objects": len(objects)}


def cross_channel_links(series, max_lag=None, threshold=0.6):
    """Find DELAYED-COPY / shared-component links between channels (the residual pass).

    For every ordered channel pair (i, j), scan lags 0..max_lag of the normalized
    cross-correlation and keep the best. A strong peak at lag L with gain g means
    channel j ~ g * channel i delayed by L samples -- structure INVISIBLE to
    per-channel decomposition (a delayed copy of noise decomposes to nothing on
    both channels, yet the pair is perfectly lawful together). This is exactly
    the pass explore_series's residuals were returned for: run it on them, and
    what "unexplained" channels share becomes the next level's structure.

    Only non-negative lags are scanned per ordered pair (i leads j); the reverse
    direction is the pair (j, i), so causality direction falls out of which
    ordering carries the peak.

    Returns dict: links (list of {src, dst, lag, gain, score} with |score| >=
    threshold, sorted by |score| desc, deterministic tiebreak by indices) and
    the full score matrix best |xcorr| per ordered pair -- the evidence.

    KEPT NEGATIVES: linear, pairwise, single-lag -- a time-varying delay, a
    nonlinear coupling, or a three-way shared source that no pair sees strongly
    are out of scope (stated, not mis-reported). Correlation is not causation:
    "i leads j" is a LAG statement, not a mechanism claim.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 2:
        raise ValueError("series must be (N, C)")
    N, C = series.shape
    # STATISTICAL GUARD: |corr| of independent noise ~ 1/sqrt(N), so with too few
    # samples the threshold is meaningless (two mean-removed 2-sample vectors
    # correlate at exactly +/-1 -- measured: a degenerate layout once produced
    # 89,700 fictitious links from pure noise before this guard existed). We
    # require the noise floor to sit well under the threshold.
    min_n = int(np.ceil((3.0 / threshold) ** 2))    # noise floor <= threshold/3
    if N < min_n:
        return {"links": [], "score_matrix": np.zeros((C, C)),
                "threshold": float(threshold), "max_lag": 0,
                "note": ("too few samples (%d < %d) for the threshold to be "
                         "meaningful; refusing to report links" % (N, min_n))}
    if max_lag is None:
        max_lag = max(1, N // 4)
    max_lag = int(min(max_lag, N - 2))

    X = series - series.mean(axis=0, keepdims=True)
    links = []
    score_matrix = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            best_s, best_lag, best_gain = 0.0, 0, 0.0
            for lag in range(0, max_lag + 1):
                a = X[: N - lag, i]          # i leads by `lag`
                b = X[lag:, j]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na < 1e-12 or nb < 1e-12:
                    continue
                s = float(np.dot(a, b) / (na * nb))
                if abs(s) > abs(best_s):
                    best_s, best_lag = s, lag
                    best_gain = float(np.dot(a, b) / (na * na))
            score_matrix[i, j] = best_s
            if abs(best_s) >= threshold:
                links.append({"src": i, "dst": j, "lag": best_lag,
                              "gain": best_gain, "score": best_s})
    links.sort(key=lambda l: (-abs(l["score"]), l["src"], l["dst"]))
    return {"links": links, "score_matrix": score_matrix,
            "threshold": float(threshold), "max_lag": int(max_lag)}


def segment_stream(x, min_seg=16, penalty=3.0):
    """CHANGE-POINT segmentation: find where a stream's statistics shift (packet
    boundaries), by binary segmentation with a BIC-style penalty.

    The cost of a segment is its Gaussian description length, n * log(var + eps):
    a homogeneous segment is cheap, a segment straddling two different sources is
    expensive. Binary segmentation greedily takes the split with the biggest cost
    reduction, recursing on both halves, and STOPS when the best split saves less
    than penalty * log(N) -- the Occam term, so a homogeneous stream honestly
    returns no boundaries rather than shattering into noise-fit pieces.

    This is the CONTINUOUS costume of holographic_segment's discrete move
    (branching-entropy boundaries in a SYMBOL stream): both find units by where
    the local statistics break. Delegation isn't clean -- symbolizing a continuous
    stream first would add quantization fragility -- so the two live side by side
    with the kinship stated.

    Returns dict: boundaries (sorted sample indices), n_segments, segments
    (list of (start, end) pairs).

    KEPT NEGATIVES: piecewise mean+variance is the model -- a source change that
    preserves both (same mean, same power, different SPECTRUM) is invisible to
    this cost; the per-segment features in packet_demux (which include a spectral
    signature) can still separate such segments once a boundary exists, but the
    boundary itself needs a statistics shift. min_seg bounds resolution: bursts
    shorter than it are absorbed into neighbours.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    eps = 1e-12

    def seg_cost(a, b):
        n = b - a
        if n < 3:
            return 0.0
        seg = x[a:b]
        # PIECEWISE-LINEAR cost: detrend the candidate segment linearly before
        # taking the variance. WHY (measured): a piecewise-CONSTANT cost splits
        # any drifting source at its own drift (a clean ramp shattered into mean
        # steps; a sine at its half-cycles) -- pieces that then genuinely differ
        # in mean, so assignment cannot reunite them. Linear detrend makes a ramp
        # ONE cheap segment while a level STEP still splits (a line fits a step
        # badly, so the straddling segment stays expensive). Oscillation remains
        # the declared model negative (curvature still splits a sine).
        t = np.arange(n, dtype=float)
        t -= t.mean()
        denom = float(np.dot(t, t))
        slope = float(np.dot(t, seg - seg.mean())) / denom if denom > 0 else 0.0
        resid = seg - seg.mean() - slope * t
        return n * float(np.log(np.var(resid) + eps))

    thresh = penalty * np.log(max(N, 2))
    boundaries = []

    def recurse(a, b):
        n = b - a
        if n < 2 * min_seg:
            return
        whole = seg_cost(a, b)
        best_gain, best_k = 0.0, None
        # candidate splits leave min_seg on both sides
        for k in range(a + min_seg, b - min_seg + 1):
            gain = whole - (seg_cost(a, k) + seg_cost(k, b))
            if gain > best_gain:
                best_gain, best_k = gain, k
        if best_k is not None and best_gain > thresh:
            recurse(a, best_k)
            boundaries.append(best_k)
            recurse(best_k, b)

    recurse(0, N)
    boundaries.sort()
    cuts = [0] + boundaries + [N]
    segments = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    return {"boundaries": boundaries, "n_segments": len(segments),
            "segments": segments}


def _segment_signature(seg):
    """A per-segment feature vector for source assignment: mean, std, delta
    continuity, and a coarse spectral signature (energy in 4 bands). WHY these:
    mean/variance catch level/power differences, delta-continuity catches
    smooth-vs-rough, the band energies catch same-power-different-spectrum -- the
    case the boundary cost is blind to but assignment need not be."""
    seg = np.asarray(seg, dtype=float)
    n = seg.size
    feats = [float(np.mean(seg)), float(np.std(seg)), _continuity_1d(seg)]
    spec = np.abs(np.fft.rfft(seg - seg.mean())) ** 2
    if spec.size >= 4 and spec.sum() > 1e-15:
        bands = np.array_split(spec, 4)
        tot = spec.sum()
        feats.extend(float(b.sum() / tot) for b in bands)
    else:
        feats.extend([0.25, 0.25, 0.25, 0.25])
    return np.array(feats)


def packet_demux(x, min_seg=16, penalty=3.0, noise_k=3.0, continuation=False):
    """Demultiplex a PACKETIZED stream: variable-length bursts from different
    sources, no cyclic stride (the case demux_series's interleave scan declares
    out of scope). Two stages:

      1. segment_stream finds the packet boundaries (statistics shifts).
      2. segments are ASSIGNED to sources by a NOISE-CALIBRATED distance: each
         segment's signature is measured twice (split-half), and the distance
         between halves of the SAME segment estimates the measurement-noise
         floor per feature. Features are weighted by 1/noise (a Fisher-style
         reliability weighting: stable-within, varying-between features
         dominate; noisy features are ignored), and two segments merge when
         their weighted distance is within noise_k (default 3) times the median
         self-distance -- "closer than 3x the measurement noise = same source."
         Self-calibrating: no magic similarity threshold to tune per dataset.

    Returns dict: boundaries, segments, assignment (source id per segment),
    n_sources, noise_floor, sources (per source: its segment (start,end) list
    and the REASSEMBLED stream, segments concatenated in order -- each ready for
    explore_series: decode each source separately, the Contact protocol for the
    packetized case). An extra spurious boundary is harmless by construction:
    the assignment reunites same-source pieces.

    KEPT NEGATIVES: inherits segment_stream's model (a boundary needs a
    statistics shift; oscillating sources over-segment at their own swings --
    piecewise-LINEAR detrending tolerates ramps, not curvature); two sources
    with genuinely identical signatures merge (indistinguishable at this scope,
    stated not hidden); assignment is CONSERVATIVE by design -- two bursts of
    the same noisy source whose sampled statistics happen to differ by a couple
    of sigma can stay split (measured: two white-noise bursts at continuity
    -0.44 vs -0.69 read as separate) -- under-merging is the safe failure
    direction, it fabricates nothing; a source that DRIFTS ACROSS its bursts
    (the level continuing where it left off) reads as different levels --
    reuniting it needs sequence-continuation reasoning, out of a bag-of-segments
    scope UNLESS continuation=True (default off, old results byte-identical),
    which closes exactly that negative: extrapolate each source's linear tail
    across the gap and reunite bursts that resume where -- and at the slope --
    the prediction says, gated by the fit's own noise; every merge carries its
    {predicted, observed, tolerance} evidence; greedy agglomeration is
    deterministic, not globally optimal.
    """
    x = np.asarray(x, dtype=float).ravel()
    seg = segment_stream(x, min_seg=min_seg, penalty=penalty)
    segments = seg["segments"]
    S = len(segments)

    sig_full = np.stack([_segment_signature(x[a:b]) for a, b in segments])
    # split-half signatures: the same segment measured twice -> the noise floor.
    halves = []
    for a, b in segments:
        mid = (a + b) // 2
        if mid - a >= 4 and b - mid >= 4:
            halves.append((_segment_signature(x[a:mid]),
                           _segment_signature(x[mid:b])))
        else:
            halves.append(None)

    # Per-segment, per-feature split-half noise: each segment's own measurement
    # uncertainty. WHY per-pair weighting below (measured): a GLOBAL median noise
    # floor is crushed by the tightest segments (clean ramps: band noise ~0.002),
    # giving huge weights to features that genuinely fluctuate on noisier
    # segments (white-noise bursts: band noise ~0.1) -- so two same-source noise
    # bursts could never merge. Each pair is judged in ITS OWN metric: features
    # weighted by 1/max(noise_i, noise_j), thresholded against the pair's own
    # self-distances under the same weights.
    feat_dim = sig_full.shape[1]
    noise_vecs = [np.abs(h[0] - h[1]) if h is not None else None for h in halves]
    valid = [v for v in noise_vecs if v is not None]
    global_noise = np.median(np.stack(valid), axis=0) if valid \
        else np.ones(feat_dim)
    # SHRINKAGE floor on the noise (measured necessity): a clean deterministic
    # segment's split halves are IDENTICAL in std/continuity/bands -- pair noise
    # exactly 0 there, so 1/(0+eps) weights blew a 0.009 band difference up to a
    # distance of 11 million between two identical ramps. Floor each feature's
    # noise at a fraction of its CROSS-SEGMENT spread, so a deterministic feature
    # gets a population-scale weight, never an infinite one.
    feat_spread = sig_full.std(axis=0)
    noise_floor_vec = 0.1 * feat_spread + 1e-9
    global_floor = 1.0

    def pair_metric(i, j):
        ni = noise_vecs[i] if noise_vecs[i] is not None else global_noise
        nj = noise_vecs[j] if noise_vecs[j] is not None else global_noise
        w = 1.0 / np.maximum(np.maximum(ni, nj), noise_floor_vec)
        d = float(np.linalg.norm((sig_full[i] - sig_full[j]) * w))
        selfs = []
        for k, h in ((i, halves[i]), (j, halves[j])):
            if h is not None:
                selfs.append(float(np.linalg.norm((h[0] - h[1]) * w)))
        # a self-distance can be 0 for a deterministic segment; the merge floor
        # is never below the sqrt(feat_dim) a matched pair costs at the noise
        # floor itself -- the metric's own unit sphere.
        floor = max(max(selfs) if selfs else global_floor,
                    float(np.sqrt(feat_dim)) * 0.5, 1e-9)
        return d, floor

    assignment = [-1] * S
    n_sources = 0
    for i in range(S):
        if assignment[i] >= 0:
            continue
        assignment[i] = n_sources
        for j in range(i + 1, S):
            if assignment[j] < 0:
                d, floor = pair_metric(i, j)
                if d <= noise_k * floor:
                    assignment[j] = n_sources
        n_sources += 1

    sources = []
    for s in range(n_sources):
        segs = [segments[i] for i in range(S) if assignment[i] == s]
        stream = np.concatenate([x[a:b] for a, b in segs]) if segs else np.zeros(0)
        sources.append({"segments": segs, "stream": stream})

    result = {"boundaries": seg["boundaries"], "segments": segments,
              "assignment": assignment, "n_sources": n_sources,
              "noise_floor": global_floor, "sources": sources}

    # Optional continuation pass (default OFF: old results byte-identical): the
    # drift negative, closed -- extrapolate each source's linear tail across the
    # gap and reunite bursts that resume where the prediction says, at the
    # predicted slope. Every merge carries its {predicted, observed, tolerance}.
    if continuation:
        cont = continuation_merges(x, segments, assignment, gate_k=noise_k)
        assignment = cont["assignment"]
        n_sources = cont["n_sources"]
        sources = []
        for s in range(n_sources):
            segs = [segments[i] for i in range(S) if assignment[i] == s]
            stream = np.concatenate([x[a:b] for a, b in segs]) if segs \
                else np.zeros(0)
            sources.append({"segments": segs, "stream": stream})
        result.update({"assignment": assignment, "n_sources": n_sources,
                       "sources": sources, "continuation_merges": cont["merges"]})

    return result


def _linear_tail(seg, tail=32):
    """Fit a line to a segment's TAIL; return (level_at_end, slope, resid_std).

    WHY linear, why the tail: the segmentation cost is piecewise-LINEAR, so the
    within-model continuation predictor is the same model -- a line -- fitted to
    the most recent samples (the part that predicts the resumption). The residual
    std is the fit's own noise, which gates the match: a prediction is only as
    trustworthy as its residuals say. (holographic_forecast's calibrated AR/
    analog producers are the upgrade path for sources that are NOT piecewise-
    linear; using them here would model outside the segmenter's own assumptions.)
    """
    seg = np.asarray(seg, dtype=float).ravel()
    n = min(seg.size, int(tail))
    y = seg[-n:]
    t = np.arange(n, dtype=float)
    t -= t.mean()
    denom = float(np.dot(t, t))
    slope = float(np.dot(t, y - y.mean())) / denom if denom > 0 else 0.0
    level_end = float(y.mean() + slope * (n - 1 - (n - 1) / 2.0))
    resid = y - y.mean() - slope * t
    return level_end, slope, float(np.std(resid))


def continuation_merges(x, segments, assignment, gate_k=3.0, tail=32):
    """Reunite sources whose bursts CONTINUE across gaps (the drift negative, closed).

    A source that drifts across its bursts (a ramp resuming where it left off)
    reads as different LEVELS to the bag-of-segments assignment -- correctly, at
    that scope. This pass adds the sequence reasoning: for each pair of segments
    in DIFFERENT sources where one ends before the other starts, extrapolate the
    earlier segment's linear tail across the gap and accept the merge when the
    later segment's HEAD lands within gate_k times the combined noise (tail-fit
    residual + head-mean standard error) of the prediction, AND the slopes agree
    within the same gate. Both level and slope must match: a different source
    passing through the predicted level at a different rate is not a continuation.

    Returns dict: assignment (relabelled, contiguous ids), merges (list of
    {earlier_seg, later_seg, predicted, observed, tolerance} -- every merge
    carries its evidence), n_sources.

    KEPT NEGATIVES: linear continuation only (the segmenter's own model); a
    curved source resuming is matched only as well as its tail is locally linear.
    One gap at a time -- a chain A..B..C reunites transitively via union-find,
    but each link is judged on its own gap. A genuine NEW source that happens to
    start exactly on another's extrapolation, at the same slope, within noise,
    will merge -- indistinguishable by construction, stated.
    """
    x = np.asarray(x, dtype=float).ravel()
    S = len(segments)
    parent = list(range(max(assignment) + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    merges = []
    for i in range(S):
        ai, bi = segments[i]
        for j in range(S):
            if i == j:
                continue
            aj, bj = segments[j]
            if aj < bi:                       # j must start after i ends
                continue
            if find(assignment[i]) == find(assignment[j]):
                continue                       # already same source
            level, slope, resid = _linear_tail(x[ai:bi], tail=tail)
            gap = aj - (bi - 1)
            predicted = level + slope * gap
            head_n = min(bj - aj, int(tail))
            head = x[aj:aj + head_n]
            # The head's own linear fit, evaluated AT ITS START (passing the head
            # reversed puts its first sample at the fit's end): comparing the
            # prediction against the head MEAN was wrong by half a window of
            # slope (measured: an exact continuation read 3.60 predicted vs 3.91
            # observed and missed) -- like must be compared with like.
            h_level, h_slope, h_resid = _linear_tail(head[::-1], tail=head_n)
            observed = h_level
            tol = gate_k * (resid + h_resid + 1e-9)
            slope_tol = gate_k * (resid + h_resid + 1e-9) / max(tail, 1)
            # DYNAMICS gate (measured necessity): a high-noise source's level
            # gate is enormous (3 sigma of sigma=1 covers anything), so without
            # this it swallowed a quiet ramp resuming at the WRONG level. The
            # continuation of a source must LOOK like the source: tail and head
            # residual scales within 4x of each other.
            eps = 1e-6
            dyn_ok = abs(np.log((resid + eps) / (h_resid + eps))) <= np.log(4.0)
            if dyn_ok and abs(predicted - observed) <= tol and \
                    abs(slope - (-h_slope)) <= max(slope_tol, 1e-6):
                union(assignment[i], assignment[j])
                merges.append({"earlier_seg": i, "later_seg": j,
                               "predicted": predicted, "observed": observed,
                               "tolerance": tol})

    # relabel to contiguous source ids, order-stable
    roots = [find(a) for a in assignment]
    remap = {}
    new_assignment = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        new_assignment.append(remap[r])
    return {"assignment": new_assignment, "merges": merges,
            "n_sources": len(remap)}


def _selftest():
    """Assert the demux contracts on constructed multiplexes.

    1. CONTACT: three unlike sources round-robin interleaved at K=3 -- the stride
       is detected (score table showing the K=3 peak and its harmonics), and each
       recovered channel matches its source exactly (interleave/deinterleave is a
       permutation: bit-exact recovery is the contract, not approximation).
    2. Occam over harmonics: K=6 also scores well (it must -- each sub-stream is
       a further downsample) but K=3 wins by the smallest-K rule.
    3. MULTI-OBJECT MESH STREAM: two animated 'meshes' serialized as 3 coordinate
       channels each (shared per-object motion, one mirrored axis) -- grouping
       recovers exactly the two objects, mirror included.
    4. Honest noise: a white-noise stream reads K=1 (nothing to separate) and one
       group per channel is NOT forced into fictitious objects.
    5. Determinism.
    """
    rng = np.random.default_rng(0)
    n = 300
    u = np.linspace(0, 1, n)

    # (1)+(2) three sources, K=3 interleave.
    s1 = np.sin(2 * np.pi * 2 * u)
    s2 = 0.8 * u ** 2
    s3 = np.cos(2 * np.pi * 5 * u) * 0.5
    inter = np.empty(3 * n)
    inter[0::3], inter[1::3], inter[2::3] = s1, s2, s3
    det = detect_interleave(inter)
    assert det["k"] == 3, det["table"]
    for rec, src in zip(det["channels"], (s1, s2, s3)):
        assert np.array_equal(rec, src)               # a permutation: bit-exact
    k6 = next(t["score"] for t in det["table"] if t["k"] == 6)
    assert k6 > det["baseline"]                        # the harmonic scores well...
    assert det["k"] == 3                               # ...and Occam still picks 3

    # (3) two objects x three channels; object B's z-axis mirrored.
    motion_a = np.sin(2 * np.pi * 1.5 * u)
    motion_b = np.cumsum(rng.standard_normal(n)) * 0.1
    obj = np.stack([motion_a * 1.0, motion_a * 0.7, motion_a * 0.4,
                    motion_b * 1.0, motion_b * 0.6, -motion_b * 0.8], axis=1)
    obj += rng.standard_normal(obj.shape) * 0.01
    g = group_channels(obj)
    assert g["groups"] == [[0, 1, 2], [3, 4, 5]], g["groups"]

    # (4) honest noise.
    noise = rng.standard_normal(600)
    detn = detect_interleave(noise)
    assert detn["k"] == 1, detn["table"]

    # (5) determinism.
    a = demux_series(inter)
    b = demux_series(inter)
    assert a["stride"] == b["stride"] and a["groups"] == b["groups"]

    # (6) THE RESIDUAL PASS: channel 1 is channel 0 (pure noise) delayed by 7 at
    # gain 0.8 -- per-channel decomposition sees nothing on either, but the pair
    # is perfectly lawful together. The link must be found with the exact lag,
    # the gain to 5%, and the reverse ordering must NOT carry the peak (the lag
    # statement is directional).
    src = rng.standard_normal(400)
    dst = np.zeros(400)
    dst[7:] = 0.8 * src[:-7]
    pair = np.stack([src, dst], axis=1)
    cx = cross_channel_links(pair)
    assert cx["links"], "delayed copy missed"
    top = cx["links"][0]
    assert (top["src"], top["dst"], top["lag"]) == (0, 1, 7), top
    assert abs(top["gain"] - 0.8) < 0.05, top
    unrel = np.stack([rng.standard_normal(300), rng.standard_normal(300)], axis=1)
    assert not cross_channel_links(unrel)["links"]      # no fictitious links

    # (7) PACKETIZED demux: variable-length bursts from two statistically
    # distinct sources (a quiet low-level source and a loud noisy one). The
    # boundaries land near the truth, exactly two sources are found, and every
    # segment is assigned to the right one. A homogeneous stream returns
    # 1 segment / 1 source -- no fictitious packets.
    lens = [40, 65, 50, 80, 45, 70]                     # variable burst lengths
    parts, truth_src = [], []
    for i, L in enumerate(lens):
        if i % 2 == 0:
            parts.append(rng.standard_normal(L) * 0.1)          # quiet source A
        else:
            parts.append(3.0 + rng.standard_normal(L) * 1.0)    # loud source B
        truth_src.append(i % 2)
    stream = np.concatenate(parts)
    pk = packet_demux(stream, min_seg=16)
    # Assignment is CONSERVATIVE (documented): same-source bursts whose sampled
    # stats differ by ~2 sigma may stay split. The hard contracts are PURITY (no
    # source ever mixes quiet and loud segments -- fabrication is the failure
    # that matters) and that at least the two true populations emerge.
    assert pk["n_sources"] >= 2, pk["assignment"]
    truth_of_seg = []
    cuts = [0] + list(np.cumsum(lens))
    for a, b in pk["segments"]:
        mid = (a + b) // 2
        k = int(np.searchsorted(cuts, mid, side="right")) - 1
        truth_of_seg.append(k % 2)
    for s in range(pk["n_sources"]):
        classes = {truth_of_seg[i] for i in range(len(pk["segments"]))
                   if pk["assignment"][i] == s}
        assert len(classes) == 1, ("source %d mixes populations" % s,
                                   pk["assignment"], truth_of_seg)
    # boundaries within a few samples of the truth
    truth_bounds = np.cumsum(lens)[:-1]
    for tb in truth_bounds:
        assert min(abs(tb - b) for b in pk["boundaries"]) <= 5, \
            (tb, pk["boundaries"])
    # the reassembled streams separate cleanly by level
    means = sorted(float(np.mean(s["stream"])) for s in pk["sources"])
    assert means[0] < 0.5 and means[-1] > 2.5, means
    homo = packet_demux(rng.standard_normal(400), min_seg=16)
    assert homo["n_sources"] == 1 and homo["boundaries"] == [], homo["boundaries"]

    # (8) CONTINUATION (the drift negative, closed): a steep ramp, a loud noise
    # burst, then the ramp resuming EXACTLY on its extrapolation. The signature
    # pass reads different levels (correct at its scope); continuation=True
    # reunites them with the {predicted, observed} evidence matching -- while a
    # wrong-level resumption and a dynamics mismatch both stay refused, and
    # default-off results are byte-identical to before.
    t1 = 0.02 * np.arange(60)
    burst = 8.0 + rng.standard_normal(120)
    t2 = 0.02 * np.arange(180, 240)
    stream_c = np.concatenate([t1, burst, t2])
    off = packet_demux(stream_c, min_seg=24)
    on = packet_demux(stream_c, min_seg=24, continuation=True)
    assert off["n_sources"] == 3 and on["n_sources"] == 2
    m0 = on["continuation_merges"][0]
    assert abs(m0["predicted"] - m0["observed"]) <= m0["tolerance"]
    t2w = 1.0 + 0.02 * np.arange(180, 240)
    onw = packet_demux(np.concatenate([t1, burst, t2w]), min_seg=24,
                       continuation=True)
    assert onw["continuation_merges"] == []            # wrong level refused

    print("holographic_demux selftest OK (K=3 found + channels recovered "
          "bit-exact; harmonic K=6 declined by Occam; 2 mesh objects grouped, "
          "mirror included; noise honestly K=1; delayed copy found at lag 7 "
          "gain %.2f; packetized: %d boundaries, 2 sources, homogeneous refused)"
          % (top["gain"], len(pk["boundaries"])))


if __name__ == "__main__":
    _selftest()
