"""Auto-scaffold: explore an unlabeled series, find its primary axis, decompose, recompose.

WHY THIS MODULE EXISTS
----------------------
The end state of the axis-role arc, as an ORCHESTRATOR. Given a multi-axis array
with NO labels and NO schema, run the loop a careful analyst would:

  1. CANDIDATE SCAFFOLDS. Try EVERY axis as the potential carrier / primary axis.
     A good scaffold is BORING (low marginal information -- cheap to index; the
     axis-role criterion) AND ORGANISING (high adjacent-slice continuity -- indexing
     it lines the content up so neighbours resemble neighbours). Boredom alone is
     not enough: a shuffled axis is boring but organises nothing. Score =
     continuity * (1 - marginal_info), both already measured by holographic_axisrole.

  2. RECTIFY. If coordinates are supplied for the winning axis and they wobble or
     sample irregularly, repair them (rectify_carrier: arc-length lift + uniform
     resample) so the scaffold is a clean uniform index before any decomposition
     reads structure along it.

  3. DECOMPOSE along the scaffold. Each payload channel, read as a 1-D series
     along the carrier, goes to the engine's MDL-gated decomposer
     (decompose_signal: topology detection + elementary-function peeling). What
     comes back is the LAW of each channel plus a residual -- structure found "on
     at least some scope," exactly the user's phrase.

  4. RECOMPOSE and audit. Rebuild each channel from its discovered components and
     measure the explained fraction (1 - residual variance / signal variance).
     The RESIDUAL is then the honest hand-off to the next level of analysis:
     everything the discovered structure does NOT explain, in the original units --
     "use what was found to help understand what remains."

  5. HONEST VERDICT. If no axis organises the content (all continuity ~ chance) or
     nothing decomposes (explained fraction ~ 0 on every channel), say "no
     structure found at this scope" rather than dressing noise as law. The
     no-invented-structure rule, at pipeline level.

REUSE, NOT REINVENTION: this module contains almost no new mathematics. It is the
wiring of analyze_axes (step 1), rectify_carrier (step 2), decompose_signal
(step 3) and plain variance accounting (step 4) into one callable loop. Its value
is that a stranger -- or an agent over /invoke -- can hand the engine a raw cube
and get back the schema, the laws, and the leftovers, with every number carrying
its measurement.

KEPT NEGATIVES (loud)
---------------------
  * SCAFFOLD SCORE IS A HEURISTIC. continuity * (1 - marginal_info) prefers
    smooth carriers; a legitimately rough-but-ordered axis (e.g. white-noise-like
    increments over a real time order) scores low continuity and can lose to a
    smoother content axis. The full score table is returned so the ranking is
    inspectable, and a near-tie is visible as a near-tie.
  * DECOMPOSITION SCOPE: decompose_signal peels elementary functions
    (polynomial / harmonic / etc. under MDL). Structure outside its dictionary
    (chaotic flows, discrete programs) lands in the residual -- correctly, but
    "unexplained" here means "unexplained BY THIS DICTIONARY," not "random."
  * CHANNEL LIMIT: payload channels are decomposed independently; joint structure
    ACROSS channels (one channel a delayed copy of another) is not searched here.
    The residuals are returned precisely so a cross-channel pass can run on them.
  * One level of recursion is implemented (decompose, hand residual back). Deeper
    automatic recursion (re-run the whole pipeline on residuals) is a caller's
    loop over this faculty -- kept out so each level's verdict stays auditable.

Only NumPy + stdlib (all heavy lifting delegated to shipped modules). Deterministic.
"""

import numpy as np

from holographic.sampling_and_signal.holographic_axisrole import (
    axis_report, rectify_carrier, _slices_along, _mean_adjacent_cosine)

# A scaffold must organise the content at least this much better than chance
# (chance adjacent cosine ~ 0 for unrelated slices). Below it, no axis is a
# scaffold and the pipeline says so instead of picking a winner among losers.
MIN_CONTINUITY = 0.2

# A channel counts as "structured" when its discovered components explain at
# least this fraction of its variance. WHY 0.5: below half, the residual is the
# bigger story and calling the channel "explained" would oversell.
MIN_EXPLAINED = 0.5

# An axis must have at least this many samples to be indexable as a scaffold.
MIN_AXIS_LEN = 8


def scaffold_scores(data, coords=None):
    """Score every axis as a candidate scaffold: boring AND organising.

    Returns a list (one dict per axis) with marginal_info, continuity (mean
    adjacent-slice cosine along the axis), and score = continuity * (1 -
    marginal_info) -- plus the ranking. The table travels so a near-tie is
    visible; the pipeline never hides how close the call was.

    Axes shorter than MIN_AXIS_LEN are excluded from candidacy (score forced to
    -1). WHY (measured): an (N, 1) column let the LENGTH-1 axis win, laying the
    N samples out as N one-sample "channels" -- each trivially constant, each
    trivially "explained", a vacuous 'structured' verdict. A scaffold you cannot
    index along is not a scaffold. Same degenerate-layout family as the
    89,700-link bug in the cross-channel pass.
    """
    data = np.asarray(data, dtype=float)
    coords = coords or {}
    rows = []
    for ax in range(data.ndim):
        if data.shape[ax] < MIN_AXIS_LEN:
            rows.append({"axis": ax, "marginal_info": 1.0, "continuity": 0.0,
                         "score": -1.0, "length": int(data.shape[ax]),
                         "note": "too short to index along"})
            continue
        rep = axis_report(data, ax, coords=coords.get(ax))
        continuity = float(max(0.0, _mean_adjacent_cosine(_slices_along(data, ax))))
        score = continuity * (1.0 - rep["marginal_info"])
        rows.append({"axis": ax, "marginal_info": rep["marginal_info"],
                     "continuity": continuity, "score": float(score),
                     "length": rep["length"]})
    rows.sort(key=lambda r: -r["score"])
    return rows


def explore_series(data, coords=None, mind=None, max_terms=6,
                   auto_demux=False, cross_channel=False,
                   handle_reversals=False):
    """The full loop: scaffold -> rectify -> decompose -> recompose -> verdict.

    Parameters
    ----------
    data : ndarray, any shape -- the unlabeled series.
    coords : dict{axis: 1-D array}, optional -- known coordinates per axis (e.g.
        timestamps); the winning axis is rectified through them when they wobble.
    mind : UnifiedMind, optional -- supplied to reuse an existing mind's
        decompose_signal; a private one is booted if omitted (dim=256, seed=0,
        deterministic either way).
    auto_demux : bool, default False (old behaviour byte-identical when off) --
        for a 1-D stream, first run demux_series: detect an interleave stride,
        split into channels, group into objects, then run THIS loop on each
        object independently -- the Contact protocol with zero hints. Returns a
        {"demux": ..., "objects": [per-object explore results]} report instead.
    cross_channel : bool, default False -- after decomposition, run
        cross_channel_links on the RESIDUAL matrix: delayed-copy / shared-
        component structure that per-channel decomposition cannot see (a delayed
        copy of noise decomposes to nothing on both channels, yet the pair is
        lawful together). Adds a "residual_links" key; found links upgrade a
        'no structure found' verdict to 'weakly structured', because linked
        residuals ARE structure, just not per-channel structure.

    Returns dict:
      scaffold        : the winning axis (int) or None when nothing organises.
      scores          : the full per-axis score table (the evidence).
      rectified       : rectify_carrier's report for the scaffold (or None).
      channels        : per payload channel: {explained_fraction, n_components,
                        summary, residual (ndarray)} -- the laws and the leftovers.
      structured_channels / n_channels : how much of the payload decomposed.
      residual_links  : (cross_channel=True only) the delayed-copy links found
                        among residuals, with the score matrix.
      verdict         : 'structured' | 'weakly structured' | 'no structure found'
                        -- decided by measured explained fractions, never vibes.
    """
    data = np.asarray(data, dtype=float)
    if mind is None:
        import lecore
        mind = lecore.UnifiedMind(dim=256, seed=0)

    # 0. optional demux pre-stage: split a raw 1-D stream into its objects and
    # explore each independently -- the Contact protocol with zero hints.
    if auto_demux and data.ndim == 1:
        from holographic.sampling_and_signal.holographic_demux import demux_series
        d = demux_series(data)
        objects = []
        for obj in d["objects"]:
            arr = obj if obj.ndim > 1 else obj.reshape(-1, 1)
            objects.append(explore_series(arr, mind=mind, max_terms=max_terms,
                                          cross_channel=cross_channel))
        return {"demux": {"stride": d["stride"], "groups": d["groups"],
                          "n_objects": d["n_objects"]},
                "objects": objects,
                "verdict": ("structured"
                            if any(o["verdict"] == "structured" for o in objects)
                            else ("weakly structured"
                                  if any(o["verdict"] != "no structure found"
                                         for o in objects)
                                  else "no structure found"))}

    # 1. scaffold selection, with the honest no-scaffold exit.
    scores = scaffold_scores(data, coords=coords)
    best = scores[0]
    if best["continuity"] < MIN_CONTINUITY:
        result = {"scaffold": None, "scores": scores, "rectified": None,
                  "channels": [], "structured_channels": 0, "n_channels": 0,
                  "verdict": "no structure found",
                  "note": "no axis organises the content (all continuity ~ chance)"}
        # The cross pass matters MOST here: a delayed copy of noise has zero
        # per-channel continuity (this exit fires) yet the PAIR is lawful. With
        # nothing explained, the raw channels ARE the residuals -- run the pass
        # on them, laid out along the top-scoring axis.
        if cross_channel and data.ndim >= 2:
            from holographic.sampling_and_signal.holographic_demux import (
                cross_channel_links)
            # Lay SAMPLES along the LONGEST axis: the no-scaffold tie-break can
            # pick a short axis, which would treat samples as channels (measured:
            # 89,700 fictitious links from noise before this fix + the sample
            # guard in cross_channel_links).
            sample_ax = int(np.argmax(data.shape))
            slabs0 = _slices_along(data, sample_ax)
            if slabs0.shape[1] >= 2:
                cx = cross_channel_links(slabs0)
                result["residual_links"] = cx
                if cx["links"]:
                    result["verdict"] = "weakly structured"
                    result["note"] = ("no per-channel structure, but channels are "
                                      "linked (delayed copies / shared components)")
        return result
    ax = best["axis"]

    # 2. lay the payload out along the scaffold; rectify its coordinates if given.
    slabs = _slices_along(data, ax)              # (N, channels)
    rectified = None
    winding = None
    if coords and ax in (coords or {}):
        c_arr = np.asarray(coords[ax], dtype=float)
        rect = rectify_carrier(c_arr, slabs)
        # Optional winding adoption (default OFF, old results byte-identical):
        # when the carrier LARGELY reverses, rectification's arc-length lift lays
        # the laps end to end -- correct, but it forfeits the multi-pass denoise.
        # winding_map is the right instrument there (measured: 6-pass noisy scan,
        # profile RMS 0.112 single-lap -> 0.047 merged, 2.4x): a 'function'
        # verdict replaces the payload with the merged profile before
        # decomposition; 'hysteresis'/'path' verdicts are reported with their
        # evidence and the payload is left as rectified (merging there would
        # fabricate -- winding_map's own refusal, honoured here).
        if handle_reversals and rect["monotone_fraction"] < 0.9 \
                and slabs.shape[1] == 1:
            from holographic.sampling_and_signal.holographic_winding import (
                winding_map)
            wm = winding_map(c_arr, slabs[:, 0])
            winding = {"verdict": wm["verdict"],
                       "n_laps": wm["n_laps"],
                       "disagreement": wm["disagreement"]}
            if wm["verdict"] == "function":
                slabs = wm["merged"].reshape(-1, 1)
            else:
                slabs = np.asarray(rect["content"])
        else:
            slabs = np.asarray(rect["content"])
        rectified = {k: rect[k] for k in ("marginal_info_before",
                                          "marginal_info_after",
                                          "monotone_fraction")}
    else:
        slabs = np.asarray(slabs)

    # 3+4. decompose each channel along the carrier; recompose; account variance.
    # decompose_signal(x, y) returns (Formula, info); Formula.generate(x) is the
    # recomposition -- measured against the channel, never trusted from a flag.
    n = slabs.shape[0]
    xs = np.linspace(0.0, 1.0, n)                # the (rectified) uniform carrier
    channels = []
    for c in range(slabs.shape[1]):
        y = slabs[:, c]
        var = float(np.var(y))
        if var <= 1e-15:
            channels.append({"channel": c, "explained_fraction": 1.0,
                             "n_components": 0, "summary": "constant",
                             "residual": np.zeros_like(y)})
            continue
        formula, info = mind.decompose_signal(xs, y, max_terms=max_terms)
        try:
            recon = np.asarray(formula.generate(xs), dtype=float)
        except Exception:
            recon = np.zeros_like(y)
        if recon.shape != y.shape:
            recon = np.zeros_like(y)
        residual = y - recon
        explained = float(max(0.0, 1.0 - np.var(residual) / var))
        channels.append({"channel": c, "explained_fraction": explained,
                         "n_components": int(info.get("n_terms", 0)),
                         "summary": str(formula)[:200],
                         "residual": residual})

    # 5. verdict from the measurements.
    structured = sum(1 for ch in channels
                     if ch["explained_fraction"] >= MIN_EXPLAINED)
    frac = structured / len(channels) if channels else 0.0
    if frac >= 0.5:
        verdict = "structured"
    elif structured > 0:
        verdict = "weakly structured"
    else:
        verdict = "no structure found"

    result = {"scaffold": ax, "scores": scores, "rectified": rectified,
              "channels": channels, "structured_channels": structured,
              "n_channels": len(channels), "verdict": verdict}
    if winding is not None:
        result["winding"] = winding

    # 6. optional residual cross-channel pass: what "unexplained" channels SHARE
    # is structure per-channel decomposition cannot see. Found links upgrade a
    # bare 'no structure found' -- linked residuals are structure.
    if cross_channel and len(channels) >= 2:
        from holographic.sampling_and_signal.holographic_demux import (
            cross_channel_links)
        residuals = np.stack([ch["residual"] for ch in channels], axis=1)
        cx = cross_channel_links(residuals)
        result["residual_links"] = cx
        if cx["links"] and result["verdict"] == "no structure found":
            result["verdict"] = "weakly structured"

    return result


def decompose_piecewise(y, mind=None, min_seg=16, penalty=3.0, max_terms=6):
    """Decompose a PIECEWISE signal: segment first, fit a law PER SEGMENT.

    A signal built from regimes (a ramp, then a harmonic, then another ramp)
    fits a single global formula badly -- the dictionary has no 'switch at
    t=100' atom. Segmenting at the statistics shifts (segment_stream, the same
    piecewise-linear/BIC instrument packet_demux uses) and decomposing each
    regime separately fits each law in its own house. MEASURED on a 3-regime
    signal, against the global-fit baseline: residual RMS 0.5001 -> 0.0013 and
    MDL bits 2723 -> 588 (4.6x better compression) -- extra harmless boundaries
    inside an oscillating regime (the segmenter's declared negative) cost a few
    bits, not correctness.

    Returns dict: segments ((start, end) pairs), pieces (per segment:
    {formula, n_terms, resid_rms, mdl_bits}), reconstruction (full-length),
    residual_rms (whole signal), total_bits, and baseline {residual_rms,
    mdl_bits} from the global fit -- the win travels WITH its baseline, so a
    signal where segmentation does NOT pay is visible as such.

    KEPT NEGATIVES: inherits segment_stream's model (regime changes need a
    statistics shift); per-segment formulas do not share parameters across
    segments (a repeated regime is paid for twice -- recipe-level dedup is the
    next rung); boundaries are hard cuts (no cross-fade).
    """
    y = np.asarray(y, dtype=float).ravel()
    if mind is None:
        import lecore
        mind = lecore.UnifiedMind(dim=256, seed=0)
    from holographic.sampling_and_signal.holographic_demux import segment_stream

    n = y.size
    xs_full = np.linspace(0.0, 1.0, n)
    f0, i0 = mind.decompose_signal(xs_full, y, max_terms=max_terms)
    rec0 = np.asarray(f0.generate(xs_full), dtype=float)
    baseline = {"residual_rms": float(np.sqrt(np.mean((rec0 - y) ** 2))),
                "mdl_bits": float(i0["mdl_bits"])}

    seg = segment_stream(y, min_seg=min_seg, penalty=penalty)
    pieces, recon = [], np.zeros(n)
    total_bits = 0.0
    for a, b in seg["segments"]:
        xx = np.linspace(0.0, 1.0, b - a)
        ff, ii = mind.decompose_signal(xx, y[a:b], max_terms=max_terms)
        rr = np.asarray(ff.generate(xx), dtype=float)
        recon[a:b] = rr
        total_bits += float(ii["mdl_bits"])
        pieces.append({"segment": (a, b), "n_terms": int(ii["n_terms"]),
                       "resid_rms": float(ii["resid_rms"]),
                       "mdl_bits": float(ii["mdl_bits"]), "formula": ff})
    return {"segments": seg["segments"], "pieces": pieces,
            "reconstruction": recon,
            "residual_rms": float(np.sqrt(np.mean((recon - y) ** 2))),
            "total_bits": total_bits, "baseline": baseline}


def _selftest():
    """Assert the loop's contracts on planted structure and on honest noise.

    1. A cube whose TRUE carrier is axis 0 (smooth evolution along it; content
       scrambled along the others): the scaffold score picks axis 0, with the
       score table showing a real margin.
    2. Planted laws (a harmonic + a trend channel) along an IRREGULARLY sampled
       carrier: rectification fires, decomposition explains most of the variance,
       verdict 'structured', and the residual really is what the law misses
       (residual variance << signal variance).
    3. An i.i.d. noise cube: verdict 'no structure found' -- either no organising
       axis, or nothing decomposes; noise is never dressed as law.
    4. Determinism.
    """
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)

    # (1) the true carrier is axis 0: smooth evolution along it, while axis 1 is
    # a set of UNRELATED channels (independent random offsets), so slicing along
    # axis 1 gives unrelated neighbours -- a genuinely bad scaffold.
    T, A = 60, 5
    chan_off = rng.standard_normal(A) * 5.0        # unrelated channel identities
    cube = np.stack([chan_off + 0.05 * t for t in range(T)], axis=0)
    cube = cube + rng.standard_normal(cube.shape) * 0.01
    rows = scaffold_scores(cube)
    assert rows[0]["axis"] == 0, rows
    assert rows[0]["score"] > rows[1]["score"] + 0.05, rows   # a real margin

    # (2) planted laws on an irregular carrier.
    t_irr = np.cumsum(rng.exponential(1.0, size=200))
    u = (t_irr - t_irr[0]) / (t_irr[-1] - t_irr[0])           # normalised position
    ch_harm = np.sin(2 * np.pi * 2 * u)          # freq 2: inside the dictionary
    ch_trend = 0.8 * u + 0.1
    series = np.stack([ch_harm, ch_trend], axis=1)            # (200, 2)
    res = explore_series(series, coords={0: t_irr}, mind=mind)
    assert res["scaffold"] == 0, res["scores"]
    assert res["rectified"] is not None
    assert res["rectified"]["marginal_info_after"] == 0.0
    assert res["verdict"] == "structured", [
        (c["channel"], c["explained_fraction"]) for c in res["channels"]]
    for ch in res["channels"]:
        assert ch["explained_fraction"] > 0.9, ch
        assert float(np.var(ch["residual"])) < 0.1 * float(np.var(series[:, ch["channel"]]))

    # (3) honest noise.
    noise = rng.standard_normal((80, 4))
    resn = explore_series(noise, mind=mind)
    assert resn["verdict"] == "no structure found", resn["verdict"]

    # (4) determinism.
    a = explore_series(series, coords={0: t_irr}, mind=mind)
    b = explore_series(series, coords={0: t_irr}, mind=mind)
    assert a["verdict"] == b["verdict"]
    assert a["scaffold"] == b["scaffold"]
    assert all(abs(x["explained_fraction"] - y["explained_fraction"]) < 1e-12
               for x, y in zip(a["channels"], b["channels"]))

    # (5) ADOPTIONS, each with its baseline:
    # winding inside explore_series -- a reversing 6-pass scan: with the flag,
    # the merged profile's decomposition residual beats the rectified path's.
    xs6 = np.linspace(0, 1, 120)
    co, ct = [], []
    for k in range(6):
        cc = xs6 if k % 2 == 0 else xs6[::-1]
        co.append(cc)
        ct.append(np.sin(2 * np.pi * 2 * cc) + rng.standard_normal(120) * 0.15)
    co = np.concatenate(co); ct = np.concatenate(ct).reshape(-1, 1)
    r_off = explore_series(ct, coords={0: co}, mind=mind)
    r_on = explore_series(ct, coords={0: co}, mind=mind, handle_reversals=True)
    assert "winding" not in r_off                       # default off: unchanged
    assert r_on["winding"]["verdict"] == "function"
    assert r_on["channels"][0]["explained_fraction"] >=         r_off["channels"][0]["explained_fraction"]      # the adoption pays

    # piecewise decomposition -- 3-regime signal: beats the global baseline on
    # BOTH residual and bits, and the baseline travels with the result.
    nseg = 100
    yp = np.concatenate([2.0 * np.linspace(0, 1, nseg),
                         np.sin(2 * np.pi * 2 * np.linspace(0, 1, nseg)) + 3.0,
                         -np.linspace(0, 1, nseg) + 1.0])
    dp = decompose_piecewise(yp, mind=mind, min_seg=24)
    assert dp["residual_rms"] < 0.1 * dp["baseline"]["residual_rms"], dp["residual_rms"]
    assert dp["total_bits"] < dp["baseline"]["mdl_bits"], (dp["total_bits"],
                                                           dp["baseline"])

    exp = [round(c["explained_fraction"], 3) for c in res["channels"]]
    print("holographic_scaffold selftest OK (carrier found, rectified 0 boredom, "
          "explained fractions %s, noise honestly refused)" % exp)


if __name__ == "__main__":
    _selftest()
