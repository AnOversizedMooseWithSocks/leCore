"""QUALITYGATE -- absolute-threshold render regression metrics, each tied to a defect that actually shipped.

WHY THIS EXISTS (Poly Studio's quality_gate.py, upstreamed in sweep 163). A preview shipped with heavy
grid-terracing (contour rings on curved surfaces, stair-stepped shadows) and the verification at the time MISSED
it, because it compared the new render against the previous render. Both had the artifact, so a difference metric
could not see it. The engine's own render sweeps have the same blind spot: sweep 151's aliasing would have been
caught by an absolute edge-tone check. So the gate measures a frame against ABSOLUTE thresholds instead, and every
threshold names the defect it catches:

  terracing   fraction of rows in the lower half showing a hard second-difference step. Grid-baked distance fields
              put iso-contour terraces into shading; an exact field does not. Calibrated on the frames that shipped:
              known-bad 0.068, known-good 0.018 (Poly Studio, 640 px). A plain first-difference test was tried and
              rejected -- it cannot tell a terrace from the legitimate gradient of a soft shadow edge.
  edge_tones  fraction of silhouette pixels (gradient > 30% of the frame max) with an intermediate tone nearby. Accumulating
              below display resolution and upscaling bakes in stair-stepped edges; true supersampling does not.
              Higher is better (1.0 = every edge pixel is partially covered).
  edge_chroma mean chroma at silhouette pixels. Compared between a converged and a single-round frame it catches a
              mis-aligned albedo map painting colour fringes onto silhouettes (fringe_ratio = converged / single).

`gate(frame, limits)` returns {metrics, failed, ok}. A metric alone is never sufficient -- LOOK at the frames too.
This gate catches regressions of KNOWN defects; it cannot see a new kind of ugly.

Pure NumPy; frames are (H,W,3) floats in 0..1 (or (H,W)).
"""
import numpy as np

# Poly Studio's shipped limits: measured headroom over the current good state, not aspirations. A gate that sits
# exactly on the current number fails on noise.
DEFAULT_LIMITS = {
    "terracing": 0.035,        # measured 0.022 good; the shipped-bad frame scored 0.068
    "edge_tones": 0.90,        # (1.000) fraction of edge pixels with an intermediate tone -- MINIMUM
    "fringe_ratio": 1.15,      # (~0.98) converged edge chroma / single-round edge chroma -- MAXIMUM
}


def _grey(img):
    a = np.asarray(img, float)
    return a.mean(-1) if a.ndim == 3 else a


def terracing(img, step=0.010, lower_half=True):
    """Fraction of (row, col) samples in the floor band whose SECOND difference down the image exceeds `step`.
    Terracing = flat plateaus separated by jumps, so it shows as spikes in the second difference; a smooth gradient
    (even a steep one) has a second difference near zero, which is what lets this see a terrace where a first
    difference cannot."""
    g = _grey(img)
    band = g[int(g.shape[0] * 0.5):, :] if lower_half else g
    if band.shape[0] < 3:
        return 0.0
    return float((np.abs(np.diff(band, 2, axis=0)) > step).mean())


def _edge_mask(g, frac=0.3):
    """Silhouette pixels: gradient magnitude above `frac` of the frame's maximum. WHY not Poly's top-1%-percentile:
    a percentile ties the mask to how much of the frame is edge -- on a small object in a big frame the top 1% spills
    onto flat floor and the metric collapses (measured: the same supersampled disc scored 0.92 at 160 px and 0.35 at
    640 px). A fraction of the strongest edge is size-independent (0.95-0.97 across 160..640 px, same disc)."""
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag > frac * float(mag.max()) if mag.max() > 0 else np.zeros_like(g, bool)


def _window(a, k, fn):
    """k x k sliding min/max (fn = np.minimum / np.maximum), edge-padded."""
    r = k // 2; pad = np.pad(a, r, mode="edge"); out = None
    for dy in range(k):
        for dx in range(k):
            s = pad[dy:dy + a.shape[0], dx:dx + a.shape[1]]
            out = s.copy() if out is None else fn(out, s)
    return out


def edge_tones(img, margin=0.15, min_range=0.05):
    """Fraction of silhouette pixels with a PARTIALLY COVERED tone in their 3x3 neighbourhood, where partial means
    'strictly between the two sides of the edge': at least `margin` of the LOCAL (5x5) tone range away from both the
    local minimum and the local maximum. A jagged, non-antialiased edge is all-or-nothing -- every pixel equals one
    side -- and scores 0; a supersampled edge scores near 1 (0.93 at 4x on a disc, dark ground or mid-grey ground
    alike). WHY local range and not absolute lo/hi tones (Poly's 0.05/0.95): the engine's default background is
    0.06 grey, which an absolute 0.05 floor counted as 'partial' and scored a jagged rasteriser 1.0. WHY the 3x3
    neighbourhood: np.gradient's central difference marks the pixels either side of a partial pixel as edge too, and
    those legitimately equal a side. Windows below `min_range` of contrast are not edges at all."""
    g = _grey(img)
    e = _edge_mask(g)
    if not e.any():
        return 0.0
    lo = _window(g, 5, np.minimum); hi = _window(g, 5, np.maximum); rng = hi - lo
    part = ((g - lo) > margin * rng) & ((hi - g) > margin * rng) & (rng > min_range)
    near = _window(part.astype(float), 3, np.maximum)
    return float(near[e].mean())


def edge_chroma(img):
    """Mean chroma (|R-G| + |G-B|) on silhouette pixels. Alone it is a number; as a RATIO between a converged frame
    and a single-round frame it is the colour-fringe detector."""
    a = np.asarray(img, float)
    if a.ndim != 3:
        return 0.0
    g = a.mean(-1); e = _edge_mask(g)
    ch = np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])
    return float(ch[e].mean()) if e.any() else 0.0


def fringe_ratio(converged, single):
    """edge_chroma(converged) / edge_chroma(single): > 1 means convergence ADDED colour to the silhouettes, which a
    correct accumulation never does (it only averages noise away)."""
    s = edge_chroma(single)
    return float(edge_chroma(converged) / s) if s > 1e-9 else 1.0


def gate(frame, limits=None, single_round=None):
    """Measure one frame against absolute limits. Returns {"metrics": {...}, "failed": [names], "ok": bool}.
    `limits` keys: terracing (max), edge_tones (min), fringe_ratio (max, needs `single_round`). Missing keys are
    not checked. WHY absolute: comparing against the previous render cannot see a defect both frames share."""
    lim = dict(DEFAULT_LIMITS); lim.update(limits or {})
    met = {"terracing": terracing(frame), "edge_tones": edge_tones(frame), "edge_chroma": edge_chroma(frame)}
    if single_round is not None:
        met["fringe_ratio"] = fringe_ratio(frame, single_round)
    failed = []
    if "terracing" in lim and met["terracing"] > lim["terracing"]:
        failed.append("terracing")
    if "edge_tones" in lim and met["edge_tones"] < lim["edge_tones"]:
        failed.append("edge_tones")
    if "fringe_ratio" in lim and "fringe_ratio" in met and met["fringe_ratio"] > lim["fringe_ratio"]:
        failed.append("fringe_ratio")
    return {"metrics": met, "limits": lim, "failed": failed, "ok": not failed}


# ------------------------------------------------------------------------------------------------ synthetic frames
def _disc(h, w, cx, cy, r, ss=1, contrast=False):
    """A shaded disc on a floor gradient, rendered with `ss`x supersampling (ss=1 is the jagged edge). `contrast`
    makes it a white disc on a black floor -- the silhouette regime edge_tones is defined for (its lo/hi bounds are
    absolute tones, so an edge between two mid-greys is invisible to it: a KEPT LIMIT, stated in the docstring)."""
    yy, xx = np.mgrid[0:h * ss, 0:w * ss] / float(ss)
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
    floor = 0.25 + 0.5 * (yy / h)                                 # a smooth vertical gradient
    shade = 0.9 - 0.4 * np.clip((yy - cy) / r, -1, 1)
    if contrast:
        floor = 0.02 * (yy / h); shade = np.ones_like(shade)
    g = np.where(inside, shade, floor)
    g = g.reshape(h, ss, w, ss).mean((1, 3))
    return np.stack([g, g, g], -1)


def _selftest():
    """Pins on synthetic frames whose defects are known by construction: a quantised floor terraces and the smooth
    one does not; a 1x edge is jagged and a 6x supersampled one is not; a chroma fringe painted onto the edges of a
    'converged' frame is caught by the ratio, while plain averaging of noisy frames is not."""
    good = _disc(120, 160, 80, 60, 30, ss=6)
    good_edge = _disc(120, 160, 80, 60, 30, ss=6, contrast=True)
    bad_edge = _disc(120, 160, 80, 60, 30, ss=1, contrast=True)
    # terracing: quantise the floor into 12 plateaus (what a coarse grid bake does to iso-contours)
    terraced = good.copy(); floor = terraced[:, :, 0]
    q = np.round(floor * 12) / 12.0
    terraced[..., :] = np.where(((np.arange(120)[:, None] - 60) ** 2 + (np.arange(160)[None, :] - 80) ** 2 < 900)[..., None],
                                terraced, q[..., None])
    t_good, t_bad = terracing(good), terracing(terraced)
    # the good frame is NOT zero: the disc's own silhouette crosses the floor band (Poly measured 0.018 on its good
    # frame for the same reason) -- the limit sits above that, and the terraced floor is several times it
    assert t_good < DEFAULT_LIMITS["terracing"] and t_bad > 3 * t_good, (t_good, t_bad)
    e_good, e_bad = edge_tones(good_edge), edge_tones(bad_edge)
    assert e_good > 0.9 and e_bad < 0.5, (e_good, e_bad)
    # fringe: add colour only where the edge is
    single = good + np.random.default_rng(0).normal(0, 0.02, good.shape)
    converged = good.copy()
    g = good.mean(-1); e = _edge_mask(g)
    converged[..., 0][e] += 0.15; converged[..., 2][e] -= 0.15
    honest = np.mean([good + np.random.default_rng(k).normal(0, 0.02, good.shape) for k in range(8)], 0)
    fr_bad, fr_honest = fringe_ratio(np.clip(converged, 0, 1), single), fringe_ratio(honest, single)
    assert fr_bad > 1.5 and fr_honest < 1.15, (fr_bad, fr_honest)
    # the gate names the failing metric, and passes the good frame
    r = gate(good_edge, single_round=good_edge + np.random.default_rng(1).normal(0, 0.02, good.shape)); assert r["ok"], r
    r = gate(terraced); assert r["failed"] == ["terracing"], r
    r = gate(bad_edge); assert "edge_tones" in r["failed"], r
    print("qualitygate selftest OK: terracing %.3f good vs %.3f terraced; edge_tones %.2f supersampled vs %.2f jagged; "
          "fringe_ratio %.2f fringed vs %.2f honest average; gate names the failing metric"
          % (t_good, t_bad, e_good, e_bad, fr_bad, fr_honest))


if __name__ == "__main__":
    _selftest()
