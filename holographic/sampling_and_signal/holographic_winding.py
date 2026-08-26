"""Winding map: when the carrier LARGELY reverses, is content a function or a path?

WHY THIS MODULE EXISTS
----------------------
rectify_carrier's declared boundary: when a carrier axis mostly moves one way,
small reversals are absorbed by the arc-length lift and the content stays a
function of the axis. But when the axis LARGELY reverses -- sweeps back and forth,
revisiting the same coordinate ranges (monotone_fraction well below ~0.9) -- the
data is a genuine TRAJECTORY over the axis, and "content as a function of the
axis" becomes a hypothesis to TEST, not an assumption to make. Two worlds:

  FUNCTION: every visit to coordinate x finds the same content (up to noise).
    The passes are then FREE REPLICATION -- merge them and the noise averages
    down by sqrt(#laps). A back-and-forth scan of a stable profile (a raster
    scan, a repeated instrument sweep, a sensor pass) lives here.

  COVERING: revisits DISAGREE systematically. The content is not a function of x
    but of a point on a covering space above x -- most commonly (x, direction):
    the up-sweep and the down-sweep each trace their own consistent branch that
    differ from each other. That is HYSTERESIS (a magnetization loop, a
    backlash-y actuator, bid/ask-crossed sweeps), and merging the branches would
    manufacture a fictitious average curve that NO pass ever traced. The honest
    structure keeps the branches; the lift x -> (x, direction) is the same
    covering-space move as the Mobius/axial encoder (data that depends on
    orientation, not just position).

This module segments the trajectory into monotone LAPS at direction reversals,
interpolates each lap onto a shared grid over the revisited range, MEASURES the
agreement between laps, and returns the verdict with the evidence:

  'function'   : all laps agree -> merged (noise-averaged) profile returned.
  'hysteresis' : laps agree WITHIN a direction, disagree ACROSS directions ->
                 per-direction branches returned; merging refused.
  'path'       : laps disagree even within a direction -> the content depends on
                 the lap itself (drift, aging, a genuine trajectory); per-lap
                 curves returned and NO merged curve is offered.

KEPT NEGATIVES (loud)
---------------------
  * The verdict is decided by an agreement THRESHOLD on normalized disagreement
    (default 0.15 of the content's spread). Borderline data lands near it; the
    raw disagreement numbers travel with the verdict so a caller can see how
    close the call was. A threshold is a judgement, not a theorem.
  * Laps shorter than a few samples cannot be interpolated onto the grid and are
    DROPPED from the comparison (counted and reported, never silently).
  * 'function' merging averages noise but also averages any real drift slower
    than a lap -- if the profile is slowly changing, the merge is a low-pass in
    lap-time. The 'path' verdict is the guard, but its threshold inherits the
    caveat above.
  * The covering lift here is by DIRECTION (2 sheets). A deeper dependence (lap
    number k, temperature, ...) is the general covering; we detect it as 'path'
    and return the per-lap curves rather than guessing the hidden variable.

Only NumPy + stdlib. Deterministic (interpolation + arithmetic; no RNG).
"""

import numpy as np

# Laps whose normalized mutual disagreement is below this agree; above, they
# differ. WHY 0.15: instrument-noise-level wiggle on a unit-spread profile sits
# well under it; a genuine hysteresis split (branches separated by a noticeable
# fraction of the spread) sits well over. Reported with every verdict.
AGREE_TOL = 0.15

# A lap must span at least this many samples to be interpolable on the grid.
MIN_LAP = 4


def split_laps(coords):
    """Segment a trajectory into MONOTONE laps at direction reversals.

    Returns a list of dicts {slice, direction (+1/-1), span (coord range)}.
    Zero steps extend the current lap (a stall is not a reversal). WHY laps are
    the unit: each lap is a monotone pass over the axis, i.e. the largest piece on
    which "content as a function of the axis" is well-posed without any lift.
    """
    coords = np.asarray(coords, dtype=float).ravel()
    d = np.diff(coords)
    signs = np.sign(d)
    # propagate through stalls: a zero step belongs to whatever direction is live
    live = 0.0
    for i in range(signs.size):
        if signs[i] == 0.0:
            signs[i] = live
        else:
            live = signs[i]
    laps = []
    start = 0
    for i in range(1, signs.size):
        if signs[i] != 0.0 and signs[i - 1] != 0.0 and signs[i] != signs[i - 1]:
            laps.append((start, i + 1))
            # WHY i+1, not i: the reversal sample (the extremum) was measured while
            # travelling the OUTGOING direction, so it belongs to the lap that just
            # ended. Sharing it leaked one other-branch sample into each new lap and
            # contaminated the branch estimates near the turning points (measured:
            # one 0.8-offset point across a 100-point grid = 0.08 RMS, exactly the
            # error the fix removed).
            start = i + 1
    laps.append((start, coords.size))
    out = []
    for a, b in laps:
        seg = coords[a:b]
        if seg.size < 2:
            continue
        direction = 1 if seg[-1] >= seg[0] else -1
        out.append({"slice": (a, b), "direction": direction,
                    "span": (float(seg.min()), float(seg.max()))})
    return out


def _lap_curves(coords, content, laps, grid):
    """Interpolate each usable lap's content onto the shared grid.

    Descending laps are flipped so np.interp sees increasing sample points; the
    CONTENT values are untouched (the flip reorders samples, it does not mirror
    the profile). Laps too short or not covering the grid are skipped and counted.
    """
    curves, kept, dropped = [], [], 0
    for lap in laps:
        a, b = lap["slice"]
        x = coords[a:b]
        y = content[a:b]
        if x.size < MIN_LAP:
            dropped += 1
            continue
        if lap["direction"] < 0:
            x = x[::-1]
            y = y[::-1]
        # only compare where this lap actually has support
        lo, hi = x[0], x[-1]
        mask = (grid >= lo) & (grid <= hi)
        if mask.sum() < MIN_LAP:
            dropped += 1
            continue
        c = np.full(grid.size, np.nan)
        c[mask] = np.interp(grid[mask], x, y)
        curves.append(c)
        kept.append(lap)
    return curves, kept, dropped


def _disagreement(curves):
    """Mean pairwise RMS between lap curves on their SHARED support, normalized by
    the content spread -- the number the verdict runs on. NaN-aware: pairs are
    compared only where both laps have support."""
    if len(curves) < 2:
        return 0.0
    stack = np.stack(curves)
    spread = float(np.nanmax(stack) - np.nanmin(stack))
    if spread <= 1e-15:
        return 0.0
    tot, npair = 0.0, 0
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            both = ~(np.isnan(curves[i]) | np.isnan(curves[j]))
            if both.sum() < MIN_LAP:
                continue
            tot += float(np.sqrt(np.mean((curves[i][both] - curves[j][both]) ** 2)))
            npair += 1
    return (tot / npair / spread) if npair else 0.0


def winding_map(coords, content, grid_size=100, agree_tol=AGREE_TOL):
    """The full analysis: laps -> agreement -> function / hysteresis / path verdict.

    Parameters: coords (N,), content (N,) sampled along a back-and-forth sweep.

    Returns dict:
      verdict     : 'function' | 'hysteresis' | 'path'
      grid        : the shared coordinate grid over the revisited range.
      merged      : the noise-averaged profile (ONLY for 'function'; None
                    otherwise -- refusing the fictitious average is the point).
      branches    : {'up': curve, 'down': curve} (for 'hysteresis'), the
                    per-direction covering sheets.
      lap_curves  : every usable lap's curve (always returned -- the evidence).
      disagreement: {'all', 'within_up', 'within_down', 'across_directions'} --
                    the normalized numbers the verdict was decided on.
      n_laps, n_dropped.

    WHY this decision shape: 'function' requires ALL laps to agree; 'hysteresis'
    is the specific covering where direction is the hidden sheet (within-direction
    agreement high, across-direction disagreement high); anything else is 'path'
    (content depends on the lap itself) and no merge is offered.
    """
    coords = np.asarray(coords, dtype=float).ravel()
    content = np.asarray(content, dtype=float).ravel()
    laps = split_laps(coords)

    # the revisited range: covered by at least two laps
    spans = [l["span"] for l in laps]
    lo = max(min(s) for s in spans) if len(spans) > 1 else min(s[0] for s in spans)
    hi = min(max(s) for s in spans) if len(spans) > 1 else max(s[1] for s in spans)
    if hi <= lo:
        # laps do not overlap: nothing is revisited; each lap is its own function
        return {"verdict": "path", "grid": None, "merged": None, "branches": None,
                "lap_curves": [], "disagreement": None,
                "n_laps": len(laps), "n_dropped": 0,
                "note": "laps do not overlap; nothing revisited"}
    grid = np.linspace(lo, hi, int(grid_size))

    curves, kept, dropped = _lap_curves(coords, content, laps, grid)
    ups = [c for c, l in zip(curves, kept) if l["direction"] > 0]
    downs = [c for c, l in zip(curves, kept) if l["direction"] < 0]

    dis_all = _disagreement(curves)
    dis_up = _disagreement(ups)
    dis_down = _disagreement(downs)
    dis_across = _disagreement([np.nanmean(np.stack(ups), axis=0)] +
                               [np.nanmean(np.stack(downs), axis=0)]) \
        if ups and downs else 0.0

    disagreement = {"all": dis_all, "within_up": dis_up,
                    "within_down": dis_down, "across_directions": dis_across}

    if dis_all <= agree_tol:
        merged = np.nanmean(np.stack(curves), axis=0)
        return {"verdict": "function", "grid": grid, "merged": merged,
                "branches": None, "lap_curves": curves,
                "disagreement": disagreement, "n_laps": len(kept),
                "n_dropped": dropped}

    within_ok = (not ups or dis_up <= agree_tol) and \
                (not downs or dis_down <= agree_tol)
    if within_ok and ups and downs and dis_across > agree_tol:
        branches = {"up": np.nanmean(np.stack(ups), axis=0),
                    "down": np.nanmean(np.stack(downs), axis=0)}
        return {"verdict": "hysteresis", "grid": grid, "merged": None,
                "branches": branches, "lap_curves": curves,
                "disagreement": disagreement, "n_laps": len(kept),
                "n_dropped": dropped}

    return {"verdict": "path", "grid": grid, "merged": None, "branches": None,
            "lap_curves": curves, "disagreement": disagreement,
            "n_laps": len(kept), "n_dropped": dropped}


def _selftest():
    """Assert the three verdicts on constructed sweeps, with the noise-win measured.

    1. FUNCTION: a stable profile scanned back and forth 6x under noise -> verdict
       'function'; the merged profile beats any single lap against ground truth
       (the multi-pass free-denoise win, measured with its baseline).
    2. HYSTERESIS: up-sweeps trace f(x)+h, down-sweeps f(x)-h -> verdict
       'hysteresis'; branches recovered; merging would land BETWEEN the branches,
       a curve no pass traced (the refusal justified by a number).
    3. PATH: the profile drifts every lap -> 'path'; no merged curve offered.
    4. Determinism.
    """
    rng = np.random.default_rng(0)
    n_per, n_laps = 120, 6
    x_up = np.linspace(0, 1, n_per)
    f = lambda x: np.sin(2 * np.pi * x) + 0.3 * x

    # (1) function: same profile every pass, noise 0.05.
    coords, content, truth_grid = [], [], None
    for k in range(n_laps):
        xs = x_up if k % 2 == 0 else x_up[::-1]
        coords.append(xs)
        content.append(f(xs) + rng.standard_normal(n_per) * 0.05)
    coords = np.concatenate(coords)
    content = np.concatenate(content)
    r = winding_map(coords, content)
    assert r["verdict"] == "function", r["disagreement"]
    true = f(r["grid"])
    merged_err = float(np.sqrt(np.nanmean((r["merged"] - true) ** 2)))
    lap_errs = [float(np.sqrt(np.nanmean((c - true) ** 2)))
                for c in r["lap_curves"]]
    assert merged_err < min(lap_errs), (merged_err, min(lap_errs))  # the win
    assert merged_err < 0.05 / np.sqrt(n_laps) * 2.5                # ~sqrt averaging

    # (2) hysteresis: direction-dependent offset h = 0.4.
    coords2, content2 = [], []
    for k in range(n_laps):
        up = (k % 2 == 0)
        xs = x_up if up else x_up[::-1]
        coords2.append(xs)
        content2.append(f(xs) + (0.4 if up else -0.4)
                        + rng.standard_normal(n_per) * 0.03)
    r2 = winding_map(np.concatenate(coords2), np.concatenate(content2))
    assert r2["verdict"] == "hysteresis", r2["disagreement"]
    mid = 0.5 * (r2["branches"]["up"] + r2["branches"]["down"])
    # the fictitious average sits ~0.4 from each branch -- no pass traced it
    gap_up = float(np.nanmean(np.abs(r2["branches"]["up"] - mid)))
    assert gap_up > 0.3, gap_up
    assert r2["merged"] is None                                      # refusal held

    # (3) path: profile drifts by 0.3 per lap.
    coords3, content3 = [], []
    for k in range(n_laps):
        xs = x_up if k % 2 == 0 else x_up[::-1]
        coords3.append(xs)
        content3.append(f(xs) + 0.3 * k + rng.standard_normal(n_per) * 0.03)
    r3 = winding_map(np.concatenate(coords3), np.concatenate(content3))
    assert r3["verdict"] == "path", r3["disagreement"]
    assert r3["merged"] is None

    # (4) determinism.
    ra = winding_map(coords, content)
    rb = winding_map(coords, content)
    assert ra["verdict"] == rb["verdict"]
    assert np.array_equal(ra["merged"], rb["merged"])

    print("holographic_winding selftest OK (function: merged err %.4f vs best lap "
          "%.4f -- the multi-pass win | hysteresis branches recovered, gap %.2f | "
          "drift -> path, merge refused)"
          % (merged_err, min(lap_errs), gap_up))


if __name__ == "__main__":
    _selftest()
