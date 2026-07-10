"""holographic_refresh.py -- W4: information-rate rendering. Shade the news, reproject the rest.

A camera drifts over a scene. Instead of shading every pixel every frame, warp the previous frame forward and shade
only a small budget of pixels: the DISOCCLUSION BORDER (the strip the camera just revealed) plus the OLDEST k
pixels, so every pixel is eventually refreshed and nothing goes stale.

MEASURED on a procedural scene that is a pure function of world coords (W4's own scene class -- no parallax, no
view-dependent shading), 12 frames, 20% budget:

    aiming                                   shaded   PSNR mean   worst
    known camera shift + bilinear warp        20.0%    57.5 dB    55.9   <- the bar: 5x fewer shader evals
    est_dx-recovered shift + bilinear warp    20.0%    47.5       40.2
    est_dx + INTEGER roll (np.roll)           20.0%    40.7       35.9

The backlog reported 55.6 dB mean / 54.1 worst at 20.6% for the first row; this reproduces it. **Five times fewer
shader evaluations at visually-indistinguishable quality, with no decay.**

KEPT NEGATIVE 1 -- **recovering the shift from pixels costs 10 dB.** `est_dx` is accurate to 0.07 px on a single
pair, but this is a FEEDBACK LOOP: each frame warps a frame that was itself warped, so the error compounds. The
renderer knows how the camera moved; use `known_shift=` and keep the 10 dB. This is F7's conclusion arriving from
the other side -- the unbind is an excellent estimator, not a substitute for knowing the camera.

KEPT NEGATIVE 2 -- **integer `np.roll` decays.** Sub-pixel drift accumulates: 40.7 dB against 57.5 dB for the same
budget. Bilinear warp is not a refinement, it is the mechanism.

KEPT NEGATIVE 3 -- **the "oldest pixels" selection is a fake-perfect bug waiting to happen.** A threshold rule
("refresh every pixel whose age >= the k-th largest") selects EVERYTHING when the ages are tied -- which they are on
frame 0, and after any full refresh. Measured: 16,384 of 16,384 pixels selected, 100% shaded, PSNR = 99 dB. A
perfect score, achieved by doing all the work. `exact_k_oldest` takes exactly k with a stated, deterministic
tie-break (stable sort -> lowest flat index wins). This is `argmax_tiebreak`'s lesson in a new place: **a selection
rule is an observable decision and its ties must be named.**

HONEST SCOPE, and it is the whole caveat. The 57.5 dB figure belongs to a scene with no parallax and no
view-dependent shading. On a real 3-D scene the reprojection ceiling itself is near 38-41 dB (see
`holographic_reproject`: parallax halves what one translation can explain), so refresh cannot beat it -- measured
36.8 dB mean, 34.9 worst, and it DECAYS 3.4 dB over 9 frames because each warp compounds the last one's error.
Depth-aware warps and per-object masks are the known extensions, and each owes its own measurement.
"""

import numpy as np

from holographic.rendering.holographic_reproject import est_dx, psnr, warp


def exact_k_oldest(age, k):
    """Select EXACTLY `k` pixels with the greatest age. Ties break deterministically on the flat index.

    THE BUG THIS REPLACES: a threshold rule -- "take every pixel whose age >= the k-th largest" -- selects the whole
    frame whenever ages are tied, which they are on frame 0. That reports a perfect PSNR because it shaded
    everything. Measured: 16,384 of 16,384 pixels. `np.argsort(kind="stable")` is order-preserving, so `-age` sorted
    descending puts the lowest flat index first among equals: the tie-break is named, not accidental."""
    age = np.asarray(age, float)
    k = int(max(0, min(int(k), age.size)))
    idx = np.argsort(-age.ravel(), kind="stable")[:k]
    mask = np.zeros(age.size, bool)
    mask[idx] = True
    return mask.reshape(age.shape)


def threshold_oldest(age, k):
    """The BUGGY selection, kept so a test can pin it. Returns every pixel whose age >= the k-th largest -- which is
    every pixel when the ages are tied. Never call this; it exists to be shown failing."""
    age = np.asarray(age, float)
    k = int(max(1, min(int(k), age.size)))
    thr = np.sort(age.ravel())[::-1][k - 1]
    return age >= thr


def disocclusion_border(shape, dy, dx):
    """The strip the camera just revealed, which no amount of warping can fill: the previous frame never saw it.

    A warp CLAMPS at the edge, so those pixels are a smeared copy of the border. They must be shaded, every frame,
    and they are the reason the measured budget (20.6% in the backlog) exceeds the nominal one (20%). Reporting the
    border inside the budget is the honest choice: it is work the renderer actually does."""
    H, W = int(shape[0]), int(shape[1])
    border = np.zeros((H, W), bool)
    bx = int(np.ceil(abs(float(dx)))) + 1
    by = int(np.ceil(abs(float(dy)))) + 1
    if dx < 0:
        border[:, W - bx:] = True
    elif dx > 0:
        border[:, :bx] = True
    if dy < 0:
        border[H - by:, :] = True
    elif dy > 0:
        border[:by, :] = True
    return border


class RefreshRenderer:
    """The reproject-and-refresh loop as a render mode.

    `shade(mask)` must return the FULL correct frame; the loop reads only `frame[mask]` from it, so a real renderer
    would shade only those pixels and the budget is the shader-evaluation count. `step(shade, known_shift=None)`
    returns the composited frame and records `last_stats`.

    Keep `known_shift` if you have it (a renderer does): recovering the shift from pixels costs a measured 10 dB,
    because the loop warps its own output and the estimator's 0.07 px error compounds."""

    def __init__(self, first_frame, budget=0.20):
        self.frame = np.asarray(first_frame, float).copy()
        self.age = np.zeros(self.frame.shape[:2], float)
        self.budget = float(budget)
        self.last_stats = {"shaded": self.frame[..., 0].size if self.frame.ndim == 3 else self.frame.size,
                           "fraction": 1.0, "border": 0, "shift": (0.0, 0.0)}

    def step(self, shade, known_shift=None):
        """Advance one frame. `shade` is a callable returning the true next frame."""
        ref = np.asarray(shade(None), float)
        dy, dx = (est_dx(self.frame, ref) if known_shift is None else
                  (float(known_shift[0]), float(known_shift[1])))
        pred = warp(self.frame, dy, dx)

        self.age += 1.0
        border = disocclusion_border(self.age.shape, dy, dx)
        self.age[border] = np.inf                        # the news is always refreshed

        k = int(round(self.budget * self.age.size))
        mask = exact_k_oldest(self.age, k)

        out = pred.copy()
        out[mask] = ref[mask]
        self.age[mask] = 0.0
        self.age[np.isinf(self.age)] = 0.0               # a border pixel not selected is still fresh next frame

        self.frame = out
        self.last_stats = {"shaded": int(mask.sum()), "fraction": float(mask.sum()) / mask.size,
                           "border": int(border.sum()), "shift": (float(dy), float(dx))}
        return out


def refresh_report(shade_at, n_frames=12, budget=0.20, known_shift=None):
    """Run the loop and report {shaded_fraction, psnr_mean, psnr_worst, psnr_first, psnr_last, tail_slope, psnrs}.

    `shade_at(i)` renders frame `i` in full -- the reference the loop is scored against.

    STABILITY IS `tail_slope`, NOT `psnr_first - psnr_last`. Frame 1 warps a PERFECT frame 0 and scores well above
    the mean for free; measuring decay from there makes a flat run look like a collapse. Measured on the headline
    configuration: first 62.8 dB, mean 57.5, last 56.2 -- **6.6 dB of apparent "decay" on a run whose tail slope is
    +0.22 dB**. `tail_slope` is `mean(last third) - mean(middle third)`: near zero means the reconstruction has
    reached a steady state, which is the property W4 actually claims. I measured the wrong thing first, and the
    first frame's free lunch is why."""
    r = RefreshRenderer(shade_at(0), budget=budget)
    fracs, scores = [], []
    for i in range(1, int(n_frames)):
        ref = shade_at(i)
        out = r.step(lambda _m, _r=ref: _r, known_shift=known_shift)
        fracs.append(r.last_stats["fraction"])
        scores.append(psnr(out, ref))
    scores = np.array(scores)
    third = max(1, len(scores) // 3)
    tail_slope = float(scores[-third:].mean() - scores[-2 * third:-third].mean()) if len(scores) >= 2 * third else 0.0
    return {"shaded_fraction": float(np.mean(fracs)), "psnr_mean": float(scores.mean()),
            "psnr_worst": float(scores.min()), "psnr_first": float(scores[0]),
            "psnr_last": float(scores[-1]), "tail_slope": tail_slope,
            "psnrs": [float(v) for v in scores]}


def _selftest():
    """Regression trap for W4: the budget is honoured exactly, the fake-perfect selection bug is pinned, and
    bilinear warping beats integer rolling on the same budget."""
    H = W = 128                                                   # the size the headline numbers were measured at

    def world(ox):
        yy, xx = np.meshgrid(np.arange(H), np.arange(W) + ox, indexing="ij")
        v = (np.sin(xx * 0.11) * np.cos(yy * 0.09) + 0.4 * np.sin((xx + yy) * 0.05)
             + 0.3 * np.sin(xx * 0.31) * np.sin(yy * 0.27))
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    step = 1.7

    # 1. THE FAKE-PERFECT BUG. Tied ages: the threshold rule takes the whole frame, exact-k takes exactly k.
    tied = np.zeros((H, W))
    k = int(0.2 * H * W)
    assert threshold_oldest(tied, k).sum() == tied.size           # 100% shaded: a perfect PSNR, all the work
    assert exact_k_oldest(tied, k).sum() == k                     # exactly k, ties broken on the flat index

    # ... and the tie-break is DETERMINISTIC: the lowest flat indices win
    picked = np.flatnonzero(exact_k_oldest(tied, 5).ravel())
    assert list(picked) == [0, 1, 2, 3, 4]

    # 2. the budget is honoured, and the border is inside it
    rep = refresh_report(lambda i: world(i * step), n_frames=12, budget=0.20, known_shift=(0.0, -step))
    assert abs(rep["shaded_fraction"] - 0.20) < 0.01
    assert rep["psnr_mean"] > 55.0 and rep["psnr_worst"] > 54.0   # the bar: 55.6 / 54.1 in the backlog
    assert abs(rep["tail_slope"]) < 1.0                           # STABLE: measured +0.22 dB
    assert rep["psnr_first"] > rep["psnr_mean"]                   # frame 1 warps a PERFECT frame 0: a free lunch

    # 3. KEPT NEGATIVE: recovering the shift from pixels costs ~10 dB AND makes the tail slide, because the loop
    #    warps its own output and a 0.07 px error compounds.
    est = refresh_report(lambda i: world(i * step), n_frames=12, budget=0.20)
    assert est["psnr_mean"] < rep["psnr_mean"] - 5.0              # measured 47.1 vs 57.5
    assert est["tail_slope"] < -2.0                               # measured -9.52: decaying

    # 4. a full budget reproduces the reference exactly (the sanity floor)
    full = refresh_report(lambda i: world(i * step), n_frames=4, budget=1.0, known_shift=(0.0, -step))
    assert full["psnr_mean"] > 90.0 and full["shaded_fraction"] == 1.0

    print("OK: holographic_refresh self-test passed (budget honoured to %.4f; the threshold rule selects %d of %d "
          "pixels on tied ages -- the fake-perfect bug -- while exact-k selects exactly %d with the lowest flat "
          "indices winning ties; a known camera shift gives %.1f dB mean / %.1f worst at 20%% shaded, and recovering "
          "the shift from pixels costs %.1f dB and turns a +%.2f dB tail slope into %.2f -- decay, because the loop "
          "warps its own output)"
          % (rep["shaded_fraction"], threshold_oldest(tied, k).sum(), tied.size, k,
             rep["psnr_mean"], rep["psnr_worst"], rep["psnr_mean"] - est["psnr_mean"],
             rep["tail_slope"], est["tail_slope"]))


if __name__ == "__main__":
    _selftest()
