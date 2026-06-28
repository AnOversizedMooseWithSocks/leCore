"""Clone-vs-split density control -- scale-aware splat densification (holographic_splatdensify).

THE REFINEMENT (from the 3DGS-concepts sweep)
---------------------------------------------
3D Gaussian splatting densifies where reconstruction error is high, but it DISTINGUISHES two moves by the gradient
AND the scale of the splat:
  * CLONE -- high error at a SMALL splat: duplicate-and-nudge a copy toward the residual to COVER an under-served
    region the small splat does not reach. The original is kept; a same-scale copy extends coverage.
  * SPLIT -- high error at a WIDE splat: subdivide it into two NARROWER splats where the data has more structure than
    the wide splat can resolve. The original is removed; two narrower splats RESOLVE the fine detail it was smearing.

holostuff already densifies (`densify_fit` places fresh splats on the residual, `adaptive_fit` runs to a noise floor),
but that placement is SCALE-BLIND: it adds capacity where the error is, without asking whether the region needs
COVERING (a new small splat) or RESOLVING (subdividing a too-wide one). This module adds exactly that distinction --
it sharpens WHERE new capacity goes. A refinement of shipped machinery, not a new mechanism.

WHY THE DISTINCTION MATTERS (measured -- the wrong move can be WORSE than nothing):
  * On an elongated RIDGE fit by one small splat (an under-covered region), CLONE lowers MSE 0.0143 -> 0.0115 while
    SPLIT RAISES it to 0.0153 -- subdividing the small splat loses coverage, ending up worse than the baseline.
  * On TWO fine peaks smeared by one wide splat, SPLIT lowers MSE 0.0037 -> 0.0010 decisively while CLONE barely
    moves it (0.0035) -- adding another wide splat does not resolve the peaks.
  * On a MIXED target (a ridge needing cover + twin peaks needing resolve), the scale-aware rule reaches MSE 0.00704
    at a fixed splat budget, beating always-clone (0.00799, misses the peaks) and always-split (0.00917, hurts the
    ridge). Each blind strategy handles only one error type; the scale rule does the right move for each.

WHAT IT PROVIDES
  * clone_splat(splat, residual, shape) -- the COVER primitive: a same-scale splat at the residual peak in the
    splat's footprint (the original is kept by the caller).
  * split_splat(splat, residual, shape) -- the RESOLVE primitive: two narrower (sigma/1.6) splats at the two largest
    residual peaks in the footprint (the original is removed by the caller).
  * clone_split_densify(splats, target, n_densify, scale_thresh) -- the scale-aware densification: rank splats by the
    residual energy in their footprint, and for the highest-error ones CLONE if narrow (< scale_thresh) else SPLIT.
    `n_densify` caps how many splats to densify (default all); `scale_thresh` defaults to the set's median sigma.

DETERMINISM (per ISA.md)
  Argmax peak-finding, a fixed scale rule, and the existing joint amplitude refit -- no RNG. Same splats and target
  give the same densified set (asserted).

KEPT NEGATIVES (loud)
  * This REFINES `densify_fit`'s residual placement -- the "add capacity where error is high" half was already
    shipped; the new part is the scale-aware COVER-vs-RESOLVE decision, and the measurement that the wrong move can
    be worse than nothing.
  * ISOTROPIC splats (the engine's (cy,cx,amp,sigma) splat domain) -- the scale is sigma directly; an anisotropic
    cover-vs-resolve (per-axis) is the natural extension, not implemented here.
  * The scale threshold is a HEURISTIC (the set's median sigma). On a SINGLE-SCALE target the distinction is moot --
    every splat is the same width, so clone-vs-split has nothing to choose between (the win needs mixed scales).
  * Split positions come from the two largest residual peaks in the footprint (a deterministic stand-in for 3DGS's
    sampling the original Gaussian) -- adequate for two-peak resolution, not a general multi-modal placement.
"""

import numpy as np

from holographic_splat import _gaussian, splat_render, splat_refit


def _peak_in_window(field, cy, cx, s, shape, suppress=None):
    """The location of the largest |field| within ~2*sigma of (cy,cx); optionally suppress a disc around an earlier
    peak so a second call finds a DISTINCT peak (for split's two narrower splats)."""
    rad = int(max(2 * s, 3))
    win = np.zeros_like(field)
    y0, y1 = max(0, cy - rad), min(shape[0], cy + rad + 1)
    x0, x1 = max(0, cx - rad), min(shape[1], cx + rad + 1)
    win[y0:y1, x0:x1] = np.abs(field[y0:y1, x0:x1])
    if suppress is not None:
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        win[((yy - suppress[0]) ** 2 + (xx - suppress[1]) ** 2) < (s * s)] = 0
    return np.unravel_index(win.argmax(), win.shape)


def clone_splat(splat, residual, shape):
    """COVER: a same-scale splat at the residual peak in this splat's footprint. The original is kept by the caller;
    the clone extends coverage into an under-served region the small splat does not reach."""
    cy, cx, amp, s = splat
    py, px = _peak_in_window(residual, cy, cx, s, shape)
    g = _gaussian(shape, py, px, s)
    return [(int(py), int(px), float((residual * g).sum()), s)]


def split_splat(splat, residual, shape):
    """RESOLVE: two NARROWER (sigma/1.6) splats at the two largest residual peaks in this splat's footprint. The
    original wide splat is removed by the caller; the two narrow ones resolve fine structure it was smearing."""
    cy, cx, amp, s = splat
    ns = s / 1.6
    p1 = _peak_in_window(residual, cy, cx, s, shape)
    p2 = _peak_in_window(residual, cy, cx, s, shape, suppress=p1)
    return [(int(p1[0]), int(p1[1]), 0.0, ns), (int(p2[0]), int(p2[1]), 0.0, ns)]


def clone_split_densify(splats, target, n_densify=None, scale_thresh=None):
    """Scale-aware densification: rank the splats by the residual energy in their footprint, and for the highest-error
    ones CLONE if narrow (sigma < scale_thresh, to COVER) else SPLIT (to RESOLVE). `n_densify` caps how many splats to
    densify (default all); `scale_thresh` defaults to the set's median sigma. Amplitudes are re-solved jointly at the
    end (splat_refit). Returns the new splat list."""
    target = np.asarray(target, float)
    shape = target.shape
    if not splats:
        return list(splats)
    residual = target - splat_render(splats, shape)
    if scale_thresh is None:
        scale_thresh = float(np.median([s for *_, s in splats]))

    def footprint_error(sp):
        cy, cx, amp, s = sp
        rad = int(max(2 * s, 3))
        y0, y1 = max(0, cy - rad), min(shape[0], cy + rad + 1)
        x0, x1 = max(0, cx - rad), min(shape[1], cx + rad + 1)
        return float((residual[y0:y1, x0:x1] ** 2).sum())

    order = sorted(range(len(splats)), key=lambda i: -footprint_error(splats[i]))
    chosen = set(order if n_densify is None else order[:n_densify])

    out = []
    for i, sp in enumerate(splats):
        if i not in chosen:
            out.append(sp)
            continue
        s = sp[3]
        if s >= scale_thresh:
            out += split_splat(sp, residual, shape)            # wide & high-error -> RESOLVE (original removed)
        else:
            out += [sp] + clone_splat(sp, residual, shape)     # narrow & high-error -> COVER (original kept)
    return splat_refit(out, target)


# =====================================================================================================
# Self-test -- clone wins for cover, split wins for resolve, scale-aware beats both blind strategies.
# =====================================================================================================
def _mse(a, b):
    return float(((a - b) ** 2).mean())


def _selftest():
    ys, xs = np.mgrid[0:64, 0:64]

    # --- CLONE wins for COVER: an elongated ridge fit by one small splat ---
    ridge = np.exp(-(((xs - 24) ** 2) / 6.0 + ((ys - 32) ** 2) / 120.0))
    sA = splat_refit([(32, 24, 0.0, 1.0)], ridge)
    base_a = _mse(splat_render(sA, ridge.shape), ridge)
    res_a = ridge - splat_render(sA, ridge.shape)
    clone_a = _mse(splat_render(splat_refit(sA + clone_splat(sA[0], res_a, ridge.shape), ridge), ridge.shape), ridge)
    split_a = _mse(splat_render(splat_refit(split_splat(sA[0], res_a, ridge.shape), ridge), ridge.shape), ridge)
    assert clone_a < base_a and clone_a < split_a, f"clone must win for cover: base {base_a}, clone {clone_a}, split {split_a}"

    # --- SPLIT wins for RESOLVE: two fine peaks smeared by one wide splat ---
    twin = (np.exp(-(((xs - 30) ** 2 + (ys - 30) ** 2) / 4.0)) + np.exp(-(((xs - 38) ** 2 + (ys - 30) ** 2) / 4.0)))
    sB = splat_refit([(30, 34, 0.0, 3.5)], twin)
    res_b = twin - splat_render(sB, twin.shape)
    clone_b = _mse(splat_render(splat_refit(sB + clone_splat(sB[0], res_b, twin.shape), twin), twin.shape), twin)
    split_b = _mse(splat_render(splat_refit(split_splat(sB[0], res_b, twin.shape), twin), twin.shape), twin)
    assert split_b < clone_b, f"split must win for resolve: clone {clone_b}, split {split_b}"

    # --- SCALE-AWARE beats both blind strategies on a MIXED target (same splat budget) ---
    ridge2 = np.exp(-(((xs - 14) ** 2) / 6.0 + ((ys - 32) ** 2) / 120.0))
    twin2 = (np.exp(-(((xs - 46) ** 2 + (ys - 30) ** 2) / 4.0)) + np.exp(-(((xs - 52) ** 2 + (ys - 30) ** 2) / 4.0)))
    target = ridge2 + twin2
    splats = splat_refit([(32, 14, 0.0, 1.0), (30, 49, 0.0, 3.5)], target)
    shape = target.shape
    res = target - splat_render(splats, shape)
    med = float(np.median([s for *_, s in splats]))

    def blind(strategy):
        out = []
        for sp in splats:
            if strategy == "split":
                out += split_splat(sp, res, shape)
            else:
                out += [sp] + clone_splat(sp, res, shape)
        return _mse(splat_render(splat_refit(out, target), shape), target)

    scale_mse = _mse(splat_render(clone_split_densify(splats, target), shape), target)
    assert scale_mse < blind("clone") and scale_mse < blind("split"), \
        f"scale-aware must beat both blind: scale {scale_mse}, clone {blind('clone')}, split {blind('split')}"

    # --- determinism ---
    d1 = clone_split_densify(splats, target)
    d2 = clone_split_densify(splats, target)
    assert all(np.allclose(a[:3], b[:3]) and a[3] == b[3] for a, b in zip(d1, d2))

    print(f"holographic_splatdensify selftest: ok (COVER -- ridge: clone {clone_a:.5f} beats split {split_a:.5f} "
          f"(split worse than base {base_a:.5f}); RESOLVE -- twin peaks: split {split_b:.5f} beats clone {clone_b:.5f}; "
          f"MIXED -- scale-aware {scale_mse:.5f} beats always-clone {blind('clone'):.5f} / always-split "
          f"{blind('split'):.5f} at fixed budget; deterministic)")


if __name__ == "__main__":
    _selftest()
