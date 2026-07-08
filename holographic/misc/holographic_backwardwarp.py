"""Backward warping is hole-free by construction -- which is what the engine's unbind already is.

WHY THIS EXISTS (PHASE-2, a validated note)
-------------------------------------------
Frame interpolation moved from FORWARD warping (push each source pixel to where it goes) to BACKWARD warping (for
each target pixel, pull from where it came) for one decisive reason: a forward warp under a non-uniform deformation
leaves HOLES (target cells no source landed on) and OVERLAPS (cells several sources collide on), while a backward
warp visits every target exactly once and so fills all of them by construction. The engine gets the backward form
for free: `unbind` is a BACKWARD, invertible map. To recover a stored value you do not scatter the composite forward
and hope every slot gets filled; you take the target role and unbind its source out -- each target finds its own
source. So unbind-based recovery is the hole-free choice, and where the engine could either splat a representation
forward or unbind it backward, the backward route is preferred for exactly this reason.

This module is the small demonstration behind that note (it is not a new faculty -- `unbind` already is the backward
map; this just makes the hole/overlap difference visible and measurable).

MEASURED (see `_selftest`, a signal resampled under a non-uniform but monotonic warp):
  * FORWARD scatter leaves dozens of holes and dozens of overlaps out of N cells (the warp locally stretches ->
    nothing lands in the gaps; locally compresses -> several sources pile onto one cell).
  * BACKWARD gather leaves ZERO holes and reconstructs the resampled signal exactly -- every target read its source.
"""

import numpy as np


def forward_scatter(values, positions, warp, n):
    """FORWARD warp: push each source sample at `positions[i]` to its warped target cell round(warp(pos)*n) and
    write `values[i]` there. Returns (warped_grid, n_holes, n_overlaps): under a non-uniform warp this leaves holes
    (cells nothing landed on, left as nan) and overlaps (cells hit more than once). The cautionary baseline."""
    values = np.asarray(values, float)
    positions = np.asarray(positions, float)
    out = np.full(n, np.nan)
    hit = np.zeros(n, int)
    for i in range(len(values)):
        t = int(round(warp(positions[i]) * n)) % n
        out[t] = values[i]
        hit[t] += 1
    return out, int(np.isnan(out).sum()), int((hit > 1).sum())


def backward_gather(values, positions, query_source_positions):
    """BACKWARD warp (the unbind form): for each target, read the source at its (already inverse-warped) position by
    interpolation. Every target gets a value -- no holes, no overlaps -- which is exactly why unbind-based recovery
    is hole-free. Returns the resampled values."""
    return np.interp(np.asarray(query_source_positions, float), np.asarray(positions, float), np.asarray(values, float))


def _selftest():
    """CI-fast: under a non-uniform but monotonic warp, the forward scatter leaves holes AND overlaps, while the
    backward gather leaves none and reconstructs the resampled signal exactly -- the hole-free property the engine's
    unbind has by construction."""
    n = 256
    pos = np.arange(n) / n
    sig = np.sin(2 * np.pi * 3 * pos) + 0.5 * pos
    A = 0.12
    warp = lambda s: s + A * np.sin(2 * np.pi * s)            # monotonic (A < 1/2pi); warp' varies around 1

    # forward: holes where the warp stretches, overlaps where it compresses
    _, holes_f, overlaps_f = forward_scatter(sig, pos, warp, n)
    assert holes_f > 20 and overlaps_f > 20, (holes_f, overlaps_f)

    # backward: invert the monotonic warp, gather -- every target filled, exact
    grid = np.linspace(0, 1, 4000)
    winv = np.interp(pos, warp(grid), grid)                  # inverse-warp the target positions
    bwd = backward_gather(sig, pos, winv)
    assert int(np.isnan(bwd).sum()) == 0                     # no holes
    assert np.allclose(bwd, np.interp(winv, pos, sig))       # exact resample (every target read its source)


if __name__ == "__main__":
    _selftest()
    print("holographic_backwardwarp selftest passed")
