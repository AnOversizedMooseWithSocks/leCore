"""Splat prune / merge + a quality-budget LOD chain (holographic_splatprune).

WHY THIS MODULE EXISTS
----------------------
From the geometry->stack backlog: the splat representation (holographic_splat) can FIT a field with Gaussian
primitives, densify it, and refit amplitudes -- but nothing yet REDUCES an existing splat set while holding quality.
This is that: prune the negligible splats, merge the redundant ones, and build a level-of-detail chain you can pick
from for a quality budget. It is the splat-domain twin of the mesh LOD policy (holographic_lod): the same
error-budget resolution selection -- there it picked a decimated mesh by screen-space pixels, here it picks a pruned
splat set by reconstruction PSNR.

THE KEY MOVE. Each splat renders as amp * gaussian, and the engine's gaussians are unit-norm, so a splat's
reconstruction ENERGY is exactly amp^2. So "which splats matter" is just "which have the largest |amplitude|" -- and
after dropping the rest, a single joint amplitude REFIT (splat_refit, a closed-form lstsq already in the engine)
lets the survivors absorb the overlap the removed ones were carrying. Contribution-ranked prune + refit degrades
gracefully and dominates naive pruning by a wide margin.

WHAT IT PROVIDES
  * splat_prune(splats, target, keep) -- keep the `keep` highest-contribution splats (by |amp|) and refit. Returns
    the pruned, refitted splat list.
  * splat_merge(splats, target, radius) -- merge splats closer than `radius` into one (amplitude-weighted centre
    and scale, summed amplitude) and refit. Returns the merged, refitted list (fewer splats).
  * splat_lod_chain(splats, target, keeps) -- prune to each count in `keeps`, measuring PSNR at each. Returns a
    fine->coarse list of (splats, count, psnr).
  * select_splat_lod(chain, min_psnr) -- the FEWEST-splat level whose PSNR still meets the budget (the cheapest
    splat set that looks right).

THE MEASUREMENT BAR (checked exactly in the self-test)
  * contribution-ranked prune+refit BEATS naive pruning (random / keep-smallest) by a large PSNR margin at the same
    kept count.
  * the LOD chain degrades gracefully -- more splats, higher PSNR, monotone.
  * merge reduces the splat count with bounded quality loss.
  * select_splat_lod returns the fewest splats meeting a PSNR budget; a tighter budget keeps more splats.

DETERMINISM (per ISA.md)
  Pruning is a stable sort by amplitude; merge is a fixed-order scan; the refit is a closed-form lstsq -- same
  splats and target give the same result (asserted).

KEPT NEGATIVES (loud)
  * No .ply / .spz EXPORTER. Those are 3D-Gaussian-splatting formats (per-splat position, scale, rotation, opacity,
    spherical-harmonic colour); the engine's splats are 2-D field primitives (cy, cx, amp, sigma) -- the format
    does not fit the representation, so shipping it would be a mislabelled stub. Stated, not faked.
  * Prune/merge operate on the ISOTROPIC splat format (splat_fit's output). The anisotropic splats (aniso_fit)
    carry a covariance and have their own optimiser; this does not prune them.
  * |amp| ranking is a proxy for true contribution when splats OVERLAP (the energies are not independent); the
    refit compensates, but a jointly-removable overlapping pair is not detected as such. Good enough in practice,
    not optimal.
  * Merge is lossy by construction (one Gaussian cannot equal two); the radius trades count for quality, and a
    large radius over a busy region loses real structure.
"""

import numpy as np

from holographic.rendering.holographic_splat import splat_render, splat_refit, psnr


def splat_prune(splats, target, keep):
    """Keep the `keep` highest-contribution splats (largest |amplitude|, since each splat's energy is amp^2 for the
    engine's unit-norm gaussians) and REFIT the survivors' amplitudes jointly so they absorb the removed overlap.
    Returns the pruned, refitted splat list. keep >= len(splats) returns a refit of the whole set."""
    target = np.asarray(target, float)
    if keep >= len(splats):
        return splat_refit(list(splats), target)
    order = sorted(range(len(splats)), key=lambda i: -abs(splats[i][2]))   # stable sort by |amp| desc
    kept = [splats[i] for i in order[:keep]]
    return splat_refit(kept, target)


def splat_merge(splats, target, radius):
    """Merge splats whose centres are closer than `radius` into a single splat -- amplitude-weighted centre and
    scale, summed amplitude -- then refit. Returns the merged, refitted list (fewer splats). A fixed-order greedy
    scan, so it is deterministic."""
    target = np.asarray(target, float)
    pts = [tuple(s) for s in splats]
    used = [False] * len(pts)
    out = []
    for i in range(len(pts)):
        if used[i]:
            continue
        group = [pts[i]]
        used[i] = True
        cy, cx = pts[i][0], pts[i][1]
        for j in range(i + 1, len(pts)):
            if used[j]:
                continue
            if (cy - pts[j][0]) ** 2 + (cx - pts[j][1]) ** 2 < radius * radius:
                group.append(pts[j])
                used[j] = True
        wsum = sum(abs(g[2]) for g in group) or 1.0                # amplitude-weighted combination
        mcy = sum(g[0] * abs(g[2]) for g in group) / wsum
        mcx = sum(g[1] * abs(g[2]) for g in group) / wsum
        ms = sum(g[3] * abs(g[2]) for g in group) / wsum
        mamp = sum(g[2] for g in group)
        out.append((int(round(mcy)), int(round(mcx)), float(mamp), float(ms)))
    return splat_refit(out, target)


def splat_lod_chain(splats, target, keeps=(40, 20, 10, 5)):
    """Build a level-of-detail CHAIN by pruning to each count in `keeps`, measuring reconstruction PSNR at each.
    Returns a fine->coarse list of (splats, count, psnr); the first level is the (refitted) full set. Pair with
    select_splat_lod to pick a level for a quality budget."""
    target = np.asarray(target, float)
    shape = target.shape
    full = splat_refit(list(splats), target)
    chain = [(full, len(full), float(psnr(splat_render(full, shape), target)))]
    for k in keeps:
        if k >= chain[-1][1]:
            continue                                               # no coarsening; skip
        pruned = splat_prune(splats, target, k)
        chain.append((pruned, len(pruned), float(psnr(splat_render(pruned, shape), target))))
    return chain


def select_splat_lod(chain, min_psnr):
    """Index of the FEWEST-splat level in `chain` whose PSNR still meets `min_psnr` -- the cheapest splat set that
    looks right. The chain is fine->coarse (PSNR falls), so this scans from the coarse end for the first level that
    clears the budget; falls back to the finest (highest-quality) level if none does."""
    pick = 0                                                       # finest = best quality, the safe fallback
    for i in range(len(chain) - 1, -1, -1):
        if chain[i][2] >= min_psnr:
            pick = i                                               # coarsest (fewest splats) clearing the budget
            break
    return pick


# =====================================================================================================
# Self-test -- contribution prune beats naive; chain degrades gracefully; merge cuts count; selection by budget.
# =====================================================================================================
def _selftest():
    from holographic.rendering.holographic_splat import splat_fit

    def gauss2d(shape, cy, cx, s):
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))

    shape = (64, 64)
    target = (1.0 * gauss2d(shape, 18, 20, 5) + 0.8 * gauss2d(shape, 40, 44, 7)
              + 0.6 * gauss2d(shape, 48, 15, 4) + 0.4 * gauss2d(shape, 12, 50, 3))
    full = splat_fit(target, 60, refit=True)
    full_psnr = psnr(splat_render(full, shape), target)

    # --- contribution-ranked prune+refit beats naive pruning by a wide margin (keep=20) ---
    keep = 20
    contrib = splat_prune(full, target, keep)
    rng = np.random.default_rng(0)
    rand_keep = [full[i] for i in rng.permutation(len(full))[:keep]]
    rand = splat_refit(rand_keep, target)
    worst_keep = [full[i] for i in sorted(range(len(full)), key=lambda i: abs(full[i][2]))[:keep]]
    worst = splat_refit(worst_keep, target)
    p_contrib = psnr(splat_render(contrib, shape), target)
    p_rand = psnr(splat_render(rand, shape), target)
    p_worst = psnr(splat_render(worst, shape), target)
    assert p_contrib > p_rand + 8 and p_contrib > p_worst + 8, \
        f"contribution prune must dominate naive: {p_contrib:.1f} vs rand {p_rand:.1f} / worst {p_worst:.1f}"

    # --- the LOD chain degrades gracefully (monotone in count) ---
    chain = splat_lod_chain(full, target, keeps=(40, 20, 10, 5))
    counts = [c[1] for c in chain]
    psnrs = [c[2] for c in chain]
    assert all(counts[i] > counts[i + 1] for i in range(len(counts) - 1)), "fewer splats down the chain"
    assert all(psnrs[i] >= psnrs[i + 1] - 1e-6 for i in range(len(psnrs) - 1)), f"PSNR must not rise: {psnrs}"

    # --- merge reduces the count with bounded quality loss ---
    merged = splat_merge(full, target, radius=4.0)
    assert len(merged) < len(full), "merge must reduce the splat count"
    assert psnr(splat_render(merged, shape), target) > full_psnr - 12, "merge quality loss must be bounded"

    # --- selection by PSNR budget: tighter budget keeps more splats ---
    loose = select_splat_lod(chain, min_psnr=30.0)
    tight = select_splat_lod(chain, min_psnr=43.0)
    assert chain[tight][1] >= chain[loose][1], "a tighter PSNR budget cannot select fewer splats"
    assert chain[loose][2] >= 30.0, "the selected level must meet the budget"

    # --- determinism ---
    assert [round(a, 6) for a in splat_render(splat_prune(full, target, 15), shape).ravel()[:5]] == \
           [round(a, 6) for a in splat_render(splat_prune(full, target, 15), shape).ravel()[:5]]

    print(f"holographic_splatprune selftest: ok (full {len(full)} splats {full_psnr:.1f} dB; prune to 20 -- "
          f"contribution {p_contrib:.1f} dB DOMINATES random {p_rand:.1f} / worst {p_worst:.1f}; LOD chain counts "
          f"{counts} -> PSNR {[round(p, 1) for p in psnrs]} (graceful); merge to {len(merged)} splats "
          f"{psnr(splat_render(merged, shape), target):.1f} dB; budget-30 keeps {chain[loose][1]}, budget-43 keeps "
          f"{chain[tight][1]}; deterministic)")


if __name__ == "__main__":
    _selftest()
