"""MCMC birth-death relocation -- conserve capacity instead of dropping it (holographic_relocate).

THE REFINEMENT (from the 3DGS-concepts sweep)
---------------------------------------------
3D-Gaussian-Splatting-as-MCMC (Kheradmand et al. 2024) replaces heuristic clone/split/prune with a principled
birth-death move: a DEAD (low-opacity, contributing-nothing) Gaussian is RELOCATED to a high-density region -- the
position of a live Gaussian sampled in proportion to opacity -- rather than dropped, so a FIXED Gaussian budget is
never wasted on dead samples. The sample count is conserved; capacity is redistributed toward where the data is.

holostuff's bounded memory does the opposite: when a store is full it EVICTS THE RAREST -- the lowest-count prototype
is DELETED (the creature's `memory_cap` path: "bounded memory: forget the rarest"). Eviction DROPS capacity; a
birth-death move CONSERVES it -- move the dead atom to an under-represented region instead. This module is that move
on the engine's "Gaussians-as-samples" structure (splats), and it ties straight into the B10 generative-denoising
sampler: birth-death IS an MCMC move, the discrete kin of running the cleanup backwards from noise.

WHAT IT PROVIDES
  * birth_death_relocate(splats, target, dead_frac) -- find the DEAD splats (|amplitude| below dead_frac of the
    largest) and RELOCATE each to the current residual peak (the most under-represented region), subtracting after
    each so successive relocations find distinct peaks. The splat COUNT is conserved (dead splats are moved, not
    removed); amplitudes are re-solved jointly at the end (splat_refit).

THE MEASUREMENT BAR (checked exactly in the self-test)
  * RELOCATE beats DROP: at a fixed budget with dead splats, relocating them to under-represented regions reaches a
    far lower reconstruction error than evicting them (which leaves the budget shrunk and unused) -- ~4x here.
  * The TARGET matters: relocating to the residual peak (high density / under-represented) beats relocating to a
    random location -- it is the MCMC-principled destination, not the move alone, that wins.
  * COUNT is conserved: the same number of splats comes out as went in (capacity is redistributed, not dropped).

DETERMINISM (per ISA.md)
  Argmax residual-peak finding + the existing joint refit -- no RNG in the relocation. Same splats and target give
  the same relocated set (asserted).

KEPT NEGATIVES (loud)
  * This is the SUCCESSOR to evict-rarest -- the DROP was already in the box (the creature's bounded memory); the new
    part is CONSERVING capacity by relocating a dead atom to an under-represented region instead of deleting it.
  * ISOTROPIC splats (the engine's "Gaussians-as-samples" structure). The same drop-vs-relocate choice applies to the
    creature's prototype eviction and any bounded store -- that broader wiring is noted, not done here (the creature's
    eviction is left unchanged; this is an additive faculty).
  * The relocation TARGET is the residual peak (a deterministic stand-in for 3DGS-MCMC's opacity-weighted sampling of
    a live Gaussian's position). It redistributes toward high RESIDUAL, which is the under-served region, not toward
    high reconstructed density per se -- the right target for COVERAGE, the honest simplification of the MCMC move.
  * If there are NO dead splats (the budget is already fully used), relocation is a NO-OP -- it ties keeping the set
    (there is nothing to redistribute). The win exists only when capacity is being wasted.
"""

import numpy as np

from holographic_splat import _gaussian, splat_render, splat_refit


def birth_death_relocate(splats, target, dead_frac=0.05):
    """Relocate the DEAD splats (|amplitude| below `dead_frac` of the largest) to under-represented regions -- each
    to the current residual peak, subtracting after each so successive relocations find distinct peaks -- CONSERVING
    the splat count (dead splats are moved, not removed). The birth-death successor to evict-rarest: redistribute
    wasted capacity instead of dropping it. Returns the new splat list (amplitudes re-solved jointly)."""
    target = np.asarray(target, float)
    shape = target.shape
    if not splats:
        return list(splats)
    sp = [list(s) for s in splats]
    amps = np.abs([s[2] for s in sp])
    thr = dead_frac * amps.max()
    dead_idx = [i for i in range(len(sp)) if amps[i] < thr]
    if not dead_idx:
        return [tuple(s) for s in sp]                       # nothing wasted -> no-op (ties keeping the set)
    residual = target - splat_render(sp, shape)
    for i in dead_idx:
        py, px = np.unravel_index(np.abs(residual).argmax(), shape)   # the most under-represented region
        g = _gaussian(shape, py, px, sp[i][3])
        a = float((residual * g).sum())
        sp[i] = [int(py), int(px), a, sp[i][3]]
        residual = residual - a * g                         # subtract so the next relocation finds a NEW peak
    return splat_refit([tuple(s) for s in sp], target)


# =====================================================================================================
# Self-test -- relocate beats drop, the residual target beats random, the count is conserved.
# =====================================================================================================
def _mse(a, b):
    return float(((a - b) ** 2).mean())


def _selftest():
    ys, xs = np.mgrid[0:64, 0:64]
    target = sum(np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 12.0))
                 for cx, cy in [(16, 16), (48, 16), (16, 48), (48, 48), (32, 32), (32, 10)])
    shape = target.shape
    useful = [(16, 16, 0.0, 3.5), (48, 16, 0.0, 3.5), (16, 48, 0.0, 3.5),
              (48, 48, 0.0, 3.5), (32, 32, 0.0, 3.5), (32, 10, 0.0, 3.5)]
    dead = [(2, 2, 0.0, 1.0)] * 6                            # 6 splats jammed in the corner -> dead on refit
    splats = splat_refit(useful + dead, target)
    amps = np.abs([s[2] for s in splats])
    thr = 0.05 * amps.max()
    n_dead = int((amps < thr).sum())
    assert n_dead >= 5, f"the test needs dead splats to relocate, got {n_dead}"

    keep = _mse(splat_render(splats, shape), target)
    drop = _mse(splat_render(splat_refit([s for s in splats if abs(s[2]) >= thr], target), shape), target)
    reloc = birth_death_relocate(splats, target)
    reloc_mse = _mse(splat_render(reloc, shape), target)

    # relocate to RANDOM (control) -- the move without the principled target
    def relocate_random(splats, target):
        sp = [list(s) for s in splats]
        a = np.abs([s[2] for s in sp])
        t = 0.05 * a.max()
        rng = np.random.default_rng(0)
        residual = target - splat_render(sp, shape)
        for i in [k for k in range(len(sp)) if a[k] < t]:
            py, px = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
            g = _gaussian(shape, py, px, sp[i][3])
            amp = float((residual * g).sum())
            sp[i] = [int(py), int(px), amp, sp[i][3]]
            residual = residual - amp * g
        return splat_refit([tuple(s) for s in sp], target)

    rand_mse = _mse(splat_render(relocate_random(splats, target), shape), target)

    assert reloc_mse < drop * 0.6, f"relocate must clearly beat drop: relocate {reloc_mse}, drop {drop}"
    assert reloc_mse < rand_mse * 0.6, f"the residual target must beat random relocation: {reloc_mse} vs {rand_mse}"
    assert len(reloc) == len(splats), "relocation must CONSERVE the splat count (move, not drop)"

    # --- no dead splats -> no-op ---
    tight = splat_refit([(16, 16, 0.0, 3.5), (48, 48, 0.0, 3.5)],
                        np.exp(-(((xs - 16) ** 2 + (ys - 16) ** 2) / 12.0)) +
                        np.exp(-(((xs - 48) ** 2 + (ys - 48) ** 2) / 12.0)))
    noop = birth_death_relocate(tight, np.exp(-(((xs - 16) ** 2 + (ys - 16) ** 2) / 12.0)) +
                                np.exp(-(((xs - 48) ** 2 + (ys - 48) ** 2) / 12.0)))
    assert len(noop) == len(tight), "no dead splats -> count unchanged"

    # --- determinism ---
    assert all(np.allclose(a[:3], b[:3]) and a[3] == b[3]
               for a, b in zip(birth_death_relocate(splats, target), birth_death_relocate(splats, target)))

    print(f"holographic_relocate selftest: ok (budget 12, {n_dead} dead: RELOCATE -> residual peaks {reloc_mse:.5f} "
          f"beats DROP {drop:.5f} (~{drop / reloc_mse:.1f}x, evict shrinks the budget) and beats RANDOM relocation "
          f"{rand_mse:.5f} (the target matters); count conserved {len(splats)}->{len(reloc)}; no-dead is a no-op; "
          f"deterministic. The birth-death successor to evict-rarest -- conserve capacity, don't drop it)")


if __name__ == "__main__":
    _selftest()
