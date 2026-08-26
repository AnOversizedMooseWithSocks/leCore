"""Holographic Gaussian splatting (B8): a scene as a superposition of Gaussian primitives."""
import numpy as np
import pytest
from holographic.rendering.holographic_splat import splat_fit, splat_render, splat_refit, splat_denoise, psnr, adaptive_fit


def _target(G=48, seed=0):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:G, 0:G]
    T = np.zeros((G, G))
    for _ in range(4):                         # a smooth few-blob target splats can represent
        cy, cx, s, a = rng.uniform(8, G - 8, 2).tolist() + [rng.uniform(3, 7), rng.uniform(0.5, 1)]
        T += a * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * s * s))
    return T / T.max()


def test_more_splats_reconstruct_better_and_compactly():
    T = _target()
    q8 = psnr(T, splat_render(splat_fit(T, 8), T.shape))
    q40 = psnr(T, splat_render(splat_fit(T, 40), T.shape))
    assert q40 > q8 and q40 > 25.0             # superposition of primitives reconstructs the field


def test_splatting_denoises():
    # fitting few smooth Gaussians to noisy data recovers the clean field (no capacity for noise).
    rng = np.random.default_rng(2)
    T = _target()
    noisy = T + 0.10 * rng.standard_normal(T.shape)
    assert psnr(T, splat_denoise(noisy, 30)) > psnr(T, noisy) + 1.0


def test_rbf_encoder_is_a_gaussian_splat_in_hv_space():
    # the bridge: holostuff's RBF scalar encoder already places a Gaussian bump in similarity space.
    from holographic.io_and_interop.holographic_encoders import ScalarEncoder
    enc = ScalarEncoder(512, lo=0.0, hi=1.0, kernel="rbf", seed=0)
    c = enc.encode(0.5)
    vals = np.linspace(0, 1, 21)
    sims = np.array([float(np.dot(c, enc.encode(v)) /
                     (np.linalg.norm(c) * np.linalg.norm(enc.encode(v)) + 1e-12)) for v in vals])
    assert abs(vals[int(sims.argmax())] - 0.5) < 0.06   # peaks at the encoded value
    assert sims.max() > 0.95 and sims.min() < sims.max() - 0.2   # smooth Gaussian-like falloff


def _hard_target(N=64):
    # a hard-edged square + a smooth blob: greedy MP overlap double-counting shows up at the edge.
    ys, xs = np.mgrid[0:N, 0:N] / N
    T = np.zeros((N, N)); T[(xs > 0.25) & (xs < 0.68) & (ys > 0.25) & (ys < 0.68)] = 1.0
    T += 0.8 * np.exp(-((xs - 0.72) ** 2 + (ys - 0.74) ** 2) / (2 * 0.10 ** 2))
    return np.clip(T, 0, 1)


def test_joint_refit_beats_greedy_and_gain_grows_with_count():
    # the 'looping': re-solving amplitudes jointly removes greedy MP's overlap double-counting.
    T = _hard_target()
    gains = []
    for K in (80, 400):
        greedy = splat_fit(T, K)
        gq = psnr(T, splat_render(greedy, T.shape))
        rq = psnr(T, splat_render(splat_refit(greedy, T), T.shape))
        assert rq > gq + 1.0                       # joint refit is a clear win
        gains.append(rq - gq)
    assert gains[1] > gains[0]                      # and the gain grows with the splat count


def test_splat_fit_refit_flag_matches_manual_refit():
    T = _hard_target()
    placed = splat_fit(T, 120)
    via_flag = splat_fit(T, 120, refit=True)        # same positions/scales, amplitudes re-solved
    via_manual = splat_refit(placed, T)
    assert np.allclose([s[2] for s in via_flag], [s[2] for s in via_manual])
    assert psnr(T, splat_render(via_flag, T.shape)) > psnr(T, splat_render(placed, T.shape))


def test_splat_refit_handles_empty():
    assert splat_refit([], _hard_target()) == []


def test_adaptive_fit_count_tracks_content_at_matched_quality():
    """ADAPT-1: with a noise threshold the splat COUNT adapts to content -- a simple field finishes in far
    fewer splats than a busy one, both at matched quality (V-Ray's adaptive sampler: sample to a noise floor,
    not a budget). The fixed-k baseline, by contrast, over-spends on the easy field and starves the busy one."""
    ys, xs = np.mgrid[0:48, 0:48] / 48.0
    def bump(cy, cx, s, a=1.0):
        return a * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * s * s))
    simple = bump(.5, .5, .18); simple /= simple.max()
    busy = sum(bump(*p) for p in [(.25, .25, .07), (.3, .7, .06), (.7, .3, .06), (.72, .72, .05),
                                  (.5, .5, .05), (.2, .55, .05), (.6, .15, .05)])
    busy /= busy.max()
    sp_s, k_s = adaptive_fit(simple, noise_thresh=0.03)
    sp_b, k_b = adaptive_fit(busy, noise_thresh=0.03)
    assert k_b > k_s + 5                                       # the busy field genuinely needs more splats
    q_s = psnr(simple, splat_render(sp_s, simple.shape))
    q_b = psnr(busy, splat_render(sp_b, busy.shape))
    assert abs(q_s - q_b) < 4.0                               # matched quality (both hit the same noise floor)
    assert q_s > 25 and q_b > 25                              # and both are genuinely good reconstructions


def test_adaptive_fit_respects_bounds():
    """ADAPT-1 bounds: k_min is honoured (and a smooth field converges below k_max), while a HARD-EDGED target
    -- which the smooth isotropic basis cannot drive to a low residual -- runs all the way to k_max. The kept
    caveat: the adaptive count is meaningful only for fields the Gaussian basis can actually represent."""
    ys, xs = np.mgrid[0:48, 0:48] / 48.0
    smooth = np.exp(-((xs - .5) ** 2 + (ys - .5) ** 2) / (2 * .2 ** 2))
    _, k = adaptive_fit(smooth, noise_thresh=0.03, k_min=6, k_max=50)
    assert 6 <= k < 50                                        # honoured k_min, converged before the ceiling
    hard = ((xs > .3) & (xs < .7) & (ys > .3) & (ys < .7)).astype(float)
    _, k_hard = adaptive_fit(hard, noise_thresh=0.005, k_min=4, k_max=25)
    assert k_hard == 25                                       # the smooth basis can't resolve a hard edge -> k_max


def _occupancy_target(S=96, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    img = np.zeros((S, S))
    for _ in range(8):
        cy, cx, sig, amp = rng.uniform(10, S-10), rng.uniform(10, S-10), rng.uniform(5, 12), rng.uniform(0.5, 1.0)
        img += amp * np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sig**2))
    return img / img.max()


def test_single_splat_bundle_region_recall_caps_with_resolution():
    # The kept negative that motivates tiling: the bundled scene's region readback is decode-via-cleanup, so
    # as the grid gets finer the bundle crosstalk grows and recall accuracy falls (here 100% -> ~75% by grid 32).
    from holographic.rendering.holographic_splat import splat_bundle, recall_region, splat_fit
    splats = splat_fit(_occupancy_target(), 30)

    def acc(grid):
        hv, ctx = splat_bundle(splats, (96, 96), dim=4096, grid=grid, levels=5, seed=0)
        return sum(abs(recall_region(hv, (gy, gx), ctx) - ctx["desc"][(gy, gx)]) < 1e-9
                   for gy in range(grid) for gx in range(grid)) / (grid*grid)

    assert acc(8) == 1.0                         # comfortably under the cap
    assert acc(32) < 0.85                        # the cap bites at fine resolution -- the negative


def test_tiled_splat_bundle_holds_region_recall_at_fine_resolution():
    # Tiling fixes it: each cell routes to a tile bundle of at most tile*tile bindings, so recall stays ~100%
    # at a resolution where the single bundle has fallen to ~75%.
    from holographic.rendering.holographic_splat import splat_bundle_tiled, recall_region_tiled, splat_fit
    splats = splat_fit(_occupancy_target(), 30)
    scene = splat_bundle_tiled(splats, (96, 96), dim=4096, grid=32, levels=5, tile=8, seed=0)
    acc = sum(abs(recall_region_tiled(scene, (gy, gx)) - scene["desc"][(gy, gx)]) < 1e-9
              for gy in range(32) for gx in range(32)) / (32*32)
    assert acc > 0.99                            # tiling holds recall where the single bundle capped
    assert len(scene["tiles"]) == 16            # 32/8 = 4 tiles per side


def test_tiled_splat_bundle_is_deterministic_and_empty_safe():
    from holographic.rendering.holographic_splat import splat_bundle_tiled, recall_region_tiled, splat_fit
    splats = splat_fit(_occupancy_target(), 30)
    a = splat_bundle_tiled(splats, (96, 96), dim=2048, grid=16, tile=8, seed=0)
    b = splat_bundle_tiled(splats, (96, 96), dim=2048, grid=16, tile=8, seed=0)
    assert all(np.array_equal(a["tiles"][k], b["tiles"][k]) for k in a["tiles"])      # bit-identical run-to-run
    assert recall_region_tiled({"tiles": {}, "tile": 8, "roles": None, "lvl": None, "levels": 5}, (0, 0)) == 0.0


def test_tiled_splat_migration_is_byte_identical_to_the_old_inline_tiling():
    """PARITY: splat_bundle_tiled now delegates its tiling to the shared TiledStore + _tile_bucket. This
    proves the delegation changed NOTHING -- the per-tile bundle vectors are bit-for-bit what the old
    inline (gy//tile, gx//tile) grouping produced. Reuse in place of the bespoke tiler, byte-identical."""
    from holographic.rendering.holographic_splat import splat_bundle_tiled, splat_fit, splat_render
    from holographic.agents_and_reasoning.holographic_ai import bind, bundle, Vocabulary
    splats = splat_fit(_occupancy_target(), 30)
    shape, dim, grid, levels, tile, seed = (96, 96), 2048, 16, 5, 8, 0
    scene = splat_bundle_tiled(splats, shape, dim=dim, grid=grid, levels=levels, tile=tile, seed=seed)

    # recompute the tiles with the ORIGINAL inline logic, independently
    H, W = shape
    rendered = splat_render(splats, (H, W))
    roles, lvl = Vocabulary(dim, seed=seed), Vocabulary(dim, seed=seed + 1)
    peak = float(np.abs(rendered).max()) + 1e-12
    tile_parts = {}
    for gy in range(grid):
        for gx in range(grid):
            ys, ye = gy * H // grid, (gy + 1) * H // grid
            xs, xe = gx * W // grid, (gx + 1) * W // grid
            energy = float(np.clip(np.abs(rendered[ys:ye, xs:xe]).max() / peak, 0.0, 1.0))
            q = int(round(energy * (levels - 1)))
            tile_parts.setdefault((gy // tile, gx // tile), []).append(
                bind(roles.get(f"cell:{gy}:{gx}"), lvl.get(f"lvl:{q}")))
    expected = {k: bundle(v) for k, v in tile_parts.items()}

    assert set(scene["tiles"].keys()) == set(expected.keys())
    for k in expected:
        assert np.array_equal(scene["tiles"][k], expected[k]), f"tile {k} changed under migration"


# ======================================================================================================
# H8 -- frequency-lifted (Gabor) splats. The lift is real; the backlog's evidence for it was not.
# ======================================================================================================
def _h8_targets(n=48):
    ys, xs = np.mgrid[0:n, 0:n]
    disk = (np.sqrt((ys - n / 2 + .5) ** 2 + (xs - n / 2 + .5) ** 2) < n * 0.28).astype(float)
    stripes = (np.sin(2 * np.pi * 5 * xs / n) > 0).astype(float)
    return disk, stripes


def test_a_zero_frequency_gabor_atom_is_exactly_a_gaussian():
    """The dictionaries NEST. That is what makes the Gaussian-vs-Gabor comparison honest: the greedy fit can always
    fall back on a Gaussian, so Gabor can never lose per primitive, and any win is a real win."""
    from holographic.rendering.holographic_splat import _gabor, _gaussian
    a = _gabor((32, 32), 16, 16, 3.0, 0.0, 0.0, 0.0)
    g = _gaussian((32, 32), 16, 16, 3.0)
    assert np.max(np.abs(a - g)) < 1e-12


def test_the_saturated_gaussian_baseline_was_a_strawman_refit_makes_it_climb():
    """The backlog's headline was 'the Gaussian basis saturates at 11.6 dB regardless of K'. That flat curve is
    greedy matching pursuit's overlap double-counting -- and splat_refit, already in the same module, removes it."""
    from holographic.rendering.holographic_splat import psnr, splat_fit, splat_refit, splat_render
    disk, _ = _h8_targets()
    mp = [psnr(splat_render(splat_fit(disk, K), disk.shape), disk) for K in (24, 96)]
    rf = [psnr(splat_render(splat_refit(splat_fit(disk, K), disk), disk.shape), disk) for K in (24, 96)]
    assert rf[1] > rf[0], rf                                 # the refit basis climbs with K...
    assert (rf[1] - rf[0]) > 2.0 * (mp[1] - mp[0]), (mp, rf)  # ...far faster than the strawman did


def test_a_gabor_atom_is_a_bandpass_primitive_so_the_win_is_content_dependent():
    """At equal PARAMETER budget (Gabor 7 numbers/atom, Gaussian 4): a grating IS a band and the carrier locks onto
    it; a sharp edge is EVERY band at once and there is nothing to lock onto."""
    from holographic.rendering.holographic_splat import (gabor_fit, gabor_render, psnr, splat_fit,
                                                         splat_refit, splat_render)
    disk, stripes = _h8_targets()

    def delta(target, K):
        gauss = splat_render(splat_refit(splat_fit(target, int(round(K * 7 / 4))), target), target.shape)
        gab = gabor_render(gabor_fit(target, K), target.shape)
        return psnr(gab, target) - psnr(gauss, target)

    d_stripes, d_disk = delta(stripes, 64), delta(disk, 64)
    assert d_stripes > 2.0, d_stripes
    assert d_disk < d_stripes / 2.0, (d_disk, d_stripes)


def test_the_splatsharpen_negative_survives_on_a_broadband_edge():
    """KEPT NEGATIVE, and the backlog predicted it would fall. Widening a basis pays only when the widening MATCHES
    the content's structure. On the very target the negative was recorded against, the lift barely moves."""
    from holographic.rendering.holographic_splat import (gabor_fit, gabor_render, psnr, splat_fit,
                                                         splat_refit, splat_render)
    disk, _ = _h8_targets()
    gauss = splat_render(splat_refit(splat_fit(disk, 112), disk), disk.shape)   # equal budget: 64*7 == 112*4
    gab = gabor_render(gabor_fit(disk, 64), disk.shape)
    assert psnr(gab, disk) - psnr(gauss, disk) < 2.5                            # not the dissolution promised


def test_spectral_detail_measures_what_psnr_cannot():
    """PSNR lives in the low frequencies, where the energy is. The splatsharpen negative is a statement about the
    HIGH ones, so it needs its own number."""
    from holographic.rendering.holographic_splat import spectral_energy_fraction
    n = 48
    ys, xs = np.mgrid[0:n, 0:n]
    grating = (np.sin(2 * np.pi * 5 * xs / n) > 0).astype(float)
    blob = np.exp(-((ys - n / 2) ** 2 + (xs - n / 2) ** 2) / (2 * 8.0 ** 2))
    assert spectral_energy_fraction(grating) > 10 * spectral_energy_fraction(blob)
    assert 0.0 <= spectral_energy_fraction(blob) < 1e-3


def test_gabor_basis_through_the_mind_and_its_guards():
    from holographic.misc.holographic_unified import UnifiedMind
    from holographic.rendering.holographic_splat import psnr
    m = UnifiedMind(dim=64, seed=0)
    _, stripes = _h8_targets()
    atoms, rendered = m.splat_field(stripes, k=48, basis="gabor")
    assert len(atoms[0]) == 7                                    # seven numbers per primitive, not four
    splats, gauss = m.splat_field(stripes, k=84)                 # equal parameter budget
    assert len(splats[0]) == 4
    assert psnr(rendered, stripes) > psnr(gauss, stripes)
    assert m.spectral_detail(stripes) > m.spectral_detail(gauss)  # the target is sharper than either fit
    with pytest.raises(ValueError):
        m.splat_field(stripes, basis="nope")
    with pytest.raises(ValueError):
        m.splat_field(stripes, basis="gabor", noise_thresh=0.03)  # adaptive count is Gaussian-calibrated
    assert any("gabor" in c.name.lower() for c in m.find_capability("gabor splat"))
