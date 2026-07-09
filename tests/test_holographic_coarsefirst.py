"""Tests for holographic_coarsefirst -- the coarse-first residual pass (the Group-B unlocker)."""
import numpy as np
from holographic.misc.holographic_coarsefirst import escalate_mask, refine_where_uncertain, gradient_uncertainty, concentration


def test_escalate_mask_by_frac_is_conservative():
    u = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    m = escalate_mask(u, frac=0.4)              # top 40% = top 2 cells
    assert m.tolist() == [False, False, False, True, True]


def test_escalate_mask_by_threshold():
    u = np.array([0.1, 0.5, 0.9])
    assert escalate_mask(u, threshold=0.5).tolist() == [False, True, True]   # >= threshold (conservative)


def test_refine_only_touches_flagged_cells():
    coarse = np.zeros((4, 4))
    unc = np.zeros((4, 4)); unc[0, 0] = 10.0
    refined, mask, n = refine_where_uncertain(coarse, unc, lambda m: np.full(int(m.sum()), 9.0), threshold=5.0)
    assert n == 1 and refined[0, 0] == 9.0 and refined.sum() == 9.0        # only the flagged cell changed


def test_field_approximation_wins_when_concentrated():
    # smooth field + a thin ridge (concentrated structure): refine 20% -> big error drop at a fraction of the cost
    H = W = 64
    ys, xs = np.mgrid[0:H, 0:W] / float(H)
    def f(Y, X): return 0.3 * np.sin(3 * Y) + 0.3 * np.cos(3 * X) + np.exp(-((X - 0.51) ** 2) / 0.0008)
    truth = f(ys, xs)
    cs = 4
    cg = f(ys[::cs, ::cs], xs[::cs, ::cs])
    coarse = np.repeat(np.repeat(cg, cs, axis=0), cs, axis=1)[:H, :W]      # blocky upsample
    unc = gradient_uncertainty(coarse)
    assert concentration(unc) > 0.3
    refined, mask, n = refine_where_uncertain(coarse, unc, lambda m: f(ys, xs), frac=0.2)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    assert rmse(refined, truth) < 0.5 * rmse(coarse, truth)               # refinement recovered most of the error


def test_concentration_low_for_uniform_uncertainty():
    rng = np.random.default_rng(0)
    uniform = np.abs(rng.standard_normal((32, 32))) + 5.0                 # roughly uniform -> low concentration
    spike = np.ones((32, 32)) * 0.01; spike[0, 0] = 100.0                # one hot cell -> high concentration
    assert concentration(uniform) < 0.2
    assert concentration(spike) > 0.8


def test_gradient_uncertainty_flags_edges():
    field = np.zeros((16, 16)); field[:, 8:] = 1.0                       # a vertical edge at column 8
    u = gradient_uncertainty(field)
    assert u[:, 7:9].sum() > u[:, :6].sum()                              # gradient concentrated at the edge


# ======================================================================================================
# The two necessary conditions, the control every claim owes, and the trap.
# ======================================================================================================
def _aa_setup(W=64, H=64):
    """A hard-edged disk: uncertainty is maximally concentrated, and the refinement is priced PER PIXEL."""
    def shade(y, x):
        u = (x - W / 2 + 0.5) / (W / 2)
        v = (y - H / 2 + 0.5) / (H / 2)
        return (np.sqrt(u * u + v * v) < 0.62).astype(float)

    def render(spp, active=None, seed=0):
        ys, xs = np.mgrid[0:H, 0:W]
        out = np.zeros((H, W))
        rng = np.random.default_rng(seed)
        sel = np.ones((H, W), bool) if active is None else active
        n = int(sel.sum())
        acc = np.zeros(n)
        for _ in range(spp):
            acc += shade(ys[sel] + rng.random(n), xs[sel] + rng.random(n))
        out[sel] = acc / spp
        return out, n * spp                                   # (image, SAMPLES TAKEN -- the real cost)
    return render


def test_coarse_first_pays_on_per_pixel_work_and_the_signal_is_what_pays():
    """The positive: 6.2x fewer samples for a modest RMSE cost. And the CONTROL every coarse-first claim owes --
    the same budget spent at RANDOM cells -- must be far worse, or the 'uncertainty signal' carried no information."""
    from holographic.misc.holographic_coarsefirst import (concentration, gradient_uncertainty,
                                                          refine_where_uncertain)
    render = _aa_setup()
    ref, _ = render(256, seed=1)
    rmse = lambda a: float(np.sqrt(np.mean((a - ref) ** 2)))

    coarse, c_cost = render(1)
    u = gradient_uncertainty(coarse)
    assert concentration(u) > 0.5                              # the gate says GO (measured 1.000)

    full, f_cost = render(16)
    cost = [0]

    def refine_fn(mask):
        img, cost[0] = render(16, active=mask, seed=2)
        return img

    out, mask, n = refine_where_uncertain(coarse, u, refine_fn, frac=0.10)
    total = c_cost + cost[0]
    assert f_cost / total > 4.0, (f_cost, total)               # measured 6.2x cheaper than uniform
    assert rmse(out) < 0.6 * rmse(coarse)                      # ...and far better than the coarse pass
    assert rmse(out) < 1.6 * rmse(full)                        # for a bounded RMSE cost

    # THE CONTROL: identical budget, random cells. If this ties, the signal was noise.
    rng = np.random.default_rng(0)
    rand = np.zeros(coarse.size, bool)
    rand[rng.choice(coarse.size, n, replace=False)] = True
    rand = rand.reshape(coarse.shape)
    img, _ = render(16, active=rand, seed=2)
    rand_out = coarse.copy()
    rand_out[rand] = img[rand]
    assert rmse(out) < 0.6 * rmse(rand_out), (rmse(out), rmse(rand_out))


def test_a_greedy_coarse_pass_destroys_the_concentration_its_refinement_needs():
    """THE TRAP, measured. Greedy matching pursuit spends its atoms exactly where the error was, flattening the very
    signal the escalator reads. Coarse-first wants a cheap, uniform, DUMB base pass."""
    from holographic.misc.holographic_coarsefirst import concentration
    from holographic.rendering.holographic_splat import _gaussian, splat_fit, splat_refit, splat_render
    n = 48
    ys, xs = np.mgrid[0:n, 0:n]
    band = (np.abs(xs - n // 2) < 10) & (np.abs(ys - n // 2) < 10)
    T = 0.6 * np.exp(-((ys - n / 2) ** 2 + (xs - n / 2) ** 2) / (2 * 16.0 ** 2))
    T[band] += 0.7 * (np.sin(2 * np.pi * 6 * xs / n) > 0)[band]

    greedy = splat_render(splat_refit(splat_fit(T, 36), T), T.shape)      # adaptive placement
    uniform = splat_render(splat_refit([(cy, cx, 0.0, 3.5) for cy in range(4, n, 8) for cx in range(4, n, 8)], T),
                           T.shape)                                        # a dumb grid, same count
    c_greedy = concentration(np.abs(T - greedy))
    c_uniform = concentration(np.abs(T - uniform))
    assert c_uniform > 2.0 * c_greedy, (c_uniform, c_greedy)   # measured 0.416 vs 0.106
    assert c_greedy < 0.3, c_greedy                            # ...and the gate correctly rules coarse-first OUT


def test_concentration_is_a_gate_and_ties_are_escalated_conservatively():
    from holographic.misc.holographic_coarsefirst import concentration, escalate_mask
    assert concentration(np.ones((16, 16))) < 1e-9             # uniform uncertainty: coarse-first cannot help
    assert concentration(np.zeros((16, 16))) == 0.0            # degenerate, no division by zero
    spike = np.zeros((16, 16)); spike[0, 0] = 1.0
    assert concentration(spike) > 0.9                          # all the work in one cell

    u = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert escalate_mask(u, frac=0.25).sum() == 1 and escalate_mask(u, frac=0.25)[1, 1]
    assert escalate_mask(u, frac=0.0).sum() == 0 and escalate_mask(u, frac=1.0).all()
    assert escalate_mask(np.zeros((4, 4)), threshold=0.0).all()   # ties INCLUDED: refine rather than skip
    assert escalate_mask(np.zeros((4, 4)), frac=0.5).sum() == 8   # a constant field still yields exactly k cells


def test_coarse_first_through_the_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    f = np.zeros((32, 32))
    f[10:22, 10:22] = 1.0
    u = m.gradient_uncertainty(f)
    assert m.uncertainty_concentration(u) > 0.5
    out, mask, n = m.refine_where_uncertain(f, u, lambda mk: np.ones_like(f), frac=0.20)
    assert n == int(mask.sum()) and n > 0
    assert np.array_equal(out[~mask], f[~mask])                 # the coarse result survives outside the mask
    assert np.array_equal(m.escalate_mask(u, frac=0.20), mask)
    assert any("coarse-first" in c.name.lower() for c in m.find_capability("is adaptive refinement worth it"))
