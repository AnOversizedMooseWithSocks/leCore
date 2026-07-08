"""Physics backlog #5: the AdaptiveSolver -- plan_waves dispatches the ocean stack like plan_render."""
import numpy as np
from holographic.simulation_and_physics.holographic_waveadaptive import plan_waves, plan_cost, method_counts, all_one_method_cost, solve_waves, METHOD_COST


def _scene():
    rng = np.random.default_rng(0); H = W = 64
    height = 0.05 * rng.standard_normal((H, W))
    height[8:16, 8:16] += np.linspace(0, 6, 8)[:, None]        # a breaking crest
    depth = np.full((H, W), 10.0); depth[:, :12] = 1.0         # a shallow shore strip
    obstacles = [(48, 48, 56, 56)]
    return height, depth, obstacles


def test_plan_picks_every_regime_with_reasons():
    height, depth, obstacles = _scene()
    plan = plan_waves(height, depth=depth, obstacles=obstacles, tile=8)
    counts = method_counts(plan)
    assert set(counts) == {"free_surface", "shallow_water", "wave_packets", "fft_ocean"}
    assert counts["fft_ocean"] == max(counts.values())        # cheap almost everywhere
    assert plan["tiles"][(8, 8)] == "free_surface" and "breaking" in plan["reasons"][(8, 8)]


def test_tie_break_deterministic_breaking_wins():
    height, depth, obstacles = _scene()
    h2 = height.copy(); h2[:8, :8] += np.linspace(0, 6, 8)[:, None]   # steep AND shallow
    p2 = plan_waves(h2, depth=depth, obstacles=obstacles, tile=8)
    assert p2["tiles"][(0, 0)] == "free_surface"              # breaking wins the tie
    a = plan_waves(height, depth=depth, obstacles=obstacles, tile=8)["tiles"]
    b = plan_waves(height, depth=depth, obstacles=obstacles, tile=8)["tiles"]
    assert a == b                                            # deterministic


def test_efficiency_win_measured():
    height, depth, obstacles = _scene()
    plan = plan_waves(height, depth=depth, obstacles=obstacles, tile=8)
    assert plan_cost(plan) < 0.25 * all_one_method_cost(plan, "free_surface")


def test_breaking_everywhere_no_discount():
    W = 64
    steep = np.tile(np.arange(W, dtype=float)[None, :], (W, 1))
    p = plan_waves(steep, tile=8)
    assert plan_cost(p) == all_one_method_cost(p, "free_surface")


def test_solve_dispatches_and_blends():
    height, depth, obstacles = _scene()
    plan = plan_waves(height, depth=depth, obstacles=obstacles, tile=8)
    tags = {"fft_ocean": lambda r, dt: r + 1.0, "wave_packets": lambda r, dt: r + 2.0,
            "shallow_water": lambda r, dt: r + 3.0, "free_surface": lambda r, dt: r + 4.0}
    out = solve_waves(plan, np.zeros((64, 64)), methods=tags, halo=0)
    assert abs(out[32, 40] - 1.0) < 1e-6                     # an open-water tile got the fft stepper


def test_blend_has_no_hard_seam():
    H = W = 64
    smooth = np.outer(np.sin(np.linspace(0, 3, H)), np.sin(np.linspace(0, 3, W)))
    plan = plan_waves(np.zeros((H, W)), tile=8)
    blended = solve_waves(plan, smooth, methods={"fft_ocean": lambda r, dt: r}, halo=2)
    assert np.max(np.abs(np.diff(blended[:, 7:10], axis=1))) < 0.2
