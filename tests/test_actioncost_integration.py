"""The action layer through the mind: the cost wall (G1), the emission-vs-actionable fill discipline (G2), and
their composition with the conditioning layer -- the storm shape (state-dependent costs) read honestly."""
import numpy as np

import lecore


def test_cost_wall_kills_and_spares_correctly():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    edge = np.random.default_rng(0).normal(10.0, 20.0, size=400)
    dead, alive = mind.net_of_costs(edge, cost=17.0), mind.net_of_costs(edge, cost=5.0)
    assert not dead["survives"] and alive["survives"]
    assert dead["wall_ratio"] < 1 < alive["wall_ratio"]
    # the portable fact travels: breakeven equals the gross mean by construction.
    assert abs(dead["breakeven_cost"] - float(edge.mean())) < 1e-9


def test_latency_artifact_vs_genuinely_predictive_event():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(1)
    n = 12000
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.9 * ar[i - 1] + rng.normal()
    rise = ar[5:] - ar[:-5]
    ev = (np.nonzero(rise > 2.0)[0] + 5)
    ev = ev[(ev > 10) & (ev < n - 30)][::3]
    trap = mind.realizable_fills(ev, ar, horizon=10, lag=1, emission_price=ar[ev - 5])
    assert trap["idealized_mean"] > 0.5 and trap["actionable_mean"] < 0.0
    assert trap["verdict"].startswith("LATENCY ARTIFACT")
    # a drift that BEGINS at the event survives the reachable entry.
    drift = np.cumsum(rng.normal(size=n) * 0.1)
    starts = np.arange(50, n - 200, 400)
    for s0 in starts:
        drift[s0:s0 + 30] += np.linspace(0, 3.0, 30)
    real = mind.realizable_fills(starts, drift, horizon=20, lag=1)
    assert real["verdict"].startswith("ACTIONABLE") and real["actionable_t"] > 5.0


def test_lag_sweep_is_the_infrastructure_budget():
    """The kept negative as behaviour: an edge that decays fast after the event weakens monotonically as the
    action lag grows -- lag=1 is a floor, and the sweep is the honest readout."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)
    n = 12000
    drift = np.cumsum(rng.normal(size=n) * 0.05)
    starts = np.arange(50, n - 100, 300)
    for s0 in starts:
        drift[s0:s0 + 8] += np.linspace(0, 1.5, 8)     # the move is over in 8 samples
    means = [mind.realizable_fills(starts, drift, horizon=8, lag=lag)["actionable_mean"]
             for lag in (1, 3, 6)]
    assert means[0] > means[1] > means[2], means


def test_state_dependent_costs_with_the_conditioning_layer():
    """G1 x C: same average cost, but concentrated in the storm state -- the composed readout the storm canon
    demands. The conditional split shows where the net edge actually lives."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(3)
    n = 600
    storm = np.arange(n) % 3 == 0                       # a third of events fire in storms
    gross = np.where(storm, rng.normal(36.0, 30.0, n), rng.normal(4.0, 10.0, n))
    costs = np.where(storm, 15.0, 3.0)                  # spreads widen exactly when the signal pays
    r = mind.net_of_costs(gross, cost=costs)
    net = gross - costs
    split = mind.conditional(net, [bool(b) for b in storm])
    # the effect survives its own storm costs, and the split shows the premium is inside the storm state.
    assert r["survives"]
    assert split["mean_inside"] > split["mean_outside"] and split["separates"]
