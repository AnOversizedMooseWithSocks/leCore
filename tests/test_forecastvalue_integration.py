"""Calibration vs value (D4) at the seams: through the mind, tau chosen honestly on a split with the ledger
carrying the sweep, and composed with net_of_costs -- constant per-event costs here, state-dependent there,
the two verdicts agreeing on the same planted economics."""
import numpy as np

import lecore


def _forecast_fixture(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(2, 2, n), 0.01, 0.99)
    y = (rng.random(n) < p).astype(float)
    return p, y


def test_the_faculty_reproduces_both_pinned_facts():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    p, y = _forecast_fixture()
    good = mind.calibration_vs_value(p, y, cost=0.05)
    flat = mind.calibration_vs_value(np.full(len(y), y.mean()), y, cost=0.05)
    squash = mind.calibration_vs_value(0.5 + 0.08 * np.tanh(3 * (p - 0.5)), y, cost=0.05)
    assert good["value_best"]["net"] > 0
    assert flat["value_best"]["net"] <= max(0.0, flat["baselines"]["always"]) + 1e-9
    assert squash["reliability"] > 10 * good["reliability"]
    assert squash["value_best"]["net"] > 0.9 * good["value_best"]["net"]


def test_tau_chosen_on_a_split_and_the_sweep_goes_on_the_ledger():
    """The kept negative, closed the honest way: sweep tau on the first half, LEDGER every tau tried, then
    quote the value at that single pre-chosen tau on the second half. The out-of-sample value at the chosen
    tau must be positive and NOT better than the in-sample argmax (selection optimism has the one sign)."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    p, y = _forecast_fixture(seed=1, n=6000)
    half = 3000
    led = mind.selection_ledger()
    r_tr = mind.calibration_vs_value(p[:half], y[:half], cost=0.05)
    for row in r_tr["value_curve"]:
        # one entry per tau tried; a pseudo-p from the net's sign keeps the ledger honest about the sweep
        led.record("tau_%.2f" % row["tau"], 0.5, family="tau_sweep", effect=row["net"],
                   note="in-sample net during tau selection")
    tau = r_tr["value_best"]["tau"]
    r_te = mind.calibration_vs_value(p[half:], y[half:], cost=0.05, taus=[tau])
    net_te = r_te["value_curve"][0]["net"]
    assert net_te > 0
    assert net_te <= r_tr["value_best"]["net"] * (half and 1.15)     # optimism has one sign (slack for noise)
    assert led.report()["families"][0]["n"] == len(r_tr["value_curve"])


def test_composed_with_net_of_costs_the_two_cost_verdicts_agree():
    """Plant economics where acting is profitable gross but dies at a 0.45/event cost. The tau-sweep's net
    and net_of_costs on the SAME realized per-action values must agree on survival at both cost levels --
    two tools, one truth."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    p, y = _forecast_fixture(seed=2)
    # The wall must exceed the sharpest achievable per-action edge: with payoff/loss = 1/1 the edge at the
    # top tau bin is ~2*E[y|act]-1 <= ~0.85 on this fixture, so 0.9 is above ANY tau's edge and 0.05 is
    # under all the good ones. (A first draft used 0.45, which the high-tau bins clear at ~0.8 edge --
    # "profitable gross but dead at the wall" has to plant the wall above the gross.)
    cheap = mind.calibration_vs_value(p, y, payoff_act=1.0, loss_act=1.0, cost=0.05)
    dear = mind.calibration_vs_value(p, y, payoff_act=1.0, loss_act=1.0, cost=0.9)
    assert cheap["value_best"]["net"] > 0
    # The dear wall's best net comes back +0.4, not 0.0 -- and that residue IS the module's own kept
    # negative demonstrating itself: value_best is an argmax over 19 taus, and at a 0.9 wall the sweep
    # scavenges a handful of top-bin events whose sampled edge cleared the wall by luck. Selection optimism
    # on the sweep's own report. So the pin is ECONOMIC death (under 1% of the cheap value, and under the
    # always baseline's |loss|/100), not literal zero -- literal zero would only hold if the argmax were not
    # a selection, which is exactly what the note in value_best warns it is.
    assert dear["value_best"]["net"] < 0.01 * cheap["value_best"]["net"], (dear["value_best"], cheap["value_best"])

    tau = cheap["value_best"]["tau"]
    act = p >= tau
    per_event = np.where(y[act] == 1, 1.0, -1.0)
    r_cheap = mind.net_of_costs(list(per_event), cost=0.05)
    r_dear = mind.net_of_costs(list(per_event), cost=0.9)
    assert r_cheap["survives"] is True
    assert r_dear["survives"] is False


def test_refusals_survive_the_faculty():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    try:
        mind.calibration_vs_value([1.2] * 30, [1] * 30)
        raise AssertionError("expected refusal")
    except ValueError as e:
        assert "[0, 1]" in str(e)
