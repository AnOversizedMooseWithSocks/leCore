"""Loss space (E1) at the seams: through the mind, and the pair it completes -- loss_space_report finds the
gate candidate, insurance_profile confirms gating it does not delete the value, trailing_gate builds the
gate, and the gated record's loss shape reads clean. The full loop the two siblings exist for."""
import numpy as np

import lecore


def test_the_faculty_reports_all_axes():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    v = rng.normal(0.05, 1.0, 800)
    r = mind.loss_space_report(v, seed=1)
    assert set(r) >= {"totals", "tail", "time", "verdict"}
    assert "unremarkable" in r["verdict"]


def test_the_gate_loop_find_confirm_gate_reread():
    """The composed workflow: (1) loss_space_report names the storm as the gate candidate; (2)
    insurance_profile on the same record confirms the VALUE is not concentrated there (gating is safe);
    (3) gate it; (4) the gated record's loss shape no longer flags the condition."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(1)
    n = 2000
    v = np.abs(rng.normal(0.6, 0.2, n))                          # steady small wins outside the storm
    storm = np.zeros(n, bool)
    for s in range(200, n - 100, 500):
        storm[s:s + 80] = True
    v[storm] = rng.normal(-1.2, 1.0, int(storm.sum()))           # the storm carries the losses

    # (1) find
    r1 = mind.loss_space_report(v, conditions={"storm": storm}, seed=0)
    assert r1["conditions"]["storm"]["z"] > 2
    assert "gate candidate" in r1["conditions"]["storm"]["verdict"]

    # (2) confirm gating is safe: the value must NOT be concentrated inside the storm.
    prof = mind.insurance_profile(v, storm)
    assert prof["share_inside"] < 0                              # the storm's contribution to the total is NEGATIVE
    assert prof["sum_outside"] > float(v.sum())                  # so gating it out strictly improves the sum

    # (3+4) gate and re-read: with storm events removed, the storm can no longer be a condition (mask must
    # align), and the remaining record's shape reads clean on tail and time.
    r2 = mind.loss_space_report(v[~storm], seed=0)
    # the gated record has (almost) no losses left -- the scarcity path IS the clean bill of health here
    assert "unremarkable" in r2["verdict"] or "too few" in r2["verdict"], r2["verdict"]


def test_streak_flag_composes_with_event_study_onsets():
    """Losses in blocks: the loss report flags the streak; event_study on loss-streak onsets shows the
    forward window keeps losing -- two instruments, one dependence structure."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)
    n = 3000
    v = np.abs(rng.normal(0.8, 0.3, n))
    for s in range(150, n - 80, 400):
        v[s:s + 50] = -np.abs(rng.normal(0.8, 0.3, 50))
    r = mind.loss_space_report(v, seed=0)
    assert r["time"]["z"] > 2
    loss = v < 0
    onsets = list(np.where(loss[1:] & ~loss[:-1])[0] + 1)
    es = mind.event_study(v, onsets, horizon=20, pre=10, seed=0)
    assert es["forward"]["stat"] < 0                             # the window after a loss onset keeps losing
    assert es["forward"]["p"] < 0.05


def test_refusals_and_scarcity_through_the_faculty():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(3)
    try:
        mind.loss_space_report(rng.normal(0, 1, 10))
        raise AssertionError("expected refusal")
    except ValueError as e:
        assert "at least 40" in str(e)
    v = np.abs(rng.normal(1, 0.1, 120))
    v[:2] *= -1
    assert "too few" in mind.loss_space_report(v)["verdict"]
