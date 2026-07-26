"""Group J reflexive compositions (J5-J9), each demonstrated at a seam: the backlog's claim is that the
SAME shipped analysis faculties serve the system's own internal decisions. Each test composes only shipped
tools -- no new machinery -- and pins the composition working on a planted system-shaped fixture. J4/J10/
J11/J12 shipped earlier (SelectionLedger, CausalIndex, program_vector, dpi_guard + the extend-don't-
duplicate rule); J1 is route_or_abstain; J2/J3 are dispositioned in NOTES_concepts.md as regen_docs/CI
process items with their concrete next steps."""
import numpy as np

import lecore


def test_j5_calibrated_envelope_scheduling_frame_times():
    """J5: frame-time budgets from the envelope forecaster instead of means. Planted load regimes: calm
    frames ~8ms, storm frames ~8ms mean but 4x spread. The MEAN-based budget covers overall yet blows
    through in storms (the backlog's 'covers 89% overall, fails exactly in the wild windows'); the
    envelope's conditional check (D1's conditional_coverage on the same record) makes the failure visible
    per regime -- the renderer's scheduler and the market tool's stops, one implementation."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    n = 4000
    storm = np.zeros(n, bool)
    for s in range(400, n - 200, 1000):
        storm[s:s + 250] = True
    frame_ms = np.where(storm, 8.0 + 4.0 * np.abs(rng.standard_normal(n)),
                        8.0 + 1.0 * np.abs(rng.standard_normal(n)))

    # calibrate a single global 90% budget on the first half (the mean-era policy), test on the second --
    # conditional_coverage's own split-conformal shape, applied to frame times instead of forecast residuals.
    rep = mind.conditional_coverage(frame_ms[:2000], frame_ms[2000:], storm[2000:], alphas=(0.1,))[0]
    assert rep["empirical_all"] > 0.88                           # the marginal bound looks fine...
    assert rep["empirical_inside"] < 0.70, rep                   # ...while storms blow through it (0.64 measured)
    assert rep["empirical_outside"] > 0.95                       # and calm frames waste headroom -- both hidden by the average


def test_j6_lookahead_lint_as_the_cache_causality_check():
    """J6: the dependency-keyed cache's claim ('this artifact reads only trailing inputs') IS lint's
    contract. A cache-key function that folds in a FUTURE-dependent normalisation fails the lint; the
    honest trailing version passes at exactly 0.0 drift -- gates and caches share the primitive."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = np.cumsum(np.random.default_rng(1).standard_normal(300))
    leaky_artifact = lambda s: s / (np.abs(s).max() or 1.0)      # peak-normalised: depends on the whole run
    causal_artifact = lambda s: np.concatenate([[0.0], np.diff(s)])
    assert mind.lookahead_lint(leaky_artifact, x)["causal"] is False
    assert mind.lookahead_lint(causal_artifact, x)["max_drift"] == 0.0


def test_j7_insurance_profile_vetoes_frequency_based_pruning():
    """J7: a rarely-hit error-path 'module' whose value concentrates in load spikes. Frequency says prune
    (2% usage); the insurance profile says its contribution inside the spike condition is where the value
    lives -- the same inside/outside decomposition that stopped the reversion being filtered out."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)
    n = 3000
    spike = rng.random(n) < 0.02
    value = np.where(spike, 50.0 + 5 * rng.standard_normal(n), 0.0 + 0.1 * rng.standard_normal(n))
    prof = mind.insurance_profile(value, spike)
    assert prof["share_inside"] > 0.9                            # the premium lives in the excluded state
    assert prof["separates"]                                     # and the profile says so loudly


def test_j8_reclocked_telemetry_makes_quiet_cheap_and_busy_dense():
    """J8: sample-when-it-moves for logs. A progress trace that stalls then sprints, reclocked on its own
    cumulative change: events concentrate in the sprint (dense where it matters), the stall is a FEW long-
    duration events (the profiler's 'time per unit progress' view), and quiet periods cost almost nothing."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    progress = np.concatenate([np.linspace(0, 1, 50),            # sprint
                               np.full(400, 1.0),                # stall
                               np.linspace(1, 2, 50)])           # sprint
    ev = mind.reclock(progress, step=0.05)
    src = np.asarray(ev["source_index"], float)
    durs = np.asarray(ev["durations"], float) if "durations" in ev else np.diff(np.concatenate([[0], src]))
    in_stall = (src > 60) & (src < 440)
    assert np.sum(in_stall) <= 2                                 # the stall is at most a couple of events
    assert float(np.max(durs)) > 20 * float(np.median(durs))     # ...whose duration screams 'stall'


def test_j9_phase_encoded_routing_distinguishes_negation_and_order():
    """J9: bag-of-words cannot tell 'a minus b' from 'b minus a'. Bind argument slots to positions on the
    circle with the CircularEncoder and the two orders separate cleanly while the same order re-encoded
    matches itself -- the market tool's signed-delta encoder serving retrieval, one representation, two
    levels of the system."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    from holographic.agents_and_reasoning.holographic_ai import bind
    enc = mind.circular_encoder(1024, period=2 * np.pi, seed=0)
    rng = np.random.default_rng(3)
    a = rng.standard_normal(1024); a /= np.linalg.norm(a)
    b = rng.standard_normal(1024); b /= np.linalg.norm(b)
    slot1, slot2 = enc.encode(0.0), enc.encode(np.pi / 2)        # argument order as position on the circle
    ab = bind(a, slot1) + bind(b, slot2)
    ba = bind(b, slot1) + bind(a, slot2)
    ab2 = bind(a, slot1) + bind(b, slot2)
    same = float(ab @ ab2 / (np.linalg.norm(ab) * np.linalg.norm(ab2)))
    crossed = float(ab @ ba / (np.linalg.norm(ab) * np.linalg.norm(ba)))
    assert same > 0.999
    assert crossed < 0.5, crossed
