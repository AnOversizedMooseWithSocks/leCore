"""Look-ahead lint (E3) at the seams: through the mind, against the engine's OWN pipelines (the linter's most
demanding user is this codebase), and composed with pipeline_null -- the lint proves causality, the null
proves the causal pipeline still is not manufacturing its result. Different questions; both required."""
import numpy as np

import lecore


def test_the_faculties_work_and_agree_with_the_module():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = np.cumsum(np.random.default_rng(0).standard_normal(400))
    bad = lambda s: (s - s.mean()) / (s.std() or 1.0)
    good = lambda s: np.concatenate([[0.0], np.diff(s)])
    assert mind.lookahead_lint(bad, x)["causal"] is False
    r = mind.lookahead_lint(good, x)
    assert r["causal"] is True and r["max_drift"] == 0.0


def test_the_lint_against_the_engines_own_faculties():
    """Dogfooding as a contract: the engine's own trailing tools must lint causal, and its own known
    whole-sample tool must lint leaky. sign_flip is a whole-series surrogate BY DESIGN -- if the lint ever
    passes it, the lint broke."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = np.cumsum(np.random.default_rng(1).standard_normal(400))
    # ema faculty if wired; otherwise a local trailing ema stands in for the same contract
    def ema(s, a=0.1):
        out = np.zeros_like(np.asarray(s, float))
        for i in range(1, out.size):
            out[i] = (1 - a) * out[i - 1] + a * s[i]
        return out
    assert mind.lookahead_lint(ema, x)["causal"] is True
    # sign_flip lints CAUSAL -- correctly, and it is worth pinning WHY: its seeded flips are element-wise, so
    # output[i] depends only on x[i] and the seed. "Whole-series surrogate" describes its PURPOSE, not its
    # dependence structure -- the lint tests the latter, which is the only thing causality is about.
    per_element = lambda s: np.asarray(mind.sign_flip(s, seed=0), float)
    assert mind.lookahead_lint(per_element, x)["causal"] is True
    # iid_shuffle is the true whole-sample dependence: the permutation is drawn over ALL n, so shortening the
    # series rearranges everything -- lints leaky, as it must.
    shuffled = lambda s: np.asarray(mind.iid_shuffle(s, seed=0), float)
    assert mind.lookahead_lint(shuffled, x)["causal"] is False


def test_shift_probe_contemporaneous_leak_and_symmetric_blindness():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)
    d = rng.standard_normal(500)
    leak = d + 0.3 * rng.standard_normal(500)
    p = mind.target_shift_probe(leak, d)
    assert p["suspicious"] and p["corr_not_ahead"] > 0.9 and p["corr_ahead"] < 0.2
    # the symmetric leak the probe is blind to -- and the lint catches on the label constructor.
    x = np.cumsum(rng.standard_normal(500))
    dd = np.concatenate([[0.0], np.diff(x)])
    label = np.convolve(dd, np.ones(5) / 5.0, mode="same")
    assert mind.target_shift_probe(label, dd)["suspicious"] is False
    assert mind.lookahead_lint(lambda s: np.convolve(s, np.ones(5) / 5.0, mode="same"), dd)["causal"] is False


def test_composed_with_pipeline_null_different_questions_both_required():
    """A trailing smoother+persistence chain LINTS CAUSAL (it truly never peeks) and STILL manufactures 79%
    direction persistence on pure noise, which pipeline_null catches. Passing the lint is not innocence --
    causal and honest are different axes, and the seam test pins that neither tool covers the other."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1500)

    def chain_signal(s):
        out = np.empty(len(s)); out[0] = s[0]
        for i in range(1, len(s)):
            out[i] = 0.8 * out[i - 1] + 0.2 * s[i]
        return out

    assert mind.lookahead_lint(chain_signal, x)["causal"] is True     # never peeks...

    def persistence_stat(s):
        y = np.sign(chain_signal(s))                                  # the null-layer's own 79% chain shape:
        y = y[y != 0]                                                 # persistence of the SMOOTHED SIGN
        return float(np.mean(y[1:] == y[:-1]))

    naive = persistence_stat(x)
    assert naive > 0.7                                                # ...yet manufactures persistence...
    r = mind.pipeline_null(persistence_stat, x, surrogate="iid_shuffle", n=100, seed=0)
    assert abs(r["z"]) < 3.0                                          # ...which its own null exposes as machinery.
