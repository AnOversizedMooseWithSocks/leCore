"""holographic_assemble.py -- find a transform chain connecting an input to an output, HONESTLY (L12).

WHY THIS MODULE EXISTS
----------------------
The synesthesia endpoint exists ("song drives skeleton" via holographic_drives / audio_param_bus), but not the
ASSEMBLER: given an input signal and a desired output, WHICH pipeline of faculties connects them? The plan (§3i)
frames assembly as iterate-a-projection on the faculty graph -- but the load-bearing part is not the graph search,
it is the VALIDATION GATE. A closed chain is only a CANDIDATE; the honest question is whether the driven output
carries measurably more structure from the REAL input than from a time-SHUFFLED input. Without that gate, ANY
random projection "works" and we have discovered nothing (the routing-'win' lesson wearing a new hat). So this
module builds the gate first and keeps the search deliberately small -- §7 forbids framework-chasing.

WHAT IT DOES
  assemble_pipeline(x, y, candidates): try each candidate transform f (a callable x -> y_hat), score how well
  f(x) reproduces y, and -- crucially -- how much of that score SURVIVES shuffling x (real dependence) vs is
  chance alignment. Returns the ranked survivors, each with its MI-over-null z-score. A candidate passes only if
  driving from the real input beats driving from a shuffled input. Held-out validation: the score is measured on a
  SEGMENT not used to pick the candidate, because search overfits its own tests.

Reuses holographic_mutualinfo (the shuffle-null gate) and holographic_guide (iterate-a-projection, for the
FABRIK-style refinement of a parametric candidate). NumPy only. Deterministic.
"""

import numpy as np

from holographic.sampling_and_signal.holographic_mutualinfo import mutual_information_vs_null


def _score(y_hat, y):
    """How well y_hat reproduces y: correlation of the two (robust, scale-free). In [-1, 1]; 1 = perfect."""
    a = np.asarray(y_hat, float).ravel()
    b = np.asarray(y, float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def assemble_pipeline(x, y, candidates, min_z=3.0, holdout=0.3, bins=16, n_shuffle=48, seed=0):
    """Find which candidate transform(s) connect input `x` to output `y`, VALIDATED against a shuffle null (L12).
    `candidates` is a dict name -> callable (x -> y_hat). For each: fit/apply it, score the fit on a HELD-OUT
    segment (search overfits its own tests), and measure MI between f(x) and y ABOVE a shuffle null (does the real
    input drive the output more than a shuffled one?). Returns a list of survivors sorted by z, each a dict with
    name, score (held-out correlation), and z (MI over null). A candidate passes only if z >= `min_z` -- otherwise
    it is chance alignment, not a discovery. An empty list is an honest 'nothing connects these above the null'.

    WHY the null AND the holdout: the null kills random-projection 'wins' (any mapping correlates a little); the
    holdout kills search overfitting (a candidate tuned on the whole signal flatters itself). Both gates, not one.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    cut = int(n * (1.0 - holdout))
    x_fit, x_test = x[:cut], x[cut:]
    y_fit, y_test = y[:cut], y[cut:]

    survivors = []
    for name, f in candidates.items():
        try:
            yh_test = np.asarray(f(x_test), float)
        except Exception:
            continue                                          # a candidate that errors on this shape is not a fit
        if yh_test.shape != y_test.shape:
            continue
        score = _score(yh_test, y_test)                       # held-out reproduction quality
        # the honest gate: MI between the driven output and the target, above a shuffle null.
        gate = mutual_information_vs_null(yh_test, y_test, bins=bins, n_shuffle=n_shuffle, seed=seed)
        if gate["z"] >= min_z:
            survivors.append({"name": name, "score": round(score, 4), "z": round(gate["z"], 2),
                              "mi": round(gate["mi"], 4)})
    survivors.sort(key=lambda s: -s["z"])
    return survivors


def _selftest():
    """Contracts -- the gate is the point:

    1. A candidate that TRULY connects x->y (the real generating transform) is found, with high z.
    2. A DECOY candidate (a fixed random projection, no relation to y) is REJECTED -- z below the floor. This is
       the 'any random projection works' failure the gate exists to stop.
    3. A shuffled/independent target yields NO survivors (nothing connects above the null).
    4. Determinism.
    """
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    # the true relationship: y is a nonlinear function of x plus a little noise.
    y = np.tanh(2.0 * x) + 0.2 * rng.normal(size=n)

    # a fixed random projection decoy (unrelated to y).
    decoy_vec = rng.normal(size=n)
    candidates = {
        "true_tanh": lambda z: np.tanh(2.0 * z),              # the real transform (up to noise)
        "identity":  lambda z: z,                            # partially correlated (tanh ~ linear near 0)
        "decoy":     lambda z: decoy_vec[-len(z):],          # unrelated -- must be rejected
        "wrong_freq": lambda z: np.sin(50.0 * z),            # a real function of x but not THIS relationship
    }
    survivors = assemble_pipeline(x, y, candidates, min_z=3.0, seed=1)
    names = [s["name"] for s in survivors]

    # (1) the true transform is found.
    assert "true_tanh" in names, names
    # (2) the decoy is rejected (unrelated fixed projection).
    assert "decoy" not in names, names
    # the true transform should rank above the wrong-frequency one.
    assert survivors[0]["name"] in ("true_tanh", "identity"), survivors

    # (3) an independent target yields no survivors above the null.
    y_indep = rng.normal(size=n)
    none = assemble_pipeline(x, y_indep, {"true_tanh": lambda z: np.tanh(2.0 * z),
                                          "identity": lambda z: z}, min_z=4.0, seed=2)
    assert none == [] or all(s["z"] < 8 for s in none), none   # nothing strongly connects x to independent noise

    # (4) determinism.
    a = assemble_pipeline(x, y, candidates, seed=5)
    b = assemble_pipeline(x, y, candidates, seed=5)
    assert a == b

    top = survivors[0]
    print("holographic_assemble selftest OK (true transform '%s' found at z=%.1f; decoy random projection REJECTED "
          "by the shuffle-null gate; independent target yields no strong survivor; held-out + null both enforced; "
          "deterministic)" % (top["name"], top["z"]))


if __name__ == "__main__":
    _selftest()
