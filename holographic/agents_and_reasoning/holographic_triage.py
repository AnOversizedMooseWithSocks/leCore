"""Little HRNNs that amortize expensive work: the triage cascade (holographic_triage).

WHY THIS EXISTS -- the shortcut family, extended
------------------------------------------------
The engine already trades computation for small baked artifacts everywhere: chunk
codebooks stand in for re-derivation, baked kernels for re-evaluation, stored VM
programs for re-planning, the fat-margin cache for re-querying. This module adds the
HRNN-shaped member of that family: a TINY trained readout (a few floats of ridge over a
few cheap features) that fronts an EXPENSIVE predicate and answers the easy cases
immediately.

THE CONTRACT, which is the whole design: the little model may only FAST-REJECT.
It never fast-accepts. Certification always runs the full machinery. WHY the asymmetry:
a cheap surrogate that can say "yes" is a false-certification machine waiting for a
distribution shift; a cheap surrogate that can only say "no" fails, at worst, by making
you pay the price you were already paying. (Branch prediction has the same shape: a
mispredicted branch is corrected by the real execution, never trusted over it.)

CALIBRATION: the rejection threshold is set from the training POSITIVES -- the lowest
score any true-positive achieved, minus a safety margin in score-spread units -- so on
the training distribution the false-reject rate is zero BY CONSTRUCTION, and the
held-out false-reject rate is measured and reported, never assumed. When the little
model is not confidently negative, it abstains and the full predicate runs: abstention
as control flow, used here for speed.

Ships with one measured instantiation: fast features for the compressibility gate
(entropy rate is already memoised; top-bin power fraction; derivative skewness;
lag-1 autocorrelation), because the full gate costs n_null+1 generator fits and most
real streams (every market series measured) are refusals -- the exact case a
fast-reject shortcut wins.

Stdlib + numpy only; deterministic given seeds; the trained cascade exports/imports
like the other model classes (save/load, npz, no pickle).
"""
import numpy as np

from holographic.sampling_and_signal.holographic_statedemand import (
    entropy_rate_report, quantize_stream)


def gate_features(x, k=4):
    """Cheap features that correlate with 'a generator exists': entropy rate (memoised,
    so repeated triage of the same bytes is near-free), top-bin power fraction (a tone
    concentrates the spectrum), absolute derivative skewness (phase-locked waveforms
    are asymmetric; Gaussianised imposters are not), and lag-1 autocorrelation."""
    x = np.asarray(x, dtype=float).ravel()
    rep = entropy_rate_report(quantize_stream(x, k), k)
    h = rep["h"] if rep["h"] is not None else np.log2(k)
    X = np.abs(np.fft.rfft(x - x.mean()))
    top = float(X.max() ** 2 / (np.sum(X ** 2) + 1e-300))
    d = np.diff(x)
    skew = float(abs(np.mean((d - d.mean()) ** 3)) / (np.std(d) ** 3 + 1e-12))
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0
    return np.array([h, top, skew, r1])


class TriageCascade:
    """Front an expensive boolean predicate with a fast-reject-only little model.

    feature_fn(x) -> small vector (cheap); full_fn(x) -> bool (expensive, the oracle).
    fit() runs the oracle once per training input, fits a ridge score, and calibrates
    the reject threshold below every training positive. __call__ fast-rejects when the
    score is confidently negative and otherwise pays for the oracle -- so the cascade's
    answer NEVER disagrees with the oracle on accepts, by construction."""

    def __init__(self, feature_fn, full_fn, safety=1.0, lam=1e-3):
        self.feature_fn, self.full_fn = feature_fn, full_fn
        self.safety, self.lam = float(safety), float(lam)
        self.w = None
        self.threshold = None
        self.stats = {}

    def fit(self, inputs):
        """Train on unlabeled inputs: the oracle provides labels (one full run each --
        the amortised cost), the ridge provides the score, the positives set the floor."""
        y = np.array([bool(self.full_fn(x)) for x in inputs], dtype=float)
        return self.fit_from_labels(inputs, y)

    def fit_from_labels(self, inputs, labels):
        """Train from PRECOMPUTED oracle labels. WHY this exists: tuning the safety knob
        re-ran the oracle per setting and timed out a 2000s budget doing it -- the
        labels never change across settings, so pay for them once and sweep free."""
        F = np.stack([self.feature_fn(x) for x in inputs])
        y = np.asarray(labels, dtype=float)
        mu, sd = F.mean(0), F.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)              # the constant-feature lesson
        A = np.hstack([(F - mu) / sd, np.ones((len(F), 1))])
        self.w = np.linalg.solve(A.T @ A + self.lam * np.eye(A.shape[1]), A.T @ (2 * y - 1))
        self.mu, self.sd = mu, sd
        scores = A @ self.w
        pos = scores[y > 0.5]
        if len(pos) == 0:
            # no positives seen: nothing is safely rejectable relative to a floor we
            # never observed, so the cascade degenerates to always-run-the-oracle.
            self.threshold = -np.inf
        else:
            spread = float(scores.std()) or 1.0
            self.threshold = float(pos.min()) - self.safety * spread
        self.stats = {"n_train": len(y), "positives": int(y.sum()),
                      "threshold": self.threshold}
        return self

    def score(self, x):
        f = (self.feature_fn(x) - self.mu) / self.sd
        return float(np.concatenate([f, [1.0]]) @ self.w)

    def __call__(self, x):
        """Returns dict{accept, path, score}: path 'fast-reject' or 'full'."""
        s = self.score(x)
        if s < self.threshold:
            return {"accept": False, "path": "fast-reject", "score": s}
        return {"accept": bool(self.full_fn(x)), "path": "full", "score": s}

    def save(self, path):
        """Export the trained cascade head (the feature/oracle functions are code, not
        state -- rebind them at load time via load(path, feature_fn, full_fn))."""
        np.savez_compressed(path, w=self.w, mu=self.mu, sd=self.sd,
                            threshold=self.threshold, safety=self.safety, lam=self.lam)
        return path

    @classmethod
    def load(cls, path, feature_fn, full_fn):
        z = np.load(path if str(path).endswith(".npz") else str(path) + ".npz",
                    allow_pickle=False)
        out = cls(feature_fn, full_fn, safety=float(z["safety"]), lam=float(z["lam"]))
        out.w, out.mu, out.sd = z["w"], z["mu"], z["sd"]
        out.threshold = float(z["threshold"])
        return out


def sweep_safety(inputs, labels, feature_fn=None, full_fn=None,
                 safeties=(1.0, 0.5, 0.25, 0.1, 0.05, 0.02), holdout=0.4, seed=0):
    """Choose the safety margin from MEASUREMENT instead of timidity. The shipped
    default (1.0) was measured too conservative on the real battery -- 9% fast at a
    perfect contract, i.e. paying for honesty we already had in surplus. This sweep
    fits ONE ridge on a train split (labels are precomputed, so all settings are
    free), then for each safety checks the HELD-OUT split: pick the largest fast-
    reject rate whose held-out false-reject count is ZERO. The contract is never
    traded: a setting with even one held-out FR is disqualified regardless of speed,
    and the recommendation ships as DATA (a report + a fitted cascade) -- the class
    default stays 1.0, because existing defaults never flip silently.
    Returns {table, chosen_safety, cascade, why}."""
    import numpy as _np
    feature_fn = feature_fn or gate_features
    rng = _np.random.default_rng(seed)
    idx = rng.permutation(len(inputs))
    n_hold = max(2, int(holdout * len(inputs)))
    hold, train = [int(i) for i in idx[:n_hold]], [int(i) for i in idx[n_hold:]]
    tr_in = [inputs[i] for i in train]
    tr_y = [labels[i] for i in train]
    table, chosen, best = [], None, None
    for sf in sorted(safeties, reverse=True):
        c = TriageCascade(feature_fn, full_fn or (lambda x: True), safety=sf)
        c.fit_from_labels(tr_in, tr_y)
        fr = fast = 0
        for i in hold:
            rejected = c.score(inputs[i]) < c.threshold
            fast += int(rejected)
            fr += int(rejected and labels[i])          # rejected a true positive
        row = {"safety": sf, "fast_rate": fast / len(hold), "false_rejects": fr}
        table.append(row)
        if fr == 0 and (best is None or row["fast_rate"] > best["fast_rate"]):
            best, chosen = row, c
    why = ("safety %.2f: held-out fast-reject %.0f%% at 0 false rejects"
           % (best["safety"], 100 * best["fast_rate"])) if best else           "no setting achieved 0 held-out false rejects: keep safety 1.0"
    return {"table": table, "chosen_safety": best["safety"] if best else 1.0,
            "cascade": chosen, "why": why}


def _selftest():
    """Asserts the CONTRACT: fast path never accepts; zero false-rejects on held-out
    positives from the calibrated classes; a real fraction of negatives short-circuits;
    accept decisions identical to the oracle's."""
    rng = np.random.default_rng(0)
    T = 600
    t = np.arange(T, dtype=float)

    # a deliberately expensive-ish oracle: the phase-sensitive skew score against 20
    # phase-randomised surrogates (the statedemand selftest's calibrated stand-in).
    def phase_rand(v, r):
        X = np.fft.rfft(v)
        ph = r.uniform(0, 2 * np.pi, len(X)); ph[0] = 0.0
        if len(v) % 2 == 0:
            ph[-1] = 0.0
        return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(v))

    def skew(v):
        d = np.diff(v)
        return float(abs(np.mean((d - d.mean()) ** 3)) / (np.std(d) ** 3 + 1e-12))

    calls = {"n": 0}

    def oracle(v):
        calls["n"] += 1
        obs = skew(v)
        null = [skew(phase_rand(v, np.random.default_rng(i))) for i in range(20)]
        return (1 + sum(1 for u in null if u >= obs)) / 21.0 <= 0.05

    def make(kind, seed):
        r = np.random.default_rng(seed)
        if kind == "saw":
            return ((t % (90 + 20 * (seed % 3))) / 90.0) * 2 - 1 + 0.05 * r.standard_normal(T)
        if kind == "white":
            return r.standard_normal(T)
        if kind == "walk":
            return np.cumsum(r.standard_normal(T))
        return phase_rand(((t % 110) / 110.0) * 2 - 1, r)          # spectrum imposter

    kinds = ["saw", "white", "walk", "imposter"]
    train = [make(kinds[i % 4], i) for i in range(32)]
    test = [make(kinds[i % 4], 100 + i) for i in range(32)]

    casc = TriageCascade(gate_features, oracle).fit(train)
    calls["n"] = 0
    results = [casc(x) for x in test]
    oracle_truth = []
    for x in test:                                     # ground truth for agreement
        oracle_truth.append(oracle(x))

    fast = [r for r in results if r["path"] == "fast-reject"]
    assert all(not r["accept"] for r in fast), "CONTRACT BROKEN: a fast path accepted"
    false_rejects = sum(1 for r, o in zip(results, oracle_truth)
                        if r["path"] == "fast-reject" and o)
    assert false_rejects == 0, "%d held-out positives fast-rejected" % false_rejects
    agree = sum(1 for r, o in zip(results, oracle_truth) if r["accept"] == o)
    assert agree == len(test), "cascade disagreed with oracle on %d" % (len(test) - agree)
    frac_fast = len(fast) / len(test)
    assert frac_fast >= 0.25, "cascade short-circuited only %.2f" % frac_fast

    # export round-trip: identical decisions
    casc.save("/tmp/triage_gate.npz")
    back = TriageCascade.load("/tmp/triage_gate.npz", gate_features, oracle)
    assert all(back(x)["accept"] == r["accept"] for x, r in zip(test[:8], results[:8]))

    # sweep: a separable battery must find a faster-than-1.0 setting at 0 held-out
    #     FR, and the contract clause must disqualify any setting with an FR.
    rng2 = np.random.default_rng(5)
    xs = [np.sin(2 * np.pi * np.arange(300.0) / p) + 0.2 * rng2.standard_normal(300)
          for p in (17, 29, 41, 53, 71, 89)] +          [rng2.standard_normal(300) for _ in range(18)]
    ys = [1] * 6 + [0] * 18
    sw = sweep_safety(xs, ys, full_fn=lambda x: True, seed=1)
    # the CONTRACT, not a conclusion: chosen setting has 0 held-out FR and is at
    # least as fast as safety 1.0 (on a cleanly separable battery 1.0 may already
    # win -- that outcome is correct, not a defect).
    assert "0 false rejects" in sw["why"], sw["why"]
    row_1 = next(r for r in sw["table"] if r["safety"] == 1.0)
    row_c = next(r for r in sw["table"] if r["safety"] == sw["chosen_safety"])
    assert row_c["false_rejects"] == 0 and row_c["fast_rate"] >= row_1["fast_rate"]
    #     HARD battery: faint positives near the negatives -- the 1.0 margin now
    #     costs real speed, and the sweep must find a faster 0-FR setting.
    xs2 = [0.5 * np.sin(2 * np.pi * np.arange(300.0) / p) + rng2.standard_normal(300)
           for p in (17, 23, 29, 37, 41, 47, 53, 61)] +           [rng2.standard_normal(300) for _ in range(24)]
    ys2 = [1] * 8 + [0] * 24
    sw2 = sweep_safety(xs2, ys2, full_fn=lambda x: True, seed=2)
    r1 = next(r for r in sw2["table"] if r["safety"] == 1.0)
    rc = next(r for r in sw2["table"] if r["safety"] == sw2["chosen_safety"])
    assert rc["false_rejects"] == 0 and rc["fast_rate"] > r1["fast_rate"], sw2["table"]

    print("holographic_triage selftest OK -- fast-rejected %.0f%% of held-out streams, "
          "0 false rejects, accept decisions == oracle, round-trip identical"
          % (100 * frac_fast))


if __name__ == "__main__":
    _selftest()
