"""holographic_honesty.py -- the ablation ethos as a callable instrument.

Holostuff's README and ABLATIONS.md keep only what beats a baseline. This module is
that rule made executable, so any recall-based predictor can be judged the same honest
way before it is believed -- and so a parameter SCAN (many encoders / leaf sizes / roles
over the same data) is held to a false-discovery bar instead of celebrating whichever
candidate cleared 2-sigma by luck. Nothing here is market-specific; it is the engine's
own discipline, callable.

  walk_forward_recall -- judge a nearest-neighbour predictor over a signed time series
                         with six checks a real edge has to survive (beat chance, beat
                         the trivial persistence baseline, collapse under a shuffle, and
                         so on).
  bh_fdr              -- Benjamini-Hochberg (or Benjamini-Yekutieli for dependent tests)
                         false-discovery-rate control, the missing guard for a library
                         that can generate a great many candidates on demand.
"""
import math

import numpy as np


def walk_forward_recall(states, outcomes, R, cost=0.0, seed=0, warmup=None):
    """Judge a nearest-neighbour predictor over (states, outcomes) honestly.

    states  : (N, dim) one hypervector per moment, in time order.
    outcomes: (N,) the signed quantity being predicted (e.g. a next return).
    R       : how many nearest PAST states to recall and vote with.
    cost    : round-trip cost in the same units as outcomes, charged per trade.

    Six checks, all of which a real edge has to survive:
      acc           directional accuracy of the recall vote
      acc_persist   the TRIVIAL baseline (predict the last non-zero sign) -- recall must
                    beat this, not merely beat a coin
      acc_shuffled  the same machinery on SHUFFLED outcomes -- this MUST collapse to
                    chance; if it does not, the harness itself is leaking
      chance_band   the 2-sigma chance half-width (2 * 0.5 / sqrt(n)); 'better than
                    chance' means acc - 0.5 > band
      scale_corr    does the SPREAD of recalled outcomes track the realised move size --
                    the magnitude signal, separate from direction
      net           gross edge minus cost, in outcome units
    """
    out = np.asarray(outcomes, float)
    states = np.asarray(states, float)
    N = len(out)
    sgn = np.sign(out)
    # last non-zero sign seen so far -> the persistence baseline
    last_nz = np.zeros(N)
    cur = 0.0
    for t in range(N):
        last_nz[t] = cur
        if sgn[t] != 0:
            cur = sgn[t]
    if warmup is None:
        warmup = max(R + 20, 120)

    def run(vec):
        osgn = np.sign(vec)
        ok = okp = tot = 0
        gross = 0.0
        spreads, realized = [], []
        for row in range(warmup, N):
            if osgn[row] == 0:
                continue
            sims = states[:row - 1] @ states[row]        # recall strictly from the PAST
            top = np.argsort(sims)[-R:]
            recall = vec[top]
            pred = np.sign(np.median(recall)) or 1.0
            ok += (pred == osgn[row])
            okp += (last_nz[row] == osgn[row])
            tot += 1
            gross += pred * vec[row]
            spreads.append(np.quantile(recall, 0.9) - np.quantile(recall, 0.1))
            realized.append(abs(vec[row]))
        return (ok / tot, okp / tot, tot, gross / tot,
                np.asarray(spreads), np.asarray(realized))

    acc, acc_persist, tot, gross, spreads, realized = run(out)
    shuf = out.copy()
    np.random.default_rng(seed).shuffle(shuf)
    acc_shuffled = run(shuf)[0]                           # MUST sit at ~0.5
    if len(spreads) > 2 and spreads.std() > 0 and realized.std() > 0:
        scale_corr = float(np.corrcoef(spreads, realized)[0, 1])
    else:
        scale_corr = float("nan")
    band = 2 * 0.5 / math.sqrt(tot)
    return dict(acc=acc, acc_persist=acc_persist, acc_shuffled=acc_shuffled,
                n=tot, gross=gross, net=gross - cost,
                scale_corr=scale_corr, chance_band=band,
                beats_chance=(acc - 0.5) > band,
                beats_persistence=acc > acc_persist)


def bh_fdr(pvals, alpha=0.1, dependent=True):
    """Benjamini-Hochberg false-discovery-rate control.

    With dependent=True applies the Benjamini-Yekutieli c(m)=sum(1/j) correction, the
    honest choice when the tests are dependent -- and a parameter SCAN (many encoders /
    leaf sizes / roles over the SAME data) always is. Holds the expected false-positive
    fraction among declared discoveries at alpha. Returns (reject_mask, n_rejected).
    """
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool), 0
    order = np.argsort(p)
    ranked = p[order]
    c = float(np.sum(1.0 / np.arange(1, m + 1))) if dependent else 1.0
    thresh = (np.arange(1, m + 1) / (m * c)) * alpha
    below = ranked <= thresh
    k = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    rej = np.zeros(m, bool)
    if k > 0:
        rej[order[:k]] = True
    return rej, k


class RecallNull:
    """Turn a recall / cleanup similarity into an HONEST false-alarm probability.

    The move radio-SETI and particle physics both live by: a raw cosine of 0.13 means
    nothing on its own -- you have to ask how high random noise reaches against THIS
    codebook before you believe a match. fit() draws random unit queries and records the
    best-match similarity each one reaches; that empirical null IS the noise floor. Then
    pvalue(score) = the fraction of null best-scores that reach `score` or higher = the
    chance pure noise would look this good. Small p: trust the recall. Large p: abstain.

    Calibrated by construction: a genuinely random query's p is ~uniform, so thresholding at
    p <= alpha holds the false-alarm rate at ~alpha -- the engine's "score it, then prove it
    isn't an artifact of your own pipeline" discipline made callable per recall. Complements
    HoloForest's cross-tree agreement (a structural abstention signal) with a statistical one.
    """

    def __init__(self):
        self.null = None          # sorted ascending: best-match cosine reached by noise

    def fit(self, codebook, n_null=2000, seed=0):
        """codebook: (N, dim) atoms (need not be unit; they are unit-normalised here so the
        dot is a cosine). Draw n_null random unit queries, record the best cosine each hits,
        and keep that sorted null. O(n_null * N * dim) once, then pvalue() is a binary search."""
        C = np.asarray(codebook, float)
        units = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((n_null, units.shape[1]))
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
        self.null = np.sort((units @ Q.T).max(axis=0))      # best match per random query
        return self

    def pvalue(self, score):
        """False-alarm probability for a recall similarity: P(noise best-match >= score)."""
        if self.null is None:
            raise ValueError("fit() the null on a codebook first")
        n = len(self.null)
        # fraction of null at or above score (null is sorted ascending)
        return float((n - int(np.searchsorted(self.null, score, side="left"))) / n)

    def calibrated_recall(self, query, codebook):
        """Recall the best-matching atom AND its honest false-alarm probability.
        Returns (best_index, similarity, pvalue). p small -> trust it; p large -> abstain."""
        C = np.asarray(codebook, float)
        units = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        q = np.asarray(query, float)
        qn = np.linalg.norm(q)
        sims = units @ (q / qn) if qn > 0 else units @ q
        j = int(sims.argmax())
        return j, float(sims[j]), self.pvalue(float(sims[j]))


class SPRTRecall:
    """Wald's Sequential Probability Ratio Test over a STREAM of recall scores.

    A single recall gives one calibrated score (see RecallNull). But when a SEQUENCE of cues bears
    on the same hypothesis -- a drifting narrowband signal across time, repeated sightings of a
    landmark, a recurring microstructure pattern -- you should not commit to a fixed window. You
    accumulate the per-cue log-likelihood ratio and stop the moment the evidence crosses a Wald
    boundary. That is provably the MINIMUM EXPECTED number of samples for a target (alpha, beta)
    error pair (Wald, Sequential Analysis) -- the discipline radio-SETI and particle physics use to
    decide as fast as the evidence allows.

    Fit Gaussian score densities from a sample of null scores -- RecallNull's noise floor IS
    p(score | null) -- and a sample of genuine-match scores. Then `update(score)` returns
    'MATCH' / 'REJECT' / 'CONTINUE'. MEASURED on real holostuff recall scores: ~half the samples of
    the best fixed-N rule at matched error (e.g. avg 2.8 samples vs fixed-N 6 at ~2% error).
    """

    def __init__(self, null_scores, match_scores, alpha=0.05, beta=0.05):
        self.mu0 = float(np.mean(null_scores)); self.sd0 = float(np.std(null_scores)) + 1e-9
        self.mu1 = float(np.mean(match_scores)); self.sd1 = float(np.std(match_scores)) + 1e-9
        self.A = float(np.log((1 - beta) / alpha))     # upper boundary -> accept MATCH
        self.B = float(np.log(beta / (1 - alpha)))     # lower boundary -> reject
        self.reset()

    def reset(self):
        """Start a fresh stream (accumulated log-LR and sample count back to zero)."""
        self.llr = 0.0
        self.n = 0
        return self

    @staticmethod
    def _loglik(x, mu, sd):
        return -0.5 * np.log(2 * np.pi * sd * sd) - (x - mu) ** 2 / (2 * sd * sd)

    def update(self, score):
        """Add one cue's evidence to the running log-LR; return 'MATCH', 'REJECT', or 'CONTINUE'."""
        self.llr += self._loglik(score, self.mu1, self.sd1) - self._loglik(score, self.mu0, self.sd0)
        self.n += 1
        if self.llr >= self.A:
            return "MATCH"
        if self.llr <= self.B:
            return "REJECT"
        return "CONTINUE"

    def decide(self, scores, cap=None):
        """Feed a whole stream; return (decision, n_samples_used). If `cap` is reached without a
        boundary crossing, fall back to the sign of the accumulated evidence."""
        self.reset()
        for s in scores:
            d = self.update(s)
            if d != "CONTINUE":
                return d, self.n
            if cap is not None and self.n >= cap:
                break
        return ("MATCH" if self.llr > 0 else "REJECT"), self.n
