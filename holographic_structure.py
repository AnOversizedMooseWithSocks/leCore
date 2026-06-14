"""Proof of meaning: verify that a sequence carries structure, rather than trust
that a fluent-looking prediction does.

A predictor that lands each step in the right neighbourhood can still emit
locally-plausible, globally-meaningless text -- word salad. So a prediction needs
a VERIFIER, and the verification cannot come from any single word's meaning in
isolation; it has to come from projecting each word onto its CONTEXT, across
ranges. This is the same principle as a running integrity check: structure shows
up as a measurable signature of the whole sequence, not of its parts.

WHAT WAS MEASURED, AND WHY A NAIVE CHECK FAILS (kept honestly):
  * SINGLE-STEP coherence -- cosine(what the context predicts, the actual word's
    meaning) -- catches shuffled text (0.21) and random words (0.08) against real
    text (0.41). But it is GAMEABLE: text generated greedily by the predictor
    itself scores 0.88, HIGHER than real text, because each step was chosen to
    maximise exactly that quantity. Local coherence is not proof of meaning.
  * The fix is the LAG-COHERENCE PROFILE: the mean similarity between each word
    and the word k positions back, for k = 1..6. Real text has a moderate, EVEN
    profile across ranges. Salad deviates in one of two ways: shuffled/random sit
    too LOW (no structure at any range); degenerate generation sits too HIGH and
    PERIODIC (an order-2 generator that fell into a 3-cycle reads ~1.0 at lag 3
    and 6). Genuine structure is neither -- it is the characteristic band of real
    text.
  * STRUCTURE SCORE = -mean z-distance of a sequence's lag-profile from a
    reference band calibrated on real text (0 = perfectly typical, more negative
    = more anomalous). Measured: real held-out text ~ -1.2 (near its own
    calibration ~ -1.4); shuffled ~ -2.2; random ~ -4.1; self-generated salad
    ~ -15. Every salad type -- including the locally-coherent self-generated one
    that single-step coherence missed -- falls well below real text. A threshold
    from the real-text score distribution separates meaning from salad.

So 'proof of meaning or structure' is the lag-profile match, and 'meaning
projected upon context' is literally what each term of the profile measures. A
meaning vector alone is inert; its usefulness is whether it sits in the structural
band that real context produces.

THE VERIFIER USED AS A PROCESS (the active payoff): steered_generate picks, among
the predictor's top candidates at each step, the word that keeps the recent
window's structure score highest -- generation that defends its own coherence.
Measured against plain greedy generation, this is a large, real win and NOT a
repeat of the topic-pull negative: greedy decoding collapses into a fixed loop
('the state of the state of the state...', structure score ~ -11.5), while steered
generation escapes the loop and stays in the real-text band (~ -0.6). The
difference from topic-pull is the lever: topic-pull re-ranked candidates by a
static topic bag and collapsed; steering by TRAJECTORY structure -- projecting each
candidate onto the unfolding context -- defends coherence as a process.

AN HONEST LIMIT, kept: the verdict cleanly rejects random words (0% pass) and
degenerate self-generated loops (the failure modes that matter for generation),
and passes real text (100%). It does NOT reliably reject SHUFFLED real text (~90%
still passes): a bag of real words keeps too much of the lag-profile even when
reordered. Catching local reordering would need a genuinely order-sensitive check
(a running composition, the Closure approach); the profile measures structure-by-
range, not exact order. For its purpose -- keeping generation out of salad and
loops -- it works; as a general grammaticality judge it is partial.

Needs: numpy, holographic_ai, a meaning space (word -> vector) and its index.
"""
import numpy as np

from holographic_ai import cosine


class StructureVerifier:
    """Calibrate the lag-coherence profile of real text, then score any sequence
    by how closely its profile matches -- the proof that it carries structure."""

    def __init__(self, vocab, M, idx, lags=(1, 2, 3, 4, 5, 6)):
        self.vocab = list(vocab)
        self.M = np.asarray(M, float)
        self.idx = dict(idx)
        self.lags = tuple(lags)
        self.ref_mean = None
        self.ref_std = None
        self.threshold = None

    def profile(self, tokens):
        """The lag-coherence profile: mean cosine between each word and the word k
        positions back, for each k in lags. The fingerprint of structure."""
        toks = list(tokens)
        out = []
        for k in self.lags:
            vals = [float(self.M[self.idx[toks[i]]] @ self.M[self.idx[toks[i - k]]])
                    for i in range(k, len(toks))
                    if toks[i] in self.idx and toks[i - k] in self.idx]
            out.append(float(np.mean(vals)) if vals else 0.0)
        return np.array(out)

    def calibrate(self, real_tokens, chunk=200, z_floor=3.0):
        """Learn the reference band (mean, std per lag) from real text, and set a
        verdict threshold from real chunks' own score distribution (mean - z_floor
        * std of those scores). Returns self."""
        toks = list(real_tokens)
        chunks = [toks[i:i + chunk] for i in range(0, max(1, len(toks) - chunk), chunk)]
        P = np.stack([self.profile(c) for c in chunks]) if chunks else np.zeros((1, len(self.lags)))
        self.ref_mean = P.mean(0)
        self.ref_std = P.std(0) + 1e-6
        scores = np.array([self._raw_score(c) for c in chunks])
        # threshold: real chunks rarely fall this low; salad reliably does
        self.threshold = float(scores.mean() - z_floor * (scores.std() + 1e-6))
        return self

    def _raw_score(self, tokens):
        p = self.profile(tokens)
        z = np.abs((p - self.ref_mean) / self.ref_std)
        return float(-z.mean())

    def structure_score(self, tokens):
        """How well a sequence's lag-profile matches real text. 0 = perfectly
        typical; more negative = more anomalous (toward salad). Requires calibrate."""
        if self.ref_mean is None:
            raise RuntimeError("call calibrate(real_tokens) first")
        return self._raw_score(tokens)

    def is_meaningful(self, tokens):
        """Verdict: does this sequence carry structure (score at or above the
        calibrated threshold)? The proof of meaning, projected onto context."""
        return self.structure_score(tokens) >= self.threshold

    def coherence_trace(self, tokens, predictor):
        """Per-position single-step coherence (cosine of the predictor's composed
        next-meaning to the actual word). Useful for inspection, but NOT proof on
        its own -- it is gameable by a generator (see module docstring)."""
        toks = list(tokens)
        out = []
        for i in range(predictor.order, len(toks)):
            if toks[i] not in self.idx:
                continue
            _, vec, _ = predictor.predict_meaning(toks[i - predictor.order:i])
            out.append(cosine(vec, self.M[self.idx[toks[i]]]))
        return np.array(out) if out else np.array([0.0])


def steered_generate(predictor, verifier, seed, length=30, beam=5, lookback=8):
    """Generate while PROJECTING meaning onto the running context: at each step,
    take the predictor's top candidates and keep the one that keeps the recent
    window's structure score highest -- generation as a process that defends its
    own coherence. Falls back to the predictor's pick if no candidate helps.

    This is the active use of the verifier. Whether it beats plain greedy is an
    empirical question, measured by the caller; the steering is a process, not a
    guarantee."""
    out = list(seed)
    for _ in range(length):
        recent = out[-predictor.order:]
        Cn, nextM = predictor._matrix()
        if len(Cn) == 0:
            break
        q = predictor.context_vector(recent)
        qn = q / (np.linalg.norm(q) + 1e-12)
        coup = Cn @ qn
        order = np.argsort(coup)[::-1][:beam]
        cands = []
        seen = set()
        for j in order:
            w = predictor._next[j]
            if w not in seen:
                seen.add(w)
                cands.append(w)
        if not cands:
            break
        # pick the candidate that maximises the recent window's structure score
        best_w, best_s = cands[0], -1e9
        for w in cands:
            window = (out + [w])[-lookback:]
            try:
                s = verifier.structure_score(window)
            except Exception:
                s = 0.0
            if s > best_s:
                best_w, best_s = w, s
        out.append(best_w)
    return out[len(seed):]
