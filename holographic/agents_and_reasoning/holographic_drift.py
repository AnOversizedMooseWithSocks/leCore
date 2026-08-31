"""DRIFT SENTINEL -- leOS's displacement-drift detector, rebuilt on lever 7's floor.

Extracted from leOS lvm/drift_detector.py + kernel_crawl_drift.py (the cp54 dig). leOS
watches every task -> response displacement and classifies it against the NEIGHBORHOOD of
similar past tasks -- pure unit-sphere geometry, no model, no phrase blacklists. leCore
already stores exactly the required experience: lever 7 IS a displacement log. So the
sentinel arrives with its experience base pre-populated wherever lever 7 has been living.

FOUR VERDICTS, each a measured geometric condition (thresholds are leOS's, tuned on
nomic-embed 768d; they transfer to any unit-normalized key because they are expressed in
NEIGHBORHOOD SIGMA, not absolute distance):

  void       fewer than 3 similar past tasks -- unexplored territory. Honestly flagged as
             low confidence rather than guessed at (the same doctrine as structured_voids).
  echo       response ~= task (cos >= 0.92): a non-answer that restates the question.
             leOS's note: this single check REPLACED a maintained phrase blacklist.
  redshift   displacement > 1.8 sigma ABOVE what similar tasks produced -- off in the
             weeds relative to established behaviour. For teaches, a redshift against the
             neighborhood of an already-taught similar question is a CONFLICT CANDIDATE:
             the new answer lands far from where this question's answers have always
             landed. That is the implicit-conflict signal the STALE benchmark grades,
             surfaced as a flag for the caller to resolve -- never an automatic overwrite.
  loop       the last 4 responses are pairwise-similar >= 0.85: the agent is circling.

WHAT THIS DOES NOT DO: it never blocks. Every verdict is advisory with the evidence
attached (magnitude, expected, deviation, confidence, neighbor count). Vetoes belong to
the caller; the sentinel's job is to make drift VISIBLE the moment it happens.
"""
import numpy as np

REDSHIFT_SIGMA = 1.8
ECHO_SIMILARITY = 0.92
LOOP_THRESHOLD = 0.85
LOOP_WINDOW = 4
VOID_MIN_NEIGHBORS = 3
NEIGHBOR_RADIUS = 0.5
NEIGHBORHOOD_K = 10


def _unit(v):
    v = np.asarray(v, np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class DriftSentinel:
    """Classify task->response displacements against the neighborhood of similar tasks."""

    def __init__(self, redshift_sigma=REDSHIFT_SIGMA, echo=ECHO_SIMILARITY,
                 loop=LOOP_THRESHOLD, loop_window=LOOP_WINDOW,
                 min_neighbors=VOID_MIN_NEIGHBORS, radius=NEIGHBOR_RADIUS,
                 k=NEIGHBORHOOD_K):
        self.redshift_sigma = float(redshift_sigma)
        self.echo = float(echo)
        self.loop = float(loop)
        self.loop_window = int(loop_window)
        self.min_neighbors = int(min_neighbors)
        self.radius = float(radius)
        self.k = int(k)
        self._log = []                       # [(task_unit, magnitude)]
        self._recent = []                    # response units, loop detection

    def note(self, task_vec, response_vec):
        """Log an observed displacement -- the experience base grows with use."""
        t, r = _unit(task_vec), _unit(response_vec)
        self._log.append((t, float(np.linalg.norm(r - t))))
        return len(self._log)

    def classify(self, task_vec, response_vec, remember=True):
        t, r = _unit(task_vec), _unit(response_vec)
        magnitude = float(np.linalg.norm(r - t))
        sim_tr = float(t @ r)

        sims = np.array([t @ lt for lt, _m in self._log]) if self._log else np.array([])
        idx = [i for i in np.argsort(-sims)[:self.k] if sims[i] >= self.radius] \
            if sims.size else []
        mags = [self._log[i][1] for i in idx]
        expected = float(np.mean(mags)) if mags else 0.0
        std = float(np.std(mags)) if len(mags) > 1 else 0.1
        confidence = (min(1.0, len(idx) / self.k) *
                      float(np.mean([sims[i] for i in idx]))) if idx else 0.0
        deviation = magnitude - expected

        verdict, warnings = "normal", []
        if len(idx) < self.min_neighbors:
            verdict = "void"
            warnings.append("only %d similar past tasks (need %d) -- prediction "
                            "confidence is low; normal for new task types"
                            % (len(idx), self.min_neighbors))
        elif sim_tr >= self.echo:
            verdict = "echo"
            warnings.append("response restates the task (cos %.3f >= %.2f) -- likely a "
                            "non-answer" % (sim_tr, self.echo))
        elif deviation > self.redshift_sigma * std:
            verdict = "redshift"
            warnings.append("displacement %.3f is %.1f sigma above the neighborhood's "
                            "%.3f -- off established behaviour for this task region"
                            % (magnitude, deviation / std, expected))
        elif expected - magnitude > 1.5 * std and len(idx) >= self.min_neighbors:
            verdict = "blueshift"
            warnings.append("displacement %.3f is well below the neighborhood's %.3f -- "
                            "suspiciously little work for this task region"
                            % (magnitude, expected))

        self._recent.append(r)
        self._recent = self._recent[-self.loop_window:]
        looped = False
        if len(self._recent) == self.loop_window:
            pair = [float(a @ b) for i, a in enumerate(self._recent)
                    for b in self._recent[i + 1:]]
            if pair and min(pair) >= self.loop:
                looped = True
                warnings.append("last %d responses are pairwise-similar >= %.2f -- "
                                "the loop is circling" % (self.loop_window, self.loop))
        if remember:
            self._log.append((t, magnitude))
        return {"verdict": verdict, "loop": looped, "magnitude": round(magnitude, 4),
                "expected": round(expected, 4), "deviation": round(deviation, 4),
                "confidence": round(confidence, 4), "neighbors": len(idx),
                "task_response_similarity": round(sim_tr, 4), "warnings": warnings}


def _selftest():
    rng = np.random.default_rng(0)
    d = 64
    s = DriftSentinel()
    # an established region: similar tasks, consistent displacement
    base = _unit(rng.standard_normal(d))
    for _ in range(12):
        t = _unit(base + 0.05 * rng.standard_normal(d))
        r = _unit(t + 0.4 * rng.standard_normal(d))         # consistent work size
        s.note(t, r)
    t = _unit(base + 0.05 * rng.standard_normal(d))

    normal = s.classify(t, _unit(t + 0.4 * rng.standard_normal(d)), remember=False)
    assert normal["verdict"] == "normal" and normal["neighbors"] >= 3, normal

    echo = s.classify(t, _unit(t + 0.01 * rng.standard_normal(d)), remember=False)
    assert echo["verdict"] == "echo", echo

    red = s.classify(t, _unit(rng.standard_normal(d)), remember=False)
    assert red["verdict"] == "redshift" and red["deviation"] > 0, red

    fresh = DriftSentinel()
    void = fresh.classify(_unit(rng.standard_normal(d)),
                          _unit(rng.standard_normal(d)), remember=False)
    assert void["verdict"] == "void" and void["confidence"] == 0.0, void

    lp = DriftSentinel()
    stuck = _unit(rng.standard_normal(d))
    out = None
    for _ in range(4):
        out = lp.classify(_unit(rng.standard_normal(d)), stuck)
    assert out["loop"], out

    return ("OK: DriftSentinel pins passed (normal in an established region; echo on a "
            "restated task; redshift %.1f sigma off the neighborhood; void honestly "
            "low-confidence; a circling loop detected in %d responses)"
            % (red["deviation"] / 0.1 if red["deviation"] else 0, LOOP_WINDOW))


if __name__ == "__main__":
    print(_selftest())
