"""holographic_analog.py -- ANALOG FORECASTING: "find the past that looks like now, and return what followed."
The most VSA-native forecaster in the engine -- it is pure content-addressable recall, and it yields a
DISTRIBUTION over outcomes natively (not just a point), which is exactly what the conformal layer wants to
calibrate.

WHY THIS EXISTS (Forecasting & Prediction backlog, F4)
------------------------------------------------------
The four other producers LEARN an operator (Propagator, reservoir) or resonate context (predictive). Analog
forecasting learns nothing: encode the recent history as one vector, recall its k nearest neighbours in the
store, and return the empirical distribution of THEIR successors. That is `HoloForest` recall (sublinear,
already shipped) pointed at time -- Pharr's BVH-for-meaning doing Lorenz's 1969 analog method. Two properties
make it the right partner for the conformal work: it produces a SET of outcomes (perfect for CRPS / an interval),
and it ABSTAINS honestly -- if nothing in the past resembles now (no near neighbour), there is no analog and the
correct answer is "I don't know," not a confident guess.

KEPT NEGATIVE (loud): analog forecasting works ONLY where the present resembles stored past. A novel regime has
no analog -- the ~0%-overlap wall again -- and the honest output there is abstention. It is pattern retrieval
with a confidence, never prophecy.

Real basis: Lorenz (1969), analog forecasting; Zhao & Giannakis (2016), kernel analog forecasting. Seat: Pharr
(sublinear content-addressable recall) + Olshausen (distributed recall). Deterministic; NumPy + stdlib.
"""
import numpy as np

from holographic_tree import HoloForest


def delay_embed(series, d):
    """Turn a 1-D time series into (context, successor) pairs the classic analog way: each context is the last
    `d` values, its successor is the next value. Returns (contexts (m,d), successors (m,)). This is the delay
    embedding Lorenz used -- a window of recent history is the 'situation' we look for an analog of."""
    s = np.asarray(series, float)
    n = len(s)
    contexts = np.stack([s[i:i + d] for i in range(n - d)])
    successors = s[d:]
    return contexts, successors


class AnalogForecaster:
    """Store (context vector -> successor) pairs; to forecast, recall the k nearest contexts and return the
    empirical distribution of their successors. `successors` may be scalars OR vectors. Sublinear recall via
    HoloForest. Abstains when the nearest analog is too weak to trust (`sim_floor`)."""

    def __init__(self, sim_floor=0.5, n_trees=4, leaf_size=64, seed=0):
        self.sim_floor = float(sim_floor)                       # below this best-cosine, there is no real analog
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.seed = seed
        self.forest = None
        self.successors = None

    def fit(self, contexts, successors):
        """Index the context vectors for sublinear recall and remember each one's successor."""
        contexts = np.asarray(contexts, float)
        self.successors = np.asarray(successors, float)
        self.forest = HoloForest(contexts.shape[1], n_trees=self.n_trees, leaf_size=self.leaf_size, seed=self.seed)
        self.forest.build(contexts)
        return self

    def forecast(self, context, k=8, beam=4):
        """Recall the k nearest stored contexts to `context` and return a dict:
          point     -- the similarity-weighted mean successor (the point forecast),
          samples   -- the k neighbours' successors (the empirical predictive distribution, for CRPS/intervals),
          weights   -- their similarities (higher = more relevant analog),
          confidence-- the best neighbour's cosine (how good the closest analog is),
          abstain   -- True when even the best analog is below sim_floor (no analog exists -> don't forecast).
        """
        if self.forest is None:
            raise RuntimeError("call fit() first")
        idx, sims = self.forest.recall_k(np.asarray(context, float), k=k, beam=beam)
        if len(idx) == 0:
            return {"point": None, "samples": np.array([]), "weights": np.array([]),
                    "confidence": 0.0, "abstain": True}
        succ = self.successors[idx]
        best = float(sims[0])
        # similarity-weighted mean successor; only positive similarities vote (a dissimilar analog shouldn't pull)
        w = np.clip(sims, 0.0, None)
        wsum = float(w.sum()) + 1e-12
        if succ.ndim == 1:
            point = float((w * succ).sum() / wsum)
        else:
            point = (w[:, None] * succ).sum(axis=0) / wsum
        return {"point": point, "samples": succ, "weights": sims,
                "confidence": best, "abstain": best < self.sim_floor}


def _selftest():
    """On a quasi-periodic series (which HAS analogs), analog forecasting beats persistence and the mean at the
    one-step horizon; it yields a successor distribution; and it ABSTAINS on a query unlike anything stored."""
    rng = np.random.default_rng(0)

    # a fast quasi-periodic signal: consecutive samples differ a lot (so persistence is a FAIR baseline, not a
    # strawman), but the pattern recurs (so real analogs exist). This is the regime analog forecasting is for.
    t = np.arange(4000)
    series = np.sin(t * 0.55) + 0.5 * np.sin(t * 0.27) + 0.03 * rng.standard_normal(len(t))
    d = 20
    contexts, successors = delay_embed(series, d)
    n_train = 3000
    af = AnalogForecaster(sim_floor=0.5, seed=0).fit(contexts[:n_train], successors[:n_train])

    # forecast each held-out point; compare to persistence (last value) and the training mean
    errs_analog, errs_persist = [], []
    train_mean = float(successors[:n_train].mean())
    errs_mean = []
    for i in range(n_train, len(contexts)):
        f = af.forecast(contexts[i], k=8)
        if f["abstain"]:
            continue
        truth = successors[i]
        errs_analog.append(abs(f["point"] - truth))
        errs_persist.append(abs(contexts[i][-1] - truth))       # persistence = repeat the last observed value
        errs_mean.append(abs(train_mean - truth))
    mae_analog = float(np.mean(errs_analog))
    mae_persist = float(np.mean(errs_persist))
    mae_mean = float(np.mean(errs_mean))
    assert mae_analog < mae_persist, (mae_analog, mae_persist)   # beats persistence
    assert mae_analog < mae_mean, (mae_analog, mae_mean)         # beats the mean (the honest baselines)

    # it yields a DISTRIBUTION (a set of successors), not just a point
    f = af.forecast(contexts[n_train], k=8)
    assert len(f["samples"]) > 1 and f["confidence"] > 0.5

    # a no-analog query has markedly LOWER confidence than a real one. (Kept negative: turning confidence into a
    # hard abstain needs a floor tuned to the CONTEXT DIMENSION -- in a low-dim raw window a random vector can hit
    # a moderate max-cosine by chance, so reliable hard abstention wants either a high floor or hypervector-encoded
    # contexts; the confidence ORDERING, however, is robust.)
    real_conf = af.forecast(contexts[n_train], k=8)["confidence"]
    alien = rng.standard_normal(d) * 10.0
    alien_conf = af.forecast(alien, k=8)["confidence"]
    assert alien_conf < real_conf, (alien_conf, real_conf)

    # with a floor set above the alien's chance level, the hard abstain fires on the no-analog query
    af_strict = AnalogForecaster(sim_floor=0.95, seed=0).fit(contexts[:n_train], successors[:n_train])
    assert af_strict.forecast(alien, k=8)["abstain"] is True

    # deterministic
    a = af.forecast(contexts[n_train], k=8)["point"]
    b = af.forecast(contexts[n_train], k=8)["point"]
    assert a == b

    print("holographic_analog selftest OK: analog MAE %.4f beats persistence %.4f and mean %.4f; yields a "
          "successor distribution (confidence %.2f); abstains on a no-analog query; deterministic"
          % (mae_analog, mae_persist, mae_mean, f["confidence"]))


if __name__ == "__main__":
    _selftest()
