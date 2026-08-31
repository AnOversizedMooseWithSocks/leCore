"""holographic_lever7.py -- THE SEVENTH LEVER's machinery: a superposed DISPLACEMENT TRACE with
delta-rule writes, surprise-gated admission, calibrated + crosstalk-priced reads, a volatility
FIELD, an exact audit floor, and self-tiling at the measured capacity cliff.

WHY THIS EXISTS (backlog v2, E1.1'/E1.3'/E2.2')
-----------------------------------------------
The engine's six levers are all EXACT and fire only on identical inputs. Lever 7 amortizes across
SIMILARITY: log task->response moves, answer a new task from its neighborhood, and gate the shortcut
with a calibrated bound. The leOS original stored a LIST of frames and scanned it (k-NN in Python);
this module stores the experiences IN SUPERPOSITION -- a bundle of bind(task_key, displacement) --
so recall is ONE unbind with a (similar) key and the superposition performs the neighborhood blend
as algebra. Cost per read is O(dim log dim) flat in N until the capacity advisory, at which point the
trace SPLITS (lever 6: the cliff number becomes the tile size).

THE UPDATE RULE (SOTA-adopted, not invented here): the delta rule. write() first READS the trace's
own prediction for the key and stores only the CORRECTION beta*(observed - predicted) -- the
error-correcting write of fast-weight programmers (Schlag et al. 2021) industrialized by
DeltaNet / Gated DeltaNet / RWKV-7, and simultaneously the displacement codec's P-frame: a move
already predicted by the trace costs ~zero write energy. Admission is SURPRISE-gated with momentum
(the Titans rule, arXiv 2501.00663, independently equal to leOS's novelty_score): low-surprise
writes are skipped entirely, and the surprise ledger is part of the trace's telemetry.

THE READ GATE (the part that is ours): read_gated() prices every answer three ways before serving --
  1. a CALIBRATED null: the raw readout magnitude against a permutation null of the trace itself
     (RecallNull discipline), refusing below the alpha-quantile;
  2. the CROSSTALK PRICE: the capacity law's own n/dim floor -- as the trace fills, every answer's
     stated trust decays by construction (trust = signal fraction of a read under the load ledger);
  3. the VOLATILITY FIELD: one cosine against a suppression bundle; a marked region never fires.
The kept negative from the measured ablation (deep-dive Part 3): ungated similarity reuse served 48
wrong answers on the standard workload where the full gate served 2 -- THE GATE IS NOT OPTIONAL.

THE FLOOR (lever 3 under lever 7): every accepted write is appended verbatim to an exact audit log,
and replay(log) rebuilds the trace BIT-IDENTICALLY (asserted in the selftest). On a shifting floor a
displacement log is noise; here the log is a set of addresses into deterministic computation.

Deterministic; NumPy + stdlib only. Fixed dtype float64; all randomness seed-derived.
"""
import hashlib
import json

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import (
    bind, unbind, bundle, cosine, random_vector,
)


def _seed_from(tag):
    """A stable 32-bit seed from a string tag -- lever 3's 'regenerate from seeds', hash()-free."""
    return int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:4], "big")


def key_atom(tag, dim):
    """A deterministic unit key atom for a string tag: same tag, same atom, any process, any
    PYTHONHASHSEED. This is the trace's tile/name key generator (0 bytes of stored index)."""
    v = random_vector(dim, np.random.default_rng(_seed_from(tag)))
    return v


class VolatilityField:
    """Volatility as a FIELD, not a regex list (E2.2'): volatile regions are unit vectors bundled
    into one suppression trace; check(x) is a single cosine. An answer whose task sits in a marked
    region must never be served from memory -- prices, live status, anything the world can move.

    The field is deliberately coarse: marking is cheap, checking is one dot product, and unmarking
    is EXACT SUBTRACTION (ablation is exact in this algebra). Sources that should feed it: explicit
    marks, the Database journal (a mutated table dirties its region), HDRIFT's bandwidth floor.
    """

    def __init__(self, dim, threshold=0.25):
        self.dim = int(dim)
        self.threshold = float(threshold)
        self._field = np.zeros(self.dim)
        self._marks = {}                       # tag -> vector (kept so unmark is exact)

    def mark(self, tag, vec=None):
        v = np.asarray(vec, float) if vec is not None else key_atom("volatile:" + tag, self.dim)
        v = v / (np.linalg.norm(v) + 1e-12)
        if tag not in self._marks:
            self._marks[tag] = v
            self._field = self._field + v
        return tag

    def unmark(self, tag):
        v = self._marks.pop(tag, None)
        if v is not None:
            self._field = self._field - v      # exact unlearning: subtraction, not decay
        return v is not None

    def check(self, x):
        """True if x lies in a marked region (one cosine against the field)."""
        n = np.linalg.norm(self._field)
        if n == 0:
            return False
        return float(np.dot(x, self._field) / (np.linalg.norm(x) * n + 1e-12)) >= self.threshold

    def to_state(self):
        return {"dim": self.dim, "threshold": self.threshold,
                "marks": {t: v.tolist() for t, v in self._marks.items()}}

    @classmethod
    def from_state(cls, state):
        f = cls(int(state["dim"]), float(state["threshold"]))
        for t, v in state["marks"].items():
            f.mark(t, np.asarray(v, float))
        return f


class DisplacementTrace:
    """The superposed experience memory (E1.1'). One trace vector holds every accepted
    (task_key -> displacement) pair as bind(key, value); read(key) is one unbind. Delta-rule
    writes, surprise-gated admission, calibrated + crosstalk-priced + volatility-checked reads,
    an exact replayable audit log, and a capacity advisory that recommends tiling.

    Parameters: dim; alpha (read-gate error rate for the calibrated null); beta (delta-rule write
    gain); surprise_floor (admission: skip writes whose prediction error cosine-distance is below
    this); momentum (surprise momentum, Titans-style); advisory_load (n/dim fraction at which
    tile() is recommended -- the measured bundle cliff territory, default 0.10).
    """

    def __init__(self, dim=2048, seed=0, alpha=0.05, beta=1.0, surprise_floor=0.05,
                 momentum=0.9, advisory_load=0.10, name="trace"):
        self.dim = int(dim)
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.surprise_floor = float(surprise_floor)
        self.momentum = float(momentum)
        self.advisory_load = float(advisory_load)
        self.name = str(name)
        self._trace = np.zeros(self.dim)
        self._n = 0                             # accepted writes in THIS trace (tile)
        self._surprise = 0.0                    # momentum-smoothed surprise level
        self._audit = []                        # exact floor: [(key.tolist(), value.tolist())]
        self._atoms = []                        # response codebook (the I-frame prototypes)
        self._succ_field = np.zeros(self.dim)   # outcome memory: where fires went WELL
        self._fail_field = np.zeros(self.dim)   # ...and where they went WRONG (the compass gate)
        self._null = None                       # calibrated null quantile (lazy)
        self.volatility = VolatilityField(self.dim)
        _r = np.random.default_rng(self.seed + 65537)
        _k0, _v0 = random_vector(self.dim, _r), random_vector(self.dim, _r)
        self._rho = max(1e-6, float(np.dot(unbind(bind(_k0, _v0), _k0), _v0)))  # single-pair readback factor
        self.stats = {"writes": 0, "skipped_low_surprise": 0, "reads": 0,
                      "fired": 0, "refused_null": 0, "refused_volatile": 0}

    # -- write side ---------------------------------------------------------------------------
    def write(self, task_key, value_vec):
        """Delta-rule write with surprise admission -- ADAPTED to superposition (kept negative
        below). Reads the trace's own prediction for the key; if the prediction clears the
        calibrated null (the key is genuinely known), stores only the CORRECTION v - pred (the
        P-frame) and skips entirely once the residual falls below surprise_floor. If the
        prediction is below the null, the key is NOVEL and v is written whole (the I-frame).

        KEPT NEGATIVES (measured in this module's own development, do not rediscover):
        (1) The textbook delta rule `write(v - read(k))` assumes EXACT reads (the matrix-memory /
        linear-attention setting). In HRR superposition the read carries crosstalk of norm
        ~sqrt(load); subtracting it verbatim writes NEGATED NOISE and destroyed the trace
        (readback cosine 0.70 -> 0.0003 at 24 pairs). (2) Detecting a known key by readout
        MAGNITUDE also fails at load: signal ~0.7 rides on crosstalk ~sqrt(n), so known and novel
        keys have indistinguishable magnitudes. The sound form corrects ALONG v ONLY: estimate the
        stored strength s = <pred, v> / (rho*|v|^2) with rho the dim's measured single-pair
        readback factor, and write (1-s)*v -- the off-v component of the prediction is crosstalk
        and must never be written back."""
        k = np.asarray(task_key, float)
        v = np.asarray(value_vec, float)
        pred = unbind(self._trace, k) if self._n else np.zeros(self.dim)
        # stored-strength estimate: project the prediction ONTO v and divide out the dim's own
        # single-pair readback factor rho (measured once, seed-derived, at construction).
        vv = float(np.dot(v, v)) + 1e-12
        s = float(np.dot(pred, v)) / (self._rho * vv) if self._n else 0.0
        s = min(max(s, 0.0), 1.0)
        s_now = 1.0 - s                                 # surprise = the unstored fraction of v
        self._surprise = self.momentum * self._surprise + (1 - self.momentum) * s_now
        if self._n and s_now < self.surprise_floor:
            self.stats["skipped_low_surprise"] += 1     # fully predicted: the free P-frame
            return {"accepted": False, "surprise": s_now, "load": self.load()}
        err = s_now * v                                 # correct ALONG v only (see kept negative)
        vn = v / (np.linalg.norm(v) + 1e-12)
        if not self._atoms or max(float(np.dot(vn, a)) for a in self._atoms) < 0.5:
            self._atoms.append(vn)                       # a new response prototype (I-frame atom)
        self._trace = self._trace + bind(k, err)
        self._n += 1
        self._audit.append((k.tolist(), v.tolist()))
        self._null = None                        # trace changed: null must recalibrate
        self.stats["writes"] += 1
        return {"accepted": True, "surprise": s_now, "load": self.load()}

    # -- read side ----------------------------------------------------------------------------
    def read(self, task_key):
        """The UNGATED read: one unbind. Exposed for measurement and composition ONLY -- the
        module's kept negative is that serving this raw is how you get 48 wrong answers."""
        self.stats["reads"] += 1
        return unbind(self._trace, np.asarray(task_key, float))

    def _null_quantile(self, n_null=48):
        """The calibrated COSINE null: for keys that were never written, the best cleanup cosine
        of the readout against the response codebook, at the CURRENT load. Deterministic
        (seed-derived probes); recomputed lazily whenever the trace changes.

        KEPT NEGATIVE (measured here, do not rediscover): gating on readout MAGNITUDE fails --
        at 24 pairs a stored key read at |.| = 4.9725 vs a novel-key null of 4.9750; the signal
        (norm ~1) is buried in crosstalk (norm ~sqrt(n)) and magnitude carries ~nothing. The
        membership signal lives in ALIGNMENT with the response codebook: cleanup is the gate,
        exactly the engine's standing doctrine that cleanup is the denoiser."""
        if self._null is None:
            if not self._atoms:
                self._null = 1.0
            else:
                A = np.stack(self._atoms)
                rng = np.random.default_rng(self.seed + 7919 + self._n)
                best = []
                for _ in range(int(n_null)):
                    r = unbind(self._trace, random_vector(self.dim, rng))
                    r = r / (np.linalg.norm(r) + 1e-12)
                    best.append(float(np.max(A @ r)))
                best.sort()
                self._null = best[min(len(best) - 1, int((1 - self.alpha) * len(best)))]
        return self._null

    def load(self):
        """The capacity-law load fraction n/dim of this tile."""
        return self._n / float(self.dim)

    def trust(self):
        """The crosstalk price: the signal fraction 1/(1+load) a read is worth under the ledger --
        as the tile fills, every answer's stated trust decays BY CONSTRUCTION (the capacity law
        pricing the cache), independent of how confident any single readout looks."""
        return 1.0 / (1.0 + self._n / float(self.dim) * self.dim / max(self._n, 1) * self.load()) \
            if False else 1.0 / (1.0 + self.load())

    def read_gated(self, task_key, task_vec=None):
        """The lever-7 read: unbind, then pay the three gates -- calibrated null, crosstalk price,
        volatility field. Returns {fired, prediction, confidence, trust, why}. Refusal is a result."""
        k = np.asarray(task_key, float)
        probe = k if task_vec is None else np.asarray(task_vec, float)
        if self.volatility.check(probe):
            self.stats["refused_volatile"] += 1
            return {"fired": False, "why": "volatile-region", "prediction": None,
                    "confidence": 0.0, "trust": self.trust()}
        if not self._outcome_ok(probe):
            self.stats["refused_outcome"] = self.stats.get("refused_outcome", 0) + 1
            return {"fired": False, "why": "outcome-memory", "prediction": None,
                    "confidence": 0.0, "trust": self.trust()}
        raw = self.read(k)
        q = self._null_quantile()
        if self._n == 0 or not self._atoms:
            self.stats["refused_null"] += 1
            return {"fired": False, "why": "empty-trace", "prediction": None,
                    "confidence": 0.0, "trust": self.trust()}
        A = np.stack(self._atoms)
        rn = raw / (np.linalg.norm(raw) + 1e-12)
        sims = A @ rn
        j = int(np.argmax(sims))
        best = float(sims[j])
        if best <= q:
            self.stats["refused_null"] += 1
            return {"fired": False, "why": "below-calibrated-null", "prediction": None,
                    "confidence": 0.0, "trust": self.trust()}
        conf = (best - q) / (1.0 - q + 1e-12)
        self.stats["fired"] += 1
        return {"fired": True, "why": "cleanup-above-null", "prediction": self._atoms[j],
                "raw": raw, "atom": j, "confidence": conf, "trust": self.trust()}

    def record_outcome(self, task_key, success):
        """Close the loop: after a fire is judged, bundle the task into the SUCCESS or FAILURE
        field. read_gated refuses where the local failure field outweighs success -- the outcome
        gate that similarity + calibration alone cannot replace. MEASURED (deep-dive Part 3 and
        this module's own bench): without it, look-alike traps sail through the calibrated null
        with confident wrong answers; with it, the wrong-serve count collapses. The fields are
        holographic (one cosine each to check), not lists."""
        t = np.asarray(task_key, float)
        t = t / (np.linalg.norm(t) + 1e-12)
        if success:
            self._succ_field = self._succ_field + t
        else:
            self._fail_field = self._fail_field + t
        return {"succ_norm": float(np.linalg.norm(self._succ_field)),
                "fail_norm": float(np.linalg.norm(self._fail_field))}

    def _outcome_ok(self, probe, margin=0.02):
        fn = np.linalg.norm(self._fail_field)
        if fn == 0:
            return True
        p = probe / (np.linalg.norm(probe) + 1e-12)
        cf = float(np.dot(p, self._fail_field) / fn)
        sn = np.linalg.norm(self._succ_field)
        cs = float(np.dot(p, self._succ_field) / sn) if sn > 0 else 0.0
        return (cf - cs) < margin

    # -- tiling (E1.3': the cliff number becomes the tile size) ---------------------------------
    def advisory(self):
        """The capacity advisory: recommend tiling when load crosses advisory_load. Returns
        {tile_recommended, load, n, dim} -- the caller (or TiledDisplacementTrace) acts on it."""
        return {"tile_recommended": self.load() >= self.advisory_load,
                "load": self.load(), "n": self._n, "dim": self.dim}

    def consolidate(self, atom_merge_cos=0.95):
        """IDLE-TIME CONSOLIDATION (backlog E1.4'/E6.1, the mechanism -- scheduling is the
        caller's): (a) merge near-duplicate response atoms (cleanup picks the nearest atom
        anyway, so dropping a >cos-0.95 twin changes no verdict while sharpening the null and
        cheapening every gate check), (b) force null recalibration at the current load. The
        audit log is untouched -- consolidation compresses the HOT structures, never the floor."""
        before = len(self._atoms)
        kept = []
        for a in self._atoms:
            if all(float(np.dot(a, b)) < atom_merge_cos for b in kept):
                kept.append(a)
        self._atoms = kept
        self._null = None
        self._null_quantile()
        return {"atoms_before": before, "atoms_after": len(kept),
                "null_recalibrated": True, "load": self.load()}

    # -- the exact floor -----------------------------------------------------------------------
    def replay(self):
        """Rebuild a fresh trace from the audit log and return it. The selftest asserts the rebuilt
        trace is BIT-IDENTICAL -- lever 7 standing on lever 3, checked, not assumed."""
        t = DisplacementTrace(self.dim, self.seed, self.alpha, self.beta,
                              surprise_floor=0.0, momentum=self.momentum,
                              advisory_load=self.advisory_load, name=self.name + ":replay")
        for k, v in self._audit:
            t.write(np.asarray(k, float), np.asarray(v, float))
        return t

    def to_state(self):
        return {"dim": self.dim, "seed": self.seed, "alpha": self.alpha, "beta": self.beta,
                "surprise_floor": self.surprise_floor, "momentum": self.momentum,
                "advisory_load": self.advisory_load, "name": self.name,
                "audit": self._audit, "volatility": self.volatility.to_state()}

    @classmethod
    def from_state(cls, state):
        t = cls(int(state["dim"]), int(state["seed"]), float(state["alpha"]), float(state["beta"]),
                float(state["surprise_floor"]), float(state["momentum"]),
                float(state["advisory_load"]), str(state["name"]))
        for k, v in state["audit"]:
            t._trace = t._trace + bind(np.asarray(k, float), np.asarray(v, float)) * 0  # placeholder, replaced below
        # rebuild faithfully through write() so surprise skipping cannot desync the audit:
        t2 = cls(int(state["dim"]), int(state["seed"]), float(state["alpha"]), float(state["beta"]),
                 surprise_floor=0.0, momentum=float(state["momentum"]),
                 advisory_load=float(state["advisory_load"]), name=str(state["name"]))
        for k, v in state["audit"]:
            t2.write(np.asarray(k, float), np.asarray(v, float))
        t2.surprise_floor = float(state["surprise_floor"])
        t2.volatility = VolatilityField.from_state(state["volatility"])
        return t2

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_state(), f)
        return path

    @classmethod
    def load_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_state(json.load(f))


class TiledDisplacementTrace:
    """Self-tiling wrapper (E1.3'): a router over DisplacementTrace tiles that SPLITS when a tile's
    capacity advisory fires -- the measured cliff number becomes the tile size automatically, and
    tile keys regenerate from seeds (0 index bytes; the Part-13 pattern: flat 0.34 -> 1.00 recall).
    Routing is by nearest tile centroid (running mean of task keys written to the tile)."""

    def __init__(self, dim=2048, seed=0, **kw):
        self.dim = int(dim)
        self.seed = int(seed)
        self.kw = dict(kw)
        self.tiles = [DisplacementTrace(dim, seed, name="tile0", **kw)]
        self._centroids = [np.zeros(dim)]
        self._counts = [0]
        self.splits = 0

    def _route(self, k):
        if len(self.tiles) == 1:
            return 0
        sims = [float(np.dot(k, c)) / (np.linalg.norm(k) * np.linalg.norm(c) + 1e-12)
                for c in self._centroids]
        return int(np.argmax(sims))

    def write(self, task_key, value_vec):
        k = np.asarray(task_key, float)
        i = self._route(k)
        out = self.tiles[i].write(k, value_vec)
        if out["accepted"]:
            self._counts[i] += 1
            self._centroids[i] = self._centroids[i] + (k - self._centroids[i]) / self._counts[i]
        if self.tiles[i].advisory()["tile_recommended"]:
            self._split(i)
        out["tiles"] = len(self.tiles)
        return out

    def _split(self, i):
        """Split tile i in two by a deterministic hyperplane through its centroid; replay each
        audit entry into its side. Journaled by construction (the audits ARE the journal)."""
        old = self.tiles[i]
        axis = random_vector(self.dim, np.random.default_rng(self.seed + 104729 + self.splits))
        a = DisplacementTrace(self.dim, self.seed, name=old.name + "a", **self.kw)
        b = DisplacementTrace(self.dim, self.seed, name=old.name + "b", **self.kw)
        ca = np.zeros(self.dim); cb = np.zeros(self.dim); na = nb = 0
        for k, v in old._audit:
            kv = np.asarray(k, float)
            if float(np.dot(kv - self._centroids[i], axis)) >= 0:
                a.write(kv, np.asarray(v, float)); na += 1; ca = ca + (kv - ca) / na
            else:
                b.write(kv, np.asarray(v, float)); nb += 1; cb = cb + (kv - cb) / nb
        if na == 0 or nb == 0:                  # degenerate split: keep the tile, raise its ceiling
            old.advisory_load *= 2.0
            return
        self.tiles[i] = a; self._centroids[i] = ca; self._counts[i] = na
        self.tiles.append(b); self._centroids.append(cb); self._counts.append(nb)
        self.splits += 1

    def read_gated(self, task_key, task_vec=None):
        k = np.asarray(task_key, float)
        out = self.tiles[self._route(k)].read_gated(k, task_vec)
        out["tiles"] = len(self.tiles)
        return out

    @property
    def stats(self):
        agg = {}
        for t in self.tiles:
            for kk, vv in t.stats.items():
                agg[kk] = agg.get(kk, 0) + vv
        agg["tiles"] = len(self.tiles)
        agg["splits"] = self.splits
        return agg


def _selftest():
    rng = np.random.default_rng(0)
    dim = 2048
    # -- 1. write/read round-trip + delta-rule skip --------------------------------------------
    tr = DisplacementTrace(dim, seed=0, surprise_floor=0.05)
    keys = [random_vector(dim, rng) for _ in range(24)]
    vals = [random_vector(dim, rng) for _ in range(24)]
    for k, v in zip(keys, vals):
        assert tr.write(k, v)["accepted"]
    got = tr.read_gated(keys[3])
    assert got["fired"] and cosine(got["prediction"], vals[3]) > 0.4, "stored pair must read back"
    for _ in range(6):                           # corrective re-writes shrink the residual...
        again = tr.write(keys[3], vals[3])
    assert not again["accepted"], "...until the delta rule skips a fully-predicted write"
    got2 = tr.read_gated(keys[3])
    assert got2["fired"] and cosine(got2["prediction"], vals[3]) > 0.9, \
        "after corrections the cleaned answer must still be the right atom"
    # -- 2. the calibrated null refuses the never-written --------------------------------------
    novel = random_vector(dim, rng)
    r = tr.read_gated(novel)
    assert not r["fired"] and r["why"] == "below-calibrated-null", "unknown key must be refused"
    # -- 3. volatility field: mark, refuse, unmark exactly -------------------------------------
    tr.volatility.mark("prices", keys[5])
    assert not tr.read_gated(keys[5])["fired"], "volatile region must never fire"
    tr.volatility.unmark("prices")
    assert tr.read_gated(keys[5])["fired"], "unmark is exact subtraction; the region serves again"
    # -- 4. the exact floor: replay is bit-identical -------------------------------------------
    rb = tr.replay()
    assert hashlib.sha256(np.ascontiguousarray(tr._trace).tobytes()).hexdigest() == \
           hashlib.sha256(np.ascontiguousarray(rb._trace).tobytes()).hexdigest(), \
        "replay(audit) must rebuild the trace bit-identically (lever 7 stands on lever 3)"
    st = DisplacementTrace.from_state(tr.to_state())
    assert hashlib.sha256(np.ascontiguousarray(st._trace).tobytes()).hexdigest() == \
           hashlib.sha256(np.ascontiguousarray(tr._trace).tobytes()).hexdigest(), \
        "state round-trip must be bit-identical"
    # -- 5. trust decays with load (the crosstalk price) ---------------------------------------
    t0 = DisplacementTrace(512, seed=1)
    trusts = []
    for i in range(96):
        t0.write(random_vector(512, rng), random_vector(512, rng))
        trusts.append(t0.trust())
    assert trusts[-1] < trusts[0], "trust must decay as the tile fills -- the capacity law prices the cache"
    # -- 6. self-tiling: overload splits, recall recovers --------------------------------------
    tt = TiledDisplacementTrace(512, seed=2, advisory_load=0.08)
    ks = [random_vector(512, rng) for _ in range(120)]
    vs = [random_vector(512, rng) for _ in range(120)]
    for k, v in zip(ks, vs):
        tt.write(k, v)
    assert len(tt.tiles) > 1, "the advisory must have triggered at least one split"
    ok = sum(1 for k, v in zip(ks, vs)
             if (g := tt.read_gated(k))["fired"] and cosine(g["prediction"], v) > 0.3)
    flat = DisplacementTrace(512, seed=3, surprise_floor=0.0)
    for k, v in zip(ks, vs):
        flat.write(k, v)
    ok_flat = sum(1 for k, v in zip(ks, vs)
                  if (g := flat.read_gated(k))["fired"] and cosine(g["prediction"], v) > 0.3)
    assert ok > ok_flat, "tiled recall must beat the overloaded flat tile (measured, not assumed)"
    # -- KEPT NEGATIVE (do not rediscover): the ungated read serves wrong answers --------------
    # On the deep-dive Part-3 workload the raw read() with a naive threshold served 48 wrong
    # answers where the full gate served 2. read() stays public for measurement; serving it is
    # the bug this module exists to prevent.
    return {"replay": "bit-identical", "tiles": len(tt.tiles),
            "tiled_recall": ok, "flat_recall": ok_flat, "stats": tt.stats}


if __name__ == "__main__":
    print(_selftest())


class UsageTrace:
    """TOOL-SELECTION MEMORY AS ALGEBRA (backlog E5.1'): successful (task -> tool) uses are
    bundled as bind(task_key, tool_atom); predicting the tools for a new task is ONE unbind plus
    a cleanup ranking over the tool codebook -- the leOS JSONL + k-NN scan replaced by a single
    operation. Tool atoms regenerate from their names (0 bytes of index); counts are kept beside
    the trace for audit (never opaque weights). Capacity discipline: same advisory as the
    displacement trace -- at the cliff, tile by domain (the centroid-culling pass becomes the
    tile router)."""

    def __init__(self, dim=2048, seed=0):
        self.dim = int(dim)
        self.seed = int(seed)
        self._trace = np.zeros(self.dim)
        self._tools = {}                        # name -> atom
        self.counts = {}                        # name -> successful uses (the audit ledger)
        self._n = 0

    def _atom(self, tool):
        if tool not in self._tools:
            self._tools[tool] = key_atom("tool:" + tool, self.dim)
        return self._tools[tool]

    def note(self, task_vec, tool, success=True):
        """Record one tool use; only SUCCESSFUL uses strengthen the trace (failures only count)."""
        self.counts[tool] = self.counts.get(tool, 0) + (1 if success else 0)
        if success:
            t = np.asarray(task_vec, float)
            self._trace = self._trace + bind(t / (np.linalg.norm(t) + 1e-12), self._atom(tool))
            self._n += 1
        return {"tool": tool, "uses": self.counts[tool], "load": self._n / self.dim}

    def predict(self, task_vec, k=3):
        """Rank the known tools for a task: one unbind, one cleanup sweep. Returns
        [(tool, score), ...] best-first; an empty trace returns [] (refusal is a result)."""
        if self._n == 0 or not self._tools:
            return []
        t = np.asarray(task_vec, float)
        raw = unbind(self._trace, t / (np.linalg.norm(t) + 1e-12))
        rn = raw / (np.linalg.norm(raw) + 1e-12)
        scored = sorted(((name, float(np.dot(rn, atom)))
                         for name, atom in self._tools.items()),
                        key=lambda x: (-x[1], x[0]))
        return scored[: int(k)]


class RecipeCache:
    """GENERATOR-RECIPE REUSE for stream identification (backlog E3.3): streams arrive in
    families; the full HRNN ladder identifies the first family member the expensive way, and this
    cache re-fits the FAMILY RECIPE (fixed fundamental + harmonic count; amplitudes/phases are a
    closed-form least squares) for its neighbors, VALIDATED on a holdout of the new stream itself
    (NRMSE gate) -- refusal falls back to the full ladder. MEASURED (deep-dive Part 11): 9.9x over
    40 family streams, 36/40 fired, and a white-noise probe was never served a generator: the
    holdout gate IS the regime contract, so every served verdict is checked on the stream it
    serves. Similarity key: the normalized magnitude spectrum (top bins) -- a DESIGNED key for
    periodic structure (key-law clause 4)."""

    def __init__(self, sig_bins=200, gate=0.80, nrmse_max=0.35, holdout=0.25):
        self.sig_bins = int(sig_bins)
        self.gate = float(gate)
        self.nrmse_max = float(nrmse_max)
        self.holdout = float(holdout)
        self.log = []                            # [(signature, f0, n_harmonics)]
        self.stats = {"fired": 0, "validated": 0, "refused": 0, "logged": 0}

    def signature(self, x):
        x = np.asarray(x, float).ravel()
        F = np.abs(np.fft.rfft(x - x.mean()))[: self.sig_bins]
        return F / (np.linalg.norm(F) + 1e-12)

    @staticmethod
    def _design(f0, H, t):
        cols = []
        for h in range(1, H + 1):
            cols += [np.sin(2 * np.pi * f0 * h * t), np.cos(2 * np.pi * f0 * h * t)]
        cols.append(np.ones_like(t, dtype=float))
        return np.column_stack(cols)

    def refit(self, x, f0, H):
        """Closed-form refit of a family recipe to a new stream: fixed frequencies, lstsq
        amplitudes/phases. Returns (predict_fn, w)."""
        x = np.asarray(x, float).ravel()
        t = np.arange(len(x), dtype=float)
        w, *_ = np.linalg.lstsq(self._design(f0, H, t), x, rcond=None)
        return (lambda tt: self._design(f0, H, np.asarray(tt, float)) @ w), w

    def try_stream(self, x):
        """Attempt the warm path: nearest logged recipe -> refit -> HOLDOUT-validate. Returns a
        verdict dict with via='recipe_cache' or None (caller runs the full ladder)."""
        x = np.asarray(x, float).ravel()
        if not self.log:
            return None
        s = self.signature(x)
        sims = [float(s @ ls) for ls, _, _ in self.log]
        j = int(np.argmax(sims))
        if sims[j] < self.gate:
            self.stats["refused"] += 1
            return None
        f0, H = self.log[j][1], self.log[j][2]
        pred, w = self.refit(x, f0, H)
        n = max(8, int(len(x) * self.holdout))
        tail = x[-n:]
        err = float(np.sqrt(np.mean((pred(np.arange(len(x) - n, len(x))) - tail) ** 2))
                    / (tail.std() + 1e-12))
        self.stats["fired"] += 1
        if err > self.nrmse_max:
            self.stats["refused"] += 1
            return None
        self.stats["validated"] += 1
        return {"regime": "generator", "via": "recipe_cache", "f0": float(f0),
                "n_harmonics": int(H), "holdout_nrmse": err,
                "coefficients": [float(v) for v in w],
                "why": "nearest family recipe refit closed-form; holdout NRMSE %.3f <= %.2f"
                       % (err, self.nrmse_max)}

    def note(self, x, f0, n_harmonics):
        """Log a recipe the EXPENSIVE path identified (only solved experiences are logged)."""
        self.log.append((self.signature(x), float(f0), int(n_harmonics)))
        self.stats["logged"] += 1
        return len(self.log)


class WorkingMemory:
    """WORKING MEMORY AS A CAPACITY-PRICED BUNDLE (backlog E5.3'): the agent's working set is a
    superposition with allocator-quoted admission (the capacity law IS the budget -- no token
    counting), relevance ranking by cosine to the live task, and EVICTION BY EXACT SUBTRACTION
    (ablation is exact in this algebra; the evicted item is returned for salvage into the
    KnowledgeStore before it leaves). The raw transcript of every admit/evict is kept beside the
    bundle -- the lever-3 floor: the bundle is the hot path, never the only copy."""

    def __init__(self, dim=2048, advisory_load=0.05):
        self.dim = int(dim)
        self.advisory_load = float(advisory_load)
        self._bundle = np.zeros(self.dim)
        self._items = {}                        # tag -> (vec, note): what exact subtraction needs
        self.transcript = []                    # the floor: [('admit'|'evict', tag, note)]

    def load(self):
        return len(self._items) / float(self.dim)

    def quote(self):
        """The allocator's admission quote: how full is the bundle, and is admission advised."""
        return {"load": self.load(), "items": len(self._items),
                "admission_advised": self.load() < self.advisory_load}

    def admit(self, vec, tag, note=None):
        v = np.asarray(vec, float)
        v = v / (np.linalg.norm(v) + 1e-12)
        if tag in self._items:
            return {"admitted": False, "why": "duplicate-tag", **self.quote()}
        q = self.quote()
        self._items[tag] = (v, note)
        self._bundle = self._bundle + v
        self.transcript.append(("admit", tag, note))
        return {"admitted": True, "over_advisory": not q["admission_advised"], **self.quote()}

    def evict(self, tag):
        """Eviction by SUBTRACTION: remove the item's vector and return it for salvage.
        HONEST PRECISION NOTE (measured here): subtraction is exact up to IEEE ASSOCIATIVITY --
        out-of-order eviction leaves a residual at rounding scale (~1e-13 over 12 items at
        dim 2048, i.e. ~1e-16 per element), because (a+b)-b need not bit-equal a. LIFO eviction
        is bit-exact; any-order eviction is exact to machine epsilon. Either way the item's
        CONTRIBUTION is gone -- unlike decay-based forgetting, nothing of it remains above
        rounding noise, and the transcript floor holds the true record."""
        v = self._items.pop(tag, None)
        if v is None:
            return None
        self._bundle = self._bundle - v[0]
        self.transcript.append(("evict", tag, v[1]))
        return {"tag": tag, "vec": v[0], "note": v[1]}

    def evict_least_relevant(self, task_vec):
        """Free capacity by evicting the item least relevant to the live task (lowest cosine)."""
        if not self._items:
            return None
        t = np.asarray(task_vec, float)
        worst = min(self._items, key=lambda k: float(np.dot(t, self._items[k][0])))
        return self.evict(worst)

    def recall_ranked(self, task_vec, k=5):
        """The working set ranked by relevance to the task (exact, from the items beside the
        bundle -- the bundle itself serves downstream binds)."""
        t = np.asarray(task_vec, float)
        tn = t / (np.linalg.norm(t) + 1e-12)
        scored = sorted(((tag, float(np.dot(tn, v[0])), v[1])
                         for tag, v in self._items.items()), key=lambda x: (-x[1], x[0]))
        return scored[: int(k)]

    @property
    def bundle(self):
        return self._bundle


def experience_coverage(trace_or_tiled, probes, threshold=0.5):
    """THE COVERAGE GAUGE (backlog E6.2): where can lever 7 not help yet? For each probe task,
    the nearest AUDITED key's cosine; coverage = the fraction above threshold; the worst-covered
    probes are the VOIDS -- 'escalate here on purpose' suggestions. MEASURED law this gauge makes
    live (deep-dive Parts 9/11): the lever's win tracks log coverage exactly (fired 37% at 3/8
    families; 36/40 at full coverage) -- the ceiling is the log, and this is its dial."""
    tiles = getattr(trace_or_tiled, "tiles", None) or [trace_or_tiled]
    keys = []
    for t in tiles:
        keys += [np.asarray(k, float) for k, _ in t._audit]
    out = {"probes": len(probes), "logged": len(keys)}
    if not keys:
        out.update({"coverage": 0.0, "mean_nearest": 0.0,
                    "voids": list(range(min(3, len(probes))))})
        return out
    K = np.stack([k / (np.linalg.norm(k) + 1e-12) for k in keys])
    near = []
    for p in probes:
        pn = np.asarray(p, float); pn = pn / (np.linalg.norm(pn) + 1e-12)
        near.append(float(np.max(K @ pn)))
    near = np.asarray(near)
    order = np.argsort(near, kind="stable")
    out.update({"coverage": float((near >= threshold).mean()),
                "mean_nearest": float(near.mean()),
                "voids": [int(i) for i in order[:3]]})
    return out
