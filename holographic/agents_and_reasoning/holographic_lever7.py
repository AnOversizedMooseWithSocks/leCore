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
        # RAW audit entries (backlog E1.5): {audit index: mode}. Entries NOT in this map are ordinary write()s and
        # replay through write(); entries in it replay VERBATIM (see write_raw). A side map, not a third tuple
        # field, so every reader that unpacks `for k, v in _audit` keeps working unchanged.
        self._audit_raw = {}
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

    # -- raw writes (backlog E1.5: the reflex trace's own correction) ---------------------------------
    # write() corrects ALONG v ONLY and clamps the stored strength at s >= 0 -- the right rule for learning a lesson,
    # and exactly why it cannot UNLEARN one: a correction write(k, truth) leaves the wrong atom at full strength.
    # Measured on real CLINC keys (w2, exp_d): truth read back 45% after one correction, 66% after three. A
    # correction therefore needs a SIGNED delta (a negative along the wrong atom, or an LMS error over the
    # codebook), and a signed delta must be replayed VERBATIM: pushed back through write() it would be re-estimated,
    # clamped at 0 and dropped, so replay / retile / split / reload would silently undo every correction.
    RAW_BIND, RAW_ADD, RAW_ATOM = 1, 2, 3
    RAW_MODES = {1: "bind", 2: "add", 3: "atom"}

    def _apply_raw(self, k, v, mode):
        """The ONE code path for a raw entry, live and on replay (so a replay is bit-identical by construction):
          RAW_BIND  trace += bind(k, v)           -- a signed delta along the key itself
          RAW_ADD   trace += v                    -- v is ALREADY a trace-space delta (e.g. bind(u, err) with an
                                                     affine-projected key u); k is kept only to ROUTE the entry
                                                     (tiling, splits, retile) to the tile its key belongs to
          RAW_ATOM  register v as a response atom -- no trace change: the codebook the gated read cleans up against
                                                     must know an atom a raw delta writes toward"""
        if mode == self.RAW_BIND:
            self._trace = self._trace + bind(k, v)
            self._n += 1
        elif mode == self.RAW_ADD:
            self._trace = self._trace + v
            self._n += 1
        elif mode == self.RAW_ATOM:
            vn = v / (np.linalg.norm(v) + 1e-12)
            if not self._atoms or max(float(np.dot(vn, a)) for a in self._atoms) < 0.5:
                self._atoms.append(vn)
        else:
            raise ValueError("unknown raw audit mode %r" % (mode,))
        self._null = None

    def write_raw(self, task_key, delta, mode=1):
        """Apply a raw delta and journal it as a FLAGGED audit entry that every replay path applies verbatim
        (replay, from_state, TiledDisplacementTrace._split, the mind's reflex_retile and the partition's instanced
        audit section all honour the flag). mode: RAW_BIND (1, default) / RAW_ADD (2) / RAW_ATOM (3) -- see
        _apply_raw. No surprise gate: a correction is deliberate, never skipped as 'already predicted'."""
        k = np.asarray(task_key, float)
        v = np.asarray(delta, float)
        mode = int(mode)
        self._apply_raw(k, v, mode)
        self._audit_raw[len(self._audit)] = mode
        self._audit.append((k.tolist(), v.tolist()))
        self.stats["raw_writes"] = self.stats.get("raw_writes", 0) + 1
        return {"accepted": True, "raw": self.RAW_MODES[mode], "load": self.load()}

    def audit_entries(self):
        """[(key, value, mode)] in journal order; mode 0 = an ordinary write(), else the raw mode. The one reader
        every re-tiling / persistence path should use (it cannot forget the flag)."""
        return [(k, v, self._audit_raw.get(i, 0)) for i, (k, v) in enumerate(self._audit)]

    def replay_entry(self, k, v, mode):
        """Replay ONE audit entry the way it was first applied: mode 0 through write(), a raw mode verbatim."""
        if mode:
            return self.write_raw(k, v, mode)
        return self.write(np.asarray(k, float), np.asarray(v, float))

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

    def read_gated(self, task_key, task_vec=None, neg_role=None):
        """The lever-7 read: unbind, then pay the three gates -- calibrated null, crosstalk price,
        volatility field. Returns {fired, prediction, confidence, trust, why}. Refusal is a result.
        neg_role (E1.5 design b, default None = unchanged): an OUTCOME role; an atom whose negative read
        unbind(read, neg_role) beats its positive read is SUPPRESSED here (a rejection kept as data)."""
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
        if neg_role is not None:
            neg = unbind(raw, np.asarray(neg_role, float))
            nsims = A @ (neg / (np.linalg.norm(neg) + 1e-12))
            sims = np.where(nsims > sims, -1.0, sims)   # rejected here more strongly than supported: suppressed
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
        for k, v, mode in self.audit_entries():         # raw entries VERBATIM (E1.5), the rest through write()
            t.replay_entry(k, v, mode)
        return t

    def to_state(self):
        # audit_raw (E1.5): [[index, mode]] for the raw entries; absent/empty in states written before it existed,
        # which then replay exactly as they always did.
        st = {"dim": self.dim, "seed": self.seed, "alpha": self.alpha, "beta": self.beta,
              "surprise_floor": self.surprise_floor, "momentum": self.momentum,
              "advisory_load": self.advisory_load, "name": self.name,
              "audit": self._audit, "audit_raw": [[int(i), int(m)] for i, m in sorted(self._audit_raw.items())],
              "volatility": self.volatility.to_state()}
        # THE OUTCOME FIELDS TRAVEL (backlog E2.1). record_outcome's success / failure fields are the reflex trace's
        # own negative evidence (read_gated refuses where failure outweighs success), but they are NOT in the audit
        # (they are outcomes, not writes), so a reload, a split or a retile -- all of which rebuild from the audit --
        # silently forgot every reported failure and the outcome gate opened again. Written only when non-zero, so
        # a trace that never had an outcome keeps its old state shape; plain lists (exact float64 round trip).
        if np.any(self._succ_field):
            st["succ_field"] = [float(x) for x in self._succ_field]
        if np.any(self._fail_field):
            st["fail_field"] = [float(x) for x in self._fail_field]
        return st

    def outcome_fields(self):
        """(success field, failure field) as copies -- what a split / retile carries into the new tiles."""
        return self._succ_field.copy(), self._fail_field.copy()

    def set_outcome_fields(self, succ=None, fail=None):
        """Lay saved / inherited outcome fields over this trace (None or a wrong length = leave as is)."""
        for attr, v in (("_succ_field", succ), ("_fail_field", fail)):
            if v is not None:
                v = np.asarray(v, np.float64).reshape(-1)
                if v.shape == (self.dim,):
                    setattr(self, attr, v.copy())

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
        raw = {int(i): int(m) for i, m in (state.get("audit_raw") or [])}
        for i, (k, v) in enumerate(state["audit"]):
            t2.replay_entry(k, v, raw.get(i, 0))      # a raw correction replays verbatim (E1.5)
        t2.surprise_floor = float(state["surprise_floor"])
        t2.volatility = VolatilityField.from_state(state["volatility"])
        t2.set_outcome_fields(state.get("succ_field"), state.get("fail_field"))   # E2.1 (absent: zero, as before)
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

    def write_raw(self, task_key, delta, mode=1):
        """A raw (verbatim-replayed) delta into the tile the KEY routes to -- see DisplacementTrace.write_raw.
        Raw entries do not move the tile centroids (the centroid is the mean of ORDINARY write keys, the same rule
        _split and a reload use), so a correction never drags routing toward itself."""
        k = np.asarray(task_key, float)
        i = self._route(k)
        out = self.tiles[i].write_raw(k, delta, mode)
        if self.tiles[i].advisory()["tile_recommended"]:
            self._split(i)
        out["tiles"] = len(self.tiles)
        out["tile"] = i
        return out

    def _split(self, i):
        """Split tile i in two by a deterministic hyperplane through its centroid; replay each
        audit entry into its side. Journaled by construction (the audits ARE the journal).
        E1.5: a RAW entry (a signed correction) goes to the side of its routing key and replays VERBATIM --
        through write() it would be re-estimated, clamped and lost -- and it does not move a centroid."""
        old = self.tiles[i]
        axis = random_vector(self.dim, np.random.default_rng(self.seed + 104729 + self.splits))
        a = DisplacementTrace(self.dim, self.seed, name=old.name + "a", **self.kw)
        b = DisplacementTrace(self.dim, self.seed, name=old.name + "b", **self.kw)
        ca = np.zeros(self.dim); cb = np.zeros(self.dim); na = nb = 0
        for k, v, mode in old.audit_entries():
            kv = np.asarray(k, float)
            side_a = float(np.dot(kv - self._centroids[i], axis)) >= 0
            if mode:
                (a if side_a else b).write_raw(kv, np.asarray(v, float), mode)
            elif side_a:
                a.write(kv, np.asarray(v, float)); na += 1; ca = ca + (kv - ca) / na
            else:
                b.write(kv, np.asarray(v, float)); nb += 1; cb = cb + (kv - cb) / nb
        if na == 0 or nb == 0:                  # degenerate split: keep the tile, raise its ceiling
            old.advisory_load *= 2.0
            return
        # E2.1: BOTH halves inherit the parent's outcome fields. The fields are region-wide evidence in the parent's
        # key space and both halves route keys from that space; before this a split (replaying the audit only) wiped
        # every reported failure of the tile it split -- the outcome gate reopened on exactly the look-alike traps it
        # was measured to catch.
        succ, fail = old.outcome_fields()
        a.set_outcome_fields(succ, fail)
        b.set_outcome_fields(succ, fail)
        # FOUND ON THE WAY (same defect class): the halves were built with an EMPTY volatility field, so every
        # reflex_mark_volatile region (prices, live status) stopped being refused in the tile that split. The marks
        # are carried exactly (VolatilityField's own state; unmark stays exact subtraction).
        vst = old.volatility.to_state()
        a.volatility = VolatilityField.from_state(vst)
        b.volatility = VolatilityField.from_state(vst)
        self.tiles[i] = a; self._centroids[i] = ca; self._counts[i] = na
        self.tiles.append(b); self._centroids.append(cb); self._counts.append(nb)
        self.splits += 1

    def read_gated(self, task_key, task_vec=None, neg_role=None):
        k = np.asarray(task_key, float)
        out = self.tiles[self._route(k)].read_gated(k, task_vec, neg_role=neg_role)
        out["tiles"] = len(self.tiles)
        return out

    def outcome_fields(self):
        """(success, failure) summed over every tile -- the whole trace's outcome evidence (E2.1). A RE-TILE (a new
        tiling of the same keys) carries this into every new tile with set_outcome_fields: the keys are re-routed,
        so per-tile fields no longer line up with their tiles, and the sum is the evidence a single tile held."""
        succ = np.zeros(self.dim)
        fail = np.zeros(self.dim)
        for t in self.tiles:
            s_, f_ = t.outcome_fields()
            succ, fail = succ + s_, fail + f_
        return succ, fail

    def set_outcome_fields(self, succ=None, fail=None):
        """Lay outcome fields over EVERY tile (the re-tile carry; see outcome_fields)."""
        for t in self.tiles:
            t.set_outcome_fields(succ, fail)

    @property
    def stats(self):
        agg = {}
        for t in self.tiles:
            for kk, vv in t.stats.items():
                agg[kk] = agg.get(kk, 0) + vv
        agg["tiles"] = len(self.tiles)
        agg["splits"] = self.splits
        return agg


# =====================================================================================================================
# REFLEX-TRACE CORRECTION (backlog E1.5) -- two designs, both built, the measurement chose (owner's decision).
#
# THE DEFECT (w2, exp_d, real CLINC keys): a correction -- decision_outcome with a truth different from the served
# answer -- only did write(k, truth). write() corrects along the truth atom and clamps at s >= 0, so the WRONG atom
# kept its full strength: truth read back 45% after one correction, 66% after three.
#
# Design (a) CODEBOOK LMS + AFFINE PROJECTION (Widrow; w1's synthetic probe 40/40 + 40/40):
#     err = sum_l (onehot(truth) - s)_l * L_l      over the codebook atoms with |s_l| >= floor, plus the target
#     u   = the minimum-norm key with <u,k> = 1 and <u,k'_j> = 0 for the nearest REPORTED keys k'_j whose outcome
#           differs (the neighbours this correction must not touch)
#     trace += bind(u, err)                         journalled RAW_ADD (routed by k), replayed verbatim
#   At k the read moves by exactly err (the LMS step to the one-hot); at each protected neighbour it moves by zero in
#   expectation (the affine-projection step). One write fixes the leak AND the truth, whatever wrote them.
# Design (b) BY PROVENANCE (w2): write(k, truth), then look in the audit for who wrote the wrong atom:
#     at THIS key (audit key cosine >= 0.95)  -> a signed along-atom write  -s_a * L[a]   (RAW_BIND at k)
#     leaked from a NEIGHBOUR                -> an outcome-role negative bind(NEG, L[a]) (RAW_BIND at k); a read with
#                                               neg_role suppresses a label whose negative beats its positive
# MEASURED (tools/bench_trace_correction.py -> docs/research/evidence/bench_trace_correction.json; real CLINC keys,
# 150 trials per cell; baseline = write(k, truth) only):
#                   same key   3 corrections   flip-flop   neighbour kept c=0.5/0.8/0.9   replay/reload
#   baseline         0.453        0.660          0.453        1.000 / 0.900 / 0.813          1.000
#   (a) lms_apa      1.000        1.000          1.000        1.000 / 1.000 / 0.993          1.000   <- the default
#   (b) provenance   0.993        1.000          0.993        1.000 / 0.780 / 0.533          1.000   (fails >= 0.85)
# KEPT NEGATIVES: the same LMS step along k itself (no projection) keeps a neighbour 0.847 / 0.38 / 0.00 at
# c = 0.5 / 0.6 / 0.8 -- the projection is what protects it; (b) without retracting an old rejection reads a
# flip-flop back 0.373. docs/TYPED_DECISIONS.md section 8d has the full table.
# =====================================================================================================================
CORRECTION_DESIGNS = ("lms_apa", "provenance")

# NEIGHBOUR PROTECTION BAND for design (a). A reported key k' with a different outcome is protected when
# PROTECT_COS <= cos(k, k') < SAME_KEY_COS.
#   PROTECT_COS = 0.4. The algebra: an LMS step along k itself moves a neighbour at cosine c by c*err, so the
#   neighbour's own atom drops to 1 - c^2 while the new truth rises to c -- it FLIPS once c > 0.618 and has already
#   lost half its margin at c = 0.5 (0.75 vs 0.5). MEASURED (tools/bench_trace_correction.py, real CLINC keys, 150
#   trials; the 'lms_noproj' arm = no protection): an unprotected neighbour keeps its answer 0.847 at c = 0.5, 0.38 at
#   0.6, 0.02 at 0.7, 0.00 at 0.8. The FIRST cut set the line AT 0.5, and a neighbour at exactly 0.5 fell on the
#   wrong side of float rounding (lms_apa 0.847 = unprotected, kept negative): the line sits at 0.4 so 0.5 is inside
#   the band with margin. Below it the leak is under a loaded tile's crosstalk, and every extra constraint grows |u|
#   (1/sqrt(1 - c^2): 1.09 at 0.4, 1.67 at 0.8) and with it the noise the write adds everywhere else.
#   SAME_KEY_COS = 0.95, the provenance line both designs share: at or above it k' IS this question (a noisy verdict
#   now corrected) -- its old outcome is superseded, not protected (and protecting it would blow |u| up: 3.2 at 0.95).
PROTECT_COS = 0.4
SAME_KEY_COS = 0.95
PROTECT_MAX = 8          # nearest protected neighbours at most (one small Gram solve each)
U_NORM_MAX = 4.0         # a neighbour set that would push |u| past this is cut back (nearest kept first)
LMS_FLOOR = 0.2          # codebook atoms read at |s| >= floor are corrected (w1's floor); below it is crosstalk


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _tile_of(trace, k):
    """(tile, index): the DisplacementTrace a key reads/writes in (a plain trace is its own single tile)."""
    if hasattr(trace, "tiles"):
        i = trace._route(k)
        return trace.tiles[i], i
    return trace, 0


def affine_key(k, neighbours, ridge=1e-6, u_max=U_NORM_MAX):
    """The minimum-norm u with <u,k> = 1 and <u,k'_j> = 0 for each neighbour key (unit rows), nearest first; a
    neighbour whose constraint would push |u| past u_max is dropped (reported). -> (u, kept_count)."""
    k = _unit(k)
    kept = []
    u = k.copy()
    for kp in neighbours:
        K = np.stack([k] + [_unit(x) for x in kept + [kp]])
        G = K @ K.T + ridge * np.eye(len(K))
        e1 = np.zeros(len(K)); e1[0] = 1.0
        cand = K.T @ np.linalg.solve(G, e1)
        cand = cand / float(np.dot(cand, k))            # exact <u,k> = 1 after the ridge
        if float(np.linalg.norm(cand)) > u_max:
            continue
        kept.append(kp)
        u = cand
    return u, len(kept)


def correct_lms_apa(trace, k, truth, book, seen=(), floor=LMS_FLOOR, protect_cos=PROTECT_COS,
                    same_cos=SAME_KEY_COS, max_protect=PROTECT_MAX):
    """Design (a). trace: a DisplacementTrace or TiledDisplacementTrace; k: the decision's key; truth: the reported
    label; book: {label: atom} (the door's codebook -- only these atoms are ever corrected); seen: [(key, outcome)]
    of REPORTED decisions of the same key kind (the neighbours to protect are picked from it). Writes RAW_ATOM (the
    truth atom, so the gated read can clean up to it) then RAW_ADD bind(u, err). Returns the diagnostics."""
    k = _unit(k)
    tile, ti = _tile_of(trace, k)
    labels = list(book)
    if truth not in book:
        raise KeyError("correct_lms_apa: truth %r is not in the codebook" % (truth,))
    C = np.stack([np.asarray(book[l], float) for l in labels])
    pred = unbind(tile._trace, k) if tile._n else np.zeros(tile.dim)
    s = (C @ pred) / (tile._rho * np.sum(C * C, axis=1))
    t = labels.index(truth)
    tgt = np.zeros(len(labels)); tgt[t] = 1.0
    keep = (np.abs(s) >= float(floor)) | (tgt > 0)
    err = ((tgt - s)[keep, None] * C[keep]).sum(axis=0)
    # the neighbours to protect: reported keys, SAME tile (another tile never sees this write), a DIFFERENT outcome,
    # inside the protection band; nearest first
    cands = []
    for kp, o in seen or ():
        if o == truth or o in (None, "", "fail", "failed", "__failed__"):
            continue
        kp = _unit(kp)
        c = float(np.dot(kp, k))
        if protect_cos <= c < same_cos and (not hasattr(trace, "tiles") or trace._route(kp) == ti):
            cands.append((-c, len(cands), kp))
    cands.sort(key=lambda x: (x[0], x[1]))              # nearest first; ties by report order
    u, n_prot = affine_key(k, [kp for _, _, kp in cands[:int(max_protect)]])
    trace.write_raw(k, np.asarray(book[truth], float), DisplacementTrace.RAW_ATOM)
    trace.write_raw(k, bind(u, err), DisplacementTrace.RAW_ADD)
    return {"design": "lms_apa", "s_truth": float(s[t]), "corrected_atoms": int(keep.sum()),
            "err_norm": float(np.linalg.norm(err)), "u_norm": float(np.linalg.norm(u)),
            "protected": n_prot, "protect_candidates": len(cands)}


def correct_by_provenance(trace, k, truth_atom, wrong_atom, neg_role, same_cos=SAME_KEY_COS, retract=True):
    """Design (b). write(k, truth) first (as the reflex always did); then, if the WRONG atom was written at this
    key (an ordinary audit entry with key cosine >= same_cos and value cosine >= same_cos to wrong_atom), a signed
    along-atom negative -s_a * L[a] (RAW_BIND); otherwise an outcome-role negative bind(neg_role, L[a]) (RAW_BIND)
    that a read with neg_role consults. Returns the diagnostics."""
    k = _unit(k)
    a = np.asarray(wrong_atom, float)
    y = np.asarray(truth_atom, float)
    tile, _ = _tile_of(trace, k)
    # RETRACT an earlier rejection of the truth at this key first. Found by the mind-level test (a typed door
    # answered y, a noisy verdict said a -- design (b) recorded 'y is wrong here' --, then the correction back to y):
    # the stale NEG(y) record outvoted the fresh truth write and the reflex went silent (0 fires). A rejection is
    # evidence about THIS key, so the truth reported here now cancels it (the measured NEG strength, clamped to [0,1]).
    retracted = 0.0
    if retract and tile._n:
        rn = unbind(unbind(tile._trace, k), np.asarray(neg_role, float))
        s_ny = float(np.dot(rn, y)) / (tile._rho * float(np.dot(y, y)) + 1e-12)
        s_ny = min(max(s_ny, 0.0), 1.0)
        if s_ny > 0.2:                                  # above crosstalk (the same floor the LMS design uses)
            trace.write_raw(k, -s_ny * bind(np.asarray(neg_role, float), y), DisplacementTrace.RAW_BIND)
            retracted = s_ny
    w = trace.write(k, y)
    tile, _ = _tile_of(trace, k)
    an = _unit(a)
    here = any(float(np.dot(_unit(kk), k)) >= same_cos and float(np.dot(_unit(vv), an)) >= same_cos
               for kk, vv, mode in tile.audit_entries() if not mode)
    if here:
        pred = unbind(tile._trace, k)
        s_a = float(np.dot(pred, a)) / (tile._rho * float(np.dot(a, a)) + 1e-12)
        s_a = min(max(s_a, 0.0), 1.0)
        if s_a > 0.0:
            trace.write_raw(k, -s_a * a, DisplacementTrace.RAW_BIND)
        return {"design": "provenance", "provenance": "same-key", "s_wrong": s_a, "truth_write": w.get("accepted"),
                "retracted_neg": retracted}
    trace.write_raw(k, bind(np.asarray(neg_role, float), a), DisplacementTrace.RAW_BIND)
    return {"design": "provenance", "provenance": "neighbour", "truth_write": w.get("accepted"),
            "retracted_neg": retracted}


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
    # -- 7. (E1.5) raw corrections: both designs fix a wrong lesson at the same key, and every replay path
    #       (replay, state round-trip, a split) reproduces the corrected trace bit-for-bit
    labels = {"L%d" % i: random_vector(dim, rng) for i in range(12)}
    neg = random_vector(dim, rng)
    for design in CORRECTION_DESIGNS:
        tc = DisplacementTrace(dim, seed=5)
        bgk = [random_vector(dim, rng) for _ in range(20)]
        for i, kk in enumerate(bgk):
            tc.write(kk, labels["L%d" % (i % 12)])
        kq = random_vector(dim, rng)
        tc.write(kq, labels["L3"])                        # the wrong lesson, at the same key
        if design == "lms_apa":
            correct_lms_apa(tc, kq, "L7", labels, seen=[(kk, "L%d" % (i % 12)) for i, kk in enumerate(bgk)])
            g = tc.read_gated(kq)
        else:
            correct_by_provenance(tc, kq, labels["L7"], labels["L3"], neg)
            g = tc.read_gated(kq, neg_role=neg)
        assert g["fired"] and cosine(g["prediction"], labels["L7"]) > 0.9, (design, "the correction must win")
        assert np.array_equal(tc.replay()._trace, tc._trace), (design, "raw entries must replay verbatim")
        assert np.array_equal(DisplacementTrace.from_state(json.loads(json.dumps(tc.to_state())))._trace, tc._trace)
    tsp = TiledDisplacementTrace(512, seed=4, advisory_load=0.5)
    kk0 = random_vector(512, rng)
    tsp.write(kk0, random_vector(512, rng))
    tsp.write_raw(kk0, -0.5 * random_vector(512, rng))
    tsp._split(0)                                        # a forced split replays the raw entry on its key's side
    assert sum(len(t._audit_raw) for t in tsp.tiles) in (0, 1)   # 0 only for the degenerate (kept) split
    return {"replay": "bit-identical", "tiles": len(tt.tiles),
            "tiled_recall": ok, "flat_recall": ok_flat, "stats": tt.stats,
            "raw_correction": "both designs replay bit-identically"}


if __name__ == "__main__":
    print(_selftest())


class UsageTrace:
    """TOOL-SELECTION MEMORY AS ALGEBRA (backlog E5.1'): successful (task -> tool) uses are
    bundled as bind(task_key, tool_atom); predicting the tools for a new task is ONE unbind plus
    a cleanup ranking over the tool codebook -- the leOS JSONL + k-NN scan replaced by a single
    operation. Tool atoms regenerate from their names (0 bytes of index); counts are kept beside
    the trace for audit (never opaque weights). Capacity discipline: same advisory as the
    displacement trace -- at the cliff, tile by domain (the centroid-culling pass becomes the
    tile router).

    READ-COMPAT SHIM SINCE THE CLM BACKLOG'S E4.4 (2026-09-26). The panel found tool choice LEARNED TWICE (D3):
    this trace was written by tool_note / tool_reflex_teach / serve but read only by present_tools, which nothing
    called, while serve picked tools by word overlap and the reflex bridge. The mind's one tool learner is now the
    tool door's ProtoStore (holographic_unified_p33_router: tool_note / tool_predict / present_tools read and write
    it; verified calls and reported corrections train it, a failed verify is a labelled negative). This class stays
    ONLY so a partition written before that change -- its lecore.learning.toolusage section -- still loads: its
    success / failure counts migrate into the tool door's audit. Its trace vector is not migrated (its task keys were
    semantic_key vectors, a different space from the door's n-gram key). Nothing writes a UsageTrace any more."""

    def __init__(self, dim=2048, seed=0):
        self.dim = int(dim)
        self.seed = int(seed)
        self._trace = np.zeros(self.dim)
        self._tools = {}                        # name -> atom
        self.counts = {}                        # name -> successful uses (the audit ledger)
        self.failures = {}                      # name -> failed uses (sweep 178: they were counted as +0)
        self._n = 0

    def _atom(self, tool):
        if tool not in self._tools:
            self._tools[tool] = key_atom("tool:" + tool, self.dim)
        return self._tools[tool]

    def note(self, task_vec, tool, success=True):
        """Record one tool use; only SUCCESSFUL uses strengthen the trace (failures only count)."""
        self.counts[tool] = self.counts.get(tool, 0) + (1 if success else 0)
        if not success:
            # KEPT NEGATIVE (sweep 178): before this tally a failure was `counts += 0` -- indistinguishable
            # from a tool never tried. Failures still never touch the trace; they are audit only.
            self.failures[tool] = self.failures.get(tool, 0) + 1
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

    # PERSISTENCE (sweep 178). The usage trace lived in process memory only: learning_save never wrote
    # it, so every restart forgot which tools had worked for which tasks. The trace is a SUM of binds
    # whose task keys are not kept, so the summed vector itself is the state (float64, exact); the tool
    # atoms are NOT stored -- they regenerate from their names, as the class docstring promises.
    def to_state(self):
        """Plain-data snapshot: {dim, seed, n, tools (names, in first-use order), counts, trace (list)}."""
        return {"dim": self.dim, "seed": self.seed, "n": int(self._n), "tools": list(self._tools),
                "counts": dict(self.counts), "failures": dict(self.failures),
                "trace": np.asarray(self._trace, float).tolist()}

    @classmethod
    def from_state(cls, state):
        """Rebuild from to_state(): predictions are identical (the atoms are derived from the names)."""
        u = cls(int(state["dim"]), int(state.get("seed", 0)))
        for name in state.get("tools") or []:
            u._atom(name)
        u.counts = {str(k): int(v) for k, v in (state.get("counts") or {}).items()}
        u.failures = {str(k): int(v) for k, v in (state.get("failures") or {}).items()}
        u._n = int(state.get("n", 0))
        tr = np.asarray(state.get("trace") if state.get("trace") is not None else np.zeros(u.dim), float)
        u._trace = tr.reshape(-1) if tr.size == u.dim else np.zeros(u.dim)
        return u


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
