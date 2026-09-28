"""holographic_protostore.py -- ONE contrastive update rule for every door that learns from verdicts.

WHY THIS EXISTS (the CLM panel, 2026-09-26; docs/research/evidence/clm_panel_20260926/)
-------------------------------------------------------------------------------------
Contrastive-LM's CLM-8B learns a System One decision by pulling the right candidate toward the state and pushing
every candidate it confused with it -- InfoNCE over a candidate set, two towers, staged hard negatives. leCore
does not copy the towers or the 8B backbone. It takes the CONCEPT and puts it where our substrate already is:
a door keeps one UNIT PROTOTYPE per option (a hypervector-sized row in the door's own feature space) and every
labelled verdict moves those rows with the InfoNCE gradient:

    p   = softmax(cos(q, A_k) / tau)            over the candidate set
    A_k += lr * (t_k - p_k) * q                 for every candidate k, then renormalise the touched rows

t is the target: one-hot on the true option, or -- when the teacher is known to be noisy -- the posterior
t_k ~ p_k * T(k -> observed) (see TeacherNoise). As tau -> 0, p becomes one-hot on the argmax and the rule is
EXACTLY SystemOne's miss-only AdaptHD (pull the truth, push the wrong pick, nothing when right). At finite tau it
also learns from narrow wins and pushes every close rival, which is what measured:

    Banking77 one pass, hashed n-grams d=2048:  InfoNCE 0.826 / 0.822  vs  AdaptHD 0.725 / 0.738 (lr 0.3)
    CLINC150:                                   InfoNCE 0.860          vs  AdaptHD 0.794
    10% wrong labels:                           InfoNCE 0.783          vs  AdaptHD (lr 1) 0.568

ONE CLASS, SEPARATE INSTANCES (panel Q5): every door (SystemOne, the router, the meaning rows, the guard, tool
choice, the verifier) owns its own ProtoStore instance -- its own feature space, codebook, label stream and
DoorCalibrator. Shared calibration was a measured defect (D2), and a write that fixed one key erased a neighbour
at cosine 0.8, so nothing learned here is shared across doors by accident.

What this module deliberately does NOT do:
  * no projection matrix / dense head (panel Q2: per-row prototypes only; the ridge-W probe lost);
  * no candidate-relative softmax as the CONFIDENCE -- it abstains worse (AURC 0.094 at 2 candidates vs 0.078
    for the absolute cosine). p_top is only ever a FEATURE fed to an isotonic calibrator;
  * no synthetic negatives -- a negative is a teacher verdict (the ranked candidate that beat the truth) or an
    explicit reported failure, never a generated sentence.

numpy + stdlib only, deterministic (no hash(), no unseeded RNG), and every learned array round-trips through
state() / from_state() so it lives in the memory partition.
"""
import hashlib
import json
import math

import numpy as np


def _unit(v):
    """Unit-normalise a vector (float64); a zero vector stays zero instead of becoming NaN."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def softmax_over(scores, tau):
    """Stable softmax of scores / tau. tau is floored so a caller passing 0 gets the argmax limit, not a NaN."""
    z = np.asarray(scores, dtype=np.float64) / max(float(tau), 1e-9)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class ProtoStore:
    """Per-option unit prototypes plus the one contrastive update rule (InfoNCE on prototypes).

    labels     : the options, in insertion order (the order is also the tie order: argmax is stable)
    A          : raw accumulators, one row per option (the update adds to these)
    P          : the unit rows actually scored against (renormalised A, touched rows only)
    tau, lr    : the rule's temperature and step; tau 0.05 = SystemOne's score_tau, lr 0.3 (measured)
    topk       : None = the softmax runs over EVERY option; k = only over the k nearest plus the truth (the
                 cheap form for thousands of options, e.g. the router's 3,952 cards)
    key_of     : optional {label: answer_key}; options sharing the truth's key are NEVER pushed (two rows that
                 serve the same answer are not rivals -- the meaning index's akey)
    confusion  : the reverse-direction miner (E1.2): M[y][k] += p_k on each verdict, i.e. how much probability
                 the truth y's questions gave to k. It is CLM's "other tower" without a second tower.

    Wave-2 asks, added by backlog G2 (each bit-identical to what the doors did by hand, pinned in _selftest):
      add_options(labels, vec_lists, keys)   many options in ONE growth of A/P (add_option copied both per call)
      update(q, truth, ..., among=labels)    the rule restricted to a candidate set (the rank door's sub-store copy)
      state(dtype="float64")                 a bit-exact reload: A and P in float64 (the default stays float32 A)
    """

    def __init__(self, dim, tau=0.05, lr=0.3, topk=None, name="door", mine=True):
        self.dim = int(dim)
        self.tau = float(tau)
        self.lr = float(lr)
        self.topk = None if topk is None else int(topk)
        self.name = str(name)
        self.mine = bool(mine)
        self.labels = []
        self._ix = {}
        self.A = np.zeros((0, self.dim), dtype=np.float64)
        self.P = np.zeros((0, self.dim), dtype=np.float64)
        self.count = []                 # labelled verdicts seen per option (the row's maturity)
        self.key_of = {}
        self.confusion = {}             # {truth: {other: accumulated probability}}
        self.n_updates = 0
        self.n_negatives = 0

    # ------------------------------------------------------------------ options
    def __len__(self):
        return len(self.labels)

    def __contains__(self, label):
        return label in self._ix

    def add_option(self, label, vecs=None, key=None):
        """Add an option (or re-seed an existing one). vecs = example encodings: the prototype starts at their
        UNIT MEAN -- exactly SystemOne.fit's accumulator, so the norm starts at 1 and lr has a stated meaning (one
        verdict with lr=1 weighs as much as the whole prior). No vecs = a zero row that the first verdict sets.
        One option at a time grows A and P by one row -- a full copy of both matrices. For many options at once use
        add_options (one copy for the whole batch); the result is the same, bit for bit."""
        return self.add_options([label], [vecs], [key])[0]

    def add_options(self, labels, vec_lists=None, keys=None):
        """Add (or re-seed) MANY options in ONE growth of A and P -> [row index per label, in order].

        WHY (backlog G2, wave-2 ask): add_option grows the matrices with np.vstack, which copies both whole arrays on
        every call -- O(n^2) copying when a door seeds n options one by one. MEASURED: the typed learning guard's build
        (364 options x 2048 dims) spent 0.52 s of 2.9 s in 728 vstacks; the router's 3,952 cards and the rank door's
        candidate sets had each grown a private workaround (state/from_state, hand-stacked sub-stores).
        SAME RESULT AS add_option IN A LOOP, bit for bit (the selftest pins it): new labels get zero rows in first-seen
        order, then each (label, vecs, key) is applied in order exactly as add_option applies it -- so a label
        repeated inside the batch ends with its LAST seeding, as sequential calls would leave it.
        vec_lists / keys: None, or one entry per label (an entry may itself be None: no seeding / no key)."""
        labels = list(labels)
        n = len(labels)
        vec_lists = [None] * n if vec_lists is None else list(vec_lists)
        keys = [None] * n if keys is None else list(keys)
        if len(vec_lists) != n or len(keys) != n:
            raise ValueError("add_options: %d labels but %d vec lists and %d keys" % (n, len(vec_lists), len(keys)))
        # 1) every label not yet present gets a row index, in first-seen order (the insertion order IS the tie order)
        fresh = []
        for label in labels:
            if label not in self._ix:
                self._ix[label] = len(self.labels)
                self.labels.append(label)
                self.count.append(0)
                fresh.append(label)
        # 2) ONE growth of both matrices for the whole batch (was: one vstack per option)
        if fresh:
            pad = np.zeros((len(fresh), self.dim))
            self.A = np.vstack([self.A, pad])
            self.P = np.vstack([self.P, pad])
        # 3) seed and key each option exactly as add_option did, in the caller's order
        out = []
        for label, vecs, key in zip(labels, vec_lists, keys):
            i = self._ix[label]
            if vecs is not None:
                V = np.asarray(vecs, dtype=np.float64).reshape(-1, self.dim)
                if len(V):
                    self.A[i] = _unit(V.mean(0))
                    self.P[i] = self.A[i]
            if key is not None:
                self.key_of[label] = key
            out.append(i)
        return out

    def index(self, label):
        """Row index of a label (KeyError if unknown)."""
        return self._ix[label]

    # ------------------------------------------------------------------ scoring
    def scores(self, q):
        """Cosine of the (unit) query to every prototype, in label order."""
        if not self.labels:
            return np.zeros(0)
        return self.P @ _unit(q)

    def rank(self, q, k=None):
        """[(label, cosine)] best first; ties keep insertion order (stable sort) -- the one stated tie rule."""
        s = self.scores(q)
        order = np.argsort(-s, kind="stable")
        if k is not None:
            order = order[:int(k)]
        return [(self.labels[int(j)], float(s[int(j)])) for j in order]

    def p_top(self, q, tau=0.02, k=None):
        """The softmax weight of the top option at a (sharper) temperature: a FEATURE for a DoorCalibrator.
        tau 0.02 ranked correctness better than the raw margin gap on Banking77 and CLINC (w1). It is never
        served as a probability on its own -- see the module docstring."""
        s = self.scores(q)
        if s.size == 0:
            return 0.0
        if k is not None:
            s = -np.sort(-s)[:int(k)]
        return float(softmax_over(s, tau).max())

    # ------------------------------------------------------------------ the rule
    def _candidates(self, s, truth_i):
        """All options, or the top-k nearest plus the truth."""
        if self.topk is None or self.topk >= len(s):
            return np.arange(len(s))
        cand = np.argsort(-s, kind="stable")[:self.topk]
        if truth_i not in cand:
            cand = np.append(cand, truth_i)
        return cand

    def update(self, q, truth, eps=0.0, m=None, curriculum_k0=None, among=None):
        """One labelled verdict: move every candidate's prototype by lr * (t - p) * q.

        eps : the teacher's estimated flip rate (TeacherNoise.eps_hat). eps > 0 replaces the one-hot target with
              the posterior of the TRUE option given the observed one -- forward correction with a known flip
              model T(k -> observed) = 1-eps if k is the observed option, else eps/m. eps = 0 is bit-identical to
              the plain rule (the target is exactly one-hot).
        m   : how many wrong candidates the teacher could have picked (default: min(8, candidates) - 1).
        curriculum_k0 : E2.2 test arm, OFF by default: per-row temperature tau*(1 + k0/n_truth) and rows with
              fewer than k0 verdicts are never pushed. MEASURED (tools/bench_contrastive.py curriculum, 3 seeds):
              clean labels within seed noise (+0.05 at K=1, -0.29 at K=5, no CI excludes 0) -- an arm, not the
              rule. Under 10% noise at K=5 it gives +1.48 but the noise-corrected target (eps 0.1) gives +1.62 and
              curriculum adds nothing on top; at K=1 the eps target HURTS (-1.16) and curriculum recovers +0.89.
              Note `count` excludes the seed wordings (the panel's probe counted them).
        among : None (default) = the rule runs over the whole store (topk applies). A list of labels = the rule runs
              over ONLY those rows, in THAT order -- the prediction, the softmax, the pushes and the miner all see
              exactly this candidate set, and no other row moves. Unknown labels are skipped, duplicates keep their
              first place, and the truth is always a candidate (appended last when absent). WHY (backlog G2, wave-2
              ask): the rank door ranks per-call candidate sets out of one shared store, and pushing every row the
              store ever held would push candidates of OTHER sets; it copied the set's rows into a temporary store to
              get this. The result is the same as that copy, bit for bit, when the rows are in the same order (the
              selftest pins it) -- without the copy.
        Returns {pred, correct, p_truth, touched}. An unknown truth is added as a new option first."""
        q = _unit(q)
        if truth not in self._ix:
            self.add_option(truth)
        y = self._ix[truth]
        if not np.any(self.A[y]):
            # A brand-new option: its first verdict IS its prototype (no rivals to push against yet).
            self.A[y] = q
            self.P[y] = q
            self.count[y] += 1
            self.n_updates += 1
            return {"pred": truth, "correct": True, "p_truth": 1.0, "touched": 1}
        # rows = the global row index of each local candidate position (None: local == global, the whole store)
        rows = None if among is None else self._among_rows(among, y)
        if rows is None:
            s = self.P @ q
            yl = y
        else:
            s = self.P[rows] @ q                     # the same product a stacked sub-store would compute
            yl = int(np.flatnonzero(rows == y)[0])
        pred = int(np.argmax(s))
        cand = self._candidates(s, yl)
        tau = self.tau
        if curriculum_k0:
            tau = self.tau * (1.0 + float(curriculum_k0) / max(self.count[y], 1))
        p = softmax_over(s[cand], tau)
        yi = int(np.flatnonzero(cand == yl)[0])
        grow = cand if rows is None else rows[cand]  # candidate positions -> global row indices
        if eps > 0:
            mm = m if m is not None else max(1, min(len(cand), 8) - 1)
            t = p * (float(eps) / mm)
            t[yi] = p[yi] * (1.0 - float(eps))
            t = t / t.sum()
        else:
            t = np.zeros_like(p)
            t[yi] = 1.0
        g = t - p
        # Options that serve the SAME answer as the truth are not rivals: never push them.
        ky = self.key_of.get(truth)
        if ky is not None:
            same = np.fromiter((self.key_of.get(self.labels[int(c)]) == ky for c in grow), bool, len(grow))
            same &= (grow != y)
            g[same] = 0.0
        sel = np.abs(g) > 1e-4                       # skip rows the verdict barely touches (cheap, deterministic)
        if curriculum_k0:
            young = np.array([self.count[int(c)] < curriculum_k0 for c in grow])
            sel &= ~young | (grow == y)
        touch = grow[sel]
        if len(touch):
            self.A[touch] += self.lr * g[sel][:, None] * q[None, :]
            nrm = np.linalg.norm(self.A[touch], axis=1, keepdims=True)
            self.P[touch] = self.A[touch] / np.maximum(nrm, 1e-12)
        if self.mine:
            row = self.confusion.setdefault(truth, {})
            for j, c in enumerate(grow):
                if int(c) != y and p[j] > 1e-3:
                    lab = self.labels[int(c)]
                    row[lab] = row.get(lab, 0.0) + float(p[j])
        self.count[y] += 1
        self.n_updates += 1
        pg = pred if rows is None else int(rows[pred])
        return {"pred": self.labels[pg], "correct": pg == y, "p_truth": float(p[yi]),
                "touched": int(len(touch))}

    def _among_rows(self, among, y):
        """update(among=...)'s candidate rows: known labels in the caller's order, first occurrence kept, the truth's
        row y appended when the caller left it out. -> int array of global row indices."""
        seen, rows = set(), []
        for label in among:
            i = self._ix.get(label)
            if i is not None and i not in seen:
                seen.add(i)
                rows.append(i)
        if y not in seen:
            rows.append(y)
        return np.asarray(rows, dtype=np.intp)

    def negative(self, q, label, weight=1.0):
        """A LABELLED NEGATIVE (E2.1): the teacher or an outcome said 'q is NOT label'. Push that one prototype
        away from q by lr * weight, renormalise. This is the single store the three old negative forms (the
        trace's failure field, the meaning index's blocked wordings, the ladder's bad payloads) now feed, so a
        negative finally TRAINS something instead of only vetoing one exact wording."""
        if label not in self._ix:
            return {"pushed": False, "why": "unknown label"}
        i = self._ix[label]
        if not np.any(self.A[i]):
            return {"pushed": False, "why": "empty prototype"}
        self.A[i] -= self.lr * float(weight) * _unit(q)
        self.P[i] = _unit(self.A[i])
        self.n_negatives += 1
        return {"pushed": True, "label": label}

    # ------------------------------------------------------------------ the miner (E1.2)
    def confusions(self, k=10, min_mass=0.0):
        """The mined confusable pairs, most confused first: [(truth, other, mass)]. Pairs mined from verdicts
        (where the probability actually went) -- the panel's replacement for decision-time reverse
        normalisation, which cost -6.0 top-1 points (kept negative)."""
        out = [(y, o, m) for y, row in self.confusion.items() for o, m in row.items() if m >= min_mass]
        out.sort(key=lambda t: (-t[2], t[0], t[1]))
        return out[:int(k)]

    # ------------------------------------------------------------------ persistence
    def state(self, dtype="float32"):
        """JSON-able meta + arrays, the container's section shape (meta, arrays).

        dtype "float32" (the default, unchanged): {"A": float32} -- half the bytes; the unit rows are recomputed on
        load, so a reloaded store ranks like the live one to ~1e-7, not bit for bit.
        dtype "float64" (backlog G2, wave-2 ask): {"A": float64, "P": float64} and meta["dtype"] = "float64" -- a
        BIT-EXACT reload. P is stored too because it cannot be recomputed exactly: a row seeded by add_option has
        P = A (normalised by a dot-product norm), and a row moved by update has P = A / (a reduction norm); the two
        norms can differ in the last bit, so from_state's recomputation would move some rows by an ulp. The tool door
        and the rank / verify stores saved A and P in float64 by hand for exactly this; this is the one spelling."""
        meta = {"dim": self.dim, "tau": self.tau, "lr": self.lr, "topk": self.topk, "name": self.name,
                "mine": self.mine, "labels": list(self.labels), "count": list(self.count),
                "key_of": dict(self.key_of), "confusion": self.confusion,
                "n_updates": self.n_updates, "n_negatives": self.n_negatives}
        if str(dtype) == "float32":
            return meta, {"A": self.A.astype(np.float32)}
        if str(dtype) != "float64":
            raise ValueError("state: dtype must be 'float32' or 'float64', not %r" % (dtype,))
        meta["dtype"] = "float64"
        return meta, {"A": np.array(self.A, dtype=np.float64), "P": np.array(self.P, dtype=np.float64)}

    @classmethod
    def from_state(cls, meta, arrays):
        """Rebuild from state(). With a float32 state the unit rows are recomputed from the accumulators; with a
        float64 state (arrays carry "P") both matrices are restored exactly. The arrays are always COPIED, so the
        store never shares memory with the caller's arrays (a float64 input used to be aliased)."""
        st = cls(meta["dim"], tau=meta["tau"], lr=meta["lr"], topk=meta.get("topk"), name=meta.get("name", "door"),
                 mine=meta.get("mine", True))
        st.labels = list(meta["labels"])
        st._ix = {l: i for i, l in enumerate(st.labels)}
        st.A = np.array(arrays["A"], dtype=np.float64).reshape(len(st.labels), st.dim)
        if arrays.get("P") is not None:
            st.P = np.array(arrays["P"], dtype=np.float64).reshape(len(st.labels), st.dim)
        else:
            nrm = np.linalg.norm(st.A, axis=1, keepdims=True)
            st.P = np.where(nrm > 0, st.A / np.maximum(nrm, 1e-12), 0.0)
        st.count = list(meta.get("count", [0] * len(st.labels)))
        st.key_of = dict(meta.get("key_of", {}))
        st.confusion = {k: dict(v) for k, v in meta.get("confusion", {}).items()}
        st.n_updates = int(meta.get("n_updates", 0))
        st.n_negatives = int(meta.get("n_negatives", 0))
        return st

    def digest(self):
        """sha256 of the learned state -- a determinism receipt (two runs, one digest)."""
        meta, arr = self.state()
        h = hashlib.sha256(json.dumps(meta, sort_keys=True).encode())
        h.update(np.ascontiguousarray(arr["A"]).tobytes())
        return h.hexdigest()


class DoorCalibrator:
    """P(this door's answer is correct | its score) -- ONE class, one instance PER DOOR (E0.6 / E0.7).

    Every door's record now carries p_correct from its own calibrator: HIGH MEANS CONFIDENT at every door. (The
    catalog router used to return a null p-value -- low meant confident -- while its reflex path returned
    1 - error: the same key, opposite meanings. That p-value now travels as p_null.)

    Fed with (score, correct) pairs from that door's outcomes only. Isotonic (PAV, SystemOne's calibrator),
    clipped to what n labels can testify to, refit on the last `window` pairs once `min_count` pairs with BOTH
    outcomes exist. Until then p_correct is None -- an uncalibrated door says so instead of inventing a number."""

    def __init__(self, door, min_count=8, window=512):
        self.door = str(door)
        self.min_count = int(min_count)
        self.window = int(window)
        self.pairs = []
        self._fit = None

    def observe(self, score, correct, refit=True):
        """Add one (score, correct) label from THIS door's outcome; refit when enough labels exist."""
        if score is None or not np.isfinite(float(score)):
            return False
        self.pairs.append((float(score), 1.0 if correct else 0.0))
        self.pairs = self.pairs[-self.window:]
        if refit:
            self.refit()
        return True

    def refit(self):
        """Refit the isotonic map from the current window (no-op below min_count or with one outcome only)."""
        # Imported here, not at module top: UnifiedMind's part 29 imports this module when `import lecore` runs, and a
        # module-top import dragged holographic_systemone and holographic_relations into lecore's REQUIRED import
        # footprint (tests/test_deptrace.py caps it -- measured 38 non-part modules with it, the cap is < 36).
        from holographic.agents_and_reasoning.holographic_systemone import IsotonicCalibrator
        ys = [c for _, c in self.pairs]
        if len(self.pairs) >= self.min_count and 0.0 < sum(ys) < len(ys):
            try:
                self._fit = IsotonicCalibrator([s for s, _ in self.pairs], ys)
            except ValueError:
                pass
        return self._fit is not None

    def calibrated(self):
        """True once a fit exists."""
        return self._fit is not None

    def p_correct(self, score):
        """Calibrated P(correct) for a score, or None while uncalibrated."""
        if self._fit is None or score is None:
            return None
        return float(self._fit.predict(float(score)))

    def state(self):
        """JSON-able: the pairs are the state; the fit is re-derived on load (deterministic)."""
        return {"door": self.door, "min_count": self.min_count, "window": self.window,
                "pairs": [[s, c] for s, c in self.pairs]}

    @classmethod
    def from_state(cls, st):
        """Rebuild from state() and refit."""
        c = cls(st.get("door", "door"), min_count=st.get("min_count", 8), window=st.get("window", 512))
        c.pairs = [(float(s), float(y)) for s, y in st.get("pairs", [])]
        c.refit()
        return c


class TeacherNoise:
    """How often is the teacher (a model end, a human, a verifier) simply wrong? Estimated from the teacher
    itself, not from our rows (E2.3).

    The old serve bar compared memory's agreement with the teacher on its most confident fifth (the 'ceiling'),
    which read 0.94-0.99 for a PERFECT teacher: it cannot tell a noisy teacher from confusable rows. Here we
    re-ask a small sample of questions with the candidates PERMUTED (so position bias cannot fake agreement) and
    count how often the two answers agree. If each ask flips to a uniformly random wrong candidate with rate e,
    P(agree) = (1-e)^2 + e^2/m, a quadratic we solve for e in [0, 0.5].

    Measured (w1, reproduced by tools/bench_contrastive.py): eps_hat 0.000 for a perfect teacher,
    0.084-0.1065 at 10% planted noise, 0.197-0.210 at 20%.
    With eps_hat the gate serves on P(correct|g) = (P(agree|g) - eps/m) / (1 - eps - eps/m): at 10% noise the
    static probe served 20.1% at 0.47% wrong, against 0.3% served before. CAVEAT kept loud: re-ask independence
    holds for the scripted stand-in; for a real model it is a hypothesis to measure."""

    def __init__(self, m=7.0, min_pairs=20):
        self.m = float(m)
        self.min_pairs = int(min_pairs)
        self.pairs = []            # [(first answer, second answer)] as strings

    def observe(self, first, second):
        """One re-ask: the teacher's two answers to the same question (candidates permuted in between)."""
        self.pairs.append((str(first), str(second)))
        self.pairs = self.pairs[-1024:]

    def agreement(self):
        """Fraction of re-asks where the two answers matched (None with no re-asks)."""
        if not self.pairs:
            return None
        return sum(1 for a, b in self.pairs if a == b) / float(len(self.pairs))

    def eps_hat(self):
        """The estimated flip rate, or 0.0 until min_pairs re-asks exist (then the gate is exactly today's)."""
        if len(self.pairs) < self.min_pairs:
            return 0.0
        a2 = self.agreement()
        A_, B_, C_ = 1.0 + 1.0 / self.m, -2.0, 1.0 - a2
        disc = max(0.0, B_ * B_ - 4.0 * A_ * C_)
        return float(max(0.0, min(0.5, (-B_ - np.sqrt(disc)) / (2.0 * A_))))

    def corrected(self, p_agree, eps=None):
        """P(the answer is correct) from P(the teacher would agree): removes the agreement a wrong teacher
        produces by chance. eps = 0 returns p_agree unchanged."""
        e = self.eps_hat() if eps is None else float(eps)
        if p_agree is None:
            return None
        if e <= 0:
            return float(p_agree)
        return float(np.clip((p_agree - e / self.m) / max(1e-9, 1.0 - e - e / self.m), 0.0, 1.0))

    def eps_bound(self, z=1.0):
        """A CONSERVATIVE flip rate: the lowest eps the re-asks support at one-sided confidence z -- the Wilson UPPER
        bound on the agreement, solved through the same quadratic as eps_hat. z <= 0 returns eps_hat itself; 0.0
        below min_pairs re-asks or for a teacher that always agreed.

        WHY a bound, and why HERE (backlog G2): the correction is steep exactly where it matters -- at 10% noise a
        calibrated P(agree) of 0.8557 already means P(correct) 0.95 -- so an eps_hat read 0.03 too high (the sampling
        error of ~100 re-asks at a 1% rate) serves bins that are only ~91% right. MEASURED by the meaning door
        (bench_meaning online, CLINC150, noisy:0.1): the point estimate met the bar on 1 of 3 seeds, z = 0.5 on 0 of
        2, z = 1 with a 3% re-ask rate on every seed (<= 2.0% wrong, <= 2.0% out-of-scope; evidence:
        docs/research/evidence/bench_meaning_online.json). It was written inside MeaningIndex.teacher_eps; it lives
        on TeacherNoise now so every door that corrects for a noisy teacher uses ONE definition (MeaningIndex calls
        this; its numbers are unchanged, bit for bit)."""
        e = float(self.eps_hat())
        z = float(z)
        if e <= 0 or z <= 0:
            return e
        n, a = float(len(self.pairs)), float(self.agreement())
        den = 1.0 + z * z / n
        a_hi = min(1.0, ((a + z * z / (2 * n)) + z * math.sqrt(a * (1 - a) / n + z * z / (4 * n * n))) / den)
        A_, B_, C_ = 1.0 + 1.0 / self.m, -2.0, 1.0 - a_hi        # (1 + 1/m) e^2 - 2 e + (1 - a) = 0, as eps_hat
        disc = max(0.0, B_ * B_ - 4.0 * A_ * C_)
        return float(max(0.0, min(0.5, (-B_ - math.sqrt(disc)) / (2.0 * A_))))

    @staticmethod
    def hold_back(rank_of_pick, eps):
        """Per-verdict quarantine: a single verdict that picks a candidate at rank >= 5 (1-based) is held back
        from training when eps > 0 -- noise lands uniformly over ranks while genuine corrections concentrate on
        the runner-up. MEASURED (tools/bench_contrastive.py teacher): CLINC150 catches 54.7% of planted errors
        and wrongly holds back 7.4% of real corrections; Banking77 61.4% / 14.8% (its confusable neighbours put
        more genuine corrections deep in the list) -- so it only runs when the teacher is measurably noisy."""
        return bool(eps > 0 and rank_of_pick is not None and rank_of_pick >= 5)

    def state(self):
        """JSON-able state."""
        return {"m": self.m, "min_pairs": self.min_pairs, "pairs": [list(p) for p in self.pairs]}

    @classmethod
    def from_state(cls, st):
        """Rebuild from state()."""
        t = cls(m=st.get("m", 7.0), min_pairs=st.get("min_pairs", 20))
        t.pairs = [(str(a), str(b)) for a, b in st.get("pairs", [])]
        return t


def negative_quality(sim_to_query, sim_to_truth_rows=(), sim_to_pick=None, flag_below=None, margin_scale=0.2):
    """E0.3 (after ECI, arXiv 2603.20990): is a negative TRUSTWORTHY?  A negative is the verdict 'q is not A,
    it is B' (A = the candidate memory served, B = the teacher's pick).

    MEASURED (tools/bench_contrastive.py negq, CLINC150, planted teacher errors vs genuine corrections, 16
    configurations tried):
      * KEPT NEGATIVE -- the first draft (quality = harmonic mean of hardness and safety, flag on low quality)
        flagged planted errors at AUROC 0.33 / 0.32: WORSE than chance, because hardness rewards exactly the
        teacher's errors (a wrong 'not A' lands on a question A owns, so A is very close). The quality number is
        still returned for RANKING useful negatives, never as the flag.
      * The flag that works is the raw TEACHER MARGIN  safety = cos(q, B) - cos(q, A): a genuine correction points
        at a B the question is also close to; a planted error points at a far B. AUROC 0.936 (meaning index) /
        0.939 (hashed n-grams), 0.41% / 0.43% false flags with a perfect teacher, catching 44% / 38% of planted
        errors at that operating point; near-miss errors 0.886 / 0.889.
      * The flag threshold depends on the feature space (-0.38 vs -0.26 for 1% false flags), so each door
        calibrates its own `flag_below`; without one, `suspect` is None (no opinion) rather than a made-up cut.
      * Without a pick (a 'new' verdict has none), the draft's safety alone -- 1 - best cosine between q and A's
        own confirmed wordings -- scores AUROC 0.827 / 0.811, and flags below `margin_scale`.

    sim_to_query      : cos(q, prototype of A)            (the hardness)
    sim_to_truth_rows : cosines between q and A's confirmed wordings   (used only when there is no pick)
    sim_to_pick       : cos(q, prototype of the teacher's pick B)
    Returns {hardness, safety, quality, suspect}."""
    h = float(np.clip(sim_to_query, 0.0, 1.0))
    if sim_to_pick is not None:
        safety = float(sim_to_pick) - float(sim_to_query)
        suspect = None if flag_below is None else bool(safety < float(flag_below))
    else:
        best = float(max(sim_to_truth_rows)) if len(sim_to_truth_rows) else 0.0
        safety = float(np.clip(1.0 - best, 0.0, 1.0))
        suspect = bool(safety < margin_scale)
    s_ = float(np.clip(safety, 0.0, 1.0))
    qual = 0.0 if (h + s_) == 0 else 2.0 * h * s_ / (h + s_)
    return {"hardness": h, "safety": safety, "quality": qual, "suspect": suspect}


def _selftest():
    """The rule's contracts, on synthetic data (cheap; the real-data numbers live in tools/bench_contrastive.py)."""
    rng = np.random.default_rng(0)
    d, C = 256, 6
    centers = rng.standard_normal((C, d))
    st = ProtoStore(d, tau=0.05, lr=0.3)
    for c in range(C):
        st.add_option("o%d" % c, [centers[c] + 0.8 * rng.standard_normal(d) for _ in range(3)])
    # 1) the rule learns: accuracy on fresh draws rises after a stream of verdicts
    test = [(c, centers[c] + 1.2 * rng.standard_normal(d)) for c in range(C) for _ in range(20)]
    acc0 = np.mean([st.rank(x, 1)[0][0] == "o%d" % c for c, x in test])
    for _ in range(300):
        c = int(rng.integers(C))
        st.update(centers[c] + 1.2 * rng.standard_normal(d), "o%d" % c)
    acc1 = np.mean([st.rank(x, 1)[0][0] == "o%d" % c for c, x in test])
    assert acc1 >= acc0, (acc0, acc1)
    # 2) tau -> 0 is exactly AdaptHD: nothing moves on a correct verdict, two rows on a miss
    a = ProtoStore(d, tau=1e-9, lr=0.5)
    a.add_option("x", [centers[0]]); a.add_option("y", [centers[1]]); a.add_option("z", [centers[2]])
    before = a.A.copy()
    r = a.update(centers[0], "x")
    assert r["correct"] and np.allclose(a.A, before)
    r = a.update(centers[0], "y")
    assert (not r["correct"]) and r["touched"] == 2
    # 3) eps = 0 is bit-identical to the plain rule
    b1, b2 = ProtoStore(d), ProtoStore(d)
    for s_ in (b1, b2):
        for c in range(3):
            s_.add_option("o%d" % c, [centers[c]])
    v = centers[1] + 0.5 * rng.standard_normal(d)
    b1.update(v, "o1"); b2.update(v, "o1", eps=0.0)
    assert b1.digest() == b2.digest()
    # 4) same-answer rows are never pushed
    k = ProtoStore(d)
    k.add_option("r1", [centers[0]], key="A"); k.add_option("r2", [centers[0] + 0.1 * rng.standard_normal(d)], key="A")
    k.add_option("r3", [centers[3]], key="B")
    r2_before = k.A[1].copy()
    k.update(centers[0], "r1")
    assert np.allclose(k.A[1], r2_before)
    # 5) state round trip
    m, arr = st.state()
    st2 = ProtoStore.from_state(json.loads(json.dumps(m)), {"A": arr["A"]})
    assert st2.labels == st.labels and np.allclose(st2.P, st.P, atol=1e-6)
    # 6) calibrator: high score -> high p, uncalibrated -> None
    cal = DoorCalibrator("t", min_count=8)
    assert cal.p_correct(0.5) is None
    for i in range(20):
        cal.observe(0.1 + i * 0.01, False)
        cal.observe(0.6 + i * 0.01, True)
    assert cal.p_correct(0.8) > cal.p_correct(0.1)
    # 7) teacher noise: a perfect teacher reads 0, a 20%-noise stand-in reads near 0.2
    tn = TeacherNoise(m=7)
    for i in range(200):
        tn.observe("a", "a")
    assert tn.eps_hat() == 0.0
    r_ = np.random.default_rng(3)
    tn2 = TeacherNoise(m=7)
    for i in range(1000):
        pick = lambda: "right" if r_.random() >= 0.2 else "w%d" % int(r_.integers(7))
        tn2.observe(pick(), pick())
    assert 0.14 < tn2.eps_hat() < 0.26, tn2.eps_hat()
    # 8) add_options == add_option in a loop, bit for bit (new, existing, repeated and unseeded labels, keys)
    seq, bat = ProtoStore(d), ProtoStore(d)
    seq.add_option("o0", [centers[0]])
    bat.add_option("o0", [centers[0]])
    batch = [("o1", [centers[1], centers[2]], "K"), ("o0", [centers[3]], None), ("o2", None, "K"),
             ("o1", [centers[4]], None), ("o3", [], None)]
    for lab, vs, ky in batch:
        seq.add_option(lab, vs, ky)
    ix = bat.add_options([b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch])
    assert ix == [1, 0, 2, 1, 3] and seq.labels == bat.labels and seq.key_of == bat.key_of
    assert seq.A.tobytes() == bat.A.tobytes() and seq.P.tobytes() == bat.P.tobytes() and seq.digest() == bat.digest()
    # 9) update(among=) == the rank door's copy-into-a-sub-store, bit for bit, and no other row moves
    big = ProtoStore(d, name="big", mine=False)
    big.add_options(["o%d" % c for c in range(C)], [[centers[c]] for c in range(C)])
    for _ in range(20):
        c = int(rng.integers(C))
        big.update(centers[c] + 1.2 * rng.standard_normal(d), "o%d" % c)
    among = ["o4", "o1", "o3"]
    sub = ProtoStore(d, tau=big.tau, lr=big.lr, name="sub", mine=False)
    sub.labels = list(among)
    sub._ix = {l_: j for j, l_ in enumerate(among)}
    sub.A = np.stack([big.A[big.index(l_)] for l_ in among])
    sub.P = np.stack([big.P[big.index(l_)] for l_ in among])
    sub.count = [big.count[big.index(l_)] for l_ in among]
    others = big.A[[big.index(l_) for l_ in big.labels if l_ not in among]].copy()
    v = centers[1] + 0.9 * rng.standard_normal(d)
    r_sub, r_big = sub.update(v, "o1"), big.update(v, "o1", among=among)
    assert r_sub == r_big, (r_sub, r_big)
    for l_ in among:
        assert big.A[big.index(l_)].tobytes() == sub.A[sub.index(l_)].tobytes()
        assert big.P[big.index(l_)].tobytes() == sub.P[sub.index(l_)].tobytes()
    assert big.A[[big.index(l_) for l_ in big.labels if l_ not in among]].tobytes() == others.tobytes()
    # 10) state(dtype="float64") reloads BIT-EXACT (A and P); the default float32 state is unchanged
    m64, a64 = big.state(dtype="float64")
    back = ProtoStore.from_state(json.loads(json.dumps(m64)), a64)
    assert back.A.tobytes() == big.A.tobytes() and back.P.tobytes() == big.P.tobytes()
    assert back.rank(v) == big.rank(v)
    a64["A"][0, 0] += 1.0
    assert back.A[0, 0] != a64["A"][0, 0]            # from_state copies: no memory shared with the caller's arrays
    m32, a32 = big.state()
    assert "dtype" not in m32 and a32["A"].dtype == np.float32 and set(a32) == {"A"}
    # 11) eps_bound: z <= 0 is eps_hat; z > 0 sits below it, tightens with more re-asks, and a perfect teacher is 0
    assert tn2.eps_bound(0) == tn2.eps_hat() and 0 < tn2.eps_bound(1.0) < tn2.eps_hat()
    assert tn.eps_bound(1.0) == 0.0
    return "ok"


if __name__ == "__main__":
    print(_selftest())
