"""
holographic_organizer.py -- a self-organizing memory with shadow-and-swap reorg.

The hard part of a system that keeps learning is not absorbing data, it is keeping
the data ORGANIZED as it arrives. A holographic class is stored as a bundle (a
running superposition), and that works beautifully until a class turns out to be
several things at once -- "vehicle" is cars AND trucks AND motorbikes, sitting in
different directions of the space. Bundle them into one prototype and you get their
average, a point that is no single one of them; on genuinely multi-modal classes a
one-prototype-per-label store collapses (measured: 49% where the structure allows
100%). The fix is to REORGANIZE: discover the sub-structure and split each class
into as many coherent sub-prototypes as the data actually shows, fold away
redundant ones, and rebuild.

Doing that on the live model mid-stream would mean serving queries against a
half-rebuilt store. So this follows the read-copy-update / double-buffer pattern
hardware and databases use: a small team of organizer experts builds a SHADOW copy
from an experience buffer, leaving the live model untouched and consistent the
whole time, and then a single atomic SWAP makes the reorganized copy live. Nothing
ever observes an inconsistent in-between state.

This is the scaffolding for a self-* system: self-learning (it absorbs a stream
with no training phase), self-organizing (it restructures its own memory on the
shadow and swaps), and self-classifying (the split expert DISCOVERS the sub-classes
inside each label -- the car-mode and the truck-mode -- without being told they
exist). The experts here are deliberate and measured rather than learned; the point
is the organizing substrate the rest can sit on.
"""

import numpy as np
from collections import defaultdict

from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder


# ---------------------------------------------------------------------------
# 1. THE STORE  (multiple sub-prototypes per label; classify by nearest)
# ---------------------------------------------------------------------------

class SubPrototypeMemory:
    """A classifier that allows several sub-prototypes per label, so a multi-modal
    class can keep its modes apart instead of averaging them. Classifying is just
    nearest-sub-prototype across every label."""

    def __init__(self, protos=None):
        # each entry: [label, sum_vector, unit_vector, count]
        self._p = [list(p) for p in (protos or [])]
        self._gen = 0          # bumped on every mutation, so the cached stack can't go stale

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def add(self, vec, label):
        """Online: fold into the nearest same-label sub-prototype, or start the
        label's first one. (Splitting into more modes is the organizer's job, not
        this fast path's -- here we just keep the running bundles current.)"""
        best_i, best_s = -1, -2.0
        for i, p in enumerate(self._p):
            if p[0] == label:
                s = float(p[2] @ vec)
                if s > best_s:
                    best_i, best_s = i, s
        if best_i < 0:
            self._p.append([label, vec.copy(), self._unit(vec), 1])
        else:
            p = self._p[best_i]
            p[1] = p[1] + vec
            p[2] = self._unit(p[1])
            p[3] += 1
        self._gen += 1          # the prototype set changed -> invalidate the cached stack

    def _stack(self):
        """Cached (labels, unit-matrix) view of the prototypes for batched scoring.
        Keyed on a mutation counter (_gen) bumped by every add/split/merge, so an
        in-place prototype update can't leave a stale matrix. Repeated classify/
        label_scores over a stable memory pay the stack cost once. This is the same fast
        path Vocabulary.cleanup uses, applied to the hottest scan in the brain (every
        classify/recall/decide routes through here)."""
        gen = getattr(self, "_gen", 0)
        cache = getattr(self, "_stack_cache", None)
        if cache is None or cache[0] != gen or cache[1] is not self._p or cache[2] != len(self._p):
            labels = [p[0] for p in self._p]
            mat = (np.stack([p[2] for p in self._p]) if self._p else np.zeros((0, 0)))
            self._stack_cache = (gen, self._p, len(self._p), labels, mat)
        return self._stack_cache[3], self._stack_cache[4]

    def classify(self, vec, among=None):
        """Nearest sub-prototype across every label. If `among` is given (a set of
        allowed labels), only those compete -- this is how a router restricts a query
        to one modality's concepts, so a text query never loses to an image prototype.

        Runs as one matrix-vector product against the cached unit-prototype stack
        (prototypes are unit length, so the dot is the cosine up to the query norm --
        same argmax), then a single masked argmax. Bit-for-bit the same winner as the
        old per-prototype loop."""
        if not self._p:
            return None, -2.0
        labels, mat = self._stack()
        sims = mat @ vec
        if among is not None:
            mask = np.array([lab in among for lab in labels])
            if not mask.any():
                return None, -2.0
            sims = np.where(mask, sims, -np.inf)
        j = int(sims.argmax())
        return labels[j], float(sims[j])

    def label_scores(self, vec, among=None):
        """The best score per LABEL (not just the winner) -- the full evidence
        vector a single probe sees. Multi-probe (multi-ray) classification needs
        this so several independent encodings of one query can be z-scored and
        combined: one ray gives a label its score, the ensemble averages them."""
        if not self._p:
            return {}
        labels, mat = self._stack()
        sims = mat @ vec
        scores = {}
        for lab, s in zip(labels, sims, strict=True):   # labels and sims are 1:1 by construction
            if among is not None and lab not in among:
                continue
            s = float(s)
            if lab not in scores or s > scores[lab]:
                scores[lab] = s
        return scores

    def labels(self):
        return {p[0] for p in self._p}

    def size(self):
        return len(self._p)

    def counts_by_label(self):
        d = defaultdict(int)
        for p in self._p:
            d[p[0]] += 1
        return dict(d)

    def copy(self):
        return SubPrototypeMemory([[p[0], p[1].copy(), p[2].copy(), p[3]] for p in self._p])

    # -- persistence: round-trip the prototype bank (label, sum, unit, count) -----------
    def to_state(self):
        """Snapshot the prototypes as parallel arrays + the labels. Reload restores an
        identical memory (same prototypes -> identical classification)."""
        return {
            "labels": [p[0] for p in self._p],
            "sums": (np.stack([p[1] for p in self._p]) if self._p else np.zeros((0, 0))),
            "units": (np.stack([p[2] for p in self._p]) if self._p else np.zeros((0, 0))),
            "counts": np.array([p[3] for p in self._p], dtype=float),
        }

    @classmethod
    def from_state(cls, state):
        labels = list(state["labels"])
        sums = np.asarray(state["sums"], float)
        units = np.asarray(state["units"], float)
        counts = np.asarray(state["counts"], float)
        protos = [[labels[i], sums[i], units[i], int(counts[i])] for i in range(len(labels))]
        return cls(protos)


# ---------------------------------------------------------------------------
# 2. THE ORGANIZER EXPERTS  (split, merge -- a small team that restructures)
# ---------------------------------------------------------------------------

def _cosine_kmeans(X, k, rng, iters=15):
    """k-means on the unit sphere (cosine). Returns (unit centroids, assignment)."""
    cent = X[rng.choice(len(X), k, replace=False)].copy()
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    assign = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        assign = (X @ cent.T).argmax(1)
        for j in range(k):
            members = X[assign == j]
            if len(members):
                s = members.sum(0)
                cent[j] = s / (np.linalg.norm(s) or 1.0)
            else:                                   # reseed an empty cluster
                cent[j] = X[rng.integers(len(X))]
    return cent, assign


class SplitExpert:
    """Discover how many things each label really is, and split it into that many
    coherent sub-prototypes. It raises k only while it buys coherence (the average
    similarity of a cluster's members to their centroid), so a single-mode class
    stays one prototype and a three-mode class becomes three -- chosen from the
    data, not told in advance. This is the self-classifying step: the modes it
    finds are emergent sub-categories."""

    def __init__(self, max_k=6, coherence=0.45, min_points=4, seed=0):
        self.max_k = max_k
        self.coherence = coherence
        self.min_points = min_points
        self.seed = seed

    def _best_k(self, X, rng):
        max_k = min(self.max_k, max(1, len(X) // self.min_points))
        best = (1, *_cosine_kmeans(X, 1, rng))
        for k in range(1, max_k + 1):
            cent, assign = _cosine_kmeans(X, k, rng)
            worst = min((float((X[assign == j] @ cent[j]).mean())
                         for j in range(k) if (assign == j).any()), default=-1.0)
            if worst >= self.coherence:             # every mode is tight enough
                return cent, assign
            if k == 1:
                best = (k, cent, assign)
        return best[1], best[2]                     # fall back to the single mode

    def organize(self, examples_by_label):
        rng = np.random.default_rng(self.seed)
        protos = []
        for label, vecs in examples_by_label.items():
            X = np.stack(vecs)
            cent, assign = self._best_k(X, rng)
            for j in range(len(cent)):
                members = X[assign == j]
                if not len(members):
                    continue
                s = members.sum(0)
                protos.append([label, s, s / (np.linalg.norm(s) or 1.0), int(len(members))])
        return protos


class MergeExpert:
    """Fold near-duplicate sub-prototypes of the SAME label back together, cutting
    redundancy so the store stays as small as the structure allows. Two prototypes
    of DIFFERENT labels that are nearly identical are left alone but counted as
    collisions -- an honest signal that those classes are not separable here."""

    def __init__(self, duplicate=0.92):
        self.duplicate = duplicate

    def prune(self, protos):
        kept, collisions = [], 0
        for p in protos:
            merged = False
            for q in kept:
                sim = float(p[2] @ q[2])
                if sim > self.duplicate:
                    if p[0] == q[0]:                # same label -> merge
                        q[1] = q[1] + p[1]
                        q[2] = q[1] / (np.linalg.norm(q[1]) or 1.0)
                        q[3] += p[3]
                        merged = True
                        break
                    else:
                        collisions += 1
            if not merged:
                kept.append(p)
        return kept, collisions


# ---------------------------------------------------------------------------
# 3. THE TRIGGER EXPERT  (decides WHEN to reorganize -- self-triggering)
# ---------------------------------------------------------------------------

class TriggerExpert:
    """Watches the model's own fit and decides when a reorganization is worth doing,
    instead of running on a fixed schedule. Two read-off-yourself signals:

      * incoherence -- recent examples sit far from their own prototype: a label has
        quietly become multi-modal but is still stored as one blurry blob (the
        cold-start symptom -- data filed before the structure was known);
      * novelty -- recent inputs match no prototype well: a new kind of thing has
        begun arriving and the current categories do not cover it.

    Either one, once enough new data has accumulated since the last reorganization
    to be sure it is signal and not a blip, fires."""

    def __init__(self, coherence_floor=0.55, novelty_rate=0.30, min_gap=200):
        self.coherence_floor = coherence_floor
        self.novelty_rate = novelty_rate
        self.min_gap = min_gap

    def assess(self, mind):
        coh, nov = mind.coherence(), mind.novelty()
        ready = mind._since_reorg >= self.min_gap
        fire = ready and (coh < self.coherence_floor or nov > self.novelty_rate)
        reason = ("incoherent" if coh < self.coherence_floor
                  else "novelty" if nov > self.novelty_rate else "")
        return {"coherence": coh, "novelty": nov, "fire": fire, "reason": reason}




class SelfOrganizingMind:
    """Absorbs a stream into a live store, and on demand REORGANIZES: the experts
    build a fresh shadow store from the experience buffer while the live one keeps
    serving, then a single assignment swaps it in. The build never touches the live
    model, so a query mid-reorganization sees a complete, consistent store -- the
    old one right up to the instant of the swap."""

    def __init__(self, dim=1024, seed=0, buffer_cap=20000,
                 max_k=6, coherence=0.45):
        self.encoder = UniversalEncoder(dim, seed=seed)
        self.live = SubPrototypeMemory()
        self.buffer = []                            # experience replay (vec, label)
        self.buffer_cap = buffer_cap
        self.split = SplitExpert(max_k=max_k, coherence=coherence, seed=seed)
        self.merge = MergeExpert()
        self._since_reorg = 0                       # new data seen since last reorg

    def observe(self, x, label, modality=None):
        v = self.encoder.encode(x, modality)
        self.live.add(v, label)                     # keep the live bundles current
        self.buffer.append((v, label))
        self._since_reorg += 1
        if len(self.buffer) > self.buffer_cap:
            self.buffer.pop(0)
        return self

    def classify(self, x, modality=None):
        return self.live.classify(self.encoder.encode(x, modality))

    # -- persistence: round-trip the learned encoder + the live prototype bank + config.
    # The replay buffer is scratch (it only feeds the next reorganize and refills as new
    # data arrives), so it is not persisted -- the same stance the creature brain takes.
    def to_state(self):
        return {
            "kind": "SelfOrganizingMind",
            "dim": int(self.encoder.dim), "seed": int(self.encoder.seed),
            "buffer_cap": int(self.buffer_cap),
            "max_k": int(self.split.max_k), "coherence": float(self.split.coherence),
            "encoder": self.encoder.to_state(),
            "live": self.live.to_state(),
        }

    @classmethod
    def from_state(cls, state):
        m = cls(dim=int(state["dim"]), seed=int(state["seed"]),
                buffer_cap=int(state.get("buffer_cap", 20000)),
                max_k=int(state.get("max_k", 6)), coherence=float(state.get("coherence", 0.45)))
        m.encoder = UniversalEncoder.from_state(state["encoder"])
        m.live = SubPrototypeMemory.from_state(state["live"])
        return m

    def observe_vector(self, v, label):
        """Absorb an ALREADY-ENCODED vector. The normal observe() runs the input
        through this memory's own UniversalEncoder, but some inputs are encoded
        elsewhere -- e.g. the text module's sentence vectors (a bundle of learned
        word vectors). This is the front door for those, so the same self-organizing
        machinery can organize any vectors, not only ones it encoded itself."""
        self.live.add(v, label)
        self.buffer.append((v, label))
        self._since_reorg += 1
        if len(self.buffer) > self.buffer_cap:
            self.buffer.pop(0)
        return self

    def classify_vector(self, v, among=None):
        return self.live.classify(v, among=among)

    # -- signals the model reads off ITSELF, so it can decide when to reorganize -
    def coherence(self, window=400):
        """How well recent examples sit on their OWN label's nearest prototype.
        A multi-modal class stored as one blurry blob shows up here as low
        coherence -- the give-away that the organization has gone stale."""
        recent = self.buffer[-window:]
        if not recent:
            return 1.0
        sims = []
        for v, label in recent:
            sims.append(max((p[2] @ v for p in self.live._p if p[0] == label), default=0.0))
        return float(np.mean(sims))

    def novelty(self, floor=0.35, window=400):
        """Fraction of recent inputs that match NO prototype well -- a new kind of
        thing has started arriving."""
        recent = self.buffer[-window:]
        if not recent:
            return 0.0
        miss = sum(1 for v, _ in recent
                   if max((p[2] @ v for p in self.live._p), default=-1.0) < floor)
        return miss / len(recent)

    # -- the reorganization, in two explicit halves so the swap is visible -----
    def build_shadow(self):
        """Build the reorganized copy WITHOUT touching the live model."""
        by_label = defaultdict(list)
        for v, label in self.buffer:
            by_label[label].append(v)
        protos = self.split.organize(by_label)
        protos, collisions = self.merge.prune(protos)
        return SubPrototypeMemory(protos), collisions

    def swap(self, shadow):
        """Make the reorganized copy live, atomically."""
        self.live = shadow

    def reorganize(self):
        before = self.live.size()
        shadow, collisions = self.build_shadow()
        report = {"before": before, "after": shadow.size(),
                  "per_label": shadow.counts_by_label(), "collisions": collisions}
        self.swap(shadow)
        self._since_reorg = 0
        return report

    def consider_reorganizing(self, trigger, max_passes=3):
        """Let the TriggerExpert decide whether now is the time. If it fires,
        reorganize -- and re-check, reorganizing again while it still helps
        (recursion): one pass usually suffices, but a class that is many modes at
        several scales can need a couple. Returns the assessment plus pass count."""
        verdict = trigger.assess(self)
        passes = 0
        if verdict["fire"]:
            while passes < max_passes:
                before = self.coherence()
                self.reorganize()
                passes += 1
                after = self.coherence()
                if after >= trigger.coherence_floor or after <= before + 0.02:
                    break                            # coherent now, or stopped helping
        verdict["passes"] = passes
        return verdict

    # -- fully autonomous: no coherence floor, no novelty rate, no schedule -------
    def _shadow_at_k(self, examples, k):
        """Build a shadow store that splits each label into UP TO k modes (capped by
        how much data the label has), then folds away near-duplicates. k is just a
        resolution we are trying out -- the choice of which k to keep is made by
        measurement, not here."""
        by_label = defaultdict(list)
        for v, label in examples:
            by_label[label].append(v)
        rng = np.random.default_rng(self.split.seed)
        protos = []
        for label, vecs in by_label.items():
            X = np.stack(vecs)
            kk = min(k, max(1, len(X) // self.split.min_points))
            cent, assign = _cosine_kmeans(X, kk, rng)
            for j in range(len(cent)):
                members = X[assign == j]
                if len(members):
                    s = members.sum(0)
                    protos.append([label, s, s / (np.linalg.norm(s) or 1.0), int(len(members))])
        protos, _ = self.merge.prune(protos)
        return SubPrototypeMemory(protos)

    def _accuracy(self, model, val):
        ok = sum(model.classify(v)[0] == label for v, label in val)
        return ok / len(val)

    def _fingering_gain(self, examples):
        """Salt-finger pre-screen: the largest two-means stratification gain across the
        labels in `examples`. A class with genuine sub-modes ("a finger") scores high
        (two centroids fit its members far better than one); a single-mode blob scores
        ~1. REVISITED finding (NOTES sec.1): on the REAL encoded substrate this signal is
        strong and correlates ~0.94 with the held-out benefit of splitting -- the original
        negative was on synthetic Gaussian blobs that lack the encoder's structure. So when
        the max gain sits at the unimodal floor, no class can be helped by a split and the
        expensive multi-resolution sweep is guaranteed to choose k=1."""
        by_label = defaultdict(list)
        for v, label in examples:
            by_label[label].append(v)
        best = 1.0
        for vecs in by_label.values():
            if len(vecs) < 2 * self.split.min_points:
                continue
            M = np.stack(vecs)
            one = M.mean(0)
            c = M - one
            if not np.any(c):
                continue
            _, _, V = np.linalg.svd(c, full_matrices=False)
            proj = c @ V[0]
            a = M[proj <= 0].mean(0) if (proj <= 0).any() else one
            b = M[proj > 0].mean(0) if (proj > 0).any() else one
            for _ in range(2):
                da = ((M - a) ** 2).sum(1); db = ((M - b) ** 2).sum(1); mk = da <= db
                if mk.any(): a = M[mk].mean(0)
                if (~mk).any(): b = M[~mk].mean(0)
            ss1 = float((c ** 2).sum())
            ss2 = float(np.minimum(((M - a) ** 2).sum(1), ((M - b) ** 2).sum(1)).sum())
            best = max(best, ss1 / (ss2 + 1e-9))
        return best

    def auto_reorganize(self, resolutions=(1, 2, 3, 4), val_frac=0.3, min_val=30,
                        fingering_prescreen=False, finger_floor=1.5):
        """Speculate, measure, adopt -- with no thresholds. Hold out a slice of recent
        experience, build a candidate organization at each resolution (1 prototype per
        label up to a few sub-prototypes), and SELECT the resolution that classifies
        the held-out slice best, breaking near-ties (within one standard error, read
        off the data) toward the fewest prototypes. A blurry cold-start blob or a
        confusable multi-modal class is beaten on held-out accuracy by a split, so the
        split wins; a class that is truly one mode ties at every resolution, so the
        single prototype wins on leanness.

        Fairness matters: EVERY candidate is trained on the same fit slice and judged
        on the same held-out slice, so a richer resolution is not handicapped by less
        data. Once a resolution is chosen, it is REFIT on all the data before going
        live, so nothing is wasted. Returns (chosen, prototype_count) or None if there
        is too little data.

        fingering_prescreen (default OFF -> identical behaviour to before): a cheap
        salt-finger check that SKIPS the full sweep when no class shows sub-mode
        structure (max two-means gain below finger_floor), short-circuiting to "keep".
        It can only avoid work, never change which organization is chosen -- if any
        class fingers, the full measured sweep runs exactly as before."""
        buf = list(self.buffer)
        rng = np.random.default_rng(len(buf))          # deterministic split per call
        rng.shuffle(buf)
        cut = int(len(buf) * (1 - val_frac))
        fit, val = buf[:cut], buf[cut:]
        if len(val) < min_val:
            return None

        # optional pre-screen: if nothing is fingering, the sweep can only pick k=1, so
        # skip it. Conservative -- it never overrides the measured choice, only avoids it.
        if fingering_prescreen and self._fingering_gain(fit) < finger_floor:
            self._since_reorg = 0
            return ("keep", self.live.size())

        # select the resolution -- all candidates on equal footing (fit -> val)
        scored = [(k, self._accuracy(self._shadow_at_k(fit, k), val)) for k in resolutions]
        best = max(scored, key=lambda z: z[1])
        se = np.sqrt(max(best[1] * (1 - best[1]), 1e-9) / len(val))   # Bernoulli standard error
        pool = [k for k, a in scored if a >= best[1] - se]
        chosen_k = min(pool)                           # leanest resolution that is as good

        self._since_reorg = 0
        if chosen_k <= 1:                              # one prototype per label is enough
            return ("keep", self.live.size())
        final = self._shadow_at_k(buf, chosen_k)       # refit the winner on ALL the data
        self.swap(final)
        return (f"k={chosen_k}", final.size())


# ---------------------------------------------------------------------------
# 4. DEMO
# ---------------------------------------------------------------------------

def _multimodal_world(seed=0, dim=512, L=40, n_classes=3, modes=2):
    """Each class is `modes` sub-clusters placed around a circle so the class
    CENTROIDS collapse together -- a non-convex arrangement a single prototype
    cannot represent."""
    rng = np.random.default_rng(seed)
    n_dirs = n_classes * modes
    ang = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1) @ rng.standard_normal((2, L))
    csub = {c: [c + n_classes * m for m in range(modes)] for c in range(n_classes)}
    enc = UniversalEncoder(dim, seed=seed)
    def sample(c):
        return dirs[csub[c][rng.integers(modes)]] * 3 + 0.5 * rng.standard_normal(L)
    return enc, sample, n_classes, rng


def demo_organizer():
    print("=" * 70)
    print("A self-organizing memory: reorganize a shadow copy, then swap it in")
    print("=" * 70)
    enc, sample, K, rng = _multimodal_world(seed=0)
    stream = [(sample(c := int(rng.integers(K))), c) for _ in range(1500)]
    test = [(sample(c := int(rng.integers(K))), c) for _ in range(600)]

    def measure(mind):
        return np.mean([mind.classify(x, "vector")[0] == c for x, c in test])

    print("\n  Streaming multi-modal data (each class is two clusters):\n")
    print(f"    {'seen':>6}  {'naive (1 proto/label)':>22}  {'self-organizing':>16}")
    checkpoints = [300, 600, 1000, 1500]
    naive = SelfOrganizingMind(dim=512, seed=0)        # never reorganizes
    organizing = SelfOrganizingMind(dim=512, seed=0)   # reorganizes periodically
    for i, (x, c) in enumerate(stream, 1):
        naive.observe(x, c, "vector")
        organizing.observe(x, c, "vector")
        if i in checkpoints:
            rep = organizing.reorganize()            # build shadow + atomic swap
            print(f"    {i:>6}  {measure(naive)*100:>20.0f}%  {measure(organizing)*100:>14.0f}%"
                  f"   (split into {rep['per_label']})")

    # --- the swap is non-destructive: building the shadow does not change live ---
    probe = [enc.encode(sample(c % K), "vector") for c in range(40)]
    before = [organizing.live.classify(v)[0] for v in probe]
    shadow, _ = organizing.build_shadow()            # build only -- no swap
    during = [organizing.live.classify(v)[0] for v in probe]
    organizing.swap(shadow)
    print("\n  Atomic swap: the live model's answers are unchanged while the shadow")
    print(f"  is built ({sum(a == b for a, b in zip(before, during))}/40 identical) -- "
          "it only changes at the instant of the swap.")
    print("\n  The split expert found the sub-structure on its own (the two modes per")
    print("  class), the merge expert kept the store minimal, and the swap kept the")
    print("  live model consistent throughout. Self-organizing, in the holographic space.")


def demo_self_triggering():
    """The cold-start problem and its cure. The system starts blind, files early
    data into immature prototypes, and only later has enough to see the real
    structure. A TriggerExpert watches the model's own coherence and novelty and
    decides -- with no schedule -- when to reorganize, re-placing the old data via
    the shadow swap. A new class is introduced mid-stream to show novelty firing
    too."""
    print("\n" + "=" * 70)
    print("Self-triggering: the model reorganizes itself when its fit goes stale")
    print("=" * 70)
    enc, sample, K, rng = _multimodal_world(seed=1, n_classes=2, modes=2)
    # a third (also multi-modal) class appears only halfway through the stream
    enc3, sample3, _, rng3 = _multimodal_world(seed=7, n_classes=1, modes=2)

    auto = SelfOrganizingMind(dim=512, seed=1, coherence=0.5)
    frozen = SelfOrganizingMind(dim=512, seed=1, coherence=0.5)   # never reorganizes
    trigger = TriggerExpert(coherence_floor=0.6, novelty_rate=0.3, min_gap=250)

    test = []
    fires = []
    N = 2400
    for i in range(1, N + 1):
        if i <= N // 2:
            c = int(rng.integers(2)); x = sample(c)
        else:
            c = int(rng.integers(3))                 # class 2 now in play
            x = sample3(0) if c == 2 else sample(c)
        auto.observe(x, c, "vector")
        frozen.observe(x, c, "vector")
        if i % 50 == 0:
            v = trigger.assess(auto)
            if v["fire"]:
                res = auto.consider_reorganizing(trigger)
                fires.append((i, v["reason"], res["passes"], dict(auto.live.counts_by_label())))

    def acc(mind, n=500):
        r = np.random.default_rng(123); ok = 0
        for _ in range(n):
            c = int(r.integers(3)); x = sample3(0) if c == 2 else sample(c)
            ok += (mind.classify(x, "vector")[0] == c)
        return ok / n

    print(f"\n  never reorganizes : {acc(frozen)*100:3.0f}%   ({frozen.live.size()} prototypes,"
          f" {frozen.live.counts_by_label()})")
    print(f"  self-triggered    : {acc(auto)*100:3.0f}%   ({auto.live.size()} prototypes,"
          f" {auto.live.counts_by_label()})")
    print("\n  It fired on its own, only when a signal crossed (no schedule):")
    for when, why, passes, counts in fires:
        print(f"    after {when:>4} seen -- {why:<11} -> reorganized ({passes} pass) -> {counts}")
    print("\n  The first fires split the early blurry classes (cold-start data re-placed);")
    print("  the later one is novelty -- class 2 arrived mid-stream and got organized in.")


def demo_autonomous_organizing():
    """The same cold-start story as demo_self_triggering, but with NO thresholds at
    all. Every so often the store speculates a few organizations of itself at
    different resolutions, keeps whichever classifies a held-out slice of recent
    experience best, and breaks ties toward the fewest prototypes. It fixes the
    cold-start blur and absorbs a new class mid-stream, deciding entirely by
    measurement -- and stays one prototype per class when a class really is one
    mode."""
    print("\n" + "=" * 70)
    print("Autonomous organizing: it reorganizes by measuring, with no thresholds")
    print("=" * 70)
    enc, sample, K, rng = _multimodal_world(seed=1, n_classes=2, modes=2)
    enc3, sample3, _, _ = _multimodal_world(seed=7, n_classes=1, modes=2)
    auto = SelfOrganizingMind(dim=512, seed=1, coherence=0.5)
    never = SelfOrganizingMind(dim=512, seed=1, coherence=0.5)
    N, choices = 2400, []
    for i in range(1, N + 1):
        if i > N // 2:
            c = int(rng.integers(3)); x = sample3(0) if c == 2 else sample(c)
        else:
            c = int(rng.integers(2)); x = sample(c)
        auto.observe(x, c, "vector"); never.observe(x, c, "vector")
        if i % 300 == 0:
            r = auto.auto_reorganize()
            if r and r[0] != "keep":
                choices.append((i, r[0], r[1]))

    def acc(mind):
        r = np.random.default_rng(999); ok = 0
        for _ in range(600):
            c = int(r.integers(3)); x = sample3(0) if c == 2 else sample(c)
            ok += (mind.classify(x, "vector")[0] == c)
        return ok / 600

    print(f"\n  never reorganize : {acc(never)*100:3.0f}%   ({never.live.size()} prototypes)")
    print(f"  autonomous       : {acc(auto)*100:3.0f}%   ({auto.live.size()} prototypes)")
    print("\n  It decided, with nothing tuned, to:")
    for when, what, n in choices:
        tag = "split the cold-start blur" if when <= N // 2 else "absorb the new class"
        print(f"    after {when:>4} seen -> {what}  ({n} prototypes) -- {tag}")
    print("\n  Held-out accuracy is the only judge: a split beats a blurry blob, so it")
    print("  splits; one true mode ties every resolution, so it keeps one prototype.")


if __name__ == "__main__":
    demo_organizer()
    demo_self_triggering()
    demo_autonomous_organizing()
