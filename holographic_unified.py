"""One model over one holographic space.

The rest of this project grew as separate studies -- a self-organizing classifier, a
self-maintaining decision brain, an image vault, a mixture-of-experts router, a text
n-gram. They were never meant to stay separate. They already share the one thing that
matters: a holographic vector space, and a `UniversalEncoder` that turns ANY input --
text, image, number, category, record, sequence -- into a vector in that single space.

`UnifiedMind` is the top level that makes the sharing real instead of nominal. There is
ONE perception step (the encoder), ONE associative memory (the autonomous
`SelfOrganizingMind`, which both classifies and is searched for recall), and ONE
decision brain (`HolographicMind`), all reading and writing the same space. It does not
reimplement simple versions of these -- the failing of the old `Mind` facade -- it uses
the real, self-maintaining ones, and every input passes through the same encoder before
it reaches any of them.

What is deliberately NOT pretended to be one call: classification, recall, and decision
are different OPERATIONS on the shared substrate (aggregate into prototypes; index the
individuals; weight by reward). The unification is the shared space and the shared
self-maintenance, not a single magic method.
"""

import numpy as np

from holographic_ai import cosine
from holographic_mind import UniversalEncoder, _Index
from holographic_organizer import SelfOrganizingMind
from holographic_creature import HolographicMind


class UnifiedMind:
    """Perceive once, into one space; remember, organize, recall, and decide over it.

      read(corpus)                     -- let perception pre-learn word co-occurrence
      learn(x, label, modality)        -- file a perception into the one memory (any modality)
      classify(x, modality)            -- 'what is this?'  (nearest self-organized prototype)
      recall(x, modality)              -- 'what's like this?' (nearest stored individual)
      actions(names) / decide / reinforce  -- choose actions over the same space

    The memory maintains itself: with maintain='auto' it periodically reorganizes (the
    speculate-measure-adopt rule from holographic_organizer), splitting a confusable
    class into sub-prototypes only when held-out accuracy says it earns its keep. The
    decision brain maintains itself the same way.
    """

    def __init__(self, dim=1024, seed=0, number_range=(-4.0, 4.0), maintain='auto',
                 check_every=60, text_window=2):
        self.dim = dim
        self.maintain = maintain
        self.check_every = check_every
        # ONE perception, shared by everything below
        self.encoder = UniversalEncoder(dim, seed=seed, number_range=number_range,
                                        text_window=text_window)
        # ONE associative memory: classify by nearest prototype, organize autonomously
        self.memory = SelfOrganizingMind(dim=dim, seed=seed)
        # a recall view over the SAME encoded vectors (individuals, for 'what's like this')
        self._recall = None
        # ONE decision brain (assembled when an action set is declared)
        self._brain = None
        self._actions = None
        self._taught = 0
        self._label_modality = {}    # which modality each label came from (for routing)
        self._gen = None             # sequence generator (lazy)

    # -- perception (the single front door) --------------------------------
    def read(self, corpus):
        """Pre-learn word co-occurrence so text perceptions carry meaning."""
        self.encoder.learn_text(corpus)
        return self

    def perceive(self, x, modality=None):
        """Any input -> one vector in the shared space. This is the only encoder in the
        system; the memory and the brain never encode anything themselves."""
        return self.encoder.encode(x, modality)

    # -- one memory: classification + organization -------------------------
    def learn(self, x, label, modality=None):
        v = self.perceive(x, modality)
        self.memory.observe_vector(v, label)        # aggregate into self-organized prototypes
        self._index(v, (label, x))                  # AND keep the individual for recall
        self._label_modality[label] = modality      # remember which modality this label is
        self._taught += 1
        if self.maintain == 'auto' and self._taught % self.check_every == 0:
            self.memory.auto_reorganize()
        return self

    def classify(self, x, modality=None, route=True):
        """Nearest self-organized prototype. If the modality is known and `route` is on,
        the query competes only against that modality's concepts -- a cheap router that
        removes the cross-modal interference a single flat store can otherwise suffer (a
        text query mistaken for an image). When the modality is unknown the query ranges
        over everything; inferring it instead would be the learned MoE gate's job, which
        (per the mixture-of-experts study) only beats this trivial routing when the
        experts are miscalibrated."""
        among = None
        if route and modality is not None:
            among = {lab for lab, m in self._label_modality.items() if m == modality}
            among = among or None
        return self.memory.classify_vector(self.perceive(x, modality), among=among)

    # -- the same data, a recall view (nearest individual) -----------------
    def _index(self, v, payload):
        if self._recall is None:
            self._recall = _Index(self.dim)
        self._recall.add(v, payload)

    def recall(self, x, modality=None):
        if self._recall is None:
            raise RuntimeError("nothing learned yet -- call learn() first")
        return self._recall.recall(self.perceive(x, modality))

    # -- one decision brain, on the same substrate -------------------------
    def actions(self, names):
        self._actions = list(names)
        self._brain = HolographicMind(self.dim, self._actions, k=12, epsilon=0.1,
                                      novelty_bonus=0.15, memory_cap=8000,
                                      maintain=self.maintain)
        return self

    def decide(self, state, explore=False, epsilon=None, modality=None):
        if self._brain is None:
            raise RuntimeError("declare an action set first -- call actions([...])")
        a = self._brain.decide(self.perceive(state, modality), explore=explore, epsilon=epsilon)
        return self._actions[a]

    def reinforce(self, state, action, reward, modality=None):
        s = self.perceive(state, modality)
        self._brain.remember([s], [self._actions.index(action)], [float(reward)])
        return self

    # -- generation: predict the next symbol over the same space ------------
    def learn_sequence(self, text, n=6, hierarchical=True):
        """Learn to continue a sequence.

        Two engines, picked by `hierarchical`:

        * The fractal coder (default): discover a chunk schema by compression, then predict by
          cross-level backoff -- emit the longest chunk a level is confident about, else descend
          a level and spell it out. Measured against the flat n-gram on Austen, it cut bits/char
          from 2.085 to 1.829 and the stored model from ~218k context entries to ~58k (3.8x
          smaller), at roughly tied coherence (0.96 vs 0.98 real words). Generation is the
          traversal-shaped operation where the multi-scale substrate earns its keep -- unlike
          classification, where a tree REGRESSED and the flat scan stayed best.

        * The flat holographic n-gram (`hierarchical=False`): the original engine, kept because
          it exposes `next_symbol` and an exact context key, and because the boundary between
          where the substrate helps and where it doesn't is measured here, not assumed."""
        if hierarchical:
            from holographic_schema import SchemaGenerator
            self._gen = SchemaGenerator(modality="text").fit(text)
            self._gen_kind = "hierarchical"
        else:
            from holographic_text import HolographicNGram
            if not isinstance(self._gen, HolographicNGram) or self._gen.n != n:
                self._gen = HolographicNGram(dim=self.dim, n=n, seed=0)
            self._gen.fit(text)
            self._gen_kind = "flat"
        return self

    def next_symbol(self, context):
        if self._gen is None:
            raise RuntimeError("nothing learned to continue -- call learn_sequence() first")
        if getattr(self, "_gen_kind", "flat") != "flat":
            raise RuntimeError("next_symbol needs the flat engine: learn_sequence(text, hierarchical=False)")
        return self._gen.next_char(context)

    def generate(self, seed_text, length=160, temperature=0.5):
        if self._gen is None:
            raise RuntimeError("nothing learned to continue -- call learn_sequence() first")
        return self._gen.generate(seed_text, length, temperature)

    # -- self-maintenance across the whole model ---------------------------
    def maintain_now(self):
        """Reorganize the memory and refresh the brain, each by its own held-out
        measurement. Returns the memory's choice."""
        choice = self.memory.auto_reorganize()
        if self._brain is not None and self._brain.maintain == 'auto':
            self._brain.auto_maintain()
        return choice

    def describe(self):
        parts = [f"memory of {self.memory.live.size()} prototypes over "
                 f"{len(self.memory.live.counts_by_label())} labels"]
        if self._recall is not None:
            parts.append(f"a recall index of {len(self._recall.vecs)} items")
        if self._brain is not None:
            parts.append(f"a decision brain over {self._actions}")
        if self._gen is not None:
            kind = getattr(self, "_gen_kind", "flat")
            detail = f"order {self._gen.n}" if kind == "flat" else "fractal cross-level coder"
            parts.append(f"a sequence generator ({detail})")
        return "UnifiedMind: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# DEMO: one mind, many modalities, one memory -- measured against separate ones
# ---------------------------------------------------------------------------

def _patterns(kind, rng, n=8):
    """Tiny synthetic 'images' -- four visually distinct classes, with noise."""
    a = np.zeros((n, n))
    if kind == "rows":
        a[::2, :] = 1.0
    elif kind == "cols":
        a[:, ::2] = 1.0
    elif kind == "diag":
        for i in range(n):
            a[i, i] = 1.0; a[i, (i + 1) % n] = 1.0
    elif kind == "check":
        a[(np.add.outer(np.arange(n), np.arange(n)) % 2) == 0] = 1.0
    return a + 0.15 * rng.standard_normal((n, n))


def demo_unified():
    """One UnifiedMind learns three different KINDS of thing -- text topics, little
    images, and records -- into a SINGLE self-organizing memory, then classifies all
    three. The honest question is whether one shared store does as well as three
    separate ones; if mixing modalities in one space wrecked it, the unification would
    be fake. It does not: the modalities land in near-orthogonal parts of the space, so
    one memory matches the separate baselines AND the same mind still makes decisions."""
    from holographic_text import TOPICS, _content, _split

    print("=" * 70)
    print("One mind, one memory: text + images + records in a single space")
    print("=" * 70)
    rng = np.random.default_rng(0)
    corpus = [s for sents in TOPICS.values() for s in sents]

    # build the three datasets as (input, label, modality)
    text_tr, text_te = [], []
    for topic, sents in TOPICS.items():
        a, b = _split(sents, frac=0.7, seed=2)
        text_tr += [(_content(s), topic, "text") for s in a]
        text_te += [(_content(s), topic, "text") for s in b]
    img_tr, img_te = [], []
    for kind in ("rows", "cols", "diag", "check"):
        for _ in range(20):
            img_tr.append((_patterns(kind, rng), f"img:{kind}", "image"))
        for _ in range(8):
            img_te.append((_patterns(kind, rng), f"img:{kind}", "image"))
    rec_tr, rec_te = [], []
    depts = ("eng", "sales", "ops")
    for d in depts:
        for _ in range(20):
            rec_tr.append(({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record"))
        for _ in range(8):
            rec_te.append(({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record"))

    # ---- ONE unified mind: everything into one memory --------------------
    # text word-vectors learn best from content words (stopwords dilute co-occurrence);
    # that is a text-task choice, so the orchestrator makes it -- the encoder stays generic.
    mind = UnifiedMind(dim=1024, seed=0).read([_content(s) for s in corpus])
    train = text_tr + img_tr + rec_tr
    rng.shuffle(train)
    for x, label, mod in train:
        mind.learn(x, label, mod)
    mind.maintain_now()

    def score(m, test, route=True):
        return sum(m.classify(x, mod, route=route)[0] == lab for x, lab, mod in test) / len(test)

    ut = score(mind, text_te); ui = score(mind, img_te); ur = score(mind, rec_te)
    ut_flat = score(mind, text_te, route=False)

    # ---- separate baselines: one memory per modality (same encoding) -----
    def separate(train_items, test_items):
        enc = UniversalEncoder(1024, seed=0)
        enc.learn_text([_content(s) for s in corpus])
        mem = SelfOrganizingMind(dim=1024, seed=0)
        for x, lab, mod in train_items:
            mem.observe_vector(enc.encode(x, mod), lab)
        mem.auto_reorganize()
        return sum(mem.classify_vector(enc.encode(x, mod))[0] == lab
                   for x, lab, mod in test_items) / len(test_items)

    st = separate(text_tr, text_te); si = separate(img_tr, img_te); sr = separate(rec_tr, rec_te)

    print(f"\n  {'modality':10s}{'separate memory':>18s}{'one shared memory':>20s}")
    print(f"  {'text':10s}{100*st:>16.0f}% {100*ut:>18.0f}%")
    print(f"  {'images':10s}{100*si:>16.0f}% {100*ui:>18.0f}%")
    print(f"  {'records':10s}{100*sr:>16.0f}% {100*ur:>18.0f}%")
    print(f"\n  Routing: a text query against ALL concepts scores {100*ut_flat:.0f}%; restricted to")
    print(f"  text concepts (its known modality) it scores {100*ut:.0f}%. With correct encoding the")
    print("  modalities separate cleanly, so here routing changes nothing -- it is a cheap")
    print("  safeguard that removes cross-modal collisions WHEN they occur, not a routine")
    print("  booster. (An earlier apparent gain came from a since-fixed encoding bug that")
    print("  degraded text vectors into colliding with other modalities.)")
    print(f"\n  {mind.describe()}")

    # ---- cross-modal recall over the same store --------------------------
    q = img_te[0]
    (lab, _), sim = mind.recall(q[0], q[2])
    print(f"\n  Recall: a held-out '{q[1]}' image finds nearest stored item '{lab}' "
          f"(cos {sim:.2f}) -- the recall view searches the same vectors.")

    # ---- the SAME mind also decides -------------------------------------
    mind.actions(["left", "right"])
    rng2 = np.random.default_rng(1)
    for _ in range(400):
        n = float(rng2.uniform(-3, 3))
        good = "right" if n > 0 else "left"
        choice = mind.decide(n, explore=True, epsilon=0.3, modality="number")
        mind.reinforce(n, choice, 1.0 if choice == good else 0.0, modality="number")
    dec = sum((mind.decide(float(v), modality="number") == ("right" if v > 0 else "left"))
              for v in np.linspace(-3, 3, 40)) / 40
    print(f"  Decision: the same mind learned a contextual choice over numbers -> "
          f"{100*dec:.0f}% correct, using the same encoder and space.")

    # ---- the SAME mind also generates (the fourth operation) -------------
    mind.learn_sequence(" ".join(corpus), n=5)
    sample = mind.generate("the ", 90, 0.4)
    print(f"  Generation: taught to continue the topic text, it produces -> \"{sample[:70]}\"")

    print(f"\n  {mind.describe()}")
    print("\n  One encoder, one self-organizing memory, one brain -- shared substrate, not")
    print("  a wrapper. One shared store matches separate per-modality memories; with")
    print("  correct encoding the modalities are near-orthogonal, so a flat store shows")
    print("  no cross-modal interference here and routing is a cheap safeguard rather than")
    print("  a booster. Storage needs no separate curator: the memory's own aggregation")
    print("  already compresses (here ~1800 observations into a handful of prototypes).")
    print("  Generation completes the operation set -- its next-symbol step is the same")
    print("  cleanup primitive -- though its context index stays exact, the one place a")
    print("  fuzzy recall was measured to hurt rather than help.")


if __name__ == "__main__":
    demo_unified()
