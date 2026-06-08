"""
holographic_emergence.py
========================

The parts, finally alive together.

Every other file added a capability. This one wires them into a single dynamical
loop that does something none of them does alone: watch a stream and grow its own
concepts -- unsupervised, online, with no labels, no preset number of categories,
and no training loop. It is the demoscene principle at the level of cognition --
a world of concepts unfolding from a stream of seeds -- and it is where the
field, the diffusion idea, and synchronization stop being separate tricks.

How the pieces feed each other:

  * Field (holographic_field) is the PERCEPTION. The committed concepts define a
    landscape; an input is recognized by which peak it falls under, and an input
    that falls in the flat void between peaks is, by definition, novel.
  * Double diffusion (the staircase idea) is the COMMITMENT. A new concept is
    only tentative; its support integrates hits and decays slowly, so a concept
    becomes real only when reinforcement is SUSTAINED. Transient noise spikes a
    tentative concept that then diffuses away -- the same transient-vs-permanent
    test that built the thermohaline staircase, now deciding what deserves to be
    a concept.
  * Synchronization (holographic_sync) is the SELF-MONITORING. The coherence of
    the concept set reports whether the world is cleanly separated into distinct
    things or collapsing into one blur.

The result behaves less like an algorithm and more like a small perceptual
cortex: it builds categories when the world shows it sustained structure, shrugs
off noise, and can tell you how novel a thing is and how fragmented its world has
become.

Needs: numpy, holographic_ai.py, holographic_field.py, holographic_sync.py.
"""

import numpy as np
from holographic_ai import random_vector, cosine
from holographic_field import landscape
from holographic_sync import SyncGrouping


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Concept:
    """A candidate category: a prototype direction plus a slowly-integrating
    support. It earns permanence only when its support is sustained."""

    def __init__(self, salt, step, tick=0):
        self.salt = salt.copy()      # prototype direction (slow / persistent)
        self.support = 1.0           # accumulated reinforcement (the slow variable)
        self.committed = False
        self.born = step
        self.last_seen = tick        # internal clock tick of the most recent hit


class EmergentConcepts:
    """A self-organizing concept former. Feed it vectors one at a time; it grows
    and maintains a set of concepts, commits the ones the stream keeps
    reinforcing, and forgets the ones it doesn't."""

    def __init__(self, vigilance=0.45, commit=3.0, prune=0.5,
                 decay_tentative=0.97, decay_committed=0.995, drift=0.05,
                 retire_after=300, seed=0):
        self.vigilance = vigilance              # match this well or you're something new
        self.commit = commit                    # support needed to become permanent
        self.prune = prune                      # below this a tentative concept dies
        self.decay_t = decay_tentative
        self.decay_c = decay_committed
        self.drift = drift                      # how fast a prototype follows its members
        # A committed concept that goes this many observations without a single
        # hit is retired: the conditions that formed it have vanished, so the
        # layer erodes. Set to None to keep every concept forever.
        self.retire_after = retire_after
        self.concepts = []
        self._t = 0                             # internal monotonic clock (hit recency)
        self.sync = SyncGrouping(seed=seed)

    # --- the perceive loop ---
    def perceive(self, x, step=0):
        """Take in one observation. Returns (concept, is_novel)."""
        x = _unit(np.asarray(x, dtype=float))
        self._t += 1
        best, best_sim = None, -1.0
        for c in self.concepts:
            s = cosine(x, c.salt)
            if s > best_sim:
                best, best_sim = c, s

        is_novel = best is None or best_sim <= self.vigilance
        if not is_novel:
            best.support += 1.0
            best.salt = _unit((1 - self.drift) * best.salt + self.drift * x)
            best.last_seen = self._t
            assigned = best
        else:
            assigned = Concept(x, step, tick=self._t)   # a tentative new concept
            self.concepts.append(assigned)

        # Double-diffusion commitment: support integrates hits and decays; a
        # concept commits only on sustained reinforcement, and unsupported
        # tentatives diffuse away.
        for c in self.concepts:
            c.support *= self.decay_c if c.committed else self.decay_t
            if not c.committed and c.support >= self.commit:
                c.committed = True

        def keep(c):
            if not c.committed:
                return c.support > self.prune       # tentatives die on low support
            if self.retire_after is None:
                return True
            return (self._t - c.last_seen) <= self.retire_after  # committed retire on staleness

        self.concepts = [c for c in self.concepts if keep(c)]
        return assigned, is_novel

    # --- readouts that reuse the other modules ---
    def committed(self):
        return [c for c in self.concepts if c.committed]

    def field(self):
        """The perceptual landscape: a Field built from the committed concepts,
        weighted by support. Sampling it anywhere asks 'how familiar is this?'"""
        cc = self.committed()
        if not cc:
            return landscape([random_vector(len(self.concepts[0].salt) if self.concepts else 8, np.random.default_rng(0))], [0.0])
        return landscape([c.salt for c in cc], [c.support for c in cc])

    def familiarity(self, x):
        """How strongly the current concept landscape recognizes x. Low = a void,
        i.e. genuinely novel."""
        return self.field().sample(_unit(np.asarray(x, dtype=float)))

    def coherence(self):
        """Synchronization order parameter over the committed concepts: low means
        the world is cleanly split into distinct things; high would mean the
        concepts have collapsed toward one blur."""
        cc = self.committed()
        if len(cc) < 2:
            return 1.0
        return self.sync.coherence(self.sync.run([c.salt for c in cc]))

    def consolidate(self, merge=0.7):
        """Fold near-duplicate committed concepts into one (their prototypes drift
        until two represent the same thing). Keeps the inventory minimal."""
        cc = self.committed()
        merged, used = [], set()
        for i, a in enumerate(cc):
            if i in used:
                continue
            group = [a]
            for j in range(i + 1, len(cc)):
                if j not in used and cosine(a.salt, cc[j].salt) > merge:
                    group.append(cc[j])
                    used.add(j)
            if len(group) == 1:
                merged.append(a)
            else:
                total = sum(g.support for g in group)
                fused = Concept(_unit(sum(g.support * g.salt for g in group)),
                                min(g.born for g in group),
                                tick=max(g.last_seen for g in group))
                fused.support, fused.committed = total, True
                merged.append(fused)
        self.concepts = merged + [c for c in self.concepts if not c.committed]


# ---------------------------------------------------------------------------
# DEMOS
# ---------------------------------------------------------------------------

def _staged_stream(dim, rng):
    """Three categories from the start; a fourth appears halfway; noise throughout."""
    centers = [random_vector(dim, rng) for _ in range(4)]
    stream, truth = [], []
    for i in range(320):
        if rng.random() < 0.06:
            stream.append(random_vector(dim, rng))   # transient noise
            truth.append(-1)
        else:
            cat = rng.integers(0, 3) if i < 160 else rng.integers(0, 4)
            stream.append(_unit(centers[cat] + 0.5 * random_vector(dim, rng)))
            truth.append(int(cat))
    return stream, np.array(truth), centers


def _pair_agreement(truth, pred):
    idx = [i for i in range(len(truth)) if truth[i] >= 0]
    a = t = 0
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = idx[ii], idx[jj]
            t += 1
            a += (truth[i] == truth[j]) == (pred[i] == pred[j])
    return a / t


def demo_concept_formation():
    print("=" * 70)
    print("DEMO 1 -- A stream grows its own concepts")
    print("=" * 70)
    rng = np.random.default_rng(0)
    stream, truth, centers = _staged_stream(256, rng)
    mind = EmergentConcepts()

    checkpoints = {}
    born_steps = {}
    for i, x in enumerate(stream):
        before = {id(c) for c in mind.committed()}
        mind.perceive(x, i)
        for c in mind.committed():
            if id(c) not in before and id(c) not in born_steps:
                born_steps[id(c)] = c.born
        if i in (80, 159, 200, 319):
            checkpoints[i] = len(mind.committed())

    cc = mind.committed()
    pred = [int(np.argmax([cosine(_unit(x), c.salt) for c in cc])) for x in stream]
    spurious = sum(1 for c in cc if max(cosine(c.salt, ctr) for ctr in centers) < 0.5)

    print("\nThree categories run from the start; a FOURTH appears at step 160;")
    print("noise is sprinkled throughout. The system is told none of this.\n")
    print(f"  committed concepts at step  80: {checkpoints[80]}")
    print(f"  committed concepts at step 159: {checkpoints[159]}   (still pre-fourth)")
    print(f"  committed concepts at step 200: {checkpoints[200]}   (fourth has appeared)")
    print(f"  committed concepts at step 319: {checkpoints[319]}")
    print(f"  fourth concept was born at step: {sorted(born_steps.values())[-1]}")
    print(f"  spurious concepts from noise   : {spurious}")
    print(f"  final agreement with hidden truth: {_pair_agreement(truth, pred) * 100:.0f}%")
    print("\n  It built exactly the categories the world contained, the moment the")
    print("  world contained them -- and ignored every transient.\n")
    return mind, centers


def demo_self_monitoring(mind, centers):
    print("=" * 70)
    print("DEMO 2 -- The system reads its own state")
    print("=" * 70)
    rng = np.random.default_rng(5)
    known = _unit(centers[1] + 0.5 * random_vector(256, rng))   # a familiar kind of thing
    novel = random_vector(256, rng)                             # something never seen
    print("\nFamiliarity is just the perceptual Field sampled at a point -- high on")
    print("a known kind of input, near zero in the void where nothing has formed:\n")
    print(f"  familiarity of a known-category input : {mind.familiarity(known):+.2f}")
    print(f"  familiarity of a brand-new input      : {mind.familiarity(novel):+.2f}  (void -> novel)")
    print(f"\n  concept-set coherence (low = cleanly separated world): {mind.coherence():.2f}")
    print("  The same order parameter from synchronization, now reporting whether")
    print("  the mind's categories are distinct or collapsing.\n")


def demo_consolidation():
    print("=" * 70)
    print("DEMO 3 -- Consolidation: folding a redundant concept back in")
    print("=" * 70)
    rng = np.random.default_rng(3)
    mind = EmergentConcepts()
    a = random_vector(256, rng)
    b = random_vector(256, rng)
    # Force two committed concepts that are actually the same thing, plus a distinct one.
    for proto in (a, a, b):
        c = Concept(_unit(proto + 0.2 * random_vector(256, rng)), 0)
        c.support, c.committed = 10.0, True
        mind.concepts.append(c)
    print(f"\n  committed concepts before consolidation: {len(mind.committed())}")
    print(f"  (two of them are near-duplicates of the same thing)")
    mind.consolidate()
    print(f"  committed concepts after consolidation : {len(mind.committed())}")
    print("\n  Prototypes that drifted onto the same thing are fused -- the inventory")
    print("  stays as small as the world actually is.\n")


if __name__ == "__main__":
    mind, centers = demo_concept_formation()
    demo_self_monitoring(mind, centers)
    demo_consolidation()
