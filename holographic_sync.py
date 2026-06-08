"""
holographic_sync.py
===================

Grouping by synchronization -- oscillators that fall into step.

This is the demoscene thread made literal (everything is an oscillator; the
structure is what locks together) and a current ML idea at once. Artificial
Kuramoto Oscillatory Neurons (Miyato et al., ICLR 2025) abstract a neuron as a
small unit vector rotating on a sphere and bind neurons by synchronizing them --
"distributed, continuous clustering" -- getting unsupervised grouping,
robustness, and reasoning. They report that SMALL oscillators bind best, which
is why each oscillator here lives in a low-dimensional space (N ~ 16), not the
full representation width.

The move that makes it ours: the substrate is already unit vectors with a
natural coupling -- VSA similarity. Give each item a little oscillator, couple
them by how similar the items are (similar attract, dissimilar repel), and run
the Kuramoto model. Items pull into alignment with their kind and away from the
rest, and groups emerge as separately synchronized directions -- no clustering
algorithm, no preset number of clusters. The count falls out of the dynamics.

It is the self-organization the field/diffusion work was circling, with a
principled mechanism: coupling is a field over pairs, the run is coupled
relaxation (cousin to double diffusion), and a locked cluster is a binding.

Needs: numpy and holographic_ai.py beside it.
"""

import numpy as np
from holographic_ai import random_vector, cosine


class SyncGrouping:
    """Couple vectors as low-dimensional oscillators by their similarity and let
    them synchronize. group() returns an emergent labelling; coherence() reports
    how unified the whole set became.

    Each item's oscillator is a unit vector in R^osc_dim. Coupling A_ij is the
    items' similarity minus a baseline, so similar items attract (align) and
    dissimilar items repel (spread to different directions)."""

    def __init__(self, osc_dim=16, strength=1.0, steps=200, dt=0.1,
                 repulsion=0.3, merge=0.6, seed=0):
        self.osc_dim = osc_dim       # small, per AKOrN -- big oscillators stop binding
        self.strength = strength     # coupling gain K
        self.steps = steps
        self.dt = dt
        self.repulsion = repulsion    # baseline subtracted from similarity -> dissimilar repel
        self.merge = merge            # oscillators closer than this (cosine) are one group
        self.seed = seed

    def _coupling(self, vectors):
        n = len(vectors)
        S = np.array([[cosine(vectors[i], vectors[j]) for j in range(n)]
                      for i in range(n)])
        A = S - self.repulsion
        np.fill_diagonal(A, 0.0)     # an oscillator does not couple to itself
        return A

    def run(self, vectors, record_every=0):
        """Evolve the oscillators under the (vector) Kuramoto model. Each step
        nudges every oscillator toward the ones it's attracted to and away from
        the ones it's repelled by, then renormalizes onto the sphere."""
        coupling = self._coupling(vectors)
        rng = np.random.default_rng(self.seed)
        X = rng.normal(size=(len(vectors), self.osc_dim))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        snapshots = []
        for t in range(self.steps):
            X = X + self.dt * self.strength * (coupling @ X)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
            if record_every and t % record_every == 0:
                snapshots.append(X.copy())
        return (X, snapshots) if record_every else X

    @staticmethod
    def coherence(X):
        """Order parameter: length of the mean oscillator. ~1 when everything
        locked into one direction (a unified whole), low when the set split into
        balanced groups."""
        return float(np.linalg.norm(np.mean(X, axis=0)))

    def read_groups(self, X):
        """Greedy cosine clustering of the locked oscillators. The number of
        groups is discovered, not supplied."""
        labels = np.full(len(X), -1, dtype=int)
        reps = []
        for i in range(len(X)):
            sims = [cosine(X[i], r) for r in reps]
            if sims and max(sims) > self.merge:
                labels[i] = int(np.argmax(sims))
            else:
                reps.append(X[i].copy())
                labels[i] = len(reps) - 1
        return labels

    def group(self, vectors):
        return self.read_groups(self.run(vectors))


# ---------------------------------------------------------------------------
# DEMOS
# ---------------------------------------------------------------------------

def _make_groups(n_groups, per_group, dim, noise, rng):
    centers = [random_vector(dim, rng) for _ in range(n_groups)]
    vectors, truth = [], []
    for g, c in enumerate(centers):
        for _ in range(per_group):
            v = c + noise * random_vector(dim, rng)
            vectors.append(v / np.linalg.norm(v))
            truth.append(g)
    return vectors, np.array(truth)


def _pair_agreement(truth, pred):
    n = len(truth)
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            agree += (truth[i] == truth[j]) == (pred[i] == pred[j])
    return agree / total


def _within_group_coherence(X, truth):
    return float(np.mean([np.linalg.norm(np.mean(X[truth == g], axis=0))
                          for g in set(truth)]))


def demo_emergent_grouping():
    print("=" * 70)
    print("DEMO 1 -- Grouping that emerges from synchronization")
    print("=" * 70)
    rng = np.random.default_rng(302)
    vectors, truth = _make_groups(3, 5, 256, 0.5, rng)   # 3 hidden groups, count unknown
    sync = SyncGrouping(seed=2)
    pred = sync.group(vectors)
    print("\n15 items drawn from 3 hidden groups (nothing is told to the algorithm")
    print("about how many groups exist). Coupled by similarity and synchronized:\n")
    print(f"  true groups      : {list(map(int, truth))}")
    print(f"  emerged groups   : {list(map(int, pred))}")
    print(f"  groups discovered: {len(set(pred))}   (true: {len(set(truth))})")
    print(f"  pairwise agreement with truth: {_pair_agreement(truth, pred) * 100:.0f}%")
    print("\n  The number of clusters was never specified -- it fell out of which")
    print("  oscillators chose to fall into step.\n")


def demo_watch_locking():
    print("=" * 70)
    print("DEMO 2 -- Watch the oscillators lock")
    print("=" * 70)
    rng = np.random.default_rng(303)
    vectors, truth = _make_groups(3, 5, 256, 0.5, rng)
    sync = SyncGrouping(seed=3)
    _, snaps = sync.run(vectors, record_every=20)
    print("\nWithin-group lock (how tightly each true group has aligned) climbing")
    print("from random toward fully locked as the dynamics run:\n")
    for i, X in enumerate(snaps):
        c = _within_group_coherence(X, truth)
        print(f"  step {i * 20:3d}: lock {c:.2f}  {'#' * int(c * 40)}")
    print()


def demo_coherence_oneness():
    print("=" * 70)
    print("DEMO 3 -- Coherence as 'one thing or many?'")
    print("=" * 70)
    rng = np.random.default_rng(7)
    sync = SyncGrouping(seed=1)
    one, _ = _make_groups(1, 12, 256, 0.5, rng)      # variations of a single thing
    many, _ = _make_groups(3, 4, 256, 0.5, rng)      # three distinct things
    c_one = sync.coherence(sync.run(one))
    c_many = sync.coherence(sync.run(many))
    print(f"\n  twelve variations of ONE thing -> coherence {c_one:.2f}  (locks into one)")
    print(f"  four each of THREE things      -> coherence {c_many:.2f}  (stays split)")
    print("\n  The order parameter physics uses for phase transitions doubles as a")
    print("  read on whether a set is one unified thing or several.\n")


if __name__ == "__main__":
    demo_emergent_grouping()
    demo_watch_locking()
    demo_coherence_oneness()
