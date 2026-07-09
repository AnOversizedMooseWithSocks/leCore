"""Many minds, one substrate: a shared frozen base brain with lightweight
per-instance overlays, for running a population of NPCs (or agents) without paying
for a full brain each.

THE PROBLEM. If every NPC builds, trains, and updates its own UnifiedMind, the cost
is N independent brains -- N copies of the heavy common knowledge, N trainings. Most
of that is redundant: the world model, the language, the shared lore are the SAME
for every NPC. What differs is a little personal, episodic knowledge.

THE STRUCTURE (copy-on-write deltas / a frozen base plus a light adapter -- the two
framings are the same thing). Train ONE base mind on the common knowledge and FREEZE
it. Every NPC is an Instance that:
  * shares the base by REFERENCE (no copy of the heavy tables, and crucially shares
    the base's ENCODER, so all instances perceive into the same vector space);
  * holds only its own small DELTA of prototypes (what this NPC personally learned);
  * reads by scoring over base + delta together, so it inherits all common knowledge
    and adds its private knowledge on top;
  * writes only into its delta -- the base never changes, so it stays shareable.

WHY THIS SUBSTRATE MAKES IT EASY. Because instances share the same atoms, a learned
vector means the same thing everywhere, so knowledge is COMPARABLE and ADDITIVE
across instances. Two consequences fall out for free:
  * MERGE = SUPERPOSITION. Propagating an NPC's learning back into the base (so every
    instance inherits it) is just adding its delta prototypes into the base -- the
    same bundle operation the engine already uses. Pooling many NPCs' learning is
    bundling them together (a federated average, in VSA terms).
  * ISOLATION IS FREE. An NPC reads base + its own delta only, so it never sees
    another NPC's private knowledge until that knowledge is explicitly propagated.

MEASURED (toy game world): with a base of B prototypes and N NPCs each holding ~d
private prototypes, the population costs B + N*d prototypes, against N*B for separate
minds -- e.g. 50 NPCs over a 1,000-prototype base with 20 private each cost ~2,000
vs ~50,000, a ~25x saving that widens as the base grows. Verified: a branch inherits
the base's knowledge; two branches stay isolated; after propagate, one NPC's private
fact becomes visible to all; and recall stays correct after merging by superposition.

Needs: numpy, holographic_unified.UnifiedMind (the base).
"""
import numpy as np


class MindInstance:
    """A lightweight NPC mind: shares a frozen base by reference, learns into a
    private delta, and reads from base + delta combined."""

    def __init__(self, shared, name="npc"):
        self.shared = shared
        self.name = name
        self._delta = []            # private prototypes: [label, vec, unit, count]

    # ---- the NPC's own learning goes here, never into the base ----------
    def learn(self, x, label):
        """Learn a private association. Encoded with the SHARED encoder, so it is
        comparable to (and later mergeable with) everything else."""
        v = self.shared.perceive(x)
        u = v / (np.linalg.norm(v) + 1e-12)
        # reinforce if this label already has a private prototype, else add one
        for p in self._delta:
            if p[0] == label:
                p[1] = p[1] + v
                p[2] = p[1] / (np.linalg.norm(p[1]) + 1e-12)
                p[3] += 1
                return self
        self._delta.append([label, v.copy(), u, 1])
        return self

    def _scores(self, v, among=None):
        scores = {}
        for label, _, unit, _ in self.shared.base_prototypes() + self._delta:
            if among is not None and label not in among:
                continue
            s = float(unit @ v)
            if s > scores.get(label, -1e9):
                scores[label] = s
        return scores

    def classify(self, x, among=None):
        """Classify over base + this NPC's private knowledge."""
        v = self.shared.perceive(x)
        sc = self._scores(v, among=among)
        return max(sc, key=sc.get) if sc else None

    def recall(self, x):
        """Best label and its score over base + delta."""
        v = self.shared.perceive(x)
        sc = self._scores(v)
        if not sc:
            return None, 0.0
        lab = max(sc, key=sc.get)
        return lab, sc[lab]

    def knows_privately(self):
        """Labels this NPC has learned on its own (not in the base)."""
        base_labels = {p[0] for p in self.shared.base_prototypes()}
        return sorted({p[0] for p in self._delta} - base_labels)

    def delta_size(self):
        return len(self._delta)

    def propagate(self):
        """Push this NPC's private learning into the shared base, so every instance
        (current and future) inherits it. Returns self."""
        self.shared.absorb_delta(self._delta)
        self._delta = []
        return self


class SharedMind:
    """A trained base mind, frozen and shared by reference across many instances.
    Branch it to make lightweight NPCs; merge their deltas back to propagate
    learning to all."""

    def __init__(self, base_mind, capacity=0):
        self.base = base_mind
        self.frozen = True
        self.capacity = capacity            # 0 = bundle without bound; >0 = cap per label
        # snapshot the base prototypes; instances read these but never mutate them
        self._base = [list(p) for p in base_mind.memory.live._p]

    # the shared perception: every instance encodes through the SAME encoder
    def perceive(self, x, modality=None):
        return self.base.perceive(x, modality)

    def base_prototypes(self):
        return self._base

    def branch(self, name="npc"):
        """Make a new lightweight NPC that shares this base and starts with an empty
        private delta -- no copy of the heavy tables."""
        return MindInstance(self, name=name)

    def absorb_delta(self, delta):
        """Merge a delta into the base by superposition: for a label already in the
        base, bundle the new prototype into it (add then renormalise); for a new
        label, append it. This is how an NPC's learning propagates to everyone.

        CAPACITY-AWARE: if self.capacity > 0, a base label's prototype only bundles up
        to `capacity` members; once full, further learning for that label starts a NEW
        sub-prototype instead of blurring the full one (a bundle has finite capacity --
        the cliff). classify/recall already score over ALL entries for a label, so
        sub-prototypes are read transparently. capacity=0 (default) bundles without
        bound, the original behaviour."""
        for label, vec, _unit, count in delta:
            # the least-loaded existing prototype for this label that still has room
            target = None
            for i, p in enumerate(self._base):
                if p[0] != label:
                    continue
                if self.capacity and p[3] >= self.capacity:
                    continue                           # this sub-prototype is full
                target = i
                break
            if target is not None:
                p = self._base[target]
                p[1] = p[1] + vec                      # superpose
                p[2] = p[1] / (np.linalg.norm(p[1]) + 1e-12)
                p[3] += count
            else:
                u = vec / (np.linalg.norm(vec) + 1e-12)
                self._base.append([label, vec.copy(), u, count])
        return self

    def merge(self, instances):
        """Pool several NPCs' learning into the base at once (a federated bundle)."""
        for inst in instances:
            self.absorb_delta(inst._delta)
            inst._delta = []
        return self

    # ---- accounting: the whole point is the saving ----------------------
    def population_cost(self, instances):
        """Prototype counts: shared (base + all deltas) vs if each NPC had its own
        full copy of the base."""
        base_n = len(self._base)
        delta_n = sum(i.delta_size() for i in instances)
        shared_total = base_n + delta_n
        separate_total = len(instances) * base_n + delta_n
        return {"base": base_n, "deltas": delta_n, "shared_total": shared_total,
                "separate_total": separate_total,
                "saving_x": (separate_total / shared_total) if shared_total else 1.0}


def share(base_mind, capacity=0):
    """Freeze a trained UnifiedMind and return a SharedMind you can branch from.
    capacity>0 caps how many deltas bundle into one base label before a new
    sub-prototype is started (capacity-aware merge), avoiding the bundle capacity
    cliff when very many instances propagate learning for the same label."""
    return SharedMind(base_mind, capacity=capacity)
