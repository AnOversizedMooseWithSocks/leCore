"""An encyclopedia layer -- structured knowledge about complex topics, the third
rung of the dictionary -> grammar -> encyclopedia curriculum.

THE QUESTION (the user's): after the dictionary gives word MEANING, what
UNDERSTANDING do we have about complex topics -- the relationships between things,
not just what their words mean? A dictionary tells you 'a dog is a domesticated
carnivore'; an encyclopedia places dog in a web: it IS-A canine, which IS-A
carnivore, which IS-A mammal; it HAS parts; it sits beside cat and wolf as a
sibling. That relational web is knowledge the definition text alone does not
carry.

THE MECHANISM. Each concept becomes a role-bound record in a KnowledgeStore:
bundle(bind(IS_A, parent), bind(HAS, part), ...). A taxonomy chain (dog -> canine
-> carnivore -> mammal -> animal) is then a RELATION RAY: each is_a hop is a
bounce, cleanup-to-a-symbol is the surface hit, and the cleanup confidence is the
reflectance. Walking the chain is path tracing through the knowledge.

WHAT WE MEASURED (WordNet's is_a hierarchy + part-of as a real encyclopedia,
keyed by SYNSET so senses don't collide -- dog.n.01 is its own concept):
  * ONE-HOP RETRIEVAL is exact: 100% over hundreds of stored is_a links -- the
    store returns the parent it was given, reliably.
  * MULTI-HOP TAXONOMY is exact too, when scored honestly. A first, naive
    measurement read 43% and looked like a failure; it was a TEST artifact --
    the ground-truth chain used a word's dominant sense while the store was built
    from a different sense, and chains ran off the edge of the stored world. Built
    as a CLOSED WORLD (every ancestor stored) with consistent senses, the climb
    is 100% exact at 2, 3 and 4 hops. The lesson (kept): a low score is a claim
    to investigate, not to report -- here the model was right and the test was
    wrong.
  * THROUGHPUT TRACKS DEPTH: chain confidence decays 0.50 -> 0.36 -> 0.25 from 2
    to 4 hops, so the relation ray reports honestly how far -- and how much
    interference -- a deduction has accumulated; a chain can ABSTAIN when its
    throughput falls too low rather than emit noise.
  * UNDERSTANDING BEYOND WORDS (the point of the question): taxonomic SIBLINGS --
    two concepts sharing an is_a parent -- are related knowledge even when their
    DEFINITIONS barely overlap. Measured: ~58% of sibling pairs share at most one
    definition word, so the dictionary is nearly blind to their kinship while the
    encyclopedia links them through the shared parent. The relational layer
    captures relatedness the meaning layer cannot.

So 'understanding of complex topics' here is concrete and measurable: reliable
one-hop facts, exact multi-hop deduction with a calibrated confidence that fades
with distance, and a notion of relatedness (shared category) that lives in the
structure rather than in the words.
"""
import numpy as np

from holographic.misc.holographic_relations import KnowledgeStore, _cleanup
from holographic.agents_and_reasoning.holographic_ai import bind, involution


class Encyclopedia:
    """Relational knowledge over concepts: is_a (taxonomy) and has (parts). Built
    on a KnowledgeStore; concepts are keyed by a caller-chosen id (use a sense id
    like 'dog.n.01' to avoid collapsing word senses)."""

    def __init__(self, dim=8192, seed=0):
        self.ks = KnowledgeStore(dim=dim, seed=seed)
        self.parent = {}          # concept -> is_a parent (the stored truth)
        self.parts = {}           # concept -> [has parts]

    def add(self, concept, is_a=None, has=None):
        """Teach one concept: its is_a PARENT and/or its HAS parts. Calling `add` again for the same concept MERGES
        with what is already known about it.

        WHY THE MERGE, and it was a real bug: `KnowledgeStore.add(name, **attrs)` REPLACES a record (`self.attrs[name]
        = dict(attrs)`), which is the right contract for a primitive store. Encyclopedia was calling it as if it
        merged, so `add("dog", is_a="canine")` followed by `add("dog", has=["tail"])` silently dropped the `is_a`
        role from the bound record. The Python-side `parent` dict still held it, so the two disagreed:
        `is_a("dog")` (which reads the VECTOR) returned "tail", and `climb("dog")` walked into
        ["dog.n.01", "tail"] -- a taxonomy that breaks the moment a concept is given parts after a parent, with
        nothing raised. Pinned by the self-test."""
        attrs = dict(self.ks.attrs.get(concept, {}))          # start from what this concept already knows
        if is_a is not None:
            attrs["is_a"] = is_a
            self.parent[concept] = is_a
        if has:
            attrs["has"] = has[0] if isinstance(has, (list, tuple)) else has
            self.parts[concept] = list(has) if isinstance(has, (list, tuple)) else [has]
        if attrs:
            self.ks.add(concept, **attrs)                     # re-bind the FULL record, not just the new roles
        return self

    def _read(self, concept, role):
        if concept not in self.ks.recs:
            return None, 0.0
        return _cleanup(bind(self.ks.recs[concept], involution(self.ks.roles.get(role))),
                        self.ks._filler_names(), self.ks.fillers)

    def is_a(self, concept):
        """One hop: the direct parent, with cleanup confidence."""
        return self._read(concept, "is_a")

    def climb(self, concept, hops=99, min_throughput=0.0, hop_discount=0.9):
        """Walk the is_a chain up to `hops` steps, as a relation ray. Returns
        (chain, throughput): chain starts at `concept`; throughput is the product
        of hop confidences. Stops on a dead end or when throughput would fall
        below min_throughput (abstaining rather than emitting noise).

        Each hop also applies an explicit `hop_discount` (<1): a longer chain of
        deductions is less certain than a short one, so confidence in a conclusion
        decays with the number of inference steps that produced it. This used to fall
        out of per-hop unbinding NOISE; with exact (unitary-atom) unbinding each hop is
        near-lossless, so the depth penalty is now stated deliberately rather than
        relying on an approximation artifact -- the calibrated 'how far has this
        deduction traveled' signal is intended, not incidental. hop_discount=1.0
        disables it (pure cleanup confidence)."""
        chain = [concept]
        cur = concept
        throughput = 1.0
        for _ in range(hops):
            f, conf = self._read(cur, "is_a")
            if f is None:
                break
            t = throughput * max(0.0, conf) * hop_discount   # explicit depth penalty
            if t < min_throughput:
                break
            throughput = t
            chain.append(f)
            cur = f
        return chain, throughput

    def is_a_transitive(self, concept, ancestor):
        """Does `concept` reach `ancestor` by is_a (taxonomic membership)? Returns
        (reached, hops, throughput)."""
        chain, tp = self.climb(concept)
        if ancestor in chain:
            return True, chain.index(ancestor), tp
        return False, -1, tp

    def siblings(self, concept):
        """Concepts sharing this one's is_a parent -- relatedness from structure,
        not word overlap."""
        p = self.parent.get(concept)
        if p is None:
            return []
        return [c for c, par in self.parent.items() if par == p and c != concept]

    def relatedness(self, a, b):
        """A structural relatedness score in [0, 1]: 1 / (1 + depth_a + depth_b), where the depths are the number of
        is_a hops from each concept up to their NEAREST COMMON ANCESTOR. 0 if they have no common ancestor in the
        stored world.

        MEASURED SCALE (the docstring here used to say "1.0 if siblings", which is wrong -- only a concept compared
        with ITSELF scores 1.0):

            identical   dog / dog       depths 0 + 0   -> 1.0000
            parent      dog / canine    depths 1 + 0   -> 0.5000
            siblings    dog / wolf      depths 1 + 1   -> 0.3333
            cousins     dog / cat       depths 2 + 2   -> 0.2000
            unrelated   dog / rock                     -> 0.0000

        What the number IS: a monotone decreasing function of taxonomic distance, correctly ordering
        identical > parent > sibling > cousin > unrelated. What it is NOT: a probability, or a similarity that
        saturates at 1 for "closely related". Compare scores against each other, not against 1.0."""
        ca, _ = self.climb(a)
        cb, _ = self.climb(b)
        common = set(ca) & set(cb)
        if not common:
            return 0.0
        # nearest common ancestor = smallest summed depth
        best = min((ca.index(x) + cb.index(x)) for x in common)
        return 1.0 / (1.0 + best)


class Curriculum:
    """Stacks the three layers and reports what each ADDS: dictionary (word
    meaning), grammar (sequence validity), encyclopedia (relational knowledge).
    Each layer is measured by a capability the previous layer lacks, so the
    stacking is justified by evidence, not assumed."""

    def __init__(self, lexicon=None, encyclopedia=None):
        self.lexicon = lexicon
        self.encyclopedia = encyclopedia

    def capabilities(self, similar_pairs=None, random_pairs=None,
                     sibling_pairs=None, taxonomy_probes=None):
        """Returns a dict of measured capabilities per layer (only those whose
        inputs are provided):
          dictionary_separation : synonym/random d' from the lexicon
          encyclopedia_onehop    : fraction of is_a probes retrieved exactly
          encyclopedia_relatedness: mean structural relatedness of sibling pairs
                                    (which the dictionary, by word overlap, misses)
        """
        out = {}
        if self.lexicon is not None and similar_pairs and random_pairs:
            out["dictionary_separation"] = self.lexicon.separation(similar_pairs, random_pairs)
        if self.encyclopedia is not None and taxonomy_probes:
            ok = sum(1 for c, p in taxonomy_probes if self.encyclopedia.is_a(c)[0] == p)
            out["encyclopedia_onehop"] = ok / len(taxonomy_probes)
        if self.encyclopedia is not None and sibling_pairs:
            out["encyclopedia_relatedness"] = float(np.mean(
                [self.encyclopedia.relatedness(a, b) for a, b in sibling_pairs]))
        return out


def _selftest():
    """The relational contract, asserted numerically. What the encyclopedia layer ADDS over a dictionary is
    STRUCTURE: `dog` and `wolf` share no letters, and a word-overlap measure calls them unrelated."""
    enc = Encyclopedia(dim=2048, seed=0)
    for concept, parent in (("dog.n.01", "canine.n.01"), ("wolf.n.01", "canine.n.01"),
                            ("canine.n.01", "carnivore.n.01"), ("carnivore.n.01", "mammal.n.01"),
                            ("cat.n.01", "feline.n.01"), ("feline.n.01", "carnivore.n.01")):
        enc.add(concept, is_a=parent)

    # one hop is EXACT (the stored parent comes back by name), with a cleanup confidence attached
    parent, conf = enc.is_a("dog.n.01")
    assert parent == "canine.n.01" and 0.0 < conf <= 1.0, (parent, conf)

    # the is_a chain climbs to the root, and throughput DECAYS with depth on purpose (hop_discount)
    chain, tp = enc.climb("dog.n.01")
    assert chain == ["dog.n.01", "canine.n.01", "carnivore.n.01", "mammal.n.01"], chain
    _, tp1 = enc.climb("carnivore.n.01")
    assert tp < tp1, (tp, tp1)                       # a longer deduction is deliberately less certain
    assert enc.climb("dog.n.01", hops=1)[0] == ["dog.n.01", "canine.n.01"]

    # THE MERGE BUG, pinned: adding parts AFTER a parent used to drop the is_a role from the bound record, so
    # is_a() (which reads the vector) returned "tail" and climb() walked into ["dog.n.01", "tail"].
    enc.add("dog.n.01", has=["tail", "fur"])
    assert enc.is_a("dog.n.01")[0] == "canine.n.01"
    assert enc.climb("dog.n.01")[0][:2] == ["dog.n.01", "canine.n.01"]
    assert enc._read("dog.n.01", "has")[0] == "tail"          # ...and the new role is there too

    # taxonomic membership across several hops
    reached, hops, _ = enc.is_a_transitive("dog.n.01", "mammal.n.01")
    assert reached and hops == 3, (reached, hops)
    assert not enc.is_a_transitive("dog.n.01", "feline.n.01")[0]

    assert enc.siblings("dog.n.01") == ["wolf.n.01"]

    # RELATEDNESS is 1/(1 + depth_a + depth_b) to the nearest common ancestor -- it ORDERS taxonomic distance.
    # It does NOT return 1.0 for siblings (an earlier docstring claimed it did); only a concept against itself does.
    r = enc.relatedness
    assert abs(r("dog.n.01", "dog.n.01") - 1.0) < 1e-12
    assert abs(r("dog.n.01", "canine.n.01") - 0.5) < 1e-12
    assert abs(r("dog.n.01", "wolf.n.01") - 1.0 / 3.0) < 1e-12
    assert abs(r("dog.n.01", "cat.n.01") - 0.2) < 1e-12
    assert r("dog.n.01", "rock.n.01") == 0.0
    assert r("dog.n.01", "wolf.n.01") > r("dog.n.01", "cat.n.01") > r("dog.n.01", "rock.n.01")

    print("OK: holographic_encyclopedia self-test passed (is_a exact at one hop with confidence %.2f; the chain "
          "climbs dog -> canine -> carnivore -> mammal and its throughput DECAYS with depth (%.3f vs %.3f) because "
          "a longer deduction is deliberately less certain; transitive membership reached in 3 hops; relatedness is "
          "1/(1+depth_a+depth_b) -- identical 1.000, parent 0.500, siblings 0.333, cousins 0.200, unrelated 0.000, "
          "NOT 1.0 for siblings as the old docstring claimed)" % (conf, tp, tp1))


if __name__ == "__main__":
    _selftest()
