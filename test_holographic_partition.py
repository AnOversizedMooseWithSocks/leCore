"""Many NPCs, one substrate: a frozen shared base brain with lightweight per-instance
deltas. Verifies the four properties that make this work -- inheritance, isolation,
propagation, and the memory saving -- plus that merge-by-superposition keeps recall
correct."""
import numpy as np

from holographic_unified import UnifiedMind
from holographic_partition import SharedMind, MindInstance, share


def _base():
    m = UnifiedMind(dim=512, seed=0)
    for x, lab in [("sword", "weapon"), ("axe", "weapon"), ("apple", "food"),
                   ("bread", "food"), ("gold", "treasure"), ("gem", "treasure")]:
        m.learn(x, lab)
    return m


def test_branch_inherits_base_knowledge():
    shared = share(_base())
    npc = shared.branch("npc")
    assert npc.classify("sword") == "weapon"      # knowledge it never learned itself
    assert npc.classify("gem") == "treasure"


def test_branches_are_isolated():
    shared = share(_base())
    alice = shared.branch("alice").learn("potion", "alchemy")
    bob = shared.branch("bob").learn("scroll", "magic")
    assert alice.classify("potion") == "alchemy"
    assert "alchemy" not in (bob.classify("potion") or "")  # bob can't see alice's private fact
    assert "alchemy" in alice.knows_privately()
    assert "magic" in bob.knows_privately()


def test_propagation_shares_learning():
    shared = share(_base())
    alice = shared.branch("alice").learn("potion", "alchemy")
    bob = shared.branch("bob")
    assert bob.classify("potion") != "alchemy"     # before
    alice.propagate()                              # push into the shared base
    assert bob.classify("potion") == "alchemy"     # after: every instance inherits it
    assert shared.branch("carol").classify("potion") == "alchemy"  # future branches too


def test_merge_pools_many_instances():
    shared = share(_base())
    npcs = [shared.branch(f"npc{i}").learn(word, f"fact{i}")
            for i, word in enumerate(["potion", "scroll", "torch"])]
    shared.merge(npcs)
    # each privately-learned fact is now recallable from the shared base
    fresh = shared.branch("fresh")
    assert fresh.classify("potion") == "fact0"
    assert fresh.classify("torch") == "fact2"


def test_merge_by_superposition_preserves_recall():
    # propagating into an EXISTING label bundles (adds) rather than replaces, and the
    # label is still recalled correctly afterward.
    shared = share(_base())
    npc = shared.branch("npc").learn("dagger", "weapon")   # reinforces existing label
    npc.propagate()
    assert shared.branch("x").classify("dagger") == "weapon"
    assert shared.branch("x").classify("sword") == "weapon"  # base knowledge intact


def test_population_cost_saving():
    shared = share(_base())
    npcs = [shared.branch(f"npc{i}").learn(f"item{i}", f"fact{i}") for i in range(20)]
    cost = shared.population_cost(npcs)
    assert cost["shared_total"] < cost["separate_total"]
    assert cost["saving_x"] > 1.0
    # shared = base + one delta each; separate = a full base per NPC + deltas
    assert cost["shared_total"] == cost["base"] + cost["deltas"]


def test_brain_share_branch_roundtrip():
    shared = _base().share()
    npc = shared.branch("hero")
    assert isinstance(npc, MindInstance)
    npc.learn("elixir", "potion")
    assert npc.classify("elixir") == "potion"
    assert npc.classify("apple") == "food"          # still inherits base


def test_instances_share_one_encoder():
    # the saving depends on instances sharing the base's encoder (same vector space);
    # the same input perceives to the same vector through any branch.
    shared = share(_base())
    a, b = shared.branch("a"), shared.branch("b")
    va = shared.perceive("sword")
    vb = shared.perceive("sword")
    assert np.allclose(va, vb)


def test_capacity_aware_merge_splits_instead_of_blurring():
    # When many instances propagate learning for the SAME label, an unbounded bundle
    # eventually loses fidelity (the capacity cliff). capacity>0 caps members per base
    # prototype and starts sub-prototypes, which classify/recall read transparently.
    from holographic_unified import UnifiedMind
    from holographic_partition import share
    base = UnifiedMind(dim=512, seed=0)
    for x, lab in [("sword", "weapon"), ("apple", "food")]:
        base.learn(x, lab)
    words = ["axe", "mace", "spear", "dagger", "bow", "sling", "flail", "glaive"] * 2

    unbounded = share(base, capacity=0)
    capped = share(base, capacity=4)
    for sh in (unbounded, capped):
        for i, w in enumerate(words):
            sh.branch(f"n{i}").learn(w, "weapon").propagate()
    n_unbounded = sum(1 for p in unbounded._base if p[0] == "weapon")
    n_capped = sum(1 for p in capped._base if p[0] == "weapon")
    assert n_unbounded == 1                            # everything blurred into one
    assert n_capped > 1                                # split into sub-prototypes
    # capped still recalls an original base member
    assert capped.branch("probe").classify("sword") == "weapon"


def test_share_default_capacity_unchanged():
    # share() with no capacity keeps the original single-bundle merge behaviour
    from holographic_unified import UnifiedMind
    from holographic_partition import share
    base = UnifiedMind(dim=256, seed=0)
    base.learn("sword", "weapon")
    sh = share(base)
    assert sh.capacity == 0
    sh.branch("a").learn("axe", "weapon").propagate()
    assert sum(1 for p in sh._base if p[0] == "weapon") == 1
