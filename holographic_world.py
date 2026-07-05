"""holographic_world.py -- a shared WORLD of vector slots you can fork, edit alone, and merge back.

This closes the multiplayer loop. A "world" here is just a named set of SLOTS, each holding a vector -- the shared
state several people (or agents, or forks) edit: scene objects, positions, parameters, whatever you name. The whole
point is that editing is SAFE and reconcilable:

    mine   = mind.workspace.fork("lab")     # a copy-on-write view of the "lab" world
    mine.set("sky", blue_vector)            # my edits accumulate in MY delta; the shared world is untouched
    ...
    res = mind.merge_forks([mine.delta, theirs.delta], policy="select")   # reconcile (agree -> merge, else surface)
    mind.apply(res["merged"], world="lab")  # write the agreed edits back to the shared world

"A world is a seed + deltas": the base can be regenerated deterministically anywhere (from a procgen/recipe seed), and
only the sparse slot edits (the delta) ever travel. This module implements the concrete slot side of that idea -- the
part merge_forks consumes -- and stays deliberately small and readable: dicts of vectors, copy-on-write on fork.

numpy/stdlib only; deterministic.
"""
import numpy as np


class World:
    """A shared world = a dict of named slots -> vectors. fork() hands out a copy-on-write editing view; apply() writes
    a merged delta back into the shared state."""

    def __init__(self, name="default", slots=None):
        self.name = name
        self.slots = {k: np.asarray(v, float) for k, v in (slots or {}).items()}   # the shared base state

    def get(self, slot):
        """The current value of a slot (or None)."""
        return self.slots.get(slot)

    def set(self, slot, vec):
        """Set a slot directly on the SHARED world (no fork). Use a fork for isolated edits you'll merge later."""
        self.slots[slot] = np.asarray(vec, float)
        return self

    def fork(self):
        """A private, copy-on-write editing view -- edits accumulate in the fork's delta, not here."""
        return Fork(self)

    def apply(self, delta):
        """Write a merged delta ({slot: vector}) into the shared world. Returns the slots changed."""
        for slot, vec in delta.items():
            self.slots[slot] = np.asarray(vec, float)
        return sorted(delta.keys(), key=str)

    def __repr__(self):
        return "World(%r, %d slots)" % (self.name, len(self.slots))


class Fork:
    """A copy-on-write editing view of a World. Reads fall through to the world's base; writes go into a PRIVATE delta,
    so your edits don't touch the shared world (or anyone else's fork) until you merge and apply. `fork.delta` is the
    {slot: vector} you hand to merge_forks."""

    def __init__(self, world):
        self.world = world
        self._base = world.slots                        # read-through to the shared base (not copied -- cheap)
        self.delta = {}                                 # your private edits: slot -> vector

    def get(self, slot):
        """This fork's view of a slot: your edit if you made one, else the shared base value."""
        return self.delta[slot] if slot in self.delta else self._base.get(slot)

    def set(self, slot, vec):
        """Edit a slot in THIS fork only (records it in the delta). Chainable."""
        self.delta[slot] = np.asarray(vec, float)
        return self

    def slots(self):
        """The fork's full view: the shared base overlaid with your private edits."""
        view = dict(self._base)
        view.update(self.delta)
        return view

    def __repr__(self):
        return "Fork(of %r, %d edits)" % (self.world.name, len(self.delta))


class WorldSpace:
    """Holds named worlds. fork(name) forks one; apply(delta, name) writes a merged delta back. This is what
    mind.workspace exposes for the fork/merge/apply loop (distinct from the DB WorkspaceManager, which handles
    database-table tiers -- a different kind of 'workspace')."""

    def __init__(self):
        self.worlds = {}

    def world(self, name="default"):
        """Get (creating if needed) the named world."""
        if name not in self.worlds:
            self.worlds[name] = World(name)
        return self.worlds[name]

    def fork(self, name="default"):
        """Fork the named world into a copy-on-write editing view."""
        return self.world(name).fork()

    def apply(self, delta, name="default"):
        """Write a merged delta back into the named world."""
        return self.world(name).apply(delta)

    def names(self):
        return sorted(self.worlds)


def _selftest():
    rng = np.random.default_rng(0)
    dim = 256
    ground = rng.standard_normal(dim); ground /= np.linalg.norm(ground)

    ws = WorldSpace()
    world = ws.world("lab")
    world.set("ground", ground)                          # a shared starting state

    # two people fork the same world
    mine = ws.fork("lab")
    theirs = ws.fork("lab")

    # I edit 'sky'; they edit 'sky' the SAME way (agree) and also add 'tree' (only them)
    blue = rng.standard_normal(dim)
    mine.set("sky", blue)
    theirs.set("sky", blue + 0.002 * rng.standard_normal(dim))
    theirs.set("tree", rng.standard_normal(dim))

    # the shared world is untouched by fork edits (copy-on-write)
    assert "sky" not in world.slots and "tree" not in world.slots
    # each fork sees its own edits + the shared base
    assert mine.get("ground") is not None and np.allclose(mine.get("sky"), blue)
    assert "tree" not in mine.slots()                    # my fork never saw their 'tree'

    # reconcile the deltas and apply the agreed result back
    from holographic_merge import merge_forks
    res = merge_forks([mine.delta, theirs.delta], policy="select")
    assert "sky" in res["merged"] and "tree" in res["merged"] and not res["conflicts"]
    changed = world.apply(res["merged"])
    assert "sky" in world.slots and "tree" in world.slots and "sky" in changed

    # a genuine conflict is surfaced, not applied
    a = ws.fork("lab"); b = ws.fork("lab")
    a.set("color", rng.standard_normal(dim))
    b.set("color", rng.standard_normal(dim))
    res2 = merge_forks([a.delta, b.delta], policy="select")
    assert not res2["merged"] and res2["conflicts"][0][0] == "color"

    print("OK: holographic_world self-test passed (fork is copy-on-write -- shared world untouched until apply; each "
          "fork sees only its own edits over the base; agreeing edits merge and apply back; a real conflict is "
          "surfaced, not silently written)")


if __name__ == "__main__":
    _selftest()
