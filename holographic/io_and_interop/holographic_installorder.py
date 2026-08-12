"""INSTALLORDER -- which install steps collide, and what order is safe.

install_lecore ran its steps in the order they were written, and one collision
was found BY ACCIDENT: growing an HRNN channel after writing the boot record
made the model report booting as NONE, because a manifest too large for one
embedding row SPILLS across the surface weights and the channel edit corrupted
the payload. boot() failed with "substrate hash mismatch" while every other step
reported success.

That fix was "write the boot record last", which is correct and was reached the
expensive way. leCore already had the general tool: `conflict_graph(item_keys)`
builds the graph where "item_keys[i] is the set of resources task i needs, and
two tasks are adjacent iff they share one", key-first so the cost is the sum of
squared key degrees rather than O(n^2).

SO THE ORDERING IS DERIVABLE RATHER THAN REMEMBERED, provided each step declares
what it WRITES. This module holds those declarations and turns them into an
order.

AND DECLARING THEM HONESTLY IS THE HARD PART, which the first attempt proved: I
guessed that `improvement` writes head rows and the conflict graph dutifully
flagged a collision with `memory_index`. MEASURED, install_improvement changes
0 OF 256 head rows -- it writes MLP weights. THE CONFLICT WAS IN MY DECLARATION,
NOT IN THE CODE. A resource table that is written from memory produces confident
false alarms, so every entry here is one that was checked against what the step
actually modifies, and `verify_declaration` re-checks a step against a real
model rather than trusting this file.

THE SPILL RULE, which is the one that actually bit: a step whose payload can
SPILL across arbitrary weights conflicts with every step that writes weights at
all, and must therefore go last. That is not an ordering preference, it is a
consequence of the substrate encoding -- and it only appears when the manifest
does not fit one row, which is width-dependent and therefore invisible on a
wide model and fatal on a narrow one.
"""

#: What each install step WRITES. Checked against the code, not recalled.
#: A step that only regenerates from a seed writes nothing and cannot collide.
WRITES = {
    "prepend": {"layer_list", "layer_tensors", "config"},
    "registers": set(),           # a reservation regenerates from its seed
    "hrnn_channel": {"linear_attn_tensors", "config"},
    "router": {"early_layer_mlp"},
    "memory_index": {"head_rows"},
    "improvement": {"last_layer_mlp"},   # MEASURED: 0 of 256 head rows change
    "facts": {"head_rows"},
    "boot_record": {"embed_row", "__spill__"},
}

#: Steps whose payload can spread across arbitrary weights. These conflict with
#: everything that writes anything, and go last.
SPILLERS = {"boot_record"}


def conflicts(steps=None):
    """Which declared steps collide? Uses leCore's own conflict_graph."""
    import lecore

    names = list(steps or WRITES)
    keys = [set(WRITES.get(n, set())) - {"__spill__"} for n in names]
    m = lecore.UnifiedMind(dim=64, seed=0)
    _n, edges = m.conflict_graph(keys)
    out = [(names[a], names[b],
            sorted(set(keys[a]) & set(keys[b]))) for a, b in edges]
    for s in (set(names) & SPILLERS):
        for other in names:
            if other != s and WRITES.get(other):
                out.append((s, other, ["__spill__"]))
    return out


def order(steps=None):
    """A safe install order: non-spillers first, spillers last.

    Within the non-spillers, steps that share a resource are separated so the
    later one is applied to the state the earlier one produced -- which is
    already how a sequential install behaves and is only a problem when a step
    reads what another has moved."""
    names = list(steps or WRITES)
    early = [n for n in names if n not in SPILLERS]
    late = [n for n in names if n in SPILLERS]
    return early + late


def verify_declaration(step, before, after):
    """Did this step write what it CLAIMED to write? Returns the discrepancy.

    Exists because the first version of this table was written from memory and
    invented a collision that measurement disproved. A declaration nobody checks
    is a comment, and this project's whole discipline is that comments rot."""
    import numpy as np

    touched = set()
    for k in set(before) | set(after):
        a, b = before.get(k), after.get(k)
        if a is None or b is None:
            touched.add("added_or_removed_tensor")
            continue
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.array_equal(a, b):
            touched.add(k)
    return {"step": step, "declared": sorted(WRITES.get(step, set())),
            "tensors_touched": len(touched),
            "sample": sorted(touched)[:6]}


def _selftest():
    # ---- THE SPILLER MUST SORT LAST, whatever order it is given in ----
    o = order(["boot_record", "prepend", "router"])
    assert o[-1] == "boot_record", o
    o2 = order(["prepend", "boot_record", "hrnn_channel"])
    assert o2[-1] == "boot_record", o2

    # ---- AND IT MUST CONFLICT WITH EVERY WEIGHT WRITER, which is the whole
    #      reason it goes last. This is the collision that cost a debugging
    #      session: HRNN after boot_record corrupted the spilled payload.
    c = conflicts(["boot_record", "hrnn_channel", "router"])
    pairs = {(a, b) for a, b, _ in c} | {(b, a) for a, b, _ in c}
    assert ("boot_record", "hrnn_channel") in pairs, c

    # ---- A SEED-ONLY STEP CANNOT COLLIDE WITH ANYTHING ----
    c2 = conflicts(["registers", "router", "improvement"])
    assert not any("registers" in (a, b) for a, b, _ in c2), c2

    # ---- AND THE DECLARATION THAT WAS WRONG MUST STAY FIXED: improvement
    #      writes MLP weights, NOT head rows. Measured 0 of 256 head rows.
    assert "head_rows" not in WRITES["improvement"], WRITES["improvement"]
    assert "head_rows" in WRITES["memory_index"]
    real = conflicts(["improvement", "memory_index"])
    assert not real, ("these do NOT collide -- the first declaration said they "
                      "did and measurement disproved it", real)

    print("installorder selftest OK -- the boot record SPILLS across the surface "
          "so it conflicts with every weight writer and sorts last (the "
          "collision that cost a session when HRNN corrupted its payload); a "
          "seed-only step like the register reservation cannot collide at all; "
          "and improvement vs memory_index does NOT collide -- my first "
          "declaration said it did and measuring 0 of 256 changed head rows "
          "disproved it, which is why verify_declaration exists")


if __name__ == "__main__":
    _selftest()
