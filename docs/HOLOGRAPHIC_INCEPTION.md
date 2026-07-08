# Holographic inception: how deep does the structure go?

*Prompted by a question: a hard drive has physical structure, and data represented by that structure,
which when executed runs an OS, which can host a VM, which can host another OS. holostuff has the first
two rungs. Can we go deeper? How far can the inception go, and how do we "format the drive"?*

The answer is yes, and the reason is the one property that makes a holographic substrate special:
**the thing you store and the space you store it in are the same kind of object.** A hypervector can
hold a value, a record, a whole nested scene — or the *recipe* for building those, executed in place.
That is exactly the hard-drive tower: structure, data in the structure, and data that becomes new
structure when run. Below is the stack, where each rung sits in the engine, and — measured — how far
down it goes.

---

## The stack

| Hard-drive layer | holostuff rung | Status |
|---|---|---|
| Platter (physical structure) | the D-dimensional float vector | already the substrate |
| Low-level format | `derived_atom(seed, name, …)` — a deterministic alphabet | already here |
| File system | role–filler records (`bind`+`bundle`), nested directories (`compose_nested`) | already here |
| **OS that executes** | **`HoloMachine` — a program encoded as a vector, run by VSA ops** | **new this round** |
| VM inside the OS | the same nesting, applied to executable structure | measured (a depth law) |

### "Formatting the drive"
Formatting is just fixing a seed. `derived_atom(seed, …)` lays down the entire alphabet
deterministically: two roles (`OP`, `ARG`), a `SLOT` role for nesting, the opcode atoms, the data
atoms, and an address function `POS(i)`. Two machines with the same seed agree bit-for-bit — the
format is reproducible, which is the whole point of a format. Nothing here is learned or random at
run time; it is a layout.

### The file system was already there
A "file" is a role-filler record: `bundle(bind(key₁, val₁), bind(key₂, val₂), …)`, read back by
unbinding a key. A "directory" is a bundle of named files. A "path" is a chain of unbinds. Nested
directories — directories inside directories — are exactly `compose_nested`, which the engine already
had (measured ceiling ~1.0 at two groups, ~0.97 at three). So the file-system rung needed no new code;
it needed only to be *named*.

### The new rung: an OS that executes
The missing piece is an interpreter — something that treats a stored vector as a *program* and runs
it. `HoloMachine` does that with a deliberately tiny, readable instruction set:

```
LOAD x   : ACC = x                 BUNDLE x : ACC = bundle([ACC, x])
BIND x   : ACC = bind(ACC, x)      PERMUTE  : ACC = permute(ACC, 1)
HALT     : stop
```

A program is a list of `(opcode, operand)`. It is assembled into **one** hypervector:

```
instruction_i = bundle( bind(OP, opcode_i), bind(ARG, operand_i) )
program       = bundle_i ( bind(POS(i), instruction_i) )
```

Instructions and data live in the same vector space — von Neumann architecture, holographically. To
run it, the interpreter unbinds each address `POS(i)`, **cleans** the opcode and operand against their
codebooks (a wide-margin classification, robust to the crosstalk from all the other instructions
bundled into the same vector), and dispatches. Because the operand is cleaned to an *exact* atom
before use, the accumulator is built from clean atoms and stays **exact** even though reading the
program is noisy.

Measured: `LOAD a; BIND b; BUNDLE c` produces `ACC == bundle(bind(a,b), c)` at cosine **1.0000**, with
the decoded instruction trace exactly correct. The substrate executed a stored program.

---

## How far does it go? Two measured cliffs

### Drive size — how big a program fits
Every instruction adds two more bound terms to the same bundle, so eventually the crosstalk
overwhelms cleanup. Instruction-decode accuracy versus program length, at two dimensions:

- dim **1024**: ~100% to ~32 instructions, then the cliff (80% by 64).
- dim **4096**: ~100% to ~128 instructions, then the cliff (87% by 192).

Capacity is finite and scales with dimension — quadruple the dimension, roughly quadruple the program.
This is the honest HRR capacity wall, kept on the record rather than hidden.

### Inception depth — how deep the nesting goes
A program is just another value, so it can be stored as a "file" on a "disk" (a bundle), and that disk
can itself be the file on a higher disk — inception. How deep before the buried program stops running?
It depends entirely on **how cluttered each level is**:

- **Clean nesting** (the program is the only file at each level): runs correctly at depth **8 and
  beyond**. A pure chain of unitary bind/unbind barely degrades, so you can nest almost arbitrarily
  deep.
- **Busy disk** (each level also holds other files): the buried program corrupts after only **~3–4
  levels**, because every level adds crosstalk from its neighbours.

That is the law, and it is a satisfying one: *you can go as deep as you like if each level is
uncluttered; a crowded disk corrupts a buried program after a few levels.* Exactly like a real drive —
the more you cram in alongside, the sooner the thing underneath is unreadable. Both limits scale with
dimension, so a wider substrate is a bigger drive that nests deeper.

---

## So: how far can the inception go?

As far as you are willing to spend dimensions on. The tower is real — platter, format, file system,
an OS that executes, and an OS-inside-the-VM by nesting — and every rung is the *same* handful of
primitives (`bind`, `bundle`, `cleanup`, `permute`, `derived_atom`) pointed at itself. The only thing
that ends the recursion is noise, and noise is bought off with dimension: a finite but honest budget,
measured at every level. We were indeed at the very beginning of that tree. We are now a few rungs up
it, with the rungs counted.
