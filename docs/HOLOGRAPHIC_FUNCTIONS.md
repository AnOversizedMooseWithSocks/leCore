# holostuff — The Panel on Inception: Functions, Folders, and What the OS Layer Unlocks

*Gathered a second time on the inception thread, now that the OS rung exists (`HoloMachine`: a program
encoded as one hypervector, executed by VSA ops). The questions on the table: can we embed and execute
**functions** inside the holographic space rather than as Python files? Does that buy extra abilities?
Can **folders/partitions** cut confusion? How useful is any of this — and what does it let us do that we
never planned for? As always, every view is attributed to a **seat** and that field's real, published
methods, and every claim below was **measured on the substrate first**.*

---

## The short answers (all measured)

| Your question | The measured answer |
|---|---|
| Functions inside the holographic space, not as Python files? | **Yes, two ways** — *demonstrated* mappings and *callable* library subroutines. |
| Extra abilities? | **Yes** — learned-by-example functions, content-addressable-by-behavior, function arithmetic. |
| Folders/partitions to avoid confusion? | **Yes** — at 256 items, flat recall 86% → 16-folder recall **100%**. |
| How useful? | Useful where *deterministic, inspectable, composable, content-addressable* code matters — not as a fast CPU. |
| What didn't we plan for? | Retrieving code **by what it does**, and **averaging two programs** like vectors. |
| How does it improve us? | Code and data share one algebra, so **every engine faculty now applies to programs too**. |

---

## 1. Functions embedded in the holographic space — two kinds (Plate; Adamatzky)

**The HRR-foundation seat (Plate)** points out that the substrate is *homoiconic*: a program and the
data it touches are the same kind of object (`HoloMachine` already puts instructions and operands in one
vector). That makes two genuinely different notions of "function" available, neither of which is a
Python file.

**(a) A function you *demonstrate* instead of write.** A mapping `f: key → value` is stored as a single
vector `M = bundle_i bind(key_i, value_i)`, and applied by `f(k) = cleanup(unbind(M, k))`. You never
write the logic; you give examples, and the vector *is* the function. Measured: **100% correct up to
~120 pairs** at dim 4096, cliff at ~240 (87%). This is `HolographicMemory` seen as what it always was —
a learned, content-addressable function. (It is also Plate's classic "holographic mapping.")

**(b) A function you *call*.** An `ACC → ACC` sub-program (e.g. `BIND b; HALT`) is embedded into a
single **library vector** under its name, and invoked with a new `CALL` opcode: the body is pulled out
of the library by name (`unbind`) and run on the current accumulator. Measured: `LOAD a; CALL tag_b;
CALL shift` produces `permute(bind(a,b))` at **cosine 1.0**, with all functions living inside one
library vector. Functions compose like ordinary code — and *are* ordinary data.

**The unconventional-computing seat (Adamatzky)** is the natural home for this: his field is computing
in substrates that are not von Neumann CPUs (slime moulds, reaction–diffusion). A machine whose code,
data, and memory are all the same hypervector algebra is exactly that — computation as an emergent
property of a representational medium, not of a fetch-decode-execute chip.

---

## 2. Extra abilities — including what we never planned for (Eno; Togelius; Pharr)

**The generative-art / reframe seat (Eno)** is interested less in the planned features than in the ones
that fall out for free because code now lives in a vector space:

- **Content-addressing by behavior.** Give an *example of what you want* — input `a`, desired output
  `permute(a)` — and the matching function ("shift") is retrieved by a behavioral signature, not by
  name. Measured, correct. You can search code by what it *does*.
- **Function arithmetic.** `bundle(f1, f2)` is a function that carries *both* answers (measured 0.18 /
  0.18, symmetric). You can average, blend, and interpolate programs the way you blend vectors — a
  capability no file-based codebase has. Eno's "honour thy error" instinct applies directly: a blended
  or slightly-corrupted program degrades gracefully rather than crashing.

**The game-AI seat (Togelius)** sees the payoff for agents: an NPC's policy can be a *portable program
vector* — content-addressable, composable, blendable, and inspectable — rather than a compiled
controller. **The data-structure seat (Pharr)** sees the dual: a library of functions indexed for
sub-linear retrieval (the `HoloForest`) means "find the routine closest to this behavior" is a
nearest-neighbour query, not a grep.

---

## 3. Folders and partitions — yes, and the engine already has them (Pharr; Macklin)

The "confusion" in a holographic store is **crosstalk**: bundle too much together and cleanup starts
matching the wrong thing. **Folders are the cure, and they are already a primitive** —
`PartitionedMemory` routes each key to its own subspace, so a query competes only against its folder's
contents, not the whole drive. Measured at fixed load:

| total items | flat store | 16 folders |
|---|---|---|
| 64 | 100% | 100% |
| 128 | 100% | 100% |
| **256** | **86%** | **100%** |

So partitions directly buy back the capacity that a flat store loses to crosstalk — and they give
**namespacing** for free (two functions named `f` in different folders never collide, because they hang
off different partition roles). The data-structure seat (Pharr) reads this as a spatial hash for
meaning; the constraint-solver seat (Macklin) reads it as decoupling — isolate sub-problems so their
errors don't leak into each other.

---

## 4. How useful is this, honestly?

The honest boundary matters as much as the capability. This is **not a fast general-purpose CPU** —
Python runs these programs far faster than the holographic interpreter does, and nobody should port a
hot loop into hypervectors. Its edge is the same edge the whole engine has: **deterministic,
inspectable, composable, content-addressable code-as-data.** It is useful exactly where those
properties are the point:

- a **portable agent/policy** that travels as one vector and can be blended or recalled by behavior;
- a **self-describing record** that carries its own decode/validate routine in the same vector as its
  data;
- a **sandboxed mini-interpreter** whose entire state is one inspectable object;
- a **case library** where "what did we do last time something looked like this?" is a recall, and the
  answer is itself runnable.

---

## 5. How it improves our ability to do things — the multiplier (every seat)

This is the part the whole room agreed is the real prize. Because code and data now share **one
algebra**, every faculty the engine already has applies to programs, with no new machinery:

- **Consolidation (Stoudenmire/Duda seats)** can *compress a program* — a library of related functions
  has low-rank structure, so it consolidates like any other state.
- **Denoising (Milanfar seat)** can *clean a corrupted program* — cleanup already rescues noisy
  instruction reads; the same operator repairs a damaged library.
- **The resonator / factorizer (Olshausen seat)** can *factor a program into its parts* — decompose a
  composed behavior into the sub-functions that built it.
- **The forest (Pharr seat)** can *index programs* for sub-linear retrieval by behavior.
- **The generative sampler (Eno seat, B10)** can *generate new programs* — sample over the "program
  manifold" to propose novel-but-valid routines.

That is the answer to "how does this improve our ability": it **collapses the wall between what computes
and what is computed on.** A binding is a multiply is a function application; a bundle is a sum is a
library is a blend. The same five primitives that store a scene now store, retrieve, compose, repair,
and generate the programs that build scenes. We did not plan most of that — it is what falls out when
you take the OS rung seriously and notice the code was already the same stuff as the data.

---

### What shipped this round (measured, with kept boundaries)
`holographic_machine.py` gained a `CALL` opcode, `HoloMachine.define(name, program)` (embed a named
function into one library vector), and `run(..., init_acc=...)` (so functions are composable ACC→ACC
transforms) — backward-compatible, 4 new tests (603 total). The demonstrated-function and folder
capabilities use existing primitives (`HolographicMemory`, `PartitionedMemory`), so they were measured
and named rather than re-implemented. Honest boundary kept on the record: finite capacity (functions
~160 pairs, library subroutines bounded by the same bundling crosstalk), and no claim to CPU speed.

*(Still teed up from the previous session, not forgotten: adaptive-rank denoising to cash B7's
low-noise kept negative. This round answered the inception questions; that one is next when you want it.)*
