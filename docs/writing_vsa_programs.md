# Writing VSA Programs in leCore

*A guide to `HoloMachine` — the small stored-program machine that lets you express custom logic as a
holographic program: a sequence of instructions encoded into ONE hypervector, executed on the same
bind/bundle/cleanup algebra as the rest of the engine. Every example below was run and its result is shown.*

> **See also:** the [code reference](REFERENCE.md) for a map of every module, and the [README](README.md) for the big picture. This guide covers `HoloMachine` specifically -- writing your own programs on the vector algebra.

## What this is for

Sometimes you don't want a permanent faculty of the mind — you want to run *your own* logic over the
vector algebra: a scientist with a dataset who wants to try several experiments, none of which needs to
become a fixture of leCore; or a game with NPCs whose behaviour should live at the VSA level so it
reacts to game data automatically when a trigger fires. `HoloMachine` is that: you write a short program,
`assemble()` it into a single vector, and `run()` it. A program is *data* — content-addressable, storable,
and composable in the same space as everything else — so it's cheap to make, throw away, and remake.

The engine's job is to provide the primitives; your program composes them. `HoloMachine` is the
composition layer for one-off / domain-specific logic that shouldn't be baked into the core.

## The model in one breath

A program is a Python list of `(opcode, operand)` tuples. `assemble()` encodes the whole list as ONE
hypervector — each instruction is `bind(POS_i, bundle(bind(OP, opcode), bind(ARG, operand)))`. `run()`
reads each address back by unbinding its position, **cleans** the opcode and operand against their
codebooks (so the accumulator is built from *exact* atoms even though the read itself is noisy), and
executes. It returns `(accumulator, trace)` — the final ACC vector and the list of decoded instructions
that actually ran.

There is one register: the **accumulator** (ACC). Instructions transform it.

## Quick start

```python
from holographic.agents_and_reasoning.holographic_machine import HoloMachine

vm = HoloMachine(dim=4096, seed=7)            # default data alphabet is 'a'..'f'
prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
acc, trace = vm.run(vm.assemble(prog))
# acc == bundle(bind(a, b), c)   -> cosine 1.0000
# trace == [('LOAD','a'), ('BIND','b'), ('BUNDLE','c')]
```

`HALT` and `PERMUTE` take no real operand — pass `""`. The data atoms `a`..`f` exist by default; to use
your own names, pass `data=[...]` to the constructor (below).

## The instruction set

| Instruction | Operand | Effect on ACC |
|---|---|---|
| `LOAD x` | data name | `ACC = x` (put a value in the accumulator) |
| `BIND x` | data name | `ACC = bind(ACC, x)` — associate (the algebra's "multiply") |
| `BUNDLE x` | data name | `ACC = bundle([ACC, x])` — superpose (the algebra's "add") |
| `PERMUTE` | `""` | `ACC = permute(ACC, 1)` — shift (encodes order/position) |
| `CALL f` | function name | run library function `f` (an ACC→ACC sub-program) on ACC |
| `APPLY g` | faculty name | `ACC = g(ACC)` via a host-supplied handler (the bridge to the mind) |
| `IFMATCH x` | data name | run the NEXT instruction only if `cosine(ACC, x) >= branch_tol` (default 0.5), else skip it |
| `REPEAT n` | count 1..8 | run the FOLLOWING `CALL` n times |
| `ITERATE f` | function name | re-apply function `f` to ACC until it converges (fixed point) or a host `stop(acc)` is met |
| `STORE r` | register name | copy ACC into named register `r` (R0..R7) — exact, no crosstalk with ACC |
| `RECALL r` | register name | load register `r` back into ACC — exact (the value returns verbatim) |
| `PUSH` | `""` | push ACC onto the permute-stack (for nesting / save-and-restore) |
| `POP` | `""` | pop the stack's top back into ACC (cleaned against the data atoms) |
| `HALT` | `""` | stop |

Operand codebooks are kept separate, which is why cleanup is reliable: `LOAD/BIND/BUNDLE/IFMATCH` operands
clean against the **data** atoms, `CALL/ITERATE` against **function names**, `APPLY` against **faculty
names**, `STORE/RECALL` against the **register names** (R0..R7), and `REPEAT` against the small-integer
**counts**.

## Functions: name a sub-program, call it by name

A function is an ACC→ACC sub-program embedded into a holographic library and invoked by name. Define it
before assembling any program that calls it.

```python
vm.define("tag_b", [("BIND", "b"), ("HALT", "")])     # ACC := bind(ACC, b)
acc, trace = vm.run(vm.assemble([("LOAD", "a"), ("CALL", "tag_b"), ("HALT", "")]))
# acc == bind(a, b)   -> cosine 1.0000 ;  trace == [('LOAD','a'), ('CALL','tag_b')]
```

Functions are data too — they're stored in the same vector space, so they compose and nest (a function
can CALL another, with a recursion-depth guard).

## Triggers: reacting to data with `IFMATCH`

This is the "react to game data when a trigger is hit" pattern, and it's the VSA-native answer to
"do we have callbacks?" — `IFMATCH` is a one-instruction conditional that gates whatever comes next on a
*similarity* test against the accumulator. Pair it with `CALL` for an if-then: *if the current state looks
like this trigger, run this response.*

```python
g = HoloMachine(dim=4096, seed=7, data=["enemy_near", "calm", "flee_signal"])
g.define("raise_alarm", [("LOAD", "flee_signal"), ("HALT", "")])      # the response

# the reactive program: IF state matches 'enemy_near', run raise_alarm
trigger = g.assemble([("IFMATCH", "enemy_near"), ("CALL", "raise_alarm"), ("HALT", "")])

# fire it on a state that MATCHES the trigger (start ACC at the current state):
acc_hit,  _ = g.run(trigger, init_acc=g.data_atoms["enemy_near"])
# -> ACC == flee_signal (cosine 1.0000): the response ran. trace: [('IFMATCH','enemy_near'), ('CALL','raise_alarm')]

# fire the SAME program on a state that does NOT match:
acc_miss, _ = g.run(trigger, init_acc=g.data_atoms["calm"])
# -> ACC == calm (cosine 1.0000): the response was skipped. trace: [('IFMATCH','enemy_near')]
```

The match is a cosine threshold, so the trigger fires not just on the exact atom but on anything *close
enough* to it — which is what you want for a noisy game state that resembles the trigger condition. Tune
the threshold per `run(..., branch_tol=...)`.

A practical NPC loop, then, is: encode the NPC's current observation into ACC (with the mind's encoder),
run a small trigger program that checks ACC against the conditions you care about and CALLs the matching
response — all at the VSA level, reusing your atoms. The host code just feeds the observation in and reads
the response out.

## Loops

```python
vm.define("shift", [("PERMUTE", ""), ("HALT", "")])
acc, _ = vm.run(vm.assemble([("LOAD", "a"), ("REPEAT", 3), ("CALL", "shift"), ("HALT", "")]))
# acc == permute(a, 3)   -> cosine 1.0000  (REPEAT runs the following CALL n times)
```

`ITERATE f` is the fixed-point loop — re-apply `f` to ACC until it stops changing (`cosine(acc, prev) >=
converge_tol`) or a host `stop(acc)` predicate says the desired output is reached. That's the
input→process→feed-back loop behind cleanup / resonator / denoise, now expressible as a program.

## Registers and a stack: `STORE`/`RECALL` and `PUSH`/`POP`

ACC is the only working register, but two extra stores let a program hold more than one value at a time
without bundling them together (which would cost capacity and crosstalk).

**A register file — `STORE r` / `RECALL r`.** Eight named slots (R0..R7) beside ACC. `STORE` copies ACC
into a slot and `RECALL` reads it back *verbatim* — the slots are separate codebook entries, so there is no
crosstalk between them or with ACC, and the read is exact (not a noisy unbind). Use it to stash a partial
result, build something else, then bring the stashed value back.

```python
prog = [("LOAD", "a"), ("BIND", "b"), ("STORE", "R0"),   # R0 = bind(a, b), set aside
        ("LOAD", "c"), ("BIND", "d"),                     # ACC = bind(c, d)
        ("RECALL", "R0"), ("HALT", "")]                   # ACC = bind(a, b) again
acc, _ = vm.run(vm.assemble(prog))
# ACC == bind(a, b), recalled exactly -> cosine 1.0000
```

**A stack — `PUSH` / `POP`.** A permute-stack for save-and-restore and nesting: `PUSH` saves ACC, `POP`
brings the most recent saved value back (cleaned against the data atoms on the way out). Handy for nested
sub-computations where you want to preserve the outer value while the inner one runs.

```python
prog = [("LOAD", "a"), ("PUSH", ""),        # save a
        ("LOAD", "b"), ("BIND", "c"),        # do other work in ACC
        ("POP", ""), ("HALT", "")]           # restore a
acc, _ = vm.run(vm.assemble(prog))
# ACC == a, popped back -> cosine 1.0000
```

**Registers and the stack survive `run_chunked` seams.** A program too long for one structure is split into
chunks (see below), and the *full* machine state — the accumulator **and** the register file **and** the
stack — is threaded across each chunk boundary in its exact representation. So a `STORE R0` in an early chunk
is readable by a `RECALL R0` many chunks later (cosine 1.0), and `PUSH`/`POP` span seams too. The exact carry
is deliberate: it adds no crosstalk at a boundary, where bundling the register file into one vector *would*
(and would compound over a long program).

**The whole state as one composable vector — `state_to_vector` / `state_from_vector`.** When you *do* want
the state as a single value — to snapshot a paused computation, store it in memory, compose it, or resume it
later — bundle it: `snap = vm.state_to_vector(acc, regs, stack)` packs ACC + the register file + the stack
into one continuation hypervector, and `vm.state_from_vector(snap, reg_names=[...], codebook=[...])` reads it
back. Atom-valued slots round-trip exact after cleanup; arbitrary continuous values are lossy and the raw
readback degrades as more slots are packed (~1/√slots — the capacity cliff). So the continuation is for
snapshot/compose/resume; the per-seam carry above stays exact. VSA-native where composability is the point,
exact where exactness is.

## Calling the mind's faculties: `APPLY` + handlers

`APPLY g` runs a real leCore faculty on ACC — but the *bare* VM has no faculties, so the host (your
code, or the mind) supplies them as a `handlers` dict mapping a faculty name to a unary `acc -> acc`
function. An `APPLY` whose faculty has no handler is a **safe no-op**, so a program always runs.

```python
fm = HoloMachine(dim=512, seed=7, data=["a", "b", "c"], faculties=["cleanup"])
codebook = {n: fm.data_atoms[n] for n in ("a", "b", "c")}
def cleanup_handler(acc):                       # a real cleanup: snap ACC to the nearest known atom
    return max(codebook.values(), key=lambda v: cosine(acc, v))

acc, _ = fm.run(fm.assemble([("APPLY", "cleanup"), ("HALT", "")]),
                init_acc=noisy_a, handlers={"cleanup": cleanup_handler})
# noisy_a had cosine 0.90 to atom 'a'; after APPLY cleanup -> cosine 1.00.
# With NO handler supplied, APPLY is a no-op and ACC is unchanged (still 0.90).
```

This is the seam between a VSA program and the engine: wire `handlers={"denoise": mind.denoise, "cleanup":
mind.cleanup, ...}` and your program can invoke the mind's measured faculties on its accumulator.

### Registering faculties on the mind: `register_apply_handler`

When you run programs *through the mind* (`mind.run_procedure(...)`), you don't pass a `handlers` dict each
time — you **register** them once and they are available to every program. `mind.register_apply_handler(name,
fn)` takes any unary `acc -> acc` closure and makes it callable as `APPLY <name>`. The closure may *capture
state*, which is what lets stateful faculties — an octree query, a Nystrom approximation over a fitted
landmark set, an `Agent`'s decision — become programmable steps:

```python
# a Nystrom landmark projection (fast approximation in a large scene), captured in a closure
B = landmarks / np.linalg.norm(landmarks, axis=1, keepdims=True)
mind.register_apply_handler("nystrom_approx", lambda acc: (lambda r: r/np.linalg.norm(r))((B @ acc) @ B))

# an agent behaviour: ACC is a state, the faculty returns the agent's learned action vector
ag = mind.agent(["grab", "lift", "place"]); ag.reward(some_state, "lift", 1.0)
mind.register_apply_handler("agent_act", lambda acc: ag.action_vec[ag.decide(acc)["action"]])

# now a VSA program can call them inline, and they CHAIN with the built-ins:
out, _ = mind.run_procedure([("APPLY", "nystrom_approx"), ("APPLY", "cleanup"), ("HALT", "")], init_acc=x)
# APPLY <name> inside a program == calling the faculty directly (measured cosine 1.0); APPLY agent_act
# picks the learned action 'lift'. This is the bridge from "the agent drives a program" to "a program
# drives the engine".
```

Honest constraint: `APPLY` is **unary** `acc -> acc`. A faculty that isn't vector-to-vector (the
`DriveSystem` *scheduler*, or a multi-argument op) can't be a bare handler — wrap it behind a closure that
fixes the extra arguments (as `agent_act` fixes the goal/allowed set), or keep it in the host loop. A
registered name overrides a built-in of the same name; non-callables are rejected.

## The faculty catalogue a program can compose (recent additions)

A program reaches the mind's faculties two ways. **(1) As an `APPLY` handler** — any faculty shaped
`acc -> acc` (cleanup, denoise, recognize, the energy/resonator steps) can be wired into `handlers` and
called inline on ACC. **(2) As host orchestration** — faculties that work on grids, positions, or encoders
(the fluid/particle/fractal layer) are called directly on the mind in your host code, with the program
handling the symbolic/decision part. Both stay on the one bind/bundle/cleanup substrate, so a faculty's
*output is a vector* you can feed straight back in as `init_acc`, a `data` atom, or a `STORE`d register.

The capabilities added recently, grouped by what they do:

**Structure encode/decode** — `encode_record(fields)` / `decode_record(vec, schema)` bundle and read back a
flat `{field: value}` record; **`encode_pairs(keys, values)`** is the batched primitive (one FFT bundle of
`bind(key_i, value_i)` over parallel arrays — *renamed from a former `encode_record(keys, values)` overload
that shadowed the record encoder*, so update any call that used the two-argument form).

**Fractal / inception (a whole self-similar volume as one vector)** — `fractal_volume(enc, period, counts,
levels, motif=…)` tiles ANY VSA object self-similarly: a synthesized fractal grain (default), a field
(`motif_grid=…`), or any hypervector (`motif=…`) — including another `fractal_volume`'s output, i.e.
inception over the engine itself. **`inception(enc, period, counts, depth)`** is that as a one-parameter
depth knob and returns `(volume, profile)`, the profile a measured table of how per-copy read falls as depth
grows (the honest capacity ceiling).

**3-D fields & immersed boundary** — `fluid_step_3d` / `smoke_step_3d` (the FFT Stable-Fluids solver in 3-D)
now take `solid=` for an obstacle the flow goes around; build the obstacle with **`sphere_mask`** and enforce
it with **`enforce_solid_3d`** (the 2-D immersed boundary, lifted).

**Particle ↔ 3-D-field coupling** — **`sample_field_3d`** reads a 3-D field at continuous positions and
**`scatter_to_field_3d`** is its exact adjoint (imprint values back onto the grid); **`drag_force_3d`** is the
fluid→body force, so a softbody couples to the 3-D fluid exactly as in 2-D (`external_force=drag_force_3d(…)`).

**Particles & the cull primitive** — **`spatial_hash_pairs(positions, radius)`** returns every close pair via
a vectorised cell list (sort + searchsorted, O(N log N + pairs), ~18× faster than the old loop and matching
brute force exactly); **`pairwise_repulsion(positions, radius, strength)`** uses it for a short-range n-body
force accumulated as a scatter, passed to `particle_system.step(force=…)` like `attractor_force`/`drag_force`.

**Softbody self-collision** — a cloth/solid from `cloth3d`/`soft_box` can call `add_self_collision(radius)` so
its non-bonded nodes repel within the radius (the same spatial-hash cull, accumulated as a scatter), keeping
the sheet from passing through itself.

Each is VSA-native — FFT *is* bind, scatter *is* the adjoint of sample, a splat scene *is* a bundle — and
composable: the output of any of them is a vector or field you can route into the next, or into a
`HoloMachine` program.

## Ephemeral by design (the scientist's use case)

Because a program is just a vector, you make one, run it, and discard it — none of it has to become part of
leCore. A scientist can hold a dataset's items as `data=[...]` atoms, then assemble and run a different
program per experiment, reusing the same machine. Nothing is registered globally; nothing persists unless
you choose to store the program vector (which you can — it's content-addressable like any other vector).

## Honest limits (the capacity wall, kept on the record)

`HoloMachine` is real HRR, so it has HRR's finite capacity — this is measured and not hidden:

- **Program length (the drive's capacity cliff).** Instruction-decode holds ~100% up to a length that
  scales with dimension — solid through roughly **18 instructions at dim 1024** (and more at higher dim),
  then near the ~20 edge the decode turns *operand-dependent* (some 20-instruction programs decode, others
  don't) and beyond it bundling crosstalk overwhelms cleanup. The way past it is **`run_chunked`** (below) —
  *not* factoring into `define`d functions, which does **not** help (see the note in that section).
- **Nesting depth.** A program can be stored as a "file" inside a "disk" inside a disk… effectively
  unbounded when each level is clean, but bounded to ~3–4 levels when each disk also holds other files (a
  busy level corrupts a buried program after a few hops). Both numbers scale with dimension.
- **The read is noisy, the result is exact.** Reading an instruction back is a noisy unbind, but operands
  are *cleaned to exact atoms* before use — so the accumulator is built from clean atoms and the computed
  value is exact even though the program-reading step is approximate.

## Running a program past the cap: `run_chunked`

A long program — a scientist's experiment protocol, a long data-processing pipeline — outgrows one
structure. `run_chunked` runs it anyway by splitting it into ≤`chunk`-instruction pieces, each its **own
clean program vector**, and **threading the accumulator** across them. The accumulator is the only thing
that crosses a seam, exactly like a re-anchored route carries only its last clean tile.

```python
vm = HoloMachine(dim=1024, seed=7)
names = [chr(ord("a") + i % 6) for i in range(60)]
long_program = [("LOAD", names[0])] + [("BIND", names[i]) for i in range(1, 60)] + [("HALT", "")]

vm.run(vm.assemble(long_program))     # 60 instructions in ONE structure -> cosine 0.08 to the right answer (the cliff)
vm.run_chunked(long_program)          # host-threaded <=14-instr chunks    -> cosine 1.00, exact, past the cap
```

`run_chunked(program, chunk=14, init_acc=None, handlers=None, ...)` returns `(acc, trace)` just like `run()`,
with the trace the chunks' traces concatenated. Notes:

- **Keep `chunk` well under the edge.** The default 14 leaves deliberate margin below the dim-1024 ~20 edge
  (sitting *on* the edge fails for some programs — measured). Raise it at higher dim (20 is solid at dim 2048+).
- **Why not `define`/`CALL` instead?** Because it does **not** work: `CALL` pulls each sub-program out of a
  *bundled library*, and bundling several function-vectors into one library re-introduces the very cliff
  (measured: the same 60-instruction program via `CALL` collapses to cosine 0.06). Functions are still great
  for *reuse*; they are not the tool for *length*. `run_chunked`'s independent host-threaded chunks are.
- **Control flow is kept intact at seams.** A chunk never ends on `IFMATCH`/`REPEAT` (the gate/repeat and the
  instruction it targets stay together), but don't rely on a single construct spanning a boundary beyond that.
  Put `HALT` at the end (a trailing one is stripped; a mid-program `HALT` stops the whole run).

## API summary

```python
vm = HoloMachine(dim=4096, seed=7, data=[...], faculties=[...])  # data/faculty names define the codebooks
vm.define(name, program)                 # embed an ACC->ACC function, callable by CALL/ITERATE/REPEAT
program_vec = vm.assemble(program)       # list of (opcode, operand) -> ONE vector
acc, trace = vm.run(program_vec,         # execute; returns (accumulator, decoded-instruction trace)
                    init_acc=None,        #   start ACC at a given vector (e.g. the current NPC state)
                    handlers={...},       #   faculty name -> unary acc->acc function (for APPLY)
                    stop=None,            #   predicate acc->bool to end an ITERATE early (goal reached)
                    branch_tol=0.5,       #   IFMATCH similarity threshold
                    max_steps=512)        #   safety cap on total instructions executed
acc, trace = vm.run_chunked(program,     # run a program TOO LONG for one structure: thread the accumulator
                    chunk=14)             #   across clean <=chunk-instruction pieces (default 14; raise at higher dim)
```

`HoloMachine` lives in `holographic_machine.py`. It is intentionally *adjacent* to the mind, not a faculty
of it — the mind exposes primitives; `HoloMachine` is how you compose your own program out of them.
