# ISA.md — the holostuff VSA instruction-set contract

*The frozen, observable semantics of the base instructions. This is the **architecture**; the FFT / BLAS /
forest implementations are **microarchitecture** below it. This document is the contract ISA-2's conformance
suite enforces, and the one place the determinism rules are stated (the executable copy lives in
`holographic_determinism.py`, which `spectral` and `chart` cite). Grounded in the live kernel
(`holographic_ai.py`), not memory.*

---

## Why a written contract (the bind_batch lesson)

An ISA is durable only if the exact **observable** semantics of its base operations are frozen while the
implementations vary underneath — that is the whole reason x86 outlived the chips that ran it. The `bind_batch`
bug is the cautionary tale in this codebase: a microarchitecture change (batched BLAS, bit-exact to 1e-12)
flipped a creature's maze trajectory, because it changed a summation order that fed a downstream `argmax` whose
tie-break the contract never pinned. The change was numerically innocent and behaviourally fatal. The lesson is
not "never vectorize" — it is **"write down the observable decision and pin it; let the continuous numbers
vary within a stated tolerance."**

## The architecture / microarchitecture boundary

- **ARCHITECTURE (pinned EXACTLY).** The observable decision a caller depends on: *which* atom `cleanup`
  returns; that `permute` and `involution` invert exactly; that `bundle`/`cosine` return the documented value
  on the zero-vector edge. A conformant implementation must reproduce these bit-for-bit where the result is a
  decision or an exact reindex.
- **MICROARCHITECTURE (may vary within a numeric tolerance).** *How* the continuous numbers are computed: an
  FFT vs a direct circular convolution, a batched vs a looped reduction. No caller can observe the last bit of
  a reduction — only the decision it feeds — so these may differ within tolerance, **provided the decision they
  feed is unchanged.** `bind_batch` is exactly such a microarchitecture variant of `bind`.

## The one determinism rule (stated once; executable in `holographic_determinism.py`)

1. **Argmax tie-break → lowest index.** Every `argmax`-style decision (`cleanup`, recall) resolves an exact tie
   to the lowest index. This is numpy's `argmax` convention, named `argmax_tiebreak` so it is citable.
2. **Eigenvector / embedding sign → largest-magnitude entry positive.** Any eigenbasis or spectral embedding
   has each column's largest-|entry| made non-negative (`fix_eigvec_signs`), removing `eigh`'s sign ambiguity.
   *Does not* resolve the basis within a degenerate eigenspace — a documented deeper limit.
3. **Reductions feeding a decision run in a fixed, documented order.** Where a downstream decision depends on a
   summation order (the bind_batch class), the order is part of the contract and pinned by a conformance test;
   where no decision depends on it, the order is microarchitecture and free.

---

## The base instructions

Each entry: signature · observable semantics · exactness class (EXACT = bit-for-bit / a decision; TOL = a
continuous value, conformant within numeric tolerance) · edge cases.

### `random_vector(dim, rng)` — mint an atom
- **Semantics.** A fresh unit-norm vector drawn from `rng`. The atom alphabet is whatever a *seeded* rng emits.
- **Class.** EXACT given `(dim, rng_state)` — the same seed yields the same atoms, so codebooks are reproducible
  (the engine's determinism rests on this). The *values* are microarchitecture; the *sequence for a fixed seed*
  is architecture.
- **Edges.** `dim > 0`. The rng is the deterministic state, never the global numpy RNG.

### `bind(a, b)` — associate (circular convolution via FFT)
- **Semantics.** Combine two vectors into a composite dissimilar to both; **commutative**; the inverse of
  `unbind`. `bind(a,b)` followed by `unbind(·, a)` recovers `b` approximately.
- **Class.** TOL. The composite is a continuous value; an FFT and a direct convolution agree within tolerance.
  **`bind_batch(A,B)` is the vectorised microarchitecture variant** — same convolution over stacked rows,
  conformant within tolerance, but **NOT pinned bit-for-bit to the looped `bind`**, which is exactly why it was
  kept out of the tie-sensitive creature path: it can only sit in front of a decision if that decision is pinned.
- **Edges.** `a`, `b` share `dim`.

### `unbind(composite, a)` — recover the bound partner
- **Semantics.** `bind(composite, involution(a))` — recovers `b` from a composite containing `bind(a,b)`.
  Recovery is **approximate**: its fidelity is bounded by how loaded the composite is (the capacity cliff).
- **Class.** TOL. The architectural guarantee is *approximate* recovery (high cosine to `b` for clean atoms at
  modest load), not an exact value.
- **Edges.** Same `dim`; quality degrades as more is bound/bundled in — a capacity question, not an error.

### `involution(a)` — the stable inverse for unbinding
- **Semantics.** The reversal used to invert `bind`. **Exactly self-inverse**: `involution(involution(a)) == a`.
- **Class.** EXACT (a reindex/conjugation, no float drift).
- **Edges.** None beyond `dim`.

### `bundle(vectors)` — superpose (and renormalize)
- **Semantics.** Sum the vectors and renormalize; the result stays *similar* to each part (how one vector stands
  for a set). **Order-independent up to float summation order.** **Information-destroying** — there is no exact
  inverse (flagged for ISA-8's reversibility audit).
- **Class.** TOL for the continuous value. **EXACT edge:** a zero-sum bundle (e.g. `a` and `-a`) returns the
  **zero vector**, not a divide-by-zero (`return total/norm if norm > 0 else total`).
- **Edges.** Empty input is caller error; the zero-sum case is pinned above.

### `cosine(a, b)` — similarity
- **Semantics.** `dot(a,b) / (‖a‖·‖b‖)`; 1.0 identical, ~0 unrelated. Drives every recall decision.
- **Class.** TOL for the value. **EXACT edge:** if either norm is 0, returns **`0.0`** (no divide-by-zero).
- **Edges.** The zero-norm case is pinned above.

### `permute(vec, shift)` — cyclic shift (encode order/position)
- **Semantics.** `np.roll(vec, shift)` — a cyclic reindex; the result is dissimilar to the original and the
  shift is reversible (`permute(·, -shift)`). The VSA primitive for order; the stack-push for ISA-5.
- **Class.** EXACT (a pure reindex — bit-for-bit, invertible, no float error).
- **Edges.** `shift` is taken mod `dim`.

### The `cleanup` decision — nearest atom
- **Semantics.** Return the codebook atom of highest cosine to the query (`int(sims.argmax())`).
- **Class.** EXACT decision. Pinned by rule 1 (lowest-index tie-break, `argmax_tiebreak`). Two implementations
  of the similarity scan may differ in the last bit of the dot products (microarchitecture) but **must** agree
  on this index (architecture), with exact ties going to the lowest index.

---

## How this is enforced (ISA-2)

A conformance suite (`test_isa_conformance.py`, ISA-2) gives the contract teeth: per instruction, a definitional
reference implementation plus golden vectors; any implementation (the FFT `bind`, `bind_batch`, a future batched
`bundle`) must match the reference **within tolerance on TOL outputs and exactly on EXACT outputs/decisions**.
It includes a regression for the bind_batch class itself — a deliberately summation-reordered op that flips a
decision must FAIL the suite. Until ISA-2 lands, `holographic_determinism._selftest` and the spectral/chart
tests pin the determinism rules; the per-instruction golden vectors are ISA-2's job.

## What this contract deliberately does NOT freeze

Per the ISA-1 negative: only the **observable** semantics above are contract. The FFT's internal rounding, the
exact bits of a reduction no decision depends on, and the basis within a degenerate eigenspace are **not**
frozen — pinning them would mistake incidental float behaviour for architecture and block legitimate
optimization (the very `bind_batch` speed-up the contract exists to make safe).

---

## The calling convention (ISA-5 — the ABI)

`CALL f` runs the named library function `f` on the current accumulator. The convention that makes nesting and
recursion well-defined, now that the machine has registers (ISA-4) and a stack (ISA-5):

- **ACC is the argument and the return value.** A function is an ACC→ACC transform: it receives the caller's
  ACC (`init_acc`) and the value it leaves in ACC is what the caller continues with. The whole function library
  obeys this — every defined function reads its input from ACC and leaves its output there.
- **Registers and the stack are FRAME-LOCAL.** Each `CALL` runs in its own frame with a fresh register file
  (R0–R7) and a fresh permute-stack. A callee therefore **cannot corrupt the caller's registers or stack** —
  they are preserved across the call automatically (measured: a callee that overwrites its R0 leaves the
  caller's R0 bit-identical, cosine 1.000). In ABI terms every register is effectively callee-saved by
  construction: the caller need not spill anything to keep a value across a `CALL`.
- **Recursion is depth-guarded.** Self-reference (a function that `CALL`s itself, with an `IFMATCH` base case)
  recurses under a fixed depth guard (8) so a missing base case cannot run away.

### The permute-stack and its safe depth (the kept negative)

The permute-stack (`PUSH` / `POP`) is a LIFO **in the vector substrate**: `PUSH` is permute+bundle (shift the
existing items one level deeper, drop ACC on top); `POP` is cleanup+inverse-permute (the top is the only
un-permuted term). It is the explicit-stack form of recursion — e.g. reversing a sequence by pushing every
element then popping — and it runs correctly **at shallow depth**.

But it is a *holographic* stack: every level rides one bundle, so depth is bounded by crosstalk exactly like the
B8 iterated-decode cliff. **Measured safe depth: ~4–8 items at dim 1024** (LIFO recovery 1.00 to depth 4, ~0.92
at 8, ~0.48 by 16; a little deeper at dim 4096). So the permute-stack is for shallow nesting of cleanup-able
items; for arbitrary intermediates at any depth, use the registers (exact, frame-local). This is the same
capacity lesson the bundled register file taught (ISA-4): superposition buys composability and pays in a
crosstalk cliff — measure it, and keep the exact path for what needs to be exact.
