# ISA_EXTENSIONS.md — the governed bind-mode extensions

*The VSA analog of x86 + SSE/AVX/AES-NI: a minimal **base** instruction set (ISA.md) plus named, opt-in
**extensions**, each justified by a measured regime win over base `bind`. The base stays RISC; specialty
operations live in governed extensions that must earn their place. Grounded in the live modules and in fresh
measurements (this session), not memory.*

---

## The model, and the standing rule

Real instruction sets grow as **base + extensions**, never by bloating the base. holostuff already does this by
instinct — the Clifford module's own docstring states the rule ("a parallel mode whose seat is a regime where it
*measurably* beats convolution"). This document makes it policy:

> **The base kernel stays minimal. A new bind mode is an EXTENSION, opt-in, in its own module, and must earn its
> place with a measured regime win over base `bind` on real data — with its cost and kept negative stated.**

## The base/extension boundary (the principle, applied)

The debatable case is `permute`: is it base or extension? The principle decides it:

> **BASE = what (almost) every faculty uses — the `holographic_ai.py` kernel. EXTENSION = regime-specific — a
> separate, opt-in module.**

By that rule the **base instruction set is frozen as** (full semantics in ISA.md): `random_vector`, `bind`
(FFT circular convolution), `unbind`, `bundle`, `permute`, `cosine`, `involution`, and the `cleanup` decision.
`permute` is **base** — it lives in the kernel and is used across the sequence, creature, and structure
faculties for order. The three extensions below are **not** base: each is a separate module, opt-in, serving a
specific data type or task the base does not.

---

## Extension 1 — Clifford-bind (the geometric product)   `holographic_clifford.py`

- **Regime.** Geometric structure, specifically **3-D rotations** and other order-sensitive (non-commutative)
  composition. Cl(3,0): a multivector is an 8-vector; the geometric product is the bind.
- **Measured win.** **Rotation composition is EXACT and is one product.** The geometric product of two rotors
  *is* the rotor of the composed rotation, so composing then applying equals applying sequentially — measured
  error **1.1e-16**. And the product is **non-commutative**, so it captures order where base `bind` (commutative
  convolution) provably cannot. Base convolution has no way to compose two rotations and apply exactly.
- **Cost / kept negative.** Cl(d) is **2^d-dimensional** — fine for Cl(3,0) (8 numbers) but a 2^d blow-up rules
  it out as a *general* high-D substrate. Use it where the structure is genuinely rotational/geometric; base
  `bind` (one FFT) remains the efficient default everywhere else.
- **Conformance** (`test_isa_extensions.py`): compose-then-apply equals sequential application to ~1e-15; a
  rotor application is length-preserving and exactly invertible.

## Extension 2 — FPE / VFA (fractional power encoding)   `holographic_fpe.py`

- **Regime.** **Continuous / spatial values** — encode a real quantity so that *nearby values are similar*,
  with the similarity profile a kernel you DESIGN (Bochner's theorem: the kernel is the phase distribution's
  characteristic function — RBF for Gaussian phases, sinc for uniform).
- **Measured win.** The designed kernel gives a **smooth, monotone similarity falloff** over continuous offset
  (cosine **1.0 → 0.9 → 0.66 → 0.41 → 0.22 → 0.11 → 0.04** across offsets 0…3 with an RBF kernel), where
  independent random atoms for the same values have **no continuity at all** (all ≈ 0 off the diagonal). Shift
  in a coordinate is a binding; the n-D kernel factors across axes.
- **Cost / kept negative.** It is an **encoder, not a general bind** — the kernel is a design choice (the
  bandwidth), and the value is the *geometry it imposes on continuous inputs*, not a replacement for `bind` on
  discrete atoms (where random near-orthogonal atoms are exactly what you want).
- **Conformance** (`test_isa_extensions.py`): the kernel falls off monotonically with offset and stays well
  above the random-atom baseline; the peak is at the encoded value.

## Extension 3 — Tensor-product bind (outer product / MPS)   `holographic_tensor.py`

- **Regime.** **High capacity / exact unbinding** when you can afford the storage. HRR's `bind` is a
  *compressed projection* of Smolensky's tensor-product binding; this keeps the uncompressed outer product
  (and a tensor-train / MPS truncation in between).
- **Measured win.** At a load that **overloads HRR**, the tensor memory recalls cleanly where convolution drowns
  in crosstalk: at 12 pairs, D=32, **HRR recall 0.29 vs tensor recall 0.87**. The outer-product store unbinds a
  stored pair near-exactly.
- **Cost / kept negative.** The storage is **D² numbers** (vs D for convolution) — the win is bought with
  dimension. And a generic full-rank binding **cannot be MPS-compressed without losing recall**, so the
  tensor-train middle ground only helps for low-entanglement structure. The frontier is
  `HRR (D) < tensor-train (≈2rD) < full tensor product (D²)` — a storage/fidelity tradeoff, not a free win.
- **Conformance** (`test_isa_extensions.py`): at a fixed overloading load, tensor recall exceeds HRR recall; a
  single stored pair round-trips near-exactly.

---

## The new-extension proposal template (the earning-its-place bar)

A new bind mode is admitted as an extension only when every line is filled — the same discipline the three above
already pass:

1. **Name & module.** A separate, opt-in module; the base kernel is **not** touched (verify base ops are
   unchanged — `conformance_report()` still passes).
2. **Regime.** The specific data type or task it serves (and the data type where it should *not* be used).
3. **The baseline it must beat.** Base `bind` (or the relevant base op) on that regime — measured, on real data.
4. **The measured win.** A number on the real substrate showing it beats the baseline *in its regime*, with the
   regime stated. No claim without a measurement (the project's standing rule).
5. **Its own conformance test.** A test in the extension-conformance suite pinning the win and any exactness
   property (composition, round-trip, kernel shape).
6. **Cost & kept negative.** What it costs (dimension, compute, a design choice) and where it does NOT help —
   stated as loudly as the win.

If a proposed mode cannot show a measured regime win, it does not become an extension — the base stays minimal.

---

*The base is RISC and frozen (ISA.md). These three extensions are governed: each names its regime, shows a
measured win this session (Clifford 1.1e-16 exact rotation; FPE 1.0→0.04 designed kernel vs flat random atoms;
tensor 0.87 vs HRR 0.29 at overload), and carries its cost and kept negative. New modes earn in by the same bar.*
