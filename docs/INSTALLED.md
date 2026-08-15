# The Installed Side — manifest schema & what installs

*This document serves two audiences: anyone publishing a leCore-installed model
(e.g. the HuggingFace model cards at https://huggingface.co/staccs) needs the
**manifest schema** — the machine-readable contract every compiled program ships
with — and anyone deciding what to attempt needs the **three-column unit
taxonomy**: what installs into frozen weights, what merely requires a host shape,
and what the substrate makes impossible. Both are stated from measurement, per the
install-aware build rule in `docs/CONVENTIONS.md` (the projector's verdict is
ground truth; documents record, they do not declare).*

---

## 1. The manifest schema (F26)

Every `compile_installed()` / `NativeHoloModel` produces a manifest;
`save_manifest()` writes the JSON sidecar. **Weights are never in the sidecar** —
weights live in the weights; the sidecar carries kinds, shapes, hashes and
certificates so an installation can be *verified*, *priced*, and *summarized on a
model card* without touching a tensor.

```json
{
  "dim": 1024,
  "chain": [["LOAD","a"], ["POWER","twist^3"], ["STORE","R1"], ...],
  "log_amplification_bound": 0.0,
  "warnings": [],
  "ops": {
    "BIND:k": {
      "kind": "circulant",                  // circulant | permutation | dense
      "residual": 4.7e-16,                  // certification: max rel err, held-out inputs
      "seconds": 0.03,                      // probe cost (setup-vs-marginal, priced)
      "spec_max": 1.41, "spec_min": 0.62,   // conditioning (circulants only)
      "payload": {
        "field": "column",                  // column (D floats) | perm (D ints) | matrix (D^2)
        "shape": [1024],
        "sha256": "9f2c1a..."               // bit-level integrity (hashlib, first 16 hex)
      },
      "quant": {                            // fp16/bf16 round-trip error of the payload --
        "fp16_max_err": 6.1e-05,            // fp16 installation is a checked claim
        "bf16_max_err": 4.9e-04
      }
    }
  }
}
```

Field semantics, and what a model card should say about each:

| Field | Meaning | Model-card guidance |
|---|---|---|
| `kind` | The certified parameterization. `circulant` = D floats (bind family), `permutation` = D ints (shift family), `dense` = D² (generic affine). Detection is **most-specific-first** (a roll is both; the cheap form wins). | State the kinds and parameter counts — they are the interpretability story. |
| `residual` | Max relative error of the installed form vs the live function on **held-out** random inputs. This is also the bound on any sub-tolerance nonlinearity that could have been smuggled through certification. | Quote it. `< 1e-10` means the layer *is* the operation. |
| `spec_max` / `spec_min`, `log_amplification_bound` | Conditioning: circulant spectrum magnitude range, and the chain's worst-case log-amplification walked **per step**. Deep non-unitary chains explode (measured: 1e8 at depth 64, 7.8e82 at 256); bound > ln(1e6) adds a warning naming the fix (`unitary=True` bake: depth-256 error 6e-15). | If `warnings` is empty, say so. If not, the card must carry the warning verbatim. |
| `payload.sha256` | Bit-level content hash of the payload actually shipped. | Publish it; installation verifies against it. |
| `quant` | fp16/bf16 round-trip error of the payload. End-to-end conformance measured at fp16: cosine 0.99999998. | State the precision the weights are published at and the corresponding error. |

Conformance itself (`verify_conformance`) is three-referee: the VM (holographic
decode), the installed chain, and a symbolic interpreter — with **instrument
validity checked first** (a decode-limited VM run is flagged, never miscounted as
a disagreement). A model card should state: *"conformance: installed == symbolic,
atol 1e-5"* and the dim it was verified at.

## 2. What installs — the three-column taxonomy (F29)

The 17-unit machine model (`holographic/misc/holographic_machinemodel.py`) maps
leCore's execution units onto GPU-architecture names. The install question is
orthogonal to the unit's job, so it gets its own columns. **INSTALLS** = becomes
frozen-weight arithmetic via the projector (matvec / permutation / spectral
power). **HOST-SHAPE** = works installed *only if* the host provides a structural
affordance (recurrent state, a token loop, a routing mechanism) — the unit is
runtime control that a host architecture can emulate. **SUBSTRATE-IMPOSSIBLE** =
refused by measurement or by theorem; stays runtime, callable via APPLY (T3).

| Unit | GPU name | Verdict | Why (measured / theorem) |
|---|---|---|---|
| `simd_lanes` | ALU / SIMD | **INSTALLS** | Elementwise linear maps are matrices by definition; nonlinear elementwise refuses (probe: abs → refused, residual 1.5). |
| `simt_width` | warp / SIMT | **INSTALLS** | Superposed carry is bundling — scaled-identity accumulation; readout is a matvec. Capacity 1/√K is the physics, installed or not. |
| `texture_unit` | texture sample | **INSTALLS** | Bake-once-sample is a fixed linear read over baked coefficients. |
| `gather_unit` | gather | **INSTALLS** | The original proof: measured into T @ r at cosine 1.000000 on the live stream — probing *is* projection. |
| `kernel_fusion` | shader fusion | **INSTALLS** (linear bodies) | Composition of certified linear ops is one matrix; the compiler already fuses REPEAT to an operator power. Nonlinear stages split the fusion at the refusal boundary. |
| `operator_power` | tensor core | **INSTALLS** | Spectral power of a circulant is exact (FFT diagonalizes); n matvecs → one. |
| `rt_core` | ray/scene intersect | **HOST-SHAPE** | Traversal is data-dependent branching — needs the host's loop; per-node tests install. |
| `rng` | counter-based RNG | **HOST-SHAPE** | Hash arithmetic is fixed-function integer work, not a float matvec; a host with integer ops can carry it, a matmul stack cannot. |
| `scheduler` | wave scheduler | **HOST-SHAPE** | Scheduling *is* control flow; the token loop is the schedule an installed program gets. |
| `occupancy_gate` | skip idle work | **HOST-SHAPE** | Gating is a data-dependent branch; hosts with routing (MoE-style) can express it; frozen matvecs cannot. |
| `t0_compiled` | registers | **INSTALLS** | A held compiled operator is *literally* what the projector emits. |
| `t1_margin_cache` | L1 cache | **HOST-SHAPE** | Hysteresis = stateful comparison; needs recurrent state (the register file pattern). |
| `t2_baked_grid` | L2 / baked | **INSTALLS** | Same shape as texture_unit: fixed linear read over baked data. |
| `t3_content_addressed` | L3 shared | **HOST-SHAPE** | Content addressing = nearest-neighbour over a store; the *scoring* installs (matvec), the *argmax + fetch* is control. |
| `t4_compressed_ram` | compressed RAM | **SUBSTRATE-IMPOSSIBLE** | Entropy coding is bit manipulation on variable-length streams — not expressible as fixed-shape linear algebra at any tolerance. Stays runtime (APPLY). |
| `t5_cold_store` | paging | **SUBSTRATE-IMPOSSIBLE** | I/O. Weights do not do I/O. |
| `t6_durable` | disk, verified | **SUBSTRATE-IMPOSSIBLE** | I/O + hashing; the *manifest* carries the hashes instead — that is this unit's installed shadow. |

Three regularities the table makes visible, worth stating once: **every pure
linear read installs**; **every data-dependent branch is host-shape** (control is
the shell, and the host's loop/routing is where the shell lives); **every
bit-manipulation or I/O unit is substrate-impossible** — and each impossible unit
has an *installed shadow* (compression → the rule-sized model file; durability →
the manifest's hashes) where the same goal is met by different means.

## 3. Substrate walls (unchanged by any of this)

Bundling SNR ~ 1/√n; the float32 write-accumulation cliff; decode capacity
(program length × SNR vs dim — measured live by the fuzzer: HALT itself can fail
to decode at dim 256 on long programs, and `verify_conformance` flags exactly
that). The installed side inherits every one of these; nothing here claims
otherwise, which is why the claims that *are* made can afford to be exact.

## 4. Mixed chains (G9/G13): `host_apply` marking

With `compile_installed(host_fallback=True)`, a refused faculty compiles as a marked
`HOST:APPLY` link instead of a dead end: `ops["HOST:<name>"] = {kind: "host_apply",
residual: <the refusal's residual>, seconds: ...}`. The chain stays one program; the
manifest states exactly which links are weights and which are runtime. A model card
for a mixed chain must list the host links by name — they are the part of the
program the weights do NOT carry.
