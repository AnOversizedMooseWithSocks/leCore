# HoloC Kernel Plan

This directory is for a pure C rewrite of the architectural kernel of the
holographic neuro-symbolic system. It is not a speed-port of whichever Python
test is slow this week. The target is the invariant substrate that every useful
layer depends on.

## Decision

The highest leverage C target is:

```text
hardware-optimized VSA substrate
    -> fixed-width holographic trace memory
    -> sparse memory-neuron grid
    -> optional capacity partitioning
```

In current `holostuff` terms, that means extracting the stable core of
`holographic_ai.py`, `holographic_core.py`, and the trace/memory pieces used by
`holographic_unified.py` into a small C library. It does not mean rewriting
`app.py`, `unified_app.py`, every experiment, or the high-level Python research
surface.

## Current Implementation

This directory now builds a small static C99 library:

```sh
make -C c test
```

Implemented:

- `include/holo_core.h`
- `include/holo_trace.h`
- `include/holo_program.h`
- `src/holo_core.c`
- `src/holo_trace.c`
- `src/holo_program.c`
- `tests/test_core.c`
- `tests/test_trace.c`
- `tests/test_program.c`

The current kernel provides deterministic key generation, unitary key
generation, FFT-backed circular-convolution bind/unbind, fixed-vector batch
binding, raw weighted accumulation, normalized bundle, permute, cleanup/top-k,
additive holographic trace memory, binary trace save/load, and tests for
algebraic roundtrip, cleanup, vectorized fixed binding, trace recall,
spectrum-native trace storage, lazy real-trace materialization, reusable C-owned
action dictionaries, a core stored-program runner for `HoloMachine`'s
LOAD/BIND/BUNDLE/PERMUTE/IFMATCH/HALT subset, and snapshot parity.
Trace memory stores its canonical state as an accumulated bound-pair spectrum,
so `learn()` updates
`sum(FFT(state) * FFT(action))` directly and `recall()` can unbind without first
transforming a real trace. The real trace is materialized lazily for `.trace`,
save/load, and compatibility checks. Fixed-vector binding reuses `FFT(fixed)`
across a row stack, and `holo_action_index` owns aligned action vectors, labels,
and precomputed norms for static action dictionaries. The default build uses a
portable radix-2 FFT;
macOS can enable Accelerate/vDSP for the same bind/unbind, `bind_fixed`, and
trace-recall contracts:

```sh
make -C c test HOLO_USE_ACCELERATE=1
```

Shared `holo_engine`, `holo_trace`, and `holo_action_index` objects are
internally serialized for concurrent operations. The lock is per engine for VSA
scratch buffers and trace state, and per action index for dictionary updates and
searches. Lifecycle calls still follow ordinary ownership rules: do not destroy
or dispose an object while another thread may be inside an API call on it.
Trace APIs that can materialize lazy cached state take `holo_trace *` rather
than `const holo_trace *`, and `holo_trace_load()` stages snapshots through a
temporary trace before replacing an existing live trace.

## Python Replacement Path

The repository root now has a Makefile. On macOS it builds the Accelerate-backed
C kernel by default:

```sh
make c
make c-test
```

Existing Python experiments can opt into the C-backed `bind`, `bind_fixed`,
`weighted_sum`, `unbind`, and `HolographicMemory` replacements without changing
their imports. The `weighted_sum` primitive is the unnormalized accumulation
used by function-valued FPE bundles; ordinary symbolic `bundle()` still
renormalizes after accumulation. The `bind_fixed` replacement uses the C path
for small row stacks, where the fixed spectrum reuse wins, and leaves wider
stacks on NumPy's batched real FFT by default. Tune that cutoff with
`HOLOSTUFF_C_BIND_FIXED_MAX_ROWS`, `HOLOSTUFF_C_BIND_FIXED_MIN_CELLS`, and
`HOLOSTUFF_C_BIND_FIXED_MAX_CELLS` (default: Accelerate builds only, at most
8 rows and 256..4096 total row cells).
Scalar builds leave `bind_fixed` on NumPy's batched real FFT unless
`HOLOSTUFF_C_BIND_FIXED_ALLOW_SCALAR=1` is set:

```sh
HOLOSTUFF_USE_C=1 python benchmark_holographic.py
make benchmark-c
make experiments-c
```

Programmatic callers can install the same backend explicitly:

```python
import holographic_c
holographic_c.install(strict=True)
```

Set `HOLOSTUFF_C_STRICT=1` to fail loudly if the shared C library is missing.
Without the environment switch, `holographic_ai.py` stays NumPy-only.

`HoloMachine.run_c_basic(...)` exposes the C stored-program path for straight
VSA programs whose instructions stay inside the core algebra. Host-bound VM
features such as CALL, APPLY, ITERATE, registers, and stack operations still run
through the Python VM because they invoke Python handlers, function libraries,
or exact host state.

## Benefit Experiment

The proof experiments compare trace-store/action-recall throughput and the
new fixed-vector batch bind against the current NumPy implementation:

```sh
make -C c bench-compare PYTHON=/Users/ratimics/develop/.venvs/holostuff/bin/python
make -C c bench-compare HOLO_USE_ACCELERATE=1 PYTHON=/Users/ratimics/develop/.venvs/holostuff/bin/python
```

The workload excludes vector setup from the timed section. It measures only the
architectural hot path:

```text
store:
    trace_spectrum += FFT(state) * FFT(action)

query:
    context = unbind(trace_spectrum, query_state)
    top action = holo_action_index_search(action_index, context)
```

On Apple clang / arm64, with `pairs=8`, `actions=8`, `queries=2048`, and seven
repeats via `make c-ci-evidence`, the spectrum-native portable scalar backend
with C-owned action-index readout shows the first boundary:

| dim | C store speedup | C query speedup | accuracy |
| ---: | ---: | ---: | ---: |
| 128 | ~7.3x | ~6.9x | 1.0 / 1.0 |
| 256 | ~3.7x | ~3.3x | 1.0 / 1.0 |
| 512 | ~2.0x | ~1.8x | 1.0 / 1.0 |

That is the useful scalar boundary after making the trace spectrum-native:
stores no longer pay an inverse FFT only to have recall transform the trace
back to frequency space. Enabling the vDSP backend moves the same architectural
loop onto hardware-optimized FFTs, vector reductions, and complex spectrum
multiplication:

| dim | Accelerate store speedup | Accelerate query speedup | accuracy |
| ---: | ---: | ---: | ---: |
| 128 | ~13.8x | ~21.9x | 1.0 / 1.0 |
| 256 | ~7.1x | ~11.3x | 1.0 / 1.0 |
| 512 | ~5.0x | ~8.5x | 1.0 / 1.0 |
| 1024 | ~2.9x | ~4.7x | 1.0 / 1.0 |

That proves the right lever: the C core pays off when it owns the architectural
loop and maps the algebra to the platform FFT. The biggest query win is not just
"C instead of Python"; it is representing trace memory in the form the algebra
actually consumes.

The same ownership boundary now exists one layer up for short stored programs:
`benchmarks/bench_program.py` compares Python `HoloMachine.run()` with the C
core runner on the same encoded program vectors and reports exact accumulator
parity plus runs/second speedups.

## C Mode Test Evidence

The metrics spine can now run a small repository test slice twice: once with the
default NumPy backend and once with `HOLOSTUFF_USE_C=1` plus a strict C library
load. This is parity evidence rather than a throughput benchmark; the benchmark
rows above remain the speed evidence.

Latest local run on July 1, 2026:

```sh
make metrics-c-tests PYTHON=python3
```

Environment: Python 3.14.5, macOS arm64, C backend
`c/build/accelerate/libholoc.dylib`.

| mode | selected tests | passed | failed | skipped |
| --- | ---: | ---: | ---: | ---: |
| NumPy/default | 5 | 5 | 0 | 0 |
| C kernel | 5 | 5 | 0 | 0 |

Selected tests:

- `test_algebra_properties::test_bind_batch_and_fixed_match_scalar_bind`
- `test_algebra_properties::test_rfft_bind_recovers_under_unbind`
- `test_algebra_properties::test_permute_inverse_is_identity_exactly`
- `test_holographic_compute::test_holographic_compute_selftest`
- `test_holographic_forward::test_holographic_forward_selftest`

The generated report recorded per-test elapsed times between 0.041s and 0.180s
for NumPy/default mode and between 0.043s and 0.178s for C-kernel mode.

## Why This Kernel

The local project learnings point to the same shape:

- `../crlplrimes/docs/holonet-retrospective.md`: Layer 1, the VSA compute
  engine, is the foundation. Layer 2+ experiments are useful only when they sit
  on a clean bind/unbind/bundle/keygen substrate.
- `../crlplrimes/README.md`: the model proposes, the grounding operator
  decides, and the trace remembers. A C kernel should make the trace cheap,
  deterministic, and auditable.
- `../crlplrimes/docs/symbolic-domain-interface.md`: domain soundness belongs
  outside the model. The kernel should expose rows, ids, and scores, not bury
  verifier policy in vector math.
- `../crlplrimes/docs/next-phase-plan.md`: flat holographic traces need
  capacity management before long-running memory can be trusted.
- `../agent-orchestration-report-2026-06-05.md`: module boundaries are
  architectural. The C layer needs a narrow contract and no hidden shared state.
- `../EGREGOREGRAMMING_101.md`: memory has tiers. The substrate should support
  immediate traces, recent/consolidated traces, and durable core snapshots
  without changing the algebra.

Together, these argue for a C substrate that is small enough to verify, fast
enough to call on every decision, and explicit enough to sit under multiple
symbolic domains.

## What To Rewrite First

### 1. Vector Symbolic Algebra

Rewrite these primitives first:

- deterministic atom/key generation
- flat-spectrum/unitary key generation
- dot/cosine/norm/normalize
- bundle and weighted bundle
- bind/unbind by circular convolution
- permute/rotate
- cleanup/top-k over an aligned matrix of candidate vectors

This is the direct replacement for the architectural center of
`holographic_ai.py`. It should be boring, tight C: fixed dimensions chosen at
init, aligned buffers, explicit workspaces, no heap allocation in the hot path.

### 2. Holographic Trace Memory

Build one reusable trace type:

```text
store(state, action, weight)
    trace_spectrum += FFT(state_key) * FFT(action_key) * weight

recall(query_state)
    action_context = unbind(trace_spectrum, query_state)
    scores = holo_action_index_search(action_index, action_context)
```

This is the "the trace remembers" piece. It gives `holostuff` a single durable
mechanism for associative memory, action scoring, relation recall, and
candidate suggestion.

### 3. Memory-Neuron Grid

Above the trace, implement the CRLPLRIMES memory-neuron pattern:

```text
fast tick:
    route state to K nearest memory neurons
    read context features from their traces
    return four features to the scorer/orchestrator

slow tick:
    after a verifier or ground-truth operator confirms an action
    store bind(state, verified_action) into the active traces
    update centroids gently
```

The kernel should expose context features, not own the final policy. The neural
or symbolic scorer learns how much to trust them.

### 4. Capacity Partitioning

Flat traces have finite capacity. Add deterministic partitioning after the basic
trace works:

- fixed anchor/router partitions for small C implementation first
- later HoloTree/HoloForest-style routing if measurements justify it
- per-partition counters and fidelity estimates
- shadow rebuild/swap hooks for self-organization experiments

This should be measured with trace rows before becoming a default.

## What Not To Rewrite First

Do not start with:

- Flask apps or UI panels
- every `holographic_*.py` experiment
- `UnifiedMind` as a monolith
- text corpus loaders or NLTK paths
- scene demos, image vault UI, or long-form tour code
- a fully holographic neural network

Those are orchestration and research surfaces. They should call the C kernel
through Python bindings once the substrate is stable.

## Proposed C Layout

```text
c/
  README.md
  Makefile
  include/
    holo_core.h          # implemented: vector engine, algebra API
    holo_trace.h         # implemented: trace memory API
    holo_memory_grid.h   # next: sparse memory-neuron grid API
  src/
    holo_core.c          # implemented: scalar ops + radix-2/vDSP FFT bind
    holo_trace.c         # implemented: cached-spectrum trace + save/load
    holo_memory_grid.c   # next
  tests/
    test_core.c          # implemented
    test_trace.c         # implemented
    test_memory_grid.c   # next
  bindings/
    python/              # optional CPython extension or cffi wrapper
```

The C API should be C99-compatible, with optional platform acceleration behind
compile-time switches:

- Apple: Accelerate/vDSP for FFT, dot products, and spectrum multiply
- Linux: FFTW or portable radix-2 fallback
- SIMD: NEON/AVX paths guarded by feature checks
- baseline: deterministic scalar fallback that always passes tests

## Kernel Contract

The C layer should guarantee:

- deterministic keygen from `(seed, name_or_id, dim)`
- no hot-path allocation after init
- explicit workspace ownership
- aligned contiguous vector storage
- stable binary snapshot with magic, version, dim, dtype, and endianness
- no verifier, policy, app, or domain-specific logic inside the algebra
- row-friendly diagnostics: stored count, fidelity, route ids, capacity load,
  score margin, and checksum

The implemented core already follows the first five rules for `holo_core` and
`holo_trace`; route ids and capacity load belong to the next memory-grid layer.

## First Milestone

The first milestone is implemented:

1. `holo_engine_create(dim, seed)`
2. `holo_keygen(engine, id, out)`
3. `holo_bind(engine, a, b, out)`
4. `holo_unbind(engine, pair, key, out)`
5. `holo_bundle(engine, vectors, weights, count, out)`
6. `holo_cleanup_topk(query, matrix, labels, k, out)`
7. tests proving:
   - unit vectors stay normalized
   - bind/unbind roundtrip works for unitary keys
   - cleanup recovers noisy vectors
   - trace store/recall recovers the right action above a margin
   - saved and loaded traces score identically

The concrete API names are `holo_engine_create`, `holo_keygen`,
`holo_keygen_unitary`, `holo_bind`, `holo_unbind`, `holo_bundle`,
`holo_cleanup_topk`, and the `holo_trace_*` family.

Next: wire Python to C behind the same public semantics as
`holographic_core.py`, then port memory-neuron context extraction.

## Opinion

The C rewrite should be a substrate, not a second product. The architectural
kernel is the thing that makes every later system possible:

```text
symbols become vectors
relations become reversible bindings
experience becomes a fixed-width trace
traces become context features
verifiers decide what is true
snapshots make the mind durable
```

That is the part worth making hardware-close.
