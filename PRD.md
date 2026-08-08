# liblecore — Product Requirements

*Status: active · ABI-0 implementation preview and rollout plan for a portable C implementation of leCore's
frozen vector algebra.*

---

## 1. Executive summary

`liblecore` is a small, portable C11 library that carries leCore's stable, caller-supplied-vector algebra into
native, Rust, Python, WebAssembly, and other host environments without requiring Python or NumPy at runtime.

The product is not a C rewrite of leCore. It is the narrow execution substrate defined by
[`docs/ISA.md`](docs/ISA.md): bind, unbind, involution, bundle, cosine similarity, permutation, cleanup, and the
supporting vector operations required to use them safely. The existing Python implementation and the definitional
reference in [`holographic/misc/holographic_reference.py`](holographic/misc/holographic_reference.py) remain the
semantic authority.

The primary product value is **one interoperable algebra across projects**, not a promised universal speedup.
Optimized FFT, SIMD, and platform backends are replaceable microarchitecture. They ship only after they preserve
the ISA's tolerant values and exact observable decisions and demonstrate a measured win in a named workload.

The canonical C source now lives in this repository under `native/liblecore/` and retains leCore's MIT license.
The current artifact is an unstable ABI-0 implementation preview: it is suitable for local conformance and
integration work, but it is not a published release and has not earned downstream adoption claims. Consumers will
ultimately use pinned release artifacts rather than copying neighboring repositories or relying on a machine-global
development checkout.

### 1.1 Current product state — 2026-08-08

The branch contains the complete intended ABI-0 kernel surface: direct and portable radix-2 f64/f32 HRR,
caller-owned batch and mixed scoring, the optional descriptor codec, CMake and `pkg-config` packaging,
amalgamation, C/C++ examples, strict Python bindings and differential fixtures, audited Rust bindings, WebAssembly
smoke tooling, ABI symbol guards, benchmarks, and bounded fuzz targets. The delivered artifacts and their local
validation map to backlog IDs in [`ENG.md` section 16](ENG.md#16-detailed-backlog).

This snapshot does **not** claim a release, green GitHub-hosted CI, a cross-platform `AUTO` crossover policy, or
Signal/NoSQLite integration. `AUTO` deliberately continues to select the direct backend until representative
adopter evidence earns a static dispatch rule. Publication, Tier 1 CI evidence, and all downstream migration gates
remain open.

## 2. Problem

Projects around leCore already need portions of its vector algebra in environments where Python is unavailable or
undesirable. Several independent implementations have appeared as a result:

- a double-precision FFT engine in `leos-c`;
- Signal's float32 holographic pilot memory, compiled for native and WebAssembly targets;
- asix's hosted and freestanding holographic schedulers;
- Holonet's MAP/Hadamard SIMD scorer;
- deterministic text-vector indexes in NoSQLite and Zero Grounded Literary LM; and
- an integer outer-product associative memory in NSRL.

These implementations prove demand, but they are not interchangeable. Some normalize binding while leCore does
not; some use circular convolution while others use element-wise multiplication; seed generators, data types,
dimensions, trace update rules, and persisted formats also differ. Reusing a function merely named `bind` can
therefore produce plausible but incorrect results.

The current alternatives have recurring costs:

1. **Semantic drift.** Independent ports change normalization, inverse, tie-breaking, and zero-vector behavior.
2. **Brittle integration.** Some consumers compile source directly from a sibling checkout instead of depending on
   a versioned artifact.
3. **No cross-language contract.** A vector's algebra, scalar type, dimension, generator, and format are often
   implicit.
4. **Repeated native plumbing.** C, Rust, Python, and WebAssembly projects each recreate build and FFI conventions.
5. **Unverifiable optimization.** A faster continuous result can still flip an exact cleanup or action decision.

## 3. Product vision

> **One semantic substrate, from NumPy to native.** A vector created for a declared liblecore profile can move
> between supported projects and runtimes without changing what its operations mean or silently changing a
> decision.

In the target state:

- leCore owns the written ISA, reference implementation, golden fixtures, and canonical C source;
- native projects link the same small library instead of maintaining their own generic HRR primitives;
- language bindings are thin and policy-free;
- new liblecore interchange uses a versioned profile descriptor, while existing application formats may carry the
  equivalent compatibility contract in their own metadata or a sidecar without rewriting payload bytes;
- exact decisions are replayable and auditable even when optimized continuous calculations are tolerance-based;
- hosts retain authority over domain policy, legality, persistence, and lifecycle; and
- unsupported profiles, incompatible data, and unavailable accelerators fail loudly rather than silently changing
  behavior.

## 4. Product principles

1. **Semantics before speed.** The ISA and reference suite decide correctness. Benchmarks decide whether an
   optimization is worth keeping.
2. **Architecture is not microarchitecture.** Circular convolution, zero behavior, and cleanup ties are contract;
   FFT implementation and SIMD width are replaceable details.
3. **Explicit profiles, never ambiguous verbs.** HRR, MAP, exact-index, and integer memories must not share an
   unqualified `bind` contract.
4. **The host owns policy.** liblecore calculates; Signal, CosyWorld, NoSQLite, and other hosts decide what may be
   stored, ranked, committed, or exposed.
5. **Portable baseline, optional acceleration.** A conformant C11 path is always available. Accelerators are
   additive and may be refused without losing capability.
6. **No silent compatibility guesses.** ABI, semantic profile, dimension, generator, normalization policy, and
   persisted format are explicit and versioned at their proper layers.
7. **No hot-path allocation.** A context allocates or receives workspace at initialization; vector operations do
   not allocate.
8. **Reproducible consumption.** Projects pin a source or binary artifact and digest. A developer's `~/develop`
   layout is never a production dependency.
9. **Keep the negative result.** If a backend fails conformance, loses on the target workload, or cannot amortize
   setup cost, record that outcome and keep it out of automatic dispatch.

## 5. Users and jobs to be done

### 5.1 leCore maintainer

**Job:** evolve the vector ISA once and verify every implementation against one reference.

Needs versioned semantics, differential tests, release discipline, and a clear separation between the Python
research engine and the stable native surface.

### 5.2 Native simulation maintainer

**Job:** use deterministic vector memory and ranking from C or WebAssembly without importing Python.

Signal is the leading example. The maintainer needs float32 support, fixed runtime cost, native/WASM parity,
explicit compatibility descriptors, and a shadow migration path that cannot silently change an action.

### 5.3 Rust application maintainer

**Job:** compile a small, pinned C dependency and call it through a minimal safe wrapper.

NoSQLite and CosyWorld already compile C through Cargo. They need an amalgamated source option, stable symbols,
caller-owned data, and no C-owned domain state exposed through Rust layouts.

### 5.4 Constrained or browser-runtime developer

**Job:** use the same vector operations in Emscripten/WASM builds with no dynamic loader and a small dependency
surface.

The developer needs C11/libm portability, static linking, bounded memory, no threads requirement, and no mandatory
filesystem or process APIs.

### 5.5 Research and systems developer

**Job:** compare implementations and introduce optimized or new algebra profiles without weakening existing
contracts.

The developer needs the slow definitional backend, golden vectors, benchmark fixtures, feature queries, and an
extension process that prevents experimental semantics from entering the base ABI accidentally.

## 6. Product scope

### 6.1 Target version 1 scope

- A stable C11 ABI with C++ linkage guards and symbol-visibility macros.
- Opaque contexts with explicit dimension, composite profile, validation mode, and backend; the profile fixes the
  scalar type, which remains queryable rather than independently configurable.
- `HRR_F64_V1`, the reference-compatible double-precision profile.
- `HRR_F32_V1`, the deployment profile for Signal and WebAssembly.
- The caller-supplied-vector ISA subset: raw HRR bind, standalone involution, involution-based unbind, bundle,
  cosine, permutation, and cleanup. `random_vector` and other atom generation remain explicitly outside version 1.
- Supporting normalization and dot operations, specified as utilities rather than frozen ISA instructions.
- Adopter-driven batch operations for pairwise bind, fixed-role bind, multi-key unbind, and codebook scoring; each
  enters the stable surface only after scalar and decision conformance.
- A separately named f64-query/f32-corpus cosine utility for NoSQLite; liblecore computes ordered scores while Rust
  retains document-ID ties and result authority.
- A direct definitional backend and a conformant radix-2 FFT backend.
- Exact lowest-index tie-breaking for cleanup. A future top-k utility, if admitted, orders by descending score and
  ascending index.
- Build and consumption paths for CMake, Make, Cargo, ctypes, and Emscripten.
- ABI, ISA, profile, library, and capability queries.
- Golden fixtures and differential conformance against the Python reference.
- A separately layered descriptor for new vector/trace interchange. The canonical codec is built and tested in the
  standard v1 distribution, but consumers may omit or decline to use it; opaque C structs are never serialized and
  existing consumers need not rewrite compatible persisted payloads.
- A shadow-adoption path for existing consumers.

### 6.2 Follow-on scope

- A policy-neutral associative-trace helper layered on the base algebra.
- A portable, language-neutral atom generator with its own version and golden vectors.
- A broader exact-index profile or index-management layer for NoSQLite and Zero LM where benchmarks justify
  sharing more than the v1 mixed-score utility.
- Optional platform FFT or SIMD backends selected only after conformance and workload measurements.
- A freestanding allocation profile if asix or another real consumer requires it.
- Separate MAP and integer associative-memory profiles after HRR adoption is stable.

### 6.3 Non-goals

liblecore will not:

- port `UnifiedMind`, rendering, physics, scene construction, service endpoints, or the broader Python engine;
- replace NumPy as leCore's default or required runtime;
- make C mandatory for installing or using the `leos-core` Python package;
- contain game rules, action legality, feature schemas, agent policy, schedulers, networking, or database lifecycle;
- define one supposedly universal meaning for every VSA operation;
- preserve existing implementation accidents that contradict the frozen ISA;
- promise bit-identical floating-point values across all compilers and architectures where the ISA class is
  tolerance-based;
- promise a speedup without a workload-specific baseline and an amortization measurement;
- load arbitrary generated Python or C source as part of the foundational ABI; or
- require a system-wide shared library or a sibling checkout at build or runtime.

## 7. Required product surfaces

### 7.1 C API

The public header is the primary product. It must be usable from C11 and C++, contain no exposed mutable object
layouts, use prefixed symbols, return status codes, document buffer and aliasing rules, and make thread behavior
explicit.

The canonical HRR operations use unnormalized binding. A consumer that historically normalizes a bound vector may
compose `bind` followed by `normalize` in its adapter; that policy must not change the meaning of base bind.

### 7.2 Semantic profiles and compatibility descriptors

A semantic profile fixes:

- algebra and immutable profile version;
- scalar type;
- the adopted ISA-subset version; and
- observable decision/reduction order and edge behavior.

A compatibility descriptor combines that profile with instance/interchange metadata: dimension, atom-generator
version or `caller_supplied`, application normalization/trace contract, payload format, byte order, length, and
checksum. Semantic profiles are checked before operations; the full compatibility contract is checked before data
is accepted. An unknown major version or mismatched algebra fails closed.

### 7.3 Distribution artifacts

Every release intended for downstream consumption provides:

- installed headers and CMake package metadata;
- static and shared library outputs where the platform supports them;
- a single-header/single-source amalgamation for Make, Cargo, and Emscripten;
- `pkg-config` metadata for local and system integration;
- checksums and the applicable MIT notices; and
- a machine-readable build descriptor containing source version, ABI version, compiler, target, and enabled
  features.

Downstream repositories pin an artifact version and digest. Local package discovery may accelerate development but
must not be the only reproducible build path.

### 7.4 Conformance kit

The conformance kit is a product surface, not internal test debris. It includes:

- hand-verifiable convolution identities;
- seeded vector fixtures with declared generator ownership;
- tolerant expected outputs and exact reindex/decision outputs;
- zero-vector, NaN/Inf, aliasing, invalid-input, batch-tail, and non-power-of-two cases;
- cross-language fixture readers; and
- a report format that records backend, compiler, target, maximum error, and decision agreement.

## 8. Functional requirements

| ID | Requirement | State | Acceptance summary |
|---|---|---|---|
| PR-F01 | Implement the caller-supplied-vector ISA subset | Implemented · locally verified | Bind, unbind, involution, bundle, cosine, permutation, and cleanup pass the Python definitional reference; `random_vector` is explicitly excluded until a portable generator is versioned. |
| PR-F02 | Support f64 and f32 HRR profiles | Implemented · locally verified preview | Both profiles have typed APIs, documented preview tolerances, and no implicit conversion; the stable f32 bound remains adopter-gated. |
| PR-F03 | Preserve exact decisions | Implemented · locally verified | Cleanup ties select the lowest stable index; replay fixtures agree across conformant backends. Any later top-k utility orders score-descending/index-ascending. |
| PR-F04 | Be zero-safe | Implemented · locally verified | Zero normalization, zero-sum bundle, and zero-norm cosine return the documented result without NaN or division by zero. |
| PR-F05 | Support repeated calls without allocation | Implemented · locally verified | Allocation instrumentation reports zero allocations after context initialization. |
| PR-F06 | Expose batch operations | Implemented · locally verified | Batch results conform to scalar operations and define layouts, strides, tails, and allowed aliasing. |
| PR-F07 | Identify compatibility | Implemented · locally verified | Callers can query and compare ABI, ISA, profile, dtype, dimension, backend, and feature versions. |
| PR-F08 | Fail loudly without rewriting raw semantics | Implemented · locally verified | Null, size, profile, and backend errors return documented statuses; raw mode preserves the ISA's NaN/Inf propagation and cleanup behavior, while an explicit checked boundary rejects non-finite input; unavailable acceleration is never silently substituted after an explicit request. |
| PR-F09 | Cross target boundaries | Implemented · CI pending | The same source builds and passes smoke tests on supported native and Emscripten targets. |
| PR-F10 | Keep policy outside the kernel | Implemented · locally reviewed | No base API refers to pilots, residents, hypotheses, databases, routes, or other domain concepts. |

## 9. Quality requirements

### 9.1 Correctness and determinism

- f64 continuous operations must satisfy the current ISA tolerance unless an ISA revision explicitly changes it.
- f32 tolerances must be measured, documented per operation, and tight enough to preserve the adopter decision
  corpus.
- raw arithmetic must preserve the ISA's documented non-finite propagation and cleanup decision; external-data
  adapters validate before invoking it.
- involution and permutation must be bit-exact reindexes.
- backend dispatch must be inspectable; every context reports the backend actually in use.
- reproducible builds disable unsafe reassociation and contraction assumptions by default.

### 9.2 Portability

Tier 1 targets for version 1 are:

- macOS arm64 with Clang;
- Linux x86_64 with GCC or Clang; and
- WebAssembly through Emscripten.

macOS x86_64, Linux arm64, and Windows x86_64 are Tier 2 candidates. They are advertised only after repeatable CI
and consumer/header smoke tests exist. Other targets may build the direct backend without becoming supported by
implication.

### 9.3 Performance

Performance is a gate on adoption, not on semantic existence:

- the direct backend is the correctness floor and supports the full declared dimension policy;
- optimized backends publish setup cost, steady-state cost, memory use, and crossover points;
- before shadow migration, each consumer declares its performance budget; absent a stronger product-specific
  reason, a replacement that regresses representative p95 latency by more than 10% retains the existing backend;
  and
- a backend that is never faster on its target fixture remains opt-in or is removed.

### 9.4 Security and robustness

- size arithmetic is overflow-checked;
- no operation reads or writes outside declared buffers;
- fuzzing covers descriptors and public entry points;
- sanitizers run in CI;
- no API executes supplied source, opens files, starts processes, or uses network access; and
- opaque contexts are never serialized or shared across incompatible library instances.

## 10. Adoption priorities

| Priority | Consumer | Product proof | Migration boundary |
|---|---|---|---|
| 1 | leCore Python | Semantic authority and differential harness | Optional ctypes test adapter only; NumPy stays default. |
| 2 | Signal | Real f32 HRR, native/WASM, replay and compatibility contracts | Replace generic algebra behind a shadow adapter; retain Signal encoding, action policy, legality, gossip, and trace versions. |
| 3 | NoSQLite | Reproducible Rust/C packaging and exact vector scoring | Extend its existing C build seam; preserve `holographic-hash-v1` and Rust-owned persistence. |
| 4 | CosyWorld | Advisory ranking behind an authoritative deterministic kernel | Rank only actions already declared legal; retain existing heuristic as fallback and verifier. |
| 5 | Zero Grounded Literary LM | Small C11/WASM exact memory | Preserve its public host ABI while replacing generic internal math. |
| 6 | crlplrimes | Removal of brittle sibling-source coupling | Keep exact verification authoritative and treat holographic ranking as experimental until measured. |

Holonet's MAP scorer, NSRL's integer memory, Solana programs, Node native addons, and freestanding asix builds are
not first-wave adopters. They inform later profiles without expanding version 1.

## 11. Success metrics

### 11.1 Version 1 release gates

- 100% of the declared caller-supplied-vector ISA subset passes the f64 tolerant/exact conformance suite.
- The f32 profile has published per-operation error bounds and 100% agreement on the committed decision corpus.
- C and C++ consumer smoke tests pass on all advertised native targets.
- Rust and Python test bindings pass without owning kernel semantics.
- Emscripten builds and executes the same golden suite subset.
- ASan and UBSan runs report no findings; public API fuzz targets complete their configured CI budget.
- Allocation tests demonstrate zero allocations after context construction.
- Release artifacts can be consumed without a sibling checkout or machine-global installation.

### 11.2 Adoption gates

- Signal completes a shadow replay with no unexplained action-selection, contract, native/WASM, or persistence
  mismatch before any default switch.
- At least one Rust consumer builds from a pinned amalgamation or package artifact in clean CI.
- At least two external downstream product repositories integrate the same unmodified pinned liblecore artifact and
  exercise a compatibility-gated operation; leCore's ctypes adapter does not count. A legacy backend may remain
  temporarily for shadow comparison and rollback, while actual duplicate retirement is a post-release outcome.
- Every adopter declares equivalent profile compatibility metadata in its existing contract, a sidecar, or the
  optional liblecore descriptor; compatible persisted payloads need not be rewritten.
- Representative performance reports include the old implementation, direct backend, optimized backend, setup
  cost, steady-state cost, and memory use.

### 11.3 Product outcome

Within the initial adoption set, generic HRR semantics have one canonical contract, one canonical source tree, and
one conformance kit; multiple conformant direct/FFT backends may coexist. Remaining domain-specific encoders and
policies are visibly adapters rather than accidental forks.

## 12. Roadmap

**Current position (2026-08-08):** the branch implements the Phase 0–2 ABI-0 code and packaging surfaces and has
locally exercised the native reference/optimized paths plus the Emscripten feature-on/off smoke. Phase exits are
evidence gates, however: hosted CI, including its WebAssembly lane, has not yet been observed green; no release
artifact has been published; and Phase 3 adopter work has not begun.

### Phase 0 — Contract freeze

Produce the ABI decision record, operation table, error model, profile registry, compatibility descriptor, and
golden-fixture schema. Resolve naming, dimension policy, allocation hooks, and the initial f32 tolerance through
tests rather than prose alone.

**Exit:** a reviewer can implement the API from the documents and fixtures without reading a consumer's source.

### Phase 1 — Reference library

Build the f64 direct-convolution backend, exact operations, checked validation, context lifecycle, C tests, and
ctypes differential harness.

**Exit:** the C implementation passes the frozen caller-supplied-vector subset independently of FFT or SIMD.
Any published reference preview remains explicitly ABI `0` and unstable.

### Phase 2 — Deployment profiles and packaging

Add f32, the portable radix-2 FFT backend, batch and mixed-score APIs, CMake/package exports, amalgamation, CI
matrix, sanitizers, Emscripten, and reproducible release metadata.

**Exit:** a clean C, C++, Rust, Python, and WASM consumer can build a pinned ABI-`0` beta artifact and run the
conformance smoke suite.

### Phase 3 — Shadow adoption

Integrate Signal behind an adapter and feature flag; establish replay and performance baselines. Exercise the Rust
packaging path through NoSQLite without changing its persisted encoder contract.

**Exit:** the Signal and NoSQLite evidence gates pass. A recorded negative result closes the investigation safely
but still blocks stable v1 until its prerequisite or product scope is explicitly changed.

### Phase 4 — Shared memory applications

Add the policy-neutral trace extension, CosyWorld advisory ranking, Zero LM migration, and crlplrimes dependency
cleanup where each project demonstrates a real benefit.

**Exit:** two or more products use the shared library in production-like paths while retaining deterministic host
fallbacks.

### Phase 5 — Earned extensions

Consider exact-index, MAP/SIMD, portable atom generation, freestanding, and integer profiles one at a time. Each
requires a named consumer, separate semantics, fixtures, and a measured reason to enter the product.

## 13. Product backlog epics

Detailed engineering tasks and dependencies live in [`ENG.md`](ENG.md). Product priority is:

| Epic | Priority | State | Outcome |
|---|---:|---|---|
| LC-E0 Contract and governance | P0 | ABI-0 implemented · local | One implementable ABI, profile registry, and release/version policy. Stable ABI v1 remains gated. |
| LC-E1 Reference conformance | P0 | Implemented · locally verified | The f64 caller-supplied-vector subset and supporting utilities are correct before optimization. |
| LC-E2 Portable optimized core | P0 | Implemented preview · adopter evidence open | f32/f64 radix-2 and batch profiles preserve local fixture decisions; the stable f32 bound and automatic optimized dispatch remain unearned. |
| LC-E3 Distribution and CI | P0 | Implemented · CI/release pending | Reproducible C/C++/Rust/Python/WASM consumption. |
| LC-E4 Signal shadow migration | P0 | Blocked · external adopter | Highest-value existing HRR consumer validates the product. |
| LC-E5 Rust adoption | P0 | Wrapper implemented · adopter blocked | NoSQLite proves pinned Cargo consumption and exact host-owned ordering. |
| LC-E6 Interchange | P0 | Implemented · locally verified | The optional descriptor component has immutable, portable bytes and fail-closed compatibility checks. |
| LC-E7 Memory and world integrations | P1 | Proposed | Policy-neutral trace helpers, CosyWorld, Zero LM, and dependency cleanup are earned follow-ons. |
| LC-E8 Additional profiles | P2 | Proposed | MAP, exact-index, fixed-point, and freestanding work only when earned. |

P0 defines the first usable product. P1 expands proven use. P2 is intentionally excluded from the critical path.

## 14. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Existing ports look compatible but are not | Silent recall or action changes | Treat ISA/reference as authority; migrate through adapters and decision replays. |
| Native work expands into a second leCore | Unbounded scope and duplicated policy | Keep version 1 to frozen primitives; require a named profile and consumer for extensions. |
| Near-equal float scores flip decisions | Nondeterministic observable behavior | Fix reduction order in reproducible backends, pin tie rules, and test decisions separately from values. |
| Seeded atoms differ across languages | Stored vectors cannot interoperate | Start with caller-supplied atoms; version and golden-test a portable generator before adoption. |
| A global shared library causes deployment drift | Works locally, fails in CI or production | Pin source/binary artifact and digest per consumer; make local discovery optional. |
| Performance claims outrun evidence | Complexity without user benefit | Require workload-specific regime maps and preserve fallbacks. |
| Public structs freeze implementation layouts | ABI breaks on internal change | Expose opaque contexts and fixed, sized descriptor records only. |
| Licensing contaminates the MIT kernel | Downstream reuse is constrained unexpectedly | Implement from leCore's MIT reference; use AGPL or unlicensed ports as behavioral oracles only unless provenance is resolved. |
| C release cadence couples to unrelated Python changes | Unnecessary downstream churn | Version the C ABI and native package independently from the Python package release train. |
| Optional C becomes mandatory by accident | Breaks leCore's NumPy-only promise | Keep Python packaging and tests functional without a compiler or native artifact. |

## 15. Adopted implementation decisions

| Decision | ABI-0 implementation |
|---|---|
| Canonical location | `native/liblecore/` in leCore; a standalone release artifact remains future publication work. |
| Library and symbols | Product `liblecore`, linker basename `lecore`, and public symbols prefixed `lecore_`. |
| Versioning | Native package semver plus ABI, ISA, profile, and format versions; the preview reports ABI `0`. |
| Allocation | Optional allocator callbacks at context creation; no allocations after creation. |
| Threading | Calls on a live context are single-thread-owned; separate contexts may run concurrently. Destruction may run on another thread only after calls quiesce and the configured deallocator is valid there. |
| Dimensions | Dimension is an unsigned 32-bit value; the direct backend supports every representable positive dimension subject to checked size/allocation limits, while radix-2 advertises its power-of-two requirement and dispatch never lies. |
| Binding normalization | Base HRR bind is unnormalized, matching the ISA. Adapters compose normalization explicitly. |
| Atom generation | Caller-supplied in the first release; `PORTABLE_ATOM_V1` is a separate gated deliverable. |
| Persistence | Separate descriptor/format layer; never dump opaque structs. |
| Dynamic generated kernels | Separate plugin ABI and backlog, not part of liblecore v1. |

## 16. Definition of product readiness

liblecore is ready to execute release milestone `LC-037` when every other P0 task in [`ENG.md`](ENG.md) is complete,
the release gates in section 11.1 pass on advertised platforms, Signal completes shadow parity with no unexplained
decision changes, and at least one external Rust consumer builds and passes its declared compatibility gate from a
pinned artifact. `LC-037` records the release; it is not one of its own prerequisites. Earlier `0.x` releases may
publish the reference and optimized library before these ecosystem gates are complete.

It is not necessary for every project under `~/develop` to adopt liblecore. Success is a trustworthy shared
substrate where it fits and an explicit refusal where a project's algebra, runtime, or economics make it the wrong
tool.
