# liblecore changelog

All notable native-kernel changes are recorded here. Native releases use their own `liblecore-v*` tag namespace and
do not share the Python package's release train.

## 0.1.0 — unreleased ABI-0 reference preview

- Added a C11 ABI with opaque contexts, caller-owned vectors, custom allocator hooks, capability queries, and typed
  status values.
- Added direct f64 and f32 HRR bind/unbind plus dense, cleanup, batch, and mixed-scoring operations.
- Added an explicitly selected portable radix-2 f64/f32 backend with precomputed plans and allocation-free calls;
  automatic selection remains on the direct oracle pending benchmark evidence.
- Added explicit shape-only and finite-input validation modes with pinned zero, non-finite, and tie behavior.
- Added an optional 96-byte little-endian interchange descriptor with CRC-64/ECMA-182 validation.
- Added static/shared CMake builds, installation exports, `pkg-config` metadata, examples, and external C/C++
  consumer tests.
- Added an explicit ABI-0 exported-symbol manifest for release and visibility drift checks.
- Added deterministic Python conformance fixtures, a strict NumPy/`ctypes` adapter, an amalgamated C distribution,
  and a reproducible direct-versus-radix benchmark baseline.
- Added bounded Clang libFuzzer targets for the numeric/configuration surface and binary descriptor parser.

This release intentionally reports ABI `0`. Adopter replay, a benchmark-backed AUTO policy, and stable ABI `1`
remain gated follow-up work.
