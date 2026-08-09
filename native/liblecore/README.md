# liblecore

*Portable C11 kernels for leCore's caller-supplied-vector algebra. Status: ABI-0 preview.*

`liblecore` makes the small, frozen HRR algebra in leCore usable from C, C++, Rust, WebAssembly, and other hosts
without importing Python. It is a semantic interoperability layer first: the direct backend follows
[`docs/ISA.md`](../../docs/ISA.md) and the definitional Python implementation in
[`holographic_reference.py`](../../holographic/misc/holographic_reference.py). It does not replace `UnifiedMind`,
generate atoms, own a vocabulary, or choose application policy.

The current preview provides:

- raw circular-convolution bind and involution-based unbind for f64 and f32;
- a dependency-free radix-2 FFT backend for explicitly selected power-of-two dimensions;
- normalize, dot, cosine, bundle, involution, permutation, and deterministic cleanup;
- pairwise, fixed-role, and unbind-all batch operations;
- ordered mixed f64-query/f32-corpus cosine scoring for Rust adopters;
- caller-selected shape-only or finite-input validation;
- a fixed 96-byte little-endian descriptor and CRC64 codec; and
- static/shared CMake builds, installable CMake and `pkg-config` packages, C/C++ consumer tests, and no runtime
  dependencies beyond the platform C math library.

Native version, ABI version, the liblecore ISA subset, semantic profiles, and interchange format are separate
contracts. The `0.x` series reports ABI `0`; its names and layouts may change before adopter replay gates justify
freezing ABI `1`. On platforms with versioned shared-library identities, ABI-0 builds therefore use the native
package's `major.minor`: all `0.1.x` releases use preview SONAME/install-name `0.1`, and an ABI-breaking preview must
bump the package minor and receive a new identity. Patch releases within one preview minor remain ABI-compatible.
Starting with ABI `1`, the public ABI major is the SONAME/install-name; every stable ABI break increments both.

## Build and test

An out-of-tree build keeps generated files away from the source checkout:

```sh
cmake -S native/liblecore -B /tmp/liblecore-build \
  -DLECORE_BUILD_TESTS=ON \
  -DLECORE_WARNINGS_AS_ERRORS=ON
cmake --build /tmp/liblecore-build --parallel
ctest --test-dir /tmp/liblecore-build --output-on-failure
```

To install to a project-local prefix:

```sh
cmake --install /tmp/liblecore-build --prefix /path/to/prefix
```

Installed CMake consumers use `find_package(lecore CONFIG REQUIRED)` and link `lecore::lecore`. Consumers using
`pkg-config` use `pkg-config --cflags --libs liblecore`; the linker spelling is `-llecore`.

## Single-source amalgamation

The checked-in [`amalgamation/lecore.h`](amalgamation/lecore.h) and
[`amalgamation/lecore.c`](amalgamation/lecore.c) provide the same ABI without CMake. Both optional components
default on; define `LECORE_ENABLE_FORMAT=0` or `LECORE_ENABLE_RADIX2=0` consistently for the source and consumer to
remove one. The amalgamation's compile-time guards verify binary32/binary64 representation and declared-precision
evaluation. Before distributing a raw or cross-compiled build, the consumer must also validate IEC 60559 NaN/Inf,
round-to-nearest, and runtime evaluation behavior for the target; the CMake and vendored Rust paths perform or
require this gate. A minimal hosted build is:

```sh
cc -std=c11 -O2 -I native/liblecore/amalgamation \
  app.c native/liblecore/amalgamation/lecore.c -lm
```

The files are generated from the canonical headers and sources and must not be edited directly. Regenerate or
verify them from the repository root with:

```sh
python3 tools/generate_liblecore_amalgamation.py
python3 tools/generate_liblecore_amalgamation.py --check
```

Configured builds expose the equivalent `liblecore_amalgamation_generate` and
`liblecore_amalgamation_check` targets. Set `LECORE_BUILD_AMALGAMATION=ON` to regenerate during the default build.
Installation places the pair in `share/liblecore/source/amalgamation`, separate from the normal installed API
headers, to make their source-distribution role explicit.

## Minimal C example

```c
#include <lecore/lecore.h>

int main(void) {
    lecore_config_v0 config;
    lecore_context *context = NULL;
    const double role[4] = {1.0, 0.0, 0.0, 0.0};
    const double value[4] = {0.0, 1.0, 0.0, 0.0};
    double bound[4];

    lecore_config_init_v0(&config);
    config.dimension = 4;
    if (lecore_context_create(&config, &context) != LECORE_OK) {
        return 1;
    }
    if (lecore_hrr_bind_f64(context, role, value, bound) != LECORE_OK) {
        lecore_context_destroy(context);
        return 1;
    }
    lecore_context_destroy(context);
    return 0;
}
```

The context owns only its immutable configuration and scratch space. Vectors remain caller-owned. Calls on a live
context are single-thread-owned; distinct contexts are independent. Destruction may run on another thread after all
calls have quiesced, provided a custom allocator's deallocator is valid there. Every allocation happens during
context creation, and the numeric hot path does not allocate. Functions return `lecore_status` and never print,
abort, or set global error state.

## Semantics that matter

- Bind and unbind are raw and unnormalized.
- `permute` follows NumPy `roll`: `out[i] = input[(i - shift) mod dimension]`.
- A zero norm normalizes to the unchanged zero vector and cosine returns positive zero.
- In shape-only mode, non-finite arithmetic propagates; cosine's exact-zero branch takes precedence and cleanup
  selects the first NaN score, matching NumPy `argmax`.
- Cleanup visits candidates in ascending order and an exact finite tie selects the lowest index.
- Exact input/output aliasing is supported only where the public header says so. Partial overlap and undocumented
  batch overlap are rejected.

`LECORE_BACKEND_AUTO` currently resolves deterministically to the direct backend. A forced radix-2 context uses the
optimized backend for power-of-two dimensions and returns `LECORE_EUNSUPPORTED` otherwise. AUTO will not select it
until adopter-backed benchmarks earn and freeze a crossover rule; the preview never silently substitutes a backend.

To measure that regime on a built shared artifact without changing dispatch policy:

```sh
python benchmarks/bench_liblecore.py --library /path/to/liblecore.so
```

## Build options

| Option | Default | Effect |
|---|---:|---|
| `LECORE_BUILD_SHARED` | `OFF` | Build a shared rather than static library. |
| `LECORE_BUILD_TESTS` | top-level builds | Build native, edge, format, and installed-consumer tests. |
| `LECORE_BUILD_EXAMPLES` | top-level builds | Build minimal C and C++ examples. |
| `LECORE_BUILD_AMALGAMATION` | `OFF` | Regenerate the checked-in single-source distribution while building. |
| `LECORE_BUILD_FUZZERS` | `OFF` | Build Clang libFuzzer targets; requires sanitizers. |
| `LECORE_ENABLE_FORMAT` | `ON` | Include the descriptor/CRC codec and install its header. |
| `LECORE_ENABLE_RADIX2` | `ON` | Include the portable power-of-two FFT backend. |
| `LECORE_WARNINGS_AS_ERRORS` | `OFF` | Promote liblecore source warnings to errors. |
| `LECORE_ENABLE_SANITIZERS` | `OFF` | Enable AddressSanitizer and UndefinedBehaviorSanitizer on Clang/GCC. |
| `LECORE_ASSUME_IEC_60559` | `OFF` | Required assertion for cross builds after target floating-point validation. |

The public-API and format fuzz targets are intentionally opt-in. A bounded local smoke run is:

```sh
cmake -S native/liblecore -B /tmp/liblecore-fuzz \
  -DCMAKE_C_COMPILER=clang \
  -DLECORE_BUILD_TESTS=OFF \
  -DLECORE_BUILD_EXAMPLES=OFF \
  -DLECORE_ENABLE_SANITIZERS=ON \
  -DLECORE_BUILD_FUZZERS=ON
cmake --build /tmp/liblecore-fuzz --parallel
/tmp/liblecore-fuzz/tests/fuzz/liblecore_fuzz_public_api \
  -runs=2000 native/liblecore/tests/fuzz/corpus/public_api
/tmp/liblecore-fuzz/tests/fuzz/liblecore_fuzz_format \
  -runs=2000 native/liblecore/tests/fuzz/corpus/format
```

The product requirements, release gates, adopter strategy, and dependency-ordered backlog are in
[`PRD.md`](../../PRD.md) and [`ENG.md`](../../ENG.md).
