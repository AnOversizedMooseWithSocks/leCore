# liblecore benchmark notes

These measurements characterize the ABI-0 preview; they do not define the API or change backend dispatch. In
particular, `LECORE_BACKEND_AUTO` remains mapped to the direct reference backend until representative adopters
provide replayable workloads and a cross-platform threshold policy is approved.

## First adapter-backed baseline (historical)

The table below records bind latency from a release build on an Apple M-series arm64 host running macOS 26.3.1,
Apple Clang 17, Python 3.14.5, and NumPy 2.4.4. Values are medians of five samples, in microseconds per call. The
original benchmark reused contexts and output buffers but timed the checked Python `Context.bind` adapter as well
as the foreign-function call. These values remain useful historical evidence, but the CI gate described below
times the public C ABI directly and must not be compared with this table. Inputs are deterministic unit vectors.

### f64 bind

| Dimension | Direct | Radix-2 | Radix-2 speedup | NumPy FFT | Max radix-2 error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 5.87 | 5.04 | 1.16x | 6.55 | 1.11e-16 |
| 16 | 5.11 | 5.06 | 1.01x | 6.33 | 2.22e-16 |
| 32 | 5.15 | 5.27 | 0.98x | 6.65 | 1.11e-16 |
| 64 | 6.28 | 5.39 | 1.16x | 6.76 | 2.22e-16 |
| 128 | 12.12 | 6.20 | 1.95x | 7.25 | 1.53e-16 |
| 256 | 37.37 | 7.64 | 4.89x | 7.94 | 1.67e-16 |
| 512 | 159.65 | 11.11 | 14.36x | 10.04 | 1.39e-16 |
| 1024 | 678.19 | 18.53 | 36.60x | 13.85 | 1.67e-16 |

### f32 bind

| Dimension | Direct | Radix-2 | Radix-2 speedup | Max radix-2 error |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4.95 | 4.92 | 1.01x | 5.96e-08 |
| 16 | 4.86 | 4.93 | 0.99x | 1.19e-07 |
| 32 | 5.01 | 5.24 | 0.95x | 8.94e-08 |
| 64 | 6.26 | 5.37 | 1.17x | 8.20e-08 |
| 128 | 12.01 | 6.10 | 1.97x | 7.45e-08 |
| 256 | 37.47 | 7.66 | 4.89x | 1.19e-07 |
| 512 | 158.08 | 11.20 | 14.12x | 8.94e-08 |
| 1024 | 673.44 | 18.41 | 36.58x | 1.04e-07 |

At dimension 1024, reported caller scratch is 8 KiB for direct f64 and 32 KiB for radix-2 f64; f32 uses 4 KiB
and 16 KiB respectively. Context creation was about 3.3 microseconds for direct and 8.3 microseconds for radix-2
at that dimension. Radix-2's plan tables are context-owned and are not included in the scratch-byte query.

The portable radix-2 implementation clearly improves on the quadratic oracle at dimensions 128 and above on this
host. Optimized NumPy FFT was still faster at dimensions 512 and 1024, so this is evidence for a portable native
backend—not a claim that it replaces platform-tuned FFT libraries.

## Ordered-kernel optimization review

The optimization review compared this implementation with pre-optimization commit `bd8da06`. Both Apple Clang 17
Release libraries were loaded by one process, given identical deterministic inputs and iteration counts, and timed
as ten alternating candidate/base pairs. Values below are median microseconds per pre-resolved public-ABI call;
improvement is the reciprocal of the median paired candidate/base ratio.

| Profile | Dimension | Direct before | Direct now | Direct improvement | Radix-2 before | Radix-2 now | Radix-2 improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 256 | 34.05 | 6.86 | 4.97x | 3.26 | 3.18 | 1.02x |
| f64 | 512 | 161.57 | 28.96 | 5.60x | 6.83 | 6.61 | 1.03x |
| f64 | 1024 | 712.05 | 113.73 | 6.26x | 14.76 | 14.18 | 1.04x |
| f32 | 256 | 34.66 | 4.18 | 8.29x | 3.33 | 3.13 | 1.06x |
| f32 | 512 | 161.48 | 14.01 | 11.50x | 6.97 | 6.46 | 1.08x |
| f32 | 1024 | 708.79 | 62.52 | 11.36x | 14.31 | 13.43 | 1.07x |

The direct result retains the ISA's ascending reduction order; direct conformance remained bit-for-bit exact.
Radix-2 maximum absolute disagreement with direct was `1.665e-16` for f64 and `1.192e-7` for f32 in this run.
These measurements do not change AUTO dispatch.

## Reproducing

Build a shared release library, then run:

```console
python3 benchmarks/bench_liblecore.py \
  --library /absolute/path/to/liblecore \
  --output /tmp/liblecore-benchmark.json
```

The default descriptive report captures host and library metadata, setup cost, scratch requirements, iteration
counts, timings, a NumPy reference for f64, and direct-versus-radix numerical error. Its bind timing uses a
pre-resolved public C ABI call; it includes the fixed Python-to-C `ctypes` transition but not `Context.bind`
validation and lock overhead. Pass `--gate` to omit the setup and NumPy metrics and emit the stricter CI-gate
metadata.

## CI regression gate

The `Linux Release performance regression` job builds the shared library with GCC in Release mode on GitHub's
Ubuntu 24.04 runner. It pins Python 3.12 and NumPy 2.4.4, then measures f64 and f32 bind at dimensions 256, 512,
and 1024. The timed callable is resolved once per cell to the exported `lecore_hrr_bind_f64` or
`lecore_hrr_bind_f32` symbol, native context handle, and input/output pointers. Context setup, Python adapter
validation, status formatting, scratch queries, and NumPy are outside the timed region. CI passes `--gate`, so
ungated setup and application-level NumPy metrics are not collected.

Before collecting samples, the harness runs three equal-count pilots per operation and calibrates from the fastest
observed per-call result toward a 30 millisecond sample target with 25% headroom. A paired run considers both
candidate and base pilots and selects one shared iteration count. If any collected candidate or base sample is
still shorter than the target, the entire batch is discarded, the shared count is increased from the shortest
sample, and both sides are measured again. Acquisition is limited to three attempts and 1,000,000 calls per
sample; failure to reach the target aborts without writing an undersized final report. This is a symmetric
measurement-integrity retry, not a second chance for a candidate that failed the regression policy.

The final report stores the raw elapsed nanoseconds for every repeat and the iteration count used. The checker
independently recomputes each median latency and rejects a report if any of the ten direct or radix-2 samples is
shorter than 10 milliseconds. This protects the gate when an implementation gets substantially faster or a pilot
runs under transient contention; fixed iteration counts or a single slow pilot would silently turn faster final
cells back into timer-noise measurements.

On pull requests, the job checks out the exact base commit into a separate directory and builds it with the same
compiler and options as the candidate. One benchmark process loads both libraries. Each candidate sample and its
base sample share an index and pair identifier, use the same inputs and iteration count, and alternate which runs
first. The checker computes candidate/base slowdown for every aligned repeat and gates the median of those paired
ratios—not the ratio of two independently aggregated medians. The committed policy permits at most a 1.35x
median paired slowdown for both direct and radix-2 in every guarded profile/dimension. The workflow performs one
reported measurement and one enforcement pass; it does not give a failing candidate an asymmetric policy retry.

The report schema marks the measurement layer as `public-c-abi`, the mode as `pre-resolved-bind`, the metrics
scope as `gate`, and the comparison as either `paired-interleaved` or bootstrap `single`. The checker requires
those values and also verifies that:

- every expected profile and dimension is present and finite;
- every candidate and base timing summary agrees with its raw repeat data;
- paired reports have the same pair identifier, repeat count, and per-backend iteration counts;
- radix-2 remains numerically close to the direct oracle within `1e-12` for f64 and `1e-4` for f32;
- the report has the expected schema, ABI, ISA, and capabilities; and
- a real `LECORE_BACKEND_AUTO` context resolves to direct for every measured profile and dimension.

The policy intentionally does not require a fixed radix-2/direct speedup. Improving the direct backend can reduce
that ratio even when both backends improve, so candidate/base paired ratios are the regression signal. The
checker supports optional per-dimension speedup floors if a future backend contract needs them.

This pull request introduces liblecore to a base branch that does not yet contain `native/liblecore`. In that one
bootstrap case, no valid base artifact can be built, so CI deliberately runs the candidate integrity, numerical,
AUTO, and measurement-duration checks without `--baseline`; it makes no candidate/base performance claim. Once
liblecore is present on the base branch, pull requests
automatically take the candidate-versus-base path. Pushes and manual runs also use the invariant-only path because
they have no pull-request base SHA.

The checker writes median public-ABI latency, optional within-candidate speedup, numerical error, and median paired
slowdown to the GitHub job summary. The raw candidate and, when available, base JSON reports are uploaded together
as a 30-day artifact even when the policy
step fails, so a regression can be inspected without rerunning CI. The workflow uses repository read permission
only and does not post or update pull-request comments.

To reproduce the CI measurement and gate locally from the repository root:

```console
cmake -S native/liblecore -B build/liblecore-performance \
  -DLECORE_BUILD_SHARED=ON \
  -DLECORE_BUILD_TESTS=OFF \
  -DLECORE_BUILD_EXAMPLES=OFF \
  -DLECORE_ENABLE_FORMAT=ON \
  -DLECORE_ENABLE_RADIX2=ON \
  -DLECORE_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/liblecore-performance --parallel 2
python3 benchmarks/bench_liblecore.py \
  --library /path/to/candidate/liblecore.so \
  --baseline-library /path/to/base/liblecore.so \
  --dimensions 256,512,1024 \
  --profiles both \
  --repeats 10 \
  --target-work 64000000 \
  --sample-target-ms 30 \
  --gate \
  --output build/liblecore-performance/liblecore-performance.json \
  --baseline-output build/liblecore-performance/liblecore-performance-base.json
python3 benchmarks/check_liblecore_performance.py \
  --report build/liblecore-performance/liblecore-performance.json \
  --baseline build/liblecore-performance/liblecore-performance-base.json \
  --policy benchmarks/liblecore_ci_policy.json
```

Omit both benchmark-side baseline arguments and checker-side `--baseline` only when bootstrapping against a
revision that has no liblecore artifact. To reproduce the normal pull-request gate, build the base revision in a
separate directory and pass both libraries to the one interleaved benchmark command shown above.

This gate covers only f32/f64 circular-convolution bind using the direct and portable radix-2 backends on one
hosted Linux configuration. It does not yet cover unbind, cleanup, batch operations, mixed f64-query/f32-corpus
cosine, allocation/setup throughput, concurrency, WebAssembly, other compilers or operating systems, or adopter
workloads. Those require their own stable workloads and policies rather than inferred thresholds from this
microbenchmark.

Passing the gate is evidence that the two explicit backends retained their expected relationship; it is not a
dispatch decision. `LECORE_BACKEND_AUTO` remains mapped to the direct reference backend. Changing `AUTO` requires
a separate, reviewed policy change supported by measurements from supported platforms and real adopters.
