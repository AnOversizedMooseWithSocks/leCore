# liblecore benchmark notes

These measurements characterize the ABI-0 preview; they do not define the API or change backend dispatch. In
particular, `LECORE_BACKEND_AUTO` remains mapped to the direct reference backend until representative adopters
provide replayable workloads and a cross-platform threshold policy is approved.

## First portable-backend baseline

The table below records bind latency from a release build on an Apple M-series arm64 host running macOS 26.3.1,
Apple Clang 17, Python 3.14.5, and NumPy 2.4.4. Values are medians of five samples, in microseconds per call. The
benchmark reuses contexts and output buffers; as a Python `ctypes` driver, it includes foreign-function call
overhead. Inputs are deterministic unit vectors. Direct-backend iteration counts are reduced at large dimensions
to keep the run bounded.

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

## Reproducing

Build a shared release library, then run:

```console
python3 benchmarks/bench_liblecore.py \
  --library /absolute/path/to/liblecore \
  --output /tmp/liblecore-benchmark.json
```

The JSON report captures host and library metadata, setup cost, scratch requirements, iteration counts, timings,
and direct-versus-radix numerical error.

## CI regression gate

The `Linux Release performance regression` job builds the shared library with GCC in Release mode on GitHub's
Ubuntu 24.04 runner. It pins Python 3.12 and NumPy 2.4.4, then measures both f64 and f32 bind at dimensions 256,
512, and 1024. Each reported latency is the median of ten samples; the target work is raised to 64,000,000 so
each sample contains enough calls to reduce timer and foreign-function-interface noise. The radix-2 timing loop
always executes at least 4,000 calls per sample, including dimension 1024. On the original Apple baseline this
puts the guarded direct and radix-2 samples in roughly the 30–80 millisecond range.

On pull requests, the job checks out the exact base commit into a separate directory and builds it with the same
compiler and options as the candidate. One benchmark process loads both libraries and measures each guarded cell
as an interleaved pair: base and candidate alternate which runs first on every repeat, with identical inputs,
iteration counts, Python environment, dimensions, and work target. The committed policy permits at most a 1.35x
candidate slowdown against that same-run base for both direct and radix-2 latency in every guarded
profile/dimension. Interleaving controls short-term frequency and scheduling drift far better than running two
complete reports in sequence; it does not pretend that a hosted runner is perfectly isolated.

If the first comparison fails, the workflow repeats the interleaved pair once and applies the same policy to that
confirmation report. Only a failure that persists in the confirmation run blocks the job. This single retry
handles a transient scheduled-runner interruption without averaging it into the score, while a repeatable
slowdown or malformed report still fails. Both initial and confirmation reports are retained.

The gate also evaluates relative invariants within the candidate report. It checks that:

- every expected profile and dimension is present and finite;
- radix-2 remains numerically close to the direct oracle within the profile-specific error limit;
- radix-2 keeps a conservative minimum speedup over direct at the guarded dimensions;
- report metadata confirms the expected schema, ABI, ISA, capabilities, and interleaved comparison mode; and
- an actual `LECORE_BACKEND_AUTO` context resolves to direct for every measured profile and dimension.

For both profiles the minimum same-run speedups are 2.0x at dimension 256, 5.0x at 512, and 10.0x at 1024.
The maximum direct-versus-radix absolute delta is `1e-12` for f64 and `1e-4` for f32. These floors sit well below
the recorded baseline ratios, but still fail if radix-2 falls back to quadratic work or suffers a major relative
regression. The conformance jobs retain the tighter, operation-wide numerical contract.

This pull request introduces liblecore to a base branch that does not yet contain `native/liblecore`. In that one
bootstrap case, no valid base artifact can be built, so CI deliberately runs the
candidate invariant checks without `--baseline`. Once liblecore is present on the base branch, pull requests
automatically take the candidate-versus-base path. Pushes and manual runs also use the invariant-only path because
they have no pull-request base SHA.

The checker writes the measured score table and every applicable policy result to the GitHub job summary. The raw
candidate and, when available, base JSON reports are uploaded together as a 30-day artifact even when the policy
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
