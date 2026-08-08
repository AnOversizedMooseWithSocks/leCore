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
and direct-versus-radix numerical error. Benchmark output is observational; changing `AUTO` requires a separate,
reviewed policy change with results from supported platforms and real adopters.
