# holostuff benchmarks

External-baseline comparisons (BLD-2): holostuff faculties measured against the off-the-shelf tool a
practitioner would actually reach for, in the project's discipline -- the case where the standard tool **wins**
is reported, not hidden. Each script is runnable and deterministic in its comparison counts and recall (the
microsecond timings are indicative and machine-dependent).

Reachable baselines were established first (the INV-2 audit): general-purpose compression has a fair stdlib
opponent (`zlib` / `lzma`), and exact nearest-neighbour is a fair opponent for sublinear recall. Denoising,
forecasting, and classification have **no** fair in-constraints standard tool -- their usual baselines (BM3D,
statsmodels/ARIMA, sklearn) are banned dependencies -- so they are left out rather than compared to a strawman.

## bench_compression.py -- rate-distortion code vs zlib/lzma

```
python benchmarks/bench_compression.py
```

holostuff's geometry-preserving code (`quant='rd'`: consolidation/KLT + water-filling + a bit-exact rANS coder)
vs int8 scalar quantization fed to `zlib`/`lzma`, with lossless float32+zlib as the cosine-1.0 reference. All
compared at matched cosine fidelity (rd targets 0.999; int8 sits at or above it). Bits per vector, lower better:

```
  structured  N=  200: rd    780.6 (cos 0.9990)  |  int8+zlib  3853.0 (cos 1.0000)  |  int8+lzma  3874.7  -> rd WINS
  structured  N= 2000: rd    111.4 (cos 0.9990)  |  int8+zlib  3809.0 (cos 1.0000)  |  int8+lzma  3823.6  -> rd WINS
  random      N=  200: rd  17317.8 (cos 0.9990)  |  int8+zlib  3827.7 (cos 1.0000)  |  int8+lzma  3847.7  -> zlib WINS
  random      N= 2000: rd   6851.3 (cos 0.9990)  |  int8+zlib  3825.8 (cos 1.0000)  |  int8+lzma  3843.4  -> zlib WINS
```

The finding: on data with real low-rank structure (rank-8 here), rd spends bits only on the directions that
carry variance and beats general-purpose compression by ~34x at N=2000 (the shared KLT basis amortizes over the
batch, so the win grows with N). On full-rank random data there is no structure to exploit, the KLT basis costs
more than it saves, and the standard tool wins. **rd is the right choice exactly when the data is low-rank --
which the engine's stored states are, and random vectors are not.**

## bench_recall.py -- HoloForest vs exact brute-force NN

```
python benchmarks/bench_recall.py
```

The sublinear approximate nearest-neighbour forest vs an exact `items @ query` scan. recall@1 / recall@8,
comparisons per query (deterministic), microseconds per query (indicative):

```
  N=   500: brute @1 100% (   500 cmp,     67us)  |  forest @1 100% @8 100% (  467 cmp = 93%,   1332us)  ->  0.1x slower
  N=  2000: brute @1 100% (  2000 cmp,    298us)  |  forest @1 100% @8 100% (  826 cmp = 41%,   2208us)  ->  0.1x slower
  N=  8000: brute @1 100% (  8000 cmp,   1565us)  |  forest @1 100% @8 100% (  952 cmp = 11%,   3067us)  ->  0.5x slower
  N= 20000: brute @1 100% ( 20000 cmp,   5591us)  |  forest @1  97% @8  97% (  616 cmp =  3%,   2326us)  ->  2.4x faster
```

The finding: the forest matches exact recall@1 up to ~10k items and uses a small, shrinking FRACTION of the
comparisons (3% at 20k) -- the structural, sublinear win is always present. But the exact scan is a single BLAS
matrix-vector product, so fast that the forest's pure-Python traversal only overtakes it on wall-clock past
~20k items. **The forest buys sublinear *work*, which is what matters when each comparison is expensive or N is
large; against a tight BLAS loop at small N, raw wall-time favours the scan.** Both directions are on the record.
