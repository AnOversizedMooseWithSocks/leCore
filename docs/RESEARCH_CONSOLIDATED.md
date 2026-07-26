# leCore — Consolidated Research Document

**All three research sweeps merged into one source. Literature current through 25 July 2026.**

---

## 0 · How to read this document

**Provenance.** This merges three separate sweeps: (1) a manual search round covering VSA/HRR arXiv, SIGGRAPH
2026, and AI mathematics; (2) an extended sweep on integer VSA, Monte Carlo PDE solvers, graphics, and
verification methodology; (3) an extended sweep on low-level foundational transplants aimed at three measured
bottlenecks. Where the rounds refined or contradicted each other, the later and better-sourced statement wins
and the correction is stated rather than quietly applied.

**Status legend**, applied to every finding:

| Tag | Meaning |
|---|---|
| **DIRECT TRANSPLANT** | Mature technique, not yet connected to VSA, attacks a named bottleneck |
| **ADJACENT** | Someone has partly made the connection already |
| **NOT TRANSFERABLE** | Violates a hard constraint — the specific one is always named |

**leCore's hard constraints** (these disqualify most ML literature): NumPy + Flask + stdlib + hashlib only in
core; no PyTorch/scipy/sklearn/autodiff; **no learned weights anywhere**; numba/CuPy/sympy opt-in only and all
tests must pass without them; fully deterministic and bit-reproducible (`PYTHONHASHSEED=0`, hashlib not
`hash()`, seeded `default_rng`); additive and backward-compatible only — existing outputs never flip; every
claim carries a baseline, a variance estimate, and loudly recorded negatives.

**The governing principle for prioritisation.** Binding, bundling and cleanup are called by every faculty in
the engine — the VM decode loop, the mesh code, the renderer, the memory tiers, retrieval. A constant-factor or
exactness improvement in one of those three primitives compounds across ~7,595 functions. **Prioritise changes
to the primitive, not features built on top of it.**

---

## 1 · The measured bottlenecks

Benchmarked on the live codebase, not recalled. Everything below is scored against these.

### Bottleneck 1 — Binding cost

| D | `rfft` | `bind` | `unbind` |
|---|---|---|---|
| 512 | 6.2 us | 21.5 us | 23.7 us |
| 1024 | 9.4 us | 30.8 us | 32.2 us |
| 4096 | 27.1 us | 89.5 us | 93.7 us |

Each bind is 2-3 real FFTs and the FFT dominates.

### Bottleneck 2 — The capacity cliff (the hardest wall in the system)

Min/max member cosine against a bundle of M random hypervectors:

| D | M=4 | M=8 | M=16 | M=32 | M=64 |
|---|---|---|---|---|---|
| 512 | 0.89 | 0.75 | 0.62 | 0.43 | **0.08** |
| 1024 | 0.96 | 0.84 | 0.72 | 0.51 | **0.19** |
| 4096 | 0.95 | 0.85 | 0.80 | 0.71 | 0.58 |

This caps a single VSA program at roughly **20-32 instructions**.

### Bottleneck 3 — Cleanup is strictly linear in codebook size

Nearest-codebook-entry by matmul at D=1024:

| K | cost |
|---|---|
| 16 | 4.6 us |
| 256 | 59 us |
| 2048 | 640 us |
| 16384 | **5,682 us** |

Every VM instruction decode pays this.

---

## 2 · Thread A — Exact and cheaper binding (Bottleneck 1)

### A1 · Number-Theoretic Transform — DIRECT TRANSPLANT (the flagship finding)

The NTT computes cyclic convolution **exactly** over Z_q with zero rounding error, using the same Cooley-Tukey
butterfly structure as the FFT but in modular integer arithmetic. It is mature and heavily engineered in
post-quantum cryptography (ML-KEM/Kyber, ML-DSA/Dilithium) and fully homomorphic encryption.

**NTT-friendly primes.** 12289 = 3*2^12+1 (NewHope/Falcon, full negacyclic NTT to n=2048); 3329 and 7681
(Kyber); 8380417 = 2^23-2^13+1 (Dilithium); **Goldilocks 2^64-2^32+1** (Plonky2/STARKs, NTT sizes to 2^32).
Full negacyclic NTT over Z_q[x]/(x^n+1) requires 2n | (q-1).

**Overflow analysis derived for leCore specifically.** For bipolar {-1,+1} vectors each convolution tap is a sum
of D products in {-1,+1}, so |tap| <= D. At D=4096 = 2^12 that needs 13 magnitude bits + sign = **14 bits: int16
is overflow-free with 8x headroom.** For integer HRR with entries bounded by +-A, |tap| <= D*A^2, so accumulator
width Q = M + ceil(log2 D) (Colbert et al., arXiv:2301.13376). For an exact modular result the prime must satisfy
q > D*A^2; **Goldilocks trivially exceeds this for any realistic HRR while also satisfying 2^32 | (q-1)** —
making it the cleanest single-modulus choice for exact D=4096 convolution. CRT splitting across smaller primes
is the standard fallback when one machine word is too small.

> **COST CAVEAT — this is an exactness win, not a proven speed win.** No published head-to-head NumPy
> NTT-vs-`numpy.fft` microsecond table exists at N=512/1024/4096. A naive pure-Python NTT is O(n^2) and vastly
> slower than `numpy.fft` (which calls C-level pocketfft); even a fully array-vectorised radix-2 NTT will likely
> run *several times slower*. **This number must be generated on the live codebase with a baseline and variance,
> and recorded honestly as a kept negative if it loses on speed.**

**Existing code to reference.** IBM `optimized-number-theoretic-transform-implementations` (C/AVX512);
`pdroalves/fft_ntt_comparison` (Python reference); `sympy.ntt` (correctness oracle; sympy is opt-in). NTTSuite
(Ding et al., arXiv:2405.11353, NYU) benchmarks seven NTT algorithms but only C++/CUDA/FPGA — reports
outperforming state of the art by 30% on FPGA, which is not a NumPy datapoint.

**Novelty.** NTT applied to HRR/VSA binding appears **unpublished**. The nearest neighbour is *"How to Build
Marcus's Algebraic Mind: Algebro-Deterministic Substrate over Galois Fields"* (arXiv:2605.21379), which contrasts
HRR's non-exact quasi-inverse against an XOR-and-shift Galois-field scheme with "bit-exact reversibility".

### A2 · Walsh-Hadamard binding — DIRECT TRANSPLANT (possibly the cleaner exactness fix)

The Hadamard convolution theorem gives `WHT(a *_d x) = WHT(a) . WHT(x)` — dyadic (XOR) convolution diagonalises
under the WHT exactly as circular convolution does under the DFT. The fast WHT is **O(D log D) in integer-only
+-1 arithmetic: no twiddle factors, no rounding, no prime, no modular reduction, no CRT.** For a binding defined
as dyadic convolution this is arguably a *simpler* exactness fix than the NTT. HTMA-Net (arXiv:2509.23103) uses
precisely this identity as multiplication-avoiding convolution.

### A3 · Alternative binding operations — mixed

From Schlegel, Neubert and Protzel, *"A comparison of Vector Symbolic Architectures"* (arXiv:2001.11797 and its
journal version):

- **MAP (Multiply-Add-Permute, Gayler) — DIRECT TRANSPLANT.** Binding = element-wise product, **O(D), no FFT at
  all**, self-inverse in the bipolar case. The cheapest possible exact binding. Price: commutative (needs a
  permutation to encode order), and it is a *different algebra*, so it must be added alongside rather than
  swapped in.
- **Binary Spatter Codes / XOR (Kanerva) — DIRECT TRANSPLANT.** Self-inverse, exact, O(D). Cost: loses the
  real-valued similarity gradients cleanup relies on — a parallel bipolar tier, not a replacement.
- **VTB (Gosmann & Eliasmith), MBAT (Gallant & Okaywe) — ADJACENT.** Both beat circular convolution on binding
  capacity in the Schlegel comparison. MBAT uses an orthonormal DxD matrix (exact inverse via transpose) but
  costs **O(D^2) per bind** — asymptotically worse than the FFT at large D, with heavy storage. VTB is
  ~O(D^1.5). Neither needs learned weights, so both are NumPy-legal.
- **FHRR phase binding (Plate) — ADJACENT.** Element-wise complex multiply, O(D), unit magnitude preserved, no
  noise accumulation. Basis of the qFHRR route (section 5).
- **GHRR** (Yeung, Zou, Imani; arXiv:2405.09689, v2 27 May 2026) generalises FHRR from U(1) scalars to U(m)
  unitary matrices: hypervectors become C^(D x m x m), binding is element-wise matrix multiply and thus
  **non-commutative**, reducing exactly to FHRR at m=1. Claims restored linear memorisation capacity even for
  non-commutative binding. **Relevance:** could encode ordered structures (opcode sequences, call stacks)
  without explicit permutation — but it is floating-point and heavier, so not an exactness win.

**Structured fast transforms.** Butterfly/Monarch/Fastfood/ACDC are the *learned-weight* generalisations —
**NOT TRANSFERABLE** as such; their unparameterised cores are the WHT/FFT butterflies already covered.
Good-Thomas prime-factor FFT and Winograd/Karatsuba are exact-arithmetic-friendly for non-power-of-2 D.

---

## 3 · Thread B — Beating the capacity cliff (Bottleneck 2, highest value)

### B1 · Sparse Superposition Codes + AMP — DIRECT TRANSPLANT, top-ranked

*The NTT-shaped finding for capacity.*

SPARCs (Barron & Joseph; Rush, Greig & Venkataramanan, *IEEE Trans. Inf. Theory* 63(3):1476-1500, 2017 /
arXiv:1501.05892) are **literally a theory of how much information a superposition of codewords can carry**,
achieving Shannon capacity on the AWGN channel with polynomial-time AMP decoding.

**The structural isomorphism is exact.** A SPARC design matrix has L sections of M columns with exactly one
nonzero per section — *identical* to leCore's L codebooks of M codewords each. A VSA bundle of role-bound fillers
**is** a noisy SPARC codeword; unbundling **is** SPARC section decoding. The Donoho-Maleki-Montanari AMP
recursion is iterative matrix-vector products plus an elementwise denoiser (soft-threshold or codebook softmax)
— **no learned weights, deterministic under a seeded RNG, NumPy-native, roughly 30 lines.**

VAMP (Xu et al., arXiv:2303.08406, 2023) extends capacity-achievability to **structured (including Hadamard)
design matrices**, dovetailing with Thread C. Rush et al. note the payoff explicitly: the fast WHT's O(N log N)
cost "allow[s] the use of much larger dictionaries (e.g., M = L = 4096) for which AMP decoding with Gaussian
matrices is infeasible."

**Gap status.** The SPARC-capacity / AMP-state-evolution transplant onto VSA *bundling capacity* has **not been
formally made.** The nearest prior art is Kleyko, Bybee, Huang, Kymn, Olshausen, Frady & Sommer, *"Efficient
Decoding of Compositional Structure in Holistic Representations,"* **Neural Computation 35(7):1159-1186 (2023)**
/ arXiv:2305.16873, which applies sparse-coding and compressed-sensing decoders plus communications-style
interference cancellation, describes these techniques as "rarely used" in HDC/VSA, and reports information rate
improving **from 1.20 to 1.40 bits/dim for smaller codebooks and from 0.60 to 1.26 bits/dim for larger
codebooks**. That is roughly **2x capacity on large codebooks from decoder changes alone** — a very strong
signal for leCore's cliff.

### B2 · Compressed-sensing unbundling — DIRECT TRANSPLANT, and leCore already ships the modules

Recovering M components from a D-dimensional superposition is sparse recovery against the codebook as
dictionary: OMP, CoSaMP, IHT, AMP/VAMP. The Donoho-Tanner phase transition gives the sharp recoverable threshold
in the (delta = n/D, rho = k/n) plane.

> **leCore already contains `cosamp` and `iht` modules that are apparently not wired into unbundling.** This is a
> near-zero-cost experiment: route the existing greedy solvers at the bundle and measure whether the ~20-32
> instruction ceiling moves.

### B3 · Linear codes for HDC — ADJACENT (peer-reviewed, directly on-topic)

Netanel Raviv, *"Linear Codes for Hyperdimensional Computing,"* **Neural Computation 36(6):1084-1120 (2024)** /
arXiv:2403.03278. Random linear codes admit **provably correct, simple** recovery algorithms to factor bundled or
bound representations — exact in the noiseless case, with noise-tolerant analog decoding flagged as open.
Finite-field linear algebra: NumPy/hashlib-friendly. This is the coding-theory route, distinct from AMP.

### B4 · Modern Hopfield / dense associative memory — ADJACENT, retrieval-cost caveat

Krotov-Hopfield F(x)=x^n gives storage proportional to N^(n-1); Demircigil et al. F(x)=exp(x) gives ~2^(N/2);
Ramsauer et al. continuous MHN retrieves in one update and is mathematically the softmax-attention rule. The
single-update rule `xi_new = X . softmax(beta X^T xi)` is a fixed matmul plus softmax — **NumPy-legal and
deterministic when the stored patterns are the codebook itself.** Exponential separation instead of raw cosine
raises effective bundle capacity.

> **Caveats.** Retrieval is still O(KD) per step (Thread C is the fix); `exp` needs a numerically careful
> reproducible implementation; and the exponential-capacity figures are **asymptotic and assume well-separated
> patterns** — real bundle crosstalk will be worse. Treat as upper bounds.

### B5 · Resonator networks — ADJACENT (already VSA-native)

Frady, Kent, Olshausen & Sommer, *Neural Computation* 32(12):2311-2331 and 2332-2388 (2020). Factorises a product
of codevectors by iterated unbind-then-cleanup across factor codebooks — **O(D * sum of codebook sizes) per
iteration instead of O(product of sizes)**. Directly relevant to the VM decode loop, which currently does
independent cleanups. Kymn et al., *"A comparative study of nonlinear cleanup rules in resonator networks,"*
*Frontiers in AI* (frai.2026.1793314, 25 June 2026) organises sign/ReLU/polynomial/softmax cleanup within a
modern-Hopfield view and compares factorisation capacity and failure modes — **the most useful recent cleanup
reference.**

> **Caveat.** Resonator networks have **no general convergence guarantee** (non-symmetric weight matrix); they
> "almost always" converge within a regime. Treat cleanup as best-effort with explicit failure detection, never
> as an exact operation.

### B6 · Residue Hyperdimensional Computing — ADJACENT

Kymn, Kleyko, Frady, Bybee, Kanerva, Sommer & Olshausen, *"Computing with Residue Numbers in High-Dimensional
Representation,"* arXiv:2311.04872 / **Neural Computation 37(1):1-37, 2024**. Unifies Residue Number Systems with
VSA: encode integer x by residues mod pairwise-coprime moduli, CRT gives uniqueness up to M = product of m_k,
using Fractional Power Encoding z(x) = z^x. **Resources scale only logarithmically with dynamic range.**

> **Exactness caveat.** The exactness lives in the residue algebra on encoded integers, *not* in the vector-space
> similarity readout. Elements are m_i-th roots of unity and decoding uses a resonator network — iterative,
> complex-valued, explicitly approximate.

Related: *"Hey Pentti, We Did (More of) It!: A Vector-Symbolic Lisp With Residue Arithmetic"* (Hanley,
Tomkins-Flanagan, Kelly; arXiv:2511.08767 / **IJCNN 2025**, doi:10.1109/IJCNN64981.2025.11229311; code at
github.com/hanleyc/residuelisp). A full stored-program symbolic computer over VSA — **directly analogous to
leCore's holographic VM** — but relies on resonator decoding, i.e. the same non-exact cleanup bottleneck.

### B7 · Capacity theory — an open gap

Clarkson, Ubaru & Yang, *"Capacity Analysis of Vector Symbolic Architectures"* (arXiv:2301.10352, IBM Research)
remains the reference: bounds on dimension needed for set-membership and set-intersection to target accuracy,
connecting VSAs to sketching and Bloom filters. **No successor was found giving capacity as a joint function of
dimension AND bits-per-dimension under quantization.** See section 9.

---

## 4 · Thread C — Sublinear and exact cleanup (Bottleneck 3)

The key question: **can a codebook be *structured* so cleanup becomes a fast transform instead of an O(K) scan?**
Yes, three ways.

### C1 · Hadamard/Walsh-structured codebook — DIRECT TRANSPLANT, top cleanup pick

If codewords are rows of a Hadamard matrix (or a randomly sign-permuted Hadamard, to break structure while
preserving the fast transform), computing **all K correlations is one fast Walsh-Hadamard transform: O(D log D)
instead of O(KD).** At D=1024, K=16384 this replaces the measured **5,682 us** matmul with a single ~D log D
transform plus argmax — an order-of-magnitude-plus projected gain that **grows with K**, exactly where the linear
scan hurts most. Integer +-1 arithmetic: exact and bit-reproducible.

This is precisely why SPARC decoders use it — Barbier & Krzakala (arXiv:1503.08040) note structured Hadamard
operators reduce decoder matrix multiplications "from O(N^2) ... to O(N ln N) and the matrix has never to be
stored in memory."

### C2 · Reed-Muller decoding = fast Hadamard transform + argmax — DIRECT TRANSPLANT

Decoding first-order Reed-Muller RM(1,m) is *exactly* a fast Hadamard transform on the 2^m received word followed
by taking the largest-magnitude component (the classic "Green machine", MacWilliams & Sloane). **Exact
maximum-likelihood nearest-codeword decode in O(D log D).** If opcode and operand atoms are RM(1,m) codewords,
every VM instruction decode becomes a fast transform rather than a K-scan.

### C3 · Lattice codebooks with fast nearest-point decoders — DIRECT TRANSPLANT (fixed dimensions)

Conway & Sloane, *"Fast quantizing and decoding algorithms for lattice quantizers and codes,"* *IEEE T-IT*
28(2):227-232 (1982). Vardy & Be'ery, *"Maximum Likelihood Decoding of the Leech Lattice,"* *IEEE T-IT*
39(4):1435-1444 (1993), achieve ML decoding of the Leech lattice in **3,595 real operations worst case, 2,955
average** — a 24-dimensional exact decode whose cost is *independent of explicit codebook size*. E8, D_n and A_n*
have linear-time nearest-point algorithms (McKilliam, Smith & Clarkson, *IEEE T-IT* 56(3), 2010).

### C4 · Syndrome and list decoding — ADJACENT

Nearest-neighbour decoding of binary linear codes (the May-Meurer-Thomae line) achieves sub-2^n complexity.
Relevant if atoms are codewords of a structured linear code. NumPy-legal.

### C5 · ANN indices — mostly NOT TRANSFERABLE

| Method | Verdict |
|---|---|
| Product quantization, transform-coding ANN (Jegou et al.) | Deterministic, NumPy-implementable, **approximate** — acceptable as opt-in accelerator only |
| HNSW, DiskANN | Deterministic only with fixed seed *and* insertion order; still approximate — **risks flipping a decision**, violating "outputs never flip" |
| LSH | Randomised, so non-deterministic unless seeded and frozen |
| ScaNN | Learned/anisotropic quantization — **NOT TRANSFERABLE (learned weights)** |

**Recommendation:** PQ and transform-coding only as an opt-in tier, never in the exactness-critical decode path.

---

## 5 · Thread D — Quantization, compression and storage

### D1 · qFHRR integer-phase representation — DIRECT TRANSPLANT (preprint)

*"qFHRR: Rethinking Fourier Holographic Reduced Representations through Quantized Phase and Integer
Arithmetic,"* Snyder, Poursiami & Parsa (George Mason University), **arXiv:2604.25939 v1, 16 April 2026.
Preprint, not peer-reviewed.**

Each dimension becomes a discrete phase index q in {0..K-1}. Binding = (q_a + q_b) mod K; unbinding =
(q_a - q_b) mod K — **exact modular integer arithmetic.** Similarity is a cosine lookup table plus integer
accumulation. Reduces from 64-bit complex to **as few as 3-4 bits per dimension.**

Author-reported fidelity versus complex FHRR:

| K | bits/dim | size reduction | bind | bundle (N=16) |
|---|---|---|---|---|
| 8 | 3 | 95.31% | 0.9497 | 0.9147 |
| 16 | 4 | 93.75% | 0.9872 | 0.9731 |
| 256 | 8 | 87.50% | 0.9999 | 0.9997 |

> **Two caveats, both load-bearing.** First: **bundling is NOT closed under the quantized representation.** The
> paper approximates it via cos/sin lookup tables to fixed-point Cartesian, accumulation, then
> `q = round((K/2pi) * atan2(I,R)) mod K` using integer CORDIC — "a hardware-efficient approximation of atan2".
> CORDIC is deterministic at a fixed iteration count, but the `round()` at a bin boundary is *itself a
> tie-arbitration point*. Second: **the paper makes no claim about bit-exactness, determinism, argmax decisions,
> or reorder-independence** — its framing is efficiency and hardware, not reproducibility. Round 3 additionally
> flagged that the per-bit fidelity figures could not be re-verified from the abstract alone; confirm against the
> paper's fidelity table before quoting them as measured.

**Net.** qFHRR gives exact *binding*, not an exact *cleanup decision*. It does **not** by itself let leCore delete
its tie-arbitration machinery. Its natural home is paths where bundling is not on the hot path — notably the
**VM decode path, which does bind/unbind plus cleanup and never bundles.**

### D2 · Asymmetric Numeral Systems — DIRECT TRANSPLANT

Duda, arXiv:1311.2540 / arXiv:0902.0271. Arithmetic-coding compression ratios at Huffman speeds; a pure
integer-table algorithm — deterministic, stdlib-implementable, already the industry lossless standard (Zstandard,
LZFSE). Reports **about 50% faster decoding than Huffman for a 256-symbol alphabet** at compression comparable to
arithmetic coding. Clean fit for storing codebooks and bundle traces losslessly. Duda's Pyramid Vector Quantizer
plus ANS is the vector-quantization variant.

### D3 · Decision-safe rate-distortion — GENUINE GAP (see section 9)

Standard rate-distortion minimises reconstruction MSE. leCore needs to preserve **argmax and cleanup decisions**
— the similarity-ranking geometry, not the vector. qFHRR is the closest empirical anchor (binding decisions
stable to about 3-4 bits); Colbert et al. (arXiv:2301.13376) supplies the accumulator-overflow half. A theory of
quantization-error propagation through bind -> bundle -> cleanup is **unpublished**.

---

## 6 · Thread E — Determinism and reproducible numerics (the deepest level)

### E1 · Reproducible summation — DIRECT TRANSPLANT, best determinism ROI

Demmel, Ahrens & Nguyen, *"Efficient Reproducible Floating Point Summation and BLAS"* (UCB/EECS-2016-121; ACM
TOMS 2020). Bitwise-identical sums **independent of summation order** via a 6-word reproducible accumulator (at
least 80 bits precision). Cost: about 9n floating-point operations plus about 3n bitwise operations; measured
**4x slowdown versus Intel MKL** on a 4096-element double dot product (Core i7-2600).

Applies directly to bundling accumulation and the cleanup matmul, guaranteeing bit-reproducibility across
microarchitectures **without integerising**. Kahan/Neumaier compensated summation is the lightweight pure-NumPy
version.

### E2 · Shewchuk adaptive-precision predicates — DIRECT TRANSPLANT

Shewchuk, *"Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates,"* *Discrete &
Computational Geometry* 18:305-363 (1997). Exact orientation and incircle predicates via adaptive expansions;
"running time depends on the degree of uncertainty of the result, and is usually small."

Hardens leCore's mesh code (Catmull-Clark, QEM, Delaunay-style operations) against non-deterministic branch
flips. **The discipline transfers directly to cleanup:** compute cosines in float, but break ties with an exact
integer or rational recomputation, so a tie can never flip across microarchitectures.

### E3 · FFT cross-microarchitecture non-determinism — CONFIRMED RISK

`numpy.fft` (pocketfft) can produce bitwise-different results across CPUs. NumPy Issue **#11926** ("FFTN
generates different results on different CPUs", stsci-hack, 10 September 2018) reports "scientifically
significant changes (up to 0.1% in image flux values) ... simply due to the CPU" — naming Xeon E5-1660 v4 versus
Xeon E7-8867 v4, attributed to SIMD vectorisation changing summation order. Corroborated by issues #14409,
#11241 and #13424. FFTW documents bitwise reproducibility only *on the same system* and only with the `ESTIMATE`
planner flag.

**This is exactly leCore's "ULP flip flips argmax" problem, confirmed in the wild — and it is the root
determinism threat to FFT-based binding.** The structural fix is Thread A: integer NTT or +-1 WHT, both exact and
order-independent. Which is why **Thread A is a determinism fix as much as a speed play.**

---

## 7 · Thread F — Graphics, Monte Carlo and signal processing

### F1 · Walk on Decomposed Subdomains — DIRECT TRANSPLANT, most transferable graphics result

*"Walk on Decomposed Subdomains: A Hybrid Monte Carlo-Deterministic Solver for Elliptic PDEs."* Clement Jambon,
Mohammad Sina Nabizadeh, Mina Konakovic Lukovic (MIT CSAIL). **ACM TOG 45(4), Article 132, July 2026,
doi:10.1145/3811340. Peer-reviewed. SIGGRAPH 2026 Best Paper.**

Decomposes the domain into simple subdomains, estimates **local solution operators** (Poisson kernels) with Monte
Carlo — small subdomains bound walk length and variance inherently — then assembles them into a **sparse global
linear system solved deterministically**, which "exactly replaces simulating discrete random walks through the
domain". Delivers "low-variance solutions orders of magnitude faster than pure Monte Carlo, without volumetric
meshing". Efficient re-solves update only the operators affected by geometry changes.

**Stated limitations.** Demonstrated in 2D; introduces a resolution-dependent **discretisation bias** (unlike
pure WoS, which is unbiased in expectation); restricted to zero-Neumann conditions.

**leCore fit — excellent.** MC kernel estimation plus a sparse solve is pure NumPy and neural-free. The
reusable-operator structure maps onto leCore's compiled-operator and baked-grid memory tiers, and the
deterministic global solve is a natural home for the determinism discipline. Its structure is also two of
leCore's own five levers stacked: *partition into a commutative monoid* and *tile the domain under an
orchestrator*.

### F2 · The WoS/WoSt lineage — all neural-free, NumPy-implementable

- Sawhney & Crane, *Monte Carlo Geometry Processing* (TOG 2020) — grid-free WoS for Poisson/Laplace
- Sawhney, Seyb, Jarosz & Crane, *Grid-free MC for PDEs with spatially varying coefficients* (SIGGRAPH 2022)
- Sawhney, Miller, Gkioulekas & Crane, *Walk on Stars* (TOG 2023) — Neumann BCs via star-shaped regions
- Miller, Sawhney, Crane & Gkioulekas, *Walkin' Robin* (TOG 43(4), 2024) — Robin BCs; reduces error "orders of
  magnitude" versus walk-on-boundary
- **Variance reduction:** boundary value caching (Miller et al., TOG 2023); mean value caching (Bakbouk & Peers,
  EGSR 2023); bidirectional formulation (Qi et al., CGF 2022); *Walking on Spheres and Talking to Neighbors*
  (arXiv:2404.17692); guiding-based importance sampling (SIGGRAPH 2025); off-centered WoS with statistical
  weighting (SIGGRAPH Asia 2025); **harmonic caching** (Zhou, d'Eon, Sawhney & Jarosz, TOG 44(6), December
  2025); robust derivative estimation (Yu et al., TOG 44(6), December 2025)
- *Gradient Domain Reconstruction for Monte Carlo PDE Solvers* (SIGGRAPH 2026 Honorable Mention) — an estimator
  targeting *differences* between query locations, then screened-Poisson reconstruction "without incurring
  additional bias"

**Deterministic-friendly.** The caching variants (boundary, mean and harmonic value caching) amortise and reuse
samples — a deterministic reuse layer matching leCore's margin-cache and content-addressed-cache tiers.

### F3 · Points as Tori — partially NOT TRANSFERABLE; the representation is the lead

*"Points as Tori: Fast Pointwise Signed Distance for Point Clouds."* Nicole Feng, Ioannis Gkioulekas, Keenan
Crane (CMU). **TOG 45(4) Article 53, doi:10.1145/3811385, 3 July 2026.** Fits point clouds locally with **tori,
which have closed-form SDFs**, giving an analytical parameterisation queryable at arbitrary resolution with "no
costly global optimization or spatial discretization, and easily parallelizable". Unifies signed distance with
winding numbers and Poisson surface reconstruction. A top-10 attendee-voted paper.

> The torus fitting is done "in a feed-forward manner using a pre-trained network" for per-point curvature and
> shift — **NOT TRANSFERABLE** under no-learned-weights. **But the theory is closed-form and could be re-derived
> with a non-learned local least-squares torus fit.** That would give leCore an analytic, resolution-independent
> point-cloud SDF feeding shrinkwrap and UV reprojection.

### F4 · Robust Planar Maps — DIRECT TRANSPLANT (philosophy and method)

*"Robust Planar Maps for 3D Vectorization."* Robert Fuchs, Keenan Crane (CMU/Roblox). **SIGGRAPH 2026 Best
Paper.** Replaces numerically difficult curve-curve intersection with tractable **curve-line** intersections,
using a spatial hierarchy as the fundamental planar-map representation; handles imperfect, disconnected inputs;
"orders of magnitude faster" than prior methods.

**The "reformulate to avoid the numerically unstable predicate" philosophy is exactly leCore's tie-band
re-arbitration mindset**, and it pairs naturally with Shewchuk (E2).

### F5 · Other SIGGRAPH 2026 hits on leCore surfaces

*Uncertainty-Aware Geometry Processing on Gaussian Process Implicit Surfaces* (Genest, Coeurjolly) ·
*Spatiotemporal FLIP* (Braun, Winchenbach, Bender, Thuerey) · *Spatio-Temporal Control Variates with ReSTIR* ·
*Implicit Minimal Surfaces for Bijective Correspondences* (Corman, Soliman, Magnet, Gillespie) · *Fast and Exact
Winding Numbers for Triangle Meshes* · *Efficient Multiscale Lanczos Eigenpair Extraction* (Braune, Dumas,
Thiery — classic numerical linear algebra, NumPy-implementable) · *Fast Sparse Matrix Permutation for Mesh-Based
Direct Solvers* · *Mixwell* (analytic grid-free 2D fluid brushes, "negligible numerical dissipation" — the same
bake-the-analytic-form instinct as leCore's carriers).

Conference context: **SIGGRAPH 2026, 19-23 July, Los Angeles; more than 1,120 submissions, a record across 53
years.** Chair Mirela Ben-Chen named the emerging themes as generative image modelling, **Monte Carlo solvers**,
and 3D vectorization.

> SIGGRAPH 2026's texture-synthesis and model-reduction papers are **overwhelmingly neural and non-transferable**.
> The neural-free by-example texture-synthesis gap in the 2026 program is real; leCore must fall back on the
> classic pre-neural line (Wei-Levoy, Portilla-Simoncelli, Lefebvre-Hoppe).

### F6 · Gaussian splatting as superposition — ADJACENT, representationally rich

A splat scene is literally a bundle of role-bound primitives. **Gaussian Wave Splatting** (Choi et al., *ACM TOG*
2025, doi:10.1145/3731163) derives a closed-form 2D-Gaussian-to-hologram transform and a wave-optics analogue of
alpha blending — an explicit bridge between superposition-of-primitives and a **hologram**, mathematically the
same object as an HRR memory trace. Gaussian Splatting Holography (THU HoloLab) does twin-image-free lensless
reconstruction from 2D-Gaussian superpositions. Implementations are CUDA and optimisation-based (**NOT
TRANSFERABLE as code**), but the identity *splat bundle is approximately a holographic memory* is a genuine
design lead.

### F7 · Denoiser-as-prior with a non-learned cleanup — DIRECT TRANSPLANT (conceptual)

RED and Plug-and-Play insert a denoiser as the implicit prior in an iterative solver. If the denoiser is
**leCore's cleanup operator** (projection onto the codebook manifold), cleanup-as-prior turns unbundling into a
PnP fixed-point iteration — precisely the resonator/AMP structure of section 3, requiring no learned network.
Training-free, deterministic, NumPy-native.

### F8 · Phase retrieval — DIRECT TRANSPLANT (conceptual)

Gerchberg-Saxton, Fienup and Wirtinger flow: recovering a hypervector from magnitude-only or degraded spectral
information is a phase-retrieval problem, and the classical alternating-projection solvers are deterministic
fixed-iteration algorithms. Maps onto FHRR unbinding when only magnitudes survive. Ozcan's lensless-holography
reconstruction-from-degraded-codes is the imaging analogue.

### F9 · Matched filtering and the SETI toolkit — DIRECT TRANSPLANT

Cleanup under heavy bundling crosstalk **is** detection of a known template in noise; the matched filter is the
optimal linear detector. The radio-astronomy machinery (incoherent harmonic summing, false-alarm thresholds —
Tarter and Siemion) maps onto setting cleanup decision thresholds under M-way superposition interference, and
dovetails with Cranmer-style false-discovery control for the argmax acceptance threshold.

---

## 8 · Thread G — AI mathematics and verification methodology

Architecturally orthogonal to the engine, but the **verification pattern** transfers directly to leCore's
kept-negatives discipline.

### G1 · The counterexample wave, February to July 2026

| Date | Event |
|---|---|
| **20 May 2026** | An internal OpenAI model **disproved the Erdos Unit Distance conjecture** — "an infinite family of examples that yield a polynomial improvement ... checked by a group of external mathematicians". Explicit bound in Sawin, arXiv:2605.20579: sets with more than n^1.014 unit-distance pairs. Human-verified exposition: Alon, Bloom, Gowers, Litt, Sawin, Shankar, Tsimerman, Wang & Matchett Wood (arXiv:2605.20695). Structure: Golod-Shafarevich (1960s) used to construct the counterexample. |
| **26 May** | Logical Intelligence (Freedman as CSO) autoformalised the paper in Lean — but **conditionally** on the deep theorem. |
| **26 June** | Boris Alexeev used OpenAI "Sol" for a complete **from-axioms** Lean formalisation: **1.2 million lines of Lean in three weeks** (mathlib is about 2.3M lines built over nine years), including real global class field theory. |
| **11 July** | Akhil Mathew with Sol found a counterexample to **Grothendieck's approximately 60-year-old SGA 3 question** — a finite flat group scheme of order 4 *not* killed by 4. "Fable" autoformalised it in about 4 hours, 1,076 lines. Buzzard verified in **under 5 minutes**: the statement uses only mathlib concepts, says what is claimed, and compiles. PR'd to mathlib. **Independently confirmed.** |
| **about 19-20 July** | Levent Alpoge with Claude Fable 5 produced a counterexample to the **Jacobian Conjecture** (open since 1939): an explicit C^3 -> C^3 map with constant Jacobian determinant -2 that is not injective. Formalised by Paul Lezeau into DeepMind's Formal Conjectures repository. Terence Tao worked through it (blog, 21 July); arithmetic **independently confirmed 21 July**. **n >= 3 refuted; n = 2 remains open; not yet journal peer-reviewed.** |

### G2 · Formal Conjectures — the transferable pattern

github.com/google-deepmind/formal-conjectures. Paper: arXiv:2605.13171. **2,615 formalised statements: 1,029 open
research conjectures plus 836 solved.** Apache 2.0, tracks monthly mathlib releases, tags immutable.

**The workflow is the lesson.** Formalising the *statement* once makes checking any candidate answer cheap:
(1) confirm the Lean statement uses trusted concepts, (2) confirm it says what is claimed, (3) run `lake build`.
**Expensive discovery, cheap verification.**

> This is the same argument as leCore's regression traps — and during these sessions it fired four times against
> the author: the flat dimension sweep, the 12x cache pessimisation, the `catmull_clark` API assumptions, and the
> hypervector-retrieval idea losing to Jaccard.

### G3 · The counter-arguments (deliberately included)

- **First Proof** (arXiv:2602.05192): 11 mathematicians posed 10 unpublished research lemmas with encrypted
  answers and a one-week window to defeat contamination. **Result mixed-to-negative** — GPT-5.1 Pro and Gemini 3
  Pro produced confident solutions but **only two were correct**; OpenAI's initial 6-of-10 claim dropped to five
  when mathematicians found a hole, and "the vast bulk of the submissions appear to be a lot of very convincing
  nonsense".
- **Selection bias.** Counterexamples are low-hanging fruit that admit concise machine-checkable refutations. An
  Imperial faculty member argued the Grothendieck counterexample's ease "just indicated that humans had not
  spent enough time thinking about the problem".
- **Human-in-the-loop confound.** Scaffolding is often unreported.
- **Contamination.** Epoch AI's May 2026 review flagged fatal errors in roughly a third of FrontierMath Tiers 1-4
  problems.
- **Buzzard's meta-point.** He trusts Lean-verified claims, *not* AI-generated informal prose — "I was not
  reading AI-generated informal mathematics".

---

## 9 · Genuine gaps — where leCore could be first

1. **NTT (or Walsh-Hadamard) binding for HRR/VSA.** Apparently unpublished as an actual VSA implementation. The
   novel framing is **exactness and determinism**, not merely speed.
2. **Formal SPARC capacity theory / AMP state-evolution transplanted onto VSA bundling capacity.** The
   literatures cite each other but nobody maps the SPARC-section = VSA-codebook isomorphism into a capacity
   theorem. leCore could produce the first "bundling capacity as a joint function of D, sparsity, code structure,
   and bits per dimension".
3. **A rate-distortion theory whose distortion metric is cleanup-decision preservation** (similarity geometry),
   not reconstruction MSE. Empirically probed by qFHRR; no propagation theory through bind -> bundle -> cleanup.
4. **A successor to Clarkson/Ubaru/Yang** giving capacity jointly in dimension *and* bits-per-dimension under
   quantization.
5. **Conway-Sloane lattice atoms as a VSA cleanup codebook** with exact sublinear recall. Mature in
   communications and quantization, unused in VSA.
6. **Integer or quantized resonator networks with exact, reorder-independent convergence guarantees.** Decoding
   is float and iterative everywhere in the literature.
7. **Any published claim that quantized or integer VSA yields a bit-exact, hardware-independent *cleanup
   decision*.** The specific claim that motivated this whole line does not exist — leCore would have to establish
   it.
8. **Cleanup-operator-as-PnP/RED-prior** — unifying resonator networks, AMP and plug-and-play into one
   training-free VSA solver. Conceptually available, unformalised.
9. **Gaussian-splat bundle and holographic memory in one engine.** Gaussian Wave Splatting proves the transform
   mathematically; nobody has run a splat scene *and* a VSA memory on the same holographic algebra — precisely
   leCore's architecture.
10. **Neural-free by-example texture synthesis** is absent from the SIGGRAPH 2026 program.

---

## 10 · Ranked shortlist — tied to measured bottlenecks

| # | Transplant | Target | Expected gain | Difficulty | Determinism risk | Novelty |
|---|---|---|---|---|---|---|
| **1** | **SPARCs + AMP / compressed-sensing unbundling** (B1/B2) | Bottleneck 2 | Adjacent work already shows about **2x info rate** (0.60 -> 1.26 bits/dim) from decoders alone; plausibly lifts the 20-32 instruction ceiling toward the Donoho-Tanner bound | Medium (AMP is about 30 lines; the theory is the work) | **Low** — fixed-iteration deterministic matmul | **High** |
| **2** | **Hadamard / Reed-Muller structured codebook** (C1/C2) | Bottleneck 3 | O(KD) -> **O(D log D)**; replaces 5,682 us at K=16384 with one transform plus argmax; gain grows with K | Medium (needs an opt-in codebook type for backward compatibility) | **Low** — integer +-1, exact | Med-High |
| **3** | **Integer NTT or +-1 WHT binding** (A1/A2) | Bottleneck 1 **plus determinism** | Eliminates FFT cross-microarchitecture risk entirely; **raw speed UNKNOWN and must be measured** (NTT likely neutral-to-slower; WHT potentially faster) | Medium (vectorised butterflies; CRT for NTT) | **Removes** risk | **High** |
| **4** | **qFHRR integer-phase representation** (D1) | Storage plus determinism | 64-bit -> 3-4 bits/dim at high bind fidelity | Medium; the atan2/CORDIC projection needs fixed-iteration determinism | Medium (localised to the projection) | Medium |
| **5** | **ReproBLAS summation + exact tie-broken argmax** (E1/E2) | Global bit-reproducibility | Bitwise-identical across CPUs at about 4x cost **on the summation hot path only** | Low-Medium | This **is** the guarantee | Low as technique, **High** as VSA discipline |
| **6** | **Walk on Decomposed Subdomains** (F1) | WoS tier | "Orders of magnitude faster than pure Monte Carlo" | Medium (MC kernels plus sparse solve, pure NumPy) | Low — deterministic global solve | Med (peer-reviewed, unapplied here) |
| **7** | **Modern-Hopfield exp-separation cleanup** (B4) | Bottlenecks 2 and 3 | Exponential separation raises effective capacity; pairs with #2 | Low (softmax matmul) | Medium (`exp` reproducibility) | Medium |

### Immediate zero-cost experiment

**Wire the existing `cosamp` and `iht` modules into unbundling.** They already ship in leCore and are apparently
unused for this purpose. Measure whether the instruction ceiling moves. Baseline, variance and kept negative,
per discipline.

---

## 11 · Standing caveats

- **The NTT-versus-`numpy.fft` microsecond comparison is a claim gap.** No authoritative NumPy benchmark exists.
  Any speed claim must be generated on the live codebase. NTT may well be *slower* than pocketfft while still
  winning decisively on exactness — record that honestly rather than burying it.
- **The "integer VSA deletes tie-arbitration" hypothesis is NOT supported by any surveyed paper.** Exactness
  holds for binding and unbinding only; similarity, cleanup and argmax remain non-exact in both qFHRR (the
  bundling projection) and Residue HDC (the float resonator). The exact route is inferred from cryptography, not
  demonstrated for VSA.
- **Modern-Hopfield capacity figures are asymptotic** and assume well-separated patterns. Real bundle crosstalk
  will be worse. Upper bounds only.
- **qFHRR fidelity figures could not be re-verified** from the abstract in round 3; confirm against the paper's
  table before quoting as measured.
- **Peer-review status.** Peer-reviewed: Kleyko et al. 2023, Raviv 2024, Rush et al. 2017, Xu et al. VAMP,
  resonator networks, Kymn et al. RHC, ReproBLAS, Shewchuk, Conway-Sloane, Vardy-Be'ery, ANS, Reed-Muller
  decoding, and the three SIGGRAPH 2026 TOG papers. Preprint or early-publication: qFHRR (2604.25939), GHRR v2,
  the Frontiers cleanup study, the Jacobian counterexample, and the SPARC-to-VSA transplant itself.
- **CUDA and learned implementations do not transfer** — Gaussian Wave Splatting, Points-as-Tori's fitting
  network, ScaNN. Only their representational ideas do.
