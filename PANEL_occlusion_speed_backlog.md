# holostuff Panel Review — the occlusion-recall speed bottleneck

*The team was convened on a specific, measured limitation: occlusion recall (RT-V) breaks the bundle capacity cliff —
recovering ~D/4 items from one bundle where the linear readout washes out at ~√D — but it costs **~170× the linear
readout** (a sequential matching pursuit, O(M·N·D) vs O(N·D)). The brief: is this an already-solved problem
elsewhere in the tech stack, and what do we change? As always, every proposal is attributed to a SEAT and that
field's REAL, published method (cited), and the load-bearing fix was prototyped and MEASURED on the real substrate
before being written down — the result is in §2.1. The user's two hunches — that our RAM/cache work isn't wired to
this, and that the 3DGS work gave us gradients-on-the-fly — both turned out correct; see §3.*

---

## 1. The problem, stated precisely

Occlusion recall IS plain **matching pursuit**: each step scores every codebook atom against the residual
(`scores = cb @ residual`, O(N·D)), picks the argmax, subtracts its contribution, and repeats M times — O(M·N·D).
The linear readout scores once — O(N·D). So the whole cost is the **M sequential rescans of the dictionary**. Two
independent levers: the **per-step rescan** (the N·D), and the **number of steps** (the M).

The panel's unanimous first observation (Milanfar's seat leads): **this is the speed problem of greedy sparse
recovery, and it has been solved three different ways in the compressed-sensing literature.** Nothing here is new;
the work is choosing the right one and wiring it to what we already have.

---

## 2. The three established fixes — each a seat, each a real method

### 2.1 Cache the Gram matrix — Batch-OMP *(Duda + Stoudenmire; the RAM hunch, MEASURED)*
**Real basis.** Rubinstein, Zibulevsky & Elad (2008), *Efficient Implementation of the K-SVD Algorithm using Batch
Orthogonal Matching Pursuit* — the basis for scikit-learn's OMP. Precompute the **Gram matrix** `G = DᵀD` once;
then the per-step correlation update never rescans the dictionary. The recent batched-OMP literature confirms the
shape: batching all signals together is the dominant optimization, with Gram precomputation contributing on top of it, and the canonical note that precomputing G makes Batch-OMP specifically a method for sparse-coding large sets of signals — i.e. it pays when the **codebook is reused across queries**, which is exactly our case (one fixed vocabulary, many recalls).

**Why it fits us even more cleanly than Batch-OMP.** Occlusion is *plain* MP (subtract the projection), not
orthogonal MP, so we don't even need the Cholesky update Batch-OMP carries for the orthogonal projection. We only
maintain the correlation vector `α = cb @ r` and update it on each subtraction:
```
α ← α − share · G[:, j]          # one Gram COLUMN, O(N) — instead of recomputing cb @ r, O(N·D)
```
**Measured on the substrate (the load-bearing prototype, kept on record):** Gram-cached occlusion recovers the
**identical atoms** (it is exact, not approximate) and runs **12× faster at D=512 and 23× faster at D=1024** — and
the speedup grows with D, because the per-step cost drops from O(N·D) to O(N), removing the D factor entirely. That
alone takes the readout from ~170× slower than linear to **~7× slower**, exact. **Kept negative:** O(N²) memory for
G and an O(N²·D) one-time precompute — so it pays when the codebook is reused (it does), not for a one-shot recall
against a throwaway dictionary.

### 2.2 Approximate the atom search — Approximate MP / MIPS *(Pharr; the acceleration-structure route)*
**Real basis.** The Approximate Matching Pursuit line (e.g. arXiv:1807.03694) replaces the per-step inner-product scan over a large dictionary with a nearest-neighbor search, accepting any sufficiently near atom instead of the exact best one. And since our atoms are unit-norm, the argmax of `cb @ r` IS a **Maximum Inner Product Search**, which reduces to ordinary nearest-neighbor search exactly when the vectors share a norm. We already own the sublinear structure for this — **`HoloForest`** — and it already reports cross-tree agreement, Pharr's "did the traversal actually find something" abstention.

**The transfer.** Run each MP atom-selection through `HoloForest` instead of the exhaustive scan: per-step drops from
O(N·D) to O(log N · D). This is the **N-factor** lever (complementary to §2.1's D-factor lever — they can combine:
Gram for exact cheap correlations, the forest for sublinear selection). **Kept negative:** it is *approximate* — the
forest may miss the true best atom, so recovery F1 can drop; this one must be **measured** against the exact readout
before it ships (Cranmer's bar), and it only earns its place at large N where the scan dominates.

### 2.3 Recover many atoms per iteration — CoSaMP / Subspace Pursuit / IHT *(Olshausen + Macklin; the M-factor route)*
**Real basis.** The batch-selection family attacks the *number of iterations* directly. CoSaMP (Needell & Tropp) improves on OMP by selecting multiple atoms per iteration, and Subspace Pursuit (Dai & Milenkovic) refines the support set iteratively; both select several atoms at once and allow previously-chosen wrong atoms to be discarded. Crucially for our second hunch, Iterative Hard Thresholding (Blumensath & Davies) uses gradient-based updates followed by hard thresholding to enforce sparsity.

**The transfer.** Replace one-atom-at-a-time MP with a batch-selection recovery: recover the strongest K candidates
per pass and refine, turning ~M passes into ~log passes. **Kept negative:** the batch-selection guarantees are
stricter (the CoSaMP/IHT recovery bounds need a small restricted-isometry constant), so at the very edge of the
capacity curve (M ≈ D/4) they may recover slightly worse than careful one-at-a-time MP — measure F1 at the edge,
keep the trade explicit.

---

## 3. The two hunches — both correct

### 3.1 "Our RAM work isn't wired up to this." — **Correct, and it's the highest-value fix.**
The Gram matrix (§2.1) and the HoloForest index (§2.2) are both **cached, reused structures** — precompute once,
amortize over every recall. That is precisely what the cache-layer backlog (`holostuff_cache_layer_backlog.md`,
promoting `ReflexCache` to a general working-set faculty) is *for*, and it has **not** been connected to the recovery
path. The readout recomputes inner products it already has the structure to avoid. Wiring the cache so the Gram /
index live as cached working-sets keyed by codebook identity is the RAM hunch made concrete — and §2.1 already
measured the payoff at 23×.

### 3.2 "The 3DGS work gave us gradients on the fly." — **Correct, and partly already in the box.**
The splat fitting (`aniso_fit` / `densify_fit`) carries a hand-derived-gradient **Adam optimizer** (no autodiff —
within the constraint), and the cache/field/vision modules already carry **finite-difference gradient** machinery.
So the engine has both halves of a general gradient-descent capability (a gradient source + an optimizer), but they
are **siloed in the splat path**. This matters here for a specific reason: IHT (§2.3) is *literally* a gradient step
plus a threshold — so the gradient machinery the 3DGS work brought in is the natural engine for the M-factor fix.
More broadly, the user's instinct is right that this is architecturally larger than splats: a general
`optimize`/`gradient_descent` faculty would serve IHT recovery, gradient-based resonator refinement, and alignment —
anywhere a problem "needs gradients." That is the real unlock the 3DGS work quietly delivered.

---

## 4. The backlog — what to change (prioritized by value over effort)

**SPEED-1 — Gram-cached occlusion recall (lead; MEASURED 23×, exact).** *(Duda, Stoudenmire)*
Add a cached `G = cb @ cbᵀ` and an `α`-update recovery path to `holographic_occlusion`; the existing per-step rescan
becomes a Gram-column update. Exact (identical recovery, asserted). New `occlusion_recall(..., gram=G)` arg, with G
optionally supplied/cached. Bar: bit-identical recovery to the current readout, ≥10× faster at D≥512. Negative kept:
O(N²) memory; pays on reuse.

**RAM-1 — wire the cache layer to the recovery path (the RAM hunch, infrastructure).** *(cache thread)*
Promote the cache-layer working-set faculty far enough to hold the SPEED-1 Gram (and the SPEED-2 index), keyed by
codebook identity, so the precompute persists across recalls instead of being rebuilt. This is the "RAM not wired up"
gap; SPEED-1 supplies the thing to cache, RAM-1 makes it durable. Bar: a second recall against the same codebook pays
zero precompute.

**GRAD-2 — promote the splat optimizer to a general `optimize` faculty (the architectural unlock).** *(Macklin + general)*
Extract the Adam loop from the splat path and pair it with the finite-difference gradient helper into one reusable
faculty: `optimize(loss_or_grad, x0, steps, ...)`. Default to analytic gradients where supplied (cheap), finite
differences where not (honest cost note: O(D) evals/gradient — fine for small D, expensive for large). Bar: reproduce
the splat fit's result through the general faculty; then it is available to everything below. Negative: no autodiff —
FD is the general fallback and is not free.

**SPEED-3 — batch-selection recovery (CoSaMP / Subspace Pursuit).** *(Olshausen, Milanfar)*
Add a multiple-atoms-per-iteration recovery beside the one-at-a-time MP, turning ~M passes into ~log. Combine with
SPEED-1's Gram for the correlations. Bar: match the one-at-a-time F1 across the capacity curve while cutting
iterations; keep the edge-of-capacity F1 trade on record. Negative: stricter recovery conditions.

**GRAD-1 — IHT recovery on the general optimizer.** *(Macklin)*
Once GRAD-2 lands, implement Iterative Hard Thresholding (gradient step + keep-K-largest) as a recovery path — the
gradient-native member of the batch-selection family, and the cleanest demonstration that the 3DGS gradient machinery
serves recovery. Bar: match MP F1 at lower wall-time on a loaded bundle. Negative: step-size tuning.

**SPEED-2 — HoloForest-accelerated atom selection (Approximate MP / MIPS).** *(Pharr)*
Route MP's atom selection through `HoloForest` (MIPS = NNS for unit-norm atoms) for sublinear selection at large N.
Bar: beat the exact readout's wall-time at large N while holding F1 within a stated tolerance — and report whether the
forest's cross-tree agreement and the recovery agree (the two abstention signals on the same selection). Negative:
approximate — measure the F1 cost before shipping; only earns its place at large N.

**Suggested sequence.** SPEED-1 first (measured, exact, biggest single win) → RAM-1 (make its Gram durable) →
GRAD-2 (the reusable optimizer the whole project wants) → SPEED-3 and GRAD-1 (the M-factor, now gradient-powered) →
SPEED-2 last (approximate, large-N only, needs the accuracy measurement). SPEED-1 + SPEED-3 together (Gram-cached
batch selection) is the asymptotic endpoint: exact cheap correlations × log iterations.

---

## 5. The honest bottom line

The bottleneck is not a wall — it is **recompute-what-you-already-know** (Eno's reframe): the readout was rescanning a
fixed dictionary M times and discarding the correlations between steps. The compressed-sensing field solved this three
ways — cache the Gram (the D factor), approximate the search (the N factor), batch the selection (the M factor) — and
**the cache route is already measured at 23× and exact**, taking occlusion from ~170× slower than linear to ~7×, with
the rest of the gap reachable by the other two. The user's two hunches were both load-bearing: the RAM/cache work is
the highest-value fix and simply wasn't wired in, and the 3DGS gradient machinery is the natural engine for the
M-factor fix and deserves promotion to a general faculty in its own right. As ever: the constraint pointed straight at
a solved problem one rung over in the stack, and the win is wiring what we already have to where it pays — measured,
with the approximate routes' accuracy costs kept loud until they clear their bar.

---

### References this review leaned on
- Rubinstein, Zibulevsky & Elad (2008), *Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
  Matching Pursuit* (Gram-precompute / Batch-OMP); Mallat & Zhang (1993), *Matching Pursuits with Time-Frequency
  Dictionaries* (the original MP).
- Needell & Tropp (2008/2010), *CoSaMP: Iterative Signal Recovery from Incomplete and Inaccurate Samples*; Dai &
  Milenkovic, *Subspace Pursuit*; Blumensath & Davies, *Iterative Hard Thresholding for Compressed Sensing* (the
  gradient-step-plus-threshold member).
- The Approximate Matching Pursuit / nearest-neighbor-atom-selection line (arXiv:1807.03694) and the MIPS↔NNS
  equivalence for equal-norm vectors (clustering-based MIPS literature).
- Recent batched-OMP implementations confirming batching as the dominant speedup with Gram precompute on top
  (arXiv:2407.06434 and the batched-OMP GPU implementation).
