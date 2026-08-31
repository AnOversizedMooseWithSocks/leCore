# Galvatron runbook — what to run, what the numbers should say

Your model is the first genuinely trained artifact in this arc. Every experiment that
cp106 had to mark BLOCKED is unblocked on it, because the gate that blocked them measures
exactly the thing your model has and the sandbox fixtures don't:

    sandbox fixtures   ppl 2272-2304 vs vocab 2,048        1.11x chance   AT-CHANCE
    galvatron          ppl 16.3      vs vocab 248,320      0.00007x       TRAINED

Everything below is ordered so each step's output tells you whether the next one is worth
running.

---

## 0. Re-run the install (cp108 fix)

```
install.bat
```

**What changed:** the `self_write` step failed at r=0.603 / 36% top decile against a 40%
bar. Root cause was an underdetermined fit, not a weak model — the readout fits one ridge
coefficient per hidden dimension and trained on half of 1400 tokens, i.e. 699 rows for
1024 features (0.68 rows/feature). A matched synthetic with a *planted* r=0.55 signal and
the identical estimator recovers only r=0.166 at that ratio, and 0.430 at 4 rows/feature.
The signal never changed; the rows did.

**Expect on the self_write line:** `~8194 tokens, 4.00 rows/feature`, with r and top-decile
materially above 0.603/36%.

- **Passes (>40%)** → the readout was always fine; proceed.
- **Still fails at 4.00 rows/feature** → now it's a real signal-strength result. Send me the
  line; the next move is the `surprise` mode (state at t+1, reference r=0.605) rather than
  `entropy`, which is a different and better-posed question for this model.
- **MemoryError** → tell me the number; the chunked fitter holds peak at one 1400-token
  forward (~1.4 GB of logits at your vocab), so an OOM would mean something else is resident.

Also confirm the two lines that already look right: `hrnn_channel ok ... 0.0e+00 (relative
0.0e+00) -- bit-identical` and `prepend ok ... drift 6.128e-14`.

## 1. Gate the artifact (1 second)

```
python tools\fixture_gate.py C:\projects\leCore\assimilation\work\assimilated
```

**Expect:** `VERDICT : TRAINED`, roughly `0.0001x chance`. This is the cheap check that
would have saved three of cp106's four builds. Exit code 0 means every learned-structure
experiment below is meaningful.

## 2. Run the doctor (the whole pipeline, one command)

```
python tools\model_doctor.py C:\projects\leCore\assimilation\work\assimilated --seqs 30 --json doctor.json
```

Six stages: gate → cosine profile → causal bypass probe → prune plan with parameter budget
→ heal attempt → verdict. On the sandbox fixture it takes 20 s; on 24 layers at hidden 1024
budget a few minutes, dominated by the bypass copies.

**The number to watch is stage 3's `screen vs truth: r=...`.** On the fixture, cosine
explained only **34%** of the variance in what actually happens when a layer is removed —
19 layers cleared the cosine screen and **zero** survived the bypass probe. If your trained
model shows the same pattern, that is a genuine result about the published angular-distance
criterion, measured on a real model rather than a toy.

Branches:

- **Some layers clear both stages** → you have a real prune candidate list with a parameter
  budget attached. That is the first honest compression result of the arc.
- **None clear** → the same verdict the fixture gave, now on a trained model, and the
  thesis's "waste" pillar needs restating rather than defending.
- **Stage 5 healing recovers held-out agreement** → this would *contradict* the sandbox
  result (cp109: no rank repaired it, held-out fell 0.400 → 0.200) and would be the most
  interesting outcome available. Send the whole `doctor.json`.

## 3. The experiments cp106 could not run

Once stage 1 says TRAINED, these become meaningful for the first time:

- **SAE → cleanup codebook.** On the fixture the learned dictionary beat a random-init null
  by +0.0016 — nothing, correctly, because there was nothing to find. On a model at ppl 16.3
  the same fit is a real test of the superposition pillar.
- **Compute-to-lookup.** It failed on the fixture with a clean mechanism: the layer's inputs
  were effective rank 124/128, essentially isotropic, so no table could cover them, and 8-NN
  tied a constant predictor. A trained model's hidden states should be far from isotropic;
  measuring that effective rank is the one number that decides whether the memory-layers
  conversion is even possible post-hoc.
- **Prune-and-heal with perplexity.** Blocked on the fixture because ppl was already at the
  ceiling and pruning "improved" it. On your model, ppl 16.3 gives healing a real gap to
  close.

## 4. What to send back

The `self_write` line from step 0, the `fixture_gate` verdict, and `doctor.json`. That's
enough to turn every remaining BLOCKED result in `docs/UNICRON_THESIS.md` into a measured
one.

---

### Standing caveats, so no number gets over-read

- Top-1 agreement measures **output change**, not quality. It is valid on any artifact
  because it is self-referential, but on a trained model prefer the perplexity deltas.
- Capacity and noise are **separate budgets** (cp104/cp105). The recall ladder extends how
  many items can be superposed; it does nothing for a degraded cue.
- Brute-force recall beats the tree at every N tested here, which is the expected result in
  high dimensions, not a leCore defect (FAISS arXiv:2401.08281 §3.1).

---

## 5. The form prior on your model's own embedding table (cp113)

The etymology work so far used distributional vectors built from text. Your model carries a
better ground truth: **248,320 learned embeddings**, a great many of them rare tokens whose
vectors were estimated from few occurrences. That is exactly the population the form prior
is for.

```
python tools\form_prior.py --selfcheck
```

Expect `rank@1 1.0`, `variance_explained 0.891` — the synthetic check where morphology
really is the generator. That number matters as a contrast: on real distributional data the
same tool reports variance explained **-0.110**, and the gap between those two is the tool
telling you whether form *generates* meaning or merely *correlates* with it.

Then, against the real table:

```python
from tools.form_prior import FormPrior
import numpy as np, json
import holographic.io_and_interop.holographic_gdnruntime as g

w = g.load_weights_dir(r"C:\projects\leCore\assimilation\work\assimilated")
if isinstance(w, tuple): w = w[0]
E = np.asarray(w[[k for k in w if k.endswith("embed_tokens.weight")][0]], np.float32)
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

tok = json.load(open(r"...\tokenizer.json", encoding="utf-8"))
vocab_map = tok["model"]["vocab"]                    # token string -> id
items = [(s.lstrip("Ġ▁").lower(), i) for s, i in vocab_map.items()]
items = [(s, i) for s, i in items if s.isalpha() and len(s) >= 4]

words = [s for s, _ in items]; rows = [i for _, i in items]
cut = int(0.85 * len(words))
fp = FormPrior().fit(words[:cut], E[rows[:cut]])
print(fp.evaluate(words[cut:], E[rows[cut:]], train_mean=E[rows[:cut]].mean(0)))
```

**What each outcome means:**

- **rank@1 far above chance** (chance is 1/n for the held-out set) → the model's embedding
  table contains recoverable form→meaning structure, and rare-token embeddings can be
  initialised from their spelling. This is the first test of the thesis on learned
  embeddings rather than corpus statistics.
- **rank@1 near chance** → the model's tokenizer has scattered morphologically related
  strings across unrelated vectors, which is the "curse of tokenization" showing up
  directly in your weights, and is itself a publishable-shaped negative.
- **variance_explained still negative** → expected, and not a failure. It is the measured
  WHICH-not-WHERE dissociation: the prior identifies the right neighbourhood without
  reproducing the vector. Use it to initialise, never to replace.

**The one that would change the plan:** if `variance_explained` comes out clearly positive
on your table, form *generates* those embeddings rather than merely correlating with them,
and a generator could stand in for part of a 248,320-row table. That is the compression
result five separate experiments have failed to find, so it would need re-running before
anyone believes it.


---

## Pre-flight audit for the next live run (cp130)

Audited before you re-run `assimilate` + `install`. Verdict: **READY**.

### What changed since your last run, and the risk of each

| change | checkpoint | touches install path? | risk |
|---|---|---|---|
| `fit_novelty_chunked` + adaptive novelty budget | cp108 | **YES** | the one real change; verified live below |
| `nov_ids` kwarg on `install()` | cp108 | **YES** | additive kwarg, default None -> old behaviour |
| `unicron_embed_repair`, `embed_repair_candidates` | cp119 | no | standalone, zero risk |
| `unicron_interstitial` + module split | cp129 | no | standalone, zero risk |
| `Lexicon.bootstrap_by_form` | cp118 | no | additive method |
| `fixture_gate`, `prune_probe`, `recall_guard`, `model_doctor`, `form_prior`, `embedding_repair` | cp107-116 | no | tools only |

Confirmed by grep: none of the new engine methods appear anywhere in
`holographic_install_lecore.py`.

### Live install, run end to end on the qwen3.5-shaped fixture

```
prepend           ok    1 layer added, drift 0.000e+00
registers         ok    16 slots, regenerable from seed
hrnn_channel      ok    4 rungs, 4096 tokens
nullspace_guard   ok
state_track       ok    4 of 16 registers
exit_calibration  ok    safe_depth 8 of 9
self_write        --    1400 tokens, 5.46 rows/feature   (at-chance fixture; cannot pass here)
final             BETTER, delta -0.004%
```

The install completes and does not degrade the model. `self_write` now reports
**`fit_tokens` and `rows_per_feature`**, which only the cp108 code path emits -- that is the
proof the fix is live.

### The numbers that will differ on your box

| hidden | corpus tokens available | fit_tokens | rows/feature | |
|---|---|---|---|---|
| 128 (fixture) | 9,000 | 1,400 | 5.46 | floor -- budget never engages |
| **1024 (yours)** | **40,000** | **8,194** | **4.00** | **budget engages** |
| 1024 | 3,000 (short corpus) | 3,000 | 1.46 | prints UNDERDETERMINED |

**Memory, at your vocab of 248,320:** the chunked fitter allocates **1.39 GB** per 1,400-token
chunk. The single-shot version we avoided would have wanted **8.14 GB**. If you see an OOM on
this step, it is not this code.

### What to expect on the self_write line

`novelty readout r=..., finds ...% of the top decile (8194 tokens, 4.00 rows/feature)`

- **passes (>40%)** -> the cp108 diagnosis was right; the readout was only starved
- **still fails at 4.00 rows/feature** -> now a real signal-strength result, not an artifact.
  Next move is the `surprise` mode rather than `entropy`
- **prints UNDERDETERMINED** -> your corpus is short; pass a longer `--doc`

### Gates, all green at time of audit

`audit_imports`, `catalog_gaps`, `skill_lint`, `tag_lint`, `structure_audit`, `usage_audit`,
`shard_tests --selfcheck` (674 files -> 4 shards), `regen_docs --check`,
`wiring_report --check`, `form_prior --selfcheck`, `recall_guard --selfcheck`,
`bench_ladder` 0.75/1.00/0.25, `bench_longmem` 1.000 across all six.

Launcher check: `install.bat` still carries the cp97 escaped-redirect fix.

### Order to run

1. `install.bat` -- watch `hrnn_channel` (expect `0.0e+00 bit-identical`) and `self_write`
2. `audit.bat` -- expect the ladder R^2 0.99858 line and a clean TOTAL
3. `python tools\fixture_gate.py <assimilated>` -- expect **TRAINED**
4. `python tools\model_doctor.py <assimilated> --json doctor.json`
5. Then the pre-registered constant-folding experiment from run twenty-three
