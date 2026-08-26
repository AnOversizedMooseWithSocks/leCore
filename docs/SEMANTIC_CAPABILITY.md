# leCore Semantic Capability

*A small file in the repo, kept fresh by CI, that lets the engine route a plain-English request to the
right module by MEANING instead of shared keywords. This is the product. It is deliberately narrow.*

> This document describes what SHIPS and how CI keeps it current. The much larger
> `leCore_model_assimilation_backlog.md` is a different kind of document: the research log of the
> "can we defrag / decode an LLM's weights" investigation. That arc produced mostly **kept
> negatives** — six compression mechanisms measured dead on three architectures — and its value now
> is the archive of what NOT to retry. The semantic index below is the one shippable result that
> came out of it. Keep the two separate: the backlog is history; this is the running system.

---

## 1. What it is, in one paragraph

`find_capability` has always matched a request to a module by **shared content words** — deterministic,
model-free, and blind to meaning. "squish a big array down for storage" shares no word with
`holographic_coldstore`; "airspeed velocity of an unladen swallow" confidently matched a physics module
on the single word *velocity*. The semantic capability adds a second path that scores by **cosine in an
embedding space**, so meaning matches even when words do not. Measured on the 12-ask suite: token
overlap ~2/12 top-1 (median rank 13 of 503); the embedding router 7/12 top-1 (median rank 1).

## 2. What ships (all of it, with real sizes)

| file | size | role |
|---|--:|---|
| `lecore_data/routing/index_128d.npz` | **128 KB** | 503 module vectors, 128d q8, ABTT correction baked in. THE shipped artifact. |
| `holographic/semantic_router/holographic_router.py` | ~5 KB | `EmbeddingRouter`: loads the index, routes a query vector by cosine. |
| `mind.route_semantic(...)` | — | the wired faculty; additive, the token `find_capability` path is unchanged. |

**No model ships.** The 96 KB index is pure data — dequantize, apply a fixed correction, take a dot
product. That is the whole runtime cost, and it is why this fits the footprint rule (core stays
NumPy/stdlib, nothing over a megabyte enters it).

## 3. How CI keeps it fresh (the loop you asked for)

`.github/workflows/semantic-coverage.yml`, on a push that touches `holographic/**` or `tools/semantic/**`:

1. **Restore the embedding cache** (runner cache, else decompress the committed ~27 MB float16 seed —
   so CI starts warm and NEVER does the hours-long cold embed).
2. **Embed only what changed** — the cache is content-addressed (`sha256(wiring||text)`), so a push that
   touches 20 modules embeds 20 docstrings, in about a minute.
3. **THE EXAM** — the 12-ask routing suite runs with `--exam --require-top5 8 --require-median 2`. A
   regression (a module renamed, a docstring rewritten into implementer jargon) **fails the build**.
   Coverage is a test, not a report.
4. **Rebuild the 96 KB index** from the refreshed cache (`tools/semantic/export_index.py`) and commit
   it — the same drift-check discipline the repo already uses for `CAPABILITIES.md`.

The heavy cache (the full 18k-entry embed of every docstring/NOTES window) is a **build intermediate**:
gitignored, lives on your disk and in the CI cache, never committed. Only the 96 KB index and the
~27 MB warm-start seed are in the repo.

## 4. How to use it

```python
import lecore
m = lecore.UnifiedMind(dim=256, seed=0)

# route by meaning -- returns [(module, cosine)] best-first
m.route_semantic("make my picture less grainy", query_vec=my_nomic64_vector)

# or route a phrase whose vector was embedded at build time (the exam's asks, an app's fixed vocabulary)
m.route_semantic("make my picture less grainy")     # uses the build-time cache if present
```

**The honest boundary — read this before relying on it.** The router needs a query **vector**. It has
one when you supply it, or when the phrase was embedded at build time. For a brand-new free-text query
with **no model present**, `route_semantic` returns `None` — and the caller falls back to the token
`find_capability`. It NEVER fabricates an embedding: an honest `None` beats a confident wrong route.
The loader also degrades — if the index or numpy is missing, it returns `None` and the token router
keeps working, so a minimal install never crashes.

## 5. The narrow roadmap (only items that improve THIS)

Ordered by leverage. Nothing here is about decoding LLM weights; that investigation is closed.

1. **N31 — offline free-text routing.** Today a new phrase needs the encoder to get a vector. A single
   ridge-regression matrix (`encoder ≈ W · token-pool`, closed form, ~2 MB) would let the engine embed
   an unseen query from the token table alone — no runtime model. This is the one piece that turns
   `route_semantic` from "cached phrases only" into "any request." `tools/semantic/distill_map.py`
   already measures whether W is good enough.
2. **Findability lint in CI.** For every capability, embed 5 stranger-phrasings; if it doesn't rank
   top-5 for its own descriptions, the docstring/aliases are the bug. Makes the docs fail like tests —
   the router becomes the test harness for its own corpus.
3. **Alias mining.** Log failed-then-rephrased queries as alias candidates; vocabulary grows from use,
   not from a bigger model.
4. **Wire `route_semantic` into the agent.** The agent still routes on token overlap. Point it at the
   cosine path (with the abstention `None` as its "I don't know") and the "airspeed velocity" class of
   confident-wrong routes goes away.

## 6. What this is NOT

- Not a language model, and not a dependency on one. The encoder is a **build tool** run to make the
  index; it is deleted from the runtime path.
- Not a describer of leCore's internals. It routes requests to modules; it does not explain the system.
- Not the assimilation program. That measured what an LLM's weights contain (answer: nothing
  compressible) and is archived. This ships one 96 KB file and keeps it honest.

---

*Determinism holds throughout: `PYTHONHASHSEED=0`, `hashlib` for cache keys (never `hash()`), q8 +
fixed ABTT transform, argsort with a name tie-break. No RNG, no model, same result every run.*
