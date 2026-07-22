# SEMANTIC BACKLOG — one goal, gated steps

*Written after the step-back audit (2026-07-17). The product is one sentence: **a plain-English request finds the
right capability.** Every item below either serves that directly or removes a hazard in its way. Items marked
GATED do not proceed until the measurement before them says so — the hierarchical-routing arc died from wiring
before measuring; we do not repeat it.*

**The map the audit produced (state before this backlog):**

| piece | state |
|---|---|
| `find_capability` (token, 2,099 entries) | THE front door; used by `suggest`/agents/Rule-0. Consults nothing semantic. |
| `route_semantic` (N28, embedding, 7/12 vs token 1/12) | Wired; works ONLY for cached phrases or a supplied vector; honest `None` otherwise. |
| `queryembed` (N31, free-text→vector, no model) | **Built and orphaned**: `route_semantic` never calls it, and its `W` artifact was never fitted. |
| `holoroute` (structured role/filler routing) | Built, synthetic selftest only, connected to nothing. |
| `bm25_rank`/`fuse_rankings` | Generic faculties; the measured dense-dominant hybrid never applied to the front door. |
| semantic tags (S4.2) | Feed the browse menu only (648 leaves, was 108). Unmeasured as a retrieval signal. |
| `scripts/` vs `tools/semantic/` | Duplicated tooling in DIFFERENT versions + a stale 64d index. Hazard: running the old copy. |

---

## S0 — Finish the tag close-out ✅ DONE
Coverage 108/2095 (5.2%) → 649/2099 (30.9%); menu reads the mind's own catalog (was silently reading the
~400-entry default); `coverage` name-collision adjudicated (3 domains, benign homonym); 18 tests; audits 0/0/0.
The two duplication-audit failures were ordering (collision entry landed before `regen_docs` ran) — both pass.

## S1 — Kill the tooling duplication ✅ DONE (reversible, executed this session)
`scripts/README.md` says the folder was DESIGNED to sit OUTSIDE the repo (sibling layout, holds model dirs);
it was committed in. The copies inside are OLDER (knowledge_index 476 vs 965 lines; export_index 68 vs 114;
distill_map 182 vs 331) — running one would cold-embed with stale wiring and pollute the cache.
Action taken: the six files duplicated by `tools/semantic/` are replaced by a README pointer; the
research-arc-only files (census/probe/run_all/distill_router) stay untouched as history; the stale
`routing_index_64d.npz` is removed (the shipped index is `lecore_data/routing/index_128d.npz`).
Reversible: `git checkout` restores anything; nothing canonical was touched.

## S2 — N31 free-text routing gate ✅ RUN. **FAILED. KEPT NEGATIVE — DOES NOT SHIP.**
Moose ran the fit at 128d (warm cache + nomic weights — the config that would actually ship):

| router | top-1 | top-5 | median |
|---|---|---|---|
| `[ceiling]` full encoder (137 MB) | 5/12 | 8/12 | 2 |
| `[floor]` SIF token-pool (23 MB q8) | 3/12 | 5/12 | 13 |
| `[m2v]` whitened table (no W) | 3/12 | 5/12 | 17 |
| `[m2v+zipf]` whitened + rank-SIF | 1/12 | 2/12 | 18 |
| `[ours]` SIF @ W | **1/12** | 2/12 | **19** |

**The learned map is WORSE than not having one** (1/12 vs floor's 3/12; median 19 vs 13). Ridge held-out R² was
+0.06 — it cost 2 top-1 and 6 median rank to apply. Nothing clears. Nothing exports. Free-text routing via a
distilled map is **dead at 128d**, and S4's embedding arm dies with it (no encoder worth shipping).

**Two method failures of mine, worth more than the numbers.** (1) **The bar was above the ceiling.** I registered
top-1 ≥6/12; the full 137 MB encoder scores 5/12. Unfalsifiable — no map could clear a bar its own upper bound
can't reach. The right reference was never the ceiling but the **floor**: does the map beat doing nothing? That's
answerable, and it's no. (2) **The export wasn't gated, and only a crash prevented a bad ship.** `--export` called
`export_query_embed` unconditionally regardless of the scores printed one line above; had it worked, a failing run
would have written an 8 MB routing-degrading artifact into `lecore_data/`. It only failed because the function sat
BELOW the `__main__` guard → NameError. That's the half-fix twin of last session's bug (then: never *called*; now:
called and not yet *defined*) — neither caught because the fit needs encoder weights the container lacks, so the
chain was never run end-to-end. **A fix you cannot execute is a hypothesis.** Bar now enforced in code; a failed
gate writes nothing.

**Honest silver lining:** the 137 MB ceiling itself only reaches 5/12 — distillation was never the weak link. The
task is hard, and a 12-ask exam can't separate these arms anyway (~0.6 SE per ask). Any reopening needs the
35-ask `catalog_exam.py` first, not a cleverer map.

## S3 — GATED on S2's numbers: ship or refuse the artifact
The plumbing is DONE and additive (absent artifact = today's byte-identical behavior). S3 is now only a decision:
if the S2 run clears the bar, commit `lecore_data/routing/query_embed_128d.npz` (~8 MB at 128d) and free-text
`route_semantic` turns on for every query, no model shipped. If not, kept negative, nothing lands.

## S4 — front-door hybrid: ASK SET BUILT, non-embedding arms MEASURED = KEPT NEGATIVE
`tools/semantic/catalog_exam.py`: **35 asks over the 2,111-entry CATALOG corpus** (distinct from the 12-ask MODULE
exam — different corpus, different baseline; conflating them is the RS-1b mistake). Asks are phrased as a stranger
types; several lifted verbatim from the client's audit, the best source of un-coached phrasing. Every gold is
VERIFIED against the live catalog before a score prints — **8 of my first-draft golds did not exist** (I wrote the
names the engine *ought* to have: `mesh_boolean`, `cleanup`, `bind`… the engine has `brep_boolean`,
`learn_cleanup`, `map_bind`). That is itself the finding: if the names *I* reach for from memory aren't the
catalog's, a stranger's odds are worse — which IS the retrieval problem.

**FIRST MEASUREMENT** — token top1 14/35, top5 21/35, median 2.0 · bm25 15/35, 22/35, 2.0 · rrf 15/35, 22/35, 2.0.
**KEPT NEGATIVE:** BM25 is **+1 ask**; RRF adds **nothing** on top. On 35 asks a 1-ask delta is ~0.5 SE — noise in
a win's clothes. **Neither ships.** BM25 does rescue one real lexical case (coldstore r46→r4), matching RS-1's
"rescues buried lexical cases" — worth remembering for a future tiebreak, worth nothing alone.

**14/35 asks are buried past top-5; 6 miss in EVERY arm** ("cut a hole in a mesh with another mesh"→brep_boolean,
"turn a point cloud into a surface"→points_to_mesh, "store a key value pair in a vector"→map_bind, "what can this
engine do"→find_capability). Zero lexical overlap — no lexical method can reach them. **DELIBERATELY NOT FIXED
with aliases:** every one would pass in a minute, and the suite would stop measuring retrieval and start measuring
whether someone had read it. The misses ARE the signal, and they are the honest argument for the embedding arm
(S2/N31) and for alias MINING from real failed-then-rephrased queries — aliases learned from USE, never from the
exam. Baseline + the negative pinned by `tests/test_catalog_exam.py` (slow lane); the BM25 test fails loudly if a
future change ever makes it genuinely dominant, which would be the signal to re-open this with real evidence.

**Remaining for S4:** the embedding arm, gated on S2. The instrument is built and waiting.

## S7 — io-KIND drive + tag lint ✅ LINT DONE, BATCHES 1-2 DONE (91→108 tagged, 104→123 edges)
**The correction that spawned this item:** the "87/2,095 (4%)" that started this arc was the client's io-kind
(`consumes`/`produces`) count — a DIFFERENT tag system from the semantic verbs raised to 649. io-kinds are the
pipeline edges nodegen turns into typed nodes downstream; live count **116/2,099 (5.5%)**, still open.
Order inside the item is load-bearing: **LINT FIRST** (their measured evidence: one wrong `field` tag routes
image→mesh through a tensor denoiser and an Aharonov-Bohm ring — a wrong tag poisons Auto Route, exactly the
"wrong branch looks done" failure the verb-tagger abstains to avoid), THEN the drive (infer where unambiguous,
abstain otherwise, hand-tag mesh_and_geometry/rendering/simulation/sampling first), carrying C7's `method=`
field on every new tag so prose titles stop needing example-regex recovery.

## S5 — Sub-branch drift gate ✅ DONE
S4.1 gated the ROOT only, so **102 capabilities under 15 branches SEMANTIC_TAXONOMY.md never listed** passed CI
for as long as they existed (`create/emit`×21, `simulate/step`×37, `analyze/measure`×15...). Doc-first, per the
doc's own rule (a branch with members has earned its place, so the DOC was what drifted): all 15 documented, with
`create/emit`'s near-duplication of `create/procedural` recorded as a kept observation rather than smuggled into
a tidy-up — re-filing 21 live tags is a change that should be MEASURED, not slipped into a docs pass.

The reverse direction found two REAL bugs: `convert/isosurface` and `simulate/pbd` were **documented but EMPTY**,
which the doc itself calls a bug. Not aspirational — their members were untagged: `occupancy_to_mesh` /
`mesh_from_sdf` / `points_to_mesh` are exactly "points↔mesh, sdf↔mesh", and `resolve_swept_collision` is a PBD
collision step the stem "resolve" had mis-filed under `analyze/pipeline`. Both populated via new
`_SEMANTIC_OVERRIDES` — the honest mechanism for names whose verb lies (a cleverer stem table would cost accuracy
on every OTHER name; the override list is pinned small by test).

Now **66 documented == 66 in use, zero drift in either direction**, gated by a test that compares the doc to the
LIVE engine rather than to a third hardcoded copy that could itself drift. Both rules proven to bite.
Also fixed one of MY OWN miss-tags, found by reading a branch's members instead of trusting its count:
"Run any faculty as a background job" was `simulate/step`; simulate/ means "evolve a physical field over time" and
a job evolves nothing.

## S6 — Commit guidance for Moose (standing)
Commit: semantictag module + catalog/unified wiring + tests + this backlog + the scripts/ reconciliation.
Nothing else pending. `index_128d.npz` untouched. Seed untouched.

---

**Explicitly NOT doing (kept negatives, do not re-propose without new evidence):**
- Fusing routers into `find_capability` by intuition (S4 exists precisely so this never happens).
- Multivec at 128d (refuted, RS-1). A wider shipped index (refuted, RS-1b). Hierarchical routing (refuted).
- Loosening the tag table to raise coverage — abstention is the contract; wrong branch > no branch is FALSE.
- `holoroute` wiring — synthetic-only evidence; it enters S4 as one more measured arm IF its encoding cost is
  paid by then, else it stays a bench result.
