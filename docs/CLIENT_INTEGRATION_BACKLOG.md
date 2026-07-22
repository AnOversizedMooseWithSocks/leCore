# CLIENT INTEGRATION BACKLOG — from the comfy-lecore audit (2026-07-16), cross-referenced 2026-07-17

*A downstream builder (ComfyUI node pack) audited the published wheel and a clone and sent 16 items. Every claim
spot-checked here reproduced against the live tree. This file records what is NEW, what we already did, and one
important correction to our own recent work. Items keep the client's numbering (C#) for traceability.*

**THE CORRECTION FIRST (honesty over comfort).** The "87/2,095 (4%) tagging" number that kicked off the semantic
arc is the client's **io-kind** count (`consumes`/`produces` — pipeline edges, the thing nodegen turns into typed
nodes). The semantic **verb** tags I raised 108 → 649 are a DIFFERENT system (the browse_semantic action menu).
Both gaps were real; the verb work stands on its own merits — but the client-valuable one, the one where "every
tag is a free node," is io-kinds, and it is still at **116/2,099 (5.5%)**. That drive is now S7 below.

---

## Verified live (all reproduce on this tree)

- C6's nonsense route: `suggest_pipeline("image","mesh")` → tensor denoiser → Aharonov-Bohm ring. Real.
- C11's trap: `mind.restore(y, mask=None, samples=None, forward=None, ...)` is the inverse-problem solver, one
  keystroke from `load()`. Real.
- C2: no `make_camera` / `make_mesh` / `make_box` / `make_sphere`; no `mind.invoke`; no `job_submit`; no
  `features()`. `CameraController` lacks `projection_matrix()` — the two-camera-class trap is real.
- C13: `grep` shows the engine never calls Python's `hash()` (only docstring mentions + a GLSL string named
  `hash`). The PYTHONHASHSEED=0 requirement is likely already vestigial — needs PROOF, not code (C13 below).

## Already addressed by us (client hit older versions)

- Their wheel/clone counts (488/888 caps) predate this tree (2,099). C1 (publish) subsumes the staleness.
- C8 overlaps our reachability program: we audit import-only entries as declared negatives INTERNALLY, but the
  client-facing half — a flag in `find_capability` results so `/tools` stops advertising uninvokable entries —
  does NOT exist. Filed as C8 with that scope only.
- Nothing else on their list was done or on our backlogs. Their #5/#6 seed S7 in SEMANTIC_BACKLOG.md.

---

## P0 (their order kept — it's correct)

**C1 — Publish the current engine to PyPI.** `leos-core` 0.1.0 is ~400 caps behind and missing the whole
catalog-navigation API (`pipeline_map`, `suggest_pipeline`, `io_kinds`, `browse_capabilities`, `set_file_root`,
job faculties). 53 of their 59 nodes are dead on the wheel. Mostly a RELEASE action (package.yml exists);
acceptance = their preflight snippet passes on `pip install -U leos-core`. **Owner: Moose (publish), us (verify).**

**C2 — JSON-drivable object faculties. ✅ DONE — and the diagnosis was wrong in a useful way.**
RULE 0 FIRST: the constructors were NOT absent. `m.render_mesh(m.mesh_box(), m.camera(...))` already rendered
in-process with no class imports, and `m.camera(...)` returns a Camera WITH `projection_matrix`. Three bugs wore
one costume: (a) DISCOVERABILITY — `find_capability("make a box")` answered "Catmull-Clark subdivision", never
`mesh_box`, so a working capability read as missing (fixed: D1-pattern aliases written from the client's own
words; all 7 of their phrasings now resolve top-3); (b) COERCION — dict args died deep in the rasteriser (fixed:
`holographic_coerce.as_mesh/as_camera` at the FACULTY boundary only; the renderer stays strict; real objects pass
through by IDENTITY so nothing existing shifts); (c) THE TWO-CAMERA TRAP — `CameraController` lacks
`projection_matrix` but has ALWAYS carried `to_camera()`; nothing called it. We call the bridge rather than mint a
third camera class. **Their acceptance passes over real HTTP** — and note it did NOT need C4: with coercion the
mesh travels as JSON, so the flagship works over the wire today. 23 tests.
*(superseded description below kept for the record)*
**C2 — JSON-drivable object faculties.** The flagship `render_mesh` cannot be called by ANY JSON client including
our own `/invoke`: it needs live `Mesh`/`Camera` objects. Fix BOTH ways (they compose): (a) faculty constructors
`make_camera/make_mesh/make_box/make_sphere` returning real objects; (b) coercion — `render_mesh` accepts
`mesh={'vertices':...,'faces':...}` / `camera={'eye':...,'target':...}`. PLUS: unify the camera protocol —
`CameraController` must satisfy `projection_matrix()/view_matrix()` or be documented as not-a-Camera; today it
fails deep inside the rasteriser. Full ritual per faculty; acceptance = their HTTP round-trip.

**C3 — `mind.invoke(name, args)`. ✅ DONE.** Hoisted onto the mind with Service._invoke's EXACT semantics
(public-only, callable check, kwargs-or-positional), and `holographic_service` now DELEGATES — pinned by a test,
so the two can never re-fork. Raises ValueError rather than returning something mistakable for a result. Their
`runtime.invoke()` already checks `hasattr(mind, "invoke")`, so it hands the job back with no pack change.
*(superseded description below kept for the record)*
**C3 — `mind.invoke(name, args)`.** Dispatch exists only inside `holographic_service.Service._invoke`; every
non-HTTP client re-implements it. Hoist to a faculty (public-only, callable-check, kwargs), make the Service
delegate. Small, high-value, removes a whole class of client drift.

## P1

**C4 — Object handles over `/invoke`. PARTIALLY OBVIATED — re-scope before building.** Their note said C2's
acceptance "also needs item 4 to pass over HTTP". It does not: MEASURED, `POST /invoke render_mesh` with a JSON
mesh + JSON camera returns an image today, because coercion lets the object be BUILT server-side from JSON
instead of shuttled as a handle. C4 remains real only for chaining outputs that are expensive or lossy to
serialise (`import_asset` → `render_mesh` on a large mesh). Worth measuring the payload cost before building a
registry + TTL: if JSON round-trips are cheap enough for realistic meshes, handles are a complication we skip.
*(original description below)*
**C4 — Object handles over `/invoke`.** `_jsonable` flattens a Mesh to a repr string, so remote mode cannot chain
rich objects. Server-side registry: `{"__handle__": id, "type", "summary"}` out, `{"__handle__": id}` in, DELETE +
TTL. Acceptance: `import_asset` → `render_mesh` over two HTTP calls without the mesh becoming JSON.

**C6 — tag lint. ✅ DONE, and it found the real cause.** `tools/tag_lint.py` + a CI gate. The nonsense route had
TWO causes, neither a typo: (1) FAKE EDGES — both edge builders read consumes×produces as a CROSS PRODUCT, right
for a CONJUNCTIVE capability (transform_selection needs mesh AND selection AND transform) but wrong for a
POLYMORPHIC one (denoise_tensor: image|field → THE SAME kind, so image→field is impossible). New
`polymorphic=True` keeps only the diagonal. (2) THE REAL ROUTE WAS UNTAGGED — `image_to_mesh`/`depth_to_mesh`/
`photo_to_3d` each say "image → MESH" in their own docstring and declared nothing, so no honest image→mesh edge
existed and the router took the dishonest one. Now tagged. **`suggest_pipeline("image","mesh")` returns
`depth_to_mesh`** — a route that runs. Also exposed a DUPLICATE: `pipelinemap._edges` and `suggest_pipeline` each
built the edge set and had already diverged (the fix landed in one; the nonsense route survived in the other) —
now pinned equal by test. KEPT NEGATIVE: your "smoke-called result's kind" is deliberately NOT built — fixtures
per kind would be a second engine to maintain, and the 3 static checks caught the entire reported failure. KEPT
NEGATIVE 2: the lint's first rule flagged any consumes/produces overlap → 5 false positives on legitimate
conjunctive tags; a lint that cries wolf gets muted. Rule tightened to `consumes == produces`.
**C5 → S7 — io-kind drive. ✅ BATCH 1 DONE (ongoing).** 91→98 both-tagged; 104→111 edges; **`timeseries`
source-only gap CLOSED**. New real routes: `field→mesh` (occupancy_to_mesh), `field→hypervector`,
`hypervector→field`, `skeleton→image` (skin_mesh→render). Every tag verified by READING the signature +
docstring — the `X_to_Y` convention LIES about types (`mesh_to_stl`→a string, `mesh_to_softbody`→a SoftBody,
`field_to_splats` takes *centers* not a field), so name-inference would manufacture the fake edges the lint
exists to catch. Four candidates abstained and pinned by test. `curve`/`skeleton` stay source-only HONESTLY:
nothing in the engine produces them (you import a skeleton, you draw a curve), and inventing a producer tag to
empty a report is the exact dishonesty we're guarding against. Their evidence shows a
wrong tag actively poisons Auto Route, so the LINT precedes the DRIVE: CI check that declared consumes/produces
match introspected signatures / a smoke-called result's actual kind. Then raise 116/2,099 the same way the verb
tags were raised — inferred where unambiguous, ABSTAIN otherwise, hand-tag the high-yield families first
(mesh_and_geometry 400, rendering 305, simulation 162, sampling 129). Also give `curve`/`skeleton`/`timeseries`
at least one consumer each (C16c) so `gaps.source_only` empties. Every tag is a free typed node downstream.

**C7 — `method` field in the catalog. ✅ DONE — and it subsumes C8.** `register_capability(method=...)`, DERIVED
when omitted (name if it's a bare identifier, else the `m.foo(` in the example — the same regex you were running,
now once, in the engine). The part no client could do: `seed_from_mind` VERIFIES every guess against a live mind
and nulls the liars — **511 of 2,040 naive guesses were wrong** (508 module names + 3 strays), now 0. `method=None`
IS your item-8 callability flag, so one field answers both questions instead of two that could disagree.
`pipeline_map()` edges carry it; your acceptance passes. Note: **exactly 4 edges have `method=None`** — the same 4
in your EXCLUDED.md. They were never regexable; they're genuinely import-only, and now they say so.
*(superseded description below)*
**C7 — `method` field in the catalog.** 33 of the tagged entries are prose-titled; the client regexes the callable
out of the example string (fragile, 4 exclusions). Add `method=` to `register_capability` + `pipeline_map` edges.
Acceptance: `all(callable(getattr(m, e["method"])) ...)`. Cheap; do together with S7 so new tags carry it.

**C8 — Callability flag. ✅ DONE via C7's `method` field** (None = import-only). 574 entries honestly flagged,
matching your reported ~570. `find_capability` results carry it; nothing advertises an uninvokable entry as callable.
*(superseded description below)*
**C8 — Callability flag on catalog entries.** 570/888 of what `find_capability` returns needs a direct class
import — fine for us, a dead end for a JSON client. Either wrap the useful ones as faculties or tag
`client="import-only"` so search results and `/tools` are honest. Acceptance: zero unflagged-but-uncallable.

**C9 — Structured args. ✅ DONE.** `params: [{name, kind, required, default}]` from `inspect` (which knows
*args/**kwargs, unlike a signature-string split), plus `primary` (the param the piped input goes to — first
required positional) and `produces`. `default` is repr'd so the card stays JSON-safe (a tuple default like
(0.8,0.8,0.8) would otherwise hand a JSON-dumping client a landmine). Fixed an inconsistency found while
building: `describe_skill("render_mesh")` resolved through the METHOD index and served params, while
`image_to_3d` is ALSO a catalog entry so it returned through the CAPABILITY branch and served none — same
question, two answers, decided by which index held the name. `method` now appears on both card kinds meaning the
same thing.
*(superseded description below)*
**C9 — Structured args in `describe_skill`/`/tools`.** Params as data (`name/kind/type/default/required`),
plus the two fields that kill client guessing: `primary` (which param takes the piped input) and `produces`.
Introspectable from signatures + _IO_SHAPES for most faculties.

**C10 — Generic `job_submit(name, args)`. ✅ DONE.** Your instinct was right — plumbing: one bucket, a new
`first` (identity) reduce, a worker that calls `mind.invoke`. Two things came out of building it that weren't in
the report. (1) `reduce="first"`: sum/min/max/bundle are for work that DECOMPOSES; `reduce_sum` happens to return
parts[0] for one part, so an atomic job would have "worked" while calling itself a sum and copying its own
result. An atomic job now says so and refuses >1 partial. (2) **A real bug**, found by measuring a claim in my own
docstring instead of asserting it: `save()` runs inside the daemon thread, and a live object in `args` (legal —
it computes the right image) made `json.dump` raise TypeError, crashing the worker thread with a traceback the
caller couldn't catch. The job had already SUCCEEDED; only the bookkeeping exploded, on stderr. Persistence is a
BONUS property, not a precondition for running — it now degrades to `persisted=False` with the reason recorded.
Bad names raise at SUBMIT, not swallowed into a failed job you must poll to discover.
*(superseded description below)*
**C10 — Generic `job_submit(name, args)`.** The whole job surface (list/status/result/cancel/pause/resume)
exists; only arbitrary START is missing — today `background=True` works solely for faculties that accept it.
Jobs are already checkpointed monoid folds; this is plumbing.

## P2 (foot-guns; each small)

**C11 ✅ DONE.** `save_state`/`load_state` aliases (delegating, no second implementation) + "not to be confused
with" cross-references across save/load/to_state/from_state/restore. **Found a WORSE trap you didn't report:**
`load` is a CLASSMETHOD that RETURNS A NEW MIND — so the pattern you described ("the Loader constructs then calls
load()") silently loads NOTHING unless you capture the return. `m.load(p)` leaves `m` untouched and empty.
Documented loudly at the call site.
*(original)* `save_state`/`load_state` aliases + "not to be confused with" cross-references in `restore`/`load`/
`to_state`/`from_state` docstrings. The trap is real; we nearly have the same confusion internally.
**C12 ✅ DONE (it already existed).** `UnifiedMind.load(path)` IS construct-from-file — a classmethod, just not
under a name anyone searches for. Added `UnifiedMind.from_file(path)` as the name you asked for; one
implementation, two honest names.
*(original)* `UnifiedMind(state_path=...)` or `UnifiedMind.from_file(path)` for atomic construct-and-restore.
**C13 ✅ DONE — and the answer is the OPPOSITE of what we both assumed. THE REQUIREMENT WAS NOT VESTIGIAL.**
Your grep found no `hash()`; my first grep agreed. **Both were wrong.** `holographic_sequence` seeded atom vectors
with `abs(hash((seed, sym)))` — salted per process, so `discover_sequential()` returned **1.64 / 1.61 / 2.17 on
identical input** under `PYTHONHASHSEED=random`. It survived greps because `hash((seed, sym))` looks nothing like
a content hash, and it survived every test because CI pins the env var — which papered over the bug rather than
preventing one. Fixed with hashlib; the static trap I added then found **two MORE** sites in
`holographic_recurrent`. A grep found one, the trap found the rest. **The engine is now deterministic with
PYTHONHASHSEED unset** (proven in subprocesses across random salts — an in-process assertion is blind to its own
salt, which is why this lived so long). The z VALUE moved 2.8379→2.2325 (atoms are differently seeded); the
DECISION did not flip. **You can drop the env var**; it's belt-and-braces now.
*(original)* PROVE determinism without `PYTHONHASHSEED=0` (the grep says the engine never calls `hash()`): run the
determinism suite with the var unset and salted hashing active; if green, delete the requirement from docs and
session ritual; if anything fails, THAT is the bug to fix. Either outcome retires an env-var footgun.
**C14 ✅ DONE.** `mind.features(names)` -> `{name: bool}` answers your whole preflight list in ONE call;
`mind.features()` maps every public faculty to True; `mind.version()` -> `{engine, capabilities_schema, dim,
seed}`. Private names always False — they're not part of the contract. You were right that a hardcoded list rots,
and the sharp edge is that it rots SILENTLY: a missing faculty and a renamed one are the same absent attribute
from outside.
*(original)* `mind.features()` → `{name: bool}` manifest (or a catalog-schema version) so clients stop hardcoding
preflight lists that rot.
**C15 ✅ DONE.** The contract is now stated in `io_kinds()`'s own docstring, where you read it: **STABLE** —
mesh, points, sdf, sdf_scene, field, image, hypervector, transform, selection, scalar (live edges; build UI
against them). **PROVISIONAL** — curve, skeleton, timeseries, spectrum. curve/skeleton have NO tagged producer
(you import a skeleton, you draw a curve), so they're `source_only` in gaps — an honest gap, pinned by a test that
fails loudly if it ever changes, so the doc can't go stale. The vocabulary only GROWS; a rename would break every
tag at once, which is why the list is validated at registration.
*(original)* Once C1 ships, document the `io_kinds()` contract (stable vs provisional kinds).
**C16 ✅ DONE.** (a) `resolve_capability_uri`'s docstring now LEADS with "URI-ONLY — returns [] for a plain
faculty name"; behaviour unchanged. (b) `render_mesh(dtype=)` added, default None = float64 byte-identical; the
cast happens at the EXIT (after the shading maths) so it cannot move a pixel — proven by test. (c) source_only
consumers: `timeseries` closed in S7 batch 1; curve/skeleton stay open honestly (see C15).
*(original)* smalls: (a) `resolve_capability_uri` docstring: URI-only, say so; (b) `render_mesh(dtype=)` to skip the
float64→float32 copy; (c) consumers for curve/skeleton/timeseries (folded into S7).

## Not ours to fix (client handles; do NOT "fix" upstream)

Their empty `suggest_pipeline("image","image")` (correct: the empty already-there pipeline; `require_step=True`
is the knob), their JSON-safe coercion, their Capability-vs-dict tolerance.

## Suggested execution order (theirs, amended with S7 pairing)

C1 (Moose publishes) → C2 → C7+C6-lint → S7 drive → C3+C9 → C4, with P2 smalls slotted into close-outs.
