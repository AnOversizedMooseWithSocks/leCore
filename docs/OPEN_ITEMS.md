# OPEN_ITEMS.md — the consolidated live backlog (verified against the running engine 2026-07-19)

Every item below was swept out of the scattered backlogs (BACKLOG.md, the client + semantic backlogs, the
promotion ledger, and the docs archived into NOTES) and then **probed against the live engine** — because the
recurring failure here is an item that was completed in another branch while its backlog line stayed open. Each
item carries its VERIFIED status: `OPEN` (probed, genuinely not built), `MOOSE` (owner action), or a pointer to
where it already shipped. This file supersedes the OPEN-WORK sections of the other backlogs as the single place to
look; the source docs keep their own history.

The audit rule that earned this sweep: **"open" is a claim to verify against live code, not a property of an
unchecked checkbox.** Three items below were marked open across the docs and are in fact DONE — see the last
section so they are not re-listed a third time.

---

## OPEN — verified not built, ready to pick up

### M9 inc3 — creature identification (shape signature resonated against archetypes)  **[RESEARCH ARC — not a knock-out]**
The VSA-meets-geometry research item: a shape signature (EGI + skeleton spectrum + proportions) as a hypervector,
resonated against archetype prototypes — "find_capability FOR SHAPES," feeding M2's rig template. Ingredient audit
2026-07-19: EGI exists as `mesh_egi_compare` (COMPARE only — the histogram is internal to
`holographic_render.egi_similarity`, not exposed as a standalone descriptor); skeleton via `skeleton_curve`;
segmentation via `mesh_parts`; VSA bind/bundle exist (`map_bind`, `superpose_batch`, `fpe_lattice_resonator`).
MISSING and genuinely un-built: a standalone shape-signature VECTOR, a prototype STORE, and the resonance/nearest
match — plus the measured baseline proving it DISCRIMINATES shapes (the whole point, and where the research risk
is). This is a full-ritual arc with its own baseline and kept negatives, NOT a bounded close-out; scoped here so it
is picked up as research, not rushed. VERIFIED OPEN: no `identify_creature`/`shape_signature`/`classify_creature`.

### M10 — standing debts (small, real)
- **Rebake perf wall:** ~~244.6s / 3002 faces~~ **RESOLVED** — `rebake_texture(method="scatter")` is the H1
  scatter/gather path (36× end-to-end), shipped in a branch; the 244s figure was the old per-texel route.
- **`mesh_orient` `components` alias:** HANDLED 2026-07-19, additively. Dropping the key now is a BREAKING change
  and C1 (publish) has not shipped, so no external caller has had a release to migrate. Instead the report carries
  a machine-readable `_deprecated: ("components",)` marker (pinned by selftest), so a post-C1 "drop deprecated
  keys" pass is mechanical. Nothing internal reads the deprecated key (verified). Drop AFTER C1, not before.
- **Poly Studio D-section — flat-vs-packaged import duality re-audit.** PARTIALLY DONE: the branch shipped
  `holographic_deptrace` (`import_footprint`) as the measuring instrument, its test recording the "import balloon"
  premise was REFUTED by measurement. Only the flat-vs-packaged duality re-audit remains, if wanted — probe
  before filing as build work (it may be another measured no-build).

### C1 — publish the current engine to PyPI  **[MOOSE]**
`leos-core` on PyPI is ~400 caps behind (`setup.py` is at 0.2.0; the live tree has 2233 caps) and missing whole
faculty families. 53 of the client's 59 nodes stay dark until this ships. Acceptance = the client's preflight
snippet passes on `pip install -U leos-core`. Owner: Moose (publish), us (verify).

### C4 — object handles over `/invoke`  **[re-scope before building]**
`_jsonable` (holographic_service.py:195) flattens a Mesh to a repr string, so remote mode cannot chain an object
output into the next call. PARTIALLY OBVIATED: with server-side coercion the flagship path never needed handles
(C2 passed over real HTTP without C4). C4 stays real ONLY for chaining outputs that are expensive or lossy to
rebuild from JSON. Measure that need on a concrete client workflow before building; do not build blind.

---

## GATED / KEPT-NEGATIVE — do not re-open without new evidence

### S4 — front-door retrieval hybrid, embedding arm
The 35-ask `catalog_exam.py` instrument is BUILT and the token baseline PINNED (14/35 top-1). The embedding arm
is gated on S2 — which **RAN AND FAILED**: at 128d the learned map scored 1/12 top-1, WORSE than the no-map floor
(3/12). N31 free-text routing is a KEPT NEGATIVE (see SEMANTIC_BACKLOG S2). Do not re-attempt the distilled map;
any reopening starts from the bigger ask set, not a cleverer map. The non-embedding arms (BM25, RRF) were measured
and are also kept negatives (+1 ask ≈ 0.5 SE, noise).

### M1 inc3 — cross-level triangle pairing   **[MEASURED NO-BUILD 2026-07-19]**
The backlog planned a second pass to pair the ~20 excess triangles a graded level boundary leaves, to recover
quad_fraction (a real gap: graded 0.71 vs uniform 0.77, persistent across box and other fixtures). REFUTED by
measurement: `quad_remesh`'s existing gate (edge shared by exactly 2 tris, near-coplanar, convex) ALREADY pairs
every geometrically-valid boundary pair. Of 15 leftover tri-pairs on the split box, ZERO are coplanar+convex; even
dropping coplanarity only 3 are convex, and those fold across the box's real edges (dihedral dot to −1.0). A
second pass recovers nothing (same gate) or makes folded/skew quads (looser gate) — strictly worse than an honest
triangle. The residual gap is INHERENT to grading: a level boundary needs transition triangles to change cell
size. If graded quad_fraction ever must rise, the lever is a FINER level field, not tri-pairing. Kept negative
recorded in `extract_quads`' docstring.

### P7 — FFT "diagonal evolution" primitive (promotion ledger)
The audit ("is there one diagonal-evolution primitive hiding in transfer/step_k/diffuse_periodic?") was DONE as
M8 — verdict CONFIRMED audit-only, the reuse is mostly correct np.fft, not fragmentation. Resolved, not open;
listed here only so the ledger's open-looking header is not mistaken for live work.

---

## CLIENT INTEGRATOR BACKLOG (11 items) -- RESOLVED 2026-07-19

All eleven items from the integrator's fresh backlog closed. Six built/fixed, five already-solved-or-doc.

BUILT/FIXED: [P1] terrain.erode runaway (default 4.0->1.0 + height-scale invariance; runaway trap pinned) ·
[P3a] rasterize_mesh coerces dict cameras · [P3b] mass_properties inertia_tensor alias · [P2]
LoadedMesh.split_by_material (wired m.split_by_material, catalogued, convert/split) · [P2] rebake size="auto" +
helpful error · [P3] SDF_COOKBOOK.md (+ SDF.__call__ ergonomic fix).

ALREADY SOLVED / DOC: [P2] optional-dep balloon -> import_footprint already gives numpy-only required set (doc'd
in PACKAGING.md) · [P3] capabilities.json/data discovery -> already __file__-fallback robust, bare import only in
a test · [P2] flat-vs-packaged duality -> code already ~consistent on packaged (2942 vs 13); doc'd canonical
style + flat shim recipe · [P2] analytic-preservation -> Mesh has no sdf_tree to drop; contract doc'd in cookbook.

NEW SIGNAL for C4: split_by_material is the first LoadedMesh-consuming faculty, and its HTTP /invoke fails because
a LoadedMesh cannot be rebuilt from JSON -- a CONCRETE motivating workflow for C4 (object handles over /invoke),
which had been waiting for exactly such a case to re-scope against. C4 remains open, now with a real driver.

## FOUND ALREADY DONE during this sweep — completed in a branch, backlog never updated

These were carrying OPEN-looking markers across the docs and are verified SHIPPED in the merged tree. Recorded so
they are not swept up as "open" a third time.

- **M16 — global worst view.** DONE: `worst_view(metric, mode="direct", ..., lipschitz=None)` — the Lipschitz /
  DIRECT-on-sphere branch-and-bound that M16 filed as the ranked "next attempt." Catalogued, wired, tested in
  test_cad_backlog. Its backlog entry read as an unbuilt research item.
- **M9 inc2 — branch segmentation.** DONE: `mesh_parts` (Reeb-graph limb/body segmentation — the branch REFUTED
  the voxel-ridge approach the backlog planned, for thin limbs) + `match_symmetric_parts`. "mantis 12 parts + 3
  symmetric pairs."
- **Promotion ledger P1–P5.** All shipped and the ledger header is stale: P1 `holographic_numerics.cg`,
  P2 `low_eigenvectors`, P3 `bisect_to_budget`, P4 `walk_knob` (meshqem, a promoted MODULE primitive with 4 live
  callers -- search helpers need no /invoke faculty), P5 `flood_fill_sign` (meshbridge). P6 was CLOSED (not a
  duplicate). Only P7 remained, and it resolved as audit-only (above).
