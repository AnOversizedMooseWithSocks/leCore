# BACKLOG.md -- the single master backlog and entry point for remaining work
<!-- Predecessors PLAN_retopo.md and PROMOTION_LEDGER.md were folded into this file and then archived into
     docs/NOTES_concepts.md (see "ARCHIVED BACKLOGS") once complete; do not recreate them. -->

Merged 2026-07-17 from (a) the Poly Studio demo backlog (engine gaps found by building an app ON leCore) and
(b) the retopo arc. Every item below was AUDITED against the live tree, not against either document's own
words. The audit rule that keeps earning its keep: **an item justified by a grep of English is a hypothesis,
not a finding** -- it must be confirmed against live code before it gets scoped, let alone built.

---

## 0. CLOSED -- the entire original Poly Studio backlog. Verified live, not remembered.

| item | faculty | status |
|---|---|---|
| A1 CAD mass properties (Tonon covariance) | `m.mass_properties` | DONE |
| A1 Oriented bounding box | `m.oriented_bbox` | DONE |
| A2 Cross-section measure | `m.mesh_section` | DONE |
| A2 Draft-angle / moldability report | `m.draft_report` | DONE |
| A2 2D constraint solver | `m.sketch2d` | DONE |
| A2 Camera from vanishing points | `m.camera_from_vanishing_points` | DONE |
| A3 Hydraulic erosion | `m.terrain_erode` | DONE |
| C1 NodeGraph.remove | catalogued (node-graph editor backend) | DONE |
| C2 Subgraph collapse/expand | catalogued | DONE |
| C2 C backend for zigrun (ccrun donation) | catalogued (native batch kernels via system C compiler) | DONE |

The B-section ergonomics (emissive/emission, SDF combinator asymmetry, extrude half-height, socket 'a') are
naming/doc items -- confirm each against the live docstrings before re-filing; do not assume they are still true.

**ONE B-ITEM IS NOW ANSWERED BY THIS ARC, and it is the headline "promote, don't re-specialize" result:**
B said *"terrain axis swap is a reflection, inverts face winding, I had to reverse triangle order by hand;
generators should emit consistent winding."* That is EXACTLY `m.mesh_orient`, built this arc for retopo. The
demo hand-rolled a terrain-specific reversal; the general move now exists. **ACTION (M3 below): make the
terrain/landscape generators call mesh_orient rather than each generator carrying its own flip.**

---

## 1. DUPLICATES -- found by auditing THIS arc's own output (the owner's instruction, applied backwards)

* **[DONE] `is_oriented` vs `face_orientation_report`.** I introduced a second checker for one property:
  isosurface.is_oriented is QUAD-ONLY (indexes q[0..3] literally, so it cannot read a triangle mesh -- every
  scan and decimation output), while the new face_orientation_report is general-degree and supersets it.
  FIXED: is_oriented now DELEGATES; name+signature kept; parity with the historical quad counter pinned.
* **[DONE] `_cg` x2** -> holographic_numerics.cg (ledger P1). Both callers delegate; real path bit-identical;
  crossfield's matvec count unchanged at 2232.
* **[NOT a duplicate, recorded so it is not "fixed"]** numerics.cg (Hermitian-PD, matvec) vs sketch2d
  (Gauss-Newton/lstsq) vs iterate.dominant_eigenvector (Fourier-diagonal closed form). Three different moves.
  Correct specialization is not fragmentation.
* **[NOT a duplicate]** P6: holographic_adaptive_sample (renderer per-pixel STOP RULE) vs R4 sizing (how big a
  quad is). Homonym. Fusing them would be the ledger's failure mode inverted.

---

## 2. OPEN WORK -- merged, holographically framed, ordered by leverage

> **LIVE STATUS lives in docs/OPEN_ITEMS.md** (verified against the running engine 2026-07-19). Most items in this
> section are DONE and kept here for their research/measurements; the genuinely-open remainder (M1 inc3, M9 inc3,
> M10 debts) is consolidated there. A 2026-07-19 sweep found M16 and M9 inc2 already SHIPPED in a branch with this
> list never updated -- so read OPEN_ITEMS.md for what is actually open, this section for how each was reasoned.

### M1 -- Graded sizing   [INCREMENTS 1+2 DONE; inc 3 mostly DISSOLVED by baseline 2026-07-17]
Uniform lattice spacing forces singularities on a curvature-varying surface: measured quad_fraction 0.48 on
the ladybird. The input EXISTS (`m.mesh_curvature`) and needs no promotion. THE BLOCKER IS STRUCTURAL, and my
own plan hid it behind the words "minor edits": position_field's operator is
`a = round((P[i]-P[j]) @ O[j] / rho); acc += P[j] + rho*(a*O[j] + c*Bt[j])` -- ONE GLOBAL rho, and the whole
construction rests on neighbours differing by an INTEGER number of rho-steps. If rho_i != rho_j that jump is
undefined and extract_quads' key round(P/rho) has no rho to divide by. Options ranked: (a) power-of-two
grading rho0*2^k so a coarse step IS two fine steps (the paper's answer); (b) per-component uniform rho +
stitch; (c) live with 0.48 (today). DONE WHEN quad_fraction rises above 0.48 at equal-or-better silhouette
with the grading pinned -- never by tuning the extractor to hide triangles.

INCREMENT 1 DONE (2026-07-17): graded_levels(mesh, target_edge, rho0) -- the balanced power-of-two size field. rho(v)=rho0*2^k(v), k seeded from the target edge (fine target -> high level), then 2:1-BALANCED so |dk|<=1 across every edge via a lowering-only relaxation (monotone, terminates, deterministic). VERIFIED: |dk|<=1 from a 4-level-apart target; grades toward the fine side; on the ladybird produces 1803/1912/919 verts across 3 curvature-driven levels. Faculty m.graded_levels, catalog 579, 6/6.
PREMISE CONFIRMED: ladybird curvature |H| varies 7.8x (p90/p50), so uniform sizing IS wasteful -- M1 is worth finishing. And a MATH CORRECTION worth keeping: I first claimed |dk|<=1 preserves POINT commensurability -- that is WRONG (a level-0 point at odd rho0-multiples is a half-integer in the 2x lattice). The real invariant is CELL-WALL nesting: 2^k lattice walls are nested for ANY dk (every coarse wall is a fine wall), so cells align; the level boundary makes a HANGING NODE (fine wall inside a coarse cell), and |dk|<=1 caps it to ONE per coarse edge. So the balance rule is about STITCH SIMPLICITY, not commensurability. This reframes increment 3 (the T-junction stitch = insert the one hanging-node midpoint into the coarse loop).
REMAINING: increment 2 = the graded position_field operator (evaluate the integer jump in the COARSER of two vertices' lattices: rho_e = rho0*2^max(ki,kj)) + level-keyed extraction (key = (cell, level)); increment 3 = the T-junction stitch. Both build ON graded_levels, now pinned.

INCREMENT 2 DONE (2026-07-17): the GRADED OPERATOR and LEVEL-KEYED EXTRACTION, both additive strict supersets. position_field(levels=) evaluates each edge's integer jump in rho_e = rho0*2^max(k_i,k_j) (the coarser of the two lattices); extract_quads(levels=) keys each vertex as (round(P/(rho0*2^k)), k) so a coarse cell never merges with a fine cell that rounds the same (the level is a distinct key axis -- the boring-axis-elevation move: level is a CARRIER, not bound into the spatial key). PROVEN strict supersets: levels=None is bit-identical (surface_retopo pin 328/0 holds); a UNIFORM level k equals rho0*2^k scaling exactly (both position_field and the whole retopo); VARYING levels produce a valid, CLOSED, adaptively-sized mesh (299 faces / 0 boundary / quad 0.70 on the split-box vs 368/0.84 uniform). HONEST NOTE, not hidden: graded quad_fraction DIPS below uniform (0.70 < 0.84) here -- EXPECTED without increment 3, because the un-stitched level boundaries triangulate. The mesh is valid and closed (what inc 2 guarantees); quad quality at the boundaries is increment 3's job. NOT tuned to hide that.
REMAINING: increment 3 = the T-junction stitch. At a level boundary a coarse cell edge has ONE fine hanging node interior to it (|dk|<=1 guarantees exactly one); insert that midpoint into the coarse loop, turning the coarse quad into a 5-gon that closes cleanly against the fine cells. This is the quad-fraction fix. Everything it needs (graded field, level key, the balance guarantee) is now built and pinned.

INCREMENT 3 FULLY RESEARCHED (2026-07-17) -- it is a BOUNDED pairing extension, not a T-junction re-mesh. Measured on the graded split-box (levels 0/1): 68 cross-level source edges (all |dk|=1, the balance guarantee holds); a coarse cell abuts up to 3 fine neighbours (mean 1.8) -- so the plan's 'always a 5-gon' was wrong; it can be a 5/6/7-gon. But the actual quad DEFICIT is small and specific: graded has 89 triangles vs uniform's 60 (~29 excess), and only 9 faces are dropped at boundaries. Of the 89 graded tris, 35 have a triangle NEIGHBOUR and 20 share an edge as mergeable pairs. RULE 0 FINDING: extract_quads ALREADY pairs tris internally, and quad_remesh is catalogued 'field-guided tris-to-quads' -- the pairing move EXISTS. So increment 3 = EXTEND extract_quads' existing tri-pairing to also consider cross-LEVEL boundary tri pairs (coplanar+convex guarded), recovering the ~20-35 pairable excess tris. TARGET METRIC: graded triangle count back toward uniform's ~60 (quad_fraction 0.70 -> ~0.83) at unchanged silhouette and 0 boundary. NOT a new algorithm -- a generalisation of the pairing already in the extractor to the level-boundary case. Build shape: after the existing pairing, a second pass over unpaired boundary tris that share an edge, merging coplanar-convex pairs into quads; default-off behind the same levels= path so uniform is untouched. This is the last piece of M1's paper-hard part, now specified from measurements not the plan's guess.

INCREMENT 3 LARGELY DISSOLVED (2026-07-17) -- the 'deficit' was a BASELINE ARTIFACT. I compared graded quad_fraction (0.70, 89 triangles) against the FINE-uniform mesh (0.84, 60 tris). Wrong baseline: the graded mesh is MOSTLY COARSE, and a COARSE-uniform mesh alone has 87 triangles (a coarser lattice has proportionally more tris per quad from the field's singular cells). So graded's TRUE boundary excess is ~16 triangles, not 29 -- and the prototype tri-pair merge found ZERO of them coplanar-convex-mergeable, because they are genuine level-boundary TRANSITIONS where a flat quad would distort the surface (the T-junction, doing its job). So there is little to fix: the graded mesh is valid, closed, and its quad fraction is APPROPRIATE for its mixed resolution -- comparing it to the fine-uniform quad fraction was the error, the same wrong-baseline mistake as the fit_camera '4x' and M11's '167 holes'. What remains of inc3 is OPTIONAL polish: an n-gon (5-gon) insertion at the ~16 boundary transitions if a downstream tool wants pure quads there, but it does not improve the mesh's correctness or fidelity. M1's functional goal -- adaptive sizing that refines curvature without breaking the extractor -- is MET by increments 1+2. KEPT NEGATIVE: 'graded quad_fraction < uniform' is NOT a defect; it is the wrong comparison. The right baseline is a uniform mesh at the SAME MIXED resolution, against which graded is on par.

### M2 -- Guide plumbing                                       [DONE 2026-07-17 -- verified, was already wired]
AUDIT RESULT: already wired, and the work was VERIFICATION, not construction. strain_directions(mesh,
deformed) exists (catalogued "retopo guide"), returns (n_faces, 3), and surface_retopo(guide_dirs=...) already
accepts it -- I built that param in R3. The plumbing FLOWS: strain -> surface_retopo -> guided=True, 197 faces.
LOAD-BEARING, measured correctly (the first metric was WRONG): guidance lifts the field's 4-RoSy alignment to
the strain from 0.092 to 0.999, climbing monotonically with guide_weight (0->1->5->20 : 0.092/0.997/0.999/
1.000). NOTE the measurement trap, pinned in the test: a single-representative |field . strain| dot reads an
ARBITRARY one of the four RoSy directions and can score an aligned field LOW (it read 0.572->0.455, "worse");
the correct metric is cos(4*(phi - guide)), aligned modulo 90 deg. Pinned end-to-end in
test_m2_strain_guides_steer_the_retopo_field.
KEPT NEGATIVE (vocabulary gap, not fixable by aliases): "put edge loops where it deforms" resolves to edge-
loop SELECTION tools, not the strain guide -- "edge loops"+"put" has too much lexical gravity toward
select_edge_loop. Same structural gap as "less grainy"->denoise; extended the other workflow phrasings
instead ("retopo that follows how the model bends" now resolves).

### M3 -- Reflection-aware transform                          [DONE 2026-07-17 -- after the filing was corrected]
CORRECTED BY MEASUREMENT, one turn after filing it. Two claims in the original M3, both false:
 1. "The terrain generator emits inconsistent winding." IT DOES NOT. Measured: terrain_to_mesh(Terrain(), 24)
    -> 1058 faces, ORIENTED True, 0 duplicated directed edges, normals 100% +Z. It is Z-up and says so.
 2. "mesh_orient is the cure." IT IS NOT, and this is the distinction worth keeping: the demo's Z-up->Y-up
    axis swap V[:, [0,2,1]] is a REFLECTION (det = -1). Measured, the swapped box is ORIENTED True with 0%
    outward normals -- CONSISTENTLY oriented and CONSISTENTLY INSIDE-OUT. mesh_orient repairs INCONSISTENCY
    (neighbours disagreeing); global inversion has no disagreement to find, so it correctly flips 0 faces.
    TWO DIFFERENT DEFECTS. I conflated them and would have shipped a cure that provably does nothing.
THE REAL GAP: there is no reflection-aware transform. m.transform_mesh / m.convert_up_axis are MISSING; the
demo hand-rolled the winding reversal because nothing owned it.
SHIPPED: m.transform_mesh(mesh, matrix) flips winding iff det < 0; m.convert_up_axis(mesh, "z", "y") rides on
it using a PROPER rotation (det=+1) so nothing inverts in the first place. Singular matrices RAISE rather than
silently collapse the mesh. Catalog 589, 6/6 stranger phrasings, convert/emit. The rule now lives in ONE place
instead of in every caller that reflects.
VERIFIED: reflection -> 100% outward preserved; rotation -> faces bit-identical (winding untouched);
convert_up_axis z->y -> 100% outward, oriented. KEPT NEGATIVE pinned in the selftest AND the pytest: the naive
column permutation still measures 0% outward while reporting oriented=True, and mesh_orient still flips 0 on
it -- the two defects stay visibly distinct so nobody re-conflates them.
ALSO FOUND, unresolved, needs its own audit: m.mirror_mesh(box, axis=0) returns oriented False -- 28
duplicated directed edges and 14 NON-MANIFOLD edges. Mirroring a symmetric box through its own centre plane
welds coincident geometry, so this may be correct-but-surprising rather than a bug. DO NOT "fix" it before
measuring against an asymmetric fixture -- that is exactly the assume-then-build this list keeps punishing.

### M4 -- Ledger P4: split `walk_knob` from the criterion      [DONE 2026-07-17 -- and it paid immediately]
SHIPPED: walk_knob(op, knob, passes, knob_cost, max_knob, ...) owns the SEARCH; `passes(out) -> (ok, fields)`
is the criterion, and it returns its own report vocabulary so the walk knows nothing about silhouettes.
silhouette_guarded is now ONLY the criterion (a 6-line closure). BIT-IDENTICAL, verified against the
decisions this arc RECORDED before the split: ladybird ask 3000 -> 6383 faces @ 0.969; opt-out -> 2914 exactly;
auto_retopo cubic refusal at knob 24, refused True, step 1.260. A split that moves an answer is a rewrite.
THE SPLIT PAID THE SAME HOUR: +topology_guarded is TWELVE LINES because the walk already existed. Measured, it
walks surface_retopo's density finer until the holes it punches actually close --
  knob 133 (density 1.504): 8 boundary edges, 452 degenerate cells -- holes
  knob 200 (density 1.000): 6 boundary edges, 192 degenerate cells -- holes
  knob 300 (density 0.667): 0 boundary edges,  66 degenerate cells -- CLOSED, a real pass walked to.
NOTE that this is a WORKAROUND, not M11's fix: the extractor still drops faces; a fine enough lattice just
drops fewer than it takes to punch through. M11 remains the real repair.

### M5 -- Ledger P5: promote flood fill / BFS   [RESOLVED 2026-07-17: graph flood already exists; wired + kept negatives]
The naive filing said "one move, numerics.flood_fill(mask_or_graph, seeds)". AUDIT REFUTES the fusion: the 22
sites split cleanly into GRAPH flood (11 sites: mesh dual-edge adjacency, no grid, no connectivity choice --
mesh_orient, connected_components, tear, ...) and GRID flood (5 sites: image mask, border-inward, with a 4-vs-8
connectivity CHOICE -- silhouette_mask, render). Exactly ONE file does both (meshtools). They share a deque and
NOTHING ELSE: a graph has no notion of 4- vs 8-connectivity, a grid has no notion of dual edges.
So the honest promotion is TWO primitives -- numerics.graph_flood(adjacency, seeds) and
numerics.grid_flood(mask, seeds, connectivity=4|8) -- that happen to share a queue, NOT one flood_fill. Fusing
them behind one signature would be the P6 homonym trap again ("flood" is the coincidence; the moves differ),
and worse, a single connectivity default would FLIP the grid sites that rely on 8 (silhouette_mask's border
flood is 8-connected -- pinned by the whole silhouette guard). 
DONE WHEN: both primitives exist; every graph site delegates to graph_flood with BIT-IDENTICAL component
counts (pinned), every grid site to grid_flood with its EXISTING connectivity (pinned), and the silhouette
masks are bit-identical before/after. Do it WHILE building M11's loop-walk extractor (a third graph consumer).

RESOLVED (2026-07-17) -- and TWO of the premises above were wrong on measurement:
(1) graph_flood does NOT need building -- it ALREADY EXISTS as holographic_island.connected_components,
whose own docstring says it is 'the generic flood fill under every island in the engine', and
route.connected_components ALREADY DELEGATES to it. So the graph-flood consolidation the plan scoped is
DONE; it predates this item.
(2) silhouette_mask's border flood is 4-CONNECTED, not 8 -- I read the code: it is a vectorised
array-shift dilation (up/down/left/right only, lines 76-79), NOT a per-pixel 8-neighbour stack. So the
'a connectivity default would flip the 8-connected silhouette' worry was based on a misread; the grid
flood is a specialised pinned dilation with no exposed connectivity knob to promote.
(3) The two remaining inline graph-floods do NOT cleanly delegate: meshseam's is seam-specialised
(set-based adjacency, excludes seam edges while building adj) and meshtools' is a HOMONYM -- it looks
like a flood but carries per-face WINDING-FLIP state computed DURING traversal (flip[gj] depends on the
neighbour's edge direction, order load-bearing via seed_face). connected_components cannot express
per-node walk-state, and forcing it would flip the pinned orientation result. KEPT NEGATIVE.
THE ONE REAL WIN, taken: island.connected_components had no direct faculty -- 'flood fill a graph'
surfaced 'Islands + sleep' (a sim concept), not the reusable primitive. Wired m.graph_connected_components
(thin delegate, no new logic) + catalog 577, 6/6. Same shape as P8/M8: the promotion the plan imagined
was mostly already done or a homonym; the actual gap was DISCOVERABILITY of the existing primitive.

### M6 -- Ledger P3: bisect_to_budget   [DONE 2026-07-17 -- promoted, both consumers delegate, both pins hold]
The ledger's "36 files match the shape" is the census trap: 151 files have lo/hi/mid code, but classifying for
the ACTUAL move (bisect a MONOTONE f(knob) to hit a numeric TARGET) leaves exactly TWO genuine consumers:
  * decimate_to (meshqem): ARITHMETIC bisection, mid=(lo+hi)//2, over an INTEGER grid knob, target=face count.
  * ratedistortion: GEOMETRIC bisection, mid=sqrt(lo*hi), over a CONTINUOUS scale, target=mean cosine. "The
    largest delta whose mean cosine still meets the target" -- decimate_to's move on a different quantity.
nodegraph's lo/hi are SDF bbox CORNERS, not a bisection (rejected). So this is a REAL promotion (2 consumers =
the threshold where it pays), NOT a homonym -- but the two differ in MIDPOINT TYPE (arithmetic vs geometric),
which the primitive must PARAMETERISE, not flatten (the P2/M4 lesson: parameterise the real difference).
BUILD SHAPE (specified so it is mechanical, but it is TIE-SENSITIVE so it wants a fresh context, not a tail):
  numerics.bisect_to_budget(probe, target, lo, hi, midpoint="arith"|"geom"|callable, max_iters, tol, cmp)
  where probe(knob)->value, and the loop brackets (grow hi until probe overshoots) then bisects tracking the
  closest-within-tol candidate from either side. decimate_to and ratedistortion each delegate.
  ACCEPTANCE, LOAD-BEARING: both call sites must be BIT-IDENTICAL after -- decimate_to's grid choice and
  ratedistortion's delta must not move by one step (these are shipped, tie-sensitive decisions; the
  constitution's "bit-identical to 1e-12 still flips a creature" applies directly). Pin the exact iter count
  and output of each before and after. This is why it is NOT a tail item: extracting the best-tracking /
  tolerance-exit logic from decimate_to's reporting while preserving every tie is delicate, and a rushed
  extraction that flips one decimation is worse than no promotion.
BOTH BIT-IDENTITY PINS CAPTURED (2026-07-17) -- the build is now pure mechanical extraction:
  * decimate_to(closed box 768 -> target 200, min_silhouette_iou=None) = 186 faces, 4 iters, budget_error 0.07
    (already verified stable through the M13 wiring, so decimate_to's bisection has not moved).
  * ratedistortion.geometry_preserving_code(default_rng(0).standard_normal((40,16)), target_cos=0.9999)
    -> delta = 0.04737815295834658, bits_per_vector 381.6 (the geometric-bisection consumer).
The primitive numerics.bisect_to_budget(probe, target, lo, hi, midpoint, max_iters, cmp): decimate_to uses
midpoint=arith (mid=(lo+hi)//2, integer) WITH best-tracking + tolerance exit; ratedistortion uses
midpoint=geom (mid=sqrt(lo*hi)) with a FIXED 28 iters and NEITHER. So best-tracking/tolerance must be OPTIONAL
-- the parameterise-the-real-difference lesson applies to the loop STRUCTURE, not only the midpoint. Reproduce
both pins EXACTLY or restore from zip.

SHIPPED: numerics.bisect_to_budget(probe, target, lo, hi, midpoint, max_iters, tol, cmp, key, bracket,
on_probe) built; decimate_to and ratedistortion BOTH delegate. BOTH PINS HOLD BIT-IDENTICAL: decimate_to
768->200 = 186 faces / 4 iters / err 0.07 (and stable across targets 100/300/500, fraction, and the guarded
path); ratedistortion delta = 0.04737815295834658 exactly. The dry-run's two predicted traps were both real
and both handled: (1) the iter counter is CALLER-owned via on_probe -- decimate_to increments report["iters"]
itself and skips the initial at(hi) exactly as before, so iters stayed 4 (a primitive counter gave 5);
(2) a `key` param was needed because decimate_to probes to a MESH not a number (found when best-track err tried
Mesh - int). Faculty m.bisect_to_budget wired; catalog 587 chars, 6/6 discoverable, analyze/measure. This is
the GENUINE promotion of the three audited: P6 homonym (don't fuse), M5 flood two-primitives (split), M6 one-
primitive-two-consumers (promote) -- and only M6 was a promotion at all.
DRY-RUN FINDING (2026-07-17, prototyped in isolation BEFORE any edit): a naive extraction reproduces
decimate_to's FACE result (186) but NOT its reported iters (prototype counted 5 vs the pinned 4), because
decimate_to increments report["iters"] at specific points and counts the initial probe(hi) differently than a
generic loop would. Likewise the geometric loop ran 29 vs 28 on a bracket off-by-one. LESSON FOR THE BUILD:
bisect_to_budget must NOT own the iteration counter -- the caller's report does, incremented where the caller
chooses. The primitive returns (result, knob) and optionally best-tracking; iter accounting stays with the
caller. This keeps decimate_to's report["iters"]=4 exact. The face result and delta are what the primitive
guarantees; the report bookkeeping stays caller-side. Found by prototype, not by a mid-splice regression.

### M7 -- Ledger P2: smallest eigenpair   [DONE 2026-07-17 -- promoted, crossfield delegates, phi bit-identical]
Generalize crossfield's `_sparse_smallest_eigvec` to numerics.smallest_eigenpair(matvec, n). Do NOT flatten
iterate's closed-form bind spectra into it. Dense-eigh sites stay dense where matrices are small (R1's routing
lesson: route, don't replace).
SHIPPED: numerics.smallest_eigenpair(matvec, n, c, seed, dtype, on_matvec) -- the two-phase shifted-inverse
solver extracted from crossfield (3.5k chars of iteration deduplicated); crossfield keeps ONLY the
Laplacian-specific parts (dual-edge matvec, Gershgorin c=2*max_deg) and delegates. PINNED BIT-IDENTICAL:
cross_field's phi sha-pinned on BOTH routes (dense 768f d2c81dd2847439ed, sparse 3072f cee8e1134fd1a71f),
plus crossfield's own dense/sparse parity selftests green. The primitive's isolated contract pinned too:
finds the TRUE smallest eigenpair of an arbitrary real-PSD matvec to 1e-6 with eigenvector alignment >0.999,
dtype=float and complex both work, and the matvec counter stays CALLER-side via on_matvec (the M6 lesson
applied preemptively). One pre-existing wiring-honesty test asserted the OLD delegation string
("numerics import cg" inside _sparse_smallest_eigvec); the promotion made the chain one level deeper
(crossfield -> smallest_eigenpair -> cg), so the test now asserts each link of the CURRENT chain. Dense-eigh
sites stay dense per the ledger note (route, don't replace). Faculty + catalog (572 chars, 6/6, analyze/
measure) + selftest + pytest pin.

### M8 -- Ledger P7: diagonal-evolution audit   [AUDITED 2026-07-17 -- verdict CONFIRMED: audit-only]
transfer/step_k/limit (iterate) and diffuse_periodic (solve_laplace) grew the same closed-form-evolution
pattern independently. AUDIT ONLY -- most are correct np.fft use, and P6 proved what assuming costs.
AUDIT RESULT (2026-07-17): 59 FFT sites (not 46), classified. 53 are PLAIN spectral use (correct as-is, a
promotion is irrelevant to them). Only ~6 share the diagonal-EVOLUTION move -- advance a linear system by
multiplying a per-frequency factor and inverting -- in TWO time-flavours of ONE pattern:
  * DISCRETE (transfer ** k, invert): iterate, transformbank, simreadout, dynamics -- the ledger's iterate cluster.
  * CONTINUOUS (exp(-t * k2), invert): fields (diffuse_periodic), reproject -- the heat/diffusion flavour.
So the ledger hypothesis is CONFIRMED: iterate and diffuse grew the same pattern independently, and it is
exactly these two flavours, not a scattered 46. backend's **k was a token co-occurrence (no FFT-with-power),
correctly excluded. VERDICT stands as AUDIT-ONLY: a promotion is POSSIBLE (numerics.diagonal_evolve(spectrum,
factor, k) with discrete/continuous factor) but the two flavours already read clearly in their own modules and
are each 2-4 lines; fusing 6 short correct call sites behind one primitive is a discoverability tax, not a win
(the inverse of M6, where the two consumers were long and genuinely shared machinery). Filed as a LATENT
promotion: revisit ONLY if a third distinct consumer appears or one flavour grows complex. Kept negative:
"6 sites match a pattern" is not "promote" -- promotion pays when the shared code is LONG and the call sites
FEW; here it is short and they are already clear. This is the mirror lesson to M6.

### M9 -- R7 skeletonizer + creature detection   [INC 1+2 DONE; inc2 branch-seg SHIPPED as m.mesh_parts/m.match_symmetric_parts (verified 2026-07-19); inc3 creature-ID still OPEN -- see OPEN_ITEMS.md]
Skeleton = RIDGE OF THE DISTANCE FIELD (the tree is saturated with SDF machinery). Creature detection =
CLEANUP AGAINST A CODEBOOK -- shape signature (EGI + skeleton spectrum + proportions) as a hypervector,
resonated against archetype prototypes; the winner names the rig template. find_capability FOR SHAPES.
Feeds M2. Composition + one ridge-tracer, not a research project.

INCREMENT 1 DONE (2026-07-17): mesh_skeleton -- the medial-axis ridge, in a new holographic_skeleton module. PREMISE TESTED FIRST (per discipline): the SDF ridge of a cylinder lands at radial distance 0.02 from the central axis and spans the height -- it IS the centerline. Only then built. And it is 'generalise on contact', not a new machine: distance-to-surface is M14's shared correspondence (closest_face_point), inside/outside is voxelize's generalised winding number; the skeleton is those two on a grid with a local-maximum filter. mesh_skeleton returns {points, depth=medial radius (local half-thickness), bounds}; interior_distance_field exposes the field itself for thickness/wall analysis. Validated on the real ladybird (46 ridge points, deepest medial point at the body core). Faculties m.mesh_skeleton / m.interior_distance_field; catalog 593, 6/6, analyze/measure. KEPT NEGATIVE (loud in the module + selftest): this is a VOXEL ridge, resolution-limited and not guaranteed connected -- the 1-D graph collapse (thin the ridge voxels to a connected curve) is increment 2.
REMAINING: increment 2 = ridge -> connected 1-D curve (graph thinning + junction detection); increment 3 = creature detection = a shape signature (EGI + skeleton spectrum + proportions) as a hypervector, resonated against archetype prototypes (find_capability FOR SHAPES, feeding M2's rig template). Increment 1 -- the field and its ridge -- is the foundation both build on, now pinned.

INCREMENT 2 DONE (2026-07-17): skeleton_curve -- the ridge collapsed to a single-branch centerline polyline. MEASURED FIRST: the ridge voxels are ONE connected component but a THICK cloud (degrees 7-11, not a thin degree-2 path), so a naive neighbour-walk fails. Tested the cheaper hypothesis -- order the ridge points along their PRINCIPAL AXIS (PCA) and average cross-section bins -- and it collapses a cylinder to a PERFECTLY straight line on its axis (radial 0.000, straightness residual 0.0000). Cheap, deterministic, NumPy-only, no thinning kernel. Faculty m.skeleton_curve; catalog 592, 6/6. KEPT NEGATIVE, load-bearing and measured: this is SINGLE-BRANCH. One global PCA axis cannot follow a bend -- on an L-shaped tube it cuts the corner (residual 0.478 from the single axis). So a bent or branched skeleton needs branch SEGMENTATION (junction detection) FIRST, then skeleton_curve per branch. This primitive IS exactly the per-branch collapse that segmentation will call.
REMAINING: increment 2-plus = branch segmentation (junction detection on the ridge graph) then per-branch skeleton_curve for a full multi-limb skeleton; increment 3 = creature detection = shape signature (EGI + skeleton spectrum + proportions) as a hypervector resonated against archetype prototypes, feeding M2's rig template -- the VSA-meets-geometry novel research. Both build on the ridge (inc 1) and the per-branch curve (inc 2), now pinned.

### M11 -- extract_quads holes  [RESOLVED 2026-07-17: extractor inherits openness, creates none; +crash fix]
MEASURED: surface_retopo turns a CLOSED 768-face box into a mesh with 6 boundary edges, is_closed False, at
density 1.0 -- while scoring 0.973 IoU, a clean silhouette PASS. CAUSE: extract_quads DROPS every source
triangle whose 3 corners land in fewer than 3 distinct lattice cells (the `distinct` filter, reported as
degenerate_cells: 192 of 768). I filed that count as a report FIELD and never asked what dropping the faces
does to the SURFACE. They are holes.
THE FIX is IFAM's own: a collapsed cell should MERGE its corners into ONE output vertex and let the incident
faces re-stitch around it, not vanish. That is the same coarse/fine move extraction already uses -- the cell
IS the anchor; a cell with <3 distinct corners is a cell the neighbours must close over.
DONE WHEN: surface_retopo on a closed source returns a closed result (boundary_edges 0), topology_delta
preserved, at unchanged silhouette. Pinned. Until then the defect is loud in extract_quads' docstring.

REMAINDER RESOLVED (2026-07-17) -- NOT a loop-walk. The ladybird's 167 output boundary edges: the SOURCE going into retopo already had 3268 (a decimated surface SCAN, massively open). The extractor takes 3268 -> 167 -- carries the scan's openness at coarser sampling, does not punch holes. DECISIVE TEST: close the scan first (mesh_repair fill_holes=True -> 0 boundary), retopo -> 0 output boundary. Extractor creates ~NO holes on closed input, scan or box; the remaining boundary was INHERITED. The loop-walk was chasing a phantom -- 'close the scan first' already exists (fill_holes). nonmanifold_edges at singular cells stay reported not forced.
BONUS CRASH FIX: face_frames divided a zero-area face's normal by zero length -> NaN -> position_field round() crash on any hole-filled mesh. Guarded (bare divide preserved, phi pins d2c81dd2/cee8e113 hold bit-identical; only zero-norm rows get a finite frame). Formerly-crashing retopo of a hole-filled scan now runs to 856 faces, 0 boundary. Pinned.

### M12 -- Displacement bake                    [DONE 2026-07-17 -- a parameter on the normal bake, with the cage]
AUDIT DONE, and it confirms the suspicion with the docstring's own words: bake_normal_map "Reuses transfer_uv's
closest-point projection (the high-poly lookup)". So the LOW->HIGH CORRESPONDENCE IS ALREADY A SHARED PRIMITIVE
(transfer_uv owns it; the normal bake is one consumer). A displacement bake is the SAME lookup recording a
different quantity: for each texel, signed distance from the low-poly point to its closest high-poly point,
measured ALONG THE LOW-POLY NORMAL -- i.e. (Q - P) . N, where P/N are what the baker already has per texel and
Q is what the projection already returns. Nothing new gets built; a third consumer gets added.
DESIGN NOTE, from the existing KEPT NEGATIVE which displacement INHERITS AND WORSENS: bake_normal_map has no
cage / ray-distance limit, so a texel far from the high-poly still grabs the nearest normal (a floating detail
bleeds). For NORMALS that is a shading artefact; for DISPLACEMENT it is a spike, because the bad value moves
GEOMETRY. So displacement should ship WITH the cage/max-distance the normal bake never got -- and that, not
the projection, is the real work in this item.
DONE WHEN: bake_displacement returns a height image whose max |value| is bounded by an explicit max_distance;
applying it to the low-poly reproduces the high-poly silhouette measurably better than the low-poly alone
(silhouette_sweep is the instrument, and it is the honest baseline).

SHIPPED: m.bake_normal_map(..., displacement=True, max_distance=D) bakes a signed height map from the SAME
closest-point cast as the normal -- one projection, two channels read out (M14's move in miniature, and the
owner's "add a dimension to one pass, project out what you need" applied literally). _high_normal_at now
returns (normal, hit_point) instead of discarding the point; displacement = (hit - texel_point) . low_normal,
clamped to max_distance. THE CAGE is the real work M12's audit identified, and it is here: a stray far hit
clamps rather than spiking, because displacement moves GEOMETRY (a normal-map artefact only shades wrong).
VERIFIED: the normal map is BIT-IDENTICAL with displacement on vs off (the shared projection did not perturb
the existing bake -- pinned), the cage clamps hard (tight cage 0.002 -> max |disp| 0.002), displacement is
signed and sane. Catalog 596, 6/6 phrasings, create/emit. This is also M14 increment 1 PROVEN: the
one-cast-many-channels principle works; the remaining M14 work is factoring the projection into a standalone
mesh_correspondence buffer so AO and attribute transfer join too, and refactoring the normal path to delegate
-- optimisation, now that the principle is demonstrated and pinned.

### M13 -- Wire the topology gate into the reducing faculties   [COMPLETE 2026-07-17 -- all six faculties]
UNBLOCKED by M4 and STARTED, deliberately on one faculty first.
THE DESIGN DECISION, made once and on purpose: the gate REPORTS by default (`topology=True`) and refuses only
on `topology="refuse"`. Enforcing by default would flip decisions that shipped -- and surface_retopo is
MEASURED to punch holes (M11), so a default refusal would break the owner's own working pipeline the day the
instrument landed. AN INSTRUMENT THAT STARTS REFUSING YESTERDAY'S WORK IS A DECISION CHANGE WEARING A
MEASUREMENT'S CLOTHES. Decisions change in one place, on purpose, never as a side effect of adding a gauge.
VERIFIED on surface_retopo: face count IDENTICAL (323) with the instrument on, off, and as recorded before it
existed; the report now carries BOTH truths at once -- silhouette 0.989 PASS, topology preserved False,
holes_created True. Pinned, including the opt-out and the refuse path.
REMAINING (each the same ~10-line pattern, one file at a time, NEVER as a sweep): mesh_cluster_decimate,
mesh_qem_decimate, mesh_decimate_to, voxel_remesh, quad_remesh. Note the first four MEASURE AS CLEAN today
(topology preserved on the box fixture, pinned in test_topology_gate_catches_what_the_silhouette_cannot), so
for them this is a regression trap, not a bug hunt. The bug is in the fifth and in surface_retopo -- M11.
COMPLETED: all five remaining faculties wired one at a time via ONE shared helper (_topology_check) so the
rule -- report by default, refuse opt-in, skip on False -- is typed once and cannot drift. Mesh-returners
(cluster/qem/voxel) ride .topology_report beside .silhouette_report; dict-returners (decimate_to, quad_remesh)
gain a "topology" key. VERIFIED: every result BIT-IDENTICAL with the instrument on vs off; decimate_to's
recorded 768->200 = 186 faces / 4 iters UNCHANGED (doubling as M6's first pin); refuse fires on a REAL
violation (SDF rebuild at res 10 FILLING an existing hole -> holes_filled=True, the owner's rule) and not on
clean ops. voxel_remesh note: an SDF rebuild legally changes topology (block-outs close holes on purpose), so
its default report is information and refuse is the caller declaring stricter intent than the documented
scope -- exactly what an opt-in is for.

### M14 -- ONE correspondence, many channels   [DONE 2026-07-17 -- shared machine; transfer + bakes delegate]

SHIPPED: build_face_grid(vertices, faces, cell_scale) + closest_face_point(p, grid, tri, lo, cell, faces) --
the shared correspondence machine. The uniform spatial hash + ring-expanding closest-point search that FOUR
sites had copied inline is now ONE pair of functions; transfer_uv and bake_normal_map's _high_normal_at both
DELEGATE. The M12 "one cast, many channels" principle is now the actual architecture: the primitive owns the
PROJECTION (face + barycentric + distance), each caller reads its own CHANNEL (transfer_uv interpolates UVs,
the bake reads normals + reconstructs the hit point for displacement). PINNED BIT-IDENTICAL: transfer_uv
out=6f296d80bb12e491 dist=326ba2806dec4bbf, bake_normal_map=740e16230c4eb938 -- both unchanged through the
refactor (the set()-iteration tie-break survived, first-seen-at-min-d2). Displacement (M12) still rides it with
normal-identity preserved. Faculty m.mesh_closest_point exposes the machine directly; catalog 589, 6/6,
analyze/measure. REMAINING (smaller now): the other two inline closest-point sites (transfer_attribute at
line ~718, and holographic_ai) can migrate opportunistically -- each pinned -- but the load-bearing two are
done and the principle is the architecture. This closes the owner's holographic reframe: correspondence is a
buffer computed once and projected many ways, not a per-bake re-cast.
MEASURED, not hypothesised. The low->high PROJECTION is the entire cost of a bake (~1.7-1.9s of a ~1.9s bake
at size 48 / 768-face high-poly; writing pixels is free by comparison). Asking for a second quantity today
re-runs the WHOLE projection on IDENTICAL hits: two normal bakes measured 1.91s then 1.74s, bit-identical
output. And the projection `_high_normal_at` already computes the closest point, barycentric weights, hit face,
and distance -- then DISCARDS everything except the normal. Normal, displacement, AO, and attribute transfer
are all just different READS at the one hit; the information is computed and thrown away.
THE MOVE (the owner's exact framing -- add channels to one pass, project out per operation):
  1. mesh_correspondence(low, low_uv, high, size, max_distance) -> a correspondence BUFFER (hit point,
     barycentric, hit face, signed distance along low normal, valid mask). ONE projection sweep, paid once.
     The cage/max_distance (M12's real work) lives HERE because EVERY channel needs it.
  2. cheap projectors reading a channel with NO new projection: bake_normal (interpolate high normal),
     bake_displacement (the stored signed distance -- M12 becomes a one-line field read), bake_ao (rays from
     stored hits), bake_transfer (any per-vertex attribute at bc).
  3. bake_maps(low, low_uv, high, channels=(...)) runs the correspondence ONCE, projects each channel. Three
     maps = ~1 projection + 3 cheap reads, not 3 projections.
WHY IT IS REUSE: the projection is transfer_uv's closest-point lookup, which bake_normal_map's docstring
claims to "reuse" but actually RE-IMPLEMENTS. mesh_correspondence is where that lookup lives ONCE, and
transfer_uv / bake_normal_map / displacement all become projectors over it. Same shape as the engine's
gather_field instinct (compile the work so one op serves many reads), transposed: there many points -> one
query; here one point -> many channels. THIS SUBSUMES M12 (displacement) and largely M11's cousin work.
DONE WHEN: bake_maps produces normal maps BIT-IDENTICAL to the standalone bake (the projection is
deterministic; only WHEN the work happens changes -- pin fused == standalone), and a 3-channel bake runs in
~1 projection's time, measured against the 3x baseline. Full build arc -- fresh context, not a tail.

### M15 -- Reference outline reused across guarded ops  [DONE 2026-07-17 -- the owner's "redoing a pass" catch]
The owner asked whether the silhouette pass redoes work. TWO findings:
 1. silhouette_mask is NOT the render pass redone -- it was explicitly built to AVOID that (no z-buffer, no
    lighting, no perspective; the docstring measures the full rasteriser at 0.5-1.5s/view and picks orthographic
    coverage instead). And WITHIN a walk, ref_cache already computes the source outline once. Both good.
 2. BUT across SEPARATE guarded ops on one source, the reference outline was re-masked from scratch each time:
    measured 7/7/7 masks for three ops on the same box, because ref_cache was a fresh {} per faculty call.
    Same waste shape as the bakes -- an invariant recomputed per operation.
FIX (additive, the reuse the owner meant): silhouette_guarded gained ref_cache=None. Default None keeps the
old per-call behaviour bit-identical; a caller running decimate+cluster+retopo on one source passes ONE dict
and pays the reference projection ONCE. MEASURED 21->7 reference masks across three ops, result BIT-IDENTICAL
(pinned: a cache that changes an answer is a bug, not an optimisation). The cache only ever grows with
REFERENCE views, never candidates, so sharing is always safe.
FOLLOW-UP (filed, not built): the natural next step is keying the reference masks on a content hash of
(source, size, views) via the existing "Dependency-keyed cache" faculty, so reuse is automatic instead of
caller-threaded -- but that touches cache lifetime/determinism and is a deliberate build, not a tail item.

### M16 -- Silhouette worst view   [DONE -- verified 2026-07-19: m.worst_view, Lipschitz/DIRECT-on-sphere branch-and-bound, the ranked "next attempt" below shipped in a branch]
The owner proposed side/top/front/perspective = 4, not 7, since front/back etc. are the same. MEASURED, and
the reality is sharper than either number:
 1. THE OPPOSITE-VIEW SYMMETRY IS ALREADY EXPLOITED. silhouette_sweep covers azimuths in [0, pi) ONLY --
    theta and theta+pi are mirror-identical under orthographic projection (on record, with the measured
    caveat that it is exact in the limit but not pixel-exact: the crab lost 15% IoU to independent
    quantisation of the two mirrors, which is why the sweep masks BOTH meshes under the SAME direction so
    truncation cancels). So "front/back are the same" is already banked; n_azimuth=6 is 6 DISTINCT directions,
    not 3 opposed pairs.
 2. A FIXED SMALL COUNT CAN MISS THE WORST VIEW. Measured on a decimated box: true worst azimuth is 135 deg;
    n_azimuth=4 (0/45/90/135) hits it DEAD ON (worst 0.9780, = dense-24 ground truth), while the current
    n_azimuth=6 (0/30/60/90/120/150) STRADDLES it and reports 0.9846 -- NON-CONSERVATIVE by +0.0066, i.e. it
    calls the mesh better than it is. On the ladybird, by contrast, n_azimuth=2 (3 views) already equals
    dense-24 exactly. THE RIGHT COUNT DEPENDS ON THE OBJECT'S SYMMETRY: a cube's worst view sits at 45 deg
    off-face; a fixed grid aligned to 30-deg steps never lands there.
WHY NOT JUST SET 4: it happens to suit the cube and would REGRESS objects whose worst view sits off the
4-grid; and lowering the default at all FLIPS shipped decisions (the reported worst changes). This is a
sampling-phase problem, not a count problem.
THE FIX IS HARDER THAN "COARSE + REFINE" -- PROTOTYPED AND MEASURED 2026-07-17, filed so the next attempt
does not repeat mine. Two prototypes:
  v1 (refine around the single coarse minimum): box 0.9782@135 correct, but LADYBIRD +0.0031 non-conservative
     -- dense found a worse view at 72 deg in a basin the single-minimum refine never visited.
  v2 (refine around EVERY coarse sample within 0.03 of the coarse min): ladybird fixed (-0.0001), but BOX g5
     +0.0012 non-conservative -- dense's worst at 8 deg is a NARROW trough BETWEEN coarse samples and below
     the refine band, so no seed reaches it. And v2 cost 84 masks, MORE than dense-90.
THE LESSON: a coarse ring can miss a narrow trough entirely, so no "coarse then refine" is unconditionally
conservative, and chasing conservativeness with more seeds erases the cost saving. The worst-view surface is
not unimodal. REAL OPTIONS for the next attempt, ranked:
  (a) LIPSCHITZ / branch-and-bound: bound how fast IoU can change per degree (it is bounded by the silhouette
      perimeter / frame) and only refine intervals whose bound COULD beat the current worst -- provably
      conservative, adapts sample count to the object. This is the honest fix; it needs the Lipschitz constant
      measured empirically first.
  (b) accept fixed sampling but pick the count from the object's rotational symmetry order (detect via the
      silhouette's autocorrelation over a cheap coarse ring), so a cube gets a 45-aligned grid.
  (c) leave the guard as-is (fixed) and only ADD adaptive as a REPORTING tool (mesh_worst_view) that callers
      invoke for a final honest number, never as the gate -- lowest risk, still useful.
DO NOT ship any of these as a DEFAULT until proven conservative on box(g4,g5)+ladybird+crab vs dense-2deg;
the prototypes show the failure is subtle and asymmetric. Opt-in reporting (option c) is the safe first ship.
DONE WHEN: the chosen method is <= dense on cost AND never above dense worst by >1e-3 on all fixtures.

### M10 -- Standing debts (small, real)
* `preview_asset` fit_camera adoption [DONE 2026-07-17, and a REAL bug found underneath]. Added fit=True
  (opt-in; default keeps the bbox-diagonal heuristic so no existing preview reframes). BUT the "4% of frame"
  premise was MY measurement error: it used direction [1.0,0.75,1.1] (long vector -> heuristic eye pushed
  far); with preview's actual eye_dir (0.55,0.35,0.7) the heuristic is only 1.14x too far -- a MODEST win, not
  4x. THE REAL PRIZE was underneath: fit_camera sized tx=ty*aspect but did NOT RETURN aspect, so the renderer
  used its default and the fit was silently wrong on every non-square frame -- the same two-paths-disagree-on-
  aspect bug that bit rasterize_mesh earlier this arc. FIXED at source: fit_camera now returns aspect; pinned
  that the subject fills the constraining axis on 4:1 / 4:3 / 1:4 frames. as_camera already allow-listed
  aspect, so no caller breaks.
* Rebake perf wall: 244.6s/3002 faces at 512^2 -- batch the candidate axis (~900k tiny NumPy calls).
* `mesh_orient` report: `components` alias deprecated in favour of `propagation_components`; drop next release.
* Poly Studio D-section (flat vs packaged import duality; optional-dep import tracing) -- re-audit before filing.

---

## 3. HOW WE DID -- honest scorecard of this arc's process

WENT WELL: every promotion kept callers bit-identical (measured, not asserted); every negative kept loud; the
structural trap caught surface_retopo with NO ONE adding it to a list.
WENT BADLY, on record so it stops repeating:
 1. I OOM-killed the box TWICE by walking a CUBIC knob at x1.5 -- and blamed the tool before reading its scope
    line, which said "BLOCK-OUT" all along.
 2. I scoped R3 at ~4x its true size (position_field/IFAM 4-PoSy and shrinkwrap already existed; the docstring
    even named the missing stage AND its paper section).
 3. I shipped a whole session of horizontally STRETCHED renders without noticing (the raster path ignored the
    frame aspect while the ray path did not -- the engine's two paths disagreed).
 4. R3c's hypothesis was BACKWARDS: decimation heals the scan, it does not shatter it.
 5. Three over-strict pins in one day (step_factor rounding; cg residual vs contract; phasor bit-equality).
    ASSERT THE CONTRACT, NOT THE WISH.
 6. I filed M3 -- a cross-source "promote, do not re-specialize" WIN -- on an unmeasured premise, and it was
    WRONG IN BOTH HALVES (the generator was already consistent; mesh_orient cannot fix global inversion). The
    narrative was so satisfying (one move, two domains, discovered twice) that I wrote it into the master
    backlog without running it. MEASUREMENT OVER NARRATIVE applies hardest to the findings that flatter you.
THE PATTERN: four items were resized by an audit BEFORE a line was written. **Audit before SCOPING now carries
more weight than audit before coding** -- an over-scoped plan wastes exactly what a duplicate wastes.
