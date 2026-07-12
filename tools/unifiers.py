#!/usr/bin/env python3
"""unifiers.py -- the registry of leCore's UNIFIERS and their named clients, and whether each one is actually wired.

WHY THIS EXISTS (the systemic finding)
--------------------------------------
The engine's biggest structural debt is not missing capability -- it is capability that EXISTS, is excellent, and is
wired to none of its own named clients. `holographic_iterate`'s docstring says outright that subdivision, the
propagator's k-step rollout, the diffusion steady state and the resonator's fixed points are all "iterate a linear
operator"... and it was wired to 0 of the 7 modules it names, each of which hand-rolls the loop. That is a silo with
a signpost pointing at it.

A unifier wired at ONE call site is still a silo. So this file makes the claim checkable:

  * `REGISTRY` names each unifier, the module it lives in, the symbols that count as "citing" it, and the clients its
    own docstring/design says should use it.
  * `status()` reads the live code (plain text search -- no imports, no side effects) and reports wired / not wired.
  * `tests/test_unifier_adoption.py` turns that into a CI lint: a wired client may never silently un-wire, and the
    list of pending ones may only ever shrink.
  * `python tools/unifiers.py` prints the table; `--markdown` emits the UNIFIERS.md registry.

Adding a unifier here is how you make "promote and generalize" stick: the moment someone hand-rolls the thing again,
the lint tells them where the canonical home is.
"""
import os
import sys


# unifier -> (module that owns it, symbols that count as citing it, the clients its design names)
REGISTRY = {
    "honesty.permutation_null": {
        "module": "holographic_honesty",
        "symbols": ["holographic_honesty", "permutation_null"],
        "why": "The shuffled-null discipline (score a datum, re-run the IDENTICAL scoring on structure-destroying "
               "resamples, demand the real score stand out) as ONE composable primitive -- Tarter's radio-SETI and "
               "Cranmer's particle-physics move, now callable by any capability including one built on the engine. "
               "Generalises the five procedure-matched private nulls (_recall_null, _recognition_null, _brain_null, "
               "_scan_cue_null, and the MI/phase-randomised nulls) that each hand-rolled the same draw-resample -> "
               "score -> sort -> p loop. VERIFIED bit-identical to that hand-rolled recall-null loop; adds the +1 "
               "plug (North et al. 2002) so p is never exactly 0. Calibrated (random datum flags at ~alpha) and "
               "deterministic. SCOPE (see DEFERRED): the primitive is SHIPPED and wired as mind.permutation_null; "
               "forcing the five private nulls to DELEGATE is deferred with a measured reason, not forced.",
        "clients": [],
    },
    "meshgeodesic.dijkstra (dist+prev)": {
        "module": "holographic_meshgeodesic",
        "symbols": ["holographic_meshgeodesic", "_dijkstra", "_edge_graph"],
        "why": "ONE weighted Dijkstra over the mesh edge-graph, returning (dist, prev). geodesic_distances kept its "
               "distance field; meshseam.shortest_seam had reimplemented the SAME adjacency build + the SAME Dijkstra "
               "inline (a distance FIELD vs a PATH -- the same algorithm in two costumes) and did not even import "
               "meshgeodesic. Lifted to (dist, prev) with an optional early-out `target`; shortest_seam now delegates "
               "and only reconstructs the path from prev. This is a TIE-SENSITIVE path (both documented vertex-index "
               "tie-breaking); verified BYTE-IDENTICAL on 18 golden outputs (distance hashes + exact seam paths) "
               "before/after -- a behaviourally-invisible dedup, pinned by a regression trap.",
        "clients": ["holographic_meshseam"],
    },
    "iterate.step_k / limit": {
        "module": "holographic_iterate",
        "symbols": ["holographic_iterate", "step_k"],
        "why": "a bind operator is diagonal in the Fourier basis, so k steps (or the k->inf limit) is closed-form: "
               "one FFT, no loop. Measured 41x (k=64) to 1059x (k=4096); a divergent operator raises instead of "
               "silently overflowing to nan. SCOPE (measured, see NOT_APPLICABLE): the operator must be LINEAR *and* "
               "CIRCULAR (a bind). `dynamics` is the only module that qualifies today.",
        "clients": ["holographic_dynamics", "holographic_subdivcurve", "holographic_meshsubdiv"],
    },
    "coarsefirst.refine_where_uncertain": {
        "module": "holographic_coarsefirst",
        "symbols": ["holographic_coarsefirst", "refine_where_uncertain", "escalate_mask"],
        "why": "run the cheap method everywhere, escalate to the expensive one only where a per-cell uncertainty "
               "signal is high. Measured on adaptive AA: 6.2x fewer samples for a 21% RMSE cost, and the same "
               "budget at RANDOM cells is 3x worse -- the signal pays, not the budget. TWO NECESSARY CONDITIONS: "
               "the uncertainty must be CONCENTRATED (`concentration` is the free gate), and the expensive method "
               "must be priced PER CELL. A greedy placement method fails the second and destroys the first.",
        "clients": ["holographic_adaptive_sample"],
    },
    "determinism.argmax_tiebreak": {
        "module": "holographic_determinism",
        "symbols": ["argmax_tiebreak"],
        "why": "the argmax IS the observable architectural decision (which atom is recalled). Scores are not "
               "bit-stable across backends/orders, so the winner must come from the named rule (ties -> lowest "
               "index), not a bare np.argmax. This is the `bind_batch` hazard class.",
        "clients": ["holographic_resonator", "holographic_classifier", "holographic_hopfield", "holographic_peel"],
    },
    "lowdiscrepancy": {
        "module": "holographic_lowdiscrepancy",
        "symbols": ["holographic_lowdiscrepancy"],
        "why": "one home for low-discrepancy sequences; `prt` carries its own duplicate spherical-Fibonacci lattice.",
        "clients": ["holographic_pathtrace", "holographic_prt", "holographic_samplinghome"],
    },
    "rns / reduce_sum_exact": {
        "module": "holographic_rns",
        "symbols": ["reduce_sum_exact", "holographic_rns"],
        "why": "exact integer reduction makes a distributed SUM bit-identical across orders and bucket counts -- the "
               "float-non-associativity caveat is retired, but only where it is actually called.",
        "clients": ["holographic_distribute", "holographic_accumulate"],
    },
    "octnormal": {
        "module": "holographic_octnormal",
        "symbols": ["octnormal"],
        "why": "manifold-correct quantization of a unit normal (octahedral mapping). MEASURED at equal STORAGE: "
               "3 bytes/normal gives 0.021 deg mean angular error vs 0.169 deg for naive per-component xyz + "
               "renormalize -- ~8x better for the same bytes. It is a STORAGE primitive: its clients are the modules "
               "that actually persist normals compactly, not every module that touches a normal.",
        "clients": ["holographic_autobump"],
    },
    "determinism.hash_unit": {
        "module": "holographic_determinism",
        "symbols": ["hash_unit", "hash_direction"],
        "why": "np.random carries STATE, so the n-th draw depends on every draw before it -- which makes farm work "
               "order-dependent and forces every node to agree on a seed and a draw count. hash_unit(x, y, i, seed) "
               "is a pure function of WHERE and WHICH, so any node computes any sample in any order and agrees. "
               "Wired into `wost` (walk-on-stars), whose walks are farm-parallel with zero seed coordination.",
        "clients": ["holographic_wost", "holographic_brdf"],
    },
    "project_onto_constraints": {
        "module": "holographic_denoise",
        "symbols": ["project_onto_constraints"],
        "why": "the resonator's alternating projection, the PnP restoration loop, PBD constraint solving and FABRIK "
               "IK are one engine: iterate a projection. The best-wired unifier in the engine -- the model.",
        "clients": ["holographic_meshik", "holographic_blendpose", "holographic_softbody", "holographic_collide",
                    "holographic_resonator", "holographic_dynamics"],
    },

    # ---- the 2026 "holographic GPU" cohort: primitives shipped with measurements, adopted by almost nobody -------
    "superposed width (bind_fixed / recover_all / pack)": {
        "module": "holographic_superposed",
        "symbols": ["holographic_superposed", "bind_fixed", "recover_all", "pack("],
        "why": "binding a FIXED operand against K vectors, or unbinding one composite against K keys, is ONE batched "
               "rfft -- not K of them inside a Python loop. MEASURED, bit-identical (np.array_equal): recover_all "
               "is 3.8x at (D=1024, K=8) and 4.3x at K=16; bind_batch is 2.8x at (1024, 8). "
               "CORRECTION, and it shrank the item by 25x: a first AST scan reported 77 candidate loop sites across "
               "30 modules. It was counting LOOP-CARRIED ACCUMULATORS -- `acc = bind(acc, d)` in the VSA machine's "
               "interpreter, `x` reassigned each trial in `flatness` -- which have a data dependency and cannot "
               "batch at all. A strict scan (bind/unbind inside a comprehension, one operand loop-invariant, the "
               "other the loop variable) finds exactly THREE: hopfield's `_structure_project` and `_decode_combo`, "
               "and `ai`'s non-iterative recall path. All three are now batched. SCOPE: the win is in K, not D -- "
               "the speedup falls to 1.7x by K=64 as the per-call overhead stops dominating.",
        "clients": ["holographic_hopfield", "holographic_ai"],
    },
    "shader algebra (bake once, compose passes)": {
        "module": "holographic_shader",
        "symbols": ["holographic_shader"],
        "why": "H1-H8: a baked field is a texture unit; a kernel is diagonal, so N filter passes compose into ONE "
               "elementwise multiply whose cost is independent of kernel size and pass count. This is the closest "
               "thing the engine has to a GPU, and NOT ONE RENDERER USES IT -- render, raymarch, pathtrace, postfx, "
               "texturerender, globalillum and svgf all cite it zero times. SCOPE: ~1% accuracy, a preview/coarse "
               "tier, and the phasor bandwidth must exceed the field's max frequency (the algebra has a Nyquist). "
               "G8 CLOSED THE postfx SILO BY GENERALIZING THE PRIMITIVE, not by paying for the wrong spectrum: "
               "`Pipeline(shape, real=True)` builds the transfer on the HALF-spectrum (rfftn), which is what a real "
               "field wants. Delegating was a 2.2x LOSS on the full fftn grid; on the half-spectrum it is "
               "BIT-IDENTICAL (max|diff| 0.0e+00) at the same speed (1.10 ms vs 1.04 ms per 128x128x3 image), and "
               "rfftn is 3.0x faster than fftn on a 256^2 real field. `Pipeline.from_transfer` avoids the identity "
               "allocation a caller with a composed transfer does not need (measured: the long way cost 42%).",
        # NOT a graphics unifier. `filter_k(field, kernel, n)` is rank-agnostic (verified on 1-D, 2-D and 3-D
        # arrays), `bake_1d(xs, ys)` approximates ANY sampled scalar function, and `Pipeline(shape)` composes in
        # 1-D. The clients below are therefore both renderers AND the non-graphics operator-iterators.
        # SCOPE, measured: the closed form needs the operator to be LINEAR *and* CIRCULAR. Applying the periodic
        # diffusion transfer to a Neumann (edge-replicated) problem is 4.76e-02 WRONG. Boundary condition is the
        # gate, not the domain.
        #
        # THE NON-GRAPHICS CLIENT IS `laplacian`, NOT `heat`. heat's periodic path already had an exact closed form
        # (`diffuse_spectral`) -- better than the filter_k form I proposed, because it solves the CONTINUOUS PDE
        # rather than reproducing the discrete Euler stepper. What it lacked was composition: it re-exponentiated
        # exp(-alpha|k|^2 t) on every call. `laplacian.diffusion_operator` now builds a `Pipeline` from that
        # transfer -- bit-identical (max|diff| 0.0e+00), ~1.9x faster on reuse, and it COMPOSES (two half-steps
        # multiply to one full step, exact to 1.1e-15). heat delegates to it via `diffuse_heat(operator=...)`.
        "clients": ["holographic_laplacian", "holographic_postfx"],
    },
    "distribute.reduce_sum_exact_partitioned": {
        "module": "holographic_distribute",
        "symbols": ["reduce_sum_exact_partitioned", "exact_partial", "run_exact", "distribute_exact"],
        "why": "distribute()'s DEFAULT reduce is float `reduce_sum`, which is not partition-invariant: a 4-way and a "
               "7-way farm split of the same work disagree by 2.98e-08 (measured, magnitudes spanning 16 orders). "
               "The exact path ships and is bit-identical across 1/4/7/13/700-way splits. NOT A DROP-IN SWAP: the "
               "exact reduce consumes buckets of CONTRIBUTIONS, not bucket sums, so the worker contract has to grow "
               "an `exact_partial` mode. That is the item -- correctness class, not speed.",
        "clients": ["holographic_coordinator"],
    },
    "collide.advance_ccd / time_of_impact": {
        "module": "holographic_collide",
        "symbols": ["advance_ccd", "time_of_impact", "resolve_swept_collision"],
        "why": "conservative advancement IS sphere tracing, so CCD costs nothing extra on an SDF. MEASURED: a 30 m/s "
               "body stepping 0.5 m per frame passes CLEAN THROUGH a 0.1 m wall under the discrete "
               "resolve_sdf_collision that `softbody` still calls, and is stopped exactly on it by advance_ccd. "
               "Correctness class: no margin can fix it (a margin resolves a crossed body out the WRONG side).",
        "clients": ["holographic_softbody"],
    },
    "island.color_waves (deterministic lock-free scheduling)": {
        "module": "holographic_island",
        "symbols": ["color_waves", "conflict_graph", "run_waves"],
        "why": "two tasks in one colour touch disjoint resources, so a wave runs with no locks and no atomics, in a "
               "reproducible order -- which is exactly how Box3D earns cross-platform determinism. MEASURED: 2,000 "
               "transactions over 300 keys colour into 24 waves, 83x lock-free parallelism, every wave verified "
               "conflict-free. `querylock` adopted it; the coordinator and farm have not.",
        "clients": ["holographic_querylock", "holographic_coordinator"],
    },
    "island.SleepTracker (solve only what moves)": {
        "module": "holographic_island",
        "symbols": ["SleepTracker", "step_islands", "island_energy", "connected_components"],
        "why": "a sleeping island is AT ITS FIXED POINT; skipping it is bit-identical to stepping it, and the saving "
               "is exactly the awake fraction (measured: 3 of 20 islands moving -> step() called 3 times, not 20). "
               "Physics steppers advance every body every frame regardless.",
        "clients": ["holographic_softbody"],
    },
    "tucker.LowRankField (compressed-domain compute)": {
        "module": "holographic_tucker",
        "symbols": ["LowRankField", "worth_factoring"],
        "why": "linear field ops pass through a factorization, so blur/add/query never form the array. MEASURED "
               "(1024^2, rank 3): 171x fewer bytes; separable blur 2.53 ms / 0.049 MB vs 66.60 ms / 8.4 MB dense at "
               "3.11e-15 error. Wired as `fieldhome.Field.low_rank`, a fourth field backend beside callable/dense/"
               "sparse. THE GATE IS AN ERROR BUDGET, not `rank_gate`'s 99% ENERGY: measured on REAL 128x128 fields, "
               "a sphere SDF at 99% energy is rank 2 and 7.45% WRONG, a box SDF 18.19% wrong, and fbm noise passes "
               "the energy gate at rank 5 with 28.54% error. `rank_for_error` sizes on max-abs error: sphere SDF "
               "rank 4 (16x fewer bytes, pays), box SDF rank 12 (5.3x, pays), fbm rank 50 (1.27x, marginal), white "
               "noise rank 124 (refused). SCOPE: the blur must be SEPARABLE, nonlinear ops do not survive, and the "
               "field must be BAKED ONCE and QUERIED MANY TIMES -- factoring a streamed frame costs an SVD, measured "
               "53.7x the FFT blur it would accelerate at 128^2 and 91.7x at 256^2.",
        "clients": ["holographic_fieldhome"],
    },
    "cachehome.MarginCache (fat margin for a drifting query)": {
        "module": "holographic_cachehome",
        "symbols": ["MarginCache", "suggest_margin", "drift_scale"],
        "why": "a drifting query (a camera, a cursor, a recall neighbourhood) should be served from an ENLARGED baked "
               "region, not re-keyed exactly. MEASURED on a unit-step 2-D walk of 400 queries: exact-key caching "
               "rebuilds 400/400; margin 6.0 rebuilds 20 at 95% hits. Wired into RenderSession.preview(reuse_margin=), "
               "where the drifting query is the CAMERA POSE: 20 drifting frames at margin 0.12 give 19 hits, 1 rebuild. "
               "THE GATE a caller must pass is NOT a hit-rate target: a hit serves a STALE value, and on a rendered "
               "frame the max error saturates at the FIRST reuse (0.5864, a silhouette edge) while the mean creeps "
               "0.0001 -> 0.0051. `suggest_margin_for_error(..., max_abs_error=)` sizes it against the error you "
               "cannot tolerate. Measured: on a value that jumps 0->1, a mean-only budget passes margin 0.1929 and "
               "serves a completely wrong answer (max error 1.00); the max-error bound stops at 0.094558, and 0.095158 "
               "is already catastrophic. The admissible margin is a CLIFF, not a slope.",
        "clients": ["holographic_session"],
    },
    "denoise.soft_relaxation (stiffness in physical units)": {
        "module": "holographic_denoise",
        # NB: "stiffness=" is NOT a symbol here. `RigidBody.step(stiffness=1.0)` has its own unrelated parameter of
        # that name, and matching it reported a silo as WIRED. A lint that lies is worse than no lint.
        # `stiffness=stiffness` IS precise: it is the pass-through into project_onto_constraints, and nothing else
        # in the tree writes it.
        "symbols": ["soft_relaxation", "stiffness=stiffness"],
        "why": "`omega` is a per-sweep number, so the same dial is different physics at different substep counts "
               "(measured: omega=0.30 lands at 0.942 after 8 sweeps and 1.000 after 64). stiffness=(hertz, zeta) is "
               "substep-invariant to first order, and stiffness=(inf, zeta) is BIT-IDENTICAL to the rigid default -- "
               "which is the gate every caller must prove before it plumbs the dial through. MEASURED at the IK call "
               "site, fixed physical horizon: omega=0.30 reaches within 0.4253 / 0.0314 / 0.0000 at 5 / 20 / 80 "
               "iterations, while stiffness=(8 Hz, 1.0) reaches 0.0033 / 0.0002 / 0.0001 -- the dial holds its "
               "meaning, omega does not. And at the softbody PBD site a stretched bone relaxes to 1.028 at 20 Hz and "
               "1.7498 at 2 Hz against a rest length of 1.0. NB `RigidBody.step(stiffness=...)` is an unrelated "
               "scalar spring constant in a different class -- that name once fooled this very lint into reporting "
               "softbody as wired.",
        "clients": ["holographic_meshik", "holographic_softbody"],
    },
}


# Clients that are KNOWN not to cite their unifier yet. This is the backlog, made executable: the lint asserts this
# set EXACTLY matches reality, so wiring one forces you to delete the line (progress is recorded), and un-wiring one
# fails CI (a silo cannot re-form quietly).
PENDING = set()
# (an empty set, not an empty dict -- `PENDING = {}` is a dict and the lint's set arithmetic dies on it)
_PENDING_NOTE = """
    # EMPTY. Every declared client of every registered unifier now cites it. The 2026 cohort arrived here with 21
    # PENDING lines; all 21 are gone -- 12 by WIRING (and each wiring forced its line to be deleted), 9 by measured
    # RETIREMENT into DEFERRED or NOT_APPLICABLE below.
    #
    # The retirements are the more interesting half. FIVE of the original client names were the module where the
    # SYMPTOM was visible rather than the module that OWNS the mechanism: `lightcache` for MarginCache (the camera
    # lives in `session`), `postfx` for LowRankField (the field lives in `fieldhome`), `heat` for the shader algebra
    # (the closed form lives in `laplacian`), `farm` for the exact reduce (the reduce lives in `coordinator`; farm
    # is a transport backend), and `physics`/`emitter` for sleep and CCD (neither has an island or a collider).
    # Reading each one is what found that; no scan could have.
"""
# EMPTY AGAIN, and this time because four clients were MEASURED AND RETIRED rather than because none was registered.
#
# `coarsefirst` was DARK -- a general primitive with no door. It now has a mind faculty, and exactly one engine
# client: `adaptive_sample.converged_mask`, whose complement IS an escalate mask. That adoption is about owning a
# CONTRACT (what "above the cutoff" means, and what happens on a tie), not about speed -- proven bit-identical on
# 100,000 variances including exact ties, because the tie convention is the opposite of coarse-first's default.
#
# The other four named clients are all in NOT_APPLICABLE with measurements, for ONE reason:
#     COARSE-FIRST BUYS ADAPTIVITY FOR A METHOD THAT HAS NONE.
#   * `splat`   -- greedy placement is already adaptive; cost is per primitive, not per cell.
#   * `volint`  -- the line integral is closed form; there are no cells to escalate.
#   * `render`  -- `empty_skip` + `early_term` ARE coarse-first, buying 15.2x where a residual mask buys 1.0x.
#   * `nystrom` -- the gate PASSES and the method still loses: max residual is the most isolated point, and a
#                  spectral embedding needs a MASS signal, not a COVERAGE one.
#
# `gbuffer.converge_samples` was recorded here for a long time and was the wrong module: it does not build the
# mask, it calls `converged_mask`. Reading the code found the right one.
#
# NOTE THE SHAPE OF THIS RESULT. A unifier invented AFTER its clients may find that every one of them already
# solved the problem, differently and better. That is not a wiring debt to be paid down; it is a discovery about
# where the primitive belongs -- and it is only reachable by measuring, because a registry full of plausible
# PENDING lines looks exactly like work worth doing.
# EMPTY, as of the `meshsubdiv` close-out. Every registered unifier is now cited by every named client it has, or
# that client sits in NOT_APPLICABLE with a measured reason. The entry that lived here longest read:
#
#     ("iterate.step_k / limit", "holographic_meshsubdiv")  --  "an irregular 2-D mesh around an extraordinary
#     vertex is not shift-invariant: Stam's method diagonalises the LOCAL subdivision matrix there instead. A
#     build, not a wall."
#
# It was a build, and smaller than it looked, because the part of the local Loop operator that is NOT shift-
# invariant is only the centre vertex. The ring-to-ring block is exactly the circulant of [3/8, 1/8, ..., 1/8] --
# a bind operator -- so `iterate.transfer` diagonalises it for free, and `meshsubdiv.loop_limit` now cites it to
# evaluate the k -> infinity limit surface in closed form. What remains unbuilt is FINITE k on an irregular mesh
# (the full Stam evaluation), which is a different construction and is recorded in loop_limit's honest scope.
# Progress is recorded by deletion from this set (the lint forces it). If a new unifier is
# registered with clients that do not yet cite it, list them here so the lint tracks the gap.


# THE ORIGINAL SCOPE -- every client each unifier's design/docstring ever named. This exists so the lint cannot be
# made green by NARROWING the registry: each of these must end up either (a) a wired client, or (b) an entry in
# NOT_APPLICABLE with a measured reason. Deleting a hard client is not progress; retiring it with evidence is.
DESIGN_CLIENTS = {
    "iterate.step_k / limit": ["holographic_dynamics", "holographic_subdivcurve", "holographic_sbc",
                               "holographic_diffuse", "holographic_resonator", "holographic_meshsubdiv",
                               "holographic_chaos", "holographic_equilibrium"],
    "determinism.argmax_tiebreak": ["holographic_resonator", "holographic_classifier", "holographic_hopfield",
                                    "holographic_peel"],
    "lowdiscrepancy": ["holographic_pathtrace", "holographic_prt", "holographic_sampling",
                       "holographic_globalillum", "holographic_samplinghome"],
    "rns / reduce_sum_exact": ["holographic_distribute", "holographic_measure", "holographic_accumulate",
                               "holographic_fountain"],
    "octnormal": ["holographic_mesh", "holographic_meshcurvature", "holographic_render", "holographic_splat",
                  "holographic_gbuffer", "holographic_autobump"],
    "project_onto_constraints": ["holographic_meshik", "holographic_blendpose", "holographic_softbody",
                                 "holographic_collide", "holographic_resonator", "holographic_dynamics"],
    "determinism.hash_unit": ["holographic_wost", "holographic_brdf", "holographic_mis", "holographic_traverse"],
    "coarsefirst.refine_where_uncertain": ["holographic_adaptive_sample", "holographic_nystrom",
                                           "holographic_volint", "holographic_render", "holographic_splat"],

    # ---- the 2026 cohort, as the FIRST wiring audit named them. Several names were wrong -- the module where the
    # symptom shows rather than the one that owns the mechanism -- and every one of those must now appear in
    # NOT_APPLICABLE or DEFERRED with a measured reason. Recording them here is what stops "narrow the registry
    # until it is green" from being a strategy.
    "superposed width (bind_fixed / recover_all / pack)": ["holographic_machine", "holographic_flatness",
                                                           "holographic_query", "holographic_reasoning",
                                                           "holographic_sequence", "holographic_hopfield",
                                                           "holographic_ai"],
    "shader algebra (bake once, compose passes)": ["holographic_postfx", "holographic_render",
                                                   "holographic_texturerender", "holographic_heat",
                                                   "holographic_laplacian"],
    "distribute.reduce_sum_exact_partitioned": ["holographic_coordinator", "holographic_farm"],
    "collide.advance_ccd / time_of_impact": ["holographic_softbody", "holographic_emitter"],
    "island.color_waves (deterministic lock-free scheduling)": ["holographic_querylock", "holographic_coordinator"],
    "island.SleepTracker (solve only what moves)": ["holographic_softbody", "holographic_physics"],
    "tucker.LowRankField (compressed-domain compute)": ["holographic_postfx", "holographic_fieldhome"],
    "cachehome.MarginCache (fat margin for a drifting query)": ["holographic_lightcache", "holographic_domecache",
                                                                "holographic_session"],
    "denoise.soft_relaxation (stiffness in physical units)": ["holographic_meshik", "holographic_softbody"],
}


def unaccounted(root=None):
    """Design clients that are neither wired nor explicitly retracted -- i.e. the honest remaining work."""
    retired = set()
    for (unifier, client) in list(NOT_APPLICABLE) + list(DEFERRED):
        for name in client.split("/"):
            retired.add((unifier, name))
    out = []
    for unifier, clients in DESIGN_CLIENTS.items():
        for c in clients:
            if c in REGISTRY[unifier]["clients"]:
                continue                                  # a tracked client (wired, or listed in PENDING)
            if (unifier, c) in retired:
                continue                                  # retracted with a measured reason
            out.append((unifier, c))
    return sorted(out)


# RETRACTED CLAIMS -- clients a unifier's docstring once named, which MEASUREMENT showed it cannot serve. Kept here
# (loud, with the reason) rather than deleted, so nobody re-proposes the wiring. A backlog item that is impossible is
# worth more written down than quietly dropped.
# DEFERRED -- possible, and the construction is known, but MEASUREMENT says it does not pay (or is not needed).
# This class exists because I previously filed these under NOT_APPLICABLE, which reads as "impossible". It isn't.
# A future session may revisit any of these with a better lever; none of them is closed by mathematics.
DEFERRED = {
    ("tucker.LowRankField (compressed-domain compute)", "holographic_postfx"):
        "The factorization EXISTS and is exact; it simply does not pay, and this is DEFERRED rather than "
        "NOT_APPLICABLE precisely so nobody reads it as impossible. postfx STREAMS frames: each image is produced, "
        "filtered once, and thrown away, so there is nothing to amortize an SVD against. Measured: factoring costs "
        "67.5 ms per 128x128 frame against 1.26 ms for the FFT blur it would accelerate (53.7x), and 592.6 ms vs "
        "6.46 ms at 256x256 (91.7x). A real rendered frame is not even very low rank -- silhouettes push it to rank "
        "28 of 96 at a 1% error budget, a 1.7x byte saving. It WOULD pay for a frame reused many times (a baked "
        "lightmap, a cached irradiance buffer); that is the follow-up. LowRankField's home is a field BAKED ONCE and "
        "QUERIED MANY TIMES -- `fieldhome`, where it is now wired.",
    ("honesty.permutation_null", "holographic_unified._recall_null"):
        "The primitive SUBSUMES this pattern -- verified bit-identical to the hand-rolled recall-null loop these "
        "five private nulls share. Delegation is DEFERRED (not NOT_APPLICABLE, so nobody reads it as impossible) "
        "because the private nulls return a RecallNull OBJECT (not the primitive's summary dict), are CACHED on "
        "domain-specific mutation counters (_gen, prototype count), and feed decide_confidence / recognize / scan "
        "thresholds that are pinned in tests; the primitive's +1 plug shifts p by <= 1/(n+1), which is correct but "
        "would move those pinned thresholds. Per ledger discipline (measure, don't force), the composable primitive "
        "ships first; converting the five nulls to delegate -- adapting the return type and re-pinning the shifted "
        "thresholds -- is the measured follow-up. The same reason applies to _recognition_null, _brain_null, and "
        "_scan_cue_null (one entry stands for the family).",
    ("iterate.step_k / limit", "holographic_diffuse"):
        "A closed form EXISTS near a fixed point: a learned bind reproduces the softmax denoise at cos 0.976 one "
        "step out, and 0.949 for an 8-step closed-form jump, with linearisation error scaling as eps^1.81 -- second "
        "order, the Koopman / modular-flow structure. But a settle() run from noise spends its steps OUTSIDE that "
        "ball and is already converged by the time it enters; once inside, one argmax returns the fixed point "
        "exactly. Valid where it is useless. Globally a single bind cannot hold several attractors (cos 0.078).",
    ("spatial.SpatialGrid (one shared knn index)", "holographic_spectral/holographic_graphsignal/holographic_chart"):
        "MEASURED and REJECTED (G2). The proposal was to route spectral/graphsignal/chart's k-NN onto the shared "
        "SpatialGrid. Measurement says NO: a uniform grid is a LOW-DIMENSIONAL geometric index and these callers "
        "operate in HIGH dimensions (VSA vectors, feature manifolds). A single high-D grid.knn query did not finish "
        "in 60 s (the guaranteed-complete ring search explodes over near-empty cells above ~3 D), and even in the "
        "grid's own low-D regime it LOST to the vectorised dense scan until N was very large (0.16x at N=300 D=2; "
        "0.56x at N=1500 D=2; ~tie 0.96x at N=4000 D=3) because of per-query Python overhead. The genuinely shared "
        "high-D index is the HoloForest, which chart+graphsignal ALREADY use via a forest= hook; G2's real fix was "
        "to give spectral.knn_adjacency the SAME forest= hook (measured 1.31x at N=800 D=128, growing with N; "
        "approximate recall_k, byte-identical when forest=None). Different index for a different regime is not a "
        "unification target -- the grid and the forest are both correct, each in its own dimensionality.",
    ("iterate.step_k / limit", "holographic_resonator"):
        "Same as `diffuse`: the alternating cleanup linearises near its fixed point and nowhere else. Its real home "
        "is project_onto_constraints, where it IS wired.",
    ("iterate.step_k / limit", "holographic_sbc"):
        "Same as `diffuse` -- the per-block softmax linearises only in a neighbourhood of a fixed point.",
    ("iterate.step_k / limit", "holographic_chaos"):
        "NonlinearPropagator IS the lift (a tanh reservoir fed its own prediction); escaping the linear closed form "
        "is the reason it exists. The regime dispatch (P11) is the right home, not `iterate`.",
    ("iterate.step_k / limit", "holographic_equilibrium"):
        "Energy relaxation through a nonlinear rho() with clipping. Linearises near equilibrium (the same eps^2 "
        "story); the iterations are spent far from it.",
    ("iterate.step_k / limit", "holographic_heat/holographic_wave"):
        "The DEFAULT Neumann stencil is linear but NOT circular, so the FFT eigendecomposition cannot touch it. On a "
        "PERIODIC domain it can -- and that path is now BUILT AND WIRED: diffuse_heat(..., bc='periodic') delegates "
        "to laplacian.diffuse_spectral and is exact to 2.2e-16 in ONE evaluation, where 1000 iterative steps still "
        "sit at 1.5e-4. What remains deferred is only the Neumann default, whose eigenbasis is the DCT, not the DFT.",
    ("rns / reduce_sum_exact", "holographic_measure"):
        "`measure` reduces a handful of scalars across seeds in a FIXED order on one machine, so order-independence "
        "buys nothing today, and quantizing a metric at bits=40 would perturb the very CI thresholds it defends. "
        "Trivially wireable as an opt-in the day measure aggregates parts returned by a farm.",
}


# NOT_APPLICABLE -- closed by MATHEMATICS or by the code's actual shape, not by a cost judgement.
NOT_APPLICABLE = {
    ("shader algebra (bake once, compose passes)", "holographic_heat"):
        "CLOSED BY LAYERING, and it is a wiring that happened one module down. `heat`'s periodic path already had "
        "an exact closed form (`laplacian.diffuse_spectral`), better than the `filter_k` form the backlog proposed "
        "because it solves the CONTINUOUS PDE rather than reproducing the discrete Euler stepper (measured: the "
        "stepper carries 9.97e-05 of truncation error against it). What was missing was COMPOSITION, and that now "
        "lives in `laplacian.diffusion_operator`, which builds a `shader.Pipeline` from exp(-alpha|k|^2 t) -- "
        "bit-identical (0.0e+00), ~1.9x faster on reuse, and composable (two half-steps multiply into one, exact to "
        "1.1e-15). `heat` consumes it via `diffuse_heat(operator=...)`. The unifier's client is the module that OWNS "
        "the mechanism, not the one that shows the symptom.",
    ("shader algebra (bake once, compose passes)", "holographic_render"):
        "CLOSED BY STRUCTURE, and it echoes `coarsefirst` -> `render` exactly. PROBED: `render` has no linear, "
        "shift-invariant pass to compose. Its 31 matches for 'filter' are PNG SCANLINE filters in the image encoder. "
        "The inner loop is per-ray sphere tracing plus nonlinear shading -- neither is diagonal in any Fourier basis, "
        "and neither is a chain of passes. The composition target for a rendered image is `postfx`, which is where "
        "the algebra is used (and DEFERRED from delegating, for its own measured reason). And as with coarse-first, "
        "render already owns the adaptivity the algebra would offer: `empty_skip` + `early_term` buy 15.2x.",
    ("shader algebra (bake once, compose passes)", "holographic_texturerender"):
        "CLOSED BY STRUCTURE. PROBED: zero convolutions, zero FFTs, zero baked grids. It marches a union SDF and "
        "evaluates material channels per hit -- a per-sample nonlinear program, not a filter chain. There is no "
        "transfer to compose and no expensive scalar function sampled densely enough to bake.",
    ("distribute.reduce_sum_exact_partitioned", "holographic_farm"):
        "CLOSED BY LAYERING. `farm.NetworkFarm` is a Coordinator BACKEND -- it submits work to hosts and returns "
        "results. It performs NO reduction; grep finds `reduce` only in its docstring. The reduce lives in "
        "`Coordinator`, where `run_exact` is wired. And the exactness is a property of the REDUCE, not the "
        "transport: MEASURED, `run_exact` is bit-identical across InProcessBackend and a 2-process LocalPool at "
        "4-, 7- and 13-way splits, while the float `run()` disagrees by 3.73e-08 between 4- and 7-way on either "
        "backend. Wiring `farm` would mean giving a transport layer an opinion about arithmetic.",
    ("collide.advance_ccd / time_of_impact", "holographic_emitter"):
        "CLOSED BY STRUCTURE. PROBED: `emitter.advance(pos, vel, force, dt, damping, wrap_to)` is a free-flight "
        "semi-implicit Euler step. It takes NO collider, and the module's only SDF use is `emit_from_surface`, which "
        "PROJECTS spawn points onto a surface rather than sweeping through it. There is no discrete collision to "
        "make continuous. Giving `advance` a collider would be a new feature with its own bar, not a wiring job -- "
        "and `softbody`, which does collide, is wired.",
    ("island.SleepTracker (solve only what moves)", "holographic_physics"):
        "CLOSED BY STRUCTURE. PROBED: `holographic_physics` is the fractional-power KINEMATICS module (five "
        "functions and one `Kinematics` class) that demonstrates encode(a+b) == bind(encode(a), encode(b)). It holds "
        "no constraint graph, no islands, and no per-frame stepper over many bodies -- there is nothing to put to "
        "sleep. The stepper with islands is `softbody`, and it is wired.",
    # These two were named as MarginCache clients in the first wiring audit, then retired together for one reason.
    # Split into per-cache entries (matching the REGISTRY clients list) so each retirement is independently
    # queryable -- the shared reason is stated once and referenced by both.
    (("cachehome.MarginCache (fat margin for a drifting query)", "holographic_lightcache")):
        "CLOSED BY STRUCTURE, not by measurement. `lightcache` and `domecache` are STATELESS per-frame SCREEN-SPACE "
        "stride caches: they bake every Nth pixel of one frame and interpolate the rest. There is no query STREAM to "
        "drift -- the query is a pixel grid, regenerated whole on every call, and the functions hold no state between "
        "frames. A fat margin needs a drifting query and a cache that outlives it. The state lives one level up, in "
        "`RenderSession`, whose camera IS the drifting query -- and that is where MarginCache is now wired. Naming "
        "these two as clients was an error in the first wiring audit.",
    (("cachehome.MarginCache (fat margin for a drifting query)", "holographic_domecache")):
        "CLOSED BY STRUCTURE, not by measurement. `lightcache` and `domecache` are STATELESS per-frame SCREEN-SPACE "
        "stride caches: they bake every Nth pixel of one frame and interpolate the rest. There is no query STREAM to "
        "drift -- the query is a pixel grid, regenerated whole on every call, and the functions hold no state between "
        "frames. A fat margin needs a drifting query and a cache that outlives it. The state lives one level up, in "
        "`RenderSession`, whose camera IS the drifting query -- and that is where MarginCache is now wired. Naming "
        "these two as clients was an error in the first wiring audit.",
    ("superposed width (bind_fixed / recover_all / pack)",
     "holographic_machine/holographic_flatness/holographic_query/holographic_reasoning/holographic_sequence"):
        "CLOSED BY DATA DEPENDENCY, not by measurement. The VSA machine's interpreter runs `acc = bind(acc, d)` -- "
        "a loop-CARRIED accumulator. Each bind consumes the previous one's output, so there is no set of K "
        "independent binds to batch; the sequence is the semantics. `run_batch` already provides the width that IS "
        "available here (28x at 256 accumulators), by widening the DATA, not the instruction stream. The same holds "
        "for `flatness` (x is redrawn every trial), `query`, `reasoning` and `sequence`. An early AST scan counted "
        "these as 77 batchable sites; it was wrong, and a strict scan finds three (in hopfield and ai).",
    ("coarsefirst.refine_where_uncertain", "holographic_nystrom"):
        "MEASURED, and it is the most interesting of the four retirements, because the coarse-first GATE PASSES and "
        "the method still loses. Adaptive Nystrom -- greedily pivot the next landmark onto the point of maximum "
        "diagonal residual r(x) = k(x,x) - k_x^T pinv(Wmm) k_x (pivoted Cholesky; Fine & Scheinberg 2001) -- scores "
        "subspace alignment 0.8434 +- 0.0207 on clusters-with-outliers, against farthest-point sampling's 0.8692 "
        "and uniform random's 0.9512 (50 trials: 10 data seeds x 5 landmark seeds, m=24, n_basis=6). It never wins "
        "on any of three datasets, and it is WORST exactly where FPS is worst -- because the point of maximum "
        "residual IS the most isolated point. The residual is a COVERAGE signal; a spectral embedding needs a MASS "
        "signal, since the leading eigenvectors carry their weight in the DENSE regions. And `concentration` cannot "
        "see this: it scores 0.328 / 0.403 / 0.321 on the three datasets, flat and comfortably 'concentrated'. The "
        "gate says CANDIDATE and the measurement says no -- which is precisely what 'necessary, not sufficient' "
        "means. A concentrated uncertainty that concentrates on the WRONG THING. "
        "(Separately measured and recorded in nystrom's own docstring: the shipped landmarks='fps' default is "
        "materially worse than 'random' on outlier-laden data, 0.869 against 0.951, random winning 47 of 50 -- and "
        "the docstring had claimed the opposite.)",
    ("coarsefirst.refine_where_uncertain", "holographic_render"):
        "MEASURED, and it is the strongest kind of retirement: the client already implements coarse-first, under "
        "other names, better. `volume_render`'s `empty_skip` (do not sample rays in empty macro-cells) and "
        "`early_term` (stop sampling a ray once it is opaque) are spatial and temporal escalation. On a two-blob "
        "scene at 96 steps they cost 22,461 field evaluations against 341,184 with both off -- 15.2x. Running "
        "coarse-first on top (render at 12 steps, escalate the top 20% by alpha gradient to 96) costs 23,137 "
        "evaluations: 1.0x, the coarse pass is pure overhead, because the pixels a gradient flags are the same "
        "pixels that hit the volume. Turn the smart base OFF and coarse-first works exactly as advertised -- "
        "115,512 evaluations against 341,184 (3.0x) at IDENTICAL RMSE (0.00145), with a random-mask control at the "
        "same budget landing 8.8x worse. So the signal is real and the escalation is real; the client simply had "
        "5x better adaptivity already. `volume_render(only=mask)` was added to run this experiment and kept: it is "
        "exact inside the mask and free outside.",
    ("coarsefirst.refine_where_uncertain", "holographic_volint"):
        "MEASURED: there is no loop to escalate. `volint` evaluates the volumetric line integral in CLOSED FORM -- "
        "one inner product per ray, because the FPE basis is a phase code and the integral of a complex exponential "
        "is a complex exponential. Its cost is flat in ray LENGTH (1023 / 1092 / 1072 / 1089 ms at L = 0.5 / 2 / 8 / "
        "64, a 128x range) and linear in ray COUNT (265 us/ray, constant). Coarse-first buys adaptivity for a method "
        "priced per cell; the closed form deleted the cells. The volumetric client coarse-first actually wants is "
        "`render.volume_render`, which marches every pixel with the same `steps` -- it is in PENDING. "
        "(volint's own docstring is careful about this: SCATTERING still wants marching; ABSORPTION does not.)",
    ("coarsefirst.refine_where_uncertain", "holographic_splat"):
        "MEASURED, and it fails BOTH of coarse-first's necessary conditions. (1) The expensive method must be priced "
        "PER CELL: a greedy matching pursuit is already adaptive -- it places its next primitive where the residual "
        "peaks -- so its cost is per PRIMITIVE and an escalate mask tells it nothing it did not know. Gabor atoms "
        "fitted to the residual of a uniform Gaussian base scored 21.0 dB over the whole residual and 21.0 dB "
        "restricted to the top-25% mask, at 0.9x the speed: zero win, some loss. (2) The uncertainty must be "
        "CONCENTRATED, and a greedy coarse pass DESTROYS the concentration its own refinement needs -- on a grating "
        "patch over a smooth background, the residual of a uniform Gaussian base scores 0.416 and the residual of a "
        "greedy base of the same size scores 0.106. Coarse-first wants a cheap, uniform, DUMB base pass. "
        "`concentration` predicted this before any code was written; I ran the experiment anyway, and it was right.",

    ("lowdiscrepancy", "holographic_sampling"):
        "`sampling` is Bridson POISSON-DISK (blue noise) -- a different family. Blue noise optimises a minimum-distance "
        "point set for stippling/splat placement; low-discrepancy optimises integration error. They are complementary, "
        "not one wrapping the other.",
    ("lowdiscrepancy", "holographic_globalillum"):
        "globalillum does not sample directions itself -- it delegates to `Sampling.cosine_hemisphere` (samplinghome). "
        "That IS the real client, and it is now wired (opt-in `low_discrepancy=True`), so globalillum gets the win for "
        "free. Wiring globalillum directly would re-create the duplicate this consolidation removed.",
    ("octnormal", "holographic_mesh/holographic_meshcurvature/holographic_render/holographic_splat/holographic_gbuffer"):
        "PROBED: none of these five quantizes normals at all -- they carry float normals in memory (splat does not "
        "even mention a normal). Wiring octnormal here would INTRODUCE lossy compression nobody asked for and change "
        "render/mesh output. octnormal is a STORAGE primitive; its real client is `autobump` (wired). The genuine "
        "future opportunity is the SERIALIZERS (`gltf`, `splatexport`), which is a format change -- a build with a "
        "bar (0.021 deg at 3 bytes/normal), not a wiring job.",
    ("determinism.hash_unit", "holographic_mis"):
        "PROBED: `mis` calls default_rng in exactly ONE place -- inside its own `_selftest`. There is no sampling "
        "path to make stateless; a seeded selftest on one machine is already reproducible. It was never a client.",
    ("determinism.hash_unit", "holographic_traverse"):
        "PROBED: same as `mis` -- its only default_rng use is inside `_selftest`. Not a sampling path.",
    ("rns / reduce_sum_exact", "holographic_fountain"):
        "fountain reduces by XOR, which is already exact, associative and commutative -- a bit-exact integer sum "
        "would add quantization for no gain. It was never a real client of this unifier.",
}


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_module(name, root=None):
    """The path of `name`.py anywhere under holographic/ (the package layout groups modules into subfolders)."""
    root = root or _repo_root()
    for dirpath, _dirs, files in os.walk(os.path.join(root, "holographic")):
        if name + ".py" in files:
            return os.path.join(dirpath, name + ".py")
    return None


def cites(client, unifier, root=None):
    """Does `client` cite `unifier`? A plain text search of the source -- no importing, so it is fast and has no
    side effects. Returns True / False, or None if the client module doesn't exist."""
    path = _find_module(client, root)
    if path is None:
        return None
    src = open(path, "r", encoding="utf-8", errors="ignore").read()
    return any(sym in src for sym in REGISTRY[unifier]["symbols"])


def status(root=None):
    """[(unifier, client, wired_bool_or_None), ...] over the whole registry, in a stable order."""
    out = []
    for unifier in REGISTRY:
        for client in REGISTRY[unifier]["clients"]:
            out.append((unifier, client, cites(client, unifier, root)))
    return out


def _markdown(root=None):
    lines = ["# leCore Unifiers — the promote-and-wire registry", "",
             "*Generated by `python tools/unifiers.py --markdown`. A unifier wired at ONE call site is still a silo,*",
             "*so this table is checked in CI by `tests/test_unifier_adoption.py`.*", ""]
    for unifier, meta in REGISTRY.items():
        rows = [(c, cites(c, unifier, root)) for c in meta["clients"]]
        wired = sum(1 for _c, w in rows if w)
        lines += ["## `%s`  — %d/%d wired" % (unifier, wired, len(rows)), "",
                  "*%s*" % meta["why"], "", "| client | status |", "|---|---|"]
        for c, w in rows:
            lines.append("| `%s` | %s |" % (c, "✅ wired" if w else ("— missing" if w is None else "❌ not wired")))
        lines.append("")
    lines += ["## Retractions", "",
              "**Closed by mathematics** (do not re-propose):", ""]
    for (u, c), why in sorted(NOT_APPLICABLE.items()):
        lines.append("* `%s` -> `%s`: %s" % (u, c, why))
    lines += ["", "**Possible, but measurement says it does not pay** (revisit with a better lever):", ""]
    for (u, c), why in sorted(DEFERRED.items()):
        lines.append("* `%s` -> `%s`: %s" % (u, c, why))
    lines.append("")
    return "\n".join(lines)


def main(argv):
    root = _repo_root()
    if "--markdown" in argv:
        print(_markdown(root))
        return 0
    for unifier, meta in REGISTRY.items():
        rows = [(c, cites(c, unifier, root)) for c in meta["clients"]]
        wired = sum(1 for _c, w in rows if w)
        print("%-30s %d/%d wired" % (unifier, wired, len(rows)))
        for c, w in rows:
            if not w:
                print("      not wired: %s" % c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
