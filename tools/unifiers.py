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
}

# Clients that are KNOWN not to cite their unifier yet. This is the backlog, made executable: the lint asserts this
# set EXACTLY matches reality, so wiring one forces you to delete the line (progress is recorded), and un-wiring one
# fails CI (a silo cannot re-form quietly).
PENDING = set()
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
    ("iterate.step_k / limit", "holographic_diffuse"):
        "A closed form EXISTS near a fixed point: a learned bind reproduces the softmax denoise at cos 0.976 one "
        "step out, and 0.949 for an 8-step closed-form jump, with linearisation error scaling as eps^1.81 -- second "
        "order, the Koopman / modular-flow structure. But a settle() run from noise spends its steps OUTSIDE that "
        "ball and is already converged by the time it enters; once inside, one argmax returns the fixed point "
        "exactly. Valid where it is useless. Globally a single bind cannot hold several attractors (cos 0.078).",
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
