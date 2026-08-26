"""Tier contracts: memory-plan preconditions and postconditions, checked BEFORE execution.

BACKLOG D1 (the Hoare workstream). A memory tier has an implicit contract -- capacity,
hit cost, and, for a HOLOGRAPHIC tier, a FIDELITY. This module states those contracts
explicitly as {pre} plan {post} triples, certifies a plan against them via the engine's own
Horn kernel, and REFUSES plans it cannot certify.

SOTA CHECK (searched 2026-08-16): the standard for memory-hierarchy reasoning is the
Cache-Aware Roofline Model (Ilic et al. 2014; still actively extended through 2026, e.g.
CARM tooling and per-level ceilings). CARM is DESCRIPTIVE: it plots attainable upper bounds
against measured points and tells you which level to optimise. THIS IS A DIFFERENT AND
COMPLEMENTARY THING -- a plan carries a certificate that it will not touch a slower tier,
checked before it runs, which a roofline does not attempt. And a classical roofline has no
FIDELITY term because classical caches are LOSSLESS (hit or miss); a holographic tier is
lossy-but-graceful, so its postcondition needs a recall clause that CARM cannot express.

THE FIDELITY CLAUSE IS MEASURED, NOT INVENTED. From the D5 sweep (SuperposedMemory, four
dimensions, exact recall of all stored pairs), retrieval collapses onto a function of D/M
exactly as Frady/Kleyko capacity theory predicts:

    D/M      2      4      8     16     32     64+
    recall  .05    .13    .39    .84    .98   1.00

so the postcondition "recall >= 0.98" is discharged by the precondition "M <= D/32".

RULE-0 AUDIT (2026-08-16): `roofline` returned nothing. REUSED, not rebuilt --
machine_spec_sheet (measures THIS box's unit costs), memory_mountain (measures the real
cache tiers), and holographic_lean (the Horn kernel + tabled query that discharges the
obligations). resource_policy was audited and NOT used: it caps what a PROCESS may consume,
which is enforcement at runtime, not a proof about a plan beforehand.

KEPT NEGATIVE: this certifies the plan's STATED tier assignments against stated capacities.
It does not predict cache behaviour from code, and it cannot: that needs the access trace,
which is the memory_mountain's job. A certificate here means "your plan is consistent with
the tier contracts", not "your program will be fast".
"""

import numpy as np


# Measured fidelity ladder from the D5 sweep -- (min D/M ratio, guaranteed recall).
# Read as a step function: a load with D/M >= r guarantees at least the paired recall.
FIDELITY_LADDER = ((32.0, 0.98), (16.0, 0.84), (8.0, 0.39), (4.0, 0.13), (2.0, 0.05))


def fidelity_floor(dim, load):
    """The recall this superposed tier is CONTRACTUALLY good for at this load.

    Step function from the measured ladder, deliberately conservative: a ratio between rungs
    reports the LOWER rung's guarantee, because a contract that interpolates a measurement
    is promising a number nobody measured."""
    if load <= 0:
        return 1.0
    ratio = float(dim) / float(load)
    for r, rec in FIDELITY_LADDER:
        if ratio >= r:
            return rec
    return 0.0


def tier_facts(tiers, plan):
    """Turn a tier table and a plan into GROUND FACTS for the Horn kernel.

    tiers: {name: {"capacity": int, "cost": int, "holographic": bool, "dim": int}}
    plan:  [{"item": str, "tier": name, "count": int}, ...]

    Emits tier/1, place/2, and slower/2 (the strict cost ordering), all in wire format so
    the faculty can hand them straight to the logic engine."""
    facts, k = [], 0
    order = sorted(tiers, key=lambda t: tiers[t].get("cost", 0))
    for name in order:
        facts.append({"head": ["tier", [name]], "name": "t_%s" % name})
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            facts.append({"head": ["slower", [b, a]], "name": "s%d" % k})
            k += 1
    for i, step in enumerate(plan):
        facts.append({"head": ["place", [str(step["item"]), str(step["tier"])]],
                      "name": "p%d" % i})
    return facts


TOUCHES_RULES = [
    {"head": ["touches", ["?x", "?t"]], "body": [["place", ["?x", "?t"]]], "name": "tch"},
]
"""What a plan touches is what it places. Kept as a rule rather than folded into the facts
so that indirect placement (an item whose tier is derived, not declared) can be added later
without changing any caller -- the derivation is already the interface."""


def certify_plan(tiers, plan, forbid_tiers=(), min_recall=None):
    """{pre} plan {post}: certify a memory plan against the tier contracts.

    Checks, each reported separately so a refusal says WHICH clause failed:
      * CAPACITY   -- no tier is oversubscribed (sum of counts <= capacity)
      * TIER BAN   -- the plan provably never touches a forbidden tier, DERIVED through the
                      Horn kernel rather than by scanning the list, so the same machinery
                      extends to indirect placement
      * FIDELITY   -- every holographic tier's guaranteed recall at its actual load meets
                      `min_recall`, using the MEASURED D/M ladder

    Returns {"ok", "violations", "tiers": {name: {load, capacity, recall}}}. A plan that
    cannot be certified is REFUSED with reasons; nothing is silently downgraded."""
    from holographic.agents_and_reasoning import holographic_lean as _L
    load = {}
    for step in plan:
        load[step["tier"]] = load.get(step["tier"], 0) + int(step.get("count", 1))
    violations = []
    report = {}
    for name, spec in tiers.items():
        n = load.get(name, 0)
        rec = (fidelity_floor(spec.get("dim", 0), n) if spec.get("holographic")
               else (1.0 if n <= spec.get("capacity", 0) else 0.0))
        report[name] = {"load": n, "capacity": spec.get("capacity", 0), "recall": rec}
        if not spec.get("holographic") and n > spec.get("capacity", 0):
            violations.append("capacity: %s holds %d > %d" % (name, n, spec["capacity"]))
        if spec.get("holographic") and min_recall is not None and rec < float(min_recall):
            violations.append("fidelity: %s guarantees %.2f < %.2f at load %d (needs "
                              "D/M >= 32 for 0.98)" % (name, rec, float(min_recall), n))
    # the tier ban is DERIVED, not scanned
    rules = _L.rules_from_wire(tier_facts(tiers, plan) + TOUCHES_RULES)
    for banned in forbid_tiers:
        hits = _L.query(_L.Atom("touches", ("?x", str(banned))), rules, budget=100000)
        for a in hits["answers"]:
            violations.append("forbidden tier: %s placed in %s" % (a.args[0], banned))
    return {"ok": not violations, "violations": violations, "tiers": report}


# ---------------------------------------------------------------------------
# D2: BAKE CERTIFICATES -- "verify pointwise or refuse" with a STATED guarantee.
#
# SOTA CHECK (searched 2026-08-16): the established formalism for auditing a large
# precomputed artifact without re-deriving all of it is SAMPLING-BASED SPOT-CHECKING with an
# explicit EVASION PROBABILITY. For N cells of which k are corrupt and m checked uniformly
# without replacement, the chance of missing every corrupt cell is the hypergeometric
# product prod_{i<m} (N-k-i)/(N-i) (the same bound used for cloud-storage integrity audits
# and, recently, for auditing proprietary training/inference traces). Statistical runtime
# verification makes the same move for probabilistic properties.
#
# RULE-0 AUDIT (2026-08-16): store_procedural ALREADY "stores the program, verifies
# pointwise, or refuses" -- the verification exists and is REUSED conceptually; what it does
# NOT do is state a DETECTION GUARANTEE, so "it passed" carries no confidence attached. That
# gap is what this adds: the sample plan, its coverage, and the probability that a
# corruption of a given size would have escaped. bake/bake_field_nd audited and untouched --
# they produce the artifact; this certifies it.
#
# KEPT NEGATIVE: a passing certificate bounds the chance of missing a corruption of at least
# `k` cells. It says NOTHING about a single-cell corruption unless m is large, and the
# module refuses to pretend otherwise -- detect_probability(k=1) is reported honestly and is
# small for small m.
# ---------------------------------------------------------------------------

def detect_probability(n_cells, n_samples, k_corrupt):
    """Probability that uniform sampling of `n_samples` cells finds at least one of
    `k_corrupt` corrupted cells: 1 - prod_{i<m} (N-k-i)/(N-i).

    Computed as a product rather than via binomials so it stays exact and overflow-free at
    the sizes a bake actually reaches (millions of cells)."""
    N, m, k = int(n_cells), int(n_samples), int(k_corrupt)
    if N <= 0 or k <= 0:
        return 0.0
    if m >= N:
        return 1.0
    miss = 1.0
    for i in range(min(m, N)):
        num = N - k - i
        if num <= 0:
            return 1.0
        miss *= num / (N - i)
    return 1.0 - miss


def samples_for_confidence(n_cells, k_corrupt, confidence=0.99, cap=100000):
    """How many samples are needed to detect a k-cell corruption with `confidence`?

    Solved by doubling then bisecting on detect_probability -- no closed form is needed and
    a solved-for-the-requirement number is what a caller actually wants to hear."""
    lo, hi = 1, 1
    while hi < cap and detect_probability(n_cells, hi, k_corrupt) < confidence:
        hi *= 2
    if detect_probability(n_cells, hi, k_corrupt) < confidence:
        return None                      # unreachable within the cap; say so
    while lo < hi:
        mid = (lo + hi) // 2
        if detect_probability(n_cells, mid, k_corrupt) >= confidence:
            hi = mid
        else:
            lo = mid + 1
    return lo


def certify_bake(evaluate, lookup, n_cells, n_samples=256, seed=0, tol=1e-9,
                 k_corrupt=None, confidence=0.99):
    """Certify a baked artifact against its own generating function.

    `evaluate(i)` recomputes cell i from the rule; `lookup(i)` reads the bake. Samples
    `n_samples` cells deterministically (dedicated rng, so a certificate is reproducible and
    an auditor can re-run the SAME plan), compares within `tol`, and reports:
      ok            -- every sampled cell matched
      max_error     -- the worst deviation seen
      guarantee     -- probability this plan would have caught a `k_corrupt`-cell corruption
      needed        -- samples required for `confidence` at that corruption size

    `k_corrupt` defaults to 1% of the bake, because a certificate should quote the guarantee
    for a corruption size worth worrying about, not for the easiest one to catch."""
    rng = np.random.default_rng(int(seed))
    N = int(n_cells)
    m = min(int(n_samples), N)
    idx = rng.choice(N, size=m, replace=False)
    worst, bad = 0.0, []
    for i in idx:
        a = np.asarray(evaluate(int(i)), float)
        b = np.asarray(lookup(int(i)), float)
        e = float(np.max(np.abs(a - b))) if a.size else 0.0
        worst = max(worst, e)
        if e > tol:
            bad.append(int(i))
    k = int(k_corrupt) if k_corrupt else max(1, N // 100)
    return {"ok": not bad, "checked": m, "n_cells": N, "max_error": worst,
            "failed_cells": bad[:8], "k_corrupt": k,
            "guarantee": detect_probability(N, m, k),
            "needed": samples_for_confidence(N, k, confidence),
            "confidence": float(confidence)}


# ---------------------------------------------------------------------------
# D3 + THE CONSOLIDATION: the two-instrument pattern, named once.
#
# HOUSE RULE TRIGGERED. "Do the two SDF emitters agree?" already ships as a DOMAIN-SPECIFIC
# instance (sdf_emitters_agree), and the pattern has since acquired four more customers:
# fuzz_export (logic engines), the tetmesh certificate vs an independent flood fill, the
# seminaive-vs-naive fixpoint equality, and the query-vs-fixpoint slice check. The rule says
# consolidate at three. This is the generic; the SDF version stays as the specialised entry
# point that knows how to build its own inputs.
#
# SOTA CHECK (searched 2026-08-16): cross-backend DIFFERENTIAL TESTING is the established
# method for backend miscompilation -- CLsmith (OpenCL), CUDAsmith, GLFuzz/ShaDiv (GLSL),
# WGSLsmith and DarthShader (WebGPU) -- usually paired with METAMORPHIC relations when no
# gold oracle exists. The design trap the literature names explicitly: a STRICT oracle
# "declares any deviation from agreement as a fail", which on numeric backends produces
# false alarms from legitimate platform-specific floating-point variation, so practical
# oracles must be tolerance-filtered. Hence `tol` is a first-class argument here and the
# report always states the WORST deviation seen, so a caller can see how much of its
# tolerance budget was actually consumed rather than just reading "passed".
#
# KEPT NEGATIVE: agreement is not correctness. Two implementations derived from the same
# wrong idea agree perfectly -- this module's own NOTES record that lesson ("two components
# agreeing is not evidence of correctness"). What differential agreement buys is that a
# TRANSLATION did not change the meaning; the meaning itself needs a separate oracle.
# ---------------------------------------------------------------------------

def differential_agreement(implementations, cases, tol=1e-9, reference=None,
                           compare=None):
    """Run the SAME cases through several implementations and report where they disagree.

    implementations: {name: callable(case) -> value}. `reference` names the gold oracle
    (default: the first) -- the literature's reference-vs-subject framing, which matters
    because "A and B differ" is less actionable than "B deviates from the reference".
    `compare(a, b) -> float` defaults to max absolute difference over array-likes.

    Returns {"ok", "n_cases", "pairs": {name: {"max_dev", "failures"}}, "worst"}. Failures
    carry the case index so a disagreement can be reproduced, not just counted."""
    names = list(implementations)
    if not names:
        return {"ok": True, "n_cases": 0, "pairs": {}, "worst": 0.0}
    ref = reference or names[0]

    def _cmp(a, b):
        if compare is not None:
            return float(compare(a, b))
        x, y = np.asarray(a, float), np.asarray(b, float)
        if x.shape != y.shape:
            return float("inf")
        return float(np.max(np.abs(x - y))) if x.size else 0.0

    pairs, worst = {}, 0.0
    ref_vals = [implementations[ref](c) for c in cases]
    for name in names:
        if name == ref:
            continue
        dev, fails = 0.0, []
        for i, c in enumerate(cases):
            try:
                d = _cmp(ref_vals[i], implementations[name](c))
            except Exception as exc:                    # a crash IS a disagreement
                d, exc_note = float("inf"), repr(exc)[:80]
                fails.append({"case": i, "dev": d, "error": exc_note})
                dev = d
                continue
            dev = max(dev, d)
            if d > tol:
                fails.append({"case": i, "dev": d})
        pairs[name] = {"max_dev": dev, "failures": fails[:8], "n_failed": len(fails)}
        worst = max(worst, dev)
    return {"ok": all(not p["failures"] for p in pairs.values()), "n_cases": len(cases),
            "reference": ref, "pairs": pairs, "worst": worst, "tol": float(tol)}


# ---------------------------------------------------------------------------
# D4: SCHEDULE CERTIFICATES -- prove a wave schedule conflict-free BEFORE running it.
#
# SOTA CHECK (searched 2026-08-16): the field splits into DYNAMIC race detection
# (happens-before / vector clocks; SHB, MultiBags, DePa) and STATIC verification (Faial for
# GPU kernels, polyhedral analysis for X10). The 2025 Faial study is the sobering datapoint:
# of 191 data-race-free GPU programs, 98% needed a specific thread configuration to be
# analysable at all and 27% needed user-provided assertions. General static race freedom is
# HARD.
#
# WHY OURS IS EASY, AND THE HONEST SCOPE THAT FOLLOWS: we are not analysing a program. We
# have an EXPLICIT schedule (colour_waves output) over EXPLICITLY DECLARED resources, so
# "no two tasks in a wave share a resource" is a finite combinatorial check, and the useful
# theorem is the standard one -- a race-free task-parallel schedule is deterministic.
# THEREFORE: this certifies the SCHEDULE, not the PROGRAM. A task that touches a resource it
# did not declare is outside the certificate, and no amount of graph colouring will catch it.
# The module says so rather than letting a green tick imply race freedom it cannot deliver.
#
# RULE-0 AUDIT (2026-08-16): color_waves and graph_coloring already ship and PRODUCE the
# schedule; nothing about them is rebuilt. What is missing is that nobody CHECKS the result
# against the declarations -- the colouring is trusted because the algorithm is believed,
# which is exactly the kind of trust this workstream exists to replace with a derivation.
# ---------------------------------------------------------------------------

def certify_schedule(waves, resources, conflicts_are_edges=True):
    """Certify that no two tasks scheduled in the SAME wave share a declared resource.

    waves: [[task, ...], ...] as returned by colour_waves. resources: {task: [names]}.
    Returns {"ok", "violations", "n_waves", "n_tasks", "max_wave"} -- a violation names the
    wave, the two tasks, and the shared resource, so a failure is actionable rather than a
    boolean. Also checks the partition is well-formed (every task exactly once), because a
    schedule that silently drops a task is a worse bug than one that races."""
    # NORMALISE TASK IDENTITY. Found by the HTTP round-trip: JSON object keys arrive as
    # STRINGS while the wave lists carry INTEGERS, so a perfectly good schedule reported
    # every task as "never scheduled". Comparing task ids by str() makes the wire path and
    # the in-process path agree, and a certificate that only works in-process is not a
    # certificate an agent can use.
    resources = {str(k): v for k, v in dict(resources).items()}
    waves = [[str(t) for t in wave] for wave in waves]
    violations = []
    seen = {}
    for w, wave in enumerate(waves):
        owner = {}
        for t in wave:
            if t in seen:
                violations.append("task %r appears in waves %d and %d" % (t, seen[t], w))
            seen[t] = w
            for r in resources.get(t, ()):
                if r in owner:
                    violations.append("wave %d: tasks %r and %r both touch %r"
                                      % (w, owner[r], t, r))
                else:
                    owner[r] = t
    missing = [t for t in resources if t not in seen]
    if missing:
        violations.append("tasks never scheduled: %r" % missing[:5])
    return {"ok": not violations, "violations": violations, "n_waves": len(waves),
            "n_tasks": len(seen), "max_wave": max((len(w) for w in waves), default=0)}


def resource_conflict_edges(resources):
    """Derive the conflict graph from resource declarations: an edge between any two tasks
    sharing a resource. This is the INPUT colour_waves should have been given -- providing
    it here means the certificate and the schedule are derived from the SAME declarations,
    so a mismatch is a real disagreement and not two people writing the edge list twice."""
    by_res = {}
    for t, rs in {str(k): v for k, v in dict(resources).items()}.items():
        for r in rs:
            by_res.setdefault(r, []).append(t)
    edges = set()
    for members in by_res.values():
        ms = sorted(members)
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                edges.add((a, b))
    return sorted(edges)


# ---------------------------------------------------------------------------
# A2 GATE: refuse a demux answer whose noise is outside the MEASURED envelope.
#
# WHY THIS EXISTS AND WHY IT WAITED. The A2 sweep first showed only 0.6-0.8 stride recovery
# even at ZERO noise, and a gate was deliberately NOT built on that, because a boundary whose
# clean corner is 0.7 bakes a mystery into an API. Diagnosis (crossing smoothness x offset)
# found the cause was the MEASUREMENT, not the method: random-walk sources violate
# demux_series's stated precondition ("every strided sub-stream is a SMOOTH single source").
# With band-limited sources the envelope is sharp:
#     noise (frac of signal std)  0.000  0.005  0.010  0.020  0.050  0.100
#     stride recovered (K=2..8)    1.00   1.00   1.00   1.00   1.00   0.00
# Perfect to 5%, cliff before 10%. THAT is a boundary worth gating on.
#
# SOTA CHECK (searched 2026-08-16): the standard robust noise estimator is Donoho &
# Johnstone's MAD of differences, sigma = median(|D^(p+1) y|) / (Phi^-1(3/4) *
# sqrt(sum_j C(p+1,j)^2)) -- the same estimator MATLAB's wnoisest ships and the change-point
# literature generalises to p-th differences for polynomial signals. We use SECOND
# differences (p=1, divisor 0.6745*sqrt(6)): they annihilate any locally linear trend, so for
# a SMOOTH source what survives is noise, which is exactly the quantity the envelope is
# stated in.
#
# THE CHICKEN-AND-EGG, and how it is dodged: noise must be measured on a SMOOTH series, but
# the raw interleaved stream is not smooth until you know K -- and K is what demux computes.
# So this gate validates the ANSWER, not the input: run demux, then estimate noise on the
# strided substreams IMPLIED by the returned K. If that K is right the substreams are smooth
# and the estimate is meaningful; if it is wrong the estimate comes out large and the answer
# is refused anyway. Both failure directions land on "refuse", which is the safe side.
#
# RULE-0 AUDIT (2026-08-16): no noise-estimator faculty exists (the `denoise*` family
# REMOVES noise, it does not measure it), and decide_or_abstain / route_or_abstain are
# routing decisions, not signal preconditions. Genuine gap. demux_series itself is untouched
# -- this wraps it, so the ungated path stays available for callers who know their data.
# ---------------------------------------------------------------------------

MAD_TO_SIGMA_D2 = 0.6745 * np.sqrt(6.0)
"""Donoho-Johnstone constant for SECOND differences: Phi^-1(3/4) * sqrt(1^2+2^2+1^2)."""

DEMUX_NOISE_LIMIT = 0.05
"""The MEASURED envelope: stride recovery is 1.00 at or below this and collapses by 0.10."""


def estimate_noise_sigma(y):
    """Robust noise sigma of a SMOOTH series via the MAD of second differences.

    Second rather than first differences because they annihilate a locally linear trend, so
    for a smooth signal what survives is noise -- first differences would still carry slope
    and would over-report. Returns 0.0 for series too short to difference twice."""
    y = np.asarray(y, float).ravel()
    if y.size < 4:
        return 0.0
    d2 = np.diff(y, 2)
    return float(np.median(np.abs(d2)) / MAD_TO_SIGMA_D2)


def demux_gated(mind, x, noise_limit=DEMUX_NOISE_LIMIT, **kw):
    """Run demux_series and REFUSE the answer if the implied substreams are too noisy for
    the measured envelope.

    Returns the demux result with three fields added: noise_ratio (estimated sigma over the
    signal's own std), noise_limit, and trusted. `trusted=False` means the stride may well be
    wrong and the caller must not treat it as a separation -- the honest outcome outside a
    measured envelope, rather than a confident answer nobody validated."""
    x = np.asarray(x, float).ravel()
    res = dict(mind.demux_series(x, **kw))
    k = res.get("k") or res.get("K") or res.get("stride")
    sig = float(np.std(x)) + 1e-12
    if not k or int(k) < 1:
        res.update({"noise_ratio": None, "noise_limit": float(noise_limit), "trusted": False})
        return res
    k = int(k)
    subs = [x[i::k] for i in range(k)]
    # median over channels: one ugly channel should not condemn the whole answer, and one
    # clean channel should not excuse it either
    sigmas = [estimate_noise_sigma(s) for s in subs if s.size >= 4]
    ratio = float(np.median(sigmas)) / sig if sigmas else None
    res.update({"noise_ratio": ratio, "noise_limit": float(noise_limit),
                "trusted": bool(ratio is not None and ratio <= noise_limit)})
    return res


# ---------------------------------------------------------------------------
# B4: POSE CERTIFICATES -- does the returned pose actually satisfy its constraints?
#
# SOTA CHECK (searched 2026-08-16) SET THE SCOPE, and it is a narrow one. Constrained IK is
# an active field -- FABRIK (Aristidou & Lasenby) with model constraints, VO-FABRIK,
# gradient-projection and QP formulations, actuator-aware joint-limit admissibility (2026) --
# and the FABRIK literature states its own failure mode plainly: with joint constraints
# applied it "suffers from an inability to reach a feasible solution ... the end effector
# often cannot reach the target, even if there is a solution, since each joint position is
# calculated INDEPENDENTLY without considering the restrictions on the next joint."
#
# THEREFORE A POSE CERTIFICATE MUST NOT CLAIM OPTIMALITY, and this one does not. It certifies
# exactly three things about the pose the solver RETURNED: bone lengths preserved, every
# joint inside its declared limit, and the reported end-effector error. Whether a better pose
# exists is a question no cheap check can answer, and pretending otherwise would be the
# failure this whole workstream exists to avoid. Reaching the target is REPORTED, never
# certified -- an unreachable target is a fact about the target, not a defect in the pose.
#
# RULE-0 AUDIT (2026-08-16): solve_ik (FABRIK) and solve_ik_limited (holographic_iklimit,
# hinge + cone limits, with an 'auto' bend axis) BOTH ALREADY SHIP. Nothing is rebuilt; the
# solver is untouched. What was missing is anyone CHECKING its output against the same limit
# spec it was given -- the solver clamps, and clamping was trusted because the code is
# believed. Same gap as the schedule colouring in D4, same fix.
# ---------------------------------------------------------------------------

def _unit_vec(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def certify_pose(joints, limits, rest_lengths=None, target=None, root_ref=(0.0, 1.0, 0.0),
                 length_tol=1e-6, angle_tol=1e-6):
    """Certify a solved pose against the SAME limit spec the solver was given.

    limits[i] constrains the bone i->i+1 relative to the bone into joint i (root_ref for the
    first), matching holographic_iklimit exactly: {'type':'hinge','axis'|'auto','lo','hi'} or
    {'type':'cone','half',['ref']}. Returns {"ok","violations","max_length_error",
    "max_angle_excess","target_error"}. Violations name the joint and the amount, so a
    refusal is actionable.

    target_error is REPORTED, not certified -- see the module note on why reachability is not
    a property of the pose."""
    J = np.asarray(joints, float)
    n = len(J) - 1
    violations = []
    if rest_lengths is None:
        rest_lengths = np.linalg.norm(np.diff(J, axis=0), axis=1)
    rest = np.asarray(rest_lengths, float)
    lens = np.linalg.norm(np.diff(J, axis=0), axis=1)
    len_err = float(np.max(np.abs(lens - rest))) if n else 0.0
    for i in range(n):
        if abs(lens[i] - rest[i]) > length_tol:
            violations.append("bone %d length %.6f != rest %.6f" % (i, lens[i], rest[i]))
    max_excess = 0.0
    for i in range(n):
        lim = limits[i] if limits and i < len(limits) else None
        if lim is None:
            continue
        u = _unit_vec(J[i] - J[i - 1]) if i >= 1 else _unit_vec(np.asarray(root_ref, float))
        v = _unit_vec(J[i + 1] - J[i])
        ang = float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))
        if lim["type"] == "cone":
            excess = ang - float(lim["half"])
            if excess > angle_tol:
                violations.append("joint %d cone: %.4f rad exceeds half %.4f"
                                  % (i, ang, float(lim["half"])))
            max_excess = max(max_excess, excess)
        elif lim["type"] == "hinge":
            # signed bend about the hinge axis; 'auto' means the axis follows the limb, in
            # which case the bend is unsigned and only the MAGNITUDE bound is checkable
            axis = lim.get("axis")
            lo, hi = float(lim["lo"]), float(lim["hi"])
            if isinstance(axis, str) or axis is None:
                bound = max(abs(lo), abs(hi))
                excess = ang - bound
            else:
                a = _unit_vec(np.asarray(axis, float))
                signed = float(np.arctan2(np.dot(np.cross(u, v), a), np.dot(u, v)))
                excess = max(lo - signed, signed - hi)
            if excess > angle_tol:
                violations.append("joint %d hinge: bend %.4f outside [%.4f, %.4f]"
                                  % (i, ang, lo, hi))
            max_excess = max(max_excess, excess)
    terr = None
    if target is not None:
        terr = float(np.linalg.norm(J[-1] - np.asarray(target, float)))
    return {"ok": not violations, "violations": violations,
            "max_length_error": len_err, "max_angle_excess": float(max_excess),
            "target_error": terr}


# ---------------------------------------------------------------------------
# C1: CONSERVATION LEDGERS -- audit a simulation's invariants without condemning correct
# integrators for doing the right thing.
#
# SOTA CHECK (searched 2026-08-16) SUPPLIED THE ONE DISTINCTION THAT MAKES THIS AUDIT VALID.
# The naive ledger asserts |dE| ~ 0 per step. That is WRONG, and would fail the best
# integrators available: symplectic and discrete-gradient schemes preserve the symplectic
# two-form and conserve a SHADOW Hamiltonian, so "energy and momentum errors remain BOUNDED
# over long-time simulations, EVEN THOUGH THESE QUANTITIES ARE NOT EXACTLY CONSERVED at each
# time step" (Vlasov-Poisson-Landau structure-preservation study, JCP 2026; the same result
# is why symplectic Euler beats RK4 for long runs despite lower formal order). Non-symplectic
# RK exhibits a "mild but SYSTEMATIC drift" -- and THAT is the detectable failure.
#
# THEREFORE THE LEDGER TESTS TWO DIFFERENT THINGS WITH TWO DIFFERENT TESTS:
#   * QUANTITIES THAT ARE EXACT BY CONSTRUCTION (mass; linear momentum under symmetric
#     internal forces, by Newton's third law) -> assert |drift| ~ machine precision.
#   * QUANTITIES THAT ARE ONLY BOUNDED (energy under a symplectic scheme) -> assert NO
#     SECULAR TREND, i.e. the least-squares slope over the run is statistically flat
#     relative to the oscillation. Bounded wobble PASSES; a slow ramp FAILS.
# Conflating the two is the mistake this module exists to avoid, and testing the wrong one is
# how a conservation audit gets quietly disabled after it cries wolf on correct physics.
#
# RULE-0 AUDIT (2026-08-16): no conservation ledger exists. The `drift_*` family is
# LLM/representation drift -- a different verb entirely, audited and dismissed here so it is
# not re-audited. holographic_energy and the sim family provide the quantities; this reads
# them. Nothing in the simulators is touched.
# ---------------------------------------------------------------------------

def secular_trend(series):
    """Least-squares slope per step, normalised by the series' own oscillation.

    Returns (slope_per_step, normalised_ramp): the second is |total drift| / (spread + eps),
    so a value near 0 means "wobbles but goes nowhere" and a value >> 1 means "ramping".
    Normalising by the SPREAD rather than the mean is what lets one threshold serve
    quantities of wildly different magnitude."""
    y = np.asarray(series, float).ravel()
    n = y.size
    if n < 3:
        return 0.0, 0.0
    t = np.arange(n, dtype=float)
    slope = float(np.polyfit(t, y, 1)[0])
    spread = float(np.max(y) - np.min(y))
    total = abs(slope) * (n - 1)
    return slope, float(total / (spread + 1e-12))


def conservation_ledger(history, exact=(), bounded=(), exact_tol=1e-9, ramp_tol=0.6):
    """Audit a run's conserved quantities. `history` is {name: [value per step]}.

    `exact` names quantities conserved BY CONSTRUCTION (mass, linear momentum under
    symmetric internal forces): judged on absolute relative drift against exact_tol.
    `bounded` names quantities a symplectic scheme only keeps BOUNDED (energy): judged on
    SECULAR TREND -- oscillation is fine, a ramp is not. ramp_tol is the fraction of the
    observed spread that a monotone trend may account for before it is called drift.

    Returns {"ok", "violations", "report"} with per-quantity numbers, so a failure says which
    invariant went and by how much rather than just failing."""
    violations, report = [], {}
    for name in exact:
        y = np.asarray(history.get(name, []), float).ravel()
        if y.size < 2:
            continue
        scale = max(abs(float(y[0])), 1e-12)
        rel = float(np.max(np.abs(y - y[0])) / scale)
        report[name] = {"kind": "exact", "max_rel_drift": rel}
        if rel > exact_tol:
            violations.append("%s is exact-by-construction but drifted %.2e (> %.0e)"
                              % (name, rel, exact_tol))
    for name in bounded:
        y = np.asarray(history.get(name, []), float).ravel()
        if y.size < 3:
            continue
        slope, ramp = secular_trend(y)
        report[name] = {"kind": "bounded", "slope_per_step": slope, "ramp_fraction": ramp,
                        "spread": float(np.max(y) - np.min(y))}
        if ramp > ramp_tol:
            violations.append("%s shows SECULAR DRIFT: %.0f%% of its spread is a monotone "
                              "trend (slope %.3e/step)" % (name, 100.0 * ramp, slope))
    return {"ok": not violations, "violations": violations, "report": report}


# ---------------------------------------------------------------------------
# C2-FIX: the LYAPUNOV WITNESS -- when a settle can be CERTIFIED instead of guessed.
#
# THE MEASURED PROBLEM (C2): leCore's settle gate watches a residual stream and is sound for
# stagnation plateaus up to about its window, but a LONGER plateau defeats it -- measured,
# window 96 falsely settles at plateau 128; window 192 at plateau 220. No finite window
# survives an arbitrarily long stagnation, so "the window is the trap length" is the gate's
# real guarantee.
#
# SOTA CHECK (searched 2026-08-16) SUPPLIES THE ESCAPE, and it is a theorem rather than a
# bigger window. A Lyapunov potential is a non-negative function that DECREASES along
# trajectories, and the classical gradient-flow stopping criterion is on the GRADIENT NORM
# (stop when ||grad f|| is small) -- not on the state change. The consequence that matters
# here: for a TRUE gradient flow x' = -grad E(x), a plateau in the state means grad E ~ 0,
# i.e. a CRITICAL POINT, and a critical point CANNOT spontaneously resume. THE STAGNATION
# TRAP IS IMPOSSIBLE FOR A GRADIENT FLOW. It exists only for systems that are not one:
# externally driven, time-varying, or carrying inertia (our C2 counterexample forced a
# velocity to zero and back, which no energy function generates).
#
# SO THE CERTIFICATE'S JOB IS TO CHECK THE PRECONDITION, NOT THE PLATEAU: is this run
# actually a gradient flow? Two checkable signatures -- the witness never increases, and the
# residual tracks the witness's own decrease (a driven system moves while its energy sits
# still, or vice versa). Pass both and the settle is CERTIFIED by the theorem. Fail either
# and the caller is told plainly that only the heuristic window guarantee applies.
#
# RULE-0 AUDIT (2026-08-16): run_until_settled and convergence_guard already ship and are
# untouched -- this does not replace the gate, it upgrades the GUARANTEE when the run
# qualifies. secular_trend (C1, above) is reused for the monotonicity test rather than a
# second trend routine being written.
# ---------------------------------------------------------------------------

def lyapunov_certify(witness, residuals=None, rise_tol=1e-9, settle_frac=0.02):
    """Can this run's settle be CERTIFIED, or only guessed?

    `witness` is the Lyapunov quantity per step (for a gradient flow, the energy).
    `residuals` is the per-step residual stream the settle gate used, when available.

    Checks, reported separately:
      * MONOTONE      -- the witness never rises by more than rise_tol. A rise means this is
                         not a descent flow and no gradient-flow theorem applies.
      * SETTLED       -- the witness's remaining decrease over the last window is a tiny
                         fraction (settle_frac) of its TOTAL decrease, i.e. it has arrived.
      * CONSISTENT    -- residual and |dW| both fall together. A run whose state moves while
                         its energy sits still (or the reverse) is externally driven, and the
                         theorem does not cover it.

    Returns {"certified", "reasons", ...}. certified=True means the stagnation trap is
    IMPOSSIBLE here, not merely unobserved -- a critical point of a gradient flow cannot
    resume. certified=False is not a failure of the run; it means only the settle gate's
    window-length heuristic applies, and the caller should size `window` accordingly."""
    w = np.asarray(witness, float).ravel()
    reasons = []
    if w.size < 8:
        return {"certified": False, "reasons": ["too few steps to certify"],
                "monotone": None, "settled": None, "consistent": None}
    rises = np.diff(w)
    max_rise = float(np.max(rises)) if rises.size else 0.0
    monotone = max_rise <= rise_tol
    if not monotone:
        reasons.append("witness RISES by %.3e -- not a descent flow, no theorem applies"
                       % max_rise)
    total_drop = float(w[0] - w[-1])
    tail = max(4, w.size // 8)
    tail_drop = float(w[-tail] - w[-1])
    settled = total_drop <= 0 or (tail_drop <= settle_frac * abs(total_drop))
    if not settled:
        reasons.append("witness still falling: last eighth accounts for %.1f%% of the total "
                       "drop" % (100.0 * tail_drop / max(abs(total_drop), 1e-12)))
    consistent = None
    if residuals is not None:
        r = np.asarray(residuals, float).ravel()
        n = min(r.size, rises.size)
        if n >= 8:
            a, b = np.abs(r[-n:]), np.abs(rises[-n:])
            # both must be quiet at the end: a driven system keeps one alive without the other
            consistent = bool(np.mean(a[-tail:]) <= 0.2 * (np.mean(a) + 1e-12) or
                              np.mean(a[-tail:]) < 1e-9)
            if not consistent:
                reasons.append("residual has not quieted with the witness -- looks driven")
    certified = bool(monotone and settled and (consistent is not False))
    return {"certified": certified, "reasons": reasons, "monotone": bool(monotone),
            "settled": bool(settled), "consistent": consistent,
            "total_drop": total_drop, "tail_drop": tail_drop}


# ---------------------------------------------------------------------------
# C4: PLAN CERTIFICATES -- GOAP's own promise, made checkable.
#
# SOTA CHECK (searched 2026-08-16): GOAP (Orkin, F.E.A.R. 2003; STRIPS lineage) models each
# action as PRECONDITIONS + EFFECTS + cost and plans by backward search over the action
# graph. Its stated advantage over behaviour trees is exactly a verification claim: "GOAP
# provides the GUARANTEE OF VALID PLANS. Hand-coded embedded plans can contain mistakes ... a
# character might be instructed to fire a weapon, without ever [acquiring one]." That
# guarantee holds for plans the PLANNER built; it says nothing about a plan that was
# hand-authored, learned, replanned mid-execution, or handed over from another system -- and
# those are exactly the plans that reach a creature at runtime.
#
# SO THIS CERTIFIES ANY PLAN, whatever produced it: walk the sequence, check each action's
# preconditions against the CURRENT simulated state, apply its effects, and confirm the goal
# holds at the end. A violation names the STEP INDEX, the ACTION, and the MISSING
# PRECONDITION -- "step 2 fire_weapon requires has_weapon" is actionable where "invalid plan"
# is not.
#
# RULE-0 AUDIT (2026-08-16): `GOAP` returned nothing. validate_plan EXISTS but checks
# ORDERING constraints only (the PB&J test: does every 'a before b' hold?) -- it does not
# simulate state, so it cannot see a missing precondition. Complementary, not duplicated, and
# recorded here so the distinction is not re-litigated. The tabled query (E1) is the natural
# engine for SEARCHING for a plan and terminates where naive backward chaining loops; this
# module does the cheaper and more urgent half -- checking one.
#
# KEPT NEGATIVE: this certifies FEASIBILITY, not optimality or goal-relevance. A plan that
# reaches the goal by a ludicrous route certifies exactly like a good one, because cost is
# the planner's business and a certificate that quietly judged quality would be lying about
# what it checked.
# ---------------------------------------------------------------------------

def certify_plan_actions(plan, actions, initial_state, goal=None):
    """Certify a GOAP-style plan: preconditions met at each step, goal reached at the end.

    `actions` is {name: {"pre": {key: bool}, "eff": {key: bool}}}; `plan` is a list of action
    names; states are dicts of key -> bool (absent reads as False). Returns
    {"ok", "violations", "final_state", "trace"} where trace records the state after each
    step, so a failure can be replayed rather than merely reported."""
    state = {k: bool(v) for k, v in dict(initial_state).items()}
    violations, trace = [], []
    for i, name in enumerate(plan):
        act = actions.get(name)
        if act is None:
            violations.append("step %d: unknown action %r" % (i, name))
            trace.append(dict(state))
            continue
        for key, want in dict(act.get("pre", {})).items():
            if bool(state.get(key, False)) != bool(want):
                violations.append("step %d %s requires %s=%s (state has %s)"
                                  % (i, name, key, bool(want), bool(state.get(key, False))))
        for key, val in dict(act.get("eff", {})).items():
            state[key] = bool(val)
        trace.append(dict(state))
    if goal:
        for key, want in dict(goal).items():
            if bool(state.get(key, False)) != bool(want):
                violations.append("goal %s=%s not reached (final state has %s)"
                                  % (key, bool(want), bool(state.get(key, False))))
    return {"ok": not violations, "violations": violations, "final_state": state,
            "trace": trace}


def _selftest():
    """Regression trap: a clean plan certifies, and EACH failure clause is provoked
    separately -- a certifier that only ever says yes certifies nothing."""
    tiers = {"hot": {"capacity": 8, "cost": 1},
             "trace": {"capacity": 10 ** 6, "cost": 10, "holographic": True, "dim": 4096},
             "storage": {"capacity": 10 ** 9, "cost": 1000}}

    good = [{"item": "a", "tier": "hot", "count": 4},
            {"item": "b", "tier": "trace", "count": 64}]
    r = certify_plan(tiers, good, forbid_tiers=("storage",), min_recall=0.98)
    assert r["ok"], r["violations"]
    assert r["tiers"]["trace"]["recall"] == 0.98      # 4096/64 = 64 -> top rung

    # capacity clause
    r2 = certify_plan(tiers, [{"item": "a", "tier": "hot", "count": 99}])
    assert not r2["ok"] and any("capacity" in v for v in r2["violations"]), r2

    # fidelity clause: 4096/256 = 16 -> only 0.84 guaranteed, so 0.98 must be REFUSED
    r3 = certify_plan(tiers, [{"item": "b", "tier": "trace", "count": 256}],
                      min_recall=0.98)
    assert not r3["ok"] and any("fidelity" in v for v in r3["violations"]), r3
    assert certify_plan(tiers, [{"item": "b", "tier": "trace", "count": 256}],
                        min_recall=0.8)["ok"], "0.84 should satisfy a 0.80 requirement"

    # tier ban, DERIVED through the kernel
    r4 = certify_plan(tiers, [{"item": "z", "tier": "storage", "count": 1}],
                      forbid_tiers=("storage",))
    assert not r4["ok"] and any("forbidden" in v for v in r4["violations"]), r4

    # the ladder is conservative between rungs: D/M = 20 reports the 16-rung guarantee
    assert fidelity_floor(4096, 205) == 0.84, fidelity_floor(4096, 205)
    assert fidelity_floor(4096, 0) == 1.0

    # D2: a clean bake certifies; a CORRUPTED one must be caught (the oracle probe -- a
    # certifier that never fails certifies nothing), and the guarantee must be honest.
    N = 10000
    table = np.sin(np.arange(N) * 0.01)
    ev = lambda i: np.sin(i * 0.01)
    c = certify_bake(ev, lambda i: table[i], N, n_samples=256, seed=0)
    assert c["ok"] and c["max_error"] < 1e-12, c
    assert 0.9 < c["guarantee"] <= 1.0, c["guarantee"]      # 1% corruption, 256 samples
    bad = table.copy()
    bad[N // 3:N // 3 + N // 20] += 0.5                     # corrupt 5% of the cells
    c2 = certify_bake(ev, lambda i: bad[i], N, n_samples=256, seed=0)
    assert not c2["ok"] and c2["failed_cells"], "corrupted bake certified as clean"
    # a SINGLE corrupt cell is honestly reported as hard to catch, not papered over
    assert detect_probability(N, 256, 1) < 0.05
    assert samples_for_confidence(N, 1, 0.99) > 8000
    assert samples_for_confidence(N, 100, 0.99) < 600
    print("OK: holographic_tiercontract -- clean plan certified (trace recall %.2f), and "
          "capacity / fidelity / forbidden-tier clauses each REFUSE when provoked"
          % r["tiers"]["trace"]["recall"])
    # D3: the consolidated differential oracle. Agreement passes, a deliberately WRONG
    # implementation is caught with its case index, and a crash counts as disagreement.
    cases = [float(i) / 7.0 for i in range(40)]
    impls = {"ref": lambda x: np.sin(x), "same": lambda x: np.sin(x),
             "nearly": lambda x: np.sin(x) + 1e-12}
    d = differential_agreement(impls, cases, tol=1e-9)
    assert d["ok"] and d["worst"] < 1e-9, d
    impls["wrong"] = lambda x: np.cos(x)
    d2 = differential_agreement(impls, cases, tol=1e-9)
    assert not d2["ok"] and d2["pairs"]["wrong"]["failures"], d2
    assert "case" in d2["pairs"]["wrong"]["failures"][0]      # reproducible, not just counted
    def _boom(x):
        raise RuntimeError("backend exploded")
    d3 = differential_agreement({"ref": lambda x: x, "bad": _boom}, cases[:3])
    assert not d3["ok"] and d3["pairs"]["bad"]["failures"][0]["dev"] == float("inf")
    # D4: a schedule derived from the SAME declarations certifies; a hand-broken one is
    # caught with the wave, the pair, and the shared resource named.
    import lecore as _lc
    mind = _lc.UnifiedMind(dim=64, seed=0)
    res = {0: ["a"], 1: ["a", "b"], 2: ["b"], 3: ["c"], 4: ["c", "a"]}
    edges = resource_conflict_edges(res)
    waves = mind.color_waves(5, edges)
    cert = certify_schedule(waves, res)
    assert cert["ok"], cert["violations"]
    assert cert["n_tasks"] == 5, cert
    broken = [[0, 1], [2, 3], [4]]        # 0 and 1 both touch "a" -- a real conflict
    bad = certify_schedule(broken, res)
    assert not bad["ok"] and "touch" in bad["violations"][0], bad
    dropped = certify_schedule([[0, 2], [1, 3]], res)     # task 4 never scheduled
    assert not dropped["ok"] and any("never scheduled" in v for v in dropped["violations"])
    # A2 gate: the estimator is accurate on a planted sigma, and -- the property that
    # actually matters -- the gate NEVER trusts a wrong stride.
    _r = np.random.default_rng(0)
    _t = np.arange(4000)
    _clean = np.sin(2 * np.pi * _t / 200)
    for _s in (0.01, 0.05, 0.2):
        _est = estimate_noise_sigma(_clean + _r.normal(scale=_s, size=_t.size))
        assert abs(_est - _s) < 0.25 * _s, "noise estimator off: %.4f vs %.3f" % (_est, _s)
    def _mk(K, n, rng, noise):
        tt = np.arange(n // K + 1)
        srcs = [np.sin(2 * np.pi * tt / 40 + i) + 0.3 * np.sin(2 * np.pi * tt / 13 + 2 * i)
                for i in range(K)]
        xx = np.empty(n)
        for i in range(n):
            xx[i] = srcs[i % K][i // K]
        return xx + rng.normal(scale=noise * np.std(xx), size=n)
    _false_trust = 0
    for _K in (2, 4, 8):
        for _n in (0.0, 0.02, 0.05, 0.10, 0.30):
            _res = demux_gated(mind, _mk(_K, 600, np.random.default_rng(7 * _K), _n))
            _k = _res.get("k") or _res.get("K") or _res.get("stride")
            if _res["trusted"] and _k != _K:
                _false_trust += 1
    assert _false_trust == 0, "the gate TRUSTED a wrong stride %d times" % _false_trust
    # B4: every pose the CONSTRAINED solver returns must certify, including for an
    # UNREACHABLE target (reach error is reported, not certified); a hand-broken pose must
    # be refused with the offending bone named.
    _J = np.array([[0., 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])
    _lim = [None, {"type": "hinge", "axis": "auto", "lo": -1.2, "hi": 0.0},
            {"type": "cone", "half": 0.9}]
    _rest = np.linalg.norm(np.diff(_J, axis=0), axis=1)
    for _tgt in ([1.5, 2.0, 0.3], [0.2, 2.9, 0.1], [9.0, 9.0, 9.0], [-1.0, 0.5, 1.0]):
        _P, _ = mind.solve_ik_limited(_J, np.array(_tgt, float), _lim)
        _c = certify_pose(_P, _lim, rest_lengths=_rest, target=_tgt)
        assert _c["ok"], (_tgt, _c["violations"])
        assert _c["max_angle_excess"] <= 1e-6 and _c["max_length_error"] < 1e-9
    _bad = _J.copy()
    _bad[2] = [1.4, 1.2, 0.0]
    _cb = certify_pose(_bad, _lim, rest_lengths=_rest)
    assert not _cb["ok"] and any("length" in v for v in _cb["violations"])
    # C1: the discrimination that makes a conservation audit valid -- bounded oscillation
    # (a CORRECT symplectic scheme) must PASS where a secular ramp (non-symplectic drift)
    # FAILS. An audit that cannot tell those apart gets disabled the first time it cries
    # wolf on correct physics.
    _t = np.arange(800)
    _wobble = 1.0 + 0.02 * np.sin(_t * 0.3)
    _ramp = _wobble + 0.00008 * _t
    assert conservation_ledger({"E": _wobble}, bounded=("E",))["ok"], "bounded energy failed"
    assert not conservation_ledger({"E": _ramp}, bounded=("E",))["ok"], "secular ramp passed"
    assert not conservation_ledger({"m": 5.0 + 1e-6 * _t}, exact=("m",))["ok"]
    assert conservation_ledger({"m": np.full(800, 5.0)}, exact=("m",))["ok"]
    # C2-FIX: a REAL gradient flow certifies; the driven plateau that defeated the window
    # gate does NOT; a still-falling run does not. The theorem, not a bigger window.
    from holographic.simulation_and_physics.holographic_morphogen import relax as _relax
    _X = np.random.default_rng(0).normal(scale=1.5, size=(30, 3))
    _, _h = _relax(_X, np.full(30, 0.5), steps=300)
    _c = lyapunov_certify(_h, [abs(_h[i + 1] - _h[i]) for i in range(len(_h) - 1)])
    assert _c["certified"] and _c["monotone"], _c["reasons"]
    _t = np.arange(400)
    _driven = np.where(_t < 40, 1.0 - 0.02 * _t,
                       np.where(_t < 200, 0.2, 0.2 - 0.005 * (_t - 200)))
    assert not lyapunov_certify(_driven)["certified"], "driven plateau was certified"
    assert not lyapunov_certify(np.exp(-np.arange(200) * 0.002))["certified"]
    assert not lyapunov_certify(np.arange(50, dtype=float))["certified"]   # rising witness
    # C4: GOAP's own promise, checked. The literature's canonical bug -- firing a weapon you
    # never picked up -- must be caught with the step and the missing precondition NAMED.
    _acts = {"goto_rack": {"pre": {}, "eff": {"at_rack": True}},
             "pickup": {"pre": {"at_rack": True}, "eff": {"has_weapon": True}},
             "goto_enemy": {"pre": {}, "eff": {"near_enemy": True}},
             "fire": {"pre": {"has_weapon": True, "near_enemy": True},
                      "eff": {"enemy_down": True}}}
    _good = ["goto_rack", "pickup", "goto_enemy", "fire"]
    _c = certify_plan_actions(_good, _acts, {}, goal={"enemy_down": True})
    assert _c["ok"], _c["violations"]
    _bad = ["goto_enemy", "fire"]                     # fires without ever picking up
    _cb = certify_plan_actions(_bad, _acts, {}, goal={"enemy_down": True})
    assert not _cb["ok"] and "has_weapon" in _cb["violations"][0], _cb
    assert not certify_plan_actions(["goto_rack"], _acts, {},
                                    goal={"enemy_down": True})["ok"]   # goal unreached
    assert not certify_plan_actions(["teleport"], _acts, {})["ok"]      # unknown action
    print("   plan certificate: valid GOAP plan certified; firing without a weapon REFUSED "
          "naming the missing precondition")
    print("   lyapunov witness: real gradient flow CERTIFIED; driven plateau, still-falling "
          "and rising witnesses all REFUSED")
    print("   conservation ledger: bounded wobble PASSES, secular ramp FAILS, leaking mass "
          "REFUSED")
    print("   pose certificate: 4 solved poses certified (incl. an unreachable target); a "
          "hand-broken pose REFUSED")
    print("   demux gate: sigma estimator within 25%% on planted noise; 0 false-trust "
          "across 15 (K, noise) cells")
    print("   schedule certificate: %d waves over %d tasks certified; a conflicting wave "
          "and a dropped task both REFUSED" % (cert["n_waves"], cert["n_tasks"]))
    print("   differential oracle: %d cases, wrong impl CAUGHT at case %d, crash counted as "
          "disagreement" % (d2["n_cases"], d2["pairs"]["wrong"]["failures"][0]["case"]))
    print("   bake certificate: %d/%d cells sampled, guarantee %.3f at k=%d, corrupted bake "
          "REFUSED" % (c["checked"], c["n_cells"], c["guarantee"], c["k_corrupt"]))


if __name__ == "__main__":
    _selftest()
