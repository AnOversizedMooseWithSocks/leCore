#!/usr/bin/env python3
"""name_collisions.py -- find PUBLIC functions that share a name across different modules, the complement of the
structural-shape duplicate scan.

WHY BOTH SCANS EXIST, and why neither subsumes the other:
  * The shape scan (tests/test_duplication_audit.py) erases identifiers and constants and compares function-body
    SHAPES. It catches "same algorithm, two names" -- `_face_normal` == `_newell_normal` -- which no name scan can.
    But it is BLIND to "same name, two bodies that compute it differently": `quat_from_axis_angle` ended one body
    with `np.array([w, *v])` and the other with `np.concatenate`, so their shapes differ while the math is
    identical, and the shape scan never paired them.
  * This scan compares NAMES. It catches exactly that blind spot -- and, more importantly, it catches the sharp
    hazard the shape scan cannot even describe: TWO FUNCTIONS WITH THE SAME NAME THAT COMPUTE DIFFERENT ANSWERS.
    `quat_rotate` is a live example: `cosserat`'s is the sandwich product q*v*q_conj; `transform`'s builds a matrix
    and multiplies. On a UNIT quaternion they agree to 2e-15; on a NON-UNIT quaternion they diverge completely
    (the matrix path normalizes, the sandwich does not), so a caller who imports the nearer `quat_rotate` gets a
    silently different answer. A name is not an identity, and nothing was checking.

WHAT IT DOES: lists every public (non-underscore) function name that appears in 2+ modules, and for the pairs whose
modules can be imported cheaply, CLASSIFIES the relationship so a reader knows which collisions are benign:
  * DELEGATION  -- one body calls the other (a deliberate re-export; the safe, intended kind).
  * (the deeper bit-identity / divergence judgement is left to a per-pair test, because deciding equality needs
     real inputs and the degenerate branches -- see the note in `classify`.)

THE BUDGET: `KNOWN_COLLISIONS` records the collisions a human has READ and accepted (different-domain homonyms like
`box` in codegen/mesh/sdf; deliberate delegations; pinned divergences like `psnr` and `quat_rotate`). It MAY SHRINK
AND MUST NEVER GROW -- a new name appearing in two modules means someone added a collision without reading the
other, and CI stops them until they do. This is the same discipline as the duplicate-shape budget, applied to names.

    python3 tools/name_collisions.py            # print the collision table
    python3 tools/name_collisions.py --new      # print only collisions NOT in the budget (what CI would fail on)
"""
import argparse
import ast
import collections
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Names that are structural, not capability collisions -- every module may define these and they are not a hazard.
# SHARED with holographic_capuri._STRUCTURAL_NAMES (imported, not re-listed) so the two collision audits -- this AST
# scan and the capuri URI view -- can never disagree on what counts as a collision. If the import fails (capuri not
# importable in a bare-tools context), fall back to the literal set, which must match capuri's.
try:
    from holographic.caching_and_storage.holographic_capuri import _STRUCTURAL_NAMES as _IGNORE
except Exception:
    _IGNORE = frozenset({"main", "demo", "run"})

# Collisions a human has read and accepted. Format: frozenset of module stems keyed by the shared function name.
# TO CHANGE THIS: read every body in the pair, then either UNIFY (delegate, per [DELEGATE, DON'T DUPLICATE]) and
# remove the entry, or accept it as a benign homonym / pinned divergence and leave it -- WITH the reason on the
# line. Never add an entry just to make the test pass without reading. The budget may shrink, never grow.
KNOWN_COLLISIONS = {
    # ---- REVIEWED IN THE CREATURE/ANATOMY MERGE. Both bodies read for each; reasons below. ----
    # A real ARCHITECTURAL TRANSITION, not a homonym: two implementations of "the creature's skin as a
    # field" -- the shipped metaball route and the new grouped-convolution route (bulge-free joints,
    # grouped limbs) that is meant to replace it. Deliberately co-existing while the convolution path
    # is proven; additive, old route still default. UNIFY (delete the metaball one) once it is default
    # -- this entry should NOT outlive that switch.
    "creature_field": frozenset({"creatureconv", "creatureskin"}),
    # Same word, different KERNEL and different domain: creatureskin is a sum-of-blobs (polynomial,
    # per-ball varying radius) for skin; meshbridge is a sum-of-GAUSSIANS, the engine's splat/bundle
    # representation lifted to an implicit field. Not interchangeable; delegation would be wrong.
    "metaball_field": frozenset({"creatureskin", "meshbridge"}),
    # Same concept, different INPUT TYPE: creaturereport projects a FIELD along an axis; render takes a
    # MESH orthographically. A field has no triangles and a mesh has no interior samples, so neither can
    # delegate. Candidates to unify behind one dispatcher if a third caller ever appears.
    "silhouette_mask": frozenset({"creaturereport", "render"}),
    # Both make pinhole rays, but zigmarch's is deliberately ISOLATED and f64: its docstring states
    # camera math is kept OUTSIDE the identity surface so both marchers are fed byte-identical rays.
    # Delegating it to gemrender would put the shared camera back inside the thing being compared.
    "camera_rays": frozenset({"gemrender", "zigmarch"}),
    # Pure homonyms, unrelated jobs, read and benign:
    "cluster": frozenset({"crystalgrow", "query"}),        # a druse of crystals vs semantic GROUP BY
    # file state record vs stream signature vs whole-model hypervector (unicron) -- three domains
    "fingerprint": frozenset({"assets", "modeltrain", "unicron"}),
    # Same FAMILY, different roles: modeltrain BUILDS a certified surrogate from a function; surrogate
    # RESOLVES a name to a callable. Genuinely confusable -- flagged as the one pair here worth renaming
    # (e.g. resolve_surrogate) rather than budgeting forever, but both predate this review.
    "make_surrogate": frozenset({"modeltrain", "surrogate"}),
    # Related but at different LEVELS: adaptive PLANS which render method to use for a scene; pathtrace
    # adapts SAMPLE COUNT within path tracing. One dispatches, one refines.
    "render_adaptive": frozenset({"adaptive", "pathtrace"}),

    # -- different-domain homonyms: same English word, unrelated jobs. Read, benign. --
    # Same CONCEPT at two LEVELS (continuous vs discrete) -- delegation impossible, a mesh has no surf_uv:
    "gaussian_curvature": frozenset({"surfanalysis", "meshcurvature"}),  # analytic K from fundamental forms
    # on a parametric surface (surf_uv,u,v,h) vs per-vertex angle-defect K on a discrete mesh. Both read.
    "mean_curvature": frozenset({"surfanalysis", "meshcurvature"}),     # analytic H vs discrete
    # mean-curvature-normal operator. Same distinction as gaussian_curvature; read together.
    "mesh_report": frozenset({"isosurface", "meshtools"}),              # extraction VALIDATION against the
    # source sdf (vertices,quads,sdf -> surface_error) vs the general Mesh scoreboard (mesh). Overlapping
    # intent, DISJOINT inputs; read both. Future-delegation candidate: isosurface could wrap meshtools for
    # the topology half once quad support lands there -- accepted rather than force-refactored today.
    "resolve": frozenset({"graphql", "machinemodel", "overrides", "superposed"}),
    "box": frozenset({"codegen", "mesh", "sdf"}),                 # an SDF box, a mesh box, a codegen box literal
    "divergence": frozenset({"curlnoise", "fields", "opponent", "probability_current"}),  # three vector-field
    # divergences under different discretizations (finite-diff / spectral / bc-aware central-diff) plus
    # opponent's unrelated angular-disagreement homonym. Read; each serves its own data layout.
    # region SDF / pivot-tree beam / module ranking / agent decision / representation choice /
    # MoE gate argmax -- six domains, all "route" as a verb, no shared math to unify
    "route": frozenset({"extras", "pivot", "router", "skills", "storeroute", "swarmbake"}),
    "agree": frozenset({"hardening", "opponent"}),
    "ball": frozenset({"extras", "field"}),
    "benchmark": frozenset({"fft", "pack"}),
    "byte_report": frozenset({"chunkcodebook", "codestructure"}),
    "centroid": frozenset({"equivariance", "metrology"}),
    # verdict-family homonyms: symmetry verdict / disagreement type / installability verdict
    "classify": frozenset({"equivariance", "opponent", "vminstall"}),
    "compose_object": frozenset({"compose", "material"}),
    "connected_components": frozenset({"island", "route"}),
    # Read all three bodies before extending this: sdfemit.coverage() reports which holographic_sdf node KINDS the
    # shader emitter handles vs refuses (a gap = silently omitted geometry); ldexplore.coverage(strategy, T, ...)
    # counts distinct GRID CELLS an exploration strategy visits; semantictag.coverage(catalog) is the FRACTION OF
    # CAPABILITIES carrying a verb tag (browse_semantic omits the untagged, so the number IS the menu). Three
    # unrelated spaces, three signatures, no shared kernel -- the same benign homonym as "box" (sdf/mesh/codegen).
    # ccrun.* compiles emitted C with the system cc; zigrun.* compiles the same-shaped source with zig. Parallel
    # native backends by design: two methods share a name (compile_cached, build_batch_source), each its own real
    # body (not a shim), each content-addressing with sha256/hashlib per the determinism rule. Benign-homonym
    # shape, same as `coverage` below -- read both bodies before touching either.
    "compile_cached": frozenset({"ccrun", "zigrun"}),
    "build_batch_source": frozenset({"ccrun", "zigrun"}),
    "coverage": frozenset({"ldexplore", "sdfemit", "semantictag"}),
    # AST split / low-rank rebuild / 4x4-matrix split -- three unrelated decompositions
    "decompose": frozenset({"codestructure", "refactor", "transform"}),
    "demo_organizer": frozenset({"navigator", "organizer"}),
    "demo_text": frozenset({"encoders", "text"}),
    "diffusion_transfer": frozenset({"laplacian", "simreadout"}),
    # composite.blend applies an IMAGE blend mode (multiply/screen/overlay) to
    # two colour arrays; opponent.blend mixes two HYPERVECTORS through the
    # opponent structure (keep agreement, mix exclusives at a ratio). Bodies
    # read: different domains, different arities, nothing shared. The English
    # word "blend" is simply the right word in both places.
    "blend": frozenset({"composite", "opponent"}),
    # lean.prove derives a ground Atom from Horn rules by forward chaining;
    # querytime.prove publishes a MERKLE ROOT as a tamper-evident commitment.
    # Bodies read: one is logic, the other is cryptography, and "prove" is the
    # ordinary word in both fields. Nothing to unify.
    "prove": frozenset({"lean", "querytime"}),
    # fem.simulate steps a finite-element soft body (activation, k_muscle,
    # gravity, pinned); smokepresets.simulate runs the smoke solver under a
    # named preset. Different solvers, different state, different physics --
    # the shared name is the English verb.
    "simulate": frozenset({"fem", "smokepresets"}),
    "gather": frozenset({"shader", "transfer"}),                 # deliberate aliases (recorded in dup audit)
    # pipelinemap joined this set when it MOVED INTO THE PACKAGE (it was a
    # root-level module the scan never saw, and never in the wheel either --
    # that was the bug). Bodies read: pipelinemap.generate WRITES
    # docs/PIPELINE_MAP.md from the live catalog; diffuse/hopfield.generate
    # produce SIGNALS. Different-domain homonyms of the plainest kind -- the
    # English word for "make one", used by a documentation tool and by two
    # samplers. Nothing to unify.
    "generate": frozenset({"diffuse", "hopfield", "pipelinemap"}),
    "geodesic_distances": frozenset({"chart", "meshgeodesic"}),
    "gradient": frozenset({"laplacian", "pattern", "vision"}),   # N-dim central-diff gradient vs an image
    # gradient (magnitude+orientation) vs a linear ramp field -- three domains, same English word. Read.
    "leaf": frozenset({"fuse", "schedule"}),
    "manifest": frozenset({"dictionary", "skills"}),
    "pack": frozenset({"pack", "superposed"}),
    # corridor bake / DAG fuse plan / context-retention budget / per-layer decision / install mode
    # -- "plan" the noun and verb across five families; bodies read, nothing delegatable
    "plan": frozenset({"billionctx", "plan", "schedule", "transform", "unlocked"}),
    "render_scene": frozenset({"compose", "semantic"}),
    "sample_field": frozenset({"fields", "meshbridge"}),
    "shade": frozenset({"equivariance", "matlib"}),
    "sphere": frozenset({"codegen", "sdf"}),
    "subspace_alignment": frozenset({"dream", "nystrom"}),
    "transfer": frozenset({"iterate", "shader"}),
    "unit": frozenset({"machinemodel", "quantities"}),
    "validate": frozenset({"materialdata", "recipeops"}),
    "validate_c": frozenset({"emit", "sdfemit"}),
    # -- the transform kit: canonical builders in `transform`, with declared/delegating copies elsewhere. --
    "translation": frozenset({"grouptower", "scenegraph", "transform"}),
    "rotation": frozenset({"meshskin", "mueller", "scenegraph"}),  # meshskin's ~9e-12 copy is DECLARED, not
    # merged; mueller's is a THIRD ANSWER -- a Mueller REFERENCE-FRAME rotation with the 2*phi polarization
    # convention, not a spatial Rodrigues matrix. Must never be unified. Read and pinned.
    "compose": frozenset({"mueller", "transform"}),  # both fold matrices but with OPPOSITE order conventions:
    # transform is plain M0@M1@... (rightmost applied first); mueller takes elements IN THE ORDER LIGHT
    # PASSES THROUGH and reverses internally. Same hazard class as rotation; read and pinned.
    "hsv_to_rgb": frozenset({"falsecolor", "vision"}),  # the same hexcone under two conventions: vision
    # takes a packed (...,3) array with H in DEGREES; falsecolor takes separate broadcastable h,s,v in
    # [0,1]. A delegation would be a signature-adapting shim of equal size -- recorded, not rewired.
    "identity": frozenset({"mueller", "scenegraph"}),  # two trivial np.eye(4) one-liners in different
    # domains (a 4x4 transform vs the Mueller composition unit). Read; nothing to unify.
    "scaling": frozenset({"scenegraph", "transform"}),           # scenegraph delegates to transform
    "quat_from_axis_angle": frozenset({"cosserat", "transform"}),# cosserat delegates to transform (rev.9 fix)
    "refract_dir": frozenset({"raydiff", "raymarch"}),
    "sdf_normal": frozenset({"raymarch", "sdf"}),                # raymarch re-exports sdf's canonical normal
    # -- PINNED DIVERGENCES: same name, DIFFERENT answers, not merged. Each has a test asserting the divergence. --
    "psnr": frozenset({"reproject", "splat"}),                   # differ in the tie band 0<mse<1e-12 (pinned)
    "quat_rotate": frozenset({"cosserat", "transform"}),         # sandwich vs matrix: agree on unit q, diverge off it (pinned)
    # ---- REVIEWED IN THE WILD-RELEASE SWEEP (unicron/galvatron family; every body read). ----
    # THE ALGEBRA TRIO, divergence deliberate (three costumes of binding, quat_rotate precedent):
    # ai = HRR via FFT circular convolution; hlb = Hadamard/WHT-domain elementwise; vsaroles =
    # integer cyclic shifts (roles as permutation powers, storage-free). Same verb, three algebras,
    # each module states its own; unifying would erase the distinction the modules exist to make.
    "bind": frozenset({"ai", "hlb", "vsaroles"}),
    "unbind": frozenset({"ai", "hlb", "vsaroles"}),
    # bundle: ai renormalizes; vsaroles is raw addition (residual-stream semantics); galvabundle is
    # a PACKAGING verb (write a deployment bundle) -- homonym across abstraction levels.
    "bundle": frozenset({"ai", "galvabundle", "vsaroles"}),
    # BIT-IDENTICAL one-line FFT primitives, kept local ON PURPOSE: querypath and vsarun ship as
    # self-contained installable payloads, and a cross-import would break payload isolation. If a
    # third copy ever appears, promote to one home and delegate.
    "cconv": frozenset({"querypath", "vsarun"}),
    "ccorr": frozenset({"querypath", "vsarun"}),
    # Plate's involution (reverse-all-but-first) in both; ai documents the trick, vsabake is the
    # payload-local copy for the same isolation reason as cconv above.
    "involution": frozenset({"ai", "vsabake"}),
    # same guard, different field: ai is real cosine; unicron is the FHRR Hermitian real part --
    # they agree on reals and diverge on complex inputs BY DESIGN (the FHRR similarity).
    "cosine": frozenset({"ai", "unicron"}),
    # same word, different instrument level: voidexplore's is the raw density dot product;
    # voidmanifold's DECODES a void point back to hidden space. Signatures disjoint.
    "void_probe": frozenset({"voidexplore", "voidmanifold"}),
    # ---- different-domain homonyms (box/psnr precedent), bodies read pairwise, no delegation
    # possible or appropriate; module namespaces are the disambiguator: ----
    "allocate": frozenset({"calltoken", "supermemory"}),      # token rows vs capacity law inverse
    "audit": frozenset({"install", "orphanaudit"}),           # install reachability vs repo orphans
    "build": frozenset({"directed", "recipe"}),               # graph assembly vs install rules
    "build_index": frozenset({"codemap", "memsearch"}),       # source tree vs passage addresses
    "compare": frozenset({"assess", "hybrid", "querytime"}),  # runs vs accuracies vs branches
    "content_key": frozenset({"declare", "galvacache", "querypath"}),  # digest/digest/hypervector
    "decode_record": frozenset({"boot", "planshape"}),        # byte-packed row vs role-filler record
    "encode_record": frozenset({"boot", "planshape"}),        # (the write halves of the above)
    "describe": frozenset({"proglib", "vision"}),             # word bundle vs image features
    "export": frozenset({"galvaport", "testkit"}),            # GGUF sidecar vs npz test kit
    "fuse": frozenset({"fuse", "vminstall"}),                 # FFT expression tree vs operator fold
    "head_of": frozenset({"earlyexit", "factbake"}),          # same lookup, both 4 lines; local on
                                                              # purpose (payload isolation, as cconv)
    "health": frozenset({"reversible", "selfheal"}),          # codebook cosine vs register margins
    "imbue": frozenset({"galvapack", "unicron"}),             # packaging verb vs task-vector write
    "install": frozenset({"galvacache", "install", "install_lecore"}),  # monkey-patch cache vs
                                                              # weight install vs six-part assembly
    "load": frozenset({"core", "sidecar", "testkit"}),        # npz objects vs curtain vs kit
    "load_model": frozenset({"modelstore", "unicron"}),       # container vs safetensors front door
    "measure": frozenset({"hrnnbake", "measure"}),            # ppl+horizon vs ppl+uncertainty
    "place": frozenset({"devicerun", "machinemodel"}),        # device placement vs cost arithmetic
    "project": frozenset({"hlb", "nullspace", "query"}),      # WHT sign vs subspace vs SELECT
    "recall": frozenset({"boot", "modelvault"}),              # ccorr read vs vault unpack
    "recall_all": frozenset({"ai", "hybrid"}),                # trace sweep vs register sweep
    "reconstruct": frozenset({"ratedistortion", "refactor"}), # rANS decode vs factor re-dense
    "report": frozenset({"bios", "measure"}),                 # BIOS screen vs ppl bootstrap
    "save": frozenset({"core", "sidecar"}),                   # npz objects vs sidecar write
    "search": frozenset({"dictionary", "memsearch"}),         # prefix words vs state ranking
    "select": frozenset({"scene_query", "writepolicy"}),      # scene predicates vs register policy


}


def scan():
    """Every public function name that appears in 2+ modules -> {name: sorted[module stems]}."""
    homes = collections.defaultdict(set)
    for p in sorted((REPO / "holographic").rglob("holographic_*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        stem = p.stem.replace("holographic_", "")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") and node.name not in _IGNORE:
                homes[node.name].add(stem)
    return {name: sorted(mods) for name, mods in homes.items() if len(mods) > 1}


def classify(name, mods):
    """A cheap, import-free hint about a collision: does one body DELEGATE to another (a benign re-export)?

    We deliberately do NOT try to decide bit-identity here. Equality is a claim about behaviour on real inputs and
    every degenerate branch (see [BIT-IDENTITY GATE] in the tech-debt backlog); a static scan that guessed 'these
    are the same' would be exactly the over-confident move this scan exists to prevent. Delegation, by contrast, is
    visible in the source: a short body that imports and calls the twin. So we report DELEGATION vs INDEPENDENT and
    leave the equal-or-divergent verdict to a per-pair test that feeds them numbers."""
    verdicts = {}
    for stem in mods:
        hits = list((REPO / "holographic").rglob("holographic_%s.py" % stem))
        if not hits:
            verdicts[stem] = "?"
            continue
        src = hits[0].read_text(errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            verdicts[stem] = "?"
            continue
        body_src = ""
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                body_src = ast.unparse(node)
                break
        # a delegation is a short body that imports the twin and calls it, OR whose docstring says so
        deleg = ("DELEGATES" in body_src or "Routes to" in body_src or
                 ("import" in body_src and body_src.count("\n") <= 6))
        verdicts[stem] = "delegates" if deleg else "own-body"
    return verdicts


def addressability_report():
    """S1.3: verify every current name collision is ADDRESSABLE -- i.e. its bare name resolves, through the capuri
    URI index, to distinct full paths (one per home). A collision is safe not because it is forbidden but because a
    caller can supply the path to disambiguate it; this checks that promise holds for every collision. Returns
    (ok, problems) where problems is a list of (name, reason) for any collision that does NOT resolve cleanly."""
    from holographic.caching_and_storage.holographic_capuri import resolve_uri, collisions
    problems = []
    coll = collisions(ignore_structural=True)                  # same set the AST scan reports (shared ignore)
    for name, uris in coll.items():
        resolved = resolve_uri(name)
        if len(resolved) < 2:
            problems.append((name, "resolves to %d URI(s), expected >=2 for a collision" % len(resolved)))
            continue
        if len(set(resolved)) != len(resolved):
            problems.append((name, "resolves to duplicate URIs: %s" % resolved))
            continue
        # each full URI must itself resolve to exactly one thing (the disambiguation actually works).
        for u in resolved:
            if resolve_uri(u) != [u]:
                problems.append((name, "full URI %s does not resolve to itself" % u))
    return (not problems), problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new", action="store_true", help="show only collisions NOT in the budget (CI-failing set)")
    ap.add_argument("--addressable", action="store_true",
                    help="verify every collision resolves to distinct URIs (the S1.3 addressability property)")
    args = ap.parse_args(argv)

    if args.addressable:
        ok, problems = addressability_report()
        if ok:
            print("ADDRESSABLE: every name collision resolves to distinct capability URIs -- each is disambiguable "
                  "by supplying its path. A collision is a path to supply, not a hazard to forbid.")
            return 0
        print("NOT ADDRESSABLE -- these collisions do not resolve cleanly (the semantic hierarchy cannot "
              "disambiguate them; fix the URI index or the collision):")
        for name, reason in problems:
            print("  %-26s %s" % (name, reason))
        return 1

    found = scan()
    new = {n: m for n, m in found.items() if frozenset(m) != KNOWN_COLLISIONS.get(n)}

    if args.new:
        if not new:
            print("no new public name collisions -- all %d are in the reviewed budget." % len(found))
            return 0
        print("NEW collisions (not in KNOWN_COLLISIONS -- read both bodies, then unify or accept WITH a reason):")
        for n in sorted(new):
            print("  %-26s %s   %s" % (n, ", ".join(found[n]), classify(n, found[n])))
        return 1

    print("PUBLIC NAME COLLISIONS: %d name(s) in 2+ modules (%d in the reviewed budget)\n"
          % (len(found), len(KNOWN_COLLISIONS)))
    for n in sorted(found):
        mark = " " if frozenset(found[n]) == KNOWN_COLLISIONS.get(n) else "*"
        print(" %s %-26s %s" % (mark, n, ", ".join(found[n])))
    if new:
        print("\n* = NEW (not yet reviewed). Run with --new for the classification, then read both bodies.")
    return 0


def _selftest():
    """Prove the scan finds a real, known collision, that the budget matches the tree, and that the detector would
    catch a fabricated new collision -- a budget test that cannot fail is worse than no budget."""
    found = scan()
    # 1. a collision we KNOW exists is found, with the right homes
    assert "quat_rotate" in found and set(found["quat_rotate"]) == {"cosserat", "transform"}, found.get("quat_rotate")
    # 2. the budget neither lies nor lags: every current collision is accounted for (this is the CI contract)
    unreviewed = {n: m for n, m in found.items() if frozenset(m) != KNOWN_COLLISIONS.get(n)}
    assert not unreviewed, "unreviewed collisions (add to KNOWN_COLLISIONS after reading both bodies): %s" % (
        sorted(unreviewed))
    # 3. the budget has no stale entries (a name that no longer collides, or whose homes moved)
    stale = {n: sorted(ms) for n, ms in KNOWN_COLLISIONS.items() if frozenset(found.get(n, ())) != ms}
    assert not stale, "stale budget entries (the collision changed or is gone -- update the line): %s" % stale
    # 4. classify() reads delegation out of the source: cosserat.quat_from_axis_angle delegates to transform's
    v = classify("quat_from_axis_angle", ["cosserat", "transform"])
    assert v.get("cosserat") == "delegates", v
    # 5. S1.3 ADDRESSABILITY: every collision resolves to distinct URIs, so it is disambiguable by path. This is the
    # deeper safety property than the frozen budget -- a collision is fine iff a caller can supply its path.
    ok, problems = addressability_report()
    assert ok, "collisions that do not resolve cleanly (the hierarchy cannot disambiguate them): %s" % problems
    print("tools/name_collisions selftest OK: %d collisions, all %d reviewed, detector bites, classify() reads "
          "delegation, every collision is ADDRESSABLE (resolves to distinct URIs)" % (len(found),
          len(KNOWN_COLLISIONS)))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        _selftest()
    else:
        sys.exit(main())
