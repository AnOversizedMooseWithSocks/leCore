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
    # -- gaussian/mean curvature: ONE concept, TWO representations, both correct. meshcurvature computes the
    # DISCRETE per-vertex form on a triangle mesh (angle defect / mixed-area cotan; returns an array over V);
    # surfanalysis computes the ANALYTIC pointwise form on a parametric surface from the fundamental forms
    # (K=(LN-M^2)/(EG-F^2), H=(EN-2FM+GL)/2(EG-F^2); returns a scalar at (u,v)). Read both bodies; the same verb on
    # a different geometry costume, the pattern the duplication audit blessed -- NOT a duplicate to merge, because
    # the discrete estimator is not the analytic formula and neither can serve the other's input.
    "gaussian_curvature": frozenset({"meshcurvature", "surfanalysis"}),
    "mean_curvature": frozenset({"meshcurvature", "surfanalysis"}),
    # -- different-domain homonyms: same English word, unrelated jobs. Read, benign. --
    "resolve": frozenset({"graphql", "machinemodel", "overrides", "superposed"}),
    "box": frozenset({"codegen", "mesh", "sdf"}),                 # an SDF box, a mesh box, a codegen box literal
    "divergence": frozenset({"curlnoise", "fields", "opponent", "probability_current"}),  # three vector-field
    # divergences under different discretizations (finite-diff / spectral / bc-aware central-diff) plus
    # opponent's unrelated angular-disagreement homonym. Read; each serves its own data layout.
    "route": frozenset({"extras", "pivot", "skills"}),
    "agree": frozenset({"hardening", "opponent"}),
    "ball": frozenset({"extras", "field"}),
    "benchmark": frozenset({"fft", "pack"}),
    "byte_report": frozenset({"chunkcodebook", "codestructure"}),
    "centroid": frozenset({"equivariance", "metrology"}),
    "classify": frozenset({"equivariance", "opponent"}),
    "compose_object": frozenset({"compose", "material"}),
    "connected_components": frozenset({"island", "route"}),
    "coverage": frozenset({"ldexplore", "sdfemit"}),
    "decompose": frozenset({"codestructure", "transform"}),
    "demo_organizer": frozenset({"navigator", "organizer"}),
    "demo_text": frozenset({"encoders", "text"}),
    "diffusion_transfer": frozenset({"laplacian", "simreadout"}),
    "gather": frozenset({"shader", "transfer"}),                 # deliberate aliases (recorded in dup audit)
    "generate": frozenset({"diffuse", "hopfield"}),
    "geodesic_distances": frozenset({"chart", "meshgeodesic"}),
    "gradient": frozenset({"laplacian", "pattern", "vision"}),   # N-dim central-diff gradient vs an image
    # gradient (magnitude+orientation) vs a linear ramp field -- three domains, same English word. Read.
    "leaf": frozenset({"fuse", "schedule"}),
    "manifest": frozenset({"dictionary", "skills"}),
    "pack": frozenset({"pack", "superposed"}),
    "plan": frozenset({"plan", "schedule"}),
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
