"""CREATURE READABILITY AS A SEARCH -- proportion, negative space, line of action, standing.

Backlog Tier 8 (A-1..A-4). The panel's call, and the reason this is a search rather than a rule table:

    Togelius, Yannakakis, Stanley & Browne 2011, "Search-Based Procedural Content Generation"

Their taxonomy is explicit that for CONTENT QUALITY you define an evaluation function and search it,
rather than hand-coding rules -- because hand-coded rules encode one person's taste at one moment and
cannot be checked. We already own the evaluation function (`silhouette_report`'s negative space, the
M-3 gate), so A-1 and A-2 become "search the spec against the metric we already trust", and no second
opinion about what reads well enters the codebase.

THE DEGENERATE OPTIMUM, MEASURED FIRST, WHICH IS WHY THE SCORE HAS MORE THAN ONE TERM. Negative space
alone is MONOTONE DECREASING in limb thickness -- measured across limb radius 0.03 -> 0.12, negative
space falls 0.470 -> 0.332 without a single local optimum. A search maximising it would drive the
limbs to zero and call a spider-legged wisp the most readable creature possible. That is not a flaw
in the metric; it is what a single-objective search does to any metric.

So the score pairs it with A-1's actual claim -- ONE DOMINANT MASS, clearly subordinate limbs -- as a
measured quantity: the share of body volume owned by the spine. That is also monotone (0.816 at limb
radius 0.03, 0.515 at 0.09) but in the OPPOSITE direction, so the two together have an interior
optimum and the search has something real to find.

KEPT NEGATIVE, LOUD: none of this reaches art direction. Spore had artists authoring parts and an art
director enforcing proportion; a two-term score is a weaker substitute and cannot tell an appealing
creature from a merely well-proportioned one. What it CAN do is refuse the obviously unreadable, and
say why with a number.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_rig import rig_of
from holographic.mesh_and_geometry.holographic_creaturetree import creature_tree_grouped
from holographic.mesh_and_geometry.holographic_creaturereport import silhouette_report, webbing_report


def mass_dominance(source, field=None, samples=6000, seed=0):
    """A-1's measurable core: what SHARE of the body's volume belongs to the spine (the dominant mass)?

    Ownership is nearest-bone-axis, the same rule `tissue_weights` uses, so this cannot disagree with
    the skinning about which part a point belongs to. A creature whose limbs have similar visual
    weight to its torso -- the readability failure the backlog names -- scores near 0.5; a clear
    big-shape/small-shape hierarchy scores high.
    """
    rig = source if hasattr(source, "tags") else rig_of(source)
    if field is None:
        field = creature_tree_grouped(rig)
    lo, hi = rig.extent()
    pad = 0.1 * rig.reference_length()
    P = np.random.default_rng(int(seed)).uniform(lo - pad, hi + pad, size=(int(samples), 3))
    inside = np.asarray(field(P), float).ravel() < 0.0
    if not inside.any():
        return 0.0
    Q = P[inside]
    D = np.empty((len(Q), len(rig.tags)))
    for j, t in enumerate(rig.tags):
        a, b = rig.segment(t)
        ab = b - a
        u = np.clip(((Q - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        D[:, j] = np.linalg.norm(Q - (a + u[:, None] * ab), axis=1)
    tags = np.array(rig.tags)
    return float(np.mean(np.char.startswith(tags[np.argmin(D, axis=1)].astype(str), "spine")))


def head_definition(source, field=None, samples=400, reach=0.6):
    """DOES THE HEAD READ AS A HEAD? -- the thickness profile along the spine, and whether there is a
    NECK between torso and head.

    Returns {'profile', 'head_ratio', 'neck_ratio', 'has_neck'}. `head_ratio` is the head's local
    half-thickness over the torso's median; `neck_ratio` is the profile's MINIMUM between the torso
    and the head over that same median -- below 1.0 means the body actually narrows before the head.

    FOUND BY LOOKING, NOT BY SCORING (the dogfooding lesson, applied again). Every readability number
    was green while the shipped quadruped's profile read as "a tube that gets fatter at one end".
    Measured along its spine, half-thickness runs 0.104, 0.114, 0.113, 0.114, 0.163 -- the head IS
    1.43x the body, but the profile is MONOTONE, so there is no neck and nothing separates the head
    mass from the torso mass. A big-shape/medium-shape hierarchy (backlog A-1) needs the separation,
    not just the size: an animal reads as headed because the silhouette PINCHES first.

    This measures the gap; it does not fix it. Fixing means a spine thickness profile that dips before
    the head, which is a spec/authoring change (`spine_profile` already exists to express it).
    """
    rig = source if hasattr(source, "tags") else rig_of(source)
    src = getattr(rig, "source", None)
    if src is None or not hasattr(src, "spine_nodes"):
        raise ValueError("head definition needs a spined creature")
    if field is None:
        field = creature_tree_grouped(rig)
    ts = np.linspace(0.0, float(reach), int(samples))
    # PROBE PERPENDICULAR TO THE SPINE, not along world +y. Measuring a body's thickness along a
    # fixed world axis reads the wrong direction the moment the body is not lying the way the author
    # assumed -- for a vertical spine it probes ALONG the body and reports the length as a thickness.
    # The perpendicular comes from the creature's own sagittal normal crossed with its spine axis,
    # which is the same body frame `dir_space` and the mirror plane use. For the default axis this is
    # exactly world +y, so no existing measurement moves.
    axis = np.asarray(getattr(src, "spine_axis", (0.0, 0.0, 1.0)), float)
    sag = np.asarray(getattr(src, "sagittal_normal", (1.0, 0.0, 0.0)), float)
    up = np.cross(axis, sag)
    n_up = float(np.linalg.norm(up))
    up = np.array([0.0, 1.0, 0.0]) if n_up < 1e-9 else up / n_up
    prof = []
    for n in src.spine_nodes:
        P = np.asarray(rig.joints[n], float)[None, :] + ts[:, None] * up[None, :]
        v = np.asarray(field(P), float).ravel()
        prof.append(float(ts[int(np.argmax(v >= 0))]) if (v >= 0).any() else float(reach))
    prof = np.asarray(prof, float)
    torso = float(np.median(prof[:-1])) or 1e-9
    head = float(prof[-1])
    neck = float(prof[1:-1].min()) if len(prof) > 2 else torso
    return {"profile": prof.tolist(), "head_ratio": head / torso, "neck_ratio": neck / torso,
            "has_neck": bool(neck < 0.92 * torso)}


def readability_score(source, field=None, res=64, samples=4000, seed=0,
                      dominance_target=0.70, dominance_weight=1.0):
    """THE EVALUATION FUNCTION the proportion search maximises (A-1 + A-2).

    Returns {'score', 'negative_space', 'dominance', 'webbing_pairs', 'feasible', 'why'}.

        negative space   the M-3 gate, already trusted, already used to judge the field rebuild
        dominance        spine share of body volume -- A-1's one-dominant-mass claim, measured
        webbing          A-2's "reject configurations whose blend regions overlap", as a HARD gate

    The dominance term is a DISTANCE FROM A TARGET, not a maximisation: pushing dominance up is just
    as degenerate as pushing negative space up (it ends in a sausage with no limbs at all). A band
    around a target is the only honest shape for a proportion objective.

    Webbing is a FEASIBILITY GATE rather than a penalty term, because a webbed creature is not a
    worse creature, it is a broken one -- and mixing a correctness failure into a quality score lets
    the search buy its way out of it with prettier proportions.
    """
    rig = source if hasattr(source, "tags") else rig_of(source)
    if field is None:
        field = creature_tree_grouped(rig)
    ns = float(silhouette_report(rig, field=field, res=int(res))["negative_space"])
    dom = mass_dominance(rig, field=field, samples=int(samples), seed=int(seed))
    web = int(webbing_report(rig, field=field)["webbing_pairs"])
    penalty = float(dominance_weight) * abs(dom - float(dominance_target))
    feasible = web == 0
    return {"score": ns - penalty, "negative_space": ns, "dominance": dom,
            "webbing_pairs": web, "feasible": feasible,
            "why": "ok" if feasible else "webbed: blend regions overlap (A-2)"}


def ground_creature(source, field=None, res=48):
    """A-4, GENERATE STANDING: the translation that puts the body's lowest point on y = 0.

    Returns {'offset', 'lowest', 'supported'}. Measured from the FIELD, not from the joints, because
    a foot's flesh reaches below its bone and a rig-only answer floats the creature by a limb radius.

    WHY IT MATTERS AT ALL, per the backlog: a creature floating in abstract space reads as an OBJECT.
    Planting it reads as an animal. `supported` reports whether at least three ground-touching
    contacts exist (a tripod is the minimum static support), so a two-legged pose that would topple is
    reported rather than silently produced.
    """
    rig = source if hasattr(source, "tags") else rig_of(source)
    if field is None:
        field = creature_tree_grouped(rig)
    lo, hi = rig.extent()
    pad = 0.15 * rig.reference_length()
    g = np.linspace(lo[1] - pad, hi[1] + pad, int(res))
    gx = np.linspace(lo[0] - pad, hi[0] + pad, int(res))
    gz = np.linspace(lo[2] - pad, hi[2] + pad, int(res))
    X, Y, Z = np.meshgrid(gx, g, gz, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    inside = np.asarray(field(P), float).ravel() < 0.0
    if not inside.any():
        return {"offset": 0.0, "lowest": 0.0, "supported": False}
    ys = P[inside][:, 1]
    lowest = float(ys.min())
    # Contacts: distinct columns whose lowest material sits within a cell of the lowest point.
    cell = float(g[1] - g[0])
    near = P[inside][ys <= lowest + cell]
    cols = {(round(float(p[0]) / max(cell, 1e-9)), round(float(p[2]) / max(cell, 1e-9))) for p in near}
    return {"offset": -lowest, "lowest": lowest, "supported": len(cols) >= 3,
            "contacts": len(cols)}


def proportion_search(spec, knobs=None, steps=3, res=56, samples=3000, seed=0):
    """A-1/A-2/A-3 AS A SEARCH: try spec variations, keep the highest-scoring FEASIBLE one.

    `knobs` maps a spec path to the candidate values to try, e.g.
    {'limb_radius': [0.03, 0.05, 0.07], 'spine_curve': [0.0, 0.12, 0.3]}. Defaults cover A-1 (limb vs
    body thickness) and A-3 (line of action -- one clear spine curve).

    COORDINATE SEARCH, DELIBERATELY, AND HERE IS WHY THE GRADIENT OPTIMIZER WAS REFUSED: the engine
    ships `optimize` (Adam with finite differences), but this objective is a GRID-SAMPLED VOLUME
    measure -- it changes in steps as sample points cross the surface, so a finite-difference gradient
    reads mostly quantisation noise. A deterministic sweep over a declared candidate set is honest
    about what it did, reproducible, and cheap at this size. It is not a claim that gradient search
    could not work on a smoother objective.

    Returns {'best', 'best_score', 'evaluated', 'trace'} -- `trace` is every candidate with its score,
    so the search's own evidence is inspectable rather than a single winner appearing by fiat.
    """
    from holographic.mesh_and_geometry.holographic_creature import Creature

    if knobs is None:
        knobs = {"limb_radius": [0.035, 0.05, 0.065, 0.08],
                 "spine_curve": [0.0, 0.12, 0.3]}

    def build(sp, limb_radius=None, spine_curve=None):
        out = dict(sp)
        if spine_curve is not None:
            out["spine"] = dict(out["spine"])
            out["spine"]["curve"] = float(spine_curve)
        if limb_radius is not None:
            out["limbs"] = [dict(l) for l in out["limbs"]]
            for l in out["limbs"]:
                l["radius"] = float(limb_radius)
        return out

    current = dict(spec)
    chosen = {}
    trace = []
    best_score = -1e9
    for _ in range(int(steps)):
        improved = False
        for key, values in sorted(knobs.items()):
            for v in values:
                cand = build(current, **{key: v})
                rig = rig_of(Creature(cand))
                sc = readability_score(rig, res=int(res), samples=int(samples), seed=int(seed))
                trace.append({key: v, "score": sc["score"], "negative_space": sc["negative_space"],
                              "dominance": sc["dominance"], "feasible": sc["feasible"]})
                if sc["feasible"] and sc["score"] > best_score + 1e-9:
                    best_score, chosen[key], current, improved = sc["score"], v, cand, True
        if not improved:
            break                          # a coordinate sweep that changes nothing has converged
    return {"best": current, "best_score": best_score, "chosen": chosen,
            "evaluated": len(trace), "trace": trace}


def _selftest():
    from holographic.mesh_and_geometry.holographic_creature import Creature, quadruped_spec

    spec = quadruped_spec()
    rig = rig_of(Creature(spec))

    # 1) THE TWO TERMS REALLY PULL IN OPPOSITE DIRECTIONS. This is the whole justification for a
    # multi-term score, so it is measured rather than asserted in prose: thin limbs must give MORE
    # negative space and MORE dominance; thick limbs the reverse. If they ever moved together, one of
    # them would be redundant and the score would be single-objective in disguise.
    def _mk(lr):
        sp = dict(spec)
        sp["limbs"] = [dict(l) for l in sp["limbs"]]
        for l in sp["limbs"]:
            l["radius"] = float(lr)
        return rig_of(Creature(sp))

    thin, thick = _mk(0.035), _mk(0.09)
    s_thin = readability_score(thin, res=48, samples=2500)
    s_thick = readability_score(thick, res=48, samples=2500)
    assert s_thin["negative_space"] > s_thick["negative_space"], (s_thin, s_thick)
    assert s_thin["dominance"] > s_thick["dominance"], (s_thin, s_thick)

    # 2) THE DEGENERATE OPTIMUM IS REALLY REFUSED. Maximising negative space alone would pick the
    # thinnest limbs available; the dominance target must pull the winner off that extreme, or the
    # multi-term score is decoration. A spindly body scores WORSE than a moderate one overall even
    # though its negative space is higher -- that inversion is the test.
    mid = _mk(0.055)
    s_mid = readability_score(mid, res=48, samples=2500)
    assert s_mid["score"] > s_thin["score"], \
        "the score must refuse the spindly extreme: mid %.4f vs thin %.4f (ns %.3f vs %.3f)" % (
            s_mid["score"], s_thin["score"], s_mid["negative_space"], s_thin["negative_space"])

    # 3) WEBBING IS A GATE, NOT A TERM. A webbed body must be infeasible no matter how it scores --
    # a correctness failure must not be purchasable with better proportions.
    from holographic.mesh_and_geometry.holographic_creatureskin import creature_field
    webbed = readability_score(rig, field=creature_field(Creature(spec), spec), res=48, samples=2000)
    assert not webbed["feasible"] and webbed["webbing_pairs"] > 0, webbed

    # 4) A-4 STANDING: the offset must actually put the lowest material at 0, and the creature must
    # have real support. Measured from the FIELD, since a rig-only answer floats it by a limb radius.
    g = ground_creature(rig, res=40)
    assert g["offset"] > 0.0 or abs(g["lowest"]) < 1e-9
    assert g["supported"], "a quadruped must have >= 3 ground contacts, got %d" % g["contacts"]

    # 5a) HEAD DEFINITION, the gap dogfooding found that no score could see. The shipped quadruped's
    # head IS bigger than its body (1.4x) yet the profile is MONOTONE -- no neck, so the silhouette
    # reads as a tube that thickens rather than as a headed animal. Asserted as the CURRENT STATE, so
    # that when a spine profile with a neck lands, this test fails and forces the number to be updated
    # rather than letting the improvement pass unnoticed.
    hd = head_definition(rig)
    assert hd["head_ratio"] > 1.2, "the head must at least be thicker than the body: %.2f" % hd["head_ratio"]
    assert not hd["has_neck"], \
        "the shipped quadruped has NO neck (profile %r) -- if it does now, that is an improvement " \
        "and this assertion is the thing that should change" % [round(x, 3) for x in hd["profile"]]

    # 5) THE SEARCH RUNS, IS FEASIBLE, AND BEATS ITS OWN STARTING POINT -- and its trace is evidence.
    base_score = readability_score(rig, res=48, samples=2500)["score"]
    out = proportion_search(spec, res=48, samples=2500, steps=2)
    assert out["evaluated"] >= 4 and out["trace"], out
    assert out["best_score"] >= base_score - 1e-9, \
        "the search must not end below where it started: %.4f vs %.4f" % (out["best_score"], base_score)
    assert any(t["feasible"] for t in out["trace"]), "no feasible candidate was found at all"

    print("creatureproportion selftest OK: thin ns %.3f dom %.3f vs thick ns %.3f dom %.3f (terms "
          "oppose), spindly refused (mid %.4f > thin %.4f), webbed field infeasible (%d pairs), "
          "standing offset %.4f with %d contacts, search %d candidates -> %.4f (from %.4f)"
          % (s_thin["negative_space"], s_thin["dominance"], s_thick["negative_space"],
             s_thick["dominance"], s_mid["score"], s_thin["score"], webbed["webbing_pairs"],
             g["offset"], g["contacts"], out["evaluated"], out["best_score"], base_score))


if __name__ == "__main__":
    _selftest()
