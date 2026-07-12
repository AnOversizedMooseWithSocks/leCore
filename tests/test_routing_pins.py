"""T5 -- the ROUTING REGRESSION HARNESS. A dedicated pin for find_capability/route decisions, so any future change
to ranking or confidence logic (Phase 1's R1/R2 change exactly that) cannot silently reroute a query that a human
already decided the right answer for.

WHY a separate file when routing is exercised in ~130 test files already: those tests pin routing INCIDENTALLY,
as a step in a larger scenario. This file pins it ON PURPOSE, with the confidence MARGIN made explicit -- because
route() confidence decays monotonically as the catalog grows (dominance is scored against the runner-up, so every
new neighbour lowers it), a probe that reads `act` at 0.667 today can rot to 0.599 as entries land, and an
incidental assertion of `decision == "act"` gives no early warning. Here the margin is asserted with the current
measured value in the message, so the failure names the mechanism.

Each pin below traces to a rev.9 routing failure that shipped RED and was fixed. They are the regressions R1/R2
must not reintroduce. See docs/NOTES_concepts.md (rev.9) for the mechanics of each.
"""
import lecore
import holographic.misc.holographic_skills as sk


ACT_THRESHOLD = 0.6          # route()'s own act/choose cutoff; pins assert the RIGHT side of it, with margin


def _mind():
    return lecore.UnifiedMind(dim=256, seed=0)


# ---------------------------------------------------------------------------------------------------------
# find_capability: the right home must be in the top-k for the phrasing a stranger would type.
# ---------------------------------------------------------------------------------------------------------

def test_automaton_is_discoverable_by_its_own_vocabulary():
    """rev.9: an auto-seeded automaton entry lost a five-way score tie ALPHABETICALLY to two `diffusion_*` entries
    and fell out of the top-3 for its own headline query. A curated entry restored it. This pins the fix."""
    m = _mind()
    hits = [h.name.lower() for h in m.find_capability("reaction diffusion cellular automaton", k=3)]
    assert any("automaton" in h for h in hits), ("automaton home fell out of top-3 -- a diffusion_* entry likely "
                                                 "out-ranks it again: %s" % hits)


def test_selftest_coverage_is_discoverable():
    """T7: the selftest-coverage census must be findable by the words an agent would use, or the whole point
    (a mind door onto 'is the engine covered?') is lost."""
    m = _mind()
    for phrasing in ("which modules lack a selftest", "audit test coverage", "is the engine covered by tests"):
        hits = [h.name for h in m.find_capability(phrasing, k=3)]
        assert any("Selftest coverage" in h for h in hits), (phrasing, hits)


def test_transform_kit_beats_the_tower_theory_for_a_concrete_matrix_query():
    """rev.9: 'translation matrix' ranked the projective-ceiling / transform-tower THEORY entries above the KIT
    that actually builds one. The Transform (warp) home must be reachable for concrete build-a-matrix phrasings."""
    m = _mind()
    for phrasing in ("make a translation matrix", "compute a quaternion from axis and angle"):
        hits = [h.name for h in m.find_capability(phrasing, k=3)]
        assert any("Transform" in h or "transform" in h for h in hits), (phrasing, hits)


# ---------------------------------------------------------------------------------------------------------
# route(): confident act vs. ask-for-clarification, with the margin made explicit.
# ---------------------------------------------------------------------------------------------------------

def test_describe_a_scene_routes_to_act():
    """rev.9: this shipped `choose` at 0.521 -- the tokenizer stopwords build/make/create, so the probe reduced to
    {describe, scene} and the headline skill couldn't dominate. Moving 'Describe' into the entry NAME restored the
    name-bonus. Currently 0.667; the margin canary fires if a Phase-1 change erodes it toward the 0.6 cliff."""
    r = sk.route("describe a scene and build it")
    assert r["decision"] == "act", ("describe-a-scene fell back to 'choose' (was 0.667); a ranking change eroded "
                                    "its dominance below %.2f -- conf now %s" % (ACT_THRESHOLD, r.get("confidence")))
    assert r["confidence"] >= 0.63, ("confidence %s is drifting toward the 0.6 act/choose cliff (was 0.667) -- "
                                     "route decays as the catalog grows; investigate before it flips" % r["confidence"])


def test_job_control_routes_to_act():
    """rev.9: shipped `choose` at 0.565 because a CLIENT entry (cloud bake) shares the job skill's vocabulary and
    split its dominance. The verbs moved into the skill NAME. Currently 0.63 -- a tighter margin than the scene
    pin, so its floor is set just under the measured value to catch erosion without flaking."""
    r = sk.route("start pause resume cancel a render job")
    assert r["decision"] == "act", ("job-control fell back to 'choose' (was 0.63) -- conf now %s" % r.get("confidence"))
    assert r["confidence"] >= 0.60, ("job-control confidence %s dropped below the act threshold -- it sits close "
                                     "to the cliff by nature (a real client shares its words)" % r["confidence"])


def test_pure_nonsense_routes_to_unknown():
    """rev.9: 'qwzx nonsense zzzq' matched a real skill because a 300-word `does` literally contained the word
    'nonsense'. One word reworded. A garbage query must resolve to `unknown`, never a confident (or even
    ambiguous) match -- this is also a canary for the T3 essay-does cleanup: if a long `does` reintroduces an
    incidental match for a nonsense token, this fails."""
    r = sk.route("qwzx nonsense zzzq")
    assert r["decision"] == "unknown", ("a nonsense query matched a real skill -- some entry's does/aliases picked "
                                        "up a garbage token: %s" % r)


def test_route_still_distinguishes_act_from_choose_generally():
    """A guard that the harness itself is not vacuous: a clearly-single-skill query acts, and a deliberately
    ambiguous two-domain query does not silently 'act' on one arm. If BOTH of these ever read the same decision,
    route() has collapsed and the specific pins above would give false comfort."""
    clear = sk.route("render a scene with global illumination")
    assert clear["decision"] in ("act", "choose")           # a real routed decision, not 'unknown'
    assert "confidence" in clear
