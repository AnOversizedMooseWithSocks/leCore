"""holographic_skills.py -- an AGENT-FRIENDLY layer over the engine: describe skills, suggest them from a plain task,
route with confidence, and autocomplete method names.

WHY
---
The engine is large (hundreds of modules, ~75 curated capability homes, hundreds of UnifiedMind methods). A human can
browse; an AGENT needs three things to use it well, and this module provides them, all deterministic and stdlib-only:

  1. SKILL DESCRIPTIONS  -- skill_card(name) / manifest(): a machine-readable card per capability (what it does, how to
     CALL it, aliases) and per UnifiedMind method (its real signature, pulled by introspection, + one-line summary).
  2. SUGGEST / AUTOCOMPLETE -- suggest(task): rank capabilities for a plain-English task WITH A CONFIDENCE, so an agent
     can pick; complete(prefix): method-name autocomplete for constructing a call.
  3. CONFIDENT ROUTING (a decision node) -- route(task): when one skill clearly wins, return {"decision":"act", ...}
     with the concrete call; when it is ambiguous, return {"decision":"choose", options:[...]} -- the "act when
     confident, ask when not" behaviour agents want, computed from the match-score gap.

Nothing here changes what the engine DOES; it only makes what the engine HAS easy to find and invoke.
"""
import inspect


# ---- confidence from raw match scores ---------------------------------------------------------------------
def _confidence(scored):
    """Turn find_scored() results into a 0..1 confidence that the TOP hit is the right one. Two factors: DOMINANCE
    (how far ahead of the runner-up) and STRENGTH (absolute match quality -- a 3+ shared-word hit is strong). A lone
    strong hit is confident; a near-tie is not.

    FACET RULE (measured, rev. 9's sequel): a runner-up that is ANOTHER ENTRY OF THE SAME MODULE is a facet of
    the same subsystem, not a competing skill -- acting on the primary is correct regardless, so it must not
    dilute dominance. The measured case: 'describe a scene and build it' -> the scene_semantic PRIMARY (3.0)
    was dragged to 'choose' (conf 0.545) by scene_semantic's OWN node-graph drill-down entry (2.5), whose
    does-text honestly documents a real describe() method. Trimming honest API documentation to win a routing
    duel would be lying to the catalog; recognising a same-home facet is the truthful fix. The rule needs BOTH
    modules tagged and equal -- untagged entries keep the historical behaviour exactly."""
    if not scored:
        return 0.0
    s0 = scored[0][1]
    top_mod = getattr(scored[0][0], "resolved_module", lambda: None)()
    s1 = 0.0
    for cap, sc in scored[1:]:
        mod = getattr(cap, "resolved_module", lambda: None)()
        if top_mod is not None and mod == top_mod:
            continue                                            # same-home facet: not a competitor
        s1 = sc
        break
    dominance = s0 / (s0 + s1) if (s0 + s1) > 0 else 1.0        # 1.0 = clear winner, 0.5 = tie
    strength = min(1.0, s0 / 3.0)                              # 3+ shared words -> full strength
    return round(dominance * strength, 3)


# ---- UnifiedMind introspection (real signatures, no instantiation) ----------------------------------------
_MIND_METHODS = None


def mind_methods():
    """Every public UnifiedMind method as {name: {"signature": "(args...)", "summary": "first docstring line"}}.
    Read straight off the class by introspection -- so an agent gets the EXACT call, always in sync with the code.
    Cached (the class doesn't change at runtime)."""
    global _MIND_METHODS
    if _MIND_METHODS is None:
        from holographic.misc.holographic_unified import UnifiedMind
        out = {}
        for name, fn in inspect.getmembers(UnifiedMind, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            try:
                sig = str(inspect.signature(fn))
                sig = sig.replace("(self, ", "(").replace("(self)", "()")   # drop the bound self
            except (ValueError, TypeError):
                sig = "(...)"
            doc = (fn.__doc__ or "").strip()
            summary = doc.split("\n")[0].strip() if doc else ""
            out[name] = {"signature": sig, "summary": summary}
        _MIND_METHODS = out
    return _MIND_METHODS


def complete(prefix, k=15):
    """Method-name AUTOCOMPLETE: UnifiedMind methods starting with `prefix` (sorted), each with its signature -- what
    an agent (or an IDE) offers as you type `mind.<prefix...`."""
    p = str(prefix).lower()
    ms = mind_methods()
    hits = sorted(n for n in ms if n.lower().startswith(p))
    return [{"name": n, "signature": ms[n]["signature"], "summary": ms[n]["summary"]} for n in hits[:k]]


# ---- skill cards ------------------------------------------------------------------------------------------
def _catalog():
    from holographic.caching_and_storage.holographic_catalog import default_catalog, seed_from_modules
    return seed_from_modules(default_catalog())


def skill_card(name):
    """A machine-readable card for ONE skill, resolved either as a catalog CAPABILITY (by name) or a UnifiedMind
    METHOD (by name). None if neither. Gives an agent the what + the how-to-call in one object."""
    cat = _catalog()
    cap = cat.get(name)
    if cap is not None:
        return {"kind": "capability", "name": cap.name, "does": cap.does, "example": cap.example,
                "native": cap.native, "aliases": list(cap.aliases)}
    ms = mind_methods()
    if name in ms:
        return {"kind": "method", "name": name, "call": "mind.%s%s" % (name, ms[name]["signature"]),
                "signature": ms[name]["signature"], "summary": ms[name]["summary"]}
    return None


def manifest(include_methods=True):
    """The full machine-readable skill list: every curated capability home, plus (optionally) every UnifiedMind method
    with its signature. What an agent loads ONCE to know the whole surface it can drive."""
    cat = _catalog()
    caps = [{"kind": "capability", "name": c.name, "does": c.does, "example": c.example, "aliases": list(c.aliases)}
            for c in cat.all()]
    if not include_methods:
        return {"capabilities": caps}
    ms = mind_methods()
    methods = [{"kind": "method", "name": n, "call": "mind.%s%s" % (n, ms[n]["signature"]),
                "summary": ms[n]["summary"]} for n in sorted(ms)]
    return {"capabilities": caps, "methods": methods, "counts": {"capabilities": len(caps), "methods": len(methods)}}


# ---- suggest + route --------------------------------------------------------------------------------------
def _rank(task, k=8):
    """Rank matches for a task, preferring CURATED homes (real skills) over auto-registered module entries -- a module
    is a pointer to a capability, not a competing skill, so it must not dilute confidence. Falls back to module hits
    only when nothing curated matches. Returns [(capability, score)] best-first."""
    scored = _catalog().find_scored(task, k=k)
    curated = [(c, s) for c, s in scored if not c.name.startswith("holographic_")]
    return curated if curated else scored


def suggest(task, k=5):
    """AUTOCOMPLETE a plain-English task to the best capabilities, each with a CONFIDENCE and the concrete call/example.
    An agent uses this to decide what to invoke. Ranked, deterministic."""
    scored = _rank(task, k=k)
    if not scored:
        return []
    conf = _confidence(scored)
    out = []
    for i, (cap, score) in enumerate(scored):
        out.append({"name": cap.name, "does": cap.does, "call": cap.example,
                    "confidence": conf if i == 0 else round(conf * (score / scored[0][1]), 3)})
    return out


def route(task, act_threshold=0.6):
    """A CONFIDENT-ROUTING decision node for agents. When one skill clearly wins (confidence >= threshold), return
    {"decision":"act", "skill":..., "confidence":...} with the call to make. When it's ambiguous, return
    {"decision":"choose", "options":[...], "prompt":...} so the agent asks or reasons instead of guessing. This is the
    'act when confident, ask when not' behaviour -- the decision is explicit and score-based, never a silent guess."""
    scored = _rank(task, k=4)
    if not scored:
        return {"decision": "unknown", "confidence": 0.0,
                "prompt": "No capability matched %r. Try mind.suggest() with different words." % task}
    conf = _confidence(scored)
    top, _ = scored[0]
    if conf >= act_threshold:
        return {"decision": "act", "confidence": conf,
                "skill": {"name": top.name, "does": top.does, "call": top.example}}
    return {"decision": "choose", "confidence": conf,
            "prompt": "Ambiguous -- did you mean one of these?",
            "options": [{"name": c.name, "does": c.does, "call": c.example} for c, _ in scored]}


def _selftest():
    # skill cards: a capability and a method both resolve
    cap = skill_card("Index (search)")
    assert cap and cap["kind"] == "capability" and cap["example"]
    ms = mind_methods()
    assert len(ms) > 100                                        # the mind has a big, introspectable surface
    some = next(iter(ms))
    m = skill_card(some)
    assert m["kind"] == "method" and m["call"].startswith("mind.%s(" % some)

    # autocomplete: method names by prefix, with signatures
    comp = complete("learn_")
    assert comp and all(c["name"].startswith("learn_") for c in comp) and comp[0]["signature"].startswith("(")

    # suggest: a task -> ranked skills with confidence + a call
    sug = suggest("draw a picture")
    assert sug and "2D image" in sug[0]["name"] and 0.0 <= sug[0]["confidence"] <= 1.0 and sug[0]["call"]

    # route: confident task -> act; vague task -> choose (or unknown)
    r_act = route("render a scene with global illumination")
    assert r_act["decision"] in ("act", "choose") and "confidence" in r_act
    r_clear = route("start pause resume cancel a render job")
    assert r_clear["decision"] == "act" and "call" in r_clear["skill"]
    r_none = route("qwzx nonsense zzzq")
    assert r_none["decision"] == "unknown"

    # manifest: the whole surface, machine-readable
    man = manifest()
    assert man["counts"]["capabilities"] > 50 and man["counts"]["methods"] > 100

    print("OK: holographic_skills self-test passed (%d capabilities + %d introspected methods; skill cards, method "
          "autocomplete, task->suggest with confidence, confident act/choose routing, machine-readable manifest)"
          % (man["counts"]["capabilities"], man["counts"]["methods"]))


if __name__ == "__main__":
    _selftest()
