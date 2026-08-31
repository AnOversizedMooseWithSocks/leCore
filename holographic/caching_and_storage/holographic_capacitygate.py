"""Capacity gate: consult every measured law BEFORE allocating, and ROUTE to the escape.

WHY THIS EXISTS. `advise_scale` already applies every measured capacity law in one call and reports
which one binds. Nothing in the engine calls it before allocating, so a request 300x past the
pair-capacity law does not fail loudly -- it succeeds and then recalls at 0.007. A law that is
measured and not consulted is worth exactly as much as a law nobody measured.

WHY IT ROUTES INSTEAD OF REFUSING. Every failing law in this engine has a MEASURED ESCAPE, and the
escape is never "more dimension":

    pair-capacity   -> celled_memory        one memory 70x past the law recalls 0.007;
                                            celled recalls 1.000 across 71 cells
    nesting depth   -> encode_tree_carrier  the flat encoder retains 0.00044 of the leaf's
                                            distinguishable share at depth 7 (10 seeds, CI
                                            [0.00043, 0.00045]); the carrier encoder recovers
                                            leaves at 0.94-1.00 at depths 7-32
    bundle readout  -> bundle_recover       linear one-shot washes out at load; cosamp/amp hold
                                            ~8.7x more than the linear ceiling suggests

So a refusal is the WRONG answer when a prescription exists. This is the abstain-not-error
discipline with the third arm attached: proceed, reroute, or abstain -- never silently collapse.

KEPT NEGATIVE. The gate does not RESIZE anything and never mutates a caller's request. It reports.
Auto-fixing a dimension is the one move that looks helpful and is not: the binding law's own
prescription for 2000 pairs over vocab 6767 is `dim >= 153536`, which no caller wants and which
routing makes unnecessary.
"""

import numpy as np  # noqa: F401  (kept for family-module consistency and future vectorised checks)


# The measured escapes, keyed by the law that binds. Each entry names the faculty a caller should
# use instead, and WHY -- so a gate result is actionable without reading this file.
_ESCAPES = {
    "pair-capacity": (
        "celled_memory",
        "cells of exactly n* pairs with an exact key->cell directory; "
        "measured 0.007 (one memory, 70x past the law) -> 1.000 (celled, 71 cells)",
    ),
    "nesting depth": (
        "encode_tree_carrier",
        "each level on its own carrier makes depth contribution linear instead of geometric; "
        "flat retains 0.00044 of the leaf share at d7, carrier recovers 0.94-1.00 at d7-32",
    ),
    "bundle readout": (
        "bundle_recover",
        "cosamp/amp sparse recovery holds ~8.7x more than the linear-readout ceiling",
    ),
    "PIC transition": (
        "one-shot decoder",
        "the PIC decoder stops paying above its measured load; switch decoder rather than dim",
    ),
    "factorization": (
        "linear-code factorization",
        "exact algebraic recovery routes AROUND the F=4 dense-code wall rather than through it",
    ),
}


def _escape_for(law_name):
    """Return (faculty, why) for a law name, or (None, None) when no escape is on record.

    Matched by PREFIX because the law strings carry their own measurement notes -- e.g.
    "nesting depth (measured: dim-independent collapse ~d5-7)". Matching the whole string would
    break the moment a measurement is refined, which is exactly when the gate must keep working.
    """
    for prefix, (faculty, why) in _ESCAPES.items():
        if law_name.startswith(prefix):
            return faculty, why
    return None, None


def capacity_gate(advice):
    """Turn an `advise_scale` result into a decision: proceed, reroute, or abstain.

    `advice` is the dict `advise_scale` returns: {'laws': [...], 'ok': bool, 'binding': str,
    'prescription': str}. Returns a dict:

        verdict     'proceed' | 'reroute' | 'abstain'
        binding     the law that binds, or None when all pass
        route       the faculty to use instead, or None
        why         one line of measured justification for the route
        failing     [{'law', 'margin', 'route', 'why', 'prescription'}, ...] for every failing law
        prescription  the binding law's own prescription, passed through unchanged

    'reroute' means at least one law fails AND every failing law has a measured escape.
    'abstain' means a law fails with no escape on record -- the honest answer, not a guess.
    """
    laws = list(advice.get("laws", []))
    failing = []
    for law in laws:
        if law.get("ok", True):
            continue
        route, why = _escape_for(str(law.get("law", "")))
        failing.append({
            "law": law.get("law"),
            "margin": law.get("margin"),
            "route": route,
            "why": why,
            "prescription": law.get("prescription"),
        })

    if not failing:
        return {"verdict": "proceed", "binding": None, "route": None, "why": None,
                "failing": [], "prescription": None}

    binding = advice.get("binding")
    # The routed answer for the BINDING law is what a caller acts on first; the rest travel with it
    # so a caller that fixes one law does not discover the next one by hitting it.
    head = next((f for f in failing if f["law"] == binding), failing[0])
    all_routed = all(f["route"] for f in failing)
    return {
        "verdict": "reroute" if all_routed else "abstain",
        "binding": binding,
        "route": head["route"],
        "why": head["why"],
        "failing": failing,
        "prescription": advice.get("prescription"),
    }


def _selftest():
    """Assert the REAL contract: the gate must route the two laws that actually fail in practice,
    must proceed when nothing fails, and must never invent a route it has no measurement for."""
    # A passing advice -> proceed, and nothing else.
    ok = {"laws": [{"law": "pair-capacity (allocate)", "ok": True, "margin": 3.0}],
          "ok": True, "binding": None, "prescription": None}
    g = capacity_gate(ok)
    assert g["verdict"] == "proceed", g
    assert g["route"] is None and g["failing"] == [], g

    # The two laws measured to fail at real sizes must both ROUTE, not abstain.
    bad = {"laws": [
        {"law": "pair-capacity (allocate)", "ok": False, "margin": 0.0033,
         "prescription": "dim >= 153536"},
        {"law": "nesting depth (measured: dim-independent collapse ~d5-7)", "ok": False,
         "margin": 0.667, "prescription": "dim is NOT the lever"},
    ], "ok": False, "binding": "pair-capacity (allocate)", "prescription": "dim >= 153536"}
    g = capacity_gate(bad)
    assert g["verdict"] == "reroute", g
    assert g["route"] == "celled_memory", g
    assert len(g["failing"]) == 2, g
    assert g["failing"][1]["route"] == "encode_tree_carrier", g
    # the prescription travels through UNCHANGED -- the gate reports, it does not resize
    assert g["prescription"] == "dim >= 153536", g

    # A law with no escape on record must ABSTAIN, never guess a faculty.
    unknown = {"laws": [{"law": "some future law", "ok": False, "margin": 0.1}],
               "ok": False, "binding": "some future law", "prescription": None}
    g = capacity_gate(unknown)
    assert g["verdict"] == "abstain", g
    assert g["route"] is None, g

    # Prefix matching must survive a refined measurement string.
    refined = {"laws": [{"law": "nesting depth (measured: dilution ratio 1/3 per level, "
                                "dim-invariant at 95% CI)", "ok": False, "margin": 0.5}],
               "ok": False, "binding": "nesting depth", "prescription": "use carrier elevation"}
    g = capacity_gate(refined)
    assert g["failing"][0]["route"] == "encode_tree_carrier", g

    print("holographic_capacitygate selftest OK")


if __name__ == "__main__":
    _selftest()
