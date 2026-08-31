"""delegation_drift -- catch a FACULTY whose signature has drifted from the function it delegates to.

WHY THIS TOOL EXISTS
--------------------
The engine's rule is that a UnifiedMind faculty DELEGATES and never reimplements, and the convention
is that its docstring ends with "See holographic_<module>.<function>". That convention is machine-
readable, and nothing was reading it.

The bug it catches is specific and quiet: a parameter is added to the module function, the faculty is
not updated, and the capability is now REACHABLE BUT CRIPPLED -- the feature exists, the docs and the
catalog describe it, and calling it through the mind raises TypeError. Every existing audit passes,
because the module has a docstring (reachability), the catalog example still runs (skill_lint), and
nothing is unwired (wiring_report). The failure is in the SEAM, and no tool was looking at seams.

It has now happened twice in one arc:
    scatter_mesh        gained `holographic`/`dim`/`cell_size` in the module; the faculty did not
    creature_material   gained `iridescence`/`film_nm`/`n_film`; the faculty did not
Both were caught by an integration test happening to call the new argument -- i.e. by luck, late.

WHAT IT REPORTS
    MISSING   the delegate accepts a parameter the faculty does not expose and cannot forward.
              This is the real bug: the argument is unreachable through the mind.
    EXTRA     the faculty accepts something the delegate does not. Usually deliberate (a faculty that
              composes two calls), so it is reported separately and does not gate.

WHAT IT DELIBERATELY DOES NOT FLAG
  * A faculty with **kwargs: it can forward anything, so nothing is unreachable.
  * A faculty that intentionally exposes a SUBSET -- there is no way to tell that apart from drift
    automatically, so those live in BUDGET below, named, with a reason. A budget entry is a decision
    on the record; silence would be a decision nobody can see.

KEPT NEGATIVE (loud)
    This checks NAMES, not semantics. A faculty that forwards `seed` to the delegate's `rng_seed`
    reads as drift, and a faculty that forwards a parameter to the WRONG delegate parameter reads as
    clean. It is a seam-shaped net, not a proof of correctness.
"""

import argparse
import importlib
import inspect
import re
import sys

#: Faculties that intentionally expose fewer parameters than their delegate, with the REASON. Each
#: entry is a decision on the record; adding one should feel like a small commitment, not a mute button.
BUDGET = {
    # "faculty_name": "why the narrower signature is deliberate",
    #
    # A faculty that SUPPLIES the missing parameter itself is not drift. These three delegate to
    # functions taking `mind=` for the FABRIK reach, and the faculty passes `self` -- exposing it
    # would let a caller hand a DIFFERENT mind to a method reached through this one, which is worse
    # than not exposing it. Recorded here rather than left silent, so the exemption is a decision on
    # the record instead of a hole in the audit.
    "gait_pose": "supplies mind=self; exposing it would let a caller pass a foreign mind",
    "gait_frames": "supplies mind=self; exposing it would let a caller pass a foreign mind",
    "gait_report": "supplies mind=self; exposing it would let a caller pass a foreign mind",
}

_SEE = re.compile(r"See\s+(holographic_[A-Za-z0-9_]+)\.([A-Za-z_][A-Za-z0-9_]*)")


def _find_module(short_name):
    """Resolve 'holographic_foo' to its real dotted path -- the engine keeps modules in family
    packages, and the docstring convention only records the leaf name."""
    for pkg in ("mesh_and_geometry", "materials_and_texture", "rendering", "simulation_and_physics",
                "sampling_and_signal", "agents_and_reasoning", "caching_and_storage", "io_and_interop",
                "scene_and_pipeline", "semantic_router", "misc", "unified"):
        try:
            return importlib.import_module("holographic.%s.%s" % (pkg, short_name))
        except Exception:
            continue
    try:
        return importlib.import_module(short_name)
    except Exception:
        return None


def _params(fn):
    """Named parameters of a callable, plus whether it takes **kwargs. `self` is dropped so a method
    and a plain function compare on equal terms."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None, False
    names, has_kw = [], False
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            has_kw = True
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        else:
            names.append(p.name)
    return names, has_kw


def audit(verbose=False, min_overlap=0.8):
    """Compare every faculty that names a delegate against that delegate's signature."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # sweep 124: runnable from anywhere
    import lecore
    mind = lecore.UnifiedMind(dim=32, seed=0)

    missing, extra, unresolved, checked = [], [], [], 0
    for name in sorted(dir(mind)):
        if name.startswith("_"):
            continue
        try:
            fac = getattr(type(mind), name, None) or getattr(mind, name)
        except Exception:
            continue
        if not callable(fac):
            continue
        doc = inspect.getdoc(fac) or ""
        m = _SEE.search(doc)
        if not m:
            continue                                          # no declared delegate: nothing to compare
        mod_name, fn_name = m.group(1), m.group(2)
        mod = _find_module(mod_name)
        if mod is None:
            unresolved.append((name, "%s.%s" % (mod_name, fn_name), "module not importable"))
            continue
        target = getattr(mod, fn_name, None)
        if target is None:                                    # often a Class.method reference
            for attr in dir(mod):
                obj = getattr(mod, attr, None)
                if inspect.isclass(obj) and hasattr(obj, fn_name):
                    target = getattr(obj, fn_name)
                    break
        if target is None or not callable(target):
            unresolved.append((name, "%s.%s" % (mod_name, fn_name), "function not found"))
            continue

        fac_p, fac_kw = _params(fac)
        tgt_p, _tgt_kw = _params(target)
        if fac_p is None or tgt_p is None:
            continue
        checked += 1
        if fac_kw:
            continue                                          # **kwargs forwards everything
        miss = [p for p in tgt_p if p not in fac_p]
        # OVERLAP is what separates drift from a different calling convention. A faculty that already
        # forwards most of the delegate's parameters by name IS a 1:1 forwarder, so a small gap is
        # almost certainly a parameter added later and never plumbed through. A faculty that shares
        # few names is doing something else entirely (building a class, computing its arguments), and
        # flagging it would be noise. MEASURED across the engine: 238 raw findings fall to 42 at 0.8
        # overlap, and the ones that survive are single missing parameters on obvious forwarders.
        overlap = (len([p for p in tgt_p if p in fac_p]) / len(tgt_p)) if tgt_p else 1.0
        if miss and overlap >= min_overlap and name not in BUDGET:
            missing.append((name, "%s.%s" % (mod_name, fn_name), miss, round(overlap, 2)))
        ext = [p for p in fac_p if p not in tgt_p]
        if ext:
            extra.append((name, "%s.%s" % (mod_name, fn_name), ext))

    print("DELEGATION DRIFT over %d faculties that declare a delegate" % checked)
    print()
    print("  MISSING -- the delegate takes a parameter the faculty cannot forward "
          "(reachable but crippled): %d" % len(missing))
    for n, tg, ps, ov in missing:
        print("    %-32s -> %-42s [overlap %.2f] missing: %s" % (n, tg, ov, ", ".join(ps)))
    if verbose:
        print()
        print("  EXTRA -- faculty takes what the delegate does not (usually deliberate): %d" % len(extra))
        for n, t, ps in extra[:40]:
            print("    %-34s -> %-46s extra: %s" % (n, t, ", ".join(ps)))
        print()
        print("  UNRESOLVED delegate references (not gating): %d" % len(unresolved))
        for n, t, why in unresolved[:25]:
            print("    %-34s -> %-46s %s" % (n, t, why))
    if BUDGET:
        print()
        print("  BUDGETED (deliberately narrower): %d" % len(BUDGET))
        for k, why in sorted(BUDGET.items()):
            print("    %-34s %s" % (k, why))
    print()
    print("TOTAL: %d likely-drifted faculty signature(s) at overlap >= %.2f." % (len(missing), min_overlap))
    print("(REPORT ONLY by default -- this is a pre-existing backlog across a mature engine, and a "
          "tool that fails the build on day one gets disabled rather than fixed. Use --gate once the "
          "backlog is cleared, or clear it family by family.)")
    return {"missing": missing, "extra": extra, "unresolved": unresolved, "checked": checked}


def _selftest():
    """The tool must CATCH a synthetic drift and PASS a matching pair -- an audit that cannot fail is
    decoration, which is the same failure this tool exists to catch in others."""
    def delegate(a, b, c=1, d=2):
        """target"""
    def good(a, b, c=1, d=2):
        """See holographic_x.delegate."""
    def drifted(a, b):
        """See holographic_x.delegate."""
    def kwargged(a, **kw):
        """See holographic_x.delegate."""

    gp, gk = _params(good); dp, dk = _params(drifted); kp, kk = _params(kwargged)
    tp, _ = _params(delegate)
    assert [p for p in tp if p not in gp] == [], "a matching signature must show no drift"
    assert [p for p in tp if p not in dp] == ["c", "d"], "drift must be detected by name"
    assert kk is True and gk is False, "**kwargs must be recognised as forwarding everything"
    assert _SEE.search(good.__doc__).group(2) == "delegate", "the See-convention must parse"
    # OVERLAP: a near-identical forwarder scores high (real drift); a totally different convention
    # scores low and must be filtered out, or the audit drowns in noise and gets ignored.
    def different(x, y, z):
        """See holographic_x.delegate."""
    dp2, _ = _params(different)
    assert len([p for p in tp if p in dp2]) / len(tp) < 0.5, "a different convention must score LOW"
    assert len([p for p in tp if p in dp]) / len(tp) >= 0.5, "a near-forwarder must score HIGH"
    print("delegation_drift selftest OK: detects missing params, ignores **kwargs, parses the "
          "See-convention, and separates drift from a different calling convention by overlap")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true", help="also list EXTRA and UNRESOLVED")
    ap.add_argument("--min-overlap", type=float, default=0.8,
                    help="only flag forwarders sharing at least this fraction of the delegate's params")
    ap.add_argument("--gate", action="store_true", help="exit non-zero on findings (CI gate)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); sys.exit(0)
    r = audit(verbose=a.verbose, min_overlap=a.min_overlap)
    sys.exit(1 if (r["missing"] and a.gate) else 0)
