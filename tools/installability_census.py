#!/usr/bin/env python3
"""tools/installability_census.py -- G1: COUNT, don't guess (the metric of "are we there").

Classifies every public UnifiedMind faculty by its INSTALLABILITY SHAPE, from the signature and
the catalog, without calling anything (calling 2,000 faculties blind is how you set the lab on
fire). Classes, per docs/INSTALLED.md's columns plus the G0 principle:

  PROBE_SHAPED     unary, no required extras -> a candidate for probe_project certification
                   (the projector then delivers the real verdict: installs or refuses)
  RESHAPEABLE      array-in/array-out with fixed extra params -> candidate after currying
  FACTORY/STATEFUL returns an object / holds state -> the object's METHODS are the candidates
                   (a second-pass census target, not counted installable here)
  CONTROL/SERVICE  loops, schedulers, file/service plumbing -> host-shape or shadowed (G0)
  OUTPUT_TEXT      produces text/serializable output -> installable via the decode head (G0)

The census is DELIBERATELY conservative: PROBE_SHAPED is a candidacy, not a verdict -- the
projector certifies or refuses each candidate individually (G15 re-runs this after each
vocabulary extension and tracks the fraction). Deterministic; prints a reproducible table.
"""
import inspect
import sys

import numpy as np

sys.path.insert(0, ".")


def census():
    import lecore
    mind = lecore.UnifiedMind(dim=64, seed=0)
    classes = {"PROBE_SHAPED": [], "RESHAPEABLE": [], "FACTORY_STATEFUL": [],
               "CONTROL_SERVICE": [], "OUTPUT_TEXT": [], "OTHER": []}
    ctrl_words = ("run", "serve", "start", "stop", "watch", "loop", "schedule", "spawn",
                  "file_", "http", "save", "load", "zip", "install", "audit", "lint")
    text_words = ("to_text", "dump", "describe", "explain", "report", "summar", "_md", "doc")
    for name in dir(mind):
        if name.startswith("_"):
            continue
        fn = getattr(mind, name)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            classes["OTHER"].append(name)
            continue
        params = [p for p in sig.parameters.values() if p.name != "self"]
        required = [p for p in params if p.default is inspect.Parameter.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        lname = name.lower()
        if any(w in lname for w in ctrl_words):
            classes["CONTROL_SERVICE"].append(name)
        elif any(w in lname for w in text_words):
            classes["OUTPUT_TEXT"].append(name)
        elif len(required) == 1:
            classes["PROBE_SHAPED"].append(name)
        elif 2 <= len(required) <= 3:
            classes["RESHAPEABLE"].append(name)
        elif len(required) == 0:
            classes["FACTORY_STATEFUL"].append(name)
        else:
            classes["OTHER"].append(name)
    return classes


def module_census(n_sample=140, dim=32, per_call_seconds=5):
    """W2 -- THE MODULE FRAME: the facade census asks "call the mind, get weights" (verdict
    8.8%); THIS asks "what math exists to compile" -- the FAC compiler consumes INNER MODULE
    FUNCTIONS directly (every installed customer so far was one), so the honest denominator is
    public module-level functions with a single required argument, probed at that level. Same
    guards as the facade probe: SIGALRM per call, exceptions classify honestly, control/effect
    names excluded up front (probing file_save teaches nothing about linear algebra and might
    write files doing it)."""
    import importlib
    import pkgutil
    import signal

    import holographic
    from holographic.io_and_interop.holographic_projector import probe_project
    ctrl_words = ("run", "serve", "start", "stop", "watch", "loop", "schedule", "spawn",
                  "file_", "http", "save", "load", "zip", "install", "audit", "lint",
                  "write", "delete", "plot", "main")
    funcs = []
    n_mods = 0
    for mi in pkgutil.walk_packages(holographic.__path__, prefix="holographic."):
        if mi.ispkg or "catalog" in mi.name or "unified" in mi.name:
            continue
        try:
            mod = importlib.import_module(mi.name)
        except Exception:
            continue
        n_mods += 1
        for nm in dir(mod):
            if nm.startswith("_") or any(w in nm.lower() for w in ctrl_words):
                continue
            fn = getattr(mod, nm)
            if not (inspect.isfunction(fn) and getattr(fn, "__module__", "") == mi.name):
                continue
            try:
                ps = [q for q in inspect.signature(fn).parameters.values()
                      if q.default is inspect.Parameter.empty
                      and q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
            except (ValueError, TypeError):
                continue
            funcs.append((mi.name + "." + nm, fn, len(ps)))
    single = sorted((n, f) for n, f, k in funcs if k == 1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(single), size=min(n_sample, len(single)), replace=False)
    verdicts = {}

    def _alarm(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _alarm)
    for i in idx:
        name, fn = single[i]
        signal.alarm(per_call_seconds)
        try:
            verdicts[name] = probe_project(lambda v, f=fn: f(v), dim)["kind"]
        except TimeoutError:
            verdicts[name] = "timeout"
        except Exception:
            verdicts[name] = "not_probe_callable"
        finally:
            signal.alarm(0)
    return {"n_modules": n_mods, "n_public_funcs": len(funcs),
            "n_single_required": len(single), "verdicts": verdicts}


def probe_sample(n_sample=80, dim=32, per_call_seconds=5):
    """The G15 follow-up, DONE instead of deferred: call a deterministic stratified sample of
    PROBE_SHAPED candidates through probe_project and report the VERDICT rate -- candidacy was
    always a signature-level claim; this is the measured one. Guards: each call runs under a
    SIGALRM budget (a hung faculty is data, not a crash), exceptions classify as
    not_probe_callable (a faculty whose one required arg is a string is a candidate the
    signature census could not exclude -- counting it honestly is the point)."""
    import signal

    import lecore
    from holographic.io_and_interop.holographic_projector import probe_project
    mind = lecore.UnifiedMind(dim=dim, seed=0)
    cand = sorted(census()["PROBE_SHAPED"])
    rng = np.random.default_rng(0)
    sample = [cand[i] for i in rng.choice(len(cand), size=min(n_sample, len(cand)), replace=False)]
    verdicts = {}

    def _alarm(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _alarm)
    for name in sample:
        fn = getattr(mind, name)
        signal.alarm(per_call_seconds)
        try:
            pr = probe_project(lambda v, f=fn: f(v), dim)
            verdicts[name] = pr["kind"]
        except TimeoutError:
            verdicts[name] = "timeout"
        except Exception:
            verdicts[name] = "not_probe_callable"
        finally:
            signal.alarm(0)
    return verdicts


def typed_probe_sample(n_sample=80, dim=32, per_call_seconds=5):
    """The named next probe: WHAT do the not-probe-callable 87.5% actually take? For each
    sampled faculty that rejects a vector, try a small typed battery (2D array via flatten
    ADAPTER, text, int, float list, dict) under the same SIGALRM budget and record the first
    type that succeeds. For 2D-array successes, run the FULL certification through the reshape
    adapter -- the reshaping lever's first measured delta on the verdict rate. Faculties that
    take text are counted OUT honestly: token-space work is the HOST's native job, not a
    projector gap."""
    import signal

    import lecore
    from holographic.io_and_interop.holographic_projector import probe_project
    mind = lecore.UnifiedMind(dim=dim, seed=0)
    base = probe_sample(n_sample=n_sample, dim=dim, per_call_seconds=per_call_seconds)
    side = int(np.sqrt(dim))
    battery = [
        ("array2d", lambda f: probe_project(
            lambda v, f=f: np.asarray(f(v.reshape(side, side)), float).reshape(-1), dim)),
        ("text", lambda f: (f("holographic memory"), {"kind": "takes_text"})[1]),
        ("int", lambda f: (f(3), {"kind": "takes_int"})[1]),
        ("float_list", lambda f: (f([1.0, 2.0, 3.0]), {"kind": "takes_list"})[1]),
        ("dict", lambda f: (f({}), {"kind": "takes_dict"})[1]),
    ]

    def _alarm(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _alarm)
    typed, adapter_verdicts = {}, {}
    for name, verdict in base.items():
        if verdict != "not_probe_callable":
            continue
        fn = getattr(mind, name)
        for tname, attempt in battery:
            signal.alarm(per_call_seconds)
            try:
                res = attempt(fn)
                typed[name] = tname
                if tname == "array2d":
                    adapter_verdicts[name] = res["kind"]
                break
            except Exception:
                continue
            finally:
                signal.alarm(0)
        else:
            typed[name] = "none_matched"
        signal.alarm(0)
    return base, typed, adapter_verdicts


def main():
    classes = census()
    total = sum(len(v) for v in classes.values())
    print("installability census over %d public faculties" % total)
    print("%-18s %6s %7s" % ("class", "count", "share"))
    for k, v in classes.items():
        print("%-18s %6d %6.1f%%" % (k, len(v), 100.0 * len(v) / max(1, total)))
    cand = len(classes["PROBE_SHAPED"]) + len(classes["RESHAPEABLE"])
    print("\ncertification CANDIDATES (probe-shaped + reshapeable): %d (%.1f%%)"
          % (cand, 100.0 * cand / max(1, total)))
    print("first 20 probe-shaped candidates:", classes["PROBE_SHAPED"][:20])
    if "--probe" in sys.argv:
        v = probe_sample()
        from collections import Counter
        counts = Counter(v.values())
        n = len(v)
        certified = sum(c for k, c in counts.items()
                        if k not in ("refused", "not_probe_callable", "timeout"))
        print("\nprobe-sample verdicts (n=%d, dim=32, deterministic sample):" % n)
        for k, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print("   %-20s %3d  %5.1f%%" % (k, c, 100.0 * c / n))
        print("CERTIFY RATE among probe-shaped sample: %.1f%%  "
              "(the verdict-level number; candidacy was %d faculties)" % (100.0 * certified / n,
                                                                          len(classes["PROBE_SHAPED"])))
    if "--modules" in sys.argv:
        r = module_census()
        from collections import Counter
        cc = Counter(r["verdicts"].values())
        n = len(r["verdicts"])
        cert = sum(v for k, v in cc.items() if k not in ("refused", "not_probe_callable", "timeout"))
        print("\nMODULE-frame census: %d modules, %d public funcs, %d single-required-arg"
              % (r["n_modules"], r["n_public_funcs"], r["n_single_required"]))
        for k, v in sorted(cc.items(), key=lambda kv: -kv[1]):
            print("   %-20s %3d  %5.1f%%" % (k, v, 100.0 * v / n))
        print("MODULE-frame verdict rate: %.1f%% (n=%d sample)   [facade frame: 8.8%% -- "
              "different question, both stand]" % (100.0 * cert / n, n))
    if "--typed" in sys.argv:
        base, typed, adapt = typed_probe_sample()
        from collections import Counter
        tc = Counter(typed.values())
        print("\ntyped probe of the not-probe-callable (n=%d):" % len(typed))
        for k, c in sorted(tc.items(), key=lambda kv: -kv[1]):
            print("   %-14s %3d  %5.1f%%" % (k, c, 100.0 * c / max(1, len(typed))))
        newly = {k: v for k, v in adapt.items() if v not in ("refused",)}
        base_cert = sum(1 for v in base.values()
                        if v not in ("refused", "not_probe_callable", "timeout"))
        n = len(base)
        print("reshape-ADAPTER certifications among array2d takers: %d (%s)"
              % (len(newly), sorted(Counter(adapt.values()).items())))
        print("VERDICT RATE: %.1f%% direct -> %.1f%% with the reshape adapter (the lever's "
              "first measured delta)" % (100.0 * base_cert / n, 100.0 * (base_cert + len(newly)) / n))


if __name__ == "__main__":
    main()
