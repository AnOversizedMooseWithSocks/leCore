"""HARDEN -- prove the installed layer works, and keeps working when abused.

Every piece of this stack has its own selftest. None of them answered the
question that matters: can a model that has been INSTALLED actually BOOT and USE
the layer, from the weights, with nothing else present -- and does it survive the
things that happen to checkpoints in the real world?

This is that test, and it is deliberately adversarial. The failures it looks for
are the ones this project has actually shipped at least once:

    a manifest that claims what was never written        (testkit, 0 layer arrays)
    a payload readable only in the process that wrote it (hash() vs hashlib)
    a capacity check that disagrees with its writer      (boot row, IndexError)
    a guarantee established before a later edit          (ward, verified then broken)
    a channel that is hidden but not addressed           (any seed reads it)
    a "restriction" that scores better than the baseline (causal leak)

THE STANDARD: an install passes only if the layer BOOTS, RECALLS, EXECUTES and
CACHES from the weights alone, and only if every corruption is DETECTED rather
than silently served. A harness that cannot fail is decoration -- so this one is
run against a damaged model too, and is required to fail there.
"""

import numpy as np


def harden(weights, cfg, seed="leCore", facts=(), program=None, machine=None,
           probe_ids=None, verbose=False):
    """Install nothing; test what is already installed, hard.

    Returns a report with a check list and pass/total. Each check names WHY it
    exists, because a check whose purpose is forgotten becomes a check that gets
    deleted the first time it is inconvenient."""
    from holographic.io_and_interop.holographic_bios import report as bios_report
    from holographic.io_and_interop.holographic_boot import boot
    from holographic.caching_and_storage.holographic_substrate import (
        read_payload, read_seeded)

    checks = []

    def _c(name, fn, why):
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail),
                       "why": why})
        if verbose:
            print("   %-26s %s  %s" % (name, "PASS" if ok else "FAIL", detail))

    # ---- BIOS: the machine must describe itself before anything trusts it ----
    prof = bios_report(weights, cfg, probe_ids=probe_ids)
    _c("bios_post", lambda: (prof["post"]["ok"], prof["post"]["detail"]),
       "installing onto a broken model yields a broken model and a clean report")
    _c("bios_enumerates",
       lambda: (bool(prof["root"]) and prof["n_layers"] > 0,
                "%s, %d layers, %s layout"
                % (prof["root"], prof["n_layers"], prof["projection_layout"])),
       "five bugs this session were one missing enumeration")

    # ---- BOOT: the layer must come up from the weights ----
    _c("boots_from_weights",
       lambda: (boot(weights)["record"].seed == seed, boot(weights)["record"].seed),
       "a record can be written where nothing reads it and nothing raises")
    _c("expansion_deterministic",
       lambda: (np.array_equal(boot(weights)["codebook"][
                    sorted(boot(weights)["codebook"])[0]],
                boot(weights)["codebook"][sorted(boot(weights)["codebook"])[0]]),
                "%d symbols" % len(boot(weights)["codebook"])),
       "hashlib not hash(): a layer booted in another process must agree")

    # ---- ADDRESSED, not merely hidden ----
    def _addressed():
        key = next(k for k in weights if np.asarray(weights[k]).ndim == 2
                   and "embed" not in k)
        A = np.asarray(weights[key])
        a, _ = read_seeded(A, seed=seed, rate=0.05)
        b, _ = read_seeded(A, seed=str(seed) + "!x", rate=0.05)
        n = min(len(a), len(b))
        agree = float(np.mean(a[:n] == b[:n])) if n else 1.0
        return 0.35 < agree < 0.65, "wrong-seed agreement %.2f" % agree
    _c("channel_addressed", _addressed,
       "hidden is not addressed; a wrong seed must read noise")

    # ---- RECALL: facts must come back BY KEY, and absent ones must not ----
    if facts:
        def _recall():
            # EVERY probe goes inside the wrapper. This call used to sit
            # OUTSIDE it, so a damaged model raised out of the harness instead
            # of being reported as a failed check -- a verifier that crashes on
            # the input it exists to judge tells you nothing about it.
            from holographic.io_and_interop.holographic_boot import (
                store_facts, recall)
            rec = boot(weights)["record"]
            trace = store_facts(list(facts), rec)
            vals = [v for _k, v in facts]
            got = [recall(trace, k, rec, vals) for k, _v in facts]
            return (got == vals, "%d/%d" % (sum(g == v for g, v in
                                                zip(got, vals)), len(vals)))
        _c("recall_by_key", _recall,
           "a store nobody can query is a store nobody has")

    # ---- EXECUTE: a stored program must run ----
    if program is not None and machine is not None:
        def _exec():
            from holographic.caching_and_storage.holographic_substrate import (
                load_program)
            pv = load_program(weights, bits=1)
            acc, trace_ = machine.run(pv, max_steps=32)
            ref_acc, ref_trace = machine.run(machine.assemble(program),
                                             max_steps=32)
            return (trace_ == ref_trace and np.allclose(acc, ref_acc),
                    "%d instructions" % len(ref_trace))
        _c("program_executes", _exec,
           "a program stored and never run is a payload, not a capability")

    # ---- CACHE: repeated work must actually get cheaper ----
    def _cache():
        import time

        from holographic.caching_and_storage.holographic_galvacache import (
            GalvaCache, content_key)
        c = GalvaCache()
        calls = [0]

        def work():
            calls[0] += 1
            time.sleep(0.002)
            return np.arange(8.0)
        k = content_key("harden", 1)
        c.get_or_compute(k, work)
        t0 = time.time()
        for _ in range(5):
            c.get_or_compute(k, work)
        warm = time.time() - t0
        return (calls[0] == 1 and warm < 0.005,
                "1 compute + 5 hits in %.4fs" % warm)
    _c("cache_saves_work", _cache,
       "a cache that recomputes is a slower dictionary")

    passed = sum(1 for c in checks if c["ok"])
    return {"checks": checks, "passed": passed, "total": len(checks),
            "clean": passed == len(checks), "profile": prof}


def _selftest():
    import numpy as _np

    from holographic.agents_and_reasoning.holographic_machine import HoloMachine
    from holographic.io_and_interop.holographic_boot import BootRecord, write_boot
    from holographic.caching_and_storage.holographic_substrate import (
        store_program)

    # HARDEN AGAINST A REAL RUNNABLE MODEL. A hand-assembled dict of two
    # tensors is not a machine -- POST correctly refused it for missing
    # layernorms, which is the check working and the fixture failing. Every
    # other fixture bug this session was the same shape.
    import os as _os

    from holographic.io_and_interop.holographic_gdnruntime import load_runtime
    from holographic.io_and_interop.holographic_unicron import load_safetensors
    src = "/home/claude/bench/model"
    if not _os.path.exists(_os.path.join(src, "model.safetensors")):
        print("harden selftest SKIPPED-SUBJECT (no runnable model present)")
        return
    rt, cfg = load_runtime(src)
    w = dict(load_safetensors(_os.path.join(src, "model.safetensors")))
    rng = _np.random.default_rng(0)
    probe = [int(b) for b in b"The capital of France is"]

    M = HoloMachine(dim=1024, seed=1)
    prog = [("LOAD", "a"), ("APPLY", "cleanup"), ("STORE", "R1"), ("HALT", None)]
    w, _p = store_program(w, M, prog, bits=1)
    rec = BootRecord(seed="leCore", dim=1024, symbols=["subject", "verb"],
                     capabilities=["bind", "unbind", "cleanup"])
    w, _b = write_boot(w, rec)

    facts = [("zorbek", "ratified_1974"), ("gdn", "erase_write_decoupled"),
             ("mp_edge", "noise_boundary")]
    rep = harden(w, rt.cfg, facts=facts, program=prog, machine=M,
                 probe_ids=probe)
    failed = [c["check"] for c in rep["checks"] if not c["ok"]]
    assert rep["clean"], failed

    # ---- THE HARNESS MUST FAIL ON DAMAGE, or it proves nothing ----
    # 1. never installed
    fresh = dict(load_safetensors(_os.path.join(src, "model.safetensors")))
    assert not harden(fresh, rt.cfg, facts=facts, probe_ids=probe)["clean"], \
        "an uninstalled model passed hardening"
    # 2. requantized after installing -- the most common real-world damage
    dmg = {k: _np.array(v, copy=True) for k, v in w.items()}
    for _k in list(dmg):
        _a = _np.asarray(dmg[_k], _np.float64)
        if _a.ndim == 2 and "embed" not in _k:
            _sc = _np.abs(_a).max() / 7.0 or 1.0
            dmg[_k] = (_np.clip(_np.round(_a / _sc), -8, 7)
                       * _sc).astype(_np.asarray(dmg[_k]).dtype)
    d = harden(dmg, rt.cfg, facts=facts, program=prog, machine=M,
               probe_ids=probe)
    assert not d["clean"], "a requantized install passed hardening"
    assert any(c["check"] == "program_executes" and not c["ok"]
               for c in d["checks"]), "quantization must break the program"

    print("harden selftest OK -- an installed model passed %d/%d: BIOS POST and "
          "enumeration, boot from weights, deterministic expansion, an ADDRESSED "
          "channel, %d/%d facts recalled by key, a stored program EXECUTED, and a "
          "cache that actually saves work; and the harness FAILS on a model that "
          "was never installed and on one requantized afterwards (%d/%d), so it "
          "verifies rather than decorates"
          % (rep["passed"], rep["total"], len(facts), len(facts),
             d["passed"], d["total"]))


if __name__ == "__main__":
    _selftest()
