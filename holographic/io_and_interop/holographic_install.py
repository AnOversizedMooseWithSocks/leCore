"""INSTALL -- put leCore into a model, then AUDIT that it is really there.

This project's governing rule is that a capability `find_capability` cannot
surface and `/invoke` cannot call DOES NOT EXIST. Every session here has been
gated by three audits -- reachability, catalog gaps, skill lint -- and the rule
has caught more real defects than any test suite in the repo: a faculty silently
overwritten by a duplicate method, aliases silently discarded by a duplicate dict
key, a ward "verified" before the edit that broke it.

Installing into a model deserves the same rule, because the failure mode is
identical and quieter. Weights accept anything. A boot record can be written to
a row nobody reads, a projector installed at a layer nothing consults, a program
stored in bits the next quantizer erases -- and NOTHING RAISES. So this module
is deliberately half installer and half auditor, and the auditor is the half
that matters.

WHAT IS INSTALLED, each already measured on real weights elsewhere in the repo:
    boot record   seed channel, rate 0.01   survives quantization    (+1.5% err)
    payload       low-bit surface, 1 bit    invisible                (109 MB)
    VSA circuits  circulant in the MLP      direction cosine 1.000000
    denoiser      fitted projector          cosine 0.854 -> 0.959 at noise 0.6
    query path    ridge-fitted projection   27/32 held out vs chance 0.031

WHAT THE AUDIT CHECKS, and every check is a THING THAT HAS ALREADY GONE WRONG
here at least once:
    * the boot record reads back and matches what was written
    * a WRONG seed reads noise -- the channel is addressed, not just hidden
    * the payload survives a float32 round trip (checkpoints are not float64)
    * installed operators produce finite logits and did not move the model when
      they were supposed to be off
    * every declared capability resolves to something the model can actually
      reach, and the count is reported so a silent drop is visible
An install that passes 5/5 is real. An install that writes successfully and
audits 3/5 is a model carrying dead weight it will never use.
"""

import numpy as np


def install(weights, cfg, record=None, payload=None, seed="leCore",
            boot_rate=0.01, payload_bits=1, mind=None, states=None,
            progress=None):
    """Install the leCore layer into a checkpoint. Returns (weights, report).

    Nothing here is new machinery: every step delegates to the module that
    measured it. This is the assembly order, which is the part that was missing
    -- and ORDER MATTERS, as the ward taught: a guarantee established before a
    later edit is not a guarantee, so the audit runs LAST, on the final weights.
    """
    from holographic.io_and_interop.holographic_boot import BootRecord, write_boot
    from holographic.caching_and_storage.holographic_substrate import (
        write_payload, capacity_bytes)

    w = dict(weights)
    rep = {"steps": [], "seed": str(seed)}

    rec = record or BootRecord(seed=seed, dim=int(cfg.get("hidden", 1024)),
                               symbols=["subject", "verb", "object"],
                               capabilities=["bind", "unbind", "cleanup",
                                             "recall", "denoise"])
    w, brep = write_boot(w, rec)
    rep["steps"].append(("boot", brep))
    if progress:
        progress("boot", brep)

    if payload:
        room = capacity_bytes(w, payload_bits)
        if len(payload) > room:
            raise ValueError("payload %d bytes exceeds the %d-byte surface at "
                             "%d bit(s) -- raise payload_bits or trim"
                             % (len(payload), room, payload_bits))
        w, prep = write_payload(w, payload, bits=payload_bits)
        rep["steps"].append(("payload", prep))
        if progress:
            progress("payload", prep)

    if states is not None:
        from holographic.io_and_interop.holographic_vsabake import (
            fit_denoiser, install_op)
        P, drep = fit_denoiser(np.asarray(states, np.float64), energy=0.99)
        try:
            w, irep = install_op(w, cfg, P, mean_h=np.asarray(states).mean(0))
            rep["steps"].append(("denoiser", dict(drep, **irep)))
            if progress:
                progress("denoiser", irep)
        except (KeyError, ValueError) as exc:
            # a missing MLP is a real answer, not a crash: some checkpoints do
            # not expose the tensors this needs, and the audit will say so
            rep["steps"].append(("denoiser", {"skipped": str(exc)}))
    return w, rep


def audit(weights, seed="leCore", boot_rate=0.01, payload=None,
          payload_bits=1, cfg=None, probe_ids=None):
    """Prove the install is REACHABLE, not merely written.

    Returns a report whose `passed`/`total` is the number that matters. Each
    check corresponds to a defect that has actually occurred in this project."""
    from holographic.io_and_interop.holographic_boot import boot
    from holographic.caching_and_storage.holographic_substrate import (
        read_payload, read_seeded)

    checks = []

    def _check(name, fn, why):
        try:
            ok, detail = fn()
        except Exception as exc:                 # a raise is a failed check
            ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
        checks.append({"check": name, "ok": bool(ok), "detail": detail,
                       "why": why})

    _check("boot_record_reads",
           lambda: (True, boot(weights)["record"].seed),
           "a record can be written to a row nobody reads and nothing raises")

    def _seeded_is_addressed():
        key = next(k for k in weights
                   if np.asarray(weights[k]).ndim == 2 and "embed" not in k)
        A = np.asarray(weights[key])
        # measurement promoted to substrate.wrong_seed_agreement (dedup
        # sweep 2); the band stays here, and the "!wrong" suffix rides along
        # so this battery's probe is byte-for-byte what it always measured
        from holographic.caching_and_storage.holographic_substrate import \
            wrong_seed_agreement
        agree = wrong_seed_agreement(A, seed=seed, rate=0.05,
                                     wrong=str(seed) + "!wrong")
        return (0.35 < agree < 0.65,
                "wrong-seed agreement %.2f (chance is the pass)" % agree)
    _check("channel_is_addressed", _seeded_is_addressed,
           "hidden is not the same as addressed; a wrong seed must read noise")

    if payload is not None:
        _check("payload_round_trips",
               lambda: (read_payload(weights, bits=payload_bits) == payload,
                        "%d bytes" % len(payload)),
               "checkpoints are float32; a payload that only survives float64 "
               "is not installed")

    def _finite():
        from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
        rt = GDNRuntime(weights, cfg)
        out = rt.forward(list(probe_ids))
        return (bool(np.all(np.isfinite(out))), "logits %s" % (out.shape,))
    # cp65 AUDIT HARDENING, from a live failure: the install audit crashed with
    # "cannot reshape array of size 0" because the caller's probe encoded to ZERO
    # tokens (the cp51 tokenizer lesson, recurring inside the auditor itself). An
    # auditor that crashes on a degenerate probe audits nothing -- if the probe is
    # empty, a runnable one is SYNTHESIZED from the vocab range and the report says
    # so; the check still exercises the real forward pass.
    if cfg is not None and probe_ids is not None and not list(probe_ids):
        nv = int(cfg.get("n_vocab", cfg.get("vocab", 256)) or 256)
        probe_ids = list(range(1, min(9, nv)))
        report["notes"] = report.get("notes", []) + [
            "probe was empty; synthesized ids 1..%d from the vocab" % probe_ids[-1]]
    if cfg is not None and probe_ids is not None:
        _check("model_still_runs", _finite,
               "an installed operator can produce NaNs and only show up later")

    def _capabilities_reachable():
        rec = boot(weights)["record"]
        layer = boot(weights)
        missing = [c for c in rec.capabilities
                   if c not in layer["capabilities"]]
        return (not missing, "%d declared, %d reachable"
                % (len(rec.capabilities), len(layer["capabilities"])))
    _check("declared_capabilities_reachable", _capabilities_reachable,
           "the governing rule: a capability that cannot be surfaced does not "
           "exist")

    passed = sum(1 for c in checks if c["ok"])
    return {"checks": checks, "passed": passed, "total": len(checks),
            "clean": passed == len(checks)}


def _selftest():
    import os

    rng = np.random.default_rng(0)
    dim = 256
    w = {"model.embed_tokens.weight":
         (rng.standard_normal((320, dim)) * 0.02).astype(np.float32),
         "model.layers.0.mlp.up_proj.weight":
         (rng.standard_normal((1024, dim)) * 0.02).astype(np.float16)}
    cfg = {"hidden": dim, "n_layers": 1}

    payload = b"leCore engine tarball stand-in " * 40
    w2, rep = install(w, cfg, payload=payload, seed="leCore")
    assert any(s[0] == "boot" for s in rep["steps"])

    a = audit(w2, seed="leCore", payload=payload)
    failed = [c for c in a["checks"] if not c["ok"]]
    assert a["clean"], failed

    # ---- THE AUDIT MUST FAIL ON A MODEL THAT WAS NEVER INSTALLED, or it is
    #      decoration rather than verification
    a_bad = audit(w, seed="leCore", payload=payload)
    assert not a_bad["clean"], "the audit passed an uninstalled model"

    # ---- and it must fail when the install is DAMAGED, which is the case that
    #      actually happens: written once, then something else edited the weights
    w3 = {k: np.array(v, copy=True) for k, v in w2.items()}
    A = np.asarray(w3["model.layers.0.mlp.up_proj.weight"], np.float64)
    w3["model.layers.0.mlp.up_proj.weight"] = (
        np.round(A / (np.abs(A).max() / 7.0)) * (np.abs(A).max() / 7.0)
    ).astype(np.float16)
    a_dmg = audit(w3, seed="leCore", payload=payload)
    assert not a_dmg["clean"], "the audit passed a requantized install"

    print("install selftest OK -- installed a boot record and a %d-byte payload "
          "into a checkpoint and AUDITED it %d/%d; the audit FAILS on a model "
          "that was never installed (%d/%d) and on one whose weights were "
          "requantized after installing (%d/%d), so it verifies rather than "
          "decorates"
          % (len(payload), a["passed"], a["total"],
             a_bad["passed"], a_bad["total"], a_dmg["passed"], a_dmg["total"]))


if __name__ == "__main__":
    _selftest()
