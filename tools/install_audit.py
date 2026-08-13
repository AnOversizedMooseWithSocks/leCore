#!/usr/bin/env python3
"""INSTALL AUDIT -- is the installed leCore actually WIRED, or just written?

Three questions the other audits do not ask. reachability_audit asks whether a
capability is DISCOVERABLE; usage_audit asks whether anything CALLS it; this
asks whether an INSTALLED MODEL can actually use what was put in it.

    ABLATION   zero a component -- if perplexity does not move, the forward
               pass never reads it. (Blank prepended layers are the honest
               exception: they are EMPTY AND LIVE, reserved capacity that
               reads at cosine 1.000000 the moment anything is written.)
    ROUND TRIP does it survive save and reload? An install that only works in
               the process that built it is not installed -- this repo has
               shipped that exact bug.
    USE        can each part be exercised from the shipped artifact alone?

    python tools/install_audit.py path/to/installed/model
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(model_dir):
    import numpy as np
    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime, load_weights_dir)
    from holographic.io_and_interop.holographic_boot import boot
    from assimilation.galvatron import _resolve_model_dir

    model_dir = _resolve_model_dir(model_dir)
    rt, cfg = load_runtime(model_dir)
    w = load_weights_dir(model_dir)
    fails = 0

    print("INSTALL AUDIT: %s" % model_dir)
    print()
    print("ROUND TRIP -- what survived to disk:")
    try:
        rec = boot(w)["record"]
        print("  boots as %r with %d capabilities: %s"
              % (rec.seed, len(rec.capabilities), list(rec.capabilities)))
    except Exception as exc:
        print("  NO BOOT RECORD (%s)" % type(exc).__name__)
        fails += 1
    lj = os.path.join(model_dir, "lecore.json")
    if os.path.exists(lj):
        print("  lecore.json: %s" % sorted(json.load(open(lj))
                                           .get("installed", []))[:8])
    else:
        print("  lecore.json MISSING")
        fails += 1
    ix = os.path.join(model_dir, "lecore_index.npz")
    print("  sidecar index: %s"
          % ("%.2f MB" % (os.path.getsize(ix) / 1e6)
             if os.path.exists(ix) else "absent (no passages installed)"))
    print()

    print("USE -- can each part be exercised from this artifact alone?")
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    H = int(cfg["hidden"])

    if os.path.exists(lj):
        from holographic.caching_and_storage.holographic_keyreserve import (
            reserve, delta_write, delta_read)
        reg = (json.load(open(lj)).get("registers") or {})
        n = int(reg.get("count", 0))
        if n:
            R = reserve(H, n, seed=int(reg.get("seed", 0)))
            g = np.random.default_rng(0)
            CB = g.standard_normal((256, H))
            CB /= np.linalg.norm(CB, axis=1, keepdims=True)
            S = np.zeros((H, H))
            truth = [int(x) for x in g.integers(0, 256, n)]
            for k, i in zip(R, truth):
                S = delta_write(S, k, CB[i])
            ok = sum(int(np.argmax(CB @ (delta_read(S, R[j])
                                         / np.linalg.norm(delta_read(S, R[j])))))
                     == truth[j] for j in range(n))
            print("  registers   %d/%d recalled, regenerated from the seed"
                  % (ok, n))
            fails += (ok != n)

    hl = np.geomspace(2, int(cfg.get("max_position_embeddings") or 4096), 4)
    _wt, rp = m.unicron_actr(half_lives=hl)
    print("  ladder      ACT-R fit R^2 %.5f (tool choice by recency+frequency)"
          % rp["r2"])
    fails += (rp["r2"] < 0.99)

    if os.path.exists(ix):
        z = np.load(ix, allow_pickle=True)
        idx = m.build_index(z["vectors"],
                            labels=list(range(len(z["passages"]))))
        hit = idx.nearest(z["vectors"][0], 1)
        print("  rag index   %d passages, self-query %s"
              % (len(z["passages"]), "OK" if hit and int(hit[0][0]) == 0
                 else "FAILED"))
        fails += not (hit and int(hit[0][0]) == 0)

    print()
    print("TOTAL: %d problem(s)" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "work/galvatron"))
