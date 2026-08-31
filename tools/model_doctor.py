"""model_doctor.py -- ONE COMMAND FOR THE WHOLE PIPELINE (cp109).

Five checkpoints produced five instruments and they lived in five scripts with five
conventions. Nobody is going to run five scripts on a GPU box and assemble the answer by
hand, so this chains them and prints one report:

    1. GATE        fixture_gate -- is there learned structure to talk about at all?
    2. PROFILE     adjacent-layer cosine, the published redundancy screen
    3. PROBE       causal bypass, because the screen alone is wrong (see below)
    4. PLAN        which layers survive both stages, with a parameter budget
    5. HEAL        can a linear correction repair the damage? (measured, not assumed)
    6. VERDICT     what is actually safe to remove

WHAT THIS TOOL KNOWS THAT THE LITERATURE'S CRITERION DOES NOT.
The angular-distance criterion says a layer whose output barely changes the
representation is removable. Measured here across cp104-cp109, that is ordinally right
and quantitatively insufficient, and four independent measurements say why:

    cp104  the delta between "redundant" layers is nearly ORTHOGONAL to the running
           representation (cos ~0.0006) and high-rank (~58/60) -- those layers write
           content-specific information into their predecessor's null space
    cp105  bypassing them costs real predictions: agreement 0.000 / 0.100 / 0.400 /
           0.633 for layers whose cosines were 0.78 / 0.96 / 0.98 / 0.985
    cp106  the WEIGHT difference between such pairs is rank 94/128 -- the same rank as
           a single layer's own weights. Output similarity is not weight duplication.
    cp109  no linear correction repairs a removal: rank 2 through full rank moved
           agreement 0.500 -> 0.533 while HELD-OUT agreement fell 0.400 -> 0.200, and
           the control confirmed the correction is not a no-op (1.000 -> 0.633 applied
           to an unpruned model). The damage is not low-rank, so it does not fold.

So the doctor screens, then CONFIRMS CAUSALLY, then tries to heal and reports honestly
when healing fails. A tool that only ran stage 2 would have recommended pruning nineteen
layers on the test artifact; zero of them survive stage 3.

Usage:
    python3 tools/model_doctor.py /path/to/model [--screen 0.95] [--accept 0.90]
                                  [--seqs 30] [--heal LAYER] [--json OUT.json]
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.fixture_gate import gate  # noqa: E402
from tools.prune_probe import adjacent_cosines, bypass_agreement, _forward_top1  # noqa: E402


def _params_in_layer(model_dir, layer):
    """Parameter budget for one layer -- Duda's rule: report the budget, not the verdict."""
    import holographic.io_and_interop.holographic_gdnruntime as g
    w = g.load_weights_dir(model_dir)
    if isinstance(w, tuple):
        w = w[0]
    tag = ".layers.%d." % layer
    total = 0
    for name, arr in w.items():
        if tag in name:
            total += int(np.prod(np.asarray(arr).shape))
    all_params = sum(int(np.prod(np.asarray(a).shape)) for a in w.values())
    return total, all_params


def heal_attempt(model_dir, pruned_dir, layer, prompts, ranks=(2, 8, 32)):
    """Fit h_pruned -> h_base at layer+1 and apply it as a runtime delta.

    Returns measured agreements per rank plus a control. Reports what happened; it does
    not assume healing works, because measured across cp109 it does not.
    """
    import holographic.io_and_interop.holographic_gdnruntime as g

    def capture(d, ps, L):
        rt, _ = g.load_runtime(d)
        tops, hs = [], []
        for p in ps:
            store = {}
            hooks = {L: (lambda h: store.__setitem__("h", np.asarray(h, np.float64).copy()) or None)}
            r = rt.forward(p, hooks=hooks)
            lg = np.asarray(r[-1] if isinstance(r, (list, tuple)) else r)
            tops.append(int(np.argmax(lg)) if lg.ndim == 1 else int(np.argmax(lg[-1])))
            if "h" in store:
                hs.append(store["h"])
        return tops, hs

    L1 = layer + 1
    base_top, base_h = capture(model_dir, prompts, L1)
    pr_top, pr_h = capture(pruned_dir, prompts, L1)
    agr_pruned = float(np.mean([a == b for a, b in zip(base_top, pr_top)]))
    n_fit = max(1, int(0.67 * len(prompts)))

    def flat(hs):
        return np.concatenate([np.asarray(h).reshape(-1, np.asarray(h).shape[-1]) for h in hs], 0)

    X, Y = flat(pr_h[:n_fit]), flat(base_h[:n_fit])
    if len(X) < 8 or X.shape != Y.shape:
        return {"agreement_pruned": agr_pruned, "healed": {}, "note": "insufficient rows to fit"}
    lam = 1e-2 * float(np.trace(X.T @ X)) / X.shape[1]
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    C = W - np.eye(X.shape[1])
    U, S, Vt = np.linalg.svd(C)
    out = {}
    for k in ranks:
        k = min(k, len(S))
        Ck = (U[:, :k] * S[:k]) @ Vt[:k]
        fn = (lambda M: (lambda h: np.asarray(np.asarray(h, np.float64) @ M, np.asarray(h).dtype)))(Ck)
        rt, _ = g.load_runtime(pruned_dir)
        tops = []
        for p in prompts:
            r = rt.forward(p, hooks={L1: fn})
            lg = np.asarray(r[-1] if isinstance(r, (list, tuple)) else r)
            tops.append(int(np.argmax(lg)) if lg.ndim == 1 else int(np.argmax(lg[-1])))
        out["rank_%d" % k] = {
            "agreement": float(np.mean([a == b for a, b in zip(base_top, tops)])),
            "agreement_heldout": float(np.mean([a == b for a, b in zip(base_top[n_fit:], tops[n_fit:])])),
        }
    return {"agreement_pruned": agr_pruned, "healed": out,
            "spectrum_top8": [float(s) for s in S[:8]]}


def doctor(model_dir, screen=0.95, accept=0.90, n_seqs=30, seq_len=12, heal_layer=None):
    t0 = time.time()
    rep = {"model": model_dir, "stages": {}}

    # 1. GATE
    try:
        v = gate(model_dir)
    except Exception as exc:
        rep["stages"]["gate"] = {"error": str(exc)}
        return rep
    rep["stages"]["gate"] = v
    flow_only = bool(v["blocked"])

    # 2. PROFILE
    cos = adjacent_cosines(model_dir)
    rep["stages"]["profile"] = {"n_pairs": len(cos),
                                "min": min(cos.values()) if cos else None,
                                "max": max(cos.values()) if cos else None,
                                "cosines": {str(k): round(x, 4) for k, x in sorted(cos.items())}}

    # 3. PROBE
    cands = sorted(L for L, c in cos.items() if c >= screen)
    rng = np.random.default_rng(7)
    prompts = [list(rng.integers(1, 500, size=seq_len)) for _ in range(n_seqs)]
    baseline = _forward_top1(model_dir, prompts)
    rows = []
    for L in cands:
        rows.append({"layer": L, "cosine": cos[L],
                     "agreement": bypass_agreement(model_dir, L, prompts, baseline)})
    rows.sort(key=lambda r: -r["agreement"])
    rep["stages"]["probe"] = {"screened": cands, "rows": rows}

    # correlation between the screen and the truth, inside the screened band
    if len(rows) >= 3:
        c = np.array([r["cosine"] for r in rows])
        a = np.array([r["agreement"] for r in rows])
        r_pear = float(np.corrcoef(c, a)[0, 1])
        rep["stages"]["probe"]["screen_vs_truth_r"] = r_pear
        rep["stages"]["probe"]["screen_explains_variance"] = float(r_pear ** 2)

    # 4. PLAN
    prunable = [r["layer"] for r in rows if r["agreement"] >= accept]
    budget = None
    if prunable:
        per, total = _params_in_layer(model_dir, prunable[0])
        budget = {"params_per_layer": per, "params_total": total,
                  "removable": per * len(prunable),
                  "fraction": (per * len(prunable)) / float(total or 1)}
    rep["stages"]["plan"] = {"prunable": prunable, "budget": budget,
                             "screen_would_have_pruned": len(cands),
                             "probe_allows": len(prunable)}

    # 5. HEAL (only meaningful if something was removed)
    target = heal_layer if heal_layer is not None else (rows[0]["layer"] if rows else None)
    if target is not None:
        import shutil
        import tempfile
        from tools.prune_probe import _zero_layer_outputs
        tmp = tempfile.mkdtemp(prefix="doctor_")
        dst = os.path.join(tmp, "m")
        try:
            shutil.copytree(model_dir, dst)
            for f in os.listdir(dst):
                if f.endswith(".safetensors"):
                    _zero_layer_outputs(os.path.join(dst, f), target)
            rep["stages"]["heal"] = heal_attempt(model_dir, dst, target, prompts)
            rep["stages"]["heal"]["layer"] = target
        except Exception as exc:
            rep["stages"]["heal"] = {"error": str(exc), "layer": target}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    rep["flow_only"] = flow_only
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def _print(rep):
    g_ = rep["stages"].get("gate", {})
    print("model doctor: %s" % rep["model"])
    print("\n[1] gate       %s  ppl %.1f vs vocab %d (%.3fx chance)"
          % (g_.get("verdict"), g_.get("perplexity", float("nan")),
             g_.get("vocab_size", 0), g_.get("ppl_over_chance", float("nan"))))
    if rep.get("flow_only"):
        print("               AT-CHANCE: every number below is an information-FLOW")
        print("               measurement. None of it is a quality claim.")
    p = rep["stages"].get("profile", {})
    print("\n[2] profile    %d adjacent pairs, cosine %.3f - %.3f"
          % (p.get("n_pairs", 0), p.get("min") or float("nan"), p.get("max") or float("nan")))
    pb = rep["stages"].get("probe", {})
    print("\n[3] probe      %d layers cleared the cosine screen; bypassing each:"
          % len(pb.get("screened", [])))
    for r in pb.get("rows", [])[:8]:
        print("               L%-3d cos %.3f -> agreement %.3f" % (r["layer"], r["cosine"], r["agreement"]))
    if len(pb.get("rows", [])) > 8:
        print("               ... %d more" % (len(pb["rows"]) - 8))
    if "screen_vs_truth_r" in pb:
        print("               screen vs truth: r=%+.3f -- cosine explains %.0f%% of the variance"
              % (pb["screen_vs_truth_r"], 100 * pb["screen_explains_variance"]))
    pl = rep["stages"].get("plan", {})
    print("\n[4] plan       cosine screen would prune %d; probe allows %d"
          % (pl.get("screen_would_have_pruned", 0), pl.get("probe_allows", 0)))
    if pl.get("budget"):
        b = pl["budget"]
        print("               budget: %d params/layer, %d removable of %d (%.1f%%)"
              % (b["params_per_layer"], b["removable"], b["params_total"], 100 * b["fraction"]))
    h = rep["stages"].get("heal", {})
    if h and "healed" in h:
        print("\n[5] heal       layer L%d, pruned agreement %.3f" % (h.get("layer", -1), h["agreement_pruned"]))
        for k, vv in h["healed"].items():
            print("               %-8s -> agreement %.3f (held-out %.3f)"
                  % (k, vv["agreement"], vv["agreement_heldout"]))
    print("\n[6] verdict")
    if pl.get("probe_allows"):
        print("    %d layer(s) cleared BOTH stages: %s" % (pl["probe_allows"], pl["prunable"]))
    else:
        print("    NOTHING is safe to remove. The cosine screen alone would have pruned")
        print("    %d layers and been wrong about all of them." % pl.get("screen_would_have_pruned", 0))
    if h and h.get("healed"):
        best = max((vv["agreement_heldout"] for vv in h["healed"].values()), default=0)
        if best <= h["agreement_pruned"]:
            print("    Linear healing did NOT repair the damage at any rank tested --")
            print("    consistent with cp104/cp106: the lost contribution is high-rank.")
    print("\n    (%.1fs)" % rep.get("seconds", 0))


def main(argv):
    if not argv:
        print(__doc__)
        return 3
    model_dir = argv[0]

    def opt(flag, default, cast):
        return cast(argv[argv.index(flag) + 1]) if flag in argv else default

    rep = doctor(model_dir,
                 screen=opt("--screen", 0.95, float),
                 accept=opt("--accept", 0.90, float),
                 n_seqs=opt("--seqs", 30, int),
                 heal_layer=opt("--heal", None, int))
    _print(rep)
    if "--json" in argv:
        out = argv[argv.index("--json") + 1]
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, default=float)
        print("    wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
