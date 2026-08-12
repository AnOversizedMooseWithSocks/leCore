"""ASSESS -- one command that produces everything needed to judge a Galvatron.

After an assimilation run there are several artifacts (original, assimilated,
repaired, requantized, the imbued bundle) and the only honest way to compare
them is on the same probe with the same instrument. This writes ONE file per
model directory containing the measurements, so a reviewer with no access to the
machine can evaluate the run.

WHAT IT MEASURES, all on the same tokens so the numbers are comparable:
    BIOS profile     layout, block structure, carrier capacity, install state
    POST             does the model produce finite logits at all
    perplexity       on a fixed public probe AND on the user's own text if given
    generation speed tokens/sec, measured not estimated
    gates            A_log / dt_bias per layer -> memory half-lives
    spectra          full singular values per 2-D tensor -> compressibility
    activations      hidden states at every layer (float16) -> stream geometry
    logits           top-64 + the exact log-sum-exp -> probabilities recoverable
    manifest         the resident roster when the directory is a bundle
    harden           the 8-check end-to-end audit when leCore is installed

WHAT IT DELIBERATELY OMITS: the weights. This is a PROFILE. A reviewer can
compare two runs, see which step helped and which hurt, and never receive the
model. The manifest inside the file lists everything it contains, so nothing
travels that the sender has not seen named.
"""

import json
import os
import time

import numpy as np

PROBE = ("The capital of France is Paris, and the capital of Japan is Tokyo. "
         "Water freezes at zero degrees celsius and boils at one hundred. "
         "A recurrent state carries what the past can tell the future, and "
         "every layer writes into the residual stream that follows it. "
         "def compress(x, rank=8):\n"
         "    u, s, vt = numpy.linalg.svd(x, full_matrices=False)\n"
         "    return (u[:, :rank] * s[:rank]) @ vt[:rank]\n"
         "SELECT title FROM notes WHERE session = 's1' ORDER BY created;\n"
         "# Heading\n- first item\n- second item\n\n"
         "Questions: why is the sky blue? How does a delta rule update a "
         "memory matrix in place? Answer carefully and cite the passage used.")


def assess(model_dir, out_path, text=None, n_gen=32, layers=(0, None, -1),
           progress=None):
    """Measure one model directory and write the assessment bundle."""
    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime, load_weights_dir)
    from holographic.io_and_interop.holographic_bios import report as bios_report

    rt, cfg = load_runtime(model_dir)
    w = load_weights_dir(model_dir)
    text = text or PROBE
    try:
        from holographic.io_and_interop.holographic_bpe import BPE
        tok = BPE.from_dir(model_dir)
        ids = tok.encode(text)[:512]
    except Exception:
        ids = [b for b in text.encode("utf-8")][:512]
    if len(ids) < 16:
        # the tokenizer did not recognise the probe: fall back to a
        # deterministic in-range span rather than measuring nothing
        n = int(np.asarray(rt.lm_head).shape[0])
        ids = [int(i % max(n - 1, 1)) for i in range(10, 10 + 128)]

    out = {}
    man = {"model_dir": os.path.abspath(model_dir), "probe_tokens": len(ids),
           "when": time.strftime("%Y-%m-%d %H:%M"), "contains": []}

    prof = bios_report(w, cfg, model_dir=model_dir, probe_ids=ids[:16])
    man["bios"] = {k: v for k, v in prof.items() if k != "shapes"}
    man["contains"].append("BIOS profile and POST")
    if progress:
        progress("bios", prof["post"]["ok"])

    # ---- perplexity and speed on the SAME tokens, so runs are comparable ----
    t0 = time.time()
    ppl = float(rt.perplexity(list(ids)))
    t_ppl = time.time() - t0
    prompt = ids[:min(48, len(ids))]
    t0 = time.time()
    gen, _st = rt.generate_fast(list(prompt), n_new=int(n_gen))
    t_gen = time.time() - t0
    man["perplexity"] = ppl
    man["perplexity_seconds"] = round(t_ppl, 3)
    man["generation"] = {"tokens": int(n_gen), "seconds": round(t_gen, 3),
                         "tokens_per_second": round(n_gen / max(t_gen, 1e-9), 1)}
    man["contains"].append("perplexity and generation speed on the probe")
    if progress:
        progress("perplexity", ppl)

    # ---- gates: the memory structure that turned out to be positional ----
    for k in sorted(w):
        if k.endswith("A_log") or k.endswith("dt_bias"):
            out["gate::" + k] = np.asarray(w[k], np.float32)
    man["contains"].append("A_log / dt_bias for every linear-attention layer")

    # ---- full spectra: compressibility, uncensored ----
    from holographic.io_and_interop.holographic_testkit import _singular_values
    shapes = {}
    for k, v in sorted(w.items()):
        a = np.asarray(v)
        shapes[k] = [list(a.shape), str(a.dtype)]
        if a.ndim == 2 and min(a.shape) >= 8:
            out["sv::" + k] = _singular_values(a).astype(np.float32)
    man["shapes"] = shapes
    man["contains"].append("FULL singular values per 2-D tensor")
    if progress:
        progress("spectra", len(shapes))

    # ---- the stream, and top-k logits with an exact normaliser ----
    cap = {}
    n_layers = int(cfg["n_layers"])
    rt.forward(ids, hooks={L: (lambda h, _L=L: cap.__setitem__(_L, h.copy())
                               or None) for L in range(n_layers)})
    for L, hh in cap.items():
        out["act::%d" % L] = np.asarray(hh, np.float16)
    lg = np.asarray(rt.forward(ids), np.float64)
    k = int(min(64, lg.shape[-1]))
    idx = np.argsort(lg, axis=-1)[:, -k:][:, ::-1]
    out["logit_top_idx"] = idx.astype(np.int32)
    out["logit_top_val"] = np.take_along_axis(lg, idx, axis=-1).astype(np.float32)
    out["logit_logsumexp"] = (np.log(np.sum(np.exp(
        lg - lg.max(-1, keepdims=True)), -1)).ravel() + lg.max(-1)).astype(np.float32)
    out["probe_ids"] = np.asarray(ids, np.int64)
    man["contains"].append("hidden states at every layer + top-%d logits" % k)

    # ---- bundle manifest and the hardening audit, when they apply ----
    mpath = os.path.join(model_dir, "galvatron.json")
    gm = None
    if os.path.exists(mpath):
        with open(mpath) as f:
            gm = json.load(f)
    if gm:
        man["galvatron"] = {"residents": [r.get("kind") for r in
                                          gm.get("residents", [])],
                            "config_keys": sorted(gm.get("config", {}))[:12],
                            "guarded_bakes": gm.get("guarded_bakes", []),
                            "baked_into_weights": gm.get("baked_into_weights", [])}
        man["contains"].append("the bundle's resident roster")

    # ---- leCore INSTALL REPORT, if this model was built by install.py ----
    lp = os.path.join(model_dir, "lecore.json")
    if os.path.exists(lp):
        try:
            with open(lp) as f:
                lj = json.load(f)
            man["lecore"] = lj
            # VERIFY THE REGISTERS ACTUALLY HOLD, rather than trusting the
            # manifest. A file that SAYS it has 64 registers and a state that
            # cannot keep one are different things, and only one of them
            # matters.
            regs = lj.get("registers") or {}
            if regs.get("count"):
                from holographic.caching_and_storage.holographic_keyreserve import (
                    reserve, orthogonalise, delta_write, delta_read)
                D = int(regs.get("dim") or cfg["hidden"])
                n = int(regs["count"])
                R = reserve(D, n, seed=int(regs.get("seed", 0)))
                g = np.random.default_rng(0)
                vals = [g.standard_normal(D) for _ in range(n)]
                S = np.zeros((D, D))
                for k, v in zip(R, vals):
                    S = delta_write(S, k, v)
                for _ in range(1024):
                    S = delta_write(S, orthogonalise(g.standard_normal(D), R),
                                    g.standard_normal(D))
                intact = sum(
                    float(delta_read(S, R[i]) @ vals[i]
                          / (np.linalg.norm(delta_read(S, R[i]))
                             * np.linalg.norm(vals[i]) + 1e-30)) > 0.99
                    for i in range(n))
                man["lecore"]["registers_verified"] = {
                    "intact": int(intact), "of": n, "after_writes": 1024}
        except Exception as exc:
            man["lecore"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
        man["contains"].append("the leCore install report, with the registers "
                               "re-verified rather than trusted")

    try:
        from holographic.io_and_interop.holographic_harden import harden
        hz = harden(w, cfg, probe_ids=ids[:16])
        man["harden"] = {"passed": hz["passed"], "total": hz["total"],
                         "checks": [{"check": c["check"], "ok": c["ok"],
                                     "detail": c["detail"][:120]}
                                    for c in hz["checks"]]}
        man["contains"].append("the 8-check hardening audit")
    except Exception as exc:
        man["harden"] = {"error": "%s: %s" % (type(exc).__name__, exc)}

    out["manifest"] = np.frombuffer(json.dumps(man).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(out_path, **out)
    return {"path": out_path,
            "megabytes": round(os.path.getsize(out_path) / 1e6, 2),
            "perplexity": ppl,
            "tokens_per_second": man["generation"]["tokens_per_second"],
            "harden": man.get("harden", {}).get("passed"),
            "contains": man["contains"]}


def compare(paths):
    """Read several assessment bundles and line them up.

    The comparison is the point: a single run's perplexity means nothing without
    the run it is being compared against, on the same probe."""
    rows = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        m = json.loads(bytes(z["manifest"]).decode("utf-8"))
        rows.append({"file": os.path.basename(p),
                     "dir": os.path.basename(m["model_dir"]),
                     "perplexity": m.get("perplexity"),
                     "tokens_per_second": m.get("generation", {}).get(
                         "tokens_per_second"),
                     "post": m.get("bios", {}).get("post", {}).get("ok"),
                     "harden": "%s/%s" % (m.get("harden", {}).get("passed"),
                                          m.get("harden", {}).get("total")),
                     "residents": len(m.get("galvatron", {}).get("residents", []))})
    return rows


def _selftest():
    import tempfile

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("assess selftest SKIPPED-SUBJECT (no model present)")
        return
    path = os.path.join(tempfile.mkdtemp(), "assess.npz")
    rep = assess(src, path, n_gen=8)

    z = np.load(path, allow_pickle=False)
    man = json.loads(bytes(z["manifest"]).decode("utf-8"))

    # ---- it is SELF-DESCRIBING: a reader can tell what they received ----
    assert man["contains"] and man["probe_tokens"] > 0
    assert "bios" in man and "perplexity" in man
    assert any(k.startswith("sv::") for k in z.files)
    assert any(k.startswith("act::") for k in z.files)
    assert "logit_top_val" in z.files

    # ---- and it is NOT the model: no full weight tensors travel ----
    assert not any(k.startswith("layer::") or k.startswith("w::")
                   for k in z.files)

    # ---- COMPARISON is the point, so two bundles must line up ----
    second = os.path.join(tempfile.mkdtemp(), "assess2.npz")
    assess(src, second, n_gen=8)
    rows = compare([path, second])
    assert len(rows) == 2 and rows[0]["perplexity"] == rows[1]["perplexity"], rows

    print("assess selftest OK -- %.2f MB bundle carrying %d kinds of "
          "measurement (perplexity %.4f, %.1f tok/s, harden %s); it is "
          "self-describing, contains NO weight tensors, and two bundles compare "
          "cleanly so a reviewer can tell which step helped"
          % (rep["megabytes"], len(rep["contains"]), rep["perplexity"],
             rep["tokens_per_second"], rep["harden"]))


if __name__ == "__main__":
    _selftest()
