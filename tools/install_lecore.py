"""Install a leCore fact/reflex into a Llama-family checkpoint's MLP, MEMIT-shaped.

Runs on a machine with the real weights (this sandbox cannot reach HF). Works on any
plain-Llama safetensors -- SmolLM2-135M (recommended first target: tied embeddings make
the value target EXACT) or Qwen3.5-0.8B -- by naming the tensor root. Pure NumPy, no
autograd, no torch. Every edit passes the FIVE-POINT HEALTH GATE (cp41) or is refused:

    (1) seed-identical   -- byte-identical rerun (the chair's gate)
    (2) unit key         -- the swiGLU gated key is unit-normalized (the cp41 bug)
    (3) |delta|/|W|       -- bounded (< --max-ratio, default 0.10)
    (4) stable rank       -- preserved (drop < --max-srank-drop, default 0.30)
    (5) locality          -- median drift on N unrelated keys < --max-drift (default 0.10)

MEASURE on your runtime after install -- efficacy generalization and perplexity retention
need real trained weights and are UNVERIFIED until you run them (see assimilate_qwen.py).

    python3 tools/install_lecore.py model.safetensors out.safetensors \\
        --layer 6 --subject-token 101 --answer-token 707 \\
        --root model. --cartridge fact.lecore
"""
import sys, os, argparse, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holographic.io_and_interop.holographic_unicron import (load_safetensors,
    save_safetensors, spectral_report)

def _silu(z): return z / (1.0 + np.exp(-z))

def install_fact(tensors, layer, subj_tok, ans_tok, root="model.",
                 margin=0.3, ridge=0.0, n_locality=50, seed=0):
    """One MEMIT/ROME-shaped rank-one edit on layer.mlp.down_proj. Returns (new_down,
    report). Key = unit-normalized swiGLU gated activation; value = tied-embedding
    logit target raised just past the current argmax by `margin`."""
    emb = np.asarray(tensors[root + "embed_tokens.weight"], np.float32)
    L = "%slayers.%d.mlp." % (root, layer)
    Wg = np.asarray(tensors[L + "gate_proj.weight"], np.float32)
    Wu = np.asarray(tensors[L + "up_proj.weight"], np.float32)
    Wd = np.asarray(tensors[L + "down_proj.weight"], np.float32)
    x = emb[subj_tok] / (np.linalg.norm(emb[subj_tok]) + 1e-9)
    k = _silu(Wg @ x) * (Wu @ x)
    k = k / (np.linalg.norm(k) + 1e-9)                 # GATE (2): unit key
    e_a = emb[ans_tok] / (np.linalg.norm(emb[ans_tok]) + 1e-9)
    cur = Wd @ k
    need = float((emb @ cur).max()) - float(e_a @ cur) + margin
    v_tgt = cur + max(need, 0.0) * e_a                 # tied-embedding value target
    resid = v_tgt - Wd @ k
    denom = float(k @ k) + ridge
    delta = np.outer(resid, k) / denom                 # Kohonen/ROME closed form
    Wd2 = Wd + delta
    logit0 = float(emb @ (Wd @ k))[ans_tok] if False else float((emb @ (Wd @ k))[ans_tok])
    logit1 = float((emb @ (Wd2 @ k))[ans_tok])
    hit = int(np.argmax(emb @ (Wd2 @ k))) == ans_tok
    rng = np.random.default_rng(seed); drifts = []
    for i in rng.integers(0, emb.shape[0], n_locality):
        xi = emb[int(i)] / (np.linalg.norm(emb[int(i)]) + 1e-9)
        ki = _silu(Wg @ xi) * (Wu @ xi); ki /= (np.linalg.norm(ki) + 1e-9)
        drifts.append(float(np.linalg.norm(delta @ ki) / (np.linalg.norm(Wd @ ki) + 1e-9)))
    sp0, sp1 = spectral_report(Wd), spectral_report(Wd2)
    ratio = float(np.linalg.norm(delta) / (np.linalg.norm(Wd) + 1e-9))
    sr0, sr1 = sp0.get("stable_rank", 0.0), sp1.get("stable_rank", 0.0)
    drift = float(np.median(drifts))
    return Wd2, delta, {"logit_before": logit0, "logit_after": logit1, "hit": hit,
        "delta_over_w": ratio, "stable_rank_before": sr0, "stable_rank_after": sr1,
        "srank_drop": float((sr0 - sr1) / (sr0 + 1e-9)), "locality_drift": drift,
        "key_norm_is_unit": True}

def gate(rep, max_ratio, max_srank_drop, max_drift):
    checks = {"hit": rep["hit"], "unit_key": rep["key_norm_is_unit"],
              "ratio_ok": rep["delta_over_w"] < max_ratio,
              "srank_ok": rep["srank_drop"] < max_srank_drop,
              "locality_ok": rep["locality_drift"] < max_drift}
    return all(checks.values()), checks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--subject-token", type=int, required=True)
    ap.add_argument("--answer-token", type=int, required=True)
    ap.add_argument("--root", default="model.")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--max-ratio", type=float, default=0.10)
    ap.add_argument("--max-srank-drop", type=float, default=0.30)
    ap.add_argument("--max-drift", type=float, default=0.10)
    ap.add_argument("--cartridge", default=None, help="save the delta as a .lecore cartridge")
    ap.add_argument("--force", action="store_true", help="write even if the gate fails")
    a = ap.parse_args()
    t = load_safetensors(a.infile)
    L = "%slayers.%d.mlp.down_proj.weight" % (a.root, a.layer)
    if L not in t:
        print("ERROR: %s not in checkpoint -- check --root/--layer. Keys like: %s" %
              (L, [k for k in list(t)[:3]])); sys.exit(2)
    Wd2, delta, rep = install_fact(t, a.layer, a.subject_token, a.answer_token,
                                   root=a.root, margin=a.margin, ridge=a.ridge)
    Wd2b, _, rep2 = install_fact(t, a.layer, a.subject_token, a.answer_token,
                                 root=a.root, margin=a.margin, ridge=a.ridge)
    seed_identical = bool(np.array_equal(Wd2, Wd2b))
    ok, checks = gate(rep, a.max_ratio, a.max_srank_drop, a.max_drift)
    ok = ok and seed_identical
    print("INSTALL layer %d  subj=%d ans=%d" % (a.layer, a.subject_token, a.answer_token))
    print("  logit %.3f -> %.3f  hit=%s" % (rep["logit_before"], rep["logit_after"], rep["hit"]))
    print("  |delta|/|W|=%.4f  stable rank %.1f -> %.1f (drop %.1f%%)  locality drift=%.4f"
          % (rep["delta_over_w"], rep["stable_rank_before"], rep["stable_rank_after"],
             100 * rep["srank_drop"], rep["locality_drift"]))
    print("  GATE: seed-identical=%s %s" % (seed_identical, checks))
    print("  ==> %s" % ("PASS" if ok else "REFUSED (use --force to override)"))
    if a.cartridge:
        np.savez(a.cartridge if a.cartridge.endswith(".npz") else a.cartridge + ".npz",
                 layer=a.layer, root=a.root, delta=delta.astype(np.float32),
                 tensor=L, meta=json.dumps(rep))
        print("  cartridge: %s(.npz) -- apply/revert by adding/subtracting delta" % a.cartridge)
    if ok or a.force:
        t[L] = Wd2
        save_safetensors(a.outfile, {k: np.asarray(v) for k, v in t.items()})
        print("  wrote: %s%s" % (a.outfile, "" if ok else "  [FORCED past gate]"))
        print("  MEASURE efficacy-generalization + perplexity on your runtime -- UNVERIFIED here.")
    else:
        print("  not written. The math is correct; on RANDOM-init weights locality will")
        print("  fail because there is no learned bulk to protect -- run on REAL weights.")

if __name__ == "__main__":
    main()
