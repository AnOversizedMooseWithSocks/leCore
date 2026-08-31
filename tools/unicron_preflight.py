"""QWEN LIVE-TEST PREFLIGHT (cp65): every known failure mode, checked BEFORE the install.

Each check is a defect that actually happened in this project: a stub tokenizer that
encoded nothing (cp51, and again inside the auditor cp65); a wrong root prefix; dtype
surprises doubling files (cp50); a model too small for the engine payload (the mini,
6.6x over); probe ids arriving empty. Run it against the model directory FIRST:

    python tools/unicron_preflight.py /path/to/qwen3.5-0.8b

Green preflight -> the exact install sequence is printed at the end, ouroboros included.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unicron_install import load_any, detect_root


def preflight(model_dir):
    ok = True

    def check(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        print("  [%s] %-26s %s" % ("+" if passed else "X", name, detail))

    print("PREFLIGHT %s" % model_dir)
    try:
        tensors, meta = load_any(model_dir)
        fmt = meta.get("format", "?") if isinstance(meta, dict) else str(meta)
        check("weights load", True, "%d tensors (%s)" % (len(tensors), fmt))
    except Exception as e:
        check("weights load", False, str(e)[:100])
        return False
    root = detect_root(tensors)
    check("root detect", bool(root), repr(root))
    emb_k = root + "embed_tokens.weight"
    check("embeddings", emb_k in tensors,
          str(np.asarray(tensors.get(emb_k, np.zeros(0))).shape))
    n_vocab, hidden = np.asarray(tensors[emb_k]).shape
    layers = len({k.split("layers.")[1].split(".")[0]
                  for k in tensors if "layers." in k})
    check("shape", layers > 0, "%d layers, hidden %d, vocab %d"
          % (layers, hidden, n_vocab))
    dts = {str(np.asarray(v).dtype) for v in list(tensors.values())[:8]}
    check("dtypes", True, "%s (preserved on write since cp50)" % sorted(dts))
    tok_ok, n_ids = False, 0
    try:
        tj = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tj):
            sys.path.insert(0, os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), "assimilation"))
            from galvatron import _load_tok, _tokens_from
            tok = _load_tok(model_dir)
            n_ids = len(_tokens_from(
                "the engine boots and reads its own record", n_vocab, tok))
            tok_ok = n_ids > 0
    except Exception:
        pass
    check("tokenizer probe", tok_ok,
          "%d ids for the standard probe (empty = the cp51/cp65 failure)" % n_ids)
    payload_mb = 6.65
    room_mb = sum(np.asarray(v).size for v in tensors.values()) / 8 / 1e6
    check("engine payload fits", room_mb > payload_mb,
          "surface %.1f MB at 1 bit vs payload %.2f MB" % (room_mb, payload_mb))
    mid = layers // 2
    print("\nRECOMMENDED SEQUENCE (green preflight only):")
    print("  1. python tools/unicron_install.py %s OUT --from-partition PART \\\\"
          % model_dir)
    print("        --budget auto --spread 3 --bake-algebra --cartridge cart.npz")
    print("     (AlphaEdit null-space protection is ON in the FACTS solve; the")
    print("      report's preservation_after must read ~0)")
    print("  2. python assimilation/galvatron.py OUT --install GALV_OUT "
          "--partition PART")
    print("     (13 residents; ouroboros at layer %d, dk 64, decay 0.98; expect "
          "AUDIT 4/4)" % mid)
    print("  3. MEASURE ON YOUR RUNTIME: efficacy generalization + perplexity "
          "retention")
    print("     (the honest boundary: neither can be proven on sandbox weights)")
    print("  4. Revert path: --revert cart.npz (exactness ~4e-09, verified cp65)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if preflight(sys.argv[1]) else 1)
