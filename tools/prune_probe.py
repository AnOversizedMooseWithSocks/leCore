"""prune_probe.py -- THE TWO-STAGE PRUNING CRITERION (cp107), earned in cp104-cp106.

The published criterion for removable transformer layers is angular distance between
adjacent hidden states: high cosine => the layer barely changes the representation =>
prune it. Measured here (cp105) that criterion is ORDINALLY right and QUANTITATIVELY
insufficient. Bypassing one layer at a time and measuring top-1 agreement gave:

    L02 (active, cos ~0.78)      agreement 0.000
    L10 (cos ~0.96, "redundant") agreement 0.100   <-- cleared stage 1, destroyed by removal
    L20 (cos ~0.98)              agreement 0.400
    L24 (cos ~0.985)             agreement 0.633
    L27 (final)                  agreement 0.067

cp104 explains why: the delta between "redundant" layers is nearly ORTHOGONAL to the
running representation (cos ~ 0.0006) and high-rank (~58/60), i.e. those layers write
content-specific information into the null space of their predecessor. cp106 confirms it
from the weights: the rank-at-90%-energy of the DIFFERENCE between such layer pairs equals
that of a single layer's own weights (94/128 vs 94/128). Near-identical outputs are
produced by genuinely different weights doing genuinely different work.

So: screen with cosine (cheap, O(layers) forwards), then CONFIRM CAUSALLY with a bypass
probe (seconds) before believing any layer is prunable.

IMPORTANT -- what the probe measures. Top-1 agreement measures OUTPUT CHANGE, i.e.
information flow. It is a quality claim only when the artifact itself models anything;
run tools/fixture_gate.py first. On an at-chance fixture the ordering is still meaningful
(flow), the quality reading is not.

Usage:
    python3 tools/prune_probe.py /path/to/model [--screen 0.95] [--accept 0.90] [--seqs 30]
"""
import os
import sys
import json
import shutil
import struct
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _forward_top1(model_dir, prompts):
    from holographic.io_and_interop import holographic_gdnruntime as g
    rt, _ = g.load_runtime(model_dir)
    out = []
    for p in prompts:
        r = rt.forward(p)
        logits = np.asarray(r[-1] if isinstance(r, (list, tuple)) else r)
        if logits.ndim == 1:
            out.append(int(np.argmax(logits)))
        else:
            out.append(int(np.argmax(logits[-1])))
    return out


def adjacent_cosines(model_dir, n_probe=20, seq_len=16):
    """Stage 1: the cheap screen. Mean cosine between adjacent layers' hidden states."""
    from holographic.io_and_interop import holographic_gdnruntime as g
    rt, cfg = g.load_runtime(model_dir)
    n = int(cfg["n_layers"])
    acc = {}
    for seed in range(n_probe):
        rng = np.random.default_rng(seed)
        toks = list(rng.integers(1, 500, size=seq_len))
        got = {}
        hooks = {L: (lambda L_: (lambda h: got.__setitem__(L_, np.asarray(h, np.float64).copy())))(L)
                 for L in range(n)}
        try:
            rt.forward(toks, hooks=hooks)
        except Exception:
            continue
        keys = sorted(got)
        for i in range(len(keys) - 1):
            a_i, b_i = keys[i], keys[i + 1]
            if b_i != a_i + 1:
                continue
            a, b = got[a_i].ravel(), got[b_i].ravel()
            c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            acc.setdefault(a_i, []).append(c)
    return {L: float(np.mean(v)) for L, v in acc.items() if v}


def _zero_layer_outputs(st_path, layer):
    """Zero a layer's output projections in place, so the residual passes through."""
    subs = ["layers.%d.linear_attn.out_proj" % layer,
            "layers.%d.self_attn.o_proj" % layer,
            "layers.%d.mlp.down_proj" % layer]
    hit = []
    with open(st_path, "r+b") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
        base = 8 + hlen
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            if any(s in name for s in subs):
                a, b = meta["data_offsets"]
                fh.seek(base + a)
                fh.write(b"\x00" * (b - a))
                hit.append(name)
    return hit


def bypass_agreement(model_dir, layer, prompts, baseline):
    """Stage 2: the causal confirmation. Copy, bypass one layer, compare top-1."""
    tmp = tempfile.mkdtemp(prefix="prune_probe_")
    dst = os.path.join(tmp, "model")
    try:
        shutil.copytree(model_dir, dst)
        for f in os.listdir(dst):
            if f.endswith(".safetensors"):
                _zero_layer_outputs(os.path.join(dst, f), layer)
        got = _forward_top1(dst, prompts)
        return float(np.mean([a == b for a, b in zip(baseline, got)]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def probe(model_dir, screen=0.95, accept=0.90, n_seqs=30, seq_len=12, candidates=None):
    """Full two-stage criterion. Returns screened candidates ranked by measured agreement."""
    cos = adjacent_cosines(model_dir)
    if candidates is None:
        candidates = sorted(L for L, c in cos.items() if c >= screen)
    rng = np.random.default_rng(7)
    prompts = [list(rng.integers(1, 500, size=seq_len)) for _ in range(n_seqs)]
    baseline = _forward_top1(model_dir, prompts)
    rows = []
    for L in candidates:
        agr = bypass_agreement(model_dir, L, prompts, baseline)
        rows.append({"layer": L, "cosine": cos.get(L), "agreement": agr,
                     "prunable": bool(agr >= accept)})
    rows.sort(key=lambda r: -r["agreement"])
    return {"screened": candidates, "rows": rows,
            "prunable": [r["layer"] for r in rows if r["prunable"]],
            "screen": screen, "accept": accept}


def main(argv):
    if not argv:
        print(__doc__)
        return 3
    model_dir = argv[0]

    def opt(flag, default, cast):
        return cast(argv[argv.index(flag) + 1]) if flag in argv else default

    screen = opt("--screen", 0.95, float)
    accept = opt("--accept", 0.90, float)
    n_seqs = opt("--seqs", 30, int)

    from tools.fixture_gate import gate
    try:
        v = gate(model_dir)
        print("fixture gate: %s (ppl %.1f vs vocab %d)" % (v["verdict"], v["perplexity"], v["vocab_size"]))
        if v["blocked"]:
            print("  NOTE: artifact is AT-CHANCE -- the ordering below is an information-FLOW")
            print("  measurement only. Do not read it as a quality claim.")
    except Exception as exc:
        print("fixture gate unavailable (%s) -- continuing, flow-only reading" % exc)

    res = probe(model_dir, screen=screen, accept=accept, n_seqs=n_seqs)
    print("\nstage 1 screen (cosine >= %.2f): %d candidate layers" % (screen, len(res["screened"])))
    print("stage 2 bypass probe (accept agreement >= %.2f):\n" % accept)
    print("  layer  cosine  agreement  verdict")
    for r in res["rows"]:
        print("  L%-4d  %.3f   %.3f      %s"
              % (r["layer"], r["cosine"] or float("nan"), r["agreement"],
                 "PRUNABLE" if r["prunable"] else "KEEP"))
    print("\nprunable: %s" % (res["prunable"] or "none -- cosine screen alone would have been wrong"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
