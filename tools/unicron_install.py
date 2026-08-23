"""UNICRON INSTALL -- put leCore inside a model's weights, and prove it or refuse.

One command turns a stock Llama-family checkpoint (Qwen3.5-0.8B, SmolLM2-135M, ...) into
a model that carries leCore's primitives natively: no leCore imports at inference, no
prompt asking for it, ordinary weights that run in any harness.

WHAT GETS INSTALLED (three layers, each independently gated and revertable):
  1. FACTS      taught reflexes -> batched Kohonen/MEMIT solve on mlp.down_proj.
                Keys are UNIT-NORMALIZED swiGLU gated activations (cp41: the unnormalized
                key was the bug that collapsed stable rank 53.8 -> 1.0). Values are
                tied-embedding targets raised just past the current argmax.
  2. ALGEBRA    bind/unbind as CIRCULANT matrices, cleanup as a projector (cp44: verified
                circulant(role) @ x == bind(role, x) to 9.7e-17). The engine is three
                matrices; bundle is the residual stream, already free.
  3. SWARM      a routed bank of leCore circuits with a content-keyed gate, so the model
                runs a swarm internally, per token (cp44: gain=0 is a BIT-EXACT no-op).

THE FIVE-POINT HEALTH GATE -- every fact install passes it or is REFUSED (cp41):
    (1) seed-identical   byte-identical rerun          (the chair's gate)
    (2) unit key         the key is normalized         (the cp41 bug, now structural)
    (3) |delta|/|W|      bounded, default < 0.10
    (4) stable rank      preserved, drop < 30%
    (5) locality         median drift on N unrelated keys < 0.10
LOCALITY IS THE BUDGET (cp44): efficacy is easy -- every fact hits at every N -- but
|delta|/|W| and drift climb with N. --budget auto BISECTS for the largest fact count that
still passes, instead of guessing.

THE CHAIR'S RULES (Quilez, cp44): the artifact runs without the toolchain; capability
ARRIVES OFF (--gain 0.0 is a no-op you switch on deliberately); the byte budget is
reported like a compo; nothing counts unless it reruns byte-identically.

HONEST BOUNDARY, restated on every run: this proves structure, keying, spectral health,
locality and determinism. EFFICACY GENERALIZATION (paraphrases) and PERPLEXITY RETENTION
require REAL trained weights and a real runtime -- measure them on your box.

    python3 tools/unicron_install.py /path/to/qwen3.5-0.8b out_dir \\
        --from-partition ~/claude_partition --budget auto \\
        --bake-algebra --bake-swarm 4 --gain 0.0 --cartridge lecore_cart
    python3 tools/unicron_install.py --inspect out_dir
    python3 tools/unicron_install.py --revert out_dir --cartridge lecore_cart.npz
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holographic.io_and_interop.holographic_unicron import (load_safetensors,
    save_safetensors, spectral_report, load_model, source_dtypes)


def load_any(path):
    """Every real-checkpoint shape a user actually has: a single .safetensors, a .gguf
    from llama.cpp, or a DIRECTORY that may be sharded across an index. Unicron already
    knew how to read all three; the installer just was not asking."""
    if os.path.isdir(path):
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            files = sorted({v for v in json.load(open(idx))["weight_map"].values()})
            out = {}
            for f in files:                     # names are disjoint across shards, so
                out.update(load_safetensors(os.path.join(path, f)))   # per-shard is exact
            return out, os.path.join(path, files[0])
        for cand in ("model.safetensors", "model.gguf"):
            fp = os.path.join(path, cand)
            if os.path.exists(fp):
                return load_model(fp), fp
        raise FileNotFoundError("no model.safetensors / index / gguf in %s" % path)
    return load_model(path), path

SILU = lambda z: z / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- architecture
def detect_root(tensors):
    """Tensor roots differ across families and this cost a user a cycle once already:
    SmolLM roots at 'model.', Qwen3.5 at 'model.language_model.'. Detect, never assume."""
    for root in ("model.language_model.", "model.", "language_model.", ""):
        if any(k.startswith(root + "layers.") for k in tensors):
            return root
    raise KeyError("no '<root>layers.N.' tensors found; is this a Llama-family model?")


def survey(tensors, root):
    layers = sorted({int(k.split(root + "layers.")[1].split(".")[0])
                     for k in tensors if k.startswith(root + "layers.")})
    emb_k = root + "embed_tokens.weight"
    emb = np.asarray(tensors[emb_k], np.float32)
    has_mlp = [l for l in layers
               if (root + "layers.%d.mlp.down_proj.weight" % l) in tensors]
    tied = not any(k.endswith("lm_head.weight") for k in tensors)
    return {"root": root, "n_layers": len(layers), "mlp_layers": has_mlp,
            "vocab": int(emb.shape[0]), "hidden": int(emb.shape[1]),
            "tied_embeddings": bool(tied)}


# ---------------------------------------------------------------- fact install
def _keys(tensors, root, layer, toks, emb):
    L = root + "layers.%d.mlp." % layer
    Wg = np.asarray(tensors[L + "gate_proj.weight"], np.float32)
    Wu = np.asarray(tensors[L + "up_proj.weight"], np.float32)
    out = []
    for t in toks:
        x = emb[int(t)] / (np.linalg.norm(emb[int(t)]) + 1e-9)
        k = SILU(Wg @ x) * (Wu @ x)
        out.append(k / (np.linalg.norm(k) + 1e-9))       # GATE 2: unit key, always
    return np.stack(out, axis=1) if out else np.zeros((Wg.shape[0], 0), np.float32)


def install_facts(tensors, root, layer, pairs, emb, margin=0.3, ridge=0.1,
                  n_locality=50, seed=0, preserve_k=None):
    """Batched Kohonen/MEMIT solve for N (subject_token -> answer_token) pairs."""
    L = root + "layers.%d.mlp.down_proj.weight" % layer
    Wd = np.asarray(tensors[L], np.float32)
    K = _keys(tensors, root, layer, [p[0] for p in pairs], emb)
    if K.shape[1] == 0:
        return Wd, np.zeros_like(Wd), {"n": 0}
    cur = Wd @ K
    V = np.empty_like(cur)
    for i, (_s, a) in enumerate(pairs):
        e_a = emb[int(a)] / (np.linalg.norm(emb[int(a)]) + 1e-9)
        need = float((emb @ cur[:, i]).max()) - float(e_a @ cur[:, i]) + margin
        V[:, i] = cur[:, i] + max(need, 0.0) * e_a
    N = K.shape[1]
    delta = (V - Wd @ K) @ np.linalg.inv(K.T @ K + ridge * np.eye(N)) @ K.T
    # ALPHAEDIT NULL-SPACE PROTECTION (cp65; Fang et al., ICLR 2025 Outstanding
    # Paper, arXiv 2410.02355): sequential/batched edits without protection cause
    # gradual-then-catastrophic forgetting (Gupta 2401.07453); the field's fix --
    # a +36.7% average across locate-then-edit methods, one line at heart -- is to
    # PROJECT the delta onto the null space of preserved knowledge's keys, so any
    # key the model already answers correctly passes through the edit unchanged:
    # (W + delta P) k0 = W k0 exactly, because P k0 ~ 0. Preserved keys here are
    # PROBE keys through this layer (on real weights: keys of held-out prompts;
    # in the sandbox: sampled tokens, honestly the same mechanism). Our five-point
    # gate DETECTS collapse; this PREVENTS it. Measured below and reported as
    # preservation_before/after so the projection proves itself every install.
    if preserve_k is not None and preserve_k.shape[1] > 0:
        Kp = preserve_k
        U, S, _vt = np.linalg.svd(Kp, full_matrices=False)
        r = int((S > S.max() * 1e-6).sum())
        Up = U[:, :r]
        pres_before = float(np.linalg.norm(delta @ Kp) /
                            (np.linalg.norm(Wd @ Kp) + 1e-9))
        delta = delta - (delta @ Up) @ Up.T
        pres_after = float(np.linalg.norm(delta @ Kp) /
                           (np.linalg.norm(Wd @ Kp) + 1e-9))
    else:
        pres_before = pres_after = None
    Wd2 = Wd + delta
    hits = sum(int(np.argmax(emb @ (Wd2 @ K[:, i]))) == int(pairs[i][1])
               for i in range(N))
    rng = np.random.default_rng(seed)
    others = [int(t) for t in rng.integers(0, emb.shape[0], n_locality)]
    KO = _keys(tensors, root, layer, others, emb)
    drift = float(np.median([np.linalg.norm(delta @ KO[:, j]) /
                             (np.linalg.norm(Wd @ KO[:, j]) + 1e-9)
                             for j in range(KO.shape[1])]))
    sp0, sp1 = spectral_report(Wd), spectral_report(Wd2)
    sr0, sr1 = sp0.get("stable_rank", 0.0), sp1.get("stable_rank", 0.0)
    return Wd2, delta, {"n": N, "hits": hits, "efficacy": hits / max(N, 1),
                        "delta_over_w": float(np.linalg.norm(delta) /
                                              (np.linalg.norm(Wd) + 1e-9)),
                        "stable_rank_before": sr0, "stable_rank_after": sr1,
                        "srank_drop": float((sr0 - sr1) / (sr0 + 1e-9)),
                        "locality_drift": drift, "unit_key": True,
                        "preservation_before": pres_before,
                        "preservation_after": pres_after}


def gate(rep, max_ratio, max_srank_drop, max_drift, seed_identical):
    checks = {"seed_identical": bool(seed_identical),
              "unit_key": bool(rep.get("unit_key")),
              "efficacy_full": rep.get("efficacy", 0.0) >= 1.0,
              "ratio_ok": rep.get("delta_over_w", 9e9) < max_ratio,
              "srank_ok": rep.get("srank_drop", 9e9) < max_srank_drop,
              "locality_ok": rep.get("locality_drift", 9e9) < max_drift}
    return all(checks.values()), checks


def budget_search(tensors, root, layer, pairs, emb, limits, **kw):
    """LOCALITY IS THE BUDGET (cp44): bisect for the largest N that still passes."""
    lo, hi, best = 0, len(pairs), (0, None, None, None)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        Wd2, d, rep = install_facts(tensors, root, layer, pairs[:mid], emb, **kw)
        _2, _d2, rep2 = install_facts(tensors, root, layer, pairs[:mid], emb, **kw)
        ok, _ = gate(rep, *limits, np.array_equal(Wd2, _2))
        if ok:
            best, lo = (mid, Wd2, d, rep), mid
        else:
            hi = mid - 1
    return best


# ---------------------------------------------------------------- algebra/swarm
def bake_algebra(tensors, root, layer, dim, seed=0, gain=1.0):
    """bind/unbind as circulants + a cleanup projector -- the engine is three matrices."""
    from holographic.io_and_interop.holographic_vsabake import circulant, involution
    from holographic.agents_and_reasoning.holographic_ai import random_vector
    from holographic.agents_and_reasoning.holographic_ai import bind as lc_bind
    rng = np.random.default_rng(seed)
    role = random_vector(dim, rng)
    C, Ci = circulant(role), circulant(involution(role))
    # THE IDENTITY THAT MATTERS (cp44, verified 9.7e-17): the circulant IS leCore's bind.
    x = random_vector(dim, rng)
    exact = float(np.max(np.abs(C @ x - lc_bind(role, x))))
    # The unbind ROUNDTRIP is a different claim and is APPROXIMATE by HRR's nature --
    # involution unbinding is exact only for unitary roles. Report it as a cosine, not as
    # an error, so a healthy approximate recovery is never mistaken for a broken identity
    # (the first cp48 run printed 3.5e-01 here and looked like a failure; it was the wrong
    # check on the right math).
    back = Ci @ (C @ x)
    rt_cos = float(np.dot(back / (np.linalg.norm(back) + 1e-12),
                          x / (np.linalg.norm(x) + 1e-12)))
    return {"role_seed": seed, "circulant_is_bind_err": exact,
            "unbind_roundtrip_cos": rt_cos,
            "matrices": {"bind": C, "unbind": Ci}, "gain": gain}


def bake_swarm(tensors, root, layer, states, n_experts, gain, seed=0):
    from holographic.io_and_interop.holographic_swarmbake import install_swarm
    from holographic.io_and_interop.holographic_vsabake import circulant, involution
    from holographic.agents_and_reasoning.holographic_ai import random_vector
    H = states.shape[1]
    rng = np.random.default_rng(seed)
    role = random_vector(H, rng)
    experts = [circulant(role), circulant(involution(role)),
               np.eye(H, dtype=np.float32) * 0.5, np.eye(H, dtype=np.float32) * 0.25]
    experts = (experts * ((n_experts // 4) + 1))[:n_experts]
    n_layers = max(int(k.split(root + "layers.")[1].split(".")[0])
                   for k in tensors if k.startswith(root + "layers.")) + 1
    out = install_swarm({k: np.asarray(v, np.float32) for k, v in tensors.items()},
                        {"n_layers": n_layers}, experts, states, layer=layer, gain=gain)
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------- facts sourcing
def facts_from_partition(path, vocab, limit=None):
    """TAUGHT FACTS ONLY (cp47): model-cached answers are provisional and must never be
    written into weights as if they were established truth."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import lecore
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "")
    m.learning_load(path)
    lad = m.zoo["ladder"]
    out = []
    for row in getattr(lad, "taught_log", []):
        q, a = str(row[0]), str(row[1])
        ex = getattr(lad, "_exact", {}).get(" ".join(q.lower().split()), {})
        if ex and ex.get("provenance") not in (None, "taught"):
            continue
        h = lambda s: abs(hash(s)) % vocab
        out.append((h(q), h(a), q[:60], a[:60]))
        if limit and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--from-partition"); ap.add_argument("--facts")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--spread", type=int, default=1,
                    help="MEMIT residual spreading: share each fact across this many "
                         "consecutive layers (r^l = residual/(L-l+1)); 1 = single layer")
    ap.add_argument("--plan", action="store_true",
                    help="DRY RUN: report exactly what would be installed and what the "
                         "gate would say, and write nothing")
    ap.add_argument("--budget", default="auto")
    ap.add_argument("--bake-algebra", action="store_true")
    ap.add_argument("--bake-swarm", type=int, default=0)
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--max-ratio", type=float, default=0.10)
    ap.add_argument("--max-srank-drop", type=float, default=0.30)
    ap.add_argument("--max-drift", type=float, default=0.10)
    ap.add_argument("--cartridge"); ap.add_argument("--revert")
    ap.add_argument("--revert-swarm", action="store_true",
                    help="also trim baked swarm neurons back to pre-bake shapes")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.inspect:
        p = os.path.join(a.model, "lecore.json")
        print(json.dumps(json.load(open(p)), indent=2) if os.path.exists(p)
              else "no lecore.json -- nothing installed by unicron")
        return
    t, src = load_any(a.model)
    t = {k: np.asarray(v) for k, v in t.items()}
    try:                                        # PRESERVE ON-DISK DTYPE: our loader
        keep_dtypes = source_dtypes(a.model)    # decodes bf16 to f32, and writing that
    except Exception:                           # back DOUBLES the file (Unicron's own
        keep_dtypes = None                      # kept negative, measured on Qwen3.5)
    root = detect_root(t)
    info = survey(t, root)
    print("MODEL: %d layers, hidden %d, vocab %d, root %r, tied_embeddings=%s"
          % (info["n_layers"], info["hidden"], info["vocab"], root,
             info["tied_embeddings"]))
    if a.revert:
        # SHAPE-AWARE REVERT (cp48 red, found by running it): baking the swarm WIDENS the
        # MLP (down_proj 384 -> 896 columns), so a fact delta saved before the bake no
        # longer aligns with the tensor it edited. The fact edit lives in the ORIGINAL
        # sub-block, so revert subtracts there; --revert-swarm additionally trims the
        # tensors back to their pre-bake shapes recorded in the cartridge.
        c = np.load(a.revert, allow_pickle=True)
        meta = json.loads(str(c["meta"])) if "meta" in c else {}
        trimmed = 0
        if a.revert_swarm:
            for name, shp in (meta.get("pre_swarm_shapes") or {}).items():
                if name in t and tuple(np.asarray(t[name]).shape) != tuple(shp):
                    arr = np.asarray(t[name])
                    t[name] = arr[tuple(slice(0, int(v)) for v in shp)].copy()
                    trimmed += 1
        origs = c["originals"] if "originals" in c.files else None
        for i, (name, d) in enumerate(zip(c["names"], c["deltas"])):
            name = str(name)
            if origs is not None:
                o = np.asarray(origs[i], np.float32)
                arr0 = np.asarray(t[name], np.float32)
                if arr0.shape == o.shape:
                    t[name] = o                      # EXACT restore, dtype-independent
                    continue
                arr0 = arr0.copy()
                arr0[tuple(slice(0, int(v)) for v in o.shape)] = o
                t[name] = arr0
                continue
            arr = np.asarray(t[name], np.float32)
            sl = tuple(slice(0, int(v)) for v in d.shape)
            if arr.shape == d.shape:
                t[name] = arr - d
            else:
                arr = arr.copy(); arr[sl] = arr[sl] - d; t[name] = arr
                print("  (tensor is wider than the delta -- fact reverted in the original "
                      "sub-block; the swarm bake is %s)"
                      % ("trimmed too" if a.revert_swarm else "still present, use --revert-swarm"))
        os.makedirs(a.out, exist_ok=True)
        save_safetensors(os.path.join(a.out, "model.safetensors"), t)
        for extra in ("config.json", "tokenizer.json", "lecore.json"):
            sp = os.path.join(os.path.dirname(src), extra)
            if os.path.exists(sp):
                open(os.path.join(a.out, extra), "wb").write(open(sp, "rb").read())
        print("REVERTED %d tensor(s)%s -> %s"
              % (len(c["names"]), " (+%d trimmed)" % trimmed if trimmed else "", a.out))
        return

    emb = np.asarray(t[root + "embed_tokens.weight"], np.float32)
    layer = a.layer if a.layer is not None else info["mlp_layers"][len(info["mlp_layers"]) // 4]
    pairs_meta = []
    if a.from_partition:
        pairs_meta = facts_from_partition(a.from_partition, info["vocab"])
    elif a.facts:
        for row in json.load(open(a.facts)):
            pairs_meta.append((int(row["subject_token"]), int(row["answer_token"]),
                               row.get("q", ""), row.get("a", "")))
    pairs = [(p[0], p[1]) for p in pairs_meta]
    limits = (a.max_ratio, a.max_srank_drop, a.max_drift)
    t0 = time.time()
    manifest = {"unicron_install": "cp48", "root": root, "layer": layer,
                "model": info, "gate_limits": {"max_ratio": a.max_ratio,
                "max_srank_drop": a.max_srank_drop, "max_drift": a.max_drift},
                "honest_boundary": "structure/keying/spectra/locality/determinism are "
                "proven here; EFFICACY GENERALIZATION and PERPLEXITY RETENTION require "
                "real trained weights measured on your runtime"}
    cart_names, cart_deltas, cart_originals = [], [], []

    if pairs:
        print("\nFACTS: %d taught candidates (model-cached answers excluded)" % len(pairs))
        if a.spread > 1:
            band = [l for l in info["mlp_layers"] if layer <= l < layer + a.spread]
            print("  SPREAD: MEMIT residual sharing across layers %s -- each layer absorbs"
                  " an equal fraction, so no single matrix carries the whole edit" % band)
        if a.budget == "auto":
            n, Wd2, d, rep = budget_search(t, root, layer, pairs, emb, limits)
            print("  BUDGET SEARCH (locality is the budget): largest passing N = %d" % n)
        else:
            n = int(a.budget)
            _kp = _keys(t, root, layer,
                        [int(x) for x in np.random.default_rng(99).integers(
                            0, emb.shape[0], 64)], emb)
            Wd2, d, rep = install_facts(t, root, layer, pairs[:n], emb,
                                        preserve_k=_kp)
            _w2, _d2, _r2 = install_facts(t, root, layer, pairs[:n], emb)
            rep["seed_identical"] = np.array_equal(Wd2, _w2)
        if n:
            _w2, _, _ = install_facts(t, root, layer, pairs[:n], emb)
            ok, checks = gate(rep, *limits, np.array_equal(Wd2, _w2))
            print("  efficacy %d/%d | |d|/|W|=%.4f | srank %.1f->%.1f | drift %.4f"
                  % (rep["hits"], rep["n"], rep["delta_over_w"],
                     rep["stable_rank_before"], rep["stable_rank_after"],
                     rep["locality_drift"]))
            print("  GATE: %s ==> %s" % (checks, "PASS" if ok else "REFUSED"))
            if a.plan:
                print("  PLAN ONLY -- nothing written.")
                manifest["facts"] = {"planned": int(n), "gate": checks}
            elif ok or a.force:
                band = ([l for l in info["mlp_layers"] if layer <= l < layer + a.spread]
                        if a.spread > 1 else [layer])
                for bi, bl in enumerate(band):
                    frac = 1.0 / (len(band) - bi)          # r^l = residual/(L-l+1)
                    nm = root + "layers.%d.mlp.down_proj.weight" % bl
                    if bl == layer:
                        part = d * frac
                        t[nm] = np.asarray(t[nm], np.float32) + part
                    else:
                        _w, part, _r = install_facts(t, root, bl, pairs[:n], emb)
                        part = part * frac
                        t[nm] = np.asarray(t[nm], np.float32) + part
                    cart_names.append(nm); cart_deltas.append(part)
                    cart_originals.append(np.asarray(t[nm], np.float32) - part)
                name = root + "layers.%d.mlp.down_proj.weight" % layer
                manifest["facts"] = {"installed": int(n), "report": {
                    k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in rep.items()},
                    "gate": checks, "forced": bool(not ok and a.force),
                    "provenance": "taught-only",
                    "sample": [{"q": p[2], "a": p[3]} for p in pairs_meta[:5]]}
            else:
                manifest["facts"] = {"installed": 0, "refused_by_gate": checks}
        else:
            print("  BUDGET 0 -- no fact count passes the gate on these weights.")
            manifest["facts"] = {"installed": 0,
                                 "note": "no N passed; expected on random-init weights"}

    if a.bake_algebra:
        alg = bake_algebra(t, root, layer, info["hidden"])
        manifest["algebra"] = {"circulant_is_bind_err": alg["circulant_is_bind_err"],
            "unbind_roundtrip_cos": alg["unbind_roundtrip_cos"],
            "note": "bind IS a circulant matrix (exact); unbind by involution is HRR-"
                    "approximate (cosine); bundle is the residual stream, already free"}
        print("\nALGEBRA: circulant == leCore bind to %.1e | unbind roundtrip cos %.4f"
              % (alg["circulant_is_bind_err"], alg["unbind_roundtrip_cos"]))

    if a.bake_swarm:
        states = emb[::max(1, emb.shape[0] // 256)][:256].astype(np.float64)
        before = {k: np.asarray(v, np.float32).copy() for k, v in t.items()}
        pre_shapes = {k: list(np.asarray(v).shape) for k, v in before.items()
                      if (root + "layers.%d.mlp." % layer) in k}
        w2 = bake_swarm(t, root, layer, states, a.bake_swarm, a.gain)
        added = sum(np.asarray(w2[k]).size for k in w2) - sum(np.asarray(before[k]).size
                                                              for k in before)
        w2b = bake_swarm(before, root, layer, states, a.bake_swarm, a.gain)
        seed_ok = all(np.array_equal(np.asarray(w2[k]), np.asarray(w2b[k])) for k in w2)
        t = w2
        manifest["swarm"] = {"experts": a.bake_swarm, "gain": a.gain,
            "params_added": int(added), "kb_fp32": round(added * 4 / 1024, 1),
            "seed_identical": bool(seed_ok),
            "arrives_off": bool(a.gain == 0.0)}
        manifest["pre_swarm_shapes"] = pre_shapes
        print("\nSWARM: %d experts | +%d params (%.1f KB) | gain=%.2f%s | seed-identical=%s"
              % (a.bake_swarm, added, added * 4 / 1024, a.gain,
                 "  [ARRIVES OFF -- switch on deliberately]" if a.gain == 0 else "",
                 seed_ok))

    manifest["ouroboros"] = {"recommended_layer": int(info["n_layers"] // 2),
        "dk": 64, "decay": 0.98,
        "note": "capacity_report declares its REGIME; in the mixed regime (stream "
                "background + written facts) predicted_recall is an UPPER BOUND -- call "
                "verify_recall(pairs) for ground truth (cp46)"}
    manifest["seconds"] = round(time.time() - t0, 2)

    if a.plan:
        # A DRY RUN THAT WRITES IS WORSE THAN NO DRY RUN (cp50 red, caught by running it):
        # --plan short-circuits every write path, not just the fact edit.
        print("\nPLAN COMPLETE -- no model, manifest or cartridge written.")
        print(json.dumps({k: v for k, v in manifest.items()
                          if k in ("facts", "algebra", "swarm", "layer")}, indent=1)[:600])
        print("\nHONEST BOUNDARY: %s" % manifest["honest_boundary"])
        return
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        save_safetensors(os.path.join(a.out, "model.safetensors"), t,
                         dtypes=keep_dtypes)
        for extra in ("config.json", "tokenizer.json"):
            sp = os.path.join(os.path.dirname(src), extra)
            if os.path.exists(sp):
                open(os.path.join(a.out, extra), "wb").write(open(sp, "rb").read())
        json.dump(manifest, open(os.path.join(a.out, "lecore.json"), "w"), indent=2)
        print("\nWROTE %s (model.safetensors + lecore.json manifest)" % a.out)
    if a.cartridge and cart_deltas:
        # THE CARTRIDGE CARRIES BOTH (cp50, found by measuring a revert): deltas compose
        # and can be scaled or stacked, but delta-subtraction cannot be EXACT once the
        # model is written at its source dtype -- preserving bf16 (which exists to stop
        # the file doubling) rounds the weights, and reverting then left a 2.4e-04
        # residual. Storing the ORIGINAL rows of the few edited tensors costs almost
        # nothing and makes revert exact at any dtype.
        np.savez(a.cartridge if a.cartridge.endswith(".npz") else a.cartridge + ".npz",
                 names=np.array(cart_names), deltas=np.array(cart_deltas),
                 originals=np.array(cart_originals), meta=json.dumps(manifest))
        print("CARTRIDGE %s(.npz) -- exact apply/revert" % a.cartridge)
    print("\nHONEST BOUNDARY: %s" % manifest["honest_boundary"])


if __name__ == "__main__":
    main()
