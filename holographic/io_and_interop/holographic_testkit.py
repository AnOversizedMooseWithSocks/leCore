"""TESTKIT -- export the smallest thing that makes experiments HONEST.

Every conclusion in this arc that later turned out wrong was wrong because the
subject was a 1.8M-parameter byte-level model standing in for a 0.8B. The list
is long enough to be embarrassing: sharded weights, split projections, a missing
vocabulary, near-full-rank matrices that made factoring look useless, matmuls
too small for a FLOP win to show, and heads that forget in 0.1 tokens.

A real checkpoint cannot travel here. But almost none of those questions need
the weights -- they need the SHAPE of the weights. This exports that: spectra,
decay rates, activation statistics and ONE representative layer, which together
are a few tens of megabytes and answer most of what the toy answers wrongly.

WHAT IT DELIBERATELY DOES NOT EXPORT: the model. No full weight tensors beyond a
single layer the caller opts into, no training data, no user text. The default
probe is a fixed public sentence, and the file lists exactly what it contains so
nothing ships that the sender did not see named.
"""

import json
import os

import numpy as np


# A REAL TOKENIZER PACKS WORDS INTO SINGLE TOKENS, so a paragraph that looked
# like 256 tokens on a byte model is 55 on a 248k vocabulary -- measured. The
# probe is now long and DIVERSE (prose, facts, code, structure, repetition),
# because activation statistics from 55 tokens of one register are thin.
DEFAULT_PROBE = (
    "The capital of France is Paris, and the capital of Japan is Tokyo. "
    "Water freezes at zero degrees celsius and boils at one hundred. "
    "A recurrent state carries what the past can tell the future, and every "
    "layer writes into the residual stream that follows it. "
    "In 1969 David Marr proposed that the cerebellum works as an associative "
    "memory, and Kanerva later formalised sparse distributed memory. "
    "def compress(x, rank=8):\n"
    "    u, s, vt = numpy.linalg.svd(x, full_matrices=False)\n"
    "    return (u[:, :rank] * s[:rank]) @ vt[:rank]\n"
    "SELECT title, author FROM notes WHERE session = 's1' ORDER BY created;\n"
    "# Heading\n- first item\n- second item\n\n"
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps "
    "over the lazy dog again, and again, and again. "
    "Questions: what happens to ice when it melts? Why is the sky blue? "
    "How does a delta rule update a memory matrix in place? "
    "Answer carefully, step by step, and cite the passage you used.")


def _singular_values(a, chunk=8192):
    """Singular values without ever materialising a huge float64 copy.

    A 248,320 x 1024 embedding table is 2 GB in float64 before LAPACK asks for
    its own workspace, and np.linalg.svd died with MemoryError on exactly that
    tensor. But for a matrix that is far taller than it is wide, the singular
    values are the square roots of the eigenvalues of the small Gram matrix
    A^T A -- 1024 x 1024 here -- and the Gram can be ACCUMULATED IN CHUNKS, so
    peak memory is one chunk rather than the whole tensor.

    Exact to floating point for the leading values, which is what every use of
    these spectra reads. The tall/wide test is arithmetic: use the Gram whenever
    the small dimension is much smaller than the large one, and the direct SVD
    otherwise (where it is cheaper and better conditioned)."""
    A = np.asarray(a)
    m, n = A.shape
    small, large = min(m, n), max(m, n)
    if large <= 4096 or large < 4 * small:
        return np.linalg.svd(np.asarray(A, np.float64), compute_uv=False)
    G = np.zeros((small, small), np.float64)
    if m >= n:
        for i in range(0, m, int(chunk)):
            B = np.asarray(A[i:i + int(chunk)], np.float64)
            G += B.T @ B
    else:
        for i in range(0, n, int(chunk)):
            B = np.asarray(A[:, i:i + int(chunk)], np.float64)
            G += B @ B.T
    ev = np.linalg.eigvalsh(G)                  # ascending, symmetric
    return np.sqrt(np.clip(ev[::-1], 0.0, None))


def export(model_dir, out_path, probe=None, layer=None, include_layer=True,
           n_singular=None, activations=True, logit_topk=64,
           layer_dtype="float16"):
    """Write a .npz test kit describing a real checkpoint.

    Contents, all named in the manifest inside the file:
      spectra          top-`n_singular` singular values of every 2-D tensor
                       -> answers "is this compressible", which the toy got wrong
      shapes/dtypes    every tensor, so layout bugs are caught without the model
      head_decay       A_log / dt_bias per layer -> does the real model also
                       forget within a token?
      activations      hidden states at every layer for the probe -> everything
                       that needs a real stream: dreamer, carrier, salience,
                       memory horizon, threshold calibration
      logits           final logits for the probe -> distillation teachers and
                       verification targets
      one_layer        (optional) every tensor of a single layer, so baking,
                       growing and factoring can be tested on REAL numbers
    """
    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime, load_weights_dir)

    rt, cfg = load_runtime(model_dir)
    w = load_weights_dir(model_dir)
    text = probe or DEFAULT_PROBE
    try:
        from holographic.io_and_interop.holographic_bpe import BPE
        ids = BPE.from_dir(model_dir).encode(text)[:512]
    except Exception:
        ids = [b for b in text.encode("utf-8")][:512]

    out = {}
    manifest = {"config": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in cfg.items()},
                "tensor_root": getattr(rt, "root", "model."),
                "probe_tokens": len(ids), "contains": []}

    # ---- spectra: the question the toy answered wrongly ----
    shapes = {}
    for k, v in sorted(w.items()):
        a = np.asarray(v)
        shapes[k] = [list(a.shape), str(a.dtype)]
        if a.ndim == 2 and min(a.shape) >= 8:
            sv = _singular_values(a)
            # EXPORT THE WHOLE SPECTRUM. Truncating at 64 CENSORED the answer:
            # on a real 0.8B every tensor reported r90 ~= 55-57, which is just
            # "more than 64" wearing a number, and compressibility -- the entire
            # question the spectra exist to answer -- could not be read at all.
            # A full spectrum is min(m,n) floats: ~4 KB per tensor, under 1 MB
            # for the whole model. The truncation saved nothing and cost the
            # measurement.
            if n_singular == 0:
                continue                    # per-layer files skip spectra: the
                                            # base file already carries them all
            keep = len(sv) if n_singular is None else int(n_singular)
            out["sv::" + k] = sv[:keep].astype(np.float32)
    manifest["shapes"] = shapes
    manifest["contains"].append(
        "spectra (%s singular values per 2-D tensor)"
        % ("FULL" if n_singular in (None, 0) else "top %d" % n_singular))

    # ---- the recurrence gates: does the real model forget in a token? ----
    for k in sorted(w):
        if k.endswith("A_log") or k.endswith("dt_bias"):
            out["gate::" + k] = np.asarray(w[k], np.float32)
    manifest["contains"].append("A_log / dt_bias for every linear-attention layer")

    # ---- a real stream ----
    if activations:
        cap = {}
        rt.forward(ids, hooks={L: (lambda h, _L=L:
                                   cap.__setitem__(_L, h.copy()) or None)
                               for L in range(int(cfg["n_layers"]))})
        for L, h in cap.items():
            # float16 halves the stream for statistics that are already noisy at
            # the fourth decimal; the manifest says so rather than pretending
            # the kit is exact
            out["act::%d" % L] = np.asarray(h, np.float16)
        # LOGITS AS TOP-K, NOT DENSE. A 248k vocabulary over 256 positions is
        # 254 MB of mostly-irrelevant numbers -- and every use here (distillation
        # teachers, argmax agreement, verification) reads the head of the
        # distribution. Storing the top `logit_topk` values and their ids is
        # ~500x smaller and answers the same questions.
        lg = np.asarray(rt.forward(ids), np.float64)
        k = int(min(logit_topk, lg.shape[-1]))
        idx = np.argsort(lg, axis=-1)[:, -k:][:, ::-1]
        out["logit_top_idx"] = idx.astype(np.int32)
        out["logit_top_val"] = np.take_along_axis(lg, idx, axis=-1).astype(np.float32)
        out["logit_logsumexp"] = (np.log(np.sum(np.exp(
            lg - lg.max(-1, keepdims=True)), -1)).ravel()
            + lg.max(-1)).astype(np.float32)      # exact normaliser, for KL
        out["probe_ids"] = np.asarray(ids, np.int64)
        manifest["contains"].append(
            "hidden states at every layer (float16) + top-%d logits with the "
            "exact log-sum-exp, so probabilities are recoverable" % k)

    # ---- one real layer, so edits can be tested on real numbers ----
    if include_layer:
        L = int(cfg["n_layers"]) - 1 if layer is None else int(layer)
        # USE THE RUNTIME'S DETECTED ROOT, never a hardcoded prefix. Moose's
        # Qwen names its tensors model.language_model.layers.*, so a literal
        # "model.layers.%d." matched NOTHING and the kit silently shipped
        # without the one thing that needed real weights -- the manifest even
        # said "layer_exported: 23" while exporting zero arrays.
        pre = "%slayers.%d." % (getattr(rt, "root", "model."), L)
        if not any(k.startswith(pre) for k in w):
            cand = sorted({k.split("layers.")[0] for k in w if "layers." in k})
            raise ValueError("no tensors under %r -- this model names them %s"
                             % (pre, cand))
        for k, v in w.items():
            if k.startswith(pre):
                a = np.asarray(v)
                out["layer::" + k] = (a.astype(layer_dtype)
                                      if a.dtype.kind == "f" else a)
        manifest["layer_exported"] = L
        manifest["contains"].append("every tensor of layer %d (real weights)" % L)

    out["manifest"] = np.frombuffer(json.dumps(manifest).encode("utf-8"),
                                    dtype=np.uint8)
    np.savez_compressed(out_path, **out)
    size = os.path.getsize(out_path)
    return {"path": out_path, "megabytes": round(size / 1e6, 2),
            "arrays": len(out), "contains": manifest["contains"],
            "layer_exported": manifest.get("layer_exported")}


def export_all(model_dir, out_dir, probe=None, n_singular=None,
               layer_dtype="float16", logit_topk=64, progress=None,
               layers=None):
    """Export EVERY layer as its own file, plus one shared base.

    WHY SEPARATE FILES, and it is a size argument rather than a style one: a
    single layer of a 0.8B is ~37 MB at float16, so all 24 in one archive is
    ~880 MB -- past what anyone wants to move around, and all of it useless if
    the transfer fails once. Per-layer files mean any single layer can be sent
    on its own, and the shared base (spectra, gates, activations, logits) is
    written once instead of 24 times.

    Produces:
        base.npz        everything that is not per-layer weights   (~15 MB)
        layer_00.npz ... layer_NN.npz    one layer of real weights each
    """
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime

    os.makedirs(out_dir, exist_ok=True)
    rt, cfg = load_runtime(model_dir)
    n_layers = int(cfg["n_layers"])

    base_path = os.path.join(out_dir, "base.npz")
    rep = export(model_dir, base_path, probe=probe, include_layer=False,
                 n_singular=n_singular, logit_topk=logit_topk)
    written = [{"file": "base.npz", "megabytes": rep["megabytes"],
                "contains": rep["contains"]}]

    # LOAD THE MODEL ONCE. The first version called export() per layer, which
    # re-read every shard and re-ran the sanity check 24 times -- minutes of
    # pointless I/O on a 0.8B, and 24 identical lines of console noise.
    from holographic.io_and_interop.holographic_gdnruntime import load_weights_dir
    w = load_weights_dir(model_dir)
    root = getattr(rt, "root", "model.")
    # ONLY THE LAYERS ASKED FOR. Writing all 24 to use three is ~860 MB of disk
    # and minutes of compression spent on files nobody opens; `layers=None`
    # still means all, but the caller should usually name a few.
    wanted = (list(range(n_layers)) if layers is None
              else [int(x) for x in layers if 0 <= int(x) < n_layers])
    for L in wanted:
        pre = "%slayers.%d." % (root, L)
        arrays = {}
        for k, v in w.items():
            if k.startswith(pre):
                a = np.asarray(v)
                arrays["layer::" + k] = (a.astype(layer_dtype)
                                         if a.dtype.kind == "f" else a)
        if not arrays:
            continue
        man = {"layer": L, "tensor_root": root,
               "config": {kk: (list(vv) if isinstance(vv, tuple) else vv)
                          for kk, vv in cfg.items()},
               "contains": ["every tensor of layer %d (%s)" % (L, layer_dtype)]}
        arrays["manifest"] = np.frombuffer(json.dumps(man).encode("utf-8"),
                                           dtype=np.uint8)
        path = os.path.join(out_dir, "layer_%02d.npz" % L)
        np.savez_compressed(path, **arrays)
        mb = round(os.path.getsize(path) / 1e6, 2)
        written.append({"file": os.path.basename(path), "megabytes": mb,
                        "layer": L})
        if progress:
            progress(L, path, mb)
    total = sum(x["megabytes"] for x in written)
    return {"out_dir": out_dir, "files": written, "total_megabytes": round(total, 1),
            "layers": len(wanted), "of_layers": n_layers,
            "note": "send base.npz plus whichever layer files are wanted; each "
                    "layer stands alone"}


def load(path):
    """Read a kit back: returns (manifest, dict-of-arrays)."""
    z = np.load(path, allow_pickle=False)
    man = json.loads(bytes(z["manifest"]).decode("utf-8"))
    return man, {k: z[k] for k in z.files if k != "manifest"}


def _selftest():
    import os
    import tempfile

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("testkit selftest SKIPPED-SUBJECT (no model present)")
        return
    path = os.path.join(tempfile.mkdtemp(), "kit.npz")
    rep = export(src, path, n_singular=32)
    man, arrays = load(path)

    # ---- the kit is SELF-DESCRIBING: a reader can tell what it received ----
    assert man["contains"] and man["config"]["n_layers"], man
    assert any(k.startswith("sv::") for k in arrays)
    assert any(k.startswith("act::") for k in arrays)
    assert "logit_top_val" in arrays and "probe_ids" in arrays
    # the top-k form must reconstruct real probabilities, or it is not a
    # substitute for the dense logits it replaces
    p_top = np.exp(arrays["logit_top_val"][0].astype(np.float64)
                   - arrays["logit_logsumexp"][0])
    assert 0.0 < p_top.sum() <= 1.0 + 1e-5, p_top.sum()
    assert p_top[0] == p_top.max(), "top-k must be sorted by value"
    assert any(k.startswith("layer::") for k in arrays)

    # ---- and it does NOT contain the model ----
    full = sum(1 for k in arrays if k.startswith("layer::"))
    total_tensors = len(man["shapes"])
    assert full < total_tensors / 2, ("a kit must not be the checkpoint",
                                      full, total_tensors)

    # ---- the spectra are usable for the question they exist to answer ----
    k = next(k for k in arrays if k.startswith("sv::") and arrays[k].size > 8)
    sv = arrays[k]
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    r90 = int(np.searchsorted(energy, 0.90)) + 1
    assert 1 <= r90 <= len(sv)

    print("testkit selftest OK -- %.2f MB, %d arrays; self-describing manifest "
          "lists %d kinds of content; carries spectra (r90=%d for a sample "
          "tensor), gates, a real stream and ONE layer of real weights (%d of "
          "%d tensors), which is not the checkpoint"
          % (rep["megabytes"], rep["arrays"], len(rep["contains"]), r90,
             full, total_tensors))


if __name__ == "__main__":
    _selftest()
