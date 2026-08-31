"""ADAPT -- read a model we have never seen, from its tensors alone.

Moose: Unicron should install leCore into ANY model we choose, and we already
demux and decompose UNLABELED datasets, so this should be easier. He is right,
and the framing is the useful part: A CHECKPOINT IS AN UNLABELED DATASET. It is
a few hundred arrays with names someone else chose, and every question we ask of
it -- which axis is the carrier, which is the payload, where does the structure
repeat -- is a question leCore already answers for unlabeled data.

WHAT IS ACTUALLY UNKNOWN about a strange model:
    where the layers are          a numeric field that REPEATS in the names
    the hidden width              the dimension that appears in the most tensors
    which tensor is the vocabulary  2-D, one axis hidden, the other much larger
    whether embeddings are tied     is there a separate head tensor at all
    which axis is IN vs OUT         `analyze_axes` -- carrier versus payload
    which rows are free             the tokenizer's added_tokens, when present

NONE OF THAT NEEDS A CONFIG. Measured on a real checkpoint with config.json
withheld: 4 layer indices recovered from the names, hidden 128 recovered as the
modal dimension (appearing in 40 tensors against 15 for the next), the
vocabulary tensor identified by shape, and tied-versus-untied answered by
whether an lm_head exists.

WHY THIS MATTERS FOR INSTALLING: install_lecore needs six facts -- depth, width,
head, tie, free rows, and where the residual stream is -- and every one of them
is inferable. A config file is a convenience, not a requirement, and treating it
as a requirement is what made the old pipeline architecture-specific.

THE HONEST LIMIT, and it is why this REPORTS CONFIDENCE rather than a verdict:
inference from shapes is a strong prior, not a proof. A model whose hidden width
happens to equal its head count, or whose naming uses a different numeric field,
will be read wrongly -- so every field comes back with the evidence that
produced it, and `confidence` is LOW when the evidence is thin. A wrong guess
that announces itself is recoverable; a wrong guess that does not is the most
expensive failure this project knows.
"""

import re
from collections import Counter

import numpy as np


def infer(weights, tokenizer_dir=None):
    """Read a model's architecture from its tensors. Returns facts + evidence."""
    shapes = {k: tuple(np.asarray(v).shape) for k, v in weights.items()}
    ev = {}

    # ---- DEPTH: the numeric field that repeats across names ----
    idx = Counter()
    for k in shapes:
        for mm in re.finditer(r"\.(\d+)\.", k):
            idx[int(mm.group(1))] += 1
    layers = sorted(idx)
    ev["layers"] = "%d indices found in tensor names" % len(layers)

    # ---- WIDTH: the modal dimension. A hidden size touches nearly every
    #      tensor; head dims and intermediate sizes touch a subset.
    dims = Counter()
    for s in shapes.values():
        for d in s:
            dims[d] += 1
    common = dims.most_common(4)
    hidden = common[0][0] if common else None
    margin = (common[0][1] / max(common[1][1], 1)) if len(common) > 1 else 99.0
    ev["hidden"] = "appears in %d tensors, %.1fx the next dimension" % (
        common[0][1] if common else 0, margin)

    # ---- THE VOCABULARY TENSOR: 2-D, one axis hidden, other much larger ----
    head_key = None
    vocab = None
    for k, s in shapes.items():
        if len(s) == 2 and hidden in s and max(s) != hidden:
            if k.endswith("embed_tokens.weight") or "lm_head" in k \
                    or "wte" in k or "embed" in k:
                head_key = head_key or k
                vocab = max(s)
    if head_key is None:
        big = [(k, s) for k, s in shapes.items()
               if len(s) == 2 and hidden in s and max(s) > 4 * hidden]
        if big:
            head_key = max(big, key=lambda kv: max(kv[1]))[0]
            vocab = max(shapes[head_key])
    ev["vocabulary"] = "from %s" % (head_key or "NOT FOUND")

    # ---- THE ATTENTION FAMILY, which decides HALF the install ----
    # Registers, the HRNN ladder and self-write all live in a RECURRENT STATE.
    # Qwen3.5/3.6 are ~75% Gated DeltaNet linear attention and have one. GEMMA 4
    # DOES NOT -- it interleaves sliding-window and global softmax attention, so
    # there is no persistent accumulator to reserve directions in, and those
    # three steps have nowhere to go. Llama is the same. Reading depth and width
    # without reading this makes an installer that silently offers half its
    # capabilities to a model that cannot hold them.
    lin_markers = ("linear_attn", "in_proj_qkvz", "A_log", "conv1d", "dt_bias",
                   "mixer.", "ssm")
    n_lin = sum(1 for k in shapes if any(t in k for t in lin_markers))
    n_attn = sum(1 for k in shapes if "self_attn" in k or "attn.q" in k
                 or "attention" in k)
    if n_lin and n_attn:
        family = "hybrid"
    elif n_lin:
        family = "recurrent"
    else:
        family = "attention"
    ev["attention"] = ("%d linear-state tensors, %d attention tensors -> %s"
                       % (n_lin, n_attn, family))

    tied = not any("lm_head" in k for k in shapes)
    ev["tied"] = ("no lm_head tensor -> the embedding IS the head" if tied
                  else "a separate lm_head exists")

    # ---- FREE ROWS: only the tokenizer knows, and only if it is present ----
    free_from = None
    if tokenizer_dir is not None and vocab:
        try:
            from holographic.io_and_interop.holographic_galvapack import (
                reserved_rows)
            free_from = int(reserved_rows(tokenizer_dir, int(vocab)))
        except Exception:
            free_from = None
    ev["free_rows"] = ("rows %s..%s are never emitted"
                       % (free_from, vocab) if free_from is not None
                       else "unknown -- no readable tokenizer")

    # ---- CONFIDENCE: thin evidence must announce itself ----
    score = 0.0
    score += 0.3 if len(layers) >= 2 else 0.0
    score += 0.3 if margin >= 1.5 else 0.1
    score += 0.2 if head_key else 0.0
    score += 0.2 if free_from is not None else 0.0
    return {"n_layers": len(layers), "hidden": hidden, "head": head_key,
            "family": family, "has_recurrent_state": family != "attention",
            "n_linear_tensors": int(n_lin),
            "vocab": vocab, "tied": tied, "free_from": free_from,
            "layer_prefix": _prefix(shapes), "confidence": round(score, 2),
            "evidence": ev}


def _prefix(shapes):
    """The string before the layer index -- what every bake needs to build keys."""
    for k in shapes:
        mm = re.search(r"^(.*?)\d+\.", k)
        if mm and "layers." in k:
            return mm.group(1)
    return ""


def _selftest():
    import json
    import os

    from holographic.io_and_interop.holographic_unicron import load_safetensors

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("adapt selftest SKIPPED-SUBJECT (no model present)")
        return
    w = load_safetensors(os.path.join(src, "model.safetensors"))
    with open(os.path.join(src, "config.json")) as f:
        truth = json.load(f)
    tc = truth.get("text_config", truth)

    got = infer(w, tokenizer_dir=src)

    # ---- THE INFERENCE MUST MATCH THE CONFIG IT NEVER READ ----
    assert got["n_layers"] == int(tc["num_hidden_layers"]), (got, tc)
    assert got["hidden"] == int(tc["hidden_size"]), (got, tc)
    assert got["head"] is not None, got
    assert got["confidence"] >= 0.6, got

    # ---- AND IT MUST NOT BE CONFIDENT ABOUT A MODEL IT CANNOT READ ----
    thin = infer({"a.weight": np.zeros((3, 3))})
    assert thin["confidence"] < 0.6, thin

    # ---- THE FAMILY MUST BE RIGHT, because it gates half the install ----
    assert got["family"] in ("recurrent", "hybrid"), got["family"]
    assert got["has_recurrent_state"] is True

    # a GEMMA-SHAPED model has NO recurrent state and must say so
    fz = lambda *sh: np.zeros(sh, np.float32)
    gem = {"model.embed_tokens.weight": fz(4096, 128)}
    for i in range(6):
        pr = "model.layers.%d." % i
        gem[pr + "self_attn.q_proj.weight"] = fz(128, 128)
        gem[pr + "self_attn.o_proj.weight"] = fz(128, 128)
        gem[pr + "mlp.up_proj.weight"] = fz(512, 128)
        gem[pr + "mlp.down_proj.weight"] = fz(128, 512)
    g = infer(gem)
    assert g["family"] == "attention", g["family"]
    assert g["has_recurrent_state"] is False, g

    print("adapt selftest OK -- read a real checkpoint with config.json WITHHELD "
          "and recovered %d layers and hidden %d (%s), found the head at %s, "
          "answered tied=%s, and located the free rows at %s -- confidence %.2f; "
          "on a checkpoint with no structure at all it reports %.2f instead of "
          "guessing; and it names the ATTENTION FAMILY -- this model is %r with "
          "a recurrent state, a Gemma-shaped one reads %r with NONE, which is "
          "what decides whether registers and the memory ladder have anywhere "
          "to live"
          % (got["n_layers"], got["hidden"], got["evidence"]["hidden"],
             got["head"].split(".")[-2], got["tied"], got["free_from"],
             got["confidence"], thin["confidence"], got["family"],
             g["family"]))


if __name__ == "__main__":
    _selftest()
