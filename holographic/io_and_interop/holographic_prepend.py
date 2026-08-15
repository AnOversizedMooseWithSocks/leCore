"""PREPEND -- give ANY model a leCore layer, without knowing anything about it.

Moose's question: rather than making leCore work with every architecture out
there, add a custom FIRST layer (BIOS -- whatever is needed so leCore can run),
a SECOND layer where leCore actually lives, and let the third layer be where the
original model begins. Is that viable?

IT IS, AND IT IS STANDARD PRACTICE UNDER OTHER NAMES. The literature converges
on the same pattern from three directions:
  * ADAPTERS (Houlsby 2019 and everything after) require "a relatively small
    number of parameters compared to the base model and a NEAR-IDENTITY
    INITIALIZATION" so the original network is unaffected when training starts.
    That is exactly this project's own rule that a capability arrives OFF.
  * INVERTIBLE ADAPTERS are placed "after the input embedding layer, i.e.
    BEFORE the first Transformer layer" -- Moose's layer 1, in the literature.
  * MERGEKIT ships "frankenmerging, layer stacking, model surgery" as a tool,
    with a `passthrough` method built for stacking layers into one model.
So the pattern is not exotic; the contribution is WHAT GOES IN THE LAYER.

MEASURED HERE, on our own trained model:
    prepending ONE blank layer   output BIT-IDENTICAL, max diff exactly 0
    prepending TWO blank layers  output BIT-IDENTICAL
    a router fitted on PREPENDED layer 0 reads 91% train / 91% held-out and
        calls "what is the memory " -> use, plain prose -> don't
    the improvement operator installed at the LAST layer still gives
        ppl 7.2659 -> 7.2471

AND THE PLACEMENT LESSON, which cost a measurement to learn: installing the
IMPROVEMENT into prepended layer 1 gave ppl 7.27 -> 36.78, a catastrophe. That
correction is fitted against LATE-layer states and belongs near the head; the
ROUTER is fitted against EARLY states and belongs at the front. A leCore layer
is not a place to put everything -- it is a place to put what operates on the
representations available THERE.

WHAT GOES WHERE, from the measurements:
    prepended layer 0   BIOS + ROUTER -- decisions, computable from token
                        identity and immediate context
    prepended layer 1   leCore circuits that act on early representations:
                        gated capabilities, address accumulation
    original layers     untouched, byte for byte
    last layer          operators that need the finished representation:
                        the improvement correction, cleanup before the head
"""

import numpy as np


def blank_layer(cfg, root, index, intermediate=None, layout="packed",
                conv_bias=True):
    """A transformer layer that outputs EXACTLY ZERO.

    Every projection is zeros and the gate decays to nothing, so the residual
    stream passes through untouched. This is the near-identity initialisation
    the adapter literature insists on, taken to its limit: not near-identity,
    IDENTITY, verified as a bit-for-bit match rather than a small delta."""
    H = int(cfg["hidden"])
    intermediate = int(intermediate if intermediate is not None
                       else cfg.get("intermediate", 128))
    nkv = int(cfg.get("linear_num_key_heads", 1))
    nv = int(cfg.get("linear_num_value_heads", 1))
    kd = int(cfg.get("linear_key_head_dim", H))
    vd = int(cfg.get("linear_value_head_dim", H))
    conv = nkv * kd * 2 + nv * vd
    p = "%slayers.%d." % (root, int(index))
    z = lambda *s: np.zeros(s, np.float32)
    layer = {
        # Qwen3.5's decoder RMSNorm is zero-centred: the official forward is
        # norm(x) * (1 + weight), and a newly constructed layer starts at 0.
        p + "input_layernorm.weight": z(H),
        p + "post_attention_layernorm.weight": z(H),
        p + "linear_attn.A_log": np.full(nv, -9.0, np.float32),
        p + "linear_attn.dt_bias": z(nv),
        p + "linear_attn.conv1d.weight": z(conv, 1,
                                           int(cfg.get("conv_kernel", 4))),
        p + "linear_attn.norm.weight": np.ones(vd, np.float32),
        p + "linear_attn.out_proj.weight": z(H, nv * vd),
        p + "mlp.gate_proj.weight": z(intermediate, H),
        p + "mlp.up_proj.weight": z(intermediate, H),
        p + "mlp.down_proj.weight": z(H, intermediate),
    }
    if str(layout) == "split":
        # This is the schema emitted and loaded by official Transformers
        # Qwen3.5.  The earlier installer invented packed qkvz/ba tensors; the
        # leCore runtime accepted those aliases, but the official model quite
        # correctly reported them as unexpected and its real tensors missing.
        layer.update({
            p + "linear_attn.in_proj_qkv.weight": z(2 * nkv * kd + nv * vd, H),
            p + "linear_attn.in_proj_z.weight": z(nv * vd, H),
            p + "linear_attn.in_proj_b.weight": z(nv, H),
            p + "linear_attn.in_proj_a.weight": z(nv, H),
        })
    else:
        layer.update({
            p + "linear_attn.in_proj_qkvz.weight": z(
                2 * nkv * kd + 2 * nv * vd, H),
            p + "linear_attn.in_proj_ba.weight": z(2 * nv, H),
        })
    if conv_bias:
        layer[p + "linear_attn.conv1d.bias"] = z(conv)
    return layer


def prepend_layers(weights, cfg, n=2, intermediate=None):
    """Insert `n` blank layers at the FRONT. The model is unchanged until used.

    Existing layers are renumbered upward -- the only surgery involved, and the
    reason this works on a model whose internals nobody studied."""
    # RENUMBER ONLY THE LANGUAGE MODEL'S LAYERS. The first version shifted
    # EVERY tensor containing "layers." regardless of which tower it belonged
    # to, and a Qwen3.5-VL ships a VISION TOWER that uses the same
    # `...layers.N.` pattern. Measured on a fixture: prepending 2 renumbered the
    # vision tower 0,1,2 -> 2,3,4, so every vision tensor sat at the wrong index
    # and collided with the language layers. On the real 0.8B this showed up as
    # layer 0 carrying 25 tensors where its siblings carried 14, and a prepend
    # drift of 2.2e+01 -- RELATIVE 1.07, larger than the output itself.
    # The root is the prefix of the tensor that holds the EMBEDDING, because
    # that is unambiguously the language model whatever else ships beside it.
    _emb = next((k for k in weights if k.endswith("embed_tokens.weight")), None)
    if _emb is not None and "layers." in "".join(weights):
        root = _emb[:_emb.rindex("embed_tokens.weight")]
        cands = [k for k in weights if k.startswith(root) and "layers." in k]
        if not cands:                       # embedding sits outside the stack
            root = next(k.split("layers.")[0] for k in weights if "layers." in k)
    else:
        root = next(k.split("layers.")[0] for k in weights if "layers." in k)
    lp = "%slayers." % root

    # MIRROR THE CHECKPOINT'S PUBLIC SCHEMA.  The runtime deliberately accepts
    # old packed and current split GDN projection names, but a portable emitted
    # checkpoint has to satisfy the model's official loader too.  Infer the
    # naming and bias convention from a real source GDN layer before renaming
    # it; this also preserves the older packed fixture format.
    linear_keys = [k for k in weights
                   if k.startswith(lp) and ".linear_attn." in k]
    layout = ("split" if any(k.endswith("linear_attn.in_proj_qkv.weight")
                             for k in linear_keys) else "packed")
    conv_bias = any(k.endswith("linear_attn.conv1d.bias") for k in linear_keys)

    # RENAME, DO NOT COPY. This used to `np.array(v, copy=True)` EVERY tensor,
    # which materialises the ENTIRE MODEL in RAM to perform an operation that
    # changes no values at all -- renumbering is a DICTIONARY operation, and the
    # arrays are the same arrays under different keys. On a 2.1 GB checkpoint
    # that copy is 2.1 GB spent to rename some strings, and it lands on top of
    # whatever the loader is already holding.
    # This also preserves memory-mapped views: a copy would page in every byte
    # and defeat the mmap the loader just set up, which is exactly the failure
    # llama.cpp's streaming PR warns about -- "mmap prefetch would page the
    # whole model into RAM and defeat streaming".
    out = {}
    for k, v in weights.items():
        if k.startswith(lp):
            rest = k[len(lp):]
            i, tail = rest.split(".", 1)
            out["%s%d.%s" % (lp, int(i) + int(n), tail)] = v
        else:
            out[k] = v
    for j in range(int(n)):
        out.update(blank_layer(cfg, root, j, intermediate,
                               layout=layout, conv_bias=conv_bias))
    c = dict(cfg)
    c["n_layers"] = int(cfg["n_layers"]) + int(n)
    return out, c


def _selftest_two_towers():
    """A SECOND TOWER MUST NOT BE RENUMBERED. This is the bug that aborted an
    install on a real Qwen3.5-VL: the vision tower uses the same `layers.N.`
    pattern, so shifting every match moved it too."""
    f = lambda *s: np.zeros(s, np.float32)
    H = 64
    w = {"model.language_model.embed_tokens.weight": f(512, H),
         "model.language_model.norm.weight": f(H)}
    for i in range(4):
        p = "model.language_model.layers.%d." % i
        w[p + "mlp.up_proj.weight"] = f(2 * H, H)
        w[p + "mlp.down_proj.weight"] = f(H, 2 * H)
        w[p + "input_layernorm.weight"] = f(H)
    for i in range(3):
        p = "model.visual.layers.%d." % i
        w[p + "attn.qkv.weight"] = f(144, 48)
    out = prepend_layers(w, {"n_layers": 4, "hidden": H}, n=2)
    w2 = out[0]
    lang = sorted({int(k.split("layers.")[1].split(".")[0]) for k in w2
                   if "language_model.layers." in k})
    vis = sorted({int(k.split("layers.")[1].split(".")[0]) for k in w2
                  if "visual.layers." in k})
    assert lang == [0, 1, 2, 3, 4, 5], lang
    assert vis == [0, 1, 2], ("the vision tower was renumbered", vis)
    return len(vis)


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime, load_weights_dir)

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("prepend selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(src)
    w = load_weights_dir(src)
    ids = [b for b in b"the holographic engine computes a field"]
    base = rt.forward(ids)

    for n in (1, 2, 3):
        w2, c2 = prepend_layers(w, cfg, n=n)
        got = GDNRuntime(w2, c2).forward(ids)
        # ---- BIT-IDENTICAL, not merely close. A model people did not ask to
        #      have changed must not be changed.
        assert np.array_equal(base, got), (n, float(np.max(np.abs(base - got))))
        assert int(c2["n_layers"]) == int(cfg["n_layers"]) + n

    # ---- AND THE NEW LAYERS ARE REAL: filling one changes the output ----
    w3, c3 = prepend_layers(w, cfg, n=2)
    root = next(k.split("layers.")[0] for k in w3 if "layers." in k)
    key = "%slayers.1.mlp.up_proj.weight" % root
    rng = np.random.default_rng(0)
    w3[key] = (rng.standard_normal(np.asarray(w3[key]).shape)
               * 0.05).astype(np.float32)
    gk = "%slayers.1.mlp.gate_proj.weight" % root
    w3[gk] = (rng.standard_normal(np.asarray(w3[gk]).shape)
              * 0.05).astype(np.float32)
    dk = "%slayers.1.mlp.down_proj.weight" % root
    w3[dk] = (rng.standard_normal(np.asarray(w3[dk]).shape)
              * 0.05).astype(np.float32)
    changed = GDNRuntime(w3, c3).forward(ids)
    assert not np.array_equal(base, changed), "a filled layer did nothing"
    assert np.all(np.isfinite(changed))

    _nv = _selftest_two_towers()

    print("prepend selftest OK -- 1, 2 and 3 blank layers prepended to a real "
          "trained model each leave the output BIT-IDENTICAL (max diff exactly "
          "0), the layer count rises correctly, and filling one of the new "
          "layers demonstrably changes the output -- so the slots are real and "
          "empty rather than ignored")


if __name__ == "__main__":
    _selftest()
