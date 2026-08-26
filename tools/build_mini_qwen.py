"""Build a MINIATURE Qwen3.5 with the REAL structure, for end-to-end testing.

Every pipeline defect this arc has cost a user a test cycle: the hardcoded
tensor root, the free-row miscount that would have eaten the vision tokens, the
dtype upcast that doubled the file, the config nested under text_config, the
bakes that never reached disk. ALL OF THEM ARE STRUCTURAL -- none needed a
0.8-billion-parameter model to reproduce, and none could be reproduced on the
toy, whose tensors are named model.layers.*, which has no vision tower, no tied
embeddings, no added tokens and no bf16.

So this builds the real thing at 1/8 scale: the same tensor names rooted at
model.language_model., the same 24-layer linear/full attention pattern, a vision
tower, TIED EMBEDDINGS (no lm_head tensor at all), added tokens above the plain
vocabulary, and bf16 on disk. Structure faithful, dimensions tiny.
"""

import json
import os

import numpy as np


def build(out_dir, shrink=8, vocab=2048, added=26, seed=0, layers=None,
          real_config="/mnt/user-data/uploads/config.json"):
    """Write a miniature but structurally faithful Qwen3.5 checkpoint."""
    from holographic.io_and_interop.holographic_unicron import save_safetensors

    with open(real_config) as f:
        real = json.load(f)
    tc = dict(real["text_config"])
    H = tc["hidden_size"] // shrink
    I = tc["intermediate_size"] // shrink
    hd = tc["head_dim"] // shrink
    lk = tc["linear_key_head_dim"] // shrink
    lv = tc["linear_value_head_dim"] // shrink
    nq = tc["num_attention_heads"]
    nkv = tc["num_key_value_heads"]
    nlv = tc["linear_num_value_heads"]
    nlk = tc["linear_num_key_heads"]
    types = tc["layer_types"]
    if layers:
        # keep the block PATTERN (linear x3 + full) while shortening, so the
        # structure stays faithful at a size that fits in memory
        types = types[:int(layers)]
    rng = np.random.default_rng(seed)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    w = {"model.language_model.embed_tokens.weight": r(vocab, H),
         "model.language_model.norm.weight": np.ones(H, np.float32)}
    for i, kind in enumerate(types):
        p = "model.language_model.layers.%d." % i
        w[p + "input_layernorm.weight"] = np.ones(H, np.float32)
        w[p + "post_attention_layernorm.weight"] = np.ones(H, np.float32)
        w[p + "mlp.gate_proj.weight"] = r(I, H)
        w[p + "mlp.up_proj.weight"] = r(I, H)
        w[p + "mlp.down_proj.weight"] = r(H, I)
        if kind == "linear_attention":
            w[p + "linear_attn.A_log"] = (rng.standard_normal(nlv) - 3.0
                                          ).astype(np.float32)
            w[p + "linear_attn.dt_bias"] = np.zeros(nlv, np.float32)
            w[p + "linear_attn.in_proj_qkvz.weight"] = r(
                2 * nlk * lk + 2 * nlv * lv, H)
            w[p + "linear_attn.in_proj_ba.weight"] = r(2 * nlv, H)
            w[p + "linear_attn.conv1d.weight"] = r(
                nlk * lk * 2 + nlv * lv, 1, tc["linear_conv_kernel_dim"]
            ).reshape(nlk * lk * 2 + nlv * lv, 1,
                      tc["linear_conv_kernel_dim"])
            w[p + "linear_attn.conv1d.bias"] = np.zeros(
                nlk * lk * 2 + nlv * lv, np.float32)
            w[p + "linear_attn.norm.weight"] = np.ones(lv, np.float32)
            w[p + "linear_attn.out_proj.weight"] = r(H, nlv * lv)
        else:
            w[p + "self_attn.q_proj.weight"] = r(nq * hd * 2, H)
            w[p + "self_attn.k_proj.weight"] = r(nkv * hd, H)
            w[p + "self_attn.v_proj.weight"] = r(nkv * hd, H)
            w[p + "self_attn.o_proj.weight"] = r(H, nq * hd)
            w[p + "self_attn.q_norm.weight"] = np.ones(hd, np.float32)
            w[p + "self_attn.k_norm.weight"] = np.ones(hd, np.float32)
    # a VISION TOWER, because a third of the real model's tensors are one and
    # nothing in this pipeline should touch them
    for i in range(2):
        p = "model.visual.blocks.%d." % i
        w[p + "attn.qkv.weight"] = r(3 * 96, 96)
        w[p + "attn.proj.weight"] = r(96, 96)
        w[p + "mlp.linear_fc1.weight"] = r(192, 96)
        w[p + "mlp.linear_fc2.weight"] = r(96, 192)

    os.makedirs(out_dir, exist_ok=True)
    # BF16 ON DISK, like the real checkpoint -- our loader decodes to float32,
    # which is exactly the asymmetry that doubled a real user's file
    save_safetensors(os.path.join(out_dir, "model.safetensors"),
                     {k: np.ascontiguousarray(v) for k, v in w.items()},
                     dtypes={k: "BF16" for k in w})

    tc.update(hidden_size=H, intermediate_size=I, vocab_size=vocab,
              head_dim=hd, linear_key_head_dim=lk, linear_value_head_dim=lv,
              num_hidden_layers=len(types))
    cfg = {"architectures": real["architectures"],
           "model_type": real["model_type"],
           "text_config": tc,
           "tie_word_embeddings": True,
           "vision_config": real["vision_config"]}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # a tokenizer whose ADDED TOKENS sit above the plain vocab, like the real
    # one -- this is what made "free rows" a dangerous over-count
    plain = vocab - added
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump({"tok%d" % i: i for i in range(plain - 30)}, f)
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump({"model": {"vocab": {"tok%d" % i: i
                                       for i in range(plain - 30)}},
                   "added_tokens": [{"id": plain - 30 + j,
                                     "content": "<extra%d>" % j}
                                    for j in range(30)]}, f)
    return {"dir": out_dir, "hidden": H, "layers": len(types), "vocab": vocab,
            "tensors": len(w),
            "megabytes": round(os.path.getsize(
                os.path.join(out_dir, "model.safetensors")) / 1e6, 2)}


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mini_qwen"))
