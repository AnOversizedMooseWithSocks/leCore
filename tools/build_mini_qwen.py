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

import argparse
import json
import os
import sys

import numpy as np

# Runnable by path from any directory, including an ilxyr executor workspace.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def official_shape_config():
    """The public Qwen3.5-0.8B text shape, embedded for offline fixtures."""
    layer_types = ["linear_attention" if i % 4 != 3 else "full_attention"
                   for i in range(24)]
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_value_heads": 16,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "layer_types": layer_types,
            "attn_output_gate": True,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
        "vision_config": {"model_type": "qwen3_5_vision"},
    }


def build(out_dir, shrink=8, vocab=512, added=26, seed=0, layers=None,
          real_config=None):
    """Write a miniature but structurally faithful Qwen3.5 checkpoint."""
    from holographic.io_and_interop.holographic_unicron import save_safetensors

    if real_config:
        with open(real_config) as f:
            real = json.load(f)
    else:
        real = official_shape_config()
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

    # A byte-complete tokenizer makes the fixture usable on arbitrary prose.
    # Added tokens occupy 256..281, exactly the boundary reserved_rows must see;
    # the remaining padded rows are genuinely free installation space.
    if int(vocab) < 256 + int(added):
        raise ValueError("vocab must be at least %d for byte tokens + added tokens"
                         % (256 + int(added)))
    from holographic.io_and_interop.holographic_bpe import _byte_encoder
    base = {_byte_encoder()[b]: b for b in range(256)}
    added_tokens = [{"id": 256 + j, "content": "<extra%d>" % j}
                    for j in range(int(added))]
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(base, f)
    with open(os.path.join(out_dir, "merges.txt"), "w") as f:
        f.write("#version: 0.2\n")
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump({"model": {"type": "BPE", "vocab": base, "merges": []},
                   "added_tokens": added_tokens}, f)
    return {"dir": out_dir, "hidden": H, "layers": len(types), "vocab": vocab,
            "tensors": len(w),
            "megabytes": round(os.path.getsize(
                os.path.join(out_dir, "model.safetensors")) / 1e6, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--config", dest="real_config",
                    help="optional real config.json; the official 0.8B shape is embedded")
    ap.add_argument("--shrink", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--added", type=int, default=26)
    ap.add_argument("--layers", type=int, default=4,
                    help="keep the first N layers of the 3-linear/1-full pattern")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    print(json.dumps(build(args.out_dir, shrink=args.shrink, vocab=args.vocab,
                           added=args.added, seed=args.seed, layers=args.layers,
                           real_config=args.real_config), sort_keys=True))


if __name__ == "__main__":
    main()
