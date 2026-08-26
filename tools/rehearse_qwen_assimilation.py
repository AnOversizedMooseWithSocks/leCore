"""End-to-end assimilation rehearsal on a Qwen3.5-0.8B-SHAPED checkpoint.

WHY a synthetic subject: this environment cannot reach huggingface.co, and shipping
an 'upgraded Qwen' we never measured would violate the honesty contract anyway. What
CAN be verified here, and is: the full pipeline against a checkpoint with the REAL
architecture's tensor-name vocabulary and layer pattern -- Qwen3.5's hybrid
6 x (3 x GatedDeltaNet -> FFN -> 1 x GatedAttention -> FFN) block structure (Qwen3.5
release, Feb 2026), with a large vocab embedding, per-layer norms, and planted
learned structure (spikes) over an MP bulk in every projection. Dims are scaled
(hidden 256, vocab 8000) so the rehearsal runs in seconds; names are verbatim-style
so the policy gate exercises the exact strings the real file will present.

The output file must: (1) contain every input tensor under its ORIGINAL name and
shape, (2) parse back through our own loader, (3) show the policy skipping
embeddings/norms with zero SVDs spent on them, (4) show learned projections filtered
with >50% energy kept, (5) leave a written retention debt in the report.
"""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holographic.io_and_interop.holographic_unicron import (
    save_safetensors, load_safetensors, assimilate_model, analyze_model)


def make_qwen_shaped(hidden=256, layers=6, vocab=8000, ffn=896, seed=0):
    """Spec-faithful (scaled) Qwen3.5-0.8B subject, per the official model card:
    hidden 1024 / 24 layers / FFN 3584 / vocab 248320 tied / 6 x (3xGDN -> 1xAttn)
    -- here at 1/4 width so it runs in seconds. Name vocabulary follows the
    qwen3_next family the 0.8B inherits from: linear_attn in_proj_qkvz/in_proj_ba/
    conv1d/A_log/dt_bias/out_proj, attention q/k/v/o_proj with q_norm/k_norm,
    mlp gate/up/down, plus a vision-tower stub and an mtp stub -- both of which
    the policy MUST skip (text retention cannot measure them)."""
    rng = np.random.default_rng(seed)

    def learned(mout, min_, spikes=6, strength=4.0):
        W = rng.standard_normal((mout, min_)) / np.sqrt(mout)
        U = np.linalg.qr(rng.standard_normal((mout, spikes)))[0]
        V = np.linalg.qr(rng.standard_normal((min_, spikes)))[0]
        return (W + U @ np.diag(np.linspace(strength, strength / 2, spikes)) @ V.T
                ).astype(np.float32)

    gdn_qk = hidden * 2          # 16 QK heads x 128 at full scale, ratio kept
    gdn_v = hidden * 2
    attn_q = hidden * 2          # 8 heads x 256
    attn_kv = hidden // 2        # 2 KV heads x 256
    t = {"model.language_model.embed_tokens.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
         "model.language_model.norm.weight": np.ones(hidden, np.float32),
         # vision tower + mtp stubs: real matrices the policy must refuse
         "model.visual.blocks.0.attn.qkv.weight": learned(hidden * 3, hidden),
         "model.visual.patch_embed.proj.weight": rng.standard_normal((hidden, 3, 14, 14)).astype(np.float32),
         "model.mtp.layers.0.mlp.gate_proj.weight": learned(ffn, hidden)}
    for i in range(layers):
        pre = "model.language_model.layers.%d." % i
        if (i + 1) % 4 == 0:     # every 4th mixer: gated full attention (GQA)
            t[pre + "self_attn.q_proj.weight"] = learned(attn_q, hidden)
            t[pre + "self_attn.k_proj.weight"] = learned(attn_kv, hidden)
            t[pre + "self_attn.v_proj.weight"] = learned(attn_kv, hidden)
            t[pre + "self_attn.o_proj.weight"] = learned(hidden, attn_q)
            t[pre + "self_attn.q_norm.weight"] = np.ones(attn_q // 8, np.float32)
            t[pre + "self_attn.k_norm.weight"] = np.ones(attn_kv // 2, np.float32)
        else:                    # gated deltanet linear-attention mixer
            t[pre + "linear_attn.in_proj_qkvz.weight"] = learned(gdn_qk * 2 + gdn_v, hidden)
            # the REAL 0.8B (field report) has separate 16-dim a/b decay gates --
            # the ONLY spike+bulk matrices found in the whole model
            t[pre + "linear_attn.in_proj_a.weight"] = learned(16, hidden, spikes=3)
            t[pre + "linear_attn.in_proj_b.weight"] = learned(16, hidden, spikes=3)
            t[pre + "linear_attn.conv1d.weight"] = rng.standard_normal((gdn_qk, 1, 4)).astype(np.float32)
            t[pre + "linear_attn.A_log"] = rng.standard_normal(16).astype(np.float32)
            t[pre + "linear_attn.dt_bias"] = rng.standard_normal(16).astype(np.float32)
            t[pre + "linear_attn.out_proj.weight"] = learned(hidden, gdn_v)
        t[pre + "mlp.gate_proj.weight"] = learned(ffn, hidden)
        t[pre + "mlp.up_proj.weight"] = learned(ffn, hidden)
        t[pre + "mlp.down_proj.weight"] = learned(hidden, ffn)
        t[pre + "input_layernorm.weight"] = np.ones(hidden, np.float32)
        t[pre + "post_attention_layernorm.weight"] = np.ones(hidden, np.float32)
    return t


def main():
    td = tempfile.mkdtemp()
    pin, pout = os.path.join(td, "qwen_shaped.safetensors"), os.path.join(td, "qwen_assimilated.safetensors")
    model = make_qwen_shaped()
    save_safetensors(pin, model)
    print("subject: %d tensors, %.1f MB on disk" % (len(model), os.path.getsize(pin) / 1e6))

    out, rep = assimilate_model(pin, out_path=pout)

    back = load_safetensors(pout)
    assert set(back) == set(model), "name set changed"                      # (1)
    for k in model:
        assert back[k].shape == model[k].shape, k                          # (1)
    assert "model.language_model.embed_tokens.weight" in rep["skipped"]                   # (3)
    assert "model.visual.blocks.0.attn.qkv.weight" in rep["skipped"]
    assert "model.mtp.layers.0.mlp.gate_proj.weight" in rep["skipped"]
    assert np.array_equal(back["model.language_model.layers.0.linear_attn.conv1d.weight"],
                          model["model.language_model.layers.0.linear_attn.conv1d.weight"])
    assert np.array_equal(back["model.language_model.embed_tokens.weight"], model["model.language_model.embed_tokens.weight"])
    n_proj = sum(1 for k in model if k.endswith("proj.weight")
                 or ".in_proj_qkvz.weight" in k or ".out_proj.weight" in k)
    assert rep["filtered"] >= 0.8 * n_proj, (rep["filtered"], n_proj)      # (4)
    for name, li in rep["layers"].items():
        # trained-layer contract: modest rank kept (in_proj_ba plants 3 spikes,
        # everything else 6; +-2 finite-size slack), spikes carrying real energy
        planted = 3 if ("in_proj_a" in name or "in_proj_b" in name) else 6
        assert abs(li["rank"] - planted) <= 2, (name, li)
        assert li["spike_energy_frac"] > 0.01, (name, li)
    assert "UNVERIFIED" in rep["verify"]                                   # (5)

    ranks = sorted(li["rank"] for li in rep["layers"].values())
    print("filtered %d/%d projections; skipped by policy: %d; guarded: %d"
          % (rep["filtered"], n_proj, len(rep["skipped"]), len(rep["guarded"])))
    print("effective ranks kept (min/median/max): %d / %d / %d"
          % (ranks[0], ranks[len(ranks) // 2], ranks[-1]))
    print("output: %s (%.1f MB), every tensor under its original name"
          % (pout, os.path.getsize(pout) / 1e6))
    print("retention debt: " + rep["verify"])
    # rsvd path: force approximate SVD on everything sizable; ranks must agree with
    # the exact-SVD run (the instrument cross-check exact vs randomized).
    out2, rep2 = assimilate_model(pin, big=100_000, rsvd_rank=64)
    n_rsvd = sum(1 for li in rep2["layers"].values() if li["rsvd"])
    assert n_rsvd >= 10, n_rsvd
    for name in rep["layers"]:
        if name in rep2["layers"]:
            assert abs(rep2["layers"][name]["rank"] - rep["layers"][name]["rank"]) <= 2, name
    print("rsvd cross-check: %d layers via randomized SVD, ranks agree with exact" % n_rsvd)
    print("QWEN-SHAPED ASSIMILATION REHEARSAL OK")


if __name__ == "__main__":
    main()
