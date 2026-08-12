# Qwen3.5-0.8B -- assimilation subject reference (official card, fetched 2026-08-08)

Source: huggingface.co/Qwen/Qwen3.5-0.8B (Apache 2.0, Feb 2026, model_type qwen3_5,
inherits qwen3_next). THIS FILE is the ground truth the rehearsal subject and the
policy encode; update it if the card changes.

## Architecture (language model)
- Causal LM **with vision encoder** (VLM; Image-Text-to-Text pipeline)
- Hidden 1024 | 24 layers | layout 6 x (3 x (GatedDeltaNet -> FFN) -> 1 x (GatedAttention -> FFN))
- Gated DeltaNet: 16 V heads + 16 QK heads, head_dim 128; causal conv1d kernel 4;
  A_log / dt_bias per-head params; in_proj_qkvz + in_proj_ba + out_proj
- Gated Attention: 8 Q heads / 2 KV heads (GQA), head_dim 256, RoPE dim 64;
  q_norm / k_norm
- FFN (dense at this size, no MoE): intermediate 3584, SwiGLU (gate/up/down)
- Token embedding 248,320 padded, LM OUTPUT TIED to embedding (~254M params,
  roughly a third of the model) | MTP trained multi-step | context 262,144
- Checkpoint tensor dtypes: MIXED F32 and BF16 (per-tensor dtype preservation
  is mandatory, not cosmetic)

## Operating notes that shaped our code
- 0.8B runs NON-thinking by default; the card explicitly warns this size is
  prone to degenerate loops and recommends presence_penalty
- Recommended sampling (non-thinking, text): temperature=1.0, top_p=1.0,
  top_k=20, presence_penalty=2.0 (HF generate has no presence_penalty;
  chat.py approximates with repetition_penalty=1.3)
- transformers >= 5.2 required; fast path warnings about fla / causal-conv1d
  are performance-only (torch fallback is correct)

## Policy implications (encoded in SKIP_PATTERNS + regime routing)
- visual.* and mtp.* are UNTOUCHABLE: text perplexity cannot measure damage
  to the vision tower or MTP heads, and we do not transform what we cannot
  measure
- embed/lm_head skip protects ~1/3 of all parameters in one stroke (tied)
- conv1d.* caught by the conv pattern; A_log/dt_bias are 1D (min_dim pass)
- Trained text projections measured HEAVY-TAILED in the field (the 256-newline
  result): regime="auto" passes them through; MP filtering applies only where
  a spectral gap actually exists
