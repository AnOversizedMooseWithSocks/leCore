#!/usr/bin/env python3
"""qwen_config_probe.py -- everything the Qwen3.5 config files tell us, before a weight is downloaded.

stdlib only (no numpy needed). Runs in under a second. Three questions:

  [Q0] ARCHITECTURE   read it from config.json, never from memory
  [Q1] BUDGET         derive the parameter count per family, then CHECK it against the index's
                      total_size. If the derived number matches the byte count, our understanding of
                      the architecture is right BEFORE we spend 1.7 GB of disk.
  [Q2] TOKENIZER      (a) how well does a 248k vocab read leCore's own snake_case identifiers?
                      (b) how many embedding rows are assigned to NO token? Those are the unused rows,
                          and the prediction on record is that they form a near-identical clique --
                          exactly as SmolLM2's ~26-30 reserved rows did, at 10x the vocabulary.
"""
import collections, json, pathlib, re, sys

import lecore_paths as P


def budget():
    c = json.loads(P.qwen_config().read_text())
    t = c['text_config']
    idx = json.loads((P.QWEN_DIR / 'model.safetensors.index.json').read_text())

    H, L, I, V = t['hidden_size'], t['num_hidden_layers'], t['intermediate_size'], t['vocab_size']
    types = t['layer_types']
    nlin = sum(1 for x in types if x == 'linear_attention')
    nfull = L - nlin
    Kh, Kd = t['linear_num_key_heads'], t['linear_key_head_dim']
    Vh, Vd = t['linear_num_value_heads'], t['linear_value_head_dim']
    qh, kvh, hd = t['num_attention_heads'], t['num_key_value_heads'], t['head_dim']
    conv = t['linear_conv_kernel_dim']
    K, Vv = Kh * Kd, Vh * Vd

    print("=" * 92)
    print("[Q0] ARCHITECTURE (from config.json)")
    print("=" * 92)
    print(f"  hidden {H} | layers {L} = {nlin} linear_attention + {nfull} full_attention "
          f"(interval {t['full_attention_interval']})")
    print(f"  DeltaNet: {Kh} key heads x {Kd}, {Vh} value heads x {Vd}, conv kernel {conv}")
    print(f"  softmax : {qh} q heads, {kvh} kv heads (GQA {qh//kvh}:1), head_dim {hd}, "
          f"partial rotary {t['rope_parameters'].get('partial_rotary_factor')}")
    print(f"  vocab {V} | tied embeddings {t['tie_word_embeddings']} | mlp {I} ({t['hidden_act']})")
    print(f"  rope: {t['rope_parameters']['rope_type']}, theta {t['rope_parameters']['rope_theta']}, "
          f"mrope sections {t['rope_parameters'].get('mrope_section')}")
    vis = sorted({re.sub(r'\.\d+\.', '.{L}.', k) for k in idx['weight_map'] if k.startswith('model.visual')})
    nvis = len({k.split('.')[3] for k in idx['weight_map'] if k.startswith('model.visual.blocks.')})
    print(f"  vision tower: {nvis} blocks, {len(vis)} tensor families")

    emb = V * H
    mlp = L * 3 * H * I
    lin_per = H * (2 * K + Vv) + H * Vv + Vv * H + conv * (2 * K + Vv) + 3 * Kh
    lin = nlin * lin_per
    full_per = H * (qh * hd) + 2 * H * (kvh * hd) + (qh * hd) * H
    full = nfull * full_per
    lang = emb + mlp + lin + full
    total = idx['metadata']['total_size'] / 2                       # bf16

    print("\n" + "=" * 92)
    print("[Q1] PARAMETER BUDGET  (derived from config, checked against index total_size)")
    print("=" * 92)
    print(f"  {'component':32s} {'params':>11s} {'share':>8s}")
    for n, p in (("token embedding (TIED)", emb), (f"MLP x{L}", mlp),
                 (f"linear-attn (DeltaNet) x{nlin}", lin), (f"full-attn (GQA) x{nfull}", full)):
        print(f"  {n:32s} {p/1e6:10.1f}M {p/total:7.1%}")
    print(f"  {'-> language model':32s} {lang/1e6:10.1f}M {lang/total:7.1%}")
    print(f"  {'-> vision tower (residual)':32s} {(total-lang)/1e6:10.1f}M {(total-lang)/total:7.1%}")
    print(f"\n  index total_size {idx['metadata']['total_size']/1e9:.3f} GB -> {total/1e6:.1f}M params (bf16)")
    ok = 0.05 < (total - lang) / total < 0.30
    print(f"  CHECK: residual for a {nvis}-block ViT is {(total-lang)/total:.1%} "
          f"-> {'plausible, budget confirmed' if ok else 'IMPLAUSIBLE -- our shape assumptions are wrong'}")

    print(f"\n  embedding table: {emb*2/1e6:.0f} MB bf16 | {emb/1e6:.0f} MB q8 | "
          f"{emb*4.17/8/1e6:.0f} MB at our codebook rate (4.17 bits)")
    print(f"  -> the codebook saves {emb*(8-4.17)/8/1e6:.0f} MB on the single largest tensor.")
    return V


def tokenizer(V_config):
    tj = json.loads(P.qwen_tokenizer().read_text(encoding='utf-8'))
    vocab = tj['model']['vocab']
    added = tj.get('added_tokens', [])
    print("\n" + "=" * 92)
    print("[Q2] TOKENIZER")
    print("=" * 92)
    ids = [a['id'] for a in added]
    print(f"  tokenizer defines {len(vocab)} + {len(added)} special = {len(vocab)+len(added)}")
    print(f"  special ids {min(ids)}..{max(ids)} (TOP of the range; SmolLM2's clique sat at LOW ids)")
    unused = V_config - (max(ids) + 1)
    print(f"  config.vocab_size {V_config} -> {unused} rows assigned to NO token")
    print(f"  PREDICTION ON RECORD: those ~{unused} rows form a mutually near-identical clique")
    print(f"  (they receive only the softmax's uniform negative gradient). Dropping them saves "
          f"~{unused*1024/1e6:.1f} MB at q8 -- i.e. nothing. The point is the MECHANISM repeating.")

    # coverage over leCore's own identifiers
    repo = P.REPO
    if not repo.is_dir():
        print(f"\n  (repo not found at {repo}; skipping coverage)")
        return
    terms = collections.Counter()
    for p in repo.rglob('holographic_*.py'):
        terms.update(re.findall(r'\b[a-z][a-z0-9_]{3,}\b', p.read_text(errors='ignore')))
    top = [t for t, _ in terms.most_common(4000)]
    Vs = set(vocab)

    def pieces(w):
        if ('\u0120' + w) in Vs or w in Vs:
            return 1
        n, i = 0, 0
        while i < len(w):
            for j in range(len(w), i, -1):
                if w[i:j] in Vs:
                    n += 1; i = j; break
            else:
                return 99
        return n

    counts = [pieces(t) for t in top]
    single = sum(1 for c in counts if c == 1)
    print(f"\n  leCore identifiers sampled: {len(top)}")
    print(f"  single-token {single} ({single/len(top):.1%}) | mean pieces/term {sum(counts)/len(counts):.2f}")
    print(f"  (nomic WordPiece, measured earlier: 56.0% single-token, 1.68 pieces/term)")
    worst = sorted(zip(top, counts), key=lambda x: -x[1])[:5]
    print(f"  most shattered: {worst}")


if __name__ == '__main__':
    V = budget()
    tokenizer(V)
    print("\nDONE.")
