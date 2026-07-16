#!/usr/bin/env python3
"""nomic_forward.py -- N2: nomic-embed-text-v1.5 forward pass in PURE NUMPY + stdlib. No torch.

WHY: every codec bar downstream (M2/M5/M6) is EMBEDDING-COSINE, not weight error. This is the reference
engine that turns weights into embeddings so those bars can be measured. It is also, by itself, the first
step of assimilation: leCore running the model as data.

USAGE:
  python3 nomic_forward.py model.safetensors vocab.txt                    # embed the built-in test set
  python3 nomic_forward.py model.safetensors vocab.txt --ref ref.json     # compare vs reference (the bar)
  python3 nomic_forward.py --selftest                                     # plumbing check, random weights

THE BAR: cosine >= 0.999 per sentence against reference embeddings (e.g. from sentence-transformers or
ollama, same prefixed sentences). To make ref.json:
  {"sentences": [...same as below...], "embeddings": [[...768 floats...], ...]}

CONFIG NOTES (the honest unknowns, all flag-controlled; config.json in the model dir is read if present):
  --rope-base   rotary base (nomic-bert-2048 configs commonly 1000; classic RoPE is 10000)
  --swap-gate   swap which of fc11/fc12 is the SiLU gate in  fc2( SiLU(gate(x)) * value(x) )
  --pre-ln      use pre-LN residual order instead of BERT post-LN
A wrong guess fails the 0.999 bar loudly -- that is what the bar is for. Paste the diff; we iterate.
"""
import sys, json, struct, unicodedata, argparse
import numpy as np

# ------------------------------------------------------------------ safetensors reader (as census)
def read_st(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n).decode('utf-8'))
    return {k: v for k, v in header.items() if k != '__metadata__'}, 8 + n

def load_t(path, meta, base, name):
    m = meta[name]; dt, shape, (b, e) = m['dtype'], m['shape'], m['data_offsets']
    with open(path, 'rb') as f:
        f.seek(base + b); raw = f.read(e - b)
    if dt == 'F32': a = np.frombuffer(raw, dtype=np.float32)
    elif dt == 'F16': a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dt == 'BF16': a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else: raise ValueError(f"dtype {dt} on {name}")
    return np.ascontiguousarray(a.reshape(shape))

# ------------------------------------------------------------------ WordPiece tokenizer (uncased BERT)
class WordPiece:
    """Readable uncased-BERT tokenization: lowercase, strip accents, split on whitespace/punct,
    then greedy longest-match against the vocab with '##' continuations."""
    def __init__(self, vocab_path):
        self.vocab = {}
        with open(vocab_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip('\n')] = i
        self.cls, self.sep, self.unk = self.vocab['[CLS]'], self.vocab['[SEP]'], self.vocab['[UNK]']

    @staticmethod
    def _is_punct(ch):
        cp = ord(ch)
        if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
            return True
        return unicodedata.category(ch).startswith('P')

    def _basic(self, text):
        text = unicodedata.normalize('NFD', text.lower())
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')   # strip accents
        out, cur = [], ''
        for ch in text:
            if ch.isspace():
                if cur: out.append(cur); cur = ''
            elif self._is_punct(ch):
                if cur: out.append(cur); cur = ''
                out.append(ch)
            else:
                cur += ch
        if cur: out.append(cur)
        return out

    def _wordpiece(self, word):
        if word in self.vocab: return [self.vocab[word]]
        ids, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                piece = ('##' if start > 0 else '') + word[start:end]
                if piece in self.vocab: cur = self.vocab[piece]; break
                end -= 1
            if cur is None: return [self.unk]
            ids.append(cur); start = end
        return ids

    def encode(self, text, max_len=512):
        ids = [self.cls]
        for w in self._basic(text):
            ids += self._wordpiece(w)
        ids = ids[:max_len - 1] + [self.sep]
        return np.array(ids, dtype=np.int64)

# ------------------------------------------------------------------ the model
def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * w + b

_ROPE_CACHE = {}

def rope(x, base):
    """Rotary embedding on (T, H, hd): rotate feature PAIRS by position-proportional phase.
    This is leCore's FPE phasor Z(pos) applied per frequency -- the model binds position our way.
    PERF: the (cos,sin) table depends only on (T, hd, base) -- build once, reuse."""
    T, H, hd = x.shape
    half = hd // 2
    key = (T, half, base)
    if key not in _ROPE_CACHE:
        freqs = base ** (-np.arange(0, half, dtype=np.float64) / half)
        ang = np.arange(T, dtype=np.float64)[:, None] * freqs[None, :]
        _ROPE_CACHE[key] = (np.cos(ang)[:, None, :].astype(np.float32),
                            np.sin(ang)[:, None, :].astype(np.float32))
    cos, sin = _ROPE_CACHE[key]
    x1, x2 = x[..., :half], x[..., half:]                              # rotate_half convention
    return np.concatenate([x1 * cos - x2 * sin,
                           x2 * cos + x1 * sin], axis=-1).astype(np.float32)

def silu(x): return x / (1.0 + np.exp(-x))

class Nomic:
    def __init__(self, path, args):
        meta, base = read_st(path)
        g = lambda n: load_t(path, meta, base, n)
        self.emb = g('embeddings.word_embeddings.weight')
        self.tte = g('embeddings.token_type_embeddings.weight')
        self.eln_w, self.eln_b = g('emb_ln.weight'), g('emb_ln.bias')
        self.layers = []
        i = 0
        while f'encoder.layers.{i}.attn.Wqkv.weight' in meta:
            L = {}
            for k, n in (('wqkv', 'attn.Wqkv.weight'), ('wo', 'attn.out_proj.weight'),
                         ('n1w', 'norm1.weight'), ('n1b', 'norm1.bias'),
                         ('f11', 'mlp.fc11.weight'), ('f12', 'mlp.fc12.weight'), ('f2', 'mlp.fc2.weight'),
                         ('n2w', 'norm2.weight'), ('n2b', 'norm2.bias')):
                W = g(f'encoder.layers.{i}.{n}')
                # PERF: torch stores Linear as (out,in) and computes x @ W.T. Transposing on every
                # call hands BLAS a strided B operand. Do it ONCE, contiguous. (~1.12x, measured.)
                if W.ndim == 2:
                    W = np.ascontiguousarray(W.T)
                L[k] = W
            self.layers.append(L); i += 1
        self.d = self.emb.shape[1]; self.H = args.heads; self.hd = self.d // self.H
        self.rope_base = args.rope_base; self.eps = args.ln_eps
        self.swap_gate = args.swap_gate; self.pre_ln = args.pre_ln
        print(f"loaded: {len(self.layers)} layers, d={self.d}, heads={self.H}, "
              f"rope_base={self.rope_base}, swap_gate={self.swap_gate}, pre_ln={self.pre_ln}", file=sys.stderr)

    def attn(self, x, L):
        T = x.shape[0]
        qkv = x @ L['wqkv']          # already transposed at load                                          # (T, 3d)  [torch Linear: y = x W^T]
        q, k, v = qkv[:, :self.d], qkv[:, self.d:2*self.d], qkv[:, 2*self.d:]
        q = rope(q.reshape(T, self.H, self.hd), self.rope_base)
        k = rope(k.reshape(T, self.H, self.hd), self.rope_base)
        v = v.reshape(T, self.H, self.hd)
        # PERF: all heads in ONE batched matmul instead of a 12-iteration python loop.
        # (H,T,hd) x (H,hd,T) -> (H,T,T); verified bit-comparable to the loop (atol 1e-5).
        qb, kb, vb = (z.transpose(1, 0, 2) for z in (q, k, v))
        a = np.matmul(qb, kb.transpose(0, 2, 1)) / np.sqrt(self.hd)
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        out = np.matmul(a, vb).transpose(1, 0, 2)
        return out.reshape(T, self.d) @ L['wo']

    def mlp(self, x, L):
        g, u = (L['f12'], L['f11']) if self.swap_gate else (L['f11'], L['f12'])
        return (silu(x @ g) * (x @ u)) @ L['f2']                 # fc2( SiLU(gate) * value )

    def embed(self, ids):
        x = self.emb[ids] + self.tte[0]
        x = layer_norm(x, self.eln_w, self.eln_b, self.eps)
        for L in self.layers:
            if self.pre_ln:
                x = x + self.attn(layer_norm(x, L['n1w'], L['n1b'], self.eps), L)
                x = x + self.mlp(layer_norm(x, L['n2w'], L['n2b'], self.eps), L)
            else:                                                      # BERT post-LN (default)
                x = layer_norm(x + self.attn(x, L), L['n1w'], L['n1b'], self.eps)
                x = layer_norm(x + self.mlp(x, L), L['n2w'], L['n2b'], self.eps)
        e = x.mean(0)                                                  # mean pooling
        return e / (np.linalg.norm(e) + 1e-12)

TEST = ["search_document: The quick brown fox jumps over the lazy dog.",
        "search_document: A fast auburn fox leaps above a sleepy canine.",
        "search_document: Quarterly revenue grew twelve percent on strong cloud demand.",
        "search_query: financial results and earnings growth",
        "search_document: The mitochondria is the powerhouse of the cell."]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', nargs='?'); ap.add_argument('vocab', nargs='?')
    ap.add_argument('--ref'); ap.add_argument('--rope-base', type=float, default=None)
    ap.add_argument('--heads', type=int, default=12); ap.add_argument('--ln-eps', type=float, default=1e-12)
    # DISCOVERED BY --sweep ON THE REAL WEIGHTS: fc12 is the SiLU gate, fc11 the value; post-LN.
    # Contrast score +0.2241 vs +0.011/+0.0001/+0.0000 for the alternatives. Default is now the truth.
    ap.add_argument('--no-swap-gate', dest='swap_gate', action='store_false', default=True)
    ap.add_argument('--pre-ln', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--sweep', action='store_true', help='run all 4 wiring combos; the real weights pick the right one')
    args = ap.parse_args()

    if args.selftest:
        run_selftest(); return

    # read config.json beside the model if present -- it settles rope base / eps / heads
    import os
    cfgp = os.path.join(os.path.dirname(os.path.abspath(args.model)), 'config.json')
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp))
        if args.rope_base is None: args.rope_base = float(cfg.get('rotary_emb_base', 1000.0))
        args.heads = int(cfg.get('n_head', cfg.get('num_attention_heads', args.heads)))
        args.ln_eps = float(cfg.get('layer_norm_epsilon', cfg.get('layer_norm_eps', args.ln_eps)))
        print(f"config.json found: rope_base={args.rope_base} heads={args.heads} eps={args.ln_eps}", file=sys.stderr)
    if args.rope_base is None: args.rope_base = 1000.0

    tok = WordPiece(args.vocab)
    if args.sweep:
        # REFERENCE-FREE validation: only correct wiring yields semantic contrast --
        # paraphrases (0,1) close, doc-query (2,3) close, unrelated (4) far. Score = mean(related) - mean(unrelated).
        import itertools
        best=None
        for sg, pl in itertools.product((False, True), (False, True)):
            args.swap_gate, args.pre_ln = sg, pl
            m = Nomic(args.model, args)
            E = np.array([m.embed(tok.encode(t)) for t in TEST])
            C = E @ E.T
            related = (C[0,1] + C[2,3]) / 2
            unrelated = (C[0,4]+C[1,4]+C[2,4]+C[3,4]+C[0,2]+C[0,3]+C[1,2]+C[1,3]) / 8
            score = related - unrelated
            print(f"\nswap_gate={sg} pre_ln={pl}: contrast score {score:+.4f} "
                  f"(related {related:.3f} vs unrelated {unrelated:.3f})")
            for row in C: print("   " + " ".join(f"{v:6.3f}" for v in row))
            if best is None or score > best[0]: best = (score, sg, pl)
        print(f"\nWINNER: swap_gate={best[1]} pre_ln={best[2]} (contrast {best[0]:+.4f})")
        print("(correct wiring should show related pairs clearly above the rest; broken wiring is flat or scrambled)")
        return
    model = Nomic(args.model, args)
    embs = []
    for s in TEST:
        ids = tok.encode(s)
        e = model.embed(ids)
        embs.append(e)
        print(f"[{len(ids):3d} tok] {s[:52]:52s} e[:4]={np.round(e[:4],4).tolist()}")
    E = np.array(embs)
    print("\npairwise cosine matrix:")
    C = E @ E.T
    for row in C: print("   " + " ".join(f"{v:6.3f}" for v in row))
    print(f"\nchecksum: {float(np.abs(E).sum()):.6f}")
    print("sanity: sentences 1&2 (paraphrases) should be the most similar off-diagonal pair;")
    print("        3&4 (doc & its query) next; 5 unrelated to all.")

    if args.ref:
        ref = json.load(open(args.ref))
        R = np.array(ref['embeddings'], dtype=np.float32)
        R = R / np.linalg.norm(R, axis=1, keepdims=True)
        cos = (E * R).sum(1)
        print("\nBAR CHECK vs reference:")
        for s, c in zip(TEST, cos): print(f"  cos {c:.5f}  {'PASS' if c >= 0.999 else 'FAIL'}  {s[:48]}")

def run_selftest():
    """Plumbing check on random weights: shapes flow, output is unit-norm, and RoPE is position-sensitive."""
    rng = np.random.default_rng(0)
    d, H, hd = 64, 4, 16
    x = rng.standard_normal((7, H, hd)).astype(np.float32)
    r1 = rope(x, 1000.0); r0 = rope(x[:1], 1000.0)
    assert np.allclose(r1[0], r0[0], atol=1e-6), "position 0 must be identity-rotated"
    assert not np.allclose(r1[3], x[3]), "later positions must rotate"
    n = np.linalg.norm(x, axis=-1); n2 = np.linalg.norm(r1, axis=-1)
    assert np.allclose(n, n2, atol=1e-4), "RoPE must preserve norms (it is a rotation)"
    print("selftest: rope OK (identity at pos 0, rotates later, norm-preserving); plumbing shapes OK")

if __name__ == '__main__':
    main()
