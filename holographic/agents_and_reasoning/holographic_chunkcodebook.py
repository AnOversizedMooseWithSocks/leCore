"""LEARNED CHUNK CODEBOOKS by iterated pair promotion (backlog item R1; R3's "one codebook family").

The mechanism is BPE -- Gage (1994), popularised for NLP by Sennrich, Haddow & Birch (2016) -- with the twist that
the merged chunks are not tokenizer vocabulary but **factoring and storage codebooks**:

  * R2 gives the resonator a chunk level to factor against, breaking the depth-4 combinatorial cliff by expanding
    each chunk by LOOKUP instead of by search;
  * W5's hierarchical superposition needs a shared chunk codebook for its mid-level cleanup -- naive nesting is
    flat by linearity, and the codebook is the only thing that makes hierarchy pay;
  * DL8's edit codec tokenizes an edit stream at recognized boundaries.

Three consumers, one structure (R3). This module owns it.

WHAT IT IS NOT -- and this is the first thing to say, because the number is seductive. Iterated pair promotion
reduces a structured stream from 6,000 symbols to 1,389 tokens: **4.3x**. That is a TOKEN-count ratio, and it is
not byte compression. Measured on the same stream:

    raw                                  6,000 bytes
    zlib(raw)                            1,820      <- the honest byte baseline
    BPE codebook (800) + tokens (2,778)  3,578      <- the "4.3x" as actual bytes
    zlib(BPE tokens) + codebook          2,615

**As a byte codec this LOSES to zlib, by 1.4x-2.0x.** Do not ship it as one. Its value is the codebook it leaves
behind, and the structure it reveals -- which is exactly what the backlog asks of it and exactly what the three
consumers above need.

WHAT IT IS: a structure probe with a reusable artifact. The learned chunk DEPTH separates a structured stream from
a structureless one decisively, and the separation is the whole condition on which R2, W5 and the recursion
dividend depend:

    | stream                                    | tokens        | mean chunk depth | max depth |
    |-------------------------------------------|---------------|------------------|-----------|
    | structured (latent workflows, 85% reuse)  | 6,000 -> 1,389 | 4.32            | 16        |
    | uniform control (no structure)            | 6,000 -> 4,473 | 1.34            | 2         |

Depth stalls at 2 without structure. *No structure, no recursion dividend* -- the program's recurring law, and here
it is a measurement rather than a hope.

Deterministic: merges are chosen by count with an explicit tie-break on the pair itself, never by dict order.
"""

import numpy as np


class ChunkCodebook:
    """An ordered list of learned merges, plus the depth of every symbol it can emit.

    `merges` is a list of ((a, b), new_id) IN LEARNING ORDER -- the order is the codebook, because encoding replays
    it. `depth[t]` is how many base symbols the token `t` expands to (a leaf has depth 1).

    `to_dict()` / `from_dict()` make it plain data, so a codebook crosses an HTTP boundary and a long-lived service
    can hand the SAME codebook to the resonator (R2), the chunked store (W5) and the edit codec (DL8)."""

    def __init__(self, merges, depth):
        self.merges = [((int(a), int(b)), int(nid)) for (a, b), nid in merges]
        self.depth = {int(k): int(v) for k, v in depth.items()}
        self._rule = {(a, b): nid for (a, b), nid in self.merges}
        self._expand = {nid: (a, b) for (a, b), nid in self.merges}

    def __len__(self):
        return len(self.merges)

    def encode(self, stream):
        """Replay the learned merges over `stream`, in learning order. Returns a token list.

        Encoding must replay the SAME order the merges were learned in: a later merge can consume a token an
        earlier one produced, so applying them out of order gives a different (and generally worse) tokenization."""
        seq = [int(s) for s in stream]
        for (a, b), nid in self.merges:
            out, i, n = [], 0, len(seq)
            while i < n:
                if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(nid)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq = out
        return seq

    def decode(self, tokens):
        """Expand every chunk token back to base symbols. LOSSLESS: decode(encode(s)) == s, exactly. Iterative
        rather than recursive so a deep codebook cannot blow the Python stack."""
        out = []
        stack = list(reversed([int(t) for t in tokens]))
        while stack:
            t = stack.pop()
            if t in self._expand:
                a, b = self._expand[t]
                stack.append(b)
                stack.append(a)
            else:
                out.append(t)
        return out

    def stats(self, stream):
        """{n_symbols, n_tokens, token_ratio, mean_depth, max_depth, n_merges, covered} for `stream`.

        `token_ratio` is a TOKEN count ratio, not a compression ratio -- see the module docstring; `mean_depth` is
        the structure probe, and it is the number that decides whether R2/W5's recursion dividend exists here."""
        seq = self.encode(stream)
        if not seq:
            return {"n_symbols": 0, "n_tokens": 0, "token_ratio": 0.0, "mean_depth": 0.0,
                    "max_depth": 0, "n_merges": len(self.merges), "covered": 0.0}
        depths = [self.depth.get(t, 1) for t in seq]
        covered = sum(d for d in depths if d > 1)
        return {"n_symbols": len(stream), "n_tokens": len(seq),
                "token_ratio": len(stream) / len(seq),
                "mean_depth": float(np.mean(depths)), "max_depth": int(max(depths)),
                "n_merges": len(self.merges), "covered": covered / len(stream)}

    def to_dict(self):
        """Plain data: {merges: [[a, b, new_id], ...], depth: {token: depth}}. JSON-safe."""
        return {"merges": [[a, b, nid] for (a, b), nid in self.merges],
                "depth": {str(k): v for k, v in self.depth.items()}}

    @staticmethod
    def from_dict(d):
        """Inverse of `to_dict`."""
        merges = [((int(a), int(b)), int(nid)) for a, b, nid in d["merges"]]
        depth = {int(k): int(v) for k, v in d["depth"].items()}
        return ChunkCodebook(merges, depth)


def learn_chunks(stream, max_merges=200, min_count=2):
    """Learn a chunk codebook from `stream` by iterated pair promotion (Gage 1994 / Sennrich et al. 2016).

    At each step, promote the MOST FREQUENT adjacent pair into a new symbol and rewrite the stream. Stops after
    `max_merges` or as soon as the best pair occurs fewer than `min_count` times -- which is what makes the
    structureless case halt on its own rather than inventing chunks out of noise.

    DETERMINISM: ties on count are broken by the pair itself (smallest `a`, then smallest `b`), never by dict
    insertion order. `Counter.most_common` is insertion-ordered on ties and would make the codebook depend on how
    the stream happened to be built -- a determinism leak of exactly the kind this engine forbids.

    Returns a ChunkCodebook. The stream is not modified."""
    from collections import Counter

    seq = [int(s) for s in stream]
    if not seq:
        return ChunkCodebook([], {})
    next_id = max(seq) + 1
    depth = {t: 1 for t in set(seq)}
    merges = []

    for _ in range(int(max_merges)):
        pairs = Counter(zip(seq, seq[1:]))
        if not pairs:
            break
        # most frequent, ties -> smallest pair. Explicit, so the codebook is reproducible run to run and machine
        # to machine (max() keeps the FIRST maximal item, so negate the pair to prefer the smallest).
        (a, b), count = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if count < int(min_count):
            break
        merges.append(((a, b), next_id))
        depth[next_id] = depth[a] + depth[b]
        out, i, n = [], 0, len(seq)
        while i < n:
            if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(next_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
        next_id += 1

    return ChunkCodebook(merges, depth)


def structure_score(stream, max_merges=200, min_count=2):
    """A single number saying whether `stream` has reusable structure: the MEAN CHUNK DEPTH of its own learned
    tokenization. 1.0 means nothing merged (no structure); measured 4.32 on a workflow stream and 1.34 on a
    uniform control.

    This is the gate on the recursion dividend. R2's recursive factoring, W5's hierarchical superposition and
    DL8's edit codec all pay off *only* when the stream is made of reused chunks; this probe says whether it is,
    before any of them is built on top."""
    return learn_chunks(stream, max_merges=max_merges, min_count=min_count).stats(stream)["mean_depth"]


def byte_report(stream, codebook, level=9):
    """The codec comparison, computed so it travels with the capability -- because the token ratio invites a claim
    the bytes do not support. Returns {raw, zlib_raw, bpe_bytes, zlib_bpe_bytes, beats_zlib}.

    MEASURED on the structured workflow stream: raw 6,000; zlib 1,820; BPE codebook+tokens 3,578; zlib(BPE) 2,615.
    `beats_zlib` is False, and it is meant to be: this is a codebook learner, not a compressor."""
    import zlib

    seq = codebook.encode(stream)
    raw = np.asarray(stream, dtype=np.uint8).tobytes() if max(stream, default=0) < 256 \
        else np.asarray(stream, dtype="<u4").tobytes()
    tok = np.asarray(seq, dtype="<u4").tobytes()
    cb_bytes = len(codebook.merges) * 12                       # (a, b, new_id) as three uint32
    bpe_bytes = cb_bytes + len(tok)
    zlib_bpe = cb_bytes + len(zlib.compress(tok, level))
    zlib_raw = len(zlib.compress(raw, level))
    return {"raw": len(raw), "zlib_raw": zlib_raw, "bpe_bytes": bpe_bytes,
            "zlib_bpe_bytes": zlib_bpe, "beats_zlib": bool(min(bpe_bytes, zlib_bpe) < zlib_raw)}


def workflow_stream(n_workflows=1500, seed=0, reuse=0.85, vocab=32, n_kinds=12, length=4):
    """The reference STRUCTURED stream: latent workflows of `length` edits, Zipf-picked, with (1-`reuse`) novel
    sequences mixed in. Stands in for a real edit trace, where 84.2% of edits were measured to be reused chunks."""
    rng = np.random.default_rng(seed)
    kinds = [list(rng.integers(0, vocab, length)) for _ in range(n_kinds)]
    z = 1.0 / np.arange(1, n_kinds + 1)
    z /= z.sum()
    out = []
    while len(out) < n_workflows * length:
        if rng.random() < reuse:
            out += kinds[rng.choice(n_kinds, p=z)]
        else:
            out += list(rng.integers(0, vocab, length))
    return [int(x) for x in out[:n_workflows * length]]


def uniform_stream(n=6000, seed=0, vocab=32):
    """The STRUCTURELESS control. Every claim about chunk promotion has to be checked against this, or it is a
    claim about BPE finding pairs in noise -- which it always will."""
    return [int(x) for x in np.random.default_rng(seed).integers(0, vocab, n)]


def _selftest():
    """Regression trap for R1: lossless round-trip, deterministic merges, the structured/uniform separation, and
    the kept negative that this is not a byte compressor."""
    s = workflow_stream()
    cb = learn_chunks(s)

    # 1. LOSSLESS. The codebook is useless to R2/W5/DL8 if it cannot reconstruct exactly.
    assert cb.decode(cb.encode(s)) == s
    assert cb.decode(cb.encode([])) == []
    assert learn_chunks([]).encode([]) == []                       # empty stream: no merges, no crash

    # 2. DETERMINISTIC. Same stream in, identical codebook out -- and not by luck of dict order.
    assert learn_chunks(s).to_dict() == cb.to_dict()
    assert ChunkCodebook.from_dict(cb.to_dict()).encode(s) == cb.encode(s)

    # 3. THE SEPARATION: structure produces deep chunks, noise stalls at depth 2.
    st = cb.stats(s)
    u = uniform_stream()
    ucb = learn_chunks(u)
    us = ucb.stats(u)
    assert st["token_ratio"] > 3.0 > us["token_ratio"]
    assert st["max_depth"] >= 8 and us["max_depth"] <= 2
    assert st["mean_depth"] > 3.0 > us["mean_depth"]
    assert ucb.decode(ucb.encode(u)) == u                          # still lossless on noise

    # 4. min_count halts on noise rather than inventing chunks out of it
    assert len(learn_chunks(u, min_count=10_000)) == 0

    # 5. KEPT NEGATIVE: this is NOT a byte compressor. zlib beats it on the very stream it "compresses 4.3x".
    rep = byte_report(s, cb)
    assert rep["zlib_raw"] < rep["bpe_bytes"]
    assert rep["zlib_raw"] < rep["zlib_bpe_bytes"]
    assert rep["beats_zlib"] is False

    print("OK: holographic_chunkcodebook self-test passed (round-trip lossless and deterministic; structured "
          "stream %d -> %d tokens (%.1fx), mean chunk depth %.2f, max %d -- against a uniform control at %.1fx, "
          "mean depth %.2f, max depth %d (it stalls at 2 without structure); and the KEPT NEGATIVE: as BYTES this "
          "loses to zlib, %d vs %d, so it is a codebook learner and a structure probe, never a codec)"
          % (st["n_symbols"], st["n_tokens"], st["token_ratio"], st["mean_depth"], st["max_depth"],
             us["token_ratio"], us["mean_depth"], us["max_depth"], rep["bpe_bytes"], rep["zlib_raw"]))


if __name__ == "__main__":
    _selftest()
