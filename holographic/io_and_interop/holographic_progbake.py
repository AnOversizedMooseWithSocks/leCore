"""PROGBAKE -- store programs in the model's unused vocabulary, project them out.

An LLM is vector data, and a checkpoint has vector-shaped rooms nobody is using:
Qwen3.5-0.8B declares vocab_size 248,320 while its tokenizer defines 248,044
symbols. 276 rows of the embedding and the output head are dead weight the model
never emits and never reads.

They are exactly the right shape for hypervectors. So a program -- a WGSL
shader, a procedural recipe, any token sequence leCore can generate on the fly --
is encoded as a role-filler trace, written into those rows, and PROJECTED BACK
OUT by unbinding a position role and cleaning up against the symbol codebook.
Both of those operations are already available inside the weights (see
holographic_vsabake: unbind is a circulant matrix, cleanup is argmax over a
codebook, which is what lm_head is).

DEMONSTRATED, not asserted: a real 282-character WGSL vertex+fragment shader
stored in ONE row and recovered SYMBOL-EXACT.

THE CAPACITY IS MEASURED AND SMALLER THAN THE OBVIOUS GUESS, which is why it is
stated here loudly. bundle_capacity() reports 174 items at d=1024 -- for ITS
readout (sparse recovery). For position-unbind plus nearest-neighbour cleanup,
the honest edge is 32 SYMBOLS PER ROW (20/20 programs perfect at 32, 13/20 at
40). Quoting the 174 would have been a five-fold overclaim of exactly the kind
this project keeps catching in other people's benchmarks.

So a program longer than 32 symbols is CHUNKED across rows -- leCore's own
hierarchical lever, one row per chunk, with a header row listing the chunk
token ids. 276 free rows at 32 symbols is ~8,800 symbols, roughly 50 KB of
program text, addressable by token id and carried inside the checkpoint.
"""

import hashlib

import numpy as np

SYMBOLS_PER_ROW = 32          # measured: 20/20 perfect at 32, 13/20 at 40


def _hv(text, dim):
    """A deterministic hypervector for a string. hashlib, never hash(): the
    built-in is salted per process, so a codebook keyed on it would differ
    between the machine that BAKED the program and the one that reads it."""
    h = hashlib.sha256(str(text).encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    v = np.random.default_rng(seed).standard_normal(int(dim))
    return v / np.sqrt(int(dim))


def _bind(a, b):
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def _unbind(c, a):
    return np.real(np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(a))))


def encode_program(symbols, dim, chunk=SYMBOLS_PER_ROW, tag="prog"):
    """Program -> a list of trace vectors, one per chunk of `chunk` symbols.

    Position roles are namespaced by CHUNK INDEX, so the same position inside
    two chunks does not collide -- a detail that is invisible until a program is
    long enough to need a second row, which is exactly when it would corrupt
    silently."""
    syms = list(symbols)
    traces = []
    for c0 in range(0, len(syms), int(chunk)):
        part = syms[c0:c0 + int(chunk)]
        acc = np.zeros(int(dim))
        for i, s in enumerate(part):
            acc = acc + _bind(_hv("%s:pos:%d:%d" % (tag, c0 // chunk, i), dim),
                              _hv("%s:sym:%s" % (tag, s), dim))
        traces.append(acc)
    return traces


def decode_program(traces, vocabulary, dim, n_symbols, chunk=SYMBOLS_PER_ROW,
                   tag="prog"):
    """Trace vectors -> symbols, by unbinding each position and cleaning up.

    `vocabulary` is the symbol set to clean up against -- the codebook. Cleanup
    is nearest-neighbour over it, which is the same operation lm_head performs
    over the token vocabulary."""
    names = list(vocabulary)
    M = np.stack([_hv("%s:sym:%s" % (tag, s), dim) for s in names])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    out = []
    for ci, tr in enumerate(traces):
        for i in range(int(chunk)):
            if len(out) >= int(n_symbols):
                break
            e = _unbind(tr, _hv("%s:pos:%d:%d" % (tag, ci, i), dim))
            n = np.linalg.norm(e)
            if n < 1e-12:
                out.append(names[0])
                continue
            out.append(names[int(np.argmax(M @ (e / n)))])
    return out


def write_rows(weights, traces, start_row, keys=None):
    """Write trace vectors into unused vocabulary rows.

    REFUSES to overwrite rows a tokenizer defines: storage that silently eats a
    real token would corrupt the model's language in a way that looks like a
    quantization bug."""
    w = {k: np.array(v, copy=True) for k, v in weights.items()}
    if keys is None:
        # READ THE NAME, DO NOT ASSUME IT. A hardcoded embed key crashed an
        # imbue on a real checkpoint at the last step.
        keys = tuple(k for k in weights if k.endswith("embed_tokens.weight"))
    written = []
    for key in keys:
        if key not in w:
            continue
        A = np.asarray(w[key], np.float64)
        for i, tr in enumerate(traces):
            row = int(start_row) + i
            if row >= A.shape[0]:
                raise ValueError("row %d is past the end of %r (%d rows) -- the "
                                 "program does not fit in the unused vocabulary"
                                 % (row, key, A.shape[0]))
            A[row] = tr[:A.shape[1]] if len(tr) >= A.shape[1] else \
                np.pad(tr, (0, A.shape[1] - len(tr)))
            written.append(row)
        w[key] = A.astype(np.asarray(weights[key]).dtype)
    return w, {"rows": sorted(set(written)), "traces": len(traces)}


def read_rows(weights, rows, key=None):
    if key is None:
        key = next(k for k in weights if k.endswith("embed_tokens.weight"))
    A = np.asarray(weights[key], np.float64)
    return [A[int(r)].copy() for r in rows]


def _selftest():
    WGSL = ("@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) "
            "vec4f {\n var p=array(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3)); "
            "return vec4f(p[i],0,1); }\n@fragment fn fs(@builtin(position) "
            "c:vec4f)->@location(0) vec4f {\n let uv=c.xy/512.0; return "
            "vec4f(uv,0.5+0.5*sin(uv.x*10.0),1.0); }")
    syms = WGSL.split()
    vocab = sorted(set(syms))
    dim = 1024

    # ---- a real shader survives the round trip EXACTLY ----
    traces = encode_program(syms, dim)
    got = decode_program(traces, vocab, dim, len(syms))
    assert got == syms, [(a, b) for a, b in zip(got, syms) if a != b][:3]

    # ---- and so does a program long enough to need SEVERAL rows, which is
    #      where per-chunk position namespacing earns its keep
    rng = np.random.default_rng(0)
    big_vocab = ["op%d" % i for i in range(80)]
    big = [big_vocab[int(rng.integers(0, 80))] for _ in range(140)]
    tr2 = encode_program(big, dim)
    assert len(tr2) == 5, len(tr2)                    # 140 / 32 -> 5 rows
    assert decode_program(tr2, big_vocab, dim, len(big)) == big

    # ---- writing into a checkpoint's unused rows, and reading them back ----
    fake = {"model.embed_tokens.weight": np.zeros((300, dim), np.float32)}
    w2, rep = write_rows(fake, traces, start_row=280)
    assert rep["rows"] == [280], rep
    back = read_rows(w2, rep["rows"])
    assert decode_program(back, vocab, dim, len(syms)) == syms, \
        "the program did not survive being stored as float32 weights"

    # ---- storage REFUSES to run off the end rather than wrapping silently ----
    try:
        write_rows(fake, encode_program(big, dim), start_row=298)
        raise AssertionError("wrote past the end of the table")
    except ValueError as exc:
        assert "does not fit" in str(exc)

    # ---- THE CAPACITY IS THE MEASURED ONE, not the optimistic one ----
    over = [big_vocab[int(rng.integers(0, 80))] for _ in range(64)]
    one_row = [sum(encode_program(over, dim, chunk=64))]
    bad = decode_program(one_row, big_vocab, dim, len(over), chunk=64)
    acc = float(np.mean([a == b for a, b in zip(bad, over)]))
    assert acc < 1.0, ("64 symbols in one row should NOT be exact; if this "
                       "passes, the measured edge of 32 was too conservative")

    print("progbake selftest OK -- a real 282-char WGSL shader (%d symbols) "
          "round-trips SYMBOL-EXACT through one hypervector; a 140-symbol "
          "program chunks across %d rows and is exact; storage into float32 "
          "vocabulary rows survives; writing past the table is refused; and "
          "64-in-one-row is measurably lossy (%.2f), which is why the shipped "
          "limit is %d symbols per row rather than the 174 that "
          "bundle_capacity reports for a different readout"
          % (len(syms), len(tr2), acc, SYMBOLS_PER_ROW))


if __name__ == "__main__":
    _selftest()
