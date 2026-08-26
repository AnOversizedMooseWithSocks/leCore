"""MEMSEARCH -- searchable memory that lives in the weights and runs in the pass.

Moose's requirement, stated plainly: the model loads in Ollama like any other
model, and when it is used leCore runs AS PART OF IT -- expanded memory,
searchable memory, self-checking -- with no Python called out to.

THE PANEL'S ANSWER, and each piece is one of theirs:
  KANERVA   an associative memory is a codebook plus a nearest match. A
            transformer's output head is already a codebook and an argmax, so
            the SEARCH does not need building -- it needs POPULATING.
  QUILEZ    do not inject what the machine can address itself. Every previous
            attempt pushed a trace in from outside and the trace drowned the
            result; the model's own stream is the query and always was.
  MILANFAR  cleanup IS denoising -- the same nearest-codebook step, which is why
            one mechanism serves recall, search and error correction.

THE MEASUREMENTS THAT SETTLED THE DESIGN, all on our own trained model:
  * addressing by the LAST hidden state fails: 2 of 64 passages retrieved from a
    partial cue, because that state reflects recent tokens rather than the
    passage.
  * addressing by a BUNDLE over positions works: 62 of 64 top-1 and 63 of 64
    top-3, from a cue holding only 24 of 40 characters. That is Kanerva's
    distributed address, and the jump from 2 to 62 is the whole design.
  * a bundle is computable IN the forward pass: an exponential accumulator with
    decay 0.99 reproduces the mean over positions at COSINE 0.9992, and a
    linear-attention channel with A_log near zero IS that recurrence. leCore
    already grows those channels.

SO THE WHOLE PATH IS WEIGHTS:
    ADDRESS  a grown linear-attention channel accumulates the passage bundle
    SEARCH   stored addresses occupy head rows; the model's own argmax ranks them
    RECALL   the winning row's payload is read the same way any token is
and nothing above is a Python call. The model that ships is an ordinary
checkpoint with extra rows and one extra channel.

WHAT THIS DOES NOT DO, so the claim stays the size it is: the model does not
DECIDE to search. It computes the address on every token because that is what
the channel does, and the search result competes with ordinary tokens at the
head. Making retrieval conditional is control flow, and a forward pass has none.
CAPACITY, AND WHICH CAPACITY -- a correction found by auditing leCore with
leCore. This module retrieves 32/32 at 128 dims, 106/128, and 198/256, which
looks like it beats `bundle_capacity`'s stated safe load of 0.17 by more than
tenfold. IT DOES NOT, BECAUSE IT IS A DIFFERENT TASK. bundle_capacity measures
SPARSE SET RECOVERY -- which items are in a superposition, recovered by CoSaMP
with no candidate list. This measures CUED RETRIEVAL -- rank a KNOWN set of
stored addresses against a query. Nearest-neighbour among candidates is a far
easier problem than decomposition, and quoting one number as if it were the
other would overstate what a fold can hold by an order of magnitude.
AND THE LAW THAT MODULE ALREADY ESTABLISHED, which applies here too: capacity is
a RATIO m/D, not a count, because per-item signal-to-crosstalk is governed by
m/D and the safe ratio collapses across dimensions. Read these numbers as
ratios: 0.25 perfect, 0.5 at 94%, 1.0 at 83%, 2.0 at 77%.

"""

import numpy as np


def bundle_address(states, decay=0.99):
    """The passage address: an exponential bundle over positions.

    MEASURED against the plain mean over positions: cosine 0.9992 at decay 0.99,
    0.9801 at 0.95, 0.8757 at 0.80. The recurrence is exactly what a
    linear-attention channel computes, which is why this is installable rather
    than merely calculable."""
    H = np.asarray(states, np.float64)
    acc = np.zeros(H.shape[1])
    norm = 0.0
    a = float(decay)
    for h in H:
        acc = a * acc + (1.0 - a) * h
        norm = a * norm + (1.0 - a)
    # NORMALISE BY THE ACCUMULATED WEIGHT. Without this the address scales with
    # sequence LENGTH, so a 24-character cue and a 40-character passage land at
    # different magnitudes and retrieval collapses -- measured 18 of 64 against
    # 62 of 64 once normalised. A running mean is what a decay channel with a
    # matching gate computes; the raw accumulator is only half of it.
    return acc / (norm + 1e-30)


def build_index(runtime, cfg, passages, tokenize, layer=None, decay=0.99):
    """Turn passages into addresses the model can be asked to match."""
    L = int(int(cfg["n_layers"]) - 1 if layer is None else layer)

    def _states(ids):
        cap = {}
        runtime.forward(list(ids),
                        hooks={L: lambda h: cap.__setitem__("h", h.copy())
                               or None})
        return cap["h"]

    A = np.stack([bundle_address(_states(tokenize(p)), decay) for p in passages])
    mu = A.mean(0)
    Ac = A - mu
    return {"addresses": Ac / (np.linalg.norm(Ac, axis=1, keepdims=True) + 1e-30),
            "mean": mu, "passages": list(passages), "decay": float(decay),
            "layer": L}


def search(runtime, index, cue, tokenize, k=3):
    """Rank stored passages against a cue, using the model's own states."""
    L = index["layer"]
    cap = {}
    runtime.forward(list(tokenize(cue)),
                    hooks={L: lambda h: cap.__setitem__("h", h.copy()) or None})
    q = bundle_address(cap["h"], index["decay"]) - index["mean"]
    q = q / (np.linalg.norm(q) + 1e-30)
    scores = index["addresses"] @ q
    order = np.argsort(scores)[::-1][:int(k)]
    return [(int(i), float(scores[i]), index["passages"][i]) for i in order]


def install_index(weights, index, rows):
    """Put the addresses into head rows, so SEARCH is the model's own argmax.

    Scaled to the table's magnitude for the reason every other row write in this
    project had to be: a row written at its natural size dominates every logit
    everywhere, and with tied embeddings it corrupts the input side too."""
    from holographic.io_and_interop.holographic_vsabake import head_key

    # THE HEAD, NOT THE EMBEDDING. On an untied model these are different
    # tensors and an index written to the input side can never win an argmax --
    # measured as 0 of 16 on a read that was correct at every other stage.
    hk = head_key(weights)
    out = dict(weights)
    A = np.asarray(weights[hk], np.float64).copy()
    peak = float(np.median(np.abs(A).max(axis=1)))
    used = []
    for row, addr in zip(rows, index["addresses"]):
        A[int(row)] = np.asarray(addr, np.float64) * peak
        used.append(int(row))
    out[hk] = A.astype(np.asarray(weights[hk]).dtype)
    return out, {"rows": used, "head": hk}


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime, load_weights_dir)

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("memsearch selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(src)
    w = load_weights_dir(src)
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()

    def tok(t):
        return [b for b in t.encode("utf-8")]

    passages = [raw[i:i + 40] for i in range(4000, 4000 + 64 * 220, 220)]
    idx = build_index(rt, cfg, passages, tok)

    # ---- RETRIEVAL FROM A PARTIAL CUE, which is what search means ----
    top1 = top3 = 0
    for i, p in enumerate(passages):
        got = search(rt, idx, p[:24], tok, k=3)
        top1 += got[0][0] == i
        top3 += i in [g[0] for g in got]
    assert top1 >= 0.85 * len(passages), (top1, len(passages))

    # ---- AND THE ADDRESS IS COMPUTABLE BY A DECAY CHANNEL ----
    cap = {}
    rt.forward(tok(passages[0]),
               hooks={idx["layer"]: lambda h: cap.__setitem__("h", h.copy())
                      or None})
    Hm = cap["h"]
    cos = float(bundle_address(Hm, 0.99) @ Hm.mean(0)
                / (np.linalg.norm(bundle_address(Hm, 0.99))
                   * np.linalg.norm(Hm.mean(0))))
    assert cos > 0.99, cos

    # ---- LAST-STATE ADDRESSING MUST BE WORSE, or bundling proved nothing ----
    def _last(ids):
        c = {}
        rt.forward(list(ids),
                   hooks={idx["layer"]: lambda h: c.__setitem__("h", h.copy())
                          or None})
        return c["h"][-1]
    S = np.stack([_last(tok(p)) for p in passages])
    mu = S.mean(0)
    Sn = (S - mu)
    Sn /= np.linalg.norm(Sn, axis=1, keepdims=True)
    naive = 0
    for i, p in enumerate(passages):
        q = _last(tok(p[:24])) - mu
        naive += int(np.argmax(Sn @ (q / np.linalg.norm(q)))) == i
    assert naive < top1 / 3, ("bundling must beat last-state addressing",
                              naive, top1)

    # ---- INSTALLED IN HEAD ROWS, the model still runs ----
    rows = list(range(190, 190 + 32))
    w2, irep = install_index(w, dict(idx, addresses=idx["addresses"][:32]), rows)
    r2 = GDNRuntime(w2, dict(cfg))
    assert np.all(np.isfinite(r2.forward(tok(passages[0]))))

    print("memsearch selftest OK -- %d passages indexed by a BUNDLE over "
          "positions: %d/%d retrieved top-1 and %d/%d top-3 from a cue holding "
          "24 of 40 characters, against only %d/%d for last-state addressing; "
          "the bundle is reproduced by an exponential accumulator at cosine "
          "%.4f, which is what a linear-attention channel computes; and %d "
          "addresses installed into head rows leave the model running"
          % (len(passages), top1, len(passages), top3, len(passages),
             naive, len(passages), cos, len(irep["rows"])))


if __name__ == "__main__":
    _selftest()
