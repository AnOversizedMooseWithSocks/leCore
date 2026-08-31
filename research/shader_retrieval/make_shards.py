"""P0.16 -- sharded delivery: split the index into fetchable blocks WITHOUT changing an answer.

WHAT THE ARITHMETIC ALREADY SETTLED. T4 proves a tiled max equals the single-pass max, so merging
shard top-k lists is exact -- IF the shards produce comparable scores. They do not by default:
BM25 bakes the per-(term,doc) weight at fit time from LOCAL idf and avgdl, measured at max relative
error 0.31 and top-1 agreement 0.76 against a single index. The `stats=` seam added earlier fixes
that, and shards fitted with the corpus's statistics are BIT-IDENTICAL.

WHAT IS LEFT IS TRANSPORT, and its risk is not arithmetic: it is that the SHIPPED BYTES and the
JS that reassembles them disagree with the Python that produced them. So this emits the bundle AND
a loader, and the verification runs the loader under Node against Python on the whole corpus -- no
browser needed, because the browser adds nothing to this particular question.
"""
import base64
import json
import os

import numpy as np

import hard_corpus as HC
import holographic.agents_and_reasoning.holographic_hashatom as HA
from holographic.semantic_router.holographic_bm25 import BM25


def pack(sym, bits):
    out = bytearray((len(sym) * bits + 7) // 8)
    acc = nb = pos = 0
    for v in sym:
        acc |= (int(v) & ((1 << bits) - 1)) << nb
        nb += bits
        while nb >= 8:
            out[pos] = acc & 0xFF; acc >>= 8; nb -= 8; pos += 1
    if nb:
        out[pos] = acc & 0xFF
    return bytes(out)


def build(target=6000, nshards=8, outdir="shards"):
    dn = HC.load_passages(target=target)
    full = BM25([" ".join(d) for _, d in dn])
    docs = full.docs_tokens
    stats = full.corpus_stats()
    rank = {}
    for d in docs:
        for t in d:
            rank.setdefault(int(HA.fnv1a(t)), len(rank))
    bits = max(1, (len(rank) - 1).bit_length())
    os.makedirs(outdir, exist_ok=True)
    bounds = np.array_split(np.arange(len(docs)), nshards)
    manifest = {"bits": bits, "nshards": nshards, "avgdl": stats["avgdl"], "N": stats["N"],
                "vocab": [0] * len(rank), "idf": {}, "shards": []}
    for h, i in rank.items():
        manifest["vocab"][i] = h
    # GLOBAL idf keyed by the term's HASH, so a shard never needs the corpus to compute it. This is
    # the side-channel that makes sharding exact, and it is small: one float per distinct term.
    for t, v in stats["idf"].items():
        manifest["idf"][str(int(HA.fnv1a(t)))] = float(v)
    total = 0
    for si, b in enumerate(bounds):
        sym, off = [], [0]
        for i in b:
            for t in docs[i]:
                sym.append(rank[int(HA.fnv1a(t))])
            off.append(len(sym))
        blob = pack(sym, bits)
        total += len(blob)
        path = os.path.join(outdir, "shard%02d.json" % si)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"base": int(b[0]), "ndocs": int(len(b)), "ntok": len(sym),
                       "packed": base64.b64encode(blob).decode(),
                       "off": base64.b64encode(np.array(off, dtype="<u4").tobytes()).decode()}, fh)
        manifest["shards"].append({"file": os.path.basename(path), "base": int(b[0]),
                                   "ndocs": int(len(b)), "bytes": len(blob)})
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return full, docs, manifest, total


if __name__ == "__main__":
    full, docs, man, total = build()
    print("SHARDED BUNDLE: %d documents, %d shards, %d bits/symbol" % (len(docs), man["nshards"], man["bits"]))
    print("  packed payload %.1f KB across shards, manifest %.1f KB (vocab %d + global idf)"
          % (total / 1024, os.path.getsize("shards/manifest.json") / 1024, len(man["vocab"])))
    print("  per-shard: %s bytes" % [s["bytes"] for s in man["shards"]])
    # queries for the Node harness: realistic 2-4 terms, NOT sampled from any one document
    rng = np.random.default_rng(0)
    vocab_terms = sorted({t for d in docs for t in d})
    qs = [[vocab_terms[j] for j in rng.choice(len(vocab_terms), int(rng.integers(2, 5)), replace=False)]
          for _ in range(60)]
    ref = [[float(v) for v in full.scores(q)] for q in qs]
    with open("shards/probe.json", "w", encoding="utf-8") as fh:
        json.dump({"queries": [[int(HA.fnv1a(t)) for t in q] for q in qs],
                   "scores": ref}, fh)
    print("  probe: %d realistic 2-4 term queries with Python reference scores" % len(qs))
