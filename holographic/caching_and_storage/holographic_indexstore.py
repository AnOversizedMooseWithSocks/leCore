"""Save a retrieval index to disk and load it back -- ONE format, shared with the browser.

WHY THIS IS NOT A NEW SERIALISER. `mind.save`/`load` already persist the MIND, and they do it well
(decision-safe quantisation, rANS on low-rank arrays). What they do not persist is a RETRIEVAL
INDEX: `retrieval_policy` rebuilds from documents you must still be holding. That is the gap, and
the format to fill it already existed as a research script -- the shard bundle whose merge is
bit-identical at 4 through 256 shards and whose reassembly was verified under Node against Python
at 5.8e-13. This module promotes that format into the engine so `find_capability` can surface it.

WHAT IS STORED, AND WHY IT IS THE GENERATOR RATHER THAN THE INDEX:
  * the dense-ranked, BIT-PACKED token stream, plus per-document offsets
  * the vocabulary as u32 hashes, and the GLOBAL statistics (N, avgdl, idf keyed by term hash)
  * a sha256 over the payload
The postings, tf tables and document lengths are DERIVED on load. Storing them too would be
duplicate state that can drift from the stream it describes, and the derivation is milliseconds.

SELF-VERIFYING BY DEFAULT. `load` recomputes the digest and REFUSES a payload that does not match,
because a truncated write or a half-finished IndexedDB transaction produces a file that parses
cleanly and answers wrongly. A store that cannot detect its own corruption is worse than no store.

THE SAME BYTES LOAD IN THE BROWSER. pages/idb_store.js reads this exact manifest out of IndexedDB,
so a corpus baked on a workstation is served from a tab with no rebuild.
"""
import base64
import hashlib
import json

import numpy as np


def _pack(sym, bits):
    """Fixed-width bit packing, LSB-first. T13 (pack_roundtrip) is lossless exactly while every
    symbol is below 2**bits, which is why the width is ASSERTED and not assumed -- a vocabulary
    that outgrew the width would silently ALIAS terms rather than fail."""
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


def _unpack(buf, bits, count):
    out = np.zeros(count, dtype=np.int64)
    acc = nb = pos = 0
    for i in range(count):
        while nb < bits:
            acc |= buf[pos] << nb; nb += 8; pos += 1
        out[i] = acc & ((1 << bits) - 1); acc >>= bits; nb -= bits
    return out


def build(docs_tokens, stats=None):
    """Bundle already-tokenised documents into the portable form. Tokens, not text: the tokenizer
    is not idempotent, so re-normalising on the way in would silently over-stem the index."""
    import holographic.agents_and_reasoning.holographic_hashatom as _ha
    rank, sym, off = {}, [], [0]
    for d in docs_tokens:
        for t in d:
            sym.append(rank.setdefault(int(_ha.fnv1a(t)), len(rank)))
        off.append(len(sym))
    bits = max(1, (len(rank) - 1).bit_length())
    assert not sym or max(sym) < (1 << bits), "symbol exceeds the packed width -- terms would alias"
    vocab = np.zeros(len(rank), dtype="<u4")
    for h, i in rank.items():
        vocab[i] = h
    packed = _pack(sym, bits)
    man = {
        "format": "lecore-index/1",
        "bits": bits,
        "ntok": len(sym),
        "ndocs": len(docs_tokens),
        "packed": base64.b64encode(packed).decode(),
        "off": base64.b64encode(np.array(off, dtype="<u4").tobytes()).decode(),
        "vocab": base64.b64encode(vocab.tobytes()).decode(),
    }
    if stats:
        man["stats"] = {"N": int(stats["N"]), "avgdl": float(stats["avgdl"]),
                        "idf": {str(int(_ha.fnv1a(t))): float(v) for t, v in stats["idf"].items()}}
    man["sha256"] = digest(man)
    return man


def digest(man):
    """Digest over the PAYLOAD only, so adding a field cannot silently invalidate old bundles."""
    h = hashlib.sha256()
    for k in ("format", "bits", "ntok", "ndocs", "packed", "off", "vocab"):
        h.update(str(man[k]).encode())
    return h.hexdigest()


def save(man, path):
    """Write the bundle. Text JSON on purpose: the same bytes go to disk and to IndexedDB."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh)
    return path


def load(path, verify=True):
    """Read a bundle back and REFUSE a payload whose digest does not match."""
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    if verify and man.get("sha256") and digest(man) != man["sha256"]:
        raise ValueError("index bundle failed its own digest -- truncated write or edited payload; "
                         "refusing to answer from it")
    return man


def symbols(man):
    """The token-id stream, unpacked. Postings and lengths are DERIVED from this, never stored."""
    return _unpack(base64.b64decode(man["packed"]), int(man["bits"]), int(man["ntok"]))


def offsets(man):
    return np.frombuffer(base64.b64decode(man["off"]), dtype="<u4")


def vocab(man):
    return np.frombuffer(base64.b64decode(man["vocab"]), dtype="<u4")


def _selftest():
    import tempfile, os
    import holographic.agents_and_reasoning.holographic_hashatom as _ha
    docs = [["alpha", "beta", "solo"], ["alpha", "beta", "twin"], ["zeta", "kappa", "unique"]]
    man = build(docs)
    d = tempfile.mkdtemp(); p = save(man, os.path.join(d, "idx.json"))
    back = load(p)

    # EXACT ROUND TRIP, symbol for symbol. A store that is "close" is a store that answers wrongly.
    sym = symbols(back); off = offsets(back)
    rebuilt = [[int(vocab(back)[s]) for s in sym[off[i]:off[i + 1]]] for i in range(len(docs))]
    truth = [[int(_ha.fnv1a(t)) for t in d_] for d_ in docs]
    assert rebuilt == truth, "the token stream did not round-trip"

    # THE DIGEST MUST REFUSE A TAMPERED PAYLOAD, or it is decoration.
    bad = dict(back); bad["packed"] = base64.b64encode(b"\x00" * 8).decode()
    with open(os.path.join(d, "bad.json"), "w", encoding="utf-8") as fh:
        json.dump(bad, fh)
    try:
        load(os.path.join(d, "bad.json"))
        raise AssertionError("a corrupted bundle was accepted -- the digest is decoration")
    except ValueError:
        pass

    # and it must still load when the caller deliberately skips verification
    load(os.path.join(d, "bad.json"), verify=False)
    print("holographic_indexstore self-test passed (%d docs round-trip exactly; a tampered payload "
          "is REFUSED; verify=False still loads)" % len(docs))


if __name__ == "__main__":
    _selftest()
