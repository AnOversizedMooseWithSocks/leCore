"""bench_dictionary_compression.py -- the honest, reproducible measurement behind shipping the dictionary as lzma.

The question was: can leCore's OWN holographic compression pack the vendored English dictionary smaller than a zip
file? The answer, measured rather than assumed, is NUANCED and worth keeping on the record:

  * The dictionary needs EXACT, verbatim recovery (a word -> its precise definition string). leCore's VSA/holographic
    compression is geometry-preserving but LOSSY (the "10-63 bytes/record beats SQLite" result is for approximate
    VECTOR recall) -- so it would garble definitions. Wrong tool for exact text.

  * leCore's LOSSLESS coder (holographic_codec.PredictiveCodec) round-trips text EXACTLY, but it is a SHARED-MODEL
    codec: the decoder needs the predictor's meaning matrix, which is built from the corpus and is far larger than the
    text it compresses (measured below: a ~7 MB model to compress ~78 KB of text). It shines when the model is already
    shared and you stream NEW text through it -- not as a self-contained FILE compressor.

  * So for the dictionary FILE, the right tool is a mature lossless BYTE coder. Among stdlib options, lzma packs this
    JSON ~45% smaller than gzip -- a real win, exactly lossless, and still stdlib-only (constitution-clean). That is
    what we ship: dictionary.json.xz.

Run: python bench_dictionary_compression.py     (uses the vendored dictionary; prints the table)
"""
import gzip
import bz2
import lzma
import json
import time


def _load_dictionary():
    """Load the vendored dictionary regardless of whether it's stored as .xz (current) or .gz (older)."""
    import holographic.misc.holographic_dictionary as hd
    path = hd._DATA_PATH
    opener = lzma.open if path.endswith(".xz") else gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def byte_coder_table(raw):
    """gzip vs bz2 vs lzma on the exact JSON bytes -- all lossless, self-contained, stdlib."""
    rows = []
    for name, fn in (("gzip -9", lambda b: gzip.compress(b, 9)),
                     ("bz2 -9", lambda b: bz2.compress(b, 9)),
                     ("lzma -9", lambda b: lzma.compress(b, preset=9))):
        t = time.time()
        out = fn(raw)
        rows.append((name, len(out), len(raw) / len(out), time.time() - t))
    return rows


def lecore_codec_point(D, n_defs=1200, dim=256):
    """Show WHY leCore's lossless codec isn't a file compressor here: it round-trips exactly, but the model it needs
    dwarfs the text. Returns (lossless, raw_bytes, rank_entropy_bytes, model_bytes, lzma_bytes) for a small sample."""
    import re
    import numpy as np
    from holographic.agents_and_reasoning.holographic_meaning_predict import MeaningPredictor
    from holographic.misc.holographic_codec import PredictiveCodec

    defs = [v.get("d", "") for _, v in list(D.items())[:n_defs]]
    sents = [re.findall(r"[a-z]+", d.lower()) for d in defs]
    toks = [t for s in sents for t in s]
    raw = " ".join(defs).encode("utf-8")

    mp = MeaningPredictor(dim=dim)
    mp.fit_space(sents, window=2)                          # builds the vocab + meaning matrix FROM the corpus
    codec = PredictiveCodec(mp)
    code = codec.compress(toks)
    lossless = list(codec.decompress(code)) == list(toks)

    ranks = np.array(code["ranks"])
    _, counts = np.unique(ranks, return_counts=True)
    p = counts / counts.sum()
    rank_bytes = float(-(p * np.log2(p)).sum()) * len(ranks) / 8      # idealised rank-stream entropy (no framing)
    model_bytes = mp.M.nbytes                                          # the model the decoder must also have
    lzma_bytes = len(lzma.compress(raw, preset=9))
    return lossless, len(raw), rank_bytes, model_bytes, lzma_bytes


def main():
    D = _load_dictionary()
    raw = json.dumps(D, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    print("dictionary: %d entries, %.2f MB of compact JSON\n" % (len(D), len(raw) / 1e6))

    print("LOSSLESS BYTE CODERS (self-contained, stdlib) -- the fair comparison for exact-recovery text:")
    print("  %-10s %10s %8s %8s" % ("coder", "size", "ratio", "time"))
    for name, size, ratio, secs in byte_coder_table(raw):
        print("  %-10s %8.2f MB   x%.2f   %5.1fs%s" % (name, size / 1e6, ratio, secs,
              "   <- shipped" if name.startswith("lzma") else ""))

    print("\nleCore LOSSLESS predictive codec on a %d-definition sample (WHY it's not a file compressor here):" % 1200)
    lossless, raw_b, rank_b, model_b, lzma_b = lecore_codec_point(D)
    print("  exact round-trip: %s" % lossless)
    print("  sample text                       : %6.1f KB" % (raw_b / 1e3))
    print("  rank-stream idealised entropy     : %6.1f KB   (excludes model + framing)" % (rank_b / 1e3))
    print("  MODEL the decoder must also carry : %6.1f KB   (built from the corpus -- the catch)" % (model_b / 1e3))
    print("  lzma of the same sample           : %6.1f KB   (self-contained)" % (lzma_b / 1e3))
    print("\nConclusion: lzma is the honest win for the dictionary FILE (lossless, stdlib, ~45%% under gzip). leCore's")
    print("holographic compression is the tool for LOSSY vector/record recall, not verbatim dictionary text.")


if __name__ == "__main__":
    main()
