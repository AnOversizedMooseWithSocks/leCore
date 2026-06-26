"""Benchmark: holostuff's geometry-preserving rate-distortion code (quant='rd') vs the general-purpose
stdlib compressors a practitioner would reach for (zlib / lzma), at MATCHED cosine fidelity (BLD-2).

The honest question: when is the engine's structure-aware code worth using over off-the-shelf compression?
Answer, measured below: rd WINS by a large margin on data with real low-rank structure -- it spends bits only
on the directions that carry variance, the consolidation/KLT subspace -- and LOSES on full-rank random data,
where there is no structure to exploit and the shared KLT basis costs more than it saves. The standard tool is
the right default when the data has no low-rank structure; that case is kept on the record, not hidden.

Run:  python benchmarks/bench_compression.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import zlib
import lzma
from holographic_ratedistortion import geometry_preserving_code, reconstruct, pack_code


def _unit_rows(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def _cos_mean(A, B):
    return float(np.mean(np.einsum("ij,ij->i", A, B) /
                         (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)))


def _int8_then(A, compressor):
    """int8 scalar quantization (the move a vector database makes) followed by a general compressor. Returns
    (bits_per_vector, reconstruction_cosine). int8 sits at-or-above rd's 0.999 fidelity, so this is a
    matched-or-better-fidelity baseline -- a fair, even slightly generous, opponent."""
    scale = np.abs(A).max(1, keepdims=True) / 127 + 1e-12
    q = np.round(A / scale).astype(np.int8)
    rec = q.astype(float) * scale
    raw = q.tobytes() + scale.astype(np.float32).tobytes()
    return len(compressor(raw)) * 8 / len(A), _cos_mean(A, rec)


def _make(kind, N, D, seed):
    rng = np.random.default_rng(seed)
    if kind == "structured":                              # rank-8: real low-rank structure, the rd win condition
        return _unit_rows(rng.standard_normal((N, 8)) @ rng.standard_normal((8, D)))
    return _unit_rows(rng.standard_normal((N, D)))        # full-rank random: nothing for the KLT to exploit


def compare_compression(D=512, target_cos=0.999, datasets=("structured", "random"), Ns=(200, 2000), seed=0):
    """Measure rd vs int8+zlib vs int8+lzma vs lossless f32+zlib, returning one row per (dataset, N)."""
    rows = []
    for kind in datasets:
        for N in Ns:
            A = _make(kind, N, D, seed)
            code = geometry_preserving_code(A, target_cos=target_cos)
            rd_bits = len(pack_code(code)) * 8 / N                   # the REAL packed size, not just the coeffs
            rd_cos = _cos_mean(A, reconstruct(code))
            z_bits, z_cos = _int8_then(A, zlib.compress)
            l_bits, _ = _int8_then(A, lambda b: lzma.compress(b))
            f_bits = len(zlib.compress(A.astype(np.float32).tobytes())) * 8 / N
            rows.append({"dataset": kind, "N": N, "rd_bits": rd_bits, "rd_cos": rd_cos,
                         "int8_zlib_bits": z_bits, "int8_zlib_cos": z_cos, "int8_lzma_bits": l_bits,
                         "f32_zlib_bits": f_bits, "rd_wins": rd_bits < z_bits})
    return rows


def _print(rows):
    print("Compression: geometry-preserving rd code vs stdlib zlib/lzma, at matched cosine fidelity")
    print("(bits/vector, lower is better; f32+zlib is lossless = the cosine-1.0 reference)")
    for r in rows:
        win = "rd WINS" if r["rd_wins"] else "zlib WINS (no low-rank to exploit)"
        print(f"  {r['dataset']:11} N={r['N']:5}: rd {r['rd_bits']:8.1f} (cos {r['rd_cos']:.4f})  |  "
              f"int8+zlib {r['int8_zlib_bits']:7.1f} (cos {r['int8_zlib_cos']:.4f})  |  "
              f"int8+lzma {r['int8_lzma_bits']:7.1f}  |  f32+zlib {r['f32_zlib_bits']:8.1f}   -> {win}")


if __name__ == "__main__":
    _print(compare_compression())
