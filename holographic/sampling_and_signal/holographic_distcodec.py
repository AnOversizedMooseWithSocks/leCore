"""holographic_distcodec.py -- C-4: the distributional codec (store the DISTRIBUTION, not the samples).

THE GAP (Rule-0 on record): "compress a point cloud to distribution moments" hit drift_train
(the ingredient) and no codec; "distributional codec" hit the atlas and the code-shape module.
This module is the codec: when the consumer needs the DISTRIBUTION a sample bank represents --
particle populations, splat sets, calibration banks, anything downstream code only ever
re-samples -- ship the drift model's d+1 moment hypervectors instead of the N points.

WHY THIS CAN PAY AT ALL (hdrift's central fact, reused): in FPE space the entire generative
model is mu (kernel mean embedding) + nu_j (d first-moment bundles) -- (d+1) x dim floats,
N-INDEPENDENT. The samples were never the asset; the density was. The codec makes the trade
explicit and PRICED:

    break_even_n = moment_bytes / bytes_per_point

below which storing the points raw is strictly cheaper and the codec says so (machine_place's
move: a unit that cannot pay reports the boundary, not a sales pitch).

QUANTIZED MOMENTS ARE THE RATE KNOB (measured before building, not assumed): coverage survives
aggressive quantization -- 8/6/4-bit moments all held coverage 1.0 with memorised_frac <= 0.016
on a two-cluster corpus (dim=2048, N=2000). The codec defaults to 6 bits with per-array scales;
the post-quantization AUDIT (generation_audit: coverage + memorisation, the H-series gate)
rides in every report, so a distribution the quantizer DID break is visible at encode time,
never discovered downstream.

WHAT DECODE RETURNS -- A MODEL, NOT THE POINTS (the honest type): distribution_decode rebuilds
a DriftModel (the encoder is a RECIPE -- n_dims/dim/bounds/bandwidth/seed -- so only numbers
ship, hdrift's own persistence discipline). Sampling from it yields points LIKE the originals,
never the originals. A caller who needs the exact points wanted a lossless codec and is told
so in the docstring and by the report's `kind` field.

KEPT NEGATIVES:
  * memorisation lives in the high-dimensional codebook-softmax regime, NOT the smooth-RBF
    regime (H-series, on record) -- this codec inherits that: it stores densities, and a
    corpus whose VALUE is its individual points (a lookup table) is the wrong customer;
  * drift_train's own refusal propagates: a corpus whose bandwidth probe collapses
    (everything at one point, or structureless) raises rather than shipping a model that
    only generates the mean;
  * the audit is a sample-based estimate (n_audit draws) -- coverage 1.0 certifies the
    audit's draw, not every future draw; k_modes must reflect the corpus's real mode count
    or coverage reads optimistically against too few targets.
"""

import json
import struct
import zlib

import numpy as np

from holographic.sampling_and_signal.holographic_hdrift import (
    DriftModel, VectorFunctionEncoder,
)

_MAGIC = b"LDC1"


def _quantize(v, bits):
    """Uniform symmetric quantization with a per-array scale. WHY per-array: mu and each nu_j
    have different dynamic ranges; one shared scale wastes levels on the smaller arrays."""
    scale = float(np.abs(v).max()) / (2 ** (bits - 1) - 1)
    if scale == 0.0:
        scale = 1.0
    q = np.round(v / scale).astype(np.int32)
    return q, scale


def _pack_ints(q, bits):
    """Pack signed ints at `bits` into bytes (offset to unsigned, then bit-pack via uint8
    views for 8, or 4-bit nibble packing). Only 4/6/8 supported -- the measured-useful set."""
    offset = q + (2 ** (bits - 1))
    if bits == 8:
        return offset.astype(np.uint8).tobytes()
    if bits == 4:
        flat = offset.astype(np.uint8).ravel()
        if len(flat) % 2:
            flat = np.append(flat, 0)
        return (flat[0::2] << 4 | flat[1::2]).tobytes()
    # 6 bits: 4 values -> 3 bytes
    flat = offset.astype(np.uint32).ravel()
    pad = (-len(flat)) % 4
    if pad:
        flat = np.append(flat, np.zeros(pad, dtype=np.uint32))
    grp = flat.reshape(-1, 4)
    b0 = (grp[:, 0] << 2 | grp[:, 1] >> 4).astype(np.uint8)
    b1 = ((grp[:, 1] & 0xF) << 4 | grp[:, 2] >> 2).astype(np.uint8)
    b2 = ((grp[:, 2] & 0x3) << 6 | grp[:, 3]).astype(np.uint8)
    return np.column_stack([b0, b1, b2]).tobytes()


def _unpack_ints(raw, n, bits):
    if bits == 8:
        offset = np.frombuffer(raw, dtype=np.uint8)[:n].astype(np.int32)
    elif bits == 4:
        b = np.frombuffer(raw, dtype=np.uint8)
        offset = np.empty(len(b) * 2, dtype=np.int32)
        offset[0::2] = b >> 4
        offset[1::2] = b & 0xF
        offset = offset[:n]
    else:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        grp = np.empty((len(b), 4), dtype=np.int32)
        grp[:, 0] = b[:, 0] >> 2
        grp[:, 1] = (b[:, 0] & 0x3) << 4 | b[:, 1] >> 4
        grp[:, 2] = (b[:, 1] & 0xF) << 2 | b[:, 2] >> 6
        grp[:, 3] = b[:, 2] & 0x3F
        offset = grp.ravel()[:n]
    return offset - (2 ** (bits - 1))


def distribution_encode(points, bits=6, dim=2048, n_audit=64, k_modes=2, mind=None):
    """Compress a sample bank to its DISTRIBUTION: train the drift model, quantize the d+1
    moment hypervectors at `bits` (4/6/8), ship moments + encoder recipe. Decode returns a
    DriftModel to sample from -- points LIKE the originals, never the originals (need
    exactness? use codec_place / residual_encode). The report prices the trade
    (break_even_n) and carries the post-quantization generation AUDIT so a broken
    distribution is visible at encode time. Returns {blob, report:{kind:'distribution',
    bytes, raw_bytes, zlib_bytes, ratio_vs_zlib, break_even_n, n_points, bits, pays,
    audit:{coverage, memorised_frac}}}."""
    assert bits in (4, 6, 8), "bits must be 4, 6 or 8 (the measured-useful set)"
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    n, d = points.shape
    raw = points.tobytes()
    zbase = zlib.compress(raw, 6)
    if mind is None:
        import lecore
        mind = lecore.UnifiedMind(dim=256, seed=0)

    model = mind.drift_train(points, dim=dim)   # drift_train's own refusal propagates
    enc = model.enc

    qmu, smu = _quantize(model.mu, bits)
    qnu, snu = zip(*[_quantize(model.nu[j], bits) for j in range(d)])
    header = dict(n_dims=int(enc.n_dims), dim=int(enc.dim),
                  bounds=[[float(a), float(b)] for a, b in model.bounds],
                  bandwidth=[float(b) for b in np.atleast_1d(enc.bandwidth)],
                  seed=int(getattr(enc, "seed", 0)), n_train=int(model.n_train),
                  bits=bits, scale_mu=smu, scale_nu=[float(s) for s in snu])
    hjson = zlib.compress(json.dumps(header, sort_keys=True).encode(), 9)
    body = _pack_ints(qmu, bits) + b"".join(_pack_ints(q, bits) for q in qnu)
    blob = _MAGIC + struct.pack("<I", len(hjson)) + hjson + zlib.compress(body, 6)

    # AUDIT WHAT WILL ACTUALLY SHIP: decode the blob we just built and audit THAT model --
    # auditing the pre-quantization model would certify a different artifact.
    shipped = distribution_decode(blob)
    X = mind.drift_generate(shipped, n=n_audit, seed=3)
    audit = mind.generation_audit(X, points, k_modes=k_modes)

    moment_bytes = len(blob)
    break_even_n = int(np.ceil(moment_bytes / (d * 8)))
    pays = bool(moment_bytes < len(zbase) and audit["coverage"] >= 0.5)
    return dict(blob=blob, report=dict(
        kind="distribution", bytes=moment_bytes, raw_bytes=len(raw),
        zlib_bytes=len(zbase), ratio_vs_zlib=len(zbase) / moment_bytes,
        break_even_n=break_even_n, n_points=n, bits=bits, pays=pays,
        audit=dict(coverage=float(audit["coverage"]),
                   memorised_frac=float(audit["memorised_frac"]))))


def distribution_decode(blob):
    """Rebuild the DriftModel from a distribution blob: encoder from its recipe (numbers
    only, deterministic), moments dequantized at their per-array scales. Sample with
    mind.drift_generate(model, ...). Raises on a foreign blob."""
    if blob[:4] != _MAGIC:
        raise ValueError("not a distribution-codec blob (bad magic)")
    hlen, = struct.unpack("<I", blob[4:8])
    h = json.loads(zlib.decompress(blob[8:8 + hlen]).decode())
    body = zlib.decompress(blob[8 + hlen:])
    bits, dim, d = h["bits"], h["dim"], h["n_dims"]
    per = {8: dim, 4: (dim + 1) // 2, 6: ((dim + 3) // 4) * 3}[bits]
    mu = _unpack_ints(body[:per], dim, bits).astype(float) * h["scale_mu"]
    nu = np.stack([_unpack_ints(body[per * (1 + j):per * (2 + j)], dim, bits).astype(float)
                   * h["scale_nu"][j] for j in range(d)])
    enc = VectorFunctionEncoder(d, dim=dim, bounds=[tuple(b) for b in h["bounds"]],
                                bandwidth=h["bandwidth"], seed=h["seed"])
    return DriftModel(enc, mu, nu, h["n_train"], bounds=[tuple(b) for b in h["bounds"]])


def _selftest():
    import lecore
    rng = np.random.default_rng(0)
    mind = lecore.UnifiedMind(dim=256, seed=0)
    pts = np.vstack([c + 0.05 * rng.standard_normal((1500, 2))
                     for c in ([0.3, 0.3], [0.7, 0.7])])

    # 1) The trade pays at this N and the shipped model's audit is healthy.
    r = distribution_encode(pts, bits=6, mind=mind)
    rep = r["report"]
    assert rep["pays"] and rep["ratio_vs_zlib"] > 3.0, rep
    assert rep["audit"]["coverage"] >= 0.9 and rep["audit"]["memorised_frac"] < 0.2, rep["audit"]

    # 2) Decode -> sample -> audit AGAIN, independently of encode's own audit.
    model = distribution_decode(r["blob"])
    X = mind.drift_generate(model, n=64, seed=11)
    a = mind.generation_audit(X, pts, k_modes=2)
    assert a["coverage"] >= 0.9 and a["memorised_frac"] < 0.2, a

    # 3) The samples are NOT the originals (distribution, not points): nearest-neighbour
    # distances must be spread, not a wall of zeros.
    dmin = np.array([np.linalg.norm(pts - x, axis=1).min() for x in X])
    assert (dmin > 1e-6).mean() > 0.9, "decode must not return memorised points"

    # 4) break_even honesty: a tiny bank must report pays=False with the boundary stated.
    tiny = pts[:64]
    rt = distribution_encode(tiny, bits=6, mind=mind)
    assert not rt["report"]["pays"] and rt["report"]["break_even_n"] > 64, rt["report"]

    # 5) 4-bit is the cheapest rung and must still cover (the measured feasibility, pinned).
    r4 = distribution_encode(pts, bits=4, mind=mind)
    assert r4["report"]["bytes"] < r["report"]["bytes"]
    assert r4["report"]["audit"]["coverage"] >= 0.9, r4["report"]["audit"]

    # 6) Determinism: identical inputs, byte-identical blob.
    assert distribution_encode(pts, bits=6, mind=mind)["blob"] == r["blob"]

    print("distcodec selftest OK -- 6-bit %.1fx, 4-bit %.1fx vs zlib, coverage %.2f"
          % (rep["ratio_vs_zlib"], r4["report"]["ratio_vs_zlib"], rep["audit"]["coverage"]))


if __name__ == "__main__":
    _selftest()
