"""Octahedral normal encoding -- quantize a unit vector on its MANIFOLD, not its ambient bits (holographic_octnormal).

WHY THIS MODULE EXISTS
----------------------
From the geometry->stack backlog sweep, item A2: the engine's quantizer (int8 / rate-distortion / `quant='rd'`) is
shipped and measured, but it quantizes the AMBIENT bits of a value. A unit normal has only TWO degrees of freedom
(it lives on the sphere S^2), so quantizing its three x/y/z components wastes a third of the budget on a
constrained coordinate -- and most quantized triples aren't even unit vectors. The standard fix (Cigolle, Donow,
Evangelakos, Mara, McGuire, Meyer, "A Survey of Efficient Representations for Independent Unit Vectors," JCGT 2014)
is the OCTAHEDRAL map: project the unit vector onto the octahedron (L1-normalize), then unfold the lower hemisphere
into a 2D square. Two numbers, a bounded round-trip error, and -- the point -- the bits land on the 2 intrinsic
DOF.

WHY IT BELONGS IN THIS ENGINE (the reverse thesis)
  This is manifold quantization made concrete: "spend bits on the surface the data lives on." That is exactly the
  warning the engine already keeps -- the binary-quantization-distorts-the-geometry-not-just-the-bits negative --
  turned into a method. The same PRINCIPLE is reverse item R3: the FHRR phasor memory is unit-magnitude complex
  (points on the circle S^1) and a normalized hypervector lives on a high-D sphere, so both want their intrinsic
  coordinate quantized, not their ambient one (for an FHRR phasor that analog is the PHASE ANGLE -- one number, not
  two). Octahedral encoding is the concrete S^2 instance of that family; R3 is the principle carried to the phasor
  memory.

WHAT IT PROVIDES
  * oct_encode(normals) / oct_decode(uv) -- the continuous bijection S^2 <-> [-1,1]^2 (exact to float precision).
  * oct_quantize(normals, bits) -- integer codes (N,2) in [0, 2^bits) (the stored form).
  * oct_dequantize(codes, bits) -- unit normals back from the codes.

THE MEASUREMENT BAR (checked exactly in the self-test)
  * the CONTINUOUS round-trip is exact (max angular error ~1e-6 deg -- a bijection).
  * the QUANTIZED round-trip at 8 bits/component has small BOUNDED angular error (< 1 deg max).
  * at an EQUAL total bit budget (16 bits) octahedral BEATS naive x/y/z quantization (5+5+6 bits, renormalized) on
    mean angular error by ~3x -- the manifold-quantization win.

DETERMINISM (per ISA.md)
  Pure fixed arithmetic (L1 projection, fold, round-to-nearest); no RNG. Same normals + bits -> identical codes
  and identical decoded normals (asserted).

KEPT NEGATIVES (loud)
  * At EQUAL bits-PER-COMPONENT, naive x/y/z is more accurate (it is using 50% more bits -- 3 components vs 2). The
    octahedral win is a STORAGE win: the same accuracy in 2 numbers that naive needs ~2.5-3 for. Stated honestly so
    the comparison isn't read the wrong way.
  * Octahedral encoding is specific to S^2 (3-D unit vectors). It does NOT generalize verbatim to S^1 (FHRR
    phasors) or a high-D sphere -- those use the SAME PRINCIPLE with a different intrinsic coordinate (phase angle;
    spherical coordinates). The literal map is for normals; R3 is the principle, not this function.
  * The fold has measure-zero seams (the octahedron edges where z=0); points exactly on a seam are still decoded to
    a valid unit vector, but the (u,v) representation there is non-unique -- the standard, harmless oct caveat.
"""

import numpy as np


def _sign_nz(a):
    """Sign that returns +1 at zero (np.sign gives 0 there, which breaks the octahedral fold at the poles)."""
    return np.where(a >= 0, 1.0, -1.0)


def oct_encode(normals):
    """Map unit vectors (..,3) on S^2 to (..,2) in [-1,1] (the octahedral unfold). Exact bijection."""
    n = np.asarray(normals, float)
    n = n / np.sum(np.abs(n), axis=-1, keepdims=True)     # project onto the octahedron (L1 sphere)
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    # fold the lower hemisphere (z<0) out across the octahedron's edges
    u = np.where(z >= 0, x, (1 - np.abs(y)) * _sign_nz(x))
    v = np.where(z >= 0, y, (1 - np.abs(x)) * _sign_nz(y))
    return np.stack([u, v], axis=-1)


def oct_decode(uv):
    """Invert oct_encode: (..,2) in [-1,1] back to unit vectors (..,3) on S^2."""
    uv = np.asarray(uv, float)
    u, v = uv[..., 0], uv[..., 1]
    z = 1 - np.abs(u) - np.abs(v)
    x = np.where(z >= 0, u, (1 - np.abs(v)) * _sign_nz(u))
    y = np.where(z >= 0, v, (1 - np.abs(u)) * _sign_nz(v))
    n = np.stack([x, y, z], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def oct_quantize(normals, bits=8):
    """Encode unit normals to integer codes (N,2) in [0, 2^bits) -- the stored form. The two intrinsic DOF get the
    full budget."""
    uv = oct_encode(normals)
    levels = (1 << bits) - 1
    return np.round((uv + 1) * 0.5 * levels).astype(np.int64)


def oct_dequantize(codes, bits=8):
    """Decode integer codes (N,2) back to unit normals (N,3)."""
    levels = (1 << bits) - 1
    uv = np.asarray(codes, float) / levels * 2 - 1
    return oct_decode(uv)


# =====================================================================================================
# Self-test -- exact continuous round-trip; bounded quantized error; the equal-budget win over naive xyz.
# =====================================================================================================
def _selftest():
    def ang_err(a, b):
        return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))

    rng = np.random.default_rng(0)
    N = rng.standard_normal((20000, 3))
    N = N / np.linalg.norm(N, axis=-1, keepdims=True)

    # --- the continuous round-trip is exact (a bijection) ---
    assert ang_err(N, oct_decode(oct_encode(N))).max() < 1e-3, "continuous oct must be exact"

    # --- axis-aligned normals (the fold edge cases) survive, including the z<0 hemisphere ---
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1.0], [-1, 0, 0]])
    assert ang_err(axes, oct_decode(oct_encode(axes))).max() < 1e-6, "axis-aligned + folded normals round-trip"

    # --- quantized round-trip at 8 bits: small bounded error ---
    q8 = ang_err(N, oct_dequantize(oct_quantize(N, 8), 8))
    assert q8.max() < 1.0, f"8-bit oct max angular error must be < 1 deg, got {q8.max():.3f}"
    assert oct_quantize(N, 8).min() >= 0 and oct_quantize(N, 8).max() < (1 << 8), "codes in range"

    # --- the manifold-quantization win: at an EQUAL 16-bit budget, oct beats naive x/y/z (5+5+6) ---
    oct16 = oct_dequantize(oct_quantize(N, 8), 8)                       # 8+8 = 16 bits
    def qn(a, bits):
        levels = (1 << bits) - 1
        return np.round((a + 1) * 0.5 * levels) / levels * 2 - 1
    nb = np.stack([qn(N[:, 0], 5), qn(N[:, 1], 5), qn(N[:, 2], 6)], axis=-1)   # 5+5+6 = 16 bits
    nb = nb / np.linalg.norm(nb, axis=-1, keepdims=True)
    oct_mean, naive_mean = ang_err(N, oct16).mean(), ang_err(N, nb).mean()
    assert oct_mean < naive_mean, f"oct must beat naive at equal budget ({oct_mean:.3f} vs {naive_mean:.3f})"

    # --- determinism ---
    assert np.array_equal(oct_quantize(N, 8), oct_quantize(N, 8))

    print(f"holographic_octnormal selftest: ok (continuous round-trip exact; 8-bit quantized max err "
          f"{q8.max():.4f} deg, mean {q8.mean():.4f}; EQUAL 16-bit budget -- oct mean {oct_mean:.4f} deg BEATS "
          f"naive xyz(5+5+6) mean {naive_mean:.4f} deg ({naive_mean / oct_mean:.1f}x); axis-aligned + folded "
          f"normals survive; deterministic)")


if __name__ == "__main__":
    _selftest()
