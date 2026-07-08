"""
holographic_rns.py  --  exact integer / fixed-point arithmetic carried over FHRR phasors (Path D's second lever).

The capacity arc says a lossy SUPERPOSITION readout of a matrix (bundle the rows, unbind, dot with the input,
no cleanup) is capped by crosstalk -- general matmul in superposition dies as the matrix grows. But matmul is
multiply-accumulate of NUMBERS, and the FHRR side of the engine already carries a number EXACTLY as a phase:
a unit phasor exp(2*pi*i*r/m) IS the residue r mod m, and BINDING phasors adds their phases -- so a product of
phasors is exp(2*pi*i*(sum r)/m), i.e. the exact sum of residues mod m, for ANY number of terms and with NO
crosstalk. That is the one thing the lossy bundle got wrong.

So carry each number as its residues over several coprime moduli (a Residue Number System), do every
multiply-accumulate as exact phasor-binding modular arithmetic, and recompose the integer with the Chinese
Remainder Theorem. The result is an EXACT integer matmul whose dynamic range FEDERATES over moduli channels
(more moduli -> bigger exact range) -- the arithmetic sibling of the storage array's federation, the same
"more channels = more capacity, coordinated by a thin recompose layer" move one rung down.

Honest scope kept on the record: exact for INTEGER / fixed-point operands within the moduli range. A float must
be QUANTIZED first, and that fixed-point rounding is the only error -- a bit-depth question, separable from and
unlike the crosstalk wall. And the FLOPs are real: the parallelism is per-modulus / per-output, native on
phasor or RNS hardware, not free on a CPU.
"""
import numpy as np

# A pool of coprime moduli (primes). More channels -> bigger exact range -- the federation axis for arithmetic.
_PRIMES = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
           199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
           317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439]


def phasor_sum_mod(residue_terms, m):
    """Exact (sum of residues) mod m via FHRR phasor binding:  prod(exp(2*pi*i*r/m)) = exp(2*pi*i*(sum r)/m).

    Binding adds phases, so the modular sum is carried in the angle EXACTLY for any number of terms -- the
    crosstalk-free accumulation a superposed bundle cannot do. This is the same phase-composition primitive as
    holographic_fhrr's binding, restricted to a single phase channel."""
    acc = np.prod(np.exp(2j * np.pi * np.asarray(residue_terms, dtype=float) / m))   # binding = phase addition
    return int(np.round(np.angle(acc) / (2 * np.pi) * m)) % m


def choose_moduli(need_range):
    """The smallest prefix of the coprime pool whose product exceeds `need_range` (the exact dynamic range the
    result needs). Returns (moduli, product). More moduli = bigger exact range -- federation for arithmetic."""
    mods, P = [], 1
    for p in _PRIMES:
        mods.append(p)
        P *= p
        if P > need_range:
            return mods, P
    raise ValueError("need_range exceeds the built-in moduli pool; extend _PRIMES")


def crt(res_per_mod, moduli):
    """Chinese Remainder Theorem: recompose per-modulus residue vectors into the exact integers (mod the
    product of the moduli). `res_per_mod[k]` is the length-M residue vector for modulus `moduli[k]`."""
    P = 1
    for m in moduli:
        P *= m
    x = np.zeros(len(res_per_mod[0]), dtype=object)
    for r, m in zip(res_per_mod, moduli):
        Mi = P // m
        x = x + (np.array(r, dtype=object) * (Mi * pow(Mi, -1, m)))     # standard CRT recomposition
    return x % P, P


def rns_matmul(W, x, moduli=None):
    """Exact integer matmul y = W @ x: every multiply-accumulate is done as phasor-binding modular arithmetic
    (one channel per modulus) and recomposed by CRT. `moduli` defaults to the smallest channel set covering the
    signed result range. Returns the exact signed integer result (centered into [-P/2, P/2))."""
    W = np.asarray(W, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)
    if moduli is None:
        ymax = int(np.abs(W.astype(object) @ x.astype(object)).max()) if W.size else 1
        moduli, _ = choose_moduli(2 * ymax + 1)
    M = W.shape[0]
    res = []
    for m in moduli:
        Wm, xm = W % m, x % m
        res.append([phasor_sum_mod((Wm[i] * xm) % m, m) for i in range(M)])   # exact MAC mod m, per row
    y, P = crt(res, moduli)
    y = np.array([int(v) for v in y])
    return np.where(y > P // 2, y - P, y).astype(np.int64)


def quantize(a, scale):
    """Fixed-point quantize: round(a * scale) as integers. Larger scale = finer resolution, but bigger
    integers, so a wider result range and more moduli channels."""
    return np.round(np.asarray(a, float) * scale).astype(np.int64)


def rns_matmul_float(W, x, scale=64, moduli=None):
    """Exact-arithmetic matmul on quantized floats: quantize W and x by `scale`, do the exact integer matmul,
    dequantize by scale**2. KEPT NEGATIVE: the result is exact for the QUANTIZED operands -- the only error is
    the fixed-point rounding set by `scale`, NOT crosstalk (it does not grow with matrix size)."""
    Wi, xi = quantize(W, scale), quantize(x, scale)
    return rns_matmul(Wi, xi, moduli).astype(float) / (scale * scale)


def _selftest():
    rng = np.random.default_rng(0)
    # (1) exact accumulation for many terms -- the thing the bundle got wrong
    for N in (10, 1000, 5000):
        terms = rng.integers(0, 9973, size=N)
        assert phasor_sum_mod(terms, 9973) == int(terms.sum() % 9973), "phasor accumulation must be exact"
    print("[rns selftest] exact modular accumulation up to N=5000 terms: 0 errors")
    # (2) exact integer matmul at a size a lossy bundle cannot do
    M, N = 256, 64
    W = rng.integers(-50, 51, size=(M, N)); x = rng.integers(-50, 51, size=N)
    y = rns_matmul(W, x)
    assert np.array_equal(y, W @ x), "RNS integer matmul must be exact"
    print(f"[rns selftest] exact integer matmul M={M}, N={N}: max|error| = {int(np.abs(y - W @ x).max())}")
    # (3) range federates over moduli channels
    _, P4 = choose_moduli(10 ** 8); _, P32 = choose_moduli(10 ** 60)
    print(f"[rns selftest] range federates: ~1e{len(str(P4)) - 1} (few moduli) -> ~1e{len(str(P32)) - 1} (more)")
    assert P32 > P4
    print("[rns selftest] OK")


if __name__ == "__main__":
    _selftest()
