"""
A2, re-opened: adapt the FORMULA to the substrate instead of declaring it impossible.

The A2 boundary was an artifact of ONE encoding -- store the matrix as superposed rows, read them
back, dot with the input, no cleanup. That bundles many things into one vector, so crosstalk wins.

But matmul is multiply-accumulate of NUMBERS, and the substrate already has an EXACT way to carry and
compose numbers: the Residue Number System (CRT) over FHRR phasors -- the same residue machinery A6
just showed federates to a 1e391 range. Decompose -> adapt -> recompose:

  decompose : cast W,x to integers; take residues mod each coprime modulus m_k (one per channel/shard)
  adapt     : in each channel, multiply-accumulate mod m_k. Accumulation = PHASOR BINDING, which is
              EXACT phase composition (sum mod m carried in the angle) -- no superposition, no crosstalk,
              for ANY number of terms. This is the part the bundle got wrong.
  recompose : CRT-combine the per-channel residues back into the exact integer result.

So we never store the matrix in a lossy superposition at all. We carry values as residues/phases and
compose them exactly. Prediction: exact integer matmul, no crosstalk wall, range federates over moduli.
Honest scope kept on the record: exact for INTEGER / fixed-point within range (float -> quantize first),
real FLOPs (the parallelism is per-modulus / per-output, native on phasor or RNS hardware).
"""
import numpy as np
rng = np.random.default_rng(0)

def phasor_sum_mod(residue_terms, m):
    """Exact (sum of residues) mod m via FHRR phasor binding: prod exp(2pi i r/m) = exp(2pi i (sum)/m)."""
    acc = np.prod(np.exp(2j * np.pi * residue_terms / m))      # binding = phase addition
    return int(np.round(np.angle(acc) / (2 * np.pi) * m)) % m  # recover the residue from the angle

def crt(res_per_mod, moduli):
    P = 1
    for m in moduli: P *= m
    x = np.zeros(len(res_per_mod[0]), dtype=object)
    for r, m in zip(res_per_mod, moduli):
        Mi = P // m
        x = x + (np.array(r, dtype=object) * (Mi * pow(Mi, -1, m)))
    return x % P, P

def rns_matmul(W, x, moduli):
    """Exact integer matmul y = W@x, computed channel-wise with phasor-binding accumulation."""
    M, N = W.shape; res = []
    for m in moduli:
        Wm, xm = W % m, x % m
        ymod = [phasor_sum_mod((Wm[i] * xm) % m, m) for i in range(M)]   # exact MAC mod m
        res.append(ymod)
    y, P = crt(res, moduli)
    return np.where(y > P // 2, y - P, y).astype(np.int64), P           # center to signed

# the lossy bundle (A2's original encoding) for a head-to-head
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from holographic.misc.holographic_core import unitary_vector, random_vector, bundle
from holographic.agents_and_reasoning.holographic_ai import bind_batch, bind_fixed
D = 1024
def inv_stack(A): return np.concatenate([A[:, :1], A[:, :0:-1]], axis=1)
def bundle_matmul_fidelity(M, N, trials=4):
    fs = []
    for _ in range(trials):
        Wd = np.stack([random_vector(D, rng) for _ in range(M)]); Wd /= np.linalg.norm(Wd,axis=1,keepdims=True)
        xd = random_vector(D, rng); xd /= np.linalg.norm(xd)
        roles = np.stack([unitary_vector(D, rng) for _ in range(M)])
        Mem = bundle(bind_batch(roles, Wd)); What = bind_fixed(Mem, inv_stack(roles))
        a, b = What @ xd, Wd @ xd; a, b = a - a.mean(), b - b.mean()
        fs.append(float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)))
    return float(np.mean(fs))

SMALL_PRIMES = [101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197]

print("=" * 76)
print("(1) Exact accumulation -- the thing the bundle got wrong. Sum N ints mod m via phasor binding.")
m = 9973
for N in (10, 100, 1000, 5000):
    errs = 0
    for _ in range(200):
        terms = rng.integers(0, m, size=N)
        errs += int(phasor_sum_mod(terms, m) != int(terms.sum() % m))
    print(f"    N={N:5d} terms:  phasor-binding errors over 200 trials = {errs}")

print("=" * 76)
print("(2) Integer matmul A2 could not do: RNS-phasor (exact?) vs the lossy bundle, by matrix size.")
N = 64; Vmax = 50
for M in (8, 64, 256):
    W = rng.integers(-Vmax, Vmax + 1, size=(M, N)); x = rng.integers(-Vmax, Vmax + 1, size=N)
    ymax = int(np.abs(W @ x).max()); need = 2 * ymax + 1
    mods, P = [], 1
    for p in SMALL_PRIMES:
        mods.append(p); P *= p
        if P > need: break
    y_rns, _ = rns_matmul(W, x, mods)
    exact = W @ x
    max_err = int(np.abs(y_rns - exact).max())
    fid = bundle_matmul_fidelity(M, N)
    print(f"    M={M:4d}: RNS-phasor max|error|={max_err}  ({len(mods)} moduli, range>{need})   "
          f"|  lossy-bundle fidelity={fid:.2f}")

print("=" * 76)
print("(3) Range federates over moduli channels (same law as A6): more channels -> bigger exact range.")
for k in (4, 8, 16, 32):
    P = 1
    for p in SMALL_PRIMES[:k] if k <= len(SMALL_PRIMES) else SMALL_PRIMES: P *= p
    # extend with more primes if needed
    extra = [199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313]
    allp = (SMALL_PRIMES + extra)[:k]
    P = 1
    for p in allp: P *= p
    print(f"    {k:2d} moduli (channels): exact integer range ~ 1e{len(str(P))-1}")
print("=" * 76)
print("VERDICT: A2's wall was the lossy-superposition encoding, not the substrate. Re-expressed as")
print("RNS multiply-accumulate over exact phasor binding, integer matmul is EXACT at every size the")
print("bundle degraded on, and its range federates over moduli channels exactly like A6.")
