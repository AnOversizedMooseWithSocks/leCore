import numpy as np, time
from scipy.linalg import hadamard   # ONLY to verify correctness of my fwht

# --- fast Walsh-Hadamard transform, vectorized, O(D log D), D = 2^m ---
def fwht(a):
    a = a.astype(np.float64).copy(); n = len(a); h = 1
    while h < n:
        a = a.reshape(n // (2*h), 2, h)
        a = np.concatenate([a[:,0,:]+a[:,1,:], a[:,0,:]-a[:,1,:]], axis=1).reshape(n)
        h *= 2
    return a

# verify against scipy's dense Hadamard matrix
for m in [3, 6, 9]:
    D = 2**m; x = np.random.default_rng(m).standard_normal(D)
    assert np.allclose(fwht(x), hadamard(D) @ x, atol=1e-9), f"fwht wrong at D={D}"
print("fwht matches dense Hadamard matrix: OK")

# --- structured holographic key operator: A = WHT . sign . scatter (matrix-free) ---
class WHTKeys:
    def __init__(self, K, D, seed=0):
        assert D & (D-1) == 0, "D must be a power of 2"
        rng = np.random.default_rng(seed)
        self.K, self.D = K, D
        self.signs = rng.choice([-1.0, 1.0], size=D)
        self.pos = rng.permutation(D)[:K]      # K distinct scatter slots
        self.scale = 1.0/np.sqrt(D)
    def apply(self, v):                        # R^K -> R^D
        x = np.zeros(self.D); x[self.pos] = v
        return fwht(x * self.signs) * self.scale
    def adjoint(self, y):                      # R^D -> R^K
        return (fwht(y) * self.scale * self.signs)[self.pos]

# verify exactness: adjoint(apply(v)) == v  (i.e. A^T A = I)
K, D = 4000, 16384
keys = WHTKeys(K, D, 0); v = np.random.default_rng(1).standard_normal(K)
print("A^T A = I (exact recovery undamaged):", np.allclose(keys.adjoint(keys.apply(v)), v, atol=1e-9))

# --- memory + speed vs dense random projection (the thing that OOM'd) ---
rng = np.random.default_rng(0)
Pdense = rng.standard_normal((K, D)).astype(np.float64); Pdense /= np.linalg.norm(Pdense,axis=1,keepdims=True)
dense_mem = Pdense.nbytes
wht_mem = keys.signs.nbytes + keys.pos.nbytes
print(f"\nkey storage:  dense P = {dense_mem/1e6:.1f} MB   |   WHT = {wht_mem/1e3:.0f} KB   "
      f"({dense_mem/wht_mem:.0f}x less)")

# encode+decode speed
t0=time.perf_counter()
for _ in range(20): h=Pdense.T@v; r=Pdense@h
t_dense=(time.perf_counter()-t0)/20
t0=time.perf_counter()
for _ in range(20): h=keys.apply(v); r=keys.adjoint(h)
t_wht=(time.perf_counter()-t0)/20
print(f"encode+decode: dense = {t_dense*1000:.1f} ms   |   WHT = {t_wht*1000:.2f} ms   "
      f"({t_dense/t_wht:.0f}x faster)")
