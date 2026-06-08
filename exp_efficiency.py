import numpy as np, time
def to_bits(V): return np.packbits(V > 0, axis=1)
def hamming(qb, Bb): return np.bitwise_count(np.bitwise_xor(qb, Bb)).sum(1)
def unit(v): return v/np.linalg.norm(v)

rng = np.random.default_rng(0); N, D = 8000, 10000
V = rng.standard_normal((N, D)).astype(np.float32); V /= np.linalg.norm(V,axis=1,keepdims=True)
Vb = to_bits(V)
print(f"=== {N} items, D={D} ===")
print(f"float32 DB {V.nbytes/1e6:.0f} MB  ->  1-bit DB {Vb.nbytes/1e6:.1f} MB "
      f"({V.nbytes/Vb.nbytes:.0f}x smaller; {64/ (Vb.nbytes/V.nbytes*32):.0f}x vs float64)")

# cleanup under MODERATE noise: query = target + c*unit_noise  (c=2 -> cos~0.45)
trials=500; c=2.0; qs=[]
for _ in range(trials):
    i=int(rng.integers(N)); nz=unit(rng.standard_normal(D).astype(np.float32))
    qs.append((i, unit(V[i]+c*nz)))
t0=time.perf_counter(); fok=sum(int(np.argmax(V@q)==i) for i,q in qs); tf=time.perf_counter()-t0
t0=time.perf_counter(); bok=sum(int(np.argmin(hamming(np.packbits(q>0),Vb))==i) for i,q in qs); tb=time.perf_counter()-t0
print(f"cleanup accuracy (cos~0.45 query):  float={fok/trials*100:.1f}%   1-bit={bok/trials*100:.1f}%")
print(f"500-query time:                     float={tf*1000:.0f} ms   1-bit={tb*1000:.0f} ms  ({tf/tb:.1f}x)")

# rank fidelity across a REAL spread of cosines (slerp query toward 200 targets by varying amounts)
q=unit(rng.standard_normal(D).astype(np.float32)); targs=[]
for amt in np.linspace(0.1,0.95,200):
    t=unit(rng.standard_normal(D).astype(np.float32)); targs.append(unit((1-amt)*q+amt*t))
T=np.array(targs); Tb=to_bits(T)
cos=T@q; est=1-2*hamming(np.packbits(q>0),Tb)/D
rc=np.corrcoef(cos,est)[0,1]
print(f"sign-similarity vs true cosine (real spread): correlation={rc:.3f}, top-10 overlap="
      f"{len(set(np.argsort(-cos)[:10])&set(np.argsort(-est)[:10]))}/10")
