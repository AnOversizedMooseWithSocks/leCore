# This one-off research script lives in archive/; put the library at the repo
# root on the path so its imports keep working when run from anywhere.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from holographic.io_and_interop.holographic_image import HolographicImage, _demo_image, _psnr, _cg

img = _demo_image(240)
hi = HolographicImage(img.shape, keep=4000, dim=16384, seed=0).store(img)
plates = hi._plates
pf = _psnr(img, hi.reconstruct())
print(f"float plate reference: {pf:.1f} dB\n")

def q_uniform(x, bits):
    lo, hi_ = x.min(), x.max(); step = (hi_-lo)/(2**bits-1)
    return np.round((x-lo)/step)*step+lo

def q_lloyd(x, bits, iters=30):
    L = 2**bits
    c = np.quantile(x, (np.arange(L)+0.5)/L)         # init at quantiles
    for _ in range(iters):
        idx = np.argmin(np.abs(x[:,None]-c[None,:]), axis=1)
        for k in range(L):
            m = idx==k
            if m.any(): c[k] = x[m].mean()
    idx = np.argmin(np.abs(x[:,None]-c[None,:]), axis=1)
    return c[idx], c

def decode(plates_q, idx_, shape, M):
    npix = shape[0]*shape[1]; out=[]
    for i,p in zip(idx_, plates_q):
        v = hi._adjoint(p); f=np.zeros(npix); f[i]=v
        out.append((M.T@f.reshape(shape)@M))
    return np.clip(np.stack(out,-1),0,1)

M = hi._M[240]
print(f"{'bits':>4} {'uniform':>9} {'Lloyd-Max':>10}")
for b in [2,3,4,5]:
    pu = _psnr(img, decode([q_uniform(p,b) for p in plates], hi._idx,(240,240),M))
    pl = _psnr(img, decode([q_lloyd(p,b)[0] for p in plates], hi._idx,(240,240),M))
    print(f"{b:>4} {pu:8.1f}dB {pl:9.1f}dB   (+{pl-pu:.1f})")
