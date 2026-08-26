# This one-off research script lives in archive/; put the library at the repo
# root on the path so its imports keep working when run from anywhere.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from holographic.io_and_interop.holographic_image import WHTKeys, _dct_matrix, _demo_image, _psnr

S=160; img=_demo_image(S); npix=S*S; M=_dct_matrix(S)
def dct2(a): return M@a@M.T
def idct2(C): return M.T@C@M
def store_coeffs(K):
    idx=[]; vals=[]
    for c in range(3):
        f=dct2(img[...,c]).ravel(); i=np.argpartition(np.abs(f),-K)[-K:]; idx.append(i); vals.append(f[i])
    return idx, vals

def recon_from_grains(keys, grains, norms, idx, biht_iters=0, tau=0.0):
    out=[]
    for c in range(3):
        v = keys.adjoint(grains[c].astype(float))          # one-bit matched filter
        v *= norms[c]/ (np.linalg.norm(v)+1e-12)            # restore scale (1 stored float)
        for _ in range(biht_iters):                         # sign-consistency refinement (BIHT)
            resid = np.sign(keys.apply(v)) - grains[c]
            v -= tau*keys.adjoint(resid)
            v *= norms[c]/(np.linalg.norm(v)+1e-12)
        f=np.zeros(npix); f[idx[c]]=v; out.append(idct2(f.reshape(S,S)))
    return np.clip(np.stack(out,-1),0,1)

K=2000
idx, vals = store_coeffs(K)
norms=[np.linalg.norm(v) for v in vals]
print(f"=== film-grain (1-bit) holographic plate, {S}x{S} colour, K={K} coeffs/chan ===")
print(f"{'grains D':>9} {'D/K':>5} {'float plate':>12} {'1-bit matched':>14} {'1-bit BIHT':>11} {'plate bytes 1b vs f64':>22}")
for D in [4096, 8192, 16384, 32768, 65536]:
    keys=WHTKeys(K, D, seed=0)
    plates=[keys.apply(v) for v in vals]
    # float-plate reference recon (exact-ish)
    fr=[]
    for c in range(3):
        v=keys.adjoint(plates[c]); f=np.zeros(npix); f[idx[c]]=v; fr.append(idct2(f.reshape(S,S)))
    p_float=_psnr(img,np.clip(np.stack(fr,-1),0,1))
    grains=[np.sign(p) for p in plates]                     # THE 1-BIT GRAIN FIELD
    p_m=_psnr(img, recon_from_grains(keys,grains,norms,idx))
    p_b=_psnr(img, recon_from_grains(keys,grains,norms,idx, biht_iters=30, tau=1.0/np.sqrt(D)))
    bits_1=3*D/8; f64=3*D*8
    print(f"{D:9d} {D/K:5.0f} {p_float:11.1f}dB {p_m:13.1f}dB {p_b:10.1f}dB   {bits_1/1e3:6.1f}KB vs {f64/1e3:6.0f}KB")
