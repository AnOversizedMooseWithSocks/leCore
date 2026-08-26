# This one-off research script lives in archive/; put the library at the repo
# root on the path so its imports keep working when run from anywhere.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from holographic.io_and_interop.holographic_image import HolographicImage, _demo_image, _psnr
img=_demo_image(240); S=240; npix=S*S; K=4000; D=16384
M=HolographicImage(img.shape,keep=K,dim=D,seed=0)._M[S]
def dct2(a): return M@a@M.T
def idct2(C): return M.T@C@M
keys=HolographicImage(img.shape,keep=K,dim=D,seed=0).keys

flats=[dct2(img[...,c]).ravel() for c in range(3)]
# per-channel top-K (current behaviour)
idx_pc=[np.argpartition(np.abs(f),-K)[-K:] for f in flats]
# shared top-K: rank by summed energy across channels, one index set for all
energy=sum(f**2 for f in flats); idx_sh=np.argpartition(energy,-K)[-K:]

def recon(idxs):
    out=[]
    for c in range(3):
        i=idxs[c] if isinstance(idxs,list) else idxs
        v=keys.adjoint(keys.apply(flats[c][i])); f=np.zeros(npix); f[i]=v; out.append(idct2(f.reshape(S,S)))
    return np.clip(np.stack(out,-1),0,1)

p_pc=_psnr(img,recon(idx_pc)); p_sh=_psnr(img,recon(idx_sh))
# honest sizes: keys + plate(4-bit) + index map (bitmask npix/8 per stored index set)
def total(idx_sets, bits=4):
    keyb=D//8 + K*2
    plate=3*(bits*D/8 + 2**bits*8)
    index=(1 if not isinstance(idx_sets,list) else 3)*np.ceil(npix/8)
    return (keyb+plate+index)/1e3
print(f"per-channel indices: {p_pc:.1f} dB   total(4-bit) = {total(idx_pc):.1f} KB  (3 index maps)")
print(f"shared    indices:   {p_sh:.1f} dB   total(4-bit) = {total(idx_sh):.1f} KB  (1 index map)")
print(f"\nindex map cost: {np.ceil(npix/8)/1e3:.1f} KB each; sharing saves {2*np.ceil(npix/8)/1e3:.1f} KB at a {p_pc-p_sh:.2f} dB fidelity cost")
