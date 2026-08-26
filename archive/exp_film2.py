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
def coeffs(K):
    idx=[];val=[]
    for c in range(3):
        f=dct2(img[...,c]).ravel(); i=np.argpartition(np.abs(f),-K)[-K:]; idx.append(i); val.append(f[i])
    return idx,val
def quantize(p, bits, dither, rng):
    lo,hi=p.min(),p.max(); step=(hi-lo)/(2**bits-1); x=(p-lo)/step
    if dither: x=x+(rng.random(x.shape)-0.5)         # film-grain: stochastic (unbiased) rounding
    return np.clip(np.round(x),0,2**bits-1)*step+lo
def recon(keys, plates_q, idx):
    out=[]
    for c in range(3):
        v=keys.adjoint(plates_q[c]); f=np.zeros(npix); f[idx[c]]=v; out.append(idct2(f.reshape(S,S)))
    return np.clip(np.stack(out,-1),0,1)

K,D=2000,16384; idx,val=coeffs(K); keys=WHTKeys(K,D,0); plates=[keys.apply(v) for v in val]
rng=np.random.default_rng(0)
pf=_psnr(img, recon(keys, plates, idx))
print(f"=== bits-per-grain sweep ({S}x{S}, K={K}, D={D}); float plate = {pf:.1f} dB ===")
print(f"{'bits':>4} {'deterministic':>14} {'dithered(film)':>15} {'plate size':>11} {'vs float64':>11}")
for b in [1,2,3,4,6,8]:
    det=_psnr(img, recon(keys,[quantize(p,b,False,rng) for p in plates],idx))
    dit=_psnr(img, recon(keys,[quantize(p,b,True,rng) for p in plates],idx))
    sz=3*b*D/8/1e3
    print(f"{b:>4} {det:12.1f}dB {dit:13.1f}dB {sz:8.1f}KB {3*D*8/1e3/sz:8.0f}x")

# ---- montage: top = "film developing" (1-bit, more grains); bottom = bits at fixed D ----
fig,ax=plt.subplots(2,5,figsize=(16,6.6))
ax[0,0].imshow(img); ax[0,0].set_title("original"); ax[0,0].axis("off")
for j,Dd in enumerate([4096,8192,32768,65536]):
    k2=WHTKeys(K,Dd,0); pl=[k2.apply(v) for v in val]; gr=[np.sign(p) for p in pl]
    o=[]
    for c in range(3):
        v=k2.adjoint(gr[c].astype(float)); v*=np.linalg.norm(val[c])/(np.linalg.norm(v)+1e-12)
        f=np.zeros(npix); f[idx[c]]=v; o.append(idct2(f.reshape(S,S)))
    r=np.clip(np.stack(o,-1),0,1)
    ax[0,j+1].imshow(r); ax[0,j+1].set_title(f"1-bit, {Dd} grains\n{_psnr(img,r):.1f} dB"); ax[0,j+1].axis("off")
ax[1,0].imshow(img); ax[1,0].set_title("original"); ax[1,0].axis("off")
for j,b in enumerate([1,2,3,4]):
    r=recon(keys,[quantize(p,b,True,rng) for p in plates],idx)
    ax[1,j+1].imshow(r); ax[1,j+1].set_title(f"{b}-bit dithered, D={D}\n{_psnr(img,r):.1f} dB"); ax[1,j+1].axis("off")
fig.suptitle("Film-grain holographic storage:  top = image developing as grain count rises (1-bit);  bottom = bit-depth at fixed grains",y=1.0)
fig.tight_layout(); fig.savefig("film_grain.png",dpi=110,bbox_inches="tight"); plt.close(fig)
print("rendered film_grain.png")
