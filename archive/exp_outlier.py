# This one-off research script lives in archive/; put the library at the repo
# root on the path so its imports keep working when run from anywhere.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from holographic.io_and_interop.holographic_image import HolographicImage, _demo_image, _psnr, _lloyd_max

img=_demo_image(240); S=240
hi=HolographicImage(img.shape,keep=4000,dim=16384,seed=0).store(img)
plates=hi._plates; D=hi.dim; M=hi._M[S]
def decode(plates_q):
    out=[]
    for i,p in zip(hi._idx,plates_q):
        v=hi._adjoint(p); f=np.zeros(S*S); f[i]=v; out.append(M.T@f.reshape(S,S)@M)
    return np.clip(np.stack(out,-1),0,1)

# distribution check: is the WHT plate actually outlier-free Gaussian?
p=plates[0]; kurt=((p-p.mean())**4).mean()/p.var()**2
print(f"plate excess kurtosis = {kurt-3:.2f} (0 = Gaussian; >0 = heavy tails/outliers)\n")

def plain(bits):
    pl=[ _lloyd_max(p,bits)[0] for p in plates]; pc=[_lloyd_max(p,bits)[1] for p in plates]
    dq=[c[idx] for idx,c in zip(pl,pc)]
    byts=3*(bits*D/8 + 2**bits*8)
    return _psnr(img,decode(dq)), byts

def outlier(b_bulk, tau, out_bits=8):
    dq=[]; byts=0
    for p in plates:
        a=np.abs(p); thr=np.quantile(a,1-tau); om=a>=thr
        q=np.empty_like(p)
        ib,cb=_lloyd_max(p[~om],b_bulk); q[~om]=cb[ib]
        io,co=_lloyd_max(p[om],out_bits); q[om]=co[io]
        dq.append(q)
        nb=(~om).sum(); no=om.sum()
        byts += nb*b_bulk/8 + no*(out_bits/8 + 2) + (2**b_bulk+2**out_bits)*8   # codes+positions+codebooks
    return _psnr(img,decode(dq)), byts

print(f"{'scheme':>22} {'PSNR':>7} {'bytes':>8}")
for b in [2,3,4]:
    ps,by=plain(b); print(f"{'Lloyd '+str(b)+'-bit':>22} {ps:6.1f}dB {by/1e3:6.1f}KB")
for tau in [0.02,0.05,0.10]:
    ps,by=outlier(2,tau); print(f"{'bulk2 + top'+str(int(tau*100))+'%@8':>22} {ps:6.1f}dB {by/1e3:6.1f}KB")
    ps,by=outlier(3,tau); print(f"{'bulk3 + top'+str(int(tau*100))+'%@8':>22} {ps:6.1f}dB {by/1e3:6.1f}KB")
