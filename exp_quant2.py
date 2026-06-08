import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from holographic_image import HolographicImage, _demo_image, _psnr, _cg

img=_demo_image(240); S=240; M=None
hi=HolographicImage(img.shape, keep=4000, dim=16384, seed=0).store(img)
M=hi._M[S]; D=hi.dim
def q_lloyd(x,bits,iters=30):
    L=2**bits; c=np.quantile(x,(np.arange(L)+0.5)/L)
    for _ in range(iters):
        idx=np.argmin(np.abs(x[:,None]-c[None,:]),1)
        for k in range(L):
            m=idx==k
            if m.any(): c[k]=x[m].mean()
    idx=np.argmin(np.abs(x[:,None]-c[None,:]),1); return c[idx]
def decode(plates, mask, lam=1e-3, iters=200):
    npix=S*S; m=np.ones(D) if mask is None else mask; out=[]
    for i,p in zip(hi._idx, plates):
        b=hi._adjoint(m*p); v=_cg(lambda x: hi._adjoint(m*hi._apply(x))+lam*x, b, iters)
        f=np.zeros(npix); f[i]=v; out.append(M.T@f.reshape(S,S)@M)
    return np.clip(np.stack(out,-1),0,1)

pl_float=hi._plates
pl_3=[q_lloyd(p,3) for p in pl_float]
pl_4=[q_lloyd(p,4) for p in pl_float]
def rand_mask(f,seed=1):
    rng=np.random.default_rng(seed); k=np.ones(D); k[rng.permutation(D)[:int(D*f)]]=0; return k

print("=== robustness survives quantization?  PSNR vs % plate destroyed ===")
fr=[0,0.2,0.4,0.6,0.7]
print(f"{'damage':>7} {'float':>8} {'4-bit':>8} {'3-bit':>8}")
curves={'float':[],'4bit':[],'3bit':[]}
for f in fr:
    mk=None if f==0 else rand_mask(f)
    a=_psnr(img,decode(pl_float,mk)); b=_psnr(img,decode(pl_4,mk)); c=_psnr(img,decode(pl_3,mk))
    curves['float'].append(a); curves['4bit'].append(b); curves['3bit'].append(c)
    print(f"{int(f*100):6d}% {a:7.1f} {b:7.1f} {c:7.1f}")

# contiguous "scratch" vs random erasure at 40%
sc=np.ones(D); sc[:int(D*0.4)]=0
p_scratch=_psnr(img,decode(pl_4,sc)); p_random=_psnr(img,decode(pl_4,rand_mask(0.4)))
print(f"\n40% destroyed (4-bit):  random={p_random:.1f} dB   contiguous scratch={p_scratch:.1f} dB")

fig,ax=plt.subplots(1,2,figsize=(12,4))
for k,mk in [('float','-o'),('4bit','-s'),('3bit','-^')]:
    ax[0].plot([x*100 for x in fr],curves[k],mk,label=k)
ax[0].set_xlabel("% plate destroyed"); ax[0].set_ylabel("PSNR dB")
ax[0].set_title("Damage tolerance survives quantization"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].imshow(decode(pl_4,sc)); ax[1].set_title(f"4-bit plate + 40% contiguous scratch\n{p_scratch:.1f} dB"); ax[1].axis("off")
fig.tight_layout(); fig.savefig("quant_robust.png",dpi=110,bbox_inches="tight"); plt.close(fig)
print("rendered quant_robust.png")
