# This one-off research script lives in archive/; put the library at the repo
# root on the path so its imports keep working when run from anywhere.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
from holographic.io_and_interop.holographic_image import HolographicPlate

N=40; npix=N*N
def corr(a,b): return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
def box_resize(img,n):
    h,w=img.shape; ys=np.linspace(0,h,n+1).astype(int); xs=np.linspace(0,w,n+1).astype(int)
    return np.array([[img[ys[i]:ys[i+1],xs[j]:xs[j+1]].mean() for j in range(n)] for i in range(n)])

# --- test images, same size ---
yy,xx=np.mgrid[0:N,0:N]; c=N/2
def disc(cx,cy,r): return ((xx-cx)**2+(yy-cy)**2)<r*r
smiley=np.zeros((N,N)); smiley[disc(c,c,0.42*N)]=0.4
smiley[disc(c-6,c-5,2.2)]=1; smiley[disc(c+6,c-5,2.2)]=1
m=(np.hypot(xx-c,yy-c+1)<0.27*N)&(np.hypot(xx-c,yy-c+1)>0.19*N)&(yy>c+1); smiley[m]=1
# asymmetric letter "R" (clearly not symmetric)
R=np.zeros((N,N)); R[6:34,10:14]=1; R[6:10,14:26]=1; R[18:22,14:26]=1
R[6:22,26:30]=1; 
for k in range(12): R[22+k,16+k:20+k]=1     # the diagonal leg -> asymmetric
gradient=(xx/N)                              # fully asymmetric dense ramp
photo=box_resize(load_sample_images().images[1][...,:3].mean(2),N); photo/=photo.max()
noise=np.random.default_rng(0).random((N,N))
# scrambled smiley: SAME intensities, positions permuted -> zero symmetry, identical energy
perm=np.random.default_rng(1).permutation(npix)
scram=smiley.ravel()[perm].reshape(N,N)

def recon_corr(img,D,seed=0):
    p=HolographicPlate((N,N),dim=D,seed=seed).store(img,"x")
    return corr(img,p.reconstruct("x"))

# ===== Experiment 1: controlled symmetry test (fixed D, fixed energy) =====
D0=8192
print("=== Exp 1: does breaking symmetry hurt? (D=%d) ==="%D0)
for name,im in [("smiley (symmetric)",smiley),("R (asymmetric)",R),
                ("smiley SCRAMBLED (no symmetry, same energy)",scram),
                ("gradient (asymmetric, dense)",gradient)]:
    cs=[recon_corr(im,D0,s) for s in range(5)]
    print(f"  {name:46s} energy={np.sum(im**2):6.1f}  recon_corr={np.mean(cs):.3f}±{np.std(cs):.3f}")

# ===== Experiment 2: what actually governs fidelity? SNR law =====
print("\n=== Exp 2: noise std vs sqrt(E/D) prediction ===")
for D in [2048,8192,32768]:
    im=noise
    p=HolographicPlate((N,N),dim=D,seed=0).store(im,"x"); rec=p.reconstruct("x")
    resid=(rec-im); E=np.sum(im**2)
    print(f"  D={D:6d}  measured_noise_std={resid.std():.4f}  predicted_sqrt(E/D)={np.sqrt(E/D):.4f}")

# ===== Experiment 3: corr vs D for different image COMPLEXITY (not symmetry) =====
Ds=[1024,2048,4096,8192,16384,32768]
series={"smiley (sym, sparse)":smiley,"R (asym, sparse)":R,
        "photo (asym, complex)":photo,"noise (asym, max complex)":noise}
curves={k:[np.mean([recon_corr(im,D,s) for s in range(3)]) for D in Ds] for k,im in series.items()}

fig=plt.figure(figsize=(14,7))
gs=fig.add_gridspec(2,4)
# top: images and their reconstructions at D=8192
ax=fig.add_subplot(gs[0,0]); ax.imshow(R,cmap="magma"); ax.set_title("asymmetric 'R'\noriginal"); ax.axis("off")
p=HolographicPlate((N,N),dim=8192,seed=0).store(R,"x")
ax=fig.add_subplot(gs[0,1]); ax.imshow(p.reconstruct("x"),cmap="magma"); ax.set_title(f"recon (corr={corr(R,p.reconstruct('x')):.2f})"); ax.axis("off")
ax=fig.add_subplot(gs[0,2]); ax.imshow(photo,cmap="magma"); ax.set_title("natural photo\noriginal"); ax.axis("off")
p2=HolographicPlate((N,N),dim=8192,seed=0).store(photo,"x")
ax=fig.add_subplot(gs[0,3]); ax.imshow(p2.reconstruct("x"),cmap="magma"); ax.set_title(f"recon (corr={corr(photo,p2.reconstruct('x')):.2f})"); ax.axis("off")
# bottom: corr-vs-D curves
axc=fig.add_subplot(gs[1,:])
for k,v in curves.items(): axc.plot(Ds,v,"-o",label=k)
axc.set_xscale("log"); axc.set_xlabel("hologram dimension D"); axc.set_ylabel("reconstruction corr")
axc.set_title("Fidelity is governed by image COMPLEXITY and D — symmetric and asymmetric sparse images overlap")
axc.legend(); axc.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("exp_critique.png",dpi=105,bbox_inches="tight"); plt.close(fig)
print("\nrendered exp_critique.png")
