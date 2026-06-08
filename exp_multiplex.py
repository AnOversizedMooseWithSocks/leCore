import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from holographic_image import _fwht, _dct_matrix, _psnr, _cg

S=128; npix=S*S; M=_dct_matrix(S)
def dct2(a): return M@a@M.T
def idct2(C): return M.T@C@M
def make_img(i):
    rng=np.random.default_rng(100+i); yy,xx=np.mgrid[0:S,0:S]/S; g=np.zeros((S,S))
    for _ in range(4): g+=rng.uniform(.3,1)*np.sin(2*np.pi*(rng.integers(1,5)*xx+rng.integers(1,5)*yy)+rng.uniform(0,6))
    g=(g-g.min())/np.ptp(g); cx,cy=rng.uniform(.3,.7,2); r=.18
    g[(((xx-cx)**2+(yy-cy)**2)<r**2) if i%2 else ((np.abs(xx-cx)+np.abs(yy-cy))<r)] = (1.0 if i%3 else 0.0)
    return g

K,D=1500,16384
rng=np.random.default_rng(0); signs=rng.choice([-1.,1.],D); perm=rng.permutation(D); sc=1/np.sqrt(D)
def pos(n): return perm[n*K:(n+1)*K]
def apply_n(n,v): x=np.zeros(D); x[pos(n)]=v; return _fwht(x*signs)*sc
def adj_n(n,y):   return (_fwht(y)*sc*signs)[pos(n)]

imgs=[make_img(i) for i in range(10)]
idxs=[]; coeffs=[]
for im in imgs:
    f=dct2(im).ravel(); i=np.argpartition(np.abs(f),-K)[-K:]; idxs.append(i); coeffs.append(f[i])

def recover(N, mask=None, lam=1e-3, iters=300):
    H=sum(apply_n(n,coeffs[n]) for n in range(N))
    if mask is None:
        Vs=[adj_n(n,H) for n in range(N)]                       # exact, per-image
    else:                                                       # JOINT masked recovery
        def app(V): return sum(apply_n(n,V[n*K:(n+1)*K]) for n in range(N))
        def adj(y): return np.concatenate([adj_n(n,y) for n in range(N)])
        Vf=_cg(lambda V: adj(mask*app(V))+lam*V, adj(mask*H), iters)
        Vs=[Vf[n*K:(n+1)*K] for n in range(N)]
    out=[]
    for n in range(N):
        f=np.zeros(npix); f[idxs[n]]=Vs[n]; out.append(np.clip(idct2(f.reshape(S,S)),0,1))
    return out

print(f"=== {D//K} distinct images in ONE plate (K={K}, D={D}) ===")
for N in [1,2,4,8,10]:
    r=recover(N); print(f"  N={N:2d} (load {N*K/D:.2f})  mean PSNR/image = {np.mean([_psnr(imgs[n],r[n]) for n in range(N)]):.1f} dB")

# graceful degradation of a multiplexed plate (N=4, load 0.37 -> cliff at ~63% destroyed)
print("\nN=4 multiplexed, destroying the shared plate (joint recovery):")
def mask_frac(f): m=np.ones(D); m[rng.permutation(D)[:int(D*f)]]=0; return m
for f in [0.0,0.3,0.5,0.7]:
    r=recover(4, None if f==0 else mask_frac(f))
    print(f"  {int(f*100):2d}% destroyed: mean PSNR {np.mean([_psnr(imgs[n],r[n]) for n in range(4)]):.1f} dB")

N=4; r=recover(N)
fig,ax=plt.subplots(2,N,figsize=(3*N,6))
for n in range(N):
    ax[0,n].imshow(imgs[n],cmap='magma'); ax[0,n].set_title(f"original {n+1}"); ax[0,n].axis("off")
    ax[1,n].imshow(r[n],cmap='magma'); ax[1,n].set_title(f"recovered\n{_psnr(imgs[n],r[n]):.1f} dB"); ax[1,n].axis("off")
fig.suptitle("Four distinct images superposed in ONE holographic plate, each recovered cleanly",y=1.0)
fig.tight_layout(); fig.savefig("multiplex.png",dpi=110,bbox_inches="tight"); plt.close(fig)
print("rendered multiplex.png")
