import sys; sys.path.insert(0, "/home/claude/work")
"""Does the picture change for a HARD shadow (small light -> sharp penumbra)? Reuse the same oracle but shrink the
light, so the penumbra is thin -- narrower than the kernel. Prediction: a big-sigma isotropic kernel now OVER-blurs
the sharp edge and loses, while a small-sigma kernel or the anisotropic (narrow-across-the-edge) kernel recovers it.
That would pin the real breakeven: it's penumbra-sharpness vs kernel-bandwidth, not occluder count."""
import numpy as np
import tools.multidim_shadow_probe as P
import tools.aniso_router_probe as A

SMALL = 0.12                        # a small area light -> sharp penumbra (was 0.8, very soft)
NIMG = 24
occ = A.make_occluders("one_big")

def visib(coords):                  # visibility oracle is light-size agnostic (u,v are passed in), reuse as-is
    return P.visibility(coords, occ)

def gt():
    g = (np.arange(NIMG)+0.5)/NIMG*(2*P.FLOOR_EXTENT)-P.FLOOR_EXTENT
    lg = (np.arange(20)+0.5)/20*(2*SMALL)-SMALL
    U,V = np.meshgrid(lg,lg,indexing="ij"); PX,PZ=np.meshgrid(g,g,indexing="ij")
    out=np.zeros((NIMG,NIMG))
    for i in range(NIMG):
        for j in range(NIMG):
            c=np.stack([np.full(U.size,PX[i,j]),np.full(U.size,PZ[i,j]),U.ravel(),V.ravel()],1)
            out[i,j]=visib(c).mean()
    return out

def samples(B,seed):
    rng=np.random.default_rng(seed)
    x=rng.uniform(-P.FLOOR_EXTENT,P.FLOOR_EXTENT,(B,2)); l=rng.uniform(-SMALL,SMALL,(B,2))
    pts=np.concatenate([x,l],1); return pts, visib(pts)

def uniform(B,seed):
    rng=np.random.default_rng(seed); N=max(1,B//(NIMG*NIMG))
    g=(np.arange(NIMG)+0.5)/NIMG*(2*P.FLOOR_EXTENT)-P.FLOOR_EXTENT; PX,PZ=np.meshgrid(g,g,indexing="ij")
    out=np.zeros((NIMG,NIMG))
    for i in range(NIMG):
        for j in range(NIMG):
            l=rng.uniform(-SMALL,SMALL,(N,2))
            c=np.stack([np.full(N,PX[i,j]),np.full(N,PZ[i,j]),l[:,0],l[:,1]],1); out[i,j]=visib(c).mean()
    return out

G=gt(); B=NIMG*NIMG*12
def err(s): return float(np.abs(s-G).mean())
pts,vis=samples(B,1)
u=uniform(B,0)
print(f"HARD shadow (light half={SMALL}, penumbra ~ sharp):")
print(f"  uniform            {err(u):.4f}")
for sig in (0.18,0.10,0.06):
    print(f"  shared  sigma={sig:<4}  {err(A.reconstruct_isotropic(pts,vis,NIMG,sig)):.4f}")
print(f"  aniso   sigma=0.18  {err(A.reconstruct_anisotropic(pts,vis,NIMG,0.18)):.4f}")
