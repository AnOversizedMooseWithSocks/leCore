import numpy as np
from holographic_image import _fwht, _dct_matrix, _psnr, _cg

def box(img, n):                       # box-resize (works for HxW or HxWxC)
    h,w = img.shape[:2]; ys=np.linspace(0,h,n+1).astype(int); xs=np.linspace(0,w,n+1).astype(int)
    if img.ndim==3:
        return np.stack([np.array([[img[ys[i]:ys[i+1],xs[j]:xs[j+1],c].mean() for j in range(n)]
                for i in range(n)]) for c in range(3)],-1)
    return np.array([[img[ys[i]:ys[i+1],xs[j]:xs[j+1]].mean() for j in range(n)] for i in range(n)])

def gallery(S=128):
    yy,xx = np.mgrid[0:S,0:S]/S; imgs=[]
    g=np.zeros((S,S,3)); h=S//2                          # 1: quadrants
    g[:h,:h]=[1,0,0]; g[:h,h:]=[0,0,1]; g[h:,:h]=[1,1,0]; g[h:,h:]=[0,1,0]; imgs.append(g)
    g=np.zeros((S,S,3))                                  # 2: horizontal RGB bands
    g[:S//3]=[1,.1,.1]; g[S//3:2*S//3]=[.1,1,.1]; g[2*S//3:]=[.1,.1,1]; imgs.append(g)
    g=np.stack([xx,1-xx,np.abs(yy-.5)*2],-1); imgs.append(g)            # 3: gradient
    r=np.sqrt((xx-.5)**2+(yy-.5)**2); g=np.stack([1-r,r,np.abs(.5-r)*2],-1); imgs.append(np.clip(g,0,1))  # 4: radial
    g=np.zeros((S,S,3)); g[...,0]=(np.sin(xx*12)*.5+.5); g[...,1]=(np.cos(yy*12)*.5+.5); g[...,2]=.5; imgs.append(g) # 5: ripples
    g=((np.floor(xx*4)+np.floor(yy*4))%2)[...,None]*np.array([1,1,1.]); g=g*np.array([1,.4,.7]); imgs.append(g) # 6: checker
    return [np.clip(im,0,1) for im in imgs]

class Archive:
    def __init__(s,S,K,D,seed=0):
        s.S,s.K,s.D=S,K,D; rng=np.random.default_rng(seed)
        s.signs=rng.choice([-1.,1.],D); s.perm=rng.permutation(D); s.sc=1/np.sqrt(D)
        s.M=_dct_matrix(S); s.plates=[np.zeros(D) for _ in range(3)]; s.idx=[]; s.fps=[]; s.N=0
    def _pos(s,i): return s.perm[i*s.K:(i+1)*s.K]
    def _ap(s,i,v): x=np.zeros(s.D); x[s._pos(i)]=v; return _fwht(x*s.signs)*s.sc
    def _aj(s,i,y): return (_fwht(y)*s.sc*s.signs)[s._pos(i)]
    def _fp(s,img): t=box(img,12).ravel(); return t/ (np.linalg.norm(t)+1e-9)
    def add(s,img):
        i=s.N; ic=[]
        for c in range(3):
            f=(s.M@img[...,c]@s.M.T).ravel(); k=np.argpartition(np.abs(f),-s.K)[-s.K:]
            ic.append(k); s.plates[c]+=s._ap(i,f[k])
        s.idx.append(ic); s.fps.append(s._fp(img)); s.N+=1
    def _joint(s,c,mask):
        N=s.N
        def app(V): return sum(s._ap(n,V[n*s.K:(n+1)*s.K]) for n in range(N))
        def adj(y): return np.concatenate([s._aj(n,y) for n in range(N)])
        Vf=_cg(lambda V: adj(mask*app(V))+1e-3*V, adj(mask*s.plates[c]),250)
        return [Vf[n*s.K:(n+1)*s.K] for n in range(N)]
    def recover(s,i,mask=None):
        out=[]; joints=[s._joint(c,mask) for c in range(3)] if mask is not None else None
        for c in range(3):
            v = s._aj(i,s.plates[c]) if mask is None else joints[c][i]
            f=np.zeros(s.S*s.S); f[s.idx[i][c]]=v; out.append(s.M.T@f.reshape(s.S,s.S)@s.M)
        return np.clip(np.stack(out,-1),0,1)
    def recall(s,q,mask=None):
        fq=s._fp(q); i=int(np.argmax([fq@fp for fp in s.fps])); return i, s.recover(i,mask)

S,K,D=128,2000,32768
imgs=gallery(S); A=Archive(S,K,D,0)
for im in imgs: A.add(im)
print(f"stored {A.N} colour images in 3 plates (one per channel), D={D}")
# clean recovery fidelity
print("clean recover PSNR:", [f"{_psnr(imgs[i],A.recover(i)):.0f}" for i in range(A.N)])
# content-addressable recall under degradations
def noisy(im): return np.clip(im+0.5*np.random.default_rng(1).standard_normal(im.shape),0,1)
def blur(im):  return box(box(im,16),S)
def occl(im):  g=im.copy(); g[20:80,20:80]=0; return g
for name,deg in [("noise",noisy),("blur",blur),("occlude",occl)]:
    hits=sum(A.recall(deg(imgs[i]))[0]==i for i in range(A.N))
    print(f"recall from {name:8s}: {hits}/{A.N} correct")

print("\n--- fixed blur + damage-tolerant recall ---")
def blur2(im): sm=box(im,16); return np.repeat(np.repeat(sm,S//16,0),S//16,1)
hits=sum(A.recall(blur2(imgs[i]))[0]==i for i in range(A.N))
print(f"recall from blur (fixed): {hits}/{A.N} correct")
# destroy 40% of every plate, then recall from a noisy query and reconstruct
rng=np.random.default_rng(7); mask=np.ones(D); mask[rng.permutation(D)[:int(D*0.4)]]=0
ok=0; ps=[]
for i in range(A.N):
    j,rec=A.recall(noisy(imgs[i]), mask=mask); ok+=(j==i); ps.append(_psnr(imgs[i],rec))
print(f"recall from noisy query with 40% of plates destroyed: {ok}/{A.N} correct, recon PSNR {np.mean(ps):.1f} dB")
