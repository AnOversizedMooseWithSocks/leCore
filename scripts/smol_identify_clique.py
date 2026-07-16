#!/usr/bin/env python3
"""smol_identify_clique.py -- what ARE the ~26 mutually-duplicate embedding rows?

The follow-up found ONE CLIQUE of ~26 rows (not 432 independent duplicates: 432 mutual pairs is
C(30,2)=435). They have NORMAL norms (1.06x the median) and ids from 1 to 48,047 with median 103.
Untrained rows would have LOW norms, so that hypothesis is out. The remaining candidate: SPECIAL /
RESERVED tokens that never appear as training targets. With tied embeddings, such rows receive only
the softmax's negative gradient -- identical for all of them -- so they drift to a common direction.

Prints the actual token strings. Two lines of stdlib JSON, no dependency.

USAGE:  python3 smol_identify_clique.py model.safetensors tokenizer.json
"""
import sys, json, struct, re
import numpy as np

def read_st(p):
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        return {k:v for k,v in json.loads(f.read(n).decode()).items() if k!='__metadata__'}, 8+n

def load(p,meta,base,name):
    m=meta[name]; dt,sh,(b,e)=m['dtype'],m['shape'],m['data_offsets']
    with open(p,'rb') as f:
        f.seek(base+b); raw=f.read(e-b)
    if dt=='BF16': a=(np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
    elif dt=='F16': a=np.frombuffer(raw,dtype=np.float16).astype(np.float32)
    else: a=np.frombuffer(raw,dtype=np.float32)
    return np.ascontiguousarray(a.reshape(sh))

meta,base=read_st(sys.argv[1])
embn=[n for n in meta if re.search(r'embed_tokens|wte|word_embeddings',n)][0]
E=load(sys.argv[1],meta,base,embn).astype(np.float32)
Ec=E-E.mean(0)
En=Ec/(np.linalg.norm(Ec,axis=1,keepdims=True)+1e-12)

best=np.zeros(len(En),np.float32)
for i in range(0,len(En),512):
    C=np.abs(En[i:i+512]@En.T)
    for r in range(C.shape[0]): C[r,i+r]=0
    best[i:i+512]=C.max(1)
ids=np.where(best>0.98)[0]
print(f"clique rows (centered, cos>0.98): {len(ids)}")

vocab=None
try:
    tj=json.load(open(sys.argv[2],encoding='utf-8'))
    vocab={v:k for k,v in tj['model']['vocab'].items()}
    for t in tj.get('added_tokens',[]): vocab[t['id']]=t['content']
except Exception as e:
    print(f"(tokenizer not readable: {e})")

norms=np.linalg.norm(E,axis=1)
print(f"\n  {'id':>7s} {'norm':>7s} {'best-cos':>9s}  token")
for i in ids[:40]:
    tok = vocab.get(int(i),'?') if vocab else '?'
    print(f"  {int(i):7d} {norms[i]:7.3f} {best[i]:9.3f}  {tok!r}")
print(f"\n  If these are <|reserved|>/<|special|>/unused tokens: the 'redundancy' is DEAD VOCABULARY.")
print(f"  Dropping all {len(ids)} rows saves {len(ids)*E.shape[1]/1e6:.3f} MB at q8 -- {len(ids)/len(E):.2%} of the table.")
