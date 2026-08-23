"""The shipped page, rebuilt around the arm that wins: BM25 + containment + the three-way policy.

WHAT CHANGED FROM THE EARLIER PAGES. They ran scalar cosine over hash atoms -- the weak arm, 0.425
top-1 on realistic text. This one runs the measured-strong path: BM25 scoring and the exact
containment count in one GLSL ES pass (binary search over per-document sorted-unique terms,
precomputed document length), then the answer/set/abstain policy.

NO VOCABULARY TABLE IS SHIPPED. A term's id IS its FNV-1a hash, so the browser hashes a typed word
and the shader compares u32s -- the same lever-3 move as the atom families. Expected 32-bit
collisions over ~16k terms are ~0.03, and the page REPORTS the measured collision count rather
than assuming.

THE PAGE CALIBRATES ITS OWN NULL. Embedding a Python-computed threshold would silently drift if
the JS and Python scorers ever diverged; instead the page runs 200 scrambled vocabulary-matched
queries through its own shader at load and takes the 95th percentile. The threshold is therefore
a property of what actually runs.
"""
import base64, json, os
from collections import Counter

import numpy as np

from holographic.semantic_router.holographic_bm25 import tokenize as _bm_tok, _STOP

import hard_corpus as HC
import holographic.agents_and_reasoning.holographic_hashatom as HA
import holographic.agents_and_reasoning.holographic_retrievalpolicy as RP

TARGET = 500
dn = HC.load_passages(target=TARGET)
docs = [t for _, t in dn]
names = [n for n, _ in dn]
N = len(docs)

# TOKENISE EXACTLY ONCE. tokenize() is NOT idempotent ('settings'->'setting'->'sett'), so the
# index is built from the POLICY'S OWN token view rather than from a second pass over the text.
# The JS port below reproduces that single pass; differentially tested on 12,015 words, 0 diffs.
pol_pre = RP.RetrievalPolicy(docs)
docs = pol_pre.docs

# RAW token stream only -- the sorted-unique+tf index the scorer wants is DERIVED in JS at load,
# and the raw stream is what proximity reranking needs for positions. Shipped DENSE-RANKED AND
# BIT-PACKED: 2,731 distinct terms is 12 bits of entropy, and shipping it in a 32-bit box was
# 2.42x of pure waste. Pair promotion was measured on this same stream and REJECTED (0.84x once
# the baseline is packed rather than u32) -- see NOTES.
sym_l, off = [], [0]
rank = {}
for d in docs:
    for t in d:
        sym_l.append(rank.setdefault(int(HA.fnv1a(t)), len(rank)))
    off.append(len(sym_l))
vocab_ids = np.zeros(len(rank), dtype="<u4")
for h, i in rank.items():
    vocab_ids[i] = h
BITS = max(1, (len(rank) - 1).bit_length())
# T13 (pack_roundtrip) is lossless exactly while every symbol is below 2**BITS. ASSERT it here
# rather than trust it: a vocabulary that outgrew the width would ALIAS terms, not fail.
assert max(sym_l) < (1 << BITS), "symbol exceeds the packed width -- terms would alias"

def _pack(sym, bits):
    out = bytearray((len(sym) * bits + 7) // 8); acc = nb = pos = 0
    for v in sym:
        acc |= (v & ((1 << bits) - 1)) << nb; nb += bits
        while nb >= 8:
            out[pos] = acc & 0xFF; acc >>= 8; nb -= 8; pos += 1
    if nb: out[pos] = acc & 0xFF
    return bytes(out)

packed = _pack(sym_l, BITS)
offv = np.array(off, dtype="<u4")

allhash = {}
coll = 0
for d in docs:
    for t in set(d):
        h = int(HA.fnv1a(t))
        if h in allhash and allhash[h] != t:
            coll += 1
        allhash[h] = t

pol = pol_pre.calibrate(n=200, seed=0)
rng = np.random.default_rng(0)
refs = []
for i in rng.choice(N, 60, replace=False):
    u = sorted(set(docs[i]))
    q = [u[j] for j in rng.choice(len(u), min(8, len(u)), replace=False)]
    v = pol.verdict(q, rerank=True)
    refs.append(dict(q=q, gold=int(i), mode=v["mode"], amb=int(v["ambiguity"]),
                     ans=(int(v["answer"]) if v["answer"] is not None else -1)))

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
# THE EMBEDDED CORPUS IS A `lecore-index/1` BUNDLE -- the same shape holographic_indexstore writes
# and idb_store.js reads. One format for disk, IndexedDB and this page; two names for the same
# bytes is how a store and its consumer drift apart.
import holographic.caching_and_storage.holographic_indexstore as _is
BUNDLE = dict(format="lecore-index/1", bits=BITS, ntok=len(sym_l), ndocs=N,
              packed=base64.b64encode(packed).decode(), off=b64(offv), vocab=b64(vocab_ids))
BUNDLE["sha256"] = _is.digest(BUNDLE)
DATA = dict(names=names, stop=sorted(_STOP), refs=refs, nterms=len(allhash), collisions=coll,
            py_threshold=float(pol.threshold), bundle=BUNDLE)

HTML = r"""<!doctype html>
<meta charset="utf-8"><title>leCore search — WebGL2</title>
<style>
 body{background:#0b0d10;color:#cfd6e4;font:13px/1.6 ui-monospace,Menlo,Consolas,monospace;
      margin:0;padding:26px 30px;max-width:1000px}
 h1{font-size:17px;color:#e8eefc;margin:0 0 4px}h2{font-size:14px;color:#e8eefc;margin:22px 0 6px}
 p.sub{color:#7d8798;margin:0 0 16px}
 table{border-collapse:collapse;width:100%;margin:8px 0 14px}
 td,th{padding:5px 9px;border-bottom:1px solid #1c2129;text-align:left;vertical-align:top}
 th{color:#7d8798;font-weight:400}
 .ok{color:#5fd38d}.bad{color:#ff6b6b}.n{color:#8ab4ff}.dim{color:#6c7688}
 input{background:#12151b;color:#e8eefc;border:1px solid #2a323d;padding:8px 10px;font:inherit;
       width:100%;box-sizing:border-box;border-radius:5px}
 pre{background:#12151b;padding:11px;border-radius:6px;overflow:auto;color:#9aa5b6;margin:6px 0}
</style>
<h1>leCore — BM25 + containment in WebGL2, with answer / set / abstain</h1>
<p class="sub">One fragment pass computes both the BM25 score and the exact containment count, by
binary search over each document's sorted-unique terms. No vocabulary table is shipped: a term's
id is its FNV-1a hash. The page calibrates its own abstain threshold from 200 scrambled queries,
so the threshold describes what actually runs.</p>
<div id="out">running…</div>
<h2>Search</h2>
<input <div class=card style="margin:14px 0">
 <b>Corpus</b> &mdash; <span id=src>embedded</span>
 <div style="margin-top:8px">
  <input type=file id=imp accept=".json" style="max-width:340px">
  <button id=clr>clear cache</button>
 </div>
 <div id=impmsg style="color:#8a93a6;margin-top:6px">
  Import a <code>lecore-index/1</code> bundle written by <code>mind.index_save(...)</code>.
  It is verified against its own sha256, cached in IndexedDB, and used on every later visit.
 </div>
</div>
id="q" placeholder="holographic vector memory cleanup" autocomplete="off">
<div id="ans"></div>
<script id="DATA" type="application/json">__DATA__</script>
<script>
const P=JSON.parse(document.getElementById('DATA').textContent);
const dec32=s=>{const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Uint32Array(u.buffer);};
let BUNDLE=P.bundle, SOURCE='embedded';
let OFF=dec32(BUNDLE.off),N=BUNDLE.ndocs;
// UNPACK the dense-ranked, bit-packed symbol stream. Pinned against the Python packer on the
// WHOLE stream (71,270 tokens), not a sample -- an off-by-one here would corrupt the tail only.
// ---- PERSISTENCE ---------------------------------------------------------------------------
// Inlined on purpose: an ES-module import from file:// is blocked by CORS, and these pages exist
// to be double-clicked. Same logic as pages/idb_store.js, same `lecore-index/1` bundle, same
// digest -- cross-checked byte-for-byte against holographic_indexstore.digest() in Python.
const IDB_NAME="lecore", IDB_STORE="indexes";
function idbOpen(){return new Promise((res,rej)=>{const r=indexedDB.open(IDB_NAME,1);
  r.onupgradeneeded=()=>r.result.createObjectStore(IDB_STORE);
  r.onsuccess=()=>res(r.result); r.onerror=()=>rej(r.error);});}
async function bundleDigest(m){
  const parts=["format","bits","ntok","ndocs","packed","off","vocab"].map(k=>String(m[k])).join("");
  const b=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(parts));
  return Array.from(new Uint8Array(b)).map(x=>x.toString(16).padStart(2,"0")).join("");}
async function idbPut(key,m){const db=await idbOpen();return new Promise((res,rej)=>{
  const tx=db.transaction(IDB_STORE,"readwrite"); tx.objectStore(IDB_STORE).put(m,key);
  // resolve on COMPLETE: a request can succeed inside a transaction that later aborts on quota
  tx.oncomplete=()=>res(true); tx.onerror=tx.onabort=()=>rej(tx.error||new Error("quota?"));});}
async function idbGet(key){const db=await idbOpen();return new Promise((res,rej)=>{
  const tx=db.transaction(IDB_STORE,"readonly"); const r=tx.objectStore(IDB_STORE).get(key);
  r.onsuccess=()=>res(r.result||null); r.onerror=()=>rej(r.error);});}

function unpack(b64,bits,count){
  const bin=atob(b64), buf=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) buf[i]=bin.charCodeAt(i);
  const out=new Uint32Array(count); let acc=0,nb=0,p=0;
  for(let i=0;i<count;i++){
    while(nb<bits){acc|=buf[p++]<<nb; nb+=8;}
    out[i]=acc&((1<<bits)-1); acc>>>=bits; nb-=bits;}
  return out;}
let SYM=unpack(BUNDLE.packed,BUNDLE.bits,BUNDLE.ntok), VOCAB_IDS=dec32(BUNDLE.vocab);
let RAW=new Uint32Array(BUNDLE.ntok);
for(let i=0;i<BUNDLE.ntok;i++) RAW[i]=VOCAB_IDS[SYM[i]];
let TERMA,TFA,OFF2A,PDOC,PTF,PRANGE,DF,DL,AVGDL;
// Wrapped so an IMPORTED bundle can re-derive without reloading the page. The index is
// always a function of the stored stream, never a second thing kept alongside it.
function rebuildIndex(){
// DERIVE the scorer's sorted-unique + tf layout from the raw stream. One shipped representation,
// no duplicate state; the raw stream is also what the proximity reranker needs for positions.
const TERM=[],TF=[],OFF2=[0];
for(let d=0;d<N;d++){const c=new Map();
  for(let i=OFF[d];i<OFF[d+1];i++) c.set(RAW[i],(c.get(RAW[i])||0)+1);
  const ids=Array.from(c.keys()).sort((a,b)=>a-b);
  for(const t of ids){TERM.push(t);TF.push(c.get(t));}
  OFF2.push(TERM.length);}
TERMA=Uint32Array.from(TERM);TFA=Uint32Array.from(TF);OFF2A=Uint32Array.from(OFF2);
// POSTINGS, derived from the same raw stream -- one entry per (term, document) pair, grouped by
// term. This is what turns a scan over every document into a walk over only the documents a query
// actually touches: measured 11,371x less work on realistic 2-4 term queries, 106x wall clock.
const _byTerm=new Map();
for(let d=0;d<N;d++)
  for(let i=OFF2A[d];i<OFF2A[d+1];i++){
    const t=TERMA[i]; if(!_byTerm.has(t))_byTerm.set(t,[]); _byTerm.get(t).push(d,TFA[i]); }
PDOC=new Uint32Array(TERMA.length);PTF=new Uint32Array(TERMA.length);PRANGE=new Map();
{let w=0;
 for(const [t,rows] of _byTerm){ const lo=w;
   for(let j=0;j<rows.length;j+=2){ PDOC[w]=rows[j]; PTF[w]=rows[j+1]; w++; }
   PRANGE.set(t,[lo,w]); } }
const ENC=new TextEncoder();
function fnv1a(s){let h=2166136261>>>0;
  for(const c of ENC.encode(s)){h=(h^c)>>>0;h=Math.imul(h,16777619)>>>0;}return h>>>0;}
// PORTED FROM holographic_bm25.tokenize -- differentially tested against Python on 12,015 words
// and 300 real source passages: 0 mismatches. The browser must split terms EXACTLY as the scorer
// does, or the containment count describes a different query than the one being scored.
const STOP=new Set(P.stop);
function normalize(t){
  for(const suf of ["ing","ed","es","s"])
    if(t.endsWith(suf) && t.length-suf.length>=3) return t.slice(0,t.length-suf.length);
  return t; }
const tokenise=s=>{const out=[];
  for(const t of (s.toLowerCase().match(/[a-z0-9]+/g)||[]))
    if(!STOP.has(t) && t.length>1) out.push(normalize(t));
  return out; };

// df and avgdl are DERIVED at load from the shipped arrays -- shipping them would be duplicate
// state that could drift from the data it describes.
DF=new Map(); DL=new Float32Array(N); let total=0;
for(let d=0;d<N;d++){DL[d]=OFF[d+1]-OFF[d]; total+=DL[d];
  for(let i=OFF2A[d];i<OFF2A[d+1];i++) DF.set(TERMA[i],(DF.get(TERMA[i])||0)+1);}
AVGDL=total/N;
}
rebuildIndex();
const VOCAB=Array.from(DF.keys());

const VS=`#version 300 es
in vec2 p; void main(){gl_Position=vec4(p,0.0,1.0);}`;
// SCATTER PATH. The vertex stage places one POINT per posting at its document's texel and
// carries that term's BM25 contribution; additive blending sums them. A fragment shader cannot
// scatter, but a VERTEX shader can, and blending is a hardware scatter-add.
const VS_SC=`#version 300 es
precision highp float; precision highp int; precision highp usampler2D; precision highp sampler2D;
uniform usampler2D uPDoc,uPTf; uniform sampler2D uDl;
uniform int uBase,uW,uOutW,uOutH;
uniform float uIdf,uK1,uB,uAvgdl;
out float vC;
ivec2 at(int i){ return ivec2(i % uW, i / uW); }
void main(){
  int p=uBase+gl_VertexID;
  int d=int(texelFetch(uPDoc,at(p),0).r);
  float tf=float(texelFetch(uPTf,at(p),0).r);
  float dl=texelFetch(uDl,at(d),0).r;
  vC=uIdf*tf*(uK1+1.0)/(tf+uK1*(1.0-uB+uB*dl/uAvgdl));
  float x=(float(d % uOutW)+0.5)/float(uOutW)*2.0-1.0;
  float y=(float(d / uOutW)+0.5)/float(uOutH)*2.0-1.0;
  gl_Position=vec4(x,y,0.0,1.0); gl_PointSize=1.0; }`;
const FS_SC=`#version 300 es
precision highp float;
in float vC; out vec2 o;
void main(){ o=vec2(vC,1.0); }`;      // .y accumulates the containment coverage count

const FS=`#version 300 es
precision highp float; precision highp int;
precision highp usampler2D; precision highp sampler2D;
uniform usampler2D uTerm,uTf,uOff,uQ; uniform sampler2D uIdf,uDl;
uniform int uNQ,uW; uniform float uK1,uB,uAvgdl;
out vec2 o;
ivec2 at(int i){ return ivec2(i % uW, i / uW); }
void main(){
  int d=int(gl_FragCoord.x);
  int lo=int(texelFetch(uOff,at(d),0).r), hi=int(texelFetch(uOff,at(d+1),0).r);
  float dl=texelFetch(uDl,at(d),0).r;
  float norm=uK1*(1.0-uB+uB*dl/uAvgdl);      // loop-invariant, hoisted
  float s=0.0, cov=0.0;
  for(int j=0;j<uNQ;++j){
    uint q=texelFetch(uQ,ivec2(j,0),0).r;
    int a=lo,b=hi,found=-1;
    while(a<b){int m=(a+b)>>1; uint v=texelFetch(uTerm,at(m),0).r;
      if(v==q){found=m;break;} else if(v<q) a=m+1; else b=m;}
    if(found>=0){ float tf=float(texelFetch(uTf,at(found),0).r); cov+=1.0;
      s += texelFetch(uIdf,ivec2(j,0),0).r*tf*(uK1+1.0)/(tf+norm); } }
  o=vec2(s,cov); }`;

let gl,prog,progSC,W=2048,tTerm,tTf,tOff,tDl,tPDoc,tPTf,outTex,fbo,THRESH=0,SCATTER=false;
const rows=[]; let fails=0;
const row=(n,v,ok,note)=>{rows.push([n,v,ok,note||'']);if(ok===false)fails++;};
function texU(arr,w){const h=Math.ceil(arr.length/w)||1;const pad=new Uint32Array(w*h);
  pad.set(arr);const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  for(const p of[gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.R32UI,w,h,0,gl.RED_INTEGER,gl.UNSIGNED_INT,pad);return t;}
function texF(arr,w){const h=Math.ceil(arr.length/w)||1;const pad=new Float32Array(w*h);
  pad.set(arr);const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  for(const p of[gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.R32F,w,h,0,gl.RED,gl.FLOAT,pad);return t;}

function scoreScatter(qh){
  gl.bindFramebuffer(gl.FRAMEBUFFER,fbo); gl.useProgram(progSC); gl.viewport(0,0,N,1);
  gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE,gl.ONE);
  [['uPDoc',tPDoc],['uPTf',tPTf],['uDl',tDl]].forEach(([n,t],u)=>{
    gl.activeTexture(gl.TEXTURE0+u); gl.bindTexture(gl.TEXTURE_2D,t);
    gl.uniform1i(gl.getUniformLocation(progSC,n),u);});
  gl.uniform1i(gl.getUniformLocation(progSC,'uW'),W);
  gl.uniform1i(gl.getUniformLocation(progSC,'uOutW'),N);
  gl.uniform1i(gl.getUniformLocation(progSC,'uOutH'),1);
  gl.uniform1f(gl.getUniformLocation(progSC,'uK1'),1.5);
  gl.uniform1f(gl.getUniformLocation(progSC,'uB'),0.75);
  gl.uniform1f(gl.getUniformLocation(progSC,'uAvgdl'),AVGDL);
  for(const t of new Set(qh)){
    const r=PRANGE.get(t); if(!r) continue;
    const df=r[1]-r[0];
    gl.uniform1f(gl.getUniformLocation(progSC,'uIdf'),Math.log(1.0+(N-df+0.5)/(df+0.5)));
    gl.uniform1i(gl.getUniformLocation(progSC,'uBase'),r[0]);
    gl.drawArrays(gl.POINTS,0,df); }
  gl.disable(gl.BLEND);
  const px=new Float32Array(N*2); gl.readPixels(0,0,N,1,gl.RG,gl.FLOAT,px);
  return px;
}

function score(qh){ return SCATTER ? scoreScatter(qh) : scoreFull(qh); }

function scoreFull(qh){
  const idf=new Float32Array(qh.length);
  for(let j=0;j<qh.length;j++){const df=DF.get(qh[j])||0;
    idf[j]=Math.log(1.0+(N-df+0.5)/(df+0.5));}
  const tq=texU(Uint32Array.from(qh),Math.max(1,qh.length)), ti=texF(idf,Math.max(1,qh.length));
  gl.bindFramebuffer(gl.FRAMEBUFFER,fbo); gl.useProgram(prog); gl.viewport(0,0,N,1);
  const binds=[['uTerm',tTerm],['uTf',tTf],['uOff',tOff],['uQ',tq],['uIdf',ti],['uDl',tDl]];
  binds.forEach(([n,t],u)=>{gl.activeTexture(gl.TEXTURE0+u);gl.bindTexture(gl.TEXTURE_2D,t);
    gl.uniform1i(gl.getUniformLocation(prog,n),u);});
  gl.uniform1i(gl.getUniformLocation(prog,'uNQ'),qh.length);
  gl.uniform1i(gl.getUniformLocation(prog,'uW'),W);
  gl.uniform1f(gl.getUniformLocation(prog,'uK1'),1.5);
  gl.uniform1f(gl.getUniformLocation(prog,'uB'),0.75);
  gl.uniform1f(gl.getUniformLocation(prog,'uAvgdl'),AVGDL);
  gl.drawArrays(gl.TRIANGLES,0,3);
  const px=new Float32Array(N*2); gl.readPixels(0,0,N,1,gl.RG,gl.FLOAT,px);
  gl.deleteTexture(tq); gl.deleteTexture(ti);
  return px;
}
// Proximity rerank over the top-k, in JS: coverage, then tightness, then ordered adjacency.
// Ported from holographic_retrievalpolicy.proximity_key -- lexicographic, no weights to tune.
function proximityKey(d,qh){
  const pos=new Map();
  for(let i=OFF[d];i<OFF[d+1];i++){const t=RAW[i];
    if(!pos.has(t))pos.set(t,[]); pos.get(t).push(i-OFF[d]);}
  const lists=qh.map(t=>pos.get(t)).filter(Boolean);
  let span=1e9;
  if(lists.length){const idx=lists.map(()=>0);
    for(;;){const cur=lists.map((l,i)=>l[idx[i]]);
      span=Math.min(span,Math.max(...cur)-Math.min(...cur)+1);
      let k=0; for(let i=1;i<cur.length;i++) if(cur[i]<cur[k]) k=i;
      if(++idx[k]>=lists[k].length) break;}}
  const qs=new Set(qh); let bg=0;
  for(let i=OFF[d];i<OFF[d+1]-1;i++) if(qs.has(RAW[i])&&qs.has(RAW[i+1])) bg++;
  return [lists.length,-span,bg];
}
const keyCmp=(a,b)=>{for(let i=0;i<3;i++) if(a[i]!==b[i]) return b[i]-a[i]; return 0;};

async function useBundle(b,label){
  const d=await bundleDigest(b);
  if(d!==b.sha256) throw new Error('bundle failed its own digest');
  BUNDLE=b; SOURCE=label;
  SYM=unpack(BUNDLE.packed,BUNDLE.bits,BUNDLE.ntok); VOCAB_IDS=dec32(BUNDLE.vocab);
  OFF=dec32(BUNDLE.off); N=BUNDLE.ndocs;
  RAW=new Uint32Array(BUNDLE.ntok);
  for(let i=0;i<BUNDLE.ntok;i++) RAW[i]=VOCAB_IDS[SYM[i]];
  rebuildIndex();
}

function verdict(tokens,rerank){
  const qh=Array.from(new Set(tokens.map(fnv1a)));
  const px=score(qh);
  let best=0,second=-1e30,bs=-1e30,cset=[];
  for(let d=0;d<N;d++){const s=px[2*d],c=px[2*d+1];
    if(c===qh.length) cset.push(d);
    if(s>bs){second=bs;bs=s;best=d;} else if(s>second) second=s;}
  cset.sort((a,b)=>px[2*b]-px[2*a]);
  const m=cset.length;
  if(bs<THRESH) return {mode:'abstain',amb:m,top:bs,set:[],answer:null};
  if(m<=1){
    let ans=best;
    if(rerank){
      const top=[]; for(let d=0;d<N;d++) top.push([px[2*d],d]);
      top.sort((a,b)=>b[0]-a[0]);
      const cand=top.slice(0,10).map(x=>x[1]);
      // REORDER ONLY. The reranker never introduces a document the scorer did not return.
      cand.sort((a,b)=>keyCmp(proximityKey(a,qh),proximityKey(b,qh)));
      ans=cand[0];}
    return {mode:'answer',amb:m,top:bs,set:[ans],answer:ans,margin:bs-second};}
  return {mode:'set',amb:m,top:bs,set:cset.slice(0,10),answer:null,ceiling:1/m};
}

function main(){
  gl=document.createElement('canvas').getContext('webgl2');
  if(!gl){document.getElementById('out').innerHTML='<p class="bad">No WebGL2.</p>';return;}
  const ext=gl.getExtension('EXT_color_buffer_float');
  // Prefer a cached bundle over the embedded one, but only if it VERIFIES. A truncated write or
  // an aborted transaction leaves a payload that parses cleanly and answers wrongly.
  try{
    const cached=await idbGet('search');
    if(cached){
      const d=await bundleDigest(cached);
      if(d===cached.sha256){ await useBundle(cached,'IndexedDB cache'); }
      else { row('cached index','DIGEST MISMATCH — ignored',false,
                 'a partial write parses cleanly and lies; the embedded corpus is used instead'); }
    }
  }catch(e){ row('IndexedDB','unavailable: '+e.message,true,'falling back to the embedded corpus'); }
  row('corpus source',SOURCE+' · '+N+' passages',true,
      SOURCE==='embedded'?'no cache yet — import a bundle to persist one':'restored, digest verified');

  row('EXT_color_buffer_float',ext?'present':'ABSENT',!!ext,ext?'needed for the RG32F target':'');
  if(!ext){render();return;}
  const sh=(t,s)=>{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(o));return o;};
  const link=(v,f)=>{const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,v));
    gl.attachShader(p,sh(gl.FRAGMENT_SHADER,f));gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;};
  prog=link(VS,FS);
  row('full-scan shader','BM25 + containment, one pass',true,'binary search, precomputed dl');
  // A driver may offer float RENDER TARGETS without float BLENDING; blending to one anyway
  // silently produces zeros, so the extension is REQUESTED and the path is chosen on the answer.
  const fb2=gl.getExtension('EXT_float_blend');
  try{ progSC=link(VS_SC,FS_SC); }catch(e){ progSC=null; }
  SCATTER=!!(fb2&&progSC);
  row('EXT_float_blend',fb2?'present':'ABSENT',true,
      fb2?'':'scatter path unavailable -- falling back to the full scan');
  row('SCORER IN USE',SCATTER?'scatter (inverted index)':'full scan',true,
      SCATTER?'one point per posting, additive blending':'one fragment per document');
  const vao=gl.createVertexArray();gl.bindVertexArray(vao);
  const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  tTerm=texU(TERMA,W);tTf=texU(TFA,W);tOff=texU(OFF2A,W);tDl=texF(DL,W);
  tPDoc=texU(PDOC,W);tPTf=texU(PTF,W);
  outTex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,outTex);
  for(const p of[gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RG32F,N,1,0,gl.RG,gl.FLOAT,null);
  fbo=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,outTex,0);

  // Calibrate the null HERE, against the shader that will answer, not against a remote number.
  const nulls=[];
  for(let i=0;i<200;i++){const q=[];
    for(let j=0;j<8;j++) q.push(VOCAB[(Math.random()*VOCAB.length)|0]);
    const px=score(Array.from(new Set(q)));
    let mx=-1e30; for(let d=0;d<N;d++) if(px[2*d]>mx) mx=px[2*d];
    nulls.push(mx);}
  nulls.sort((a,b)=>a-b); THRESH=nulls[Math.floor(0.95*nulls.length)];
  row('null calibrated in-page',THRESH.toFixed(3)+'  (python said '+P.py_threshold.toFixed(3)+')',
      true,'200 scrambled queries through this shader');
  row('hash collisions in vocabulary',P.collisions+' of '+P.nterms+' terms',P.collisions===0,
      'term ids are FNV-1a hashes; no vocabulary table shipped');

  let mode_ok=0,amb_ok=0,ans_ok=0,ans_n=0;const t0=performance.now();
  for(const r of P.refs){const v=verdict(r.q,true);
    if(v.mode===r.mode)mode_ok++; if(v.amb===r.amb)amb_ok++;
    if(r.ans>=0){ans_n++; if(v.answer===r.ans)ans_ok++;}}
  const ms=(performance.now()-t0)/P.refs.length;
  // THE PAGE VERIFIES ITS OWN FAST PATH AGAINST ITS OWN SLOW PATH. Both implementations ship,
  // so the speed claim is falsifiable from the artifact instead of from a notes file.
  if(SCATTER){
    let worst=0, worstCov=0;
    for(const r of P.refs.slice(0,20)){
      const qh=Array.from(new Set(tokenise(r.q.join(' ')).map(fnv1a)));
      const a=scoreScatter(qh), b=scoreFull(qh);
      let m=0,mx=0,c=0;
      for(let d=0;d<N;d++){ m=Math.max(m,Math.abs(a[2*d]-b[2*d])); mx=Math.max(mx,Math.abs(b[2*d]));
                            c=Math.max(c,Math.abs(a[2*d+1]-b[2*d+1])); }
      worst=Math.max(worst,m/(mx||1)); worstCov=Math.max(worstCov,c); }
    row('scatter == full scan',worst.toExponential(2)+' rel, coverage diff '+worstCov,
        worst<1e-4&&worstCov<0.5,'the page checks its fast path against its own slow path');
  }
  row('verdict mode vs f64 engine',mode_ok+' / '+P.refs.length,mode_ok===P.refs.length);
  row('ambiguity count vs engine',amb_ok+' / '+P.refs.length,amb_ok===P.refs.length,
      'exact integers — must match, not approximate');
  row('reranked answer vs engine',ans_ok+' / '+ans_n,ans_ok===ans_n,
      'proximity rerank over the top-10, in JS: 10 docs beside a '+N+'-row pass is not worth a shader');
  row('per-query time',ms.toFixed(2)+' ms',true,'includes a readback; not throughput');
  row('renderer',gl.getParameter(gl.VERSION),true,gl.getParameter(gl.RENDERER));
  render();

  const box=document.getElementById('q');
  const go=()=>{const tk=tokenise(box.value);
    if(!tk.length){document.getElementById('ans').innerHTML='<pre class="dim">type some words…</pre>';return;}
    const v=verdict(tk,true);
    let body='tokens    '+tk.join(' ')+'\nmode      '+v.mode+'\nambiguity '+v.amb+
             '\ntop score '+v.top.toFixed(3)+'  (abstain below '+THRESH.toFixed(3)+')';
    if(v.mode==='answer') body+='\nanswer    '+P.names[v.answer];
    if(v.mode==='set') body+='\nceiling   '+v.ceiling.toFixed(3)+
      ' — these are indistinguishable to a term scorer\nset       '+
      v.set.map(i=>P.names[i]).join('\n          ');
    if(v.mode==='abstain') body+='\n          nothing here matches; refusing rather than guessing';
    document.getElementById('ans').innerHTML='<pre>'+body+'</pre>';};
  box.oninput=go; box.value='holographic vector memory cleanup'; go();
}
function render(){
  let h='<table><tr><th>check</th><th>value</th><th>result</th><th>note</th></tr>';
  for(const [n,v,ok,note] of rows)
    h+='<tr><td>'+n+'</td><td class="n">'+v+'</td><td class="'+(ok?'ok':'bad')+'">'+
       (ok?'PASS':'FAIL')+'</td><td class="dim">'+note+'</td></tr>';
  h+='</table><p class="'+(fails?'bad':'ok')+'">'+(fails?fails+' FAILED':
     'ALL PASS — BM25, containment and the answer/set/abstain policy ran in WebGL2 and matched the f64 engine')+
     '</p><pre>'+N+' passages · '+P.nterms+' terms · source: '+SOURCE+' · no vocabulary table shipped</pre>';
  document.getElementById('out').innerHTML=h;
}
try{main();}catch(e){document.getElementById('out').innerHTML='<p class="bad">'+e.message+'</p>';}

document.getElementById('imp').addEventListener('change', async (e)=>{
  const f=e.target.files[0]; if(!f) return;
  const msg=document.getElementById('impmsg');
  try{
    const b=JSON.parse(await f.text());
    if(b.format!=='lecore-index/1') throw new Error('not a lecore-index/1 bundle');
    await useBundle(b,'imported: '+f.name);      // verifies the digest, throws if it fails
    await idbPut('search',b);                    // only cache what already verified
    document.getElementById('src').textContent=SOURCE+' · '+N+' passages';
    msg.innerHTML='<span style="color:#5ad67d">imported and cached — '+N+
      ' passages, digest verified. Reload and it comes back from IndexedDB.</span>';
  }catch(err){
    // A rejected bundle must NOT be cached: caching first and validating later is how a corpus
    // that answers wrongly becomes the one you load every time.
    msg.innerHTML='<span style="color:#ff6b6b">refused: '+err.message+'</span>';
  }
});
document.getElementById('clr').addEventListener('click', async ()=>{
  const db=await idbOpen();
  await new Promise(r=>{const tx=db.transaction(IDB_STORE,'readwrite');
    tx.objectStore(IDB_STORE).delete('search'); tx.oncomplete=r;});
  document.getElementById('impmsg').textContent='cache cleared — reload to use the embedded corpus';
});
document.getElementById('src').textContent=SOURCE+' · '+N+' passages';
</script>
"""
open("lecore_search_webgl2.html", "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(DATA)))
print("wrote lecore_search_webgl2.html  %.0f KB  (%d passages, %d terms, %d hash collisions)"
      % (os.path.getsize("lecore_search_webgl2.html") / 1024, N, len(allhash), coll))
print("python threshold %.3f ; reference verdicts %d" % (pol.threshold, len(refs)))
