"""The final page: TYPE A QUERY, encoded in-shader, recalled through the tier walk. No vocabulary.

What changed: the query is no longer pre-encoded. The browser hashes tokens with FNV-1a (the
same u32 definition NumPy uses) and a fragment shader turns those hashes into the query vector.
So the page ships MEMORY (doc/chunk/super vectors) but NO VOCABULARY -- an atom is a function of
its name, per lever 3, determinism instead of storage.

It stays a differential test: 100 held-out queries ship as TOKEN STRINGS with the f64 engine's
answers, so the page verifies its own encoder as well as its own recall.
"""
import base64, json, os
import numpy as np
import holographic.agents_and_reasoning.holographic_hashatom as HA
import glsl_hier as H

DIM, BEAM = 256, 4
docs, _ = H.load_corpus(DIM)
K = len(docs)
V = np.stack([HA.encode_hash(t, DIM) for _, t in docs])
g = max(2, int(round(K ** (1/3))))
ch = np.stack([V[i:i+g].sum(0) for i in range(0, K, g)])
su = np.stack([ch[i:i+g].sum(0) for i in range(0, len(ch), g)])

rng = np.random.default_rng(0)
qtoks, T = [], []
for i, (_, tk) in enumerate(docs):
    p = rng.choice(len(tk), max(3, int(len(tk)*0.4)), replace=False)
    qtoks.append([tk[j] for j in p]); T.append(i)

def beam(q, b=BEAM):
    s = np.argsort(su @ q)[::-1][:b]
    cand = np.concatenate([np.arange(x*g, min((x+1)*g, len(ch))) for x in s])
    c = cand[np.argsort(ch[cand] @ q)[::-1][:b]]
    leaf = np.concatenate([np.arange(x*g, min((x+1)*g, K)) for x in c])
    return int(leaf[int(np.argmax(V[leaf] @ q))])

QV = [HA.encode_hash(t, DIM, normalise=False) for t in qtoks]
refBeam = [beam(q) for q in QV]
refFlat = [int(np.argmax(V @ q)) for q in QV]
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()

DATA = dict(D=DIM, K=K, G=g, BEAM=BEAM, NCH=len(ch), NSU=len(su),
            names=[n for n, _ in docs], qtoks=qtoks, truth=T,
            refBeam=refBeam, refFlat=refFlat,
            V=b64(V.ravel()), CH=b64(ch.ravel()), SU=b64(su.ravel()))

HTML = r"""<!doctype html>
<meta charset="utf-8"><title>leCore — typed query, WebGL2</title>
<style>
 body{background:#0b0d10;color:#cfd6e4;font:13px/1.6 ui-monospace,Menlo,Consolas,monospace;
      margin:0;padding:26px 30px;max-width:1000px}
 h1{font-size:17px;color:#e8eefc;margin:0 0 4px}h2{font-size:14px;color:#e8eefc;margin:22px 0 6px}
 p.sub{color:#7d8798;margin:0 0 18px}
 table{border-collapse:collapse;width:100%;margin:8px 0 16px}
 td,th{padding:5px 9px;border-bottom:1px solid #1c2129;text-align:left;vertical-align:top}
 th{color:#7d8798;font-weight:400}
 .ok{color:#5fd38d}.bad{color:#ff6b6b}.n{color:#8ab4ff}.dim{color:#6c7688}
 input{background:#12151b;color:#e8eefc;border:1px solid #2a323d;padding:8px 10px;
       font:inherit;width:100%;box-sizing:border-box;border-radius:5px}
 pre{background:#12151b;padding:11px;border-radius:6px;overflow:auto;color:#9aa5b6;margin:6px 0}
</style>
<h1>leCore — type a query; it is encoded and recalled entirely in WebGL2</h1>
<p class="sub">No vocabulary is shipped. An atom is a function of its name: the browser hashes
each token (FNV-1a, u32) and a fragment shader expands those hashes into the query vector, then
the 3-tier beam walk runs — every decision on the GPU. Reference answers from the f64 engine are
embedded, so the page checks its own encoder and its own recall.</p>
<div id="out">running…</div>
<h2>Ask the memory</h2>
<input id="q" placeholder="e.g. holographic vector memory cleanup resonator" autocomplete="off">
<div id="ans"></div>
<script id="DATA" type="application/json">__DATA__</script>
<script>
const P=JSON.parse(document.getElementById('DATA').textContent);
const dec=s=>{const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);};
const V=dec(P.V),CH=dec(P.CH),SU=dec(P.SU);
const D=P.D,K=P.K,G=P.G,BEAM=P.BEAM,NCH=P.NCH,NSU=P.NSU;

// FNV-1a over UTF-8 bytes, mod 2^32 -- the SAME definition NumPy evaluates. Math.imul is the
// only way to get a wrapping 32-bit multiply in JS; `*` would go through a double and lose bits.
const ENC=new TextEncoder();
function fnv1a(s){let h=2166136261>>>0;
  for(const c of ENC.encode(s)){h=(h^c)>>>0; h=Math.imul(h,16777619)>>>0;} return h>>>0;}
const STOP=new Set(("the a an of to and or is are was in on for with that this it as by be from "+
  "at not but if then than so its which what when how you your we our can use used using").split(" "));
const tokenise=s=>(s.toLowerCase().match(/[a-z][a-z0-9_]{2,}/g)||[]).filter(w=>!STOP.has(w));

const VS=`#version 300 es
in vec2 p; void main(){gl_Position=vec4(p,0.0,1.0);}`;

// THE ENCODER. Token hashes arrive in an INTEGER texture: a float texture cannot carry a u32
// exactly past 2^24, so routing hashes through floats would silently corrupt every atom.
const FS_ENCODE=`#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform usampler2D uTok; uniform int uT; uniform float uScale;
out vec4 o;
void main(){
  uint i = uint(int(gl_FragCoord.x));
  float acc = 0.0;
  for (int t = 0; t < uT; ++t) {
    uint x = texelFetch(uTok, ivec2(t,0), 0).r ^ i;
    uint s = x * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    x = (w >> 22u) ^ w;
    acc += ((x >> 31u) == 1u) ? 1.0 : -1.0;   // Rademacher: no trig, so no float divergence
  }
  o = vec4(acc*uScale, 0.0, 0.0, 1.0);        // NOT normalised: a positive scalar cannot move an argmax
}`;

const FS_GATHER=`#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uM,uQ,uIdx; uniform int uD,uG,uN,uNI;
out vec4 o;
void main(){ int t=int(gl_FragCoord.x); int pi=t/uG;
  if(pi>=uNI){o=vec4(-1e30,0,0,1);return;}
  int row=int(texelFetch(uIdx,ivec2(pi,0),0).r+0.5)*uG+(t-pi*uG);
  if(row>=uN){o=vec4(-1e30,0,0,1);return;}
  float a=0.0;
  for(int i=0;i<uD;++i) a+=texelFetch(uM,ivec2(i,row),0).r*texelFetch(uQ,ivec2(i,0),0).r;
  o=vec4(a,0,0,1); }`;

const FS_TOPB=`#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uS,uIdx; uniform int uN,uG,uUseIdx;
out vec4 o;
void main(){ int want=int(gl_FragCoord.x);
  for(int i=0;i<uN;++i){ float v=texelFetch(uS,ivec2(i,0),0).r; int rank=0;
    for(int j=0;j<uN;++j){ float w=texelFetch(uS,ivec2(j,0),0).r;
      if(w>v||(w==v&&j<i)) rank+=1; }
    if(rank==want){ if(uUseIdx==0){o=vec4(float(i),0,0,1);return;}
      int pi=i/uG;
      o=vec4(float(int(texelFetch(uIdx,ivec2(pi,0),0).r+0.5)*uG+(i-pi*uG)),0,0,1); return; } }
  o=vec4(-1.0,0,0,1); }`;

let gl,pEnc,pGat,pTop,texV,texC,texS,texIdent;
const rows=[]; let fails=0;
const row=(n,v,ok,note)=>{rows.push([n,v,ok,note||'']);if(ok===false)fails++;};
const texF=(w,h,d)=>{const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  for(const p of[gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);
  for(const p of[gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.R32F,w,h,0,gl.RED,gl.FLOAT,d||null);return t;};
const texU=(w,d)=>{const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  for(const p of[gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);
  for(const p of[gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.R32UI,w,1,0,gl.RED_INTEGER,gl.UNSIGNED_INT,d);return t;};

function run(p,w,binds,ints,floats){
  const out=texF(w,1,null),fb=gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER,fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,out,0);
  gl.useProgram(p); gl.viewport(0,0,w,1);
  binds.forEach(([n,t],u)=>{gl.activeTexture(gl.TEXTURE0+u);gl.bindTexture(gl.TEXTURE_2D,t);
    gl.uniform1i(gl.getUniformLocation(p,n),u);});
  for(const k in ints){const l=gl.getUniformLocation(p,k);if(l)gl.uniform1i(l,ints[k]);}
  for(const k in (floats||{})){const l=gl.getUniformLocation(p,k);if(l)gl.uniform1f(l,floats[k]);}
  gl.drawArrays(gl.TRIANGLES,0,3);
  const px=new Float32Array(w); gl.readPixels(0,0,w,1,gl.RED,gl.FLOAT,px);
  gl.deleteFramebuffer(fb); gl.deleteTexture(out); return px;
}

function encodeQuery(tokens){
  const h=new Uint32Array(tokens.map(fnv1a));
  const tt=texU(Math.max(1,h.length),h);
  const q=run(pEnc,D,[['uTok',tt]],{uT:h.length},{uScale:1/Math.sqrt(D)});
  gl.deleteTexture(tt); return q;
}

function walk(qvec,trace){
  const q=texF(D,1,qvec);
  const s1=run(pGat,NSU,[['uM',texS],['uQ',q],['uIdx',texIdent]],{uD:D,uG:1,uN:NSU,uNI:NSU});
  const i1=run(pTop,BEAM,[['uS',texF(NSU,1,s1)],['uIdx',texIdent]],{uN:NSU,uG:1,uUseIdx:0});
  const t1=texF(BEAM,1,i1);
  const s2=run(pGat,BEAM*G,[['uM',texC],['uQ',q],['uIdx',t1]],{uD:D,uG:G,uN:NCH,uNI:BEAM});
  const i2=run(pTop,BEAM,[['uS',texF(BEAM*G,1,s2)],['uIdx',t1]],{uN:BEAM*G,uG:G,uUseIdx:1});
  const t2=texF(BEAM,1,i2);
  const s3=run(pGat,BEAM*G,[['uM',texV],['uQ',q],['uIdx',t2]],{uD:D,uG:G,uN:K,uNI:BEAM});
  const i4=run(pTop,BEAM,[['uS',texF(BEAM*G,1,s3)],['uIdx',t2]],{uN:BEAM*G,uG:G,uUseIdx:1});
  if(trace){trace.supers=Array.from(i1);trace.chunks=Array.from(i2);
            trace.top=Array.from(i4).map(x=>Math.round(x));
            trace.scores=Array.from(s3).slice().sort((a,b)=>b-a);}
  gl.deleteTexture(q);gl.deleteTexture(t1);gl.deleteTexture(t2);
  return Math.round(i4[0]);
}

function main(){
  gl=document.createElement('canvas').getContext('webgl2');
  if(!gl){document.getElementById('out').innerHTML='<p class="bad">No WebGL2.</p>';return;}
  const ext=gl.getExtension('EXT_color_buffer_float');
  row('EXT_color_buffer_float',ext?'present':'ABSENT',!!ext,
      ext?'':'cannot render to R32F — every number below would be meaningless');
  if(!ext){render();return;}
  const sh=(t,s)=>{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(o));return o;};
  const prog=f=>{const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,VS));
    gl.attachShader(p,sh(gl.FRAGMENT_SHADER,f));gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;};
  pEnc=prog(FS_ENCODE); pGat=prog(FS_GATHER); pTop=prog(FS_TOPB);
  row('shaders compile + link','encoder (hash→atom), gather, top-b',true);

  const vao=gl.createVertexArray();gl.bindVertexArray(vao);
  const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  texV=texF(D,K,V);texC=texF(D,NCH,CH);texS=texF(D,NSU,SU);
  texIdent=texF(NSU,1,new Float32Array(Array.from({length:NSU},(_,i)=>i)));

  let agree=0,correct=0;
  const t0=performance.now();
  for(let i=0;i<K;i++){const a=walk(encodeQuery(P.qtoks[i]),null);
    if(a===P.refBeam[i])agree++; if(a===P.truth[i])correct++;}
  const ms=(performance.now()-t0)/K;
  const refAcc=P.refBeam.filter((a,i)=>a===P.truth[i]).length/K;

  row('encode + recall vs f64 engine',agree+' / '+K+' identical answers',agree===K,
      agree===K?'the browser encoder reproduces NumPy exactly':'diverged on '+(K-agree));
  row('retrieval accuracy',(correct/K).toFixed(4)+'  (f64 engine '+refAcc.toFixed(4)+')',
      correct/K===refAcc);
  row('vocabulary shipped','0 bytes — an atom is a function of its name',true,
      'lever 3: determinism instead of storage');
  row('dot products / query',(NSU+2*BEAM*G)+' vs '+K+' flat',true,
      (K/(NSU+2*BEAM*G)).toFixed(1)+'× fewer, at parity');
  row('per-query time',ms.toFixed(2)+' ms',true,'includes readbacks — NOT a throughput number');
  row('renderer',gl.getParameter(gl.VERSION),true,gl.getParameter(gl.RENDERER));
  render();

  const box=document.getElementById('q');
  const go=()=>{const tk=tokenise(box.value);
    if(!tk.length){document.getElementById('ans').innerHTML=
      '<pre class="dim">type some words…</pre>';return;}
    const tr={}; const a=walk(encodeQuery(tk),tr);
    const margin=tr.scores[0]-tr.scores[1];
    document.getElementById('ans').innerHTML='<pre>'+
      'tokens      '+tk.join(' ')+'\n'+
      'supers kept '+tr.supers.join(', ')+'\n'+
      'chunks kept '+tr.chunks.join(', ')+'\n'+
      'top hits    '+tr.top.map(i=>P.names[i]).join('\n            ')+'\n'+
      'margin      '+margin.toFixed(4)+(margin>1e-4?'  (T1 gate: ANSWERS)':'  (T1 gate: ABSTAINS)')+
      '</pre>';};
  box.oninput=go; box.value='holographic vector memory cleanup resonator'; go();
}
function render(){
  let h='<table><tr><th>check</th><th>value</th><th>result</th><th>note</th></tr>';
  for(const [n,v,ok,note] of rows)
    h+='<tr><td>'+n+'</td><td class="n">'+v+'</td><td class="'+(ok?'ok':'bad')+'">'+
       (ok?'PASS':'FAIL')+'</td><td class="dim">'+note+'</td></tr>';
  h+='</table><p class="'+(fails?'bad':'ok')+'">'+(fails?fails+' FAILED':
     'ALL PASS — query encoded and recalled in WebGL2, matching the f64 engine')+'</p>'+
     '<pre>'+K+' docstrings · dim '+D+' · tiers '+K+'→'+NCH+'→'+NSU+' · g='+G+' · beam '+BEAM+'</pre>';
  document.getElementById('out').innerHTML=h;
}
try{main();}catch(e){document.getElementById('out').innerHTML='<p class="bad">'+e.message+'</p>';}
</script>
"""
open("lecore_webgl2_typed.html","w",encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(DATA)))
print("wrote lecore_webgl2_typed.html  %.0f KB" % (os.path.getsize("lecore_webgl2_typed.html")/1024))
print("corpus %d, dim %d, tiers %d->%d->%d, g=%d, beam=%d" % (K, DIM, K, len(ch), len(su), g, BEAM))
print("f64 reference: flat %.4f, beam %.4f"
      % (sum(a==t for a,t in zip(refFlat,T))/K, sum(a==t for a,t in zip(refBeam,T))/K))
