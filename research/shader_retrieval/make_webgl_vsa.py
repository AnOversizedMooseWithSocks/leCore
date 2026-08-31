"""The whole thing in one page: retrieval AND the VSA algebra, both zero-vocabulary, in WebGL2.

Two families, two jobs, both generated from the same u32 name hash so NOTHING is shipped except
the corpus memory itself:
  * Rademacher atoms -> typed query encoding + 3-tier beam recall (bundling vocabulary)
  * FHRR phasor atoms -> bind / unbind / bundle / cleanup (binding vocabulary, exact)
The record is BUILT in a shader from role and filler name hashes, so for the algebra half not
even the record is stored -- it is a pure function of the names you type.

Both halves are differential tests: f64 answers are embedded and compared on load.
"""
import base64, json, os
import numpy as np
import holographic.agents_and_reasoning.holographic_hashatom as HA
import holographic.agents_and_reasoning.holographic_phasor as PH
import glsl_hier as H

DIM, BEAM = 256, 4
docs, _ = H.load_corpus(DIM); K = len(docs)
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
refBeam = [beam(HA.encode_hash(t, DIM, normalise=False)) for t in qtoks]

ROLES = ["colour", "size", "material", "owner", "era"]
FILLS = ["red", "large", "metal", "moose", "modern"]
CANDS = FILLS + ["blue", "small", "wood", "green", "tiny", "glass", "stone", "ancient"]
rec = PH.bundle([PH.bind(PH.atom(r, DIM), PH.atom(f, DIM)) for r, f in zip(ROLES, FILLS)])
refRoles = [PH.cleanup(PH.unbind(rec, PH.atom(r, DIM)), CANDS, DIM)[0] for r in ROLES]

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()
DATA = dict(D=DIM, K=K, G=g, BEAM=BEAM, NCH=len(ch), NSU=len(su),
            names=[n for n, _ in docs], qtoks=qtoks, truth=T, refBeam=refBeam,
            roles=ROLES, fills=FILLS, cands=CANDS, refRoles=refRoles,
            V=b64(V.ravel()), CH=b64(ch.ravel()), SU=b64(su.ravel()))

HTML = r"""<!doctype html>
<meta charset="utf-8"><title>leCore in WebGL2</title>
<style>
 body{background:#0b0d10;color:#cfd6e4;font:13px/1.6 ui-monospace,Menlo,Consolas,monospace;
      margin:0;padding:26px 30px;max-width:1000px}
 h1{font-size:17px;color:#e8eefc;margin:0 0 4px}h2{font-size:14px;color:#e8eefc;margin:24px 0 6px}
 p.sub{color:#7d8798;margin:0 0 16px}
 table{border-collapse:collapse;width:100%;margin:8px 0 14px}
 td,th{padding:5px 9px;border-bottom:1px solid #1c2129;text-align:left;vertical-align:top}
 th{color:#7d8798;font-weight:400}
 .ok{color:#5fd38d}.bad{color:#ff6b6b}.n{color:#8ab4ff}.dim{color:#6c7688}
 input{background:#12151b;color:#e8eefc;border:1px solid #2a323d;padding:8px 10px;font:inherit;
       width:100%;box-sizing:border-box;border-radius:5px;margin-bottom:6px}
 pre{background:#12151b;padding:11px;border-radius:6px;overflow:auto;color:#9aa5b6;margin:6px 0}
</style>
<h1>leCore — retrieval <span class="dim">and</span> the VSA algebra, executed in WebGL2</h1>
<p class="sub">No vocabulary is shipped. Every atom is a function of its name: the browser hashes
each token (FNV-1a, u32) and shaders expand those hashes on demand. Two families, two jobs —
Rademacher for bundling/retrieval, FHRR phasors for exact bind/unbind. f64 answers are embedded,
so both halves are differential tests.</p>
<div id="out">running…</div>

<h2>1 · Ask the memory <span class="dim">— 100 docstrings, 3-tier beam walk</span></h2>
<input id="q" placeholder="holographic vector memory cleanup resonator" autocomplete="off">
<div id="ans"></div>

<h2>2 · Build a record and take it apart <span class="dim">— bind · bundle · unbind · cleanup</span></h2>
<input id="rec" placeholder="colour:red size:large material:metal owner:moose" autocomplete="off">
<div id="recans"></div>

<script id="DATA" type="application/json">__DATA__</script>
<script>
const P=JSON.parse(document.getElementById('DATA').textContent);
const dec=s=>{const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);};
const V=dec(P.V),CH=dec(P.CH),SU=dec(P.SU);
const D=P.D,K=P.K,G=P.G,BEAM=P.BEAM,NCH=P.NCH,NSU=P.NSU;
const ENC=new TextEncoder();
function fnv1a(s){let h=2166136261>>>0;
  for(const c of ENC.encode(s)){h=(h^c)>>>0;h=Math.imul(h,16777619)>>>0;}return h>>>0;}
const STOP=new Set(("the a an of to and or is are was in on for with that this it as by be from "+
 "at not but if then than so its which what when how you your we our can use used using").split(" "));
const tokenise=s=>(s.toLowerCase().match(/[a-z][a-z0-9_]{2,}/g)||[]).filter(w=>!STOP.has(w));

const VS=`#version 300 es
in vec2 p; void main(){gl_Position=vec4(p,0.0,1.0);}`;
const HDR=`#version 300 es
precision highp float; precision highp int;
precision highp sampler2D; precision highp usampler2D;
`;
// One atom generator, shared by every shader below. Phases in TURNS: the only constant that has
// to agree across NumPy, GLSL and JS is 2^32, never pi.
const ATOM=`
float atomTurn(uint b, uint i){ uint x=b^i;
  uint s = x * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; x = (w >> 22u) ^ w;
  return float(x)/4294967296.0; }
vec2 atomC(uint b, uint i){ float t=atomTurn(b,i)*6.283185307179586;
  return vec2(cos(t),sin(t)); }
float atomR(uint b, uint i){ uint x=b^i;
  uint s = x * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; x = (w >> 22u) ^ w;
  return ((x>>31u)==1u)?1.0:-1.0; }
vec2 cmul(vec2 a, vec2 b){ return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
`;
const FS_ENCODE=HDR+ATOM+`
uniform usampler2D uTok; uniform int uT; uniform float uScale;
out vec4 o;
void main(){ uint i=uint(int(gl_FragCoord.x)); float a=0.0;
  for(int t=0;t<uT;++t) a+=atomR(texelFetch(uTok,ivec2(t,0),0).r,i);
  o=vec4(a*uScale,0,0,1); }`;      // not normalised: a positive scalar cannot move an argmax
const FS_GATHER=HDR+`
uniform sampler2D uM,uQ,uIdx; uniform int uD,uG,uN,uNI;
out vec4 o;
void main(){ int t=int(gl_FragCoord.x); int pi=t/uG;
  if(pi>=uNI){o=vec4(-1e30,0,0,1);return;}
  int row=int(texelFetch(uIdx,ivec2(pi,0),0).r+0.5)*uG+(t-pi*uG);
  if(row>=uN){o=vec4(-1e30,0,0,1);return;}
  float a=0.0; for(int i=0;i<uD;++i)
    a+=texelFetch(uM,ivec2(i,row),0).r*texelFetch(uQ,ivec2(i,0),0).r;
  o=vec4(a,0,0,1); }`;
const FS_TOPB=HDR+`
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
// The record is BUILT in a shader from name hashes -- for the algebra half, not even the record
// is stored. It is a pure function of the names typed into the box.
const FS_RECORD=HDR+ATOM+`
uniform usampler2D uRole,uFill; uniform int uT;
out vec2 o;
void main(){ uint i=uint(int(gl_FragCoord.x)); vec2 acc=vec2(0.0);
  for(int t=0;t<uT;++t) acc+=cmul(atomC(texelFetch(uRole,ivec2(t,0),0).r,i),
                                  atomC(texelFetch(uFill,ivec2(t,0),0).r,i));
  o=acc; }`;
// bind (uConj=0) / unbind (uConj=1) against a GENERATED atom. |atom|=1, so the conjugate is the
// TRUE inverse -- no pseudo-inverse, no normalisation.
const FS_BINDC=HDR+ATOM+`
uniform sampler2D uZ; uniform uint uKey; uniform int uConj;
out vec2 o;
void main(){ uint i=uint(int(gl_FragCoord.x)); vec2 k=atomC(uKey,i);
  if(uConj==1) k.y=-k.y;
  o=cmul(texelFetch(uZ,ivec2(int(i),0),0).rg,k); }`;
const FS_CLEAN=HDR+ATOM+`
uniform sampler2D uZ; uniform usampler2D uNames; uniform int uD;
out vec4 o;
void main(){ int c=int(gl_FragCoord.x); uint b=texelFetch(uNames,ivec2(c,0),0).r;
  float acc=0.0;
  for(int i=0;i<uD;++i){ vec2 z=texelFetch(uZ,ivec2(i,0),0).rg; vec2 a=atomC(b,uint(i));
    acc+=z.x*a.x+z.y*a.y; }
  o=vec4(acc,0,0,1); }`;

let gl,pEnc,pGat,pTop,pRec,pBind,pClean,texV,texC,texS,texIdent;
const rows=[]; let fails=0;
const row=(n,v,ok,note)=>{rows.push([n,v,ok,note||'']);if(ok===false)fails++;};
function mkTex(w,h,comps,data){const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  for(const p of[gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);
  for(const p of[gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);
  const [ifmt,fmt]=comps===2?[gl.RG32F,gl.RG]:[gl.R32F,gl.RED];
  gl.texImage2D(gl.TEXTURE_2D,0,ifmt,w,h,0,fmt,gl.FLOAT,data||null);return t;}
function texU(u){const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  for(const p of[gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);
  for(const p of[gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.R32UI,u.length,1,0,gl.RED_INTEGER,gl.UNSIGNED_INT,u);return t;}
function run(p,w,comps,binds,ints,floats,uints){
  const out=mkTex(w,1,comps,null),fb=gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER,fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,out,0);
  gl.useProgram(p); gl.viewport(0,0,w,1);
  binds.forEach(([n,t],u)=>{gl.activeTexture(gl.TEXTURE0+u);gl.bindTexture(gl.TEXTURE_2D,t);
    gl.uniform1i(gl.getUniformLocation(p,n),u);});
  for(const k in (ints||{})){const l=gl.getUniformLocation(p,k);if(l)gl.uniform1i(l,ints[k]);}
  for(const k in (floats||{})){const l=gl.getUniformLocation(p,k);if(l)gl.uniform1f(l,floats[k]);}
  for(const k in (uints||{})){const l=gl.getUniformLocation(p,k);if(l)gl.uniform1ui(l,uints[k]);}
  gl.drawArrays(gl.TRIANGLES,0,3);
  const px=new Float32Array(w*comps);
  gl.readPixels(0,0,w,1,comps===2?gl.RG:gl.RED,gl.FLOAT,px);
  gl.deleteFramebuffer(fb); gl.deleteTexture(out); return px;
}
function encodeQuery(tokens){
  const t=texU(new Uint32Array(tokens.map(fnv1a)));
  const q=run(pEnc,D,1,[['uTok',t]],{uT:tokens.length},{uScale:1/Math.sqrt(D)});
  gl.deleteTexture(t); return q;
}
function walk(qv,tr){
  const q=mkTex(D,1,1,qv);
  const s1=run(pGat,NSU,1,[['uM',texS],['uQ',q],['uIdx',texIdent]],{uD:D,uG:1,uN:NSU,uNI:NSU});
  const i1=run(pTop,BEAM,1,[['uS',mkTex(NSU,1,1,s1)],['uIdx',texIdent]],{uN:NSU,uG:1,uUseIdx:0});
  const t1=mkTex(BEAM,1,1,i1);
  const s2=run(pGat,BEAM*G,1,[['uM',texC],['uQ',q],['uIdx',t1]],{uD:D,uG:G,uN:NCH,uNI:BEAM});
  const i2=run(pTop,BEAM,1,[['uS',mkTex(BEAM*G,1,1,s2)],['uIdx',t1]],{uN:BEAM*G,uG:G,uUseIdx:1});
  const t2=mkTex(BEAM,1,1,i2);
  const s3=run(pGat,BEAM*G,1,[['uM',texV],['uQ',q],['uIdx',t2]],{uD:D,uG:G,uN:K,uNI:BEAM});
  const i3=run(pTop,BEAM,1,[['uS',mkTex(BEAM*G,1,1,s3)],['uIdx',t2]],{uN:BEAM*G,uG:G,uUseIdx:1});
  if(tr){tr.supers=Array.from(i1);tr.chunks=Array.from(i2);
         tr.top=Array.from(i3).map(Math.round);
         tr.scores=Array.from(s3).sort((a,b)=>b-a);}
  gl.deleteTexture(q);gl.deleteTexture(t1);gl.deleteTexture(t2);
  return Math.round(i3[0]);
}
function buildRecord(pairs){
  const tR=texU(new Uint32Array(pairs.map(p=>fnv1a(p[0]))));
  const tF=texU(new Uint32Array(pairs.map(p=>fnv1a(p[1]))));
  const z=run(pRec,D,2,[['uRole',tR],['uFill',tF]],{uT:pairs.length});
  gl.deleteTexture(tR);gl.deleteTexture(tF); return z;
}
function askRole(recZ,role,cands){
  const zt=mkTex(D,1,2,recZ);
  const pr=run(pBind,D,2,[['uZ',zt]],{uConj:1},null,{uKey:fnv1a(role)});
  const nt=texU(new Uint32Array(cands.map(fnv1a)));
  const s=run(pClean,cands.length,1,[['uZ',mkTex(D,1,2,pr)],['uNames',nt]],{uD:D});
  gl.deleteTexture(zt);gl.deleteTexture(nt);
  let bi=0; for(let i=1;i<s.length;i++) if(s[i]>s[bi]) bi=i;
  const srt=Array.from(s).sort((a,b)=>b-a);
  return {name:cands[bi],margin:srt[0]-srt[1]};
}

function main(){
  gl=document.createElement('canvas').getContext('webgl2');
  if(!gl){document.getElementById('out').innerHTML='<p class="bad">No WebGL2.</p>';return;}
  const ext=gl.getExtension('EXT_color_buffer_float');
  row('EXT_color_buffer_float',ext?'present':'ABSENT',!!ext,
      ext?'needed for R32F and RG32F targets':'every number below would be meaningless');
  if(!ext){render();return;}
  const sh=(t,s)=>{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(o));return o;};
  const prog=f=>{const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,VS));
    gl.attachShader(p,sh(gl.FRAGMENT_SHADER,f));gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;};
  pEnc=prog(FS_ENCODE);pGat=prog(FS_GATHER);pTop=prog(FS_TOPB);
  pRec=prog(FS_RECORD);pBind=prog(FS_BINDC);pClean=prog(FS_CLEAN);
  row('shaders compile + link','encode, gather, top-b, record, bind/unbind, cleanup',true,'6 programs');

  const vao=gl.createVertexArray();gl.bindVertexArray(vao);
  const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  texV=mkTex(D,K,1,V);texC=mkTex(D,NCH,1,CH);texS=mkTex(D,NSU,1,SU);
  texIdent=mkTex(NSU,1,1,new Float32Array(Array.from({length:NSU},(_,i)=>i)));

  let agree=0,correct=0;const t0=performance.now();
  for(let i=0;i<K;i++){const a=walk(encodeQuery(P.qtoks[i]),null);
    if(a===P.refBeam[i])agree++; if(a===P.truth[i])correct++;}
  const ms=(performance.now()-t0)/K;
  const refAcc=P.refBeam.filter((a,i)=>a===P.truth[i]).length/K;
  row('retrieval: encode+recall vs f64',agree+' / '+K+' identical',agree===K,
      'the browser encoder reproduces NumPy');
  row('retrieval accuracy',(correct/K).toFixed(4)+'  (f64 '+refAcc.toFixed(4)+')',correct/K===refAcc);

  const recZ=buildRecord(P.roles.map((r,i)=>[r,P.fills[i]]));
  let algOK=0;
  const detail=P.roles.map((r,i)=>{const a=askRole(recZ,r,P.cands);
    if(a.name===P.refRoles[i]&&a.name===P.fills[i])algOK++;
    return r+' → '+a.name+'   (f64 '+P.refRoles[i]+', margin '+a.margin.toFixed(1)+')';});
  row('algebra: bind·bundle·unbind·cleanup',algOK+' / '+P.roles.length+' roles recovered',
      algOK===P.roles.length,'record built in-shader from name hashes');
  row('vocabulary stored','0 bytes — atoms and the record are functions of their names',true,
      'lever 3: determinism instead of storage');
  row('dot products / query',(NSU+2*BEAM*G)+' vs '+K+' flat',true,
      (K/(NSU+2*BEAM*G)).toFixed(1)+'× fewer, at parity');
  row('per-query time',ms.toFixed(2)+' ms',true,'includes readbacks — NOT throughput');
  row('renderer',gl.getParameter(gl.VERSION),true,gl.getParameter(gl.RENDERER));
  render();
  document.getElementById('recans').innerHTML='<pre>'+detail.join('\n')+'</pre>';

  const box=document.getElementById('q');
  const go=()=>{const tk=tokenise(box.value);
    if(!tk.length){document.getElementById('ans').innerHTML='<pre class="dim">type some words…</pre>';return;}
    const tr={};const a=walk(encodeQuery(tk),tr);const m=tr.scores[0]-tr.scores[1];
    document.getElementById('ans').innerHTML='<pre>tokens      '+tk.join(' ')+
      '\nsupers kept '+tr.supers.join(', ')+'\nchunks kept '+tr.chunks.join(', ')+
      '\ntop hits    '+tr.top.map(i=>P.names[i]).join('\n            ')+
      '\nmargin      '+m.toFixed(4)+(m>1e-4?'  (T1 gate: ANSWERS)':'  (T1 gate: ABSTAINS)')+'</pre>';};
  box.oninput=go; box.value='holographic vector memory cleanup resonator'; go();

  const rbox=document.getElementById('rec');
  const rgo=()=>{const pairs=(rbox.value.match(/[a-z0-9_]+:[a-z0-9_]+/gi)||[])
      .map(s=>s.toLowerCase().split(':'));
    if(!pairs.length){document.getElementById('recans').innerHTML=
      '<pre class="dim">type role:filler pairs…</pre>';return;}
    const z=buildRecord(pairs);
    const cands=Array.from(new Set(pairs.map(p=>p[1]).concat(P.cands)));
    document.getElementById('recans').innerHTML='<pre>'+
      pairs.map(p=>{const a=askRole(z,p[0],cands);
        return 'unbind '+p[0].padEnd(10)+'→ '+a.name.padEnd(10)+
               (a.name===p[1]?'ok':'MISS (stored '+p[1]+')')+'   margin '+a.margin.toFixed(1);})
      .join('\n')+'\n\nrecord: '+pairs.length+' bound pairs bundled into '+D+
      ' complex components, built in a shader, nothing stored</pre>';};
  rbox.oninput=rgo;
  rbox.value=P.roles.map((r,i)=>r+':'+P.fills[i]).join(' ');
}
function render(){
  let h='<table><tr><th>check</th><th>value</th><th>result</th><th>note</th></tr>';
  for(const [n,v,ok,note] of rows)
    h+='<tr><td>'+n+'</td><td class="n">'+v+'</td><td class="'+(ok?'ok':'bad')+'">'+
       (ok?'PASS':'FAIL')+'</td><td class="dim">'+note+'</td></tr>';
  h+='</table><p class="'+(fails?'bad':'ok')+'">'+(fails?fails+' FAILED':
     'ALL PASS — retrieval and the VSA algebra both ran in WebGL2 and matched the f64 engine')+
     '</p><pre>'+K+' docstrings · dim '+D+' · tiers '+K+'→'+NCH+'→'+NSU+' · g='+G+
     ' · beam '+BEAM+' · Rademacher atoms for bundling, FHRR phasors for binding</pre>';
  document.getElementById('out').innerHTML=h;
}
try{main();}catch(e){document.getElementById('out').innerHTML='<p class="bad">'+e.message+'</p>';}
</script>
"""
open("lecore_webgl2_vsa.html","w",encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(DATA)))
print("wrote lecore_webgl2_vsa.html  %.0f KB" % (os.path.getsize("lecore_webgl2_vsa.html")/1024))
print("retrieval beam acc %.4f | algebra roles %s" % (
    sum(a==t for a,t in zip(refBeam,T))/K, refRoles))
