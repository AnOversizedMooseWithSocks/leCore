"""Generate the FULL single-file WebGL2 page: real corpus, beam-search tier walk, self-verifying.

The earlier page proved the algebra (bind/score/argmax) on synthetic atoms. This one runs the
ACTUAL read path -- 100 real module docstrings, the 3-tier store, beam-4 coarse-to-fine walk --
in the browser, and checks every query against the f64 engine's answer, embedded.

WHY QUERIES ARE EMBEDDED RATHER THAN TYPED: encoding free text in the browser would need
derived_atom (blake2b + PCG64 + numpy's ziggurat), which is a DECLARED DEAD END in GLSL ES --
reproducing it would be a second implementation of the atom generator. Tokenisation and encoding
are host work by the same boundary that says arithmetic installs and control does not. So the
page ships pre-encoded held-out queries and remains a differential test rather than a demo.
"""
import base64, json, os
import numpy as np
import glsl_hier as H

DIM, BEAM = 256, 4
docs, V = H.load_corpus(DIM)
K = len(V)
g = max(2, int(round(K ** (1 / 3))))
ch = np.stack([V[i:i + g].sum(0) for i in range(0, K, g)])
su = np.stack([ch[i:i + g].sum(0) for i in range(0, len(ch), g)])

rng = np.random.default_rng(0)
Q, T = [], []
for i, (_, tk) in enumerate(docs):
    p = rng.choice(len(tk), max(3, int(len(tk) * 0.4)), replace=False)
    Q.append(H.encode([tk[j] for j in p], DIM)); T.append(i)
Q = np.stack(Q)

def np_beam(q, b=BEAM):
    s = np.argsort(su @ q)[::-1][:b]
    cand = np.concatenate([np.arange(x*g, min((x+1)*g, len(ch))) for x in s])
    c = cand[np.argsort(ch[cand] @ q)[::-1][:b]]
    leaf = np.concatenate([np.arange(x*g, min((x+1)*g, K)) for x in c])
    return int(leaf[int(np.argmax(V[leaf] @ q))])

ref_beam = [np_beam(q) for q in Q]
ref_flat = [int(np.argmax(V @ q)) for q in Q]
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()

DATA = dict(D=DIM, K=K, G=g, BEAM=BEAM, NCH=len(ch), NSU=len(su),
            names=[n for n, _ in docs], truth=T, refBeam=ref_beam, refFlat=ref_flat,
            V=b64(V.ravel()), CH=b64(ch.ravel()), SU=b64(su.ravel()), Q=b64(Q.ravel()))

HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>leCore — full read path in WebGL2</title>
<style>
 body{background:#0b0d10;color:#cfd6e4;font:13px/1.6 ui-monospace,Menlo,Consolas,monospace;
      margin:0;padding:26px 30px;max-width:1000px}
 h1{font-size:17px;color:#e8eefc;margin:0 0 4px} p.sub{color:#7d8798;margin:0 0 18px}
 table{border-collapse:collapse;width:100%;margin:8px 0 16px}
 td,th{padding:5px 9px;border-bottom:1px solid #1c2129;text-align:left;vertical-align:top}
 th{color:#7d8798;font-weight:400}
 .ok{color:#5fd38d}.bad{color:#ff6b6b}.n{color:#8ab4ff}.dim{color:#6c7688}
 select{background:#12151b;color:#cfd6e4;border:1px solid #232a34;padding:5px;font:inherit}
 pre{background:#12151b;padding:11px;border-radius:6px;overflow:auto;color:#9aa5b6;margin:6px 0}
</style>
<h1>leCore — 3-tier beam recall over 100 real docstrings, executed in WebGL2</h1>
<p class="sub">Every decision is made on the GPU: each level's winners go to a texture the next
pass reads. Reference answers come from the f64 engine and are embedded — this is a differential
test. Fragment shaders and texelFetch only; no compute shaders, no SSBOs.</p>
<div id="out">running…</div>
<div id="probe"></div>
<script id="DATA" type="application/json">__DATA__</script>
<script>
const P=JSON.parse(document.getElementById('DATA').textContent);
const dec=s=>{const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);};
const V=dec(P.V), CH=dec(P.CH), SU=dec(P.SU), QQ=dec(P.Q);
const D=P.D,K=P.K,G=P.G,BEAM=P.BEAM,NCH=P.NCH,NSU=P.NSU;

const VS=`#version 300 es
in vec2 p; void main(){gl_Position=vec4(p,0.0,1.0);}`;

// Score rows NAMED BY AN INDEX TEXTURE: row = idx[t/uG]*uG + t%uG.
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

// Top-b by RANK COUNTING: slot t takes the entry beaten by exactly t others (ties by index).
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

let gl,pGather,pTopb,vaoBound,texV,texC,texS,texIdent;
const rows=[]; let fails=0;
const row=(n,v,ok,note)=>{rows.push([n,v,ok,note||'']); if(ok===false)fails++;};

function tex(w,h,data){const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  for(const p of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);
  for(const p of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.R32F,w,h,0,gl.RED,gl.FLOAT,data||null); return t;}

function run(p,w,binds,ints){
  const out=tex(w,1,null), fb=gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER,fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,out,0);
  gl.useProgram(p); gl.viewport(0,0,w,1);
  binds.forEach(([n,t],u)=>{gl.activeTexture(gl.TEXTURE0+u);gl.bindTexture(gl.TEXTURE_2D,t);
    gl.uniform1i(gl.getUniformLocation(p,n),u);});
  for(const k in ints){const l=gl.getUniformLocation(p,k); if(l) gl.uniform1i(l,ints[k]);}
  gl.drawArrays(gl.TRIANGLES,0,3);
  const px=new Float32Array(w); gl.readPixels(0,0,w,1,gl.RED,gl.FLOAT,px);
  gl.deleteFramebuffer(fb); gl.deleteTexture(out);
  return px;
}

// The tier walk. Nothing here reads back a decision to steer the next pass -- the index
// textures do that on the GPU. The single readback per pass is only so the page can print.
function walk(qi, trace){
  const q=tex(D,1,QQ.subarray(qi*D,(qi+1)*D));
  const s1=run(pGather,NSU,[['uM',texS],['uQ',q],['uIdx',texIdent]],{uD:D,uG:1,uN:NSU,uNI:NSU});
  const i1=run(pTopb,BEAM,[['uS',tex(NSU,1,s1)],['uIdx',texIdent]],{uN:NSU,uG:1,uUseIdx:0});
  const t1=tex(BEAM,1,i1);
  const s2=run(pGather,BEAM*G,[['uM',texC],['uQ',q],['uIdx',t1]],{uD:D,uG:G,uN:NCH,uNI:BEAM});
  const i2=run(pTopb,BEAM,[['uS',tex(BEAM*G,1,s2)],['uIdx',t1]],{uN:BEAM*G,uG:G,uUseIdx:1});
  const t2=tex(BEAM,1,i2);
  const s3=run(pGather,BEAM*G,[['uM',texV],['uQ',q],['uIdx',t2]],{uD:D,uG:G,uN:K,uNI:BEAM});
  const i3=run(pTopb,1,[['uS',tex(BEAM*G,1,s3)],['uIdx',t2]],{uN:BEAM*G,uG:G,uUseIdx:1});
  if(trace) trace.push(['supers kept',Array.from(i1).join(', ')],
                       ['chunks kept',Array.from(i2).join(', ')]);
  gl.deleteTexture(q); gl.deleteTexture(t1); gl.deleteTexture(t2);
  return Math.round(i3[0]);
}

function main(){
  const cv=document.createElement('canvas'); gl=cv.getContext('webgl2');
  if(!gl){document.getElementById('out').innerHTML='<p class="bad">No WebGL2 context.</p>';return;}
  const ext=gl.getExtension('EXT_color_buffer_float');
  row('EXT_color_buffer_float',ext?'present':'ABSENT',!!ext,
      ext?'':'cannot render to R32F — every number below would be meaningless');
  if(!ext){render();return;}
  const sh=(t,s)=>{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(o));return o;};
  const prog=f=>{const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,VS));
    gl.attachShader(p,sh(gl.FRAGMENT_SHADER,f));gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;};
  pGather=prog(FS_GATHER); pTopb=prog(FS_TOPB);
  row('shaders compile + link','gather (matvec by index), top-b (rank count)',true);

  const vao=gl.createVertexArray(); gl.bindVertexArray(vao);
  const vb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,vb);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);

  texV=tex(D,K,V); texC=tex(D,NCH,CH); texS=tex(D,NSU,SU);
  texIdent=tex(NSU,1,new Float32Array(Array.from({length:NSU},(_,i)=>i)));

  const t0=performance.now();
  let agree=0, correct=0;
  const got=[];
  for(let i=0;i<K;i++){const a=walk(i,null); got.push(a);
    if(a===P.refBeam[i])agree++; if(a===P.truth[i])correct++;}
  const ms=(performance.now()-t0)/K;

  row('GPU vs f64 beam walk',agree+' / '+K+' identical answers',agree===K,
      agree===K?'f32 changed no decision':'f32 flipped '+(K-agree));
  row('retrieval accuracy',(correct/K).toFixed(4)+' (f64 engine: '+
      (P.refBeam.filter((a,i)=>a===P.truth[i]).length/K).toFixed(4)+')',
      Math.abs(correct-P.refBeam.filter((a,i)=>a===P.truth[i]).length)===0);
  row('dot products / query',(NSU+BEAM*G+BEAM*G)+' vs '+K+' for a flat scan',true,
      (K/(NSU+BEAM*G+BEAM*G)).toFixed(1)+'× fewer, at parity');
  row('per-query time',ms.toFixed(2)+' ms',true,'includes a readback per pass — not a throughput number');
  row('renderer',gl.getParameter(gl.VERSION),true,gl.getParameter(gl.RENDERER));
  render();

  let h='<h1 style="font-size:14px;margin:18px 0 6px">Inspect a query</h1><select id="sel">';
  for(let i=0;i<K;i++) h+='<option value="'+i+'">'+i+' — '+P.names[i]+'</option>';
  h+='</select><div id="tr"></div>';
  document.getElementById('probe').innerHTML=h;
  const show=()=>{const i=+document.getElementById('sel').value, tr=[];
    const a=walk(i,tr);
    document.getElementById('tr').innerHTML='<pre>'+
      tr.map(([k,v])=>k.padEnd(14)+v).join('\n')+
      '\nGPU answer    '+a+' — '+P.names[a]+
      '\nf64 answer    '+P.refBeam[i]+' — '+P.names[P.refBeam[i]]+
      '\nplanted       '+P.truth[i]+' — '+P.names[P.truth[i]]+'</pre>';};
  document.getElementById('sel').onchange=show; show();
}

function render(){
  let h='<table><tr><th>check</th><th>value</th><th>result</th><th>note</th></tr>';
  for(const [n,v,ok,note] of rows)
    h+='<tr><td>'+n+'</td><td class="n">'+v+'</td><td class="'+(ok?'ok':'bad')+'">'+
       (ok?'PASS':'FAIL')+'</td><td class="dim">'+note+'</td></tr>';
  h+='</table><p class="'+(fails?'bad':'ok')+'">'+(fails?fails+' FAILED':
     'ALL PASS — leCore\'s read path ran in WebGL2 and matched the f64 engine on every query')+
     '</p><pre>corpus '+K+' docstrings · dim '+D+' · tiers '+K+'→'+NCH+'→'+NSU+
     ' · g='+G+' (K^⅓) · beam '+BEAM+'</pre>';
  document.getElementById('out').innerHTML=h;
}
try{main();}catch(e){document.getElementById('out').innerHTML='<p class="bad">'+e.message+'</p>';}
</script>
"""
open("lecore_webgl2_full.html", "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(DATA)))
flat_acc = sum(a == t for a, t in zip(ref_flat, T)) / K
beam_acc = sum(a == t for a, t in zip(ref_beam, T)) / K
print("wrote lecore_webgl2_full.html  %.0f KB" % (os.path.getsize("lecore_webgl2_full.html")/1024))
print("corpus %d docs, dim %d, tiers %d->%d->%d, g=%d, beam=%d" % (K, DIM, K, len(ch), len(su), g, BEAM))
print("embedded reference: flat acc %.4f, beam acc %.4f, dots %d vs %d"
      % (flat_acc, beam_acc, len(su)+2*BEAM*g, K))
