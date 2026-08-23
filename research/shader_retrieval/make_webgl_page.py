"""Generate a SINGLE-FILE, SELF-VERIFYING WebGL2 page that runs leCore's VSA read path.

The page carries its own reference data: atoms and expected f64 results are computed HERE by
the authoritative engine and embedded as base64 Float32 blobs, so opening the file in a browser
is a DIFFERENTIAL TEST, not a demo. It prints a PASS/FAIL table and the T1 gate verdict.
No network, no build step, no dependencies -- a 4k-intro shape: the page IS the experiment.
"""
import base64, json
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, unitary_vector

D, K, TARGET, TILE = 256, 32, 7, 8
rng = np.random.default_rng(0)
k = unitary_vector(D, rng)
V = np.stack([unitary_vector(D, rng) for _ in range(K)])
x = V[TARGET]
ki = np.empty_like(k); ki[0] = k[0]; ki[1:] = k[1:][::-1]

ref_bind = bind(x, k)
ref_probe = unbind(ref_bind, k)
ref_scores = V @ ref_probe
srt = np.sort(ref_scores)[::-1]
margin = float(srt[0] - srt[1])

def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()

DATA = dict(D=D, K=K, TILE=TILE, target=TARGET,
            argmax=int(np.argmax(ref_scores)), margin=margin,
            k=b64(k), ki=b64(ki), x=b64(x), V=b64(V.ravel()),
            refBind=b64(ref_bind), refProbe=b64(ref_probe), refScores=b64(ref_scores))

HTML = """<!doctype html>
<meta charset="utf-8">
<title>leCore VSA read path — WebGL2 differential test</title>
<style>
 body{background:#0b0d10;color:#cfd6e4;font:14px/1.55 ui-monospace,Menlo,Consolas,monospace;
      margin:0;padding:28px 32px;max-width:900px}
 h1{font-size:17px;color:#e8eefc;margin:0 0 4px} p.sub{color:#7d8798;margin:0 0 20px}
 table{border-collapse:collapse;width:100%;margin:10px 0 18px}
 td,th{padding:6px 10px;border-bottom:1px solid #1c2129;text-align:left}
 th{color:#7d8798;font-weight:400}
 .ok{color:#5fd38d} .bad{color:#ff6b6b} .n{color:#8ab4ff}
 pre{background:#12151b;padding:12px;border-radius:6px;overflow:auto;color:#9aa5b6}
</style>
<h1>leCore — bind · score · tiled-argmax, executed as WebGL2 fragment shaders</h1>
<p class="sub">Reference values were computed by the f64 engine and embedded. This page is a
differential test, not a demo. Fragment shaders and texelFetch only — no compute, no SSBOs.</p>
<div id="out">running…</div>
<script id="DATA" type="application/json">__DATA__</script>
<script>
const D_ = JSON.parse(document.getElementById('DATA').textContent);
const dec = s => { const b = atob(s), u = new Uint8Array(b.length);
  for (let i=0;i<b.length;i++) u[i]=b.charCodeAt(i); return new Float32Array(u.buffer); };
const kArr=dec(D_.k), kiArr=dec(D_.ki), xArr=dec(D_.x), VArr=dec(D_.V);
const refBind=dec(D_.refBind), refProbe=dec(D_.refProbe), refScores=dec(D_.refScores);
const D=D_.D, K=D_.K, TILE=D_.TILE;

const VS = `#version 300 es
in vec2 p; void main(){ gl_Position = vec4(p,0.0,1.0); }`;

// bind(x,k): circulant GATHER, k[(j-i) mod D]. The matrix is never materialised.
const FS_BIND = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uX; uniform sampler2D uK; uniform int uD;
out vec4 o;
void main(){ int j=int(gl_FragCoord.x); float a=0.0;
  for(int i=0;i<uD;++i){ int q=j-i; if(q<0) q+=uD;
    a += texelFetch(uK,ivec2(q,0),0).r * texelFetch(uX,ivec2(i,0),0).r; }
  o=vec4(a,0.0,0.0,1.0); }`;

// codebook scores: one fragment per atom, the dot product is the fragment's loop.
const FS_SCORE = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uV; uniform sampler2D uQ; uniform int uD;
out vec4 o;
void main(){ int r=int(gl_FragCoord.x); float a=0.0;
  for(int i=0;i<uD;++i){ a += texelFetch(uV,ivec2(i,r),0).r * texelFetch(uQ,ivec2(i,0),0).r; }
  o=vec4(a,0.0,0.0,1.0); }`;

// tiled max reduction — T4 (tiled_max_eq_global) says regrouping cannot move the answer.
const FS_MAX = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uS; uniform int uN; uniform int uTile;
out vec4 o;
void main(){ int t=int(gl_FragCoord.x); float b=-1e30;
  for(int i=0;i<uTile;++i){ int j=t*uTile+i;
    if(j<uN) b=max(b, texelFetch(uS,ivec2(j,0),0).r); }
  o=vec4(b,0.0,0.0,1.0); }`;

const rows=[]; let fails=0;
function row(name, val, ok, note){ rows.push([name,val,ok,note||'']); if(ok===false) fails++; }

function main(){
  const cv=document.createElement('canvas');
  const gl=cv.getContext('webgl2');
  if(!gl){ document.getElementById('out').innerHTML='<p class="bad">No WebGL2 context.</p>'; return; }
  // Rendering to a 32-bit float target needs this extension in WebGL2. If absent, say so
  // rather than silently falling back to half precision and reporting a bogus error figure.
  const ext=gl.getExtension('EXT_color_buffer_float');
  row('EXT_color_buffer_float', ext?'present':'ABSENT', !!ext,
      ext?'':'cannot render to R32F — results below would be meaningless');
  if(!ext){ render(); return; }

  const sh=(t,s)=>{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o)); return o;};
  const prog=fs=>{const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,VS));
    gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p;};
  const pBind=prog(FS_BIND), pScore=prog(FS_SCORE), pMax=prog(FS_MAX);
  row('shaders compile + link','bind, score, tiled-max',true);

  const vao=gl.createVertexArray(); gl.bindVertexArray(vao);
  const vb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,vb);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);

  const tex=(w,h,data)=>{const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.R32F,w,h,0,gl.RED,gl.FLOAT,data||null); return t;};

  function run(p, outW, binds, ints){
    const out=tex(outW,1,null);
    const fb=gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER,fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,out,0);
    gl.useProgram(p); gl.viewport(0,0,outW,1);
    binds.forEach(([n,t],u)=>{gl.activeTexture(gl.TEXTURE0+u);gl.bindTexture(gl.TEXTURE_2D,t);
      gl.uniform1i(gl.getUniformLocation(p,n),u);});
    for(const kk in ints) gl.uniform1i(gl.getUniformLocation(p,kk),ints[kk]);
    gl.drawArrays(gl.TRIANGLES,0,3);
    const px=new Float32Array(outW); gl.readPixels(0,0,outW,1,gl.RED,gl.FLOAT,px);
    return px;
  }

  const tX=tex(D,1,xArr), tK=tex(D,1,kArr), tKi=tex(D,1,kiArr), tV=tex(D,K,VArr);
  const gB=run(pBind,D,[['uX',tX],['uK',tK]],{uD:D});
  const tB=tex(D,1,gB);
  const gP=run(pBind,D,[['uX',tB],['uK',tKi]],{uD:D});
  const tQ=tex(D,1,gP);
  const gS=run(pScore,K,[['uV',tV],['uQ',tQ]],{uD:D});
  const tS=tex(K,1,gS);
  const gM=run(pMax,Math.ceil(K/TILE),[['uS',tS]],{uN:K,uTile:TILE});

  const relerr=(a,b)=>{let m=0,s=0;for(let i=0;i<b.length;i++){m=Math.max(m,Math.abs(a[i]-b[i]));
    s=Math.max(s,Math.abs(b[i]));} return m/s;};
  const abserr=(a,b)=>{let m=0;for(let i=0;i<b.length;i++)m=Math.max(m,Math.abs(a[i]-b[i]));return m;};
  const amax=a=>{let bi=0;for(let i=1;i<a.length;i++)if(a[i]>a[bi])bi=i;return bi;};

  const eB=relerr(gB,refBind), eP=relerr(gP,refProbe), eS=abserr(gS,refScores);
  row('bind  GPU vs f64 rFFT', eB.toExponential(3)+' rel', eB<1e-5);
  row('unbind GPU vs f64',      eP.toExponential(3)+' rel', eP<1e-5);
  row('scores GPU vs f64',      eS.toExponential(3)+' abs', eS<1e-5);

  let tiled=-1e30; for(const v of gM) tiled=Math.max(tiled,v);
  let single=-1e30; for(const v of gS) single=Math.max(single,v);
  row('tiled max == single pass (T4)', tiled.toFixed(6)+' vs '+single.toFixed(6),
      Math.abs(tiled-single)<1e-6);

  const ag=amax(gS);
  row('ARGMAX', 'gpu='+ag+'  f64='+D_.argmax+'  planted='+D_.target, ag===D_.argmax);
  const safety=D_.margin/(2*eS);
  row('T1 gate (margin > 2·eps)',
      'margin '+D_.margin.toFixed(6)+' vs 2·eps '+(2*eS).toExponential(3)+
      '  → safety ×'+safety.toFixed(0), D_.margin>2*eS, 'gate '+(D_.margin>2*eS?'ANSWERS':'ABSTAINS'));
  row('renderer', gl.getParameter(gl.VERSION)+' / '+gl.getParameter(gl.SHADING_LANGUAGE_VERSION), true);
  render();
}

function render(){
  let h='<table><tr><th>check</th><th>value</th><th>result</th><th>note</th></tr>';
  for(const [n,v,ok,note] of rows)
    h+='<tr><td>'+n+'</td><td class="n">'+v+'</td><td class="'+(ok?'ok':'bad')+'">'+
       (ok?'PASS':'FAIL')+'</td><td>'+note+'</td></tr>';
  h+='</table><p class="'+(fails?'bad':'ok')+'">'+
     (fails? fails+' FAILED' : 'ALL PASS — the VSA read path ran in WebGL2 and matched the f64 engine')+
     '</p><pre>D='+D+'  K='+K+'  tile='+TILE+'  — fragment shaders + texelFetch only</pre>';
  document.getElementById('out').innerHTML=h;
}
try { main(); } catch(e) {
  document.getElementById('out').innerHTML='<p class="bad">'+e.message+'</p>'; }
</script>
"""
open("lecore_webgl2.html", "w", encoding="utf-8").write(
    HTML.replace("__DATA__", json.dumps(DATA)))
import os
print("wrote lecore_webgl2.html  %.1f KB   (D=%d K=%d, f64 argmax=%d, margin=%.6f)"
      % (os.path.getsize("lecore_webgl2.html") / 1024, D, K, int(np.argmax(ref_scores)), margin))
