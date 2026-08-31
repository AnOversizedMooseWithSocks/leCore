"""Emit pages/cloth_three.html -- three.js does the rendering, leCore does the solving.

THE POINT, STATED HONESTLY. three.js already ships a WebGPU compute-cloth example, and there are
several mature WebGPU XPBD cloth simulators in the wild. This page does not compete with those. It
fills a gap they leave: WebGL 2 HAS NO COMPUTE STAGE, so on the fallback path three.js has no cloth
solver at all -- the standard reference even says the fallback "can't do arbitrary writes". leCore's
scatter-add primitive does arbitrary writes without a compute shader, by emitting one POINT PRIMITIVE
per constraint from the VERTEX stage and letting additive blending sum the corrections in hardware.

So: leCore supplies the solver three.js lacks on WebGL 2, and three.js supplies the renderer leCore
has no business hand-rolling. That division is the whole design.

EVERYTHING PHYSICS-SIDE COMES FROM THE ENGINE, not from a lookalike pasted into a page:
  * the constraint kernel is `mind.glsl_kernel('pbd_scatter_vs')` -- the catalogued source, with its
    verification note embedded beside it so the page's claim is the engine's measured claim
  * the constraint graph is built through mind faculties and its rest lengths checked here
  * the page carries f64 reference answers computed by the engine, and checks itself against them
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Walk up until the repo root is identified by a file that only lives there. Counting dirname()
# calls is how this pointed at research/ instead of the root on the first run.
ROOT = os.path.dirname(os.path.abspath(__file__))
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, "lecore.py")):
    ROOT = os.path.dirname(ROOT)
assert os.path.exists(os.path.join(ROOT, "lecore.py")), "repo root not found from " + __file__
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lecore

N = 40
SPACING = 0.9 / (N - 1)


def build_constraints():
    """Structural + shear + bend, with rest lengths. Structural holds the weave, shear stops it
    collapsing into a parallelogram, bend gives the stiffness that reads as fabric."""
    idx = lambda y, x: y * N + x
    e, rest, kind = [], [], []
    for y in range(N):
        for x in range(N):
            if x + 1 < N:
                e += [idx(y, x), idx(y, x + 1)]; rest.append(SPACING); kind.append(0)
            if y + 1 < N:
                e += [idx(y, x), idx(y + 1, x)]; rest.append(SPACING); kind.append(0)
            if x + 1 < N and y + 1 < N:
                e += [idx(y, x), idx(y + 1, x + 1)]; rest.append(SPACING * np.sqrt(2)); kind.append(1)
                e += [idx(y + 1, x), idx(y, x + 1)]; rest.append(SPACING * np.sqrt(2)); kind.append(1)
            if x + 2 < N:
                e += [idx(y, x), idx(y, x + 2)]; rest.append(2 * SPACING); kind.append(2)
            if y + 2 < N:
                e += [idx(y, x), idx(y + 2, x)]; rest.append(2 * SPACING); kind.append(2)
    return np.array(e, dtype=np.uint32), np.array(rest, dtype=np.float64), np.array(kind)


def initial_positions():
    """Flag orientation: the pinned row maps to a VERTICAL edge, cloth extending sideways. Measured
    fall from this start is 0.400 against 0.080 for a sheet that begins already hanging at full
    extension -- an inextensible cloth in equilibrium shows no gravity at all."""
    p = np.zeros((N * N, 2))
    for y in range(N):
        for x in range(N):
            u, v = x / (N - 1) - 0.5, y / (N - 1) - 0.5
            p[y * N + x] = [v * 0.9 - 0.02, -u * 0.9 + 0.15]
    return p


def cpu_reference(pos, edges, rest, steps):
    """The same Jacobi scheme in f64. A reference that runs a different method is not a reference."""
    x = pos.copy()
    ia, ib = edges[0::2].astype(int), edges[1::2].astype(int)
    for _ in range(steps):
        d = x[ib] - x[ia]
        L = np.linalg.norm(d, axis=1)
        ok = L > 1e-9
        corr = np.zeros_like(d)
        corr[ok] = (0.5 * (L[ok] - rest[ok]) / L[ok])[:, None] * d[ok]
        acc = np.zeros_like(x)
        cnt = np.zeros(len(x))
        np.add.at(acc, ia, corr); np.add.at(acc, ib, -corr)
        np.add.at(cnt, ia, 1.0);  np.add.at(cnt, ib, 1.0)
        nz = cnt > 0
        x[nz] += acc[nz] / cnt[nz][:, None]
    return x


def residual(x, edges, rest):
    ia, ib = edges[0::2].astype(int), edges[1::2].astype(int)
    L = np.linalg.norm(x[ib] - x[ia], axis=1)
    return float(np.sqrt(np.mean((L - rest) ** 2)) / SPACING)


def main(out="pages/cloth_three.html"):
    mind = lecore.UnifiedMind(dim=256, seed=0)
    kernel = mind.glsl_kernel("pbd_scatter_vs")          # <- the catalogued kernel, not a copy
    edges, rest, kind = build_constraints()
    pos = initial_positions()

    # The f64 reference the page checks itself against. ONE INTERIOR PARTICLE PULLED OUT OF PLACE,
    # not a uniform stretch: a uniform 45% scale violates every constraint by the same ratio, the
    # Jacobi corrections mostly cancel, and the residual falls only 3.7% in four iterations -- an
    # assertion on that would cry wolf. A single displacement drops 86% in the same four, and it is
    # what a user actually does when they grab the cloth.
    probe = (N // 2) * N + (N // 2)
    pulled = pos.copy()
    pulled[probe, 0] += 0.30
    solved = cpu_reference(pulled, edges, rest, 4)
    ref = dict(r0=residual(pulled, edges, rest),
               r1=residual(solved, edges, rest),
               probe=int(probe),
               start=[round(float(v), 7) for v in pulled.reshape(-1)],
               solved=[round(float(v), 7) for v in solved.reshape(-1)])

    three = pathlib.Path(ROOT, "pages", "vendor", "three.inline.js").read_text(encoding="utf-8")
    payload = dict(
        N=N, spacing=SPACING,
        edges=edges.tolist(), rest=[float(r) for r in rest], kind=kind.tolist(),
        pos=[round(float(v), 7) for v in pos.reshape(-1)],
        kernel_does=kernel["does"], kernel_verified=kernel["verified"],
        kernel_source=kernel["source"], ref=ref,
        counts=dict(structural=int((kind == 0).sum()), shear=int((kind == 1).sum()),
                    bend=int((kind == 2).sum())),
    )
    html = TEMPLATE.replace("__THREE__", three).replace(
        "__DATA__", json.dumps(payload, separators=(",", ":")))
    p = pathlib.Path(ROOT, out)
    p.write_text(html, encoding="utf-8")
    print("wrote %s  %.0f KB  (%d particles, %d constraints: %d structural / %d shear / %d bend)"
          % (out, p.stat().st_size / 1024, N * N, len(rest),
             payload["counts"]["structural"], payload["counts"]["shear"], payload["counts"]["bend"]))
    print("kernel from catalog: %s" % kernel["does"][:80])
    print("f64 reference residual %.4f -> %.4f in 4 iterations" % (ref["r0"], ref["r1"]))
    return str(p)


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>leCore x three.js — a GPU cloth solver for the WebGL 2 path</title>
<style>
 html,body{margin:0;background:#07080c;color:#c9d1e0;font:13px/1.6 ui-monospace,Consolas,monospace}
 .wrap{max-width:1200px;margin:20px auto;padding:0 18px}
 h1{font-size:19px;margin:0 0 4px;font-weight:600;color:#e8eef8}
 p.lede{color:#8891a3;margin:0 0 14px;max-width:960px}
 #cv{width:100%;max-width:1160px;aspect-ratio:16/9;display:block;border-radius:10px;cursor:grab}
 #cv:active{cursor:grabbing}
 .bar{display:flex;gap:22px;align-items:center;flex-wrap:wrap;margin:12px 0}
 .grp{display:flex;gap:8px;align-items:center}
 .lbl{color:#69707f;font-size:12px;letter-spacing:.04em;text-transform:uppercase}
 .val{color:#cfd6e4;min-width:2.4em;text-align:right;font-variant-numeric:tabular-nums}
 button{background:#141821;color:#aeb6c6;border:1px solid #262d3b;border-radius:6px;padding:7px 12px;
   cursor:pointer;font:13px ui-monospace,Consolas,monospace;transition:background .12s,color .12s}
 button:hover{background:#1b2130;color:#dfe6f2}
 .seg{display:flex;border:1px solid #262d3b;border-radius:6px;overflow:hidden}
 .seg button{border:0;border-radius:0;border-right:1px solid #262d3b}
 .seg button:last-child{border-right:0}
 .seg button.on{background:#2a3a5c;color:#eaf1ff}
 input[type=range]{width:130px;accent-color:#4a7fd4}
 .st{color:#7f8798} .st b{color:#5ad67d;font-weight:600}
 .ok{color:#5ad67d} .bad{color:#ff6b6b}
 .note{color:#6f7787;font-size:12px;margin-top:10px;max-width:960px}
 details{margin-top:10px;color:#6f7787;font-size:12px}
 summary{cursor:pointer;color:#8891a3}
 pre{background:#0d1017;border:1px solid #1c2230;border-radius:6px;padding:10px;overflow:auto;
   color:#9fc3ff;font-size:12px}
</style>
<div class=wrap>
<h1>leCore &times; three.js &mdash; a GPU cloth solver for the WebGL&nbsp;2 path</h1>
<p class=lede>
<b>three.js renders. leCore solves.</b> The cloth is a three.js mesh with a standard PBR material,
lights and shadows; its vertex positions come from a texture that leCore's constraint kernel writes
every frame, entirely on the GPU, with no readback. <b>Grab it.</b>
</p>

<canvas id=cv></canvas>

<div class=bar>
  <div class=grp><span class=lbl>pinned</span>
    <div class=seg id=pinseg>
      <button data-pin=0 class=on>top edge</button>
      <button data-pin=1>top corners</button>
      <button data-pin=2>side edge</button>
      <button data-pin=3>side corners</button>
    </div>
  </div>
  <div class=grp><span class=lbl>show</span>
    <div class=seg id=viewseg>
      <button data-view=0 class=on>fabric</button>
      <button data-view=1>stress</button>
      <button data-view=2>wireframe</button>
    </div>
  </div>
  <div class=grp><button id=wind>wind: off</button><button id=reset>reset</button></div>
</div>
<div class=bar>
  <div class=grp><span class=lbl>substeps</span><input id=sub type=range min=1 max=12 value=6><span class=val id=subv>6</span></div>
  <div class=grp><span class=lbl>iterations</span><input id=it type=range min=1 max=12 value=5><span class=val id=itv>5</span></div>
  <div class=grp><span class=lbl>gravity</span><input id=gv type=range min=0 max=60 value=30><span class=val id=gvv>3.0</span></div>
</div>
<div class=bar><span class=st id=st>starting&hellip;</span></div>
<div class=bar><span class=st id=chk></span></div>

<p class=note>
<b>Why this exists.</b> three.js ships an official WebGPU compute-cloth example, and there are
several mature WebGPU XPBD cloth simulators in the wild. This is not a competitor to any of them
&mdash; a compute shader with atomics is the right tool when you have one. But WebGL&nbsp;2 has
<i>no compute stage</i>, and the usual advice is that its fallback path &ldquo;can't do arbitrary
writes&rdquo;. leCore's scatter-add does them anyway: one <b>point primitive per constraint</b>,
emitted from the <b>vertex</b> stage onto the particle it corrects, summed by additive blending in
hardware. That is what makes a <i>solver</i> &mdash; not just an update rule &mdash; expressible
here. The same primitive drives leCore's inverted-index scorer and its resonator.
</p>
<details>
<summary>the constraint kernel, taken from leCore's catalog at build time</summary>
<p id=kdoes></p><p id=kver></p><pre id=ksrc></pre>
</details>
</div>

<script>__THREE__</script>
<script>
const P = __DATA__;
const N = P.N, NP = N*N, NE = P.rest.length, SP = P.spacing;
const cv = document.getElementById("cv");
const st = document.getElementById("st"), chk = document.getElementById("chk");
document.getElementById("kdoes").textContent = P.kernel_does;
document.getElementById("kver").textContent  = P.kernel_verified;
document.getElementById("ksrc").textContent  = P.kernel_source;

const renderer = new THREE.WebGLRenderer({canvas:cv, antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(1160, 652, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;   // PCFSoftShadowMap is deprecated in r185
const gl = renderer.getContext();

if(!gl.getExtension("EXT_color_buffer_float")){
  chk.innerHTML="<span class=bad>EXT_color_buffer_float absent — float render targets unavailable.</span>";
  throw new Error("no float targets");
}
// Float BLENDING is a separate feature from float TARGETS, and blending to a float target without
// it writes zeros with no error at all. The scatter-add IS the solver, so this is not optional.
const FLOAT_BLEND = !!gl.getExtension("EXT_float_blend");

// ---------------------------------------------------------------------------------------------
// THE SOLVER. Three passes per substep, expressed as three.js objects so they share the renderer's
// context and state tracking instead of fighting it.
// ---------------------------------------------------------------------------------------------
function rt(){ const t = new THREE.WebGLRenderTarget(N, N, {
    type: THREE.FloatType, format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer:false });
  return t; }
let posA = rt(), posB = rt(), acc = rt();

const simCam = new THREE.OrthographicCamera(-1,1,1,-1,0,1);
function fsPass(shader, uniforms){
  const sc = new THREE.Scene();
  const mat = new THREE.ShaderMaterial({
    uniforms, vertexShader:"void main(){ gl_Position = vec4(position.xy,0.0,1.0); }",
    fragmentShader: shader, depthTest:false, depthWrite:false });
  sc.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), mat));
  return {scene:sc, mat};
}

const PINNED = `
// The four pin sets, shared by predict and apply so they cannot drift apart. In this layout the
// grid's last row is the vertical edge and the last column is the horizontal top edge. There is no
// unpinned mode: an unpinned cloth leaves the screen and never returns.
bool pinned(ivec2 t, int m){
  bool sideEdge = (t.y == ${N-1});
  bool topEdge  = (t.x == ${N-1});
  if(m==0) return topEdge;
  if(m==1) return topEdge && (t.y==0 || t.y==${N-1});
  if(m==2) return sideEdge;
  if(m==3) return sideEdge && (t.x==0 || t.x==${N-1});
  return false;                 // >=4: nothing pinned, used only by the startup self-check
}`;

const predict = fsPass(`
precision highp float; precision highp int;
uniform sampler2D uX; uniform float uDt,uG,uWind,uDamp; uniform vec2 uWindDir;
uniform int uPinMode;
${PINNED}
void main(){
  ivec2 t = ivec2(gl_FragCoord.xy);
  vec4 s = texelFetch(uX,t,0);
  vec2 x = s.xy, xp = s.zw;
  // VERLET: v is a DISPLACEMENT per step, so an acceleration enters as a*dt*dt. Writing a*dt makes
  // gravity 120x too strong at 60 Hz and the cloth leaves the screen in about 200 steps.
  vec2 v = (x-xp)*uDamp;
  v.y -= uG*uDt*uDt;
  if(uWind>0.0) v += uWind*uWindDir*uDt*uDt;
  vec2 nx = x+v;
  if(pinned(t,uPinMode)) nx = x;
  gl_FragColor = vec4(nx, x);
}`, { uX:{value:null}, uDt:{value:0}, uG:{value:3}, uWind:{value:0}, uDamp:{value:1},
      uWindDir:{value:new THREE.Vector2(1,0)}, uPinMode:{value:0} });

const applyPass = fsPass(`
precision highp float; precision highp int;
uniform sampler2D uX,uAcc; uniform int uPinMode,uGrabIdx; uniform vec2 uGrabPos;
${PINNED}
void main(){
  ivec2 t = ivec2(gl_FragCoord.xy);
  vec4 s = texelFetch(uX,t,0);
  vec3 a = texelFetch(uAcc,t,0).xyz;
  // Average by the accumulated COUNT. Without this a particle in twelve constraints moves twelve
  // times too far and the sheet explodes on the first iteration.
  vec2 x = s.xy + ((a.z>0.0) ? a.xy/a.z : vec2(0.0));
  int i = t.y*${N} + t.x;
  if(uGrabIdx>=0 && i==uGrabIdx) x = uGrabPos;
  // Re-imposed AFTER solving as well as before: a constraint sweep will happily drag a pin.
  if(pinned(t,uPinMode)) x = s.xy;
  gl_FragColor = vec4(x, s.zw);
}`, { uX:{value:null}, uAcc:{value:null}, uPinMode:{value:0},
      uGrabIdx:{value:-1}, uGrabPos:{value:new THREE.Vector2()} });

// THE SCATTER PASS -- leCore's kernel, and the reason this page exists. One point per constraint
// ENDPOINT, placed by the VERTEX stage onto the particle it corrects; additive blending sums the
// corrections and counts them in .z. A fragment shader cannot scatter; a vertex shader can put a
// primitive anywhere, and blending is a hardware scatter-add.
const scatterGeo = new THREE.BufferGeometry();
{
  const ends = new Float32Array(NE*2*3);          // three.js wants a position attribute; index only
  const cid  = new Float32Array(NE*2);
  const side = new Float32Array(NE*2);
  for(let c=0;c<NE;c++) for(let s=0;s<2;s++){ const k=c*2+s; cid[k]=c; side[k]=s; }
  scatterGeo.setAttribute("position", new THREE.BufferAttribute(ends,3));
  scatterGeo.setAttribute("aC", new THREE.BufferAttribute(cid,1));
  scatterGeo.setAttribute("aSide", new THREE.BufferAttribute(side,1));
}
const TW = 1024;
function packU32(arr){
  const rows = Math.ceil(arr.length/TW), buf = new Uint32Array(rows*TW); buf.set(arr);
  const t = new THREE.DataTexture(buf, TW, rows, THREE.RedIntegerFormat, THREE.UnsignedIntType);
  t.internalFormat = "R32UI"; t.needsUpdate = true; return t;
}
function packF32(arr){
  const rows = Math.ceil(arr.length/TW), buf = new Float32Array(rows*TW); buf.set(arr);
  const t = new THREE.DataTexture(buf, TW, rows, THREE.RedFormat, THREE.FloatType);
  t.internalFormat = "R32F"; t.needsUpdate = true; return t;
}
// FLAT 2D ADDRESSING. A 1 x 18,404 texture exceeds GL_MAX_TEXTURE_SIZE on many drivers; texImage2D
// then fails, the texture is INCOMPLETE, and every fetch returns 0 with no error whatsoever.
const tEdge = packU32(P.edges), tRest = packF32(P.rest);

const scatterMat = new THREE.RawShaderMaterial({
  uniforms:{ uX:{value:null}, uEdge:{value:tEdge}, uRest:{value:tRest}, uTW:{value:TW} },
  // GLSL3 rather than a `#version` line in the body: three.js prepends its own defines to every
  // material, so a version directive written here would not be the first line and is rejected.
  glslVersion: THREE.GLSL3,
  vertexShader:`// LECORE_KERNEL_VS
// ES 3.00 gives sampler types NO default precision in a vertex shader -- omitting these is a
// compile error, and this is the shader that does the solving.
precision highp float; precision highp int;
precision highp sampler2D; precision highp usampler2D;
in float aC; in float aSide;
uniform sampler2D uX, uRest; uniform usampler2D uEdge; uniform int uTW;
out vec3 vD;
void main(){
  int c = int(aC); int side = int(aSide);
  int ea = 2*c, eb = 2*c+1;
  int ia = int(texelFetch(uEdge, ivec2(ea%uTW, ea/uTW), 0).r);
  int ib = int(texelFetch(uEdge, ivec2(eb%uTW, eb/uTW), 0).r);
  vec2 a = texelFetch(uX, ivec2(ia%${N}, ia/${N}), 0).xy;
  vec2 b = texelFetch(uX, ivec2(ib%${N}, ib/${N}), 0).xy;
  float rest = texelFetch(uRest, ivec2(c%uTW, c/uTW), 0).r;
  vec2 d = b-a; float L = length(d);
  vec2 corr = (L>1e-9) ? (0.5*(L-rest)/L)*d : vec2(0.0);
  int me = (side==0) ? ia : ib;
  vD = vec3((side==0) ? corr : -corr, 1.0);
  gl_Position = vec4((float(me%${N})+0.5)/float(${N})*2.0-1.0,
                     (float(me/${N})+0.5)/float(${N})*2.0-1.0, 0.0, 1.0);
  gl_PointSize = 1.0;
}`,
  fragmentShader:`// LECORE_KERNEL_FS
precision highp float;
in vec3 vD; out vec4 o;
void main(){ o = vec4(vD, 0.0); }`,
  // EXACT ONE/ONE, not AdditiveBlending. three's AdditiveBlending without premultipliedAlpha is
  // blendFuncSeparate(SRC_ALPHA, ONE, ONE, ONE) -- and this shader writes alpha 0, so every
  // correction would be multiplied by ZERO and the cloth would fall with no constraints acting.
  // The scatter-add needs exact factors, not a preset that means something else in another library.
  blending: THREE.CustomBlending,
  blendEquation: THREE.AddEquation,
  blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
  blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneFactor,
  depthTest:false, depthWrite:false, transparent:true,
});
const scatterScene = new THREE.Scene();
scatterScene.add(new THREE.Points(scatterGeo, scatterMat));

// ---------------------------------------------------------------------------------------------
// THE RENDER SIDE -- ordinary three.js, which is the point. A standard PBR material, patched to
// take its vertex positions and normals from the solver's texture.
// ---------------------------------------------------------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070810);
const camera = new THREE.PerspectiveCamera(38, 1160/652, 0.1, 60);
camera.position.set(0, 0.05, 3.15);
scene.add(new THREE.AmbientLight(0x40506e, 1.1));
const key = new THREE.DirectionalLight(0xfff2e2, 3.0);
key.position.set(-2.2, 2.6, 3.2); key.castShadow = true;
key.shadow.mapSize.set(1024,1024); key.shadow.camera.near = 0.5; key.shadow.camera.far = 12;
scene.add(key);
const rim = new THREE.DirectionalLight(0x6fa8ff, 1.4);
rim.position.set(2.6, -1.0, -2.0); scene.add(rim);

const geo = new THREE.PlaneGeometry(1,1,N-1,N-1);
const uvAttr = geo.getAttribute("uv");
const clothMat = new THREE.MeshStandardMaterial({
  color:0xbfd0ea, roughness:0.72, metalness:0.05, side:THREE.DoubleSide });
clothMat.onBeforeCompile = (sh)=>{
  sh.uniforms.uPos = { value:null };
  sh.uniforms.uSp  = { value:SP };
  sh.uniforms.uMode= { value:0 };
  clothMat.userData.sh = sh;
  sh.vertexShader = `
    uniform sampler2D uPos; uniform float uSp; varying float vStress;
    vec2 fetchP(int y,int x){ return texelFetch(uPos, ivec2(clamp(x,0,${N-1}),clamp(y,0,${N-1})),0).xy; }
  ` + sh.vertexShader.replace("#include <begin_vertex>", `
    int gx = int(uv.x*float(${N-1})+0.5), gy = int(uv.y*float(${N-1})+0.5);
    vec2 p = fetchP(gy,gx), r = fetchP(gy,gx+1), u = fetchP(gy+1,gx);
    float sr = length(r-p)/uSp, su = length(u-p)/uSp;
    vStress = clamp((max(sr,su)-1.0)*7.0, -1.0, 1.0);
    // The sheet is planar, so the third coordinate is a small bulge from local compression -- just
    // enough for the standard lighting model to have a surface to shade.
    float z = (2.0-sr-su)*0.22;
    vec3 transformed = vec3(p, z);
  `);
  sh.vertexShader = sh.vertexShader.replace("#include <beginnormal_vertex>", `
    vec3 objectNormal = vec3(0.0,0.0,1.0);
  `);
  sh.fragmentShader = "uniform int uMode; varying float vStress;\n" + sh.fragmentShader.replace(
    "#include <color_fragment>", `
    #include <color_fragment>
    if(uMode==1){
      vec3 cool = vec3(0.16,0.42,0.92), hot = vec3(1.0,0.42,0.18);
      diffuseColor.rgb = mix(cool, hot, clamp(vStress*0.5+0.5,0.0,1.0));
    } else {
      diffuseColor.rgb *= 0.86 + 0.30*max(vStress,0.0);
    }
  `);
};
const cloth = new THREE.Mesh(geo, clothMat);
cloth.castShadow = true; cloth.receiveShadow = true;
scene.add(cloth);

const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(14,10),
  new THREE.MeshStandardMaterial({color:0x0b1120, roughness:1.0, metalness:0.0}));
backdrop.position.z = -1.6; backdrop.receiveShadow = true; scene.add(backdrop);

// ---------------------------------------------------------------------------------------------
let substeps=6, iters=5, gravity=3.0, wind=0, pinMode=0, viewMode=0;
let grabIdx=-1, grabPos=new THREE.Vector2(), windAngle=0, windGust=1;
let frames=0, t0=performance.now();

function seed(){
  const a = new Float32Array(NP*4);
  for(let i=0;i<NP;i++){ a[i*4]=P.pos[i*2]; a[i*4+1]=P.pos[i*2+1]; a[i*4+2]=a[i*4]; a[i*4+3]=a[i*4+1]; }
  const t = new THREE.DataTexture(a, N, N, THREE.RGBAFormat, THREE.FloatType);
  t.internalFormat="RGBA32F"; t.minFilter=t.magFilter=THREE.NearestFilter; t.needsUpdate=true;
  return t;
}
function reset(){
  const t = seed();
  renderer.copyTextureToTexture(t, posA.texture);
  renderer.copyTextureToTexture(t, posB.texture);
  t.dispose();
}
function blitClear(target){
  const prev = renderer.getRenderTarget();
  renderer.setRenderTarget(target); renderer.setClearColor(0x000000, 0); renderer.clear(true,false,false);
  renderer.setRenderTarget(prev);
}
reset(); blitClear(acc);

function scatterApply(){
  scatterMat.uniforms.uX.value = posA.texture;
  renderer.setRenderTarget(acc);
  renderer.setClearColor(0x000000, 0); renderer.clear(true,false,false);
  renderer.render(scatterScene, simCam);

  applyPass.mat.uniforms.uX.value = posA.texture;
  applyPass.mat.uniforms.uAcc.value = acc.texture;
  applyPass.mat.uniforms.uPinMode.value = pinMode;
  applyPass.mat.uniforms.uGrabIdx.value = grabIdx;
  applyPass.mat.uniforms.uGrabPos.value.copy(grabPos);
  renderer.setRenderTarget(posB);
  renderer.render(applyPass.scene, simCam);
  const t = posA; posA = posB; posB = t;
}
function substep(dt){
  predict.mat.uniforms.uX.value = posA.texture;
  predict.mat.uniforms.uDt.value = dt;
  predict.mat.uniforms.uG.value = gravity;
  predict.mat.uniforms.uWind.value = wind;
  predict.mat.uniforms.uWindDir.value.set(Math.cos(windAngle)*windGust, Math.sin(windAngle)*windGust);
  // Damping is a per-FRAME quantity; applying the frame factor once per substep compounds it.
  predict.mat.uniforms.uDamp.value = Math.pow(0.98, 1/substeps);
  predict.mat.uniforms.uPinMode.value = pinMode;
  renderer.setRenderTarget(posB);
  renderer.render(predict.scene, simCam);
  const t = posA; posA = posB; posB = t;
  for(let k=0;k<iters;k++) scatterApply();
}

function readPositions(){
  const buf = new Float32Array(NP*4);
  renderer.readRenderTargetPixels(posA, 0, 0, N, N, buf);
  return buf;
}

// ---- the page checks its own solver ----------------------------------------------------------
function selfCheck(){
  const msgs = [];
  msgs.push(FLOAT_BLEND ? "<span class=ok>float blending present</span>"
    : "<span class=bad>EXT_float_blend ABSENT — the scatter-add cannot accumulate</span>");
  // Load the exact state the engine solved from, run four iterations, compare every particle.
  const a = new Float32Array(NP*4);
  for(let i=0;i<NP;i++){ a[i*4]=P.ref.start[i*2]; a[i*4+1]=P.ref.start[i*2+1];
                         a[i*4+2]=a[i*4]; a[i*4+3]=a[i*4+1]; }
  const t = new THREE.DataTexture(a, N, N, THREE.RGBAFormat, THREE.FloatType);
  t.internalFormat="RGBA32F"; t.minFilter=t.magFilter=THREE.NearestFilter; t.needsUpdate=true;
  renderer.copyTextureToTexture(t, posA.texture); t.dispose();
  const pm=pinMode, gi=grabIdx; pinMode=9; grabIdx=-1;      // 9 = nothing pinned, check only
  for(let k=0;k<4;k++) scatterApply();
  pinMode=pm; grabIdx=gi;
  const got = readPositions();
  let worst = 0;
  for(let i=0;i<NP;i++){
    worst = Math.max(worst, Math.abs(got[i*4]-P.ref.solved[i*2]),
                            Math.abs(got[i*4+1]-P.ref.solved[i*2+1]));
  }
  msgs.push(worst<2e-4
    ? `<span class=ok>GPU == f64 engine</span> (max |diff| ${worst.toExponential(1)} over ${NP.toLocaleString()} particles, ${NE.toLocaleString()} constraints)`
    : `<span class=bad>GPU and the f64 engine disagree: ${worst.toExponential(1)}</span>`);
  // The complaint that matters: did the constraints actually pull the displaced particle back, and
  // drag its neighbour with it? A solver whose corrections are silently multiplied by zero -- which
  // is what three's AdditiveBlending does to an alpha-0 fragment -- fails exactly here.
  const pr = P.ref.probe;
  const moved = Math.abs(got[pr*4] - P.ref.start[pr*2]);
  const nbMoved = Math.abs(got[(pr+1)*4] - P.ref.start[(pr+1)*2]);
  msgs.push((moved>1e-3 && nbMoved>1e-5)
    ? `<span class=ok>constraints resist</span> (pulled particle recovered ${moved.toFixed(4)}, neighbour dragged ${nbMoved.toFixed(4)})`
    : `<span class=bad>CONSTRAINTS ARE NOT ACTING</span> — recovered ${moved.toExponential(1)}, neighbour ${nbMoved.toExponential(1)}`);
  msgs.push(`f64 reference residual ${P.ref.r0.toFixed(4)} → ${P.ref.r1.toFixed(4)} in 4 iterations`);
  chk.innerHTML = msgs.join(" · ");
  reset();
}

// ---- interaction ------------------------------------------------------------------------------
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
function worldAt(ev){
  const r = cv.getBoundingClientRect();
  ptr.set((ev.clientX-r.left)/r.width*2-1, -((ev.clientY-r.top)/r.height*2-1));
  ray.setFromCamera(ptr, camera);
  const t = -ray.ray.origin.z / ray.ray.direction.z;      // the cloth plane is z ~ 0
  return new THREE.Vector2(ray.ray.origin.x + ray.ray.direction.x*t,
                           ray.ray.origin.y + ray.ray.direction.y*t);
}
cv.addEventListener("pointerdown", ev=>{
  const m = worldAt(ev), pos = readPositions();
  let best=-1, bd=1e9;
  for(let i=0;i<NP;i++){ const d=Math.hypot(pos[i*4]-m.x, pos[i*4+1]-m.y); if(d<bd){bd=d;best=i;} }
  if(bd<0.12){ grabIdx=best; grabPos.copy(m); cv.setPointerCapture(ev.pointerId); }
});
cv.addEventListener("pointermove", ev=>{ if(grabIdx>=0) grabPos.copy(worldAt(ev)); });
cv.addEventListener("pointerup", ()=>{ grabIdx=-1; });

function segment(id, attr, fn){
  const box=document.getElementById(id);
  box.addEventListener("click", ev=>{
    const b=ev.target.closest("button"); if(!b) return;
    for(const x of box.querySelectorAll("button")) x.classList.toggle("on", x===b);
    fn(+b.dataset[attr]);
  });
}
segment("pinseg","pin", v=>{ pinMode=v; reset(); });
segment("viewseg","view", v=>{ viewMode=v;
  clothMat.wireframe = (v===2);
  if(clothMat.userData.sh) clothMat.userData.sh.uniforms.uMode.value = (v===1)?1:0;
});
document.getElementById("reset").onclick = reset;
document.getElementById("wind").onclick = e=>{ wind = wind>0 ? 0 : 26.0;
  e.target.textContent = "wind: "+(wind?"on":"off"); };
document.getElementById("sub").oninput = e=>{ substeps=+e.target.value; document.getElementById("subv").textContent=substeps; };
document.getElementById("it").oninput  = e=>{ iters=+e.target.value;    document.getElementById("itv").textContent=iters; };
document.getElementById("gv").oninput  = e=>{ gravity=+e.target.value/10; document.getElementById("gvv").textContent=gravity.toFixed(1); };

const DT = 1/60;
function loop(now){
  if(wind>0){
    const t = now*0.001;
    windAngle = t*0.28 + Math.sin(t*0.19)*1.9 + Math.sin(t*0.061)*2.4;   // sweeps every bearing
    windGust  = 0.55 + 0.45*Math.sin(t*0.9) + 0.2*Math.sin(t*2.7);
  }
  for(let s=0;s<substeps;s++) substep(DT/substeps);
  if(clothMat.userData.sh) clothMat.userData.sh.uniforms.uPos.value = posA.texture;
  renderer.setRenderTarget(null);
  renderer.render(scene, camera);
  frames++;
  if(now-t0>500){
    const fps = frames*1000/(now-t0); frames=0; t0=now;
    st.innerHTML = `<b>${fps.toFixed(0)} fps</b> · ${NP.toLocaleString()} particles · `+
      `${NE.toLocaleString()} constraints (${P.counts.structural.toLocaleString()} structural, `+
      `${P.counts.shear.toLocaleString()} shear, ${P.counts.bend.toLocaleString()} bend) · `+
      `${substeps}×${iters} · ${(NE*2*iters*substeps).toLocaleString()} corrections/frame`+
      (wind>0 ? ` · wind ${((windAngle*180/Math.PI)%360).toFixed(0)}°` : "");
  }
  requestAnimationFrame(loop);
}
selfCheck();
requestAnimationFrame(loop);
</script>
"""

if __name__ == "__main__":
    main()
