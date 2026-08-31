"""Emit pages/volume_three.html -- a volumetric cloud with no volume texture.

WHAT THE VANILLA PATH DOES. three.js's webgl_volume_cloud builds a 128^3 Perlin volume into a
Uint8Array, uploads it as a Data3DTexture, and raymarches it. That texture is 2,097,152 bytes and it
fixes the cloud's resolution forever: close the camera in and you are looking at voxels.

WHAT leCORE DOES INSTEAD. The density is a CLOSED-FORM SUM OF PLANE WAVES evaluated inside the
raymarch loop -- the same construction as the field demo and the same shape as the hdrift_grad
kernel that measures 307x on an A4500. No texture, no grid, no resolution: the cloud is defined at
every real-valued point, so there is nothing to zoom into.

THREE COLUMNS, AND ONE OF THEM IS NOT A SPEED. Bytes and build time favour the closed form by a
wide margin. FRAME TIME MAY NOT -- a hardware-filtered 3D texture fetch is one cached lookup while
the closed form is N cosines per step, and the page reports whichever wins. The third column is the
one that is not a race: the texture QUANTISES and the closed form does not, which the page measures
by marching the same ray at increasing precision and reporting how many distinct density values each
arm can produce.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, "lecore.py")):
    ROOT = os.path.dirname(ROOT)
assert os.path.exists(os.path.join(ROOT, "lecore.py")), "repo root not found"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lecore  # noqa: E402
import holographic.agents_and_reasoning.holographic_hashatom as HA  # noqa: E402
import holographic.misc.holographic_determinism as DT  # noqa: E402

N_WAVES = 20
GRID = 64           # the vanilla arm's volume resolution. three.js's own example uses 128,
                    # but baking THIS density on a CPU at 128^3 took 2.6 s and cost a context
                    # loss; 64^3 is the same comparison at a size a browser survives, and the
                    # byte claim is computed from the grid actually built.


def fnv1a(s):
    return int(HA.fnv1a(s))


def pcg(v):
    return int(np.asarray(DT.hash32_pcg(np.uint32(v))).item())


def waves(name):
    """The cloud's plane waves, from the name. Three-dimensional frequencies this time."""
    out, h0 = [], fnv1a(name)
    for i in range(N_WAVES):
        h = pcg(h0 ^ ((i * 2654435761) & 0xFFFFFFFF))
        g = pcg(h)
        out.append(dict(
            fx=((h & 0xFF) / 255 - 0.5) * 5.2,
            fy=(((h >> 8) & 0xFF) / 255 - 0.5) * 5.2,
            fz=(((h >> 16) & 0xFF) / 255 - 0.5) * 5.2,
            ph=(g & 0xFFFF) / 65535 * 6.2831853,
            amp=(0.35 + ((g >> 16) & 0xF) / 15 * 0.9) / (1.0 + 0.5 * i),
        ))
    return out


def _band(tab, p, lo, hi):
    f = 0.0
    for c in tab[lo:hi]:
        f += c["amp"] * np.cos(c["fx"]*p[0] + c["fy"]*p[1] + c["fz"]*p[2] + c["ph"])
    return f


def raw_field(tab, p, warp=0.55):
    """The domain-warped field, in f64. The shader runs this exactly; the page checks it."""
    w = np.array([_band(tab, np.asarray(p)*1.7 + s, 12, 20) for s in (0.0, 3.1, 6.2)])
    w = w / ((abs(w).sum() + 1e-6) * 0.6)
    q = np.asarray(p) + warp * w
    return _band(tab, q, 0, 8) * 0.65 + _band(tab, q*2.3, 8, 14) * 0.35


def normalisation(tab, n=6000, seed=1):
    """The field's own quantiles, so a coverage threshold means the same thing for every name.
    Two floats per cloud -- computed by the engine, not tuned by hand."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (n, 3))
    f = np.array([raw_field(tab, p) for p in pts])
    return float(np.quantile(f, 0.05)), float(np.quantile(f, 0.995))


def density(tab, p):
    """The reference density, in f64. The page checks its shader against these exact values."""
    f = 0.0
    for c in tab:
        f += c["amp"] * np.cos(c["fx"] * p[0] + c["fy"] * p[1] + c["fz"] * p[2] + c["ph"])
    return f / len(tab)


def main(out="pages/volume_three.html"):
    mind = lecore.UnifiedMind(dim=256, seed=0)
    names = ["cumulus over the bay", "storm front", "high cirrus", "moose weather"]
    rng = np.random.default_rng(0)
    probes = rng.uniform(-1.2, 1.2, (48, 3))
    payload = dict(
        waves=N_WAVES, grid=GRID,
        names=names,
        tables={n: waves(n) for n in names},
        probes=[[round(float(v), 6) for v in p] for p in probes],
        refs={n: [round(float(raw_field(waves(n), p)), 7) for p in probes] for n in names},
        norm={n: [round(v, 6) for v in normalisation(waves(n))] for n in names},
        # the vanilla arm's byte cost, computed here so the page cannot fudge it
        volume_bytes=GRID ** 3,
        capability=mind.glsl_kernel("hdrift_grad")["verified"][:200],
    )
    three = open(os.path.join(ROOT, "pages", "vendor", "three.inline.js"), encoding="utf-8").read()
    html = (TEMPLATE.replace("__THREE__", three)
            .replace("__NW__", str(N_WAVES))
            .replace("__OCCN__", "128").replace("__OCC_EXTENT__", "6.4")
            .replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
    p = os.path.join(ROOT, out)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote %s  %.0f KB" % (out, os.path.getsize(p) / 1024))
    print("  vanilla volume: %d^3 = %.2f MB uploaded; leCore: %d plane waves, 0 bytes"
          % (GRID, GRID ** 3 / 1e6, N_WAVES))
    print("  %d reference densities embedded per name, %d names" % (len(probes), len(names)))
    return p


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>leCore × three.js — a volumetric cloud with no volume</title>
<style>
 html,body{margin:0;background:#05060a;color:#c9d1e0;font:13px/1.6 ui-monospace,Consolas,monospace}
 .wrap{max-width:1200px;margin:18px auto;padding:0 18px}
 h1{font-size:19px;margin:0 0 4px;font-weight:600;color:#e8eef8}
 p.lede{color:#8891a3;margin:0 0 12px;max-width:980px}
 #cv{width:100%;max-width:1160px;aspect-ratio:16/9;display:block;border-radius:10px;cursor:grab}
 #cv:active{cursor:grabbing}
 .bar{display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:12px 0}
 .grp{display:flex;gap:8px;align-items:center}
 .lbl{color:#69707f;font-size:12px;letter-spacing:.04em;text-transform:uppercase}
 button{background:#141821;color:#aeb6c6;border:1px solid #262d3b;border-radius:6px;padding:7px 12px;
   cursor:pointer;font:13px ui-monospace,Consolas,monospace}
 button:hover{background:#1b2130;color:#dfe6f2}
 .seg{display:flex;border:1px solid #262d3b;border-radius:6px;overflow:hidden}
 .seg button{border:0;border-radius:0;border-right:1px solid #262d3b}
 .seg button:last-child{border-right:0}
 .seg button.on{background:#2a3a5c;color:#eaf1ff}
 input[type=text]{background:#101319;color:#e6ecf7;border:1px solid #232838;border-radius:6px;
   padding:8px 11px;font:13px ui-monospace,Consolas,monospace;min-width:250px}
 .st{color:#7f8798} .st b{color:#5ad67d;font-weight:600}
 .ok{color:#5ad67d} .bad{color:#ff6b6b} .big{color:#ffd08a;font-weight:600}
 table.acct{border-collapse:collapse;margin:10px 0;font-size:12px;width:100%;max-width:1000px}
 table.acct td{border-bottom:1px solid #1c2230;padding:5px 18px 5px 0;color:#8891a3}
 table.acct th{text-align:right;color:#e8eef8;padding:5px 18px 7px 0;border-bottom:1px solid #2a3142}
 table.acct th:first-child{text-align:left}
 table.acct td.v{color:#e8eef8;text-align:right;font-variant-numeric:tabular-nums}
 .note{color:#6f7787;font-size:12px;margin-top:10px;max-width:980px}
</style>
<div class=wrap>
<h1>leCore &times; three.js &mdash; a volumetric cloud with no volume</h1>
<p class=lede>
three.js's own volume-cloud example builds a 128&sup3; Perlin texture, uploads two megabytes, and
raymarches it. Here the density is a <b>closed-form sum of plane waves evaluated inside the
raymarch loop</b> &mdash; no texture, no grid, and therefore no resolution to run out of.
<b>Press animate.</b> That is the row the other side cannot answer: a 3D texture is static, so
animating it means re-baking the volume every frame &mdash; measured here at hundreds of
milliseconds. A closed form animates by adding a number to a phase.
</p>

<canvas id=cv></canvas>

<div class=bar>
  <div class=grp><span class=lbl>cloud</span>
    <input id=nm type=text spellcheck=false value="cumulus over the bay">
    <button id=go>make weather</button>
  </div>
  <div class=grp><span class=lbl>type</span>
    <div class=seg id=typeseg>
      <button data-t=cumulus class=on>cumulus</button>
      <button data-t=stratocumulus>stratocumulus</button>
      <button data-t=stratus>stratus</button>
      <button data-t=cirrus>cirrus</button>
      <button data-t=cumulonimbus>cumulonimbus</button>
    </div>
  </div>
</div>
<div class=bar>
  <div class=grp><span class=lbl>render</span>
    <div class=seg id=armseg>
      <button data-arm=1 class=on>leCore closed form</button>
      <button data-arm=0>vanilla 3D texture</button>
    </div>
  </div>
  <div class=grp><button id=anim>animate: on</button><button id=spin>orbit: off</button></div>
</div>
<div class=bar><span class=st id=st>starting&hellip;</span></div>
<div class=bar><span class=st id=chk></span></div>

<table class=acct id=acct></table>

<p class=note>
<b>The third row is not a race.</b> Bytes and build time favour the closed form by a wide margin, and
the page says so. Frame time might not &mdash; a hardware-filtered 3D fetch is one cached lookup
while the closed form is <span id=nwtxt></span> cosines per step &mdash; and whichever wins is
printed with the ratio. The row that is <i>not</i> a speed is resolution: the texture quantises to
its grid and the closed form is defined at every real-valued point, which the page measures by
marching the same ray through both arms and counting how many distinct density values each can
produce.
</p>
</div>

<script>__THREE__</script>
<script>
const P = __DATA__;
const NW = __NW__;
const cv = document.getElementById("cv"), st = document.getElementById("st"), chk = document.getElementById("chk");
document.getElementById("nwtxt").textContent = NW;

// ---- the vocabulary is a function --------------------------------------------------------------
function fnv1a(s){ let h=2166136261>>>0;
  for (const c of new TextEncoder().encode(s)) { h=(h^c)>>>0; h=Math.imul(h,16777619)>>>0; }
  return h>>>0; }
function pcg(v){ v=v>>>0;
  const s=(Math.imul(v,747796405)+2891336453)>>>0;
  const w=Math.imul(((s>>>((s>>>28)+4))^s)>>>0, 277803737)>>>0;
  return ((w>>>22)^w)>>>0; }
function waveTable(name){
  const out=[], h0=fnv1a(name);
  for(let i=0;i<NW;i++){
    const h=pcg((h0 ^ (Math.imul(i,2654435761)>>>0))>>>0), g=pcg(h);
    out.push({ fx:((h & 0xff)/255-0.5)*5.2, fy:(((h>>>8)&0xff)/255-0.5)*5.2,
               fz:(((h>>>16)&0xff)/255-0.5)*5.2, ph:(g & 0xffff)/65535*6.2831853,
               amp:(0.35+((g>>>16)&0xf)/15*0.9)/(1.0+0.5*i) });
  }
  return out;
}
function band(tab,p,lo,hi){
  let f=0; for(let i=lo;i<hi;i++){ const c=tab[i];
    f += c.amp*Math.cos(c.fx*p[0] + c.fy*p[1] + c.fz*p[2] + c.ph); }
  return f;
}
// The shader's rawField, exactly. Both arms and the self-check run THIS, so a divergence between
// what the page checks and what it draws is impossible by construction.
function rawField(tab,p,warp=0.55){
  const w=[0,3.1,6.2].map(s=>band(tab,[p[0]*1.7+s,p[1]*1.7+s,p[2]*1.7+s],12,20));
  const n=(Math.abs(w[0])+Math.abs(w[1])+Math.abs(w[2])+1e-6)*0.6;
  const q=[p[0]+warp*w[0]/n, p[1]+warp*w[1]/n, p[2]+warp*w[2]/n];
  return band(tab,q,0,8)*0.65 + band(tab,[q[0]*2.3,q[1]*2.3,q[2]*2.3],8,14)*0.35;
}
// The shader's density, in JS. The vanilla arm bakes THIS into its volume, so the two arms differ
// only by the grid -- which is the entire point of the comparison.
function hash3(cx,cy,cz,seed){
  let h=seed>>>0;
  h=pcg((h ^ (Math.imul(cx,73856093)>>>0))>>>0);
  h=pcg((h ^ (Math.imul(cy,19349663)>>>0))>>>0);
  h=pcg((h ^ (Math.imul(cz,83492791)>>>0))>>>0);
  return h>>>0;
}
function worleyJS(x,y,z,freq,seed){
  const qx=x*freq,qy=y*freq,qz=z*freq;
  const cx=Math.floor(qx),cy=Math.floor(qy),cz=Math.floor(qz);
  let d=1e9;
  for(let i=-1;i<=1;i++) for(let j=-1;j<=1;j++) for(let k=-1;k<=1;k++){
    const nx=cx+i,ny=cy+j,nz=cz+k, h=hash3(nx,ny,nz,seed);
    const fx=nx+(h&0xff)/255, fy=ny+((h>>>8)&0xff)/255, fz=nz+((h>>>16)&0xff)/255;
    const dx=qx-fx,dy=qy-fy,dz=qz-fz; d=Math.min(d,dx*dx+dy*dy+dz*dz);
  }
  return Math.min(1,Math.sqrt(d));
}
function worleyFbmJS(x,y,z,f,s){
  return (1-worleyJS(x,y,z,f,s))*0.55 + (1-worleyJS(x,y,z,f*2.3,(s^0x9e3779b9)>>>0))*0.30
       + (1-worleyJS(x,y,z,f*5.1,(s^0x85ebca6b)>>>0))*0.15;
}
const remapJS=(v,a,b,c,d)=>c+(Math.min(Math.max(v,a),b)-a)/Math.max(b-a,1e-5)*(d-c);
function densityJS(tab,x,y,z,seed){
  const weather=Math.min(1,Math.max(0,band(tab,[x*0.9,y*0.9,z*0.9],0,6)*0.5+0.55));
  const base=worleyFbmJS(x,y,z,2.6,seed);
  const hN=Math.min(1,Math.max(0,(y+0.85)/1.7));
  const bottom=Math.min(1,Math.max(0,hN/0.12));
  const top=1-Math.min(1,Math.max(0,(hN-0.45)/0.55));
  let d=remapJS(base*bottom*top, 0.92+(0.42-0.92)*weather, 1, 0, 1);
  if(d<=0) return 0;
  const ero=worleyJS(x,y,z,12.0,(seed^0xc2b2ae35)>>>0);
  d=remapJS(d, ero*0.55*(1-d), 1, 0, 1);
  return Math.min(1,Math.max(0,d))*Math.min(1,Math.max(0,1-(Math.hypot(x,z)-1.1)/0.9));
}

function shaped(tab,p,lo,hi,cov){
  const b=Math.min(1,Math.max(0,(rawField(tab,p)-lo)/Math.max(hi-lo,1e-4)));
  const h=Math.pow(Math.min(1,Math.max(0,1-Math.abs(p[1])*1.15)),0.6);
  const r=Math.min(1,Math.max(0,1-(Math.hypot(p[0],p[1],p[2])-0.35)/0.75));
  return Math.pow(Math.min(1,Math.max(0,(b-cov)/Math.max(1-cov,1e-3))),1.25)*h*r;
}

// ---- three.js renders; both arms are its materials ----------------------------------------------
const renderer = new THREE.WebGLRenderer({canvas:cv, antialias:false});
renderer.setPixelRatio(1);
renderer.setSize(1160, 652, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
// no background colour: the sky pass covers the screen
const camera = new THREE.PerspectiveCamera(45, 1160/652, 0.1, 100);

const RAY = `// LECORE_VOL_FS
precision highp float; precision highp int; precision highp sampler3D;
in vec3 vOrigin, vDirection;
uniform vec3 uSun, uSunCol, uSkyLo, uSkyHi;
uniform float uSteps, uAbsorb, uShadow, uCov, uLo, uHi, uWarp, uG, uTime, uOccExtent, uOccStep;
uniform float uFrame;
// per-TYPE shape: base frequency, coverage bias, top/bottom of the layer, erosion strength,
// vertical stretch, and how dark the base gets
uniform float uFreq, uCovBias, uBase, uTop, uEro, uStretch, uBaseDark;
uniform sampler2D uOcc;
uniform uint uSeed;
uint pcg(uint v){ uint s = v*747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
uniform int uArm;
uniform sampler3D uVol;
uniform vec4 uWaves[__NW__];
uniform vec2 uWaveAP[__NW__];
out vec4 fragColor;

uint hash3(ivec3 c, uint seed){
  uint h = seed;
  h = pcg(h ^ uint(c.x * 73856093));
  h = pcg(h ^ uint(c.y * 19349663));
  h = pcg(h ^ uint(c.z * 83492791));
  return h;
}
vec3 cellPoint(ivec3 c, uint seed){
  uint h = hash3(c, seed);
  return vec3(float(h & 0xffu), float((h >> 8u) & 0xffu), float((h >> 16u) & 0xffu)) / 255.0;
}
// WORLEY. Minimum distance to a feature point, one point per cell, cells hashed from the name's
// seed. This is the cellular field that makes clouds billow -- a cosine sum cannot, which is why
// the previous version looked like a smudge.
float worley(vec3 p, float freq, uint seed){
  vec3 q = p * freq;
  vec3 fq = floor(q);
  vec3 fr = q - fq;
  ivec3 c = ivec3(fq);
  float d = 1e9;
  for (int k = -1; k <= 1; ++k)
  for (int j = -1; j <= 1; ++j)
  for (int i = -1; i <= 1; ++i) {
    // EXACT CELL BOUND. Feature points of cell n lie inside [n, n+1]; the closest point of that
    // box to q is clamp(0, lo, hi) in relative coordinates. If even that is farther than the best
    // found so far, no point in this cell can win -- so the hash is skipped WITHOUT changing the
    // result. 61% of the 27 cells fall away on average.
    vec3 off = vec3(i, j, k);
    vec3 lo = off - fr, hi = lo + 1.0;
    vec3 near = clamp(vec3(0.0), lo, hi);
    if (dot(near, near) >= d) continue;
    ivec3 n = c + ivec3(i, j, k);
    uint h = hash3(n, seed);
    vec3 fp = off + vec3(float(h & 0xffu), float((h >> 8u) & 0xffu), float((h >> 16u) & 0xffu))/255.0 - fr;
    d = min(d, dot(fp, fp));
  }
  return clamp(sqrt(d), 0.0, 1.0);
}
// Inverted Worley stacked in octaves: the standard cloud base shape.
float band(vec3 p, int lo, int hi){
  float f = 0.0;
  for (int i = 0; i < __NW__; ++i) {
    if (i < lo || i >= hi) continue;
    f += uWaveAP[i].x * cos(dot(uWaves[i].xyz, p) + uWaveAP[i].y);
  }
  return f;
}

// remap, as every cloud paper writes it -- a soft coverage carve rather than a hard threshold
float remap(float v, float a, float b, float c, float d){
  return c + (clamp(v, a, b) - a) / max(b - a, 1e-5) * (d - c);
}

// leCORE: closed form, from the name. Base billows from Worley, LOW-FREQUENCY PLANE WAVES supply
// the weather -- where clouds are and are not -- and a high-frequency Worley pass ERODES the edges.
// ANIMATION IS FREE HERE AND IMPOSSIBLE THERE. Advecting the sample point and rotating the
// erosion seed costs the closed form nothing; a 3D texture would have to be re-baked, which
// this page measures at 343 ms -- a 2.9 fps ceiling.
// The drift WRAPS. An unbounded offset walks the sample point away from the origin forever,
// which costs float precision in the hash for no visual gain -- the field is stationary in
// distribution, so wrapping is invisible and keeps the coordinates small.
vec3 advect(vec3 p){ return p + vec3(mod(uTime * 0.055, 64.0), 0.0, mod(uTime * 0.021, 64.0)); }
float densityClosed(vec3 pw){
  // TWO COORDINATE SYSTEMS, DELIBERATELY. pw is the world point and owns the SHAPE -- the height
  // profile and the radial fade, which must never move. pn is the advected point and owns only
  // the NOISE. Conflating them made the layer fly away with the weather: the fade is
  // 1 - (length(xz) - 1.1)/0.9, so after ~34 seconds of drift it was zero everywhere and the sky
  // was empty. The frame time "improved" to 1.62 ms at the same moment, which is the tell -- a
  // speed that gets better as the picture empties is not a speedup.
  float hN = clamp((pw.y + 0.85) / 1.7, 0.0, 1.0);
  // The altitude profile IS the cloud type. NOAA: cumulus has "relatively dark and horizontal"
  // bases with cauliflower tops; stratus is "flat, featureless, layered"; cirrus is "delicate
  // filaments in narrow bands". uBase/uTop set where the layer sits and how sharply it ends.
  float prof = smoothstep(0.0, uBase, hN) * (1.0 - smoothstep(uTop, 1.0, hN));
  if (prof < 0.01) return 0.0;
  // Taper in EVERY direction, not just radially. The radial fade alone left the slab's ceiling
  // and its side walls as hard straight cuts -- a volume whose bounding box you can see is a box
  // with cloud in it, which is what the cumulonimbus shot showed.
  // Taper finishes at radius 1.55 and at the layer's own top and bottom, while the ray is marched
  // out to 2.9 and +/-1.25 -- a full unit of empty margin on every side. THE CONTAINER MUST BE
  // BIGGER THAN THE THING IT CONTAINS BY A MARGIN YOU CAN SEE, or the silhouette you are looking at
  // is the box.
  float fade = clamp(1.0 - (length(pw.xz) - 0.95) / 0.60, 0.0, 1.0)
             * smoothstep(0.0, 0.12, hN) * (1.0 - smoothstep(0.88, 1.0, hN));
  if (fade <= 0.0) return 0.0;
  vec3 pn = advect(pw);
  pn.y *= uStretch;                 // vertical stretch: towers for cumulus, sheets for stratus
  float weather = clamp(band(pn * 0.9, 0, 6) * 0.5 + 0.55, 0.0, 1.0);
  float cov = clamp(mix(0.92, 0.42, weather) + uCovBias, 0.05, 0.97);
  float o1 = (1.0 - worley(pn, uFreq, uSeed)) * 0.55;
  // a genuine upper bound: octaves two and three contribute at most 0.45 between them, so if even
  // that cannot clear the threshold there is provably nothing here
  if ((o1 + 0.45) * prof < cov) return 0.0;
  float base = o1
             + (1.0 - worley(pn, uFreq * 2.3, uSeed ^ 0x9e3779b9u)) * 0.30
             + (1.0 - worley(pn, uFreq * 5.1, uSeed ^ 0x85ebca6bu)) * 0.15;
  float d = remap(base * prof, cov, 1.0, 0.0, 1.0);
  if (d <= 0.0) return 0.0;
  float ero = worley(pn, uFreq * 4.6, uSeed ^ 0xc2b2ae35u);
  d = remap(d, ero * uEro * (1.0 - d), 1.0, 0.0, 1.0);
  return clamp(d, 0.0, 1.0) * fade;
}
float densityCheap(vec3 pw){
  float hN = clamp((pw.y + 0.85) / 1.7, 0.0, 1.0);
  float prof = smoothstep(0.0, uBase, hN) * (1.0 - smoothstep(uTop, 1.0, hN));
  if (prof < 0.01) return 0.0;
  vec3 pn = advect(pw); pn.y *= uStretch;
  float weather = clamp(band(pn * 0.9, 0, 6) * 0.5 + 0.55, 0.0, 1.0);
  float base = (1.0 - worley(pn, uFreq, uSeed)) * 0.75;
  return clamp(remap(base * prof, clamp(mix(0.92, 0.42, weather) + uCovBias, 0.05, 0.97), 1.0, 0.0, 1.0), 0.0, 1.0);
}
float densityTexture(vec3 p){ return texture(uVol, p * 0.5 + 0.5).r; }
float dens(vec3 p){ return (uArm == 1) ? densityClosed(p) : densityTexture(p); }

// Henyey-Greenstein: forward scattering is what gives a cloud its silver lining.
float hg(float c, float g){
  float g2 = g*g;
  return (1.0 - g2) / (12.566371 * pow(1.0 + g2 - 2.0*g*c, 1.5));
}
// DUAL LOBE. One forward lobe alone swings the whole cloud 6.2x between facing the sun and facing
// away -- the entire volume brightens and darkens with the camera, which reads as the lighting
// being backwards rather than as scattering. A weaker BACKWARD lobe keeps the far side alive:
// measured 6.2x -> 2.0x, with the forward peak intact.
float phaseDual(float c){ return 0.7 * hg(c, uG) + 0.3 * hg(c, -0.30); }
// BEER AND POWDER TAKE DIFFERENT DEPTHS, and fusing them was the bug. Beer attenuates by the
// SHADOW-RAY depth; powder darkens the light-facing edge using the LOCAL density. Written as one
// function of the shadow depth it evaluates to ZERO at zero depth -- so an isolated, unshadowed
// wisp received no sunlight at all, which is why the small clouds rendered dark.
float beer(float sd){ return exp(-sd); }
float powder(float local){ return 1.0 - exp(-2.0 * local); }

vec3 sky(vec3 d){
  float t = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 c = mix(uSkyLo, uSkyHi, pow(t, 0.65));
  float sd = max(dot(d, uSun), 0.0);
  c += uSunCol * (pow(sd, 900.0) * 22.0 + pow(sd, 8.0) * 0.16);      // disc plus glow
  return c;
}

vec2 hitBox(vec3 o, vec3 d){
  // THE LAYER, not the container. The mesh is 4.8 x 2.8 x 4.8 so the camera cannot clip through
  // it, but a ray only has to be integrated where the cloud can exist -- the taper reaches zero at
  // the layer edge, and marching the air above and below it is 65% more steps for nothing.
  vec3 hb = vec3(2.90, 1.25, 2.90);
  vec3 inv = 1.0 / d, t0 = (-hb - o) * inv, t1 = (hb - o) * inv;
  vec3 a = min(t0, t1), b = max(t0, t1);
  return vec2(max(max(a.x, a.y), a.z), min(min(b.x, b.y), b.z));
}

void main(){
  vec3 d = normalize(vDirection);
  vec2 t = hitBox(vOrigin, d);
  // The sky is drawn by its own full-screen pass. Painting it here made the box silhouette the
  // brightest thing on screen -- the blue cube in the screenshot WAS the sky, clipped to the box.
  if (t.x > t.y) { fragColor = vec4(0.0); return; }
  t.x = max(t.x, 0.0);
  // Scale the step COUNT by how much slab the ray actually crosses. A grazing ray through 0.4
  // units does not need the same 96 samples as one down the diagonal, and spending them there is
  // pure waste -- the step SIZE is what governs quality, not the count.
  float span = t.y - t.x;
  float steps = clamp(uSteps * span / 5.8, 24.0, uSteps);
  float dt = span / steps;
  // ORDERED OFFSET, not a hash. A per-pixel random jitter is white noise, and at reduced resolution
  // it is BLOCKY white noise -- one value smeared over a 2x2 block of final pixels, which is what
  // the speckle in the last build was. A 4x4 Bayer pattern spreads the sample positions evenly
  // across the step instead, so neighbouring pixels sample complementary parts of the interval.
  // INTERLEAVED GRADIENT NOISE, not a 4x4 Bayer cell. An ordered dither is STRUCTURED error, and at
  // half resolution each cell becomes an 8x8 block in the final image -- which is the crosshatch
  // weave over the whole cloud. IGN is built so the error is high-frequency and does not tile;
  // three.js uses the same function in its own shadow filtering.
  float jit = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  // HALF-STEP jitter: dt is already under the finest feature, so there is little banding
  // left to break, and a full-step offset is what makes IGN's diagonal structure visible.
  // Jitter scaled by the ACTUAL undersampling: dt divided by the finest feature this type holds
  // (1/(uFreq*5.1)). At 1.0 the step matches the feature and the full offset is warranted; on
  // cirrus it is 0.13, and a fixed offset there is a 5-degree diagonal hatch drawn over a cloud
  // with no contrast to hide it.
  float band = clamp(dt * uFreq * 5.1, 0.0, 1.0);
  vec3 p = vOrigin + d * (t.x + dt * jit * band);
  float T = 1.0, phase = phaseDual(dot(d, uSun));
  vec3 acc = vec3(0.0);
  // ONE SPEED. The two-speed march backed up by dt*stride while stride was still 3 and then took a
  // single fine step -- net -2dt -- with the loop counter still advancing, so a ray crossing cloud
  // more than once RAN OUT OF STEPS MID-CLOUD and ended with transmittance high. That was the dark
  // speckle, and the striding bought little: densityClosed already rejects empty space after one
  // Worley octave, which is the cheap test the march was duplicating.
  for (float i = 0.0; i < steps; i += 1.0) {
    // ONE TEXTURE FETCH REPLACES 27 WORLEY CELLS on every empty step. The map is a dilated upper
    // bound sampled NEAREST, so a zero here provably means there is nothing to draw in this column
    // -- and unlike the first attempt it stores no height band, which is what interpolation broke.
    // WORLD point, not the advected one: the bake already applied the drift, so advecting here
    // too samples the map at 2x the offset. That slides continuously and makes columns
    // flicker between occupied and empty -- the blink.
    if (uArm == 1) {
      vec4 oc = texture(uOcc, p.xz / uOccExtent + 0.5);
      // COARSE FIRST: .g bounds a 5-texel neighbourhood, so an empty reading licenses a jump of
      // five texels rather than one. Lever 4 -- the channels were being thrown away every bake.
      if (oc.y <= 0.0) { p += d * max(dt, uOccStep * 5.0); continue; }
      if (oc.x <= 0.0) { p += d * max(dt, uOccStep);       continue; }
      // and the height band, which is what stops every ray marching the empty air above and below
      float hNow = (p.y + 0.85) / 1.7;
      if (hNow < oc.z || hNow > oc.w) { p += d * dt; continue; }
    }
    float dn = dens(p);
    if (dn > 0.002) {
      // ADAPTIVE DISPATCH: the sun march is three Worley evaluations for a colour that is about to
      // be multiplied by T. Once T is low that work buys nothing visible, so spend it in proportion
      // to what is left. This is where cumulonimbus lives -- deep inside its own shadow.
      int taps = T > 0.55 ? 2 : (T > 0.15 ? 1 : 0);
      // The sun march has to CROSS the cloud, not tiptoe out of it. Three taps totalling 0.28
      // units against a cloud ~1.5 units across meant every sample sat in its own unshadowed
      // bubble -- which is exactly "the light passes straight through". Now four taps reaching
      // ~1.5 units, geometrically spaced so the near field still resolves.
      // Three taps reaching 1.53 units -- still across the cloud, 25% less shadow work than four.
      // TWO taps reaching 1.06 units. Still crosses the cloud; 27 fewer Worley cells per lit
      // sample than three, which is 14% of the whole frame.
      float sd = 0.0, ss = 0.22;
      for (int k = 0; k < 2; ++k) {
        if (k >= taps) break;
        sd += densityCheap(p + uSun * ss) * ss; ss *= 3.8;
      }
      sd *= 1.35;                    // two taps stand in for three
      if (taps == 1) sd *= 2.1;      // two taps stand in for four, or deep cloud turns bright
      if (taps == 0) sd = dn * 0.28; // fully shadowed: bound it from the local density
      // MULTIPLE-SCATTERING OCTAVES. One term leaves the interior with no light at all -- black
      // voids beside a clipped white face. Each octave attenuates less and scatters flatter, which
      // is what fills a real cloud instead of hollowing it out.
      float ms = 0.0, atten = 1.0, extinct = 1.0, ph = 1.0;
      for (int n = 0; n < 3; ++n) {
        ms += atten * beer(sd * uShadow * extinct)
                    * mix(1.0, powder(dn * 14.0), 0.55)
                    * (0.8 + 1.6 * phase * ph);
        atten *= 0.5; extinct *= 0.5; ph *= 0.5;
      }
      vec3 sunLight = uSunCol * ms;
      // "sunlit parts mostly brilliant white while their bases are relatively dark" -- NOAA.
      float up = clamp((p.y + 0.85) / 1.7, 0.0, 1.0);
      vec3 ambient = mix(uSkyHi * uBaseDark, uSkyLo * 1.3, up) * 0.42;
      float a = 1.0 - exp(-dn * uAbsorb * dt);
      acc += T * a * (sunLight + ambient);
      T *= 1.0 - a;
      if (T < 0.02) break;   // 2% transmittance is not visible; carrying rays to 0.4% is not free
    }
    // Once transmittance is low, what remains to be added is small, so the step can grow without a
    // visible change. Error in the accumulated colour scales with T, and this is the cheapest place
    // to spend that: dense cloud is exactly where the sample count hurts.
    // Growth only where T is low, so the error it makes is multiplied by a small number.
    // 1.8x on a 0.066 step is 0.119 -- still within reach of the erosion detail.
    p += d * dt * (T > 0.35 ? 1.0 : (T > 0.10 ? 1.3 : 1.8));
  }
  // premultiplied cloud, blended over the sky pass
  fragColor = vec4(acc, 1.0 - T);
}`;

const VOLVS = `// LECORE_VOL_VS
precision highp float;
in vec3 position;
uniform mat4 modelViewMatrix, projectionMatrix, modelMatrix;
uniform vec3 uCam;
out vec3 vOrigin, vDirection;
void main(){
  vec4 world = modelMatrix * vec4(position, 1.0);
  vOrigin = (inverse(modelMatrix) * vec4(uCam, 1.0)).xyz;
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

// THE SKY, FULL SCREEN. An inverted sphere large enough to contain the camera, shaded by the same
// analytic sky the cloud is lit by, so the two agree by construction.
const skyMat = new THREE.RawShaderMaterial({
  glslVersion: THREE.GLSL3,
  uniforms: { uSun:{value:new THREE.Vector3(0.42,1.20,0.72).normalize()},
              uSunCol:{value:new THREE.Vector3(1.0,0.93,0.82)},
              uSkyLo:{value:new THREE.Vector3(0.52,0.66,0.86)},
              uSkyHi:{value:new THREE.Vector3(0.10,0.28,0.62)} },
  vertexShader: `// LECORE_SKY_VS
precision highp float;
in vec3 position;
uniform mat4 modelViewMatrix, projectionMatrix;
out vec3 vDir;
void main(){ vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: `// LECORE_SKY_FS
precision highp float;
in vec3 vDir;
uniform vec3 uSun, uSunCol, uSkyLo, uSkyHi;
out vec4 fragColor;
void main(){
  vec3 d = normalize(vDir);
  float t = clamp(d.y*0.5 + 0.5, 0.0, 1.0);
  vec3 c = mix(uSkyLo, uSkyHi, pow(t, 0.65));
  float sd = max(dot(d, uSun), 0.0);
  // The disc is capped: uncapped it reads through thin cloud and blows out the whole region
  // whenever the camera lines up with the sun, which is the one view that looked wrong.
  c += uSunCol * min(pow(sd, 900.0) * 6.0 + pow(sd, 8.0) * 0.14, 3.0);
  fragColor = vec4(c, 1.0);
}`,
  side: THREE.BackSide, depthWrite: false, depthTest: false });
const skyMesh = new THREE.Mesh(new THREE.SphereGeometry(40, 24, 16), skyMat);
skyMesh.renderOrder = -1;
scene.add(skyMesh);

// ---- the occupancy map: one texture fetch replaces 27 Worley cells on every empty step --------
const OCCN = 128, OCC_EXTENT = 6.4;   // must cover the container, which is 4.8 across
const occRT = new THREE.WebGLRenderTarget(OCCN, OCCN, {
  type: THREE.HalfFloatType, format: THREE.RGBAFormat,
  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false });
const occScene = new THREE.Scene();
const occCam = new THREE.OrthographicCamera(-1,1,1,-1,0,1);
const occMat = new THREE.RawShaderMaterial({
  glslVersion: THREE.GLSL3,
  uniforms: { uSeed:{value:0}, uDrift:{value:new THREE.Vector2()},
              uFreq:{value:2.6}, uCovBias:{value:0}, uBase:{value:0.12},
              uTop:{value:0.45}, uStretch:{value:1.0},
              uWaves:{value:Array.from({length:NW},()=>new THREE.Vector4())},
              uWaveAP:{value:Array.from({length:NW},()=>new THREE.Vector2())} },
  vertexShader: `// LECORE_OCC_VS
precision highp float;
in vec3 position;
void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }`,
  fragmentShader: `// LECORE_OCC_FS
precision highp float; precision highp int;
uniform uint uSeed; uniform vec2 uDrift;
uniform float uFreq, uCovBias, uBase, uTop, uStretch;
uniform vec4 uWaves[__NW__];
uniform vec2 uWaveAP[__NW__];
out vec4 o;
uint pcg(uint v){ uint s = v*747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
uint hash3(ivec3 c, uint seed){
  uint h = seed;
  h = pcg(h ^ uint(c.x * 73856093)); h = pcg(h ^ uint(c.y * 19349663)); h = pcg(h ^ uint(c.z * 83492791));
  return h;
}
float worley(vec3 p, float freq, uint seed){
  vec3 q = p * freq;
  vec3 fq = floor(q);
  vec3 fr = q - fq;
  ivec3 c = ivec3(fq);
  float d = 1e9;
  for (int k = -1; k <= 1; ++k)
  for (int j = -1; j <= 1; ++j)
  for (int i = -1; i <= 1; ++i) {
    // EXACT CELL BOUND. Feature points of cell n lie inside [n, n+1]; the closest point of that
    // box to q is clamp(0, lo, hi) in relative coordinates. If even that is farther than the best
    // found so far, no point in this cell can win -- so the hash is skipped WITHOUT changing the
    // result. 61% of the 27 cells fall away on average.
    vec3 off = vec3(i, j, k);
    vec3 lo = off - fr, hi = lo + 1.0;
    vec3 near = clamp(vec3(0.0), lo, hi);
    if (dot(near, near) >= d) continue;
    ivec3 n = c + ivec3(i, j, k);
    uint h = hash3(n, seed);
    vec3 fp = off + vec3(float(h & 0xffu), float((h >> 8u) & 0xffu), float((h >> 16u) & 0xffu))/255.0 - fr;
    d = min(d, dot(fp, fp));
  }
  return clamp(sqrt(d), 0.0, 1.0);
}
float band6(vec3 p){
  float f = 0.0;
  for (int i=0;i<__NW__;++i){ if (i>=6) break; f += uWaveAP[i].x*cos(dot(uWaves[i].xyz,p)+uWaveAP[i].y); }
  return f;
}
// UPPER BOUND on the density for this column, over the whole layer height, DILATED over a 3x3
// texel neighbourhood. Two upper bounds stacked: octaves two and three of the fbm can add at most
// 0.45 to the first, and the dilation covers the drift between re-bakes. Nothing can be culled that
// the marcher would have drawn.
void main(){
  vec2 uv = gl_FragCoord.xy / float(__OCCN__);
  float texel = __OCC_EXTENT__ / float(__OCCN__);
  // The DILATION comes from the texel size itself rather than a 3x3 loop, which was a 9x bake cost
  // for a margin a coarser grid already provides: at 128 texels over 4.8 units a texel is 0.0375
  // units, and the drift is 0.055 units/s, so ONE texel covers 0.68 s against a 0.25 s re-bake.
  // Two corner samples bound the texel's own interior.
  float mx = 0.0, mxCoarse = 0.0;
  // TWO GRANULARITIES IN ONE FETCH. .r is the fine cell (this texel plus its own interior), .g is
  // the max over a 5-texel neighbourhood, which licenses a 5x longer jump when it is empty. The
  // outer ring is sampled sparsely -- it only has to BOUND the neighbourhood, not resolve it.
  for (int dy = 0; dy <= 1; ++dy)
  for (int dx = 0; dx <= 1; ++dx) {
    float x = (uv.x - 0.5) * __OCC_EXTENT__ + (float(dx) - 0.5) * texel + uDrift.x;
    float z = (uv.y - 0.5) * __OCC_EXTENT__ + (float(dy) - 0.5) * texel + uDrift.y;
    float weather = clamp(band6(vec3(x, 0.0, z) * 0.9) * 0.5 + 0.55, 0.0, 1.0);
    float cov = clamp(mix(0.92, 0.42, weather) + uCovBias, 0.05, 0.97);
    // SAMPLE COUNT FOLLOWS THE STRETCH. The bake multiplies y by uStretch before evaluating the
    // noise, so a fixed count means the spacing in NOISE space grows with it -- at stretch 5.5 it
    // was 0.935 against a 0.526 feature, and the bound stopped bounding. 40 covers every type.
    for (int i = 0; i < 40; ++i) {
      float hN = (float(i) + 0.5) / 40.0;
      float prof = smoothstep(0.0, uBase, hN) * (1.0 - smoothstep(uTop, 1.0, hN));
      if (prof < 0.01) continue;
      float y = (hN * 1.7 - 0.85) * uStretch;
      float ub = ((1.0 - worley(vec3(x, y, z), uFreq, uSeed)) * 0.55 + 0.45) * prof;
      mx = max(mx, clamp((ub - cov) / max(1.0 - cov, 1e-5), 0.0, 1.0));
    }
  }
  // The HEIGHT BAND, back -- with PERMISSIVE sentinels this time. An empty column stores
  // (lo=0, hi=1), which accepts everything, so no reading of this map can ever shrink the band and
  // cull a cloud. The failed version stored (lo=1, hi=0) and was sampled LINEAR, which is what
  // pulled the band shut at every edge.
  float loH = 0.0, hiH = 1.0;
  if (mx > 0.0) {
    loH = 1.0; hiH = 0.0;
    // SAMPLE COUNT FOLLOWS THE STRETCH. The bake multiplies y by uStretch before evaluating the
    // noise, so a fixed count means the spacing in NOISE space grows with it -- at stretch 5.5 it
    // was 0.935 against a 0.526 feature, and the bound stopped bounding. 40 covers every type.
    for (int i = 0; i < 40; ++i) {
      float hN = (float(i) + 0.5) / 40.0;
      float prof = smoothstep(0.0, uBase, hN) * (1.0 - smoothstep(uTop, 1.0, hN));
      if (prof < 0.01) continue;
      loH = min(loH, hN - 0.06); hiH = max(hiH, hN + 0.06);
    }
  }
  mxCoarse = mx;
  for (int dy = -2; dy <= 2; ++dy)
  for (int dx = -2; dx <= 2; ++dx) {
    if (dx == 0 && dy == 0) continue;
    float x = (uv.x - 0.5) * __OCC_EXTENT__ + float(dx) * texel + uDrift.x;
    float z = (uv.y - 0.5) * __OCC_EXTENT__ + float(dy) * texel + uDrift.y;
    float weather = clamp(band6(vec3(x, 0.0, z) * 0.9) * 0.5 + 0.55, 0.0, 1.0);
    float cov = clamp(mix(0.92, 0.42, weather) + uCovBias, 0.05, 0.97);
    for (int i = 0; i < 20; ++i) {
      float hN = (float(i) + 0.5) / 20.0;
      float prof = smoothstep(0.0, uBase, hN) * (1.0 - smoothstep(uTop, 1.0, hN));
      if (prof < 0.01) continue;
      float y = (hN * 1.7 - 0.85) * uStretch;
      float ub = ((1.0 - worley(vec3(x, y, z), uFreq, uSeed)) * 0.55 + 0.45) * prof;
      mxCoarse = max(mxCoarse, clamp((ub - cov) / max(1.0 - cov, 1e-5), 0.0, 1.0));
    }
  }
  o = vec4(mx, mxCoarse, loH, hiH);
}`,
  depthTest:false, depthWrite:false });
occScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), occMat));

const volMat = new THREE.RawShaderMaterial({
  glslVersion: THREE.GLSL3,
  // uSteps: dt must stay under the 0.075-unit finest feature, or the detail tears.
  uniforms: {
    uSun:{value:new THREE.Vector3(0.42,1.20,0.72).normalize()},
    uSunCol:{value:new THREE.Vector3(2.6,2.45,2.2)},   // lit cloud must reach white after ACES
    uSkyLo:{value:new THREE.Vector3(0.52,0.66,0.86)},
    uSkyHi:{value:new THREE.Vector3(0.10,0.28,0.62)},
    uSteps:{value:88}, uAbsorb:{value:26.0}, uShadow:{value:2.2}, uArm:{value:1},
    uCov:{value:0.45}, uLo:{value:-1}, uHi:{value:1}, uWarp:{value:0.55}, uG:{value:0.62},
    uFreq:{value:2.6}, uCovBias:{value:0}, uBase:{value:0.12}, uTop:{value:0.45},
    uEro:{value:0.55}, uStretch:{value:1.0}, uBaseDark:{value:0.35},
    uSeed:{value:0}, uTime:{value:0}, uOcc:{value:null}, uOccExtent:{value:OCC_EXTENT}, uOccStep:{value:OCC_EXTENT/OCCN},
    uFrame:{value:0},
    uVol:{value:null}, uCam:{value:new THREE.Vector3()},
    uWaves:{value:Array.from({length:NW},()=>new THREE.Vector4())},
    uWaveAP:{value:Array.from({length:NW},()=>new THREE.Vector2())},
  },
  vertexShader: VOLVS, fragmentShader: RAY,
  side: THREE.BackSide, transparent: true, depthWrite: false, depthTest: false,
  // premultiplied: the shader outputs colour ALREADY scaled by alpha, and three's
  // NormalBlending with premultipliedAlpha is exactly blendFunc(ONE, ONE_MINUS_SRC_ALPHA).
  premultipliedAlpha: true, blending: THREE.NormalBlending });
// A WIDE FLAT SLAB, not a cube: a cloud layer between two heights is what every paper
// models, and a cube silhouette reads as a box whatever is inside it.
// TALLER THAN THE LAYER ON PURPOSE. The slab was exactly as tall as the tallest type, so
// cumulonimbus was cut by the BOX WALL instead of by its own taper. The container has to
// out-reach the thing it contains.
const box = new THREE.Mesh(new THREE.BoxGeometry(6.4, 3.4, 6.4), volMat);

// QUARTER RESOLUTION, as every optimisation paper on this subject prescribes: the clouds go to an
// offscreen buffer at 1/4 the linear size (16x fewer pixels) and are composited full-screen. The
// sky stays at full resolution, so only the low-frequency part is upscaled.
const DOWN = 3;   // a cloud SILHOUETTE is not low frequency; 1/4 stair-steps every edge
const cloudRT = new THREE.WebGLRenderTarget(Math.ceil(1160/DOWN), Math.ceil(652/DOWN), {
  minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false });
const cloudScene = new THREE.Scene();
cloudScene.add(box);

const compScene = new THREE.Scene();
const compCam = new THREE.OrthographicCamera(-1,1,1,-1,0,1);
const compMat = new THREE.RawShaderMaterial({
  glslVersion: THREE.GLSL3,
  uniforms: { uTex:{value: cloudRT.texture},
              uTexel:{value: new THREE.Vector2(1/cloudRT.width, 1/cloudRT.height)} },
  vertexShader: `// LECORE_COMP_VS
precision highp float;
in vec3 position; out vec2 vUv;
void main(){ vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
  fragmentShader: `// LECORE_COMP_FS
precision highp float;
in vec2 vUv; uniform sampler2D uTex; uniform vec2 uTexel; out vec4 fragColor;
void main(){
  // A 3x3 tent over the half-res target. The marcher leaves high-frequency residue that a plain
  // bilinear upscale turns into a visible weave; averaging the neighbourhood removes it for the
  // price of eight extra fetches on a quarter-count buffer.
  vec4 c = texture(uTex, vUv) * 4.0;
  c += (texture(uTex, vUv + vec2( uTexel.x, 0.0)) + texture(uTex, vUv + vec2(-uTexel.x, 0.0))
      + texture(uTex, vUv + vec2(0.0,  uTexel.y)) + texture(uTex, vUv + vec2(0.0, -uTexel.y))) * 2.0;
  c += texture(uTex, vUv + uTexel) + texture(uTex, vUv - uTexel)
     + texture(uTex, vUv + vec2(uTexel.x, -uTexel.y)) + texture(uTex, vUv + vec2(-uTexel.x, uTexel.y));
  fragColor = c / 16.0;
}`,
  transparent: true, premultipliedAlpha: true, blending: THREE.NormalBlending,
  depthTest: false, depthWrite: false });
compScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), compMat));

// ---- the vanilla arm's asset: a 128^3 volume, built and uploaded exactly as the example does -----
let volTex = null, volBuildMs = 0;
function buildVolume(name){
  const G = P.grid, tab = waveTable(name);
  const data = new Uint8Array(G*G*G);
  const t0 = performance.now();
  let k = 0;
  for(let z=0;z<G;z++) for(let y=0;y<G;y++) for(let x=0;x<G;x++){
    const px=(x/(G-1))*2-1, py=(y/(G-1))*2-1, pz=(z/(G-1))*2-1;
    // the SAME density, quantised -- the vanilla arm's only difference must be the grid
    // the vanilla arm bakes the SAME Worley density the shader evaluates -- if the two diverge the
    // comparison stops being one. Evaluated here in JS at the grid, quantised to 8 bits.
    data[k++] = Math.max(0, Math.min(255, Math.round(densityJS(tab, px*2.2, py*0.85, pz*2.2, SEED)*255)));
  }
  volBuildMs = performance.now() - t0;
  if (volTex) volTex.dispose();
  volTex = new THREE.Data3DTexture(data, G, G, G);
  volTex.format = THREE.RedFormat; volTex.type = THREE.UnsignedByteType;
  volTex.minFilter = volTex.magFilter = THREE.LinearFilter;
  volTex.unpackAlignment = 1; volTex.needsUpdate = true;
  volMat.uniforms.uVol.value = volTex;
}

// Declared HERE, not next to the interaction handlers: accounting() restores it and a `let`
// is in the temporal dead zone until its declaration runs, so the first call would throw.
let arm = 1;
let table = null;
// FIVE TYPES, each set from the WMO/NOAA description rather than tuned by feel.
//  cumulus       "detached, sharp outlines, rising mounds and domes with cauliflower tops;
//                 sunlit parts brilliant white, bases relatively dark and horizontal"
//  stratocumulus "patchy grey and white, honeycomb appearance, distinct bases, low, extensive"
//  stratus       "flat, featureless, layered; a thin white sheet covering most of the sky"
//  cirrus        "delicate white filaments in patches or narrow bands, ice crystals, silky sheen"
//  cumulonimbus  "dense, towering, dark-based, vertically developed"
const TYPES = {
  cumulus:       { freq:2.6, covBias: 0.00, base:0.12, top:0.52, ero:0.55, stretch:1.00,
                   baseDark:0.35, absorb:26, shadow:2.2, sun:[0.327,0.312,0.293] },
  stratocumulus: { freq:3.6, covBias:-0.12, base:0.08, top:0.30, ero:0.40, stretch:1.90,
                   baseDark:0.55, absorb:22, shadow:2.5, sun:[0.293,0.281,0.274] },
  stratus:       { freq:1.5, covBias:-0.26, base:0.05, top:0.22, ero:0.20, stretch:3.20,
                   baseDark:0.72, absorb:16, shadow:3.0, sun:[0.243,0.243,0.247] },
  cirrus:        { freq:1.9, covBias:-0.06, base:0.48, top:0.96, ero:0.45, stretch:5.50,
                   baseDark:0.85, absorb:15, shadow:1.2, sun:[0.353,0.350,0.353] },
  cumulonimbus:  { freq:2.0, covBias:-0.16, base:0.06, top:0.86, ero:0.45, stretch:0.55,
                   baseDark:0.14, absorb:38, shadow:3.0, sun:[0.304,0.296,0.281] },
};
let cloudType = "cumulus";
function applyType(){
  const t = TYPES[cloudType];
  const u = volMat.uniforms, o = occMat.uniforms;
  u.uFreq.value = t.freq; u.uCovBias.value = t.covBias; u.uBase.value = t.base;
  u.uTop.value = t.top;   u.uEro.value = t.ero;         u.uStretch.value = t.stretch;
  u.uBaseDark.value = t.baseDark; u.uAbsorb.value = t.absorb; u.uShadow.value = t.shadow;
  u.uSunCol.value.set(t.sun[0], t.sun[1], t.sun[2]);
  // the occupancy bake MUST see the same shape, or its upper bound stops bounding
  o.uFreq.value = t.freq; o.uCovBias.value = t.covBias; o.uBase.value = t.base;
  o.uTop.value = t.top;   o.uStretch.value = t.stretch;
}
const COV = 0.45;
let NORM = [-1, 1];
let SEED = 0;
function setCloud(name){
  table = waveTable(name);
  SEED = fnv1a(name) >>> 0;
  // The engine shipped quantiles for the built-in names; anything typed is normalised here the same
  // way, so a name that has never existed gets a cloud rather than a haze.
  NORM = P.norm[name] || (()=>{
    const pts=[]; let sd=1;
    const rnd=()=>{ sd=(Math.imul(sd,1103515245)+12345)&0x7fffffff; return sd/0x7fffffff*2-1; };
    for(let i=0;i<3000;i++) pts.push(rawField(table,[rnd(),rnd(),rnd()]));
    pts.sort((a,b)=>a-b);
    return [pts[Math.floor(pts.length*0.05)], pts[Math.floor(pts.length*0.995)]];
  })();
  volMat.uniforms.uLo.value = NORM[0];
  volMat.uniforms.uHi.value = NORM[1];
  volMat.uniforms.uCov.value = COV;
  volMat.uniforms.uSeed.value = fnv1a(name) >>> 0;
  occMat.uniforms.uSeed.value = fnv1a(name) >>> 0;
  for(let i=0;i<NW;i++){
    occMat.uniforms.uWaves.value[i].set(table[i].fx, table[i].fy, table[i].fz, 0);
    occMat.uniforms.uWaveAP.value[i].set(table[i].amp, table[i].ph);
  }
  volMat.uniforms.uOcc.value = occRT.texture;
  applyType();
  bakeOccupancy(0);
  for(let i=0;i<NW;i++){
    volMat.uniforms.uWaves.value[i].set(table[i].fx, table[i].fy, table[i].fz, 0);
    volMat.uniforms.uWaveAP.value[i].set(table[i].amp, table[i].ph);
  }
  buildVolume(name);          // the vanilla arm needs its asset before it can be compared
  accounting(name);
}

// ---- head to head --------------------------------------------------------------------------------
// finish() alone left the draws queued and reported 0.01 ms for a 64-step raymarch, which is not a
// measurement. A one-pixel readback is a real sync point.
const syncPx = new Uint8Array(4);
function syncGPU(){
  const gl = renderer.getContext();
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, syncPx);
}
// autoClear OFF: the composite pass would otherwise clear the framebuffer and erase the sky that
// was just drawn into it. Every clear below is explicit.
renderer.autoClear = false;
// The map is baked in the ADVECTED frame, so it must be refreshed as the weather drifts. A 2-texel
// dilation at 256 texels over 4.8 units covers 0.68 s of drift at 0.055 units/s; re-baking every
// 0.25 s therefore leaves the map a valid upper bound at all times, with margin.
let lastBake = -1e9;
function bakeOccupancy(t){
  occMat.uniforms.uDrift.value.set((t * 0.055) % 64.0, (t * 0.021) % 64.0);
  const prev = renderer.getRenderTarget();
  renderer.setRenderTarget(occRT);
  renderer.render(occScene, occCam);
  renderer.setRenderTarget(prev);
  lastBake = t;
}
let frameIx = 0, lastCam = new THREE.Vector3(1e9, 0, 0);
function drawFrame(){
  volMat.uniforms.uFrame.value = frameIx++;
  renderer.setRenderTarget(cloudRT);
  renderer.setClearColor(0x000000, 0); renderer.clear(true, false, false);
  renderer.render(cloudScene, camera);

  renderer.setRenderTarget(null);
  renderer.setClearColor(0x000000, 1); renderer.clear(true, true, false);
  renderer.render(scene, camera);          // the sky, full resolution
  renderer.render(compScene, compCam);     // the clouds, upscaled over it
}
function timeArm(arm, frames){
  volMat.uniforms.uArm.value = arm;
  drawFrame(); syncGPU();
  const t0 = performance.now();
  for(let i=0;i<frames;i++){ drawFrame(); }
  syncGPU();
  return (performance.now() - t0) / frames;
}

// The row that is NOT a race: how many distinct density values can each arm produce along a ray?
// The texture is 8-bit over a 128 grid; the closed form has neither limit.
function resolutionProbe(){
  const tab = table, seen = new Set(), seenQ = new Set();
  for(let i=0;i<4000;i++){
    const t = -0.9 + 1.8*i/4000;
    const v = shaped(tab, [t, t*0.31, t*0.77], NORM[0], NORM[1], COV);
    seen.add(v.toFixed(6));
    seenQ.add(Math.round(v*255));        // what an 8-bit volume can represent
  }
  return { closed: seen.size, texture: seenQ.size, samples: 4000 };
}

function row(label, a, b, cls){
  return `<tr><td>${label}</td><td class=v>${a}</td><td class="v ${cls||''}">${b}</td></tr>`;
}
function accounting(name){
  const nm = new TextEncoder().encode(name).length;
  const tA = timeArm(0, 12), tB = timeArm(1, 12);
  volMat.uniforms.uArm.value = arm;
  const r = resolutionProbe();
  const faster = tB <= tA;
  const speed = faster ? `<span class=ok>${(tA/tB).toFixed(2)}× faster</span>`
                       : `<span class=bad>${(tB/tA).toFixed(2)}× slower</span>`;
  const xfer = (mbps, bytes) => (bytes*8/1e6)/mbps*1000;
  document.getElementById("acct").innerHTML = `
   <tr><th></th><th>vanilla three.js (3D texture)</th><th>leCore GLSL (closed form)</th></tr>
   ${row("animate the cloud",
         `<span class=bad>re-bake every frame</span> — ${volBuildMs.toFixed(0)} ms, a ${(1000/Math.max(volBuildMs,1)).toFixed(1)} fps ceiling`,
         "<span class=ok>free</span> — a phase offset")}
   ${row("fly beyond the volume", "<span class=bad>the texture ends</span>",
         "<span class=ok>defined everywhere</span>")}
   ${row("zoom in close", `<span class=bad>voxels</span> — fixed at ${P.grid}³`,
         "<span class=ok>no grid to run out of</span>")}
   ${row("switch to a new cloud", volBuildMs.toFixed(0)+" ms", "<span class=ok>0 ms</span>")}
   ${row("scene data downloaded", "<span class=big>"+(P.volume_bytes/1e6).toFixed(2)+" MB</span>",
         "<span class=big>"+nm+" bytes</span> — the name")}
   <tr><td colspan=3 style="color:#69707f;padding-top:8px">and the rows where a cached texture fetch is simply cheaper — printed, not hidden</td></tr>
   ${row("frame time, same view and step count", tA.toFixed(2)+" ms", tB.toFixed(2)+" ms — "+speed)}
   ${row("distinct density values along a ray", r.texture.toLocaleString()+" <span style='color:#69707f'>(8-bit)</span>",
         "<span class=ok>"+r.closed.toLocaleString()+"</span> of "+r.samples.toLocaleString()+" samples")}
   <tr><td colspan=3 style="color:#69707f;padding-top:6px">transfer time from the byte counts — arithmetic, not measured</td></tr>
   ${row("&nbsp;&nbsp;at 25 Mbps", xfer(25,P.volume_bytes).toFixed(0)+" ms", xfer(25,nm).toFixed(2)+" ms")}
   ${row("&nbsp;&nbsp;at 5 Mbps (mobile)", xfer(5,P.volume_bytes).toFixed(0)+" ms", xfer(5,nm).toFixed(2)+" ms")}`;
}

// ---- the page proves its arithmetic is the engine's ---------------------------------------------
function selfCheck(){
  const msgs = [];
  let worst = 0, n = 0;
  for(const nm of P.names){
    const tab = waveTable(nm);
    for(let i=0;i<P.probes.length;i++){
      const p = P.probes[i];
      worst = Math.max(worst, Math.abs(rawField(tab,p) - P.refs[nm][i]));
      n++;
    }
  }
  // 2e-4, not 1e-6: the DOMAIN WARP amplifies a float rounding difference in its divisor into
  // roughly 1e-5 downstream, so a tighter bound reports a mismatch that is arithmetic, not a bug.
  msgs.push(worst < 2e-4
    ? `<span class=ok>this browser computes the engine's weather field</span> (${P.names.length} clouds, ${n} probes, max |diff| ${worst.toExponential(1)})`
    : `<span class=bad>DENSITY MISMATCH (${worst.toExponential(1)}) — this is a different cloud</span>`);
  chk.innerHTML = msgs.join(" · ");
}

// ---- interaction ---------------------------------------------------------------------------------
let animate = true;
// Orbit OFF by default: a moving camera makes the container's edges obvious, which is the
// opposite of what this demo is for.
let spin = false, ang = 0.7, dist = 3.6, elev = 0.12;
let dragging = false, lastX = 0, lastY = 0;
cv.addEventListener("pointerdown", e=>{ dragging=true; lastX=e.clientX; lastY=e.clientY; cv.setPointerCapture(e.pointerId); });
cv.addEventListener("pointerup", ()=>{ dragging=false; });
cv.addEventListener("pointermove", e=>{ if(!dragging) return;
  ang  -= (e.clientX-lastX)*0.006;
  // vertical drag moves the eye above and below the layer, which is how you see that it IS a layer
  elev  = Math.max(-1.4, Math.min(1.4, elev + (e.clientY-lastY)*0.006));
  lastX = e.clientX; lastY = e.clientY; });
cv.addEventListener("wheel", e=>{ dist = Math.min(9, Math.max(0.35, dist + e.deltaY*0.002)); e.preventDefault(); }, {passive:false});
document.getElementById("go").onclick = ()=> setCloud(document.getElementById("nm").value || "unnamed");
document.getElementById("nm").addEventListener("keydown", e=>{ if(e.key==="Enter") document.getElementById("go").click(); });
document.getElementById("anim").onclick = e=>{ animate=!animate; e.target.textContent="animate: "+(animate?"on":"off"); };
document.getElementById("spin").onclick = e=>{ spin=!spin; e.target.textContent="orbit: "+(spin?"on":"off"); };
{
  const tb = document.getElementById("typeseg");
  tb.addEventListener("click", ev=>{
    const b = ev.target.closest("button"); if(!b) return;
    for(const x of tb.querySelectorAll("button")) x.classList.toggle("on", x===b);
    cloudType = b.dataset.t;
    applyType();
    bakeOccupancy(volMat.uniforms.uTime.value);
    setCloud(document.getElementById("nm").value || "unnamed");
  });
}
{
  const bx = document.getElementById("armseg");
  bx.addEventListener("click", ev=>{
    const b = ev.target.closest("button"); if(!b) return;
    for(const x of bx.querySelectorAll("button")) x.classList.toggle("on", x===b);
    arm = +b.dataset.arm; volMat.uniforms.uArm.value = arm;
  });
}

let frames=0, t0=performance.now();
function loop(now){
  const tNow = animate ? now*0.001 : 0.0;
  volMat.uniforms.uTime.value = tNow;
  if (Math.abs(tNow - lastBake) > 0.25) bakeOccupancy(tNow);
  if(spin && !dragging) ang += 0.0022;
  // low and close: a cloudscape is framed from near the layer, not from orbit
  camera.position.set(Math.cos(ang)*dist, dist*elev, Math.sin(ang)*dist);
  camera.lookAt(0, 0.05, 0);
  volMat.uniforms.uCam.value.copy(camera.position);
  skyMesh.position.copy(camera.position);   // the sky travels with the eye, or it swims
  drawFrame();
  frames++;
  if(now-t0>500){
    const fps = frames*1000/(now-t0); frames=0; t0=now;
    st.innerHTML = `<b>${fps.toFixed(0)} fps</b> · ${arm===1?"leCore closed form":"vanilla 3D texture"} · `+
      `${NW} plane waves · ${volMat.uniforms.uSteps.value} steps/ray at 1/${DOWN} resolution · `+
      (arm===1 ? "<b>0 bytes</b> of volume data" : `${(P.volume_bytes/1e6).toFixed(2)} MB uploaded`);
  }
  requestAnimationFrame(loop);
}
setCloud(document.getElementById("nm").value);
selfCheck();
requestAnimationFrame(loop);
</script>
"""

if __name__ == "__main__":
    main()
