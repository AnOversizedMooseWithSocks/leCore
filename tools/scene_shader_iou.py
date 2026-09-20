"""scene_shader_iou.py -- I1 acceptance (needs Playwright Chromium; the speaker scene from /home/claude/speaker_render.py): the SAME exact scene, compiled to GLSL and hosted in WebGL2, must draw
the same silhouette the engine's own sphere-trace draws through the same camera. Per-pixel nearest-part id
from both sides, compared as a mask (speaker parts vs floor) -> IoU, plus per-part id agreement.

The shader is not hand-written: every part's map() comes from the engine's own emitter (sdf_shader), renamed
map0..mapN and combined by min; the host computes ray directions from exactly the basis Camera.ray_dirs uses.
"""
import io
import json
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/claude/leCore")
sys.argv = ["x"]
import speaker_render as S
import lecore
from holographic.rendering.holographic_render import Camera

W, H = 400, 250
m = lecore.UnifiedMind(dim=256, seed=0)
parts = S.parts
names = [p[0] for p in parts]
FLOOR = names.index("floor")

# ---- 1. the emitted maps, one per part, renamed ------------------------------------------------
helpers, maps = {}, []
for i, (name, sdf, _) in enumerate(parts):
    src = m.sdf_shader(sdf)
    body = src[:src.find("vec3 calcNormal")]
    for h in re.findall(r"(float sd\w+\([^)]*\)\s*\{[^\n]*\})", body):       # primitive helpers, deduped by text
        helpers[h.split("(")[0]] = h
    mm = re.search(r"float map\(vec3 p\)\s*\{(.*?)\n\}", body, re.S)
    maps.append("float map%d(vec3 p){%s\n}" % (i, mm.group(1)))
glsl_parts = "\n".join(helpers.values()) + "\n" + "\n".join(maps)
scene_fn = "float mapAll(vec3 p, out int id){ float d=1e9; id=-1;\n" + "\n".join(
    "  { float di=map%d(p); if(di<d){ d=di; id=%d; } }" % (i, i) for i in range(len(parts))) + "\n  return d; }"

cam = Camera(eye=S.cam.eye, target=S.cam.target, fov_deg=S.cam.fov_deg, aspect=W / H)
r, u, f = cam._basis()
t = float(np.tan(np.radians(cam.fov_deg) / 2.0))
frag = """#version 300 es
precision highp float; precision highp int;
uniform vec3 uEye, uR, uU, uF; uniform float uTan, uAspect; uniform vec2 uRes;
out vec4 o;
%s
%s
void main(){
  vec2 px = gl_FragCoord.xy;                              // WebGL y is up; Camera.ray_dirs has row 0 at the TOP
  float ndc_x = (2.0*(px.x+0.5)/uRes.x - 1.0)*uAspect*uTan;
  float ndc_y = (2.0*(px.y+0.5)/uRes.y - 1.0)*uTan;
  vec3 dir = normalize(ndc_x*uR + ndc_y*uU + uF);
  float tt = 0.0; int id = -1; int hit = -1;
  for(int i=0;i<256;i++){ vec3 p = uEye + tt*dir; int k; float d = mapAll(p, k); if(d < 0.0005){ hit = k; break; } tt += d; if(tt > 40.0) break; }
  float g = hit < 0 ? 0.0 : float(hit + 1) / 32.0;
  o = vec4(g, g, g, 1.0);
}""" % (glsl_parts, scene_fn)
html = """<!doctype html><html><body style="margin:0"><canvas id=c width=%d height=%d></canvas><script>
const gl = document.getElementById('c').getContext('webgl2', {preserveDrawingBuffer:true, antialias:false});
function sh(t, s){ const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if(!gl.getShaderParameter(x, gl.COMPILE_STATUS)) { document.title = 'ERR:' + gl.getShaderInfoLog(x); } return x; }
const vs = sh(gl.VERTEX_SHADER, '#version 300 es\\nvoid main(){ vec2 v = vec2((gl_VertexID&1)*4-1, (gl_VertexID&2)*2-1); gl_Position = vec4(v, 0, 1); }');
const fs = sh(gl.FRAGMENT_SHADER, %s);
const pr = gl.createProgram(); gl.attachShader(pr, vs); gl.attachShader(pr, fs); gl.linkProgram(pr); gl.useProgram(pr);
const U = n => gl.getUniformLocation(pr, n);
gl.uniform3f(U('uEye'), %s); gl.uniform3f(U('uR'), %s); gl.uniform3f(U('uU'), %s); gl.uniform3f(U('uF'), %s);
gl.uniform1f(U('uTan'), %r); gl.uniform1f(U('uAspect'), %r); gl.uniform2f(U('uRes'), %d, %d);
gl.viewport(0,0,%d,%d); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.finish(); document.title = document.title || 'OK';
</script></body></html>""" % (W, H, json.dumps(frag), ",".join("%r" % float(v) for v in cam.eye), ",".join("%r" % float(v) for v in r),
                                 ",".join("%r" % float(v) for v in u), ",".join("%r" % float(v) for v in f), t, W / H, W, H, W, H)
open("/home/claude/speaker_shader.html", "w").write(html)

# ---- 2. WebGL render through Playwright -----------------------------------------------------
from playwright.sync_api import sync_playwright
t0 = time.time()
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--no-sandbox", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": W, "height": H})
    pg.goto("file:///home/claude/speaker_shader.html")
    pg.wait_for_timeout(800)
    title = pg.title()
    png = pg.locator("#c").screenshot()
    b.close()
if title.startswith("ERR"):
    raise SystemExit("shader failed to compile: " + title[:800])
gl_img = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=float)
gl_id = np.rint(gl_img / 255.0 * 32.0).astype(int) - 1                 # -1 = miss
print("WebGL raymarch of the emitted maps: %.1fs at %dx%d" % (time.time() - t0, W, H))

# ---- 3. the engine's own sphere-trace of the same scene, same camera -------------------------
eye, dirs = cam.ray_dirs(W, H)
D = dirs.reshape(-1, 3)
P = np.broadcast_to(np.asarray(eye, float), D.shape).copy()
tt = np.zeros(len(D)); hit = -np.ones(len(D), int); alive = np.ones(len(D), bool)
t0 = time.time()
for _ in range(256):
    idx = np.nonzero(alive)[0]
    if not idx.size:
        break
    Q = P[idx] + tt[idx, None] * D[idx]
    ds = S.scene._stack(Q)                        # (parts, n)
    k = ds.argmin(0); d = ds[k, np.arange(len(idx))]
    done = d < 0.0005
    hit[idx[done]] = k[done]; alive[idx[done]] = False
    tt[idx] += d
    alive[idx[tt[idx] > 40.0]] = False
ref_id = hit.reshape(H, W)
print("NumPy sphere-trace of the same scene: %.1fs" % (time.time() - t0))

# ---- 4. compare --------------------------------------------------------------------------------
ref_mask = (ref_id >= 0) & (ref_id != FLOOR)
gl_mask = (gl_id >= 0) & (gl_id != FLOOR)
inter = (ref_mask & gl_mask).sum(); union = (ref_mask | gl_mask).sum()
same_id = (ref_id == gl_id).mean()
print("SILHOUETTE IoU (speaker vs floor/background): %.4f   (acceptance >= 0.98)" % (inter / union))
print("per-pixel nearest-part id agreement: %.4f" % same_id)
print("speaker pixels: ref %d, shader %d" % (ref_mask.sum(), gl_mask.sum()))
Image.fromarray((np.clip(gl_id + 1, 0, 31) * 8).astype(np.uint8)).save("/home/claude/speaker_shader_ids.png")
Image.fromarray((np.clip(ref_id + 1, 0, 31) * 8).astype(np.uint8)).save("/home/claude/speaker_ref_ids.png")
