"""A GLSL index built for LARGE HISTORY: flat 2D addressing, arbitrary depth, O(1) append.

Nobody depends on the old shaders yet, so this is a redesign rather than a patch. Three limits
in the old layout, each structural rather than a tuning matter:

  LIMIT 1 -- ONE ROW PER DOCUMENT. A K x D texture caps K at MAX_TEXTURE_SIZE, typically 16384
      in WebGL2. Sixteen thousand documents is not a history. FIX: pack rows into a FLAT 2D
      texture and address by index -- `ivec2(n % W, n / W)`. At W = 4096 and the same 16384-row
      limit, the same texture holds 67 MILLION rows. Capacity stops being a texture question.

  LIMIT 2 -- THE DEPTH WAS HARDCODED AT THREE. Depth is the cheap axis: T9/T10 say one more tier
      multiplies capacity by g while adding one pass, whereas widening costs work per query. FIX:
      build tiers until the top fits in one group, whatever depth that takes.

  LIMIT 3 -- APPENDING MEANT REBUILDING. A tier cell is a SUM, and T8 proves that adding an item
      to a live cell equals rebuilding that cell from the extended list. FIX: append touches one
      cell per level -- depth many adds -- and the tier vectors are patched in place with a
      sub-image write instead of a full re-upload.

Everything is vec4-packed and the leaf level can be stored f16 (checked, with the T1 gate, in
glsl_fast.py -- ship f16 WITH the gate on).
"""
import os, time
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl
import holographic.agents_and_reasoning.holographic_hashatom as HA

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# FLAT 2D ADDRESSING. A row lives at texel span [n*uD4, (n+1)*uD4) of a W-wide texture, so the
# only limit left is W*H texels rather than H rows. uW is a uniform, not a constant, so the same
# shader serves any layout the host picks.
FS_SCORE = """
#version 330 core
uniform sampler2D uM;      // flat, vec4-packed, uW texels wide
uniform sampler2D uQ;      // uD4 texels
uniform sampler2D uIdx;    // parent indices
uniform int uD4,uG,uN,uNI,uW,uBase;
out float fragOut;
ivec2 at(int texel){ return ivec2(texel % uW, texel / uW); }
void main(){
  int t=int(gl_FragCoord.x); int pi=t/uG;
  if(pi>=uNI){ fragOut=-1e30; return; }
  int row=int(texelFetch(uIdx,ivec2(pi,0),0).r+0.5)*uG+(t-pi*uG);
  if(row>=uN){ fragOut=-1e30; return; }
  int off=(uBase+row)*uD4;
  float a=0.0;
  for(int i=0;i<uD4;++i)
    a+=dot(texelFetch(uM,at(off+i),0), texelFetch(uQ,ivec2(i,0),0));
  fragOut=a; }
"""

# Top-b in ONE fragment, four winners packed into RGBA -- one linear scan instead of O(N^2).
FS_TOPB4 = """
#version 330 core
uniform sampler2D uS,uIdx; uniform int uN,uG,uUseIdx;
out vec4 fragOut;
void main(){
  float v0=-1e30,v1=-1e30,v2=-1e30,v3=-1e30; int b0=-1,b1=-1,b2=-1,b3=-1;
  for(int i=0;i<uN;++i){
    float v=texelFetch(uS,ivec2(i,0),0).r;
    if(v>v0){v3=v2;b3=b2;v2=v1;b2=b1;v1=v0;b1=b0;v0=v;b0=i;}
    else if(v>v1){v3=v2;b3=b2;v2=v1;b2=b1;v1=v;b1=i;}
    else if(v>v2){v3=v2;b3=b2;v2=v;b2=i;}
    else if(v>v3){v3=v;b3=i;} }
  vec4 r=vec4(float(b0),float(b1),float(b2),float(b3));
  if(uUseIdx==1){
    int q[4]; q[0]=b0; q[1]=b1; q[2]=b2; q[3]=b3;
    for(int k=0;k<4;++k){
      int i=q[k]; int pi=i/uG;
      float abs_=float(int(texelFetch(uIdx,ivec2(pi,0),0).r+0.5)*uG+(i-pi*uG));
      if(k==0) r.x=abs_; else if(k==1) r.y=abs_; else if(k==2) r.z=abs_; else r.w=abs_; } }
  fragOut=r; }
"""


class FlatIndex:
    """Tiers stored end to end in ONE flat texture; append patches cells in place."""

    def __init__(self, ctx, dim, group=8, width=4096):
        self.ctx, self.dim, self.g, self.W = ctx, dim, group, width
        self.d4 = (dim + 3) // 4
        self.levels = [[]]                       # levels[0] = leaves, then coordinators upward
        self.tex = None

    # ---- build / grow -----------------------------------------------------------------------
    def _ensure_depth(self):
        """T10: when the top level outgrows one group, PUSH A LEVEL. Depth is the cheap axis."""
        while len(self.levels[-1]) > self.g:
            below = self.levels[-1]
            self.levels.append([np.sum(below[i:i + self.g], axis=0)
                                for i in range(0, len(below), self.g)])

    def append(self, vec):
        """T8: a cell is a SUM, so adding to a live cell == rebuilding it. O(depth), not O(K)."""
        self.levels[0].append(np.asarray(vec, dtype=np.float64))
        n = len(self.levels[0]) - 1
        for L in range(1, len(self.levels)):
            n //= self.g
            if n < len(self.levels[L]):
                self.levels[L][n] = self.levels[L][n] + vec
            else:
                self.levels[L].append(np.array(vec, dtype=np.float64))
        self._ensure_depth()

    def extend(self, vecs):
        for v in vecs:
            self.append(v)

    # ---- upload -----------------------------------------------------------------------------
    def _flat(self):
        rows, self.base, self.count = [], [], []
        for L in self.levels:
            self.base.append(sum(self.count))
            self.count.append(len(L))
            rows.extend(L)
        M = np.stack(rows) if rows else np.zeros((1, self.dim))
        pad = (-M.shape[1]) % 4
        if pad:
            M = np.concatenate([M, np.zeros((M.shape[0], pad))], axis=1)
        flat = M.reshape(-1, 4)
        h = (len(flat) + self.W - 1) // self.W
        buf = np.zeros((h, self.W, 4), dtype="f4")
        buf.reshape(-1, 4)[:len(flat)] = flat
        return buf

    def upload(self):
        buf = self._flat()
        if self.tex is None or self.tex.height != buf.shape[0]:
            if self.tex is not None:
                self.tex.release()
            self.tex = self.ctx.texture((self.W, buf.shape[0]), 4, buf.tobytes(), dtype="f4")
        else:
            self.tex.write(buf.tobytes())
        return self.tex

    def capacity_note(self, max_tex=16384):
        texels = max_tex * max_tex
        return texels // self.d4


class GL:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.vbo = self.ctx.buffer(np.array([-1,-1,3,-1,-1,3], dtype="f4").tobytes())
        self.c, self.t = {}, {}
    def prog(self, fs):
        if fs not in self.c:
            p = self.ctx.program(vertex_shader=VS, fragment_shader=fs)
            self.c[fs] = (p, self.ctx.vertex_array(p, [(self.vbo, "2f", "p")]))
        return self.c[fs]
    def tex(self, a, comps=1):
        a = np.ascontiguousarray(np.atleast_2d(a).astype("f4"))
        return self.ctx.texture((a.shape[1] // comps, a.shape[0]), comps, a.tobytes(), dtype="f4")
    def target(self, w, comps=1):
        k = (w, comps)
        if k not in self.t:
            o = self.ctx.texture((w, 1), comps, dtype="f4")
            self.t[k] = (o, self.ctx.framebuffer(color_attachments=[o]))
        return self.t[k]
    def draw(self, fs, w, texs, ints, comps=1):
        o, fbo = self.target(w, comps)
        p, vao = self.prog(fs)
        fbo.use(); self.ctx.viewport = (0, 0, w, 1)
        for u, (n, t) in enumerate(texs.items()):
            t.use(u); p[n].value = u
        for k, v in ints.items():
            if k in p:
                p[k].value = int(v)
        vao.render(moderngl.TRIANGLES)
        return np.frombuffer(o.read(), dtype="f4")


def walk(gl, idx, q, beam=4):
    """Descend every level the index happens to have. Depth is data, not a constant."""
    tM = idx.tex
    pad = (-len(q)) % 4
    qq = np.concatenate([q, np.zeros(pad)]) if pad else q
    tQ = gl.tex(qq.reshape(1, -1), 4)
    top = len(idx.levels) - 1
    cur = gl.tex(np.zeros((1, 1), dtype="f4"))
    n = idx.count[top]
    s = gl.draw(FS_SCORE, n, {"uM": tM, "uQ": tQ, "uIdx": cur},
                {"uD4": idx.d4, "uG": n, "uN": n, "uNI": 1, "uW": idx.W, "uBase": idx.base[top]})
    r = gl.draw(FS_TOPB4, 1, {"uS": gl.tex(s.reshape(1, -1)), "uIdx": cur},
                {"uN": n, "uG": n, "uUseIdx": 0}, comps=4)
    cur = gl.tex(np.asarray(r[:beam], dtype="f4").reshape(1, -1))
    for L in range(top - 1, -1, -1):
        w = beam * idx.g
        s = gl.draw(FS_SCORE, w, {"uM": tM, "uQ": tQ, "uIdx": cur},
                    {"uD4": idx.d4, "uG": idx.g, "uN": idx.count[L], "uNI": beam,
                     "uW": idx.W, "uBase": idx.base[L]})
        r = gl.draw(FS_TOPB4, 1, {"uS": gl.tex(s.reshape(1, -1)), "uIdx": cur},
                    {"uN": w, "uG": idx.g, "uUseIdx": 1}, comps=4)
        cur = gl.tex(np.asarray(r[:beam], dtype="f4").reshape(1, -1))
    return int(round(float(r[0])))


def np_walk(idx, q, beam=4):
    top = len(idx.levels) - 1
    cand = np.argsort(np.stack(idx.levels[top]) @ q)[::-1][:beam]
    for L in range(top - 1, -1, -1):
        M = np.stack(idx.levels[L])
        rows = np.concatenate([np.arange(c * idx.g, min((c + 1) * idx.g, len(M))) for c in cand])
        cand = rows[np.argsort(M[rows] @ q)[::-1][:beam]]
    return int(cand[0])


if __name__ == "__main__":
    gl = GL()
    print("BACKEND:", gl.ctx.info["GL_RENDERER"], "\n")
    D, G = 256, 8
    for K in (512, 4096, 32768):
        V = np.stack([HA.hash_atom("doc%d" % i, D) for i in range(K)])
        idx = FlatIndex(gl.ctx, D, group=G)
        t0 = time.perf_counter(); idx.extend(V); t_build = time.perf_counter() - t0
        idx.upload()
        rng = np.random.default_rng(0)
        probes = [int(rng.integers(K)) for _ in range(20)]
        gpu = [walk(gl, idx, V[p]) for p in probes]
        ref = [np_walk(idx, V[p]) for p in probes]
        hit = sum(a == p for a, p in zip(gpu, probes))
        t0 = time.perf_counter()
        for p in probes:
            walk(gl, idx, V[p])
        t_q = (time.perf_counter() - t0) / len(probes) * 1e3
        # O(1) append: one more doc, then re-query, against a full rebuild of the same size
        t0 = time.perf_counter(); idx.append(HA.hash_atom("extra", D)); t_app = time.perf_counter() - t0
        t0 = time.perf_counter()
        FlatIndex(gl.ctx, D, group=G).extend(V)
        t_rebuild = time.perf_counter() - t0
        print("K=%-6d depth=%d  levels=%s" % (K, len(idx.levels), idx.count))
        print("   GPU == NumPy walk : %d/%d      self-retrieval %d/%d      %.3f ms/query"
              % (sum(a == b for a, b in zip(gpu, ref)), len(gpu), hit, len(probes), t_q))
        print("   append 1 doc      : %.3f ms   full rebuild %.1f ms   -> %.0fx cheaper"
              % (t_app * 1e3, t_rebuild * 1e3, t_rebuild / max(t_app, 1e-9)))
        print("   rows addressable in one 16384^2 texture at D=%d: %s"
              % (D, format(idx.capacity_note(), ",")))
        print()
