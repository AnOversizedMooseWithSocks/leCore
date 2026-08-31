"""Benchmark the GLSL read path -- honestly.

WHAT CANNOT BE MEASURED HERE, said first so no number below gets misread: there is no GPU. Mesa
llvmpipe is a SOFTWARE RASTERISER running on the same CPU as NumPy, so any wall time is a
CPU-vs-CPU comparison and says NOTHING about throughput on real hardware. The GPU crossover
question stays open and still needs the A4500.

WHAT CAN BE MEASURED HERE, and is hardware-independent or nearly so:
  A. WORK COUNTS -- dot products, passes, texel fetches per query. Exact arithmetic, no timing.
  B. SCALING SLOPE in corpus size K. A slope is far more transferable than an absolute time: if
     the flat scan is O(K) and the tier walk is O(1) in K, that structure holds on any backend.
  C. READBACK SHARE -- how much of a query is glReadPixels rather than shading. This is the
     number that justifies the design decision to keep every between-level decision ON the GPU.
  D. BATCHED vs PER-QUERY -- one draw for all queries against one draw each, which is the
     difference between a benchmark and a deployment.
"""
import os, time
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl
import holographic.agents_and_reasoning.holographic_hashatom as HA

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

FS_GATHER = """
#version 330 core
uniform sampler2D uM,uQ,uIdx; uniform int uD,uG,uN,uNI;
out float fragOut;
void main(){ int t=int(gl_FragCoord.x); int pi=t/uG;
  if(pi>=uNI){fragOut=-1e30;return;}
  int row=int(texelFetch(uIdx,ivec2(pi,0),0).r+0.5)*uG+(t-pi*uG);
  if(row>=uN){fragOut=-1e30;return;}
  float a=0.0; for(int i=0;i<uD;++i)
    a+=texelFetch(uM,ivec2(i,row),0).r*texelFetch(uQ,ivec2(i,0),0).r;
  fragOut=a; }
"""

FS_TOPB = """
#version 330 core
uniform sampler2D uS,uIdx; uniform int uN,uG,uUseIdx;
out float fragOut;
void main(){ int want=int(gl_FragCoord.x);
  for(int i=0;i<uN;++i){ float v=texelFetch(uS,ivec2(i,0),0).r; int rank=0;
    for(int j=0;j<uN;++j){ float w=texelFetch(uS,ivec2(j,0),0).r;
      if(w>v||(w==v&&j<i)) rank+=1; }
    if(rank==want){ if(uUseIdx==0){fragOut=float(i);return;}
      int pi=i/uG;
      fragOut=float(int(texelFetch(uIdx,ivec2(pi,0),0).r+0.5)*uG+(i-pi*uG)); return; } }
  fragOut=-1.0; }
"""

# BATCHED flat scan: one draw for the whole query set. x = document, y = query. This is what a
# deployment does; the per-query loop above is what a benchmark does when nobody is looking.
FS_BATCH = """
#version 330 core
uniform sampler2D uM,uQ; uniform int uD;
out float fragOut;
void main(){
  int doc=int(gl_FragCoord.x), q=int(gl_FragCoord.y);
  float a=0.0;
  for(int i=0;i<uD;++i)
    a+=texelFetch(uM,ivec2(i,doc),0).r*texelFetch(uQ,ivec2(i,q),0).r;
  fragOut=a; }
"""


class GL:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.vbo = self.ctx.buffer(np.array([-1,-1,3,-1,-1,3], dtype="f4").tobytes())
        self.c = {}

    def prog(self, fs):
        if fs not in self.c:
            p = self.ctx.program(vertex_shader=VS, fragment_shader=fs)
            self.c[fs] = (p, self.ctx.vertex_array(p, [(self.vbo, "2f", "p")]))
        return self.c[fs]

    def tex(self, a):
        a = np.ascontiguousarray(np.atleast_2d(a).astype("f4"))
        return self.ctx.texture((a.shape[1], a.shape[0]), 1, a.tobytes(), dtype="f4")

    def target(self, w, h=1):
        out = self.ctx.texture((w, h), 1, dtype="f4")
        return out, self.ctx.framebuffer(color_attachments=[out])

    def draw(self, fs, out, fbo, texs, ints, read=True):
        p, vao = self.prog(fs)
        fbo.use(); self.ctx.viewport = (0, 0, out.width, out.height)
        for u, (n, t) in enumerate(texs.items()):
            t.use(u); p[n].value = u
        for k, v in ints.items():
            if k in p:
                p[k].value = int(v)
        vao.render(moderngl.TRIANGLES)
        if not read:
            self.ctx.finish()          # without this the timing measures QUEUEING, not work
            return None
        return np.frombuffer(out.read(), dtype="f4")


def bench(g, K, D=256, BEAM=4, reps=30):
    rng = np.random.default_rng(0)
    names = ["doc%d" % i for i in range(K)]
    V = np.stack([HA.hash_atom(n, D) for n in names])
    g_ = max(2, int(round(K ** (1/3))))
    ch = np.stack([V[i:i+g_].sum(0) for i in range(0, K, g_)])
    su = np.stack([ch[i:i+g_].sum(0) for i in range(0, len(ch), g_)])
    Q = np.stack([V[int(rng.integers(K))] for _ in range(reps)])

    tV, tC, tS = g.tex(V), g.tex(ch), g.tex(su)
    identK = g.tex(np.arange(K, dtype="f4"))
    identS = g.tex(np.arange(len(su), dtype="f4"))
    oK, fK = g.target(K); oS, fS = g.target(len(su))
    oB, fB = g.target(BEAM); oBG, fBG = g.target(BEAM*g_)
    o1, f1 = g.target(1)

    def flat(q, read=True):
        tQ = g.tex(q)
        s = g.draw(FS_GATHER, oK, fK, {"uM": tV, "uQ": tQ, "uIdx": identK},
                   {"uD": D, "uG": 1, "uN": K, "uNI": K}, read=read)
        tQ.release()
        return s

    def beam(q):
        tQ = g.tex(q)
        s1 = g.draw(FS_GATHER, oS, fS, {"uM": tS, "uQ": tQ, "uIdx": identS},
                    {"uD": D, "uG": 1, "uN": len(su), "uNI": len(su)})
        t1 = g.tex(s1)
        i1 = g.draw(FS_TOPB, oB, fB, {"uS": t1, "uIdx": identS},
                    {"uN": len(su), "uG": 1, "uUseIdx": 0})
        ti1 = g.tex(i1)
        s2 = g.draw(FS_GATHER, oBG, fBG, {"uM": tC, "uQ": tQ, "uIdx": ti1},
                    {"uD": D, "uG": g_, "uN": len(ch), "uNI": BEAM})
        t2 = g.tex(s2)
        i2 = g.draw(FS_TOPB, oB, fB, {"uS": t2, "uIdx": ti1},
                    {"uN": BEAM*g_, "uG": g_, "uUseIdx": 1})
        ti2 = g.tex(i2)
        s3 = g.draw(FS_GATHER, oBG, fBG, {"uM": tV, "uQ": tQ, "uIdx": ti2},
                    {"uD": D, "uG": g_, "uN": K, "uNI": BEAM})
        t3 = g.tex(s3)
        i3 = g.draw(FS_TOPB, o1, f1, {"uS": t3, "uIdx": ti2},
                    {"uN": BEAM*g_, "uG": g_, "uUseIdx": 1})
        for t in (tQ, t1, ti1, t2, ti2, t3):
            t.release()
        return i3

    def timed(fn, arg, n=None):
        n = n or reps
        fn(arg)                                    # warm the pipeline and the shader cache
        t0 = time.perf_counter()
        for i in range(n):
            fn(Q[i % len(Q)])
        return (time.perf_counter() - t0) / n * 1e3

    t_flat = timed(flat, Q[0])
    t_flat_noread = timed(lambda q: flat(q, read=False), Q[0])
    t_beam = timed(beam, Q[0])

    # NumPy baseline, same work, f64
    t0 = time.perf_counter()
    for i in range(reps):
        int(np.argmax(V @ Q[i % len(Q)]))
    t_np = (time.perf_counter() - t0) / reps * 1e3

    # batched: every query in one draw
    tQall = g.tex(Q)
    oBa, fBa = g.target(K, len(Q))
    g.draw(FS_BATCH, oBa, fBa, {"uM": tV, "uQ": tQall}, {"uD": D})
    t0 = time.perf_counter()
    for _ in range(5):
        g.draw(FS_BATCH, oBa, fBa, {"uM": tV, "uQ": tQall}, {"uD": D})
    t_batch = (time.perf_counter() - t0) / 5 / len(Q) * 1e3

    dots_flat, dots_beam = K, len(su) + 2 * BEAM * g_
    return dict(K=K, g=g_, dots_flat=dots_flat, dots_beam=dots_beam,
                t_np=t_np, t_flat=t_flat, t_flat_noread=t_flat_noread,
                t_beam=t_beam, t_batch=t_batch)


if __name__ == "__main__":
    g = GL()
    print("BACKEND:", g.ctx.info["GL_RENDERER"])
    print("*** SOFTWARE RASTERISER ON THE SAME CPU AS NUMPY -- times below are NOT GPU numbers.")
    print("*** Read the WORK COUNTS and the SLOPE; ignore the absolute milliseconds.\n")
    print("A + B: work per query, and how each scales with corpus size (D=256, beam=4)")
    print("  K      g   dots_flat  dots_beam   NumPy_f64  GL_flat  GL_flat(no readback)  GL_beam  GL_batched")
    res = []
    for K in (100, 200, 400, 800, 1600):
        r = bench(g, K)
        res.append(r)
        print("  %-6d %-3d %-10d %-11d %-10.3f %-8.3f %-21.3f %-8.3f %.4f"
              % (r["K"], r["g"], r["dots_flat"], r["dots_beam"], r["t_np"], r["t_flat"],
                 r["t_flat_noread"], r["t_beam"], r["t_batch"]))

    def slope(key):
        x = np.log2([r["K"] for r in res]); y = np.log2([max(r[key], 1e-9) for r in res])
        return float(np.polyfit(x, y, 1)[0])

    print("\nB: measured scaling exponent (d log time / d log K) -- the transferable number")
    for k, lbl in (("dots_flat", "flat, work"), ("dots_beam", "beam, work"),
                   ("t_flat", "flat, GL"), ("t_beam", "beam, GL"),
                   ("t_batch", "flat, GL batched"), ("t_np", "flat, NumPy")):
        print("   %-18s %+.2f" % (lbl, slope(k)))

    print("\nC: readback share of a single-query flat scan")
    for r in res:
        share = 100 * (r["t_flat"] - r["t_flat_noread"]) / r["t_flat"]
        print("   K=%-6d readback %5.1f%% of the query  (%.3f ms of %.3f ms)"
              % (r["K"], share, r["t_flat"] - r["t_flat_noread"], r["t_flat"]))

    print("\nD: batching, same flat scan, one draw for all queries")
    for r in res:
        print("   K=%-6d per-query %.4f ms batched vs %.3f ms one-at-a-time  -> %.0fx"
              % (r["K"], r["t_batch"], r["t_flat"], r["t_flat"] / max(r["t_batch"], 1e-9)))
