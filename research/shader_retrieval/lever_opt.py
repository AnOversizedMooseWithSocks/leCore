"""Walk the levers at the three limits the benchmark actually found. Measure each, keep the losses.

THE MEASURED LIMITS (not guessed -- from bench_glsl.py):
  L1  crossover: below K~150 the tier walk LOSES to a flat scan, because six fixed passes cost
      more than the dot products they save. T6 says the WORK crossover is K=27 at beam 4, so the
      gap between 27 and 150 is pure per-pass overhead. LEVER 5 (tile under an orchestrator,
      read backwards): FEWER, FATTER passes. Score-and-select fuse into one pass when the level
      is small, taking 6 passes to 4.
  L2  storage: the leaf matrix is K x D floats and dominates everything. LEVER 3 (determinism
      instead of storage) + LEVER 6 (a measured limit is a tile size): a doc vector is a SUM OF
      HASH ATOMS, so it is a FUNCTION of its token ids. Keep only the coordinator tiers and
      REGENERATE the beam's leaves from token ids in-shader. T7 says the coordinators cost at
      most 3/4 of the leaf level at ANY grouping >= 2, so this is a strict win in bytes.
  L3  per-draw overhead was ~1/3 of a query. Already answered by batching, measured at 2-3x, and
      not re-litigated here.

Every arm is checked for IDENTICAL ANSWERS first. A speedup that changes the answer is not a
speedup, and a compression that changes the answer is not compression.
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

# LEVER 5, READ BACKWARDS: a top level of only `uN` rows does not need a pass to score and
# another to select. Each output fragment recomputes all uN scores itself and emits the one whose
# rank it wants. That is uN x more arithmetic on a TINY level, traded against a whole pass and
# its render-target allocation -- the trade only pays while uN is small, which is exactly the
# regime the crossover lives in.
FS_SCORESEL = """
#version 330 core
uniform sampler2D uM,uQ; uniform int uD,uN;
out float fragOut;
void main(){
  int want=int(gl_FragCoord.x);
  int best=-1; float bv=0.0;
  for(int r=0;r<uN;++r){
    float a=0.0;
    for(int i=0;i<uD;++i) a+=texelFetch(uM,ivec2(i,r),0).r*texelFetch(uQ,ivec2(i,0),0).r;
    int rank=0;
    for(int s=0;s<uN;++s){
      if(s==r) continue;
      float b=0.0;
      for(int i=0;i<uD;++i) b+=texelFetch(uM,ivec2(i,s),0).r*texelFetch(uQ,ivec2(i,0),0).r;
      if(b>a || (b==a && s<r)) rank+=1; }
    if(rank==want){ best=r; bv=a; break; } }
  fragOut=float(best); }
"""

# LEVER 3: the leaf vector is NOT stored. Each candidate row is rebuilt from its token ids --
# tok[off .. off+len) -- as a sum of hash atoms, and scored against the probe in the same pass.
# Storage for the leaf level drops from K*D floats to the token id list.
FS_REGEN = """
#version 330 core
uniform sampler2D uQ,uIdx; uniform usampler2D uTok; uniform usampler2D uOff;
uniform int uD,uG,uN,uNI;
out float fragOut;
uint pcg(uint v){ uint s=v*747796405u+2891336453u;
  uint w=((s>>((s>>28u)+4u))^s)*277803737u; return (w>>22u)^w; }
void main(){
  int t=int(gl_FragCoord.x); int pi=t/uG;
  if(pi>=uNI){fragOut=-1e30;return;}
  int row=int(texelFetch(uIdx,ivec2(pi,0),0).r+0.5)*uG+(t-pi*uG);
  if(row>=uN){fragOut=-1e30;return;}
  uint off=texelFetch(uOff,ivec2(row,0),0).r;
  uint len=texelFetch(uOff,ivec2(row+1,0),0).r-off;
  float acc=0.0;
  for(int i=0;i<uD;++i){
    float v=0.0;
    for(uint k=0u;k<len;++k){
      uint h=texelFetch(uTok,ivec2(int(off+k),0),0).r;
      v += ((pcg(h ^ uint(i))>>31u)==1u)?1.0:-1.0; }
    acc += v*texelFetch(uQ,ivec2(i,0),0).r; }
  fragOut=acc; }          // NOT normalised: doc norms differ, so this is a DIFFERENT ranking
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
    def texU(self, u):
        u = np.ascontiguousarray(np.asarray(u, dtype="<u4"))
        return self.ctx.texture((len(u), 1), 1, u.tobytes(), dtype="u4")
    def target(self, w):
        o = self.ctx.texture((w, 1), 1, dtype="f4")
        return o, self.ctx.framebuffer(color_attachments=[o])
    def draw(self, fs, o, fbo, texs, ints):
        p, vao = self.prog(fs)
        fbo.use(); self.ctx.viewport = (0, 0, o.width, 1)
        for u, (n, t) in enumerate(texs.items()):
            t.use(u); p[n].value = u
        for k, v in ints.items():
            if k in p:
                p[k].value = int(v)
        vao.render(moderngl.TRIANGLES)
        return np.frombuffer(o.read(), dtype="f4")


def build(K, D, ntok=14, seed=0):
    rng = np.random.default_rng(seed)
    vocab = ["w%d" % i for i in range(4000)]
    toks = [[vocab[int(j)] for j in rng.choice(len(vocab), ntok, replace=False)] for _ in range(K)]
    V = np.stack([HA.encode_hash(t, D, normalise=False) * np.sqrt(D) for t in toks]) / np.sqrt(D)
    return toks, V


def run(g, K, D=256, BEAM=4, reps=25):
    toks, V = build(K, D)
    gg = max(2, int(round(K ** (1/3))))
    ch = np.stack([V[i:i+gg].sum(0) for i in range(0, K, gg)])
    su = np.stack([ch[i:i+gg].sum(0) for i in range(0, len(ch), gg)])
    rng = np.random.default_rng(1)
    Q = np.stack([V[int(rng.integers(K))] for _ in range(reps)])

    tV, tC, tS = g.tex(V), g.tex(ch), g.tex(su)
    identS = g.tex(np.arange(len(su), dtype="f4"))
    oS, fS = g.target(len(su)); oB, fB = g.target(BEAM)
    oBG, fBG = g.target(BEAM*gg); o1, f1 = g.target(1)

    flat_tok = np.concatenate([[HA.fnv1a(t) for t in d] for d in toks]).astype("<u4")
    offs = np.cumsum([0] + [len(d) for d in toks]).astype("<u4")
    tTok, tOff = g.texU(flat_tok), g.texU(offs)

    def baseline(q):
        tQ = g.tex(q)
        s1 = g.draw(FS_GATHER, oS, fS, {"uM": tS, "uQ": tQ, "uIdx": identS},
                    {"uD": D, "uG": 1, "uN": len(su), "uNI": len(su)})
        t1 = g.tex(s1)
        i1 = g.draw(FS_TOPB, oB, fB, {"uS": t1, "uIdx": identS},
                    {"uN": len(su), "uG": 1, "uUseIdx": 0})
        ti1 = g.tex(i1)
        s2 = g.draw(FS_GATHER, oBG, fBG, {"uM": tC, "uQ": tQ, "uIdx": ti1},
                    {"uD": D, "uG": gg, "uN": len(ch), "uNI": BEAM})
        t2 = g.tex(s2)
        i2 = g.draw(FS_TOPB, oB, fB, {"uS": t2, "uIdx": ti1},
                    {"uN": BEAM*gg, "uG": gg, "uUseIdx": 1})
        ti2 = g.tex(i2)
        s3 = g.draw(FS_GATHER, oBG, fBG, {"uM": tV, "uQ": tQ, "uIdx": ti2},
                    {"uD": D, "uG": gg, "uN": K, "uNI": BEAM})
        t3 = g.tex(s3)
        r = g.draw(FS_TOPB, o1, f1, {"uS": t3, "uIdx": ti2},
                   {"uN": BEAM*gg, "uG": gg, "uUseIdx": 1})
        for t in (tQ, t1, ti1, t2, ti2, t3):
            t.release()
        return int(round(float(r[0])))

    def fused(q):
        """LEVER 5: the top level's score+select collapse into ONE pass."""
        tQ = g.tex(q)
        i1 = g.draw(FS_SCORESEL, oB, fB, {"uM": tS, "uQ": tQ}, {"uD": D, "uN": len(su)})
        ti1 = g.tex(i1)
        s2 = g.draw(FS_GATHER, oBG, fBG, {"uM": tC, "uQ": tQ, "uIdx": ti1},
                    {"uD": D, "uG": gg, "uN": len(ch), "uNI": BEAM})
        t2 = g.tex(s2)
        i2 = g.draw(FS_TOPB, oB, fB, {"uS": t2, "uIdx": ti1},
                    {"uN": BEAM*gg, "uG": gg, "uUseIdx": 1})
        ti2 = g.tex(i2)
        s3 = g.draw(FS_GATHER, oBG, fBG, {"uM": tV, "uQ": tQ, "uIdx": ti2},
                    {"uD": D, "uG": gg, "uN": K, "uNI": BEAM})
        t3 = g.tex(s3)
        r = g.draw(FS_TOPB, o1, f1, {"uS": t3, "uIdx": ti2},
                   {"uN": BEAM*gg, "uG": gg, "uUseIdx": 1})
        for t in (tQ, ti1, t2, ti2, t3):
            t.release()
        return int(round(float(r[0])))

    def regen(q):
        """LEVER 3: the leaf matrix is not stored; leaves are rebuilt from token ids in-shader."""
        tQ = g.tex(q)
        s1 = g.draw(FS_GATHER, oS, fS, {"uM": tS, "uQ": tQ, "uIdx": identS},
                    {"uD": D, "uG": 1, "uN": len(su), "uNI": len(su)})
        t1 = g.tex(s1)
        i1 = g.draw(FS_TOPB, oB, fB, {"uS": t1, "uIdx": identS},
                    {"uN": len(su), "uG": 1, "uUseIdx": 0})
        ti1 = g.tex(i1)
        s2 = g.draw(FS_GATHER, oBG, fBG, {"uM": tC, "uQ": tQ, "uIdx": ti1},
                    {"uD": D, "uG": gg, "uN": len(ch), "uNI": BEAM})
        t2 = g.tex(s2)
        i2 = g.draw(FS_TOPB, oB, fB, {"uS": t2, "uIdx": ti1},
                    {"uN": BEAM*gg, "uG": gg, "uUseIdx": 1})
        ti2 = g.tex(i2)
        s3 = g.draw(FS_REGEN, oBG, fBG,
                    {"uQ": tQ, "uIdx": ti2, "uTok": tTok, "uOff": tOff},
                    {"uD": D, "uG": gg, "uN": K, "uNI": BEAM})
        t3 = g.tex(s3)
        r = g.draw(FS_TOPB, o1, f1, {"uS": t3, "uIdx": ti2},
                   {"uN": BEAM*gg, "uG": gg, "uUseIdx": 1})
        for t in (tQ, t1, ti1, t2, ti2, t3):
            t.release()
        return int(round(float(r[0])))

    def flatscan(q):
        identK = g.tex(np.arange(K, dtype="f4"))
        oK, fK = g.target(K)
        s = g.draw(FS_GATHER, oK, fK, {"uM": tV, "uQ": g.tex(q), "uIdx": identK},
                   {"uD": D, "uG": 1, "uN": K, "uNI": K})
        identK.release(); oK.release()
        return int(np.argmax(s))

    def timed(fn):
        fn(Q[0])
        t0 = time.perf_counter()
        for i in range(reps):
            fn(Q[i % len(Q)])
        return (time.perf_counter() - t0) / reps * 1e3

    base_ans = [baseline(q) for q in Q]
    fused_ans = [fused(q) for q in Q]
    regen_ans = [regen(q) for q in Q]
    bytes_full = (K + len(ch) + len(su)) * D * 4
    bytes_regen = (len(ch) + len(su)) * D * 4 + len(flat_tok) * 4 + len(offs) * 4
    return dict(K=K, g=gg, t_flat=timed(flatscan), t_base=timed(baseline),
                t_fused=timed(fused), t_regen=timed(regen),
                same_fused=sum(a == b for a, b in zip(base_ans, fused_ans)),
                same_regen=sum(a == b for a, b in zip(base_ans, regen_ans)),
                n=len(Q), bytes_full=bytes_full, bytes_regen=bytes_regen)


if __name__ == "__main__":
    g = GL()
    print("BACKEND:", g.ctx.info["GL_RENDERER"], "-- software; read the RATIOS, not the ms\n")
    print("  K     g   flat    6-pass  fused(L5)  regen(L3)   same_fused  same_regen   bytes_full  bytes_regen  shrink")
    for K in (100, 200, 400, 800):
        r = run(g, K)
        print("  %-5d %-3d %-7.3f %-7.3f %-10.3f %-11.3f %-11s %-12s %-11d %-12d %.2fx"
              % (r["K"], r["g"], r["t_flat"], r["t_base"], r["t_fused"], r["t_regen"],
                 "%d/%d" % (r["same_fused"], r["n"]), "%d/%d" % (r["same_regen"], r["n"]),
                 r["bytes_full"], r["bytes_regen"], r["bytes_full"] / r["bytes_regen"]))
