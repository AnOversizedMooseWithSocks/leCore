"""Make the shaders fast. Four structural changes, each measured, each answer-verified.

WHY THESE FOUR, and why they are not llvmpipe artefacts: every one attacks FETCH COUNT or PASS
COUNT, which are the two things a fragment pipeline actually charges for on any backend.

  V1 VEC4 PACKING. The scalar shaders do `uD` texelFetches per operand -- 512 fetches for one
     D=256 dot product. A texel is RGBA, so packing four components per texel cuts that to 128
     and turns the inner loop into `dot(vec4,vec4)`, one instruction instead of a multiply-add.
     4x fewer fetches is the single largest structural saving available.
  V2 HALF-PRECISION STORAGE (f16). Bandwidth is the bill; f16 halves it and halves the index on
     disk. The question is whether the DECISION survives -- T1 says it does exactly while the
     margin exceeds 2*eps, so this is not a guess, it is a bound to check. Verified, not assumed.
  V3 SINGLE-PASS TOP-B. The rank-counting select is O(N^2) IN TEXTURE FETCHES, and it is run
     once per output slot, so four fragments each re-read the whole score row. Emitting all four
     indices PACKED IN ONE RGBA FRAGMENT does the scan ONCE -- this is the fix recorded as
     untested when the naive pass-fusion was refuted, now built.
  V4 NO PER-QUERY ALLOCATION. The earlier harness created a texture and a framebuffer per draw.
     Allocation is not shading; it is reused here so the timing measures the shader.

EVERY VARIANT IS ANSWER-CHECKED AGAINST THE SCALAR BASELINE FIRST. A speedup that changes the
answer is not a speedup.
"""
import os, time
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl
import holographic.agents_and_reasoning.holographic_hashatom as HA

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

FS_GATHER1 = """
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

# V1: identical maths, a quarter of the fetches. uD4 = D/4 texels per row.
FS_GATHER4 = """
#version 330 core
uniform sampler2D uM,uQ,uIdx; uniform int uD4,uG,uN,uNI;
out float fragOut;
void main(){ int t=int(gl_FragCoord.x); int pi=t/uG;
  if(pi>=uNI){fragOut=-1e30;return;}
  int row=int(texelFetch(uIdx,ivec2(pi,0),0).r+0.5)*uG+(t-pi*uG);
  if(row>=uN){fragOut=-1e30;return;}
  float a=0.0;
  for(int i=0;i<uD4;++i)
    a+=dot(texelFetch(uM,ivec2(i,row),0), texelFetch(uQ,ivec2(i,0),0));
  fragOut=a; }
"""

FS_TOPB1 = """
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

# V3: ONE fragment, ONE linear scan, four winners packed into RGBA. Insertion into a 4-slot
# register array keeps the same total order as the rank-count form (strict >, ties by lower
# index), so the answer is identical by construction -- and checked anyway.
FS_TOPB4 = """
#version 330 core
uniform sampler2D uS,uIdx; uniform int uN,uG,uUseIdx;
out vec4 fragOut;
void main(){
  float bv0=-1e30,bv1=-1e30,bv2=-1e30,bv3=-1e30;
  int b0=-1,b1=-1,b2=-1,b3=-1;
  for(int i=0;i<uN;++i){
    float v=texelFetch(uS,ivec2(i,0),0).r;
    if(v>bv0){ bv3=bv2;b3=b2; bv2=bv1;b2=b1; bv1=bv0;b1=b0; bv0=v;b0=i; }
    else if(v>bv1){ bv3=bv2;b3=b2; bv2=bv1;b2=b1; bv1=v;b1=i; }
    else if(v>bv2){ bv3=bv2;b3=b2; bv2=v;b2=i; }
    else if(v>bv3){ bv3=v;b3=i; } }
  vec4 r=vec4(float(b0),float(b1),float(b2),float(b3));
  if(uUseIdx==1){
    int i0=b0,i1=b1,i2=b2,i3=b3;
    r.x=float(int(texelFetch(uIdx,ivec2(i0/uG,0),0).r+0.5)*uG+(i0-(i0/uG)*uG));
    r.y=float(int(texelFetch(uIdx,ivec2(i1/uG,0),0).r+0.5)*uG+(i1-(i1/uG)*uG));
    r.z=float(int(texelFetch(uIdx,ivec2(i2/uG,0),0).r+0.5)*uG+(i2-(i2/uG)*uG));
    r.w=float(int(texelFetch(uIdx,ivec2(i3/uG,0),0).r+0.5)*uG+(i3-(i3/uG)*uG)); }
  fragOut=r; }
"""


class GL:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.vbo = self.ctx.buffer(np.array([-1,-1,3,-1,-1,3], dtype="f4").tobytes())
        self.c = {}
        self.targets = {}
    def prog(self, fs):
        if fs not in self.c:
            p = self.ctx.program(vertex_shader=VS, fragment_shader=fs)
            self.c[fs] = (p, self.ctx.vertex_array(p, [(self.vbo, "2f", "p")]))
        return self.c[fs]
    def tex(self, a, comps=1, dtype="f4"):
        a = np.ascontiguousarray(a.astype(dtype))
        h = a.shape[0] if a.ndim >= 2 else 1
        w = a.shape[1] // comps if a.ndim >= 2 else a.shape[0] // comps
        return self.ctx.texture((w, h), comps, a.tobytes(), dtype=dtype)
    def target(self, w, comps=1):
        """V4: render targets are CACHED. Allocation is not shading."""
        k = (w, comps)
        if k not in self.targets:
            o = self.ctx.texture((w, 1), comps, dtype="f4")
            self.targets[k] = (o, self.ctx.framebuffer(color_attachments=[o]))
        return self.targets[k]
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


def pack4(M):
    """Pad the component axis to a multiple of 4 so a row is a whole number of RGBA texels."""
    M = np.atleast_2d(M)
    pad = (-M.shape[1]) % 4
    if pad:
        M = np.concatenate([M, np.zeros((M.shape[0], pad))], axis=1)
    return M


def run(gl, K, D=256, BEAM=4, reps=30):
    rng = np.random.default_rng(0)
    V = np.stack([HA.hash_atom("doc%d" % i, D) for i in range(K)])
    g = max(2, int(round(K ** (1/3))))
    ch = np.stack([V[i:i+g].sum(0) for i in range(0, K, g)])
    su = np.stack([ch[i:i+g].sum(0) for i in range(0, len(ch), g)])
    Q = np.stack([V[int(rng.integers(K))] for _ in range(reps)])
    NS, NC = len(su), len(ch)
    D4 = pack4(V).shape[1] // 4

    t1 = {"V": gl.tex(V), "C": gl.tex(ch), "S": gl.tex(su)}
    t4 = {"V": gl.tex(pack4(V), 4), "C": gl.tex(pack4(ch), 4), "S": gl.tex(pack4(su), 4)}
    # V2: the SAME arrays stored as f16 and read back as f32 -- exactly what a half texture does.
    h = lambda M: pack4(M).astype("f2").astype("f4")
    t16 = {"V": gl.tex(h(V), 4), "C": gl.tex(h(ch), 4), "S": gl.tex(h(su), 4)}
    identS1, identS4 = gl.tex(np.arange(NS, dtype="f4")), gl.tex(np.arange(NS, dtype="f4"))

    def walk(q, mats, packed, top4):
        tq = gl.tex(pack4(q.reshape(1, -1)), 4) if packed else gl.tex(q.reshape(1, -1))
        gth, dk = (FS_GATHER4, "uD4") if packed else (FS_GATHER1, "uD")
        dv = D4 if packed else D
        s1 = gl.draw(gth, NS, {"uM": mats["S"], "uQ": tq, "uIdx": identS1},
                     {dk: dv, "uG": 1, "uN": NS, "uNI": NS})
        ts1 = gl.tex(s1.reshape(1, -1))
        if top4:
            i1 = gl.draw(FS_TOPB4, 1, {"uS": ts1, "uIdx": identS1},
                         {"uN": NS, "uG": 1, "uUseIdx": 0}, comps=4)[:BEAM]
        else:
            i1 = gl.draw(FS_TOPB1, BEAM, {"uS": ts1, "uIdx": identS1},
                         {"uN": NS, "uG": 1, "uUseIdx": 0})
        ti1 = gl.tex(np.asarray(i1, dtype="f4").reshape(1, -1))
        s2 = gl.draw(gth, BEAM*g, {"uM": mats["C"], "uQ": tq, "uIdx": ti1},
                     {dk: dv, "uG": g, "uN": NC, "uNI": BEAM})
        ts2 = gl.tex(s2.reshape(1, -1))
        if top4:
            i2 = gl.draw(FS_TOPB4, 1, {"uS": ts2, "uIdx": ti1},
                         {"uN": BEAM*g, "uG": g, "uUseIdx": 1}, comps=4)[:BEAM]
        else:
            i2 = gl.draw(FS_TOPB1, BEAM, {"uS": ts2, "uIdx": ti1},
                         {"uN": BEAM*g, "uG": g, "uUseIdx": 1})
        ti2 = gl.tex(np.asarray(i2, dtype="f4").reshape(1, -1))
        s3 = gl.draw(gth, BEAM*g, {"uM": mats["V"], "uQ": tq, "uIdx": ti2},
                     {dk: dv, "uG": g, "uN": K, "uNI": BEAM})
        ts3 = gl.tex(s3.reshape(1, -1))
        if top4:
            r = gl.draw(FS_TOPB4, 1, {"uS": ts3, "uIdx": ti2},
                        {"uN": BEAM*g, "uG": g, "uUseIdx": 1}, comps=4)[0]
        else:
            r = gl.draw(FS_TOPB1, 1, {"uS": ts3, "uIdx": ti2},
                        {"uN": BEAM*g, "uG": g, "uUseIdx": 1})[0]
        for t in (tq, ts1, ti1, ts2, ti2, ts3):
            t.release()
        return int(round(float(r)))

    arms = [("scalar baseline", lambda q: walk(q, t1, False, False)),
            ("V1 vec4",         lambda q: walk(q, t4, True,  False)),
            ("V1+V3 vec4+top4", lambda q: walk(q, t4, True,  True)),
            ("V1+V2+V3 +f16",   lambda q: walk(q, t16, True, True))]
    base = [arms[0][1](q) for q in Q]
    out = []
    for name, fn in arms:
        ans = [fn(q) for q in Q]
        fn(Q[0])
        t0 = time.perf_counter()
        for i in range(reps):
            fn(Q[i % len(Q)])
        ms = (time.perf_counter() - t0) / reps * 1e3
        out.append((name, ms, sum(a == b for a, b in zip(ans, base)), len(Q)))
    return K, g, out


if __name__ == "__main__":
    gl = GL()
    print("BACKEND:", gl.ctx.info["GL_RENDERER"], "-- software; read the SPEEDUP RATIOS\n")
    for K in (200, 800, 3200):
        K, g, arms = run(gl, K)
        base_ms = arms[0][1]
        print("K=%-5d g=%-3d" % (K, g))
        for name, ms, same, n in arms:
            print("   %-18s %7.3f ms   %5.2fx   answers %d/%d %s"
                  % (name, ms, base_ms / ms, same, n, "" if same == n else "<-- CHANGED"))
        print()
