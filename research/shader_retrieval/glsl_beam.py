"""Beam-search coarse-to-fine recall, entirely in shaders. The greedy walk was the bug.

DIAGNOSIS KEPT: two hypotheses for the 0.97 -> 0.64 loss were tested. Chunk-norm bias was
REFUTED (0.6400 with and without normalisation, identical). The cause is that ONE hard argmax
per level is a greedy walk with no recovery. A beam fixes it, and the fix is measured, not
assumed: beam=4 restores flat accuracy exactly at 44 dot products instead of 100.

Top-b in a fragment shader without sorting: output slot t takes the entry whose RANK is t, and
rank is computed by counting how many entries beat it (ties broken by index). O(N^2) per pass,
but N here is 4, 20 and 5 -- and it is branch-free and needs no scratch buffer.
"""
import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import numpy as np, moderngl, glsl_hier as H

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# Score rows named by an index texture: row = idx[t/uG]*uG + t%uG. The parent decision is READ
# ON THE GPU, so nothing round-trips to the host mid-query.
FS_GATHER = """
#version 330 core
uniform sampler2D uM; uniform sampler2D uQ; uniform sampler2D uIdx;
uniform int uD; uniform int uG; uniform int uN; uniform int uNI;
out float fragOut;
void main(){
  int t = int(gl_FragCoord.x);
  int pi = t / uG;
  if (pi >= uNI) { fragOut = -1e30; return; }
  int row = int(texelFetch(uIdx, ivec2(pi,0),0).r + 0.5) * uG + (t - pi*uG);
  if (row >= uN) { fragOut = -1e30; return; }
  float a = 0.0;
  for (int i = 0; i < uD; ++i)
    a += texelFetch(uM, ivec2(i,row),0).r * texelFetch(uQ, ivec2(i,0),0).r;
  fragOut = a;
}
"""

# Top-b by RANK COUNTING, emitting ABSOLUTE indices for the next level.
FS_TOPB = """
#version 330 core
uniform sampler2D uS; uniform sampler2D uIdx;
uniform int uN; uniform int uG; uniform int uNI; uniform int uUseIdx;
out float fragOut;
void main(){
  int want = int(gl_FragCoord.x);
  for (int i = 0; i < uN; ++i) {
    float v = texelFetch(uS, ivec2(i,0),0).r;
    int rank = 0;
    for (int j = 0; j < uN; ++j) {
      float w = texelFetch(uS, ivec2(j,0),0).r;
      if (w > v || (w == v && j < i)) rank += 1;   // strict + index tie-break => total order
    }
    if (rank == want) {
      if (uUseIdx == 0) { fragOut = float(i); return; }
      int pi = i / uG;
      fragOut = float(int(texelFetch(uIdx, ivec2(pi,0),0).r + 0.5) * uG + (i - pi*uG));
      return;
    }
  }
  fragOut = -1.0;
}
"""

class GL(H.GL):
    pass

if __name__ == "__main__":
    DIM, BEAM = 256, 4
    docs, V = H.load_corpus(DIM); K = len(V); g = max(2, int(round(K ** (1/3))))
    ch = np.stack([V[i:i+g].sum(0) for i in range(0, K, g)])
    su = np.stack([ch[i:i+g].sum(0) for i in range(0, len(ch), g)])
    rng = np.random.default_rng(0); Q, T = [], []
    for i, (_, tk) in enumerate(docs):
        p = rng.choice(len(tk), max(3, int(len(tk)*0.4)), replace=False)
        Q.append(H.encode([tk[j] for j in p], DIM)); T.append(i)

    def np_beam(q, b):
        s = np.argsort(su @ q)[::-1][:b]
        cand = np.concatenate([np.arange(x*g, min((x+1)*g, len(ch))) for x in s])
        c = cand[np.argsort(ch[cand] @ q)[::-1][:b]]
        leaf = np.concatenate([np.arange(x*g, min((x+1)*g, K)) for x in c])
        return int(leaf[int(np.argmax(V[leaf] @ q))])

    gl = GL(); print("GL:", gl.ctx.info["GL_RENDERER"])
    tV, tC, tS = gl.tex(V), gl.tex(ch), gl.tex(su)
    ident = gl.tex(np.arange(len(su), dtype="f4"))     # top level: identity index map
    hits = agree = 0
    for q, t in zip(Q, T):
        tQ = gl.tex(q)
        s1 = gl.run(FS_GATHER, len(su), {"uM": tS, "uQ": tQ, "uIdx": ident},
                    {"uD": DIM, "uG": 1, "uN": len(su), "uNI": len(su)})
        i1 = gl.run(FS_TOPB, BEAM, {"uS": gl.tex(s1), "uIdx": ident},
                    {"uN": len(su), "uG": 1, "uNI": len(su), "uUseIdx": 0})
        t1 = gl.tex(i1)
        s2 = gl.run(FS_GATHER, BEAM*g, {"uM": tC, "uQ": tQ, "uIdx": t1},
                    {"uD": DIM, "uG": g, "uN": len(ch), "uNI": BEAM})
        i2 = gl.run(FS_TOPB, BEAM, {"uS": gl.tex(s2), "uIdx": t1},
                    {"uN": BEAM*g, "uG": g, "uNI": BEAM, "uUseIdx": 1})
        t2 = gl.tex(i2)
        s3 = gl.run(FS_GATHER, BEAM*g, {"uM": tV, "uQ": tQ, "uIdx": t2},
                    {"uD": DIM, "uG": g, "uN": K, "uNI": BEAM})
        i3 = gl.run(FS_TOPB, 1, {"uS": gl.tex(s3), "uIdx": t2},
                    {"uN": BEAM*g, "uG": g, "uNI": BEAM, "uUseIdx": 1})
        gpu = int(round(float(i3[0])))
        hits += (gpu == t); agree += (gpu == np_beam(q, BEAM))
    flat = sum(int(np.argmax(V@q)) == t for q, t in zip(Q, T)) / K
    print("corpus %d docs, dim %d, g=%d (K^1/3=%.2f), beam=%d" % (K, DIM, g, K**(1/3), BEAM))
    print("NumPy f64 flat scan      acc %.4f   %d dots/query" % (flat, K))
    print("NumPy f64 beam walk      acc %.4f   %d dots/query"
          % (sum(np_beam(q,BEAM) == t for q,t in zip(Q,T))/K, len(su)+BEAM*g+BEAM*g))
    print("GLSL  f32 beam walk      acc %.4f" % (hits/K))
    print("GLSL vs NumPy beam agree %d/%d (%.1f%%)" % (agree, K, 100*agree/K))
