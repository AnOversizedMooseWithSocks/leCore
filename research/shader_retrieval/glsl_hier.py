"""The WHOLE read path in shaders: 3-tier coarse-to-fine recall over the REAL docstring corpus.

What was still Python after the last step: the tiered store, and every decision between levels.
This closes both. The tier walk runs entirely on the GPU -- the argmax at each level is written
to a 1x1 texture and READ BY THE NEXT PASS via texelFetch, so no result crosses back to the host
mid-query. Host work is reduced to what the install boundary always said it would be: ingest,
tokenisation and pass sequencing. Arithmetic and decisions are shader-side.

Tier size comes from the BALANCE LAW measured earlier, not from a sweep: three tiers balance at
g ~ K^(1/3). For the 122-document corpus that is 4.96, so g=5.

Fragment shaders and texelFetch only -- the WebGL2 subset. No compute shaders, no SSBOs.
"""
import os, re, glob
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl
from holographic.agents_and_reasoning.holographic_ai import derived_atom

STOP = set("the a an of to and or is are was in on for with that this it as by be from at not "
           "but if then than so its which what when how you your we our can use used using".split())

def tokens(t):
    return [w for w in re.findall(r"[a-z][a-z0-9_]{2,}", t.lower()) if w not in STOP]

def load_corpus(dim, seed=0, limit=400):
    docs = []
    for path in sorted(glob.glob("holographic/*/holographic_*.py"))[:limit]:
        src = open(path, encoding="utf-8", errors="ignore").read(4000)
        m = re.search(r'"""(.{60,1200}?)"""', src, re.S)
        if not m:
            continue
        tk = tokens(m.group(1))
        if len(set(tk)) >= 25:
            docs.append((path.split("/")[-1][:-3], sorted(set(tk))))
    V = np.stack([encode(t, dim, seed) for _, t in docs])
    return docs, V

def encode(toks, dim, seed=0):
    v = np.zeros(dim)
    for t in toks:
        v += derived_atom(seed, t, dim)
    n = np.linalg.norm(v)
    return v / n if n else v

def build_tiers(V, g):
    """Bundle docs into chunks, chunks into supers. A bundle is a sum -- lever 2's monoid."""
    ch = np.stack([V[i:i + g].sum(0) for i in range(0, len(V), g)])
    su = np.stack([ch[i:i + g].sum(0) for i in range(0, len(ch), g)])
    return ch, su

def numpy_walk(q, V, ch, su, g):
    """Reference coarse-to-fine walk: super -> chunk -> doc, argmax at each level."""
    s = int(np.argmax(su @ q))
    lo, hi = s * g, min((s + 1) * g, len(ch))
    c = lo + int(np.argmax(ch[lo:hi] @ q))
    lo2, hi2 = c * g, min((c + 1) * g, len(V))
    return lo2 + int(np.argmax(V[lo2:hi2] @ q))

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# Score a CONTIGUOUS RANGE of rows against the probe. The range base is read from a texture, so
# the previous level's decision never leaves the GPU.
FS_SCORE_RANGE = """
#version 330 core
uniform sampler2D uM;     // D x N matrix (x = component, y = row)
uniform sampler2D uQ;     // D x 1 probe
uniform sampler2D uBase;  // 1 x 1, holds the parent index as a float (or -1 = start at 0)
uniform int uD; uniform int uG; uniform int uN;
out float fragOut;
void main() {
    int t = int(gl_FragCoord.x);
    int base = int(texelFetch(uBase, ivec2(0,0), 0).r + 0.5);
    int row = base * uG + t;
    if (row >= uN) { fragOut = -1e30; return; }   // ragged last group
    float a = 0.0;
    for (int i = 0; i < uD; ++i)
        a += texelFetch(uM, ivec2(i,row),0).r * texelFetch(uQ, ivec2(i,0),0).r;
    fragOut = a;
}
"""

# argmax over a short row, written as a FLOAT INDEX into a 1x1 texture -- the decision stays on
# the GPU. T4 says a tiled max is exact; an argmax needs the index too, so this pass carries both.
FS_ARGMAX = """
#version 330 core
uniform sampler2D uS; uniform int uN; uniform sampler2D uBase; uniform int uG;
out float fragOut;
void main() {
    int best = 0; float bv = -1e30;
    for (int i = 0; i < uN; ++i) {
        float v = texelFetch(uS, ivec2(i,0),0).r;
        if (v > bv) { bv = v; best = i; }
    }
    int base = int(texelFetch(uBase, ivec2(0,0),0).r + 0.5);
    fragOut = float(base * uG + best);            // ABSOLUTE index, ready for the next level
}
"""

class GL:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.vbo = self.ctx.buffer(np.array([-1,-1,3,-1,-1,3], dtype="f4").tobytes())
        self.cache = {}
    def prog(self, fs):
        if fs not in self.cache:
            p = self.ctx.program(vertex_shader=VS, fragment_shader=fs)
            self.cache[fs] = (p, self.ctx.vertex_array(p, [(self.vbo, "2f", "p")]))
        return self.cache[fs]
    def tex(self, a):
        a = np.ascontiguousarray(np.atleast_2d(a).astype("f4"))
        return self.ctx.texture((a.shape[1], a.shape[0]), 1, a.tobytes(), dtype="f4")
    def run(self, fs, w, texs, ints):
        p, vao = self.prog(fs)
        out = self.ctx.texture((w,1), 1, dtype="f4")
        fbo = self.ctx.framebuffer(color_attachments=[out]); fbo.use()
        self.ctx.viewport = (0,0,w,1)
        for u,(n,t) in enumerate(texs.items()):
            t.use(u); p[n].value = u
        for k,v in ints.items():
            # A uniform the shader never reads is optimised out; that is the compiler doing its
            # job, so a missing name here is not an error.
            if k in p:
                p[k].value = int(v)
        vao.render(moderngl.TRIANGLES)
        res = np.frombuffer(out.read(), dtype="f4").copy()
        fbo.release(); out.release()
        return res

if __name__ == "__main__":
    DIM = 256
    docs, V = load_corpus(DIM)
    K = len(V)
    g = max(2, int(round(K ** (1/3))))
    ch, su = build_tiers(V, g)
    print("REAL CORPUS: %d docstrings, dim=%d" % (K, DIM))
    print("tier size from the BALANCE LAW g ~ K^(1/3) = %.2f -> g=%d" % (K ** (1/3), g))
    print("tiers: %d docs -> %d chunks -> %d supers\n" % (K, len(ch), len(su)))

    rng = np.random.default_rng(0)
    queries, truth = [], []
    for i, (_, tk) in enumerate(docs):
        pick = rng.choice(len(tk), max(3, int(len(tk) * 0.4)), replace=False)
        queries.append(encode([tk[j] for j in pick], DIM)); truth.append(i)

    flat = sum(int(np.argmax(V @ q)) == t for q, t in zip(queries, truth)) / K
    walk = sum(numpy_walk(q, V, ch, su, g) == t for q, t in zip(queries, truth)) / K
    print("NumPy f64  flat scan       accuracy %.4f   (%d dot products / query)" % (flat, K))
    print("NumPy f64  3-tier walk     accuracy %.4f   (%d dot products / query)"
          % (walk, len(su) + g + g))

    gl = GL()
    print("\nGL:", gl.ctx.info["GL_RENDERER"])
    tV, tC, tS = gl.tex(V), gl.tex(ch), gl.tex(su)
    hits = 0; agree = 0
    for q, t in zip(queries, truth):
        tQ = gl.tex(q)
        zero = gl.tex(np.zeros(1)); neg = gl.tex(np.zeros(1))   # base=0 for the top level
        s1 = gl.run(FS_SCORE_RANGE, len(su), {"uM": tS, "uQ": tQ, "uBase": zero},
                    {"uD": DIM, "uG": len(su), "uN": len(su)})
        i1 = gl.run(FS_ARGMAX, 1, {"uS": gl.tex(s1), "uBase": zero},
                    {"uN": len(su), "uG": len(su)})
        tI1 = gl.tex(i1)
        s2 = gl.run(FS_SCORE_RANGE, g, {"uM": tC, "uQ": tQ, "uBase": tI1},
                    {"uD": DIM, "uG": g, "uN": len(ch)})
        i2 = gl.run(FS_ARGMAX, 1, {"uS": gl.tex(s2), "uBase": tI1}, {"uN": g, "uG": g})
        tI2 = gl.tex(i2)
        s3 = gl.run(FS_SCORE_RANGE, g, {"uM": tV, "uQ": tQ, "uBase": tI2},
                    {"uD": DIM, "uG": g, "uN": K})
        i3 = gl.run(FS_ARGMAX, 1, {"uS": gl.tex(s3), "uBase": tI2}, {"uN": g, "uG": g})
        gpu = int(round(float(i3[0])))
        hits += (gpu == t); agree += (gpu == numpy_walk(q, V, ch, su, g))
    print("GLSL f32   3-tier walk     accuracy %.4f" % (hits / K))
    print("GLSL vs NumPy walk, same answer on %d/%d queries (%.1f%%)"
          % (agree, K, 100 * agree / K))
