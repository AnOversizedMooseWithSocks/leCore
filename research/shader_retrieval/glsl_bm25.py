"""Put the arm that actually WINS on the GPU: BM25 scoring and the containment count, in GLSL.

WHY THIS AND NOT MORE COSINE WORK. The retrieval measurements settled it: on realistic overlapping
source text BM25 reaches 0.875 top-1 against a 0.858 Bayes ceiling, while the hash-dense arm this
GLSL path was built around manages 0.425. The shaders have been carrying the WEAK arm. Worse, the
policy that turned 0.875 into 0.965-when-answering needs the CONTAINMENT COUNT, which the GPU
path could not compute at all.

BOTH ARE THE SAME SHAPE AND THE SAME PASS. For each document fragment, walk that document's token
ids once and compare each against the (short) query. Occurrence counts give BM25's tf; how many
DISTINCT query terms were seen gives containment. One loop, two outputs, no scatter -- which
matters because a fragment shader cannot scatter, and the obvious posting-list formulation would
need exactly that.

IT ALSO REUSES A LAYOUT THAT ALREADY EXISTS: the concatenated token-id + offset textures built
for regenerate-on-demand. The compressed index and the lexical scorer want the same bytes.

DIFFERENTIAL TARGET, stated precisely: the shader is pinned against the SAME Okapi formula in
NumPy, bit-for-bit on the ranking. It is NOT pinned against mind.bm25_rank -- that harness gap
(score correlation 0.846) is already on record and is a formula/tokenisation difference, not
something this shader introduces or fixes.
"""
import os, time
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# One fragment per document. Walks the document once; counts occurrences of each query term in a
# register array (queries are short, so this fits); emits BM25 score and containment coverage.
FS_BM25 = """
#version 330 core
uniform usampler2D uTok;    // concatenated term ids for every document
uniform usampler2D uOff;    // uOff[d]..uOff[d+1] is document d's span
uniform usampler2D uQ;      // query term ids
uniform sampler2D  uIdf;    // idf per query term
uniform int uNQ, uW;
uniform float uK1, uB, uAvgdl;
out vec2 fragOut;           // .x = BM25 score, .y = distinct query terms present
ivec2 at(int i){ return ivec2(i % uW, i / uW); }
void main(){
  int d = int(gl_FragCoord.x);
  uint lo = texelFetch(uOff, at(d), 0).r;
  uint hi = texelFetch(uOff, at(d+1), 0).r;
  float dl = float(hi - lo);
  float tf[16];
  for (int j = 0; j < 16; ++j) tf[j] = 0.0;
  for (uint p = lo; p < hi; ++p) {
    uint t = texelFetch(uTok, at(int(p)), 0).r;
    for (int j = 0; j < uNQ; ++j)
      if (t == texelFetch(uQ, ivec2(j,0), 0).r) tf[j] += 1.0;
  }
  float s = 0.0; float cov = 0.0;
  for (int j = 0; j < uNQ; ++j) {
    if (tf[j] > 0.0) {
      cov += 1.0;
      float denom = tf[j] + uK1 * (1.0 - uB + uB * dl / uAvgdl);
      s += texelFetch(uIdf, ivec2(j,0), 0).r * tf[j] * (uK1 + 1.0) / denom;
    }
  }
  fragOut = vec2(s, cov);
}
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
    def texu(self, a, w):
        a = np.asarray(a, dtype="<u4")
        h = (len(a) + w - 1) // w
        buf = np.zeros(h * w, dtype="<u4"); buf[:len(a)] = a
        return self.ctx.texture((w, h), 1, buf.tobytes(), dtype="u4")
    def texf(self, a):
        a = np.ascontiguousarray(np.atleast_2d(np.asarray(a, dtype="f4")))
        return self.ctx.texture((a.shape[1], 1), 1, a.tobytes(), dtype="f4")
    def draw(self, fs, w, texs, ints, floats):
        p, vao = self.prog(fs)
        o = self.ctx.texture((w, 1), 2, dtype="f4")
        fbo = self.ctx.framebuffer(color_attachments=[o]); fbo.use()
        self.ctx.viewport = (0, 0, w, 1)
        for u, (n, t) in enumerate(texs.items()):
            t.use(u); p[n].value = u
        for k, v in ints.items():
            if k in p: p[k].value = int(v)
        for k, v in floats.items():
            if k in p: p[k].value = float(v)
        vao.render(moderngl.TRIANGLES)
        r = np.frombuffer(o.read(), dtype="f4").reshape(w, 2).copy()
        fbo.release(); o.release()
        return r


def build_index(docs, W=4096):
    vocab = {}
    tok, off = [], [0]
    for d in docs:
        for t in d:
            tok.append(vocab.setdefault(t, len(vocab)))
        off.append(len(tok))
    return vocab, np.array(tok, dtype="<u4"), np.array(off, dtype="<u4"), W


def cpu_bm25(docs, off, tok, qids, idf, k1=1.5, b=0.75):
    """The SAME Okapi formula in NumPy -- the differential reference."""
    dl = np.diff(off).astype(float)
    avgdl = dl.mean()
    s = np.zeros(len(docs)); cov = np.zeros(len(docs))
    for d in range(len(docs)):
        span = tok[off[d]:off[d + 1]]
        for j, q in enumerate(qids):
            tf = float((span == q).sum())
            if tf > 0:
                cov[d] += 1
                s[d] += idf[j] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[d] / avgdl))
    return s, cov


if __name__ == "__main__":
    import hard_corpus as HC
    gl = GL()
    print("BACKEND:", gl.ctx.info["GL_RENDERER"], "\n")
    dn = HC.load_passages(target=800)
    docs = [t for _, t in dn]
    vocab, tok, off, W = build_index(docs)
    N = len(docs)
    dl = np.diff(off).astype(float)
    avgdl = float(dl.mean())
    df = np.zeros(len(vocab))
    for d in docs:
        for t in set(d):
            df[vocab[t]] += 1

    tTok, tOff = gl.texu(tok, W), gl.texu(off, W)
    rng = np.random.default_rng(0)
    agree1 = 0; agree_cov = 0; maxerr = 0.0; n = 0
    t_gpu = t_cpu = 0.0
    for trial in range(40):
        i = int(rng.integers(N))
        u = sorted(set(docs[i]))
        q = [u[j] for j in rng.choice(len(u), min(8, len(u)), replace=False)]
        qids = np.array([vocab[t] for t in q], dtype="<u4")
        idf = np.array([np.log(1.0 + (N - df[v] + 0.5) / (df[v] + 0.5)) for v in qids], dtype="f4")
        t0 = time.perf_counter()
        r = gl.draw(FS_BM25, N, {"uTok": tTok, "uOff": tOff, "uQ": gl.texu(qids, len(qids)),
                                 "uIdf": gl.texf(idf)},
                    {"uNQ": len(qids), "uW": W},
                    {"uK1": 1.5, "uB": 0.75, "uAvgdl": avgdl})
        t_gpu += time.perf_counter() - t0
        t0 = time.perf_counter()
        s_cpu, c_cpu = cpu_bm25(docs, off, tok, qids, idf)
        t_cpu += time.perf_counter() - t0
        agree1 += int(int(np.argmax(r[:, 0])) == int(np.argmax(s_cpu)))
        agree_cov += int(np.array_equal(r[:, 1].astype(int), c_cpu.astype(int)))
        rel = np.max(np.abs(r[:, 0] - s_cpu)) / (np.max(np.abs(s_cpu)) + 1e-30)
        maxerr = max(maxerr, float(rel)); n += 1
    print("GLSL BM25 vs the SAME Okapi formula in NumPy, hard corpus, %d passages, %d queries" % (N, n))
    print("   top-1 ranking identical            %d/%d" % (agree1, n))
    print("   containment coverage EXACT (ints)  %d/%d" % (agree_cov, n))
    print("   max relative score error           %.3e  (f32 vs f64)" % maxerr)
    print("   per query: GPU %.2f ms   NumPy reference %.2f ms   (software raster, ratio only)"
          % (1e3 * t_gpu / n, 1e3 * t_cpu / n))
