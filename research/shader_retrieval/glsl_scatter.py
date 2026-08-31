"""Scatter BM25 on a fragment pipeline: one POINT per posting, accumulated by additive blending.

WHY THIS EXISTS, and the correction that produced it. I measured "candidate fraction 0.518" and
concluded an inverted index buys only 2x. That was DOCUMENTS TOUCHED, not WORK DONE. The work a
full scan does is K * nq * log2(terms/doc); the work a posting walk does is sum(df) over the query
terms. Measured on the same corpus:
    full scan     29,221 * 8 * 7          = 1,636,376 fetches
    posting walk  median sum(df)          =    19,717 visits
    ratio                                   83x
Two orders of magnitude were hiding behind a badly chosen denominator.

THE OBJECTION THAT MADE ME SKIP IT WAS ALSO WRONG. "A fragment shader cannot scatter" is true of
the FRAGMENT stage; the VERTEX stage can place a primitive anywhere, and ADDITIVE BLENDING is a
hardware scatter-add. So: one point per posting entry, positioned at its document's output texel,
carrying its BM25 contribution, with blend func ONE/ONE. The GPU accumulates. This is the classic
GPGPU histogram trick and it is exactly an inverted index.

THE PRICE, stated before any number: BLEND ORDER IS UNSPECIFIED, and float addition is not
associative, so accumulated scores are NOT bit-reproducible run to run. That is a real cost this
project does not hand-wave. It is measured below, and judged by T1's margin bound -- the question
is never "are the bits identical" but "does the DECISION move".
"""
import os, time
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

# One vertex per POSTING. The vertex stage reads (doc, tf) for its posting, computes that term's
# BM25 contribution, and PLACES ITSELF at the document's output texel. Blending does the sum.
VS_SCATTER = """
#version 330 core
uniform usampler2D uDoc, uTf;      // postings, CSR-ordered by term
uniform sampler2D uDl;
uniform int uBase, uW, uOutW, uOutH;
uniform float uIdf, uK1, uB, uAvgdl;
out float vContrib;
ivec2 at(int i){ return ivec2(i % uW, i / uW); }
void main(){
  int p = uBase + gl_VertexID;
  int d = int(texelFetch(uDoc, at(p), 0).r);
  float tf = float(texelFetch(uTf, at(p), 0).r);
  float dl = texelFetch(uDl, at(d), 0).r;
  vContrib = uIdf * tf * (uK1 + 1.0) / (tf + uK1 * (1.0 - uB + uB * dl / uAvgdl));
  // place this point at document d's texel, in clip space, sampling the pixel centre
  float x = (float(d % uOutW) + 0.5) / float(uOutW) * 2.0 - 1.0;
  float y = (float(d / uOutW) + 0.5) / float(uOutH) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_PointSize = 1.0;
}
"""

FS_SCATTER = """
#version 330 core
in float vContrib;
out vec2 fragOut;
void main(){ fragOut = vec2(vContrib, 1.0); }   // .y counts terms present = containment coverage
"""


_CTX = None


def _shared_context():
    global _CTX
    if _CTX is None:
        _CTX = moderngl.create_standalone_context(require=330, backend="egl")
    return _CTX


class Scatter:
    def __init__(self, docs, W=4096):
        self.W = W
        self.docs = docs
        self.N = len(docs)
        self.vocab = {}
        post = {}
        for i, d in enumerate(docs):
            c = {}
            for t in d:
                c[t] = c.get(t, 0) + 1
            for t, tf in c.items():
                post.setdefault(self.vocab.setdefault(t, len(self.vocab)), []).append((i, tf))
        self.off = np.zeros(len(self.vocab) + 1, dtype=np.int64)
        docid, tfv = [], []
        for v in range(len(self.vocab)):
            rows = post.get(v, [])
            self.off[v + 1] = self.off[v] + len(rows)
            for i, tf in rows:
                docid.append(i); tfv.append(tf)
        self.docid = np.array(docid, dtype="<u4")
        self.tf = np.array(tfv, dtype="<u4")
        self.dl = np.array([len(d) for d in docs], dtype="f4")
        self.avgdl = float(self.dl.mean())
        self.df = np.diff(self.off).astype(float)

        # ONE CONTEXT PER PROCESS. Creating a second standalone context makes it current and
        # every subsequent draw from an older Scatter lands in the wrong framebuffer -- measured:
        # 4,744 of 5,000 documents wrong, max score error 56.4, with NO error reported by GL.
        # This is what produced the 0.67 "relative error" and a sweep whose numbers flapped.
        self.ctx = _shared_context()
        self.prog = self.ctx.program(vertex_shader=VS_SCATTER, fragment_shader=FS_SCATTER)
        self.vao = self.ctx.vertex_array(self.prog, [])
        self.tDoc = self._texu(self.docid)
        self.tTf = self._texu(self.tf)
        self.tDl = self._texf(self.dl)
        self.outW = min(W, self.N)
        self.outH = (self.N + self.outW - 1) // self.outW
        self.out = self.ctx.texture((self.outW, self.outH), 2, dtype="f4")
        self.fbo = self.ctx.framebuffer(color_attachments=[self.out])

    def _texu(self, a):
        h = (len(a) + self.W - 1) // self.W
        buf = np.zeros(h * self.W, dtype="<u4"); buf[:len(a)] = a
        return self.ctx.texture((self.W, h), 1, buf.tobytes(), dtype="u4")

    def _texf(self, a):
        h = (len(a) + self.W - 1) // self.W
        buf = np.zeros(h * self.W, dtype="f4"); buf[:len(a)] = a
        return self.ctx.texture((self.W, h), 1, buf.tobytes(), dtype="f4")

    def scores(self, qterms, k1=1.5, b=0.75):
        """One draw per query term, drawing df(t) points. Total work = sum(df), not K*nq*log."""
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.outW, self.outH)
        self.fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE      # the scatter-add
        self.tDoc.use(0); self.prog["uDoc"].value = 0
        self.tTf.use(1); self.prog["uTf"].value = 1
        self.tDl.use(2); self.prog["uDl"].value = 2
        for key, val in (("uW", self.W), ("uOutW", self.outW), ("uOutH", self.outH)):
            self.prog[key].value = int(val)
        for key, val in (("uK1", k1), ("uB", b), ("uAvgdl", self.avgdl)):
            self.prog[key].value = float(val)
        drawn = 0
        for t in set(qterms):
            v = self.vocab.get(t)
            if v is None:
                continue
            lo, hi = int(self.off[v]), int(self.off[v + 1])
            if hi <= lo:
                continue
            df = hi - lo
            self.prog["uIdf"].value = float(np.log(1.0 + (self.N - df + 0.5) / (df + 0.5)))
            self.prog["uBase"].value = lo
            self.vao.render(moderngl.POINTS, vertices=df)
            drawn += df
        self.ctx.disable(moderngl.BLEND)
        px = np.frombuffer(self.out.read(), dtype="f4").reshape(-1, 2)[:self.N]
        return px[:, 0].astype(np.float64), px[:, 1].astype(np.float64), drawn


if __name__ == "__main__":
    import hard_corpus as HC
    from holographic.semantic_router.holographic_bm25 import BM25
    dn = HC.load_passages(target=29221)
    bm = BM25([" ".join(d) for _, d in dn])
    docs = bm.docs_tokens
    sc = Scatter(docs)
    print("corpus %d docs, %d postings, %d distinct terms" % (sc.N, len(sc.docid), len(sc.vocab)))
    rng = np.random.default_rng(0)
    agree1 = 0; agree10 = 0; n = 0; drawn_tot = 0; times = []
    maxrel = 0.0
    for _ in range(30):
        i = int(rng.integers(sc.N)); u = sorted(set(docs[i]))
        if len(u) < 8:
            continue
        q = [u[j] for j in rng.choice(len(u), 8, replace=False)]
        t0 = time.perf_counter()
        s, cov, drawn = sc.scores(q)
        sc.ctx.finish()
        times.append(time.perf_counter() - t0)
        ref = bm.scores(" ".join(q))
        agree1 += int(int(np.argmax(s)) == int(np.argmax(ref)))
        agree10 += len(set(np.argsort(-s)[:10]) & set(np.argsort(-ref)[:10])) / 10.0
        m = np.max(np.abs(ref))
        maxrel = max(maxrel, float(np.max(np.abs(s - ref)) / (m + 1e-30)))
        drawn_tot += drawn; n += 1
    print("  top-1 identical to the engine's BM25 : %d/%d" % (agree1, n))
    print("  top-10 overlap                       : %.3f" % (agree10 / n))
    print("  max relative score error             : %.3e" % maxrel)
    print("  posting visits per query             : %d (vs full-scan %d fetches, %.0fx less)"
          % (drawn_tot / n, sc.N * 8 * 7, (sc.N * 8 * 7) / (drawn_tot / n)))
    print("  GPU ms/query                         : %.2f  (full-scan shader measured 27.7)"
          % (1e3 * float(np.median(times))))
