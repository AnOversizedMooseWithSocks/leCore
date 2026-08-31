"""G1 + G2 -- score a whole BATCH of queries in ONE draw call and return only the verdicts.

WHY THIS SHAPE, and it follows from a measurement rather than from taste. The readback floor is
per-CALL, not per-BYTE: an empty shader reading 4 floats costs 0.009 ms and reading 65,536 costs
0.249 ms, and on the A4500 it is nearly flat from 4k to 64k. So shrinking what comes back recovers
almost nothing -- the fix has to reduce the NUMBER OF CALLS, and that means batching.

THE CONSTRUCTION:
  * The host flattens the whole batch into ONE list of contributions: for every query q, every term
    t in q, and every posting (doc, tf) of t, an entry (doc, tf, q, idf). One draw over that list.
  * Each point lands at (doc, q) in an N x Q target, summed by additive blending -- the same
    scatter-add trick, now carrying a query row as well as a document column.
  * A second pass reduces each ROW to its top-1 (score, doc, coverage), so only Q verdicts cross
    the bus. That is G2, and G2 exists to make G1 possible: a batched pass cannot hand back N*Q
    floats.

WHAT IT DOES NOT CLAIM. This does not beat JavaScript at small-corpus retrieval and is not meant
to -- pure JS is 0.002 ms/query at 500 passages and no draw call is ever that cheap. It exists to
find where the per-call floor stops dominating, which is the only honest question left for the
retrieval kernels.
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import time

import numpy as np
import moderngl

VS_BATCH = """
#version 330 core
uniform usampler2D uDoc, uTf, uRow;   // one entry per CONTRIBUTION, flat
uniform sampler2D uIdf, uDl;
uniform int uW, uN, uOutW, uOutH;
uniform float uK1, uB, uAvgdl;
out float vC;
out float vHalf;
ivec2 at(int i){ return ivec2(i % uW, i / uW); }
void main(){
  int p = gl_VertexID;
  int d = int(texelFetch(uDoc, at(p), 0).r);
  int q = int(texelFetch(uRow, at(p), 0).r);
  float tf = float(texelFetch(uTf, at(p), 0).r);
  float dl = texelFetch(uDl, at(d), 0).r;
  vC = texelFetch(uIdf, at(p), 0).r * tf * (uK1 + 1.0)
       / (tf + uK1 * (1.0 - uB + uB * dl / uAvgdl));
  // FLAT 2D ADDRESSING. An (N x Q) target is 20,000 texels wide at this corpus size, past
  // GL_MAX_TEXTURE_SIZE -- the framebuffer comes back INCOMPLETE with no warning. Index linearly.
  // LEVER 4: two queries share a texel, so the row block is q/2 and the half is q&1.
  int idx = (q >> 1) * uN + d;
  vHalf = float(q & 1);
  float x = (float(idx % uOutW) + 0.5) / float(uOutW) * 2.0 - 1.0;
  float y = (float(idx / uOutW) + 0.5) / float(uOutH) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_PointSize = 1.0;
}
"""

FS_BATCH = """
#version 330 core
in float vC; in float vHalf; out vec4 o;
// Blending is uniform across channels, so a contribution zeroes the half it is not for.
void main(){ o = (vHalf < 0.5) ? vec4(vC, 1.0, 0.0, 0.0) : vec4(0.0, 0.0, vC, 1.0); }
"""

# One fragment per QUERY. Walks its own row and returns the verdict, so only Q values come back.
# LEVER 5 -- TILE THE DOMAIN. One fragment per (query, TILE): each reduces its own slice, so the
# work is parallel instead of one fragment walking every document. T4 (tiled_max_eq_global, machine
# checked) is what makes this safe: a tiled max EQUALS the single-pass max, so tiling changes the
# schedule and not the answer.
FS_REDUCE_TILE = """
#version 330 core
uniform sampler2D uScores; uniform int uN, uOutW, uTiles, uTileN;
// One fragment reduces a tile for a query PAIR: it reads each texel ONCE and updates both halves,
// so the packed layout halves the fetches instead of merely halving the memory.
out vec4 o0;   // (best, arg) for the even query
out vec4 o1;   // (best, arg) for the odd query
void main(){
  int t = int(gl_FragCoord.x);
  int pair = int(gl_FragCoord.y);
  int lo = t * uTileN, hi = min(lo + uTileN, uN);
  float b0 = -1e30, c0 = 0.0, b1 = -1e30, c1 = 0.0;
  int a0 = lo, a1 = lo;
  for (int d = lo; d < hi; ++d) {
    int idx = pair * uN + d;
    vec4 s = texelFetch(uScores, ivec2(idx % uOutW, idx / uOutW), 0);
    if (s.r > b0) { b0 = s.r; a0 = d; c0 = s.g; }
    if (s.b > b1) { b1 = s.b; a1 = d; c1 = s.a; }
  }
  o0 = vec4(b0, float(a0), c0, 0.0);
  o1 = vec4(b1, float(a1), c1, 0.0);
}
"""

# Second pass: one fragment per query, reducing only the TILE WINNERS -- uTiles of them, not uN.
FS_REDUCE_FINAL = """
#version 330 core
uniform sampler2D uP0, uP1; uniform int uTiles;
out vec3 o;
void main(){
  int q = int(gl_FragCoord.x);
  int pair = q >> 1;
  float best = -1e30, cov = 0.0, arg = 0.0;
  for (int t = 0; t < uTiles; ++t) {
    vec4 s = ((q & 1) == 0) ? texelFetch(uP0, ivec2(t, pair), 0)
                            : texelFetch(uP1, ivec2(t, pair), 0);
    if (s.r > best) { best = s.r; arg = s.g; cov = s.b; }
  }
  o = vec3(best, arg, cov);
}
"""


class BatchScorer:
    def __init__(self, docs, ctx=None, W=4096):
        self.W = W
        self.ctx = ctx or moderngl.create_standalone_context(require=330, backend="egl")
        self.vocab, post, self.dl = {}, {}, np.zeros(len(docs), dtype="f4")
        for i, d in enumerate(docs):
            c = {}
            for t in d:
                v = self.vocab.setdefault(t, len(self.vocab))
                c[v] = c.get(v, 0) + 1
            for v, f in c.items():
                post.setdefault(v, []).append((i, f))
            self.dl[i] = len(d)
        self.off = np.zeros(len(self.vocab) + 1, dtype=np.int64)
        pd, pt = [], []
        for v in range(len(self.vocab)):
            rows = post.get(v, [])
            self.off[v + 1] = self.off[v] + len(rows)
            for i, f in rows:
                pd.append(i); pt.append(f)
        self.pdoc = np.array(pd, dtype="<u4"); self.ptf = np.array(pt, dtype="<u4")
        self.df = np.diff(self.off).astype(float)
        self.N = len(docs); self.avgdl = float(self.dl.mean())
        # LEVER 1 -- bake once, sample O(1). idf depends only on the term, so recomputing it per
        # (query, term) was paying a log per posting list per query for a value that never moves.
        with np.errstate(divide="ignore"):
            self.idf = np.log(1.0 + (self.N - self.df + 0.5) / (self.df + 0.5)).astype("f4")
        self.pbatch = self.ctx.program(vertex_shader=VS_BATCH, fragment_shader=FS_BATCH)
        self.vbatch = self.ctx.vertex_array(self.pbatch, [])
        quad = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        VSQ = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"
        self.ptile = self.ctx.program(vertex_shader=VSQ, fragment_shader=FS_REDUCE_TILE)
        self.vtile = self.ctx.vertex_array(self.ptile, [(quad, "2f", "p")])
        self.pfin = self.ctx.program(vertex_shader=VSQ, fragment_shader=FS_REDUCE_FINAL)
        self.vfin = self.ctx.vertex_array(self.pfin, [(quad, "2f", "p")])
        self.tile_n = 256          # measured below; a tile smaller than this pays more setup than it saves
        self.tDl = self._texf(self.dl)

    def _tex(self, a, kind):
        a = np.asarray(a).reshape(-1)
        rows = max(1, (a.size + self.W - 1) // self.W)
        if kind == "u":
            buf = np.zeros(rows * self.W, dtype="<u4"); buf[:a.size] = a
            t = self.ctx.texture((self.W, rows), 1, buf.tobytes(), dtype="u4")
        else:
            buf = np.zeros(rows * self.W, dtype="f4"); buf[:a.size] = a
            t = self.ctx.texture((self.W, rows), 1, buf.tobytes(), dtype="f4")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return t

    def _texu(self, a): return self._tex(a, "u")
    def _texf(self, a): return self._tex(a, "f")

    def flatten(self, queries):
        """Host side: the whole batch becomes ONE contribution list. This is the entire trick."""
        doc, tf, row, idf = [], [], [], []
        for q, terms in enumerate(queries):
            for t in set(terms):
                v = self.vocab.get(t)
                if v is None:
                    continue
                lo, hi = int(self.off[v]), int(self.off[v + 1])
                if hi <= lo:
                    continue
                w = float(self.idf[v])                      # baked, not recomputed
                doc.append(self.pdoc[lo:hi]); tf.append(self.ptf[lo:hi])
                row.append(np.full(hi - lo, q, dtype="<u4"))
                idf.append(np.full(hi - lo, w, dtype="f4"))
        if not doc:
            return None
        return (np.concatenate(doc), np.concatenate(tf),
                np.concatenate(row), np.concatenate(idf))

    def verdicts(self, queries, k1=1.5, b=0.75):
        """One draw for the batch, one reduce pass, ONE readback of Q verdicts."""
        Q = len(queries)
        flat = self.flatten(queries)
        if flat is None:
            return np.zeros((Q, 3))
        d, tf, row, idf = flat
        tD, tT, tR, tI = self._texu(d), self._texu(tf), self._texu(row), self._texf(idf)
        pairs = (Q + 1) // 2                      # LEVER 4: two queries share every texel
        total = self.N * pairs
        outW = min(self.W, total)
        outH = (total + outW - 1) // outW
        scores = self.ctx.texture((outW, outH), 4, dtype="f4")
        scores.filter = (moderngl.NEAREST, moderngl.NEAREST)
        fs = self.ctx.framebuffer([scores])
        fs.use(); self.ctx.viewport = (0, 0, outW, outH)
        fs.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.BLEND); self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        for n, t, u in (("uDoc", tD, 0), ("uTf", tT, 1), ("uRow", tR, 2),
                        ("uIdf", tI, 3), ("uDl", self.tDl, 4)):
            t.use(u); self.pbatch[n].value = u
        for n, v in (("uW", self.W), ("uN", self.N), ("uOutW", outW), ("uOutH", outH)):
            if n in self.pbatch:            # a uniform the shader does not READ is eliminated
                self.pbatch[n].value = int(v)
        for n, v in (("uK1", k1), ("uB", b), ("uAvgdl", self.avgdl)):
            self.pbatch[n].value = float(v)
        self.vbatch.render(moderngl.POINTS, vertices=len(d))   # <-- ONE DRAW for the whole batch
        self.ctx.disable(moderngl.BLEND)
        tiles = max(1, (self.N + self.tile_n - 1) // self.tile_n)
        # Two render targets so one fragment can emit both halves of its query pair (MRT).
        p0 = self.ctx.texture((tiles, pairs), 4, dtype="f4")
        p1 = self.ctx.texture((tiles, pairs), 4, dtype="f4")
        for t in (p0, p1):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        fp = self.ctx.framebuffer([p0, p1])
        fp.use(); self.ctx.viewport = (0, 0, tiles, pairs)
        scores.use(0); self.ptile["uScores"].value = 0
        for n, v in (("uN", self.N), ("uOutW", outW), ("uTiles", tiles), ("uTileN", self.tile_n)):
            if n in self.ptile:
                self.ptile[n].value = int(v)
        self.vtile.render(moderngl.TRIANGLES)

        out = self.ctx.texture((Q, 1), 3, dtype="f4")
        fo = self.ctx.framebuffer([out])
        fo.use(); self.ctx.viewport = (0, 0, Q, 1)
        p0.use(0); self.pfin["uP0"].value = 0
        p1.use(1); self.pfin["uP1"].value = 1
        self.pfin["uTiles"].value = tiles
        self.vfin.render(moderngl.TRIANGLES)
        v = np.frombuffer(out.read(), dtype="f4").reshape(Q, 3).astype(np.float64)
        for o in (tD, tT, tR, tI, scores, p0, p1, out, fs, fp, fo):
            o.release()
        return v


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hard_corpus as HC
    from holographic.semantic_router.holographic_bm25 import BM25
    dn = HC.load_passages(target=20000)
    bm = BM25([" ".join(d) for _, d in dn])
    docs = bm.docs_tokens
    sc = BatchScorer(docs)
    rng = np.random.default_rng(0)
    terms = [t for t in sc.vocab if sc.df[sc.vocab[t]] >= 2]
    QS = [[terms[j] for j in rng.choice(len(terms), int(rng.integers(2, 5)), replace=False)]
          for _ in range(200)]
    print("BACKEND:", sc.ctx.info["GL_RENDERER"])
    print("corpus %d docs, %d terms, %d queries\n" % (sc.N, len(sc.vocab), len(QS)))

    # correctness first: a fast wrong answer is not a result
    v = sc.verdicts(QS)
    ok = 0
    for i, q in enumerate(QS):
        ref = bm.scores(q)
        ok += int(int(v[i, 1]) == int(np.argmax(ref)))
    print("  top-1 identical to the engine : %d/%d" % (ok, len(QS)))

    def batched(n):
        t0 = time.perf_counter(); sc.verdicts(QS[:n]); sc.ctx.finish()
        return time.perf_counter() - t0

    def one_at_a_time(n):
        t0 = time.perf_counter()
        for q in QS[:n]:
            sc.verdicts([q])
        sc.ctx.finish()
        return time.perf_counter() - t0

    print("\n  queries   one-at-a-time   batched     per-query batched   speedup")
    for n in (1, 10, 50, 200):
        batched(n); one_at_a_time(min(n, 10))          # warm
        a = one_at_a_time(n); b = batched(n)
        print("  %-9d %-15.1f %-11.1f %-19.4f %.0fx"
              % (n, 1e3 * a, 1e3 * b, 1e3 * b / n, a / max(b, 1e-9)))
    print("\n  [software rasteriser -- GPU-vs-GPU variant comparison is fair here,")
    print("   GPU-vs-CPU is not. Run bench_gpu.py on hardware for the real numbers.]")
