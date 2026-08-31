"""Run the shader set on REAL HARDWARE and report correctness AND timing. Software raster is refused.

WHY THIS EXISTS. Every timing figure produced in the development container came from Mesa llvmpipe,
which is a CPU rasteriser. A "GPU vs CPU" number measured there is CPU-vs-CPU with extra copies,
and labelling it does not make it a hardware result -- so one such verdict ("the perfect-recall
candidate pass is slower on the GPU") has been RETRACTED rather than caveated. This harness is the
thing that replaces it.

IT REFUSES TO PRODUCE A TIMING TABLE ON A SOFTWARE RASTERISER unless --allow-software is passed,
and when it does, it stamps every row. A benchmark that silently accepts llvmpipe is how the
retracted number happened in the first place.

RUN IT:
    pip install numpy moderngl
    python bench_gpu.py                      # normal run, refuses software raster
    python bench_gpu.py --json results.json  # also write machine-readable results
    python bench_gpu.py --allow-software     # only if you know why you want that

WHAT IT MEASURES, per kernel: a CORRECTNESS differential against a NumPy reference computed here
(so a wrong shader fails loudly on the target machine, not just in the container), then wall-clock
for the GPU path and the NumPy path over repeated runs, reporting the MEDIAN and the spread.
Correctness is reported first and a failing kernel prints no timing at all -- a fast wrong answer
is not a result.

WINDOWS / MINGW64 NOTE: the backend is left to moderngl's default (WGL) unless EGL is requested,
because forcing EGL is what a Linux container needs and what a Windows box does not have.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

try:
    import moderngl
except ImportError:                                     # pragma: no cover - environment probe
    print("moderngl is required: pip install moderngl numpy")
    sys.exit(2)

SOFTWARE_MARKERS = ("llvmpipe", "softpipe", "swiftshader", "software", "lavapipe")


def make_context(force_egl=False):
    if force_egl:
        return moderngl.create_standalone_context(require=330, backend="egl")
    try:
        return moderngl.create_standalone_context(require=330)
    except Exception:
        return moderngl.create_standalone_context(require=330, backend="egl")


VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

FS_DIFFUSE = """
#version 330 core
uniform sampler2D uT; uniform int uW, uH; uniform float uR;
out float fragOut;
float at(int x, int y){ x = clamp(x,0,uW-1); y = clamp(y,0,uH-1); return texelFetch(uT, ivec2(x,y),0).r; }
void main(){
  int x = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
  float c = at(x,y);
  float n = (x>0 ? at(x-1,y) : c) + (x<uW-1 ? at(x+1,y) : c)
          + (y>0 ? at(x,y-1) : c) + (y<uH-1 ? at(x,y+1) : c);
  fragOut = c + uR * (n - 4.0*c);
}
"""

FS_PLANEWAVE = """
#version 330 core
uniform sampler2D uX, uW, uMag, uArg;
uniform int uK, uNP;
out float fragOut;
void main(){
  int i = int(gl_FragCoord.x);
  if (i >= uNP) { fragOut = 0.0; return; }
  float x = texelFetch(uX, ivec2(i,0),0).r;
  float g = 0.0;
  for (int k = 0; k < uK; ++k) {
    float w = texelFetch(uW, ivec2(k,0),0).r;
    float m = texelFetch(uMag, ivec2(k,0),0).r;
    float a = texelFetch(uArg, ivec2(k,0),0).r;
    g += m * w * sin(a - w * x);
  }
  fragOut = g;
}
"""

FS_FORM = """
#version 330 core
uniform sampler2D uBasis, uParams;
uniform int uL, uW, uTexW;
out float fragOut;
// FLAT 2D ADDRESSING. A (lights x pixels) texture is 65,536 rows tall at 256x256 -- past
// GL_MAX_TEXTURE_SIZE on most devices, which does not raise, it samples garbage. Index linearly
// and wrap, the same fix the flat index used for deep candidate arrays.
ivec2 at(int i){ return ivec2(i % uTexW, i / uTexW); }
void main(){
  int px = int(gl_FragCoord.x) + int(gl_FragCoord.y) * uW;
  float s = 0.0;
  for (int l = 0; l < uL; ++l)
    s += texelFetch(uBasis, at(px * uL + l),0).r * texelFetch(uParams, ivec2(l,0),0).r;
  fragOut = s;
}
"""


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.quad = ctx.buffer(np.array([-1,-1, 3,-1, -1,3], dtype="f4").tobytes())

    def prog(self, fs):
        p = self.ctx.program(vertex_shader=VS, fragment_shader=fs)
        return p, self.ctx.vertex_array(p, [(self.quad, "2f", "p")])

    def texf(self, a, w=None, h=1):
        a = np.ascontiguousarray(np.asarray(a, dtype="f4"))
        w = w or a.size
        t = self.ctx.texture((w, h), 1, a.tobytes(), dtype="f4")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return t


def time_it(fn, repeats):
    """Median and spread over repeats, with one untimed warm-up -- the first call pays for shader
    validation and buffer allocation on most drivers, and including it makes a GPU look slow."""
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return float(np.median(ts)), float(ts.std())


def bench_diffuse(h, repeats):
    rng = np.random.default_rng(0)
    H, W = 512, 512
    T0 = rng.random((H, W)); T0[H//3:H//2, W//4:W//2] += 5.0
    r, steps = 0.24, 40
    prog, vao = h.prog(FS_DIFFUSE)
    a = h.ctx.texture((W, H), 1, np.ascontiguousarray(T0, dtype="f4").tobytes(), dtype="f4")
    b = h.ctx.texture((W, H), 1, dtype="f4")
    for t in (a, b): t.filter = (moderngl.NEAREST, moderngl.NEAREST)
    fa, fb = h.ctx.framebuffer([a]), h.ctx.framebuffer([b])
    prog["uW"].value = W; prog["uH"].value = H; prog["uR"].value = r

    def gpu():
        src, fdst = a, fb
        for _ in range(steps):
            fdst.use(); h.ctx.viewport = (0,0,W,H)
            src.use(0); prog["uT"].value = 0
            vao.render(moderngl.TRIANGLES)
            src = b if src is a else a
            fdst = fa if fdst is fb else fb
        h.ctx.finish()
        return np.frombuffer(src.read(), dtype="f4").reshape(H, W).astype(np.float64)

    def cpu():
        c = T0.copy()
        for _ in range(steps):
            l = np.concatenate([c[:, :1], c[:, :-1]], 1); rr = np.concatenate([c[:, 1:], c[:, -1:]], 1)
            u = np.concatenate([c[:1, :], c[:-1, :]], 0); d = np.concatenate([c[1:, :], c[-1:, :]], 0)
            c = c + r * (l + rr + u + d - 4.0*c)
        return c

    got, ref = gpu(), cpu()
    err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30))
    return err, time_it(gpu, repeats), time_it(cpu, repeats), "%dx%d, %d steps" % (H, W, steps)


def bench_planewave(h, repeats):
    rng = np.random.default_rng(0)
    K, NP = 2048, 4096
    w = rng.uniform(-40, 40, K); mag = rng.random(K); arg = rng.uniform(-np.pi, np.pi, K)
    X = rng.uniform(0, 1, NP)
    prog, vao = h.prog(FS_PLANEWAVE)
    tX, tW, tM, tA = (h.texf(X), h.texf(w), h.texf(mag), h.texf(arg))
    out = h.ctx.texture((NP, 1), 1, dtype="f4")
    fbo = h.ctx.framebuffer([out])

    def gpu():
        fbo.use(); h.ctx.viewport = (0,0,NP,1)
        for n, t, u in (("uX",tX,0),("uW",tW,1),("uMag",tM,2),("uArg",tA,3)):
            t.use(u); prog[n].value = u
        prog["uK"].value = K; prog["uNP"].value = NP
        vao.render(moderngl.TRIANGLES); h.ctx.finish()
        return np.frombuffer(out.read(), dtype="f4").astype(np.float64)

    def cpu():
        return (mag * w * np.sin(arg[None, :] - w[None, :] * X[:, None])).sum(1)

    got, ref = gpu(), cpu()
    err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30))
    return err, time_it(gpu, repeats), time_it(cpu, repeats), "%d particles x %d waves" % (NP, K)


def bench_form(h, repeats):
    rng = np.random.default_rng(0)
    W, H, L = 256, 256, 64
    B = rng.random((W*H, L)) * 200.0
    params = rng.random(L)
    prog, vao = h.prog(FS_FORM)
    TEXW = 2048
    flat = np.ascontiguousarray(B, dtype="f4").reshape(-1)
    rows = (flat.size + TEXW - 1) // TEXW
    buf = np.zeros(rows * TEXW, dtype="f4"); buf[:flat.size] = flat
    tb = h.ctx.texture((TEXW, rows), 1, buf.tobytes(), dtype="f4")
    tp = h.texf(params)
    for t in (tb, tp): t.filter = (moderngl.NEAREST, moderngl.NEAREST)
    out = h.ctx.texture((W, H), 1, dtype="f4")
    fbo = h.ctx.framebuffer([out])

    def gpu():
        fbo.use(); h.ctx.viewport = (0,0,W,H)
        tb.use(0); prog["uBasis"].value = 0
        tp.use(1); prog["uParams"].value = 1
        prog["uL"].value = L; prog["uW"].value = W; prog["uTexW"].value = TEXW
        vao.render(moderngl.TRIANGLES); h.ctx.finish()
        return np.frombuffer(out.read(), dtype="f4").astype(np.float64)

    def cpu():
        return B @ params

    got, ref = gpu(), cpu()
    err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30))
    # the contract that matters for this one: does the QUANTISED image agree, and by how much margin
    qg = np.clip(np.round(got), 0, 255).astype(int)
    qr = np.clip(np.round(ref), 0, 255).astype(int)
    frac = np.abs(ref - np.floor(ref) - 0.5)
    extra = "%d/%d px differ, rounding margin %.2e" % (int((qg != qr).sum()), qg.size, float(frac.min()))
    return err, time_it(gpu, repeats), time_it(cpu, repeats), "%dx%d, %d lights; %s" % (W, H, L, extra)


FS_CONST = "#version 330 core\nuniform float uV;\nout float o;\nvoid main(){ o = uV; }\n"


def bench_readback(h, repeats):
    """The FLOOR: a shader that computes nothing, timed at each kernel's readback size.

    WHY THIS IS HERE. On an A4500 the diffusion kernel does FORTY passes and image formation does
    ONE, yet both landed near 2.52 ms -- while the plane-wave kernel, which reads back 16-64x fewer
    floats, was the only one under 2 ms. That is the shape of a fixed per-call cost, not of
    compute. Without this row, "image formation is 0.37x" reads as "the GPU is bad at GEMV" when it
    may mostly be the readback. A benchmark that cannot separate those two is measuring the bus and
    calling it the kernel.
    """
    prog, vao = h.prog(FS_CONST)
    rows = []
    for label, (w, hh) in (("4096 floats", (4096, 1)), ("65536 floats", (256, 256)),
                           ("262144 floats", (512, 512))):
        out = h.ctx.texture((w, hh), 1, dtype="f4")
        fbo = h.ctx.framebuffer([out])

        def once():
            fbo.use(); h.ctx.viewport = (0, 0, w, hh)
            prog["uV"].value = 1.0
            vao.render(moderngl.TRIANGLES); h.ctx.finish()
            return np.frombuffer(out.read(), dtype="f4")

        got = once()
        ms, sd = time_it(once, repeats)
        rows.append("%s %.2f ms" % (label, ms * 1e3))
        err = float(np.max(np.abs(got - 1.0)))
        out.release(); fbo.release()
    return err, (0.0, 0.0), (0.0, 0.0), "FLOOR (empty shader): " + ", ".join(rows)



# ---------------------------------------------------------------------------------------------
# Retrieval kernels. These carry the arc's largest claims and had NO hardware numbers before now.
# ---------------------------------------------------------------------------------------------

VS_SCATTER = """
#version 330 core
uniform usampler2D uPDoc, uPTf;
uniform sampler2D uDl;
uniform int uBase, uW, uOutW, uOutH;
uniform float uIdf, uK1, uB, uAvgdl;
out float vC;
ivec2 at(int i){ return ivec2(i % uW, i / uW); }
void main(){
  int p = uBase + gl_VertexID;
  int d = int(texelFetch(uPDoc, at(p), 0).r);
  float tf = float(texelFetch(uPTf, at(p), 0).r);
  float dl = texelFetch(uDl, at(d), 0).r;
  vC = uIdf * tf * (uK1 + 1.0) / (tf + uK1 * (1.0 - uB + uB * dl / uAvgdl));
  float x = (float(d % uOutW) + 0.5) / float(uOutW) * 2.0 - 1.0;
  float y = (float(d / uOutW) + 0.5) / float(uOutH) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_PointSize = 1.0;
}
"""

FS_SCATTER = """
#version 330 core
in float vC; out vec2 o;
void main(){ o = vec2(vC, 1.0); }
"""

FS_FULLSCAN = """
#version 330 core
uniform usampler2D uTerm, uTf, uOff, uQ;
uniform sampler2D uIdf, uDl;
uniform int uNQ, uW, uOutW, uN;
uniform float uK1, uB, uAvgdl;
out vec2 o;
ivec2 at(int i){ return ivec2(i % uW, i / uW); }
void main(){
  int d = int(gl_FragCoord.x) + int(gl_FragCoord.y) * uOutW;
  if (d >= uN) { o = vec2(-1e30, 0.0); return; }
  int lo = int(texelFetch(uOff, at(d), 0).r), hi = int(texelFetch(uOff, at(d + 1), 0).r);
  float dl = texelFetch(uDl, at(d), 0).r;
  float norm = uK1 * (1.0 - uB + uB * dl / uAvgdl);
  float s = 0.0, cov = 0.0;
  for (int j = 0; j < uNQ; ++j) {
    uint q = texelFetch(uQ, ivec2(j, 0), 0).r;
    int a = lo, b = hi, found = -1;
    while (a < b) { int m = (a + b) >> 1; uint v = texelFetch(uTerm, at(m), 0).r;
      if (v == q) { found = m; break; } else if (v < q) a = m + 1; else b = m; }
    if (found >= 0) { float tf = float(texelFetch(uTf, at(found), 0).r); cov += 1.0;
      s += texelFetch(uIdf, ivec2(j, 0), 0).r * tf * (uK1 + 1.0) / (tf + norm); }
  }
  o = vec2(s, cov);
}
"""


_FALLBACK_REASON = []


def _corpus(target=20000):
    """Real stratified repo passages when importable; a labelled synthetic Zipf corpus otherwise."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))          # research/shader_retrieval -> repo root
    for pth in (here, root):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    cwd = os.getcwd()
    try:
        os.chdir(root)                                     # hard_corpus globs relative to the CWD
        import hard_corpus as HC
        from holographic.semantic_router.holographic_bm25 import BM25
        dn = HC.load_passages(target=target)
        docs = [list(d) for d in BM25([" ".join(d) for _, d in dn]).docs_tokens]
        return docs, "repo corpus (stratified, family coverage asserted)"
    except Exception as exc:
        _FALLBACK_REASON.append("%s: %s" % (type(exc).__name__, str(exc)[:90]))
    finally:
        os.chdir(cwd)
    if True:
        rng = np.random.default_rng(0)
        V = 20000
        zipf = 1.0 / np.arange(1, V + 1) ** 1.07
        zipf /= zipf.sum()
        docs = [["t%d" % t for t in rng.choice(V, int(rng.integers(80, 220)), p=zipf)]
                for _ in range(target)]
        why = _FALLBACK_REASON[-1] if _FALLBACK_REASON else "repo not importable"
        return docs, ("SYNTHETIC Zipf corpus -- NOT REAL TEXT. Fell back because: " + why)


def _index(docs):
    vocab, post, dl = {}, {}, np.zeros(len(docs), dtype="f4")
    for i, d in enumerate(docs):
        c = {}
        for t in d:
            c[vocab.setdefault(t, len(vocab))] = c.get(vocab.setdefault(t, len(vocab)), 0) + 1
        for v, f in c.items():
            post.setdefault(v, []).append((i, f))
        dl[i] = len(d)
    off = np.zeros(len(vocab) + 1, dtype=np.int64)
    pdoc, ptf = [], []
    for v in range(len(vocab)):
        rows = post.get(v, [])
        off[v + 1] = off[v] + len(rows)
        for i, f in rows:
            pdoc.append(i); ptf.append(f)
    return (vocab, np.array(pdoc, dtype="<u4"), np.array(ptf, dtype="<u4"), off, dl,
            np.diff(off).astype(float))


_SHARED = {}


def _shared():
    """ONE corpus, index and query set for every retrieval bench in a run.

    Scatter's headline is a ratio AGAINST the full scan, so the two must see identical data and
    identical queries or the ratio is comparing fixtures. Built once, cached, reused.
    """
    if not _SHARED:
        docs, label = _corpus()
        vocab, pdoc, ptf, off, dl, df = _index(docs)
        rng = np.random.default_rng(0)
        terms = [v for v in range(len(vocab)) if df[v] >= 2]
        qs = [[terms[j] for j in rng.choice(len(terms), int(rng.integers(2, 5)), replace=False)]
              for _ in range(24)]
        _SHARED.update(docs=docs, label=label, vocab=vocab, pdoc=pdoc, ptf=ptf, off=off,
                       dl=dl, df=df, qs=qs, N=len(docs), avgdl=float(dl.mean()), W=4096)
    return _SHARED


def bench_scatter(h, repeats):
    S = _shared()
    docs, label, vocab = S["docs"], S["label"], S["vocab"]
    pdoc, ptf, off, dl, df = S["pdoc"], S["ptf"], S["off"], S["dl"], S["df"]
    N, avgdl, W, qs = S["N"], S["avgdl"], S["W"], S["qs"]
    prog = h.ctx.program(vertex_shader=VS_SCATTER, fragment_shader=FS_SCATTER)
    vao = h.ctx.vertex_array(prog, [])

    def texu(a):
        a = np.asarray(a, dtype="<u4").reshape(-1)
        rows = max(1, (a.size + W - 1) // W)
        buf = np.zeros(rows * W, dtype="<u4"); buf[:a.size] = a
        t = h.ctx.texture((W, rows), 1, buf.tobytes(), dtype="u4")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST); return t

    def texf(a):
        a = np.asarray(a, dtype="f4").reshape(-1)
        rows = max(1, (a.size + W - 1) // W)
        buf = np.zeros(rows * W, dtype="f4"); buf[:a.size] = a
        t = h.ctx.texture((W, rows), 1, buf.tobytes(), dtype="f4")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST); return t

    tD, tT, tL = texu(pdoc), texu(ptf), texf(dl)
    outW = min(W, N); outH = (N + outW - 1) // outW
    out = h.ctx.texture((outW, outH), 2, dtype="f4")
    fbo = h.ctx.framebuffer([out])

    def gpu_one(q):
        fbo.use(); h.ctx.viewport = (0, 0, outW, outH)
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        h.ctx.enable(moderngl.BLEND); h.ctx.blend_func = moderngl.ONE, moderngl.ONE
        tD.use(0); prog["uPDoc"].value = 0
        tT.use(1); prog["uPTf"].value = 1
        tL.use(2); prog["uDl"].value = 2
        for k, v in (("uW", W), ("uOutW", outW), ("uOutH", outH)):
            prog[k].value = int(v)
        for k, v in (("uK1", 1.5), ("uB", 0.75), ("uAvgdl", avgdl)):
            prog[k].value = float(v)
        for v in set(q):
            lo, hi = int(off[v]), int(off[v + 1])
            if hi <= lo: continue
            prog["uIdf"].value = float(np.log(1.0 + (N - (hi - lo) + 0.5) / ((hi - lo) + 0.5)))
            prog["uBase"].value = lo
            vao.render(moderngl.POINTS, vertices=hi - lo)
        h.ctx.disable(moderngl.BLEND)
        return np.frombuffer(out.read(), dtype="f4").reshape(-1, 2)[:N, 0].astype(np.float64)

    def cpu_one(q):
        s = np.zeros(N)
        for v in set(q):
            lo, hi = int(off[v]), int(off[v + 1])
            if hi <= lo: continue
            idf = np.log(1.0 + (N - (hi - lo) + 0.5) / ((hi - lo) + 0.5))
            ids = pdoc[lo:hi]; tf = ptf[lo:hi].astype(float)
            s[ids] += idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl[ids] / avgdl))
        return s

    err = 0.0; top1 = 0
    for q in qs:
        g, c = gpu_one(q), cpu_one(q)
        err = max(err, float(np.max(np.abs(g - c)) / (np.max(np.abs(c)) + 1e-30)))
        top1 += int(np.argmax(g) == np.argmax(c))

    def gpu(): [gpu_one(q) for q in qs]; h.ctx.finish()
    def cpu(): [cpu_one(q) for q in qs]
    note = "%s, %d docs, %d postings, %d queries (2-4 terms); top-1 %d/%d" % (
        label, N, len(pdoc), len(qs), top1, len(qs))
    return err, time_it(gpu, max(3, repeats // 3)), time_it(cpu, max(3, repeats // 3)), note




def bench_fullscan(h, repeats):
    """The baseline scatter's ratio is measured AGAINST -- one fragment per document, binary search
    per query term. Same corpus, same index, same queries as bench_scatter, by construction."""
    S = _shared()
    pdoc, ptf, off, dl, df = S["pdoc"], S["ptf"], S["off"], S["dl"], S["df"]
    docs, N, avgdl, W, qs = S["docs"], S["N"], S["avgdl"], S["W"], S["qs"]
    vocab = S["vocab"]
    # the full scan wants the transposed layout: per-doc sorted-unique terms + tf
    term_l, tf_l, doff = [], [], [0]
    for d in docs:
        c = {}
        for t in d:
            v = vocab[t]; c[v] = c.get(v, 0) + 1
        for v in sorted(c):
            term_l.append(v); tf_l.append(c[v])
        doff.append(len(term_l))
    term = np.array(term_l, dtype="<u4"); tfv = np.array(tf_l, dtype="<u4")
    doffv = np.array(doff, dtype="<u4")

    def texu(a):
        a = np.asarray(a, dtype="<u4").reshape(-1)
        rows = max(1, (a.size + W - 1) // W)
        buf = np.zeros(rows * W, dtype="<u4"); buf[:a.size] = a
        t = h.ctx.texture((W, rows), 1, buf.tobytes(), dtype="u4")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST); return t

    def texf(a):
        a = np.asarray(a, dtype="f4").reshape(-1)
        rows = max(1, (a.size + W - 1) // W)
        buf = np.zeros(rows * W, dtype="f4"); buf[:a.size] = a
        t = h.ctx.texture((W, rows), 1, buf.tobytes(), dtype="f4")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST); return t

    prog, vao = h.prog(FS_FULLSCAN)
    tT, tF, tO, tL = texu(term), texu(tfv), texu(doffv), texf(dl)
    outW = min(W, N); outH = (N + outW - 1) // outW
    out = h.ctx.texture((outW, outH), 2, dtype="f4")
    fbo = h.ctx.framebuffer([out])

    def gpu_one(q):
        qids = np.array(sorted(set(q)), dtype="<u4")
        idf = np.array([np.log(1.0 + (N - df[v] + 0.5) / (df[v] + 0.5)) for v in qids], dtype="f4")
        tq, ti = texu(qids), texf(idf)
        fbo.use(); h.ctx.viewport = (0, 0, outW, outH)
        for n, t, u in (("uTerm", tT, 0), ("uTf", tF, 1), ("uOff", tO, 2),
                        ("uQ", tq, 3), ("uIdf", ti, 4), ("uDl", tL, 5)):
            t.use(u); prog[n].value = u
        for k, v in (("uNQ", len(qids)), ("uW", W), ("uOutW", outW), ("uN", N)):
            prog[k].value = int(v)
        for k, v in (("uK1", 1.5), ("uB", 0.75), ("uAvgdl", avgdl)):
            prog[k].value = float(v)
        vao.render(moderngl.TRIANGLES)
        r = np.frombuffer(out.read(), dtype="f4").reshape(-1, 2)[:N, 0].astype(np.float64)
        tq.release(); ti.release()
        return r

    def cpu_one(q):
        s2 = np.zeros(N)
        for v in set(q):
            lo, hi = int(off[v]), int(off[v + 1])
            if hi <= lo: continue
            idf = np.log(1.0 + (N - (hi - lo) + 0.5) / ((hi - lo) + 0.5))
            ids = pdoc[lo:hi]; tf = ptf[lo:hi].astype(float)
            s2[ids] += idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl[ids] / avgdl))
        return s2

    err = 0.0; top1 = 0
    for q in qs:
        g, c = gpu_one(q), cpu_one(q)
        err = max(err, float(np.max(np.abs(g - c)) / (np.max(np.abs(c)) + 1e-30)))
        top1 += int(np.argmax(g) == np.argmax(c))

    def gpu(): [gpu_one(q) for q in qs]; h.ctx.finish()
    def cpu(): [cpu_one(q) for q in qs]
    note = "same corpus/queries as scatter; %d docs; top-1 %d/%d" % (N, top1, len(qs))
    return err, time_it(gpu, max(3, repeats // 3)), time_it(cpu, max(3, repeats // 3)), note


VS_PBD = """
#version 330 core
uniform sampler2D uX, uRest;
uniform usampler2D uEdge;
uniform int uNP, uW;
out vec3 vDelta;
void main(){
  int c = gl_VertexID / 2, side = gl_VertexID - c * 2;
  int ia = int(texelFetch(uEdge, ivec2(2*c, 0), 0).r);
  int ib = int(texelFetch(uEdge, ivec2(2*c+1, 0), 0).r);
  vec2 a = texelFetch(uX, ivec2(ia % uW, ia / uW), 0).rg;
  vec2 b = texelFetch(uX, ivec2(ib % uW, ib / uW), 0).rg;
  float rest = texelFetch(uRest, ivec2(c, 0), 0).r;
  vec2 d = b - a; float L = length(d);
  vec2 corr = (L > 1e-8) ? (0.5 * (L - rest) / L) * d : vec2(0.0);
  int me = (side == 0) ? ia : ib;
  vDelta = vec3((side == 0) ? corr : -corr, 1.0);
  float x = (float(me % uW) + 0.5) / float(uW) * 2.0 - 1.0;
  float y = (float(me / uW) + 0.5) / float((uNP + uW - 1) / uW) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0); gl_PointSize = 1.0;
}
"""
FS_PBD = "#version 330 core\nin vec3 vDelta; out vec3 o;\nvoid main(){ o = vDelta; }\n"
FS_PBD_APPLY = """
#version 330 core
uniform sampler2D uX, uAcc;
out vec2 o;
void main(){
  ivec2 t = ivec2(gl_FragCoord.xy);
  vec2 x = texelFetch(uX, t, 0).rg;
  vec3 a = texelFetch(uAcc, t, 0).rgb;
  o = x + ((a.z > 0.0) ? a.xy / a.z : vec2(0.0));
}
"""


def bench_pbd(h, repeats):
    """Jacobi PBD: scatter-add by blending, then ping-pong. NOT comparable to a sequential
    Gauss-Seidel sweep -- the CPU reference here is Jacobi too, and the physical invariant
    (constraint residual falling) is checked as well as the numbers."""
    n, iters = 64, 40
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(n), np.arange(n))
    X0 = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    X0 += 0.35 * rng.standard_normal(X0.shape)
    edges = []
    for y in range(n):
        for x in range(n):
            i = y * n + x
            if x + 1 < n: edges.append((i, i + 1))
            if y + 1 < n: edges.append((i, i + n))
    edges = np.array(edges, dtype=np.int64); rest = np.ones(len(edges))
    NP = len(X0); W = 64; H = (NP + W - 1) // W
    ps = h.ctx.program(vertex_shader=VS_PBD, fragment_shader=FS_PBD)
    pa, va = h.prog(FS_PBD_APPLY)
    vs = h.ctx.vertex_array(ps, [])
    buf = np.zeros((H, W, 2), dtype="f4"); buf.reshape(-1, 2)[:NP] = X0
    xa = h.ctx.texture((W, H), 2, buf.tobytes(), dtype="f4")
    xb = h.ctx.texture((W, H), 2, dtype="f4")
    acc = h.ctx.texture((W, H), 3, dtype="f4")
    for t in (xa, xb, acc): t.filter = (moderngl.NEAREST, moderngl.NEAREST)
    e = np.zeros(2 * len(edges), dtype="<u4"); e[0::2] = edges[:, 0]; e[1::2] = edges[:, 1]
    te = h.ctx.texture((len(e), 1), 1, e.tobytes(), dtype="u4")
    tr = h.ctx.texture((len(rest), 1), 1, np.asarray(rest, dtype="f4").tobytes(), dtype="f4")
    facc, fa, fb = h.ctx.framebuffer([acc]), h.ctx.framebuffer([xa]), h.ctx.framebuffer([xb])

    def gpu():
        h.ctx.viewport = (0, 0, W, H)
        src, fdst = xa, fb
        # reset the start state so repeated timing runs are identical, not cumulative
        xa.write(buf.tobytes())
        for _ in range(iters):
            facc.use(); facc.clear(0.0, 0.0, 0.0, 0.0)
            h.ctx.enable(moderngl.BLEND); h.ctx.blend_func = moderngl.ONE, moderngl.ONE
            src.use(0); ps["uX"].value = 0
            te.use(1); ps["uEdge"].value = 1
            tr.use(2); ps["uRest"].value = 2
            ps["uNP"].value = NP; ps["uW"].value = W
            vs.render(moderngl.POINTS, vertices=2 * len(edges))
            h.ctx.disable(moderngl.BLEND)
            fdst.use()
            src.use(0); pa["uX"].value = 0
            acc.use(1); pa["uAcc"].value = 1
            va.render(moderngl.TRIANGLES)
            src = xb if src is xa else xa
            fdst = fa if fdst is fb else fb
        h.ctx.finish()
        return np.frombuffer(src.read(), dtype="f4").reshape(-1, 2)[:NP].astype(np.float64)

    def cpu():
        X = X0.copy()
        for _ in range(iters):
            a2 = np.zeros_like(X); cnt = np.zeros(len(X))
            d = X[edges[:, 1]] - X[edges[:, 0]]
            L = np.linalg.norm(d, axis=1); safe = L > 1e-8
            corr = np.zeros_like(d)
            corr[safe] = (0.5 * (L[safe] - rest[safe]) / L[safe])[:, None] * d[safe]
            np.add.at(a2, edges[:, 0], corr); np.add.at(a2, edges[:, 1], -corr)
            np.add.at(cnt, edges[:, 0], 1.0); np.add.at(cnt, edges[:, 1], 1.0)
            nz = cnt > 0
            X[nz] += a2[nz] / cnt[nz][:, None]
        return X

    g, c = gpu(), cpu()
    err = float(np.max(np.abs(g - c)) / (np.max(np.abs(c)) + 1e-30))
    resid = lambda P: float(np.sqrt(np.mean(
        (np.linalg.norm(P[edges[:, 1]] - P[edges[:, 0]], axis=1) - rest) ** 2)))
    note = "%dx%d cloth, %d constraints, %d Jacobi iters; residual %.4f -> %.4f (GPU) / %.4f (CPU)" % (
        n, n, len(edges), iters, resid(X0), resid(g), resid(c))
    return err, time_it(gpu, max(3, repeats // 3)), time_it(cpu, max(3, repeats // 3)), note




def bench_batch(h, repeats):
    """G1+G2: the whole query batch in ONE draw, and only the VERDICTS come back.

    The readback floor is per-CALL, not per-byte -- an empty shader reading 4 floats costs 0.009 ms
    and 65,536 floats costs 0.249 ms. So the lever is fewer calls, not smaller ones. Compared here
    against the SAME kernel driven one query at a time, on the same corpus and queries.
    """
    import glsl_batch as GB
    S = _shared()
    sc = GB.BatchScorer(S["docs"], ctx=h.ctx)
    inv = {v: k for k, v in sc.vocab.items()}
    qs = [[inv[v] for v in q if v in inv] for q in S["qs"]]
    qs = [q for q in qs if q] or [[next(iter(sc.vocab))]]
    v = sc.verdicts(qs)
    top1 = 0
    for i, q in enumerate(qs):
        s2 = np.zeros(sc.N)
        for t in set(q):
            vv = sc.vocab[t]; lo, hi = int(sc.off[vv]), int(sc.off[vv + 1])
            if hi <= lo: continue
            idf = np.log(1.0 + (sc.N - (hi - lo) + 0.5) / ((hi - lo) + 0.5))
            ids = sc.pdoc[lo:hi]; tf = sc.ptf[lo:hi].astype(float)
            s2[ids] += idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * sc.dl[ids] / sc.avgdl))
        top1 += int(int(v[i, 1]) == int(np.argmax(s2)))
    err = 0.0 if top1 == len(qs) else 1.0

    def gpu(): sc.verdicts(qs); h.ctx.finish()
    def one(): [sc.verdicts([q]) for q in qs]; h.ctx.finish()
    g = time_it(gpu, max(3, repeats // 3))
    o = time_it(one, max(3, repeats // 3))
    note = ("%d queries in ONE draw; one-at-a-time %.1f ms vs batched %.1f ms = %.1fx; "
            "top-1 %d/%d" % (len(qs), o[0] * 1e3, g[0] * 1e3, o[0] / max(g[0], 1e-9),
                             top1, len(qs)))
    return err, g, o, note

BENCHES_MORE = (("bm25 full scan", bench_fullscan),
                ("pbd cloth (Jacobi)", bench_pbd),
                ("batched scatter (G1+G2)", bench_batch))

BENCHES_RETRIEVAL = (("scatter inverted index", bench_scatter),) + BENCHES_MORE


BENCHES = (("readback floor", bench_readback),
           ("diffuse (ping-pong)", bench_diffuse),
           ("hdrift plane waves", bench_planewave),
           ("image formation", bench_form)) + BENCHES_RETRIEVAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--json", default=None)
    ap.add_argument("--egl", action="store_true", help="force the EGL backend (Linux headless)")
    ap.add_argument("--allow-software", action="store_true")
    args = ap.parse_args()

    ctx = make_context(force_egl=args.egl)
    info = {k: ctx.info.get(k, "?") for k in ("GL_RENDERER", "GL_VENDOR", "GL_VERSION")}
    soft = any(m in str(info["GL_RENDERER"]).lower() for m in SOFTWARE_MARKERS)
    print("device : %s" % info["GL_RENDERER"])
    print("vendor : %s   GL %s" % (info["GL_VENDOR"], info["GL_VERSION"]))
    print("host   : %s %s, numpy %s" % (platform.system(), platform.machine(), np.__version__))
    if soft:
        print("\n!! SOFTWARE RASTERISER DETECTED.")
        print("   A GPU-vs-CPU timing measured here is CPU-vs-CPU with extra copies. This harness")
        print("   refuses to print a timing table on software raster; correctness still runs.")
        if not args.allow_software:
            print("   Pass --allow-software if you understand that every row would be mislabelled.")
    print()

    rows = []
    print("  %-22s %-12s %-16s %-16s %s" % ("kernel", "max rel err", "GPU ms", "NumPy ms", "notes"))
    for name, fn in BENCHES:
        try:
            err, (gms, gsd), (cms, csd), note = fn(Harness(ctx), args.repeats)
        except Exception as exc:
            print("  %-22s FAILED: %s" % (name, exc))
            rows.append({"kernel": name, "error": str(exc)})
            continue
        ok = err < 1e-4
        gcell = "-" if (soft and not args.allow_software) else "%.2f +-%.2f" % (gms*1e3, gsd*1e3)
        ccell = "-" if (soft and not args.allow_software) else "%.2f +-%.2f" % (cms*1e3, csd*1e3)
        if not ok:
            gcell = ccell = "(suppressed)"
            note = "CORRECTNESS FAILED -- a fast wrong answer is not a result. " + note
        print("  %-22s %-12.3e %-16s %-16s %s" % (name, err, gcell, ccell, note))
        rows.append({"kernel": name, "max_rel_err": err, "correct": bool(ok),
                     "gpu_ms": gms*1e3, "gpu_sd_ms": gsd*1e3,
                     "cpu_ms": cms*1e3, "cpu_sd_ms": csd*1e3,
                     "note": note, "software_raster": bool(soft)})
    if not (soft and not args.allow_software):
        by = {r["kernel"]: r for r in rows if r.get("correct")}
        sc, fs = by.get("scatter inverted index"), by.get("bm25 full scan")
        if sc and fs:
            print("\n  THE HEADLINE -- both on THIS device, same corpus, same queries:")
            print("    bm25 full scan      %8.2f ms" % fs["gpu_ms"])
            print("    scatter (inverted)  %8.2f ms" % sc["gpu_ms"])
            print("    scatter is %.1fx the full scan%s"
                  % (fs["gpu_ms"] / max(sc["gpu_ms"], 1e-9),
                     "" if not soft else "  [SOFTWARE RASTER -- not a hardware result]"))
        print("\n  speedup = NumPy / GPU, medians:")
        for r in rows:
            if r.get("correct") and r["gpu_ms"] > 0 and r["cpu_ms"] > 0:
                print("    %-22s %.2fx" % (r["kernel"], r["cpu_ms"] / max(r["gpu_ms"], 1e-9)))
    out = {"device": info, "host": platform.platform(), "software_raster": bool(soft),
           "repeats": args.repeats, "results": rows}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print("\n  wrote %s" % args.json)


if __name__ == "__main__":
    main()
