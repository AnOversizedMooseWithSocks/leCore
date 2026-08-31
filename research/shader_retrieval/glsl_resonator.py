"""The RESONATOR in GLSL -- and it is PBD wearing a different costume.

RULE 0 FOUND THIS, NOT INVENTION. `project_onto_constraints`'s own docstring says it plainly: the
SBC resonator, `denoise(method='pnp')` and a PBD constraint sweep are ONE ENGINE -- alternating
projection, Macklin's observation. PBD is already ported and wins 51x on an A4500. The resonator is
the same shape and was never ported, so the capability was sitting one costume change away.

WHAT IT DOES: given a composite s = x1 (*) x2 (*) ... (*) xF and one codebook per factor, recover
which entry of each codebook was bound in. Elementwise +/-1 binding, so a factor is its own inverse
and unbinding is another multiply.

THE SHAPE, assembled from parts this repo had already verified:
    probe   one fragment per (factor, dim)   -- elementwise, the cheapest shader there is
    score   one fragment per (factor, entry) -- a gather/matvec, the same shape as bm25_score
    argmax  tiled reduce over entries        -- T4 (tiled_max_eq_global) says tiling is exact
    gather  one fragment per (factor, dim)   -- pick the winning row back out
Four passes per ITERATION regardless of how many factors there are, because all F factors are
updated from the SAME state -- JACOBI, which is exactly what PBD taught. A Gauss-Seidel sweep would
need 4F passes and converge in fewer iterations; on a GPU that trade goes the other way.

KEPT NEGATIVE, inherited and re-stated: Jacobi is NOT the sequential sweep. It converges more
slowly per iteration and the iteration cap is real. Convergence here is OBSERVED, not proved.
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# probe[f][i] = composite[i] * prod over g != f of est[g][i].  All F rows in one pass.
FS_PROBE = """
#version 330 core
uniform sampler2D uComp, uEst; uniform int uF, uD;
out float o;
void main(){
  int i = int(gl_FragCoord.x);
  int f = int(gl_FragCoord.y);
  float v = texelFetch(uComp, ivec2(i, 0), 0).r;
  for (int g = 0; g < uF; ++g)
    if (g != f) v *= texelFetch(uEst, ivec2(i, g), 0).r;
  o = v;
}
"""

# score[f][m] = dot(codebook[f][m], probe[f]).  A gather, the same shape as the BM25 scorer.
FS_SCORE = """
#version 330 core
uniform sampler2D uBook, uProbe; uniform int uD, uM, uTexW;
out float o;
ivec2 at(int i){ return ivec2(i % uTexW, i / uTexW); }
void main(){
  int m = int(gl_FragCoord.x);
  int f = int(gl_FragCoord.y);
  float s = 0.0;
  int base = (f * uM + m) * uD;
  for (int i = 0; i < uD; ++i)
    s += texelFetch(uBook, at(base + i), 0).r * texelFetch(uProbe, ivec2(i, f), 0).r;
  o = s;
}
"""

# Tiled argmax over the entries of each factor's row. T4 makes the tiling exact.
FS_ARGMAX = """
#version 330 core
uniform sampler2D uScore; uniform int uM;
out vec2 o;
void main(){
  int f = int(gl_FragCoord.x);
  float best = -1e30; int arg = 0;
  for (int m = 0; m < uM; ++m) {
    float v = texelFetch(uScore, ivec2(m, f), 0).r;
    if (v > best) { best = v; arg = m; }
  }
  o = vec2(best, float(arg));
}
"""

# SOFT PROJECTION, the published resonator step: est[f] = sign( sum_m score[f][m] * book[f][m] ).
# No argmax during iteration -- selecting a single codebook entry is omega=1 with no damping, and
# it limit-cycles immediately (measured: 1/30 recovery, GPU and CPU agreeing exactly, which is how
# we know the scheme and not the shader was at fault).
FS_GATHER = """
#version 330 core
uniform sampler2D uBook, uScore; uniform int uD, uM, uTexW;
out float o;
ivec2 at(int i){ return ivec2(i % uTexW, i / uTexW); }
void main(){
  int i = int(gl_FragCoord.x);
  int f = int(gl_FragCoord.y);
  float acc = 0.0;
  for (int m = 0; m < uM; ++m)
    acc += texelFetch(uScore, ivec2(m, f), 0).r * texelFetch(uBook, at((f * uM + m) * uD + i), 0).r;
  o = (acc >= 0.0) ? 1.0 : -1.0;
}
"""


class Resonator:
    def __init__(self, codebooks, ctx=None, texw=2048):
        self.F = len(codebooks)
        self.M, self.D = codebooks[0].shape
        self.texw = texw
        self.ctx = ctx or moderngl.create_standalone_context(require=330, backend="egl")
        quad = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        mk = lambda fs: (lambda p: (p, self.ctx.vertex_array(p, [(quad, "2f", "p")])))(
            self.ctx.program(vertex_shader=VS, fragment_shader=fs))
        self.pProbe, self.vProbe = mk(FS_PROBE)
        self.pScore, self.vScore = mk(FS_SCORE)
        self.pArg, self.vArg = mk(FS_ARGMAX)
        self.pGath, self.vGath = mk(FS_GATHER)
        flat = np.concatenate([np.asarray(c, dtype="f4").reshape(-1) for c in codebooks])
        rows = max(1, (flat.size + texw - 1) // texw)
        buf = np.zeros(rows * texw, dtype="f4"); buf[:flat.size] = flat
        self.tBook = self.ctx.texture((texw, rows), 1, buf.tobytes(), dtype="f4")
        self.tBook.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.books = [np.asarray(c, dtype=np.float64) for c in codebooks]

    def _tex(self, w, h, comps=1, data=None):
        t = self.ctx.texture((w, h), comps,
                             None if data is None else np.ascontiguousarray(data, dtype="f4").tobytes(),
                             dtype="f4")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return t

    def factor(self, composite, iters=40, restarts=20, seed=0):
        """Run the resonator `restarts` times from different states and keep the run whose
        RECONSTRUCTION matches the composite best -- the verify step, not a vote. A resonator that
        cycles produces a confident wrong answer, so the only safe selector is re-binding the
        estimate and comparing."""
        best, best_score = None, -1e30
        rng = np.random.default_rng(seed)
        for r in range(int(restarts)):
            init = None if r == 0 else rng.choice([-1.0, 1.0], (self.F, self.D))
            cand = self._one(composite, iters, init)
            rec = np.ones(self.D)
            for f in range(self.F):
                rec = rec * self.books[f][cand[f]]
            sc = float(rec @ np.asarray(composite, dtype=np.float64))
            if sc > best_score:
                best, best_score = cand, sc
            if best_score >= self.D - 1e-6:          # exact reconstruction: nothing can beat it
                break
        return best

    def _one(self, composite, iters=40, init=None):
        D, F, M = self.D, self.F, self.M
        comp = self._tex(D, 1, 1, np.asarray(composite, dtype="f4"))
        est = self._tex(D, F, 1, (np.ones((F, D), dtype="f4") if init is None
                                  else np.asarray(init, dtype="f4")))
        probe = self._tex(D, F); score = self._tex(M, F)
        win = self._tex(F, 1, 2); nxt = self._tex(D, F)
        fbP = self.ctx.framebuffer([probe]); fbS = self.ctx.framebuffer([score])
        fbW = self.ctx.framebuffer([win]); fbE = self.ctx.framebuffer([nxt])
        last = None
        for _ in range(int(iters)):
            fbP.use(); self.ctx.viewport = (0, 0, D, F)
            comp.use(0); self.pProbe["uComp"].value = 0
            est.use(1); self.pProbe["uEst"].value = 1
            self.pProbe["uF"].value = F
            if "uD" in self.pProbe: self.pProbe["uD"].value = D
            self.vProbe.render(moderngl.TRIANGLES)

            fbS.use(); self.ctx.viewport = (0, 0, M, F)
            self.tBook.use(0); self.pScore["uBook"].value = 0
            probe.use(1); self.pScore["uProbe"].value = 1
            for k, v in (("uD", D), ("uM", M), ("uTexW", self.texw)):
                self.pScore[k].value = v
            self.vScore.render(moderngl.TRIANGLES)

            fbW.use(); self.ctx.viewport = (0, 0, F, 1)
            score.use(0); self.pArg["uScore"].value = 0
            self.pArg["uM"].value = M
            self.vArg.render(moderngl.TRIANGLES)

            fbE.use(); self.ctx.viewport = (0, 0, D, F)
            self.tBook.use(0); self.pGath["uBook"].value = 0
            score.use(1); self.pGath["uScore"].value = 1
            for k, v in (("uD", D), ("uM", M), ("uTexW", self.texw)):
                if k in self.pGath: self.pGath[k].value = v
            self.vGath.render(moderngl.TRIANGLES)

            cur = np.frombuffer(win.read(), dtype="f4").reshape(F, 2)[:, 1].astype(int)
            est, nxt = nxt, est
            fbE = self.ctx.framebuffer([nxt])
            if last is not None and np.array_equal(cur, last):
                break                                  # fixpoint: the estimates stopped moving
            last = cur
        out = tuple(int(v) for v in last)
        for o in (comp, est, nxt, probe, score, win, fbP, fbS, fbW, fbE):
            try: o.release()
            except Exception: pass
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D, F, M = 512, 3, 16
    print("BACKEND:", moderngl.create_standalone_context(require=330, backend="egl").info["GL_RENDERER"])
    print("\n  D    F  M    trials  GPU exact  CPU (same Jacobi)  chance")
    for D, F, M, n in ((256, 3, 8, 20), (512, 3, 16, 20), (512, 4, 16, 20), (1024, 4, 16, 20)):
        gpu_ok = cpu_ok = 0
        for t in range(n):
            books = [rng.choice([-1.0, 1.0], (M, D)) for _ in range(F)]
            truth = tuple(int(rng.integers(M)) for _ in range(F))
            comp = np.ones(D)
            for f in range(F):
                comp = comp * books[f][truth[f]]
            r = Resonator(books)
            gpu_ok += int(r.factor(comp) == truth)
            # the SAME scheme in NumPy, restarts and all -- a reference that runs a different
            # method is not a reference, it is a second experiment.
            def cpu_once(init):
                est = [np.ones(D) for _ in range(F)] if init is None else [init[f] for f in range(F)]
                prev = None
                for _ in range(40):
                    probes = []
                    for f in range(F):
                        pr = comp.copy()
                        for g in range(F):
                            if g != f: pr = pr * est[g]
                        probes.append(pr)
                    cur = [int(np.argmax(books[f] @ probes[f])) for f in range(F)]
                    nw = [np.sign(books[f].T @ (books[f] @ probes[f])) for f in range(F)]
                    est = [np.where(v == 0, 1.0, v) for v in nw]
                    if prev == cur: break
                    prev = cur
                return tuple(prev)
            crng = np.random.default_rng(0)
            cbest, cscore = None, -1e30
            for r in range(20):
                cand = cpu_once(None if r == 0 else crng.choice([-1.0, 1.0], (F, D)))
                rec = np.ones(D)
                for f in range(F): rec = rec * books[f][cand[f]]
                sc = float(rec @ comp)
                if sc > cscore: cbest, cscore = cand, sc
                if cscore >= D - 1e-6: break
            cpu_ok += int(cbest == truth)
        print("  %-4d %-2d %-4d %-7d %-10s %-18s %.4f"
              % (D, F, M, n, "%d/%d" % (gpu_ok, n), "%d/%d" % (cpu_ok, n), 1.0 / M ** F))
