"""RUN leCore's VSA read path AS ACTUAL GLSL, on a real GL context, and check the numbers.

Everything before this was emission and proof. The standing kept negative was blunt: "the
emitted shader is NOT executed by any test here -- there is no GPU and no browser." This closes
that for GLSL. Mesa llvmpipe gives a software GL 4.5 core context headlessly (surfaceless EGL),
so the shaders below are COMPILED BY A REAL GLSL COMPILER AND EXECUTED.

DELIBERATE CONSTRAINT: fragment shaders only, texelFetch only, no compute shaders and no SSBOs --
because the target is WebGL2 (GLSL ES 3.00), which has neither. Running under desktop GLSL 330
core is the closest honest proxy available here; the DIALECT DIFFERENCE IS DECLARED, not hidden.

Three passes, which is the whole algebra the browser tier needs:
  1. BIND      -- circular convolution as a circulant gather: k[(j-i) mod D]. The `mod` on the
                  index domain is domain repetition; the matrix is never materialised.
  2. SCORE     -- the codebook matvec, one fragment per atom.
  3. ARGMAX    -- a tiled max reduction, which T4 proves equals the single-pass max.
"""
import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, unitary_vector

VERT = """
#version 330 core
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# PASS 1 -- bind(x, k) as a circulant gather. One fragment per output component.
FRAG_BIND = """
#version 330 core
uniform sampler2D uX;      // D x 1
uniform sampler2D uK;      // D x 1
uniform int uD;
out float fragOut;
void main() {
    int j = int(gl_FragCoord.x);
    float acc = 0.0;
    for (int i = 0; i < uD; ++i) {
        int idx = j - i;
        if (idx < 0) idx += uD;             // DOMAIN REPETITION: the circulant index wrap
        acc += texelFetch(uK, ivec2(idx, 0), 0).r * texelFetch(uX, ivec2(i, 0), 0).r;
    }
    fragOut = acc;
}
"""

# PASS 2 -- codebook scores. One fragment per atom; the dot product is the fragment's loop.
FRAG_SCORE = """
#version 330 core
uniform sampler2D uV;      // D x K codebook (x = component, y = atom)
uniform sampler2D uQ;      // D x 1 probe
uniform int uD;
out float fragOut;
void main() {
    int row = int(gl_FragCoord.x);
    float acc = 0.0;
    for (int i = 0; i < uD; ++i) {
        acc += texelFetch(uV, ivec2(i, row), 0).r * texelFetch(uQ, ivec2(i, 0), 0).r;
    }
    fragOut = acc;
}
"""

# PASS 3 -- tiled max reduction. T4 (tiled_max_eq_global) says regrouping cannot move the answer.
FRAG_MAXRED = """
#version 330 core
uniform sampler2D uS;
uniform int uN;            // valid entries in the input row
uniform int uTile;
out float fragOut;
void main() {
    int t = int(gl_FragCoord.x);
    float best = -1e30;
    for (int i = 0; i < uTile; ++i) {
        int j = t * uTile + i;
        if (j < uN) best = max(best, texelFetch(uS, ivec2(j, 0), 0).r);
    }
    fragOut = best;
}
"""

class GL:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.progs = {}

    def prog(self, frag):
        if frag not in self.progs:
            p = self.ctx.program(vertex_shader=VERT, fragment_shader=frag)
            self.progs[frag] = (p, self.ctx.vertex_array(p, [(self.vbo, "2f", "in_pos")]))
        return self.progs[frag]

    def tex(self, arr):
        a = np.ascontiguousarray(np.atleast_2d(arr).astype("f4"))
        return self.ctx.texture((a.shape[1], a.shape[0]), 1, a.tobytes(), dtype="f4")

    def run(self, frag, out_w, textures, ints):
        p, vao = self.prog(frag)
        out = self.ctx.texture((out_w, 1), 1, dtype="f4")
        fbo = self.ctx.framebuffer(color_attachments=[out])
        fbo.use(); self.ctx.viewport = (0, 0, out_w, 1)
        for unit, (name, t) in enumerate(textures.items()):
            t.use(unit); p[name].value = unit
        for k, v in ints.items():
            p[k].value = int(v)
        vao.render(moderngl.TRIANGLES)
        res = np.frombuffer(out.read(), dtype="f4").copy()
        fbo.release(); out.release()
        return res

if __name__ == "__main__":
    g = GL()
    print("GL:", g.ctx.info["GL_RENDERER"], "|", g.ctx.info["GL_VERSION"])
    print("    fragment-shader only, texelFetch only -- the WebGL2 subset.\n")

    for D in (256, 512):
        rng = np.random.default_rng(0)
        K = 64
        k = unitary_vector(D, rng)
        V = np.stack([unitary_vector(D, rng) for _ in range(K)])
        target = 7
        x = V[target]

        # --- PASS 1: bind on the GPU vs leCore's f64 rFFT bind
        tX, tK = g.tex(x), g.tex(k)
        gpu_bind = g.run(FRAG_BIND, D, {"uX": tX, "uK": tK}, {"uD": D}).astype(np.float64)
        ref_bind = bind(x, k)
        e_bind = np.max(np.abs(gpu_bind - ref_bind)) / np.max(np.abs(ref_bind))

        # --- unbind on the GPU is the same shader with the involuted key
        ki = np.empty_like(k); ki[0] = k[0]; ki[1:] = k[1:][::-1]
        tB, tKi = g.tex(gpu_bind), g.tex(ki)
        gpu_probe = g.run(FRAG_BIND, D, {"uX": tB, "uK": tKi}, {"uD": D}).astype(np.float64)
        ref_probe = unbind(ref_bind, k)
        e_unbind = np.max(np.abs(gpu_probe - ref_probe)) / np.max(np.abs(ref_probe))

        # --- PASS 2: scores
        tV, tQ = g.tex(V), g.tex(gpu_probe)
        gpu_s = g.run(FRAG_SCORE, K, {"uV": tV, "uQ": tQ}, {"uD": D}).astype(np.float64)
        ref_s = V @ ref_probe
        e_score = np.max(np.abs(gpu_s - ref_s))

        # --- PASS 3: tiled max reduction (T4), then the decision
        tS = g.tex(gpu_s)
        tile = 8
        red = g.run(FRAG_MAXRED, (K + tile - 1) // tile, {"uS": tS}, {"uN": K, "uTile": tile})
        gpu_max, ref_max = float(np.max(red)), float(np.max(ref_s))

        margin = float(np.sort(ref_s)[::-1][0] - np.sort(ref_s)[::-1][1])
        eps = float(np.max(np.abs(gpu_s - ref_s)))
        print("D=%-5d K=%d" % (D, K))
        print("   bind   GPU vs f64 rFFT : max rel err %.3e" % e_bind)
        print("   unbind GPU vs f64      : max rel err %.3e" % e_unbind)
        print("   scores GPU vs f64      : max abs err %.3e" % e_score)
        print("   tiled max == single-pass max (T4): %s   (%.6f vs %.6f)"
              % (abs(gpu_max - ref_max) < 1e-5, gpu_max, ref_max))
        print("   ARGMAX gpu=%d  f64=%d  target=%d  -> %s"
              % (int(np.argmax(gpu_s)), int(np.argmax(ref_s)), target,
                 "DECISION MATCHES" if np.argmax(gpu_s) == np.argmax(ref_s) else "FLIPPED"))
        print("   T1 gate: margin %.6f vs 2*eps %.3e -> safety %.1f, gate %s\n"
              % (margin, 2 * eps, margin / max(2 * eps, 1e-30),
                 "ANSWERS" if margin > 2 * eps else "ABSTAINS"))
