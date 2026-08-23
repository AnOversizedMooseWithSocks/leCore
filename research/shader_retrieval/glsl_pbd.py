"""P2.2 -- a Position-Based Dynamics step in GLSL, and it needs BOTH shader tricks at once.

WHY IT IS THE INTERESTING ONE. Diffusion needed ping-pong. Scatter BM25 needed additive blending.
PBD needs both: each distance constraint touches TWO particles, so applying corrections is a
scatter-add, and the solver iterates, so the state ping-pongs. This is the first faculty here that
composes the two, which is the real test of whether the pieces fit.

JACOBI, NOT GAUSS-SEIDEL, AND THAT IS A REAL CHOICE WITH A COST. The engine's CPU projector sweeps
constraints in order -- each sees the previous one's correction (Gauss-Seidel), which converges
faster. A GPU cannot do that: all constraints run at once and see the SAME input state (Jacobi).
So the shader is NOT bit-comparable to a sequential sweep, and pretending otherwise would be the
mistake. What IS comparable is Jacobi-on-GPU against Jacobi-on-CPU, plus the physical invariant
both must satisfy -- constraint residual falling monotonically. Both are measured below.

Macklin's own position-based-fluids work uses Jacobi for exactly this reason, and compensates with
more iterations; the measurement below shows what that costs here.
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

VS_SCATTER = """
#version 330 core
uniform sampler2D uX;          // positions, RG32F, one texel per particle
uniform usampler2D uEdge;      // constraint endpoints, 2 texels per constraint
uniform sampler2D uRest;       // rest length per constraint
uniform int uNP, uW;
out vec3 vDelta;               // .xy = position correction, .z = 1 (the count, for averaging)
void main(){
  int c = gl_VertexID / 2;          // two vertices per constraint: one per endpoint
  int side = gl_VertexID - c * 2;
  int ia = int(texelFetch(uEdge, ivec2(2 * c, 0), 0).r);
  int ib = int(texelFetch(uEdge, ivec2(2 * c + 1, 0), 0).r);
  vec2 a = texelFetch(uX, ivec2(ia % uW, ia / uW), 0).rg;
  vec2 b = texelFetch(uX, ivec2(ib % uW, ib / uW), 0).rg;
  float rest = texelFetch(uRest, ivec2(c, 0), 0).r;
  vec2 d = b - a;
  float L = length(d);
  vec2 corr = (L > 1e-8) ? (0.5 * (L - rest) / L) * d : vec2(0.0);
  int me = (side == 0) ? ia : ib;
  vDelta = vec3((side == 0) ? corr : -corr, 1.0);
  float x = (float(me % uW) + 0.5) / float(uW) * 2.0 - 1.0;
  float y = (float(me / uW) + 0.5) / float((uNP + uW - 1) / uW) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_PointSize = 1.0;
}
"""

FS_SCATTER = """
#version 330 core
in vec3 vDelta; out vec3 fragOut;
void main(){ fragOut = vDelta; }
"""

# Apply the averaged correction. Averaging by the accumulated COUNT is what makes Jacobi stable:
# a particle in ten constraints would otherwise move ten times as far as one in a single
# constraint and the solver would explode.
VS_QUAD = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"
FS_APPLY = """
#version 330 core
uniform sampler2D uX, uAcc;
out vec2 fragOut;
void main(){
  ivec2 t = ivec2(gl_FragCoord.xy);
  vec2 x = texelFetch(uX, t, 0).rg;
  vec3 a = texelFetch(uAcc, t, 0).rgb;
  fragOut = x + ((a.z > 0.0) ? a.xy / a.z : vec2(0.0));
}
"""


def cpu_jacobi(X, edges, rest, iters):
    """The SAME Jacobi scheme in NumPy: all constraints read the same state, corrections averaged."""
    X = X.copy()
    for _ in range(iters):
        acc = np.zeros_like(X)
        cnt = np.zeros(len(X))
        a, b = X[edges[:, 0]], X[edges[:, 1]]
        d = b - a
        L = np.linalg.norm(d, axis=1)
        safe = L > 1e-8
        corr = np.zeros_like(d)
        corr[safe] = (0.5 * (L[safe] - rest[safe]) / L[safe])[:, None] * d[safe]
        np.add.at(acc, edges[:, 0], corr)
        np.add.at(acc, edges[:, 1], -corr)
        np.add.at(cnt, edges[:, 0], 1.0)
        np.add.at(cnt, edges[:, 1], 1.0)
        nz = cnt > 0
        X[nz] += acc[nz] / cnt[nz][:, None]
    return X


def residual(X, edges, rest):
    d = X[edges[:, 1]] - X[edges[:, 0]]
    return float(np.sqrt(np.mean((np.linalg.norm(d, axis=1) - rest) ** 2)))


class GL:
    def __init__(self, W=64):
        self.W = W
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.pscatter = self.ctx.program(vertex_shader=VS_SCATTER, fragment_shader=FS_SCATTER)
        self.papply = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_APPLY)
        self.quad = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        self.vscatter = self.ctx.vertex_array(self.pscatter, [])
        self.vapply = self.ctx.vertex_array(self.papply, [(self.quad, "2f", "p")])

    def run(self, X, edges, rest, iters):
        NP = len(X); W = self.W; H = (NP + W - 1) // W
        buf = np.zeros((H, W, 2), dtype="f4"); buf.reshape(-1, 2)[:NP] = X
        xa = self.ctx.texture((W, H), 2, buf.tobytes(), dtype="f4")
        xb = self.ctx.texture((W, H), 2, dtype="f4")
        acc = self.ctx.texture((W, H), 3, dtype="f4")
        for t in (xa, xb, acc):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        e = np.zeros(2 * len(edges), dtype="<u4"); e[0::2] = edges[:, 0]; e[1::2] = edges[:, 1]
        te = self.ctx.texture((len(e), 1), 1, e.tobytes(), dtype="u4")
        tr = self.ctx.texture((len(rest), 1), 1, np.asarray(rest, dtype="f4").tobytes(), dtype="f4")
        facc = self.ctx.framebuffer(color_attachments=[acc])
        fa = self.ctx.framebuffer(color_attachments=[xa])
        fb = self.ctx.framebuffer(color_attachments=[xb])
        src, dst, fdst = xa, xb, fb
        for _ in range(iters):
            facc.use(); self.ctx.viewport = (0, 0, W, H)
            facc.clear(0.0, 0.0, 0.0, 0.0)
            self.ctx.enable(moderngl.BLEND); self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            src.use(0); self.pscatter["uX"].value = 0
            te.use(1); self.pscatter["uEdge"].value = 1
            tr.use(2); self.pscatter["uRest"].value = 2
            self.pscatter["uNP"].value = NP; self.pscatter["uW"].value = W
            self.vscatter.render(moderngl.POINTS, vertices=2 * len(edges))
            self.ctx.disable(moderngl.BLEND)
            fdst.use(); self.ctx.viewport = (0, 0, W, H)
            src.use(0); self.papply["uX"].value = 0
            acc.use(1); self.papply["uAcc"].value = 1
            self.vapply.render(moderngl.TRIANGLES)
            src, dst = dst, src
            fdst = fa if fdst is fb else fb
        out = np.frombuffer(src.read(), dtype="f4").reshape(-1, 2)[:NP].astype(np.float64)
        for o in (xa, xb, acc, te, tr, facc, fa, fb):
            o.release()
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gl = GL()
    print("BACKEND:", gl.ctx.info["GL_RENDERER"], "\n")
    print("  cloth     particles  constraints  iters  GPUvsCPU rel err   residual: start -> GPU -> CPU")
    for n in (16, 32, 48):
        gx, gy = np.meshgrid(np.arange(n), np.arange(n))
        X0 = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
        X0 += 0.35 * rng.standard_normal(X0.shape)          # a disturbed cloth, not a perfect grid
        edges = []
        for y in range(n):
            for x in range(n):
                i = y * n + x
                if x + 1 < n: edges.append((i, i + 1))
                if y + 1 < n: edges.append((i, i + n))
        edges = np.array(edges, dtype=np.int64)
        rest = np.ones(len(edges))
        for iters in (10, 60):
            g = gl.run(X0, edges, rest, iters)
            c = cpu_jacobi(X0, edges, rest, iters)
            err = float(np.max(np.abs(g - c)) / (np.max(np.abs(c)) + 1e-30))
            print("  %-9s %-10d %-12d %-6d %-18.3e %.4f -> %.4f -> %.4f"
                  % ("%dx%d" % (n, n), len(X0), len(edges), iters, err,
                     residual(X0, edges, rest), residual(g, edges, rest),
                     residual(c, edges, rest)))
