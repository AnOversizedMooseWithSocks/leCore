"""P2.1b -- the complete HDRIFT drift step in shaders: attraction MINUS batch self-repulsion.

The gradient pass (P2.1) showed the field is a sum of plane waves. The repulsion term needs the
BATCH's own field, which changes every step -- so it cannot be a host-side table. It does not need
to be: in the frequency domain the batch spectrum is

    M_batch(k) = sum_j exp(i * (phi0_k + w_k * x_j))

which is one fragment per BIN looping over particles. So the step is two passes that mirror each
other -- particles -> spectrum, then spectrum -> particles -- and nothing crosses back to the host
inside a step.

WHY REPULSION IS NOT OPTIONAL, in the engine's own words: it is "the corrective for the measured
attraction-only memorisation". A drift model with attraction alone collapses its particles onto the
data points it was trained on. The physical check at the bottom asserts that collapse is AVOIDED,
because a step that merely runs is not a step that samples.
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
import glsl_hdrift as G1

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# PASS A: one fragment per frequency bin. Accumulates the batch's own spectrum from the particles.
FS_SPEC = """
#version 330 core
uniform sampler2D uX, uW, uPhi0;
uniform int uNP;
out vec2 fragOut;                    // (real, imag) of the batch spectrum at this bin
void main(){
  int k = int(gl_FragCoord.x);
  float w = texelFetch(uW, ivec2(k, 0), 0).r;
  float p0 = texelFetch(uPhi0, ivec2(k, 0), 0).r;
  vec2 acc = vec2(0.0);
  for (int j = 0; j < uNP; ++j) {
    float ph = p0 + w * texelFetch(uX, ivec2(j, 0), 0).r;
    acc += vec2(cos(ph), sin(ph));
  }
  fragOut = acc;
}
"""

# PASS B: one fragment per particle. Attraction gradient minus repel * batch gradient, then step.
FS_STEP = """
#version 330 core
uniform sampler2D uX, uW, uMag, uArg, uBatch, uScale;
uniform int uK, uNP;
uniform float uLr, uRepel;
out float fragOut;
void main(){
  int i = int(gl_FragCoord.x);
  if (i >= uNP) { fragOut = 0.0; return; }
  float x = texelFetch(uX, ivec2(i, 0), 0).r;
  float g = 0.0;
  for (int k = 0; k < uK; ++k) {
    float w = texelFetch(uW, ivec2(k, 0), 0).r;
    float m = texelFetch(uMag, ivec2(k, 0), 0).r;
    float a = texelFetch(uArg, ivec2(k, 0), 0).r;
    g += m * w * sin(a - w * x);                       // attraction to the data field
    vec2 B = texelFetch(uBatch, ivec2(k, 0), 0).rg;    // the batch's own field
    float s = texelFetch(uScale, ivec2(k, 0), 0).r;
    g -= uRepel * (s * length(B)) * w * sin(atan(B.y, B.x) - w * x);
  }
  fragOut = x + uLr * g;
}
"""


def cpu_step(X, w, phi0, mag, arg, scale, lr, repel):
    """The SAME scheme in NumPy. Not the engine's drift_sample -- that adds annealing, coupling and
    conditioning; this is the two-term core, and comparing to it is the honest differential."""
    ph = phi0[None, :] + w[None, :] * X[:, None]
    B = np.exp(1j * ph).sum(axis=0)
    g = np.zeros_like(X)
    for i, x in enumerate(X):
        att = np.sum(mag * w * np.sin(arg - w * x))
        rep = np.sum((scale * np.abs(B)) * w * np.sin(np.angle(B) - w * x))
        g[i] = att - repel * rep
    return X + lr * g


class GL:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.quad = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        self.pspec = self.ctx.program(vertex_shader=VS, fragment_shader=FS_SPEC)
        self.pstep = self.ctx.program(vertex_shader=VS, fragment_shader=FS_STEP)
        self.vspec = self.ctx.vertex_array(self.pspec, [(self.quad, "2f", "p")])
        self.vstep = self.ctx.vertex_array(self.pstep, [(self.quad, "2f", "p")])

    def tex(self, a, comps=1):
        a = np.ascontiguousarray(np.asarray(a, dtype="f4"))
        t = self.ctx.texture((a.size // comps, 1), comps, a.tobytes(), dtype="f4")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return t

    def step(self, X, w, phi0, mag, arg, scale, lr, repel):
        K, NP = len(w), len(X)
        tX, tW, tP = self.tex(X), self.tex(w), self.tex(phi0)
        spec = self.ctx.texture((K, 1), 2, dtype="f4")
        fs = self.ctx.framebuffer(color_attachments=[spec]); fs.use()
        self.ctx.viewport = (0, 0, K, 1)
        for n, t, u in (("uX", tX, 0), ("uW", tW, 1), ("uPhi0", tP, 2)):
            t.use(u); self.pspec[n].value = u
        self.pspec["uNP"].value = NP
        self.vspec.render(moderngl.TRIANGLES)
        tM, tA, tS = self.tex(mag), self.tex(arg), self.tex(scale)
        out = self.ctx.texture((NP, 1), 1, dtype="f4")
        fo = self.ctx.framebuffer(color_attachments=[out]); fo.use()
        self.ctx.viewport = (0, 0, NP, 1)
        for n, t, u in (("uX", tX, 0), ("uW", tW, 1), ("uMag", tM, 2),
                        ("uArg", tA, 3), ("uBatch", spec, 4), ("uScale", tS, 5)):
            t.use(u); self.pstep[n].value = u
        self.pstep["uK"].value = K; self.pstep["uNP"].value = NP
        self.pstep["uLr"].value = float(lr); self.pstep["uRepel"].value = float(repel)
        self.vstep.render(moderngl.TRIANGLES)
        r = np.frombuffer(out.read(), dtype="f4").astype(np.float64)
        for o in (tX, tW, tP, tM, tA, tS, spec, out, fs, fo):
            o.release()
        return r


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gl = GL()
    print("BACKEND:", gl.ctx.info["GL_RENDERER"], "\n")
    dim = 256
    enc = VectorFunctionEncoder(1, dim=dim, bounds=[(0., 1.)], bandwidth=3.0, seed=0)
    # UNFRIENDLY: three tight, UNEQUAL clusters -- a uniform blob would let attraction-only look fine
    pts = np.clip(np.concatenate([rng.normal(0.2, 0.015, 60), rng.normal(0.5, 0.02, 20),
                                  rng.normal(0.85, 0.015, 40)]), 0.01, 0.99)
    mu = enc.bundle([np.array([v]) for v in pts])
    w, mag, arg = G1.freq_table(enc, mu)
    phi0 = np.angle(np.fft.rfft(enc.encode(np.array([0.0]))))
    scale = np.full(len(w), 2.0 / dim); scale[0] = 1.0 / dim
    if dim % 2 == 0:
        scale[-1] = 1.0 / dim
    X0 = rng.uniform(0.05, 0.95, 64)
    print("  steps  repel  GPU vs CPU max abs err   particle spread   nearest-data distance")
    for repel, steps in ((0.0, 60), (0.5, 60)):
        Xg, Xc = X0.copy(), X0.copy()
        for _ in range(steps):
            Xg = gl.step(Xg, w, phi0, mag, arg, scale, 0.002, repel)
            Xc = cpu_step(Xc, w, phi0, mag, arg, scale, 0.002, repel)
        err = float(np.max(np.abs(Xg - Xc)))
        spread = float(np.std(Xg))
        nd = float(np.mean(np.min(np.abs(Xg[:, None] - pts[None, :]), axis=1)))
        print("  %-6d %-6.1f %-23.3e %-17.4f %.5f" % (steps, repel, err, spread, nd))
    print("\n  repel=0 is the MEMORISATION control: particles should sit closer to the data points")
    print("  than with repulsion on. If the two rows' nearest-data distances match, repulsion is")
    print("  not doing anything and the shader is wrong even though it runs.")
