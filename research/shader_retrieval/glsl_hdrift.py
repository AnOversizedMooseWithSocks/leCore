"""P2.1 -- the HDRIFT field gradient in GLSL. The blocker was "needs the FPE encoder in-shader";
measurement dissolved it.

WHAT UNBLOCKED IT. HDRIFT's expensive inner piece is grad_x <enc(x), mu> -- attraction to the data
field -- and porting `VectorFunctionEncoder` looked like porting a whole subsystem. Probing the
encoder instead of reading it showed the structure: its rfft has CONSTANT UNIT MAGNITUDES and
phases that are LINEAR IN x, measured identical at x=0.2 and x=0.6 to 1e-3. So

    <enc(x), mu> = sum_k |M_k| cos(arg M_k - w_k x)     -- a SUM OF PLANE WAVES
    d/dx         = sum_k |M_k| w_k sin(arg M_k - w_k x)

There is no encoder to port. There is a frequency table (w, |M|, arg M) computed once on the host
and a fragment that sums plane waves -- which is the oldest shader in the demoscene, and the reason
this faculty turned out to be the cheapest of the P2 set rather than the dearest.

FIRST ATTEMPT AT THE RECONSTRUCTION WAS OFF BY A CONSTANT 0.625 at every x: I doubled every rfft
bin except DC, forgetting the NYQUIST bin is also real and must not be doubled. A constant offset
is the signature of a mishandled DC/Nyquist term, and it does not affect the GRADIENT at all --
which is exactly why the gradient is what gets tested here, and the offset is recorded rather than
hidden.
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# One fragment per particle. Loops the frequency table and accumulates the plane-wave gradient.
FS_GRAD = """
#version 330 core
uniform sampler2D uX;      // particle positions, one texel each
uniform sampler2D uW;      // omega per bin
uniform sampler2D uMag;    // |M| per bin
uniform sampler2D uArg;    // arg M per bin
uniform int uK, uNP;
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
    g += m * w * sin(a - w * x);
  }
  fragOut = g;
}
"""


def freq_table(enc, mu, h=1e-4, at=0.2):
    """(w, |M|, arg M) with the DOUBLING that a real signal's rfft needs: interior bins count
    twice, DC and Nyquist once. Getting that wrong shifts the FIELD by a constant and leaves the
    GRADIENT untouched -- so the gradient is the honest thing to test."""
    P = lambda x: np.angle(np.fft.rfft(enc.encode(np.array([x]))))
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    w = wrap(P(at) - P(at - h)) / h
    M = np.fft.rfft(mu)
    n = enc.dim
    scale = np.full(len(M), 2.0 / n)
    scale[0] = 1.0 / n
    if n % 2 == 0:
        scale[-1] = 1.0 / n
    return w, np.abs(M) * scale, np.angle(M)


if __name__ == "__main__":
    ctx = moderngl.create_standalone_context(require=330, backend="egl")
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS_GRAD)
    vao = ctx.vertex_array(prog, [(quad, "2f", "p")])
    print("BACKEND:", ctx.info["GL_RENDERER"], "\n")
    rng = np.random.default_rng(0)
    print("  dim    data pts  particles  field offset (const)  gradient max abs err   rel err")
    for dim in (64, 256, 1024):
        enc = VectorFunctionEncoder(1, dim=dim, bounds=[(0., 1.)], bandwidth=3.0, seed=0)
        # UNFRIENDLY DATA: three tight clusters, not a uniform sample -- a multimodal field has
        # steep gradients between the modes, which is where a plane-wave sum is worst.
        pts = np.concatenate([rng.normal(c, 0.02, 40) for c in (0.2, 0.5, 0.85)])
        pts = np.clip(pts, 0.01, 0.99)
        mu = enc.bundle([np.array([v]) for v in pts])
        w, mag, arg = freq_table(enc, mu)
        X = rng.uniform(0.05, 0.95, 96)
        tex = lambda a: ctx.texture((len(a), 1), 1, np.ascontiguousarray(a, dtype="f4").tobytes(),
                                    dtype="f4")
        tX, tW, tM, tA = tex(X), tex(w), tex(mag), tex(arg)
        for t in (tX, tW, tM, tA):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        out = ctx.texture((len(X), 1), 1, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[out]); fbo.use()
        ctx.viewport = (0, 0, len(X), 1)
        for name, t, u in (("uX", tX, 0), ("uW", tW, 1), ("uMag", tM, 2), ("uArg", tA, 3)):
            t.use(u); prog[name].value = u
        prog["uK"].value = len(w); prog["uNP"].value = len(X)
        vao.render(moderngl.TRIANGLES)
        g = np.frombuffer(out.read(), dtype="f4").astype(np.float64)
        # reference: central difference of the ACTUAL encoder, not of my reconstruction
        hh = 1e-5
        ref = np.array([(enc.encode(np.array([x + hh])) @ mu - enc.encode(np.array([x - hh])) @ mu)
                        / (2 * hh) for x in X])
        # the constant field offset, reported because it is the one thing the gradient hides
        field_direct = np.array([enc.encode(np.array([x])) @ mu for x in X])
        field_pw = np.array([float(np.sum(mag * np.cos(arg - w * x))) for x in X])
        off = float(np.mean(field_pw - field_direct))
        err = float(np.max(np.abs(g - ref)))
        print("  %-6d %-9d %-10d %-22.3e %-22.3e %.3e"
              % (dim, len(pts), len(X), off, err, err / (np.max(np.abs(ref)) + 1e-30)))
        for o in (tX, tW, tM, tA, out, fbo):
            o.release()
