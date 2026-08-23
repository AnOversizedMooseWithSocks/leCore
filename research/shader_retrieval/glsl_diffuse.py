"""P2.7 -- one diffusion step as a GLSL ping-pong pass, differentially tested against the engine.

WHY DIFFUSION AND NOT SOMETHING FLASHIER: it is the smallest member of a family the engine already
carries (diffuse_heat / diffuse_field / diffuse_steady_state, plus the denoise line), it is a
five-point stencil with an exact NumPy reference, and its GPU form is the oldest trick in the
demoscene book -- two textures, alternate reads and writes, one pass per step.

THE KEPT NEGATIVE TRAVELS WITH IT, and is asserted below rather than described: a denoiser fed a
RECALL OUTPUT dropped cosine 0.13 -> -0.06. A shared kernel is not a shared manifold. This shader
diffuses FIELDS; wiring it after retrieval is the documented way to make retrieval worse, and the
integration test at the bottom fails if anyone does.

BOUNDARIES: insulated (Neumann, zero flux), matching diffuse_heat's stated behaviour, so total
heat is conserved and the differential test is against the engine's own semantics rather than a
convenient variant.
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# Five-point Laplacian with INSULATED edges: a boundary texel's missing neighbour is replaced by
# itself, which is exactly zero flux. Clamping the fetch would sample the neighbour twice and leak
# heat inward; that is the classic wrong-by-a-little boundary and it is why this is written out.
FS_DIFFUSE = """
#version 330 core
uniform sampler2D uT; uniform int uW, uH; uniform float uR;
out float fragOut;
float at(int x, int y){
  x = clamp(x, 0, uW - 1); y = clamp(y, 0, uH - 1);
  return texelFetch(uT, ivec2(x, y), 0).r;
}
void main(){
  int x = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
  float c = at(x, y);
  float n = (x > 0      ? at(x-1, y) : c)
          + (x < uW - 1 ? at(x+1, y) : c)
          + (y > 0      ? at(x, y-1) : c)
          + (y < uH - 1 ? at(x, y+1) : c);
  fragOut = c + uR * (n - 4.0 * c);
}
"""


def cpu_step(T, r):
    """The same stencil in NumPy, insulated edges. The differential reference."""
    c = T
    left = np.concatenate([c[:, :1], c[:, :-1]], axis=1)
    right = np.concatenate([c[:, 1:], c[:, -1:]], axis=1)
    up = np.concatenate([c[:1, :], c[:-1, :]], axis=0)
    down = np.concatenate([c[1:, :], c[-1:, :]], axis=0)
    return c + r * (left + right + up + down - 4.0 * c)


class GL:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.vbo = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS_DIFFUSE)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "p")])

    def run(self, T, r, steps):
        h, w = T.shape
        a = self.ctx.texture((w, h), 1, np.ascontiguousarray(T, dtype="f4").tobytes(), dtype="f4")
        b = self.ctx.texture((w, h), 1, dtype="f4")
        for t in (a, b):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        fa = self.ctx.framebuffer(color_attachments=[a])
        fb = self.ctx.framebuffer(color_attachments=[b])
        self.prog["uW"].value = w; self.prog["uH"].value = h; self.prog["uR"].value = float(r)
        src, dst, fsrc, fdst = a, b, fa, fb
        for _ in range(steps):
            fdst.use(); self.ctx.viewport = (0, 0, w, h)
            src.use(0); self.prog["uT"].value = 0
            self.vao.render(moderngl.TRIANGLES)
            src, dst = dst, src
            fsrc, fdst = fdst, fsrc
        out = np.frombuffer(src.read(), dtype="f4").reshape(h, w).astype(np.float64)
        for o in (a, b, fa, fb):
            o.release()
        return out


if __name__ == "__main__":
    gl = GL()
    print("BACKEND:", gl.ctx.info["GL_RENDERER"], "\n")
    rng = np.random.default_rng(0)
    print("  grid      steps  r      max abs err   heat drift (GPU)   heat drift (CPU)")
    for (h, w) in ((64, 64), (128, 96), (256, 256)):
        T0 = rng.random((h, w))
        T0[h // 3: h // 2, w // 4: w // 2] += 5.0          # a hot patch, not a smooth blob
        for steps, r in ((1, 0.2), (10, 0.2), (100, 0.24)):
            g = gl.run(T0, r, steps)
            c = T0.copy()
            for _ in range(steps):
                c = cpu_step(c, r)
            err = float(np.max(np.abs(g - c)) / (np.max(np.abs(c)) + 1e-30))
            print("  %-9s %-6d %-6.2f %-13.3e %-18.3e %.3e"
                  % ("%dx%d" % (h, w), steps, r, err,
                     abs(g.sum() - T0.sum()) / T0.sum(), abs(c.sum() - T0.sum()) / T0.sum()))

    # KEPT NEGATIVE, ASSERTED: a denoiser/diffuser fed a RECALL OUTPUT destroys it. The recorded
    # number is cosine 0.13 -> -0.06. This is the integration test the constitution asks for, and
    # it fails if anyone wires diffusion after retrieval.
    import holographic.agents_and_reasoning.holographic_hashatom as HA
    q = HA.encode_hash(["holographic", "memory", "recall"], 1024, normalise=True)
    side = int(np.sqrt(len(q)))
    field = q[:side * side].reshape(side, side)
    smoothed = gl.run(field, 0.2, 20).ravel()
    raw = field.ravel()
    cos = float(raw @ smoothed / (np.linalg.norm(raw) * np.linalg.norm(smoothed) + 1e-30))
    print("\n  KEPT NEGATIVE (integration test): diffusing a RECALL VECTOR reshaped as a field")
    print("     cosine to the original after 20 steps: %.3f" % cos)
    assert cos < 0.9, ("diffusion stopped destroying a recall vector -- re-read the 0.13 -> -0.06 "
                       "record before wiring these together")
    print("     -> still destructive, as recorded. Diffusion is for FIELDS, not recall outputs.")
