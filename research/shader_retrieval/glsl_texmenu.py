"""P2.6 -- the procedural texture menu through the emitter, into GLSL, EXECUTED, and compared.

THE INTERESTING OUTPUT IS THE REFUSAL LIST. `emit` accepts a narrow language on purpose: annotated
float parameters, straight-line assignments, bounded literal-count loops, one return, and only
intrinsics it knows. Most of the texture menu will NOT fit, and WHICH ONES and WHY is the useful
measurement -- it says exactly where the emitter's language stops for real workloads instead of
for a toy.

This also closes a loop opened earlier in this arc: the `glsl` dialect was a PINNED NEGATIVE that
was flipped deliberately, and its only execution evidence so far was hand-written shaders. Here
the emitter GENERATES the GLSL, a fragment shader RUNS it, and the output is compared to the same
function evaluated in Python.
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

from holographic.io_and_interop.holographic_emit import emit_source, EmitError

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# Kernels written in the emitter's language on purpose. Each is a real per-pixel texture rule, not
# a toy: they are the arithmetic cores of the menu's wave / marble / rings / checker entries.
KERNELS = {
    "stripes":  ("def f(x: float, y: float) -> float:\n"
                 "    return 0.5 + 0.5 * sin(x * 12.0)\n"),
    "rings":    ("def f(x: float, y: float) -> float:\n"
                 "    r = sqrt(x * x + y * y)\n"
                 "    return 0.5 + 0.5 * sin(r * 24.0)\n"),
    "marble":   ("def f(x: float, y: float) -> float:\n"
                 "    v = sin(x * 6.0 + sin(y * 11.0) * 2.0)\n"
                 "    return 0.5 + 0.5 * v\n"),
    "fbm4":     ("def f(x: float, y: float) -> float:\n"
                 "    s = 0.0\n"
                 "    a = 0.5\n"
                 "    for i in range(4):\n"
                 "        s = s + a * sin(x * 3.0 + i) * cos(y * 3.0 + i)\n"
                 "        a = a * 0.5\n"
                 "    return 0.5 + 0.5 * s\n"),
    "wood":     ("def f(x: float, y: float) -> float:\n"
                 "    r = sqrt(x * x + y * y) * 9.0\n"
                 "    return r - floor(r)\n"),               # floor: now in the table
    "voronoi":  ("def f(x: float, y: float) -> float:\n"
                 "    best = 9.0\n"
                 "    for i in range(cells):\n"              # variable trip count -> refuse
                 "        best = min(best, abs(x - i))\n"
                 "    return best\n"),
    # fmod is DELIBERATELY still refused (C and GLSL disagree on the sign of the remainder), so
    # the checker is written the way a shader author would: floor of a halved sum, doubled back.
    "checker":  ("def f(x: float, y: float) -> float:\n"
                 "    s = floor(x) + floor(y)\n"
                 "    return s - 2.0 * floor(s * 0.5)\n"),
    "checker_fmod": ("def f(x: float, y: float) -> float:\n"
                     "    return fmod(floor(x) + floor(y), 2.0)\n"),   # must STILL refuse
}


def py_eval(src, X, Y):
    ns = {"sin": np.sin, "cos": np.cos, "sqrt": np.sqrt, "abs": np.abs,
          "min": np.minimum, "max": np.maximum, "exp": np.exp, "log": np.log, "pow": np.power,
          "floor": np.floor, "trunc": np.trunc}
    exec(compile(src, "k", "exec"), ns)
    f = ns["f"]
    out = np.zeros_like(X)
    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            out[j, i] = f(float(X[j, i]), float(Y[j, i]))
    return out


if __name__ == "__main__":
    ctx = moderngl.create_standalone_context(require=330, backend="egl")
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
    N = 64
    xs = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(xs, xs)
    print("BACKEND:", ctx.info["GL_RENDERER"], "\n")
    print("  kernel     emit     executed   max abs err vs Python   refusal reason")
    emitted = refused = 0
    for name, src in KERNELS.items():
        try:
            body = emit_source(src, "glsl")
        except EmitError as exc:
            refused += 1
            print("  %-10s REFUSED  %-10s %-23s %s" % (name, "-", "-", str(exc)[:52]))
            continue
        emitted += 1
        fs = ("#version 330 core\nuniform float uN;\nout float o;\n" + body +
              "\nvoid main(){ float x = (gl_FragCoord.x - 0.5) / (uN - 1.0) * 2.0 - 1.0;\n"
              "             float y = (gl_FragCoord.y - 0.5) / (uN - 1.0) * 2.0 - 1.0;\n"
              "             o = f(x, y); }\n")
        prog = ctx.program(vertex_shader=VS, fragment_shader=fs)
        vao = ctx.vertex_array(prog, [(quad, "2f", "p")])
        tex = ctx.texture((N, N), 1, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[tex]); fbo.use()
        ctx.viewport = (0, 0, N, N)
        prog["uN"].value = float(N)
        vao.render(moderngl.TRIANGLES)
        got = np.frombuffer(tex.read(), dtype="f4").reshape(N, N).astype(np.float64)
        ref = py_eval(src, X, Y)
        err = float(np.max(np.abs(got - ref)))
        print("  %-10s ok       %-10s %-23.3e %s" % (name, "yes", err, ""))
        tex.release(); fbo.release()
    print("\n  %d emitted and executed, %d refused by the emitter's language." % (emitted, refused))
    print("  The refusals are the measurement: they name exactly where the language stops.")
