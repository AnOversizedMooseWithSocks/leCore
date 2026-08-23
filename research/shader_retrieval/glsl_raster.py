"""P2.4 -- the installed image-formation chain as a fragment shader: a linear map, run per pixel.

WHY THIS IS THE FRAGMENT STAGE'S OWN JOB. raster_program_pgm runs an INSTALLED chain -- scene
params in, pixels out -- and the engine already certifies linear formation models (splatting,
basis lighting) as RECTANGULAR. A rectangular map from L light intensities to W*H pixels is
exactly `pixel = dot(basis_row, params)`, which is one fragment per pixel with a loop over L.
Nothing is translated here; the shader form is the same arithmetic the certificate describes.

THE QUANTISATION BOUNDARY IS THE INTERESTING PART, and the faculty already names it: "pixels
clipped to [0,255] ints at emit (quantization is the SERIALIZER's job, stated -- the chain stays
float and certified)". So there are TWO contracts to check, not one:
  * the FLOAT chain, where f32-vs-f64 differences are expected and bounded
  * the QUANTISED output, where they must vanish -- or the picture differs by a pixel value, and
    "byte-exactness vs the live path is a testable contract" stops being true.
A shader that matched in float and disagreed after rounding would pass a naive test and ship a
different image.
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# One fragment per pixel; the basis is a W*H x L texture, params a length-L texture.
FS_FORM = """
#version 330 core
uniform sampler2D uBasis;    // L x (W*H): row = pixel, column = light
uniform sampler2D uParams;   // L x 1
uniform int uL, uW;
out float fragOut;
void main(){
  int px = int(gl_FragCoord.x) + int(gl_FragCoord.y) * uW;
  float s = 0.0;
  for (int l = 0; l < uL; ++l)
    s += texelFetch(uBasis, ivec2(l, px), 0).r * texelFetch(uParams, ivec2(l, 0), 0).r;
  fragOut = s;
}
"""


def basis_for(w, h, lights, seed=0):
    """A REAL formation basis, not a random matrix: each light contributes an inverse-square
    falloff lobe from its own position, so rows are smooth and highly correlated -- which is the
    unfriendly case for a dot product in f32, because neighbouring pixels differ in the last bits."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w]
    P = np.stack([xs.ravel() / max(w - 1, 1), ys.ravel() / max(h - 1, 1)], axis=1)
    B = np.zeros((w * h, lights))
    for l in range(lights):
        c = rng.uniform(-0.2, 1.2, 2)
        d2 = ((P - c) ** 2).sum(axis=1) + 0.05
        B[:, l] = 1.0 / d2
    return B / B.max() * 200.0


if __name__ == "__main__":
    ctx = moderngl.create_standalone_context(require=330, backend="egl")
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS_FORM)
    vao = ctx.vertex_array(prog, [(quad, "2f", "p")])
    print("BACKEND:", ctx.info["GL_RENDERER"], "\n")
    print("  image     lights  float max abs err   QUANTISED pixels differing   worst |delta|")
    rng = np.random.default_rng(0)
    for (w, h) in ((8, 8), (32, 24), (64, 64)):
        for L in (3, 16, 64):
            B = basis_for(w, h, L)
            params = rng.uniform(0.0, 1.0, L)
            ref = B @ params
            tb = ctx.texture((L, w * h), 1, np.ascontiguousarray(B, dtype="f4").tobytes(), dtype="f4")
            tp = ctx.texture((L, 1), 1, np.ascontiguousarray(params, dtype="f4").tobytes(), dtype="f4")
            for t in (tb, tp):
                t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            out = ctx.texture((w, h), 1, dtype="f4")
            fbo = ctx.framebuffer(color_attachments=[out]); fbo.use()
            ctx.viewport = (0, 0, w, h)
            tb.use(0); prog["uBasis"].value = 0
            tp.use(1); prog["uParams"].value = 1
            prog["uL"].value = L; prog["uW"].value = w
            vao.render(moderngl.TRIANGLES)
            got = np.frombuffer(out.read(), dtype="f4").reshape(h * w).astype(np.float64)
            ferr = float(np.max(np.abs(got - ref)))
            # THE CONTRACT THAT MATTERS: the SERIALISED image, quantised exactly as the faculty does
            qg = np.clip(np.round(got), 0, 255).astype(int)
            qr = np.clip(np.round(ref), 0, 255).astype(int)
            ndiff = int((qg != qr).sum())
            worst = int(np.max(np.abs(qg - qr))) if ndiff else 0
            print("  %-9s %-7d %-18.3e %-28s %d"
                  % ("%dx%d" % (w, h), L, ferr, "%d of %d" % (ndiff, w * h), worst))
            for o in (tb, tp, out, fbo):
                o.release()
