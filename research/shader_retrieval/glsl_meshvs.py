"""P2.3 -- rigid mesh transforms through the VERTEX stage, read back by transform feedback.

WHY THE VERTEX STAGE AND NOT ANOTHER FRAGMENT PASS. Everything shipped so far in this arc used the
fragment stage: diffusion (ping-pong), BM25 (one fragment per document), PBD (blending). Vertex
processing is what a GPU was BUILT for, and `mesh_program_obj` already certifies rigid transforms
as BLOCKDIAG with a byte-exact CPU path -- so this is the one faculty where the shader form is the
NATIVE form rather than a translation. Transform feedback is the honest way to get the result back:
the vertex stage actually runs, and its output is captured rather than re-derived in a fragment.

UNFRIENDLY DATA ON PURPOSE: a real subdivided mesh from the engine (mesh_box + catmull-clark), not
a symmetric lattice, and a CHAIN of transforms rather than one -- error compounds across a chain,
which is the thing worth measuring about a 9+3-parameter-per-step representation.
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

VS_XFORM = """
#version 330 core
in vec3 in_pos;
uniform vec3 uC0, uC1, uC2;   // the rotation's three COLUMNS -- 9 params, no mat3 padding
uniform vec3 uT;              // + 3 -- exactly mesh_program_obj's per-step budget
out vec3 out_pos;
void main(){ out_pos = uC0 * in_pos.x + uC1 * in_pos.y + uC2 * in_pos.z + uT; }
"""


def rigid(seed):
    """A rotation from a random axis-angle, plus a translation. Not axis-aligned, on purpose:
    an axis-aligned rotation has zeros where a real one has round-off."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(3); a /= np.linalg.norm(a)
    th = float(rng.uniform(0.3, 2.5))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    t = rng.uniform(-2, 2, 3)
    return R, t


class GL:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.prog = self.ctx.program(vertex_shader=VS_XFORM, varyings=["out_pos"])

    def chain(self, V, steps):
        buf = self.ctx.buffer(np.ascontiguousarray(V, dtype="f4").tobytes())
        out = self.ctx.buffer(reserve=buf.size)
        for R, t in steps:
            vao = self.ctx.vertex_array(self.prog, [(buf, "3f", "in_pos")])
            for k in range(3):                       # column k of R = R[:, k]
                self.prog["uC%d" % k].value = tuple(float(v) for v in R[:, k])
            self.prog["uT"].value = tuple(float(x) for x in t)
            vao.transform(out, mode=moderngl.POINTS)
            buf, out = out, buf
        return np.frombuffer(buf.read(), dtype="f4").reshape(-1, 3).astype(np.float64)


if __name__ == "__main__":
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    mesh = mind.mesh_box()
    for _ in range(2):                       # subdivide: a real irregular mesh, not 8 corners
        mesh = mind.mesh_catmull_clark(mesh)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = mesh.faces
    gl = GL()
    print("BACKEND:", gl.ctx.info["GL_RENDERER"])
    print("mesh: %d vertices from mesh_box + 2x catmull-clark\n" % len(V))
    print("  chain steps   max abs err   max rel err   rigidity drift (edge lengths)")
    for n in (1, 4, 16, 64):
        steps = [rigid(s) for s in range(n)]
        g = gl.chain(V, steps)
        c = V.copy()
        for R, t in steps:
            c = c @ R.T + t
        err = float(np.max(np.abs(g - c)))
        rel = err / (float(np.max(np.abs(c))) + 1e-30)
        # A rigid chain must PRESERVE edge lengths; drift here is the physical invariant, and it
        # catches a wrong matrix convention that a coordinate diff might not.
        e = F[0] if hasattr(F, "__getitem__") else None
        d0 = np.linalg.norm(V[1:] - V[:-1], axis=1)
        d1 = np.linalg.norm(g[1:] - g[:-1], axis=1)
        drift = float(np.max(np.abs(d1 - d0)))
        print("  %-13d %-13.3e %-13.3e %.3e" % (n, err, rel, drift))
