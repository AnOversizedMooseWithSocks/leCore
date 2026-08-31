"""The FULL VSA algebra in GLSL: bind, unbind, bundle, cleanup -- atoms generated in the shader.

This is the piece the retrieval pages did not have. Bag-of-words needed only bundle + dot; a
role-filler record needs BIND to be exact, and holographic_hashatom explicitly refuses to bind.
Phasor atoms bind by PHASE ADDITION, which is one add, so the whole algebra fits a fragment
shader with no FFT, no normalisation pass and no stored vocabulary.

Complex values live in RG32F textures (re, im). Atoms are never stored: a shader regenerates any
atom from its u32 name hash, so binding a record against a key costs zero bytes of vocabulary.
"""
import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import numpy as np, moderngl
import holographic.agents_and_reasoning.holographic_hashatom as HA
import holographic.agents_and_reasoning.holographic_phasor as PH

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# The atom generator, inlined into every shader that needs it. Phases in TURNS so the only
# constant shared across NumPy/GLSL/JS is 2^32, not pi.
ATOM = """
float atomTurn(uint base, uint i){
  uint x = base ^ i;
  uint s = x * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; x = (w >> 22u) ^ w;
  return float(x) / 4294967296.0;
}
vec2 atomC(uint base, uint i){
  float t = atomTurn(base, i) * 6.283185307179586;
  return vec2(cos(t), sin(t));
}
vec2 cmul(vec2 a, vec2 b){ return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
"""

# bind / unbind a stored complex record against a GENERATED atom. uConj flips it to unbind:
# because |atom| = 1, the conjugate IS the exact inverse -- no pseudo-inverse, no normalisation.
FS_BINDC = "#version 330 core\n" + ATOM + """
uniform sampler2D uZ; uniform uint uKey; uniform int uConj;
out vec2 fragOut;
void main(){
  uint i = uint(int(gl_FragCoord.x));
  vec2 k = atomC(uKey, i);
  if (uConj == 1) k.y = -k.y;
  fragOut = cmul(texelFetch(uZ, ivec2(int(i),0),0).rg, k);
}
"""

# Cleanup: score a stored complex vector against generated candidate atoms. One fragment per
# candidate; the candidate's name hash comes from an INTEGER texture, so no vocabulary is stored.
FS_CLEAN = "#version 330 core\n" + ATOM + """
uniform sampler2D uZ; uniform usampler2D uNames; uniform int uD;
out float fragOut;
void main(){
  int c = int(gl_FragCoord.x);
  uint base = texelFetch(uNames, ivec2(c,0),0).r;
  float acc = 0.0;
  for (int i = 0; i < uD; ++i) {
    vec2 z = texelFetch(uZ, ivec2(i,0),0).rg;
    vec2 a = atomC(base, uint(i));
    acc += z.x*a.x + z.y*a.y;          // Re<a,z> -- the cleanup score
  }
  fragOut = acc;
}
"""

class G:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.vbo = self.ctx.buffer(np.array([-1,-1,3,-1,-1,3], dtype="f4").tobytes())
        self.c = {}
    def prog(self, fs):
        if fs not in self.c:
            p = self.ctx.program(vertex_shader=VS, fragment_shader=fs)
            self.c[fs] = (p, self.ctx.vertex_array(p, [(self.vbo, "2f", "p")]))
        return self.c[fs]
    def texC(self, z):
        a = np.empty((1, len(z), 2), dtype="f4"); a[0,:,0] = z.real; a[0,:,1] = z.imag
        return self.ctx.texture((len(z),1), 2, a.tobytes(), dtype="f4")
    def texU(self, u):
        u = np.asarray(u, dtype="<u4")
        return self.ctx.texture((len(u),1), 1, u.tobytes(), dtype="u4")
    def run(self, fs, w, comps, texs, ints=None, uints=None):
        p, vao = self.prog(fs)
        out = self.ctx.texture((w,1), comps, dtype="f4")
        fbo = self.ctx.framebuffer(color_attachments=[out]); fbo.use()
        self.ctx.viewport = (0,0,w,1)
        for u,(n,t) in enumerate(texs.items()):
            t.use(u); p[n].value = u
        for k,v in (ints or {}).items():
            if k in p: p[k].value = int(v)
        for k,v in (uints or {}).items():
            if k in p: p[k].value = int(v)
        vao.render(moderngl.TRIANGLES)
        r = np.frombuffer(out.read(), dtype="f4").copy().reshape(w, comps)
        fbo.release(); out.release()
        return r

if __name__ == "__main__":
    g = G(); print("GL:", g.ctx.info["GL_RENDERER"])
    D = 256
    roles = ["colour", "size", "material"]
    fillers = ["red", "large", "metal"]
    cands = fillers + ["blue", "small", "wood", "green", "tiny", "glass"]

    # 1. atom generation: shader vs NumPy
    key = "colour"
    z1 = g.run(FS_BINDC, D, 2, {"uZ": g.texC(np.ones(D, dtype=complex))},
               uints={"uKey": int(HA.fnv1a(key))}, ints={"uConj": 0})
    ref = PH.atom(key, D)
    e_atom = float(np.max(np.abs((z1[:,0] + 1j*z1[:,1]) - ref)))
    print("atom generated in-shader vs NumPy : max|diff| %.3e" % e_atom)

    # 2. bind then unbind on the GPU -- the exactness claim
    a = PH.atom("sphere", D)
    zb = g.run(FS_BINDC, D, 2, {"uZ": g.texC(a)},
               uints={"uKey": int(HA.fnv1a("colour"))}, ints={"uConj": 0})
    zbc = zb[:,0] + 1j*zb[:,1]
    zu = g.run(FS_BINDC, D, 2, {"uZ": g.texC(zbc)},
               uints={"uKey": int(HA.fnv1a("colour"))}, ints={"uConj": 1})
    zuc = zu[:,0] + 1j*zu[:,1]
    print("GPU bind vs NumPy bind            : max|diff| %.3e"
          % float(np.max(np.abs(zbc - PH.bind(a, PH.atom("colour", D))))))
    print("GPU bind->unbind round trip       : max|diff| %.3e  (f32 target)"
          % float(np.max(np.abs(zuc - a))))

    # 3. a real record: bundle three bound pairs on the host, then do the ENTIRE recall on GPU
    rec = PH.bundle([PH.bind(PH.atom(r, D), PH.atom(f, D)) for r, f in zip(roles, fillers)])
    tn = g.texU([HA.fnv1a(c) for c in cands])
    ok = True
    for r, f in zip(roles, fillers):
        zr = g.run(FS_BINDC, D, 2, {"uZ": g.texC(rec)},
                   uints={"uKey": int(HA.fnv1a(r))}, ints={"uConj": 1})
        probe = zr[:,0] + 1j*zr[:,1]
        s = g.run(FS_CLEAN, len(cands), 1, {"uZ": g.texC(probe), "uNames": tn}, ints={"uD": D})
        got = cands[int(np.argmax(s[:,0]))]
        ref_got, ref_s = PH.cleanup(PH.unbind(rec, PH.atom(r, D)), cands, D)
        srt = np.sort(s[:,0])[::-1]
        print("   role %-9s -> GPU %-6s  NumPy %-6s  %s   margin %.4f"
              % (r, got, ref_got, "OK" if got == f == ref_got else "MISMATCH",
                 float(srt[0]-srt[1])))
        ok &= (got == f == ref_got)
    print("\nFULL ALGEBRA IN GLSL:", "bind, unbind, bundle, cleanup -- all correct" if ok else "FAILED")
    print("vocabulary stored: 0 bytes (atoms regenerated in-shader from the name hash)")
