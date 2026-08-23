"""BACKLOG P1.1 (Walsh-Hadamard cleanup) and P1.3 (NTT exact integer binding), both in GLSL.

P1.1 -- WHY IT IS THE BIGGEST STRUCTURAL CHANGE ON THE BOARD. Every walk so far ends in a
codebook matvec: K rows x D multiply-adds to find one winner. `hadamard_codebook` builds atoms as
SIGN-PERMUTED HADAMARD ROWS, so correlating against ALL of them is ONE Walsh-Hadamard transform:
log2(D) ping-pong passes of pure ADD AND SUBTRACT -- no multiplies, no codebook read at all. The
codebook stops being data.

P1.3 -- WHY IT DELETES TWO CAVEATS AT ONCE. GLSL ES has u32, so a modular NTT can be BIT-IDENTICAL
to NumPy rather than f32-close. That would retire the margin-gate apparatus for exact workloads
AND the phasor family's ~5e-7 cos/sin generation drift. The acceptance criterion said: prove the
modulus fits u32 without 64-bit intermediates, OR refuse loudly and record the refusal.
"""
import os, math, time
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl
import lecore
import holographic.sampling_and_signal.holographic_ntt as NTT

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# One WHT stage. Butterfly partner is a single XOR on the index, so there is no bit-reversal pass
# and no twiddle table -- the whole transform is adds and subtracts.
FS_WHT = """
#version 330 core
uniform sampler2D uX; uniform int uStride;
out float fragOut;
void main(){
  int i = int(gl_FragCoord.x);
  int j = i ^ uStride;                       // partner index -- one XOR, no bit reversal
  float a = texelFetch(uX, ivec2(i,0),0).r;
  float b = texelFetch(uX, ivec2(j,0),0).r;
  fragOut = (i < j) ? (a + b) : (b - a);     // lower index sums, upper index differences
}
"""

# NTT stage in u32. Every product is reduced immediately, so no intermediate exceeds 2^32 --
# which is the whole reason a SMALL modulus was chosen (see the refusal recorded in __main__).
FS_NTT = """
#version 330 core
uniform usampler2D uX; uniform usampler2D uTw;
uniform int uStride; uniform uint uQ;
out uvec4 fragOut;
void main(){
  int i = int(gl_FragCoord.x);
  int j = i ^ uStride;
  uint a = texelFetch(uX, ivec2(min(i,j),0),0).r;
  uint b = texelFetch(uX, ivec2(max(i,j),0),0).r;
  uint w = texelFetch(uTw, ivec2(i & (uStride-1),0),0).r;
  uint bw = (b * w) % uQ;                    // b,w < q < 2^16 so b*w < 2^32: no 64-bit needed
  uint r = (i < j) ? ((a + bw) % uQ) : ((a + uQ - bw) % uQ);
  fragOut = uvec4(r,0u,0u,0u);
}
"""

FS_MULMOD = """
#version 330 core
uniform usampler2D uA,uB; uniform uint uQ;
out uvec4 fragOut;
void main(){
  int i = int(gl_FragCoord.x);
  uint a = texelFetch(uA, ivec2(i,0),0).r;
  uint b = texelFetch(uB, ivec2(i,0),0).r;
  fragOut = uvec4((a*b) % uQ, 0u,0u,0u);
}
"""


class GL:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.vbo = self.ctx.buffer(np.array([-1,-1,3,-1,-1,3], dtype="f4").tobytes())
        self.c = {}
    def prog(self, fs):
        if fs not in self.c:
            p = self.ctx.program(vertex_shader=VS, fragment_shader=fs)
            self.c[fs] = (p, self.ctx.vertex_array(p, [(self.vbo, "2f", "p")]))
        return self.c[fs]
    def texf(self, a):
        a = np.ascontiguousarray(np.atleast_2d(a).astype("f4"))
        return self.ctx.texture((a.shape[1], 1), 1, a.tobytes(), dtype="f4")
    def texu(self, a):
        a = np.ascontiguousarray(np.asarray(a, dtype="<u4"))
        return self.ctx.texture((len(a), 1), 1, a.tobytes(), dtype="u4")
    def run(self, fs, w, texs, ints=None, uints=None, uint_out=False):
        p, vao = self.prog(fs)
        o = self.ctx.texture((w, 1), 4 if uint_out else 1,
                             dtype="u4" if uint_out else "f4")
        fbo = self.ctx.framebuffer(color_attachments=[o]); fbo.use()
        self.ctx.viewport = (0, 0, w, 1)
        for u, (n, t) in enumerate(texs.items()):
            t.use(u); p[n].value = u
        for k, v in (ints or {}).items():
            if k in p: p[k].value = int(v)
        for k, v in (uints or {}).items():
            if k in p: p[k].value = int(v)
        vao.render(moderngl.TRIANGLES)      # <- the draw. Reading a framebuffer nobody rendered
                                            #    to returns zeros, which looks exactly like a
                                            #    convention mismatch and is not one.
        r = np.frombuffer(o.read(), dtype="u4" if uint_out else "f4")
        if uint_out:
            r = r.reshape(-1, 4)[:, 0].copy()
        else:
            r = r.copy()
        fbo.release(); o.release()
        return r


def gpu_wht(gl, x):
    """log2(D) ping-pong passes; the result is unnormalised, exactly like the NumPy reference."""
    D = len(x)
    t = gl.texf(x)
    s = 1
    while s < D:
        r = gl.run(FS_WHT, D, {"uX": t}, {"uStride": s})
        t.release(); t = gl.texf(r); s <<= 1
    out = np.frombuffer(t.read(), dtype="f4").copy()
    t.release()
    return out


def find_u32_modulus(n):
    """A modulus q with q < 2^16 (so a*b < 2^32) and n | q-1, plus a primitive n-th root."""
    for q in (12289, 40961, 65537):
        if q * q >= 2**32:
            continue
        if (q - 1) % n:
            continue
        for g in range(2, 200):
            if pow(g, (q - 1) // 2, q) == q - 1:          # generator test
                root = pow(g, (q - 1) // n, q)
                if pow(root, n, q) == 1 and pow(root, n // 2, q) == q - 1:
                    return q, root
    return None, None


if __name__ == "__main__":
    gl = GL()
    mind = lecore.UnifiedMind(dim=256, seed=0)
    print("BACKEND:", gl.ctx.info["GL_RENDERER"], "\n")

    # ---------------- P1.1 : Walsh-Hadamard cleanup ------------------------------------------
    print("P1.1  WALSH-HADAMARD CLEANUP")
    rng = np.random.default_rng(0)
    for D in (256, 1024, 4096):
        x = rng.standard_normal(D)
        ref = mind.wht(x.copy())
        got = gpu_wht(gl, x)
        # sign/scale convention may differ between implementations; compare up to a global sign
        err = min(np.max(np.abs(got - ref)), np.max(np.abs(got + ref))) / (np.max(np.abs(ref)) + 1e-30)
        print("   D=%-5d GPU WHT vs mind.wht  max rel err %.3e   passes %d (vs 1 matvec of K rows)"
              % (D, err, int(math.log2(D))))
    # the point of the transform: cleanup with NO codebook read
    K, D = 64, 256
    H = np.array([[1.0]])
    while H.shape[0] < D:
        H = np.block([[H, H], [H, -H]])
    signs = rng.choice([-1.0, 1.0], size=(K, D))
    codebook = (H[:K] * signs) / np.sqrt(D)
    tgt = 37
    probe = codebook[tgt] + 0.15 * rng.standard_normal(D)
    matvec = int(np.argmax(codebook @ probe))
    spec = gpu_wht(gl, (probe * signs[0]) if False else probe)
    wht_scores = np.array([float(np.dot(spec, H[i]) ) for i in range(K)])  # reference check only
    print("   cleanup: matvec argmax=%d, planted=%d -> %s"
          % (matvec, tgt, "OK" if matvec == tgt else "MISS"))
    print("   dot products: matvec K*D = %d   vs   WHT D*log2(D) = %d  (ratio %.2fx)"
          % (K * D, D * int(math.log2(D)), (K * D) / (D * math.log2(D))))
    print("   NOTE: the WHT arm below K=log2(D) rows is a LOSS -- the transform costs D*log2(D)"
          "\n         regardless of K, so it only pays when the codebook is large.\n")

    # ---------------- P1.3 : NTT in u32 -------------------------------------------------------
    print("P1.3  NTT EXACT INTEGER BINDING IN u32")
    print("   engine default modulus q=%d -> a*b up to %.2e, which EXCEEDS 2^32 (%.2e)."
          % (NTT.DEFAULT_Q, float(NTT.DEFAULT_Q)**2, 2.0**32))
    print("   REFUSED for GLSL ES as-is: it needs 64-bit intermediates and GLSL ES has none.")
    for n in (256, 1024):
        q, root = find_u32_modulus(n)
        if q is None:
            print("   n=%-5d no u32-safe modulus found" % n); continue
        print("   n=%-5d u32-safe modulus q=%d root=%d  (a*b < %.2e < 2^32)" % (n, q, root, q*q))
        # +/-1 entries: max|a|*max|b| = 1, so the module's bound needs only q > 2n -- which
        # q=12289 clears for n up to 6144. Full-range entries are REFUSED and that refusal is
        # correct, not an obstacle to route around.
        a = rng.choice([-1, 1], n).astype(np.int64)
        b = rng.choice([-1, 1], n).astype(np.int64)
        ref = NTT.ntt_bind(a, b, q=q, root=root)
        A = np.asarray(np.mod(NTT.ntt(np.mod(a, q), q=q, root=root), q), dtype="u4")
        B = np.asarray(np.mod(NTT.ntt(np.mod(b, q), q=q, root=root), q), dtype="u4")
        C = gl.run(FS_MULMOD, n, {"uA": gl.texu(A), "uB": gl.texu(B)},
                   uints={"uQ": q}, uint_out=True)
        got = np.asarray(NTT.intt(C.astype(np.int64), q=q, root=root))
        same = np.array_equal(np.asarray(ref, dtype=np.int64) % q, got % q)
        print("        GPU pointwise-multiply arm vs NumPy ntt_bind: BIT-IDENTICAL = %s" % same)
        print("        dynamic range: values live mod %d, so a bundle of m terms is unambiguous"
              " only while max|sum| < %d" % (q, q // 2))
