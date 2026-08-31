"""Evaluate the SAME atom definition in GLSL and pin it against NumPy, bit for bit.

The encoder is one fragment shader: output component i loops over the query's token hashes and
accumulates the sign bit of lowbias32(tokenHash ^ i). Token hashes arrive in an INTEGER texture
(R32UI / usampler2D, both WebGL2 features) because a float texture cannot carry a u32 exactly
above 2^24 -- routing hashes through floats would silently corrupt every atom past that.

The query is deliberately NOT normalised: a positive scalar cannot move an argmax, so the
browser skips a whole reduction pass.
"""
import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import numpy as np, moderngl
import holographic.agents_and_reasoning.holographic_hashatom as HA
import glsl_hier as H

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

FS_ENCODE = """
#version 330 core
uniform usampler2D uTok;    // R32UI: one FNV-1a token hash per texel
uniform int uT;             // token count
uniform float uScale;       // 1/sqrt(D)
out float fragOut;
void main(){
  uint i = uint(int(gl_FragCoord.x));
  float acc = 0.0;
  for (int t = 0; t < uT; ++t) {
    uint x = texelFetch(uTok, ivec2(t,0), 0).r ^ i;
    uint s = x * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    x = (w >> 22u) ^ w;
    acc += ((x >> 31u) == 1u) ? 1.0 : -1.0;   // Rademacher: no trig, so no float divergence
  }
  fragOut = acc * uScale;
}
"""

if __name__ == "__main__":
    ctx = moderngl.create_standalone_context(require=330, backend="egl")
    vbo = ctx.buffer(np.array([-1,-1,3,-1,-1,3], dtype="f4").tobytes())
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS_ENCODE)
    vao = ctx.vertex_array(prog, [(vbo, "2f", "p")])
    print("GL:", ctx.info["GL_RENDERER"])

    def gpu_encode(tokens, D):
        h = np.array([HA.fnv1a(t) for t in tokens], dtype="<u4")
        tt = ctx.texture((len(h), 1), 1, h.tobytes(), dtype="u4")
        out = ctx.texture((D, 1), 1, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[out]); fbo.use(); ctx.viewport = (0,0,D,1)
        tt.use(0); prog["uTok"].value = 0
        prog["uT"].value = len(h); prog["uScale"].value = 1.0/np.sqrt(D)
        vao.render(moderngl.TRIANGLES)
        r = np.frombuffer(out.read(), dtype="f4").copy()
        fbo.release(); out.release(); tt.release()
        return r

    D = 256
    toks = ["holographic","vector","memory","bind","cleanup","resonator","codebook"]
    cpu = HA.encode_hash(toks, D, normalise=False).astype("f4")
    gpu = gpu_encode(toks, D)
    print("query encode, GPU vs NumPy: BIT-IDENTICAL =", np.array_equal(cpu, gpu),
          " max|diff| = %.3e" % np.max(np.abs(cpu.astype(np.float64)-gpu.astype(np.float64))))

    # ...and the whole pipeline on the real corpus, encoded with the SHADER-NATIVE family.
    docs, _ = H.load_corpus(D)
    K = len(docs)
    V = np.stack([HA.encode_hash(t, D) for _, t in docs])
    rng = np.random.default_rng(0)
    hits_cpu = hits_gpu = agree = 0
    for i, (_, tk) in enumerate(docs):
        pick = rng.choice(len(tk), max(3, int(len(tk)*0.4)), replace=False)
        qt = [tk[j] for j in pick]
        qc = HA.encode_hash(qt, D, normalise=False)
        qg = gpu_encode(qt, D).astype(np.float64)
        ac, ag = int(np.argmax(V @ qc)), int(np.argmax(V @ qg))
        hits_cpu += ac == i; hits_gpu += ag == i; agree += ac == ag
    print("real corpus %d docs, dim %d, HASH-ATOM family (no vocabulary shipped):" % (K, D))
    print("   NumPy encode + flat scan : acc %.4f" % (hits_cpu/K))
    print("   GLSL  encode + flat scan : acc %.4f" % (hits_gpu/K))
    print("   same answer on %d/%d queries" % (agree, K))
