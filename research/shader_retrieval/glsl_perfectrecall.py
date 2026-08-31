"""Wire PERFECT RECALL to the GLSL stack: the Bloom/tile candidate pass as a fragment shader.

WHAT PORTS AND WHAT MUST NOT. PerfectRecallIndex is two halves with different natures:
  * the CANDIDATE half -- tile probes culled by an AND of query bits, then per-doc filter tests --
    is pure bitwise arithmetic over fixed-width words. That is a fragment shader, and it is the
    part that costs O(N) on the host.
  * the VERIFY half -- exact sha256 term-hash set membership -- is NOT ported and must not be.
    It is what buys ZERO FALSE POSITIVES, and moving a correctness guarantee onto a substrate
    whose float behaviour this project has spent an entire arc bounding would be trading the one
    exact thing in the module for speed. It stays on the host, and it only ever sees candidates.

So the shader answers "which documents COULD contain all these terms" -- a superset, by design --
and the host turns that into the exact answer. The shader's contract is therefore not "matches the
final answer" but the stricter, checkable one: IT MUST REPRODUCE THE HOST'S CANDIDATE SET EXACTLY.
A shader that dropped one candidate would silently break the module's zero-false-negative claim,
which is its entire reason to exist -- so that is what the differential test asserts.

RENDERING SHAPE, which is the module's own metaphor and is load-bearing here: the tile probe is an
irradiance probe (bake once, test a whole tile in one AND), the per-doc test is the depth pass, and
the host verify is the final shading of surviving fragments.
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import moderngl

VS = "#version 330 core\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.,1.);}"

# One fragment per DOCUMENT. Tests (doc_filter AND query) == query, word by word, and first checks
# the document's TILE probe so a culled tile costs one early-out instead of `words` fetches.
FS_CANDIDATE = """
#version 330 core
uniform usampler2D uDocF;    // per-doc filter words, row-major: doc * uWords + w
uniform usampler2D uTileF;   // per-tile probe words: tile * uTileWords + w
uniform usampler2D uQ;       // query words at doc resolution
uniform usampler2D uQT;      // query words at tile resolution
uniform int uWords, uTileWords, uTile, uN, uW;
out uint fragOut;            // 1 = candidate, 0 = culled
ivec2 at(int i){ return ivec2(i % uW, i / uW); }
void main(){
  int d = int(gl_FragCoord.x) + int(gl_FragCoord.y) * uW;
  if (d >= uN) { fragOut = 0u; return; }
  int t = d / uTile;
  // IRRADIANCE PROBE: if the tile lacks any query bit, nothing in it can contain every term.
  for (int w = 0; w < uTileWords; ++w) {
    uint qw = texelFetch(uQT, at(w), 0).r;
    if (qw != 0u) {
      uint tw = texelFetch(uTileF, at(t * uTileWords + w), 0).r;
      if ((tw & qw) != qw) { fragOut = 0u; return; }
    }
  }
  // DEPTH TEST: the document's own filter must hold every query bit.
  for (int w = 0; w < uWords; ++w) {
    uint qw = texelFetch(uQ, at(w), 0).r;
    if (qw != 0u) {
      uint dw = texelFetch(uDocF, at(d * uWords + w), 0).r;
      if ((dw & qw) != qw) { fragOut = 0u; return; }
    }
  }
  fragOut = 1u;
}
"""


class GLCandidates:
    """Uploads an index's filters once; answers candidate sets per query on the GPU."""

    def __init__(self, idx, channel="token", W=2048):
        self.idx, self.channel, self.W = idx, channel, W
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS_CANDIDATE)
        quad = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(quad, "2f", "p")])
        df = np.asarray(idx.doc_filters[channel], dtype="<u4").reshape(-1)
        tf = np.asarray(idx.tile_filters[channel], dtype="<u4").reshape(-1)
        self.tDoc, self.tTile = self._u(df), self._u(tf)
        self.N = idx.n
        self.outW = min(W, max(1, self.N))
        self.outH = (self.N + self.outW - 1) // self.outW
        self.out = self.ctx.texture((self.outW, self.outH), 1, dtype="u4")
        self.fbo = self.ctx.framebuffer(color_attachments=[self.out])

    def _u(self, a):
        a = np.asarray(a, dtype="<u4").reshape(-1)
        h = max(1, (len(a) + self.W - 1) // self.W)
        buf = np.zeros(h * self.W, dtype="<u4"); buf[:len(a)] = a
        t = self.ctx.texture((self.W, h), 1, buf.tobytes(), dtype="u4")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return t

    def candidates(self, qwords, qtwords):
        tq, tqt = self._u(qwords), self._u(qtwords)
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.outW, self.outH)
        for n, t, u in (("uDocF", self.tDoc, 0), ("uTileF", self.tTile, 1),
                        ("uQ", tq, 2), ("uQT", tqt, 3)):
            t.use(u); self.prog[n].value = u
        for n, v in (("uWords", self.idx.words), ("uTileWords", self.idx.tile_words),
                     ("uTile", self.idx.tile), ("uN", self.N), ("uW", self.W)):
            self.prog[n].value = int(v)
        self.vao.render(moderngl.TRIANGLES)
        px = np.frombuffer(self.out.read(), dtype="<u4")[:self.N]
        tq.release(); tqt.release()
        return np.nonzero(px)[0]
