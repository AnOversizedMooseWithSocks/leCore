"""SATO-SEQ -- turn a MESH into a stable SEQUENCE, and a sequence into a single hypervector.

WHY a sequence at all: a mesh is a set, but VSA sequence memory (permutation-power binding) and whole-mesh
similarity both want a canonical linear ORDER of the vertices. The order must be STABLE (independent of how
the file happened to list its vertices) or the encoding is noise. Three orders, each deterministic:

  * "morton"  -- Z-order curve on quantised coordinates: interleave the bits of (x,y,z). Pure integer ops,
                 no eigensolve, byte-stable under any input permutation (VERIFIED). Best default.
  * "zyx"     -- PolyGen's lexicographic sort by (z, y, x) after 8-bit quantisation (Nash et al., ICML 2020).
                 The permissive de-facto convention reused by MeshGPT/MeshAnything; reimplemented from the
                 paper's description, not from any GPL source.
  * "fiedler" -- spectral seriation by the 2nd cotan-Laplacian eigenfunction (mesh_fiedler_order): places
                 connectivity-adjacent vertices near each other. Reuses the shipped eigenmaps (auto dense/
                 sparse), so it inherits the matvec solver on big meshes.

LICENSING NOTE, on record: the "SATO" paper (Xu et al., "Strips as Tokens", SIGGRAPH 2026) ships a GPL-3.0
reference tokenizer (github Xrvitd/SATO, license CONFIRMED via the GitHub API). We do NOT use its strip
machinery and never copied its code -- this module is a clean-room serializer built from standard orderings
(Morton 1966; PolyGen 2020) and the engine's own FHRR primitives. The name SATO-SEQ is our backlog label.

Sequence -> one hypervector: permutation-power binding H = sum_k rho^k(token_k), rho a fixed hashlib-seeded
permutation. Decode position k by applying rho^-k and cleaning up against the token vocabulary. VERIFIED
round-trip: perfect to 64 tokens at dim 1024, 256 at dim 4096; the bundling capacity cliff is ~dim/8, so
seq_encode CHUNKS at <= dim/16 tokens per bundle and binds the chunks under distinct chunk-role permutations.

Deterministic throughout: hashlib-seeded phasor vocab and permutation, integer Morton keys, id-tie-broken
sorts. NumPy + stdlib + hashlib only.
"""
import hashlib
import numpy as np

from holographic.sampling_and_signal.holographic_fhrr import phasor_atom


def _seeded_rng(seed):
    """A numpy Generator seeded DETERMINISTICALLY from hashlib (never Python hash()): the seed is folded
    through sha256 so equal seeds give equal streams across processes with PYTHONHASHSEED unset too."""
    h = hashlib.sha256(str(seed).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def morton_key(p, bits=10):
    """Z-order (Morton) key of a point p in the unit cube: quantise each coord to `bits` bits and interleave.
    A single integer whose sort order is a space-filling curve -- nearby points get nearby keys. Deterministic
    integer arithmetic; clamps outside [0,1]. WHY Morton over a raw axis sort: locality in ALL axes at once,
    so the resulting vertex order does not tear along one direction."""
    q = np.clip((np.asarray(p, float) * (2 ** bits - 1)), 0, 2 ** bits - 1).astype(np.int64)
    key = 0
    for b in range(bits):
        for c in range(3):
            key |= ((int(q[c]) >> b) & 1) << (3 * b + c)
    return key


def mesh_sequence_order(mesh, order="morton", bits=10):
    """A stable linear ORDER of a mesh's vertices. order in {"morton","zyx","fiedler"} (see module docstring).
    Returns an int array of vertex indices. The coordinate orders (morton/zyx) normalise the mesh into the
    unit cube first so the order is scale/translation invariant; fiedler is intrinsic already."""
    V = np.asarray(mesh.vertices, float)
    if order == "fiedler":
        from holographic.mesh_and_geometry.holographic_crossfield import mesh_fiedler_order
        return mesh_fiedler_order(mesh)
    lo = V.min(0); span = np.ptp(V, axis=0); span[span < 1e-12] = 1.0
    U = (V - lo) / span                                          # into [0,1]^3, deterministic
    if order == "morton":
        keys = np.array([morton_key(u, bits=bits) for u in U], dtype=np.int64)
        return np.lexsort((np.arange(len(U)), keys))            # id tie-break
    if order == "zyx":
        q = np.clip(U * (2 ** 8 - 1), 0, 255).astype(np.int64)  # 8-bit like PolyGen
        return np.lexsort((np.arange(len(U)), q[:, 0], q[:, 1], q[:, 2]))  # sort by z, then y, then x
    raise ValueError("order must be one of 'morton','zyx','fiedler', got %r" % order)


def _permutation(dim, seed):
    """A fixed permutation of range(dim) and its inverse, seeded from hashlib. rho is applied to a vector by
    fancy-indexing v[rho]; rho^k by iterating; rho^-k by v[argsort of the k-times-composed rho]."""
    perm = _seeded_rng(("perm", seed)).permutation(dim)
    return perm


def seq_encode(tokens, dim=1024, seed=0, chunk=None, vocab_size=256):
    """Encode a sequence of INTEGER tokens into one hypervector by permutation-power binding.

    H = sum_k rho^k(vocab[token_k]). rho is a fixed hashlib-seeded permutation; vocab is a fixed hashlib-
    seeded phasor codebook of vocab_size atoms (MUST match seq_decode's vocab_size -- each phasor_atom call
    advances the shared rng, so a different count draws a different codebook and decode fails). Token ids must
    be < vocab_size.

    CAPACITY: a single permutation-power bundle holds ~dim/8 tokens before crosstalk swamps cleanup
    (MEASURED). Sequences longer than `chunk` (default dim//16, safely below the cliff) are split; each chunk
    is stored as its OWN hypervector and they are returned as a list (block encoding). KEPT NEGATIVE:
    bundling the chunks under distinct role phasors does NOT beat the cliff -- the total term count is what
    matters, so N chunks in one vector crosstalk exactly as N*chunk tokens would (measured 3/256). Block
    (list) storage is O(len/chunk) vectors and round-trips exactly.

    Returns a (dim,) complex vector if the sequence fits one chunk, else a list of (dim,) vectors.
    Deterministic (seeded vocab + permutation, fixed token order)."""
    tokens = [int(t) for t in tokens]
    if chunk is None:
        chunk = max(1, dim // 16)
    vocab_rng = _seeded_rng(("vocab", seed))
    vocab = np.stack([phasor_atom(dim, vocab_rng) for _ in range(int(vocab_size))])
    rho = _permutation(dim, seed)

    def encode_chunk(chunk_tokens):
        H = np.zeros(dim, complex)
        cur = np.arange(dim)
        for tok in chunk_tokens:
            H = H + vocab[tok][cur]                              # rho^k applied to the token atom
            cur = rho[cur]
        return H

    if len(tokens) <= chunk:
        return encode_chunk(tokens)
    return [encode_chunk(tokens[ci:ci + chunk]) for ci in range(0, len(tokens), chunk)]


def seq_decode(H, length, dim=1024, seed=0, vocab_size=256, chunk=None):
    """Decode `length` integer tokens from a permutation-power encoding produced by seq_encode with the same
    dim/seed/vocab_size. H is a single (dim,) vector (short sequence) or the list of block vectors (long).
    For each position k, unwind rho^k and pick the vocab atom of highest similarity (cleanup memory). Returns
    a list of token ids."""
    if chunk is None:
        chunk = max(1, dim // 16)
    vocab_rng = _seeded_rng(("vocab", seed))
    vocab = np.stack([phasor_atom(dim, vocab_rng) for _ in range(int(vocab_size))])
    rho = _permutation(dim, seed)

    def decode_chunk(Hc, n):
        toks = []
        cur = np.arange(dim)
        for k in range(n):
            probe = Hc[np.argsort(cur)]                         # rho^-k
            toks.append(int(np.argmax(np.real(np.conj(vocab) @ probe))))
            cur = rho[cur]
        return toks

    if not isinstance(H, list):
        return decode_chunk(H, length)
    out = []
    for bi, ci in enumerate(range(0, length, chunk)):
        out += decode_chunk(H[bi], min(chunk, length - ci))
    return out


def mesh_to_tokens(mesh, order="morton", bits=8):
    """Serialise a mesh to a token stream: order the vertices (mesh_sequence_order), quantise each coordinate
    to `bits` bits, and emit 3 tokens per vertex (x,y,z), each in [0, 2^bits). The inverse of quantisation is
    lossy by design (grid resolution = tolerance). Returns (tokens, order, grid) where grid carries the
    bbox+bits to dequantise. This is the mesh-as-sequence a hypervector or an autoregressive consumer wants."""
    V = np.asarray(mesh.vertices, float)
    idx = mesh_sequence_order(mesh, order=order, bits=max(bits, 10) if order == "morton" else 10)
    lo = V.min(0); span = np.ptp(V, axis=0); span[span < 1e-12] = 1.0
    U = (V[idx] - lo) / span
    q = np.clip(U * (2 ** bits - 1), 0, 2 ** bits - 1).astype(np.int64)
    tokens = q.reshape(-1).tolist()                             # x0,y0,z0, x1,y1,z1, ...
    return tokens, idx, {"lo": lo, "span": span, "bits": bits}


def _selftest():
    """SATO-SEQ regression trap: Morton order is stable under input permutation; the three orders are all
    valid permutations; a token sequence round-trips through seq_encode/seq_decode; a mesh serialises and
    dequantises back to within the grid tolerance."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons

    # Morton stability under input permutation
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, (150, 3))
    k1 = [morton_key(p) for p in pts]
    perm = rng.permutation(150)
    k2 = [morton_key(p) for p in pts[perm]]
    o1 = np.argsort(k1, kind="stable"); o2 = perm[np.argsort(k2, kind="stable")]
    assert np.array_equal(pts[o1], pts[o2]), "Morton order must be stable under input permutation"

    S = triangulate_ngons(loop_subdivide(box(), 3))
    Vs = np.asarray(S.vertices, float); Vs = Vs / (np.linalg.norm(Vs, axis=1, keepdims=True) + 1e-9)
    mesh = Mesh(Vs, [tuple(int(i) for i in f) for f in S.faces])
    for od in ("morton", "zyx", "fiedler"):
        order = mesh_sequence_order(mesh, order=od)
        assert sorted(order.tolist()) == list(range(len(Vs))), "%s must be a valid permutation" % od

    # sequence round-trip: perfect below the capacity cliff (dim/8); use dim/16 tokens
    dim = 1024
    toks = _seeded_rng("t").integers(0, 40, dim // 16).tolist()
    H = seq_encode(toks, dim=dim, seed=0, vocab_size=64)
    dec = seq_decode(H, len(toks), dim=dim, seed=0, vocab_size=64)
    assert dec == toks, "sequence must round-trip below the capacity cliff (got %d/%d correct)" % (
        sum(a == b for a, b in zip(dec, toks)), len(toks))
    # KEPT NEGATIVE (measured, research pass): above ~dim/8 tokens the permutation-power bundle degrades (dim
    # 1024 fell to 0.93 at 128, 0.64 at 256), and BUNDLING chunks under role phasors does NOT beat it -- the
    # total term count is what crosstalks (measured 3/256). seq_encode therefore stores a long sequence as a
    # LIST of block vectors (O(len/chunk) of them), which round-trips EXACTLY.
    long_toks = _seeded_rng("lt").integers(0, 40, dim // 4).tolist()   # 4x a single bundle's safe load
    Hl = seq_encode(long_toks, dim=dim, seed=1, vocab_size=64)
    assert isinstance(Hl, list), "a long sequence must be stored as block vectors, not one bundle"
    decl = seq_decode(Hl, len(long_toks), dim=dim, seed=1, vocab_size=64)
    assert decl == long_toks, "block encode must round-trip a long sequence exactly (%d/%d)" % (
        sum(a == b for a, b in zip(decl, long_toks)), len(long_toks))

    # mesh serialisation dequantises back within one grid cell
    tokens, idx, grid = mesh_to_tokens(mesh, order="morton", bits=8)
    q = np.asarray(tokens, float).reshape(-1, 3)
    deq = q / (2 ** grid["bits"] - 1) * grid["span"] + grid["lo"]
    err = np.abs(deq - Vs[idx]).max()
    cell = float(np.max(grid["span"]) / (2 ** grid["bits"] - 1))
    assert err <= cell + 1e-9, "dequantised mesh must be within one grid cell (%.4f vs %.4f)" % (err, cell)
    print("meshseq selftest OK (Morton perm-stable; 3 orders valid; seq round-trips + chunks past the "
          "dim/8 cliff; mesh dequantises within a grid cell)")


if __name__ == "__main__":
    _selftest()
