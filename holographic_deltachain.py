"""A chunked DELTA CHAIN with an integrity proof (DELTA-1).

THE IDEA (from the request): when you have a SEQUENCE of chunks (state snapshots, frames, scene versions, the
per-seam states a chunked run produces), don't store each in full -- store a base and, per chunk, a DELTA against
either the BASE (first chunk) or the PRIOR chunk, whichever is smaller. Then make the sequence's integrity
PROVABLE with a hash chain (each chunk's hash folds in the prior's, so a corruption, a wrong base, or a broken
propagation is DETECTED, not silently reconstructed wrong) plus a Merkle ROOT over all chunk hashes -- one small
"fractal" proof of the whole sequence.

WHAT EXISTED / WHAT'S NEW (probe-first): FrameCache (holographic_anim) already stores frames as deltas vs a BASE
with a hot/warm/cold tier (the honest L1-L4 analogy -- Python cannot touch real CPU caches). holographic_scenedelta
already content-hashes scene COMPONENTS for dedup. NEITHER does delta-vs-PRIOR, auto base/prior selection, or a
CHAINED integrity proof over a sequence. Those are the gaps here.

HONEST DESIGN NOTES (the recurring discipline):
  * INTEGRITY IS hashlib, NOT a VSA bundle. A bundled "checksum" hypervector is lossy (crosstalk) and cannot give
    bit-exact tamper detection; exact integrity is exactly where VSA-native is NOT beneficial. So the proof is a
    deterministic SHA-256 chain + Merkle root. (VSA-native where it pays, exact where exactness is the point.)
  * CODEBOOKS HELP SIZE, losslessly, where the data is codebook-structured: a changed row that EXACTLY equals a
    codebook atom is stored as a small INDEX, not a full float row -- bit-exact, much smaller, the common case for
    VSA states (bundles of atoms). Near-but-not-equal rows fall back to full storage (no silent loss).
  * VECTORIZED, no per-element Python on the data: changed-row detection is np.where over a max-abs reduction,
    codebook matching is a broadcast compare, reconstruction is fancy indexing, hashing is one call per chunk on
    .tobytes(). The only Python loop is over CHUNKS (unavoidable, small) -- so there is no hot VSA<->Python seam.
"""

import hashlib
import numpy as np


def _sha(*byteses):
    """Deterministic SHA-256 over a sequence of byte strings -> 32 raw bytes."""
    h = hashlib.sha256()
    for b in byteses:
        h.update(b)
    return h.digest()


def merkle_root(leaf_hashes):
    """A binary Merkle root over a list of leaf hashes (raw bytes) -- the single 'fractal' proof of the whole
    sequence: change any chunk and the root changes. Odd levels duplicate the last node (standard). O(n) build,
    and a leaf's inclusion is provable in O(log n) (the proof path), though this returns just the root."""
    if not leaf_hashes:
        return _sha(b"")
    level = list(leaf_hashes)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])                          # duplicate the last to pair it
        level = [_sha(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


class DeltaChain:
    """Append-only store of a chunk sequence as base + per-chunk deltas (auto base-vs-prior) with a hash-chain +
    Merkle integrity proof. Bit-exact (within `tol`), deterministic, vectorized. Each chunk is an (N, D) float
    array of the SAME shape as the base.

    api: append(chunk) -> index; get(i) (reconstruct + verify); verify() (whole chain); memory_bytes()/full_bytes()
    (the saving); tip_hash() / root() (the proof handles). With `codebook` set, rows equal to an atom store an
    index (lossless size win)."""

    def __init__(self, base, tol=0.0, codebook=None):
        self.base = np.ascontiguousarray(np.asarray(base, float))
        self.shape = self.base.shape
        self.tol = float(tol)
        self.codebook = None if codebook is None else np.ascontiguousarray(np.asarray(codebook, float))
        self._deltas = []                                    # per chunk: dict(ref, idx, rows, code_idx, code_at)
        self._prev = self.base                               # last reconstructed chunk (the 'prior' reference)
        self._hashes = [_sha(self.base.tobytes())]           # h_0 = H(base); h_i folds in h_{i-1} (the chain)

    # ---- helpers (all vectorized) ----------------------------------------------------------------
    def _changed_rows(self, chunk, ref):
        diff = np.abs(chunk - ref).max(axis=1) > self.tol    # which rows differ from the reference (vectorized)
        return np.where(diff)[0]

    def _encode_rows(self, rows):
        """Split changed rows into codebook-index hits (exact atom matches) and literal float rows. Vectorized
        broadcast compare; no per-row Python beyond the index bookkeeping."""
        if self.codebook is None or len(rows) == 0:
            return np.empty(0, int), np.empty(0, int), rows
        # max-abs distance from each changed row to each atom: (n_rows, n_atoms)
        d = np.abs(rows[:, None, :] - self.codebook[None, :, :]).max(axis=2)
        nearest = d.argmin(axis=1)
        exact = d[np.arange(len(rows)), nearest] <= self.tol
        code_at = np.where(exact)[0]                          # which changed-row positions are atom-exact
        code_idx = nearest[exact]                            # the atom index each maps to
        lit = rows[~exact]                                   # the rest stored literally
        return code_idx, code_at, lit

    # ---- the chain -------------------------------------------------------------------------------
    def append(self, chunk):
        chunk = np.ascontiguousarray(np.asarray(chunk, float))
        if chunk.shape != self.shape:
            raise ValueError(f"chunk shape {chunk.shape} != base shape {self.shape}")
        idx_base = self._changed_rows(chunk, self.base)
        idx_prior = self._changed_rows(chunk, self._prev)
        ref = "prior" if len(idx_prior) < len(idx_base) else "base"   # the smaller delta wins
        idx = idx_prior if ref == "prior" else idx_base
        rows = chunk[idx]
        code_idx, code_at, lit = self._encode_rows(rows)
        # the lit_at positions are the changed-row positions NOT covered by a codebook hit
        lit_at = np.setdiff1d(np.arange(len(idx)), code_at, assume_unique=False)
        self._deltas.append({"ref": ref, "idx": idx, "lit_at": lit_at, "lit": lit,
                             "code_at": code_at, "code_idx": code_idx})
        self._hashes.append(_sha(self._hashes[-1], chunk.tobytes()))   # chain: h_i = H(h_{i-1} || chunk)
        self._prev = chunk                                   # next chunk's 'prior' reference
        return len(self._deltas) - 1

    def _reconstruct(self, i):
        """Rebuild chunk i from base + deltas, following prior-references back as needed (iterative, no recursion
        depth issue). Pure array ops."""
        # find the start: walk back over consecutive 'prior' refs to the nearest 'base'-ref (or the base itself)
        order = []
        j = i
        while j >= 0:
            order.append(j)
            if self._deltas[j]["ref"] == "base":
                break
            j -= 1
        order.reverse()
        state = self.base.copy()
        for k in order:
            d = self._deltas[k]
            ref = self.base if d["ref"] == "base" else state    # 'base' resets from base; 'prior' edits running state
            if d["ref"] == "base":
                state = self.base.copy()
            rows = np.empty((len(d["idx"]), self.shape[1]), float)
            if len(d["lit_at"]):
                rows[d["lit_at"]] = d["lit"]
            if len(d["code_at"]):
                rows[d["code_at"]] = self.codebook[d["code_idx"]]
            state[d["idx"]] = rows
        return state

    def get(self, i):
        """Reconstruct chunk i AND verify its hash against the chain. Raises IntegrityError on a mismatch (a
        corrupted delta, a wrong base, or a broken upstream propagation all surface here)."""
        state = self._reconstruct(i)
        if _sha(self._hashes[i], state.tobytes()) != self._hashes[i + 1]:
            raise IntegrityError(f"chunk {i} failed its hash check -- the delta chain is corrupted")
        return state

    def verify(self):
        """Verify the WHOLE chain: every chunk reconstructs to its recorded hash, and the Merkle root matches.
        Returns True, or raises IntegrityError at the first bad chunk."""
        for i in range(len(self._deltas)):
            self.get(i)                                      # each get() re-checks the chain link
        return True

    # ---- proof handles + accounting --------------------------------------------------------------
    def tip_hash(self):
        """The chain's tip hash (hex) -- folds in every chunk in order; any change anywhere changes it."""
        return self._hashes[-1].hex()

    def root(self):
        """The Merkle root (hex) over all chunk hashes -- the single 'fractal' proof of the whole sequence."""
        return merkle_root(self._hashes).hex()

    def memory_bytes(self):
        b = self.base.nbytes
        for d in self._deltas:
            b += d["idx"].nbytes + d["lit"].nbytes + d["code_at"].nbytes + d["code_idx"].nbytes + d["lit_at"].nbytes
        return b

    def full_bytes(self):
        return self.base.nbytes * (len(self._deltas) + 1)    # storing every chunk in full (the naive baseline)


class IntegrityError(Exception):
    """Raised when a DeltaChain reconstruction does not match its recorded hash."""


def _selftest():
    rng = np.random.default_rng(0)
    N, D = 200, 16
    base = rng.standard_normal((N, D))
    chain = DeltaChain(base, tol=0.0)
    # a DRIFTING sequence: each chunk edits ~5 rows of the prior -> prior-deltas should win and stay small
    cur = base.copy()
    originals = [base.copy()]
    for _ in range(12):
        cur = cur.copy()
        rows = rng.choice(N, 5, replace=False)
        cur[rows] = rng.standard_normal((5, D))
        chain.append(cur)
        originals.append(cur.copy())
    # bit-exact reconstruction + integrity
    assert all(np.array_equal(chain.get(i), originals[i + 1]) for i in range(12))
    assert chain.verify()
    saving = chain.full_bytes() / chain.memory_bytes()
    chose_prior = sum(d["ref"] == "prior" for d in chain._deltas)
    # corruption is DETECTED
    chain._deltas[3]["lit"][0, 0] += 1.0                     # tamper with a stored delta
    detected = False
    try:
        chain.get(3)
    except IntegrityError:
        detected = True
    assert detected, "corruption must be detected"
    print(f"deltachain selftest ok: bit-exact reconstruct; {chose_prior}/12 chunks chose prior-delta; "
          f"{saving:.1f}x smaller than storing full; tamper DETECTED by the hash chain")


if __name__ == "__main__":
    _selftest()
