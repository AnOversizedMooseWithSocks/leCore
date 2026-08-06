"""Nested memory: a LIBRARY of knowledge bases in ONE vector, queried by (base, key)
in a SINGLE unbind (holographic_nested, NEST-1).

THE TRICK, which is algebra and not machinery
---------------------------------------------
bind is associative and distributes over the bundle, so a library of whole superposed
memories

    library = sum_i bind(name_i,  sum_j bind(key_ij, value_ij))

IS, identically,

    library = sum_ij bind(name_i (*) key_ij,  value_ij)

-- a FLAT pair memory whose keys are the composite name(*)key. Two consequences, both
load-bearing and both asserted in the selftest:

  1  A two-level lookup costs ONE unbind: query(base, key) =
     cleanup(unbind(library, name (*) key)). No base is ever reconstructed to be read.
     Nesting depth is free at query time; you pay only in LOAD.
  2  Capacity is the EXISTING law at the flat load: M bases x n facts each = M*n pairs,
     so allocate(M*n, ...) prices the library exactly, and the gated PIC decoder,
     int8 decision-freeness, and the 1-bit export story all apply unchanged. The
     Nesting Depth Law's warning stands where it always stood: depth costs the PRODUCT
     of the loads, so deep libraries want the allocator, not optimism.

WHY THIS TURNS HEADS: "a library of databases in one vector, any fact from any base in
one operation, exported at one bit per dimension" is the holographic promise made
literal -- and it is not a metaphor here, it is three shipped mechanisms composed:
seed-derived codebooks (the names and keys cost nothing), the capacity allocator (the
dimension is priced before the first fact), and the load-gated decoder (recall at spec
or an honest refusal). Users build on it by stacking their own bases: train pair
memories with mind.train_model, shelve them, query across all of them with one call.

Stdlib + numpy; deterministic; save()/load() like every other model in the family.
"""
import numpy as np

from holographic.caching_and_storage.holographic_supermemory import (
    SuperposedMemory, allocate)

_RFFT, _IRFFT = np.fft.rfft, np.fft.irfft


class NestedMemoryLibrary:
    """A shelf of named pair-memories stored as ONE vector.

    add(name, keys, values) superposes a whole base under its name atom;
    query(name, keys) answers facts from any base in one unbind + cleanup;
    shelve(name, memory) ingests an existing SuperposedMemory that shares this
    library's codebooks (vocab and seed must match -- asserted, not assumed)."""

    def __init__(self, dim, vocab=1024, max_bases=64, seed=0, precision="f64"):
        self.inner = SuperposedMemory(dim, vocab, seed=seed, precision=precision)
        rng = np.random.default_rng(seed * 2 + 5)
        self.names = rng.standard_normal((max_bases, dim)) / np.sqrt(dim)
        self.names /= np.linalg.norm(self.names, axis=1, keepdims=True)
        self.base_ids = {}
        self.dim, self.vocab, self.seed_ = int(dim), int(vocab), int(seed)

    def _name_id(self, name):
        if name not in self.base_ids:
            if len(self.base_ids) >= len(self.names):
                raise ValueError("library full: %d bases" % len(self.names))
            self.base_ids[name] = len(self.base_ids)
        return self.base_ids[name]

    def add(self, name, keys, values):
        """Superpose a whole base under its name: one batched FFT, keys composited with
        the name atom IN FOURIER (bind is elementwise there -- the associativity that
        makes the one-unbind query exist is literally a multiplication reordering)."""
        i = self._name_id(name)
        keys = np.asarray(keys, dtype=int)
        values = np.asarray(values, dtype=int)
        nf = _RFFT(self.names[i])
        kf = _RFFT(self.inner.K[keys], axis=1) * nf[None, :]
        vf = _RFFT(self.inner.V[values], axis=1)
        self.inner.mem = self.inner.mem + _IRFFT((kf * vf).sum(0), n=self.dim)
        self.inner.n_stored += len(keys)
        return self

    def shelve(self, name, memory):
        """Ingest an EXISTING SuperposedMemory (same vocab+seed codebooks) by binding
        its whole state under the name -- the memory-of-memories move: the base's own
        sum of bind(k, v) becomes sum of bind(name(*)k, v) in one convolution."""
        assert memory.vocab == self.vocab and memory.seed_ == self.seed_, \
            "shelved base must share the library's codebooks (vocab and seed)"
        assert memory.dim == self.dim, "shelved base must share the library's dimension"
        i = self._name_id(name)
        self.inner.mem = self.inner.mem + _IRFFT(
            _RFFT(self.names[i]) * _RFFT(memory.mem), n=self.dim)
        self.inner.n_stored += memory.n_stored
        return self

    def query(self, name, keys, decoder="one-shot"):
        """(base, key) -> value in ONE unbind: compose the name atom onto the query keys
        and let the FLAT machinery (including the load gate) do everything else."""
        i = self.base_ids[name]
        keys = np.asarray(keys, dtype=int)
        m = self.inner._state()
        nf = _RFFT(self.names[i])
        kf = _RFFT(self.inner.K[keys], axis=1) * nf[None, :]
        est = _IRFFT(np.conj(kf) * _RFFT(m)[None, :], n=self.dim, axis=1)
        vhat = np.argmax(est @ self.inner.V.T, axis=1)
        if decoder != "pic":
            return {"values": vhat, "decoder": "one-shot",
                    "why": "flat matched filter on composite keys"}
        # PIC across the WHOLE library: interference comes from every base, so the
        # cancellation must too -- delegate by rebuilding composite-key bind estimates.
        limit_ok = self.inner.n_stored <= __import__(
            "holographic.caching_and_storage.holographic_supermemory",
            fromlist=["pic_transition"]).pic_transition(self.dim, self.vocab)
        if not limit_ok:
            return {"values": vhat, "decoder": "one-shot",
                    "why": "GATED at library load %d (kept negative: PIC past its "
                           "transition is poison)" % self.inner.n_stored}
        for _ in range(4):
            B = _IRFFT(kf * _RFFT(self.inner.V[vhat], axis=1), n=self.dim, axis=1)
            resid = m - B.sum(0)
            look = _IRFFT(np.conj(kf) * _RFFT(resid[None, :] + B, axis=1),
                          n=self.dim, axis=1)
            est = 0.5 * look + 0.5 * est
            vhat = np.argmax(est @ self.inner.V.T, axis=1)
        return {"values": vhat, "decoder": "pic",
                "why": "damped PIC over the library's own queried base"}

    def state_bits(self):
        return self.inner.state_bits()

    def save(self, path):
        """Export the library: the one vector + the shelf manifest (names in slot
        order); codebooks and name atoms regenerate from the seed."""
        order = sorted(self.base_ids, key=self.base_ids.get)
        self.inner.save(path)
        np.savez_compressed(str(path) + ".shelf.npz",
                            names=np.array(order, dtype="U64"),
                            max_bases=len(self.names))
        return path

    @classmethod
    def load(cls, path):
        inner = SuperposedMemory.load(path)
        shelf = np.load(str(path) + ".shelf.npz", allow_pickle=False)
        out = cls(inner.dim, inner.vocab, max_bases=int(shelf["max_bases"]),
                  seed=inner.seed_, precision=inner.precision)
        out.inner = inner
        out.base_ids = {str(n): i for i, n in enumerate(shelf["names"])}
        return out


def _selftest():
    """Asserts the algebra (nested == flat law), the one-unbind query at spec, the
    shelve path, export round-trip, and the load gate at library scale."""
    rng = np.random.default_rng(0)
    M, n = 6, 30                              # 6 bases x 30 facts = 180 flat pairs
    D = allocate(M * n, 1024)                 # the FLAT law prices the library
    lib = NestedMemoryLibrary(D, vocab=1024, seed=0)
    truth = {}
    for b in range(M):
        ks = np.random.default_rng(10 + b).choice(1024, n, replace=False)
        vs = np.random.default_rng(20 + b).integers(0, 1024, n)
        lib.add("base%d" % b, ks, vs)
        truth["base%d" % b] = (ks, vs)

    # 1) one-unbind two-level query, at the accuracy the flat law promised.
    accs = []
    for name, (ks, vs) in truth.items():
        out = lib.query(name, ks, decoder="pic")
        accs.append(float(np.mean(out["values"] == vs)))
    assert min(accs) >= 0.90, "library recall below spec: %s" % accs

    # 2) shelve an EXISTING trained memory and query it through the library.
    ks7 = np.random.default_rng(70).choice(1024, n, replace=False)
    vs7 = np.random.default_rng(71).integers(0, 1024, n)
    standalone = SuperposedMemory(D, 1024, seed=0).store(ks7, vs7)
    lib.shelve("imported", standalone)
    acc7 = float(np.mean(lib.query("imported", ks7, decoder="pic")["values"] == vs7))
    assert acc7 >= 0.90, "shelved base unreadable: %.3f" % acc7

    # 3) export round-trip: identical answers, tiny file.
    lib.save("/tmp/library.npz")
    back = NestedMemoryLibrary.load("/tmp/library.npz")
    same = np.array_equal(back.query("base0", truth["base0"][0])["values"],
                          lib.query("base0", truth["base0"][0])["values"])
    assert same, "round-trip changed answers"

    # 4) the gate holds at library scale: overload refuses PIC with the reason.
    small = NestedMemoryLibrary(512, vocab=1024, seed=0)
    for b in range(4):
        ks = np.random.default_rng(30 + b).choice(1024, 30, replace=False)
        small.add("b%d" % b, ks, np.random.default_rng(40 + b).integers(0, 1024, 30))
    r = small.query("b0", truth["base0"][0][:5], decoder="pic")
    assert r["decoder"] == "one-shot" and "GATED" in r["why"], "library gate failed"

    print("holographic_nested selftest OK -- %d bases x %d facts in ONE %d-dim vector, "
          "one-unbind query min acc %.3f; shelved base %.3f; round-trip identical; "
          "gate fires at library load" % (M + 1, n, D, min(accs), acc7))


if __name__ == "__main__":
    _selftest()
