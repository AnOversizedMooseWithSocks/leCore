"""holographic_hypervector.py -- the first-class HYPERVECTOR datatype (consolidation backlog D1).

WHY THIS EXISTS
---------------
The whole engine has been operating on one datatype implicitly: a high-dimensional numpy vector that carries meaning.
Everywhere, that vector travels as a bare `np.ndarray` -- so its dimension, which encoder made it, and "what it is"
all live in the caller's head instead of on the object. `Hypervector` gives that datatype a name and a home, WITHOUT
changing anything: it is a THIN wrapper -- the raw array is always one attribute away (`.array`, `.raw()`, or
`np.asarray(hv)` with no copy), because the hot paths work on the array directly and must never pay for the wrapper.

    MAKE  (the 'make' side is the ENCODERS -- they are the constructors):
        Hypervector.wrap(array, tag=...)            -- attach metadata to an existing vector
        Hypervector.encode(encoder, value)          -- build one FROM DATA via any encoder (scalar/text/record/FPE/
                                                       UniversalEncoder) or a plain callable
    CONSUME  (the five VSA VERBS, as methods, each returning a Hypervector):
        hv.bind(other)     -- associate / rigid-transform  (circular convolution)
        hv.unbind(other)   -- query / detach               (correlation)
        hv.bundle(*others) -- superpose into a set/memory  (normalized sum)
        hv.permute(shift)  -- order / protect              (cyclic shift)
        hv.cleanup(book)   -- recognize / denoise          (snap to the nearest atom in a codebook)
    READ  (readouts):
        hv.cosine(other)   -- similarity                   hv.decode(encoder) -- back to a value, if decodable

The verbs accept a Hypervector OR a raw array on the other side, so the wrapper mixes freely with existing code.
"""
import numpy as np


def _arr(x):
    """The raw array of `x` -- x.array if it's a Hypervector, else np.asarray(x). Lets the verbs take either, so
    the wrapper drops into code that still passes bare arrays."""
    return x.array if isinstance(x, Hypervector) else np.asarray(x, float)


def _short(value):
    """A short 'what am I' tag from an encoded value (for readability, not identity)."""
    s = repr(value)
    return s if len(s) <= 40 else s[:37] + "..."


class Hypervector:
    """A high-dimensional vector that carries meaning: the raw array + its dim + which encoder made it + a tag,
    with the five VSA verbs as methods. Thin -- the raw array is never hidden."""

    __slots__ = ("array", "dim", "encoder", "tag")

    def __init__(self, array, encoder=None, tag=None):
        self.array = np.asarray(array, float)     # the raw vector -- always directly accessible
        self.dim = int(self.array.shape[-1])      # its dimension
        self.encoder = encoder                    # which encoder made it (or None)
        self.tag = tag                            # a human-readable "what am I" label (or None)

    # --- MAKE: constructors (the encoders are the 'make' side) ---
    @classmethod
    def wrap(cls, array, encoder=None, tag=None):
        """Attach metadata to an EXISTING raw vector -- no math, just give the array a name and a home."""
        return cls(array, encoder=encoder, tag=tag)

    @classmethod
    def encode(cls, encoder, value, tag=None):
        """Build a hypervector FROM DATA using any encoder -- a UniversalEncoder, a scalar/text/record/FPE encoder
        (anything with `.encode(value)`), or a plain callable. This is the 'make' side: encoders are the constructors.
        """
        enc_fn = encoder.encode if hasattr(encoder, "encode") else encoder
        vec = enc_fn(value)
        return cls(vec, encoder=encoder, tag=tag if tag is not None else _short(value))

    # --- CONSUME: the five verbs (each returns a Hypervector) ---
    def bind(self, other):
        """Bind -- associate / rigidly transform. Circular convolution of self with `other`. Invertible by unbind."""
        from holographic_ai import bind
        return Hypervector(bind(self.array, _arr(other)), tag=self._tag2(other, "*"))

    def unbind(self, other):
        """Unbind -- query / detach. Correlation: recover what was bound into self with `other`."""
        from holographic_ai import unbind
        return Hypervector(unbind(self.array, _arr(other)), tag=self._tag2(other, "/"))

    def bundle(self, *others):
        """Bundle -- superpose into a set / memory. Normalized sum of self and `others` (order-independent)."""
        from holographic_ai import bundle
        mat = np.stack([self.array] + [_arr(o) for o in others])
        return Hypervector(bundle(mat), tag=self._tagN(others, "+"))

    def permute(self, shift=1):
        """Permute -- order / protection. A cyclic shift by `shift` (its inverse is permute(-shift))."""
        from holographic_ai import permute
        tag = None if self.tag is None else ("perm(%s,%d)" % (self.tag, int(shift)))
        return Hypervector(permute(self.array, int(shift)), encoder=self.encoder, tag=tag)

    def cleanup(self, codebook):
        """Cleanup -- recognize / denoise. Snap to the nearest atom in `codebook`, which may be a Vocabulary, a
        dict {tag: vector/Hypervector}, or an (N, dim) array. Returns the cleaned Hypervector (tag = winning atom)."""
        from holographic_ai import nearest, Vocabulary
        if isinstance(codebook, Vocabulary):
            name, _sim = codebook.cleanup(self.array)
            return Hypervector(codebook.vectors[name], tag=str(name))
        if isinstance(codebook, dict):
            names = list(codebook.keys())
            mat = np.stack([_arr(codebook[k]) for k in names])
            i, _ = nearest(self.array, mat)
            return Hypervector(mat[i], encoder=self.encoder, tag=str(names[i]))
        mat = np.asarray(codebook, float)
        i, _ = nearest(self.array, mat)
        return Hypervector(mat[i], encoder=self.encoder, tag="atom%d" % i)

    # --- READ: readouts ---
    def cosine(self, other):
        """Cosine similarity to another hypervector or raw array."""
        from holographic_ai import cosine
        return float(cosine(self.array, _arr(other)))

    def decode(self, encoder=None):
        """Decode back to a value via the encoder that made it (or a supplied one), if that encoder can decode.
        Returns None if there is no decoder."""
        enc = encoder if encoder is not None else self.encoder
        dec = getattr(enc, "decode", None)
        return dec(self.array) if callable(dec) else None

    # --- keep the raw array CHEAP: the thin-wrapper promise (hot paths must not pay for the wrapper) ---
    def raw(self):
        """The underlying numpy array, no copy."""
        return self.array

    def __array__(self, dtype=None):
        """np.asarray(hv) returns the raw array (no copy), so a Hypervector drops straight into any numpy call."""
        return self.array if dtype is None else self.array.astype(dtype)

    def __len__(self):
        return self.dim

    def __repr__(self):
        return "Hypervector(dim=%d, tag=%r%s)" % (self.dim, self.tag,
                                                  ", encoded" if self.encoder is not None else "")

    # --- tiny tag helpers (readability only; tags never affect the math) ---
    def _tag2(self, other, op):
        o = other.tag if isinstance(other, Hypervector) else None
        if self.tag is None and o is None:
            return None
        return "(%s%s%s)" % (self.tag, op, o)

    def _tagN(self, others, op):
        tags = [self.tag] + [o.tag if isinstance(o, Hypervector) else None for o in others]
        tags = [t for t in tags if t is not None]
        return op.join(tags) if tags else None


def _selftest():
    from holographic_encoders import ScalarEncoder
    from holographic_ai import bind, unbind, bundle, permute, cosine

    enc = ScalarEncoder(dim=1024, seed=0)

    # MAKE from data via an encoder (the 'make' side)
    a = Hypervector.encode(enc, 0.3, tag="a")
    b = Hypervector.encode(enc, 0.7, tag="b")
    assert a.dim == 1024 and a.encoder is enc and a.tag == "a"

    # the raw array is CHEAP to get back (no copy) -- the thin-wrapper promise
    assert a.raw() is a.array
    assert np.asarray(a) is a.array                              # np.asarray(hv) -> the raw array, no copy

    # CONSUME: the five verbs as methods, each matching the bare-array op exactly
    assert np.array_equal(a.bind(b).array, bind(a.array, b.array))
    assert np.array_equal(a.bind(b).unbind(b).array, unbind(bind(a.array, b.array), b.array))
    assert np.array_equal(a.bundle(b).array, bundle(np.stack([a.array, b.array])))
    assert np.array_equal(a.permute(3).array, permute(a.array, 3))

    # verbs accept a raw array on the other side too (mixes with existing code)
    assert np.array_equal(a.bind(b.array).array, bind(a.array, b.array))

    # cleanup snaps to the nearest atom in a codebook (dict form)
    book = {"a": a, "b": b}
    noisy = Hypervector.wrap(a.array + 0.05 * np.random.default_rng(0).standard_normal(1024))
    assert noisy.cleanup(book).tag == "a"                       # recognized as 'a'

    # READ: cosine + a bind/unbind round-trip recovers b better than chance
    rec = a.bind(b).unbind(a)
    assert rec.cosine(b) > cosine(a.array, b.array)             # unbind recovered b, not a

    print("OK: holographic_hypervector self-test passed (encode from data; five verbs as methods match the bare ops; "
          "raw array returned with no copy; cleanup recognizes 'a'; cosine round-trip recovers b) -- %r" % a)


if __name__ == "__main__":
    _selftest()
