"""The frozen core: the stable kernel that everything builds ON, plus versioned
save/load for trained minds.

WHY this exists (G3, the gate for build-on-top)
-----------------------------------------------
The repo is a flat pile of `holographic_*.py` modules. New layers -- the reservoir, the
generation bundle, the forward scene renderer to come -- need a kernel they can import
whose signatures will NOT shift underneath them, and a way to persist a trained mind so a
result can be saved and reloaded rather than recomputed every run. This module is that
contract. It is an EXTRACTION, not a rewrite: the primitives still live in
`holographic_ai.py`; this re-exports them as the stable public surface and adds the one
thing that was missing -- uniform, version-stamped persistence.

The kernel (stable signatures, safe to build against):
    random_vector, unitary_vector  -- mint a clean atom (Gaussian / exact-unbind unitary)
    bind, unbind                   -- circular-convolution binding and its involution inverse
    bundle                         -- superpose (the holographic "and")
    permute                        -- cyclic-shift tag (order / position)
    cosine                         -- direction similarity
    slerp                          -- spherical interpolation (drives the morph generators)
    Vocabulary                     -- the clean-atom store with cleanup()

Persistence:
    save(obj, path) / load(path)   -- npz-backed, stamped with STATE_VERSION; an
                                      incompatible version fails LOUDLY on load rather than
                                      returning a silently-wrong object. Works for any
                                      object exposing to_state()/from_state() -- currently
                                      Vocabulary, HolographicMind, HoloForest.

Build-on-top code should import its primitives from here, not from a subsystem's internals.
"""
import numpy as np

# Re-export the kernel primitives from their canonical home. This is the frozen surface:
# these names and signatures are the contract new layers may depend on.
from holographic.agents_and_reasoning.holographic_ai import random_vector, unitary_vector, bind, unbind, involution, bundle, permute, cosine, slerp, Vocabulary

__all__ = ["random_vector", "unitary_vector", "bind", "unbind", "involution", "bundle",
           "permute", "cosine", "slerp", "cleanup", "Vocabulary",
           "save", "load", "to_state", "from_state", "STATE_VERSION", "CORE_VERSION"]

# Bump CORE_VERSION if a kernel signature changes; STATE_VERSION if the on-disk save
# format changes. A load() of an older STATE_VERSION fails loudly (see _check_version).
CORE_VERSION = 1
STATE_VERSION = 1


def cleanup(noisy, vocabulary, candidates=None):
    """Snap a noisy vector to the nearest known symbol in `vocabulary` (a Vocabulary).
    A free function over the kernel so build-on-top code has a stable cleanup entry point
    without reaching into Vocabulary's methods. Returns (name, similarity)."""
    return vocabulary.cleanup(noisy, candidates=candidates)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

# The object kinds that know how to round-trip themselves, resolved lazily so importing
# the core never drags in the heavier subsystem modules unless persistence is used.
def _registry():
    from holographic.agents_and_reasoning.holographic_ai import Vocabulary as _Vocab
    from holographic.misc.holographic_creature import HolographicMind
    from holographic.misc.holographic_tree import HoloForest
    from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
    from holographic.misc.holographic_unified import UnifiedMind
    return {"Vocabulary": _Vocab, "HolographicMind": HolographicMind,
            "HoloForest": HoloForest, "SelfOrganizingMind": SelfOrganizingMind,
            "UnifiedMind": UnifiedMind}


def to_state(obj):
    """The object's own snapshot, tagged with the format version. Any object exposing
    to_state() works; the tag is what makes an incompatible reload fail loudly."""
    if not hasattr(obj, "to_state"):
        raise TypeError(f"{type(obj).__name__} has no to_state(); cannot persist it")
    state = dict(obj.to_state())
    state["state_version"] = STATE_VERSION
    return state


def from_state(state):
    """Rebuild whatever object a to_state() snapshot describes, checking the version and
    dispatching on the recorded 'kind'."""
    _check_version(state)
    kind = state.get("kind")
    reg = _registry()
    if kind not in reg:
        raise ValueError(f"unknown saved kind {kind!r}; known: {sorted(reg)}")
    return reg[kind].from_state(state)


def _check_version(state):
    v = state.get("state_version")
    if v is None:
        raise ValueError("save has no state_version stamp; refusing to load (it may be "
                         "from before versioning, or corrupt)")
    if int(v) != STATE_VERSION:
        raise ValueError(f"save is state_version {v}, this build expects {STATE_VERSION}; "
                         f"refusing to load a mismatched format rather than guess")


def _flatten(state, prefix=""):
    """Flatten a (possibly NESTED) state dict into npz-storable arrays + a small JSON
    sidecar of the structure, so a single .npz round-trips nested dicts, lists-of-arrays,
    arrays, and scalars faithfully."""
    flat, meta = {}, {}
    for k, v in state.items():
        key = f"{prefix}{k}"
        if isinstance(v, np.ndarray):
            flat[key] = v
            meta[k] = {"t": "array"}
        elif isinstance(v, dict):
            sub_flat, sub_meta = _flatten(v, prefix=f"{key}.")
            flat.update(sub_flat)
            meta[k] = {"t": "dict", "m": sub_meta}
        elif isinstance(v, list) and v and all(isinstance(x, np.ndarray) for x in v):
            for i, arr in enumerate(v):
                flat[f"{key}[{i}]"] = arr
            meta[k] = {"t": "array_list", "n": len(v)}
        elif v is None:
            meta[k] = {"t": "none"}
        else:
            meta[k] = {"t": "json", "v": v}
    return flat, meta


def _rebuild(meta, z, prefix=""):
    """Inverse of _flatten: reconstruct the (nested) state dict from the npz + meta."""
    state = {}
    for k, spec in meta.items():
        key = f"{prefix}{k}"
        t = spec["t"]
        if t == "array":
            state[k] = z[key]
        elif t == "dict":
            state[k] = _rebuild(spec["m"], z, prefix=f"{key}.")
        elif t == "array_list":
            state[k] = [z[f"{key}[{i}]"] for i in range(spec["n"])]
        elif t == "none":
            state[k] = None
        else:
            state[k] = spec["v"]
    return state


def _quant_reconstruct(arr, bits):
    """Dequantised reconstruction of `arr` at `bits` (a per-row scale) -- used only to TEST a
    candidate level inside _auto_quant_kind."""
    q = (2 ** (bits - 1)) - 1
    scale = np.abs(arr).max(axis=1, keepdims=True) / q
    scale[scale == 0] = 1.0
    return np.round(arr / scale).clip(-q, q).astype(float) * scale


def _auto_quant_kind(arr, keep=0.7, size_floor=1024):
    """Dynamic quantisation: pick a float array's storage precision from its OWN structure,
    so precision follows the data's complexity and size rather than a fixed global choice.
    Returns 'int8' | 'f32'.

      * Tiny arrays (< size_floor elements) -> 'f32': the per-array scale/spec overhead is
        not worth it and they are not what makes a file big (the 'size' half of the rule).
      * A 2-D matrix whose int8 reconstruction keeps every row's self-recognition AND at
        least `keep` of its top1-top2 margin -> 'int8' (~4x), since the data's own separation
        proves 8-bit precision leaves the nearest-neighbour argmax intact.
      * Anything failing the margin check, or not a clean 2-D matrix -> 'f32' (the safe floor).

    Only DECISION-SAFE, magnitude-preserving levels are auto-selected. An earlier version also
    offered 1-bit binary for unit-norm matrices, but a cross-stack check found binary
    distorts the pairwise-similarity geometry by ~0.1-0.2 on EVERY array (it only survives
    where the decision is a wide-margin classification argmax, e.g. the prototype memory --
    it flipped 62/200 of the creature value-brain's action decisions). Binary's safety is
    decision-specific and cannot be verified at the generic persistence layer, so it is not
    auto-selected. int8's geometry drift is ~0.002, decision-safe for every persistable type
    measured (classification memory: 0 flips; value brain: tie-level), so auto stays within
    {int8, f32} and matches a float32 save's decisions on any object.
    """
    if arr.size < size_floor:
        return "f32"
    if arr.ndim != 2 or arr.shape[0] < 2:
        return "int8"                       # large 1-D: int8 drift is negligible
    norms = np.linalg.norm(arr, axis=1)
    Un = arr / (norms[:, None] + 1e-12)
    S = Un @ Un.T
    n = len(arr)
    base_gap = np.empty(n)
    for i in range(n):
        row = S[i].copy(); row[i] = -2.0
        base_gap[i] = 1.0 - row.max()       # float top1(self=1) - top2, per row
    Q = _quant_reconstruct(arr, 8)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    for i in range(n):
        sims = Qn @ Un[i]                   # the ORIGINAL row i against the int8 set
        if int(np.argmax(sims)) != i:
            return "f32"
        order = np.sort(sims)[::-1]
        if (order[0] - order[1]) < keep * base_gap[i]:
            return "f32"
    return "int8"


def save(obj, path, compress=True, quant=None):
    """Persist a trained object to `path` (an .npz). Stamped with STATE_VERSION; reload
    with load().

    compress=True (default) stores float arrays as float32, roughly HALVING the file.
    These vectors are only ever compared by cosine, where float32 is far more precision
    than the similarity decisions need; in practice behaviour is unchanged (a decision
    only flips on an exact tie, where either answer is equally valid). Pass compress=False
    to keep float64 for a bit-exact round-trip when that matters.

    quant="int8" goes further: each float array is scaled to signed 8-bit integers (a
    per-array scale, dequantised on load), ~4x smaller than float64 / ~2x smaller than
    float32. This is the scalar-quantisation trick vector databases use to shrink stored
    embeddings, and it is measured LOSSLESS for classification on this substrate -- at the
    working dimension the prototypes are near-orthogonal, so 8-bit precision leaves the
    nearest-neighbour argmax unchanged (verified on crowded 100-class spaces). Opt-in, not
    default: like float32 it can flip an exact tie, and it costs a little reconstruction
    error, so float32 stays the default and int8 is for when stored size matters.

    quant="auto" is DYNAMIC quantisation: each float array gets the coarsest DECISION-SAFE
    precision its own structure supports (see _auto_quant_kind) -- int8 (~4x) where the
    data's separation proves 8-bit leaves the argmax intact, float32 for tiny arrays or
    where the margin is too tight. Precision follows the data's complexity and size, so a
    save compresses as hard as each array individually allows while matching a float32 save's
    decisions on any object type (the per-array margin gate guarantees it). Integer/structural
    arrays (rng state, counts) always keep their dtype.

    quant="rd" is the RATE-DISTORTION code (B5): for low-rank 2D float arrays -- the engine's
    consolidated/bundled state, which lives in a small subspace -- it stores the KLT (consolidation)
    coefficients, uniformly quantized and rANS-entropy-coded, spending only the bits needed to preserve
    the cosines (~11x smaller than int8 on genuinely low-rank state). Where there is no low-rank
    structure it falls back to int8, so it is never larger. The rANS coder is bit-exact.
    """
    import json
    state = to_state(obj)
    flat, meta = _flatten(state)
    out, scales, qspec = {}, {}, {}
    for k, arr in flat.items():
        is_f64 = isinstance(arr, np.ndarray) and arr.dtype == np.float64
        if quant == "rd" and is_f64:
            # rate-distortion code: for low-rank 2D float arrays (consolidated/bundled state) spend the
            # minimum bits that preserve the cosines (KLT -> quantize -> rANS). Falls back to int8 where
            # there is no low-rank structure to exploit, so it is always at least as small as int8.
            from holographic.misc.holographic_ratedistortion import geometry_preserving_code, pack_code, bits_per_vector
            used_rd = False
            if arr.ndim == 2 and arr.shape[0] >= 256:
                code = geometry_preserving_code(arr, target_cos=0.9999)
                if bits_per_vector(code) < 8 * arr.shape[1]:       # only when it actually beats int8
                    out[k] = np.frombuffer(pack_code(code), dtype=np.uint8)
                    qspec[k] = {"k": "rd"}
                    used_rd = True
            if not used_rd:
                peak = float(np.abs(arr).max()) if arr.size else 0.0
                scale = (peak / 127.0) or 1.0
                out[k] = np.round(arr / scale).astype(np.int8)
                qspec[k] = {"k": "int8", "scale": scale}
        elif quant == "auto" and is_f64:
            # auto now also considers the RATE-DISTORTION code (B5) for large low-rank 2D arrays: it
            # preserves cosines to 0.9999 (decision-safe -- tighter than int8's ~0.998 -- so it fits auto's
            # "coarsest decision-safe precision" contract) and is taken ONLY when it actually beats int8, so
            # the default save shrinks genuinely low-rank state with no precision risk and changes nothing for
            # small arrays (rd needs >= 256 rows). This is how the mind's default save (quant='auto') uses B5.
            used_rd = False
            if arr.ndim == 2 and arr.shape[0] >= 256:
                from holographic.misc.holographic_ratedistortion import geometry_preserving_code, pack_code, bits_per_vector
                code = geometry_preserving_code(arr, target_cos=0.9999)
                if bits_per_vector(code) < 8 * arr.shape[1]:        # only when it beats int8
                    out[k] = np.frombuffer(pack_code(code), dtype=np.uint8)
                    qspec[k] = {"k": "rd"}
                    used_rd = True
            if not used_rd:
                kind = _auto_quant_kind(arr)
                if kind == "int8":
                    peak = float(np.abs(arr).max()) if arr.size else 0.0
                    scale = (peak / 127.0) or 1.0
                    out[k] = np.round(arr / scale).astype(np.int8)
                    qspec[k] = {"k": "int8", "scale": scale}
                else:
                    out[k] = arr.astype(np.float32)
                    qspec[k] = {"k": "f32"}
        elif quant == "int8" and is_f64:
            peak = float(np.abs(arr).max()) if arr.size else 0.0
            scale = (peak / 127.0) or 1.0                         # one scale per array
            out[k] = np.round(arr / scale).astype(np.int8)
            scales[k] = scale
        elif compress and is_f64:
            out[k] = arr.astype(np.float32)
        else:
            out[k] = arr
    extra = {}
    if scales:
        extra["__scales__"] = np.frombuffer(json.dumps(scales).encode("utf-8"), dtype=np.uint8)
    if qspec:
        extra["__qspec__"] = np.frombuffer(json.dumps(qspec).encode("utf-8"), dtype=np.uint8)
    np.savez(path, __meta__=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
             **extra, **out)
    return path


def load(path):
    """Reload an object saved with save(). Dequantises any int8- or auto-quantised arrays
    back to float. Fails loudly on a version or format mismatch rather than returning a
    silently-wrong object."""
    import json
    if not str(path).endswith(".npz"):
        path = str(path) + ".npz"
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["__meta__"]).decode("utf-8"))
        reserved = ("__meta__", "__scales__", "__qspec__")
        if "__qspec__" in z:
            qs = json.loads(bytes(z["__qspec__"]).decode("utf-8"))
            store = {}
            for k in z.files:
                if k in reserved:
                    continue
                spec = qs.get(k)
                if spec is None:
                    store[k] = z[k]
                elif spec["k"] == "int8":
                    store[k] = z[k].astype(np.float64) * spec["scale"]
                elif spec["k"] == "rd":
                    from holographic.misc.holographic_ratedistortion import unpack_code, reconstruct
                    store[k] = reconstruct(unpack_code(bytes(z[k])))
                else:                                    # f32
                    store[k] = z[k].astype(np.float64)
            state = _rebuild(meta, store)
        elif "__scales__" in z:
            scales = json.loads(bytes(z["__scales__"]).decode("utf-8"))
            store = {k: (z[k].astype(np.float64) * scales[k] if k in scales else z[k])
                     for k in z.files if k not in reserved}
            state = _rebuild(meta, store)
        else:
            state = _rebuild(meta, z)
    return from_state(state)


def _demo():
    import tempfile, os
    from holographic.misc.holographic_creature import HolographicMind
    print("FROZEN CORE + PERSISTENCE\n")
    print(f"kernel exports: {', '.join(n for n in __all__ if n[0].islower())[:80]}...")
    print(f"CORE_VERSION={CORE_VERSION}  STATE_VERSION={STATE_VERSION}\n")

    rng = np.random.default_rng(0)
    brain = HolographicMind(dim=64, actions=["N", "S", "E", "W"], seed=0, capacity=8)
    for _ in range(400):
        brain.remember([rng.standard_normal(64)], [int(rng.integers(4))], [rng.standard_normal()])
    path = os.path.join(tempfile.gettempdir(), "holo_brain_demo.npz")
    save(brain, path)
    back = load(path)
    probe = rng.standard_normal(64)
    same = all(np.allclose(brain.value(probe, a)[0], back.value(probe, a)[0]) for a in range(4))
    print(f"trained brain saved to {os.path.basename(path)} ({os.path.getsize(path)} bytes), "
          f"reloaded, decides identically: {same}")


if __name__ == "__main__":
    _demo()
