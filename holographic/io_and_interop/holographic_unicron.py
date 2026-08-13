"""UNICRON -- consume trained models and read their weights informatively.

WHY this exists: talking to an LLM is the lowest-bandwidth way to understand it.
The weight matrices themselves carry a readable signal: random-matrix theory says an
UNTRAINED layer's singular-value spectrum follows the Marchenko-Pastur bulk, and
TRAINING pushes learned structure OUT of the bulk (spectral outliers) and makes the
tail HEAVY (Martin & Mahoney, "Traditional and Heavy-Tailed Self Regularization in
Neural Network Models", ICML 2019 -- their ESD power-law alpha predicts test accuracy
WITHOUT any data). This module reads those signals with NumPy alone.

What it does, in order:
  load_safetensors / load_model  -- parse model files with stdlib+NumPy only.
        safetensors is (8-byte LE header length)+(JSON header)+(raw tensor bytes):
        no pickle, no torch, no security surface. .npz supported as the native twin.
  spectral_report                -- per-matrix RMT readout: MP bulk edge, outlier
        count/fraction (the learned signal), heavy-tail alpha (Hill), stable rank,
        spacing-ratio regime (delegates to holographic_quantumstats.level_statistics).
  analyze_model                  -- the readout over every 2D weight in a model.
  fingerprint                    -- one hypervector per MODEL: bind(layer-role,
        metric-encoding), bundle across layers. Models become points in FHRR space;
        compare by cosine, compose/ablate by +/- (the HDRIFT model-algebra pattern).
  compare_models                 -- matched-layer metric deltas between two models
        (teacher vs student: is distillation actually copying spectral structure?).

KEPT NEGATIVES (do not reinvent):
  * hash() is banned -- layer-role vectors are seeded from hashlib.sha256 of the
    layer NAME so fingerprints are stable across processes (PYTHONHASHSEED-proof).
  * The MP edge needs the NOISE sigma, not the raw std -- a planted low-rank spike
    inflates np.std(W) and hides its own outliers. We estimate sigma from the
    MEDIAN singular value against the MP median (robust to a few spikes).
  * Hill's alpha on the FULL spectrum is meaningless (the bulk is not a power law);
    it must run on the top tail only (we use the top 10%, min 10 values).
  * torch .pt/.bin files are pickle archives: NOT parsed here, by decision --
    unpickling arbitrary files is an arbitrary-code-execution surface. Convert to
    safetensors/npz upstream. This is NOT_APPLICABLE, not DEFERRED.
"""

import os
import json
import struct
import hashlib
import zipfile

import zlib
import tempfile

import numpy as np

# Delegations -- Rule 0 said these exist; do not reimplement.
from holographic.sampling_and_signal.holographic_quantumstats import level_statistics


# --------------------------------------------------------------------------- loading

# safetensors dtype strings -> (numpy dtype used to read raw bytes, post-decode)
# bf16 has no numpy dtype: read as uint16, shift into the high half of a float32.
_ST_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def _decode_bf16(raw_u16):
    """bfloat16 -> float32 exactly: bf16 IS the top 16 bits of an IEEE float32,
    so a left shift into a uint32 reinterpreted as float32 is a lossless decode."""
    u32 = raw_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def load_safetensors(path, return_dtypes=False):
    """Parse a .safetensors file into {name: ndarray} with stdlib + NumPy only.
    return_dtypes=True additionally returns {name: on-disk dtype string}, so a
    caller can hand it back to save_safetensors and keep the file size honest.

    Format: first 8 bytes = little-endian uint64 length N of the JSON header;
    next N bytes = JSON mapping tensor name -> {dtype, shape, data_offsets};
    the rest = the concatenated raw tensor bytes the offsets index into.
    bf16 tensors are decoded losslessly to float32 (see _decode_bf16)."""
    # MEMORY-MAP THE PAYLOAD, DO NOT READ IT. `blob = f.read()` pulls the whole
    # checkpoint into RAM before a single tensor is touched, which is precisely
    # the anti-pattern safetensors was designed to avoid -- the format exists so
    # the OS can page bytes in on demand rather than duplicating the file.
    # Field-caught on a real 2.1 GB model: the install finished, the file wrote
    # correctly, and reading it back for VERIFICATION died with MemoryError
    # while the installed and original copies were still held.
    # np.memmap is numpy-only, needs no dependency, and gives the same zero-copy
    # behaviour the safetensors library gets from mmap.
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len
    try:
        blob = np.memmap(path, dtype=np.uint8, mode="r", offset=data_start)
    except Exception:
        # a filesystem that cannot map (some network shares) falls back to the
        # old behaviour rather than failing -- slower and hungrier, but correct
        with open(path, "rb") as f:
            f.seek(data_start)
            blob = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        raw = blob[a:b]
        shape = tuple(meta["shape"])
        dt = meta["dtype"]
        # DO NOT MATERIALISE. The previous line ended in `.copy()` on EVERY
        # tensor, which pages in the whole mapping and defeats the memmap that
        # was just set up -- and for a BF16 checkpoint the eager _decode_bf16
        # DOUBLED it again, because bf16 decodes to float32. A 2.1 GB bf16 model
        # therefore cost 4.2 GB before anything was computed.
        # llama.cpp's streaming PR names this exact trap: enabling streaming
        # AUTO-DISABLES mmap because "mmap prefetch would page the whole model
        # into RAM and defeat streaming". A copy is the same defeat.
        # A _LazyTensor holds the OFFSET, not the bytes, and decodes the one
        # tensor a caller actually touches. Every consumer here goes through
        # np.asarray(), which triggers __array__ -- so nothing else changes.
        if dt not in _ST_DTYPES and dt != "BF16":
            raise ValueError("unsupported safetensors dtype: %s" % dt)
        out[name] = _LazyTensor(blob, a, b, shape, dt)
    if return_dtypes:
        return out, {k: header[k]["dtype"] for k in out}
    return out


class _LazyTensor:
    """A tensor that is an OFFSET until someone asks for its values.

    numpy calls __array__ on any np.asarray/np.array, so this behaves as an
    ndarray everywhere in this codebase without a single call site changing.
    `.shape` and `.dtype` answer from the header, so the many places that only
    inspect geometry -- the architecture inference, the layer-type detection,
    the size reports -- never touch a byte of the payload."""

    __slots__ = ("_blob", "_a", "_b", "shape", "_dt", "_cache")

    def __init__(self, blob, a, b, shape, dt):
        self._blob = blob
        self._a = int(a)
        self._b = int(b)
        self.shape = tuple(shape)
        self._dt = dt
        self._cache = None

    @property
    def dtype(self):
        return np.dtype(np.float32) if self._dt == "BF16" \
            else np.dtype(_ST_DTYPES[self._dt])

    @property
    def size(self):
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def ndim(self):
        return len(self.shape)

    def __array__(self, dtype=None, copy=None):
        if self._cache is None:
            raw = bytes(self._blob[self._a:self._b])
            if self._dt == "BF16":
                arr = _decode_bf16(np.frombuffer(raw, dtype=np.uint16))
            else:
                arr = np.frombuffer(raw, dtype=_ST_DTYPES[self._dt])
            self._cache = arr.reshape(self.shape)
        return (self._cache if dtype is None
                else self._cache.astype(dtype, copy=False))

    def __getattr__(self, name):
        # ANYTHING ELSE AN NDARRAY HAS, materialise and delegate. Enumerating
        # the surface by hand fails on the first attribute nobody thought of --
        # this hit `.T` immediately. A lazy value must be INDISTINGUISHABLE from
        # the real one or it is a trap rather than an optimisation.
        # THE UNDERSCORE GUARD IS LOAD-BEARING: without it, __getattr__ is
        # reached for `_cache` before __init__ has set it, calls np.asarray,
        # which reads `_cache`, which calls __getattr__ -- RecursionError in
        # every module at once. A fallback that can invoke itself is not a
        # fallback.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)

    def __getitem__(self, k):
        return np.asarray(self)[k]

    def __len__(self):
        return int(self.shape[0]) if self.shape else 0


def _encode_bf16(f32):
    """float32 -> bfloat16 raw uint16, round-to-nearest-EVEN on the dropped 16 bits
    (plain truncation biases every value toward zero; RNE is what hardware does).
    Values already representable in bf16 round-trip exactly through decode."""
    u32 = np.ascontiguousarray(f32, np.float32).view(np.uint32)
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)


def save_safetensors(path, tensors, dtypes=None):
    """Write {name: ndarray} as .safetensors. `dtypes` maps name -> safetensors
    dtype string ("BF16", "F16", "F32", ...) to OVERRIDE the array's own dtype on
    disk -- the round-trip fidelity fix: our loader decodes BF16 to float32
    losslessly, so without this override a load->save cycle silently DOUBLES the
    file (measured live on Qwen3.5: 2x size, kept negative). Exists so the
    selftest can round-trip WITHOUT any external model file."""
    inv = {v: k for k, v in _ST_DTYPES.items()}
    dtypes = dtypes or {}
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):  # sorted: byte-deterministic output
        arr = np.ascontiguousarray(tensors[name])
        want = dtypes.get(name)
        if want == "BF16":
            raw = _encode_bf16(arr).tobytes()
            dt = "BF16"
        elif want is not None and want in _ST_DTYPES:
            arr = arr.astype(_ST_DTYPES[want])
            raw = arr.tobytes()
            dt = want
        else:
            dt = inv.get(arr.dtype.type)
            if dt is None:
                raise ValueError("unsupported dtype for save: %r" % (arr.dtype,))
            raw = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header, sort_keys=True).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def load_model(path):
    """Front door: .safetensors or .npz -> {name: ndarray}. torch pickle files are
    refused on purpose (arbitrary-code-execution surface; see module negatives)."""
    p = str(path)
    if p.endswith(".safetensors"):
        return load_safetensors(p)
    if p.endswith(".gguf"):
        return load_gguf(p)
    if p.endswith(".npz"):
        with np.load(p) as z:
            return {k: z[k] for k in z.files}
    if p.endswith((".pt", ".bin", ".pth")) or zipfile.is_zipfile(p):
        raise ValueError("torch pickle checkpoints are refused (unpickling is an "
                         "ACE surface); convert to .safetensors or .npz first")
    raise ValueError("unknown model format: %s" % p)


# --------------------------------------------------------------------------- spectra

def spectral_report(W, spacing=False):
    """Random-matrix readout of one weight matrix. Returns a plain dict.

    Signals and WHY each is informative:
      mp_edge        Marchenko-Pastur bulk edge for a pure-noise matrix of this
                     shape and (robustly estimated) noise scale. Anything above it
                     did not come from initialization noise.
      n_outliers /   count and fraction of singular values above the edge --
      outlier_frac   the learned, low-rank signal training injected.
      alpha          Hill estimator of the ESD power-law tail exponent (top 10%).
                     Martin & Mahoney: heavier tail (smaller alpha, ~2-4) tracks
                     better-trained layers; ~6+ looks like noise.
      stable_rank    ||W||_F^2 / ||W||_2^2 -- how spread the energy is.
      regime         (optional, spacing=True) spacing-ratio verdict on the
                     eigenvalues of W W^T via holographic_quantumstats -- Poisson
                     vs GOE-like level repulsion, no unfolding needed.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("spectral_report wants a 2D matrix, got shape %r" % (W.shape,))
    n, m = W.shape
    if n < m:                       # convention: tall matrix, q = m/n <= 1
        n, m = m, n
    sv = np.linalg.svd(W, compute_uv=False)  # descending
    ev = sv * sv                    # eigenvalues of W^T W (the ESD lives here)
    q = m / n
    # Robust noise scale: match the MEDIAN eigenvalue to the MP median instead of
    # using np.std(W) -- KEPT NEGATIVE: raw std is inflated by planted spikes and
    # hides the very outliers we are hunting. MP median has no closed form; a
    # numeric quantile of the MP density is cheap and exact enough.
    grid = np.linspace((1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2, 2001)[1:-1]
    dens = np.sqrt(((1 + np.sqrt(q)) ** 2 - grid) * (grid - (1 - np.sqrt(q)) ** 2)) / (2 * np.pi * q * grid)
    cdf = np.cumsum(dens); cdf /= cdf[-1]
    mp_median_unit = grid[int(np.searchsorted(cdf, 0.5))]
    sigma2 = np.median(ev) / (n * mp_median_unit)
    mp_edge = sigma2 * n * (1 + np.sqrt(q)) ** 2          # eigenvalue-scale edge
    # small tolerance: finite-size fluctuation of the top bulk eigenvalue
    thresh = mp_edge * (1.0 + 3.0 * n ** (-2.0 / 3.0))    # Tracy-Widom width scale
    n_out = int(np.sum(ev > thresh))
    # Hill alpha on the TOP TAIL only (kept negative: full-spectrum Hill is garbage)
    k = max(10, int(0.10 * ev.size))
    k = min(k, ev.size - 1)
    tail = ev[:k]
    alpha = float("nan")
    if tail[-1] > 0 and ev[k] > 0:
        alpha = 1.0 + k / float(np.sum(np.log(tail / ev[k])))
    rep = {
        "shape": (int(W.shape[0]), int(W.shape[1])),
        "spectral_norm": float(sv[0]),
        "fro_norm": float(np.sqrt(ev.sum())),
        "stable_rank": float(ev.sum() / ev[0]) if ev[0] > 0 else 0.0,
        "mp_edge": float(np.sqrt(thresh)),   # reported on the singular-value scale
        "n_outliers": n_out,
        "outlier_frac": float(n_out / ev.size),
        "alpha": float(alpha),
    }
    if spacing and ev.size >= 32:
        stats = level_statistics(np.sort(ev))
        rep["regime"] = stats.get("verdict", stats.get("regime", "?")) \
            if isinstance(stats, dict) else str(stats)
    return rep


def analyze_model(tensors, min_dim=8, spacing=False):
    """Run spectral_report over every >=2D tensor (matrices; higher-rank tensors are
    flattened to (d0, rest) -- the convention conv/attention analyses use). Returns
    {"layers": {name: report}, "summary": {...}} with model-level medians, because a
    single number per model is what fingerprints and comparisons consume."""
    layers = {}
    for name, t in tensors.items():
        t = np.asarray(t)
        if t.ndim < 2 or min(t.shape[0], int(np.prod(t.shape[1:]))) < min_dim:
            continue
        W = t.reshape(t.shape[0], -1)
        layers[name] = spectral_report(W, spacing=spacing)
    if not layers:
        return {"layers": {}, "summary": {}}
    med = lambda key: float(np.median([r[key] for r in layers.values()
                                       if np.isfinite(r[key])]))
    summary = {"n_layers": len(layers),
               "median_alpha": med("alpha"),
               "median_stable_rank": med("stable_rank"),
               "median_outlier_frac": med("outlier_frac"),
               "total_outliers": int(sum(r["n_outliers"] for r in layers.values()))}
    return {"layers": layers, "summary": summary}


# ----------------------------------------------------------------------- fingerprint

def _role_vec(name, dim):
    """Deterministic FHRR role phasor for a layer name. hashlib, never hash():
    the fingerprint must be identical across processes and years."""
    seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(-np.pi, np.pi, dim))


# metric -> (center, scale) for phase encoding; chosen so typical trained-layer
# values land well inside (-pi, pi) without wrapping.
_METRIC_SCALE = {"alpha": (4.0, 4.0), "stable_rank": (0.0, 200.0),
                 "outlier_frac": (0.0, 0.25), "spectral_norm": (0.0, 50.0)}


def _metric_vec(report, dim):
    """Encode a layer report as one phasor vector: each metric gets its own role
    (hash of the metric NAME) and a fractional-power-style phase proportional to
    the normalized value -- similar metrics => similar phases => high cosine.
    WHY not RecordEncoder: that encodes for exact recall; here we want SMOOTH
    similarity in the metric values, which phase-proportional encoding gives."""
    acc = np.zeros(dim, dtype=np.complex128)
    for key, (c, s) in _METRIC_SCALE.items():
        v = report.get(key, float("nan"))
        if not np.isfinite(v):
            continue
        t = np.clip((v - c) / s, -1.0, 1.0)
        base = _role_vec("metric::" + key, dim)
        acc += np.exp(1j * np.angle(base) * t)   # fractional power binding: base^t
    n = np.abs(acc); n[n == 0] = 1.0
    return acc / n


def fingerprint(analysis, dim=1024):
    """One hypervector for a whole model: bundle over layers of
    bind(role(layer name), encode(layer metrics)). Two checkpoints of the SAME
    architecture share roles, so cosine(fingerprint_a, fingerprint_b) reads how
    similar their per-layer spectral structure is -- the distillation question.
    Model algebra applies: fp_teacher - fp_student highlights what training
    changed (the HDRIFT compose/ablate pattern, on models-of-models)."""
    acc = np.zeros(dim, dtype=np.complex128)
    for name, rep in analysis["layers"].items():
        acc += _role_vec(name, dim) * _metric_vec(rep, dim)
    n = np.linalg.norm(acc)
    return acc / n if n > 0 else acc


def cosine(a, b):
    """Real part of the normalized Hermitian inner product -- the FHRR similarity."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.real(np.vdot(a, b)) / (na * nb))


def compare_models(analysis_a, analysis_b):
    """Matched-layer metric deltas (b - a) plus fingerprint cosine. The distillation
    audit: a student copying the teacher's FUNCTION should be drifting toward the
    teacher's spectral structure (alpha down toward it, outliers appearing in the
    same layers) -- if the deltas are noise, the distillation is memorizing, not
    inheriting."""
    la, lb = analysis_a["layers"], analysis_b["layers"]
    common = sorted(set(la) & set(lb))
    deltas = {name: {k: float(lb[name][k] - la[name][k])
                     for k in ("alpha", "stable_rank", "outlier_frac", "spectral_norm")
                     if np.isfinite(la[name].get(k, np.nan))
                     and np.isfinite(lb[name].get(k, np.nan))}
              for name in common}
    fa, fb = fingerprint(analysis_a), fingerprint(analysis_b)
    return {"n_common": len(common), "n_only_a": len(set(la) - set(lb)),
            "n_only_b": len(set(lb) - set(la)),
            "fingerprint_cosine": cosine(fa, fb), "layer_deltas": deltas}




# ------------------------------------------------------------------------------ gguf

# ggml tensor type ids we DEQUANTIZE (llama.cpp convention). Everything else is
# refused BY NAME so the caller knows exactly which quant to convert upstream --
# implementing every k-quant here would be a maintenance tax with no RMT payoff
# (the spectrum of a heavily quantized matrix is the quantizer's, not training's).
_GGML_F32, _GGML_F16, _GGML_Q8_0, _GGML_BF16 = 0, 1, 8, 30
_GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

def _gguf_read_str(f):
    """GGUF string: u64 length + raw utf-8 bytes (no terminator)."""
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")

# kv value readers by GGUF type id; arrays (9) recurse on the element type.
_GGUF_SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
                 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
                 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}

def _gguf_read_value(f, t):
    if t in _GGUF_SCALARS:
        fmt, size = _GGUF_SCALARS[t]
        return struct.unpack(fmt, f.read(size))[0]
    if t == 8:
        return _gguf_read_str(f)
    if t == 9:                               # array: elem type + count + values
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_gguf_read_value(f, et) for _ in range(n)]
    raise ValueError("unknown GGUF kv type: %d" % t)

def _dequant_q8_0(raw, count):
    """Q8_0: blocks of 32 values, each block = f16 scale + 32 int8; value = scale*q.
    Vectorized: view the interleaved block layout with a structured dtype."""
    blk = np.dtype([("d", "<f2"), ("q", "i1", (32,))])
    b = np.frombuffer(raw, dtype=blk, count=(count + 31) // 32)
    out = (b["d"].astype(np.float32)[:, None] * b["q"].astype(np.float32)).reshape(-1)
    return out[:count]

def load_gguf(path):
    """Parse a GGUF file (llama.cpp models) into {name: ndarray} -- stdlib+NumPy.
    Layout: magic+version, tensor_count, kv metadata, tensor infos (name/dims/type/
    offset), then the data section aligned to general.alignment (default 32).
    F32/F16/BF16 read directly; Q8_0 dequantized; other quants refused by name
    (see the WHY-comment above _GGML_F32). NOTE: GGUF stores dims INNERMOST-FIRST
    (ne[0] = fastest-moving), so the numpy shape is the dims REVERSED -- getting
    this backwards silently transposes every matrix and poisons the RMT q ratio."""
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != _GGUF_MAGIC:
            raise ValueError("not a GGUF file (bad magic)")
        if version < 2:
            raise ValueError("GGUF v1 uses u32 counts; only v2+ supported")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        meta = {}
        for _ in range(n_kv):
            key = _gguf_read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            meta[key] = _gguf_read_value(f, t)
        infos = []
        for _ in range(n_tensors):
            name = _gguf_read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            gtype, off = struct.unpack("<IQ", f.read(12))
            infos.append((name, dims, gtype, off))
        align = int(meta.get("general.alignment", 32))
        pos = f.tell()
        data_start = ((pos + align - 1) // align) * align
        f.seek(0, 2); end = f.tell()
        out = {}
        for name, dims, gtype, off in infos:
            count = 1
            for d in dims:
                count *= int(d)
            shape = tuple(int(d) for d in reversed(dims))   # innermost-first!
            f.seek(data_start + off)
            if gtype == _GGML_F32:
                arr = np.frombuffer(f.read(4 * count), dtype=np.float32)
            elif gtype == _GGML_F16:
                arr = np.frombuffer(f.read(2 * count), dtype=np.float16).astype(np.float32)
            elif gtype == _GGML_BF16:
                arr = _decode_bf16(np.frombuffer(f.read(2 * count), dtype=np.uint16))
            elif gtype == _GGML_Q8_0:
                nbytes = ((count + 31) // 32) * 34          # 2 (f16 d) + 32 (i8)
                arr = _dequant_q8_0(f.read(nbytes), count)
            else:
                raise ValueError("GGUF tensor %r has unsupported ggml type %d; "
                                 "convert this quant to f16/f32 upstream" % (name, gtype))
            out[name] = arr.reshape(shape).copy()
        if end < data_start:
            raise ValueError("truncated GGUF data section")
    return out

def save_gguf(path, tensors, quant=None):
    """Minimal GGUF v3 writer (F32, or Q8_0 for names listed in `quant`). Exists for
    the same reason save_safetensors does: the loader's test must round-trip with no
    external download. Not a general exporter -- kv metadata is just alignment."""
    quant = set(quant or ())
    infos, blobs, off = [], [], 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(np.asarray(tensors[name], dtype=np.float32))
        dims = tuple(reversed(arr.shape))                   # innermost-first on disk
        if name in quant:
            flat = arr.reshape(-1)
            pad = (-flat.size) % 32
            fp = np.concatenate([flat, np.zeros(pad, np.float32)]).reshape(-1, 32)
            d = np.max(np.abs(fp), axis=1) / 127.0
            d[d == 0] = 1.0
            q = np.clip(np.round(fp / d[:, None]), -127, 127).astype(np.int8)
            blk = np.empty(fp.shape[0], dtype=np.dtype([("d", "<f2"), ("q", "i1", (32,))]))
            blk["d"] = d.astype(np.float16); blk["q"] = q
            raw, gt = blk.tobytes(), _GGML_Q8_0
        else:
            raw, gt = arr.tobytes(), _GGML_F32
        infos.append((name, dims, gt, off)); blobs.append(raw)
        off += len(raw) + ((-len(raw)) % 32)                # tensor data 32-aligned
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", _GGUF_MAGIC, 3, len(infos), 1))
        k = b"general.alignment"
        f.write(struct.pack("<Q", len(k))); f.write(k)
        f.write(struct.pack("<II", 4, 32))                  # type u32, value 32
        for name, dims, gt, o in infos:
            nb = name.encode("utf-8")
            f.write(struct.pack("<Q", len(nb))); f.write(nb)
            f.write(struct.pack("<I", len(dims)))
            f.write(struct.pack("<%dQ" % len(dims), *dims))
            f.write(struct.pack("<IQ", gt, o))
        pad = (-f.tell()) % 32
        f.write(b"\x00" * pad)
        for raw in blobs:
            f.write(raw)
            f.write(b"\x00" * ((-len(raw)) % 32))

# -------------------------------------------------------------------- subspace overlap

def subspace_overlap(A, B, k=8, side="left"):
    """How much do two matrices' top-k singular SUBSPACES agree? Returns principal-angle
    cosines (descending) and their mean square -- the chordal similarity in [0, 1].

    WHY this beats scalar metrics: two layers can share every spectral statistic and
    still encode in orthogonal directions. The principal angles between span(U_A[:, :k])
    and span(U_B[:, :k]) are the singular values of U_A^T U_B (Bjorck & Golub 1973) --
    exact, no iteration. Calibration: identical subspaces -> 1.0; independent random
    k-subspaces of R^n -> mean cos^2 concentrates at k/n (the chance level to report
    against -- an overlap is only evidence ABOVE that floor)."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    if side == "right":
        A, B = A.T, B.T
    Ua = np.linalg.svd(A, full_matrices=False)[0][:, :k]
    Ub = np.linalg.svd(B, full_matrices=False)[0][:, :k]
    cos = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    n = A.shape[0]
    return {"cosines": cos.tolist(), "overlap": float(np.mean(cos ** 2)),
            "chance": float(min(k, n) / n), "k": int(k)}




# ------------------------------------------------------------- localization / filtering

def vector_localization(W, k=10):
    """WHERE does the learned information live? Porter-Thomas test on singular vectors.

    RMT prediction (Thamm, Staats & Rosenow, Phys. Rev. E 106, 054124): a NOISE
    singular vector has i.i.d.-Gaussian entries -- excess kurtosis 0, inverse
    participation ratio (IPR = sum v_i^4) at 3/n. A LEARNED vector localizes on the
    coordinates that matter: kurtosis and IPR rise above the Gaussian baseline.
    Returns per-vector stats for the top-k left and right singular vectors, each with
    its Gaussian expectation, so the caller reads evidence ABOVE baseline -- the same
    report-against-chance discipline subspace_overlap uses."""
    W = np.asarray(W, np.float64)
    U, sv, Vt = np.linalg.svd(W, full_matrices=False)
    k = min(k, sv.size)
    out = []
    for i in range(k):
        row = {"index": i, "sigma": float(sv[i])}
        for tag, v in (("left", U[:, i]), ("right", Vt[i, :])):
            n = v.size
            v2 = v * v
            row[tag + "_ipr"] = float(np.sum(v2 * v2))
            row[tag + "_ipr_gauss"] = 3.0 / n           # E[sum v^4], v uniform on sphere
            m2 = np.mean(v2)
            row[tag + "_kurtosis"] = float(np.mean(v2 * v2) / (m2 * m2) - 3.0)
        out.append(row)
    return {"vectors": out,
            "n_localized": int(sum(1 for r in out
                                   if r["left_ipr"] > 2.0 * r["left_ipr_gauss"]
                                   or r["right_ipr"] > 2.0 * r["right_ipr_gauss"]))}


def rmt_filter(W, keep=None, mode="truncate"):
    """RMT-guided weight filtering: keep the spectral OUTLIERS (learned signal),
    discard the Marchenko-Pastur bulk (initialization noise that training never
    overwrote -- Thamm/Staats/Rosenow measured that MOST of a trained network's
    spectrum is still random). Staats, Thamm & Rosenow (PRE 108, L022302, 2023)
    show this boundary is the principled noise/information cut.

    keep=None uses the matrix's own MP edge (spectral_report); an int forces a rank.
    mode="truncate" zeroes the bulk; mode="shrink" additionally debiases each kept
    singular value by the noise floor (sqrt(max(s^2 - edge^2, 0)) -- the spiked-model
    correction: an observed spike rides ON the bulk, so its raw value overstates the
    signal). Returns (W_filtered, info). NOT the manifold `denoise` faculty (that
    projects hypervectors onto a learned manifold); NOT Tucker/TT compression (that
    minimizes reconstruction error with no noise model) -- this cut is a NOISE MODEL,
    which is why it can IMPROVE on the raw matrix instead of only approximating it."""
    W = np.asarray(W, np.float64)
    U, sv, Vt = np.linalg.svd(W, full_matrices=False)
    rep = spectral_report(W)
    edge = rep["mp_edge"]
    r = int(keep) if keep is not None else int(np.sum(sv > edge))
    r = max(0, min(r, sv.size))
    s_kept = sv[:r].copy()
    if mode == "shrink":
        s_kept = np.sqrt(np.maximum(s_kept ** 2 - edge ** 2, 0.0))
    Wf = (U[:, :r] * s_kept) @ Vt[:r, :]
    return Wf, {"rank_kept": r, "mp_edge": float(edge), "mode": mode,
                "energy_kept": float(np.sum(sv[:r] ** 2) / max(np.sum(sv ** 2), 1e-300))}


# ----------------------------------------------------------------------- trajectories

def checkpoint_trajectory(analyses, dim=1024):
    """READ A TRAINING RUN: given per-checkpoint analyze_model results (in time
    order), return the model's path through FHRR space -- fingerprint of each
    checkpoint, cosine of each step, cumulative distance from start -- plus
    per-layer metric time-series. Theory anchor: squared singular values under SGD
    follow Dyson Brownian motion toward a bulk+tail stationary state (Olsen et al.,
    arXiv 2507.12709), so a HEALTHY run shows monotone drift away from init that
    decelerates (steps shorten as spectra settle); a step cosine that DROPS mid-run
    marks a regime change worth investigating (lr event, data shift, divergence).
    This function only reports the measurements -- verdicts stay with the caller."""
    fps = [fingerprint(a, dim=dim) for a in analyses]
    step_cos = [cosine(fps[i], fps[i + 1]) for i in range(len(fps) - 1)]
    from_start = [cosine(fps[0], f) for f in fps]
    layers0 = set(analyses[0]["layers"])
    common = sorted(layers0.intersection(*[set(a["layers"]) for a in analyses[1:]])) \
        if len(analyses) > 1 else sorted(layers0)
    series = {name: {k: [float(a["layers"][name][k]) for a in analyses]
                     for k in ("alpha", "stable_rank", "outlier_frac")}
              for name in common}
    return {"n_checkpoints": len(analyses), "step_cosines": step_cos,
            "cosine_from_start": from_start, "layer_series": series,
            "fingerprints": fps}




# ------------------------------------------------------------------------ transformation

def transform_model(tensors, mode="shrink", keep=None, min_dim=8, factored=True, guard=True):
    """UPGRADE a whole model: rmt_filter every weight matrix (keep learned outliers,
    discard the still-random Marchenko-Pastur bulk), and store each filtered layer in
    FACTORED form (U*s, V) when that is smaller than the dense matrix. Returns
    (new_tensors, report) -- report has per-layer rank, parameter counts, and the
    model-level compression ratio.

    THE HONESTY CONTRACT, load-bearing: spectral surgery alone proves NOTHING about
    capability. The claim "smaller and just as capable" is a FUNCTIONAL claim and
    must be measured on the model's task -- which is why functional_retention()
    exists and why the selftest refuses to pass on spectra alone. This function
    reports what it changed; whether the change was an upgrade is a measurement the
    caller owes. Small tensors (min_dim) and 1D params pass through UNTOUCHED --
    biases and norms are cheap and filtering them buys nothing.

    factored=True stores name+".U" (m x r, = U*s) and name+".V" (r x n) instead of
    the dense (m x n) whenever r*(m+n) < m*n -- an ACTUAL size reduction on disk and
    an actual FLOP reduction at inference (two thin matmuls), not just zeroed
    singular values. reconstruct_model() is the exact inverse."""
    new, rep = {}, {"layers": {}, "params_in": 0, "params_out": 0}
    for name, t in tensors.items():
        t = np.asarray(t)
        rep["params_in"] += t.size
        if t.ndim < 2 or min(t.shape[0], int(np.prod(t.shape[1:]))) < min_dim:
            new[name] = t
            rep["params_out"] += t.size
            continue
        W = t.reshape(t.shape[0], -1).astype(np.float64)
        U, sv, Vt = np.linalg.svd(W, full_matrices=False)
        edge = spectral_report(W)["mp_edge"]
        n_out = int(np.sum(sv > edge))
        # THE GUARD, measured into existence: a matrix with (almost) no spectral
        # outliers is not necessarily useless -- random-FEATURE layers (ELM,
        # reservoirs, random projections) are functionally load-bearing while
        # spectrally indistinguishable from noise. Filtering one deletes a working
        # layer (selftest pins the -31-point accuracy collapse). guard=True passes
        # such layers through untouched. Discriminator = outlier ENERGY fraction,
        # never outlier count (count-gating guarded EVERY realistic trained layer
        # in the first Qwen-shaped rehearsal: 0/32 filtered; kept negative).
        _ev = sv * sv
        _spikeE = float(np.sum(_ev[:n_out] - edge ** 2)) if n_out else 0.0
        if guard and keep is None and (n_out == 0 or _spikeE < 0.01 * float(_ev.sum())):
            new[name] = t
            rep["params_out"] += t.size
            rep["layers"][name] = {"rank": int(sv.size), "of": int(sv.size),
                                   "energy_kept": 1.0, "factored": False,
                                   "guarded": True}
            continue
        r = int(keep) if keep is not None else max(1, n_out)
        r = min(r, sv.size)
        s_kept = sv[:r].copy()
        if mode == "shrink":
            s_kept = np.sqrt(np.maximum(s_kept ** 2 - edge ** 2, 0.0))
        m_, n_ = W.shape
        if factored and r * (m_ + n_) < m_ * n_:
            new[name + ".U"] = (U[:, :r] * s_kept).astype(t.dtype)
            new[name + ".V"] = Vt[:r, :].astype(t.dtype)
            p_out = r * (m_ + n_)
        else:
            new[name] = ((U[:, :r] * s_kept) @ Vt[:r, :]).reshape(t.shape).astype(t.dtype)
            p_out = t.size
        rep["params_out"] += p_out
        rep["layers"][name] = {"rank": r, "of": int(sv.size),
                               "energy_kept": float(np.sum(sv[:r] ** 2) / max(np.sum(sv ** 2), 1e-300)),
                               "factored": bool(factored and r * (m_ + n_) < m_ * n_)}
    rep["compression"] = float(rep["params_out"] / max(rep["params_in"], 1))
    return new, rep


def reconstruct_model(tensors):
    """Exact inverse of transform_model's factored storage: every name.U/name.V pair
    multiplies back into a dense `name`; everything else passes through."""
    out, done = {}, set()
    for k in tensors:
        if k.endswith(".U") and k[:-2] + ".V" in tensors:
            base = k[:-2]                     # "w1.weight.U" -> "w1.weight"
            out[base] = np.asarray(tensors[base + ".U"]) @ np.asarray(tensors[base + ".V"])
            done.add(base + ".U"); done.add(base + ".V")
    for k, v in tensors.items():
        if k not in done:
            out.setdefault(k, v)
    return out


def pca_net_train(X, y, hidden=256, k=8, n_classes=None, seed=0, reg=1e-3):
    """Train a small model whose FIRST layer is genuinely learned (no autodiff):
    W1 = A @ P where P = top-k principal directions of the data (learned structure,
    low-rank + spiked -- exactly what rmt filtering preserves) and A is a random
    expansion; readout by ridge. The instrument transform_model's honest test needs:
    a trained matrix the filter should keep, next to elm_train's random matrix the
    filter should not touch."""
    rng = np.random.default_rng(seed)
    n_classes = n_classes or int(np.max(y)) + 1
    Xc = X - X.mean(0)
    P = np.linalg.svd(Xc, full_matrices=False)[2][:k]        # k x d, learned from data
    A = rng.standard_normal((hidden, k)) / np.sqrt(k)
    W1 = A @ P + rng.standard_normal((hidden, X.shape[1])) * 0.01
    b1 = rng.standard_normal(hidden) * 0.1
    H = np.tanh(X @ W1.T + b1)
    T = np.eye(n_classes)[np.asarray(y, int)]
    W2 = np.linalg.solve(H.T @ H + reg * np.eye(hidden), H.T @ T).T
    return {"w1.weight": W1, "w1.bias": b1, "w2.weight": W2}


def elm_train(X, y, hidden=256, n_classes=None, seed=0, reg=1e-3):
    """Train a small real model with NO autodiff: an Extreme Learning Machine
    (random tanh hidden layer + least-squares readout, Huang et al. 2006). Exists as
    the measurement instrument for transform_model's honesty contract -- a model we
    can train, export, transform, and re-evaluate entirely inside NumPy. Returns
    {name: array} in the same shape a checkpoint takes, so the whole Unicron surface
    applies to it."""
    rng = np.random.default_rng(seed)
    n_classes = n_classes or int(np.max(y)) + 1
    W1 = rng.standard_normal((hidden, X.shape[1])) / np.sqrt(X.shape[1])
    b1 = rng.standard_normal(hidden) * 0.1
    H = np.tanh(X @ W1.T + b1)
    T = np.eye(n_classes)[np.asarray(y, int)]
    # ridge readout: the only "training", one solve
    W2 = np.linalg.solve(H.T @ H + reg * np.eye(hidden), H.T @ T).T
    return {"w1.weight": W1, "w1.bias": b1, "w2.weight": W2}


def elm_predict(tensors, X):
    """Forward pass for elm_train models (dense or factored storage transparently --
    reconstruct_model handles the .U/.V pairs)."""
    t = reconstruct_model(tensors)
    H = np.tanh(X @ t["w1.weight"].T + t["w1.bias"])
    return np.argmax(H @ t["w2.weight"].T, axis=1)


def functional_retention(tensors_before, tensors_after, X, y, predict=elm_predict):
    """THE measurement transform_model's claim depends on: accuracy before vs after
    on held-out data. Any model with a NumPy-callable predict(tensors, X) plugs in.
    Returns the two accuracies and their difference -- no verdict words, numbers."""
    y = np.asarray(y, int)
    acc_b = float(np.mean(predict(tensors_before, X) == y))
    acc_a = float(np.mean(predict(tensors_after, X) == y))
    return {"acc_before": acc_b, "acc_after": acc_a, "delta": acc_a - acc_b}




# --------------------------------------------------------------------------- assimilation

def rsvd(W, k, seed=0, oversample=10, power=2):
    """Randomized SVD (Halko, Martinsson & Tropp 2011): top-k factors of a huge matrix
    from k+p Gaussian probes and `power` subspace iterations -- O(mnk) instead of the
    full O(mn*min(m,n)). Exists because the Qwen-class embedding table (250k x 2k) is
    ~1e12 flops under exact SVD just to LOOK at it. Deterministic under the seed.
    WHY power iterations: weight spectra decay slowly through the MP bulk; without
    q>=1 the probe subspace leaks bulk energy and the top singular values bias low."""
    W = np.asarray(W, np.float64)
    m, n = W.shape
    k = min(k, min(m, n))
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(W @ rng.standard_normal((n, min(k + oversample, n))))[0]
    for _ in range(power):
        Q = np.linalg.qr(W @ (W.T @ Q))[0]
    U_s, sv, Vt = np.linalg.svd(Q.T @ W, full_matrices=False)
    return (Q @ U_s)[:, :k], sv[:k], Vt[:k, :]


def _mp_edge_from_sv(sv, shape):
    """Marchenko-Pastur edge (singular-value scale) computed from an ALREADY
    COMPUTED spectrum -- exists because calling spectral_report just for the edge
    re-runs a full SVD, and on a real 0.8B checkpoint that doubled an already
    slow pass (measured live: the console sat silent long enough to be reported
    as a hang). Same robust median-matching sigma as spectral_report."""
    n, m = max(shape), min(shape)
    ev = np.asarray(sv, np.float64) ** 2
    q = m / n
    grid = np.linspace((1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2, 2001)[1:-1]
    dens = np.sqrt(((1 + np.sqrt(q)) ** 2 - grid) * (grid - (1 - np.sqrt(q)) ** 2)) / (2 * np.pi * q * grid)
    cdf = np.cumsum(dens); cdf /= cdf[-1]
    sigma2 = np.median(ev) / (n * grid[int(np.searchsorted(cdf, 0.5))])
    thresh = sigma2 * n * (1 + np.sqrt(q)) ** 2 * (1.0 + 3.0 * n ** (-2.0 / 3.0))
    return float(np.sqrt(thresh))


# Name-pattern policy for transformer checkpoints: decide CHEAP (string match) before
# computing EXPENSIVE (SVD). Embedding tables and output heads are lookup structures --
# per-token rows, not learned linear maps; their spectrum is not a training readout and
# low-ranking them clamps the vocabulary. Norms/biases are 1D and pass min_dim anyway,
# but conv stems are listed because flatten-(d0,rest) SVD on a 3D conv mixes kernel
# axes with channels -- a transform convention hazard already on the repo ledger.
# "visual"/"mtp" added after reading the official Qwen3.5-0.8B card: the 0.8B is
# a VLM with a vision encoder and multi-token-prediction weights. Our retention
# instrument is TEXT perplexity/chat -- it cannot measure vision or MTP damage,
# and the honesty contract forbids transforming what we cannot measure. Also on
# the card: the LM output is TIED to the 248320x1024 embedding (~1/3 of all
# params), so the embed skip alone already protects a third of the model.
SKIP_PATTERNS = ("embed", "lm_head", "wte", "wpe", "tok_embeddings", "conv",
                 "patch_embed", "norm", "ln_", "visual", "mtp")


def _policy_skip(name):
    low = name.lower()
    return any(pat in low for pat in SKIP_PATTERNS)


def spectral_regime(sv, edge, band=(0.75, 1.30)):
    """Which world does this spectrum live in? Returns "spike_bulk" or "heavy_tail".

    THE FIELD RESULT THIS ENCODES (Qwen3.5-0.8B, measured live): MP-edge filtering
    DESTROYED a real LLM -- original answered "water", assimilated emitted 256
    newlines. Cause: two of our own research anchors are in tension, and only one
    applies per layer. Spike+bulk (Thamm/Staats/Rosenow, small nets): learned
    signal sits in isolated outliers ABOVE a noise bulk, with a spectral GAP at
    the edge -- MP filtering is valid and beneficial. Heavy-tailed (Martin &
    Mahoney, modern well-trained nets): the ESD decays as a continuous power law,
    there is NO gap, and everything past the "edge" is still learning -- cutting
    there amputates the model. Discriminator: density of singular values inside a
    band around the edge. A gap means the band is nearly empty; a power law
    crosses it densely."""
    sv = np.asarray(sv, np.float64)
    # Count ABOVE the edge only: the MP bulk's own top sits just BELOW the edge,
    # so a two-sided band always reads dense and misfires (measured: the pca_net
    # instrument model was misrouted heavy_tail on first cut of this detector).
    # A spike+bulk spectrum leaves the region just above the edge EMPTY -- spikes
    # sit far above; a power law crosses it densely.
    just_above = int(np.sum((sv > edge) & (sv < band[1] * edge)))
    return "heavy_tail" if just_above >= max(3, 0.02 * sv.size) else "spike_bulk"


def assimilate_model(in_path_or_tensors, out_path=None, mode="shrink", guard=True,
                     policy=True, big=2_000_000, rsvd_rank=256, seed=0,
                     progress=None, regime="auto"):
    """UNICRON'S FULL PASS, one front door: load -> analyze -> filter/defragment ->
    re-export a WORKING model. Steps, and why each exists:

      1. LOAD     safetensors/gguf/npz (torch pickle refused, standing contract).
      2. POLICY   name-pattern skip (embed/lm_head/conv/norm) decided by string
                  match BEFORE any SVD -- the cheap gate in front of the expensive
                  compute. policy=False disables.
      3. FILTER   per matrix: exact SVD when small, randomized SVD (rsvd) when
                  size > `big` elements; MP-edge rank cut with the untrained-layer
                  guard (random != useless, -31.5 points on record); mode="shrink"
                  debiases kept spikes by the noise floor.
      4. EXPORT   DENSE, under ORIGINAL tensor names and shapes -- the output loads
                  wherever the input loaded (llama.cpp / HF / our own loader). The
                  disk file is not smaller (same shapes); what changed is CONTENT:
                  the still-random MP bulk is gone. The report carries the effective
                  ranks, so the factored small format (transform_model) remains
                  available for leCore-native deployment where size shrinks too.

    Returns (tensors, report). report["verify"] states the retention debt in plain
    words: the output is a claim until perplexity/eval runs before-vs-after on the
    caller's runtime -- assimilation without that measurement is narrative."""
    tensors = load_model(in_path_or_tensors) if isinstance(in_path_or_tensors, str) \
        else in_path_or_tensors
    out, rep = {}, {"layers": {}, "skipped": [], "guarded": [], "heavy_tail": [],
                    "filtered": 0,
                    "params": int(sum(np.asarray(t).size for t in tensors.values()))}
    for name, t in tensors.items():
        t = np.asarray(t)
        if t.ndim < 2 or min(t.shape[0], int(np.prod(t.shape[1:]))) < 8 \
                or (policy and _policy_skip(name)):
            out[name] = t
            if policy and t.ndim >= 2 and _policy_skip(name):
                rep["skipped"].append(name)
            continue
        # float32 SVD: the rank decision and the reconstruction both tolerate it
        # easily (bf16 containers carry ~3 decimal digits anyway), and it halves
        # the time and memory of the dominant cost on real checkpoints.
        W = t.reshape(t.shape[0], -1).astype(np.float32)
        if progress:
            progress(name, W.shape)
        if W.size > big:
            U, sv, Vt = rsvd(W, rsvd_rank, seed=seed)
            # MP edge still needs the FULL spectrum's bulk scale; estimate sigma from
            # a row sample instead of the (unavailable) full sv set. Row energies are
            # bulk-dominated, so Frobenius/size is a serviceable sigma^2 here.
            n_, m_ = max(W.shape), min(W.shape)
            sigma2 = float(np.mean(W[np.random.default_rng(seed).integers(0, W.shape[0], 512)] ** 2))
            edge = np.sqrt(sigma2 * n_) * (1 + np.sqrt(m_ / n_))
            approx = True
        else:
            U, sv, Vt = np.linalg.svd(W, full_matrices=False)
            edge = _mp_edge_from_sv(sv, W.shape)   # NOT spectral_report: no 2nd SVD
            approx = False
        n_out = int(np.sum(sv > edge))
        # REGIME ROUTING (regime="auto", the post-Qwen default): MP filtering is
        # only applied where the MP model FITS -- spike+bulk spectra with a real
        # gap at the edge. Heavy-tailed layers pass through UNTOUCHED, because on
        # them the cut removes learning, not noise (256-newlines field result).
        # regime="force" restores the old unconditional behaviour for study.
        if regime == "auto" and spectral_regime(sv, edge) == "heavy_tail":
            out[name] = t
            rep["heavy_tail"].append(name)
            continue
        # THE GUARD, corrected by measurement (first rehearsal filtered 0/32): the
        # discriminator is outlier ENERGY fraction, not outlier COUNT. Trained layers
        # legitimately have FEW outliers relative to width (Thamm et al.: most of a
        # trained spectrum stays random -- the finding, not a defect), so a count
        # threshold guards everything. A functionally-random layer (ELM/reservoir)
        # has outliers carrying ~0% of energy; a trained layer's spikes carry real
        # energy. Kept negative: never gate MP filtering on outlier count.
        full_energy = float(np.sum(W.astype(np.float64) ** 2))
        spike_energy = float(np.sum(sv[:n_out] ** 2 - edge ** 2)) if n_out else 0.0
        if guard and (n_out == 0 or spike_energy < 0.01 * full_energy):
            out[name] = t
            rep["guarded"].append(name)
            continue
        r = max(1, n_out) if not approx else max(1, min(n_out, rsvd_rank))
        s_kept = sv[:r].copy()
        if mode == "shrink":
            s_kept = np.sqrt(np.maximum(s_kept ** 2 - edge ** 2, 0.0))
        Wf = (U[:, :r] * s_kept) @ Vt[:r, :]
        out[name] = Wf.reshape(t.shape).astype(t.dtype)
        rep["filtered"] += 1
        rep["layers"][name] = {"rank": int(r), "of": int(min(W.shape)),
                               "energy_kept": float(np.sum(sv[:r] ** 2)) / max(full_energy, 1e-300),
                               "spike_energy_frac": spike_energy / max(full_energy, 1e-300),
                               "rsvd": bool(approx)}
    if out_path:
        save_safetensors(out_path, {k: np.ascontiguousarray(v) for k, v in out.items()})
        rep["out_path"] = out_path
    rep["verify"] = ("UNVERIFIED until measured: run your eval (perplexity / task "
                     "accuracy) on the input and output files on your runtime; "
                     "ship only if the delta is acceptable.")
    return out, rep




# ------------------------------------------------------------------------ dissection

def head_structure(W, candidates=(2, 4, 8, 16, 32)):
    """BLIND head-count discovery for a projection matrix: which reshape
    (heads, head_dim, in) reflects the model's real multi-head block structure?

    Two delegated instruments agree or the answer is not trusted:
      * holographic_axisrole (mind.analyze_axes) must call the head axis an
        INDEX/carrier -- heads are parallel slots, not content (probed live:
        coupling 1.0, role 'index' on planted head structure).
      * the per-slice stable rank ELBOW finds the boundary: merging two real
        heads into one slice ~doubles slice rank, while splitting one head in
        half leaves rank unchanged -- so the true head count is the smallest K
        whose rank stops shrinking when K doubles.

    KEPT NEGATIVE (probed, on record): demux_series is the WRONG tool here --
    head layout is BLOCK concatenation, not round-robin striding; the stride
    finder returns a spurious stride on blocked data.
    """
    from holographic.sampling_and_signal.holographic_axisrole import analyze_axes
    W = np.asarray(W, np.float64)
    m = W.shape[0]
    rows = []
    for K in candidates:
        if m % K or m // K < 2:
            continue
        Wh = W.reshape(K, m // K, -1)
        ranks = []
        for h in range(K):
            sv = np.linalg.svd(Wh[h], compute_uv=False)
            e = sv * sv
            ranks.append(float(e.sum() / e[0]) if e[0] > 0 else 0.0)
        ax = analyze_axes(Wh)
        rows.append({"heads": K, "mean_slice_stable_rank": float(np.mean(ranks)),
                     "head_axis_role": ax["per_axis"][0]["role"]})
    inferred = None
    # WHY the "reason" field: a bare None told the caller nothing about whether
    # the matrix had no head structure or the candidate list simply never
    # bracketed it (measured: a 4-head q_proj with candidates starting at 2 has
    # no doubling pair to compare when the shape divides poorly).
    reason = ("no candidate pair bracketed an elbow; try candidates that both "
              "divide the row count and include K and 2K")
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if b["heads"] == 2 * a["heads"] and \
                b["mean_slice_stable_rank"] > 0.75 * a["mean_slice_stable_rank"] \
                and rows[i]["head_axis_role"] == "index":
            inferred = a["heads"]
            reason = "elbow: rank survives doubling from %d to %d" % (
                a["heads"], b["heads"])
            break
    if not rows:
        reason = "no candidate head count divides this matrix's row count"
    return {"candidates": rows, "inferred_heads": inferred, "reason": reason}


def depth_sharing(mats):
    """HOW MUCH of a model is depth-REPEATED structure? Stack same-role matrices
    from every layer into (L, m, n) and read the layer-mode spectrum (mode-0
    unfolding SVD -- delegates to holographic_tucker's unfold; same machinery as
    tucker_compress's rank gate). shared_frac = energy of the top layer-mode:
    ~1.0 means the layers are one matrix wearing L costumes (store a shared
    basis + tiny per-layer cores -- a real structural-compression lever);
    ~1/L means every layer learned its own thing and depth is NOT redundant.
    This is a MEASUREMENT of the "LLMs are wastefully structured" hypothesis,
    per role, per model -- not a verdict."""
    from holographic.caching_and_storage.holographic_tucker import unfold
    X = np.stack([np.asarray(w, np.float64) for w in mats])
    sv = np.linalg.svd(unfold(X, 0), compute_uv=False)
    e = sv * sv
    return {"n_layers": int(X.shape[0]),
            "layer_mode_spectrum": (sv / sv[0]).tolist() if sv[0] > 0 else sv.tolist(),
            "shared_frac": float(e[0] / e.sum()) if e.sum() > 0 else 0.0,
            "chance": float(1.0 / X.shape[0])}




# ---------------------------------------------------------------------------- imbue

def task_vector(base, finetuned):
    """A CAPABILITY as an object: tau = W(finetuned) - W(base), per tensor. The
    fine-tune's learning, extracted from its checkpoint as a thing you can hold,
    scale, add, and subtract -- the weight-space form of the drift-model algebra
    (HDRIFT compose/ablate), one level down, on models themselves. Only tensors
    present in BOTH with matching shapes contribute; everything else is reported."""
    tau, skipped = {}, []
    for name, wb in base.items():
        wf = finetuned.get(name)
        if wf is None or np.asarray(wf).shape != np.asarray(wb).shape:
            skipped.append(name)
            continue
        tau[name] = np.asarray(wf, np.float64) - np.asarray(wb, np.float64)
    return tau, {"n_tensors": len(tau), "skipped": skipped}


def imbue(target, tau, scale=1.0, policy=True):
    """WRITE a capability INTO a model: target + scale * tau, per tensor -- the
    Galvatron operation. Grounded in measured task-arithmetic (Ilharco et al.,
    "Editing Models with Task Arithmetic", ICLR 2023): fine-tune deltas act as
    composable vectors ON MODELS SHARING THE SAME BASE.

    THE LINEAGE LAW, pinned by this module's selftest with a measured failure:
    a delta only means anything in the basis it was learned in. Transplanting
    between models with DIFFERENT initializations scrambles both capabilities
    (basis mismatch); between same-base siblings it transfers the skill. For
    real LLMs this reads: donor fine-tune and target must descend from the SAME
    base checkpoint. imbue() cannot check lineage from weights alone -- the
    caller owns that claim, and the retention/eval debt applies doubly here.
    policy=True leaves embeddings/norms/visual/mtp untouched (same gate as
    assimilation: do not write where you cannot measure)."""
    out = {}
    for name, w in target.items():
        d = tau.get(name)
        if d is None or (policy and _policy_skip(name)) \
                or np.asarray(d).shape != np.asarray(w).shape:
            out[name] = w
            continue
        out[name] = (np.asarray(w, np.float64) + scale * np.asarray(d, np.float64)
                     ).astype(np.asarray(w).dtype)
    return out




# --------------------------------------------------------------------- archive

def _tensor_hash(a):
    """Content identity for exact-parity checks: hashlib over the raw bytes of a
    canonical (C-contiguous, declared-dtype) view. hashlib, never hash() -- the
    archive's parity claims must survive process restarts and years."""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode()
                          + str(a.shape).encode()).hexdigest()


def regenerate(recipe):
    """Materialize a tensor from a RECIPE -- leCore's seed-determinism rung: for
    tensors the engine itself created (ELM random features, projector bridges,
    instrument inits), the generator IS the storage. Supported kinds:
    standard_normal / uniform / projector (the galvatron bridge). Every recipe
    carries the sha256 of what it must produce; regenerate() verifies it, so a
    recipe can never silently drift from its data."""
    kind = recipe["kind"]
    shape = tuple(recipe["shape"])
    if kind == "standard_normal":
        a = np.random.default_rng(int(recipe["seed"])).standard_normal(shape)
        # exactness demands the ORIGINAL operation: x/sqrt(d) and x*(1/sqrt(d))
        # differ in the last ulp, and the hash check caught exactly that. A
        # recipe stores the operation, not a mathematically-equal cousin.
        if "div" in recipe:
            a = a / float(recipe["div"])
        else:
            a = a * float(recipe.get("scale", 1.0))
    elif kind == "uniform":
        a = np.random.default_rng(int(recipe["seed"])).uniform(
            float(recipe.get("low", 0.0)), float(recipe.get("high", 1.0)), shape)
    elif kind == "projector":
        from holographic.agents_and_reasoning.holographic_galvatron import _projector
        a = _projector(int(recipe["d_in"]), int(recipe["d_out"]), recipe["tag"])
    else:
        raise ValueError("unknown recipe kind %r" % kind)
    a = a.astype(recipe.get("dtype", "float64"))
    h = _tensor_hash(a)
    if h != recipe["sha256"]:
        raise ValueError("recipe drift: regenerated hash %s != stored %s"
                         % (h[:12], recipe["sha256"][:12]))
    return a


def generator_audit(tensor):
    """Is this tensor's generator DISCOVERABLE? Delegates to HRNN's two-stage
    compressibility gate (holographic_hrnn.compressibility_gate) rather than
    asserting. Returns {"discoverable": bool, "stage": ...}.

    WHY THIS EXISTS, and why the archive never seed-searches: a seed-born
    tensor is deterministic GIVEN the seed but statistically white, so the gate
    rejects it (measured: passed=False at stage1) exactly as it rejects trained
    weights. A seed can be KNOWN, never DISCOVERED -- which is why the RECIPE
    rung takes caller-supplied provenance and verifies it by hash, instead of
    hunting for a generator that no measurement could confirm."""
    from holographic.agents_and_reasoning.holographic_hrnn import compressibility_gate
    g = compressibility_gate(np.asarray(tensor, np.float64).ravel())
    return {"discoverable": bool(g["passed"]), "stage": g.get("stage")}


def archive_models(models, reference=None, recipes=None):
    """Archive a FLEET of models with leCore's storage ladder, per tensor:

      rung 0 SAME    identical to the reference tensor -> store a pointer
      rung 1 RECIPE  known provenance (caller-supplied recipe) -> store the
                     recipe, hash-verified on regeneration; the seed rung
      rung 2 DELTA   differs from reference -> store zlib(delta bytes) if it
                     pays (fine-tune deltas are small-magnitude and compress;
                     the task-vector insight applied to STORAGE)
      rung 3 RAW     zlib(raw) or plain raw, whichever is smaller -- the
                     honesty rung; never pretend structure that is not there

    KEPT NEGATIVE, stated where it belongs: TRAINED weights are NOT seed-
    compressible -- they are the residue of data the archive never saw; no
    seed search is attempted, ever. The recipe rung is for leCore-born
    tensors whose generator is KNOWN, not discovered.

    models: {model_name: {tensor_name: array}}. reference: model name or a
    weights dict (default: first model). recipes: {(model, tensor): recipe}.
    Returns (archive, report). restore_model(archive, name) is bit-exact."""
    names = list(models)
    if reference is None:
        reference = names[0]
    ref = models[reference] if isinstance(reference, str) else reference
    ref_name = reference if isinstance(reference, str) else "<external>"
    recipes = recipes or {}
    arc = {"reference_name": ref_name, "reference": {}, "models": {}}
    rep = {"per_model": {}, "raw_bytes": 0, "archive_bytes": 0, "rungs": {}}
    for tname, t in ref.items():
        a = np.ascontiguousarray(t)
        arc["reference"][tname] = a
        rep["archive_bytes"] += a.nbytes
    for mname in names:
        entry, mrep = {}, {"SAME": 0, "RECIPE": 0, "DELTA": 0, "RAW": 0}
        for tname, t in models[mname].items():
            a = np.ascontiguousarray(t)
            rep["raw_bytes"] += a.nbytes
            r = ref.get(tname)
            rec = recipes.get((mname, tname))
            if rec is not None:
                rec = dict(rec, sha256=_tensor_hash(a), dtype=str(a.dtype),
                           shape=list(a.shape))
                regenerate(rec)                      # verify BEFORE trusting
                entry[tname] = ("RECIPE", rec)
                rep["archive_bytes"] += 200          # recipe overhead estimate
            elif r is not None and np.ascontiguousarray(r).shape == a.shape                     and np.array_equal(np.ascontiguousarray(r), a):
                entry[tname] = ("SAME", None)
            elif r is not None and np.ascontiguousarray(r).shape == a.shape                     and np.ascontiguousarray(r).dtype == a.dtype:
                # EXACT delta = XOR of byte views. Field-caught kept negative:
                # arithmetic delta (ref + (a - ref)) is NOT bit-exact in IEEE
                # float -- hash parity failed on it. XOR zeroes the shared bits
                # of near-siblings (compresses well) and decode is exact BY
                # CONSTRUCTION, not by numerical luck.
                rbytes = np.ascontiguousarray(r).view(np.uint8).ravel()
                blob = zlib.compress(np.bitwise_xor(
                    a.view(np.uint8).ravel(), rbytes).tobytes(), 6)
                if len(blob) < 0.9 * a.nbytes:
                    entry[tname] = ("DELTA", blob)
                    rep["archive_bytes"] += len(blob)
                else:                                # delta did not pay: honesty rung
                    zb = zlib.compress(a.tobytes(), 6)
                    payload = zb if len(zb) < a.nbytes else a
                    entry[tname] = ("RAW", payload)
                    rep["archive_bytes"] += len(zb) if len(zb) < a.nbytes else a.nbytes
            else:
                zb = zlib.compress(a.tobytes(), 6)
                payload = zb if len(zb) < a.nbytes else a
                entry[tname] = ("RAW", payload)
                rep["archive_bytes"] += len(zb) if len(zb) < a.nbytes else a.nbytes
            mrep[entry[tname][0]] += 1
        arc["models"][mname] = {"tensors": entry,
                                "dtypes": {k: str(np.asarray(v).dtype)
                                           for k, v in models[mname].items()},
                                "shapes": {k: list(np.asarray(v).shape)
                                           for k, v in models[mname].items()}}
        rep["per_model"][mname] = mrep
    rep["ratio"] = rep["raw_bytes"] / max(rep["archive_bytes"], 1)
    return arc, rep


def restore_model(archive, name):
    """Bit-exact reconstruction from the archive: pointer / regenerate / ref+delta
    / decompress, per rung. Exactness is the contract -- parity is asserted by
    hash in the selftest, not assumed."""
    ref = archive["reference"]
    entry = archive["models"][name]
    out = {}
    for tname, (rung, payload) in entry["tensors"].items():
        shape = tuple(entry["shapes"][tname])
        dtype = entry["dtypes"][tname]
        if rung == "SAME":
            out[tname] = np.array(ref[tname], copy=True)
        elif rung == "RECIPE":
            out[tname] = regenerate(payload)
        elif rung == "DELTA":
            x = np.frombuffer(zlib.decompress(payload), dtype=np.uint8)
            rbytes = np.ascontiguousarray(ref[tname]).view(np.uint8).ravel()
            out[tname] = np.bitwise_xor(x, rbytes).view(dtype).reshape(shape).copy()
        else:
            raw = payload if isinstance(payload, np.ndarray) else                 np.frombuffer(zlib.decompress(payload), dtype=dtype).reshape(shape)
            out[tname] = np.array(raw, copy=True).reshape(shape)
    return out




# ----------------------------------------------------------------- middle-out

def middle_out_encode(W, n_refine=6, base_bits=3, max_bits=9):
    """PROGRESSIVE weight code: one artifact, many fidelity points. A coarse base
    (base_bits uniform quantization of the whole matrix) plus successive-
    approximation refinement layers, each halving the remaining quantization
    error. Decode any PREFIX -- the stream is truncatable at load time, so a
    single stored artifact serves a 3-bit edge deployment and a 9-bit server
    deployment with NO re-encode and no cut decision (which is what made the
    heavy-tail/spike regime split so treacherous: there is no rank to choose).

    HONEST RATIO CLAIM -- there is none, and that is the measured finding:
    per-byte quality is at PARITY with plain uniform quantization, never better.
    Three refutations are pinned in this module's selftest and must not be
    reinvented:
      (1) low-rank + bit-plane middle-out LOSES to uniform quantization on
          Frobenius error (heavy-tail 256x512: 168 KB at rel 0.096 vs uniform
          8-bit 136 KB at rel 0.017) -- the greedy rate-distortion allocator
          picks rank moves that lose on the cumulative curve;
      (2) it also fails to win on the ruler that actually matters for weights
          (function): on the ELM instrument, uniform and low-rank both saturate
          accuracy at the same budget -- a tie, not a win;
      (3) per-layer SENSITIVITY-allocated bits do not beat the BEST FLAT
          setting either (allocated 4772 B @ acc 1.000 vs flat-3 4579 B @ acc
          1.000) -- allocation only looks like a win against a strawman
          (flat-4), which is exactly the baseline-discipline trap.
    So: middle-out ships for PROGRESSIVITY, not compression. Claiming otherwise
    would be shipping a bad result as a win.

    Returns {"base": ..., "refinements": [...], "shape", "scale", "bits"}.
    """
    W = np.asarray(W, np.float64)
    scale = float(np.max(np.abs(W))) + 1e-30
    levels = int(np.clip(max_bits, base_bits, 16))
    q_full = np.rint(W / scale * (2 ** (levels - 1) - 1)).astype(np.int32)
    keep = levels - base_bits
    base = (q_full >> keep) << keep                      # top base_bits planes
    out = {"shape": tuple(W.shape), "scale": scale, "levels": levels,
           "base_bits": int(base_bits),
           "base": zlib.compress(base.astype(np.int32).tobytes(), 6),
           "refinements": []}
    # each refinement layer = the next bit-plane down (successive approximation)
    for i in range(min(int(n_refine), keep)):
        sh = keep - 1 - i
        plane = ((q_full >> sh) & 1).astype(np.uint8)
        out["refinements"].append(zlib.compress(np.packbits(plane).tobytes(), 6))
    return out


def middle_out_decode(code, n_refine=None):
    """Decode a middle-out stream using its base plus the first `n_refine`
    refinement layers (None = all). Fewer layers = smaller memory, coarser
    weights, SAME artifact -- the truncatable read."""
    shape = tuple(code["shape"])
    levels, base_bits = int(code["levels"]), int(code["base_bits"])
    q = np.frombuffer(zlib.decompress(code["base"]), dtype=np.int32).reshape(shape).copy()
    keep = levels - base_bits
    n = len(code["refinements"]) if n_refine is None else int(n_refine)
    for i in range(min(n, len(code["refinements"]))):
        sh = keep - 1 - i
        bits = np.unpackbits(np.frombuffer(
            zlib.decompress(code["refinements"][i]), dtype=np.uint8))
        plane = bits[:int(np.prod(shape))].reshape(shape).astype(np.int32)
        q = q | (plane << sh)
    return q.astype(np.float64) / (2 ** (levels - 1) - 1) * code["scale"]


def middle_out_bytes(code, n_refine=None):
    """Byte cost of a given truncation point -- so the caller can pick a budget
    with a number in hand instead of a hope."""
    n = len(code["refinements"]) if n_refine is None else int(n_refine)
    return len(code["base"]) + sum(len(r) for r in code["refinements"][:n])




# ------------------------------------------------------- compressed residency

class LazyWeights:
    """Weights that live COMPRESSED in RAM and materialize per tensor on demand.

    The model's own storage becomes a cache hierarchy: middle-out codes are the
    cold store, the LRU holds the hot working set, and a tensor is decoded only
    when the forward pass actually reaches it. Because a transformer touches
    layers strictly in order, the working set is tiny -- this is the classic
    demoscene/streaming trade (keep it packed, unpack at the point of use) applied
    to a model's parameters.

    Drop-in for a plain weights dict: GDNRuntime does `w[name]` and `name in w`
    and needs no change. Bit-exactness is the contract -- at full refinement depth
    a lazily decoded tensor equals the eagerly quantized one exactly, so logits
    are unchanged; truncating refinement layers trades fidelity for footprint
    with a knob the caller sets, never silently.

    HONEST LIMIT: this is a RAM-footprint lever, not a speed lever -- decode costs
    time on a cache miss. Measure both before claiming either."""

    def __init__(self, weights, max_cached=8, n_refine=6, base_bits=3,
                 max_bits=9, skip=("norm", "bias", "A_log", "dt_bias")):
        self._codes, self._raw, self._lru, self._max = {}, {}, [], int(max_cached)
        self._n_refine = n_refine
        self.stats = {"hits": 0, "misses": 0, "decoded_bytes": 0}
        for name, t in weights.items():
            a = np.asarray(t)
            # tiny/1-D tensors stay raw: coding overhead exceeds the win, and
            # norms are the tensors quantization hurts most (policy parity with
            # assimilation -- do not compress what you cannot afford to blur)
            if a.ndim < 2 or a.size < 4096 or any(k in name for k in skip):
                self._raw[name] = a
            else:
                self._codes[name] = middle_out_encode(
                    a, n_refine=n_refine, base_bits=base_bits, max_bits=max_bits)

    def __contains__(self, name):
        return name in self._raw or name in self._codes

    def __iter__(self):
        return iter(list(self._raw) + list(self._codes))

    def keys(self):
        return list(self)

    def __len__(self):
        return len(self._raw) + len(self._codes)

    def __getitem__(self, name):
        if name in self._raw:
            return self._raw[name]
        for i, (k, v) in enumerate(self._lru):
            if k == name:
                self._lru.append(self._lru.pop(i))
                self.stats["hits"] += 1
                return v
        self.stats["misses"] += 1
        v = middle_out_decode(self._codes[name], n_refine=self._n_refine)
        self.stats["decoded_bytes"] += v.nbytes
        self._lru.append((name, v))
        while len(self._lru) > self._max:
            self._lru.pop(0)
        return v

    def stored_bytes(self):
        """Actual resident footprint of the compressed store (+ raw passthrough)."""
        c = sum(middle_out_bytes(v) for v in self._codes.values())
        r = sum(a.nbytes for a in self._raw.values())
        return {"coded": c, "raw": r, "total": c + r,
                "dense": c and sum(int(np.prod(v["shape"])) * 4
                                   for v in self._codes.values()) + r}


def source_dtypes(model_dir_or_file):
    """The ON-DISK dtype of every tensor, read from the safetensors header.

    WHY THIS IS NEEDED: numpy has no bfloat16, so our loader decodes BF16 to
    float32 on read. export_portable then faithfully preserves float32 and
    writes a file DOUBLE the original -- Moose's 1.75 GB bf16 Qwen came back as
    3.5 GB holding the same numbers, and preserving the in-memory dtype was
    exactly the wrong thing to preserve. The dtype that matters is the one the
    file had, not the one the decoder produced."""
    import json as _json

    out = {}
    files = []
    if os.path.isdir(model_dir_or_file):
        for f in sorted(os.listdir(model_dir_or_file)):
            if f.endswith(".safetensors"):
                files.append(os.path.join(model_dir_or_file, f))
    else:
        files.append(model_dir_or_file)
    for path in files:
        try:
            with open(path, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                head = _json.loads(fh.read(n).decode("utf-8"))
        except (OSError, ValueError):
            continue
        for name, meta in head.items():
            if name != "__metadata__" and isinstance(meta, dict):
                out[name] = meta.get("dtype", "F32")
    return out


def export_portable(weights, out_path, n_refine=None, dtype=None, like=None):
    """Decode a compressed/lazy store back to a PLAIN safetensors file at a chosen
    fidelity -- the bridge to every standard harness.

    Ollama, LM Studio and llama.cpp consume GGUF, which is produced from a normal
    Hugging Face directory by llama.cpp's convert_hf_to_gguf.py; none of them
    expose a custom-loader hook. So the portable artifact is deliberately BORING:
    ordinary tensors under ordinary names, indistinguishable from any other
    checkpoint. Ship the compact leCore artifact, decode at the fidelity the
    target deployment wants, and the result converts and runs like any model.

    DTYPE IS PRESERVED unless one is named. The default used to be F32, which
    doubled the file whenever the input was float16 -- measured on a real run as
    a 1.7 GB assimilated model becoming a 3.4 GB repaired one holding the same
    numbers.

    WHAT DOES NOT TRAVEL, stated plainly: residents (memory, dreamer, ward,
    council, capability calls) are runtime behaviour, not weights. A portable
    export is the model ALONE. Residents require leCore's runtime (or a hooked
    harness); that is a property of every activation-space method, not a
    limitation of this one."""
    if isinstance(weights, LazyWeights):
        out = {}
        for name in weights:
            out[name] = np.ascontiguousarray(
                weights._raw[name] if name in weights._raw
                else middle_out_decode(weights._codes[name], n_refine=n_refine))
    else:
        out = {k: np.ascontiguousarray(np.asarray(v)) for k, v in weights.items()}
    # PRESERVE THE DTYPE THAT CAME IN. This defaulted to "F32" and silently
    # UPCAST every float16 tensor, so a repaired model came out DOUBLE the size
    # of the assimilated one it was built from -- 3.4 GB against 1.7 GB on a
    # real run, with identical numbers. An exporter that changes precision
    # without being asked is a compressor running in reverse.
    _MAP = {"float16": "F16", "float32": "F32", "float64": "F64",
            "bfloat16": "BF16", "int8": "I8", "uint8": "U8",
            "int16": "I16", "int32": "I32", "int64": "I64"}
    if dtype is not None:
        dts = {k: dtype for k in out}
    elif like:
        # MATCH THE SOURCE FILE, not the decoded array. This is the only way to
        # round-trip a bf16 checkpoint at its original size.
        src = source_dtypes(like)
        dts = {k: src.get(k, _MAP.get(str(np.asarray(v).dtype), "F32"))
               for k, v in out.items()}
    else:
        dts = {k: _MAP.get(str(np.asarray(v).dtype), "F32")
               for k, v in out.items()}
    save_safetensors(out_path, out, dtypes=dts)
    return {"path": out_path, "tensors": len(out),
            "bytes": os.path.getsize(out_path)}




# ------------------------------------------------------------- delta storage

def delta_lineage(model, candidates, k=64):
    """WHICH BASE was this fine-tune derived from? Ranks candidate bases by
    weight-space evidence alone -- no model cards, no metadata, no honest
    seller required.

    WHY THIS EXISTS: TStore (arXiv 2604.17104, May 2026) names "Missing Lineage
    Metadata" as a fundamental limitation of delta compression at scale --
    ZipLLM relies on Hugging Face model-card metadata to decide which models to
    pair, and model cards are optional. Delta storage is worthless without
    correct pairing, and leCore already had the instrument: fine-tuning moves
    weights a little, so the true base is the candidate whose per-tensor
    subspaces still align. Scores by mean cosine of leading singular vectors
    (basis overlap), which survives the small rotations a fine-tune induces.

    Returns candidates ranked best-first with their scores AND the margin over
    the runner-up -- a lineage call with no margin is a guess, and the caller
    should be able to see that."""
    scored = []
    for name, cand in candidates.items():
        sims, n = [], 0
        for tname, wf in model.items():
            wb = cand.get(tname)
            if wb is None:
                continue
            A = np.asarray(wf, np.float64)
            B = np.asarray(wb, np.float64)
            if A.ndim != 2 or A.shape != B.shape or min(A.shape) < 2:
                continue
            r = int(min(k, min(A.shape)))
            Ua = np.linalg.svd(A, full_matrices=False)[0][:, :r]
            Ub = np.linalg.svd(B, full_matrices=False)[0][:, :r]
            # principal-angle overlap: how much of one basis lives in the other
            sv = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
            sims.append(float(np.mean(sv)))
            n += 1
            if n >= 8:                      # a handful of tensors decides it
                break
        scored.append({"name": name, "score": float(np.mean(sims)) if sims else 0.0,
                       "tensors_compared": n})
    scored.sort(key=lambda r: -r["score"])
    margin = (scored[0]["score"] - scored[1]["score"]) if len(scored) > 1 else 1.0
    return {"ranked": scored, "best": scored[0]["name"] if scored else None,
            "margin": float(margin)}


def delta_encode(base, finetuned, energy=0.9999, bits=8, tol=1e-12, mode="lowrank"):
    """Store a fine-tune as a DELTA, not as a second model.

    Why this is a different problem from compressing weights -- and why the
    answer flips: a trained weight matrix is high-entropy and heavy-tailed, which
    is exactly why low-rank lost to plain quantization four times over (see
    middle_out_encode's pinned refutations). A DELTA is not a trained matrix. It
    is the RESIDUE of one task's learning on top of another's, and it is
    structurally thin: tensors the fine-tune never touched come back EXACTLY
    zero, and the ones it did touch concentrate in few directions (the empirical
    basis of the LoRA family). Measured on the pca_net instrument: the delta of a
    (256,60) layer was exactly rank-8, reconstructing to rel 0.0000 at 3384 B
    against 18374 B dense 8-bit -- 5.4x, LOSSLESS.

    HONEST CAVEAT, stated because it bounds the claim: on that instrument the
    BASE was also rank-8, so the ratio is instrument-bound, not proof that
    deltas beat bases in general. What IS general and measured here: unchanged
    tensors cost ZERO, and the rank needed is discovered from the delta's own
    spectrum rather than assumed. Price it on a real fine-tune pair before
    quoting a number.

    Returns {"tensors": {...}, "report": {...}} -- per tensor either
    {"kind": "unchanged"}, {"kind": "lowrank", U, V, ...} or {"kind": "dense"},
    whichever is smaller, so the codec can never lose to storing the delta plainly."""
    out, rep = {}, {"unchanged": 0, "lowrank": 0, "dense": 0,
                    "delta_bytes": 0, "dense_bytes": 0, "skipped": []}
    for name, wb in base.items():
        wf = finetuned.get(name)
        if wf is None or np.asarray(wf).shape != np.asarray(wb).shape:
            rep["skipped"].append(name)
            continue
        d = np.asarray(wf, np.float64) - np.asarray(wb, np.float64)
        dense_sz = d.size * (bits / 8.0)
        rep["dense_bytes"] += dense_sz
        if np.max(np.abs(d)) <= tol:
            out[name] = {"kind": "unchanged"}
            rep["unchanged"] += 1
            continue
        if d.ndim != 2 or min(d.shape) < 2:
            out[name] = {"kind": "dense", "d": d}
            rep["dense"] += 1
            rep["delta_bytes"] += dense_sz
            continue
        if mode == "qlr":
            # D-QRELO recipe (Li et al., Findings of ACL 2026, arXiv 2604.16940):
            # coarse ONE-BIT quantization captures the delta's dominant structure,
            # then low-rank approximates the SMALLER residual error. The reported
            # motivation matches what we measured independently: large-scale SFT
            # inflates delta magnitude and singular values, so a pure low-rank fit
            # of the whole delta degrades -- splitting it costs 1 bit per weight
            # and buys a much easier residual.
            sign = np.sign(d)
            alpha = float(np.mean(np.abs(d)))          # optimal 1-bit scale (L1)
            q1 = alpha * sign
            resid = d - q1
            Ur, Sr, Vtr = np.linalg.svd(resid, full_matrices=False)
            er = np.cumsum(Sr * Sr) / max(np.sum(Sr * Sr), 1e-300)
            rr = int(np.searchsorted(er, energy)) + 1
            sz = d.size / 8.0 + rr * (d.shape[0] + d.shape[1]) * (bits / 8.0)
            if sz < dense_sz:
                out[name] = {"kind": "qlr", "sign": sign.astype(np.int8),
                             "alpha": alpha, "U": (Ur[:, :rr] * Sr[:rr]),
                             "V": Vtr[:rr], "rank": rr}
                rep["lowrank"] += 1
                rep["delta_bytes"] += sz
                continue
        U, S, Vt = np.linalg.svd(d, full_matrices=False)
        e = np.cumsum(S * S) / max(np.sum(S * S), 1e-300)
        r = int(np.searchsorted(e, energy)) + 1
        lr_sz = r * (d.shape[0] + d.shape[1]) * (bits / 8.0)
        if lr_sz < dense_sz:
            out[name] = {"kind": "lowrank", "U": (U[:, :r] * S[:r]), "V": Vt[:r],
                         "rank": r}
            rep["lowrank"] += 1
            rep["delta_bytes"] += lr_sz
        else:
            # low-rank must EARN it: a fat delta stays dense rather than paying
            # factor overhead for nothing (the earn-your-bytes rule again)
            out[name] = {"kind": "dense", "d": d}
            rep["dense"] += 1
            rep["delta_bytes"] += dense_sz
    rep["ratio"] = rep["dense_bytes"] / max(rep["delta_bytes"], 1e-9)
    return {"tensors": out, "report": rep}


def delta_apply(base, delta, scale=1.0):
    """Rebuild the fine-tuned model from base + delta (scale<1 interpolates --
    the same knob task-vector arithmetic uses)."""
    out = {}
    for name, wb in base.items():
        rec = delta["tensors"].get(name)
        w = np.asarray(wb, np.float64)
        if rec is None or rec["kind"] == "unchanged":
            out[name] = np.asarray(wb)
            continue
        if rec["kind"] == "dense":
            d = rec["d"]
        elif rec["kind"] == "qlr":
            d = rec["alpha"] * rec["sign"].astype(np.float64) + rec["U"] @ rec["V"]
        else:
            d = rec["U"] @ rec["V"]
        out[name] = (w + scale * d).astype(np.asarray(wb).dtype)
    return out




# ------------------------------------------------------------- the front door

def full_report(model, sample_layers=8, roles=("mlp.gate_proj.weight",
                                               "self_attn.q_proj.weight"),
                candidate_bases=None, progress=None):
    """ONE CALL, THE WHOLE PICTURE: hand Unicron a checkpoint and get back what
    it is, what can be done to it, and -- just as loudly -- what CANNOT.

    This is the front door over the whole arc: spectral regime census (which
    layers even have a filterable gap), head structure, depth redundancy per
    role, optional lineage detection, and a RANKED list of size levers where
    every entry carries the measured evidence for or against it. The refuted
    levers are listed too, with their numbers, because a report that only lists
    what might work will get someone to try MP filtering on a heavy-tailed model
    again -- which is exactly how the Qwen 256-newline collapse happened.

    Returns {"census", "heads", "depth", "lineage", "levers", "warnings"}.
    Nothing here is a promise about a downstream eval: every lever's entry says
    what was measured and on what."""
    if isinstance(model, str):
        model = load_model(model)
    names = [n for n in sorted(model)
             if np.asarray(model[n]).ndim == 2
             and min(np.asarray(model[n]).shape) >= 8]
    census = {"heavy_tail": 0, "spike_bulk": 0, "policy_skipped": 0,
              "examined": 0, "filterable": []}
    step = max(1, len(names) // max(sample_layers, 1))
    for i, n in enumerate(names[::step]):
        if _policy_skip(n):
            census["policy_skipped"] += 1
            continue
        W = np.asarray(model[n], np.float64)
        if W.size > 4_000_000:                     # sampling keeps the door fast
            W = W[:2048, :2048]
        sv = np.linalg.svd(W, compute_uv=False)
        edge = _mp_edge_from_sv(sv, W.shape)
        regime = spectral_regime(sv, edge)
        census["examined"] += 1
        if regime == "heavy_tail":
            census["heavy_tail"] += 1
        else:
            census["spike_bulk"] += 1
            census["filterable"].append(n)
        if progress:
            progress(i, n, regime)

    heads = None
    for n in names:
        if "q_proj" in n or "qkv" in n:
            try:
                heads = head_structure(np.asarray(model[n], np.float64))
            except Exception:
                heads = None
            break

    depth = {}
    for role in roles:
        mats = [np.asarray(model[n], np.float64) for n in names
                if n.endswith(role) and not any(p in n.lower()
                                                for p in ("visual", "mtp"))]
        if len(mats) >= 2 and len({m.shape for m in mats}) == 1:
            depth[role] = depth_sharing(mats)

    lineage = None
    if candidate_bases:
        lineage = delta_lineage(model, candidate_bases)

    frac_ht = census["heavy_tail"] / max(census["examined"], 1)
    levers = []
    if census["spike_bulk"]:
        levers.append({
            "lever": "unicron_assimilate (MP filter, regime-routed)",
            "applies_to": "%d of %d examined layers with a real MP gap"
                          % (census["spike_bulk"], census["examined"]),
            "evidence": "regime router passes heavy-tail layers untouched; on "
                        "Qwen3.5-0.8B only the 16-dim DeltaNet gates qualified",
            "verdict": "worth trying, eval required"})
    levers.append({
        "lever": "unicron_delta_store (+ unicron_lineage for pairing)",
        "applies_to": "storing MANY fine-tunes of one base",
        "evidence": "measured exactly rank-8 of 60 on a learning instrument, "
                    "lossless, 5.4x vs dense; unchanged tensors cost ZERO",
        "verdict": "strongest measured lever in the arc"})
    levers.append({
        "lever": "unicron_lazy_weights (+ unicron_middleout)",
        "applies_to": "RAM footprint at serve time",
        "evidence": "2.67x smaller resident store, argmax sequence identical",
        "verdict": "footprint only -- a cache miss costs a decode, not a speedup"})
    warnings = []
    if frac_ht > 0.5:
        warnings.append(
            "HEAVY-TAIL DOMINANT (%.0f%% of examined layers): this model's "
            "knowledge-bearing matrices have NO separable noise floor. Forcing "
            "a rank cut here is what produced the measured 256-newline collapse."
            % (100 * frac_ht))
    warnings.append("REFUTED for weights, do not retry without new evidence: "
                    "low-rank middle-out, per-layer sensitivity allocation, "
                    "distributional codec, Gaussian splats (rel 0.98-0.99 -- "
                    "weight matrices are permutation-invariant, so no spatial "
                    "method applies), and long-range KV prediction on this "
                    "instrument (worse than plain quantization at every bit "
                    "width). Honest uniform quantization is a very strong "
                    "baseline; price against it FIRST.")
    return {"census": census, "heads": heads, "depth": depth,
            "lineage": lineage, "levers": levers, "warnings": warnings}


# --------------------------------------------------------------------------- selftest

def _selftest():
    import tempfile, os
    rng = np.random.default_rng(0)

    # 1) safetensors round-trip is byte-faithful, including a bf16 decode check.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "toy.safetensors")
        t = {"w": rng.standard_normal((32, 16)).astype(np.float32),
             "b": np.arange(7, dtype=np.int32)}
        save_safetensors(p, t)
        back = load_safetensors(p)
        assert np.array_equal(back["w"], t["w"]) and np.array_equal(back["b"], t["b"])
        assert load_model(p)["w"].shape == (32, 16)
    # bf16: decode of a hand-built header must be lossless for representable values
    vals = np.array([1.0, -2.5, 0.15625], dtype=np.float32)
    u16 = (vals.view(np.uint32) >> 16).astype(np.uint16)  # these values are exact in bf16
    assert np.array_equal(_decode_bf16(u16), vals)

    # 2) RMT readout: pure noise has ~no outliers; a planted rank-5 spike shows
    #    EXACTLY 5, even though the spikes inflate the raw std (the kept negative
    #    the median-based sigma estimate exists to survive).
    n, m = 800, 400
    noise = rng.standard_normal((n, m)) / np.sqrt(n)
    r_noise = spectral_report(noise)
    assert r_noise["n_outliers"] <= 2, r_noise      # finite-size slack, near-zero
    U = np.linalg.qr(rng.standard_normal((n, 5)))[0]
    V = np.linalg.qr(rng.standard_normal((m, 5)))[0]
    spiked = noise + U @ np.diag([8, 7, 6, 5, 4]) @ V.T
    r_spiked = spectral_report(spiked)
    assert r_spiked["n_outliers"] == 5, r_spiked
    # heavy tail moves alpha DOWN relative to noise
    assert r_spiked["alpha"] < r_noise["alpha"], (r_spiked["alpha"], r_noise["alpha"])

    # 3) Fingerprints: identical model -> cosine 1; same arch trained differently ->
    #    high but < 1; different arch (different layer names) -> near zero.
    A = {"layers": {"l%d" % i: spectral_report(
            rng.standard_normal((256, 128)) / 16.0) for i in range(6)}}
    A2 = {"layers": dict(A["layers"])}
    B = {"layers": {"l%d" % i: spectral_report(
            rng.standard_normal((256, 128)) / 16.0
            + (np.linalg.qr(rng.standard_normal((256, 3)))[0]
               @ np.linalg.qr(rng.standard_normal((128, 3)))[0].T) * 3.0)
         for i in range(6)}}
    C = {"layers": {"other%d" % i: v for i, v in enumerate(B["layers"].values())}}
    fA, fA2, fB, fC = (fingerprint(x) for x in (A, A2, B, C))
    assert abs(cosine(fA, fA2) - 1.0) < 1e-12
    assert 0.5 < cosine(fA, fB) < 0.999          # same roles, different metrics
    assert abs(cosine(fA, fC)) < 0.2             # foreign roles decorrelate

    # 4) compare_models sees the planted change in the right direction.
    cmp_ = compare_models(A, B)
    assert cmp_["n_common"] == 6
    mean_dof = np.mean([d["outlier_frac"] for d in cmp_["layer_deltas"].values()])
    assert mean_dof > 0, cmp_                     # B has MORE learned structure

    # 5) torch pickle refusal is a contract, not an accident.
    try:
        load_model("fake.pt"); raise AssertionError("should have refused .pt")
    except ValueError:
        pass
    # 6) GGUF round-trip: F32 bit-exact; Q8_0 dequant within block-quant error;
    #    dims order (innermost-first) proven by shape survival of a non-square matrix.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "toy.gguf")
        W = rng.standard_normal((48, 20)).astype(np.float32)
        save_gguf(p, {"wf": W, "wq": W}, quant={"wq"})
        g = load_gguf(p)
        assert g["wf"].shape == (48, 20) and np.array_equal(g["wf"], W)
        err = np.max(np.abs(g["wq"] - W)) / np.max(np.abs(W))
        assert err < 0.01, err                    # 8-bit block quant: <1% rel error
        assert load_model(p)["wf"].shape == (48, 20)
        # spectral readout survives Q8_0: outlier count unchanged on a spiked matrix
        S = (rng.standard_normal((256, 128)) / 16.0
             + (np.linalg.qr(rng.standard_normal((256, 3)))[0]
                @ np.linalg.qr(rng.standard_normal((128, 3)))[0].T) * 3.0)
        p2 = os.path.join(td, "spiked.gguf")
        save_gguf(p2, {"s": S.astype(np.float32)}, quant={"s"})
        assert spectral_report(load_gguf(p2)["s"])["n_outliers"] == \
               spectral_report(S)["n_outliers"] == 3

    # 7) Subspace overlap calibration: identical -> 1.0 exactly; independent random
    #    -> at the k/n chance floor (within sampling slack); a shared planted
    #    3-subspace under fresh noise -> the top-3 cosines are high, rest at chance.
    X = rng.standard_normal((200, 100))
    so = subspace_overlap(X, X, k=6)
    assert abs(so["overlap"] - 1.0) < 1e-10
    Y = rng.standard_normal((200, 100))
    so2 = subspace_overlap(X, Y, k=6)
    assert abs(so2["overlap"] - so2["chance"]) < 3 * so2["chance"], so2
    U = np.linalg.qr(rng.standard_normal((200, 3)))[0]
    mk2 = lambda: rng.standard_normal((200, 100)) / 14.0 + \
        U @ np.diag([6, 5, 4]) @ np.linalg.qr(rng.standard_normal((100, 3)))[0].T
    so3 = subspace_overlap(mk2(), mk2(), k=6)
    assert min(so3["cosines"][:3]) > 0.9, so3     # shared signal directions found
    assert max(so3["cosines"][3:]) < 0.6, so3     # noise directions stay near chance

    # 8) Porter-Thomas localization: a planted vector concentrated on 4 coordinates
    #    reads localized; pure-noise top vectors sit at the Gaussian IPR baseline.
    v_loc = np.zeros(300); v_loc[:4] = 0.5              # unit norm, 4-sparse
    w_r = rng.standard_normal(150); w_r /= np.linalg.norm(w_r)
    L = np.outer(v_loc, w_r) * 9.0 + rng.standard_normal((300, 150)) / np.sqrt(300)
    loc = vector_localization(L, k=5)
    top = loc["vectors"][0]
    assert top["left_ipr"] > 10 * top["left_ipr_gauss"], top   # 4-sparse: IPR ~ 1/4
    assert loc["n_localized"] >= 1
    pure = vector_localization(rng.standard_normal((300, 150)), k=5)
    for r in pure["vectors"]:
        assert r["left_ipr"] < 3.0 * r["left_ipr_gauss"], r    # noise stays at baseline

    # 9) RMT filter: on noise + planted rank-3 signal, the filtered matrix is CLOSER
    #    to the clean signal than the raw observation is -- the noise-model payoff
    #    plain low-rank approximation of the OBSERVED matrix cannot claim. And
    #    "shrink" beats "truncate" (spikes ride on the bulk; debiasing helps).
    n2, m2 = 600, 300
    Us = np.linalg.qr(rng.standard_normal((n2, 3)))[0]
    Vs = np.linalg.qr(rng.standard_normal((m2, 3)))[0]
    S_true = Us @ np.diag([6.0, 5.0, 4.0]) @ Vs.T
    Obs = S_true + rng.standard_normal((n2, m2)) / np.sqrt(n2)
    for md in ("truncate", "shrink"):
        Wf, info = rmt_filter(Obs, mode=md)
        assert info["rank_kept"] == 3, info
        assert np.linalg.norm(Wf - S_true) < np.linalg.norm(Obs - S_true), md
    e_tr = np.linalg.norm(rmt_filter(Obs, mode="truncate")[0] - S_true)
    e_sh = np.linalg.norm(rmt_filter(Obs, mode="shrink")[0] - S_true)
    assert e_sh < e_tr, (e_sh, e_tr)

    # 10) Trajectory: growing planted signal across 4 "checkpoints" -> cosine from
    #     start decreases monotonically, and the layer series sees outliers appear.
    def ckpt(strength):
        return {"layers": {"l0": spectral_report(
            rng.standard_normal((256, 128)) / 16.0
            + strength * (np.linalg.qr(rng.standard_normal((256, 3)))[0]
                          @ np.linalg.qr(rng.standard_normal((128, 3)))[0].T))}}
    traj = checkpoint_trajectory([ckpt(s_) for s_ in (0.0, 1.0, 2.0, 4.0)])
    cs = traj["cosine_from_start"]
    assert cs[0] == 1.0 and all(cs[i + 1] <= cs[i] + 1e-9 for i in range(3)), cs
    of = traj["layer_series"]["l0"]["outlier_frac"]
    assert of[-1] > of[0], of

    # 11) TRANSFORMATION with the honesty contract enforced, both directions:
    #     (a) a model whose big matrix is LEARNED (pca_net) compresses with measured
    #         functional retention; (b) a model whose big matrix is RANDOM-BUT-
    #         FUNCTIONAL (elm) is DESTROYED by unguarded filtering (-31 points was
    #         the live measurement) and SAVED by the guard. Random != useless.
    def blobs(n_per, noise_seed):
        r_c = np.random.default_rng(50)                  # centers FIXED across splits
        cents = r_c.standard_normal((4, 40)) * 3.0       # (instrument error kept on
        r = np.random.default_rng(noise_seed)            #  record: fresh centers per
        X = np.concatenate([c + r.standard_normal((n_per, 40)) for c in cents])
        y = np.repeat(np.arange(4), n_per)               #  split = testing on a
        return X, y                                      #  different task)
    Xtr, ytr = blobs(200, 20); Xte, yte = blobs(100, 21)

    modelP = pca_net_train(Xtr, ytr, hidden=256, k=8, seed=0)
    newP, repP = transform_model(modelP)
    assert repP["compression"] < 0.7, repP               # actually smaller
    assert repP["layers"]["w1.weight"]["factored"], repP # learned layer got factored
    frP = functional_retention(modelP, newP, Xte, yte)
    assert frP["acc_before"] > 0.9, frP                  # instrument healthy
    assert frP["delta"] > -0.03, frP                     # retention within 3 points

    modelE = elm_train(Xtr, ytr, hidden=256, seed=0)
    frE_bad = functional_retention(modelE, transform_model(modelE, guard=False)[0], Xte, yte)
    assert frE_bad["delta"] < -0.10, frE_bad             # unguarded: destroys it (pinned)
    newE, repE = transform_model(modelE, guard=True)
    assert repE["layers"]["w1.weight"].get("guarded"), repE
    frE = functional_retention(modelE, newE, Xte, yte)
    assert frE["delta"] > -0.03, frE                     # guard saves the random layer

    # round-trip: factored storage survives the safetensors container
    with tempfile.TemporaryDirectory() as td:
        p3 = os.path.join(td, "t.safetensors")
        save_safetensors(p3, {k: np.ascontiguousarray(v, np.float32) for k, v in newP.items()})
        fr2 = functional_retention(modelP, load_safetensors(p3), Xte, yte)
        assert abs(fr2["acc_after"] - frP["acc_after"]) < 0.02, (frP, fr2)

    # 12) rsvd agrees with exact SVD on the spike structure of a big-ish matrix
    #     (top singular values within 1%, subspace overlap ~1) at a fraction of cost.
    nB, mB = 2000, 700
    UB = np.linalg.qr(rng.standard_normal((nB, 6)))[0]
    VB = np.linalg.qr(rng.standard_normal((mB, 6)))[0]
    Wb = UB @ np.diag([30, 25, 20, 15, 12, 10.0]) @ VB.T + rng.standard_normal((nB, mB)) / np.sqrt(nB)
    Ur, sr, Vr = rsvd(Wb, 6, seed=0)
    sv_exact = np.linalg.svd(Wb, compute_uv=False)[:6]
    assert np.max(np.abs(sr - sv_exact) / sv_exact) < 0.01, (sr, sv_exact)
    Ue = np.linalg.svd(Wb, full_matrices=False)[0][:, :6]
    assert np.min(np.linalg.svd(Ue.T @ Ur, compute_uv=False)) > 0.99

    # 13) assimilate: policy skips by NAME with no SVD; guard protects the random
    #     layer; the learned layer is filtered; output round-trips through the
    #     container under ORIGINAL names; the functional model still works.
    modelQ = dict(modelP)                                # trained pca_net from (11)
    modelQ["model.embed_tokens.weight"] = rng.standard_normal((500, 64)).astype(np.float32)
    outQ, repQ = assimilate_model(modelQ)
    assert "model.embed_tokens.weight" in repQ["skipped"], repQ["skipped"]
    assert np.array_equal(outQ["model.embed_tokens.weight"], modelQ["model.embed_tokens.weight"])
    assert "w1.weight" in repQ["layers"], repQ           # learned layer filtered
    frQ = functional_retention(modelQ, outQ, Xte, yte)
    assert frQ["delta"] > -0.03, frQ
    outE2, repE2 = assimilate_model(modelE)              # random-feature model
    assert "w1.weight" in repE2["guarded"], repE2        # guard still on duty here

    # 14) BF16 WRITE path: values representable in bf16 round-trip exactly; RNE
    #     rounding within 1 ulp for the rest; and a BF16 load->save cycle keeps
    #     the FILE SIZE (the 2x-doubling regression measured live on Qwen3.5).
    with tempfile.TemporaryDirectory() as td:
        exact = np.array([1.0, -2.5, 0.15625, 3.0], np.float32)
        p4 = os.path.join(td, "bf.safetensors")
        save_safetensors(p4, {"w": exact}, dtypes={"w": "BF16"})
        back3, dts = load_safetensors(p4, return_dtypes=True)
        assert dts["w"] == "BF16" and np.array_equal(back3["w"], exact)
        vals = rng.standard_normal(4096).astype(np.float32)
        save_safetensors(p4 + "b", {"w": vals}, dtypes={"w": "BF16"})
        approx2 = load_safetensors(p4 + "b")["w"]
        assert np.max(np.abs(approx2 - vals) / np.maximum(np.abs(vals), 1e-6)) < 2 ** -8
        big = {"w": rng.standard_normal((64, 64)).astype(np.float32)}
        save_safetensors(p4 + "c", big, dtypes={"w": "BF16"})
        sz1 = os.path.getsize(p4 + "c")
        t2, d2 = load_safetensors(p4 + "c", return_dtypes=True)
        save_safetensors(p4 + "d", t2, dtypes=d2)
        assert os.path.getsize(p4 + "d") == sz1, "load->save changed file size"

    # 15) REGIME ROUTING, the Qwen field lesson pinned: a heavy-tailed matrix
    #     (continuous power-law ESD, no edge gap -- the well-trained-LLM regime)
    #     must pass through assimilation UNTOUCHED; the spike+bulk matrix from the
    #     same run must still be filtered. Both through one call.
    nH, mH = 600, 300
    Uh = np.linalg.qr(rng.standard_normal((nH, mH)))[0][:, :mH]
    Vh = np.linalg.qr(rng.standard_normal((mH, mH)))[0]
    sv_pl = (np.arange(1, mH + 1) ** -0.7) * 8.0          # smooth power law
    Wheavy = (Uh * sv_pl) @ Vh.T
    Wspike = (rng.standard_normal((nH, mH)) / np.sqrt(nH)
              + np.linalg.qr(rng.standard_normal((nH, 4)))[0]
              @ np.diag([7, 6, 5, 4.0])
              @ np.linalg.qr(rng.standard_normal((mH, 4)))[0].T)
    outR, repR = assimilate_model({"heavy.weight": Wheavy.astype(np.float32),
                                   "spiky.weight": Wspike.astype(np.float32)})
    assert "heavy.weight" in repR["heavy_tail"], repR
    assert np.array_equal(outR["heavy.weight"], Wheavy.astype(np.float32))
    assert repR["layers"].get("spiky.weight", {}).get("rank") == 4, repR
    # regime="force" restores the old cut on the heavy-tailed matrix (for study)
    outF, repF = assimilate_model({"heavy.weight": Wheavy.astype(np.float32)},
                                  regime="force")
    assert "heavy.weight" in repF["layers"] or "heavy.weight" in repF["guarded"]

    # 16) CROSS-FACULTY seam (unicron <-> residualcodec), post-merge: for HEAVY-
    #     TAILED layers -- where regime routing refuses rank truncation -- the
    #     honest size lever is error-bounded residual coding. Measured at probe
    #     scale on a real-size matrix: 5.22x vs zlib at bf16-class error, alpha
    #     1.770->1.769, stable rank unchanged; ~300s per 80k values, so this is a
    #     COLD-STORAGE lever, not a hot path (priced in the codec atlas). Here at
    #     selftest scale the same contract is pinned fast.
    #     KEPT NEGATIVE from the first attempt at this seam: distcodec is the
    #     WRONG codec for weights -- it ships a DISTRIBUTION; a decoded layer is a
    #     fresh sample that merely resembles the original, which is meaningless
    #     for a neural net. Weights need decode ~= original: residual, not dist.
    from holographic.sampling_and_signal.holographic_residualcodec import (
        residual_encode, residual_decode)
    mS, nS = 48, 96
    Us = np.linalg.qr(rng.standard_normal((nS, mS)))[0][:, :mS]
    Vs2 = np.linalg.qr(rng.standard_normal((mS, mS)))[0]
    Wht = ((Us * ((np.arange(1, mS + 1) ** -0.7) * 5.0)) @ Vs2.T).astype(np.float32)
    scale = float(np.max(np.abs(Wht)))
    outR2 = residual_encode(Wht.astype(np.float64).ravel(),
                            max_error=scale * 2 ** -8, min_seg=512)
    Wq2 = np.asarray(residual_decode(outR2["blob"])).reshape(Wht.shape).astype(np.float32)
    assert np.max(np.abs(Wq2 - Wht)) <= scale * 2 ** -8 + 1e-9
    rA2, rB2 = spectral_report(Wht), spectral_report(Wq2)
    assert abs(rB2["alpha"] - rA2["alpha"]) / rA2["alpha"] < 0.05
    assert abs(rB2["stable_rank"] - rA2["stable_rank"]) / rA2["stable_rank"] < 0.05

    # 17) DISSECTION: blind head-count recovery on planted 8-head structure, and
    #     the elbow logic's failure modes pinned (finer split leaves rank flat).
    heads8 = []
    for h in range(8):
        Uh8 = np.linalg.qr(rng.standard_normal((16, 2)))[0]
        Vh8 = np.linalg.qr(rng.standard_normal((64, 2)))[0]
        heads8.append(Uh8 @ np.diag([4.0, 3.0]) @ Vh8.T + 0.05 * rng.standard_normal((16, 64)))
    Wheads = np.concatenate(heads8, axis=0)          # (128, 64), 8 head blocks
    hs = head_structure(Wheads)
    assert hs["inferred_heads"] == 8, hs

    # 18) DEPTH SHARING calibration: L copies of one base + noise -> shared_frac
    #     near 1; independent layers -> near the 1/L chance floor.
    base = rng.standard_normal((64, 32))
    shared = [base + 0.05 * rng.standard_normal((64, 32)) for _ in range(12)]
    indep = [rng.standard_normal((64, 32)) for _ in range(12)]
    dsh, din = depth_sharing(shared), depth_sharing(indep)
    assert dsh["shared_frac"] > 0.9, dsh["shared_frac"]
    assert din["shared_frac"] < 3.0 * din["chance"], (din["shared_frac"], din["chance"])

    # 19) IMBUE, the Galvatron operation, measured in BOTH directions with the
    #     instrument models. Same shared random basis (ELM W1, one seed), eight
    #     output classes, two disjoint 4-class tasks on separate input regions:
    #       base       = trained on task1 only
    #       donor      = base's sibling additionally trained on task2
    #       tau        = donor - base           (the capability, extracted)
    #       imbued     = base + tau             (capability written in)
    #     Contract: imbued gains task2 (donor-level) while KEEPING task1.
    #     LINEAGE LAW negative: the same tau applied to a DIFFERENT-init model
    #     fails to deliver task2 -- deltas are basis-bound.
    r1 = np.random.default_rng(60); r2 = np.random.default_rng(61)
    c1 = r1.standard_normal((4, 40)) * 3.0            # task1 lives here
    c2 = r1.standard_normal((4, 40)) * 3.0 + 12.0     # task2 far away
    def mk(cents, n, rr, off):
        X = np.concatenate([c + rr.standard_normal((n, 40)) for c in cents])
        return X, np.repeat(np.arange(4) + off, n)
    X1, y1 = mk(c1, 150, r2, 0); X2, y2 = mk(c2, 150, r2, 4)
    X1t, y1t = mk(c1, 80, r2, 0); X2t, y2t = mk(c2, 80, r2, 4)
    base = elm_train(X1, y1, hidden=256, n_classes=8, seed=7)
    donor = elm_train(np.vstack([X1, X2]), np.concatenate([y1, y2]),
                      hidden=256, n_classes=8, seed=7)     # SAME basis seed
    tau, tinfo = task_vector(base, donor)
    imbued = imbue(base, tau, policy=False)               # instrument has no embeds
    accs = lambda t: (float(np.mean(elm_predict(t, X1t) == y1t)),
                      float(np.mean(elm_predict(t, X2t) == y2t)))
    a_base, a_donor, a_imb = accs(base), accs(donor), accs(imbued)
    assert a_base[0] > 0.9 and a_base[1] < 0.4, a_base    # base: task1 only
    assert a_imb[1] > 0.9, (a_base, a_imb)                # GAINED task2
    assert a_imb[0] > 0.85, a_imb                         # KEPT task1
    stranger = elm_train(X1, y1, hidden=256, n_classes=8, seed=99)  # different basis
    a_str = accs(imbue(stranger, tau, policy=False))
    assert a_str[1] < 0.6, a_str                          # lineage law: no transfer

    # 20) ARCHIVE: the storage ladder on a fleet of sibling models. Three ELM
    #     fine-tunes sharing one seeded random basis: W1 rides the RECIPE rung
    #     (seed, not data), shared tensors ride SAME, fine-tune W2s ride DELTA,
    #     and reconstruction is BIT-EXACT (hash parity per tensor). The kept
    #     negative rides along: a trained tensor with no reference lands on
    #     RAW -- the archive never invents a seed for the residue of data.
    Xa, ya = mk(c1, 150, r2, 0)
    base_m = elm_train(Xa, ya, hidden=256, n_classes=8, seed=7)
    ft1 = elm_train(np.vstack([Xa, X2]), np.concatenate([ya, y2]),
                    hidden=256, n_classes=8, seed=7)
    ft2 = dict(base_m)
    ft2["w2.weight"] = base_m["w2.weight"] * 1.001 + 0.001
    fleet = {"base": base_m, "ft1": ft1, "ft2": ft2}
    # RECIPE rung eligibility is verified against the LIVE generator: w1.weight
    # is the first draw from default_rng(seed) scaled 1/sqrt(d) -- checked, not
    # assumed, so the recipe can never drift from elm_train's actual code
    w1_regen = np.random.default_rng(7).standard_normal(
        base_m["w1.weight"].shape) / np.sqrt(Xa.shape[1])
    rec = ({(mn, "w1.weight"): dict(kind="standard_normal", seed=7,
                                    shape=list(base_m["w1.weight"].shape),
                                    div=float(np.sqrt(Xa.shape[1])))
            for mn in fleet}
           if np.array_equal(w1_regen, base_m["w1.weight"]) else {})
    arc, arep = archive_models(fleet, reference="base", recipes=rec)
    for mn in fleet:
        back = restore_model(arc, mn)
        for tn in fleet[mn]:
            assert _tensor_hash(back[tn]) == _tensor_hash(
                np.ascontiguousarray(fleet[mn][tn])), (mn, tn)
    assert arep["ratio"] > 2.0, arep["ratio"]
    assert arep["per_model"]["ft2"]["DELTA"] >= 1        # near-sibling -> delta
    if rec:
        assert arep["per_model"]["base"]["RECIPE"] >= 1  # seed rung exercised
    # the RECIPE rung's justification, MEASURED via HRNN rather than asserted:
    # a seed-born tensor is statistically white, so no gate can find its
    # generator -- provenance must be supplied, never discovered.
    assert generator_audit(np.random.default_rng(11).standard_normal(4096)
                           )["discoverable"] is False
    assert generator_audit(base_m["w2.weight"])["discoverable"] is False

    lone, lrep = archive_models({"stranger": {"W": rng.standard_normal((64, 64))
                                              @ np.diag(np.arange(1, 65.0))}},
                                reference={"nothing": np.zeros(1)})
    assert lrep["per_model"]["stranger"]["RAW"] == 1     # honesty rung

    # 20) MIDDLE-OUT: progressive decode contract + the pinned refutations.
    Wmo = rng.standard_normal((64, 96))
    codemo = middle_out_encode(Wmo, n_refine=5, base_bits=3, max_bits=9)
    errs, sizes = [], []
    for n in range(len(codemo["refinements"]) + 1):
        rec = middle_out_decode(codemo, n_refine=n)
        errs.append(float(np.linalg.norm(Wmo - rec) / np.linalg.norm(Wmo)))
        sizes.append(middle_out_bytes(codemo, n_refine=n))
    # progressive: every extra layer strictly helps and strictly costs
    for i in range(1, len(errs)):
        assert errs[i] < errs[i - 1], (i, errs)
        assert sizes[i] > sizes[i - 1], (i, sizes)
    assert errs[-1] < 0.02, errs[-1]
    # full-depth decode must equal a direct 9-bit quantization: the stream is a
    # RE-ORDERING of the same information, not a different code
    sc = float(np.max(np.abs(Wmo)))
    direct = np.rint(Wmo / sc * 255) / 255 * sc
    # all 6 planes (9 bits - 3 base bits), not 5: a prefix is only exact when the
    # stream is COMPLETE -- the first version of this assert sent 5 and blamed
    # the codec, an instrument error caught by the codec being right.
    full = middle_out_decode(middle_out_encode(Wmo, n_refine=6, base_bits=3, max_bits=9))
    assert np.max(np.abs(full - direct)) < 1e-12, "prefix code must reconstruct exactly"
    # REFUTATION PINNED (do not reinvent): at matched bytes, middle-out is at
    # parity with flat uniform quantization -- never better. Measured here so a
    # future "win" is immediately suspect.
    b_mo = middle_out_bytes(codemo, n_refine=3)
    q6 = np.rint(Wmo / sc * 31).astype(np.int16)
    b_flat = len(zlib.compress(q6.tobytes(), 6))
    e_flat = float(np.linalg.norm(Wmo - q6 / 31 * sc) / np.linalg.norm(Wmo))
    e_mo = errs[3]
    assert e_mo >= 0.5 * e_flat or b_mo >= 0.5 * b_flat, \
        "a middle-out WIN over flat quantization contradicts 3 measurements -- " \
        "hunt the bug or the strawman baseline before believing it"

    # 21) COMPRESSED RESIDENCY: a lazy store must be BIT-EXACT at full depth,
    #     genuinely smaller, and correct under LRU eviction (the eviction path is
    #     where a cache silently serves stale tensors if the LRU is wrong).
    lw_src = {"model.layers.0.mlp.gate_proj.weight": rng.standard_normal((96, 128)),
              "model.layers.1.mlp.gate_proj.weight": rng.standard_normal((96, 128)),
              "model.layers.2.mlp.gate_proj.weight": rng.standard_normal((96, 128)),
              "model.norm.weight": rng.standard_normal(96)}
    lw = LazyWeights(lw_src, max_cached=1, n_refine=6, base_bits=3, max_bits=9)
    for name, t in lw_src.items():
        direct = np.asarray(t) if np.asarray(t).ndim < 2 or np.asarray(t).size < 4096 \
            else middle_out_decode(middle_out_encode(t, n_refine=6, base_bits=3,
                                                     max_bits=9))
        assert np.array_equal(lw[name], direct), name
    # re-read after eviction (max_cached=1 guarantees each read above evicted the
    # previous): values must be identical, not merely close
    for name in lw_src:
        assert np.array_equal(lw[name], lw[name])
    assert lw.stats["misses"] >= 3
    sb = lw.stored_bytes()
    assert sb["total"] < 0.7 * sb["dense"], sb
    # portable export round-trips through the ordinary loader
    _pp = os.path.join(tempfile.mkdtemp(), "portable.safetensors")
    rep_pp = export_portable(lw, _pp)
    back_pp = load_safetensors(_pp)
    for name in lw_src:
        assert np.allclose(back_pp[name], lw[name], atol=1e-6), name
    assert rep_pp["tensors"] == len(lw_src)

    # 22) DELTA STORAGE: unchanged tensors cost nothing, touched ones go
    #     low-rank at their OWN discovered rank, and round-trip is exact enough
    #     to preserve function. Uses the ELM instrument BECAUSE its W1 is frozen
    #     random -- so its delta is exactly zero, which is the property under
    #     test (a fine-tune does not touch everything).
    rdA = np.random.default_rng(80); rdB = np.random.default_rng(81)
    ca = rdA.standard_normal((4, 40)) * 3.0
    cb = rdA.standard_normal((4, 40)) * 3.0 + 12.0
    mkd = lambda cents, n, off: (
        np.concatenate([c + rdB.standard_normal((n, 40)) for c in cents]),
        np.repeat(np.arange(4) + off, n))
    XA, yA = mkd(ca, 150, 0); XB, yB = mkd(cb, 150, 4)
    XAt, yAt = mkd(ca, 60, 0); XBt, yBt = mkd(cb, 60, 4)
    d_base = elm_train(XA, yA, hidden=256, n_classes=8, seed=7)
    d_ft = elm_train(np.vstack([XA, XB]), np.concatenate([yA, yB]),
                     hidden=256, n_classes=8, seed=7)
    dpack = delta_encode(d_base, d_ft)
    assert dpack["report"]["unchanged"] >= 1, dpack["report"]
    assert dpack["report"]["ratio"] > 1.5, dpack["report"]
    rebuilt = delta_apply(d_base, dpack)
    acc_ft = float(np.mean(elm_predict(d_ft, XBt) == yBt))
    acc_rb = float(np.mean(elm_predict(rebuilt, XBt) == yBt))
    assert abs(acc_rb - acc_ft) < 0.02, (acc_ft, acc_rb)   # function preserved
    assert float(np.mean(elm_predict(rebuilt, XAt) == yAt)) > 0.85
    # scale=0 must return the base exactly -- the interpolation knob is honest
    at_zero = delta_apply(d_base, dpack, scale=0.0)
    for _k in d_base:
        assert np.allclose(at_zero[_k], d_base[_k]), _k

    # 22b) D-QRELO mode (1-bit dominant + low-rank residual, arXiv 2604.16940)
    #      must round-trip and preserve function too. Both modes are kept: the
    #      literature's motivation is LARGE-SFT deltas, which this instrument
    #      does not produce, so no ratio winner is declared here -- the honest
    #      statement is that both are available and must be priced per subject.
    qpack = delta_encode(d_base, d_ft, mode="qlr")
    q_rebuilt = delta_apply(d_base, qpack)
    acc_q = float(np.mean(elm_predict(q_rebuilt, XBt) == yBt))
    assert abs(acc_q - acc_ft) < 0.02, (acc_ft, acc_q)
    assert qpack["report"]["ratio"] > 1.0, qpack["report"]

    # 22c) LINEAGE from weights alone: the true base must win, with a margin.
    #      (TStore, arXiv 2604.17104, names missing lineage metadata as an open
    #      limitation of delta compression at scale; this answers it from the
    #      weights instead of from a model card.)
    stranger = elm_train(XA, yA, hidden=256, n_classes=8, seed=99)
    other = elm_train(XB, yB, hidden=256, n_classes=8, seed=123)
    lin = delta_lineage(d_ft, {"true_base": d_base, "stranger": stranger,
                               "other": other})
    assert lin["best"] == "true_base", lin
    assert lin["margin"] > 0.01, lin

    # 23) FRONT DOOR: one call must classify regimes, find structure, and carry
    #     the refutations. Built on a qwen-shaped subject so the census is real.
    import sys as _sys
    _sys.path.insert(0, "tools")
    try:
        from rehearse_qwen_assimilation import make_qwen_shaped
        subj = make_qwen_shaped(hidden=128, layers=6, vocab=800, ffn=448)
    except Exception:
        subj = {"model.layers.%d.mlp.gate_proj.weight" % i:
                rng.standard_normal((96, 128)) for i in range(4)}
    frep = full_report(subj, sample_layers=6)
    assert frep["census"]["examined"] >= 2, frep["census"]
    assert frep["levers"] and frep["warnings"]
    # the refutation list must ALWAYS ship -- a report that only lists what may
    # work is how a refuted lever gets retried
    assert any("REFUTED" in w for w in frep["warnings"])

    print("holographic_unicron selftest OK")









if __name__ == "__main__":
    _selftest()
