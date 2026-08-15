"""Safe, NumPy-only checkpoint I/O used by UNICRON and model runtimes.

This module owns the file-format boundary so spectral analysis and model
transformation do not have to share a growing monolith with binary parsers.
Safetensors payloads are memory-mapped, BF16 values are decoded on demand, and
pickle-based torch checkpoints remain deliberately unsupported.
"""

import json
import os
import struct
import zipfile
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np


# safetensors dtype strings -> numpy dtype used to read raw bytes.
# BF16 has no NumPy dtype: read as uint16, shift into a float32.
_ST_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def _decode_bf16(raw_u16):
    """Decode bfloat16 exactly by placing its bits in an IEEE float32."""
    u32 = raw_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


class SafetensorWeights(Mapping):
    """Read-only, file-backed mapping over one or more safetensors shards.

    Native NumPy dtypes remain mmap views. BF16 has no NumPy dtype, so only the
    tensor currently requested is decoded to float32 and held in a bounded LRU.
    """

    def __init__(self, paths, max_cached=8):
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        self._entries = OrderedDict()
        self._cache = OrderedDict()
        self._max_cached = max(0, int(max_cached))
        self.stats = {"hits": 0, "misses": 0, "decoded_bytes": 0}
        for path in paths:
            path = os.fspath(path)
            with open(path, "rb") as fh:
                raw = fh.read(8)
                if len(raw) != 8:
                    raise ValueError("truncated safetensors header in %s" % path)
                (hdr_len,) = struct.unpack("<Q", raw)
                header = json.loads(fh.read(hdr_len).decode("utf-8"))
            data_start = 8 + hdr_len
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                if name in self._entries:
                    raise ValueError("duplicate tensor %r across safetensors shards" % name)
                self._entries[name] = (path, data_start, dict(meta))

    @property
    def dtypes(self):
        return {name: entry[2]["dtype"] for name, entry in self._entries.items()}

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, name):
        if name in self._cache:
            self.stats["hits"] += 1
            value = self._cache.pop(name)
            self._cache[name] = value
            return value
        self.stats["misses"] += 1
        path, data_start, meta = self._entries[name]
        a, b = map(int, meta["data_offsets"])
        shape = tuple(int(v) for v in meta["shape"])
        disk_dtype = meta["dtype"]
        if disk_dtype == "BF16":
            raw = np.memmap(path, dtype="<u2", mode="r",
                            offset=data_start + a, shape=((b - a) // 2,))
            value = _decode_bf16(raw).reshape(shape)
            self.stats["decoded_bytes"] += int(value.nbytes)
        else:
            if disk_dtype not in _ST_DTYPES:
                raise ValueError("unsupported safetensors dtype: %s" % disk_dtype)
            dtype = np.dtype(_ST_DTYPES[disk_dtype]).newbyteorder("<")
            value = np.memmap(path, dtype=dtype, mode="r",
                              offset=data_start + a,
                              shape=((b - a) // dtype.itemsize,)).reshape(shape)
        if self._max_cached:
            self._cache[name] = value
            while len(self._cache) > self._max_cached:
                self._cache.popitem(last=False)
        return value


def load_safetensors(path, return_dtypes=False, lazy=False, max_cached=8):
    """Parse safetensors into an eager dict or a lazy file-backed mapping."""
    store = SafetensorWeights(path, max_cached=max_cached)
    if lazy:
        if return_dtypes:
            return store, store.dtypes
        return store
    out = {name: np.array(store[name], copy=True) for name in store}
    if return_dtypes:
        return out, store.dtypes
    return out


def _encode_bf16(f32):
    """Encode float32 as BF16 with round-to-nearest-even."""
    u32 = np.ascontiguousarray(f32, np.float32).view(np.uint32)
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)


def save_safetensors(path, tensors, dtypes=None):
    """Write tensors in safetensors format while preserving declared dtypes."""
    inv = {v: k for k, v in _ST_DTYPES.items()}
    dtypes = dtypes or {}
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
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
    encoded_header = json.dumps(header, sort_keys=True).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded_header)))
        fh.write(encoded_header)
        for blob in blobs:
            fh.write(blob)


# llama.cpp GGUF identifiers supported by this lightweight reader/writer.
_GGML_F32, _GGML_F16, _GGML_Q8_0, _GGML_BF16 = 0, 1, 8, 30
_GGUF_MAGIC = 0x46554747


def _gguf_read_str(fh):
    (length,) = struct.unpack("<Q", fh.read(8))
    return fh.read(length).decode("utf-8")


_GGUF_SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
                 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
                 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def _gguf_read_value(fh, value_type):
    if value_type in _GGUF_SCALARS:
        fmt, size = _GGUF_SCALARS[value_type]
        return struct.unpack(fmt, fh.read(size))[0]
    if value_type == 8:
        return _gguf_read_str(fh)
    if value_type == 9:
        (element_type,) = struct.unpack("<I", fh.read(4))
        (length,) = struct.unpack("<Q", fh.read(8))
        return [_gguf_read_value(fh, element_type) for _ in range(length)]
    raise ValueError("unknown GGUF kv type: %d" % value_type)


def _dequant_q8_0(raw, count):
    """Decode Q8_0 blocks of one f16 scale and 32 signed values."""
    block = np.dtype([("d", "<f2"), ("q", "i1", (32,))])
    values = np.frombuffer(raw, dtype=block, count=(count + 31) // 32)
    out = (values["d"].astype(np.float32)[:, None]
           * values["q"].astype(np.float32)).reshape(-1)
    return out[:count]


def load_gguf(path):
    """Parse F32/F16/BF16/Q8_0 tensors from a GGUF v2+ file."""
    with open(path, "rb") as fh:
        magic, version = struct.unpack("<II", fh.read(8))
        if magic != _GGUF_MAGIC:
            raise ValueError("not a GGUF file (bad magic)")
        if version < 2:
            raise ValueError("GGUF v1 uses u32 counts; only v2+ supported")
        n_tensors, n_kv = struct.unpack("<QQ", fh.read(16))
        metadata = {}
        for _ in range(n_kv):
            key = _gguf_read_str(fh)
            (value_type,) = struct.unpack("<I", fh.read(4))
            metadata[key] = _gguf_read_value(fh, value_type)
        infos = []
        for _ in range(n_tensors):
            name = _gguf_read_str(fh)
            (n_dims,) = struct.unpack("<I", fh.read(4))
            dims = struct.unpack("<%dQ" % n_dims, fh.read(8 * n_dims))
            ggml_type, offset = struct.unpack("<IQ", fh.read(12))
            infos.append((name, dims, ggml_type, offset))
        alignment = int(metadata.get("general.alignment", 32))
        position = fh.tell()
        data_start = ((position + alignment - 1) // alignment) * alignment
        fh.seek(0, 2)
        end = fh.tell()
        out = {}
        for name, dims, ggml_type, offset in infos:
            count = int(np.prod(dims))
            shape = tuple(int(dim) for dim in reversed(dims))
            fh.seek(data_start + offset)
            if ggml_type == _GGML_F32:
                array = np.frombuffer(fh.read(4 * count), dtype=np.float32)
            elif ggml_type == _GGML_F16:
                array = np.frombuffer(fh.read(2 * count), dtype=np.float16).astype(np.float32)
            elif ggml_type == _GGML_BF16:
                array = _decode_bf16(np.frombuffer(fh.read(2 * count), dtype=np.uint16))
            elif ggml_type == _GGML_Q8_0:
                nbytes = ((count + 31) // 32) * 34
                array = _dequant_q8_0(fh.read(nbytes), count)
            else:
                raise ValueError("GGUF tensor %r has unsupported ggml type %d; "
                                 "convert this quant to f16/f32 upstream"
                                 % (name, ggml_type))
            out[name] = array.reshape(shape).copy()
        if end < data_start:
            raise ValueError("truncated GGUF data section")
    return out


def save_gguf(path, tensors, quant=None):
    """Write a minimal GGUF v3 file with F32 and optional Q8_0 tensors."""
    quant = set(quant or ())
    infos, blobs, offset = [], [], 0
    for name in sorted(tensors):
        array = np.ascontiguousarray(np.asarray(tensors[name], dtype=np.float32))
        dims = tuple(reversed(array.shape))
        if name in quant:
            flat = array.reshape(-1)
            padding = (-flat.size) % 32
            padded = np.concatenate([flat, np.zeros(padding, np.float32)]).reshape(-1, 32)
            scale = np.max(np.abs(padded), axis=1) / 127.0
            scale[scale == 0] = 1.0
            values = np.clip(np.round(padded / scale[:, None]), -127, 127).astype(np.int8)
            block = np.empty(padded.shape[0],
                             dtype=np.dtype([("d", "<f2"), ("q", "i1", (32,))]))
            block["d"] = scale.astype(np.float16)
            block["q"] = values
            raw, ggml_type = block.tobytes(), _GGML_Q8_0
        else:
            raw, ggml_type = array.tobytes(), _GGML_F32
        infos.append((name, dims, ggml_type, offset))
        blobs.append(raw)
        offset += len(raw) + ((-len(raw)) % 32)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IIQQ", _GGUF_MAGIC, 3, len(infos), 1))
        key = b"general.alignment"
        fh.write(struct.pack("<Q", len(key)))
        fh.write(key)
        fh.write(struct.pack("<II", 4, 32))
        for name, dims, ggml_type, tensor_offset in infos:
            encoded_name = name.encode("utf-8")
            fh.write(struct.pack("<Q", len(encoded_name)))
            fh.write(encoded_name)
            fh.write(struct.pack("<I", len(dims)))
            fh.write(struct.pack("<%dQ" % len(dims), *dims))
            fh.write(struct.pack("<IQ", ggml_type, tensor_offset))
        fh.write(b"\x00" * ((-fh.tell()) % 32))
        for raw in blobs:
            fh.write(raw)
            fh.write(b"\x00" * ((-len(raw)) % 32))


def load_model(path):
    """Load safe model formats and refuse pickle-based torch checkpoints."""
    model_path = str(path)
    if model_path.endswith(".safetensors"):
        return load_safetensors(model_path)
    if model_path.endswith(".gguf"):
        return load_gguf(model_path)
    if model_path.endswith(".npz"):
        with np.load(model_path) as archive:
            return {name: archive[name] for name in archive.files}
    if model_path.endswith((".pt", ".bin", ".pth")) or zipfile.is_zipfile(model_path):
        raise ValueError("torch pickle checkpoints are refused (unpickling is an "
                         "ACE surface); convert to .safetensors or .npz first")
    raise ValueError("unknown model format: %s" % model_path)


def _selftest():
    """Exercise the safe format boundary without external model fixtures."""
    import tempfile

    tensors = {
        "f32": np.arange(24, dtype=np.float32).reshape(6, 4),
        "f16": np.linspace(-1.0, 1.0, 12, dtype=np.float16).reshape(3, 4),
    }
    with tempfile.TemporaryDirectory() as directory:
        safetensors_path = os.path.join(directory, "model.safetensors")
        save_safetensors(safetensors_path, tensors)
        loaded = load_model(safetensors_path)
        assert np.array_equal(loaded["f32"], tensors["f32"])
        assert np.array_equal(loaded["f16"], tensors["f16"])

        lazy = load_safetensors(safetensors_path, lazy=True, max_cached=1)
        assert isinstance(lazy["f32"], np.memmap)
        assert lazy.stats["misses"] == 1

        npz_path = os.path.join(directory, "model.npz")
        np.savez(npz_path, **tensors)
        assert np.array_equal(load_model(npz_path)["f32"], tensors["f32"])

        refused_path = os.path.join(directory, "unsafe.pt")
        with open(refused_path, "wb") as fh:
            fh.write(b"not a safe model format")
        try:
            load_model(refused_path)
        except ValueError as exc:
            assert "pickle checkpoints are refused" in str(exc)
        else:
            raise AssertionError("pickle checkpoint was not refused")

    print("checkpointio selftest OK")


if __name__ == "__main__":
    _selftest()
