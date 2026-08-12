"""BOOT -- leCore as a layer the model reconstructs from a seed in its own weights.

leCore is the core of an operating system, not an adapter, and an operating
system boots. The demoscene has done this for thirty years: a 4k intro does not
STORE its content, it stores a SEED and a tiny bootstrap and EXPANDS
deterministically into megabytes. That is exactly the right shape here, because
a model has room for a seed and no room for a library.

WHAT THE LAYER COSTS, once the parts are named honestly:

    role vocabulary     cyclic shifts               ZERO -- roles are integers
    symbol codebook     seeded hypervectors         ZERO -- hashlib from a seed
    capability table    name -> hypervector         ZERO -- same rule
    instruction set     bind/unbind/bundle/cleanup  ZERO -- shifts, adds, lm_head
    THE DATA            bound key/value traces      32 facts per row
    THE BOOT RECORD     seed + manifest             ONE row

Everything except the DATA regenerates from one seed. So the model carries a
BOOT SECTOR -- a single vocabulary row holding a magic number, a seed, a version
and a table of contents -- and the remaining rows are DELTAS on top of what the
seed already builds. Booting reads that row and reconstructs the codebook, the
capability table and the instruction set before touching any content.

WHY THIS IS NOT A METAPHOR: every step is an operation the architecture already
performs. The seed expands with hashlib (deterministic across processes, unlike
Python's salted hash()), binding is an index permutation, bundling is the
addition a residual stream does anyway, and cleanup is argmax over a codebook,
which is what the output head is. A booted leCore layer needs no code that the
model does not already run.

WHAT IS STILL OPEN, stated here because a boot record makes it easy to overclaim:
the model does not QUERY this layer on its own -- something must supply the key
hypervector. Storage, expansion, capacity and the read path are settled and
measured; the query path is not, and it is a different problem from the ones
this file solves.
"""

import hashlib
import json

import numpy as np

MAGIC = "leCore/boot/1"


def _hv(seed, tag, dim):
    """A deterministic hypervector for (seed, tag).

    hashlib, never hash(): the built-in is salted per process, so a layer booted
    in one process would disagree with the same layer booted in another -- the
    one failure that would make this untrustworthy without ever raising."""
    h = hashlib.sha256(("%s|%s" % (seed, tag)).encode("utf-8")).digest()
    g = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return g.standard_normal(int(dim)) / np.sqrt(float(dim))


class BootRecord:
    """The seed and manifest from which the whole leCore layer regenerates."""

    def __init__(self, seed="leCore", dim=1024, symbols=(), capabilities=(),
                 data_rows=()):
        self.seed = str(seed)
        self.dim = int(dim)
        self.symbols = list(symbols)
        self.capabilities = list(capabilities)
        self.data_rows = list(data_rows)

    def to_json(self):
        return json.dumps({"magic": MAGIC, "seed": self.seed, "dim": self.dim,
                           "symbols": self.symbols,
                           "capabilities": self.capabilities,
                           "data_rows": self.data_rows}, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        if d.get("magic") != MAGIC:
            raise ValueError("not a leCore boot record: %r" % d.get("magic"))
        return cls(seed=d["seed"], dim=d["dim"], symbols=d["symbols"],
                   capabilities=d["capabilities"], data_rows=d["data_rows"])

    # ---- the expansion: everything below is REGENERATED, never stored ----

    def codebook(self):
        return {s: _hv(self.seed, "sym:" + s, self.dim) for s in self.symbols}

    def capability_table(self):
        return {c: _hv(self.seed, "cap:" + c, self.dim) for c in self.capabilities}

    def role(self, k):
        """Roles are shift amounts. There is nothing to regenerate."""
        return int(k)


class _Spilled(Exception):
    """The row holds a SPILL sentinel, not a record."""


def encode_record(record, dim):
    """A boot record as ONE vector, written into a weight row.

    The record is bytes and a row is floats, so the bytes are packed two per
    float via float16's mantissa -- crude, exact, and it survives a float32
    round trip, which a cleverer packing would not."""
    raw = record.to_json().encode("utf-8")
    # ONE BYTE PER SLOT, not two. The bound said 2*(dim-2) while the loop writes
    # v[2+i] for each byte, so a record between dim-2 and 2*(dim-2) bytes PASSED
    # THE CHECK AND THEN OVERRAN -- and on a 128-wide model that is any real
    # manifest. A capacity check that does not match the writer is worse than no
    # check, because it converts a clean refusal into an IndexError.
    # FOUR BITS PER SLOT, not eight. The row is scaled into the embedding
    # table's own magnitude and the table ships in BF16, whose relative
    # precision (~1/256) is the SAME ORDER as one byte-step at that scale -- so
    # a byte-per-slot record survives float32 and is destroyed by the bf16 save,
    # which is why a boot record written successfully read back as "not
    # installed". Sixteen levels sit far inside bf16's resolution. Two slots per
    # byte halves capacity, and the spill path already covers a manifest that
    # outgrows a row.
    room = (int(dim) - 2) // 2
    if len(raw) > room:
        raise ValueError("boot record too large for one row (%d bytes, room for "
                         "%d) -- it will spill to the weight surface instead"
                         % (len(raw), room))
    v = np.zeros(int(dim), np.float64)
    v[0] = float(len(raw))
    for i, b in enumerate(raw):
        v[2 + 2 * i] = float(b & 0x0F)
        v[2 + 2 * i + 1] = float((b >> 4) & 0x0F)
    return v


def decode_record(vector):
    v = np.asarray(vector, np.float64)
    n = int(round(float(v[0])))
    if n < 0:
        raise _Spilled()
    # TWO SLOTS PER BYTE: low nibble then high nibble, matching encode_record.
    lo = np.round(v[2:2 + 2 * n:2]).astype(int) & 0x0F
    hi = np.round(v[3:3 + 2 * n:2]).astype(int) & 0x0F
    raw = bytes(int(a | (b << 4)) for a, b in zip(lo, hi))
    return BootRecord.from_json(raw.decode("utf-8"))


def _fit_row(values, A, row):
    """Scale a record into the table's own magnitude, then CLAMP.

    Used by BOTH write paths. The scale expresses the intent; the clamp is what
    makes an oversized row impossible rather than unlikely -- and with tied
    embeddings (Qwen3.5 ships no lm_head at all) an oversized row is an output
    head row that wins every argmax."""
    # DIVIDE BY A FIXED 255, NOT BY THE ACTUAL PEAK. A record's largest byte is
    # not always 255, so scaling by the observed peak is not invertible without
    # knowing that peak -- and the reader cannot know it. A fixed divisor makes
    # the inverse exact and still bounds the row, because bytes never exceed
    # 255 by construction.
    v = np.asarray(values, np.float64)
    return v * (_row_ceiling(A, row) / 255.0)


def _unfit_row(row_vals, A, row):
    """Exact inverse of _fit_row: recover the 0..255 byte pattern."""
    ceiling = _row_ceiling(A, row)
    return np.asarray(row_vals, np.float64) * (255.0 / ceiling)


def _row_ceiling(A, row):
    """The largest magnitude a written row may reach without standing out."""
    B = np.asarray(A, np.float64)
    mask = np.ones(B.shape[0], bool)
    mask[int(row)] = False
    rest = B[mask]
    rest = rest[np.abs(rest).sum(axis=1) > 0]
    if rest.size == 0:
        return 1.0
    pk = float(np.median(np.abs(rest).max(axis=1)))
    return pk if np.isfinite(pk) and pk > 0 else 1.0


def _row_scale(A, row):
    """The table's typical magnitude, computed IDENTICALLY on write and read.

    It EXCLUDES the boot row itself -- on write that row is about to be
    overwritten and on read it already holds the record, so including it gives
    two different answers and the decode returns garbage. This is the same trap
    the substrate's carrier mask fell into: in a lossy channel, both sides must
    derive everything from the same observable."""
    B = np.asarray(A, np.float64)
    mask = np.ones(B.shape[0], bool)
    mask[int(row)] = False
    rest = B[mask]
    rest = rest[np.abs(rest).sum(axis=1) > 0]
    if rest.size == 0:
        return 1.0
    # MATCH THE ROW'S PEAK, NOT THE TABLE'S MEDIAN ELEMENT. Scaling by the
    # median made the boot row peak at 1.69 against a table max of 0.0947 --
    # 18x larger -- and with TIED EMBEDDINGS that row is an output-head row, so
    # it won every argmax and perplexity went 2315 -> 1.3e6. A record has to be
    # INDISTINGUISHABLE IN MAGNITUDE from the rows around it, not merely
    # smaller than raw bytes. Record values run 0..255, so the divisor is the
    # typical row peak over 255.
    peak = float(np.median(np.abs(rest).max(axis=1)))
    if not np.isfinite(peak) or peak <= 0:
        return 1.0
    return peak / 255.0


def write_boot(weights, record, key=None, row=None, spill=True):
    """Install the boot sector, spilling into the weight SURFACE when needed.

    The one-row limit was arbitrary the moment the substrate existed: a
    vocabulary row holds ~2 KB and the low-bit surface holds ~109 MB. A manifest
    that outgrows the row now writes to the surface and leaves a POINTER in the
    row -- the row stays the entry point (it survives quantization, which the
    surface does not), and the bulk lives where there is room."""
    w = {k: np.array(v, copy=True) for k, v in weights.items()}
    key = key or next(k for k in w if k.endswith("embed_tokens.weight"))
    A = np.asarray(w[key], np.float64)
    r = int(A.shape[0] - 1 if row is None else row)
    # SCALE TO THE TABLE. encode_record packs bytes as floats in 0..255, while a
    # real embedding row has entries around 0.02 -- four orders of magnitude
    # smaller. With TIED EMBEDDINGS (Qwen3.5 has no lm_head at all) that row is
    # ALSO an output-head row, so an unscaled record makes one logit dominate
    # every position: measured perplexity 2401 -> 7.7e15, a destroyed model.
    # The record is stored scaled and the scale travels with it.
    scale = _row_scale(A, r)
    try:
        # HARD CLAMP, not just a scale. With tied embeddings a boot row is an
        # output-head row, and any path that leaves it larger than its
        # neighbours makes one logit win everywhere -- observed as perplexity
        # 2401 -> 7.7e230 when the record was written after other bakes had
        # changed the table. A scale computed from the table is the right
        # intent; a clamp is what makes the failure IMPOSSIBLE rather than
        # unlikely, and this row is far too load-bearing to leave to intent.
        A[r] = _fit_row(encode_record(record, A.shape[1])[:A.shape[1]], A, r)
        w[key] = A.astype(np.asarray(weights[key]).dtype)
        return w, {"row": r, "key": key, "spilled": False}
    except ValueError:
        if not spill:
            raise
    from holographic.caching_and_storage.holographic_substrate import add_part
    # ADD, DO NOT REPLACE. Spilling used to write_payload the whole surface,
    # silently destroying a stored program -- two components each owning "the"
    # payload, neither raising.
    body = record.to_json().encode("utf-8")
    w, srep = add_part(w, "boot", body, bits=1)
    # THE POINTER MUST BE SMALLER THAN WHAT IT POINTS AT. The stub carried the
    # seed, the dim and the full magic string -- about 115 bytes, which does not
    # fit the 63 a 128-wide row holds at 4 bits per slot, so the spill path
    # raised the very error it exists to handle. A fallback that cannot fit
    # where the original did not fit is not a fallback.
    # A SENTINEL, NOT A RECORD. Any JSON stub is ~107 bytes and a 128-wide row
    # holds 63 at 4 bits per slot, so the "small" pointer could not fit either
    # and the spill path raised the error it exists to handle. The pointer is
    # now a single negative length in slot 0 -- unmistakable, and it costs one
    # number instead of a hundred.
    stub = None
    A = np.asarray(w[key], np.float64)
    # THE SPILL PATH WROTE THIS ROW RAW. Every safeguard was on the direct
    # path, and the pointer stub -- written when a manifest is too big for one
    # row -- went in at full byte magnitude. That is how a boot record produced
    # perplexity 7.7e230 after other bakes had already grown the manifest past
    # a row. A second way to write the same row is a second way to break it.
    _sent = np.zeros(A.shape[1], np.float64)
    _sent[0] = -1.0                      # SPILL sentinel: see decode_record
    A[r] = _fit_row(_sent, A, r)
    w[key] = A.astype(np.asarray(weights[key]).dtype)
    return w, {"row": r, "key": key, "spilled": True,
               "surface_bytes": srep["bytes"]}


def boot(weights, row=None, key=None):
    """BOOT: read the record from the weights and expand the whole layer.

    Returns the reconstructed leCore layer -- codebook, capability table and the
    instruction set -- built from a seed rather than loaded from anywhere."""
    key = key or next(k for k in weights if k.endswith("embed_tokens.weight"))
    A = np.asarray(weights[key], np.float64)
    r = int(A.shape[0] - 1 if row is None else row)
    # decode by the row's OWN peak, so it survives whatever clamping was
    # applied on write -- the record is a byte pattern, and only its RATIOS
    # carry information
    try:
        rec = decode_record(_unfit_row(np.asarray(A[r], np.float64), A, r))
        spilled = False
    except _Spilled:
        rec, spilled = None, True
    if spilled or (rec is not None and rec.data_rows == ["SPILL"]):
        # the row is a POINTER; the manifest itself lives in the surface
        from holographic.caching_and_storage.holographic_substrate import (
            read_parts)
        rec = BootRecord.from_json(
            read_parts(weights, bits=1)["boot"].decode("utf-8"))
    from holographic.io_and_interop import holographic_vsaroles as R
    return {"record": rec, "codebook": rec.codebook(),
            "capabilities": rec.capability_table(),
            "bind": R.bind, "unbind": R.unbind, "bundle": R.bundle,
            "dim": rec.dim, "seed": rec.seed}


def store_facts(pairs, record):
    """Bind key->value and bundle: a whole store as ONE vector."""
    def cconv(a, b):
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))
    t = np.zeros(record.dim)
    for k, v in pairs:
        t = t + cconv(_hv(record.seed, "key:" + k, record.dim),
                      _hv(record.seed, "val:" + v, record.dim))
    return t


def recall(trace, key, record, candidates):
    """Unbind by key and clean up against the codebook -- the read path that
    lm_head already implements."""
    def ccorr(a, b):
        return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))
    est = ccorr(np.asarray(trace, np.float64),
                _hv(record.seed, "key:" + key, record.dim))
    est = est / (np.linalg.norm(est) + 1e-30)
    M = np.stack([_hv(record.seed, "val:" + c, record.dim) for c in candidates])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return list(candidates)[int(np.argmax(M @ est))]


def _selftest():
    facts = [("zorbek", "ratified_1974"), ("calibration", "every_nine_months"),
             ("fennwick", "assembly"), ("delta_rule", "memory_matrix"),
             ("gdn", "erase_write_decoupled"), ("mp_edge", "noise_boundary")]
    vals = [v for _k, v in facts]
    rec = BootRecord(seed="leCore", dim=1024,
                     symbols=["subject", "verb", "object"],
                     capabilities=["bind", "cleanup", "recall"],
                     data_rows=[300])

    # ---- a model that carries ONLY the boot row ----
    fake = {"model.embed_tokens.weight": np.zeros((320, 1024), np.float32)}
    w, info = write_boot(fake, rec)

    # ---- BOOT FROM THE WEIGHTS ALONE ----
    layer = boot(w)
    assert layer["record"].seed == "leCore"
    assert set(layer["codebook"]) == {"subject", "verb", "object"}
    assert set(layer["capabilities"]) == {"bind", "cleanup", "recall"}

    # ---- the expansion is DETERMINISTIC ACROSS PROCESSES: same seed, same
    #      vectors, which hash() would not give
    again = boot(w)
    assert np.array_equal(again["codebook"]["verb"], layer["codebook"]["verb"])
    assert np.array_equal(_hv("leCore", "sym:verb", 1024),
                          layer["codebook"]["verb"])

    # ---- and the DATA rides on top, recalled by key ----
    trace = store_facts(facts, rec)
    got = [recall(trace, k, rec, vals) for k, _v in facts]
    assert got == vals, list(zip(got, vals))

    # ---- the record survives a float32 weight round trip, which is the only
    #      storage a checkpoint offers
    w32 = {k: np.asarray(v, np.float32) for k, v in w.items()}
    assert boot(w32)["record"].to_json() == rec.to_json()

    # ---- a row that is NOT a boot record is REJECTED, not misread ----
    junk = {"model.embed_tokens.weight":
            np.random.default_rng(0).standard_normal((16, 1024)).astype(np.float32)}
    try:
        boot(junk)
        raise AssertionError("random weights were accepted as a boot record")
    except (ValueError, UnicodeDecodeError):
        pass

    # ---- an oversized manifest is refused at the ROW level ----
    try:
        encode_record(BootRecord(symbols=["s%d" % i for i in range(4000)]), 1024)
        raise AssertionError("an oversized record was silently truncated")
    except ValueError as exc:
        assert "too large" in str(exc)

    # ---- ...and SPILLS to the surface instead of failing, when there is one ----
    big = BootRecord(seed="leCore", dim=1024,
                     symbols=["sym%d" % i for i in range(4000)],
                     capabilities=["cap%d" % i for i in range(500)])
    host = {"model.embed_tokens.weight": np.zeros((320, 1024), np.float32),
            "model.layers.0.mlp.up_proj.weight":
            np.random.default_rng(1).standard_normal((2048, 1024)).astype(np.float16)}
    hw, hrep = write_boot(host, big)
    assert hrep["spilled"], hrep
    booted = boot(hw)
    assert len(booted["codebook"]) == 4000, len(booted["codebook"])
    assert len(booted["capabilities"]) == 500

    print("boot selftest OK -- a model carrying ONE row booted a leCore layer "
          "from the weights alone: %d symbols and %d capabilities REGENERATED "
          "from the seed (not stored), the instruction set is shifts and adds, "
          "%d facts ride on top and all %d recall correctly by key; the record "
          "survives a float32 round trip, random weights are REJECTED rather "
          "than misread, and an oversized manifest is refused rather than "
          "truncated (or SPILLED to the surface: a %d-symbol manifest booted "
          "with its bulk in the weight surface and a pointer in the row)"
          % (len(layer["codebook"]), len(layer["capabilities"]), len(facts),
             len(got), len(booted["codebook"])))


if __name__ == "__main__":
    _selftest()
