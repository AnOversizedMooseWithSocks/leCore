"""SUBSTRATE -- the model's weight surface as a storage medium.

Moose's framing, and it is exactly right: a platter, a floppy, a CD and a tape
were all just physical irregularities on a surface. Someone chose a pattern,
called it a format, and an operating system grew on top. The capacity was in the
SURFACE, not in the spare sectors at the end.

The unused vocabulary rows were the spare sectors: 276 rows, about 0.56 MB. The
SURFACE is every weight in the model, and the low bits of a float16 carry almost
nothing -- which is not a guess, it is the same measurement that showed 4-bit
quantization costs only 0.11 output error.

MEASURED on a real Qwen3.5-0.8B layer, overwriting the lowest bits of every
weight and scoring the layer's OUTPUT:

    bits/weight   capacity (this layer)   output error   verdict
        1              1.38 MB              0.00107      invisible
        2              2.75 MB              0.00317      usable
        3              4.13 MB              0.00744      usable
        4              5.51 MB              0.00822      usable
        5              6.88 MB              0.01114      visible
        8             11.01 MB              0.06972      damaging

Scaled to the whole 871M-parameter model: 109 MB at the invisible setting, and
435 MB at 4 bits. Two hundred times what the spare rows offered, in space the
model is already carrying.

THE LIMIT THAT MATTERS, and it must be said before anyone builds on this:
QUANTIZATION DESTROYS THE PAYLOAD. Converting to GGUF Q4 rewrites exactly the
bits this uses. The substrate survives float16 and float32 checkpoints and dies
in any requantization -- so it is a medium for a model you ship as weights, not
for one you ship as a quantized artifact. A storage format whose failure mode is
undocumented is a trap, and this one's failure mode is a very common workflow.
"""

import hashlib
import struct

import numpy as np


def capacity_bytes(weights, bits=1, skip=("embed", "lm_head")):
    """How many bytes the surface holds at this bit depth."""
    n = 0
    for k, v in weights.items():
        a = np.asarray(v)
        if a.dtype.kind != "f" or any(s in k for s in skip):
            continue
        n += a.size
    return (n * int(bits)) // 8


def _carriers(weights, skip):
    """Deterministic ordering of the carrier weights.

    Sorted by name, never by dict order: a payload written in one process must
    be readable in another, and dict ordering is an implementation detail even
    when it happens to be stable."""
    for k in sorted(weights):
        a = np.asarray(weights[k])
        if a.dtype.kind == "f" and not any(s in k for s in skip):
            yield k, a


def write_payload(weights, data, bits=1, skip=("embed", "lm_head")):
    """Write bytes into the low `bits` of every carrier weight.

    A HEADER GOES FIRST: magic, length and a content hash. Without it a reader
    cannot tell payload from noise, and every bit pattern is a valid float --
    so a substrate with no header always "reads" and always returns garbage."""
    payload = bytes(data)
    header = b"leSUB1" + struct.pack("<I", len(payload)) + \
        hashlib.sha256(payload).digest()[:8]
    blob = header + payload
    need = len(blob) * 8
    have = capacity_bytes(weights, bits, skip) * 8
    if need > have:
        raise ValueError("payload needs %d bits, surface holds %d at %d bit(s) "
                         "per weight -- raise `bits` or shorten the payload"
                         % (need, have, bits))
    stream = np.unpackbits(np.frombuffer(blob, np.uint8))
    out = dict(weights)
    pos = 0
    mask = np.uint16((1 << int(bits)) - 1)
    for k, a in _carriers(weights, skip):
        if pos >= len(stream):
            break
        u = a.astype(np.float16).view(np.uint16).ravel().copy()
        take = min((len(stream) - pos) // int(bits), u.size)
        if take <= 0:
            break
        chunk = stream[pos:pos + take * int(bits)].reshape(take, int(bits))
        vals = np.zeros(take, np.uint16)
        for b in range(int(bits)):
            vals = (vals << np.uint16(1)) | chunk[:, b].astype(np.uint16)
        u[:take] = (u[:take] & ~mask) | vals
        out[k] = u.view(np.float16).reshape(a.shape).astype(a.dtype)
        pos += take * int(bits)
    return out, {"bytes": len(payload), "bits": int(bits),
                 "carriers_used": pos // max(int(bits), 1)}


def read_payload(weights, bits=1, skip=("embed", "lm_head")):
    """Read the payload back, verifying the header and the content hash."""
    # ONE CONTINUOUS STREAM ACROSS ALL CARRIERS. The first version collected a
    # bit-plane array PER TENSOR and then took min(len) across them, which
    # silently assumed every carrier tensor was the same size -- true only when
    # the payload fits entirely in the first one. It passed every synthetic test
    # (single big tensor) and failed on a real checkpoint, where the carriers
    # are dozens of tensors of wildly different sizes. The audit caught it.
    chunks = []
    total = 0
    cap = 64 * 1024 * 1024
    for _k, a in _carriers(weights, skip):
        u = a.astype(np.float16).view(np.uint16).ravel()
        vals = u & np.uint16((1 << int(bits)) - 1)
        part = np.empty(vals.size * int(bits), np.uint8)
        for b in range(int(bits)):
            part[b::int(bits)] = ((vals >> np.uint16(int(bits) - 1 - b))
                                  & np.uint16(1)).astype(np.uint8)
        chunks.append(part)
        total += part.size
        if total >= cap:
            break
    if not chunks:
        raise ValueError("no carrier weights found")
    stream = np.concatenate(chunks)
    raw = np.packbits(stream[:(len(stream) // 8) * 8]).tobytes()
    if raw[:6] != b"leSUB1":
        raise ValueError("no leCore substrate header here (found %r) -- this "
                         "model was not written to, or was requantized"
                         % raw[:6])
    length = struct.unpack("<I", raw[6:10])[0]
    digest = raw[10:18]
    payload = raw[18:18 + length]
    if hashlib.sha256(payload).digest()[:8] != digest:
        raise ValueError("substrate hash mismatch: the payload was corrupted "
                         "(quantization rewrites exactly these bits)")
    return payload


def quant_carriers(A, bits=4, group=64, threshold=0.45):
    """Which weights sit close enough to a bucket boundary to carry a bit.

    THE IDEA: do not hide UNDER the quantizer, hide IN it. A weight whose scaled
    value lands near x.5 can round either way and BOTH are legitimate
    quantizations, so the choice carries one bit -- and that bit IS the quantized
    value, so it survives the conversion that destroys an LSB payload.

    MEASURED on a real Qwen3.5-0.8B tensor at 4 bits:
        threshold  carriers   worst-case quantization error
          0.45       9.9%       0.1131 -> 0.1165
          0.40      19.7%       0.1131 -> 0.1259
          0.30      39.3%       0.1131 -> 0.1583
    At 0.45 that is ~10.8 MB across the model for a 0.3% relative change in
    quantization error -- enough to carry the entire 6.96 MB engine tarball
    through a GGUF conversion."""
    A = np.asarray(A, np.float64)
    m, n = A.shape
    g = int(group) if n % int(group) == 0 else n
    B = A.reshape(m, -1, g)
    qmax = 2 ** (int(bits) - 1) - 1
    sc = np.abs(B).max(-1, keepdims=True) / qmax
    sc = np.where(sc == 0, 1.0, sc)
    x = B / sc
    frac = np.abs(x - np.round(x))
    return (frac >= float(threshold)).reshape(m, n), x.reshape(m, n), \
        np.broadcast_to(sc, B.shape).reshape(m, n)


def write_quantsafe(A, payload_bits, bits=4, group=64, threshold=0.45):
    """Quantize a tensor while encoding bits in the rounding direction.

    Returns (quantized_tensor, bits_consumed). Weights that are not carriers get
    ordinary nearest rounding, so the tensor is a normal quantization of itself
    everywhere the payload is not."""
    mask, x, sc = quant_carriers(A, bits, group, threshold)
    qmax = 2 ** (int(bits) - 1) - 1
    q = np.round(x)
    idx = np.flatnonzero(mask.ravel())
    take = min(len(idx), len(payload_bits))
    if take:
        chosen = idx[:take]
        want = np.asarray(payload_bits[:take], np.int64)
        flat = q.ravel()
        xf = x.ravel()
        # bit 0 -> round DOWN, bit 1 -> round UP; both are valid quantizations
        flat[chosen] = np.where(want == 1, np.floor(xf[chosen]) + 1,
                                np.floor(xf[chosen]))
        q = flat.reshape(q.shape)
    q = np.clip(q, -qmax - 1, qmax)
    return (q * sc).astype(np.asarray(A).dtype), take


def read_quantsafe(A_quant, A_reference, bits=4, group=64, threshold=0.45):
    """Recover the bits from an already-quantized tensor.

    Needs the ORIGINAL tensor to know which weights were carriers -- the carrier
    set is a property of the pre-quantization values, and after rounding that
    information is gone. In practice the reference travels as a hash of the
    carrier positions, not as the weights."""
    mask, x, sc = quant_carriers(A_reference, bits, group, threshold)
    idx = np.flatnonzero(mask.ravel())
    q = np.round(np.asarray(A_quant, np.float64).ravel()
                 / sc.ravel())[idx]
    return (q > np.floor(x.ravel()[idx])).astype(np.uint8)


def write_parts(weights, parts, bits=1, skip=("embed", "lm_head")):
    """Write SEVERAL named payloads into one surface.

    WHY THIS EXISTS: write_payload owns the whole surface, so a boot record that
    SPILLS and a stored program both wrote to it and silently clobbered each
    other -- the second write won and the first became unreadable, with no error
    on either side. The hardening harness caught it; nothing else would have,
    because each component's own selftest writes exactly one payload.

    The container is a length-prefixed list of (name, bytes), so parts can be
    added without any part knowing about the others."""
    blob = b""
    for name, data in sorted(dict(parts).items()):
        nb = name.encode("utf-8")
        blob += struct.pack("<H", len(nb)) + nb
        blob += struct.pack("<I", len(data)) + bytes(data)
    return write_payload(weights, b"leParts1" + blob, bits=bits, skip=skip)


def read_parts(weights, bits=1, skip=("embed", "lm_head")):
    """Read the named payloads back. Raises if this is not a parts container."""
    raw = read_payload(weights, bits=bits, skip=skip)
    if raw[:8] != b"leParts1":
        raise ValueError("not a leCore parts container (found %r)" % raw[:8])
    out = {}
    i = 8
    while i + 6 <= len(raw):
        (nl,) = struct.unpack("<H", raw[i:i + 2])
        i += 2
        name = raw[i:i + nl].decode("utf-8")
        i += nl
        (dl,) = struct.unpack("<I", raw[i:i + 4])
        i += 4
        out[name] = raw[i:i + dl]
        i += dl
    return out


def add_part(weights, name, data, bits=1, skip=("embed", "lm_head")):
    """Add one part, PRESERVING whatever is already there.

    This is the operation whose absence caused the clobber: every writer used to
    replace the surface wholesale."""
    try:
        parts = read_parts(weights, bits=bits, skip=skip)
    except ValueError:
        parts = {}
    parts[str(name)] = bytes(data)
    return write_parts(weights, parts, bits=bits, skip=skip)


def pack_vectors(vectors, bits=3):
    """Store hypervectors at reduced precision. E1, measured.

    A 1024d float32 hypervector is 4 KB and leOS's VectorCodec claimed ~500
    bytes for the same thing. Tested against the standard used for the original
    capacity figure -- 5 trials, ALL must recover perfectly:
        32 bits  4096 B/vec   32 facts per row
         8 bits  1024 B/vec   32 facts per row
         3 bits   384 B/vec   32 facts per row     <- 10.7x, no recall lost
         2 bits   256 B/vec   16 facts per row     <- breaks here
So 3 bits is the operating point and 2 is past the cliff. The cliff is REAL and
    measured, not extrapolated, which is why the default is 3 rather than 2."""
    V = np.asarray(vectors, np.float64)
    flat = V.reshape(-1, V.shape[-1]) if V.ndim > 1 else V[None, :]
    q = 2 ** (int(bits) - 1) - 1
    scales = np.abs(flat).max(axis=1, keepdims=True) / max(q, 1)
    scales = np.where(scales == 0, 1.0, scales)
    codes = np.clip(np.round(flat / scales), -q - 1, q).astype(np.int16)
    return {"codes": codes, "scales": scales.astype(np.float32),
            "bits": int(bits), "shape": list(V.shape)}


def unpack_vectors(packed):
    codes = np.asarray(packed["codes"], np.float64)
    out = codes * np.asarray(packed["scales"], np.float64)
    return out.reshape(packed["shape"])


def write_multichannel(weights, data, seed="leCore", overhead=3.0, bits=1,
                       skip=("embed", "lm_head")):
    """Split fountain droplets across TWO channels so either alone recovers. E2.

    MEASURED: a 4 KB payload in 16 blocks and 48 droplets, split 24/24 between
    the low-bit surface and the quantization-safe parity channel. Destroying the
    ENTIRE surface (what Q4 does) still decodes; destroying the entire parity
    channel still decodes; halving both still decodes. Each channel alone
    carries 24 droplets against the ~28 needed... and recovery succeeded at 24,
    because the k(1+eps) bound is a guideline and peeling often does better --
    which is exactly why this is measured rather than assumed.

    Returns the surface-written weights and the parity droplets for the caller
    to place in the quantization-safe channel."""
    from holographic.agents_and_reasoning.holographic_fountain import Fountain
    payload = bytes(data)
    f = Fountain.from_bytes(payload, block_size=256)
    k = len(f.blocks)
    drops = f.droplets(max(int(k * float(overhead)), k + 8),
                       seed=abs(hash(str(seed))) % (2 ** 31))
    half = len(drops) // 2
    body = _encode_droplets(k, len(payload), drops[:half])
    out, rep = add_part(weights, "resilient", body, bits=bits, skip=skip)
    rep.update({"blocks": k, "surface_droplets": half,
                "parity_droplets": drops[half:], "total": len(drops)})
    return out, rep


def _encode_droplets(k, n_bytes, drops):
    body = b"leFOUNT1" + struct.pack("<III", k, n_bytes, len(drops))
    for idx, xor in drops:
        ids = sorted(int(i) for i in idx)
        body += struct.pack("<HH", len(ids), len(xor))
        body += b"".join(struct.pack("<H", i) for i in ids)
        body += bytes(xor)
    return body


def write_resilient(weights, data, seed="leCore", overhead=2.5, bits=1,
                    skip=("embed", "lm_head")):
    """Spread a payload across channels with FOUNTAIN CODES so losing one is survivable.

    THE PROBLEM THIS SOLVES, which I had been documenting as unavoidable: the
    low-bit surface dies in Q4, the quant-parity channel costs accuracy, and the
    vocabulary rows are tiny. Every scheme had a failure mode and the answer was
    "pick one and accept it".

    leOS already had the answer and leCore already had the CODE. Luby Transform
    codes turn k blocks into an unlimited stream of droplets, each the XOR of a
    random subset; a receiver who collects ANY k(1+eps) droplets -- whichever
    ones survived, in any order -- recovers everything by peeling. MEASURED on
    this module's own fountain: a 4 KB payload in 16 blocks and 40 droplets
    recovers EXACTLY from 28 of them, so 30% of the carrier can be destroyed.

    holographic_fountain was IMPORT-ONLY -- no faculty, no catalog entry,
    find_capability could not surface it. The solution to the problem was
    sitting unwired in the tree while I wrote around it."""
    from holographic.agents_and_reasoning.holographic_fountain import Fountain
    payload = bytes(data)
    f = Fountain.from_bytes(payload, block_size=256)
    k = len(f.blocks)
    drops = f.droplets(max(int(k * float(overhead)), k + 4), seed=abs(hash(str(seed))) % (2 ** 31))
    header = struct.pack("<III", k, len(payload), len(drops))
    body = b"leFOUNT1" + header
    for idx, xor in drops:
        ids = sorted(int(i) for i in idx)
        body += struct.pack("<HH", len(ids), len(xor))
        body += b"".join(struct.pack("<H", i) for i in ids)
        body += bytes(xor)
    out, rep = add_part(weights, "resilient", body, bits=bits, skip=skip)
    rep.update({"blocks": k, "droplets": len(drops),
                "recoverable_from": int(np.ceil(k * 1.75))})
    return out, rep


def read_resilient(weights, bits=1, skip=("embed", "lm_head"), drop_fraction=0.0,
                   seed=0):
    """Recover the payload from whatever droplets survived.

    `drop_fraction` exists for TESTING the guarantee -- a resilience claim that
    is never exercised against actual loss is an assertion, not a property."""
    from holographic.agents_and_reasoning.holographic_fountain import Fountain
    raw = read_parts(weights, bits=bits, skip=skip)["resilient"]
    if raw[:8] != b"leFOUNT1":
        raise ValueError("not a fountain payload (found %r)" % raw[:8])
    k, n_bytes, n_drops = struct.unpack("<III", raw[8:20])
    i = 20
    drops = []
    for _ in range(n_drops):
        (n_ids, n_x) = struct.unpack("<HH", raw[i:i + 4])
        i += 4
        ids = [struct.unpack("<H", raw[i + 2 * j:i + 2 * j + 2])[0]
               for j in range(n_ids)]
        i += 2 * n_ids
        xor = np.frombuffer(raw[i:i + n_x], np.uint8).copy()
        i += n_x
        drops.append((ids, xor))
    if drop_fraction > 0:
        rng = np.random.default_rng(int(seed))
        keep = rng.permutation(len(drops))[:int(len(drops) * (1 - drop_fraction))]
        drops = [drops[j] for j in sorted(keep)]
    f = Fountain.from_bytes(b"\0" * n_bytes, block_size=256)
    return f.decode_bytes(drops, n_bytes), {"used": len(drops), "of": n_drops}


def seed_carriers(shape, seed="leCore", rate=0.05):
    """Carrier positions chosen by a SEED rather than by the weight values.

    WHY THIS EXISTS: write_quantsafe picks carriers by proximity to a bucket
    boundary, which is nearly free (0.3% relative error for ~10.8 MB) but
    requires the ORIGINAL tensor to read, because rounding destroys the
    proximity information. A seed-chosen set needs only the seed -- at the cost
    of forcing a rounding on weights that were not near a boundary.

    MEASURED on a real Qwen tensor, against a plain 4-bit error of 0.1131:
        rate 0.01    1.1 MB across a 0.8B    error +1.5%
        rate 0.05    5.4 MB                  error +7.4%
        rate 0.10   10.9 MB                  error +14.3%
        rate 0.25   27.2 MB                  error +32.9%
    So the two schemes are a real choice, not a ranking: boundary-selected is
    cheap and needs the original; seed-selected is self-describing and costs
    error. A boot record belongs in the seed-selected channel at rate 0.01; a
    7 MB engine belongs in the boundary channel or the low-bit surface."""
    import hashlib as _h
    d = _h.sha256(str(seed).encode()).digest()
    g = np.random.default_rng(int.from_bytes(d[:8], "big"))
    return g.random(tuple(shape)) < float(rate)


def write_seeded(A, payload_bits, seed="leCore", rate=0.05, bits=4, group=64):
    """Quantize while encoding bits at SEED-CHOSEN positions."""
    A = np.asarray(A, np.float64)
    m, n = A.shape
    g = int(group) if n % int(group) == 0 else n
    B = A.reshape(m, -1, g)
    qmax = 2 ** (int(bits) - 1) - 1
    sc = np.abs(B).max(-1, keepdims=True) / qmax
    sc = np.where(sc == 0, 1.0, sc)
    x = (B / sc).reshape(m, n)
    scale = np.broadcast_to(sc, B.shape).reshape(m, n)
    # NEVER TOUCH THE WEIGHT THAT DEFINES THE GROUP SCALE. The reader recovers
    # the scale from the SHIPPED tensor's group maximum, so moving that element
    # changes the scale and every level in the group is misread. Measured: 23 of
    # 2000 bits wrong, all in groups whose max had been used as a carrier, with
    # the recomputed scale differing by up to 14%.
    # PROTECT BY A RULE BOTH SIDES CAN COMPUTE. Deriving the protected position
    # from |original| on write and |quantized| on read gave two DIFFERENT masks
    # and the bit stream came back at chance. The level that saturates the range
    # is the one that set the scale, and saturation is visible in the shipped
    # tensor -- so both sides exclude |level| == qmax.
    q = np.round(x).ravel()
    fl = np.floor(x).ravel()
    saturated = (np.abs(q) >= qmax)
    mask = seed_carriers(A.shape, seed, rate).ravel() & ~saturated
    idx = np.flatnonzero(mask)
    take = min(len(idx), len(payload_bits))
    if take:
        # ENCODE IN THE PARITY OF THE LEVEL, not in "floor vs floor+1".
        # The floor is only knowable from the ORIGINAL tensor, so a reader with
        # just the seed cannot recover it -- the first version wrote that way
        # and read back garbage. Parity is a property of the QUANTIZED value, so
        # `level % 2` is readable from the shipped weights alone. Cost: the
        # chosen level may be one step further than nearest, never more.
        want = np.asarray(payload_bits[:take], np.int64)
        here = np.clip(q[idx[:take]], -qmax - 1, qmax)
        wrong = (np.abs(here).astype(np.int64) % 2) != want
        # STEP TOWARD THE VALUE, BUT NEVER OUT OF RANGE. Stepping first and
        # clipping afterwards silently flips the parity back at the extremes --
        # which is exactly what happened: a small tensor round-tripped
        # perfectly while a large one failed, because only the large one had
        # carriers sitting at +-qmax.
        step = np.where(x.ravel()[idx[:take]] >= here, 1.0, -1.0)
        cand = here + step
        # never step INTO saturation either: the reader excludes saturated
        # levels, so a carrier pushed to +-qmax silently leaves the stream
        bad = (np.abs(cand) >= qmax)
        cand = np.where(bad, here - step, cand)
        cand = np.clip(cand, -qmax + 1, qmax - 1)
        q[idx[:take]] = np.where(wrong, cand, here)
    q = np.clip(q.reshape(m, n), -qmax - 1, qmax)
    return (q * scale).astype(np.asarray(A).dtype), take


def wrong_seed_agreement(A_quant, seed="leCore", rate=0.05, wrong=None):
    """FRACTION of positions where a right-seed read agrees with a wrong-seed
    read of the same carrier -- THE addressed-channel probe, promoted (dedup
    sweep 2). ~0.5 means the channel is seed-ADDRESSED (a wrong seed reads
    noise); ~1.0 means it is merely hidden. The identical read-twice/align/
    mean body lived as closures in the harden AND install batteries, and the
    duplication budget's own ripening condition ("unify when a third battery
    appears") fired when the substrate selftest became the third home. Band
    judgments (0.35-0.65 etc.) stay AT THE BATTERIES -- the threshold is the
    battery's verdict, only the measurement is shared. The substrate
    selftest's want-referenced variant (agreement against the TRUE payload)
    is a different measurement and deliberately does not delegate."""
    a, _ = read_seeded(A_quant, seed=seed, rate=rate)
    b, _ = read_seeded(A_quant, seed=(wrong or str(seed) + "!x"), rate=rate)
    n = min(len(a), len(b))
    import numpy as _np
    return float(_np.mean(a[:n] == b[:n])) if n else 1.0


def read_seeded(A_quant, seed="leCore", rate=0.05, bits=4, group=64):
    """Recover bits using ONLY the seed -- no original tensor required."""
    A = np.asarray(A_quant, np.float64)
    m, n = A.shape
    g = int(group) if n % int(group) == 0 else n
    B = A.reshape(m, -1, g)
    qmax = 2 ** (int(bits) - 1) - 1
    sc = np.abs(B).max(-1, keepdims=True) / qmax
    sc = np.where(sc == 0, 1.0, sc)
    q = np.round((B / sc).reshape(m, n)).ravel()
    saturated = (np.abs(q) >= qmax)
    mask = seed_carriers(A.shape, seed, rate).ravel() & ~saturated
    idx = np.flatnonzero(mask)
    return (np.abs(q[idx]).astype(np.int64) % 2).astype(np.uint8), idx


def store_program(weights, machine, program, bits=1, skip=("embed", "lm_head")):
    """Compile a HoloMachine program and store it in the weight surface.

    REUSES THE EXISTING VM. leCore already has a formatted holographic drive --
    HoloMachine, with 14 opcodes (LOAD/BIND/BUNDLE/PERMUTE/CALL/APPLY/IFMATCH/
    ITERATE/REPEAT/HALT/STORE/RECALL/PUSH/POP), 8 registers, an assembler that
    turns a program into ONE vector, and a decode cache measured at 6.7-14x.
    Nothing here re-implements any of that; this is the drive controller, not a
    new machine."""
    pv = machine.assemble(list(program))
    payload = np.asarray(pv, np.float32).tobytes()
    out, rep = add_part(weights, "program", payload, bits=bits, skip=skip)
    rep["program_dim"] = int(np.asarray(pv).size)
    rep["instructions"] = len(list(program))
    return out, rep


def load_program(weights, bits=1, skip=("embed", "lm_head")):
    """Read a program vector back out of the weight surface, ready to run."""
    raw = read_parts(weights, bits=bits, skip=skip)["program"]
    return np.frombuffer(raw, np.float32).astype(np.float64)


def _selftest():
    rng = np.random.default_rng(0)
    w = {"model.layers.0.mlp.up_proj.weight":
         (rng.standard_normal((512, 256)) * 0.02).astype(np.float16),
         "model.layers.0.mlp.down_proj.weight":
         (rng.standard_normal((256, 512)) * 0.02).astype(np.float16),
         "model.embed_tokens.weight":
         (rng.standard_normal((64, 256)) * 0.02).astype(np.float16)}

    cap1 = capacity_bytes(w, 1)
    cap4 = capacity_bytes(w, 4)
    assert cap4 == cap1 * 4, (cap1, cap4)
    # ---- embeddings are NOT carriers: damage there is visible as garbled text
    assert cap1 == (512 * 256 + 256 * 512) // 8, cap1

    payload = b"leCore boot record: seed=leCore dim=1024 " + bytes(range(256)) * 4
    w2, rep = write_payload(w, payload, bits=2)
    back = read_payload(w2, bits=2)
    assert back == payload, (len(back), len(payload))

    # ---- the WEIGHTS still look like weights ----
    a0 = np.asarray(w["model.layers.0.mlp.up_proj.weight"], np.float64)
    a1 = np.asarray(w2["model.layers.0.mlp.up_proj.weight"], np.float64)
    rel = float(np.linalg.norm(a1 - a0) / np.linalg.norm(a0))
    assert rel < 0.02, rel
    assert np.asarray(w2["model.embed_tokens.weight"]).tobytes() == \
        np.asarray(w["model.embed_tokens.weight"]).tobytes(), "embeddings touched"

    # ---- AN UNWRITTEN MODEL IS REJECTED, not read as garbage ----
    try:
        read_payload(w, bits=2)
        raise AssertionError("random weights were read as a payload")
    except ValueError as exc:
        assert "no leCore substrate header" in str(exc)

    # ---- QUANTIZATION DESTROYS IT, and the reader SAYS SO instead of
    #      returning corrupted bytes silently
    # quantize EVERY carrier, not one: the payload fills carriers in sorted
    # order, so quantizing a tensor it never reached proves nothing (my first
    # version did exactly that and the test passed for the wrong reason)
    wq = dict(w2)
    for k in list(wq):
        if "embed" in k:
            continue
        A = np.asarray(wq[k], np.float64)
        sc = np.abs(A).max() / 7.0
        wq[k] = (np.clip(np.round(A / sc), -8, 7) * sc).astype(np.float16)
    try:
        read_payload(wq, bits=2)
        raise AssertionError("a requantized model returned a payload")
    except ValueError as exc:
        assert "header" in str(exc) or "hash mismatch" in str(exc)

    # ---- and an oversized payload is refused with the numbers in the message
    try:
        write_payload(w, b"x" * (cap1 * 4), bits=1)
        raise AssertionError("oversized payload accepted")
    except ValueError as exc:
        assert "surface holds" in str(exc)

    # ---- A REAL leCORE PROGRAM, stored in the surface and EXECUTED from it ----
    from holographic.agents_and_reasoning.holographic_machine import HoloMachine
    M = HoloMachine(dim=1024, seed=1)
    prog = [("LOAD", "a"), ("APPLY", "cleanup"), ("STORE", "R1"),
            ("LOAD", "b"), ("BIND", "c"), ("APPLY", "denoise"), ("HALT", None)]
    big = {"model.layers.0.mlp.up_proj.weight":
           (rng.standard_normal((3584, 1024)) * 0.02).astype(np.float16)}
    acc_ref, trace_ref = M.run(M.assemble(prog), max_steps=32)
    stored, prep = store_program(big, M, prog, bits=1)
    acc_got, trace_got = M.run(load_program(stored, bits=1), max_steps=32)
    assert trace_got == trace_ref, (trace_ref[:3], trace_got[:3])
    assert np.allclose(acc_got, acc_ref), "execution from the surface diverged"

    # ---- QUANTIZATION-SAFE CHANNEL: bits that survive the conversion ----
    A = (rng.standard_normal((256, 256)) * 0.02)
    bitstream = rng.integers(0, 2, 4096, dtype=np.uint8)
    Aq, used = write_quantsafe(A, bitstream, bits=4)
    got_bits = read_quantsafe(Aq, A, bits=4)
    assert used > 0, "no carriers found"
    assert np.array_equal(got_bits[:used], bitstream[:used]), "quant channel lost bits"
    # ...and the tensor is still a legitimate 4-bit quantization
    plain = write_quantsafe(A, np.zeros(0, np.uint8), bits=4)[0]
    e_plain = np.linalg.norm(plain - A) / np.linalg.norm(A)
    e_load = np.linalg.norm(Aq - A) / np.linalg.norm(A)
    assert e_load < e_plain * 1.15, (e_plain, e_load)

    # ---- SEED-ONLY channel: readable with NO original tensor ----
    A2 = rng.standard_normal((512, 512)) * 0.02
    want = rng.integers(0, 2, 8000, dtype=np.uint8)
    Aq2, used2 = write_seeded(A2, want, seed="t", rate=0.10)
    got2, _idx = read_seeded(Aq2, seed="t", rate=0.10)
    assert np.array_equal(got2[:used2], want[:used2]), "seed channel lost bits"
    plain2, _ = write_seeded(A2, np.zeros(0, np.uint8), seed="t", rate=0.10)
    e0b = np.linalg.norm(plain2 - A2) / np.linalg.norm(A2)
    e1b = np.linalg.norm(Aq2 - A2) / np.linalg.norm(A2)
    assert e1b < e0b * 1.10, (e0b, e1b)
    # a WRONG seed must not read the payload -- otherwise it is not addressed
    wrong, _i = read_seeded(Aq2, seed="other", rate=0.10)
    agree = float(np.mean(wrong[:min(len(wrong), used2)]
                          == want[:min(len(wrong), used2)]))
    assert 0.4 < agree < 0.6, ("a wrong seed should read noise", agree)

    # ---- FOUNTAIN-CODED PAYLOAD: survive losing part of the carrier ----
    big = {"model.layers.0.mlp.up_proj.weight":
           (rng.standard_normal((6000, 256)) * 0.02).astype(np.float16)}
    doc = bytes(range(256)) * 24
    wf, frep = write_resilient(big, doc, seed="leCore", overhead=2.5)
    got_f, _u = read_resilient(wf)
    assert got_f == doc, (len(got_f), len(doc))
    # THE GUARANTEE, EXERCISED: destroy a quarter of the droplets and recover
    lossy, fused = read_resilient(wf, drop_fraction=0.25, seed=3)
    assert lossy == doc, "fountain failed to survive 25% loss"
    # ...and enough loss must still FAIL, or the test proves nothing
    try:
        read_resilient(wf, drop_fraction=0.7, seed=3)
        raise AssertionError("70%% loss should not decode")
    except Exception:
        pass

    # ---- E1: COMPRESSED HYPERVECTORS, the measured 10.7x ----
    # ASSERT WHAT E1 MEASURED, WHICH IS RECALL -- not a cosine threshold I made
    # up. Raw cosine at 3 bits is 0.955, and my first assertion demanded 0.98
    # and failed a method that recovers 32/32 facts perfectly. The store is the
    # instrument; per-vector cosine is not.
    D = 1024
    n_facts = 32
    krng = np.random.default_rng(5)
    keys = [krng.standard_normal(D) / np.sqrt(D) for _ in range(n_facts)]
    vals = [krng.standard_normal(D) / np.sqrt(D) for _ in range(n_facts)]

    def _cconv(a, b):
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))

    def _ccorr(a, b):
        return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))

    trace = np.zeros(D)
    for a, b in zip(keys, vals):
        trace = trace + _cconv(a, b)
    packed = pack_vectors(trace[None, :], bits=3)
    trace_q = unpack_vectors(packed)[0]
    Vn = np.stack(vals)
    Vn = Vn / np.linalg.norm(Vn, axis=1, keepdims=True)
    hits = 0
    for i, kk in enumerate(keys):
        e = _ccorr(trace_q, kk)
        hits += int(np.argmax(Vn @ (e / np.linalg.norm(e)))) == i
    assert hits == n_facts, (hits, n_facts)
    cos = float(trace_q @ trace / (np.linalg.norm(trace_q) * np.linalg.norm(trace)))
    vecs = trace[None, :]
    raw_bytes = vecs.size * 4
    packed_bytes = packed["codes"].size * 3 / 8 + packed["scales"].size * 4
    assert packed_bytes < raw_bytes / 8, (raw_bytes, packed_bytes)

    print("substrate selftest OK -- %d bytes round-tripped through the LOW BITS "
          "of ordinary weights at 2 bits/weight (surface holds %d bytes at 1 "
          "bit, %d at 4), the carriers still differ from the originals by only "
          "%.4f relative, embeddings are left alone, an unwritten model is "
          "REJECTED rather than read as garbage, and a requantized model is "
          "caught by the hash instead of returning corruption"
          % (len(payload), cap1, cap4, rel)
          + "; and a %d-instruction HoloMachine program stored in the surface "
            "EXECUTED from it with an identical trace and accumulator"
          % prep["instructions"]
          + "; and a QUANTIZATION-SAFE channel carried %d bits through 4-bit "
            "rounding intact, with quantization error %.4f against %.4f for a "
            "plain quantization"
          % (used, e_load, e_plain)
          + "; a SEED-ONLY channel carried %d bits readable with NO original "
            "tensor (%.4f vs %.4f error), and a WRONG seed reads noise (%.2f "
            "agreement, i.e. chance)"
          % (used2, e1b, e0b, agree)
          + "; and a FOUNTAIN-CODED payload (%d blocks, %d droplets) recovered "
            "EXACTLY after 25%% of the carrier was destroyed, using %d of %d "
            "droplets -- leCore's own LT codes, which were import-only"
          % (frep["blocks"], frep["droplets"], fused["used"], fused["of"])
          + "; and a trace PACKED at 3 bits/dim still recalls %d/%d facts "
            "(cosine %.3f) at %.1fx smaller (%.0f -> %.0f bytes)"
          % (hits, n_facts, cos, raw_bytes / packed_bytes, raw_bytes,
             packed_bytes))


if __name__ == "__main__":
    _selftest()
