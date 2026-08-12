"""holographic_codecatlas.py -- C-1: the codec ATLAS + honest router (machine_map applied to compression).

THE GAP (Rule-0 on record, two sweep rounds, ~10 phrasings): the engine ships ~10 codecs, each
with its own `pays` flag and its own kept negatives -- and NOTHING runs them side by side.
"Which codec should I use for this array" routed to machine_map (a compute map, not a codec
map); "compare compressors on my data" routed to time-travel audit. The atlas closes that.

TWO CALLS, mirroring machine_map / machine_place:

  codec_atlas()          the SPEC SHEET: every codec unit with the real module+symbol, what it
                         applies to, when it pays, and its kept negatives -- so a session reads
                         the family in one place instead of rediscovering it per arc.
  codec_place(x, ...)    the ROUTER: run the cheap gates first, then MEASURE every applicable
                         codec on the caller's actual data, and return a ranked table where
                         'store raw' is a first-class row. A codec that cannot say "store raw"
                         is not honest.

BASELINE DISCIPLINE: every row is priced against zlib(raw bytes) -- the strongest honest
general-purpose baseline in the original space. A ratio quoted against raw float32 flatters
every codec; the atlas refuses to quote it as the headline.

DELEGATION, NOT REIMPLEMENTATION: lossless rows use stdlib zlib/lzma (the same codecs
cold_store trusts); lossy rows delegate to holographic_tucker (tucker/tt/low-rank). The atlas
adds ZERO new codecs. Sequence-predictive (compress_lossless) and set-delta (pack_images)
units are LISTED in the atlas with their preconditions but not auto-run by codec_place --
they need trained predictors / image families the router cannot conjure; the table says so.

KEPT NEGATIVES (inherited loudly, so the router can enforce them):
  * high-entropy data does not compress -- the win there is refusal, and the entropy gate
    prices it BEFORE any expensive factoring runs (Quilez: don't march empty space);
  * energy gates lie on error-sensitive fields -- when the caller states max_error, the
    lossy rows are gated by rank_for_error's budget, never by 99% energy;
  * lossy rows exist ONLY when the caller states a max_error -- the atlas never volunteers
    loss (the sentinel's discipline: noise is never fake-compressed, exactness never
    silently traded).
"""

import lzma
import zlib

import numpy as np

from holographic.caching_and_storage.holographic_tucker import (
    tucker_compress, tucker_reconstruct, tucker_size,
    tt_compress, tt_reconstruct, tt_bytes,
)


# ---------------------------------------------------------------------------
# The spec sheet. Static knowledge: module+symbol, preconditions, negatives.
# WHY a static table: the units' *existence* and *contracts* are facts of the
# codebase; only their performance on the caller's data is measured (codec_place).
# ---------------------------------------------------------------------------
CODEC_UNITS = [
    dict(name="model_weights", kind="lossy", symbol="holographic_unicron (assimilate/transform/filter)",
         applies="trained neural-network weight matrices (safetensors/gguf checkpoints)",
         auto=False,
         pays_when="the layer's spectrum is SPIKE+BULK (a real gap at the MP edge): "
                   "keep outliers, drop the still-random bulk, store thin factors. "
                   "Route via mind.unicron_assimilate; regime detection is built in.",
         negatives="the Qwen3.5-0.8B field result: every knowledge-bearing projection of a "
                   "well-trained LLM read HEAVY-TAILED (no gap) and MP filtering DESTROYED "
                   "the model (256-newline collapse). Heavy-tail layers must pass through; "
                   "their honest size lever is error-bounded residual coding "
                   "(residualcodec: measured 5.22x vs zlib at bf16-class error with the RMT "
                   "readout invariant) -- COLD STORAGE only at ~300s/80k values. Distcodec is "
                   "REFUSED for weights: it ships a distribution; a decoded layer is a fresh "
                   "sample, not the layer."),
    dict(name="raw", kind="lossless", symbol="(identity)",
         applies="anything", auto=True,
         pays_when="never smaller; it is the refusal row every ranking must contain",
         negatives="none -- honesty itself"),
    dict(name="zlib", kind="lossless", symbol="zlib.compress (stdlib; cold_store's fast codec)",
         applies="any bytes / any array's raw bytes", auto=True,
         pays_when="repetition or low byte-entropy exists at byte granularity",
         negatives="high-entropy data (random floats, hypervectors) returns ~1.0x; that is data, not a bug"),
    dict(name="lzma", kind="lossless", symbol="lzma.compress (stdlib; cold_store codec='lzma')",
         applies="any bytes / any array's raw bytes", auto=True,
         pays_when="same as zlib but packs smaller on text/structured data; slower",
         negatives="cost grows fast with size; a hot path should not sit behind lzma"),
    dict(name="lowrank", kind="lossy", symbol="holographic_tucker.tucker_compress (2-D)",
         applies="2-D float arrays, caller-stated max_error", auto=True,
         pays_when="smooth/structured fields; gate is an ERROR budget (rank_for_error), never 99% energy",
         negatives="an SDF passing the energy gate at rank 2 was 7.45% wrong -- energy gates lie; "
                   "white noise gates to near-full rank and must be refused"),
    dict(name="tucker", kind="lossy", symbol="holographic_tucker.tucker_compress (n-D)",
         applies=">=3-D float arrays, caller-stated max_error", auto=True,
         pays_when="structure along SEVERAL axes at once (field over x,y,t; frame stacks; volumes)",
         negatives="never CP (a best rank-R CP approximation may not exist for 3+ modes)"),
    dict(name="tt", kind="lossy", symbol="holographic_tucker.tt_compress",
         applies=">=3-D float arrays, many modes", auto=True,
         pays_when="storage linear in mode count; wins over tucker as modes grow",
         negatives="same refusal as tucker on structureless data"),
    dict(name="residual_codec", kind="lossless/near-lossless", symbol="holographic_residualcodec.residual_encode",
         applies="1-D float signals (lawful/regime-structured)", auto=True,
         pays_when="a piecewise law explains the signal: exact mode pays modestly (float64 mantissa "
                   "ceiling, ~1.0-1.1x), quant mode under a budget pays big (measured 8.5x vs zlib)",
         negatives="model head ~100-400 B/segment loses on short signals (gate refuses); exact-mode "
                   "ratio is capped by irreducible low-mantissa planes"),
    # Listed, not auto-run: preconditions the router cannot conjure from bare data.
    dict(name="rate_distortion", kind="lossy", symbol="mind.rate_distortion_report",
         applies="a SET of vectors where pairwise GEOMETRY is the contract", auto=False,
         pays_when="low-rank vector sets (~3x measured); refuses near-orthogonal sets (0.95x, pays=False)",
         negatives="incompressible unit vectors can code LARGER than float32"),
    dict(name="pack_images", kind="lossless", symbol="mind.pack_images",
         applies="an image FAMILY sharing structure (logo suites, sprite variants)", auto=False,
         pays_when="shared structure across files (measured 2x vs per-file PNG)",
         negatives="LOSES 16x on individually-compressible content; run mind.pack_benchmark, do not guess"),
    dict(name="event_codec", kind="lossless", symbol="mind.record_physics_trace / replay_physics_trace",
         applies="deterministic simulation traces with sparse interruptions", auto=False,
         pays_when="event SPARSITY (663 events replaced 9,600 rows; 13.7x) -- not a codebook",
         negatives="DeltaChain loses on dense mutation; quantized impulse codebooks amplify loss"),
    dict(name="sequence_predictive", kind="lossless", symbol="mind.compress_lossless (tokens)",
         applies="token sequences with a TRAINED predictor (learn_sequence first)", auto=False,
         pays_when="the predictor ranks the truth highly; compression<->prediction duality -- "
                   "its value is MEASURING understanding, not shrinking files",
         negatives="NOT a file codec, measured: ~11 tokens/s vs zlib's ~10^8 bytes/s (7 orders), "
                   "and the varint-coded rank stream reached 5.8 bits/token vs the predictor's "
                   "3.4-bit estimate (rank coding is not an arithmetic coder); zlib beat it on "
                   "the same source text; no predictor, no codec -- raises rather than pretending"),
    dict(name="generator", kind="model", symbol="mind.compressibility_check -> stream sentinel recorder",
         applies="1-D signals; a pass certifies a generator AT THIS HORIZON only", auto=False,
         pays_when="~30 floats replace the window (sentinel's cheapest-faithful-form rung)",
         negatives="extrapolating past the horizon is the caller's declared risk"),
    dict(name="cold_store", kind="tier", symbol="mind.cold_store / mind.cool",
         applies="INACTIVE data (residency policy, not a codec choice)", auto=False,
         pays_when="freeing live RAM / spilling to disk -- even when bytes barely shrink",
         negatives="high-entropy hypervectors barely compress; the win is the freed object"),
]


def codec_atlas():
    """The compression family's spec sheet: every codec unit with its real module+symbol,
    what it applies to, when it pays, and its kept negatives. Static contracts only --
    measure performance on YOUR data with codec_place(x). Mirrors machine_map's shape."""
    return [dict(u) for u in CODEC_UNITS]


# ---------------------------------------------------------------------------
# Cheap gates. WHY first: factoring white noise COSTS more than storing it
# (measured: rank 197/256) -- the gate prices the refusal before the work.
# ---------------------------------------------------------------------------
def byte_entropy(raw):
    """Shannon entropy of the byte histogram, bits/byte in [0, 8]. A cheap ceiling:
    zlib cannot beat ~entropy/8 of the size at byte granularity, so ~7.9+ predicts
    a refusal without running the compressor."""
    if len(raw) == 0:
        return 0.0
    counts = np.bincount(np.frombuffer(raw, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(raw)
    return float(-(p * np.log2(p)).sum())


def _to_bytes(x):
    """Canonical raw bytes for anything the router accepts. Arrays go through
    np.ascontiguousarray so the byte view is deterministic regardless of stride
    history (a transposed view must not hash differently from its copy)."""
    if isinstance(x, (bytes, bytearray)):
        return bytes(x), None
    a = np.ascontiguousarray(np.asarray(x))
    return a.tobytes(), a


def _rank_for_error(s, shape_other, max_abs_err):
    """Smallest rank whose spectral tail bounds the max reconstruction error.
    WHY the Frobenius tail as the bound: ||X - X_r||_max <= ||X - X_r||_F, and the
    Frobenius tail is sqrt(sum of squared dropped singular values) -- conservative,
    never optimistic, which is the direction an error BUDGET must fail in."""
    tail = np.sqrt(np.cumsum((s ** 2)[::-1])[::-1])
    ok = np.where(tail <= max_abs_err)[0]
    return int(ok[0]) if len(ok) else len(s)


def codec_place(x, max_error=None, try_lossy=None):
    """Route data to its honest codec: MEASURE every applicable unit on x and rank by bytes.
    Lossless rows always run (raw / zlib / lzma). Lossy rows (low-rank, tucker, tt) run ONLY
    when the caller states max_error -- the atlas never volunteers loss. Returns
    {rows: [...ranked by bytes...], best: name, raw_bytes, baseline: 'zlib', entropy_bits_per_byte,
    notes} where every row carries {name, bytes, ratio_vs_zlib, ratio_vs_raw, exact, max_abs_error,
    pays}. `pays` means: strictly smaller than the zlib baseline AND (if lossy) inside the budget.
    Refusal is first-class: on incompressible data best='raw' or 'zlib' and that is the finding."""
    raw, arr = _to_bytes(x)
    n_raw = len(raw)
    ent = byte_entropy(raw)
    rows = [dict(name="raw", bytes=n_raw, exact=True, max_abs_error=0.0)]

    # WHY still run zlib above the entropy gate: the gate is a ceiling argument at BYTE
    # granularity; multi-byte structure (float patterns) can still slip under it. zlib is
    # cheap enough to be its own verdict; the gate's job is to skip the EXPENSIVE units.
    z = zlib.compress(raw, 6)
    rows.append(dict(name="zlib", bytes=len(z), exact=True, max_abs_error=0.0))
    l = lzma.compress(raw, preset=1)
    rows.append(dict(name="lzma", bytes=len(l), exact=True, max_abs_error=0.0))
    zlib_bytes = len(z)

    notes = []
    if ent > 7.5:
        notes.append("byte entropy %.2f/8: near-incompressible at byte granularity; "
                     "expensive lossy units gated off unless a max_error budget re-opens them" % ent)

    # 1-D float signals route through the predictive residual codec (C-2): exact mode always
    # (it self-refuses via its own pays gate, so a raw-mode blob is never listed as a row);
    # quant mode only under a stated budget -- loss is never volunteered.
    if arr is not None and np.issubdtype(arr.dtype, np.floating) and arr.ndim == 1 and arr.size >= 128:
        from holographic.sampling_and_signal.holographic_residualcodec import residual_encode
        re_ = residual_encode(arr)
        if re_["report"]["mode"] == "exact":
            rows.append(dict(name="residual_codec", bytes=re_["report"]["bytes"],
                             exact=True, max_abs_error=0.0))
        else:
            notes.append("residual_codec refused (exact mode did not pay): %d B vs %d zlib"
                         % (re_["report"]["bytes"], zlib_bytes))
        if max_error is not None:
            rq = residual_encode(arr, max_error=float(max_error))
            if rq["report"]["mode"] == "quant":
                rows.append(dict(name="residual_codec(quant)", bytes=rq["report"]["bytes"],
                                 exact=False, max_abs_error=rq["report"]["max_abs_error"]))

    lossy_wanted = (max_error is not None) if try_lossy is None else bool(try_lossy)
    if lossy_wanted and arr is not None and np.issubdtype(arr.dtype, np.floating) and arr.ndim >= 2:
        budget = float(max_error) if max_error is not None else None
        if arr.ndim == 2:
            # Low-rank: gate by the ERROR budget, never energy (the SDF lesson).
            s = np.linalg.svd(arr, compute_uv=False)
            r = _rank_for_error(s, arr.shape, budget)
            fac_bytes = r * (arr.shape[0] + arr.shape[1] + 1) * arr.itemsize
            if r < min(arr.shape) and fac_bytes < n_raw:
                U, sv, Vt = np.linalg.svd(arr, full_matrices=False)
                rec = (U[:, :r] * sv[:r]) @ Vt[:r]
                err = float(np.abs(arr - rec).max())
                rows.append(dict(name="lowrank(r=%d)" % r, bytes=fac_bytes,
                                 exact=False, max_abs_error=err))
            else:
                notes.append("lowrank refused: rank %d of %d needed at budget %.3g -- factoring would not pay"
                             % (r, min(arr.shape), budget))
        else:
            for meth, comp, rec_fn, size_fn in (
                    ("tucker", lambda: tucker_compress(arr, energy=0.9999), tucker_reconstruct, tucker_size),
                    ("tt", lambda: tt_compress(arr, tol=budget * 0.5), tt_reconstruct, tt_bytes)):
                try:
                    code = comp()
                    rec = rec_fn(code)
                    err = float(np.abs(arr - rec).max())
                    b = int(size_fn(code)) * arr.itemsize if meth == "tucker" else int(size_fn(code))
                    if err <= budget and b < n_raw:
                        rows.append(dict(name=meth, bytes=b, exact=False, max_abs_error=err))
                    else:
                        notes.append("%s refused: err %.3g vs budget %.3g, %d bytes vs %d raw"
                                     % (meth, err, budget, b, n_raw))
                except Exception as e:  # a unit's failure is a note, never the router's crash
                    notes.append("%s errored: %s" % (meth, e))

    for row in rows:
        row["ratio_vs_raw"] = n_raw / row["bytes"] if row["bytes"] else float("inf")
        row["ratio_vs_zlib"] = zlib_bytes / row["bytes"] if row["bytes"] else float("inf")
        budget_ok = row["exact"] or (max_error is not None and row["max_abs_error"] <= max_error)
        row["pays"] = bool(row["bytes"] < zlib_bytes and budget_ok and row["name"] != "raw")
    rows.sort(key=lambda r: r["bytes"])
    # WHY best excludes budget-violating rows even if smallest: a codec outside the
    # caller's stated error contract has not compressed the caller's data, it has
    # compressed different data.
    valid = [r for r in rows if r["exact"] or (max_error is not None and r["max_abs_error"] <= max_error)]
    best = valid[0]["name"] if valid else "raw"
    return dict(rows=rows, best=best, raw_bytes=n_raw, baseline="zlib",
                entropy_bits_per_byte=ent, notes=notes)


def _selftest():
    rng = np.random.default_rng(0)

    # 1) Smooth 2-D field: lowrank must appear, beat zlib, and respect the budget.
    t = np.arange(96) / 11.0
    X = np.add.outer(np.sin(t), np.cos(t)) + 0.5 * np.outer(np.cos(t / 3), np.sin(t / 2))
    r = codec_place(X, max_error=1e-6)
    lr = [row for row in r["rows"] if row["name"].startswith("lowrank")]
    assert lr and lr[0]["pays"], "lowrank must pay on a rank-2-ish field: %s" % r["rows"]
    assert lr[0]["max_abs_error"] <= 1e-6, "budget violated: %g" % lr[0]["max_abs_error"]
    assert r["best"].startswith("lowrank"), r["best"]

    # 2) White noise: REFUSAL is the finding. No lossy row may pay; best is raw or zlib-ish.
    N = rng.standard_normal((64, 64))
    rn = codec_place(N, max_error=0.01)
    assert not any(row["pays"] and not row["exact"] for row in rn["rows"]), \
        "a lossy unit claimed to pay on white noise: %s" % rn["rows"]
    assert rn["entropy_bits_per_byte"] > 7.0, rn["entropy_bits_per_byte"]

    # 3) Repetitive bytes: zlib pays, and raw never claims pays.
    rb = codec_place(b"abcabcabc" * 500)
    zrow = [row for row in rb["rows"] if row["name"] == "lzma"][0]
    assert zrow["ratio_vs_raw"] > 5, zrow
    assert all(not row["pays"] for row in rb["rows"] if row["name"] == "raw")

    # 4) No max_error => NO lossy rows, ever (loss is never volunteered).
    rq = codec_place(X)
    assert all(row["exact"] for row in rq["rows"]), rq["rows"]

    # 5) 3-D structured stack: tucker or tt must pay inside the budget.
    V = np.stack([X * (1 + 0.01 * k) for k in range(24)])
    rv = codec_place(V, max_error=1e-4)
    assert any(row["name"] in ("tucker", "tt") and row["pays"] for row in rv["rows"]), rv["rows"]

    # 5b) 1-D lawful signal: the residual codec must appear and its quant row must win big.
    t2 = np.arange(1200.)
    sig = np.concatenate([np.sin(2 * np.pi * t2[:400] / 23), 0.002 * t2[400:800] - 0.3,
                          0.5 * np.cos(2 * np.pi * t2[:400] / 41)])
    r1 = codec_place(sig + 0.01 * rng.standard_normal(1200), max_error=1e-3)
    qrow = [row for row in r1["rows"] if row["name"] == "residual_codec(quant)"]
    assert qrow and qrow[0]["pays"] and r1["best"] == "residual_codec(quant)", r1["rows"]

    # 6) Atlas lists every declared unit and each carries its negatives.
    atlas = codec_atlas()
    assert len(atlas) == len(CODEC_UNITS) and all(u["negatives"] for u in atlas)

    # 7) Determinism: identical input, identical byte counts.
    assert codec_place(X, max_error=1e-6)["rows"] == r["rows"]

    print("holographic_codecatlas selftest OK (%d atlas units)" % len(atlas))


if __name__ == "__main__":
    _selftest()
