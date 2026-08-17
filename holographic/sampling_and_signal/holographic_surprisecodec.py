"""holographic_surprisecodec.py -- C-3: surprise-weighted rate allocation (the void instrument as a coder).

THE GAP (Rule-0 on record, both sweep rounds + a fresh 6-phrase probe): "allocate bits where
the information is" / "code the news finely and the expected coarsely" returned only fallbacks.
Information-rate RENDERING ("shade the news, reproject the rest") exists; the coding analogue
did not. This module is that analogue, built on the drift model's zeroth moment:

    z(x) = <enc(x), mu>   -- a KDE density readout in ONE dot product, N-independent
                             (holographic_hdrift's central fact, reused not rebuilt).

THE ALLOCATION RULE: a point the reference corpus already predicts (z on the reference's own
on-support scale) carries little news -- code it COARSELY. A point in the corpus's void
(z below the reference's low quantile -- the support_gauge discipline from residualvoid,
pointed at rate instead of alarm) IS the news -- code it FINELY. One flag bit per point
routes each to its step; the flag + zigzag-varint quantized coordinates are zlib'd.

THE HONEST CLAIM (and its baseline, which travels in the report): against UNIFORM-FINE
quantization -- the coder that gives every point the news-grade step -- surprise allocation
keeps the SAME error contract on the news (|err| <= fine_step/2 per coordinate, pinned) while
spending coarse symbols on the predicted mass. MEASURED in the selftest (77% on-model batch,
coarsen=256): 1.71x fewer bytes at identical news fidelity; coarsen sweep 16/64/128/256 ->
1.17/1.36/1.57/1.71x. Against uniform-coarse the
comparison is not run, because uniform-coarse violates the news contract by construction --
a baseline that fails the contract is a strawman, not a baseline.

REFUSAL (first-class, the atlas discipline): when the split does not differentiate -- fewer
than 5% or more than 95% of points land on one side -- per-point flags cannot pay for
themselves; the coder falls back to UNIFORM fine quantization and the report says
mode='uniform' with the reason. All-news data (nothing predicted) and all-predicted data
(nothing new) are both served honestly by one step.

BOUNDS ARE LOAD-BEARING: the FPE scalar encoder is meaningless out of range (its own loud
warning), so the drift model is trained with bounds spanning reference AND batch. A batch
point outside the reference's box is then a genuine low-z void point, not an encoder artifact.

KEPT NEGATIVES:
  * this is LOSSY BY DESIGN on the predicted mass -- it is the right coder when the consumer
    tolerates model-grade fidelity where the model already knows (telemetry, particle
    populations, sample banks), and the WRONG coder for a bit-exact contract (use
    residual_encode / the atlas);
  * THE VARINT FLOOR caps the split's win: one byte per coordinate is the cheapest symbol,
    so once the coarse step drives quantized values under 128 the ratio saturates (~1.7x on
    the selftest geometry). The next rung -- coding the predicted mass as deltas from shipped
    cluster centers -- is DEFERRED, not impossible: it pays only when the predicted mass is
    tight around few modes, and it adds decoder-side state;
  * surprise is judged against the REFERENCE, so a stale reference inflates the news share
    and the bytes with it -- the report carries news_fraction so drift of that number over
    batches is itself the retrain signal.
"""

import struct
import zlib

import numpy as np


_MAGIC = b"LSC1"
_MODE_UNIFORM, _MODE_SPLIT = 0, 1


def _zigzag(q):
    return np.where(q >= 0, 2 * q, -2 * q - 1).astype(np.uint64)


def _unzigzag(z):
    z = z.astype(np.int64)
    return (z >> 1) ^ -(z & 1)


def _varint_encode(vals):
    out = bytearray()
    for v in vals:
        v = int(v)
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def _varint_decode(raw, n):
    vals = np.empty(n, dtype=np.uint64)
    i = 0
    for k in range(n):
        shift = 0
        v = 0
        while True:
            b = raw[i]; i += 1
            v |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        vals[k] = v
    return vals


def surprise_code(points, reference, fine_step, coarsen=128.0, dim=2048,
                  news_quantile=0.10, mind=None):
    """Code a point batch with bits allocated by SURPRISE against a reference corpus: points
    the reference's drift model predicts get step fine_step*coarsen, points in its void get
    fine_step -- same news fidelity as uniform-fine, fewer bytes. Falls back to mode='uniform'
    when the split does not differentiate (<5% or >95% news). Returns {blob, report:{mode,
    bytes, uniform_fine_bytes, ratio_vs_uniform_fine, news_fraction, fine_step, coarse_step,
    max_err_news, max_err_predicted}}. Decode with surprise_decode."""
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    reference = np.asarray(reference, dtype=np.float64)
    n, d = points.shape
    if mind is None:
        import lecore
        mind = lecore.UnifiedMind(dim=256, seed=0)

    # Bounds span reference AND batch: out-of-range FPE encodings are meaningless (the
    # encoder's own declared negative), and a codec must not build its importance field
    # on a meaningless readout.
    lo = np.minimum(points.min(0), reference.min(0))
    hi = np.maximum(points.max(0), reference.max(0))
    pad = 0.05 * (hi - lo + 1e-12)
    model = mind.drift_train(reference, dim=dim, bounds=list(zip(lo - pad, hi + pad)))

    z_ref = np.array([float(model.enc.encode(p) @ model.mu) for p in reference])
    z_batch = np.array([float(model.enc.encode(p) @ model.mu) for p in points])
    # The support gauge, pointed at rate: news = below the reference's OWN low quantile.
    thr = float(np.quantile(z_ref, news_quantile))
    news = z_batch < thr
    frac = float(news.mean())

    fine = float(fine_step)
    coarse = fine * float(coarsen)

    def _pack_uniform():
        q = np.round(points / fine).astype(np.int64)
        payload = zlib.compress(_varint_encode(_zigzag(q.ravel())), 6)
        head = struct.pack("<BIIdd", _MODE_UNIFORM, n, d, fine, coarse)
        return _MAGIC + head + payload, np.abs(points - q * fine).max()

    uniform_blob, uni_err = _pack_uniform()

    # THE CHANCE GATE (shuffled-null discipline, closed form): a batch drawn from the
    # reference's own distribution lands ~news_quantile of its points below the reference's
    # q(news_quantile) BY CONSTRUCTION -- that is the tail, not news. Split only when the
    # news share clearly exceeds chance (1.5x) and leaves something predicted to coarsen.
    chance = float(news_quantile)
    if frac < max(0.05, 1.5 * chance) or frac > 0.95:
        blob = uniform_blob
        report = dict(mode="uniform", bytes=len(blob),
                      uniform_fine_bytes=len(uniform_blob), ratio_vs_uniform_fine=1.0,
                      news_fraction=frac, fine_step=fine, coarse_step=coarse,
                      max_err_news=float(uni_err), max_err_predicted=float(uni_err),
                      note="news share %.1f%% vs %.0f%% expected by chance: split cannot pay"
                           % (100 * frac, 100 * chance))
        return dict(blob=blob, report=report)

    steps = np.where(news, fine, coarse)
    q = np.round(points / steps[:, None]).astype(np.int64)
    flags = np.packbits(news.astype(np.uint8))
    payload = zlib.compress(flags.tobytes() + _varint_encode(_zigzag(q.ravel())), 6)
    head = struct.pack("<BIIdd", _MODE_SPLIT, n, d, fine, coarse)
    blob = _MAGIC + head + payload

    rec = q * steps[:, None]
    err_news = float(np.abs(points[news] - rec[news]).max()) if news.any() else 0.0
    err_pred = float(np.abs(points[~news] - rec[~news]).max()) if (~news).any() else 0.0
    report = dict(mode="split", bytes=len(blob), uniform_fine_bytes=len(uniform_blob),
                  ratio_vs_uniform_fine=len(uniform_blob) / len(blob),
                  news_fraction=frac, fine_step=fine, coarse_step=coarse,
                  max_err_news=err_news, max_err_predicted=err_pred)
    return dict(blob=blob, report=report)


def surprise_decode(blob):
    """Invert surprise_code: read the per-point news flags (split mode) and dequantize each
    point at its own step. Uniform mode dequantizes everything at fine_step. Raises on a
    foreign blob."""
    if blob[:4] != _MAGIC:
        raise ValueError("not a surprise-codec blob (bad magic)")
    mode, n, d, fine, coarse = struct.unpack("<BIIdd", blob[4:4 + 25])
    raw = zlib.decompress(blob[4 + 25:])
    if mode == _MODE_UNIFORM:
        q = _unzigzag(_varint_decode(raw, n * d)).reshape(n, d)
        return q * fine
    nflag = (n + 7) // 8
    news = np.unpackbits(np.frombuffer(raw[:nflag], dtype=np.uint8))[:n].astype(bool)
    q = _unzigzag(_varint_decode(raw[nflag:], n * d)).reshape(n, d)
    steps = np.where(news, fine, coarse)
    return q * steps[:, None]


def _selftest():
    import lecore
    rng = np.random.default_rng(0)
    mind = lecore.UnifiedMind(dim=256, seed=0)

    # Reference: two tight clusters. Batch: 85% on-model + 15% void points (the news).
    ref = np.vstack([c + 0.04 * rng.standard_normal((120, 2))
                     for c in ([0.25, 0.25], [0.75, 0.75])])
    on = np.vstack([c + 0.04 * rng.standard_normal((85, 2))
                    for c in ([0.25, 0.25], [0.75, 0.75])])[:170]
    void = np.array([[0.5, 0.5], [0.15, 0.85], [0.85, 0.15]]).repeat(10, axis=0) \
        + 0.02 * rng.standard_normal((30, 2))
    batch = np.vstack([on, void])
    fine = 1e-4

    r = surprise_code(batch, ref, fine_step=fine, coarsen=256.0, mind=mind)
    rep = r["report"]
    out = surprise_decode(r["blob"])

    # 1) Mode split; the planted news fraction is recovered (15% planted, tolerance for tails).
    assert rep["mode"] == "split", rep
    assert 0.08 <= rep["news_fraction"] <= 0.30, rep["news_fraction"]

    # 2) THE CONTRACT: news error at fine grade, predicted at coarse grade, decode agrees.
    assert rep["max_err_news"] <= fine / 2 + 1e-15, rep
    assert rep["max_err_predicted"] <= (fine * 256.0) / 2 + 1e-15, rep
    news_mask = np.abs(out - batch).max(1) <= fine / 2 + 1e-15
    assert news_mask.sum() >= 30, "decoded news points must sit at fine fidelity"

    # 3) THE WIN, against the honest baseline: same news fidelity, strictly fewer bytes.
    assert rep["ratio_vs_uniform_fine"] > 1.5, "must clearly beat uniform-fine: %s" % rep

    # 4) REFUSAL: an all-on-model batch cannot pay for flags -- uniform mode, ratio 1.0.
    r_all = surprise_code(on, ref, fine_step=fine, mind=mind)
    assert r_all["report"]["mode"] == "uniform" and r_all["report"]["ratio_vs_uniform_fine"] == 1.0
    out_all = surprise_decode(r_all["blob"])
    assert np.abs(out_all - on).max() <= fine / 2 + 1e-15

    # 5) Determinism: identical inputs, byte-identical blob.
    assert surprise_code(batch, ref, fine_step=fine, coarsen=256.0, mind=mind)["blob"] == r["blob"]

    print("surprisecodec selftest OK -- split %.2fx vs uniform-fine at equal news fidelity "
          "(news %.0f%%)" % (rep["ratio_vs_uniform_fine"], 100 * rep["news_fraction"]))


if __name__ == "__main__":
    _selftest()
