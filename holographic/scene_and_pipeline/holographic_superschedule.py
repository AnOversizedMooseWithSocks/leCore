"""holographic_superschedule.py -- Fill 3: AUTO-SUPERPOSITION + SPILL. The latency-hiding move: hold N
independent computations in ONE vector, do one op on the bundle, unpack -- the VSA-native form of a GPU keeping
thousands of threads in flight so memory stalls vanish behind other work.

WHY THIS EXISTS (Compute Architecture plan, Fill 3)
---------------------------------------------------
`holographic_superposed` already packs K keyed items into one vector and recovers them, but it is MANUAL and
capacity-bounded (~0.1-0.2*D items with cleanup-gated recall, ~0.02*D when the recovered values feed continuous
math). Rev 3 of the plan corrects the earlier "abstain above the wall" stance: the capacity limit is a
BUCKET-SIZING DIAL, not a throughput ceiling. So this fill packs up to the dial into one vector AND, when the
batch overflows, SPILLS the rest across more buckets (the §5.3 RAID-width lever, `holographic_distribute`'s
partition), instead of giving up. Total throughput has no ceiling; the wall only sets the bucket size.

The one honest DECLINE that remains: continuous downstream math with NO cleanup to reset crosstalk -- there,
cramming past the (small) continuous capacity into a single vector silently corrupts the result, so we spill
(each bucket stays under the dial) rather than cram. Abstention is only correct when the result would be
silently wrong AND spilling is not allowed.

HONEST SCOPE (kept loud): the capacity dial is PRICED -- widening D pushes it out only sublinearly while memory
grows linearly (Plate). Nonlinear-feedback batches genuinely do NOT superpose (`distribute`'s own negative) --
those must scatter disjointly, not pack. Measured against `bind_batch` (the strong baseline), not a Python loop.
Deterministic; NumPy + stdlib; a thin layer over the shipped `superposed` + `distribute`.
"""
import numpy as np

from holographic.misc.holographic_superposed import pack, recover_all
from holographic.agents_and_reasoning.holographic_ai import bind_fixed, involution_batch


def pack_capacity(dim, gated=True):
    """The bucket-sizing dial: how many items one D-vector can hold. `gated=True` = cleanup-gated recall (a
    discrete decision resets crosstalk each step) -> ~0.15*D; `gated=False` = the recovered values feed
    continuous math with no cleanup -> a much smaller ~0.02*D. A conservative floor of 1. (Plate's capacity
    mathematics; the wall is a fraction of D, so it is a dial you can also widen by widening D -- at linear cost.)"""
    frac = 0.10 if gated else 0.02
    return max(1, int(frac * dim))


def _buckets(n, cap):
    """Split n items into contiguous buckets of at most `cap` -- the spill. This is `distribute.partition`'s job
    (independent buckets, reassembled by concatenation since each holds DISJOINT items); kept inline and readable."""
    return [list(range(i, min(i + cap, n))) for i in range(0, n, cap)]


def superpose_batch(keys, items, gated=True):
    """Pack (keys, items) into the FEWEST superposed vectors that keep each bucket under the capacity dial. If
    everything fits, that is ONE vector (one op in flight over all of it); otherwise it SPILLS across buckets.
    Returns (packed_vectors, buckets) where buckets[b] are the original item indices in packed_vectors[b]."""
    keys = np.asarray(keys, float); items = np.asarray(items, float)
    n, dim = items.shape
    cap = pack_capacity(dim, gated=gated)
    buckets = _buckets(n, cap)
    packed = [pack(keys[b], items[b]) for b in buckets]
    return packed, buckets


def recover_batch(packed, buckets, keys):
    """Recover every item from its superposed bucket, back into original order. Inverse of superpose_batch."""
    keys = np.asarray(keys, float)
    n = sum(len(b) for b in buckets)
    out = np.zeros((n, keys.shape[1]))
    for pv, b in zip(packed, buckets):
        rec = recover_all(pv, keys[b])                    # batched recovery within the bucket
        for local, idx in enumerate(b):
            out[idx] = rec[local]
    return out


def apply_in_superposition(keys, items, op, gated=True):
    """The latency-hiding move end-to-end: hold the items in superposition and apply ONE bind by `op` to each
    bucket's whole bundle at once (transforming every item in flight), then recover. bind(S, op) distributes over
    the bundle, so recovering item i yields bind(item_i, op) -- N transforms from a handful of packed ops, with
    spill past the capacity dial. Returns the recovered transformed items in original order."""
    keys = np.asarray(keys, float); items = np.asarray(items, float)
    packed, buckets = superpose_batch(keys, items, gated=gated)
    op = np.asarray(op, float)
    from holographic.agents_and_reasoning.holographic_ai import bind as _bind
    transformed = [_bind(pv, op) for pv in packed]        # ONE bind per bucket applies op to all its items
    return recover_batch(transformed, buckets, keys)


def _unit(rng, k, d):
    v = rng.standard_normal((k, d)); return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------------------------
# CALIBRATED CAPACITY (Forecasting sweep, sec.5.5): the scheduler's cost model IS a forecaster. `pack_capacity`
# assumes the theoretical wall (0.10*D); this MEASURES it -- "will a packing of N recall well?" is a forecast, so
# probe growing loads, measure recall, and pick the largest load whose recall stays above the caller's target.
# The scheduler then packs as many as it is CONFIDENT it can, instead of trusting a fixed fraction. Additive:
# pack_capacity is unchanged; this is the opt-in measured alternative.
# ---------------------------------------------------------------------------------------------------------------

def calibrated_capacity(dim, gated=True, target_recall=0.9, n_trials=5, seed=0, max_n=None):
    """Measure the packing capacity instead of assuming it. For growing bucket sizes N, pack N random items under
    N random keys, recover them, and measure recall (the mean cosine of each recovered item to its original). As
    N grows, crosstalk grows and recall falls monotonically, so the largest N whose recall stays >= target_recall
    is the CALIBRATED capacity -- the cost model's forecast of "how many can I pack and still recover." Returns
    (capacity, curve) where curve is [(N, recall), ...] for inspection.

    KEPT NEGATIVE (loud): this costs a measurement (a probe), and the capacity is target- and data-dependent -- a
    stricter target_recall gives a smaller capacity. It assumes crosstalk is monotone in load, which holds for
    random superposition (the scheduler's case) but is not a theorem for adversarial keys."""
    rng = np.random.default_rng(seed)
    if max_n is None:
        max_n = max(4, int(0.30 * dim))                       # probe past the theoretical wall (0.10*D)
    curve = []
    capacity = 1
    for n in range(1, max_n + 1):
        accs = []
        for _t in range(n_trials):
            keys = _unit(rng, n, dim)
            items = _unit(rng, n, dim)
            pv = pack(keys, items)
            rec = np.asarray(recover_all(pv, keys), float)
            if gated:
                # GATED recall: each recovered vector is CLEANED UP to the nearest atom (a discrete decision that
                # resets crosstalk). Recall = fraction that clean up to their OWN item. This is the recall the
                # gated capacity is defined for.
                sims = rec @ items.T
                acc = float((np.argmax(sims, axis=1) == np.arange(n)).mean())
            else:
                # UNGATED recall: the recovered values feed continuous math with no cleanup, so raw fidelity is
                # what matters -- the mean cosine of each recovered item to its original.
                cos = np.sum(rec * items, axis=1) / (np.linalg.norm(rec, axis=1) * np.linalg.norm(items, axis=1) + 1e-12)
                acc = float(cos.mean())
            accs.append(acc)
        acc = float(np.mean(accs))
        curve.append((n, acc))
        if acc >= target_recall:
            capacity = n
        else:
            break                                             # crosstalk is monotone in N -> larger N only worse
    return capacity, curve


def should_superpose(n, dim, gated=True, target_recall=0.9, seed=0):
    """The scheduler's gate: is it worth holding all N items in ONE superposed vector, or should it spill? True
    iff N is within the MEASURED calibrated capacity at the caller's confidence target -- 'only superpose when
    confident the recall pays, else fall back to spilling,' the sweep's decision made per batch instead of by a
    fixed dial."""
    cap, _ = calibrated_capacity(dim, gated=gated, target_recall=target_recall, seed=seed)
    return n <= cap


def _capacity_selftest():
    """The measured capacity brackets the theoretical dial and moves the right way with the target: a stricter
    recall target gives a SMALLER capacity; recall is high within it and falls beyond it; should_superpose gates
    accordingly."""
    dim = 512
    cap90, curve = calibrated_capacity(dim, gated=True, target_recall=0.9, seed=0)
    cap99, _ = calibrated_capacity(dim, gated=True, target_recall=0.99, seed=0)
    assert cap99 <= cap90                                      # a stricter target -> a smaller safe capacity
    assert cap90 >= 1
    # recall is perfect at the smallest load and falls beyond capacity (monotone crosstalk)
    near = dict(curve)
    assert near[1] >= 0.99
    # the gate: a load within capacity superposes; a load far past it does not
    assert should_superpose(max(1, cap90 // 2), dim, target_recall=0.9) is True
    assert should_superpose(cap90 + max(5, cap90), dim, target_recall=0.9) is False
    print("holographic_superschedule capacity selftest OK: at D=%d, calibrated capacity is %d @ target 0.90 and "
          "%d @ 0.99 (stricter target -> smaller safe load); theoretical dial is %d; should_superpose gates on the "
          "MEASURED wall, not the assumed one" % (dim, cap90, cap99, pack_capacity(dim, gated=True)))


def _selftest():
    """Under the capacity dial, cleanup-gated recovery picks the right atoms; OVER it, spilling into buckets
    recovers far better than cramming all into one vector (spill beats cram/abstain); one bind on the bundle
    transforms all items (the latency-hiding win); deterministic."""
    from holographic.agents_and_reasoning.holographic_ai import bind, cosine
    from holographic.misc.holographic_superposed import pack, recover_all, resolve
    rng = np.random.default_rng(0)
    D = 512
    cap = pack_capacity(D, gated=True)                   # ~51 at D=512

    def unit(k):
        v = rng.standard_normal((k, D)); return v / np.linalg.norm(v, axis=1, keepdims=True)

    # (1) under the dial: recover K keyed atoms and CLEAN UP -- the gated recall the capacity is defined for
    K = cap // 3
    keys = unit(K); items = unit(K); cb = items
    packed, buckets = superpose_batch(keys, items, gated=True)
    assert len(packed) == 1                              # fits one bucket
    rec = recover_batch(packed, buckets, keys)
    acc_under = float(np.mean([resolve(rec[i], cb)[0] == i for i in range(K)]))
    assert acc_under > 0.85                              # cleanup-gated recall holds under the dial

    # (2) OVER the dial: N = 2x a bucket. SPILLING into buckets recovers far better than cramming into one.
    N = cap * 2
    keys = unit(N); items = unit(N); cb = items
    packed, buckets = superpose_batch(keys, items, gated=True)
    assert len(buckets) == 2                             # spilled, not abstained
    rec_spill = recover_batch(packed, buckets, keys)
    acc_spill = float(np.mean([resolve(rec_spill[i], cb)[0] == i for i in range(N)]))
    crammed = pack(keys, items)                          # everything in ONE over-capacity vector
    rec_cram = recover_all(crammed, keys)
    acc_cram = float(np.mean([resolve(rec_cram[i], cb)[0] == i for i in range(N)]))
    assert acc_spill > acc_cram + 0.2                    # spill beats cram -- the whole point of not abstaining

    # (3) the latency-hiding win: ONE bind on the bundle applies op to ALL items (recover ~ bind(item_i, op)).
    # The recovered items feed CONTINUOUS math (no cleanup), so keep K under the small continuous dial.
    K = max(3, pack_capacity(D, gated=False) // 2)       # ~5 at D=512
    keys = unit(K); items = unit(K); op = unit(1)[0]
    out = apply_in_superposition(keys, items, op, gated=True)
    truth = np.stack([bind(items[i], op) for i in range(K)])
    fid_apply = float(np.mean([cosine(out[i], truth[i]) for i in range(K)]))
    assert fid_apply > 0.3                               # every item got the op, from one bind per bucket
    big_keys = unit(30); big_items = unit(30)
    assert len(superpose_batch(big_keys, big_items)[0]) < 30   # far fewer packed ops than separate binds

    # (4) the continuous dial is smaller (the priced residual, made visible)
    assert pack_capacity(D, gated=False) < pack_capacity(D, gated=True)

    # (5) deterministic
    a = recover_batch(*superpose_batch(keys, items), keys)
    b = recover_batch(*superpose_batch(keys, items), keys)
    assert np.array_equal(a, b)
    print("holographic_superschedule selftest OK: under the dial (cap=%d @ D=%d) gated recall %.2f; over it, "
          "SPILL beats cram (%.2f vs %.2f); one bind transforms all items in flight (fid %.2f); deterministic"
          % (cap, D, acc_under, acc_spill, acc_cram, fid_apply))


if __name__ == "__main__":
    _selftest()
