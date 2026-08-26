#!/usr/bin/env python3
"""The benchmark pack -- Cranmer's gate as repo infrastructure (panel item 7).

Three pieces, each honest about what it measures:

1. MQAR WITH DISTRACTORS (the full Zoology-shaped task, not just the memory half):
   token sequences interleave (key, value) adjacent pairs with distractor tokens
   drawn from a disjoint range; queries arrive after. The ENCODER is a one-pass
   scanner that stores value-range tokens immediately following key-range tokens --
   the same vocabulary-range knowledge every published baseline's tokenisation
   embeds. This closes the recorded scope gap on the earlier MQAR claim ("memory
   half only; no distractor parsing").

2. UEA-MTSCA LOADER (.ts parser, stdlib+numpy): real multivariate archives fetched
   from sktime's GitHub tree (allowed-domain constraint respected). Feeds
   easy_model's classifier as (T, d) sequences with string labels.

3. CI HARNESS: run any benchmark fn across seeds -> mean, sd, and a bootstrap 95%
   interval, emitted as a markdown row. Claims without intervals do not leave this
   file.

Not a test suite: nothing here gates CI. It is the instrument that turns measured
results into claimable ones.
"""
import os
import sys
import numpy as np

# the pack runs from tools/ or the repo root -- resolve the repo root either way.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)                       # data/uea paths are repo-relative


# ---------------------------------------------------------------- 1: MQAR+distractors
def mqar_distractor_task(n_pairs, vocab, seq_noise=3.0, seed=0):
    """Build one Zoology-shaped episode: token stream with distractors, then queries.
    Key tokens in [0, vocab), value tokens in [vocab, 2*vocab), distractors in
    [2*vocab, 3*vocab) -- disjoint ranges, as in the published task's tokenisation."""
    rng = np.random.default_rng(seed)
    keys = rng.choice(vocab, n_pairs, replace=False)
    vals = rng.integers(0, vocab, n_pairs)
    stream = []
    for k, v in zip(keys, vals):
        for _ in range(rng.poisson(seq_noise)):
            stream.append(int(2 * vocab + rng.integers(0, vocab)))   # distractor
        stream.append(int(k))
        stream.append(int(vocab + v))
    for _ in range(rng.poisson(seq_noise)):
        stream.append(int(2 * vocab + rng.integers(0, vocab)))
    return np.array(stream), keys, vals


def mqar_encode_and_recall(stream, queries, vocab, dim=None, seed=0):
    """One-pass scan: a value-range token immediately after a key-range token is a
    pair; everything else is ignored. Store superposed, recall the queries."""
    from holographic.caching_and_storage.holographic_supermemory import (
        SuperposedMemory, allocate)
    pairs = [(int(a), int(b) - vocab) for a, b in zip(stream[:-1], stream[1:])
             if a < vocab and vocab <= b < 2 * vocab]
    if not pairs:
        return np.full(len(queries), -1)
    ks = np.array([p[0] for p in pairs])
    vs = np.array([p[1] for p in pairs])
    D = dim or allocate(len(pairs), vocab)
    mem = SuperposedMemory(D, vocab, seed=seed)
    mem.store(ks, vs)
    return mem.recall(np.asarray(queries, dtype=int), decoder="pic")["values"]


def bench_mqar_distractors(n_pairs=64, vocab=512, seed=0):
    stream, keys, vals = mqar_distractor_task(n_pairs, vocab, seed=seed)
    got = mqar_encode_and_recall(stream, keys, vocab, seed=seed)
    return float(np.mean(got == vals))


# --------------------------------------------------------------------- 2: UEA loader
def load_ts(path):
    """Minimal .ts parser (stdlib+numpy): dimensions ':'-separated, values
    ','-separated, trailing ':label'. Returns (list of (T, d) arrays, labels)."""
    seqs, labels = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line.startswith(("@", "#")):
            continue
        parts = line.split(":")
        labels.append(parts[-1].strip())
        dims = [np.array([float(c) for c in p.split(",")]) for p in parts[:-1]]
        seqs.append(np.stack(dims, axis=1))
    return seqs, labels


def bench_uea(name="BasicMotions", root="data/uea", seed=0):
    """Train easy_model on the archive's own train split, score the test split."""
    import lecore
    tr_x, tr_y = load_ts("%s/%s_TRAIN.ts" % (root, name))
    te_x, te_y = load_ts("%s/%s_TEST.ts" % (root, name))
    m = lecore.UnifiedMind(dim=512, seed=seed)
    model = m.easy_model(tr_x, labels=tr_y, seed=seed)
    pred = model.ask(te_x)["answer"]
    return float(np.mean([p == t for p, t in zip(pred, te_y)]))


# --------------------------------------------------------------------- 3: CI harness
def ci_run(fn, seeds=range(5), n_boot=2000):
    """mean, sd, bootstrap 95% CI across seeds -- the interval every claim needs."""
    xs = np.array([fn(seed=s) for s in seeds], dtype=float)
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(xs, len(xs)).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(xs.mean()), "sd": float(xs.std()),
            "ci95": (float(lo), float(hi)), "runs": [float(x) for x in xs]}


def row(name, r):
    return "| %-32s | %.3f | %.3f | [%.3f, %.3f] | %d seeds |" % (
        name, r["mean"], r["sd"], r["ci95"][0], r["ci95"][1], len(r["runs"]))


def main():
    print("| benchmark | mean | sd | 95% CI | n |")
    print("|---|---|---|---|---|")
    print(row("MQAR+distractors kv=64 V=512",
              ci_run(lambda seed: bench_mqar_distractors(64, 512, seed=seed))))
    print(row("MQAR+distractors kv=128 V=1024",
              ci_run(lambda seed: bench_mqar_distractors(128, 1024, seed=seed),
                     seeds=range(3))))
    try:
        print(row("UEA BasicMotions (test acc)",
                  ci_run(lambda seed: bench_uea(seed=seed), seeds=range(3))))
    except FileNotFoundError:
        print("| UEA BasicMotions | fetch data/uea first (sktime GitHub) |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
