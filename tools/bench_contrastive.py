#!/usr/bin/env python3
"""bench_contrastive.py -- ONE harness for leCore's contrastive backlog: reproduce the panel, then gate honestly.

WHY THIS EXISTS (backlog E0.1 / E0.3 / E0.4 / E1.2 / E2.2, docs/BACKLOG_contrastive.md)
------------------------------------------------------------------------------------
The CLM panel (2026-09-26) measured the contrastive ideas with nine scratch probes, each with its own loader,
split and statistics (docs/research/evidence/clm_panel_20260926/). Scratch probes are evidence, not a harness:
three different "gate" methods produced coverage-at-0.97 numbers from 3.3% to 37.7% for the SAME index. This file
is the one place those numbers are reproduced -- same protocol, same seeds, to the printed digit -- and the one
place every later gate is measured, with ONE gate protocol (gate_protocol, E0.4), ONE bootstrap, ONE hash split.

The InfoNCE arms run through the SHARED rule, holographic.agents_and_reasoning.holographic_protostore.ProtoStore,
never a private copy: if the shared rule drifts, this harness is what notices. The baselines (miss-only AdaptHD at
lr 0.3 and lr 1, the positive-only "every confirmed wording joins the row" rule) are spelled out here because they
ARE the baselines -- they mirror SystemOne.observe and MeaningIndex.link exactly as the panel's probe did.

DATA (not vendored; pass --data DIR): clinc150_full.json (CLINC150, Larson et al. 2019, CC BY 3.0) and
banking77_train.csv / banking77_test.csv (Banking77, Casanueva et al. 2020, CC BY 4.0). The sha256 of every input
file is written into the output, so a number can always be traced to the exact bytes it came from.
Banking77 ships no validation split: hash_val_split() carves one by sha256 of the TEXT (deterministic, and a
duplicated wording lands on one side only). The learned meaning index behind the 0.8413 baseline is NOT in the
repo either: the `learned` and `prior-trap` subcommands take --index PATH (the sweep-181 idx_clinc.json).

SUBCOMMANDS (each prints its section and merges it into docs/research/evidence/bench_contrastive.json)
  repro       E0.1  the w1 one-pass baselines: InfoNCE (ProtoStore) vs AdaptHD lr 0.3 / 1 vs positive-only, clean and
                    10% teacher noise, Banking77 seeds 0/1 and CLINC150; plus E1.2's kept negative (decision-time
                    reverse normalisation, -6.0 top-1 points)
  teacher     E0.1  w1 probe_gate: teacher self-agreement eps_hat (TeacherNoise) and the eps-corrected static gate
  learned     E0.1  w4 p0/p2b/p3 on the learned index (--index): baseline 0.8413, InfoNCE 0.8616, candidate-relative
                    softmax AURC 0.080 / 0.086 / 0.094 vs absolute cosine 0.078
  prior-trap  E0.4  w4 p4 on the learned index (--index): why the gate protocol exists (37.7% -> 3.3% coverage swing;
                    val holds 3.2% out-of-scope vs test's 18%)
  gate        E0.4  the protocol applied to the one-pass arms: 3 seeds, CLINC150 + Banking77 (hash-carved val), paired
                    bootstrap CIs, configurations counted
  negq        E0.3  negative-quality metric (hardness x safety, after ECI) on planted teacher errors
  miner       E1.2  ProtoStore.confusions() vs a centroid-cosine miner: precision@K of TRAIN-mined pairs on TEST
  curriculum  E2.2  ProtoStore.update(curriculum_k0=3) vs the plain rule, 3 seeds, paired CIs
  systemone   hook  SystemOne(scorer="contrastive") on the Banking77 one-pass protocol, once it exists
  all               everything above (learned / prior-trap only when --index is given)

WHAT IT MEASURED (2026-09-26; every number is in bench_contrastive.json with the command that reproduces it)
  * E0.1 REPRODUCED: all 49 checked numbers match the panel probes' printed 4 digits -- Banking77 InfoNCE 0.8263 /
    0.8218 vs AdaptHD 0.7253 / 0.7377 (lr 0.3) and 0.6847 / 0.6974 (lr 1); CLINC150 0.8598 vs 0.7938 / 0.7504; 10%
    noise 0.7828 vs positive-only 0.7403 vs AdaptHD lr 1 0.5682; eps_hat 0 / 0.0840-0.1065 / 0.1972-0.2103; corrected
    gate 20.07% served at 0.47% wrong; AURC 0.0780 absolute vs 0.0797 / 0.0855 / 0.0942 softmax; learned index 0.8413
    -> 0.8616 (+2.02, CI [+1.29, +2.71]). ProtoStore makes the probe's decisions at every step; its prototypes differ
    from the probe's arrays by <= 1.5e-13 (normalisation epsilon: x / max(|x|, 1e-12) vs x / (|x| + 1e-12)).
    Two PANEL 3-digit strings are double roundings (0.1065 -> "0.107", 0.08546 -> "0.086"); the probe digits match.
  * E0.4: InfoNCE beats AdaptHD lr 0.3 on every gate statistic, 3 seeds, paired CIs excluding 0 -- CLINC150 AURC
    0.0770 vs 0.1183, coverage at calibrated P 0.95 62.5% vs 44.6%; Banking77 0.0903 vs 0.1472, 59.9% vs 36.7%.
    Loud: realised precision at the P 0.95 threshold was 0.944 on Banking77 (0.006 under target), and p_top (tau
    0.02, candidate-relative) serves 1.7x the out-of-scope questions of the absolute g at the same target (CLINC
    6.3% vs 3.8%) although its AURC is marginally lower.
  * E0.3 KEPT NEGATIVE (the draft as specified): quality = HM(hardness, 1 - best cosine to A's wordings) flags planted
    teacher errors at AUROC 0.33 (meaning index) / 0.32 (n-grams) -- WORSE than chance: hardness rewards exactly the
    teacher's errors. Clamping ECI's margin at 0, or ranking by the teacher's pick rank, gives no usable threshold.
    PASS: the raw teacher MARGIN, cos(q, prototype of the pick B) - cos(q, prototype of A): AUROC 0.936 / 0.939 at
    0.41% / 0.43% false flags (recall 44% / 38%); 0.886 / 0.889 when the planted errors are near-misses.
  * E1.2: the ProtoStore miner beats centroid cosine on CLINC150 (P@50 +6..+14 points, CI > 0 on 3/3 seeds); on
    Banking77 it is not distinguishable (CI includes 0 in 17 of 18 seed x K cells; both near ceiling). Kept negative
    reproduced: decision-time reverse normalisation -6.04 top-1 points (-0.07 with same-label exclusion, lam 0.5).
  * E2.2: curriculum_k0=3 on clean streams is within seed noise (K=1 +0.05, K=5 -0.29 points; no seed's CI excludes
    0): kept as an arm, not the rule. Under 10% noise it helps (K=5 +1.48, 3/3 seeds) but the noise-corrected target
    (eps 0.1) does as much (+1.62) and curriculum adds nothing on top (-0.02); at K=1 the eps target HURTS (-1.16)
    and the curriculum rescues it (+0.89 over eps alone, 2/3 seeds).

Usage (from the repo root; ~7 min of CPU for `all` -- 20 min wall on a box loaded 4x -- encodings cached with --cache):
    export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python3 tools/bench_contrastive.py --data DIR [--cache DIR] [--index idx_clinc.json] all
    python3 tools/bench_contrastive.py --data DIR repro

numpy + stdlib only (the learned-index sparse algebra is done with postings, not scipy, so CI's numpy-only venv can
import this file). Deterministic: hashlib, seeded Generators, stable sorts.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_JSON = os.path.join(ROOT, "docs", "research", "evidence", "bench_contrastive.json")
DATA_FILES = ("clinc150_full.json", "banking77_train.csv", "banking77_test.csv")
# The stated DEPLOYMENT prior every precision is computed under: 4,500 in-scope to 1,000 out-of-scope questions,
# i.e. CLINC150's test mix (18.2% out-of-scope). Val holds 100 out-of-scope per 3,000 (3.2%): a threshold chosen on
# val's raw mix is optimistic by construction -- the prior-trap subcommand measures by how much.
PRIOR = (4500.0, 1000.0)
DIM = 2048


# =============================================================================================== data (pinned)
def sha256_file(path):
    """sha256 of a file's bytes (streamed) -- the pin that ties every number to the exact input it came from."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def pinned_inputs(data_dir, index=None):
    """{file: sha256} for the dataset files present in data_dir (+ the learned index when one is used)."""
    out = {}
    for fn in DATA_FILES:
        p = os.path.join(data_dir, fn) if data_dir else ""
        out[fn] = sha256_file(p) if p and os.path.exists(p) else None
    if index:
        out["index:" + os.path.basename(index)] = sha256_file(index)
    return out


def load_clinc(data_dir):
    """CLINC150 exactly as the panel's probes read it: train as {intent: [texts in file order]}, the rest as lists."""
    with open(os.path.join(data_dir, "clinc150_full.json")) as f:
        d = json.load(f)
    train = {}
    for t, i in d["train"]:
        train.setdefault(i, []).append(t)
    return {"train": train, "test": [tuple(x) for x in d["test"]], "val": [tuple(x) for x in d["val"]],
            "oos_train": [t for t, _ in d["oos_train"]], "oos_val": [t for t, _ in d["oos_val"]],
            "oos_test": [t for t, _ in d["oos_test"]]}


def _csv_rows(path):
    with open(path, newline="") as f:
        return [(r["text"], r["category"]) for r in csv.DictReader(f)]


def load_banking(data_dir):
    """Banking77 exactly as the probes read it: train {intent: [texts in file order]}, test [(text, intent)]."""
    train = {}
    for t, i in _csv_rows(os.path.join(data_dir, "banking77_train.csv")):
        train.setdefault(i, []).append(t)
    return {"train": train, "test": _csv_rows(os.path.join(data_dir, "banking77_test.csv"))}


def hash_fraction(text, salt=""):
    """A deterministic uniform number in [0, 1) from sha256(salt + text). Never hash(): PYTHONHASHSEED-proof."""
    return int(hashlib.sha256((salt + text).encode("utf-8")).hexdigest()[:16], 16) / float(1 << 64)


BANKING_VAL_SALT = "bench_contrastive/banking77-val/v1:"


def hash_val_split(train, frac=0.1, salt=BANKING_VAL_SALT):
    """Carve a validation split out of a train dict by sha256 of the TEXT (Banking77 ships none).

    -> (train_rest {intent: [texts]}, val [(text, intent)]). Deterministic on any machine and Python version; the
    same wording always lands on the same side, so a duplicated training sentence cannot leak into val. Intents are
    visited in sorted order and texts in file order, so the result does not depend on dict insertion order."""
    rest, val = {}, []
    for intent in sorted(train):
        for t in train[intent]:
            if hash_fraction(t, salt) < frac:
                val.append((t, intent))
            else:
                rest.setdefault(intent, []).append(t)
    return rest, val


def oos_half_a(texts):
    """w4's split of oos_test, verbatim: half A (even sha256 of the question) CALIBRATES, half B (odd) is REPORTED."""
    return np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 == 0 for q in texts], bool)


# =============================================================================================== the encoder
def encode_texts(texts, dim=DIM, lo=3, hi=5, chunk=6000, cache=None, dtype=np.float64, normalise=True):
    """SystemOne's hashed_ngram_encode (char 3..5-grams, sha256-seeded gaussian atoms), unit rows, BIT-FOR-BIT the
    panel probe's encode_all: computed n-gram-major (each atom generated once per chunk and scattered), in sorted
    n-gram order per text, then divided by (norm + 1e-12) -- SystemOne._unit's normalisation.

    Why chunks give the same bits: a text's vector is the sum of its own n-gram atoms added in globally sorted order,
    which is its own sorted order whatever other texts share the chunk. Chunking only bounds the posting dict's
    memory (the box OOM-kills near 2 GB). cache: a directory for .npy files keyed by sha256(dim, texts).
    normalise=False returns the raw bundles: numpy's Generator gives the SAME first d normals whatever length is
    drawn, so the first d columns of a raw d_max encoding ARE the raw d encoding (bench_curves encodes once)."""
    path = None
    if cache:
        h = hashlib.sha256(("%d|%d|%d|%s|%d|" % (dim, lo, hi, np.dtype(dtype).str, normalise)).encode())
        for t in texts:
            h.update(t.encode("utf-8") + b"\x00")
        path = os.path.join(cache, "enc_%s.npy" % h.hexdigest()[:24])
        if os.path.exists(path):
            return np.load(path)
    H = np.zeros((len(texts), dim), dtype)
    for a in range(0, len(texts), chunk):
        part = texts[a:a + chunk]
        post = {}
        for j, text in enumerate(part):
            t = " " + text.lower() + " "
            for n in range(lo, hi + 1):
                for i in range(len(t) - n + 1):
                    g = t[i:i + n]
                    d = post.setdefault(g, {})
                    d[j] = d.get(j, 0) + 1
        B = np.zeros((len(part), dim), np.float64)
        for g in sorted(post):
            seed = int.from_bytes(hashlib.sha256(g.encode()).digest()[:8], "big") % (2 ** 32)
            v = np.random.default_rng(seed).standard_normal(dim)
            idx = np.fromiter(post[g].keys(), np.int64)
            c = np.fromiter(post[g].values(), np.float64)
            B[idx] += c[:, None] * v[None, :]
        if normalise:
            B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
        H[a:a + len(part)] = B
        del post, B
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, H)
    return H


class Encoded:
    """Every distinct text of a dataset encoded ONCE; rows looked up by text (identical text -> identical vector)."""

    def __init__(self, texts, dim=DIM, cache=None, dtype=np.float64):
        uniq, seen = [], set()
        for t in texts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        self.row = {t: i for i, t in enumerate(uniq)}
        self.dim = dim
        self.H = encode_texts(uniq, dim=dim, cache=cache, dtype=dtype)

    def __call__(self, texts):
        return self.H[[self.row[t] for t in texts]]


def dataset_texts(ds, extra=()):
    """All texts of a loaded dataset dict (every split), in a fixed order, plus `extra` texts.
    Every subcommand encodes the SAME universe per dataset (CLINC150: all splits; Banking77: all splits + CLINC150's
    oos_test, its stand-in out-of-scope set), so one cached encoding serves them all."""
    out = [t for i in sorted(ds["train"]) for t in ds["train"][i]]
    for k in ("test", "val"):
        out += [t for t, _ in ds.get(k, [])]
    for k in ("oos_train", "oos_val", "oos_test"):
        out += list(ds.get(k, []))
    return out + list(extra)


def universe(name, clinc, bank, cache):
    """The cached encoding of one dataset's text universe (see dataset_texts)."""
    if name == "clinc":
        return Encoded(dataset_texts(clinc), cache=cache)
    return Encoded(dataset_texts(bank, clinc["oos_test"]), cache=cache)


# =============================================================================================== statistics
def auroc(score, ok):
    """AUROC with average ranks for ties (the w1 probe's implementation, verbatim): P(score of a positive > score of
    a negative). nan when one class is empty."""
    score, ok = np.asarray(score, float), np.asarray(ok, bool)
    pos, neg = score[ok], score[~ok]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    # average 1-based rank of a tie group occupying sorted positions i..j is (i + j) / 2 + 1 = cumcount - (cnt-1)/2:
    # the probe's while-loop, vectorised (ranks are exact halves, so the sum -- and the AUROC -- is bit-identical)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def paired_bootstrap(a, b, n=1000, seed=0):
    """Paired bootstrap of mean(b) - mean(a) over the SAME items (a, b: per-item values, e.g. 0/1 correct).

    Paired because two arms answering the same questions share most of their variance; resampling them
    independently would inflate the interval. -> {a, b, diff, lo, hi, b_only, a_only, n, seed} with a 95%
    percentile interval. b_only / a_only count items one arm gets right and the other wrong (McNemar's cells).
    Deterministic: one seeded Generator, the (n x N) index matrix drawn at once (w4's p2 procedure)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    N = len(a)
    idx = rng.integers(0, N, size=(n, N))
    diff = b[idx].mean(1) - a[idx].mean(1)
    return {"a": float(a.mean()), "b": float(b.mean()), "diff": float(b.mean() - a.mean()),
            "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)),
            "b_only": int(((b > 0) & ~(a > 0)).sum()), "a_only": int(((a > 0) & ~(b > 0)).sum()),
            "n": int(n), "seed": int(seed)}


def paired_bootstrap_stat(stat_a, stat_b, n_items, n=1000, seed=0):
    """Paired bootstrap of stat_b(idx) - stat_a(idx) for statistics that are not per-item means (AURC, AUROC,
    precision@K): both arms are evaluated on the SAME resampled item indices. -> {diff, lo, hi, n, seed}."""
    rng = np.random.default_rng(seed)
    full = np.arange(n_items)
    d0 = stat_b(full) - stat_a(full)
    ds = np.empty(n)
    for r in range(n):
        idx = rng.integers(0, n_items, size=n_items)
        ds[r] = stat_b(idx) - stat_a(idx)
    return {"diff": float(d0), "lo": float(np.percentile(ds, 2.5)), "hi": float(np.percentile(ds, 97.5)),
            "n": int(n), "seed": int(seed)}


# =============================================================================================== E0.4 gate protocol
def _served_counts(sig, ok, is_in):
    """For every DISTINCT score u (ascending): counts of items with sig >= u -- right (in-scope and correct),
    served in-scope, served out-of-scope. Vectorised form of 'for thr in np.unique(sig)': identical integers."""
    u = np.unique(sig)
    o = np.argsort(sig, kind="stable")
    lo = np.searchsorted(sig[o], u, side="left")

    def suffix(x):
        return np.concatenate([np.cumsum(x[::-1])[::-1], [0]])[lo]
    return (u, suffix((ok & is_in)[o].astype(np.int64)), suffix(is_in[o].astype(np.int64)),
            suffix((~is_in)[o].astype(np.int64)))


def prior_rates(right, served_in, oos, prior=PRIOR):
    """Precision under the deployment prior from RATES (w4 p2b's prec_rates, same float expression):
    right = in-scope served-and-correct / n_in, served_in = in-scope served / n_in, oos = out-of-scope served / n_oos.
    A split's own in/out mix never enters: that is the point (val's 3.2% out-of-scope would flatter every gate)."""
    n_in, n_oos = prior
    return right * n_in / np.maximum(served_in * n_in + oos * n_oos, 1e-9)


def calibrate_threshold(sig, ok, is_in, target=None, max_wrong=None, prior=PRIOR):
    """The LOWEST threshold whose prior-weighted precision on the calibration items is >= target (w4 p2b's
    gate_thr: scan distinct values, keep the last one that passes), or -- with max_wrong -- whose served-WRONG share
    of all questions (in-scope wrong + out-of-scope served, under the prior) is <= max_wrong. inf if none passes."""
    sig = np.asarray(sig, np.float64)
    ok = np.asarray(ok, bool) & np.asarray(is_in, bool)
    is_in = np.asarray(is_in, bool)
    n_in, n_oos = int(is_in.sum()), int((~is_in).sum())
    u, c_right, c_in, c_oos = _served_counts(sig, ok, is_in)
    right = c_right / n_in
    served_in = c_in / n_in
    oos = c_oos / max(n_oos, 1)
    if max_wrong is None:
        good = prior_rates(right, served_in, oos, prior) >= target
    else:
        wrong_all = ((served_in - right) * prior[0] + oos * prior[1]) / (prior[0] + prior[1])
        good = wrong_all <= max_wrong
    return float(u[good].min()) if good.any() else float("inf")


def realised(sig, ok, is_in, thr, prior=PRIOR):
    """What a threshold actually does on the REPORT items: prior-weighted precision, in-scope coverage (served and
    right / n_in), in-scope served-wrong / n_in, out-of-scope served / n_oos, served share of all questions."""
    sig = np.asarray(sig, np.float64)
    is_in = np.asarray(is_in, bool)
    ok = np.asarray(ok, bool) & is_in
    s = sig >= thr
    n_in, n_oos = is_in.sum(), (~is_in).sum()
    right = (ok & s & is_in).sum() / n_in
    served_in = (s & is_in).sum() / n_in
    oos = (s & ~is_in).sum() / max(n_oos, 1)
    prec = right * prior[0] / max(served_in * prior[0] + oos * prior[1], 1e-9)
    return {"thr": float(thr), "precision": float(prec), "coverage": float(right),
            "wrong_in": float(served_in - right), "oos_served": float(oos),
            "wrong_all": float(((served_in - right) * prior[0] + oos * prior[1]) / (prior[0] + prior[1])),
            "served_all": float((served_in * prior[0] + oos * prior[1]) / (prior[0] + prior[1]))}


def risk_coverage(sig, ok, is_in, prior=PRIOR):
    """The prior-weighted risk-coverage curve (w4 p3's method): items sorted by score (stable), out-of-scope items
    weighted so the report mix equals the prior. -> (order, precision-at-each-prefix, served-fraction, weights).

    NOTE on p3's weight: it wrote w_oos = N_OOS/n_oos * (N_IN/n_in), which is only right when n_in == N_IN (4,500,
    CLINC's test). The general weight is N_OOS/n_oos * (n_in/N_IN) -- the SAME floating value at n_in == 4,500, so
    the panel's numbers reproduce bit-for-bit and Banking77 (3,080 in-scope) is weighted correctly."""
    sig = np.asarray(sig, np.float64)
    is_in = np.asarray(is_in, bool)
    ok = np.asarray(ok, bool) & is_in
    n_in, n_oos = int(is_in.sum()), int((~is_in).sum())
    w_oos = (prior[1] / n_oos * (n_in / prior[0])) if n_oos else 0.0
    w = np.where(is_in, 1.0, w_oos)
    o = np.argsort(-sig, kind="stable")
    cw = np.cumsum(w[o])
    cr = np.cumsum((ok * w)[o])
    return o, cr / cw, cw / cw[-1], w


def aurc(sig, ok, is_in, prior=PRIOR):
    """Area under the prior-weighted risk-coverage curve (lower is better): mean risk over every served prefix.
    Threshold-FREE, which is why it is the headline gate statistic -- coverage at exactly P = 0.97 swung 37.7% ->
    3.3% between two calibration samples of one index (prior-trap)."""
    o, prec, _, w = risk_coverage(sig, ok, is_in, prior)
    return float(np.sum((1 - prec) * w[o]) / np.sum(w))


def gate_protocol(scores, correct, is_oos, is_cal, targets=(0.95, 0.97), max_wrong=(), prior=PRIOR,
                  served_fracs=(0.2, 0.3, 0.4, 0.5, 0.6), oracle=True):
    """THE gate protocol (backlog E0.4). Every gate from E1 onward is measured through this function.

    scores  : one confidence per question (higher = more sure); any scale -- cosine, lead, calibrated p
    correct : the door's top answer is right (ignored for out-of-scope questions: serving one is always wrong)
    is_oos  : the question matches no option
    is_cal  : True = CALIBRATION item, False = REPORT item. The convention: calibrate on val in-scope + the sha256
              half A of oos_test; report on test in-scope + half B (oos_half_a). Nothing chosen on report items.
    targets : precisions a threshold is calibrated for; max_wrong: served-wrong shares (coverage at <= x% wrong)

    Returns (all rates on the REPORT items, precision always under the stated prior, never the split's own mix):
      n                  item counts per side
      top1               in-scope top-1 on the report items
      aurc               area under the risk-coverage curve (threshold-free headline)
      precision_at_served {0.2: ..., 0.6: ...} precision when serving that share of all questions
      calibrated         {"P0.97": realised precision / coverage / wrong / out-of-scope served at the threshold chosen
                          on the CALIBRATION items} -- the only threshold a shipping claim may use
      oracle             the same targets with the threshold chosen ON the report items: a SYMMETRIC DIAGNOSTIC only
                          (every arm gets the same advantage), never a shipping number
    """
    sig = np.asarray(scores, np.float64)
    is_oos = np.asarray(is_oos, bool)
    is_cal = np.asarray(is_cal, bool)
    is_in = ~is_oos
    ok = np.asarray(correct, bool) & is_in
    cal, rep = is_cal, ~is_cal
    out = {"n": {"cal_in": int((cal & is_in).sum()), "cal_oos": int((cal & is_oos).sum()),
                 "rep_in": int((rep & is_in).sum()), "rep_oos": int((rep & is_oos).sum())},
           "prior": list(prior)}
    rs, rk, ri = sig[rep], ok[rep], is_in[rep]
    out["top1"] = float(rk[ri].mean())
    out["aurc"] = aurc(rs, rk, ri, prior)
    o, prec, frac, _ = risk_coverage(rs, rk, ri, prior)
    out["precision_at_served"] = {"%.2f" % f: float(prec[min(np.searchsorted(frac, f), len(prec) - 1)])
                                  for f in served_fracs}
    out["calibrated"] = {}
    for P in targets:
        thr = calibrate_threshold(sig[cal], ok[cal], is_in[cal], target=P, prior=prior)
        out["calibrated"]["P%.2f" % P] = realised(rs, rk, ri, thr, prior)
    for W in max_wrong:
        thr = calibrate_threshold(sig[cal], ok[cal], is_in[cal], max_wrong=W, prior=prior)
        out["calibrated"]["W%.3f" % W] = realised(rs, rk, ri, thr, prior)
    if oracle:
        out["oracle"] = {}
        for P in targets:
            idx = np.where(prec >= P)[0]
            k = idx.max() + 1 if len(idx) else 0
            out["oracle"]["P%.2f" % P] = {"coverage": float(rk[o][:k].sum() / ri.sum())}
    return out


def assemble(val_sig, val_ok, test_sig, test_ok, oos_sig, oos_texts):
    """Lay out the E0.4 split: [val in-scope | test in-scope | oos_test], with oos half A on the calibration side.
    -> (scores, correct, is_oos, is_cal) ready for gate_protocol."""
    half_a = oos_half_a(oos_texts)
    nv, nt, no = len(val_sig), len(test_sig), len(oos_sig)
    scores = np.concatenate([val_sig, test_sig, oos_sig])
    correct = np.concatenate([np.asarray(val_ok, bool), np.asarray(test_ok, bool), np.zeros(no, bool)])
    is_oos = np.concatenate([np.zeros(nv + nt, bool), np.ones(no, bool)])
    is_cal = np.concatenate([np.ones(nv, bool), np.zeros(nt, bool), half_a])
    return scores, correct, is_oos, is_cal


# =============================================================================================== reporting helpers
def match(value, panel, nd=None):
    """Does a reproduced value print as the panel's number? panel is the STRING the panel printed ('0.826', '20.1'),
    so its digit count is the comparison precision (no rounding judgement of ours)."""
    if nd is None:
        nd = len(panel.split(".")[1]) if "." in panel else 0
    return "%.*f" % (nd, value) == panel


def check(rows, name, value, panel, probe=None, scale=1.0, note=None):
    """Append one reproduction row {metric, reproduced, panel, probe, match} (value * scale is compared)."""
    v = float(value) * scale
    r = {"metric": name, "reproduced": round(v, 6), "panel": panel, "match_panel": match(v, panel)}
    if probe is not None:
        r["probe"] = probe
        r["match_probe"] = match(v, probe)
    if note:
        r["note"] = note
    rows.append(r)
    return r


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def write_section(out_path, name, payload):
    """Merge one subcommand's section into the output JSON (other sections kept), atomically."""
    doc = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                doc = json.load(f)
        except ValueError:
            doc = {}
    doc["about"] = ("tools/bench_contrastive.py -- the contrastive backlog's one harness (E0.1/E0.3/E0.4/E1.2/E2.2). "
                    "Each section records the command that reproduces it and the sha256 of every input.")
    doc.setdefault("sections", {})[name] = payload
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, default=_json_default)
    os.replace(tmp, out_path)


def section_header(args, name, extra=""):
    """The reproduce-me block every section carries."""
    cmd = "PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python3 tools/bench_contrastive.py --data DATA"
    if getattr(args, "index", None) and name in ("learned", "prior_trap"):
        cmd += " --index idx_clinc.json"
    return {"command": cmd + " " + name.replace("_", "-") + extra,
            "inputs_sha256": pinned_inputs(args.data, args.index if name in ("learned", "prior_trap") else None),
            "python": sys.version.split()[0], "numpy": np.__version__}


def printj(obj):
    print(json.dumps(obj, indent=1, default=_json_default), flush=True)


# =============================================================================================== the one-pass stream (w1)
def w1_protocol(train, test, K, n_stream, seed):
    """The w1 probe's split, verbatim: labels sorted; init = the FIRST K training wordings per intent (file order);
    stream = every other training wording, shuffled with random.Random(seed), first n_stream kept.
    -> (labels, init [(text, y)], stream [(text, y)], test [(text, y)])."""
    labels = sorted(train)
    L = {l: i for i, l in enumerate(labels)}
    init = [(t, L[l]) for l in labels for t in train[l][:K]]
    stream = [(t, L[l]) for l in labels for t in train[l][K:]]
    random.Random(seed).shuffle(stream)
    stream = stream[:n_stream]
    return labels, init, stream, [(t, L[l]) for t, l in test]


def noise_draws(n, noise, seed):
    """The probe's teacher-noise draws: identical for every arm (flip = which verdicts are wrong, pick = which wrong
    candidate). The wrong label itself is chosen among the ARM'S OWN current top-8, as the probe did."""
    rng = np.random.default_rng(seed + 99)
    return rng.random(n) < noise, rng.random(n)


def _noisy_label(s, y, flip, pick):
    if not flip:
        return y
    top8 = np.argsort(-s, kind="stable")[:8]
    wrong = [int(k) for k in top8 if k != y]
    return wrong[int(pick * len(wrong))]


def run_arm(arm, labels, Hi, yi, Hs, ys, noise=0.0, seed=0, flips=None):
    """One pass of one learning rule over the stream: decide first (prequential), then learn from the (possibly
    noisy) label. -> {P: final unit prototypes (C x dim), preq, store (ProtoStore or None), preds, labels_used}.

    arm = {"rule": "infonce" | "adapthd" | "positive" | "frozen", "lr", "tau", "topk", "eps", "curriculum_k0"}
      infonce  : THE SHARED RULE -- ProtoStore(dim, tau, lr, topk).update(q, label)            (never a copy)
      adapthd  : SystemOne.observe's miss-only rule: on a miss, A[label] += lr q, A[pred] -= lr q, renormalise
      positive : MeaningIndex.link-like: every confirmed wording joins its row, A[label] += lr q
      frozen   : the init prototypes, no learning (what learning adds is measured against this)"""
    C, dim = len(labels), Hi.shape[1]
    flip, pick = flips if flips is not None else noise_draws(len(ys), noise, seed)
    rule = arm["rule"]
    preds = np.empty(len(ys), np.int64)
    used = np.empty(len(ys), np.int64)
    if rule == "infonce":
        from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
        store = ProtoStore(dim, tau=arm.get("tau", 0.05), lr=arm.get("lr", 0.3), topk=arm.get("topk"),
                           name="bench", mine=True)
        for c in range(C):
            store.add_option(labels[c], Hi[yi == c])
        if arm.get("seed_count"):
            # DIAGNOSTIC (E2.2): count the seeding wordings as the row's maturity, the probe's `nrow` semantics
            for c in range(C):
                store.count[c] = int((yi == c).sum())
        L = store._ix
        for n in range(len(ys)):
            q = Hs[n]
            y = int(ys[n])
            s = store.scores(q)
            lab = _noisy_label(s, y, flip[n], pick[n])
            used[n] = lab
            r = store.update(q, labels[lab], eps=arm.get("eps", 0.0), curriculum_k0=arm.get("curriculum_k0"))
            preds[n] = L[r["pred"]]
        return {"P": store.P, "preq": float((preds == ys).mean()), "store": store, "preds": preds, "used": used}
    # the baselines, spelled out (they ARE the baselines; the probe's arithmetic, operation for operation)
    lr = arm.get("lr", 1.0)
    A = np.zeros((C, dim))
    for c in range(C):
        mu = Hi[yi == c].mean(0)
        A[c] = mu / (np.linalg.norm(mu) + 1e-12)
    P = A / np.linalg.norm(A, axis=1, keepdims=True)
    for n in range(len(ys)):
        q = Hs[n]
        s = P @ q
        pred = int(np.argmax(s))
        y = int(ys[n])
        preds[n] = pred
        lab = _noisy_label(s, y, flip[n], pick[n])
        used[n] = lab
        if rule == "frozen":
            continue
        if rule == "adapthd":
            if pred != lab:
                A[lab] += lr * q
                A[pred] -= lr * q
                for i in (lab, pred):
                    P[i] = A[i] / (np.linalg.norm(A[i]) + 1e-12)
            continue
        if rule == "positive":
            A[lab] += lr * q
            P[lab] = A[lab] / (np.linalg.norm(A[lab]) + 1e-12)
            continue
        raise ValueError(rule)
    return {"P": P, "preq": float((preds == ys).mean()), "store": None, "preds": preds, "used": used}


def top1_eval(P, Ht, yt):
    """Test top-1 and the per-question pieces the gate needs: s1, s2, correct, p_top at tau 0.02."""
    St = Ht @ P.T
    pr = np.argmax(St, axis=1)
    ok = pr == yt
    srt = -np.sort(-St, axis=1)
    z = (St - srt[:, :1]) / 0.02
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return {"top1": float(ok.mean()), "ok": ok, "pred": pr, "s1": srt[:, 0], "s2": srt[:, 1], "ptop02": p.max(1),
            "St": St}


def reverse_normalisation(P, Hbank, ybank, Ht, yt, C, tau=0.05):
    """E1.2's KEPT NEGATIVE, the w1 probe's arithmetic: normalise each row's score at decision time by its log-
    partition over a bank of the last 256 stream questions (querybank normalisation / inverted softmax),
    s'_k = s_k - tau * log mean_j exp(s_jk / tau). Plus the false-negative rule (bank questions labelled k are not
    negatives for row k) and damped versions (lam * the correction). -> {variant: top-1}."""
    bank = Hbank @ P.T
    St = Ht @ P.T
    out = {}
    bk = tau * np.log(np.mean(np.exp((bank - bank.max(0, keepdims=True)) / tau), axis=0)) + bank.max(0)
    out["qbnorm"] = float((np.argmax(St - bk[None, :], axis=1) == yt).mean())
    E = np.exp((bank - bank.max()) / tau)
    E[np.arange(len(ybank)), ybank] = 0.0
    cnt = len(ybank) - np.bincount(ybank, minlength=C)
    bk2 = tau * np.log(E.sum(0) / np.maximum(cnt, 1) + 1e-300) + bank.max()
    out["qbnorm_excl"] = float((np.argmax(St - bk2[None, :], axis=1) == yt).mean())
    for lam in (0.25, 0.5):
        out["qbnorm_excl_lam%.2f" % lam] = float((np.argmax(St - lam * bk2[None, :], axis=1) == yt).mean())
    return out


def _prep(ds, enc, K, n_stream, seed):
    labels, init, stream, test = w1_protocol(ds["train"], ds["test"], K, n_stream, seed)
    return (labels, enc([t for t, _ in init]), np.array([y for _, y in init]),
            enc([t for t, _ in stream]), np.array([y for _, y in stream]),
            enc([t for t, _ in test]), np.array([y for _, y in test]))


ARMS_W1 = {
    "infonce_t0.05_lr0.3": {"rule": "infonce", "tau": 0.05, "lr": 0.3},
    "adapthd_lr0.3": {"rule": "adapthd", "lr": 0.3},
    "adapthd_lr1": {"rule": "adapthd", "lr": 1.0},
    "positive_only_lr1": {"rule": "positive", "lr": 1.0},
}


# =============================================================================================== subcommand: repro (E0.1)
def cmd_repro(args):
    """The w1 one-pass table, reproduced through ProtoStore: Banking77 (3,000-verdict stream, seeds 0 and 1, clean;
    seed 0 at 10% noise) and CLINC150 (4,000-verdict stream, seed 0). K = 5 seed wordings per intent, d = 2048."""
    t0 = time.time()
    res = {"protocol": "w1 probe_proto: init = first K=5 train wordings/intent (unit mean), one shuffled pass over "
                       "n_stream other train wordings (decide, then learn), test top-1 with the final prototypes; "
                       "noise = with prob r the label is a uniformly random WRONG intent among the arm's top-8",
           "runs": {}}
    rows = []
    bank, clinc = load_banking(args.data), load_clinc(args.data)
    enc = universe("banking", clinc, bank, args.cache)
    for seed, noise in ((0, 0.0), (1, 0.0), (0, 0.1)):
        labels, Hi, yi, Hs, ys, Ht, yt = _prep(bank, enc, 5, 3000, seed)
        run = {}
        for name, arm in ARMS_W1.items():
            r = run_arm(arm, labels, Hi, yi, Hs, ys, noise=noise, seed=seed)
            ev = top1_eval(r["P"], Ht, yt)
            run[name] = {"prequential_acc": round(r["preq"], 4), "test_top1": round(ev["top1"], 4),
                         "auroc_gap": round(auroc(ev["s1"] - ev["s2"], ev["ok"]), 4),
                         "auroc_ptop_tau0.02": round(auroc(ev["ptop02"], ev["ok"]), 4)}
            if name == "infonce_t0.05_lr0.3" and noise == 0.0:
                rn = reverse_normalisation(r["P"], Hs[-256:], ys[-256:], Ht, yt, len(labels))
                run[name]["reverse_normalisation_top1"] = {k: round(v, 4) for k, v in rn.items()}
                run[name]["reverse_normalisation_cost_points"] = round(100 * (rn["qbnorm"] - ev["top1"]), 2)
            print("banking seed %d noise %.1f %-22s top1 %.4f  (%.0fs)" % (seed, noise, name, ev["top1"],
                                                                        time.time() - t0), flush=True)
        res["runs"]["banking_seed%d_noise%.1f" % (seed, noise)] = run
    del enc
    enc = universe("clinc", clinc, bank, args.cache)
    labels, Hi, yi, Hs, ys, Ht, yt = _prep(clinc, enc, 5, 4000, 0)
    run = {}
    for name, arm in ARMS_W1.items():
        r = run_arm(arm, labels, Hi, yi, Hs, ys, noise=0.0, seed=0)
        ev = top1_eval(r["P"], Ht, yt)
        run[name] = {"prequential_acc": round(r["preq"], 4), "test_top1": round(ev["top1"], 4)}
        print("clinc seed 0 %-22s top1 %.4f  (%.0fs)" % (name, ev["top1"], time.time() - t0), flush=True)
    res["runs"]["clinc_seed0_noise0.0"] = run
    del enc
    R = res["runs"]
    b0, b1, bn, c0 = (R["banking_seed0_noise0.0"], R["banking_seed1_noise0.0"], R["banking_seed0_noise0.1"],
                      R["clinc_seed0_noise0.0"])
    check(rows, "Banking77 InfoNCE seed 0", b0["infonce_t0.05_lr0.3"]["test_top1"], "0.826", "0.8263")
    check(rows, "Banking77 InfoNCE seed 1", b1["infonce_t0.05_lr0.3"]["test_top1"], "0.822", "0.8218")
    check(rows, "Banking77 AdaptHD lr0.3 seed 0", b0["adapthd_lr0.3"]["test_top1"], "0.725", "0.7253")
    check(rows, "Banking77 AdaptHD lr0.3 seed 1", b1["adapthd_lr0.3"]["test_top1"], "0.738", "0.7377")
    check(rows, "Banking77 AdaptHD lr1 seed 0", b0["adapthd_lr1"]["test_top1"], "0.685", "0.6847")
    check(rows, "Banking77 AdaptHD lr1 seed 1", b1["adapthd_lr1"]["test_top1"], "0.697", "0.6974")
    check(rows, "CLINC150 InfoNCE", c0["infonce_t0.05_lr0.3"]["test_top1"], "0.860", "0.8598")
    check(rows, "CLINC150 AdaptHD lr0.3", c0["adapthd_lr0.3"]["test_top1"], "0.794", "0.7938")
    check(rows, "CLINC150 AdaptHD lr1", c0["adapthd_lr1"]["test_top1"], "0.750", "0.7504")
    check(rows, "Banking77 10% noise InfoNCE", bn["infonce_t0.05_lr0.3"]["test_top1"], "0.783", "0.7828")
    check(rows, "Banking77 10% noise positive-only", bn["positive_only_lr1"]["test_top1"], "0.740", "0.7403")
    check(rows, "Banking77 10% noise AdaptHD lr1", bn["adapthd_lr1"]["test_top1"], "0.568", "0.5682")
    rn = b0["infonce_t0.05_lr0.3"]["reverse_normalisation_top1"]
    base = b0["infonce_t0.05_lr0.3"]["test_top1"]
    check(rows, "E1.2 kept negative: reverse normalisation cost (points)", 100 * (rn["qbnorm"] - base), "-6.0",
          note="0.8263 -> 0.7659 in the probe (banking_qb.json)")
    check(rows, "E1.2 reverse normalisation with same-label exclusion, lam 0.5 (points)",
          100 * (rn["qbnorm_excl_lam0.50"] - base), "-0.07", note="banking_qb.json: 0.8256 vs 0.8263")
    res["reproduction"] = rows
    res["configs_tried"] = len(ARMS_W1)
    res["seconds"] = round(time.time() - t0, 1)
    return res


# =============================================================================================== subcommand: teacher (E0.1)
def gate_build(ds_name, ds, oos_tr, oos_te, K=10, n_cal=1500, n_test=1500, n_oos=300, seed=0):
    """w1 probe_gate's static index, verbatim: a MeaningIndex (repo code) holds the first K wordings per intent
    (learn=True links), a calibration stream (train wordings beyond K + oos_train) and a test stream (test +
    oos_test) are ranked ONCE with answers(k=8). -> (mi, rint, cal records, test records)."""
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    tr = ds["train"]
    mi = MeaningIndex()
    rint = {}
    for intent in sorted(tr):
        rid = mi.add_row(tr[intent][0], akey=intent)
        rint[rid] = intent
        for t in tr[intent][1:K]:
            mi.link(rid, t, learn=True)
    rng = random.Random(seed)
    cal = [(t, i) for i in sorted(tr) for t in tr[i][K:]]
    rng.shuffle(cal)
    cal = cal[:n_cal] + [(t, "oos") for t in oos_tr[: n_cal // 10]]
    rng.shuffle(cal)
    te = list(ds["test"])
    rng.shuffle(te)
    te = te[:n_test] + [(t, "oos") for t in oos_te[:n_oos]]

    def rec(q, want):
        ranked = mi.answers(q, k=8)
        return {"q": q, "g": MeaningIndex.confidence(ranked), "want": want, "rids": [r for r, _, _ in ranked],
                "cands": [rint.get(r) for r, _, _ in ranked], "scores": [s for _, s, _ in ranked]}
    return mi, rint, [rec(q, w) for q, w in cal], [rec(q, w) for q, w in te]


def scripted_teacher(r, eps, rng, mode="uniform"):
    """The scripted model end (bench_meaning.ScriptedModel / probe_gate): the right candidate's 1-based rank, or 0
    for 'new' when it is not listed; with probability eps a uniformly random WRONG listed candidate instead.
    mode="nearmiss" (E0.3's robustness check, NOT the panel's stand-in): an error picks the BEST-ranked wrong
    candidate -- the confusable mistake a real model is likelier to make. The rng is consumed identically in both
    modes, so the same verdicts are wrong. -> (pick, planted)."""
    right = [k for k, c in enumerate(r["cands"]) if c == r["want"]]
    if rng.random() < eps:
        wrong = [k for k in range(len(r["cands"])) if k not in right]
        if not wrong:
            return 0, True
        k = rng.choice(wrong)
        return (wrong[0] if mode == "nearmiss" else k) + 1, True
    return (right[0] + 1) if right else 0, False


def teacher_eval(C, T, eps, seed=0, n_reask=100, p_serve=0.95):
    """probe_gate.evaluate, with the SHARED TeacherNoise doing the eps estimate and the correction."""
    from holographic.agents_and_reasoning.holographic_systemone import IsotonicCalibrator
    from holographic.agents_and_reasoning.holographic_protostore import TeacherNoise
    rng = random.Random(seed + 7)
    picks, noisy = [], []
    for r in C:
        pk, nz = scripted_teacher(r, eps, rng)
        picks.append(pk)
        noisy.append(nz)
    g = np.array([r["g"] for r in C])
    agree = np.array([1.0 if p == 1 else 0.0 for p in picks])
    cal = IsotonicCalibrator(list(g), list(agree))
    order = np.argsort(-g, kind="stable")[:max(20, len(g) // 5)]
    ceiling = float(agree[order].mean())                  # the RETIRED teacher-relative ceiling, for the record
    idx = list(range(len(C)))
    random.Random(seed + 11).shuffle(idx)
    idx = idx[:n_reask]
    rng2 = random.Random(seed + 13)
    m_bar = float(np.mean([max(1, len(r["cands"]) - 1) for r in C]))
    tn = TeacherNoise(m=m_bar, min_pairs=20)
    for i in idx:
        a, _ = scripted_teacher(C[i], eps, rng2)
        b, _ = scripted_teacher(C[i], eps, rng2)
        tn.observe(a, b)
    eps_hat = tn.eps_hat()
    dis = [(p, nz) for p, nz in zip(picks, noisy) if p != 1]
    n_noise = sum(1 for p, nz in dis if nz)
    n_gen = sum(1 for p, nz in dis if not nz)
    res = {"eps": eps, "n_cal": len(C), "ceiling": round(ceiling, 4), "reask_agree": round(tn.agreement(), 3),
           "eps_hat": round(eps_hat, 4), "eps_hat_raw": eps_hat, "m_bar": round(m_bar, 2),
           "hold_back_rank>=5": {
               "planted_held": round(sum(1 for p, nz in dis if nz and TeacherNoise.hold_back(p, max(eps_hat, 1e-9)))
                                     / max(1, n_noise), 3),
               "genuine_held": round(sum(1 for p, nz in dis if (not nz) and TeacherNoise.hold_back(p, 1.0))
                                     / max(1, n_gen), 3), "n_planted": n_noise, "n_genuine": n_gen}}
    tg = np.array([r["g"] for r in T])
    pa = np.array([cal.predict(x) for x in tg])
    top_ok = np.array([bool(r["cands"]) and r["cands"][0] == r["want"] for r in T])
    is_oos = np.array([r["want"] == "oos" for r in T])
    has = np.array([bool(r["cands"]) for r in T])

    def gate(serve):
        serve = serve & has
        ins = ~is_oos
        return {"in_correct": round(float((serve & ins & top_ok).sum() / ins.sum()), 4),
                "in_WRONG": round(float((serve & ins & ~top_ok).sum() / ins.sum()), 4),
                "oos_served": round(float((serve & is_oos).sum() / max(1, is_oos.sum())), 4)}
    res["gate_abs_0.95"] = gate(pa >= p_serve)
    pc = np.array([tn.corrected(x, eps_hat) for x in pa])
    res["gate_noisecorrected_eps_hat"] = gate(pc >= p_serve)
    pc_true = np.array([tn.corrected(x, eps) for x in pa])
    res["gate_noisecorrected_eps_true"] = gate(pc_true >= p_serve)
    return res


def cmd_teacher(args):
    """probe_gate on CLINC150 and Banking77 (K = 10 wordings per intent), teacher noise 0 / 0.1 / 0.2."""
    t0 = time.time()
    clinc = load_clinc(args.data)
    res = {"protocol": "w1 probe_gate: static MeaningIndex with K=10 wordings/intent; 1,600 calibration verdicts "
                       "(1,500 train + 100 oos_train), 1,800 test questions (1,500 + 300 oos_test); 100 re-asks "
                       "for eps_hat (TeacherNoise with m = mean candidates - 1); serve bar 0.95", "runs": {}}
    for name in ("clinc", "banking"):
        ds = clinc if name == "clinc" else load_banking(args.data)
        _, _, C, T = gate_build(name, ds, clinc["oos_train"], clinc["oos_test"])
        res["runs"][name] = [teacher_eval(C, T, e) for e in (0.0, 0.1, 0.2)]
        print(name, [(r["eps"], r["eps_hat"], r["gate_noisecorrected_eps_hat"]) for r in res["runs"][name]],
              "(%.0fs)" % (time.time() - t0), flush=True)
    rows = []
    c, b = res["runs"]["clinc"], res["runs"]["banking"]
    check(rows, "eps_hat perfect teacher, CLINC150", c[0]["eps_hat_raw"], "0.000")
    check(rows, "eps_hat perfect teacher, Banking77", b[0]["eps_hat_raw"], "0.000")
    check(rows, "eps_hat at 10% noise, CLINC150 (low end)", c[1]["eps_hat_raw"], "0.084", "0.0840")
    check(rows, "eps_hat at 10% noise, Banking77 (high end)", b[1]["eps_hat_raw"], "0.107", "0.1065",
          note="the panel's 0.107 is 0.1065 rounded twice; the 4-digit probe value is the strict check")
    check(rows, "eps_hat at 20% noise, Banking77 (low end)", b[2]["eps_hat_raw"], "0.197", "0.1972")
    check(rows, "eps_hat at 20% noise, CLINC150 (high end)", c[2]["eps_hat_raw"], "0.210", "0.2103")
    check(rows, "eps-corrected static gate, CLINC150 10% noise: served correct (%)",
          c[1]["gate_noisecorrected_eps_hat"]["in_correct"], "20.1", "20.07", scale=100)
    check(rows, "eps-corrected static gate, CLINC150 10% noise: served wrong (%)",
          c[1]["gate_noisecorrected_eps_hat"]["in_WRONG"], "0.47", scale=100)
    check(rows, "uncorrected gate today, CLINC150 10% noise: served (%)", c[1]["gate_abs_0.95"]["in_correct"],
          "0.3", "0.33", scale=100)
    check(rows, "retired ceiling, perfect teacher, Banking77", b[0]["ceiling"], "0.941", "0.9406")
    check(rows, "retired ceiling, perfect teacher, CLINC150", c[0]["ceiling"], "0.988", "0.9875",
          note="0.988 is 0.9875 rounded twice; the 4-digit probe value is the strict check")
    for r in c + b:
        r.pop("eps_hat_raw", None)
    res["reproduction"] = rows
    res["configs_tried"] = 1
    res["seconds"] = round(time.time() - t0, 1)
    return res


# =============================================================================================== the learned index (numpy-only)
class MeaningSpace:
    """A MeaningIndex's own sparse space as numpy arrays -- the vectors rank() scores with, no scipy.

    Phrasing vectors are meaning_features x idf (exactly MeaningIndex._build); x_hat = x / |x|. A query is
    _query_vec(text) (learned association expansion included) divided by its norm INCLUDING the unknown-word mass,
    so cos(q, p) is bit-for-bit the index's formula up to summation order (w4 common.py did this with scipy; the
    order of a handful of float additions is the only difference, ~1e-16)."""

    def __init__(self, mi):
        A = mi._build()
        self.mi = mi
        self.A = A
        pid = np.asarray(mi._ent_pid, np.int64)
        fid = np.asarray(mi._ent_fid, np.int64)
        val = np.asarray(mi._ent_val, np.float64)
        self.n_p, self.F = len(mi._pid_row), len(mi._fnames)
        self.pn = np.asarray(A["norm"], np.float64)
        self.prow = np.asarray(A["prow"], np.int64)
        self.rids = list(A["rix"])
        self.nr = len(self.rids)
        w = val * A["idf"][fid]
        # phrasing rows, features ascending (CSR canonical order), values already unit-normalised
        o = np.lexsort((fid, pid))
        pid_s, fid_s, w_s = pid[o], fid[o], w[o] / np.maximum(self.pn[pid[o]], 1e-12)
        starts = np.searchsorted(pid_s, np.arange(self.n_p + 1))
        self.xf = [fid_s[starts[i]:starts[i + 1]] for i in range(self.n_p)]
        self.xv = [w_s[starts[i]:starts[i + 1]] for i in range(self.n_p)]
        # unit row centroids (Rocchio): the sum of a row's unit phrasings, normalised -- the prototypes' init
        Csum = np.zeros((self.nr, self.F))
        np.add.at(Csum, (self.prow[pid_s], fid_s), w_s)
        cnorm = np.sqrt((Csum * Csum).sum(1))
        self.C0 = Csum / np.maximum(cnorm, 1e-12)[:, None]
        # phrasings grouped by row (for best-wording max)
        self.order = np.argsort(self.prow, kind="stable")
        self.rstarts = np.searchsorted(self.prow[self.order], np.arange(self.nr + 1))

    def queries(self, texts):
        """-> list of (fids, unit values) per question."""
        out = []
        for t in texts:
            q, qn = self.mi._query_vec(t)
            f = np.fromiter(q.keys(), np.int64, len(q))
            v = np.fromiter(q.values(), np.float64, len(q)) / max(qn, 1e-12)
            out.append((f, v))
        return out

    def phrase_cos(self, qf, qv):
        """cos(q, every phrasing) via the feature postings (MeaningIndex.rank's inner loop)."""
        A = self.A
        st, P_, W_ = A["starts"], A["p_pid"], A["p_w"]
        if len(qf) == 0:
            return np.zeros(self.n_p)
        seg = [(st[f], st[f + 1]) for f in qf]
        idx = np.concatenate([P_[a:b] for a, b in seg])
        wts = np.concatenate([W_[a:b] * v for (a, b), v in zip(seg, qv)])
        return np.bincount(idx, weights=wts, minlength=self.n_p) / np.maximum(self.pn, 1e-12)

    def best_and_cen(self, Q):
        """-> (BEST: max over each row's phrasings, clipped at 0; per-row lists of phrasing cosines on demand)."""
        B = np.zeros((len(Q), self.nr))
        for i, (f, v) in enumerate(Q):
            s = self.phrase_cos(f, v)[self.order]
            B[i] = np.maximum(np.maximum.reduceat(s, self.rstarts[:-1]), 0.0)
        return B

    @staticmethod
    def dot(Q, PT):
        """(n_q x nr) = Q @ PT for sparse unit queries and a dense F x nr prototype matrix."""
        out = np.zeros((len(Q), PT.shape[1]))
        for i, (f, v) in enumerate(Q):
            if len(f):
                out[i] = v @ PT[f]
        return out


def meaning_train(space, rule, tau, eta, epochs, seed):
    """w4 p2b's train(), verbatim: prototypes start at the unit centroids and learn from the index's OWN wordings
    (row labels), shuffled per epoch. infonce: P_k += eta/tau * x * (1[k=y] - softmax(cos/tau)_k); adapthd: miss-only
    +-eta. Norms tracked exactly. -> F x nr unit prototypes (transposed)."""
    PT = np.ascontiguousarray(space.C0.T)
    nrm2 = (PT * PT).sum(0)
    rng = np.random.default_rng(seed)
    nr = space.nr
    for _ in range(epochs):
        for i in rng.permutation(space.n_p):
            f, v, y = space.xf[i], space.xv[i], space.prow[i]
            if len(f) == 0:
                continue
            blk = PT[f]
            cos = (v @ blk) / np.sqrt(np.maximum(nrm2, 1e-12))
            if rule == "adapthd":
                yh = int(np.argmax(cos))
                if yh == y:
                    continue
                g = np.zeros(nr)
                g[y] = eta
                g[yh] = -eta
            else:
                z = cos / tau
                z -= z.max()
                p = np.exp(z)
                p /= p.sum()
                g = -p
                g[y] += 1.0
                g *= eta / tau
            new = blk + np.outer(v, g)
            nrm2 += (new * new).sum(0) - (blk * blk).sum(0)
            PT[f] = new
    return PT / np.sqrt(np.maximum(nrm2, 1e-12))[None, :]


def top2_signal(R):
    """-> (top row, confidence g = s1 + (s1 - s2) with the index's rules: only s > 0 rows are candidates, and a
    question with none gets -9 (below every real score), has-candidate mask)."""
    o = np.argsort(-R, axis=1, kind="stable")[:, :2]
    sv = np.take_along_axis(R, o, axis=1)
    s1, s2 = sv[:, 0], np.where(sv[:, 1] > 0, sv[:, 1], 0.0)
    return o[:, 0], np.where(s1 > 0, 2 * s1 - s2, -9.0), s1 > 0


class LearnedIndex:
    """The sweep-181 learned CLINC150 index (602 rows over 151 intents), loaded exactly as w4's common.load()."""

    def __init__(self, path, clinc):
        from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
        with open(path) as f:
            st = json.load(f)
        self.ri = st["row_intent"]
        self.mi = MeaningIndex.from_state(st["meaning"])
        self.sp = MeaningSpace(self.mi)
        self.row_int = np.array([self.ri[r] for r in self.sp.rids])
        self.clinc = clinc
        self.Q, self.BEST = {}, {}
        for s in ("val", "oos_val", "test", "oos_test"):
            texts = [t for t, _ in clinc[s]] if s in ("val", "test") else clinc[s]
            self.Q[s] = self.sp.queries(texts)
            self.BEST[s] = self.sp.best_and_cen(self.Q[s])
        self.truth = {s: np.array([i for _, i in clinc[s]]) for s in ("val", "test")}

    def readout(self, PTn, s, mode="blend"):
        cos = MeaningSpace.dot(self.Q[s], PTn)
        return cos if mode == "proto" else 0.2 * self.BEST[s] + 0.8 * cos

    def signals(self, PTn, mode="blend"):
        """-> {split: (ok_in or None, g)} for val / oos_val / test / oos_test."""
        out = {}
        for s in ("val", "oos_val", "test", "oos_test"):
            top, g, has = top2_signal(self.readout(PTn, s, mode))
            ok = (has & (self.row_int[top] == self.truth[s])) if s in self.truth else None
            out[s] = (ok, g, top)
        return out


def _protocol_from_signals(sg, clinc, oos_sig=None, test_sig=None):
    """E0.4 layout from LearnedIndex.signals (optionally replacing the test / oos signal, e.g. softmax p_max)."""
    return assemble(sg["val"][1], sg["val"][0], sg["test"][1] if test_sig is None else test_sig, sg["test"][0],
                    sg["oos_test"][1] if oos_sig is None else oos_sig, clinc["oos_test"])


def cmd_learned(args):
    """w4 on the learned index: baseline (0.8413), InfoNCE on its own wordings (0.8616), and the candidate-relative
    softmax trap (AURC 0.080 / 0.086 / 0.094 vs 0.078 for the absolute cosine)."""
    if not args.index:
        return {"skipped": "needs --index PATH: the sweep-181 learned index idx_clinc.json is NOT in the repo "
                           "(scratchpad final/default/idx_clinc.json in the session that built it)"}
    t0 = time.time()
    clinc = load_clinc(args.data)
    LI = LearnedIndex(args.index, clinc)
    sp = LI.sp
    res = {"protocol": "w4 p0/p2b/p3: the learned index scored as MeaningIndex.rank does (0.2 best wording + 0.8 "
                       "centroid); InfoNCE prototypes trained on the index's own 8,591 wordings (tau .05, eta .03, "
                       "4 epochs, blend readout: w4's val-chosen config); AURC under the 4,500:1,000 prior on test "
                       "in-scope + oos_test half B",
           "index": {"rows": sp.nr, "wordings": sp.n_p, "features": sp.F,
                     "answer_keys": len({LI.mi.rows[r].get("akey") for r in sp.rids})}}
    C0T = np.ascontiguousarray(sp.C0.T)
    base = LI.signals(C0T)
    res["base"] = {"top1_val": float(base["val"][0].mean()), "top1_test": float(base["test"][0].mean())}
    ok_t = base["test"][0]
    half_b = ~oos_half_a(clinc["oos_test"])

    def a_(ok, gi, go):
        sig = np.concatenate([gi, go[half_b]])
        okk = np.concatenate([ok, np.zeros(int(half_b.sum()), bool)])
        isin = np.concatenate([np.ones(len(gi), bool), np.zeros(int(half_b.sum()), bool)])
        return aurc(sig, okk, isin)
    # p3's `ok` is the top row's intent == truth (no has-candidate mask): kept as run
    ok3 = lambda sg: LI.row_int[sg["test"][2]] == LI.truth["test"]
    res["aurc"] = {"base": a_(ok3(base), base["test"][1], base["oos_test"][1])}
    PT_inf = {sd: meaning_train(sp, "infonce", 0.05, 0.03, 4, sd) for sd in (0, 1, 2)}
    PT_ad = meaning_train(sp, "adapthd", 1.0, 0.03, 4, 0)
    res["infonce"] = []
    for sd, PTn in PT_inf.items():
        sg = LI.signals(PTn)
        bt = paired_bootstrap(ok_t, sg["test"][0], n=2000, seed=0)
        res["infonce"].append({"seed": sd, "top1_val": float(sg["val"][0].mean()),
                               "top1_test": float(sg["test"][0].mean()), "vs_base": bt})
        if sd == 0:
            sg0 = sg
    sga = LI.signals(PT_ad)
    res["adapthd_eta0.03"] = {"top1_test": float(sga["test"][0].mean())}
    ok0 = ok3(sg0)
    res["aurc"]["infonce_abs"] = a_(ok0, sg0["test"][1], sg0["oos_test"][1])
    res["aurc"]["adapthd"] = a_(ok3(sga), sga["test"][1], sga["oos_test"][1])
    PTn = PT_inf[0]
    for kk in (None, 8, 2):
        pm = []
        for s in ("test", "oos_test"):
            z = MeaningSpace.dot(LI.Q[s], PTn) / 0.05
            if kk is None:
                z -= z.max(1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(1, keepdims=True)
                pm.append(p.max(1))
            else:
                z = -np.sort(-z, axis=1)[:, :kk]
                z -= z[:, :1]
                p = np.exp(z)
                p /= p.sum(1, keepdims=True)
                pm.append(p[:, 0])
        res["aurc"]["infonce_softmax_%s" % ("all" if kk is None else "top%d" % kk)] = a_(ok0, pm[0], pm[1])
    # the same comparison through gate_protocol (calibrated + oracle numbers), for the record
    res["gate_protocol"] = {"base": gate_protocol(*_protocol_from_signals(base, clinc)),
                            "infonce_seed0": gate_protocol(*_protocol_from_signals(sg0, clinc))}
    rows = []
    check(rows, "learned index baseline top-1 (test)", res["base"]["top1_test"], "0.8413")
    check(rows, "learned index baseline top-1 (val)", res["base"]["top1_val"], "0.8473")
    check(rows, "InfoNCE on the learned index, seed 0", res["infonce"][0]["top1_test"], "0.8616")
    check(rows, "InfoNCE vs baseline (points)", 100 * res["infonce"][0]["vs_base"]["diff"], "2.0")
    check(rows, "  CI low (points)", 100 * res["infonce"][0]["vs_base"]["lo"], "1.3")
    check(rows, "  CI high (points)", 100 * res["infonce"][0]["vs_base"]["hi"], "2.7")
    check(rows, "AURC absolute cosine (InfoNCE)", res["aurc"]["infonce_abs"], "0.078", "0.0780")
    check(rows, "AURC softmax p_max, all rows", res["aurc"]["infonce_softmax_all"], "0.080", "0.0797")
    check(rows, "AURC softmax p_max, top 8", res["aurc"]["infonce_softmax_top8"], "0.086", "0.0855",
          note="the panel's 0.086 is 0.0855 (0.08546) rounded twice; the 4-digit probe value is the strict check")
    check(rows, "AURC softmax p_max, top 2", res["aurc"]["infonce_softmax_top2"], "0.094", "0.0942")
    check(rows, "AURC base index", res["aurc"]["base"], "0.0941")
    check(rows, "AURC AdaptHD (eta .03)", res["aurc"]["adapthd"], "0.0911")
    check(rows, "AURC change InfoNCE vs base (%)", 100 * (res["aurc"]["infonce_abs"] / res["aurc"]["base"] - 1),
          "-17")
    res["reproduction"] = rows
    res["configs_tried"] = "1 here (w4's val grid: 4 InfoNCE x 4 epoch checkpoints x 2 readouts, 3 AdaptHD x 4 x 2)"
    res["seconds"] = round(time.time() - t0, 1)
    return res


# =============================================================================================== subcommand: prior-trap (E0.4)
def _oracle_prefix(sig, ok, is_in, P, w=None):
    """p0/p4's (pre-protocol) threshold picker: sort by signal, largest prefix whose (weighted) precision >= P.
    -> (coverage, threshold, oos served). Kept only to REPRODUCE the trap; gate_protocol replaces it."""
    if w is None:
        w = np.ones(len(sig))
    o = np.argsort(-sig, kind="stable")
    c = np.cumsum((ok * w)[o])
    n = np.cumsum(w[o])
    idx = np.where(c / n >= P)[0]
    if not len(idx):
        return 0.0, np.inf, 0.0
    k = idx.max() + 1
    thr = sig[o][k - 1]
    served = sig >= thr
    return (ok & served & is_in).sum() / is_in.sum(), thr, (served & ~is_in).sum() / max((~is_in).sum(), 1)


def cmd_prior_trap(args):
    """w4 p4 + the protocol swing: the same learned index, three ways to pick a P = 0.97 threshold."""
    if not args.index:
        return {"skipped": "needs --index PATH (the learned index is not in the repo)"}
    t0 = time.time()
    clinc = load_clinc(args.data)
    LI = LearnedIndex(args.index, clinc)
    sg = LI.signals(np.ascontiguousarray(LI.sp.C0.T))
    n_v, n_ov, n_t, n_ot = len(clinc["val"]), len(clinc["oos_val"]), len(clinc["test"]), len(clinc["oos_test"])
    sv = np.concatenate([sg["val"][1], sg["oos_val"][1]])
    okv = np.concatenate([sg["val"][0], np.zeros(n_ov, bool)])
    iv = np.concatenate([np.ones(n_v, bool), np.zeros(n_ov, bool)])
    stt = np.concatenate([sg["test"][1], sg["oos_test"][1]])
    okt = np.concatenate([sg["test"][0], np.zeros(n_ot, bool)])
    it = np.concatenate([np.ones(n_t, bool), np.zeros(n_ot, bool)])
    res = {"val_oos_share": n_ov / (n_v + n_ov), "test_oos_share": n_ot / (n_t + n_ot), "P": {}}
    w_oos = (n_ot / n_t) / (n_ov / n_v)
    for P in (0.97, 0.95):
        _, thr_raw, _ = _oracle_prefix(sv, okv, iv, P)
        _, thr_w, _ = _oracle_prefix(sv, okv, iv, P, w=np.where(iv, 1.0, w_oos))
        rr = {}
        for nm, thr in (("unweighted_val", thr_raw), ("prior_reweighted_val", thr_w)):
            s = stt >= thr
            rr[nm] = {"thr": float(thr), "coverage": float((okt & s & it).sum() / it.sum()),
                      "precision_raw_mix": float((okt & s).sum() / max(s.sum(), 1)),
                      "oos_served": float((s & ~it).sum() / (~it).sum())}
        gp = gate_protocol(*_protocol_from_signals(sg, clinc), targets=(P,))
        rr["e0.4_protocol"] = gp["calibrated"]["P%.2f" % P]
        res["P"]["%.2f" % P] = rr
    rows = []
    r97, r95 = res["P"]["0.97"], res["P"]["0.95"]
    check(rows, "P0.97 coverage, threshold on prior-reweighted val (%)", r97["prior_reweighted_val"]["coverage"],
          "37.7", scale=100)
    check(rows, "P0.97 coverage, E0.4 protocol (val + oos half A) (%)", r97["e0.4_protocol"]["coverage"], "3.3",
          scale=100)
    check(rows, "P0.97 E0.4 realised precision", r97["e0.4_protocol"]["precision"], "0.993")
    check(rows, "P0.97 coverage, threshold on UNweighted val (%)", r97["unweighted_val"]["coverage"], "40.4",
          scale=100)
    check(rows, "P0.97 realised precision, UNweighted val", r97["unweighted_val"]["precision_raw_mix"], "0.967")
    check(rows, "P0.95 coverage, prior-reweighted val (%)", r95["prior_reweighted_val"]["coverage"], "56.9",
          scale=100)
    check(rows, "P0.95 coverage, E0.4 protocol (%)", r95["e0.4_protocol"]["coverage"], "45.5", scale=100)
    check(rows, "val out-of-scope share (%)", res["val_oos_share"], "3.2", scale=100)
    check(rows, "test out-of-scope share (%)", res["test_oos_share"], "18", scale=100)
    res["reproduction"] = rows
    res["lesson"] = ("coverage at EXACTLY one precision is a knife-edge statistic: the same index reads 37.7% or 3.3% "
                     "at P=0.97 depending on which 100-odd out-of-scope questions calibrate it. Report AURC and "
                     "precision at fixed served fractions first; a calibrated operating point only with its realised "
                     "precision, out-of-scope served and a paired CI (gate_protocol).")
    res["configs_tried"] = 3
    res["seconds"] = round(time.time() - t0, 1)
    return res



# =============================================================================================== subcommand: gate (E0.4 applied)
GATE_ARMS = ("infonce_t0.05_lr0.3", "adapthd_lr0.3", "adapthd_lr1")


def _gate_signals(P, H):
    """Per-question signals for a prototype matrix: top-1 index, g = s1 + (s1 - s2) (absolute cosine plus its lead,
    the meaning index's confidence form) and p_top at tau 0.02 (the candidate-relative feature ProtoStore feeds a
    DoorCalibrator)."""
    S = H @ P.T
    srt = -np.sort(-S, axis=1)
    z = (S - srt[:, :1]) / 0.02
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return np.argmax(S, axis=1), 2 * srt[:, 0] - srt[:, 1], p.max(1)


def gate_dataset(name, clinc, bank, cache, seeds=(0, 1, 2), n_boot=1000):
    """The E0.4 protocol on the w1 one-pass arms for one dataset. CLINC150 calibrates on its own val; Banking77 on
    the hash-carved 10% of its train (the stream never sees those wordings). Out-of-scope for BOTH is CLINC150's
    oos_test, split by sha256 parity (for Banking77 that is out-of-DOMAIN, an easier out-of-scope: stated)."""
    if name == "clinc":
        train, val, test, n_stream = clinc["train"], clinc["val"], clinc["test"], 4000
    else:
        train, val = hash_val_split(bank["train"])
        test, n_stream = bank["test"], 3000
    oos = clinc["oos_test"]
    enc = universe(name, clinc, bank, cache)
    Hv, Ht, Ho = enc([t for t, _ in val]), enc([t for t, _ in test]), enc(oos)
    half_a = oos_half_a(oos)
    out = {"n_val": len(val), "n_test": len(test), "n_oos": len(oos), "seeds": {}}
    for seed in seeds:
        labels, init, stream, test_p = w1_protocol(train, test, 5, n_stream, seed)
        L = {l: i for i, l in enumerate(labels)}
        yv = np.array([L[i] for _, i in val])
        yt = np.array([y for _, y in test_p])
        Hi, yi = enc([t for t, _ in init]), np.array([y for _, y in init])
        Hs, ys = enc([t for t, _ in stream]), np.array([y for _, y in stream])
        per = {}
        items = {}
        for arm in GATE_ARMS:
            r = run_arm(ARMS_W1[arm], labels, Hi, yi, Hs, ys, noise=0.0, seed=seed)
            (pv, gv, tv), (pt, gt, tt), (_, go, to) = (_gate_signals(r["P"], Hv), _gate_signals(r["P"], Ht),
                                                        _gate_signals(r["P"], Ho))
            okv, okt = pv == yv, pt == yt
            per[arm] = {}
            for sname, (sv, st_, so) in (("g_lead", (gv, gt, go)), ("ptop_tau0.02", (tv, tt, to))):
                lay = assemble(sv, okv, st_, okt, so, oos)
                gp = gate_protocol(*lay, targets=(0.95, 0.97), max_wrong=(0.015,))
                per[arm][sname] = gp
                rep = ~lay[3]
                thr = calibrate_threshold(lay[0][lay[3]], lay[1][lay[3]], ~lay[2][lay[3]], target=0.95)
                items[(arm, sname)] = {"sig": lay[0][rep], "ok": lay[1][rep], "isin": ~lay[2][rep],
                                       "cov95": okt & (st_ >= thr)}
            items[(arm, "ok")] = okt
        # paired comparisons, InfoNCE (b) against each AdaptHD arm (a), on the SAME report questions
        cmp_ = {}
        for base in ("adapthd_lr0.3", "adapthd_lr1"):
            c = {"top1": paired_bootstrap(items[(base, "ok")], items[("infonce_t0.05_lr0.3", "ok")], n=n_boot)}
            for sname in ("g_lead", "ptop_tau0.02"):
                A, B = items[(base, sname)], items[("infonce_t0.05_lr0.3", sname)]
                c["coverage_P0.95_" + sname] = paired_bootstrap(A["cov95"], B["cov95"], n=n_boot)
                c["aurc_" + sname] = paired_bootstrap_stat(
                    lambda idx, A=A: aurc(A["sig"][idx], A["ok"][idx], A["isin"][idx]),
                    lambda idx, B=B: aurc(B["sig"][idx], B["ok"][idx], B["isin"][idx]), len(A["sig"]), n=n_boot)
            cmp_["infonce_vs_" + base] = c
        out["seeds"][str(seed)] = {"arms": per, "paired": cmp_}
        print("gate %s seed %d: top1 %s  aurc(g) %s" % (
            name, seed, {a: round(per[a]["g_lead"]["top1"], 4) for a in GATE_ARMS},
            {a: round(per[a]["g_lead"]["aurc"], 4) for a in GATE_ARMS}), flush=True)
    return out


def _seed_summary(runs, path):
    """mean / min / max over seeds of a nested value (path = list of keys under each seed)."""
    vals = []
    for sd in runs["seeds"].values():
        v = sd
        for k in path:
            v = v[k]
        vals.append(v)
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)),
            "per_seed": [float(x) for x in vals]}


def cmd_gate(args):
    """E0.4 applied: the one-pass arms through gate_protocol, 3 seeds, both datasets, paired bootstrap CIs."""
    t0 = time.time()
    clinc, bank = load_clinc(args.data), load_banking(args.data)
    res = {"protocol": ("E0.4: calibrate on val in-scope + sha256 half A of CLINC oos_test; report on test in-scope + "
                        "half B; precision under the 4,500:1,000 prior; AURC; precision at 20-60% served; realised "
                        "precision / coverage / wrong / out-of-scope at thresholds calibrated for P 0.95 / 0.97 and "
                        "for <= 1.5% served-wrong; oracle thresholds only as a symmetric diagnostic; 3 seeds; paired "
                        "bootstrap (1,000 resamples, seed 0) InfoNCE minus AdaptHD on the same report questions"),
           "configs": {"arms": list(GATE_ARMS), "signals": ["g_lead", "ptop_tau0.02"],
                       "datasets": ["clinc", "banking"]},
           "configs_tried": len(GATE_ARMS) * 2 * 2, "datasets": {}}
    for name in ("clinc", "banking"):
        runs = gate_dataset(name, clinc, bank, args.cache)
        summ = {}
        for arm in GATE_ARMS:
            for sname in ("g_lead", "ptop_tau0.02"):
                summ["%s/%s" % (arm, sname)] = {
                    "top1": _seed_summary(runs, ["arms", arm, sname, "top1"]),
                    "aurc": _seed_summary(runs, ["arms", arm, sname, "aurc"]),
                    "cov_P0.95": _seed_summary(runs, ["arms", arm, sname, "calibrated", "P0.95", "coverage"]),
                    "prec_P0.95": _seed_summary(runs, ["arms", arm, sname, "calibrated", "P0.95", "precision"]),
                    "oos_P0.95": _seed_summary(runs, ["arms", arm, sname, "calibrated", "P0.95", "oos_served"]),
                    "cov_W1.5%": _seed_summary(runs, ["arms", arm, sname, "calibrated", "W0.015", "coverage"])}
        runs["summary"] = summ
        res["datasets"][name] = runs
    res["seconds"] = round(time.time() - t0, 1)
    return res


# =============================================================================================== subcommand: negq (E0.3)
def _hm(h, s):
    """Harmonic mean of hardness and safety (ECI's aggregation: a deficiency in EITHER sinks the quality)."""
    h, s = np.clip(h, 0.0, 1.0), np.clip(s, 0.0, 1.0)
    return np.where(h + s > 0, 2 * h * s / np.maximum(h + s, 1e-300), 0.0)


NEGQ_VARIANTS = {
    # name: (what safety measures, one line)  -- every variant is reported, none is dropped for losing
    "draft": "1 - max cos(q, the negative row A's own confirmed wordings) (holographic_protostore.negative_quality)",
    "eci_wording_margin": "ECI Delta_max: max cos(q, wordings of the teacher's pick B) - max cos(q, wordings of A); "
                          "a 'new' verdict's positive is q itself (cos 1)",
    "prototype_margin": "cos(q, prototype of B) - cos(q, prototype of A) ('new' -> B = q)",
    "pick_rank": "1 - (rank of B - 2) / 6: genuine corrections concentrate on the runner-up, noise spreads over "
                 "ranks 2..8 ('new' -> 1)",
}


def _negatives(recs, feats, eps, seed, mode="uniform"):
    """Run the scripted teacher over ranked records; every verdict that REJECTS the top candidate A is a negative
    'q is not A'. planted = A was actually right (a teacher error). -> list of dicts with the variant features."""
    rng = random.Random(seed + 7)
    out = []
    for r, f in zip(recs, feats):
        pick, _ = scripted_teacher(r, eps, rng, mode)
        if pick == 1 or not r["cands"]:
            continue
        b = pick - 1 if pick >= 2 else None
        out.append({"planted": r["cands"][0] == r["want"], "h": f["proto"][0], "mA": f["best"][0],
                    "mB": f["best"][b] if b is not None else 1.0, "pA": f["proto"][0],
                    "pB": f["proto"][b] if b is not None else 1.0, "rank": pick if b is not None else 0,
                    "wA": f["wA"]})
    return out


def _variant_scores(negs, variant):
    """-> (safety raw, quality) arrays; quality = HM(hardness, clip(safety)), the thing a store would rank by."""
    from holographic.agents_and_reasoning.holographic_protostore import negative_quality
    h = np.array([n["h"] for n in negs])
    if variant == "draft":
        q = [negative_quality(n["h"], n["wA"]) for n in negs]
        return np.array([x["safety"] for x in q]), np.array([x["quality"] for x in q])
    if variant == "eci_wording_margin":
        s = np.array([n["mB"] - n["mA"] for n in negs])
    elif variant == "prototype_margin":
        s = np.array([n["pB"] - n["pA"] for n in negs])
    elif variant == "pick_rank":
        s = np.array([1.0 if n["rank"] == 0 else 1.0 - (n["rank"] - 2) / 6.0 for n in negs])
    else:
        raise ValueError(variant)
    return s, _hm(h, s)


def _flag_threshold(genuine_scores, max_false=0.01):
    """Flag a negative when its score is BELOW t; t chosen so at most max_false of GENUINE negatives are flagged."""
    v = np.sort(np.asarray(genuine_scores, np.float64))
    k = int(np.floor(max_false * len(v)))
    return float(v[k]) if len(v) else -np.inf


NEGQ_TEACHERS = (("0.1", 0.1, "uniform"), ("0.2", 0.2, "uniform"), ("nearmiss_0.1", 0.1, "nearmiss"))


def negq_space(space_name, recs_cal, feats_cal, recs_rep, feats_rep, n_boot=1000):
    """Every variant, two scores each (quality and raw safety), on one feature space.
    A variant PASSES only with AUROC >= 0.8 (uniform 10% noise) AND <= 1% false flags (perfect teacher) at the
    val-calibrated threshold AND a threshold that actually flags something (planted recall > 0): a flag that never
    fires has zero false flags by construction, and the clipped variants produce exactly that (kept loud)."""
    out = {}
    cal_g = _negatives(recs_cal, feats_cal, 0.0, 0)
    rep_g = _negatives(recs_rep, feats_rep, 0.0, 0)
    rep_n = {nm: _negatives(recs_rep, feats_rep, e, 0, mode) for nm, e, mode in NEGQ_TEACHERS}
    for v in NEGQ_VARIANTS:
        for which in ("quality", "safety"):
            k = 1 if which == "quality" else 0
            thr = _flag_threshold(_variant_scores(cal_g, v)[k])
            ff = float((_variant_scores(rep_g, v)[k] < thr).mean())
            r = {"threshold_from_val": thr, "false_flags_perfect_teacher": ff,
                 "n_genuine_perfect": len(rep_g), "noisy": {}}
            for e, negs in rep_n.items():
                sc = _variant_scores(negs, v)[k]
                pl = np.array([n["planted"] for n in negs])
                au = auroc(-sc, pl)
                ci = paired_bootstrap_stat(lambda idx: 0.0, lambda idx, sc=sc, pl=pl: auroc(-sc[idx], pl[idx]),
                                           len(sc), n=n_boot)
                r["noisy"][e] = {"auroc": au, "auroc_ci": [ci["lo"], ci["hi"]], "n_planted": int(pl.sum()),
                                          "n_genuine": int((~pl).sum()),
                                          "planted_recall_at_thr": float((sc[pl] < thr).mean()) if pl.any() else None,
                                          "genuine_flagged_at_thr": float((sc[~pl] < thr).mean())}
            r["degenerate_threshold"] = not r["noisy"]["0.1"]["planted_recall_at_thr"]
            r["passes"] = bool(r["noisy"]["0.1"]["auroc"] >= 0.8 and ff <= 0.01 and not r["degenerate_threshold"])
            out["%s/%s" % (v, which)] = r
    if True:   # the draft's own built-in flag (suspect = safety < 0.2), no threshold fitting at all
        s_g = _variant_scores(rep_g, "draft")[0]
        s_n = _variant_scores(rep_n["0.1"], "draft")[0]
        pl = np.array([n["planted"] for n in rep_n["0.1"]])
        out["draft/builtin_suspect_s<0.2"] = {"false_flags_perfect_teacher": float((s_g < 0.2).mean()),
                                              "planted_recall": float((s_n[pl] < 0.2).mean()) if pl.any() else None}
    return out


def _meaning_negq_setup(clinc, K):
    """Meaning-index rows with K confirmed wordings each (learn=True links, as probe_gate), ranked with the repo's
    answers(k=8); per candidate: best-wording cosine and prototype (unit centroid) cosine in the index's own space."""
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    tr = clinc["train"]
    mi = MeaningIndex()
    rint = {}
    for intent in sorted(tr):
        rid = mi.add_row(tr[intent][0], akey=intent)
        rint[rid] = intent
        for t in tr[intent][1:K]:
            mi.link(rid, t, learn=True)
    sp = MeaningSpace(mi)
    rpos = {r: i for i, r in enumerate(sp.rids)}
    C0T = np.ascontiguousarray(sp.C0.T)

    def build(pairs):
        recs, feats = [], []
        for q, want in pairs:
            ranked = mi.answers(q, k=8)
            rids = [r for r, _, _ in ranked]
            recs.append({"q": q, "want": want, "cands": [rint[r] for r in rids]})
            (f, v), = sp.queries([q])
            pc = sp.phrase_cos(f, v)
            proto = (v @ C0T[f]) if len(f) else np.zeros(sp.nr)
            best, wA = [], []
            for j, r in enumerate(rids):
                ri = rpos[r]
                seg = pc[sp.order[sp.rstarts[ri]:sp.rstarts[ri + 1]]]
                best.append(float(seg.max()))
                if j == 0:
                    wA = [float(x) for x in seg]
            feats.append({"best": best, "proto": [float(proto[rpos[r]]) for r in rids], "wA": wA})
        return recs, feats
    return build


def _ngram_negq_setup(clinc, K, cache):
    """The same K wordings as ProtoStore rows in the hashed n-gram space (SystemOne's encoder, d = 2048)."""
    from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
    tr = clinc["train"]
    labels = sorted(tr)
    enc = universe("clinc", clinc, None, cache)
    store = ProtoStore(DIM, name="negq")
    W = {}
    for l in labels:
        W[l] = enc(tr[l][:K])
        store.add_option(l, W[l])

    def build(pairs):
        recs, feats = [], []
        H = enc([q for q, _ in pairs])
        for (q, want), h in zip(pairs, H):
            ranked = store.rank(h, 8)
            cands = [l for l, _ in ranked]
            recs.append({"q": q, "want": want, "cands": cands})
            ws = [W[l] @ h for l in cands]
            feats.append({"best": [float(x.max()) for x in ws], "proto": [float(c) for _, c in ranked],
                          "wA": [float(x) for x in ws[0]]})
        return recs, feats
    return build


def cmd_negq(args, K=10):
    """E0.3: can a negative-quality score flag a teacher's error ('not A' when A is right) without flagging real
    corrections? Rows = CLINC150 intents with K = 10 confirmed wordings; calibration = val + oos_val (perfect
    teacher: the flag threshold allowing 1% false flags); report = test + oos_test, perfect teacher (false flags)
    and 10% / 20% planted noise (AUROC, planted recall). ACCEPTANCE: AUROC >= 0.8 AND <= 1% false flags."""
    t0 = time.time()
    clinc = load_clinc(args.data)
    cal = list(clinc["val"]) + [(t, "oos") for t in clinc["oos_val"]]
    rep = list(clinc["test"]) + [(t, "oos") for t in clinc["oos_test"]]
    res = {"protocol": cmd_negq.__doc__.split("\n", 1)[1].strip(), "variants": NEGQ_VARIANTS,
           "acceptance": "AUROC >= 0.8 at flagging planted errors (10% noise) AND <= 1% false flags, perfect teacher",
           "spaces": {}}
    for space, setup in (("meaning_index", lambda: _meaning_negq_setup(clinc, K)),
                         ("hashed_ngram_protostore", lambda: _ngram_negq_setup(clinc, K, args.cache))):
        build = setup()
        rc, fc = build(cal)
        rr, fr = build(rep)
        res["spaces"][space] = negq_space(space, rc, fc, rr, fr)
        print("negq", space, {k: (round(v["noisy"]["0.1"]["auroc"], 3), round(v["noisy"]["nearmiss_0.1"]["auroc"], 3),
                                  round(v["false_flags_perfect_teacher"], 4), v["passes"])
                              for k, v in res["spaces"][space].items() if "noisy" in v}, flush=True)
    res["configs_tried"] = sum(1 for sp in res["spaces"].values() for k in sp if "noisy" in sp[k])
    res["verdict"] = {}
    for sp_name, sp in res["spaces"].items():
        passing = [k for k, v in sp.items() if v.get("passes")]
        res["verdict"][sp_name] = ("PASS: " + ", ".join(passing)) if passing else \
            "KEPT NEGATIVE: no variant reaches AUROC >= 0.8 with <= 1% false flags at a threshold that fires"
    res["verdict"]["draft_as_specified"] = (
        "draft/quality (HM of hardness and 1 - best cosine to A's wordings) is a KEPT NEGATIVE whenever its AUROC "
        "is below 0.5: hardness REWARDS the teacher's errors -- a false negative is the hardest negative there is")
    res["seconds"] = round(time.time() - t0, 1)
    return res


# =============================================================================================== subcommand: miner (E1.2)
def _pair_rank(M, C):
    """Unordered pairs (a < b) ranked by a symmetric score matrix, highest first; ties by pair index (stable)."""
    iu = np.triu_indices(C, 1)
    sc = (M + M.T)[iu]
    o = np.argsort(-sc, kind="stable")
    return iu[0][o] * C + iu[1][o], sc[o]


def cmd_miner(args, seeds=(0, 1, 2), Ks=(20, 50, 100), n_boot=1000):
    """E1.2: do the pairs ProtoStore mines from TRAIN verdicts (M[y][k] += p_k) predict the pairs the deployed
    prototypes actually confuse on TEST better than a centroid-cosine miner? precision@K against the test-confused
    unordered pairs (>= 1 and >= 2 confusions), 3 seeds, paired bootstrap over test questions."""
    t0 = time.time()
    clinc, bank = load_clinc(args.data), load_banking(args.data)
    res = {"protocol": cmd_miner.__doc__.split("\n", 1)[1].strip(),
           "miners": {"protostore_M": "ProtoStore.confusion mass, M[a][b] + M[b][a] (the E1.2 miner)",
                      "centroid_cosine": "cosine between unit centroids of every TRAIN wording the store saw (true "
                                         "labels) -- the baseline named in the acceptance",
                      "prototype_cosine": "cosine between the final learned prototypes",
                      "stream_errors": "count of prequential misses (y, pred) during the stream; ties by pair id"},
           "datasets": {}}
    for name, ds, n_stream in (("banking", bank, 3000), ("clinc", clinc, 4000)):
        enc = universe(name, clinc, bank, args.cache)
        per = {}
        for seed in seeds:
            labels, Hi, yi, Hs, ys, Ht, yt = _prep(ds, enc, 5, n_stream, seed)
            C = len(labels)
            r = run_arm(ARMS_W1["infonce_t0.05_lr0.3"], labels, Hi, yi, Hs, ys, seed=seed)
            st = r["store"]
            M = np.zeros((C, C))
            for y, row in st.confusion.items():
                for o_, m in row.items():
                    M[st.index(y), st.index(o_)] += m
            cen = np.zeros((C, len(Hi[0])))
            np.add.at(cen, np.concatenate([yi, ys]), np.vstack([Hi, Hs]))
            cen /= np.linalg.norm(cen, axis=1, keepdims=True)
            E = np.zeros((C, C))
            miss = r["preds"] != ys
            np.add.at(E, (ys[miss], r["preds"][miss]), 1.0)
            miners = {"protostore_M": _pair_rank(M, C)[0], "centroid_cosine": _pair_rank(cen @ cen.T, C)[0],
                      "prototype_cosine": _pair_rank(r["P"] @ r["P"].T, C)[0], "stream_errors": _pair_rank(E, C)[0]}
            pred = np.argmax(Ht @ r["P"].T, axis=1)
            a, b = np.minimum(pred, yt), np.maximum(pred, yt)
            err_pair = np.where(pred != yt, a * C + b, -1)

            def prec_at(order, idx, K, need):
                ep = err_pair[idx]
                cnt = np.bincount(ep[ep >= 0], minlength=C * C)
                return float((cnt[order[:K]] >= need).mean())
            full = np.arange(len(yt))
            cnt_all = np.bincount(err_pair[err_pair >= 0], minlength=C * C)
            out = {"test_top1": float((pred == yt).mean()), "n_pairs": C * (C - 1) // 2,
                   "confused_pairs": {"ge1": int((cnt_all >= 1).sum()), "ge2": int((cnt_all >= 2).sum())},
                   "precision_at_K": {}, "paired_vs_centroid": {}}
            for need in (1, 2):
                out["chance_ge%d" % need] = out["confused_pairs"]["ge%d" % need] / out["n_pairs"]
                for K in Ks:
                    for mn, order in miners.items():
                        out["precision_at_K"]["%s/K%d/ge%d" % (mn, K, need)] = prec_at(order, full, K, need)
                    out["paired_vs_centroid"]["K%d/ge%d" % (K, need)] = paired_bootstrap_stat(
                        lambda idx, K=K, need=need: prec_at(miners["centroid_cosine"], idx, K, need),
                        lambda idx, K=K, need=need: prec_at(miners["protostore_M"], idx, K, need),
                        len(yt), n=n_boot)
            per[str(seed)] = out
            print("miner %s seed %d: P@50(ge1) M %.2f cen %.2f proto %.2f err %.2f" % (
                name, seed, *[out["precision_at_K"]["%s/K50/ge1" % m] for m in miners]), flush=True)
        wins = {k: sum(1 for s in per.values() if s["paired_vs_centroid"][k]["lo"] > 0) for k in
                per[str(seeds[0])]["paired_vs_centroid"]}
        mean_p = {k: float(np.mean([s["precision_at_K"][k] for s in per.values()]))
                  for k in per[str(seeds[0])]["precision_at_K"]}
        res["datasets"][name] = {"seeds": per, "mean_precision_at_K": mean_p,
                                 "seeds_where_M_beats_centroid_CI_above_0": wins}
    res["configs_tried"] = 4
    res["seconds"] = round(time.time() - t0, 1)
    return res


# =============================================================================================== subcommand: curriculum (E2.2)
def cmd_curriculum(args, seeds=(0, 1, 2), n_boot=1000):
    """E2.2: ProtoStore.update(curriculum_k0=3) against the plain rule on the Banking77 one-pass protocol, K = 1
    (where the panel saw +0.4..+0.7) and K = 5 seed wordings, clean and 10% noise, 3 seeds, paired CIs. Plus a
    DIAGNOSTIC arm whose row maturity counts the seeding wordings (the probe's `nrow`; ProtoStore counts verdicts
    only, so its rows all start 'young')."""
    t0 = time.time()
    bank, clinc = load_banking(args.data), load_clinc(args.data)
    enc = universe("banking", clinc, bank, args.cache)
    arms = {"plain": {"rule": "infonce", "tau": 0.05, "lr": 0.3},
            "curriculum_k0=3": {"rule": "infonce", "tau": 0.05, "lr": 0.3, "curriculum_k0": 3},
            "curriculum_k0=3_seedcount": {"rule": "infonce", "tau": 0.05, "lr": 0.3, "curriculum_k0": 3,
                                          "seed_count": True}}
    # under teacher noise the fair rival is not only the plain rule but the PRINCIPLED fix, the noise-corrected
    # target (eps = the true flip rate, E2.3): does the curriculum add anything once the target is corrected?
    noisy_arms = {"eps=0.1": {"rule": "infonce", "tau": 0.05, "lr": 0.3, "eps": 0.1},
                  "curriculum_k0=3+eps=0.1": {"rule": "infonce", "tau": 0.05, "lr": 0.3, "eps": 0.1,
                                              "curriculum_k0": 3}}
    res = {"protocol": cmd_curriculum.__doc__.split("\n", 1)[1].strip(), "settings": {}}
    rows = []
    for K in (1, 5):
        for noise in (0.0, 0.1):
            key = "K%d_noise%.1f" % (K, noise)
            st = {"seeds": {}}
            arms_here = dict(arms, **(noisy_arms if noise > 0 else {}))
            for seed in seeds:
                labels, Hi, yi, Hs, ys, Ht, yt = _prep(bank, enc, K, 3000, seed)
                oks = {}
                for an, arm in arms_here.items():
                    r = run_arm(arm, labels, Hi, yi, Hs, ys, noise=noise, seed=seed)
                    oks[an] = top1_eval(r["P"], Ht, yt)["ok"]
                st["seeds"][str(seed)] = {an: float(o.mean()) for an, o in oks.items()}
                st["seeds"][str(seed)]["paired"] = {an: paired_bootstrap(oks["plain"], oks[an], n=n_boot)
                                                    for an in arms_here if an != "plain"}
                if noise > 0:
                    st["seeds"][str(seed)]["paired"]["curriculum_k0=3+eps=0.1_vs_eps=0.1"] = paired_bootstrap(
                        oks["eps=0.1"], oks["curriculum_k0=3+eps=0.1"], n=n_boot)
            plain = [st["seeds"][str(s)]["plain"] for s in seeds]
            st["plain_seed_spread_points"] = 100 * (max(plain) - min(plain))
            for an in list(arms_here) + (["curriculum_k0=3+eps=0.1_vs_eps=0.1"] if noise > 0 else []):
                if an == "plain":
                    continue
                d = [st["seeds"][str(s)]["paired"][an]["diff"] for s in seeds]
                st[an + "_mean_diff_points"] = 100 * float(np.mean(d))
                st[an + "_seeds_CI_excludes_0"] = sum(1 for s in seeds if st["seeds"][str(s)]["paired"][an]["lo"] > 0
                                                      or st["seeds"][str(s)]["paired"][an]["hi"] < 0)
                st[an + "_beats_seed_spread"] = bool(abs(np.mean(d)) * 100 > st["plain_seed_spread_points"])
            res["settings"][key] = st
            print("curriculum", key, {an: round(100 * float(np.mean([st["seeds"][str(s)][an] for s in seeds])), 2)
                                      for an in arms_here}, flush=True)
    s0 = res["settings"]
    check(rows, "ProtoStore InfoNCE, Banking77 K=1 seed 0, clean", s0["K1_noise0.0"]["seeds"]["0"]["plain"], "0.8114",
          note="banking_k1_curr.json infonce_t0.05_lr0.3 (the curriculum probe's baseline)")
    check(rows, "ProtoStore InfoNCE, Banking77 K=1 seed 0, 10% noise", s0["K1_noise0.1"]["seeds"]["0"]["plain"],
          "0.7695")
    res["reproduction"] = rows
    res["configs_tried"] = len(arms) - 1 + len(noisy_arms)
    res["seconds"] = round(time.time() - t0, 1)
    return res


# =============================================================================================== subcommand: systemone (hook)
# ---- SYSTEMONE HOOK ------------------------------------------------------------------------------------------------
# SystemOne(scorer="contrastive") (E1.1's first consumer) is being added by another worker. Until it exists this
# subcommand prints "not available yet" and records that; once it exists the SAME subcommand measures it on the w1
# Banking77 protocol, no edit needed here. If its constructor or fit/observe/decide signature differs from the
# existing scorers', adapt cmd_systemone below -- it is the only place that calls it.
# --------------------------------------------------------------------------------------------------------------------
def systemone_contrastive_available():
    """True once SystemOne accepts scorer="contrastive" (being added by another worker). Probed, never assumed."""
    try:
        from holographic.agents_and_reasoning.holographic_systemone import SystemOne
        SystemOne(lambda t: np.zeros(8), scorer="contrastive")
        return True
    except Exception:
        return False


def cmd_systemone(args):
    """SYSTEMONE HOOK -- E1.1's first consumer: SystemOne(scorer="contrastive") on the Banking77 one-pass protocol (fit on K = 5 examples
    per intent, observe the 3,000-verdict stream, decide the test set). ACCEPTANCE: top-1 >= 0.80 against measured
    AdaptHD 0.725 / 0.738 (lr 0.3). Prints 'not available yet' until the scorer exists.

    Measured THROUGH SystemOne itself (fit / observe / decide -- the mind's typed door), never a bench-side copy. The
    baseline arm is the SAME class with scorer="prototype" and observe(lr=0.3): SystemOne's own miss-only AdaptHD, which
    must land on the panel's AdaptHD lr 0.3 numbers (0.725 / 0.738) -- if it does not, the harness is wrong, not the
    scorer. Also reported per arm: the AUROC of the calibrator's feature for correctness on the test set (p_top at tau
    0.02 for the contrastive scorer, the margin gap for the prototype scorer) -- the reason the contrastive scorer's
    calibrator is fed p_top."""
    if not systemone_contrastive_available():
        print("systemone: SystemOne(scorer='contrastive') not available yet", flush=True)
        return {"status": "not available yet", "acceptance": "Banking77 one-pass top-1 >= 0.80 (seeds 0, 1)"}
    from holographic.agents_and_reasoning.holographic_systemone import SystemOne
    t0 = time.time()
    bank, clinc = load_banking(args.data), load_clinc(args.data)
    enc = universe("banking", clinc, bank, args.cache)
    out = {"status": "measured", "protocol": "w1: K=5 examples/intent (file order), 3,000-verdict shuffled one-pass "
                                             "stream through SystemOne.observe (decide first, then learn), test top-1 "
                                             "with SystemOne.decide; encoder = the harness's cached hashed char "
                                             "3-5-grams d=2048 (SystemOne.hashed_ngram_encode's arithmetic)",
           "acceptance": "contrastive top-1 >= 0.80 on seeds 0 and 1", "seeds": {}}
    arms = (("contrastive", {"scorer": "contrastive"}, 1.0), ("prototype_adapthd_lr0.3", {"scorer": "prototype"}, 0.3))
    for seed in (0, 1):
        labels, init, stream, test = w1_protocol(bank["train"], bank["test"], 5, 3000, seed)
        lookup = lambda text: enc.H[enc.row[text]]
        ex = {}
        for t, y in init:
            ex.setdefault(labels[y], []).append(t)
        row = {}
        for arm, kw, lr in arms:
            ta = time.time()
            s1 = SystemOne(lookup, **kw)
            s1.fit({"intent": {"type": "choice", "options": labels, "examples": ex}})
            preq = [s1.observe(t, {"intent": labels[y]}, lr=lr)["intent"]["was_correct"] for t, y in stream]
            dec = [s1.decide(t)["intent"] for t, _ in test]
            ok = np.array([d["ranked"][0][0] == labels[y] for d, (_, y) in zip(dec, test)])
            feat = np.array([d.get("p_top", d["margin_gap"]) for d in dec])
            row[arm] = {"top1": round(float(ok.mean()), 4), "prequential": round(float(np.mean(preq)), 4),
                        "auroc_cal_feature": round(auroc(feat, ok), 4),
                        "cal_feature": "p_top_tau0.02" if kw["scorer"] == "contrastive" else "margin_gap",
                        "seconds": round(time.time() - ta, 1)}
            print("systemone seed %d %-24s top1 %.4f  (%.0fs)" % (seed, arm, row[arm]["top1"], time.time() - t0),
                  flush=True)
        out["seeds"][str(seed)] = row
    out["passes"] = all(v["contrastive"]["top1"] >= 0.80 for v in out["seeds"].values())
    out["configs_tried"] = 1
    out["seconds"] = round(time.time() - t0, 1)
    return out


# =============================================================================================== main
COMMANDS = {"repro": cmd_repro, "teacher": cmd_teacher, "learned": cmd_learned, "prior-trap": cmd_prior_trap,
            "gate": cmd_gate, "negq": cmd_negq, "miner": cmd_miner, "curriculum": cmd_curriculum,
            "systemone": cmd_systemone}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="folder with clinc150_full.json, banking77_train/test.csv")
    ap.add_argument("--index", default=None, help="the learned meaning index idx_clinc.json (learned, prior-trap)")
    ap.add_argument("--cache", default=None, help="folder for cached encodings (.npy); speeds up re-runs")
    ap.add_argument("--out", default=OUT_JSON, help="output JSON (sections are merged)")
    ap.add_argument("cmd", choices=sorted(COMMANDS) + ["all"])
    args = ap.parse_args(argv)
    names = [c for c in COMMANDS] if args.cmd == "all" else [args.cmd]
    for name in names:
        key = name.replace("-", "_")
        t0 = time.time()
        payload = section_header(args, key)
        payload.update(COMMANDS[name](args))
        if "skipped" in payload and os.path.exists(args.out):
            with open(args.out) as f:
                if key in json.load(f).get("sections", {}):
                    # a run without --index must not erase a measured learned-index section
                    print("[%s] skipped (%s); the existing section is kept" % (name, payload["skipped"]))
                    continue
        write_section(args.out, key, payload)
        printj({key: {k: v for k, v in payload.items() if k in ("reproduction", "seconds", "skipped")}})
        print("[%s] done in %.0fs" % (name, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
