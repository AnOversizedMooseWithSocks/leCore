"""systemone_bench.py -- the honest measurement for the typed-decision door (sweep 171).

WHAT THIS MEASURES AND WHY THESE BASELINES
------------------------------------------
Task: SST-2 binary sentiment (real labeled data, fetched from the HuggingFace datasets-server
with stdlib urllib; cached to tools/_sst2_cache.json so reruns are offline and deterministic).
A vendor decision-model claim ("Jev, ~68% on our own workflows") is not reproducible here, so we
do NOT compare against it -- we compare against the strongest honest baselines in the original
space, per the baseline discipline:

  majority        always predict the train-majority label (the floor every paper must beat)
  token-overlap   Jaccard of state tokens vs the SAME k exemplars leCore sees, argmax
                  (the cheap classical move; if the substrate cannot beat this, say so loudly)
  labels-only     leCore systemone with NO examples (prototype = the encoded label word) --
                  expected to be poor; kept in the table so nobody mistakes zero-shot for magic
  few-shot k      leCore systemone with k exemplars/class + isotonic calibration on a separate
                  labeled pool; reports accuracy, coverage (abstentions are a RESULT, so we score
                  both accuracy-if-forced and accuracy-on-answered), Brier and ECE.

Variance: mind.measure (the existing harness) across exemplar-sampling seeds -- mean, spread,
bootstrap CI. The eval slice is FIXED so the spread isolates exemplar luck.

Run:  PYTHONHASHSEED=0 python3 tools/systemone_bench.py [--n-eval 200] [--seeds 5] [--dim 512]
No network and no cache -> the script REFUSES with instructions rather than inventing data.
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sst2_cache.json")
_BASE = ("https://datasets-server.huggingface.co/rows?dataset=stanfordnlp%%2Fsst2"
         "&config=default&split=%s&offset=%d&length=100")


def _fetch_split(split, total):
    rows = []
    for off in range(0, total, 100):
        with urllib.request.urlopen(_BASE % (split, off), timeout=60) as r:
            rows += [(x["row"]["sentence"].strip(), int(x["row"]["label"]))
                     for x in json.load(r)["rows"]]
    return rows


def load_sst2(n_train=1200, n_eval=400):
    """Cache-first loader. The cache makes the benchmark a fixed artifact: same rows, same
    numbers, forever -- determinism instead of storage-of-trust in a remote service."""
    if os.path.exists(_CACHE):
        d = json.load(open(_CACHE))
        return d["train"], d["eval"]
    try:
        train = _fetch_split("train", n_train)
        ev = _fetch_split("validation", n_eval)
    except Exception as e:
        raise SystemExit("REFUSED: cannot fetch SST-2 (%s) and no cache at %s. Run once with "
                         "network, or copy a cache file in; inventing stand-in data would make "
                         "every number below a lie." % (e, _CACHE))
    json.dump({"train": train, "eval": ev}, open(_CACHE, "w"))
    return train, ev


LABELS = ["negative", "positive"]   # SST-2: 0 = negative, 1 = positive


def sample_fewshot(train, k, seed):
    """k exemplars per class + a disjoint calibration pool of 64, deterministically per seed."""
    rng = np.random.default_rng(seed)
    by = {0: [t for t, y in train if y == 0], 1: [t for t, y in train if y == 1]}
    ex, cal = {}, []
    for y in (0, 1):
        idx = rng.permutation(len(by[y]))
        ex[LABELS[y]] = [by[y][i] for i in idx[:k]]
        cal += [(by[y][i], {"sent": LABELS[y]}) for i in idx[k:k + 32]]
    return ex, cal


def token_overlap_predict(state, class_tokens, majority):
    """Jaccard argmax against per-class exemplar token sets; exact tie falls to majority --
    the baseline always answers (no abstention), which is exactly its handicap and its point."""
    s = set(state.lower().split())
    scores = [len(s & class_tokens[c]) / max(1, len(s | class_tokens[c])) for c in LABELS]
    if scores[0] == scores[1]:
        return majority
    return LABELS[int(np.argmax(scores))]


def run_once(enc, margin, train, ev, k, seed):
    ex, cal = sample_fewshot(train, k, seed)
    from holographic.agents_and_reasoning.holographic_systemone import SystemOne
    so = SystemOne(enc, margin=margin)
    so.fit({"sent": {"type": "choice", "options": LABELS, "examples": ex}})
    so.calibrate(cal)
    answers = so.decide_map([t for t, _ in ev])
    truth = [LABELS[y] for _, y in ev]
    picked = [a["sent"]["ranked"][0][0] for a in answers]           # forced pick (top-1)
    val = [a["sent"]["value"] for a in answers]                     # abstention-aware
    forced = float(np.mean([p == t for p, t in zip(picked, truth)]))
    answered = [i for i, v in enumerate(val) if v is not None]
    cov = len(answered) / len(ev)
    on_ans = (float(np.mean([val[i] == truth[i] for i in answered])) if answered else 0.0)
    ps = np.array([a["sent"]["p"] for a in answers], dtype=np.float64)
    ys = np.array([p == t for p, t in zip(picked, truth)], dtype=np.float64)
    brier = float(np.mean((ps - ys) ** 2))
    edges = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(ps, edges) - 1, 0, 9)
    ece = float(sum((which == b).mean() * abs(ps[which == b].mean() - ys[which == b].mean())
                    for b in range(10) if (which == b).any()))
    # token-overlap on the SAME exemplars (same information budget -- the fair fight)
    ctok = {c: set(" ".join(ex[c]).lower().split()) for c in LABELS}
    maj = LABELS[int(np.mean([y for _, y in train]) >= 0.5)]
    tok = float(np.mean([token_overlap_predict(t, ctok, maj) == tr
                         for (t, _), tr in zip(ev, truth)]))
    return {"forced": forced, "coverage": cov, "on_answered": on_ans,
            "brier": brier, "ece": ece, "token_overlap": tok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--k", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--encoder", choices=["perceive", "ngram"], default="perceive",
                    help="perceive: the mind's text encoder (margin 0.1). ngram: hashed char "
                         "3-5 gram bundle (margin 0.02) -- see hashed_ngram_encode's measured "
                         "numbers and kept negatives.")
    args = ap.parse_args()

    import lecore
    from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode
    mind = lecore.UnifiedMind(dim=args.dim, seed=0)
    memo = {}
    if args.encoder == "ngram":
        enc, margin = hashed_ngram_encode(dim=2048), 0.02
    else:
        raw = lambda t: mind.perceive(t, "text")
        enc, margin = (lambda t: memo.setdefault(t, raw(t))), 0.1
    train, ev = load_sst2()
    ev = ev[:args.n_eval]
    maj = LABELS[int(np.mean([y for _, y in train]) >= 0.5)]
    maj_acc = float(np.mean([LABELS[y] == maj for _, y in ev]))
    print("SST-2 | eval n=%d | train pool=%d | majority baseline: %.3f"
          % (len(ev), len(train), maj_acc))

    # labels-only (zero-shot): one deterministic run, no exemplar sampling to vary over.
    from holographic.agents_and_reasoning.holographic_systemone import SystemOne
    z = SystemOne(enc)
    z.fit({"sent": {"type": "choice", "options": LABELS}})
    zp = [a["sent"]["ranked"][0][0] for a in z.decide_map([t for t, _ in ev])]
    z_acc = float(np.mean([p == LABELS[y] for p, (_, y) in zip(zp, ev)]))
    print("labels-only (zero-shot, forced): %.3f   <- kept loud; a label is one word of evidence"
          % z_acc)

    for k in args.k:
        # mind.measure = the existing variance harness: mean, spread, bootstrap CI across seeds.
        res = mind.measure(lambda seed: run_once(enc, margin, train, ev, k, seed)["forced"],
                           seeds=range(args.seeds))
        one = [run_once(enc, margin, train, ev, k, s) for s in range(args.seeds)]
        mean = lambda key: float(np.mean([r[key] for r in one]))
        print("few-shot k=%-3d forced acc: %s" % (k, _fmt(res)))
        print("             coverage %.3f | on-answered %.3f | brier %.3f | ece %.3f | "
              "token-overlap(same k): %.3f"
              % (mean("coverage"), mean("on_answered"), mean("brier"), mean("ece"),
                 mean("token_overlap")))


def _fmt(res):
    """Render whatever shape the measure harness returns without assuming its keys."""
    if isinstance(res, dict):
        m = res.get("mean"); ci = res.get("ci") or res.get("ci95")
        if m is not None and ci is not None:
            return "%.3f  CI[%.3f, %.3f]" % (m, ci[0], ci[1])
    return str(res)


if __name__ == "__main__":
    main()
