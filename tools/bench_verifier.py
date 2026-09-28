"""tools/bench_verifier.py -- E5.2: do verifier PROTOTYPES rank a swarm's verified outcomes better than the reflex?

WHAT IS MEASURED (CLM backlog E5.2, wave 2)
-------------------------------------------
swarm_step now teaches per-tool SUCCESS / FAILURE prototypes (a two-option ProtoStore 'verify:<tool>' per tool) from
every VERIFIED step -- a passed verify is a success example of its state, a FAILED verify a labelled failure example
(before E5.2 a refused step taught nothing). mind.verify_precheck(state, actions) scores an action as
cos(state, success) - cos(state, failure). The acceptance bar: its AUROC at telling a step that will pass from one
that will fail must beat the CURRENT REFLEX's -- verify_decision (score, forward/backward margins, valid) and the
reflex bridge's confidence -- on recorded runs, and its p_correct (door_calibrator('verify')) must be calibrated.

THE RECORDED RUNS (a labelled log, frozen beside the results)
-------------------------------------------------------------
tools/bench_swarm_reflex.py's frozen workloads (3 seeds, 4 agents x 40 draws, a pool of 60 vague tasks, 2 wordings
each, Zipf 1.1): every draw is SEARCHED with no reflex -- find_capability(q, 16), candidates submitted to the
deterministic judge in router order until one passes -- and every submission is one row (seed, t, agent, task,
wording, state, action, router rank, verified ok / failed): 2,691 rows, 425 passed, sha256 3f1bbbaa31df... The rows
are a pure function of the frozen workloads and each wording's 16 candidates, so the candidate lists are what is
frozen, in docs/research/evidence/bench_verifier_runs.json, and the rows are rebuilt from them by default (--fresh
redraws them from the live catalog: the sweep-179 lesson, a benchmark whose questions move between runs cannot
compare runs).

PROTOCOLS
---------
  (every split is run under SALTS = 5 hash salts and reported as mean +- sd with per-salt wins: one split of ~600
   correlated rows moved the verifier's AUROC by +-0.09 in the probe, so a single split would be an anecdote)
  split_by_state  the headline: a sha256 hash split of the task WORDINGS (70 train / 30 test). Per seed a fresh mind
                  replays the train rows through the REAL path -- swarm_step(verify=judge) -- so the reflex learns
                  what it learns today (passed steps, via decision_outcome) and the verifier learns both outcomes;
                  every test row is then scored by each signal (nothing learned from test rows). A test wording was
                  never seen; its task's OTHER wording may have been (a paraphrase).
  split_by_task   both wordings of a task on the same side: NOVEL tasks only -- the honest ceiling on generalisation.
  prequential     every row in stream order, scored BEFORE its verify runs, then learned through swarm_step.
  utility         the swarm itself (one shared mind, round robin, the reflex answering repeats first as in
                  bench_swarm_reflex): judge calls with candidates tried in router order vs in verify_precheck order
                  (the default strong-evidence rule, and the raw-score order it replaced), with and without the reflex
                  answering repeats first, plus the min_p 0.05 pre-filter. Every accepted step still ran its verify;
                  the pre-check only reorders (or, with min_p, drops -- counted as 'truth dropped').

Run:  PYTHONHASHSEED=0 python tools/bench_verifier.py [--json docs/research/evidence/bench_verifier.json] [--fresh]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ.setdefault("PYTHONHASHSEED", "0")

import bench_swarm_reflex as SWARM  # noqa: E402  (the frozen workloads and their constants: one source of truth)

EVIDENCE = os.path.join(ROOT, "docs", "research", "evidence")
RUNS = os.path.join(EVIDENCE, "bench_verifier_runs.json")
SEEDS = 3
TRAIN_FRACTION = 7          # of 10 hash buckets
SALTS = 5                   # hash-split salts: one split of ~600 correlated rows swings AUROC by +-0.09 (measured)


# --------------------------------------------------------------------------------------------- the run log ----

def _new_mind(truth_of):
    """A fresh mind with the deterministic judge as a mind verb (swarm_step's verify calls verbs by name)."""
    import lecore
    m = lecore.UnifiedMind(dim=SWARM.MIND_DIM, seed=0)
    m.judge_calls = 0

    def judge(state, action):
        m.judge_calls += 1
        return truth_of.get(state) == action
    m.judge = judge
    return m


def build_runs(cands=None):
    """The labelled log: every judged submission of a no-reflex search over each seed's frozen workload. The rows are
    a pure function of the frozen workloads and each wording's candidate list, so only the candidate lists are
    frozen on disk (`cands`, keyed 'seed|wording'); None asks the live catalog (find_capability, k=16)."""
    rows, cands = [], (dict(cands) if cands is not None else {})
    live = not cands
    for seed in range(SEEDS):
        tasks, streams = SWARM.workload(seed)
        m = _new_mind({}) if live else None
        t = 0
        for step in range(SWARM.TASKS_PER_AGENT):
            for a in range(SWARM.N_AGENTS):
                tid, wi = streams[a][step]
                q = tasks[tid]["wordings"][wi]
                key = "%d|%s" % (seed, q)
                if key not in cands:
                    if m is None:
                        raise KeyError("the frozen run log has no candidates for %r (rebuild with --fresh)" % key)
                    cands[key] = [c.name for c in m.find_capability(q, k=SWARM.K_CANDIDATES)]
                for r, name in enumerate(cands[key]):
                    ok = name == tasks[tid]["truth"]
                    rows.append({"seed": seed, "t": t, "agent": a, "tid": tid, "wording": wi, "state": q,
                                 "action": name, "rank": r, "ok": bool(ok)})
                    t += 1
                    if ok:
                        break
    return {"rows": rows, "candidates": cands}


def load_runs(fresh=False):
    """The run log: rebuilt from the frozen candidate lists (bench_verifier_runs.json, 135 KB instead of the 656 KB
    the full rows took), or drawn fresh from the live catalog and frozen."""
    if not fresh and os.path.exists(RUNS):
        with open(RUNS, encoding="utf-8") as f:
            return build_runs(json.load(f)["candidates"])
    d = build_runs()
    with open(RUNS, "w", encoding="utf-8") as f:
        json.dump({"candidates": d["candidates"]}, f, sort_keys=True, separators=(",", ":"))
    return d


# ------------------------------------------------------------------------------------------------- metrics ----

def auroc(score, y):
    """Mann-Whitney AUROC with MID-RANKS for ties (many signals tie at 0 on an unseen tool; ties count half)."""
    s = np.asarray(score, float)
    y = np.asarray(y, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    o = np.argsort(s, kind="stable")
    ss = s[o]
    r = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(len(s))
    ranks[o] = r
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def ece(p, y, bins=10):
    """Expected calibration error of p against 0/1 outcomes (equal-width bins); rows with p None are skipped."""
    pairs = [(float(a), float(b)) for a, b in zip(p, y) if a is not None]
    if not pairs:
        return None
    P = np.array([a for a, _ in pairs])
    Y = np.array([b for _, b in pairs])
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = (P >= lo) & ((P < hi) if b < bins - 1 else (P <= hi))
        if sel.any():
            e += sel.mean() * abs(P[sel].mean() - Y[sel].mean())
    return float(e)


def _bucket(key, salt=0):
    return int(hashlib.sha256(("verifier-split:%d:%s" % (salt, key)).encode()).hexdigest()[:8], 16) % 10


# ----------------------------------------------------------------------------------------------- the path ----

def _submit(m, row):
    """One recorded row through the real path: swarm_step with the judge as its verify (a failed verify raises)."""
    try:
        m.swarm_step(row["state"], row["action"], done_when="judge passes", evidence={"candidate": row["action"]},
                     worker="agent%d" % row["agent"],
                     verify={"verb": "judge", "args": {"state": row["state"], "action": row["action"]}}, expect=True)
        return True
    except ValueError:
        return False


def _features(m, row):
    """Every signal for one (state, action) row, read BEFORE anything is learned from it."""
    vd = m.verify_decision(row["state"], row["action"])
    ck = vd["checks"]
    rd = m.reflex_decide(row["state"], key="ngram")
    v = rd.get("value")
    conf = float(rd.get("confidence", 0.0) or 0.0)
    pre = m.verify_precheck(row["state"], [row["action"]])
    return {"reflex:verify_score": 0.0 if vd["score"] is None else float(vd["score"]),
            "reflex:forward_margin": float(ck["forward_margin"]),
            "reflex:backward_margin": float(ck["backward_margin"]),
            "reflex:valid": 1.0 if vd["valid"] else 0.0,
            "reflex:bridge_confidence": conf if v == row["action"] else (-conf if v is not None else 0.0),
            "router:rank": -float(row["rank"]),
            "verifier:score": float(pre["scores"][row["action"]]),
            "verifier:p_correct": pre["p_correct"][row["action"]]}


def run_split(runs, key_of, salt=0):
    """Train on the train rows through swarm_step (per seed, a fresh mind), score the test rows. -> summary."""
    feats, ys, per_seed = {}, [], []
    for seed in range(SEEDS):
        R = [r for r in runs["rows"] if r["seed"] == seed]
        truth = {r["state"]: r["action"] for r in R if r["ok"]}
        m = _new_mind(truth)
        train = [r for r in R if _bucket(key_of(r), salt) < TRAIN_FRACTION]
        test = [r for r in R if _bucket(key_of(r), salt) >= TRAIN_FRACTION]
        for r in train:
            _submit(m, r)
        sf, sy = {}, []
        for r in test:
            for k, v in _features(m, r).items():
                sf.setdefault(k, []).append(v)
            sy.append(r["ok"])
        for k, v in sf.items():
            feats.setdefault(k, []).extend(v)
        ys.extend(sy)
        per_seed.append({"seed": seed, "train_rows": len(train), "test_rows": len(test), "test_pass": int(sum(sy)),
                         "verifier_auroc": auroc(sf["verifier:score"], sy),
                         "reflex_best_auroc": max(auroc(sf[k], sy) or 0.0 for k in sf if k.startswith("reflex:")),
                         "verify_calibration": m.door_calibration_report().get("verify")})
    return _summarise(feats, ys, per_seed)


def _summarise(feats, ys, per_seed=None):
    y = np.array(ys, bool)
    rank = np.array(feats["router:rank"])
    out = {"rows": int(len(y)), "pass": int(y.sum()), "auroc": {}, "auroc_router_tiebreak": {}}
    for k, v in feats.items():
        if k == "verifier:p_correct":
            continue
        out["auroc"][k] = auroc(v, y)
        # the pre-check's ORDER breaks ties by the given (router) order; score the same way for every signal
        out["auroc_router_tiebreak"][k] = auroc(np.array(v, float) + 1e-6 * rank, y)
    reflex = {k: a for k, a in out["auroc"].items() if k.startswith("reflex:")}
    best = max(reflex, key=lambda k: reflex[k] or 0.0)
    out["reflex_best"] = {"signal": best, "auroc": reflex[best]}
    out["verifier_minus_reflex_best"] = (out["auroc"]["verifier:score"] or 0.0) - (reflex[best] or 0.0)
    pc = feats["verifier:p_correct"]
    out["p_correct"] = {"available": int(sum(p is not None for p in pc)), "ece": ece(pc, ys),
                        "mean_p": float(np.mean([p for p in pc if p is not None])) if any(p is not None for p in pc)
                        else None, "pass_rate": float(y.mean()) if len(y) else None}
    if per_seed is not None:
        out["per_seed"] = per_seed
    return out


def run_prequential(runs):
    """Every row in stream order: score before its verify, then learn through swarm_step (per seed)."""
    feats, ys, seen_flags = {}, [], []
    for seed in range(SEEDS):
        R = sorted([r for r in runs["rows"] if r["seed"] == seed], key=lambda r: r["t"])
        m = _new_mind({r["state"]: r["action"] for r in R if r["ok"]})
        seen = set()
        for r in R:
            for k, v in _features(m, r).items():
                feats.setdefault(k, []).append(v)
            ys.append(r["ok"])
            seen_flags.append(r["state"] in seen)
            seen.add(r["state"])
            _submit(m, r)
    out = _summarise(feats, ys)
    sf = np.array(seen_flags)
    sub = {k: [v for v, s in zip(vals, sf) if s] for k, vals in feats.items()}
    out["seen_state_rows"] = _summarise(sub, [y for y, s in zip(ys, sf) if s])
    return out


def run_utility(runs, use_precheck, reflex_first=True, min_p=None, strong="default"):
    """The swarm (bench_swarm_reflex's shared_judged condition) with candidates in router order or pre-check order."""
    res = {"judge_calls": 0, "tasks": 0, "solved": 0, "reflex_served": 0, "dropped_truth": 0, "per_seed": []}
    for seed in range(SEEDS):
        tasks, streams = SWARM.workload(seed)
        truth = {w: t["truth"] for t in tasks for w in t["wordings"]}
        m = _new_mind(truth)
        for step in range(SWARM.TASKS_PER_AGENT):
            for a in range(SWARM.N_AGENTS):
                tid, wi = streams[a][step]
                q = tasks[tid]["wordings"][wi]
                res["tasks"] += 1
                if reflex_first:
                    d = m.reflex_decide(q, key="ngram")
                    if d.get("value") is not None:
                        res["reflex_served"] += 1
                        res["solved"] += d["value"] == truth[q]
                        continue
                cands = runs["candidates"]["%d|%s" % (seed, q)]
                order = cands
                if use_precheck:
                    kw = {} if strong == "default" else {"strong": strong}
                    pre = m.verify_precheck(q, cands, min_p=min_p, **kw)
                    order = pre["order"]
                    res["dropped_truth"] += truth[q] in pre["dropped"]
                for name in order:
                    if _submit(m, {"state": q, "action": name, "agent": a}):
                        res["solved"] += 1
                        break
        res["per_seed"].append(m.judge_calls)
        res["judge_calls"] += m.judge_calls
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=os.path.join(EVIDENCE, "bench_verifier.json"))
    ap.add_argument("--fresh", action="store_true", help="redraw the run log from the live catalog")
    args = ap.parse_args(argv)
    t0 = time.time()
    load0 = open("/proc/loadavg").read().split()[:3] if os.path.exists("/proc/loadavg") else None
    runs = load_runs(fresh=args.fresh)
    blob = json.dumps(runs, sort_keys=True).encode()
    out = {"config": {"seeds": SEEDS, "train_buckets_of_10": TRAIN_FRACTION, "agents": SWARM.N_AGENTS,
                      "draws_per_agent": SWARM.TASKS_PER_AGENT, "k_candidates": SWARM.K_CANDIDATES,
                      "runs_sha256": hashlib.sha256(blob).hexdigest(), "rows": len(runs["rows"]),
                      "rows_pass": int(sum(r["ok"] for r in runs["rows"]))}}
    print("run log: %d rows, %d passed (sha256 %s...)" % (len(runs["rows"]), out["config"]["rows_pass"],
                                                          out["config"]["runs_sha256"][:12]), flush=True)
    for name, key_of in (("split_by_state", lambda r: "%d|%s" % (r["seed"], r["state"])),
                         ("split_by_task", lambda r: "%d|task%d" % (r["seed"], r["tid"]))):
        per_salt = [run_split(runs, key_of, salt) for salt in range(SALTS)]
        v = [p["auroc"]["verifier:score"] for p in per_salt]
        rb = [p["reflex_best"]["auroc"] for p in per_salt]
        out[name] = {"salts": per_salt, "verifier_auroc": {"mean": float(np.mean(v)), "sd": float(np.std(v)), "each": v},
                     "reflex_best_auroc": {"mean": float(np.mean(rb)), "sd": float(np.std(rb)), "each": rb,
                                           "signals": [p["reflex_best"]["signal"] for p in per_salt]},
                     "router_auroc": {"mean": float(np.mean([p["auroc"]["router:rank"] for p in per_salt]))},
                     "verifier_wins": int(sum(a > b for a, b in zip(v, rb))),
                     "ece_mean": float(np.mean([p["p_correct"]["ece"] for p in per_salt if p["p_correct"]["ece"] is not None]))}
        print("%-15s verifier AUROC %.3f +- %.3f vs reflex best %.3f +- %.3f (router %.3f) -- verifier wins %d of %d "
              "salts | ECE %.3f" % (name, out[name]["verifier_auroc"]["mean"], out[name]["verifier_auroc"]["sd"],
                                   out[name]["reflex_best_auroc"]["mean"], out[name]["reflex_best_auroc"]["sd"],
                                   out[name]["router_auroc"]["mean"], out[name]["verifier_wins"], SALTS,
                                   out[name]["ece_mean"]), flush=True)
    out["prequential"] = run_prequential(runs)
    s = out["prequential"]
    print("prequential     verifier AUROC %.3f vs reflex best %s %.3f (router %.3f) | ECE %.3f | seen-state rows: "
          "verifier %.3f vs reflex %.3f" % (s["auroc"]["verifier:score"], s["reflex_best"]["signal"],
                                            s["reflex_best"]["auroc"], s["auroc"]["router:rank"], s["p_correct"]["ece"],
                                            s["seen_state_rows"]["auroc"]["verifier:score"],
                                            s["seen_state_rows"]["reflex_best"]["auroc"]), flush=True)
    out["utility"] = {"router_order": run_utility(runs, False),
                      "precheck_order": run_utility(runs, True),
                      "precheck_raw_score_order": run_utility(runs, True, strong=None),
                      "router_order_no_reflex": run_utility(runs, False, reflex_first=False),
                      "precheck_order_no_reflex": run_utility(runs, True, reflex_first=False),
                      "precheck_raw_score_order_no_reflex": run_utility(runs, True, reflex_first=False, strong=None),
                      "precheck_prefilter_min_p_0.05": run_utility(runs, True, min_p=0.05)}
    for k, u in out["utility"].items():
        print("utility %-36s judge calls %4d  solved %d/%d  reflex served %d  truth dropped %d"
              % (k, u["judge_calls"], u["solved"], u["tasks"], u["reflex_served"], u["dropped_truth"]), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    out["loadavg_start"] = load0
    out["loadavg_end"] = open("/proc/loadavg").read().split()[:3] if os.path.exists("/proc/loadavg") else None
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return out


if __name__ == "__main__":
    main()
