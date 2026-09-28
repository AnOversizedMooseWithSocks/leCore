"""tools/bench_swarm_reflex.py -- does a swarm that SHARES judged experience beat isolated agents?

The SwarmWorld comparison (Pal, Wang, Buehler, arXiv 2608.26081), run on leCore's own machinery after sweep 177
connected the reflex arc to the judge (verified swarm_step -> reflex_learn) and to the partition (the reflex
bridge now persists). Everything here is deterministic: same seed, same numbers.

THE TASK (a stand-in for "propose a design, let the simulator decide"):
    A task is a hidden target capability plus a VAGUE description of it -- 2 words from the target's own
    description mixed with 2 distractor words from an unrelated card. An agent solves it by SEARCH: it asks
    find_capability for 16 candidates and submits them, in ranked order, to a deterministic JUDGE (the stand-in
    for the paper's simulator) until one passes. Every judge call is one unit of COST. Calibrated before this
    file was written (120 tasks): the first candidate is right 21% of the time, 86% are solvable within 16,
    and a solve averages 4.5 judge calls -- a real search, not a lookup. The judge knows the target; agents
    never see it.

    Before searching, an agent asks its reflex (reflex_decide, key='ngram'). If the reflex answers, the task
    is served at ZERO judge calls. Whether that answer was right is scored by the bench, never charged.

CONDITIONS (4 agents, interleaved round-robin -- one "tick" each -- 40 tasks per agent):
    isolated_judged    4 separate minds. Each learns only from its OWN judged steps.   (best-of-N baseline)
    shared_judged      4 agents on ONE mind (= one leCore service). Learns from judged steps.
    shared_selfreport  4 agents on one mind, the OLD learning path: an agent takes the router's first
                       candidate WITHOUT judging it and reports it as the outcome (decision_outcome).
                       The control for fix #2: what the reflex learns when nobody checks.
    restart            after shared_judged, learning_save -> a FRESH mind learning_load -> ask every task
                       wording again. The control for fix #1 (before sweep 177 this fired 0 times).

WORKLOAD: a pool of 60 tasks, drawn per agent with Zipf(1.1) weights (some subproblems are common, most are
rare -- the overlap a real swarm sees). Each task has TWO wordings; 75% of draws use wording 0, 25% wording 1,
so we can report exact-wording repeats and new-wording ("paraphrase") repeats separately.

Run:  PYTHONHASHSEED=0 python tools/bench_swarm_reflex.py            (3 seeds, ~1-2 minutes on a laptop CPU)
      PYTHONHASHSEED=0 python tools/bench_swarm_reflex.py --json out.json
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

# run from anywhere: the repo root is this file's parent's parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONHASHSEED", "0")

N_AGENTS = 4
TASKS_PER_AGENT = 40
POOL = 60
K_CANDIDATES = 16
ZIPF_S = 1.1
P_ALT_WORDING = 0.25
MIND_DIM = 256


# --------------------------------------------------------------------------------------------- workload ----

def _words(text):
    """Content words of a capability description (4+ letters, lowercase)."""
    return re.findall(r"[a-z]{4,}", (text or "").lower())


def make_workload(seed):
    """Build the task pool and each agent's stream. Returns (tasks, streams) where
    tasks[tid] = {"truth": card name, "wordings": [q0, q1]} and streams[a] = [(tid, wording_index), ...]."""
    from holographic.caching_and_storage import holographic_catalog as C
    cards = [c for c in C.default_catalog().all() if len(_words(c.does)) >= 8]
    rng = np.random.default_rng(seed)
    chosen = [cards[i] for i in rng.permutation(len(cards))[:POOL]]

    def vague(card):
        # 2 of the target's own words + 2 distractors from a random other card, shuffled together
        own = list(rng.choice(_words(card.does), 2, replace=False))
        other = cards[int(rng.integers(len(cards)))]
        noise = list(rng.choice(_words(other.does), 2, replace=False))
        ws = own + noise
        rng.shuffle(ws)
        return " ".join(ws)

    tasks = [{"truth": c.name, "wordings": [vague(c), vague(c)]} for c in chosen]
    weights = 1.0 / np.arange(1, POOL + 1) ** ZIPF_S
    weights /= weights.sum()
    streams = []
    for _ in range(N_AGENTS):
        tids = rng.choice(POOL, TASKS_PER_AGENT, p=weights)
        alts = rng.random(TASKS_PER_AGENT) < P_ALT_WORDING
        streams.append([(int(t), int(a)) for t, a in zip(tids, alts)])
    return tasks, streams

# FROZEN WORKLOADS (sweep 179). The tasks are drawn from the LIVE catalog, so adding one card anywhere reshuffled
# the draw and every number moved -- measured: sweep 179 added one catalog card and this bench changed on every
# seed while the code under test was bit-identical. A benchmark whose questions change between runs cannot
# compare runs. So the workload is frozen to a fixture beside the results and read back by default;
# --fresh redraws from the live catalog (and --freeze writes that draw as the new fixture).
EVIDENCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "research", "evidence")


def workload(seed, fresh=False, freeze=False):
    """(tasks, streams) for one seed: the frozen fixture unless fresh=True. freeze=True writes what was drawn."""
    path = os.path.join(EVIDENCE, "bench_swarm_reflex_workload_seed%d.json" % seed)
    if not fresh and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d["tasks"], [[tuple(x) for x in s_] for s_ in d["streams"]]
    tasks, streams = make_workload(seed)
    if freeze:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "tasks": tasks, "streams": streams}, f, indent=0)
    return tasks, streams



# ---------------------------------------------------------------------------------------------- agents ----

def new_mind(tasks, counter):
    """A fresh UnifiedMind with the JUDGE attached as a mind verb (swarm_step's verify calls mind verbs by name).
    The judge is deterministic: pass iff the candidate is the task's hidden target. `counter` counts calls."""
    import lecore
    m = lecore.UnifiedMind(dim=MIND_DIM, seed=0)

    def judge_design(task, candidate):
        counter["judge_calls"] += 1
        return tasks[int(task)]["truth"] == candidate
    m.judge_design = judge_design
    return m


def agent_tick(mind, agent, tid, wi, tasks, mode, stats, seen_by_scope):
    """One agent handles one task on `mind`. mode: 'judged' or 'selfreport'. Updates `stats` in place.

    The scope key tells us, BEFORE acting, whether this learning scope (the shared mind, or this agent's own
    mind) has handled this exact wording before, or only another wording of the same task -- so reflex fires
    can be split into exact repeats vs paraphrases."""
    q = tasks[tid]["wordings"][wi]
    truth = tasks[tid]["truth"]
    if (tid, wi) in seen_by_scope:
        kind = "repeat"
    elif any(t == tid for t, _ in seen_by_scope):
        kind = "paraphrase"
    else:
        kind = "new"
    seen_by_scope.add((tid, wi))
    stats["by_kind"][kind]["n"] += 1

    # 1. ANSWER FROM EXPERIENCE FIRST (zero judge calls)
    d = mind.reflex_decide(q, key="ngram")
    if d.get("value") is not None:
        ok = d["value"] == truth
        stats["reflex_served"] += 1
        stats["reflex_wrong"] += (not ok)
        stats["correct"] += ok
        stats["by_kind"][kind]["reflex"] += 1
        stats["by_kind"][kind]["reflex_wrong"] += (not ok)
        return

    # 2. THE EXPENSIVE PATH: search
    cands = [c.name for c in mind.find_capability(q, k=K_CANDIDATES)]
    if mode == "selfreport":
        # the old path: trust the first candidate, report it as the outcome, no judge
        if not cands:
            stats["unsolved"] += 1
            return
        pick = cands[0]
        s = mind.swarm_step(q, pick, done_when="the agent says so", evidence={"candidate": pick}, worker=agent)
        mind.decision_outcome(s["id"], pick)          # self-reported: the reflex learns whatever this says
        stats["searched"] += 1
        stats["correct"] += (pick == truth)
        stats["search_wrong"] += (pick != truth)
        return

    # judged: submit candidates in order until the judge passes one (each submission = 1 judge call)
    for name in cands:
        try:
            mind.swarm_step(q, name, done_when="judge_design passes", evidence={"candidate": name}, worker=agent,
                            verify={"verb": "judge_design", "args": {"task": tid, "candidate": name}}, expect=True)
        except ValueError:
            continue                                   # refused by the judge: never reaches the bus or the reflex
        stats["searched"] += 1
        stats["correct"] += 1
        return
    stats["searched"] += 1
    stats["unsolved"] += 1                             # no candidate passed within K


def _blank_stats():
    return {"tasks": 0, "judge_calls": 0, "correct": 0, "reflex_served": 0, "reflex_wrong": 0,
            "searched": 0, "search_wrong": 0, "unsolved": 0, "seconds": 0.0,
            "by_kind": {k: {"n": 0, "reflex": 0, "reflex_wrong": 0} for k in ("new", "repeat", "paraphrase")}}


def run_condition(tasks, streams, shared, mode):
    """Round-robin the agents' streams (agent 0 task 0, agent 1 task 0, ...). Returns (stats, shared_mind)."""
    t0 = time.time()
    stats = _blank_stats()
    counter = {"judge_calls": 0}
    if shared:
        one = new_mind(tasks, counter)
        minds = [one] * N_AGENTS
        scopes = [set()] * N_AGENTS                    # ONE scope: everything any agent did
    else:
        minds = [new_mind(tasks, counter) for _ in range(N_AGENTS)]
        scopes = [set() for _ in range(N_AGENTS)]
    for step in range(TASKS_PER_AGENT):
        for a in range(N_AGENTS):
            tid, wi = streams[a][step]
            agent_tick(minds[a], "agent%d" % a, tid, wi, tasks, mode, stats, scopes[a])
            stats["tasks"] += 1
    stats["judge_calls"] = counter["judge_calls"]
    stats["seconds"] = round(time.time() - t0, 2)
    return stats, (minds[0] if shared else None)


def run_restart(tasks, trained_mind):
    """Fix #1's control: persist the shared mind's learning, boot a fresh mind from it, and ask every wording
    of every task again. Counts fires on wordings the swarm SOLVED (should fire, and be right) and on wordings
    it never solved (should stay refused)."""
    root = tempfile.mkdtemp(prefix="lecore_scratch_bench_")
    os.makedirs(os.path.join(root, "learning"), exist_ok=True)
    trained_mind.learning_save(root)
    counter = {"judge_calls": 0}
    fresh = new_mind(tasks, counter)
    fresh.learning_load(root)
    out ={"asked": 0, "fired": 0, "fired_right": 0, "fired_wrong": 0}
    known = set()
    for rec in trained_mind.decision_ledger()._rows.values():
        if rec.via == "swarm" and rec.outcome is not None:
            known.add(rec.state)
    out["known_wordings"] = len(known)
    for t in tasks:
        for q in t["wordings"]:
            out["asked"] += 1
            d = fresh.reflex_decide(q, key="ngram")
            if d.get("value") is not None:
                out["fired"] += 1
                if d["value"] == t["truth"]:
                    out["fired_right"] += 1
                else:
                    out["fired_wrong"] += 1
    out["fired_on_known"] = sum(1 for t in tasks for q in t["wordings"]
                                if q in known and fresh.reflex_decide(q, key="ngram").get("value") is not None)
    return out


# ------------------------------------------------------------------------------------------------ main ----

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", default=None, help="also write every number to this file")
    ap.add_argument("--fresh", action="store_true", help="redraw tasks from the live catalog, not the fixture")
    ap.add_argument("--freeze", action="store_true", help="write the drawn tasks as the new fixture")
    args = ap.parse_args(argv)

    conds = [("isolated_judged", False, "judged"), ("shared_judged", True, "judged"),
             ("shared_selfreport", True, "selfreport")]
    results = {name: [] for name, _, _ in conds}
    results["restart"] = []
    for seed in range(args.seeds):
        tasks, streams = workload(seed, fresh=args.fresh, freeze=args.freeze)
        for name, shared, mode in conds:
            st, mind = run_condition(tasks, streams, shared, mode)
            results[name].append(st)
            if name == "shared_judged":
                results["restart"].append(run_restart(tasks, mind))
        print("seed %d done" % seed, flush=True)

    def mean(name, key):
        return float(np.mean([r[key] for r in results[name]]))

    def sd(name, key):
        return float(np.std([r[key] for r in results[name]]))

    print("\n%d agents x %d tasks, pool %d, %d seeds (mean +- sd)" % (N_AGENTS, TASKS_PER_AGENT, POOL, args.seeds))
    hdr = "%-18s %14s %12s %14s %13s %9s" % ("condition", "judge calls", "accuracy", "reflex served",
                                            "reflex wrong", "seconds")
    print(hdr)
    print("-" * len(hdr))
    for name, _, _ in conds:
        n = mean(name, "tasks")
        print("%-18s %8.1f+-%-4.1f %6.3f+-%.3f %8.1f+-%-4.1f %7.1f+-%-4.1f %9.1f" % (
            name, mean(name, "judge_calls"), sd(name, "judge_calls"),
            mean(name, "correct") / n, float(np.std([r["correct"] / r["tasks"] for r in results[name]])),
            mean(name, "reflex_served"), sd(name, "reflex_served"),
            mean(name, "reflex_wrong"), sd(name, "reflex_wrong"), mean(name, "seconds")))

    print("\nreflex fires by what the learning scope had seen before (summed over seeds):")
    for name, _, _ in conds:
        parts = []
        for k in ("new", "repeat", "paraphrase"):
            n = sum(r["by_kind"][k]["n"] for r in results[name])
            f = sum(r["by_kind"][k]["reflex"] for r in results[name])
            w = sum(r["by_kind"][k]["reflex_wrong"] for r in results[name])
            parts.append("%s %d/%d fired (%d wrong)" % (k, f, n, w))
        print("  %-18s %s" % (name, "; ".join(parts)))

    rs = results["restart"]
    print("\nrestart (learning_save -> fresh mind -> learning_load), summed over seeds:")
    print("  wordings the swarm had solved: %d, fired after reboot on %d; all wordings asked %d: fired %d "
          "(%d right, %d wrong)" % (sum(r["known_wordings"] for r in rs), sum(r["fired_on_known"] for r in rs),
                                    sum(r["asked"] for r in rs), sum(r["fired"] for r in rs),
                                    sum(r["fired_right"] for r in rs), sum(r["fired_wrong"] for r in rs)))
    print("  (before sweep 177 the same reboot fired 0 times: the seen gate and label book were not saved)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"config": {"agents": N_AGENTS, "tasks_per_agent": TASKS_PER_AGENT, "pool": POOL,
                                  "k": K_CANDIDATES, "zipf_s": ZIPF_S, "p_alt_wording": P_ALT_WORDING,
                                  "seeds": args.seeds}, "results": results}, f, indent=1)
    return results


if __name__ == "__main__":
    main()
