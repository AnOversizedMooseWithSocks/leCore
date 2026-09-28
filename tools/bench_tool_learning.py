"""tools/bench_tool_learning.py -- do TOOL CALLS teach the decision learners, through the real agent path?

Sweep 178 connected tool calling to the decision records: AgentLoop reports a JUDGED tool call against the
route decision (verify=), the menu band's "choose" is reported against the same id, and route_tiered(
reflex=True) answers a repeat from that experience. This bench measures it on the path agents actually use
-- mind.tool_loop / AgentLoop -- rather than on swarm_step (tools/bench_swarm_reflex.py).

THE TASK: a hidden target TOOL (a callable catalog card) and a request made of 4 words from its own
description. Calibrated before this file was written (100 requests): the loop's gate answers 33% directly
(all correct), offers a MENU for 65% (the target always in it), refuses 2%; vaguer requests (2 own + 2
distractor words) are refused 85% of the time -- the gate will not act on them, by design, so they cannot
teach a tool loop anything and are not used here.

HOW AN AGENT HANDLES ONE REQUEST (every condition, same rules):
    out = AgentLoop(mind, scripted_model, verify=?, reflex=?).run(request)
      * gate 'answer' -> the model calls the manifest's tools in order until the JUDGE passes one, then DONE.
      * gate 'menu'   -> the loop does not act (a menu is not a licence to act); the agent CHOOSES by judging
                         the menu's options in order, then reports its choice against the gate's id.
      * gate 'refuse' -> unsolved.
    One JUDGE CALL per tool tried = the cost. Tools are stubbed (invoke returns at once): this measures the
    learning loop, not the tools. The judge is deterministic: pass iff the tool is the target's method.

CONDITIONS (4 agents x 40 requests, round-robin, pool of 60 targets, Zipf(1.1), 2 wordings each, 3 seeds):
    no_learning        isolated minds, reflex off, nothing reported              (the plain best-of-N baseline)
    isolated_judged    isolated minds, reflex on, judged calls/choices reported (each agent learns alone)
    shared_judged      ONE mind for all 4 agents, reflex on, judged reports    (the swarm)
    shared_selfreport  one mind, reflex on, the FIRST tool / first menu option reported WITHOUT judging,
                       and the agent stops there (it believes itself) -- the pre-177 learning path
    restart            after shared_judged: learning_save -> fresh mind -> learning_load, every wording asked

Run:  PYTHONHASHSEED=0 python tools/bench_tool_learning.py [--seeds 3] [--json out.json]
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONHASHSEED", "0")

N_AGENTS = 4
TASKS_PER_AGENT = 40
POOL = 60
OWN_WORDS = 4
ZIPF_S = 1.1
P_ALT_WORDING = 0.25
K_TOOLS = 6
MIND_DIM = 256


def _words(text):
    return re.findall(r"[a-z]{4,}", (text or "").lower())


def make_workload(seed):
    """tasks[tid] = {card, method, wordings[2]}; streams[a] = [(tid, wording), ...]."""
    from holographic.caching_and_storage import holographic_catalog as C
    cards = [c for c in C.default_catalog().all() if getattr(c, "method", None) and len(_words(c.does)) >= 8]
    rng = np.random.default_rng(seed)
    chosen = [cards[i] for i in rng.permutation(len(cards))[:POOL]]

    def request(card):
        return " ".join(rng.choice(_words(card.does), OWN_WORDS, replace=False))

    tasks = [{"card": c.name, "method": c.method, "wordings": [request(c), request(c)]} for c in chosen]
    w = 1.0 / np.arange(1, POOL + 1) ** ZIPF_S
    w /= w.sum()
    streams = []
    for _ in range(N_AGENTS):
        tids = rng.choice(POOL, TASKS_PER_AGENT, p=w)
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
    path = os.path.join(EVIDENCE, "bench_tool_learning_workload_seed%d.json" % seed)
    if not fresh and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d["tasks"], [[tuple(x) for x in s_] for s_ in d["streams"]]
    tasks, streams = make_workload(seed)
    if freeze:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "tasks": tasks, "streams": streams}, f, indent=0)
    return tasks, streams



def new_mind():
    """A mind whose invoke is a stub: a tool 'runs' and returns its name. The engine is not exercised."""
    import lecore
    m = lecore.UnifiedMind(dim=MIND_DIM, seed=0)
    m.invoke = lambda name, args=None: {"ran": name}
    return m


def handle(mind, task, q, mode, cost):
    """One agent, one request. Returns a dict describing what happened. `cost` counts judge calls."""
    from holographic.agents_and_reasoning.holographic_agentloop import AgentLoop
    target = task["method"]

    def judge(tool):
        cost["judge"] += 1
        return tool == target

    verdicts = {}                                       # call index -> verdict (so no call is judged twice)
    learning = mode in ("judged", "selfreport")
    judged = mode == "judged"

    def verify(t, tool, args, result):                  # AgentLoop's judge hook (judged mode only)
        v = judge(tool)
        verdicts[len(verdicts)] = v
        return v

    tried = []

    def model(prompt):
        """Scripted model: try the manifest's tools in order; stop when the last one passed (or, in
        self-report mode, after the first call -- it believes itself)."""
        tools = [ln.strip().split(" -- ")[0] for ln in prompt.split("TOOLS:")[1].split("\n\n")[0].splitlines()
                 if ln.strip()]
        if tried:
            if mode == "selfreport":
                return "DONE: believed"
            last_ok = verdicts.get(len(tried) - 1)
            if last_ok is None:                         # not judged by the loop (no_learning): judge here
                last_ok = judge(tried[-1])
                verdicts[len(tried) - 1] = last_ok
            if last_ok:
                return "DONE: judged"
        rest = [t for t in dict.fromkeys(tools) if t not in tried]
        if not rest:
            return "DONE: gave up"
        tried.append(rest[0])
        return '{"tool": "%s", "args": {}}' % rest[0]

    loop = AgentLoop(mind, model, max_steps=K_TOOLS + 1, k_tools=K_TOOLS, verify=(verify if judged else None),
                     reflex=learning)
    out = loop.run(q)
    gate = out["gate"]
    res = {"tier": gate.get("tier"), "from_experience": False, "experience_wrong": False, "solved": False}
    if gate.get("tier") == "answer":
        via_reflex = bool(gate.get("hits")) and gate.get("reason", "").startswith("answered from experience")
        res["from_experience"] = via_reflex
        if mode == "selfreport":
            first = out["steps"][0] if out["steps"] else None
            res["solved"] = bool(first and first["tool"] == target)
            if first and gate.get("id"):
                mind.decision_outcome(gate["id"], first["card"])    # believed, never judged
        else:
            res["solved"] = any(verdicts.values())
        if via_reflex:
            first = out["steps"][0]["tool"] if out["steps"] else None
            res["experience_wrong"] = first != target
        res["calls"] = len(tried)
        return res
    if out.get("decision") == "choose":
        options = [name for name, _ in out.get("options", [])]
        cat = mind._capability_catalog()
        chosen = None
        for name in options:
            c = cat.get(name) if hasattr(cat, "get") else None
            meth = getattr(c, "method", None)
            if mode == "selfreport":
                chosen = name if meth else None
                res["solved"] = meth == target
                break
            if meth and judge(meth):
                chosen, res["solved"] = name, True
                break
        if learning and chosen and gate.get("id"):
            mind.decision_outcome(gate["id"], chosen)
        return res
    return res                                          # refused


def _blank():
    return {"requests": 0, "judge_calls": 0, "solved": 0, "answer": 0, "menu": 0, "refuse": 0,
            "from_experience": 0, "experience_wrong": 0, "seconds": 0.0,
            "by_kind": {k: {"n": 0, "exp": 0, "exp_wrong": 0} for k in ("new", "repeat", "paraphrase")}}


def run_condition(tasks, streams, shared, mode):
    t0 = time.time()
    st = _blank()
    cost = {"judge": 0}
    if shared:
        one = new_mind()
        minds, scopes = [one] * N_AGENTS, [set()] * N_AGENTS
    else:
        minds, scopes = [new_mind() for _ in range(N_AGENTS)], [set() for _ in range(N_AGENTS)]
    for step in range(TASKS_PER_AGENT):
        for a in range(N_AGENTS):
            tid, wi = streams[a][step]
            scope = scopes[a]
            kind = "repeat" if (tid, wi) in scope else ("paraphrase" if any(t == tid for t, _ in scope) else "new")
            scope.add((tid, wi))
            r = handle(minds[a], tasks[tid], tasks[tid]["wordings"][wi], mode, cost)
            st["requests"] += 1
            st[r["tier"] if r["tier"] in ("answer", "menu", "refuse") else "refuse"] += 1
            st["solved"] += r["solved"]
            st["from_experience"] += r["from_experience"]
            st["experience_wrong"] += r["experience_wrong"]
            b = st["by_kind"][kind]
            b["n"] += 1
            b["exp"] += r["from_experience"]
            b["exp_wrong"] += r["experience_wrong"]
    st["judge_calls"] = cost["judge"]
    st["seconds"] = round(time.time() - t0, 1)
    return st, (minds[0] if shared else None)


def run_restart(tasks, trained):
    root = tempfile.mkdtemp(prefix="lecore_scratch_toolbench_")
    os.makedirs(os.path.join(root, "learning"), exist_ok=True)
    trained.learning_save(root)
    fresh = new_mind()
    fresh.learning_load(root)
    out = {"asked": 0, "from_experience": 0, "right": 0, "wrong": 0}
    for t in tasks:
        for q in t["wordings"]:
            out["asked"] += 1
            r = fresh.route_tiered(q, k=K_TOOLS, z_answer=0.1, z_refuse=-0.5, reflex=True)
            if r.get("via") == "reflex":
                out["from_experience"] += 1
                out["right" if r["answer"] == t["card"] else "wrong"] += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--fresh", action="store_true", help="redraw tasks from the live catalog, not the fixture")
    ap.add_argument("--freeze", action="store_true", help="write the drawn tasks as the new fixture")
    args = ap.parse_args(argv)
    conds = [("no_learning", False, "none"), ("isolated_judged", False, "judged"),
             ("shared_judged", True, "judged"), ("shared_selfreport", True, "selfreport")]
    R = {n: [] for n, _, _ in conds}
    R["restart"] = []
    for seed in range(args.seeds):
        tasks, streams = workload(seed, fresh=args.fresh, freeze=args.freeze)
        for name, shared, mode in conds:
            st, mind = run_condition(tasks, streams, shared, mode)
            R[name].append(st)
            if name == "shared_judged":
                R["restart"].append(run_restart(tasks, mind))
            print("seed %d %-18s done in %5.1fs" % (seed, name, st["seconds"]), flush=True)

    def col(n, k):
        return np.array([r[k] for r in R[n]], float)

    print("\n%d agents x %d requests, pool %d, %d seeds (mean +- sd)" % (N_AGENTS, TASKS_PER_AGENT, POOL, args.seeds))
    hdr = "%-18s %13s %13s %11s %11s %13s %12s" % ("condition", "judge calls", "solved", "answer", "menu",
                                                  "from exper.", "exper. wrong")
    print(hdr)
    print("-" * len(hdr))
    for n, _, _ in conds:
        req = col(n, "requests")
        print("%-18s %7.1f+-%-5.1f %6.3f+-%.3f %11.1f %11.1f %13.1f %12.1f" % (
            n, col(n, "judge_calls").mean(), col(n, "judge_calls").std(),
            (col(n, "solved") / req).mean(), (col(n, "solved") / req).std(),
            col(n, "answer").mean(), col(n, "menu").mean(), col(n, "from_experience").mean(),
            col(n, "experience_wrong").mean()))
    print("\nanswered from experience, by what the learning scope had seen (summed over seeds):")
    for n, _, _ in conds:
        parts = []
        for k in ("new", "repeat", "paraphrase"):
            nn = sum(r["by_kind"][k]["n"] for r in R[n])
            e = sum(r["by_kind"][k]["exp"] for r in R[n])
            w = sum(r["by_kind"][k]["exp_wrong"] for r in R[n])
            parts.append("%s %d/%d (%d wrong)" % (k, e, nn, w))
        print("  %-18s %s" % (n, "; ".join(parts)))
    rs = R["restart"]
    print("\nrestart (shared_judged -> learning_save -> fresh mind -> learning_load), summed over seeds: "
          "asked %d wordings, answered from experience %d (%d right, %d wrong)"
          % (sum(r["asked"] for r in rs), sum(r["from_experience"] for r in rs),
             sum(r["right"] for r in rs), sum(r["wrong"] for r in rs)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"config": {"agents": N_AGENTS, "requests_per_agent": TASKS_PER_AGENT, "pool": POOL,
                                  "own_words": OWN_WORDS, "k_tools": K_TOOLS, "zipf_s": ZIPF_S,
                                  "p_alt_wording": P_ALT_WORDING, "seeds": args.seeds}, "results": R}, f, indent=1)
    return R


if __name__ == "__main__":
    main()
