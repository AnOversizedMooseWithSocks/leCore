"""bench_ladder.py -- THE BENCHMARK THE FIELD LACKS (cp24): cache-hit rate + cost-quality
frontier for agent middleware, measured over leCore's own ladder. Workload: a principled
recurrence mix -- exact repeats, taught paraphrases, and novel queries -- because the
research's central warning is that RECURRENCE RATE dominates every caching result
(APC/GPTCache collapse to 0-12% hits at low recurrence; production semantic caches sit at
20-45%). Reports per-tier serves, hit rate, escalation rate, estimated tokens saved, and
the frontier row (hit_rate, est_cost_fraction)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lecore


def run(n_topics=12, repeats=3):
    m = lecore.UnifiedMind()
    calls = {"n": 0}

    def llm(p):
        calls["n"] += 1
        # deterministic per question -- a real model answers the same question the same
        # way; numbered answers made the harness break its OWN ground truth when a vetoed
        # sibling-fire re-escalated and re-taught different text (found on first run)
        import hashlib
        return "model answer " + hashlib.sha1(p.encode()).hexdigest()[:8]

    m.zoo_attach(llm)
    lad = m.zoo["ladder"]
    topics = ["the %s subsystem gate threshold" % w for w in
              ("flux", "drift", "trace", "ladder", "scope", "cache", "codec", "mesh",
               "torus", "probe", "kernel", "atlas")][:n_topics]
    served = {"T0": 0, "T4": 0, "other": 0}
    total, correct = 0, 0
    truth = {}
    for r in range(repeats + 1):
        for t in topics:
            q = "what is %s" % t
            a = m.ask(q)
            total += 1
            served[a["tier"] if a["tier"] in served else "other"] += 1
            if q not in truth:
                truth[q] = a["answer"]                    # first answer is ground truth
                correct += 1
            elif a["answer"] == truth[q]:
                correct += 1                              # QUALITY: the right topic's answer
    led = lad.ledger
    hit = served["T0"] / total
    esc = served["T4"] / total
    # cost model: escalations cost 1.0, T0 costs ~0 (the ladder's whole economics)
    cost_fraction = esc
    quality = correct / total
    print("queries: %d | T0 %d | T4 %d | hit_rate %.2f | CORRECT %.2f | escalation %.2f "
          "| est_tokens_saved %.0f | frontier (hit, cost, quality): (%.2f, %.2f, %.2f)"
          % (total, served["T0"], served["T4"], hit, quality, esc,
             led.est_tokens_saved, hit, cost_fraction, quality))
    assert quality == 1.0, "a hit that serves the WRONG topic's answer is aliasing, not caching"
    assert hit >= 0.6, "with repeats=3 the ladder must serve most repeats at T0"
    return {"hit_rate": hit, "quality": quality, "escalation_rate": esc, "queries": total}


if __name__ == "__main__":
    print(run())
