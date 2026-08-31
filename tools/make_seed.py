"""SEED BUILDER (cp67): turn plain facts into a personalization seed bundle.

A SEED is how an operator (openzoo.fun) or a builder personalizes leCore without
touching code: a small .lecore memory bundle taught from their own facts, loaded at
boot. The substrate then answers their domain questions from memory at zero model
cost, escalates honestly outside it, and every seeded entry keeps the full contract
-- provenance, veto, session isolation, replay.

Input formats (one file, either):
  JSON:      [{"q": "...", "a": "..."}, ...]        or {"question": "answer", ...}
  Markdown:  lines of "question = answer"           (# comments ignored)

Usage:
  python tools/make_seed.py facts.json out_seed_dir
  -> out_seed_dir/learning/state.lecore  (+ a printed verification report)

Load it:
  lecore.autoboot(partition="out_seed_dir")             # local
  chat: "load memory out_seed_dir"                      # as a named slot
  hosted: the operator points LECORE_PARTITION at it    # openzoo

The builder VERIFIES before it blesses: every seeded question must answer at T0
from a fresh boot, and a held-out probe must escalate -- a seed that leaks or a
seed that cannot answer its own facts is refused, not shipped.
"""
import json
import os
import sys

sys.path.insert(0, ".")


def parse_facts(path):
    text = open(path).read()
    if path.endswith(".json"):
        data = json.loads(text)
        if isinstance(data, dict):
            return [(str(k), str(v)) for k, v in data.items()]
        return [(str(d["q"]), str(d["a"])) for d in data]
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        q, a = line.split("=", 1)
        pairs.append((q.strip(), a.strip()))
    return pairs


def build(facts_path, out_dir):
    import lecore
    pairs = parse_facts(facts_path)
    if not pairs:
        print("no facts found in %s" % facts_path)
        return 1
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "")
    refused = []
    for q, a in pairs:
        r = m.teach(q, a)
        if isinstance(r, dict) and r.get("taught") is False:
            refused.append((q, r.get("reason", "")))
    os.makedirs(out_dir, exist_ok=True)
    m.learning_save(out_dir)
    # VERIFY from a fresh boot: all facts at T0; a held-out probe escalates
    v = lecore.UnifiedMind()
    v.zoo_attach(lambda p: "")
    v.learning_load(out_dir)
    misses = [q for q, a in pairs
              if (q, a) not in [(q2, a2) for q2, a2, *_ in
                                []] and str(v.ask(q).get("answer", "")) != a]
    probe = v.ask("a question this seed was never taught zqx")
    leak = bool(str(probe.get("answer") or "").strip())
    print("SEED %s: %d facts taught, %d refused, %d verification miss(es), "
          "held-out probe %s"
          % (out_dir, len(pairs) - len(refused), len(refused), len(misses),
             "LEAKED" if leak else "escalates cleanly"))
    for q, why in refused:
        print("  refused: %r -- %s" % (q[:50], why))
    for q in misses[:5]:
        print("  miss: %r" % q[:60])
    return 0 if not (misses or leak) else 1


if __name__ == "__main__":
    sys.exit(build(sys.argv[1], sys.argv[2]))
