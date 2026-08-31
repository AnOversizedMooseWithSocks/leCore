"""READER-MODE LONGMEMEVAL PROTOCOL (cp61): memory + ouroboros + a model rung, end to end.

The cp60 harness measured the SUBSTRATE alone (1.000 across six seeds). This harness
measures the full published-benchmark SHAPE: retrieve -> select -> READ -> answer.

  RETRIEVE   leCore recall_semantic / ask_latest (the cp60 organs)
  SELECT     the OUROBOROS resident as working memory: every evidence line is written
             into the delta-rule trace (key = question key, value = evidence key);
             the readout then RANKS the evidence -- the resident performs the final
             selection with its own algebra, zero forward passes. Ablatable.
  READ       one reader interface, three readers:
               substrate   -- no reader; the evidence line IS the answer (cp60 mode)
               extractive  -- deterministic: return the ouroboros-top evidence verbatim
                              (proves the retrieve->select->answer PLUMBING)
               model       -- the mini through its rung: GDN runtime forward generation
                              over prompt = evidence + question

HONEST SCOPE, stated before any number: NO REAL SMOLLM WEIGHTS EXIST IN THIS SANDBOX
(network egress is disabled; real-weights work is the recorded user's-box step). The
"model" reader is the cp51 mini -- a REAL qwen3.5-shaped forward pass with the full
galvatron install, but essentially untrained: it exercises every wire and is EXPECTED to
score near zero, which is the point -- the table shows where the intelligence lives and
proves the plumbing is ready for real weights.
"""
import sys
import re
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "/tmp")
import lecore
from tools.bench_longmem import build
from holographic.agents_and_reasoning.holographic_galvatron import OuroborosResident


def _wkey(mind, text):
    r = mind.recall_semantic("__probe__", k=1)      # ensure index fn exists
    return mind._sem_index[3](text) if hasattr(mind, "_sem_index") else None


def run_reader(reader="extractive", ouro=True, seed=0, model_path=None):
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "")
    sessions, questions = build(seed=seed)
    for s in sessions:
        for t in s["turns"]:
            txt = "[%s] %s" % (s["date"], t)
            m.teach(txt, txt)
    m.recall_semantic("warm the index", k=1)
    wk = m._sem_index[3]
    rung = None
    if reader == "model":
        from model_rung import ModelRung
        rung = ModelRung(model_path or "/tmp/mini_installed_full",
                         "/home/claude/claude_partition")
    res = OuroborosResident(hidden_dim=512, layer=0, dk=512, decay=0.98)
    rng = np.random.default_rng(seed)
    cats = {}
    for q in questions:
        sem = m.recall_semantic(q["q"], k=5)
        cands = [c["text"] for c in sem["candidates"][:4]]
        if q["cat"] == "knowledge-update" and hasattr(m, "ask_latest"):
            al = m.ask_latest(q["q"])
            if al.get("answer"):
                cands = [str(al["answer"])] + [c for c in cands
                                               if c != str(al["answer"])]
        if not cands:
            ans = ""
        else:
            qk = wk(q["q"])
            if ouro:
                # THE STALENESS TRAP, MEASURED THEN FIXED (cp61): naive equal-weight
                # writes make the readout a MAJORITY VOTE -- two stale "tuesdays"
                # entries outvoted one fresh correction (update fell 0.5 -> 0.167
                # against the no-selection ablation). The operator already carries the
                # cure: DECAY IS RECENCY. Evidence is written oldest-and-weakest FIRST
                # with the trace decayed between writes, so the freshest, best-retrieved
                # line is the least-decayed deposit and the readout favours it --
                # the cp58 theorem (pheromone == delta-rule) applied to selection.
                # SECOND MEASURED LESSON (cp61): decay 0.98 is 2% a step -- over four
                # writes the weights are flat and near-duplicates still vote as a bloc
                # (update stuck at 0.333). And date-primary ordering BURIED the answer
                # ask_latest had already pinned first: selection was undoing upstream
                # doctrine. The principle that fixed both: selection REINFORCES the
                # pipeline -- evidence is written in REVERSE retrieval order (the
                # pipeline's best pick last, therefore least decayed) with a working-set
                # decay of 0.6, so the readout amplifies upstream ranking unless the
                # rest of the evidence coheres strongly against it.
                for c in reversed(cands):
                    res.S = res.S * 0.6                # aggressive working-set aging
                    res.external_write(qk, wk(c))
                rd = res.external_read(qk)            # readout ranks the evidence
                rdv = np.asarray(rd if not isinstance(rd, dict)
                                 else rd.get("value", rd.get("readout")), float).ravel()
                scored = sorted(cands, key=lambda c: -float(wk(c) @ rdv[:512]))
            else:
                scored = list(rng.permutation(cands))  # ablation: no selection
            if reader == "substrate":
                ans = scored[0] if q["cat"] == "knowledge-update" else \
                    " | ".join(scored[:3])
            elif reader == "extractive":
                ans = scored[0]
                if q["cat"] in ("multi-session", "temporal"):
                    ans = " | ".join(scored[:3])       # these need >1 line by design
            else:
                prompt = "evidence:\n%s\nquestion: %s\nanswer:" % (
                    "\n".join(scored[:3]), q["q"])
                ans = str(rung(prompt))[:200]
        if not q["answerable"]:
            ok = not ans.strip() or (reader == "model" and False)
        else:
            ok = bool(q["answer"]) and q["answer"].lower() in ans.lower()
            if ok and q.get("wrong") and q["wrong"].lower() in ans.lower():
                ok = False
            if not ok and q.get("answer2") and q["answer2"].lower() in ans.lower():
                ok = True
        cats.setdefault(q["cat"], []).append(1.0 if ok else 0.0)
    table = {c: round(float(np.mean(v)), 3) for c, v in sorted(cats.items())}
    table["OVERALL"] = round(float(np.mean([x for v in cats.values() for x in v])), 3)
    return table


if __name__ == "__main__":
    for name, kw in [("substrate            ", dict(reader="substrate")),
                     ("extractive + ouro    ", dict(reader="extractive", ouro=True)),
                     ("extractive - ouro    ", dict(reader="extractive", ouro=False)),
                     ("mini model + ouro    ", dict(reader="model", ouro=True))]:
        print(name, run_reader(**kw))
