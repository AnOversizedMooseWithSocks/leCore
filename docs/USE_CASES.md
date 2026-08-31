# Three swarms, one substrate: use cases that run

Each scenario below is a `# guide-check` block: it runs verbatim in CI
(`tests/test_guide_examples.py`), so what this page says the engine supports, the engine
demonstrably supports. The doors are thin compositions of rails described in
`WHY_A_HOLOGRAPHIC_VM.md`; the pin is `tests/test_mcp_server.py::test_swarm_use_cases_sweep_125`.

## 1. A customer-service swarm that learns from its humans

Agents answer what memory knows, **escalate honestly** what it does not, and when a human
resolves the escalation the answer is taught with the human's name in its provenance and
propagated — so every other agent in the swarm serves it next time without a model call.

```python
# guide-check
import lecore, os, tempfile
t = tempfile.mkdtemp()
agent = lecore.UnifiedMind(dim=256, seed=0)
agent.teach("how do I reset my password", "use the link on the login page; it expires in 15 minutes")
assert agent.serve("how do I reset my password")["via"] == "memory"          # answered, no model
r = agent.serve("can I transfer my subscription to another account")
assert r["via"] == "escalate"                                                # to a human, honestly
assert [e["question"] for e in agent.escalations()] == ["can I transfer my subscription to another account"]
done = agent.resolve("can I transfer my subscription to another account",
                     "yes -- support moves it once both accounts are verified",
                     by="jane@support", propagate=os.path.join(t, "bundle"))
assert done["taught"] and done["cleared"] and agent.escalations() == []
agent.commons_pool([os.path.join(t, "bundle")], os.path.join(t, "commons"))
other = lecore.UnifiedMind(dim=256, seed=0)                                  # another agent in the swarm
other.memory_import(os.path.join(t, "commons"))
assert other.serve("can I transfer my subscription to another account")["via"] == "memory"
```

The ledger (`escalations()`) is the swarm's honest account of what it does not yet know;
`wisdom`/`bequeath` carry lessons the same way facts travel here.

## 2. A development swarm with one understanding of the codebase

Multiple streams of agents work concurrently. The codebase is studied **once**: each file's
digest is taught as a fact fingerprinted to that file, so it persists in the partition
across restarts and sessions, every agent that boots the partition already knows the code,
and an edit marks *exactly* the affected facts stale — the swarm re-syncs only those.

```python
# guide-check
import lecore, os, tempfile, time
t = tempfile.mkdtemp()
repo = os.path.join(t, "repo", "pkg"); os.makedirs(repo)
open(os.path.join(repo, "billing.py"), "w").write('"""Billing: computes invoices from metered usage and applies the customer plan discount."""\n')
open(os.path.join(repo, "auth.py"), "w").write('"""Auth: verifies session tokens and refreshes them shortly before they expire."""\n')
dev1 = lecore.UnifiedMind(dim=256, seed=0)
assert dev1.codebase_sync(os.path.join(t, "repo"))["taught"] == 2            # studied once
dev1.learning_save(os.path.join(t, "partition"))
dev2 = lecore.UnifiedMind(dim=256, seed=0)                                   # a second agent, after a restart
dev2.learning_load(os.path.join(t, "partition"))
assert dev2.ask("what does pkg/billing.py do")["tier"] == "T0"               # already knows the code
time.sleep(1.1)
open(os.path.join(repo, "auth.py"), "a").write("def revoke(token):\n    return None\n")
assert dev2.stale_facts(root=os.path.join(t, "repo"))["stale"] == ["what does pkg/auth.py do"]
s = dev2.codebase_sync(os.path.join(t, "repo"), only_stale=True)
assert s["taught"] == 1 and s["skipped"] == 1                                # re-synced only the changed file
```

For agents on different machines the same partition rides `memory_export`/`memory_import`,
and `study()` over MCP gives any connected model the tree with citations.

## 3. A lab: roles on one bus, one memory, each focused on its task

Some agents answer questions, some research what was discussed, some write reports when
findings arrive, some run experiments on data as it lands. Each is a **role**: a focused
handler subscribed to a topic on the shared bus, holding the same mind — what one learns,
all recall; the topic keeps each one on task; roles chain without knowing each other.

```python
# guide-check
import lecore, os, tempfile
lab = lecore.UnifiedMind(dim=256, seed=0)
corpus = tempfile.mkdtemp()
open(os.path.join(corpus, "paper.md"), "w").write("# Results\n\n" + "The catalyst yield rises with temperature up to 340 kelvin then falls sharply. " * 3)
study = lab.study(corpus)
lab.role("answerer", "chat", lambda p, mind: mind.serve(str(p)), emit="answers")
lab.role("researcher", "chat", lambda p, mind: study["ask"](str(p)), emit="findings")
lab.role("reporter", "findings", lambda p, mind: {"report": p["chunks"][0][:80] if p.get("answerable") else "no evidence", "cited": p.get("citations", [])}, emit="reports")
lab.role("experimenter", "data", lambda p, mind: mind.table_analyze([{"v": float(x)} for x in p["series"]], "v", tasks=("regimes",)), emit="reports")
lab.teach("who runs the lab", "dr. moose")
bus = lab.bus()
bus.publish("chat", "who runs the lab")                                      # answerer serves it from memory
bus.publish("chat", "what happens to the catalyst yield above 340 kelvin")   # researcher -> findings -> reporter
bus.publish("data", {"series": [1] * 18 + [9] * 18})                          # experimenter -> reports
handled = {r["name"]: r["handled"] for r in lab.roles()}
assert handled == {"answerer": 2, "researcher": 2, "reporter": 2, "experimenter": 1}
assert len(bus.history("reports")) == 3
```

Across machines the bus fans out with `distributed_bus`; compute fans out with `farm`.
