# Where to start (the map for 50+ documents)

Pick your seat; read two documents, not fifty.

## "I just want to try it" (5 minutes)
1. **README.md** — the receipts table (every claim = a number + the command that
   reproduces it) and the 60-second quickstart.
2. Run `run.bat` / `./run.sh` → http://127.0.0.1:7860 → say **`commands`**.

## "I'm evaluating whether this is real" (30 minutes)
1. **README.md** → run the three commands in the receipts table yourself
   (`tools/bench_longmem.py`, `tools/bench_ladder.py`, `tools/swarm_audit.py`).
2. **docs/WHATS_NEW.md** — every recent feature with one copy-paste example.
3. **docs/ABLATIONS.md** — the kept negatives. Slop hides failures; this doesn't.

## "I operate openzoo.fun (or any hosted deployment)"
1. **docs/OPENZOO_OPERATIONS.md** — the operator manual: all 26 hosted tools with
   descriptions generated from source, env vars, boot, the bundle, security
   boundaries, cost levers, adoption steps, troubleshooting.
2. **docs/OPENZOO_INTEGRATION.md** — the why behind each lever, with measured numbers.

## "I'm building on the engine locally"
1. **docs/WHATS_NEW.md** → then `commands` in the chat, and `m.tool_find("<task>")`
   / `m.explain("<topic>")` in Python — the engine explains itself from its own
   docstrings (2,200+ faculties indexed).
2. **REFERENCE.md** (generated) and **docs/FEATURE_GUIDE.md** for depth.
3. **docs/UNICRON_INSTALL.md** for installing leCore *inside* a model
   (AlphaEdit-protected; qwen3.5:0.8b runbook included).

## "I want the history and the proofs"
- **CHANGELOG_lever7.md** — 80 checkpoint entries, every number with its command.
- **docs/LEVER7.md** — the arc in one page. **docs/BENCHMARKS.md** — the tables.
- **docs/RELEASE_CHECKLIST.md** — what's verified, what awaits a real box, and every
  known limitation stated in print.
