# Release checklist and known limitations (v0.7.77, lever 7 / checkpoint 77)

## Verified in this release (sandbox, deterministic)
- Ten CI gates green; full shimmed suite green; standard bench pinned 0.75 / 1.00 /
  0.25; LongMemEval-protocol 1.000 across six categories and six seeds, no model.
- All module selftests green: apilearn (live local HTTP loop), runtimerung (opt-outs),
  attribution (deterministic addresses), splatmem (abstention + cliff), swarm audit
  (12 groups, 0 import failures, 20x5 matrix, 0 unintended gaps).
- The shipped memory bundle re-distilled from the current partition: 62 generic
  entries, every exclusion itemized by reason, leakage audit 0/4.
- Chat front door headless-verified: teach/veto/workspaces/artifacts/memory slots/
  void loop/api verbs/commands cheatsheet.

## To run on a real box before calling it fully proven (in order)
1. `python -m pytest` on the full 679-file suite (the sandbox uses tools/ci_sim.py, a
   shim -- it says so in its own output).
2. `python tools/unicron_preflight.py <qwen3.5-0.8b dir>` -- must be green -- then the
   install runbook in docs/UNICRON_INSTALL.md (AlphaEdit protection on; expect
   preservation_after ~0, galvatron AUDIT 4/4), then measure the two things a sandbox
   cannot: efficacy generalization and perplexity retention.
3. Official LongMemEval-S with a real reader model (`tools/bench_longmem_reader.py`
   is the harness; the protocol run is at 1.000, the official set needs real weights).
4. `lean backtest_is_honest.lean` -- the exported theorem; external Lean is the
   authority.

## Known limitations, stated rather than hidden
- Raw `ask()` with a model rung attached serves the rung's cached output ungrounded;
  `ask_grounded()` is the sanctioned door (engine floor since cp68) and what the chat
  and hosted zoo_ask use. Migrating raw ask() is deliberate future work.
- Fuzzy-path answers carry session tags but not payload-level provenance across
  sessions (recorded since cp53).
- plan_warm's goal key is not paraphrase-robust.
- Interest pheromone does not persist hosted-side.
- The installer's fact keys are embedding-derived; runtime lens readback of installed
  facts needs ROME-style forward-derived keys (--forward-keys, designed cp70, not yet
  built) -- on trained weights the two coincide.
- EvoEdit-style dynamic null-space projector (ACL 2026) is the known upgrade path
  beyond the static AlphaEdit projector now shipped.
- misc/ sits at 151 modules against a soft budget of 150 (structure_audit note);
  housekeeping, not behavior.
