# CI green-up round 3 -- what's in this zip and what needs your tree

The failing commit (p16_unicron, devicerun, gdnruntime) isn't on any pushed branch, so two of
the four fixes are files you drop in, and two run against your working tree.

## 1. tests/test_routing_seed_canonical.py -- FIXED, drop in (verified on main)
The 400..700 band was another absolute-count pin that the corpus outgrew (703 modules + 18
asks = 721 seed rows) -- the same disease as the exam bars last round. The ceiling is now
DERIVED from the on-disk module count (n_modules + 60), so it never rots again, while an
18k md-window bloat still overshoots it by ~25x. Passes on main (638 rows / 619 modules) and
passes your branch's numbers by arithmetic (721 <= 763).

## 2. tests/test_knowledge_index_corpus.py -- already fixed last round
The failing run predates the test-file update; main's head already has the recalibrated pins
(require-median 2.5 / fused-top1 5). If your branch forked before that landed, cherry-pick or
drop in the copy from the previous zip. No new change here.

## 3. tools/fix_p16_unicron.py -- RUN IN YOUR TREE (dry-run first)
Fixes both test_unified_split failures mechanically:
  * DEDUPE: the dupes are inside p16 itself (owner == module in the failure tuples). Python
    keeps the LAST definition in a class body, so the first unicron_runtime / unicron_vm_install
    / unicron_imbue bodies are dead code that has never executed -- the tool deletes exactly
    those, behaviour-neutral by construction.
  * SPLIT: moves the tail methods (past the ~1600-line boundary, on a method edge) into
    holographic_unified_p17_unicron2.py / _UnifiedPart17, mirrors the import in the shim, and
    inserts Part17 into the UnifiedMind bases tuple. unified_sources() derives from live bases,
    so no third edit. Base order is irrelevant while no name is duplicated (the test pins this).
  * CRLF-preserving byte writes, idempotent, --dry-run prints the full plan first.
Verified end-to-end against a synthetic p16 with your exact failure shape + main's real shim
wiring pattern. Afterwards: pytest tests/test_unified_split.py -q, then the usual audits --
the split moves method bodies, so run skill_lint (delete /tmp/lecore_lint_memo.json first) and
reachability_audit before trusting green.

## 4. test_gpu_fallback_equivalence -- NEEDS YOUR HANDS (can't be done blind)
devicerun + gdnruntime now import the backend; the test demands each new wired module get a
REAL fallback-equivalence test, and its philosophy is explicit: assert the documented result,
not merely no-raise. Writing those asserts requires the modules' contracts, which only exist
in your tree. The mechanical half:

    covered = {"rendering/holographic_shader", "simulation_and_physics/holographic_fluid",
               "simulation_and_physics/holographic_memoryhome",
               "unified/holographic_unified_p12_proc_texture",
               "io_and_interop/holographic_devicerun",
               "io_and_interop/holographic_gdnruntime"}

then one test per module, shaped like the existing four (exercised through its FACULTY, with
use_gpu(True) returning False as the precondition), asserting the numeric contract from each
module's docstring/_selftest. Rule-0 the faculty names first (find_capability "run on device",
"gdn runtime", etc.) so the tests go through the front door.

## 5. The canceled shard (image 2)
full-suite(3) died at 19m51s -- that's the 20-minute timeout-minutes budget, not a new failure.
The F's it collected are the four above; once they're green the shard should fit the budget
again. If it doesn't, that's a separate conversation about the shard partition, not this fix.
