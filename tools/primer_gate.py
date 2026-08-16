#!/usr/bin/env python3
"""Ship gate for docs/PRIMER_for_openzoo_ai.md: the zip must not rebuild unless this exits 0.

WHY: a document that names measured numbers, live transcripts, and prior art is a contract
like any generated artifact -- an edit that silently drops a receipt hash or a citation is a
regression. Needles are checked against WHITESPACE-NORMALIZED text so a reflowed line can
never fail the gate, and the gate's exit code is what the ship pipeline conditions on."""
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, so the
# gate runs from anywhere -- its first real run blocked the ship on exactly this line's absence

t = pathlib.Path("docs/PRIMER_for_openzoo_ai.md").read_text()
flat = " ".join(t.split())

NEEDLES = [
    "external_write -> external_read cosine: 1.000",
    "46b57dd6d42b2b793378f9b32c63f01a0a5eecc1e927262163241369848637a3",
    "0.935", "0.951", "0.767", "182,010", "116 GFLOP/s", "121 ns", "90 GB/s",
    "recall 1.000 @ 9.7", "27.1 ms", "the only 1.000 in the table",
    "ROME 2022", "SISA (Bourtoule", "fractional power", "Lehman & Stanley",
    "zkLLM 2024", "TOPLOC, DiFR", "RaBitQ (SIGMOD 2024)", "deterministic worst-case",
    "Tribase/TRIM", "hierarchical_pack", "bundle_capacity", "unicron_swarm_mind",
    "quantization grain", "24/24", "8e-17", "candy-wrapper",
    "the snake eats its tail", "keeping the doormat", "ledger of its own refuted ideas",
    "470", "171",
    "svgf_denoise", "B-rep boolean", "hertz + damping ratio", "2.62", "1.25 ms",
    "render_critique_loop", "lecore_invoke", "humanoid",
    "zoo_ask", "counterfactual", "savesVsDirect", "one verb", "payment happened",
    "right-to-forget as one call",
    "$LEOS", "the data never moves", "revocation that actually executes",
    "the receipt is the product", "royalty",
    "single-sided", "ask ladder", "buy wall", "triangular arbitrage",
    "Store the rule, not the dump", "$TOKEN",
]
SCARS = ["Part I", "Part II", "Part III", "Part IV", "This part exists",
         "Also adjacent, and named", "The updated one-sentence"]

missing = [n for n in NEEDLES if n not in flat]
present_scars = [s for s in SCARS if s in t]
if missing or present_scars:
    print("GATE FAILED. missing=%r scars=%r" % (missing, present_scars))
    sys.exit(1)

import lecore
m = lecore.UnifiedMind(dim=64, seed=0)
VERBS = ["bundle_capacity", "hierarchical_pack", "recursive_factor", "unicron_vm_install",
         "unicron_swarm_mind", "advise_scale", "cosamp_recall", "void_map",
         "route_or_abstain", "tiered_memory", "cold_store", "codec_place", "machine_map",
         "memory_mountain", "local_pool", "shared_workspace", "farm", "wgsl_bind_batch",
         "vm_decode_plan"]
dead = [v for v in VERBS if not hasattr(m, v)]
if dead:
    print("GATE FAILED. dead verbs=%r" % dead)
    sys.exit(1)
print("primer gate: %d needles, %d scars absent, %d verbs live -- GREEN" %
      (len(NEEDLES), len(SCARS), len(VERBS)))
