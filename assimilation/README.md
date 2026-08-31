# assimilation/ -- Unicron vs Qwen3.5-0.8B, start to finish

Three commands, run from the repo root. A private venv appears at
`assimilation/.venv` on first run; your system Python is never touched and no
Hugging Face account or token is ever needed (the weights are public and the
download is anonymous by construction).

Linux / macOS:

    ./assimilation/assimilate.sh --eval    # 1. download + assimilate + MEASURE
    ./assimilation/install.sh              # 2. leCore INTO the assimilated model
                                           #    (-> work/galvatron = the point)
    ./assimilation/chat.sh --both --galvatron   # 3. untouched vs Galvatron, timed

Windows (same flags, same behaviour):

    assimilation\assimilate.bat --eval
    assimilation\chat.bat --both
    assimilation\chat.bat

Layout after a run:

    assimilation/work/original/       the untouched download
    assimilation/work/assimilated/    the Unicron output (same tensor names/shapes,
                                      loads exactly like the original) + per-shard
                                      *.unicron_report.json rank reports
                                      + *.lecore.safetensors -- the FACTORED form:
                                      each filtered layer as its thin (U,V) pair.
                                      This is the model's true information size
                                      (2x smaller on the rehearsal subject; the
                                      dense file stays full-shape only because
                                      transformers/llama.cpp demand the original
                                      architecture). Loads via leCore's
                                      unicron_reconstruct; a transformers shim
                                      that RUNS the factored form is the planned
                                      next step.

What "assimilate" does and why: see `holographic_unicron.assimilate_model` --
Marchenko-Pastur filtering keeps each projection's learned spectral outliers and
drops the still-random bulk; embeddings/norms are policy-skipped; layers whose
outliers carry <1% of energy are guarded (random != useless, measured).

The honesty contract: `--eval` prints perplexity before vs after. Until that (or
your own harness) has run, the assimilated model is an UNVERIFIED claim -- the
report says so in as many words. A bad delta is a result worth keeping, not a
failed run.

Nothing here touches the leCore engine's dependencies: torch/transformers live
only in this folder's venv, as the measurement-and-runtime instrument. The
engine that rewrites the weights remains NumPy + stdlib.

## The chat has a mind now (external memory, cp86)

Every chat mode boots a leCore memory beside the model (default
`assimilation/work/chat_memory`, or `--memory DIR`, or `$LECORE_PARTITION`;
`--no-memory` for the old raw behaviour). Taught knowledge answers FIRST with
provenance -- `[MEMORY taught] ...` -- before any tokens are spent; unknowns fall
through to the model honestly. In-chat verbs: `teach: q = a` (stored and saved
immediately), `wrong: q` / `veto: q` (tombstoned across restarts). This is the
same job a harness like OpenWebUI's store performs, played by the engine itself:
numpy + stdlib, no new dependencies in the venv, continuity between sessions
verified (teach -> restart -> answers; veto -> restart -> stays dead).
