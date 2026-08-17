# assimilation/ -- Unicron vs Qwen3.5-0.8B, start to finish

Three commands, run from the repo root. A private venv appears at
`assimilation/.venv` on first run; your system Python is never touched and no
Hugging Face account or token is ever needed (the weights are public and the
download is anonymous by construction).

Linux / macOS:

    ./assimilation/assimilate.sh --eval    # 1. download + assimilate + MEASURE
    ./assimilation/chat.sh --both          # 2. same prompt to both models, side by side
    ./assimilation/chat.sh                 # 3. just talk to the assimilated one

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
