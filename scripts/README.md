# leCore assimilation scripts

Drop this `scripts/` folder as a SIBLING of the `leCore/` repo folder
(lecore_paths.py resolves everything from that layout; run it alone to verify):

    <parent>/leCore/          the repository
    <parent>/scripts/         this folder
        nomic_text/           config.json, model.safetensors, vocab.txt
        qwen3.5_0.8b/         config.json, tokenizer.json, model.safetensors-...
        smol/                 SmolLM2 (optional)
        .knowledge_cache.json created/extended on first index run

## the one command

    python3 run_all.py            # lint -> paths -> config probe -> router floor ->
                                  # distill map -> census -> VSA analysis, cached per stage
    python3 run_all.py --list     # what would run and why
    python3 run_all.py --report   # rebuild ASSIMILATION_REPORT.md, run nothing

## the pieces (each also runs standalone)
    lecore_paths.py           layout truth; loud errors on missing files
    lint_scripts.py           ERRORs halt (alias shadowing, syntax); WARNs advise
    qwen_config_probe.py      architecture + budget + tokenizer, NO weights needed
    distill_router.py         the token-table routing floor (N30)
    distill_map.py            is the encoder a linear correction? (N31; needs the cache)
    llm_census.py             families, duplicates, q8/q4, embedding-table quant
    model_analysis_program.py the VSA program: regime -> demux -> scout -> frontier -> codecs
    knowledge_index.py        builds the content-addressed embedding cache + routing suite
    nomic_forward.py          the NumPy BERT-style forward pass (imported by the index)
    lecore_agent.py           the routing agent (N28 wiring pending)
    align_models.py, smol_*  , nomic_vision_census.py -- follow-up probes

## CI
`.github/workflows/semantic-coverage.yml` goes in the REPO (not here). Prerequisite:
commit knowledge_index.py, nomic_forward.py, lint_scripts.py to tools/semantic/ in the
repo and set NOMIC_WEIGHTS_URL / NOMIC_VOCAB_URL repo variables. First run: press the
"Run workflow" button (the cold embed populates the Actions cache, ~1-2 h, once).

Discipline that applies everywhere: PYTHONHASHSEED=0; single-threaded BLAS for
bit-identical sums; failures are never cached; every claim ships with its baseline.
