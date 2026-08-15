# Qwen runtime right-sizing benchmark

`tools/benchmark_qwen_runtime.py` measures the exact resumed-forward and
full-vocabulary negative-log-likelihood path used by the Qwen acceptance run.
It starts a fresh process for every chunk size, records peak RSS and throughput,
checks per-token loss parity, and emits a checksummed JSON evidence document.

Run the benchmark with the same frozen CPU environment as the formal project:

```bash
python3 -m venv .venv-qwen-acceptance
. .venv-qwen-acceptance/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experiments/qwen35_acceptance/requirements-cpu.txt
```

Keeping the benchmark and formal runner on the exact Torch 2.11/Torchvision
0.26 pair prevents an environment change from masquerading as a chunk-size or
backend speedup.

Run it against the larger, emitted checkpoint when possible:

```bash
python tools/benchmark_qwen_runtime.py \
  /models/qwen-installed /corpora/evaluation.txt \
  --chunk-sizes 64,128,256 \
  --tokens 256 \
  --gdn-backend c \
  --concurrent-runtimes 2 \
  --full-run-peak-mib 12000 \
  --ram-hourly-usd 32=0.40,64=0.80,128=1.60 \
  --fixed-overhead-minutes 20 \
  --output qwen-runtime-benchmark.json
```

Hourly prices are deliberately supplied at run time: region, architecture,
purchase model, and AWS pricing change. The report recommends the fastest chunk
whose per-token losses match the smallest chunk within the recorded tolerance
(default: `1e-6` nats, below the acceptance logit tolerance). It
then projects memory conservatively as:

```text
max(single evaluator peak * concurrent evaluators, full-run observed peak)
  * headroom factor
  + system reserve
```

The first fitting class among 32, 64, and 128 GiB is reported, together with the
margin for every class and a cost projection using the supplied current rates.
The report includes the Git commit, source hashes, corpus hash, model manifest,
CPU, thread controls, and all raw measurements so it can be archived with ilxyr.
For the compiled recurrence, each trial also records whether `c` actually became
active or was refused by the first-call parity gate. Run once with `numpy` and
once with `c` to produce a like-for-like backend comparison.

The generator will not increase its default 128-token chunk without this
evidence. Bind the report into the next preregistered ilxyr project with:

```bash
python experiments/qwen35_acceptance/generate.py \
  /models/qwen /corpora/install.txt /corpora/evaluation.txt /experiments/qwen-v4 \
  --benchmark-report qwen-runtime-benchmark.json
```

The report hash becomes part of the runner-policy digest and experiment ID.
Both evaluator processes then use one parent-frozen chunk schedule; they do not
independently reselect resume boundaries from fluctuating free-memory readings.
Archive the report itself with the v4 project/results: its schema is
`lecore.qwen_runtime_benchmark.v1`, while `benchmark_report_sha256` in the
generated runner policy is the immutable link from the project to those raw
measurements. A digest without its matching report is not sufficient evidence.

This benchmark covers the evaluation runtime only. It must not be used alone to
claim that an entire acceptance run fits a smaller machine. Pass the largest
observed peak from a completed full run with `--full-run-peak-mib`, and validate
the selected configuration with the miniature end-to-end fixture before the
next preregistered formal run.
