# Qwen3.5 integration status

Qwen loading, tokenizer/config parsing, the text-only NumPy runtime, and
diagnostics are supported integration work. The two checkpoint-changing paths
remain experiments and are opt-in.

## Safe default: download the untouched checkpoint

Linux/macOS:

```bash
./assimilation/assimilate.sh
```

Windows:

```bat
assimilation\assimilate.bat
```

This downloads public `Qwen/Qwen3.5-0.8B` weights into
`assimilation/work/original` and stops. It does not edit tensors.

## Experimental layer-prepending installer

The newer design prepends blank layers and installs leCore facilities into the
new layers, reserved recurrent directions, and tokenizer-unused rows. It does
not spectrally filter the original tensors. It is gated until a complete real
Qwen acceptance run succeeds:

```bash
./assimilation/install.sh --experimental assimilation/work/original assimilation/work/installed
```

The acknowledgement is intentional. The last recorded full-model attempt
reached checkpoint emission but exhausted memory during verification; fixture
success is not a substitute for a real checkpoint result.

Generate the preregistered ilxyr acceptance project with
`experiments/qwen35_acceptance/generate.py`. Its runner checks reference logits,
thousands of paired token positions with block-bootstrap confidence, peak
memory, disk reload, official text generation, and official image-input
execution.

## Research-only spectral control

The original Unicron spectral experiment is retained for reproduction and
negative controls only:

```bash
./assimilation/assimilate.sh --research-spectral --eval --no-imbue
```

It is disabled by default because the real Qwen study did not demonstrate a
benefit: 18 of 265 eligible tensors changed, the assimilated checkpoint
regressed, repair reverted most changes, and the remaining apparent gain was
inside an underpowered confidence interval. Do not describe this path as a
proven optimization or compression result.

## Scope boundary

leCore's runtime executes the text stack. Vision and MTP tensors pass through
unchanged, but that is not a vision test. Any checkpoint intended for ordinary
use must also pass the official Transformers image-text smoke test in the ilxyr
acceptance contract.
