# Test fixtures

## smollm2_slice.safetensors  (NOT in the repo -- 33 MB)

A slice of SmolLM2-135M: 4 layers, 4096 vocabulary rows, 38 tensors, 16.5M
parameters, BF16, tied embeddings, plain (ungated) attention, no qk-norm.
Produced with:

    python tools/make_test_model.py <smollm2-135m-dir> <out-dir> 4 4096

It is deliberately NOT committed: 33 MB of weights does not belong in a source
tree, and it regenerates in one command from a public Apache-2.0 checkpoint.
Drop it here (with `smollm2_slice.config.json`, which IS committed) to run the
trained-weights tests.

## Why two fixtures

`tools/build_mini_qwen.py` generates STRUCTURE -- real tensor names rooted at
`model.language_model.`, the 24-layer linear/full attention pattern, a vision
tower, tied embeddings, added tokens above the plain vocabulary, BF16 on disk.
It caught eight structural defects that would otherwise have cost a user a test
cycle each.

The SmolLM2 slice supplies TRAINED STATISTICS -- real spectra, real heavy tails,
real activation geometry. This is not interchangeable with the synthetic one,
and the difference was measured rather than assumed:

    bake          random weights        trained weights
    vsa_bind      REVERTED  +3.1%       KEPT  +0.0001%
    boot_record   REVERTED +14.3%       KEPT  +0.0001%

A circuit installed into trained weights costs essentially nothing; the same
circuit in random weights is pure damage. Testing on either fixture alone gives
a confident and wrong answer about the other.
