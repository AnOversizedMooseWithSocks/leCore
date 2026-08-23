# shader_retrieval — the GLSL / retrieval arc

Run from the repo root: `PYTHONPATH=. python3 research/shader_retrieval/<script>.py`

## Hardware benchmark (the one to run on a real GPU)

`bench_gpu.py` — standalone (numpy + moderngl). **Refuses to print a timing table on a software
rasteriser**, because every timing figure this arc produced came from Mesa llvmpipe, which is a CPU.
Correctness runs regardless; a kernel that fails correctness prints no timing at all.

```sh
pip install numpy moderngl
python3 research/shader_retrieval/bench_gpu.py --json results_<machine>.json
```

## Layout

- `glsl_*.py` — one kernel each, with its differential test against an exact NumPy reference.
  The verified shader **sources** are also carried inside the engine at
  `holographic/io_and_interop/holographic_glslkernels.py`, discoverable via `mind.glsl_kernel(name)`.
- `hard_corpus.py` — the stratified repo corpus fixture. It **asserts family coverage** rather than
  taking a target count, because truncating a sorted file list is a biased sample that looks
  complete from the inside.
- `bench_docstring.py` — the fully held-out benchmark (docstring → code). This is the default for
  retrieval claims; the self-sampled corpus measures lookup, not retrieval.
- `bench_revdict.py` — reverse dictionary on the vendored 144k-word dictionary.
- `p01_*.py`, `p19_*.py` — retriever/stage-2/ambiguity harnesses.
- `make_search_page.py`, `make_webgl_*.py`, `make_shards.py` — artifact generators.

## Reading the results

Every number these produce is in `docs/NOTES_concepts.md` **with its kept negatives**. Notably:
the scatter scorer gives up bit-reproducibility, diffusion conserves heat only to f32, PBD is
Jacobi and not comparable to a sequential sweep, raster byte-exactness is conditional on the
scene's rounding margin, and **no GPU timing exists yet for any of them**.
