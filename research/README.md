# research/ — one folder per closed or open research arc

Everything here is a **one-off experiment harness**, not part of the `holographic/` engine and not
imported by it. The engine never depends on this directory; the audits
(`tools/reachability_audit.py`, `tools/catalog_gaps.py`) confirm that.

These lived in the repo root for a long time — 74 loose `.py` files sitting beside `setup.py` and
`lecore.py`, which made it impossible to tell an entry point from a scratch script. They are
organised by the arc that produced them, following the precedent `scripts/README.md` already set.

## Running them

From the **repo root**, with the package importable:

```sh
PYTHONPATH=. python3 research/<arc>/<script>.py
```

Scripts inside an arc import each other by bare name (`import hard_corpus`), which works because
Python puts the *script's own directory* on `sys.path`. `PYTHONPATH=.` supplies the `holographic`
package. Both are needed; neither alone is enough.

## The arcs

| folder | what it is | status |
|---|---|---|
| `shader_retrieval/` | GLSL/WebGL2 kernels, BM25 + retrieval-policy benchmarks, corpus fixtures, the search-page generators, `bench_gpu.py` | open — GPU timing unmeasured |
| `capacity_laws/` | capacity/tiling/nesting sweeps, Monte-Carlo probes, compressibility nulls | closed |
| `program_induction/` | program induction, RNN probes, VM triggers, declarative ladders | closed |
| `demos/` | creature and material demos | closed |

## What is NOT here

`tools/` (standing audits and the delivery zip), `benchmarks/` (maintained benchmark suites,
including BEIR), `tests/` (pytest), and `scripts/` (the closed LLM-assimilation arc, which has its
own README). Those are all maintained; this directory is where experiments live.

**The verdicts these harnesses produced live in `docs/NOTES_concepts.md`, not here.** A script whose
result matters has its number recorded there with its kept negatives; the script is the provenance,
not the finding.
