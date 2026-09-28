# Evidence: the CLM panel swarm (2026-09-26)

These are the raw scripts and outputs behind the contrastive backlog, `docs/BACKLOG_contrastive.md`. That file is
a local working note: backlogs are gitignored by owner directive, so it is not repo content. The question was how
the ideas in Contrastive-LM's CLM-8B, JEV's typed decisions and NOOA's verified steps can improve leCore's reflex
arc and substrate.

## How it was run

- **The panel.** Four workers shared one leCore service on port 8080. The memory partition was mounted, with 516
  taught rows, at generation `state-20260926-192013Z`.
- **Seats.** Each worker held two or three expert-panel seats and noted them with `panel_seat`. The seats are
  listed in the backlog.
- **What each worker did.** It measured first, then wrote its position with `panel_note`, then taught each
  measurement with `teach`.
- **The digest.** `panel_digest.json` is the output of `panel_deliberate` over the five questions. Its law: when
  the panel agrees, the digest stays silent; only disagreement is reported.

## Folders

- **`w1-contrastive/`** (van den Oord, Gutmann, Widrow)
  - `probe_proto.py`: InfoNCE prototypes against miss-only AdaptHD, one pass, clean and 10% noise. Outputs:
    `banking_*.json`, `clinc_all.json`.
  - `probe_gate.py`: teacher self-agreement ε̂ and the corrected gate. Outputs: `gate_*.json`.
  - `probe_trace.py`, `probe_trace_apa.py`: reflex-trace correction, today's write against LMS plus an affine
    projection. Synthetic keys.
- **`w2-hd/`** (Plate, Kanerva, Frady)
  - `exp_a_resonator.py`, `exp_a2_permroles.py`: composing a method call as a role-filler sum and as a bound
    product; resonator exits; permutation roles.
  - `exp_b_direction.py`: exchange direction in the encoding against the question. This is the probe that found
    the positional-truth error in sweep 181.
  - `exp_c_null_center.py`: NULL centroid, centering, and hypervector ranking against exact sparse ranking.
  - `exp_d_reflex_contrastive.py`: correcting the real reflex trace on CLINC keys; signed write against an
    outcome-role record.
  - `exp_e_projection_cost.py`: the cost of each projection.
- **`w3-reflex/`** (Kohonen, and the JEV and NOOA lenses)
  - `e02_bench.py`, `e02_attrib.py`: routing on held-out catalog aliases, and where route_tiered's 16.7 ms goes.
  - `e02_service.py`: `serve` round trip and the 20-alias probe.
  - `panel.py`, `teach1.py`, `teach2.py`: the panel notes and the teaches.
  - Defects G1–G4 and D1–D4 are written up in the backlog.
- **`w4-measure/`** (Duda, Cranmer, Olshausen)
  - `p0_baseline.py`: re-measures the sweep-181 baseline (0.8413, not 0.832).
  - `p1_centering.py`: E1.4 centering and contrastive readout, a kept negative.
  - `p2_proto.py`, `p2b_proto.py`: InfoNCE on the learned meaning index at k = 20 and k = 100 wordings.
  - `p3_compare.py`: candidate-relative softmax against absolute cosine as the confidence (AURC).
  - `p4_prior_trap.py`: why the gate protocol E0.4 is needed (calibration-sample swing, val vs test
    out-of-scope prior).
  - `common.py`: shared loaders. `*.txt` / `*.json` are the outputs.

## Rerunning, honestly

- **Scratch paths.** The scripts were written in a session scratch directory and still name it
  (`/tmp/claude-0/.../scratchpad/...`). The paths are kept as run. To rerun, change them:
  - `DATA` at the top of the w1 and w2 scripts;
  - `S` in `w4-measure/common.py`;
  - the `sys.path.insert` line in the w3 scripts. That line only finds `inv.py`, the 10-line `POST /invoke`
    client, which is vendored next to them.
- **Datasets are not vendored.**
  - CLINC150 (`clinc150_full.json`, CC BY 3.0): Larson et al. 2019, https://github.com/clinc/oos-eval
  - Banking77 (`banking77_train.csv` / `banking77_test.csv`, CC BY 4.0): Casanueva et al. 2020,
    https://github.com/PolyAI-LDN/task-specific-datasets
- **Environment.** Run from the repo root with `PYTHONHASHSEED=0`. The `w3-reflex/e02_service.py` probe needs the
  service on port 8080.
- **The model end is a scripted stand-in.** It answers from dataset labels, with optional planted noise, exactly
  as in sweep 181. These numbers measure the learning loop, not any real model's judgement.
- **These probes are evidence, not the implementation.** The shared `ProtoStore`, the trace correction and the
  harness `tools/bench_contrastive.py` are backlog items. Each one must reproduce its number here under the E0.4
  gate protocol before it lands.

None of these files is collected by pytest: `testpaths = ["tests"]`, and the repo-walking tests scan only
`holographic/`.
