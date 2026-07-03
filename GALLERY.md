# leCore — Gallery: visual output & measured behaviour

*A showcase of what the engine actually produces. The 3-D renders, procedural patterns, reaction–diffusion
frame, and the four data charts are **generated fresh from the engine** by `make_gallery.py`; the rest come from
the committed test/benchmark harness (`figures/`). Everything is pure NumPy — no GPU, no pretrained models. The
performance/behaviour numbers are from single-thread sandbox runs, so treat them as ballpark, not spec.*

*(Visual companion to [`REFERENCE.md`](REFERENCE.md), which maps the code.)*

---

## Rendering & 3-D

A from-scratch Monte-Carlo path tracer on signed-distance geometry — diffuse, metal, and dielectric materials,
soft sky lighting — all in NumPy.

![Path-traced spheres on a checker floor](gallery/render_spheres.png)

*Three spheres — diffuse red, gold metal (GGX), blue — on a checker floor. 240×200, 48 spp, 4 bounces.*

![A glass sphere refracting the scene behind it](gallery/render_glass.png)

*A clear **glass** sphere in front of two coloured spheres: the material returns an index of refraction (IOR 1.5),
so the tracer bends rays through it — you can see the scene behind flipped and warped through the glass. 64 spp,
6 bounces.*

![A Menger-sponge fractal](gallery/render_fractal.png)

*A **Menger sponge** (3 recursion levels) ray-marched from its signed-distance function. The geometry is
*generated*, not stored — the SDF is a few bytes whether it resolves to 100 k or 250 k faces.*

---

## Procedural & generative

Richness from a tiny deterministic kernel.

![Procedural pattern fields](gallery/patterns.png)

*Procedural pattern fields (`holographic_pattern`): fBm, value noise, checker, dots — each a **field over world
position**, a solid 3-D texture that wraps any surface with no UV unwrap.*

![Vector reaction–diffusion](gallery/reaction_diffusion.png)

*A vector-valued reaction–diffusion cellular automaton (`holographic_automaton`) — Turing patterns in hypervector
space, 24 steps, projected to RGB.*

![A zoo of reaction–diffusion patterns](gallery/ca_zoo.png)

*The same machinery under different couplings — from the test suite.*

![Film grain / image processing](gallery/film_grain.png)

*Image-processing output from the test suite.*

---

## Content-addressable memory & superposition

Store many things in one space; recall by content; degrade gracefully instead of failing hard.

![The holographic image archive](gallery/archive.png)

*The holographic image archive (`holographic_archive`): images superposed into Walsh–Hadamard key plates and any
one recovered by content — exact when undamaged, graceful under an erasure mask.*

![Holographic image reconstruction](gallery/holo_capable.png)

*Reconstruction quality across stored images — recovered beside original, from the test harness.*

![Superposition multiplexing](gallery/multiplex.png)

*Many signals bundled into one hypervector and pulled back apart by content — superposition as storage.*

---

## The deterministic learning creature

![The learning creature](gallery/creature_viz.png)

*The reinforcement-learning forager (`holographic_creature`) — deterministic, debuggable, learns online without
catastrophic forgetting, and can say in human sense-terms why it chose a move.*

---

## Data & measured behaviour (the non-3-D story)

How the algebra actually behaves — every claim with a baseline and a spread. These four are generated fresh.

![Core op cost vs dimension](gallery/perf_core_ops.png)

*Cost of the two core operations vs hypervector dimension — `bind` (FFT circular convolution) and a 16-way
`bundle` (superposition), microseconds per op, single thread. The whole algebra is cheap.*

![Compression vs SQL](gallery/compression_vs_sql.png)

*The measured answer to "how well does our store compress vs SQL?" — bytes/record for the engine's low-rank
rate-distortion code vs SQLite on the same structured data, as the table grows. The VSA store's shared basis
**amortises**, so per-record cost **falls with N** and crosses *under* SQLite at a few thousand rows (~10 vs
~32 B/record by 50 k). gzip beats both on raw bytes, but gives no query — the VSA bytes *are* the fuzzy index.*

![Memory capacity curve](gallery/capacity_curve.png)

*Key→value recall accuracy vs how many pairs are packed into one vector, at three dimensions. The honest capacity
**cliff** — and how adding dimensions moves it to the right (capacity ≈ order D).*

![Graceful degradation under corruption](gallery/graceful_degradation.png)

*Recall accuracy as the memory vector is progressively zeroed out. It declines **gracefully** rather than
crashing — the hallmark of distributed/holographic storage.*

![Capacity vs load (harness)](gallery/bench_capacity.png)
![Recall under corruption (harness)](gallery/bench_corruption.png)
![Quantization robustness (harness)](gallery/quant_robust.png)
![Throughput (harness)](gallery/bench_throughput.png)
![Scaling under stress (harness)](gallery/stress_scaling.png)
![An ablation result (harness)](gallery/improve_final.png)

*More measured curves from the benchmark/stress harness — capacity, corruption tolerance, quantization
robustness, throughput, adversarial scaling, and a before/after ablation. Every gain is measured against a proper
baseline.*

---

*The 3-D renders, procedural patterns, reaction–diffusion frame, and the four data charts regenerate any time
with `python make_gallery.py`; the rest are produced by the test suite and benchmark harness.*
