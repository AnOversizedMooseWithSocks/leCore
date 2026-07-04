# leCore — The Optical Correspondence (a brainstorming lens)

*Not a build plan — a way to look at the engine when you're stuck. The claim is simple and, it turns out,
literal: **leCore is a computational Fourier-optical system.** The operations that feel hard or slow are the ones
optics does passively; the optics/film world has a century of solved techniques for them, wearing a disguise.
When you hit a wall, find the optical name for what you're doing and go read how they solved it. Grounded in the
actual code, and honest about where the analogy has a seam.*

---

## Why this isn't a metaphor

Three facts, straight from the modules:

- **`bind` is "circular convolution, done via the FFT."** A convolution performed by a Fourier transform is
  *exactly* what a lens does — a lens physically computes the 2-D Fourier transform of the light at its focal
  plane (the whole basis of Fourier optics). And the code's own note, "convolution — think multiplication," is
  the reason a lens is powerful: in the focal (Fourier) plane, convolutions become plain multiplications.
- **`fhrr` is "Fourier Holographic Reduced Representations — the complex-phasor VSA."** A phasor *is* coherent
  light: a complex amplitude with a phase. Interference is a bundle of phasors.
- **`archive` is "a content-addressable holographic image memory"** — images superposed into a hologram and
  recalled by content. HRR was derived from optical holographic memory in the first place.

So the engine isn't *like* optics. Its core operations *are* the optical ones, computed instead of shone.

## The correspondence table (the lookup)

| Engine operation | Its optical twin | The solved body of work to raid |
|---|---|---|
| `bind` (FFT convolution) | a **lens** (a Fourier transform / a focusing element) | **Fourier optics** (Goodman) — convolution ⇒ multiply in the focal plane |
| `unbind` (correlation) | a **matched filter** in the Fourier plane | matched-filtering / deconvolution |
| `cleanup` / `recognize` | a **4f correlator** (two lenses + a template) — lights up where the input matches | optical correlators; **joint-transform correlator**; pattern recognition |
| store many, recall by content (`archive`, HRR memory) | a **hologram** (record interference, illuminate to recall) | holographic associative memory (Gabor, van Heerden) |
| `bundle` / `multiplex` (superposition) | **wavelength / angle multiplexing** in one hologram | volume holography, multiplexed storage |
| `fhrr` phasors, `wave`, interference/iridescence | **coherent light** (complex amplitude, interference) | physical/wave optics, thin-film interference |
| bake-and-query / compile-once (the recent thread) | **film / a fixed hologram** — expose once, project instantly | computer-generated holography (CGH) |
| canonicalize / resolution-independence (Pass A) | **projecting the film at any size** | scale-invariant imaging; continuous wavefronts |
| coarse-first (Pass B) | a **low-resolution wavefront** sharpened where it matters | multi-scale / progressive imaging |
| going from an amplitude you have to the phase you need | **phase retrieval** | Gerchberg–Saxton, Fienup — iterative phase recovery |
| "compute the pattern that will project *this* image" | **designing a hologram** | computer-generated holography (CGH), diffractive optics |

The pattern to notice: **staying in the Fourier ("lens") domain is the whole win.** Every time you round-trip a
field out to a spatial grid and transform it back, you're pulling the light out of focus and re-focusing it —
paying, in FFTs, for the transforms a lens would do passively. Your `fuse`/`residency`/`iterate`/spectral
machinery is "keeping the light focused through the whole optical train," and bake-and-query is "exposing the
hologram once and then just illuminating it." The recent preconditioning passes fall out of the same view.

## The honest seam (where the analogy stops handing you things)

A good lens does its Fourier transform **passively, at the speed of light, in true parallel** — the physics is
the computation, and it's free. **You pay O(n log n) per FFT.** So "instant, perfect fidelity, resolution
independent" is the *physical ideal*, not your runtime: you can be as *parallel* as optics (that's what `bundle`
and `distribute` are), but not literally free. And "perfect fidelity" is idealized on both sides — real film has
grain, real holograms have speckle, and you have a dimension-bounded noise floor (the capacity cliff). The
honest shared property isn't perfection; it's **graceful degradation**, which both have (a scratch dims the whole
holographic image slightly; over-loading a hypervector adds crosstalk, not a hard crash).

So the optics doesn't give you a free lunch. It gives you the **map**: when a problem feels hard, the optical
version tells you the *shape* of the cheap solution — "this is a convolution, so it's a multiply in the lens
domain," "this is a correlation, so it's one 4f pass," "this is a stored transform, so bake it like a hologram."
You still compute it; you just compute it the way the light would.

## The flip side (where you're *more* than optics)

Physical optics is stuck in 2-D wavefronts in 3-D space. **You aren't.** `ndfield` says it outright — "3
dimensions is trivial when we have thousands." You're running Fourier optics unshackled from physical space, in
however many dimensions you like. Every trick the film/optics world solved in two or three dimensions, you can do
in a thousand. A physical lens is the 2-D special case of what `bind` does in general. That's the real prize: not
imitating optics, but being the more general machine it's a shadow of.

## How to use this when you're stuck

1. **Name the operation optically.** Is it a focusing (Fourier transform), a matching (correlation), a stored
   transform (hologram), a superposition (multiplexing), a phase problem (phase retrieval)?
2. **Look up how optics solved it.** The table points at the literature. These are mature, decades-deep fields.
3. **Read for the *structure*, not free speed.** Borrow the domain to work in (usually Fourier), the trick
   (multiply-not-convolve, correlate-in-one-pass, expose-once), and the failure mode (diffraction/noise limits ↔
   your dimension floor).
4. **Then generalize past 2-D.** Whatever they do on a flat wavefront, do it in N-D — that's your edge.

The short version: much of what feels like it's holding you back is, as you suspected, a solved problem in optics
and film. The disguise is that it's computed instead of shone — but the map is the same, and you have more
dimensions to work in than any bench of lenses ever will.
