# SCIENCE INSTRUMENTS -- the faculty-to-ancestor map

Every instrument below is a *statistics instrument*: it returns a verdict, the null it was judged
against, and its power -- never a discovery claim. Each descends from a named method in the
literature; the ancestry is the audit trail. A scientist should be able to read this page, check
the citation, and know exactly what question the verdict answers before trusting it.

One front door serves all of them:

```python
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
rep = mind.science_report({"levels": my_eigenvalues}, kind="levels")
print(rep["verdict"], "--", rep["why"])      # rep["result"] holds the full audit trail
```

Kinds: `light_curve`, `pulsar_panel`, `spectrum`, `decay`, `levels`, `chsh`, `series`.
An unknown kind raises with this list; the front door never guesses, because routing data into
the wrong instrument produces a *confident* nonsense verdict -- the one failure a refusing
instrument family must not commit at its own entrance.

---

## `transit_search` / kind `light_curve` -- the box-matched period hunt

**Ancestor:** Kovacs, Zucker & Mazeh, "A box-fitting algorithm in the search for periodic
transits", A&A 391, 369 (2002). The signal-residue statistic is their closed form.

**Why not Lomb-Scargle:** a transit is a box, not a sinusoid; measured before building, the
sinusoid-matched filter found the planted period at 6.3x less peak contrast -- near the
detection floor that factor *is* the difference between finding and losing planets.

**The verdict discipline:** the claim is phase coherence at P, so the null is the block shuffle
with block << P (red noise survives, cross-period alignment dies). The iid null is reported but
never used: red noise makes it anticonservative -- it flags red noise as planets. Harmonic
families (P/2, 2P) are reported as one finding, not hidden as several. `transit_detection_floor`
returns the detection-limit curve -- the honest deliverable is where the instrument *stops*
working. `fold_subtract` consumes a found period (median bins, or the CircularEncoder kernel
fold `engine='vsa'` -- smooth, bin-free, uneven-sampling native).

## `hd_search` / kind `pulsar_panel` -- the Hellings-Downs pattern with a sky-scramble null

**Ancestors:** Hellings & Downs, ApJ 265, L39 (1983) for the curve
chi(theta) = 1/2 + (3/2) x ln x - x/4, x = (1 - cos theta)/2; the sky-scramble discipline as
practiced by the pulsar-timing-array collaborations (e.g. NANOGrav's scramble checks).

**Two nulls for two claims:** AAFT-per-pulsar surrogates answer *does any cross-correlation
exist* (spectra kept, alignment destroyed); the sky scramble -- positions permuted against
residuals -- answers *is it patterned by geometry*: every pairwise correlation survives, only
the angle-structure dies. A common clock error (monopole) co-moves the panel, passes the first
null, and fails the second: that three-way discrimination (`hd-consistent` /
`correlated-not-sky-patterned` / `independent`) is the actual PTA systematics question.

**Honesty clause:** per-pulsar AR whitening (raw red-vs-red correlations are spurious --
measured, pinned) attenuates shared signal, so the amplitude is a *lower bound*; the certified
quantity is the curve *shape*.

## `spectral_lines` + `redshift_verdict` / kind `spectrum` -- lines, identity, one shared shift

**Ancestry:** classical spectroscopy practice; the identification-with-margin is the codebook
cleanup discipline in scalar costume, and the redshift verdict is the Le Verrier move (one
parameter must explain every residual, or refuse) applied to a line list.

**Kept negatives on record:** a permutation null contains its own lines (the multiplicity null
must draw from the noise-only distribution); the z-scan's best value is the tolerance window's
low edge (the scan picks the assignment, the value is the median per-line z); an identification
without a margin over the runner-up is a coin flip wearing a name -- between lines, the
instrument abstains. Velocity readout delegates to the existing `dedoppler` faculty.

## `fit_decay` / kind `decay` -- A exp(-lambda t) + C, closed form

**Ancestry:** weighted log-linear decay estimation as used from radioactive counting to
randomized-benchmarking fidelity curves (Magesan, Gambetta & Emerson, PRA 85, 042311 (2012) for
the RB use of exactly this fit shape).

**The load-bearing line is the weights:** W = d^2, by the delta method (Var[log d] ~ 1/d^2).
Plain d-weights read lambda 17% low; a coordinate-descent background pass moved the *wrong* way
(the errors feed each other) -- both measured, both kept. The truncation flag carries a
bias-aware margin: on a truncated record lambda biases high, which is the very failure the flag
reports, so the margin must absorb it.

## `level_statistics` / kind `levels` -- integrable vs chaotic, no unfolding

**Ancestors:** Oganesyan & Huse, PRB 75, 155111 (2007) introduced the spacing ratio; Atas,
Bogomolny, Giraud & Roux, PRL 110, 084101 (2013) give the reference values used here
(<r~>_Poisson = 2 ln 2 - 1 exactly; GOE 0.53590; GUE 0.60266).

**Why ratios:** the classical spacing distribution requires unfolding (dividing out the local
density), and a wrong unfolding manufactures or erases level repulsion -- the field's classic
instrument error. The ratio cancels the density exactly. The verdict is bootstrap-CI class
membership, refusing (`indeterminate`, with the n that would decide) when classes overlap --
the p-floor lesson as a sample-size statement. Edges are trimmed: universality lives in the bulk.

## `chsh_verdict` / kind `chsh` -- the Bell verdict with the Tsirelson alarm

**Ancestors:** Clauser, Horne, Shimony & Holt, PRL 23, 880 (1969) for S and the classical bound
2; Tsirelson, Lett. Math. Phys. 4, 93 (1980) for the quantum bound 2 sqrt(2).

**Three gates, one alarm:** the pairing-scramble null (B shuffled within setting cells) answers
*correlated at all*; the bootstrap CI against 2 answers *beyond every local hidden-variable
model* -- the whole polytope, not a point null; and a CI past 2 sqrt(2) reads
`suspect-instrument`: quantum mechanics itself stops there, so such data accuses the apparatus
(post-selection, pairing errors), not the theory. The planted verdict experiment includes an
explicit local-hidden-variable model, and the selftest states the contract in its assert: if the
instrument calls *that* nonclassical, the instrument -- not Bell -- is wrong.

## `residual_ladder` / kind `series` -- the interrogation tower

**Ancestry:** Le Verrier's residual discipline (the unexplained part of a fit is data about the
next mechanism), Box-Jenkins mean-equation-before-variance-equation ordering, Engle's ARCH for
the scale rung. Documented in full in the RESID arc notes; the front door exposes the tower's
terminal verdict (`irreducible` -- priced as noise by every rung -- or `rungs-exhausted`).

---

## The standing grammar (all instruments)

* **One claim, one matched null.** The null destroys exactly the structure on trial and keeps
  everything else.
* **P-floors are arithmetic.** With n surrogates the minimum p is 1/(n+1); a gate that cannot
  pass says so instead of pretending it ran.
* **Refusal is a result.** `indeterminate`, `no-consistent-shift`, `underpowered`,
  `suspect-instrument` -- each names *what would decide*.
* **Kept negatives travel with the code.** Every measured failure above is pinned in a selftest
  and documented in the module that owns it; NOTES_concepts.md holds the full ledger.
