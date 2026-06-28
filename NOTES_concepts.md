# Concept notes -- physics/nature analogies vs. the substrate

This project runs on one rule: an idea earns its place only by beating a baseline
under honest measurement, and negatives are kept. These are six natural-process
analogies considered as possible improvements, filtered by that rule. The point
of writing them down is the *reasoning* and the one *measured boundary* -- not a
pile of new features. Most of these are deliberately NOT implemented; the value
is knowing why.

## 1. Double diffusion / salt fingering  -- TESTED, clean negative (a boundary)

Physics: heat diffuses ~100x faster than salt, so a layer that is stable in
temperature but unstable in salinity throws "fingers" of mixing that neither
variable makes alone. The fast and slow signals, and the gap between them, are
the whole mechanism.

Mapping: a memory prototype has a FAST signal (its centroid -- every observation
nudges it) and a SLOW signal (internal stratification -- members quietly
splitting into sub-modes, which takes many observations to show). The analogy
predicts a cheap pre-screen for `auto_reorganize`: a class can be STABLE in
held-out accuracy while UNSTABLE in internal variance -- the finger -- a split
forming before accuracy degrades. If the variance signature ranked classes by
how much a split helps, the expensive resolution sweep could run only on
"fingering" classes.

What the measurement said: it does not transfer to the hyperdimensional
substrate, and the reason is quantified. A two-means stratification signal
(how much better two centroids fit a class's members than one) separates truly
bimodal classes from unimodal blobs by **3.9 sigma at dim 8**, but the
separation decays to ~0.3 sigma -- pure noise -- by dim 128 and stays there at
512 (the working dimension). Concentration of measure is the culprit: in high
dimensions a unimodal Gaussian blob looks just as "splittable" to k=2 as a
genuinely bimodal class, so the local variance gradient the fingers need does
not exist. (`_exp_double_diffusion.py` reasoning, dimension sweep recorded here.)

The useful conclusion: this EXPLAINS why the existing accuracy-only trigger is
the right design. The cheap variance pre-screen the analogy suggests is not a
missed optimization -- it is mathematically unavailable at 512-d. The slow
signal has to be held-out accuracy itself, which is exactly what
`auto_reorganize` already measures. A real salt-finger memory would need a
low-dimensional variable to stratify on; the substrate deliberately has none.

REVISITED (after the projection/consolidation work, concept 7, which discovers a
~22-36-dim subspace the prototypes share -- the low-dimensional variable the note
said we lacked). Re-measured the two-means stratification gain, this time on the
REAL encoded multimodal world (`_multimodal_world`, classes with genuine sub-modes
through the UniversalEncoder) rather than synthetic Gaussian blobs. Two honest
findings: (a) the signal is NOT dead at 512-d on the real substrate -- bimodal vs
unimodal classes separate by ~7 sigma, essentially identical full-512-d and
in-basis (7.17 vs 7.01). The original 0.3-sigma decay was an artifact of
structureless synthetic blobs; the encoder's real structure carries the signal.
(b) The gain PREDICTS the payoff: correlation 0.94 between a class's two-means gain
and the held-out accuracy benefit of actually splitting it (bimodal gain ~39 vs
unimodal ~1.07). So the pre-screen the original declared unavailable does work
here. SHIPPED as a CONSERVATIVE, default-off option on `auto_reorganize`
(`fingering_prescreen=True`): when no class fingers (max gain < finger_floor) it
skips the expensive 4-resolution sweep and short-circuits to "keep" -- it can only
AVOID work, never change the measured choice (verified: identical organization to
the full sweep on both bimodal -> k=2 and unimodal -> keep worlds; ~21% faster on a
stable memory, 10.7 -> 8.4 ms). The lesson cuts both ways: the original negative
was real for the test it ran, but the test bed (synthetic blobs) was not the
substrate, and a capability unlocked later (the consolidation lens) made it worth
re-measuring on real data -- where it became a small honest win. Pinned in
test_holographic_organizer.py.

## 2. Surface tension  -- NOT implemented (likely a refinement, and we have a curation negative)

Water minimizes surface area; a membrane resists deformation in proportion to
curvature. Mapping: the reorganization gate could resist splitting a prototype
cluster that is already smooth/coherent (high tension) and yield for a lumpy one
(low tension), replacing the flat 1-SE leanness margin with a coherence-scaled
one. Plausible and measurable, but (a) it is a refinement of a gate that already
works, and (b) the codebase has a clean negative on layering a curation
controller over the organizer's own aggregation. Worth a small experiment only
if the 1-SE rule is later shown to mis-fire; not a priority.

## 3. Gravity lensing  -- NOT implemented (mostly re-describes existing machinery)

Mass bends geodesics; light bends around it and can multiply into rings. Mapping:
high-frequency prototypes should bend nearby queries toward themselves. But that
is already what the ReflexCache's hot set and any frequency weighting do -- the
analogy re-describes a prior we have. Its one non-redundant prediction is
*multiple images*: a query near a massive prototype routed two legitimate ways,
which maps to keeping both a coarse and a fine prototype for a heavily trafficked
class -- and that is what the multi-resolution organizer already does on demand.
No new mechanism falls out.

## 4. Flocking (boids)  -- TESTED, clean negative

Three local rules (separation, alignment, cohesion) produce global coordination
with no leader. Mapping is to a RECORDED negative: averaging independently
trained minds was worse than picking the best single one (their policies differ
too much to average), so the UI trains several and keeps the best. Flocking
proposes a third option that negative did not rule out: candidate policies
nudged toward LOCAL agreement only with their nearest neighbours in policy
space -- not global averaging (which destroyed them) and not single best-pick
(the current winner), but local alignment that keeps global diversity.

Tested as a reversible decision-time value blend: each candidate's action scores
pulled 30% toward its single most-similar peer's scores, then the committee
votes. Two regimes, 16x16 mazes:

  * Well-trained candidates: flock ties best-pick at 100% everywhere AND beats
    plain averaging where averaging collapses (maze 5: flock 100%, average 0%) --
    so local alignment really does dodge the averaging failure. Encouraging, but
    best-pick is already saturated, so no separation.
  * Under-trained candidates (the regime that matters -- where a committee should
    earn its keep): best-pick **100%**, flock **67% mean, 0% worst** (collapsed
    on 2 of 6 mazes).

The why is decisive: when candidates disagree, local alignment can pull a GOOD
mind toward a confidently-wrong neighbour -- flocking has no notion of which
neighbour is correct, only who is nearby. Best-pick's probe, even noisy, directly
MEASURES escape and selects a winner. Measurement beats consensus; the recorded
negative stands and best-pick remains the champion. (`_exp_flock.py` reasoning.)

This sharpens a principle the whole project already runs on: where a cheap
measurement of the real objective is available (probe escape rate, held-out
accuracy, compression bits), trust it over any structural prior about how the
candidates "should" relate. Flocking is a structural prior; it loses to the
measurement.

## 5. Prism / spectral decomposition  -- partially realised; premise CHECKED and refuted for the open problem

A prism separates superposed frequencies by refractive index. The holographic
substrate IS superposition, and unbind/cleanup IS separation, so this is less an
analogy than a description of what the engine already does (the ResonatorNetwork
factoring a scene into per-object attribute atoms is a literal prism).

The target it seemed to point at: the survival forager bundles "food east" +
"danger north" + walls into ONE superposed state prototype, and the guess was
that this fusion fragments learning and causes the recorded wall-pocket
dithering. Before building a state-decomposition mechanism, the premise was
checked cheaply -- instrument a long cluttered-world life and ask whether
dithering correlates with state ALIASING (distinct physical cells sharing a
near-identical bundled state, which is what fusion would cause). It does NOT:
only 3.5% of distinct-cell state pairs alias (cos > 0.85), and 89% of visited
cells get a STABLE chosen action. The forager is not confused about where it is;
it makes a locally-consistent but globally-trapped choice at pocket entrances.
Fusion is not the disease, so a prism is not the cure -- the premise check saved
building the wrong mechanism.

Where the prism analogy IS already real and correct: the ResonatorNetwork in the
compositional-scene work, which genuinely splits a superposed scene back into its
attribute bands. That is the legitimate home of the idea; the forager is not.

## 7. Projection (one 3-D object, many 2-D shadows)  -- TESTED, the first WIN, with a measured hazard

The concept: a single complex object casts many different shadows depending on
the projection plane; conversely, where many flat patterns OVERLAP, the overlap
is a registration mark that they are projections of one higher object (the
Contact-blueprint reading: the pages only made sense stacked).

Where it is literally true in this system: the creature brain's state vectors
are bundles of a SMALL atom vocabulary, so its thousands of 512-D prototypes
must all lie in the span of those atoms -- they are high-dimensional shadows of
one intrinsically low-rank object. Measured on trained brains: 99.9% of
prototype energy sits in **22-24 of 512 dimensions** (forage and 16x16 maze
both). `HolographicMind.consolidate()` discovers the subspace by SVD over the
prototypes themselves -- the overlap IS the registration mark -- and re-stores
the entire memory as coefficients in it. Results at full behavioural parity:
**21x smaller memory, ~5x faster decide()** (forage 122 -> 120 stars at
1.36 -> 0.25 ms/decision; 16x16 maze 90% -> 95% escapes).

The hazard, found by measuring before integrating: **a shadow hides new
structure**. A brain consolidated in a poison-free world compressed to rank 9,
and the danger sense then carried only **4% of its energy inside the basis** --
poison was nearly invisible to its values. So consolidation ships with a
residual guard (the flux-guard pattern, fourth appearance): every incoming
state's out-of-basis energy is tracked as a slow EMA, and when it grows past a
threshold the basis EXPANDS from a small ring of recent raw states (new
orthogonal directions appended; old prototypes get zero coefficients there,
which is exact -- they truly had no such component). Measured under a world
shift: basis 9 -> 13, danger in-basis energy 4% -> 100%, learning continues in
the grown space. Compress when the world is stable, grow when it is not.

Both behaviours are pinned in test_holographic_brain.py. This is the only one
of the seven concepts to produce shipped machinery -- and its hazard would have
shipped too, silently, without the measure-first step.

## 6. Demoscene  -- the operating constraint, not a feature

Maximum effect from minimal, fully deterministic code; seeded RNG everywhere so
every result reproduces. This is already the house style and the discipline the
other five are held to: any addition must stay tiny and seed-reproducible, or it
does not go in.

## 8. Chained concepts -> meaning (the user's follow-on: projection + composition)

The thesis: if things are decomposed and structured properly, basic concepts
chain into complex ones, and the structure should surface not just THAT two
things are similar but WHY and HOW. Tested, and the thesis holds -- with one
deep, measured qualifier about what makes it hold.

In this algebra the why/how has an executable form. Entities are role-bound
records (bundle of bind(role, filler)) -- exactly what the UniversalEncoder
already builds from dicts -- and four operations recover the relationship
(holographic_relations.py, all numbers from the 10-entity demo world):

  EXPLAIN  per-role decode of two records with a match verdict ("france is like
           belgium BECAUSE currency=franc, language=french, continent=europe;
           UNLIKE because the capitals differ") -- 4/4 roles correct.
  NAME     unbind a filler from a record and clean up against the roles:
           "paris relates to france AS capital" -- 40/40 = 100%.
  MAP      "what is the dollar of mexico?" -- name the role the probe fills in
           one record, read that role out of the other -- 360/360 = 100%.
  CHAIN    "the language of the country with the currency of the country whose
           capital is X" -- 100% at two AND three hops.

THE QUALIFIER, which is the real finding: meaning survives composition only
when it touches SYMBOLS between steps. The direct algebraic relation object
M = bind(rec_b, involution(rec_a)) -- one bind, conceptually the purest "the
relationship as a first-class vector" -- scores ~94% on the same mapping task,
its failures are 20/22 pure HRR cross-term noise (only 2 honest filler
ambiguities), and it does NOT improve with dimension (96/94/90% at
1024/2048/4096). Routing every hop through a cleanup (geometry -> symbol ->
geometry) is exact on the same data: the discrete vocabulary acts as error
correction between compositions. The symbolic layer is not decoration on the
geometry -- it is what makes chained meaning reliable. (This is also why the
3-hop chain survives: each hop snaps to a symbol before the noise can compound,
the same per-hop-cleanup law the maze corridor reflex and the generation work
each found in their own domains.)

Integration: holographic_relations.py (KnowledgeStore with explain/name/
the_x_of/find/ask) and UnifiedMind.explain(x1, x2), which articulates the
per-role verdict for any two record dicts using the mind's own encoder, with
cleanup candidates drawn from the inputs themselves. One honest detail pinned
in passing: numeric fillers (wheels: 4 vs 2) decode correctly but at near-zero
confidence, because scalar encodings make nearby numbers near-neighbours BY
DESIGN -- the verdict is right and the low confidence honestly reports that the
call was close. Pinned in test_holographic_relations.py.

CROSS-MODAL CODA: the same four operations answer "why is this IMAGE like that
one" with zero new machinery -- the existing auto-tagger (colour from HSV,
shape from geometry, texture from the DCT) turns raw pixels into the same
role-bound records, and explain/name/map/chain apply verbatim
(holographic_scene.explain_objects). Measured end-to-end on generated shapes
where ground truth is known: tagger 36/36, explanation verdicts 72/72 = 100%
over all pairs, chains working over stores of tagged images. This is the
thesis's strongest form: meaning recovered identically for words, numbers, and
pictures, because all three decompose into the same role-filler structure
before the relations machinery ever sees them.

UNIFICATION CODA: the operations now live ON the unified mind, not beside it.
find/read_role/ask/explain run over the records absorb() stored and the filler
vocabulary learn() registered from experience -- and the measured payoff is the
genuinely new result: a class prototype built from SIX noisy, incomplete
observations (one random role dropped per copy) still decodes its roles at
100% (read 40/40, explain verdicts 180/180, 3-hop chains exact), because the
superposition's linearity reinforces shared role-filler terms while dropouts
average out. The mind explains concepts it LEARNED. One bug the suite caught
during integration, kept for the record: the new read(label, role) shadowed
the mind's existing read() (text pre-reading) -- renamed to read_role; the
collision broke absorb() instantly and visibly, which is what 200 tests are
for.

INCEPTION CODA -- the loop closed: the relations decode turned INWARD.
(1) explain_splits(): when the organizer splits a class, the sub-prototypes'
roles are decoded and contrasted, NAMING what the split separated. Separation
is judged by contrast (each mode's winner genuinely absent from the other --
measured ~0.5 for real structure vs <=0.1 for incidental skew), and the
statistic's first outing caught the organizer making an accuracy-sufficient
but structurally ARBITRARY split (it separated the noise role on one XOR
label, because the other label's clean split already resolved the confusion)
-- the explanation honestly reports what the split did, not what was assumed.
(2) The creature brain gained describe()/why_differ() (its states are
role-bound sense bundles; present roles decode 373/373, absent stay silent
427/427, the 0.18 floor in a real gap). Pointed at a caught dither, the brain
articulated its own bug: the oscillation states were sense-identical and it
was choosing E at value +0.43 while sensing wall_E=yes -- valuing moves into
walls. The articulation named the one-line fix (walls join poison in the
`among` veto), and the fix SOLVED the recorded cluttered-world open problem:
stars 5.1 -> 19.8 (the reflex ceiling), dither 79% -> 43%, deaths 0%. The
earlier scorecard's elimination had narrowed the cause to 'a value-estimate
trap'; introspection finished the job by naming WHICH values were wrong.
Self-discovery (the encoder registers its own experience's vocabulary),
inception (the system explaining itself with itself), and the measured loop
from explanation to repair.

PROPAGATION CODA -- the capabilities pushed into the MAIN models, every
consumer updated. The safety reflexes moved inside HolographicMind.decide()
itself (pass the senses, get the vetoes; among/flags are now thin translations)
-- which immediately fixed a drift the directive was about: the showcase app's
creature panel had its own inline episode loop with NO vetoes, so the on-camera
creature could still suicide into poison it could see; it now routes through
the one model mechanism. And UnifiedMind keeps a JOURNAL: both roads to
auto_reorganize (learn's auto pass and maintain_now) go through one narrating
wrapper, so every consumer -- console, tour, absorb -- gets the mind's own
account of its maintenance for free, with splits NAMED by the contrast decode
where the data is record-shaped. The introspection demo now ends with the
brain describing its best and worst prototype for an action in its own sense
vocabulary ('food north, no danger' vs 'danger north, walls around') -- the
why, visible. One model, one mechanism, all callers.

REAL-DATA CODA -- the relations operations leave the toy world. The 712-sprite
library absorbs into one mind as IMAGE + RECORD per label (colour/texture from
pixels via auto_tags; family/facing/frame parsed from the names -- and the
honest negative first: the shape tag is USELESS on pixel art, every sprite
'rectangle', a zero-information role, so it stays out of the record). The new
measurement: each label's prototype superposes an image vector with the
role-bound record, and role decode survives the mixing at 100% (750/750) --
the image component is near-orthogonal noise to the bound terms, linearity
again. The cross-modal loop closes at 96%: SEE a sprite -> classify -> SAY its
colour in symbols (the misses are sibling walk-frames confusing classify, not
decode failures). And the console gained the 'Countries (records)' dataset
(eight noisy observations per country, 97% held-out -- the noisy-prototype
result, live) plus a relations panel wired to find/read_role/ask/explain over
the mind's own memory: the WHY, clickable.

SHOWCASE CODA -- the WHY went visual. app.py gained 'Compare two sprites':
per-role verdicts decoded next to the real images, plus SEE->SAY live (the
mind classifies each unnamed image against the library and speaks the colour
from the matched prototype -- dim 2048 is load-bearing: at 1024 the image
classify confused recoloured sibling variants and answered with the VARIANT's
colour; measured, then fixed by using the library's measured configuration).
Absorbing with FAMILY labels closed the real-data inception loop: all 31
families split, the journal naming facing/frame for walk-cycle families and
colour for the npc grab-bag -- the organizer's account matching the data's
actual structure. And one more propagation drift caught: the showcase's
embedded unified panel had a SECOND organize endpoint that never learned the
journal story. The lesson stands -- one mechanism is only one mechanism if
every surface actually routes through it.

(8) FRACTALS -- tested, and the organizer's flat sweep SURVIVES the challenger.
The idea: self-similar memory organization -- the same speculate-measure-adopt
split applied recursively at every node, so each label finds its own depth
(recursive bisection, accepts gated on an internal held-out slice, the
finished candidate competing fairly). Measured three ways. (a) Easy uneven
world (1/1/2/4/6 modes): the fractal matches flat accuracy at ~30% fewer
prototypes -- real, but only where everything saturates. (b) Hard noisy world
(1/2/3/5/8 modes, 5 seeds): a WASH -- +2/-3/0 vs the shipped flat selection,
with more prototypes and ~5x the build time; the per-node accepts are gated on
a noisier internal slice and pay for it. (c) Extending the flat sweep to k<=8
instead: binds on 1 of 5 seeds (+2% at +9 prototypes) -- the lean-1SE rule
almost always settles at k<=4 anyway, so even the cheap version of 'deeper'
is not earning. Verdict: the lean-1SE flat sweep already harvests the
compression the fractal promises, and accuracy already gates the depth;
recursive splitting adds machinery without measured value. Recursive
self-similarity stays where it is load-bearing and measured: the RP-tree /
HoloForest index, whose recursive median splits ARE the fractal idea doing
real work (slime mould veins resolving a broad mass into hierarchy). The
organizer's resolution-selection has now outlasted four challengers
(per-label greedy, climb-while-earning, merge-grain sweep, fractal bisection)
-- a deliberate, repeatedly re-validated trade.

---

Scorecard, all testable ideas now measured (two wins, three negatives, two parked):
  (8) chained concepts -> meaning -- WIN, shipped: explain/name/map/chain over
      role-bound records (4/4, 100%, 100%, 100% through 3 hops), with the law
      that the direct symbol-free algebraic route is ~94% and dimension does
      not save it -- meaning survives composition only by touching symbols
      between steps.
  (7) projection -- WIN, shipped: consolidate() compresses the brain's memory
      21x and speeds decide() ~5x at parity, by discovering the low-rank object
      all the prototypes are shadows of; its measured hazard (a shadow hides new
      structure -- danger at 4% in-basis energy after a poison-free
      consolidation) shipped WITH its cure (the expanding residual guard,
      4% -> 100% under a world shift).
  (1) double diffusion -- closed, useful boundary result: the cheap variance
      pre-screen is mathematically unavailable at 512-d (separation decays from
      3.9 sigma at dim 8 to 0.3 sigma noise by dim 128), which is exactly why the
      accuracy-only reorganization trigger is the correct design.
  (4) flocking -- closed, clean negative: local consensus ties best-pick when all
      candidates are good and LOSES when they disagree (67% vs 100%), because
      measurement of the real objective beats a structural prior about how
      candidates relate.
  (5) prism -- premise checked and refuted for the forager (dithering is not
      caused by state fusion: 3.5% aliasing, 89% stable per-cell action); the
      idea is already correctly realised in the ResonatorNetwork.
  (2) surface tension, (3) gravity lensing -- parked with reasons (a refinement
      of a working gate; a re-description of existing machinery).

Three honest negatives and a parked pair -- and the wall-pocket dithering's real
cause is now narrowed by elimination: NOT aliasing, NOT poison risk, NOT memory
depth. It is a stable, locally-consistent bad choice at pocket entrances -- a
value-estimate trap, not a perception or representation problem. That points the
next real attempt at the VALUE side (e.g. an exploration bonus that decays with
revisit count -- repulsion from recently-exhausted ground), not at any of these
six analogies. The analogies did their job: they generated hypotheses, the
measurements killed the wrong ones cheaply, and the elimination sharpened the
real target.

GENERATION-FIDELITY CODA (user-caught: no capitals or punctuation in generated
text). The ENGINES were innocent -- the fractal coder takes raw characters --
but the diet was scrubbed (every console loader lowercased and
isalpha-filtered the corpus before the coder ever saw it) and the doorway was
too (BOTH generate endpoints lowercased the seed). The loaders now feed TRUE
corpus text; measured on Austen: ~12% more bits/char (1.949 -> 2.175) and 8
points of word coherence, for output that reads as PROSE -- capitals, commas,
apostrophes, sentences. The flat n-gram gained a fold_case switch (default
preserves every pinned number). Two engine lessons earned while debugging the
toy world corpus: (1) tiling a small block x N makes the block itself the
optimal compression unit -- the chunk schema honestly learns a corpus-sized
mega-chunk and generation replays it wholesale, seed ignored; vary the passes
instead. (2) A seed that encodes to nothing at coarse chunk levels now
descends to finer conditioning instead of trusting the unconditional prior
(bits/char unchanged at 2.175 -- regression-free). Garbage-scrubbed in,
garbage-scrubbed out: fidelity problems are usually PIPELINE problems.


PROVENANCE CODA (user asked: can stored information be traced to its source?).
The architecture is well-suited because the stores aren't opaque weights. The
recall index already keeps every item with its payload (find = exact source).
The generative sequence model now records, per context->token transition, which
DOCUMENTS taught it -- a doc-counter beside each count, no cost when unused --
and attribute(text) ranks sources by the transitions a passage uses. The
measurements drew the honest boundary: attributing GIVEN text works (70% top-1
on a clean 4-book Gutenberg split, 92% on 5, 8/9 spliced windows localized);
attributing freely-GENERATED low-order text does NOT (after the seed it drifts
into transitions every source shares -> near-uniform), and the UI admits it
rather than faking confidence. A measurement settled HOW to attribute, against
intuition: coarsest-chunk-level-first (70%) beat atom-only (42%) and all-levels
summed (48%), because an author's characteristic multi-char chunks ARE the
distinctive signal while single-char transitions are universal. An idf
refinement measured a wash and was dropped. The general lesson: provenance is
cheap and reliable when you store WHO alongside WHAT and ask the well-posed
question (attribute given text), and the right granularity is an empirical
question, not an assumed one.

COHERENT-ATTRIBUTION CODA (user's principle: a passage comes from one source, so
the source with the most UNIQUE information should resolve the ambiguous tokens
-- 'butterfly' is in three books but 'fillet' in one, so attribute the butterfly
to the fillet's source). Realized as SPECIFICITY weighting: each transition's
vote scaled by 1 / (sources that taught it), so unique evidence dominates the
tally and pins the source while shared evidence merely confirms. Measured win
where the principle predicts -- ambiguous short passages (+2-4 top-1 at 60-100
chars) and margin (a Melville probe 0.37 -> 0.68) -- and a wash where evidence
saturates (long passages already have enough distinctive signal either way). The
STRONGER sequential form (a running prior: let the leader-so-far bias later
tokens) was measured across prior strengths 0.5/1/2 and was a wash-to-slightly-
negative: specificity already captures the insight, and sequential feedback
risks committing hard to an early wrong guess. Kept the static version, dropped
the running prior -- the same discipline as the flocking/fractal/idf negatives:
the structurally-appealing addition has to BEAT the simpler measured mechanism,
and here it did not.
SEQUENCE-ALIGNMENT CODA (user: a sentence shares words with many sources --
even ones of OPPOSITE message, because they share the topic word -- so how is
provenance resolved? meaning is in the ORDERING, and nature already solves
this). The bag-of-transitions answers "whose STYLE"; it CANNOT answer "whose
MATERIAL", and is actively fooled by the adversarial case: feed it a bearish
sentence whose words all appear in a bullish source and it answers 'bull' with
full confidence -- the shared words swamp the single differing one. The fix is
genome alignment: identify a fragment by its longest CONTIGUOUS verbatim match,
not its token composition. align() scores maximal spans by length x specificity
(a long run of ' and the ' is common and discounted; a distinctive clause is
decisive). Measured: 100% top-1 on verbatim-clause probes (bag 97%) at a ~3.5x
margin, and it gets the bull/bear theses BOTH right where the bag picks wrong.
trace() unifies them -- report style AND material, lead with the decisive one:
a long verbatim span => assembled/quoted (alignment, near-certain); no span =>
paraphrase/original-in-style (the bag). The deeper point the user was right
about: the bag treats a passage as an unordered SET of transitions, discarding
the sequence that carries the meaning; alignment restores it. Two questions,
two methods, the system now picks the right one per text. (This is also why
coarsest-chunk-first beat atom-level earlier -- a chunk is a short learned
span; alignment is that idea taken to its limit, matching the longest span of
all.)


SEQUENCE / ORDER / TIME CODA (user: the same PB&J steps in the wrong order are
not a recipe; order deserves consideration across retrieval, storage,
organization, generation). Finding: the stack is order-FREE BY DESIGN where that
is correct -- topic = bag of words, class = bundle, record = set of bindings,
and infer() even routes word-lists to the bag (97.5% vs 93.8% via the sequence
path). The gap was not lost order in encoding (a scramble is ~0.03 cosine) but
that nothing could QUERY order back out. SequenceMemory adds it with the same
bind/bundle/permute primitives: step(i)=100%, position_of=100%, precedes=100%,
validate() runs the PB&J check and names the out-of-order step. A what-next
encoding measured ~64% (bundle-capacity ceiling) so next-step stays with the
exact list; this memory owns the ORDER RELATIONS no bag can answer. Where order
already lived (sequential generation/PPM/alignment, the creature's ordered
recent-action memory, the maze corridor reflex) the sweep confirmed it. The
principle: order is a SEPARATE axis from content -- some jobs need only content,
some only order, most need both, and each is applied where it measurably belongs
rather than forced everywhere.

SELF-DISCOVERED SEQUENTIALITY CODA (user: let the organizer DETECT a sequential
class -- but no cheating, no magic numbers; discover structure, do not declare
it). The mechanism is a PERMUTATION TEST, the honest way to establish a signal
is real: a class is sequential iff its members' true order predicts the next
element better than the SAME members shuffled. The shuffle is the class's OWN
null hypothesis -- nothing external assumed. A leave-one-out transition model is
scored by next-element MARGIN (how much higher the true next ranks than the
rest; the graded margin replaced argmax accuracy, which saturated on small step
vocabularies and missed a 6-step recipe -- a measured fix, kept). The statistic
is a z-score against the shuffled spread; z>2 is the standard two-sigma bar, a
STATEMENT ('exceeds noise') not a tuned constant -- the only honest place to
draw the line, and the data draws it. Measured: sequential ~+16, bag ~0,
graceful degradation through noise. Discovery feeds SELF-ASSEMBLY: a passing
class has its canonical order reconstructed from partial member sequences by a
pairwise-precedence vote (exact recovery from drop-one observations -- the mind
rebuilds a whole it only ever saw in fragments). Touches the constellation: SELF-
DISCOVERY (the class proves its own nature), DECOMPOSITION (members are fragments
of a whole order), COMPOSABILITY (the recovered order plugs into the existing
precedes/validate ops), SEQUENCE/ORDER/TIME (the axis itself), and the
demoscene/no-magic-number discipline (the threshold is a statistical statement,
reproducible and unfudged). The deeper principle: structure should be MEASURED
into existence against the data's own null, never asserted -- the same
speculate-measure-adopt that gates every split, now gating a representation
CHANGE (bag -> sequence) rather than a parameter.

RECURSIVE-DISCOVERY CODA (the sequentiality test applied to ITSELF, fractally).
Once a class is found sequential, each step is tested for its own internal order
where sub-observations exist, and the SAME permutation test recurses one layer
down -- a nested plan unfolds into a tree whose shape was never declared. The
discipline that matters is the TERMINATION: recursion stops not at a chosen
depth but where the data stops carrying order -- an atomic step (no
sub-observations) bottoms out, and crucially a step that HAS sub-observations in
unordered form (a garnish ingredient-bag) is correctly NOT expanded, told apart
from a real sub-recipe by z alone. So the tree is the data's own: layers
(structure at every scale), recursion (the test calls itself), fractals (the
same shape repeating until it doesn't), decomposition (a step is a sub-plan),
composability (each layer plugs into precedes/validate), and self-discovery
(every layer proves its own nature). The whole edifice rests on one honest move
made once and reused everywhere: measure structure into existence against the
data's own null, and let the measurement -- not a parameter -- decide where to
stop.

SELF-PROOF + CONTEXT-BINDING CODA (user: structure needs self-made proofs of
validity before its meaning is useful; and steps are generic until context binds
them -- 'open the book', 'density 5g', 'F=ma'). Two moves. PROOF: a class can
score z>2 and still be inconsistent (a precedence CYCLE admits no ordering);
prove_executable runs a topological feasibility check and gates registration on
it. The payoff was immediate and is the whole point of self-proof -- it caught a
REAL bug: the canonical-order recovery used a score heuristic that misplaced a
rare step against a 4-0 majority edge; the proof flagged the resulting violation,
and the fix (a true topological sort honouring every majority edge) followed.
Structure validating structure found an error before it shipped -- the proof is
not ceremony, it is a debugger. CONTEXT-BINDING: extract_template separates the
generic schema (stable positions) from context slots (varying positions) by
per-position entropy, split at the natural largest gap -- no magic cutoff, the
observations set the scale. This is the relations work's role-filler binding
arriving at the level of STEPS: a law/template is generic, a scenario binds its
slots, and the same vector substrate holds both. Touches: integration (proof +
binding join sequence discovery), decomposition (a step = schema + slots),
composability (templates compose with bindings as laws compose with scenarios),
projection (a generic template casts a specific instance when context is applied
-- the shadow shaped by what fills it), and the no-magic-number discipline (both
the cycle bar and the slot gap are the data's own scale). The principle, sharper
now: structure must be MEASURED into existence AND PROVED consistent before its
meaning is trusted -- statistics propose, proof disposes.

EXECUTION CODA -- the loop closes (discover -> prove -> bind -> act). execute_plan
RUNS a discovered, proven, slotted plan under an honest contract: a step fires
only when its preconditions (earlier steps in the discovered order) have fired
AND its context slots bind; else it BLOCKS with a reason, and blocks cascade
truthfully. Three honesty guarantees: no assumed success (a missing binding stops
the step), no out-of-order cheating (a late step attempted early blocks, naming
what it needs), no running the unproven (an unregistered plan raises). The
templated step fires as its CONTEXT-BOUND form ('cut into 2 pieces') -- the
generic schema + scenario binding from the previous round becomes a concrete
action. The full constellation is now one pipeline on one substrate:
sequence/order/time, self-discovery (permutation test), self-proof (topological
feasibility), decomposition+recursion+fractals (hierarchy unfolds), context-
binding+projection (template casts a specific instance under context),
composability+integration (each stage feeds the next), and the no-magic-number
discipline throughout. The arc: structure measured into existence, proved
consistent, filled with context, and ACTED upon -- nothing asserted, nothing
assumed, every failure informative.

PLAN-REPLAY CODA (the discovered plan composed with action, and knowing its own
boundary). A creature that has discovered and proven its maze's canonical route
can REPLAY it -- drive navigation from the plan instead of re-deciding every step
-- and the honest part is validation: each move must actually advance, and a
blocked move means the plan has hit the edge of where it applies. Measured: the
plan escapes 10/10 in its own maze, and in a DIFFERENT maze it detects exactly
where it breaks (5/5, naming the blocked cell) rather than falsely reporting
success. The break point is information -- the seam at which the creature would
re-learn only the changed segment, not the whole maze. This composes the whole
sequence pipeline with the policy: discover structure -> prove it -> act on it ->
know where it stops being valid. The principle that has run through all of this,
once more: structure is trusted only as far as it is proven, and a plan that
knows its own boundary is more useful than one that pretends to always work.

RAYTRACING / THROUGHPUT CODA (the path-tracing parallel, and a revisit of prior
negatives). A relation chain IS a ray bouncing through the holographic space:
each hop is a bounce, the cleanup-to-a-symbol is the surface intersection, and
the cleanup CONFIDENCE is that bounce's reflectance. Path tracing accumulates
THROUGHPUT -- the product of reflectances along the path -- and uses it both as
the sample's contribution and as a Russian-roulette TERMINATION signal (kill a
path that has lost too much energy to matter). Both transfer exactly. ask_traced
(on KnowledgeStore and UnifiedMind) accumulates the product of per-hop cleanup
confidences, and the measurement is clean: on a dense, interfering 20-country
store, throughput SEPARATES correct chains (mean ~0.23) from wrong ones (~0.10),
and abstaining on the low-throughput half lifts answered-accuracy from 71% to
85% (+14 points). So throughput is a calibrated 'how much should you trust this
chained answer', and a chain whose throughput decays below a floor ABSTAINS
(returns None) instead of confidently emitting noise -- the ray that ran out of
energy contributes nothing. This is the first CALIBRATED CONFIDENCE in the
chaining machinery, and it falls straight out of the raytracing analogy: the
forward query is the camera ray, each memory hop a bounce, and the accumulated
product is exactly throughput. Surfaced in the console relations panel (answer +
throughput + per-hop confidences, with an honest 'abstained' when too low).

PRIOR-NEGATIVE REVISIT (do the new tools overturn any?). Reviewed the kept
negatives against the sequence/proof/throughput machinery. FLOCKING: retested
the idea that a competence signal could fix its core flaw ('no notion of which
neighbour is correct'). It does not -- in the under-trained regime where a
committee should earn its keep, no candidate escapes well enough to yield a
competence signal to weight by, so there is nothing reliable to align toward;
and once candidates ARE good enough to measure, best-pick already wins. The wall
is intrinsic: measurement requires the thing already work well enough to
measure, so consensus never beats it. Negative RE-CONFIRMED from a new angle.
FRACTAL ORGANIZER and SURFACE-TENSION/CURATION: their failures were about
layering machinery over a gate that already harvests the available signal (the
lean-1SE sweep, the organizer's own aggregation); the new tools do not change
that calculus -- recursive self-similarity stays where it is load-bearing (the
RP-tree/HoloForest index) and the permutation test stays where it discovers
order, neither displacing a working gate. The honest scorecard: the raytracing
parallel produced a genuine NEW capability (calibrated chain confidence), while
the revisit confirmed the old negatives still stand -- which is itself the
discipline working: a good negative is durable, and the bar for overturning one
is a measured win, not a fresh analogy.

MULTI-RAY CODA (one ray is a point sample; fire many and combine). Path tracing
fires dozens of rays per pixel and averages -- each noisy, the ensemble
converging -- and the parallel transfers to navigating the holographic space. The
honest journey mattered here: the FIRST attempts failed. Word-dropout views were
too CORRELATED (same bag minus a word -> same errors), and a naive throughput
VOTE across independent feature lenses (word/char/bigram/skip) only tied the best
single lens, because a confident-but-wrong weak lens (skip at 17%) dragged the
vote down -- the exact failure that sinks flocking. The fix is the path tracer's
discipline plus normalization: Z-SCORE each ray's per-label evidence before
summing, so an outlier ray cannot dominate. Measured: with lenses ranging
100/100/50/17%, the z-scored ensemble hits the BEST single lens's accuracy BLIND
-- without being told which lens to trust -- and on a noisy text task
classify_robust lifted 89% -> 100% while never regressing on clean queries
(100% -> 100%, stable across seeds). Wired as UnifiedMind.classify_robust (resampled
views, z-combined, returns an agreement signal = fraction of rays backing the
winner) and surfaced in the console classify panel. The concepts in play:
RAYTRACING (many rays, averaged), PROJECTION (each view a shadow of the input from
a different angle; the ensemble is the form no single shadow shows), COMPOSABILITY
(the views combine without rebuilding the encoder), INTEGRATION (multi-ray sits on
the same prototypes single-ray uses), and the recurring DISCIPLINE: the
structurally-appealing version (naive vote) had to be beaten by a measured one
(z-scored combine), and the failure modes were kept visible. The throughput work
gave one ray a calibrated confidence; this gives many rays a way to agree.

MULTI-RAY CHAINS -- TESTED, a clean negative for accuracy, with a kept artifact.
The thread's original target: fire several throughput-traced relation chains to
one answer and combine them (as multi-ray classification combined feature views).
It does NOT work for chains, and the WHY is the cleanup law itself. A relation
route is either through a UNIQUE-valued key (capital: one country per capital ->
find lands exactly, the chain is already 100%) or through a SHARED value
(currency: 8 for 40 countries -> find is fundamentally ambiguous, returns *some*
matching entity). Combining routes cannot manufacture information that is in no
single route: when one route is unique it is already exact (nothing to add), and
when all routes are shared they fail for the SAME reason (correlated errors), so
averaging just averages noise. Measured every way: naive throughput-vote across
routes made a perfect single route WORSE (100% -> 75%, a confident-wrong shared
route corrupting the unique one -- flocking's failure again); reliability-
weighting recovered the 100% but only MATCHED the best route, never beat it; and
where all routes were shared, even reliability-weighting (27%) lost to the best
single (52%). The contrast with multi-ray CLASSIFICATION is the lesson: there,
the feature-lens views had DE-CORRELATED errors (word vs char vs bigram fail on
different inputs), so z-scored combination genuinely recovered errors. Here the
routes share the cleanup law's one failure mode, so they do not de-correlate.
Multi-ray helps only when the rays' errors are independent -- the honest
precondition, and chains do not meet it.

THE KEPT ARTIFACT: route_reliability. The experiment surfaced a genuine,
self-measured signal -- a route's trustworthiness is 1 / mean fan-out of its
role's values (unique role = exact key = 1.0; shared role = ambiguous = low). No
magic number, the data's own fan-out inverted. It cleanly ranks which find()
operations to trust (capital 1.0, currency 0.20, continent 0.15), which is the
defensible thing to keep from a negative result. Concepts: this is DECOMPOSITION
(a chain factored into per-hop reliabilities), SELF-DISCOVERY (reliability
measured from the data, not declared), and the recurring DISCIPLINE -- the
structurally-appealing idea (more rays = better chains) was beaten by measurement,
the failure modes kept visible, and the one real discovery (route reliability)
extracted and pinned. A good negative leaves something behind.

PROJECTION-CREATES CODA (the shadow that creates new things -- a concept the user
flagged twice). Investigated whether the holographic space supports ANALOGY, and
the finding splits cleanly along retrieval vs generation. RETRIEVAL analogy
(a:b::c:?, FIND the existing d) hits a uniqueness wall: the cleanup law makes
every entity an exact key, so there is no graded 'nearness' for a transform to
climb -- shared-role projection finds the right REGION (another asian country)
but not the unique answer, and exact transform-set-matching finds nothing because
every entity differs from every other on the same role count. Multi-ray's lesson
again from a new angle: the structure that makes single lookups exact removes the
gradient retrieval-analogy needs. GENERATION, by contrast, works at 100%. blend()
synthesizes a NOVEL entity -- one record's frame with another's values projected
onto chosen roles ('france with japanese language and the yen') -- that exists in
no training data and decodes back to exactly the intended mix (100% over 40 random
blends). project_transform() does analogy AS CREATION: the a->b per-role delta
projected onto c generates a coherent new hybrid (japan's geography, germany's
distinctive capital+language), exact. The principle: synthesizing a SPECIFIED new
thing is well-posed where SEARCHING for an existing analogue is not -- creation
sidesteps the wall retrieval hits. Both wired (KnowledgeStore.blend/
project_transform/decode_record, UnifiedMind.blend over learned classes) and shown
in the tour. Concepts: PROJECTION (cast attributes onto a frame -- the shadow),
DECOMPOSITION (factor a record into roles), COMPOSABILITY (recombine factored
parts into a new whole), INTEGRATION (over the mind's own learned classes, decoded
from prototypes), and the recurring DISCIPLINE: the appealing idea (retrieval
analogy) was tested and found walled, the honest split (retrieval fails,
generation works) measured and kept, and the working half built. A good negative
names exactly where the line is -- here, between finding and making.

SCENE-BLEND CODA (projection-creation lifted to multi-object scenes -- the
resonator's factor-and-recombine finally load-bearing for GENERATION). The record
blend projects one entity's attributes onto another's frame; this does the same a
level up, where the parts must first be DISCOVERED. Given two scene VECTORS
(objects unknown), blend_scenes factors each into its objects via the resonator,
projects one factor across (scene A's forms wearing scene B's palette, or any
factor), and recomposes a NOVEL scene vector that holds the hybrid coherently --
and it factors back to exactly the intended objects (measured 100% across all
three factors and 2-4 separable objects). The full loop is decompose -> project
-> recompose, all through the resonator, with nothing known in advance but the
two composite vectors. This is the cleanest realization yet of 'projection of
complex shapes like a shadow to create new things': two scenes go in, their
structure and palette are factored apart and re-cast, and a scene neither
contained comes out. Honest boundary: recovery rides the resonator's capacity --
separable objects recover exactly, colliding objects degrade as multi-object
factoring always does (the same ~5-object ceiling charted in the scene panel).
Wired into SceneCoder.blend_scenes, shown in the showcase scene panel (a third
'projection blend' demo: scene A, scene B, and the synthesized blend side by
side, with the blend's recovered tags proving coherence) and in the tour.
Concepts: PROJECTION (cast a factor across), DECOMPOSITION (factor the scene into
objects, each into attributes), COMPOSABILITY (recombine factored parts into a new
whole), INTEGRATION (the resonator that was a recovery tool now drives
generation), and RAYTRACING's sibling -- a scene projected through a new 'lens'
casts a shadow that is itself a new scene. Creation over composites, proven by the
round-trip.

SCENE-MORPH CODA (projection unfolded over time -- projection meets sequence). The
question: does the holographic space support a CONTINUOUS morph between two
scenes? The honest answer found by measurement: NOT at the attribute level. A
linear interpolation of one colour atom red->blue is a CROSSFADE-WITH-SNAP -- the
colour readout decays from red and rises toward blue smoothly, but the resonator
REPORTS red until t~0.55 then flips hard to blue. The cleanup law holds discrete
coherent states with a hard boundary; there is no graded in-between colour. So a
smooth per-attribute morph is a clean boundary result, not a capability. The morph
that WORKS given the boundary: a SEQUENCE of discrete coherent frames. A control
parameter sweeps 0->1 and the objects adopt scene B's attribute one at a time
(object k flips at (k+1)/n), so every frame is a fully coherent scene that factors
EXACTLY, the first is A and the last has B's full pattern -- continuous control,
discrete honest outputs. And the loop closes with the sequence machinery: the
morph as a token sequence (flip-count per frame) passes the sequentiality
permutation test (z~10 vs its own shuffle, shuffle ~0.8) -- PROJECTION generates
the frames, SEQUENCE-DISCOVERY confirms the order, two threads of the project
meeting. Wired into SceneCoder.morph_scenes, surfaced in the scene panel as a
morph strip under the projection-blend demo (each frame drawn, all factoring
exactly), and in the tour. Concepts: PROJECTION (cast a factor across, now over
time), SEQUENCE/ORDER/TIME (the frames are a provably ordered run), COMPOSABILITY
(projection + sequence compose into a morph), INTEGRATION (the two big threads
join), and the recurring DISCIPLINE: the appealing 'smooth morph' was tested and
found walled (crossfade-with-snap), the boundary measured and kept, and the honest
working version (an ordered sequence of exact frames) built and proven ordered by
the existing permutation test. The morph is the shadow cast frame by frame.

CARDINALITY CODA (the last piece of scenes-as-fully-manipulable-composites: the
object COUNT itself morphs, and is self-measured). Two discoveries, one honest
boundary. (1) COUNT IS SELF-DISCOVERED: the scene is an unnormalised
superposition of near-orthogonal unit products, so round(||v||^2) IS the object
count -- measured 96% exact over n=1..7 (misses at high n where cross-terms
accumulate, the honest boundary; a peel-until-energy-drops alternative reached
90% and hit the resonator's capacity wall at n=6, so the norm wins). The
unnormalised-superposition decision, made long ago so explain-away could subtract
unit-scale atoms, pays off a second time: the count was sitting in the norm all
along. Nobody tells the system n anymore. (2) THE SCENE VECTOR IS ALGEBRAICALLY
EDITABLE: removing an object is factoring the scene and SUBTRACTING that object's
product (explain-away, built as a recovery step, repurposed as an editor); adding
is vector ADDITION of a new product. A chain of such edits walked a 3-object
scene down to one and up into a different 2-object scene, count discovered at
every frame, each frame factoring exactly at its discovered count, the final
edited vector holding EXACTLY the target scene -- never re-encoded from scratch.
morph_cardinality packages the chain (counts per frame [3,2,1,1,2] in the tour's
demo). Wired into SceneCoder (count_objects/add_object/remove_object/
morph_cardinality), surfaced in the scene panel (the blend caption now reports
the self-measured counts: 'nobody told the system how many') and the tour.
Concepts: SELF-DISCOVERY (the count measured from the vector's own norm),
DECOMPOSITION+COMPOSABILITY (objects subtracted and added as algebra),
SEQUENCE/TIME (the morph is an ordered chain of edits), COMPRESSION's cousin (the
norm carries a whole scene statistic for free), and INTEGRATION (explain-away,
the resonator, and the morph machinery all reusing one substrate). The composite
is no longer just decodable -- it is COUNTABLE and EDITABLE, which closes
'scenes as fully manipulable composite structures'.

PERCEPTION-AS-COMPOSITE + CREATURE STRESS CODA. The scene machinery applied to
the creature's WORLD: every visible thing (exit, poison, wall) is an object =
bind(type, position), the view their unnormalised superposition (WorldView). Two
properties fall out of the algebra. COUNT: round(||view||^2) is the number of
visible things. CHANGE: the DIFF of two snapshots is itself a composite of the
changes -- appeared objects positive, vanished negative -- so round(||diff||^2)
COUNTS the changes and peeling the diff (count-driven: exactly as many peels as
the diff's own norm says, NO threshold) NAMES each one. Identical products cancel
EXACTLY, so unchanged content vanishes from the diff. Measured on mutated 16x16
mazes (set-difference ground truth, after fixing my own test's double-counting of
cancelling mutations): change-count 100%, change-naming 100% over 40 trials with
1-3 simultaneous mutations. INTEGRATION, the round's centerpiece: a wall dropped
on the creature's learned route makes replay_plan break at EXACTLY that cell and
WorldView independently NAMES that wall -- perception explains the plan failure,
6/6 at 9x9 and 5/5 at 16x16, two systems cross-validating on one substrate. (A
replay_plan(reset=False) flag was needed and is documented: without fixed_seed a
reset re-carves a DIFFERENT maze -- the stress test surfaced that landmine.)

STRESS RESULTS (the honest sweep). 16x16 perfect mazes: 100% escape across SIX
seeds at the default budget. Braided+poison forks (11x11, braid 0.5): 100% on
both seeds tried. Full pipeline at 16x16: route z=81 over 97 cells, proven
executable, mutate->break->explain 5/5. THE WALL: 20x20 fails for some seeds.
Diagnosis by elimination: more episodes/steps lifted seed 5 from 0% to 83%
(budget-shaped), but seed 11 stayed 0% under bigger budgets AND a longer
discount horizon (gamma 0.985) -- the BOOTSTRAP problem: learning needs one
successful episode, and epsilon-greedy exploration reaching the deepest cell of
a 20x20 perfect maze within any reasonable step budget is exponentially unlucky
for some layouts. No reward ever arrives, so there is nothing to learn from. The
honest fix is a future thread: curiosity/novelty-driven exploration or a
curriculum (learn small, transfer deeper), not more of the same budget. Concepts:
SELF-DISCOVERY (count and changes from the vectors' own norms), DECOMPOSITION
(the world factored into typed objects), INTEGRATION (plan machinery + perception
cross-validating), COMPRESSION (one norm carries the change count), and the
DISCIPLINE: the stress test was allowed to find a real wall, the wall was probed
to a specific diagnosis, and the boundary recorded instead of papered over.

BOOTSTRAP-RESCUE CODA (the wall named its fix; the fix had to be measured into
shape). The 20x20 thread, pulled end to end. BASELINE: under the standard
decaying-epsilon schedule, seed 11 trains to ZERO escapes in 400 episodes, and
plain probes 0% at EVERY budget and horizon tried -- the wall is real. (Honest
refinement found while pinning: sustained HIGH epsilon occasionally escapes --
7/80 at constant 0.5 -- so the mechanism is not 'luck is hopeless' but 'the
loop-attractor policy locks in as epsilon decays, before luck consolidates'.
The first phrasing was overstated; the docstrings were corrected.) THE RESCUE,
three pieces, each forced by a measured failure: (1) CURIOSITY -- a first-visit
cell bonus of exit_reward / n_free_cells, the world's own arithmetic, full
coverage summing to exactly one exit reward -- finds the first success within
episodes (4 vs never). But the policy could not HOLD it, and the reason is
sharp: visited-ness is not in the creature's state, so curiosity returns are
unlearnable noise -- curiosity drives behaviour, it cannot be encoded; hence the
HANDOVER (off at first escape). (2) REHEARSAL -- store successful trajectories,
re-remember one per episode -- consolidates rare successes (87 then 263 training
escapes) instead of letting hundreds of failures drown them. (3) CAPACITY --
greedy still looped in a 14-cell attractor at 256/15; 512/30 (and memory_cap
scaled as 800*k) let the policy hold the maze: 263/400 training escapes, greedy
4/6. THEN MEASUREMENT CUT THE OTHER WAY: on seed 5, where a bigger budget
already found successes by luck (83% plain), the SAME protocol HURT -- 0% with
it, 33% with rehearsal alone: curiosity noise and rehearsed early meanders
degrade a signal that is already arriving. So the integration is a RESCUE
SUMMONED BY SELF-MEASUREMENT: candidates run plain; only a candidate finishing
with zero escapes (starvation -- the data's own signal, no threshold) flips the
bootstrap on for subsequent candidates. bootstrap="auto" is learn_maze's
default; True forces, False disables. VERIFIED: seed 5 routes to the plain path
(83%), seed 11 starves, rescues, and probes 100%; the 16x16 six-seed sweep
stays 100%. Pinned end-to-end (plain 0% vs rescued >=2/3 on the formerly
impossible seed; flag modes at 9x9). Concepts: SLIME-MOLD's spirit (coverage-
driven exploration laying crumbs), SELF-DISCOVERY (starvation observed, not
configured), COMPRESSION (one success rehearsed into many examples),
SEQUENCE/TIME (the winning trajectory as the thing consolidated), INTEGRATION
(speculate-measure-adopt applied to the training protocol itself), and the
DISCIPLINE at its sharpest: every piece of the design -- the bonus magnitude,
the handover, the rehearsal, the capacity, and above all the decision to make
the whole thing a rescue rather than a default -- was forced by a specific
measured failure, and the one overstated claim found along the way was
corrected in place.

MARKET-DATA CODA (real numeric time series -- DEX candles -- on the substrate;
the honest split between memory and prophecy). User-supplied DAI/WETH data, 100
one-minute candles, checked in at data/dai_weth_ohlcv.json. WINS, each measured:
(1) A CANDLE IS ONE RECORD -- five roles bound to SCALAR codes (ScalarEncoder's
graded similarity: near numbers stay near, which symbols cannot give numbers)
and bundled; round-trip decode error 1.6-2.9 bp vs the data's own 8.2 bp return
sd -- resolution finer than the signal. (2) THE PERMUTATION TEST REDISCOVERS
MARKET STRUCTURE: price levels provably ordered (z=+6.8 vs their own shuffle),
return signs indistinguishable from shuffle (z=-0.6) -- the efficient-market
property, found by the same instrument that proves creature routes. (3) NOVELTY
at the candle level (similarity to nearest prior, flagged z>2 below mean -- the
data's own scale) catches both real anomalies: the 2685-volume/17bp-range
candle (z=4.2) and the +21bp swing (z=5.7). KEPT NEGATIVES: (a) PREDICTION IS A
COIN FLIP -- walk-forward next-sign from the nearest motif scores 49%, and so
does EVERYTHING (majority 55%, persistence 54%, anti-persistence 46%), all
inside the binomial 39-61% chance band at n=82; no edge demonstrable by any
method tried, and the test suite PINS the honest claim (inside the band), not a
win. Recall is memory, not prophecy. (b) WINDOW-level novelty dilutes
single-candle anomalies (one outlier role among 18 in a sum) and is biased
early -- novelty must run at the granularity of the thing that can be anomalous.
Wired as holographic_market.py (CandleCoder: encode/decode_candle, feature_vec,
window_vec/nearest_motif, novelty) with the real dataset checked in and four
tests pinning wins and negatives alike. Concepts: SEQUENCE/ORDER/TIME (the
permutation test on real time), COMPRESSION (a candle in one vector),
SELF-DISCOVERY (anomalies at the data's own scale), DECOMPOSITION (roles
unbound back to numbers), and the DISCIPLINE where it matters most -- a domain
that breeds overclaiming got the chance band printed next to every accuracy.
MARKET ROUND 2 -- SCALE CODA (the user's 'bigger dataset' turned out to be a
DIFFERENT INSTRUMENT-REGIME, and the engine's instruments responded). The
upload, named fiveMin.json, is actually ~1-second SOL ticks: 15,793 points over
2.2 days (15,448 jupiter ticks in ~50-tick bursts every ~5 minutes plus 419
coingecko 5-minute points; one 21-hour hole). Checked in compressed as
data/sol_5min.npz (37KB). All analysis WITHIN-BURST only (gap <= 2s) -- a
return is never computed across a hole. THE REGIME FLIP, the round's
centerpiece: the same sequentiality test that called DAI-minute return signs
shuffle-like (z=-0.6) calls tick signs STRONGLY ORDERED (z=+44.1, levels
z=+59.5, shuffled control -0.9). Tick momentum (+0.198 sign autocorr, 59.9%
sign-repeat) is real structure and the permutation test found it: one
instrument, two regimes, opposite honest verdicts. THE PREDICTION REDESIGN,
forced by the data's shape: 88% of ticks are FLAT, and the first test broke on
zero-inflation (persistence compared against an often-zero last sign). Honest
framings: (A) 3-class next tick {U,D,F} -- always-predict-flat 90.2% beats
motif 89.2% and persist3 86.4%, kept negative; (B) direction of NEXT MOVE
(target nonzero, persistence = last NONZERO sign, motif calls the first
nonzero sign within 30 ticks of the matched window), n=1454, band 50+/-2.6%:
persistence 60.2% OUTSIDE the band -- the momentum edge is REAL and now
PROVEN -- and the holographic motif 54.1%, also outside chance (genuine
signal) but DECISIVELY BELOW the trivial rule. Measurement beats
sophistication: the flocking lesson again. The motif's value is memory and
novelty, not direction-calling; the measured direction tool here is
last-nonzero-sign. Implementation note that mattered: 15k windows of K=6
position-bound scalar codes at dim 512, rows L2-normalised into one float32
matrix, walk-forward similarity as a growing matvec -- 8s to encode, 1s to
walk. Concepts: SEQUENCE/TIME (the order test doing real science on real
time), SELF-DISCOVERY (the verdicts come from the data, and they DIFFER per
regime), HONESTY UNDER TEMPTATION (a 54.1%-outside-the-band result that a
less careful round would headline -- pinned instead as 'below persistence'),
and the DISCIPLINE: the zero-inflation bug was found because the first
numbers looked impossible, and the redesign was recorded, not hidden.

MARKET ROUND 3 -- STRUCTURE MAP + RAY-PROJECTED TARGETS (the user's reframe:
find the structure first, then project identified patterns forward for price
targets 'with proofs' -- and the proofs came). LOCATING THE STRUCTURE, three
measurements: (1) the momentum is INTRA-BURST ONLY -- sign persistence 60.5%
+/- 2.7 within bursts, 49.4% (chance) across the ~5-minute holes, and no
burst-to-burst drift carry-over (45.7% +/- 12): sub-minute microstructure that
dies at every gap. (2) Bursts are mildly drifty (13% exceed binomial z>2
imbalance vs 5% expected). (3) MOVE-SHAPES DO NOT RECUR: vs order-shuffled
within-burst surrogates (same marginal, destroyed order) nearest-neighbour
similarity shows no excess (z=-0.6) -- the order structure is momentum and
drift, NOT repeating chart patterns. (A first version of that test saturated
at 0.9997 similarity because 88%-flat windows all match each other -- the
statistic must run on the MOVE sequence; recorded.) THE RAY PROJECTION, the
user's idea built honestly: at a matched K-move pattern, R rays = the R most
similar past windows, each carrying its next-H-move cumulative return; the
bundle's quantiles are the price-target distribution. Proper-scored
(pinball + coverage), walk-forward, selection done cleanly: R=80 chosen on the
FIRST half only, scored on the untouched second half -- rays beat the
unconditional outcome distribution by 0.134 bp/point, paired z=+3.3, with ~13%
tighter 80% intervals near nominal coverage (85%) while the baseline
over-covers (89%). THE INTERPRETATION, kept precise: the pattern does NOT call
direction (persistence owns that contest, prior round) -- it locates the
CURRENT CONTEXT'S OUTCOME SCALE, yielding sharper calibrated targets. And the
kept negative inside the win: ray-similarity CONFIDENCE gates difficulty, not
skill -- the confident quartile improves the unconditional baseline exactly as
much (pinball 4.35 vs 4.38), so it is a when-to-trust gauge, not an edge.
Wired as move_series + RayProjector in holographic_market.py, three pinned
tests (intra-burst momentum location; no shape recurrence; the held-out ray
win at z>2), a tour line projecting a live target. Concepts: RAYTRACING
(rays through pattern-space instead of relation-space, same idea: many paths,
each carrying evidence, bundled with confidence), PROJECTION-CREATES (the
target distribution is generated from matched precedents), SELF-DISCOVERY
(every threshold is the data's own scale -- p99 encode range, binomial bands,
split-half selection), and the DISCIPLINE: the win was only claimed after the
selection was quarantined to half the data and the improvement survived on
the rest.

MARKET ROUND 4 -- THE HORIZON MAP (the user asked plainly: how accurately can
it predict future prices, per timeframe? The honest answer is a table, and the
table is mostly the diffusion law). MOVES-HORIZON SWEEP (R frozen at the
previously validated 80, nothing newly selected; second-half scoring):
DIRECTION is predictable ONE move ahead only -- persistence 58.3% (outside the
4.6% band); by H=2 it is 52.1% (inside) and stays at chance; rays never beat
persistence on direction at any H. POINT prediction NEVER beats predict-zero:
the median target's MAE matches |outcome| at every horizon (2.27 vs 2.28 bp at
H=1, slightly worse beyond), with error growing like sqrt(H) (2.3 -> 7.2 bp
from H=1 to 8) -- pure diffusion. INTERVALS, the validated value, decay
honestly: the rays' pinball advantage over the unconditional distribution is
z=+4.8 / +2.2 / +3.3 at H=1/2/3, a wash at H=5, gone (z=-1.3) at H=8.
WALL-CLOCK: within bursts the per-second direction edge dilutes to ~51%
(50.8-51.5% at 5-30s; the 60% momentum is per-MOVE and flat ticks wash it
out), MAE-zero 1.2 -> 5.4 bp from 5s to 30s; ACROSS the holes (one close per
burst, ~5-min grid) direction is CHANCE at 5 and 10 minutes (54.8 +/- 4.9,
n=409) -- confirming the structure map at the price level. THE SUMMARY, kept
plain: this data is diffusion plus a one-move memory; the only validated
predictive product is the calibrated target INTERVAL 1-3 moves out; point
forecasts beyond 'no change' are not achievable, and nothing carries across
the gaps. Pinned: held-out rays win at H=1, gone by H=8 -- the suite enforces
that the claim stays horizon-qualified. Concepts: SEQUENCE/TIME (the horizon
as the experimental axis), HONESTY (the user asked for accuracy and the
answer includes 'predict-zero is unbeatable for points'), SELF-DISCOVERY
(every verdict from bands and paired z, no thresholds), and the project's
oldest rule -- the negative is the result: knowing WHERE prediction ends is
the useful thing a measurement engine can say about a market.

PHYSICS CODA (the user asked: what if the price were an object with mass and
velocity -- can the system have a base understanding of physics, and does it
help? The answer split into a discovery, a compression win, and a falsified
metaphor, all measured). THE DISCOVERY: additive kinematics is NATIVE to the
substrate. For the fractional-power scalar code, encode(a+b) ==
bind(encode(a), encode(b)) EXACTLY (frequency phases multiply) -- translation
in value-space IS the binding operation. So x += v is one binding; v += a is
the same trick one level up; a 15-step constant-acceleration trajectory
integrates by pure vector algebra with max decode error 0.06; and velocity is
READ OFF two observed positions by unbinding (+3.44 vs true +3.50). The system
does not need to be taught linear motion -- it lives in the encoder. (Two test
bugs found and kept: the discrete closed form is x0+v0*t+a*t(t-1)/2, and the
first 'failure' was exactly the t(t+1)/2-vs-t(t-1)/2 difference, a*t; the
range-boundary case also needed honest parameters.) THE BOUNDARIES:
trajectories must stay inside the encoder's lo..hi (the decode grid), and
MULTIPLICATIVE dynamics (damping, oscillation) are not native -- binding adds,
never scales; nonlinear physics must be learned, not inherited. THE MARKET
VERDICT, against the validated tools: at H=1 the two-number particle STATE
(v = last move, a = its change) gives ray-targets EQUIVALENT to the 5-move
shape rays (paired z=+0.4) while beating the unconditional distribution
(z=+3.5) -- at one step ahead the market's structure IS kinematic, and physics
buys understanding-as-COMPRESSION (five numbers were never needed; two
suffice). At H=3 the shape still wins (z=-2.1): outcome-scale context exceeds
the instantaneous state. AND THE FALSIFIED METAPHOR, pinned: prices have NO
INERTIA -- kinematic extrapolation as a point forecast loses to predict-zero
at every horizon (3.43 vs 2.41 bp at H=1, 17.5 vs 4.1 at H=3). The velocity's
SIGN persists one tick (that is the momentum, already known); its MAGNITUDE
mean-reverts immediately. The price is not a coasting mass; it is diffusion
with a one-tick memory -- and the physics framing is precisely what made that
sentence sayable. ('Mass' as volume/liquidity: untestable on volumeless ticks;
recorded, not guessed.) Wired as holographic_physics.py (Kinematics:
trajectory-by-binding, read_velocity, range guard), four pinned tests
including the no-inertia negative and the H=1 equivalence. Concepts:
INTEGRATION at its deepest (an entire domain of law inherited from the
encoder's algebra, zero new machinery), COMPRESSION (two state numbers
replacing five pattern numbers, proven equivalent), SELF-DISCOVERY (state read
off data by unbinding), and HONESTY (the metaphor was tested, half survived,
and the dead half is pinned so it stays dead).


MARKET ROUND 5 -- SCALE + CROSS-INSTRUMENT (the user pulled 1000 DAI/WETH
candles from the GeckoTerminal URL; 10x the first slice, ~35h, a 3x-tighter
chance band, with real outliers including a 53652-volume bar). EVERY STRUCTURAL
FINDING HELD AND SHARPENED: record round-trip MAE 1.54bp vs the data's 10bp
return sd (finer than the signal); permutation test levels z=+100, return signs
z=-1.0 (STILL shuffle-like -- the efficient-market verdict survived 10x the data
and the tighter band); candle novelty flagged the 53652-volume/biggest-range
bar at z=8.1 (top novelties candles 30/31 at z=25/10 -- a sharp local regime).
PREDICTION STILL CHANCE: next-sign motif 49.7%, majority 49.9%, persist 52.9%,
flip 47.1%, ALL inside the now-+/-3.2% band -- no minute-scale directional edge
in DAI/WETH exists to find, and the tighter band makes that a stronger
statement, not a weaker one. THE KEY RESULT -- CROSS-INSTRUMENT REPRODUCTION:
the ray-projected calibrated-interval win, first validated on SOL ticks,
REPRODUCES on DAI candles (a different instrument, different microstructure):
held-out H=3 pinball improvement +0.31 bp/pt, paired z=+2.98, ~9% tighter
intervals (32.7 vs 36.0bp) near nominal coverage. The product generalizes. AND
THE HONEST DIFFERENCE: the edge lives at a DIFFERENT horizon per instrument --
H=1-3 on momentum-driven SOL ticks, H=3-5 on mean-reverting DAI candles (whose
-0.175 lag-1 autocorr also kills directional persistence, 41-49%, at/below
chance) -- same conclusion (the interval is the product, not the direction)
reached from opposite microstructure. Pinned: structure-holds-at-scale and
ray-win-reproduces-on-second-instrument. Saved data/dai_weth_big.json.
Concepts: HONEST MEASUREMENT (the negative got STRONGER with more data, the
mark of a real null), CROSS-VALIDATION (a claim that survives a second
instrument is a claim worth keeping), SELF-DISCOVERY (the band tightened by
sqrt(n) exactly as predicted and the verdicts moved accordingly), and the
project's throughline: the engine does not predict prices -- it locates and
calibrates uncertainty, and that capability now holds across two markets.

COMPRESSION ROUND -- VIDEO + AUDIO ON THE SUBSTRATE (the user connected the
physics COMPRESSION win to the project's compression machinery and asked what
video and audio coding could teach). THE CONNECTION, which turned out to be
exact: the physics round proved translation-in-value-space IS binding
(encode(a+b)==bind(encode(a),encode(b))). A spatial SHIFT is translation in
pixel coordinates, so a rigidly moving object is ONE operator applied
repeatedly -- precisely the redundancy a video codec removes with keyframes +
motion-compensated prediction. The lesson video teaches (store the TRANSFORM,
not the state) is one the binding algebra already embodies. WHAT WORKS,
measured: (1) MOTION COMPENSATION ZEROES THE RESIDUAL -- for whole-pixel rigid
translation a one-number motion search recovers the shift exactly and
frame[t]-shift(frame[t-1]) is L2 0.000; the sequence collapses to keyframe +
one int/frame. (2) GOP CODING WINS STRICTLY on rigid motion: ~10% smaller AND
+0.4 dB vs per-frame INTRA storage (the residual coder spends coefficients on
almost nothing). THE HONEST BOUNDARY, equally measured and the reason it is not
magic: (3) DEFORMATION BREAKS IT -- a growing/morphing object is not a rigid
shift, the motion model is wrong, residuals stay large, error accumulates
against lossy reconstructions, and GOP LOSES -3.7 dB at matched budget. Motion
compensation pays exactly when motion is the dominant change -- the same
condition under which the algebra represents the change exactly. (The journey
kept its negatives: naive frame-differencing first looked WORSE than per-frame,
2136 vs 562 DCT coeffs, because a moving sharp edge makes broadband residual
energy -- which is exactly why motion compensation, not differencing, is the
codec's actual trick; recorded.) AUDIO, the unifying probe: MP3 keeps the
loudest frequency components, and a 1-D signal's DCT IS that spectral basis, so
the SAME HolographicImage machinery compresses a tone+transient signal (8x at
34 dB) and survives 30% erasure with NO extra loss -- spatial (image) and
spectral (audio) are the same operation on the substrate; the compressor is
basis-agnostic. Wired as holographic_video.py (HolographicVideo: encode/decode
GOP coder, estimate_shift, intra_baseline) with five pinned tests including the
deformation negative and the audio unification. Concepts: COMPRESSION (the
round's spine -- temporal redundancy removed by the same binding that removed
the physics state redundancy), INTEGRATION (image DCT machinery + physics
shift-as-binding + holographic plates, one pipeline), TIME/SEQUENCE (the GOP is
an ordered predict-residual chain), and HONESTY (the win is fenced by the
deformation boundary and the differencing dead-end, both measured and kept).
VERSIONED HISTORY + SUB-PIXEL MOTION (the user connected the video round to two
things: rollback after a caught reorganization mistake, and the lost record of
HOW the store changed -- proposing each version be stored as a frame; plus the
greenlit sub-pixel motion extension). VERSION HISTORY IS A VIDEO. A sequence of
store versions is exactly inter-frame-redundant: consecutive versions share
almost everything, so the GOP structure (keyframe + deltas) stores the whole
timeline cheaply and any version is recoverable. THE TWIST that makes it its own
thing: rollback needs the EXACT prior state, so the deltas are LOSSLESS, where
video's were lossy spectral truncation -- the two complete the compression
picture (lossy spectral for perceptual data; lossless sparse-delta for state and
history). THE GIT LESSON, learned by measurement: version rows by a STABLE ID,
not by position. A naive entry-wise diff calls a row DELETION an 86%-of-matrix
change (every later row's index shifts -- an alignment artifact); content-keyed
rows make a delete cost ONE id and the real change sizes show reorganization is
genuinely sparse (insert/split/relabel/merge at 0-9%), compressing the history
~29x losslessly. WHAT WORKS, measured: (1) checkout(v) reconstructs ANY version
EXACTLY (lossless, verified entry-wise across a 12-edit run); (2) reorganization
history compresses (full snapshots / delta storage > 3x, up to ~29x on
structural edits); (3) PROOF-GATED COMMITS + ROLLBACK -- the rescue the user
asked for: a buggy reorganization (collapses a prototype to a degenerate
near-zero row) FAILS its coherence proof, is REJECTED, and the store stays at the
last valid version, while the rejected attempt is still in the audit log; an
explicit rollback(v) restores a past version EXACTLY and is itself recorded
(nothing erased -- the timeline is append-only). THE HONEST BOUNDARY (the
'deformation' analog from video): a DENSE update (a gradient/learning step
nudging every entry) changes 100% of the matrix, so delta coding does NOT
compress it -- versioning is for STRUCTURAL history (reorganization, edits,
discrete commits), not dense trajectories; pinned. SUB-PIXEL MOTION (the
extension): a fractional drift (1.7 px/frame) is recovered EXACTLY by a Fourier-
shift motion search (residual to numerical zero) where integer search rounds and
leaves residual -- because a pixel shift is a phase ramp in frequency, the
scalar code's fractional-power principle in 2-D; the same property powers both
the physics win and now sub-pixel video. Wired as holographic_history.py
(VersionedStore: commit/checkout/rollback, proof-gating, audit log, byte
accounting) and fourier_shift/estimate_subpixel_shift in holographic_video.py;
ten pinned tests across the two. Concepts: COMPRESSION (history as inter-frame
redundancy, now LOSSLESS), TIME/HISTORY (the always-now substrate gains a
replayable past), PROOF+INTEGRITY (commits gated by validity, the existing proof
discipline made operational as a merge gate), ROLLBACK (the missing undo, built
on compression), and HONESTY (the dense boundary and the position-vs-id artifact
both measured and kept). The thread the user pulled turned the video codec into
a version-control system for the mind's own state.

SPRITE-PACK v2 -- ORIENTATION CHOICE (the user asked to recheck every layer of
the stack for compression now that the substrate has advanced, starting with the
712-sprite test set whose last pack was already strong at 68,632 bytes / 8.7% of
the loose GIFs). THE DISCIPLINE PAID OFF BY RULING THINGS OUT FIRST: the
advancements that helped video did NOT transfer naively -- consecutive animation
frames differ in 29% of pixels (not near-zero), only 1% of planes are exact
dups, and XOR/left/up delta filters all made it WORSE (143K vs 65K), the same
lesson video taught (differencing breaks the byte runs LZMA exploits). Greedy
similarity-reordering of planes also did not pay -- LZMA's match window already
finds most cross-sprite structure, so the permutation (1,120 bytes) cost more
than the adjacency saved. THE REAL WIN, found by measuring orientations: these
are CHARACTER sprites, and a column down the body is more self-similar than a row
across it, so storing index planes COLUMN-MAJOR gives LZMA longer runs -- index
body 54,292 vs 64,884 bytes (-16%), full pack 58,041 vs 68,632 (-15%), bit-exact
lossless. Transpose is free; pack() now tries both orientations and keeps the
smaller, recording the choice in a 1-byte flag (v2 format), so it can NEVER
regress below row-major; v1 blobs still decode (back-compat verified). Two pinned
tests (the 712-set shrinks <62KB and round-trips exact; mixed-size sprites
round-trip through the transpose path, exercising per-sprite (h,w)). Concepts:
COMPRESSION (a genuine 15% lossless gain on real data by changing
REPRESENTATION, not adding a coder -- the project's oldest compression lesson),
SELF-DISCOVERY (the orientation is chosen per-pack by measurement, never
assumed), HONESTY (three plausible advancements -- inter-frame delta, reordering,
predictive filters -- were each measured and REJECTED, and only the one that
actually won was kept), and the throughline: the win came from understanding the
DATA (vertical character structure), exactly as the original v1 win came from
understanding that sprites are palette images. Checking every layer meant
re-measuring, and the measurement found 15% that assumption would have missed.

ORIENTATION PRINCIPLE GENERALIZED (the user asked whether the sprite-pack v2 win
applies to the other places we compress). The win abstracted to a PRINCIPLE: try
cheap REVERSIBLE reorientations of the data and keep whichever the coder likes
best, per-instance, recorded in a flag -- a free, safe option that adapts to the
data and can never regress. Swept all three real byte-compressors in the stack.
(1) image_vault.py -- the general adaptive chooser, the highest-leverage target:
its `palette` and `lzma` methods do the same index-plane / raw-pixel LZMA where
transpose helped, but tried only one orientation. Added _orient(): both methods
now try row- and column-major and keep the smaller, prefixing a 1-byte flag to
the body; decode reads the flag and untransposes per-image (respecting each
image's own (h,w), verified on mixed sizes). VERIFIED ADAPTIVE: on character
sprites it chooses transpose (flag 1, palette body 64,884->54,292-class win) and
round-trips exact; on HORIZONTALLY-structured data (X-gradients) it correctly
chooses row (flag 0) -- proving the choice is made by measurement, not assumed,
which is exactly why it is safe. (2) pack_sprites.py already has v2 transpose
(prior round). (3) holographic_pack.py -- the zlib DELTA packer for related
truecolour images: MEASURED and DECLINED. On its intended data (a base image
with localized edits) transpose changed the size by 6 bytes (16,338 vs 16,344) --
the delta residual is already structure-free, so there is no directional
redundancy to exploit; it only helps on sprites, which are not this module's job.
Adding it there would be complexity for zero gain, so it was NOT added. The
honest throughline: the principle generalizes precisely where the DATA has
directional structure (sprites: vertical character self-similarity), and the
measurement is what tells you where that is -- the same principle that found the
original win is the one that says where to stop. Pinned: two vault tests (adaptive
orientation chooses correctly on sprites vs gradients and both round-trip; mixed
sizes through the transpose decode). Concepts: COMPRESSION (one principle now
serving three layers, applied only where measured to pay), SELF-DISCOVERY (the
orientation chosen per-set from the bytes), HONESTY (the delta-packer case
measured and rejected -- a negative kept as a design decision), and INTEGRATION
(the vault inherits the sprite win automatically, no special-casing).


DICTIONARY-FIRST CURRICULUM (the user asked what happens if the brain learns a
DICTIONARY before any other reading, plus a GRAMMAR book and an ENCYCLOPEDIA).
Could not feed Webster's literally (no open-web access here), but used WordNet --
a real machine-readable dictionary, 82k noun definitions + synonyms + is-a
hierarchy -- and measured the MECHANISM honestly. THE HYPOTHESIS: a word's
meaning = bundle of the meaning vectors of the words in its DEFINITION; a
dictionary is self-referential (definitions use defined words), so this is a
fixed-point iteration on the definition graph -- the resonator/cleanup dynamic
applied to a lexicon. Downstream test: do synonyms become more similar than
random word pairs (a d-prime)? RESULTS: random vectors d'=+0.0 (the null,
synonyms no closer than random); ONE definition pass d'=+1.5 (the dictionary
bootstraps meaning hard -- a few defining words beat nothing); ITERATED, d' PEAKS
at ~3 passes (d'=+1.9) then DECAYS as meaning over-diffuses through the graph --
the same fixed-point-then-collapse sweet spot the resonator shows, now on a
dictionary. CO-OCCURRENCE reading alone (Brown corpus) only d'=+0.5: thousands of
noisy sentences carry far LESS signal than concentrated definitions. THE
CURRICULUM VERDICT, two-sided and the valuable part: dictionary-THEN-reading
BEATS reading-alone (+0.8 d', confirming the user's intuition that seeding
helps), BUT full-rate reading WASHES OUT the clean definitional structure (1.9 ->
1.3) -- the dictionary is so much cleaner than prose that reading must REFINE,
not overwrite; gentle (low-rate) reading preserves the seed (-> 1.8, approaching
dict-alone). The honest rule: learn the dictionary first, then read CAREFULLY.
THE THREE LAYERS map onto three subsystems: DICTIONARY -> meaning vectors (this
module, the strongest seed for raw word meaning); GRAMMAR -> SEQUENCE structure
(holographic_sequence's sequentiality_z scores which word orders are valid);
ENCYCLOPEDIA -> RELATIONAL fact (holographic_relations' KnowledgeStore + the
ask/raytrace machinery -- is-a/has-a, already built). Wired as
holographic_lexicon.py (Lexicon: bootstrap/read/similarity/nearest/separation),
five pinned tests (mechanism, recursion sweet spot, gentle-reading-preserves-seed
on a hermetic synthetic dictionary; a WordNet-gated scale check). Concepts:
RECURSION/FIXED-POINT (the dictionary as a self-referential system bootstrapping
meaning, peaking then collapsing), COMPRESSION (a definition is concentrated
meaning -- a few words outvalue thousands of sentences), SELF-DISCOVERY (meaning
emerges from the definition graph's own structure, no labels), CURRICULUM
(seed-then-refine, with the measured caveat that the seed must be protected), and
HONESTY (the intuition was confirmed AND corrected -- seeding helps, but
over-reading hurts, and both directions were measured).

ENCYCLOPEDIA LAYER -- UNDERSTANDING BEYOND WORDS (the user asked, after the
dictionary, to add a basic encyclopedia and see what understanding we have about
complex topics aside from word meaning). Built from WordNet's is_a hierarchy +
part-of as a real encyclopedia, keyed by SYNSET so senses don't collide
(dog.n.01 is its own concept), loaded into a KnowledgeStore where each concept is
bundle(bind(is_a,parent), bind(has,part)) and a taxonomy chain is a RELATION RAY
(each is_a hop a bounce, cleanup-to-symbol the surface hit, cleanup confidence
the reflectance). RESULTS: (1) one-hop is_a retrieval EXACT, 100% over hundreds of
links. (2) MULTI-HOP TAXONOMY exact too -- but only once measured honestly. A
first naive measurement read 43% and looked like failure; it was a TEST artifact
-- the ground-truth chain used a word's dominant sense while the store was built
from a different sense, and chains ran off the edge of the stored world. Rebuilt
as a CLOSED WORLD (every ancestor stored) with consistent senses, the climb is
100% exact at 2, 3 AND 4 hops. KEPT LESSON: a low score is a claim to
investigate, not to report -- here the model was right and the test was wrong,
and saying so is the honest move. (3) THROUGHPUT TRACKS DEPTH: chain confidence
decays 0.50 -> 0.36 -> 0.25 from 2 to 4 hops, so the relation ray reports how far
a deduction has traveled and can ABSTAIN when it fades. (4) THE POINT OF THE
QUESTION -- understanding beyond words: taxonomic SIBLINGS (sharing an is_a
parent) are related knowledge even when their DEFINITIONS barely overlap;
measured, ~58% of sibling pairs share at most ONE definition word, so the
dictionary is nearly blind to their kinship while the encyclopedia links them
through the shared parent. Relatedness lives in the STRUCTURE, not the words.
Wired as holographic_encyclopedia.py (Encyclopedia: add/is_a/climb/
is_a_transitive/siblings/relatedness; Curriculum: stacks the layers and reports a
capability each adds that the previous lacks). Six pinned tests (one-hop,
multi-hop+throughput, abstention, structural-relatedness, curriculum, WordNet-
gated closed-world scale). Two of my own test assertions were WRONG first (dog
and rose DO meet at 'organism'; siblings score 1/3 not 1) and the code was right
-- fixed the tests, kept the lesson. THE FULL CURRICULUM now stands: dictionary
(holographic_lexicon, word meaning) -> grammar (holographic_sequence's
sequentiality_z, valid order) -> encyclopedia (this, relational knowledge), each
justified by a measured capability the previous layer lacks. Concepts: RELATIONS/
RAYTRACING (taxonomy as path tracing with calibrated throughput), HONESTY (the
43%-was-the-test story, twice over with my own assertions), SELF-DISCOVERY
(relatedness emerges from the graph), COMPRESSION (a concept's place in the web
is knowledge its definition text does not carry), and CURRICULUM (three kinds of
structure for three subsystems, stacked on evidence).

WIRED TO THE BRAIN -- CURRICULUM NATIVE IN UnifiedMind + LIVE PANEL (the user
said: keep going, and don't leave functionality in tests -- wire it to the
brain). The dictionary and encyclopedia had lived as standalone modules; now they
are CAPABILITIES OF THE MIND ITSELF. UnifiedMind gained: learn_dictionary(defs)
-- bootstraps word meaning by fixed-point iteration on the definition graph and
writes the result straight into its OWN text encoder's word vectors, so every
downstream text perception inherits definitional meaning; define(word) -- nearest
words by that learned meaning (empty for words never in the dictionary, so no
spurious neighbours, a bug caught by an honest unknown-word test); learn_encyclo-
pedia(facts) -- absorbs each concept as a role-bound record into its OWN memory
via the existing learn(...,'record') path, so is_a knowledge is filed exactly
like every other record; climb/is_a -- walk the is_a chain over the mind's own
memory as a path-traced ray (read_role per hop, throughput the accumulated
confidence, abstains when it fades). The brain now defines 'dog' -> canine/fox/
animal AND climbs dog -> canine -> carnivore -> mammal -> animal -> organism, the
dictionary living in its encoder and the encyclopedia in its memory, ONE mind.
WIRED TO THE APP too: a 'Dictionary + encyclopedia (curriculum)' dataset (hand-
built, no network) trains one UnifiedMind on the dictionary then the encyclopedia
and reports one-hop is_a accuracy (100%); a /api/unified/curriculum endpoint and
a '3 3/4 - curriculum' panel let you type a word and see its meaning-neighbours
(dictionary layer) beside its is_a chain climbed with throughput (encyclopedia
layer), side by side over the same brain. Five app tests (dataset offered + needs
no network, loads + learns both layers, query returns meaning + chain, unknown
word handled, panel present) and one brain test (learns both natively, climbs
exactly, throughput decays). Concepts: INTEGRATION (the curriculum is in the
model the whole app already drives, not a side table -- the user's explicit
ask), SELF-DISCOVERY (meaning from definitions, knowledge from relations, both in
the mind's own stores), HONESTY (the unknown-word spurious-neighbour bug found
and fixed by test), and CURRICULUM (dictionary -> encyclopedia, learned natively
and queryable live).

QUESTION ROUTER -- ANSWERING vs COMPLETING (the user noticed that typing into the
UI makes the brain COMPLETE the sentence, and asked how to get it to respond
conversationally / answer a question). The honest framing first: this mind is NOT
a language model and does not converse -- the sentence-completion they saw is the
generation (n-gram/chunk) path, which only continues text. But the mind holds
real knowledge, and most questions have a SHAPE that maps to one of its actual
operations. Added UnifiedMind.answer(question): a template router (keyword/regex
matching -- explicitly NOT natural-language understanding, and it says so) that
recognizes question forms and dispatches to the real operation: 'what is X' /
'define X' / 'what is X like' -> define() meaning neighbours + climb() is_a
chain; 'is X a Y' -> is_a() taxonomic membership (both polarities); 'what is the
ROLE of CONCEPT' -> read_role() over absorbed records; 'classify: TEXT' / 'what
kind of text is TEXT' -> classify(); 'what is like TEXT' -> recall(). Anything it
cannot map FALLS THROUGH to generation, LABELLED as completion ('this is
generation, not an answer'), and if there is no sequence model it returns
'unknown' with examples rather than crashing or fabricating -- the key honesty
property: the system never pretends to answer when it is only continuing text.
WIRED TO THE APP: a /api/unified/answer endpoint and a prominent '2 - ask
(question router)' panel at the top, with Enter-to-ask and per-kind rendering
(is_a shows Yes/No + chain + throughput; define shows meaning + is_a; role/
classify/recall show their result; completion is shown as labelled continuation).
So the user types a question and gets an answer from knowledge when the shape is
recognized, and an honest 'I'm just continuing text' when it is not. Six tests
(router maps each form, honest on unmappable, role on records; endpoint routes,
panel present). The unknown-input edge (no sequence model) was found and guarded.
Concepts: HONESTY (the central one -- distinguishing ANSWERING from COMPLETING,
and never faking the former with the latter), INTEGRATION (the router is in the
brain and the app, over the same operations everything else uses), and
SELF-DISCOVERY (the question's own shape selects the operation). The router does
not add knowledge -- it routes to the knowledge already there, which is exactly
why it is honest about what it cannot answer.

REAL PHOTOGRAPHS -- THE IMAGE STACK ON CONTINUOUS-TONE DATA (the user found a
wallpaper repo, dharmx/walls, organized by category, and wanted real rasterized
images to test -- everything image so far had been GIF sprites). Pulled the
'mountain' category (47 real photos, up to 6000px, ~12k colours each -- the
opposite regime to 88-colour sprites), downsampled to a uniform working set, kept
four small 96x96 samples in-repo (features/photo_sample) so the tests are
self-contained. WHAT THE PHOTOS TAUGHT, measured and HONEST: (1) JPEG BEATS OUR
DCT CODER on efficiency and we say so -- HolographicImage keeps top-K GLOBAL DCT
coefficients in a fixed-size plate while JPEG uses 8x8 block DCT with
entropy-coded coefficients; JPEG ~31dB at ~2KB where the plate needs ~5KB. The
plate is NOT a competitive photo codec and is not claimed to be. (2) THE HONEST
WIN -- the plate is ROBUST where JPEG is BRITTLE: erase a random 50% of the
holographic coefficients and PSNR does not move (28.7dB at 0% and at 50%), every
coefficient carrying a little of the whole image, while JPEG loses ~15dB after
10% byte loss and often fails to decode at all. Graceful degradation is the
property the distributed representation buys, and photographs show it as clearly
as sprites did. (3) THE VAULT'S ADAPTIVE CHOOSER GENERALIZES to a new data type:
on sprites it chose shared-palette+LZMA (lossless, low-colour); on 12k-colour
photos that path is correctly UNAVAILABLE and the chooser picks lossy WebP (~90KB
/38dB, beating JPEG here) -- same code, opposite verdict, driven entirely by the
data, validating the 'measure every encoder, keep the smallest, report honestly'
design. (4) THE ORIENTATION PRINCIPLE STAYS IDLE, correctly -- transposing photo
planes costs ~4% (photos have no column-vs-row self-similarity like character
sprites), so the vault keeps row-major; the principle applies only where the data
has directional structure and the measurement says so. Wired as
holographic_photos.py (load_photo_folder, robustness_curve) with four pinned
tests (continuous-tone not palette, graceful degradation, vault picks lossy,
orientation idle). Concepts: HONESTY (the headline -- we are WORSE than JPEG on
efficiency and say so plainly; the real contribution is robustness, a different
axis), COMPRESSION (lossy DCT is the photo tool where palette was the sprite
tool), ROBUSTNESS (the distributed plate's flat degradation under loss, now shown
on real photos), and INTEGRATION (the same vault chooser, validated to make the
opposite-but-correct call on a data type it was not built for). The wallpaper
repo (3.3GB) was pulled, one category extracted, the rest discarded -- only four
tiny samples ship.

FRACTAL STRUCTURE -- leOS's self-similarity detector, ported to real data. leOS's
fractal_detector looked for an abstract pattern recurring across scales of an LLM
displacement log, with the payoff that a self-similar log compresses to a few
Iterated-Function-System rules. holostuff has no such log, but it has data where
self-similarity GENUINELY lives, and the same three tools apply honestly. THE
INSTRUMENT, verified first against known fractals before trusting it: box-counting
dimension recovers Sierpinski 1.59 (true 1.585), filled square 1.95 (true 2.0), a
line 0.98 (true 1.0). WHAT IT FOUND ON REAL DATA, negatives kept: (1) NATURAL vs
SYNTHETIC IMAGES -- natural-photo edge maps have fractal dimension ~1.55-1.6
(rough, scale-invariant, the known statistics of natural scenes) while a smooth
synthetic circle's edge is ~1.0 (a clean 1-D curve); fractal dimension is a real
natural-vs-synthetic signal and a texture/complexity descriptor the smooth-shape
vision work never had. (2) MARKET SELF-AFFINITY -- DAI/WETH minute returns have
Hurst ~0.30 (strongly mean-reverting), INDEPENDENTLY reaching the same verdict as
the market rounds' -0.175 lag-1 autocorrelation and permutation tests, from a
different direction; a control random walk reads ~0.53 as it must. (3) IFS
COMPRESSION WORKS ONLY ON SELF-SIMILAR DATA (the central honest result) -- a
Barnsley fern (8k+ points) regenerates from 4 affine maps = 28 numbers (>500x
compression, coverage error ~0) because it IS the attractor of those maps; random
points fit that same IFS no better than random (error ~0.59 vs ~0.59), so the
compression CORRECTLY fails. Self-similarity is a property of the data, and the
measurement says whether it is present -- the same discipline as the orientation
and motion-compensation rounds (the transform pays exactly where the structure
supports it). WIRED TO THE BRAIN: UnifiedMind.fractal_dimension(x) reads an
image's edge-roughness or a series' self-affinity (as 2-Hurst) directly as a
perceptual quantity, and self_affinity(series) gives the Hurst verdict. WIRED TO
THE APP: a 'fractal dimension' demo in the vision panel tabulates natural-photo
vs synthetic-shape edge dimensions over the real photo samples, showing the
natural-vs-synthetic gap live. Built as holographic_fractal.py
(box_counting_dimension, image_fractal_dimension, edge_mask, hurst_exponent, IFS
with barnsley_fern + generate, ifs_compresses), five module tests + two app
tests. Concepts: FRACTALS/SELF-SIMILARITY (named in the project's own comments as
something leOS did that we hadn't ported -- now ported and measured), COMPRESSION
(IFS as the extreme compression that pays off iff the data is self-similar, with
the negative kept), SELF-DISCOVERY (the structure's dimension read from the data
itself), HONESTY (instrument validated on known fractals first; the IFS negative
on random data kept as the boundary), and CONVERGENCE (the Hurst verdict on the
market data agrees with the autocorrelation finding reached a year of rounds
earlier -- two lenses, one truth).

CONTEXT-CONDITIONED GENERATION -- the honest answer to 'why isn't the brain an
LLM, and how close can deeper conditioning get?' (the user's framing: conversation
feels like looping to find a response that fits the query's shape -- and they are
right that the LOOP exists; they had already built it three times as the router,
the resonator's settle-iterate, and coarse_to_fine's escalation). The explanation
made concrete and TESTED: the missing piece is not the loop, it is a high-capacity
LEARNED P(next | context). An LLM's next-token distribution is conditioned on the
whole prior context through a function fit by gradient descent over trillions of
tokens into billions of parameters; holostuff's generator is a shallow n-gram
P(next | last few tokens). To check whether merely DEEPENING the conditioning on
this substrate helps, built ContextGenerator (holographic_generation.py): generate
at the WORD level, where a word n-gram gives local fluency, and re-rank its
candidates by alignment between each candidate's learned MEANING vector and a
running TOPIC vector (a decaying bundle of generated content words, seeded from
the prompt), with a tunable topic_weight (0 = bare n-gram baseline). THE MEASURED
RESULT, a KEPT NEGATIVE and the whole point: topic-pull re-ranking does NOT buy
genuine coherence. At word n-gram order 2, ~85% of contexts have exactly ONE
continuation, so the topic term has nothing to choose among -- the curve is flat.
At order 1 (mean ~4 candidates) it has room, but then the honest failure shows:
moderate topic_weight slightly LOWERS coherence, and only extreme weight raises
the coherence NUMBER while lexical diversity COLLAPSES (0.78 -> 0.09) into
degenerate repetition ('government operation of government operation of...'),
which even keeps transition_validity high because the repeated bigram was seen in
training. The high-weight 'coherence' is the metric being gamed by a topic vector
collapsing onto a few high-frequency words, not real on-topic language -- which is
why a third metric, lexical diversity, is reported to expose exactly that. WHY
THIS IS THE EXPECTED, INSTRUCTIVE OUTCOME: you can only re-rank structure already
present in the candidates; a shallow proposer offers none, so no amount of
topic-pull conjures coherence it never had. The ceiling is set by the proposer,
confirming by measurement the argument for why the brain is not an LLM. Three
honest metrics in the API (topic_coherence, transition_validity, diversity) and a
sweep() that runs the curve. WIRED: UnifiedMind.learn_word_generator / generate_words
/ topic_pull_tradeoff (the char-level generate() is untouched and still the
default); app endpoint /api/unified/topic_pull lazily trains a word generator from
the loaded prose and a '4 1/4 - topic pull' panel renders the coherence/validity/
diversity table plus baseline-vs-heavy-pull samples, with the lesson stated in the
UI. Six module tests (hermetic synthetic two-topic corpus) + two app tests. The
char generator analysis that preceded this: HolographicNGram is character-level
(per-context bundle of next-char atoms, backoff, temperature sampling), so the
topic pull had to operate at the word level where meaning vectors live -- resolved
during the build, as planned. Concepts: HONESTY (the headline is a negative,
reported plainly, with a diversity metric added specifically to stop the coherence
number from lying), MEASUREMENT (the same sweep-and-keep-the-curve discipline as
the curriculum's reading-washes-out result), and the throughline that the project
is a from-scratch readable VSA engine whose generation ceiling is the n-gram
proposer -- deepening the conditioning is too weak a lever, and the experiment
shows precisely why.

FOUNTAIN (RATELESS ERASURE) CODES -- the last clean idea from the leOS deep dive,
and a SECOND robustness axis complementing the holographic plate. leOS had a
synesthetic fountain; this is Luby's LT code, built from scratch. k source blocks
become an UNLIMITED stream of droplets, each the XOR of a random subset (subset
size from the Robust Soliton distribution); a receiver collecting ANY k(1+eps)
droplets -- whichever survived, in any order -- recovers ALL k blocks EXACTLY by
PEELING (a degree-1 droplet reveals its block, XOR it out of every droplet
containing it, repeat). ON-THEME, TWO WAYS: (1) a droplet (XOR of a random subset)
is the binary sibling of a BUNDLE (superposition of a random subset) -- the same
'combine a random handful' move the VSA core uses, over GF(2); (2) peeling decode
IS the 'loop until resolved' pattern that recurs throughout the project (resonator
settle, coarse_to_fine escalation, the router) -- which is exactly the looping the
user raised when asking why the brain isn't an LLM, here applied to exact erasure
recovery. THE TWO ROBUSTNESS AXES, MEASURED, AND WHEN EACH IS RIGHT: the
HOLOGRAPHIC PLATE is analog and graceful -- erase a random 50% of one stored
representation and PSNR barely moves (28.7dB at 0% and 50%), failure mode 'part of
one representation corrupted', survived LOSSILY (quality decays, nothing exact).
The FOUNTAIN CODE is digital and exact -- failure mode 'whole packets lost in
transit/storage', survived LOSSLESSLY: lose 20-40% of a provisioned droplet stream
and the blob returns bit-for-bit, lose so much that fewer than k droplets survive
and NOTHING returns, because k blocks cannot come from fewer than k droplets (an
information floor, not a flaw). The price of rateless exactness is a ~20% droplet
overhead (mean ~1.20x k for reliable decode at large k). A KEPT CAVEAT: the clean
~1.20x cliff is LARGE-k behaviour; at small k (tens of blocks) the Robust Soliton
guarantees loosen and overhead is higher and more variable (a 1.5x stream can fail
a fraction of the time around k=45) -- LT codes are asymptotic, so size blocks or
overhead up for small payloads (this surfaced as a flaky test and was fixed by
testing at representative k, the failure kept as the lesson). APPLIED to protecting
a real structured blob against whole-block loss (provision the stream for the
expected loss so survivors clear the threshold -- how rateless coding is actually
used). NOT forced into a brain perception method (it is a codec, not a
perception/reasoning op) -- wired honestly where it fits: holographic_fountain.py
(robust_soliton, Fountain with from_bytes/droplets/decode/decode_bytes,
recovery_curve), six module tests, and an /api/fountain endpoint + an 'Erasure
robustness: the other axis' app panel with a channel-loss slider that shows exact
recovery and the recovery-curve cliff live, placed beside the plate's graceful-decay
demo so the two axes sit together. Concepts: ROBUSTNESS (the second, exact-erasure
axis the project lacked), COMPRESSION/REDUNDANCY (rateless coding as the dual of
compression -- spend overhead to buy exactness over a lossy channel), the
LOOP-UNTIL-RESOLVED throughline (peeling as the same settle pattern as the
resonator and coarse_to_fine, the direct sequel to the 'why isn't it an LLM'
thread), BUNDLE<->XOR (the VSA superposition move over GF(2)), and HONESTY (the
information floor reported as a hard cliff, the small-k overhead caveat kept after
a flaky test exposed it). NOTE: with this, the well-aligned leOS ideas are
exhausted -- what remains there is LLM-coupled (needs a second model/displacement
log) or OS/agent infrastructure with no home in a from-scratch NumPy VSA engine.

PREDICTIVE LOOP -- the ACTIVE layer the substrate was missing (the user's goal:
move from a store that retrieves to a system that DOES something with what it
holds, toward query-and-generate; explicitly NOT trying to be an LLM, building a
new thing). Ported the architecture (not the substrate) from a friend's
Closure-SDK, whose upper stack is Rao & Ballard predictive coding on S3
quaternions; rebuilt on holostuff's bind/bundle/permute as holographic_predictive.py
(PredictiveMemory + a zread soft-read primitive). THE LIVING CYCLE, one step at a
time: predict the next symbol from recent context, measure surprise =
1-cosine(predicted,actual), correct error-gated (reinforce if right, nudge or
create an entry if wrong, update size scaled by surprise -- familiar input barely
moves the model, novel input moves it), and report free energy (smoothed running
prediction error) and valence (its signed change). WHAT IS GENUINELY NEW for this
engine: (1) PREDICTION BY RESONANCE not exact match -- each entry is an
order-aware context VECTOR (via permute) paired with a next symbol; prediction
resonates the query context against all stored contexts by cosine, so a context
NEVER SEEN EXACTLY still predicts sensibly when a similar one was seen (verified:
unseen 'my cat' predicts 'sat' at confidence ~0.5 from the cat-contexts; exact
n-gram backoff scores such contexts blind). Confidence (the resonance score)
separates a memorised continuation (~1.0) from a generalisation (~0.5). (2)
SURPRISE as a first-class per-step signal driving error-gated learning -- the
model spends effort where it was wrong, not uniformly (the old learn() wrote
everything equally). (3) FREE-ENERGY CONVERGENCE -- on a periodic stream surprise
falls to ~0 within one period, accuracy reaches 100%, and free energy converges
1.0->0.0 (the model becoming a fixed point of the stream it sees); on non-
repeating prose it honestly does NOT converge, because each context is mostly
novel -- the signal tracks reality, not a target. (4) GENERATION BY ANTICIPATION
-- predict next, append, repeat; the same predictor that learns runs forward to
produce a sequence. MEASURED on Brown news (word-level, open vocabulary, a hard
regime): held-out accuracy RISES with exposure (the learning curve) and it scores
~7% on contexts never seen exactly where exact lookup is at 0% -- the
generalisation is the new capability, kept honestly alongside the modest absolute
numbers (open-vocabulary next-word on sparse news is genuinely hard). A DESIGN
CORRECTION kept as a lesson: the first free-energy definition (distance from a
decaying running state to nearest context) oscillated meaninglessly -- a category
mismatch; redefined as the low-pass of per-step surprise, which is exactly
predictive coding's expected-surprise free energy, and it then converged cleanly
on periodic data. WIRED: UnifiedMind.build_predictor / observe_sequence /
anticipate / generate_predictive / prediction_report; app endpoint
/api/unified/predictive lazily trains from the loaded prose and a '4 1/8 -
predictive loop' panel shows the learning curve, free-energy endpoints, the
unseen-context generalisation score, and a generation-by-anticipation sample.
Seven module tests + two app tests. ALSO ported from Closure-SDK: zread, a
coupling-weighted order-aware soft read (their ZREAD / soft attention) used by the
predictor and exposed as a primitive. NOT ported: the S3 quaternion substrate and
Hopf-fiber channel split (a parallel bet, not a component -- holostuff gets type/
position/role from separate roles and permute already), and the 'geometric
computer / Turing-complete' framing (their research program). Concepts: PREDICTION
(the engine now anticipates, not just stores), SURPRISE/ERROR-GATING (learn where
wrong), GENERALISATION-BY-RESONANCE (predict across similar contexts, the genuine
new power vs exact lookup), the LOOP-UNTIL-RESOLVED throughline (predict-correct
as the active cousin of the resonator settle and coarse_to_fine escalation), and
HONESTY (free energy redefined after a measured failure; modest absolute accuracy
reported plainly with the generalisation win as the real result). This is the
first rung of query-and-generate: a memory that expects its input and changes
when wrong.

MEANING-LEVEL PREDICTION -- generation with structure (the next rung after the
predictive loop). The symbol predictor returns one stored next symbol (right or
wrong, nothing between). This layer (holographic_meaning_predict.py,
MeaningPredictor) changes the OUTPUT: it COMPOSES a next-MEANING vector as the
coupling-weighted ZREAD blend of the next-meanings of every resonating context,
SETTLES it by iterated cleanup in a meaning space (the resonator pattern), and
reads off the nearest word. The prediction is a point composed from many entries,
so it can land where no single entry sits, and -- crucially -- even when the exact
word is wrong it is wrong toward semantically NEAR words. Reported with TWO
metrics: exact-symbol accuracy (precision) and SEMANTIC RANK (percentile of the
actual next word's meaning under the composed prediction; 0.5 chance, 1.0 always
nearest) -- the second is where composition earns its keep over a hard lookup.
THE TEST-DATA LESSON (the user insisted on good data -- dictionary/encyclopedia
curriculum as prior, then a real corpus -- precisely because weak data tells you
nothing; and the data OVERTURNED the obvious guess). Compared two meaning spaces
as the compose/settle space, measured at scale on Brown news AND Reuters, the same
double dissociation both times: (1) CO-OCCURRENCE (syntagmatic) meaning -- a word
is the sum of what appears NEAR it -- predicts the NEXT word well (semantic rank
~0.85, actual next word in the top ~15%); the DICTIONARY-CURRICULUM (paradigmatic)
space is near chance at it (~0.50). (2) The reverse for RELATEDNESS ('what is this
/ what is like X'): the dictionary curriculum separates WordNet-related words at
d-prime ~0.8-0.93 vs ~0.43-0.48 for co-occurrence. THE KEPT PRINCIPLE: next-word
prediction is a DISTRIBUTIONAL task, so the predictor composes/settles in
co-occurrence space; the dictionary/encyclopedia prior is not discarded but is the
right space for the relatedness query -- match the space to the question. (My prior
assumption that the richer dictionary meaning would help prediction was WRONG, and
the good test data is what showed it -- exactly the user's point.) The settle step
is iterated cleanup against the meaning space (start from the composed vector, snap
toward the nearest few meanings, re-blend, converge), the same loop-until-resolved
pattern as the resonator and coarse_to_fine. WIRED: UnifiedMind.build_meaning_predictor
/ anticipate_meaning / meaning_prediction_report; cooccurrence_space and
relatedness_dprime as reusable helpers; set_space to plug in any prior (e.g. a
dictionary space for the relatedness query). Six module tests (compose+settle,
semantic rank beats chance, space construction, match-the-space mechanism,
relatedness d-prime, brain wiring). Concepts: COMPOSITION (the prediction is built
from many entries, not recalled from one -- the move from lookup toward
generation), SETTLE/RESONATE (iterated cleanup to a clean attractor), HONEST
MEASUREMENT (semantic rank credits right-neighbourhood predictions; the
match-the-space dissociation replicated on two corpora; a wrong prior assumption
overturned by good data), and CURRICULUM (the dictionary/encyclopedia prior placed
where it actually helps -- relatedness, not sequence). The path forward this opens:
anticipation now produces a meaning, not just a symbol, so a generated continuation
can be steered in meaning space (toward a topic, an answer shape) while staying
distributionally plausible -- query-and-generate taking shape on the substrate's
own terms.

PROOF OF STRUCTURE -- verifying meaning, not trusting it (the user's insight:
prediction can yield word salad, so it needs proof of meaning/structure, and
meaning is only useful PROJECTED onto context or process). Built
holographic_structure.py (StructureVerifier + steered_generate). THE CORE FINDING,
which proves the user right by measurement: SINGLE-STEP coherence -- cosine(what
the context predicts, the actual word's meaning) -- separates real text (0.41)
from shuffled (0.21) and random (0.08), BUT IS GAMEABLE: text generated greedily
by the predictor itself scores 0.88, HIGHER than real text, because every step was
chosen to maximise exactly that. Local coherence is not proof of meaning. THE
PROOF that works is the LAG-COHERENCE PROFILE: mean similarity between each word
and the word k positions back, k=1..6. Real text has a moderate, EVEN profile;
salad deviates two ways -- shuffled/random sit too LOW (no structure at any range),
degenerate generation sits too HIGH and PERIODIC (an order-2 generator that fell
into a 3-cycle reads ~1.0 at lag 3 and 6). STRUCTURE SCORE = -mean z-distance of a
sequence's profile from a band calibrated on real text (0 = typical, more negative
= anomalous): real held-out ~ -1.2, shuffled ~ -2.2, random ~ -4.1, self-generated
salad ~ -15. Every salad type, INCLUDING the locally-coherent self-generated loop
that single-step coherence rated highly, falls well below real text -- the proof
catches what the naive check missed. 'Meaning projected upon context' is literally
each profile term; a meaning vector alone is inert, its usefulness is whether it
sits in the band real context produces. THE VERIFIER AS A PROCESS (the active
payoff, and NOT a repeat of the topic-pull negative): steered_generate picks, among
the predictor's top candidates, the word that keeps the recent window's structure
score highest. Greedy decoding collapses into a fixed loop ('the city of the city
of the city...', score ~ -15); steered generation escapes it and stays in the
real-text band (~ -0.8). The difference from the failed topic-pull is the lever --
topic-pull re-ranked by a static topic bag and collapsed; steering by TRAJECTORY
structure, projecting each candidate onto the unfolding context, defends coherence
as a process. AN HONEST LIMIT, kept: the verdict rejects random words (0% pass) and
degenerate loops (the failure modes that matter for generation) and passes real
text, but does NOT reliably reject SHUFFLED real text (~90% passes) -- a bag of real
words keeps too much of the lag-profile; catching exact-order corruption would need
a running-composition (the Closure approach), and the profile measures structure-
by-range, not order. For keeping generation out of salad and loops it works; as a
general grammaticality judge it is partial. WIRED: UnifiedMind.verify_structure /
generate_structured (and build_meaning_predictor now also calibrates the verifier);
the app's predictive panel gained a proof-of-structure block showing greedy-vs-
steered scores and samples side by side. Six module tests + one app test. Concepts:
PROOF/VERIFICATION (structure measured, not assumed -- every prediction can be
checked, echoing the Closure 'composition is an integrity check' thesis on this
substrate), MEANING-IN-CONTEXT (coherence is fit-to-context across ranges, not a
word's meaning alone), the LOOP-UNTIL-RESOLVED throughline (steered generation as a
process that defends its own structure), and HONESTY (the gameable single-step
measure reported as a failure that motivated the real one; the shuffle limitation
kept). This answers the user's point directly: prediction now carries a proof of
structure, and meaning is put to work by projection onto the unfolding context.

QUERY-AND-GENERATE -- the synthesis toward the user's stated goal (query the
system and get a response that fits the query's shape; build something with that
functionality, not an LLM). holographic_respond.py (respond, respond_report,
query_target, relevance) ties the predictive layers together. A query implies a
TARGET region in meaning space -- the bundle of its content words' meanings.
Generation runs forward with the meaning predictor, and each step is chosen under
TWO forces: STRUCTURE (keep the running window's lag-coherence profile in the
real-text band -- holographic_structure -- stay coherent, escape loops) and
QUERY-PULL (prefer candidates whose meaning points toward the query target -- stay
on-query). MEASURED (Brown news), the honest operating curve: relevance rises
monotonically with query_weight (0.47 at 0 -> 0.53 at 2 -> 0.60 at 5 -> 0.66 at
10) while structure holds in the band through query_weight ~5 (-1.1 vs unsteered
-0.8) and only degrades at a hard pull (10 -> -2.9). THE LOAD-BEARING FINDING, and
why this is NOT a repeat of the failed topic-pull: the structure guard is what
makes the on-query window exist. At a hard pull (query_weight=8), WITH the guard
structure is ~-2.0; WITHOUT it (struct_weight=0, the exact topic-pull regime)
structure collapses to ~-6.7 for the same relevance. Pulling toward a topic alone
collapses into salad (the earlier negative); pulling toward the query INSIDE the
structure band answers on-query and stays coherent -- the two forces together are
the capability. respond_report returns the response WITH its relevance and
structure, so an answer is never trusted blindly -- both are measured. Note this is
the random-words lesson applied: relevance is reported against what the query
implies, and the structure guard keeps us above the salad floor, but the system is
not claimed to be fluent -- it is coherent, on-query generation whose two
properties are measured. WIRED: UnifiedMind.respond / respond_report; app endpoint
/api/unified/respond and a '4 3/8 - query & generate' panel with a query box that
shows the steered response (relevance, structure) beside the unsteered baseline so
the query-pull is visible. Five module tests + two app tests. Concepts:
QUERY->GENERATE (the actual goal -- ask, get a structured on-topic continuation
back, built entirely from the substrate's predict/compose/verify machinery), DUAL
STEERING (two forces, structure as the guard that makes topic-pull safe -- the
direct redemption of the earlier topic-pull negative now that a structure proof
exists), MEANING-AS-PROCESS (the query target is meaning projected onto the
generation process, the thing the user said meaning needs), and HONEST MEASUREMENT
(relevance and structure both reported, the operating window and the guard's
necessity both measured). The arc is now whole: store -> anticipate (predictive
loop) -> compose+settle (meaning prediction) -> prove structure (verifier) ->
answer a query inside that structure (this). Housekeeping: the recovery zip was
rebuilt at the start of this round to capture the prior structure round, which had
shipped late.

DELIBERATION -- think before you speak (the user's idea: a human doesn't emit the
first thing thought; they form an abstract thought, piece it together, assign
language, let it surface as inner speech, iterate, and only then speak -- fast
sometimes, slow other times depending on context). holographic_deliberate.py
(Deliberator) builds this from the parts already present. The loop: GIST (the
query's meaning target -- the abstract anchor of what to say) -> DRAFT (realize the
gist into words with the meaning predictor + structure guard -- the inner-speech
surfacing) -> JUDGE (quality = relevance on-gist + struct_weight * structure score)
-> ITERATE (a first greedy draft, then stochastic diverse drafts; keep the best;
STOP EARLY once quality clears target_quality). The iteration count IS the thinking
time and it ADAPTS: easy queries settle in 1-2 passes, hard ones run the full
budget (measured range 1-8 across Brown queries) -- that adaptivity is the
'sometimes fast, sometimes slow'. MEASURED: deliberated best-of-adaptive-N quality
~0.40-0.43 vs single greedy pass ~0.34-0.36 (the loop helps). TWO KEPT NEGATIVES
that shaped the design (the user's 'start with abstract thoughts, piece them
together' suggested elaborating a plan before language; measurement said keep the
plan simple): (1) rolling the meaning predictor FORWARD into a meaning trajectory
as a plan drifts into function words ('in','the','of') because the predictor is
syntagmatic -- it predicts what FOLLOWS, not a semantic arc -- and realizing to it
HURT relevance (0.47 vs 0.57 flat); (2) enriching the gist with its meaning-
NEIGHBOURS was neutral (0.554 vs 0.566). CONCLUSION, kept honestly: on this
substrate the query's own meaning target is already the right abstract anchor; the
human-like gain is in the iterate-and-keep-best LOOP, not in elaborating the plan,
so the deliberator keeps the gist simple and spends effort on the loop. The trace
(every draft + its score, the kept one marked) is returned so the inner
deliberation is VISIBLE rather than hidden -- the 'internal thought surfacing'. The
first draft is deterministic (greedy); diversity for the later drafts comes from
temperature sampling among the predictor's candidates. WIRED: UnifiedMind.deliberate
(lazily builds a Deliberator over the meaning predictor + verifier); app endpoint
/api/unified/deliberate and a '4 1/2 - deliberate' panel with a query box that shows
the thinking time, the kept response, and the full draft trace. Five module tests +
two app tests. Concepts: DELIBERATION/THINK-BEFORE-SPEAK (draft-judge-iterate, the
loop-until-resolved throughline applied to language production), ADAPTIVE EFFORT
(thinking time scales with difficulty -- the contextual fast/slow the user
described), INNER SPEECH MADE VISIBLE (the draft trace), and HONEST MEASUREMENT (the
loop's gain measured; two plan-elaboration ideas tried and kept as negatives, so the
design follows the data not the intuition). This composes with query-and-generate:
the deliberator wraps the steered responder in a judge-and-revise loop, so the
system now pauses to choose its words and only says the best draft it found.
MULTI-JUDGE NEGOTIATION -- competing pressures resolving before something surfaces
(the agreed extension of deliberation). holographic_deliberate.Deliberator gained
judges() + negotiate(): instead of one quality number, several judges score each
draft -- COHERENCE (structure score mapped through the verifier's threshold),
RELEVANCE (on-query cosine), and NOVELTY (type-token ratio, which falls when a draft
loops/repeats). The negotiated score is the MINIMUM across the normalized judges --
the binding pressure -- so the kept draft is the most BALANCED, not one that wins a
single axis while failing another; the loop stops early once the negotiated score
clears a target. The judges genuinely pull against each other (coherence likes
common, sometimes repetitive text; novelty penalizes repetition). MEASURED: with the
structure guard already suppressing most loops, the novelty judge is mostly a SAFETY
NET -- it matches the single-quality loop on repetition (type-token ~0.92) in the
typical case and rescues the occasional repetitive draft (0.85 -> 0.96 on the one
query that needed it). The per-judge trace (every draft's coherence/relevance/novelty
breakdown, the kept one marked) makes the tension visible. WIRED:
UnifiedMind.negotiate; four module tests. Backward compatible -- the single-quality
deliberate() is untouched. Concepts: NEGOTIATION (min-of-judges = balance the
weakest pressure), COMPETING-PRESSURES (coherence vs novelty as a real tension), and
HONEST MEASUREMENT (novelty reported as a safety net, not oversold).

CROSS-DOMAIN STRUCTURE -- does the recent machinery help numbers/images? (the user's
question). Investigated the most general piece, the structure verifier, honestly.
The text verifier scores a sequence by how closely its LAG-COHERENCE PROFILE matches
a real-text band; the same idea is the lag-AUTOCORRELATION profile for a continuous
series and the SPATIAL autocorrelation for an image. holographic_signal_structure.py
(SignalStructureVerifier + lag_autocorr_profile + spatial_coherence_profile +
volatility_clustering + clustering_zscore). VERDICT PER DOMAIN, measured and kept:
(1) IMAGES -- TRANSFERS CLEANLY: spatial coherence is stable like prose, so a natural
patch scores ~ -0.6 against a real-patch band while white noise and pixel-shuffled
versions crash to ~ -14 -- the same clean separation text gives. (2) MARKET/RETURNS
-- TRANSFERS ONLY WITH THE RIGHT SIGNATURE AND ENOUGH DATA: raw returns are nearly
uncorrelated, so a symmetric profile-deviation does NOT separate real from shuffled
(a flat shuffle can even score HIGHER, because intermittent volatility structure
widens the band -- a kept negative). The structure that DOES distinguish real returns
is volatility CLUSTERING (|returns| positively autocorrelated); the honest statistic
is directional -- lag-1 autocorr of |returns| vs a shuffled control. On a long
synthetic GARCH series this is unmistakable (z~9); on the short real DAI/WETH sample
(~100 returns) it is present but only ~1 sigma -- too little data to call, the regime
where honest measurement says 'not enough signal yet'. THE GENERAL LESSON, kept: the
verifier idea (compare a sample's autocorrelation signature to a band of real data)
is genuinely cross-domain, but the SIGNATURE must match the domain's actual structure
-- stable spatial coherence for images, intermittent volatility clustering for
returns, an even lag-profile for text. The machinery transfers; choosing what to
autocorrelate is the domain knowledge. Five module tests. Concepts: GENERALITY (one
proof-of-structure idea across text/series/image), SIGNATURE-MATCHING (the transfer
is not automatic -- it failed for returns under the wrong statistic, kept honestly),
and the throughline that real signals of every kind carry an autocorrelation
fingerprint that noise and shuffles lack -- the same insight as the text verifier,
now domain-general. (Note: complements the existing fractal Hurst/box-counting
structure measures -- a second, autocorrelation-based structure signature.)

COMPRESSION -- better structure means better compression, made literal (the user's
sweep principle, and the highest-leverage low-hanging fruit). A predictor IS a
compressor: if you can anticipate the next symbol you spend fewer bits recording
which one came. holographic_compress.py (PredictiveCompressor +
structure_compression_correlation) rank-codes a sequence under the meaning
predictor -- at each position rank the vocabulary by the settled next-meaning, and
the actual symbol's RANK is its cost (log2(rank+2)); encoder and decoder run the
same predictor over the symbols so far, so ranks are reproducible (an idealized
rank-coder measuring the information content of the ranks, not a byte-level
arithmetic coder). MEASURED on Brown news against a uniform baseline of
log2(vocabulary) ~11.8 bits/symbol: real text ~7.0 bits/sym (ratio 0.59), shuffled
real words ~8.9 (0.75), random ~10.4 (0.88) -- MORE STRUCTURE, FEWER BITS, exactly
the claim. The STRUCTURE SCORE PREDICTS COMPRESSIBILITY: across windows from real to
fully shuffled, structure score vs compression ratio correlates ~-0.6 (higher
structure -> lower ratio). And it is NOT just word frequency: a unigram
frequency-only model costs ~9.5 bits/sym on the same text; the predictor's ~7.0
beats it by exploiting ORDER and context -- the structure a frequency table cannot
see. This sits beside the fractal IFS compressor already in the stack (compresses a
self-similar fern ~500x but not random data): two kinds of structure (temporal/
predictive here, spatial/self-similar there), two kinds of compression, one
principle -- structure is what makes a thing shorter to describe. WIRED:
UnifiedMind.compress_cost / structure_compresses; the app's predictive proof block
now shows real-vs-shuffled compression ratios beside the structure scores. Five
module tests. Concepts: PREDICTION=COMPRESSION (the deep identity, now a hard
bit-metric that validates the predictor), STRUCTURE->COMPRESSION (quantified
correlation), and the throughline tying the predictive loop, the structure verifier,
and the fractal compressor into one statement.

WIRING SWEEP -- made sure recent capability isn't orphaned (the user's 'make sure
everything is wired up'). The cross-domain signal-structure verifier (last round's
holographic_signal_structure) was standalone; now wired into the brain:
UnifiedMind.verify_image_structure (spatial-autocorrelation signature of a real
image vs noise/corruption; calibrates on supplied real patches or the image's own
patches) and UnifiedMind.volatility_structure (the volatility-clustering z-score for
a return series). Two brain-wiring tests added. So every recent layer -- predictive
loop, meaning prediction, structure verifier, query-and-generate, deliberation,
multi-judge negotiation, cross-domain signal structure, and now compression -- is
reachable from UnifiedMind and (where it has a demo) surfaced in the app. SWEEP
NOTE on the other themes the user listed: recursion/fractals/self-similarity and
demoscene (generate-big-from-small) are the fractal/IFS compressor, now unified with
predictive compression under structure->compression; decomposition/projection
('shadow of a complex shape') is the resolution coarse-to-fine and meaning-space
projection; sequence/order/time is the predictive loop; composability/integration is
bind/bundle + ZREAD; self-assembly/self-discovery is the slime mold and resolution
stabilisation; layers/inception is the hierarchy. The compression build is the one
that turned a stated principle into a measured, wired capability with a hard metric.

SELF-DISCOVERY OF STRUCTURE -- find the units with no labels (the sweep continued;
hits self-discovery, decomposition, sequence/order/time, and better-structure->
better-compression at once). holographic_segment.py (Segmenter + boundary_f1 +
chunk_compression). THE IDEA (Harris 1955, Saffran 1996): inside a unit the next
symbol is tightly constrained, at a unit's end many symbols can follow, so
uncertainty about the next symbol PEAKS at boundaries. Strip the spaces from text
and the word boundaries are still recoverable from this signal alone. ON THIS
SUBSTRATE: for each EXACT context (last K symbols) accumulate a bundle of the symbol
atoms that followed; project that bundle onto every symbol atom (the next-symbol
readout) and take its ENTROPY -- high entropy = many possible successors = boundary.
Boundaries are the local entropy peaks above a percentile. MEASURED (Brown news,
spaces removed): branching-entropy boundaries hit F1 ~0.6 against the true word
boundaries vs ~0.2 for a random cut at the same rate -- words genuinely
self-discovered from an unsegmented stream (sample chunks: 'county','grand','jury',
'said','friday'). A KEPT NEGATIVE that chose the method: doing the readout via
RESONANCE (blending over SIMILAR contexts -- the ZREAD that HELPS prediction)
DESTROYS the signal (F1 fell to ~0.26, near random) because it smears the
next-symbol distribution across neighbours; boundary discovery needs the EXACT
context's successor diversity, not a generalised one -- generalisation and
segmentation want opposite things. THE COMPRESSION PAYOFF (better structure ->
better compression, again, now reached by DISCOVERING the units): coding the stream
as the discovered chunks costs ~2.1 bits/char (unigram over chunks) vs ~4.2
bits/char over single characters -- finding the right decomposition roughly halves
the description length. WIRED: UnifiedMind.discover_units; app endpoint
/api/unified/discover and a '4 3/4 - self-discovery' panel that strips the corpus's
spaces, shows the recovered boundaries (F1 vs a random cut), the discovered units,
and the chunk-vs-char compression. Six module tests + two app tests. Concepts:
SELF-DISCOVERY (units found, not given), DECOMPOSITION (segment a raw stream into
its own parts -- which COMPOSE upward: a chunk can become a symbol for a higher
layer, and COMPRESS downward: units are where the code resets), PREDICTION-AS-
STRUCTURE-SENSOR (the predictor's uncertainty is the boundary signal), and the
throughline that structure, compression, and decomposition are three views of one
thing -- the predictable regularities that make a stream shorter to describe and
splittable into reusable parts. This is the natural feeder for a recursive/layered
('inception') next step: segment, treat chunks as symbols, predict/compress at the
chunk layer, repeat.

FACTORIZATION BY SEARCHING IN SUPERPOSITION -- the inverse of binding, the deep/
novel capability found by looking outward (searched the VSA literature; this is the
Resonator Network of Frady, Kent, Olshausen & Sommer, Neural Computation 2020, and
the "computing in superposition" property the surveys call the distinguishing power
of VSA). holographic_resonator.py (ResonatorNetwork + map_codebook + map_bind).
THE PROBLEM: binding combines several vectors into one composite; the hard inverse
is, given only the composite and the codebooks of possible parts, recover which part
came from each codebook. Brute force is the PRODUCT of codebook sizes (three of 100
= a million; it explodes with more factors). THE SOLUTION: a resonator searches in
SUPERPOSITION -- each factor's estimate is a weighted blend of ALL its codebook's
vectors at once; hold the others fixed, unbind them from the composite to estimate
the remaining factor, CLEAN UP toward its codebook (similarity to every codevector,
superpose them back by those similarities, take sign), update all factors
simultaneously, iterate until re-binding the nearest codevectors reproduces the
composite. The true factors "resonate out" of the mixture while the rest cancel.
Resonators aren't guaranteed to converge but a converged run is always correct, so
RANDOM RESTARTS make it a reliable solver. MEASURED on this substrate: 3 codebooks
of 50 (125,000 combinations) solved 20/20 with ~2 median restarts; 3 of 100
(1,000,000 combinations) ~11/20 at dim 3000 -- the classic dimension-vs-capacity
tradeoff (capacity ~quadratic in dimension, per the papers). A KEPT NEGATIVE that
shaped the build: with the engine's native CIRCULAR-CONVOLUTION binding the
resonator does NOT converge (0-1/20) -- unbinding by involution amplifies the
cross-term noise each step. Factorization needs a SELF-INVERSE, noise-stable bind,
so this module uses MAP (bipolar, elementwise-product) binding internally and is
explicit it is a different algebra from the rest of the engine. THE LESSON: the
operation you can invert in superposition depends on the algebra you bind with --
circular convolution is great for storage/cleanup but MAP is what factors. WIRED:
UnifiedMind.factor_composite; app endpoint /api/unified/factorize and a '4 7/8 -
factorize' panel that binds three random vectors and recovers them, showing the
combinatorial space searched-not-enumerated. Six module tests + two app tests.
Concepts: DECOMPOSITION (pull a single bound representation apart into the
independent factors that composed it -- distinct from holographic_segment, which
cuts a stream; this factors a vector), COMPUTING-IN-SUPERPOSITION (the distinguishing
power of VSA: entertain all solutions at once and let the right one resonate out),
RAYTRACING/PROJECTION (combinatorial search by repeated projection onto codebooks
and settling -- the loop-until-resolved shape shared with cleanup, coarse_to_fine,
the meaning settler and the predictive correct step, now solving an exponential
search), and SELF-ASSEMBLY (the factors assemble themselves out of an uninformative
superposition). This is the most powerful single primitive added: the true inverse
of composition, which makes the whole bind/compose side of the engine reversible.

GOING BOTH DIRECTIONS: LOSSLESS CODEC + SOURCE ATTRIBUTION (the user asked: can we
now trace the source of information used in generation, and is there a deterministic
compress-to-a-seed/decompress-back?). holographic_codec.py (PredictiveCodec +
SourceAttributor). THE CODEC -- a predictor IS a compressor, and now reversibly: the
predictor ranks the vocabulary at each step, each token is encoded by its RANK, and
because the DECODER runs the IDENTICAL predictor over the tokens decoded so far it
reproduces the identical ranking and recovers the exact token. The round-trip is
EXACTLY lossless (verified token-for-token). The compressed object is the seed (first
`order` tokens) + the rank stream; the model is shared like a codebook. HONEST ANSWER
TO 'COMPRESS TO A SEED, DECOMPRESS BACK': it is real and exactly lossless, but its
size is bounded by the data's STRUCTURE, not by wishful thinking. Measured (Brown):
real text rank-stream entropy ~7.3 bits/token vs ~11.6 uniform baseline (ratio ~0.63,
lossless); RANDOM tokens barely move (~0.74) -- no free lunch, and there cannot be (no
method losslessly shrinks all inputs -- a counting argument; the seed can never be
smaller than the data's true information content); and a PERFECTLY PERIODIC stream
rank-codes to ~0 bits/token (every token is the top prediction, so the sequence
collapses to the seed alone). That last case is the demoscene/fractal 'seed' dream made
literal, and it is the SAME statement as the IFS fern compressing ~500x while random
data will not: compression is the search for the shortest GENERATOR, and the predictor
is that generator for a sequence. So the bidirectional machinery did not repeal
information theory -- it gave a clean, lossless way to spend exactly as few bits as the
structure allows. A coding note kept: Elias-gamma on the ranks was WORSE than baseline
(ratio 1.1) because the meaning predictor's exact-word rank is often large (great
neighbourhood, weak exact); the achievable size is the rank-stream ENTROPY (what an
arithmetic coder reaches), which is what is reported. SOURCE ATTRIBUTION -- the thing
that was hard before, now tractable because resonance couplings are exposed: tag each
stored (context->next) entry with its SOURCE; for a token in context, the predictor's
resonance gives a coupling to every stored context, and the token's provenance is the
sources of the highest-coupling entries that ALSO predict the realized token; aggregate
over a passage -> a provenance distribution. MEASURED on a two-source corpus (news vs
romance): a held-out news passage attributes ~0.74 to news, a romance passage ~0.58 to
romance -- a real, majority-correct signal, imperfect because distinct sources share
common language (an honest ceiling). WIRED: UnifiedMind.compress_lossless /
decompress_lossless / attribute_sources; app endpoint /api/unified/codec and a
'5 - lossless codec' panel showing the exact round-trip and the structure-vs-random
ratios. Six module tests + two app tests. Concepts: BIDIRECTIONALITY (compress<->
decompress as exact inverses, the same go-both-ways theme as bind<->factor), MDL/
KOLMOGOROV (the seed bounded by information content; structure compresses, noise does
not -- demonstrated, not asserted), and PROVENANCE (generation/prediction traced back
to the stored material it resonated with). Together with the resonator this completes
a reversibility sweep: the engine can now both COMPOSE and FACTOR (bind<->resonator),
and both PREDICT-FORWARD and COMPRESS-AND-RESTORE (generate<->codec), with attribution
showing WHERE the information came from.

UI OVERHAUL: SEARCHABLE / CATEGORIZED / COLLAPSIBLE EXAMPLE CARDS (the user: the
demo panels were boring/confusing to find, and scrolling everywhere is annoying;
make each example a tagged, categorized, searchable card that expands on click).
unified_app.py page only -- no Python logic, core, or module changes. WHAT CHANGED:
(1) a sticky TOOLBAR at the top of <main> with a search box and category pills
(All / Setup / Memory / Predict / Generate / Structure / Compress / Reason) plus a
live result count. (2) every example panel is now a COLLAPSIBLE CARD: a clickable
head (twirl + clean title + one-line description + tag chips) and a body that is
hidden until the card is opened -- click to expand and use the example. (3) SEARCH
filters cards live by title/tags/category/description and auto-expands matches and
highlights the query; CATEGORY pills filter by group; a "no examples match" hint
shows when a query is empty. IMPLEMENTATION (deliberately low-risk): a JS CATALOG
maps a distinctive substring of each card's <h2> to {category, tags, description},
and buildCards() post-processes the DOM at load -- it never rewrites the panels
themselves, so all existing controls, ids, and onclick wiring are untouched (the
panel functions still find their $("pout") etc. because only DOM nodes are moved,
not renamed). FIXED ALONG THE WAY: an orphaned source-tracing/attribution block
from an earlier round was floating OUTSIDE any card (malformed, with stray closing
divs) -- wrapped it into a proper '5 1/4 - source tracing' card with an h2 so it is
now tagged and searchable like the rest. Every one of the 15 cards maps to exactly
one catalog entry (verified by test). Two app tests added: the card system is
present (search box, pills, buildCards, filterCards, CATALOG) and every card has a
catalog entry (unique h2->category mapping). Suite 435. Concepts: NAVIGABILITY
(find an example by searching or by category instead of scrolling), PROGRESSIVE
DISCLOSURE (cards collapsed by default, expand to use -- less overwhelming), and
keeping the change SURGICAL (runtime DOM transform over a catalog, so no panel
behaviour or test wiring broke). Note: this was presentation only; the underlying
demos and their measured claims are unchanged.

MANY MINDS, ONE SUBSTRATE: SHARED FROZEN BASE + PER-INSTANCE DELTAS (the user is
building a game with many NPCs and foresaw a scaling wall -- every NPC building/
training/updating its own brain. Their instinct, correct and well-known: branch from
a shared parent and propagate learning back, OR freeze a base and add a lightweight
per-instance layer that propagates back -- the SAME idea, copy-on-write deltas /
frozen-base-plus-adapter / structural sharing). holographic_partition.py (SharedMind
+ MindInstance + share). THE STRUCTURE: train ONE base mind on common knowledge and
FREEZE it; every NPC is an Instance that (1) shares the base BY REFERENCE -- including
its ENCODER, so all instances perceive into the same vector space -- (2) holds only
its own small DELTA of prototypes (what it personally learned), (3) reads by scoring
over base+delta together (inherits common knowledge, adds private on top), (4) writes
only into its delta so the base stays shareable. WHY THIS SUBSTRATE MAKES IT EASY:
because instances share the same atoms, a learned vector means the same thing
everywhere, so knowledge is COMPARABLE and ADDITIVE across instances -- two free
consequences: MERGE = SUPERPOSITION (propagating an NPC's learning into the base, or
pooling many NPCs, is just bundling their delta prototypes -- a federated average in
VSA terms; existing-label merge bundles+renormalises, new-label appends), and
ISOLATION IS FREE (an NPC reads base+own-delta only, never another's private
knowledge until it is explicitly propagated). MEASURED: with base B prototypes and N
NPCs each ~d private, the population costs B+N*d vs N*B for separate minds -- 50 NPCs
over a small base showed ~4-5x in the demo, growing with base size (50 over a
1,000-prototype base with 20 private each ~= 2,000 vs ~50,000, ~25x). Verified four
properties: INHERITANCE (a branch classifies shared items it never learned),
ISOLATION (two branches don't see each other's private deltas), PROPAGATION (after
propagate/merge a private fact becomes classifiable by every instance, present and
future), and MERGE-BY-SUPERPOSITION preserves recall (reinforcing an existing label
keeps it and the base intact). WIRED: UnifiedMind.share() -> SharedMind; SharedMind
.branch(name) -> MindInstance with .learn/.classify/.recall/.knows_privately/
.propagate; SharedMind.merge(instances) and .population_cost(instances). App endpoint
/api/unified/population and a '6 - many NPCs, one mind' panel (new 'Scale' card
category) showing inheritance, isolation, propagation, and the saving for a chosen
population size. Eight module tests + two app tests. Concepts: STRUCTURAL SHARING /
COPY-ON-WRITE (the heavy base shared by reference, only diffs per instance),
FROZEN-BASE+ADAPTER (= the same thing, the user's two framings unified), FEDERATED
MERGE = BUNDLING (superposition makes propagation trivial and principled), and the
key enabler SHARED ATOMS (one encoder across all instances is what makes vectors
comparable/mergeable). HONEST LIMIT to keep in mind for the game: superposition merge
has capacity limits -- bundling very many distinct deltas into one base label will
eventually degrade recall (the classic VSA capacity cliff); for now each NPC's
private facts are few and distinct so it is comfortably within capacity, and a future
rung could add capacity-aware merge or per-instance episodic layers. This is the
scaling answer: NPCs are partitioned instances of one mind, not N separate minds.

EXPOSING THE REAL DATASETS IN THE UI: SPRITES + IMAGE REPOSITORY (the user: make sure
the UI surfaces examples that use the sprites, the wallpaper/image repository, the
dictionary+encyclopedia curriculum, Reuters, books, and other sizeable datasets).
unified_app.py page + two endpoints only -- no core/module changes. WHAT WAS ALREADY
THERE (dataset picker, now relabelled for clarity): world records, Dictionary +
encyclopedia (curriculum), this project's own source, Reuters categories, Brown
genres, Books (Gutenberg authors), Europarl languages. WHAT WAS MISSING and is now
exposed as demo cards under a new 'Images' category: (1) SPRITE PACK
(/api/unified/sprites) -- loads the real 712-sprite set (features/sprites) and packs it
as ONE body against shared references via pack_sprites.pack (the proven, bit-exact
path the tests use), comparing to per-file optimized PNG: measured ~7.6x smaller (200
sprites: ~18.9 KB vs ~143.5 KB) and bit-exact. KEPT NOTE: holographic_pack.benchmark
chokes on the real set (mixed RGBA/grayscale shapes -> stack broadcast error), so the
endpoint uses pack_sprites.pack which handles the mixed set; honest baseline is
per-file PNG ("each sprite alone"). (2) IMAGE VAULT (/api/unified/vault) -- ingests the
photo repository (features/photo_sample, falling back to sprites), groups
near-duplicates into perceptual clusters by a size/format-invariant 16x16 fingerprint,
runs query-by-example (an image matches itself at sim 1.0), and reports honest
size/fidelity per encoder via ImageVault.report. Both are self-contained (no corpus
load needed). WIRING: two new cards ('7 - sprite pack', '8 - image vault') added to the
JS CATALOG with the new 'Images' category and tags; books/Gutenberg label clarified to
'Books (Gutenberg authors)'. Card system still maps every one of 18 cards uniquely
(verified). Three app tests added (sprite pack smaller+lossless; vault clusters+self-
query; both panels present), gated with pytest.skip when the asset folders are absent
so CI stays green without the binary assets. Concepts: ONE SPACE FOR PICTURES TOO (the
image repository lives in the same perceptual-memory shape the engine uses for words),
CROSS-ITEM STRUCTURE = COMPRESSION (a sprite set is one structured body, not N files --
the same better-structure->better-compression principle as the predictive codec and
the IFS fern, now on game assets), and DATA COVERAGE (the sizeable datasets the project
actually has -- sprites, images, dictionary+encyclopedia, Reuters, Brown, books -- are
all reachable from the UI). Presentation/coverage only; no measured claims changed.

ON-DEMAND EXPERIMENT PANELS FOR THE HEAVY DATASETS (the user's workflow directive:
don't add long-running tests that run the big data every suite invocation; instead
wire the heavy experiments into the UI as buttons the user runs and pastes/screenshots
results from -- image data, market data, large text). unified_app.py page + three
endpoints; tests kept LIGHT (small-slice or skip-gated, never the full heavy run).
WHAT WAS WIRED: (1) MARKET STRUCTURE (/api/unified/market, '9 - market structure' card,
new 'Market' category) -- the market data (data/sol_5min.npz ~1,500 within-burst moves;
data/dai_weth_big.json ~1,000 candle returns) was fully coded (holographic_market,
holographic_signal_structure) but NEVER exposed in the UI. The card tests for
VOLATILITY CLUSTERING: |returns| lag-1 autocorrelation vs a shuffle of the same
returns. MEASURED NOW (the bigger data changed the verdict): SOL ticks z=+12 to +15
(acf1 ~0.41), DAI/WETH big z=+4 (acf1 ~0.12) -- both real structure (>2 sigma), where
the OLD ~100-return DAI/WETH sample was only ~1 sigma (kept lesson: too little data,
not absence of structure); shuffle controls collapse to ~0; raw signed returns stay
efficient-market-like. (2) BIG-TEXT RUN (/api/unified/bigtext, '10 - big-text run'
card, new 'Experiment' category) -- a heavy run on a large slice of the loaded corpus,
reporting structure score (real vs shuffled vs random), lossless codec ratio, and
self-discovered word-boundary F1 all at once, with a tokens knob (500-20,000). This is
exactly the kind of run kept OUT of the suite. (3) the image vault/sprite cards from
last round remain (the photo sample is small; the vault can also run over the full
712-sprite set as images). WIRING: three cards added to the JS CATALOG with two new
categories (Market, Experiment); every one of 20 cards still maps uniquely (verified).
THREE light tests added: market returns a sane shape with shuffle-acf ~0 (skip if data
absent), bigtext on a SMALL 500-token slice round-trips lossless, both panels present.
WORKFLOW PRINCIPLE recorded: heavy/large-dataset experiments live in the UI as
on-demand panels (run -> copy/paste/screenshot results back) so the test suite stays
fast and hermetic; automated tests use tiny inputs or skip-gates. Concepts: SEPARATION
OF EXPERIMENT FROM REGRESSION (interactive heavy runs vs fast deterministic tests),
DATA COVERAGE (market data now reachable alongside images and text), and the kept
market lesson that STRUCTURE NEEDS ENOUGH DATA TO SEE (the same signature went from
~1 sigma to >4-12 sigma purely with more samples).

WIRING AUDIT + SHARED/CACHED TRAINING + MIT LICENSE (the user: thoroughly check every
UI example is wired correctly; let each experiment train on relevant material;
experiments that share training should reuse it instead of retraining unless a fresh
brain is needed; add an MIT license). unified_app.py + LICENSE + README.
WIRING AUDIT (programmatic, kept as the method): cross-checked every onclick/oninput
handler against defined JS functions, every JS fetch/post target against the Flask
routes, and every card against having a working control -- ALL 20 cards have a button
and a live endpoint, no handler lacks a function, no fetch lacks a route, no route is
orphaned. ~14 experiments need a trained mind; 6 are self-contained (sprites, image
vault, factorize, population, market, and the datasets list). TRAINING CACHE: added
TRAINED (dataset_id -> (state_snapshot, build_result)) and build_cached(dataset_id,
fresh=False). The load endpoint routes through it: first load of a dataset trains and
caches the FULL STATE (the mind plus everything lazily built on it -- meaning
predictor, verifier, codec), later loads restore that snapshot instantly (measured:
cold ~1.3s, warm ~0.001s, same mind object so derived structures are shared), and
fresh=True retrains a new mind. Because every experiment reads STATE["mind"], sharing
a dataset shares ALL trained/derived data for free; switching datasets and back is
instant. Added a 'fresh' checkbox to the pull+train panel (with a description that
training is cached and shared) and a /api/unified/trained status endpoint (active +
cached dataset names). RELEVANT-MATERIAL GUIDANCE: the prose experiments' empty-state
messages now name the datasets to train (Reuters / Brown / Books), and the curriculum
one names the curriculum dataset -- so each experiment points you at the right
training. MIT LICENSE: added a standard MIT LICENSE file (copyright
AnOversizedMooseWithSocks, the repo owner) and a License section in the README. Three
light app tests added (cache reuse + fresh retrain via the fast 'world' dataset; the
trained-status endpoint; an empty-state message names a dataset), made order-
independent (the cache persists across tests in one process, so the cache test forces
a fresh baseline first). Concepts: TRAIN ONCE, SHARE EVERYWHERE (one trained mind per
dataset, reused across experiments; the same structural-sharing instinct as the NPC
partition round, now applied to the app's own experiments), EXPLICIT FRESHNESS (opt
in to retraining when an experiment needs a clean brain), and DISCOVERABILITY (each
experiment names its relevant training). No measured claims changed; this is
plumbing, guidance, and licensing.

CUMULATIVE / STACKED TRAINING + VISIBLE TRAINING PROVENANCE (the user: train on
several datasets in sequence -- e.g. dictionary+encyclopedia first to lay a base
structure, then books, then Reuters or country info -- and SEE what the current brain
has been trained on, to judge whether a fresh brain is needed). unified_app.py only.
WHAT CHANGED: the training cache (added last round, then keyed by a single dataset id)
is now keyed by the FULL ORDERED STACK -- a tuple of dataset ids. build_stack(ids)
trains one mind on the datasets in order, reusing cached PREFIXES: building
(curriculum, gutenberg, reuters) caches (curriculum,), (curriculum, gutenberg), and
the full triple, so extending a stack reuses the work already done. build() was
refactored into _absorb_into(mind, id) (and build_curriculum into
_absorb_curriculum_into(mind)) which train a dataset INTO an existing mind --
absorb()/learn_sequence() already ADD to the one memory, so calling them again layers
the new dataset on top rather than replacing. STATE now carries trained_ids and
trained_on (ordered display names); each _absorb merges labels, appends to the stack,
refreshes seq_raw to the newly added prose, and drops any stale _meaning_pred/
_verifier/_codec so prose experiments rebuild against the fuller brain. STACK DIM:
stacked minds use a uniform dim=2048 (curriculum's dim, enough capacity for the
dictionary), so every layer shares one atom space -- required for the layers to be in
the same vector space. SAFETY (deep-copy isolation, measured ~0.075s small / ~1.6s a
big 2048-d corpus-trained mind -- fine for a deliberate action): cache snapshots are
deep copies, and "add on top" deep-copies the current mind before absorbing, so a
cached prefix is NEVER corrupted when a longer stack builds on it (verified: the pure
curriculum base keeps its prototype count after world is added on top). UI: the
pull+train panel now has two buttons -- "Start fresh" (mode=replace, new base) and
"Add on top" (mode=add, layer onto the current brain) -- a "rebuild (ignore cache)"
checkbox, and a persistent readout "current brain trained on: 1. X -> 2. Y -> 3. Z"
(renderStack), refreshed on page load from /api/unified/trained (which now reports
trained_on, prototypes, labels, and cached_stacks). MEASURED demo: curriculum base
(19 prototypes) + Countries on top (29 prototypes) -- the count grows because the
second layer adds, confirming stacking. Four light app tests (cache reuse/rebuild;
status reports the stack; cumulative stacking accumulates prototypes; add-on-top does
not corrupt the cached base) -- all fast (small datasets), none long-running. Concepts:
CURRICULUM/LAYERED TRAINING (lay a base structure first, then specialise -- the same
recursion/layers theme, now in how the brain is taught), TRAIN-ONCE-REUSE across
stacks and prefixes, COPY-ON-WRITE SAFETY (deep-copy so shared prefixes stay valid),
and PROVENANCE (the brain shows its training history so the user can decide on a fresh
brain). The single-dataset behaviour is preserved via build_cached -> build_stack((id,)).

CAPACITY-AWARE LAYERING + DELIVERY TOWN BRAIN FEEDBACK (the user: capacity-aware
layering is extremely important; also feedback from a game (Delivery Town) using the
holo brain for NPC navigation, listing brain-side change requests -- some game-specific
and skipped, the rest applicable). holographic_creature.py (HolographicMind) +
holographic_partition.py.
CAPACITY-AWARE LAYERING (the priority, and the root of the cliff flagged in the NPC
and stacked-training rounds): a prototype is a SUPERPOSITION (bundle) of its members,
and a bundle has FINITE capacity -- fold too many DISTINCT members into one and the
unit can no longer resemble any of them; mean member-cosine collapses ~1/sqrt(count)
(MEASURED on D=512: one bundle's fidelity falls 0.35->0.06 as members go 8->256, while
capping at 16/bundle holds ~0.25 at the cost of more prototypes -- the cliff and its
fix, both numbers). FIX: a `capacity` cap (HolographicMind, default 0 = off/unbounded
folding = today's behaviour). In _absorb, when the nearest prototype is merge-close but
already holds `capacity` members, it does NOT blur it further -- it starts a fresh
SUB-prototype for the same situation. value()/classify already score over all
prototypes for an action/label, so sub-prototypes are read transparently. New
capacity_report() diagnostic: prototypes, max_count, mean_count, and 'overloaded'
(count past the sqrt(dim) soft cap). The SAME guard added to the shared-base NPC merge
(SharedMind.absorb_delta + share(capacity=)): when many instances propagate learning
for one label, capacity>0 splits into sub-prototypes instead of one over-loaded bundle
(MEASURED: 30 propagations into 'weapon' -> 1 blurred prototype unbounded vs 6 capped,
recall preserved). This directly answers the capacity cliff warned about for deep
stacks and many-NPC merges. 'split, don't blur' is the same move the scaling RP-tree
uses for storage, now on the value memory and the federated merge.
DELIVERY TOWN FEEDBACK (brain-side changes that apply here; game-side Change 1
[orthogonal compass] and Change 6 [their brain_compat.py contract] are theirs, not
ours -- but every change here is ADDITIVE with a behaviour-preserving default so their
compat gate passes and an older game still runs on the newer brain):
  * Change 2 -- TIERED veto in decide(): new soft=() kwarg splits the veto into HARD
    (permanent: wall) vs SOFT (temporary: red/traffic) blocks. When every direction is
    blocked, instead of lifting to ALL actions (a blind pick among permanent walls),
    lift only to the SOFT-blocked ones -- wait on a temporary block (often the goal
    direction) rather than guess. soft=() reproduces today's behaviour exactly.
    MEASURED: boxed by walls + traffic_N (goal_N) -> chooses N (wait), not a wall.
  * Change 3 (minimal) -- penalize_recent(amount, n): an online 'stuck' signal. When
    an external watchdog detects a loop it nudges DOWN the value of the last n
    (state,action) pairs in the recent buffer, so the loop teaches itself instead of
    only being rescued. Needs maintain=True/'auto' (the buffer); safe no-op otherwise.
    MEASURED: a repeated move's value 1.0 -> 0.5.
  * Change 4 -- blind_floor (attr, default 0.0 = off): when the brain has no memory for
    any allowed action (max support < blind_floor) and a goal_<dir> token is in senses,
    FOLLOW THE COMPASS rather than guess. MEASURED: blind + goal_E -> chooses E.
  * Change 5 (merge tuning) -- noted as measure-first; capacity= is the sharper lever
    for the same granularity concern, so addressed structurally rather than by sweep.
  * Change 3 fuller (TD bootstrapping) -- deferred as a research spike, as the doc
    itself recommends (changes brain behaviour everywhere; measure on the maze bench).
Seven HolographicMind tests + two partition tests, all fast. Concepts: SPLIT-DON'T-BLUR
(finite bundle capacity met by sub-prototypes, on both the policy memory and the
federated merge -- the recursion/layers theme applied to capacity), and ADDITIVE-WITH-
DEFAULTS (every change off or identity by default, so existing benches/demos/the game's
compat gate are byte-for-byte unchanged). HONEST: capacity= trades memory (more
prototypes) for fidelity; the right cap depends on dim and how distinct the members
are -- a tuning knob with a measurable diagnostic (capacity_report), not a free win.

DESIGN STANCE CORRECTION + MEASURED CAPACITY DEFAULT (the user: don't preserve
backward compatibility for downstream consumers like the Delivery Town game -- holostuff
is the CORE; others build on top and adapt to our changes. So core decisions stand on
their own merits and measurements, never on "so a downstream game's compat gate passes"
or "so an older game still runs"). Applied: scrubbed the game-compat framing from the
HolographicMind code and tests -- soft= (tiered hard/soft veto), penalize_recent
(online stuck-signal), and blind_floor (blind-state compass fallback) are now documented
as general core capabilities on their own reasoning (wait on a temporary block rather
than guess; let a loop teach itself before episode end since learning is Monte-Carlo;
replace a coin flip with the compass when the value memory is empty here), not as
"feedback Change N" kept additive for a downstream gate. They remain off/identity by
DEFAULT not for compatibility but because each is opt-in by nature: soft= needs the
caller to name which sense prefixes are temporary (the core cannot know game vocabulary),
and blind_floor needs goal_<dir> tokens the core does not itself emit.
THE CAPACITY DEFAULT, DECIDED BY MEASUREMENT (not by compat, and not by assumption):
A/B on a harsh fixed obstacle world where one egocentric situation recurs constantly,
4 seeds, 200 episodes: capacity=0 (unbounded folding) blurred the value memory into a
near-degenerate ~1.0 star/life; a cap rescued performance (~4.2 at cap=sqrt(dim)=16,
~2.6 at cap=64) -- BUT a tight cap fragments memory hard (~108 -> ~8700 prototypes) and
is high-variance (+/-2 stars). The gauntlet, meanwhile, passes with capacity=0, so
unbounded is not degenerate everywhere. CONCLUSION: capacity-aware layering genuinely
matters, but the right cap trades fidelity against memory and is task-dependent, so
defaulting to a single guessed cap would itself violate measure-first. capacity=0
stays the default as a MEASURED decision (no cap unless you tune one), with
capacity_report() exposing prototype load so the cap can be set by data per task. The
capacity-aware MERGE in the shared base (share(capacity=)) has no return-denoising
tension -- it is pure recall fidelity -- so a cap there is an unambiguous good when many
instances propagate into one label. Net: the features are core-merit features; the one
real default question (capacity) was answered with numbers, and the answer is "it
depends, here is the diagnostic to decide," which is the honest core stance.

GEOMETRY PROPOSAL EVALUATED: WS1 RUNG 0 (UNITARY HRR) -- ADOPTED PER-SUBSYSTEM, NOT
GLOBALLY (an external design note proposed exact-unbinding geometries: unitary HRR ->
FHRR torus -> qFHRR (WS1), hyperbolic embedding (WS2, measure-first), rotor/GA binding
(WS3, study). Evaluated WS1 Rung 0, the cheap one, on the existing benchmarks with
negatives kept). THE CLAIM IS TRUE AT THE BINDING LEVEL: our unbind is bind(c,
involution(a)), and the involution is the EXACT circular-convolution inverse only when
an atom's FFT magnitudes are all 1. Gaussian atoms have a spread spectrum, so a single
unbind recovers its target at only cosine ~0.71; a UNITARY atom (mint a Gaussian, divide
its FFT by |.| so every component is unit-magnitude, ifft back -- stays real, 1x storage,
bind/unbind byte-for-byte unchanged) makes that single unbind EXACT (1.0). Added
unitary_vector() to holographic_ai.py and a unitary= flag on Vocabulary.
BUT MEASURED AGAINST THE BENCHMARKS, THE WIN IS NARROW AND THE BLAST RADIUS IS REAL:
  * Key->value capacity (the doc's headline target): NO GAIN -- at many pairs the error
    is cross-term crosstalk BETWEEN pairs, which unitary atoms don't reduce. Wash with
    Gaussian (8-48 pairs).
  * Resonator/scene: nothing to fix -- it already uses MAP binding (bipolar, exact
    self-inverse), not circular convolution.
  * Few-factor role-binding (records, relations, sequence roles): a GENUINE win --
    exact unbind widens the cleanup margin and lifts accuracy under stress (16
    role/filler pairs at dim 256: 0.971 -> 0.982; margins wider across 3-8 roles).
  * MEASURED NEGATIVES (kept on record): unitary atoms HURT every mechanism that reads
    the SPREAD of a bundle as signal. (a) sequentiality_z momentum detection on a
    2-symbol alphabet: real SOL tick signs (+0.20 lag-1 autocorr) read z=+44 Gaussian
    vs z=-4.77 unitary (WRONG). (b) branching-entropy segmentation: word-boundary F1
    fell below the random-cut baseline. (c) the creature's permute+bundle working
    memory: the starved-maze bootstrap-rescue gauntlet went from cracking the maze to
    rescued=0.0 under unitary.
DECISION: do NOT flip the global Vocabulary default to unitary (too broad, real
regressions). Per the doc's own "adopt per subsystem where it wins" guidance: Vocabulary
defaults to GAUSSIAN; unitary is opt-in. ADOPTED unitary in KnowledgeStore (so relations
+ encyclopedia, pure role-binding, get the cleaner margin). KEPT Gaussian everywhere the
bundle-spread mechanisms live. Two robustness fixes fell out: (1) the encyclopedia/
unified climb() depth-decay used to come from per-hop unbind NOISE; with cleaner
role-unbinding it had to be made an EXPLICIT hop_discount (0.9) -- a deliberate "a longer
derivation is less certain" signal, not an artifact. (2) sequentiality_z now mints its
OWN Gaussian atoms deterministically per symbol, so its verdict no longer depends on the
caller's vocab or call order (this also fixed a test-ordering fragility). WS1 Rung 1
(full complex FHRR, 2x storage) NOT built: Rung 0's narrow win doesn't yet justify it,
and the doc gates Rung 1 behind "if Rung 0 helps but the resonator wants more" -- the
resonator can't use it. WS2/WS3 not started (measure-first / study, as the doc itself
ordered). Three pin tests added (unitary makes single unbind exact; Vocabulary mint
flag; unitary widens role-filler margin). Concept: the torus isn't a better shape, it's
our binding with the approximation removed -- and that removal is a NET GOOD only on
clean role-unbinding, a net BAD on bundle-spread readouts; measurement drew the line,
not theory.

RECURRENT-LAYER PROPOSAL EVALUATED: WS-R RESERVOIR -- BUILT, MEASURED ON REAL DATA, KEPT
AS A NEGATIVE (an external note proposed a recurrent layer: gradient-free Echo State
Network/reservoir first (WS-R), numpy LSTM only if the reservoir falls short (WS-L,
crosses the no-gradient line), associative-LSTM advanced (WS-A)). Built WS-R in
holographic_recurrent.py: EchoStateNetwork (dense random reservoir, leaky-tanh, ONE
ridge-regression readout -- no backprop, no epochs), VSAReservoir (the native one:
permute is the fixed recurrent operator, bind folds in input, tanh the one nonlinearity,
renormalise -- the engine's own kit + one tanh), ReservoirCharModel (next-char), and
ReservoirSequenceClassifier (read the FINAL state, prototype + cosine cleanup). The
reservoir adds the nonlinearity the linear permute+bundle recurrence lacks, gradient-free.
MEASURED ON REAL CORPORA (per the user's directive: no toy data except as a control):
  * Generation, next-char on Gutenberg's Alice (the doc's named ~62% baseline): n-gram
    0.583 vs ESN 0.42 vs VSA reservoir 0.30 -- N-GRAM WINS.
  * Language ID on real UDHR (6 languages): bag-of-trigrams 0.97-0.99 vs reservoir
    final-state 0.33-0.36 -- BAG WINS DECISIVELY.
  * Genre on real Brown (6 categories): bag-of-words 0.30 vs reservoir 0.28 -- BAG WINS.
  * CONTROL (order-only, same character multiset in opposite order -- the one place a
    bag is structurally blind): reservoir 1.00 vs bag 0.50. The MECHANISM works; real
    tasks just don't reward it, because real classes separate on symbol statistics that
    n-grams and bags capture directly and a FIXED random projection captures less
    sharply. The abcd/dcba task is labelled a CONTROL, not a result.
VERDICT: WS-R does not beat the existing baselines on this repo's real corpora, so it is
NOT adopted as a default (no model= switch added to the core generation path -- adding a
losing option to UnifiedMind.generate would be surface area for nothing). It is kept as a
clean, self-contained module + demo + tests (6, the real-data A/Bs NLTK-gated) and
documented as a measured negative, exactly the repo convention ("the reservoir didn't
beat the n-gram on X is a result, not wasted work"). WS-L (LSTM) correctly NOT built: the
doc gates it behind "only if the reservoir falls short in a way that justifies crossing
the gradient line" -- the reservoir didn't fall short in a way more gradients would fix
(the gap is the fixed random feature extractor vs explicit n-gram/bag counts, not a lack
of learned gates), and building backprop-through-time to lose by less is not worth
crossing the no-gradient line. WS-A deferred. Concept: a reservoir is the TEMPORAL
blessing-of-dimensionality (random recurrent operator spreads a sequence into a near-
orthogonal trajectory) -- elegant and gradient-free, but on these corpora the explicit
count-based baselines are simply stronger, and measurement says so plainly.

BRAIN BUG FIX (penalize_recent IndexError) + G6 PROPERTY TESTS (a bug report:
penalize_recent throws IndexError: index N out of bounds when called after
auto_maintain/reorganize resized a per-action prototype array; timing-sensitive, repro'd
in a 15-day game with maintain='auto'. Plus a "close the gaps" design note: G1 variance
harness, G2 ablation table, G3 frozen core+persistence, G4 adversarial gauntlet, G5
perception ceiling, G6 algebra property tests).
THE BUG, ROOT-CAUSED DEEPER THAN REPORTED: penalize_recent computes j from _unit[a] then
indexes _cnt[a][j]/_ret[a][j]. The reporter guessed lockstep drift in reorganize, but
reorganize keeps the four per-action banks (_unit/_sum/_ret/_cnt) in lockstep. The real
fault: (1) after auto_maintain SWAPS in a new memory (or a basis change re-dimensions the
banks), self._buf still holds (state, action) entries recorded against the OLD banks/old
width, so a buffered state may not map onto the new banks at all; and (2) a separate, more
serious latent crash -- auto_maintain's candidate rebuild (_rebuilt_from -> _blank) created
candidates at the ORIGINAL self.dim, so after consolidate() projected the brain to a
smaller subspace, _absorb vstacked a projected (narrower) state onto a full-width empty
bank and raised ValueError outright. _blank also silently dropped the capacity setting.
FIXES (root cause, not band-aid): _blank() now builds candidates at the brain's CURRENT
working dimension (_state_dim() = basis rank if consolidated else dim) and inherits the
basis, and carries capacity=self.capacity; auto_maintain clears self._buf after a swap
(the buffer is meaningless against the new banks and self-heals as new experience
arrives); penalize_recent defends itself -- skips a buffered entry whose action index is
stale, whose state width no longer matches the bank, or whose argmax j would index past a
since-resized array. Verified: 3000 steps of maintain='auto' with regime shifts +
consolidation + constant penalize_recent, no exception, four banks stay in lockstep.
G6 ADOPTED (the highest-value, cheapest gap, and the one this very bug motivates): new
test_algebra_properties.py asserts INVARIANTS over many random draws, bounding the WORST
case, not a demo outcome -- the silent numerical/structural class demo tests miss. Nine
properties: bind/unbind round-trip band (Gaussian mean ~0.71, min >0.5 -- a kept measured
fact: the involution is approximate on Gaussian atoms), unitary atoms make it exact
(min >0.999), bind hides its operands (leakage <0.2), permute-inverse is exact identity
(<1e-12), permute decorrelates, bundle stays similar to members within the 1/sqrt(count)
capacity band, cleanup is exact under bounded noise, the Walsh-Hadamard key operator is an
exact isometry (norm preserved + exact clean round-trip, <1e-9), and -- closing the loop
on the bug -- the brain's four prototype banks stay in lockstep through a long maintain
+consolidate stream (a property test of STRUCTURE). G1-G5 NOT built this round (each
substantial; G1 variance harness is a large retrofit of every headline number, G2 ablation
a system-wide study, G3 a core extraction + persistence layer): evaluated and left
sequenced as the doc orders, with G6 done first because it is cheap, parallel, and
directly guards the exact regression class just reported. Concept: point the "measured,
not promised" discipline at the algebra ITSELF -- invariants with worst-case bounds catch
the degradations that leave every demo green.

G1 VARIANCE HARNESS BUILT + LOAD-BEARING CLAIMS RETROFITTED ON REAL DATA (the credibility
gap: most headline numbers were single-seed point estimates on a system built from RANDOM
vectors, so a figure could be a lucky seed). holographic_measure.py: measure(run_once,
seeds) runs a scored experiment once per seed and returns mean, sample std, and a 95%
PERCENTILE-BOOTSTRAP CI (no distributional assumptions); assert_robust(stats, floor)
passes only if the LOWER CI bound clears the floor (a lucky-seed mean is not enough);
is_fragile flags a claim whose std is >= half its margin above the floor; report()
formats "mean +/- std (95% CI [lo,hi], n)". Used REAL corpora throughout (per the
standing directive -- no toy data except as a control): MEASURED, each across seeds:
  * next-char accuracy, Gutenberg Alice 6-gram (the ~62% headline): 0.611 +/- 0.001
    (CI [0.610, 0.612]) -- essentially seed-proof, a useful credibility result.
  * language ID, UDHR 6 languages: 0.990 +/- 0.007 (CI [0.984, 0.994]) -- solid.
  * word-boundary F1, spaceless Brown news (the ~0.61 headline): 0.602 +/- 0.010
    (CI [0.596, 0.609]) -- solid.
  * topic classification, Reuters 5-category: 0.819 +/- 0.044 (CI [0.793, 0.847]) -- the
    one with a REAL spread (a single seed could read 0.77-0.87), so it is reported with
    its interval and its test asserts the lower CI clears a conservative 0.72, not a
    lucky 0.87.
  * resonator 3x50 (125k space) and key->value peel recall (40@512): both 1.000 +/- 0.000
    -- rock-solid across seeds.
The load-bearing tests now assert_robust on the LOWER CI bound (test_variance.py: 8 tests
-- 3 hermetic for the harness itself, including a test that a mean clearing a floor still
FAILS when its lower CI does not; 5 real-corpus claims, NLTK-gated). README gained a
"Variance and credibility" table; tour prints the live Alice spread; holographic_measure.py
--main prints the full real-corpus variance table. CONCEPT: point the "measured, not
promised" discipline at the numbers THEMSELVES -- a claim isn't its mean, it's a
distribution, and the honest report is the spread plus a lower-bound test. Finding: the
core text claims are not seed-fragile (tight spreads), and the one with genuine variance
(Reuters classify) is now reported honestly with its band rather than as a point estimate.
Of the "close the gaps" note, G1 and G6 are now done; G2 (ablation table) is the natural
next -- it needs G1's noise band to judge "within the noise", which now exists.

G2 ABLATION TABLE BUILT -- WHERE IS VSA ACTUALLY LOAD-BEARING? (the highest-insight gap;
unblocked by G1, which provides the noise band needed to judge "within the noise"). For
each subsystem, run the DUMBEST honest non-holographic baseline on the SAME real data and
metric, measure both across seeds via the variance harness, and let the 95% CIs decide:
holo lower CI > baseline upper CI => "VSA load-bearing"; baseline lower > holo upper =>
"baseline wins"; overlap => "uniformity". holographic_ablate.py (verdict(), one holo+
baseline pair per subsystem on real data, ablation_table(), _demo() prints it) +
ABLATIONS.md (the committed deliverable) + test_ablations.py (6 tests, verdicts pinned,
NLTK-gated). MEASURED VERDICTS (real corpora):
  * topic classify (Reuters 5-cat): holo 0.830+/-0.047 vs bag-of-words 0.611+/-0.062 =>
    VSA LOAD-BEARING (a real ~0.22 win -- the holographic encoding folds co-occurrence
    structure in where a sparse bag centroid can't generalise).
  * key->value, NOISY keys (noise=0.5): VSA 0.889+/-0.069 vs exact dict 0.000+/-0.000 =>
    VSA LOAD-BEARING (the sharpest: a hash either matches or doesn't, so a perturbed key
    scores a flat 0; cosine cleanup recovers the value from an approximate cue -- this is
    the mechanism the whole engine leans on). Note: on EXACT keys a dict is trivially 1.0;
    VSA's win is specifically the approximate-cue case.
  * language ID (UDHR 6-lang): holo 0.990+/-0.008 vs bag-of-trigrams 0.994+/-0.004 =>
    UNIFORMITY (the trigram IDEA tells languages apart, not the VSA encoding; the bag even
    edges ahead).
  * segmentation (Brown spaceless): holo entropy 0.604+/-0.011 vs EXACT count-based
    branching entropy 0.612+/-0.000 => NOT LOAD-BEARING (the entropy idea finds
    boundaries; the exact estimator marginally edges the holographic one, as an exact
    method should -- razor-thin, lands between uniformity and a hair's-breadth baseline
    win; the test accepts either non-load-bearing verdict).
  * recall index (forest, 2000 items dim128): HoloForest 0.817+/-0.028 @ 41% of
    comparisons vs exact scan 1.000+/-0.000 @ 100% => BASELINE WINS on raw recall, but the
    honest reading is the forest's win is SCALE (sublinear, ~41% comparisons), not
    accuracy -- the row attaches comparison_fraction and is read "decorative for accuracy,
    load-bearing for cost".
THROUGHLINE (now in README + ABLATIONS.md): VSA is load-bearing exactly where the problem
is APPROXIMATE or COMPOSITIONAL (recover from a corrupted cue, fold structure into a
representation) and decorative where an exact countable statistic already settles the task.
A sharper self-description than "does text and memory", and it says where the next unit of
effort belongs: the damage-tolerant / approximate-cue corners, not the ones a Counter wins.
Of the "close the gaps" note: G1 (variance), G2 (ablation), G6 (algebra properties) now
done. Remaining: G3 (frozen core + persistence -- gates build-on-top), G4 (adversarial/
worst-case gauntlet), G5 (perception ceiling). G3 is the natural next: it gates building
anything else on the kernel and adds save/load for trained minds.

GENERATION LOW-HANGING BUNDLE BUILT -- PROCEDURAL OUTPUT IN FOUR MODALITIES (the
highest-value roadmap's Tier 1, and the third of its "if you do only three things" after
G1+G2 and unitary HRR; the on-ramp to the generative work). holographic_generate.py drives
decoders the engine already has, no learned distribution, no gradients; each generator
beats the dumbest honest baseline on a measurable metric:
  * IMAGE/VIDEO morph: slerp between two stored images IN THE DCT-COEFFICIENT DOMAIN
    (interpolate coefficient direction + magnitude), inverse-transform per frame. Honest
    win is ANTI-GHOSTING, measured precisely: a pixel crossfade midpoint IS the
    double-exposure 0.5a+0.5b (distance 0 -- both pictures visible at once); the
    coefficient-domain morph blends STRUCTURE so its midpoint sits ~0.06 away from that
    double image. (Measured the wrong metric first -- frame-evenness -- where a crossfade
    trivially wins by construction; the RIGHT metric is ghosting, and slerp wins it
    clearly. Kept that as the lesson.) morph_video() threads keyframes (de-dupes shared
    endpoints: 3 transitions x 8 = 7+7+8 = 22 frames).
  * TEXT: nucleus (top-p) decoding + optional repetition penalty over the existing
    holographic n-gram distribution. Honest win is COHERENCE: real-word fraction
    0.79 (plain temperature) -> ~1.00 (nucleus p=0.85), because trimming the unlikely tail
    removes the garbage chars that break words -- at a modest diversity cost (distinct-4gram
    0.87 -> 0.77). That is the real top-p tradeoff, reported not hidden. The repetition
    penalty defaults OFF: measured, this n-gram does not loop (repeat-rate ~0), so it is a
    knob not a fix -- kept that negative honestly rather than claiming a repetition win.
  * AUDIO: sonify a symbolic sequence to a real 16-bit WAV (stdlib wave) -- each distinct
    symbol -> a fixed chromatic pitch, deterministic and repeatable, short sine tones with
    a click-avoiding fade. This is faithful RENDERING, not a learned synthesiser; the
    honest claim (asserted) is distinct symbols -> distinguishable repeatable pitches, NOT
    "sounds good".
test_generate.py: 8 tests (image/audio hermetic, text coherence A/B NLTK-gated on real
Alice). README "Procedural generation" section; tour "procedural:" line (distinct from the
pre-existing "generation:" line, which documents WHY the heavier learned-P(next|context)
path is still needed -- the two are complementary: procedural ships now, learned generation
is the next tier). CONCEPT: generation here is driving STORED STRUCTURE forward (morph,
re-rank, render), not learning a distribution -- cheap, honest, shippable, and the on-ramp
to native generation (run the resonator FORWARD to compose new attribute scenes rather than
interpolate stored ones). Of the roadmap: Tier 0 (G1+G2) done, Tier 1 (unitary HRR +
generation bundle + G6 property tests) now ALL done. Next per the ranking: Tier 2 -- G3
(frozen core + persistence, the gate) then the reservoir (already built, kept as a negative
on generation but it is the 4-way-leverage component), then forward scene render.

G3 FROZEN CORE + PERSISTENCE BUILT -- THE GATE FOR BUILD-ON-TOP (the roadmap's Tier 2
first item; "nothing in build-on-top is safe to layer until the core stops shifting", and
a trained mind could not be saved at all). Done as a NON-BREAKING EXTRACTION per the flat-
module preference -- a new facade module + purely additive methods on existing classes, no
edits to any existing logic.
  * holographic_core.py: re-exports the kernel primitives (random_vector, unitary_vector,
    bind, unbind, involution, bundle, permute, cosine, slerp, Vocabulary) + a free
    cleanup() -- the STABLE surface build-on-top code imports. It is an EXTRACTION not a
    rewrite: the functions still live in holographic_ai.py, so existing files are
    untouched; new layers import the frozen surface instead of a subsystem's internals.
    CORE_VERSION + STATE_VERSION stamps.
  * Versioned save/load: save(obj,path)/load(path) round-trip any object exposing
    to_state()/from_state() through a SINGLE npz (a _flatten that stores arrays + lists-of-
    arrays + a tiny JSON meta sidecar, allow_pickle=False). Stamped with STATE_VERSION;
    an incompatible or UNSTAMPED state fails LOUDLY on load (refuses rather than returning
    a silently-wrong object) -- tested.
  * Persistence added to the three LEAF objects that hold the core learned state:
    - Vocabulary: dim/unitary + every name->vector (exact round-trip).
    - HolographicMind: the four per-action banks (_sum/_unit/_ret/_cnt) + the consolidation
      _basis + config; reloads deciding IDENTICALLY, including the hard case of a
      CONSOLIDATED brain (low-rank basis present), verified across many probes; banks stay
      in lockstep. The recent-experience buffer + transient EMAs are deliberately NOT saved
      (self-healing scratch; saving them would only reintroduce the stale-buffer hazard the
      maintenance code guards against). Captured self._seed in the ctor for this (additive).
    - HoloForest: trees are a pure function of (seed, tree idx, node id), so we save only
      items + config and REBUILD deterministically -- a saved-then-loaded forest recalls
      identically. Captured _seed/_leaf_size/_n_trees in the ctor (additive).
  * UnifiedMind persistence DEFERRED ON PURPOSE: it is a deep composite (encoder,
    SelfOrganizingMind, lazy sub-objects, journals, format gates) whose faithful round-trip
    is out of proportion to this round; shipping a fragile half-version would be worse than
    not shipping it. Kept as an honest deferral rather than a half-done claim. The three
    leaf objects are what the reservoir / generation / forward-render layers actually need
    to persist.
  * Verified the core is a SUFFICIENT build surface: every primitive the reservoir and the
    generation bundle use is re-exported by holographic_core (the document's "recurrent
    layer builds against the kernel only" criterion).
test_core_persistence.py: 9 tests (kernel re-export identity, cleanup parity, exact
Vocabulary round-trip, trained-brain + CONSOLIDATED-brain identical decisions, forest
identical recall, version-mismatch + unstamped-state + no-to_state() all raise). README
"Frozen core + persistence" section; tour "core:" line (saves a consolidated brain to npz,
reloads, shows identical decisions). VALIDATED the three edited modules (holographic_ai,
holographic_creature, holographic_tree) broke nothing: full brain/tree/navigator/slime
suites + the creature gauntlet INCLUDING the rescue_cracks canary all pass. Roadmap: Tier 0
(G1+G2) + Tier 1 (unitary HRR, generation bundle, G6) + Tier 2's G3 now done. Next per the
ranking: the reservoir is already built (kept as a negative on generation but it is the
4-way-leverage sequencer), so the live Tier 2 item is FORWARD COMPOSITIONAL SCENE RENDER --
run the resonator forward to compose NEW attribute scenes, render via the decomposition
renderer, vary over time for procedural video. That is the real native-generation step the
generative direction is aiming at, and it now has a buildable, persistable core under it.

FORWARD COMPOSITIONAL GENERATION BUILT -- NATIVE GENERATION (the roadmap's live Tier 2
item and the real payoff of the generative direction: step up from the morph bundle
[interpolate what's STORED] to composing what was NEVER stored). holographic_compose.py
runs the EXISTING decomposition machinery FORWARD: the scene path normally goes backward
(image -> auto_tags -> SceneCoder.factor_scene resonates the scene vector into its
colour/shape/texture atoms); forward it is pick tags -> encode_scene (bind + UNNORMALISED
superpose) -> a scene vector -> make_scene renders it to a real RGB image. No new model, no
gradients -- existing structure driven forward, built on the frozen core + the SceneCoder/
make_scene that already existed.
THE HONEST BAR (this is the whole point): a generated scene is meaningful only if it can be
ANALYSED STRAIGHT BACK to the spec it was built from, so every generator is measured by
ROUND-TRIP, and the combinations are drawn NOVEL (excluded from a "seen" set) so a correct
round-trip proves COMPOSITION not recall. MEASURED:
  * novel single-object compose->factor: 40/40 exact; the ENTIRE composable space
    (7 colours x 4 shapes x 4 textures = 112) round-trips >=97%.
  * novel multi-object scenes (explain-away peel): 2-obj 30/30, 3-obj 30/30, 4-obj 29/30.
  * render->auto-tag fidelity: shape 40/40, colour 40/40 -- the generated PIXELS read back
    as the composed shape/colour, so the image is a real analysable picture not noise
    (texture is carried in the vector but not painted by make_scene, so render fidelity is
    judged on shape+colour).
  * animation: hold base tags, sweep ONE attribute through a sequence, compose+render a
    frame each; the trajectory is "real" iff every frame's vector factors back to its
    intended value -- 100% on-target for a colour sweep AND a shape sweep, each frame
    carrying its rendered image (procedural video by composition).
test_compose.py: 8 tests (single + multi-object round-trip, novelty-not-recall, render
fidelity, valid image, colour-sweep + shape-sweep animation faithfulness, full-space
coverage). README "Forward compositional generation" section; tour "compose:" line.
CONCEPT: generation here is RUN THE FACTORISER BACKWARDS -- the same resonator that
DECOMPOSES a scene into atoms COMPOSES new scenes from chosen atoms, and the proof it
worked is that decomposition recovers exactly what composition put in. That round-trip is
what makes "native generation" an honest claim rather than "we made a picture": the
artefact is verifiable against its own spec. Roadmap status: Tier 0 (G1+G2), all of Tier 1
(unitary HRR, generation bundle, G6), and Tier 2 (G3 core+persistence, reservoir already
built, AND now forward scene render) -- the entire high-value spine of the roadmap is done.
Remaining are Tier 3 conditionals (FHRR only if a resonator result demands it; G4
adversarial gauntlet; G5 perception ceiling as a reservoir rider) and Tier 4 defer/declines
(numpy LSTM, hyperbolic/rotors, price prediction, learned-distribution gen). The procedural
+ compositional generation path the user wanted is now in place end to end: interpolate
stored plates (morph), decode with better sampling (nucleus text), sonify sequences (audio),
AND compose+render NEW verifiable scenes with animation (this) -- all measured, all honest.

WIRED THE GENERATIVE + PERSISTENCE WORK INTO THE LIVE APP (user directive: "make sure we
wire everything up. Don't leave functionality in test files"). The recent capability was
built into modules + proven in tour + tests, but NOT reachable from the UnifiedMind console
(unified_app.py). Added FOUR panels, each a real Flask endpoint + frontend card + JS +
CATALOG entry (so they show up in the searchable/categorised card UI):
  * /api/unified/compose -- "compose a scene": run the resonator FORWARD to compose a NOVEL
    multi-object scene, render it (returns a base64 PNG data URL), and prove it by
    round-trip (factor the vector back + auto-tag the rendered pixels), plus a colour-sweep
    animation strip. Mind-independent (uses SceneCoder + make_scene).
  * /api/unified/morph -- "morph": coefficient-domain slerp between two rendered shapes vs a
    pixel crossfade, with the two midpoints shown side by side and their ghosting distances
    (crossfade midpoint IS the double-exposure -> ~0; coeff morph blends structure -> >0).
  * /api/unified/nucleus -- "nucleus text": top-p decoding over the loaded mind's FLAT
    n-gram vs plain temperature, reporting the real-word coherence gain. Correctly resolves
    the flat HolographicNGram among _gens (hierarchical schema gens have no _distribution
    and get a clear decline). Verified live: nucleus 0.97 real-word vs temperature 0.79.
  * /api/unified/persist -- "save & reload": snapshot the mind's learned meaning space
    (encoder._text.context word->vector) into a Vocabulary, save through holographic_core's
    versioned save(), reload via load(), and verify identical vectors AND identical
    nearest-neighbour structure for a probe word, plus that a bumped state_version is
    refused. This is the user-facing face of G3 persistence (full UnifiedMind serialisation
    stays deferred; the learned meaning space is the meaningful, cleanly-persistable slice).
TWO REAL BUGS surfaced BECAUSE of the wiring (the point of wiring): (1) make_scene's
palette had no "grey" while COLOURS includes "grey", so composing a grey object crashed the
renderer -- FIXED (grey now renders). (2) test_every_card_has_a_catalog_entry hardcoded a
DUPLICATE copy of the catalog keys (brittle, and exactly the "logic stranded in a test"
the directive warns against) -- rewrote it to DERIVE the keys from the app's own CATALOG
via the page source, which then caught a substring collision ("factorize" inside
"factorizer" in the compose card's h2) -- reworded the h2 to "resonator". Audited all recent
test files: only test fns + tiny fixtures, no stranded reusable functionality.
test_app_generative.py: 9 tests hitting the LIVE routes (panels present + wired, compose
round-trips, morph beats crossfade on ghosting, nucleus needs/uses a dataset, persist
round-trips the meaning space, version guard). Measurement tools (holographic_measure,
holographic_ablate) intentionally stay CLI/tour-facing (heavy multi-seed corpus loops are
wrong for a web request); the reservoir stays unfeatured (it is a kept negative). Tour gains
an "app:" line confirming 4/4 endpoints live; README notes the wiring under the generative
sections. Everything reachable from the UI now, not stranded in the library or tests.

PROPAGATION SWEEP -- making the new advancements utilized at ALL levels of the stack
(user directive). Audited where each advancement lives vs where it SHOULD be used, then
filled the genuine gaps (measure-first, backward-compatible defaults, keep negatives).
FINDINGS + ACTIONS:
  1. PERSISTENCE was SILOED: only Vocabulary/HolographicMind/HoloForest could round-trip,
     but the BIGGEST stateful object -- SelfOrganizingMind (== UnifiedMind.memory) -- had
     none. Added to_state/from_state to SubPrototypeMemory, TextEncoder, UniversalEncoder,
     and SelfOrganizingMind; registered SOM in the core registry; upgraded core save/load
     _flatten/_rebuild to recurse into NESTED dicts (so a composite object persists through
     one npz). SOM now round-trips with IDENTICAL classifications.
     - REAL CORRECTNESS BUG surfaced by this: a reloaded mind diverged on NEVER-SEEN words
       because the Vocabulary rng was reset to seed-start while the original had advanced
       during training, so unknown-word index atoms minted differently. FIX: persist the
       Vocabulary rng bit-generator state in to_state, restore it in from_state -- now even
       post-reload mints match a never-saved run, verified by identical classifications on
       unknown-word probes. (Captured self.seed on Vocabulary too; additive.)
     - App /api/unified/persist UPGRADED from saving just word-vectors to saving the WHOLE
       SelfOrganizingMind (prototypes + labels + encoder), verifying identical
       classifications on reload.
  2. NUCLEUS DECODING was siloed in holographic_generate + one app panel; the CORE
     generator (HolographicNGram.generate, used everywhere incl. /api/unified/generate)
     still did plain temperature. PROPAGATED: added top_p=1.0 param to
     HolographicNGram.generate (top_p=1.0 is BYTE-IDENTICAL to the old behaviour --
     backward compatible; top_p<1.0 = native nucleus). Threaded top_p through
     UnifiedMind.generate (try/except so hierarchical schema gens that lack it are safe)
     and the main /api/unified/generate route + a top-p control on the primary Generate
     panel. Measured on real Alice: default temp real-word 0.889 -> native nucleus(0.85)
     0.985. holographic_generate.generate_text kept as the richer path (it also has the
     repetition penalty); no duplication that matters.
  3. UNITARY ATOMS in holographic_mind._roles / RecordEncoder -- MEASURED, NOT ADOPTED
     (kept negative). Records do role-filler binding (the unitary win case), so measured
     Gaussian vs unitary field-recovery under stress: 8-12 fields both perfect; 24 fields
     Gaussian 0.990 vs unitary 0.994 (+0.003, CIs OVERLAP -- within the noise). For the
     RecordEncoder's actual few-field use there is NO measurable gain, and switching risks
     the categorical-symbol path that feeds bundle-based similarity, so NOT adopted. The
     discipline: measured, doesn't help here, don't adopt.
TESTS: +SelfOrganizingMind round-trip (incl. unknown-word probes) in test_core_persistence;
+backward-compatible nucleus test in test_holographic_text; app persist test upgraded to
whole-memory. All edits ADDITIVE (new methods + captured seeds + json import + nested
flatten + a backward-compatible top_p default). VALIDATED: the foundational Vocabulary
change rippled nowhere -- brain/tree/navigator/slime/image/vision/scene/measurement suites
+ the creature gauntlet INCLUDING the rescue_cracks canary all pass. Net: persistence spans
the stack, nucleus is native to the core generator and reachable from the main panel, and
the one advancement that DIDN'T help where it was a candidate (unitary in records) is on
the record as a measured negative.

SPEED + COMPRESSION PASS, AND "ONE BRAIN" INTEGRATION (user directive: optimize speed/
compression where possible, and make sure everything routes through the main brain --
"I don't want a bunch of different brains").
ONE-BRAIN INTEGRATION: audited the *Mind/*Brain landscape. Architecture was already
well-composed -- UnifiedMind is the main brain and OWNS its parts (SelfOrganizingMind as
.memory, HolographicMind as ._brain assembled on actions(), SharedMind for NPCs); the
design comments even say "ONE perception / ONE memory / ONE decision brain". The
FRAGMENTATION was in the app: the compose + morph endpoints built their OWN standalone
SceneCoder / HolographicArchive instead of using STATE["mind"]. FIX: gave UnifiedMind an
OWNED scene faculty -- lazy self._scene = SceneCoder(min(dim,1024), seed) (a singleton on
the same substrate), plus methods compose_scene / decompose_scene / render_scene /
morph_scene. Routed the app's /compose and /morph endpoints through STATE["mind"] (falling
back to a fresh UnifiedMind only if none loaded, so the panels still work standalone).
Captured self.seed on UnifiedMind for the owned faculties (additive). Now there is one
brain with a scene faculty, not a parallel scene engine.
SPEED: profiled the hot primitives -- found Vocabulary.cleanup was ~1925us on 500
candidates because it LOOPED in Python calling cosine() per name. Vectorised it to ONE
cached matrix-vector product (stack @ noisy / ||noisy||; stored atoms are unit-length so
the dot is the cosine up to the query norm -> same argmax). ~9.5x faster (1925 -> 202us),
bit-for-bit identical answer (verified vs the old loop on 20 probes). Cache (_matrix())
rebuilds only when the atom set changes; the explicit-candidate-subset path keeps the
simple loop (small, one-off). This speeds the WHOLE stack since cleanup is used in text
classification, scene factoring, recall, brain perception. Checked the other
max(key=lambda: cosine) scans -- they are small-N classifier scans (4-6 classes), NOT
hotspots, so left alone (don't optimise what isn't measured hot).
COMPRESSION: holographic_core.save now stores float arrays as float32 by DEFAULT ->
~halves every saved mind (SelfOrganizingMind 126200 -> 64752 bytes). Vectors are only
compared by cosine where float32 is ample; on realistic probes behaviour is unchanged
(a decision only flips on an exact tie). HONEST nuance kept: float32 CAN flip a near-tied
classification, so save(compress=False) is the bit-exact opt-out, and the two bit-exact
round-trip tests (Vocabulary vectors, SOM classifications) use compress=False; a new test
asserts compress=True halves the file AND preserves classifications on 60 realistic probes
(>=58/60). brain round-trip stays identical at float32 (atol 1e-4). TESTS: +float32
compression test; existing exact tests pinned to compress=False. All edits additive/
behaviour-preserving. VALIDATED: cleanup vectorisation broke nothing -- brain/relations/
encyclopedia/sequence/segment/mind/resonator + creature gauntlet incl. rescue_cracks canary
all pass; scene integration green across unified/scene/compose/app suites. Net: one brain
that owns its scene faculty, a ~9x faster cleanup felt across the stack, and half-size
saves by default with an exact opt-out.

UNIFIEDMIND INTERNAL-CONSISTENCY / DEDUP AUDIT (user directive: make UnifiedMind utilise
all its capabilities consistently; find duplicate efforts to fold into one generic helper;
find where some features do things better -- compression/speed/proofs -- and apply the best
approach uniformly; note that a lot was developed in tests/siloed and now under one roof may
still need optimisation).
FINDINGS + ACTIONS:
  1. DUPLICATE + INCONSISTENT SPEED (the big one): the brain's HOTTEST scan,
     SubPrototypeMemory.classify / label_scores (every UnifiedMind classify/recall/decide/
     classify_robust routes through it), was STILL a Python per-prototype loop -- the exact
     pattern already vectorised in Vocabulary.cleanup, and the fast (mat @ vec).argmax form
     already existed elsewhere (holographic_mind.py:279, organizer kmeans:145). Applied the
     best approach uniformly: cached unit-matrix product + masked argmax. ~17x faster at 64
     prototypes (56->3.2us), gap widens with prototype count. Winner BIT-IDENTICAL to the
     old loop; score matches to machine epsilon (matrix sum order vs per-element loop, ~1e-16
     -- verified zero label diffs over 40 probes incl. the `among` modality restriction).
     - REAL STALENESS BUG caught by measuring: add() can fold a vector into an existing
       same-label prototype, mutating its unit IN PLACE without changing the prototype count,
       which left the cached matrix stale (classify returned 0.72 instead of 1.0 for a
       prototype's own unit). FIX: a _gen mutation counter bumped on every add(), and the
       _stack cache keyed on it (plus list identity + length). Now correct. This is why the
       project measures optimisations -- it would have silently corrupted online learning.
  2. SMALL-N CLASSIFIER SCANS (LanguageID.identify ~265, TopicSorter.classify ~334 in
     holographic_text.py): max(key=lambda: cosine) over a HANDFUL of profiles/prototypes
     (one per language / per topic). NOT hot (called once per query, tiny N), and the
     dict-comprehension max is readable. Deliberately LEFT ALONE -- vectorising would trade
     readability (a stated preference) for no measurable gain. Don't optimise what isn't hot.
  3. COMPRESSION CONSISTENCY: the persist panel now saves the WHOLE SelfOrganizingMind at
     float32 by default (the core save() compression from the prior pass) -- verified end to
     end: 19 prototypes, identical classifications, ~half the bytes. The codec/archive
     already quantise; no redundant uncompressed store found.
  4. PROOFS/VERIFICATION: round-trip verification is present where it is meaningful (compose
     -> factor-back; codec roundtrip_ok; verify_structure lag-coherence). find() already uses
     the smart coarse_to_fine escalating resolve for large stores. No duplicate verification
     logic to fold; the recall-type paths return a similarity score which is the right
     confidence signal for an approximate store (a hard round-trip only makes sense for the
     exactly-invertible compose/codec paths). Left as-is by design.
TESTS: +2 in test_holographic_organizer (vectorised classify matches the loop incl. `among`;
cache invalidates on in-place update). All edits additive/behaviour-preserving. VALIDATED:
organizer/unified/mind/relations/encyclopedia + text/sequence/segment/predictive/structure +
the creature gauntlet INCLUDING rescue_cracks canary all pass -- the foundational classify
change is behaviour-identical through the whole brain. Net: the brain's core scan now uses
the one fast pattern everywhere it's hot, with a correctness fix the speedup surfaced; the
genuinely-tiny scans stay readable; compression and verification are already consistent where
they belong.

THIRD-PARTY STATIC-ANALYSIS PASS (user directive: use third-party tools to analyse the
code, look online if needed, do another pass). Installed and ran ruff 0.15, vulture 2.16,
bandit 1.9 (pip --break-system-packages). Triaged by signal -- fixed real issues, left
deliberate compact-style and false positives alone.
FIXED (real issues):
  * B033 (ruff): STOPWORDS set in holographic_text.py contained "did" TWICE -- removed the
    stray duplicate (harmless to behaviour since a set dedupes, but a copy-paste slip).
  * F841 dead computations in PRODUCTION code: holographic_unified.py built a `bound`
    string that was immediately superseded by the vals/parts fill loop and never read
    (leftover from an earlier approach) -- removed. holographic_schema.py computed `pos` and
    `gen_atoms` in the source-attribution path, never read -- removed (kept `segments=[]`
    which IS used). unified_app self-discovery endpoint fetched `mind = STATE["mind"]` and
    never used it -- removed.
  * F401 unused imports (22): removed across the stack. Notably several `cosine` imports
    became unused BECAUSE the earlier vectorisation replaced per-name cosine loops with the
    matrix path -- confirms that refactor was thorough. After this, F811 (redefinition) went
    to zero on its own (the unused module-level cosine that shadowed in-method imports was
    the cause).
  * F541 f-strings without placeholders (17): dropped the stray f-prefix on constant
    strings in _demo()/__main__ print blocks. Cosmetic.
  * bandit B306 MEDIUM: unified_app persist endpoint used tempfile.mktemp() (deprecated,
    TOCTOU race). Switched to mkstemp() (atomic create + fd), os.close the fd, use the path.
    Verified the persist panel still round-trips.
  * B905 zip-without-strict: added strict=True to the two zips with a real 1:1 invariant
    (SubPrototypeMemory.label_scores labels/sims; UnifiedMind._record_items vecs/payloads).
    Turns a silent truncation into a loud failure -- consistent with the project's
    fail-loudly stance (version guard). Verified the invariants actually hold on real data.
LEFT ALONE (deliberate / false positive):
  * vulture flagged app.py @app.route handlers as "unused functions" -- FALSE POSITIVE
    (Flask calls them via the decorator; vulture can't see that). The flagged METHODS
    (penalize_recent, capacity_report, replay_plan, is_a_transitive, quantize, changes) are
    all exercised by tests/tour -- genuine public API, not dead. Net: NO genuinely dead
    PRODUCTION code found -- the consolidation passes worked.
  * ruff E702 (semicolons), E741 (ambiguous `l`), E731 (lambda assign), B007/B008: compact
    scientific-code style and intentional; readable-code preference says leave them.
  * Two F841 in _demo()/__main__ scaffolding (organizer `test`, scene `true_tags`): demo
    illustration, not production logic -- left.
All edits additive/behaviour-preserving (removals of provably-dead code + one security
hardening + two fail-loud guards). VALIDATED: full batched suite incl. the rescue_cracks
canary, tour smoke, both apps import -- all green at 539 (no new tests; this was cleanup +
hardening). Tooling recorded here so the pass can be re-run: `ruff check --select F,B`,
`vulture *.py --min-confidence 60`, `bandit -r app.py unified_app.py -ll`.

REVISIT PASS (user: recent capabilities may make old experiments bear fruit -- salt
fingering, surface tension, gravity lensing, spectral decomposition). Checked each named
concept HONESTLY against what we've actually built lately (vectorised cleanup/classify,
float32 compression, full-stack persistence, the owned scene faculty + faster resonator,
nucleus decoding, and the projection/consolidation low-rank lens). The discipline: only
re-open where a recent capability touches what ORIGINALLY blocked the idea; do not
manufacture revivals.
  * SALT FINGERING (#1) -- RE-OPENED and shipped a small win. The original block was "no
    low-dimensional variable to stratify on"; the consolidation work later produced exactly
    that (the ~22-36-dim shared subspace). Re-measuring on the REAL encoded substrate (not
    synthetic blobs) showed the signal was never actually dead there (~7 sigma) and predicts
    split benefit at r=0.94. Added the conservative default-off fingering_prescreen to
    auto_reorganize (skips the sweep when nothing fingers; never changes the choice; ~21%
    faster on stable memories). See the REVISITED note in sec.1.
  * SURFACE TENSION (#2) -- still parked. It was a REFINEMENT of a gate that already works,
    plus a standing curation negative. Nothing recent touches the gate or that negative, so
    there is no new opening. (Would only matter if the 1-SE rule were shown to mis-fire.)
  * GRAVITY LENSING (#3) -- still parked. It RE-DESCRIBES existing machinery (frequency
    weighting + the multi-resolution organizer's coarse/fine images). Recent work didn't
    create a non-redundant prediction, so still no new mechanism falls out.
  * PRISM / SPECTRAL DECOMPOSITION (#5) -- already real in the ResonatorNetwork, which the
    recent work made an OWNED, faster faculty on UnifiedMind (scene()/decompose_scene). So
    the idea is now better-integrated than when parked, but the forager premise it once
    seemed to point at stays refuted (no state aliasing). No new separate mechanism; the
    improvement was integration, already done in the one-brain pass.
NET: one genuine re-opening that measurement turned into a small shipped optimization, and
three concepts honestly confirmed as still-correctly-parked (a refinement, a re-description,
and an already-realised idea) -- recent capabilities did not change their blockers. The
honest correction worth keeping: salt fingering's original negative was measured on
synthetic Gaussian blobs, not the real encoded substrate; re-measuring on real data (made
worth doing by the consolidation lens) flipped it. +1 test (fingering pre-screen invariants).

## REVISIT PASS -- re-examining the shelved concepts against recent capabilities

Prompt: a lot of new capability has landed (vectorised cleanup + prototype classify,
full-stack persistence, the owned scene faculty on UnifiedMind, native nucleus decoding,
float32 saves). The salt-fingering note set the precedent: a clean negative became a small
win once a LATER capability (the consolidation lens) lifted its specific blocker, and the
lesson was "re-measure when a capability unlocks, on REAL data." So each shelved concept was
re-checked against the new capabilities -- does any stated blocker actually lift? Measured,
not asserted. The honest result: the verdicts HOLD. Writing down the re-check because a
negative re-check is itself the deliverable here -- it records that these were reconsidered
with the new tools and did not move, and WHY.

1. Salt fingering -- ALREADY revisited and shipped (fingering_prescreen, default-off, can
   only avoid work never change the choice). Nothing further to do; this IS the precedent.

2. Surface tension (coherence-scaled split margin vs the flat 1-SE rule) -- RE-MEASURED, and
   the blocker holds for a sharper reason than before. The recent vectorised classify makes
   per-prototype coherence cheap, so the "too expensive to justify a refinement" cost
   argument is gone -- but cost was never the real blocker. Measured the flat 1-SE gate on
   the standard beds: it splits the bimodal world (k=2) and keeps the unimodal world (k=1)
   on 6/6 seeds each -- ZERO errors. There is no mistake for a coherence-scaled margin to
   correct. A refinement that makes a perfect gate more complex is not a win. Verdict
   unchanged: not implemented, now with a measurement showing the gate is error-free on the
   exact task surface tension targets.

3. Gravity lensing (bend hot queries toward massive prototypes; "multiple images") -- the
   one non-redundant prediction (keep coarse+fine prototypes for a trafficked class) is still
   exactly what the multi-resolution organizer does on demand. Vectorised classify now scans
   all sub-prototypes in ONE matrix product, so a frequency-weighted bend is a pure prior
   with no error to fix. No new mechanism. Verdict unchanged.

5. Prism / spectral decomposition -- its legitimate home, the ResonatorNetwork, is now a
   first-class faculty on UnifiedMind (compose_scene/decompose_scene). Re-measured the
   resonator's ceiling by object count on the owned faculty: 30/30, 30/30, 29/30, 30/30,
   29/30 exact scene round-trips for 1..5 objects. It is already near-perfect; the 1/30
   residual is HRR cross-term noise, not a structural gap a spectral pre-separation would
   fix. The analogy is fully realised and needs nothing. Verdict unchanged (now with a
   ceiling measurement on the new faculty).

The throughline (consistent with the whole project): a capability unlock is a reason to
RE-MEASURE a shelved idea, not to assume it now works. Salt fingering moved because a new
capability supplied the precise thing its blocker named (a low-dim subspace) AND re-measuring
on real data showed a real, correlated payoff. The others did not move because their blockers
were never "missing capability" -- they were "the existing mechanism already does this" or
"there is no error to fix," and the new tools, measured, confirm that rather than overturn it.
No new features this pass: the deliverable is the kept, re-verified negatives.

## FRACTAL SWEEP -- self-similarity, inception of structure, regenerate-from-seed

Prompt: look for ways to make things more fractal across levels -- same above same below,
inception of structure/functionality, deterministic reconstruction from seed values. Held
to the project rule: self-similarity earns its place only as a MEASURED win, not a metaphor.

AUDIT (what is already fractal -- mapped, not reinvented):
  * REGENERATE-FROM-SEED is already the deep theme and already realised at multiple levels:
    Vocabulary.to_state stores the RECIPE (seed + ordered names) not the matrix and replays
    get() to reconstruct -- measured 170x smaller than a raw 400x512 float32 matrix (4824 vs
    819200 bytes), exact round-trip. derived_atom(seed,name) mints each atom as a pure
    function of (seed,name) via blake2b (order-independent), so a derived vocabulary persists
    as just names. HoloForest rebuilds its trees from seed; consolidate() rebuilds its basis
    by SVD. The leaf-level fractal compression is DONE and measured.
  * The honest BOUNDARY the codebase already respects: SEED-DERIVED structure (atoms, trees,
    basis) compresses to a recipe; LEARNED structure (prototypes, the encoder's co-occurrence
    context) is genuinely not seed-derivable and is stored explicitly. Conflating them would
    be a cheat. Verified the prototype/context stores correctly keep raw vectors -- not a
    missed fractal opportunity, the correct call.

NEW WIN (a level that was NOT yet self-similar, now is, measured):
  * Scene composition was FLAT -- a scene is a bag of objects (encode_scene = unnormalised
    superposition). Tested whether the SAME machinery works one level up: bind each sub-scene
    to a group-role atom, superpose the groups, then unbind one group and factor it. IT WORKS
    -- the unbind is noisy (cos ~0.58, cross-talk from the other group) but the resonator's
    explain-away cleans it to EXACT sub-scene recovery. So the algebra is already self-similar
    enough for recursion; it just was not exposed. Measured ceiling (honest, kept): 2 groups
    1.00, 3 groups 0.97, 4 groups 0.89, 5 groups 0.82 (2 objects each); 3x3=9 objects 0.87.
    The same capacity limit the flat scene has, one level up. SHIPPED as UnifiedMind.
    compose_nested / decompose_nested (the same bind+superpose / unbind+factor, recursively),
    with group-role atoms minted DERIVED (seed-reconstructable -- the whole nesting regenerates
    from one seed; verified identical super-scene across two minds of the same seed). Wired
    into the app as the "nested scene" panel (/api/unified/nested) with per-group round-trip
    proof + rendered sub-scenes, through the loaded mind's own faculty (one brain). Tour gains
    a "nested:" line; app now 5/5 generative endpoints. Tests: 2 in test_compose (round-trip +
    seed-determinism, and the 3-group >=0.9 boundary) and 2 app-wiring tests.
    One bug the wiring tests caught and I fixed: inserting the nested JS accidentally consumed
    the `async function morphScene(){` declaration line, orphaning the morph body -- the morph
    panel test failed instantly and visibly (what the wiring tests are for), restored.

THROUGHLINE: the fractal principle was already correctly applied where it is VALID
(regenerate seed-derived structure; store learned structure) -- so the sweep did not bolt on
metaphor. The one genuinely new level (recursive scene-of-scenes) was shipped because the
algebra already supported it and the round-trip MEASURES exact at the useful depth, with the
decay honestly reported. Same above, same below -- where measurement says it holds.

## "IS THE APPROACH DATED?" -- a literature pass with measurement

Prompt: someone called the approach dated; search for better ways. Did a real literature
pass (VSA/HDC surveys and comparisons) and checked the one concrete lead by MEASURING it on
this substrate, per the project rule.

HONEST ASSESSMENT (sourced): largely NOT dated.
  * The core primitives are current and well-founded: Plate's real-valued HRR (circular
    convolution binding), resonator networks for factoring (Frady/Kent/Sommer 2020), cleanup
    memory. None are superseded; they are the standard VSA toolkit in the 2022/2023 Kleyko
    surveys and the Schlegel et al. 2021 comparison.
  * VSA/HDC is a LIVE area, not a relic: "vector symbolic architecture" is just the older
    name for hyperdimensional computing (both current), and neuro-symbolic AI rose in 2025
    specifically to address LLM hallucination with interpretable symbolic reasoning -- which
    is exactly what this from-scratch, deterministic, interpretable engine is. A "dated"
    critique most likely compares it to deep learning, which is a CATEGORY difference (the
    project deliberately uses no frameworks/pretrained models/GPU), not obsolescence.

THE ONE CONCRETE LEAD -- FHRR -- CHECKED BY MEASUREMENT:
  * Schlegel, Neubert & Protzel (2021, the most-cited recent cross-VSA comparison) find FHRR
    (Fourier HRR: complex unit-phasor atoms, bind = elementwise complex multiply) performs
    best across their benchmarks. The project defaults to real-valued HRR.
  * Measured on THIS substrate (dim 256, pairs in one key->value trace): FHRR holds far more
    pairs -- 40 pairs real-HRR 0.61 vs FHRR 0.90; 60 pairs 0.40 vs 0.74. The advantage is
    real and large under load.
  * CORRECTION to a natural assumption: the project's existing `unitary` atoms (unit-magnitude
    SPECTRUM, real domain) do NOT capture this -- unitary-HRR tracks real-HRR (0.41 vs 0.40 at
    60 pairs), not FHRR (0.74). The win comes from staying in the complex phasor domain.
  * BOUNDARIES, also measured (kept, honest): (a) at LOW load (<=~10 pairs/256-d, or the
    few-factor records the project normally builds at 512-1024) both are 1.000 -- FHRR changes
    nothing, so the readable real-valued default loses nothing by staying default. (b) FHRR
    does NOT raise the nested-scene composition ceiling: I measured the OUTER group-binding
    layer (real-HRR perfect to 12 groups), so that ceiling is the resonator factoring a NOISY
    UNBOUND sub-scene -- a different bottleneck FHRR can't fix. (I was about to assume FHRR
    would help nesting; the measurement refuted it.)

ACTION: shipped FHRR as a self-contained, opt-in module (holographic_fhrr.py: phasor atoms,
bind/unbind/bundle/sim, PhasorVocabulary, PhasorMemory), exposed through UnifiedMind as an
owned faculty (high_capacity_memory(); one brain, seed-deterministic, singleton). Did NOT rip
out the real-HRR core -- it is the right readable default and not dated, and FHRR only wins in
the high-load key-value regime the project rarely hits. Demonstrated in the tour (real-HRR 52%
vs FHRR 95% at 40 pairs), and tested (test_holographic_fhrr.py: exact algebra, the high-load
WIN >0.1 margin, the low-load NON-win both at 1.000, the owned faculty). The deliverable is an
honest assessment + one measured capability earning its place where it measurably wins, with
its boundaries kept as recorded negatives. The approach is current; FHRR is a tool added for a
specific regime, not a verdict that the core was wrong.

## VECTOR DATABASES -- a literature pass, and the one transferable technique (quantization)

Prompt: lots of talk about vector databases storing info/files; any recent developments to
improve the brain? Searched current VDB/ANN literature (2025-2026) and assessed against the
project's hard constraints (from-scratch NumPy, deterministic, no external services).

HONEST ASSESSMENT: the PRODUCTS don't fit and wouldn't help; one TECHNIQUE transfers.
  * Vector databases (Pinecone, Milvus, Qdrant, Weaviate, pgvector, FAISS) are external
    services / heavy C++ deps for million-to-billion-scale retrieval. Adopting one violates
    the project's premise (no frameworks, no services, runs anywhere, deterministic) -- the
    same category mismatch as "use a transformer." And the scale is wrong: this is a KB-scale
    engine, not a billion-vector store.
  * HNSW (the dominant VDB index) specifically CONFLICTS with the project's core principle:
    its graph is built by order-dependent incremental insertion with random level assignment,
    so it is NOT seed-reproducible the way everything here is. The project's HoloForest
    (random-projection trees) is the right approximate index precisely because it rebuilds
    deterministically from a seed.
  * The literature actually VALIDATES the project's design at its scale: the Feb-2026 filtered-
    ANN study found approximate indexes are often chosen even when an EXACT sequential scan
    gives perfect recall at comparable latency -- i.e. brute-force wins at small/moderate
    scale (which is where this project lives). And the Nov-2025 B+ANN paper identifies HNSW's
    weakness as "fine-grained pairwise computations" and moves toward batching vectors so
    distance can use MATRIX MULTIPLICATION instead of pairwise dot-products -- which is exactly
    the vectorised cleanup/classify this project already adopted. The brain is aligned with
    where ANN research is heading, not behind it.

THE TRANSFERABLE TECHNIQUE -- QUANTIZATION -- MEASURED:
  * Vector DBs shrink stored embeddings with scalar (int8), binary, and product quantization
    (ScaNN's anisotropic PQ, etc.). The project stored learned vectors at float32. Measured
    int8 and binary (sign) quantization of prototype units vs classification fidelity:
    - Clean 6-class world: int8 1.000, binary 1.000 (both lossless).
    - Crowded 100-class noisy space: int8 1.000, binary 1.000 STILL lossless -- because at
      dim 512 the prototypes are near-orthogonal (concentration of measure), so even 1 sign
      bit/dim preserves the nearest-neighbour argmax.
  * SHIPPED int8 as an opt-in save level: holographic_core.save(obj, path, quant="int8") --
    each float array -> signed 8-bit ints with a per-array scale, dequantised on load. Measured
    8881 bytes vs 26944 (float32) vs 51544 (float64) for a trained SelfOrganizingMind: ~3x
    smaller than float32, ~5.8x vs float64, classification lossless (acc 1.000, same-class on
    120 probes). float32 stays the DEFAULT (int8 can flip an exact tie, same as float32 only
    a touch more, so it is opt-in for when stored size matters). Vocabulary recipe path
    untouched. Empty-array edge guarded. Pinned in test_core_persistence.
  * Binary (32x) NOT shipped: lossless on near-orthogonal data but fragile on genuine near-ties
    (sub-modes of one class), and the project's stores are small enough that int8 is plenty.
    Kept as a measured note. The RP-trees already use sign-based routing, so the project is
    not missing the binary idea where it matters.

NET: vector databases are the wrong tool at this scale and break determinism; the brain's
recall is already aligned with current ANN design (batched matrix distance + deterministic
RP-trees + exact-wins-at-small-scale). The one part worth borrowing -- scalar quantization --
is now an opt-in int8 save level, measured lossless at the working dimension. An honest "no"
to the products, a measured "yes" to the one technique that fits.

## DYNAMIC QUANTIZATION -- precision per array from the data's own complexity and size

Prompt: what if quantization were dynamic, depending on the complexity or size of the data?
Followed the project rule -- measure whether a data signal actually predicts where coarse
quantization is safe, build the selector only if it does, keep the boundaries.

THE SIGNAL (measured): vector SEPARATION predicts quantization headroom. The min over rows of
(1 - best-other-cosine) -- the worst row's margin -- tracks complexity: high-dim separated
~0.87, low-dim packed ~0.45, near-duplicate twins ~0.01. And quantization is safe across a
huge range: binary (1 sign bit/dim) stays lossless for classification until the margin is
~0.01, where even float32 fails under noise because the data itself is ambiguous. So coarse
precision is safe almost everywhere -- the question is finding the few arrays where it is not.

THE SELECTOR (measured to engage, not collapse to one level): _auto_quant_kind picks the
COARSEST level whose reconstruction keeps every row's self-recognition AND >= 0.7 of its
float top1-top2 margin (so a noisy query will not flip). On synthetic regimes it correctly
adapts: high-dim separated -> binary; low-dim 24d/8d -> int8; near-duplicate twins -> int8;
tight cluster -> int8. On a real trained mind it chose a MIX -- {f32:2, int8:1, bin:1} --
genuinely per-array, not one fixed choice.

THE SAFETY RULE (the real subtlety): binary preserves DIRECTION but destroys MAGNITUDE, so it
is safe only for UNIT-NORM arrays (the cosine-compared `units`), never for magnitude-carrying
accumulators (`sums`) that continued learning needs. So binary is in the candidate ladder only
when the rows are unit-norm (+/-0.05); otherwise the ladder starts at int8 (8-bit magnitude,
same precision class as the float32 default). Plus a SIZE floor: arrays under ~1024 elements
stay float32 (the per-array scale/spec overhead is not worth it, and tiny arrays are not what
makes a file big -- the 'size' half of the prompt). Verified: after an auto-save reload, the
mind keeps observing and stays uncorrupted (acc 1.000), because the accumulators were never
binarised.

SHIPPED: holographic_core.save(obj, path, quant="auto"). Per float64 array -> binary (np.
packbits of the signs, ~32x), int8 (scale, ~4x), or float32, recorded in a per-array __qspec__
the loader decodes. MEASURED on a trained SelfOrganizingMind: float32 26944B, fixed int8 8881B,
auto 6246B -- ~4.3x vs float32, ~1.4x smaller than fixed int8 -- and classifications match
float32 EXACTLY (200/200 same-class). The win over fixed int8: it compresses harder where the
data allows (binary on the well-separated unit-norm matrix) AND is safer where it does not
(it can only ever pick a level the measured separation supports). float32 stays the default;
int8 and auto are opt-in. Pinned in test_core_persistence (the dynamic mix engages, fidelity
matches float32, continued learning survives, the unit-norm gate, the size floor). Demoed in
the tour. The resolution of last round's caution: dynamic selection with a margin gate is what
makes the aggressive binary level SAFE to offer, which a blanket default could not be.

## APPLY RECENT WORK ACROSS THE STACK -- a cross-stack audit that caught a real boundary

Prompt: make sure the recent improvements are applied to the main brain and all levels of the
stack where applicable. Audited each recent capability at each level (library -> UnifiedMind
faculty -> app endpoint/UI -> tour), and -- the valuable part -- ran the quantization save
levels against EVERY persistable object, not just the SelfOrganizingMind they were measured on.

ALREADY WIRED (verified): nested composition (UnifiedMind.compose_nested/decompose_nested +
app "nested scene" panel + tour); FHRR high-capacity memory (UnifiedMind.high_capacity_memory()
+ app panel + tour); int8/auto quantization (core.save + /api/unified/persist endpoint + the
persist panel's float32/int8/auto selector + tour). So the surfaces were all connected.

THE BOUNDARY THE AUDIT CAUGHT (the reason cross-stack testing matters): quant="auto" was
measured only on the SelfOrganizingMind (classification). Run against the CREATURE VALUE-BRAIN
(HolographicMind) it flipped 62/200 action decisions -- a silent corruption. Root cause: auto's
binary level. Measured why: binary distorts the pairwise-similarity (Gram) geometry by
~0.117-0.204 on EVERY array, including the prototype memory's (0.204). It only LOOKED safe on
the SelfOrganizingMind because wide-margin classification argmax is robust to a 0.2 similarity
shift; the value brain's finer linear readback is not. int8's Gram drift is ~0.002, so int8 is
decision-safe everywhere (classification 0 flips, value brain 1/200 = tie-level, recall forest
0 flips). Binary's safety is DECISION-SPECIFIC and cannot be verified at the generic persistence
layer (which does not know whether the consumer does wide-margin argmax or fine value readback).

THE FIX: quant="auto" now adapts only among the magnitude-preserving, decision-safe levels
{int8, float32} -- int8 where the margin gate proves it lossless, float32 for tiny/marginal
arrays. The 1-bit binary auto-selection from the previous round is REMOVED (helpers deleted),
because the cross-stack check showed it is not generically safe. This walks back last round's
headline "auto picks binary, 4.3x" to an honest "auto picks int8/float32, ~3x, decision-safe on
every brain type." A correction kept in the open, exactly the project's rule: built it, a
broader test revealed the boundary, corrected it. quant="auto" verified across SelfOrganizingMind
(0 flips), HolographicMind value brain (1/200 tie-level), HoloForest (0 flips); pinned by a new
value-brain safety test in test_core_persistence. (Aside, noted not fixed: a consolidated
HolographicMind's value() rejects a raw unprojected probe -- a pre-existing issue independent of
quantization, it fails before any save.)

LESSON: "lossless on the object I measured" is not "lossless on the stack." A quantization
level's safety depends on how the CONSUMER reads the vectors (wide-margin argmax vs fine
magnitude readback), so a generic persistence option must only auto-select levels that are
decision-safe for the most sensitive consumer -- which is int8, not binary.

## CREATURE: AUDIT FOR LATEST TECH -- already optimized, one robustness fix, one kept negative

Prompt: make sure the creature stuff is updated and optimized for the latest tech. Audited the
HolographicMind value-brain's hot paths against the recent improvements, MEASURING on a REAL
trained creature (real GridWorld + CreatureEncoder + episodes), not synthetic random vectors --
which matters: a random-vector brain has NO low-rank structure, so consolidation found rank 94
of 94 and made decide() 8x SLOWER (pure projection overhead, no compression). On a REAL creature
the prototypes are bundles of a small sense-atom vocabulary, so consolidation finds rank 9 and
decide() is 5.4x faster (827 -> 152 us) -- the NOTES claim holds, on real data only.

ALREADY OPTIMIZED (verified): value() is the vectorised matrix-product pattern already (sims =
U @ state, cosine to every prototype at once); decide() projects the state ONCE via perceive_vec
then loops the 4 actions; consolidation (the projection/low-rank work) gives the 5.4x. So the
creature already carries the latest tech.

KEPT NEGATIVE (measured, not shipped): batching the 4 per-action matmuls into one concatenated
(Ntotal x rank) product. Measured 55 vs 58 us -- only ~5% -- because the matmul was never the
bottleneck (rank 9, tiny); the per-action top-k (argpartition over ~1000-2000 prototypes) and the
weighted readback dominate and are unchanged by batching. It was also NOT bit-identical (tie
ordering shifts when sims are sliced from a combined array vs computed per action). 5% for added
caching/offset/invalidation complexity AND a behaviour change is not worth it -- not shipped.

SHIPPED (a real robustness update): value() now projects a RAW full-dim probe into the low-rank
basis when the brain is consolidated. decide() and app.py perceive (project) first, but DIRECT
value() callers (the core save/reload demo, the tour persistence check, _greedy) pass a raw probe
-- which CRASHED on a consolidated brain (matmul dim mismatch, confirmed live). The fix is a cheap
side-effect-free lift (state @ basis when the width is the full dim; already-projected vectors are
left alone). It deliberately does NOT call perceive_vec, which also feeds the flux-guard ring --
calling that inside value() would double-count out-of-basis energy on every per-action call.
Verified: raw-probe value() now matches the pre-projected path exactly (idempotent), decide()
unchanged, the rescue_cracks canary passes (creature behaviour preserved). Pinned in
test_holographic_brain.

(Noted from the previous round, now FIXED by this: the "consolidated value() rejects a raw probe"
gap was this same bug.) Net: the creature was already optimized with the latest tech; the audit's
deliverable is the confirming measurement, one robustness fix, and one honest negative.

## PORTING UPSTREAM FROM A SIBLING PROJECT (TuneFM) -- verified, then adopted/adapted

A sibling project that adopted the holostuff methodology on market data proposed five
improvements to send back upstream "because they help every application, not just trading."
Treated the document the holostuff way: VERIFIED every claim against the actual source (all
five were accurate about what holostuff did/didn't have), then let MEASUREMENT on this
substrate decide what earns its place. All five adopted, two adapted from the proposal.

1. rfft bind + bind_batch/bind_fixed. The core bind used the COMPLEX fft; atoms are real, so
   the REAL fft is exact (measured equal to ~7e-17) and ~1.5x cheaper -- switched it, and the
   full algebra suite + the rescue_cracks canary pass, so every caller gets 1.5x for free.
   Added bind_batch/bind_fixed (vectorised): ~2x even at 3 fillers, 5.5x at 64. Wired into
   RecordEncoder.encode (verified bit-identical to the per-field loop). The new primitives are
   in real use, not siloed.

2. ScalarEncoder RBF kernel + kernel_at. Default stays sinc (uniform phases). kernel="rbf"
   mints Gaussian phases -> a non-negative monotone RBF kernel; kernel_at(dx) returns the
   similarity the encoder analytically realises (Bochner). VERIFIED the encoder IS its kernel
   (measured cosine matches kernel_at to 0.001) and MEASURED the real win: recovering a
   bimodal density within the range, RBF gets corr 0.73 and resolves both modes where sinc --
   one lobe over the range by construction -- gets 0.40 and sees one. (Did NOT find the
   negative-lobe density corruption the document claimed in my configs, so did not claim it;
   the mode-resolution win is the measured justification.)

3. holographic_honesty.py -- the ablation ethos as a callable instrument. walk_forward_recall
   (six checks a recall predictor must survive: beat chance, beat the persistence baseline,
   collapse under a shuffle, magnitude correlation, net of cost) and bh_fdr (Benjamini-
   Hochberg/Yekutieli false-discovery control -- the real gap: grep for benjamini returned
   nothing, and a library that generates many candidates needs FDR). The module passes its
   OWN audit: a planted edge clears chance with its shuffle collapsing, pure noise does not,
   bh_fdr rejects 5 planted discoveries and spares 200 nulls. (Did NOT rewire the known-flaky
   market test onto it -- left available for adoption rather than destabilise that test.)

4. HoloForest.recall(with_agreement=True). The trees are independently seeded, so their
   agreement is a free abstention signal. Returns (best, agreement=fraction of trees whose
   own pick equals the forest's). Stored item -> 1.00, random query -> 0.59, so it separates
   known from unknown. Guarded so the DEFAULT path is byte-identical (verified, incl. cosine-
   tie order) and not slowed (per-tree work only when the flag is set).

5. HolographicArchive.verify(). ADAPTED from the document's sketch, which assumed a bucket-of-
   members API this archive does not have (it is WHT disjoint-slot image superposition). The
   real, checkable version: reconstruct the most collision-prone stored images and confirm
   each recalls back to its OWN index -- the disjoint-slot orthonormal-key guarantee, checked
   on this build rather than assumed, using only the archive's own API. 6/6 exact clean and
   after 4-bit plate quantisation; catches identity loss if quantisation ever corrupts it.

Deliberately NOT taken (the document was honest about provenance and so is this): the WHT
plate memory, median-split trees, and resonator peeling were re-derivations on their side of
things holostuff already had. Net: cross-project port, every claim verified against source,
every change measured before adoption, two adapted to the real APIs, all backward-compatible.

## VENDORING REAL MARKET + ON-CHAIN DATA FROM A SIBLING PROJECT (not just code, data)

The sibling project (TuneFM-SOL) that produced the upstream-port document also shipped its
real datasets and some app/infra code. Brief: add its market and trading datasets alongside
the existing images/text/records, and extract anything else useful. Treated it the holostuff
way -- vendor REAL data in a lean, native form, wire it into the existing machinery, and be
explicit about what was deliberately left out.

WHAT WAS VENDORED (real, checked-in, lean):
  * data/sol_market.npz (0.60 MB) -- real SOL/USDT bars from Binance, multi-timeframe (5m/1h/1d)
    plus BTC/ETH 1d cross-assets, each [time,open,high,low,close,volume,taker_buy,ofi] (taker_buy
    = aggressive buy volume, ofi = order-flow-imbalance sign -- microstructure the permutation
    tests and CandleCoder can chew on), plus a funding-rate [time,rate] series. Built from the
    richest source file (the 62 MB -big export) but capped to recent bars and float32, so it is
    0.60 MB, not 62. load_sol_market(timeframe=...) in holographic_market.py; feeds the EXISTING
    CandleCoder once prices are normalized to a working level (the coder lives in bp-space around
    1.0 -- a real contract detail, documented in the test: SOL round-trips to 0.13% normalized).
  * data/onchain_traders.json (122 KB) -- realized on-chain Jupiter Perpetuals trades the sibling
    read off Solana's public ledger: 58 wallet profiles + 869 realized trades. HONEST by
    construction: every profile carries trade COUNT and a per-trade edge t-stat beside its PnL, so
    a wallet green on a handful of trades reads as luck, not skill (the same n-problem the engine
    keeps flagging). load_onchain_traders() in holographic_market.py.

WIRED IN (reachable, not siloed):
  * load_onchain_world() in unified_app.py turns the wallets into role-bound RECORDS (win_rate,
    leverage, hold, bias, blowups bucketed to categoricals) LABELLED by honest skill: "skilled"
    only when edge_t_stat >= 2 AND positive per-trade net, "burned" if liquidated and negative,
    else "neutral" (distribution 4/37/17). Registered as the "onchain" dataset in DATASETS, so it
    appears in the app's dataset picker next to world/reuters/brown/... Measured: the records are
    genuinely learnable -- absorb a 70/30 split, held-out classify accuracy 0.95 (vs 0.33 chance
    for the 3 classes). A tour line demonstrates both the SOL candles and the onchain records.

DELIBERATELY NOT VENDORED (and why -- this matters as much as what was taken):
  * onchain.py -- a live Solana/Helius RPC fetcher (hand-rolled borsh event decoder). It is
    network-dependent infrastructure that does not fit holostuff's offline, minimal-dependency
    philosophy, and the sandbox blocks Solana RPC anyway. Its OUTPUT (the trader JSON) is the
    useful artifact; the fetcher is not. Its honest caveat is preserved in spirit in the labeling
    above ("up over a window is survivorship/variance until proven; report count + per-trade edge
    t-stat, treat green-on-12-trades as luck").
  * app.py (734 KB FastAPI app) and index.html (407 KB) -- the sibling's whole web app. Far too
    large and infra-specific; holostuff has its own app surface. Not vendored.
  * The 62 MB / 12 MB / 9 MB raw soldata exports, the quiz CSVs, TEAM_REGISTRY.md -- raw or
    project-specific; the useful market signal was distilled into the 0.60 MB npz instead.

The sibling's README is itself a substantial honest-measurement document (same methodology,
including holographic experiments); it was read for provenance but not vendored wholesale.
Net: two real datasets added in lean native form, wired into the existing CandleCoder, dataset
registry, tour, and tests; honest labeling baked in; clear boundary on what infra was left out.

## FILESYSTEM: the recovery zip was silently dropping data/*.npz

While vendoring sol_market.npz, found the close-out zip's exclude list carried a blanket
-x "*.npz", meant to skip transient model snapshots -- but it ALSO dropped the checked-in
datasets data/sol_5min.npz and data/sol_market.npz from the recovery zip. Since this tree is
not a git repo, the zip is the ONLY recovery path, so the tick dataset had quietly been at risk
the whole time. Fixed by removing the blanket *.npz exclude from the close-out rebuild so
data/*.npz is kept; transient snapshots are written to /tmp or root scratch (handled by the
"_"-scratch and /tmp conventions) rather than under data/, so they are still not shipped.

## WHERE THE RECENT IMPROVEMENTS REACH ELSEWHERE -- an audit, with one unlock and one kept negative

After the upstream port, traced each improvement to see where else it applies or what it unlocks.
Measured every candidate; applied what earned its place, kept the negatives.

THE UNLOCK -- bh_fdr controls the project's own ablation table. The flagship "is VSA
load-bearing?" table (holographic_ablate.py) scans ~6 subsystems and decides each verdict on
that subsystem's OWN 95% CI. But a table is a SCAN, and scanning enough subsystems means one can
clear a per-test bar by luck -- exactly the exposure the honesty module's bh_fdr exists for. Added
fdr_verdicts(): each subsystem gets a PAIRED PERMUTATION p-value (holo vs baseline; seeds are
shared so scores pair by seed -- a sign-flip test, enumerated exactly for the handful of seeds;
falls back to a two-sample label-permutation test when an arm ran fewer seeds), then bh_fdr
(Benjamini-Yekutieli, dependent=True since the subsystems share data/methodology) holds the
false-discovery rate among the surviving "load-bearing" calls across the WHOLE family. On the real
table both load-bearing verdicts (topic-classify, noisy key->value) are unanimous 6-seed wins
(p=0.0156) and SURVIVE the family-wise bar. HONEST PROPERTY found while testing: BY is conservative
-- a single unanimous 6-seed win (finest p reachable = 1/64) does NOT survive BY across a 6-test
family on its own (top-rank threshold ~0.007); the two real verdicts survive because they SHARE the
win (the rank-2 threshold is more lenient). So 6 seeds is near the floor for clearing BY-FDR; more
seeds tighten it. Wired into the ablation _demo (p + FDR columns) and the tour. Pinned in
test_ablations. This makes the engine's core epistemic instrument rigorous against multiple-testing.

APPLIED (consistent, behaviour-safe) -- bind_batch in KnowledgeStore.add. The relations record
builder used the same bundle([bind(role,filler) for ...]) pattern already vectorised in
RecordEncoder; switched it to one batched FFT. Identical to the loop at 1e-12, relations recall is
exact-key (wide margin, robust to the ~1e-16 batched-vs-scalar difference), all relations tests
pass. A small, consistent application -- faster as records widen, no behaviour change.

KEPT NEGATIVE -- bind_batch in the creature encoder. CreatureEncoder.encode binds role->value for
every sense each step (a hot path: encode ~169us actually costs MORE than decide ~80us). bind_batch
measured 1.38x there and identical to the loop at 1e-12 -- BUT batched and scalar FFT differ at
~1e-16, and that is enough to flip a knife-edge tie-break in the starved-maze rescue trajectory:
the rescue_cracks CANARY FAILED. Reverted. The creature's deterministic reproducibility outweighs a
per-step 1.4x. Note the asymmetry with RecordEncoder (which tolerated the identical change): record
classification is wide-margin argmax, the maze rescue is tie-sensitive -- the SAME 1e-16 perturbation
is harmless in one consumer and trajectory-changing in the other. This is why the canary gates
compute-path changes and bit-exactness matters for the creature specifically.

ASSESSED, NOT FORCED (no high-value internal consumer right now, so left as available capability):
  * HoloForest with_agreement (abstention signal) -- the forest is used for scale benchmarks and
    the recall-index ablation, not for any decision that should abstain; the unified mind's
    classify/cleanup run on vectorised prototype matrices, not the forest. Wiring agreement somewhere
    just to use it would be decorative. Kept as an available signal for callers.
  * ScalarEncoder RBF / kernel_at (non-negative density kernel) -- no existing path reads a scalar
    bundle as a density where sinc's lobes bite; forcing RBF into CandleCoder (whose flaky test and
    bp-space contract argue against churn) would be fishing for a win. Available, not forced.
  * archive verify() pattern -> other memories -- does NOT transfer: HolographicMemory/
    PartitionedMemory are intentionally LOSSY (finite capacity is a measured feature, already
    instrumented by capacity_curve/recall_all), so "verify exact recall" is the wrong check for them.
    The archive's verify() is specific to an exact-recall store with stored ground truth.

Net: one genuine unlock (FDR over the ablation family), one consistent safe application, one kept
negative with a sharp lesson about why bit-exactness matters more in the creature than elsewhere,
and three capabilities honestly left unforced rather than wired in decoratively.

## ADVISORY-PANEL DESIGN REVIEW -- sixteen lenses, debated to one build + three queued

Ran the cross-disciplinary panel as a design review: each seat proposed one change grounded in
its field's real published method, then the proposals were clustered, cross-examined against
holostuff's constraints, and measured before belief. (Attributed to seats/methods, not to the
individuals as personal opinion.) Two proposals were measured during the debate to keep the
convergence evidence-based:
  * Duda/ANS: the int8 stream carries ~6.85 bits/symbol of entropy vs 8 bits stored -> ANS would
    save ~14% losslessly on top of int8. Real but modest; a bit-exact NumPy coder is fiddly. QUEUED.
  * Tarter/Cranmer null-calibration: the random-query null is already well-behaved; a clean match
    sits far above it. Cheap, low-risk, broadly useful. BUILT.

BUILT -- RecallNull (holographic_honesty.py): turns a recall/cleanup similarity into an HONEST
false-alarm probability. fit() draws random queries against a codebook and records the best-match
cosine each reaches (the empirical noise floor); pvalue(score) = fraction of that null reaching
score or higher = the chance noise alone would look this good. calibrated_recall(query, codebook)
returns (idx, score, p). MEASURED: a clean stored atom -> p~0; random queries are well-calibrated
(P(p<=0.05)=0.043, P(p<=0.20)=0.201, so a p<=alpha gate has false-alarm rate ~alpha); it tracks the
capacity cliff per recall (recalling pair-0 from a filling key->value trace, score 0.71->0.15 but p
stays ~0 because it is still above the noise floor -- the calibration CONFIRMS each recall is real);
and it refuses to over-claim when a signal is genuinely swamped (p rises toward 1). Complements the
two existing abstention signals: HoloForest cross-tree AGREEMENT (structural) + RecallNull FALSE-
ALARM PROBABILITY (statistical). Pinned in test_holographic_honesty, demoed in the tour.

QUEUED with evidence (in priority order, for a future round):
  1. Tero (2007) flow-conductance Physarum solver (Adamatzky seat) -- tubes thicken with Poiseuille
     flux; a genuinely different algorithm from the current elitist-ant pheromone. Bar: beat
     elitist-ant on the braided maze at equal cost.
  2. ANS entropy-coded save level (Duda seat) -- measured ~14% lossless on int8, if a bit-exact
     NumPy coder round-trips cleanly.
  3. L1 / compressed-sensing archive recovery (Ozcan seat) -- behind a measurement: does it beat the
     archive's CG least-squares past ~60% plate erasure?

PARKED with rationale: sparse thinning / exact-inverse unbind / XPBD cleanup (overlap FHRR +
resonator, small expected gain, hot-path risk); SAH tree split (the ablation already shows median
split is not the bottleneck -- "scale win, not accuracy"); tensor-train codebook + Stam Helmholtz
binding (research-grade); SDF / quality-diversity (peripheral to the core algebra, fine as app
extensions). The panel's real output was the DEBATE cutting 15 proposals to 1 build + 3 evidenced
queue items + honest parks -- the engine's own method applied to its own roadmap.

## DENOISING & GAUSSIAN SPLATS -- the measured cluster shipped (panel addendum II)

Built the four measured breakthroughs that share one engine ("one operation seen several ways":
a denoiser is a map of the manifold signals live on, and holostuff already owned those maps).
All additive, opt-in, backward-compatible, pure NumPy, deterministic.

B1 -- holographic_hopfield.py: modern continuous Hopfield cleanup (Ramsauer 2020 / Krotov &
Hopfield 2016 / Demircigil 2017). dense_cleanup(q, codebook, beta, steps) = z<-V^T softmax(beta Vq)
iterated; HopfieldCleanup.fit/cleanup/denoise. KEPT NEGATIVE: ties one-shot NN on IDENTITY (NN
already optimal; at beta->inf it REPRODUCES the hard decision exactly -> backward compatible).
The real win is CONTINUOUS-VECTOR DENOISING: a recovered vector at cosine 0.45 cleans to ~1.0
(measured, mean over trials = 1.000 across dim/noise). A single high-noise draw can occasionally
miss; the mean is what we ship (test averages over trials).

B10 -- generate() in holographic_hopfield.py: iterate the cleanup from PURE NOISE with annealed
beta-up / noise-down = a tiny holographic diffusion. Measured: nearest-pattern cosine 0.5->1.0 in
~8-12 steps; generation and denoising are the SAME operation in different regimes. KEPT NEGATIVE:
over a BARE codebook this returns stored atoms (degenerate sampler) -- the interesting regime is a
COMPOSED/continuous manifold.

B8 -- holographic_splat.py: a splat scene IS a superposition (bundle). splat_fit (matching pursuit
with isotropic Gaussian atoms) / splat_render / splat_denoise. MEASURED on a real (log-return,
log-volume) SOL density: ~20 superposed Gaussians -> ~31 dB at ~3.5% of pixels; fitting few splats
to NOISY data denoises it (+~5 dB to clean, no capacity for noise). BRIDGE pinned in test: the RBF
ScalarEncoder's similarity profile is a Gaussian bump (peaks at the encoded value) = Gaussian
splatting in the hypervector domain. SCOPE/kept-negative: isotropic matching pursuit only;
anisotropic covariances + gradient refinement (full 3DGS) deliberately out of scope.

B7 -- holographic_denoise.py: denoising as MANIFOLD PROJECTION (Milanfar: a denoiser is a map of
the signal manifold; consolidation IS that map) + the Plug-and-Play/RED loop (Venkatakrishnan 2013;
Romano-Elad-Milanfar 2017). fit_manifold (SVD = consolidation) / manifold_denoise (project) /
codebook_denoise (re-exports dense_cleanup) / pnp_restore (data-fidelity <-> denoise, any denoiser).
MEASURED on real SOL price windows: projection denoising WINS as noise grows (+3.85 dB at sigma=0.8)
but HURTS at low noise (-1.4 dB over-smoothing -- the Donoho/Milanfar threshold-selection problem,
KEPT NEGATIVE) and DESTROYS random no-manifold data (-5 dB, honest control, pinned in test). pnp
inpainting test: restoration beats the masked measurement.

Tests: test_holographic_hopfield.py (3), test_holographic_splat.py (3), test_holographic_denoise.py
(3) = +9 (569 -> 578). Tour block added (denoise+splats line). Wired as standalone opt-in modules;
no change to bind/value/decide/cleanup-defaults -> creature tie-sensitive path untouched (canary not
required). STILL QUEUED (each behind its measurement bar, next passes): B2 sparse block codes +
scaled resonator; B3 SPRT streaming recall; B4 propagator binding (needs a learnable-dynamics
signal); B5 rate-distortion ANS save level (bit-exact coder is the fiddly part); B6 Tero flow /
fragment assembly; B9 non-local-means via content-addressable recall (bar: beat manifold projection
on textured/non-low-rank signals).

## B9 -- NON-LOCAL-MEANS DENOISING VIA CONTENT-ADDRESSABLE RECALL (shipped)

Built B9 from the queue. "Find the patches that look like this one and average them" (Buades-Coll-
Morel NLM 2005; BM3D Dabov 2007) IS content-addressable recall -- so it runs on holostuff's own
index. Added HoloForest.recall_k(query, k, beam) -> (indices, cosines) ranked over the same unioned
candidate set recall() uses (stays SUB-LINEAR; default recall() untouched, byte-identical). Added
nlm_denoise(patches, k, h, use_forest) in holographic_denoise.py: per patch, recall its k nearest
(forest sub-linear, or exact cosine fallback for small sets / determinism), softmax(cosine/h) weight,
average -- cancels iid noise across near-duplicates (~1/sqrt(k)).

MEASURED (real SOL motif-windows, M motifs x R=8 repeats + noise): NLM-via-forest 11.7 dB vs rank-8
projection 7.3 dB vs raw 4.6 dB -- and the sub-linear forest path (11.67) matches exact kNN (11.77).
COMPLEMENTARITY confirmed and pinned as a KEPT NEGATIVE test: on low-rank-but-NOT-self-similar data
(every patch unique), projection WINS (2.8 dB) and NLM has nothing to average (0.5 dB). So B7
(manifold projection) and B9 (NLM) cover DIFFERENT structure: low-rank-not-similar vs
self-similar-not-low-rank. Tests: +3 in test_holographic_denoise.py (nlm beats projection on
self-similar; projection beats nlm without self-similarity; recall_k finds near-duplicates) ->
578 -> 581. Tour line added. recall_k is a pure addition; default forest recall path unchanged.

REMAINING QUEUE (each behind its bar): B3 SPRT streaming recall (builds on RecallNull; clean
optimality bar); B2 sparse block codes + scaled resonator; B5 rate-distortion ANS save level (fiddly
bit-exact coder); B4 propagator binding (mechanism real, prediction an honest near-negative on
markets -- ship as content-addressable-trajectory capability); B6 Tero flow / fragment assembly.

## B3 -- SPRT STREAMING RECALL (shipped): sample-optimal sequential detection

Built B3 from the queue. RecallNull turns ONE recall into a calibrated false-alarm probability;
SPRTRecall (holographic_honesty.py) turns a STREAM of cues for the same hypothesis into a Wald
sequential test: accumulate the per-cue log-LR log(p(score|match)/p(score|null)) and decide the
moment it crosses a Wald boundary A=log((1-beta)/alpha) / B=log(beta/(1-alpha)). RecallNull's noise
floor IS p(score|null); match density is fit from genuine noisy-target recalls. API: SPRTRecall(
null_scores, match_scores, alpha, beta).reset()/.update(score)->MATCH|REJECT|CONTINUE/.decide(stream,
cap)->(decision, n). Gaussian-fit densities.

MEASURED on real recall scores (the optimality bar): across the overlapping regime SPRT reaches a
target (alpha,beta) error pair in ~HALF the samples of the best fixed-N rule -- e.g. avg 2.8 cues vs
fixed-N 6 at ~2% error (also 1.7 vs 3, and 4.8 vs 9 at heavier overlap). Wald optimality confirmed.
KEPT NEGATIVE / boundary: when the cue carries NO per-sample information (signal fully swamped, match
and null distributions identical -- e.g. sigma=9 noise), neither SPRT nor fixed-N can decide and SPRT
just hits the cap; streaming only helps when each cue carries SOME evidence. Tests: +2 in
test_holographic_honesty.py (clear streams decide MATCH/REJECT; SPRT uses fewer samples than fixed-N
at matched error) -> 581 -> 583. Tour line added (sequential recall). Pure addition; nothing else
touched.

REMAINING QUEUE: B2 sparse block codes + scaled resonator; B5 rate-distortion ANS save level (fiddly
bit-exact coder); B4 propagator binding (mechanism real, prediction honest near-negative on markets
-- ship as content-addressable-trajectory capability); B6 Tero flow / fragment assembly.

## B4 -- PROPAGATOR BINDING (shipped): dynamics as an algebra of binds

Built B4 from the queue. holographic_dynamics.py / Propagator: learn a fixed bind operator U so
state(t+1) ~ bind(U, state(t)). In HRR's Fourier domain bind is elementwise multiply, so the learned
operator is a per-frequency least-squares transfer H[k]=sum X1 conj(X0)/sum|X0|^2 (Koopman-in-Fourier
/ DMD / the same FFT-on-a-torus Stam and Puckette use). step(state)=bind(U,state) LITERALLY (pinned
in a test) -> prediction is one bind; recall_at(state,k) applies a Wiener-regularised inverse operator
k times -> the trajectory is CONTENT-ADDRESSABLE.

MEASURED (honest, complete picture):
  * POSITIVE CONTROL (dynamics that ARE a bind): next-state prediction cosine 0.997 vs persistence
    0.528 -- when dynamics are bind-shaped the propagator recovers the operator and predicts full
    states. This is the method's honest SCOPE.
  * DURABLE WIN: content-addressable round-trip (forward k / back k) cosine ~0.9995 -- past states
    recoverable regardless of predictability.
  * KEPT NEGATIVE (real SOL returns, scalar next-return): propagator RMSE 0.0088 vs mean 0.0063 --
    ties/loses to mean; near-efficient-market returns have no linear structure. Also a structural
    kept-negative: the bind operator is a CIRCULAR convolution, so as a next-VALUE predictor on a
    shifted signal it suffers wrap-around and an unconstrained full operator would do better -- the
    bind framing buys the exact content-addressable round-trip, not best-in-class scalar prediction.
  * Inverse uses conj(H)/(|H|^2+eps) (Wiener-regularised) -- the Plate tradeoff made explicit (exact
    deconvolution is precise but amplifies near-null frequencies; regularised is robust).

Tests: test_holographic_dynamics.py (+4: step IS a bind; predicts bind-shaped dynamics; trajectory
content-addressable; rollout shape) -> 583 -> 587. Tour line added. Pure new module, nothing else
touched.

REMAINING QUEUE: B2 sparse block codes + scaled resonator; B5 rate-distortion ANS save level (fiddly
bit-exact coder); B6 Tero flow / fragment assembly. (B1,B3,B4,B7,B8,B9,B10 shipped.)

## MÖBIUS / NON-ORIENTABLE TOPOLOGY (shipped): matching representation topology to data

Prompted by a question -- circles, sign flips, and noise recur throughout the engine; would a Mobius
strip define some things better than a circle? Searched the literature (neural population activity
traces a manifold whose TOPOLOGY MATCHES the variable: ring for head-direction, torus for grid cells,
Klein bottle / Mobius for ORIENTATION). holostuff binds by circular convolution, so its native shape
is the circle/torus -- right for a directed angle, WRONG for two cases:

  * AXIAL data (theta == theta+pi: orientation, director/nematic fields, phase-mod-pi). On a circle
    theta and theta+pi are ANTIPODAL (sim -1) though they are identical. Correct base = projective
    line RP^1 = the Mobius double-cover's base. Fix = double-angle map theta -> 2*theta.
  * SIGN-FLIPPING data f(t+T) = -f(t) (antiperiodic, a Mobius double-cover in time): all energy in
    ODD harmonics; the periodic/circular basis is blind to it.

Built holographic_mobius.py: AxialEncoder (double-angle phasor encoder, theta and theta+pi map to the
SAME hypervector), antiperiodic_fraction / antiperiodic_split (diagnose + extract the sign-flipping
component).

MEASURED:
  * axial recovery error (values reported as theta OR theta+pi at random): naive circle 0.470 rad vs
    Mobius double-angle 0.002 rad; sim(theta,theta+pi) naive -0.22 vs Mobius +1.00.
  * sign-flipping signal: ~100% of energy antiperiodic (periodic component ~1e-14).

KEPT NEGATIVE / SCOPE: use ONLY for genuinely axial or sign-flipping data -- on DIRECTED data the
circle is correct and the double-angle encoder WRONGLY merges theta with theta+pi (it discards the
half-turn on purpose). Also NAMES an old kept negative: binary quantization maps to +-1, itself a
Z2/antipodal (Mobius) identification -- exactly why it distorted circular geometry, and exactly why it
would be right for axial/sign-flip data. Same lesson: topology must match the data.

Tests: test_holographic_mobius.py (+6: axial identifies theta==theta+pi; naive circle disagrees;
recovers orientation despite pi-flips; merges half-turn on purpose [scope]; antiperiodic fraction
detects sign flip; split reconstructs) -> 587 -> 593. Tour block added. Pure new module.

## STRUCTURE-FIRST COMPUTATION + REORGANIZATION (measured, no module): the fruit-fly-connectome parallel

Prompted by the embodied fly-connectome result (Shiu et al. Nature 2024 wired a leaky-integrate-and-
fire model STRAIGHT FROM the FlyWire connectome, no training, ~95% sensorimotor accuracy; Eon 2026
drove a physics fly from the wiring alone). Load-bearing claim across the coverage: STRUCTURE CARRIES
COMPUTATION (biological wiring beat random graphs / standard nets). That is holostuff's thesis. Backed
it with proofs on real Brown-corpus data:
  * PROOF 1 (structure = computation, no training): bundled per-class prototypes (just bind+bundle, no
    gradients) classify held-out documents at 0.76 vs 0.17 chance (6 classes). The structure IS the
    classifier -- the engine's analog of wiring driving behavior.
  * PROOF 2 (learning = structural REORGANIZATION, honest): the RAW document cloud's effective rank
    GROWS with samples (8.9 -> 45) -- accumulation is NOT learning. But the TASK structure is low-rank:
    consolidating the prototypes (SVD, = our consolidation faculty) to rank 6 preserves accuracy
    exactly (0.76), rank 4 keeps most (0.70), rank 2 breaks it (0.43). Learning is the reorganization
    onto the low-rank task subspace, separating it from high-rank sample noise -- consolidation is the
    holostuff move that mirrors a connectome being a specific low-complexity wiring that holds behavior.
Written up with the Mobius findings in MOBIUS_AND_STRUCTURE.md (outputs).

REMAINING QUEUE: B2 sparse block codes + scaled resonator; B5 rate-distortion ANS save level; B6 Tero
flow / fragment assembly. (B1,B3,B4,B7,B8,B9,B10 shipped; Mobius shipped.)

## HOLOGRAPHIC MACHINE (shipped): inception -- a program encoded as a vector, executed by the substrate

Prompted by an inception question: a hard drive has physical structure, data in that structure, and --
executed -- an OS, a VM, an OS inside the VM. holostuff had the lower rungs (vector = platter,
derived_atom = format, role-filler + nested composition = file system); the missing rung is an OS that
EXECUTES. Built holographic_machine.py / HoloMachine: a program is encoded as ONE hypervector and run by
the engine's own bind/bundle/cleanup. Instructions and data share one vector space (von Neumann,
holographically). "Format the drive" = fix a seed (lays down roles OP/ARG/SLOT, opcode atoms, data atoms,
POS(i) addresses, all via derived_atom). Instruction set: LOAD/BIND/BUNDLE/PERMUTE/HALT.
  instruction = bundle(bind(OP,opcode), bind(ARG,operand)); program = bundle_i bind(POS(i), instruction_i).
run() unbinds each address, CLEANS opcode+operand against codebooks (wide-margin, robust to crosstalk),
dispatches. Operands are cleaned to exact atoms before use, so ACC is EXACT despite noisy reads.

MEASURED:
  * Correctness: LOAD a; BIND b; BUNDLE c -> ACC == bundle(bind(a,b),c) cosine 1.0000; trace exact.
  * DRIVE SIZE (capacity cliff, KEPT NEGATIVE): instruction-decode ~100% up to a length that scales with
    dim -- ~32 instructions reliable at dim 1024, ~128 at dim 4096 -- then bundling crosstalk overwhelms
    cleanup. Capacity is finite; the cliff is the honest HRR wall.
  * INCEPTION DEPTH (the law): a program nested as the ONLY file at each level survives 8+ levels deep
    (a pure unitary bind chain barely degrades); a program buried among OTHER files on each disk corrupts
    after ~3-4 levels. Depth is set by clutter per level -- nest as deep as you like if each level is
    clean, only a few levels on a busy disk. Both scale with dim.

The stack: platter (vector) -> format (derived_atom) -> file system (bind/bundle/compose_nested) -> OS
(HoloMachine.run) -> VM-in-OS (nest a program inside a disk inside a disk). Written up in
HOLOGRAPHIC_INCEPTION.md (outputs).

Tests: test_holographic_machine.py (+6: executes exactly; HALT stops; PERMUTE; 32-instr decodes fully;
clean nesting deep; busy-disk depth floor [kept negative]) -> 593 -> 599. Tour block added. Pure new module.

REMAINING QUEUE: B2 sparse block codes + scaled resonator; B5 rate-distortion ANS save level; B6 Tero
flow / fragment assembly. (B1,B3,B4,B7,B8,B9,B10 shipped; Mobius shipped; HoloMachine shipped.)

## HOLOGRAPHIC FUNCTIONS + CALL (shipped): functions embedded and executed in the holographic space

Prompted by an inception follow-up: can we embed and execute FUNCTIONS within the holographic space
(not as Python files)? Folders/partitions to reduce confusion? What does it enable, including things
we didn't plan for? Measured all of it on the substrate, then shipped the load-bearing piece.

MEASURED (real numbers):
  * A function you DEMONSTRATE instead of write: a key->value (or input->output) mapping stored as ONE
    vector M=bundle(bind(k_i,v_i)); apply f(k)=cleanup(unbind(M,k)). 100% to ~120 pairs at dim 4096,
    cliff at ~240 (87%). This is HolographicMemory used as a learned, content-addressable function --
    no code written, only examples given.
  * Functions in a holographic LIBRARY, called by name: define ACC->ACC sub-programs, bundle them into
    ONE library vector, CALL by name -> the body is extracted (unbind) and run on the current ACC.
    'LOAD a; CALL tag_b; CALL shift' == permute(bind(a,b)) cosine 1.0. Functions compose like data.
  * FOLDERS/PARTITIONS reduce confusion: at 256 items a flat HolographicMemory recalls 86%, a 16-folder
    PartitionedMemory recalls 100% -- partitioning cuts crosstalk per query (folders already exist as a
    primitive; this just names/measures the benefit).
  * DIDN'T PLAN FOR: (a) behavioral content-addressing -- retrieve a function by an EXAMPLE of what it
    does (a->permute(a) retrieves 'shift'); (b) function arithmetic -- bundle(f1,f2) is a function that
    carries BOTH answers (0.18/0.18 symmetric), i.e. you can average programs like vectors.

SHIPPED: holographic_machine.py extended -- OPCODES gains CALL; HoloMachine.define(name, program) embeds
a named ACC->ACC function into a single library vector; run() gains init_acc + CALL dispatch (extract by
name, run on current ACC, recursion-guarded). Backward compatible (init_acc defaults None; non-CALL
programs unchanged). The other capabilities use existing primitives (HolographicMemory = demonstrated
function; PartitionedMemory = folders), so they were measured/named, not re-implemented.

WHY IT MATTERS (the multiplier): code and data now share one algebra, so EVERY engine faculty applies to
programs too -- consolidate (compress a program), denoise (clean a corrupted program), factorize a
program into parts, index programs for content-addressable retrieval, even generate new programs with the
B10 sampler. The honest boundary: this is not a fast general CPU (Python is faster); its edge is
deterministic, inspectable, composable, content-addressable code-as-data. Written up in
HOLOGRAPHIC_FUNCTIONS.md (outputs).

Tests: test_holographic_machine.py (+4: CALL runs a library function; CALL composes; library is one
vector; run backward-compatible) -> 599 -> 603. Tour line added.

REMAINING QUEUE: B2 sparse block codes; B5 rate-distortion ANS; B6 Tero flow. Also teed up: adaptive-rank
denoising (cash B7's low-noise kept negative). (B1,B3,B4,B7,B8,B9,B10 + Mobius + HoloMachine + CALL shipped.)

## B5 -- RATE-DISTORTION GEOMETRY-PRESERVING CODE (shipped): KLT -> quantize -> rANS

Built B5 from the queue. holographic_ratedistortion.py: spend the minimum bits that preserve the
DECISION GEOMETRY (cosines), not raw values, by chaining three pieces the engine half-owned -- the
classic transform-coding pipeline:
  consolidate (KLT/SVD)  ->  uniform scalar quantize the coefficients  ->  rANS entropy code
Consolidation IS the KLT (decorrelates), so one quantization step on the coefficients is near
rate-distortion-optimal and the entropy coder spends bits proportional to each component's entropy
(water-filling for free). rANS (Duda's ANS) codes to the Shannon limit.

THE FIDDLY PART, DONE FIRST (the gate): a pure-NumPy bit-exact static rANS coder. Verified 40/40 random
streams round-trip EXACTLY (the determinism rule depends on it), coding within ~0.3% of entropy vs
int8's flat 8 bits/sym. Only after the gate passed was anything wired to it.

MEASURED (honest, complete):
  * WIN on genuinely low-rank engine state (bundled sense states, energy fully at rank 16): matches
    int8 fidelity (cosine 0.99998) at ~191 bits/vec vs int8's 2048 -- ~11x smaller than int8, ~43x vs
    float32. At target 0.9999, ~7x. File format (save_rd/load_rd) measured 6.2x smaller than int8.
  * KEPT NEGATIVE: on full-rank data (SOL RETURNS, ~rank 64/64) no subspace to exploit -> rd loses;
    it auto-falls-back to int8 in the save path so it is never larger. Like B7, helps only where real
    low-rank structure exists.
  * METHODOLOGICAL NEGATIVE: participation-ratio "effective rank" misleads -- smooth SOL PRICE windows
    looked rank ~4 but a heavy spectral tail needs rank ~40 for high cosine (rank-8 only reaches 0.93).
    Judge low-rank-ness by energy concentration / truncation cosine, not the participation ratio.

WIRED: holographic_core.save gains quant="rd" (beside int8/auto): for low-rank 2D float arrays it stores
the packed rd code (basis f32 + rANS bytes) and falls back to int8 where rd wouldn't beat it -- so it is
always at least as small as int8 and never breaks a save (SelfOrganizingMind round-trips, classifications
preserved). load reconstructs. Standalone save_rd/load_rd (.rdc) provided too. Bit-exact rANS keeps the
determinism guarantee.

Tests: test_holographic_ratedistortion.py (+5: rANS bit-exact; rANS ~entropy; geometry code preserves
cosines; beats int8 on low-rank; kept-negative full-rank) + test_core_persistence.py (+2: rd save level
safe/non-breaking; rd activates+shrinks low-rank) -> 603 -> 610. Tour line added.

REMAINING QUEUE: B2 sparse block codes + scaled resonator (the +5-orders-of-magnitude capacity lever,
freshly grounded -- Hersche/Langenegger); B6 Tero flow / fragment assembly. Also teed up: adaptive-rank
denoising (cash B7's low-noise kept negative). (B1,B3,B4,B5,B7,B8,B9,B10 + Mobius + HoloMachine + CALL shipped.)

## HOLOGRAPHIC KAN (shipped): a deterministic Kolmogorov-Arnold readout on holostuff encoders

A panel/user question -- KANs (Kolmogorov-Arnold Networks: a function as a SUM of learnable univariate
splines, F(x)=sum_j psi_j(x_j)) sounded related to our encoder/bundle work. Checked the literature (Liu
et al. 2024; learnable univariate functions on edges, nodes just sum; B-spline basis; adaptive grid;
interpretable; slower than MLP, curse-of-dimensionality on splines). It IS related, and we built the
connection deterministically. Two threads, one module holographic_kan.py:

  * THREAD 1 -- AdaptiveScalarEncoder: a ScalarEncoder whose grid ADAPTS to the data via a monotonic
    empirical-CDF warp -- KAN's "move the spline knots to where the data is", fit once and frozen
    (stays deterministic). basis(x) = similarities of encode(warp(x)) to grid anchors = the spline
    basis (the RBF encoder's similarity profile is a Gaussian BUMP, exactly a B-spline-like basis).
  * THREAD 2 -- HolographicKAN: a single-layer KAN. Each feature -> its encoder's basis activations ->
    psi_j(x_j)=a_j . basis_j(x_j); prediction = SUM over features (the Kolmogorov-Arnold inner sum =
    holostuff's bundle). Output is LINEAR in the coefficients a, so they are fit by ridge LEAST SQUARES
    -- NO backprop. psi_j are recoverable/plottable (KAN's interpretability), all deterministic.

So: a KAN whose splines are holostuff encoder bumps and whose training is a linear solve -- KAN's idea
in holostuff's idiom (deterministic, interpretable, structure-first).

MEASURED:
  * additive target f=sin(2pi x1)+4(x2-.5)^2: test R^2 0.999; recovered psi_1 vs sin corr 1.000,
    psi_2 vs quadratic corr 1.000 (interpretable parts recovered); linear readout only R^2 0.54.
  * adaptive grid beats uniform on a SKEWED feature (R^2 0.41 vs 0.25 -- resolution follows density);
    on UNIFORM data the warp is ~identity, a kept tie (no help, no harm, costs a stored CDF).
  * KEPT NEGATIVE: the single-layer additive form cannot represent feature INTERACTIONS (x1*x2 -> R^2
    ~0), while the additive control x1^2+x2^2 -> R^2 0.997. Boundary shown; needs a 2nd layer or
    explicit interaction features.

Relation kept honest: cousins not twins -- KAN learns its univariate functions by backprop and is a
neural-net approximator; ours fixes the encoder, learns only the linear readout, and is structure-first.
The shared heart (sum of univariate basis-bump functions = bundle of per-feature encodings) is real.

Tests: test_holographic_kan.py (+6: additive fit+recovery; beats linear; adaptive>uniform on skewed;
kept-negative interactions; warp maps skewed->uniform; warp identity before fit) -> 610 -> 616. Tour
line added. Pure new module (encoders + cosine + least squares; no kernel/compute-path change).

REMAINING B-QUEUE (on hold per user): B2 sparse block codes + scaled resonator; B6 Tero flow / fragment
assembly. Teed up: adaptive-rank denoising (B7 low-noise negative). (B1,B3,B4,B5,B7,B8,B9,B10 + Mobius +
HoloMachine + CALL + Holographic-KAN shipped.)

## GENERATIVE COMPRESSOR, part 1: the recipe-store (shipped)

Follows the "proven structure has no noise" debate. A structure BUILT by a deterministic proof carries no
noise, so it serialises to its generator losslessly: store the recipe, not the expanded vectors, and
replaying reproduces it BIT-EXACT. This is the easy, exact half -- when we are the builder we already hold
the proof, so there is no search and no residual.

holographic_recipe.py -- StructureRecipe: a tiny replayable build-graph. Ops atom/bind/bundle/permute/
normalize each produce one result from a seed + earlier results; you build THROUGH the recipe so you get
both the vectors and the generator. `raw` stores a literal vector verbatim -- the escape hatch for
non-constructed data (the measured/lossy regime; stored float32). save/load is JSON (readable) with raw
payloads as binary float32.

MEASURED: a 2000x512 derived codebook (~4.1 MB) -> ~68 KB recipe (~60x), replay max abs error 0.0
(bit-exact). Deep nested structure recovered exactly at depth 8 -- no capacity cliff, because the recipe
names its leaves and replays rather than reading them out of a bounded superposition. KEPT NEGATIVE: the
`raw` escape hatch -- non-constructed/random data has no short recipe, so ratio ~0.99x (no win, no harm);
the compression is exactly the constructed fraction (half/half -> ~2x). Constructed ops replay bit-exact;
raw payloads are float32 (the regime where a residual coder belongs).

Tests: test_holographic_recipe.py (+5). 616 -> 621.

## GENERATIVE COMPRESSOR, part 2: the decompose search (shipped)

The hard half: find the construction behind FOREIGN data. holographic_symbolic.py -- MDL-gated symbolic
regression. Rather than enumerate EML/operator trees (combinatorial; EML eval is expensive -- the EML
debate's kept negative), search a dictionary of elementary basis functions (SINDy-style; Brunton-Proctor-
Kutz 2016) by deterministic greedy forward selection, and choose the model by Minimum Description Length:
total bits = model bits (#terms x [index + coefficient]) + residual bits (Gaussian coding cost). MDL is
the gate: a term is kept only if it shortens the code, so the law is the shortest program explaining the
data -- the parsimony that makes extrapolation valid. On noise it adds nothing (honest refusal).

MEASURED: recovered 2*sin(1.5x)+0.5x from noisy data (2 terms of 17), seed ~70x smaller than the data,
extrapolation RMS 0.016. THE MDL GATE CURES OVERFITTING (solves the generative-compression debate's kept
negative): MDL extrapolation 0.016 vs un-gated max-fit 7.6e5 (explodes out of range). On pure noise MDL
keeps 0 terms -> just the mean -> refuses to manufacture a law (no free lunch, enforced). The recovered
Formula is a generative SEED -- the measured-regime analogue of a StructureRecipe: build 2 finds the
recipe, build 1 stores it, the residual is what a B5 rate-distortion coder takes.

KEPT NEGATIVES: the dictionary bounds what is discoverable (a law outside the basis, or a rate off the
frequency grid, is not found); the MDL coefficient-cost is a knob not a law; this is the tractable proxy
for the full EML-tree search (the uniform single-operator tree remains the theoretical, far larger space).

Tests: test_holographic_symbolic.py (+5). 621 -> 626.

THE TWO HALVES TOGETHER = the generative compressor the panel mapped: decompose (build 2) -> seed ->
generate/store (build 1) -> residual coder (B5), with the MDL/RecallNull parsimony gate keeping
extrapolation honest. CONSTRUCTED data: exact, no search (build 1). MEASURED data: search + residual
(build 2 + B5). Two regimes, two tools, one pipeline.

REMAINING B-QUEUE (still on hold per user): B2 sparse block codes + scaled resonator; B6 Tero flow /
fragment assembly. Also teed up: adaptive-rank denoising (B7 low-noise negative).

## RECIPE-STORE macro op + one-call decompose pipeline (shipped)

Two follow-ups closing loose ends before resuming the B-list.

(1) MACRO/LOOP OP for the recipe-store. The straight-line recipe stored N explicit ops for a regular
structure (a 2000-atom codebook -> ~60x). Added a `repeat(count, template)` op: a parameterised iteration
captured as ONE op, with a declarative template (local refs; {i} substituted in atom names; permute shift
can be the loop index "i"); the iteration's output is its last template result, so repeat emits `count`
results. `atom_range(prefix, count, unitary)` is sugar over it. Refactored handles to absolute RESULT
indices (a counter `_n_results`) since a macro produces many results per op. MEASURED: the 2000-atom
codebook now collapses to a 96-byte recipe -> ~42,000x (was 60x), replay BIT-EXACT, save/load round-trips.
A positional sequence (bundle_i permute(item_i, i)) is a 201-byte recipe matching the manual build exactly.
The win: the recipe now compresses REGULAR structure to its rule, not just per-vector.

(2) ONE-CALL DECOMPOSE PIPELINE. Gave the symbolic `Formula` the SAME seed interface as StructureRecipe
(to_recipe/from_recipe/save/load/recipe_bytes/compression_ratio) -- a Formula IS the measured-regime
recipe (it generates a scalar signal where StructureRecipe generates vectors). Added `compress_signal(x,y,
path=None)`: decompose -> seed in one call. MEASURED: foreign data -> 135-byte seed file that reloads and
regenerates (in-window RMS 0.005) + extrapolates (RMS 0.016); to_recipe round-trips exact; residual (B5's
job) reported. This closes the pipeline end-to-end and is a concrete step toward the integration review's
"unify the seed/structure representation" recommendation (both regimes now share one seed interface).

Tests: +3 recipe (atom_range ratio+bit-exact; repeat template vs manual; macro save/load), +2 symbolic
(Formula save/load roundtrip; compress_signal end-to-end). 626 -> 631. Additive (no compute-path change).

NOW RESUMING THE B-LIST. Standing queue: B2 (sparse block codes + scaled resonator), B6 (Tero flow /
fragment assembly), plus the integration-review additions B7 (typed holographic structure: recipe=EML-tree
=program=scene), B8 (denoised structure decoding -- push the inception cliff deeper), B9 (manifold-aware
decompose). Also teed up: adaptive-rank denoising (B7-original low-noise negative).

## B2 -- SPARSE BLOCK CODES + SCALED RESONATOR (shipped)

The long-queued capacity lever, and -- per the blend discussion -- the deconfounder a superposition search
needs. holographic_sbc.py. An SBC atom is B integers (one active position per block); dense form is the
one-hot expansion, D = B*L. bind = (a+b) mod L per block (block-local circular convolution of one-hots) --
EXACT, lossless, where dense circular-convolution binding accumulates crosstalk. The resonator factors a
product into one atom per codebook by annealed alternating projection (soft superposition estimates;
deterministic annealing beta 0.5->12 to explore-then-commit; random init to break the symmetric trap;
restarts), and verifies itself with a hard CONFIDENCE check: do the recovered factors RECONSTRUCT the
product? validated=True <=> correct.

WHY annealing+restarts: a fixed-temperature softmax collapses to spurious fixed points (measured: ~0.13
accuracy); signed-linear cleanup stalls too; deterministic annealing + reconstruction-validated restarts
fixed it.

MEASURED (head-to-head at fixed D=256, F=3): SBC beats dense at every alphabet with signal -- N=10 1.00 vs
0.90, N=25 0.25 vs 0.15, N=50 0.05 vs 0.00 (consistent, modest edge). The confidence check tracks
correctness EXACTLY (validated<=>correct, precision ~1.0); coverage drops with alphabet so it verifies or
abstains rather than guessing. KEPT NEGATIVES: absolute capacity modest (both collapse by N~100; more
blocks/restarts raise both); SBC is a PARALLEL representation requiring sparse-block-coded data (beside the
dense kernel, not inside it); exact reconstruction-validation makes it abstain under product corruption
(honest but conservative).

Tests: test_holographic_sbc.py (+5: exact block bind/unbind; clean factorization; confidence=>correctness;
high coverage at N=10; abstains on corruption). 631 -> 636. New standalone module, no compute-path change.

THREAD TO CIRCLE BACK TO: the resonator's verified-factorization is the deconfounder for the underexploited
"superposition-parallel candidate search" in the decompose pipeline (blend candidate sub-expressions, let
the resonator factor which are present, verify by reconstruction). That is the next step on the blend thread.

B-LIST STANDING: B2 DONE. Remaining: B6 (Tero flow / fragment assembly), B7 (typed holographic structure),
B8 (denoised structure decoding), B9 (manifold-aware decompose); teed up: adaptive-rank denoising.

## STRUCTURAL DECOMPOSE: the verified resonator as the inverse of build-1 (shipped, blend thread)

Picking the blend thread back up with B2's resonator now in hand. The honest scoping: the resonator's
unique power is factoring a BOUND PRODUCT of unknowns -- and a bound product is DISSIMILAR to its factors,
so you cannot read them off naively (measured: per-factor readout of a product is chance). That is exactly
where the superposition-parallel + deconfounded + VERIFIED search earns its keep -- and it applies to
compositional STRUCTURE (scenes, recipes, trees), not flat numeric sums (where greedy/matching-pursuit
already deconfounds, so the resonator adds nothing there -- kept scope).

holographic_sbc.py gains `decompose_structure(composed, codebooks, L)` -> {picks, factors, verified,
present}: recover the generating recipe of a composed structure via the verified resonator -- the
structural INVERSE of build-1's recipe-store (build 1: recipe->structure forward; this: structure->recipe
inverse). `sbc_identity` lets a factor be detected ABSENT, so you can blend candidates INCLUDING an
'absent' option and factor which are PRESENT -- the literal "blend candidate sub-expressions, factor which
are present" idea.

MEASURED: naive per-factor readout of a product (2,5,8) gives (2,0,0) = chance; the resonator recovers
(2,5,8) verified. Presence detection: a structure missing its third factor -> present=[True,True,False],
verified. Recovered recipe reconstructs the structure exactly. Superposition-parallel: resolves 1 of N^F
combinations (1000) without enumerating them. KEPT NEGATIVES: applies to compositional/product structure
with known codebooks, NOT to numeric-signal decompose (greedy already deconfounds sums); capacity is the
resonator's (modest, from B2); aliasing -- if two factor-combos reconstruct the same product, verification
cannot distinguish them (rare).

This closes the loop the integration review wanted: build-1 (store known structure as recipe, forward) and
this (decompose foreign structure to recipe, inverse), with build-2 the numeric-signal analogue. Three
decompose regimes now: numeric signal -> law (build 2, greedy/MDL); composed structure -> recipe (this,
verified resonator); known structure -> recipe (build 1, no search).

Tests: +3 in test_holographic_sbc.py (recover+verify; detect absent factor; naive-fails-resonator-succeeds).
636 -> 639. Additive (extends holographic_sbc; no compute-path change).

## DECOMPOSE: multiplicative mode via the log transform (shipped)

From the prime-factorization discussion: the deep takeaway was that MULTIPLICATIVE structure becomes
ADDITIVE in the right basis (a logarithm to a prime basis turns x into +). The transferable nugget: our
greedy/MDL decompose only finds ADDITIVE (sum-of-terms) laws, but a multiplicative law a*x^p*exp(cx)*...
becomes additive under log y. So holographic_symbolic.py gains a multiplicative path:
  * `_eval_atom` gains a `log` kind (log|x|); `log_dictionary()` = {log|x|, x, x^2, x^3} -- the log-images
    of power and exp-of-polynomial factors.
  * `Formula` gains `log_space`: generate() exponentiates, so the additive log-fit becomes a PRODUCT law.
  * `symbolic_regress(..., multiplicative=True)` fits log(y) (requires y>0); term selection runs in log
    space, resid_rms is reported in the ORIGINAL space (comparable across modes).
  * `compress_signal(..., mode='additive'|'multiplicative'|'auto')`. auto switches to multiplicative only
    if it is COMPETITIVE in-sample AND generalizes better on a held-out tail.

MEASURED: recovered exp(0.707 + 1.51 log x + 0.29 x) = 2*x^1.5*exp(0.3x) -- the exact product law, extrap
relRMS 0.017, which the flat additive basis only approximates. auto-selection over 6 seeds: additive data
-> additive 6/6 (never a false positive), multiplicative data -> multiplicative 4/6 (else falls back to
additive, which still fits). KEPT NEGATIVES: needs y>0 (and x>0 for the log|x| power-law basis);
multiplicative mode FAILS on additive laws (log of a sum is not a sum -- measured resid 0.378 vs 0.031);
auto-selection between the two FAMILIES is genuinely hard when both fit in-sample, so the selector is a
conservative heuristic (never false-positive; ~4/6 true-positive), erring toward additive.

Tests: +5 in test_holographic_symbolic.py (recovers product law; requires y>0; log_space roundtrip;
auto never false-positives on additive; multiplicative beats additive extrapolation). 639 -> 644.
Additive (extends the decompose; no kernel/compute-path change).

## B7 KEYSTONE: one typed holographic structure (shipped)

The integration review's headline was "substrate-integrated, orchestration-siloed": the engine's four
"structure" types -- a build RECIPE, an assembled PROGRAM (HoloMachine), an EML/expression TREE, and a
composed nested SCENE (UnifiedMind.compose_nested) -- are not four things. They are ONE directed graph of
the same primitives replayed to a vector, and StructureRecipe already IS that graph. holographic_typed.py
makes the unification concrete and MEASURED with adapters that reproduce each source bit-exactly:
  * program_to_recipe(machine, program)   == HoloMachine.assemble   (cosine 1.000000, max|diff| 0.0)
  * tree_to_recipe(dim, seed, tree)        == encode_tree (direct)   (cosine 1.000000, max|diff| 0.0)
  * nested_scene_to_recipe(mind, groups)   == mind.compose_nested    (cosine 1.000000, max|diff| <1e-9)
The union alphabet across all three is just {atom, raw, bind, bundle, superpose} -- one small primitive
set, not five class hierarchies. A new `superpose` op (un-normalized sum) was added to StructureRecipe so
it can reproduce compose_nested's raw np.sum (the renormalizing `bundle` could not); backward-compatible.

WIRED INTO UnifiedMind (the de-siloing): typed_structure() -> a fresh recipe at the mind's dim/seed;
realize(recipe) -> the single replay path; tree_structure(tree) and nested_scene_structure(groups) emit the
mind's own compositions AS typed structures (verified cosine 1.0 against the source ops).

WHAT IS / ISN'T UNIFIED (kept honestly):
  * Unified: the forward ENCODING. Name-addressable leaves (program opcodes, tree roles, scene group keys)
    become atoms; rng-drawn leaves (a SceneCoder sub-scene) ride as `raw` payloads -- so the scene's
    constructed STRUCTURE unifies into named atoms while its rng leaves stay raw (the constructed-vs-measured
    split again; raw round-trips to float32 ~1e-8, constructed all-atom round-trips truly bit-exact).
  * NOT unified: SEMANTICS (program execution / the CALL library, an EML node's scalar eval) -- those are
    layers above the encoding; program_to_recipe rejects CALL as out of scope. And the INVERSE (decode a
    foreign vector to a structure) is the resonator's job (decompose_structure), bounded by crosstalk -- a
    structure here is a GENERATOR, not a parser. B8 (denoised decode) and B9 (manifold) target this one type.

Tests: +7 in test_holographic_typed.py (program/tree/scene bit-exact; one-alphabet; CALL out of scope;
UnifiedMind wiring; superpose constructed round-trip). 644 -> 651. Additive: a new recipe primitive +
a new module + new UnifiedMind methods; no kernel/compute-path change.

## B8: denoised structure decoding -- per-peel cleanup pushes the decode depth cliff (shipped)

The B7 keystone unified the forward ENCODING; B8 attacks the INVERSE. A composed structure decoded by
ITERATED unbinding accumulates crosstalk, and -- the crux -- without cleanup that noise is carried into
the next query and COMPOUNDS. holographic_peel.py demonstrates this on a linked list
M = superpose_i bind(node_i, node_{i+1}) (itself a B7 typed structure via chain_recipe):
  * MEASURED (16-node chain, dim 512): raw traversal (no cleanup) decodes ~2/15 hops then craters and
    the carried vector diverges; per-peel cleanup decodes 15/15. ~2 -> full chain. Per-peel cleanup is
    the whole game.
  * Hard argmax cleanup and the B1 dense-Hopfield cleanup TIE on the discrete pointer (both 15/15) --
    exactly B1's kept negative: snapping to the nearest atom is Bayes-optimal for identity, so soft
    cannot beat it there.
  * The soft (Hopfield) update earns its keep on CONTINUOUS payloads: recovering off-grid scalar-encoded
    values from a superposition, the soft blend beats hard snap-to-grid (cosine ~0.996 vs ~0.990) -- it
    returns a mixture of nearby grid atoms, landing between grid points where the true value lives.

KEPT NEGATIVES / scope: a commutative-bind chain has an INTRINSIC predecessor leak (unbinding node_i
surfaces node_{i-1} as a clean atom, since node_{i-1} bound it as its value and node_i*involution(node_i)
=delta). A forward traversal KNOWS its predecessor, so traverse() explains it away -- standard history-aware
decode, reported not hidden. Permuting the key/value does NOT fix it (permute distributes through the
convolution and the cancellation returns); disjoint key/value codebooks would, at the cost of chainability.
SBC block codes (B2) bind losslessly -> no leak and no cliff to push (so this is a DENSE-HRR technique).
The soft-vs-hard continuous win is modest (~0.6%). Reuses dense_cleanup (B1) and StructureRecipe (B7).

Tests: +6 in test_holographic_peel.py (full-chain decode vs raw crater; hard/soft tie on pointers; correct
sequence; chain is a typed structure bit-exact; soft beats hard on continuous values; diverged hop marked).
651 -> 657. Additive: a new module; no kernel/compute-path change.

## B9: manifold-aware decompose -- detect topology, decompose on the right manifold (shipped)

The build-2 decompose assumes a flat line (a sum of elementary functions over an open interval). Many
signals live on a curved domain -- a RING (periodic), an antiperiodic MOBIUS band (only odd harmonics),
or a TORUS (two periods). On the wrong manifold a periodic signal needs many terms and EXTRAPOLATES BY
DIVERGING (a polynomial shoots off where the true signal repeats). holographic_manifold.py detects the
topology, then decomposes on the matched basis -- the decompose-side twin of the Mobius/AxialEncoder.

DETECTION: detrend -> FFT for a candidate fundamental (the LOWEST significant peak; a strong harmonic is
not the fundamental) -> VALIDATE by how well a harmonic basis at that period actually fits (robust to FFT
leakage). Commensurate peaks -> periodic (ring; mobius if the odd-only fit is as good); an incommensurate
peak -> torus. A poor best-fit (R^2<0.9) is guarded back to "line" (no spurious rings).

MEASURED: detected an OFF-GRID period (P~5, off the elementary freq grid) as ring; the matched harmonic
basis EXTRAPOLATES (RMS 0.024) where the flat-line polynomial DIVERGES/fails (RMS ~1.0 ring, ~5.4 mobius).
line/ring/mobius classify correctly and survive 5% noise (3/3 seeds each) on a 2-cycle window.

KEPT NEGATIVES:
  * TORUS needs a window long enough to RESOLVE the two incommensurate tones (Rayleigh, span>=1/df). On a
    short 2-cycle window the tones merge into one blurred peak and detection falls back to line rather than
    guessing -- a reported limitation, not a silent error (resolved correctly on a long window).
  * The MOBIUS (odd-only) basis vs the full ring basis is a TIE on extrapolation under noise (~0.005 each):
    MDL on the full-ring basis already prunes the spurious even harmonics, so the odd-only restriction's
    value is STRUCTURAL (guaranteed antiperiodicity, half the basis size), not measured accuracy.

Feeds straight into symbolic_regress (build 2) via a topology-matched dictionary -- no new search machinery.
Tests: +8 in test_holographic_manifold.py (detect line/ring/mobius; noise robustness; accurate period;
matched extrapolates vs flat-line diverges; dictionary shapes incl odd-only; records topology; torus window
requirement; line not forced to ring). 657 -> 665. Additive: a new module; no kernel/compute-path change.

## B6: Physarum flow-conductance maze solver (Tero et al. 2007) -- shipped

The elitist-ant slime solver (holographic_slime) is stochastic: random walkers laying pheromone into one
HRR field, needing many rounds + elitist reinforcement on a braided maze to avoid a longer tube.
holographic_flow.py implements the PRINCIPLED dynamics the organism actually uses (Tero, Kobayashi,
Nakagaki 2007): the maze is a tube network, flux from source(start) to sink(goal) is a weighted
graph-Laplacian solve L p = b (Poiseuille Q_ij = D_ij(p_i-p_j), conservation at every node), and tubes
adapt dD/dt = f(|Q|)-D with saturating f(Q)=|Q|^mu/(1+|Q|^mu). Iterate solve->adapt and the network
collapses onto the shortest source-sink path.

MEASURED vs the elitist ant on braided 16x16 mazes (same maze, same optimum): both find the OPTIMAL path
(seeds 3/7/11/15 -> 84/38/46/42 steps); Tero is DETERMINISTIC (identical reruns) and ~100-340x FASTER
(~90ms vs the ant's 10-32s). The bar -- beat elitist-ant on the braided maze at equal cost -- cleared
decisively. Path extracted by thresholding surviving tubes (BFS), falling back to a widest-path
(Dijkstra on 1/D) route.

KEPT NEGATIVES / scope: Tero is CENTRALIZED -- each step solves the WHOLE graph's Laplacian (O(N^3) dense),
whereas the ant is decentralized (local diffusion, one holographic field, plus the hierarchical partition
for huge mazes). So this is the principled-physics complement to the holographic ant, operating on the
DECODED adjacency, not itself a holographic method. It needs an explicit source+sink (the ant can diffuse
with no goal). The Baker/Rosetta-seat extension -- fragment assembly as a flow over an energy-conductance
landscape -- is NOT built; this delivers the maze bar, which was the gate.

Tests: +6 in test_holographic_flow.py (optimal on braided mazes; deterministic; picks short route through a
loop; disconnected/missing-endpoint -> None; wrapper reports optimum + determinism). 665 -> 671. Additive:
a new module; no kernel/compute-path change.

## Adaptive-rank denoising -- cashing the fixed-rank low-noise negative (shipped)

B7-original's manifold denoiser (fixed rank-8 projection) had a kept negative: at LOW noise it over-smooths
(projecting onto rank-8 discards real signal detail), measured -0.57 dB harm on real SOL windows. The
teed-up fix (Donoho/Milanfar threshold selection) is now in holographic_denoise.py:
  * fit_manifold_full(samples, rank) -> a GENEROUS basis + its singular values.
  * estimate_sigma(x) -> Donoho's MAD-of-finest-detail noise estimate (parameter-free).
  * adaptive_manifold_denoise(x, basis, mean, sigma=None) -> project, then HARD-THRESHOLD the coefficients
    at the universal level kappa*sigma*sqrt(2 ln r) (Donoho-Johnstone shrinkage in the manifold basis).
MEASURED on real SOL windows: at sigma=0.3 fixed rank-8 HARMS (-0.57 dB) while adaptive is neutral (-0.10);
at sigma=0.8 fixed +5.56 dB, adaptive +4.23 dB. The negative is cashed -- adaptive never meaningfully harms
across the noise range. KEPT NEGATIVE: adaptive does NOT match the ORACLE fixed-rank's peak high-noise gain
(when the true rank is known); the value is robustness to UNKNOWN noise, not beating the oracle. A contiguous
top-r* variant was rejected (it truncates real detail at low noise just like fixed rank); individual
coefficient thresholding keeps detail wherever it sits. +5 tests in test_holographic_denoise_adaptive.py.

## B6 part 2 -- fragment assembly as flow search (the Baker/Rosetta seat) (shipped)

The maze solver finds a min-cost path on a grid; the Baker/Rosetta seat's fragment assembly -- choose a
fragment per position to minimise an energy, consecutive fragments overlap-agreeing -- is a min-cost path on
a layered (position x fragment) TRELLIS. Same search. holographic_assembly.py builds the trellis (placement
energy encoded as unit hops via relay nodes, so the unit-length Tero solver's shortest path == the min-energy
assembly) and recovers the chosen fragments as a B7 StructureRecipe (each fragment bound to its position).
MEASURED vs exact DP (Viterbi): complete library -> assembles the target EXACTLY (energy 0); with a true
fragment missing -> forced mismatches, and the flow assembly MATCHES the DP optimum (energy 9 == 9), i.e. the
GLOBAL best, not a greedy one. KEPT NEGATIVE: this is the combinatorial CORE (a placement-mismatch energy, a
Rosetta-score stand-in), not a protein force field; the relay encoding bloats the graph by total energy
(fine small; weight edges by length directly for large). +4 tests in test_holographic_assembly.py.

These two finish the B-list: every breakthrough (B1-B10), the three integration-review items (typed
structure / denoised decode / manifold decompose), B6 (Tero flow) and its fragment-assembly generalisation,
plus the teed-up adaptive-rank denoiser, are shipped and measured with negatives intact. 671 -> 680.

## Integration plan, Tier 1 -- the DECOMPOSE / DENOISE / FIT faculties wired into UnifiedMind (shipped)

The integration plan's audit found 14 modules built since the last review with ZERO references in
UnifiedMind: the substrate was shared (every module is bind/bundle/cleanup on the one kernel) but the
orchestration was siloed. UnifiedMind was strong on one half of the loop -- COMPOSE / RECALL / PREDICT /
GENERATE (build structure, act) -- and everything built since is the OTHER half: DECOMPOSE / DENOISE /
SEARCH (take a foreign signal apart, on the right manifold, cleaned). This ships Tier 1: the three
highest-value, mostly-thin-wrapping faculties, each unifying several modules behind one honest entry
point -- the same move B7's typed_structure() made for composition.

  * decompose_signal(x, y) -- one faculty over manifold + symbolic + the multiplicative mode + mobius.
    Detect the domain topology (detect_topology), then route the basis: line -> compress_signal(mode=
    'auto') picks an additive OR multiplicative (log-transform) law by the measured conservative rule
    (competitive in-sample AND better on a held-out tail); ring/mobius/torus -> decompose_on_manifold's
    matched harmonic basis (mobius = ODD harmonics only, the antiperiodic space). Returns a Formula --
    already a savable seed (.generate/.save/.load), the measured-regime twin of a StructureRecipe.
    info normalised across both branches: topology, period, mode, n_terms, resid_rms, compression_ratio.
    Ergonomic shorthand: decompose_signal(y) fits a lone signal on a unit index grid.
  * denoise(x, method='auto', samples=/codebook=) -- one callable over denoise + hopfield. 'adaptive'
    (noise-thresholded low-rank projection, the safe default), 'manifold' (fixed-rank), 'codebook'
    (modern-Hopfield cleanup), 'nlm' (non-local means on x's own near-duplicates), 'pnp' (Plug-and-Play
    /RED restoration with the adaptive map as prior). 'auto' picks codebook if a codebook is given else
    adaptive manifold if samples are given. DECISION KEPT HONEST: NLM and PnP stay opt-in -- deciding
    self-similar-vs-low-rank automatically is itself a measurement, not faked with an unvalidated
    heuristic. And denoise REFUSES a lone vector with no prior (samples/codebook): a denoiser is a map
    of a manifold, and there is no free lunch.
  * fit_function(X, y) -- the KAN readout as a faculty: a single-layer Kolmogorov-Arnold fit
    (HolographicKAN) at this mind's seed, exposing .predict and .feature_function(j, ts). A lone feature
    vector is taken as one column.

KEPT NEGATIVES, surfaced through the faculties rather than buried in the modules: fixed-rank projection
over-smooths at low noise (use 'adaptive', ~neutral there); a manifold projection only helps where real
low-rank structure exists (it destroys structureless signal); NLM only helps where near-duplicates exist;
a single-layer additive KAN cannot represent feature interactions (e.g. x1*x2) -- additive by construction.
And the multiplicative law is auto-selected only on a LINE domain and needs y > 0; a torus needs a window
long enough to resolve both tones or detection falls back to line (the Rayleigh limit).

THE WIRING IS PROVEN, NOT NOMINAL. The plan's hardest prior lesson (its section 6) was that naive
cross-module chaining once REGRESSED -- a denoiser fed a recall output dropped cosine -- because a shared
KERNEL is not a shared MANIFOLD. So Tier 1 lands with test_integration.py running a cross-faculty pipeline
THROUGH the mind end to end: detect topology -> decompose_signal -> seed.save -> realize (reload + generate)
-> denoise, with each hop's prior matched to its input, asserting the END materially improves on the noisy
input (no silent regression). A faculty that only imports is still a silo; these run. A live confirmation
fell out of writing the test: the pipeline signal sin(x) + 0.4 sin(3x) is purely odd-harmonic, so the
topology detector correctly classified it as MOBIUS (not ring) -- the antiperiodic branch firing on real
input, exactly as the manifold work intended.

Tests: +7 in test_integration.py (faculties present; periodic-law recovery + bit-exact seed roundtrip +
bounded periodic extrapolation; multiplicative auto-select + single-array shorthand; the end-to-end
pipeline with a no-regression assertion; denoise routing + the honest no-prior refusal + the codebook map;
real-SOL high-noise denoise gain; KAN additive recovery + the interaction-limit boundary). 680 -> 687.
Additive: three new methods on UnifiedMind, each a thin lazy-import wrapper over already-measured modules;
no kernel or compute-path change; fully backward-compatible (new methods only, no signature changes).

NOT YET WIRED (the rest of the integration plan, sequenced next): Tier 2 -- resolve the factor_composite
duplication by delegating to the B2 SBC resonator (the real de-siloing), decode_structure via peel, and the
opt-in energy cleanup; Tier 3 -- the flow-search and dynamics faculties; Tier 4 -- rd-quant save and the
generative reconcile. This entry covers Tier 1 only.

## Integration plan, Tier 2 (item 5) -- the factor_composite de-siloing (shipped)

The plan's "real de-siloing": the audit found TWO factorizers -- the original dense MAP/bipolar
ResonatorNetwork (reached via UnifiedMind.factor_composite) and the newer SBC resonator
(holographic_sbc.decompose_structure / sbc_resonator) with measured higher capacity and a
reconstruction-confidence check -- and called for one. Reading both made the honest shape of the fix
clear, and it is NOT the literal "replace factor_composite's internals" the one-line plan implied:

  * The two factor DIFFERENT objects in DIFFERENT algebras. Dense MAP binding is the elementwise
    sign-product (self-inverse); SBC binding is per-block modular addition of one-hots (block-local
    circular convolution). The SBC resonator CANNOT factor a dense MAP composite, and you cannot
    faithfully transcode an existing dense composite into SBC and recover the same indices. Verified by
    reading the modules; the B2 module states it outright ("SBC lives beside the dense kernel, not
    inside it").
  * factor_composite's dense contract is PINNED by test_brain_factor_composite (dense codebooks, a
    MAP-bound triple, must solve) and by the backward-compatibility rule. So the dense path could not be
    deleted -- only delegated-past and deprecated.

What shipped, the honest version:
  * NEW FACULTY UnifiedMind.decompose_structure(composed, codebooks, L) -- the SBC factorizer, which had
    NO mind-level entry point before (only a bare module fn and a tour line), is now a first-class faculty
    the mind speaks directly. Returns {picks, factors, verified, present}: verified<=>the picks rebuild the
    product (it verifies or abstains, never guesses), and present[f] is False when factor f resolved to the
    SBC identity (presence detection).
  * factor_composite is now ONE entry point: given an `L` it routes to decompose_structure (the preferred,
    validated path) and maps the result onto a superset of the old contract (factors/solved + verified/
    present + backend='sbc'); without `L` it runs the legacy dense ResonatorNetwork and emits a
    DeprecationWarning steering new code to the SBC path (backend='dense'). The dense return keys are
    unchanged, so the pinned test still passes.

This is genuine de-siloing -- one factorizer for new code, exposed by name -- without faking an algebra
bridge. KEPT NEGATIVE / boundary, on the record: the dense MAP path is RETAINED (deprecated, not removed)
because the SBC resonator is a different algebra that cannot factor dense MAP composites; "one factorizer"
means one PREFERRED, mind-exposed factorizer with the legacy path honestly labelled, not a single
implementation covering both algebras. CI runs plain `pytest -q` (no warning escalation), so the
DeprecationWarning is informative, not breaking; the one legacy test that triggers it now asserts it via
pytest.warns, keeping CI output clean and the intent explicit.

Tests: +3 in test_integration.py (decompose_structure faculty factors + verifies + presence; factor_composite
routes to SBC and agrees with decompose_structure, presence survives routing; the dense path is
backward-compatible AND deprecated). test_holographic_resonator.py's brain test updated to expect the
deprecation (no count change). 687 -> 690. Additive: one new method + a router on the existing method; no
kernel/compute-path change; the dense path's behaviour and return keys are unchanged.

NEXT (still queued): Tier 2 remainder -- decode_structure via peel (the B8 denoised per-peel decode), and
opt-in energy cleanup (cleanup(..., energy=True) -> hopfield.dense_cleanup, pinned to argmax at high beta);
then Tier 3 (flow-search + dynamics faculties) and Tier 4 (rd-quant save + generative reconcile).

## Integration plan, Tier 2 (items 4 & 6) -- decode_structure (peel) + opt-in energy cleanup (shipped)

The two smaller Tier 2 wirings, finishing the DECODE side. Both modules were already shipped and measured
(B8 peel, B1 dense-Hopfield); the work was exposing them on the mind, with their negatives intact.

  * NEW FACULTIES chain_structure(n) and decode_structure(memory, nodes) -- B7's forward chain object and
    its B8 inverse, on the one substrate. chain_structure builds the linked list M = superpose_i
    bind(node_i, node_{i+1}) as a StructureRecipe at the mind's dim/seed (realize() gives M); decode_structure
    traverses it back by iterated unbinding with PER-PEEL CLEANUP. The crux B8 measured, now visible through
    the mind: each recovered pointer is noisy and that noise COMPOUNDS into the next hop, so a raw traversal
    (cleanup=None) craters after ~1-2 hops while per-peel cleanup decodes all 15 of a 16-node chain. cleanup
    in {None, 'hard', 'soft'}; hard and soft TIE on the discrete pointer (B1's kept negative -- argmax is
    Bayes-optimal for identity). Named distinctly from decompose_structure on purpose: decode_structure is the
    SEQUENCE inverse (traverse a chain), decompose_structure is the PRODUCT inverse (factor a bound product) --
    different structures, different inverses; the docstrings cross-reference so the pair is not confused.
  * OPT-IN ENERGY CLEANUP -- the B1 plan, finally wired exactly where B1 specified: Vocabulary.cleanup gained
    an energy=False flag (beta, steps). With energy=True the query is first denoised by the modern-Hopfield
    update z <- V^T softmax(beta*V z) against the candidate codebook, THEN the usual nearest-symbol readout
    runs. At beta->inf the softmax is one-hot, so the returned identity is BIT-FOR-BIT the plain argmax (the
    pinned guarantee). It is a kernel change but purely additive and default-off, so every existing caller is
    unaffected (verified: the full core suite passes unchanged). KEPT NEGATIVE, restated: on identity it ties
    hard argmax -- the value is cleaning CONTINUOUS vectors (the Tier 1 denoise faculty and peel's soft path),
    not changing which discrete symbol wins.

With these, the mind speaks the whole inverse half of the loop: decompose_signal (a law), decompose_structure
(a product's factors), decode_structure (a chain's sequence), denoise (a manifold), fit_function (a function) --
each a thin faculty over measured code, each proven through the mind.

Tests: +2 in test_integration.py (decode_structure round-trips a chain through the mind -- per-peel decodes 15/15,
raw craters <=3, soft ties hard; energy cleanup is opt-in and matches argmax bit-for-bit at high beta). 690 -> 692.
Additive: two new mind faculties + one default-off optional kwarg on Vocabulary.cleanup; no change to any existing
code path (core engine, algebra, peel, typed, recipe, unified suites all pass unchanged).

TIER 2 IS COMPLETE (items 4, 5, 6). REMAINING: Tier 3 -- one flow-search faculty (solve_maze via flow.solve_maze_flow,
assemble via assembly.assemble) and learn_dynamics (dynamics.Propagator, keeping the market kept-negative); Tier 4 --
UnifiedMind save requesting quant='rd' on low-rank arrays, and reconciling the generative paths (vector generate ->
hopfield.generate, splat -> scene/archive). Also pending from the modules' own B-list: NEW B7/B8 were the typed
structure and denoised decode (now both wired); no further B-list items remain unshipped per the last close-out.

## Integration plan, Tier 3 -- the SEARCH & DYNAMICS faculties (shipped)

Min-cost search (on a graph, on a trellis) and learned linear dynamics, now faculties of the mind. All
three modules were shipped and measured already; the work was exposing them on UnifiedMind with their
negatives intact, and -- where natural -- returning the search result as a B7 typed structure.

  * NEW FACULTY solve_maze(world) -> delegates to flow.solve_maze_flow (the deterministic Tero
    flow-conductance model: Physarum tubes thicken with Poiseuille flux until the network collapses onto
    the shortest path). Same (path, info) interface as the stochastic slime solver, but DETERMINISTIC and
    ~100x faster, and it lands EXACTLY on the optimum on braided mazes (extracted_len == optimal).
  * NEW FACULTY assemble(target, library) -> delegates to assembly.assemble, built at the mind's dim/seed.
    Rosetta-style fragment assembly (a fragment per position minimising a placement energy, consecutive
    fragments overlap-agreeing) cast as the SAME min-cost flow the maze solver runs, on a (position x
    fragment) trellis. Returns the assembled string, energy, chosen (pos, fragment) list, and a B7
    StructureRecipe binding each fragment to its position -- the search result AS a typed structure the mind
    can realize(). It attains the GLOBAL (Viterbi) optimum, not a greedy one. KEPT NEGATIVE: the energy is a
    placement-mismatch / Rosetta-score STAND-IN, the combinatorial core, not a protein force field. Per the
    plan, the DP oracle assemble_optimal_energy stays a module reference function, NOT a mind method.
  * NEW FACULTY learn_dynamics(states) -> delegates to dynamics.Propagator.learn. Learns a fixed operator U
    with state(t+1) ~ bind(U, state(t)) -- in HRR's Fourier domain a per-frequency complex transfer, i.e. the
    Koopman/DMD operator in Fourier coordinates. The Propagator exposes .step (one-step prediction = a single
    bind), .rollout(state, k), and .recall_at(state, k) (recover the state k steps BEFORE one now). KEPT
    NEGATIVE on real SOL returns: prediction only TIES a trivial mean predictor (near-efficient-market returns
    have almost no linear structure for a fixed operator to exploit) -- it shines on signals with genuine
    linear dynamics (audio, fluids, a bind-shaped control). The CONTENT-ADDRESSABLE round-trip (forward k then
    back k -> the start at cosine ~1.0) is the durable win regardless, and is what the integration test pins.

Tests: +3 in test_integration.py (solve_maze finds the optimal path and is deterministic; assemble is optimal,
matches the exact DP under forced mismatch, and returns a realizable B7 typed structure; learn_dynamics predicts
bind-shaped dynamics far past persistence and round-trips a trajectory at cosine >0.99). 692 -> 695. Additive:
three new mind faculties, each a thin lazy-import delegate over an already-measured module; no kernel or
compute-path change; backward-compatible (new methods only).

TIER 3 COMPLETE. REMAINING: Tier 4 (lowest urgency) -- (9) UnifiedMind's save path requesting quant='rd' on
low-rank arrays (B5 is already in holographic_core.save; just request it from the mind), and (10) reconciling the
generative paths: point a vector-level generate at hopfield.generate (B10 diffusion) and connect splat (B8) to
the scene/archive representation.

## Integration plan, Tier 4 -- persistence (rate-distortion) + the generative faculties (shipped)

The last tier, and the one where the plan's estimate was furthest off: item 9 ("UnifiedMind save uses
quant='rd' -- just request it. Small.") assumed a save path existed. It did NOT -- UnifiedMind had no
to_state/save at all. So this built one, honestly scoped, with a round-trip test (the project's bar for
persistence), which makes it a real feature, not a flag flip.

  * NEW: UnifiedMind.to_state / from_state / save / load, and UnifiedMind registered in
    holographic_core._registry() (lazy import, no load-time cycle), so the kernel's versioned save handles
    it like every other persistable object and quant='rd'/'auto'/'int8' all apply. save persists the mind's
    LEARNED GENERALIZATION: the encoder (perception), the SelfOrganizingMind (the prototype classifier), the
    HolographicMind decision brain, and the routing/format bookkeeping classify reads. MEASURED: classify AND
    decide are bit-for-bit identical after save->load across quant levels (the encoder, memory, and brain each
    already had verified round-trips; this composes them).
    DOCUMENTED BOUNDARY / KEPT NEGATIVE: the verbatim recall index of individuals (`_recall`) is NOT persisted
    -- its payloads are arbitrary original inputs (raw arrays, dicts, strings) that do not round-trip through a
    structured array save -- and the lazy/derived faculties (sequence & plan memory, the text/word generators,
    meaning predictors, the scene coder, the FHRR high-capacity memory) are rebuilt on use, not stored. What
    round-trips is the trained generalization (classify + decide), proven; recall raises "nothing learned yet"
    after a bare load (re-learn for it). Also honest: on a SMALL mind quant='rd' finds no low-rank 2D array to
    activate on and falls back to int8 (marginally larger only by per-array qspec overhead) -- rd's ~11x win is
    on genuinely low-rank consolidated/bundled state, exactly as the B5 module documents.
  * NEW FACULTY generate_vector(codebook) -> delegates to hopfield.generate (B10): generate a hypervector by
    denoising FROM PURE NOISE -- anneal beta up and injected noise down, walking onto the codebook manifold.
    Generation and denoising are the same operation in different regimes; this is the vector-level twin of the
    text generate(). KEPT NEGATIVE: over a bare codebook it converges to a stored atom (degenerate) -- feed a
    composed/continuous manifold for novel-but-valid samples.
  * NEW FACULTY splat_field(target, k, denoise=False) -> delegates to holographic_splat: represent a 2-D field
    as a SUPERPOSITION of K Gaussian primitives by matching pursuit (a splat scene IS a bundle; the RBF
    ScalarEncoder is already a Gaussian splat in hypervector space). Reconstructs compactly and, with
    denoise=True, denoises (smooth Gaussians have no capacity for noise). KEPT NEGATIVE / SCOPE: isotropic
    splats + fixed scales (the honest matching-pursuit baseline); anisotropic covariances, gradient refinement
    (full 3DGS), and storing archive images AS splat bundles are documented build targets, not done here.

Tests: +3 in test_integration.py (the mind save/load round-trips classify AND decide identically with quant='rd',
and the un-persisted recall index raises after a bare load -- the boundary asserted; generate_vector lands on the
manifold and is seed-deterministic; splat_field reconstructs >25 dB and denoises). 695 -> 698. Additive: four new
mind methods (save/load/to_state/from_state) + two generative faculties + one lazy registry key in core; no change
to any existing save/load logic or compute path (the full persistence suite passes unchanged).

=== THE INTEGRATION PLAN IS COMPLETE (Tiers 1-4, all ten items). ===
UnifiedMind now speaks both halves of the loop on one substrate:
  FORWARD : perceive / classify / recall / decide / generate (text) / compose (recipe, typed, nested scene)
  INVERSE : decompose_signal (a law) / decompose_structure (a product's factors) / decode_structure (a chain) /
            denoise (a manifold) / fit_function (a function)
  SEARCH  : solve_maze (Tero flow) / assemble (fragment assembly, as a typed structure)
  DYNAMICS: learn_dynamics (prediction is one bind; content-addressable trajectories)
  STORAGE : save / load (rate-distortion), GENERATIVE: generate_vector (B10 diffusion) / splat_field (B8 splats)
The de-siloing is real -- one factorizer for new code (the dense path deprecated, not faked away); the wiring is
proven by test_integration.py running each faculty THROUGH the mind end to end (the §6 lesson: a shared kernel is
not a shared manifold); and every faculty carries its measured negatives. The corollary the plan set out to
enforce holds: there is one MIND the primitives serve, not a drawer of disconnected experiments beside it.

## Wiring check -- integration plan verified against the live code (clean, with two flagged boundaries)

Re-ran the plan's own audit (the 14 modules it found with ZERO references in UnifiedMind) plus a
faculty-presence + full-integration-suite pass. Verdict: every plan item is wired and works; two honest
boundaries are flagged below (neither a missed item).

Module references in UnifiedMind now (was 0/14):
  * 12/14 wired DIRECTLY: symbolic, kan, sbc, peel, manifold, flow, assembly, hopfield, splat, dynamics,
    denoise (+ the antiperiodic concept via manifold).
  * ratedistortion: 0 direct refs but reachable TRANSITIVELY -- UnifiedMind.save -> holographic_core.save
    (quant='rd') -> holographic_ratedistortion. Intended (the plan said rd lives in core.save; the mind now
    requests it). VERIFIED reachable.
  * machine (HoloMachine VM): 0 refs -- INTENTIONALLY standalone per plan §5 ("a VM, adjacent to the mind,
    not a faculty"). Correct.

Faculty presence (live UnifiedMind, all callable): decompose_signal, denoise, fit_function, chain_structure,
decode_structure, decompose_structure, factor_composite (routing+deprecated dense), solve_maze, assemble,
learn_dynamics, save/load/to_state/from_state, generate_vector, splat_field; plus Vocabulary.cleanup(energy=).
UnifiedMind public-method count 86 -> 99. All 18 integration tests pass.

Duplication table (§4) -- all four rows now closed:
  * factor_composite -> SBC resonator: DONE (routes to decompose_structure on L; dense deprecated).
  * vector generate -> hopfield.generate (B10): DONE (generate_vector).
  * save path -> quant='rd': DONE (UnifiedMind.save requests it via core.save).
  * compress_lossless vs symbolic/recipe: the one row left as a doc cross-link -- now CLOSED: compress_lossless's
    docstring documents the boundary (lossless entropy coding of discrete TOKENS vs decompose_signal's lossy
    generating LAW over a CONTINUOUS signal; both kept, different levels).

TWO FLAGGED BOUNDARIES (honest, neither is a missed plan item):
  1. holographic_mobius MODULE: the plan's only mobius reference was "(mobius for the antiperiodic basis)" in
     decompose_signal -- that role IS wired and tested (manifold.detect_topology classifies 'mobius' and
     manifold_dictionary builds the ODD-harmonic basis itself; sin(x)+0.4sin(3x) decodes as mobius). But the
     standalone holographic_mobius module -- its AxialEncoder (the double-angle map for AXIAL data: theta ==
     theta+pi, orientation/director fields) and antiperiodic_split helper -- is NOT on the mind's call path
     (referenced only in a docstring, the tour, and its own tests). The AxialEncoder is a distinct ENCODER
     capability the plan never listed for wiring; it remains a standalone study, like machine. CANDIDATE for a
     future encoder faculty (e.g. perceive(..., axial=True)) if wanted -- not done, not claimed.
  2. splat -> archive: splat_field is wired (a 2-D field as a Gaussian-splat superposition + denoiser). The
     DEEPER integration item 10 gestured at -- storing ARCHIVE images AS splat bundles beside the WHT plates --
     is the addendum's documented build target and is NOT done. splat is connected to the mind as a faculty,
     not (yet) fused into the archive store.

No code paths regressed (persistence, brain, organizer, scene, relations, schema, resonator, algebra suites all
green). The only change this check made was the compress_lossless docstring cross-link.

## Axial perception + the splat-bundle archive (shipped) -- the two wiring-check boundaries closed

The wiring check flagged two modules whose CAPABILITY was reachable but whose own code was not on the
mind's path. Both are now wired, each through its real published method, each measured.

AXIAL MODALITY (holographic_mobius.AxialEncoder -> the encoder). An axial value is one where theta and
theta+pi mean the SAME thing -- an unoriented line, a director/nematic field, a crystal axis. On a circle
they sit apart; the fix is the double-angle map theta -> 2*theta onto RP^1 (the Mobius base). UniversalEncoder
now builds an AxialEncoder(dim//2) and exposes modality="axial": it takes the real [Re, Im] embedding of the
phasor, which PRESERVES the FHRR cosine and lands the value in the SAME real space as every other modality,
so the one memory can learn / classify / recall orientations correctly. UnifiedMind gains axial_similarity()
and decode_axial(). MEASURED (dim 512): sim(theta, theta+pi)=+1.00 (same orientation) where the plain number
modality gives +0.76 and cannot see the flip as identity; sim(theta, theta+pi/2)=-0.17 (orthogonal); decode is
mod pi (1.2 and 1.2+pi both read 1.20); and a flipped A-orientation still classifies as A. OPT-IN: a bare float
infers as "number" (infer() cannot tell axial from a plain angle), so axial must be declared.

SPLAT-BUNDLE ARCHIVE (holographic_splat + new holographic_splat_archive). A 2-D field is a SUPERPOSITION of
Gaussian primitives -- a bundle. SplatArchive stores a gallery as splat codes (cy, cx, amp, sigma) per channel
BESIDE the WHT-plate archive. Because matching pursuit orders splats by decreasing residual energy, the stored
list is already importance-sorted, which buys: PROGRESSIVE REFINEMENT for free (recover(i, k) renders a k-prefix
-- a coarser valid preview; gallery 27.3 dB full vs 19.6 dB at K/4), an EXACT region query (the splats whose
centre lies in a box ARE what is there), and a fixed tunable byte budget. holographic_splat also gains the
addendum's named HRR functions: splat_bundle() encodes a scene as ONE hypervector (quantised per-region peak
occupancy bound to region roles, bundled) and recall_region() reads a region back by unbinding its role and
cleaning up against orthogonal level atoms -- RELIABLE (100% exact-level recall up to 36 regions at dim 4096)
but COARSE (a quantised occupancy, not the splats). UnifiedMind gains splat_archive().

KEPT NEGATIVES (measured, not hidden):
  * The splat archive is LOSSY and, on the DCT-friendly _gallery, the WHT plates BEAT it on quality at a matched
    byte budget (WHT keep=120 reconstructs near-exactly: 163.7 dB at 75 KB vs splat 27.3 dB at 55 KB). The
    addendum's "match or beat WHT quality" bar is NOT met for quality on these smooth images -- DCT is ideal for
    gradients. The splat archive's real value is the ADDED region-query + progressive-refinement + compact code,
    not quality parity; it sits BESIDE the plates, not in place of them. No damage-tolerant joint recovery either
    (the plates' strength under erasure). Isotropic splats only (anisotropic covariances / gradient refinement =
    full 3DGS, out of scope).
  * recall_region is coarse (quantised levels), not a continuous descriptor; the exact per-splat content is
    SplatArchive.region.

LESSON BANKED: this engine's unbind is unbind(composite, key) -- composite FIRST. A reversed call
(unbind(key, composite)) returns ~orthogonal noise and silently destroys recall; it cost a full mis-diagnosis as
a "VSA capacity cliff" before a one-line bind/unbind round-trip check (cosine ~0, not ~1) exposed the real cause.
Always sanity-check the round-trip before blaming capacity.

Tests: +3 (axial modality theta==theta+pi incl. flip-invariant classify; splat archive recover/refine/region/
recall; splat_bundle superposition carries region signal). 698 -> 701.

## Two investigations that did NOT earn a build (kept negatives + the mechanism, so they aren't re-tried blindly)

Two reframes were proposed -- "bitspace as a loss surface" and "primes as local minima" -- and prototyped
and measured on the real substrate (exp_bitspace.py / exp_primes.py, not shipped). Both are accurate
DESCRIPTIONS of things the engine already does, but neither produced a refinement worth shipping. The
measurements and the reason each failed:

BITSPACE AS A LOSS SURFACE -- per-component bit allocation vs B5's single global step. B5
(geometry_preserving_code) quantizes all KLT coefficients with ONE delta found by bisection to hit a target
mean cosine. Tested a per-component allocation that greedily descends a pairwise-cosine-error surface over
bit-allocations, at a MATCHED bit budget, on a controlled low-rank codebook and on real SOL price windows.
  * B5 already achieves recall@1 = 0.94-1.00 at every budget tested (down to target_cos 0.97). There is
    essentially NO recall headroom to recover.
  * On idealized low-rank data the per-component code MATCHES B5's recall and roughly HALVES the pairwise-
    geometry error at the same bits (e.g. 0.019 -> 0.009) -- a real but modest win on a metric recall does
    not need.
  * On REAL SOL windows the per-component greedy is WORSE: it stalls at ~45 bits (flat-ish spectrum, no
    dominant low-rank structure to allocate selectively) and gets recall 0.74 vs B5's 0.94-0.99. The single
    global step is the right move for near-flat spectra.
  WHY: the KLT orders directions by BETWEEN-vector variance, so one global step already crushes mainly the
  non-discriminative directions, and a uniform step is the water-filling solution for the retained
  components. B5 is already near rate-distortion-optimal for the recall geometry. Not worth the squeeze.

PRIMES AS LOCAL MINIMA -- log-prime matching pursuit as a factorization/compression algorithm. The idea:
decompose a value's log as a sum over a log-prime basis {log2, log3, log5, ...} by greedy residual descent,
so prime-power values land on exact lattice minima.
  * The ALGORITHM FAILS. Coordinate/greedy descent over log-primes recovers ONLY pure powers of the first
    basis prime (2^16, 2^10 correct); EVERY multi-prime smooth number is recovered WRONG -- 3^10 -> "2^16",
    360 -> "2^8", 2^6*3^6 -> "2^16". Reason: smooth numbers are DENSE in log space (2^16 ~= 3^10 to 0.10 in
    log), so the residual landscape is a thicket of shallow SPURIOUS near-minima, not clean isolated ones;
    greedy descent lands on the wrong one. The "local minima" intuition is actively misleading here.
  * EXACT factorization (trial division -- not matching pursuit) does work: ~2x on a smooth-integer signal
    (901 vs 1680 raw bits). But it is WORSE than raw on functional signals and worse than nothing on random
    data, and -- decisively -- the engine's own symbolic compressor nails a functional signal like y=3x^2
    EXACTLY (44 bits, residual 5e-13) where the prime code spends 723. Prime factorization only helps on
    smooth-INTEGER signals with no functional form, which the engine does not process (its inputs are float
    hypervectors, prices, structured states).
  WHY: the reframe restates the already-documented observation ("prime powers compress dramatically; large
  primes/random do not") but does not become a buildable capability -- the MP version is mathematically
  wrong, and the exact version has no home in the engine's data. Not worth the squeeze.

THE COMMON THREAD (the part worth keeping): both phrases name the move the engine already makes -- pick the
representation where the hard thing becomes a downhill walk (log turns x into +, KLT decorrelates, the
Hopfield energy turns recall into descent). They are good descriptions of the design, not new leverage over
it. Measured, written down, moved on.

## Honesty woven into recognition -- calibrated confidence + abstention as CORE (shipped)

The honesty layer (holographic_honesty: RecallNull / SPRTRecall / bh_fdr) was a standalone MEASUREMENT
harness -- the tour, the tests and holographic_ablate.py called it to VALIDATE the engine, but the mind
itself never used it. It is now part of how the UnifiedMind RECOGNISES, on both readout paths.

The move it encodes: a raw recall cosine means nothing on its own (radio-SETI and particle physics live by
this) -- you ask how high pure NOISE reaches against THIS codebook before believing a match. RecallNull
draws random unit queries against the mind's own prototypes and records the best cosine each reaches; that
empirical null IS the noise floor, and pvalue(score) = the fraction of noise reaching `score` or higher =
the honest false-alarm probability. Small p: trust the recall. Large p: ABSTAIN.

Wired (all auto-maintained on the mind's OWN data -- no external calibration set, no new persisted state):
  * _recognition_null  -- a RecallNull over the class PROTOTYPE codebook (memory.live._stack()), rebuilt
    only when the prototype set changes (keyed on the store mutation counter _gen), so steady state is free.
  * recognize(x)       -- CORE calibrated recognition: (label, similarity, pvalue).
  * classify(x, abstain=alpha) -- the label only if p <= alpha, else (None, sim). Default abstain=None
    preserves the original always-name-a-nearest-label behaviour EXACTLY (the (label, score) tuple shape is
    unchanged), so every existing caller is untouched.
  * recall_calibrated / recall(x, abstain) -- the SAME treatment for the INDIVIDUAL store (a _recall_null
    over a capped sample of self._recall.vecs), so BOTH memory readouts can say "I have nothing like this".
    (Exact-scan winner -- on a large store it can name a truly-nearest item the sublinear forest misses --
    and the capped sample is a documented under-estimate of the true floor.)
  * stream_recognize(cues) -- Wald's SPRT over a stream of cues bearing on the same thing; null density =
    the mind's noise floor, match density = its own examples' self-similarity (the quantity coherence()
    reads). Decides MATCH / REJECT as fast as the evidence allows.
  * recognize_batch(queries) -- bh_fdr (Benjamini-Hochberg/Yekutieli) over the per-query p-values, so
    scanning many queries cannot manufacture matches by luck (the look-elsewhere discipline).

MEASURED (dim 512, three single-token text classes): a learned member recognises at p=0.000; gibberish at
p~0.5 so classify(abstain=.05) returns None; the SPRT stream of canine cues decides MATCH in 1 sample
(well-separated densities); the FDR batch keeps the 3 real members and drops the gibberish; recall abstains
on an unseen query. KEPT NEGATIVE / scope: single-token text only matches what was literally learned (no
co-occurrence), so an UNLEARNED synonym like 'terrier' is correctly noise to this mind -- the honesty layer
flags it, which is the point, not a failure to generalise.

AUDIT (the second half of the task -- is anything else "just a callable method" that should be core?).
Enumerated all 103 public UnifiedMind methods. Finding: honesty was the ONE cross-cutting *property* (vs
*operation*) that belonged in the core, and it is now there, on both readouts. The rest fall into two
groups, both correctly placed:
  * the core loop itself (perceive / learn / classify / recall / recognize / decide / reinforce / save), and
  * on-demand TRANSFORMATIONS (decompose_signal, denoise, fit_function, decompose_structure,
    factor_composite, chain/decode_structure, solve_maze, assemble, learn_dynamics, generate_vector,
    splat_field/archive, typed_structure, compose/decompose scene/nested, blend, ...). These are things you
    INVOKE, not background properties of recognition; forcing them into the core loop is the "a faculty must
    earn its method" anti-pattern the integration plan warns against.
denoise is DELIBERATELY standalone for a MEASURED reason (integration plan section 6): a denoiser fed a
recall output dropped cosine 0.13 -> -0.06 -- a shared kernel is not a shared manifold -- so chaining it
into recall REGRESSES. ONE opportunity is flagged, NOT forced: the calibrated null could replace the
organizer's fixed-floor novelty(0.35) and DRIVE reorganization (reorganize when calibrated-novel inputs
accumulate, not on a fixed schedule). That is a change to AUTONOMOUS behaviour and needs its own measurement
plus a design decision (the learn buffer has a prototype-formation lag), so it is recorded here as a
measured follow-up rather than wired blind.

Tests: +4 honesty integration tests through the mind (recognize calibrated + classify abstain; SPRT stream
MATCH vs REJECT; FDR-controlled batch; recall abstains on unseen). 701 -> 705.

## Reorganize when INCOHERENT, not on a clock -- a kept negative (calibrated novelty) and a win (coherence gate) (shipped)

The audit that wove honesty into recognition flagged ONE opportunity it did not take blindly: the
calibrated noise floor could DRIVE reorganization -- reorganize when calibrated-novel inputs arrive,
not on a fixed schedule. "Build it and let's find out", so this is the measured outcome.

Setup (exp_calibrated_maintain.py, scratch -- not shipped). A prequential stream where reorganization
genuinely matters: each class is two ANTIPODAL modes on a circle, so the class centroid collapses and the
single (blurry) prototype online `add` keeps is useless -- only a SPLIT (auto_reorganize's job) classifies
it. Two new classes arrive mid-stream, so a trigger's RESPONSIVENESS shows up in post-shift accuracy. The
honest frame: auto_reorganize is SELF-VALIDATING (it holds out recent data, tries k=1..4, adopts the best,
defaults "keep"), so running it never hurts accuracy -- it only costs compute. The question is therefore
the accuracy-vs-COST frontier: hold accuracy with FEWER expensive passes.

NEGATIVE (the flagged idea). A calibrated-NOVELTY trigger does NOT work. 8 seeds: 75.5+/-9.5% overall,
46.9+/-6.4% on the new classes -- the FLOOR -- at 3.4 passes. It fires rarely and ineffectively because
NOVELTY detects "matches nothing", but online `add` always leaves SOMETHING to match, so the signal stays
low even when the store badly needs reorganizing. And CALIBRATION added nothing over the organizer's fixed
cosine floor (novelty(0.35)): both sat at the ~45% floor. The value of reorganizing here is fixing
INCOHERENCE, which novelty cannot see -- a standing property, not a new-thing-arriving event.

WIN (what the negative pointed to). COHERENCE -- mean similarity of recent inputs to their own prototype --
IS the signal. A coherence-gated trigger (reorganize when coherence drops below a floor) gets, 8 seeds,
85.5+/-1.6% overall at 5.8 passes: it BEATS the comparable fixed schedule (k=80: 82.9% at 8.0 passes) on
BOTH accuracy and cost, and matches the best schedule (k=40: 86.8% at 16.0 passes) at about a THIRD of its
passes -- by reorganizing only when the store is actually incoherent and skipping the passes a coherent
store does not need.

Wired into UnifiedMind as an OPT-IN coherence_floor (default None -> the original fixed schedule, so every
existing test is untouched). One subtlety that mattered: the gate must read a RESPONSIVE coherence window --
the default window=400 is too smooth to register a mid-stream shift (it left the gate stuck at the 44%
floor), so the gate reads coherence(window=check_every), checked EVERY observation, with a cooldown of
check_every//2. MIND-LEVEL verification (dim 512, check_every=40, 4 seeds) replicates the organizer result:
schedule 86.6% overall / 88.7% new at 16 passes; coherence gate 86.2% / 78.2% at 6.2 passes -- the SAME
overall accuracy at ~1/3 the passes (the gate is slightly less aggressive on the brand-new classes, but
stays well above the floor). The right floor is DATA-DEPENDENT (the coherence scale moves with dimension
and class structure), so it is a parameter, not a constant -- the kept caveat.

Tests: +2 (coherence gate reorganizes fewer times than the schedule at comparable accuracy and above the
never-reorganize floor; coherence_floor round-trips through save/reload). 705 -> 707.

## Tier-0 panel fixes: sublinear+calibrated recall, a procedure-matched null, rd-in-auto, calibration coverage (shipped)

The panel reviewed the live mind and asked first for FIXES to what the honesty/coherence work had just added,
not new features. Four landed.

1. recall_calibrated was sublinear-DEFEATING (Pharr). It did its own exact O(n) scan for the winner, throwing
away the HoloForest the recall path uses. Now the winner comes through recall() itself -- the sublinear forest
on a big store, the exact scan on a small one -- so honest abstention costs nothing the acceleration structure
did not already cost. Verified: on a 5000-item store the null fit + recall stays ~1s (forest), and
recall_calibrated now returns the SAME winner and score as recall().

2. The recall null was ANTI-CONSERVATIVE (Cranmer). It was a RecallNull fit on a capped SAMPLE of the
individuals (max-over-sample < max-over-all), which under-estimates the floor and inflates false matches.
Replaced with a PROCEDURE-MATCHED null: draw random unit queries, run them through the SAME recall path, and
take the score distribution that produces. Calibrated by construction (the null IS what noise scores under the
real procedure) and it inherits the procedure's sublinearity. The earlier "documented under-estimate" is gone,
not just noted.

3. A calibration COVERAGE diagnostic (Cranmer). calibration_report(n) draws pure-noise vectors and reports the
empirical false-alarm RATE -- the fraction whose p-value falls at or below each alpha -- on both readout paths.
MEASURED: at alpha 0.01/0.05/0.10/0.20 the prototype path fires at 0.008/0.059/0.098/0.200 and the individual
path at 0.008/0.062/0.093/0.200 -- it tracks alpha, so thresholding at alpha holds the false-alarm rate at
alpha. This is the radio-SETI / HEP coverage check, run on the mind's own geometry, and the proof that the
abstention the honesty layer added is trustworthy.

4. The mind's default save now uses B5 where it helps (Duda). B5's rate-distortion code was in the kernel but
only reachable via quant='rd'; the default 'auto' never asked for it. Now 'auto' itself tries the
rate-distortion code for LARGE low-rank 2D arrays (>= 256 rows), taking it only when it beats int8 and only
because it preserves cosines to 0.9999 (tighter than int8's ~0.998, so it fits auto's decision-safe contract).
Small minds are untouched (rd needs >= 256 rows); a 512x256 rank-8 array drops to ~200 bits/vector vs int8's
2048 (~10x) and round-trips at >= 0.999 row-cosine. The mind's default save now uses the rate-distortion code
automatically wherever the state is genuinely low-rank.

Tests: +4 (recall_calibrated agrees with recall and can abstain; recognition p-values are calibrated on noise
for both paths; auto picks rd for a large low-rank array and round-trips decision-safe; the mind's default save
round-trips classify-identical). 707 -> 711.

## Honesty reaches action; the SPRT's real regime; an auto coherence floor (shipped)

Three pieces: the flagship carries the calibrated-recognition idea from PERCEPTION into the decision brain
(Togelius's seat -- an agent that knows when it is guessing), plus two Tier-0 finishers.

1. Calibrated decide (Togelius). The creature brain already returns a `support` from value() -- the best cosine
the current state reaches against an action's prototypes -- and used a HAND-SET absolute `blind_floor` on it,
the same uncalibrated-threshold problem the coherence floor had. decide_confidence(state) now turns that raw
support into a false-alarm p-value via a PROCEDURE-MATCHED brain null (_brain_null: run the brain's own value()
on random unit states, take the best-support distribution -- calibrated by construction, value() used as a
black box). It returns (action, pvalue): p small means the brain has genuinely been somewhere like here and the
value estimate can be trusted; p large means it is guessing. decide(..., explore_if_unrecognized=alpha) makes
that actionable -- when p > alpha the value estimate is built on nothing, so take a safe random move among the
allowed actions instead of committing. MEASURED: a familiar state -> the learned-good action at p=0.000; a
never-seen state -> p=0.300; the brain null is calibrated on noise (false-alarm 0.066/0.113 at alpha 0.05/0.10);
explore_if_unrecognized=0.1 spreads a novel state's action ~uniformly (guessing) while a familiar state stays
locked on its trusted action. This is the honesty layer's RecallNull machinery, over the brain's experienced
states instead of perceptual prototypes -- and it replaces the hand-set blind_floor with a calibrated one.

2. The SPRT's real regime (a Tier-0 demo finisher). Wald's sequential test was decribed as saving ~half a fixed
window's samples, but the tour's distinct learned items are WELL-SEPARATED from noise, so stream_recognize
decides in ~1 sample -- correctly (decide as fast as the evidence allows). The sample-savings appear only when
the match and null densities OVERLAP -- a faint or drifting signal. MEASURED across overlap regimes (matched
error throughout): well-separated -> avg 1.1 samples, fixed-N needs 3 (~64% fewer); overlapping -> 2.2, fixed-N
4 (~46% fewer); heavy overlap -> 4.4, fixed-N 10 (~56% fewer). On REAL noisy cues the count adapts the same way
-- a clear sighting decides in 1, a borderline one spends up to 5. The honest framing: the SPRT is correctly
decisive when the evidence is strong; its efficiency is a property of the OVERLAP regime, not a number to force
on separated densities.

3. An auto-calibrated coherence floor (a Tier-0 finisher). The opt-in coherence gate fired below a hand-set
absolute level (0.65) that depends on dimension and structure. coherence_floor='auto' removes the absolute
level: track recent coherence and reorganize when it drops below ~90% of its own recent PEAK -- a RELATIVE
retention that transfers across data scales. Honestly, this trades an absolute parameter for a relative one,
not for nothing; but the relative one needs no per-dataset retuning. MEASURED (6 seeds, the antipodal-bimodal
shift stream): fixed schedule 81.1% at 11.0 passes; hand-set 0.65 -> 80.0% at 6.5; AUTO -> 82.1% at 6.7;
never-reorganize 48.5% (the floor). AUTO matches the hand-set floor's accuracy-vs-cost with no absolute
threshold. The 'auto' sentinel round-trips through save/load like a numeric floor; the baseline resets after a
reorganize because the store has changed.

Tests: +5 (decide_confidence low for familiar / high for novel; the brain recognition null is calibrated on
noise; explore_if_unrecognized guesses randomly on novel states and commits on familiar; the SPRT spends more
samples as densities overlap and beats fixed-N at matched error; the auto coherence floor matches the hand-set
floor without an absolute threshold and round-trips). 711 -> 716.

## The scan faculty: streaming detection + look-elsewhere control in one pass (shipped)

A1 from the revised backlog -- the last piece of the honesty arc, and the one Tier-1 item with zero code:
Siemion's seat asked for a single faculty that scans an astronomical channel count the way SETI must --
decide each channel as fast as its own evidence allows, AND control the trials factor across all of them.
`scan(channels, alpha, beta, fdr)` is pure assembly of parts already shipped: per channel, Wald's SPRT
(B3) decides MATCH/REJECT over that channel's stream of cues; then Benjamini-Hochberg/Yekutieli FDR
(`bh_fdr`) runs across the channels' calibrated p-values. A channel is a CONFIRMED detection only when the
SPRT decided MATCH *and* its p-value survives FDR -- the two disciplines combined. Each channel is a stream
bearing on one hypothesis (a frequency bin over time, a sky position, a recurring pattern).

The load-bearing detail was a calibration bug caught and fixed before shipping -- the engine's usual lesson.
The channel p-value needs a noise floor for the channel's mean score, and the obvious floor -- the existing
`_recognition_null` -- is WRONG twice over: it scores prototype ROWS (noise mean ~0.086), but recognize()
returns the max LABEL score (sub-prototypes aggregated); and recognize() first runs perceive(), which is NOT
the identity even for a raw vector -- it lifts the vector onto the encoder geometry, raising the max label
score of pure noise to ~0.117. Calibrating to either wrong floor made 69-76 of 80 pure-noise channels look
significant. The fix is a PROCEDURE-MATCHED floor (the recurring principle): run random unit vectors through
recognize() itself -- the exact path a channel cue takes, perceive and routing included -- and resample the
channel-mean null from that, by channel length. With the right floor, pure-noise channel p-values are uniform
again (~8 of 80 below 0.10, as a calibrated detector should be).

Measured (a weak/drifting target, 256-d, eight clear + four faint signal channels among eighty noise):

- **Detection.** All eight clear signal channels and all four faint ones detected; zero of eighty noise
  channels detected -- false-discovery proportion 0.00 at an FDR target of 0.10.
- **The look-elsewhere value.** Across the eighty pure-noise channels, naive per-channel thresholding at
  p<=0.10 flags ~11 false positives (as uniform p-values should); BH-FDR holds the detections to 0. That gap
  is exactly the trials factor the FDR controls.
- **The sequential value.** The SPRT spends 1.0 samples on a clear channel but ~1.8 on a faint one -- it
  decides as fast as each channel's own evidence allows, the Wald property, now per channel across a scan.
- **Deterministic** run-to-run (the null draws are seeded per channel length -- Macklin's tie-break rule).

Tests: +1 (scan detects signals, controls the look-elsewhere -- naive false positives cut by FDR -- spends
more SPRT samples on faint channels than clear, and is bit-identical run-to-run). 716 -> 717.

## Calibrated soft confidence for the resonator: a graded answer on approximate inputs (shipped)

A2 from the revised backlog -- Olshausen's resonator network with Cranmer's calibrated detector. The SBC
resonator already had a confidence signal: `verified`, True iff the recovered factors rebuild the product
EXACTLY (precision ~1.0). That certificate is perfect on exact products and uselessly brittle on approximate
ones: the moment the input is a noisy bind, exact reconstruction fails and `verified` goes False even when the
resonator found exactly the right factors. `resonator_confidence` (exposed through `decompose_structure(...,
confidence=True)` and `factor_composite(..., confidence=True)`) adds the graded answer -- (picks, verified,
agreement, pvalue), where `agreement` is the fraction of blocks the factors rebuild (the soft version of the
boolean, which is agreement==1.0) and `pvalue` is a calibrated false-alarm probability.

The calibration is the whole subtlety, and it is the same lesson scan taught. The obvious null -- the agreement
RANDOM PICKS would score -- assumes a factorization matching ~1/L of the blocks by chance (~0.06 here). But the
resonator OPTIMISES reconstruction, so on pure noise it still manufactures ~0.27 agreement, far above that
chance line. Calibrating to the random-picks null therefore rates pure noise as a near-certain detection
(measured p ~ 0.02-0.003). The honest null is PROCEDURE-MATCHED: the agreement the SAME resonator reaches on
STRUCTURELESS input (random SBCs through the real factorizer), which includes its overfitting. That null is a
property of the search configuration -- stable across different random codebooks of the same shape (mean
0.262-0.269 over three) -- so it is cached per codebook set; the first confidence call pays for it, the rest
are free.

Measured (B=16, L=16, three factors, eight atoms each; true factors (2,5,1)):

- **The rescue.** Corrupting one to five of the sixteen product blocks, the resonator still recovers the true
  factors every time -- but `verified` is True only at zero corruption, False for all the rest. The calibrated
  p holds at 0.010 (trust) straight through, exactly where the boolean fails. Agreement falls smoothly 1.00 ->
  0.94 -> 0.88 -> 0.81 -> 0.75 -> 0.69 as the blocks corrupt.
- **Abstention on noise, calibrated.** Over eighty pure-noise products the median p is 0.84 -- it abstains.
  Its false-alarm rate is conservative (3 of 80 below p=0.10, against a nominal 8): block agreement is discrete,
  so the p-value is stepwise, and conservative is the safe direction for a detector.
- **The kept lesson on one noise product** (agreement 0.250): the random-picks null gives p=0.022 (false
  confidence); the procedure-matched null gives p=0.842 (abstains). Same principle that fixed scan's floor.

Backward-compatible: `confidence` defaults False, so `decompose_structure` and `factor_composite` return exactly
what they did. Tests: +1 (the rescue holds through three corrupted blocks with verified False; noise abstains
with a controlled false-alarm rate). 717 -> 718.

## Pluggable assembly energy + structure-compare: the Rosetta move and a fold comparator (shipped)

A3 from the revised backlog -- the Baker seat, and the first Tier-2 item that was genuinely unbuilt. `assemble`
already cast fragment assembly as the same min-cost flow search the maze solver runs, but with one hardcoded
energy: Hamming mismatch, every disagreement costing the same. That is the documented stand-in, not a Rosetta
score. Two additions make it the real thing.

**Pluggable energy.** `assemble(target, library, ..., energy=callable)` (and `assemble_optimal_energy(...,
energy=)`, and the mind's `assemble(..., energy=)`) lets the caller supply any non-negative placement energy;
it defaults to the Hamming stand-in, so every existing call is unchanged. The point is the Rosetta move -- not
every substitution costs the same. The energy is rounded to integer hops for the relay-encoded trellis (supply
an integer energy for an exact search; the reported energy is the exact unrounded sum), and the flow search
still finds the GLOBAL optimum under whatever energy it is given (it matches the Viterbi DP). Measured: with a
toy substitution matrix where same-group swaps cost 1 and cross-group swaps cost 4, the target "EAAE" assembles
to "BABE" under Hamming (three plain mismatches, cost 3) but to "EEEE" under substitution (cost 4) -- because
"BABE"'s three mismatches are cross-group B-for-vowel swaps that cost 12 under the substitution energy. Each
assembly is the unique global optimum under its OWN energy, both matching the DP.

**Structure-compare.** `compare_structures(a, b)` superposes two assembled structures and reads their overlap
two ways: `placement_overlap`, the exact overlap coefficient of the (position, fragment) sets (the shared
local motifs of two folds); and `holographic_overlap`, the SAME quantity read from the SUPERPOSITION via
consolidation -- stack both structures' role-bound (pos (x) frag) vectors and take the effective rank (the
consolidation SVD), where a shared placement is the same vector so the combined rank COLLAPSES by the number
shared, giving (rank_A + rank_B - rank_AB)/min as the overlap. On clean structures the two reads agree exactly
(measured 1.00/1.00 identical, 0.33/0.33 sharing one of three, 0.00/0.00 disjoint) -- the holographic read
validated against the exact count, and the form you use when a structure is only available as a hypervector.

A tie-break caught the test first, which is on-theme for the Macklin determinism work next: the original test
instance had a Hamming TIE (two assemblies both cost 2), and which one the flow search returned was sensitive to
suite ordering -- it passed alone and failed in the full run. The fix was not to special-case the tie but to
pick an instance with UNIQUE optima (enumerated to confirm), so the chosen assembly is deterministic. Atom names
in the comparator are hashed deterministically (Python's str hash is process-randomised) for the same reason.

Tests: +1 (the substitution energy changes the optimum and each matches the DP; the holographic overlap matches
the exact placement overlap on identical / partial / disjoint structures; deterministic). 718 -> 719.

## One iterate-a-projection engine + a determinism audit of the calibrated paths (shipped)

A4 from the revised backlog -- the Macklin seat, two pieces.

**One engine under three faculties.** Macklin's observation was that the resonator's alternating cleanup, the
PnP/RED denoise loop, and a position-based-dynamics constraint sweep are the SAME object he builds: project onto
each constraint in turn until they jointly hold. `project_onto_constraints(x, projections, iters, tol, omega)`
(holographic_denoise, exposed as a mind faculty) is that engine -- sweep a list of projections, optionally
under-relaxed (omega<1, PBD's stability trick), with early-stop on convergence. The unification is made
load-bearing, not just claimed: `pnp_restore` now LITERALLY calls it (two projections -- a data-fidelity step
then the denoiser), bit-for-bit identical to before (the denoise suite passes unchanged). Demonstrated as three
instances of the one engine:

- **POCS.** Alternating projection onto two subspaces (sharing a 1-D direction) converges in 29 sweeps to a
  point in their intersection, off-axis residual ~5e-13 -- von Neumann's theorem, and it matches the exact
  projection onto the intersection.
- **A resonator.** Given factor-cleanup projections (unbind the others, snap to a codebook) the SAME engine
  recovers a bound product's factors at reconstruction cosine 1.000 -- WITH restarts. A single restart converges
  to a spurious fixed point (recovered the wrong factors): an honest reminder that the real resonator's restarts
  are not decoration, they are how alternating projection escapes the non-convexity.
- **PnP.** `pnp_restore` == `project_onto_constraints([data_fidelity, denoiser])`, bit-identical.

**The determinism audit (the heart of the Macklin ask), expanded to the new paths.** The assemble tie-break two
items ago -- a test that passed alone and failed under suite ordering -- is the class of bug this audit exists to
catch. Every calibrated/null path added this program (recall_calibrated, decide_confidence, the auto coherence
floor, scan, resonator confidence, compare_structures) is now run TWICE on a freshly rebuilt setup -- so its null
is RECOMPUTED, not reused from cache -- with numpy's GLOBAL RNG scrambled in between. All return bit-identical:
the paths draw only from their own seeded `default_rng(self.seed)`, never the global stream (the thing that, had
it leaked in, would have made results depend on whatever ran before). A clean result, but the value is the
guarantee it locks in -- and the cache-clear in the resonator case makes it test the null COMPUTATION, not just
the cache. Determinism here is not luck; it is audited.

Tests: +2 (the engine as POCS / resonator / PnP, all three matching; and the determinism audit -- six calibrated
paths bit-identical across a fresh rebuild with the global RNG scrambled). 719 -> 721.

## The inverse problem through the mind: inpaint an erased plate, and validate the noise estimate (shipped)

A5 from the revised backlog -- the Milanfar/Ozcan seats, and a DEMO not a build (the re-audit found the machinery
already wired). The genuine work was integration: the PnP/RED loop and the noise-adaptive denoiser were callable,
but there was no clean mind entry for "restore a degraded measurement" -- a caller had to hand-build the forward
and adjoint operators every time. `restore(y, mask=..., samples=...)` is that entry: pass a 0/1 mask and the
forward operator and its transpose are filled in (a diagonal mask is its own transpose), the prior is THIS mind's
adaptive manifold denoiser fit from `samples`. It does NOT reimplement anything -- it delegates to
`denoise(method='pnp')`, so the inverse problem is one mind call built on the existing loop, not a silo.

Measured end to end, through the mind, on an erased archive plate:

- The mind's OWN splat archive holds a 40-image low-rank gallery (recover round-trips a plate at 29.9 dB).
- One plate has a 5x5 block (25 of 256 px) erased plus light noise -- the degraded measurement.
- A SINGLE adaptive denoise of the masked input reaches 19.3 dB; `restore` (the PnP/RED LOOP) reaches **38.5 dB**
  -- the loop beats the one-shot by **19 dB**. The reason is exactly Milanfar's: the one-shot projection is
  dragged toward zero by the erased pixels, while the loop holds the observed pixels to the measurement and fills
  only the erased ones from the manifold. Reconstruction-under-erasure as a SOLVED inverse problem, in the mind.

- **Noise-estimate validation, with its kept negative.** Donoho's MAD estimate is accurate at moderate-to-high
  noise (true 0.20 -> estimated 0.221; true 0.10 -> 0.126) and the adaptive denoiser's `sigma=None` self-estimate
  matches SUPPLYING the true sigma (identical PSNR-to-clean) -- no oracle needed, which is the whole point of the
  adaptive path. The kept negative: at LOW noise it OVER-estimates (true 0.02 -> 0.061), because the estimate
  assumes the clean signal is smoother than the noise, and a textured low-rank image's own finest detail inflates
  the MAD. It is honest where it holds and honest where it does not.

Tests: +1 (restore inpaints the erased plate through the mind and beats the one-shot by >5 dB, the archive
round-trips the gallery, and the sigma estimate is accurate at the tested level with sigma=None matching the
truth). 721 -> 722.

## Capacity / SNR vs the cliff, and calibration coverage vs load (shipped)

A6 from the revised backlog -- the Plate and Cranmer seats, and the ONE genuinely new diagnostic the re-audit
identified (everything else on the list was either built or a demo). It answers two questions about the same
store geometry in one report, `capacity_report`.

**Where the store sits relative to the noise-wins cliff (Plate, HRR capacity theory).** Random unit vectors in
D dimensions have pairwise cosine ~N(0, 1/D), so a random query's BEST cosine to N stored rows -- the noise
floor -- is the max of N such, ~sqrt(2 ln N / D) by extreme-value theory. A genuine match sits at a much higher
cosine. The report reads off:

- `dprime` = (match - floor_mean) / floor_std: the SNR, in noise-sigmas, that a real match clears the crosstalk.
  Measured: a roomy store (D=512, 8 classes) sits at d'=23; a loaded one (D=64, 20 classes) at d'=6.6 -- the
  diagnostic CAPTURES load, the loaded store visibly closer to the cliff.
- the measured floor vs the HRR bound: 0.063 vs 0.090 (roomy), 0.229 vs 0.306 (loaded) -- the same order, the
  measured floor sitting a bit BELOW the asymptotic bound at small N (honest: sqrt(2 ln N) overestimates the
  expected max for small N, so the store is slightly safer than the bound says). The geometry follows theory.
- `headroom` = n_cliff / N where n_cliff = exp(D match^2 / 2): the roomy store could grow ~10^50x before the
  rising floor reaches the match level; the loaded one only ~10^5x. That enormous high-D headroom IS the point
  of distributed codes, now a live readout instead of a slogan.

**Whether calibrated coverage holds as the store GROWS (Cranmer).** Tier 0 validated the false-alarm rate at a
FIXED store; the open question was whether p<=alpha still holds the rate at alpha as N grows and the floor rises.
The report builds random codebooks of increasing size (64 -> 256 -> 1024) in the mind's D, fits the procedure-
matched null on each, and measures the false-alarm rate: it stays ~alpha (0.006 / 0.04 / 0.062 at alpha=0.05) --
the null re-fits to the rising floor, so the look-elsewhere discipline is load-robust. Materially above alpha at
the largest N would have meant the null was under-sampling the bigger store; it does not.

The diagnostic is the capacity complement to `calibration_report` (fixed-store coverage) and `resolution_profile`,
and is deterministic (seeded by the mind -- it passed the A4 audit's bit-identical bar by construction).

Tests: +1 (the operating point ranks roomy above loaded, the measured floor tracks the HRR bound, headroom is
larger for the high-D store, coverage holds <=~alpha at every load, and the report is bit-identical run-to-run).
722 -> 723.

## A spectral/audio FHRR modality, and dynamics on audio frames -- closing the market-returns loop (shipped)

A7 from the revised backlog -- Puckette and Stam, and the close of a loop the dynamics work (B4) deliberately
left open. B4's learned propagator (`learn_dynamics`: state(t+1) ~ bind(U, state(t)), a per-frequency transfer
in HRR's Fourier domain) only TIED a trivial mean predictor on real market RETURNS, the correct result kept on
record -- near-efficient-market returns have almost no linear structure for a fixed operator to exploit. The
honest test was always going to be a signal that DOES have linear structure, and audio is the canonical one.

**The audio modality (Puckette).** `spectral_encode(frame)` is the phase vocoder in the complex domain: a real
frame's DFT splits into a unit-magnitude PHASOR per bin (the phase -- an FHRR vector, every component on the
unit circle, so it binds / bundles / recalls in `high_capacity_memory` like any minted atom) and a MAGNITUDE
per bin (the timbre). Silent bins take phasor 1 by convention so the phasor vector is unit-magnitude EVERYWHERE,
a valid FHRR vector rather than a spectrum with holes. `spectral_decode` re-attaches the magnitudes and inverts
the DFT, exact to 5e-14 (the phasor key plus the magnitude lose nothing). Encoding several BROADBAND sounds
(fundamental plus harmonics plus a little noise) and cramming them into one phasor trace, each recalls by its
key cleanly (3/3, off-diagonal phasor similarity ~0). **Negative on record:** a pure TONE is too sparse for
phase alone to separate -- its silent bins dominate the unit-phasor encoding (three distinct tones sit at
fhrr_sim ~0.99, indistinguishable), so for sparse sounds the MAGNITUDE carries identity, not the phase. The
modality is honest about which half of the (phasor, magnitude) split is discriminative for which kind of sound.

**Dynamics on audio (Stam, the proving ground).** A sustained multi-sinusoid framed with a hop evolves frame to
frame by exactly the per-bin phase advance the propagator is built to learn (the same advance `spectral_encode`'s
phasors carry -- the operator and the encoding are two faces of one spectral structure). Through `learn_dynamics`,
held-out one-step prediction error is 0.001, against persistence 1.64 (it ignores the advance, so a hop that
moves the phase a half-turn makes the last frame nearly anti-correlated) and mean 1.00 (it averages the
oscillation away). The propagator beats both by three orders of magnitude -- audio HAS the linear structure
market returns lacked, the loop closed with a positive set against the kept negative. On a HARDER case
(non-integer-cycle frequencies plus noise -> spectral leakage and a corrupted transfer) the error rises to 0.169
-- approximate, not exact, the honest cost of a fixed operator on non-stationary input -- but still beats
persistence (1.59) and mean (1.00) by ~6-10x. And the content-addressable round-trip holds regardless of signal:
forward four frames then back four returns the start at cosine 1.0.

Tests: +1 (the modality round-trips exactly with unit-magnitude phasors; broadband sounds recall by key from one
shared FHRR trace; dynamics through the mind beats persistence and mean on a sustained tone; the propagator's
predicted next frame, run back through the modality, matches the true next frame's encoding -- the two faculties
wired; and the forward-then-back round-trip returns the start). 723 -> 724.

## learn_dynamics on a fluid field -- the second positive against the market negative (shipped)

A8 from the revised backlog -- the Stam seat, and the third validated regime for the B4 dynamics operator after
audio (A7) and the kept market negative. Stam's "Stable Fluids" and his FFT fluid solver work on a periodic
(toroidal) domain, doing the hard step in Fourier space -- the same FFT-on-a-torus the engine's bind already
is. A passive scalar's LINEAR advection-diffusion step is exact there: in Fourier each mode k just rotates
(advection: phase -2*pi*k*shift/N) and decays (diffusion: e^{-nu*k^2}), i.e. a per-bin complex transfer --
precisely the operator `learn_dynamics` fits.

**The clean Stam case.** A bump plus two low modes, advected on a 256-point torus, framed into a sequence of
fields. Through `m.learn_dynamics`, held-out one-step prediction error is 0.011, against persistence 0.34 (the
field has moved, so the old field is stale) and mean 1.15 (averaging the moving structure away). The propagator
recovers the advection-diffusion operator almost exactly -- a fluid field HAS the linear structure a fixed bind
operator exploits, where near-efficient-market returns did not. Two further properties, both measured:

- **Surrogate solver.** The learned operator rolls out 8 steps from a single field and tracks the true
  simulation to ~3.5% relative error -- learn the fluid operator from a handful of frames, then simulate
  forward with one bind per step.
- **Content-addressable trajectory.** The operator's own forward-k-then-back-k returns the start at cosine 1.0
  (even with diffusion: the Wiener-regularised inverse exactly undoes the operator's own forward map). The
  earlier confusion -- recalling the TRUE future field gave cosine ~0 -- was the honest distinction that an
  imperfect learned operator inverts its OWN trajectory, not the ground truth it only approximates.

**The honest limit, kept on record.** A NONLINEAR Burgers field (u_t + u u_x = nu u_xx) forms shocks -- the wave
steepens, energy cascades to high modes -- and no single fixed LINEAR operator captures that. Measured: the
propagator does WORSE than persistence on a shock-forming Burgers field (error 0.054 vs 0.006; worse still,
0.125 vs 0.015, for a stronger shock). The propagator is for linear or linearizable dynamics; nonlinear flow
with shock formation is exactly where it fails, and that negative sits beside the audio and linear-fluid wins.

The faculty's own docstring now records all three regimes (audio, linear fluid, the Burgers limit) so the
measured boundary travels with the code. This is a validation of existing machinery through the mind, not a new
build -- the same shape as the PnP restoration demo.

Tests: +1 (linear advection-diffusion beats persistence and mean through the mind; the operator rolls out 8
steps as a surrogate within 10%; the forward-then-back round-trip returns the start; and a shock-forming Burgers
field is the honest case where the propagator loses to persistence). 724 -> 725.

## Multi-terminal network design -- the Tokyo-rail Physarum, as a typed graph-memory (shipped)

A9 from the revised backlog -- the Adamatzky seat, and a genuine BUILD (not a validation): the multi-terminal
generalisation of the single-source `solve_maze` flow solver. Tero et al. (2010, *Rules for Biologically
Inspired Adaptive Network Design*) showed Physarum grows a network connecting many food sources that rivals the
Tokyo rail network on cost, efficiency, and fault tolerance. `tero_network` reproduces that on a graph: drive
flow between ALL pairs of terminals, and the tubes that survive form the connecting network.

**The mechanism, with two real improvements over a naive port.** (1) The Laplacian depends only on the
conductivities, so every terminal pair is solved in ONE multi-right-hand-side factorisation per step (A P = B
with a column per pair), not one solve per pair. (2) Summing raw flux over pairs pinned every tube open (the
saturating response f = q^mu/(1+q^mu) hits 1 when flux is large, and most edges carry large summed flux). The
fix is to adapt each tube toward the MEAN saturated response OVER pairs -- i.e. toward how many terminal pairs
route through it -- so trunk tubes thicken and unused tubes die. Without it the network is the whole grid.

**The cost / fault-tolerance trade-off, measured (7x7 grid, 5 terminals, MST baseline 24 hops).** The `mu`
feedback tunes exactly what Tero describes:
- mu=4: a near-minimal Steiner TREE -- 21 edges, 0 cycles, and SHORTER than the 24-hop terminal-MST (the flow
  shares trunk segments through Steiner cells the pairwise-MST cannot), a genuine Steiner approximation.
- mu=2: a fault-tolerant network -- 36 edges with 4 redundant loops (alternate routes that survive an edge cut)
  at modest extra cost.
- mu<=1: the full redundant mesh.

**Wired in as a B7 typed structure.** `design_network` returns the network BOTH as raw edges and as a typed
graph-memory: a StructureRecipe building M = superpose over edges of bind(node_u, node_v) -- the same
construction `chain_structure` uses for a linked list. That realise()s to one hypervector, and the engine's own
unbind + cleanup recalls a node's neighbours: unbind M by a node atom, snap the result to the node codebook,
and the true network-neighbours come back above every non-neighbour (node (0,0) -> {(0,1),(1,0)} at similarity
0.15 vs 0.06 for non-neighbours). The network is not a side artefact; it is an object the mind can store, query,
and decode like any other structure.

A note kept honest: the dense Laplacian solve is O(n^3) per step, so this is for modest graphs (tens of nodes),
the regime where the flow model's interpretability is the point; a sparse/conjugate-gradient solve would be the
scaling path if needed.

Tests: +1 (high mu gives a tree no longer than the MST with zero cycles; low mu gives a strictly larger mesh
with cycles; the default network connects every terminal and its typed graph-memory recalls a terminal's actual
neighbours above all non-neighbours). 725 -> 726.

## Cross-modal recall: the exact image archive, reachable from the mind, queried by description (shipped)

A10 from the revised backlog -- the Ozcan seat. The re-audit found the cross-modal machinery was ALREADY built
in `HolographicArchive` (the DCT/Walsh-Hadamard plate store): `add(image, tags=[...], nums={...})` attaches a
hypervector address (a bundle of tag atoms, plus bind(attr, scalar.encode(v)) for numeric attributes), and
`recall_by_tags(words=[...])` returns the best-matching image by cosine of the query address to the stored ones
-- Ozcan's describe-then-retrieve. The gap was pure integration: only the LOSSY `splat_archive` was wired into
the mind; the EXACT, tag-addressable archive was unreachable.

**Wired in** as `image_archive`. The mind now has both archives and the right one for the job: splats for a
compact resolution-independent bundle, plates for bit-exact recall AND cross-modal addressing. Measured through
the mind on a 4-image gallery:

- **Exact recovery** at full keep (all DCT coefficients): max pixel error 6e-15 -- a single adjoint per channel
  inverts the superposition exactly. (Fewer coefficients trade exactness for compression, the archive's other
  mode.)
- **Tag -> image**, soft-AND over the query: `['round','large']` returns the ring, `['round','small']` the
  circle -- the image matching the MOST query tags wins, because each shared tag adds its atom's correlation to
  the address match.
- **Robust under damage**: describe-then-retrieve still reconstructs the gradient from `['smooth']` at 0.002
  error with 40% of the plate ERASED -- the joint masked recovery the archive is built for, now driven by a
  text query instead of a degraded picture.

**The improvement: the reverse direction.** The archive could go tag -> image but not image -> tags. Added
`tags_of(i, candidates)`: rank a candidate vocabulary by each word's correlation to stored image i's address --
the description the archive would give the image. 'ring' comes back as round + large (0.72 each) over smooth
(0.01). Cross-modal recall is now bidirectional: describe to retrieve, or retrieve to describe.

Tests: +1 (the mind's image_archive recovers every image exactly; tag queries return the right image including
soft-AND; the reverse ranks an image's own tags on top; and describe-then-retrieve survives 40% plate erasure).
726 -> 727.

## Generation over a composed subspace -- the B10 sampler's interesting regime (shipped)

A11 from the revised backlog -- the Eno seat, and the regime B10 (generative denoising) explicitly flagged as
the one worth reaching. B10's sampler runs the denoiser BACKWARDS from pure noise -- anneal beta up, injected
noise down, iterate the cleanup -- and walks a random vector onto the signal manifold, a sample generated by
denoising. Its KEPT NEGATIVE: over a BARE codebook it converges to a stored atom (a degenerate sampler that can
only return what is already in the box). The interesting regime, per the docstring, is a COMPOSED manifold.

**The composed-manifold denoiser.** A valid composed structure is a bundle over slots of bind(role, filler) for
fillers drawn from a vocabulary -- one of V^S structures, far too many to enumerate as a codebook. So the
denoiser is slot-wise: for each role, unbind the slot, dense_cleanup its filler toward the vocabulary, rebind,
then bundle and renormalise. `generate_structure` drops that projection into B10's annealed diffusion in place
of the bare-codebook cleanup, so the random start walks onto the manifold of role-filler STRUCTURES instead of
collapsing to an atom. (The slot-wise projection is itself an instance of 'iterate a projection' -- the same
shape as the resonator and the PnP loop.)

**Measured through the mind** (3 slots, 6 fillers, V^S = 216 possible structures, 1024-d):
- **Diversity:** 10 distinct valid structures from 10 seeds -- the sampler explores the composition space, it
  does not fall into one attractor.
- **Validity by construction:** re-encoding each generated vector's decoded fillers reproduces it at cosine
  1.0 -- the output genuinely IS a composition (bundle of role-bound fillers), and every slot unbinds to a
  vocabulary atom.
- **Not a stored atom:** the generated structure is nearly orthogonal to every single filler (max |cos| < 0.4)
  -- a composition, not a verbatim atom.
- **The kept negative confirmed for contrast:** `generate_vector` over the bare filler codebook returns a
  single stored atom at cosine ~1.0 (degenerate) -- exactly what generating over the composed manifold avoids.

So generation and denoising remain the same operation (Eno's process-not-object framing), and pointing that
operation at the composition manifold turns a degenerate atom-recaller into a generator of novel-but-valid
structure -- the bar B10 set, now cleared.

Tests: +1 (10 seeds each produce a valid composition that re-encodes to itself and is orthogonal to every bare
filler; at least four are distinct; and bare-codebook generation collapses to a stored atom). 727 -> 728.

## A fractal scene from a single seed vector -- one kernel, repeated to depth (shipped)

A12 from the revised backlog -- the Quilez seat, the demoscene aesthetic stated in the engine's terms: maximal
richness from a tiny deterministic kernel, infinite detail by recursion. The existing `nested_scene_structure`
does scenes-of-scenes for one level; the ask was a single seed vector driving one kernel repeated to ARBITRARY
depth, with `fractal_dimension` reported.

**The kernel lives in one vector.** `fractal_seed(offsets, scale)` encodes a fractal kernel -- N copies of the
plane, each contracted by `scale` and translated to an offset -- as a holographic bundle: sum over copies of
bind(pos_role, grid_atom[offset]) plus bind(scale_role, scale_atom). The whole generator is carried in the
geometry of one hypervector. `fractal_scene(seed, depth)` decodes it with pure VSA -- unbind the position role
and threshold the grid atoms to recover WHICH cells are offsets, unbind the scale role and clean up to recover
the scale -- then expands that one kernel to `depth` (each level places a contracted copy of the whole scene,
N^depth points), and reports the box-counting dimension. This is an IFS whose maps are read out of a vector.

**Measured through the mind:**
- Seed A (a Sierpinski kernel: 3 copies at scale 1/2) decodes to exactly 3 offsets and scale 0.5, and its
  expanded scene has box-dimension 1.57 against the self-similar log3/log2 = 1.585.
- Seed B (5 copies at scale 1/3) decodes to 5 offsets and scale 1/3, box-dimension 1.51 against log5/log3 =
  1.465.
- The two seeds give DISTINCT measured dimensions -- the seed genuinely drives the scene -- and expansion is
  deterministic (same seed, identical points). The small box-counting gaps (0.01, 0.04) are the expected
  finite-sample / finite-range bias of box counting, not error in the construction.

So a single vector encodes a generator, the engine decodes it, and one kernel repeated to depth yields a
self-similar scene of a predictable, measured fractal dimension -- the SDF/demoscene move (one kernel, domain
repetition, infinite detail, deterministic from a seed) on the VSA substrate. An honest scope note: the offsets
are snapped to a grid codebook so they decode by exact cleanup, and the dimension is set by N and scale (both
recovered exactly), not by sub-grid offset precision.

Tests: +1 (a Sierpinski seed decodes to 3 copies at scale 1/2 with box-dimension near log3/log2; a 5-copy
scale-1/3 seed lands near log5/log3; the two dimensions are distinct; and expansion is deterministic). 728 -> 729.

## Anisotropic splats and a 3-D extension -- the real 3DGS primitive, fit from scratch (shipped)

A13 from the revised backlog -- the Drettakis seat, and a deliberately-deferred Tier-4 scope: the splat work
(B8) used ISOTROPIC (circular) Gaussians by explicit choice; 3D Gaussian Splatting's actual primitive is an
ANISOTROPIC, oriented Gaussian with a full covariance, fit by differentiable optimisation. A13 builds that core
in NumPy, true to the project's minimal-framework rule (analytical gradients + a tiny hand-written Adam, no
autodiff library).

**The primitive and the fit.** Each splat is (center, amplitude, L), L the lower-triangular Cholesky factor of
the INVERSE covariance, so the Gaussian is amp * exp(-0.5 ||L^T (x - center)||^2) and L lower-triangular keeps
the precision positive-definite for free. `aniso_fit` warm-starts from the isotropic matching pursuit (so the
covariances only have to specialise), then descends the reconstruction MSE with analytical gradients for the
amplitude, center, and L. The whole thing is dimension-general: a 2-D image and a 3-D volume share one fit.
Wired in as `splat_aniso` (the anisotropic, n-D twin of `splat_field`).

**Measured through the mind** -- anisotropy is decisive exactly where structure is oriented:
- 2-D, two elongated oriented ridges, K=4: isotropic ~18 dB -> anisotropic ~64 dB. A circular Gaussian cannot
  match an elongated ridge; one aligned anisotropic splat does, and four nearly reconstruct two ridges.
- 3-D, an elongated ellipsoid, K=3: isotropic ~24 dB -> anisotropic ~61 dB. An ellipsoid IS one anisotropic
  Gaussian.
- The learned splats are genuinely anisotropic (inverse-covariance eigenvalue ratio > 3 on the ridges), and
  re-rendering the returned (center, amp, L) code reproduces the fit.

**KEPT NEGATIVE / honest scope.** The loss is non-convex, so the fit finds a LOCAL optimum: more splats do NOT
help monotonically -- a clean K=4 fit (66 dB) beat a messier K=8 one (52 dB) in testing -- and the result
depends on the isotropic warm start. And this is the from-scratch CORE of 3DGS only: no tile rasteriser, no
spherical-harmonic view-dependent colour, no GPU speed; it runs on small fields where the optimisation is the
point, not a real-time renderer. That boundary is the honest Tier-4 label -- the primitive and its
differentiable fit, not the production system.

Tests: +1 (anisotropic beats the isotropic warm start by >15 dB in both 2-D and 3-D on oriented structure, the
splats are measurably anisotropic, and the splat code re-renders to the fit). 729 -> 730.

## Tensor-product / tensor-train (MPS) bind vs HRR -- the capacity comparison (shipped)

A14 from the revised backlog -- the Stoudenmire seat, the heaviest and most speculative item, and the LAST of
the program (A1-A14 all delivered). HRR's bind(a,b) = circular convolution is a compressed projection of
Smolensky's tensor-product binding a (X) b; tensor networks (MPS / matrix-product states) interpolate by
truncating the tensor product to a low bond rank. `tensor_bind` builds the uncompressed outer-product bind and
its rank-r 2-site MPS truncation so all three points on the rank spectrum can be measured against the engine's
circular convolution. The result is bucketed honestly -- a real but storage-bought capability, not a free win.

**The mechanism.** The outer-product bundle M = sum_i outer(k_i, v_i) recalls value_i as M^T k_i =
sum_j (k_j . k_i) v_j. The crosstalk coefficient is the key inner product (~1/sqrt(D) for random keys), so the
crosstalk is suppressed by 1/sqrt(D) -- whereas HRR's unbind produces full-magnitude pseudo-random crosstalk
vectors. That single difference is why the tensor product recalls so much better at a fixed load.

**Measured (D=128) against HRR:**
- At a fixed LOAD M=16, recall cosine is 0.25 (HRR) vs 0.95 (tensor product) -- the tensor product is far more
  faithful, because it spends D^2 = 16384 numbers against HRR's D = 128.
- With ORTHOGONAL keys at M = D, the tensor product is EXACT (recall 1.000) -- the key inner products are
  Kronecker deltas, zero crosstalk -- where circular convolution cannot be (0.10). A genuine qualitative
  advantage for structured keys.
- A rank-8 binding matrix (values in a rank-8 subspace) MPS-truncates LOSSLESSLY: recall 0.862 preserved while
  storage drops from 16384 to 2048 numbers (8x). Truncating below the true rank (rank 4) is lossy (0.66). This
  is the tensor-network's real capability -- exploit low rank / low entanglement.

**KEPT NEGATIVE (the honest bucket).** At a fixed RECALL THRESHOLD the capacity-per-stored-number of HRR and
the tensor product is the SAME (both ~ (1-t^2)/(t^2 D) per number) -- HRR's compression gives up nothing on
that frontier; it simply chooses the compact, low-absolute-capacity end while the tensor product chooses the
high-fidelity, high-storage end. And a generic (full-rank) binding cannot be MPS-compressed without losing
recall, and even the lossless low-rank form (>= 2D numbers) still costs more than HRR's D. So the tensor /
tensor-train bind is a DIFFERENT point on the storage-vs-fidelity tradeoff -- worth it when you need exact
high-capacity structured recall and can afford the storage, or when an existing bound tensor is genuinely
low-rank -- not a way to beat HRR's efficiency on generic bindings. That is the correct, measured place for it.

Tests: +1 (the tensor product beats HRR at fixed load and is exact for orthogonal keys; the MPS truncation is
lossless at the true rank and lossy below it, at a fraction of the full storage but still above HRR's). 730 -> 731.

## Integrating Path D -- federation and width, the "as above, so below" arc (shipped)

A parallel investigation ("Path D": computing and storing INSIDE the holographic space) was merged in from a
separate session. It arrived as a self-contained bundle -- two new modules, plus experiments, figures, and the
frontier-program / dataset-benchmark / distribution-candidate docs -- and it touched none of the existing
engine code, so the integration was additive: the bundle lives under `path_d/`, the two reusable modules were
hoisted to the top level and (the real work) WIRED INTO `UnifiedMind` as faculties, the same discipline every
other module shipped under. They imported cleanly against the frozen kernel with zero API drift, and both their
selftests and the headline experiments reproduced on this tree before anything was written down.

**The through-line Path D found.** One D-dimensional vector holds only ~0.1 x D items faithfully (~0.02 x D for
*continuous* compute with no cleanup to absorb crosstalk), and that budget is CONSERVED -- you do not beat it by
encoding harder, you FEDERATE: more vectors = more total dimensions = more capacity, coordinated by a thin layer.
The same move recurs at every scale (storage, lookup, resilience, the neural-network forward pass), which is the
engine's recurring lesson seen one rung up: the within-vector property becomes the across-shard property by the
same linearity.

**Two faculties wired and measured through the mind:**
- `storage_array` (holographic_array.HoloArray) -- a federated, RAID-style symbol store. A shard is a running
  sum, so a PARITY shard is the real-valued sibling of a fountain XOR droplet and reconstructs a lost shard
  EXACTLY by subtraction. Measured: 150 symbols at D=1024 auto-grow to 3 shards at ~0.89 recall (one shard would
  have cliffed); lose a shard and parity restores recall exactly (0.89), where a zeroed shard drops it to ~0.55.
  KEPT NEGATIVE / information floor: `n_parity` parity survives at most `n_parity` losses -- it cannot recover
  more than it has parity, mirroring the fountain's "too few droplets -> nothing."
- `superpose_compute` (holographic_superposed) -- the WIDTH faculty: evaluate K computations at once inside one
  vector (Kanerva/Kleyko computing-in-superposition), the parallel-readout complement to the mind's DEPTH side
  (recursion, `peel` traversal, the inception depth law). Measured: a single keyed item recovers EXACTLY with a
  unitary key (cosine 1.0); six candidates packed into one vector are scored in parallel and the winner is
  resolved cleanup-gated against a codebook. KEPT NEGATIVE / the conservation law: recovery fidelity decays with
  width (mean cosine ~0.50 at K=4 -> ~0.12 at K=64) -- width is bounded, and you buy more by spending DEPTH, not
  by widening one flat bundle.

The bundle's own headline (reproduced here) is the distributed forward pass: a single weight-vector is faithful
to 16 classes (~0.02 x D), and federating to 8 shards holds 96 (~6x) -- the same federation move that fixes
storage, applied to the matmul. The bundle also carries the frontier-program and dataset-benchmark docs (now
under `path_d/docs/`) and a second lever, RNS-phasor arithmetic, that lives in the Path D experiments but is not
yet an engine module -- a clear, honestly-labelled next step rather than a claim.

Tests: +3 (one mind-level integration test wiring both faculties -- federation grows shards and parity restores
a lost one exactly; superpose is exact at K=1, resolves the right winner, and decays with width -- plus a CI
selftest wrapper for each new module). 731 -> 734.

## Exact RNS-phasor arithmetic -- the matmul wall was the encoding, not the substrate (shipped)

P1 of the Path D integration -- the second lever, and the one that had no engine module (it lived only in the
Path D experiments). General matmul read out of a lossy SUPERPOSITION (bundle the matrix rows, unbind, dot the
input, no cleanup) is capped by crosstalk: the bundled rows interfere on readout, so fidelity collapses as the
matrix grows. But matmul is multiply-accumulate of NUMBERS, and the FHRR side of the engine already carries a
number exactly as a phase -- a unit phasor exp(2*pi*i*r/m) IS the residue r mod m, and binding phasors adds
their phases. So a product of phasors is exp(2*pi*i*(sum r)/m): the exact sum of residues mod m, for ANY number
of terms, with no crosstalk. That is the single thing the bundle got wrong.

`holographic_rns.py` carries each number as residues over coprime moduli (a Residue Number System), does every
multiply-accumulate as that exact phasor-binding modular arithmetic (one channel per modulus), and recomposes
the integer with the Chinese Remainder Theorem. Wired into the mind as `exact_matmul`.

**Measured through the mind:**
- Modular accumulation via phasor binding is EXACT for thousands of terms (0 errors at N=5000) -- the
  crosstalk-free MAC the bundle could not do.
- Integer matmul at M=256, N=64 is EXACT (max|error| = 0), exactly where the lossy superposed readout of the
  same matmul manages only ~0.11 fidelity.
- A float matmul (fixed-point, scale given) is exact for the QUANTIZED operands.
- The exact dynamic range FEDERATES over moduli channels (~1e8 with a few -> ~1e62 with more) -- the arithmetic
  sibling of the storage array's federation: more channels = more range, coordinated by the thin CRT recompose.

**KEPT NEGATIVE / scope:** exact for INTEGER / fixed-point operands within range. A float is quantized first and
the only error is that fixed-point rounding (set by `scale`) -- a bit-depth question, separable from and unlike
the crosstalk wall (it does not grow with matrix size). And the FLOPs are real: the parallelism is per-modulus /
per-output, native on phasor or RNS hardware, not free on a CPU. The faculty delegates the phase composition to
the same primitive holographic_fhrr binding uses, one phase channel at a time.

Tests: +2 (a mind-level test -- exact integer matmul where the lossy bundle gets ~0.11, the FHRR-binding
accumulation identity, fixed-point exactness, and range federation -- plus a CI selftest wrapper for the module).
734 -> 736.

## Recursive pivot-tree index -- sublinear recall as cleanup applied recursively (shipped)

P2 of the Path D integration -- the forest / data-structure seat (Pharr), and the cleanest realisation yet of
the engine's long-standing sublinear-retrieval wish. The crash that preceded it is the kept lesson: a content
index that summarizes items UPWARD into a bundle hits the capacity wall, and recall collapses to ~0.23 -- the
bundle blurs as it grows. A B-tree never does that. Its internal nodes hold PIVOTS (separators), stored
explicitly, so the wall never bites; in VSA a node is then a small cleanup memory of (pivot -> child), and
routing a query is a nearest-pivot decision applied RECURSIVELY -- the same `cleanup` primitive the mind already
uses, one level per hop, inception as the addressing fabric. `holographic_pivot.py` builds the tree by recursive
k-means (NumPy only, no sklearn -- the minimal-frameworks rule) and routes with a beam; wired in as `pivot_index`.

**Measured through the mind** (216 well-separated leaves, so the exhaustive ceiling is high and any drop is the
routing, not the data):
- Greedy top-1 routing matches the exhaustive scan -- ~0.88 vs ~0.90 -- while touching only ~18 pivots instead
  of all 216, an order of magnitude fewer comparisons (~O(log N), the tree's whole point).
- A beam-5 search lands the true leaf in the candidate set 100% of the time, after which an exact key-unbind (or
  a final scan of the few candidates) finishes -- against the naive summary index's 0.23.

**KEPT NEGATIVE:** each hop is an approximate nearest-pivot decision, so a wrong turn at beam=1 can lose a query
on overlapping (not well-separated) data -- the beam is the honest knob that buys recall back, trading a few
more comparisons for it. The build cost is the recursive k-means. The routing delegates to the same nearest-
codebook cleanup the mind's recall uses; the tree is just that cleanup stacked, depth-many.

Tests: +2 (a mind-level test -- greedy top-1 matches exhaustive at a fraction of the comparisons, with beam
recall of the true leaf -- plus a CI selftest wrapper for the module). 736 -> 738.

## Sketch-routed array recall -- breaking the broadcast wall, content-addressably (shipped)

P3 of the Path D integration. The storage array already recalls in O(1) when you have the directory (item ->
shard), and routerlessly by BROADCAST when you don't -- but broadcast asks every shard, so it costs O(shards)
and soft-erodes as the array grows (the false-alarm tax of many value-cleanup votes). The fix is the same
content-addressable trick one rung up: summarize each shard by a SKETCH = bundle of its keys (the holographic
'and' of what it holds, one extra vector per shard), match a query's key against the sketches in one matmul to
pick the top-c candidate shards, and unbind+cleanup ONLY those. Routing by key-sketch is a CLEAN decision -- a
key sits ~1/sqrt(load) inside its own shard's sketch, far above the 1/sqrt(D) noise from the others -- so it
stays accurate exactly where the broadcast value-vote drowns. Added to HoloArray as `routed_recall(g, c)` and
advertised on the `storage_array` faculty (which now exposes directory / broadcast / sketch-routed recall).

**Measured through the mind (64 shards, ~1920 items):**
- directory recall 1.00, sketch-routed(c=8) 0.99, broadcast 0.95 -- routed tracks the directory while broadcast
  erodes with shard count (0.97 at 32 shards -> 0.95 at 64), and the gap widens as the array grows.
- routed touches only c=8 of the shards, not all 64 -- O(c) unbinds instead of O(shards), the sublinear win,
  while matching the exhaustive directory.

The sketch is built lazily and rebuilt whenever the shard count changes; it reuses the engine's own
bundle/unbind/derived_atom, adding an index, not a new algebra. (Path D also asked whether a 2-level
sketch-of-sketches buys fully sublinear ROUTING; that runs into the per-vector capacity wall on the upper
sketches and is kept in the experiments as a measured open question, not wired as a claim.)

Tests: +1 (a mind-level test: sketch routing stays above 0.95 and tracks the directory at 64 shards while
unbinding only c=8 of them, at least as accurate as full broadcast; the module's own selftest gains a 48-shard
routed-recall check). 738 -> 739.

## Distributed forward pass -- federation applied to compute, with depth cured two ways (shipped)

P4 of the Path D integration, and the headline the whole "as above, so below" arc was driving at: federation,
which fixed storage, fixes COMPUTE too. A linear layer's weight rows stored in ONE bundled vector cap out at
C ~ 0.02 x D classes -- recovering a row carries crosstalk from the other C-1 rows, and the continuous logit
<w_hat_c, x> has no cleanup to absorb it, so fidelity dies as the matrix grows. FEDERATE the rows across K
weight-memory shards (row c in shard c mod K) and recovering a row only carries crosstalk from its ~C/K
shard-mates, so the wall moves to C ~ K x 0.02 x D. `holographic_compute.py` implements the federated readout
and the depth cures; wired in as `distributed_forward`.

**Measured through the mind:**
- A 64-class forward pass: exact classifier 1.00, single-vector readout (K=1) 0.73, federated (K=8) 0.999 --
  federation moves the class wall, and at K=8 the federated pass tracks the exact classifier. In the Path D
  sweep this is 16 classes faithful on one vector -> 96 on eight shards (~6x), the same federation that fixed
  storage applied to the matmul.

**Depth, the second question** (a deep net feeds each layer's noisy output into the next, so crosstalk can
COMPOUND), cured two ways, both wired:
- EXACT arithmetic per layer (`exact_matmul` / P1): no crosstalk to compound at all, so a deep integer forward
  pass is exact at any depth (verified: a 2-layer integer net reproduces the float result exactly). The depth
  decay was arithmetic crosstalk, not a depth wall.
- CLEANUP-GATING (`cleanup_books`): `softclean`, a soft dense-Hopfield, snaps each hidden activation back onto
  the manifold of valid activations (keep the scale, denoise the direction), resetting crosstalk between layers.
  The primitive is robust -- a crosstalk-corrupted activation goes from cos 0.78 to cos 1.00 with its clean
  prototype.

**KEPT NEGATIVE / honest scope:** federation buys FIDELITY / capacity, not fewer FLOPs -- total unbinds are
still C, grouped into K vectors; the parallelism is across the K shards, native on neuromorphic hardware. And
the end-to-end ACCURACY benefit of cleanup-gating needs a well-formed (trained) activation manifold, as in
exp_A1's trained MLP; with untrained or class-mean weights it is seed-dependent, so it is NOT asserted as an
always-win -- only the cleanup primitive (which robustly denoises onto the manifold) and the exact-arithmetic
depth cure (which is exact) are. The faculty delegates to the engine's own bundle/unbind, adding federation and
the depth cures, never a new algebra.

Tests: +2 (a mind-level test -- federation moves the class wall, K=8 tracking the exact classifier; exact_matmul
per layer exact at depth; the softclean cleanup primitive denoising onto the manifold; the cleanup_books path
wired through the mind -- plus a CI selftest wrapper for the module). 739 -> 741.

## Bucket A under federation -- selection, sequence, and the archive wired to the same lever (shipped)

The last of the Path D advancements. The Bucket-A experiments re-opened three more single-vector walls under the
distributed premise, and all three are the SAME conservation law -- a superposed readout capped by per-vector
crosstalk, federated across K shards -- applied to different tasks. So they wire to the faculties that already
embody the federation move, not to new redundant ones.

- **A3 hypothesis selection** and **A4 sequence memory** are the width faculty, federated. `superpose_compute`
  gained a `shards=K` parameter: it spreads the items across K vectors (item i -> shard i mod K) and recovers
  each shard separately, moving the width wall ~K-fold, plus a `decoded` output (per-item cleanup to a codebook)
  for the sequence case. Measured through the mind: picking the planted match out of 160 candidates goes from
  0.38 (one vector) to 1.00 (K=8); recalling a 160-symbol sequence goes from 0.58 to 1.00. One call now serves
  both -- pass a `query` to select, pass position-atom `keys` + a symbol `codebook` to recall a sequence.
- **A5 federated archive** is the storage array's federation applied to the CONTENT archive. `FederatedArchive`
  (new in holographic_archive.py, wired as `federated_archive`) routes image i to shard i mod K over K aligned
  HolographicArchive shards. Measured: at a FIXED total dimension, a monolithic archive and a 4-shard federated
  one recover 64 images at the SAME quality (corr 0.965 vs 0.965) -- capacity federates (total = K x per-shard)
  while recovery is conserved, the conservation law holding for images exactly as it did for symbols.
- **A2** (dense continuous matmul in superposition) is the FOIL, not a new capability -- it is the lossy bundle
  `exact_matmul` (P1) replaces, and it stays on the record as the kept negative (it never gets good because it
  has no cleanup; A3/A4 win precisely because they END in a discrete cleanup -- argmax / codebook snap). **A6**
  (residue integer range) is `exact_matmul`'s own range federation over moduli, already shipped with P1.

This closes the Path D integration: every advancement its experiments demonstrated is now a UnifiedMind faculty
or a measured property of one -- federation for storage (`storage_array`), width (`superpose_compute`, now with
shards), the archive (`federated_archive`), and the forward pass (`distributed_forward`); exact arithmetic
(`exact_matmul`); and sublinear lookup (`pivot_index`, `routed_recall`). The pure conservation-law measurements
(block federation, depth-vs-width, the factor wall on the existing resonator) remain evidence in the experiments,
not invented methods -- a faculty has to earn its place.

Tests: +2 (a mind-level test that federating `superpose_compute` moves the selection AND sequence-length walls,
and one that the federated archive conserves recovery at fixed total dim while federating capacity). 741 -> 743.

## Federation / conservation diagnostic -- the through-line as a callable readout (shipped)

The honest way to wire the Path D conservation MEASUREMENTS into the mind -- rather than leave them only in the
experiments -- is as a diagnostic, the same family as `capacity_report` and `calibration_report`. (Forcing a
measurement into a capability faculty would be a fake faculty; a diagnostic whose job IS to measure and report
is the right shape.) `federation_report` operationalizes the 'as above, so below' law on the mind's own
dimension and kernel, delegating to `storage_array`:

- `per_vector_budget` -- the largest single-shard load whose recall still clears the threshold (measured ~51
  symbols at D=1024, 0.90 recall: ~0.05 x D -- the figure depends on the threshold);
- `federated` -- a spot check that K aligned shards hold ~K x that budget at the same recall (4 shards -> 204
  symbols at 0.94);
- `conservation_ratio` -- partitioning the dimension in half holds total capacity (a half-D vector holds ~half
  the budget, so two tie one full vector): measured 0.98, the block-federation finding that federation buys
  capacity from more DIMENSIONS, not for free;
- `recommended_shards` -- ceil(target / per-vector budget), a planning readout (500 items -> 10 shards).

This is the federation-aware companion to `capacity_report` (which charts a single vector's noise-wins cliff):
together they cover the per-vector cliff AND how federation moves it. It wraps `experiment_below_federation`
(conservation under partitioning) and `experiment_array_scale` (the per-vector budget and its scaling). The
other two conservation measurements are the SAME law in different costumes and are referenced here rather than
given redundant methods: `experiment_depth_vs_width` (escape the per-vector wall by recursion/DEPTH instead of
width -- the mind's `encode_tree`/`peel` are the depth half) and `experiment_factor_wall` (the factorization
search cliff vs dimension, measured on the resonator the mind already exposes via `decompose_structure`).

Honest scope: the budget is the DISCRETE-symbol (cleanup-gated) ~0.05-0.1 x D regime; continuous compute with no
cleanup is the lower ~0.02 x D regime (see `distributed_forward`); and federation buys fidelity and capacity,
not fewer FLOPs.

Tests: +1 (a mind-level test: the diagnostic measures a per-vector budget in the conservation-law range, K
shards hold ~K x it at recall, the partition-conservation ratio is ~1, and the shard recommendation matches
ceil(target / budget)). 743 -> 744.

## Gradient-free substrate-native learning -- reservoir + prototype classifier wired as faculties (shipped)

Translation (the RNS lever) made the substrate RUN trained networks exactly, but not TRAIN them. That gap is
closed with gradient-free learning methods the field already proved -- adopted rather than reinvented -- two of
them wired as UnifiedMind faculties on machinery the engine already had.

`reservoir` (holographic_reservoir.HolographicESN) -- an Echo-State Network whose recurrence IS holostuff's
`permute` (a cyclic shift, hence norm-preserving / orthogonal = the echo-state property). The reservoir is
FIXED; only a linear readout is trained, by one closed-form ridge solve -- truly derivative-free and
deterministic. Diagnostic finding: the permutation recurrence EQUALS a classical random-matrix ESN on NARMA10
(NRMSE 0.560 vs 0.562), so the engine's native operator is a real reservoir with no penalty. Tuned, NARMA10
reaches a literature-grade NRMSE 0.367 +/- 0.001 over 5 seeds (1.59x over a linear-on-raw baseline; the
reservoir features carry it -- state-only equals state+input). It also LEARNS autoregressive text generation
(readout learned by ridge, the substrate generating from it). KEPT NEGATIVES: the first untuned cut was 1.18
(worse than the mean) -- leak too low for NARMA's step-level dynamics, fixed by leak=1.0 + centered input;
chaotic free-running prediction diverges pointwise after ~one Lyapunov time (the climate is learnable, the
weather is not); periodic free-run tracking is loose; the readout learns a linear map of FIXED features.

`prototype_classifier` (holographic_classifier.HolographicClassifier) -- the HDC/VSA learner. Encode each
example (bind a feature-id atom with a ScalarEncoder level, bundle over features), bundle a class's examples
into one prototype (a one-shot centroid), then perceptron retraining: on a miss, pull the correct prototype
toward the example and push the wrong one away (add/subtract on bundled vectors, no gradients). Measured (test
acc, 3 seeds): digits 0.902 -> 0.949, breast_cancer 0.934 -> 0.949, wine 0.981 (saturated) while raw
nearest-centroid is only 0.667 -- the encoding lifts the centroid model dramatically. KEPT NEGATIVE (the
field's own verdict): retraining beats the one-shot centroid, but the classifier lands just BELOW a tuned
linear model (logistic regression, by 0.006-0.016) -- traded for a dead-simple gradient-free rule. Kept
genuinely gradient-free (the perceptron rule), not the SGD-based methods that wear the HDC label.

Both are the TRULY derivative-free corner. The local-gradient methods remain queued: Equilibrium Propagation on
the modern-Hopfield cleanup (free + nudged phases, contrastive-Hebbian, makes attractors learned), and
Forward-Forward / Mono-Forward (layer-local goodness, deeper nets). Standing caveat: native learning at
small/moderate scale, not a route to frontier scale. Real basis: Jaeger / Maass (reservoir computing); Kanerva
/ Rahimi / Imani / Kleyko / Hernandez-Cano (HDC prototypes, AdaptHD / OnlineHD); Scellier & Bengio (EP); Hinton
(Forward-Forward).

Tests: +4 (reservoir selftest: fixed reservoir + ridge readout learns one-step prediction and is deterministic;
classifier selftest: one-shot + perceptron retraining beats chance, does not hurt, and is deterministic; plus
two integration tests running both faculties end-to-end through UnifiedMind). 744 -> 748.

## Equilibrium Propagation -- the learning rule for the energy-based Hopfield cleanup (shipped)

The reservoir and prototype classifier are TRULY derivative-free, but both only learn a linear map of fixed
features. Equilibrium Propagation (Scellier & Bengio, 2017) is the LOCAL-GRADIENT method that learns the
HIDDEN weights of an energy-based net, so it fits a NONLINEAR task they cannot -- and it is exactly the
learning rule for the energy-based (Hopfield) memory the engine uses as a FIXED cleanup (B1): where cleanup
relaxes a query to a stored attractor, EP learns the weights so the energy minima ENCODE a task.

`holographic_equilibrium.EquilibriumNet` -- a 1-hidden-layer continuous Hopfield net (hard-sigmoid rho =
clip[0,1]; symmetric weights Wxh, Who by construction). No backprop. Relaxations of the same circuit: a FREE
phase (clamp the input, relax to an energy minimum = the prediction) and NUDGED phases (add +/- beta *
1/2||o - y||^2 to the energy, relax again). The weight update is the contrastive difference of the nudged
equilibria, dW ~ (1/2beta)(rho(s_-) (x) rho(s_-) - rho(s_+) (x) rho(s_+)), which estimates the loss gradient.
We use SYMMETRIC nudging (Laborieux 2021): the +beta and -beta pair cancels the leading O(beta) bias.

VALIDATED honestly:
- Gradient correctness: the symmetric EP update matches the true gradient (central finite differences over the
  free-phase loss) to COSINE 1.000 on a tiny net -- EP's defining property, measured, not assumed.
- Nonlinear learning: on two interleaving moons (noise 0.10) EP reaches ~0.92 test accuracy vs a linear
  least-squares foil's ~0.85 -- the hidden layer earns its keep; a linear readout on fixed features (the
  reservoir / classifier regime) cannot separate the moons.

KEPT NEGATIVES (on the record):
- EP LANDS BELOW exact backprop: a tanh MLP trained by real backprop reaches ~1.00 on the same moons; EP's
  ~0.92 is the cost of a biased finite-beta gradient estimate. EP is local-gradient, not a free lunch.
- It needs SYMMETRIC weights, and costs THREE relaxations per update (free + two nudged) -- far more compute
  than the one-shot reservoir / classifier rules.
- Instability if pushed: large lr / weight-init drives the relaxation to collapse (~0.5, chance); the working
  regime needs a converged free phase (longer t_free, smaller dt) and a modest lr.
- Two bugs found-and-fixed during the build, kept as lessons: (1) the hard-sigmoid derivative must be INCLUSIVE
  on [0,1] -- else a state initialized at 0 has zero force and never moves; the gradient check caught it as
  cosine 0.000. (2) The two-moons split must be SHUFFLED -- an index split left the test set single-class,
  reading as acc 0.000; a harness bug, not an EP bug, caught only because below-chance accuracy is a red flag.

Both the truly-derivative-free corner (reservoir, classifier) and now the local-gradient corner (EP) are
shipped; Forward-Forward / Mono-Forward (layer-local goodness, deeper nets) is the one method still queued.
Standing caveat: native learning at small / moderate scale, not frontier scale. Real basis: Scellier & Bengio
(2017); Laborieux et al. (2021).

Tests: +2 (an EP selftest -- the symmetric update matches finite differences to cosine > 0.9 AND it learns two
moons past a linear foil, deterministically; plus an integration test running the EP faculty end-to-end
through UnifiedMind). 748 -> 750.

## Forward-Forward -- backprop-free depth from local objectives, with a loud kept negative (shipped)

Forward-Forward (Hinton 2022) is the last family and the DEPTH corner: it stacks many layers, each trained by
its OWN local objective with no gradient flowing between them -- depth without a global backward pass and
without EP's settling. Mechanism: replace backprop's forward+backward with TWO forward passes. POSITIVE data ->
train each layer to high "goodness" (mean squared activity); NEGATIVE data (the same input with a WRONG label
embedded) -> train each layer to low goodness. Each layer's local loss is a logistic on (goodness - theta); its
weights move by the gradient of THAT loss alone. Every layer L2-NORMALIZES its output before the next sees it,
so a later layer can't read the length an earlier one already separated. Classification is label-embedded:
prepend a one-hot label; at test try each label, forward, and pick the highest accumulated goodness.

`holographic_forward.ForwardForwardNet`. Two implementation fixes were needed and are kept as lessons: (1) a
single global theta fails because layer goodness scales differ wildly (layer 0 ~1-4, layer 1 ~0.015 after
normalization) -- the fix is a PER-LAYER adaptive threshold (EMA of the goodness) so each logistic stays
centered and has gradient; (2) prediction must sum goodness over ALL layers (the constant 'a label is present'
part cancels across candidates, leaving each layer's learned label-vs-input compatibility). With those, the
mechanism works: on separable 4-class blobs it classifies at 100% with a positive-minus-negative goodness gap
of ~+2.4 on held-out data.

KEPT NEGATIVE (MEASURED, loud -- the most humbling of the program):
- At the small scale tested this compact FF is a WORKING but WEAK classifier. It TRAILS a plain linear /
  logistic model on EVERY task tried: two-moons ~0.88 (a tie with linear, NO nonlinear advantage -- unlike EP);
  overlapping 4-class blobs 0.95 vs linear 0.99; sklearn digits (its natural high-dim habitat) 0.88 vs logistic
  0.97. It beats linear only on a radial task where linear PROVABLY fails (~0.69 vs 0.47), and even there weakly.
- FF's published accuracy (Hinton's ~1.4% MNIST error) needs the full-scale recipe -- many layers, large width,
  long training, carefully built negatives -- not reachable in a compact CI-fast module. What this module
  contributes is the MECHANISM (backprop-free, settling-free depth from local objectives), a conceptual route,
  NOT a competitive number.
- Local-gradient, not derivative-free (like EP). Goodness-based label inference costs one forward pass per class
  at test. Sensitive to the goodness threshold and the negative-data quality.
- The stronger Mono-Forward (2025) refinement (per-layer LOCAL supervised projections to logits) is reported to
  match tuned backprop; it is the natural next step for a competitive FF-family accuracy and is NOT built here.

This closes the four-family learning program. Honest summary of the whole arc: the TRULY derivative-free corner
(reservoir, prototype classifier) is competitive at small scale; the LOCAL-GRADIENT corner splits -- Equilibrium
Propagation genuinely learns nonlinear functions and beats linear (two-moons 0.92 vs 0.85), while Forward-Forward
demonstrates backprop-free depth but trails linear at this scale. None reaches frontier scale; the engine now
holds a clear-eyed MAP of what substrate-native learning buys and where each method's boundary lies. Real basis:
Hinton (2022); Mono-Forward (2025).

Tests: +2 (an FF selftest -- the local-goodness mechanism classifies a separable task and separates positive
from negative goodness, deterministically; plus an integration test running the FF faculty end-to-end through
UnifiedMind). 750 -> 752.

## NONLINEAR DYNAMICS COMPANION (shipped): learning a chaotic flow the linear propagator cannot

This is the above/below examination's strongest "ABOVE" candidate, realised: the unlocked LEARNING aimed
straight at the most embarrassing kept negative in the dynamics line. `learn_dynamics` (Propagator) fits ONE
per-frequency complex transfer -- the linear Koopman/DMD operator. Exact for linearisable flow (it recovers
advection-diffusion almost perfectly), but a single fixed linear map cannot follow a state-dependent
nonlinearity: the record already carried "on a shock-forming Burgers field the linear propagator does WORSE
than persistence (0.054 vs 0.006; 0.125 vs 0.015)". The fix the negative itself named was "a learned lift".

holographic_chaos.py / `mind.learn_chaos` is that lift, and it DELEGATES to the reservoir (holographic_reservoir)
rather than re-implementing a learner -- a fixed nonlinear echo-state expansion read out by a TRAINED linear
map learns the one-step evolution operator a linear transfer structurally cannot. NonlinearPropagator.learn
captures per-coordinate normalisation, fits the reservoir readout (one ridge solve) to map state(t) ->
state(t+1); `predict_sequence` gives one-step-ahead forecasts, `free_run` closed-loop rollout. lorenz_trajectory
(RK4) gives the selftest a known chaotic system with no external-data dependency.

MEASURED (Lorenz '63, the canonical reservoir-computing test, RK4 dt=0.02):
- ONE-STEP is a clean WIN and NOT a strawman. Reservoir one-step ~0.0014 relative error vs the BEST linear map
  (full DMD) ~0.059 and persistence ~0.071 -- ~40x better than best-linear, ~50x better than persistence. The
  engine's own circulant propagator only ties persistence. A linear map sits at the chaos floor because the
  Lorenz flow is state-dependent; the nonlinear reservoir genuinely learned the local evolution operator.
- Deterministic (same seed -> identical readout). Closed-loop free-run tracks the attractor ~10x longer than
  persistence.

KEPT NEGATIVES (loud, on the record -- the boundaries, established by sweeps, not guessed):
- Closed-loop horizon is only ~ONE Lyapunov time -- far short of the ~5 the one-step error implies. A 0.0014
  one-step error under clean chaotic growth would give ~5 Lyapunov times; getting ~1 means the AUTONOMOUS system
  diverges faster than chaos alone: the well-known reservoir free-run STABILITY problem. State-noise helps only
  marginally (noise=1e-2 is the sweet spot; 1e-1 shortens it to ~0.2), bigger reservoirs help modestly
  (dim 500->1500 took 0.4->0.8), and -- the key finding -- the recurrence MIXING is NOT the lever: cyclic-shift,
  random-permutation, AND an inline unitary-bind (random circulant orthogonal) recurrence all cap at ~0.4-0.9
  Lyapunov times. The wall is closed-loop stability, not mixing. Cracking it is a research problem of its own;
  this module does not claim to.
- HIGH-DIMENSIONAL PDE FIELDS are out of reach for a single global reservoir. Forecasting a 48-D Burgers field
  one-step lands ~0.27-0.35 relative error, far worse than persistence; Equilibrium Propagation as a memoryless
  field regressor was also ~0.08 (worse than persistence). Pathak et al. (2018) forecast the chaotic
  Kuramoto-Sivashinsky equation with LOCAL/parallel reservoirs precisely because one global readout cannot, and
  EP's sweet spot is low-output (classification-shaped) targets, not 48-D field regression. The win above is a
  genuine LOW-dimensional nonlinear-dynamics result, said plainly.
- On MILD dissipative Burgers the per-step change is tiny (persistence ~0.012), a punishing baseline with almost
  nothing to win regardless of the learner -- which is why the clean win lives on chaos (weak persistence), not
  on a smooth dissipative flow.

As-above-so-below: this is the LEARNING (a within-/across-vector trained map) wired ONE RUNG UP to fix a
system-level dynamics negative, delegating to the reservoir faculty rather than re-building it -- the
examination's prediction made load-bearing by a cross-faculty test. Real basis: Jaeger echo-state networks;
Pathak et al. (2018) reservoir forecasting of spatiotemporal chaos; Lorenz (1963).

Tests: +2 (a chaos selftest -- the nonlinear learner beats best-linear by >10x on the chaotic one-step map and
beats persistence's free-run, deterministically, without overclaiming the ~1-Lyapunov-time horizon; plus an
integration test running `learn_chaos` end-to-end through UnifiedMind). 752 -> 754.

## Sparse cleanup readout + geometry-aware denoise -- match the MAP to the MANIFOLD (shipped)
A measured negative drove this: on a CONTINUOUS manifold (recovering UN-stored in-between points along a
photo-to-photo path in real SD latents), the softmax modern-Hopfield cleanup TIES or LOSES to plain
nearest-neighbour -- softmax 0.983, NN 0.998 -- because the dense blend weights in far codebook atoms and
OVER-SMOOTHS. This is the documented metastable-mixing / "fuzzy-memory" failure of the softmax update
(Ramsauer et al. 2020), and the field's 2024-25 fix is a SPARSE readout.

Two VSA-native fixes, both prototyped and measured before wiring:
- SPARSE CLEANUP READOUT. `dense_cleanup(..., readout='sparsemax')` replaces the softmax blend with a
  sparsemax simplex projection (Martins & Astudillo 2016) -- the Hopfield-Fenchel-Young move (Santos,
  Niculae, McNamee, Martins 2024-25; sparse Hopfield Hu 2023 / Wu 2024) -- so the readout blends ONLY the
  relevant patterns. Measured on the continuous SD-latent manifold: sparse 0.999 > NN 0.998 > softmax 0.983;
  it REVERSES the softmax-loses-to-NN result. It does NOT regress discrete recall (still exact at high beta,
  where sparsemax is also one-hot -- pinned by test). Default stays 'softmax', bit-for-bit unchanged.
- GEOMETRY-AWARE DENOISE SELECTOR. `denoise(method='geometry', samples=/codebook=)` reads the set's
  `effective_rank` (the consolidation/SVD spectrum knee) and routes: LOW-rank-relative-to-count (a continuous
  manifold) -> project onto that subspace (Milanfar's denoiser-as-manifold-map, RED 2017; a tensor-network
  truncation in Stoudenmire's reading); HIGH-rank (distinct atoms) -> codebook recall. Measured:
  manifold-projection recovers UN-stored in-between points at 1.000 vs the softmax blend's 0.983.

KEPT NEGATIVES (travel with the methods):
- The sparse-beats-NN margin is THIN (~0.001, e.g. tour 0.996 vs 0.995); its CLEAR, robust win (every seed in
  the variance harness) is over the softmax blend, not over NN. Reported, not oversold.
- Manifold projection only helps where the manifold is genuinely low-rank: forced onto the HIGH-rank distinct
  photos it COLLAPSES recall to 67% (measured, and asserted by test). That failure is exactly why the router
  reads the rank first -- match the map to the manifold, never project high-rank data.
- The softmax blend over-smooths continuous manifolds; use the sparse readout (or projection) there.

The deepest VSA-native handle (recognised, not newly built): a continuous manifold should be HELD as a
function -- Vector Function Architecture / fractional power encoding (Frady, Kleyko, Kymn, Olshausen, Sommer,
Computing on Functions) -- which the RBF ScalarEncoder already is; the codebook-of-samples was the wrong
construction for a continuous quantity. The two wired items fix the readout and the routing; the functional
representation was in the encoder all along.

As-above-so-below: the readout lives in the KERNEL (`dense_cleanup`, below) and is threaded UNCHANGED through
the mind's `denoise` (above) -- a test pins `mind.denoise(method='codebook', readout=...)` bit-for-bit to the
kernel call, so the mind delegates and does not re-implement. The geometry router delegates to `effective_rank`
+ `fit_manifold`/`manifold_denoise` + `codebook_denoise` (no new math), and a cross-faculty test proves it
picks projection on a low-rank manifold and recall on a high-rank set, with the high-rank-projection failure
kept as the guard. Real basis: Ramsauer et al. (2020); Martins & Astudillo (2016); Santos, Niculae, McNamee,
Martins (2024-25, Hopfield-Fenchel-Young); Hu (2023), Wu (2024); Romano, Elad, Milanfar (2017, RED); Frady,
Kleyko, Kymn, Olshausen, Sommer (Computing on Functions / VFA).

Tests: +9 (sparsemax-simplex; softmax-readout-unchanged; high-beta-pins-to-hard-NN for both readouts;
sparse-beats-softmax and not-worse-than-NN on a continuous manifold; sparse-does-not-regress discrete recall;
effective_rank separates the geometries; the geometry router matches the right map AND projection-on-high-rank
is worse; the kernel<->mind above/below delegation; and a variance harness bootstrapping the margins across 12
seeds). 754 -> 763.

## Sparse readout in the SBC resonator -- the same fix one rung up, a measured capacity win (shipped)
The denoise finding generalised exactly as predicted: the SBC resonator's alternating projection
(`sbc_resonator`) updates each factor estimate with an annealed SOFTMAX blend over its codebook -- the same
dense blend whose metastable mixing hurt continuous-manifold cleanup. Swapping it for the sparse readout
(`readout='sparsemax'`, delegating to the shared `_sparsemax`) blends only the relevant atoms each step.

MEASURED (F=3, B=16, L=16, all-factors-correct over 40 trials), softmax -> sparse:
- CLEAN CAPACITY rises sharply: N=25 0.47->0.62; N=50 0.00->0.12; N=80 0.00->0.25 -- sparse RECOVERS
  factorizations where the softmax blend collapses to exactly zero. The sparse blend escapes the
  metastable/limit-cycle traps that cap the softmax resonator at high alphabet.
- APPROXIMATE input (corrupted product blocks): helps at low corruption (clean 0.80->0.95) and TIES under
  heavy corruption (corrupt=4 both 0.62). KEPT NEGATIVE: the win is largest on clean high-alphabet capacity;
  heavy corruption is a tie, and absolute capacity is still modest (N=80 0.25, but vs softmax's 0.00).
- It NEVER regresses (sparse >= softmax in every cell). The annealed beta still drives explore->commit
  (sparsemax keeps a sparse-but-broad set at low beta, one atom at high beta), so the search schedule holds.

The CONFIDENCE null is matched to the readout. `_resonator_noise_null` / `resonator_confidence` now take the
readout and re-fit the procedure-matched noise floor under it (the cache key includes the readout) -- the
recurring rule: a null that does not match the actual procedure lies, and sparse manufactures a different
noise-floor agreement than softmax. Default stays `readout='softmax'`, bit-for-bit unchanged; sparsemax is
the measured-better opt-in (recommended, not yet the default, per the engine's backward-compatibility rule).

As-above-so-below: the readout switch lives in the kernel resonator (`sbc_resonator`, below), threads
unchanged through `resonator_confidence` / `decompose_structure` and up through the mind's
`decompose_structure` and `factor_composite` (above) -- a test pins `mind.decompose_structure(readout=...)`
to the SBC factorizer's picks, so the mind delegates and does not re-implement. This is the SECOND faculty to
take the readout fix (cleanup was the first), confirming the panel's read: wherever a softmax blend appears,
the sparse readout is a candidate -- and here it cleared a real bar (a capacity win, not a thin margin). Real
basis: Frady et al. (2020), Kymn, Olshausen et al. (2024) resonator networks; Martins & Astudillo (2016)
sparsemax; Santos, Niculae, McNamee, Martins (2024-25, Hopfield-Fenchel-Young); Ramsauer et al. (2020).

Tests: +5 (softmax-default-unchanged picks; the capacity win sparse>softmax at N=25/50 with no regression at
N=10; the confidence null is recomputed per readout; the mind delegates AND threads the readout through both
`decompose_structure` and `factor_composite`). 763 -> 768.

## Sparse readout in the generative attractor -- the same fix cures generative mode collapse (shipped)
The third application of the readout finding, and the one that revealed a NEW axis. `generate` and
`generate_structure` are annealed cleanup attractors (denoise from pure noise, beta up / noise down); both run
through `dense_cleanup` (`generate_structure` slot-wise via `_structure_project`), so the sparse readout
threads in directly. The bar was "cleaner valid samples"; what it actually cleared was DIVERSITY.

MEASURED (dim=1024-2048, recon-validity = cosine(z, reencode(decode(z))); diversity = fraction distinct combos
over the seeds):
- VALIDITY is a perfect tie: BOTH readouts produce structures that reencode their decoded combination at
  cosine 1.000 in every config -- so the decoded combos are trustworthy and the structures are genuinely valid.
- DIVERSITY diverges sharply: softmax generation MODE-COLLAPSES (many random seeds settle into the SAME few
  structures -- diversity as low as 0.03, i.e. one structure for 30 seeds, and typically 0.13-0.5), while
  sparsemax stays diverse (0.6-1.0, nearly every seed a distinct valid structure). The mechanism is the SAME
  metastable mixing: the softmax blend's wide blended basins funnel different noise starts to one attractor;
  sparse's distinct basins let them settle into different valid structures. So the readout fix shows up here
  as a cure for generative mode collapse, at NO validity cost.

KEPT NEGATIVE: over a CONTINUOUS codebook, `generate` is UNAFFECTED by the readout -- both softmax and sparse
snap to a stored coarse atom (validity-to-manifold 1.000, novelty ~0). The old "bare codebook -> stored atom"
negative holds for both readouts; the sparse win is specific to `generate_structure` (the discrete composed
manifold), not to continuous-manifold generation. Pinned by a test.

The creature is the boundary the analogy does NOT cross, and that is worth recording: `decide` ends in a HARD
argmax over scores, and the value readout it argmaxes is a clipped-cosine (ReLU-kernel) weighted average over
the top-k prototypes -- already sparse/thresholded, and a one-shot estimate, not an iterated soft-blended
attractor. There is no softmax blend there for sparsemax to improve, so the creature's decision path was left
unchanged (an honest non-fit, not a forced one).

As-above-so-below: the readout switch lives in the kernel attractors (`generate`, `_structure_project`,
`generate_structure`, below) and threads up through the mind's `generate_vector` / `generate_structure`
(above); a test pins `mind.generate_structure(readout=...)` to the kernel generator's exact output, so the
mind delegates. This is the THIRD faculty to take the readout fix (cleanup -> resonator -> generator), and it
adds a new line to the unifier: a softmax blend in an ITERATED attractor is a sparse candidate not only for
accuracy/capacity but for SAMPLE DIVERSITY. Real basis: the cleanup attractor as diffusion (Ramsauer et al.
2020 modern Hopfield; Hopfield-Fenchel-Young, Santos/Niculae/Martins 2024-25); Martins & Astudillo (2016)
sparsemax; the VSA generate-by-denoising framing (Frady/Kymn/Olshausen/Sommer resonator + cleanup line).

Tests: +5 (softmax-default-unchanged output for both generate and generate_structure; sparse structures stay
valid at recon-cosine ~1; the mode-collapse cure sparse-diversity > softmax-diversity; the continuous-generate
kept negative; the mind delegates AND threads the readout). 768 -> 773.

## LEARNED ENERGY MEMORY (shipped): training the cleanup's attractors instead of storing them

The panel's audit found the whole formal backlog (A1-A14, incl. tensor-train via tensor_bind rank) already
shipped; the ONE genuinely-unbuilt thing the seats' real methods converged on was that
holographic_equilibrium's docstring CLAIMED EP "is the rule that LEARNS those attractors" of the energy
memory -- but nothing actually trained a cleanup's energy. EP ran as a standalone classifier; the cleanup
stayed fixed (classical = snap to a stored atom; modern-Hopfield dense_cleanup = relax against a fixed
codebook). holographic_energy.py / `mind.learn_cleanup` makes good on that claim. It DELEGATES to
EquilibriumNet (not a new learner) to train a denoising AUTO-ASSOCIATOR -- (sample+noise -> sample) pairs,
hidden bottleneck ~ D/2 forcing the attractor set onto the low-dim manifold -- whose `cleanup(x)` clamps x,
relaxes, and reads the free-phase output: a noisy query PROJECTED onto a LEARNED manifold instead of snapped
to the nearest stored sample. torus_bump_manifold gives the selftest a known continuous nonlinear manifold
(a Gaussian bump at a continuous position on a latent_dim-torus; curved, NOT low-rank, so SVD/consolidation
can't denoise it and a finite codebook can only QUANTIZE the continuous position) with no external data.

The result is GEOMETRIC -- when learning the energy beats storing the codebook is the whole point:
- vs the FIXED SOFT energy cleanup (dense_cleanup): unconditional win on a continuous manifold at every
  codebook size (1-D EP ~0.33 vs soft 0.43-0.51; 2-D EP ~0.43 vs soft 0.45-0.56) -- the soft cleanup returns
  a softmax MIXTURE that blurs on a continuum while the learned net projects. Apples-to-apples: learned energy
  memory beats fixed energy memory.
- vs storing DATA (hard 1-NN codebook of RANDOM manifold samples) the win is DIMENSIONAL. On a 2-D manifold at
  MATCHED MEMORY (EP weights 2*D*hidden vs an equal-byte codebook, K~=48): EP ~0.43 vs hard-1NN ~0.49-0.50.
  Tiling a d-manifold with samples costs ~grid^d points (curse of dimensionality); a fixed-size learned
  projector scales with the manifold's intrinsic structure, not its volume. Deterministic (Who allclose).

KEPT NEGATIVES (loud -- they ARE the boundary, measured by sweeps):
- DISCRETE atoms are the wrong job. Queries = noisy versions of a finite stored set -> HARD 1-NN returns the
  EXACT atom (~0.02-0.03) and is unbeatable; the learned approximate energy (~0.21) loses. This is B1's
  "single-item identity is a tie" SHARPENED: against the hard cleanup it's a loss, not a tie. Use the existing
  cleanup for discrete recall.
- In 1-D the curse does NOT bite, so a matched-memory random-sample codebook BEATS the learned energy (1-D
  K=32 ~0.27 vs EP ~0.33). The advantage over storing data REQUIRES manifold dimension >= 2. In 1-D, just
  store the samples.
- The win over a codebook is at MATCHED memory, not unbounded -- give the codebook 2-4x more bytes and it wins
  even in 2-D (2-D K=100 ~0.41, K=200 ~0.35 vs EP 0.43). And EP inherits its weakness at very high output
  dimension (this targets moderate D, low intrinsic-dim manifolds; it is NOT a high-D field denoiser -- cf.
  the chaos module's 48-D Burgers negative).

This is the LEARNING reaching the engine's most fundamental fixed object (the cleanup) -- the examination's
"below" unlock realised, the apex of the learning arc (reservoir -> classifier -> EP -> FF -> learn_chaos ->
learn_cleanup): the through-line was "make the fixed objects trainable", and the cleanup was the last and
deepest one. It is also the natural LEARNED prior for the Plug-and-Play/RED loop the engine already runs
(Milanfar: a denoiser is a map of the signal manifold -- now a LEARNED map). Real basis: Krotov-Hopfield /
Ramsauer et al. (the energy memory); Scellier-Bengio (2017) / Laborieux (2021) Equilibrium Propagation (the
local learning rule); Romano-Elad-Milanfar (RED) for the prior framing.

Tests: +2 (an energy selftest -- the learned energy beats both the soft cleanup AND a matched-memory
random-sample codebook on a continuous 2-D manifold, deterministically, while the hard 1-NN cleanup wins on
discrete atoms (kept negative); plus an integration test running `learn_cleanup` end-to-end through
UnifiedMind and beating the fixed soft cleanup). 773 -> 775.

## Grounded answering -- short, accurate, constructed sentences from retrieved knowledge (shipped)
The text-generation review's honest finding drove this: the engine's RELATIONAL layer answers questions
correctly and traceably (is_a chains, role lookups, learned-meaning similarity, classification), while its
GENERATIVE layer is locally fluent but globally incoherent (measured: longest verbatim run 3-8 words and
85-100% novel 4-grams, so NOT snippet-copying -- but no sentence-level meaning; and the structure verifier
rates a Markov walk as MORE typical than real text, so it cannot certify coherence). The right way to "answer
a question with a sentence that makes sense" is therefore NOT to generate -- it is to RETRIEVE the facts and
REALIZE them.

`answer_text(question)` (faculty) = `realize_answer(answer(question))`. It delegates ALL retrieval to the
existing `answer()` router (which maps a question to the brain's real operations) and adds the one missing
piece: a surface-realization layer (`holographic_answer.realize_answer`) that builds a short sentence from the
retrieved STRUCTURE -- the is_a chain, the role value, the learned-meaning neighbours. Template-based NLG over
a holographic knowledge base, the standard pre-neural move, deliberately using the parts that WORK and NOT the
free n-gram walk.

The three properties, MEASURED on a known encyclopedia + dictionary battery:
- ACCURATE: 11/11 on known questions. "Yes -- a dog is a mammal, which is an animal."; "No -- a salmon is a
  fish, and ultimately an organism, not a bird." (correct no, and it explains what it IS); "The capital of
  france is paris."; "A dog is a mammal -- more broadly, an organism. It's closely related to cat, wolf...".
- NO FABRICATION: 3/3 honest abstentions on unknowns. "I don't have dragon in my knowledge, so I can't say
  whether it's an animal." -- the calibrated-abstention discipline applied to language: unknown concept,
  low-confidence recall/classify, or a question that falls through to the generation path -> abstain, never
  invent. Confidence/score floors gate role/recall/classify; an is_a "no" is only emitted when the subject is
  actually known (chain length > 1), else it abstains.
- NOT VERBATIM: the sentence is CONSTRUCTED from the structure (a new sentence, not a copied source line);
  verbatim only where the answer simply IS a stored value (a capital, a parent class) -- i.e. only when that
  is what was asked for. Article (a/an) and natural-list rendering handle 1/2/3+-link chains gracefully.

KEPT CAVEAT (loud): the "closely related to" neighbours come from the dictionary-meaning space, which groups
by shared DEFINITION words -- so they can include attributes ("four", "wood", "leaves"), not only clean
taxonomic siblings. The is_a/role parts are exact; the relatedness part is associative, consistent with the
meaning_predict finding that the dictionary space separates related words at d'~0.76 (good, not perfect). A
concept-only filter on the neighbours is the obvious next refinement.

As-above-so-below: the realizer is a pure function over the `answer()` struct (fast deterministic unit tests
pin every branch); `answer_text` is exactly `realize_answer(answer(q))` (a test pins the delegation). Real
basis: template-based natural-language generation over a knowledge base (the standard grounded-QA architecture
before neural LMs); the engine's own relational faculties (climb/is_a/read_role/define/classify/recall) supply
the content; the calibrated-abstention thread supplies the "don't fabricate" floor.

Tests: +10 (realizer form/accuracy per kind; low-confidence and unknown-subject abstention; recall/classify
score gates; completion/unknown abstain; helpers; and end-to-end: answer_text delegates to answer(), is
accurate, and does not fabricate on unknowns). 775 -> 785.

## VSA-native question routing -- understand the question from a blend of word meanings (shipped)
The answerer's reach was limited by `answer()`'s brittle regex templates: natural or verbose phrasing
("could you tell me whether a dog is an animal", "do you happen to know the capital of japan") missed the
template and abstained even though the brain knew the answer. The fix is the engine's OWN machinery, exactly
as proposed: the text encoder already turns a string into a BUNDLE of its word meanings (the VSA blend), and
that bundle is a good INTENT signal, so route the question by encoding it (`mind.perceive`) and matching it to
per-intent prototypes (each the mean bundle of several example phrasings), then dispatch to the brain's real
operations.

The crux, kept honest: bundling is COMMUTATIVE, so "is a dog an animal" and "is an animal a dog" blend to
nearly the same vector -- the blend gives the KIND of question but not WHICH concept is subject vs object. So
intent comes from the blend, and the ARGUMENTS come from a concept-scan: find the words the mind actually
KNOWS (its class labels + lexicon) and use their ORDER (first found = subject, last = ancestor) to assign the
roles the commutative bundle cannot. Intent-by-blend + arguments-by-order is the whole design; each half does
the job the other can't.

MEASURED:
- INTENT routing: on natural/verbose phrasings the regex abstained on (0/8), the blend routed 7/8 correctly
  (the miss is "what's a salmon" leaning IS_A because "a salmon" appears in the IS_A examples).
- END TO END through `answer_text`: 5/6 of those phrasings now answer correctly via the VSA fallback ("Yes --
  a dog is a mammal, which is an animal." from "could you tell me whether..."), and the DIRECTION case
  "is an animal a dog" -> "No -- an animal is an organism, not a dog." is resolved by the order-scan, which the
  blend alone cannot do.
- BACKWARD-COMPATIBLE: `answer()` tries the exact templates FIRST and calls the VSA router only when they miss,
  so every templated question is byte-for-byte unchanged (tests pin that the `via` field is not 'vsa' there).
- ABSTENTION PRESERVED / NO FABRICATION: arguments must be concepts the mind knows; an unknown concept yields
  no usable pair and the router returns None -> the honest abstention fires. KEY FIX found by measurement: the
  define-fallback is restricted to DEFINE/SIMILAR intents -- describing the lone KNOWN concept of an IS_A
  question whose SUBJECT is unknown ("is a dragon an animal", animal known, dragon not) would answer the WRONG
  thing, so those abstain instead. Better to abstain than to mislead.

KEPT NEGATIVES (loud): short questions with overlapping content words can confuse adjacent intents (mitigated
to abstention, not a wrong answer); classify/recall need an explicit text PAYLOAD not a concept, so they stay
with the templates (this router covers the relational intents is_a/role/define/similar); and the intent
prototypes come from a fixed example set, so a heavily padded wording can fall below the intent floor and
abstain rather than route. The router is the broad net; the templates remain the precise one.

As-above-so-below: the router returns an `answer()`-style struct, so `realize_answer` and the abstention floors
apply unchanged -- the new path reuses the whole grounded-answer pipeline, it does not fork it. Real basis:
the VSA blend (bundle of word-meaning atoms) as a bag-of-words intent signal; the engine's own
perceive/encode (the one text encoder) and its relational faculties (is_a/climb/read_role/define) for content;
binding/order, not bundling, for argument roles (the standard VSA lesson that superposition is order-free).

Tests: +7 (intent classification of natural phrasings; the order-based direction fix; natural role/define
answers; abstain-when-subject-unknown; answer_text answers natural phrasings via the fallback; templated
questions unchanged/backward-compatible; abstention preserved on natural unknowns). 785 -> 792.

## Ordered lists -- recipes, directions, instructions: how well, measured (assessment + correction)
Asked how well the engine handles ordered lists, the honest answer is: well, and better than its own docstring
claimed. Ordered sequences live in `SequenceMemory` via PERMUTATION-positional encoding -- each step's atom is
rotated by its 1-based position and bundled into one vector (a scrambled order is near-orthogonal, cosine
~0.03). The mind exposes it as `learn_plan(name, steps)`, `step_at(name, i)`, `precedes(name, a, b)`, and
`validate_plan(name_or_steps, constraints)`.

MEASURED (dim 2048):
- REAL lists -- an 8-step pancake recipe, 7-step driving directions, 6-step chair assembly -- recall in order
  EXACTLY (8/8, 7/7, 6/6). precedes is correct both ways; validate_plan passes correct ordering rules and, on
  an impossible rule, returns False naming the exact offending pair.
- CAPACITY is far past the old "~8" claim. Forced-choice step recall (which cleans up against the step list)
  is 100% out to length 40, ~99.7% at 80, ~92.5% at 120 -- a graceful decline, not a hard cliff. Position
  decoding (the harder token->slot direction, used by position_of/precedes) tracks it: 100% to ~40, ~99% at
  80, ~93% at 120. The "~8" docstrings were corrected to these measured numbers.
- ROBUST cleanup: even with 1000 competing distractor atoms in the vocabulary, step recall at length 12 stays
  100% -- the permutation signal is clean enough that the right token wins without needing a small candidate
  set. (So the strength is the encoding, not a forced choice.)

KEPT NEGATIVES (loud, pinned in tests):
- REPEATS are half-handled. A recurring step ("stir" at positions 1/3/5) recalls correctly at EVERY slot
  (position -> element is position-indexed), but the INVERSE position_of (element -> position) is an argmax and
  returns only ONE of the occurrences. An all-occurrences query would need to threshold the per-position
  scores instead of taking the argmax.
- Each step is a WHOLE-STRING ATOM, so this is exact, order-faithful recall of stored step labels, not
  generation or paraphrase: a reworded query step ("beat eggs" vs the stored "beat the eggs") is a different
  atom and will not match. There is no fuzzy step matching yet (a natural place to reuse the learned-meaning
  space or an edit-distance fallback), and the step CONTENTS are opaque to the order machinery -- it captures
  the meaning that lives in the ORDER, not the meaning inside each step.

Net: for directions/recipes/instructions at realistic lengths (a handful to a few dozen steps) the engine
stores them faithfully and answers position, "what is step i", precedence, and constraint-violation queries
exactly. The two real gaps are all-occurrences-for-repeats and fuzzy step matching.

Tests: +3 (exact ordered recall + precedence + validation on a realistic recipe; capacity pinned exact at 20
and >=88% at 120; the repeat recall/position_of-limit pinned as a kept negative). 792 -> 795.

## Executable procedures: HoloMachine wired INTO the mind (de-silo; milestone 1 of 4) (shipped)
HoloMachine -- the stored-program VM whose opcodes ARE VSA operations (LOAD/BIND/BUNDLE/PERMUTE/CALL),
where a program is one hypervector and a function library is one vector callable by name -- had been kept
deliberately ADJACENT to UnifiedMind ("a program is just another value; leave the interpreter standalone").
That was the silo. This milestone makes it a FACULTY: the mind owns a `_machine()` built at the mind's own
dim and seed (the same share-the-substrate move `_seq_mem` makes), so a procedure's accumulator is a vector
in the mind's OWN space and the format is deterministic.

New faculties (all thin delegations to the machine, nothing re-implemented):
- `learn_procedure(name, program)` -- store a named executable ACC->ACC recipe in the library; composable
  (a procedure may CALL procedures defined earlier).
- `run_procedure(name_or_program, init_acc=None)` -- execute and return (accumulator, trace). `init_acc`
  seeds the accumulator with a vector from the mind's own space -- the bridge that makes a procedure an
  operation ON the mind's data, not just on the machine's data atoms.
- `decode_step(name_or_program, i)` -- read instruction i back as (opcode, operand): the von Neumann
  encoding means a stored procedure is DATA you can inspect, not only run.
- `procedure_to_recipe(program)` -- express a procedure as a typed B7 StructureRecipe, bit-exactly.

Distinct from `learn_plan` on purpose: a plan stores an ordered list of opaque step LABELS ("beat the eggs");
a PROCEDURE stores a recipe of real operations that DOES something. The two are the read/exec halves of the
same "ordered steps" idea on one substrate.

MEASURED / as-above-so-below: the de-silo is proven LOAD-BEARING, not nominal -- a procedure run through the
mind is BIT-FOR-BIT identical to the same program through a bare HoloMachine at the same dim & seed
(np.array_equal on the accumulator), so the faculty truly delegates. Correctness holds (LOAD a; BIND b;
BUNDLE c == bundle(bind(a,b),c) at cosine 1.0); CALL-composition computes the right result through the mind;
decode_step reads instructions back exactly; and `realize(procedure_to_recipe(prog))` == `assemble(prog)` at
cosine 1.0. KEPT NEGATIVE (inherited, documented in the VM): instruction decode is a noisy cleanup whose
capacity scales with dim (~32 instructions at dim 1024, ~128 at 4096) before bundle crosstalk wins -- the
honest HRR capacity wall, not hidden; for realistic recipe lengths it is exact. Real basis: von Neumann
stored-program model expressed holographically (instructions and data in one vector space); HRR
bind/bundle/cleanup as the execution engine.

This is milestone 1 of 4. Next, on this foundation: richer holostuff opcodes (CLEANUP/ENCODE/FACTOR/DENOISE
that call the mind's faculties), a goal-addressable procedure-memory faculty (recall WHICH recipe achieves a
goal), and recipe generation/completion (predict the next op).

Tests: +7 (bit-identical delegation; real-VSA-op correctness; accumulator seeded from a mind vector;
CALL-composition through the mind; decode_step as data; bit-exact procedure->recipe bridge; unknown-procedure
guard). 795 -> 802.

## Richer opcodes: APPLY <faculty> -- a procedure invokes the engine's faculties as steps (milestone 2/4) (shipped)
Milestone 1 wired the VM into the mind but its opcodes were still pure kernel algebra (LOAD/BIND/BUNDLE/
PERMUTE). This milestone lets a procedure call the engine's higher faculties as steps, via ONE general,
extensible opcode rather than a opcode-per-faculty sprawl:

  APPLY <faculty>   means   ACC := faculty(ACC)

The VM gains the opcode and a faculty-name operand codebook (`fac_atoms`, alongside the data and function
codebooks); `run(..., handlers=...)` takes a host-supplied dict {faculty_name -> unary acc->acc map}. The
bare VM has no handlers, so APPLY is a SAFE NO-OP there -- a program with APPLY still assembles, decodes, and
runs everywhere. The mind supplies the handlers (`_procedure_handlers`), each delegating to a real faculty:
- `cleanup` -> the dense associative cleanup (`hopfield.dense_cleanup`) against the procedure's value-atom
  codebook: relax the accumulator toward the nearest known value.
- `denoise` -> the mind's general manifold denoiser.

MEASURED: APPLY cleanup is a real capability a plain list of kernel ops cannot match -- a procedure
SELF-CORRECTS a noisy accumulator. Seeding the accumulator with a heavily corrupted value atom (cosine-to-
truth ~0.07 at sigma 0.5, dim 1024) and running `[APPLY cleanup; HALT]` recovers it to ~0.79 (and to ~1.0 at
the tour's larger dim). Backward-compat is exact (a procedure without APPLY is byte-for-byte identical to a
bare HoloMachine -- the run() signature change is safe), the bare VM runs APPLY programs as a no-op,
decode_step reads APPLY back as data, and a procedure containing APPLY is still a typed B7 structure
reproduced bit-exactly.

KEPT NEGATIVES / honest scope: `denoise` helps only when the accumulator carries low-rank/self-similar
structure; on bare random value atoms there is no manifold, so `cleanup` is the operative denoiser there.
And APPLY is deliberately limited to UNARY acc->acc maps -- the opcodes Moose floated that do NOT fit this
shape are out of scope: FACTOR/RESONATE produce MULTIPLE outputs (no single accumulator to write back), and
value-ENCODE needs a value in the mind's space rather than a faculty applied to the accumulator. APPLY is the
extension point if a unary form of any of those is later wanted (register it in `_procedure_handlers` and
`DEFAULT_FACULTIES`).

As-above-so-below: the opcode lives in the VM (so programs stay self-contained and decodable), but the
SEMANTICS come from the mind's faculties through the handler hook -- the same delegation the whole de-silo is
built on. Real basis: dense associative memory / modern Hopfield cleanup (Ramsauer et al. 2020) as the
recover-toward-the-codebook step; the von Neumann stored-program model extended with a host-call instruction.

Tests: +4 (APPLY cleanup recovers a noisy accumulator; backward-compat bit-identical; APPLY decodes and the
bare VM runs it as a no-op; APPLY procedure->recipe stays bit-exact). 802 -> 806.

## Procedure memory: goal-addressable recall over the library (milestone 3/4) (shipped)
With procedures stored as data in one library vector, the question is whether you can recall the RIGHT one by
what it ACCOMPLISHES -- something a plain list of callables cannot do without manual bookkeeping. Two faculties:
- `recall_procedure(input_vec, output_vec)` -- given ONE (input -> output) example, return (name, score) of
  the stored procedure whose behaviour best reproduces it.
- `recall_and_apply(input_vec, output_vec, new_input)` -- recall that procedure, then run it on NEW input:
  learn an operation from one example, then reuse it (analogy/transfer, VSA-native).

MEASURED (dim 1024): over a MIXED library (bind-b/c/d, permute, bundle-e), behavioural recall identified the
right procedure from a single example 100% of the time, and recall-and-apply transferred the recalled
transform to fresh input correctly 100% of the time. The elegant special case also holds: for a single-bind
transform the operation can be recovered ALGEBRAICALLY in O(1) -- unbind the input from the output and clean
the result against the transforms' keys -- 100% identification with NO candidate runs.

HONEST COST / boundary: the general faculty is BEHAVIOURAL -- it runs each candidate on the input and matches
the output, so it is O(library size) in executions. That is the honest price of recalling an ARBITRARY
procedure by goal: the VSA encoding does not magically avoid running the candidates for general programs. The
O(1) algebraic shortcut is real but transform-specific (bind-parameterised transforms only), so it is shown in
the tests rather than wired as the default. A precomputed behavioural FINGERPRINT (run each procedure once on a
fixed probe, then match goals against the stored fingerprints) would make per-query recall execution-free and
even sublinear via the HoloForest -- a natural next step, noted but not built, and exact only for transforms a
fixed probe characterises (linear ones).

As-above-so-below: recall_and_apply composes the M1 run faculty (executes through the same VM, with the M2
APPLY handlers available), so it is the engine's content-addressable-recall competence pointed at PROCEDURES,
reusing the procedure machinery rather than a parallel store. Real basis: matched-filter / behavioural
identification; HRR unbind+cleanup for the algebraic special case.

Tests: +3 (recall the right procedure from one example over a mixed library; recall-and-apply transfers to new
input; empty-library recall returns None). 806 -> 809.

## Recipe completion: predict the next opcode from a partial recipe (milestone 4/4) (shipped)
The last upgrade closes the loop from running and recalling procedures to GENERATING them: given a partial
recipe, predict the likely next opcode. Two faculties:
- `learn_recipe_grammar(recipes, order=2)` -- learn the opcode-sequence statistics of a set of valid recipes
  (only the opcode stream, i.e. the control SHAPE), into a dedicated token-level predictive model kept
  separate from the mind's prose predictor.
- `complete_procedure(partial)` -- predict (opcode, confidence) for the next step of a partial recipe; an
  empty partial predicts the typical FIRST opcode.

Delegates to the existing PredictiveMemory (token-level n-gram with error-gated writes) rather than a new
predictor -- opcodes are just its symbols. MEASURED on a grammar (LOAD, BIND, then 0-2 of
{BUNDLE,APPLY,PERMUTE}, then HALT): it learned the HARD constraints -- after LOAD it predicts BIND at
confidence 1.0, and after two middle ops it predicts HALT -- and 100% of next-opcode predictions on held-out
partial recipes were grammar-VALID continuations.

KEPT NEGATIVES: it is an n-gram over opcodes, so it predicts by the frequency of transitions it has SEEN --
a recipe shape absent from training is not anticipated well; it predicts the single most-likely next opcode
(the model's soft mode gives a blended estimate); and it learns the opcode SHAPE, not the operands (predicting
the right argument is a larger-vocabulary problem left for later). The degenerate empty-context prediction
returns the right first opcode but at confidence 0 (the zero context vector), which is correct if unsmooth.

As-above-so-below: the grammar is a thin token-level wrapper over PredictiveMemory, so recipe generation
reuses the same predictive-coding machinery the mind uses for sequences -- one predictor design, pointed at
opcodes. Real basis: n-gram / predictive-coding next-symbol modelling.

This completes the 4-milestone procedure arc: M1 de-siloed the VM into the mind, M2 let procedures call the
mind's faculties (APPLY), M3 made the library goal-addressable (recall by example), M4 makes recipes
predictable. Tests: +2 (grammar predicts valid next opcodes incl. the hard LOAD->BIND and the HALT cap;
no-grammar returns None). 809 -> 811.

## Fingerprint fast-path for procedure recall (milestone 5/4 -- the natural follow-on) (shipped)
M3 left an honest cost: behavioural recall runs EVERY candidate through the VM (O(library) executions per
query). This milestone removes that cost for the case it can, measured first. Two pieces:
- `index_procedures()` -- run each procedure ONCE on a canonical probe and cache the output (a behavioural
  FINGERPRINT). One-time O(library) cost, amortised across all later recalls.
- `recall_procedure(..., method=...)` -- 'fingerprint' recovers the transform's kernel from the single
  example and matches the IMPLIED fingerprint with ZERO program runs; 'behavioral' is the M3 scan; 'auto'
  (default) tries the shortcut, trusts it only when its match clears `fp_floor`, and otherwise falls back.

MEASURED (dim 1024): the shortcut is EXACT (confidence ~1.00) for the LINEAR / convolution class, and that
class is larger than expected -- it is bind AND permute and their compositions. The reason is a clean
identity: permutation is convolution by a shifted delta, so it commutes with binding exactly as a key does,
giving bind(P, unbind(permute(X), X)) == permute(P). Across a mixed library (binds, permute, additive bundle,
and a nonlinear cleanup procedure), 'auto' identified the right procedure 100% of the time -- linear ones via
the zero-run shortcut, the rest via the fallback -- and ran ~30x faster than the behavioural scan on a
bind/permute workload (8 ms vs 222 ms for 80 queries). Backward-compatible: with no index, 'auto' IS the
behavioural scan (100% unchanged).

KEPT NEGATIVES / measured boundary: the fingerprint is reliable only for the convolution class. An ADDITIVE
transform (BUNDLE, i.e. x+c) lands borderline (~0.48) because the constant part partially aligns with the
probe, and a genuinely NONLINEAR procedure (one with an APPLY cleanup/denoise step) scores near zero (~0.01).
Both sit below the 0.5 gate, so 'auto' correctly routes them to the behavioural fallback -- the gate is what
makes the speed-up SAFE rather than a source of silent wrong answers. The confidence separation is wide
(1.00 for exact matches vs <=0.48 for everything else), so the gate is robust. One more measured caveat: the
exactness assumes the QUERY INPUT is unitary (so the unbind that recovers the kernel is clean); for a non-
unitary input the recovered kernel is noisier (~0.67 confidence in the tour) -- still well above the gate, so
recall still succeeds, just with less margin.

As-above-so-below: index_procedures runs through the SAME run_procedure faculty (M1) with the SAME APPLY
handlers available (M2), and the fast-path is just the engine's unbind+cleanup competence applied to a
precomputed library -- no parallel machinery. The fingerprints could be dropped into a HoloForest to make
recall sublinear in library size as well (noted, not built; exact only for the linear class a fixed probe
characterises). Real basis: convolution-algebra / transfer-function identification (the kernel of a
shift-invariant linear map is recoverable from one input-output pair); HRR unbind+cleanup.

Tests: +2 (fingerprint 'auto' matches behavioural across a mixed library and is backward-compatible without an
index; the shortcut is exact for bind AND permute and gated below 0.5 for a nonlinear procedure). 811 -> 813.

## Procedure synthesis: CONSTRUCT a procedure for a goal (milestone 6) (shipped)
recall_procedure (M3) finds a procedure already in the library; this milestone builds the missing constructive
counterpart -- `synthesize_procedure(input_vec, output_vec, max_depth=2)` SEARCHES for a short program that maps
input -> output, even when none is stored. It runs a bounded breadth-first search over the VM's operations
(BIND/BUNDLE/PERMUTE x the data atoms), returns the SHORTEST program (as (opcode, operand) pairs ending in
HALT) whose execution reaches the target, and VERIFIES it by running it before returning.

MEASURED (dim 1024): it constructs correct programs for single-op and composite goals -- bind, bind-then-bind,
and the order-SENSITIVE permute-then-bundle (it picks the right order; binding two atoms it may pick either
order, which is fine because binding commutes) -- each verified to map X -> target. Crucially the synthesized
program GENERALISES: run on a fresh input it performs the same operation (cosine >0.99 to the transform's
truth on the new input), so it captures the TRANSFORM, not the example pair -- the structured moves it searches
are what make one example enough. An unreachable target returns None honestly; a depth-3 composite is found
when max_depth=3 is allowed.

KEPT NEGATIVES: the search branches by (ops x operands) per step, so it is EXPONENTIAL in depth -- practical
only for short programs (depth 2-3); it constructs programs only over the KNOWN operations and operands (it
cannot invent a new atom or a nonlinear step); and it may return an EQUIVALENT program rather than a unique
'intended' one. This is the panel's search theme (Baker's landscape search, the flow solver) on the program
space itself: a deterministic, verified pre-screen, with the honest exponential wall stated rather than hidden.

As-above-so-below: synthesis applies the SAME kernel ops the VM executes and VERIFIES through the SAME
run_procedure faculty (M1), so a synthesized program is immediately runnable, decodable (M1), recallable (M3),
and reducible to a typed B7 structure -- it drops straight into everything already built. Real basis: bounded
program search / enumerative program synthesis, verified by execution.

Tests: +3 (synthesize single + composite + order-sensitive programs, all verified; a synthesized program
generalises to a new input; an unreachable target returns None). 813 -> 816.

## Control flow: IFMATCH (conditional) and ITERATE (the fixed-point loop) (milestone 7) (shipped)
Until now the VM had only straight-line execution plus CALL (subroutines) and HALT -- no conditionals, no
loops. So the one pattern that drives most of the engine -- input -> process -> feed the result back as input
-> repeat until the desired output -- could not be written as a PROGRAM, even though the engine runs exactly
that loop inside cleanup, the resonator, denoise, and the diffusion sampler. This milestone adds the two
missing primitives, both reusing the existing machinery:

- `IFMATCH x` -- execute the NEXT instruction only if cosine(ACC, x) >= branch_tol, else skip it (a one-
  instruction conditional; pair it with CALL for an if-then). Implemented by giving run() an explicit program
  counter so a branch can skip forward.
- `ITERATE f` -- re-apply library function f to ACC until it CONVERGES (cosine to the previous ACC >=
  converge_tol, a fixed point), OR a host `stop(acc)` predicate marks the desired OUTPUT reached, OR max_loop
  is hit. The loop body is a named library function, so ITERATE reuses CALL's library-pull. Its trace entry is
  the 4-tuple (op, f, iterations, reason) where reason is 'converged' / 'goal' / 'maxloop' -- the loop tells
  you why it stopped and after how many passes (the "benchmark the result before exiting" visibility).

MEASURED (dim 1024): ITERATE of a one-step cleanup body on a noisy accumulator IS the fixed-point loop --
at low noise (sigma 0.3) it converges to the clean atom (cosine 0.11 -> 1.00) in ~2 iterations, reason
'converged'; the goal predicate exits the instant the output crosses the target (reason 'goal'); and a non-
converging body (a PERMUTE that rotates forever) correctly hits the cap (reason 'maxloop'). IFMATCH branches
both ways: the guarded CALL runs on a match (ACC -> bind(a,b)) and is skipped on a mismatch (ACC stays a),
with the trace showing exactly which path was taken. Backward-compatible: a program with no control flow is
byte-for-byte identical to a bare VM; IFMATCH (data operand) is a typed B7 structure bit-exactly, while
ITERATE (runtime library lookup, like CALL) is out of scope for the recipe bridge.

KEPT NEGATIVES: ITERATE converges to a FIXED POINT, which is the clean atom only when the input is inside its
basin of attraction -- at higher noise (sigma 0.5, 0.7) the loop still converges in ~2 iterations but to a
partial recovery (cosine 0.81, 0.68), not the exact atom (the cleanup's basin shrinks with noise; an honest
property of attractor dynamics, not a bug). IFMATCH is a forward-only skip of ONE instruction (no backward
jumps), so it expresses if-then, not arbitrary goto; there is no general counted FOR loop (convergence/goal/cap
cover the AI case, and a count-as-operand would be awkward in the atom codebook) -- noted as the honest scope.

As-above-so-below: the loop body and the conditional run through the SAME run()/CALL/APPLY machinery, so an
ITERATE can drive a procedure that itself uses APPLY cleanup/denoise (M2), CALLs sub-procedures (M1), or was
synthesized (M6) -- control flow composes with everything already built. This is the engine's own fixed-point
nature (Hopfield/resonator/denoise all iterate to attractors) finally expressible at the program level. Real
basis: fixed-point iteration / attractor dynamics; the von Neumann stored-program model with conditional and
loop control.

Tests: +4 (ITERATE converges to the fixed point; goal and cap exits; IFMATCH branches both ways; control flow
is backward-compatible and the recipe bridge accepts IFMATCH but rejects ITERATE). 816 -> 820.

## matmul in the loop: exact_matmul as an APPLY faculty (backlog VM-1) (shipped)
The control-flow milestone gave the VM a fixed-point loop; this gives the loop a real LINEAR-ALGEBRA step.
`set_matmul(W)` configures a matrix and `APPLY matmul` then does ACC := W @ ACC, carried by the engine's
EXACT RNS matmul (residue-number-system phasor multiply-accumulate -- no crosstalk). With a dim x dim W the
accumulator keeps its shape, so `ITERATE [APPLY matmul]` is a recurrent linear map iterated to a fixed point
-- the literal input -> process-by-a-matrix -> feed-back pattern, the shape of so much of AI.

MEASURED (the marquee demo): a column-stochastic transition matrix iterated by `ITERATE [APPLY matmul]` IS
power iteration -- it converges to the matrix's STATIONARY DISTRIBUTION (the dominant eigenvector, lambda=1).
On a 64-state chain it reached the stationary distribution at cosine 0.9993 in 3 iterations (reason
'converged'), a real iterative algorithm expressed entirely as a VM program. Disabling it (`set_matmul(None)`)
makes APPLY matmul a safe no-op, so the opcode is harmless until configured; the bare VM is unaffected.

KEPT NEGATIVES / scope: the matmul is EXACT for integer / fixed-point operands within range; a float matrix
and vector are fixed-point QUANTISED first, so the only error is that rounding (set by the scale), NOT the
crosstalk wall -- on large-magnitude operands (standard-normal values ~+-3) the default-scale rounding is
visible (~0.12 abs error on a raw matmul), while on well-scaled data like probability distributions it is
negligible (hence the 0.9993 convergence). One configured matrix at a time, and this step treats ACC as a raw
vector -- a deliberate, honest departure from the VSA algebra to do ordinary linear algebra inside the loop.

As-above-so-below: the matmul handler is just another entry in the same APPLY registry as cleanup/denoise
(M2), so it composes with everything -- a loop body can matmul then clean, a conditional can gate a matmul, a
synthesized program could include one. The RNS matmul Moose added becomes the process step in the AI loop the
control-flow milestone made expressible. Real basis: power iteration / Perron-Frobenius (stochastic matrix ->
stationary distribution); Residue Number System exact integer matmul.

Tests: +2 (ITERATE [APPLY matmul] converges to the stationary distribution; disabled matmul is a no-op).
820 -> 822.

## Counted loop: REPEAT n runs the next CALL n times (backlog VM-2) (shipped)
ITERATE is a convergence/goal WHILE loop; this adds the counted FOR loop the VM was missing. `REPEAT n` runs
the FOLLOWING instruction n times -- expected to be a CALL, so the body is a named library function (any block
of work). The count is a small-integer operand drawn from a dedicated codebook (cnt:1 .. cnt:COUNT_MAX, default
8), mirroring how IFMATCH gates the next instruction; REPEAT consumes the CALL that follows it.

MEASURED: REPEAT n; CALL shiftone (a one-PERMUTE body) yields permute(X, n) exactly for n = 1, 3, 5 -- an
exact, countable proof the loop ran the right number of times -- with the trace showing [REPEAT, CALL]. It
decodes back as (REPEAT, count) data and runs on the bare VM. KEPT NEGATIVES / scope: the count is bounded to
the count-atom set (1..8 by default, raise COUNT_MAX to extend); REPEAT repeats exactly ONE following
instruction, which must be a CALL (wrap any multi-op body in a function) -- if the next instruction is not a
CALL, REPEAT is a safe no-op and the next instruction runs once. Like the other control flow, REPEAT is runtime
(it consumes a runtime CALL), so the structural recipe bridge declines it alongside CALL/ITERATE.

As-above-so-below: REPEAT reuses CALL's library-pull and threads the same handlers/loop knobs through the
recursion, so a REPEATed body can APPLY faculties, ITERATE, or CALL further -- it composes with the rest of the
control flow. The VM now has both loop kinds: ITERATE (loop until a fixed point / goal) and REPEAT (loop a
fixed number of times). Real basis: the counted-loop / bounded-iteration control primitive.

Tests: +2 (REPEAT runs the next CALL n times, exact via permute; REPEAT decodes and runs on the bare VM).
822 -> 824.

## Control flow composes: nesting + a worked program (backlog VM-3) (shipped)
Control flow is only real if it nests, so this validates that and ships a worked program as the proof. MEASURED
compositions all reach the right result: a counted loop of convergence loops (REPEAT 2; CALL refine, where
refine ITERATEs -- REPEAT>CALL>ITERATE>CALL>APPLY) denoises to the clean atom; a convergence loop whose body
CALLs (ITERATE double_clean -- ITERATE>CALL) converges; and runs are bit-identical run-to-run (determinism
holds through the nested control flow). The depth guard caps recursion; ITERATE's own iterations do not consume
depth (they reuse one level), so loops nest freely within the guard.

The WORKED PROGRAM is a complete little routine in one procedure: `ITERATE clean_step; IFMATCH c; CALL tag;
HALT` -- denoise the input to a clean atom, branch on the cleaned result, and tag it only if it is 'c'. On an
input that cleans to c it runs [ITERATE, IFMATCH, CALL] and the accumulator becomes bind(c, tag); on an input
that cleans to d it runs [ITERATE, IFMATCH] and skips the tag. Loop + conditional + call working together on
the one substrate -- the proof that the VM is now a real little language. (A learning from the build, kept: a
conditional must come AFTER the denoise, not before -- raw noise at high dimension has cosine ~0.15 to the true
atom, below the IFMATCH gate, so the natural and correct order is process-then-branch.)

As-above-so-below: the worked program uses ITERATE (M7), IFMATCH (M7), CALL (M1) and APPLY cleanup (M2) in one
assembled vector, run through the one VM -- every control and faculty primitive composing in a single program.
Real basis: structured-program composition; the denoise-then-classify routine is the engine's own recall
pipeline expressed at the program level.

Tests: +2 (nested control flow composes and is deterministic; the worked denoise->classify->tag program runs
both branches correctly). 824 -> 826.

## Automatic data-analysis pipeline as a VSA program (PIPE-1, shipped)

The reason the VM grew control flow: a real, useful program that loops a process until it converges and
branches on the result. Handed a 1-D signal, `run_analysis_pipeline` runs ONE HoloMachine program -- not
Python control flow --

    APPLY analyze ; ITERATE _denoise_step ; APPLY decompose ; IFMATCH structured ; CALL _train_validate ; APPLY save ; HALT

-- where each APPLY delegates to a real faculty and the looping/branching is the recent VM (ITERATE/IFMATCH/
CALL). The accumulator carries the signal through the denoise loop; then decompose hands the program back a
`structured` or `noise` FLAG atom, and the IFMATCH branches on it -- the data decides the path.

Measured on a structured signal (1 + 2t + 3t^2 + noise, 256 points): analyze reports a line topology, the
ITERATE denoise loop settles the signal, decompose finds the 2-term quadratic law at explained variance
0.998, the IFMATCH fires, the CALL'd train+validate confirms the law extrapolates to the unseen last 20%
(held-out error 0.10 of the signal's std), and save stores the 256-point signal as a 157-byte generative law
(decompose's compression_ratio 6.5x). On PURE NOISE the same program denoises, decompose finds nothing
(explained variance 0.0, zero terms), the IFMATCH SKIPS the CALL -- train and validate never run -- and save
reports raw_only. Both branches, one program, driven only by what the data turns out to be (executed opcodes
APPLY/ITERATE/APPLY/IFMATCH/CALL/APPLY with structure present vs APPLY/ITERATE/APPLY/IFMATCH/APPLY without).

The denoise loop body needed its own fix. A denoiser is a map of a manifold and a lone signal has none --
denoise(auto) on a bare vector correctly refuses ("no free lunch"). So the prior is built FROM the signal:
its sliding windows form a trajectory (Hankel) matrix that is LOW-RANK for any smooth/structured signal
(classic SSA/Cadzow), so projecting those windows onto their own dominant subspace removes the noise.
`_denoise_signal` delegates that projection to denoise(method='adaptive') and reconstructs by anti-diagonal
averaging; it cuts noise ~3.4x on a structured signal and is ~idempotent (cosine 0.9999 under a second pass),
so the ITERATE settles in a couple of steps instead of spinning to max_loop.

Kept negatives (inherited, surfaced not hidden): decompose_signal fits line-domain elementary laws
(polynomial, exponential) and harmonic laws well and reports NO structure on noise -- a clean branch
discriminator -- but it is NOT a universal fitter: a bare sine on a LINE domain is detected as "line" rather
than "ring" and missed (zero terms), so a purely periodic signal with no trend can slip through as if
structureless. This is the SINGLE-LEVEL pipeline: it finds the dominant law and a residual but does not yet
recurse into the residual; peeling structure layer by layer ("every level") is the natural follow-on (the B8
peel module is exactly that engine). The accumulator's role changes mid-program (signal -> flag atom) -- a
deliberate state transition, not a type error, since each cosine compares same-length vectors.

As above, so below: the pipeline's decompose records the same n_terms a direct decompose_signal call returns
on the final denoised signal, and the program runs through the same run_procedure/HoloMachine path as every
other procedure -- a custom hand-written program of (opcode, operand) tuples executes identically. The faculty
orchestrates, it does not fork. Real basis: Milanfar's denoiser-as-manifold-map (the adaptive projection),
Broomhead-King / Cadzow SSA (the low-rank trajectory prior), and the engine's own decompose_signal
(topology -> matched basis -> MDL-gated law).

Tests: +5 (826 -> 831), in test_procedure_faculty.py.

## Recursive peel: accessing structure on every level (PIPE-1 follow-on, shipped)

PIPE-1's single decompose finds the dominant law and a residual but stops -- "every level" was the part it
left open. recursive=True turns the single `APPLY decompose` into `ITERATE _peel_step`: decompose the
dominant law, peel its residual, decompose THAT, layer by layer, until nothing structured remains. The loop
is the recent VM's ITERATE; it converges on the engine's OWN MDL verdict -- decompose returns n_terms==0 when
its gate admits no term (the same gate that returns 0 on pure noise) -- so peeling stops exactly when the
residual is noise. An `APPLY assess` then records the ladder and flags the train/save branch.

The stop criterion matters and was measured. The first try gated each level on "did it explain >= 30% of the
residual?" -- and that FAILED on the very case the peel exists for: a line trend UNDER a comparable sine. The
trend explains only ~0.3 of the variance noiseless, and less with noise, so the floor rejected the real first
level and peeling never started (0 levels on a noisy trend+sine). The fix is to trust decompose's MDL gate,
not the explained fraction: a level is REAL if the gate admitted a term (n_terms >= 1), however modest its
share. A level can be real and small.

Measured: a noisy trend+sine (0.5 + 2t + sin(2*pi*5t) + 0.2 noise) peels into 3 levels -- line(trend) ->
mobius(periodic) -> line(cleanup) -- cumulative explained 0.997, residual down to 0.04, where a SINGLE
decompose explains only ~0.29 (it fits the trend but is thrown by the sine, or vice versa -- not both at
once). poly+exp is captured in ONE level (its additive dictionary fits both at once) and peeling correctly
stops at one -- it does not invent layers. Pure noise yields zero levels (raw_only, training skipped); a hard
mix (a trend + two sines of different frequency) ALSO yields zero -- detect_topology sees "line" and the MDL
fit admits no term against the strong oscillations, an honest inherited limit of decompose_signal's line/ring
detection.

Kept negatives: on a NOISELESS signal the peel runs to completion and can use a couple of extra "cleanup"
levels -- the harmonic fits leave small Gibbs residue that is itself fit-able -- so 4 levels on a noiseless
trend+sine where ~2 are conceptual; a 1%-of-input negligible-residual guard plus the MDL gate bound it, and
on any NOISY (i.e. real) signal it halts at the noise floor (3 levels, not endless). trend+two-sines finds
nothing because the FIRST topology detection fails on the mixed signal -- peeling can only go as deep as
decompose_signal can see at each step.

As above, so below: each peel level is a real decompose_signal call (the recorded topology and n_terms of
level 1 match a direct decompose_signal on the denoised signal), the ladder of laws is saved through the same
path as a single law (save was unified to handle one law or a ladder), and the whole thing is the same VSA
program with one APPLY swapped for an ITERATE. Real basis: matching-pursuit / iterative residual
decomposition (peel the dominant component, recurse on the residual) under the engine's MDL-gated
decompose_signal.

Tests: +4 (831 -> 835), in test_procedure_faculty.py.

## Deep synthesis, meet-in-the-middle, and the bind/permute collapse (SYN-1: a measured negative + its flip)

SYN-1 asked: extend program synthesis past the depth-2/3 the forward BFS reaches, via a meet-in-the-middle
bidirectional search (forward from the input, backward from the output with inverse ops, meet in the middle --
halving the search exponent). Before building it, the precondition was measured -- and the measurement killed
the plan, in the engine's usual humbling way.

The cleanly-invertible ops are BIND (inverse: unbind) and PERMUTE (inverse: shift back). But that algebra
COLLAPSES. Measured at cosine 1.0000 on every case: two binds are one bind by the product, bind(bind(x,a),b)
== bind(x, a*b); a permute slides through a bind onto x, permute(bind(x,a)) == bind(permute(x), a); so ANY
interleaving of k binds and m permutes applied to x equals permute(x, m) bound by the product of all the
operands -- a depth-(k+m) program is a depth-<=2 canonical one. There is nothing DEEP to find in the
invertible algebra, so a bidirectional search through it buys nothing the M5 fingerprint (one example recovers
the kernel for exactly this linear/convolution class) does not already give.

And the ops where depth genuinely matters -- BUNDLE (its superposition normalizes, which breaks
bind-commutativity, so bind/bundle programs do NOT collapse) and the nonlinear APPLY/ITERATE/IFMATCH/CALL --
are precisely the ops that do not invert cleanly (bundle's inverse is "subtract and de-normalize," lossy with
an unknown scale; APPLY/cleanup are many-to-one). So the meet-in-the-middle backward search has no clean
target on the only programs whose depth is real. The forward-only BFS (M6) at depth 2-3 is the right tool;
its limit is the branching factor, not an algorithm a bidirectional trick could fix. Meet-in-the-middle is
NOT built -- it would be dead complexity. That is the negative, kept.

The flip side is constructive. The collapse is exactly a program OPTIMIZER: `canonicalize_procedure` reduces
any bind/permute program to its minimal form -- the k binds become one bind by the product, the m permutes
stay m unit shifts (this VM's PERMUTE is a fixed shift of 1) -- and verifies the reduction by execution
(cosine 1.0). Measured: a 5-op program (bind;permute;bind;permute;bind) reduces to 3 ops (2 permutes + 1
product bind); five binds reduce to one. It is also an EQUIVALENCE oracle for the invertible algebra: two
differently-written programs that compute the same function reduce to the SAME canonical form (bind a;
permute; bind b and permute; bind b; bind a both canonicalize to permute; bind(a*b)). BUNDLE and the
nonlinear ops are honest BARRIERS -- a program containing one is refused (fully_collapsible=False), not given
a wrong partial answer.

As above, so below: canonicalize verifies through the same run_procedure path as everything else (it executes
the original and the canonical program and compares), and stores the product operand as a real codebook atom
so the canonical program is runnable. Real basis: the convolution algebra (binding is circular convolution --
commutative and associative; a cyclic shift is convolution with a shifted unit impulse) -- the same
FFT-on-a-torus operator the whole engine rests on, here used to prove its own programs flatten.

Tests: +4 (835 -> 839), in test_procedure_faculty.py.

## Sublinear procedure recall: the forest index is premature; vectorize the scan instead (REC-1)

REC-1 asked: index the procedure fingerprints in a HoloForest so recall is sub-linear instead of an O(N)
scan. Measured first, in the engine's usual way -- and the measurement said no, then said what to do instead.

A HoloForest over N fingerprints was built and timed against the linear scan, with realistic queries (the M5
implied kernel matches its fingerprint at cosine ~0.95-1.0, so the nearest neighbour is well-separated). Two
findings: (1) the forest is SLOWER than a linear scan for every realistic library size -- 3-8x slower at
N=50-1000, crossing over only around N~4000 procedures -- and that regime is UNREACHABLE, since the single
`define` library vector holds at most a few hundred functions before bundle crosstalk corrupts decode, so a
4000-procedure library cannot exist in this architecture; the sub-linear index is premature twice over. (2)
Accuracy is fine at realistic noise (forest top-1 == linear top-1 == 1.0 when the query is close to its
target), so the rejection is about speed, not correctness.

The measurement pointed at the real fix. The existing fingerprint recall was an O(N) scan implemented as a
PYTHON LOOP -- one cosine call per candidate. The bottleneck was the loop, not the algorithm. Replacing it
with a VECTORIZED scan -- cache the fingerprints as one unit-normalized matrix at index time, then compute
every cosine in a single matrix-vector product (mat @ qhat) and argmax -- is 6-26x faster than the loop AND
3-7x faster than the forest, at every realistic N. Recall stays O(N) but runs at BLAS speed; the named-subset
path keeps the dict loop (rare). The right answer to "an O(N) scan is slow" was to vectorize it, not to reach
for a sub-linear index that does not pay until a scale the system cannot reach.

As above, so below: the vectorized path returns the SAME identity and the SAME cosine the per-candidate loop
would (rows are unit-normalized so mat @ qhat is exactly cosine), verified against the loop on a real library.
Real basis: a linear nearest-neighbour scan as a single GEMV -- the standard "the constant factor, not the
big-O, was the problem" lesson, measured rather than assumed.

Tests: +1 (839 -> 840), in test_procedure_faculty.py.

## Operand prediction in recipe completion (GEN-1, shipped)

complete_procedure predicts the next OPCODE of a partial recipe from a learned grammar (M4); GEN-1 adds the
OPERAND -- the full next instruction. learn_recipe_grammar now trains a second PredictiveMemory over the
JOINT (opcode, operand) token stream ("OPCODE|operand"), and complete_instruction predicts the next joint
token and splits it back into (opcode, operand, confidence). The opcode grammar is untouched, so
complete_procedure is unchanged.

Measured, with the honest boundary front and centre. When operand USAGE is PATTERNED -- two templates,
a->b->c and d->e->f, the operand determined by context -- operand prediction generalizes to held-out recipes
at accuracy 1.00 (it learns the context->operand map, returning ("BIND","b") after "BIND a" and ("BIND","e")
after "BIND d"). When operands are ARBITRARY per recipe, the operand is unknowable: held-out operand-
prediction accuracy falls to chance (0.23, vs 1/6 = 0.17 for six operands) -- correctly, a random operand
cannot be anticipated. The opcode SHAPE, meanwhile, is predicted operand-independently in BOTH cases (the
opcode grammar ignores operands), so complete_procedure stays the robust call and the operand is a bonus only
where it is patterned.

Kept negative (and a sharp one): the CONFIDENCE is not a reliable abstention signal for operands. An n-gram
context seen only ONCE returns confidence 1.00 -- a single random observation looks as certain as a
thousand-fold pattern -- so a high score does not mean the operand is really predictable. The honest
discriminator is held-out GENERALIZATION, not the confidence the model reports; that is why GEN-1 is
documented by an accuracy measurement, not a confidence threshold.

As above, so below: complete_instruction delegates to the same PredictiveMemory class the opcode grammar uses
(a parallel model over joint tokens, seeded distinctly so the two never mix), and with no grammar it returns
(None, None, 0.0) -- backward-safe. Real basis: an n-gram / Markov sequence model over instruction tokens
(the recipe grammar), here over the joint opcode-operand alphabet rather than opcodes alone.

Tests: +3 (840 -> 843), in test_procedure_faculty.py.

## Above/below sweep: the cleanup-matvec pattern, and a denoiser promoted out of the pipeline (shipped)

An "as above, so below" sweep -- looking for a technique or primitive that is load-bearing in one place and
applies elsewhere. Two findings, both acted on.

(1) The VECTORIZED-RECALL pattern. The core Vocabulary.cleanup already snaps a query to the nearest atom with
ONE matrix-vector product against a cached stack of the stored vectors, not a Python loop of per-name cosines
("a constant-factor win, not a big-O one" -- the argmax is identical because stored atoms are unit length, so
the dot IS the cosine up to the query's norm). The sweep found four more recall paths still written as the
loop, and gave each the same cached-matrix treatment, every one bit-for-bit identical to the loop it replaced:

  * ScalarEncoder.decode -- was re-encoding a 200-point grid AND cosine-scanning it on EVERY call; now the grid
    encodings are built once and cached as a unit-normalized matrix, decode is one matvec. Measured ~224x.
  * archive recall + recall_by_tags -- were looping over stored fingerprints / tag-addresses; now cached
    matrices (invalidated on add()), one matvec each, untagged images masked to score -1 as before. ~4-16x.
  * Lexicon.nearest -- was a per-word cosine over the whole vocabulary; now a matvec against a cached
    (row-normalized) meaning matrix + a top-k, same ranking. Measured ~20-40x on a 500-10000-word vocabulary.
    The cache remembers which meaning dict it was built from, so a re-bootstrap rebuilds it automatically.
  * market nearest_motif -- the per-window cosine loop became one matvec (no cache: past windows are passed in).

  Deliberately LEFT (the principle is "earn it by measurement," not vectorize everything): the region-router's
  scan is over num_partitions (small N, no win); a benchmark accuracy helper is not a faculty path; the
  resonator and reasoning factor-recall already use per-factor matrices. Recorded so the non-action is a choice.

(2) A PRIMITIVE PROMOTED below. The data-analysis pipeline owned a private _denoise_signal: the only way to
clean a LONE 1-D signal (which the denoise faculty otherwise can't do -- a single vector has no manifold, and
nlm needs a patch SET, not a raw signal). It builds the prior from the signal ITSELF -- a smooth signal's
sliding-window Hankel matrix is low-rank (Broomhead-King / Cadzow SSA), so the windows project onto their own
subspace and the signal rebuilds by anti-diagonal averaging. That is a general capability, not a pipeline
detail, so it moved DOWN into holographic_denoise as trajectory_denoise and is exposed as
denoise(method='trajectory') -- the second prior-free denoiser beside nlm. The pipeline's _denoise_signal is
now a one-line delegate to it (bit-for-bit identical, max abs diff 0.0), so the pipeline and any other caller
share one implementation. As above, so below: the faculty method delegates to the module function, and the
pipeline (above) delegates to the faculty method.

Real basis: matrix-vector recall is just the inner-product nearest-neighbour every cleanup already is; SSA /
Cadzow trajectory denoising is the lone-signal classic. Honest negative carried in the new method's docstring:
trajectory denoise has nothing to recover from a STRUCTURELESS signal (its trajectory is full-rank) -- the
prior is the signal's own structure, so a signal without structure can only be shrunk, not restored.

Tests: +5 (843 -> 848): vectorized-recall == loop pinned for scalar decode, archive recall, and lexicon
nearest; the trajectory method shown to clean a lone signal and to be the exact denoiser the pipeline delegates
to. The four faculties' own existing suites still pass unchanged (the behavior is identical, only faster).

### Above/below sweep, second pass (the honest non-finding)

A follow-up pass looked for MORE of the same -- duplicated primitives, faculties re-implementing kernel
machinery, hand-set thresholds that should be data-derived. The honest result is mostly a CONFIRMATION that
the engine is already well-factored, which is worth recording so the absence of changes is a measured choice,
not a missed opportunity:

  * `bind` is centralized -- zero faculties re-implement the FFT circular-convolution by hand; everyone
    delegates to the one kernel operator (the discipline the trajectory-denoise promotion just reinforced,
    already holding for the core algebra).
  * The procedure-matched nulls (`_recognition_null`, `_scan_cue_null`, `_brain_null`) are DELIBERATELY
    separate -- each must match its own recall procedure, so a shared generic null would be anti-conservative.
    Consolidating them would VIOLATE the project's own "procedure-matched nulls" rule; leaving them apart is
    correct.
  * `box_resize` is defined once and shared by the archive and the splat-archive; `learn_dynamics` works on a
    given state SEQUENCE, not a delay-embedded 1-D signal, so it shares no Hankel construction with the new
    trajectory denoiser -- no duplication to fold.
  * The handful of inline cosines that aren't the shared helper are mostly DIFFERENT operations -- a COMPLEX
    inner product (mobius) and a mean-centred correlation (a couple of measurement spots) -- which the plain
    real `cosine` helper would silently get wrong, so they correctly stay bespoke.

The one genuine residue: the Flask app (`app.py`) carried a private `box(img, n)` that duplicated the colour
branch of the shared `box_resize` (and even hard-coded 3 channels, so it would crash on a grayscale image the
shared function handles). It now imports and calls `box_resize` -- one duplicate removed, bit-identical on the
3-channel images it is used on. Cosmetic and app-layer (no test-count change), but a real DRY fix.

Net: the high-value above/below seam was the vectorized-recall pattern and the lone-signal denoiser (first
pass); the second pass confirms the rest of the codebase already honours the discipline, with one app-layer
duplicate folded away. Writing the non-finding down is the same honesty the rest of the project runs on.


## TopK resonator readout: the high-load option (shipped)

The SBC resonator's alternating-projection blend had two readouts -- softmax (the original, blends all atoms)
and sparsemax (Martins & Astudillo 2016, blends only the relevant ones, curing softmax's metastable mixing
and raising capacity at the MIDDLE of the load range). A third, TopK (Gao et al. 2024 -- the k-sparse
autoencoder readout; a point in the same Hopfield-Fenchel-Young energy family), keeps exactly the k largest
atoms and softmaxes over just those. MEASURED on the capacity cliff (all-factors-correct, F=3, B=16, L=64,
vs codebook size N): TopK wins at the HIGHEST load -- at N=110, where softmax, sparsemax, AND alpha-entmax all
collapse to 0.05, topk(k=8) is the only readout still recovering factors (0.23), and it leads at N=50 (0.60
vs sparsemax 0.47). A fixed k keeps k candidates alive where adaptive methods over-prune.

Kept negatives: k must be chosen (k=4 underperformed k=8 badly), and TopK ties or slightly LOSES to sparsemax
in the MIDDLE of the range (N=80: 0.12 vs 0.25) -- so it is the high-load option, not a new default.
alpha-entmax (the principled softmax<->sparsemax interpolation at alpha=1.5) was ALSO prototyped and DECLINED:
it merely tracks sparsemax, finding no sweet spot the annealed resonator benefits from -- the extreme (TopK)
wins, not the interpolation. The Hopfield-Fenchel-Young framework (Santos et al. 2025) is the theory that
unifies softmax/sparsemax/entmax/TopK as one energy family; the measurement still rules.

Above/below: TopK is a readout PRIMITIVE, so it lives in holographic_hopfield beside _sparsemax (the shared
cleanup home, no reimpl) and is imported by the resonator -- one readout family, used wherever a softmax over
similarities is. Threaded through decompose_structure, resonator_confidence, and the mind's
decompose_structure faculty with a `k` parameter; the calibrated-confidence null caches per (readout, k) so
its p-value stays matched to the chosen readout. Backward-compatible: default stays softmax.

Tests: +1 (848 -> 849): test_topk_readout_recovers_and_verifies in test_holographic_sbc.py.


## Support-weighted soft predict: MAP-correct on stochastic successors (shipped)

Re-reading Closure-SDK (faltz009) through the lens of "the substrate stores but does not compute" confirmed
the engine ALREADY owns the predictive loop it inspired (PredictiveMemory: build_predictor / anticipate /
generate_predictive, plus the meaning-level MeaningPredictor). A re-derivation prototype only reconfirmed
that holostuff does context-sensitive prediction (A->B after P, A->D after Q: 1.00 vs 0.55 for Markov),
generalizes to held-out sequences of the same grammar (1.00), and rolls out STABLY over 500 steps with no
drift -- the discrete cleanup resets the representation each step, so the limit is context ORDER (the n-gram
curse), not drift, and under-ordered it degrades to wrong-but-VALID, never garbage. No new faculty earned a
place.

But the Closure-style probabilistic test (a context with two successors at 70/30) surfaced a REAL BUG in the
existing faculty: the SOFT read (zread blend of next-vectors) weighted candidates by resonance only, so two
equally-resonant entries (same context, cosine 1.0) contributed EQUALLY regardless of how often each was
seen -- a 70/30 split blended 50/50 and decoded to the 30% (minority) symbol. The fix is the engine's own
frequency-weighted-superposition insight (the same trick that lets the scene coder count and recover objects
from an unnormalised sum): weight the soft blend by each entry's SUPPORT (its reinforcement count, already
tracked). MEASURED MAP-correct across mixes 60/40..90/10. The same revisit showed the HARD read is itself
fragile near a tie (at 60/40 it returned the minority), so support-weighted soft is now the reliable read on
stochastic streams. zread gained an optional `weights` parameter (default None = unchanged); it has exactly
one caller, so the change is fully backward-compatible.

The honest outcome of the Closure re-read: not a new capability (the predictive loop, itself ported from
Closure, was already shipped) but a precise fix to it -- plus verification that the engine's scattered
prediction faculties ARE a from-scratch attention/predictor that learns WITHOUT autodiff, the project's hard
constraint, vindicated by an independent geometric computer reaching the same place.

Tests: +2 (849 -> 851): test_zread_support_weighting_picks_the_frequent_value and
test_soft_predict_is_map_correct_on_stochastic_successor in test_holographic_predictive.py.


## Soft (sharpened) cleanup for the older resonators (above/below sweep, shipped)

The SBC resonator's readout lesson -- a softmax-SHARPENED blend beats the raw-similarity superposition --
swept downward to the two older resonators, which both used hard/linear cleanups. MEASURED, both win at
high codebook load, with the size of the win set by how strong each resonator already was:

  * BIPOLAR resonator (holographic_resonator.py, elementwise-product binding, sign cleanup): replacing
    sign(B.T @ (B @ est)) with sign(B.T @ softmax(beta * cosines)) gives a 3-25x recovery improvement
    (F=3, D=1024, 5 restarts): N=50 hard 0.30 -> soft 0.93; N=70 0.03 -> 0.77; N=90 0.00 -> 0.53. And it
    needs far fewer restarts -- one soft run (0.47) beats five hard restarts (0.30) at N=50.
  * CIRCULAR-CONVOLUTION resonator (holographic_reasoning.py, the engine's native bind/unbind, continuous
    cleanup): smaller win because it is already much stronger (linear handles N<=30 perfectly). Through the
    real class, N=35 0.90 -> 1.00, N=45 0.90 -> 0.95 (beta=25). The scene coder, which uses it, is already
    near-ceiling on its tiny codebooks (colours 7, shapes 4, textures 4): linear recovers K=7 objects at
    1.00 and K=9 at 0.96; soft reaches 1.00 at K=9 (+0.04, marginal).

THE BETA SWEET SPOT is unimodal and load-bearing (kept negative): too soft (beta<=5 on cosines) is too flat
to converge and fails completely; too sharp (beta>=80, or beta applied to RAW unnormalised similarities,
which is effectively one-hot) collapses the search and also fails. The win lives in the middle -- beta~30
bipolar, ~15-30 circular-conv -- a sharpening that keeps a thin tail of competitors alive so the resonator
can still explore. (The first prototype FAILED because beta=8 on raw sims of scale ~1024 was effectively
one-hot; normalising to cosines and using a fair beta is what surfaced the win -- recorded so the scaling
trap is on the record.)

SCOPING HONESTY: the biggest win (bipolar, 3-25x) lands on factor_composite's LEGACY dense path; the
actively-used circular-conv callers (scene coder, the CRT ResidueSystem in holographic_extras) run small
codebooks where the current cleanup is already near-ceiling, so the practical gain there is marginal. The
lesson genuinely transfers, but its high-impact regime (large codebooks / high load) is one the current
callers rarely hit. Wired therefore as a backward-compatible OPTION (a `beta` parameter on
ResonatorNetwork.factor, default None = the original linear cleanup, no default changed) so any high-load
use -- larger CRT moduli, the legacy path, a future large-codebook factorization -- gets the win for free,
without churning the paths that don't need it.

TWO CLEAN NON-FINDINGS from the same sweep, written down because the absence of a change is a measured
choice: (1) the MoE gate already does sparse top-1 routing, and its docstring records that BLENDING experts
was measured to HURT -- so top-k (k>1) routing would re-introduce a known failure; the lesson does not
transfer. (2) the MeaningPredictor already embodies the frequency lesson by construction: fit_transitions
stores one entry PER OCCURRENCE (no merge), so its coupling-weighted blend is frequency-weighted by
repetition -- verified MAP-correct (after [p,a] with b 70%/c 30% it returns 'b'), and its settle already
uses a top-5 readout. The zread support-weighting fix it would have needed is already there, reached a
different way.

Tests: +1 (851 -> 852): test_resonator_soft_cleanup_beta_recovers_where_linear_fails in test_holographic.py.


## Magic-number sweep: the beta=25 cleanup sharpness, checked and justified (sweep, no code change)

A sweep for magic numbers -- arbitrary constants that gate behaviour and might be derived or calibrated
instead of hand-set. FINDING FIRST: the codebase already runs a strong no-magic-number discipline --
statistical z-floors that say "exceeds noise" rather than a tuned cutoff, natural-largest-gap splits, the
auto coherence floor derived from the store's own distribution, the calibrated decide_confidence that
replaced a hand-set blind_floor. Most thresholds are already data-derived statements, not magic.

The one prominent remaining magic number is beta=25.0 -- the softmax-cleanup SHARPNESS, the default across
Vocabulary.cleanup, dense_cleanup, codebook_denoise, and HopfieldCleanup. Three derivation hypotheses were
prototyped and MEASURED; none cleanly beats a sensible fixed value, so the constant earns its place:

  1. SHOULD beta scale with dimension? The principled window is ln(N) < beta < sqrt(2D) -- the softmax must
     beat the noise pack (N-1 random cosines ~ N(0, 1/sqrt(D))) but not amplify a noise fluctuation into a
     false winner. MEASURED the resonator's sweet-spot beta at D=512..4096: it sits in 15-40 and does NOT
     cleanly track sqrt(D) -- lower beta is favoured on HARDER problems (low D), and beta barely matters when
     D is large (codewords already well separated). No clean formula. beta=25 is within the acceptable range
     at every D (optimal at 2048); a slightly lower 15-20 is marginally more robust across D.
  2. DOES sparsemax remove the beta dependence? For PLAIN (non-iterative) cleanup, YES and cleanly: on a
     continuous manifold, softmax recovery climbs with beta (needs it high enough to stop over-smoothing;
     beta=25 good-not-maximal) while SPARSEMAX recovers ~1.000 across the ENTIRE beta=4..150 range, D- and
     density-independent -- the simplex projection self-adapts its sparsity, so beta stops mattering. The
     magic number's RISK is thus already neutralised where it matters most: the codebase ships sparsemax as a
     measured-better option for the continuous case. KEPT NEGATIVE: in the ITERATIVE resonator sparsemax is
     NOT beta-robust -- it just shifts the sweet spot LOW (beta=8: 1.00 vs softmax 0.53; beta=60: collapses
     to 0.60), because its projection reaches one-hot at lower beta and one-hot breaks the resonator search.
  3. DOES annealing beta (soft->sharp, the SBC resonator's own design) remove the need for a fixed value? For
     the DENSE circular-convolution resonator, NO -- annealing from a very soft beta0=0.5 HURTS at every D
     (0.53-0.80 vs the fixed value's 0.67-1.00): the early super-soft iterations blend everything and waste
     the descent. Annealing works for the SBC (sparse block code) resonator but does not transfer to the dense one.

Bottom line: beta=25 is a JUSTIFIED default, not loose magic -- safe and even conservative for plain cleanup
(where sparsemax already neutralises any beta-sensitivity), and a defensible fixed compromise for the
resonator (where beta is an irreducible tuning knob that does not derive cleanly from the geometry). The
actionable nuances: prefer sparsemax when cleanup beta-sensitivity is a concern; the resonator's beta sweet
spot is readout-dependent (softmax ~25, sparsemax ~8) and mildly load-dependent (lower at low D). No code
change and no test-count change -- writing the measured non-finding down so the three derivation attempts are
not re-run.

## De-Doppler drift detection: a binding-is-a-shift, in radio astronomy (DDD, shipped)

The detection cluster (Tarter, Siemion, Cranmer seats) had every primitive a turboSETI triage needs --
streaming SPRT, FDR, calibrated nulls -- but no faculty that put them on the field's actual signal: a
narrowband technosignature that DRIFTS in frequency (the Doppler shift from relative motion). The whole
detector turned out to be two engine primitives reused, no new machinery:

- A Doppler drift is a cyclic SHIFT of the spectrum over time, and a cyclic shift is the engine's `permute`
  (np.roll) -- the SAME rigid-shift transform holographic_video.py uses for motion-compensated compression,
  and equally a binding: bind(x, delta_k) == permute(x, k), exact to 1e-9. So "de-Doppler integration" -- the
  matched filter that recovers a drifting signal a stationary detector loses -- is just permute-ing each frame
  back by the drift before summing. The bank sweeps every candidate drift; the peak reports BOTH the signal and
  its drift rate.
- The look-elsewhere control over the (drift x channel) grid is `bh_fdr` (dependent / Benjamini-Yekutieli --
  the drift cells overlap, so the tests ARE dependent), exactly as `scan` controls it across channels.

MEASURED on synthetic spectrograms at the field's S/N>=10 regime:
- Stationary integration LOSES a drifting signal (~2.9 sigma, noise level); de-drift at the right rate RECOVERS
  it (~6.2 sigma). The de-drift via the bind kernel is bit-identical to np.roll (the shift-is-a-bind identity).
- Look-elsewhere: naive per-cell thresholding fires on ~100% of pure-noise scans (1664 cells); bh_fdr on ~0%.
- ROC: recall ~96% at 0% false-positive at integrated ~12 sigma (the field's <1%-FP-@-95% bar).
- ON-OFF cadence: a STRONG stationary RFI (S/N 12) that WOULD be detected on its own is rejected ~100% (it
  persists in the OFF pointing), while the drifting ON-only signal is kept ~94%.

KEPT NEGATIVE: below ~10 sigma integrated, recall falls off -- the dependent-FDR correction over the many cells
is conservative (a lone weak signal needs ~5 sigma to clear the multiple-testing bar). Not a flaw but a match:
turboSETI's own search threshold is S/N>=10 precisely because it scans so many places. The detector is as honest
about the cost of the bank as the field is. Structurally this is also the learn_dynamics move (a known operator
advancing/reversing a state): de-Doppler integration == recall_at over the drift operator.

Wired as `UnifiedMind.detect_drifting(waterfall, drifts=None, alpha=0.01, off=None)` beside the scan/stream
detection faculties, delegating to holographic_dedoppler (dedoppler_bank + detect_drifting). Deterministic.
Tests: +3 (852 -> 855), in test_holographic_dedoppler.py.

## Above/below sweep after de-Doppler: pain-point hunt, one optimization shipped (sweep + DDD-opt)

The discipline (Moose): after unblocking something, old negatives may be retired and new paths open; and sweep
for places we are SLOW, skip compression, or do work we don't need. Findings, all measured:

- **de-Doppler bank vectorization -- KEPT NEGATIVE.** Replacing the bank's per-(drift,frame) `permute` loop with
  a fancy-index gather is FASTER for small inputs (6.7x at T=24,F=96) but SLOWER at scale (0.5x -- a regression --
  at T=128,F=2048): np.roll's C implementation beats a fancy-index gather once arrays are large. The shipped loop
  is the right choice. The genuine fast algorithm is the Taylor-tree / fast-folding de-Doppler (turboSETI's,
  O(F log T)), a real algorithm not a quick vectorize -- noted as the scale path, not built.
- **Codebase already well-vectorized.** A grep for per-item cosine/dot loops in hot paths found only ONE
  (`holographic_archive.py` `tags_of`, on small tag lists -- not a pain point). The GEN-1 vectorized-recall sweep
  did its job.
- **SHIPPED FIX -- resonator confidence null keyed by SHAPE, not content.** A cProfile of a confidence workload
  showed `_resonator_noise_null` dominating (12 of 13 s, 833k FFTs). The cache key hashed full codebook CONTENT,
  forcing a multi-second cold fit for every new codebook set. Measured: across five random codebook contents of
  one shape the p-value is IDENTICAL for every decision-relevant agreement (>=0.45 -- the regime where a
  factorization is trustworthy); content only shifts the deep-abstain tail (agreement ~0.27, answer "abstain"
  regardless). So the content hash was redundant work. Dropped it; the null is now keyed on shape
  (B, L, codebook sizes, restarts, iters, readout, k). Result: three DIFFERENT codebook sets of one shape now
  cost ONE cold fit (7828 ms) + two 13 ms cache hits, not three cold fits; one fit per shape for a whole run /
  test suite. Calibration verified UNCHANGED (the corruption sweep p-values match the pre-change A2 result
  exactly: p 0.010 while factors recover, rising to 0.57 as they fail). The resonator's FFT inner loop has a
  further ~30-50% available by caching per-factor rffts across the leave-one-out binds, but that touches the
  tie-sensitive per-block kernel (the bind_batch lesson) so it was NOT taken -- the cache-key fix is numerically
  identical and safe.

Tests: +1 (855 -> 856), in test_integration.py (test_resonator_null_keyed_by_shape_not_content).

## Self-verifying storage: the holographic Merkle tree (BLD-1, shipped)

The one genuinely new capability the cross-project comparison surfaced: tamper-evidence as an O(log n) property
of the structure itself, with no prior engine analog -- and it rides entirely on the two kernel primitives, so
it stays in-substrate. A segment tree whose COMBINE is bundle over POSITION-bound items: leaf_i = bind(pos_i,
item_i); each internal node = sum of its children; root = the whole-store composite. DETECT by rebuilding the
root from current items and comparing; LOCALISE by descending from the root, following the child whose composite
no longer matches its committed value -- the changed item in <= log2(n)+1 comparisons (root check + descent
depth), independent of n.

Measured (the BLD-1 bar, deterministic): a single full tamper localised 40/40 in <= log2(64)+1 = 7 checks, 0
false positives on clean data; a slot SWAP detected and localised (position binding defeats the bundle's
commutativity -- without it a reorder is invisible).

KEPT NEGATIVES (measured, load-bearing):
1. LINEAR, NOT CRYPTOGRAPHIC -- the headline. The root is a linear combination, so items -> root is
   R^(n*D) -> R^D, many-to-one for n>1: collisions EXIST and are CONSTRUCTIBLE. A key-aware adversary picks any
   change da to item a, then changes item b by db = deconv(-bind(pos_a, da), pos_b) so bind(pos_b, db) exactly
   cancels it -- the root is bit-for-bit unchanged (measured: cosine 1.0000, an invisible forgery). A
   cryptographic Merkle tree resists this (a hash collision is hard; cancelling a linear sum is a division). So
   the guarantee is evidence of ACCIDENTAL corruption / uncoordinated tampering, NOT tamper-proofing.
2. SINGLE-TAMPER LOCALISATION -- the descent follows one differing child, so several uncoordinated changes are
   detected at the root but only one path is returned per pass.
3. O(n) SPACE -- localisation needs the per-node composites kept; a root-only commitment is O(1) but detect-only.

A guess the plan got WRONG, kept on the record: quantising the stored checksums to save space was expected to
create a small-tamper detection floor (the "keep leaves in capacity" worry that bounds superposition elsewhere).
Measured, it does NOT bind here -- detection stays 100% down to 2-bit checksums even at n=1024, because in high
dimension a tamper always pushes some component across a quantiser boundary. So the checksums are kept exact
float and the quant option is not exposed.

Tests: +5 (856 -> 861), in test_holographic_verify.py (selftest wrapper + single-tamper localise-in-log-checks +
position-binding-catches-reorder + the linear-collision kept negative + the verify_store faculty round-trip).

## External-baseline benchmark harness (BLD-2, shipped)

The field's most convincing habit -- "N times the standard tool for a real job" -- adopted with the project's
discipline: the case where the standard tool WINS stays on the record. INV-2 first audited which baselines are
even reachable in NumPy/Flask-only (sklearn/faiss/statsmodels are banned): general-purpose compression has a
fair stdlib opponent (zlib/lzma), exact NN is a fair opponent for sublinear recall; denoising/forecasting/
classification have no fair in-constraints standard tool, so they are NOT benchmarked rather than measured
against a strawman.

Two runnable, deterministic comparisons in benchmarks/ (numbers checked into benchmarks/README.md):

bench_compression.py -- the geometry-preserving rd code (quant='rd': KLT + water-filling + rANS) vs int8 fed to
zlib/lzma, at matched cosine fidelity. MEASURED (bits/vector): on rank-8 structured data rd wins ~34x at N=2000
(111 vs ~3800; the KLT basis amortizes over the batch, so the win grows with N); on FULL-RANK RANDOM data rd
LOSES (6851 vs 3826 at N=2000) -- no low-rank to exploit, the basis costs more than it saves. The kept negative:
rd is the right tool exactly when the data is low-rank, which the engine's stored states are and random vectors
are not.

bench_recall.py -- the HoloForest (sublinear approximate NN) vs an exact items@query scan. MEASURED: the forest
matches exact recall@1 up to ~10k items and touches a shrinking fraction of the comparisons (467/826/952/616 =
93%/41%/11%/3% as N goes 500->20000); recall@1 holds 100% to 8k, 97% at 20k. The kept negative: on WALL-TIME the
exact scan is one BLAS matvec, so fast that the forest's pure-Python traversal only overtakes it past ~20k items.
The forest buys sublinear WORK (what matters when comparisons are expensive or N is large), not raw wall-clock at
small N against a tight BLAS loop.

Tests: +3 (861 -> 864), in test_benchmarks.py (rd-wins-on-structure-loses-on-random + reproducibility +
forest-sublinear-and-near-exact).

## INV-5: unblockable-negative profiling pass (shipped)

The discipline: unblocking something (denoising, learned energy, SBC, the rd-code, the vectorized-recall
pattern) can retire an old negative -- old results may no longer hold -- so re-profile the heavy non-confidence
workloads and re-scan for missed loops. cProfile + a STATIC scan for the loops INV-1's cosine-grep could not
see: loops that call a similarity HELPER rather than a literal cosine / @ / np.dot.

Found ONE real win, on the record as a table:
- FHRR PhasorVocabulary.cleanup -- a per-candidate `for nm in names: fhrr_sim(noisy, self.vectors[nm])`. INV-1
  missed it precisely because fhrr_sim is a helper, not a literal cosine. VECTORIZED to two REAL matvecs
  (real(vdot(b,q)) == Re(b).Re(q) + Im(b).Im(q)) with the stacked real/imag matrices cached -- ~11x at k=2000
  (19.6 -> 1.8 ms), argmax-identical, sims matching to ~1e-16. The honest subtlety: the OBVIOUS vectorization
  `np.real(B.conj() @ q)` is NO faster than the loop, because B.conj() allocates a full complex copy whose cost
  matches the matvec -- the real-matvec-no-copy form is the actual win. SHIPPED.
- archive recall_by_tags -- its for-loops are TAG set-logic (words/nums), not similarity-over-N; not a target.
- Lexicon.nearest -- already vectorized in the prior above/below sweep; the residual loops format the top-k.
- creature decide -- has per-action loops, but it is the TIE-SENSITIVE maze path the bind_batch lesson keeps
  hands off (a 1e-12-identical change once flipped a trajectory). NOT touched, by policy.
- recurrent/_next_dist, schema/_ppm_dist -- small-alphabet n-gram/PPM distributions, not similarity-over-N.

So the pass confirms the engine is otherwise well-vectorized (reinforcing INV-1 and OOS-1): one genuine missed
loop, fixed; everything else either already done, not a similarity loop, or deliberately tie-protected.

Tests: +1 (864 -> 865), in test_holographic_fhrr.py (vectorized cleanup matches a brute-force loop on both the
all-atoms and the subset paths, and the cache invalidates when a new atom is minted).

## BLD-3: theory-and-guarantees document (shipped)

The project's notion of rigor, finally consolidated. THEORY.md gathers the load-bearing claims into one place
and tags each by what backs it: [CITED] for a literature theorem (Plate HRR capacity, Smolensky tensor binding,
Ramsauer/Krotov modern Hopfield, Wald SPRT, Benjamini-Hochberg/Yekutieli FDR, Tero flow, Duda ANS, Dasgupta
RP-trees), [MEASURED] for a result proven here with a named test pointer, [KEPT NEGATIVE] for a measured limit.
The rule: no claim appears without either a citation or a test. Sections -- the algebra (the
bind=convolution=phase=tensor identity), capacity (the cliff + the modern-Hopfield superset + FHRR), geometry &
compression (consolidation/KLT + rd vs zlib), search (RP-forest + Tero), detection & honesty (RecallNull + SPRT
+ FDR), self-verifying storage (the linear-collision negative), determinism (the bit-exact tie-break
discipline), and a table of the standing kept-negatives.

A test (test_theory_references.py) parses THEORY.md and asserts every `test_file.py::test_function` it cites
resolves to a real function -- so the document is SELF-BACKING and cannot rot into asserting a test that no
longer exists. This is the doc's own guarantee, in the engine's measure-don't-assert spirit.

Tests: +1 (865 -> 866), in test_theory_references.py.

## BLD-4 & BLD-5 (de-Doppler scale/precision): prototyped/reasoned, NOT wired -- kept so they aren't re-tried

Both were the conditional ("only if scale/precision matters") tail of the BLD list. Resolved honestly rather
than built speculatively:

BLD-5 -- sub-bin de-drift via a fractional (Fourier phase-ramp) shift instead of the bank's integer-bin
`permute(wf[t], -int(round(d*t)))`. PROTOTYPED and MEASURED. First, a real bug to note: the 2-D `fourier_shift`
is NOT 1-D-safe -- on a 1-D frame its `reshape([F,1])` broadcasts `(F,)*(F,1)` into an `(F,F)` array (it
silently produced a spurious 50-sigma peak until a correct 1-D phase-ramp shift was used). With the correct 1-D
shift: a MODEST peak-SNR gain on sub-bin drifts, largest at half-integer rates where integer rounding is worst
(+0.7 sigma at drift 0.5; marginal at 0.3; tied on integer drifts). CRUCIALLY, NO drift-rate-resolution benefit:
on a fine (0.05) drift grid the INTEGER bank already recovers sub-bin drifts (0.25/0.5/0.75) with ZERO error --
the matched-filter peak lands on the true drift regardless of the per-frame rounding. So the item fails its own
bar ("strictly better drift-rate recovery"): a fraction-of-a-sigma SNR gain, no resolution gain, an FFT-per-frame
cost. Does NOT earn its place; the integer de-drift stays. Kept negative.

BLD-4 -- Taylor-tree / fast-folding de-Doppler (O(F log T) vs the bank's O(F*T*n_drift)). PARKED, on the same
discipline as OOS-1 (the native-kernel rejection): a SCALE-ONLY optimization with no demonstrated current
bottleneck (the bank runs fine at the verified spectrogram sizes), and the tree's bit-identity is non-trivial
for the FRACTIONAL drift grid the bank searches (the classic Taylor tree computes integer-slope sums). The right
trigger is a real SETI-scale search that profiling shows the loop cannot clear -- with numbers in hand, not
before. The Taylor-tree is the known approach when that day comes.

## BLD-7: N-dimensional fractional power encoding + compute-on-functions (shipped)

The backlog framed FPE as a thing to BUILD to beat the "locality-preserving RBF approximation." Measurement
flipped the premise: the 1-D ScalarEncoder is ALREADY a fractional power encoder. encode(x) =
irfft(exp(i*scale*x*phases)) is literally "raise the base to power x"; its kernel_at is the Bochner kernel
(sinc/rbf); and because the engine's bind is circular convolution -- spectrum multiply = phase add --
bind(encode(x), encode(s)) == encode(x+s) to numerical exactness (cosine 1.00000). So 1-D FPE, with
shift-as-bind and a designed kernel, has been here all along; "beat the RBF encoder" is moot because it IS the
FPE/RBF encoder. (Pinned in test_holographic_fpe.py::test_scalar_encoder_is_already_fpe_shift_as_bind_and_kernel.)

The genuine addition (holographic_fpe.py, faculty m.vector_function_encoder) is the step up from a scalar to a
VECTOR domain and to FUNCTIONS, which the engine did not have:
  * N-D encoding -- a point in R^n is encoded by binding one per-axis 1-D FPE per coordinate. A shift along any
    axis is still ONE binding (2-D shift-as-bind cosine 1.00000), and the kernel is the PRODUCT of the per-axis
    kernels (measured 0.920 vs the 0.914 product at a 2-D offset) -- an n-D RBF for rbf axes.
  * Compute on functions -- f: R^n -> R as a weighted superposition of encoded points, f = sum_i w_i encode(p_i);
    querying reads sum_i w_i kernel(q, p_i) (a holographic KDE: high at placed points 0.81/0.69/0.78, low at an
    empty spot 0.06); and the WHOLE function translates by ONE binding, bind(f, encode(delta)) = sum_i w_i
    encode(p_i + delta) -- the rigid-shift-is-a-bind trick the motion compensator uses, lifted to a function.

KEPT NEGATIVES (measured in the selftest): the standing capacity cliff applies -- a function is a bundle, so
query separation (placed vs empty) decays as atoms pile up (+0.39 at K=2 -> +0.01 at K=128); and where a scalar
suffices the n-D machinery buys nothing, since 1-D FPE IS the ScalarEncoder. Reuses the verified ScalarEncoder
per axis (DRY) and the engine's bind/cosine -- no new dependency, nothing learned.

Tests: +9 (866 -> 875), 8 in test_holographic_fpe.py + 1 faculty test in test_integration.py.

## EXP-5 + EXP-6: the spectral structure kernel + the Laplacian eigenbasis basis-selector (shipped)

One operator, two readings. holographic_spectral.py builds the graph and Hodge Laplacians from a point cloud or
a simplicial complex and exposes their eigendecomposition (C2: eigenvector signs fixed -- largest-magnitude
component positive -- so the basis is reproducible, the bind_batch tie-break class of bug).

EXP-5 (the operator + its sanity): the cycle-graph Laplacian eigenbasis IS the DFT (eigenvalues 4 sin^2(pi
k/n)); a ring signal reconstructs from it to 1e-15. And the Hodge Laplacian's HARMONIC dimension equals the
Betti numbers -- 4-cycle (1,1), filled triangle (1,0), two components (2,0) -- the spectral route to topology
EXP-7 will use.

EXP-6 (the basis-selector, generalising decompose_signal's hand-picked list): the path-graph Laplacian
eigenbasis IS the DCT/elementary basis (line denoise 1.134 == DCT 1.134, identical to 1e-6), and the cycle's IS
the harmonic basis -- so the Laplacian eigenbasis SUBSUMES the line->elementary / ring->harmonic special cases.
And it extends to manifolds the topology detector can only call "line": on a sphere, the kNN-Laplacian low
eigenvectors recover a smooth degree-2 field (denoise 1.925 from a noisy 6.112) where the line/index-order basis
barely helps (5.235). The data-driven basis is measurably right where the hand-picked fallback is not. Faculty
m.spectral_basis(points, k, n_basis) -> a SpectralBasis with decompose/reconstruct/denoise.

KEPT NEGATIVES (C1/C2, on the record in the module): dense eigh -> moderate N (a huge sparse graph would need
scipy, a banned dependency); and where the manifold genuinely IS a simple line or ring, the hand-picked basis is
cheaper and exact -- the data-driven basis earns its place only where the topology is unknown or off the
hand-coded list. Only NumPy; nothing learned.

Tests: +8 (875 -> 883), 7 in test_holographic_spectral.py + 1 faculty test in test_integration.py.

## EXP-7: principled topology by persistent homology (shipped)

detect_topology names a 1-D signal's shape from a hand-coded menu (line/ring/mobius/torus via harmonic fits).
EXP-7 (holographic_topology.py, faculty manifold_topology) reads topology straight off a point cloud: build a
Vietoris-Rips complex at scale eps, count holes by dimension (Betti numbers B0/B1/B2 = components/loops/voids
via B_k = n_k - rank(d_k) - rank(d_{k+1})), and keep the signature that PERSISTS across a scale band auto-set
from the cloud's median NN distance. Reproduces detect_topology on its cases (contractible -> (1,0,0) "line",
loop -> (1,1,0) "ring") and extends to ones it can't name: torus (1,2,1) and sphere (1,0,1). B1 orders
line(0)<circle(1)<torus(2); B2 is what tells a sphere (1) from a line (0), both B1=0.

THE FIX THAT MADE IT WORK: the first cut computed Betti via dense real np.linalg.matrix_rank and TIMED OUT on
the torus (the d3 rank on a VR complex is too big for an O(n^3) SVD). Reducing the boundary matrices over GF(2)
-- columns as integer bitmasks, XOR twist reduction (the standard PH approach) -- is exact and drops the torus
from a timeout to ~0.5s. It is Z/2 homology (= integer homology absent 2-torsion, which none of these shapes
has), and it AGREES with EXP-5's Hodge-Laplacian harmonic-dimension Betti route on every fixed complex (pinned
in the selftest -- two operators, one answer).

KEPT NEGATIVES (measured): PH is finicky on small/noisy/UNEVENLY sampled clouds -- a sine's delay embedding
(dense at turning points, sparse between) fragments at the median scale (reads (32,0,0), not a ring), so this
reads a WELL-SAMPLED manifold's topology, not an arbitrary trajectory; the delay-embedding bridge to
detect_topology's 1-D signals needs even resampling first. It is BLIND to non-topological geometry by design
(a circle and an ellipse are both (1,1,0)) -- pair it with EXP-6 geometry. And cost grows fast with the
complex, so the cloud is subsampled to max_points and dense reduction bounds it to moderate N (C1).

Tests: +9 (883 -> 892), 8 in test_holographic_topology.py + 1 faculty test in test_integration.py.

## EXP-8: the Helmholtz-Hodge decomposition of an edge flow (shipped)

The same boundary operators that count holes (EXP-5/7) take an edge flow APART. Added to holographic_spectral.py
(hodge_decomposition, denoise_flow; faculties on UnifiedMind). Any flow on a graph splits into three L2-ORTHOGONAL
parts: flow = gradient + curl + harmonic.
  * GRADIENT = d1^T phi -- curl-free transport from a vertex potential (the source-to-sink part).
  * CURL     = d2 psi   -- divergence-free circulation around the filled triangles (local rotation).
  * HARMONIC = remainder -- both div-free AND curl-free: the GLOBAL circulation wrapping the holes. Its dimension
    is exactly B1, so the harmonic part IS the flow's topology (meets EXP-7 on one complex).
Computed by least-squares solves of the graph Laplacian (for phi) and the triangle Laplacian (for psi); the
harmonic part is what neither explains.

MEASURED: the split sums to the flow at 1e-16 and the parts are orthogonal to 1e-15; the harmonic part of a
random flow is div-free and curl-free to 1e-15; denoising a transport flow (drop curl) returns 1.088 from a noisy
1.915 and well past naive edge-smoothing's 3.592 (which over-smooths). KEPT NEGATIVE: on a TREE (no cycles, no
triangles) curl and harmonic are exactly zero -- nothing to circulate -- so all flow is gradient; this falls
straight out of the topology (B1=0, no triangles). For the Tero flow solver and graph-signal denoising.

Tests: +5 (892 -> 897), 4 in test_holographic_spectral.py + 1 faculty test in test_integration.py.

## EXP-9: Clifford Cl(3,0) geometric algebra as a parallel binding mode (shipped)

A second way to bind, alongside circular-convolution bind -- the geometric product of Cl(3,0) (8-dim
multivectors, holographic_clifford.py, faculty m.clifford()). Like tensor_bind, NOT a drop-in: a parallel mode
whose seat is GEOMETRIC structure, specifically 3D rotations. The product is built from the blade Cayley table
(blades as bitmasks over {e1,e2,e3}; product = XOR of masks + a reordering sign; e_i^2=+1).

THE WIN IT IS BUILT FOR (measured): a rotor R = cos(t/2) - sin(t/2) B rotates a vector by the sandwich v' =
R v R~. Composing rotations is EXACT -- the geometric product of two rotors IS the rotor of the composed
rotation (max err ~1e-15 over 200 random pairs vs applying them in sequence) -- and NON-COMMUTATIVE (the two
orders of a pair land ~0.66 apart on a probe vector). HRR's circular convolution is COMMUTATIVE, so it returns
one answer for both orders and carries that whole order-gap as unavoidable error; that gap, which convolution
provably cannot close, is the concrete sense in which Clifford beats it on rotations. Rotors are length-
preserving and exactly invertible by their reverse (~1e-15).

KEPT NEGATIVES (measured / on the record): 2^d DIMENSION GROWTH -- Cl(n,0) needs 2^n components (Cl(3,0)=8,
Cl(10,0)=1024), affordable only for low-dim geometric domains, not a general high-D substrate (HRR's fixed-D
FFT bind is the right tool there). And it binds VERSORS, not arbitrary atoms -- a unit rotor times its reverse
is the identity (clean unbind), but a random multivector times its reverse is NOT, so this is a geometric-
transform algebra, not a general key->value memory like HRR. Narrow win-condition: a parallel tool for the
rotation-shaped corner, like tensor_bind is for the capacity corner.

Tests: +9 (897 -> 906), 8 in test_holographic_clifford.py + 1 faculty test in test_integration.py.

## BLD-8: optimal transport by Sinkhorn (shipped)

A transport-geometry distance between distributions, for the case bin-wise metrics get wrong. Euclidean/cosine
compare two distributions height-by-height and are blind to WHERE the mass sits: two histograms with no overlap
are maximally far no matter how far apart they actually are (a peak at bin 12 reads as distant from bin 10 as a
peak at bin 40 does). The Wasserstein (earth-mover's) distance measures the least work to MOVE one onto the
other -- mass times the ground distance it travels -- so it keeps growing as distributions move apart even with
no shared support. holographic_transport.py (faculty m.wasserstein(a, b, cost, eps)) computes it by the Sinkhorn
algorithm: add an entropy term -> a Gibbs kernel K = exp(-C/eps) and a pair of alternating diagonal rescalings
(u <- a/(Kv), v <- b/(K^T u)) converging to the transport plan P; the distance is <P, C>.

MEASURED: matches the 1-D closed form W1 = sum |CDF_a - CDF_b| to ~1e-3 (W=10.000 on a shift-10 pair). The win:
a shift of 5/10/20 reads as a distance of 5/10/20, while Euclidean saturates flat at ~0.53 for every
non-overlapping shift and cosine collapses to ~0 -- both unable to tell a near miss from a far one. A custom
cost matrix changes the geometry (a ring/cyclic cost lets mass wrap around, shrinking the distance).

KEPT NEGATIVES (measured): THE EPS KNOB -- too LARGE blurs the plan toward the independent coupling and inflates
the distance (a same-mean narrow-vs-wide pair with true W1~3.6 reads ~4.9 at eps=50); too SMALL underflows the
kernel between separated supports (exp(-C/eps) rounds to 0) into a broken answer (eps=0.01 on a shift-10 pair
returns 3.4 instead of 10). The default RULE scales eps to the cost (0.02 * median nonzero cost), sharp without
underflowing for well-conditioned cost matrices; a wide cost range wants an explicit eps or a log-domain solver
(out of scope). Also O(n*m) per iteration (dense kernel + matvecs). And the entropic self-distance W_eps(a,a) is
a small positive bias, not exactly 0 (debias via the Sinkhorn divergence if an exact zero is needed) -- here
self << cross is what matters.

Tests: +9 (906 -> 915), 8 in test_holographic_transport.py + 1 faculty test in test_integration.py.

## Above/below sweep: the Tero flow solver wired to the Hodge decomposition (shipped)

A sweep of the three new geometry kernels (spectral / topology / transport) for where they belong lower or
apply elsewhere. The genuine, load-bearing finding: the Tero flow-conductance solver (holographic_flow.py)
computes a Poiseuille flux Q_uv = D_uv(p_u - p_v) on every edge each step, uses it to thicken tubes, and
DISCARDS it once it has the path. That flux is exactly the edge flow EXP-8's Helmholtz-Hodge decomposition
takes apart.

WIRED: `tero_flux(nbr, start, goal)` exposes the converged signed flux (refactored to share a `_tero_converge`
helper with tero_solve, bit-identical -- flow tests still green). Faculty `flow_circulation(nbr, start, goal)`
splits it: GRADIENT = net source->goal transport (its divergence is the injected current, max 1.000 = I0),
HARMONIC = circulation around the graph's loops (dimension = B1, the loop count EXP-5/7 already measure). A maze
graph has no filled triangles -> no curl. Returns {loops (B1), redundancy (harmonic energy fraction),
transport_energy, circulation_energy, flux, edges, n_vertices}.

MEASURED: on a 4x4 grid, B1=9 = E-V+1, gradient divergence exactly I0 at source/goal (0 elsewhere), harmonic
subspace dim == B1. The harmonic FRACTION is a previously-hidden read on the flow -- how much of the converged
flux circulates rather than transports: EXACTLY 0 on a tree (forced route), and on a loopy grid it varies with
mu (5x5 grid: 0.73 at mu=4 down to 0 at mu=1 -- at high mu, competing thick tubes leave more circulating flux).
This connects three kernels: the flow solver + EXP-8 (Hodge) + EXP-7/5 (B1 topology).

A RESTRAINT kept on the record (also a sweep outcome): the flow solver builds its OWN conductance-weighted
Laplacian, deliberately NOT rerouted through the shared graph_laplacian -- the dynamics is tie-sensitive and a
different summation order could flip a trajectory (the bind_batch lesson). The shared HELPER was extracted
inside the module; the shared KERNEL was left alone. Not everything that looks duplicated should be merged.
Two other candidates were audited and rejected as forced: decompose_signal (1-D symbolic regression, not a
point-cloud basis projection) and Wasserstein-into-the-honesty-layer (the score distributions are already
characterized by their thresholds/likelihood ratios).

Tests: +6 (915 -> 921), 5 in test_holographic_flow.py + 1 faculty test in test_integration.py.

## Above/below sweep of the geometry toolkit: spectral denoise wired, transport/topology kept standalone (shipped)

After the geometry toolkit (BLD-7, EXP-5/6/7/8/9, BLD-8) the standing above/below discipline was run over the
three new kernels (spectral / topology / transport), grounded in a live-code audit. The honest result: most
connections were ALREADY in place, and exactly one genuine gap was found and wired.

ALREADY WIRED (confirmed, not re-done): the Tero flow solver -> Hodge split (transport vs circulation,
flow_circulation faculty) was already built; the spectral and GF(2) boundary-operator routes corroborate (the
selftest pins their Betti agreement) rather than duplicate; the flow solver's conductance-weighted Laplacian is
deliberately NOT routed through the shared graph_laplacian (tie-sensitive dynamics, the bind_batch lesson).

THE ONE GENUINE WIRING -- denoise(method='spectral', points=...): the unified denoise faculty could map a LINEAR
subspace (manifold/adaptive, which need an example set) but had no map for a CURVED manifold's geometry, and none
of its methods could clean a lone scalar field on a point cloud. The EXP-5/6 graph-Laplacian eigenbasis is exactly
that map. New signature params: points=<(N,d) coordinates>, spectral_k=10, spectral_nbasis=12; x is the field over
those N points. MEASURED on a smooth field over a 2-sphere: cleans error 4.078 -> 0.862, where the geometry-blind
options barely move it (trajectory/SSA 3.113, fixed DCT low-pass 4.182) -- a linear/1-D prior cannot see a curved
manifold's smoothness. It is the only denoiser in the faculty needing no example set and no codebook, just the
cloud's own geometry (the nonlinear-manifold completion of Milanfar's "denoiser = manifold map" framing).

KEPT STANDALONE (a finding, not a failure): Wasserstein found NO existing bin-wise distribution comparison to
improve -- the market compares price WINDOWS by cosine and forecasts by proper score, the de-Doppler search uses
the field-standard permute+matched-filter bank; persistent homology is blocked from the 1-D detect_topology by
its delay-embedding uneven-sampling negative. Forcing either would be churn.

Tests: +1 (921 -> 922), test_spectral_denoise_faculty in test_integration.py.

## Self-hosting: PnP restoration and B10 generation moved into VSA programs (shipped)

Following the self-hosting audit (which mapped the boundary: a Python procedure is movable into a VSA program
iff it is ORCHESTRATION whose every step is a hypervector->hypervector map and whose state is one accumulator),
two of the engine's remaining canonical iterate-to-fixed-point loops were re-expressed as programs running on
the HoloMachine -- joining PIPE-1 (the data-analysis pipeline) and the matmul-iterate (a recurrent linear map)
that were already there. Neither replaces its fast Python faculty; each adds the BEING-DATA form (a stored,
composable, recipe-savable procedure -- process, not object).

RESTORE -- Plug-and-Play/RED as a program: ITERATE [APPLY datafit; APPLY denoise]. `restore_procedure(y,
forward, adjoint, samples, mu)` fits a manifold prior from `samples`, configures the inverse problem
(`set_inverse_problem`), and runs the two-step body to a fixed point. The `datafit` handler is the gradient
step ACC <- ACC - mu*adjoint(forward(ACC)-y); the `denoise` handler is the prior. MEASURED: on a half-masked
low-rank signal (raw rel-error 0.863) it recovers to rel-error 0.167 -- the SAME error-to-truth as the Python
`pnp_restore` -- converging in 6 ITERATE iterations where the Python loop runs a fixed 40.

GENERATE -- B10 diffusion as a program: ITERATE [APPLY diffuse] from a noise seed. `generate_procedure(codebook,
steps, seed)` configures a self-cooling `diffuse` handler (`set_generator`) that anneals beta up and injected
noise down per call; ITERATE halts when the sharpened cleanup reaches a fixed point on the manifold. MEASURED:
lands on the codebook manifold at cosine 1.000 (the same sample `hopfield.generate` produces), in 13 iterations
(the 12-step schedule + the convergence check). The generative PROCESS is now a stored, composable procedure.

The schedule wrinkle is the one non-obvious bit: `generate` is a SCHEDULED loop (beta/noise change per step),
which a bare ITERATE (same body each step) cannot express -- so the schedule lives in the stateful `diffuse`
handler (it carries the step counter and advances), and ITERATE's fixed-point stop naturally coincides with the
cooled, sharpened cleanup converging. PnP needs no such trick: it is a TRUE fixed-point iterate (same
datafit+denoise body each step), the cleanest ITERATE fit.

KEPT NEGATIVES: (1) the PROCEDURE TAX -- a noisy unbind-and-clean per instruction read makes the program form
slower than the direct Python loop; this is the price of being data, not a faster path. (2) the numerics never
leave NumPy -- datafit/denoise/diffuse are NumPy faculties behind APPLY; the FFT bind / SVD / Sinkhorn cannot
themselves become programs (they ARE the substrate -- circular). (3) B10's own negative travels: a BARE codebook
generation converges to a stored atom; feed a composed/continuous manifold for novel-but-valid samples. (4) the
capacity cliff still bounds program length (~32 instr at dim 1024, ~128 at 4096) -- these procedures are short
(1-2 body instructions under an ITERATE), well inside it.

Tests: +2 (922 -> 924), test_restore_procedure_pnp_as_program + test_generate_procedure_diffusion_as_program in
test_integration.py.

## Persistent topology: simplex budget + reused distance matrix (deployable speed) (shipped)

Deployment feedback: persistent_topology was too slow for a rolling study (~4s/call on the user's windows, and
~32s on a 250-point cloud here). The cost was twofold: (1) the pairwise distance matrix was REBUILT at every one
of the 7 band scales (and again in _median_nn) -- 8x redundant; (2) on a DENSE cloud with no clean low-dim
topology (a market delay embedding is the canonical case), the VR complex EXPLODES at the wider scales -- ~120k
tetrahedra at the top of the band on a 250-point blob -- and the GF(2) reduction over them dominates.

THE FIX (no accuracy change on the legitimate use case): the distance matrix is computed ONCE and threaded
through (`_median_nn`, `betti_at_scale`, `_build_complex` all take an optional precomputed D), and a SIMPLEX
BUDGET caps the triangle/tetrahedron enumeration (`tri_budget`/`tet_budget`, default 15000). A well-sampled
low-dim manifold's complex is sparse and never approaches the budget (a 250-point sphere peaks at ~6k tris /
~10k tets); a blob hits it at the wide scales. Hitting the budget IS the signal the cloud is not a clean
manifold there: that scale returns B1/B2 as None (unreliable) and is skipped, `histogram['dense_scales']` reports
how many band scales exploded, and if EVERY scale is too dense the result is said plainly as "dense (no clean
topology)" rather than a misleading Betti number.

MEASURED: a 250-point Gaussian blob went 32.5s -> 0.39s (83x), with `dense_scales: 3` flagging that 3 of 7 band
scales were too dense to read. The four manifolds still classify correctly (line/circle ~0s, torus 0.16s,
sphere 0.17s) and the sine-embedding kept-negative still holds (it does NOT read as a clean ring).

HONEST SCOPE (unchanged by this fix -- a speed/honesty fix, not a capability claim): persistent homology still
tracks the shape of a FIXED point cloud, not the dynamics of a 1-D signal -- on a market delay embedding its
B0 counts shadow volatility (point spread), which is why a sine fragments rather than reading as a loop. The
budget makes it FAST and makes it SAY when the cloud is a blob; it does not make it the right tool for 1-D
time-series regime detection. That remains a kept negative.

Tests: +2 (924 -> 926), test_dense_cloud_is_capped_and_flagged + test_budget_does_not_change_clean_manifolds in
test_holographic_topology.py.

## Fast topology promoted to a gate; SpectralBasis scaled with a Chebyshev partial eigensolver (shipped)

Two things off the back of the persistent-homology speedup (83x, prior section). A performance audit profiled the
remaining faculties at REALISTIC sizes (not test sizes -- small inputs hide scale-dependent cost): Wasserstein/
Sinkhorn early-stops and returns small histograms; the de-Doppler bank, Kuramoto sync, creature, and the learning
modules are all bounded or cached. Persistent topology was the lone heavy outlier (a unique combo of redundant
recompute + combinatorial explosion), and it was already fixed. The one latent O(n^3) left was SpectralBasis's
eigh. So: a dividend from the topology fix, and the eigh.

THE GATE (the dividend). Now that naming a cloud's topology is sub-second even on a structureless blob,
persistent homology becomes a first-class GATE. `is_manifold(points)` runs `manifold_topology` and returns
{is_manifold, topology, betti, dense_scales}; is_manifold is True iff the cloud is ONE connected piece (B0 == 1)
and at most `max_dense_scales` band scales were too dense to read. It earns its keep on the spectral denoiser,
whose premise is a smooth field on a CONNECTED manifold -- exactly what the gate checks. `denoise(method=
'spectral', check_manifold=True)` runs the gate first and raises on a non-manifold (with an escape hatch:
check_manifold=False) rather than silently returning graph low-pass. MEASURED: the spectral map cleans a 2-sphere
field 3.74 -> 1.08, but on a random 4-D blob it barely moves (4.37 -> 4.20) -- the gate names which case you are
in for free. Default off keeps the path overhead-free and backward-compatible.

THE EIGENSOLVER (the hard half) -- and the KEPT NEGATIVES of FOUR failed approaches before the one that works.
SpectralBasis built its modes with np.linalg.eigh: ALL n eigenvectors at O(n^3) to keep the lowest ~12. Fine to
~1500 points (0.6s), painful at 3000 (4.4s), worse at 5000 (22s). A partial solver should compute only the
smooth modes. The honest difficulty is DEGENERACY: a 2-sphere's Laplacian carries 2l+1 modes at each eigenvalue
(cumulative block boundaries are perfect squares 1,4,9,16,25), so a COUNT cutoff like n_basis=12 lands INSIDE a
degenerate block (l=3 spans modes 10-16) and the smooth subspace at that cutoff is itself ambiguous. Measured
failures, kept on record:
  * Shifted Lanczos (M = sigma*I - L, top-k Ritz): the wanted modes cluster near sigma at the top of the shift;
    single-vector Lanczos converges to the WRONG subspace (projector diff ~4, denoise 8.0 vs eigh 1.2). 14-28x
    speed but useless.
  * Block subspace iteration on the same shift: the smooth modes compress to a tiny relative spread near sigma
    (convergence ratio ~1), so it never converges (projector diff ~2-3.5, denoise 8.5-15.8).
  * Unshifted Lanczos on L, smallest-k Ritz: works at n=400 (0.92 == 0.92) but DEGRADES with n (n=1500: 2.48 vs
    1.37) -- single-vector Lanczos cannot capture a degenerate subspace, and the degeneracy worsens the larger
    the cloud.
  * Graph-Tikhonov low-pass x = (I + gamma*L)^-1 fn via CG (NO eigendecomposition): robust and fast (0.002-0.17s)
    but a SOFT filter ATTENUATES the signal modes along with the noise (best 3.2 vs eigh 1.2). The field z^2-0.5x
    lives in l<=2; a gentle filter that suppresses l>=3 also suppresses l<=2, so it cannot match a hard k-mode
    cutoff. Fundamentally not a substitute -- a kept negative about soft vs hard spectral filtering.
  * Nystrom (landmark eigh + kNN extension): preserves the hard cutoff on the subsample but the extension error
    GROWS with n (1.67 at 800 -> 3.62 at 3000), i.e. it degrades exactly where eigh is too slow to use. Useless.

THE METHOD THAT WORKS: Chebyshev-filtered subspace iteration (ChebFSI) -- what real sparse eigensolvers use. A
degree-d Chebyshev polynomial of [lambda_cut, lambda_max] stays bounded on that interval and grows fast BELOW
lambda_cut, so it AMPLIFIES the wanted low-eigenvalue subspace by orders of magnitude; a block subspace
iteration then converges THROUGH the degeneracy (the block captures degenerate modes; the filter gives a real
gap). lambda_cut is estimated by a few cheap Lanczos steps (eigenvalue ESTIMATES at the extreme converge even
when the VECTORS do not under degeneracy). The decisive piece is the SPARSE matvec: the kNN Laplacian has ~k
nonzeros per row, so v -> Lv is O(n*k) (verified byte-identical to the dense knn_laplacian, max diff 2e-14),
never forming the n x n matrix. Dense-matvec ChebFSI matched eigh EXACTLY (projector diff 0.00) but was barely
faster (the dense matvec is the bottleneck); the sparse matvec is what turns it into a real speedup.

MEASURED (tuned oversample=14, outer=6, deg=24): projector diff to the exact eigh is 0.000 at n=1800, 0.013 at
3000, 0.310 at 4500 (denoise within ~2% throughout), with speedup GROWING as eigh's O(n^3) bites: 1.7x at 1800,
3.8x at 3000, 7.1x at 4500. SpectralBasis switches to ChebFSI above `partial_threshold` (default 2000) and keeps
the exact dense eigh below -- faster there AND bit-identical, so every existing test/selftest (small clouds)
is unchanged.

KEPT NEGATIVES (travelling in the docstrings and tests): ChebFSI is an APPROXIMATION whose projector error grows
slowly with n; and it lifts the EIGH O(n^3), NOT the O(n^2) kNN distance build, which itself caps practical use
at a few thousand points -- a spatial index would be the next rung. Below threshold the exact eigh is used
(faster and exact). The is_manifold gate inherits manifold_topology's scope: it reads a WELL-SAMPLED manifold,
and a disconnected manifold (B0 > 1) reads as not-a-manifold by design (the gate wants one connected piece).

Tests: +5 (926 -> 931). test_cheb_eigenbasis_matches_full_eigh_at_scale +
test_spectral_basis_thresholds_to_exact_below_cutoff in test_holographic_spectral.py; test_is_manifold_gate_faculty
+ test_spectral_denoise_check_manifold_guard + test_spectral_denoise_scales_to_large_cloud in test_integration.py.

## Backlog triage + D1: the honesty discipline as a structural lint on protocol-vectors (shipped)

A forwarded "VSA-program backlog" proposed porting a session of analysis methods into VSA programs. The
engine's rule is to ground against the LIVE code, not the proposal, and the audit was the usual humbling one:
most of it is already built. A1 (program convention + interpreter) is `holographic_machine.py` (the
stored-program VM) plus the mind's `learn_procedure`/`run_procedure`/`index_procedures`/`canonicalize_procedure`
layer. C1 (the "honesty harness keystone") is already `walk_forward_recall` -- the same six checks the backlog
describes (a shuffled-outcomes null that must collapse to chance, a persistence baseline the signal must beat, a
chance band, scale-correlation, net-of-cost). E3 (massive parallel scan) is `scan` (SPRT-per-channel + honest
per-channel-length FDR). A3 (the exact arbiter) is `RecallNull.pvalue` / `SPRTRecall.decide` / `bh_fdr`, and the
self-hosting audit already drew the boundary. So C1/E3/A1/A3 are shipped; the backlog author rebuilds the harness
by hand because they didn't know to call it `walk_forward_recall` -- the denoising/self-hosting lesson again.

TWO THINGS THE TRIAGE FOUND WRONG, kept on record so they aren't rebuilt:
  * A2 (the "superposed null engine," the backlog's claimed single biggest speed win) is the capacity cliff in
    disguise. The null is ALREADY batched -- `RecallNull.fit` scores all n_null random queries in one matmul,
    `(units @ Q.T).max(axis=0)` -- so there is no sequential loop to parallelize. And running N independent
    computations in one superposed bundle and reading N exact scalars back is exactly the thing the engine has
    measured cannot be done (a 2048-d bundle recalls ~100% of 64 items, ~0% of 2048); the crosstalk would land
    in the null that must stay exact. The backlog even says "verify the superposed run equals N separate runs
    first" -- the verification would just re-derive the cliff.
  * The market half (B, D2, E1/E2) bets against a documented negative: SOL returns came back efficient-market-
    like (shuffle-indistinguishable, survived 10x the data), and the dynamics operator only ties a trivial mean
    predictor. Faster search over absent structure finds the absence faster.

THE ONE GENUINELY-NEW, ON-MISSION IDEA -- and the one thing built (D1). Turn the honesty discipline from a habit
you maintain into a STRUCTURAL PROPERTY you can check. Because a protocol (an analysis procedure) is
program-as-data, its step structure can be READ BACK from its program vector (the VM's own unbind+cleanup), and
anti-patterns become structural queries. `holographic_protocol.py`:
  * `build_protocol(machine, steps)` assembles an ordered list of faculty-step NAMES (encode, combination_search,
    calibrated_null, fdr, oos_split, decide, ...) into one program vector. The names match real mind faculties
    (recall, RecallNull, bh_fdr, walk_forward_recall), so a protocol is a real analysis program, not a toy.
  * `protocol_role_sequence` decodes the program VECTOR position-by-position and maps each APPLY's faculty to a
    ROLE (SEARCH / NULL / FDR / SPLIT / DECIDE / ENCODE / NEUTRAL) via a default, extensible taxonomy.
  * `audit_protocol` evaluates three rules over the recovered structure: (R1) a SEARCH with no procedure-matched
    NULL -- the canonical artifact-factory; (R2) a searched-and-scored FAMILY (SEARCH + DECIDE) with no FDR /
    look-elsewhere control; (R3) selecting then scoring (SEARCH then a later DECIDE) with no out-of-sample SPLIT
    between them. Returns {sound, roles, sequence, violations}.
Wired as the mind faculty `audit_procedure(steps=[...])` (or a prebuilt program+n_steps).

WHY this is the SOUND version of D1 (and where the backlog over-reached): the backlog imagined checking "a whole
space of protocols in ONE operation" -- but reading a per-protocol property out of a SUPERPOSITION of protocols
is the same A2 capacity cliff. So that framing is dropped: the audit reads ONE protocol vector at a time, which
is genuinely holographic (structure recovered by unbind+cleanup) AND correct.

MEASURED (the selftest IS the earns-its-place measurement): the protocol structure round-trips EXACTLY from the
program vector at protocol length (decode reliable for ~6-step protocols at dim 4096); a complete honest protocol
reads sound; each of the three anti-patterns is flagged on a protocol built to contain it; and a no-search
procedure (a restoration loop: datafit -> denoise) is NOT flagged -- targeted, not trigger-happy.

KEPT NEGATIVES (in the docstrings and tests): it is a STRUCTURAL lint on DECLARED steps, not a data-flow analysis
-- the single-accumulator VM does not encode per-step data lineage, so "scores the exact rows it selected on" is
approximated by the ORDER check (no SPLIT between SEARCH and DECIDE), not by tracking data identity. The per-step
decode is bounded by the program vector's capacity, so a protocol must be SHORT to read reliably (the procedure
tax; longer protocols need a larger dim). And an unknown faculty name carries NO obligation (role NEUTRAL), so a
missing taxonomy entry fails OPEN (no false alarm), not closed.

Tests: +8 (931 -> 939). test_holographic_protocol.py (selftest + structure-round-trips + complete-is-sound +
the three anti-pattern flags + no-search-not-flagged) and test_audit_procedure_faculty in test_integration.py.

## D3: the findings registry -- a research log as a holographic knowledge structure (shipped)

The second genuinely-new idea from the forwarded backlog (D3), built after D1 and measured the same way. A
research log that you query by similarity and that detects its OWN contradictions. The substrate was already
present -- the relations layer (`holographic_relations.KnowledgeStore`) encodes role-bound records and runs
explain / name / analogy-as-unbind on them -- so the only thing missing was the operation a research log
actually needs: contradiction detection.

A FINDING is a structured claim: a SUBJECT affects an OBJECT with a POLARITY (+1 helps/strengthens, -1
hurts/backfires), optionally under a CONDITION (a regime: a horizon, a session, an asset). It is encoded the way
every record in the engine is: `finding = bind(SUBJ, subject) + bind(OBJ, object) [+ bind(COND, condition)]`, so
the existing explain/analogy operations compose with it for free. `holographic_knowledge.FindingRegistry`:
  * `add(subject, object, polarity, condition=None, note=None)` -- stores the structured record plus two
    vectors: the FULL finding (subj+obj+cond, for query) and the CLAIM (subj+obj only, for tension pairing).
  * `query(subject=, object=, condition=)` -- recall by similarity to a PARTIAL claim. Bundle the binds for the
    given slots, cosine against findings. ROLE-SENSITIVE: object=momentum recalls findings where momentum is the
    OBJECT, not where it is the subject (the dividend of structured encoding over a bag of words).
  * `tensions(claim_tol=0.85)` -- THE HEADLINE. For every pair whose CLAIMS match (cosine of their subj+obj
    bindings >= tol) and whose POLARITIES are opposite, report a tension, classified FLAT (same/absent condition
    -- genuinely conflicting, one must be wrong) or CONDITIONED (different conditions -- reconcilable, the
    outcome is conditioned on the differing dimension).

The backlog's exact example works: "ER strengthens momentum at 10d" (+1, horizon_10d) vs "ER backfires intraday"
(-1, intraday) is flagged CONDITIONED (reconcilable), while a planted "bracket convex" (+1) vs "bracket drift
masquerades as convexity" (-1), both unconditioned, is flagged FLAT (must resolve).

THE EXACT-DOOR DISCIPLINE (the engine's rule, again): RETRIEVAL is holographic -- the cosine over the bound claim
finds candidate conflicts -- but the VERDICT is EXACT: the polarity sign and the condition equality decide. With
unitary atoms two identical claims give claim-cosine 1.0 and "same subject, different object" gives ~0.5, so
claim_tol=0.85 cleanly requires BOTH subject and object to match (different objects are different claims, not
contradictions). Wired as the lazily-cached mind faculty `finding_registry()`.

MEASURED (the selftest IS the earns-its-place gate): query by subject/object recalls the right findings and is
role-sensitive (a finding with the token in the SUBJECT slot is NOT matched by an OBJECT query); the conditioned
tension and the flat contradiction are each classified correctly; and exactly the two genuine tensions are found
-- no false positives from random similarity, and same-polarity findings about the same claim are NOT flagged
(they agree).

KEPT NEGATIVES (in the docstrings and tests): findings are STRUCTURED claims, NOT free prose -- turning a
2300-line narrative log into structured claims is an NLP step this engine does not do (no embeddings, no parser);
that is the manual / future-LLM boundary, stated plainly. And the tension scan is O(n^2) pairwise claim-cosine --
fine for a few thousand findings; a HoloForest pre-filter is the standard sublinear answer at larger scale,
noted but not needed yet.

Tests: +8 (939 -> 947). test_holographic_knowledge.py (selftest + query-by-subject + role-sensitivity + the
flat-vs-conditioned classification + no-false-positives + same-direction-not-a-tension + signed-polarity) and
test_finding_registry_faculty in test_integration.py.


## D3 made durable: the findings log persists by storing claims, not vectors (shipped)

A research log that evaporates when the session ends is half a tool. D3's FindingRegistry now saves and loads --
but the design follows the determinism rule the whole engine runs on, rather than the obvious route of pickling
the vectors.

THE DESIGN. The saved file holds ONLY the structured claims (subject, object, polarity, condition, note) plus the
dimension and the seed. It does NOT hold a single vector. The vectors -- the bound finding vectors used for recall
and the claim vectors used for tension pairing -- are a deterministic function of the claims and the seed, so on
load they are REBUILT by simply re-adding each finding. This is the demoscene move the engine uses everywhere
(reproduce the structure from the seed instead of storing it): the file is tiny (a JSON list of claims), and the
restored registry is not an approximation of the original but bit-for-bit the same object.

MEASURED (the selftest and tests are the earns-its-place gate): after save -> load, the findings list is identical,
every rebuilt finding vector and claim vector is np.array_equal to the original's (so recall and tension verdicts
are not merely close but exact), and the conditioned-vs-flat tension classification reproduces. A test asserts the
file contains NO vector data (only dim/seed/findings). And a "keeps growing" test reloads a log, adds an
opposite-polarity finding under a different condition, and confirms the new conditioned tension is detected against
the restored findings -- the durable-log use case end to end. Through the mind, m.finding_registry().save(path)
persists the session's log and FindingRegistry.load(path) restores it standalone.

WHY IT MATTERS HERE. This shipped alongside an image-generation push (VSA-diffusion generate_structure rendering
novel valid scenes, and the splat sharpness finding that the right splat scale is content-dependent -- small
splats sharpen edges but hurt smooth fills). Those findings were logged into D3, which surfaced the genuine
conditioned tension (small_splats -> fidelity, +1 at sharp_content vs -1 at smooth_content), and D1 audited the
splat-config TUNING procedure (flagging tune-and-report without a held-out split). Persistence is what lets that
accumulated, reconciled knowledge outlive the conversation it was measured in.

Tests: +4 (947 -> 951). test_holographic_knowledge.py (save/load round-trips findings+tensions+query;
load-rebuilds-vectors-from-seed-not-file with the no-vectors-in-file assertion; loaded-registry-keeps-growing) and
test_finding_registry_persists_across_minds in test_integration.py.


## Holographic vector-graphics (SVG): the sharp, resolution-independent cousin of the splat archive (shipped)

The splat work kept fighting one thing: a Gaussian basis BLURS sharp edges, so a crisp square needs smaller splats
(which then hurt smooth content) or supersampling (which spreads a fixed splat budget too thin). The fix turned out
to be not a better Gaussian but a different primitive. An SVG <rect>/<circle>/<polygon> has analytically EXACT
edges at any zoom -- the rasteriser computes exact coverage from the maths -- so representing/generating images as
vector primitives makes the sharpness problem disappear, and makes the result resolution-INDEPENDENT for free.

This is a structural match the engine already had, the same way a splat scene is a bundle: a vector-graphics scene
is ALSO a bundle of role-bound primitives. holographic_svg.py's HolographicSVG encodes a scene -- a list of
(type, x, y, size, colour) primitives -- into ONE hypervector: each primitive is bundle(bind(TYPE,t),
bind(X, encX(x)), bind(Y, encY(y)), bind(SIZE, encS(s)), bind(COLOUR, c)), and the scene is the bundle of those
primitives each bound to a SLOT. Discrete attributes (type, colour) decode by cleanup; continuous ones (position,
size) by the ScalarEncoder's grid decode -- the continuous analogue of cleanup. SVG emission is pure string
formatting (no new dependency -- NumPy and Flask only).

MEASURED (the selftest and tests are the earns-its-place gate):
- ROUND-TRIP: a 3-primitive scene encodes to one vector and decodes back with type and colour EXACT and
  position/size within ~0.016 on [0,1] (at dim 4096) -- a faithful content-addressable picture.
- MORPH AS ARITHMETIC: interpolating two scenes' HYPERVECTORS and decoding the blend matches a direct parameter
  lerp at the midpoint to within ~0.014. So (1-t)*vA + t*vB really does interpolate the picture -- vector
  arithmetic in the holographic space, rendered crisp through SVG. This is the same morph the splat experiment did
  in pixel space, now exact and resolution-free.
- GENERATE: the composed-manifold diffusion (generate_structure, the B10 sampler) runs over a discrete primitive
  codebook (type x grid-cell x colour) and produces 6/6 distinct novel scenes, valid by construction, rendered
  crisp.
- RESOLUTION INDEPENDENCE (shown in the figure): the same scene stored as a 48px raster and upscaled is visibly
  blocky; rendered from the SVG at 480px it is razor-sharp. No splat blur, no supersampling.

Wired as the svg_canvas() faculty on UnifiedMind (cached, built at the mind's dim/seed), beside the generative
faculties. KEPT NEGATIVE / SCOPE: primitives are isotropic (one size, a palette colour) -- anisotropic
width/height, rotation, gradients/strokes, and bezier paths are the honest next step, the same boundary the
anisotropic-splat work drew. And round-trip fidelity scales with dimension (a few primitives are faithful at
2048+; a crowded scene wants more -- the bundle's finite capacity, shown not hidden).

The through-line: the engine generates compositional structures well, and SVG is simply the SHARP, resolution-
independent renderer that compositional generation deserved -- the sharpness ceiling the splat work measured was a
property of the Gaussian basis, not of the engine, exactly as the matmul and splat-sharpness corrections kept
showing (a quality question is empirical, not a structural wall).

Tests: +8 (951 -> 959). test_holographic_svg.py (selftest + round-trip + morph-midpoint + generate-diverse-and-
deterministic + well-formed-SVG + geometry-scales-with-size + too-many-primitives-rejected) and
test_svg_canvas_faculty in test_integration.py.


## Learned-energy generation: a measured negative (the EP autoencoder denoises but does not generate)

The SVG modality closed with an obvious-looking next lever: swap the hand-built grid codebook the composed-manifold
diffusion uses for a manifold LEARNED from data -- use LearnedEnergyMemory (the EP autoencoder from the June-24
work) as the diffusion denoiser, so generation samples a learned manifold. Probed it; it does not work, and the
measurement says clearly why. NOT SHIPPED.

WHAT WAS MEASURED.
- On the established 2-D bump manifold (2000 samples, the selftest's working config), the learned energy DENOISES
  near-manifold points with ~0.43 relative error -- a RELATIVE win over a matched-byte codebook (the kept positive)
  but modest in absolute terms. From PURE noise a single cleanup lands ~0.51 OFF the manifold; a Langevin walk
  (seed on the manifold, iterate add-noise -> cleanup) does best -- ~0.33 off, 30/36 latent cells covered, novel
  (0.32 from any stored sample) -- but still imprecise.
- On a LOW-DIM SMOOTH manifold (a 9-D scene-parameter family: three circles in a cluster whose centre moves with
  two latents) it FAILS outright. The decisive test: the raw AVERAGE of two nearby training scenes is already
  on-manifold (0.014-0.017) -- the manifold is locally convex, so plain interpolation generates a valid in-between
  scene -- and the learned-energy cleanup makes it WORSE (0.32-1.12) at every bottleneck size (n_hidden in 2,3,4).
  A figure (learned_svg_generation.png) shows it: clean clusters in, a single collapsed/distorted blob out.

WHY (the load-bearing lesson). A good DENOISER is not a good GENERATOR. Denoising only needs a small local
correction from a point already near the manifold; generation needs the energy's MINIMA to lie ON the manifold so
that descending from afar lands on it. A single-noise EP autoencoder's free-state attractors are not reliably on
the manifold -- they collapse toward a smeared mean -- so it denoises (locally) yet distorts when asked to generate.

THE CONSTRUCTIVE OUTCOME.
- For COMBINATIONAL generation the engine's right tool is the composed-manifold diffusion (generate_structure),
  already shipped (9/9 distinct valid SVG scenes).
- For CONTINUOUS in-style variation, simple INTERPOLATION in the composed/parameter space already works (the raw
  midpoint is on-manifold to 0.014) -- which is exactly what the shipped SVG morph does. No learned energy needed
  on locally-convex manifolds.
- A learned GENERATIVE manifold (for genuinely non-convex cases) would need a denoiser trained ACROSS noise levels
  -- a real diffusion model, not this single-noise autoencoder. Filed as backlog VG-2.

The kept negative travels in LearnedEnergyMemory's docstring (a SCOPE paragraph: denoiser, not generator) so a
future caller meets it at the point of use. No test/count change -- nothing shipped; the negative is the result.


## Splat edge artifacts are under-reconstruction (density + greedy fit), fixed by a joint refit -- not a basis floor (measured, corrected)

This entry CORRECTS a too-quick conclusion. The question was splat edge blur, and the first pass blamed two things;
only one held up, and the panel's 3D-graphics seat (and the actual 3DGS literature) pointed at the real cause.

WHAT STILL HOLDS: SUPERSAMPLING A FIXED SPLAT SET IS A NO-OP. Render the same fitted splats direct@N vs
8x->downscale: coarse 14.89 -> 14.88 dB, fine 20.95 -> 20.82, edge PSNR identical, visually indistinguishable. A
sum of Gaussians is band-limited -- there is no high-frequency content to alias, so area-averaging equals point-
sampling. Supersampling fixes ALIASING, which a Gaussian sum does not have. (It DOES become relevant once splats go
SUB-pixel -- then point-sampling aliases them -- which is the last-mile of edge sharpness and where the SVG modality,
with analytic exact-coverage edges, is simply the better tool.)

WHAT WAS WRONG: calling the residual blur/lumpiness a "basis floor" ("a smooth basis cannot represent a hard edge").
At a FINITE output resolution the AA target is itself band-limited, and a sum of Gaussians can approximate it
arbitrarily well with enough density -- so the artifacts were UNDER-RECONSTRUCTION, not a basis limit. Two causes,
both fixable: (1) too few splats (K=140 for a 4096-pixel image is sparse -- the exact under-reconstruction adaptive
density control targets); (2) GREEDY matching pursuit fits each amplitude against the residual at placement time, so
overlapping splats systematically double-count, and it never goes back to fix it.

THE FIX, MEASURED AND SHIPPED: `splat_refit` -- one JOINT least-squares solve over all amplitudes (positions/scales
fixed), the "looping" step. On the real engine fit it adds ~2-4 dB (square+blob 23.64 -> 27.11 at K=400; rings 33.27
-> 35.74), and the gain GROWS with the splat count (more overlap to disentangle). It is closed-form and gradient-FREE,
so it stays inside the NumPy-only rule -- distinct from the gradient optimisation of positions/scales/opacity that
full 3DGS does (that needs autodiff and stays out of scope). Wired as `splat_fit(..., refit=True)` and defaulted ON in
the `splat_field` faculty.

THE 3DGS GROUNDING (web): adaptive density control is the core of 3D Gaussian Splatting -- clone/split Gaussians in
under-reconstructed (high-error) regions, prune elsewhere, iterating throughout optimisation, with the number of
Gaussians set automatically by image complexity. The amplitude refit is the gradient-free half of that loop; true
clone/split densification + position/scale optimisation is the natural next step (backlog), needing the autodiff the
project avoids -- the existing gradient-free `aniso_fit` is where to push it.

THE LESSON (again): a quality artifact is empirical, not a structural wall. "Smooth basis can't do edges" was the same
kind of premature-floor claim the matmul and splat-sharpness corrections already caught; measuring it found a real,
shippable fix instead.

Tests: +4 (959 -> 963). test_holographic_splat.py (joint_refit_beats_greedy_and_gain_grows_with_count,
splat_fit_refit_flag_matches_manual_refit, splat_refit_handles_empty) and test_splat_field_joint_refit in
test_integration.py.

## Low-discrepancy sampling: even coverage where random clumps and a grid aliases (shipped)

The rendering-engine lessons arc opens at its cheapest, broadest-backed item (the panel's pick): a low-discrepancy
sampler. The recurring graphics fact -- random points clump into holes, a regular grid aliases, and the
blue-noise / low-discrepancy middle ground is what production renderers actually sample on (Pharr's PBRT sampling
chapter; Roberts 2018) -- applies anywhere the engine PLACES points to COVER rather than to draw an INDEPENDENT
sample: generation seeds, codebook / anchor placement, the sub-pixel jitter a temporal-accumulation pass will need.

`holographic_lowdiscrepancy.low_discrepancy(n, d, seed)` is Roberts' generalised R-sequence -- the d-dimensional
golden-ratio / plastic-constant additive recurrence, one line of NumPy, no state, deterministic, and PROGRESSIVE
(any prefix is itself well-distributed, so you can keep taking points). Measured: 64 points cover ~28% tighter than
the mean of random (dispersion 0.16 vs 0.23), and as a quasi-Monte-Carlo integrator the same points estimate a
smooth integral with ~13x less error than plain Monte Carlo at equal count (0.0011 vs 0.0143) -- the downstream
payoff of even coverage, not just a prettier scatter. Wired as the `low_discrepancy_sample` faculty (defaults to
the mind's seed). KEPT SCOPE: this is a COVERAGE tool; where genuine independence is wanted (bootstrap, noise
injection) default_rng stays -- these points are correlated by construction. It is the sampler the later
jitter-accumulate (ACCUM-1) and anchor-placement (CACHE-1/3) backlog items will draw on.

Tests: +2 (963 -> 965). test_holographic_lowdiscrepancy.py (the module _selftest: coverage beats random, QMC beats
MC, deterministic + progressive) and test_low_discrepancy_sample_faculty in test_integration.py.

## Throughput-gated traversal: Russian roulette for holographic paths (shipped)

The second rendering-engine-lessons item, and the one with the broadest cross-seat backing on the panel. The
identity behind it -- in the FFT/phasor domain a bind is elementwise complex MULTIPLICATION, so a chain of
binds is a running PRODUCT of per-step transfer functions, exactly a ray's THROUGHPUT -- means a holographic
traversal (a multi-hop recall, the resonator's iterative peeling, a recursive descent) is a ray bouncing
through the space, its recoverable signal attenuating until every further step is noise. Path tracers terminate
such a path with Russian roulette once its throughput is negligible; this ports that move.

`holographic_traverse.gated_traverse(step, start, floor, max_steps, min_steps)` drives a step function --
step(state) -> (next_state, throughput, payload) -- with a cheap running confidence (a cleanup cosine, a
convergence margin) and STOPS the instant it falls below the floor, abstaining on that step (not recording
noise). Measured on a directed linked list stored in superposition: the gate recovers every valid hop and then
abstains the moment the chain is exhausted (signal gone) -- the correct prefix [1..10] in order, stopping where
the past-end unbind is noise (throughput 0.03 vs ~0.3 for valid hops), i.e. 10 steps vs a fixed depth of 30.
Crucially it needs NO ground truth: the cheap confidence tracks the true recoverable cosine closely enough to
know when the ray has gone dark. Wired as the `gated_traverse` faculty (between the recall and resonator
faculties it serves).

KEPT NEGATIVE / SCOPE: the gate keys on LOW confidence (the ray dark); it does NOT catch a CONFIDENT-but-WRONG
step -- the capacity-ambiguity regime where crosstalk returns a wrong atom at moderate confidence -- which is a
calibration problem (the calibrated-null / MIS items), not a throughput one. And this is the deterministic FLOOR
(right for FOLLOWING a path); the unbiased STOCHASTIC Russian roulette (terminate with prob 1-T, boost survivors
by 1/T) is for ACCUMULATING a sum and is a separate, not-yet-measured extension. The directed chain reused the
RAY-3 lesson in passing -- a bundle of bind(x_i, x_{i+1}) is undirected, so the links are permuted (a direction
role) to make traversal unambiguous.

Tests: +2 (965 -> 967). test_holographic_traverse.py (the module _selftest: gating logic on a known profile,
and the real directed-chain traversal) and test_gated_traverse_faculty_recovers_chain_then_abstains in
test_integration.py.

## Adaptive splat count: sample to a noise floor, not a budget (shipped)

The third rendering-engine-lessons item: V-Ray's adaptive sampler (and 3DGS densification in spirit) ported to
splats. A path tracer doesn't spend a fixed number of rays per pixel -- it spends until the variance is below a
noise floor, few where the image is smooth and many where it is busy. The splat fitter previously took a fixed K
regardless of content; this makes the COUNT adaptive.

`holographic_splat.adaptive_fit(target, noise_thresh, k_min, k_max, refit)` runs the same matching pursuit as
splat_fit but stops once the residual RMS falls below noise_thresh * the target's range (bounded [k_min, k_max]),
returning (splats, k_used). Exposed through the existing `splat_field` faculty as a `noise_thresh` argument
(default None keeps the fixed-k path byte-for-byte). Measured at noise_thresh=0.03: a one-blob field fit to 33.2
dB with 13 splats, a seven-blob field to 33.0 dB with 36 -- matched quality, count tracking content -- where a
fixed k=20 over-spends on the easy field (36.5 dB, splats wasted) and starves the busy one (27.0 dB). The count
is orthogonal to the joint refit: the count is WHERE the splats go, the refit is HOW STRONG they are.

KEPT CAVEAT (in the docstring and a test): the threshold gates the GREEDY residual, so quality is only
APPROXIMATELY equalised; and a HARD-EDGED target the smooth isotropic basis cannot represent simply runs to k_max
rather than converging -- the adaptive count is meaningful only for fields the Gaussian basis can actually fit.
The gradient optimisation of positions and anisotropic covariances (full 3DGS) still needs autodiff and stays out
of scope; this item moves only the COUNT.

Tests: +3 (967 -> 970). test_holographic_splat.py (adaptive_fit_count_tracks_content_at_matched_quality,
adaptive_fit_respects_bounds) and test_splat_field_adaptive_count in test_integration.py.

## SHARP-1: a Mitchell-Netravali reconstruction kernel does NOT sharpen the scalar decode (measured negative)

The rendering-lessons backlog's SHARP-1 proposed giving the ScalarEncoder a Mitchell-Netravali reconstruction
kernel: the encoder is a Fourier phase encoder whose similarity kernel is the characteristic function of its phase
distribution (Bochner), already shipping rbf (Gaussian phases -> all-positive, blurs) and sinc (uniform phases ->
the ideal band-limited reconstruction filter, sharp but it rings). Mitchell is the production reconstruction filter
that lives BETWEEN those two -- band-limited negative lobes like sinc, but with the ringing tamed. The hypothesis:
its negative lobes would sharpen the scalar decode, the same negative-lobe sharpening the splat joint-refit already
exploits. Prototyped it thoroughly; it does not yield a measurable win. NOT SHIPPED.

WHAT WAS MEASURED.
- The Mitchell kernel IS realisable in this encoder: its frequency response (the phase distribution it needs) is
  non-negative to 0.2% (clip the tiny negative lobes, sample phases from it). kernel_at then matches the Mitchell
  cubic by Bochner. So the kernel exists and is correct -- that part worked.
- At matched main-lobe width it sits exactly where theory says: peak side-lobe (ringing) 0.022 vs sinc's 0.166 vs
  rbf's 0.008 -- the reconstruction-filter compromise, confirmed.
- But DECODE accuracy is a TIE across all three kernels: single value, value-under-noise (sigma 0.3-1.0), and a
  value bundled with 0-8 distractors all land within noise of each other. The decode is an argmax over a fine grid
  -- a peak DETECTION -- and argmax is insensitive to side-lobe shape, the only thing the kernel choice changes. So
  the negative lobes have nothing to grip.
- A multi-value DENSITY read-out (a SUM, where side-lobes DO shape the output) is messy and does not favour
  Mitchell: sinc's ringing corrupts it badly (1 peak recovered for 3-7 separated values), rbf is adequate, and
  Mitchell's own residual ringing OVER-counts on structured inputs (10 peaks for 5 equally-spaced values). On
  random dense sets Mitchell (3.12 mean peak-count error) only ties rbf (3.63) -- both poor -- so it does not beat
  the simplest existing kernel.

WHY (the load-bearing lesson). Negative-lobe reconstruction-filter sharpening helps where you SUM the kernel and
the OUTPUT IS that sum -- image reconstruction, where the splat joint-refit's ~51%-negative amplitudes measurably
sharpened edges. The scalar decode is the opposite kind of operation: a peak DETECTION (argmax), which reads only
where the main lobe is highest and ignores the side-lobes entirely. The rendering lesson is real but DOMAIN-bound
to reconstruction, not detection -- and the encoder already brackets the genuine sharpness/ringing tradeoff with
its existing rbf and sinc kernels.

THE CONSTRUCTIVE OUTCOME.
- No new kernel is wired: per the project's own rule an option earns its place by measurement, and Mitchell earns
  none here (ties rbf at best, regresses on structured density read-outs). The rbf/sinc pair already spans the axis.
- The place negative-lobe sharpening DID pay is reconstruction -- the shipped splat joint-refit -- so its natural
  generalisation (SHARP-2: a tunable negative-lobe sharpening for the splat/image reconstruction path) is the branch
  of this lesson worth pursuing, not the scalar encoder.

No test/count change -- nothing shipped; the negative is the result.

## Directed structure: a permutation direction role for sequences and graphs (shipped)

The fourth rendering-lessons item -- RAY-3, the one Plate's seat argued to pull early for substrate-correctness.
A memory of edges bundled as bind(x_i, x_{i+1}) is UNDIRECTED: unbinding by a node returns BOTH neighbours,
predecessor and successor, at equal strength (measured ~0.33 vs ~0.33 at the operating dimension), so a traversal
cannot tell forward from backward -- the "predecessor leak". The engine's existing chain_structure (B7) carries
exactly this leak and relies on holographic_peel's per-peel history-aware cleanup to suppress it at decode time.

RAY-3 fixes it at ENCODE time. Binding the successor through a fixed PERMUTATION first -- bind(x_i, perm(x_{i+1}))
-- breaks the symmetry: unbinding by x_i and undoing the permutation recovers the successor (~0.34), while the
predecessor term lands in the permuted subspace as noise (~0.00). The permutation does at encoding time what the
peel cleanup does at decoding time. It also generalises past linear chains: any set of directed EDGES bundles the
same way (a graph), and a branching node returns its whole successor set from one unbind (a 0 -> {1,2,3} node
hands back all three, ~0.40 each, cleanly above the non-successors).

`holographic_directed` ships build()/encode_directed (M = superpose bind(node_i, perm(node_j))), successors()
(perm_inv . unbind . cleanup, with topk / thresh for branching), and make_step() (a gated_traverse-ready
closure). Wired as three faculties: directed_structure (build a sequence or graph), directed_successor (one
forward hop), and directed_traverse (a forward walk gated by recovery confidence -- the RAY-3 substrate under the
RAY-1 throughput gate, so the directed chain and the Russian-roulette walk compose into a clean forward traversal
that stops when the chain runs out).

NOTE on the landscape: the permutation-direction-role mechanism already lived inside sequentiality_z's order test
(bind(a, permute(b,1)) as its transition model), but it was buried in a statistical probe, not exposed as a
general directed encoding; and SequenceMemory is a different representation entirely (POSITIONAL -- each element
rotated by its absolute position, for "what's at position i" / "does A precede B"), not edge/transition based.
RAY-3 is the additive, first-class directed-EDGE structure -- traversable, graph-capable, gated.

Tests: +3 (970 -> 973). test_holographic_directed.py (the module _selftest: the directed-vs-undirected
predecessor-leak contrast, graph branching, gated-walk composition) and
test_directed_structure_forward_only_and_graph + test_directed_traverse_walks_chain_forward in test_integration.py.

## Multiple Importance Sampling: Veach's balance heuristic for combining estimators (shipped)

The fifth rendering-lessons item -- MIS-1, and the first genuinely NEW machinery in the arc rather than a port.
The engine has several estimators of the SAME quantity that each win in a different regime: exact 1-NN (Bayes-
optimal on discrete atoms), the soft dense-Hopfield blend (wins on continuous off-grid values, the B1 kept
negative), the manifold projection (smooth low-rank data), the forest (sublinear/approximate). Until now you PICK
one by hand. Veach's MIS combines them, weighting each by its per-query reliability.

The load-bearing WARNING MIS encodes -- and the thing this module MEASURES -- is that NAIVELY AVERAGING estimators
reliable in different regimes makes things WORSE: the average carries each estimator's error into the other's
regime, landing below the better single. On a coarse sharp-kernel ScalarEncoder manifold with a 50/50 mix of
on-grid + off-grid cues, naive averaging of hard 1-NN and soft Hopfield scores error 0.0061 -- worse than soft
alone at 0.0040. The BALANCE HEURISTIC (w_i = r_i / sum_j r_j, the per-query reliability) lands at 0.0037, beating
both singles AND the naive average.

`holographic_mis` ships combine_estimators(pairs, power) (the Veach balance/power heuristic primitive: pairs of
(estimate, reliability), w_i = r_i^power / sum r_j^power, power=1 balance / 2 power) and mis_recover(q, codebook,
beta, power) (combines hard 1-NN and soft Hopfield per-query, reliability = the cosine distribution's peakiness: a
sharp single winner trusts the exact atom, a close runner-up trusts the interpolating blend). Wired as two
faculties: combine_estimators and mis_recover.

SCOPE / KEPT NEGATIVE: MIS beats EVERY single only in the CROSSOVER regime where neither estimator dominates. When
one dominates the whole regime (e.g., a very sharp kernel where soft wins almost everywhere), MIS MATCHES that
dominant estimator within a few percent rather than beating it -- mixing in the weak one costs a little. The
ALWAYS-TRUE win is over NAIVE AVERAGING; the win over the best single needs a genuine mix -- which is exactly the
MIS property: its value is when no single strategy is uniformly best.

Tests: +2 (973 -> 975). test_holographic_mis.py (the module _selftest: naive-averaging-worse-than-best, MIS beats
naive and both singles, on the mix) and test_mis_recover_beats_naive_average_and_singles in test_integration.py.

## Gradient-cached decode: Ward's irradiance gradients for smooth maps (shipped)

The seventh rendering-lessons item -- CACHE-1, and the second of Group B's "combining & caching" pair (MIS was the
first). The engine evaluates smooth maps (a manifold decode, a splat field), and the naive dense read is a fine
grid + nearest-neighbour snap (the decode argmax, piecewise constant). Greg Ward's irradiance caching does better:
store the value AND its local gradient (Jacobian) at SPARSE anchors and interpolate FIRST-ORDER -- each anchor
extrapolates its own linear model v_i + J_i.(q - a_i) to the query, blended by 1/distance.

MEASURED on a smooth splat / Gaussian-mixture field (analytic gradients): first-order gradient interp cuts error
~28% at a fixed 25 anchors vs the nearest-neighbour baseline (0.135 vs 0.189), and first-order @25 anchors roughly
MATCHES nearest-neighbour @49 -- gradients ~HALVE the anchor count a smooth decode needs.

KEPT NEGATIVE (the load-bearing part): the blend MUST be local. A naive GLOBAL weighting (every anchor contributes,
weight ~1/distance, no cutoff) lets a distant anchor dump a wildly wrong long-range linear extrapolation into the
query -- measured ~2.7x WORSE than the local version (0.363 vs 0.135). This rediscovers exactly why Ward's
irradiance caching carries a validity radius + neighbour clamping. So the borrowable unit is the whole PACKAGE:
sparse anchors + stored gradients + a validity-radius locality guard.

`holographic_cache` ships gradient_cache(anchors, values, jacobians) (scalar OR vector fields), gradient_cache_fd
(build from a field function alone via central finite differences), and interp_first_order(cache, q,
validity_radius, global_weights=False) (Ward first-order interp with the validity-radius guard; global_weights=True
exposes the negative). Wired as two faculties: gradient_cache and cache_interp.

Tests: +2 (975 -> 977). test_holographic_cache.py (the module _selftest: gradients beat nearest-neighbour at fixed
anchors, ~halve the count, and global weights fail) and
test_gradient_cache_first_order_beats_nearest_and_global_weights_fail in test_integration.py.

## Robust accumulation: harmonic weights + firefly clamping (shipped)

The eighth rendering-lessons item -- ACCUM-2 and ACCUM-3, two cheap robustness fixes for the engine's averaging
paths (consolidation over a growing store, HoloForest vote-averaging, any iterate-and-average).

ACCUM-2 (harmonic weights, TAA's lesson). A fixed-alpha exponential blend x <- (1-a)x + a*sample never fully
converges -- it keeps forgetting old samples, so its variance plateaus. The harmonic (1/n) running average
x <- x + (sample - x)/n weights every sample equally and converges. MEASURED on a stationary noisy stream: harmonic
error falls with N (0.0073 @ N=50 -> 0.0012 @ N=200 -> 0.0004 @ N=800), while the fixed-alpha EMA flatlines at
~0.034. KEPT CAVEAT: on a DRIFTING target the forgetful EMA tracks BETTER (0.031 vs harmonic 0.043) -- so
schedule='ema' stays available for non-stationary accumulation.

ACCUM-3 (firefly clamping, V-Ray's adaptivity clamp / TAA history rectification). One outlier estimate (a firefly
recall/vote with a huge magnitude) skews a mean. Clamping each sample's deviation from the MEDIAN to k robust-scales
(the median deviation) winsorizes the outliers. MEASURED: with 5 injected fireflies, plain mean error 0.0467 vs
clamped 0.0004 (~100x more robust); on clean data, clamped == plain (no loss).

`holographic_accumulate` ships robust_accumulate(samples, schedule, alpha, clamp_k) (schedule 'harmonic'/'ema'/'mean'
+ optional firefly clamp_k -- the two compose), plus harmonic_accumulate and clamped_accumulate conveniences. Wired
as one faculty: robust_accumulate. NOT forced into consolidation/forest internals (that would risk a regression);
shipped as the available robust accumulator for those paths, demonstrated on the canonical stationary-stream and
firefly cases.

Tests: +2 (977 -> 979). test_holographic_accumulate.py (the module _selftest: harmonic converges + EMA plateaus +
drift caveat + firefly clamp robust/no-loss) and
test_robust_accumulate_harmonic_converges_and_clamp_resists_fireflies in test_integration.py.

## Denoise-by-downscale: find a pattern by coarsening until noise averages out (shipped)

XDATA-1, the entry point of Group G (the cross-data-type through-line) and the first rendering lesson that is
explicitly NOT about images. The lesson: "patterns can be found by downscaling to eliminate noise." Downsampling an
image pools neighbouring pixels so independent noise averages out while structure survives -- a MANIFOLD operation,
not an image one. The engine already owns its forms: consolidation (low-rank SVD) is downscaling for CORRELATED
VECTORS (pool across samples; the shared subspace reinforces, per-coordinate noise cancels), and low-pass filtering
is downscaling for SIGNALS.

MEASURED on two non-image data types:
- LOW-RANK: a rank-3 subspace INVISIBLE in any single noisy vector (per-sample subspace energy ~0.03) is recovered
  by pooling many samples -- subspace overlap grows with the sample count (0.22 @ N=100 -> 0.91 @ N=2000), the
  averaging mechanism. (Requires the signal above the SVD/BBP recovery threshold; below it, fails safe -- nothing.)
- LOW-FREQUENCY SIGNAL: slow sinusoids buried under 2x noise (full-res corr 0.47) recovered by keeping the top-k
  spectral components -- corr 0.90 to the clean signal.

THE HONEST PART (fail-safe detection): keeping the top-k components ALWAYS concentrates a little, even on pure noise
(you select the largest of many random components -- the FFT noise concentration was 0.10 vs a uniform 0.023). So
"a pattern was found" is NOT read off the concentration; it is decided against a PERMUTATION NULL (shuffle to
destroy the structure, keep the noise level, recompute the score). Signal scores land ~60 sigma (low-rank) / ~14
sigma (signal) above the null; pure noise lands AT the null -> found=False. The faculty does not hallucinate a
pattern in noise.

`holographic_downscale` ships downscale_lowrank (SVD subspace), downscale_lowfreq (top-k FFT), and
find_pattern_by_downscale(data, kind='vectors'/'signal', k, n_null, seed) -> PatternResult(pattern, score,
null_mean, null_std, found). Wired as one faculty: find_pattern_by_downscale.

Tests: +2 (979 -> 981). test_holographic_downscale.py (the module _selftest: recover buried subspace + buried
sinusoids, both found; pure noise of either type -> nothing) and
test_find_pattern_by_downscale_recovers_buried_pattern_and_noise_fails_safe in test_integration.py.

## Looping denoise as diffusion on an arbitrary manifold (shipped)

XDATA-2, the diffusion half of Group G. "A looping denoising process": iterate a denoiser and it walks onto the
manifold (DENOISING) or, from pure noise, walks ONTO it (GENERATING) -- the same operation in two regimes (B10).
The engine already ran this over the discrete CODEBOOK (hopfield.generate); XDATA-2 generalizes it to a LEARNED or
COMPOSED manifold given as a point cloud (a curved manifold, or a consolidation subspace from
find_pattern_by_downscale -- the two halves of Group G compose).

The denoiser is a dense-Hopfield step over the manifold's samples: x <- softmax(beta * S.x) @ S (a soft move toward
the local samples). Iterating settles a point onto the manifold; annealing beta UP while injecting DECREASING noise
turns it into a diffusion sampler.

MEASURED on a curved manifold (a unit RING in R^D -- the case where interpolation provably leaves the manifold):
- IDEMPOTENT DENOISE: a noisy ring point settles from ring-distance 0.59 to 0.029 and stays there (further steps
  flat). The 0.029 floor is the sample-spacing limit (N=48 discrete samples), not error.
- BEATS INTERPOLATION: the chord midpoint of two ring samples is off the ring (0.74); the denoiser settles it back
  on (0.029). Looping-denoise beats interpolation for staying on a curved manifold.
- NOVEL-BUT-VALID GENERATION: from-noise annealed diffusion lands on the ring (dist ~0.02, valid) BETWEEN the stored
  samples (dist-to-nearest-stored ~0.04, novel) -- where bare-codebook generation just returns a stored sample
  (dist-to-stored 0, degenerate).

`holographic_diffuse` ships manifold_denoise_step (one dense-Hopfield step), settle (iterate -- denoise), and
generate (annealed diffusion -- from-noise generation). Wired as two faculties: manifold_denoise and
manifold_generate.

Tests: +2 (981 -> 983). test_holographic_diffuse.py (the module _selftest: idempotent settling, interpolation
beaten, novel-but-valid generation, codebook degeneracy) and
test_manifold_denoise_settles_and_generate_is_novel_but_valid in test_integration.py.

## Looping negative-lobe sharpening for arbitrary signals (shipped)

XDATA-3, the SHARPEN half of Group G and the partner to SHARP-2 -- closing the denoise/generate/sharpen trio. A
smooth basis (low-rank reconstruction, Gaussian splat, over-consolidated truncation) LOW-PASSES a signal,
attenuating its high-frequency detail. Sharpening counteracts that by repeatedly adding a high-pass (negative-lobe)
correction -- the mechanism the splat joint-refit used (its ~51%-negative amplitudes sharpened edges), now
data-type-agnostic.

THE HONEST SUBTLETY: the naive loop (iterated unsharp x <- x + a(x - blur(x))) DIVERGES -- its high-freq gain
(1+a)^k is unbounded, recovering detail for a few steps then exploding (measured: error 0.22 -> 0.069 at iter 6,
then -> 38 by iter 10). The stable loop is VAN CITTERT (residual-fitting deconvolution, x <- x + lam(y - blur(x))):
its accumulated operator converges to the INVERSE blur (a negative-lobe sharpening filter) with bounded eigenvalues,
so it CONVERGES.

MEASURED on a 1-D signal (slow component + a localized high-frequency burst, Gaussian-blurred sigma=3):
- NO NOISE: looping sharpening recovers the burst and converges -- relative error 0.222 -> 0.001, no blow-up.
- WITH NOISE (kept negative): Van Cittert recovers up to an OPTIMUM then amplifies high-freq NOISE (over-sharpening).
  The principled stop is Morozov's DISCREPANCY PRINCIPLE (halt when residual ||y - blur(x)|| <= noise norm): lands
  near the optimum (err ~0.12 vs blurred 0.22); running UNGUARDED over-sharpens to ~0.45.
- lam above the stability bound (~2/||blur||^2) DIVERGES into ringing (err -> 1300+) -- why lam is bounded.

`holographic_sharpen` ships _gauss_blur (default FFT low-pass) and sharpen_loop(x, blur, sigma, lam, iters,
noise_level) (Van Cittert with the discrepancy-principle guard; blur is the smoothing operator, callable). Wired as
one faculty: sharpen_loop.

Tests: +2 (983 -> 985). test_holographic_sharpen.py (the module _selftest: no-noise recovery+convergence, noise
guard-beats-unguarded, over-large-step divergence) and
test_sharpen_loop_recovers_detail_converges_and_guard_beats_oversharpening in test_integration.py.

This closes Group G -- the cross-data-type through-line: denoise-by-downscale (XDATA-1), looping diffusion denoise +
generate (XDATA-2), and looping sharpen (XDATA-3) are all ONE manifold operation, applicable to any data type the
engine holds, each with its honest negative and fail-safe/stability guard.

## Smooth/sharp two-layer representation (shipped)

CACHE-2, the architectural move borrowed from irradiance caching (cache the smooth indirect light, compute the
sharp direct light). The principle: NO SINGLE basis is cheap across a signal that is smooth in places and sharp in
others. The same split recurs in the negative-lobe sharpening finding, the SVG (smooth morph + exact vector edges),
and manifold-plus-residual decompose. At a fixed budget, split:
  smooth layer = the k_smooth lowest-frequency coefficients (cheap dense basis), and
  sharp layer  = the k_sharp largest residual coefficients, in a basis where the sharp content is sparse.

The earlier attempt was only a MODEST win (15.7 vs 13.7 dB) because its sharp basis was weak (pixel-exact). The win
here is LARGE because the sharp basis is the RIGHT one for the sharp content: localized spikes are BROADBAND in
frequency but SPARSE in the SAMPLE domain, so a sparse sample-domain residual holds them in a handful of
coefficients (a low-frequency basis would need a great many).

MEASURED on a signal = two slow sinusoids + 6 spikes, at a budget covering both layers (k_smooth=6, k_sharp=6):
- SPLIT 40.4 dB vs single-FFT 28.0 vs single-sparse 18.3 -- the split wins by a wide margin.
- 30% of the signal energy sits in the residual the low-frequency layer provably cannot hold (the spikes).
- KEPT CAVEAT: at too SMALL a budget (k_smooth=4, k_sharp=4) the split LOSES (23.5 vs single-FFT 27.5) -- it cannot
  afford enough of either layer; the win needs a budget large enough to hold both layers' essential coefficients.

ANSWER to the backlog's open research question "what is the right sharp basis in the hypervector domain": whichever
one the sharp content is sparse in -- sample-sparse for spikes, a wavelet basis for edges. CACHE-2 is the STORAGE
counterpart to XDATA-3's RECOVERY: CACHE-2 stores the detail explicitly (the sharp layer); XDATA-3 recovers it from
an over-smoothed estimate. Complementary store-vs-recover.

`holographic_twolayer` ships TwoLayerCode, smooth_sharp_split(x, k_smooth, k_sharp), smooth_sharp_reconstruct(code),
and the single-basis baselines _fft_topk / _sparse_topk. Wired as two faculties: smooth_sharp_split +
smooth_sharp_reconstruct.

Tests: +2 (985 -> 987). test_holographic_twolayer.py (the module _selftest: split beats both single bases at
sufficient budget, sharp positions exact, small-budget caveat) and
test_smooth_sharp_split_beats_single_basis_at_fixed_budget in test_integration.py.

## FHRR phase-domain morph (shipped)

PHASE-1, borrowing phase-based frame interpolation's move into the PHASE domain (phase shift = motion), not
amplitude blending. Under large motion, amplitude blending GHOSTS (two faint copies fading through each other); a
phase shift MOVES the feature. FHRR is already the engine's phase domain (every atom = a vector of complex unit
phasors), so the engine gets phase-domain interpolation for free: shift each component's phase along the shortest
arc, staying on the unit-phasor manifold, instead of blending the complex vectors (the amplitude-domain morph).

MEASURED on an FHRR fractional-power position encoding (a feature moving a large distance, xA=0.1 -> xB=0.9):
- UNIFORM MOTION (the win): the phase morph moves the decoded feature at CONSTANT velocity -- tracks the ideal
  trajectory exactly (max deviation 0.000). The amplitude blend STALLS near each endpoint and rushes through the
  middle (an eased S-curve, deviation 0.057), because the phase of a weighted complex sum is biased toward the
  heavier endpoint.
- ENERGY / VALIDITY: the phase morph is a valid unit phasor at every t (|z_j|=1). The amplitude blend COLLAPSES
  where components fall out of phase -- mean magnitude 0.75 at the midpoint (toward 0.64 for independent states),
  so it is not even a valid FHRR vector without renormalising.

THE HONEST NEGATIVE (kept loud): phase-domain morphing is NOT a free win under arbitrarily large change. The morph
uses the SHORTEST ARC per component, which WRAPS once a component's phase difference exceeds pi -- past that it
takes the wrong way round and stops tracking the true intermediate (measured: at a separation where phase diffs
reach ~1.6*pi, deviation 0.983 -- completely lost). And near-orthogonal endpoints have no well-defined intermediate
for ANY method. So the win holds while the change keeps per-component phase differences under pi; beyond that it
degrades gracefully on energy (still unit phasors) but not on tracking.

`holographic_phasemorph` ships phase_morph(a,b,t) (shortest-arc phase interpolation) and amplitude_morph(a,b,t) (the
baseline blend). Wired as one faculty: phase_morph. This connects to the WiFi/CSI phase-as-information thread on
record: phase IS the information, and interpolating it directly is what FHRR's phasor domain makes natural.

Tests: +2 (987 -> 989). test_holographic_phasemorph.py (the module _selftest: uniform-motion win, energy
preservation, wrapping negative) and test_phase_morph_uniform_motion_and_energy_with_wrapping_negative in
test_integration.py.

## Adaptive iteration count for the resonator (shipped)

ADAPT-2, the variance-gate applied to iteration COUNT rather than sample count. The SBC resonator
(holographic_sbc.sbc_resonator / decompose_structure) factors a bound product by annealed alternating projection. It
already returned early per RESTART once the picks verified, but its INNER loop always ran a fixed 50 iters even after
the estimate had converged. Adaptive sampling is the engine's own pattern (the SPRT in the recall path), and the
resonator has an even cleaner stop signal: an EXACT reconstruction.

The opt-in `early_stop=True` (default off, bit-identical when off) stops the moment the picks RECONSTRUCT the product
exactly. Because no further iteration can improve a verified answer, this returns the SAME verified answer the fixed
count would, only sooner -- so it is RISK-FREE (accuracy never changes, iters never increase).

MEASURED (B=24, L=7, F=3):
- EASILY-SOLVABLE workload (codebook N=10): early-stop cut average iters ~62% (68 -> 26) at IDENTICAL accuracy
  (19/20 == 19/20). A single solvable problem went 50 -> 6 iters, same verified picks.
- HARD / mostly-unsolved workload (N=50): 0% change, 0 harm -- unsolved problems never verify, so they run the full
  search either way. The win is workload-dependent: large where the fixed count over-computes an easy problem, a
  clean no-op where the search genuinely needs the iterations.

Wired additively: `early_stop=False, min_iters=5, stats=None` on sbc_resonator and decompose_structure (module), and
`early_stop=False, stats=None` on the UnifiedMind decompose_structure faculty. Pass stats={} to read stats['iters']
(the inner iterations actually run) so the saving is measurable. Existing SBC suite passes unchanged (back-compat).

Tests: +2 (989 -> 991). test_holographic_adaptive_resonator.py (the module _adapt2_selftest: matched accuracy at
lower avg iters on solvable, no-op on hard) and test_decompose_structure_early_stop_matches_at_lower_cost in
test_integration.py.

## Adaptive curvature-driven cache anchor placement (shipped)

CACHE-3, irradiance caching's adaptive record density instead of a uniform grid. Uniform placement wastes anchors on
flat regions and under-resolves the bends; the GI literature reports ~7x fewer records for the same quality with
adaptive density. The same waste applies to any cache or codebook over a field with non-uniform smoothness.

THE RULE (equidistribution). For piecewise-linear reconstruction the error on an interval of width h scales like
|f''|*h^2, so to make every interval contribute equally: |f''|*h^2 = const -> h ~ |f''|^(-1/2) -> anchor DENSITY ~
|f''|^(1/2). Estimate the curvature, raise to the 1/2 power, add a small floor so flat regions still get a few
anchors, place anchors at equal-mass quantiles of that density (inverse-CDF sample).

MEASURED on a gentle slope + one sharp narrow bump:
- adaptive placement matches uniform quality at ~7.5x FEWER anchors (uniform needs 239 to match adaptive-32), and at
  a fixed count is far better (N=32: uniform RMSE 0.070 vs adaptive 0.0017) -- the bump resolved, not stepped over.
- HONEST CONTROL (kept scope): on a UNIFORMLY-smooth field (a plain sinusoid) adaptive does NOT beat uniform
  (uniform-32 0.0106 vs adaptive-32 0.0084, ~tied) -- no curvature concentration to exploit. The win is quality
  MOVED to where the field needs it, not free quality; it is specifically a property of NON-uniform smoothness.

Ties to ADAPT-1 (residual-peak splat placement, gradient-ish) and CACHE-1 (the irradiance cache whose anchors this
places), and to the Group H AO-1 local-crowding hypothesis. `holographic_adaptive_cache` ships adaptive_anchors(x,
y, n, floor, power) and reconstruct_from_anchors(x, anchor_x, y). Wired as two faculties.

Tests: +2 (991 -> 993). test_holographic_adaptive_cache.py (the module _selftest: adaptive beats uniform at fixed N
and at ~7x fewer anchors, ~tied on a smooth field) and test_adaptive_anchors_beat_uniform_on_nonuniform_field in
test_integration.py.

## Backward warping is hole-free (shipped / validated)

PHASE-2, a validated note (not a new faculty -- unbind already is the backward map). Frame interpolation moved from
FORWARD warping (push each source pixel to where it goes) to BACKWARD warping (for each target pixel, pull from where
it came) because a forward warp under a non-uniform deformation leaves HOLES (target cells no source landed on) and
OVERLAPS (cells several sources collide on), while a backward warp visits every target exactly once and fills them
all by construction. The engine gets the backward form for free: unbind is a BACKWARD, invertible map -- to recover a
stored value you take the target role and unbind its source out (a gather), not scatter the composite forward and
hope every slot fills.

MEASURED (a signal resampled under a non-uniform but monotonic warp warp(s) = s + 0.12*sin(2pi*s), N=256):
- FORWARD scatter: 62 holes + 39 overlaps out of 256 cells (the warp locally stretches -> gaps; locally compresses
  -> collisions).
- BACKWARD gather: 0 holes, reconstruction RMSE 0.0 -- every target read its source exactly.

The note for the engine: wherever it could either splat a representation forward or unbind it backward, the backward
route is the hole-free one to prefer. `holographic_backwardwarp` ships forward_scatter (the cautionary baseline) and
backward_gather (the unbind form) as the demonstration behind the note; no UnifiedMind faculty (unbind already
exists).

Tests: +1 (993 -> 994). test_holographic_backwardwarp.py (the module _selftest: forward leaves holes+overlaps,
backward leaves none and is exact).

## Multi-resolution pyramid / mipmap (shipped)

SCALE-1, making coarse-to-fine an explicit ARCHIVE (mipmaps / flow pyramids / 3DGS densification). Keep the signal at
several resolutions, read the level a query needs, refine toward fine only where it matters. The engine already leans
this way implicitly (recursive/fractal structure, HoloForest's coarse descent, consolidation's low-rank-first); this
makes the multi-resolution archive explicit.

THE DECISIVE PROPERTY: anti-aliasing on a COARSE read. You cannot get a low-resolution view by SUBSAMPLING the full
store -- content above the coarse Nyquist FOLDS into the low band and corrupts it (aliasing). A mipmap level was
LOW-PASS FILTERED before downsampling, so its coarse view is clean. Each level is also smaller (cheap coarse read),
and the levels are a progressive code (coarsest is a usable approximation, finer levels add detail back, exact at top).

MEASURED (a low-freq signal + a high freq ABOVE the coarse Nyquist; pyramid [1024, 512, 256, 128, 64]):
- a 1/8 coarse query matches the true low-frequency band ~11x better than a naive subsample (mipmap RMSE 0.035 vs
  naive 0.388), which aliases the high frequency into a spurious low tone (the aliased bin has >100x the spurious
  energy under naive vs mipmap).
- each level is half the size of the one below (cheap LOD read); the full level reconstructs exactly.

RELATION TO CACHE-2 (kept honest): CACHE-2's smooth/sharp split is a fixed TWO-level decomposition tuned to a storage
budget; SCALE-1 is the multi-LEVEL spatial hierarchy with the distinct anti-aliased-LOD property (each coarse level
is a smaller, alias-free array readable on its own). Same family, different job. Relates to XDATA-1 (downscale =
low-pass = the same anti-aliasing, here stacked into a pyramid). `holographic_multires` ships build_pyramid,
upsample_to, naive_subsample (the baseline). Wired as two faculties: multires_pyramid + pyramid_reconstruct.

Tests: +2 (994 -> 996). test_holographic_multires.py (the module _selftest: anti-aliased coarse query beats naive
subsample, levels halve, full level exact) and test_multires_pyramid_anti_aliased_coarse_query in test_integration.py.

## Re-anchoring is load-bearing for deep traversal (shipped / audited)

RAY-2, the path-traced form of "a shared kernel is not a shared manifold." In the FFT/phasor domain a bind is
elementwise complex multiplication, so a chain of binds is a ray whose recoverable signal ATTENUATES multiplicatively
with each hop. The fix is next-event estimation: connect to a KNOWN anchor (the codebook) at every bounce via cleanup
-- re-project the intermediate state onto the manifold each step. Without it the state drifts off-manifold and the
signal collapses.

THE AUDIT (the VALIDATE half): every deep-composition / traversal faculty already re-anchors at each step --
gated_traverse (RAY-1) and directed_traverse (RAY-3) clean up inside their step; the peel-based decode_structure
cleans up per peel (measured 2 -> 15 hops); the pack/recover and nested-decode paths resolve each item to the
codebook. No deep path is missing the discipline, so there is no cleanup to add -- RAY-2 is a validation, not a build.

THE CONTRAST (what the existing tests omit): the traverse self-test shows the RE-ANCHORED traversal works, but never
shows it FAILING without re-anchoring -- the whole claim. This drives the engine's REAL gated_traverse over a directed
linked list two ways, identical except for the one line that carries the CLEANED node forward vs the RAW one.

MEASURED (a 12-hop directed linked list in superposition):
- RE-ANCHORED reaches every hop (12/12) in order, then the throughput gate abstains exactly when the chain runs out
  (the signal is genuinely gone, not lost to drift).
- RAW collapses almost immediately (~1 hop): the carried noise compounds each hop, throughput falls through the
  floor, and the gate stops the dark ray. Per-hop cost: one codebook argmax (O(vocab)) -- cheap, and plainly
  justified, since without it the traversal does not survive past the first hop.

Complements peel's BUNDLE result (iterated decode 2 -> 15) with the CHAIN case, on the real faculty.
`holographic_reanchor` ships directed_linked_list (build the superposed chain) and make_steps (the re-anchored vs raw
step functions); no new faculty (gated_traverse already is the faculty -- this audits it).

Tests: +2 (996 -> 998). test_holographic_reanchor.py (the module _selftest: re-anchored reaches all hops, raw
collapses early) and test_reanchoring_is_load_bearing_for_deep_traversal in test_integration.py.

## Jittered sub-pixel splat accumulation -- KEPT NEGATIVE (ACCUM-1)

ACCUM-1, the "TAA/DLSS done correctly" idea. The splat fit places every splat at an INTEGER grid position (residual
peak) and the joint refit keeps positions fixed, so the natural idea: jitter the FIT at sub-pixel offsets across
passes (Halton/golden-ratio) and accumulate, letting splats land between grid points to sharpen sub-pixel edges. The
honest question: does it sharpen PAST the joint refit? MEASURED ANSWER: NO.

MEASURED (a continuous target with a sharp SUB-PIXEL feature, K splats, scored at high resolution):
- REFIT-ONLY (base grid): RMSE 0.0201 -- grid-aligned splats can't sit on the sub-pixel feature.
- JITTERED accumulation (fit K/j on j sub-pixel-shifted grids, accumulate, joint-refit): RMSE 0.0022 -- better than
  base, BUT only because it SAMPLES the target at sub-pixel offsets (supersampling), not because of jittering.
- THE CONTROL THAT SETTLES IT: given the SAME sub-pixel samples, fitting DIRECTLY on a 4x-finer grid (an ordinary
  refit at higher resolution) is RMSE 0.0011 -- STRICTLY BETTER than the jittered accumulation. A global greedy +
  joint refit over all sub-pixel positions beats fitting each shifted grid independently and summing.
- AND with NO new info (shifted grids interpolated from the base grid), jittering can't manufacture sub-pixel detail
  the base samples never held.

THE NEGATIVE: jittered sub-pixel accumulation is NOT a sharpening tool. The only lever is the SAMPLING RESOLUTION of
the target -- if you have sub-pixel samples, fit directly on them (a finer-grid refit wins); if you don't, jittering
adds nothing. Pixel-aligned placement + joint refit, at sufficient sampling resolution, is already the right answer.
Consistent with the earlier no-op (supersampling a band-limited Gaussian sum has nothing to anti-alias). Nothing is
wired -- `holographic_jittersplat` records the experiment and the negative.

Tests: +1 (998 -> 999). test_holographic_jittersplat.py (the module _selftest: jittered beats base only by
supersampling; a finer-grid refit beats jittered -- jittering doesn't sharpen past the refit).

## Anisotropic splat-fit adaptive stop -- C3 (cross-cutting: ADAPT-2 -> image gen)

The first cross-cutting transfer: the resonator's adaptive-stop (ADAPT-2) applied to splat_aniso's gradient
fit, which optimises the covariances for a FIXED 200 Adam steps. Stop when the reconstruction MSE has converged.

THE CRITERION (and why the obvious one fails twice):
- Relative-improvement-over-a-window vs the CURRENT MSE FAILS on a near-perfectly-fittable field: the fit
  descends geometrically toward zero, so each window still halves the error (> tol relative) forever and the
  stop never fires. FIX: measure window improvement against the INITIAL error (a fixed scale) -- it fires
  whether the fit plateaus at a residual floor OR descends geometrically toward zero.
- Adam's momentum needs ~30 steps to warm up; during the warm-up the MSE barely moves, so a naive test
  mistakes it for convergence and stops at step ~20 with a terrible fit. FIX: a min_steps floor (default 40).

MEASURED: ~20-40% fewer steps on under-fit fields (a busy 9-blob field stops near ~121 of 200), less on a
near-perfectly fittable one, at a few-percent MSE cost.

THE KEPT CAVEAT (the honest difference from ADAPT-2): the resonator early-stop is FREE because it has an EXACT
reconstruction certificate (stop when the picks verify -- same answer, sooner). A continuous gradient fit has
only a SOFT plateau, so stopping ALWAYS costs a little MSE -- this is a speed/quality KNOB, not a free lunch.
Off by default (early_stop=False is bit-identical to the fixed-step fit).

Wired as early_stop= on aniso_fit and the splat_aniso faculty (pass stats={} to read stats['steps']).

Tests: +2 (999 -> 1001). test_holographic_aniso_earlystop.py (the module _c3_selftest) and
test_aniso_early_stop_saves_steps_at_small_cost in test_integration.py.

## Adaptive-stop diffusion -- B3 (cross-cutting: ADAPT-2 -> text gen)

The resonator's adaptive-stop applied to generate_structure (the B10 composed-manifold diffusion), which runs
a FIXED annealing schedule. The structure being built -- read as the hard combination of fillers per slot
(_decode_combo: unbind each role, argmax the filler) -- SETTLES well before the schedule ends, so stop once it
has been stable for `patience` steps past a `min_steps` floor (default steps//2, past the high-noise phase).

ENO'S CONDITION (don't amputate novelty): stop on STABILITY, not first-convergence. The late, lower-noise part
of the walk is where different seeds diverge into different structures; cutting it off on the first converged
step would collapse diversity. Stability-for-`patience`-steps past a floor preserves it -- MEASURED: 20 distinct
structures both ways, and the SAME structure as the full run on every seed (so the stop changes WHEN it lands,
not WHERE).

WHY IT IS FREE (unlike C3): a continuous splat fit has only a soft plateau, so stopping always costs a little
MSE. Here the hard decoded combination is an effective CERTIFICATE -- once it is stable, the output is
determined. The early-stopped z is mid-anneal (slightly less sharp: validity ~0.967), so on stopping we apply
one final crisp `_structure_project` at full beta with NO noise, which sharpens the settled combination and
restores validity to 1.000. Same structure, full validity, ~50% fewer steps.

Wired as early_stop=/min_steps= on generate_structure (module + faculty; pass stats={} to read stats['steps']).
Off by default (early_stop=False is bit-identical to the fixed schedule).

Tests: +2 (1001 -> 1003). test_holographic_diffusion_earlystop.py (the module _b3_selftest) and
test_generate_structure_early_stop_matches_full_at_half_the_steps in test_integration.py.

## Splat-render sharpening -- C4 (cross-cutting: XDATA-3 -> image gen) -- KEPT NEGATIVE

THE PROPOSAL (Milanfar's seat, RED/Van Cittert): a splat render is a sum of smooth Gaussians, hence
over-smoothed (splat_aniso's own negative says a few Gaussians cannot hold high frequency), so sharpen it with
the XDATA-3 negative-lobe loop to recover edge detail. High upside on paper.

THE MEASURED ANSWER: it does NOT work, for a STRUCTURAL reason (not tuning). Van Cittert deconvolution assumes
the smooth signal is blur(truth) -- a CONVOLUTION of what you want. A splat render is not that: it is a sparse
sum of Gaussians, ~= blur(the splat CENTRES), a handful of spikes. Deconvolving it drives toward those centres
(spikes/ringing), NOT toward the discarded edges. Sharpening the render at every sigma/iters tested makes it
WORSE (relative error rises ~5-8%).

THE DECISIVE CONTROL: the SAME 2-D Van Cittert sharpener on a GENUINE Gaussian blur of the truth RECOVERS ~42%
of the error (0.37 -> 0.22). The machinery works; the negative is specifically that a splat render is
sum-of-Gaussians(centres), not blur(truth).

THE LESSON: the image-domain twin of the ACCUM-1 jitter negative and the generate_vector bare-codebook negative
-- you cannot manufacture detail that was never stored. A lossy smooth basis THREW AWAY the high frequency; no
negative-lobe loop recovers information that is not in the render. Sharpening un-low-passes a genuinely
low-passed signal; it cannot un-throw-away a lossy approximation.

No faculty, no tour line (the finding is the negative). The 2-D Van Cittert (gauss_blur2 + vc_sharpen2 in
holographic_splatsharpen.py) is the vehicle for the control.

Tests: +1 (1003 -> 1004). test_holographic_splatsharpen.py (the module _selftest: control recovers from a true
blur, negative shows the splat render cannot be improved at any setting).

## Robust reward/value accumulation -- D2 (cross-cutting: ACCUM-3 -> creature brain)

ACCUM-3's outlier clamping applied to the creature brain's value memory. Each prototype keeps a running-mean
return (`_ret[a][j] += alpha*(ret - mean)`); a single freak reward (a jackpot, a sensor glitch) folds straight
in and drags the estimate. robust_returns winsorises the residual to +/- k * `_ret_dev` before it lands, where
`_ret_dev` is the running typical |residual| (the reward NOISE scale).

THE DESIGN CHOICE (why ONE global scalar, not a per-prototype array): `_ret` is touched at 8+ sites (init,
append, evict, reorganize, clone, save/load); a parallel per-prototype scale array would be invasive and would
break old saves. ONE running scalar suffices because the noise SCALE -- unlike the mean -- is roughly constant
across prototypes, so a global |residual| estimate winsorises a mean-1 prototype and a mean-5 prototype equally
well (measured). It is cheap, serialises trivially (it is transient scratch -- like the existing EMAs, it is NOT
persisted and re-seeds from the first post-reload residual; only the FLAG is saved, via _STATE_FIELDS).

MEASURED: under 8% outlier rewards, ~3x lower value error than the plain running average (1.57 -> 0.53 in
isolation; the integration/selftest assert the brain's value() is markedly closer to the true mean). On CLEAN
data: no cost (0.0561 vs 0.0550). The win is ~3x, not ACCUM-3's ~100x, because the floored-alpha EMA already
damps outliers somewhat -- winsorisation adds the rest.

Off by default (robust_returns=False -> the plain update path is bit-identical). Wired as robust_returns= on
HolographicMind and the actions() faculty; carried through _blank/_clone and persisted via _STATE_FIELDS.

Tests: +2 (1004 -> 1006). test_holographic_robust_returns.py (the module _d2_selftest: lower error under
outliers, no clean-data cost, flag survives save/load) and test_robust_returns_resists_outlier_rewards in
test_integration.py.

## Coarse-to-fine splat densification -- C1 (cross-cutting: SCALE-1 + ADAPT-1 + 3DGS -> image gen)

3D-Gaussian-Splatting densification, from scratch. The one-shot aniso_fit places all K splats by matching
pursuit then runs ONE joint gradient fit; its kept negative is that the non-convex loss makes the result
depend on the warm start (a poor local optimum, sometimes divergence). densify_fit grows the set in STAGES:
place a fraction on the current residual (coarse scales first), jointly optimise ALL, place more where the
re-optimised fit still errs, optimise again.

THE MEASURED WIN (and why it is real, not just more steps): on a multi-scale target (broad blob + small sharp
details) densify reaches MSE ~1e-6 where the one-shot plateaus near ~1e-3 -- and the one-shot CANNOT close the
gap at ANY step count (measured 280/450/700 steps: it stays ~1e-3 and then DIVERGES past ~300, the non-convex
instability the negative warns of). So the staged placement is a strictly better WARM START, landing the final
joint fit in a basin the one-shot never finds. At MATCHED total compute (splat-steps) densify already wins; the
trade is that it uses several optimisation rounds, and the win is specific to MULTI-SCALE content (on a
single-scale field the one-shot is already near-optimal).

NOT manufacturing detail (contrast C4): C4 tried to sharpen detail the splats discarded and failed (you cannot
recover what was not stored). C1 does the opposite -- it finds a better ARRANGEMENT of the detail genuinely
present in the target. Different operation, different (positive) result.

REFACTOR: the Adam loop was extracted into the shared `_aniso_optimize(target, centers, amps, Ls, ...)` so
aniso_fit (iso warm start) and densify_fit (staged warm start) use ONE gradient engine -- no duplication; the
C3 early-stop lives in the helper. aniso_fit is bit-identical after the refactor (its selftest + the splat
suite confirm). Wired as the `splat_densify` faculty (pass stats={} to read stats['stages']).

Tests: +2 (1006 -> 1008). test_holographic_densify.py (the module _c1_selftest: densify reaches a markedly
better optimum than the one-shot) and test_splat_densify_beats_one_shot_on_multiscale in test_integration.py.

## Adaptive encoder resolution -- A3 (cross-cutting: CACHE-3 -> encoder) -- the one promising below-stack item

CACHE-3's equidistribution (place resolution by density) applied to the ScalarEncoder. The sweep's premise was
that the kernel is already near-optimal, so below-stack transfers are mostly negative -- A3 is the exception.

THE MAPPING: the ScalarEncoder is NOT a grid of kernels -- it is a Fourier-phase encoder whose kernel is
shift-invariant (uniform resolution across [lo,hi] by construction, Bochner). So "place kernels adaptively" has
no discrete kernels to move; the equivalent is to WARP the input axis by the value-density CDF, stretching dense
regions so they get finer effective resolution. fit_resolution(samples) fits that monotonic warp; encode warps
x before the phase rotation, decode unwarps the result.

THE FLOOR (the irradiance-caching validity-radius lesson, AGAIN): a PURE CDF warp drives sparse regions to
~zero resolution, where decodes go catastrophic (measured: sparse-region error 0.0073 uniform -> 0.2569 warped,
~35x WORSE) -- and those catastrophic tail decodes drag down the AVERAGE too. Mixing the CDF with the identity
(floor=0.2: keep >= 20% resolution everywhere) bounds the sparse loss to ~4x and LIFTS the in-distribution win
from ~15% to ~73%. Local weights with a validity radius, third appearance (after irradiance caching and the
splat refit).

MEASURED: non-uniform (bimodal) distribution ~55-73% lower decode error under noise; UNIFORM distribution ties
(warp = identity -- the control proving the gain is from density structure, not the machinery). KEPT CAVEAT: a
REALLOCATION, not free -- dense decodes ~4x better, sparse/out-of-distribution ~4x worse (floor-bounded). Fit
only when decoding in-distribution values.

REFACTOR: encode() split into _phase_encode(u) (the raw Fourier encoding) + the warp; decode() builds its grid
with _phase_encode and unwarps the result. Unfitted (no fit_resolution call) -> warp is the identity ->
bit-identical to the plain encoder. A primitive enhancement (no UnifiedMind faculty -- the mind has no scalar
faculty to attach it to; a faculty must earn its method).

Tests: +2 (1008 -> 1010). test_holographic_adaptive_encoder.py (the module _a3_selftest: win on non-uniform,
tie on uniform, unfitted bit-identical) and test_adaptive_encoder_resolution_on_nonuniform_data in
test_integration.py.

## Low-discrepancy exploration -- D1 (cross-cutting: SAMPLE-1 -> creature) -- KEPT NEGATIVE

THE PROPOSAL (Togelius's seat, caveat on record): SAMPLE-1's low-discrepancy sampling covers a space more evenly
than i.i.d. random, so drive the creature's exploration from a low-discrepancy sequence instead of epsilon-random.

THE MEASURED ANSWER: it actively HURTS (not just neutral), for a structural reason. SAMPLE-1's win is for placing
each sample INDEPENDENTLY. A creature WALKS its state space, and a walk ACCUMULATES displacement. A
low-discrepancy sequence over the four moves is BALANCED (N vs S, E vs W spread evenly in time), so the steps
CANCEL and the agent stays pinned near start. A random walk's runs and imbalances ARE the diffusive drift that
explores. Measured (open grid, 400 steps): random ~162 distinct cells, low-discrepancy ~12 -- an order of
magnitude WORSE.

THE LESSON: this pins down WHY a transfer that pays for direct sampling fails for sequential exploration. Low
discrepancy MINIMISES the imbalance of a point set; spatial exploration NEEDS the imbalance (displacement is the
cumulative SUM of the steps; a balanced sum is ~zero). Opposed goals. Togelius's caveat ("buys almost nothing
over a handful of discrete actions") is stronger than predicted -- harmful, not neutral. The real coverage lever
is count-based / novelty exploration, partly already in the brain's novelty_bonus.

No faculty, no tour line (the finding is the negative).

Tests: +1 (1010 -> 1011). test_holographic_ldexplore.py (the module _selftest: low-discrepancy covers far fewer
cells than random; sanity that random does drift).

## MIS-weighted steered generation -- B1 (cross-cutting: MIS-1 -> text gen) -- KEPT NO-OP

THE PROPOSAL (Pharr's seat, balance heuristic, precondition on record): steered_generate keeps the candidate
with the best verifier (coherence) score among the predictor's top-beam, discarding the predictor's ranking at
selection. So combine the predictor's coupling score and the verifier's coherence score by the balance
heuristic, weighting each by reliability, instead of letting the verifier override.

THE MEASURED ANSWER: NO-OP -- the balance combination gives results IDENTICAL to verifier-only, structurally.
steered_generate already uses the predictor as the candidate GATE (it restricts to the top-`beam` before the
verifier picks). WITHIN that beam the coupling scores are nearly flat (all are the most-probable continuations),
so after the softmax that puts them on a common scale the predictor's factor is ~uniform and cannot move the
argmax of the product. The verifier dominates -> MIS == verifier. The predictor's information is ALREADY fully
spent on gating; re-using it as a within-beam weight is redundant.

MEASURED (loop-trap corpus: a frequent 'ping pong' cycle mixed with coherent clauses): the verifier DOES escape
the greedy loop (distinct-token ratio ~0.44 vs greedy ~0.15 -- the setup is real), but the balance combination
matches the verifier EXACTLY on both fluency (valid-bigram rate) and anti-looping (distinct ratio).

THE LESSON: MIS combines two estimators of the SAME quantity over a COMMON support on a common scale (Pharr's
precondition). The predictor does not estimate over the same set as the verifier -- it FILTERS to its top-beam
first -- so there is nothing for the balance heuristic to balance. A gate followed by a re-ranker is not the MIS
setting. (Compare D1: another transfer that fails because the operation is structurally different from the one
the technique was built for.)

No faculty, no tour line (the finding is the no-op).

Tests: +1 (1011 -> 1012). test_holographic_misgen.py (the module _selftest: verifier escapes the loop -- setup
real; MIS == verifier on distinct ratio -- the no-op).

## Phase-domain image morph -- C2 (cross-cutting: phase vocoder / PHASE-1 -> the image morph path)

The SAME phase-domain lesson PHASE-1 already established for FHRR vectors, now applied to morph_scene (which only
did DCT-coefficient slerp). NOT a new principle -- the rediscovery is honest: PHASE-1 + its wrapping bound were
already on record for vectors; C2 is its application to images.

THE MAPPING: morph in the 2-D FFT domain, interpolating each bin's MAGNITUDE linearly and PHASE along the
shortest arc. By the Fourier shift theorem a translation is a phase ramp, so phase interpolation SLIDES a
translated feature to its intermediate position (a compact moving blob) where the DCT slerp interpolates the
feature's SHAPE and SMEARS it into an elongated oval. VERIFIED VISUALLY (rendered the frames, did not trust the
scalar): at shift 6 the DCT midpoint is a stretched oval, the phase midpoint a compact round blob sliding cleanly.

MEASURED (blob on 48x48, metric = midpoint peak / endpoint peak): shift 6 -> DCT 0.85 vs phase 0.97 (phase
WINS); shift 16 -> DCT 0.70 vs phase 0.67 (the wrap -- phase slightly WORSE). Through morph_scene on a 28x28
field: phase 0.83 vs dct 0.73.

THE BOUND (kept loud, same as the vector version): phase is mod 2*pi. For a LARGE translation the bin phase
differences exceed pi, the shortest arc wraps, and the morph falls back to a ghosted crossfade (rendered: BOTH
methods show two blobs at shift 24). The win holds only within the per-step displacement that keeps bin phase
differences under pi -- the Nyquist limit on phase, exactly the phase-unwrapping problem the vocoder lives with.
So method='phase' is for small-motion morphs, method='dct' for arbitrary structure change.

Wired as morph_scene(method='phase') (default 'dct' -> bit-identical). Added morph_image_phase to the existing
holographic_phasemorph.py rather than a new module (same family).

Tests: +2 (1012 -> 1014). test_holographic_phasemorph_image_c2_selftest (the module _c2_selftest: phase slide
beats crossfade small, wraps large) and test_morph_scene_phase_slides_translation_better_than_dct in
test_integration.py.

## Re-anchored lookahead for the creature -- D4 (cross-cutting: RAY-1 -> model-based planning) -- KEPT NEGATIVE

THE PROPOSAL (the research item; Baker / Adamatzky): the creature is purely REACTIVE (best learned value in the
current state, no forward model). Give it one -- a per-action transition operator learned from its experience --
roll it out a few steps to imagine each action's consequences, pick the best rollout, RE-ANCHORING each predicted
state (RAY-1: clean up every hop or the rollout compounds error and decays).

PART 1 -- THE MECHANISM WORKS (RAY-1 confirmed in a new domain). Forward model: delta_a = normalize(mean
unbind(next, state)); predict(s,a) = bind(s, delta_a). Naive rollout DECAYS with depth (cosine to true 0.65 ->
0.53 over four steps). Re-anchoring the predicted state to a codebook of seen states each step holds it
ON-MANIFOLD (cosine ~constant 0.77 across depth). Re-anchoring is as load-bearing here as for bind-chain traversal.

PART 2 -- THE APPLICATION IS REDUNDANT (the kept negative). Re-anchored lookahead ranks the four actions
IDENTICALLY to the plain reactive value -- 98-100% action-rank agreement -- so it can never decide differently;
on stars it ties the reactive policy and loses its small epsilon (marginally worse). PRECISE ROOT CAUSE
(diagnosed): the four predicted leaves sit at 0.974-0.994 PAIRWISE COSINE -- a single per-action bind displacement
COLLAPSES all actions to nearly the same predicted next-state, because in the egocentric sense-space the AVERAGE
sense-change is similar across directions (directional specificity is lost in the averaging). So the lookahead
bonus varies by std 0.0075 across actions while the reactive value varies by std 0.37 -- lookahead carries no
differentiated signal. Secondary structural reason: the creature's value IS the Monte-Carlo discounted return,
already horizon-aware, so model-based planning recomputes (through a noisy model) what the model-free value
already encodes.

THE CROSS-CUTTING LESSON (throughline with C4, D1, B1): the re-anchoring transfer is mechanically sound, but the
creature's state space does not admit a forward model good enough for the application. The structural mismatch is
egocentric-sense-space averaging -- a per-action linear/bind operator cannot capture the position-dependent,
action-specific consequences a real lookahead needs, so it predicts the same future for every action and the
planner is blind. A right technique applied to an operation whose shape defeats it -- the same failure as the
splat sharpener (a sum is not a blur), LD exploration (independent points for a sequential walk), and MIS
generation (a gate, not a second estimator).

No faculty, no tour line (the finding is the negative).

Tests: +1 (1014 -> 1015). test_holographic_lookahead.py (the module _selftest, CI-fast ~1.7s: the forward model
collapses the actions -- leaves > 0.9 pairwise cosine -- so lookahead-vs-reactive rank agreement > 0.9).

## The cross-cutting PROBE SWEEP -- A1, A2, B2, B4, D3, D5 -- SIX KEPT NEGATIVES

The cross-cutting backlog's probe items: transfers the panel pre-judged as no-ops. All six measured on the real
substrate, all six confirmed the prior. They ship as ONE shared module (holographic_probesweep.py) with one
measurement+assert and one test each -- no faculty, no tour line. The reason each fails is the artifact, and it is
the C4/D1/B1/D4 throughline: a sound technique applied to an operation whose shape defeats it. Two failure classes:

CONCENTRATION OF MEASURE (the kernel is already near-optimal -- no slack to win):
  A1 LD/blue-noise codebook [SAMPLE-1 -> kernel]. Riesz repulsion vs i.i.d. atoms: max coherence ~3% lower, MEAN
     coherence unchanged, at every dim 64..1024; capacity identical (40 pairs in d=512: 0.70 recall both); 500
     steps barely move it. Random atoms are already near-uniform on the sphere. NO-OP.
  B4 LD sampling in generation [SAMPLE-1 -> text]. LD-noise diffusion == i.i.d.-noise diffusion on diversity (41 vs
     42 distinct atoms) and validity (0.507 vs 0.510) -- the diffusion is ATTRACTOR-dominated (cleanup decides the
     landing, not the noise). A categorical token draw is a single pick and cannot use LD at all. NO-OP.
  D5 Observation denoising [XDATA-1/2 -> creature]. Snapping the noisy state to the seen-state manifold does not
     improve the decision -- argmax preserved ~equally from raw and denoised (0.68 vs 0.68 low noise) because the
     value's similarity-weighting already absorbs noise -- and OVER-SMOOTHS at low noise (value err 0.34 -> 0.42).
     The high-dim encoder is its own denoiser. NO-OP.

WRONG-SHAPED OPERATION (the technique's precondition does not hold):
  A2 Negative-lobe cleanup sharpening [XDATA-3 -> kernel]. Deconvolving the similarity profile (subtract
     alpha*(G-I)@s) then argmax HURTS discrete cleanup (correlated atoms: 0.18 -> 0.00, amplifies noise) and is a
     no-op for orthogonal atoms (G~=I). Hard NN is already Bayes-optimal for 'which atom'. NEGATIVE.
  B2 Throughput-gated generation [RAY-1 -> text]. The running coherence does NOT separate in-distribution from a
     garbage seed (means -37 vs -39, overlapping) -- steered generation pulls any start back to coherent
     continuations, so there is no incoherent tail for a gate to catch. The coherence defense is redundant; the
     abstention has nothing to fire on. REDUNDANT NO-OP.
  D3 MIS-combined decision [MIS-1 -> creature]. In typical states the soft blend == the veto (0 lethal choices over
     84 sensed-danger states, because value already disfavours danger). But the veto's value is the RESIDUAL the
     value misestimates (survival bench: ~0.6%/step -> 67-73% of long lives die without it); a soft penalty picks a
     lethal move whenever the value margin exceeds the penalty, so it cannot give the guarantee. A safety
     constraint is not an estimator to blend. NEGATIVE.

Tests: +6 (1015 -> 1021). test_holographic_probesweep.py (one test per probe; the module _selftest runs all six
asserts, CI-fast ~0.8s).

## Creature-brain performance pass: value_batch / budget knob -- and a MEASURED NEGATIVE on batched value

Three requests for holographic_creature, all bound by the hard constraint that the creature is
TIE-SENSITIVE (a 1e-16 difference at the top-k boundary flips a maze trajectory -- the bind_batch
lesson). So every change had to be BIT-IDENTICAL to enter the decision path, or stay out. Each was
measured before shipping.

REQUEST 1 -- batched value() for speed: MEASURED NO SPEEDUP, kept negative. The premise was that a
single stacked `U_all @ state` plus a segment-reduce would cut the hot path. It does not, for two
reasons measured on real trained brains (banks ~84/action, dim 512, 6000 value calls):
  - No speedup at any scale. A bit-identical per-action-gemv batch is ~2% faster (the value()
    wrapper overhead is negligible); a stacked one-matmul is 73% SLOWER (the per-call concatenate
    costs more than it saves); a PERSISTENT stack (no per-call concat) is 0.94-1.01x at banks
    84/300/800; a padded-tensor vectorised top-k is 1.03-1.05x. BLAS already runs four small
    matrix-vector products about as fast as one big one, and the per-action top-k (argpartition)
    is irreducible -- it cannot be merged because k is per action.
  - The stacked matmul is also NOT bit-identical to the per-action product: BLAS gemv blocks by row
    count, so `concat(U) @ s` differs from `[U_a @ s]` at ~1e-16 -- the exact tie-break hazard that
    kept bind_batch out of the encoder. (An M-INDEPENDENT reduction `(U*s).sum(1)` IS batchable
    bit-identically, but does not match gemv, so adopting it would shift the whole baseline.)
  A cProfile of a real run shows why the lever was misjudged: decide() is 44% of the run and
  encode() (the per-sense binds -> FFTs; _raw_fft alone is 25% of total) is the OTHER 44%. value()
  is genuinely efficient; the cost is real work, not Python overhead. The other half (encode) is the
  bind_batch territory already known to be tie-unsafe. value_batch is still SHIPPED as the requested
  API (bit-identical), honestly documented as an interface convenience, not a hot-path win.

REQUEST 3 -- skip the basis-width check in value(): mostly a NO-OP. decide() runs perceive_vec once,
so value() already receives a projected state and its width-check evaluates FALSE -- no redundant
projection happens. The only real saving is for a CONSOLIDATED brain handed a RAW state scoring
several actions: value_batch projects ONCE instead of once per action (measured 1.11x on that path).
Shipped as `_value_projected` (value() minus the branch, bit-identical) and folded into value_batch.

REQUEST 2 -- cheaper auto_maintain: shipped as a caller-controlled BUDGET knob (grains / refresh on
auto_maintain, plus instance defaults), NOT an auto-gate. Measured findings:
  - The proposed surprise auto-gate is UNRELIABLE: self.surprise is an EMA of reward prediction
    error, which tracks reward NOISE as much as regime change, so it sat at 0.77 (floor 0.4) after
    stable training -- it is not a "nothing shifted" signal. So the knob is caller-controlled.
  - Savings are large: full 8-way 31 ms/tick; 1 grain + refresh (4 cand) 17 ms; 1 grain, no refresh
    (2 cand) 9 ms; keep-only (1 cand) 3 ms.
  - It is a speed/SELECTION-thoroughness trade, not free. Trimming fold grains is the safer lever
    (both families still compete -- no missed-shift risk -- and here 1 grain picked the SAME memory
    as 3, same held-out value); dropping the refresh family is sharper (it changed the selection
    from a refresh to a preserve), so a stable courier that turns refresh off should periodically run
    a full tick. Default (grains=(0.9,0.82,0.75), refresh=True) reproduces current behaviour exactly
    -- the rescue-cracks canary passes unchanged.

Tests: +6 (1021 -> 1027). test_holographic_creature_batch.py (value_batch / _value_projected
bit-identity on un-consolidated and consolidated-raw states; budget default-is-full and lean-trims).

## Corridor planning (re-anchoring): plan() / replan_needed -- the way past the per-structure capacity cap

A user stress-testing fleet turn-by-turn navigation hit a "capacity limit" on how many steps a set of
directions could hold. That limit is the HRR cliff, and it is a property of the ENCODING and the dimension,
not a holostuff wall -- and the fix was already in the box:

  - MEASURED cliff: a route stored as ONE undirected bundle (consecutive bind(tile_i, tile_{i+1})) decodes
    only ~1 tile at dim 512, ~3 at 1024, ~5 at 2048 before crosstalk + the predecessor leak win.
  - MEASURED fix: the SAME route as a DIRECTED structure (the permutation direction role, RAY-3) walked
    with the throughput gate (RAY-1) decodes its full reliable prefix -- 15/15 at dim 512-1024 for a
    16-route, 23/23 at dim 2048. The cap is just that prefix length; more dim buys more.

The way PAST the per-structure cap is re-anchoring, exactly Russian roulette for a decaying ray: don't push
one structure past its reliable depth -- bake a CORRIDOR (the next ~12-16 downhill steps, short enough to
decode cleanly), execute it, and re-anchor at the decision point. Arbitrarily long routes become a sequence
of cap-sized clean corridors; the brain is consulted once per corridor, not once per tile.

holographic_plan.py is the API for that pattern, built ENTIRELY on existing tested pieces (holographic_directed
+ holographic_traverse):
  - plan(start, field_step, max_steps, floor, action_of, is_branch) rolls out the goal field's downhill path
    (field_step is the caller's gradient/flow/policy step; stops at is_branch or max_steps), bakes it as a
    directed chain, and returns a Plan(memory, nodes, route, actions, throughputs, stopped, ds): the compact
    plan hypervector, the decoded tile route, the decoded direction labels, and a per-step throughput.
  - replan_needed(plan, executed, tile_ok, floor) is the cheap per-tick guard -- True (re-anchor) on
    exhaustion / next-step throughput below floor / a blocked next tile; else execute the baked step. No
    value() calls, no decode work.
Wired into UnifiedMind as plan() / replan_needed (general functionality belongs in the mind, not siloed in
the creature). KEPT NEGATIVE: a corridor that REVISITS a tile (a tight loop) can confuse cleanup, since two
steps map to near-identical vectors -- straight corridors to the next decision point are distinct by
construction; loops want segmenting.

This is also the right answer to the batched-value request: baking a corridor collapses the ~72% trivial
straight-line steps into near-free executions of a baked plan, so the per-tick value() loop that batching
tried (and measured-failed) to speed up barely runs on those steps at all -- the brain only fires at the
decision points. value_batch remains the (bit-identical, no-speedup) API for the genuine decisions that
remain.

Tests: +2 (1027 -> 1029). test_holographic_plan.py (selftest: at-cap corridor decodes fully, over-cap
reports only its reliable prefix, replan_needed gating) + a UnifiedMind integration test.

## Creature-mind migration, Phase 0: the creature reaches the planning faculty

The audit for the creature<->UnifiedMind migration (holostuff_creature_migration_plan.md) found the real
silo is ACCESS, not extraction: the creature's surface is almost all the RL layer (value/decide/remember/
auto_maintain/...), it does not hoard general functionality, and UnifiedMind currently WRAPS it
(self._brain). The genuinely-general faculties (plan, directed_structure, traverse, recall, denoise) live in
the mind and the creature simply cannot call them.

Phase 0 (behavior-preserving, shippable now) closes the part the navigation user needs: HolographicMind.plan
/ replan_needed, delegating to the SAME holographic_plan module UnifiedMind.plan uses -- so an NPC on a
creature can bake a corridor and re-anchor without the engine first inverting the creature<->mind
relationship. These two are substrate-level (operate on supplied vectors, not the creature's value memory),
so they add the capability with zero weight, no nesting (the creature does NOT build a UnifiedMind, which
would build its own nested brain), no circular import, and -- confirmed -- a bit-identical decision path (the
rescue-cracks canary passes unchanged).

Phases 1-2 (invert the dependency so the creature is a layer ON UnifiedMind; optionally unify the encoder/
memory onto the shared substrate, which is behavior-CHANGING and needs a canary re-baseline) are real
architectural commitments left for Moose to direct -- see the migration plan for the options (composition
recommended over inheritance for this shape), constraints (tie-sensitivity, the wide standalone surface, the
circular-wrapping problem), and sequencing.

Tests: +1 (1029 -> 1030). test_creature_plan_bakes_a_corridor in test_holographic_creature_batch.py.

## CreatureMind: the creature as a LAYER on the one mind (the architecture, made concrete)

Moose's architecture, stated plainly: there is ONE mind (UnifiedMind) where all general functionality lives
-- the single encoder, the memory, recall, planning, denoising, the decision machinery -- and specialized
minds (creature behavior, image generation, ...) are thin LAYERS on top that inherit every faculty and add
only their domain wiring. The creature mind is the reference DEMO of that pattern. UnifiedMind's own encoder
docstring already declares the intent -- "this is the only encoder in the system; the memory and the brain
never encode anything themselves" -- and the standalone creature's separate CreatureEncoder is exactly the
deviation from it.

holographic_creature_mind.CreatureMind(UnifiedMind) writes the target shape down: it subclasses UnifiedMind
(inherits the whole faculty suite), names its actions, and expresses the creature loop entirely over
inherited faculties -- `sense` is the one encoder's `perceive` in record mode (no separate encoder), `act`
is the inherited `decide`, `learn` is the inherited `reinforce`, and `plan` / `recall` / `denoise` are
there for free. So a CreatureMind is, in ONE object, a full mind that also acts, learns, and navigates --
the specialization is a handful of convenience methods, nothing rebuilt. It is the template for any other
specialized mind: subclass UnifiedMind and wire your domain on top.

MEASURED (selftest): a CreatureMind senses through the one encoder (bit-identical to perceive(...,'record')),
learns the rewarded action via inherited decide/reinforce, and bakes a corridor via the inherited plan
faculty -- all on one object.

This sits BESIDE the standalone HolographicMind (the lower-level RL engine UnifiedMind still wraps today)
during the migration; the migration plan's later phases retire that duplication (one encoder, one memory)
and move the RL methods out of UnifiedMind's core into the layer -- a wide change touching the tests/tour
that call unified.reinforce/decide, with a canary re-baseline, since the creature is a teaching demo not a
frozen artifact. CreatureMind is where that migration lands, written down now so the destination is real.

Also fixed in passing: plan()'s decode walked up to len(nodes)+2 steps, so a permissive floor let the
terminal node's noisy unbind oscillate PAST the corridor end ([1..6,5,6,5]); capped to exactly the
corridor's edge count (len(nodes)-1), the decode never invents tiles beyond what was rolled out.

Tests: +1 (1030 -> 1031). test_holographic_creature_mind.py (CreatureMind selftest: one encoder for senses,
act/learn on inherited machinery, inherited planning -- the layer-on-one-mind pattern).

## MEASURED: is the bespoke creature value memory a redundant old path? -- No. (a kept result + negative)

The long-running confusion was whether HolographicMind's prototype value memory is a "tumor" duplicating the
unified mind, or an essential component. Audit first: HolographicMind imports only the shared kernel
(bind/bundle/permute/Vocabulary), and through UnifiedMind its ENCODING already goes through the one encoder
(decide/reinforce call perceive; the creature encoder is bypassed) and the calibrated honesty layer is wired
in. The one thing genuinely separate is its value MEMORY (per-action prototype banks + soft k-NN returns
regression) -- and the modern advancements (Hopfield cleanup, resonator, denoise) address clean recall /
factorization, not value regression, so it was not obvious they would help it.

So we measured it instead of guessing (exp_value_memory.py). Task: 16 egocentric situations (4 food dirs x 4
distractor combos, best action = move toward food), train on 12, test greedy accuracy on all 16 (in-sample +
held-out generalization). All learners handed the SAME perceive-encoded vectors (encoder not a variable),
SAME episode stream, SAME exploration -- only the value memory differs. 6 seeds, dim 512.

  RESULT (greedy accuracy, chance 0.25):
    bespoke per-action memory : in 0.96   gen 0.75
    unified-memory, hard class: in 0.57   gen 0.21  (NEGATIVE -- naive replacement fails)
    unified-memory, soft k-NN : in 0.57   gen 0.25  (NEGATIVE -- mechanism-matched rival ALSO fails)

The bespoke memory wins decisively, and -- the important negative -- the gap is NOT just soft-vs-hard: a soft
k-NN value regression built on the unified SelfOrganizingMind (the fair, mechanism-matched rival) still
collapses to chance generalization. The bespoke's edge is its PER-ACTION prototype organization: action a's
bank holds every state where a was taken, so value(s,a) is a soft k-NN Q-value regression over similar states
across ALL situations -- which generalizes ("states like this, when you went east, paid off"). The
situation-class + value-table structure first buckets the state into a class (losing the cross-situation
regression), and even soft-weighting over classes does not recover it (it only approaches the bespoke at a
tuned high novelty_floor: floor=0.7 gives soft gen 0.67, still under 0.75 and floor-sensitive).

VERDICT: the value memory is a deliberate, measured-essential RL engine, not a redundant old path. Keep it.
The remaining "confusion" to fix is only the SURFACE: the public pattern for an agent is
CreatureMind(UnifiedMind) (not building from HolographicMind directly), and the one real duplication left is
the standalone CreatureEncoder still used by navigator/moe/lookahead/core/app -- a minor cleanup (route them
through the mind's encoder), not an excision. The measurement saved a wrong refactor; the negative is kept.

Tests: +0 (no faculty changed; HolographicMind docstring updated to record the verdict, exp_value_memory.py
kept as provenance).

## Follow-up: is the standalone CreatureEncoder a stray duplicate? -- No (the value-memory lesson, again)

After the value-memory result, the last "loose thread" looked like the standalone CreatureEncoder used
outside UnifiedMind. Looking closely (not assuming) settled it the same way: it is NOT redundant. Through
UnifiedMind the encoding already goes through the ONE encoder (perceive; the brain/memory never encode
themselves), so the rule holds where stated. CreatureEncoder is the creature DOMAIN's encoder: role/filler
binding (the shared primitive) PLUS (a) build_state's action-memory -- a working memory of recent moves
perceive has no notion of (app's maze console uses it) -- and (b) the `seen` role/value tracking that
HolographicMind.describe REQUIRES to decode a state back into sense terms. The rescue canary is tie-sensitive
to its exact output (the kept-negative in encode(): a 1e-16 change flips the trajectory). And the modules
reuse it ON PURPOSE: navigator is an explicit "inception" demo (same brain + same encoder, new world), not
accidental duplication -- and it is three modules (navigator/lookahead/app), not the five I'd loosely said
(moe/core use only the engine, never the encoder).

So, like the per-action value memory, CreatureEncoder earns its place; routing those uses through perceive
would drop describe + action-memory and re-baseline the whole tie-sensitive suite. Corrected the docstrings
(HolographicMind + CreatureEncoder) to say this accurately instead of mislabeling it "duplication to tidy."
The genuine, low-value, optional leftover is only the ~2-line role/filler bind+bundle that appears in both
encoders; factoring it to a shared helper is cosmetic and touches the tie-sensitive path, so it is not done.

Tests: +0 (docstrings/comments only; no behavior changed).

---

## plan_route: a whole arbitrarily-long route in one call, by chaining cap-sized corridors

A delivery-game user kept hitting the ~15-tile cap and reported it "still exists." They were right about
the surface fact and wrong about its scope, and the distinction matters: the cap bounds ONE baked structure,
not the route you can navigate. Measured the two paths at dim 512 on a 45-tile straight route:

  * cram all 45 tiles into ONE plan(max_steps=45): the directed structure is overstuffed and the decode
    COLLAPSES -- it came back with **1** step, not a clean 15-prefix. The cliff is steeper the more you cram;
    "~15" is the reliable depth for a corridor-SIZED structure, not a floor you get for free at any length.
  * chain cap-sized corridors, re-anchoring at each leg's reliably-decoded end: the full **44-step** route
    decodes EXACTLY. Re-anchoring resets the HRR accumulation each leg, so each leg stays inside its capacity.

This is the same move plan() + replan_needed already enable (bake a corridor, drive it, re-anchor on the
gate) -- but a user calling plan() ONCE and expecting the whole route hits the wall. plan_route runs that
loop internally: it chains plan() corridors, re-anchors at `nodes[route[-1]]` (the last RELIABLY-decoded
tile -- never past it, so a short-but-clean leg just re-anchors sooner), breaks on field_end/branch only
when the decode actually reached the leg's last tile, and caps the whole route at `max_total`. Returns a
Route (full action sequence, the chained corridors, stop reason, re-anchor count, step total). Wired as a
method on UnifiedMind AND HolographicMind (delegating to the module, like plan/replan_needed); CreatureMind
inherits it. Verified: 39/40-tile routes decode exactly through all three minds.

KEPT NEGATIVE: `corridor` must stay at/under the dim's reliable decode depth (default 14, safe at dim
512-1024). Set it too high and that leg overstuffs its OWN structure -- the same cliff, per leg: measured
corridor=30 at dim 512 does NOT recover the full route (it skips/corrupts tiles). plan_route does not, and
cannot, rescue an over-long leg; it only removes the cliff by keeping each leg small. The realtime courier
still wants plan() + replan_needed (bake-as-you-go, reacts to traffic); plan_route is for getting the WHOLE
route in hand at once (display / validate / pre-plan a leg).

Tests: +4 (1031 -> 1035). test_holographic_plan.py grew the selftest (45-tile chained route exact vs a
collapsed single plan; max_total prefix; over-long-corridor negative) and added three named API tests;
test_integration.py added a plan_route-through-the-mind test. Files: holographic_plan.py (plan_route + Route),
holographic_unified.py / holographic_creature.py (the wired method), test_holographic_plan.py, test_integration.py,
tour.py.

---

## chunk_route: the explicit-sequence twin -- scaling to GPS/experiment size by chunking

The capacity question came back as a scaling worry: would the ~15 cap make holostuff useless for GPS
navigation or a long experiment plan? Measured the answer rather than asserting it. A 200-step route at
dim 512: crammed into ONE structure it decodes 1 step (the cliff); chunked into <=14-element pieces with
re-anchoring it replays all 199 steps EXACTLY in 15 chunks, in 52 ms. So the worry is unfounded -- chunking
makes EFFECTIVE length unbounded at LINEAR cost (~N/14 pieces), and each piece is one compact vector. The
per-piece cap is physics (a fixed-width structure can't hold unbounded order, like any bounded buffer);
chunking is the standard, correct workaround, not a hack.

The thin orchestration layer this needs already half-existed: plan_route (prior section) chunks a route you
DISCOVER by following a goal field. The genuine gap was the EXPLICIT case -- a sequence you ALREADY HAVE (GPS
waypoints from a planner, a scientist's fixed protocol, any known list). You had to hand-write a nearest-match
field_step to feed plan_route. chunk_route closes that: hand it the ordered list, it splits by position into
<=chunk pieces (overlapping by one element so each piece re-anchors exactly where the last ended -- nothing
skipped or double-counted), bakes each as a clean directed structure, and returns the full replayable sequence
plus the chunk vectors. It is implemented directly on build + gated_traverse (not by wrapping plan_route), so
it does not depend on nearest-match rediscovery -- it knows the order and chunks it. Wired on UnifiedMind AND
HolographicMind; CreatureMind inherits. Verified: a 200-element list replays exactly through all three.

The relationship to the rest: this is the third chunking mechanism in the engine, one per kind of long thing.
Routes you discover -> plan_route. Sequences you hold -> chunk_route. Programs that outgrow one structure ->
HoloMachine define/CALL (sub-programs called from a short top-level program). All three are the same lesson --
keep each structure inside its capacity and coordinate the pieces -- which is the project's recurring theme of
beating a hard limit with composition rather than pretending the limit isn't there.

KEPT NEGATIVE (same shape as plan_route's): `chunk` must stay at/under the dim's reliable decode depth (default
14, safe at dim 512-1024); an over-long chunk overstuffs its own piece -- the cliff, per chunk. And the elements
must be DISTINGUISHABLE (a codebook): a sequence that revisits the same element can confuse a chunk's cleanup,
since two steps map to near-identical vectors. chunk_route removes the cliff by keeping pieces small; it cannot
rescue an over-long piece or a non-distinguishable alphabet.

Tests: +3 (1035 -> 1038). test_holographic_plan.py: chunk_route replays an explicit 200-step sequence exactly
in ~15 compact chunks, and degenerate (empty / single-element / fits-in-one-chunk) inputs are safe;
test_integration.py: a 200-step explicit sequence through the mind. Files: holographic_plan.py (chunk_route),
holographic_unified.py / holographic_creature.py (the wired method), test_holographic_plan.py, test_integration.py,
tour.py.

---

## run_chunked: VSA programs past the single-program cap (chunking transfer backlog, item P1)

First build off the chunking-transfer sweep. The sweep's headline question -- does the chunk-and-re-anchor
lesson unlock more complex VSA programs? -- measured YES, with a load-bearing negative. A 60-instruction
HoloMachine program at dim 1024 (single-program cap ~20-32) decodes to garbage as one structure (cosine 0.08
to the intended bind-chain). The OBVIOUS fix -- factor it into define()d functions and CALL them -- ALSO
FAILS (cosine 0.06): CALL pulls each sub-program out of a BUNDLED library, and bundling several
function-vectors into one library vector re-introduces the very cliff (the docstring's "busy disk"
crosstalk). The fix that works is the true chunk_route analog: each chunk is its OWN clean program vector and
the HOST threads the accumulator across them -- 60 instructions then run at cosine 1.000.

Shipped HoloMachine.run_chunked(program, chunk=14, ...): strips a trailing HALT, splits into <=chunk pieces
(never ending a chunk on IFMATCH/REPEAT so a gate/repeat stays with the instruction it targets), assembles
each piece to its own vector, runs them threading the ACC, and stops the whole run if a mid-program HALT
fires. Returns (acc, trace) like run(), trace concatenated. Verified exact on three different long programs
(60 binds; 50 binds different phase; 40 mixed bind/bundle/permute), bit-equivalent to run() on a short
program, control-construct-intact at a forced chunk=1 seam, and mid-HALT stops the run.

KEPT NEGATIVE / the operand-dependent edge: the chunk size must be WELL UNDER the cliff, not on it. At dim
1024 the decode is solid through ~18 instructions but turns OPERAND-DEPENDENT right at ~20 -- a 20-instruction
chunk decoded for two operand sequences and FAILED for a third in the same program (cosine 0.448). This is
the same lesson as plan_route's over-long-corridor negative, sharper: near the edge, success depends on the
specific operands, so the default 14 leaves deliberate margin. The reliable length grows with dim (chunk=20
is solid at dim 2048+), so raise chunk at higher dim. And the CALL-the-library route is kept as a test
(test_call_library_does_NOT_chunk_a_long_program_kept_negative) so nobody reaches for it.

run_chunked is a HoloMachine method, NOT a UnifiedMind faculty -- the VM stays adjacent to the mind (the
integration plan's standing decision), so there is no UnifiedMind wiring or integration test, only the
machine's own tests. The user guide (writing_vsa_programs.md) gained a "Running a program past the cap"
section and the limits section was corrected (the old "factor into functions" advice was the measured
negative).

Tests: +4 (1038 -> 1042). test_holographic_machine.py: run_chunked past the cap (60 instr exact vs a
collapsed single program), equivalence to run() on a short program, constructs-intact + mid-HALT, and the
CALL-library kept negative. Files: holographic_machine.py (run_chunked), test_holographic_machine.py, tour.py,
writing_vsa_programs.md, README (test-list entry + counts).

---

## RouteIndex: sub-linear random access into a chunked route (chunking transfer backlog, item X3)

Second build off the chunking-transfer sweep, Pharr's acceleration-structure angle. A long route is now many
chunks (plan_route / chunk_route), and "where am I on it?" should be a jump, not a replay from the start.
RouteIndex is a BVH over the chunks: index each chunk by a SUMMARY vector (the bundle of its tiles), then
locate a query two-level -- nearest chunk summary (level 1), then nearest tile within that chunk (level 2).
Measured on a 200-tile route: 200/200 tiles located exactly at ~28-30 comparisons per query vs 200 for a flat
scan (~6.9x fewer). Why the bundle summary is a usable index: a tile in a chunk has cosine ~1/sqrt(chunk_size)
to that chunk's summary (it is one of its components) and ~0 to the others, so argmax over summaries is the
right chunk -- the same bundle-crosstalk that CAPS a single structure is what makes the summary discriminative
here. The global_step the locate returns accounts for the one-tile overlap between chunks, so it recovers the
true route index exactly (verified g == t for sampled tiles).

Shipped holographic_plan.RouteIndex(route): precomputes normalized chunk summaries and per-chunk start offsets
in __init__; .locate(query) -> (chunk, position_in_chunk, global_step); .n_chunks. Built once, queried many --
the courier asking its position every tick is the repeated-query case this amortises. Wired as m.index_route(
route) -> RouteIndex on UnifiedMind AND HolographicMind; CreatureMind inherits. Empty route is safe (returns
(-1,-1,-1)). Verified locating across all three minds (tile 137 -> chunk 9, pos 11, step 137, exact).

Tests: +3 (1042 -> 1045). test_holographic_plan.py: RouteIndex locates every tile in the right chunk at the
exact position with sub-linear comparisons, and the empty-route case is safe; test_integration.py: random
access through the mind. Files: holographic_plan.py (RouteIndex + the bundle/cosine import),
holographic_unified.py / holographic_creature.py (index_route), test_holographic_plan.py, test_integration.py,
tour.py, README (test-list entry + counts).

---

## Determinism / tie-break audit of the chunking seams (chunking transfer backlog, item C2)

Macklin's discipline applied to the three seams just added to the plan module (plan_route, chunk_route,
RouteIndex): a bit-exact change must stay bit-exact, and a query must never resolve on a knife-edge tie. The
audit came back CLEAN -- no fix needed, so the deliverable is a regression test that locks the property in.
Measured: chunk_route and plan_route produce identical actions AND bit-identical chunk vectors run-to-run at a
fixed seed; RouteIndex summaries are bit-identical and .locate is deterministic; a deliberately ambiguous query
(equidistant between two tiles in different chunks) resolves the SAME chunk every call (numpy argmax breaks ties
by lowest index); and 1e-12 perturbations of a tile query flipped the locate 0/200 times (the tiles are
well-separated, so a query near a real tile has a clear winner -- not tie-sensitive). This is expected -- the
seams are built on build / gated_traverse / bundle / argmax, all deterministic given the seed -- but it is the
exact class of bug the bind_batch lesson warns about, so it is now asserted, not assumed.

Tests: +1 (1045 -> 1046). test_holographic_plan.py: test_chunking_seams_are_deterministic_and_not_tie_sensitive.
Audit-only; no faculty changed. Files: test_holographic_plan.py, README (counts).

---

## Chunked sequence memory: order queries exact past the single-bundle cap (chunking transfer backlog, item S3)

Third build off the chunking-transfer sweep, Plate/Olshausen's positional-encoding angle. SequenceMemory stores
order as position = rotation, bundled (element i is permute(atom, i+1), all summed into one vector), and answers
step / position_of / precedes / validate by un-rotating a position and reading it off. That single bundle caps
with length -- but FAR more gracefully than the directed-chain route did. Measured at dim 2048, vector-only
positional decode accuracy: ~100% at length 50, ~96% at 100, 69% at 200, 29% at 400, 15% at 800 (the route's
directed structure, by contrast, collapsed to one step at ~45 tiles -- iterative traversal compounds error,
direct positional read does not, so the positional encoding is the more robust of the two). add(..., chunk=K)
stores the sequence as positional blocks of <=K, each its own clean bundle, and routes a position query to the
one block it lives in (divmod(i, K)); measured chunked accuracy is 100% at EVERY length tested. Gain grows from
+0% (short) to +85% at N=800 -- load-bearing for long sequences, a pure no-op on short ones.

IMPORTANT, narrower than P1/X3: step (position -> element) cleans against the KEPT element list, so it is always
exact regardless of bundle quality -- the chunking benefit is NOT visible there. The win shows in the ORDER
queries that decode positions FROM the vector: precedes and position_of (and vector-only step against the full
vocab). Those are exactly SequenceMemory's distinctive value -- the recipe-vs-pile-of-steps relation -- so the
gain lands where it matters, but the framing has to be honest about which queries it helps. Same recurring
negative as the other chunkers: K must stay at/under the dim's reliable bundle length, or each block hits the
same cliff (default margin 14).

Shipped backward-compatibly: storage went from a 2-tuple (vector, elements) to a 3-tuple (repr, elements, chunk)
-- index 1 (the element list, read by app.py and the mind) is unchanged, chunk=0 is the original single-vector
path, and nothing external reads index 0. add gains chunk=0; a _probe(repr, chunk, i) helper centralizes the
block routing so step / position_of / precedes / validate each route through it. Wired through the mind as
learn_plan(name, steps, chunk=K) -- step_at / precedes / validate_plan are automatically chunk-aware; verified a
200-step protocol exact through the mind where a single bundle slips (tour: 200-step plan precedes 22/33 single
-> 33/33 chunked).

Tests: +3 (1046 -> 1049). test_holographic_sequence.py: chunked storage keeps long-sequence order queries exact
and is a no-op on short ones; backward-compatible default storage shape. test_integration.py: learn_plan chunked
keeps a long protocol exact through the mind. Files: holographic_sequence.py, holographic_unified.py,
test_holographic_sequence.py, test_integration.py, tour.py, README (test-list entry + counts).

---

## Where chunking helps -- and where it doesn't: the S1 overlap-add negative (chunking transfer backlog, item S1)

S1 was the most seductive item in the chunking-transfer sweep: chunk_route's one-element boundary overlap looks
exactly like the phase vocoder's weighted overlap-add, so processing a long signal as overlapping windowed
chunks "should" be the same chunk-and-re-anchor win. Prototyped on the FPE substrate (VectorFunctionEncoder: a
continuous function f is the bundle f = sum_i y_i encode(x_i), read by an inner product = kernel sum). It is a
clean MEASURED NEGATIVE -- chunking is not just a no-op, it is HARMFUL.

Measured (raw inner-product readout, shape correlation vs the noise-free designed-kernel sum): a SINGLE bundle
reconstructs the function with corr ~1.0 at every domain length tested -- N=120 (0.91), 400 (1.00), 800 (1.00),
1500 (1.00) -- while hard-cut chunking and proper Hann overlap-add both sit at corr ~0. The reconstruction does
NOT degrade with domain length, so there is no capacity problem for chunking to solve, and breaking the global
kernel sum into windows only introduces boundary-incomplete neighbourhoods and per-window normalisation error.

WHY the rhyme fails -- and the principle it buys. FPE codes are shift-invariant powers of ONE base, so
<encode(q), encode(x_i)> is the SAME kernel for every pair at a given distance: the finite-dimension error is a
DETERMINISTIC sidelobe of that kernel, not a √N pile of independent random noise. And the readout <query, sum_i
w_i encode(x_i)> = sum_i w_i <query, encode(x_i)> distributes over the superposition EXACTLY (linearity). So
the kernel sum is computed exactly, the kernel decay localises it, and a longer domain changes nothing about the
local readout. Contrast the route / sequence / program: there the task is to DECODE a specific item back out of
a superposition by cleanup, and every other item's crosstalk eats into that recovery -- which is precisely what
caps, and precisely what chunking bounds. The sharpened rule:

    Chunking helps DECODE-VIA-CLEANUP -- recover/identify a SPECIFIC item from a superposition, where the other
    items' crosstalk caps recovery (routes, sequences, programs: plan_route, chunk_route, run_chunked, chunked
    SequenceMemory all live here).
    Chunking does NOT help LINEAR-FUNCTIONAL EVALUATION -- evaluate <query, superposition> (a kernel-density /
    function query), which is exact by linearity regardless of how many terms are bundled (FPE function readout
    lives here).

This reconciles cleanly with the pre-existing FPE capacity cliff (test_capacity_cliff_is_a_kept_negative): that
cliff measures absolute DETECTION separation (is THIS point placed vs empty, cosine-normalised), which decays as
the bundle norm grows with K -- a detection/decode behaviour. The S1 metric is relative SHAPE fidelity, which is
preserved. Both true; they are different behaviours of the same bundle, and the decode-vs-evaluate split is what
separates them. The same caution likely weakens S2 (overlapping-block denoising of a long signal) -- aggregate
block denoising is closer to evaluation than to per-item decode -- and is flagged in the backlog to be checked
before any build.

Tests: +1 (1049 -> 1050). test_holographic_fpe.py: test_function_shape_reconstruction_does_not_cap_so_overlap_
add_chunking_is_a_no_op (pins the corr ~1.0 evidence). No faculty built -- this is a recorded negative that
sharpens the theory of the whole chunking arc. Files: test_holographic_fpe.py, holostuff_chunking_transfer_
backlog.md (S1 marked negative + build order updated), README (count).

---

## Does chunking help text / image generation? Tiled splat scenes (chunking transfer backlog, item X2)

Asked whether the chunking arc transfers to generation. Settled it with the decode-vs-evaluate principle (from
the S1 negative) plus a code audit and a measurement, and the answer splits cleanly.

TEXT generation: NO. The generators (generate -> n-gram, generate_structured -> steered_generate) condition each
step on a BOUNDED context -- the predictor's n-gram order plus the last `lookback` tokens -- so a longer
generation never piles into a capping superposition. There is no decode-from-a-long-bundle cliff for chunking to
fix; generating length 30 vs 300 uses the identical per-step context. (Long generations can still drift or loop,
but that is bounded MEMORY, not a capacity cliff -- fixing it would need a hierarchical long-range summary, a
different and speculative mechanism, not this lesson.)

IMAGE: YES, in the content-addressable splat SCENE. splat_bundle encodes a scene as grid*grid bind(cell_role,
occupancy_level) terms in ONE hypervector, and recall_region reads a cell back by unbind + cleanup -- a textbook
decode-via-cleanup readout. So as the grid gets finer the bundle's own crosstalk grows and region recall caps:
measured at dim 4096, accuracy is ~100% at grid 8, 98% at 16, 88% at 24, 75% at 32. This is the SAME cap chunking
bounds for routes / sequences / programs, and the chunk here is a TILE. splat_bundle_tiled routes each cell to a
tile bundle (floor-divide the grid index by `tile`), so a tile holds at most tile*tile bindings no matter how
fine the TOTAL grid is -- the per-bundle load is fixed and recall holds ~100% at any resolution (measured 75% ->
100% at grid 32). Costs one hypervector per tile (proportional storage, the price of exceeding a single vector's
capacity -- the same trade chunk_route and chunked SequenceMemory make). This is the splat side of backlog item
X2, and it confirms the principle predicts WHERE the lesson lands: image RECALL/representation that decodes from
a superposition (yes), not text generation with bounded context (no), and not the FPE function readout that is
linear-exact (S1, no).

NOTE the precise complement already in the box: SplatArchive.region stores splats as an EXPLICIT list and is
exact per-splat -- so it never had this cap. The tiled bundle is the COMPACT, content-addressable, coarse-but-
robust path; tiling is what lets it stay accurate at fine resolution.

Shipped: holographic_splat.splat_bundle_tiled / recall_region_tiled (global cell roles, so recall needs no
remapping; routes a cell to its tile bundle and reuses recall_region's unbind+cleanup). Wired onto UnifiedMind
as splat_scene(field, grid, tile, levels, k) -> tiled scene and splat_region(scene, cell) -> occupancy.
Determinism-clean (tile bundles bit-identical run-to-run) and empty-tile safe.

Tests: +4 (1050 -> 1054). test_holographic_splat.py: the single-bundle cap (negative, acc<0.85 at grid 32),
the tiled fix (acc>0.99 at grid 32), determinism + empty safety. test_integration.py: splat_scene region recall
exact at fine resolution through the mind. Files: holographic_splat.py, holographic_unified.py,
test_holographic_splat.py, test_integration.py, tour.py, README, holostuff_chunking_transfer_backlog.md (X2).


## StructuredIndex -- the shared content-address index (and where Merkle already lives)

Three places were independently growing the same primitive: "given a pile of vectors, find the one this query
points at, without scanning all of them." RouteIndex (a flat two-level summary scan), a chunked sequence, and
the content store (which already grows a per-bucket HoloForest). The request was to stop duplicating it -- one
abstraction the rest draw from, so a future caller does not re-hit the same limit by re-inventing the lookup.

StructuredIndex (holographic_tree.py) is that, as a thin payload-carrying wrapper over the HoloForest RP-tree:
build(keys, payloads); locate() is the sub-linear path with a free cross-tree agreement/abstention signal;
locate_k() is sub-linear top-k; locate_exact() is the flat guaranteed-nearest. The payload is the point -- you
file vectors and get back a LABEL (a URI for the store, (chunk, step) for a route, a step index for a sequence),
not a row number. Wired as the mind faculty structured_index(keys, payloads).

TWO RULES ARE BAKED INTO IT, both measured the hard way in the design probe so a future caller meets them as
documentation rather than rediscovering them as "limits":
  1. Key on the ITEMS THEMSELVES. A hyperplane tree only routes a query to the right leaf when query ~= key.
     Filing items under a bundle-SUMMARY the query is weakly correlated with mis-routes them -- a tile has cosine
     only ~0.27 to its chunk summary, which an exhaustive argmax still resolves but a greedy tree descent does
     not (measured: locating a route by chunk-summary THROUGH a tree collapsed to ~1/200, while the same tree
     over the tiles themselves routed home). This is the decode-vs-evaluate constraint wearing a routing hat.
  2. Never store the index as a BUNDLE. Superposing the keys and recovering one by unbind+cleanup is decode-via-
     cleanup and caps with set size (measured: 200 -> 127 -> 15 recovered as the set grows). The index must be a
     navigable STRUCTURE (this tree) or, below the crossover, an explicit scanned list -- never a superposition.

HONEST CROSSOVER (kept, not hidden): the forest carries a large fixed constant (n_trees x leaf_size x beam
candidates), so a flat scan WINS until the set is in the low thousands -- measured ~30 vs ~470 comparisons for a
few-hundred-chunk route, the forest only pulling ahead past ~6000 items. So locate_exact is not a fallback, it is
the correct call below the crossover; RouteIndex's flat scan is therefore this index at its small-n operating
point (exact, cheap), and the content store's HoloForest is it at its at-scale one. The abstraction unifies them
without a regressive refactor -- the working flat path stays, and new callers reach for the one primitive.

MERKLE (the question that prompted checking, and the payoff of checking): holostuff ALREADY has a holographic
Merkle tree -- holographic_verify.CompositionTree, mind faculty verify_store. leaf = bind(pos, item), node =
bundle(children), root = the commitment; detect by rebuilding and comparing the root, localise a changed item in
<= log2(n) composite comparisons. Its kept negative is the right caveat: the root is LINEAR, so collisions exist
and a key-aware adversary can cancel a change by deconvolution -- evidence of ACCIDENTAL corruption, NOT
cryptographic tamper-proofing. The clean separation: StructuredIndex is for LOOKUP (recover an item), the Merkle
tree is for INTEGRITY (has anything changed, which one). And they sit on opposite sides of the decode-vs-evaluate
line -- comparing two whole composites by cosine is an EVALUATION, which does not cap, which is exactly why the
Merkle tree's detection survives at any store size while a lookup that must DECODE an item does not.

Tests: +7 (1054 -> 1061). test_holographic_tree.py: sub-linear content routing, payload labels, exact+flat scan,
ranked top-k, the agreement signal, payload/key mismatch guard. test_integration.py: one structured_index faculty
serving both content (payload=URI-like) and route (payload=(chunk,step)) lookups. Files: holographic_tree.py,
holographic_unified.py, test_holographic_tree.py, test_integration.py, tour.py, README.


CHUNKING-TRANSFER, THE LAST THREE (X1 tiled scene factorization, C1 chunk dedup, R1 re-anchored rollout):
the sweep that asked "is the recent route lesson a general capacity primitive?" closes here with two builds and
one kept negative, all from the same decode-vs-evaluate test -- does the operation DECODE a specific item from a
superposition (helped by chunking/tiling/re-anchoring) or merely EVALUATE a linear form (not helped, because a
linear map has no capacity cliff to relieve).

X1 (WIN). A multi-object scene is a superposition the resonator must FACTOR -- a decode-via-cleanup, so it has the
capacity problem tiling addresses, and unlike S1 the S1 negative does not pre-empt it. Measured: at dim 1024 the
whole-scene factorization caps at ~5 objects and collapses past it (~30% recovery at 15 objects across 8 seeds);
splitting the objects into spatial tiles of <= cap each, factoring every sub-scene, and merging lifts recovery to
~93%. The tile size plays the chunk's role exactly: it must stay at/under the per-tile cap or each tile re-hits
the same cliff (tiles of 5 at dim 1024 still leave a little within-tile crosstalk -- 10-15/15 per seed -- which a
smaller tile or higher dim removes). This is chunk_route's move and splat_bundle_tiled's move on the resonator:
beat a fixed structure's capacity with composition, at the honest price of keeping the tiles. SceneCoder.
factor_scene_tiled / mind decompose_scene_tiled.

C1 (WIN, with its honest bound). A long route that REVISITS the same corridor, or a program with repeated motifs,
stores the same compact chunk vector many times. Content-address the store -- keep each unique chunk once and
replace repeats with a reference -- and storage shrinks by EXACTLY the repetition ratio: measured 65% on a
17-corridor loop with 6 distinct chunks, 0% on a no-repeat control (dedup can only save what actually repeats),
references rebuilding the original sequence bit-for-bit. This is the storage twin of StructuredIndex: that finds
an item BY content, this stores items BY content so identical ones coalesce -- and comparing whole chunk vectors
by cosine is an EVALUATION (not a decode), so two genuinely distinct chunks never collide at high dim, no cap.
holographic_plan.dedup_chunks / mind dedup_chunks.

R1 (KEPT NEGATIVE, joining S1). Re-anchoring a learned propagator's long rollout onto the consolidation manifold
does NOT help -- and the reason is the same line. A route's per-hop cleanup ACCUMULATES crosstalk, so re-anchoring
rescues it; a linear propagator rollout is repeated application of one operator, an EVALUATION. Measured on a
trajectory in the propagator's exact model class (per-frequency phase advance == circular convolution, the audio
sweet spot): the free rollout's drift is ~0 over 50 steps -- the operator TRACKS the trajectory, there is no drift
to fix -- and projecting the state onto the rank-r training-state manifold every few steps only makes it worse
(mean drift 0.0001 -> 0.5+), because the manifold of training states is a SUBSET of where the true trajectory goes
and re-projection discards valid forward signal. On a trajectory OUTSIDE the model class the prediction is wrong
within the manifold (a phase error), which manifold projection cannot fix either. So re-anchoring helps
decode-via-cleanup chains (routes, sequences, programs), never a linear-operator rollout. Pinned in
test_holographic_dynamics.py so nobody "fixes" a non-problem.

The whole 13-item sweep is now resolved: P1/X3/S3/C2 shipped, X2/X4 shipped (X4, the multi-terminal Tero "Tokyo
rail" network design, was already on disk), S1/R1 kept negatives, X1/C1 shipped here. The decode-vs-evaluate line
predicted every outcome: chunking/tiling/re-anchoring helps wherever an item must be DECODED from a superposition,
and is inert (or harmful) wherever the query is a linear EVALUATION with no capacity cliff.

Tests: +6 (1061 -> 1067). test_holographic_scene.py: tiled scene factorization beats the capped whole scene across
seeds. test_holographic_plan.py: dedup saves at the repetition ratio + exact rebuild, and saves nothing without
repetition. test_holographic_dynamics.py: re-anchoring a rollout does not help (R1 kept negative). test_integration
.py: the dedup_chunks and decompose_scene_tiled faculties through the mind. Files: holographic_scene.py,
holographic_plan.py, holographic_unified.py, test_holographic_scene.py, test_holographic_plan.py,
test_holographic_dynamics.py, test_integration.py, tour.py, README.


SCHEMA-GUIDED TYPED PLANS + descend (the structured branching output the planning work was missing):
a colleague who hit the route/planning case asked for a first-class Plan/PlanNode type (primary action, named
contingency branches, scope, confidence) plus a decode helper, so the bind/bundle tree is not re-derived by
hand each time. The audit found the ENCODING already proven (holographic_typed.encode_tree / StructureRecipe lay
down exactly this role-filler tree, bit-exact) and a NAME collision (holographic_plan.Plan is the corridor
planner's namedtuple). New module holographic_planshape.py ships the type and the decode.

THE LOAD-BEARING IDEA (why a user-provided SHAPE is the right design, not sugar): the shape IS the decode key.
Decoding a foreign vector with no shape is the resonator's blind, crosstalk-bounded parse (decompose_structure),
which caps -- the typed module says so itself, a StructureRecipe is a GENERATOR not a parser. With the shape
KNOWN, decode is a deterministic walk: unbind exactly the roles the shape names, clean against exactly the
codebooks it gives, recurse on exactly the field it marks recursive. No search. The decode-vs-evaluate line
again -- a known structure turns a decode-SEARCH into clean unbinds, so a 3- or 4-level plan round-trips every
action and scope EXACTLY where the blind parse would crater. Measured: per-node branch fan-out holds past 16 at
dim 1024 (the cap is the per-node bundle width, the HRR capacity bound -- a huge node nests, the same lesson).

WHAT SHIPPED. encode_record / decode_record: the GENERAL "bring your own shape" path -- a flat record of named
symbolic fields (a scientific decision record, a classified state) <-> one vector, decoded against per-field
codebooks. PlanNode + PlanShape + encode_plan / decode_plan: the concrete contingency tree and its schema-guided
round-trip. descend(vec, situation, shape): the walk to the branch matching the current situation -- the genuine
generalisation of the machine's IFMATCH from one gated instruction (fire iff cosine(state, x) >= tol) to a named
branch tree, matching the situation against branch condition keys by cosine and ABSTAINING (returning the node's
primary action -- "no contingency applies") when none clears a MEASURED noise floor. Mind faculties: plan_shape,
encode_plan, decode_plan, descend, encode_record, decode_record.

PANEL GROUNDING (seats + real published methods, no fabricated opinions). Plate (HRR): recursive role-filler
binding with per-level normalisation (bundle already does it); the per-node fan-out cap is his capacity bound,
kept as the negative. Olshausen (resonator networks): the resonator is the UNKNOWN-structure tool the schema
path AVOIDS -- it stays the fallback. Togelius (game AI): descend is a behavior-tree selector ("best child whose
condition fires, else fall through"); the decodable, readable tree is the explainability his field wants.
Cranmer (calibrated detection): the branch gate is a MEASURED noise floor and confidence is the MEASURED decode
cosine, not a magic threshold (a simple cousin of RecallNull; upgrade to the full null for a controlled
false-match RATE). Macklin (bit-exact tie-breaks): argmax ties broken deterministically, encode and descend
run-to-run identical (a determinism test pins it). Eno (the reframe): the output shape is not formatting, it is
the choice of which structure to impose, which determines what is recoverable -- so the shape is a first-class
decode key, optional (omit it and you are back to the resonator's discovery).

KEPT HONESTLY. confidence rides on the PlanNode OBJECT (builder metadata); it is NOT encoded into the vector,
because forcing a scalar into the bundle decodes lossily and would betray the honest-number rule -- decode
instead fills the returned node's confidence with the measured decode cosine. And schema-guided decode NEEDS the
schema: a truly foreign vector of unknown shape falls back to the capping resonator. This is explicitly "decode
a structure you have the shape for", which is the plan/protocol case -- exactly why providing the shape is the
right design, not a limitation of it.

Tests: +11 (1067 -> 1078). test_holographic_planshape.py: the module self-test, flat-record round-trip +
measured per-field confidence, plan round-trip exact, a deep tree held by schema guidance, descend walking to the
right branch, descend abstaining when no branch applies, descend matching a state VECTOR (and abstaining on an
unrelated one), and determinism of encode/descend/noise-floor. test_integration.py: the plan faculties
(encode_plan/decode_plan/descend) and the general record faculty through UnifiedMind. Files:
holographic_planshape.py, holographic_unified.py, test_holographic_planshape.py, test_integration.py, tour.py,
README.


GRAPH-SIGNAL DENOISING (reverse-transfer RT-III1 -- mesh smoothing mapped back onto the concept graph):
the DCC reverse-transfer sweep asked which 3-D operation is a special case of a general operation the stack
LACKS. Mesh smoothing (filter a signal -- vertex positions -- on the mesh graph) is one: holostuff is full of
graphs (the codebook similarity graph, the HoloForest, the store adjacency, the scene/sequence chains), and
`graph_memory` only does cosine k-means CLUSTERING, never a Laplacian or spectral FILTER. New module
holographic_graphsignal.py is that filter -- denoise/regularize a set of vectors over its own k-NN similarity
graph (non-local means on the concept graph).

THE TAUBIN POINT. A naive graph-Laplacian smooth denoises but SHRINKS: every step pulls mass toward the graph
mean (its transfer (1-lam*k)^n < 1 for every graph frequency k>0, DC included), so the whole codebook collapses.
Taubin's lam|mu pair (Taubin 1995) alternates a shrink step (lam>0) with an un-shrink step (mu<0, |mu|>lam) so
the combined transfer (1-lam*k)(1-mu*k) is ~1 at low frequency (DC preserved -> no shrink) and <1 at high
frequency (noise removed) -- the classic no-shrink low-pass.

MEASURED ON A CURVED HIGH-RANK MANIFOLD (the regime where a LOCAL graph beats a GLOBAL-linear method), with the
kept negative. (1) Taubin robustly AVOIDS the shrink -- mean norm stays ~0.88-0.98 (toward the clean norm as
noise rises) where the naive Laplacian always collapses to ~0.41-0.54. Unambiguous. (2) Graph filtering BEATS
per-vector denoising (consolidation onto the global low-rank subspace) ONLY at HIGH noise: rel-noise 1.2 ->
Taubin quality 0.865 vs consolidate 0.837, winning 6/6 seeds; moderate noise ties; LOW noise (rel-0.5) ->
consolidation wins 0/6 (0.968 vs 0.953) and the graph filter OVER-SMOOTHS. KEPT NEGATIVE: the local k-NN graph
helps precisely when noise is high enough to corrupt the global linear subspace while the curved manifold's
local neighbourhoods survive; when the signal is already clean, the global linear denoiser is better and the
graph filter only blurs it. The decode-vs-evaluate cousin: the graph filter pays when the structure is genuinely
non-linear/local, not when a few global components already capture it.

The doc's flagged failure mode (building the k-NN graph is O(n^2)) is handled by REUSING the HoloForest's
sub-linear `recall_k` for the neighbours (its own docstring already names it "the neighbour-search step that
non-local-means denoising needs"; `graph_denoise(..., sublinear=True)`), and by holding the graph as sparse
neighbour lists so the filter step is O(n*k), not a dense n*n matvec. Mind faculty: graph_denoise(vectors, k,
method='taubin'|'laplacian', sublinear). PANEL SEATS: Milanfar ("A Tour of Modern Image Filtering" links
denoisers to graph Laplacians) + Taubin (the no-shrink lam|mu surface filter).

This is RT-III1, the panel's #1 reverse-transfer pick (cleanest bar, reuses the HoloForest). The full DCC
reverse-transfer backlog (11 items, Groups I-VI) is captured as Part II of holostuff_crosscutting_backlog.md;
the remaining ◆ picks in order are RT-II1 (nonlinear manifold chart), RT-IV1 (steering/anisotropic kernel), and
RT-I1 (operator-limit / spectral-iteration).

Tests: +6 (1078 -> 1084). test_holographic_graphsignal.py: the module self-test, Taubin denoises + avoids the
shrink (norm kept where naive collapses), graph beats per-vector at high noise across seeds, the low-noise kept
negative (per-vector wins), and knn_graph determinism + row-normalisation. test_integration.py: the graph_denoise
faculty beating per-vector at high noise through the mind. Files: holographic_graphsignal.py,
holographic_unified.py, test_holographic_graphsignal.py, test_integration.py, tour.py, README,
holostuff_crosscutting_backlog.md.


NONLINEAR MANIFOLD CHART (reverse-transfer RT-II1 -- UV unwrapping mapped back onto the concept manifold):
the second DCC reverse-transfer pick. UV unwrapping (LSCM/ARAP/Tutte) is the least-holostuff item on the
backlog and secretly the most general -- distortion-minimizing FLATTENING of a curved 2-manifold to a low-D
chart, the embedding problem the whole stack faces and only solved LINEARLY by `consolidation` (an SVD). A
LINEAR projection FOLDS a curved manifold: points far apart ALONG the manifold land on top of each other in the
2-D chart. New module holographic_chart.py is the nonlinear extension.

TWO METHODS, both pure NumPy, both reuse RT-III1's k-NN graph. (1) ISOMAP (Tenenbaum-de Silva-Langford 2000) --
the primary, geodesic-PRESERVING chart: approximate along-manifold distance by shortest paths on the k-NN graph
(Floyd-Warshall), then classical-MDS to 2-D; this UNROLLS the curve. (2) LAPLACIAN EIGENMAPS (Belkin-Niyogi
2003) -- the graph-spectral cousin: the bottom non-trivial eigenvectors of the SAME graph Laplacian whose
high-frequency components RT-III1's Taubin filter REMOVES (so the two reverse-transfer items are one operator
used two ways). It preserves LOCAL neighbourhood structure but distorts GLOBAL distances -- kept as the honest
secondary, not the default.

MEASURED on a swiss roll lifted into D=256 (the canonical curved 2-manifold whose ambient variance defeats a
linear projection). Isomap BEATS linear SVD/consolidation robustly: geodesic-distance correlation ~0.83 vs
~0.76 and class separation (4 bands adjacent on the manifold but FOLDED by SVD) ~0.86 vs ~0.76, winning 5/5
seeds on both (on a clean roll the geo-corr gap is wider, 0.95 vs 0.52). Laplacian Eigenmaps preserves local
neighbourhoods but its global geo-corr trails SVD here -- the kept nuance.

FAILURE MODE the doc flagged (honest, not a bug): a chart assumes disk topology (genus 0). A CLOSED manifold (a
torus, genus 1) cannot flatten to a plane without a SEAM -- cut it first, and the `topology` faculty finds the
genus that says where. A 1-manifold ring charts to a circle with no cut; a genus>0 surface needs the cut. High
curvature also makes some distortion unavoidable (LSCM's own limit). The geodesic step is Floyd-Warshall O(N^3)
-- fine for a few hundred points; subsample to landmarks or reuse the HoloForest neighbours (RT-III1's O(N^2)
graph-build fix) for more. Determinism: the eigenvector sign is pinned (largest-|entry| made positive) so the
chart is bit-stable -- the sign/order tie the determinism fence warns about. Mind faculty:
manifold_chart(vectors, dim, method='isomap'|'spectral', k, sublinear). PANEL SEATS: Olshausen (representation
geometry) + the consolidation thread + Tutte (graph drawing) + Lévy/Liu (LSCM/ARAP).

Two of the four DCC reverse-transfer ◆ items now shipped (RT-III1 graph-Laplacian, RT-II1 manifold chart). The
remaining ◆ picks in order: RT-IV1 (steering/anisotropic kernel), RT-I1 (operator-limit / spectral-iteration).
Full backlog: Part II of holostuff_crosscutting_backlog.md.

Tests: +7 (1084 -> 1091). test_holographic_chart.py: the module self-test, Isomap beats SVD on geodesic
fidelity across seeds, Isomap separates classes the linear chart folds, the chart is deterministic, the spectral
method runs, and geodesics stay finite when the raw k-NN graph starts disconnected (connectivity repair).
test_integration.py: the manifold_chart faculty beating linear SVD on a curved manifold through the mind. Files:
holographic_chart.py, holographic_unified.py, test_holographic_chart.py, test_integration.py, tour.py, README,
holostuff_crosscutting_backlog.md.


THE DETERMINISM CONTRACT (ISA-1 -- the first item of the VSA ISA backlog, Part III of the cross-cutting
backlog): the "learn from assembly" lens found that holostuff has ALREADY built a VSA instruction-set
architecture (the kernel is the instruction set; HoloMachine the assembler+interpreter; StructureRecipe the IR;
the resonator the disassembler), and an ISA is durable only if the EXACT OBSERVABLE semantics of its base
instructions are a frozen contract while implementations vary underneath. The audit found the cost of NOT having
that contract, paid right now: the determinism/tie-break behaviour was specified FOUR different ways --
cleanup's implicit numpy argmax (ties->lowest index, written nowhere), spectral's "largest-magnitude component
positive" (explicitly citing "the same bit-exact-tie class as the bind_batch bug"), flow's private weighted
Laplacian, and -- the fourth, added two builds ago during RT-II1 -- chart's private `_fix_signs` reinventing the
same sign rule. Same bug class, re-litigated four times, code duplication as the price.

WHAT SHIPPED. (1) ISA.md -- the written contract: per-instruction observable semantics (bind/unbind/bundle/
permute/cosine/involution/random_vector + the cleanup decision), each tagged EXACT (a decision / exact reindex,
pinned bit-for-bit) or TOL (a continuous value, conformant within numeric tolerance), with the real edge cases
grounded from the kernel (zero-sum bundle -> zero vector; zero-norm cosine -> 0.0; permute exact+invertible;
involution exactly self-inverse). The ARCHITECTURE/MICROARCHITECTURE boundary stated explicitly: the observable
DECISION is architecture (pinned); how the continuous numbers are computed (FFT vs direct, batched vs looped) is
microarchitecture (free within tolerance, provided the decision it feeds is unchanged) -- bind_batch is exactly
such a variant, which is why it could be bit-exact to 1e-12 yet flip a trajectory through an unpinned argmax
tie. (2) holographic_determinism.py -- the executable embodiment of the ONE determinism rule: `fix_eigvec_signs`
(the reconciled sign convention) and `argmax_tiebreak` (names cleanup's lowest-index convention so it is
citable). (3) THE DE-SILO: spectral.sign_fix and chart._fix_signs are now thin delegates to the shared utility,
BIT-EXACT (copy=False preserves spectral's in-place behaviour; the 19 spectral/chart tests still pass unchanged)
-- the fourth scattered copy is gone, the convention has one home.

THE ANTICIPATED NEGATIVE, kept: a contract must not over-specify. ISA.md freezes only the OBSERVABLE semantics
callers depend on (the argmax decision, unbind's approximate-recovery guarantee, the edge-case returns), NOT the
FFT's internal rounding, the bits of a reduction no decision observes, or the basis within a degenerate
eigenspace -- pinning those would mistake incidental float behaviour for architecture and block the very
optimization (the bind_batch speed-up) the contract exists to make safe. SEATS: Cranmer (reproducible-analysis /
frozen re-runnable contract, RECAST) + Macklin (the bit-exact tie-break lesson; this also answers his standing
determinism-audit request). NEXT in the spine: ISA-2 (the conformance suite + reference implementations + a
regression for the bind_batch class itself -- the contract's teeth), then ISA-3 (the extension discipline).

Tests: +7 (1091 -> 1098). test_holographic_determinism.py: the sign rule is deterministic / sign-invariant
(V and -V -> the same fixed basis) / idempotent, the copy flag preserves each call site's behaviour, the argmax
tie-break picks the lowest index, and the de-silo is bit-exact (spectral.sign_fix and chart._fix_signs now equal
the shared rule). Files: holographic_determinism.py, ISA.md, holographic_spectral.py, holographic_chart.py,
test_holographic_determinism.py, tour.py, README, holostuff_crosscutting_backlog.md.


THE CONFORMANCE SUITE (ISA-2 -- the second item of the VSA ISA spine; the teeth for ISA.md): a contract with no
enforcement is just prose, so this is the enforcement. For each base instruction there is now a DEFINITIONAL
reference implementation (holographic_reference.py) -- the simplest, obviously-correct version: `ref_bind` is a
direct O(D^2) circular convolution (NOT an FFT), `ref_involution` the explicit reversal, `ref_permute` the
explicit index roll -- verified against the production kernel to MACHINE EPSILON (bind vs direct conv: 1e-16;
involution/permute: exact). The FFT `bind` genuinely IS circular convolution, now provable by a slow reference.

THE TOL/EXACT SPLIT, made callable (the ISA-1 boundary enforced). `value_conformant` (continuous outputs match
within numeric tolerance) for bind/unbind/bundle/cosine; `exact_conformant` (bit-for-bit) for involution/permute;
`decision_conformant` (same cleanup pick under argmax_tiebreak) for the observable decision. `run_conformance`
checks every production op against its reference and returns {op: passed/class/max_diff}; exposed as the mind
faculty `conformance_report()` (the conformance harness made first-class, beside calibration_report). A
vectorized op is "conformant" iff it passes here -- which is exactly what makes the §7 vectorization sweep safe
to pursue.

THE CENTERPIECE -- the bind_batch-class regression, caught BY CONSTRUCTION. The bug was a value-conformant
change (bit-exact to 1e-12) that flipped a creature's trajectory through an unpinned argmax tie. The suite
catches the whole class because it checks the DECISION separately and exactly: a similarity vector perturbed by
a SUB-TOLERANCE amount (1e-12) passes `value_conformant` but fails `decision_conformant` -- a value-only suite
would accept it, the contract's decision check rejects it. And the literal mechanism is pinned too: the same
numbers summed in two orders (x = [1e16, 1, -1e16, -1]) give -1.0 vs 0.0, and on a near-tie that flips the
argmax. GOLDEN VECTORS are the hand-verifiable convolution identities (bind(a, delta0)==a; bind(a, delta_k)==
roll(a,k); bind commutative; the round-trip recovers b as the cleanup WINNER -- approximate because involution
is an exact inverse only for unitary vectors, not random ones, so the guarantee is the decision, not a 1e-9
match) -- goldens that cannot rot the way frozen float arrays would. EDGE CASES pinned: zero-sum bundle -> zero
vector; zero-norm cosine -> 0.0.

SEATS: Cranmer (the conformance/measurement discipline; golden tests as the spec made executable). NEXT in the
spine: ISA-3 (the extension discipline -- document Clifford/tensor/FPE as named, opt-in ISA extensions, base
kernel stays minimal). NOTE (honest): the doc suggested ISA-2 lets flow's Laplacian be de-duplicated, but the
audit shows flow's `_weighted_laplacian` is a DIFFERENT construction (edge-conductance for the Tero solve) than
the k-NN similarity Laplacian in graphsignal/chart -- not a duplicate to merge, so that de-silo is not forced.

Tests: +9 (1098 -> 1107). test_isa_conformance.py: all base ops conform to their references (TOL/EXACT), the
convolution-identity golden vectors, exact ops bit-for-bit + self-inverse/invertible, the zero-vector edges, and
the bind_batch-class regression (a value-conformant change that flips a decision is caught; a summation reorder
flips an argmax). test_integration.py: the conformance_report faculty passing for the live kernel. Files:
holographic_reference.py, ISA.md (referenced), holographic_unified.py, test_isa_conformance.py,
test_integration.py, tour.py, README, holostuff_crosscutting_backlog.md.


THE GOVERNED EXTENSIONS (ISA-3 -- the third item of the VSA ISA spine; closes the Tier 0-1 do-now block): real
instruction sets grow as base + extensions, never by bloating the base. The audit confirmed holostuff already
does this by instinct (the Clifford module's docstring states the rule), so ISA-3 makes it POLICY:
ISA_EXTENSIONS.md is the VSA analog of x86 + SSE/AVX/AES-NI -- a minimal base kernel plus named, opt-in bind-mode
EXTENSIONS, each justified by a MEASURED regime win over base `bind`.

THE BASE/EXTENSION BOUNDARY (the principle, applied to the debatable `permute` case): BASE = what (almost) every
faculty uses, the holographic_ai.py kernel; EXTENSION = regime-specific, a separate opt-in module. By that rule
the base instruction set is frozen as random_vector / bind / unbind / bundle / permute / cosine / involution /
the cleanup decision (full semantics in ISA.md) -- and `permute` is BASE (it lives in the kernel, used across
the sequence/creature/structure faculties for order). The three extensions are NOT base.

THE THREE EXTENSIONS, each with a regime win MEASURED FRESH this session: (1) Clifford-bind (the geometric
product, holographic_clifford.py) -- regime 3-D rotations; win: rotation composition is EXACT and is one product
(the geometric product of two rotors IS the composed rotor, error 1.1e-16), and non-commutative so it captures
order base convolution cannot; cost: 2^d-dimensional, rules it out as a general substrate. (2) FPE/VFA
(holographic_fpe.py) -- regime continuous/spatial values; win: a DESIGNED Bochner kernel makes nearby values
similar (smooth monotone falloff 1.0->0.9->0.66->0.41->0.22->0.11->0.04 over offset 0..3) where independent
random atoms have NO continuity (all ~0 off-diagonal); cost: it is an encoder, not a general bind, and the
kernel is a design choice. (3) Tensor-product bind (holographic_tensor.py) -- regime high capacity at the cost
of D^2 storage; win: at a load that overloads HRR, tensor recall 0.87 vs HRR 0.28 (D=32, 12 pairs); cost: D^2
numbers vs D, and a generic full-rank binding cannot be MPS-compressed without losing recall (the frontier is
HRR(D) < tensor-train(~2rD) < full tensor(D^2)).

THE NEW-EXTENSION PROPOSAL TEMPLATE (the earning-its-place bar, baked into the doc): name & module (base
untouched -- conformance_report still passes); regime (and where NOT to use it); the base-bind baseline it must
beat; the MEASURED win on real data with the regime stated; its own conformance test; cost & kept negative
stated as loudly as the win. No measured regime win -> not an extension; the base stays minimal. SEATS:
Stoudenmire (the tensor/capacity extension) + Plate (what stays in the minimal base ISA). The honesty/anticipated
negative (the boundary is debatable) is resolved by picking the principle and applying it consistently rather
than case-by-case.

Tiers 0-1 of the ISA spine are now complete (ISA-1 contract, ISA-2 conformance teeth, ISA-3 extension
discipline) -- the do-now block that makes the whole engine safer to optimize and systematizes a design
holostuff already followed by instinct. NEXT: ISA-4 (accumulator -> a small register file; register pressure is
literally a capacity-cliff question), the first machine-model item.

Tests: +4 (1107 -> 1111). test_isa_extensions.py: Clifford exact rotation composition (length-preserving +
invertible), FPE's designed kernel is continuous and beats random atoms, tensor-bind's higher recall at an
overloading load, and the base kernel stays minimal/unchanged when the extensions are imported. Files:
ISA_EXTENSIONS.md, test_isa_extensions.py, tour.py, README, holostuff_crosscutting_backlog.md.


THE REGISTER FILE (ISA-4 -- the first machine-model item of the ISA spine): HoloMachine ran everything through
ONE accumulator (ACC). ISA-4 grows it to a handful of named slots (REGISTERS R0..R7) with two new opcodes,
STORE r (ACC -> slot) and RECALL r (slot -> ACC). Backward-compatible: the two opcodes are additive, existing
programs are untouched (all 14 prior machine tests pass), and the operand is cleaned against a new reg_atoms
codebook exactly like REPEAT's counts or APPLY's faculty names.

THE DESIGN HINGES ON THE KEPT NEGATIVE (the lovely VSA-native one): how is the register file held? Two options,
and the measurement decides. (A) SEPARATE NAMED SLOTS (a Python dict in run()): reads are EXACT (the value is
returned verbatim, cosine 1.000, bit-for-bit via np.array_equal), no crosstalk, no capacity limit. (B) ONE
BUNDLE (the machine's existing "disk" pattern, bundle of bind(reg_role_i, value_i) with a distinct role per
register): the slots share the crosstalk budget, so readback degrades as registers pile in -- "register pressure"
is LITERAL, the capacity cliff applied to the register file. MEASURED: a bundled file reads back perfectly to ~16
registers at dim 1024, then degrades (32 -> 0.99, 64 -> 0.93/0.91); at dim 4096 it holds 64. So register count is
a CAPACITY QUESTION for the bundled rep, not a free choice. The machine therefore holds slots SEPARATELY -- the
measurement is the justification for the design, exactly the project's pattern (ship the working design, keep the
negative that rules out the alternative on record).

THE BAR (registers save re-derivation): a value needed again after ACC moves on costs a full re-derivation
without registers but ONE RECALL with them. For a k-instruction intermediate M, registers replace k instructions
with 1 (plus 1 STORE) -- fewer instructions, and the recalled M is EXACT regardless of how it was produced (a
re-derivation that ran a lossy APPLY step would not even reproduce M; RECALL always does). SEATS: Plate (the
clean HRR slot/role algebra the register file is built from) + Eno (a small, well-chosen set of named slots as a
generative constraint -- a handful, not unlimited).

NEXT in the spine: ISA-5 (a documented calling convention + a permute-stack for recursion -- and the kept
negative there is the same family: stack depth is bounded by crosstalk, like the B8 iterated-decode cliff).

Tests: +6 (1111 -> 1117). test_isa_registers.py: exact read, bit-for-bit recall of an intermediate, the
re-derivation-instruction saving, 8 independent slots, and the bundled-file capacity-cliff kept negative; plus a
mind-level integration test (exact recall through the mind's machine + a register-free program still runs).
Files: holographic_machine.py, test_isa_registers.py, test_integration.py, tour.py, README,
holostuff_crosscutting_backlog.md.


THE CALLING CONVENTION + PERMUTE-STACK (ISA-5 -- the second machine-model item of the ISA spine): two parts, a
documented ABI and a substrate stack. (1) THE CALLING CONVENTION (ISA.md's new ABI section): CALL f runs library
function f as an ACC->ACC transform -- ACC is the argument in and the return value out, and the whole function
library obeys it. The preservation guarantee is the nice part: registers (R0..R7) and the permute-stack are
FRAME-LOCAL -- each CALL runs in its own run() frame with a fresh register file and a fresh stack, so a callee
CANNOT corrupt the caller's registers or stack (measured: a callee overwriting its R0 leaves the caller's R0
bit-identical, cosine 1.000). In ABI terms every register is callee-saved by construction -- the caller spills
nothing to keep a value across a CALL. Recursion (self-CALL with an IFMATCH base case) runs under the existing
depth guard (8).

(2) THE PERMUTE-STACK (PUSH/POP opcodes + module-level stack_push/stack_pop): a LIFO in the vector substrate.
PUSH is permute+bundle (shift the existing items one level deeper, drop ACC on top); POP is cleanup +
inverse-permute (the top is the only un-permuted term -- clean it out, peel it off, un-shift the rest). It is the
explicit-stack form of recursion -- e.g. reversing a sequence by pushing every element then popping pops them in
reverse, the textbook stack-replaces-recursion pattern -- and it runs correctly through the machine (push a,b,c,d
-> first POP yields 'd').

THE KEPT NEGATIVE, measured (the spine's recurring lesson, a third time): the permute-stack is a HOLOGRAPHIC
stack -- every level rides one bundle, so depth is bounded by crosstalk exactly like the B8 iterated-decode
cliff. SAFE DEPTH ~4-8 items at dim 1024 (LIFO recovery 1.00 to depth 4, ~0.92 at 8, ~0.48 by 16; a little
deeper at dim 4096). So the permute-stack is for shallow nesting of cleanup-able items; for arbitrary
intermediates at any depth, use the registers (exact, frame-local). This is the SAME capacity lesson the bundled
register file taught (ISA-4) and the bundled disk before it: superposition buys composability and pays in a
crosstalk cliff -- measure it, keep the exact path for what must be exact. SEATS: Plate (the HRR role/permute
algebra the convention and stack are built from) + the machine thread.

NEXT in the spine (Tier 3): ISA-6 (a macro layer -- parameterized recipe/procedure templates over the assembly,
the procedure abstraction is already halfway there). Then ISA-7 (a small HLL, research) and ISA-8
(reversible/quantum bind, the frontier).

Tests: +6 (1117 -> 1123). test_isa_callstack.py: frame-local registers (the ABI guarantee) and frame-local stack,
the permute-stack LIFO primitive, reverse-via-stack through the machine (the bar), and the depth-cliff kept
negative; plus a mind-level integration test (reverse-via-stack + frame-local registers through the mind's
machine). Files: holographic_machine.py, ISA.md, test_isa_callstack.py, test_integration.py, tour.py, README,
holostuff_crosscutting_backlog.md.


THE MACRO LAYER (ISA-6 -- Tier 3, the first layer above assembly): parameterized recipe TEMPLATES, in
holographic_template.py. A template is a StructureRecipe with named HOLES filled at instantiation -- "tag a value
under a role" (pair), "a two-field record" (record), "an order-bearing pair" (ordered_pair) -- written once and
instantiated with different arguments. Because a StructureRecipe replays BIT-EXACT (atoms are regenerated from
the seed by name), instantiating with different arguments produces the correct DISTINCT structures
deterministically (the recipe's exactness carries -- the bar). A starter library (STARTER_LIBRARY) ships three;
the mind exposes `instantiate_template(name, **args)` and `template_names()`.

THE KEPT NEGATIVE -- MACRO HYGIENE, designed out: atoms are derived from NAMES, so two atoms with the same name
(and kind) are the SAME vector. A template that creates an internal role atom named "role" would COLLIDE with a
caller who fills a hole with an atom also named "role" -- role and value become one vector (capture) and the
binding degenerates. The fresh-atom discipline: template-INTERNAL atoms are namespaced under a reserved prefix
("@tmpl:<name>:") that a caller's bare names cannot hit -- a gensym keyed by template name (so the same template
stays deterministic across instantiations). The witness of capture is cosine(internal_role, caller_value) at
matched kind: ~0 (-0.04 measured) with the discipline, 1.0 without it. `RecipeTemplate` is hygienic by
construction; `_UnhygienicTemplate` (tests only) exhibits the capture the discipline prevents. (Roles are unitary
so the single-binding `pair` recovers its value EXACTLY on unbind; the 2-field `record` recovers each field
approximately -- a 2-item bundle -- but cleanly separable, the right value winning by a wide margin.) SEATS:
Puckette (Pd/Max -- a composition language of parameterized patches over real-time primitives) + the
recipe/procedure thread.

NEXT in the spine (Tier 3, research-heavy): ISA-7 (a small higher-level language that lowers to the recipe IR --
the DCC material-node-graph-as-recipe and the typed-structure unification are already early forms) and ISA-8
(reversible/quantum bind, the frontier).

Tests: +9 (1123 -> 1132). test_holographic_template.py (8): bit-exact determinism, distinct structures from
distinct args, exact recovery for a single-binding template, separable record fields, order-sensitivity of
ordered_pair, the starter library, the hygiene/capture kept negative, and the reserved-namespace convention;
plus a mind-level integration test (instantiate_template -> distinct bit-exact + exact pair recovery through the
mind). Files: holographic_template.py, holographic_unified.py, test_holographic_template.py, test_integration.py,
tour.py, README, holostuff_crosscutting_backlog.md.


THE STRUCTURE LANGUAGE (ISA-7 -- the top of the assembly tower; the last buildable spine item before the
frontier): a small declarative language that LOWERS to the recipe IR, in holographic_lang.py. The surface is
S-expressions: a bare symbol is an atom; (bind a b) / (bundle ...) / (permute a n) lower to the matching recipe
ops; and the ISA-6 templates appear as language forms -- (record name moose), (pair x) -- so the macro layer
becomes language constructs. parse/unparse round-trip the surface (parse o unparse == identity); compile_spec
lowers an AST to a StructureRecipe; realize_spec materialises the vector. The mind exposes compile_structure and
realize_structure. The whole framing: the typed unification (program = tree = scene = record = one
StructureRecipe) IS the IR this language targets -- assembly (the kernel) -> macros (templates) -> language, one
tower over one substrate.

THE BAR met: a declarative spec compiles to a CORRECT recipe and realizes BIT-EXACT -- (bind a b) realizes
exactly to bind(atom a, atom b); a (record ...) form is bit-identical to the ISA-6 template instantiated
directly (the layers agree); same spec -> same vector. And it round-trips on the surface.

THE KEPT NEGATIVE / SCOPE BOUNDARY (heeded, not discovered): a general-purpose language is large and easy to
over-scope, so ISA-7 is SCOPED TO ONE DOMAIN -- structure description. There are NO variables, NO control flow,
NO user-defined functions: just atoms, the base binds, and the fixed template library. The scope is enforced, not
aspirational: an unknown form (while ...) is a ValueError, not a silent no-op, and template/base arities are
checked. This is the "do not build a general language up front" discipline made into test_scope_boundary. SEATS:
Puckette (a declarative DSP language over primitives -- Pd is exactly "a surface that lowers to a small set of
real-time ops") + Eno (language-as-generative-system) + Plate (the IR).

NEXT in the spine -- the frontier (Tier 4, research): ISA-8 (the reversible-computing / error-correction model --
cleanup AS error correction, unbind as the exact inverse of bind). That is the last spine item, and the most
speculative.

Tests: +9 (1132 -> 1141). test_holographic_lang.py (8): surface round-trip, correct lowering of the base forms,
bit-exact determinism, template-forms-agree-with-ISA-6, nested composition recovery, compile-returns-a-replayable
-recipe, the scope-boundary kept negative (unknown form / bad arity are errors), and parser rejection of
malformed input; plus a mind-level integration test (realize_structure + compile_structure + agreement with
instantiate_template through the mind). Files: holographic_lang.py, holographic_unified.py,
test_holographic_lang.py, test_integration.py, tour.py, README, holostuff_crosscutting_backlog.md.


THE REVERSIBLE / ERROR-CORRECTION MODEL (ISA-8 -- the frontier, the LAST item of the VSA ISA spine): names what
the engine has been all along and ships one measured payoff. In holographic_reversible.py + ISA_REVERSIBLE.md.
THREE parts, honestly labelled by what they are. (a) THE REVERSIBILITY AUDIT (framing, but testable):
bind/unbind/permute/involution are REVERSIBLE (exact inverse -- verified: unbind o bind == identity for a unitary
key, permute by -shift, involution self-inverse); bundle/superpose/cleanup are INFORMATION-DESTROYING (sum or
projection -- no exact inverse). The organizing read: the lossy ops are where the coherence budget is spent, and
CLEANUP IS ERROR CORRECTION (snap to the codebook manifold, discard the accumulated error). (b) THE AUTO-CLEANUP
SCHEDULER (THE PRACTICAL CORE, measured): a long program accumulates crosstalk and drifts toward a cliff where
cleanup would snap to the WRONG atom; the scheduler inserts cleanup BEFORE the cliff using an ORACLE-FREE health
signal -- cosine of the running vector to its nearest atom (1.0 on a clean atom, falling as it drifts; the
capacity diagnostic's SNR proxy). 'adaptive' cleans only when health < floor; 'fixed' cleans every k. This
generalizes the shipped coherence-gate from store-MAINTENANCE to program-EXECUTION. MEASURED (bursty damage):
adaptive holds the output above a 0.9 fidelity threshold (frac-below 0.000) at 5 CLEANUPS; the best fixed cadence
that matches that fidelity (k=3) needs 16 -- ~1/3, echoing the coherence-gate's "matched accuracy at ~1/3 the
passes." Fixed cadences using fewer (k=4->12, k=6->8) drop below threshold. (c) THE QUANTUM-GATE CONNECTION
(framing only): FHRR's bind is a diagonal unitary (per-frequency phase rotation), structurally gate-like, which is
why unitarity makes bind exactly invertible -- useful for capacity-as-coherence-budget reasoning, nothing more.

THE LOUD NEGATIVE (the most important honesty note on the spine): this is an ANALOGY, NOT physics. VSA is NOT a
quantum computer -- no exponential superposition, no entanglement, no quantum speedup. We borrow the DISCIPLINE
(error budget + correct-before-the-cliff + reversibility bookkeeping); we do not overclaim the physics. The
practical do-able core is the scheduler (b); (a) and (c) are scaffolding. KEPT honest within (b): under CONSTANT
damage a fixed cadence is already near-optimal, so the adaptive win is specific to VARIABLE damage rates (when
the right fixed k cannot be known in advance); and the health floor must trigger early enough that the nearest
atom is still the true one when cleanup fires. SEATS: Stoudenmire (quantum-inspired/tensor networks; the
FHRR-as-diagonal-unitary framing) + the FHRR/honesty/coherence threads.

*** THE VSA ISA SPINE IS COMPLETE: ISA-1 (determinism contract) -> ISA-2 (conformance suite) -> ISA-3 (extension
discipline) -> ISA-4 (register file) -> ISA-5 (calling convention + permute-stack) -> ISA-6 (macros) -> ISA-7
(structure language) -> ISA-8 (reversible/error-correction model). Eight items, each with its kept negative on
record; the recurring lesson across the whole spine -- superposition buys composability and pays in a crosstalk
cliff, so measure the budget and keep an exact path -- appeared as the bundled disk, the bundled register file,
the permute-stack depth cliff, and finally as the coherence budget the auto-cleanup scheduler manages. ***

Tests: +7 (1141 -> 1148). test_holographic_reversible.py (6): the audit classification (+ unknown-op error),
reversible ops actually round-trip, lossy ops destroy information (bundle mixes, cleanup is idempotent), the
health signal tracks drift, the adaptive-beats-fixed scheduler bar (~5 vs ~16 cleanups at matched fidelity), and
the no-cleanup degradation control; plus a mind-level integration test (reversibility_audit + run_with_auto_cleanup
through the mind). Files: holographic_reversible.py, ISA_REVERSIBLE.md, holographic_unified.py,
test_holographic_reversible.py, test_integration.py, tour.py, README, holostuff_crosscutting_backlog.md.


ANISOTROPIC / STEERING KERNELS (RT-IV1 -- the DCC reverse-transfer item deferred while the ISA spine was built;
now picked up): a direction-dependent metric for the FPE encoder. holographic_fpe.py's VectorFunctionEncoder now
accepts a PER-AXIS bandwidth (a list, one per axis) as well as a scalar -- a diagonal anisotropic kernel: SMALL
bandwidth on an axis = a wide, smooth kernel there; LARGE bandwidth = a sharp one. This is the bounded form of
Milanfar's steering kernel (Takeda/Farsiu/Milanfar 2007) -- n bandwidths, not a per-point covariance that
overfits -- and it is the same object as an anisotropic Gaussian splat (per-splat covariance), the cross-
connection Drettakis' seat noted. Backward-compatible: a scalar bandwidth broadcasts to all axes (the original
isotropic behaviour). holographic_steering.py adds steer_bandwidths (fit per-axis bandwidths from the data's
directional smoothness -- sharp axis -> large bandwidth) and kernel_regress (FPE-kernel-weighted Nadaraya-Watson),
plus the mind faculty steering_regress.

THE BAR met, in the RIGHT REGIME (this took several honest iterations to locate): on DENSE, strongly-directional
data -- a sharp ridge/edge, constant along one axis and sharp across another -- the steered anisotropic kernel
beats the best isotropic RBF by ~8% (grid-vs-grid), pooling the many same-value samples ALONG the flat direction
while staying sharp across the edge. This is the regime steering kernels are actually designed for (image edges,
dense samples).

THE KEPT NEGATIVES (loud, because anisotropy is easy to oversell -- several iterations of the prototype kept
failing until the regime was right): (1) On SPARSE scattered data the advantage collapses to ~1-3% -- not enough
samples to pool along the flat direction; isotropic stays the honest baseline. (2) On ISOTROPIC data (equal
structure both axes) anisotropy gives ~0%, as it must not. (3) The framing "low frequency = can pool widely" is
WRONG when the low-frequency axis still spans a full period over the domain -- there must be a genuine
low-VARIATION direction, not merely low frequency. (4) The STEERING ESTIMATE is unreliable on scattered data: a
per-axis gradient estimated from scattered points is polluted by the OTHER axes varying and can point the WRONG
way (an early prototype steered backwards) -- it needs dense/grid sampling (neighbours that differ in just one
axis) to estimate cleanly, and a perfectly-flat axis gives gradient 0 (guarded against nan). A full per-point
covariance is worse still (the splat module's own anisotropy negative). So: diagonal bandwidths, dense
directional data, isotropic as the fallback. SEATS: Milanfar (steering-kernel regression) + Drettakis
(anisotropic splats).

NEXT (the reverse-transfer thread): RT-I1 (operator_limit / spectral-iteration -- the subdivision = dynamics =
diffusion = resonator unification, the most beautiful but most O(n^3)-haunted; do it in the Fourier/structured
form). The broader BACKLOG.md items (external-baseline harness, theory-and-guarantees doc) also remain.

Tests: +8 (1148 -> 1156). test_holographic_steering.py (7): FPE per-axis bandwidth is anisotropic, scalar
bandwidth is backward-compatible, per-axis length is checked, the anisotropic-beats-isotropic dense-ridge bar,
steering recovers the right direction on dense data, steering handles a perfectly-flat axis (no nan), and the
isotropic-data no-advantage kept negative; plus a mind-level integration test (steering_regress beats a matched
isotropic baseline on the dense ridge through the mind). Files: holographic_fpe.py, holographic_steering.py,
holographic_unified.py, test_holographic_steering.py, test_integration.py, tour.py, README,
holostuff_crosscutting_backlog.md.


SPECTRAL ITERATION (RT-I1 -- the last and most conceptual DCC reverse-transfer item; the unification the backlog
called "the most beautiful but most O(n^3)-haunted"): diagonalise an iterated bind operator once, evaluate any
level or the limit in closed form. In holographic_iterate.py. THE KEY INSIGHT that dissolves the O(n^3) worry: a
bind is circular convolution, which is DIAGONAL in the Fourier basis -- so the eigenvalues of the bind operator U
are simply its rfft spectrum and the eigenvectors are the Fourier modes. The eigendecomposition is FREE (it is
the FFT), never a dense SVD at D=4096 (exactly what the topology module timed out on). This is "live in the
Fourier/structured form where the spectrum is free."

The unification the backlog pointed at: subdivision (Stam's exact eval), the dynamics propagator's k-step rollout
(learn_dynamics), the diffusion sampler's steady state (hopfield.generate), and the resonator's fixed points are
ALL "iterate a linear operator." Given U: (1) the k-step iterate is ONE eval -- raise the transfer to the k-th
power -- matching k sequential binds to FFT tolerance (~1e-15, MEASURED: a 20-step jump == 20 binds to 9e-16);
(2) the limit is closed-form -- decaying modes (|eigenvalue|<1) vanish, persistent modes (|.|~1) remain, a
contractive operator's limit is 0 with NO iteration; (3) convergence/stall is READ OFF the spectrum before
running -- the regime from max|eigenvalue| (contractive -> decays, marginal -> persists, divergent -> blows up),
and the power-iteration rate from the spectral gap |lambda_2|/|lambda_1| (small gap -> slow / near-degenerate
stall). Mind faculties propagator_jump (one-eval k-step) and propagator_spectrum (regime + gap, without running).

THE KEPT NEGATIVES: only LINEAR operators diagonalise this way; the TRUE resonator is nonlinear (alternating
projection + cleanup) and needs delay-embedding -- the spectral prediction is exact for the linear iterate (the
dynamics propagator, power iteration) and only a HEURISTIC for the nonlinear resonator (the dynamics module's own
nonlinearity negative). So the clean exact results are the linear iterate; the "predict a resonator stall from the
spectrum" bar is met in its linear cousin (power-iteration convergence from the spectral gap), with the nonlinear
caveat on the record. Eigenvector sign is pinned (largest-|entry| positive) for determinism -- the ISA-1 fence.
SEATS: Stam (exact subdivision eval = an eigendecomposition of the refinement matrix) + Stoudenmire (spectral/
low-rank) + Koopman/DMD.

*** THE DCC REVERSE-TRANSFER THREAD IS COMPLETE: RT-III1 (graph-Laplacian denoise) -> RT-II1 (nonlinear manifold
chart) -> RT-IV1 (steering kernels) -> RT-I1 (spectral iteration). Four reverse-transfers from the 3D/DCC domain
into the engine, each measured with its kept negative; the reverse-transfer paid (the engine gained a graph
filter, a curved-manifold chart, a direction-dependent metric, and a free closed-form operator iterate). ***

Tests: +8 (1156 -> 1164). test_holographic_iterate.py (7): the eigendecomposition is the free rfft, the k-step
jump matches the k-bind rollout, the contractive limit is closed-form zero, a divergent operator has no finite
limit, the regime is read off the spectrum before running, the spectral gap predicts power-iteration speed, and
the dominant eigenvector is deterministic + unit + the power-iteration fixed direction; plus a mind-level
integration test (propagator_jump matches the learned propagator's rollout + propagator_spectrum reads a regime).
Files: holographic_iterate.py, holographic_unified.py, test_holographic_iterate.py, test_integration.py, tour.py,
README, holostuff_crosscutting_backlog.md.


================================================================================
FORWARD DCC -- the explicit polygon-geometry thread BEGINS (FWD-1 + FWD-2, the Step-0 vertical slice).

THE GAP THIS OPENS: holostuff's geometry has always been IMPLICIT/native -- an SDF is a function (field), a
splat scene is a bundle (splat), a scene-graph is recursive bind/bundle. All mature, all measured. But the
EXPLICIT side -- an actual indexed polygon mesh of the kind Blender/three.js/glTF speak -- did not exist:
grep confirmed no half-edge, no marching cubes, no gltf/glb, no Mesh class anywhere in 133 modules. The Forward
DCC backlog's Tier 0 is the GATE: stand up a conformant mesh kernel (FWD-1) and the binary boundary to a
three.js front end (FWD-2), and prove the whole boundary end-to-end FIRST as a minimal vertical slice before
building the toolkit on top. That slice is this entry.

FWD-1 -- holographic_mesh.py (the mesh kernel). A Mesh is vertices (V,3 float) + faces (tuples; tris, quads,
n-gons all allowed) + optional normals/uvs/colours. The half-edge adjacency (half_edges() -> origin/face/nxt/
twin arrays, cached, deterministic) is the load-bearing structure: it makes neighbour/face/one-ring queries
O(local) instead of O(scan), and it REJECTS non-manifold input loudly (the same directed edge twice -> ValueError)
rather than silently building a corrupt topology. On TOP of it: euler_characteristic / is_closed / is_manifold /
genus (the closed cube reads V8 E12 F6 chi=2 g=0, MEASURED in the selftest), vertex_normals (Newell's method,
area-weighted), triangulate (fan, convex only -- the honest scope limit), to_buffers/from_buffers (flat indexed
float32 position/normal/uv + a triangle index buffer, the GPU-ready form), to_obj/from_obj (topology-preserving
round-trip). Primitives box / tetrahedron / grid for tests and demos.

THE KEPT NEGATIVE (loud, in the module docstring): NumPy is the WRONG tool for tight per-element mesh-edit loops.
Half-edge traversal and incremental edits (split/collapse/flip) are pointer-chasing, not vectorizable; the
selftest prints the build rate (~1.5M half-edges/s) as evidence that this is Python-loop bound. So this kernel is
correct, deterministic, and fine for the geometry SIZES the engine actually manipulates -- but it will NOT scale
to interactive million-poly editing without a compiled core. "NumPy-only" is the ENGINE's rule, not an
interactive-mesh-editor's rule, and pretending otherwise would be the kind of unmeasured claim this project exists
to avoid. The vectorized paths (Euler, normals, buffer build) are fast; the edit loops are the ceiling.

FWD-2 -- holographic_gltf.py (the three.js boundary). mesh_to_glb(mesh) -> bytes: a real glTF 2.0 binary
container (12-byte header + JSON chunk + BIN chunk), POSITION/NORMAL/TEXCOORD_0/COLOR_0 accessors + a triangle
index buffer, the REQUIRED POSITION min/max bounds, a default PBR material, uint16 indices for small meshes else
uint32, little-endian throughout, sort_keys on the JSON so the output is BYTE-REPRODUCIBLE (the determinism rule
reaching all the way to the wire format). glb_to_mesh parses it back; validate_glb returns a structural-conformance
dict (magic / version / chunk lengths / chunk order / 4-byte alignment / position bounds). The cube emits a
1300-byte .glb that round-trips positions/normals/uvs and is structurally valid -- MEASURED in the selftest.

THE INDEPENDENT CHECK: the .glb was loaded with the real third-party pygltflib OFFLINE (scratch only, NOT added to
the suite -- it's a banned dependency, used once as an external oracle exactly the way an external baseline should
be): it loaded cleanly, reported 8 vertices, correct min/max. So "three.js-loadable" is not a hope, it's verified
against an independent glTF reader. The delta/patch channel (ARCH-2, send only what changed) is explicitly
DEFERRED with a loud note in the module -- it is the backlog's highest-value architectural addition and earns its
own thread, not a rushed corner of this one.

WIRED AS FACULTIES (the close-out ritual, additive + backward-compatible): mesh_box / mesh_tetrahedron /
mesh_grid / mesh_euler / mesh_to_gltf / mesh_from_gltf on UnifiedMind, inserted before the SEARCH & DYNAMICS
section. These are explicit-geometry I/O, NOT VSA hypervector ops -- the docstrings say so plainly; the bridge that
makes a mesh a hypervector (mesh <-> SDF <-> splat) is FWD-11/ARCH work, deliberately not faked here.

SEATS: Drettakis (the glTF/three.js boundary and the splat<->mesh bridge to come) + Pharr (indexed buffers and the
acceleration structures meshes feed) + Macklin (the half-edge as the substrate a constraint/edit solver would ride,
and the determinism discipline on connectivity). Connectivity is naturally EXACT (integer indices, no float drift);
normals are TOL (continuous, feeding no decision) -- the ISA-1 fence applied to the new layer.

A NON-OURS NEGATIVE SURFACED BY THE FULL REGRESSION (recorded honestly, not absorbed silently): the full suite run
turned up that test_holographic_market.py::test_big_dai_structure_holds_at_scale is FLAKY on the UNTOUCHED upload
(fails ~1 run in 3). Root cause pinned: it is hash-seed dependent -- with PYTHONHASHSEED fixed it passes 3/3 every
time. The market return-SIGN sequence is genuinely near-random (the efficient-market verdict the test asserts), so
it sits right at the z<2.0 boundary, and hash-seed-dependent set/dict iteration order in the bundling chain
occasionally nudges the order-sensitive float sum across the line. This is a real (minor) determinism-contract gap
-- the contract pins RNG seeds but not hash-seed iteration order -- and it is ORTHOGONAL to the mesh slice (the
slice touches none of that code). Fix is a separate task (pin PYTHONHASHSEED in conftest, OR canonicalize the
bundling order, OR widen the test's statistical band). Left on the record, not papered over.

THE STRATEGIC FORK STILL OWED TO THE OWNER (the backlog's own "decide before FWD-1"): mesh-first (chase Blender
parity) vs native-first (play to the SHIPPED SDF/splat strengths and reach usefulness sooner). The slice plus all
of Tier 1 (UV/smoothing/geodesics/curvature, ported from the already-shipped chart/graphsignal/steering modules)
are valuable under EITHER answer -- so building the slice DE-RISKED the decision rather than pre-empting it. The
ordering of everything below FWD-2 waits on that call.

Tests: +30 (1164 -> 1194). test_holographic_mesh.py (15): Euler invariants on box/tetra/grid, half-edge
reciprocity + cycle closure, neighbour/face queries, outward + unit normals, buffer + OBJ round-trips, the OBJ
slash-face form, non-manifold rejection, degenerate-face rejection, deterministic index buffer, sorted edges.
test_holographic_gltf.py (11): structural validity, 4-byte alignment, position/normal/uv round-trips, triangle
count, position bounds, BYTE-reproducibility, tetra+grid meshes, uint16 index path, bad-magic rejection, file
round-trip. test_integration.py (+4): the mind exposes the mesh faculties, the Euler invariant holds THROUGH the
mind, the cube->glb->cube boundary round-trips through the mind (THE vertical slice), and that boundary is
byte-reproducible through the mind. Files: holographic_mesh.py, holographic_gltf.py, holographic_unified.py,
test_holographic_mesh.py, test_holographic_gltf.py, test_integration.py, tour.py, README, NOTES_concepts.md.


================================================================================
CONSOLIDATION -- the chunkers/tilers/stores converge onto ONE routing fabric (StructuredIndex keying +
TiledStore). The capacity-cliff cure ("route each item to a bounded-load chunk") had been re-grown five
times -- splat tiles, chunk_route, the instruction chunker, the RP-tree forest, the FacetStore buckets --
each module's docstring even NAMING the others ("the same trade chunk_route makes"). This collapses the
duplication onto the shared primitive that already existed (StructuredIndex), the de-siloing the integration
plan flagged.

THE ORGANISING LAW (recovered from the RAM/addressing thread, "as above so below"): the capacity cliff is a
property of ONE bundle, not of the problem; you escape it HORIZONTALLY (RAID-style, capacity = K x per-vector
budget, the HoloArray) and you ADDRESS the shards by a PIVOT -- and the pivot you pick IS the regime. Hash of
the key -> the page-table / LBA / DHT regime: deterministic routing with ZERO comparisons (this is "RAM" --
you COMPUTE where it is, you do not search). Random projection -> nearest-neighbour content recall. Floor-
divide a coordinate -> spatial tiles. One fabric, one parameter.

WHAT SHIPPED (additive, backward-compatible):
  * StructuredIndex gains `keying=` (holographic_tree.py). 'projection' (default) is the original RP-tree
    content recall, BYTE-FOR-BYTE (the 15 existing tree tests are the parity net, green). 'hash' is the RAM /
    page-table regime -- a blake2b address (NOT Python's salted hash, which would reshuffle buckets every
    process and break the determinism rule), ~O(1), exact, with absent keys returning None. 'spatial' floor-
    divides a coordinate into a tile. MEASURED on the substrate before writing it: at N=5000, hash routes in
    1.04 comparisons and spatial in 2.01 (vs 5000 for a flat scan) while projection takes ~594 -- the RAM
    thread's law confirmed (computed-address routing is zero-comparison; NN routing is sublinear-not-zero).
  * `_tile_bucket(coord, tile)` -- the floor-divide route, now in ONE place; the spatial index, TiledStore,
    AND the splat tiler all call it instead of each re-deriving `gy // tile, gx // tile`.
  * `TiledStore` -- the splat-tiler's core, generalised. The KEY DESIGN INSIGHT the audit forced: the clients
    vary on TWO axes, not one. ROUTING (how key -> bucket: projection / hash / spatial) is shared. STORAGE
    (what a bucket HOLDS: explicit keys you FIND, vs a bounded bundle you DECODE) is the one thing that
    differs -- which is why TiledStore is a SIBLING class, not a flag on the index. Bundling is FORBIDDEN in
    an index (rule 2: a superposed index caps with set size) yet CORRECT in a bounded-load tile (the decode
    cap never bites because floor-divide caps each tile at tile**ndim cells). One law, two storage shapes.

FIRST MIGRATION (proven byte-identical): splat_bundle_tiled / recall_region_tiled now DELEGATE their tiling
to TiledStore + _tile_bucket. The splat module owns only its encode (role-bound occupancy) and decode; the
floor-divide routing and bounded grouping live once, in the shared store. A PARITY TEST recomputes every
tile bundle with the OLD inline (gy//tile, gx//tile) logic and asserts np.array_equal -- the delegation
changed NOTHING, bit-for-bit. Build-time and recall-time tiling now provably route identically (same
_tile_bucket), closing a latent class of "the two floor-divides drifted" bug.

THE TWO RULES STILL HOLD (and are now enforced in ONE place instead of rediscovered per caller): KEY ON THE
ITEMS THEMSELVES (a tree only routes when query ~= key; a weakly-correlated summary mis-routes -- the
~0.27-cosine measurement), and NEVER STORE THE INDEX AS A BUNDLE (decode-via-cleanup caps with set size).
TiledStore is the sanctioned exception to the second: it bundles, but only within a bounded-load tile, which
is exactly why it is a separate object with its own docstring saying so.

STILL TO MIGRATE (the remaining clients, each its own backward-compatible + parity-tested increment, in risk
order): RouteIndex -> keying='sequential' (the two-level chunk-summary, its own docstring already admits it
is "this index at its small-n operating point"); the direct HoloForest-wrapping sites (ablate, creature,
denoise/NLM, mind, unified, uri) -> structured_index(keying='projection') (byte-identical, removes the near-
copies); FacetStore -> structured-address keying + recursion (its bi-level "prefix outside, forest inside" IS
index-of-indexes). And the optional axes the RAM thread named but this increment did not need yet: raid=True
(shards backed by HoloArray -- parity + grow), and halo= for the coupling operations (the convolution-as-bind
tiling, where a feature near a tile edge spreads into the neighbour -- overlap-add, which bind's bilinearity
makes clean: bind(f, g) = sum over tiles of bind(f_tile, g)).

SEATS: Pharr (the BVH / acceleration-structure framing of the index), Duda (the addressing / page-table
information view), Stoudenmire (the low-rank shard view) -- and the RAM/addressing thread's own systems-
engineering lesson: fifty years of storage and interconnect design converges on "balanced tree of fixed-
budget nodes, routed by a pivot or a structured address, never summarising content upward." The engine
keeps re-deriving it.

Tests: +10 (1194 -> 1204). test_holographic_tree.py (+7): hash keying is zero-comparison exact (~1
comparison at N=2000) and returns None for absent keys; the hash route is deterministic across processes
(blake2b, not salted hash); hash carries payloads; spatial routes by floor-divide and is exact; locate_exact
agrees with the routed locate for the computed keyings; locate_k refuses non-projection keyings (k-NN is a
content query); TiledStore routes + groups with bounded per-tile load. test_holographic_splat.py (+1): the
byte-identical migration parity test (new tiles == old inline tiling, np.array_equal). test_integration.py
(+2): the three keying regimes reachable through the mind faculty, and the splat tiler + spatial index
provably sharing ONE route. Files: holographic_tree.py, holographic_splat.py, holographic_unified.py,
test_holographic_tree.py, test_holographic_splat.py, test_integration.py, tour.py, README, NOTES_concepts.md.


--------------------------------------------------------------------------------
CONSOLIDATION, increment 2 -- RouteIndex -> the shared 'sequential' keying. The route chunker's two-level
random-access index (nearest chunk SUMMARY, then nearest tile within the chunk) was the next member of the
chunking family to fold onto the one fabric. It is now a keying on StructuredIndex, and RouteIndex delegates.

WHAT SHIPPED (additive, backward-compatible):
  * StructuredIndex gains keying='sequential' (holographic_tree.py): keys are a SEQUENCE of chunks (each a
    (chunk_size, dim) array); locate routes two-level by EXACT scan -- argmax over the chunk summaries (a
    normalised bundle per chunk), then argmax over the chosen chunk's RAW tiles -- returning the (chunk,
    position) coordinate. It reproduces RouteIndex's computation exactly: same normalising bundle for the
    summary, query normalised, tiles kept RAW at level 2 (so NO normalisation drift -- the trap that makes the
    forest-wrapping sites non-trivial, see below). locate_exact delegates to locate (already exact); locate_k
    still refuses (sequential is not nearest-neighbour).
  * RouteIndex (holographic_plan.py) now DELEGATES: it keeps its public surface (self.chunks, n_chunks, and a
    _summaries property that reads the index's summaries for the determinism audit) and its route-specific
    global-step bookkeeping, but the summary computation + two-level routing live in the shared index. A
    PARITY TEST recomputes the old inline two-level scan and asserts locate() returns the identical
    (chunk, pos, global_step) for every tile on a real route -- byte-identical, the delegation changed nothing.

WHY THE FOREST-WRAPPING DE-DUPS WERE DEFERRED (the honest call, kept on record): the six direct
HoloForest(...) sites (ablate, creature, denoise, mind, unified, uri) looked like the biggest duplication
win, but they are NOT byte-identical swaps. HoloForest.recall ranks by RAW DOT PRODUCT
(items[cand] @ query); StructuredIndex unit-normalises its keys, so the RP-tree splits differently and the
candidate set can change. And the sites are mostly the wrong shape to migrate blindly: ablate and creature
are benchmark/demo blocks whose ground truth is itself a dot-product argmax; denoise and the unified graph
use recall_k where the normalised tree shifts the neighbours; mind is a performance-critical hot path tied to
ReflexCache; uri belongs to the FacetStore migration. Forcing them would risk behaviour change for little
gain. The clean path for them later is either a normalize=False option on the projection keying, or migrating
only the ones whose vectors are already unit-norm, each with a parity test. Deferred, not forgotten.

STILL TO MIGRATE (unchanged from increment 1, minus RouteIndex which is now done): FacetStore -> structured-
address keying + recursion (its bi-level "prefix outside, forest inside" IS index-of-indexes -- and its inner
hot-bucket forest is the uri.py site above, so the two land together); the forest-wrapping de-dups (with the
normalisation caveat above); and the optional axes the RAM thread named -- raid=True (HoloArray-backed shards:
parity + grow) and halo= (the convolution-as-bind overlap-add tiling).

Tests: +2 (1204 -> 1206). test_holographic_plan.py (+1): the RouteIndex byte-identical migration parity test
(locate == old inline two-level scan for every tile; summaries bit-identical). test_integration.py (+1):
RouteIndex's routing IS StructuredIndex(keying='sequential'), and an independent sequential index over the
same chunks routes a tile to the same (chunk, position). Files: holographic_tree.py, holographic_plan.py,
test_holographic_plan.py, test_integration.py, tour.py, README, NOTES_concepts.md.


--------------------------------------------------------------------------------
CONSOLIDATION, increment 3 -- the content store delegates, and the forest-de-dup unlock ships. The deferred
forest-wrapping sites were blocked by a real mismatch: HoloForest.recall ranks by RAW DOT PRODUCT, while
StructuredIndex unit-normalised its keys (so the RP-tree split differently). This adds the one parameter that
removes the block and migrates the most on-theme site -- the content store, which StructuredIndex's own
docstring already CLAIMED was "this index at its at-scale operating point" but which was still wrapping a raw
forest itself.

WHAT SHIPPED (additive, backward-compatible):
  * StructuredIndex projection keying gains normalize=True (default = unchanged). normalize=False keeps keys
    RAW, so the tree splits on raw vectors and locate ranks by raw dot product -- making the index
    BYTE-IDENTICAL to a bare HoloForest over the same items. MEASURED: 0/300 query mismatches vs a bare forest
    on deliberately non-unit-norm items. This is the unlock the increment-2 note flagged: a site that already
    wraps a raw forest can now delegate with zero behaviour change.
  * FacetStore (holographic_uri.py) now DELEGATES its hot-bucket content search to StructuredIndex
    (keying='projection', normalize=False), filing each record under its own content vector and carrying the
    record as the payload. build_indexes builds the shared index; nearest() calls locate (returning the record
    + the cost directly). A PARITY TEST proves nearest() returns the SAME record a bare HoloForest would, for
    every query -- so the docstring's claim is now literally true, not aspirational. The prefix-LISTING layer
    (put / list / common_prefixes / tree) is untouched: it is a distinct *listable keyspace* capability, a
    sibling to the lookup index, not a keying of it (the honest scoping from increment 2's findings).

WHAT'S DELIBERATELY NOT DONE (and why -- keep the negatives loud): the other five forest-wrapping sites are
now UNBLOCKED (normalize=False makes them byte-identical too, including the recall_k sites since recall_k
already ranks by cosine over whatever tree was built), but they are NOT worth migrating right now: ablate and
creature are benchmark/demo blocks (calling the forest directly is not a "duplicate implementation", it is
just USING the primitive); mind is a performance-critical hot path tied to ReflexCache where the payload
indirection buys nothing; denoise/unified-graph are recall_k uses that would gain only indirection. Migrating
them would add wrapper overhead for no functional gain. The de-dup that MATTERED was the content store (a
genuine "individual solution" in the chunking/store family, and the one the docstring named); that is done.

WHERE THE CONSOLIDATION STANDS: the routing fabric (StructuredIndex: projection / hash=RAM / spatial /
sequential keying, + normalize toggle, + TiledStore for the decode-a-bundle storage axis) is built, and the
three genuine "individual solutions" in the chunking/tiling/store family now delegate to it -- splat tiler
(spatial), RouteIndex (sequential), FacetStore hot bucket (projection). The capacity-cliff cure lives ONCE.
Remaining are EXTENSIONS, not consolidations: the optional axes the RAM thread named -- raid=True (HoloArray-
backed shards: parity + grow, for the at-scale degradation-tolerant case) and halo= (the convolution-as-bind
overlap-add tiling, where bind's bilinearity makes the tiling clean: bind(f,g) = sum_tiles bind(f_tile, g)).
Those are net-new capability and can be picked up from the backlog rather than as cleanup.

Tests: +3 (1206 -> 1209). test_holographic_tree.py (+1): normalize=False is byte-identical to a bare
HoloForest (0/300 mismatches on non-unit-norm items). test_holographic_uri.py (+1): FacetStore.nearest()
returns the same record as the bare forest it replaced, every query. test_integration.py (+1): a hot bucket's
content search IS a StructuredIndex (keying='projection'), the de-siloing made real. Files: holographic_tree.py,
holographic_uri.py, holographic_unified.py, test_holographic_tree.py, test_holographic_uri.py,
test_integration.py, README, NOTES_concepts.md.


--------------------------------------------------------------------------------
FWD-7 -- the explicit mesh can finally be EDITED, not just described. The kernel (FWD-1) shipped the half-edge
substrate and the Euler invariants but was effectively read-only. This adds the LOCAL EULER OPERATORS -- the
bounded connectivity rewrites known since Baumgart/Mantyla that every higher modeling op (subdivide, bevel,
decimate, remesh) decomposes into.

SCOPE (honest): this ships the PRIMITIVE Euler-operator layer (flip / split / collapse / split_face) -- the
substrate FWD-7's user-facing modeler VERBS (extrude / bevel / inset / loop-cut / bridge / dissolve) decompose
into (extrude = a loop of MEV+MEF, loop-cut = a ring of split_edge, dissolve = KEV/KEF). Those verbs are the
REMAINING FWD-7 work. The primitives already satisfy FWD-7's stated bar -- manifold-preserving on valid input,
deterministic per the ISA contract, and undoable via the make/kill round-trips -- so this is the foundation
the verbs stand on, not FWD-7 complete. NOTE ON ORDERING: the forward backlog puts Tier 1 (FWD-3/4/5/6, the
ADAPT-SHIPPED items that wire chart/graphsignal/steering onto meshes) ABOVE FWD-7 (Tier 2) by leverage; these
primitives were built first as the natural continuation of the kernel, but Tier 1 is the higher-leverage path
and is where the work resumes next. The panel converged here over the bigger forward items: the [Stam] seat
(subdivision surfaces ARE sequences of Euler operators; his exact Catmull-Clark evaluation is the rigor
reference) called it the floor everything else stands on; the [Pharr] seat set the bar (the result must stay
a renderer-valid manifold, not merely "run"); the [Cranmer] seat supplied the measurement (make/kill inverse
pairs give an exact do-then-undo round-trip -- the cleanest correctness witness, not an in-sample fit). FWD-11
(the mesh<->SDF<->splat VSA bridge, the [Plate]/[Quilez]/[Drettakis] seats) is the higher-value follow-on but
bigger and benefits from having real edit operators first.

WHAT SHIPPED (holographic_eulerops.py; additive, four standalone operators + four UnifiedMind faculties):
  * flip_edge(mesh, a, b)        -- rotate the shared edge of two triangles. V/E/F (hence chi) unchanged: the
    purest rewrite. The Delaunay-remeshing primitive. Its own inverse (flipping the new edge c-d restores the
    old). PRECONDITION kept loud: refuses if c-d already exists (would be shared by 3 faces -> non-manifold).
  * split_edge(mesh, a, b)       -- insert a midpoint vertex, splitting the incident triangle(s). V+1, chi
    unchanged. Returns (new_mesh, m). The refinement primitive.
  * collapse_edge(mesh, keep, remove) -- the INVERSE of split_edge: merge an edge's endpoints. V-1, chi
    unchanged. The decimation/LOD primitive. GUARDED by the LINK CONDITION: keep and remove may share
    neighbours only at the apexes of their shared faces; otherwise the contraction would weld the surface
    onto itself, so it returns None. Not every edge is collapsible -- a true property of meshes, made
    operational (the caller must handle the refusal), not a code shortcoming.
  * split_face(mesh, f, i, j)    -- cut a polygon with a diagonal between two corners (MEF). E+1, F+1, chi
    unchanged. The one operator that works on n-gons, not just triangles.

DESIGN CHOICE (readability over pointer surgery): each operator uses the half-edge adjacency to FIND the local
patch (which faces share the edge, the opposite apexes) then REWRITES THE FACE LIST and lets the new Mesh
rebuild its own half-edge table -- so the combinatorics stay legible and there is no cache to invalidate or
twin pointer to fix by hand. Cost: a rebuild per edit, consistent with the kernel's recorded "NumPy is the
wrong tool for per-element mesh loops" negative (which this module inherits and re-flags).

DETERMINISM (ISA EXACT class): new vertices are APPENDED (index = old count); faces rewritten in face order;
vertex removal reindexes by one fixed rule (drop the index, decrement higher ones). Pure function of
(mesh, selection) -> byte-identical out (asserted). No float comparison ever chooses connectivity.

KEPT NEGATIVES: (1) collapse_edge is not always legal (link condition) -- refuses rather than break the mesh;
(2) flip into an existing edge is illegal -- refuses; (3) flip/split/collapse require triangle faces on the
touched faces and raise otherwise (split_face is the n-gon operator); (4) the per-element Python-loop bound
remains -- fine for interactive single edits, a compiled core is still the eventual need for heavy remeshing.

Tests: +12 (1209 -> 1221). test_holographic_eulerops.py (+11): flip chi/V/E/F-invariance and flip-back
round-trip; the flip duplicate-edge refusal; split_edge vertex-add + chi preservation; the split->collapse
exact make/kill round-trip; the collapse link-condition refusal (bipyramid equator) and a legal collapse
(bipyramid -> tetrahedron); split_edge's non-triangle rejection; split_face n-gon chi preservation and its
adjacent-corner rejection; operator determinism. test_integration.py (+1): the operators as UnifiedMind
faculties preserving the invariants end-to-end (split->collapse restores; flip stays a closed manifold).
Files: holographic_eulerops.py (new), test_holographic_eulerops.py (new), holographic_unified.py (4 faculties),
test_integration.py, README, NOTES_concepts.md, tour.py.


--------------------------------------------------------------------------------
FWD-4 -- the first TIER 1 ADAPT-SHIPPED item: mesh smoothing is the shipped Taubin filter, wired onto a mesh.
The forward backlog's real insight is that the matured intrinsic-geometry toolkit turns the "conventional" DCC
items into adaptations of shipped faculties, and the panel converged here (over the out-of-order FWD-7
primitives) because Tier 1 is the highest-leverage path AND valuable under either fork of the native-vs-mesh
strategic decision. FWD-4 is the cleanest of the four: `graphsignal.taubin_filter(vectors, nbr_idx, nbr_w)`
already exists and is tested, so mesh smoothing is THREE substitutions and nothing else -- vertex positions as
the signal, the mesh 1-ring as `nbr_idx`, cotangent weights as `nbr_w`. This is a WIRE, not a re-implementation.

WHAT SHIPPED (holographic_meshsmooth.py; additive; one UnifiedMind faculty `mesh_smooth`):
  * cotangent_adjacency(mesh) / uniform_adjacency(mesh) -> (nbr_idx, nbr_w) in the (V, k_max) padded,
    row-normalised format the shipped filter consumes. COTANGENT = the discrete Laplace-Beltrami weight
    w_ij = (cot a + cot b)/2 over the two adjacent triangles' opposite angles -- geometry-aware (accounts for
    triangle shape), so it approximates true surface diffusion, not mesh-connectivity diffusion. Triangulates
    internally (cotangents are triangle angles); clamps negative (obtuse) weights to >= 0.
  * taubin_smooth(mesh, lam, mu, iters, weights) -> a new Mesh, smoothed positions, FACES UNTOUCHED (so all
    connectivity and Euler invariants preserved -- only vertices move). Delegates the filtering to the shipped
    graphsignal.taubin_filter. laplacian_smooth ships as the shrinking baseline.

WHY TAUBIN not naive Laplacian: lambda-only Laplacian smoothing SHRINKS the surface toward its centroid;
Taubin alternates a shrink (lambda>0) and a larger un-shrink (mu<0, |mu|>lambda) step, preserving low-frequency
extent while removing high-frequency noise.

MEASURED (the [Milanfar] denoiser-as-manifold-map bar, the [Cranmer] no-shrink test): on a noisy unit sphere
(subdiv-3 icosphere, sigma=0.05), Taubin cut radial error 0.0400 -> 0.0177 (a 56% denoise) and KEPT the mean
radius at 1.009 (no shrink), while the naive Laplacian collapsed the mean radius to 0.894. Connectivity and chi
unchanged. Deterministic (byte-identical positions run-to-run).

KEPT NEGATIVES (loud):
  * Fixed strength over-smooths an already-clean mesh (it is a low-pass) -- proper use needs a noise estimate;
    the faculty exposes lam/mu/iters and does NOT auto-tune (the sigma-estimate discipline, deferred).
  * COTANGENT IS NOT UNIFORMLY BETTER. On THIS regular sphere with isotropic noise, UNIFORM weights denoise a
    touch better (0.0146 vs 0.0177) -- a near-regular mesh with directionless noise has no triangle-shape
    variation for cotangent to exploit. Cotangent's real edge is IRREGULAR meshes / feature preservation; it is
    the default for that reason, but we keep the honest finding rather than assert a superiority that isn't
    there in this case.
  * Cotangent weights can go negative on obtuse triangles -> clamped to >= 0 (the standard intrinsic/clamped
    cotangent mitigation; exact on well-shaped meshes, a documented approximation on very obtuse ones).

ORDERING NOTE: this resumes the backlog at its real highest-leverage point. Remaining Tier 1: FWD-6 (curvature
/ feature detection via steering + Laplacian -- which FWD-4's crease-aware mode wants), then FWD-3 (UV via
manifold_chart on mesh edges) and FWD-5 (geodesics via chart.geodesic_distances), both of which need the
seam/atlas machinery (ARCH-4) because they replace manifold_chart/geodesic_distances' k-NN graph with explicit
mesh edges. The FWD-7 modeler verbs (extrude/bevel/inset/loop-cut/bridge/dissolve) stand on the already-shipped
primitive Euler operators and are Tier 2.

Tests: +10 (1221 -> 1231). test_holographic_meshsmooth.py (+9): Taubin denoises; no-shrink; the Laplacian
baseline shrinks; connectivity + chi preserved (vertices-only); both weightings denoise (no false cotangent
superiority); adjacency is row-normalised + rectangular; cotangent weights non-negative; quad topology kept;
determinism. test_integration.py (+1): mesh_smooth as a UnifiedMind faculty denoising without shrink end-to-end.
Files: holographic_meshsmooth.py (new), test_holographic_meshsmooth.py (new), holographic_unified.py (faculty),
test_integration.py, README, NOTES_concepts.md, tour.py.


--------------------------------------------------------------------------------
FWD-6 -- Tier 1, item four: mesh curvature & feature detection, the item with the most RIGOROUS reference of
the forward set because discrete differential geometry hands us exact identities to check against. Three
measurements, each reusing shipped machinery or grounded in a hard invariant:

WHAT SHIPPED (holographic_meshcurvature.py; additive; three UnifiedMind faculties + a confidence faculty):
  * mean_curvature(mesh) -> |H| via the discrete mean-curvature-normal operator K(x_i)=(1/A_i) sum_j w_ij
    (x_i-x_j) = 2 H_i n_i, with w_ij the cotangent edge weights REUSED from FWD-4 (curvature and smoothing are
    the same operator -- one applied, one measured) and A_i the barycentric vertex area. On a unit sphere
    H=1/R=1.
  * gaussian_curvature(mesh) -> angle defect / area (K=1/R^2=1 on the unit sphere), with the strongest check in
    the module: gauss_bonnet_defect(mesh) = (total angle defect) - 2*pi*chi, which is ~0 to FLOATING POINT for
    a closed mesh by discrete Gauss-Bonnet -- the curvature estimate validated against the Euler characteristic
    FWD-1 computes.
  * dihedral_angles(mesh) / detect_creases(mesh, threshold_deg) -> sharp-edge detection via the angle between
    adjacent face normals. A cube's 12 edges are 90-degree creases; a smooth sphere has none.
  * curvature_confidence(mesh) -> per-vertex [0,1] reliability from 1-ring regularity (the noise negative made
    actionable).

THE [MILANFAR] STRUCTURE-TENSOR / STEERING CONNECTION: curvature is the surface's local shape -- the directions
and rates it bends -- which is exactly the anisotropic local metric a steering kernel encodes (structure tensor
on geometry, not image gradients). The curvature field + crease set are what an adaptive operator STEERS by
(subdivide where |K| is high, smooth along creases not across them, split shading normals at sharp edges). This
ships the scalar curvatures + crease set (the measurable core); the anisotropic steering of downstream
operators is the consumer.

MEASURED (the exact references): on a subdiv-3 unit icosphere -- Gauss-Bonnet total defect = 2*pi*chi to 6e-14
(machine precision); mean Gaussian K=1.022, mean |H|=1.010 (both ~1); per-vertex coefficient-of-variation 0.07
(the noise negative). On a cube -- exactly 12 creases, each a 90-degree dihedral; the 6 flat triangulation
diagonals correctly excluded; a smooth sphere yields 0 creases. Deterministic.

KEPT NEGATIVES (loud):
  * Per-vertex curvature is NOISY on coarse/irregular meshes -- the MEAN over a closed surface is accurate and
    the Gauss-Bonnet TOTAL is exact, but individual vertex values vary (CoV 0.07 even on a regular sphere). The
    estimate needs a reasonably regular 1-ring; curvature_confidence scores per-vertex reliability so a caller
    can down-weight rather than trust blindly.
  * The exact Gauss-Bonnet check is for CLOSED meshes; an open mesh carries a boundary (geodesic-curvature)
    term, so its total defect is not 2*pi*chi and the check is skipped there.
  * Angle defect + the cotangent operator assume TRIANGLE faces; n-gons are triangulated for the computation
    (face normals / dihedral use Newell so they work on n-gons directly).

REFACTOR (additive, backward-compatible): exposed `cotangent_edge_weights(mesh)` from holographic_meshsmooth
(the raw, un-clamped Laplace-Beltrami edge weights), used internally by FWD-4's cotangent_adjacency and reused
by FWD-6's mean_curvature -- so the cotangent computation lives in one place. FWD-4's selftest + 9 tests
re-verified green after the refactor.

Tests: +13 (1231 -> 1244). test_holographic_meshcurvature.py (+12): Gauss-Bonnet exact on a closed mesh (and on
a cube); unit-sphere mean/Gaussian curvature ~1; barycentric vertex areas sum to surface area; cube = 12
creases at 90deg; triangulated cube still 12 (flat diagonals excluded); smooth sphere = 0 creases; per-vertex
noise negative; confidence in range; determinism. test_integration.py (+1): curvature + creases as UnifiedMind
faculties end-to-end with the exact Gauss-Bonnet validation. Files: holographic_meshcurvature.py (new),
test_holographic_meshcurvature.py (new), holographic_meshsmooth.py (additive cotangent_edge_weights),
holographic_unified.py (3 faculties), test_integration.py, README, NOTES_concepts.md, tour.py.


--------------------------------------------------------------------------------
FWD-5 -- Tier 1, the geodesic item: distance ALONG the surface, and the foundation FWD-3 (UV) stands on. The
shipped chart.geodesic_distances computes geodesics as shortest paths on a graph (Floyd over a k-NN graph), and
chart.classical_mds embeds any distance matrix. The honest ADAPT-SHIPPED move: run the same shortest-path idea
on the EXPLICIT MESH EDGE graph (true surface connectivity, real Euclidean edge lengths) instead of a k-NN
approximation. FWD-3 will feed the resulting geodesic matrix to the shipped classical_mds for the UV chart.

WHAT SHIPPED (holographic_meshgeodesic.py; additive; two UnifiedMind faculties):
  * geodesic_distances(mesh, source) -- single-source Dijkstra along mesh edges (Euclidean weights) -> distance
    to every vertex. Along the surface, not the straight line through the void.
  * geodesic_matrix(mesh) -- all-pairs (repeated Dijkstra); the distance matrix FWD-3's classical-MDS UV chart
    consumes.
  * geodesic_soft_selection(mesh, source, radius, falloff) -- a [0,1] falloff by geodesic distance that does NOT
    bleed to vertices near in 3-D space but far across the surface (the geodesic-vs-Euclidean win).

MEASURED (vs the analytic great circle on a unit sphere): geodesic from the pole correlates with arccos(z) at
corr=0.994; antipode (north->south) geodesic = 3.136 ~ pi, GREATER than the Euclidean diameter (2.0). The
geodesic-vs-Euclidean contrast made concrete: a soft selection at radius 2.5 EXCLUDES the antipode (geodesic
~pi > 2.5) that a Euclidean ball of the same radius would INCLUDE (straight-line 2.0 < 2.5) -- the "even on the
surface, no bleed" property. Deterministic (Dijkstra ties break on integer vertex index).

KEPT NEGATIVES (loud):
  * The edge-graph geodesic is APPROXIMATE. It overestimates the polyhedron's own face-crossing geodesic (edge
    restriction), and vs a SMOOTH surface sits a few percent high overall (+7.7% net on the sphere) with a tiny
    chord-effect undercut possible near the source (edges are chords, slightly shorter than arcs; <1%). The
    earlier draft asserted a clean ">= true geodesic" bound -- that is WRONG against a smooth reference (61/258
    sphere vertices slightly undercut), so the test now measures the net overestimate AND the bounded undercut
    rather than claim a one-sided bound that does not hold. Accept where the mesh is fine and curvature mild.
  * All-pairs is O(V * E log V) -- fine here, not for very large meshes; a heat-method solve is the next step.

Tests: +10 (1244 -> 1254). test_holographic_meshgeodesic.py (+9): great-circle correlation; net overestimate
with bounded undercut; antipode farthest + exceeds Euclidean; soft-selection excludes the antipode a Euclidean
ball includes (no bleed); soft-selection in range; matrix symmetric + zero diagonal; reachability on a closed
mesh; flat-grid along-grid >= straight-line sanity; determinism. test_integration.py (+1): geodesic + soft
selection as UnifiedMind faculties with the geodesic-vs-Euclidean contrast end-to-end. Files:
holographic_meshgeodesic.py (new), test_holographic_meshgeodesic.py (new), holographic_unified.py (2 faculties),
test_integration.py, README, NOTES_concepts.md, tour.py. NEXT: FWD-3 (UV unwrapping) feeds geodesic_matrix to
the shipped classical_mds -- the last Tier 1 item, needing seam handling for closed surfaces (ties to ARCH-4).


--------------------------------------------------------------------------------
FWD-3 -- Tier 1 CLOSED: UV unwrapping, the payoff of FWD-5 and the backlog's sharpest irony. UV, the
"least-holostuff" item on the original DCC list, is a near-direct reuse of shipped, tested faculties. The shipped
chart.classical_mds embeds any distance matrix; chart.manifold_chart's Isomap is exactly "classical MDS of the
GEODESIC matrix". So UV unwrapping = feed the mesh's OWN geodesic distances (FWD-5's geodesic_matrix, on explicit
edges) to the shipped classical_mds; the 2-D embedding IS the UV chart. Machinery shipped, substitution is "mesh
geodesics in place of k-NN geodesics".

WHAT SHIPPED (holographic_meshuv.py; additive; two UnifiedMind faculties):
  * uv_unwrap(mesh, method) -- (V,2) UV packed to ~[0,1]^2. 'isomap' (geodesic-preserving, wins on curved),
    'planar' (linear PCA, exact on developable), 'spectral' (Laplacian eigenmaps).
  * uv_distortion(mesh, uv) -- per-edge STRETCH spread (log-ratio std): 0 = isometric, grows with curvature.
  * flat_grid_mesh / hemisphere_cap / puncture -- developable reference, curved test surface, closed->disk seam.

MEASURED: flat isotropic patch unwraps near-isometric (stretch spread 0.049); curved hemisphere cap 0.233
(Gauss -- unavoidable); punctured sphere 0.507 (closed needs a real seam). The bar's "charts don't overlap after
packing" met EXACTLY: both flat and cap unwraps are 100% orientation-consistent (zero flipped triangles ->
locally injective). The bar's "beats a baseline": on the CURVED cap, Isomap (0.233) beats a naive linear PCA
projection (0.456) -- geodesic preservation wins where the surface bends.

KEPT NEGATIVES (loud):
  * Disk-topology required. A CLOSED surface has no boundary and cannot flatten to a disk without a CUT; direct
    unwrapping distorts badly. puncture() opens it crudely (one vertex), and the test MEASURES the punctured
    sphere distorting far more than a cap -- the seam-need made concrete. A good seam (a cut path placed by the
    topology/genus faculty) is ARCH-4, deferred.
  * On a DEVELOPABLE (flat) surface the linear 'planar' projection is EXACTLY isometric (0.000) and BEATS Isomap
    (0.049) -- Isomap carries the edge-graph geodesic's small approximation error. Isomap is NOT universally
    better; it wins on curved surfaces (its purpose) and slightly loses to linear on flat ones. This mirrors
    chart.py's own philosophy ("linear SVD remains the right choice when the manifold is flat"). Pick by curvature.
  * The unwrap is sensitive to triangulation ANISOTROPY: a FAN triangulation (all diagonals one way) biases the
    edge-graph geodesic and distortion GROWS with resolution (measured 0.14 -> 0.17, 5x5 -> 15x15); an ISOTROPIC
    (alternating-diagonal) mesh behaves correctly, distortion SHRINKS toward isometric as it refines
    (0.063 -> 0.044). flat_grid_mesh uses alternating diagonals for this reason -- the bias is documented, not
    hidden by picking a passing triangulation. (This diagnosis corrected a first-draft threshold of <0.06 on a
    fan grid that fails at 0.14; the honest finding is the anisotropy, kept on record.)

Tests: +10 (1254 -> 1264). test_holographic_meshuv.py (+9): flat near-isometric; non-degenerate UV; no flipped
triangles (no overlap, both surfaces); curved cap distorts more than flat; Isomap beats planar on curved; planar
beats Isomap on flat (the honest reverse, planar exact); punctured sphere is a disk and distorts most; unit-square
packing; determinism. test_integration.py (+1): UV unwrap through the mind, flip-free, Isomap-beats-planar on the
cap end-to-end. Files: holographic_meshuv.py (new), test_holographic_meshuv.py (new), holographic_unified.py
(2 faculties), test_integration.py, README, NOTES_concepts.md, tour.py.

TIER 1 COMPLETE: FWD-4 (smoothing), FWD-5 (geodesics), FWD-6 (curvature/creases), FWD-3 (UV) all shipped -- the
cheap, high-leverage wires that turn the matured intrinsic-geometry toolkit (chart, graphsignal, steering,
spectral-iteration) onto explicit meshes. NEXT: Tier 2 -- FWD-7 modeler VERBS (extrude/bevel/inset/loop-cut/
bridge/dissolve, decomposing into the shipped primitive Euler operators), FWD-8 subdivision (reuses
spectral-iteration), FWD-9 rig/skinning (reuses moe), FWD-10 IK (reuses iterate-a-projection); then Tier 3
FWD-11 (mesh<->SDF<->splat bridge). ARCH items interleave (ARCH-4 atlas/seams would give FWD-3 a real seam).


--------------------------------------------------------------------------------
FWD-7 (core) -- Tier 2 LEAD: the modeller VERBS, built on the explicit mesh kernel. The shipped primitive Euler
operators (holographic_eulerops: flip/split_edge/collapse/split_face) are the atomic invariant-preserving moves;
these are the human-facing operations on top. Backlog thesis: the verbs DECOMPOSE into Euler operators -- the
honest frame, with one explicit caveat (extrude needs MEV, which the four shipped primitives don't include, so it
is a direct face-list construction in the primitives' style, not a literal call sequence we can't make).

WHAT SHIPPED (holographic_meshverbs.py; additive; three UnifiedMind faculties):
  * extrude_face(mesh, face, distance) -- lift a face along its normal + side walls. The iconic verb.
  * inset_face(mesh, face, ratio) -- shrink a face toward its centroid + surrounding ring (in-plane).
  * dissolve_vertex(mesh, vertex) -- remove a vertex + its umbrella, fan-triangulate the hole (Euler KEV). The
    decimation cousin collapse_edge (shipped) instead merges the vertex onto a neighbour.
  Shared helper _ring_walls wires the side/ring walls with windings that SUPPLY the freed directed edges (the
  manifold-balance condition). Walls triangulated -> output stays pure-triangle (safe for cotangent/curvature).

MEASURED -- each verb produces a VALID mesh (the bar) with an EXACT geometric signature:
  * All three PRESERVE chi (=2) and keep a closed mesh CLOSED + MANIFOLD, on both a triangle mesh (icosphere) and
    a QUAD mesh (box, degree-3 vertices) -- robust across mesh types.
  * extrude: cap moves EXACTLY `distance` (0.300) along the face normal and ONLY along it; outward extrude
    increases signed volume (box 1.0 -> 1.5).
  * inset: central-face area EXACTLY (1-ratio)^2 of the original; central face stays coplanar (normal preserved).
  * dissolve: removes EXACTLY one vertex (icosphere 66 -> 65).
  Deterministic (new vertices are pure functions of input positions; faces appended in fixed order).

KEPT NEGATIVES / SCOPE (loud):
  * CORE three only. bevel, bridge, loop-cut are the FWD-7 remainder, deferred: bevel/bridge need vertex
    DUPLICATION with an offset/correspondence (fiddly, easy to get subtly wrong); a general loop-cut needs robust
    loop tracing on an arbitrary triangle mesh. Three correct measured verbs > six shaky ones.
  * extrude is NOT a literal composition of the four shipped primitives (it needs MEV, not in the set) -- the
    decomposition is the conceptual model, the direct construction is the honest implementation. Said plainly in
    the module docstring rather than overclaimed.
  * dissolve_vertex fan-triangulates the hole from one ring vertex -- valid TOPOLOGICALLY (covers the polygon,
    stays manifold) but not a quality remesh for a wildly non-convex link; a curvature-aware fill is out of scope.

Tests: +11 (1264 -> 1275). test_holographic_meshverbs.py (+10): extrude/inset/dissolve each preserve
chi+closed+manifold; extrude cap moves exactly distance along normal; extrude increases volume; inset area =
(1-ratio)^2; inset coplanar; dissolve removes one vertex; all three on a quad box; determinism.
test_integration.py (+1): the three verbs through the mind with the exact extrude signature end-to-end. Files:
holographic_meshverbs.py (new), test_holographic_meshverbs.py (new), holographic_unified.py (3 faculties),
test_integration.py, README, NOTES_concepts.md, tour.py. NEXT: FWD-7 remainder (bevel/bridge/loop-cut) OR
FWD-8 subdivision (reuses spectral-iteration), FWD-9 rig/skinning (reuses moe), FWD-10 IK (iterate-a-projection).


--------------------------------------------------------------------------------
FWD-8 -- Tier 2: mesh subdivision (Loop, for triangle meshes). Subdivision is two operations braided, and naming
them honestly is the point: (1) REFINE -- split each triangle into 4 (a topological op; an Euler-operator
sequence per the Stam seat; the genuinely NEW part), and (2) SMOOTH -- move every vertex to a Loop-weighted
neighbour average (a graph-signal LOW-PASS, the SAME family as FWD-4's Taubin on the shipped graphsignal, whose
smooth limit lives in the low-frequency eigenspace holographic_spectral computes -- the "reuses spectral-iteration"
half). So: refinement new, smoothing is the spectral low-pass the engine already owns.

WHAT SHIPPED (holographic_meshsubdiv.py; additive; one UnifiedMind faculty):
  * loop_subdivide(mesh, levels=1) -- Loop subdivision with the proper masks: interior edge vertex 3/8(a+b)+
    1/8(c+d), boundary edge midpoint; interior vertex reposition (1-n*beta)v + beta*sum(nbrs) with Warren's
    beta=(1/n)(5/8-(3/8+1/4 cos 2pi/n)^2), boundary 3/4 v + 1/8(prev+next); retriangulate 1->4. Non-triangle
    input is triangulated first (Loop is a triangle scheme). Returns a new triangle Mesh.

MEASURED (the bar -- a valid mesh with the exact subdivision properties):
  * Each level multiplies faces by EXACTLY 4 (icosphere(1) 32 -> 128 -> 512) and gives V'=V+E (one new vertex
    per edge; 18+48=66). chi preserved, closed mesh stays a closed manifold.
  * AFFINE REPRODUCTION (the exact rigor reference, the Stam seat's ask): a FLAT mesh stays flat to machine
    precision (<1e-12 in z) because the Loop masks are barycentric -- the discrete analogue of Catmull-Clark
    reproducing a plane.
  * SMOOTHING (low-pass signature, made geometric): dihedral-angle spread on a cube drops 0.740 -> 0.102 over
    two levels -- the low-pass character of the smoothing step, dramatic and clear.
  Deterministic (weighted averages; edges visited in sorted order -> fixed new-vertex indices).

KEPT NEGATIVES (loud):
  * Loop is a TRIANGLE scheme. A quad mesh (box) is triangulated first, so the result reflects that
    triangulation, not a Catmull-Clark quad refinement. Catmull-Clark (the quad scheme) is a separate operator,
    not shipped.
  * The limit surface is NOT the input's circumscribed smooth shape -- subdividing an inscribed icosphere does
    not reproduce the exact sphere (subdivision surfaces have their own limit). The exact-reproduction guarantee
    is for AFFINE/planar input only; for curved input the scheme smooths toward its own limit (the honest claim).

Tests: +9 (1275 -> 1284). test_holographic_meshsubdiv.py (+8): faces x4; V'=V+E; chi + closed manifold; flat
stays flat (affine reproduction, exact); smooths an angular cube (spread roughly halved or more); two levels x16;
all-triangle output from a quad input; determinism. test_integration.py (+1): subdivision through the mind --
quadruple, chi/manifold preserved, flat-stays-flat, cube-smoothed end-to-end. Files: holographic_meshsubdiv.py
(new), test_holographic_meshsubdiv.py (new), holographic_unified.py (1 faculty), test_integration.py, README,
NOTES_concepts.md, tour.py. NEXT: FWD-9 rig/skinning (reuses moe), FWD-10 IK (iterate-a-projection); or the
FWD-7 remainder (bevel/bridge/loop-cut); then Tier 3 FWD-11 (mesh<->SDF<->splat bridge).


--------------------------------------------------------------------------------
FWD-10 -- Tier 2, the cleanest reuse on the list: inverse kinematics (FABRIK) expressed LITERALLY through the
shipped iterate-a-projection engine. IK asks: given a chain of fixed-length bones and a TARGET for the tip, where
must the joints go so the tip reaches it while every bone keeps its length? FABRIK (Forward And Backward Reaching
IK) is exactly "iterate a projection onto constraints" -- each reaching pass projects each joint onto the sphere
of correct distance from its neighbour, root and target pinned. The engine ALREADY owns that loop:
holographic_denoise.project_onto_constraints (the mind's project_onto_constraints faculty -- Macklin's one object
under the resonator, the PnP denoiser, and PBD) sweeps a list of projection callables in order until they jointly
hold, and that sweep IS FABRIK's forward/backward reaching. So this module does not reimplement the iteration; it
BUILDS the kinematic-chain projections and hands them to the shipped sweeper. Reuse is literal, not a resemblance.

WHAT SHIPPED (holographic_meshik.py; additive; one UnifiedMind faculty solve_ik):
  * solve_ik(joints, target, iters=20, tol=None) -- pose a chain (n+1,3) so the tip reaches target; pure call into
    project_onto_constraints over the chain projections. Returns (new_joints, n_sweeps).
  * chain(n, length, axis) -- a straight test chain.
  Projections: forward reach (pin tip to target, then end->root move inner joint onto radius-L sphere of outer);
  backward reach (pin root, then root->end move outer joint onto radius-L sphere of inner). One sweep = one
  forward + one backward FABRIK pass.

MEASURED (the bar):
  * REACHABLE target (within total chain length): tip reaches it to <1e-6 in 30 sweeps.
  * Every BONE LENGTH preserved to 1e-9 (the hard constraint FABRIK maintains exactly) and ROOT fixed to 1e-12.
  * UNREACHABLE target (beyond reach): chain fully EXTENDS -- tip at distance (total length=4.000) from root,
    pointing straight at the target (cos > 1-1e-6). The correct degenerate outcome, measured not failed.
  * Convergence MONOTONE in sweeps. Works on longer chains (8 bones). Deterministic (pure geometry, no RNG; a
    zero-length direction falls back to a fixed axis -- deterministic tie-break, the Macklin bit-exact lesson).

KEPT NEGATIVES (loud):
  * Plain FABRIK has NO joint-angle limits and no obstacle avoidance -- position constraints only. A per-joint
    cone projection would slot into the SAME sweep, but is not shipped.
  * An UNREACHABLE target cannot be reached by any solver -- the honest outcome is the fully-extended chain.
  * FABRIK returns A solution, not THE solution -- a redundant chain has many poses reaching a target; this is
    the one the sweep lands on from the given start (deterministic but start-dependent).

Tests: +9 (1284 -> 1293). test_holographic_meshik.py (+8): reaches reachable target; preserves every bone length;
root fixed; unreachable fully extends; extended chain points at target; convergence monotone in sweeps; longer
chain (8 bones); determinism. test_integration.py (+1): IK through the mind via its own project_onto_constraints,
reachable hit + bones/root preserved + unreachable extended end-to-end. Files: holographic_meshik.py (new),
test_holographic_meshik.py (new), holographic_unified.py (1 faculty solve_ik), test_integration.py, README,
NOTES_concepts.md, tour.py. Tier 2 now: FWD-7 core, FWD-8, FWD-10 shipped. NEXT: FWD-9 rig/skinning (LBS as a
mixture of expert bone-transforms <-> moe); or the FWD-7 remainder (bevel/bridge/loop-cut); then Tier 3 FWD-11.


--------------------------------------------------------------------------------
FWD-9 -- Tier 2, the last core item: skinning/rigging (linear blend skinning) as a SOFT mixture of expert
bone-transforms. Skinning deforms a vertex as a WEIGHTED COMBINATION of what each bone's transform would do to it,
weights summing to one -- structurally a mixture of experts (bones = experts, skin weights = gate).

THE HONEST REUSE FINDING (reported, not buried -- like FWD-8's spectral nuance): holostuff's mixture of experts
(holographic_moe.GatedMixture) is the HARD, SPARSE, LEARNED kind (top-1 router, gate = the creature brain, only
the chosen expert runs). LBS is the OPPOSITE regime: SOFT, DENSE, FIXED (every bone contributes, painted weights
form a partition of unity, no learning, no winner-take-all). So the moe connection is real but CONCEPTUAL, not a
literal call: skinning is the soft/dense cousin of the engine's hard/sparse GatedMixture. Same experts+gating
skeleton, different gating regime. (Contrast FWD-10, where the iterate-a-projection reuse WAS literal.)

WHAT SHIPPED (holographic_meshskin.py; additive; one UnifiedMind faculty skin_mesh):
  * linear_blend_skin(vertices, transforms, weights) -- v' = sum_b w_b (M_b v); weights row-normalised. (V,3) out.
  * skin_mesh(mesh, transforms, weights) -- same, returns a new Mesh (deformed vertices, faces untouched).
  * make_transform / rotation(axis,angle) -- build the 4x4 bone transforms (Rodrigues rotation + translation).

MEASURED (the bar):
  * RIGID REPRODUCTION (the partition-of-unity guarantee, LBS's analogue of subdivision's affine reproduction):
    if every bone shares one rigid transform M, LBS reproduces M EXACTLY (1e-12) on every vertex for ANY weights.
  * Single-bone (weight 1) vertex = exactly that bone's transform; identity transforms leave the mesh fixed;
    translation interpolation is the weighted midpoint; skin_mesh leaves faces untouched. Deterministic.

THE KEPT NEGATIVE, MEASURED TO CLOSED FORM (the point of the module): LBS averages the bone MATRICES, not the
rotations, so a vertex blended 50/50 across a large relative TWIST collapses toward the bone axis (the infamous
"candy-wrapper" artifact). Exact, not vague: a unit ring twisted by theta has blended radius EXACTLY |cos(theta/2)|
-- 0.5 at 120 degrees, 0.000 (full collapse) at 180. The test asserts that closed form at several angles.
Dual-quaternion skinning fixes it by blending rotations properly -- the honest next step, not shipped.

Tests: +10 (1293 -> 1303). test_holographic_meshskin.py (+9): shared rigid transform reproduced exactly for any
weights; identity fixed; single-bone exact; unnormalized weights treated as partition of unity; translation
interpolation; candy-wrapper = cos(theta/2) at several angles; full collapse at 180; skin_mesh preserves faces;
determinism. test_integration.py (+1): skinning through the mind -- rigid reproduction + faces preserved + the
candy-wrapper closed form end-to-end. Files: holographic_meshskin.py (new), test_holographic_meshskin.py (new),
holographic_unified.py (1 faculty skin_mesh), test_integration.py, README, NOTES_concepts.md, tour.py.

TIER 2 CORE COMPLETE: FWD-7 core (extrude/inset/dissolve), FWD-8 subdivision, FWD-9 skinning, FWD-10 IK -- the
rig (skeleton+IK) -> skin animation pipeline. The honest reuse ledger across Tier 2: IK's iterate-a-projection was
LITERAL; subdivision's spectral low-pass and skinning's moe-mixture were CONCEPTUAL cousins (named precisely, not
overclaimed). NEXT: FWD-7 remainder (bevel/bridge/loop-cut); Tier 3 FWD-11 (mesh<->SDF<->splat bridge); ARCH items
(ARCH-4 atlas/seams -> a real FWD-3 seam; ARCH-1 StructureRecipe validator+edit-ops mirroring the Euler operators).


--------------------------------------------------------------------------------
FWD-11 -- Tier 3, the highest-value item: the mesh <-> SDF <-> splat bridge. A surface can be carried three ways
-- explicit MESH (verts+faces), implicit SDF (a scalar field, negative inside, zero level-set = the surface), or
SPLAT field (a superposition of Gaussians, holographic_splat). Same geometry, three costumes -- the project's
recurring thesis. This is the bridge that converts between them and measures the round-trip.

THE GENUINELY NEW PIECE: isosurface extraction (SDF -> mesh). The mesh kernel's own header says "no marching
cubes" -- so extracting a mesh from an implicit field was the one missing direction. Supplied here via MARCHING
TETRAHEDRA (not cubes) on purpose: a tiny unambiguous case set (per tet 0/1/2 triangles by how many of 4 corners
are inside) vs marching cubes' 256 cases + ambiguous faces; and MANIFOLD BY CONSTRUCTION (a crossing lives on a
grid edge shared by every tet touching it, welded by edge identity; the tet's quad split is interior, so adjacent
patches always agree -- no cracks). Cube split into 6 tets sharing a main diagonal (Kuhn decomposition).

WHAT SHIPPED (holographic_meshbridge.py; additive; two UnifiedMind faculties):
  * mesh_from_sdf(sdf, bounds, res, level) [faculty] / marching_tetrahedra(values, axes, level) -- extract the
    level-set isosurface of a sampled field as a watertight, OUTWARD-oriented triangle Mesh. The bridge's core.
  * mesh_to_sdf(mesh, points) [faculty] -- signed distance from a mesh (vectorised closest-point-on-triangle,
    Ericson's region test; sign from the nearest face normal). The reverse direction.
  * sample_field / sphere_sdf / metaball_field -- grid sampler, analytic sphere SDF, and the splat-as-implicit
    Gaussian sum (a bundle of Gaussians thresholded is an isosurface).

MEASURED (the bar, against analytic references):
  * SDF -> mesh: the analytic unit sphere extracts to a CLOSED MANIFOLD (chi=2), 100% OUTWARD-oriented faces,
    vertices on the sphere (mean r=0.999 +/- 0.001). A radius-0.7 sphere -> r=0.699 +/- 0.001. Resolution scales
    (res 12/20/28 -> 1536/4392/9216 faces).
  * mesh -> SDF: a sphere mesh's signed distance matches analytic |p|-1 at probes (<0.05), correct inside/outside
    sign (origin negative, far point positive).
  * SPLAT -> mesh: a sum of Gaussian splats (metaball field) iso-extracts to a closed-manifold blob -- the splat
    representation entering the mesh world through the SAME extractor. Deterministic.

KEPT NEGATIVES (loud):
  * mesh_to_sdf signs by the NEAREST FACE NORMAL -- exact for convex-ish closed meshes, can mis-sign deep
    concavities or thin sheets (generalized winding number is the fix, not shipped). The magnitude is always right.
  * Marching-tet resolution is the grid's: sharp features below the cell size are rounded; the round-trip recovers
    the SHAPE to grid resolution, not the original connectivity.
  * It emits edge-welded triangle soup (no triangle-quality guarantee) -- a downstream Taubin smooth/remesh
    (FWD-4) is the cleanup, which is exactly why those faculties exist.

Tests: +10 (1303 -> 1313). test_holographic_meshbridge.py (+9): SDF->mesh closed manifold sphere; vertices on
sphere; radius scales; 100% outward orientation; resolution scaling; mesh->SDF matches analytic; sign correct;
splat->mesh closed blob; determinism. test_integration.py (+1): the full bridge through the mind (SDF->mesh,
mesh->SDF, splat->mesh) end-to-end. Files: holographic_meshbridge.py (new), test_holographic_meshbridge.py (new),
holographic_unified.py (2 faculties), test_integration.py, README, NOTES_concepts.md, tour.py.

TIER 3 OPENED with the bridge. The FWD backlog is now: Tier 1 (FWD-3/4/5/6) DONE; Tier 2 core (FWD-7 core, FWD-8,
FWD-9, FWD-10) DONE; Tier 3 FWD-11 DONE. REMAINING: FWD-7 remainder (bevel/bridge/loop-cut); ARCH items (ARCH-4
atlas/seams -> a real FWD-3 seam; ARCH-1 StructureRecipe validator+edit-ops mirroring the Euler operators; ARCH-3
geometry-weighted graph ops; etc.). The mesh DCC suite is now broadly complete end to end.


--------------------------------------------------------------------------------
ARCH-1 -- the first §ARCH item: turn the 3-D DCC concepts INWARD on the engine's own structures. FWD-7 gave the
MESH its local invariant-preserving editors (the Euler operators: flip/split/collapse, each preserving chi + the
manifold). ARCH-1 is the exact mirror for the StructureRecipe (the one build-graph program/tree/scene all reduce
to, B7): a VALIDATOR (check well-formedness -- the recipe's is_manifold) + EDIT OPERATORS that rewrite a recipe
while preserving its meaning.

THE PARALLEL (the point): a mesh Euler op preserves a topological invariant; a recipe edit op preserves the
REALIZED VECTOR -- for the SAME reason: it is a local rewrite that is an IDENTITY of the underlying algebra. bind
is circular convolution (commutative); bundle/superpose are sums (commutative). So:
  * commute_bind  -- bind(a,b)=bind(b,a)            <-> flip_edge (its own inverse, preserves the invariant)
  * reorder_members -- bundle(any order) is equal   <-> a parameterised flip (invertible by the inverse perm)
  * substitute_atom -- rename a leaf                 <-> a vertex-position move (structure fixed, result changes, reversible)

WHAT SHIPPED (holographic_recipeops.py; additive; four UnifiedMind faculties):
  * validate(recipe) [faculty validate_recipe] -> (ok, problems) -- every op references only EARLIER existing
    results (DAG, no forward/dangling/out-of-range refs), raw indices + repeat templates in range.
  * commute_bind(recipe, handle) [recipe_commute_bind] -- swap a bind's args. Vector-preserving, OWN INVERSE.
  * reorder_members(recipe, handle, perm) [recipe_reorder_members] -- permute a bundle/superpose's members.
    Vector-preserving, invertible by the inverse perm.
  * substitute_atom(recipe, handle, new_name) [recipe_substitute_atom] -- rename an atom leaf. Validity-preserving,
    result changes predictably, invertible by renaming back.
  Each returns a NEW recipe (originals untouched, as the mesh operators returned new meshes). _op_index_for_handle
  maps an absolute result handle to its op position (repeat produces several results, so it's not just `handle`).

MEASURED (the bar):
  * validate ACCEPTS a well-formed recipe and REJECTS a corrupted one (a forward/out-of-range reference, a bad raw
    index) -- with human-readable problems.
  * commute_bind + reorder_members leave the realized vector BIT-EXACT to FFT precision (1e-12) and the recipe
    valid; commute_bind applied twice literally restores the op (own inverse); reorder undone by the inverse perm.
  * substitute_atom CHANGES the realized vector and reverses EXACTLY by substituting the original name back.
  * Edits don't mutate the original recipe; deterministic.

KEPT NEGATIVES (loud):
  * These are the VECTOR-PRESERVING / structure-preserving edits (the recipe's Euler-operator CORE). Edits that
    REMOVE/RESIZE ops (flatten a nested superpose, splice out dead results) require re-indexing every downstream
    handle -- the recipe analogue of the mesh face-list reindex in collapse/dissolve -- and are deferred; the
    in-place edits are correct and complete on their own (as flip_edge is).
  * "bit-exact" is up to FFT/float round-off (~1e-12), an algebraic identity (FP-equal not literally bit-equal) --
    the same honest caveat the bind_batch vectorization carries.

Tests: +15 (1313 -> 1328). test_holographic_recipeops.py (+14): validate accepts/rejects (forward ref, bad raw);
commute_bind preserves vector + own inverse + rejects non-bind; reorder preserves vector + inverts + rejects
non-permutation; substitute_atom changes + reverses; edits keep validity; edits don't mutate the original;
determinism. test_integration.py (+1): the recipe editors through the mind end-to-end. Files:
holographic_recipeops.py (new), test_holographic_recipeops.py (new), holographic_unified.py (4 faculties),
test_integration.py, README, NOTES_concepts.md, tour.py. Faculty count -> 250 (round milestone). NEXT §ARCH:
ARCH-4 atlas/seams (-> a real FWD-3 seam); ARCH-3 geometry-weighted graph ops; ARCH-5 subdivision-for-structures;
ARCH-6 rig+IK-for-structures; ARCH-7 representation routing. Plus FWD-7 remainder (bevel/bridge/loop-cut).


--------------------------------------------------------------------------------
ARCH-4 -- seam cutting / atlas: opening a closed surface into a disk by vertex duplication. THE FWD-3 PAYBACK.
FWD-3 (UV unwrap) shipped with a kept negative -- a CLOSED surface needs a CUT to flatten, and its only opener was
`puncture` (delete a vertex, leaving a tiny hole that unwraps badly). ARCH-4 supplies the real thing: cut along a
SEAM (an edge path) by DUPLICATING the seam's interior vertices, opening the surface into a disk that keeps ALL
its geometry.

THE SUBTLE PART (why FWD-3 deferred it): a seam arc does NOT separate the surface, so you cannot 2-colour faces
left/right globally -- the sides are LOCAL. Fix: ORIENT the seam (v0->...->vk) and at each interior vertex
duplicate the fan on the side matching the path direction (the fan containing the face carrying directed edge
v_i->v_{i+1}). That side is defined by the single path orientation, so the duplicated side is consistent all along
the seam and the two lips line up -> a manifold. (Get this wrong -> non-manifold mess.)

WHAT SHIPPED (holographic_meshseam.py; additive; two UnifiedMind faculties):
  * cut_seam(mesh, seam) [mesh_cut_seam] -- cut open along an ordered vertex path, duplicating interior seam
    vertices on a consistent side. Returns a new (open) Mesh. _components_of_incident_faces splits a vertex's
    umbrella into its two fans (faces sharing a non-seam edge through the vertex); _face_with_directed_edge
    picks the consistent side.
  * shortest_seam(mesh, a, b) [mesh_shortest_seam] -- shortest edge path (Dijkstra), e.g. a meridian.
  * _boundary_loop_count -- verify the cut made exactly one boundary (a disk).

TOPOLOGY: cutting a closed genus-0 surface (chi=2) along a simple arc of k edges duplicates its k-1 interior
vertices and splits each of k seam edges into two -> dchi = (k-1)-k = -1: chi 2->1, a DISK.

MEASURED (the bar):
  * cut_seam(sphere, meridian) -> a DISK: chi=1, NOT closed, manifold, exactly ONE boundary loop, V grown by
    (interior seam vertices = 15 for the icosphere meridian).
  * ROBUST PAYBACK (always true): the cut is NON-DESTRUCTIVE -- preserves all 512 faces; the puncture DELETES 4
    faces (loses geometry). A real seam keeps the whole surface.
  * DISTORTION PAYBACK (good seam): a pole-to-equator seam unwraps at 0.405 < the puncture's 0.507.
  * Deterministic.

KEPT NEGATIVE (measured, loud): SEAM CHOICE MATTERS. A FULL pole-to-pole meridian opens a valid disk but unwraps
WORSE than the puncture (0.579 > 0.507) -- it makes a long thin lune. One cut never makes a sphere unwrap WELL
(Gauss); a good atlas uses several cuts / multiple charts (the rest of ARCH-4, deferred). The win is
"non-destructive, and beats the puncture with a sensible seam", not "distortion-free". The first-draft assumed any
meridian beats the puncture -- WRONG (the full meridian doesn't); the honest finding (seam-dependent) is kept.

Tests: +10 (1328 -> 1338). test_holographic_meshseam.py (+9): cut opens to a disk (chi=1, manifold, open); one
boundary loop; interior vertices duplicated; preserves every face; puncture deletes faces but cut doesn't;
well-chosen seam beats puncture distortion; full meridian is worse (the kept negative); shortest_seam is a valid
edge path; determinism. test_integration.py (+1): seam cutting through the mind (disk, non-destructive, beats
puncture) end-to-end. Files: holographic_meshseam.py (new), test_holographic_meshseam.py (new),
holographic_unified.py (2 faculties), test_integration.py, README, NOTES_concepts.md, tour.py. (Note: edges()
yields sorted TUPLES not frozensets -- a test-only normalisation fix, no code change.) NEXT §ARCH: ARCH-3
(geometry-weighted graph ops), ARCH-5 (subdivision-for-structures), ARCH-6 (rig+IK-for-structures), ARCH-7
(representation routing). Plus FWD-7 remainder (bevel/bridge/loop-cut).


--------------------------------------------------------------------------------
ARCH-7 -- representation routing: the POLICY layer on top of FWD-11's mesh<->SDF<->splat bridge. FWD-11 built the
conversions; ARCH-7 decides WHEN to use them. Different operations are natural in different representations, so
route to the one that makes an operation easy, do it there, convert back. (Same shape as the decode-vs-evaluate
principle for vectors -- use the representation the operation actually fits.)

THE FLAGSHIP: CSG (constructive solid geometry). Boolean union/intersection/difference have NO native mesh
implementation (robust mesh booleans need surface-surface intersection, never built). On an SDF they are trivial
exact FIELD ops: union=min(dA,dB), intersection=max(dA,dB), difference=max(dA,-dB). So the router takes meshes ->
SDF (mesh_to_sdf) -> combine fields -> extract back to mesh (marching tetrahedra, FWD-11). Crucially this lets a
boolean CHANGE TOPOLOGY -- two separate spheres become ONE blob when overlapping, stay TWO when not -- which a mesh
cannot do to itself; the field merges/keeps-separate automatically.

WHAT SHIPPED (holographic_route.py; additive; four UnifiedMind faculties):
  * REPRESENTATION_CAPABILITIES -- the routing table (which ops each representation supports: sdf owns
    booleans/inside_test/offset, mesh owns boundary/render/subdivide, splat owns blend/scatter).
  * representation_for(op) [route_representation] -- the routing decision.
  * route_csg(op, A, B, res, bounds) [mesh_csg] -- the flagship boolean via SDF routing. Returns a Mesh.
  * connected_components(mesh) [mesh_connected_components], mesh_volume(mesh) [mesh_volume] -- the measurements.

MEASURED (the bar):
  * the table sends booleans -> "sdf" and boundary/render -> "mesh"; "union" is explicitly NOT a mesh capability
    (that is WHY routing exists).
  * OVERLAPPING spheres: union merges to ONE connected component, a closed manifold (topology merged). SEPARATE
    spheres: union stays TWO components (separation preserved).
  * GEOMETRICALLY correct, not just topologically -- inclusion-exclusion holds to a few percent: vol(uni) 6.50 ~
    vA+vB-vInt 6.55; vA 3.82 ~ vInt+vDiff 3.77.
  * Deterministic.

KEPT NEGATIVES (loud):
  * Resolution is the grid's (FWD-11 inherited): sharp intersection seams round at the cell size. Volumes converge
    to the truth FROM BELOW (marching-tet under-fills) -- hence the inclusion-exclusion checks carry a few-percent
    tolerance, not machine precision.
  * route_csg trusts mesh_to_sdf's sign, reliable for convex-ish inputs but mis-signs deep concavities in an INPUT
    mesh; the spheres are convex so the combined field is exact. A non-convex input needs a winding-number sign
    (the FWD-11 fix, deferred).
  * The table is a small curated policy (published strengths), not a learned cost model.

Tests: +14 (1338 -> 1352). test_holographic_route.py (+13): table routes booleans->sdf + boundary->mesh; union not
a mesh capability; unknown op raises; overlapping union -> 1 component; union is closed manifold; separate union ->
2 components; intersection smaller than inputs; difference smaller than minuend; inclusion-exclusion for union;
intersection+difference recovers A; components of a single sphere = 1; determinism. test_integration.py (+1): CSG
routing through the mind (policy + merged-topology union + inclusion-exclusion). Files: holographic_route.py (new),
test_holographic_route.py (new), holographic_unified.py (4 faculties), test_integration.py, README,
NOTES_concepts.md, tour.py. Faculty count -> 256. NEXT §ARCH: ARCH-3 (geometry-weighted graph ops), ARCH-5
(subdivision-for-structures), ARCH-6 (rig+IK-for-structures). Plus FWD-7 remainder (bevel/bridge/loop-cut).


--------------------------------------------------------------------------------
ARCH-3 -- geometry-weighted graph operations on hypervectors: the COTANGENT LAPLACIAN, turned inward. On a mesh
(FWD-4) the cotangent Laplacian weights edges by the actual geometry (angles) and respects the shape where uniform
combinatorial weights distort it. The engine's graphs (knn_adjacency over stored vectors) are BINARY (every edge
1). The natural geometry of the hypervector world is COSINE SIMILARITY, so a similarity-WEIGHTED graph is the
cotangent analogue.

WHAT SHIPPED (holographic_simgraph.py; additive; three UnifiedMind faculties):
  * similarity_adjacency(vectors, k, weighted) [similarity_graph] -- a kNN graph; weighted=True -> each edge carries
    the cosine similarity (the geometry), weighted=False -> the engine's existing BINARY kNN graph (reused verbatim).
  * spectral_embedding(vectors, k, dims, weighted) [graph_spectral_embedding] -- Laplacian eigenmaps (low
    eigenvectors of the weighted graph Laplacian) via holographic_spectral's graph_laplacian/laplacian_eigenbasis.
  * ring_order(vectors, k, weighted) [graph_ring_order] -- recovered cyclic coordinate atan2(e2,e1) for ring points.

MEASURED (the bar):
  * POSITIVE (clean): the weighted similarity-graph eigenmap RECOVERS a ring -- recovered cyclic order tracks the
    true angle to |corr|=0.998 from high-D hypervectors. The geometry-weighted op recovers intrinsic manifold
    structure.
  * WHERE WEIGHTING WINS: under NON-UNIFORM sampling (points bunched into arcs) the weighted graph recovers the ring
    BETTER than binary (0.917 > 0.812 at seed 0; weighted wins 6/6 seeds) -- the cosine weighting corrects sampling
    density, exactly as the cotangent Laplacian corrects an irregular mesh.
  * weighted adjacency entries ARE the cosine similarities (varying, in (0,1]); binary entries are all 1.
  * Deterministic.

KEPT NEGATIVES (loud, measured -- the honest headline):
  * Under UNIFORM sampling / well-separated data, similarity-weighting and the BINARY graph essentially TIE (ring
    recovery 0.998 either way). This is a REAL difference from the mesh: a mesh's edge LENGTHS vary by orders of
    magnitude so cotangent-vs-uniform differs sharply, but in high dimension the CONCENTRATION OF MEASURE makes a
    kNN graph's edges nearly equal in strength, so weighting has little to correct. Geometry-weighting here helps
    most under IRREGULAR SAMPLING, not universally.
  * Downstream tasks the mesh weighting would help (cluster label propagation, vector denoising by graph smoothing)
    showed NO weighted-over-binary gain on well-separated high-D clusters in this engine, same concentration reason
    -- measured during development and kept; the module ships the operations + the regime where weighting
    demonstrably helps (irregular sampling on a continuous manifold), not an overclaim that weighting always wins.

Tests: +10 (1352 -> 1362). test_holographic_simgraph.py (+9): weighted eigenmap recovers a ring; weighted edges
carry varying similarities; binary edges all 1; adjacency symmetric; weighting wins under non-uniform sampling;
weighting ties binary under uniform sampling (kept negative); embedding shape; ring_order length; determinism.
test_integration.py (+1): geometry-weighted graph through the mind (ring recovery + weights + non-uniform win).
Files: holographic_simgraph.py (new), test_holographic_simgraph.py (new), holographic_unified.py (3 faculties),
test_integration.py, README, NOTES_concepts.md, tour.py. Faculty count -> 259. NEXT §ARCH: ARCH-5
(subdivision-for-structures, mirrors FWD-8 inward), ARCH-6 (rig+IK-for-structures, mirrors FWD-9/10 inward). Plus
FWD-7 remainder (bevel/bridge/loop-cut).


--------------------------------------------------------------------------------
ARCH-5 -- subdivision curves on hypervector sequences: FWD-8's Loop subdivision, turned inward onto a 1-manifold.
FWD-8 subdivided a MESH (2-manifold): refine (1 tri -> 4) + low-pass smooth toward a limit surface. ARCH-5 does the
same to the engine's own 1-D structure -- a SEQUENCE of hypervectors is a polyline through vector space (what the
sequence faculties encode) -- via CHAIKIN corner-cutting (the curve analogue of Loop, generator of a quadratic
B-spline limit): each edge (p_i,p_{i+1}) -> (3/4 p_i + 1/4 p_{i+1}, 1/4 p_i + 3/4 p_{i+1}), which both REFINES
(doubles the count) and SMOOTHS (corner-cutting is a low-pass filter).

THE MESH PROPERTIES MAP ACROSS EXACTLY:
  Loop faces x4/level            <-> Chaikin points x2/level
  Loop flat-stays-flat (affine)  <-> Chaikin straight-line-of-vectors-stays-straight (affine)
  Loop -> limit surface          <-> Chaikin -> limit curve
  Loop dihedral-spread shrinks   <-> Chaikin roughness (2nd-diffs) shrinks

WHAT SHIPPED (holographic_subdivcurve.py; additive; one UnifiedMind faculty):
  * chaikin_subdivide(points, closed) -- one level of corner-cutting.
  * subdivide_sequence(points, levels, closed) [subdivide_sequence] -- `levels` of Chaikin on a vector sequence;
    returns the refined (M,dim) sequence.

MEASURED (the bar):
  * REFINE: open polyline n -> 2(n-1)/level ([6,10,18,34]); closed -> 2n/level ([6,12,24,48]).
  * AFFINE REPRODUCTION: a straight line of vectors (linear ramp) stays ON the line to 2e-15 -- the exact analogue
    of FWD-8's "flat stays flat".
  * CONVERGENCE: curve length deltas [21.9,5.7,2.2,0.9,0.4] shrink (approaches a limit curve).
  * LOW-PASS: a zig-zag's roughness [128,16,2,0.25,0.03] shrinks ~8x/level (corner cutting removes high freqs).
  * Deterministic.

KEPT NEGATIVE (loud): Chaikin is APPROXIMATING, not interpolating -- the limit curve cuts the original control
points' corners and does NOT pass through interior control points (nearest 0.25, not ~0). This is the EXACT mirror
of FWD-8's negative (Loop approximates -> a curved icosphere smooths to Loop's own limit, not the exact sphere). An
INTERPOLATING scheme (Dyn-Levin-Gregory 4-point) keeps the control points but is less smooth and needs >=4 points
with boundary special-casing -- the classic approximating/interpolating trade-off, deferred. Also: the open scheme
cuts the end corners too (first/last control points not preserved; an endpoint-preserving boundary rule is separate).

Tests: +10 (1362 -> 1372). test_holographic_subdivcurve.py (+9): open/closed counts double; single-level count;
straight line stays straight; curve length converges; zig-zag roughness shrinks; approximating (control points not
interpolated); short sequence unchanged; determinism. test_integration.py (+1): subdivide_sequence through the mind
(refine + affine + low-pass). Files: holographic_subdivcurve.py (new), test_holographic_subdivcurve.py (new),
holographic_unified.py (1 faculty), test_integration.py, README, NOTES_concepts.md, tour.py. Faculty count -> 260
(round milestone). NEXT §ARCH: ARCH-6 (rig+IK-for-structures, mirrors FWD-9/10 inward) -- the last §ARCH item. Plus
FWD-7 remainder (bevel/bridge/loop-cut).


--------------------------------------------------------------------------------
ARCH-6 -- rig + IK for STRUCTURES via blendshape posing: FWD-9 (linear blend skinning) + FWD-10 (IK) turned inward.
The LAST inward mirror, and it closes the §ARCH block. A "rig" is a set of pose-TARGET structures (blendshapes)
p_1..p_m; a pose is a soft weighted blend pose(w)=normalize(sum w_i p_i). The two halves of FWD-9/10 map across:
  * FORWARD = SKINNING (FWD-9): given weights, the pose is the blend -- FWD-9's soft mixture of bone transforms,
    one rung up (mixing whole structures, not transforms).
  * INVERSE = IK (FWD-10): given a GOAL structure, SOLVE the blend weights to reach it -- via the SAME
    project_onto_constraints sweeper FWD-10 used for FABRIK. The "joint angles" are the weights; the swept
    constraints are FIT-the-goal (least-squares gradient step = FABRIK's reach) + VALID-CONVEX-BLEND (simplex
    projection = FABRIK's bone-length projection). Literal reuse, distinct from 3-D mesh IK (this is IK in the
    engine's semantic vector space).

WHAT SHIPPED (holographic_blendpose.py; additive; two UnifiedMind faculties):
  * blend_pose(targets, weights) [blend_pose] -- forward skinning/blendshape map: normalize(sum w_i targets_i).
  * solve_pose(targets, goal, iters) [solve_pose] -- IK: solve the blend weights via project_onto_constraints
    ([fit, simplex_project]). Returns a valid convex blend. _simplex_project = Duchi et al. 2008 simplex projection.

MEASURED (the bar):
  * FORWARD: a one-hot weight reproduces that target exactly (1e-12); a mix leans toward its targets.
  * IK REACHABLE: goal IS a known interior blend -> recovers the weights (L1 err 0.000) and the achieved pose
    matches the goal (residual 1e-15). The analogue of FWD-10 hitting a reachable target exactly.
  * IK UNREACHABLE: random goal outside the span -> CLOSEST valid blend (residual 18.94 <= best single target 22.37,
    a GUARANTEE: the simplex it searches contains every vertex) but cannot reach (residual > 1). The analogue of
    FWD-10's chain fully extending toward an out-of-reach target.
  * solved weights are always a valid convex blend (w>=0, sum=1).
  * Deterministic.

CRITICAL IMPLEMENTATION NOTE (kept): the least-squares step size MUST come from the Lipschitz constant (largest
eigenvalue of the Gram P P^T, ~dim for random targets); a fixed step diverges and the simplex projection collapses
to a vertex (the bug found in the probe). mu = 1/L fixes it -> exact recovery.

KEPT NEGATIVES (loud):
  * The IK CANNOT reach a goal outside the targets' convex blend -- returns the closest valid blend (the honest
    analogue of FWD-10's unreachable target). Reaching arbitrary goals needs a richer rig (more targets), not a
    better solver.
  * Returns A best convex blend, not THE only one: linearly-dependent targets -> non-unique weights (the POSE is
    still optimal, the weights just aren't identifiable) -- the same "a-solution-not-the-solution" caveat as FWD-10.
  * Forward map is a blend in the AMBIENT vector space (FWD-9's linear-blend analogue), not a nonlinear pose
    manifold -- the same scope as linear blend skinning (whose own negative was the candy-wrapper collapse).

Tests: +10 (1372 -> 1382). test_holographic_blendpose.py (+9): one-hot blend is that target; mix leans to targets;
IK recovers a reachable blend; reachable pose matches goal; unreachable is closest valid blend; unreachable cannot
reach; solved weights valid simplex; simplex projection lands on simplex; determinism. test_integration.py (+1):
blendshape posing through the mind (forward + reachable IK + closest-blend). Files: holographic_blendpose.py (new),
test_holographic_blendpose.py (new), holographic_unified.py (2 faculties), test_integration.py, README,
NOTES_concepts.md, tour.py. Faculty count -> 262.

*** §ARCH BLOCK COMPLETE: ARCH-1 (recipe Euler ops), ARCH-2 (delta protocol, prior session), ARCH-3 (geometry-
weighted graph ops), ARCH-4 (real seam), ARCH-5 (subdivision curves), ARCH-6 (rig+IK for structures), ARCH-7
(representation routing) all shipped. Each turned a piece of the FWD mesh pipeline inward onto the engine's own
structures. *** NEXT: FWD-7 modeler remainder (bevel/bridge/loop-cut) is the main remaining FWD thread.


--------------------------------------------------------------------------------
FWD-7 REMAINDER -- bevel, bridge, loop-cut: the three modeler verbs FWD-7 deferred because they need vertex
DUPLICATION or edge-loop TRACING (FWD-7 shipped the face-list-rewrite verbs extrude/inset/dissolve). This ships
them, reusing the vertex-fan/umbrella logic from the ARCH-4 seam and the unused-vertex compaction (reindex) from
dissolve.

WHAT SHIPPED (holographic_meshverbs2.py; additive; three UnifiedMind faculties):
  * bevel_vertex(mesh, vertex, ratio) [mesh_bevel_vertex] -- chamfer a corner: pull each incident edge back toward
    its neighbour by ratio, chamfer every incident face, cap the hole with a new face. Needs the cyclic neighbour
    order (the umbrella) + compaction of the removed corner vertex.
  * bridge_loops(verts, loop_a, loop_b, closed) [mesh_bridge] -- join two equal-length ordered vertex loops with a
    band of quads (build a tube between two openings).
  * loop_cut(mesh, start_face, start_edge) [mesh_loop_cut] -- trace the perpendicular quad loop (enter a quad
    through one edge, leave through the OPPOSITE edge, cross to the neighbour) and split every crossed quad in two.

MEASURED (the bar):
  * BEVEL a cube corner (degree 3): closed manifold, chi PRESERVED (2); the 3 incident quads become PENTAGONS and a
    TRIANGULAR cap appears (face sizes [3,4,4,4,5,5,5]); V = 8-1+3 (corner removed, 3 new). New verts sit on the
    incident edges at the ratio.
  * BRIDGE two squares -> an open tube: 4 quads, chi=0, exactly TWO boundary loops, manifold.
  * LOOP-CUT a cube: closed manifold, chi PRESERVED (2), +4 faces (the ring crosses 4 quads). LOOP-CUT a grid(3,3):
    chi PRESERVED (1), +3 faces (the open strip crosses 3 quads).
  * Deterministic.

TWO BUGS FOUND IN THE PROBE (kept as notes): (1) loop-cut first split quads after reordering the cycle to start at
an arbitrary vertex -> inconsistent winding across the strip -> non-manifold ("directed edge appears twice"). FIX:
split using the quad's OWN native cyclic order (v0,v1,v2,v3 from the entering-edge position) so adjacent cells wind
oppositely on the shared cut edge. (2) bevel left the removed corner vertex ORPHANED in the array -> chi wrong (3
not 2). FIX: _compact drops unused vertices and reindexes (the dissolve/seam reindex).

KEPT NEGATIVES (loud):
  * BEVEL is the VERTEX bevel (chamfer a corner). The EDGE bevel (widen an edge into a chamfer face, splitting BOTH
    endpoints' fans) is the harder two-sided split, deferred -- the same fan-consistency the seam solved for one
    path, here needed on both sides. ratio must be in (0,1); boundary/non-manifold vertices out of scope.
  * BRIDGE requires two EQUAL-LENGTH, ALREADY-ALIGNED loops (caller supplies the correspondence); resampling/matching
    unequal loops (the general bridge) is deferred.
  * LOOP-CUT needs QUADS (the opposite-edge trace is undefined on triangles); the trace stops at a boundary (open
    cut) or when it returns to the start (closed ring).

Tests: +12 (1382 -> 1394). test_holographic_meshverbs2.py (+11): bevel closed-manifold + chi preserved; bevel
pentagons+cap; bevel vertex count; bevel new verts near corner; bridge open tube; bridge two boundary loops; bridge
unequal loops raises; loop-cut box chi+4faces; loop-cut grid chi+3faces; loop-cut on triangle raises; determinism.
test_integration.py (+1): all three verbs through the mind. Files: holographic_meshverbs2.py (new),
test_holographic_meshverbs2.py (new), holographic_unified.py (3 faculties), test_integration.py, README,
NOTES_concepts.md, tour.py. Faculty count -> 265.

*** With this, the FWD direct-modeling verb set is complete: extrude/inset/dissolve (FWD-7) + bevel/bridge/loop-cut
(remainder). The broad FWD DCC pipeline + the full §ARCH inward-mirror block are both done. ***


--------------------------------------------------------------------------------
HOLOGRAPHIC SCENE-GRAPH ALGEBRA -- the capstone that joins the FWD mesh kernel to the ARCH-1 recipe algebra. A
scene graph (leaves are meshes, edges are transforms) read TWO ways at once: as GEOMETRY (instance + merge) and as
STRUCTURE (encode to a StructureRecipe). The point is that the two views are CONSISTENT -- VSA is geometry, and the
scene is one object wearing both costumes.

WHAT SHIPPED (holographic_scenegraph.py; additive; seven UnifiedMind faculties):
  * SceneNode(transform, mesh, children) [scene_graph] -- a node: a 4x4 transform, an optional leaf mesh, optional
    children.
  * identity/translation/scaling/rotation/compose_transforms [scene_translation/scene_scaling/scene_rotation/
    scene_compose_transforms] -- 4x4 transform builders (rotation is Rodrigues).
  * flatten_scene(node) [scene_flatten] -- the GEOMETRY view: instance every leaf through its accumulated transform
    (parent transforms composed down the graph) and MERGE into one Mesh.
  * scene_to_recipe(node, dim, seed) [scene_to_recipe] -- the STRUCTURE view: encode as a StructureRecipe
    (transforms BOUND to content via bind, siblings BUNDLED), realising to one hypervector. Leaf/transform atom
    names are content hashes (hashlib).

THE CONSISTENCY THEOREM (the unification, measured): swapping two siblings leaves the flattened GEOMETRY identical
(same sorted vertices, same face count -- a mesh merge is commutative) AND the realised VECTOR identical (bundle is
commutative). So a structural edit from ARCH-1 (recipe_reorder_members on the sibling bundle) is a no-op on the
geometry too -- the two representations agree. The scene recipe is a WELL-FORMED recipe (passes ARCH-1's validate),
so the recipe Euler operators apply to scenes.

MEASURED (the bar): INSTANCING -- a scene of 2 cubes flattens to one mesh V=16 F=12, the +x instance lands with its
centroid at its translation; NESTED transforms compose to +2 (parent then child); CONSISTENCY -- sibling swap
leaves geometry AND vector identical; distinct transforms -> distinct structure vectors; the scene is a valid
recipe; deterministic (same scene -> byte-identical mesh and vector).

KEPT NEGATIVES (loud):
  * flatten_scene INSTANCES and concatenates -- it does NOT weld coincident vertices or boolean-merge overlapping
    geometry (that is mesh_csg / ARCH-7's job); two touching cubes flatten to two components, not one solid. This is
    scene assembly, not constructive solid geometry.
  * scene_to_recipe encodes the scene's STRUCTURE (which transform holds which content), not its geometry -- the
    mesh hash distinguishes meshes but the vector is a structural index, and recovering geometry is scene_flatten's
    job, not the vector's.
  * the encoding bundles siblings, so (decode ceiling) a node with very many children loses per-child
    recoverability from the root vector -- the same capacity cliff every bundle carries; wide scenes index
    structurally but are not meant to be decoded child-by-child from the root.

Tests: +14 (1394 -> 1408). test_holographic_scenegraph.py (+13): transform builders (translation/rotation/scaling/
compose); instancing merges; instance lands at translation; nested transforms compose; identity node; sibling-swap
geometry identical; sibling-swap vector identical; valid recipe; distinct scenes -> distinct vectors; determinism.
test_integration.py (+1): the full scene-graph algebra through the mind. Files: holographic_scenegraph.py (new),
test_holographic_scenegraph.py (new), holographic_unified.py (7 faculties), test_integration.py, README,
NOTES_concepts.md, tour.py. Faculty count -> 272.

*** This is the geometry capstone: the FWD mesh pipeline + the §ARCH inward mirror now meet in one object -- a scene
that is simultaneously a pile of triangles and a composed hypervector, with the two provably consistent. The
standing thesis (VSA is geometry) made concrete: the scene graph IS the recipe. ***


--------------------------------------------------------------------------------
QEM DECIMATION -- the quadric error metric (Garland-Heckbert, SIGGRAPH 1997), the one genuinely-missing piece of a
principled mesh simplifier. From the geometry->stack backlog sweep: the engine already had the guarded
collapse_edge (eulerops, the link-condition refusal made operational), the heapq priority-descent with
deterministic ties (HoloForest/Dijkstra), the curvature read-out (meshcurvature), and the greedy-by-error loop
shape (matching pursuit) -- ONLY the cost function (the quadric) was absent. This supplies it and wires it to the
shipped collapse.

WHAT SHIPPED (holographic_meshqem.py; additive; two UnifiedMind faculties):
  * vertex_quadrics(mesh) -- per-vertex 4x4 error quadrics Q_v = sum over incident faces of (plane plane^T),
    plane = [n, -n.p]; v^T Q_v v is the summed squared distance from v to its incident planes.
  * contraction_target(Q, p_i, p_j) -- the optimal merged position argmin v^T Q v and its cost; singular 3x3 ->
    best of {midpoint, endpoints}.
  * qem_decimate(mesh, target_faces) [mesh_qem_decimate] -- greedily collapse the lowest-cost edge (deterministic
    ties by vertex index) via the guarded collapse_edge, ACCUMULATING quadrics through each collapse (the survivor
    inherits Q_keep + Q_remove), until <= target_faces.
  * surface_deviation(mesh_a, mesh_b) [mesh_surface_deviation] -- (mean, max) point-to-surface distance (a
    decimation quality metric; uses the closed-form point-to-triangle distance).

WHY IT BELONGS IN THIS ENGINE (the reverse thesis, concrete): a quadric is Q_v = sum of (plane plane^T) -- an
OUTER-PRODUCT ACCUMULATION = a BUNDLE of plane constraints in matrix form, with the collapse cost read out as a
quadratic. That is bind/bundle/readout in a different costume. So QEM IS a general "merge the two items whose
combined representation loses the least" operator, which is exactly reverse item R2 (prototype compaction in the
creature -- combine redundant prototypes instead of evicting), and the same shape as the splat merge / codebook
merge. Build it once for meshes; it is the merge operator everywhere. (R2 is a future wire on this.)

MEASURED (the bar): icosphere V66 F128 -> QEM F64, closed manifold, chi PRESERVED (2); QEM BEATS a naive
shortest-edge->midpoint baseline on MEAN point-to-surface error (~1.8x) AND dramatically on MAX error (~3x -- naive
spikes where it collapses a feature edge); a vertex's own quadric vanishes at the vertex (it lies on its incident
planes); the cost is never negative; deterministic.

KEPT NEGATIVES (loud):
  * QEM minimizes squared distance to incident PLANES, not the true surface or any invariant -- so on a sphere the
    optimal points sit slightly OFF-RADIUS (|r-1| a touch worse than the chord-midpoint baseline) while being
    CLOSER to the actual surface (point-to-surface, the honest metric, is much better, esp. max). The plane metric
    is the right one; radius fidelity is not what it optimizes.
  * CLOSED meshes are in scope; OPEN-mesh boundary preservation (the standard high-weight perpendicular-plane
    penalty per boundary edge) is deferred.
  * The loop recomputes edge costs each pass (clear + correct); the incremental heap-with-lazy-deletion that makes
    QEM near-linear is the standard perf upgrade, deferred (the panel's "delegate the heavy grind" call) -- this is
    the readable, correct version for moderate meshes.
  * collapse_edge REFUSES manifold-breaking collapses (link condition); the decimator tries the next-cheapest edge
    and HALTS if no safe collapse remains -- so it may stop above target_faces. A true mesh property, operational.

Tests: +11 (1408 -> 1419). test_holographic_meshqem.py (+10): quadric vanishes at its vertex; quadric symmetric;
cost non-negative; singular -> midpoint fallback; decimate preserves closed manifold + chi; reaches target; QEM
beats naive mean error; QEM beats naive max error; surface_deviation zero for identical mesh; determinism.
test_integration.py (+1): QEM through the mind, beating naive. Files: holographic_meshqem.py (new),
test_holographic_meshqem.py (new), holographic_unified.py (2 faculties), test_integration.py, README,
NOTES_concepts.md, tour.py. Faculty count -> 274.

*** First item off the geometry->stack backlog: the sweep found the engine had every part of a decimator but the
quadric; this is the quadric, and (per the reverse thesis) it is the general error-minimizing MERGE operator -- the
same Sigma-nn^T-and-read-the-cost shape the creature's prototype compaction (R2) wants. ***


--------------------------------------------------------------------------------
OCTAHEDRAL NORMAL ENCODING -- quantize a unit vector on its MANIFOLD, not its ambient bits (Cigolle, Donow,
Evangelakos, Mara, McGuire, Meyer, JCGT 2014). From the geometry->stack backlog item A2: the shipped quantizer
(int8/quant='rd') quantizes a value's ambient bits, but a unit normal has only 2 DOF (it lives on S^2), so
quantizing 3 x/y/z components wastes a third of the budget on a constrained coordinate. The octahedral map projects
onto the octahedron (L1) and unfolds the lower hemisphere into a 2D square -- 2 numbers, bounded error, bits on the
intrinsic DOF.

WHAT SHIPPED (holographic_octnormal.py; additive; two UnifiedMind faculties):
  * oct_encode(normals) / oct_decode(uv) -- the continuous bijection S^2 <-> [-1,1]^2 (exact to float precision).
  * oct_quantize(normals, bits) [oct_encode_normals] -- integer codes (N,2) in [0, 2^bits).
  * oct_dequantize(codes, bits) [oct_decode_normals] -- unit normals back.
  * _sign_nz -- sign that returns +1 at zero (np.sign gives 0, which breaks the fold at the poles, e.g. [0,0,-1]).

WHY IT BELONGS (reverse thesis): manifold quantization made concrete -- "spend bits on the surface the data lives
on" -- which is the engine's binary-quantization-distorts-the-geometry negative turned into a method. The same
PRINCIPLE is reverse item R3: the FHRR phasor memory is unit-magnitude complex (S^1) and a normalized hypervector
lives on a high-D sphere, so both want their intrinsic coordinate quantized (for a phasor that analog is the PHASE
ANGLE -- one number, not two). Octahedral is the concrete S^2 instance; R3 is the principle carried to the phasor
memory (a future wire).

MEASURED (the bar): continuous round-trip EXACT (max ~1e-6 deg, a bijection); 8-bit quantized round-trip small +
BOUNDED (max 0.93 deg, mean 0.34); at an EQUAL 16-bit budget octahedral (8+8) BEATS naive x/y/z (5+5+6,
renormalized) on mean angular error 0.34 vs 1.20 deg (~3.5x) -- the manifold-quantization win; axis-aligned + the
z<0 pole survive (the fold edge case, fixed by _sign_nz); decode outputs unit vectors; deterministic.

KEPT NEGATIVES (loud):
  * At EQUAL bits-PER-COMPONENT naive xyz is more accurate -- because it spends 50% more bits (3 comps vs 2). The
    octahedral win is a STORAGE win (same accuracy in 2 numbers naive needs ~2.5-3 for), stated so the per-component
    numbers aren't misread.
  * Octahedral is specific to S^2 (3-D unit vectors). It does NOT generalize verbatim to S^1 (FHRR phasors) or a
    high-D sphere -- those use the SAME PRINCIPLE with a different intrinsic coordinate. The literal map is for
    normals; R3 is the principle, not this function.
  * The fold has measure-zero seams (the octahedron edges, z=0) where the (u,v) representation is non-unique; points
    there still decode to a valid unit vector -- the standard harmless oct caveat.

Tests: +9 (1419 -> 1428). test_holographic_octnormal.py (+8): continuous round-trip exact; axis-aligned + poles
roundtrip; 8-bit bounded error; codes in range; decode outputs unit vectors; more bits -> lower error; oct beats
naive at equal budget; determinism. test_integration.py (+1): octahedral through the mind, beating naive. Files:
holographic_octnormal.py (new), test_holographic_octnormal.py (new), holographic_unified.py (2 faculties),
test_integration.py, README, NOTES_concepts.md, tour.py. Faculty count -> 276.

*** Second item off the geometry->stack backlog (item A2, paired with QEM). The concrete S^2 case of manifold
quantization -- the same "quantize the intrinsic DOF" principle reverse item R3 wants for the FHRR phasor memory. ***


--------------------------------------------------------------------------------
SPECTRAL BANDWIDTH + A SINGULARITY CROSS-CHECK -- the genuinely-new parts of the fractal-optics backlog's
"fractal-dimension/bandwidth probe" (item 2). The DE-DUP discipline applied to the engine itself: the audit found
fractal DIMENSION is already shipped (box-counting + R/S Hurst in holographic_fractal; the fractal_dimension /
self_affinity faculties), so this ships ONLY the two missing pieces the review named, not another dimension.

WHAT SHIPPED (holographic_bandwidth.py; additive; two UnifiedMind faculties):
  * spectral_bandwidth(x, energy_fraction) [spectral_bandwidth] -- the fraction of Nyquist holding that energy
    fraction; the number that drives a band-limited encoder's bandwidth knob (the next item). Small for band-limited
    content, near 1 for broadband. GENUINELY NEW (the review's "the probe's real job is bandwidth measurement").
  * spectral_dimension(x) -- the power-spectrum-slope dimension D=(5-gamma)/2 (Berry & Klein), a fast estimator used
    as a cross-check term (NOT the engine's primary dimension).
  * fractal_confidence(x) [fractal_confidence] -- (d_spectral, d_increment, agree): two INDEPENDENT slope estimators
    and whether they agree -- the singularity flag. The shipped single-estimator dimension silently returns a wrong
    number for a step/tone; this catches it.

MEASURED (the bar): bandwidth smooth sinusoid 0.0007 << white noise 0.947 of Nyquist; rougher fBm (lower Hurst) ->
more bandwidth; on clean fBm of known H the two slope estimators AGREE and bracket D=2-H (spectral 1.70 / increment
1.65 at H=0.3); a STEP (isolated singularity) -> spectral 2.03 / increment 1.50 DISAGREE -> flag fires; a pure tone
-> disagree -> flag fires; deterministic.

A MEASURED FINDING (kept): the cross-check uses spectral-slope vs increment-variance, NOT the shipped R/S Hurst --
because R/S reads a DIFFERENT number on the same clean fBm (it is a range statistic weighting coarse/low-frequency
trend-dominated scales, while the slope methods fit the whole power law). They measure different things, so R/S is a
poor naive co-validator here. The honest cross-check is slope-vs-slope. (R/S stays correct for what it ships for --
series persistence.) An instance of the backlog discipline: check the live code, build only the delta, and report
exactly where two methods legitimately disagree.

KEPT NEGATIVES (loud):
  * spectral_bandwidth is an ENERGY rolloff: a fractal's front-loaded 1/f^b energy can read a small bandwidth even
    though its self-similar detail extends higher -- band-limiting to the energy-bandwidth keeps the bulk, discards
    the fine detail (the fundamental fractal trade; superoscillation is the standing proof it can't be cheated for
    free). Honest about fidelity-for-a-budget, not lossless bandwidth.
  * the power-spectrum-slope dimension is the one FOOLED by singularities -- it exists here only as a cross-check
    term, never the engine's reported dimension; trust a dimension only when agree.
  * 1-D signals; higher-D the slope relation is approximate, and images already use the shipped box-counting.

Tests: +10 (1428 -> 1438). test_holographic_bandwidth.py (+9): bandwidth separates smooth/broadband; bandwidth in
[0,1]; rougher fBm -> more bandwidth; spectral D recovers fBm; increment D recovers fBm; cross-check agrees on fBm;
cross-check flags a step; cross-check flags a pure tone; determinism. test_integration.py (+1): bandwidth +
cross-check through the mind. Files: holographic_bandwidth.py (new), test_holographic_bandwidth.py (new),
holographic_unified.py (2 faculties), test_integration.py, README, NOTES_concepts.md, tour.py. Faculty count -> 278.

*** First fractal-optics backlog item. The de-dup lesson again: the engine already had fractal dimension three ways,
so the real work was the BANDWIDTH driver (for the next item, the band-limited-encoding faculty) and the cross-check
the single-estimator dimension lacked. Build only the delta. ***


--------------------------------------------------------------------------------
AUTO-BANDWIDTH KDE VIA THE ENCODER -- the disciplined form of the fractal-optics backlog's "band-limited-encoding
faculty" (Item N). A LIVE AUDIT of the encoder reshaped the ask, and the slog produced several kept negatives before
landing on what actually delivers.

THE AUDIT (what the review's premise got wrong about the live code):
  * The SINC kernel's bandwidth is NOT tunable -- its width is fixed at scale=1/(hi-lo); the `bandwidth` parameter
    only affects the RBF phases. So "tune the sinc ideal filter to Nyquist" does not apply; only RBF bandwidth is
    selectable. KEPT NEGATIVE.
  * The encoder is a SCALAR encoder, not a function encoder -- reconstructing an oscillatory function by bundling
    weighted samples + Nadaraya-Watson collapses to the mean and does not benefit from bandwidth tuning. KEPT
    NEGATIVE (the failed approach; measured RMSE ~0.7 = predicting the mean).
  * The encoder's DOCUMENTED use is the RBF kernel as a KDE ("a bundle of encoded points reads as a proper KDE"),
    and THERE the bandwidth IS the band-limit with a real optimum (U-shaped error). The faculty lands here.

WHAT SHIPPED (holographic_kde.py; additive; two UnifiedMind faculties):
  * kde_bandwidth(samples, lo, hi, method) [kde_bandwidth] -- RBF bandwidth by 'lcv' (leave-one-out likelihood,
    robust) or 'silverman' (cheap fallback).
  * density_estimate(samples, lo, hi, query, dim, seed, method) [density_estimate] -- KDE via the encoder (bundle of
    encoded samples, density ~ bundle . encode(x)), bandwidth auto-selected, output normalized to integrate ~1.
    Returns (density_at_query, bandwidth).

THE KEY BUG FOUND + KEPT: LCV REQUIRES a NORMALIZED kernel. The encoder's kernel is unnormalized (its integral grows
with width), so naive LCV collapses to the WIDEST bandwidth (measured: it picked bw=2, the floor). The selection
normalizes the Gaussian per candidate (1/(std*sqrt(2pi))) and then LCV works -- landing near the ground-truth
optimum on both bimodal (bw 39 vs optimum 40) and unimodal (bw 22 vs optimum 20) densities. The encoder is still
used for the actual estimate; only the selection normalizes.

MEASURED (the bar): bimodal density -- LCV bw 39 near optimum 40, shape RMSE 0.17 BEATS the fixed default (bw 1.8)
1.16 by 6.8x and Silverman 0.45; estimate correlation 0.99 with truth; unimodal -- LCV near optimum, beats default
~7x; the density integrates to ~1; a too-small dim (16 vs 1024) gives worse correlation at the same bandwidth (the
capacity negative); deterministic.

KEPT NEGATIVES (loud):
  * SINC bandwidth is not tunable in the shipped encoder (only RBF) -- the review's sinc-ideal-filter knob does not
    apply.
  * LCV requires normalized kernels (the bug above) -- naive LCV on the encoder's unnormalized kernel collapses.
  * Silverman's rule (fallback) over-smooths MULTIMODAL data (~2.6x vs LCV's ~6.8x) -- the standard caveat.
  * Bandwidth selection fixes the SMOOTHING match, NOT capacity: a too-small dim cannot be rescued by any bandwidth.
  * Function reconstruction (vs density estimation) is NOT this encoder's job (the failed Nadaraya-Watson approach).

Tests: +10 (1438 -> 1448). test_holographic_kde.py (+9): LCV beats default bimodal; LCV near optimum bimodal; LCV
near optimum unimodal; estimate correlates with truth; Silverman beats default but worse than LCV; density
integrates to ~1; capacity negative (small dim worse); silverman bandwidth is a number; determinism.
test_integration.py (+1): auto-bandwidth KDE through the mind. Files: holographic_kde.py (new),
test_holographic_kde.py (new), holographic_unified.py (2 faculties), test_integration.py, README, NOTES_concepts.md,
tour.py. Faculty count -> 280.

*** Second fractal-optics backlog item. The audit reshaped the review's over-promise (tune the sinc to Nyquist)
into what the shipped encoder actually supports: auto-bandwidth KDE, where the bandwidth IS the band-limit and LCV
matches it to the data 6.8x better than the default. The kept negatives (sinc not tunable, NW collapses, LCV needs
normalization) are the audit working. ***


--------------------------------------------------------------------------------
SCREEN-SPACE-ERROR LOD POLICY -- the geometry->stack backlog's "geometric screen-space-error policy." The piece that
turns QEM decimation + surface_deviation (both shipped) into an actual DECISION: which simplification to show.

THE REVERSE-THESIS CONNECTION (why it belongs here): this is the engine's own error-budget RESOLUTION SELECTION
carried to meshes. coarse_to_fine refines a query only until an error budget is met; multires_pyramid keeps a signal
at several scales; the equidistribution rule places resolution where needed. select_lod is that rule for geometry --
the coarsest level of a decimation chain whose error, projected to the screen, meets a pixel budget. The principle
is the one the engine already uses for signals and queries; only the domain (meshes) and the budget unit (pixels)
are new.

WHAT SHIPPED (holographic_lod.py; additive; two UnifiedMind faculties):
  * build_lod_chain(mesh, targets) [mesh_lod_chain] -- QEM-decimate to coarser levels at face-count fractions,
    measuring each level's surface deviation (mean, max) from the ORIGINAL. Returns fine->coarse LODLevel records;
    level 0 is the original (zero error).
  * screen_space_error(world_error, distance, screen_height_px, fov_rad) -- project a world error to screen pixels:
    sse = world_error * screen_height / (2 * distance * tan(fov/2)).
  * select_lod(chain, distance, pixel_threshold, ...) [mesh_select_lod] -- index of the COARSEST level whose max
    screen-space error stays under the pixel budget (the cheapest mesh that looks right at that distance).

MEASURED (the bar): chain off an icosphere F[128, 64, 32, 16] with max deviation [0.0, 0.072, 0.105, 0.174]
(monotone -- fewer faces, growing error); screen error falls with distance; LOD selection by distance [0,0,0,2,3]
across 2..200 units (full detail up close, F16 far away) -- monotone coarsening; the choice is TIGHT (at d=50 it
picks F32 at 1.97px while F16 would breach the 2px budget); a tighter pixel threshold or higher screen resolution
forces a finer level; deterministic.

KEPT NEGATIVES (loud):
  * the error driving the policy is GEOMETRIC surface deviation (a Hausdorff-style distance), not a perceptual or
    silhouette metric -- a coarse mesh can be within the pixel budget on average yet show a visible silhouette
    break. The policy is exactly as good as surface_deviation is.
  * the projection ignores foreshortening and screen position (the standard conservative LOD estimate, not a
    per-pixel bound).
  * the chain inherits QEM's limits (closed meshes, boundary handling) -- this selects among levels, it does not
    improve them.

Tests: +13 (1448 -> 1461). test_holographic_lod.py (+12): chain has several levels; first level is the original
zero-error; face count strictly decreases; deviation only grows; screen error falls with distance; screen error
scales with resolution; LOD coarsens with distance; full detail up close / coarser far; selection is tight; tighter
threshold never coarser; higher resolution never coarser; determinism. test_integration.py (+1): LOD policy through
the mind. Files: holographic_lod.py (new), test_holographic_lod.py (new), holographic_unified.py (2 faculties),
test_integration.py, README, NOTES_concepts.md, tour.py. Faculty count -> 282.

*** geometry->stack backlog item, completing the QEM decimation story (decimate -> measure -> SELECT). The reverse
thesis again: a geometric LOD policy is the engine's error-budget resolution selection (coarse_to_fine) in the mesh
domain -- same rule, different units. ***


--------------------------------------------------------------------------------
BINDING-STABILITY REGIME TEST -- the fractal-optics backlog's "band-limit-preservation regime test" (Trefethen
transient-growth / pseudospectra spirit), grounded in the engine's actual bind. The investigation measured all three
relevant operations on the real substrate; the Trefethen framing came up empty and the real story is a LINEAR one.

WHAT THE INVESTIGATION FOUND (all measured, in the self-test):
  * LINEAR ops preserve the band-limit -- bind, bundle, permute all map a white spectrum to a white spectrum
    (high-frequency-energy fraction ~0.5 throughout). No spectral concentration.
  * The CLEANUP shows NO transient growth -- a pure HIGH-FREQUENCY perturbation of a stored atom, iterated through
    the dense-associative (modern-Hopfield) cleanup, contracts MONOTONICALLY to zero (one step at usable beta). The
    non-normal transient amplification Trefethen's lens looks for does not appear.
  * So the real stability axis is a LINEAR property of the binding KEY: its SPECTRAL FLATNESS. unbind(bind(x,k),k)
    returns x convolved with |K|^2, equal to x only when |K|=1 everywhere -- a UNITARY key (flatness 1.0). A random
    key (flatness ~0.5) DISTORTS, and the distortion compounds catastrophically over a chain.

WHAT SHIPPED (holographic_flatness.py; additive; two UnifiedMind faculties):
  * spectral_flatness(v) [spectral_flatness] -- Wiener entropy (geometric/arithmetic mean of the power spectrum),
    (0,1]; 1.0 = unitary, distortion-free key.
  * binding_distortion(key, seed, trials) -- the measured single-round bind/unbind distortion (ground truth flatness
    predicts).
  * binding_stability(v, tol) [binding_stability] -- {'flatness', 'distortion', 'stable'}: the regime diagnostic for
    a key.

THE DE-DUP (what is NOT new): the stable regime itself is already shipped -- unitary_vector mints flat-spectrum
atoms, and holographic_array / holographic_assembly already use them "for exact unbind." What was missing, and is the
genuinely-new contribution, is the DIAGNOSTIC: measuring where any vector sits on the stability spectrum, and the
regime test confirming flatness predicts distortion. Answers "is this key safe to bind/unbind repeatedly?"

MEASURED (the bar): flatness unitary 1.000 vs random 0.594; a unitary key is EXACT (chain-64 bind/unbind error <
1e-9), a random key distorts ~0.93 and compounds; flatness PREDICTS distortion -- across keys blended unitary->random
the flatness falls [1.0, 0.95, 0.81, 0.57, 0.3] and distortion rises [0.0, 0.33, 0.75, 1.13, 1.47] monotonically;
linear ops preserve a white spectrum; the cleanup contracts monotonically; deterministic.

KEPT NEGATIVES (loud):
  * the stable regime is NOT new (unitary_vector exists); this adds the MEASUREMENT. And unitarity is a mint CHOICE,
    not a free default -- the engine's own record notes a starved-maze bootstrap that went to zero under unitary
    atoms (their flatness removes a redundancy some paths rely on). Flatness tells you the binding cost, not that
    unitary is always right.
  * the Trefethen transient-growth framing, taken literally, came up EMPTY -- the honest result is a linear-stability
    story (key flatness), not a non-normal-dynamics one. Reported as found.
  * flatness governs binding (convolution) specifically; bundle capacity and cleanup confusability are separate axes.

Tests: +10 (1461 -> 1471). test_holographic_flatness.py (+9): flatness separates unitary/random; unitary key exact;
random key lossy; unitary chain stays exact; flatness predicts distortion monotonically; linear ops preserve white
spectrum; cleanup contracts monotonically (no transient growth); binding stability report; determinism.
test_integration.py (+1): binding stability through the mind. Files: holographic_flatness.py (new),
test_holographic_flatness.py (new), holographic_unified.py (2 faculties), test_integration.py, README,
NOTES_concepts.md, tour.py. Faculty count -> 284.

*** Third fractal-optics backlog item. The Trefethen lens looked for transient growth and found none -- a clean
negative -- so the honest deliverable is the LINEAR stability diagnostic the data actually pointed at: spectral
flatness predicts binding distortion, with unitary keys (already shipped) as the flatness=1 exact regime. ***


--------------------------------------------------------------------------------
SPLAT PRUNE / MERGE + A QUALITY-BUDGET LOD CHAIN -- the geometry->stack backlog's "splat prune/merge + exporter."
The splat-domain twin of the mesh LOD policy: reduce an existing splat set while holding quality, and pick a level
for a budget -- there the budget was screen-space pixels, here it is reconstruction PSNR.

THE KEY MOVE: each splat renders as amp * gaussian and the engine's gaussians are UNIT-NORM, so a splat's
reconstruction energy is exactly amp^2 -- "which splats matter" is "which have the largest |amp|". Drop the rest,
then one joint amplitude REFIT (splat_refit, the closed-form lstsq already in the engine) lets the survivors absorb
the overlap the removed ones carried. Contribution-ranked prune + refit degrades gracefully and dominates naive
pruning by a wide margin.

WHAT SHIPPED (holographic_splatprune.py; additive; four UnifiedMind faculties):
  * splat_prune(splats, target, keep) [splat_prune] -- keep the top-`keep` splats by |amp|, refit.
  * splat_merge(splats, target, radius) [splat_merge] -- merge splats closer than radius (amplitude-weighted centre
    and scale, summed amplitude), refit; reduces count.
  * splat_lod_chain(splats, target, keeps) [splat_lod_chain] -- prune to each count, measuring PSNR; returns
    fine->coarse (splats, count, psnr).
  * select_splat_lod(chain, min_psnr) [splat_select_lod] -- the fewest-splat level meeting the PSNR budget.

MEASURED (the bar): full 60 splats 44.5 dB; prune to 20 -- contribution 38.3 dB DOMINATES random 18.3 / worst (keep
smallest) 16.6 (a ~20 dB margin); LOD chain counts [60,40,20,10,5] -> PSNR [44.5,43.7,38.3,32.0,29.0] (graceful,
monotone); merge to 29 splats 39.9 dB (bounded loss); budget-30 keeps 10 splats, budget-43 keeps 40 (tighter budget
-> more splats); deterministic.

KEPT NEGATIVES (loud):
  * NO .ply / .spz exporter. Those are 3D-Gaussian-splatting formats (per-splat position, scale, rotation, opacity,
    spherical-harmonic colour); the engine's splats are 2-D field primitives (cy, cx, amp, sigma) -- the format does
    not fit the representation, so shipping it would be a mislabelled stub. Stated, not faked.
  * prune/merge operate on the ISOTROPIC splat format (splat_fit's output); the anisotropic splats (aniso_fit) carry
    a covariance and have their own optimiser; this does not prune them.
  * |amp| ranking is a proxy for true contribution when splats OVERLAP (energies not independent); the refit
    compensates but a jointly-removable overlapping pair is not detected as such -- good enough, not optimal.
  * merge is lossy by construction (one Gaussian cannot equal two); a large radius over a busy region loses real
    structure.

Tests: +12 (1471 -> 1483). test_holographic_splatprune.py (+11): contribution prune beats random; beats keeping
smallest; keep-all returns full; prune reduces count; LOD chain counts decrease; LOD PSNR degrades gracefully; merge
reduces count; merge loss bounded; tighter budget keeps more; selection meets budget; determinism.
test_integration.py (+1): splat prune/LOD through the mind. Files: holographic_splatprune.py (new),
test_holographic_splatprune.py (new), holographic_unified.py (4 faculties), test_integration.py, README,
NOTES_concepts.md, tour.py. Faculty count -> 288.

*** geometry->stack backlog item. The splat twin of the mesh LOD policy (decimate->measure->select becomes
prune->measure->select), same error-budget resolution selection in a different domain. The .ply/.spz exporter was
DECLINED honestly -- the format is for 3D Gaussians and the engine's splats are 2-D field primitives. ***


--------------------------------------------------------------------------------
SCENE COMPONENT DELTA -- the geometry->stack backlog's reverse item R6 ("cluster/scene delta"). The investigation
measured it on the real scene-graph and found the honest scope, which is the point worth recording:

THE SAVING IS AUTOMATIC. scene_to_recipe names every component (mesh, transform) by CONTENT HASH, so two scenes that
share a subtree already share the identical atom; stored in any content-addressed table they dedup for FREE --
measured 3.86x fewer stored components across a base + 8 variants each changing one of four subtrees (54 -> 14). There
is NO new delta ALGEBRA to invent; content-addressing already does the sharing (the same reason a content-addressed
blob store dedups a repo). This is the thin-item outcome flagged before the probe -- reported plainly.

WHAT SHIPPED (holographic_scenedelta.py; additive; two UnifiedMind faculties) -- only the genuinely-useful,
NOT-automatic operations:
  * scene_delta(base, variant) [scene_delta] -- {'added', 'removed'} content-hashed component ids: the explicit DIFF,
    so a variant is TRANSMITTED as its delta (send base once, then small deltas) rather than re-sent whole.
  * apply_scene_delta(base_components, delta) -- rebuild the variant's component set from base + delta (exact).
  * scene_components(scene) -- the content-hashed component-id set (the handle sharing keys on).
  * scene_dedup_saving(scenes) [scene_dedup_saving] -- {'naive', 'unique', 'saving_x'}: quantify the automatic
    sharing.

MEASURED (the bar): a one-subtree change -> delta 1+1 vs full 6 components; base+delta rebuilds the variant exactly;
an identical scene -> empty delta; dedup across 9 scenes saves 3.86x (54 -> 14 components); deterministic.

KEPT NEGATIVES (loud):
  * the dedup saving is AUTOMATIC from content-addressed atoms, NOT a contribution of this module -- it exposes and
    measures it and adds the transmittable diff. Stated, not dressed up as a new mechanism. (This was the thin item
    flagged before probing; the probe confirmed it.)
  * the delta is over COMPONENTS (the heavy mesh/transform atoms); the scene TREE wiring is rebuilt by the recipe, so
    a delta that only re-wires shared components reads as an empty component delta though the scene changed.
  * sharing requires BIT-IDENTICAL components (the hash is exact) -- a near-but-not-identical mesh does not dedup;
    this is why geometric quantization (making near-identical things identical) matters upstream.

Tests: +9 (1483 -> 1492). test_holographic_scenedelta.py (+8): one-subtree change is small; reconstruction exact;
identical scene empty delta; dedup saving above 1x; dedup accounting consistent; variants share most components;
apply-delta round trip with added+removed; determinism. test_integration.py (+1): scene delta through the mind.
Files: holographic_scenedelta.py (new), test_holographic_scenedelta.py (new), holographic_unified.py (2 faculties),
test_integration.py, README, NOTES_concepts.md, tour.py. Faculty count -> 290.

*** Reverse item R6, probed honestly. The reverse thesis held -- a scene delta IS a content-addressed component diff
-- but the dedup turned out AUTOMATIC (content-hashing already shares), so the only genuinely-new deliverables are the
explicit diff (for transmission) and the saving measurement. Shipped those; recorded the rest as a kept finding. This
was the thin item flagged in advance; the measurement confirmed it. ***
