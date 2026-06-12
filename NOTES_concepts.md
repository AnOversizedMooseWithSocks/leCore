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