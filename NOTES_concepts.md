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
