# Holographic VSA

[![tests](https://github.com/AnOversizedMooseWithSocks/holostuff/actions/workflows/ci.yml/badge.svg)](https://github.com/AnOversizedMooseWithSocks/holostuff/actions/workflows/ci.yml)

A from-scratch, **numpy-only** holographic / vector-symbolic engine - and a small
web UI on top of it.

One idea runs through the whole thing: represent *everything* - a number, a word,
a record, a fact, a creature's situation, an image - as a point in a very high
dimensional space, and combine those points with a few reversible algebraic
operations (`bind`, `bundle`, `permute`). Out of that one substrate you get
associative memory, learned word meaning, structured records, symbolic reasoning,
a little reinforcement-learning creature, and a damage-tolerant image archive.
No neural-network framework, no pretrained models, no GPU - just numpy. (The UI,
image I/O, tests, and plots use Flask / Pillow / pytest / matplotlib.)

If you only read one thing: run it (below), click **Run full system tour**, and
watch all twelve subsystems work in ~30 seconds.

### One model on top (`holographic_unified.py`)

The subsystems grew as separate studies, but they were always meant to be one model,
and they already share the one thing that makes that possible: a single holographic
space, with a `UniversalEncoder` that turns *any* input -- text, image, number,
category, record, sequence -- into a vector in it. `UnifiedMind` is the top level that
makes the sharing real rather than nominal. There is **one perception** (the encoder),
**one associative memory** (the autonomous, self-maintaining `SelfOrganizingMind`, which
both classifies and is searched for recall), and **one decision brain**
(`HolographicMind`), all reading and writing the same space. Crucially it does *not*
reimplement simple versions of these the way the earlier `Mind` facade did -- it uses
the real, self-organizing ones, and every input passes through the same encoder before
it reaches any of them.

The honest test of a unification is whether the shared substrate costs anything. One
`UnifiedMind` was taught text topics, little images, and records *into a single memory*,
then asked to classify all three: it matches three separate per-modality memories within
the noise of the test sets (images and records identical; text within a sentence or
two), because the modalities land in near-orthogonal regions of the space. The same mind
recalls the nearest stored item to a query and, given an action set, learns a contextual
decision -- all over the same encoder and space. (`demo_unified()` shows it.)

What is *not* pretended to be one call: classification, recall, and decision are
different **operations** on the shared substrate (aggregate into prototypes; index the
individuals; weight by reward), and generation is a fourth. The unification is the shared
space and shared self-maintenance, not a single magic method.

Two of the obvious "make the brain run everything" ideas were tested and only one earned
its keep -- which is the point of testing them. **A learned curation controller** (a
brain whose actions are store/skip, learning which incoming items are worth keeping) is a
clean *negative*: the self-organizing memory already compresses by aggregation -- ~1800
redundant observations collapse to a handful of prototypes at full accuracy -- so a
store/skip policy has nothing to add and, by pruning, actually hurts. Storage is already
decided well, by aggregation plus the autonomous (measurement-driven) reorganization
gate, which is a better mechanism for that than reinforcement learning. **Routing** (the
brain/gate deciding which modality's concepts a query competes against) is built in too,
but its value came with an honest correction. It first looked like a small accuracy win
(text 75% -> 79%); tracking that down revealed an *encoding bug* -- a list of tokens
tagged `modality="text"` was falling through to the order-sensitive sequence encoder
instead of the order-insensitive sentence bundle, degrading text vectors until they
collided with image and record prototypes. Routing had been cleaning up the bug, not real
interference. With it fixed, unified text classification jumped to parity with the
dedicated pipeline (92% on the topic set, ~88% on pulled Reuters), the modalities separate
cleanly, and routing now changes nothing on this data: it is a cheap, correct *safeguard*
that removes cross-modal collisions if they occur, not a routine booster. The learned MoE
gate would only beat trivial modality routing when the modality must be *inferred* and the
experts are miscalibrated -- the mixture-of-experts study's finding. All reflected in
`UnifiedMind`: routing as a safeguard, no curation layer (on purpose), encode bug fixed.

Still sequenced and honestly unfinished: making **generation** a sequential operation
over the same store rather than a side n-gram, so "learn language" runs through the one
memory too.

**Generation** is now folded in -- as far as measurement allows, which turned out to be
an instructive boundary. Its next-symbol *prediction* was always a holographic operation:
the distribution after a context is a superposition of symbol atoms, read back by cosine
cleanup -- the same primitive the classifier uses -- so that half already lived on the
shared substrate, and `UnifiedMind.learn_sequence` / `generate` expose it as the model's
fourth operation. The other half, the *context index*, was the open question: could the
exact-string lookup become a holographic nearest-key recall over the shared store? Three
variants say no. Pure nearest-context recall drops next-symbol accuracy from 67% to 56%
at order 6; a hybrid (exact when seen, holographic backoff when unseen) is no better and
worsens at higher order (69% -> 49% at order 8). The reason is clean: string backoff
falls to a well-populated *shorter* context (reliable statistics), while fuzzy recall
finds a *same-length but different* context (unreliable), and precise context matching is
exactly what lets a higher-order model exploit longer context. So generation keeps an
exact key -- it is the one operation here that needs precision and is hurt, not helped, by
associative recall. That is the honest end of the list: perception, classification,
recall, decision, and a generator's prediction all share the one space and its
primitives; generation's indexing is the measured exception, kept exact on purpose.

**Try it.** `python unified_app.py` opens a console (http://127.0.0.1:5001) that PULLS a
real corpus on request -- Reuters categories, Brown genres, Gutenberg authors, or Europarl
languages, downloaded via NLTK from GitHub, the same place the test data came from --
trains one `UnifiedMind` on it, and lets you classify, recall the nearest stored item,
watch the memory organize itself into sub-prototypes, and generate new text in the
corpus's style, all against the one trained mind.

---

## Quick start

### Windows
Double-click **`run.bat`** - it installs dependencies on first run, starts the
server, and opens `http://127.0.0.1:5000`.

### Any platform
    pip install -r requirements.txt
    python app.py            # then open http://127.0.0.1:5000

If `pip install` fails with **"Access is denied" / WinError 5** (a system-wide
Python you can't write to), install into a local virtual environment instead -
which is exactly what `run.bat` now does automatically:

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python app.py

(or add `--user` to the `pip install` to install just for your account.)

The web UI has these panels: **System tour** (runs every subsystem once),
**Compression & speed** (single- and many-file size, encode/decode timing, and corruption resilience vs JPEG/PNG),
**Batch operations** (superposition capacity, 1-bit memory, cleanup throughput), **Creature** (trains the forager
with lethal poison present, then animates its random-vs-trained life step-by-step -- with a live energy bar and star
tally; an Obstacles mode adds walls to route around with the optimal route drawn
in, and a Labyrinth mode has it learn the way out of a maze -- all on a prototype memory built from the same classifier),
**Test suite** (runs the full pytest suite), **Query & recall** (the interactive image demo - degrade an
image, optionally destroy part of the plate, watch it get recalled), **Recall by description**
(cross-modal recall - describe an image in words and get the matching one back from the tag address space),
**Set packer** (delta-code a set of related images against one reference), and **Image vault** (the general store: relate by fingerprint, compress adaptively across lossless and lossy encoders with an honest table, and query by example). The Test suite panel auto-discovers and runs every test_*.py (162 at last count; three skip without the optional NLTK corpora). The package also ships the real 712-sprite set packed to ~67 KB at `features/sprites.hsp` (which doubles as a live demo of the sprite packer), and the UI uses it in two places: the Image vault runs relate/compress/query on the whole set, and the learning creature is drawn as a real walking sprite (`amg2`) that turns to face the direction it moves and cycles its two walk frames -- with its baked-in background keyed out (flood-filled from the edges) so it shows real transparency over the grid instead of an opaque tile. The creature also runs on an energy mechanic: it starts each life with 100 energy, every step costs 1, each star it reaches gives +3, and stepping on poison empties the battery -- instant death -- so collecting stars and staying alive are the same goal. Finally, a **Vision** panel shows that the image is just numbers: RGB->HSV colour and dominant-colour extraction, Sobel edges with Hough line/circle detection and Harris corners, a geometric shape classifier, and unsupervised *emergent* classes that fall out of clustering simple feature descriptors -- then a VSA prototype classifier (bundle + cosine cleanup) labels held-out shapes, tying the vision work back to the holographic engine. The panel reports each step's accuracy honestly, including where unsupervised clustering tops out. A final **Compositional scene** panel takes the opposite stance to a holistic descriptor: it reads the DCT coefficient layout as a texture tag (finally using the DCT as a feature, not just for compression), pairs it with HSV colour and geometric shape for automatic per-object tags, then encodes each object as a product of attribute atoms and a scene as their superposition -- so a ResonatorNetwork can factor the parts back out. Multi-object scenes now decompose reliably up to ~5 objects: the old ~50%-at-three ceiling turned out to be a scale bug (normalising the scene) plus missing refinement, not a real capacity limit -- keeping the scene as an unnormalised superposition and adding coordinate-descent sweeps recovers 3-4 objects at 100%. A **Scaling** panel confronts the deepest limit head-on: one holographic trace is a bundle with finite capacity (a 2048-d memory recalls 100% of 64 pairs but ~0% of 2048), so instead of one flat store it grows a deterministic recursive tree -- each node a seeded random hyperplane splitting items at the median, each leaf a small memory kept inside capacity, queries descending with a beam that back-tracks into nearby cells. This is the random projection tree of Dasgupta & Freund and, in spirit, how slime mould beats the size limit of pure diffusion by resolving a broad mass into a hierarchical vein network. The flat memory collapses with scale while the tree holds 100%, and search reaches ~96% recall at a fraction of a full scan's comparisons; per-leaf query 'flux' shows the thick-vein / thin-vein structure. A HoloForest of several differently-seeded trees breaks the single tree's recall ceiling, reaching ~100% recall at a fraction of a full scan's comparisons. Finally, a **Content addresses** panel realises the original partitioning idea the way AWS S3 does: no folders, just a flat keyspace where each object's name encodes the hierarchy. The auto-tags (colour/shape/texture) generate a deterministic URI like `red/circle/smooth`, the key *is* the partition path, and a FacetStore supports S3-style prefix listing and CommonPrefixes roll-up. Where the RP-tree splits by meaningless random hyperplanes, this splits by meaning -- readable, queryable paths -- at the honest cost of bucket skew, with key depth as the lever. And the resonator closes the loop: it recovers an item's URI from its content vector alone, so the address is computed from the content. And the skew problem is now handled: build_indexes gives any hot bucket its own in-bucket HoloForest, so content search inside a popular prefix stays sub-linear -- the bi-level structure (semantic prefix outside, geometric forest inside) realised.

### From the command line
    python tour.py                    # guided tour of all subsystems (~20s)
    python holographic_creature.py    # any module runs its own demo
    python holographic_encoders.py    # numbers / text / records demos
    pytest -q                         # the whole test suite (60 tests)

---

## What it can do (with results from `tour.py`)

Everything below lives on the *same* vector substrate and the same memory.

**Numbers.** Encode a real value as a unit vector so nearby numbers get nearby
vectors, then read the number back out - even from a noisy vector.
`decode(encode(7.2)) = 7.19`; a noisy vector of `4.0` still decodes to `3.92`.

**Text.** Learn word vectors from raw co-occurrence (no gradients, no labels).
After a few passes over a tiny corpus: `cat ~ dog = 0.77` but `cat ~ car = 0.36`;
nearest word to `truck` is `car (0.88)`. The geometry carries meaning.

**Mixed records.** Pack a number, a category, and a free-text note into one
2048-d vector and read each field back individually: `price -> 140.7` (stored
142.5), `trend -> up`. Record-to-record similarity reflects all fields at once.

**Key -> value memory.** Store many `key -> value` pairs superposed in a *single*
vector and recall them by content, with cleanup snapping noisy results to the
nearest known symbol. A handful of `country -> capital` facts in one 1024-d
vector, all recalled correctly; it degrades gracefully when overloaded. And the
overload point moves: `recall_all` adds **successive cancellation** -- peel the
single clearest pair, subtract `bind(key, clean_value)` back out of the trace,
and the residual gets sharper for the rest; repeat (recursively) until the whole
"exposure" is developed. This roughly doubles how many pairs come back cleanly
from one trace (at 1024-d, 80 pairs go from 84% one-shot to 100%, and 160 from
42% to 70%), the way a decoder cancels interference or film develops the
strongest signal first. It is orthogonal to partitioning, so the two COMPOUND:
peeling *inside* each of 8 regions recovers 100% of 320 pairs where a single
one-shot trace gets 16% and peeling-alone has collapsed -- two compression
filters stacked, gist-then-residual. The one honest catch is error propagation:
once the clearest remaining guess is itself wrong, subtracting it injects noise,
so peeling rescues a trace inside a sane regime and partitioning is what keeps it
there. (`python holographic_ai.py` prints the full table.)

**Reasoning.** A resonator network factors a composite vector
(`subject (x) relation (x) object`, three things bound together) back into its
parts knowing only the vocabularies - recovering 6/6 facts where a single unbind
cannot. Also included: split-conformal error bars, an epistemic "how well do I
know this" map, and a semantic compass.

**A learning creature.** A grid-world forager with a purely holographic mind
(perceive -> decide -> learn by remembering experiences; no neural net). From a
random baseline of `-0.27` reward / `0.2` stars, after 120 episodes it reaches
about `+9.1` reward / `9.6` stars - it taught itself to find stars. It plays an
energy game: it starts with 100 energy, each step costs 1, each star gives +3,
and poison is lethal (energy to 0, instant death), so chasing stars and staying
alive are one and the same. The full demo in the UI also shows poison avoidance
-- it learns to collect stars AND route around the hazards, surviving most
worlds where a random walker dies. Two obstacle modes sit on the same machinery:
**Obstacles** scatters impassable walls into the forage world (kept connected,
so it is always solvable) and draws the BFS-optimal route behind the creature's
learned one -- here a working memory of recent moves collects ~1.6x more stars
than a reactive brain, because the same trick that helps when blind also helps
when the straight line is blocked. **Labyrinth** carves a fixed 7x7 perfect maze
and the creature learns the single way out over repeated tries (the classic
rat-in-a-maze), escaping in ~16 steps against an optimal of 16. The honest limit:
a 9x9 maze (a ~28-step solution) is mostly beyond this brain, because far-apart
corridors look identical through its egocentric senses. The module demo
(`python holographic_creature.py`) adds a working-memory scene with limited
vision plus the walls-and-labyrinth scene.

The creature's memory is the same holographic kit the image side uses, not a
separate store. Rather than hoarding every experience (it meets the same
egocentric situation -- "star east, wall north" -- in thousands of different
cells), the brain keeps one **prototype** per distinct situation: a new
experience is cosine-matched to that action's prototypes and either joins the
nearest class or starts a new one (the same bundle-and-cosine **classifier** the
vision panel uses), and each prototype is a **superposition** of its members
with a denoised mean return (the "superpose a gallery into plates" trick storing
experience instead of pixels). Because averaging the returns cancels the noise of
early exploration, the compressed memory is usually a *better* value estimate,
not just a smaller one: the reactive forager folds ~7,700 raw experiences into
~350 prototypes (about 22x) while foraging better, and the working-memory modes
compress 3-7x. Those prototypes are content-addressable, so `demo_introspect()`
indexes them with the same **recursive** HoloForest the image vault uses and
recalls the most similar past situation from a noisy cue in roughly a tenth of a
full scan. Honest caveat: that approximate index is for associative recall, not
the control loop -- choosing a move needs the full weighted neighbourhood, and
the approximation drops enough of it to wreck the maze policy, so deciding still
uses the exact scan, which the prototype compression already made cheap.

We tried the rest of the toolkit on the creature too, and kept only what
measurably helped -- the honest part of "use everything." Putting the recursive
forest in the *decision* loop dropped maze escapes from 93% to 0% (approximate
recall loses neighbours the value estimate needs). Deepening the working-memory
window to crack a 9x9 maze did not work (0% at both shallow and deep memory) and
at depth 10 it ballooned the prototype set to ~23k by making every state unique,
so generalization collapsed -- the 9x9 aliasing is a real ceiling for an
egocentric brain, not a tuning miss. Averaging an *ensemble* of independently
trained minds was worse than picking the best single one (their policies differ
too much to average), which is why the UI trains several candidate minds and
keeps the best -- a branching search over policies that beats voting. The net:
the toolkit lives in the creature where it belongs (a classifier and layered
superpositions for the memory, a recursive branching partition to index it), and
stays out of the one place it hurts.

**The creature, repurposed: a navigator over the data.** The grid was always a
testbed for a mind that perceives, decides, learns from what happened, and
adjusts -- and the rest of this project quietly built the *world* for it: a
recursive HoloTree that partitions data into branching regions, exactly like a
maze of corridors. `holographic_navigator.py` closes the loop ("inception"): the
*same* `HolographicMind` and `CreatureEncoder` that found the star now learn to
navigate the index to find what you asked for. The map is literal -- a cell it
stands on becomes a region (tree leaf) it is examining; "food is to the east"
becomes "the best match so far is strong / weak"; stepping toward the star
becomes examining the next-most-promising region; "I reached the star" becomes "I
have arrived at the answer, commit." The point is not to re-implement the tree's
routing but to spend its effort *adaptively*: a fixed beam reads the same number
of regions for every query, but difficulty varies enormously (here ~40% of cues
find their true neighbour in the very first region, while a third need four or
more), so a fixed beam overpays on the easy majority. The navigator senses how
confident the answer looks -- the margin between the best match and the runner-up
-- and decides arrive-or-keep-moving, learning to commit at once on easy queries
and search hard only on ambiguous ones. On a 2000-item tree it reaches **94%
recall at ~120 comparisons, matching the widest fixed beam's 96%/500 at ~4x fewer
comparisons**, from a policy of ~50 prototypes -- the same find-it / keep-looking
instinct that solved the maze, now buying retrieval efficiency. (It is trained
against an exact-scan ground truth, then deployed using only its learned senses;
this is the *access* half of "organize and access", and the natural next step is
to let the same loop decide where new items should live.)

And the organize half follows from the same instinct. Real query streams are
skewed -- a few items get most of the traffic -- so the navigator grows habits:
a small `ReflexCache` of the items it commits to most, checked *before* it
descends the tree, exactly the way slime mould thickens the veins it travels
often and lets the rest wither (and the way the engine's ReflexArc lets a
familiar input skip the expensive path). A use-reinforced hot set recognises
popular queries instantly, and a flux guard prunes the habit on an unpredictable
stream so it never costs more than it saves. On a skewed (Zipf) workload it cuts
the average query from **~125 comparisons at 96% to ~84 at 99%** -- recognising
most queries on sight -- while on a uniform stream the veins are pruned and it
falls back to the plain cost. Because each `find` can narrate its path, the
navigator has a literal train of thought: a familiar query returns
"recognised instantly -- a familiar query", an unfamiliar one walks the regions
("best match 0.18, a tie -- look further ... 0.90, clear -- arrive here"). The
little navigator in the data world, thinking out loud. (`python
holographic_navigator.py` runs both halves.)

**One general-purpose mind for any input.** Everything above rests on a single
fact: once you encode something into a hypervector, the same operations work on
it -- `bundle`/`bind`/`cosine` do not care where the vector came from. So the
hypervector is a universal interchange format, and the machines built on it (a
prototype classifier, the recursive index, the creature's brain) are a component
library that snaps onto *any* encoded input. `holographic_mind.py` is the front
door. A `UniversalEncoder` turns text, numbers, categories, raw feature vectors
(an audio MFCC frame, an image embedding), images, structured records (a dict of
fields), or sequences into one unit vector in one shared space -- different
modalities, identical representation. A `Mind` then takes a direction -- classify,
recall, or decide, either stated or inferred by `assemble()` from the shape of
what you hand it -- and lazily assembles the matching structure: teach it labelled
examples and it grows a classifier, pour items in and it grows a searchable index,
give it states/actions/rewards and it grows the creature's brain. The same `Mind`
class, measured end to end across modalities, handles text topic-labelling,
structured records with unseen field values, noisy 12x12 image patterns, tone
pitch from an FFT spectrum, associative recall from a partial cue, and a learned
contextual-decision policy -- each at high accuracy, with only the input format
and the direction changing between them. Honest scope: "general purpose" here
means a modality-agnostic representation plus a small library of measured machines
auto-wired to the job -- not a universal solver. The intelligence lives in the
components; what is new is that one interface and one representation now span all
of them, so the same mind can be pointed at a sentence one minute and a control
problem the next. (`python holographic_mind.py` prints the measured table.)

**A holographic mixture of experts, with a learned gate.** The general Mind routes
by a rule -- which verb you called, what type the input is. A true mixture of
experts needs the missing piece: a *learned* gate that, per input, decides which
specialist to trust, trained from outcomes. `holographic_moe.py` adds it, and the
gate is the creature's own brain -- encode the input, let the brain `decide` which
expert to consult, reward it when the expert it chose was right. Sparse top-1
routing (only the chosen expert runs), learned with no gradients, by the same
perceive/decide/remember loop that learned to forage. This closes a loop opened
earlier: blind ensembling was measured to *hurt* when experts disagree, because a
confident-but-wrong specialist drags the average down. The answer is not to mix
but to *route* -- and the gate learns to. With three cross-modal specialists
(text, image, audio) that each know only their own labels, the gate sees only the
encoded vector, never the modality, and still learns to send each input to the
right expert: **100% accuracy, matching an oracle router, against 43% for the best
single expert and 21% for random routing**. Given two experts that own different
halves of the number line (same modality), it learns to route by *value*, reaching
92% where any single expert caps at 50%. So the brain genuinely learns to route
from reward -- the capability is real.

But the honest comparison demands a gate-free baseline, and it changes the verdict:
just route to whichever expert is *surest of itself*. For a bank of holographic
specialists this is hard to beat, because an unfamiliar input naturally produces
low similarity -- "I don't know this" falls out for free -- so confidence already
tracks competence. Measured against it, confidence routing matches the learned gate
on the cross-modal task (100% vs 100%) and slightly *beats* it on the number-line
task (100% vs 92%, the gate's boundary error). So the finding is: a learned gate
works, but it is not needed for a homogeneous bank of well-calibrated holographic
experts -- confidence routing is simpler and at least as good. A learned gate earns
its keep only when confidence is unreliable: heterogeneous or miscalibrated experts
that are confidently wrong out of domain (the classic overconfident-model failure),
which is outside what this homogeneous bank can show. This is the same lesson the
forest-in-loop and the ensembling experiments taught -- keep the machinery only
where it measurably beats the simpler thing. (`python holographic_moe.py` prints
every baseline side by side.)

And there is a regime where it does beat it, which completes the story. Drop one
*heterogeneous* expert into the bank -- a linear+softmax model instead of a
holographic one -- and it behaves the way most real models do off their home turf:
it extrapolates with growing logits and is *confidently wrong*. On a task split
into two regions, where a calibrated holographic specialist owns one and the
overconfident linear specialist owns the other, the linear expert reports ~90%
confidence on the region where it is always wrong. That breaks the gate-free
heuristic -- confidence routing drops to ~72% -- while the learned gate, routing by
reliability it discovered from reward and ignoring confidence magnitude entirely,
holds at **100%, matching the oracle**, against 52% for either single expert. So
the full verdict: a learned gate is unnecessary for a homogeneous bank of
well-calibrated holographic experts, and exactly what you want the moment an expert
can be confidently wrong -- the normal case once the experts are heterogeneous.
(`demo_heterogeneous` in the same file shows it.)

**A self-organizing memory that reorganizes a shadow copy and swaps it in.** The
hard part of a system that keeps learning is keeping its data ORGANIZED as it
arrives. A holographic class is stored as a bundle, which is fine until a class is
several things at once -- "vehicle" is cars and trucks and motorbikes, in different
directions of the space. Bundle those into one prototype and you get their average,
a point that is none of them; on genuinely multi-modal classes a
one-prototype-per-label store collapses (measured: 49% where the structure allows
100%). `holographic_organizer.py` fixes this the way databases and CPUs do
in-place updates safely -- read-copy-update / double buffering. A small team of
organizer experts builds a SHADOW copy of the store from an experience buffer: a
`SplitExpert` discovers how many modes each label really has (raising k only while
it buys coherence, so a one-mode class stays one prototype and a three-mode class
becomes three -- chosen from the data, the self-classifying step), and a
`MergeExpert` folds away near-duplicates and flags cross-label collisions. The live
model is never touched during the build, so a query mid-reorganization sees a
complete, consistent store; then a single atomic SWAP makes the reorganized copy
live. Streaming multi-modal data, a naive one-prototype store sits at ~50% while
the self-organizing one climbs to **100% after each reorganization**, having found
the two modes per class on its own; and the swap is verified non-destructive (the
live model's answers are identical while the shadow is built, changing only at the
swap). This is the scaffolding for the self-* goal: self-learning (it absorbs a
stream with no training phase), self-organizing (it restructures its own memory on
the shadow and swaps), self-classifying (it discovers the sub-categories inside each
label). (`python holographic_organizer.py` runs it.)

And it can pull the trigger itself, which is what makes the cold-start problem
tractable. A system that starts blind has nothing to classify against; it files its
first data into immature prototypes and only later has enough to see the real
structure -- so early data ends up in the wrong place. A `TriggerExpert` watches two
signals it reads off the model itself -- *incoherence* (recent examples sitting far
from their own prototype: a class has gone multi-modal but is still one blurry blob)
and *novelty* (recent inputs matching no prototype: a new kind of thing has begun
arriving) -- and fires a reorganization with no schedule, only when a signal crosses
and enough new data has accumulated to be sure. Because the reorganization rebuilds
from the experience buffer, the early, badly-filed data gets re-placed in the swap.
Streaming two multi-modal classes and then introducing a third halfway through, the
self-triggering store goes from 52% to **100%, firing exactly twice on its own**:
once on incoherence (splitting the early blurry classes -- the cold-start data
re-placed) and once on novelty (organizing in the class that arrived mid-stream),
while a store that never reorganizes stays at 52%. And it does not thrash: on data
that really is one mode per class, coherence stays high and it never fires. The
reorganization can also recurse -- reorganize, re-check, and reorganize again while
it still buys coherence -- for structure that is modal at several scales. This is
the loop the self-* goal needs: notice your own organization has gone stale, fix it
on a shadow, swap it in, without being told when. (`demo_self_triggering` shows it.)

And, like the brain, it now does this with **no thresholds at all**. `auto_reorganize`
holds out a slice of recent experience, speculates a few organizations of itself at
different resolutions (one prototype per label, up to a few sub-modes each), and keeps
whichever *classifies the held-out slice best*, breaking near-ties (within one
standard error, read off the data) toward the fewest prototypes. Held-out accuracy is
the only judge, and it replaces both hand-set signals at once: a blurry cold-start
blob is beaten on accuracy by a split, so it splits; a new class that arrives
mid-stream is absorbed when a finer organization starts predicting it; and a class
that really is one mode ties at every resolution, so the single prototype wins on
leanness and nothing over-splits. On the same streaming cold-start-plus-new-class
test it reaches 100% (vs 56% for never reorganizing), choosing `k=2` to split the
early blur and `k=2` again to take in the new class -- matching, and here slightly
beating, the hand-tuned trigger, with nothing tuned. (`demo_autonomous_organizing`
shows it narrating each choice.)


**The same tools, turned on the brain itself (inception).** The brain that runs the
system -- the creature's `HolographicMind`, the same class the navigator and the MoE
gate are built from -- can go stale just like any other memory. It never forgets:
its bundles only grow, and near-duplicate prototypes (cosine below the merge
threshold) pile up. That redundancy turns out to be the exact thing that makes it
stale. When the world *shifts* -- the right action for a situation changes -- each of
those duplicates still holds the old value up, and an online update only ever touches
the single nearest one, so the value barely moves and the orchestrator cannot
unlearn. Measured plainly: after a regime shift where every situation's best action
changes, a plain brain is stuck near chance across thousands of steps (its old action
still reads 0.87 while the new correct one reads 0.19), carrying ~440 prototypes.
The fix is the data-organizer's own merge tool, run on the brain's value memory:
fold the duplicate prototypes into one, combining their returns by count. A single
prototype that every update touches *can* be unlearned where a cloud of duplicates
cannot -- so folding both compresses the memory (~440 -> ~70 prototypes, 6x) and
restores adaptation (recovery to 100%). And the brain triggers it itself, off two
signals it reads about its own state: *redundancy* (it has gone bloated) and
*surprise* (an EMA of how far its value predictions miss the returns it actually
gets -- which spikes the moment the world moves). With `maintain=True` it reorganized
itself six times over the run, unprompted, and ended both fresh and lean; with it off
(the default) the brain is byte-for-byte its old self.

And the last hand-hold is now gone too. `maintain='auto'` runs the whole thing with
**no behavioural thresholds at all** -- no surprise floor, no redundancy floor, no
fixed fold grain. Instead the brain keeps a window of recent experience and, every so
often, *speculates*: it builds a handful of reorganised versions of itself -- fold its
duplicates at a few grains (compress, forget nothing); rebuild from recent experience
(forget the stale regime) -- and measures each one the way that actually matters, by
the reward its greedy decisions would have earned on a held-out slice of that window.
A rebuild wins as soon as its decisions are better than the best fold -- it does not
have to win by a margin, because the costs are asymmetric: a needless rebuild in a
stable world just re-derives the same policy from still-valid recent experience and
costs nothing, while a missed rebuild after a shift strands the brain on a stale
policy. When a rebuild is chosen it is refit on the *full* recent window, not just the
slice it was selected on. Otherwise the brain compresses without forgetting, taking the
leanest candidate that is statistically as good as the best. One rule, and it does the
right thing in both regimes: while the world holds still it picks a fold (matching the
6x compression to 72 prototypes), and the moment the world shifts it picks a rebuild and
recovers to 100% -- with nothing tuned for either case. The only knobs left are resource
budgets (how large a window, how often to look), not behavioural thresholds.

That eager-commit rule is itself a correction. The first version made a rebuild win only
if it beat the best fold by a full standard error -- the same instinct that, in the data
organizer, made splitting candidates (trained on a fit slice) lose to a "keep" model
trained on all the data. On easy shifts it was invisible, but a hard, noisy, narrow-gap
shift exposed it: right after the shift the recent window is still half old experience,
which flatters the stale memory enough that the one-SE margin keeps the gate sitting on
"keep" for thousands of steps while the world has plainly moved -- the autonomous brain
crawling back on online relearning alone instead of committing. Dropping the margin (the
costs are asymmetric, so "better at all" is the right bar) and deploying the rebuild on
the full recent window roughly halves the recovery time on those hard shifts and fires a
rebuild where the old gate fired none -- while the stationary case shows no churn and no
loss of accuracy or leanness. The same conservatism, found in two different self-* gates,
fixed the same way: judge on held-out data, then commit and deploy on all the (currently
valid) data. This is the inception the self-* goal asks for, fully closed: the
orchestrator does not rot while the things it orchestrates stay fresh, and it does not
need a human to tell it when or how hard to clean itself. (`demo_self_maintaining` shows
the autonomous brain narrating each choice; the data organizer's trigger runs the very
same speculate-measure-adopt rule, judged by held-out classification accuracy.)


**Text, from a system that knows no language.** There is no dictionary here and no
grammar -- every word and every letter begins as a meaningless random vector. What
the engine can learn is the *statistical structure* of text, and `holographic_text.py`
shows that structure alone carries surprisingly far. It ships small original datasets
(topical sentences across cooking / space / sports / money, and a short four-language
set) and answers four questions, each measured honestly:

- *Can it learn?* Random indexing gives each word a vector that is the sum of the
  words it appears near. After reading the corpus, words used alike sit closer than
  words that are not (same-topic cosine ~5x the cross-topic one; `oven` lands nearest
  `cake`/`bread`, `ball` nearest `team`/`striker`) -- distributional meaning, with no
  labels and no training loop. On a corpus this small the signal is real but modest,
  and the writeup says so.
- *Can it analyze?* A character-trigram classifier identifies English / Spanish /
  French / German from raw letters at ~83% on held-out sentences, knowing nothing
  about any of them -- each language is just a direction built from its common
  letter-triples.
- *Can it organize?* Representing each sentence by the bundle of its learned word
  vectors, a prototype-per-topic classifier labels held-out sentences at ~92%, and
  clustering the sentences with no labels recovers the topics at ~82% purity. The
  unsupervised result leans directly on step 1: it only works this well *because* the
  learned meanings pull related sentences together (raw random atoms cluster at ~50%).
- *Can it produce?* A holographic character n-gram -- storing each context's next
  character as a superposition, read back by cleanup, backing off to shorter contexts
  -- predicts the next letter ~51% of the time (vs ~19% for always guessing a space)
  and the words it emits are ~100% real, despite working one character at a time with
  no notion of a word. The honest verdict: it reads like the training text up close
  and drifts into nonsense from afar -- it learned letter and word statistics, not
  meaning or grammar. (`python holographic_text.py` runs all four.)

The built-in datasets are deliberately tiny -- enough to read and test offline. To
see how far the *same* code goes on real text, `holographic_text.py` can also pull
public-domain corpora from NLTK (which hosts them on GitHub): Project Gutenberg
books, the Universal Declaration of Human Rights in many languages, and the
genre-labeled Brown corpus. Nothing about the algorithms changes -- only the amount
of text they learn from -- and everything gets sharper: language ID rises to **97%
across 11 languages**, word neighbours become genuine (`woman` -> `man, lady, ladies,
person`; `happy` -> `sorry, glad`), genre classification reaches **73% across five
Brown genres on 145 real documents**, and a 6-gram trained on *Alice in Wonderland*
predicts the next letter **62%** of the time with **96% real words**, generating
recognisable Carroll ("the cook and the mouse shook its head ... she was not a
serpent"). The scaling is opt-in (`pip install nltk`, one download) and the module
still runs fully offline on the built-in data without it. The limits stay the honest
ones -- distributional, not grammatical; an n-gram, not an understanding -- they just
arrive much later with real data. (`demo_text_scaled()` runs these.)

Generation in particular turns out not to be English-specific. The same character
n-gram, with nothing language-aware in it, was trained on five European languages
(European Parliament proceedings) and generates text that reads unmistakably as each:
**next-letter prediction holds at 63-68% across English, French, German, Spanish and
Italian, with 85-97% real words**, and the samples keep what makes each language look
like itself -- French accents, German compounds, Spanish endings. This is a capability
test, not a new mechanism (the generator is unchanged; only the text differs), and the
limit is the same honest one -- it spells and chains words plausibly without knowing
what any of them mean. (`demo_text_multilingual()` runs it.)

These text demos at first used a plain local k-means and single prototypes rather
than the self-organizing memory built earlier -- so the obvious question is whether
that heavier machinery helps here. Wiring text into it (a new `observe_vector` front
door lets the memory ingest the sentence vectors it did not encode itself) and
measuring gives a clean, honest negative that is itself the point. On the topic
classifier the autonomous memory scores **92%, identical to one-prototype-per-topic,
and keeps exactly one prototype each (zero reorganizations)** -- while naively forcing
three sub-prototypes per topic also scores 92% but spends 3x the memory. Text topics
are linearly separable, so a single prototype per class is already optimal, and the
autonomous memory *measures* that and refuses to split. The same holds on the
heterogeneous Brown genres (it keeps one prototype per genre, matching the baseline).
So the self-organizing machinery's contribution on text is discipline, not accuracy:
the same self-* logic that splits genuinely multi-modal data here verifies that text
does not need splitting and declines to over-engineer. (For blind clustering the plain
k-means stays ahead; the coherence-driven split is fragile on these embeddings -- a
limit worth stating plainly. `demo_text_self_organizing()` shows the comparison.)

That clean negative held only because the topics were easy. Pushed onto genuinely HARD
text -- the Reuters financial categories, whose vocabularies overlap heavily
(crude / trade / money-fx / interest all read alike) and whose classes are internally
multi-modal -- the picture flips. A single averaged prototype mis-files the off-centre
members (~77% across seeds), splitting each class into ~3 sub-prototypes genuinely
helps (~82%), and the autonomous memory now *fires*: it measures the gain on held-out
data and reorganizes, reaching ~81% (76.8% -> 80.7% averaged over seeds). The hard
problem also exposed a real bug: the gate had been comparing split candidates (trained
on a fit slice) against a "keep" model trained on ALL the data, so the splits were
handicapped and the gate under-fired. Fixing it to judge every resolution on equal
footing and then refit the winner on all the data made the autonomous version fire
reliably -- a fairness fix that only a hard, confusable dataset would surface. And the
discipline still cuts both ways: on sentiment (movie reviews), where the failure is the
representation carrying no good/bad signal rather than multi-modality, splitting cannot
help and the memory correctly declines. So the honest full picture is: the machinery is
inert on easy, linearly separable topics (and says so), earns its keep on hard
confusable ones (and fires), and refuses to chase problems splitting cannot fix.
(`demo_text_hard()` runs the Reuters and sentiment stress tests.)

One more question the hard data invites: now that the gate fires, is the rule that
*picks how far to split* itself any good? It chooses one resolution for all labels and,
among resolutions that tie on held-out accuracy within one standard error, keeps the
leanest. Three alternatives were measured against it on Reuters (hard) and the clean
topics (easy): a per-label resolution chosen by greedy coordinate-ascent, a "climb while
each step still earns more accuracy than its own noise" rule, and sweeping the duplicate-
merge grain alongside k. All three were worse or a wash -- per-label and climb both
under-fired (78-79% vs 80%), and sweeping the merge grain changed nothing. The current
rule already sits close to the pure best-k accuracy (80% vs 81%) at roughly half the
prototypes AND still picks one prototype per class on the easy topics, which pure best-k
would over-split on noise. So unlike the two gates above, this one is not a hidden
conservatism bug -- it is a deliberate accuracy-for-leanness trade that measurement says
is already well placed, and it was left alone. The discipline is in checking, and in not
"fixing" what the data says is right.


**A damage-tolerant image archive.** Store a gallery of images superposed into a
few plates; recall the clean original from a noisy / blurred / occluded query,
even after destroying a large fraction of the plate. The bundled demo recalls
6/6 images from each corruption, and still 6/6 (reconstructing at ~52 dB) with
40% of every plate destroyed. Two further capabilities sit on the same store:
**cross-modal recall** (`recall_by_tags`) addresses an image by word/number tags
bundled into a hypervector, so "radial pink" alone returns the right image (6/6
on the demo gallery, no picture needed); and **quantized plates** (`quantize(4)`)
shrink the store ~8x (844 KB -> 107 KB) with content recall still 6/6 and
recovery degrading gracefully (79 -> 30 dB). Both work together - the tag
addresses live outside the plates, so cross-modal recall is unaffected by
quantization.

**A delta set-packer for related images** (`holographic_pack.py`). A follow-on, `pack_sprites.py`, handles palette GIF sprite sets a different way, and the honest lesson there is that delta coding was the WRONG tool: on a real 712-sprite set, unifying everything to one shared 88-colour palette and compressing the index planes with LZMA packs to 69 KB (bit-exact) -- 11.5x under the loose GIFs and 2.3x under zipping the folder -- while every delta variant did worse. The win was the representation, not the diff. Generalising past sprites, `image_vault.py` is a format-, size- and codec-agnostic store that NORMALISES any input to RGBA, RELATES images by a size-invariant fingerprint (similarity, clustering, query-by-example), and COMPRESSES adaptively -- it measures shared-palette+LZMA, LZMA over related-ordered pixels, and per-image PNG, plus optional lossy JPEG/WebP for photographs (measured with PSNR), then keeps whichever is smallest for that set and reports the comparison honestly. Lossless modes are bit-exact; pull any image back by index/name or hand it an example to find the nearest stored ones. Single-file
codecs compress each image alone, so a family of images that shares structure
pays for that shared content in every file. The packer stores a set as one
reference plus per-image deltas (residual mod 256, zlib'd), bit-exact and in 8-bit
integers throughout. On a six-logo suite that shares a background and ring it packs
to ~39% of per-file PNG and beats gzip-ing the whole set; honestly, on images that
are already compressible on their own (gradients, photos) per-file PNG/JPEG win,
and the built-in benchmark shows both so the choice is clear. A lossy
Walsh-Hadamard tier was prototyped and dropped because it never beat JPEG.

**More engine pieces** (each with a runnable demo): residue (exact integer)
arithmetic on vectors, signed-distance regions of the sphere, a predictive filter
that stays quiet on the expected, a unified scalar "field" abstraction,
two-timescale diffusion, Kuramoto-style emergent grouping, a tool orchestrator
with circuit-breakers and reusable skeletons, online unsupervised concept
formation, and a hypervector reaction-diffusion cellular automaton.

---

## How it works

**The core.** Atoms are random high-dimensional vectors, nearly orthogonal by the
blessing of dimensionality. `bind` (a reversible element-wise combine) ties two
together into something dissimilar to both; `unbind` recovers a partner.
`bundle` (normalised sum) overlays things into a set you can still query. A
`Vocabulary` mints clean atoms and `cleanup` snaps a noisy vector back to the
nearest known one. That is the whole toolkit; every subsystem is a different way
of arranging those operations.

**The image archive** adds three image-specific steps:

1. *DCT, keep the big coefficients.* Each colour channel goes through a pure-numpy
   orthonormal 2-D DCT; only the largest `K` coefficients are kept (their
   positions stored as a small bitmask, counted honestly in the size).
2. *Spread with structured keys.* The kept coefficients are scattered and run
   through a **Walsh-Hadamard transform** with a fixed random sign pattern, so
   each one is smeared across *all* `D` plate values - no plate value is special.
   The key operator is matrix-free and an exact isometry, so an undamaged plate
   decodes exactly with one adjoint pass.
3. *Recover.* Undamaged - one multiply. Damaged - a mask marks survivors and a
   small conjugate-gradient solve recovers from what is left, graceful until the
   survivors drop below the stored coefficient count.

Multiple images share one plate via *disjoint* key-slot pools (keeping the
combined keys orthonormal), and content-addressable recall keeps a tiny thumbnail
fingerprint *outside* the plate so recognition survives even when the plate is
wrecked.

---

## Benchmarks vs. existing tech (image archive)

Measured by `bench_vs_jpeg.py` on a 240x240 colour image. Reproduce: `python bench_vs_jpeg.py`.

**Plain compression - JPEG/PNG win, and that's fine.** The hologram is not a
compressor: PNG 1.5 KB, JPEG q85 5.3 KB (29.8 dB), hologram 4-bit 42.2 KB
(27.9 dB). Use a real codec if you want small files.

**Corruption resilience - the hologram wins enormously.** Corrupt the same
*fraction* of a JPEG file vs. plate cells (mean PSNR over 8 trials; 0 dB = no
longer decodes):

| corrupted | JPEG q85 | Hologram |
|-----------|----------|----------|
| 0.1%      | 9.2 dB   | 27.9 dB  |
| 1%        | 4.0 dB   | 27.9 dB  |
| 10%       | 0.0 dB   | 27.8 dB  |
| 40%       | 0.0 dB   | 27.0 dB  |

A JPEG dies at a tenth of a percent (its headers, DC terms, and entropy-coder
state are single points of failure); the hologram is essentially untouched at 40%,
because corruption is just uniform noise with no privileged bytes to destroy. See
`figures/bench_corruption.png`.

**Also measured:** the Walsh-Hadamard keys use ~3,200x less memory than a dense
random-projection matrix and run ~57x faster, with identical fidelity; conjugate-
gradient decoding gives ~8x the usable capacity of a matched filter; 10 images
multiplex into one plate with no crosstalk.

**Batch retrieval (1-bit vs float).** Finding the right stored item from a noisy
query, over a 10,000-item database (`bench_batch.py`): 1-bit sign hypervectors
with Hamming similarity match float32 cosine on accuracy (**100% vs 100% recall@1**
on a 20%-corrupted query) while using **32x less memory** (10 MB vs 328 MB). In
pure numpy the float matmul is faster (BLAS is hard to beat without a dedicated
popcount kernel); the 1-bit win is the footprint, which fits in cache at scale.

---

## Project layout (flat on purpose - everything imports cleanly)

The engine (pure numpy):

    holographic_ai.py         bind/bundle/cleanup, key->value memory, learner, reflex, drift
    holographic_unified.py    TOP LEVEL: one encoder + one self-organizing memory + one brain
    unified_app.py            web console to test the unified mind on pulled corpora
    holographic_encoders.py   numbers (scalar/fractional-power), text, mixed records
    holographic_reasoning.py  resonator, conformal intervals, epistemic map, compass
    holographic_creature.py   grid-world + a holographic RL mind (the forager)
    holographic_navigator.py  the same mind, repurposed to navigate the data tree
    holographic_mind.py       general-purpose front door: any input -> one mind
    holographic_moe.py        mixture of experts with a learned holographic gate
    holographic_organizer.py  self-organizing memory: reorganize a shadow, then swap
    holographic_text.py       text from scratch: learn / analyze / organize / produce
    holographic_extras.py     residue arithmetic, SDF regions, predictive filter
    holographic_field.py      scalar field abstraction (one field, many roles)
    holographic_diffusion.py  two-timescale double diffusion
    holographic_sync.py       Kuramoto-style emergent grouping
    holographic_orchestrator.py  tool planner with circuit-breakers + skeletons
    holographic_emergence.py  online unsupervised concept formation
    holographic_automaton.py  hypervector reaction-diffusion CA (demoscene)
    holographic_image.py      WHT keys, DCT codec, quantised plate, damage decode
    holographic_archive.py    content-addressable multi-image memory
    holographic_pack.py       lossless delta set-packer for related images
    pack_sprites.py           palette-indexed packer for GIF sprite sets (+ bench_sprites.py)
    image_vault.py            format-agnostic store: relate, compress, retrieve any images

The app and tour:

    app.py        Flask UI (system tour + test runner + image recall)
    tour.py       command-line tour of every subsystem
    run.bat       Windows launcher

Tests (162 total):

    test_holographic.py           core engine (bind/bundle/memory/reflex/drift)
    test_holographic_image.py     image store / WHT / quantisation
    test_holographic_archive.py   archive recall + damage
    test_holographic_pack.py      delta set-packer round-trip + size
    test_pack_sprites.py          sprite packer round-trip + size
    test_image_vault.py           vault round-trip, query, clustering, lossy tier
    test_holographic_vision.py    HSV / edges / Hough / shape ID / clustering
    test_holographic_scene.py     compositional tags + multi-object resonator
    test_holographic_tree.py      capacity curve, RP-tree + forest recall
    test_holographic_uri.py       S3-style keyspace, prefixes, bi-level buckets
    test_holographic_navigator.py learned data-tree navigator vs fixed beam
    test_holographic_mind.py      universal encoder + classify / recall / decide
    test_holographic_moe.py       learned gate routes to specialists, beats single
    test_holographic_organizer.py self-organizing + autonomous reorg (no thresholds)
    test_holographic_text.py      word learning, language ID, topic sort, generate, scale, hard, multilingual
    test_holographic_brain.py     self-maintaining, autonomous, hard-shift recovery
    test_holographic_unified.py   top level: one memory across modalities, recall, decide

Research / provenance (run from this folder, e.g. `python exp_wht.py`):

    exp_*.py, bench_vs_jpeg.py (add --fig for the corruption figure),
    bench_sprites.py, benchmark_holographic.py, stress_holographic.py,
    make_test_image.py
    figures/   rendered results

---

## Honest limitations

- **Not a competitive image compressor** - use JPEG/WebP/AVIF for small files.
- **Hard capacity / damage cliff** - image recovery is graceful only until the
  surviving plate cells drop below the stored coefficient count (the demo archive
  sits at load 0.37, so its cliff is ~63%; the UI caps damage at 70%).
- **Small-scale by design** - this is a clear, readable, from-scratch engine for
  learning and experimenting, not a tuned production system. The text corpus is
  tiny, the creature world is small, and the vectors are modest. It is built to be
  understood and extended.
