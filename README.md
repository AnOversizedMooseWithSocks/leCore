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
watch all fourteen subsystems work in ~30 seconds -- it now ends with the unified mind self-assembling from a bare pile of examples.

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
corpus's style, all against the one trained mind. A fifth dataset needs no network at
all and is the inception option: **"This project's own source"** -- the mind learns its
own code, classifies which subsystem a pasted snippet belongs to (held-out ~62% over
five subsystems sharing heavy vocabulary, vs 20% chance), and generates code in the
project's style. Getting that number honest transferred an old lesson to a new format:
pure-punctuation tokens are code's stopwords -- shared by every module, they dilute the
bag exactly like prose stopwords dilute topics, and dropping them lifted held-out
accuracy from 42% to 70% in the controlled comparison (generation keeps its punctuation;
code without parentheses is not code). Queries in this dataset also keep their case
(`HoloTree` is not `holotree`) and run untagged, so the mind self-discovers what it is
being shown.

**Self-discovery and self-assembly** were the next two gaps, and closing them surfaced
one re-measured trap and two clean negatives -- all kept. *Self-discovery:* the routing
safeguard used to depend on the caller's bookkeeping -- pass no modality tag and it
silently vanished. Now the encoder itself names the modality it would use
(`UniversalEncoder.infer`, the single source of truth `encode` also dispatches through,
so the tag used for routing can never disagree with the encoding used), and
`learn`/`classify` discover untagged inputs. The trap: naive type dispatch sends a LIST
of tokens to the order-sensitive sequence encoder -- the *same* encoding bug fixed once
already, sneaking back in through inference -- and scores 93.8% on the mixed-modality
demo. With the one rule that a list of strings is a bag of words, inferred routing
scores 97.5%, *exactly* matching caller-declared tags. *Self-assembly:*
`UnifiedMind.absorb(examples)` builds a working mind from a bare pile of
`(input, label)` pairs -- discovers each item's modality, pre-reads the text it sees so
word vectors carry co-occurrence meaning before anything is filed, learns everything
into the one memory, and runs a maintenance pass; it is deliberately sugar over
`read`/`learn`/`maintain_now` so there is nothing to drift out of sync. The negatives:
wiring the learned *navigator* into unified recall lost badly on the mind's own store
(48% recall@1 at ~130 comparisons, where the fixed-beam forest gets 89% within ~512 --
its arrive/keep-moving margin sense was tuned on uniform random vectors and the unified
store is clustered), so recall keeps the dumb-but-honest index and the navigator stays
a study. And the recall index's *switch-over to the forest* turned out to be 16x too
eager: measured, a single numpy matmul scan is exact AND faster than the tree's
Python-level routing until roughly 4,000 items (at the old 256-item threshold the scan
is ~7x faster and the forest already costs recall), so the threshold now sits at the
crossover -- below it the forest paid more wall-clock for less accuracy.

**Structure before meaning, on a new format.** The destination demands that the same
machinery handle text, code, images, and more by discovering each format's *structure*
first -- and the schema module always claimed its compress-by-merging move was
modality-blind ("idioms and indentation in code"), but code had never been measured. It
now is, and the corpus is -- recursively -- this project's own source: ~500k characters
of it. The discovered schema cuts held-out bits/char from 2.98 (flat character model) to
2.28 (the fractal coder), 24% fewer bits, the same shape of win it earned on Austen. The
emergent chunks, found with no labels from raw characters, ARE Python syntax:
`def __init__`, `rng = np.random.default_rng(`, `)\n        return `, whole indentation
idioms, the banner comment lines. And a compression gate can tell code from prose --
each schema claims its own held-out format -- with one honest caveat that is itself a
finding: feed the code expert this project's RAW source, which is half English
docstrings, and it becomes a *better English model* than a prose expert trained on less,
so the gate mis-routes. Representative corpora are part of the mechanism, not an
optional nicety. (Both results are locked in as tests.)

The third format closed the set: the same primitive on **images**, measured on the
project's own 712-sprite set (each pixel an opaque colour-code atom in raster order). The
honest shape of this one is that the schema is *data-hungry* here: at 60 training sprites
the rare chunks starve and the fractal coder LOSES to the flat pixel model (1.91 vs 1.96
bits/pixel); at 150 sprites it wins on every split tried (1.49 -> 1.30, and 23% fewer
bits at deeper settings -- the same magnitude as text and code). Structure exists in the
format; feeding the statistics is part of the mechanism. And the unified mind now *uses*
the format work: `learn_sequence` takes a modality (so it learns to continue code, not
just prose) and a name, so one mind holds MANY sequence schemas at once instead of a
single slot that silently overwrote. Unnamed generation routes the seed through the
compression gate -- whoever compresses the seed best understands it -- which is
content-level self-discovery, needed exactly where type inference goes blind: code and
prose are both `str`. Measured: a `def encode(self, x):` seed routes to the code schema
and continues as code; a prose seed routes to prose. One routing primitive, reused at
every level of the stack.

The same gate then moved into **classification**, and the measurement that justified it
was a surprise: not a booster, a *correctness fix*. First the encoder needed `"code"` as
a first-class text-like modality -- before that, a declared code tag silently fell to the
opaque-symbol path and two nearly identical snippets encoded as orthogonal (measured
cosine 0.04). With that fixed, one mind learned documentation and source code about the
*same* subsystems (heavy shared vocabulary -- the adversarial case) into the one memory.
The finding: routing's gain over a flat scan is ZERO on this data (bag-of-token vectors
already separate docs from code -- the safeguard story again, one level down), but the
old type-only path was actively destructive -- tags declared at learn time put code
labels in a "code" pool, an untagged classify inferred "text", and the routing safeguard
then *excluded the true labels from competition entirely*: 24% accuracy, 66% cross-pool
leakage, worse than no routing at all. The compression gate, fitted on the mind's own
learned samples (capped, refit only when the corpus grows by a third), identified the
sub-format on 100% of held-out queries and recovered declared-tag accuracy exactly --
~2s one-time fit, ~10ms per untagged string query at steady state. So `classify` now
discovers in two stages: type (`encoder.infer`) for everything, then content (the gate)
exactly where type goes blind. All pinned as tests.

**The slime mould moved into recall** -- the organize half of the navigator study,
salvaged from its negative. The learned navigator lost its place in unified recall
(recorded above), but its `ReflexCache` -- veins thicken toward what you ask most, with
a flux guard so the habit never costs more than it saves -- is separable, and it now
fronts the recall index's big regime (it moved to `holographic_tree`, since a reflex
fronts index machinery, not navigators). Measured at 16k items: on a Zipf workload the
reflex answers 70% of queries, recall@1 *rises* 96.8% -> 99.0% (a popular noisy cue
snaps to the right hot item where the forest's beam sometimes misses), and cost drops
3x; under a popularity *shift* it re-adapts within one rebuild period; on a uniform
stream the flux guard deactivates it and the cost is a wash -- measured, not promised.
Integrating it surfaced a separate embarrassment worth recording: every big-regime
recall had been re-stacking the entire store into a matrix at a measured **54 ms per
call** at 16k items -- thirty times the cost of the search it was preparing for. The
matrix is now cached, so the whole path went from ~56 ms to 0.5-1.7 ms per query.
And `absorb(pile, sequences=True)` completes self-assembly: the one call now returns a
mind that classifies, recalls, AND generates -- one named sequence schema per text-like
sub-format it discovered, fitted from the same capped samples the classify gate uses,
with unnamed generation routed by the same compression gate. The tour's closing segment
runs exactly that.

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
**Set packer** (delta-code a set of related images against one reference), and **Image vault** (the general store: relate by fingerprint, compress adaptively across lossless and lossy encoders with an honest table, and query by example). The Test suite panel auto-discovers and runs every test_*.py (268 at last count; up to five skip without NLTK or its downloaded corpora). The package also ships the real 712-sprite set packed to ~67 KB at `features/sprites.hsp` (which doubles as a live demo of the sprite packer), and the UI uses it in two places: the Image vault runs relate/compress/query on the whole set, and the learning creature is drawn as a real walking sprite (`amg2`) that turns to face the direction it moves and cycles its two walk frames -- with its baked-in background keyed out (flood-filled from the edges) so it shows real transparency over the grid instead of an opaque tile. The creature also runs on an energy mechanic: it starts each life with 100 energy, every step costs 1, each star it reaches gives +3, and stepping on poison empties the battery -- instant death -- so collecting stars and staying alive are the same goal. Finally, a **Vision** panel shows that the image is just numbers: RGB->HSV colour and dominant-colour extraction, Sobel edges with Hough line/circle detection and Harris corners, a geometric shape classifier, and unsupervised *emergent* classes that fall out of clustering simple feature descriptors -- then a VSA prototype classifier (bundle + cosine cleanup) labels held-out shapes, tying the vision work back to the holographic engine. The panel reports each step's accuracy honestly, including where unsupervised clustering tops out. A final **Compositional scene** panel takes the opposite stance to a holistic descriptor: it reads the DCT coefficient layout as a texture tag (finally using the DCT as a feature, not just for compression), pairs it with HSV colour and geometric shape for automatic per-object tags, then encodes each object as a product of attribute atoms and a scene as their superposition -- so a ResonatorNetwork can factor the parts back out. Multi-object scenes now decompose reliably up to ~5 objects: the old ~50%-at-three ceiling turned out to be a scale bug (normalising the scene) plus missing refinement, not a real capacity limit -- keeping the scene as an unnormalised superposition and adding coordinate-descent sweeps recovers 3-4 objects at 100%. A **Scaling** panel confronts the deepest limit head-on: one holographic trace is a bundle with finite capacity (a 2048-d memory recalls 100% of 64 pairs but ~0% of 2048), so instead of one flat store it grows a deterministic recursive tree -- each node a seeded random hyperplane splitting items at the median, each leaf a small memory kept inside capacity, queries descending with a beam that back-tracks into nearby cells. This is the random projection tree of Dasgupta & Freund and, in spirit, how slime mould beats the size limit of pure diffusion by resolving a broad mass into a hierarchical vein network. The flat memory collapses with scale while the tree holds 100%, and search reaches ~96% recall at a fraction of a full scan's comparisons; per-leaf query 'flux' shows the thick-vein / thin-vein structure. A HoloForest of several differently-seeded trees breaks the single tree's recall ceiling, reaching ~100% recall at a fraction of a full scan's comparisons. Finally, a **Content addresses** panel realises the original partitioning idea the way AWS S3 does: no folders, just a flat keyspace where each object's name encodes the hierarchy. The auto-tags (colour/shape/texture) generate a deterministic URI like `red/circle/smooth`, the key *is* the partition path, and a FacetStore supports S3-style prefix listing and CommonPrefixes roll-up. Where the RP-tree splits by meaningless random hyperplanes, this splits by meaning -- readable, queryable paths -- at the honest cost of bucket skew, with key depth as the lever. And the resonator closes the loop: it recovers an item's URI from its content vector alone, so the address is computed from the content. And the skew problem is now handled: build_indexes gives any hot bucket its own in-bucket HoloForest, so content search inside a popular prefix stays sub-linear -- the bi-level structure (semantic prefix outside, geometric forest inside) realised.

### From the command line
    python tour.py                    # guided tour of all subsystems (~20s)
    python holographic_creature.py    # any module runs its own demo
    python holographic_encoders.py    # numbers / text / records demos
    pytest -q                         # the whole test suite (268 tests)

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
so generalization collapsed. We recorded that as "a real ceiling for an
egocentric brain, not a tuning miss" -- and the revision of that claim is part of
the record now: it was not a tuning miss, but it WAS a framing miss (see the
maze gauntlet below; the ceiling fell to 100% without changing the brain, the
senses, or the memory -- only WHERE decisions are spent). Averaging an *ensemble* of independently
trained minds was worse than picking the best single one (their policies differ
too much to average), which is why the UI trains several candidate minds and
keeps the best -- a branching search over policies that beats voting. And the
newest entry: **schema-discovered macro-actions** -- letting the compress-by-merging
schema read a *trained* creature's own trajectories (episodes joined by unique
separators so merges never cross a boundary) and handing the discovered idioms to a
fresh learner as extra actions. The discovery itself works perfectly -- the emergent
chunks are exactly the straight-line runs (`E+E+E`, `W+W+W+W`) a forager's behavior
contains -- but using them LOSES, robustly: open-loop macros drop the clean world
from 9.9 to 6.8 stars and the poison world from 6.3 to a catastrophic 2.5 (blind
commitment walks into poison); making them interruptible (stop on a star or sensed
danger) rescues the catastrophe but still loses everywhere (5.7 and 5.5), across
three seeds. The why is principled: a reactive policy that senses every step can
already produce `EEE` by deciding `E` three times -- the chunk only adds value where
deciding is expensive or perception is poor, and here it throws away exactly what
the per-step sense-decide loop provides, while doubling the action set thins the
exploration statistics. Same lesson as the curation controller: discovered
structure must beat what the substrate already does, and here it does not. The net:
the toolkit lives in the creature where it belongs (a classifier and layered
superpositions for the memory, a recursive branching partition to index it), and
stays out of the places it hurts.

**The maze gauntlet -- gamified debugging.** The mazes are now designed to mirror
challenges the system itself faced, on the principle that a puzzle the creature cracks
without cheating usually carries a lesson back to the brain -- and the first lessons ran
the other way, system to creature. The 9x9 maze ceiling is ALIASING in a costume:
far-apart corridors look identical through egocentric senses, exactly as code and prose
both look like "text". Two system cures were tried, three seeds each, same senses,
nothing global. The COMPRESSION cure (a decaying bundled trace of past actions, the
anti-23k-prototypes move) is a clean negative at every decay tried -- 0% even on the 7x7
that exact mem=4 solves at 97%, because permute-by-age ORDER is precisely the
information that breaks aliasing and bundling erases it; compression must preserve the
distinctions the task needs (the nearest-key-generation lesson again). The
DECIDE-ONLY-AT-CHOICES cure won completely: a corridor reflex (`run_episode(...,
corridor_reflex=True)`) auto-walks forced cells using nothing but the wall senses, so
the brain spends decisions and credit only at junctions. The diagnosis is quantitative:
per-step framing discounted a 26-step exit to gamma^26 ~ 0.07 at the first decision --
nearly invisible -- while junction granularity puts it near 0.4, learnable. Escapes:
9x9 0% -> 100%, 11x11 100%, 13x13 67% (and training runs ~10x faster, since an episode
is ~8 decisions instead of 90). The honest control is pinned in the tests: corridor-
following with RANDOM junction choices already escapes easy mazes (73% at 9x9 -- a
perfect maze has a small junction graph), so the gauntlet requires the brain to BEAT
that control, which it does at every size and triples at 13x13 (15% random vs 67%
trained). The transferable lesson runs both ways: where the system spends its
machinery only at real choice points, the brain must spend its decisions the same way
-- and the credit-horizon arithmetic says WHY ceilings appear when it does not.
`test_creature_gauntlet.py` holds all of it in place.

The gauntlet's second round added LOOPS and HAZARDS, and its first catch was a bug in
the first round's winner. **Braided mazes** (a fraction of dead-ends opened, so multiple
valid routes exist -- the maze costume of competing reorganization candidates) are where
corridor-following can cycle and the brain must add real routing on top: measured at
11x11 braid=0.5, the no-reflex baseline escapes 0%, reflex+brain 100%, and the
reflex-with-random control 50% -- the brain doubles the control. **The poisoned fork**
(braided maze plus hazards, each placed only if a poison-free route to the exit still
exists, so the maze stays honest: solvable, but one arm of some fork is lethal and looks
like the safe one) mirrors confusable classes with asymmetric cost -- and designing it
exposed that the corridor reflex would auto-walk the creature into poison it could
*see*: the fast path had no anomaly handoff. Measured, naive reflex with a trained
brain: 7% deaths, 93% escapes; with a one-line danger yield (the reflex returns control
to the brain when the way forward is sensed as danger): 0% deaths, 100% escapes, three
seeds -- while the random control with the same yield died 88% of the time, so the
brain's contribution stays enormous. The lesson is the flux guard's, running in both
directions: every fast path in the system -- the reflex cache, the format gate, and now
the corridor reflex -- must hand back control at anomalies, and the gauntlet is where
that class of bug gets caught in costume before it gets caught in production.

**The 16x16 room, any seed.** The escalation demanded reliability with no cherry-picked
maze and no map knowledge -- the creature only ever has its senses, learning each maze
the way a rat does, by living in it. Three walls fell, each one a system lesson. First,
ENERGY: a 16x16 maze's optimal path runs 80-108 steps against the then-default battery
of 100, so starvation beat intelligence on most seeds -- the budget must match the world
before the brain even gets a vote. That finding raised the creature's default battery to
300, and the whole any-seed sweep re-validated at the new default (same worst 95%, mean
99%) with no explicit energy override anywhere. Second, the CREDIT HORIZON arithmetic struck again one level
up: at gamma=0.9 the training is bimodal -- runs land at ~100% or collapse to 0% with
nothing between, the brain committing early to a wrong junction policy and then greedily
cycling it to death; gamma=0.97 took the failing maze/brain combinations from 1% to 98%
mean, and IMPROVED the smaller gauntlet mazes too (13x13: 67% -> 100%); epsilon floors
and longer training did nothing -- the horizon was the lever. Third, the stray collapse
that remained (15-run grid: mean 93%, worst 0%) is closed by SPECULATE-MEASURE-ADOPT
over whole policies: `learn_maze()` trains a candidate, probes its real escape rate over
a few greedy lives, and restarts with a different brain seed if it measures incompetent
-- the organizer's rule applied to training runs, and the same train-several-keep-the-
best pattern the UI already uses. Validated across 8 maze seeds: worst 95%, mean 99%,
with the restart visibly rescuing the nastiest seed. The honest frontier is recorded
with it: ZERO-SHOT transfer -- one mind trained across 30 mazes, dropped into 10 it
never saw -- measured 41% against a 36% random-junction control, i.e. no real
competence transfers; the learned junction policy is maze-specific. Earned per maze,
reliable on any seed; portable across mazes is the open problem.

**Survival foraging -- fair, and harsh on purpose.** Foraging and obstacle worlds now run
until the creature DIES (poison or starvation; `max_steps=None`), with the score being
stars collected in a life -- the energy arithmetic keeps every life finite (stars +3,
moves -1, average star ~4.7 steps away, so even a perfect forager runs a slow deficit
from its 300 battery). The fairness is in the baselines, which use the creature's exact
senses: a naive greedy chaser, and a danger-aware greedy chaser that is the bar learning
must clear. The harshness did its job immediately, three findings deep. FIRST, the
bombshell: the trained brain -- whose poison avoidance looked solid in every 50-step test
-- died on poison in 67-73% of full lives (a ~0.6%/step residual risk that short caps
simply cannot see; risks COMPOUND), collecting 13-25 stars where the two-line
danger-aware reflex collected 136 with zero deaths. Fixed by the danger reflex: lethal
moves are vetoed BELOW the brain (`decide` gained `among`, the routing lesson applied to
actions -- compete only within the survivable pool), because irreversible mistakes are
reflex business, not learned preference (the corridor reflex's danger yield and
auto_maintain's asymmetric-cost rule, a third time). Deaths: 0%. An honest negative on
the way: blinding the brain to the danger senses it no longer needs did NOT recover
efficiency (and got worse with training). SECOND, the real thief was DITHERING: the
memoryless forager spent a measured 60% of its steps stepping back where it stood two
steps before, starving at 28 stars; working memory (mem=3) cuts dithering to 10% and
lifts it to ~121 stars -- 89% of the danger-aware reflex's ceiling, the same ratio it
achieves in the clean world, so what remains is chase efficiency, not poison. THIRD, the
open problem -- recorded, then SOLVED by the system's own introspection. The brain
gained describe()/why_differ() (the relations decode turned inward: its states are
role-bound sense bundles, so a prototype reads back out in sense terms -- measured
373/373 present roles decoded, 427/427 absent roles correctly silent). Pointed at a
caught dither, the brain articulated its own bug with precision: the two 'oscillation'
states were sense-IDENTICAL, and it was choosing E at value +0.43 while its own senses
said wall_E=yes -- valuing moves into walls it could see, burning energy on no-ops. The
articulation named the fix: walls join poison in the `among` veto (the wall reflex).
Measured, three seeds: stars 5.1 -> 19.8 (the danger-aware reflex's ~20 ceiling),
dither 79% -> 43%, deaths 0%. Found by introspection, fixed by one line, measured
before believed.

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

**One shared encoder for any input.** Everything above rests on a single
fact: once you encode something into a hypervector, the same operations work on
it -- `bundle`/`bind`/`cosine` do not care where the vector came from. So the
hypervector is a universal interchange format, and the machines built on it (a
prototype classifier, the recursive index, the creature's brain) are a component
library that snaps onto *any* encoded input. `holographic_mind.py` holds the
shared pieces: a `UniversalEncoder` that turns text, numbers, categories, raw
feature vectors (an audio MFCC frame, an image embedding), images, structured
records (a dict of fields), or sequences into one unit vector in one shared
space -- and that can *name* the modality it would use (`infer`), which is what
lets the unified mind discover an input's kind instead of being told (see
below). This file used to also hold a `Mind` facade with an `assemble()` that
guessed the task from data shape; it was retired, because it re-implemented thin
versions of machinery that exists for real elsewhere -- exactly the failing
`UnifiedMind` was built to fix. Its one good idea, building a working mind
straight from a pile of examples, lives on as `UnifiedMind.absorb()`, running on
the real self-organizing memory instead of a toy one.

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
is already well placed, and it was left alone. The rule has now outlasted a fourth challenger: FRACTAL recursive bisection (the same split applied self-similarly at every node, each accept measured) was a wash on hard uneven data and wins leanness only where accuracy saturates -- full numbers in the design notes; recursive self-similarity stays where it measurably pays, the HoloForest index. The discipline is in checking, and in not
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
    holographic_unified.py    TOP LEVEL: one encoder + one memory + one brain + named sequence schemas
    unified_app.py            web console to test the unified mind on pulled corpora
    holographic_encoders.py   numbers (scalar/fractional-power), text, mixed records
    holographic_reasoning.py  resonator, conformal intervals, epistemic map, compass
    holographic_creature.py   grid-world + a holographic RL mind (the forager)
    holographic_navigator.py  the same mind, repurposed to navigate the data tree
    holographic_mind.py       the shared UniversalEncoder (with modality self-discovery)
    holographic_moe.py        mixture of experts with a learned holographic gate
    holographic_organizer.py  self-organizing memory: reorganize a shadow, then swap
    holographic_schema.py     structure by compression: chunk schemas, fractal coder, the gate
    holographic_text.py       text from scratch: learn / analyze / organize / produce
    holographic_tree.py       recursive RP-tree + HoloForest + slime-mould ReflexCache
    holographic_graph_memory.py  routed-descent memory -- a recorded negative for classification
    holographic_slime.py      slime-mould maze solver (discover, then thin to shortest)
    holographic_vision.py     pixels -> features: HSV, edges, Hough, shapes, emergent classes
    holographic_scene.py      compositional scenes: per-object tags, resonator factoring
    holographic_uri.py        content addresses: S3-style keyspace + bi-level buckets
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

Tests (268 total):

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
    test_holographic_mind.py      universal encoder + modality self-discovery + index regime
    test_holographic_moe.py       learned gate routes to specialists, beats single
    test_holographic_organizer.py self-organizing + autonomous reorg (no thresholds)
    test_holographic_text.py      word learning, language ID, topic sort, generate, scale, hard, multilingual
    test_holographic_schema.py    schema discovery across text / CODE / IMAGES + the gates
    test_holographic_slime.py     slime maze solving + tube thinning
    test_holographic_orchestrator.py  typed tool chains, circuit-breakers
    test_holographic_graph_memory.py  routed descent -- pins the recorded negative
    test_holographic_brain.py     self-maintaining, autonomous, hard-shift recovery
    test_holographic_unified.py   top level: one memory across modalities, self-discovery,
                                  absorb, named schemas routed by the compression gate
    test_holographic_relations.py meaning as the recovered relationship: explain/name/map/chain
    test_creature_gauntlet.py     the maze gauntlet: gamified debugging, system lessons in mazes
    test_app_creature.py          the app's creature endpoint round-trip

Research / provenance -- one-off scripts whose results are recorded above and in
`figures/`; none is imported by the library, the app, or the tests. They were
moved out of the root into `archive/` to keep the working set readable, and each
adds the repo root to its own import path so it still runs from anywhere:

    archive/exp_*.py, archive/bench_vs_jpeg.py (add --fig for the corruption figure),
    archive/bench_batch.py, archive/bench_sprites.py, archive/bench_fig.py,
    archive/make_test_image.py        (e.g. `python archive/exp_wht.py`)
    benchmark_holographic.py, stress_holographic.py    (still at the root: these are
                                       the live measurement suites, not one-offs)
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

## Design notes

- **Relations -- meaning as the recovered relationship** (`holographic_relations.py`,
  `UnifiedMind.explain()`): similarity says THAT two things are alike; these
  operations say WHY and HOW. Over role-bound records (what the encoder already
  builds from dicts): EXPLAIN decodes the per-role verdict ("france is like
  belgium BECAUSE currency/language/continent match; UNLIKE because the capitals
  differ" -- 4/4), NAME recovers how a filler relates ("paris relates to france
  AS capital" -- 100%), MAP answers "what is the dollar of mexico?" (100%), and
  CHAIN composes hops ("the language of the country with the currency of the
  country whose capital is X" -- exact through three hops). The measured law
  that shaped the API: meaning survives composition only when it touches
  SYMBOLS between steps -- the direct algebraic relation vector scores ~94%,
  its failures are pure HRR noise, and dimension does not save it (96/94/90% at
  1024/2048/4096), while routing each hop through a cleanup is exact: the
  discrete vocabulary is the error correction that makes chained meaning
  reliable. And the operations are CROSS-MODAL with zero new machinery
  (`holographic_scene.explain_objects`): two raw images go through the existing
  auto-tagger (colour/shape/texture), the tags become role-bound records, and
  the system answers why one image is like another -- "shape SHARED (circle),
  colour differs (red vs green)" -- measured end-to-end on generated shapes at
  72/72 = 100% verdicts (the tagger itself is 36/36 on ground truth), with
  chains working over image stores too ("the colour of the rectangle-shaped
  object"). And the operations are UNIFIED, not a side library: the mind runs
  them on its OWN memory -- find(role, filler) scans the records absorb()
  already stored, read_role() decodes a role from a learned class prototype
  against the filler vocabulary learn() registered from experience, ask()
  chains multi-hop questions over the absorbed store, and explain() takes
  either fresh dicts or two LEARNED labels. The measured payoff of unification:
  classes built from six noisy, incomplete observations each (one random role
  dropped per copy) still decode perfectly -- read 40/40, explain verdicts
  180/180 over all pairs, 3-hop chains 100% -- because superposition linearity
  reinforces the shared role-filler terms while the dropouts average out. The
  mind explains concepts it LEARNED, not just records it is handed. And the
  INCEPTION step: explain_splits()/explain_organization() turn the relations
  decode on the mind's own memory ORGANIZATION -- when the organizer splits a
  class, the sub-prototypes' roles are decoded and contrasted, naming what the
  split separated ('A divided because one mode is colour=red/shape=circle, the
  other blue/square'). Separation is judged by CONTRAST (each mode's winner
  genuinely absent from the other: ~0.5 for real structure vs <=0.1 for
  incidental skew), and the statistic's first outing caught the organizer
  red-handed: one XOR label's split turned out to separate the NOISE role --
  accuracy-sufficient, structurally arbitrary -- and the explanation reported
  it honestly. The same introspection on the creature's brain
  (describe()/why_differ()) articulated and SOLVED the cluttered-forager open
  problem (see the survival section). And the capabilities live in the MAIN
  MODELS, propagated to every consumer: the safety reflexes moved INSIDE the
  brain (`HolographicMind.decide(senses=...)` vetoes moves into seen poison or
  walls; run_episode's flags, the demos, and the showcase app's own creature
  loop all route through the one mechanism -- the on-camera creature no longer
  suicides or wall-bumps), and the unified mind keeps a JOURNAL: every road to
  auto_reorganize narrates itself ("reorganized: 'A' went from 1 to 2
  sub-prototypes, the modes differ in colour, shape"), maintain_now adds the
  decision brain's measured keep/fold/refresh verdict to the same entry (the
  whole self-maintenance story in one place -- and honestly: a brain that
  self-maintained mid-stream reports "kept", because by maintenance time
  nothing WAS stale), with the console's organize panel showing the mind's own
  account verbatim. The operations now run on REAL data twice over: the
  712-sprite library absorbs as image + auto-tag/name record per label (roles:
  colour, texture, family, facing, frame), and the measured new result is that
  role decode survives a MIXED prototype -- an image vector superposed with the
  record -- at 100% (750/750), with the cross-modal loop closing at 96%: SEE a
  sprite, classify it, SAY its colour in symbols. And the console gained a
  'Countries (records)' dataset (ten countries from eight noisy observations
  each, 97% held-out) plus a RELATIONS panel: explain two learned labels
  per-role, find by attribute, chain a two-hop ask -- the mind answering WHY
  over its own memory, in the browser -- the ask row takes arbitrary chains
  ('capital>currency, currency>language'), each hop cleaned up to a symbol. PROVENANCE -- can a generated or pasted passage be traced to its sources?
  The stores already hold the answer: the recall index keeps every absorbed item
  with its payload (find() returns exact provenance), and the sequence model now
  records, for every context->token transition, WHICH source documents taught it
  (a doc-counter beside each count, zero cost when unused). attribute(text) ranks
  the fitted sources by the transitions a passage actually uses -- and the
  measurements drew a sharp, honest boundary. Attributing GIVEN text is the
  well-posed question: 70% top-1 on a clean four-book Gutenberg split, 92% on
  five, 8/9 windows localized in spliced text -- and the level matters
  (coarsest-chunk-first measured 70% vs 42% atom-only and 48% all-levels,
  because an author's characteristic multi-character chunks are the signal while
  'th'->'e' is shared by everyone). Attributing freely-GENERATED low-order text
  is NOT well-posed: after the seed it drifts into transitions every source
  shares, so the ranking goes near-uniform -- and the UI says so rather than
  faking confidence. An inverse-document-frequency refinement was tried and
  measured a wash (the multi-order context already carries the distinctiveness),
  so it was dropped. COHERENT RESOLUTION (the default) realizes a sharper
  principle: a passage usually comes from ONE source, so a transition only one
  source taught (the word 'fillet' in one book) is near-certain provenance while
  a shared one ('butterfly' in three) is weak -- so each transition's vote is
  weighted by its SPECIFICITY (inverse number of sources that taught it), and the
  unique tokens PIN the source while the shared tokens confirm rather than smear.
  Measured: lifts confidence on ambiguous short passages (+2-4 points top-1 at 60-100
  chars, a wash where evidence already saturates) and sharpens the margin
  (a Melville probe went 0.37 -> 0.68). A sequential RUNNING-PRIOR on top of this
  (let the leader-so-far bias later tokens) was tried and measured a
  wash-to-negative -- specificity already captures the insight, and a feedback
  loop risks runaway commitment to an early wrong guess -- so it was kept out.
  The console gained a provenance panel: generate then 'Trace sources', or paste
  any passage and see the source ranking as weighted bars.

  SEQUENCE ALIGNMENT then closed the gap the bag could not: it answers "whose
  STYLE" but not "whose actual MATERIAL", and a sentence sharing every word with
  sources of OPPOSITE message (a bullish vs bearish thesis differing only in
  'up'/'down') is attributed by the bag to the wrong source. Meaning is in the
  ORDERING, and nature solves this the way genome alignment does -- identify a
  fragment by its longest contiguous verbatim match, not its token composition.
  align() scores maximal verbatim spans by length x specificity; measured 100%
  top-1 on verbatim-clause probes (bag 97%) at a ~3.5x margin, and it gets the
  bull/bear theses BOTH right where the bag confidently picks wrong. trace()
  reports STYLE and MATERIAL and leads with whichever is decisive (a long
  verbatim span => quoted/assembled; none => paraphrase/original-in-style); the
  console shows the verdict, its basis, the deciding span, and both rankings.

  SEQUENCE / ORDER / TIME as a first-class property (holographic_sequence.py).
  A sweep prompted by a sharp observation -- the same steps of a peanut-butter
  sandwich in the wrong order are not a worse recipe, they are not a recipe --
  found the stack treats most things as order-FREE, rightly (topic = bag of
  words, class = bundle of examples, record = set of bindings; "what is this
  about" does not depend on order). But some meaning lives ONLY in the sequence
  (plans, recipes, proofs, protocols, timelines), and nothing could QUERY order.
  SequenceMemory fixes that with the same primitives (bind/bundle/permute): each
  step rotated by its position, order recoverable. Measured 100%: step(i) reads
  the i-th step, position_of(x) finds where x occurs, precedes(a, b) answers
  whether a precedes b, and validate(constraints) runs the PB&J test -- does
  every 'a before b' rule hold? -- naming exactly which step is out of order.
  (A what-comes-next encoding measured ~64%, the bundle-capacity ceiling the
  scaling work charted, so next-step is left to the exact list; this memory owns
  the ORDER RELATIONS no bag store can answer.) Wired into the unified mind
  (learn_plan/step_at/precedes/validate_plan) over the SHARED symbol space. The
  encoder already made the right call elsewhere: a word-list infers as an
  order-free bag for classification (97.5% vs 93.8% via the sequence path) --
  order is restored where it carries meaning, not blanket-applied.

  SELF-DISCOVERED SEQUENTIALITY -- the organizer learning, without being told
  and without a magic number, that a class is ORDERED. The honest test is a
  permutation test against the data's OWN shuffle: does the real order of a
  class's members predict the next element better than the same members with
  order destroyed? A transition model is built leave-one-out and scored by how
  much higher the true next element ranks than the others (a graded margin, not
  argmax -- argmax saturates on small step vocabularies); the baseline is the
  mean over shuffled copies, so the class is its own null hypothesis. The result
  is a z-score (signal in units of the null's own spread), and z>2 -- the
  standard 'two sigma above noise' bar, a statement not a tuned constant --
  calls a class sequential. Measured: ~+16 for genuinely ordered classes, ~0 for
  an order-free bag of the same elements (real order indistinguishable from
  shuffled), degrading gracefully through partial noise (still strong at 30%
  scrambled, at the boundary near 50%, silent once order is gone). A class that
  passes gets its canonical order SELF-ASSEMBLED from the members by a pairwise-
  precedence vote -- the mind reconstructs a sequence it was never shown whole,
  exactly, even from drop-one partial observations -- and gains order queries
  (precedes/validate). The mind absorbs sequential and bag classes mixed,
  discovers which is which, and acts only on the real structure: order as a
  DISCOVERED organizational property, measured into existence, not declared.

  RECURSIVE / FRACTAL discovery -- the same order-test applied at every layer.
  Once a class proves sequential, each of its steps is tested for its OWN
  internal order (where the data provides sub-observations), and the structure
  unfolds into a tree the mind was never given the shape of: a nested recipe's
  top order is recovered, its expandable steps (make_sauce, prep) recurse into
  their sub-recipes, and the recursion STOPS honestly -- at atomic steps with no
  sub-observations, and (the real test) at steps that HAVE sub-observations but
  in unordered form (a garnish whose ingredients carry no order is correctly
  NOT expanded, told from an ordered sub-recipe by the permutation test alone).
  No depth is declared, no shape assumed; each layer is measured into existence
  by the same z>2 bar, and self-assembly recovers each layer's canonical order.
  Sequence discovery made fractal: structure all the way down, until the data
  says stop.

  SELF-PROOF + CONTEXT-BINDING -- structure must prove itself before its meaning
  is trusted, and steps are generic until context fills them. Two additions. (1)
  A discovered order can score z>2 yet be INCONSISTENT: if members' pairwise
  precedences form a cycle (A before B, B before C, C before A) no ordering
  satisfies them and the plan cannot be executed. prove_executable does a
  topological feasibility check -- structure earns trust by passing, not by z
  alone -- and gating registration on it immediately caught a real bug: a
  score-heuristic canonical sort had misplaced a rare step against a 4-0
  majority; the proof surfaced it, and a proper topological sort fixed it
  (structure validating structure found an error before it shipped). (2)
  extract_template discovers the generic SCHEMA and its context-bound SLOTS in a
  repeated step: 'the material has density X' is fixed words plus a slot that
  varies across observations (5g, 3g, ...), separated by per-position entropy and
  split at the natural largest GAP (the data's own scale, no constant). This is a
  physical law -- 'F = m*a' is generic until a scenario BINDS the values; the
  schema is the law, the slots are where context enters, exactly as 'open the
  book' leaves 'book' to be filled from prior context.

  EXECUTION -- the closed loop, from discovering structure to ACTING on it.
  execute_plan runs a discovered, PROVEN plan under an honest contract: a step
  fires only when every step that must precede it has already fired AND its
  context slots can be bound from the scenario; otherwise it BLOCKS, reported
  with its reason (an unmet precondition naming the steps still needed, or an
  unbound slot naming the missing context) rather than silently assumed away. A
  templated step fires as its bound form ('cut into 2 pieces' when context
  supplies pieces=2); without the binding it blocks, and steps behind it
  cascade-block truthfully. An unproven plan cannot be run -- you cannot execute
  what discovery and proof never registered. The full arc stands: discover order
  (permutation test) -> prove it executable (topological feasibility) -> bind
  context into its slots -> RUN it, every stage measured or proven, every failure
  informative.

  WIRED THROUGH THE STACK (not living in tests). absorb() now AUTO-DISCOVERS
  order: hand it ordered list-examples and it runs the permutation test on each
  class, proves the winners executable, and registers them -- order is a property
  of self-assembly, not a manual call. Verified at scale: 240 mixed
  procedure/bag examples absorbed in one call, all four procedures identified
  with EXACT canonical-order recovery and both bag classes left alone. The
  CREATURE uses it: a trained maze brain's successful escape routes are captured
  (capture_route) and the sequence machinery discovers their route is genuinely
  ordered (z up to ~68 vs its own shuffle) and proves it executable -- the
  creature acts, then understands the structure of its own action, surfaced live
  in the showcase maze panel. And the UI exposes the whole pipeline: a
  plan-discovery panel (/api/plan) absorbs noisy procedure observations mixed
  with bag distractors and shows, with no labels, the mind discovering which are
  ordered, proving them, recovering the order, and executing under the honest
  contract (in-order fires, out-of-order blocks). And the discovered plan composes with action:
  replay_plan drives navigation from a proven route instead of re-deciding each
  step, validating every move -- in its own maze the plan escapes 10/10, in a
  CHANGED maze it detects exactly where it breaks (the blocked cell) rather than
  falsely succeeding, so the creature knows the boundary of its learned structure
  and the seam where it would need to re-learn.

  THROUGHPUT (the raytracing parallel). A relation chain is a ray bouncing
  through the holographic space -- each hop a bounce, the cleanup-to-a-symbol the
  surface intersection, the cleanup confidence that bounce's reflectance. Path
  tracing accumulates THROUGHPUT (the product of reflectances) and terminates
  paths that lose too much energy; both transfer. ask_traced accumulates the
  per-hop confidences, and the product is a calibrated trust in the chained
  answer: on a dense interfering store it separates correct chains (~0.23) from
  wrong (~0.10), and abstaining on the low-throughput half lifts answered
  accuracy from 71% to 85%. A chain whose throughput decays below a floor
  ABSTAINS rather than emitting noise -- the ray that ran out of energy
  contributes nothing. The console relations panel shows the answer, its
  throughput, and the per-hop confidences. (A revisit of the kept negatives
  against the new machinery re-confirmed them: competence-weighted flocking still
  loses to best-pick -- in the regime a committee should help, no candidate is
  good enough to yield a signal to align toward, and once they are, best-pick
  already wins -- and the fractal/curation negatives still stand, the new tools
  not displacing a gate that already harvests its signal. A negative is overturned
  by a measured win, not a fresh analogy.)

  MULTI-RAY (many rays per pixel). One query is a noisy point sample; path tracing
  fires many rays and averages, and the same recovers errors a single encoding
  makes. classify_robust fires several word-resampled views of a text query and
  combines them -- the crucial step is Z-SCORING each ray's per-label evidence
  before summing, so a confident-but-wrong view cannot dominate (the naive vote's
  failure, and flocking's). Measured: with feature lenses ranging 100/100/50/17%
  the z-scored ensemble reaches the best single lens BLIND, and on a noisy text
  task it lifted classification 89% -> 100% with no regression on clean queries.
  Each view is a SHADOW of the input from a different angle; the ensemble is the
  form no single shadow shows. The console classify panel reports the multi-ray
  label and the fraction of rays that agreed.

  MULTI-RAY CHAINS, by contrast, are a clean NEGATIVE for accuracy -- and the
  contrast is the lesson. Firing several throughput-traced relation routes to one
  answer and combining them does not help: a route through a unique key is already
  exact (nothing to add), and routes through shared values fail for the SAME
  reason (correlated errors), so combining averages noise -- naive voting even
  made a perfect route worse (100% -> 75%), reliability-weighting only matched the
  best route, and where all routes were ambiguous the combo (27%) lost to the best
  single (52%). Multi-ray helps only when the rays' errors are INDEPENDENT, which
  the feature-lens classification views are and the chain routes are not. The kept
  artifact is route_reliability: a self-measured 1 / mean-fan-out (unique role =
  exact key = 1.0, shared role = ambiguous = low) that ranks which find()
  operations to trust, no magic number -- a good negative leaves something behind.

  PROJECTION TO CREATE NEW THINGS. Casting one record's attributes onto another's
  frame synthesizes a NOVEL entity -- 'france with japanese language and the yen'
  -- that exists in no training data and decodes back to exactly the intended
  blend (100% over random blends). blend() does this directly; project_transform()
  does analogy AS GENERATION (the a->b per-role delta projected onto c creates a
  coherent hybrid: japan's geography, germany's distinctive capital and language).
  The honest split the investigation found: retrieval analogy (FIND the existing
  d) hits a uniqueness wall -- the cleanup law makes every entity an exact key, so
  there is no graded nearness for a transform to climb -- but GENERATION (MAKE the
  specified new thing) is well-posed and exact. Creation sidesteps the wall
  retrieval hits; the line a good negative names is the one between finding and
  making. Wired into the knowledge store and the unified mind (synthesizing over
  its own learned classes), and shown in the tour. And projection lifts to multi-object
  SCENES, where the parts must first be discovered: blend_scenes takes two scene
  vectors (objects unknown), factors each into its objects via the resonator,
  projects one factor across (scene A's forms wearing scene B's palette), and
  recomposes a NOVEL scene that factors back to exactly the intended hybrid
  (100% across all three factors and 2-4 separable objects). The full decompose
  -> project -> recompose loop, all through the resonator -- the part that was a
  recovery tool now drives generation. Honest boundary: recovery rides the
  resonator's capacity (separable objects exact; colliding objects degrade as
  multi-object factoring always does). Shown as a third 'projection blend' demo in
  the scene panel and in the tour. Projection then unfolds over TIME: a smooth
  attribute morph is impossible (interpolating one colour atom red->blue is a
  crossfade-with-snap -- the resonator reports red until t~0.55 then flips hard,
  the cleanup law holding discrete coherent states), so the honest morph is a
  SEQUENCE of discrete coherent frames: a control parameter sweeps 0->1 and the
  objects adopt B's attribute one at a time, every frame factoring exactly, A
  first and B's full pattern last. morph_scenes builds it, and the loop closes
  with the sequence machinery -- the morph as a flip-count token sequence passes
  the sequentiality permutation test (z~10 vs its own shuffle), so projection
  generates the frames and sequence-discovery confirms the order. Shown as a morph
  strip in the scene panel and in the tour. And cardinality itself morphs: the
  object COUNT is self-measured (the scene is an unnormalised superposition of
  near-orthogonal unit products, so round(||v||^2) IS the count -- 96% exact over
  n=1..7, nobody tells the system n), and the scene vector is ALGEBRAICALLY
  EDITABLE -- removing an object is subtracting its factored product (explain-away
  repurposed as an editor), adding is adding one. morph_cardinality chains such
  edits from a 3-object scene down to one and up into a different 2-object scene,
  the count discovered at every frame, each frame factoring exactly, the final
  edited vector holding exactly the target -- never re-encoded. The composite is
  countable and editable, not just decodable. The same algebra becomes the creature's
  PERCEPTION: WorldView encodes the world's contents (exit, poison, walls) as a
  superposition of type(x)position products, so the count is the norm and the
  DIFF of two snapshots is itself a composite of the changes -- appeared objects
  positive, vanished negative, unchanged content cancelling exactly. The diff's
  norm counts the changes and count-driven peeling names them (100%/100% over
  mutated 16x16 mazes). The integration: a wall dropped on the creature's learned
  route makes replay_plan break at exactly that cell and WorldView independently
  names that wall -- perception explains the plan failure (6/6 at 9x9, 5/5 at
  16x16). Stress sweep: 16x16 escapes 100% across six seeds, braided+poison forks
  100%; 20x20 is the measured wall -- partly budget-shaped (one seed 0%->83% with
  more episodes/steps) and partly the BOOTSTRAP problem (another seed stays 0%
  under any budget or horizon: epsilon-greedy exploration never finds the first
  success in a deep-enough maze, so no reward signal ever arrives). The honest
  next step there is curiosity-driven exploration or a curriculum, recorded as a
  future thread. That thread is now pulled: the BOOTSTRAP RESCUE. (And a
  refinement found while pinning it: the wall's mechanism is not 'luck is
  hopeless' -- sustained high epsilon occasionally escapes -- but 'the
  loop-attractor policy locks in as epsilon decays, before luck consolidates';
  plain probes 0% at every budget tried.) Curiosity (first-visit bonus =
  exit_reward / n_free_cells, the world's own arithmetic; off at first escape,
  because visited-ness is not in the creature's state and the crumbs are
  unlearnable after their job is done) finds the first success (episode 4 vs
  never); rehearsal (one stored successful trajectory re-remembered per episode)
  consolidates it; capacity (512/30 where 256/15 loops in a 14-cell attractor)
  holds it. And measurement cut both ways: the same protocol HURT the seed where
  luck already sufficed (83% plain -> 0% with it; rehearsal alone 33%), so the
  integration is a rescue summoned by self-measurement -- candidates run plain,
  and only an observed starvation (zero training escapes, the data's own signal,
  no threshold) enables the bootstrap for subsequent candidates
  (bootstrap="auto", learn_maze's default). Verified: the luck-sufficient seed
  routes plain to 83%, the formerly impossible seed starves, rescues, and probes
  100%, and the 16x16 six-seed regression stays 100%.
  A WIRING SWEEP then made sure nothing
  stayed hidden in tests: the app's labyrinth pane carried a stale "no
  reactive brain can hold a 16x16 maze" early-return from before the gauntlet
  broke that ceiling, so the panel now shows TWO SOLVERS, ONE SUBSTRATE -- the
  brain that LEARNED the maze on the left (learn_maze protocol; escapes the
  braided 16x16 in 42 steps on camera) and the slime-mold colony computing the
  optimal 38-step tube on the right; the forage modes now TRAIN with the
  brain's safety reflexes (the veto shapes the experience learned from, not
  just the final moves -- measured in the app's own walls world: 48 -> 83
  stars across six lives, both deathless); the unified mind's decide() passes
  senses through to the same model-level vetoes; and every decision frame in
  the creature animation now carries the brain's own account, decoded live
  ("senses food_x=west, wall_S=yes -> W (value +0.79)") -- introspection on
  camera. GENERATION FIDELITY (user-caught): generated text had no capitals or
  punctuation -- the engines were innocent (the fractal coder takes raw
  characters), but every console loader fed them a scrubbed token diet
  (lowercased, isalpha-filtered), and BOTH generate endpoints lowercased the
  seed. The loaders now feed TRUE corpus text, measured on Austen: ~12% more
  bits/char (1.949 -> 2.175) and 8 points of word coherence for output that
  reads as PROSE -- capitals, commas, apostrophes, sentences. The flat n-gram
  gained a fold_case switch (default preserves every pinned number). And one
  engine lesson the debugging earned: tiling a small corpus (block x N) makes
  the whole block the OPTIMAL compression unit, so the chunk schema learns a
  corpus-sized mega-chunk and generation replays it wholesale -- the schema
  was right, the diet was degenerate; varied, independently-shuffled passes
  fix it. A seed that encodes to nothing at coarse chunk levels now descends
  to finer conditioning instead of trusting the unconditional prior
  (bits/char unchanged at 2.175). The showcase app also joined in with a
  'Compare two sprites' panel: pick (or randomize) two real sprites, see the
  per-role verdict decoded holographically next to the actual images, and
  below it the cross-modal loop live -- the mind is shown each IMAGE with no
  name, classifies it against the whole library, and states the colour in
  symbols (SEE -> SAY; the first run builds the relations memory over all 712
  sprites, ~1 min, then instant). Absorbing the library with FAMILY as the
  label closed the inception loop on real data too: every family split, and
  the journal named the splits by the genuine within-family modes -- facing
  and frame for the walk-cycle families, colour for the npc grab-bag. Building
  the panel also caught a propagation miss the suite could not see: the
  showcase's embedded unified panel had its own organize endpoint that never
  learned to show the journal story -- fixed, both consoles now narrate. Pinned in
  test_holographic_relations.py, test_holographic_unified.py, and
  test_holographic_brain.py.
- **Projection consolidation** (`HolographicMind.consolidate()`): the brain's
  thousands of 512-D prototypes are shadows of one low-rank object (the span of
  its sense-atom vocabulary -- measured: 99.9% of their energy in 22-24
  dimensions), so the memory is re-stored as coefficients in the SVD-discovered
  subspace: **21x smaller, ~5x faster decisions, at behavioural parity** (forage
  122 -> 120 stars, 16x16 maze 90% -> 95%). The measured hazard ships with its
  cure: a shadow hides new structure (a poison-free consolidation left the
  danger sense at 4% in-basis energy -- nearly invisible), so a residual guard
  tracks out-of-basis energy and EXPANDS the basis when the world grows
  structure the shadow cannot show (measured under a shift: rank 9 -> 13,
  danger 4% -> 100% visible). Compress when stable, grow at anomaly -- the
  flux-guard pattern's fourth appearance. Pinned in test_holographic_brain.py.
- **`NOTES_concepts.md`** records natural-process analogies (double diffusion /
  salt fingering, surface tension, gravity lensing, flocking, prism/spectral
  decomposition, demoscene) considered as possible improvements, and what honest
  measurement said about each. Two were tested to clean negatives -- the
  salt-finger variance pre-screen is mathematically unavailable at 512-d
  (separation decays from 3.9 sigma at dim 8 to noise by dim 128), and
  flocking-style local policy consensus loses to measured best-pick when
  candidates disagree -- and a third (prism) had its premise refuted cheaply
  before it was built (wall-pocket dithering is not caused by state fusion). The
  value is the recorded reasoning: the analogies generated hypotheses, the
  measurements killed the wrong ones early, and the elimination sharpened the
  real open problem (a value-estimate trap, not a representation one).
