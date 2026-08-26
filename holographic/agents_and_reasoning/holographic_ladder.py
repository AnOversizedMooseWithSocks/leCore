"""holographic_ladder.py -- the abstraction ladder: climb a corpus into a tower of levels (L1).

WHY THIS MODULE EXISTS
----------------------
We kept running the SAME seven-step loop by hand on different substrates: letters->words (BPE), verts->mesh->
instance->scene (canonical element + delta chain), transforms->grammar->corpus. Each time: take many things,
consolidate to atoms, find patterns of atoms, promote patterns to a new alphabet, and repeat one level up. Every
RUNG of that loop already exists in the engine. What did not exist is the ORCHESTRATOR that runs the loop
generically and, crucially, KNOWS WHEN TO STOP. That is this module. It contains NO new math -- it delegates the
pattern-finding to the chunk codebook (iterated pair promotion) and measures with an honest MDL bit count.

THE STOPPING GATE IS THE WHOLE POINT
------------------------------------
Pair promotion will happily memorize noise, so "climb until it stops" would build an infinite tower of garbage.
The ladder climbs only while each new level's MDL gain (bits saved) clears a floor. When the gain drops below the
floor, the ladder has hit the ceiling FOR THIS DATA -- and that is a RESULT to log loudly, not a failure. Most
data has a shallow ceiling; a deep tower is the rare, real pocket. (Wolfram's framing: reducibility pockets are
rare; finding one is the win. We take the lesson, not the apparatus.)

DESIGN DECISIONS CARRIED IN FROM THE PLAN + PANEL
-------------------------------------------------
* STABLE ATOM IDENTITY (sweep-mandated): a promoted pattern's identity is a hashlib content hash over its
  constituent atom ids -- reproducible across rebuilds and sessions, never Python's salted hash(). Without this a
  persisted tower loaded next session has orphaned atoms.
* LEVEL ISOLATION (sweep-mandated): each level tags its atoms with its depth, and a contamination check verifies
  a level-k cleanup never snaps to a level-j atom -- the shared-kernel-is-not-a-shared-manifold lesson, enforced.
* GAIN PRE-GATE (Quílez Q5, "don't march empty space"): before the expensive promotion pass, a CONSERVATIVE
  upper bound on achievable gain (the zeroth-order entropy floor) prunes a level that cannot possibly pay. A
  conservative bound never over-estimates gain, so it never wrongly prunes a real pocket.
* prior_tower= (Quílez Q7, temporal reprojection): reserved from day one (default None = batch). The streaming
  climb will reuse the previous window's tower as a prior instead of rebuilding; today the batch path exercises
  the None branch so the parameter is additive later.

Deterministic (PYTHONHASHSEED=0, hashlib for ids, seeded rng). Additive: a new module, nothing existing changes.
NumPy / stdlib only.
"""

import hashlib
import math
import numpy as np

from holographic.agents_and_reasoning.holographic_chunkcodebook import learn_chunks


# ---------------------------------------------------------------------------------------------------------
# Atom identity -- stable across rebuilds and sessions (hashlib, never hash()).
# ---------------------------------------------------------------------------------------------------------

def atom_id(parts, depth):
    """A STABLE content id for a promoted atom: a short hex hash over its constituent atom ids and its level
    depth. Reproducible across rebuilds/sessions because it uses hashlib, not Python's salted hash(). Without a
    stable id, a tower saved this session has orphaned atoms next session. `parts` is a tuple of the child atom
    ids (ints or strings); `depth` is the level this atom lives at."""
    key = ("|".join(str(p) for p in parts) + "@" + str(depth)).encode("utf-8")
    return "a" + hashlib.sha256(key).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------------
# Honest MDL cost + the conservative gain pre-gate (Quílez Q5).
# ---------------------------------------------------------------------------------------------------------

def corpus_bits(corpus, n_symbols):
    """Honest MDL cost of a `corpus` (a list of sequences of symbol ids) under an alphabet of `n_symbols`: the
    zeroth-order coded length -- sum over symbols of -log2 p(symbol), where p is the empirical frequency. This is
    the length an optimal prefix code achieves; it is the number every level-up must beat. WHY zeroth order (not
    a fancier model): it is the honest floor the promotion is trying to improve on, and it is cheap and exact."""
    counts = {}
    total = 0
    for seq in corpus:
        for s in seq:
            counts[s] = counts.get(s, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    bits = 0.0
    for c in counts.values():
        p = c / total
        bits -= c * math.log2(p)
    # add the alphabet's own storage (a code table costs bits too -- honest accounting, not just the stream).
    bits += n_symbols * math.log2(max(2, n_symbols))
    return bits


def _achievable_gain_bound(corpus, current_bits):
    """A cheap CONSERVATIVE proxy for how many bits a re-alphabeting could save (Quílez Q5, 'don't march empty
    space'). Zeroth-order entropy cannot see the repeated-pair structure that chunking exploits, so it is the
    WRONG bound (it roughly equals current_bits). Instead we ask a cheap general compressor (zlib) how
    compressible the symbol stream is: if zlib -- which finds repeated substrings, the same thing pair-promotion
    does -- cannot shrink the stream, chunking will not either. Returns an OPTIMISTIC upper bound on achievable
    gain (so the pre-gate only prunes when even the optimistic case fails): current_bits x (1 - zlib_ratio),
    where zlib_ratio is compressed/raw. WHY optimistic-and-conservative-together: we want to prune ONLY levels
    that provably cannot pay, so the bound must never UNDER-state achievable gain -- zlib on the raw bytes is a
    strong, cheap lower bound on redundancy, and we scale current_bits by it."""
    import zlib
    # pack the symbol stream to bytes for zlib. Map symbol ids to a compact byte range where possible.
    syms = [s for seq in corpus for s in seq]
    if not syms:
        return 0.0
    lo = min(syms)
    packed = bytes(((s - lo) & 0xFF) for s in syms)            # cheap 1-byte packing (collisions only inflate size => conservative)
    comp = zlib.compress(packed, 6)
    ratio = len(comp) / max(len(packed), 1)                    # < 1 means compressible
    redundancy = max(0.0, 1.0 - ratio)                         # fraction zlib could remove
    return current_bits * redundancy                           # optimistic bits a re-alphabeting might save


# ---------------------------------------------------------------------------------------------------------
# The sequence lens -- the only lens shipped in L1 (structure lens is L2).
# ---------------------------------------------------------------------------------------------------------

def _sequence_lens_promote(corpus, max_merges, min_count):
    """One promotion pass with the SEQUENCE lens (adjacency = position): delegate to the chunk codebook's pair
    promotion, then re-express the corpus over the promoted alphabet using the codebook's OWN encoder. Returns
    (new_corpus, new_atoms) where new_atoms maps each promoted symbol id to the (a,b) child pair it stands for.
    Pure delegation -- the codebook assigns the ids and does the encoding; we just read its merge table."""
    # flatten to one stream with a sentinel the codebook won't merge across (keeps sequences from bleeding).
    SEP = -1
    stream = []
    for seq in corpus:
        stream.extend(seq)
        stream.append(SEP)
    cb = learn_chunks(stream, max_merges=max_merges, min_count=min_count)
    # cb.merges is [((a,b), new_id), ...] with ids ALREADY assigned by the codebook. Read them directly.
    new_atoms = {new_id: (a, b) for ((a, b), new_id) in getattr(cb, "merges", [])}
    # re-express each sequence with the codebook's own encoder (applies the merges in learned order).
    new_corpus = [cb.encode(list(seq)) for seq in corpus]
    return new_corpus, new_atoms


def _structure_lens_promote(corpus, max_merges, min_count):
    """One promotion pass with the STRUCTURE lens (adjacency = SHARED GROUP, not position): promote co-occurring
    SETS of atoms into a single part, regardless of order. This is the verts->part->assembly climb -- a repeated
    sub-assembly is the same part wherever it appears, in any vertex order, so promotion must be
    order-INDEPENDENT (unlike the sequence lens). Each corpus entry is a GROUP (a bag of atom ids: a scene's
    parts, a part's verts). We find the most frequent UNORDERED atom PAIR co-occurring in a group, promote it to
    a new atom, and replace both members with it -- the canonical-element move (one representative for a repeated
    structure). Returns (new_corpus, new_atoms).

    WHY unordered pairs and not the sequence codebook: the pinned negative is that morphemic CONCATENATION gives
    zero decomposability -- levels must promote role-filler/set records, not ordered strings. A scene is a SET of
    parts; two scenes with the same parts in different order are the same scene. Promoting ordered pairs here
    would miss that and re-derive the sequence lens badly."""
    from collections import Counter
    # count unordered co-occurring pairs within each group.
    pair_counts = Counter()
    for group in corpus:
        uniq = sorted(set(group))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pair_counts[(uniq[i], uniq[j])] += 1
    if not pair_counts:
        return [list(g) for g in corpus], {}

    # promote the most frequent pairs (up to max_merges) that clear min_count, deterministically ordered.
    promoted = [(pair, cnt) for pair, cnt in pair_counts.items() if cnt >= min_count]
    # deterministic order: by count DESC, then by pair (content order) -- never dict/insertion order.
    promoted.sort(key=lambda pc: (-pc[1], pc[0]))
    promoted = promoted[:max_merges]

    next_id = max((s for g in corpus for s in g), default=0) + 1
    new_atoms = {}
    pair_to_id = {}
    for (pair, _cnt) in promoted:
        pair_to_id[pair] = next_id
        new_atoms[next_id] = pair
        next_id += 1

    def apply(group):
        s = set(group)
        # apply promotions in the same deterministic order; replacing a co-present pair with its new atom.
        for (a, b), _ in promoted:
            if a in s and b in s:
                s.discard(a); s.discard(b); s.add(pair_to_id[(a, b)])
        return sorted(s)

    new_corpus = [apply(g) for g in corpus]
    return new_corpus, new_atoms


# ---------------------------------------------------------------------------------------------------------
# The runner: climb.
# ---------------------------------------------------------------------------------------------------------

def climb(corpus, lens="sequence", max_depth=8, min_gain=0.02, max_merges=200, min_count=2,
          prior_tower=None):
    """Climb a `corpus` into a TOWER of levels: consolidate -> find patterns -> promote to a new alphabet ->
    repeat, STOPPING when the MDL gain of the next level drops below `min_gain` (as a fraction of current bits).
    Returns the tower: a list of level dicts, each with depth, alphabet size, corpus, bits, baseline (bits at the
    previous level), the promoted atoms (id -> child pair) with STABLE hashlib ids, and -- on the terminal level
    -- the refusal reason and the measured gain that failed the gate.

    `lens` selects the adjacency definition: 'sequence' = position (letters->words->phrases, transform streams),
    'structure' = shared group / spatial bucket (verts->part->assembly -- order-independent set promotion).
    `prior_tower` (Quilez Q7, temporal reprojection) is reserved for the streaming climb -- default None rebuilds.

    WHY min_gain as a FRACTION and the pre-gate: pair promotion memorizes noise, so the floor is the
    anti-memorization gate. Before each expensive promotion pass, a CONSERVATIVE zlib-redundancy bound (Q5) prunes
    a level that provably cannot pay, so we never 'march empty space'. A ladder that tops out at depth k is a
    result, logged loudly, not an apology."""
    lenses = {"sequence": _sequence_lens_promote, "structure": _structure_lens_promote}
    if lens not in lenses:
        raise ValueError("unknown lens %r; ship 'sequence' or 'structure'." % lens)
    promote = lenses[lens]
    if prior_tower is not None:
        # Q7 reserved: the streaming path will seed from the prior tower's alphabets. Not built yet; the batch
        # path (prior_tower=None) is what we prove. Flagged, not silently ignored.
        raise NotImplementedError("prior_tower (streaming reprojection, Quilez Q7) is reserved for the streaming "
                                  "climb; the batch climb requires prior_tower=None.")

    # normalise: a corpus is a list of sequences (lists) of hashable symbol ids.
    corpus = [list(seq) for seq in corpus]
    n_syms = len(set(s for seq in corpus for s in seq))
    bits = corpus_bits(corpus, n_syms)
    tower = [{"depth": 0, "alphabet_size": n_syms, "corpus": corpus, "bits": bits,
              "baseline": None, "lens": lens, "atoms": {}, "negatives": [], "terminal": False,
              "kit": _level_kit(lens, 0, n_syms)}]

    for depth in range(1, max_depth + 1):
        prev = tower[-1]

        # Q5 PRE-GATE: a cheap conservative bound on achievable gain (zlib redundancy). If even the optimistic
        # bound cannot clear min_gain, refuse WITHOUT running the promotion pass -- 'don't march empty space'.
        bound = _achievable_gain_bound(prev["corpus"], prev["bits"])
        if bound < min_gain * prev["bits"]:
            prev["terminal"] = True
            prev["negatives"].append("pre-gate: zlib redundancy bounds achievable gain at %.1f bits < %.1f (%.0f%% "
                                      "of %.1f) -- no promotion pass run (Quilez Q5)"
                                      % (bound, min_gain * prev["bits"], min_gain * 100, prev["bits"]))
            prev["refusal"] = "pre-gate: cannot beat the floor"
            break

        # the expensive pass: promote patterns and re-express (lens-dispatched).
        new_corpus, new_atoms = promote(prev["corpus"], max_merges, min_count)
        new_n = len(set(s for seq in new_corpus for s in seq))
        new_bits = corpus_bits(new_corpus, new_n)
        gain = prev["bits"] - new_bits

        if gain < min_gain * prev["bits"]:
            # the level did not pay -- STOP here, this is the ceiling. Record the measured gain that failed.
            prev["terminal"] = True
            prev["negatives"].append("level-up to depth %d gained %.1f bits (%.1f%%) < floor %.1f%% -- ladder "
                                      "topped out (a result, not a failure)"
                                      % (depth, gain, 100 * gain / max(prev["bits"], 1e-9), min_gain * 100))
            prev["refusal"] = "gain %.1f%% below %.1f%% floor" % (100 * gain / max(prev["bits"], 1e-9),
                                                                  min_gain * 100)
            break

        # the level paid: promote. Re-key the promoted atoms with STABLE hashlib ids (sweep-mandated).
        stable_atoms = {}
        for nid, (a, b) in new_atoms.items():
            sid = atom_id((a, b), depth)
            stable_atoms[sid] = {"children": (a, b), "raw_id": nid, "depth": depth}
        tower.append({"depth": depth, "alphabet_size": new_n, "corpus": new_corpus, "bits": new_bits,
                      "baseline": prev["bits"], "lens": lens, "atoms": stable_atoms,
                      "negatives": [], "terminal": False, "kit": _level_kit(lens, depth, new_n)})
    else:
        # ran out of depth without topping out -- mark the last level terminal by exhaustion, honestly.
        tower[-1]["terminal"] = True
        tower[-1]["refusal"] = "reached max_depth=%d (not a natural ceiling -- raise max_depth to continue)" % max_depth

    return tower


def tower_summary(tower):
    """A compact human-readable summary of a climbed tower: per level, its depth, alphabet size, bits, and the
    gain over the previous level; plus the terminal refusal reason. For logging a climb result honestly."""
    lines = []
    for lv in tower:
        gain = "" if lv["baseline"] is None else " (gain %.1f%%)" % (100 * (lv["baseline"] - lv["bits"]) / max(lv["baseline"], 1e-9))
        term = "  TERMINAL: " + lv.get("refusal", "") if lv.get("terminal") else ""
        lines.append("depth %d: %d atoms, %.1f bits%s%s" % (lv["depth"], lv["alphabet_size"], lv["bits"], gain, term))
    return "\n".join(lines)


def _level_kit(lens, depth, alphabet_size):
    """The INVARIANT KIT for a level (§3a): each slot is either WIRED (names the existing primitive it delegates
    to) or a DECLARED NEGATIVE (a string reason it does not apply here, prefixed 'no '). An EMPTY slot with no
    reason is an audit error -- that is how kit_gaps stops us silently re-deriving these per substrate. A level
    with 'no lod' is a decision to record, not a hole to trip over later. The enforced form of 'we also do X at
    every level'."""
    kit = {
        "delta":      "canonical atom + child pair (stable hashlib id over children)",
        "chunk":      "holographic_chunkcodebook.learn_chunks (pair promotion)",
        "cleanup":    "holographic cleanup / snap against this level's alphabet",
        "canonical":  "the promoted atom is the canonical element; children are the delta",
        "seed":       "atom regenerable from (children, depth) via atom_id -- deterministic",
        "cache":      "corpus_bits memoizable; dependency-keyed cache applies",
        "scale":      "auto_scale when this level's bundle capacity saturates (sqrt(N/D) per level)",
        "superpose":  "superposed gather: all alphabet atoms scored in one dot product at cleanup",
        "decompose":  ("reconstruct(): child-pair expansion -- LOSSLESS round-trip to the original corpus"
                       if lens == "sequence" else
                       "reconstruct(): expands to the SET of base part-types (structure lens is order- and "
                       "count-agnostic by design -- not a lossless round-trip, a declared semantic)"),
        "bake":       "bake this level's alphabet signatures once, sample O(1) at cleanup",
    }
    # slots whose applicability DEPENDS on the level -- declared explicitly ('no ...'), never left empty:
    kit["lod"] = ("progressive: coarse alphabet first" if lens == "sequence"
                  else "no lod: structure-lens alphabet is an unordered set (no natural coarsening order)")
    kit["tile"] = ("domain_repeat / partitioned promotion" if alphabet_size > 8
                   else "no tile: alphabet too small to partition (%d atoms)" % alphabet_size)
    kit["fuse"] = "no fuse yet: fuse-promotion (pipeline->one op) is L7; declared negative until then"
    kit["guide"] = "no guide yet: constrained movement (guide_structure) is L10; declared negative until then"
    return kit


def kit_report(level):
    """Report a level's invariant-kit slots as WIRED (a delegate named), DECLARED-NEGATIVE (a reason, prefixed
    'no '), or SILENT-GAP (empty with no reason -- an audit error). Returns
    {'wired': [...], 'declared_negative': [...], 'silent_gaps': [...]}. Target: 0 silent gaps. The L6 enforcement
    that turns 'we also do X at every level' into a budget, like the reachability audit did for wiring."""
    kit = level.get("kit", {})
    EXPECTED = {"delta", "chunk", "scale", "decompose", "cleanup", "bake", "seed", "tile", "lod", "canonical",
                "cache", "fuse", "superpose", "guide"}
    wired, declared_negative, silent_gaps = [], [], []
    for slot in sorted(EXPECTED):
        val = kit.get(slot)
        if val is None or (isinstance(val, str) and not val.strip()):
            silent_gaps.append(slot)
        elif isinstance(val, str) and val.startswith("no "):
            declared_negative.append(slot)
        else:
            wired.append(slot)
    return {"wired": wired, "declared_negative": declared_negative, "silent_gaps": silent_gaps}


def contamination_check(tower):
    """LEVEL ISOLATION test (sweep-mandated): verify that no atom id is shared between two different levels -- a
    level-k cleanup must never be able to snap to a level-j atom. Returns (ok, collisions). This is the
    shared-kernel-is-not-a-shared-manifold lesson enforced as a check: distinct levels must have distinct atom
    id-spaces, which the hashlib id (keyed on depth) guarantees by construction -- this verifies it held."""
    seen = {}
    collisions = []
    for lv in tower:
        for aid in lv.get("atoms", {}):
            if aid in seen and seen[aid] != lv["depth"]:
                collisions.append((aid, seen[aid], lv["depth"]))
            seen[aid] = lv["depth"]
    return (len(collisions) == 0, collisions)


def identify_level(corpus, max_merges=100, min_count=2):
    """'What am I looking at?' -- classify a corpus by WHICH LADDER OPERATIONS PAY on it (L8, §3e), returning a
    signature of MEASUREMENTS, not a label, so a wrong guess stays auditable. The four diagnostics:

      * `compressible` (bits saved fraction by one promotion pass) -- is there a level ABOVE this one? High =
        undiscovered structure; ~0 = you are at the top.
      * `hit_rate` (fraction of promoted pairs that recur) -- density of repeated structure.
      * `order_dependent` (does the SEQUENCE lens beat the STRUCTURE lens?) -- which adjacency the data wants:
        True => streams/sequences, False => sets/assemblies. This picks the lens WITHOUT guessing.
      * `regime` -- Wolfram's taxonomy as a measurement: 'repetitive' (one atom dominates), 'nested/structured'
        (compresses well, a level above), or 'irreducible' (no compression above the null -- refuse the climb).

    Quilez Q6 (carry the derivative): `compress_sensitivity` returns d(gain)/d(merges) between a half-budget and
    full-budget pass -- a FREE by-product of running two passes, telling `climb` HOW HARD the structure resists,
    not just whether it exists. Returns a dict of these numbers + the inferred `lens` and `regime`. Reuses the
    ladder's own promotion machinery -- no new math."""
    corpus = [list(seq) for seq in corpus]
    if not any(corpus):
        return {"compressible": 0.0, "hit_rate": 0.0, "order_dependent": False, "regime": "empty",
                "lens": "sequence", "compress_sensitivity": 0.0, "n_symbols": 0}
    n0 = len(set(s for seq in corpus for s in seq))
    b0 = corpus_bits(corpus, n0)

    # sequence-lens pass at HALF and FULL budget (Q6: the two passes give the derivative for free).
    half_c, half_a = _sequence_lens_promote(corpus, max(1, max_merges // 2), min_count)
    full_c, full_a = _sequence_lens_promote(corpus, max_merges, min_count)
    b_half = corpus_bits(half_c, max(1, len(set(s for seq in half_c for s in seq))))
    b_full = corpus_bits(full_c, max(1, len(set(s for seq in full_c for s in seq))))
    gain_half = (b0 - b_half) / max(b0, 1e-9)
    gain_full = (b0 - b_full) / max(b0, 1e-9)
    # sensitivity: how much extra gain the second half of the budget bought (Q6 derivative).
    sensitivity = gain_full - gain_half

    # structure-lens pass (order-independent) to decide which adjacency the data wants.
    struct_c, struct_a = _structure_lens_promote(corpus, max_merges, min_count)
    b_struct = corpus_bits(struct_c, max(1, len(set(s for seq in struct_c for s in seq))))
    gain_struct = (b0 - b_struct) / max(b0, 1e-9)

    order_dependent = gain_full > gain_struct                  # sequence beats structure => order carries info
    lens = "sequence" if order_dependent else "structure"
    best_gain = max(gain_full, gain_struct)
    hit_rate = len(full_a) / max(1, max_merges)

    # NULL BASELINE (kept negative: high-D noise has basins too). Short sequences over a small alphabet compress
    # BY CHANCE -- and the STRUCTURE lens especially, since ~5 symbols drawn from 16 make many pairs co-occur by
    # accident. So measure the SAME lens's compression on a SHUFFLED corpus (same symbols, structure destroyed)
    # and credit only gain ABOVE that null. Crucially the null uses the WINNING lens, so its own spurious
    # compression cancels: real structure survives the shuffle, chance co-occurrence does not.
    rng = np.random.default_rng(0)
    flat = [s for seq in corpus for s in seq]
    shuffled = list(flat)
    rng.shuffle(shuffled)
    null_corpus, k = [], 0
    for L in (len(seq) for seq in corpus):
        null_corpus.append(shuffled[k:k + L]); k += L
    win_promote = _sequence_lens_promote if order_dependent else _structure_lens_promote
    null_c, _ = win_promote(null_corpus, max_merges, min_count)
    b_null = corpus_bits(null_c, max(1, len(set(s for seq in null_c for s in seq))))
    null_gain = (b0 - b_null) / max(b0, 1e-9)
    gain_over_null = best_gain - null_gain                     # structure that survives the shuffle (chance cancels)

    # regime (Wolfram taxonomy as a measurement): dominant-symbol => repetitive; compression ABOVE THE NULL =>
    # nested; at/below null => irreducible (refuse -- what looks like structure is chance).
    counts = {}
    tot = 0
    for seq in corpus:
        for s in seq:
            counts[s] = counts.get(s, 0) + 1; tot += 1
    dominant = max(counts.values()) / max(tot, 1)
    if dominant > 0.6:
        regime = "repetitive"
    elif gain_over_null > 0.05:
        regime = "nested/structured"
    else:
        regime = "irreducible"

    return {"compressible": round(best_gain, 4), "gain_over_null": round(gain_over_null, 4),
            "hit_rate": round(hit_rate, 4), "order_dependent": bool(order_dependent), "lens": lens,
            "regime": regime, "compress_sensitivity": round(sensitivity, 4), "n_symbols": n0}


def adaptive_pipeline(corpus, null_margin=0.05, max_depth=8, min_gain=0.02):
    """MEASUREMENT-DRIVEN adaptive dispatcher (the panel convergence point): run identify_level, then dispatch to
    the method its REGIME names -- rather than hard-coding one method for all data. The four seats' one ask:

      * ABSTAIN (SETI -- Tarter/Siemion): if the data does not beat its shuffle null by `null_margin`, REFUSE.
        A 'cleaned' version of noise is a fabricated signal -- the cardinal sin. Return method='abstain' + reason.
      * REPETITIVE (Quilez taxonomy): one atom dominates -> the cheap answer is run-length / domain-fold, NOT a
        climb. Return method='fold' with the dominant symbol; a climb here would pay for nothing.
      * NESTED/STRUCTURED: real hierarchy above the null -> CLIMB, with the lens identify_level PICKED per-signal
        (Puckette -- the lens is the analysis window, chosen not guessed). Return method='climb' + the tower.
      * (irreducible-but-above-null is treated as 'store raw' -- structure too weak to climb but not pure noise.)

    Returns a dict: `method` ('abstain' | 'fold' | 'climb' | 'store_raw'), `regime`, `lens`, `gain_over_null`, a
    plain-language `reason`, and the method's payload (`tower` for climb, `dominant` for fold). A readable,
    refusable dispatch on numbers we already compute -- no new learner, no black box. Quilez's cost discipline:
    the cheap sufficient method is chosen, never an expensive one that gains nothing."""
    sig = identify_level(corpus)
    regime = sig["regime"]
    lens = sig["lens"]
    gon = sig["gain_over_null"]

    # ABSTAIN gate (SETI): null-indistinguishable input is refused -- do not fabricate structure.
    if gon < null_margin and regime != "repetitive":
        return {"method": "abstain", "regime": regime, "lens": lens, "gain_over_null": gon,
                "reason": "data does not beat its shuffle null (gain_over_null %.3f < %.2f) -- refusing to "
                          "'clean' noise into a fabricated signal (SETI gate)" % (gon, null_margin),
                "tower": None, "dominant": None}

    if regime == "repetitive":
        # cheap answer: run-length / domain-fold on the dominant symbol (Quilez -- never pay for a climb here).
        counts = {}
        for seq in corpus:
            for s in seq:
                counts[s] = counts.get(s, 0) + 1
        dominant = max(counts, key=counts.get) if counts else None
        return {"method": "fold", "regime": regime, "lens": lens, "gain_over_null": gon,
                "reason": "one symbol dominates -> run-length/domain-fold is the cheap sufficient method; a climb "
                          "would pay for nothing (Quilez cost discipline)",
                "tower": None, "dominant": dominant}

    if regime == "nested/structured":
        # real hierarchy above the null -> climb, with the lens PICKED per-signal (Puckette).
        tower = climb(corpus, lens=lens, max_depth=max_depth, min_gain=min_gain)
        return {"method": "climb", "regime": regime, "lens": lens, "gain_over_null": gon,
                "reason": "nested structure above the null -> climb with the '%s' lens (picked by identify_level, "
                          "not guessed -- the lens is the analysis window, Puckette)" % lens,
                "tower": tower, "dominant": None}

    # irreducible but above the null margin: weak structure, not worth a climb -- store raw honestly.
    return {"method": "store_raw", "regime": regime, "lens": lens, "gain_over_null": gon,
            "reason": "structure too weak to climb but above the null -> store raw; do not fabricate a hierarchy",
            "tower": None, "dominant": None}


# ---------------------------------------------------------------------------------------------------------
# Cartography (L13): 'raytracing' the holographic space, made measurable. A RAY is a deterministic 1-parameter
# path (an interpolation between two atoms); a HIT is the path entering a cleanup BASIN (its nearest alphabet
# atom exceeds a cosine margin). The atlas reports basin coverage, margins, and dead zones -- but only counts as
# STRUCTURE above a band-limited random-alphabet NULL (Quilez Q8), because high-D noise has basins too.
# ---------------------------------------------------------------------------------------------------------

def _march_ray(a, b, alphabet, steps, margin):
    """March a straight ray from atom `a` to atom `b` in `steps`, testing at each step whether the point sits in a
    cleanup BASIN. The honest basin test is a SEPARATION margin: the nearest atom's cosine must exceed the
    RUNNER-UP's by `margin`. WHY not absolute cosine: with a small alphabet in high-D, argmax-cosine ALWAYS has a
    winner above any fixed absolute threshold (the nearest of N random atoms is trivially 'close'), so an absolute
    margin makes every point a hit -- including the null. A real basin means the point is DISTINCTIVELY nearer one
    atom than the next, which is exactly the gap that makes cleanup unambiguous. Returns (hit_fraction, n_basins):
    the fraction of steps distinctively inside one basin, and how many distinct basins the ray passed through.
    Pure numpy: one matmul (alphabet @ point). No training."""
    hits = 0
    basins = set()
    for k in range(steps + 1):
        u = k / steps
        p = (1.0 - u) * a + u * b
        n = np.linalg.norm(p)
        if n < 1e-12:
            continue
        p = p / n
        cos = alphabet @ p                     # cosine to every atom (rows are unit-norm)
        order = np.argsort(cos)                # ascending; last is the winner, second-last the runner-up
        j = int(order[-1])
        runner = cos[order[-2]] if len(order) > 1 else -1.0
        if cos[j] - runner >= margin:          # DISTINCTIVELY nearest -> a genuine basin
            hits += 1
            basins.add(j)
    return hits / (steps + 1), len(basins)


def _band_limit(alphabet, keep_frac, rng):
    """Band-limit an alphabet to a fraction `keep_frac` of its spectral energy (Quilez Q8, analytic prefiltering):
    project each atom onto the top principal directions of the alphabet and drop the rest, then renormalise. WHY:
    the null model must be measured on band-limited random atoms too, or the baseline does not match the
    measurement -- two atoms that differ only in high-frequency detail should tie at the coarse grain, honestly,
    rather than one winning by aliasing. Deterministic given the alphabet (SVD) -- rng unused here but kept for
    signature parity with the null builder. Returns a band-limited, row-normalised copy."""
    # SVD of the alphabet; keep the top-k right singular directions (the 'low-frequency' shared structure).
    U, S, Vt = np.linalg.svd(alphabet, full_matrices=False)
    k = max(1, int(round(keep_frac * Vt.shape[0])))
    basis = Vt[:k]                             # (k, dim) top directions
    proj = (alphabet @ basis.T) @ basis        # reconstruct using only the kept directions
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return proj / norms


def chart_space(alphabet, rays=64, steps=24, margin=0.1, band_keep=0.5, seed=0):
    """Chart a holographic ALPHABET as a measured atlas (L13, §3j): march random rays between atoms and record
    where they enter cleanup BASINS. A point is in a basin when its nearest atom is DISTINCTIVELY nearer than the
    runner-up (top cosine minus second cosine >= `margin`) -- a separation gap, not an absolute threshold, because
    with a small alphabet in high-D the nearest atom is always trivially 'close' (an absolute margin makes
    everything a hit, including the null). Returns a dict with `hit_fraction` (mean fraction of a ray inside some
    basin), `basins_per_ray` (mean distinct basins a ray crosses), `coverage` (fraction of atoms that are SOME
    ray's nearest at least once -- dead atoms never win), and the honest verdict `structure_over_null`: the
    hit_fraction MINUS that of a BAND-LIMITED random alphabet of the same size (Quilez Q8). A region counts as
    structure only ABOVE the matched null -- high-D noise has basins too, so an atlas without its null is a
    Rorschach test.

    `alphabet` is an (N, dim) array of atom vectors (normalised here). Delegates the hit test to argmax-cosine
    (the op cleanup uses); no new math. Deterministic given `seed`. WHY band-limit the NULL and not the real
    alphabet: the baseline must match the measurement's grain, or a real alphabet's fine detail flatters it."""
    A = np.asarray(alphabet, float)
    if A.ndim != 2 or A.shape[0] < 2:
        raise ValueError("chart_space needs an (N>=2, dim) alphabet; got shape %r" % (A.shape,))
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    A = A / norms
    rng = np.random.default_rng(seed)
    n = A.shape[0]

    def _atlas(alpha):
        hf_sum, bpr_sum = 0.0, 0.0
        won = set()
        for _ in range(rays):
            i, j = rng.integers(n), rng.integers(n)
            if i == j:
                j = (j + 1) % n
            hf, nb = _march_ray(alpha[i], alpha[j], alpha, steps, margin)
            hf_sum += hf
            bpr_sum += nb
            # record which atoms are ever the nearest along this ray (for coverage / dead-zone detection).
            for k in range(steps + 1):
                u = k / steps
                p = (1.0 - u) * alpha[i] + u * alpha[j]
                nn = np.linalg.norm(p)
                if nn < 1e-12:
                    continue
                won.add(int(np.argmax(alpha @ (p / nn))))
        return hf_sum / rays, bpr_sum / rays, len(won) / n

    hit_fraction, basins_per_ray, coverage = _atlas(A)
    # band-limited random-alphabet NULL (Q8): same N, same dim, but no real structure -- band-limited to match grain.
    null_raw = rng.standard_normal(A.shape)
    null = _band_limit(null_raw, band_keep, rng)
    null_hf, _, _ = _atlas(null)

    return {"hit_fraction": round(hit_fraction, 4), "basins_per_ray": round(basins_per_ray, 3),
            "coverage": round(coverage, 4), "null_hit_fraction": round(null_hf, 4),
            "structure_over_null": round(hit_fraction - null_hf, 4), "n_atoms": n}


# ---------------------------------------------------------------------------------------------------------
# The up/down/sideways sweep (L5) -- a capability that works in only one direction is an incomplete faculty.
# ---------------------------------------------------------------------------------------------------------

def sweep_directions(corpus, max_depth=6, min_gain=0.02):
    """The UP / DOWN / SIDEWAYS completeness sweep (§6): does the ladder's structure-finding hold in all three
    directions, or only one? A capability that works on a whole but not its parts (or vice versa) is an INCOMPLETE
    faculty -- this turns that design check into a runnable measurement, the way kit_gaps did for the invariant
    kit. Returns a dict per direction with an `ok` flag and the measurement behind it, plus `gaps` (directions
    that failed) so a missed direction is caught, not assumed.

      * DOWN -- does structure survive DECOMPOSITION? Climb the corpus, expand the top level's atoms back to their
        child components (reconstruct_corpus), and confirm re-climbing the components RECOVERS structure (the parts
        are themselves analyzable, not opaque). A faculty that only sees wholes fails here.
      * UP -- does structure survive EMBEDDING? Splice the corpus into a LARGER corpus (as a sub-stream among other
        material) and confirm the climb still finds compression -- the structure is not destroyed by being a
        component of something bigger. A faculty that only sees isolated inputs fails here.
      * SIDEWAYS -- which COSTUMES (lenses) does the data wear? Climb under each lens and report which ones find
        structure. Most data favours one lens; a datum that compresses under BOTH wears two costumes. A faculty
        that hard-codes one lens misses the other costume.

    Delegates entirely to climb / identify_level / reconstruct_corpus -- no new math, just the three probes.
    Deterministic."""
    corpus = [list(seq) for seq in corpus]
    report = {"down": {}, "up": {}, "sideways": {}, "gaps": []}

    # pick the lens the data actually wants (so DOWN/UP test the RIGHT operation, not an arbitrary one).
    sig = identify_level(corpus)
    lens = sig["lens"]

    # GUARD (consistency with the rest of the system): if the corpus itself has no structure above its shuffle
    # null, the sweep is vacuous -- noise 'compresses' a little by chance (especially under the structure lens),
    # so raw compression is not enough. Report all directions as not-applicable rather than fabricating structure.
    if sig["gain_over_null"] < 0.05 and sig["regime"] != "repetitive":
        report["down"] = {"ok": False, "reason": "corpus is irreducible (no structure above the null)"}
        report["up"] = {"ok": False, "reason": "corpus is irreducible (no structure above the null)"}
        report["sideways"] = {"lenses_that_fit": [], "all": {"sequence": False, "structure": False}, "ok": False,
                              "reason": "corpus is irreducible (no structure above the null)"}
        report["gaps"] = ["down", "up", "sideways"]
        report["preferred_lens"] = lens
        report["complete"] = False
        report["irreducible"] = True
        return report

    # --- DOWN: decompose the top level and confirm the components are still analyzable. ---
    tower = climb(corpus, lens=lens, max_depth=max_depth, min_gain=min_gain)
    if len(tower) >= 2:
        components = reconstruct_corpus(tower)                 # the atoms expanded back to base symbols
        comp_sig = identify_level(components)
        down_ok = comp_sig["gain_over_null"] > 0.05           # the parts still carry structure ABOVE THE NULL
        report["down"] = {"ok": bool(down_ok), "levels": len(tower),
                          "component_gain_over_null": comp_sig["gain_over_null"]}
    else:
        # the corpus did not climb at all -- DOWN is vacuously not applicable (no whole to decompose).
        report["down"] = {"ok": False, "levels": len(tower), "reason": "corpus did not climb -- nothing to decompose"}
    if not report["down"]["ok"]:
        report["gaps"].append("down")

    # --- UP: embed the corpus in a larger one and confirm structure survives EMBEDDING, above the null. ---
    rng = np.random.default_rng(0)
    n_syms = max((s for seq in corpus for s in seq), default=0) + 1
    # distractor material: random symbols OUTSIDE the corpus's alphabet, so the embedding is honest (not more of
    # the same structure). Splice the real corpus among the distractors.
    distract = [[int(x) for x in rng.integers(n_syms, n_syms + 8, size=len(seq))] for seq in corpus]
    larger = []
    for real, noise_seq in zip(corpus, distract):
        larger.append(noise_seq)
        larger.append(real)                                   # real structure interleaved with distractors
    up_sig = identify_level(larger)
    up_ok = up_sig["gain_over_null"] > 0.05                   # still beats the null when embedded, not just compresses
    report["up"] = {"ok": bool(up_ok), "gain_over_null": up_sig["gain_over_null"]}
    if not up_ok:
        report["gaps"].append("up")

    # --- SIDEWAYS: which lenses find structure ABOVE THE NULL? (raw compression alone over-credits noise.) ---
    costumes = {}
    for L in ("sequence", "structure"):
        t = climb(corpus, lens=L, max_depth=max_depth, min_gain=min_gain)
        # a lens 'fits' only if it compresses AND the corpus beats the null under that lens's adjacency.
        raw_gain = len(t) >= 2 and t[1]["bits"] < t[0]["bits"]
        costumes[L] = bool(raw_gain and sig["gain_over_null"] > 0.05)
    report["sideways"] = {"lenses_that_fit": [L for L, ok in costumes.items() if ok], "all": costumes,
                          "ok": any(costumes.values())}
    if not report["sideways"]["ok"]:
        report["gaps"].append("sideways")

    report["preferred_lens"] = lens
    report["irreducible"] = False
    report["complete"] = (len(report["gaps"]) == 0)
    return report


# ---------------------------------------------------------------------------------------------------------
# The bank-vs-formula gate (L9, Quilez Q1 'store the formula, not the samples'). The demoscene economy: a
# generator (seed + params) beats a baked table UNLESS evaluation is expensive enough AND reused enough to
# amortize the storage. This is that decision as a MEASURED function -- the reusable wheel that the transform
# bank, the generator bank (fit_deterministic), and any cache should consult, rather than banking by reflex.
# ---------------------------------------------------------------------------------------------------------

def bank_or_formula(eval_cost_us, hit_rate, n_entries, bytes_per_entry, lookup_cost_us=0.5,
                    regen_from_seed=True):
    """Decide whether to BANK a set of computed values or keep the FORMULA and regenerate on demand (Quilez Q1).
    The demoscene rule made measurable: bank only when the amortized saving beats the storage+lookup cost. A bank
    of things a cheap formula gives you for free is NEGATIVE storage.

    Inputs (all measured, none guessed): `eval_cost_us` (microseconds to compute ONE entry from its formula),
    `hit_rate` (fraction of queries that reuse an existing entry, in [0,1]), `n_entries` (bank size),
    `bytes_per_entry` (storage per baked entry), `lookup_cost_us` (cost of a bank lookup), `regen_from_seed` (if
    True the formula is (seed, params) -- cold entries evict to nothing, the kit's seed slot).

    Returns a verdict dict: `bank` (bool), `saving_us_per_query` (amortized time saved by banking, negative means
    banking LOSES), `storage_bytes`, and a plain-language `reason`. The amortized model, honestly: WITHOUT a bank
    every query pays eval_cost. WITH a bank, a MISS (fraction 1-hit_rate) must BUILD the entry (eval + lookup)
    while a HIT pays only lookup -- so the mean cost with a bank is (1-hit_rate)*eval + lookup, and the saving per
    query is `hit_rate*eval_cost - lookup_cost`. Banking pays iff that is positive; the break-even hit rate is
    lookup_cost/eval_cost. WHY hit_rate multiplies eval and not (eval-lookup): a miss pays BOTH the eval to build
    AND the lookup, so the lookup is a flat per-query tax that does not scale with reuse -- only the saved evals
    on hits amortize the bake. (An earlier version used hit_rate*(eval-lookup), which overcounted the saving by
    crediting the lookup against hits; the fix is this corrected accounting.) Deterministic; pure arithmetic."""
    per_query_saving = hit_rate * eval_cost_us - lookup_cost_us
    storage_bytes = int(n_entries * bytes_per_entry)
    bank = per_query_saving > 0.0
    breakeven = lookup_cost_us / eval_cost_us if eval_cost_us > 0 else float("inf")
    if not bank:
        if eval_cost_us <= lookup_cost_us:
            reason = ("formula wins: eval (%.1f us) is already <= a lookup (%.1f us) -- banking is negative "
                      "storage, exactly what Q1 warns against" % (eval_cost_us, lookup_cost_us))
        else:
            reason = ("formula wins: hit_rate %.3f below break-even %.3f (lookup/eval) -- entries are not reused "
                      "enough to pay back the bake (saving %.2f us/query <= 0)"
                      % (hit_rate, breakeven, per_query_saving))
    else:
        econ = "cold entries evict to nothing (regenerable from seed)" if regen_from_seed else \
               "entries are NOT seed-regenerable -- storage is a real, permanent cost"
        reason = ("bank wins: hit_rate %.3f x eval %.1f - lookup %.1f = %.2f us saved per query (break-even "
                  "hit_rate %.3f); %s" % (hit_rate, eval_cost_us, lookup_cost_us, per_query_saving, breakeven, econ))
    return {"bank": bool(bank), "saving_us_per_query": round(per_query_saving, 3),
            "storage_bytes": storage_bytes, "reason": reason}
# ---------------------------------------------------------------------------------------------------------

def _persistence_predict(history):
    """The honest BASELINE forecaster: predict the next symbol is the same as the last one. A ladder predictor
    must BEAT this above the null, or it abstains -- the SETI discipline on the time axis (a forecast that cannot
    beat 'same as last' is a null result). Returns the last symbol, or None on empty history."""
    return history[-1] if len(history) else None


def _chunk_continuations(stream, cb, order):
    """Build a next-CHUNK frequency table over the codebook-encoded stream: for each context of `order` encoded
    chunks, count what chunk follows. This is a flat n-gram over the PROMOTED alphabet -- which is the
    hierarchical predictor, because each chunk decodes to many raw symbols. Returns dict[context_tuple] ->
    dict[next_chunk] -> count."""
    enc = cb.encode(list(stream))
    table = {}
    for i in range(len(enc) - order):
        ctx = tuple(enc[i:i + order])
        nxt = enc[i + order]
        table.setdefault(ctx, {})
        table[ctx][nxt] = table[ctx].get(nxt, 0) + 1
    return table, enc


def ladder_predict(history, order=2, max_merges=200, min_count=2, null_margin=0.02, seed=0):
    """Predict what comes next after `history` (a 1-D list of symbol ids) using the ladder's learned HIERARCHICAL
    alphabet -- the compression<->prediction duality (a good compressor is a good predictor). Learns a chunk
    codebook over the history, then predicts the next CHUNK by matching the context against continuation counts
    over the promoted alphabet, and decodes that chunk back to raw symbols -- so one prediction can emit a whole
    learned pattern, not one flat symbol. Falls to a shorter context when the deep match is unseen (the
    recursive_factor pattern, on prediction).

    THE GATE (SETI, on the time axis): the predictor must BEAT the persistence baseline ('next = last') on
    held-out continuations by at least `null_margin`, or it ABSTAINS. A forecast that cannot beat 'same as last'
    is a null result, said loudly -- never a confidently-hallucinated trend.

    Returns a dict: `prediction` (list of raw symbols, the decoded next chunk), `confidence` (the winning
    continuation's share of its context's counts), `method` ('ladder' | 'persistence' | 'abstain'),
    `beats_persistence` (accuracy improvement over persistence on held-out, in [-1,1]), and `reason`. Delegates
    the chunking to learn_chunks -- no new math, just the model turned around."""
    history = list(history)
    if len(history) < 4:
        return {"prediction": [_persistence_predict(history)] if history else [], "confidence": 0.0,
                "method": "persistence", "beats_persistence": 0.0,
                "reason": "history too short to learn a model -- falling back to persistence"}

    # split history: fit on the front, MEASURE on the held-out tail (search overfits its own tests).
    cut = max(4, int(len(history) * 0.75))
    fit, test = history[:cut], history[cut:]
    # PREDICTION wants MODERATE chunking, not maximal: aggressive BPE collapses a repeating stream into a handful
    # of giant chunks, leaving too few tokens to model transitions (compression's goal is the OPPOSITE of
    # prediction's here). Cap merges so the encoded stream keeps enough tokens to build a continuation table.
    pred_merges = min(max_merges, max(4, cut // 8))
    cb = learn_chunks(fit, max_merges=pred_merges, min_count=min_count)
    enc_probe = cb.encode(fit)
    # If chunking collapsed the stream too far (a perfectly periodic signal merges ENTIRELY, regardless of the
    # cap), the continuation table would be empty -- so predict over RAW symbols instead (the ladder's depth-0
    # model). A raw n-gram already captures a short period; chunks only help when they leave enough tokens.
    if len(enc_probe) < 3 * order + 4:
        cb = learn_chunks([], max_merges=0, min_count=min_count)   # identity codebook: encode == raw symbols
    table, enc_fit = _chunk_continuations(fit, cb, order)

    def predict_chunk(ctx_encoded):
        """Predict the next chunk for an encoded context, backing off to shorter contexts when unseen."""
        for o in range(order, 0, -1):
            ctx = tuple(ctx_encoded[-o:])
            if ctx in table and table[ctx]:
                nxt = max(table[ctx].items(), key=lambda kv: kv[1])
                total = sum(table[ctx].values())
                return nxt[0], nxt[1] / total
        return None, 0.0

    # MEASURE on held-out: does the ladder predictor beat persistence at predicting the next raw symbol?
    ladder_hits = pers_hits = trials = 0
    for i in range(len(test) - 1):
        ctx_raw = (fit + test[:i + 1])
        enc_ctx = cb.encode(ctx_raw)
        chunk, _conf = predict_chunk(enc_ctx)
        if chunk is not None:
            decoded = cb.decode([chunk])
            pred_sym = decoded[0] if decoded else None
        else:
            pred_sym = None
        actual = test[i + 1]
        if pred_sym == actual:
            ladder_hits += 1
        if _persistence_predict(ctx_raw) == actual:
            pers_hits += 1
        trials += 1

    if trials == 0:
        beats = 0.0
    else:
        beats = (ladder_hits - pers_hits) / trials

    # THE GATE: abstain if the ladder does not beat persistence above the margin.
    if beats < null_margin:
        # persistence itself is the honest forecast if IT is informative; otherwise abstain.
        return {"prediction": [_persistence_predict(history)], "confidence": 0.0, "method": "persistence",
                "beats_persistence": round(beats, 4),
                "reason": "ladder model does not beat persistence on held-out (%.1f%% <= %.1f%%) -- returning the "
                          "persistence baseline, not a fabricated trend (SETI gate on the time axis)"
                          % (100 * beats, 100 * null_margin)}

    # the ladder wins: predict the next chunk from the FULL history, decode to raw symbols.
    cb_full = learn_chunks(history, max_merges=min(max_merges, max(4, len(history) // 8)), min_count=min_count)
    if len(cb_full.encode(history)) < 3 * order + 4:
        cb_full = learn_chunks([], max_merges=0, min_count=min_count)   # identity: predict over raw symbols
    table_full, _ = _chunk_continuations(history, cb_full, order)
    enc_hist = cb_full.encode(history)

    def predict_full(ctx_encoded):
        for o in range(order, 0, -1):
            ctx = tuple(ctx_encoded[-o:])
            if ctx in table_full and table_full[ctx]:
                nxt = max(table_full[ctx].items(), key=lambda kv: kv[1])
                total = sum(table_full[ctx].values())
                return nxt[0], nxt[1] / total
        return None, 0.0

    chunk, conf = predict_full(enc_hist)
    prediction = cb_full.decode([chunk]) if chunk is not None else [_persistence_predict(history)]
    return {"prediction": list(prediction), "confidence": round(conf, 4), "method": "ladder",
            "beats_persistence": round(beats, 4),
            "reason": "ladder hierarchical predictor beats persistence on held-out by %.1f%% -- predicting the next "
                      "learned chunk (%d raw symbol(s) ahead)" % (100 * beats, len(prediction))}


def ladder_forecast_calibrated(series, order=2, alpha=0.1, min_history=12, seed=0):
    """Forecast the next value of a numeric `series` with the ladder predictor, wrapped in a CALIBRATED prediction
    interval (Cranmer: an uncalibrated forecast is not a measurement). Rolls ladder_predict over the series to
    gather (prediction, actual) residuals on data it did not fit, calibrates a conformal forecaster on those
    residuals, and returns the next-step POINT forecast plus an interval with MEASURED coverage -- not an assumed
    one. `alpha` sets the miss rate (0.1 -> 90% intervals).

    Returns a dict: `point` (next-value forecast), `interval` (lo, hi), `half_width`, `coverage` (the target),
    `empirical_coverage` (fraction of held-out actuals the intervals actually covered -- the honest check), and
    `n_calibration` (how many residuals backed the interval). Falls back to a point forecast with no interval when
    the history is too short to calibrate honestly. Delegates the interval to the existing ConformalForecaster --
    no new calibration math."""
    series = list(series)
    if len(series) < min_history + 2:
        pred = ladder_predict(series, order=order, seed=seed)
        pt = pred["prediction"][0] if pred["prediction"] else (series[-1] if series else 0.0)
        return {"point": pt, "interval": None, "half_width": None, "coverage": None,
                "empirical_coverage": None, "n_calibration": 0,
                "reason": "history too short to calibrate an interval (need >= %d) -- point forecast only"
                          % (min_history + 2)}

    # roll the predictor: at each step predict the next value from the prefix, record (pred, actual).
    preds, actuals = [], []
    for i in range(min_history, len(series) - 1):
        prefix = series[:i + 1]
        p = ladder_predict(prefix, order=order, seed=seed)
        pt = p["prediction"][0] if p["prediction"] else prefix[-1]
        preds.append(float(pt))
        actuals.append(float(series[i + 1]))

    # calibrate a conformal forecaster on the residuals (delegated -- the interval math already exists).
    from holographic.mesh_and_geometry.holographic_conformal import ConformalForecaster
    cf = ConformalForecaster(alpha=alpha)
    cf.calibrate(np.asarray(preds, float), np.asarray(actuals, float))

    # measured (empirical) coverage on the held-out pairs -- the honest check Cranmer demands.
    covered = sum(1 for pr, ac in zip(preds, actuals) if cf.covers(pr, ac))
    empirical = covered / max(1, len(preds))

    # the next-step forecast from the FULL series, wrapped in the calibrated interval.
    p_full = ladder_predict(series, order=order, seed=seed)
    point = float(p_full["prediction"][0]) if p_full["prediction"] else float(series[-1])
    iv = cf.predict(point)
    return {"point": point, "interval": iv["interval"], "half_width": iv["half_width"],
            "coverage": iv["coverage"], "empirical_coverage": round(empirical, 4),
            "n_calibration": len(preds),
            "reason": "ladder point forecast wrapped in a conformal %.0f%% interval; measured coverage %.1f%% on "
                      "held-out (Cranmer: an uncalibrated forecast is not a measurement)"
                      % (100 * (1 - alpha), 100 * empirical)}


def _selftest_forecast_calibrated():
    """ladder_forecast_calibrated contracts: on a predictable periodic series the calibrated interval achieves
    close to its target coverage on held-out data, and returns a finite interval around the point forecast; on a
    short series it returns a point forecast with no interval, said plainly."""
    periodic = [0, 1, 2, 3, 4] * 30
    r = ladder_forecast_calibrated(periodic, order=2, alpha=0.1)
    assert r["interval"] is not None and r["half_width"] is not None
    assert r["n_calibration"] > 10
    assert 0.7 <= r["empirical_coverage"] <= 1.0, r["empirical_coverage"]
    lo, hi = r["interval"]
    assert lo <= r["point"] <= hi                              # the point sits inside its own interval
    short = ladder_forecast_calibrated([0, 1, 2], order=2)
    assert short["interval"] is None and short["n_calibration"] == 0   # too short: honest point-only
    print("ladder_forecast_calibrated OK (periodic series: %d calibration residuals, measured coverage %.0f%% vs "
          "90%% target, point %.1f inside its interval; short series returns point-only honestly)"
          % (r["n_calibration"], 100 * r["empirical_coverage"], r["point"]))


def reconstruct_corpus(tower):
    """Expand a climbed `tower` back to its ORIGINAL corpus of base symbols (the inverse of climb -- the
    `decompose` kit slot made real and TESTED, not just claimed). The top level's corpus is expressed in promoted
    atom ids; this walks every level's child map, recursively replacing each promoted id with its (a,b) children
    until only base symbols remain. Returns the reconstructed corpus (list of sequences). By construction this is
    LOSSLESS for the SEQUENCE lens -- climb only re-expresses, never discards -- so reconstruct(climb(corpus)) ==
    corpus EXACTLY.

    The STRUCTURE lens is DIFFERENT by design: it treats each group as a SET of part TYPES (it dedupes on entry
    and promotes unordered pairs), so it answers 'which parts are present', NOT 'how many, in what order'.
    reconstruct of a structure tower therefore returns the SET of base symbols per group -- neither original order
    NOR duplicate counts survive. That is the structure lens's premise (order and multiplicity carry no info for
    an instanced scene), stated loudly so it is a documented semantic, not a silent loss. Found by the down-sweep:
    a tower you cannot decompress is useless, and the kit claimed 'round-trips by construction' without a function
    to prove it.

    WHY the sequence lens round-trips exactly: every promotion maps a pair (a,b) -> one new id and records
    (new_id -> (a,b)); expansion is the exact inverse. Base symbols are ids that were never a promotion target."""
    child_map = {}
    for lv in tower:
        for info in lv.get("atoms", {}).values():
            child_map[info["raw_id"]] = info["children"]

    def expand(sym):
        if sym not in child_map:
            return [sym]
        a, b = child_map[sym]
        return expand(a) + expand(b)

    top = tower[-1]["corpus"]
    return [[base for sym in seq for base in expand(sym)] for seq in top]


def _make_planted_corpus(seed=0, n=200):
    """A synthetic corpus with PLANTED 2-level structure: base atoms combine into fixed 'words' (pairs), words
    combine into fixed 'phrases' (pairs of words), and each sequence is a few phrases in a row. A correct ladder
    recovers the words at depth 1 and the phrases at depth 2, then STOPS (random phrase ORDER has no depth-3
    structure to find). Returns the corpus (list of sequences of ints). The variety is real: phrase choice and
    count vary per sequence, so promotion has genuine repeated structure to compress but no infinite ladder."""
    rng = np.random.default_rng(seed)
    words = [(0, 1), (2, 3), (4, 5), (6, 7)]                    # four 2-symbol words over 8 base atoms
    phrases = [(words[0], words[1]), (words[2], words[3]),      # four 2-word phrases (the depth-2 structure)
               (words[0], words[3]), (words[2], words[1])]
    corpus = []
    for _ in range(n):
        k = int(rng.integers(2, 5))                            # 2-4 phrases per sequence (varied length)
        seq = []
        for _ in range(k):
            ph = phrases[int(rng.integers(len(phrases)))]      # varied phrase choice
            seq.extend(list(ph[0]) + list(ph[1]))              # 4 base symbols per phrase
        corpus.append(seq)
    return corpus


def _make_instanced_scene_corpus(seed=0, n=200):
    """A synthetic STRUCTURE corpus: scenes that are BAGS of parts, where each part is a bag of vertex atoms, and
    a few parts RECUR across scenes in varied order (an instanced scene -- the same chair in many rooms). A
    correct structure-lens climb promotes the recurring vertex-sets into parts regardless of order. Returns a
    list of groups (each group a list of atom ids). The recurrence is real but the ORDER is shuffled per scene,
    so only an order-INDEPENDENT lens finds it."""
    rng = np.random.default_rng(seed)
    parts = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]     # four recurring vertex-sets (parts)
    corpus = []
    for _ in range(n):
        k = int(rng.integers(2, 4))                            # 2-3 parts per scene
        scene = []
        for _ in range(k):
            p = list(parts[int(rng.integers(len(parts)))])
            scene.extend(p)
        rng.shuffle(scene)                                     # SHUFFLE: order carries no information
        corpus.append(scene)
    return corpus


def _selftest():
    """Contracts (cross-seed, not single-seed thresholds):

    1. On a corpus with PLANTED 2-level structure, climb builds a tower that COMPRESSES (bits fall level over
       level) and then STOPS with a terminal refusal on record -- it does not climb forever.
    2. On PURE-NOISE input (uniform random), climb refuses early (shallow tower) -- the anti-memorization gate.
    3. The Q5 pre-gate fires on noise (refusal mentions the floor) OR the gain gate fires -- either way it stops.
    4. STABLE ATOM IDS: two climbs of the same corpus produce byte-identical atom id sets (hashlib, determinism).
    5. LEVEL ISOLATION: contamination_check passes (no atom id shared across levels).
    6. prior_tower is reserved (raises NotImplementedError), and a non-sequence lens raises (L2 not here yet).
    """
    # (1) planted structure compresses a LOT then stops. (One BPE pass promotes pairs AND pairs-of-pairs, so the
    #     2-level structure is captured in one ladder rung -- depth 1 with a large gain -- then depth 2 is refused
    #     because the remaining phrase ORDER is random. The contract is 'big compression then a logged stop', not
    #     a fixed depth: how many rungs a real structure takes depends on the lens's own internal depth.)
    corpus = _make_planted_corpus(seed=0)
    tower = climb(corpus, min_gain=0.02, max_depth=8)
    assert len(tower) >= 2, "planted structure should climb at least one level"
    assert tower[1]["bits"] < 0.5 * tower[0]["bits"], "the first level should compress the planted structure a lot"
    assert tower[-1]["terminal"] and "refusal" in tower[-1], "the climb must stop with a reason on record"

    # (2) noise refuses early / shallow.
    rng = np.random.default_rng(1)
    noise = [list(rng.integers(0, 16, size=6)) for _ in range(200)]
    noise_tower = climb(noise, min_gain=0.05, max_depth=8)
    assert noise_tower[-1]["terminal"]
    assert len(noise_tower) <= len(tower), "noise should not out-climb planted structure"

    # (3) something refused loudly with a numeric reason (pre-gate floor OR the gain gate).
    assert any("floor" in n or "topped out" in n or "gain" in n
               for lv in noise_tower for n in lv.get("negatives", []))

    # (4) determinism of atom ids across two climbs.
    t2 = climb(_make_planted_corpus(seed=0), min_gain=0.02, max_depth=8)
    ids_a = tuple(sorted(a for lv in tower for a in lv.get("atoms", {})))
    ids_b = tuple(sorted(a for lv in t2 for a in lv.get("atoms", {})))
    assert ids_a == ids_b, "atom ids must be bit-identical across climbs (hashlib, not hash())"

    # (5) level isolation.
    ok, coll = contamination_check(tower)
    assert ok, ("atom id shared across levels: %r" % coll)

    # (6) reserved / guarded paths + the STRUCTURE lens (L2).
    try:
        climb(corpus, prior_tower=[{}]); assert False, "prior_tower should be reserved"
    except NotImplementedError:
        pass
    try:
        climb(corpus, lens="nonsense"); assert False, "unknown lens should raise"
    except ValueError:
        pass

    # (7) STRUCTURE lens: an instanced scene (recurring parts in SHUFFLED order) compresses under the structure
    #     lens, and the sequence lens does WORSE on it -- because order carries no information here, so
    #     order-dependent promotion cannot find the parts. This is the whole reason the structure lens exists.
    scene = _make_instanced_scene_corpus(seed=0)
    struct_tower = climb(scene, lens="structure", min_gain=0.02, max_depth=6)
    assert struct_tower[1]["bits"] < struct_tower[0]["bits"], "structure lens should compress recurring parts"
    seq_on_scene = climb(scene, lens="sequence", min_gain=0.02, max_depth=6)
    struct_gain = struct_tower[0]["bits"] - struct_tower[-1]["bits"]
    seq_gain = seq_on_scene[0]["bits"] - seq_on_scene[-1]["bits"]
    assert struct_gain >= seq_gain, ("structure lens should beat sequence on shuffled parts: %.1f vs %.1f"
                                     % (struct_gain, seq_gain))
    # structure lens atom ids are also stable + isolated.
    assert contamination_check(struct_tower)[0]

    # (8) identify_level (L8): classifies regime + picks the lens WITHOUT guessing, and calls noise IRREDUCIBLE
    #     (the shuffle-null cancels chance co-occurrence -- high-D noise has basins too).
    seq_sig = identify_level(corpus)
    assert seq_sig["order_dependent"] and seq_sig["lens"] == "sequence"     # planted sequence wants sequence lens
    assert seq_sig["regime"] == "nested/structured"
    scene_sig = identify_level(scene)
    assert not scene_sig["order_dependent"] and scene_sig["lens"] == "structure"   # shuffled parts want structure
    noise_sig = identify_level(noise)                          # reuse the real noise corpus from (2), not a re-seeded one
    assert noise_sig["regime"] == "irreducible", noise_sig     # noise is NOT structure above the null
    rep_sig = identify_level([[5] * 8 for _ in range(80)])
    assert rep_sig["regime"] == "repetitive"

    # (9) KIT (L6): every level carries an invariant kit with NO silent gaps -- each slot is WIRED or a declared
    #     negative ('no ...' with a reason). This is the enforced 'we also do X at every level'.
    for lv in tower:
        rep = kit_report(lv)
        assert not rep["silent_gaps"], ("level %d has silent kit gaps: %r" % (lv["depth"], rep["silent_gaps"]))
        assert rep["wired"], "a level should have some wired kit slots"

    # (10) ADAPTIVE PIPELINE (panel convergence): the dispatcher routes each regime to the right method and
    #      ABSTAINS on noise (the SETI gate -- never fabricate structure). The lens is picked per-signal.
    assert adaptive_pipeline(corpus)["method"] == "climb"                # structured -> climb
    assert adaptive_pipeline(corpus)["lens"] == "sequence"              # with the picked lens
    assert adaptive_pipeline(scene)["lens"] == "structure"             # a different signal picks a different lens
    assert adaptive_pipeline(noise)["method"] == "abstain"             # noise is refused, not 'cleaned'
    assert adaptive_pipeline([[5] * 8 for _ in range(80)])["method"] == "fold"   # repetitive -> cheap fold

    # (11) LADDER_PREDICT (forecasting -- compression<->prediction duality): a PERIODIC signal is predicted by the
    #      ladder and BEATS persistence hugely (next != last on a cycle); NOISE falls to persistence and does NOT
    #      claim to beat it (the SETI gate on the time axis -- no fabricated trends).
    periodic = [0, 1, 2, 3] * 40
    pr = ladder_predict(periodic, order=2)
    assert pr["method"] == "ladder" and pr["beats_persistence"] > 0.5, pr   # ladder wins big on a cycle
    assert pr["prediction"] == [0], pr                                       # correct next in cycle after ...2,3
    rng2 = np.random.default_rng(0)
    noise_seq = list(rng2.integers(0, 8, size=300))
    nr = ladder_predict(noise_seq, order=2)
    assert nr["method"] == "persistence" and nr["beats_persistence"] < 0.02, nr   # noise: no fabricated trend

    # (12) SWEEP (up/down/sideways, L5): structured data is COMPLETE in all three directions; a shuffled scene
    #      wears only the STRUCTURE costume (the sequence lens fails on it -- correct); NOISE is irreducible with
    #      all three directions flagged as gaps (no fabricated structure in any direction).
    sweep_seq = sweep_directions(corpus)
    assert sweep_seq["complete"] and not sweep_seq["gaps"], sweep_seq
    sweep_scene = sweep_directions(scene)
    assert sweep_scene["sideways"]["lenses_that_fit"] == ["structure"], sweep_scene   # one costume only
    sweep_noise = sweep_directions(noise)
    assert sweep_noise["irreducible"] and set(sweep_noise["gaps"]) == {"down", "up", "sideways"}, sweep_noise

    # (10) RECONSTRUCT (down-sweep): the SEQUENCE-lens tower round-trips EXACTLY -- reconstruct(climb(corpus)) ==
    #      corpus. This proves the 'decompose' kit slot, which was CLAIMED without a function until the sweep. The
    #      STRUCTURE lens is order/count-agnostic by design, so it recovers the SET of base symbols, not the order.
    seq_tower = climb(corpus)                                   # corpus is the planted sequence corpus from (1)
    assert reconstruct_corpus(seq_tower) == corpus, "sequence-lens tower must round-trip exactly"
    struct_recon = reconstruct_corpus(struct_tower)
    orig_sets = [set(g) for g in scene]
    recon_sets = [set(g) for g in struct_recon]
    assert recon_sets == orig_sets, "structure-lens tower must recover the SET of base part-types per group"

    # (11) CHART_SPACE (L13): a well-separated alphabet has real cleanup basins that beat a band-limited random
    #      null; a clumped alphabet (atoms nearly identical) has NO distinctive basins and reads BELOW the null.
    #      Deterministic given the seed. This is the cartography: structure only counts above the matched null.
    rng = np.random.default_rng(0)
    sep = rng.standard_normal((8, 128)); sep /= np.linalg.norm(sep, axis=1, keepdims=True)
    atlas = chart_space(sep, rays=60, seed=0)
    assert atlas["structure_over_null"] > 0.02, ("separated alphabet must beat the null: %r" % atlas)
    assert chart_space(sep, rays=60, seed=0) == atlas, "chart_space must be deterministic given the seed"
    base = rng.standard_normal(128); base /= np.linalg.norm(base)
    clump = base + 0.03 * rng.standard_normal((8, 128)); clump /= np.linalg.norm(clump, axis=1, keepdims=True)
    clump_atlas = chart_space(clump, rays=60, seed=0)
    assert clump_atlas["structure_over_null"] < atlas["structure_over_null"], "clumped alphabet has weaker basins"

    # (12) BANK-OR-FORMULA gate (L9, Quilez Q1): the amortized decision is honest -- cheap-to-eval or rarely-reused
    #      entries are NOT banked (negative storage); expensive-and-reused entries are. Break-even = lookup/eval.
    assert bank_or_formula(0.3, 1.0, 10, 64)["bank"] is False           # eval cheaper than a lookup -> formula
    assert bank_or_formula(24.0, 0.005, 1000, 4096)["bank"] is False    # reuse below break-even -> formula
    assert bank_or_formula(5000.0, 0.9, 100, 4096)["bank"] is True      # costly + reused -> bank
    # the saving is the corrected amortized model H*eval - lookup (a miss builds AND looks up)
    v = bank_or_formula(100.0, 0.5, 10, 8)
    assert abs(v["saving_us_per_query"] - (0.5 * 100.0 - 0.5)) < 1e-6, v

    depth = len(tower) - 1
    print("holographic_ladder selftest OK (planted 2-level corpus climbs to depth %d then stops: %s; noise tops "
          "out at depth %d; STRUCTURE lens compresses shuffled instanced parts and beats the sequence lens on "
          "them; atom ids bit-identical across climbs; level isolation clean; prior_tower reserved)"
          % (depth, tower[-1]["refusal"][:36], len(noise_tower) - 1))


if __name__ == "__main__":
    _selftest()
    _selftest_forecast_calibrated()
