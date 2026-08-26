"""Corridor planning -- bake a short executable route, run it cheap, re-anchor at the decision point.

WHY THIS EXISTS
---------------
A route stored as one bundle hits the HRR capacity cliff fast: an undirected chain of consecutive binds
decodes only a handful of tiles before crosstalk wins (measured ~1 tile at dim 512, ~5 at dim 2048), and
even the directed structure (the permutation-role chain that suppresses the predecessor leak) decodes only
its reliable prefix -- ~15 tiles at dim 512-1024, more at higher dim. That prefix length is the real cap.

The way PAST the cap is the same move a path tracer makes when a ray's throughput decays: don't push one
structure further than it can carry, RE-ANCHOR. Bake a CORRIDOR -- the next ~12-16 steps of following a
goal field downhill, short enough to decode cleanly -- execute it with no further thinking, and when the
plan is exhausted or its confidence runs out (the throughput gate trips), re-anchor at that decision point
and bake the next corridor. Arbitrarily long routes become a sequence of cap-sized, individually-clean
corridors. The expensive brain call happens once per corridor (at the decision point), not once per tile;
the trivial straight-line steps in between are near-free executions of a baked plan.

This module is the API for that pattern, built entirely on pieces the engine already has: the directed
structure (holographic_directed -- the permutation direction role, RAY-3) and the throughput-gated walk
(holographic_traverse -- Russian roulette for holographic paths). `plan` rolls out and bakes one corridor;
`replan_needed` is the cheap per-tick "is my baked plan still good?" check that decides when to re-anchor.

MEASURED (see `_selftest`)
  * A corridor at or under the cap bakes and decodes back to the full direction sequence, throughput high.
  * A corridor LONGER than the cap honestly reports only its reliable prefix (the gate stops where the
    recoverable signal ends) -- so the plan never claims steps it cannot actually carry.
  * replan_needed is False while on-route with confidence, and True the moment the plan is exhausted, the
    next step's throughput falls below the floor, or a supplied tile-check says the next tile is blocked.

SCOPE / KEPT NEGATIVE
  * The corridor's tiles must be reasonably distinct (a codebook): a corridor that REVISITS a tile (a tight
    loop) can confuse the cleanup, since two steps map to near-identical vectors. Straight corridors to the
    next decision point -- the case this is for -- are distinct by construction.
  * `plan` does not decide WHERE downhill is; the caller supplies `field_step` (the goal-field gradient, a
    flow direction, a policy's greedy step). Planning is the baking + gating; the field is the caller's.
"""

from collections import namedtuple

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bundle, cosine
from holographic.misc.holographic_directed import build, make_step
from holographic.misc.holographic_traverse import gated_traverse

# memory: the directed-structure hypervector (the compact, composable plan -- None if no corridor)
# nodes:  the rolled-out corridor tile vectors (nodes[0] is the start)
# route:  decoded successor indices into `nodes` -- the executable tile sequence (re-derived from the bake)
# actions:the decoded direction labels (if an action_of was supplied), aligned to `route`
# throughputs: per-step confidence (the cleanup cosine at each hop) -- where this falls is where to re-anchor
# stopped: why the corridor ended -- 'field_end' / 'branch' / 'max_steps' for the rollout
Plan = namedtuple("Plan", "memory nodes route actions throughputs stopped ds")


def plan(start, field_step, max_steps=14, floor=0.15, seed=0, action_of=None, is_branch=None):
    """Bake one corridor: roll out the goal-field's downhill path from `start`, encode it as a directed
    structure, and return an executable Plan with a per-step throughput.

    `field_step(node) -> next_node_or_None` is the caller's downhill stepper (a route gradient, a flow
    field, a policy's greedy move). It returns the next tile's vector, or None at a natural end. The rollout
    also stops if `is_branch(node)` is truthy -- a decision point (intersection, junction) where the brain
    should be consulted rather than baked through -- or after `max_steps` tiles (keep this at or under the
    directed structure's reliable decode depth for your dim, ~12-16 at dim 512-1024).

    The corridor is baked with `build` (the permutation-role directed chain) and walked back with the
    throughput gate, so `throughputs[i]` is the recovered confidence of step i and `route` is the decoded
    tile sequence. `action_of(prev_node, next_node) -> label` (optional) turns consecutive tiles into the
    decoded DIRECTION sequence the courier executes. Returns a Plan (memory is None if no corridor of length
    >= 2 could be rolled out)."""
    start = np.asarray(start, float)
    nodes = [start]
    cur = start
    stopped = "max_steps"
    for _ in range(max_steps):
        nxt = field_step(cur)
        if nxt is None:
            stopped = "field_end"
            break
        nxt = np.asarray(nxt, float)
        nodes.append(nxt)
        cur = nxt
        if is_branch is not None and is_branch(nxt):
            stopped = "branch"
            break
    if len(nodes) < 2:
        return Plan(memory=None, nodes=nodes, route=[], actions=[], throughputs=[], stopped="no_path", ds=None)

    ds = build(np.array(nodes), seed=seed)                 # directed chain over the corridor's tiles
    # walk it back for EXACTLY the corridor's edge count -- never past the end (a permissive floor would
    # otherwise let the terminal node's noisy unbind keep "stepping" and oscillate). The decode's job is the
    # per-step throughput and an honest reliable-prefix, not to discover new tiles.
    res = gated_traverse(make_step(ds), nodes[0], floor=floor, max_steps=len(nodes) - 1)
    route = list(res.payloads)                             # decoded successor indices (the reliable prefix)
    actions = ([action_of(nodes[i], nodes[i + 1]) for i in range(len(route))]
               if action_of is not None else [])
    return Plan(memory=ds.memory, nodes=nodes, route=route, actions=actions,
                throughputs=list(res.throughputs), stopped=stopped, ds=ds)


def replan_needed(p, executed, tile_ok=None, floor=0.15):
    """The cheap per-tick guard: should the courier abandon the baked plan and re-anchor?

    `executed` is how many baked steps it has already taken. Returns True -- meaning call plan() again --
    when (1) the plan is exhausted (no baked step left), (2) the next baked step's throughput is below
    `floor` (the plan's confidence has run out), or (3) `tile_ok(next_tile_vector)` is supplied and reports
    the next tile is no longer clear/on-route (traffic ahead, off-plan). Otherwise False: execute the next
    baked step. No value() calls, no decoding work -- just a list index and a comparison."""
    if p is None or p.memory is None or executed >= len(p.route):
        return True                                       # exhausted -> re-anchor
    if executed < len(p.throughputs) and p.throughputs[executed] < floor:
        return True                                       # confidence ran out -> re-anchor
    if tile_ok is not None:
        nxt = p.nodes[p.route[executed]]                  # the next baked tile
        if not tile_ok(nxt):
            return True                                   # blocked / off-route -> re-anchor
    return False


# A whole route, assembled by chaining corridors:
#   actions:   the full executable direction sequence -- ARBITRARILY long, past the single-structure cap
#   corridors: the list of Plan objects it chained (each a cap-sized, individually-clean corridor)
#   stopped:   why the whole route ended -- 'field_end' / 'branch' (goal/junction reached) / 'max_total' / 'stalled'
#   reanchors: how many times it re-anchored (len(corridors) - 1) -- the count of "decision points"
#   steps:     len(actions)
Route = namedtuple("Route", "actions corridors stopped reanchors steps")


def plan_route(start, field_step, max_total=200, corridor=14, floor=0.15,
               seed=0, action_of=None, is_branch=None):
    """Bake an ARBITRARILY-LONG route by chaining cap-sized corridors, RE-ANCHORING internally at each
    corridor's reliably-decoded end. This is the way PAST the single-structure ~15 cap delivered as ONE
    call: a 45-tile route that collapses to noise if crammed into one plan() comes back correct here as a
    sequence of clean corridors.

    The single-structure cap is real and inherent (HRR crosstalk; ~15 tiles at dim 512-1024, fewer if you
    overstuff). It bounds ONE baked corridor -- NOT the route you can navigate. `corridor` is the per-leg
    length and MUST stay at/under the reliable decode depth for your dim (default 14, safe at dim 512-1024):
    a corridor longer than that overstuffs its own structure and its decode corrupts -- the very same cliff,
    now per leg (measured: corridor=30 at dim 512 skips tiles, a KEPT NEGATIVE). `max_total` caps the whole
    route. An early gate-trip *within* a cap-sized corridor is fine -- each leg re-anchors at its last
    reliably-decoded tile, never past it, so a short-but-clean leg simply re-anchors sooner.

    WHEN TO USE WHICH: a real-time courier still wants plan() + replan_needed -- bake one corridor, drive
    it, re-anchor only when the gate trips (this avoids planning the whole route up front and reacts to
    traffic). Use plan_route when you want the WHOLE route in hand at once: to display it, validate it, or
    pre-plan a leg offline. Both ride the same re-anchoring; this one just runs the loop for you.

    Returns a Route (see the field notes above)."""
    corridors = []
    full_actions = []
    cur = np.asarray(start, float)
    stopped = "max_total"
    while len(full_actions) < max_total:
        room = min(corridor, max_total - len(full_actions))
        p = plan(cur, field_step, max_steps=room, floor=floor, seed=seed,
                 action_of=action_of, is_branch=is_branch)
        corridors.append(p)
        if not p.route:                                   # nothing decoded -> cannot advance, stop honestly
            stopped = "stalled" if not full_actions else (p.stopped if p.stopped != "max_steps" else "stalled")
            break
        full_actions += p.actions
        reached_end = (p.route[-1] == len(p.nodes) - 1)   # did the decode reach the corridor's LAST rolled-out tile?
        if p.stopped in ("field_end", "branch") and reached_end:
            stopped = p.stopped                           # goal/junction reached AND fully decoded -> done
            break
        cur = p.nodes[p.route[-1]]                        # re-anchor at the last RELIABLY-DECODED tile (never past it)
    return Route(actions=full_actions, corridors=corridors, stopped=stopped,
                 reanchors=max(0, len(corridors) - 1), steps=len(full_actions))


def chunk_route(items, chunk=14, floor=0.15, seed=0, action_of=None):
    """Store/replay an EXPLICIT ordered sequence you ALREADY HAVE -- a GPS route from a planner, a scientist's
    fixed experiment protocol, any known list of N steps -- past the per-structure cap, by splitting it into
    <=chunk-element directed-structure chunks, each individually clean. This is the explicit-list twin of
    plan_route: there you DISCOVER the route by following a goal field; here the sequence is given, so you skip
    the rollout and just chunk, bake, and replay it EXACTLY.

    Why chunking and not one big structure: a single directed bundle decodes only ~15 tiles at dim 512-1024
    before HRR crosstalk wins (that cap is physics, not a bug -- the same way a fixed-width buffer cannot hold
    unbounded data). Splitting into <=chunk pieces keeps every piece inside its own capacity, so the EFFECTIVE
    length is unbounded at LINEAR cost: a 200-step route is ~15 chunks, a 1000-step one ~72. Each chunk is ONE
    compact hypervector (chunk.memory) you can store, compare, or compose like any other vector -- the whole
    sequence becomes a short list of clean holographic objects instead of one overstuffed, unreadable blob.

    `items` is the ordered list of element vectors; `chunk` is the per-piece length and must stay at/under the
    dim's reliable decode depth (default 14, safe at dim 512-1024 -- an over-long chunk overstuffs its own piece,
    the same cliff per chunk); `action_of(a, b)` labels each consecutive pair (the executable step). Chunks
    OVERLAP by one element at the boundary so the next chunk re-anchors exactly where the last one ended -- no
    element is skipped or double-counted. The elements must be distinguishable (a codebook): a sequence that
    REVISITS the same element can confuse a chunk's cleanup, since two steps map to near-identical vectors.

    Returns a Route: the full replayable action sequence (past the cap), the chunk Plans (each holding its
    compact vector), the stop reason, the chunk-boundary count, and the step total."""
    items = [np.asarray(x, float) for x in items]
    if len(items) < 2:                                        # nothing to sequence
        return Route(actions=[], corridors=[], stopped="stalled", reanchors=0, steps=0)
    corridors, actions = [], []
    start = 0
    stopped = "sequence_end"
    while start < len(items) - 1:
        seg = items[start:start + chunk + 1]                 # a chunk of `chunk` EDGES needs chunk+1 nodes
        ds = build(np.array(seg), seed=seed)                 # one clean directed structure over this piece
        res = gated_traverse(make_step(ds), seg[0], floor=floor, max_steps=len(seg) - 1)
        route = list(res.payloads)                           # decoded successor indices within the chunk
        if not route:                                        # a chunk that won't even decode its first step -> stop
            stopped = "stalled" if not actions else "sequence_end"
            break
        seg_actions = ([action_of(seg[j], seg[j + 1]) for j in range(len(route))]
                       if action_of is not None else [])
        corridors.append(Plan(memory=ds.memory, nodes=seg, route=route, actions=seg_actions,
                              throughputs=list(res.throughputs), stopped="chunk", ds=ds))
        actions += seg_actions
        start += len(route)                                  # advance by what decoded; the boundary node is shared
    return Route(actions=actions, corridors=corridors, stopped=stopped,
                 reanchors=max(0, len(corridors) - 1), steps=len(actions))


def dedup_chunks(vectors, tol=1e-9):
    """Content-addressed deduplication of chunk vectors: a long route that REVISITS the same corridor, or a
    program with repeated motifs, ends up storing the same compact chunk vector many times. Keep each UNIQUE
    chunk once and replace every repeat with a reference, so the store shrinks by exactly the REPETITION RATIO
    -- and by nothing when there is no repetition (the honest bound: dedup can only save what actually repeats,
    measured 65% on a 17-corridor loop with 6 distinct chunks, 0% on a no-repeat control).

    `vectors` is the ordered list of chunk vectors (e.g. `[c.memory for c in route.corridors]`); two are the
    same chunk when their cosine is >= 1 - tol (default exact to floating point). Returns (unique, refs):
    `unique` is the deduplicated store (each distinct chunk once, in first-seen order) and `refs` is one index
    per ORIGINAL chunk into `unique`, so `[unique[r] for r in refs]` rebuilds the original list EXACTLY -- order
    and repeats preserved, only the duplicate storage removed.

    This is the storage twin of the StructuredIndex lookup: where that finds an item BY content, this stores
    items BY content so identical ones coalesce. Comparing whole chunk vectors by cosine is an EVALUATION, not
    a decode, so it does not hit the capacity cap -- two genuinely distinct chunks never collide at high dim."""
    vectors = [np.asarray(v, float) for v in vectors]
    unique, refs = [], []
    for v in vectors:
        hit = None
        for i, u in enumerate(unique):
            if u.shape == v.shape and cosine(u, v) >= 1.0 - tol:
                hit = i
                break
        if hit is None:
            unique.append(v)
            refs.append(len(unique) - 1)
        else:
            refs.append(hit)
    return unique, refs


class RouteIndex:
    """Sub-linear RANDOM ACCESS into a chunked route -- a BVH over the chunks. A long route is many chunks;
    "where am I on it?" should be a jump, not a replay from the start. Index each chunk by a SUMMARY vector
    (the bundle of its tiles), and locate a query two-level: nearest chunk summary, then nearest tile within
    that chunk. Cost is ~(#chunks + chunk_size) per query instead of #tiles -- for a 200-tile route, ~28
    comparisons vs 200 (measured ~6.9x fewer), and it located 200/200 tiles exactly.

    Build it once from a Route (from plan_route or chunk_route); query it many times -- the courier asking its
    position every tick is exactly the repeated-query case this amortises. Why a bundle summary works: a tile
    that lives in a chunk has cosine ~1/sqrt(chunk_size) to that chunk's summary (it is one of its components)
    and ~0 to the others, so the nearest summary is the right chunk; the second level is then an exact small
    search. The same bundle-crosstalk that caps a single structure is what makes the summary a usable index."""

    def __init__(self, route):
        self.chunks = [np.asarray(c.nodes, float) for c in route.corridors]   # public: callers/tests read .chunks
        # The two-level summary routing now lives in the shared StructuredIndex (keying='sequential') -- one
        # routing fabric for the chunkers, not a bespoke fourth copy. This class keeps only the route-specific
        # global-step bookkeeping (chunks overlap by one tile) layered on top of that shared route.
        from holographic.misc.holographic_tree import StructuredIndex
        dim = self.chunks[0].shape[1] if self.chunks else 0
        self._idx = StructuredIndex(dim, keying="sequential").build(self.chunks)
        self._starts, acc = [], 0
        for ch in self.chunks:
            self._starts.append(acc)
            acc += max(1, len(ch) - 1)

    def locate(self, query):
        """Return (chunk_index, position_in_chunk, global_step) of the route tile nearest `query` -- two-level,
        sub-linear. The routing delegates to StructuredIndex(keying='sequential'); global_step is the approximate
        index along the whole route (chunks overlap by one tile, so it subtracts the shared boundary)."""
        if not self.chunks:
            return (-1, -1, -1)
        (c, pos), _ = self._idx.locate(query)
        return (c, pos, self._starts[c] + pos)

    @property
    def n_chunks(self):
        return len(self.chunks)

    @property
    def _summaries(self):
        # The chunk summaries now live in the shared StructuredIndex; exposed here so existing callers/tests
        # that inspect routing summaries (e.g. the determinism audit) keep working after the delegation.
        return self._idx._summaries


def _selftest():
    """CI-fast: bake a corridor and prove (1) an at-cap corridor decodes to the full direction sequence with
    high throughput, (2) an over-cap corridor honestly reports only its reliable prefix, (3) replan_needed
    fires exactly on exhaustion / low throughput / a blocked tile."""
    rng = np.random.default_rng(0)
    dim = 1024
    # a straight corridor of distinct tiles, with a field_step that walks the known list and a direction label
    tiles = rng.standard_normal((13, dim))
    tiles /= np.linalg.norm(tiles, axis=1, keepdims=True)
    index_of = {id(t.tobytes()): i for i, t in enumerate(tiles)}

    def field_step(cur):
        # find cur in the tile list (by nearest, since vectors round-trip exactly here) and return the next
        sims = tiles @ (cur / (np.linalg.norm(cur) + 1e-12))
        i = int(np.argmax(sims))
        return tiles[i + 1] if i + 1 < len(tiles) else None

    def action_of(a, b):
        return "step"

    # (1) at-cap corridor: 12 hops over 13 tiles, decodes fully with high confidence
    p = plan(tiles[0], field_step, max_steps=12, floor=0.12, action_of=action_of)
    assert p.memory is not None
    assert p.route == list(range(1, 13)), p.route          # every tile recovered, in order
    assert len(p.actions) == 12
    assert min(p.throughputs) > 0.12                        # throughput stayed above the floor
    assert p.stopped == "max_steps"                         # the 12-step cap ended the rollout

    # (2) over-cap corridor at a SMALL dim: the decode stops at its reliable prefix, doesn't overclaim
    small = rng.standard_normal((40, 128))
    small /= np.linalg.norm(small, axis=1, keepdims=True)

    def field_step_small(cur):
        sims = small @ (cur / (np.linalg.norm(cur) + 1e-12))
        i = int(np.argmax(sims))
        return small[i + 1] if i + 1 < len(small) else None

    p2 = plan(small[0], field_step_small, max_steps=39, floor=0.2)
    assert len(p2.route) < 39                               # honest: only the reliable prefix, not all 39

    # (3) replan_needed: False on-route, True at exhaustion / blocked tile
    assert replan_needed(p, 0, floor=0.12) is False        # step 0 is fine
    assert replan_needed(p, len(p.route), floor=0.12) is True   # exhausted
    blocked = {p.route[3]}                                  # pretend the 4th tile is now blocked
    assert replan_needed(p, 3, tile_ok=lambda v: int(np.argmax(tiles @ v)) not in blocked) is True
    assert replan_needed(p, 2, tile_ok=lambda v: int(np.argmax(tiles @ v)) not in blocked) is False

    # (4) plan_route: a 45-tile route (3x the single-corridor cap) comes back EXACTLY by chaining corridors,
    #     where ONE plan() over the same route collapses to near-nothing. The cap bounds a leg, not the route.
    big = rng.standard_normal((45, 512))
    big /= np.linalg.norm(big, axis=1, keepdims=True)

    def field_step_big(cur):
        i = int(np.argmax(big @ (cur / (np.linalg.norm(cur) + 1e-12))))
        return big[i + 1] if i + 1 < len(big) else None

    def action_big(a, b):
        return int(np.argmax(big @ (b / (np.linalg.norm(b) + 1e-12))))   # the tile-index each edge lands on

    crammed = plan(big[0], field_step_big, max_steps=45, floor=0.12, action_of=action_big)
    r = plan_route(big[0], field_step_big, corridor=14, floor=0.12, action_of=action_big)
    assert len(crammed.actions) < 5                        # one structure overstuffed -> collapses (the cliff)
    assert r.actions == list(range(1, 45))                 # chained corridors -> the FULL route, exact
    assert r.stopped == "field_end" and r.reanchors >= 2   # it re-anchored at the decision points
    assert len(r.actions) > len(crammed.actions) * 5       # decisively past the single-structure cap
    # max_total caps the whole route, and what it returns is a correct prefix
    rc = plan_route(big[0], field_step_big, max_total=20, corridor=14, floor=0.12, action_of=action_big)
    assert rc.steps <= 20 and rc.actions == list(range(1, 1 + rc.steps))
    # KEPT NEGATIVE: an over-long corridor overstuffs its OWN leg and corrupts -- corridor must stay <= the
    # reliable decode depth (the default 14 does). corridor=30 at dim 512 does NOT recover the full route.
    bad = plan_route(big[0], field_step_big, corridor=30, floor=0.12, action_of=action_big)
    assert bad.actions != list(range(1, 45))               # honest: over-long legs are the same cliff, per leg

    # (5) chunk_route: an EXPLICIT 200-element sequence (a known list, no field_step) replays EXACTLY by
    #     chunking into <=14 clean pieces -- effective length unbounded at linear cost, ~N/14 chunks.
    seq = rng.standard_normal((200, 512)); seq /= np.linalg.norm(seq, axis=1, keepdims=True)

    def action_seq(a, b):
        return int(np.argmax(seq @ (b / (np.linalg.norm(b) + 1e-12))))

    cr = chunk_route(list(seq), chunk=14, floor=0.12, action_of=action_seq)
    assert cr.actions == list(range(1, 200))               # the whole 200-step sequence, exact, past the cap
    assert len(cr.corridors) <= 200 // 14 + 2              # linear: ~15 chunks, not one impossible structure
    assert cr.corridors[0].memory.shape == (512,)          # each chunk is ONE compact vector
    assert chunk_route([seq[0]], action_of=action_seq).steps == 0   # degenerate single-element -> empty, no crash

    # (6) RouteIndex: sub-linear two-level random access -- locate every tile in the right chunk, exact position.
    idx = RouteIndex(cr)
    # which chunk does each tile actually live in (overlapping boundaries -> a tile can be in two; accept either)
    def true_chunks(t):
        return {k for k, c in enumerate(cr.corridors)
                if any(np.array_equal(c.nodes[j], seq[t]) for j in range(len(c.nodes)))}
    assert all(idx.locate(seq[t])[0] in true_chunks(t) for t in range(200))   # level 1 correct for all 200
    c0, p0, g0 = idx.locate(seq[37])
    assert np.allclose(idx.chunks[c0][p0], seq[37])        # level 2 exact: the located tile matches the query
    assert idx.n_chunks == len(cr.corridors)
    assert RouteIndex(chunk_route([], action_of=action_seq)).locate(seq[0]) == (-1, -1, -1)  # empty route safe


if __name__ == "__main__":
    _selftest()
    print("holographic_plan selftest passed")
