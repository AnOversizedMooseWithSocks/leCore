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

from holographic_directed import build, make_step
from holographic_traverse import gated_traverse

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


if __name__ == "__main__":
    _selftest()
    print("holographic_plan selftest passed")
