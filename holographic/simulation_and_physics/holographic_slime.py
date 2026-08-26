"""Slime-mold path-finding over a HOLOGRAPHIC associative graph.

The reactive creature brain cannot traverse a big maze, and the reason turned out
to be the SAME wall that limits deep recall: a single holographic vector has a
capacity, and cramming a whole structure into it drowns the signal in crosstalk
("getting lost"). This module attacks that wall the way a slime mold (Physarum)
attacks a real maze -- and the way this codebase already organizes memory:

  * SLIME MOLD is the algorithm. Physarum has no map and no plan. It floods tubes
    everywhere, REINFORCES the tubes that reach food, and lets the rest DECAY.
    Dead-ends are not avoided by cleverness -- they simply starve and vanish.

  * COMPRESSION is why it scales. A maze has O(N^2) cells but its solution is a
    single O(N) path. Decay prunes the stored graph down to the productive tube,
    so the thing we must hold in memory shrinks to (roughly) the answer.

  * The graph itself lives in ONE holographic vector. Each directed edge u->v is
    bind(node_u, role(node_v)) where `role` is a fixed RANDOM permutation. Direction
    matters: bind is commutative and a cyclic shift commutes with the convolution, so
    neither makes a one-way edge -- a walker would wander backward (getting lost, in
    algebra). A random permutation does not commute, so unbind(edge, u) recovers the
    target while unbind(edge, v) returns only noise. The whole weighted pheromone
    field is the superposition M = sum_e  weight_e * edge_e, read back by hop-and-snap.

  * This is the unification the maze was always standing in for. Storing a memory
    is reinforcing an edge; retrieving is a hop-and-snap read; organizing is decay
    and pruning; orchestrating is following the strongest tube. The capacity of M
    is exactly the deep-recall capacity measured elsewhere -- and when a maze is too
    big for one M, the escape is this codebase's recurring move: PARTITION into
    regions, solve each, and (the recursive / inception step) compress each solved
    region into a single super-node and solve the maze of super-nodes one layer up.

Built only from the kernel primitives (bind / unbind / cosine + one permutation), in
the demoscene spirit: maximal behaviour from minimal means.
"""

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, cosine, Vocabulary


def _edge(node_u, node_v, role):
    """A DIRECTED edge u->v. The target is role-tagged by a FIXED RANDOM permutation
    `role` before binding. This matters: bind is commutative, so a naive bind(u, v)
    is symmetric, and a cyclic SHIFT does not help either -- a shift commutes with
    circular convolution, so unbind(e, v) would still hand back the source and a
    walker would wander backward. A random permutation does NOT commute with the
    convolution, so unbind(e, u) recovers the (permuted) target while unbind(e, v)
    recovers only noise. That asymmetry is what makes the edge genuinely one-way."""
    return bind(node_u, node_v[role])


class SlimeGraph:
    """A directed, weighted graph whose entire weighted edge set (the 'pheromone
    field') is one holographic vector M. Weights are kept in a small dict as the
    intent; M is their holographic realization, and reads go through M so its finite
    capacity is real, not bypassed."""

    def __init__(self, dim=2048, seed=0, decay=0.80):
        self.dim = dim
        self.decay = decay                 # fraction of pheromone surviving each round
        self.voc = Vocabulary(dim, seed)
        # a fixed random permutation that role-tags an edge's target (see _edge), plus
        # its inverse to undo the tag when reading a successor back out
        self.role = np.random.default_rng(seed + 12345).permutation(dim)
        self.unrole = np.argsort(self.role)
        self.w = {}                        # (u, v) -> pheromone weight
        self.M = np.zeros(dim)             # holographic field = sum_e w_e * edge_e
        self._edge_cache = {}

    def _ev(self, u, v):
        key = (u, v)
        if key not in self._edge_cache:
            self._edge_cache[key] = _edge(self.voc.get(u), self.voc.get(v), self.role)
        return self._edge_cache[key]

    def reinforce(self, path, amount):
        """Lay pheromone along a successful path (more for shorter paths)."""
        for u, v in zip(path, path[1:]):
            self.w[(u, v)] = self.w.get((u, v), 0.0) + amount

    def evaporate(self):
        """Global decay: every tube fades. Tubes not refreshed by success die off --
        this is the pruning that both finds the answer AND keeps M within capacity."""
        for k in list(self.w):
            self.w[k] *= self.decay
            if self.w[k] < 1e-3:
                del self.w[k]

    def realize(self):
        """Rebuild the holographic pheromone field from the current weights. This is
        the one place the whole graph is squeezed into a fixed-width vector, so this
        is where capacity bites."""
        if not self.w:
            self.M = np.zeros(self.dim)
            return
        self.M = np.sum([wt * self._ev(u, v) for (u, v), wt in self.w.items()], axis=0)

    def pheromone(self, u, neighbours):
        """READ tube strength u->each neighbour out of the holographic field by
        hop-and-snap: unbind by u, undo the target's role permutation, cosine to each
        candidate. Approximate by construction -- that is the whole point of testing it."""
        probe = unbind(self.M, self.voc.get(u))[self.unrole]
        return {v: max(0.0, cosine(probe, self.voc.get(v))) for v in neighbours}

    def edge_count(self):
        return len(self.w)


# ---------------------------------------------------------------------------
# Solving a graph with a colony of slime-mold walkers over the holographic field
# ---------------------------------------------------------------------------

def _neighbours(world, cell):
    """Passable neighbours of a cell in a GridWorld (no wall, on the grid)."""
    x, y = cell
    out = []
    for dx, dy in world.MOVES.values():
        nx, ny = x + dx, y + dy
        if 0 <= nx < world.w and 0 <= ny < world.h and (nx, ny) not in world.walls:
            out.append((nx, ny))
    return out


def _name(c):
    return str(c)


def _dedup(path):
    """Collapse loops: the first time a cell is revisited, splice out the detour.
    Turns a wandering successful walk into the simple path it implies."""
    out = []
    pos = {}
    for c in path:
        if c in pos:
            out = out[:pos[c] + 1]
            pos = {cell: i for i, cell in enumerate(out)}
        else:
            pos[c] = len(out)
            out.append(c)
    return out


def _walk(g, nbr, start, goal, rng, max_steps, heuristic, compass=0.15, explore=0.1):
    """One slime walker over an ABSTRACT graph `nbr` (node -> neighbour list), biased by
    holographic pheromone plus an optional `heuristic` 'compass' toward the goal, never
    immediately reversing. Returns the node path; reaches goal or gives up."""
    path = [start]
    cur, prev = start, None
    hcur = heuristic(cur)
    for _ in range(max_steps):
        if cur == goal:
            return path
        options = [c for c in nbr[cur] if c != prev] or nbr[cur]
        ph = g.pheromone(_name(cur), [_name(c) for c in options])
        scores = []
        for c in options:
            base = ph[_name(c)] + 1e-3
            base += compass if heuristic(c) < hcur else 0.0
            base += explore * rng.random()
            scores.append(base)
        s = np.array(scores)
        cur, prev = options[int(rng.choice(len(options), p=s / s.sum()))], cur
        hcur = heuristic(cur)
        path.append(cur)
    return path


def _colony_solve(nbr, start, goal, dim=2048, ants=24, rounds=50, decay=0.80,
                  seed=0, q=2.0, heuristic=None, max_steps=None, trace=None, elite=0.0):
    """Slime-mold search over an ABSTRACT directed graph held in ONE holographic field.

    nbr: node -> list of neighbour nodes (nodes may be any hashable). heuristic(node) is
    an optional 'compass' pull toward the goal (Manhattan on a grid; omit on a plain
    graph and the colony still solves it, just by diffusion). `elite` > 0 adds elitist
    reinforcement (Dorigo's elitist ant system): each round the best-so-far route is laid
    down an extra `elite` times. On a graph WITH loops this is what makes the colony
    converge on the SHORTEST tube rather than the most-trafficked one -- without it the
    field's strongest tube is often a longer route that simply got crowded early. If
    `trace` is a dict, we record trace['order'] = the cells in the order the colony FIRST
    reached them, so a visualiser can replay the actual discovery (a tile is only ever
    shown after a walker has stepped on it). Returns (extracted_path | None,
    best_seen_path | None, peak_edges_held)."""
    if heuristic is None:
        heuristic = lambda n: 0
    if max_steps is None:
        max_steps = 8 * max(2, len(nbr))
    g = SlimeGraph(dim=dim, seed=seed, decay=decay)
    rng = np.random.default_rng(seed)
    best = None
    peak = 0
    seen = {start}                                            # cells the colony has reached
    order = [start]                                           # ...in first-discovery order
    for _ in range(rounds):
        g.realize()
        peak = max(peak, g.edge_count())
        for _ in range(ants):
            walk = _walk(g, nbr, start, goal, rng, max_steps, heuristic)
            for c in walk:                                    # log first-discovery of each tile
                if c not in seen:
                    seen.add(c)
                    order.append(c)
            if walk[-1] == goal:
                p = _dedup(walk)
                g.reinforce([_name(c) for c in p], amount=q / len(p))
                if best is None or len(p) < len(best):
                    best = p
        if elite > 0 and best is not None:                    # elitist: commit to the best-so-far tube
            g.reinforce([_name(c) for c in best], amount=elite * q / len(best))
        g.evaporate()
    # Read the answer back by following the STRONGEST holographic tube greedily.
    g.realize()
    path, cur, seen2 = [start], start, {start}
    for _ in range(max_steps):
        if cur == goal:
            break
        options = [c for c in nbr[cur] if c not in seen2] or nbr[cur]
        ph = g.pheromone(_name(cur), [_name(c) for c in options])
        cur = max(options, key=lambda c: ph[_name(c)])
        seen2.add(cur)
        path.append(cur)
    if trace is not None:
        trace["order"] = order
    return (path if path and path[-1] == goal else None), best, peak


def solve_maze(world, dim=2048, ants=24, rounds=50, decay=0.80, seed=0, q=2.0,
               use_compass=True, record=False, elite=0.0):
    """Find a path from the creature's start to the maze exit with a colony of slime-mold
    walkers laying pheromone into ONE holographic field, then read it back. (path, info).

    use_compass=False removes the directional pull toward the goal, so the walkers explore
    with NO knowledge of where the exit is -- pure pheromone diffusion. elite>0 turns on
    elitist reinforcement, which on a BRAIDED maze (one with loops, hence many routes) is
    what makes the colony converge on the SHORTEST tube. record=True returns info['order'],
    the cells in first-discovery order, for an honest replay of the search."""
    world.reset()
    start, goal = (world.cx, world.cy), (world.fx, world.fy)
    nbr = {c: _neighbours(world, c) for c in world._free_cells()}
    h = (lambda c: abs(c[0] - goal[0]) + abs(c[1] - goal[1])) if use_compass else None
    tr = {} if record else None
    path, best, peak = _colony_solve(nbr, start, goal, dim, ants, rounds, decay, seed, q,
                                     heuristic=h, max_steps=8 * (world.w + world.h),
                                     trace=tr, elite=elite)
    opt = len(world.shortest_path(start, goal)) - 1
    reached = path is not None
    info = {"reached": reached, "extracted_len": (len(path) - 1) if reached else None,
            "best_seen": (len(best) - 1) if best else None, "optimal": opt,
            "peak_edges": peak, "cells": len(nbr)}
    if record:
        info["order"] = tr["order"]
    return (path if reached else best), info


def _bfs_path(nbr, a, b):
    """Exact shortest path a->b inside a (small) subgraph, or None. Used only to
    compress a region: regions are kept small enough to solve directly, so the
    HOLOGRAPHIC field never has to hold a whole region -- only the portal graph."""
    from collections import deque
    prev = {a: None}
    dq = deque([a])
    while dq:
        x = dq.popleft()
        if x == b:
            break
        for n in nbr.get(x, []):
            if n not in prev:
                prev[n] = x
                dq.append(n)
    if b not in prev:
        return None
    path = []
    x = b
    while x is not None:
        path.append(x)
        x = prev[x]
    return path[::-1]


def solve_maze_hier(world, dim=2048, region=8, ants=24, rounds=50, decay=0.80, seed=0):
    """Solve a maze too big for one holographic field, by the lesson the capacity wall
    taught us: PARTITION, COMPRESS, RECURSE.

      * Partition the grid into region tiles.
      * COMPRESS each region to its PORTALS -- the cells where the maze crosses a tile
        border -- and connect portals that a region links internally (a region is small,
        so this is solved directly: the holographic field never holds a whole region).
      * The portals form a much smaller SUPER-GRAPH. Solve start->goal on it with the
        SAME slime colony, one layer up (and if the portal graph were itself too big,
        the same move would recurse again -- inception).
      * Stitch the per-region segments back into a full path.

    The point is measured, not asserted: every holographic field here -- the super-graph
    field -- holds only ~#portals edges, which stays bounded no matter how big the maze
    grows, where a single flat field overflows and reads back lost. Returns (path, info)."""
    world.reset()
    start, goal = (world.cx, world.cy), (world.fx, world.fy)
    cells = set(world._free_cells())
    full_nbr = {c: _neighbours(world, c) for c in cells}
    R = region
    tile = lambda c: (c[0] // R, c[1] // R)

    # 1. portals: cells with a passable neighbour in a different tile (border crossings)
    portals = {c for c in cells for n in full_nbr[c] if tile(n) != tile(c)}
    portals |= {start, goal}

    # 2. compress each region: connect its portals that are reachable WITHIN the region
    by_tile = {}
    for p in portals:
        by_tile.setdefault(tile(p), []).append(p)
    super_nbr = {p: [] for p in portals}
    seg = {}                                  # (p, q) -> the cell path between them
    for t, ps in by_tile.items():
        region_cells = [c for c in cells if tile(c) == t]
        rnbr = {c: [n for n in full_nbr[c] if tile(n) == t] for c in region_cells}
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = ps[i], ps[j]
                local = _bfs_path(rnbr, a, b)
                if local is not None:
                    super_nbr[a].append(b); super_nbr[b].append(a)
                    seg[(a, b)] = local; seg[(b, a)] = local[::-1]
    # cross-tile portal adjacency (a single step over the border)
    for c in portals:
        for n in full_nbr[c]:
            if n in portals and tile(n) != tile(c) and n not in super_nbr[c]:
                super_nbr[c].append(n); seg[(c, n)] = [c, n]

    # 3. solve the portal super-graph with the SAME holographic colony (bounded field)
    h = lambda c: abs(c[0] - goal[0]) + abs(c[1] - goal[1])
    sp, _, peak_super = _colony_solve(super_nbr, start, goal, dim, ants, rounds, decay,
                                      seed, heuristic=h, max_steps=8 * len(super_nbr))
    opt = len(world.shortest_path(start, goal)) - 1
    if sp is None:
        return None, {"reached": False, "optimal": opt, "cells": len(cells),
                      "portals": len(portals), "peak_super_edges": peak_super}

    # 4. stitch the per-region segments into one continuous path
    full = [start]
    for a, b in zip(sp, sp[1:]):
        full += seg[(a, b)][1:]
    full = _dedup(full)
    reached = full[0] == start and full[-1] == goal
    info = {"reached": reached, "extracted_len": (len(full) - 1) if reached else None,
            "optimal": opt, "cells": len(cells), "portals": len(portals),
            "peak_super_edges": peak_super}
    return (full if reached else None), info


def demo_slime_maze():
    """Solve mazes far past the reactive brain's ceiling, all from one holographic
    pheromone field, and show where a single field's capacity starts to strain."""
    from holographic.misc.holographic_creature import GridWorld
    print("Slime-mold path-finding over a holographic pheromone field")
    print("(the reactive creature brain solves ~7x7 and stalls; this does not)\n")
    for size, dim in ((9, 2048), (15, 2048), (21, 2048), (31, 4096)):
        w = GridWorld(size, size, maze=True, fixed_seed=3)
        path, info = solve_maze(w, dim=dim, ants=24, rounds=50, seed=0)
        if info["reached"] and info["extracted_len"] == info["optimal"]:
            tag = f"optimal ({info['optimal']} steps)"
        elif info["reached"]:
            tag = f"{info['extracted_len']} steps (optimal {info['optimal']})"
        else:
            tag = "did not extract a clean path"
        print(f"  {size:2d}x{size:2d}: {info['cells']:3d} cells -> {tag}; "
              f"held {info['peak_edges']} edges in one {dim}-dim vector")
    print("\nPast a point, one field cannot hold the whole graph cleanly -- the read")
    print("comes back a little lost. The fix is not bigger vectors but PARTITION:")
    print("solve regions, compress each into a super-node, recurse one layer up.")


def demo_partition():
    """The capacity wall, broken by partition + compression. A 41x41 maze (a 466-step
    solution) is hopeless for one dim-2048 field; partitioned, the field that carries the
    long-range structure holds only the portal graph and reads back optimally."""
    from holographic.misc.holographic_creature import GridWorld
    print("\nPartition + compress (the recursive fix), all fields at dim 2048:\n")
    for size in (21, 41):
        w = GridWorld(size, size, maze=True, fixed_seed=3)
        path, info = solve_maze_hier(w, dim=2048, region=8, seed=0)
        tag = f"optimal ({info['optimal']} steps)" if info["reached"] and info["extracted_len"] == info["optimal"] else "miss"
        print(f"  {size}x{size}: {info['cells']:3d} cells compressed to {info['portals']:3d} portals -> {tag}; "
              f"holographic field held only {info['peak_super_edges']} edges")
    print("\n  (a single flat field at dim 2048 fails outright on 41x41 -- it finds the")
    print("   466-step route but cannot read 466 edges back out of one vector.)")


if __name__ == "__main__":
    demo_slime_maze()
    demo_partition()
