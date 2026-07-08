"""Slime-mold pathfinding over a holographic field: it traverses mazes far past the
reactive brain's ceiling, its edges are genuinely directed (a random-permutation role
tag, since neither bind's commutativity nor a cyclic shift gives a one-way edge), and
the recursive partition keeps the holographic field bounded on mazes a flat field cannot
hold."""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import unbind, cosine
from holographic.misc.holographic_creature import GridWorld
from holographic.simulation_and_physics.holographic_slime import solve_maze, solve_maze_hier, SlimeGraph, _edge, _neighbours


def test_edges_are_directed_not_symmetric():
    # bind is commutative and a shift commutes with convolution, so neither makes a
    # one-way edge. A random-permutation role tag does: u->v is recoverable from u
    # but not from v, or a walker wanders back the way it came.
    g = SlimeGraph(dim=2048, seed=0)
    a, b = g.voc.get("a"), g.voc.get("b")
    e = _edge(a, b, g.role)
    forward = unbind(e, a)[g.unrole]            # recover the target from the source
    backward = unbind(e, b)[g.unrole]           # try to recover the source from the target
    assert cosine(forward, b) > 0.5
    assert cosine(forward, b) > cosine(backward, a) + 0.3


def test_solves_small_maze_optimally():
    w = GridWorld(9, 9, maze=True, fixed_seed=3)   # 24-step maze the reactive brain barely managed
    _, info = solve_maze(w, dim=2048, ants=24, rounds=50, seed=0)
    assert info["reached"]
    assert info["extracted_len"] == info["optimal"]


def test_scales_well_past_reactive_ceiling():
    w = GridWorld(13, 13, maze=True, fixed_seed=7)  # ~48 steps; reactive brain scored 0 here
    _, info = solve_maze(w, dim=2048, ants=24, rounds=50, seed=0)
    assert info["reached"]
    assert info["extracted_len"] <= info["optimal"] + 4


def test_partition_solves_with_a_bounded_field():
    # The capacity-wall fix: partition + compress, so the holographic field stays small.
    w = GridWorld(21, 21, maze=True, fixed_seed=3)
    path, info = solve_maze_hier(w, dim=2048, region=8, seed=0)
    assert info["reached"]
    assert info["extracted_len"] == info["optimal"]          # perfect maze -> unique path
    assert info["peak_super_edges"] < info["cells"]          # field held less than the whole maze
    nb = {c: set(_neighbours(w, c)) for c in w._free_cells()}
    assert all(b in nb[a] for a, b in zip(path, path[1:]))   # a real, contiguous path
    assert path[0] == (w.cx, w.cy) and path[-1] == (w.fx, w.fy)
