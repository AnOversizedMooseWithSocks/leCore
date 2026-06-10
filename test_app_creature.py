"""The Labyrinth UI demo is the slime-mold colony solving a BRAIDED 16x16 maze: loops mean
many routes out, and the colony must thin its tubes to the SHORTEST. This pins that
contract -- the maze really has loops, the colony discovers (no precomputed path, trails
only on reached tiles), and on the demo layout the emergent route is the true optimum."""


def _free_and_loops(after):
    W, H = after["w"], after["h"]
    walls = {tuple(c) for c in after["walls"]}
    free = {(x, y) for x in range(W) for y in range(H) if (x, y) not in walls}
    edges = sum(((x + dx, y + dy) in free) for (x, y) in free for dx, dy in ((1, 0), (0, 1)))
    return free, edges - (len(free) - 1)            # independent loops (>0 => braided)


def test_labyrinth_is_braided_slime_discovery_finding_shortest():
    import app
    cfg = app._MODES["maze"]
    assert cfg.get("size") == 16 and cfg.get("braid", 0) > 0
    after = app._slime_rollout(cfg["layout"])
    assert after["w"] == 16 and after["h"] == 16
    assert after["escaped"] is True
    free, loops = _free_and_loops(after)
    assert loops >= 2                               # genuinely braided: multiple routes out
    assert after["steps"] == after["opt"]           # demo layout: colony finds the SHORTEST tube
    assert tuple(after["route"][0]) == (1, 1)       # starts at the entrance
    discovered = {tuple(c) for c in after["explore"]}
    assert all(tuple(c) in discovered for c in after["route"])   # route only uses reached tiles
