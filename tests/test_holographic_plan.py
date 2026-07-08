"""CI wrapper for corridor planning (holographic_plan): bake a short route on the directed substrate, run
it cheap, re-anchor when the throughput gate trips. The selftest proves an at-cap corridor decodes to the
full direction sequence, an over-cap corridor reports only its reliable prefix (no overclaiming), and
replan_needed fires exactly on exhaustion / low throughput / a blocked tile. plan_route then chains
cap-sized corridors to return a whole arbitrarily-long route in one call -- past the single-structure cap."""
import numpy as np
from holographic.scene_and_pipeline.holographic_plan import _selftest, plan, plan_route, chunk_route, RouteIndex, dedup_chunks


def test_plan_selftest():
    _selftest()


def _line(N, dim, seed=2):
    """A straight corridor of N distinct tiles plus a field_step that walks it and an index-labelling action."""
    rng = np.random.default_rng(seed)
    tiles = rng.standard_normal((N, dim)); tiles /= np.linalg.norm(tiles, axis=1, keepdims=True)

    def field_step(cur):
        i = int(np.argmax(tiles @ (cur / (np.linalg.norm(cur) + 1e-12))))
        return tiles[i + 1] if i + 1 < N else None

    def action_of(a, b):
        return int(np.argmax(tiles @ (b / (np.linalg.norm(b) + 1e-12))))

    return tiles, field_step, action_of


def test_plan_route_chains_past_the_single_structure_cap():
    # A 45-tile route is 3x the single-corridor cap. One plan() over it collapses; plan_route recovers it all.
    tiles, field_step, action_of = _line(45, 512)
    crammed = plan(tiles[0], field_step, max_steps=45, floor=0.12, action_of=action_of)
    route = plan_route(tiles[0], field_step, corridor=14, floor=0.12, action_of=action_of)
    assert len(crammed.actions) < 5                          # overstuffed single structure -> the cliff
    assert route.actions == list(range(1, 45))               # chained corridors -> the FULL route, exact
    assert route.stopped == "field_end"
    assert route.reanchors >= 2                              # it re-anchored at the decision points
    assert route.steps > len(crammed.actions) * 5           # decisively past the single-structure cap


def test_plan_route_respects_max_total_and_is_a_correct_prefix():
    tiles, field_step, action_of = _line(60, 512)
    route = plan_route(tiles[0], field_step, max_total=25, corridor=14, floor=0.12, action_of=action_of)
    assert route.steps <= 25
    assert route.actions == list(range(1, 1 + route.steps))  # what it returns is a correct prefix of the route


def test_plan_route_overlong_corridor_is_a_kept_negative():
    # An over-long corridor overstuffs its OWN leg -- the same cliff, per leg. corridor must stay <= the
    # reliable decode depth (the default 14 does). This is on the record, not hidden.
    tiles, field_step, action_of = _line(45, 512)
    bad = plan_route(tiles[0], field_step, corridor=30, floor=0.12, action_of=action_of)
    good = plan_route(tiles[0], field_step, corridor=14, floor=0.12, action_of=action_of)
    assert good.actions == list(range(1, 45))                # the safe per-leg length recovers the full route
    assert bad.actions != list(range(1, 45))                 # the over-long leg does not -- honestly recorded


def test_chunk_route_replays_an_explicit_sequence_past_the_cap():
    # The EXPLICIT-list twin: a known 200-step sequence (no field_step) replays EXACTLY by chunking into
    # <=14 clean pieces. This is the GPS-route / experiment-protocol case: you already have the sequence.
    tiles, _field_step, action_of = _line(200, 512)
    r = chunk_route(list(tiles), chunk=14, floor=0.12, action_of=action_of)
    assert r.actions == list(range(1, 200))                  # whole sequence, exact, past the single-structure cap
    assert len(r.corridors) <= 200 // 14 + 2                 # LINEAR cost: ~15 chunks, not one impossible structure
    assert all(c.memory.shape == (512,) for c in r.corridors)  # each chunk is ONE compact vector


def test_chunk_route_handles_degenerate_inputs():
    tiles, _f, action_of = _line(10, 512)
    assert chunk_route([tiles[0]], action_of=action_of).steps == 0   # single element -> empty, no crash
    assert chunk_route([], action_of=action_of).steps == 0           # empty -> empty, no crash
    short = chunk_route(list(tiles), chunk=14, floor=0.12, action_of=action_of)
    assert short.actions == list(range(1, 10)) and len(short.corridors) == 1  # fits in one chunk


def test_route_index_locates_sub_linearly_and_exactly():
    # Random access into a long chunked route: every tile is located in the right chunk at the exact position,
    # via a two-level (nearest chunk summary -> nearest tile) search -- far fewer comparisons than a flat scan.
    tiles, _f, action_of = _line(200, 512)
    route = chunk_route(list(tiles), chunk=14, floor=0.12, action_of=action_of)
    idx = RouteIndex(route)
    assert idx.n_chunks == len(route.corridors)

    def true_chunks(t):
        return {k for k, c in enumerate(route.corridors)
                if any(np.array_equal(c.nodes[j], tiles[t]) for j in range(len(c.nodes)))}
    # level 1: every tile lands in a chunk that actually contains it (boundaries overlap, so accept either)
    assert all(idx.locate(tiles[t])[0] in true_chunks(t) for t in range(200))
    # level 2: the located position is exact, and the global step recovers the route index
    for t in (0, 37, 99, 137, 199):
        c, p, g = idx.locate(tiles[t])
        assert np.allclose(idx.chunks[c][p], tiles[t])
        assert g == t                                        # overlap-adjusted global step == the true index
    # sub-linear: comparisons per query are ~(#chunks + chunk_size), not #tiles
    per_query = idx.n_chunks + max(len(c) for c in idx.chunks)
    assert per_query < 200


def test_route_index_empty_route_is_safe():
    assert RouteIndex(chunk_route([], action_of=lambda a, b: 0)).locate(np.zeros(512)) == (-1, -1, -1)


def test_chunking_seams_are_deterministic_and_not_tie_sensitive():
    # Macklin's discipline: the chunking seams (plan_route / chunk_route / RouteIndex) must be bit-deterministic
    # run-to-run and must not resolve a query on a knife-edge tie. Audited clean; this locks it in.
    tiles, field_step, action_of = _line(80, 512)
    a, b = (chunk_route(list(tiles), chunk=14, floor=0.12, seed=0, action_of=action_of) for _ in range(2))
    assert a.actions == b.actions
    assert all(np.array_equal(x.memory, y.memory) for x, y in zip(a.corridors, b.corridors))   # BIT-identical
    pa, pb = (plan_route(tiles[0], field_step, corridor=14, floor=0.12, seed=0, action_of=action_of) for _ in range(2))
    assert pa.actions == pb.actions
    assert all(np.array_equal(x.memory, y.memory) for x, y in zip(pa.corridors, pb.corridors))
    i1, i2 = RouteIndex(a), RouteIndex(a)
    assert np.array_equal(i1._summaries, i2._summaries)                                          # summaries bit-identical
    assert all(i1.locate(tiles[t]) == i2.locate(tiles[t]) for t in range(80))                    # locate deterministic
    # an ambiguous query (between two tiles in different chunks) resolves the SAME way every call (argmax tie-break)
    mid = tiles[10] + tiles[40]; mid /= np.linalg.norm(mid)
    assert len({i1.locate(mid) for _ in range(20)}) == 1


# ---- C1: content-addressed chunk deduplication ----------------------------------------------------

def test_dedup_chunks_saves_at_repetition_ratio_and_rebuilds_exactly():
    # A route revisiting corridors / a program with repeated motifs: only the distinct chunks are stored,
    # references recover the original order and repeats exactly.
    rng = np.random.default_rng(0); dim = 256
    segs = [rng.standard_normal(dim) for _ in range(5)]      # 5 distinct chunk vectors
    pattern = [0, 1, 2, 0, 1, 2, 3, 0, 1, 4]                 # 10 chunks, 5 distinct
    chunks = [segs[p] for p in pattern]
    unique, refs = dedup_chunks(chunks)
    assert len(unique) == 5                                  # the saving == the repetition ratio
    assert refs == pattern                                   # references recover original order/repeats
    rebuilt = [unique[r] for r in refs]
    assert all(np.array_equal(rebuilt[i], chunks[i]) for i in range(len(chunks)))   # exact reconstruction


def test_dedup_chunks_saves_nothing_without_repetition():
    rng = np.random.default_rng(1); dim = 256
    chunks = [rng.standard_normal(dim) for _ in range(8)]    # all distinct
    unique, refs = dedup_chunks(chunks)
    assert len(unique) == 8 and refs == list(range(8))       # honest bound: no repeats -> no saving


def test_routeindex_migration_is_byte_identical_to_the_old_inline_two_level_routing():
    """PARITY: RouteIndex now delegates its two-level routing to StructuredIndex(keying='sequential'). This
    proves the delegation changed NOTHING -- locate() returns the exact same (chunk, pos, global_step) the old
    inline summary-scan produced, for every tile on a real route. Reuse in place of the bespoke index, byte-
    identical."""
    from holographic.agents_and_reasoning.holographic_ai import bundle
    tiles, field_step, action_of = _line(80, 512)
    route = chunk_route(list(tiles), chunk=14, floor=0.12, seed=0, action_of=action_of)

    # the ORIGINAL RouteIndex computation, inline and independent
    chunks = [np.asarray(c.nodes, float) for c in route.corridors]
    summaries = []
    for ch in chunks:
        s = bundle(list(ch)); n = np.linalg.norm(s); summaries.append(s / n if n > 0 else s)
    summaries = np.array(summaries) if summaries else np.zeros((0,))
    starts, acc = [], 0
    for ch in chunks:
        starts.append(acc); acc += max(1, len(ch) - 1)

    def old_locate(query):
        if not chunks:
            return (-1, -1, -1)
        q = np.asarray(query, float); nq = np.linalg.norm(q); q = q / nq if nq > 0 else q
        c = int(np.argmax(summaries @ q)); ch = chunks[c]; pos = int(np.argmax(ch @ q))
        return (c, pos, starts[c] + pos)

    idx = RouteIndex(route)
    for t in range(80):
        assert idx.locate(tiles[t]) == old_locate(tiles[t]), f"tile {t} routed differently under migration"
    # summaries bit-identical too (they moved into the shared index)
    assert np.array_equal(idx._summaries, summaries)
