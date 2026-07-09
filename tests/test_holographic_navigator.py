"""Tests for holographic_navigator: the creature's brain, learning to navigate
the data tree, should match a wide fixed beam's recall at far lower cost."""

import numpy as np
import pytest

from holographic.agents_and_reasoning.holographic_ai import random_vector
from holographic.agents_and_reasoning.holographic_navigator import DataWorld, CreatureEncoder, HolographicMind, train, evaluate, fixed_beam_curve


def _trained_world(seed=0):
    rng = np.random.default_rng(seed)
    items = np.stack([random_vector(128, rng) for _ in range(600)])
    world = DataWorld(items, leaf_size=32, seed=seed, max_regions=12, noise=0.5)
    enc = CreatureEncoder(256, seed=1)
    mind = HolographicMind(256, DataWorld.ACTIONS, k=12, epsilon=0.3,
                           novelty_bonus=0.1, memory_cap=4000, seed=3)
    train(world, enc, mind, queries=2500)
    return world, enc, mind


def test_dataworld_basic_roundtrip():
    # With a clean (noise-free) cue and enough effort, the navigator's frontier
    # should contain the exact item, so committing on it is correct.
    rng = np.random.default_rng(1)
    items = np.stack([random_vector(64, rng) for _ in range(200)])
    world = DataWorld(items, leaf_size=16, seed=0, max_regions=8, noise=0.0)
    world.reset(rng)
    # exhaust the frontier, then arrive -- best item should be the true NN
    done = False
    while not done:
        _, _, _, done = world.step("keep_moving")
    assert world.correct()


def test_navigator_matches_recall_at_lower_cost():
    world, enc, mind = _trained_world(seed=0)
    recall, comps = evaluate(world, enc, mind, queries=300)
    base = fixed_beam_curve(world, beams=(1, 2, 4, 8, 12), queries=300)
    widest = base[-1]                       # the most thorough fixed beam
    # The navigator should be accurate...
    assert recall >= 0.80
    # ...while spending markedly fewer comparisons than the widest fixed beam
    # that it is competitive with.
    assert comps < 0.6 * widest["comparisons"]
    # And it should have learned a compact policy (a handful of prototypes).
    assert mind.prototype_count() < 400


def test_reflex_habits_help_on_skew_and_dont_hurt_on_uniform():
    from holographic.agents_and_reasoning.holographic_navigator import Navigator, _zipf_workload
    world, enc, mind = _trained_world(seed=0)
    items = world.items

    def run(workload, use_reflex):
        nav = Navigator(world, enc, mind, hot_size=32)
        r = np.random.default_rng(123)
        ok = comps = 0
        for i in workload:
            q = items[i] + world.noise * random_vector(world.dim, r)
            q = q / np.linalg.norm(q)
            truth = int((items @ q).argmax())
            pred, c, _ = (nav.find(q) if use_reflex
                          else world.search(q, enc, mind))
            ok += (pred == truth); comps += c
        return ok / len(workload), comps / len(workload)

    n = len(items)
    skew = _zipf_workload(n, 2500, 1.3, seed=5)
    s_recall, s_comps = run(skew, False)
    r_recall, r_comps = run(skew, True)
    # On a skewed stream the habits should clearly cut cost without losing recall.
    assert r_comps < 0.8 * s_comps
    assert r_recall >= s_recall - 0.02

    uni = _zipf_workload(n, 2500, 0.0, seed=6)
    _, su_comps = run(uni, False)
    _, ru_comps = run(uni, True)
    # On an unpredictable stream the flux guard keeps it from costing much more.
    assert ru_comps < 1.3 * su_comps


# ======================================================================================================
# The faculty, and the claim it rests on: an adaptive budget beats a fixed beam on BOTH readings.
# ======================================================================================================
def _items(n=600, dim=256, seed=0):
    from holographic.agents_and_reasoning.holographic_ai import random_vector
    rng = np.random.default_rng(seed)
    return np.stack([random_vector(dim, rng) for _ in range(n)])


def test_navigator_faculty_trains_finds_and_guards():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    with pytest.raises(ValueError):
        m.navigator_find(np.zeros(256))                       # training is an explicit step, not a lazy one

    items = _items()
    world = m.train_navigator(items, queries=800, seed=0)
    assert world["items"] == 600 and world["regions"] > 1 and world["depth"] > 1

    rng = np.random.default_rng(7)
    from holographic.agents_and_reasoning.holographic_ai import random_vector
    cue = items[7] + 0.5 * random_vector(256, rng)
    cue /= np.linalg.norm(cue)
    hit = m.navigator_find(cue)
    assert hit["index"] == 7 and hit["comparisons"] > 0 and isinstance(hit["trace"], list)
    assert m.navigator_find(cue, explain=True)["trace"]        # explain returns a trace


def test_the_learned_budget_beats_a_fixed_beam_on_both_readings():
    """The claim the module is built on, checked against ITS OWN baseline -- the tree's fixed-beam curve, which is
    the strongest honest baseline in the original space, not a strawman.

    (1) No fixed beam matches the navigator's recall for fewer comparisons.
    (2) At the navigator's comparison budget, every fixed beam has strictly worse recall.
    Quoting only one of those would be cherry-picking the axis that flatters it."""
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    m.train_navigator(_items(), queries=800, seed=0)
    b = m.navigator_benchmark(queries=120)

    rec, comps = b["recall"], b["comparisons"]
    assert rec > 0.9, rec

    # (1) any fixed beam reaching this recall costs strictly more
    matching = [r for r in b["fixed_beams"] if r["recall"] >= rec]
    assert matching, "no fixed beam reaches the navigator's recall at all"
    assert min(r["comparisons"] for r in matching) > 1.5 * comps

    # (2) at the navigator's budget, every fixed beam is strictly worse
    affordable = [r for r in b["fixed_beams"] if r["comparisons"] <= comps]
    assert affordable and max(r["recall"] for r in affordable) < rec - 0.05


def test_navigator_survives_a_real_http_invoke():
    """State lives on the mind and every method speaks plain data, so a long-lived service trains once and searches
    across /invoke calls -- no stateless twin needed."""
    import json
    import threading
    import urllib.request
    from http.server import HTTPServer

    import holographic_service as svc_mod
    from holographic.misc.holographic_unified import UnifiedMind

    mind = UnifiedMind(dim=64, seed=0)
    items = _items(n=300)
    svc = svc_mod.Service(mind=mind)
    httpd = HTTPServer(("127.0.0.1", 0), svc_mod.make_handler(svc))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % httpd.server_address[1]

    def invoke(name, args):
        body = json.dumps({"name": name, "args": args}).encode()
        req = urllib.request.Request(base + "/invoke", data=body, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())

    try:
        r = invoke("train_navigator", {"items": items.tolist(), "queries": 300, "seed": 0})
        assert r["ok"] and r["result"]["items"] == 300
        cue = (items[3] / np.linalg.norm(items[3])).tolist()
        r = invoke("navigator_find", {"cue": cue})             # the trained agent persisted between calls
        assert r["ok"] and r["result"]["index"] == 3
    finally:
        httpd.shutdown()
        httpd.server_close()
