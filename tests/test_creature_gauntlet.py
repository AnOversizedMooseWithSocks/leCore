"""The maze gauntlet -- gamified debugging.

Each scenario here mirrors a challenge the SYSTEM itself faced, translated into
a maze the creature must solve without cheating (egocentric senses only, nothing
global). The principle: if the creature can crack the puzzle, the lesson it took
usually transfers back to the brain -- and the lessons that built this file ran
the other way, system to creature:

  * ALIASING (far-apart corridors look identical) is the maze costume of the
    sub-format problem (code and prose both look like "text").
  * The cure that worked is the system's own DECIDE-ONLY-AT-CHOICES boundary
    lesson (exact scan below the crossover; the gate only where type goes
    blind): a corridor reflex auto-walks forced cells so the brain spends its
    decisions and its credit only at junctions. Per-step framing discounted a
    26-step exit to gamma^26 ~ 0.07 (invisible); junction granularity puts it
    near 0.4 (learnable). The recorded 9x9 ceiling fell 0% -> 100%.
  * The honest CONTROL is pinned too: corridor-following with RANDOM junction
    choices solves easy mazes often (a perfect maze has a small junction graph),
    so the brain must beat that control, not just escape.
  * And a recorded NEGATIVE stays a negative: a decaying bundled action trace
    (compression of history) scored 0% at every decay tried, even on the 7x7
    that exact mem=4 solves at 97% -- permute-by-age ORDER is the information
    that breaks aliasing, and bundling erases it. Compression must preserve the
    distinctions the task needs (the nearest-key-generation lesson again).
"""

import numpy as np

from holographic.misc.holographic_creature import HolographicMind, CreatureEncoder, GridWorld, run_episode, _forced_dir
import pytest

DIM = 256


def _train_maze(size, maze_seed, brain_seed, episodes, mem=4, reflex=True,
                max_steps=90):
    enc = CreatureEncoder(DIM, seed=1)
    mind = HolographicMind(DIM, GridWorld.ACTIONS, k=15, epsilon=0.50,
                           novelty_bonus=0.2, memory_cap=12000, seed=brain_seed)
    world = GridWorld(size, size, maze=True, fixed_seed=maze_seed)
    for ep in range(episodes):
        mind.epsilon = max(0.05, 0.50 * (1.0 - ep / episodes))
        run_episode(world, enc, mind, learn=True, explore=True, mem=mem,
                    corridor_reflex=reflex, max_steps=max_steps)
    return world, enc, mind


def _escape_rate(world, enc, mind, n=20, mem=4, reflex=True, random_policy=False,
                 max_steps=90):
    got = 0
    for _ in range(n):
        run_episode(world, enc, mind, learn=False, explore=random_policy,
                    eval_epsilon=(1.0 if random_policy else 0.05), mem=mem,
                    corridor_reflex=reflex, max_steps=max_steps)
        got += world.escaped
    return got / n


def test_corridor_reflex_only_walks_forced_cells():
    # the reflex must fire ONLY in corridors: two open directions, one behind us
    assert _forced_dir({"wall_N": "yes", "wall_S": "yes"}, "E") == "E"   # E-W corridor
    assert _forced_dir({"wall_N": "yes", "wall_E": "yes"}, "N") == "W"   # a turn
    assert _forced_dir({"wall_N": "yes"}, "E") is None                   # junction (3 open)
    assert _forced_dir({"wall_N": "yes", "wall_E": "yes", "wall_S": "yes"}, "E") is None  # dead end
    assert _forced_dir({"wall_N": "yes", "wall_S": "yes"}, None) is None  # episode start


def test_gauntlet_9x9_aliasing_wall_falls():
    # THE headline: the recorded 9x9 ceiling (0% escapes, far-apart corridors
    # alias) falls when decisions are spent only at junctions. Measured 100% on
    # all three seeds; one seed with a conservative floor keeps the test honest
    # and fast.
    world, enc, mind = _train_maze(9, maze_seed=5, brain_seed=2, episodes=240)
    assert _escape_rate(world, enc, mind) >= 0.8


def test_gauntlet_brain_beats_the_reflex_alone():
    # the honest control, pinned: at 13x13 corridor-following with RANDOM junction
    # choices mostly fails (measured 15%), while the trained brain escapes ~67% --
    # the brain's contribution must stay real, not a side effect of the reflex.
    world, enc, mind = _train_maze(13, maze_seed=5, brain_seed=2, episodes=240)
    trained = _escape_rate(world, enc, mind, n=20)
    fresh = HolographicMind(DIM, GridWorld.ACTIONS, k=15, epsilon=1.0,
                            novelty_bonus=0.2, memory_cap=12000, seed=2)
    random_rate = _escape_rate(world, enc, fresh, n=20, random_policy=True)
    assert trained >= 0.35
    assert trained > random_rate


def test_gauntlet_7x7_floor_holds():
    # the regression floor: the maze the old framing already solved must stay
    # solved under the new one (reflex on, junction-granularity credit)
    world, enc, mind = _train_maze(7, maze_seed=7, brain_seed=2, episodes=200)
    assert _escape_rate(world, enc, mind) >= 0.9


def test_gauntlet_braided_maze_loops():
    # BRAIDS mirror competing valid candidates (multiple routes exist, as in
    # reorganization where several resolutions can all work). Loops also let
    # corridor-following CYCLE, so the brain must add real routing on top.
    # Measured, 3 seeds: baseline mem=4 without the reflex 0%; reflex+brain 100%;
    # reflex+random control 50%. One seed, conservative floors, and the brain
    # must beat the control.
    world = GridWorld(11, 11, maze=True, fixed_seed=5, braid=0.5)
    enc = CreatureEncoder(DIM, seed=1)
    mind = HolographicMind(DIM, GridWorld.ACTIONS, k=15, epsilon=0.50,
                           novelty_bonus=0.2, memory_cap=12000, seed=2)
    for ep in range(240):
        mind.epsilon = max(0.05, 0.50 * (1.0 - ep / 240))
        run_episode(world, enc, mind, learn=True, explore=True, mem=4,
                    corridor_reflex=True, max_steps=110)
    trained = _escape_rate(world, enc, mind, max_steps=110)
    fresh = HolographicMind(DIM, GridWorld.ACTIONS, k=15, epsilon=1.0,
                            novelty_bonus=0.2, memory_cap=12000, seed=2)
    rnd = _escape_rate(world, enc, fresh, random_policy=True, max_steps=110)
    assert trained >= 0.8
    assert trained > rnd


def test_gauntlet_poisoned_fork_yield_and_survive():
    # THE POISONED FORK mirrors confusable classes with ASYMMETRIC cost: one arm
    # of a fork is lethal and looks like the safe one. Two things are pinned:
    # (1) the danger-yielding reflex (automation hands back control at anomalies
    #     -- the flux-guard lesson) keeps a trained creature ALIVE: measured 0%
    #     deaths / 100% escapes vs 7% deaths naive, 3 seeds;
    # (2) the brain's contribution stays huge: the random control with the same
    #     yielding reflex died 88% of the time and escaped 12%.
    world = GridWorld(11, 11, maze=True, fixed_seed=5, braid=0.5, maze_poison=3)
    assert world._route_exists((world.cx, world.cy), (world.fx, world.fy),
                               blocked=world.poison)        # the maze stays honest
    enc = CreatureEncoder(DIM, seed=1)
    mind = HolographicMind(DIM, GridWorld.ACTIONS, k=15, epsilon=0.50,
                           novelty_bonus=0.2, memory_cap=12000, seed=2)
    for ep in range(240):
        mind.epsilon = max(0.05, 0.50 * (1.0 - ep / 240))
        run_episode(world, enc, mind, learn=True, explore=True, mem=4,
                    corridor_reflex=True, max_steps=110)
    esc = died = 0
    for _ in range(20):
        run_episode(world, enc, mind, learn=False, explore=False,
                    eval_epsilon=0.05, mem=4, corridor_reflex=True, max_steps=110)
        esc += world.escaped
        died += (not world.alive and not world.escaped)
    assert esc / 20 >= 0.8
    assert died / 20 <= 0.1


def test_gauntlet_16x16_any_seed():
    # THE 16x16 ROOM, any seed, no map knowledge. Three walls fell to get here,
    # each a system lesson: (1) ENERGY -- optimal paths run 80-108 steps against
    # the then-default battery of 100, so the budget must match the world before
    # intelligence even enters (that finding raised the default to 300, which
    # re-validated at the same worst 95% / mean 99% across 8 maze seeds); (2) the CREDIT HORIZON again, one level up --
    # at gamma=0.9 training is bimodal (runs land at ~100% or collapse to 0%,
    # the brain committing early to a wrong junction policy and greedily cycling
    # it); gamma=0.97 took the failing combinations from 1% to 98% mean and
    # IMPROVED the smaller mazes too; (3) the stray collapses that remained are
    # handled by SPECULATE-MEASURE-ADOPT over whole policies (learn_maze trains a
    # candidate, probes its real escape rate, restarts if incompetent -- the
    # organizer's rule applied to training runs). Validated across 8 maze seeds:
    # worst 95%, mean 99%. Pinned here on the historically nastiest seed (3,
    # whose first candidate collapses and must be rescued by the restart) plus a
    # second seed, against the random-junction control. The honest frontier is
    # recorded too: ZERO-SHOT transfer to never-seen mazes measured 41% vs a 36%
    # control -- the learned junction policy is maze-specific; like a rat, the
    # creature earns each maze by living in it.
    from holographic.misc.holographic_creature import learn_maze

    def wf(ms):
        return lambda: GridWorld(16, 16, maze=True, fixed_seed=ms)   # default 300

    for ms in (3, 7):
        enc, mind, _ = learn_maze(wf(ms), seed=2, max_steps=300)
        got = 0
        for _ in range(20):
            w = wf(ms)()
            run_episode(w, enc, mind, learn=False, explore=False, eval_epsilon=0.05,
                        mem=4, corridor_reflex=True, max_steps=300)
            got += w.escaped
        assert got / 20 >= 0.8, f"maze seed {ms} fell below the floor"

    # the brain must beat the reflex-with-random control on the same budget
    fresh = HolographicMind(DIM, GridWorld.ACTIONS, k=15, epsilon=1.0,
                            novelty_bonus=0.2, memory_cap=12000, seed=2)
    rnd = 0
    for _ in range(20):
        w = wf(7)()
        run_episode(w, enc, fresh, learn=False, explore=True, eval_epsilon=1.0,
                    mem=4, corridor_reflex=True, max_steps=500)
        rnd += w.escaped
    assert got / 20 > rnd / 20


@pytest.mark.slow
def test_gauntlet_survival_foraging_poison():
    # SURVIVAL FORAGING, fair and harsh: lives run until DEATH, score = stars per
    # life, and the baselines use the creature's exact senses. This framing found
    # two real problems the 50-step caps masked: (1) compounding risk -- the
    # capped-trained brain died on poison in 67-73% of full lives (~0.6%/step
    # residual risk, invisible in short tests); fixed by the danger reflex
    # (lethal moves vetoed below the brain via decide's `among` -- the routing
    # lesson for actions; asymmetric costs make irreversible mistakes reflex
    # business, not learned preference). (2) DITHERING -- a memoryless forager
    # spent 60% of its steps oscillating, starving at 28 stars; mem=3 cuts it to
    # 10% and lifts it to ~121 (89% of the danger-aware greedy reflex's 136, the
    # same ratio as the clean world). Floors here are conservative; the harsh
    # bar is the NAIVE greedy chaser, which dies on poison almost immediately
    # (8.2 stars) -- the brain must crush it on both stars and survival.
    #
    # SLOW (slowest-tests pass): a survival-foraging test whose stars-per-life and death-rate contrast is a product
    # of full-length training-then-lives-until-death (~19 s). Like its wall-reflex sibling, the margin is a
    # consolidation effect that shrinking the run would make fragile, so it is marked slow rather than trimmed.
    import numpy as np
    rng = np.random.default_rng(9)

    def naive_greedy_life(world):
        senses = world.reset()
        while world.age < 10000:
            toward = []
            if senses.get("food_x") == "east":  toward.append("E")
            if senses.get("food_x") == "west":  toward.append("W")
            if senses.get("food_y") == "south": toward.append("S")
            if senses.get("food_y") == "north": toward.append("N")
            cand = ([d for d in toward if f"wall_{d}" not in senses]
                    or [d for d in GridWorld.ACTIONS if f"wall_{d}" not in senses])
            senses, r, ate, done = world.step(cand[int(rng.integers(len(cand)))])
            if done:
                break
        return world.stars, ((world.cx, world.cy) in world.poison)

    kw = dict(width=7, height=7, n_poison=2, seed=3)
    naive_stars, naive_deaths = zip(*(naive_greedy_life(GridWorld(**kw))
                                      for _ in range(15)))

    enc = CreatureEncoder(DIM, seed=1)
    mind = HolographicMind(DIM, GridWorld.ACTIONS, k=15, epsilon=0.45,
                           novelty_bonus=0.2, memory_cap=12000, seed=2)
    world = GridWorld(**kw)
    for ep in range(180):
        mind.epsilon = max(0.05, 0.45 * (1.0 - ep / 180))
        run_episode(world, enc, mind, learn=True, explore=True, mem=3,
                    max_steps=100, danger_reflex=True)
    stars, deaths = [], 0
    for _ in range(10):
        world = GridWorld(**kw)
        run_episode(world, enc, mind, learn=False, explore=False, eval_epsilon=0.05,
                    mem=3, max_steps=None, danger_reflex=True)
        stars.append(world.stars)
        deaths += ((world.cx, world.cy) in world.poison)

    assert deaths == 0                                   # the veto makes death impossible
    assert np.mean(stars) >= 60                          # measured ~121; harsh floor
    assert np.mean(stars) > np.mean(naive_stars) * 3     # crush the naive chaser
    assert np.mean(naive_deaths) > 0.5                   # ...which really does die


@pytest.mark.slow
def test_gauntlet_wall_reflex_solves_the_cluttered_open_problem():
    # THE INTROSPECTION-NAMED FIX: describe() on a caught dither showed the
    # brain choosing E at value +0.43 while its own senses said wall_E=yes --
    # it was valuing moves into walls it could see. Vetoing wall moves through
    # the same `among` mechanism as danger solved the recorded open problem:
    # stars 5.1 -> 19.8 (the danger-aware reflex ceiling is ~20), dither
    # 79% -> 43%, deaths 0%, three seeds. Pinned: one seed, both conditions,
    # the fix must roughly double the stars and stay deathless.
    #
    # SLOW, irreducibly (slowest-tests pass): the fixed-condition star count only clears its floor of 12 after the
    # full 180-episode training. Measured: at 120 episodes it collapses to 5.9, at 100 to 10.9 -- both below the
    # floor, and non-monotone, because the wall-reflex advantage needs the full run to CONSOLIDATE into the value
    # memory. Shrinking it would make the assertion fragile, so it is marked slow (~38 s) rather than trimmed.
    import numpy as np

    def bench(wall_reflex, n=8):
        enc = CreatureEncoder(DIM, seed=1)
        mind = HolographicMind(DIM, GridWorld.ACTIONS, k=15, epsilon=0.45,
                               novelty_bonus=0.2, memory_cap=12000, seed=2)
        world = GridWorld(7, 7, n_poison=2, n_walls=8, seed=3)
        for ep in range(180):
            mind.epsilon = max(0.05, 0.45 * (1 - ep / 180))
            run_episode(world, enc, mind, learn=True, explore=True, mem=3,
                        max_steps=100, danger_reflex=True, wall_reflex=wall_reflex)
        stars, deaths = [], 0
        for _ in range(n):
            world = GridWorld(7, 7, n_poison=2, n_walls=8, seed=3)
            run_episode(world, enc, mind, learn=False, explore=False,
                        eval_epsilon=0.05, mem=3, max_steps=None,
                        danger_reflex=True, wall_reflex=wall_reflex)
            stars.append(world.stars)
            deaths += ((world.cx, world.cy) in world.poison)
        return float(np.mean(stars)), deaths

    base, d0 = bench(wall_reflex=False)
    fixed, d1 = bench(wall_reflex=True)
    assert d1 == 0
    assert fixed >= 12                                 # measured 19.8; harsh floor
    assert fixed > base * 1.5                          # the fix must really be the fix


def test_worldview_counts_and_names_changes():
    # PERCEPTION AS COMPOSITE: the world's contents as a superposition of
    # type(x)position products. The DIFF of two snapshots is a composite of the
    # changes (appeared positive, vanished negative): round(||diff||^2) COUNTS
    # them and count-driven peeling NAMES them -- no threshold, the diff's own
    # norm says when to stop. Ground truth is the set difference of contents
    # (mutations that cancel are correctly no-change).
    import numpy as np
    from holographic.misc.holographic_creature import GridWorld, WorldView
    rng = np.random.default_rng(0)
    wv = WorldView(dim=2048, width=16, height=16, seed=0)

    def contents(w):
        s = {("exit", (w.fx, w.fy))}
        s |= {("wall", c) for c in w.walls}
        return s

    ok = tot = 0
    for trial in range(10):
        w = GridWorld(width=16, height=16, maze=True, seed=int(rng.integers(100)))
        w.reset()
        before, v1 = contents(w), wv.view(w)
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.5 and w.walls:
                w.walls.discard(list(w.walls)[int(rng.integers(len(w.walls)))])
            else:
                free = [(x, y) for x in range(16) for y in range(16)
                        if (x, y) not in w.walls and (x, y) != (w.fx, w.fy)
                        and (x, y) != (w.cx, w.cy)]
                w.walls.add(free[int(rng.integers(len(free)))])
        after, v2 = contents(w), wv.view(w)
        app, van = wv.changes(v1, v2)
        ok += (sorted(app) == sorted(after - before)
               and sorted(van) == sorted(before - after))
        tot += 1
    assert ok == tot                                  # measured 100%


def test_perception_explains_plan_break():
    # INTEGRATION: two systems on one substrate cross-validate. A wall dropped on
    # the learned route makes replay_plan break at exactly that cell, and
    # WorldView independently NAMES that wall as the change -- perception explains
    # the plan failure. (replay_plan(reset=False) drives the mutated world as-is;
    # a reset would re-carve a different maze since seed != fixed_seed.)
    import io, contextlib
    from holographic.misc.holographic_creature import GridWorld, WorldView, learn_maze, capture_route, replay_plan
    from holographic.misc.holographic_unified import UnifiedMind

    def mk():
        return GridWorld(width=9, height=9, maze=True, seed=5)

    with contextlib.redirect_stdout(io.StringIO()):
        enc, mind, rate = learn_maze(mk, dim=256, episodes=150, mem=2)
        routes = capture_route(mk, enc, mind, mem=2, trials=8)
    m = UnifiedMind(dim=2048, seed=0)
    m.learn_sequences([(r, "route") for r in routes])
    m.discover_sequential()
    canon = m._seq_mem().seqs["route"][1]
    wv = WorldView(dim=2048, width=9, height=9, seed=0)

    agree = tot = 0
    for trial in range(3):
        w = mk(); w.reset()
        v1 = wv.view(w)
        cell = canon[3 + trial * 3]
        r_, c_ = int(cell[1:cell.index('c')]), int(cell[cell.index('c') + 1:])
        target = (c_, r_)
        if target in ((w.fx, w.fy), (w.cx, w.cy)):
            continue
        w.walls.add(target)
        v2 = wv.view(w)
        status, where, _, intended = replay_plan(w, canon, reset=False)
        app, _ = wv.changes(v1, v2)
        agree += (status == "broke" and app == [("wall", target)])
        tot += 1
    assert tot >= 2 and agree == tot
    # control: the unmutated maze still escapes under reset=False
    w = mk(); w.reset()
    assert replay_plan(w, canon, reset=False)[0] == "escaped"


@pytest.mark.slow  # bootstrap-rescue over a starved maze budget; measured ~43-61s, exceeds the 15s per-test budget --
                    # this file already marks its other two irreducibly-slow tests (lines above); this one was
                    # missed and was silently hitting the watchdog on every default run
def test_bootstrap_rescue_cracks_the_starved_maze():
    # THE END-TO-END RESULT, the robust discriminator: on the hard 20x20 seed-11 maze the plain protocol probes
    # 0% (under the decaying-epsilon schedule the loop-attractor policy locks in before luck finds the exit),
    # while the adaptive rescue -- plain candidate starves, starvation summons curiosity+rehearsal at adequate
    # capacity -- takes the SAME maze to a competent probe.
    #
    # PERF (slowest-tests pass): the original ran dim=512/episodes=400/max_steps=800 twice = ~140 s, the single
    # slowest test in the suite. Swept for the cheapest config that PRESERVES THE CONTRAST (plain starves to 0,
    # rescue clears 2/3) and verified it is STABLE, not a fluke: dim=320/episodes=180/max_steps=450 gives
    # plain=0.00 and rescue=1.00 on three consecutive deterministic runs (rescue is well clear of the 2/3 bar,
    # so the margin is real), at ~42 s -- a 3.3x cut with the discriminator intact. The contrast is fragile below
    # this (episodes=200 dropped rescue to 0.50; the rescue needs enough training to CONSOLIDATE), which is why
    # the config is pinned here with the measurement rather than trimmed further.
    import io, contextlib
    from holographic.misc.holographic_creature import GridWorld, learn_maze

    def mk():
        return GridWorld(20, 20, maze=True, fixed_seed=11)

    with contextlib.redirect_stdout(io.StringIO()):
        _, _, plain = learn_maze(mk, dim=320, episodes=180, mem=4, max_steps=450,
                                 k=20, candidates=1, bootstrap=False)
        _, _, rescued = learn_maze(mk, dim=320, episodes=180, mem=4, max_steps=450,
                                   k=20, candidates=4, bootstrap="auto")
    assert plain == 0.0                     # the wall is real
    assert rescued >= 2 / 3                 # the rescue cracks it


def test_learn_maze_bootstrap_flag_modes():
    # The adaptive rescue's API: bootstrap=False is exactly the old plain
    # protocol, bootstrap="auto" (default) runs plain when not starved -- both
    # must train a 9x9 maze to competence (where luck has always sufficed, the
    # rescue stays dormant and behavior is unchanged).
    import io, contextlib
    from holographic.misc.holographic_creature import GridWorld, learn_maze

    def mk():
        return GridWorld(width=9, height=9, maze=True, fixed_seed=5)

    with contextlib.redirect_stdout(io.StringIO()):
        _, _, r_off = learn_maze(mk, dim=256, episodes=150, mem=2, bootstrap=False)
        _, _, r_auto = learn_maze(mk, dim=256, episodes=150, mem=2)  # default auto
    assert r_off >= 2 / 3 and r_auto >= 2 / 3
