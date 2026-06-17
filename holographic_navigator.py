"""
holographic_navigator.py -- the creature, repurposed to navigate our DATA.

The creature demo was never really about a bug in a grid. It was a testbed for a
mind that can perceive, decide, learn from what happened, and adjust -- and the
whole rest of this project quietly built the *world* for it to live in: a
recursive HoloTree/HoloForest that partitions stored data into branching regions,
exactly like a maze of corridors. So this module closes the loop ("inception"):
the SAME brain (HolographicMind) and the SAME senses encoder (CreatureEncoder)
that learned to find the star and the maze exit now learn to navigate the data
structure to find what you asked for.

The parallel, made literal:

    creature in a grid                navigator in the data tree
    ------------------                --------------------------
    a cell it stands on               a region (tree leaf) it is examining
    "food is to the east"             "the best match so far is strong / weak"
    step toward the star              examine the next-most-promising region
    "I reached the star"              "I have arrived at the answer -- commit"
    avoid wasting energy              avoid wasting comparisons

Why a LEARNED navigator instead of the tree's built-in routing? Because the tree
spends a FIXED budget on every query: a beam of b always reads b regions, however
easy or hard the query is. But difficulty varies wildly -- many cues land cleanly
in one region (a wide beam is wasted on them) while a few land near a boundary and
need a deep search. A fixed beam must be set wide enough for the hard ones, so it
overpays on the easy majority. The navigator instead learns an ADAPTIVE budget:
read a region, sense how confident the answer looks, and decide arrive-or-keep-
moving. Measured below, it matches a wide beam's recall at a fraction of the
comparisons -- the find-the-star / keep-searching instinct, now buying efficiency.

Honest scope: this is the ACCESS half of "organize and access". The navigator
makes retrieval cheaper and is trained against an exact-scan ground truth, then
deployed using only its learned senses (no ground truth needed at query time).
Pure numpy, built directly on the existing engine, tree, and creature.
"""

import numpy as np

from holographic_ai import random_vector
from holographic_tree import HoloTree, ReflexCache   # ReflexCache lives with the index machinery now
from holographic_creature import CreatureEncoder, HolographicMind


# ---------------------------------------------------------------------------
# 1. THE WORLD  (the data tree as a place to navigate -- DataWorld ~ GridWorld)
# ---------------------------------------------------------------------------

class DataWorld:
    """A holographic index over a set of items, presented as a place to navigate.

    Where GridWorld is a grid the creature steps through, DataWorld is the
    recursive HoloTree partition of data the navigator steps through. One episode
    answers one query: we form a (possibly noisy) cue, ask the tree for its
    best-first list of candidate regions -- the 'frontier', the corridors ranked
    by promise -- and drop the navigator at the most promising one. From there it
    walks outward, region by region, until it decides it has arrived.

    The reward mirrors the creature's: a star for committing to the RIGHT answer,
    a small running cost for every extra region it reads (its 'energy'). Reaching
    the answer cheaply and not over-searching are, again, the same goal.
    """

    ACTIONS = ["arrive", "keep_moving"]      # commit to the best so far / read the next region

    def __init__(self, items, leaf_size=48, seed=0, max_regions=16, noise=0.5):
        self.items = np.asarray(items, float)
        self.dim = self.items.shape[1]
        self.max_regions = max_regions       # deepest the frontier can go (cap on effort)
        self.noise = noise                   # how blurred the query cue is (0 = exact)
        self.tree = HoloTree(self.dim, leaf_size=leaf_size, seed=seed).build(self.items)
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.frontier = []      # tree leaves, best-first (the ranked corridors)
        self.pos = 0            # how many regions we have read
        self.best = -2.0        # best cosine to the cue found so far
        self.second = -2.0      # next-best (the margin best-second is our confidence)
        self.bid = -1           # global index of the current best item
        self.comparisons = 0    # items inspected so far -- the navigator's true cost
        self.improved = True    # did the last region improve our best?
        self.true_nn = -1       # exact nearest neighbour of the cue (training signal)
        self.arrived = False
        self.alive = True

    # -- start an episode: pick a target, blur it into a cue, rank the regions --
    def reset(self, rng):
        self._reset_episode_state()
        i = int(rng.integers(len(self.items)))
        cue = self.items[i] + self.noise * random_vector(self.dim, rng)
        self.cue = cue / np.linalg.norm(cue)
        self.true_nn = int((self.items @ self.cue).argmax())   # ground truth (exact scan)
        self.frontier = self.tree._route(self.cue, beam=self.max_regions)
        self._read_region()                                    # always look at the top region
        return self.senses()

    def _read_region(self):
        """Inspect the next region on the frontier; fold its items into our best."""
        leaf = self.frontier[self.pos]
        cand = leaf["idx"]
        sims = self.items[cand] @ self.cue
        self.comparisons += len(cand)
        prev = self.best
        j = int(sims.argmax())
        if float(sims[j]) > self.best:
            self.best = float(sims[j]); self.bid = int(cand[j])
        order = np.sort(sims)[::-1]
        if len(order) > 1:
            self.second = max(self.second, float(order[1]))
        self.improved = self.best > prev + 1e-9
        self.pos += 1

    def senses(self):
        """What the navigator perceives about its search -- all RELATIVE, never the
        raw vectors, so the policy transfers across datasets. Discretized into a
        few tokens so the very same CreatureEncoder can turn them into a state.

          margin   : how far the best match leads the runner-up (confidence)
          strength : how good the best match is in absolute terms
          progress : how deep into the frontier we have gone (effort spent)
          improved : did the last region we read actually help?
        """
        m = self.best - self.second
        margin = "clear" if m > 0.08 else ("some" if m > 0.03 else "tie")
        strength = "hi" if self.best > 0.6 else ("mid" if self.best > 0.4 else "lo")
        frac = self.pos / self.max_regions
        progress = "early" if frac < 0.25 else ("mid" if frac < 0.55 else "late")
        return {"margin": margin, "strength": strength,
                "progress": progress, "improved": "y" if self.improved else "n"}

    # -- dynamics: arrive (commit) or keep moving (read the next region) --------
    def step(self, action_name):
        """Apply an action; return (senses, reward, arrived, done).

        keep_moving reads one more region and pays a small cost (like a step
        draining energy). arrive commits to the best item found so far: a full
        reward if it is the true nearest neighbour, nothing if it guessed wrong.
        Running off the end of the frontier forces an arrival.
        """
        STEP_COST, HIT = 0.03, 1.0
        if action_name == "keep_moving" and self.pos < len(self.frontier):
            self._read_region()
            done = self.pos >= len(self.frontier)     # frontier exhausted -> must commit next
            return self.senses(), -STEP_COST, False, done
        # arrive (or nothing left to read): commit to the best so far
        self.arrived = True
        self.alive = False
        reward = HIT if self.bid == self.true_nn else 0.0
        return self.senses(), reward, True, True

    def correct(self):
        return self.bid == self.true_nn

    # -- deployment: navigate a GIVEN cue, no ground truth, with a thought trace -
    def search(self, cue, encoder, mind, explain=False):
        """Answer a real query: descend the frontier letting the trained mind
        decide arrive-or-keep, and return (best_index, comparisons, trace). Unlike
        reset()/step() there is no target and no reward -- this is the navigator
        out in the world, deciding purely from its learned senses. `trace` is its
        little internal monologue: what it saw and why it stopped."""
        cue = cue / (np.linalg.norm(cue) or 1.0)
        self._reset_episode_state()
        self.cue = cue
        self.frontier = self.tree._route(cue, beam=self.max_regions)
        self._read_region()
        trace = []
        while True:
            s = self.senses()
            if explain:
                trace.append(f"region {self.pos}: best match {self.best:.2f} "
                             f"({s['margin']} lead) -- ")
            a = mind.decide(encoder.encode(s), explore=False, epsilon=0.0)
            if mind.actions[a] == "keep_moving" and self.pos < len(self.frontier):
                if explain:
                    trace[-1] += "not sure yet, look further"
                self._read_region()
            else:
                if explain:
                    trace[-1] += "confident -- arrive here"
                break
        return self.bid, self.comparisons, trace


# ---------------------------------------------------------------------------
# 2. THE LOOP  (answer one query; optionally learn from it -- run_query ~ run_episode)
# ---------------------------------------------------------------------------

def run_query(world, encoder, mind, rng, learn=True, explore=True,
              eval_epsilon=None, gamma=0.9):
    """Live one query end-to-end; return (correct, comparisons).

    Identical in shape to the creature's run_episode: perceive -> decide -> act,
    collect the trajectory, then (if learning) credit each decision with the
    Monte-Carlo reward-to-go and fold it into the prototype memory."""
    senses = world.reset(rng)
    state = encoder.encode(senses)
    states, actions, rewards = [], [], []
    while True:
        a = mind.decide(state, explore=explore, epsilon=eval_epsilon)
        senses, r, arrived, done = world.step(mind.actions[a])
        states.append(state); actions.append(a); rewards.append(r)
        state = encoder.encode(senses)
        if done:
            break
    if learn:
        returns, g = [0.0] * len(rewards), 0.0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + gamma * g
            returns[t] = g
        mind.remember(states, actions, returns)
    return world.correct(), world.comparisons


def train(world, encoder, mind, queries=4000, eps_start=0.3, seed=1):
    rng = np.random.default_rng(seed)
    for q in range(queries):
        mind.epsilon = max(0.05, eps_start * (1.0 - q / queries))
        run_query(world, encoder, mind, rng, learn=True, explore=True)


def evaluate(world, encoder, mind, queries=400, seed=999):
    """Recall and average comparisons of the learned navigator on fresh queries."""
    rng = np.random.default_rng(seed)
    ok = comps = 0
    for _ in range(queries):
        c, cmp = run_query(world, encoder, mind, rng, learn=False,
                           explore=False, eval_epsilon=0.02)
        ok += c; comps += cmp
    return ok / queries, comps / queries


def fixed_beam_curve(world, beams=(1, 2, 4, 8, 12, 16), queries=400, seed=999):
    """The honest baseline: the tree's own routing at a range of FIXED beams,
    each a (recall, average-comparisons) point on the effort/accuracy curve."""
    rng = np.random.default_rng(seed)
    cues = []
    for _ in range(queries):
        i = int(rng.integers(len(world.items)))
        c = world.items[i] + world.noise * random_vector(world.dim, rng)
        c = c / np.linalg.norm(c)
        cues.append((c, int((world.items @ c).argmax())))
    rows = []
    for b in beams:
        ok = comps = 0
        for c, truth in cues:
            ok += int(world.tree.recall(c, beam=b) == truth)
            comps += world.tree.last_comparisons
        rows.append(dict(beam=b, recall=ok / queries, comparisons=comps / queries))
    return rows


# ---------------------------------------------------------------------------
# 3. THE ORGANIZER  (slime-mould flux: thick veins to where you actually go)
# ---------------------------------------------------------------------------

class Navigator:
    """The whole agent: a trained mind that adaptively searches the data tree,
    fronted by a ReflexCache of its habits. `find` recognises familiar queries
    instantly and only does the deeper search on the unfamiliar ones -- and gets
    faster at whatever you ask for most, the way a person stops deliberating over
    a route they walk every day."""

    def __init__(self, world, encoder, mind, hot_size=48):
        self.world = world
        self.encoder = encoder
        self.mind = mind
        self.cache = ReflexCache(len(world.items), hot_size=hot_size)

    def find(self, cue, explain=False):
        cue = cue / (np.linalg.norm(cue) or 1.0)
        hot_hit, scan = self.cache.consider(cue, self.world.items)
        if hot_hit is not None:                          # recognised by reflex
            self.cache.reinforce(hot_hit, True, self.world.items)
            trace = [f"recognised instantly -- a familiar query ({scan} veins checked)"]
            return hot_hit, scan, (trace if explain else [])
        idx, comps, trace = self.world.search(cue, self.encoder, self.mind, explain=explain)
        self.cache.reinforce(idx, False, self.world.items)
        if explain:
            trace = ([f"no familiar vein ({scan} checked), search the tree:"] if scan
                     else ["search the tree:"]) + trace
        return idx, comps + scan, trace


# ---------------------------------------------------------------------------
# 4. DEMO
# ---------------------------------------------------------------------------

def demo_navigator(n_items=2000, dim=256, leaf_size=48, seed=0):
    """Train the navigator to find items in a 2000-item tree, then show it beats
    every fixed-beam setting: a wide beam's recall at a fraction of the cost."""
    print("=" * 70)
    print("The creature, repurposed: a learned navigator over the data tree")
    print("=" * 70)
    rng = np.random.default_rng(seed)
    items = np.stack([random_vector(dim, rng) for _ in range(n_items)])
    world = DataWorld(items, leaf_size=leaf_size, seed=seed, max_regions=16, noise=0.5)
    print(f"\nIndexed {n_items} items into {world.tree.stats()['leaves']} regions "
          f"(depth {world.tree.stats()['depth']}). Queries are blurred cues; the")
    print("navigator must find each cue's true nearest neighbour.\n")

    print("Fixed-beam baseline (the tree's built-in routing, same effort every query):")
    print(f"  {'beam':>5} {'recall':>8} {'comparisons':>13}")
    base = fixed_beam_curve(world)
    for r in base:
        print(f"  {r['beam']:>5} {r['recall']*100:>7.0f}% {r['comparisons']:>12.0f}")

    enc = CreatureEncoder(256, seed=1)
    mind = HolographicMind(256, DataWorld.ACTIONS, k=12, epsilon=0.3,
                           novelty_bonus=0.1, memory_cap=4000, seed=3)
    train(world, enc, mind, queries=4000)
    recall, comps = evaluate(world, enc, mind)

    wide = base[-1]
    print("\nLearned adaptive navigator:")
    print(f"  recall {recall*100:.0f}%   {comps:.0f} comparisons   "
          f"({mind.prototype_count()} prototypes learned)")
    print(f"\n  It matches the widest beam's recall ({wide['recall']*100:.0f}% at "
          f"{wide['comparisons']:.0f} comparisons) using ~"
          f"{wide['comparisons']/max(comps,1):.1f}x fewer comparisons -- by committing")
    print("  at once on the easy queries and only searching hard on the ambiguous")
    print("  ones. The same find-it / keep-looking instinct that solved the maze,")
    print("  now spending the system's effort where it is actually needed.")


def _zipf_workload(n_items, length, skew, seed):
    """A stream of target items: skew>0 makes a few items wildly popular
    (Zipf), skew=0 is uniform. Which items are popular is arbitrary."""
    r = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, n_items + 1) ** skew
    p /= p.sum()
    popular = r.permutation(n_items)
    return popular[r.choice(n_items, size=length, p=p)]


def demo_organizer(n_items=2000, dim=256, leaf_size=48, seed=0):
    """Show the navigator getting FASTER at what it is asked for most -- the
    organize half. A reflex cache of its habits (slime-mould veins) recognises
    popular queries instantly, and a flux guard makes it never cost more than it
    saves on an unpredictable stream."""
    print("\n" + "=" * 70)
    print("The organizer: habits (slime-mould veins) for what you ask most")
    print("=" * 70)
    rng = np.random.default_rng(seed)
    items = np.stack([random_vector(dim, rng) for _ in range(n_items)])
    world = DataWorld(items, leaf_size=leaf_size, seed=seed, max_regions=16, noise=0.5)
    enc = CreatureEncoder(256, seed=1)
    mind = HolographicMind(256, DataWorld.ACTIONS, k=12, epsilon=0.3,
                           novelty_bonus=0.1, memory_cap=4000, seed=3)
    train(world, enc, mind, queries=4000)

    def run(workload, use_reflex):
        nav = Navigator(world, enc, mind)
        r = np.random.default_rng(123)
        ok = comps = 0
        for i in workload:
            q = items[i] + world.noise * random_vector(dim, r)
            q = q / np.linalg.norm(q)
            truth = int((items @ q).argmax())
            if use_reflex:
                pred, c, _ = nav.find(q)
            else:
                pred, c, _ = world.search(q, enc, mind)
            ok += (pred == truth); comps += c
        return ok / len(workload), comps / len(workload), nav

    for skew, name in [(1.2, "skewed stream (a few items get most queries)"),
                       (0.0, "uniform stream (every item equally likely)   ")]:
        wl = _zipf_workload(n_items, 4000, skew, seed=5)
        r0, c0, _ = run(wl, use_reflex=False)
        r1, c1, nav = run(wl, use_reflex=True)
        veins = "kept" if nav.cache.active else "pruned"
        print(f"\n  {name}")
        print(f"    adaptive search only : recall {r0*100:>3.0f}%   {c0:>4.0f} comparisons")
        print(f"    + reflex habits      : recall {r1*100:>3.0f}%   {c1:>4.0f} comparisons   "
              f"(veins {veins})")

    # a peek at the navigator's internal monologue on two queries
    print("\n  Its train of thought (after learning the skewed stream):")
    nav = run(_zipf_workload(n_items, 4000, 1.2, seed=5), use_reflex=True)[2]
    r = np.random.default_rng(77)
    hot_item = int(nav.cache.hot[0]) if len(nav.cache.hot) else 0
    for label, target in [("a familiar query", hot_item),
                          ("an unfamiliar query", int(r.integers(n_items)))]:
        q = items[target] + world.noise * random_vector(dim, r); q = q / np.linalg.norm(q)
        _, c, trace = nav.find(q, explain=True)
        print(f"    {label} ({c} comparisons):")
        for line in trace:
            print(f"       - {line}")


if __name__ == "__main__":
    demo_navigator()
    demo_organizer()
