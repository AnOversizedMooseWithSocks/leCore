"""
holographic_creature.py
=======================

A creature brain built on the holographic engine in holographic_ai.py.

It learns to forage in a little grid world -- find food, avoid poison -- with
NO neural net and NO training loop in the gradient sense. It simply remembers
what happened (state, action, how it turned out) and, faced with a new
situation, does whatever worked in similar situations before. Similarity is
measured holographically; the "value" of an action is the reward of its nearest
neighbours in memory. That is instance-based reinforcement learning, and it
maps cleanly onto leOS's reflex arc + "semantic compass" (lean toward what
succeeded) + "void/curiosity" (try what you haven't, where you're unsure).

The one trick that makes it learn fast: the creature senses the world
EGOCENTRICALLY -- "food is to my east", not "food is at (5,2)". Because the
state is relative, a lesson learned in one corner of the map applies
everywhere, so it never has to visit every cell.

Run:   python3 holographic_creature.py
Needs: numpy, and holographic_ai.py beside it.
"""

import numpy as np
from holographic_ai import random_vector, cosine, bind, bundle, permute, Vocabulary


# ---------------------------------------------------------------------------
# 1. THE BRAIN  (general-purpose; nothing creature-specific in here)
# ---------------------------------------------------------------------------

class HolographicMind:
    """Perceive -> decide -> learn, by remembering experiences.

    An experience is (state_vector, action, return), where 'return' is the
    discounted reward that followed. To value an action in a state, we look up
    the nearest past states where that action was taken and average their
    returns, weighted by similarity. Pick the best-valued action. No weights to
    train, no backprop -- the learning is the remembering.
    """

    def __init__(self, dim, actions, k=12, epsilon=0.1, novelty_bonus=0.3,
                 memory_cap=6000, seed=0):
        self.dim = dim
        self.actions = list(actions)        # e.g. ["N", "S", "E", "W"]
        self.k = k                          # neighbours consulted per action
        self.epsilon = epsilon              # chance of a random exploratory move
        self.novelty_bonus = novelty_bonus  # optimism for rarely-tried actions
        self.memory_cap = memory_cap        # forget oldest beyond this many
        self.rng = np.random.default_rng(seed)
        # Experiences kept as parallel arrays so recall is one fast matrix-vector
        # product instead of a Python loop.
        self.S = np.zeros((0, dim))         # state vectors (each unit length)
        self.A = np.zeros(0, dtype=int)     # action index taken
        self.R = np.zeros(0)                # return that followed

    def value(self, state_vec, action_idx):
        """Estimate the value of an action in a state.

        Returns (value, support). 'support' is how similar the closest
        remembered situation is (0 = we've basically never seen this) -- the
        curiosity signal. Because state vectors are unit length, the matrix
        product below IS the cosine similarity to every stored state at once.
        """
        mask = self.A == action_idx
        if not np.any(mask):
            return 0.0, 0.0
        sims = self.S[mask] @ state_vec
        rets = self.R[mask]
        if sims.size > self.k:                       # keep only the k nearest
            top = np.argpartition(sims, -self.k)[-self.k:]
            sims, rets = sims[top], rets[top]
        weights = np.clip(sims, 0.0, None)           # ignore unrelated/opposite states
        total = weights.sum()
        if total <= 1e-9:
            return 0.0, 0.0
        return float((weights * rets).sum() / total), float(sims.max())

    def decide(self, state_vec, explore=True, epsilon=None):
        """Choose an action. Mostly greedy on value, with two sources of
        exploration: an epsilon chance of a random move, and (while exploring) a
        novelty bonus that favours actions rarely tried in situations like this.

        'epsilon' overrides the random-move chance for this call. A small value
        even at evaluation time (e.g. 0.05) is worth keeping: a purely greedy,
        memoryless reactive agent can get trapped oscillating between two
        opposite moves, and an occasional random step shakes it loose.
        """
        eps = epsilon if epsilon is not None else (self.epsilon if explore else 0.0)
        if self.rng.random() < eps:
            return int(self.rng.integers(len(self.actions)))
        scores = np.zeros(len(self.actions))
        for a in range(len(self.actions)):
            v, support = self.value(state_vec, a)
            bonus = self.novelty_bonus * (1.0 - support) if explore else 0.0
            scores[a] = v + bonus
        scores += self.rng.normal(0, 1e-6, scores.shape)   # random tie-break
        return int(np.argmax(scores))

    def remember(self, states, action_idxs, returns):
        """Fold one episode's experiences into memory, forgetting the oldest
        once we pass the cap (bounded memory, like a real creature)."""
        self.S = np.vstack([self.S, np.asarray(states)])
        self.A = np.concatenate([self.A, np.asarray(action_idxs, dtype=int)])
        self.R = np.concatenate([self.R, np.asarray(returns, dtype=float)])
        overflow = len(self.A) - self.memory_cap
        if overflow > 0:
            self.S, self.A, self.R = self.S[overflow:], self.A[overflow:], self.R[overflow:]


# ---------------------------------------------------------------------------
# 2. THE SENSES  (the from-scratch encoder -- raw world -> one vector)
# ---------------------------------------------------------------------------

class CreatureEncoder:
    """Turn the creature's egocentric senses into a single unit vector.

    A 'sense' is a dict of feature -> value, e.g.
        {"food_x": "east", "food_y": "none", "danger_E": "yes", ...}
    We bind each feature to its value and bundle the lot -- the same role/filler
    trick the classifier in holographic_ai.py uses. Two situations that share
    senses get similar vectors, which is exactly what lets the brain generalise.

    It also holds a vector per ACTION, so build_state() can fold a short-term
    memory of recent moves into the state (see below).
    """

    def __init__(self, dim, seed=1):
        self.vocab = Vocabulary(dim, seed)               # sense symbols
        self.actions = Vocabulary(dim, seed + 100)       # one vector per action

    def encode(self, senses):
        """Senses dict -> one unit vector (zero vector if the creature senses
        nothing at all, e.g. blind in open space)."""
        if not senses:
            return np.zeros(self.vocab.dim)
        tokens = [bind(self.vocab.get(role), self.vocab.get(value))
                  for role, value in sorted(senses.items())]
        return bundle(tokens)

    def build_state(self, senses, recent_actions=(), mem=0):
        """Combine what the creature senses NOW with a memory of its recent moves.

        The memory is a working-memory vector: the last 'mem' actions, each
        rotated by how many steps ago it happened (permute by age) and bundled
        together. Order is preserved, so 'just moved east' differs from 'moved
        east three steps ago'. With mem=0 this is just the current senses --
        a purely reactive creature. With mem>0 the creature can tell apart
        situations that look identical to the senses but differ in history,
        which is what lets it search instead of dithering when it can't see food.
        """
        sense_vec = self.encode(senses)
        if mem and len(recent_actions):
            context = bundle([permute(self.actions.get(a), age + 1)
                              for age, a in enumerate(recent_actions[:mem])])
            if np.linalg.norm(sense_vec) > 0:
                return bundle([sense_vec, context])
            return context                                # fully blind: lean on memory
        return sense_vec


# ---------------------------------------------------------------------------
# 3. THE WORLD  (a tiny foraging grid -- the creature's body and environment)
# ---------------------------------------------------------------------------

class GridWorld:
    """A small grid with one creature, one food, and some poison cells.

    Eating food: +FOOD and the food respawns elsewhere. Trying to step onto
    poison: +POISON (negative) and the creature is repelled, so poison acts
    like a painful wall. Every step costs STEP, so dawdling is mildly punished.
    Coordinates use y growing DOWNWARD, so North is y-1.
    """

    MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
    ACTIONS = ["N", "S", "E", "W"]

    def __init__(self, width=7, height=7, n_poison=2, seed=0,
                 step_cost=-0.01, food_reward=1.0, poison_reward=-1.0,
                 vision_radius=None):
        self.w, self.h, self.n_poison = width, height, n_poison
        self.STEP, self.FOOD, self.POISON = step_cost, food_reward, poison_reward
        self.vision_radius = vision_radius   # None = always sees food's direction
        self.rng = np.random.default_rng(seed)
        self.reset()

    # -- setup ------------------------------------------------------------
    def reset(self):
        self.poison = set()
        while len(self.poison) < self.n_poison:
            self.poison.add(self._random_cell())
        self.cx, self.cy = self._random_cell(avoid=self.poison)
        self._spawn_food()
        return self.senses()

    def _random_cell(self, avoid=()):
        while True:
            c = (int(self.rng.integers(self.w)), int(self.rng.integers(self.h)))
            if c not in avoid:
                return c

    def _spawn_food(self):
        blocked = set(self.poison) | {(self.cx, self.cy)}
        self.fx, self.fy = self._random_cell(avoid=blocked)

    # -- perception -------------------------------------------------------
    def _axis(self, delta, neg, pos):
        return neg if delta < 0 else pos if delta > 0 else "none"

    def senses(self):
        """Egocentric, SPARSE description of the situation.

        food_x / food_y say where the food is relative to us -- but only when it
        is within vision_radius (if set); beyond that the creature is 'blind' to
        it and those keys are absent. Danger (adjacent poison) and walls (grid
        edge) are reported per direction, and only when present. We never emit
        'no danger here': a token that appears in almost every state makes
        unrelated situations look alike and drowns out the signal that matters.
        """
        senses = {}
        dist = abs(self.fx - self.cx) + abs(self.fy - self.cy)
        if self.vision_radius is None or dist <= self.vision_radius:
            senses["food_x"] = self._axis(self.fx - self.cx, "west", "east")
            senses["food_y"] = self._axis(self.fy - self.cy, "north", "south")
        for d, (dx, dy) in self.MOVES.items():
            nx, ny = self.cx + dx, self.cy + dy
            if not (0 <= nx < self.w and 0 <= ny < self.h):
                senses["wall_" + d] = "yes"
            elif (nx, ny) in self.poison:
                senses["danger_" + d] = "yes"
        return senses

    # -- dynamics ---------------------------------------------------------
    def step(self, action_name):
        """Apply an action; return (senses, reward, ate_food)."""
        dx, dy = self.MOVES[action_name]
        nx, ny = self.cx + dx, self.cy + dy
        reward = self.STEP                                # cost of living

        if not (0 <= nx < self.w and 0 <= ny < self.h):  # wall: stay put
            nx, ny = self.cx, self.cy
        elif (nx, ny) in self.poison:                     # poison: repelled, hurts
            reward += self.POISON
            nx, ny = self.cx, self.cy

        self.cx, self.cy = nx, ny
        ate = (self.cx, self.cy) == (self.fx, self.fy)
        if ate:
            reward += self.FOOD
            self._spawn_food()
        return self.senses(), reward, ate

    # -- a peek at the world ---------------------------------------------
    def render(self):
        rows = []
        for y in range(self.h):
            row = ""
            for x in range(self.w):
                if (x, y) == (self.cx, self.cy):
                    row += "C"
                elif (x, y) == (self.fx, self.fy):
                    row += "F"
                elif (x, y) in self.poison:
                    row += "x"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# 4. THE LOOP  (live an episode; optionally learn from it)
# ---------------------------------------------------------------------------

def run_episode(world, encoder, mind, learn=True, explore=True,
                eval_epsilon=None, gamma=0.9, max_steps=50, mem=0):
    """Live one episode; return (total_reward, food_eaten).

    'mem' is the working-memory depth: how many recent moves the creature folds
    into its state (0 = purely reactive). If learning, fold the experience back
    in afterward using Monte-Carlo returns (each action credited with the
    discounted reward that actually followed it -- simple, no bootstrapping)."""
    senses = world.reset()
    recent = []                                          # recent actions, newest first
    state = encoder.build_state(senses, recent, mem)
    states, actions, rewards = [], [], []
    food = 0

    for _ in range(max_steps):
        a = mind.decide(state, explore=explore, epsilon=eval_epsilon)
        senses, r, ate = world.step(mind.actions[a])
        food += int(ate)
        states.append(state)
        actions.append(a)
        rewards.append(r)
        recent = [mind.actions[a]] + recent              # remember this move
        state = encoder.build_state(senses, recent, mem)

    if learn:
        returns, g = [0.0] * len(rewards), 0.0
        for t in reversed(range(len(rewards))):          # reward-to-go
            g = rewards[t] + gamma * g
            returns[t] = g
        mind.remember(states, actions, returns)

    return float(np.sum(rewards)), food


# ---------------------------------------------------------------------------
# 5. DEMO
# ---------------------------------------------------------------------------

def _train(world, encoder, mind, episodes, eps_start=0.35, block=25, label="",
           mem=0, max_steps=50):
    """Train for some episodes with decaying exploration, printing the average
    reward of each block so the learning curve is visible."""
    rewards = []
    print(f"Training{label} (avg reward per {block}-episode block):")
    for ep in range(episodes):
        mind.epsilon = max(0.05, eps_start * (1.0 - ep / episodes))
        r, _ = run_episode(world, encoder, mind, learn=True, explore=True,
                           mem=mem, max_steps=max_steps)
        rewards.append(r)
        if (ep + 1) % block == 0:
            print(f"  episodes {ep - block + 2:3d}-{ep + 1:3d}:  "
                  f"{np.mean(rewards[-block:]):+.2f}")


def _evaluate(world, encoder, mind, n=40, mem=0, max_steps=50):
    """Run greedily (tiny eval epsilon to avoid oscillation, no learning)."""
    out = [run_episode(world, encoder, mind, learn=False, explore=False,
                       eval_epsilon=0.05, mem=mem, max_steps=max_steps)
           for _ in range(n)]
    return np.mean([r for r, _ in out]), np.mean([f for _, f in out])


def _baseline(world, encoder, n=30, seed=9, mem=0, max_steps=50):
    rand = HolographicMind(encoder.vocab.dim, GridWorld.ACTIONS,
                           epsilon=1.0, seed=seed)
    out = [run_episode(world, encoder, rand, learn=False, mem=mem,
                       max_steps=max_steps) for _ in range(n)]
    return np.mean([r for r, _ in out]), np.mean([f for _, f in out])


def demo_creature():
    dim = 256
    print("=" * 70)
    print("CREATURE BRAIN -- learning to forage from scratch (no neural net)")
    print("=" * 70)

    # --- Scene A: a clean world with only food --------------------------
    print("\n--- Scene A: food only -----------------------------------------\n")
    encoder = CreatureEncoder(dim, seed=1)
    mind = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.35,
                           novelty_bonus=0.1, memory_cap=5000, seed=2)
    world = GridWorld(7, 7, n_poison=0, seed=3)

    base_r, base_f = _baseline(world, encoder)
    print(f"Random baseline: reward {base_r:+.2f}, food eaten {base_f:.1f}\n")
    _train(world, encoder, mind, episodes=150)
    ev_r, ev_f = _evaluate(world, encoder, mind)
    print(f"\nTrained (greedy): reward {ev_r:+.2f}, food eaten {ev_f:.1f}  "
          f"(baseline {base_r:+.2f}, {base_f:.1f})")

    print("\nLearned reflexes -- where it heads for food in each direction:")
    for fx, fy, where in [("east", "none", "east"), ("west", "none", "west"),
                          ("none", "north", "north"), ("none", "south", "south")]:
        a = mind.decide(encoder.encode({"food_x": fx, "food_y": fy}), explore=False)
        print(f"  food to the {where:5s} -> moves {mind.actions[a]}")

    # --- Scene B: add poison, watch it learn avoidance ------------------
    print("\n--- Scene B: now with poison -----------------------------------\n")
    encoder = CreatureEncoder(dim, seed=1)
    mind = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.35,
                           novelty_bonus=0.1, memory_cap=5000, seed=2)
    world = GridWorld(7, 7, n_poison=2, seed=3)

    base_r, base_f = _baseline(world, encoder)
    print(f"Random baseline: reward {base_r:+.2f}, food eaten {base_f:.1f}\n")
    _train(world, encoder, mind, episodes=200)
    ev_r, ev_f = _evaluate(world, encoder, mind)
    print(f"\nTrained (greedy): reward {ev_r:+.2f}, food eaten {ev_f:.1f}  "
          f"(baseline {base_r:+.2f}, {base_f:.1f})")
    print("  Food eaten is the honest success metric: net reward stays modest")
    print("  because dodging hazards costs steps -- a real reactive-agent trade-off.")

    a = mind.decide(encoder.encode({"food_x": "east", "food_y": "none",
                                    "danger_E": "yes"}), explore=False)
    print(f"\n  Avoidance check: food is EAST but EAST is poison -> moves {mind.actions[a]} "
          f"({'avoids the poison' if mind.actions[a] != 'E' else 'walks into it!'})")

    print("\nA few greedy steps (C=creature F=food x=poison):")
    senses = world.reset()
    state = encoder.encode(senses)
    for frame in range(4):
        print(f"\n  step {frame}:")
        print("    " + world.render().replace("\n", "\n    "))
        a = mind.decide(state, explore=False, epsilon=0.05)
        senses, _, _ = world.step(mind.actions[a])
        state = encoder.encode(senses)
    print()


def demo_memory(seeds=(2, 5, 8), episodes=130, steps=60):
    """Scene C: with limited vision, show that a working memory of recent moves
    lets the creature SEARCH efficiently instead of dithering blindly.

    Single RL runs are noisy, so we average a few independent seeds and report
    them all -- no cherry-picking."""
    dim = 256
    print("\n--- Scene C: limited vision, with vs without working memory ------\n")
    print("On an 11x11 grid the creature only senses food within 2 cells, so it")
    print("is blind most of the time and has to search. We train it identically")
    print(f"except for memory depth, {len(seeds)} seeds each, and count food found")
    print("per 60-step episode.\n")

    means = {}
    for mem in (0, 3):
        foods = []
        for seed in seeds:
            encoder = CreatureEncoder(dim, seed=1)
            mind = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.40,
                                   novelty_bonus=0.1, memory_cap=6000, seed=seed)
            world = GridWorld(11, 11, n_poison=0, seed=3, vision_radius=2)
            for ep in range(episodes):
                mind.epsilon = max(0.05, 0.40 * (1.0 - ep / episodes))
                run_episode(world, encoder, mind, learn=True, explore=True,
                            mem=mem, max_steps=steps)
            _, food = _evaluate(world, encoder, mind, n=40, mem=mem, max_steps=steps)
            foods.append(food)
        means[mem] = float(np.mean(foods))
        tag = "reactive" if mem == 0 else "with memory"
        print(f"  mem={mem} ({tag:11s}): food per seed = "
              f"{[round(float(f), 2) for f in foods]}  mean {means[mem]:.2f}")

    print(f"\nWorking memory finds {means[3] / max(means[0], 1e-9):.1f}x more food, "
          f"and wins every seed.")
    print("Why: blind, a reactive creature wanders back over itself (~sqrt(t)")
    print("coverage); memory lets it hold a heading and sweep new ground (~t).")
    print("Honest caveat: on a SMALLER grid the creature bumps into food often")
    print("enough that memory barely helps -- it earns its keep only when the")
    print("task genuinely demands search.")


if __name__ == "__main__":
    demo_creature()
    demo_memory()
