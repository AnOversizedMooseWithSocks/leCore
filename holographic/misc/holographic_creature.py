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
from holographic.agents_and_reasoning.holographic_ai import random_vector, bind, bundle, permute, Vocabulary


# ---------------------------------------------------------------------------
# 1. THE BRAIN  (general-purpose; nothing creature-specific in here)
# ---------------------------------------------------------------------------

class HolographicMind:
    """Perceive -> decide -> learn, by remembering experiences as PROTOTYPES.

    THE THREE MINDS -- one division of labour, so this never gets confusing again:
      - UnifiedMind     : THE ONE MIND (holographic_unified.py) -- every general faculty (the single encoder,
                          the memory, recall, planning, denoising, the decision machinery). Everything builds
                          on it; it uses THIS class internally for value-learning.
      - CreatureMind    : a SPECIALIZED LAYER on UnifiedMind (holographic_creature_mind.py) -- the reference
                          demo; subclass UnifiedMind and add domain wiring. Build agents THIS way.
      - HolographicMind : the RL ENGINE  <<< THIS CLASS.  A per-action prototype value memory + greedy policy,
                          used by UnifiedMind. NOT an agent-building pattern -- build agents from CreatureMind,
                          never directly from here.

    WHY THIS ENGINE IS KEPT (measured, exp_value_memory.py): its value memory substantially BEATS
    value-learning built on the unified SelfOrganizingMind -- in-sample 0.96 vs 0.57, generalization 0.75 vs
    0.25 (chance) -- and the gap is NOT just soft-vs-hard: a mechanism-matched soft k-NN rival on the unified
    memory ALSO fails (0.57 / 0.25). The bespoke edge is its PER-ACTION prototype organization (a Q-value
    regression by soft k-NN over similar states), which a situation-class + value-table does not replicate.
    So the value memory earns its place; the unified memory does not do this out of the box.

    HOW TO USE IT: do NOT build a new agent directly from this class -- it is the engine, not the pattern. The
    public shape for creature behaviour is holographic_creature_mind.CreatureMind(UnifiedMind): a thin LAYER
    on the one UnifiedMind that inherits the single encoder, the memory, planning, recall, and the decision
    machinery, and uses this engine internally. Through UnifiedMind the encoding goes through the ONE encoder
    (decide/reinforce call perceive, bypassing the creature encoder), and the memory/brain never encode
    themselves -- the "one encoder" rule holds where it is stated.

    ON THE CreatureEncoder (holographic_creature.py): it is NOT redundant duplication to retire. It is the
    creature DOMAIN's encoder -- role/filler binding (the shared primitive) PLUS two things perceive does not
    provide: build_state's action-memory (a working memory of recent moves) and the `seen` tracking that
    HolographicMind.describe REQUIRES to decode a state back into sense terms. The rescue canary is
    tie-sensitive to its EXACT output (see the kept-negative in CreatureEncoder.encode: a 1e-16 change flips
    the trajectory). Creature-style agents reuse it on PURPOSE -- navigator's "inception" demo and app's maze
    console run the same brain+encoder in a different world. Like the value memory, it earns its place; routing
    these through perceive would drop describe/action-memory and re-baseline the whole tie-sensitive suite. The
    value memory and this encoder both stay, measured/grounded.

    An experience is (state_vector, action, return). A naive store would keep
    every single one, but the creature meets the same egocentric situation again
    and again -- "star east, wall north" happens in a thousand different cells --
    so a flat list is mostly duplicates. Instead we do what the image side does
    with classes: keep one PROTOTYPE per distinct situation.

      * Classification: a prototype is a class. A new experience is matched
        (cosine) to the existing prototypes for that action; if it is close
        enough it BELONGS to that class, otherwise it starts a new one. This is
        the same bundle-and-cosine classifier the vision panel uses.
      * Layering: a prototype is a SUPERPOSITION -- we fold each member's state
        vector into a running sum (the bundle) and keep a running mean of the
        returns. Averaging denoises the Monte-Carlo returns, so the compressed
        memory is often a BETTER value estimate than the raw pile, not just a
        smaller one.

    To value an action in a state we cosine-match it to that action's prototypes
    and take a similarity-weighted average of their mean returns -- the same knn
    formula as before, but over a few hundred prototypes instead of thousands of
    raw rows. No weights to train, no backprop -- the learning is the remembering
    (and the organising). The compressed memory is also exposed via prototypes()
    so it can be indexed by the recursive HoloForest for associative recall, the
    same way the image vault indexes its plates (see demo_introspect()).
    """

    def __init__(self, dim, actions, k=12, epsilon=0.1, novelty_bonus=0.3,
                 memory_cap=6000, seed=0, merge=0.92, ret_alpha=0.1,
                 maintain=False, reorg_duplicate=0.85, redundancy_floor=0.35,
                 surprise_floor=0.4, maintain_gap=400, buffer_cap=1200, check_every=400,
                 capacity=0, robust_returns=False, value_backend="table"):
        self.dim = dim
        self.actions = list(actions)        # e.g. ["N", "S", "E", "W"]
        self.k = k                          # prototypes consulted per action
        self.epsilon = epsilon              # chance of a random exploratory move
        self.novelty_bonus = novelty_bonus  # optimism for rarely-tried actions
        self.memory_cap = memory_cap        # max prototypes PER ACTION (evict the rarest)
        self.merge = merge                  # fold into a prototype when cosine >= this
        # CAPACITY-AWARE LAYERING: a prototype is a bundle (superposition) of its
        # members, and a bundle has FINITE capacity -- fold too many distinct members
        # into one and the unit can no longer resemble any of them (the cosine readout
        # collapses ~1/sqrt(count): the capacity cliff). `capacity` caps how many
        # members fold into a single prototype; when a matching prototype is already
        # full, the experience starts a NEW sub-prototype for that action instead of
        # blurring the full one further. classify()/value() score over ALL of an
        # action's prototypes, so sub-prototypes are read transparently.
        #   Measured tradeoff (don't set this blind): on a harsh fixed obstacle world
        #   where the same egocentric situation recurs constantly, unbounded folding
        #   (capacity=0) blurred the value memory into a near-degenerate policy
        #   (~1 star/life), while a cap rescued it (~4 at cap=sqrt(dim)) -- but a tight
        #   cap fragments memory hard (~100 -> ~8700 prototypes) and is high-variance.
        #   So the right cap trades fidelity against memory and is task-dependent;
        #   capacity=0 (no cap) is the default, and a cap is a measured knob, not a
        #   free win -- capacity_report() exposes the load so you can tune it.
        self.capacity = capacity
        # BLIND-STATE COMPASS FALLBACK: when the brain has essentially no memory for
        # any allowed action here (max support < blind_floor), a random pick is the
        # worst policy for "I'm lost". If a goal_<dir> token is present in senses, take
        # that direction instead of guessing -- replacing a coin flip with "follow the
        # compass," which is strictly better when the value memory is empty here.
        # 0.0 = off (choose on value+novelty even when blind); set a small positive
        # floor (e.g. 0.15) to turn it on.
        self.blind_floor = 0.0
        self.ret_alpha = ret_alpha          # floor on the return step size (recency)
        # Self-maintenance: keep the brain itself from going stale. The base mind
        # never forgets -- its bundles only grow -- and it lets near-duplicate
        # prototypes (cosine below `merge`) pile up. When the world SHIFTS, those
        # duplicates each hold the old value up and online updates only touch one at
        # a time, so the orchestrator that runs on this brain cannot unlearn.
        #   maintain=True  -- threshold mode: fold when a hand-set signal crosses.
        #   maintain='auto'-- AUTONOMOUS: no behavioural thresholds at all. The brain
        #     keeps a window of recent experience, periodically SPECULATES a few
        #     reorganised versions of itself (fold its duplicates at a couple of
        #     grains; rebuild from recent experience), measures each one's fit on a
        #     held-out slice of that window, and adopts whichever predicts reality
        #     best -- breaking near-ties toward the leanest. The data decides; the
        #     only knobs left are resource budgets (window size, how often to look).
        self.maintain = maintain
        self.reorg_duplicate = reorg_duplicate
        self.redundancy_floor = redundancy_floor
        self.surprise_floor = surprise_floor
        self.maintain_gap = maintain_gap
        self.buffer_cap = buffer_cap        # size of the recent-experience window (a budget)
        self.check_every = check_every      # look this often, in new experiences (a budget)
        # MAINTENANCE BUDGET (Request 2): how many candidate memories auto_maintain builds and
        # scores each tick. The defaults reproduce the full 8-way search exactly. A deployment
        # that knows it is quiet (a stable courier) can trim these -- fewer fold grains, and/or
        # drop the REFRESHING family that only earns its keep after a regime shift -- so a calm
        # stretch stops paying full model-selection cost on every tick. Set on the instance (or
        # overridden per call). NOTE these are CALLER controls, not an auto-gate on `surprise`:
        # surprise here is an EMA of reward prediction error, which tracks reward NOISE as much
        # as regime change, so it stays high in a noisy-but-stable world and is not a reliable
        # "nothing shifted" signal -- the caller, who knows the deployment, decides the cadence.
        self.maintain_grains = (0.9, 0.82, 0.75)   # fold grains tried in each family
        self.maintain_refresh = True               # include the refreshing (forget-old) family
        self.surprise = 0.0                 # EMA of |actual return - predicted value|
        self._since_reorg = 0
        self._buf = []                      # recent (state, action, return) for self-checking
        self._added = 0                     # new experiences since the last autonomous look
        self.reorganizations = 0
        self.last_choice = None             # what the autonomous step last decided to do
        self.rng = np.random.default_rng(seed)
        self._seed = seed                   # remembered so a trained brain can be persisted
        n = len(self.actions)
        # Per action, parallel arrays so recall is one matrix-vector product:
        #   _sum  : running SUM of member state vectors (the un-normalised bundle)
        #   _unit : its unit-length mean -- the class direction we match against
        #   _ret  : running MEAN return of the members
        #   _cnt  : how many experiences the prototype represents (its support)
        self._sum = [np.zeros((0, dim)) for _ in range(n)]
        self._unit = [np.zeros((0, dim)) for _ in range(n)]
        self._ret = [np.zeros(0) for _ in range(n)]
        self._cnt = [np.zeros(0) for _ in range(n)]
        # ROBUST RETURNS (D2, opt-in): an outlier reward (a fluke jackpot, a sensor glitch) folded straight into
        # a prototype's running-mean return swings the value estimate. When robust_returns is on, winsorise the
        # residual to +/- k * `_ret_dev` before folding it in, where `_ret_dev` is ONE running estimate of the
        # typical |residual| (the reward NOISE scale -- shared across prototypes because the noise scale, unlike
        # the mean, is roughly constant). Measured ~3x lower value error under outlier rewards, no cost on clean
        # data. A single scalar, so it stays cheap and serialises trivially; off by default (bit-identical).
        self.robust_returns = bool(robust_returns)
        self._ret_dev = None                # lazy: seeded from the first residual so the scale starts sensibly
        self.experiences = 0                # total raw experiences absorbed (compression stat)
        # VSA VALUE BACKEND (opt-in). Default 'table' = the prototype machinery above, BIT-IDENTICAL. 'holo'
        # routes value()/_absorb() to a two-bundle HolographicValueHead, so the whole per-action policy is a
        # FIXED-SIZE, savable hypervector program (Q_a, N_a) instead of a growing (vector, scalar) table, and
        # learning is one bundling step. It is fixed-size so it needs no consolidation -- the projection /
        # maintain machinery below is simply unused in this mode (value/_absorb return before reaching it).
        self.value_backend = value_backend
        self._holo = value_backend in ("holo", "routed")
        if self._holo:
            from holographic.agents_and_reasoning.holographic_valuehead import HolographicValueHead, RoutedValueHead
            self._value_head = (RoutedValueHead(dim, n) if value_backend == "routed"
                                else HolographicValueHead(dim, n))
        else:
            self._value_head = None
        # PROJECTION consolidation (see consolidate()): once set, every incoming
        # state is projected into this shared low-rank basis -- with a residual
        # guard that grows the basis when the world develops structure the
        # current shadow cannot show.
        self._basis = None                  # dim x r orthonormal basis (None = raw)
        self._oob = 0.0                     # EMA of out-of-basis state energy
        self._raw_ring = []                 # small ring of recent RAW states (for expansion)
        self.expand_at = 0.05               # guard threshold on _oob

    def perceive_vec(self, v):
        """Route any incoming state through the projection, if one is set.

        After consolidate(), the memory lives in a low-rank subspace -- the one
        ~22-D object all those 512-D prototypes were shadows of. A raw state is
        projected to its coefficients (cosines are preserved because everything
        meaningful lies in the basis). THE GUARD: the energy a state carries
        OUTSIDE the basis is tracked as a slow EMA, because a shadow hides new
        structure -- measured, a brain consolidated in a poison-free world left a
        danger sense with only 4% of its energy inside the basis, i.e. poison was
        nearly invisible. When out-of-basis energy grows past `expand_at`, the
        basis is EXPANDED from a small ring of recent raw states (new orthogonal
        directions appended; old prototypes get zero coefficients there, which is
        correct -- they truly had no such component). Compress when the world is
        stable, grow when it is not: the flux-guard pattern, fourth appearance."""
        v = np.asarray(v, float)
        if self._basis is None or v.shape[-1] != self._basis.shape[0]:
            return v                                  # raw mode, or already projected
        self._raw_ring.append(v.copy())
        if len(self._raw_ring) > 240:
            self._raw_ring.pop(0)
        c = v @ self._basis
        tot = float(v @ v)
        if tot > 1e-12:
            res = 1.0 - float(c @ c) / tot
            self._oob += 0.02 * (res - self._oob)
            if self._oob > self.expand_at:
                self._expand_basis()
                c = v @ self._basis                   # re-project in the grown basis
        return c

    def consolidate(self, energy=0.999):
        """PROJECTION: discover the low-rank subspace the stored prototypes share
        (SVD over all of them -- the overlap between concepts IS the registration
        mark that they are shadows of one object) and re-store the whole memory as
        coefficients in it. Measured on trained brains: 99.9% of prototype energy
        sits in ~22-24 of 512 dimensions (the span of the sense-atom vocabulary),
        giving ~21x smaller memory and ~5x faster decide() at behavioural parity
        (forage 122 -> 120 stars; 16x16 maze 90% -> 95% escapes). Returns the rank."""
        V, _, _ = self.prototypes()
        if len(V) < 2:
            return None
        _, s, Vt = np.linalg.svd(V, full_matrices=False)
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        r = int(np.searchsorted(e, energy)) + 1
        B = Vt[:r].T
        for a in range(len(self.actions)):
            if len(self._unit[a]):
                C = np.asarray(self._unit[a]) @ B
                n = np.linalg.norm(C, axis=1, keepdims=True)
                self._unit[a] = C / np.where(n == 0, 1, n)
                self._sum[a] = np.asarray(self._sum[a]) @ B
            else:
                self._unit[a] = np.zeros((0, r))
                self._sum[a] = np.zeros((0, r))
        # the recent-experience buffer must live in the same space
        self._buf = [((np.asarray(s_, float) @ B), a_, r_) for s_, a_, r_ in self._buf]
        self._basis = B
        self._oob = 0.0
        self._raw_ring = []
        return r

    def _expand_basis(self):
        """Grow the basis with the directions the recent raw states carry outside
        it (Gram-Schmidt via SVD of the residuals). Existing coefficient banks get
        zero columns for the new directions."""
        if not self._raw_ring:
            self._oob = 0.0
            return
        R = np.stack(self._raw_ring)
        resid = R - (R @ self._basis) @ self._basis.T
        _, s, Vt = np.linalg.svd(resid, full_matrices=False)
        if not len(s) or s[0] <= 1e-9:
            self._oob = 0.0
            return
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        add = max(1, int(np.searchsorted(e, 0.9)) + 1)        # capture 90% of residual
        add = min(add, self._basis.shape[0] - self._basis.shape[1])
        if add <= 0:
            self._oob = 0.0
            return
        newdirs = Vt[:add].T
        self._basis = np.hstack([self._basis, newdirs])
        for a in range(len(self.actions)):                    # zero-pad old coefficients
            if len(self._unit[a]):
                z = np.zeros((len(self._unit[a]), add))
                self._unit[a] = np.hstack([self._unit[a], z])
                self._sum[a] = np.hstack([self._sum[a], z])
        self._buf = [(np.concatenate([s_, np.zeros(add)]), a_, r_)
                     for s_, a_, r_ in self._buf]
        self._oob = 0.0

    def value(self, state_vec, action_idx):
        """Estimate the value of an action in a state.

        Returns (value, support). 'support' is how similar the closest prototype
        is (0 = we've basically never seen this) -- the curiosity signal. Because
        prototype directions are unit length, the matrix product below IS the
        cosine similarity to every prototype at once.
        """
        if self._holo:                               # VSA backend: value is <s,Q_a>/<s,N_a>, a single dot
            return self._value_head.value(state_vec, action_idx)
        U = self._unit[action_idx]
        if not len(U):
            return 0.0, 0.0
        # Robustness: decide() perceives (projects) the state once before the per-action
        # loop, so value() normally gets a vector already in the prototype space. But a
        # DIRECT caller (a demo, a save/reload check, _greedy) may pass a RAW full-dim probe
        # to a consolidated brain, whose prototypes live in the low-rank basis. Project it
        # here if its width is the full dim -- a cheap, side-effect-free lift (unlike
        # perceive_vec, which also feeds the flux-guard ring, so we must NOT call that here
        # or a direct value() call would double-count out-of-basis energy). An already-
        # projected vector (basis width) is left untouched.
        state_vec = np.asarray(state_vec, float)
        if self._basis is not None and state_vec.shape[-1] == self._basis.shape[0]:
            state_vec = state_vec @ self._basis
        sims = U @ state_vec
        rets = self._ret[action_idx]
        if sims.size > self.k:                       # keep only the k nearest prototypes
            top = np.argpartition(sims, -self.k)[-self.k:]
            sims, rets = sims[top], rets[top]
        weights = np.clip(sims, 0.0, None)           # ignore unrelated/opposite prototypes
        total = weights.sum()
        if total <= 1e-9:
            return 0.0, 0.0
        return float((weights * rets).sum() / total), float(sims.max())

    def _value_projected(self, state_vec, action_idx):
        """value() WITHOUT the basis-width check -- the hot-path fast path (Request 3).

        decide() runs perceive_vec ONCE before its per-action loop, so by the time values are
        scored the state is already in prototype space; the public value() still re-checks the
        width on every call to stay safe for direct callers. This private variant assumes the
        caller has already projected (or never consolidated), so it skips the branch. The numbers
        are BIT-IDENTICAL to value() on an already-projected (or raw, un-consolidated) state -- it
        only drops the conditional, never changes the math. Used by value_batch below."""
        if self._holo:                               # VSA backend (hot path mirrors value())
            return self._value_head.value(state_vec, action_idx)
        U = self._unit[action_idx]
        if not len(U):
            return 0.0, 0.0
        sims = U @ state_vec
        rets = self._ret[action_idx]
        if sims.size > self.k:
            top = np.argpartition(sims, -self.k)[-self.k:]
            sims, rets = sims[top], rets[top]
        weights = np.clip(sims, 0.0, None)
        total = weights.sum()
        if total <= 1e-9:
            return 0.0, 0.0
        return float((weights * rets).sum() / total), float(sims.max())

    def value_batch(self, state_vec, action_idxs=None):
        """Score several actions for one state in a single call: returns (values, supports) as
        aligned arrays. Equivalent to calling value() for each action, but it does the
        basis projection ONCE up front (not once per action) and then takes the no-branch fast
        path per action -- so the result is BIT-IDENTICAL to a value() loop while a consolidated
        brain handed a RAW state pays for one projection instead of len(action_idxs).

        HONEST NOTE on speed: this is an API convenience, not a hot-path win. Measured on real
        trained brains, scoring all actions in one batched call is within a few percent of the
        per-action loop -- the per-action prototype banks are small and BLAS already runs four
        little matrix-vector products about as fast as one stacked one, while the per-action
        top-k (argpartition) is irreducible and cannot be merged. The one real saving is the
        single projection for the consolidated-brain raw-state case above. (A stacked one-matmul
        form was measured to be both SLOWER -- the concatenate costs more than it saves -- and
        NOT bit-identical to the per-action product, differing at ~1e-16, the same tie-break
        hazard that kept bind_batch out of the encoder. So it is deliberately not used.)

        action_idxs: which actions to score (default: all). state_vec may be raw or projected."""
        idxs = list(range(len(self.actions))) if action_idxs is None else list(action_idxs)
        state_vec = np.asarray(state_vec, float)
        # project ONCE (Request 3) -- a side-effect-free lift, exactly as value() does per call,
        # so the per-action results below match value()'s bit-for-bit. NOT perceive_vec: that
        # feeds the flux-guard ring and must run only once per real perception, not per scoring.
        if self._basis is not None and state_vec.shape[-1] == self._basis.shape[0]:
            state_vec = state_vec @ self._basis
        values = np.zeros(len(idxs))
        supports = np.zeros(len(idxs))
        for i, a in enumerate(idxs):
            values[i], supports[i] = self._value_projected(state_vec, a)
        return values, supports

    def decide(self, state_vec, explore=True, epsilon=None, among=None,
               senses=None, avoid=("danger", "wall"), soft=()):
        """Choose an action. Mostly greedy on value, with two sources of
        exploration: an epsilon chance of a random move, and (while exploring) a
        novelty bonus that favours actions rarely tried in situations like this.

        'epsilon' overrides the random-move chance for this call. A small value
        even at evaluation time (e.g. 0.05) is worth keeping: a purely greedy,
        memoryless reactive agent can get trapped oscillating between two
        opposite moves, and an occasional random step shakes it loose.

        'senses' puts the SAFETY REFLEXES inside the brain itself: when the
        current senses dict is passed, any action whose direction is flagged by
        a prefix in `avoid` is vetoed below the value estimate. Both vetoes are
        measured: the danger veto ended the compounding-poison deaths the
        survival bench exposed (67-73% of full lives -> 0%), and the wall veto
        -- named by the brain's own introspection, which caught it choosing E at
        value +0.43 while sensing wall_E=yes -- solved the cluttered-world open
        problem (stars 5.1 -> 19.8, dither 79% -> 43%). Putting them HERE means
        every caller of the brain (episode runner, demos, the showcase app's
        own loop) gets the same safety by passing what the creature senses;
        callers that pass nothing are byte-for-byte unchanged.

        'soft' tiers the veto into HARD vs SOFT blocks. Some obstacles are
        permanent (a wall: moving into it does nothing) and some are temporary (a
        red light or a car ahead: wait and it clears). When every direction is
        blocked, instead of lifting the veto to ALL actions (a blind pick that may
        drive at a permanent wall), lift it only to the SOFT-blocked ones -- the
        temporary blocks worth waiting on -- and only fall back to all actions if
        nothing is soft either (truly walled in). Prefixes in `soft` name the
        temporary kinds; `soft=()` means no tiering (every block is treated alike).

        'among' restricts the choice to a subset of action indices -- the same
        routing move classify() uses for labels (compete only within the
        legitimate pool). When both `among` and `senses` are given the vetoes
        intersect. If everything is vetoed (fully surrounded), the restriction
        lifts: the brain's call. Exploration respects the restriction too -- a
        random move is random among the allowed, never a coin-flip into poison.

        'blind_floor': when the brain has no memory for any allowed action (max
        support < blind_floor) and a goal_<dir> token is present in senses, take the
        goal direction rather than guessing. Off when blind_floor == 0.0."""
        state_vec = self.perceive_vec(state_vec)
        idxs = list(range(len(self.actions))) if among is None else list(among)
        if senses is not None:
            ok = [i for i in idxs
                  if not any(f"{p}_{self.actions[i]}" in senses for p in avoid)]
            if ok:
                idxs = ok                            # some direction is unblocked
            elif soft:
                # every direction hard-or-soft blocked: prefer WAITING on a soft
                # (temporary) block over a blind pick among permanent walls
                soft_ok = [i for i in idxs
                           if any(f"{p}_{self.actions[i]}" in senses for p in soft)]
                idxs = soft_ok or idxs
            # else: fully walled with nothing soft -- brain's call, keep idxs as-is
        eps = epsilon if epsilon is not None else (self.epsilon if explore else 0.0)
        if self.rng.random() < eps:
            return int(idxs[int(self.rng.integers(len(idxs)))])
        scores = np.full(len(self.actions), -np.inf)
        supports = {}
        for a in idxs:
            v, support = self.value(state_vec, a)
            supports[a] = support
            bonus = self.novelty_bonus * (1.0 - support) if explore else 0.0
            scores[a] = v + bonus
        # blind-state compass fallback: no memory anywhere here, but the compass
        # points somewhere -- follow it instead of guessing.
        if (senses is not None and self.blind_floor > 0.0
                and max(supports.values(), default=0.0) < self.blind_floor):
            goal_dirs = [i for i in idxs if f"goal_{self.actions[i]}" in senses]
            if goal_dirs:
                return int(goal_dirs[0] if len(goal_dirs) == 1
                           else max(goal_dirs, key=lambda a: scores[a]))
        scores[idxs] += self.rng.normal(0, 1e-6, len(idxs))   # random tie-break
        return int(np.argmax(scores))

    def describe(self, vec, encoder, floor=0.18):
        """INTROSPECTION: decode a state or prototype back into SENSE terms.

        The states are role-bound bundles (bind(sense, value)), so each role the
        encoder has ever seen is unbound from the vector and cleaned up against
        the values experience registered (encoder.seen) -- the relations decode,
        turned on the brain's own memory. A role whose best value scores below
        `floor` is reported absent (senses are sparse, so most roles ARE absent
        from most states; the floor separates real membership from cross-term
        noise, and the measurement that set it is in the tests). Works on
        consolidated minds too: coefficient vectors are lifted back through the
        basis before unbinding. Returns {role: (value, confidence)}."""
        v = np.asarray(vec, float)
        if self._basis is not None and v.shape[-1] == self._basis.shape[1]:
            v = self._basis @ v                       # lift the shadow back up
        from holographic.agents_and_reasoning.holographic_ai import bind, involution, cosine
        out = {}
        for role, values in sorted(encoder.seen.items()):
            est = bind(v, involution(encoder.vocab.get(role)))
            best, score = None, -2.0
            for val in values:
                s = cosine(est, encoder.vocab.get(val))
                if s > score:
                    best, score = val, s
            if score >= floor:
                out[role] = (best, float(score))
        return out

    def why_differ(self, vec_a, vec_b, encoder, floor=0.18):
        """Why did two situations deserve different treatment? The per-role
        verdict between two states/prototypes, in the creature's own sense
        vocabulary: [(role, value_a, value_b, shared)] where None means the role
        is absent from that state. The same explain operation the unified mind
        runs on its records, turned inward on the policy's memory."""
        da, db = self.describe(vec_a, encoder, floor), self.describe(vec_b, encoder, floor)
        out = []
        for role in sorted(set(da) | set(db)):
            va = da.get(role, (None,))[0]
            vb = db.get(role, (None,))[0]
            out.append((role, va, vb, va == vb))
        return out

    # ---- planning: corridor baking + re-anchoring (the navigation faculty) ----------------------
    # MIGRATION NOTE (Phase 0): these expose the SAME corridor-planning capability UnifiedMind has
    # (UnifiedMind.plan / replan_needed), by delegating to the one shared module both use -- so an NPC
    # running on a creature can bake a route and re-anchor without the engine having to invert the
    # creature<->mind relationship first. The general logic lives in holographic_plan (and the directed
    # / gated faculties under it), not duplicated here; this is just the creature reaching it. The full
    # "creature builds on UnifiedMind" inversion (and access to the mind's recall / recognize / denoise,
    # which touch the mind's OWN memory and so need the relationship settled) is a later phase. These two
    # are substrate-level (they operate on supplied vectors, not the creature's value memory), so they add
    # the capability with zero weight and leave the decision path bit-identical.

    def plan(self, start, field_step, max_steps=14, floor=0.15, action_of=None, is_branch=None):
        """Bake one CORRIDOR -- a short executable route to the next decision point -- on the directed
        substrate, the way past the per-structure capacity cap. `field_step(node) -> next_or_None` is the
        caller's downhill stepper (a goal gradient, a flow field, this brain's own greedy move); the rollout
        stops at `is_branch(node)` (a junction worth a real decide()) or `max_steps` (keep it at/under the
        dim's reliable decode depth, ~15 at dim 512-1024). Returns a Plan (the plan hypervector, the decoded
        tile route, the decoded direction labels via `action_of`, and a per-step throughput). The courier
        executes the baked steps and re-anchors via replan_needed(); the brain is consulted once per
        corridor, not once per tile. Same capability as UnifiedMind.plan()."""
        from holographic.scene_and_pipeline.holographic_plan import plan as _plan
        return _plan(start, field_step, max_steps=max_steps, floor=floor, seed=self._seed,
                     action_of=action_of, is_branch=is_branch)

    def replan_needed(self, p, executed, tile_ok=None, floor=0.15):
        """The cheap per-tick guard for a baked Plan: re-anchor (call plan() again) when the plan is
        exhausted, the next baked step's throughput is below `floor`, or `tile_ok(next_tile)` reports the
        next tile is blocked; else execute the next baked step. No value() calls, no decode work."""
        from holographic.scene_and_pipeline.holographic_plan import replan_needed as _replan
        return _replan(p, executed, tile_ok=tile_ok, floor=floor)

    def plan_route(self, start, field_step, max_total=200, corridor=14, floor=0.15,
                   action_of=None, is_branch=None):
        """Bake a WHOLE arbitrarily-long route in one call, chaining cap-sized corridors and re-anchoring
        internally at each leg's reliably-decoded end -- the way past the per-structure ~15 cap delivered as
        a single result. `corridor` is the per-leg length and must stay at/under the dim's reliable decode
        depth (default 14, safe at dim 512-1024); `max_total` caps the whole route. Use this to get a full
        route in hand (display / validate / pre-plan); a real-time courier reacting to traffic still wants
        plan() + replan_needed. Returns a Route. Same capability as UnifiedMind.plan_route()."""
        from holographic.scene_and_pipeline.holographic_plan import plan_route as _plan_route
        return _plan_route(start, field_step, max_total=max_total, corridor=corridor, floor=floor,
                           seed=self._seed, action_of=action_of, is_branch=is_branch)

    def chunk_route(self, items, chunk=14, floor=0.15, action_of=None):
        """Store/replay an EXPLICIT ordered sequence you ALREADY HAVE (a GPS route, a fixed protocol, any known
        list) past the per-structure cap, by splitting it into <=chunk-element clean pieces. The explicit-list
        twin of plan_route: the sequence is given, so it skips the rollout and just chunks, bakes, and replays
        it exactly. Effective length is unbounded at linear cost (~N/chunk pieces); each chunk is one compact
        vector. `chunk` must stay at/under the dim's reliable decode depth (default 14). Returns a Route. Same
        capability as UnifiedMind.chunk_route()."""
        from holographic.scene_and_pipeline.holographic_plan import chunk_route as _chunk_route
        return _chunk_route(items, chunk=chunk, floor=floor, seed=self._seed, action_of=action_of)

    def index_route(self, route):
        """Build a sub-linear RANDOM-ACCESS index over a chunked route (from plan_route / chunk_route): a BVH
        over the chunks, so "where am I?" is a jump not a replay. Build once, query many via .locate(query) ->
        (chunk, position, global_step). Same capability as UnifiedMind.index_route()."""
        from holographic.scene_and_pipeline.holographic_plan import RouteIndex
        return RouteIndex(route)

    def penalize_recent(self, amount=0.5, n=4):
        """Online 'stuck' signal: nudge DOWN the value of the last `n` (state, action)
        pairs the brain acted on, without waiting for the episode to finish. Learning
        here is Monte-Carlo at episode end, so a creature trapped in a loop never
        finishes and never learns the one lesson it needs -- it just gets rescued. This
        lets a detected loop leave a mark immediately: the prototypes for the moves that
        led into it get their mean return lowered, so next time the brain is less likely
        to repeat them.

        Requires the recent-experience buffer, which exists under maintain=True or
        'auto' (self._buf holds recent (state, action, return)). Returns the number of
        (state, action) pairs penalised. `amount` is subtracted from each pair's
        prototype mean return (scaled by the prototype's recency step), matching the
        units of the returns the brain already learns from."""
        if not self._buf:
            return 0
        hit = 0
        for s, a, _r in self._buf[-n:]:
            if not (0 <= a < len(self._unit)):
                continue                              # action index from a stale buffer
            U = self._unit[a]
            # A maintenance pass (reorganize / auto_maintain swap / basis change) can
            # rebuild or re-dimension the prototype banks AFTER this (s, a) was buffered,
            # so a buffered state may no longer match the current bank's width, and the
            # four per-action arrays are only guaranteed in lockstep at rest. Skip any
            # buffer entry that doesn't line up cleanly rather than indexing past an
            # array that maintenance has since resized.
            if not len(U) or U.shape[1] != np.asarray(s).shape[-1]:
                continue                              # state lives in a different space now
            sims = U @ s
            j = int(sims.argmax())
            if j >= len(self._cnt[a]) or j >= len(self._ret[a]):
                continue                              # banks out of lockstep -> don't index
            if sims[j] >= self.merge:                 # the prototype this move folded into
                alpha = max(1.0 / (self._cnt[a][j] + 1.0), self.ret_alpha)
                self._ret[a][j] -= alpha * amount     # lower its learned value
                hit += 1
        return hit

    def remember(self, states, action_idxs, returns):
        """Fold one episode's experiences into the prototype memory."""
        states = np.asarray([self.perceive_vec(s) for s in states], float)
        action_idxs = np.asarray(action_idxs, dtype=int)
        returns = np.asarray(returns, dtype=float)
        for s, a, r in zip(states, action_idxs, returns):
            if self.maintain:                        # track prediction error (surprise)
                pred, _ = self.value(s, int(a))
                self.surprise += 0.02 * (abs(float(r) - pred) - self.surprise)
                self._buf.append((s.copy(), int(a), float(r)))   # recent-experience window
                if len(self._buf) > self.buffer_cap:
                    self._buf.pop(0)
                self._added += 1
            self._absorb(s, int(a), float(r))
            self._since_reorg += 1
        if self.maintain == 'auto':
            if self._added >= self.check_every and len(self._buf) >= self.check_every:
                self.auto_maintain()
        elif self.maintain and self.should_reorganize():
            self.reorganize()

    def _absorb(self, state, a, ret):
        """Classify one experience into a prototype (or start a new one), then
        layer it in: extend the bundle, update the mean return and support."""
        self.experiences += 1
        if self._holo:                               # VSA backend: learning is one bundling step
            self._value_head.absorb(state, a, ret)
            return
        U = self._unit[a]
        if len(U):
            sims = U @ state
            j = int(sims.argmax())
            # capacity-aware: if the nearest prototype is close enough to merge BUT is
            # already holding `capacity` members, don't blur it further -- fall through
            # to start a fresh sub-prototype for this same situation instead.
            full = self.capacity and self._cnt[a][j] >= self.capacity
            if sims[j] >= self.merge and not full:    # close enough: it joins this class
                self._sum[a][j] = self._sum[a][j] + state
                c = self._cnt[a][j]
                # Recency-weighted return: a true mean at first (1/(c+1)), then a
                # floored step size so the prototype keeps tracking the creature's
                # IMPROVING policy instead of being dragged down forever by the
                # noisy returns of early exploration. (Standard RL running average.)
                alpha = max(1.0 / (c + 1.0), self.ret_alpha)
                resid = ret - self._ret[a][j]
                if self.robust_returns:                  # D2: clamp an outlier reward's pull to +/- k robust-scales
                    if self._ret_dev is None:
                        self._ret_dev = abs(resid) + 1e-6   # seed the scale from the first residual seen
                    clamp = 3.0 * self._ret_dev          # winsorise: a fluke can move the value by at most this
                    resid_w = max(-clamp, min(clamp, resid))
                    self._ret[a][j] += alpha * resid_w
                    self._ret_dev += 0.05 * (abs(ret - self._ret[a][j]) - self._ret_dev)   # track the noise scale
                else:
                    self._ret[a][j] += alpha * resid     # plain running average (unchanged default path)
                self._cnt[a][j] = c + 1.0
                v = self._sum[a][j]; nv = np.linalg.norm(v)
                self._unit[a][j] = v / nv if nv > 0 else v
                return
        norm = np.linalg.norm(state) or 1.0          # otherwise: a brand-new class
        self._sum[a] = np.vstack([self._sum[a], state])
        self._unit[a] = np.vstack([self._unit[a], state / norm])
        self._ret[a] = np.append(self._ret[a], ret)
        self._cnt[a] = np.append(self._cnt[a], 1.0)
        if len(self._ret[a]) > self.memory_cap:      # bounded memory: forget the rarest
            j = int(self._cnt[a].argmin())
            self._sum[a] = np.delete(self._sum[a], j, axis=0)
            self._unit[a] = np.delete(self._unit[a], j, axis=0)
            self._ret[a] = np.delete(self._ret[a], j)
            self._cnt[a] = np.delete(self._cnt[a], j)

    def prototype_count(self):
        """Total prototypes kept across all actions (the compressed memory size)."""
        return int(sum(len(r) for r in self._ret))

    def capacity_report(self):
        """How loaded the prototype bundles are -- the capacity diagnostic. Returns
        {'prototypes', 'max_count', 'mean_count', 'overloaded'}: 'overloaded' counts
        prototypes holding more members than the dimension comfortably supports
        (count > ~sqrt(dim)), where the unit bundle starts to lose resemblance to its
        members. With capacity-aware layering on, no prototype exceeds `capacity`, so
        'overloaded' stays at 0."""
        soft_cap = self.dim ** 0.5
        counts = [c for a in range(len(self.actions)) for c in self._cnt[a]]
        if not counts:
            return {"prototypes": 0, "max_count": 0, "mean_count": 0.0, "overloaded": 0,
                    "soft_cap": round(soft_cap, 1)}
        counts = np.asarray(counts, float)
        return {"prototypes": int(len(counts)), "max_count": int(counts.max()),
                "mean_count": float(round(counts.mean(), 2)),
                "overloaded": int((counts > soft_cap).sum()),
                "soft_cap": round(soft_cap, 1)}

    def prototypes(self):
        """The compressed memory laid out for indexing: (unit vectors, action per
        row, mean return per row). This is what a HoloForest indexes for
        associative recall -- the creature's experience as content-addressable
        plates, exactly like the image vault."""
        vecs, acts, rets = [], [], []
        for a in range(len(self.actions)):
            for i in range(len(self._unit[a])):
                vecs.append(self._unit[a][i]); acts.append(a); rets.append(self._ret[a][i])
        V = np.stack(vecs) if vecs else np.zeros((0, self.dim))
        return V, np.asarray(acts, dtype=int), np.asarray(rets, dtype=float)

    # -- self-maintenance: the organizer's tools, turned on the brain itself ----
    def redundancy(self):
        """Fraction of prototypes that have a near-duplicate within the same action
        (cosine >= reorg_duplicate). High means the memory has gone bloated -- and,
        worse, that stale duplicates can hold an out-of-date value up because online
        updates only touch one of them at a time."""
        total = dup = 0
        for a in range(len(self.actions)):
            U = self._unit[a]
            if len(U) < 2:
                total += len(U); continue
            sims = U @ U.T
            np.fill_diagonal(sims, -1.0)
            dup += int((sims.max(axis=1) >= self.reorg_duplicate).sum())
            total += len(U)
        return dup / total if total else 0.0

    def should_reorganize(self):
        return (self._since_reorg >= self.maintain_gap
                and (self.redundancy() > self.redundancy_floor
                     or self.surprise > self.surprise_floor))

    def reorganize(self, duplicate=None):
        """Fold near-duplicate prototypes (per action) into one, combining their
        returns by count -- the MergeExpert principle from the data organizer,
        applied to the brain's own value memory. Folding the duplicates both shrinks
        the memory and, crucially, lets the merged prototype actually track change:
        one prototype that every update touches can be unlearned, where a cloud of
        duplicates cannot. Returns (before, after) prototype counts."""
        dup = self.reorg_duplicate if duplicate is None else duplicate
        before = self.prototype_count()
        for a in range(len(self.actions)):
            U, S, R, C = self._unit[a], self._sum[a], self._ret[a], self._cnt[a]
            ks, kr, kc = [], [], []
            for i in range(len(U)):
                placed = False
                for j in range(len(ks)):
                    ku = ks[j] / (np.linalg.norm(ks[j]) or 1.0)
                    if float(ku @ U[i]) >= dup:
                        tot = kc[j] + C[i]
                        kr[j] = (kr[j] * kc[j] + R[i] * C[i]) / tot
                        ks[j] = ks[j] + S[i]; kc[j] = tot
                        placed = True; break
                if not placed:
                    ks.append(S[i].copy()); kr.append(float(R[i])); kc.append(float(C[i]))
            if ks:
                self._sum[a] = np.stack(ks)
                self._unit[a] = self._sum[a] / (np.linalg.norm(self._sum[a], axis=1,
                                                               keepdims=True) + 1e-12)
                self._ret[a] = np.array(kr); self._cnt[a] = np.array(kc)
        self._since_reorg = 0
        self.surprise = 0.0
        self.reorganizations += 1
        return before, self.prototype_count()

    # -- fully autonomous variant: no thresholds, decide by measured fit ---------
    def _state_dim(self):
        """The width of the space the brain currently operates in: the basis rank after
        consolidate()/_expand_basis(), else the raw dim. Candidate memories must be
        built at THIS width, because the recent-experience buffer they are rebuilt from
        already lives in the projected space."""
        return self._basis.shape[1] if self._basis is not None else self.dim

    def _blank(self):
        d = self._state_dim()
        m = HolographicMind(self.dim, self.actions, k=self.k, merge=self.merge,
                            ret_alpha=self.ret_alpha, capacity=self.capacity,
                            robust_returns=self.robust_returns)
        if d != self.dim:
            # the buffer's states live in the projected space, so the candidate's empty
            # prototype banks must have that width too (otherwise the first _absorb
            # vstacks a d-wide state onto a self.dim-wide empty array and mismatches)
            m._sum = [np.zeros((0, d)) for _ in range(len(self.actions))]
            m._unit = [np.zeros((0, d)) for _ in range(len(self.actions))]
            m._basis = self._basis                     # same subspace, so value()/decide line up
        return m

    def _clone(self, src=None):
        """A copy of a memory (this brain's, or a candidate's) we can fold freely."""
        src = self if src is None else src
        m = self._blank()
        m._sum = [x.copy() for x in src._sum]; m._unit = [x.copy() for x in src._unit]
        m._ret = [x.copy() for x in src._ret]; m._cnt = [x.copy() for x in src._cnt]
        m._ret_dev = src._ret_dev                  # D2: carry the running robust scale into the clone
        return m

    def _rebuilt_from(self, experiences):
        """A memory built fresh from a slice of recent experience -- i.e. the brain
        with the stale regime forgotten."""
        m = self._blank()
        for s, a, r in experiences:
            m._absorb(s, a, r)
        return m

    def _greedy(self, mind, s):
        return int(np.argmax([mind.value(s, a)[0] for a in range(len(self.actions))]))

    def _policy_value(self, mind, val):
        """Off-policy estimate of how good a candidate's GREEDY decisions are: on the
        held-out experiences where the candidate would have taken the action that was
        actually logged, average the reward that action actually got. This judges
        decisions, not value magnitudes -- so folding that preserves the argmax costs
        nothing, which is what lets compression happen. Returns (mean, rewards)."""
        rew = [r for s, a, r in val if self._greedy(mind, s) == a]
        return (float(np.mean(rew)) if rew else 0.0), np.array(rew)

    def auto_maintain(self, grains=None, refresh=None):
        """Speculate, measure, adopt -- with no behavioural thresholds. We build two
        families of candidate memories and judge them on a held-out slice of recent
        experience:

          * PRESERVING -- keep, or fold the duplicates at a couple of grains. These
            retain every situation the brain has seen; folding only compresses.
          * REFRESHING -- rebuild from recent experience (optionally folded). These
            FORGET the old regime, which is what you want after a shift but wasteful
            otherwise.

        Refreshing is allowed to win as soon as it predicts recent reality better than
        the best preserving option -- the costs are asymmetric, so it does not have to
        win by a margin. A needless refresh in a stable world merely rebuilds from
        recent (still-valid) experience and costs nothing measurable; a MISSED refresh
        after a shift strands the brain on a stale policy. (An earlier one-standard-
        error margin here was too timid: on hard, noisy shifts it left the gate sitting
        on 'keep' while the world had already moved, because right after a shift the
        recent window still holds enough old experience to flatter the stale memory.)
        When a refresh is chosen it is refit on the FULL recent window, not just the fit
        slice it was selected on -- the same "select on held-out, deploy on all the
        (recent) data" discipline the organizer uses, kept leakage-free by rebuilding
        only after the choice is made. Within the winning family we still take the
        leanest candidate that is statistically as good as the best. So a regime shift
        draws a refresh; a quiet stretch draws a fold; neither is tuned for.

        BUDGET (Request 2): `grains` is the tuple of fold grains tried in each family
        (default: the instance's self.maintain_grains); `refresh` toggles the refreshing
        family (default: self.maintain_refresh). Both default to the full 8-way search.
        A quiet deployment can pass fewer grains and refresh=False to score as few as one
        candidate per tick -- the caller's call, since (as the constructor notes) surprise
        is not a reliable auto-signal. With refresh=False the brain stops watching for a
        regime shift, so a stable courier that turns it off should periodically run a full
        tick (plain auto_maintain()) to re-check, exactly the asymmetric cost above."""
        grains = self.maintain_grains if grains is None else tuple(grains)
        refresh = self.maintain_refresh if refresh is None else bool(refresh)
        self._added = 0
        buf = list(self._buf)
        self.rng.shuffle(buf)
        cut = int(len(buf) * 0.7)
        fit, val = buf[:cut], buf[cut:]
        if len(val) < 20:
            return

        preserving = [("keep", self._clone())]
        for g in grains:
            c = self._clone(); c.reorganize(duplicate=g)
            preserving.append((f"fold@{g}", c))
        if refresh:
            base = self._rebuilt_from(fit)
            refreshing = [("refresh", base)]
            for g in grains:
                c = self._clone(base); c.reorganize(duplicate=g)
                refreshing.append((f"refresh+fold@{g}", c))
        else:
            refreshing = []                     # stable-deployment budget: don't pay to forget

        def score(group):
            return [(name, m, (pv := self._policy_value(m, val))[0], pv[1]) for name, m in group]
        P, R = score(preserving), score(refreshing)
        bestP = max(P, key=lambda z: z[2])
        bestR = max(R, key=lambda z: z[2]) if R else None      # R empty when refresh=False
        if bestR is not None and bestR[2] > bestP[2]:   # recent decisions better -> the world moved
            group, best = R, bestR
        else:                                    # compress without forgetting (or no refresh family)
            group, best = P, bestP
        # 2-sigma band read off the WINNING family's pooled rewards (not chosen by hand)
        se = best[3].std() / np.sqrt(len(best[3])) if len(best[3]) > 1 else 0.0
        # within the winning family, take the leanest whose decisions are statistically
        # as good as the best
        pool = [z for z in group if z[2] >= best[2] - 2 * se]
        chosen = min(pool, key=lambda z: z[1].prototype_count())

        self.last_choice = chosen[0]
        if chosen[0] != "keep":
            m = chosen[1]
            if chosen[0].startswith("refresh"):  # deploy refit on the FULL recent window
                m = self._rebuilt_from(buf)
                if "@" in chosen[0]:
                    m.reorganize(duplicate=float(chosen[0].split("@")[1]))
            self._sum, self._unit = m._sum, m._unit
            self._ret, self._cnt = m._ret, m._cnt
            # the recent-experience buffer was recorded against the OLD prototype banks
            # (and possibly an old basis width); after a swap/refresh it no longer maps
            # onto the new banks, so clear it rather than leave stale (s, a) entries that
            # an online consumer like penalize_recent would have to second-guess.
            self._buf = []
            self.reorganizations += 1
        return chosen[0], chosen[1].prototype_count()

    # -- persistence: round-trip a TRAINED brain --------------------------------
    # Saves the learned value memory (the four per-action banks), the projection
    # basis if the brain has consolidated, and the config needed to reconstruct an
    # identical decision-maker. The recent-experience buffer and transient EMAs are
    # deliberately NOT persisted -- they are self-healing scratch state that refills
    # as the reloaded brain sees new experience, and saving them would only risk the
    # stale-buffer hazards the maintenance code already guards against.
    _STATE_FIELDS = ("k", "epsilon", "novelty_bonus", "memory_cap", "merge", "ret_alpha",
                     "maintain", "reorg_duplicate", "redundancy_floor", "surprise_floor",
                     "maintain_gap", "buffer_cap", "check_every", "capacity",
                     "blind_floor", "expand_at", "experiences", "reorganizations", "robust_returns")

    def to_state(self):
        """A snapshot of this trained brain: config + the learned per-action banks +
        the consolidation basis. Reload with HolographicMind.from_state()."""
        cfg = {f: getattr(self, f) for f in self._STATE_FIELDS}
        return {
            "kind": "HolographicMind",
            "dim": int(self.dim),
            "actions": list(self.actions),
            "seed": int(self._seed) if hasattr(self, "_seed") else 0,
            "config": cfg,
            "sum": [a.copy() for a in self._sum],
            "unit": [a.copy() for a in self._unit],
            "ret": [a.copy() for a in self._ret],
            "cnt": [a.copy() for a in self._cnt],
            "basis": (self._basis.copy() if self._basis is not None else None),
        }

    @classmethod
    def from_state(cls, state):
        """Rebuild a trained HolographicMind from a to_state() snapshot. The reloaded
        brain decides identically to the saved one (same banks, same basis)."""
        cfg = dict(state.get("config", {}))
        m = cls(int(state["dim"]), list(state["actions"]), seed=int(state.get("seed", 0)),
                k=cfg.get("k", 12), merge=cfg.get("merge", 0.92),
                ret_alpha=cfg.get("ret_alpha", 0.1), capacity=cfg.get("capacity", 0),
                maintain=cfg.get("maintain", False))
        for f in cls._STATE_FIELDS:
            if f in cfg:
                setattr(m, f, cfg[f])
        m._sum = [np.asarray(a, float) for a in state["sum"]]
        m._unit = [np.asarray(a, float) for a in state["unit"]]
        m._ret = [np.asarray(a, float) for a in state["ret"]]
        m._cnt = [np.asarray(a, float) for a in state["cnt"]]
        m._basis = (np.asarray(state["basis"], float) if state.get("basis") is not None else None)
        return m


# ---------------------------------------------------------------------------
# 2. THE SENSES  (the from-scratch encoder -- raw world -> one vector)
# ---------------------------------------------------------------------------

class CreatureEncoder:
    """Turn the creature's egocentric senses into a single unit vector -- the creature DOMAIN's encoder.

    RELATION TO THE ONE ENCODER (read first, so this is never mistaken for a stray duplicate): the system's
    general encoder is UnifiedMind.perceive (UniversalEncoder), and through UnifiedMind the brain/memory get
    their vectors from it -- they never encode themselves. THIS class is the standalone creature domain's
    encoder, used by the creature's own training/demo harness and by creature-style agents that deliberately
    reuse the same machinery in another world (holographic_navigator's "inception" demo, app's maze console,
    holographic_lookahead). It is NOT redundant with perceive: it does role/filler binding (the shared
    primitive) PLUS two things perceive does not -- build_state's action-memory (below) and the `seen`
    tracking that HolographicMind.describe needs to decode a state back into sense terms. And the rescue
    canary is tie-sensitive to its EXACT output (see the kept-negative in encode()). So it stays a focused,
    deliberate component, the same way the per-action value memory does (measured, exp_value_memory.py).

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
        # SELF-DISCOVERY: the encoder remembers which roles and values its own
        # experience has contained -- this becomes the cleanup vocabulary that
        # lets the brain DESCRIBE its prototypes back in sense terms (see
        # HolographicMind.describe). Never declared, only observed.
        self.seen = {}                                   # role -> set of values

    def encode(self, senses):
        """Senses dict -> one unit vector (zero vector if the creature senses
        nothing at all, e.g. blind in open space)."""
        if not senses:
            return np.zeros(self.vocab.dim)
        for role, value in senses.items():
            self.seen.setdefault(role, set()).add(value)
        # NOTE (kept negative): bind_batch here is ~1.4x faster and identical to the loop at
        # 1e-12, BUT batched vs scalar FFT differ at ~1e-16, and that is enough to flip a
        # knife-edge tie-break in the starved-maze rescue trajectory -- the rescue_cracks
        # canary fails. The creature's deterministic reproducibility outweighs a per-step
        # 1.4x, so the per-sense loop stays. (RecordEncoder tolerates the same change because
        # classification is wide-margin; the maze rescue is tie-sensitive. See NOTES.)
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


class FastCreatureEncoder(CreatureEncoder):
    """Compiled, fully in-VSA perception: the per-step role/filler BIND (an FFT convolution) is the last
    Python<->VSA boundary cost in the perceive loop, and it recomputes the SAME bind every time a
    (role, value) feature recurs. This caches each bound atom the first time it is seen, so after a brief
    warm-up perception is a GATHER + bundle (a sum) -- pure array ops, no per-step FFT. build_state inherits
    this `encode` unchanged, so the whole maze loop (perceive -> decide -> learn) runs without a per-step
    convolution.

    BIT-IDENTICAL to CreatureEncoder.encode: it caches the EXACT same scalar bind and bundles in the same
    sorted order, so the output vector is unchanged -- only the redundant FFTs are skipped. (Kept as an
    opt-in subclass anyway, so the tie-sensitive rescue canary keeps using the plain encoder by default.)
    `perception_codebook()` exposes the cached atoms as one (n_features, dim) matrix -- the compiled codebook
    that turns perception into a single gather/matmul, the in-VSA form of the senses dict.
    """

    def __init__(self, dim, seed=1):
        super().__init__(dim, seed)
        self._bind_cache = {}                            # (role, value) -> precomputed bound atom
        self.binds_done = 0                              # FFT binds actually computed (warm-up cost)
        self.binds_saved = 0                             # FFT binds avoided by the cache (the steady-state win)

    def encode(self, senses):
        if not senses:
            return np.zeros(self.vocab.dim)
        for role, value in senses.items():
            self.seen.setdefault(role, set()).add(value)
        tokens = []
        for role, value in sorted(senses.items()):       # same sorted order as the base encoder
            t = self._bind_cache.get((role, value))
            if t is None:
                t = bind(self.vocab.get(role), self.vocab.get(value))   # compute the convolution ONCE
                self._bind_cache[(role, value)] = t
                self.binds_done += 1
            else:
                self.binds_saved += 1
            tokens.append(t)
        return bundle(tokens)

    def perception_codebook(self):
        """The compiled codebook: (features, dim) matrix of cached bound atoms + the feature key list. With it,
        perceiving a set of active features is one gather-and-sum (or an indicator @ matrix) -- perception as
        a single array op, entirely inside the holographic space."""
        keys = sorted(self._bind_cache.keys())
        return np.stack([self._bind_cache[k] for k in keys]) if keys else np.zeros((0, self.vocab.dim)), keys


# ---------------------------------------------------------------------------
# 3. THE WORLD  (a tiny foraging grid -- the creature's body and environment)
# ---------------------------------------------------------------------------

class GridWorld:
    """A small grid with one creature, one star (food), some poison cells, and
    -- optionally -- impassable WALLS.

    The creature runs on an energy battery -- the survival mechanic that gives
    the foraging a point:

      * it starts each life with START_ENERGY (300 by default -- raised from
        100 when the 16x16 gauntlet showed optimal paths of 80-108 steps,
        i.e. the old battery starved the creature even on a perfect run),
      * every move drains MOVE_ENERGY (1), so it is slowly dying just by living,
      * reaching a star refills it by STAR_ENERGY (3) and the star respawns,
      * stepping onto poison empties the battery at once -- instant death.

    Stars are the ONLY way to top the battery up, so "collect as many stars as
    you can" and "stay alive as long as you can" are the very same goal. The life
    ends the instant the creature touches poison or the battery hits empty.

    Two obstacle modes sit on top of the same machinery:

      * n_walls > 0 scatters that many random impassable walls into the forage
        world (kept connected, so the world is always solvable) -- now the
        creature must route AROUND obstacles to reach a star, not just head
        straight for it.
      * maze=True carves a proper labyrinth (a 'perfect' maze: exactly one route
        between any two cells), drops the creature at one end and the EXIT at the
        far end, and the goal becomes finding the way out. Reaching the exit ends
        the life as an ESCAPE (a win), rewarded by EXIT.

    Walls differ from poison: a wall is impassable (you stay put if you try to
    enter it, just like the grid edge) and is sensed as 'wall_<dir>'; poison can
    be entered but kills, and is sensed as 'danger_<dir>'.

    The reward channel (STEP / FOOD / POISON / EXIT) is kept on its own small,
    well-tested scale -- it is the LEARNING signal, deliberately separate from
    the integer energy the game runs on. They point the same way (stars/exit
    good, poison fatal, moving mildly costly), so a brain trained to maximise
    reward also plays the energy game well.

    Coordinates use y growing DOWNWARD, so North is y-1.
    """

    MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
    ACTIONS = ["N", "S", "E", "W"]

    def __init__(self, width=7, height=7, n_poison=2, seed=0,
                 step_cost=-0.01, food_reward=1.0, poison_reward=-1.0,
                 start_energy=300, move_energy=1, star_energy=3,
                 vision_radius=None, n_walls=0, maze=False, exit_reward=3.0,
                 fixed_seed=None, braid=0.0, maze_poison=0):
        self.w, self.h, self.n_poison = width, height, n_poison
        self.STEP, self.FOOD, self.POISON = step_cost, food_reward, poison_reward
        self.EXIT = exit_reward               # reward for reaching the maze exit
        # The energy mechanic (see the class docstring above).
        self.start_energy = start_energy     # full battery at birth
        self.move_energy = move_energy        # drained per move
        self.star_energy = star_energy        # refilled per star
        self.vision_radius = vision_radius    # None = always sees food's direction
        self.n_walls = n_walls                # random obstacles in forage mode
        self.maze = maze                      # carve a labyrinth instead?
        self.braid = braid                    # fraction of dead-ends to open (0 = perfect maze)
        self.maze_poison = maze_poison        # hazards in a BRAIDED maze (safe route kept)
        # When set, every reset() rebuilds the SAME layout (we reseed the world's
        # rng from it). That is how the creature gets to LEARN one fixed maze over
        # many tries -- the classic rat-in-a-maze setup -- instead of facing a new
        # random labyrinth every episode.
        self.fixed_seed = fixed_seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    # -- setup ------------------------------------------------------------
    def reset(self):
        if self.fixed_seed is not None:      # rebuild the identical layout each life
            self.rng = np.random.default_rng(self.fixed_seed)
        self.walls = set()
        self.escaped = False                 # set True when a maze is solved
        if self.maze:
            self._carve_maze()               # fills self.walls; passages are the rest
            free = self._free_cells()
            start = (1, 1) if (1, 1) not in self.walls else min(free)
            self.cx, self.cy = start
            dist = self._bfs_dist(start)     # exit = the deepest cell in the maze
            self.fx, self.fy = max(free, key=lambda c: dist.get(c, -1))
            # THE POISONED FORK: hazards in a maze make sense only when an
            # alternative route exists (in a PERFECT maze every corridor is the
            # only way somewhere, so poison would simply make it unsolvable).
            # With braiding on, maze_poison cells are placed one at a time and
            # each is kept only if a poison-free route from start to exit still
            # exists -- the maze stays honest: solvable, but one arm of some fork
            # is deadly and looks just like the safe one.
            self.poison = set()
            if self.maze_poison > 0 and self.braid > 0:
                exit_c = (self.fx, self.fy)
                near_start = {(self.cx + dx, self.cy + dy)
                              for dx, dy in self.MOVES.values()} | {start}
                cand = [c for c in free if c != exit_c and c not in near_start]
                self.rng.shuffle(cand)
                for c in cand:
                    if len(self.poison) >= self.maze_poison:
                        break
                    trial = self.poison | {c}
                    if self._route_exists(start, exit_c, blocked=trial):
                        self.poison = trial
        else:
            self._scatter_walls(self.n_walls)
            self.poison = set()
            while len(self.poison) < self.n_poison:
                self.poison.add(self._random_cell(avoid=self.poison))
            self.cx, self.cy = self._random_cell(avoid=self.poison)
            self._spawn_food()
        self.energy = self.start_energy      # battery starts full
        self.age = 0                         # steps lived this life
        self.stars = 0                       # stars collected (or 1 once escaped)
        self.alive = True                    # poison or an empty battery ends it
        return self.senses()

    def _random_cell(self, avoid=()):
        """A uniformly random FREE cell -- never a wall, and never in `avoid`."""
        avoid = set(avoid) | self.walls
        while True:
            c = (int(self.rng.integers(self.w)), int(self.rng.integers(self.h)))
            if c not in avoid:
                return c

    def _spawn_food(self):
        blocked = set(self.poison) | {(self.cx, self.cy)}
        self.fx, self.fy = self._random_cell(avoid=blocked)

    # -- walls, mazes, and routes ----------------------------------------
    def _free_cells(self):
        """Every passable (non-wall) cell."""
        return [(x, y) for x in range(self.w) for y in range(self.h)
                if (x, y) not in self.walls]

    def _route_exists(self, a, b, blocked=()):
        """Is there a passable route a -> b that avoids `blocked` cells?"""
        from collections import deque
        blocked = set(blocked)
        seen, q = {a}, deque([a])
        while q:
            x, y = q.popleft()
            if (x, y) == b:
                return True
            for dx, dy in self.MOVES.values():
                nxt = (x + dx, y + dy)
                if (0 <= nxt[0] < self.w and 0 <= nxt[1] < self.h
                        and nxt not in self.walls and nxt not in blocked
                        and nxt not in seen):
                    seen.add(nxt)
                    q.append(nxt)
        return False

    def _bfs_dist(self, start):
        """Breadth-first step-distances from `start` over passable cells."""
        from collections import deque
        dist = {start: 0}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for dx, dy in self.MOVES.values():
                nxt = (x + dx, y + dy)
                if (0 <= nxt[0] < self.w and 0 <= nxt[1] < self.h
                        and nxt not in self.walls and nxt not in dist):
                    dist[nxt] = dist[(x, y)] + 1
                    q.append(nxt)
        return dist

    def shortest_path(self, start, goal):
        """The shortest passable route from start to goal, as a list of cells
        including both ends (or [] if there is none). Plain BFS -- this is the
        'optimal route' we draw behind the creature so you can see how close its
        learned path came to the best possible one."""
        from collections import deque
        if start == goal:
            return [start]
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            x, y = cur
            for dx, dy in self.MOVES.values():
                nxt = (x + dx, y + dy)
                if (0 <= nxt[0] < self.w and 0 <= nxt[1] < self.h
                        and nxt not in self.walls and nxt not in prev):
                    prev[nxt] = cur
                    q.append(nxt)
        if goal not in prev:
            return []
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        return path[::-1]

    def _all_free_connected(self):
        """True if every open cell can reach every other open cell."""
        free = self._free_cells()
        return bool(free) and len(self._bfs_dist(free[0])) == len(free)

    def _scatter_walls(self, n):
        """Drop up to n random impassable walls -- but keep a wall only if the
        open space stays fully connected afterward. That guarantees the world is
        always solvable and no star can spawn in a sealed-off pocket. We give up
        after a bounded number of tries so a too-dense request can't loop."""
        self.walls = set()
        attempts = 0
        while len(self.walls) < n and attempts < n * 30 + 30:
            attempts += 1
            c = (int(self.rng.integers(self.w)), int(self.rng.integers(self.h)))
            if c in self.walls:
                continue
            self.walls.add(c)
            if not self._all_free_connected():
                self.walls.discard(c)        # this one would wall off a region

    def _carve_maze(self):
        """Carve a 'perfect' maze (exactly one route between any two cells) with
        a depth-first recursive backtracker. Start completely solid, then tunnel:
        from a passage cell, jump two cells to an unvisited neighbour and knock
        out the wall in between. Needs odd width/height so the passages land on
        the lattice and the outer ring stays as the maze's border."""
        self.walls = {(x, y) for x in range(self.w) for y in range(self.h)}
        start = (1, 1)
        self.walls.discard(start)
        stack, visited = [start], {start}
        while stack:
            x, y = stack[-1]
            nbrs = [(x + dx, y + dy, dx, dy)
                    for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
                    if 0 <= x + dx < self.w and 0 <= y + dy < self.h
                    and (x + dx, y + dy) not in visited]
            if not nbrs:
                stack.pop()
                continue
            nx, ny, dx, dy = nbrs[int(self.rng.integers(len(nbrs)))]
            self.walls.discard((x + dx // 2, y + dy // 2))   # knock out the wall between
            self.walls.discard((nx, ny))
            visited.add((nx, ny))
            stack.append((nx, ny))

        # Optional braiding: a perfect maze is all dead-ends and a single tortuous
        # route, which is brutal for a reactive brain. Opening a fraction of the
        # dead-ends adds loops and alternative routes -- still clearly a maze, but
        # one the creature has a fair chance of learning to escape.
        if self.braid > 0:
            interior = lambda c: 0 < c[0] < self.w - 1 and 0 < c[1] < self.h - 1
            dead_ends = [c for c in self._free_cells()
                         if interior(c) and self._passable_degree(c) == 1]
            for c in dead_ends:
                if self.rng.random() >= self.braid:
                    continue
                x, y = c                                     # open a wall that leads somewhere new
                cand = [(x + dx, y + dy) for dx, dy in self.MOVES.values()
                        if interior((x + dx, y + dy)) and (x + dx, y + dy) in self.walls]
                if cand:
                    self.walls.discard(cand[int(self.rng.integers(len(cand)))])

    def _passable_degree(self, cell):
        """How many passable neighbours a cell has (1 == a dead-end)."""
        x, y = cell
        return sum((x + dx, y + dy) not in self.walls
                   and 0 <= x + dx < self.w and 0 <= y + dy < self.h
                   for dx, dy in self.MOVES.values())

    # -- perception -------------------------------------------------------
    def _axis(self, delta, neg, pos):
        return neg if delta < 0 else pos if delta > 0 else "none"

    def senses(self):
        """Egocentric, SPARSE description of the situation.

        food_x / food_y say where the star (or maze exit) is relative to us --
        but only when it is within vision_radius (if set); beyond that the
        creature is 'blind' to it and those keys are absent. Walls (a grid edge
        OR an impassable wall cell) and danger (adjacent poison) are reported per
        direction, and only when present. We never emit 'no wall here': a token
        that appears in almost every state makes unrelated situations look alike
        and drowns out the signal that matters.
        """
        senses = {}
        dist = abs(self.fx - self.cx) + abs(self.fy - self.cy)
        if self.vision_radius is None or dist <= self.vision_radius:
            senses["food_x"] = self._axis(self.fx - self.cx, "west", "east")
            senses["food_y"] = self._axis(self.fy - self.cy, "north", "south")
        for d, (dx, dy) in self.MOVES.items():
            nx, ny = self.cx + dx, self.cy + dy
            if not (0 <= nx < self.w and 0 <= ny < self.h) or (nx, ny) in self.walls:
                senses["wall_" + d] = "yes"   # grid edge or an impassable wall
            elif (nx, ny) in self.poison:
                senses["danger_" + d] = "yes"
        return senses

    # -- dynamics ---------------------------------------------------------
    def step(self, action_name):
        """Apply an action; return (senses, reward, ate_star, done).

        Order of bookkeeping matters. The move is paid for FIRST -- every step
        burns MOVE_ENERGY, even one that bumps a wall -- and only then do we look
        at what the creature stepped onto:

          * A wall (grid edge or wall cell) is impassable: the creature stays put.
          * Poison is LETHAL: step on it and the battery is zeroed and the life is
            over, right where it stands.
          * The star: in forage mode it refuels the battery by STAR_ENERGY and
            respawns; in maze mode it is the EXIT, so reaching it ends the life as
            an escape (a win).
          * If the battery reaches empty, the creature dies of exhaustion.

        'done' is True on any terminal: death, or a maze escape.
        """
        if not self.alive:                                # already dead/escaped: no-op
            return self.senses(), 0.0, False, True

        dx, dy = self.MOVES[action_name]
        nx, ny = self.cx + dx, self.cy + dy
        reward = self.STEP                                # cost of living
        self.age += 1                        # one more step lived
        self.energy -= self.move_energy                   # moving drains the battery

        if not (0 <= nx < self.w and 0 <= ny < self.h) or (nx, ny) in self.walls:
            nx, ny = self.cx, self.cy                     # wall or edge: stay put
        elif (nx, ny) in self.poison:                     # poison: step on it and die
            self.cx, self.cy = nx, ny
            self.energy = 0
            self.alive = False
            reward += self.POISON
            return self.senses(), reward, False, True

        self.cx, self.cy = nx, ny
        ate = (self.cx, self.cy) == (self.fx, self.fy)
        if ate:
            self.stars += 1
            if self.maze:                                 # reached the exit: escaped!
                reward += self.EXIT
                self.escaped = True
                self.alive = False                        # the life's task is done
                return self.senses(), reward, True, True
            reward += self.FOOD                           # a star: refuel and respawn
            self.energy += self.star_energy
            self._spawn_food()

        if self.energy <= 0:                              # battery empty: death by exhaustion
            self.energy = 0
            self.alive = False
            return self.senses(), reward, ate, True

        return self.senses(), reward, ate, False

    # -- a peek at the world ---------------------------------------------
    def render(self):
        goal = "exit" if self.maze else "star"
        status = (f"energy {self.energy:3d}/{self.start_energy}   {goal}s {self.stars}"
                  + ("   [ESCAPED]" if self.escaped else "" if self.alive else "   [DEAD]"))
        rows = [status]
        for y in range(self.h):
            row = ""
            for x in range(self.w):
                if (x, y) == (self.cx, self.cy):
                    row += "C"
                elif (x, y) == (self.fx, self.fy):
                    row += "E" if self.maze else "*"
                elif (x, y) in self.walls:
                    row += "#"
                elif (x, y) in self.poison:
                    row += "x"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# 4. THE LOOP  (live an episode; optionally learn from it)
# ---------------------------------------------------------------------------

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _forced_dir(senses, last_action):
    """If the creature stands in a CORRIDOR -- exactly two open directions, one of
    them where it just came from -- return the only way forward, else None. Uses
    nothing but the wall senses the brain already gets.

    One override, found by the poisoned-fork gauntlet: if the way forward is
    sensed as DANGER, the reflex yields and hands control back to the brain.
    Without it the auto-walk can march the creature into poison it can see --
    measured on the poisoned fork (braided 11x11, 3 hazards, safe route exists):
    the naive reflex with a trained brain still died in 7% of evaluations and
    escaped 93%; with the yield, deaths 0% and escapes 100%, three seeds. The
    system lesson runs in both directions here: a fast path must hand back
    control at anomalies (the reflex cache's flux guard, the gate falling back
    when it has no corpus), and so must this one."""
    opens = [d for d in GridWorld.ACTIONS if f"wall_{d}" not in senses]
    fwd = [d for d in opens if d != OPPOSITE.get(last_action)]
    if len(opens) == 2 and len(fwd) == 1 and f"danger_{fwd[0]}" not in senses:
        return fwd[0]
    return None


def run_episode(world, encoder, mind, learn=True, explore=True,
                eval_epsilon=None, gamma=0.9, max_steps=50, mem=0,
                corridor_reflex=False, danger_reflex=False, wall_reflex=False,
                curiosity=0.0, return_trajectory=False):
    """Live one episode; return (total_reward, stars_collected).

    max_steps=None is SURVIVAL MODE: the life runs until the creature dies --
    poison or an empty battery -- with no step cap (a hard safety ceiling of
    10,000 steps guards against a truly immortal loop). This is the harsh
    framing for foraging: collect as many stars as possible without dying.
    The energy arithmetic makes every life finite anyway (stars give +3, moves
    cost 1, and the average star is ~4.7 steps away on a 7x7, so even a perfect
    forager runs a slow deficit) -- and it surfaces what short caps mask:
    per-step risks COMPOUND over a long life, so a policy with a 1%% chance per
    step of touching poison looks fine in a 50-step test and is dead by ~300.

    The episode otherwise ends at max_steps OR the moment the creature dies --
    whichever comes first. 'stars' is the honest success metric: how many stars
    it reached before its life ended.

    'mem' is the working-memory depth: how many recent moves the creature folds
    into its state (0 = purely reactive). If learning, fold the experience back
    in afterward using Monte-Carlo returns (each action credited with the
    discounted reward that actually followed it -- simple, no bootstrapping).

    'curiosity' (off by default) is the BOOTSTRAP fix for worlds too deep for
    luck: each cell earns a small intrinsic bonus the FIRST time it is visited
    this episode, so exploration is rewarded densely instead of only at the
    (possibly never-reached) exit. The honest magnitude is the world's own
    arithmetic -- exit_reward / n_free_cells -- so touring the ENTIRE maze sums
    to exactly one exit reward and curiosity can never dominate the real signal
    once the exit is found. Measured: under the standard decaying-epsilon
    schedule, 20x20 seed 11 trains to ZERO escapes (the loop-attractor policy
    locks in before luck finds the exit; sustained high epsilon occasionally
    escapes but never consolidates -- plain probes 0% at every budget tried);
    curiosity turns the early walk into a covering walk that finds the first
    success while exploration is still high.

    'corridor_reflex' is the DECIDE-ONLY-AT-CHOICES lesson, imported from the
    rest of the system (exact scan below the crossover, the gate only where type
    goes blind -- machinery only where there is a real choice). When on, corridor
    cells are auto-walked by a pure sense-driven reflex arc and the BRAIN spends
    its decisions -- and its credit -- only at junctions and dead ends. This is
    what broke the 9x9 maze ceiling: per-step framing discounted a 26-step exit
    to gamma^26 ~ 0.07 at the first decision (nearly invisible), while junction
    granularity puts it near 0.4 (learnable). Measured, 3 seeds each: 9x9 escapes
    went 0% -> 100%, 11x11 100%, 13x13 67% -- against an honest control of the
    reflex with RANDOM junction choices (73% / 67% / 15%), so the brain's
    contribution is real at every size and triples the hardest one. Default off:
    foraging worlds are open space, and the reflex is a maze-shaped prior."""
    if max_steps is None:
        max_steps = 10000                                # survival: death decides
    senses = world.reset()
    recent = []                                          # recent actions, newest first
    state = encoder.build_state(senses, recent, mem)
    states, actions, rewards = [], [], []
    _visited = {(world.cx, world.cy)}                    # for the curiosity bonus
    stars = 0
    steps = 0
    last = None

    while steps < max_steps:
        # THE DANGER REFLEX: lethal moves are vetoed BELOW the brain. Found by
        # the survival bench: a brain whose poison avoidance looked solid in
        # 50-step tests died on poison in 67-73% of full LIVES (a ~0.6%%/step
        # residual risk compounds), collecting 13-25 stars where a two-line
        # danger-aware reflex collected 136 with zero deaths. Irreversible
        # mistakes belong to reflexes, not learned preferences -- the corridor
        # reflex's danger yield and auto_maintain's asymmetric-cost rule, again.
        # The brain still does all the foraging; the veto only removes suicide
        # from the menu (decide's `among` -- the routing lesson, for actions).
        # The safety reflexes (danger and wall vetoes) live in the BRAIN now
        # (see HolographicMind.decide's `senses`/`avoid` -- both measured: the
        # danger veto ended compounding poison deaths 67-73% -> 0%, the wall
        # veto -- named by the brain's own introspection -- solved the
        # cluttered-world open problem, stars 5.1 -> 19.8). These flags just
        # translate to the model-level mechanism, so every other caller of the
        # brain gets identical safety by passing what the creature senses.
        avoid = tuple([p for p, on in (("danger", danger_reflex),
                                       ("wall", wall_reflex)) if on])
        a = mind.decide(state, explore=explore, epsilon=eval_epsilon,
                        senses=(senses if avoid else None), avoid=avoid)
        name = mind.actions[a]
        senses, r, ate, done = world.step(name)
        stars += int(ate)
        steps += 1
        recent = [name] + recent                         # remember this move
        last = name
        r_total = r
        if curiosity and (world.cx, world.cy) not in _visited:
            _visited.add((world.cx, world.cy))
            r_total += curiosity                         # first visit this episode
        if corridor_reflex:
            while not done and steps < max_steps:
                fwd = _forced_dir(senses, last)
                if fwd is None:                          # junction/dead end: brain's turn
                    break
                senses, r, ate, done = world.step(fwd)   # forced cell: reflex walks it
                stars += int(ate)
                steps += 1
                recent = [fwd] + recent
                last = fwd
                r_total += r
                if curiosity and (world.cx, world.cy) not in _visited:
                    _visited.add((world.cx, world.cy))
                    r_total += curiosity                 # forced cells count too
        states.append(state)
        actions.append(a)
        rewards.append(r_total)                          # the DECISION earns the segment
        state = encoder.build_state(senses, recent, mem)
        if done:                                         # poison death or empty battery
            break

    if learn:
        returns, g = [0.0] * len(rewards), 0.0
        for t in reversed(range(len(rewards))):          # reward-to-go
            g = rewards[t] + gamma * g
            returns[t] = g
        mind.remember(states, actions, returns)
        if return_trajectory:                            # for success REHEARSAL
            return float(np.sum(rewards)), stars, (states, actions, returns)

    return float(np.sum(rewards)), stars


# ---------------------------------------------------------------------------
# 5. DEMO
# ---------------------------------------------------------------------------

def _train(world, encoder, mind, episodes, eps_start=0.35, block=25, label="",
           mem=0, max_steps=50, danger_reflex=False, wall_reflex=False):
    """Train for some episodes with decaying exploration, printing the average
    reward of each block so the learning curve is visible. The reflex flags
    pass through to run_episode (and so to the brain's own senses-based veto)
    -- training WITH the vetoes matters as much as playing with them: the
    cluttered-world fix measured 19.8 vs 5.1 stars precisely because the veto
    shaped the experience the brain learned from, not just the final moves."""
    rewards = []
    print(f"Training{label} (avg reward per {block}-episode block):")
    for ep in range(episodes):
        mind.epsilon = max(0.05, eps_start * (1.0 - ep / episodes))
        r, _ = run_episode(world, encoder, mind, learn=True, explore=True,
                           mem=mem, max_steps=max_steps,
                           danger_reflex=danger_reflex, wall_reflex=wall_reflex)
        rewards.append(r)
        if (ep + 1) % block == 0:
            print(f"  episodes {ep - block + 2:3d}-{ep + 1:3d}:  "
                  f"{np.mean(rewards[-block:]):+.2f}")


def _evaluate(world, encoder, mind, n=40, mem=0, max_steps=50,
              danger_reflex=False, wall_reflex=False):
    """Run greedily (tiny eval epsilon to avoid oscillation, no learning)."""
    out = [run_episode(world, encoder, mind, learn=False, explore=False,
                       eval_epsilon=0.05, mem=mem, max_steps=max_steps,
                       danger_reflex=danger_reflex, wall_reflex=wall_reflex)
           for _ in range(n)]
    return np.mean([r for r, _ in out]), np.mean([f for _, f in out])


def _survive(world, encoder, mind, n=15, mem=0, danger_reflex=True,
             wall_reflex=False):
    """SURVIVAL evaluation: each life runs until the creature DIES (poison or
    empty battery), and the score is stars collected in a life. This is the
    harsh framing that exposed what 50-step caps mask -- per-step risks
    compound: a policy with a ~0.6%/step chance of touching poison looked fine
    capped and died in 67-73% of full lives. Returns (mean stars, mean
    lifespan, poison-death rate)."""
    stars, ages, pdeaths = [], [], 0
    for _ in range(n):
        run_episode(world, encoder, mind, learn=False, explore=False,
                    eval_epsilon=0.05, mem=mem, max_steps=None,
                    danger_reflex=danger_reflex, wall_reflex=wall_reflex)
        stars.append(world.stars)
        ages.append(world.age)
        pdeaths += ((world.cx, world.cy) in world.poison)
    return float(np.mean(stars)), float(np.mean(ages)), pdeaths / n


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

    # --- Scene A: a clean world with only stars -------------------------
    print("\n--- Scene A: stars only ----------------------------------------\n")
    encoder = CreatureEncoder(dim, seed=1)
    mind = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.35,
                           novelty_bonus=0.1, memory_cap=5000, seed=2)
    world = GridWorld(7, 7, n_poison=0, seed=3)

    base_r, base_f = _baseline(world, encoder)
    print(f"Random baseline: reward {base_r:+.2f}, stars collected {base_f:.1f}\n")
    _train(world, encoder, mind, episodes=150)
    ev_r, ev_f = _evaluate(world, encoder, mind)
    print(f"\nTrained (greedy): reward {ev_r:+.2f}, stars collected {ev_f:.1f}  "
          f"(baseline {base_r:+.2f}, {base_f:.1f})")

    print("\nLearned reflexes -- where it heads for a star in each direction:")
    for fx, fy, where in [("east", "none", "east"), ("west", "none", "west"),
                          ("none", "north", "north"), ("none", "south", "south")]:
        a = mind.decide(encoder.encode({"food_x": fx, "food_y": fy}), explore=False)
        print(f"  star to the {where:5s} -> moves {mind.actions[a]}")

    # --- Scene B: add poison, watch it learn avoidance ------------------
    print("\n--- Scene B: now with poison (lethal) --------------------------\n")
    encoder = CreatureEncoder(dim, seed=1)
    mind = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.35,
                           novelty_bonus=0.1, memory_cap=5000, seed=2)
    world = GridWorld(7, 7, n_poison=2, seed=3)

    base_r, base_f = _baseline(world, encoder)
    print(f"Random baseline: reward {base_r:+.2f}, stars collected {base_f:.1f}\n")
    _train(world, encoder, mind, episodes=200)
    ev_r, ev_f = _evaluate(world, encoder, mind)
    print(f"\nTrained (greedy): reward {ev_r:+.2f}, stars collected {ev_f:.1f}  "
          f"(baseline {base_r:+.2f}, {base_f:.1f})")
    print("  Stars collected is the honest success metric: poison is now lethal,")
    print("  so a single wrong step ends the life -- learning to avoid it matters.")

    # THE SURVIVAL TEST -- forage until the battery dies, no step cap. This
    # framing found two real problems the capped test masked: (1) compounding
    # poison risk (the capped-trained brain died on poison in 67-73% of full
    # lives) -- fixed by the danger reflex, lethal moves vetoed below the brain;
    # (2) DITHERING -- a memoryless forager spent a measured 60% of its steps
    # oscillating back where it was two steps before, starving at 28 stars;
    # working memory (mem=3) cuts dithering to 10% and lifts it to ~121 stars,
    # 89% of the danger-aware greedy reflex's 136 -- the same ratio it achieves
    # in the clean world, so the remaining gap is the chase itself, not poison.
    enc_s = CreatureEncoder(dim, seed=1)
    mind_s = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.45,
                             novelty_bonus=0.2, memory_cap=12000, seed=2)
    for ep in range(180):
        mind_s.epsilon = max(0.05, 0.45 * (1.0 - ep / 180))
        run_episode(world, enc_s, mind_s, learn=True, explore=True, mem=3,
                    max_steps=100, danger_reflex=True)
    s, a, pd = _survive(world, enc_s, mind_s, mem=3)
    print("\n  SURVIVAL (life ends only at death; mem=3 + danger reflex):")
    print(f"    stars per life {s:.0f}, lifespan {a:.0f} steps, poison deaths {pd*100:.0f}%")

    # Avoidance reflex over all four directions (honest aggregate, not one
    # cherry-picked probe): does it head for a clear star, and turn away when
    # that direction is poison?
    dirs = [("E", "east", "food_x"), ("W", "west", "food_x"),
            ("N", "north", "food_y"), ("S", "south", "food_y")]
    seek = avoid = 0
    for d, val, axis in dirs:
        clear = {axis: val, ("food_y" if axis == "food_x" else "food_x"): "none"}
        seek += mind.actions[mind.decide(encoder.encode(clear), explore=False)] == d
        blocked = mind.decide(encoder.encode({**clear, "danger_" + d: "yes"}), explore=False)
        avoid += mind.actions[blocked] != d
    print(f"\n  Reflex check: heads for a clear star {seek}/4, turns away when that "
          f"way is poison {avoid}/4.")

    print("\nA few greedy steps (C=creature *=star x=poison):")
    senses = world.reset()
    state = encoder.encode(senses)
    for frame in range(4):
        print(f"\n  step {frame}:")
        print("    " + world.render().replace("\n", "\n    "))
        a = mind.decide(state, explore=False, epsilon=0.05)
        senses, _, _, done = world.step(mind.actions[a])
        state = encoder.encode(senses)
        if done:
            print("\n    " + world.render().replace("\n", "\n    "))
            print("    (the creature's life ended)")
            break
    print()


def demo_memory(seeds=(2, 5, 8), episodes=130, steps=60):
    """Scene C: with limited vision, show that a working memory of recent moves
    lets the creature SEARCH efficiently instead of dithering blindly.

    Single RL runs are noisy, so we average a few independent seeds and report
    them all -- no cherry-picking."""
    dim = 256
    print("\n--- Scene C: limited vision, with vs without working memory ------\n")
    print("On an 11x11 grid the creature only senses a star within 2 cells, so it")
    print("is blind most of the time and has to search. We train it identically")
    print(f"except for memory depth, {len(seeds)} seeds each, and count stars found")
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
        print(f"  mem={mem} ({tag:11s}): stars per seed = "
              f"{[round(float(f), 2) for f in foods]}  mean {means[mem]:.2f}")

    print(f"\nWorking memory finds {means[3] / max(means[0], 1e-9):.1f}x more stars, "
          f"and wins every seed.")
    print("Why: blind, a reactive creature wanders back over itself (~sqrt(t)")
    print("coverage); memory lets it hold a heading and sweep new ground (~t).")
    print("Honest caveat: on a SMALLER grid the creature bumps into stars often")
    print("enough that memory barely helps -- it earns its keep only when the")
    print("task genuinely demands search.")


def demo_obstacles(seeds=(2, 7, 11), episodes=240):
    """Scene D: obstacles. First random WALLS in the forage world (the creature
    must route around them, and a working memory roughly triples how many stars
    it manages), then a fixed LABYRINTH it learns to escape.

    As elsewhere we average a few seeds and report them all -- no cherry-picking.
    """
    dim = 256
    print("\n--- Scene D: walls to route around, and a labyrinth to escape ----\n")

    # (a) foraging with random walls: reactive vs working memory ---------------
    print("(a) A 7x7 forage world with 2 poison + 8 random walls. Walls are")
    print("    impassable, so the creature can't just head straight at a star --")
    print("    it has to go around. We train identically except for memory depth")
    print("    and count stars per 100-step life.\n")
    means = {}
    for mem in (0, 3):
        stars = []
        for seed in seeds:
            enc = CreatureEncoder(dim, seed=1)
            mind = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.45,
                                   novelty_bonus=0.2, memory_cap=12000, seed=seed)
            world = GridWorld(7, 7, n_poison=2, n_walls=8, seed=3)
            for ep in range(episodes):
                mind.epsilon = max(0.05, 0.45 * (1.0 - ep / episodes))
                run_episode(world, enc, mind, learn=True, explore=True, mem=mem, max_steps=100)
            _, s = _evaluate(world, enc, mind, n=30, mem=mem, max_steps=100)
            stars.append(s)
        means[mem] = float(np.mean(stars))
        tag = "reactive" if mem == 0 else "with memory"
        print(f"  mem={mem} ({tag:11s}): stars per seed = "
              f"{[round(float(s), 2) for s in stars]}  mean {means[mem]:.2f}")
    print(f"\n    Working memory collects {means[3] / max(means[0], 1e-9):.1f}x more "
          f"stars among the walls -- the same tool that helps when blind also helps\n"
          f"    when the straight line is blocked.")

    # SURVIVAL framing -- and the cluttered-world open problem, SOLVED by the
    # system's own introspection: describe() on a caught dither revealed the
    # brain valuing moves into walls it could sense (the 'oscillation' was
    # wall-bumping in place). The wall reflex (veto wall moves through the same
    # `among` mechanism as danger) took stars from 5.1 to 19.8 -- the
    # danger-aware reflex's ceiling -- with dither 79% -> 43% and deaths 0%.
    enc_s = CreatureEncoder(dim, seed=1)
    mind_s = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.45,
                             novelty_bonus=0.2, memory_cap=12000, seed=2)
    world_s = GridWorld(7, 7, n_poison=2, n_walls=8, seed=3)
    for ep in range(180):
        mind_s.epsilon = max(0.05, 0.45 * (1.0 - ep / 180))
        run_episode(world_s, enc_s, mind_s, learn=True, explore=True, mem=3,
                    max_steps=100, danger_reflex=True, wall_reflex=True)
    s, a, pd = _survive(world_s, enc_s, mind_s, mem=3, wall_reflex=True)
    print(f"\n    SURVIVAL among walls (wall reflex on): stars {s:.0f}/life, "
          f"lifespan {a:.0f}, poison deaths {pd*100:.0f}%")

    # (b) a fixed labyrinth the creature learns to escape ----------------------
    print("\n(b) A fixed 7x7 perfect maze (one route between any two cells). The")
    print("    creature starts in a corner and must find the EXIT, learning this")
    print("    one labyrinth over repeated tries -- the classic rat-in-a-maze.\n")
    probe = GridWorld(7, 7, maze=True, fixed_seed=7)
    optimal = len(probe.shortest_path((probe.cx, probe.cy), (probe.fx, probe.fy))) - 1
    escapes = []
    for seed in seeds:
        enc = CreatureEncoder(dim, seed=1)
        mind = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.50,
                               novelty_bonus=0.2, memory_cap=12000, seed=seed)
        world = GridWorld(7, 7, maze=True, fixed_seed=7)
        for ep in range(episodes):
            mind.epsilon = max(0.05, 0.50 * (1.0 - ep / episodes))
            run_episode(world, enc, mind, learn=True, explore=True, mem=4, max_steps=90)
        got = 0
        for _ in range(20):
            run_episode(world, enc, mind, learn=False, explore=False,
                        eval_epsilon=0.05, mem=4, max_steps=90)
            got += world.escaped
        escapes.append(got / 20.0)
    print(f"  optimal escape = {optimal} steps. Escape rate per seed = "
          f"{[round(e * 100) for e in escapes]}%  mean {np.mean(escapes) * 100:.0f}%")
    print("\n  Honest limit, revised: a 9x9 used to be beyond this brain -- far-apart")
    print("  corridors look identical through its egocentric senses. That ceiling")
    print("  fell (0% -> 100%) without changing the brain or the senses: with")
    print("  corridor_reflex=True decisions are spent only at junctions, so the")
    print("  exit's credit survives the discount. See test_creature_gauntlet.py;")
    print("  the new frontier is 13x13 (67% vs a 15% reflex-with-random control).")


def demo_introspect(episodes=240, seed=7):
    """Scene E: the creature's memory is the same holographic kit as the image
    side -- a classifier, layered superpositions, and a recursive index.

    (a) Classification + layering: the brain keeps one PROTOTYPE per distinct
        situation (cosine-matched class), each a bundle of its members with a
        denoised mean return. We report how many raw experiences collapse into
        how few prototypes.
    (b) Recursive recall: we index those prototypes with the SAME HoloForest the
        image vault uses, and recall the most similar past situation from a noisy
        cue in a fraction of a full scan -- recall@comparisons, honestly.
    """
    from holographic.misc.holographic_tree import HoloForest
    dim = 256
    print("\n--- Scene E: the creature's memory as a holographic store --------\n")

    # Train a working-memory creature in the walls world -- it meets a rich set of
    # situations, so the prototype set is sizable enough for the index to matter.
    enc = CreatureEncoder(dim, seed=1)
    mind = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.45,
                           novelty_bonus=0.2, memory_cap=12000, seed=seed)
    world = GridWorld(7, 7, n_poison=2, n_walls=8, seed=3)
    for ep in range(episodes):
        mind.epsilon = max(0.05, 0.45 * (1.0 - ep / episodes))
        run_episode(world, enc, mind, learn=True, explore=True, mem=3, max_steps=100)

    protos = mind.prototype_count()
    print("(a) Classification + layering (the prototype memory)")
    print(f"    absorbed {mind.experiences} raw experiences -> kept {protos} prototypes "
          f"({mind.experiences / max(protos, 1):.0f}x smaller).")
    print("    Each prototype is a superposed bundle of similar states with the")
    print("    average of their returns -- the bundle+cosine classifier and the")
    print("    'superpose a gallery into plates' trick, now storing experience.\n")

    # (b) index the prototypes with a recursive HoloForest and recall from noisy cues
    vecs, _, _ = mind.prototypes()
    forest = HoloForest(dim, n_trees=4, leaf_size=64, seed=0).build(vecs)
    rng = np.random.default_rng(0)
    hits = comps = trials = 0
    for _ in range(200):
        i = int(rng.integers(len(vecs)))
        cue = vecs[i] + 0.35 * random_vector(dim, rng)        # a fuzzy "situation like this"
        cue = cue / np.linalg.norm(cue)
        truth = int((vecs @ cue).argmax())                    # exact nearest (full scan)
        got = forest.recall(cue, beam=4)
        hits += (got == truth); comps += forest.last_comparisons; trials += 1
    st = forest.trees[0].stats()
    flux = forest.trees[0].flux()
    busy = sum(1 for f in flux if f > np.mean(flux)) if flux else 0
    print("(b) Recursion, branching & partitioning (HoloForest, the image vault's index)")
    print(f"    one tree recursively splits the {protos} prototypes on a random")
    print("    hyperplane at each node -- a BINARY BRANCHING that PARTITIONS the")
    print(f"    memory into {st['leaves']} leaf cells at depth {st['depth']} "
          f"(~{st['avg_leaf']:.0f} prototypes each).")
    print("    Recalling the most similar past situation to a noisy cue then agrees")
    print(f"    with an exact scan {100 * hits / trials:.0f}% of the time while comparing ~"
          f"{comps / trials:.0f} prototypes, not all {protos} "
          f"({protos / max(comps / trials, 1):.1f}x fewer); query 'flux' concentrates")
    print(f"    on {busy}/{st['leaves']} cells -- the thick-vein paths the cues take.\n")
    print("    Honest note: this index is for ASSOCIATIVE recall, not the control")
    print("    loop. Picking a move needs the full weighted neighbourhood, and the")
    print("    approximation drops enough of it to hurt (it wrecks the maze policy),")
    print("    so deciding still uses the exact scan -- which the compression above")
    print("    already made cheap. Tested-and-kept-out, not overlooked.")

    # (c) INTROSPECTION: the brain DESCRIBES its own memory in sense terms.
    # Prototypes are role-bound sense bundles, so describe() decodes them back
    # (measured: present roles 373/373, absent roles silent 427/427) -- and
    # why_differ() gives the per-role verdict between two situations. Below,
    # the most- and least-valuable prototype for one action, in the creature's
    # own vocabulary -- the same machinery whose first real outing articulated
    # (and thereby solved) the wall-bumping open problem.
    print("\n(c) The brain describes its own memory (introspection)")
    V, acts, rets = mind.prototypes()
    a0 = 0                                            # action "N"
    rows = [(i, rets[i]) for i in range(len(V)) if acts[i] == a0]
    if len(rows) >= 2:
        best = max(rows, key=lambda z: z[1])
        worst = min(rows, key=lambda z: z[1])
        print(f"    action '{mind.actions[a0]}': best prototype (return "
              f"{best[1]:+.2f}) vs worst ({worst[1]:+.2f}) -- its own verdict:")
        for role, va, vb, same in mind.why_differ(V[best[0]], V[worst[0]], enc):
            print(f"      {role:10s}: {str(va):8s} vs {str(vb):8s}  "
                  f"{'same' if same else 'DIFFERS'}")


def learn_maze(world_factory, dim=256, episodes=240, gamma=0.97, mem=4,
               max_steps=500, candidates=3, probe=6, accept=2/3, seed=2,
               k=15, bootstrap="auto"):
    """Learn to escape a maze reliably -- the rat-in-a-maze protocol, hardened for
    big mazes and ANY maze seed. Returns (encoder, mind, measured_probe_rate).

    Two lessons are baked in, both measured on 16x16 mazes (optimal paths 80-108
    steps against the old default battery of 100, which starved the creature
    even on a perfect run -- the default is now 300, and bigger worlds may
    still need start_energy raised to match):

    * gamma=0.97, NOT the foraging default of 0.9. The credit-horizon arithmetic
      that broke the per-step framing strikes again one level up: a 16x16 needs
      enough junction decisions that 0.9 starves the exit signal (0.9^20 ~ 0.12)
      and training turns BIMODAL -- runs land at ~100% or collapse to 0% with
      nothing between, the brain committing early to a wrong junction policy and
      then greedily cycling it. At 0.97 the same failing maze/brain combinations
      went 1% -> 98% mean, and the smaller gauntlet mazes IMPROVED too (13x13
      67% -> 100%). Epsilon floors and longer training did not help; the
      horizon was the lever.

    * THE BOOTSTRAP RESCUE (bootstrap="auto", the default), for mazes too deep
      for luck -- and ONLY for those, because measurement cut both ways. On
      20x20 seed 11 the plain protocol probes 0% at every budget tried (under
      the decaying epsilon schedule, zero training escapes -- the loop-attractor
      policy locks in before luck finds the exit; sustained high epsilon
      occasionally escapes but never consolidates); the rescue -- CURIOSITY (a
      first-visit cell bonus of exit_reward / n_free_cells, the world's own
      arithmetic, full coverage summing to exactly one exit reward, switched
      off at the candidate's first escape because visited-ness is not in the
      creature's state and the crumbs are unlearnable noise after their job is
      done) plus REHEARSAL (successful trajectories stored, one re-remembered
      per episode, so a rare success is consolidated instead of drowned) plus
      CAPACITY (256/15 holds through 16x16; 20x20 needed dim=512, k=30) --
      took that seed from impossible to a 67% probe. BUT on 20x20 seed 5,
      where a bigger budget already found successes by luck (83% plain), the
      same protocol HURT (0% with it; rehearsal alone 33%): curiosity noise
      and rehearsed early meanders degrade a signal that was already arriving.
      So the protocol is a RESCUE, summoned by self-measurement: candidates
      run PLAIN, and only when a candidate finishes training with zero escapes
      (starvation -- the data's own signal, no threshold) do subsequent
      candidates enable the bootstrap. bootstrap=True forces it always on,
      False never. Speculate-measure-adopt, applied to the protocol itself.

    * SPECULATE-MEASURE-ADOPT over whole policies, the organizer's rule applied
      to training runs: even at gamma=0.97 a stray combination still collapsed
      (15-run grid: grand mean 93%, worst run 0%). So this function trains a
      candidate, PROBES its real escape rate over a few greedy lives, adopts it
      if it measures competent, and otherwise restarts with a different brain
      seed. No map knowledge is involved -- the creature is simply allowed to
      notice it has not learned to escape and start over, which is the same
      self-measurement discipline the rest of the system runs on (and the same
      train-several-keep-the-best pattern the UI already uses for foraging)."""
    best = (None, None, -1.0)
    starved = False                        # did a plain candidate get ZERO escapes?
    for c in range(candidates):
        use_boot = (bootstrap is True) or (bootstrap == "auto" and starved)
        enc = CreatureEncoder(dim, seed=1)
        mind = HolographicMind(dim, GridWorld.ACTIONS, k=k, epsilon=0.5,
                               novelty_bonus=0.2, memory_cap=800 * k,
                               seed=seed + 97 * c)
        world = world_factory()
        world.reset()
        cur = (world.EXIT / max(1, len(world._free_cells()))) if use_boot else 0.0
        successes = []
        n_escapes = 0
        for ep in range(episodes):
            mind.epsilon = max(0.05, 0.5 * (1.0 - ep / episodes))
            out = run_episode(world, enc, mind, learn=True, explore=True, mem=mem,
                              corridor_reflex=True, max_steps=max_steps,
                              gamma=gamma, curiosity=cur,
                              return_trajectory=use_boot)
            if world.escaped:
                n_escapes += 1
                cur = 0.0                  # curiosity's job (first success) is done
                if use_boot:
                    successes.append(out[2])
            if use_boot and successes:     # consolidate: one success per episode
                s_, a_, r_ = successes[ep % len(successes)]
                mind.remember(s_, a_, r_)
        if n_escapes == 0:
            starved = True                 # starvation observed: summon the rescue
        got = 0
        world = world_factory()
        for _ in range(probe):
            run_episode(world, enc, mind, learn=False, explore=False,
                        eval_epsilon=0.05, mem=mem, corridor_reflex=True,
                        max_steps=max_steps)
            got += world.escaped
        rate = got / probe
        if rate > best[2]:
            best = (enc, mind, rate)
        if rate >= accept:
            break                          # measured competent: adopt and stop
    return best


def demo_self_maintaining(dim=256, seed=0):
    """The orchestrator brain keeping ITSELF fresh, with no thresholds to tune. We
    run a contextual decision task, SHIFT the world (every situation's best action
    changes), and watch each brain recover. The plain brain cannot: stale duplicate
    prototypes hold the old values up and online updates only touch one at a time.
    The autonomous brain keeps a window of recent experience and, on its own,
    speculates a few reorganised versions of itself, measures which one would have
    made the best DECISIONS on held-out recent experience, and adopts it -- folding
    duplicates in quiet stretches, rebuilding from recent experience when the world
    moves. The data decides; nothing here is tuned for either case."""
    print("=" * 70)
    print("A brain that keeps itself fresh -- fully autonomously (no thresholds)")
    print("=" * 70)
    rng = np.random.default_rng(seed)
    C, A, shift_at, steps = 24, 3, 2500, 5500
    base = [rng.standard_normal(dim) for _ in range(C)]
    base = [b / np.linalg.norm(b) for b in base]
    best = [int(rng.integers(A)) for _ in range(C)]
    def state(i):
        v = base[i] + 0.02 * rng.standard_normal(dim)
        return v / np.linalg.norm(v)

    plain = HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                            novelty_bonus=0.2, seed=seed)
    auto = HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                           novelty_bonus=0.2, seed=seed, maintain='auto')

    def acc(b):
        ok = 0
        for _ in range(400):
            i = int(rng.integers(C)); ok += (b.decide(state(i), explore=False, epsilon=0.0) == best[i])
        return ok / 400

    curve_p, curve_a, choices = [], [], []
    for t in range(1, steps + 1):
        if t == shift_at:
            best = [(x + 1) % A for x in best]
        i = int(rng.integers(C)); s = state(i)
        for b in (plain, auto):
            a = b.decide(s, explore=True, epsilon=0.25)
            prev = b.reorganizations
            b.remember([s], [a], [1.0 if a == best[i] else 0.0])
            if b is auto and b.reorganizations > prev:
                choices.append((t, b.last_choice))
        if t > shift_at and (t - shift_at) % 750 == 0:
            curve_p.append(round(acc(plain) * 100))
            curve_a.append(round(acc(auto) * 100))

    print("\n  Post-shift recovery (greedy accuracy, every 750 steps after the shift):")
    print(f"    plain brain   : {curve_p}   ({plain.prototype_count()} prototypes)")
    print(f"    autonomous    : {curve_a}   ({auto.prototype_count()} prototypes)")
    print("\n  The autonomous brain decided, with nothing tuned, to:")
    for when, what in choices:
        tag = "before the shift" if when < shift_at else "after the shift "
        print(f"    step {when:>4} ({tag}) -> {what}")
    print("\n  Folds while the world holds still (compress, forget nothing); a rebuild")
    print("  from recent experience once it moves (forget the stale regime). The plain")
    print("  brain, with no upkeep, stays stuck on the old policy. Same recovery and")
    print("  leanness as the hand-tuned version -- but chosen by measurement, not by me.")


def _d2_selftest():
    """D2: robust_returns winsorises outlier rewards so a fluke (a jackpot, a sensor glitch) cannot swing a
    prototype's running-mean value -- measured markedly lower value error under outlier rewards, with no cost on
    clean data; and the flag survives save/load (old saves default to off)."""
    import numpy as _np
    dim = 256

    def trial(robust, outliers, seed):
        rng = _np.random.default_rng(seed)
        brain = HolographicMind(dim, ["a", "b"], merge=0.8, robust_returns=robust)
        s = rng.standard_normal(dim)
        s /= _np.linalg.norm(s) + 1e-12                              # one fixed state -> one prototype for action a
        for _ in range(150):
            r = rng.normal(20.0, 5.0) if (outliers and rng.random() < 0.08) else rng.normal(1.0, 0.3)
            brain.remember([s], [0], [float(r)])                     # noisy/outlier rewards, true mean 1.0
        est, _ = brain.value(s, 0)
        return abs(est - 1.0)

    err_plain = _np.mean([trial(False, True, s) for s in range(8)])
    err_robust = _np.mean([trial(True, True, s) for s in range(8)])
    assert err_robust < err_plain * 0.6, (err_robust, err_plain)    # robust markedly closer under outliers

    clean_plain = _np.mean([trial(False, False, s) for s in range(8)])
    clean_robust = _np.mean([trial(True, False, s) for s in range(8)])
    assert clean_robust < clean_plain * 1.3 + 0.02, (clean_robust, clean_plain)   # no meaningful cost on clean data

    b = HolographicMind(dim, ["a", "b"], robust_returns=True)
    b2 = HolographicMind.from_state(b.to_state())
    assert b2.robust_returns is True                                # the flag survives save/load


if __name__ == "__main__":
    demo_creature()
    demo_memory()
    demo_obstacles()
    demo_introspect()
    demo_self_maintaining()


def capture_route(world_factory, encoder, mind, mem=2, max_steps=300, trials=8):
    """Run a trained maze brain and capture its successful escape routes as
    ordered cell-sequences (the corridor reflex walks forced cells, so a route is
    the junction-to-junction path the brain actually took). Returns a list of
    routes, each a list of 'rYcX' cell names in visit order -- ready to hand to
    UnifiedMind.learn_sequences so the brain can DISCOVER and PROVE the canonical
    structure of its own successful behaviour. Acting, then understanding the
    structure of the action: the sequence machinery turned on the creature."""
    routes = []
    for _ in range(trials):
        w = world_factory()
        senses = w.reset()
        recent = []
        cells = [f"r{w.cy}c{w.cx}"]
        state = encoder.build_state(senses, recent, mem)
        last = None
        for _ in range(max_steps):
            a = mind.decide(state, explore=False, epsilon=0.05, senses=senses)
            senses, r, ate, done = w.step(GridWorld.ACTIONS[a])
            last = GridWorld.ACTIONS[a]
            cells.append(f"r{w.cy}c{w.cx}")
            recent = [last] + recent
            while not done:                            # corridor reflex
                fwd = _forced_dir(senses, last)
                if fwd is None:
                    break
                senses, r, ate, done = w.step(fwd)
                last = fwd
                cells.append(f"r{w.cy}c{w.cx}")
                recent = [fwd] + recent
            state = encoder.build_state(senses, recent, mem)
            if done:
                if w.escaped:
                    routes.append(cells)
                break
    return routes


def replay_plan(world, route, reset=True):
    """Drive navigation from a DISCOVERED route plan instead of re-deciding every
    step -- and VALIDATE each move honestly. The creature steps toward each next
    cell in the plan; if a move fails to advance (a wall blocks it), the plan has
    hit the boundary of its validity and that break point is REPORTED, not papered
    over. Returns (status, step_index, blocked_from, intended_cell) where status
    is 'escaped', 'dead', 'broke' (the maze differs from where the plan was
    learned -- the informative case), or 'ran_out'.

    This is the discovered-plan machinery composed with action: a proven plan
    knows where it stops applying, so a creature can replay what it learned and
    detect exactly where reality has changed -- the seam at which it would need to
    re-learn only the changed segment, not the whole maze."""
    def _cell(name):
        r = int(name[1:name.index('c')])
        c = int(name[name.index('c') + 1:])
        return c, r
    w = world
    if reset:
        w.reset()                       # NB: without fixed_seed a reset re-carves a
                                        # NEW maze; pass reset=False to drive an
                                        # already-prepared (e.g. mutated) world as-is
    for i, nxt in enumerate(route[1:]):
        tx, ty = _cell(nxt)
        ox, oy = w.cx, w.cy
        if w.cx < tx:
            a = "E"
        elif w.cx > tx:
            a = "W"
        elif w.cy < ty:
            a = "S"
        elif w.cy > ty:
            a = "N"
        else:
            continue
        senses, r, ate, done = w.step(a)
        if (w.cx, w.cy) == (ox, oy):                  # blocked: plan broke here
            return ("broke", i, (ox, oy), nxt)
        if done:
            return ("escaped" if w.escaped else "dead", i, None, None)
    return ("ran_out", len(route), None, None)


class WorldView:
    """The creature's world as a COUNTABLE, DIFFABLE composite -- the scene
    machinery applied to perception. Every visible thing (exit, poison cell,
    wall cell) is an object = bind(type, position); the view is their
    unnormalised superposition. Two properties fall out of the algebra:

      * COUNT: round(||view||^2) is the number of visible things (the same
        norm-counting as SceneCoder.count_objects -- near-orthogonal unit
        products).
      * CHANGE: the DIFFERENCE of two views is itself a composite of the
        changes -- appeared objects sit in it positively, vanished ones
        negatively. round(||diff||^2) counts the changes, and peeling the diff
        (positively for appeared, negatively for vanished) NAMES each one.
        Termination is count-driven: we peel exactly as many objects as the
        diff's own norm says changed -- no confidence threshold, the data's own
        scale.

    So the creature can notice 'something changed', know HOW MUCH changed, and
    say WHAT changed, all from vector algebra on two snapshots -- perception-
    level change detection nearly for free."""

    TYPES = ("exit", "poison", "wall")

    def __init__(self, dim=2048, width=16, height=16, seed=0):
        from holographic.agents_and_reasoning.holographic_ai import random_vector
        rng = np.random.default_rng(seed)
        self.dim = dim
        self._t = {t: random_vector(dim, rng) for t in self.TYPES}
        self._p = {(x, y): random_vector(dim, rng)
                   for x in range(width) for y in range(height)}

    def _obj(self, typ, pos):
        from holographic.agents_and_reasoning.holographic_ai import bind
        return bind(self._t[typ], self._p[pos])

    def view(self, world):
        """Encode a GridWorld's current contents as one composite vector."""
        objs = [("exit", (world.fx, world.fy))]
        objs += [("poison", p) for p in getattr(world, "poison", ())]
        objs += [("wall", w) for w in getattr(world, "walls", ())]
        if not objs:
            return np.zeros(self.dim)
        return np.sum([self._obj(t, p) for t, p in objs], axis=0)

    def count(self, view_vec):
        return int(round(float(np.linalg.norm(view_vec)) ** 2))

    def _best(self, vec):
        from holographic.agents_and_reasoning.holographic_ai import cosine
        best, bs = None, -2.0
        for t in self.TYPES:
            for pos in self._p:
                s = cosine(vec, self._obj(t, pos))
                if s > bs:
                    bs, best = s, (t, pos)
        return best, bs

    def changes(self, view_old, view_new):
        """What changed between two snapshots? Returns (appeared, vanished) as
        lists of (type, position). The diff's norm COUNTS the changes; we peel
        exactly that many, each time taking whichever sign (appeared vs
        vanished) currently explains the residual better -- count-driven, no
        threshold."""
        diff = np.asarray(view_new, float) - np.asarray(view_old, float)
        n = self.count(diff)
        appeared, vanished = [], []
        resid = diff.copy()
        for _ in range(n):
            (ta, pa), sa = self._best(resid)
            (tv, pv), sv = self._best(-resid)
            if sa >= sv:
                appeared.append((ta, pa))
                resid = resid - self._obj(ta, pa)
            else:
                vanished.append((tv, pv))
                resid = resid + self._obj(tv, pv)
        return appeared, vanished
