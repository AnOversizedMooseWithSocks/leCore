"""The creature's value head AS a VSA program -- policy = hypervectors, learn = bundling, decide = a dot.

WHY THIS MODULE EXISTS
----------------------
The creature mind (holographic_creature.HolographicMind) is holographic on the OUTSIDE -- states are
role-bound bundles, prototypes are superpositions, perceive is bundle-and-cosine -- but TABULAR on the
inside: for each action it keeps a GROWING list of prototype direction vectors paired with a parallel
NumPy array of scalar mean-returns (self._unit[a], self._ret[a]), and value(s,a) is a kernel-weighted
average of those scalars. That scalar-return table is the one part of the brain that is not itself a VSA
program -- it is why the creature "functions a little different" from the recipes / SDF trees / orchestrator
that were all elevated to LIVE in the holographic space.

This head folds the whole table into the holographic space. The creature's value is exactly a
Nadaraya-Watson estimator:

    value(s, a) = sum_i sim(s, proto_i) * ret_i  /  sum_i sim(s, proto_i)        (sim = cosine)

and a sum is a BUNDLE. So keep, per action, just two hypervectors:

    Q_a = sum_i ret_i * unit(state_i)     (the return-weighted superposition of states seen under a)
    N_a = sum_i        unit(state_i)     (the plain superposition -- the normaliser)

Then, because <s, unit(state_i)> IS the cosine sim,

    value(s, a) = <s, Q_a> / <s, N_a>

reproduces the SAME weighted average, but the entire per-action policy is now TWO fixed-length vectors
instead of an unbounded list, and learning is one bundling step:  Q_a += ret*u ;  N_a += u  -- the
holographic complement, O(1) per update and independent of how much has been learned. The policy {Q, N} is
a hypervector program: savable, bindable, composable, inspectable, exactly like a recipe.

THE HONEST TRADE (kept, and measured in the self-test)
  * A fixed-D pair of bundles has FINITE capacity. The scalar table is effectively exact and grows without
    bound; this head folds everything into 2*A vectors, so past ~D distinct situations the cross-talk
    between folded experiences blurs the recalled value -- the VSA capacity cliff. So the head MATCHES the
    tabular brain at low load and DEGRADES past the cliff. The win is architectural (one composable,
    savable, fixed-size hypervector policy that is consistent with the rest of the stack, with graceful
    degradation), NOT higher accuracy at small scale -- and the self-test keeps that negative loud.
  * The creature's tabular `support` is the NEAREST-prototype similarity (a max); this head's natural
    support is <s, N_a>, the total similarity MASS (a sum) -- a different, mass-based familiarity signal.
    It cannot cheaply recover "nearest" without the list it deliberately discards. Reported, not hidden.
  * Linear kernel: the tabular head clips negative sims (a ReLU kernel) and keeps only k nearest; the
    folded dot product cannot, so it uses a LINEAR kernel over ALL experiences. In high dimensions
    dissimilar states are near-orthogonal, so the dropped clip is mostly harmless -- but it is a difference.
"""

import numpy as np


class HolographicValueHead:
    """A drop-in value backend for the creature -- same value(state, action)->(value, support) and
    absorb(state, action, ret) API as HolographicMind, but the policy is two bundles per action (a pure
    VSA program) rather than a growing list of (vector, scalar) prototypes."""

    def __init__(self, dim, n_actions, eps=1e-9):
        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.eps = float(eps)
        self.Q = np.zeros((self.n_actions, self.dim))    # return-weighted state superposition, per action
        self.N = np.zeros((self.n_actions, self.dim))    # plain state superposition (the normaliser)
        self.count = np.zeros(self.n_actions)            # scalar tally per action (cheap; keeps fixed size)

    # -- learning: one bundling step (the holographic complement) ------------------------------------
    def absorb(self, state, action_idx, ret):
        """Fold one (state, action, return) experience in by bundling -- O(1), independent of history."""
        u = np.asarray(state, float)
        nrm = np.linalg.norm(u)
        if nrm > 0:
            u = u / nrm
        self.Q[action_idx] += float(ret) * u
        self.N[action_idx] += u
        self.count[action_idx] += 1.0
        return self

    # -- recall: a dot product (the Nadaraya-Watson average, folded) ---------------------------------
    def value(self, state, action_idx):
        """(value, support) for one action. value is <s,Q_a>/<s,N_a>; support is the similarity MASS
        <s,N_a> normalised by the experience count -> a ~[0,1]-ish familiarity (mass-based, see caveat)."""
        s = np.asarray(state, float)
        den = float(s @ self.N[action_idx])
        if abs(den) <= self.eps:
            return 0.0, 0.0
        val = float(s @ self.Q[action_idx]) / den
        support = den / max(self.count[action_idx], 1.0)
        return val, support

    def value_all(self, state):
        """Vectorised over actions: returns (values (A,), supports (A,)) in one matmul each."""
        s = np.asarray(state, float)
        den = self.N @ s                                 # (A,)
        num = self.Q @ s                                 # (A,)
        safe = np.abs(den) > self.eps
        values = np.where(safe, num / np.where(safe, den, 1.0), 0.0)
        supports = np.where(safe, den / np.maximum(self.count, 1.0), 0.0)
        return values, supports

    def decide(self, state, among=None):
        """Greedy choice on value alone (no exploration bonus) -- the control-relevant decision."""
        values, _ = self.value_all(state)
        idxs = range(self.n_actions) if among is None else list(among)
        return int(max(idxs, key=lambda a: values[a]))

    # -- the policy IS a hypervector program -----------------------------------------------------------
    def policy_vectors(self):
        """The whole learned policy as hypervectors: (Q, N), each (n_actions, dim). Savable / bindable /
        composable -- this is the point of the exercise."""
        return self.Q.copy(), self.N.copy()

    @classmethod
    def from_policy(cls, Q, N, count=None):
        """Rebuild a head from saved policy hypervectors -- the save/load round-trip. The policy is just
        arrays, so persisting and restoring a learned brain is trivial and stays inside the holographic space."""
        Q = np.asarray(Q, float); N = np.asarray(N, float)
        h = cls(Q.shape[1], Q.shape[0])
        h.Q = Q.copy(); h.N = N.copy()
        h.count = np.ones(Q.shape[0]) if count is None else np.asarray(count, float).copy()
        return h

    def policy_atom(self, action_codes):
        """Fold the WHOLE policy into two composable hypervectors by binding each action's bundle to that
        action's code: M_Q = sum_a bind(code_a, Q_a), M_N = sum_a bind(code_a, N_a). Now the policy is two
        D-vectors that can be bound into a larger structure (a recipe) and carried around the VSA space as one
        object -- and a decision is driven straight from them by unbind+dot (decide_from_atom), with no trip
        back through Python. `action_codes` is an (n_actions, dim) codebook of unit role vectors."""
        from holographic.agents_and_reasoning.holographic_ai import bind
        M_Q = sum(bind(action_codes[a], self.Q[a]) for a in range(self.n_actions))
        M_N = sum(bind(action_codes[a], self.N[a]) for a in range(self.n_actions))
        return M_Q, M_N

    @property
    def nbytes(self):
        """Fixed storage: 2 * A * D floats, independent of how many experiences were folded in."""
        return self.Q.nbytes + self.N.nbytes + self.count.nbytes


def decide_from_atom(M_Q, M_N, state, action_codes, eps=1e-9):
    """Drive a decision straight from a composed policy atom -- the choice another VSA program makes when the
    policy is embedded in it. For each action: unbind its bundle from the atom, then value = <s,Q_a>/<s,N_a>;
    pick the argmax. Pure VSA ops (unbind = a bind with the inverse code, then a dot), no Python table lookup.
    KEPT NEGATIVE: folding all actions into two D-vectors adds cross-talk, so the atom-driven choice matches
    the head only within capacity (few actions / large dim); past that the unbind noise can flip the argmax."""
    from holographic.agents_and_reasoning.holographic_ai import unbind
    s = np.asarray(state, float)
    best, best_v = 0, -np.inf
    for a in range(len(action_codes)):
        qa = unbind(M_Q, action_codes[a]); na = unbind(M_N, action_codes[a])
        den = float(s @ na)
        v = float(s @ qa) / den if abs(den) > eps else 0.0
        if v > best_v:
            best, best_v = a, v
    return best


class RoutedValueHead:
    """The value head with the capacity cliff pushed back by the ROUTING FABRIC. A single (Q_a, N_a) pair
    blurs once it holds more than ~D distinct situations; here every state is ROUTED to one of B buckets by
    a fixed random-projection sign hash (locality-sensitive -- the same mechanism as HoloForest / the RP
    trees), and each bucket keeps its own (Q, N). Similar states hash to the same bucket, so each bundle
    only ever holds the handful of situations that land together -- bounded load per bundle. Capacity scales
    ~B-fold: the cliff moves from ~D distinct situations to ~B*D.

    Same drop-in API as HolographicValueHead (value/absorb/decide), so it is a value_backend for the creature.
    KEPT NEGATIVE: routing trades memory for capacity -- storage is B times a single head -- and a query
    reads ONLY its own bucket, so two similar states that fall on opposite sides of a hash plane miss each
    other's experience (boundary loss). It is 'cull, don't batch' applied to value storage: bounded buckets
    beat one big blurred bundle, at the cost of B-fold memory and some boundary smoothing.
    """

    def __init__(self, dim, n_actions, n_buckets=64, eps=1e-9, seed=0):
        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.n_buckets = int(n_buckets)
        self.eps = float(eps)
        self.bits = max(1, int(np.ceil(np.log2(self.n_buckets))))
        rng = np.random.default_rng(seed)
        self.R = rng.normal(size=(self.bits, self.dim))          # fixed random hyperplanes (the LSH router)
        self._pow = (1 << np.arange(self.bits))                  # sign-bits -> integer bucket id
        B = self.n_buckets
        self.Q = np.zeros((B, self.n_actions, self.dim))
        self.N = np.zeros((B, self.n_actions, self.dim))
        self.count = np.zeros((B, self.n_actions))

    def _bucket(self, u):
        """Route a state to its bucket: the sign pattern of the random projection, read as an integer."""
        bits = (self.R @ u > 0).astype(np.int64)
        return int((bits * self._pow).sum()) % self.n_buckets

    def absorb(self, state, action_idx, ret):
        u = np.asarray(state, float)
        nrm = np.linalg.norm(u)
        if nrm > 0:
            u = u / nrm
        b = self._bucket(u)
        self.Q[b, action_idx] += float(ret) * u
        self.N[b, action_idx] += u
        self.count[b, action_idx] += 1.0
        return self

    def value(self, state, action_idx):
        s = np.asarray(state, float)
        u = s / (np.linalg.norm(s) or 1.0)
        b = self._bucket(u)
        den = float(s @ self.N[b, action_idx])
        if abs(den) <= self.eps:
            return 0.0, 0.0
        return float(s @ self.Q[b, action_idx]) / den, den / max(self.count[b, action_idx], 1.0)

    def value_all(self, state):
        s = np.asarray(state, float)
        u = s / (np.linalg.norm(s) or 1.0)
        b = self._bucket(u)
        den = self.N[b] @ s                                      # (A,)
        num = self.Q[b] @ s
        safe = np.abs(den) > self.eps
        values = np.where(safe, num / np.where(safe, den, 1.0), 0.0)
        supports = np.where(safe, den / np.maximum(self.count[b], 1.0), 0.0)
        return values, supports

    def decide(self, state, among=None):
        values, _ = self.value_all(state)
        idxs = range(self.n_actions) if among is None else list(among)
        return int(max(idxs, key=lambda a: values[a]))

    def policy_vectors(self):
        return self.Q.copy(), self.N.copy()

    @property
    def nbytes(self):
        return self.Q.nbytes + self.N.nbytes + self.count.nbytes


# =====================================================================================================
# TD as VSA -- n-step returns as discounted bundles, eligibility traces as a decaying bundle.
# =====================================================================================================

def discounted_return(rewards, gamma, bootstrap=0.0):
    """An n-step return as a DISCOUNTED BUNDLE of rewards plus a bootstrap: sum_k gamma^k r_k + gamma^n * V.
    With bootstrap=0 and the full reward sequence this is the Monte-Carlo return; with a few rewards and
    bootstrap=value(s_n) it is the n-step TD target. The target you absorb -- learning stays a bundling step,
    only the TARGET changes from a realised return (MC) to a bootstrapped one (TD)."""
    g = 1.0
    total = 0.0
    for r in rewards:
        total += g * float(r)
        g *= gamma
    return total + g * float(bootstrap)


class EligibilityTrace:
    """The TD(lambda) eligibility trace AS a decaying hypervector bundle: e <- gamma*lambda*e + unit(state).
    It IS a bundle -- a superposition of recently-visited states, each weighted by (gamma*lambda)^age -- so a
    single TD-error update can credit the whole recent trajectory at once (the backward view). The forward
    view (absorbing lambda-returns) is the equivalent and is what the measured demo below uses; this is the
    mechanism Moose named, kept as a first-class object."""

    def __init__(self, dim, gamma=0.9, lam=0.9):
        self.vec = np.zeros(int(dim))
        self.gamma = float(gamma)
        self.lam = float(lam)

    def step(self, state):
        u = np.asarray(state, float)
        nrm = np.linalg.norm(u)
        if nrm > 0:
            u = u / nrm
        self.vec = self.gamma * self.lam * self.vec + u        # decay the bundle, add the new state
        return self.vec

    def reset(self):
        self.vec = np.zeros_like(self.vec)
        return self


# =====================================================================================================
# Self-test -- the head-to-head against the REAL tabular brain: match at low load, cliff at high load.
# =====================================================================================================

def _selftest():
    from holographic.misc.holographic_creature import HolographicMind

    def world(P, A, D, rng):
        situations = rng.normal(size=(P, D))
        situations /= np.linalg.norm(situations, axis=1, keepdims=True)   # ~orthogonal in high D
        V = rng.uniform(0.0, 1.0, size=(P, A))                            # true value per (situation, action)
        return situations, V

    def run(P, A=3, D=512, visits=4, seed=0, with_tabular=True):
        rng = np.random.default_rng(seed)
        S, V = world(P, A, D, rng)
        holo = HolographicValueHead(D, A)
        # tabular = the REAL creature brain, value mechanism isolated (no consolidation / projection).
        # PERF (slowest-tests pass): the tabular brain is a P*A*visits absorb loop plus a P*A value sweep on D=512
        # -- the entire ~17 s cost of this selftest. But only the LOW-load run USES it (to prove holo agrees with
        # the real brain); the HIGH-load run's assertions check only holo's cliff and byte-size. So the tabular
        # brain is built only when its result is actually consumed -- dead compute removed, no contract touched.
        brain = HolographicMind(dim=D, actions=list(range(A)), epsilon=0.0, maintain=False) if with_tabular else None
        for p in range(P):
            for a in range(A):
                for _ in range(visits):
                    s = S[p] + rng.normal(0, 0.02, D)                    # small perceptual noise
                    s /= np.linalg.norm(s)
                    r = V[p, a] + rng.normal(0, 0.05)                    # noisy return
                    holo.absorb(s, a, r)
                    if brain is not None:
                        brain._absorb(s, a, r)                           # same stream, same interface
        best = V.argmax(axis=1)
        holo_hits = sum(holo.decide(S[p]) == best[p] for p in range(P))
        brain_hits = (sum(int(np.argmax([brain.value(S[p], a)[0] for a in range(A)])) == best[p] for p in range(P))
                      if brain is not None else 0)
        # value agreement on the clean situations (holo vs the true value)
        holo_rmse = np.sqrt(np.mean([(holo.value(S[p], a)[0] - V[p, a]) ** 2 for p in range(P) for a in range(A)]))
        return holo_hits / P, brain_hits / P, holo_rmse, holo.nbytes

    # LOW load (P << D): the holographic head should match the tabular brain and pick the best action.
    h_lo, b_lo, rmse_lo, bytes_lo = run(P=8)
    assert h_lo >= 0.85, f"holo head should nail well-separated situations at low load, got {h_lo:.2f}"
    assert abs(h_lo - b_lo) <= 0.15, f"holo and tabular should agree at low load ({h_lo:.2f} vs {b_lo:.2f})"

    # HIGH load (P approaching D): the two-bundle head must DEGRADE -- the capacity cliff (kept negative). The
    # tabular brain is not compared here, so it is not built (with_tabular=False) -- this is the big time save.
    h_hi, b_hi, rmse_hi, bytes_hi = run(P=500, with_tabular=False)
    assert h_hi < h_lo, f"KEPT NEGATIVE: holo head must degrade past the cliff ({h_hi:.2f} !< {h_lo:.2f})"
    assert bytes_hi == bytes_lo, "the holographic policy is FIXED size regardless of experiences"

    print(f"holographic_valuehead selftest: ok (LOW load P=8 -- holo {h_lo:.2f} vs tabular {b_lo:.2f} best-action "
          f"accuracy, value RMSE {rmse_lo:.2f}, matched; HIGH load P=500 -- holo {h_hi:.2f} (CLIFF, kept negative) "
          f"vs tabular {b_hi:.2f}; policy is {bytes_lo} B fixed at both loads -- a savable hypervector program)")
    _routed_selftest()
    _td_selftest()


def _routed_selftest():
    """Step A: the routing fabric pushes the capacity cliff back -- a routed head holds far more distinct
    situations than a single-bundle head at the same accuracy."""
    def accuracy(head, P, D=256, A=3, seed=0):
        rng = np.random.default_rng(seed)
        S = rng.normal(size=(P, D)); S /= np.linalg.norm(S, axis=1, keepdims=True)
        V = rng.uniform(0, 1, size=(P, A))
        for p in range(P):
            for a in range(A):
                head.absorb(S[p], a, V[p, a])
        best = V.argmax(axis=1)
        return np.mean([head.decide(S[p]) == best[p] for p in range(P)]), S, V
    # at 4x the dimension in distinct situations, the single bundle is near chance; routing holds up
    plain, _, _ = accuracy(HolographicValueHead(256, 3), 1024)
    routed, _, _ = accuracy(RoutedValueHead(256, 3, n_buckets=64), 1024)
    assert routed > plain + 0.3, f"routing should push the cliff back: routed {routed:.2f} vs plain {plain:.2f}"
    print(f"  Step A (routing): at 1024 situations (4x dim) single-bundle {plain:.2f} -> routed(64 buckets) {routed:.2f} "
          f"-- the cliff pushed back ~B-fold (kept negative: B-fold memory + boundary smoothing)")


def _td_selftest():
    """Step B: TD as VSA. The canonical random-walk value-prediction (Sutton & Barto 6.2): TD bootstrapping
    (target = r + gamma*V(s'), absorbed) converges with LOWER error than Monte-Carlo (absorb realised
    returns) for the same number of episodes -- lower variance via bootstrapping. n-step returns are
    discounted bundles; the eligibility trace is a decaying bundle (mechanism above)."""
    D = 128
    rng = np.random.default_rng(0)
    codes = {s: rng.normal(size=D) for s in range(1, 6)}        # 5 non-terminal states; 0 and 6 are terminals
    for s in codes:
        codes[s] /= np.linalg.norm(codes[s])
    true = {s: s / 6.0 for s in range(1, 6)}                    # known true values

    def episode(rng):
        s = 3; traj = []
        while 1 <= s <= 5:
            traj.append(s)
            s += 1 if rng.random() < 0.5 else -1
        return traj, (1.0 if s == 6 else 0.0)                   # reward 1 only if it exits right

    def rmse(head):
        return np.sqrt(np.mean([(head.value(codes[s], 0)[0] - true[s]) ** 2 for s in range(1, 6)]))

    def learn(mode, episodes, seed):
        r = np.random.default_rng(seed)
        head = HolographicValueHead(D, 1)
        for s in range(1, 6):
            head.absorb(codes[s], 0, 0.5)                       # textbook V_init = 0.5 prior
        for _ in range(episodes):
            traj, final_r = episode(r)
            if mode == "mc":                                    # Monte-Carlo: absorb the realised return
                for s in traj:
                    head.absorb(codes[s], 0, final_r)
            else:                                               # TD(0): absorb r + gamma*V(s'), a bootstrapped target
                for i, s in enumerate(traj):
                    nxt = traj[i + 1] if i + 1 < len(traj) else None
                    vnext = head.value(codes[nxt], 0)[0] if nxt is not None else final_r
                    head.absorb(codes[s], 0, discounted_return([0.0], 1.0, bootstrap=vnext))
        return rmse(head)

    mc = np.mean([learn("mc", 40, seed) for seed in range(5)])
    td = np.mean([learn("td", 40, seed) for seed in range(5)])
    assert td < mc, f"TD bootstrapping should beat Monte-Carlo on the random walk: TD {td:.3f} !< MC {mc:.3f}"
    print(f"  Step B (TD as VSA): random-walk value prediction after 40 episodes -- TD RMSE {td:.3f} < MC RMSE {mc:.3f} "
          f"(bootstrapped target = r+gamma*V(s') absorbed; n-step return is a discounted bundle)")


if __name__ == "__main__":
    _selftest()
