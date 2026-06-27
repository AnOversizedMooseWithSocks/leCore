"""exp_value_memory.py -- HEAD-TO-HEAD: the bespoke creature value memory vs. value-learning built on the
unified mind's memory (SelfOrganizingMind). The decision this informs: is the bespoke per-action prototype
memory a redundant OLD path that the unified memory can replace, or does it earn its place?

FAIRNESS: both learners are handed the SAME perceive-encoded state vectors (so the ENCODER is not a
variable -- only the value MEMORY is), the SAME episode stream (same seed), and the SAME epsilon-greedy
exploration. We compare the VALUE estimate's greedy policy, not the full decide() (which adds novelty/danger
policy on the bespoke side) -- isolating the memory.

TASK: 16 egocentric situations = 4 food directions x 4 distractor combos; the best action is "move toward
food". The learner is NOT told the rule; it learns per-situation from rewards (+1 best, -0.5 otherwise). We
train on 12 situations and test greedy accuracy on all 16 -- in-sample on the 12, GENERALIZATION on the 4
held-out (same food dirs, distractor combos never trained) -- which is the holographic similarity test both
memories rely on.
"""
import sys; sys.path.insert(0, "/home/claude/work")
import numpy as np
from holographic_unified import UnifiedMind
from holographic_creature import HolographicMind
from holographic_organizer import SelfOrganizingMind

ACTIONS = ["N", "S", "E", "W"]
DIRS = ["N", "S", "E", "W"]


def all_situations():
    """16 situations; best action == food direction. Distractors d1,d2 are irrelevant noise features."""
    sits = []
    for food in DIRS:
        for d1 in ("0", "1"):
            for d2 in ("0", "1"):
                senses = {"food": food, "d1": d1, "d2": d2}
                sits.append((senses, DIRS.index(food)))   # (senses, best_action_idx)
    return sits


class UnifiedValueLearner:
    """Value-learning on the UNIFIED memory: classify the situation into a class (the same prototype
    management the bespoke memory uses, but it is the ONE memory), and track mean return per
    (class, action) in a small table. The candidate replacement for the bespoke per-action banks."""

    def __init__(self, dim, n_actions, seed=0, novelty_floor=0.5):
        self.mem = SelfOrganizingMind(dim=dim, seed=seed)   # the unified memory (vector API only)
        self.n_actions = n_actions
        self.floor = novelty_floor
        self.vtable = {}        # (class_label, action) -> [sum_return, count]
        self._next = 0

    def _situation(self, sv, grow):
        lab, sim = self.mem.classify_vector(sv)
        if grow and (lab is None or sim < self.floor):     # a new kind of situation -> mint a class
            lab = self._next; self._next += 1
            self.mem.observe_vector(sv, lab)
        return lab

    def learn(self, sv, action, ret):
        lab = self._situation(sv, grow=True)
        e = self.vtable.get((lab, action), [0.0, 0])
        e[0] += ret; e[1] += 1
        self.vtable[(lab, action)] = e

    def value(self, sv, action):
        lab = self._situation(sv, grow=False)              # read-only: never grow memory on a value query
        e = self.vtable.get((lab, action))
        return e[0] / e[1] if e else 0.0

    def greedy(self, sv):
        return max(range(self.n_actions), key=lambda a: self.value(sv, a))


class SoftUnifiedValueLearner:
    """Fairer rival: value-learning on the unified memory but with the SAME soft k-NN returns-regression
    the bespoke memory uses -- cosine-weighted average of returns over the nearest situation-classes (via
    label_scores), with per-class per-action returns. Isolates whether the bespoke's edge is its MEMORY or
    just its soft mechanism."""

    def __init__(self, dim, n_actions, seed=0, novelty_floor=0.5, k=12):
        self.mem = SelfOrganizingMind(dim=dim, seed=seed)
        self.n_actions = n_actions
        self.floor = novelty_floor
        self.k = k
        self.cval = {}          # class_label -> per-action [sum_return, count]
        self._next = 0

    def _mint_or_match(self, sv):
        lab, sim = self.mem.classify_vector(sv)
        if lab is None or sim < self.floor:
            lab = self._next; self._next += 1
            self.mem.observe_vector(sv, lab)
        return lab

    def learn(self, sv, action, ret):
        lab = self._mint_or_match(sv)
        e = self.cval.setdefault(lab, [[0.0, 0] for _ in range(self.n_actions)])
        e[action][0] += ret; e[action][1] += 1

    def value(self, sv, action):
        scores = self.mem.live.label_scores(sv)                 # {class_label: best cosine} over the unified store
        if not scores:
            return 0.0
        items = sorted(scores.items(), key=lambda kv: -kv[1])[:self.k]
        num = den = 0.0
        for lab, sim in items:
            w = max(sim, 0.0)                              # clip-at-0, mirroring the bespoke weighting
            e = self.cval.get(lab)
            if e and e[action][1] > 0 and w > 0:
                num += w * (e[action][0] / e[action][1]); den += w
        return num / den if den > 1e-9 else 0.0

    def greedy(self, sv):
        return max(range(self.n_actions), key=lambda a: self.value(sv, a))


def run_one(seed, n_present=40, eps=0.2, dim=512, floor=0.5, checkpoints=(10, 20, 40)):
    """One trial: train both learners on the SAME stream, return greedy accuracy at each checkpoint for
    in-sample (trained) and held-out (generalization) situations."""
    rng = np.random.default_rng(seed)
    mind = UnifiedMind(dim=dim, seed=seed)                  # the ONE encoder, shared by both learners
    enc = {}                                                # cache perceive() per situation (deterministic)

    def vec(senses):
        key = tuple(sorted(senses.items()))
        if key not in enc:
            enc[key] = mind.perceive(dict(senses), "record")
        return enc[key]

    sits = all_situations()
    order = rng.permutation(len(sits))
    train_idx = sorted(order[:12]); held_idx = sorted(order[12:])   # 12 trained, 4 held out

    old = HolographicMind(dim, ACTIONS, seed=seed)          # bespoke per-action prototype memory
    new = UnifiedValueLearner(dim, len(ACTIONS), seed=seed, novelty_floor=floor)
    soft = SoftUnifiedValueLearner(dim, len(ACTIONS), seed=seed, novelty_floor=floor)

    def reward(best, a):
        return 1.0 if a == best else -0.5

    def greedy_old(sv):
        vals = [old.value(sv, a)[0] for a in range(len(ACTIONS))]
        return int(np.argmax(vals))

    def accuracy(greedy_fn, idxs):
        return float(np.mean([greedy_fn(vec(sits[i][0])) == sits[i][1] for i in idxs]))

    results = {k: [] for k in ("old_in", "old_gen", "new_in", "new_gen", "soft_in", "soft_gen")}
    presented = 0
    for rep in range(max(checkpoints)):
        for i in rng.permutation(train_idx):               # one pass over the trained situations
            senses, best = sits[i]; sv = vec(senses)
            # SAME exploration draw drives ALL THREE so the streams match; only the memory differs
            explore = rng.random() < eps
            rand_a = int(rng.integers(len(ACTIONS)))
            a_old = rand_a if explore else greedy_old(sv)
            a_new = rand_a if explore else new.greedy(sv)
            a_soft = rand_a if explore else soft.greedy(sv)
            old.remember([sv], [int(a_old)], [reward(best, int(a_old))])
            new.learn(sv, int(a_new), reward(best, int(a_new)))
            soft.learn(sv, int(a_soft), reward(best, int(a_soft)))
            presented += 1
        if (rep + 1) in checkpoints:
            results["old_in"].append(accuracy(greedy_old, train_idx))
            results["old_gen"].append(accuracy(greedy_old, held_idx))
            results["new_in"].append(accuracy(new.greedy, train_idx))
            results["new_gen"].append(accuracy(new.greedy, held_idx))
            results["soft_in"].append(accuracy(soft.greedy, train_idx))
            results["soft_gen"].append(accuracy(soft.greedy, held_idx))
    return results, checkpoints


def main():
    seeds = range(6)
    floor = 0.5
    agg = {k: [] for k in ("old_in", "old_gen", "new_in", "new_gen", "soft_in", "soft_gen")}
    cps = None
    for s in seeds:
        res, cps = run_one(s, floor=floor)
        for k in agg:
            agg[k].append(res[k])
    arr = {k: np.array(v) for k, v in agg.items()}         # shape (n_seeds, n_checkpoints)

    print(f"Head-to-head: bespoke value memory vs unified-memory value-learners")
    print(f"  {len(list(seeds))} seeds, dim 512, 12 trained / 4 held-out situations, novelty_floor={floor}")
    print(f"  greedy policy accuracy (chance = 0.25), mean across seeds\n")
    hdr = f"  {'passes':>7} | {'BESPOKE':>16} | {'UNIFIED-hard':>16} | {'UNIFIED-soft':>16}"
    print(hdr); print(f"  {'':>7} | {'in':>7} {'gen':>7} | {'in':>7} {'gen':>7} | {'in':>7} {'gen':>7}")
    for j, cp in enumerate(cps):
        print(f"  {cp:>7} | {arr['old_in'][:,j].mean():>7.2f} {arr['old_gen'][:,j].mean():>7.2f} "
              f"| {arr['new_in'][:,j].mean():>7.2f} {arr['new_gen'][:,j].mean():>7.2f} "
              f"| {arr['soft_in'][:,j].mean():>7.2f} {arr['soft_gen'][:,j].mean():>7.2f}")
    print()
    fin = -1
    print(f"  FINAL ({cps[fin]} passes), mean +/- std across seeds:")
    for name, ki, kg in (("bespoke", "old_in", "old_gen"),
                         ("unified-hard", "new_in", "new_gen"),
                         ("unified-soft", "soft_in", "soft_gen")):
        print(f"    {name:>13}:  in-sample {arr[ki][:,fin].mean():.3f}+/-{arr[ki][:,fin].std():.3f}"
              f"   generalization {arr[kg][:,fin].mean():.3f}+/-{arr[kg][:,fin].std():.3f}")

    # robustness: sweep the unified learners' one hyperparameter so the result is not a cherry-picked floor
    print("\n  unified-memory robustness across novelty_floor (final in / gen):")
    for fl in (0.3, 0.4, 0.5, 0.6, 0.7):
        hi = hg = si = sg = 0.0
        for s in seeds:
            res, _ = run_one(s, floor=fl)
            hi += res["new_in"][-1]; hg += res["new_gen"][-1]
            si += res["soft_in"][-1]; sg += res["soft_gen"][-1]
        n = len(list(seeds))
        print(f"    floor={fl}:  hard in {hi/n:.3f} gen {hg/n:.3f}   |   soft in {si/n:.3f} gen {sg/n:.3f}")


if __name__ == "__main__":
    main()
