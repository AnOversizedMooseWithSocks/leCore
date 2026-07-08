"""
holographic_moe.py -- a mixture of experts with a LEARNED, holographic gate.

The general Mind dispatches by a rule (which verb you called, what type the input
is). That is routing, but it is not a mixture of experts -- MoE's defining piece
is a *learned* gate that, per input, picks which specialist to trust, trained from
outcomes. This module adds exactly that, and reuses the creature's brain as the
gate: encode the input, let the brain DECIDE which expert to consult, and reward
it when the expert it chose was right. Sparse top-1 routing -- only the chosen
expert runs -- and the gate learns the routing with no gradients, just the same
perceive/decide/remember loop that learned to forage and to navigate.

Why this is the real thing and not ensembling: averaging experts that disagree
was already measured to HURT (see the navigator work). Here the experts are
specialists that each know only part of the problem and are confidently wrong
outside it, so mixing them is hopeless -- you have to ROUTE. The demo measures the
learned gate against the honest baselines (best single expert, random routing, and
an oracle that always picks the right specialist) so the gate has to earn it.
"""

import numpy as np

from holographic.misc.holographic_creature import HolographicMind
from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder, _PrototypeClassifier


class Expert:
    """A specialist: a prototype classifier over the shared encoding, trained only
    on its own slice of the world, plus the set of labels it actually knows (so we
    can score an oracle router and so a wrong route is correctly penalised)."""

    def __init__(self, name):
        self.name = name
        self.clf = _PrototypeClassifier()
        self.labels = set()

    def learn(self, vec, label):
        self.clf.learn(vec, label)
        self.labels.add(label)

    def predict(self, vec):
        return self.clf.predict(vec)


class LinearExpert:
    """A deliberately DIFFERENT kind of expert: a ridge-fit linear read-out with a
    softmax. Linear/softmax models extrapolate with ever-growing logits, so they
    are famously OVERCONFIDENT outside their training region -- their reported
    confidence stops tracking competence. It is here to expose the one regime where
    a learned gate matters: confidence routing is fooled by this miscalibration,
    while a gate trained on outcomes is indifferent to confidence scale. (temp is
    the softmax sharpness; a high value models an uncalibrated, overconfident model
    -- the classic neural-net-out-of-distribution failure, in miniature.)"""

    def __init__(self, name, temp=20.0):
        self.name = name
        self.temp = temp
        self.labels = set()
        self.classes = []
        self.W = None

    def fit(self, vecs, labels):
        self.classes = sorted(set(labels))
        self.labels = set(self.classes)
        idx = {c: i for i, c in enumerate(self.classes)}
        X = np.stack(vecs)
        Y = np.zeros((len(labels), len(self.classes)))
        for r, c in enumerate(labels):
            Y[r, idx[c]] = 1.0
        self.W = np.linalg.solve(X.T @ X + 1e-2 * np.eye(X.shape[1]), X.T @ Y)
        return self

    def predict(self, vec):
        logits = vec @ self.W
        e = np.exp(self.temp * (logits - logits.max()))
        p = e / e.sum()
        return self.classes[int(p.argmax())], float(p.max())


class GatedMixture:
    """Experts + a learned holographic gate (the creature's brain) that routes
    each encoded input to one expert, trained from reward."""

    def __init__(self, dim=1024, seed=0, number_range=(-4.0, 4.0)):
        self.dim = dim
        self.encoder = UniversalEncoder(dim, seed=seed, number_range=number_range)
        self.experts = []
        self.gate = None
        self._rel = {}                   # expert_idx -> [wins, uses] for the hybrid gate

    # -- build specialists -------------------------------------------------
    def add_expert(self, name, examples):
        """examples: list of (input, label, modality). Trains one specialist on
        just this slice."""
        e = Expert(name)
        for x, label, modality in examples:
            e.learn(self.encoder.encode(x, modality), label)
        self.experts.append(e)
        return self

    def add_linear_expert(self, name, examples, temp=20.0):
        """Add a heterogeneous specialist: a linear+softmax model (see LinearExpert)
        trained on its own slice. examples: list of (input, label, modality)."""
        le = LinearExpert(name, temp=temp)
        le.fit([self.encoder.encode(x, m) for x, _, m in examples],
               [lab for _, lab, _ in examples])
        self.experts.append(le)
        return self

    def add_prebuilt_expert(self, expert):
        """Plug in any object exposing .predict(vec)->(label, conf), .labels, .name."""
        self.experts.append(expert)
        return self

    def _ensure_gate(self):
        if self.gate is None:
            actions = [str(i) for i in range(len(self.experts))]
            self.gate = HolographicMind(self.dim, actions, k=12, epsilon=0.4,
                                        novelty_bonus=0.15, memory_cap=8000, seed=1)

    # -- train the gate from reward (was the routed expert right?) ---------
    def train_gate(self, examples, epochs=10, eps_start=0.45, seed=0):
        """examples: list of (input, label, modality). The gate picks an expert,
        we ask it, and reward 1 if its answer matched the truth."""
        self._ensure_gate()
        rng = np.random.default_rng(seed)
        order = list(range(len(examples)))
        for ep in range(epochs):
            self.gate.epsilon = max(0.05, eps_start * (1 - ep / epochs))
            rng.shuffle(order)
            for j in order:
                x, label, modality = examples[j]
                vec = self.encoder.encode(x, modality)
                a = self.gate.decide(vec, explore=True)
                pred, _ = self.experts[a].predict(vec)
                reward = 1.0 if pred == label else 0.0
                self.gate.remember([vec], [a], [reward])

    # -- inference and baselines ------------------------------------------
    def route(self, vec):
        return self.gate.decide(vec, explore=False, epsilon=0.0)

    def predict(self, x, modality=None):
        vec = self.encoder.encode(x, modality)
        a = self.route(vec)
        return self.experts[a].predict(vec)[0], self.experts[a].name

    def predict_with(self, expert_idx, x, modality=None):
        vec = self.encoder.encode(x, modality)
        return self.experts[expert_idx].predict(vec)[0]

    def oracle_label(self, x, label, modality=None):
        """An upper bound: route to an expert that actually knows this label, then
        return its prediction (still its real answer, not a freebie)."""
        vec = self.encoder.encode(x, modality)
        for e in self.experts:
            if label in e.labels:
                return e.predict(vec)[0]
        return None

    # A gate-FREE baseline: route to whichever expert is surest of itself. For a
    # bank of holographic specialists this is a strong baseline, because an
    # unfamiliar input simply produces low similarity -- so confidence already
    # tracks competence, with nothing to train.
    def confidence_route(self, vec):
        best, best_i = -2.0, 0
        for i, e in enumerate(self.experts):
            _, s = e.predict(vec)
            if s > best:
                best, best_i = s, i
        return best_i

    def predict_by_confidence(self, x, modality=None):
        vec = self.encoder.encode(x, modality)
        return self.experts[self.confidence_route(vec)].predict(vec)[0]

    # -- the schema gate: route by who COMPRESSES the raw input best -------
    def set_expert_schema(self, expert_idx, data, modality="text", cuts=(0, 120, 350)):
        """Give an expert a learned schema over its own domain's raw inputs, so routing can be
        a description-length contest. Deterministic and training-free, unlike the RL gate."""
        from holographic.simulation_and_physics.holographic_schema import SchemaGenerator
        self.experts[expert_idx].schema = SchemaGenerator(modality=modality, cuts=cuts).fit(data)
        return self

    def schema_route(self, raw_input):
        """Route to the expert whose schema needs the fewest bits to encode the input -- the
        one that understands it. Measured against the LEARNED gate on short-text routing it won
        in both regimes tested: 84% vs ~47% on author/style routing, 61% vs 42% on Reuters topic
        routing, with no training and no seed dependence. The standing caveat is the same as for
        confidence routing -- it assumes understanding tracks competence, so for MISCALIBRATED
        experts (understand the input, answer it wrong) the reward-trained gate is still the one
        to use. Kept as an option, not a replacement, on that measured boundary."""
        from holographic.simulation_and_physics.holographic_schema import compression_gate
        pool = [(i, e.schema) for i, e in enumerate(self.experts)
                if getattr(e, "schema", None) is not None]
        if not pool:
            raise RuntimeError("no expert has a schema -- call set_expert_schema() first")
        return compression_gate(raw_input, pool)[0][1]

    def predict_by_schema(self, raw_input, modality=None):
        i = self.schema_route(raw_input)
        return self.experts[i].predict(self.encoder.encode(raw_input, modality))[0]

    # -- the hybrid gate: compression routing, demoted by a thin reward signal ----
    def _reliability(self, i):
        w, u = self._rel.get(i, (0, 0))
        return (w + 1) / (u + 2)                 # smoothed; unused -> 0.5 (a constant offset)

    def route_hybrid(self, raw_input):
        """Route by compression, biased by each expert's surprise penalty -log2(reliability).
        With no feedback this is identical to schema_route; feedback demotes liars."""
        import math
        from holographic.simulation_and_physics.holographic_schema import compression_gate
        pool = [(i, e.schema) for i, e in enumerate(self.experts)
                if getattr(e, "schema", None) is not None]
        if not pool:
            raise RuntimeError("no expert has a schema -- call set_expert_schema() first")
        bias = {i: -math.log2(self._reliability(i)) for i, _ in pool}
        return compression_gate(raw_input, pool, bias)[0][1]

    def observe_hybrid(self, expert_idx, correct):
        w, u = self._rel.get(expert_idx, (0, 0))
        self._rel[expert_idx] = (w + int(bool(correct)), u + 1)
        return self

    def calibrate_hybrid(self, examples):
        """One online pass: route by the hybrid gate, see whether the chosen expert was right,
        update its reliability. Cheap -- one scalar per expert, no gradients, no RL brain."""
        for x, label, modality in examples:
            i = self.route_hybrid(x)
            pred, _ = self.experts[i].predict(self.encoder.encode(x, modality))
            self.observe_hybrid(i, pred == label)
        return self

    def predict_by_hybrid(self, x, modality=None):
        i = self.route_hybrid(x)
        return self.experts[i].predict(self.encoder.encode(x, modality))[0]


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

def _make_modalities(rng):
    """Build labelled examples in three modalities for cross-modal experts."""
    corpus = ["the cat sat", "a dog barked", "the kitten purred", "the puppy ran",
              "the car drove", "a truck honked", "the engine roared", "the sedan parked"]
    text = ([("the cat sat", "animal", None), ("a dog barked", "animal", None),
             ("the kitten purred", "animal", None), ("the puppy slept", "animal", None),
             ("the car drove", "vehicle", None), ("a truck honked", "vehicle", None),
             ("the engine roared", "vehicle", None), ("the sedan parked", "vehicle", None)])

    def img(kind):
        g = np.zeros((10, 10))
        if kind == "h": g[::2, :] = 1
        elif kind == "v": g[:, ::2] = 1
        else: g[::2, ::2] = 1; g[1::2, 1::2] = 1
        return g + 0.25 * rng.standard_normal((10, 10))
    image = [(img(k), "img_" + k, "image") for k in ("h", "v", "x") for _ in range(4)]

    def tone(f):
        t = np.linspace(0, 1, 256, endpoint=False)
        return np.abs(np.fft.rfft(np.sin(2 * np.pi * f * t) + 0.3 * rng.standard_normal(256)))
    audio = [(tone(f), "tone_" + n, "vector")
             for n, f in {"lo": 5, "hi": 40}.items() for _ in range(6)]
    return text, image, audio, corpus


def demo_moe():
    print("=" * 70)
    print("A holographic mixture of experts -- the brain learns to route")
    print("=" * 70)
    rng = np.random.default_rng(0)

    # ---- scenario 1: cross-modal specialists -----------------------------
    text, image, audio, corpus = _make_modalities(rng)
    moe = GatedMixture(dim=1024, seed=0)
    moe.encoder.learn_text(corpus)
    moe.add_expert("text", text)       # knows animal/vehicle only
    moe.add_expert("image", image)     # knows img_h/v/x only
    moe.add_expert("audio", audio)     # knows tone_lo/hi only

    train = text + image + audio
    moe.train_gate(train, epochs=12)

    # held-out test set across all three modalities
    test = []
    for s, lab in [("the cat ran", "animal"), ("the truck drove", "vehicle"),
                   ("a dog sat", "animal"), ("the engine parked", "vehicle")]:
        test.append((s, lab, None))
    for k in ("h", "v", "x"):
        for _ in range(4):
            g = np.zeros((10, 10))
            if k == "h": g[::2, :] = 1
            elif k == "v": g[:, ::2] = 1
            else: g[::2, ::2] = 1; g[1::2, 1::2] = 1
            test.append((g + 0.25 * rng.standard_normal((10, 10)), "img_" + k, "image"))
    for n, f in {"lo": 5, "hi": 40}.items():
        for _ in range(6):
            t = np.linspace(0, 1, 256, endpoint=False)
            test.append((np.abs(np.fft.rfft(np.sin(2 * np.pi * f * t)
                         + 0.3 * rng.standard_normal(256))), "tone_" + n, "vector"))

    def acc(fn):
        return np.mean([fn(x, lab, mod) == lab for x, lab, mod in test])

    learned = acc(lambda x, lab, mod: moe.predict(x, mod)[0])
    oracle = acc(lambda x, lab, mod: moe.oracle_label(x, lab, mod))
    conf = acc(lambda x, lab, mod: moe.predict_by_confidence(x, mod))
    rng2 = np.random.default_rng(7)
    def rand_route(x, lab, mod):
        vec = moe.encoder.encode(x, mod)
        return moe.experts[rng2.integers(len(moe.experts))].predict(vec)[0]
    randomr = acc(rand_route)
    singles = [acc(lambda x, lab, mod, i=i: moe.predict_with(i, x, mod))
               for i in range(len(moe.experts))]

    print("\n  Scenario 1 -- three cross-modal specialists (text / image / audio),")
    print("  each knows ONLY its own labels. The gate sees just the encoded vector")
    print("  (never the modality) and must learn where to send each input.\n")
    print(f"    best single expert : {max(singles)*100:>3.0f}%   (only right on its own modality)")
    print(f"    random routing     : {randomr*100:>3.0f}%")
    print(f"    confidence routing : {conf*100:>3.0f}%   (gate-free: trust the surest expert)")
    print(f"    LEARNED gate (MoE) : {learned*100:>3.0f}%")
    print(f"    oracle routing     : {oracle*100:>3.0f}%   (upper bound)")

    # ---- scenario 2: same modality, routing must be by CONTENT -----------
    # Two experts each accurate on one half of the number line; the gate has to
    # learn to route by value, proving it is not just detecting input type. The
    # encoder's range is set to the data so low and high values are separable,
    # and each half has two sub-labels with a margin so the experts do real work.
    moe2 = GatedMixture(dim=1024, seed=2, number_range=(0.0, 1.0))
    def lo_lab(v): return "lo_a" if v < 0.18 else "lo_b"     # gap around 0.18-0.30
    def hi_lab(v): return "hi_a" if v < 0.68 else "hi_b"     # gap around 0.68-0.80
    lo_examples, hi_examples = [], []
    for _ in range(50):
        v = rng.uniform(0.02, 0.16); lo_examples.append((v, lo_lab(v), None))
        v = rng.uniform(0.30, 0.48); lo_examples.append((v, lo_lab(v), None))
        v = rng.uniform(0.52, 0.66); hi_examples.append((v, hi_lab(v), None))
        v = rng.uniform(0.80, 0.98); hi_examples.append((v, hi_lab(v), None))
    moe2.add_expert("low-range", lo_examples)
    moe2.add_expert("high-range", hi_examples)
    moe2.train_gate(lo_examples + hi_examples, epochs=14)

    test2 = []
    for _ in range(60):
        v = rng.uniform(0.02, 0.16); test2.append((v, lo_lab(v), None))
        v = rng.uniform(0.30, 0.48); test2.append((v, lo_lab(v), None))
        v = rng.uniform(0.52, 0.66); test2.append((v, hi_lab(v), None))
        v = rng.uniform(0.80, 0.98); test2.append((v, hi_lab(v), None))
    learned2 = np.mean([moe2.predict(x, mod)[0] == lab for x, lab, mod in test2])
    conf2 = np.mean([moe2.predict_by_confidence(x, mod) == lab for x, lab, mod in test2])
    single2 = max(np.mean([moe2.predict_with(i, x, mod) == lab for x, lab, mod in test2])
                  for i in range(2))
    print("\n  Scenario 2 -- two experts on different halves of the number line,")
    print("  SAME modality. The gate must route by content (value), not type.\n")
    print(f"    best single expert : {single2*100:>3.0f}%")
    print(f"    confidence routing : {conf2*100:>3.0f}%   (gate-free baseline)")
    print(f"    LEARNED gate (MoE) : {learned2*100:>3.0f}%")
    print("\n  The gate learned the routing from reward alone -- the brain deciding")
    print("  which mind to think with. Honest finding: for a homogeneous bank of")
    print("  holographic specialists, gate-free confidence routing is just as good,")
    print("  because an unfamiliar input naturally looks unfamiliar (low similarity).")
    print("  A learned gate earns its keep when confidence is unreliable instead --")
    print("  heterogeneous or miscalibrated experts that are confidently wrong.")


def demo_heterogeneous():
    """The regime where a learned gate finally beats the gate-free baseline: a
    heterogeneous bank where one expert is MISCALIBRATED (confidently wrong out of
    its domain). Confidence routing gets fooled; the outcome-trained gate does not.
    """
    print("\n" + "=" * 70)
    print("When the learned gate matters: a miscalibrated, heterogeneous expert")
    print("=" * 70)
    rng = np.random.default_rng(0)
    L = 48
    moe = GatedMixture(dim=512, seed=0)
    # Two regions, well separated; region A holds classes 0,1 and region B holds 2,3.
    cA, cB = rng.standard_normal(L) * 3, rng.standard_normal(L) * 3
    sub = rng.standard_normal((4, L)) * 0.6
    def sample(c):
        return (cA if c < 2 else cB) + sub[c] + 0.4 * rng.standard_normal(L)

    # holographic specialist on region A (classes 0,1); overconfident linear
    # specialist on region B (classes 2,3) -- neither is good on both regions.
    A_ex = []
    for _ in range(300):
        c = int(rng.integers(2)); A_ex.append((sample(c), c, "vector"))
    B_ex = []
    for _ in range(300):
        c = 2 + int(rng.integers(2)); B_ex.append((sample(c), c, "vector"))

    moe.add_expert("holo-A", A_ex)                       # well-calibrated specialist
    moe.add_linear_expert("linear-B", B_ex, temp=25.0)   # overconfident specialist
    moe.train_gate(A_ex + B_ex, epochs=8)

    test = [(sample(c := int(rng.integers(4))), c, "vector") for _ in range(800)]
    def acc(fn): return np.mean([fn(x, lab, mod) == lab for x, lab, mod in test])
    single = max(acc(lambda x, lab, mod, i=i: moe.predict_with(i, x, mod)) for i in range(2))
    conf = acc(lambda x, lab, mod: moe.predict_by_confidence(x, mod))
    learned = acc(lambda x, lab, mod: moe.predict(x, mod)[0])
    oracle = acc(lambda x, lab, mod: moe.oracle_label(x, lab, mod))

    # how overconfident is the linear expert where it is WRONG (region A)?
    misconf = np.mean([moe.experts[1].predict(moe.encoder.encode(sample(rng.integers(2)),
                       "vector"))[1] for _ in range(200)])
    print(f"\n  The linear expert reports {misconf*100:.0f}% confidence on region A --")
    print("  where it is always WRONG. That is what breaks the gate-free heuristic.\n")
    print(f"    best single expert : {single*100:>3.0f}%")
    print(f"    confidence routing : {conf*100:>3.0f}%   (fooled by the overconfident expert)")
    print(f"    LEARNED gate (MoE) : {learned*100:>3.0f}%   (routes by learned reliability, not confidence)")
    print(f"    oracle routing     : {oracle*100:>3.0f}%")
    print("\n  This is the missing half of the story: a learned gate is unnecessary")
    print("  for a homogeneous bank of calibrated holographic experts (confidence")
    print("  routing ties it), but it is exactly what you want once an expert can be")
    print("  confidently wrong -- which is the normal case for heterogeneous models.")


if __name__ == "__main__":
    demo_moe()
    demo_heterogeneous()
