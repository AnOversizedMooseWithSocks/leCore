"""
holographic_diffusion.py
========================

Double-diffusive dynamics, borrowed from the ocean.

When hot salty water meets cold fresh water, something strange happens. Heat
and salt share the same fluid but heat diffuses about a hundred times faster, so
a parcel pushed across the boundary loses its heat almost at once while keeping
its salt. That single rate difference lets a gravitationally STABLE column become
spontaneously unstable -- it grows thin "salt fingers," and over time the water
self-organizes into a staircase of well-mixed layers separated by sharp
interfaces. Structure appears out of a smooth gradient, powered only by the gap
between two relaxation rates.

The transferable principle: keep two co-located quantities that relax at
different rates. The FAST one (heat) equilibrates with the surroundings and
tracks the present; the SLOW one (salt) holds the persistent state, the memory.
Their divergence is the signal -- the fingering, where something real is
happening -- and when the divergence is SUSTAINED rather than a transient heat
flicker, the slow component restructures into a new layer (the staircase step).

This generalizes the PredictiveFilter (one prediction -> two timescales) and is
self-similar across levels: run it on a scalar stream and it segments regimes;
on a vector stream it separates persistent identity from fluctuating context; on
a Field, a fast field minus a slow field is a band-pass that lights up edges; on
the two-model loop, the rate ratio is a stability knob -- too close and they
collapse together, the right gap gives productive structure.

Needs: numpy and holographic_ai.py beside it.
"""

import numpy as np
from holographic.agents_and_reasoning.holographic_ai import random_vector, cosine


class DoubleDiffusion:
    """Track a stream with two components at different relaxation rates.

    heat  -- fast EMA, tracks the present (equilibrates with the surroundings)
    salt  -- slow component, holds the persistent state and steps to a new
             LAYER only when the heat/salt divergence is sustained

    observe(x) returns (divergence, layer_index, started_new_layer). Works on a
    scalar (pass [x]) or a vector alike.
    """

    def __init__(self, fast=0.5, slow=0.02, threshold=0.3, trigger=1.2):
        self.fast = fast            # heat relaxation: high = tracks quickly
        self.slow = slow            # salt relaxation: low = remembers
        self.threshold = threshold  # divergence that counts as an unstable interface
        self.trigger = trigger      # accumulated instability that breaks into a new layer
        self.heat = None
        self.salt = None
        self.instability = 0.0
        self.layer = 0

    def observe(self, x):
        x = np.asarray(x, dtype=float)
        if self.heat is None:
            self.heat = x.copy()
            self.salt = x.copy()
            return 0.0, self.layer, False

        # Fast component equilibrates toward the observation, like heat.
        self.heat = (1 - self.fast) * self.heat + self.fast * x

        # Divergence between fast and slow: the strained interface.
        divergence = float(np.linalg.norm(self.heat - self.salt))

        started = False
        if divergence > self.threshold:
            self.instability += divergence - self.threshold   # pressure builds
        else:
            self.instability *= 0.5                            # transients diffuse away

        if self.instability > self.trigger:
            # The interface breaks through: a new well-mixed layer forms.
            self.salt = self.heat.copy()
            self.layer += 1
            self.instability = 0.0
            started = True
        else:
            # Within a layer the slow component mixes gently, like slow salt.
            self.salt = (1 - self.slow) * self.salt + self.slow * x

        return divergence, self.layer, started


# ---------------------------------------------------------------------------
# DEMOS
# ---------------------------------------------------------------------------

def demo_staircase():
    print("=" * 70)
    print("DEMO 1 -- A thermohaline staircase: segment a stream into layers")
    print("=" * 70)
    rng = np.random.default_rng(0)
    stream = []
    for i in range(120):
        base = 0.0 if i < 40 else (1.0 if i < 80 else 2.0)   # two real regime shifts
        if 18 <= i <= 20:
            base = 0.9                                        # a transient spike (noise)
        stream.append(base + rng.normal(0, 0.05))

    dd = DoubleDiffusion()
    layers, salt = [], []
    for i, x in enumerate(stream):
        _, _, started = dd.observe([x])
        salt.append(float(dd.salt[0]))
        if started:
            layers.append(i)

    print("\nStream: flat near 0, a brief spike at steps 18-20, then sustained")
    print("jumps to 1.0 (step 40) and 2.0 (step 80).\n")
    print(f"  layers committed at steps : {layers}")
    print("  (the transient spike at 18-20 forms no layer -- heat diffused it away)")
    print(f"  salt staircase (every 10) : {[round(salt[i], 2) for i in range(0, 120, 10)]}")
    print("\n  The slow component steps 0 -> 1 -> 2 in discrete layers; the fast one")
    print("  absorbed the noise. Transient vs permanent, told apart by two rates.\n")


def demo_identity():
    print("=" * 70)
    print("DEMO 2 -- Identity vs context: salt keeps what heat washes away")
    print("=" * 70)
    rng = np.random.default_rng(1)
    a, b = random_vector(64, rng), random_vector(64, rng)
    dd = DoubleDiffusion(fast=0.5, slow=0.05, threshold=0.6, trigger=2.0)

    cos_a, cos_b = [], []
    for i in range(80):
        identity = a if i < 40 else b                 # the persistent thing
        observation = identity + 0.3 * random_vector(64, rng)   # plus churning context
        dd.observe(observation)
        cos_a.append(cosine(dd.salt, a))
        cos_b.append(cosine(dd.salt, b))

    print("\nEach observation is a stable identity plus fresh random context. The")
    print("slow component should hold the identity while the context averages out:\n")
    print(f"  cos(salt, A):  early {cos_a[10]:+.2f}   late-A {cos_a[35]:+.2f}   after-switch {cos_a[70]:+.2f}")
    print(f"  cos(salt, B):  early {cos_b[10]:+.2f}   late-A {cos_b[35]:+.2f}   after-switch {cos_b[70]:+.2f}")
    print("\n  Salt locks onto A, then re-forms onto B when the identity truly")
    print("  changes -- the same fingering that builds the staircase, on vectors.\n")


if __name__ == "__main__":
    demo_staircase()
    demo_identity()
