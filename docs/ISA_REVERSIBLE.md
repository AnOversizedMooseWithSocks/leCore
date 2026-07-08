# ISA_REVERSIBLE.md — the reversible / error-correction model (ISA-8, the frontier)

*The last item of the VSA ISA spine, and the most conceptual. It names what the engine has been all along, imports
a discipline from reversible/error-correcting computing, and ships one practical, measured payoff — an auto-cleanup
scheduler. The honest framing matters more here than anywhere else on the spine, so the loud negative comes first.*

## The loud negative, up front: this is an ANALOGY, not physics

VSA is **not** a quantum computer. There is no exponential superposition, no physical entanglement, no quantum
speedup. FHRR's `bind` happens to be a diagonal-unitary operator — a per-frequency phase rotation, which is
*structurally* gate-like — and that is a genuinely useful lens for capacity bounds. But it is a lens. What ISA-8
actually adopts is the **discipline** of reversible/error-correcting computing — track an error budget, correct
before the cliff, keep reversibility bookkeeping — not any claim about the physics. Overclaiming the quantum
connection would be exactly the kind of unmeasured assertion this project exists to refuse.

So of the three parts below, **(b) the scheduler is the practical, measured core**; (a) the reversibility audit
is doable, testable framing; (c) the quantum-gate connection is conceptual scaffolding. They are labelled as such.

## (a) The reversibility audit — which instructions are reversible

VSA assembly is partly reversible. Classifying the base instructions (verified empirically in
`holographic_reversible.py`):

| Instruction | Class | Why |
|---|---|---|
| `bind` | **reversible** | exact inverse is `unbind` by the same (unitary) key |
| `unbind` | **reversible** | exact inverse is `bind` by the same key |
| `permute` | **reversible** | exact inverse is `permute` by the negated shift |
| `involution` | **reversible** | self-inverse: `involution(involution(x)) == x` |
| `bundle` | **lossy** | a sum — the summands are not exactly recoverable; coherence is spent here |
| `superpose` | **lossy** | a raw sum (no renormalization) — same |
| `cleanup` | **lossy** | projection to the nearest codebook atom — discards the residual |

The reading that organises everything: the **lossy** instructions are exactly where information (coherence) is
spent or restored, and `cleanup` is *error correction* — it snaps a drifted vector back onto the codebook
manifold, throwing away the accumulated error (and a little signal with it). `capacity` is the coherence budget;
re-anchoring and the coherence-gate are error-correction rounds. This is the same lesson the whole ISA spine kept
re-learning (the bundled disk, the bundled register file, the permute-stack): superposition buys composability
and pays in a crosstalk cliff.

## (b) The error budget + auto-cleanup scheduler (the measured payoff)

A long "program" — a sequence of binds/unbinds/perturbations on a vector — accumulates crosstalk and drifts from
the truth, eventually crossing a cliff where `cleanup` would snap to the *wrong* atom. The scheduler inserts a
`cleanup` **before** the cliff, using an **oracle-free health signal**: the cosine of the running vector to its
nearest codebook atom (1.0 on a clean atom, falling as it drifts into no-man's-land — the capacity diagnostic's
SNR proxy). This generalises the shipped coherence-gate from store-*maintenance* to program-*execution*.

- **adaptive**: clean only when `health < floor` — correct just-in-time, matching cleanups to the actual damage.
- **fixed**: clean every *k* steps regardless.

**Measured (bursty damage — heavy steps interleaved with calm, the regime where matching damage pays):** the
adaptive scheduler holds the program output above a 0.9 fidelity threshold (frac-below = 0.000) at **5 cleanups**;
the *best fixed cadence that matches that fidelity* (k=3) needs **16** — roughly **a third**, echoing the
coherence-gate's "matched accuracy at ~⅓ the passes." Fixed cadences that try to use fewer cleanups (k=4 → 12,
k=6 → 8) start dropping below the threshold. The win is entirely from matching cleanups to the bursty damage:
clean right after each burst, skip the calm.

*Kept honest:* under **constant** damage a fixed cadence is already near-optimal, so the adaptive advantage is
specific to **variable** damage rates — exactly when you cannot know the right fixed *k* in advance. And the
health signal is a proxy: it must trigger early enough (a floor near the drift the bursts cause) that the nearest
atom is still the true one when the cleanup fires.

## (c) The quantum-gate connection (framing only)

FHRR represents a hypervector as unit-magnitude phasors and binds by multiplying them — a **diagonal unitary**, a
per-frequency phase rotation. That is the same shape as a layer of single-qubit phase gates, which is why FHRR
sits, as the tensor-network seat put it, "a stop on the road from quantum amplitudes to classical hypervectors,"
and why unitarity is what makes `bind` exactly invertible (audit row 1). Useful for reasoning about capacity as a
coherence budget — and nothing more. See the loud negative.

---

*Seats: Stoudenmire (quantum-inspired / tensor networks; the FHRR-as-diagonal-unitary framing) + the
FHRR/honesty/coherence threads. The practical core is the scheduler; the rest is the discipline it borrows.*
