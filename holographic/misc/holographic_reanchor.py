"""Re-anchoring is load-bearing for deep traversal -- the audit, and the contrast the other tests don't show.

WHY THIS EXISTS (RAY-2)
-----------------------
In the FFT/phasor domain a `bind` is elementwise complex multiplication, so a chain of binds is a ray whose
recoverable signal ATTENUATES multiplicatively with each hop. The path-traced fix is next-event estimation: connect
to a KNOWN anchor at every bounce instead of hoping the path stays valid. Here the anchor is the codebook and the
connection is `cleanup` -- re-project the intermediate state onto the manifold (snap it to the nearest stored atom)
every step. "A shared kernel is not a shared manifold": without that re-projection the accumulated state drifts off
the manifold and the signal collapses.

THE AUDIT (the VALIDATE half). Every deep-composition / traversal faculty in the engine already re-anchors at each
step: `gated_traverse` (RAY-1) and `directed_traverse` (RAY-3) clean up inside their step before continuing; the
peel-based `decode_structure` cleans up per peel (measured: iterated decode 2 -> 15 hops); the pack/recover and
nested-decode paths resolve each recovered item to the codebook. The audit found NO deep path missing the discipline,
so there is no cleanup to add -- RAY-2 is a validation, not a build.

THE CONTRAST (what the existing tests omit). The traverse self-test shows the RE-ANCHORED traversal works, but never
shows it FAILING WITHOUT re-anchoring -- which is the whole claim. This module drives the engine's real
`gated_traverse` on a directed linked list two ways, identical except for the one line that matters:
  * RE-ANCHORED step: carry the CLEANED node (nearest codebook atom) forward.
  * RAW step:         carry the raw unbound vector forward, no cleanup.

MEASURED (see `_selftest`, a 12-hop directed linked list in superposition):
  * RE-ANCHORED reaches every hop (12/12) in order, then the throughput gate abstains exactly when the chain runs
    out -- the signal is genuinely gone, not lost to drift.
  * RAW collapses almost immediately (~1 hop): the carried noise compounds each hop, throughput falls through the
    floor, and the gate stops the dark ray. The per-hop re-anchor cost is one codebook argmax (O(vocab)) -- cheap,
    and plainly justified, since without it the traversal does not survive past the first hop.
"""

import numpy as np
from holographic.agents_and_reasoning.holographic_ai import random_vector, bind, unbind
from holographic.misc.holographic_traverse import gated_traverse


def directed_linked_list(n_hops, dim=1024, seed=0, n_distractors=10):
    """Build a directed linked list of `n_hops` edges stored in one superposition: M = sum_i bind(node_i,
    permute(node_{i+1})). Returns dict {M, chain, cb, perm} where `cb` is the codebook (chain nodes + distractors)
    used for cleanup and `perm` is the direction permutation. The same construction the traverse self-test uses."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(dim)
    def unit():
        v = random_vector(dim, rng); return v / np.linalg.norm(v)
    chain = [unit() for _ in range(n_hops + 1)]
    cb = np.array(chain + [unit() for _ in range(n_distractors)])
    M = np.zeros(dim)
    for i in range(n_hops):
        M = M + bind(chain[i], chain[i + 1][perm])           # directed link: the successor is permuted
    return {"M": M, "chain": chain, "cb": cb, "perm": perm}


def make_steps(ll):
    """Return (reanchored_step, raw_step) for `gated_traverse` over the linked list `ll`. The two are identical
    except for the one line that matters: the re-anchored step carries the CLEANED node forward, the raw step
    carries the raw unbound vector forward (so its noise compounds)."""
    M, cb, perm = ll["M"], ll["cb"], ll["perm"]
    inv = np.argsort(perm)
    def reanchored_step(cur):
        raw = unbind(M, cur)[inv]                            # unbind the link, undo the direction permutation
        cs = cb @ (raw / (np.linalg.norm(raw) + 1e-9)); j = int(np.argmax(cs))
        return (cb[j], float(cs[j]), j)                      # carry the CLEANED node forward (re-anchor)
    def raw_step(cur):
        raw = unbind(M, cur)[inv]
        rn = raw / (np.linalg.norm(raw) + 1e-9)
        cs = cb @ rn; j = int(np.argmax(cs))
        return (rn, float(cs[j]), j)                         # carry the RAW vector forward (no cleanup)
    return reanchored_step, raw_step


def _selftest():
    """CI-fast: on a 12-hop directed linked list, the engine's gated_traverse with a RE-ANCHORED step recovers every
    hop in order, while the SAME traversal with a RAW (no-cleanup) step collapses almost immediately -- re-anchoring
    is load-bearing, and the throughput gate stops the dark ray either way."""
    L = 12
    ll = directed_linked_list(L, dim=1024, seed=0)
    reanchored_step, raw_step = make_steps(ll)
    start = ll["chain"][0]

    g_re = gated_traverse(reanchored_step, start, floor=0.20, max_steps=L + 5, min_steps=1)
    assert g_re.payloads == list(range(1, L + 1)), g_re.payloads   # every hop, in order
    assert g_re.stopped == "floor"                                 # abstains when the chain is exhausted

    g_raw = gated_traverse(raw_step, start, floor=0.20, max_steps=L + 5, min_steps=1)
    assert len(g_raw.payloads) <= 3, g_raw.payloads                # raw collapses early (noise compounds)
    assert len(g_raw.payloads) < len(g_re.payloads) - 5            # decisively worse than re-anchored


if __name__ == "__main__":
    _selftest()
    print("holographic_reanchor selftest passed")
