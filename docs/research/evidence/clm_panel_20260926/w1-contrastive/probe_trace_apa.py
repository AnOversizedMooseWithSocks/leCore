"""probe_trace_apa.py -- w1-contrastive scratch probe (NOT repo code). SYNTHETIC keys/labels.
w2-hd measured that a signed negative write erases a NEIGHBOUR key (cos 0.8) whose own truth is the wrong label.
Widrow lens: that is LMS interference; the affine-projection fix writes the correction along the key component
ORTHOGONAL to the nearest stored key with a different outcome: k_perp = (k - c k')/(1 - c^2), so <k_perp,k>=1 and
<k_perp,k'>=0. Arms: repo (along target), signed LMS on codebook, signed LMS on k_perp (APA)."""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/leCore")
from holographic.agents_and_reasoning.holographic_lever7 import DisplacementTrace, key_atom
from holographic.agents_and_reasoning.holographic_ai import bind, unbind

def unit(v): return v / np.linalg.norm(v)

def run(c, n_bg=40, dim=2048, n_labels=40, trials=40, floor=0.2):
    L = np.stack([unit(key_atom("label:%d" % i, dim)) for i in range(n_labels)])
    out = {}
    for arm in ("repo_along_target", "signed_codebook", "signed_apa"):
        kb, kn = 0, 0
        for t in range(trials):
            tr = DisplacementTrace(dim=dim, seed=0, advisory_load=1.0)
            for i in range(n_bg):
                tr.write(key_atom("bg:%d" % i, dim), L[i % n_labels])
            kp = unit(key_atom("nb:%d" % t, dim))                     # the neighbour: its truth IS a
            r = key_atom("ort:%d" % t, dim); r = unit(r - (r @ kp) * kp)
            k = unit(c * kp + np.sqrt(1 - c * c) * r)                  # the query key, cos c to the neighbour
            a, b = t % n_labels, (t + 7) % n_labels
            tr.write(kp, L[a])
            # the reflex at k now leaks a (served wrong); the truth for k is b -> correction
            pred = unbind(tr._trace, k)
            s = (L @ pred) / (tr._rho * np.sum(L * L, 1))
            tgt = np.zeros(n_labels); tgt[b] = 1.0
            if arm == "repo_along_target":
                tr.write(k, L[b])
            else:
                e = tgt - s
                keep = (np.abs(s) >= floor) | (tgt > 0)
                err = (e[keep, None] * L[keep]).sum(0)
                key = k if arm == "signed_codebook" else (k - c * kp) / (1 - c * c)
                tr._trace = tr._trace + bind(key, err); tr._n += 1
            rk = L @ unit(unbind(tr._trace, k)); rn = L @ unit(unbind(tr._trace, kp))
            kb += int(np.argmax(rk) == b); kn += int(np.argmax(rn) == a)
        out[arm] = {"k_reads_truth": "%d/%d" % (kb, trials), "neighbour_keeps_its_answer": "%d/%d" % (kn, trials)}
    return out

print(json.dumps({"cos_%.2f" % c: run(c) for c in (0.8, 0.9)}, indent=1))
