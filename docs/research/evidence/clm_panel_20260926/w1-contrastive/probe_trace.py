"""probe_trace.py -- w1-contrastive scratch probe (NOT repo code). SYNTHETIC keys/labels.
Does a CORRECTION on the reflex trace erase the wrong label? The repo's DisplacementTrace.write corrects ALONG the
target only ((1-s)*v). A key first written with a WRONG label a, then corrected to b, reads back a blend of a and b.
Arm 'lms_codebook' adds the Widrow term along the OTHER codebook atoms only: -s_l * a_l for labels whose estimated
strength s_l clears a floor -- the LMS error projected onto the codebook span (never the crosstalk)."""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/leCore")
from holographic.agents_and_reasoning.holographic_lever7 import DisplacementTrace, key_atom
from holographic.agents_and_reasoning.holographic_ai import bind, unbind

def run(n_load, dim=2048, n_labels=40, n_probe=40, floor=0.2, seed=0):
    rng = np.random.default_rng(seed)
    labels = [key_atom("label:%d" % i, dim) for i in range(n_labels)]
    L = np.stack([l / np.linalg.norm(l) for l in labels])
    out = {}
    for arm in ("repo_along_target", "lms_codebook"):
        tr = DisplacementTrace(dim=dim, seed=0, advisory_load=1.0)
        r2 = np.random.default_rng(seed + 1)
        keys = [key_atom("bg:%d" % i, dim) for i in range(n_load)]
        for i, k in enumerate(keys):                        # background load: correct pairs
            tr.write(k, labels[i % n_labels])
        wins, margins = 0, []
        for j in range(n_probe):
            k = key_atom("probe:%d" % j, dim)
            a, b = j % n_labels, (j + 7) % n_labels         # a = wrong first write, b = the correction
            tr.write(k, labels[a])
            if arm == "repo_along_target":
                tr.write(k, labels[b])
            else:
                pred = unbind(tr._trace, k)
                s = (L @ pred) / (tr._rho * np.sum(L * L, 1))  # estimated stored strength per atom
                t = np.zeros(n_labels); t[b] = 1.0
                e = t - s
                keep = (np.abs(s) >= floor) | (t > 0)        # only atoms the key measurably holds, + the target
                err = (e[keep, None] * L[keep]).sum(0)
                tr._trace = tr._trace + bind(k, err); tr._n += 1
            rd = unbind(tr._trace, k); rd /= np.linalg.norm(rd)
            c = L @ rd
            wins += int(np.argmax(c) == b)
            margins.append(float(c[b] - c[a]))
        out[arm] = {"correction_wins": "%d/%d" % (wins, n_probe), "median_margin_b_minus_a": round(float(np.median(margins)), 3)}
    return out

res = {n: run(n) for n in (0, 50, 150)}
print(json.dumps(res, indent=1))
