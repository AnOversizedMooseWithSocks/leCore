"""P3: threshold-FREE comparison of the confidence signals (the gate's input), because coverage at exactly P=0.97 turned
out to be a knife-edge statistic for the base (37.7% with one calibration sample, 3.3% with another).

Report split = CLINC test in-scope (4500) + oos_test half B (by sha256 parity), precision under the 4500:1000 prior.
For each arm: precision at fixed SERVED fractions, AURC (area under the risk-coverage curve, lower is better), and
coverage at P in {0.95, 0.96, 0.97, 0.98} with the ORACLE threshold on the report split (diagnostic only: symmetric
for every arm, but tuned on what it reports -- never a shipping number).
Arms: base (index today), InfoNCE prototypes (val-chosen cfg tau .05 eta .03 ep4, seed 0), AdaptHD (eta .03 ep4),
InfoNCE with softmax p_max as the signal (the candidate-relative trap), literal E1.4 (lam .5, from P1).
"""
import hashlib, importlib.util, sys, time
import numpy as np
sys.argv = ["p2b_proto.py", "learned"]
t0 = time.time()
# reuse P2's machinery without running its experiment: exec the module up to its experiment section
src = open("p2b_proto.py").read()
src = src[:src.index("RES = {\"setting\": SETTING}")]
exec(compile(src, "p2b_proto_head", "exec"))

HALF_A = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 == 0 for q, _ in d["oos_test"]])


def signal(R_in, R_oos):
    out = []
    for R in (R_in, R_oos):
        o = np.argsort(-R, axis=1, kind="stable")[:, :2]
        sv = np.take_along_axis(R, o, axis=1)
        s1, s2 = sv[:, 0], np.where(sv[:, 1] > 0, sv[:, 1], 0.0)
        out.append((o[:, 0], np.where(s1 > 0, 2 * s1 - s2, -9)))
    (ti, gi), (to, go) = out
    truth = np.array([i for _, i in d["test"]])
    return row_int[ti] == truth, gi, go


def curves(name, ok_in, g_in, g_oos):
    g_oos = g_oos[~HALF_A]
    w_oos = 1000.0 / len(g_oos) * (4500.0 / len(g_in))
    sig = np.concatenate([g_in, g_oos]); ok = np.concatenate([ok_in, np.zeros(len(g_oos), bool)])
    w = np.concatenate([np.ones(len(g_in)), np.full(len(g_oos), w_oos)])
    o = np.argsort(-sig, kind="stable")
    cw = np.cumsum(w[o]); cr = np.cumsum((ok * w)[o]); prec = cr / cw; frac = cw / cw[-1]
    aurc = float(np.sum((1 - prec) * w[o]) / cw[-1])
    at = []
    for f in (0.2, 0.3, 0.4, 0.5, 0.6):
        k = np.searchsorted(frac, f)
        at.append("%.0f%%:%.3f" % (100 * f, prec[k]))
    cov = []
    for P in (0.95, 0.96, 0.97, 0.98):
        idx = np.where(prec >= P)[0]
        k = idx.max() + 1 if len(idx) else 0
        cov.append("P%.2f %.1f%%" % (P, 100 * (ok[o][:k] & (np.arange(k) >= 0)).sum() / len(g_in)))
    print("%-34s AURC %.4f | precision at served fraction %s | oracle cov %s" % (name, aurc, " ".join(at), " ".join(cov)),
          flush=True)


C0T = np.ascontiguousarray(C0.T)
ok, gi, go = signal(readout(C0T, "test", "blend"), readout(C0T, "oos_test", "blend"))
curves("base (index today)", ok, gi, go)
# literal E1.4 on the base: g - lam * cos(q, mu_c), mu_c = mean unit centroid
mu_c = C0.mean(0); nmu = np.linalg.norm(mu_c)
cq_in = np.asarray(QH["test"] @ mu_c).ravel() / nmu; cq_oos = np.asarray(QH["oos_test"] @ mu_c).ravel() / nmu
curves("literal E1.4 (lam 0.5, val-chosen)", ok, np.where(gi > -9, gi - 0.5 * cq_in, -9), np.where(go > -9, go - 0.5 * cq_oos, -9))
for rule, tau, eta, ep in (("infonce", 0.05, 0.03, 4), ("adapthd", 1.0, 0.03, 4)):
    PTn = train(rule, tau, eta, ep, 0, (ep,))[ep]
    ok, gi, go = signal(readout(PTn, "test", "blend"), readout(PTn, "oos_test", "blend"))
    curves("%s (val-chosen, seed 0)" % rule, ok, gi, go)
    if rule == "infonce":
        pm = []
        for s in ("test", "oos_test"):
            z = np.asarray(QH[s] @ PTn) / tau
            z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            pm.append(p.max(1))
        curves("infonce, signal = softmax p_max", ok, pm[0], pm[1])
        # the CLM trap proper: softmax over only the k SUPPLIED candidates (the typed prompt shows 8; a Choice may show 2)
        for kk in (8, 2):
            pk = []
            for s in ("test", "oos_test"):
                z = np.asarray(QH[s] @ PTn) / tau
                z = -np.sort(-z, axis=1)[:, :kk]
                z -= z[:, :1]; p = np.exp(z); p /= p.sum(1, keepdims=True)
                pk.append(p[:, 0])
            curves("infonce, p_max over top-%d only" % kk, ok, pk[0], pk[1])
print("total %.1fs" % (time.time() - t0))
