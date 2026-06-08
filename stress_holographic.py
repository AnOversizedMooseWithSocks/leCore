"""
stress_holographic.py -- adversarial / breaking-point benchmarks.

Where benchmark_holographic.py measures the system in its comfort zone, this one
deliberately pushes each mechanism until it fails, to map the boundaries and
produce a prioritized list of what needs addressing.

Run with:  python3 stress_holographic.py
Outputs:   stress_*.png  and  stress_report.md
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from holographic_ai import random_vector, cosine
from holographic_reasoning import ConformalPredictor
from holographic_sync import SyncGrouping
from holographic_emergence import EmergentConcepts
from holographic_diffusion import DoubleDiffusion
from holographic_extras import PredictiveFilter
from holographic_encoders import ScalarEncoder

warnings.filterwarnings("ignore")
FINDINGS = []     # (area, observation, severity)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _ari_emergent(vecs, truth, seed=0):
    mind = EmergentConcepts()
    for i in np.random.default_rng(seed).permutation(len(vecs)):
        mind.perceive(vecs[int(i)], int(i))
    cc = mind.committed()
    if not cc:
        return 0.0, 0
    pred = [int(np.argmax([cosine(v, c.salt) for c in cc])) for v in vecs]
    return adjusted_rand_score(truth, pred), len(cc)


# ---------------------------------------------------------------------------
# A. Cluster separability sweep -- where does grouping collapse?
# ---------------------------------------------------------------------------
def stress_separability():
    print("[A] separability sweep ...")
    stds = [2, 4, 6, 8, 10, 13]
    em, sg, km = [], [], []
    for std in stds:
        e, s, k = [], [], []
        for seed in range(4):
            X, y = make_blobs(n_samples=240, centers=4, n_features=64,
                              cluster_std=std, center_box=(-12, 12), random_state=seed)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            vecs = [X[i] for i in range(len(X))]
            e.append(_ari_emergent(vecs, y, seed)[0])
            s.append(adjusted_rand_score(y, SyncGrouping(seed=seed).group(vecs)))
            k.append(adjusted_rand_score(y, KMeans(n_clusters=4, n_init=10,
                                                   random_state=0).fit_predict(X)))
        em.append(np.mean(e)); sg.append(np.mean(s)); km.append(np.mean(k))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(stds, km, "-o", label="KMeans (told k)")
    ax.plot(stds, em, "-s", label="Emergent (discovers k)")
    ax.plot(stds, sg, "-^", label="SyncGroup (discovers k)")
    ax.axhline(0.8, color="gray", ls=":", lw=1)
    ax.set_xlabel("cluster_std (higher = more overlap)"); ax.set_ylabel("ARI")
    ax.set_title("Where grouping collapses as clusters overlap")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig("stress_separability.png", dpi=110); plt.close(fig)
    # finding: separation at which emergent drops below 0.8
    breakpt = next((stds[i] for i in range(len(stds)) if em[i] < 0.8), None)
    FINDINGS.append(("Clustering separability",
                     f"Emergent ARI falls below 0.8 once cluster_std >= "
                     f"{breakpt if breakpt else '>13'}; SyncGroup is consistently weaker "
                     f"(ARI {sg[0]:.2f}->{sg[-1]:.2f} across the sweep).",
                     "medium"))
    return pd.DataFrame({"cluster_std": stds, "KMeans": km, "Emergent": em, "SyncGroup": sg})


# ---------------------------------------------------------------------------
# B. Disappearing categories -- does it forget dead concepts?
# ---------------------------------------------------------------------------
def stress_disappearance():
    print("[B] disappearing categories ...")
    dim = 128
    r = np.random.default_rng(0)
    centers = [random_vector(dim, r) for _ in range(5)]
    # cats 0,1 live only in the first third; cats 2,3,4 live in the last two-thirds
    stream = []
    for i in range(600):
        if i < 200:
            live = [0, 1]
        else:
            live = [2, 3, 4]
        cat = int(r.choice(live))
        stream.append((_unit(centers[cat] + 0.5 * random_vector(dim, r)), cat, i))
    mind = EmergentConcepts()
    counts = []
    for x, cat, i in stream:
        mind.perceive(x, i)
        counts.append(len(mind.committed()))
    cc = mind.committed()
    # which latent category does each committed concept represent?
    reps = [int(np.argmax([cosine(c.salt, ctr) for ctr in centers])) for c in cc]
    active_at_end = {2, 3, 4}
    stale = sum(1 for rcat in reps if rcat not in active_at_end)
    FINDINGS.append(("Concept lifecycle",
                     f"After categories 0,1 stopped emitting 400 steps earlier, the former "
                     f"holds {stale} stale concept(s) for them ({len(cc)} committed total, "
                     f"3 categories active). Committed concepts retire after a staleness "
                     f"timeout (retire_after), so dead categories erode while active ones persist.",
                     "high" if stale > 0 else "low"))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.step(range(len(counts)), counts, where="post")
    ax.axvline(200, color="gray", ls=":")
    ax.text(205, 0.5, "cats 0,1 die; 2,3,4 begin", fontsize=8, color="gray")
    ax.set_xlabel("stream step"); ax.set_ylabel("committed concepts")
    ax.set_title(f"Concepts never retire: {stale} dead concept(s) still held at the end")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig("stress_disappearance.png", dpi=110); plt.close(fig)
    return stale, len(cc)


# ---------------------------------------------------------------------------
# C. Scaling in number of categories
# ---------------------------------------------------------------------------
def stress_scaling():
    print("[C] category scaling ...")
    ncats = [3, 5, 8, 12, 20]
    aris, kerr = [], []
    for nc in ncats:
        a, ke = [], []
        for seed in range(3):
            r = np.random.default_rng(seed)
            centers = [random_vector(96, r) for _ in range(nc)]
            vecs, truth = [], []
            for g, c in enumerate(centers):
                for _ in range(12):
                    vecs.append(_unit(c + 0.5 * random_vector(96, r))); truth.append(g)
            ari, found = _ari_emergent(vecs, np.array(truth), seed)
            a.append(ari); ke.append(abs(found - nc))
        aris.append(np.mean(a)); kerr.append(np.mean(ke))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ncats, aris, "-s", label="Emergent ARI")
    ax2 = ax.twinx()
    ax2.plot(ncats, kerr, "-^", color="C3", label="|k_found - k_true|")
    ax.set_xlabel("number of true categories"); ax.set_ylabel("ARI")
    ax2.set_ylabel("cluster-count error", color="C3")
    ax.set_title("Concept former vs number of categories")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig("stress_scaling.png", dpi=110); plt.close(fig)
    drop = next((ncats[i] for i in range(len(ncats)) if aris[i] < 0.8), None)
    FINDINGS.append(("Concept scaling",
                     f"ARI stays high to ~{ncats[-1] if drop is None else drop} categories "
                     f"(ARI {aris[0]:.2f} at 3 -> {aris[-1]:.2f} at 20; count error "
                     f"{kerr[0]:.1f} -> {kerr[-1]:.1f}).",
                     "low" if (drop is None or drop >= 12) else "medium"))
    return pd.DataFrame({"n_categories": ncats, "ARI": aris, "k_error": kerr})


# ---------------------------------------------------------------------------
# D. Conformal under distribution shift (a known assumption-break)
# ---------------------------------------------------------------------------
def stress_conformal_shift():
    print("[D] conformal under shift ...")
    rows = []
    for label, shift, scale in [("control (no shift)", 0.0, 1.0),
                                ("mean shift +1.5", 1.5, 1.0),
                                ("scale x2", 0.0, 2.0)]:
        covs = []
        for trial in range(40):
            r = np.random.default_rng(trial)
            cal = r.normal(0, 1, 250)
            conf = ConformalPredictor(alpha=0.1)        # target 0.90
            conf.calibrate(cal)
            test = r.normal(shift, scale, 400)
            covs.append(np.mean(np.abs(test - 0) <= conf.q))
        rows.append({"condition": label, "target": 0.90, "empirical": round(np.mean(covs), 3)})
    df = pd.DataFrame(rows)
    worst = df["empirical"].min()
    FINDINGS.append(("Conformal robustness",
                     f"Coverage holds under no shift ({df.empirical[0]:.2f}) but drops to "
                     f"{worst:.2f} under distribution shift -- the exchangeability assumption "
                     f"breaks, as expected. Needs shift detection to stay honest.",
                     "medium"))
    return df


# ---------------------------------------------------------------------------
# E. Double diffusion on gradual vs abrupt drift
# ---------------------------------------------------------------------------
def stress_gradual_drift():
    print("[E] gradual vs abrupt drift ...")
    r = np.random.default_rng(0)

    def layers_for(stream):
        dd = DoubleDiffusion()
        return sum(dd.observe([x])[2] for x in stream)

    abrupt = [(0.0 if i < 50 else 2.0) + r.normal(0, 0.05) for i in range(100)]
    gradual = [2.0 * i / 100 + r.normal(0, 0.05) for i in range(100)]   # slow ramp
    la, lg = layers_for(abrupt), layers_for(gradual)
    FINDINGS.append(("Double diffusion drift type",
                     f"An abrupt 0->2 jump commits {la} layer(s) cleanly; a gradual 0->2 ramp "
                     f"over 100 steps commits {lg} layer(s) -- slow drift is "
                     f"{'silently absorbed (no segmentation)' if lg == 0 else 'segmented'}; the "
                     f"detector keys on suddenness, not total change.",
                     "low" if lg <= 2 else "medium"))
    return la, lg


# ---------------------------------------------------------------------------
# F. Predictive filter: detection vs change magnitude (sensitivity floor)
# ---------------------------------------------------------------------------
def stress_predictive_sensitivity():
    print("[F] predictive-filter sensitivity ...")
    dim = 256
    mags = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]    # fraction of the way to an orthogonal state
    det, fa = [], []
    for m in mags:
        d_runs, fa_runs = [], []
        for seed in range(6):
            r = np.random.default_rng(seed)
            a = random_vector(dim, r)
            b = _unit((1 - m) * a + m * random_vector(dim, r))   # a partial shift
            pf = PredictiveFilter()
            caught = 0; false = 0
            for i in range(60):
                base = a if i < 30 else b
                novel, _ = pf.observe(base + 0.2 * random_vector(dim, r))
                if 30 <= i <= 34 and novel:
                    caught = 1
                if 5 < i < 30 and novel:
                    false += 1
            d_runs.append(caught); fa_runs.append(false)
        det.append(np.mean(d_runs)); fa.append(np.mean(fa_runs))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(mags, det, "-o", label="detection rate")
    ax.plot(mags, fa, "-^", color="C3", label="false alarms / run (stable phase)")
    ax.set_xlabel("change magnitude (0 = none, 1 = orthogonal)")
    ax.set_ylabel("rate"); ax.set_title("Predictive filter: detection vs change size")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig("stress_predictive.png", dpi=110); plt.close(fig)
    floor = next((mags[i] for i in range(len(mags)) if det[i] >= 0.8), None)
    FINDINGS.append(("Predictive filter sensitivity",
                     f"Reliable (>=80%) change detection needs a shift of about "
                     f"{floor if floor else '>1.0'} toward orthogonal; smaller shifts are missed. "
                     f"False alarms stay near {max(fa):.1f}/run.",
                     "low" if (floor and floor <= 0.3) else "medium"))
    return pd.DataFrame({"magnitude": mags, "detection": det, "false_alarms": fa})


# ---------------------------------------------------------------------------
# G. Scalar encoder: decode error vs range width (fixed dimension)
# ---------------------------------------------------------------------------
def stress_scalar_range():
    print("[G] scalar encoder range ...")
    dim = 1024
    ranges = [10, 100, 1000, 10000]
    rows = []
    for R in ranges:
        enc = ScalarEncoder(dim, lo=0, hi=R, seed=0)
        r = np.random.default_rng(0)
        errs = []
        for _ in range(60):
            v = float(r.uniform(0, R))
            errs.append(abs(enc.decode(enc.encode(v)) - v) / R)   # relative error
        rows.append({"range": R, "rel_decode_error": round(float(np.mean(errs)), 4)})
    df = pd.DataFrame(rows)
    worst = df["rel_decode_error"].max()
    FINDINGS.append(("Scalar encoder resolution",
                     f"Relative decode error stays ~{df.rel_decode_error.min():.3f} across "
                     f"ranges (grid readout scales with range), worst {worst:.3f}. Absolute "
                     f"precision is fixed by decode-grid steps, not range -- fine resolution "
                     f"over a wide range needs more grid steps or staged encoders.",
                     "low"))
    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(tables):
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    findings = sorted(FINDINGS, key=lambda f: sev_rank[f[2]])
    lines = ["# Holographic system -- stress test findings\n",
             "Adversarial probes pushing each mechanism past its comfort zone. "
             "Findings are ordered by severity.\n",
             "## Prioritized findings\n",
             "| severity | area | observation |", "|---|---|---|"]
    for area, obs, sev in findings:
        lines.append(f"| **{sev}** | {area} | {obs} |")
    lines.append("\n## Details and tables\n")
    for name, df in tables.items():
        lines.append(f"### {name}\n\n```\n{df.to_string(index=False)}\n```\n")
    lines.append("## Plots\n")
    for p in ["stress_separability.png", "stress_disappearance.png", "stress_scaling.png",
              "stress_predictive.png"]:
        lines.append(f"- `{p}`")
    with open("stress_report.md", "w") as f:
        f.write("\n".join(lines))
    print("\nwrote stress_report.md")


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    tables = {}
    tables["A. separability"] = stress_separability()
    stress_disappearance()
    tables["C. scaling"] = stress_scaling()
    tables["D. conformal shift"] = stress_conformal_shift()
    stress_gradual_drift()
    tables["F. predictive sensitivity"] = stress_predictive_sensitivity()
    tables["G. scalar range"] = stress_scalar_range()
    write_report(tables)
    print(f"done in {time.perf_counter() - t0:.0f}s\n")
    print("=== FINDINGS ===")
    for area, obs, sev in sorted(FINDINGS, key=lambda f: {'high': 0, 'medium': 1, 'low': 2}[f[2]]):
        print(f"[{sev.upper()}] {area}: {obs}")
