import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np, os
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
Ms = [8, 64, 256]; bundle = [-0.13, 0.04, 0.06]; rns = [1.0, 1.0, 1.0]
xp = np.arange(len(Ms)); w = 0.38
a = ax[0]
a.bar(xp - w/2, bundle, w, color="#c0392b", label="lossy bundle (the A2 boundary)")
a.bar(xp + w/2, rns, w, color="#239b56", label="RNS-phasor (decompose→adapt→recompose)")
a.axhline(0, color="#888", lw=.8); a.set_xticks(xp); a.set_xticklabels([f"M={m}" for m in Ms])
a.set_ylabel("matmul fidelity (1.0 = exact)"); a.set_ylim(-0.25, 1.1); a.legend(fontsize=8.5)
a.set_title("(a) Same integer matmul, two encodings:\nbundle stays noise; the adapted formula is EXACT (error 0)")
a = ax[1]
ks = [4, 8, 16, 32]; digits = [8, 16, 34, 71]
a.bar([str(k) for k in ks], digits, color="#2c7fb8")
a.set_xlabel("moduli channels (federated)"); a.set_ylabel("exact integer range (decimal digits)")
a.set_title("(b) Range federates over channels — the A6 law,\nnow carrying exact arithmetic instead of stored integers")
for i, d in enumerate(digits): a.text(i, d + 1, f"1e{d}", ha="center", fontsize=8.5)
fig.suptitle("A2 resolved by adapting the formula, not abandoning it: matmul → RNS multiply-accumulate over exact FHRR phasor binding.\n"
             "The crosstalk wall was the lossy weight-superposition encoding; carrying values as residues/phases composes them exactly.", fontsize=9.8)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(os.path.dirname(__file__), "a2_resolved_rns.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print("plot ->", out)
