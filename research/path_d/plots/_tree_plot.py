import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = json.load(open("_tree_cache.json")); R = c["results"]; exh = c["exhaustive"]; K = c["K"]
dep=[r["depth"] for r in R]; t1=[r["top1_b1"] for r in R]; rb5=[r["rec_b5"] for r in R]
cb1=[r["comp_b1"] for r in R]; cb5=[r["comp_b5"] for r in R]

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a=ax[0]
a.axhline(exh, color="#2c7fb8", ls="--", lw=1.5, label=f"exhaustive ceiling ({exh:.3f}, scans all {K})")
a.plot(dep, t1, "o-", color="#239b56", ms=7, label="pivot-tree greedy top-1")
a.scatter([4], [0.23], color="#c0392b", marker="x", s=90, zorder=4)
a.annotate("naive summary\nindex (0.23)", (4, 0.23), textcoords="offset points", xytext=(-86,2), fontsize=8, color="#c0392b")
a.set_xticks(dep); a.set_xlabel("tree depth"); a.set_ylabel("recall (top-1)")
a.set_ylim(0,1.03); a.grid(alpha=.3); a.legend(fontsize=8, loc="lower center")
a.set_title("(a) Routing = exhaustive at every depth (no r^d compounding)")

a=ax[1]
a.plot(dep, cb1, "o-", color="#239b56", ms=7, label="greedy (beam 1)")
a.axhline(K, color="#2c7fb8", ls="--", lw=1.2, label=f"exhaustive ({K})")
a.set_yscale("log"); a.set_xticks(dep); a.set_xlabel("tree depth"); a.set_ylabel("comparisons / query")
a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
a.set_title("(b) ...at 86x fewer comparisons (sublinear)")

a=ax[2]
a.plot(dep, rb5, "o-", color="#8e44ad", ms=7, label="true shard in beam-5 set")
a.axhline(0.23, color="#c0392b", ls=":", lw=1.2, label="naive summary index (0.23)")
a.set_xticks(dep); a.set_xlabel("tree depth"); a.set_ylabel("routing-recall @ beam 5")
a.set_ylim(0,1.03); a.grid(alpha=.3); a.legend(fontsize=8, loc="center right")
a.set_title("(c) Beam lands truth in a tiny set -> array's exact recall finishes")

fig.suptitle("The recursive holographic B-tree: pivots not summaries, cleanup-within-cleanup. "
             "Routing survives depth -- greedy matches brute force at 86x fewer comparisons.", fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.95])
out=os.path.join(os.path.dirname(__file__),"pivot_tree.png"); fig.savefig(out,dpi=130,bbox_inches="tight")
print("plot ->", out)
