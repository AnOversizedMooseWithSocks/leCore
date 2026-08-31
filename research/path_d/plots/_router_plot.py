import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = json.load(open("_router_cache.json")); p1 = c["part1"]; p2 = c["part2"]
K=[r[0] for r in p1]; da=[r[1] for r in p1]; ra=[r[2] for r in p1]
ba=[(r[0], r[3]) for r in p1 if r[3] is not None]
dt=[r[4]*1e6 for r in p1]; rt=[r[5]*1e6 for r in p1]
bt=[(r[0], r[6]*1e6) for r in p1 if r[6] is not None]

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a=ax[0]
a.plot(K, da, "o-", color="#2c7fb8", label="directory (exact key, O(1))", ms=5)
a.plot(K, ra, "o-", color="#239b56", label="sketch-routed (top-8)", ms=5)
a.plot([x for x,_ in ba], [y for _,y in ba], "o-", color="#c0392b", label="full broadcast", ms=5)
a.set_xscale("log",base=2); a.set_xlabel("# shards (50 items each)"); a.set_ylabel("recall accuracy")
a.set_ylim(-.03,1.03); a.grid(alpha=.3,which="both"); a.legend(fontsize=8)
a.set_title("(a) The router holds where broadcast collapses")

a=ax[1]
a.plot(K, dt, "o-", color="#2c7fb8", label="directory  O(1)", ms=5)
a.plot(K, rt, "o-", color="#239b56", label="sketch-routed  ~flat", ms=5)
a.plot([x for x,_ in bt], [y for _,y in bt], "o-", color="#c0392b", label="broadcast  O(shards)", ms=5)
a.set_xscale("log",base=2); a.set_yscale("log"); a.set_xlabel("# shards"); a.set_ylabel("time per query (us)")
a.grid(alpha=.3,which="both"); a.legend(fontsize=8)
a.set_title("(b) ...at ~88x lower cost at 1024 shards")

a=ax[2]
for label, rec, comp in p2:
    col = "#239b56" if label == "1-level" else "#c0392b"
    mk = "s" if label == "1-level" else "o"
    a.scatter([comp], [rec], color=col, marker=mk, s=70, zorder=3)
    a.annotate(label, (comp, rec), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
a.axhline(0.9, color="0.6", ls=":", lw=1)
a.set_xlabel("routing comparisons / query"); a.set_ylabel("routing-recall (true shard found)")
a.set_ylim(-.03,1.03); a.grid(alpha=.3); a.set_title("(c) Naive hierarchy crashes: capacity caps fan-out")

fig.suptitle("Pushing the limit: a sketch router breaks the broadcast wall (flat accuracy, ~flat cost) -- "
             "but naive sketch-of-sketches blows the per-vector budget (the conserved law, one level up)",
             fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.95])
out=os.path.join(os.path.dirname(__file__),"array_router.png"); fig.savefig(out,dpi=130,bbox_inches="tight")
print("plot ->", out)
