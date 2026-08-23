from _scale_lib import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = load_cache(); scale = c["scale"]; ceil = c["ceiling"]
N=[r[0] for r in scale]; sh=[r[1] for r in scale]; da=[r[2] for r in scale]; ba=[r[3] for r in scale]
dt=[r[4]*1e6 for r in scale]; bt=[r[5]*1e6 for r in scale]
K=[r[0] for r in ceil]; cda=[r[1] for r in ceil]; cba=[r[2] for r in ceil]

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a=ax[0]
a.plot(N, da, "o-", color="#239b56", label="directory recall", markersize=4)
a.plot(N, ba, "o-", color="#c0392b", label="broadcast recall", markersize=4)
a.set_xscale("log"); a.set_xlabel("items streamed in"); a.set_ylabel("recall accuracy")
a.set_ylim(-0.03,1.03); a.grid(alpha=0.3, which="both")
a2=a.twinx(); a2.plot(N, sh, "--", color="#2c7fb8", lw=1.2); a2.set_ylabel("# shards", color="#2c7fb8")
a2.tick_params(axis="y", labelcolor="#2c7fb8")
a.set_title("(a) Streaming load: storage holds, broadcast erodes"); a.legend(loc="lower left", fontsize=8)

a=ax[1]
a.plot(K, cda, "o-", color="#239b56", label="directory (routed)", markersize=4)
a.plot(K, cba, "o-", color="#c0392b", label="broadcast (routerless)", markersize=4)
a.axhline(0.9, color="0.6", ls=":", lw=1); a.axhline(0.5, color="0.6", ls=":", lw=1)
a.set_xscale("log", base=2); a.set_xlabel("# shards in the array"); a.set_ylabel("recall accuracy")
a.set_ylim(-0.03,1.03); a.grid(alpha=0.3, which="both")
a.annotate("46,080 items\n@ 0.98", (1024, 0.98), textcoords="offset points", xytext=(-78,-6), fontsize=7, color="#239b56")
a.set_title("(b) Pushed to 1024 shards: directory flat, broadcast soft-erodes"); a.legend(fontsize=8)

a=ax[2]
a.plot(sh, dt, "o-", color="#239b56", label="directory  O(1)", markersize=4)
a.plot(sh, bt, "o-", color="#c0392b", label="broadcast  O(shards)", markersize=4)
a.set_xlabel("# shards"); a.set_ylabel("time per query (us)"); a.set_yscale("log"); a.grid(alpha=0.3, which="both")
a.set_title("(c) The price of routerless: O(shards)/query"); a.legend(fontsize=8)

fig.suptitle("How far the aligned array goes: directory-routed storage scales ~unbounded (920x one vector at 0.97); "
             "routerless broadcast soft-erodes and costs O(shards)", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95])
out=os.path.join(os.path.dirname(__file__), "array_scale.png"); fig.savefig(out, dpi=130, bbox_inches="tight")
print("plot ->", out)
print(f"build {c['N_MAX']} items in {c['build_t']:.1f}s; broadcast >=90% to {c['k90']} shards, >=50% to {c['k50']}")
