import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = json.load(open("_fwd_cache.json")); Cs = c["Cs"]; Ks = c["Ks"]; res = c["res"]; D = c["D"]
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.3))
cols = {1: "#c0392b", 2: "#e67e22", 4: "#2c7fb8", 8: "#239b56"}
a = ax[0]
for K in Ks:
    a.plot(Cs, res[str(K)]["fid"], "o-", color=cols[K], ms=5, label=f"{K} shard{'s' if K>1 else ''}")
a.axhline(0.90, color="#555", ls="--", lw=1, label="fidelity = 0.90")
a.annotate("single vector\n(the blocked\nPath-D flagship)", (16, 0.90), textcoords="offset points",
           xytext=(20, -52), fontsize=8, color="#c0392b",
           arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
a.set_xlabel("layer width  C  (number of classes superposed)"); a.set_ylabel("logit fidelity vs exact")
a.set_ylim(0.3, 1.02); a.grid(alpha=.3); a.legend(fontsize=8.5, title="weight-memory")
a.set_title("(a) Federating the weight-memory moves the forward-pass wall")
a = ax[1]
a.plot(Cs, res["1"]["ax"], "k--", lw=1.4, label="exact matmul")
a.plot(Cs, res["1"]["asu"], "o-", color=cols[1], ms=5, label="superposed, 1 vector")
a.plot(Cs, res["8"]["asu"], "s-", color=cols[8], ms=5, label="superposed, 8 shards")
a.set_xlabel("layer width  C  (number of classes)"); a.set_ylabel("classification accuracy")
a.set_ylim(0, 1.03); a.grid(alpha=.3); a.legend(fontsize=8.5)
a.set_title("(b) ...so the classifier stays correct to a far wider layer")
fig.suptitle("As below, so above on the compute floor: the superposed forward pass was capped at C~0.02xD "
             "per VECTOR,\nnot per problem. Federate the weight-memory and faithful layer width scales with "
             "shard count -- the storage fix, applied to compute.", fontsize=9.7)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(os.path.dirname(__file__), "distributed_forward_pass.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print("plot ->", out)
