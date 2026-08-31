import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = json.load(open("_factor_cache.json")); rows = c["rows"]
F = [r["F"] for r in rows]; dense = [r["dense"] for r in rows]; sbc = [r["sbc"] for r in rows]
mat = json.load(open("_factor_dim_cache.json"))
Fd = [3, 4, 5, 6]

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2))
a = ax[0]
a.plot(F, dense, "o-", color="#c0392b", ms=7, label="dense / monolithic (one resonator)")
a.plot(F, sbc, "s-", color="#239b56", ms=7, label="SBC / block-distributed (+ thin layer)")
a.axvspan(3.5, 5.5, color="#f1c40f", alpha=0.12)
a.annotate("distribution buys\n~1 extra factor", (4.5, 0.5), ha="center", fontsize=8.5, color="#7d6608")
a.set_xlabel("number of factors  F   (search space = 8^F)"); a.set_ylabel("factorization solve rate")
a.set_ylim(-0.03, 1.05); a.grid(alpha=.3); a.legend(fontsize=8.5, loc="upper right")
a.set_title("(a) The factorization wall, broken the router's way\n(distribute the search), matched D=1024")

a = ax[1]
cols = {"1024": "#85929e", "2048": "#5499c7", "4096": "#1f618d"}
for Dlab in ["1024", "2048", "4096"]:
    a.plot(Fd, mat[Dlab], "o-", ms=6, color=cols[Dlab], label=f"D={Dlab}")
a.set_xlabel("number of factors  F"); a.set_ylabel("SBC factorization solve rate")
a.set_xticks(Fd); a.set_ylim(-0.03, 1.05); a.grid(alpha=.3); a.legend(fontsize=9, title="total dimensions")
a.set_title("(b) ...and adding dimension shifts the wall right\n(factorization capacity scales with D, like storage)")

fig.suptitle("As below, so above on the algebra floor: the factorization wall yields to the SAME move as "
             "the lookup wall —\ndistribute the search with a thin layer, and add dimension. Joint search → "
             "steeper exchange rate, but the mechanism is identical.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(os.path.dirname(__file__), "factor_wall.png"); fig.savefig(out, dpi=130, bbox_inches="tight")
print("plot ->", out)
