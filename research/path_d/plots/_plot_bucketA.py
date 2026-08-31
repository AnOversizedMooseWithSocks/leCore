import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
b234 = json.load(open("_batch234_cache.json")); bB = json.load(open("_batchB_cache.json"))
fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2))
WIN="#239b56"; BLK="#c0392b"; MID="#2c7fb8"; ORG="#e67e22"

# A1 (measured)
d=[1,2,3]; ex=[1.0,1.0,1.0]; no=[0.976,0.418,0.149]; cl=[0.999,0.694,0.201]
a=ax[0,0]; a.plot(d,ex,"k--",label="exact MLP"); a.plot(d,no,"o-",color=BLK,label="superposed, no cleanup")
a.plot(d,cl,"s-",color=MID,label="+ inter-layer cleanup")
a.set_xticks(d); a.set_xlabel("hidden layers (depth)"); a.set_ylabel("test accuracy"); a.set_ylim(0,1.05)
a.grid(alpha=.3); a.legend(fontsize=8); a.set_title("A1 deep forward pass — width federates, DEPTH is the wall")

# A2 dense matmul (boundary)
Ms,dense = bB["A2"]
a=ax[0,1]; a.plot(Ms,dense["1"],"o-",color=BLK,label="1 vector"); a.plot(Ms,dense["4"],"s-",color=ORG,label="4 shards")
a.axhline(0.9,color="#555",ls="--",lw=1,label="0.90"); a.set_ylim(-0.1,1.02)
a.set_xlabel("matrix rows  M"); a.set_ylabel("product fidelity (corr)"); a.grid(alpha=.3); a.legend(fontsize=8)
a.set_title("A2 dense continuous matmul — boundary (no cleanup → precision wall)")

# A3 selection (win)
Hs,sel,rnk = b234["A3"]
a=ax[0,2]; a.plot(Hs,sel["1"],"o-",color=BLK,label="1 vector"); a.plot(Hs,sel["8"],"s-",color=WIN,label="8 shards")
a.axhline(0.95,color="#555",ls="--",lw=1,label="0.95"); a.set_ylim(0,1.05)
a.set_xlabel("# hypotheses  H"); a.set_ylabel("pick-the-best accuracy"); a.grid(alpha=.3); a.legend(fontsize=8)
a.set_title("A3 hypothesis selection — wall 64→256 (WIN)")

# A4 sequence (win)
Ts,acc = b234["A4"]
a=ax[1,0]; a.plot(Ts,acc["1"],"o-",color=BLK,label="1 vector"); a.plot(Ts,acc["8"],"s-",color=WIN,label="8 shards")
a.axhline(0.9,color="#555",ls="--",lw=1,label="0.90"); a.set_ylim(0,1.05)
a.set_xlabel("sequence length  T"); a.set_ylabel("fraction recalled"); a.grid(alpha=.3); a.legend(fontsize=8)
a.set_title("A4 sequence memory — wall 64→256 (WIN)")

# A5 archive (conservation)
Ns,mono,fed = bB["A5"]
a=ax[1,1]; a.plot(Ns,mono,"o-",color=BLK,label="monolithic"); a.plot(Ns,fed,"s--",color=WIN,label="federated K=4")
a.set_ylim(0,1.05); a.set_xlabel("# images (fixed total dim)"); a.set_ylabel("recovery corr")
a.grid(alpha=.3); a.legend(fontsize=8); a.set_title("A5 federated archive — quality CONSERVED (curves overlap)")

# A6 residue range (win)
ks,ranges,acc6 = bB["A6"]
digits=[len(str(r)) for r in ranges]
a=ax[1,2]; a.plot(digits,acc6["1"],"o-",color=BLK,label="1 vector"); a.plot(digits,acc6["8"],"s-",color=WIN,label="8 shards")
a.axhline(0.95,color="#555",ls="--",lw=1,label="0.95"); a.set_ylim(0,1.05)
a.set_xlabel("faithful integer range (decimal digits)"); a.set_ylabel("round-trip accuracy")
a.grid(alpha=.3); a.legend(fontsize=8); a.set_title("A6 residue range — 1e34 → 1e391 (WIN)")

fig.suptitle("Bucket A swept: federation moves the single-vector wall for selection (A3), sequences (A4), and integer range (A6); "
             "the archive (A5) conserves\nquality at fixed dimensions; dense continuous matmul (A2) is a confirmed precision boundary; "
             "and depth (A1) is the genuine frontier cleanup only partly tames.", fontsize=10)
fig.tight_layout(rect=[0,0,1,0.95])
out=os.path.join(os.path.dirname(__file__),"bucket_A_sweep.png")
fig.savefig(out,dpi=125,bbox_inches="tight"); print("plot ->",out)
