# This one-off research script lives in archive/; put the library at the repo
# root on the path so its imports keep working when run from anywhere.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from holographic.misc.holographic_creature import GridWorld, CreatureEncoder, HolographicMind, _train
import io, contextlib

def rollout(seed, encoder, mind=None, steps=60):
    """One episode on a fixed world layout; record the creature's path + stars eaten."""
    w = GridWorld(7, 7, n_poison=2, seed=seed)             # constructor already reset()s
    poison = list(w.poison); path = [(w.cx, w.cy)]; eaten = []
    foods = [(w.fx, w.fy)]; senses = w.senses()
    rng = np.random.default_rng(123)
    for _ in range(steps):
        if mind is None:                                    # random policy (the 'before')
            a = int(rng.integers(4))
        else:
            a = mind.decide(encoder.encode(senses), explore=False)
        senses, r, ate, done = w.step(GridWorld.ACTIONS[a])
        path.append((w.cx, w.cy))
        if ate:
            eaten.append((w.cx, w.cy)); foods.append((w.fx, w.fy))
        if done:                                            # poison death or empty battery
            break
    return dict(poison=poison, path=path, eaten=eaten, foods=foods, w=w.w, h=w.h)

def draw(ax, ep, title):
    W, H = ep["w"], ep["h"]
    ax.set_xlim(-.5, W-.5); ax.set_ylim(H-.5, -.5); ax.set_aspect("equal")
    ax.set_xticks(range(W)); ax.set_yticks(range(H)); ax.grid(True, color="#1e2c47", lw=.6)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    for (x, y) in ep["poison"]:
        ax.add_patch(plt.Rectangle((x-.5, y-.5), 1, 1, color="#ff5c7a", alpha=.35))
        ax.plot(x, y, "x", color="#ff5c7a", ms=12, mew=3)
    px = [p[0] for p in ep["path"]]; py = [p[1] for p in ep["path"]]
    ax.plot(px, py, "-", color="#2dd4bf", lw=2, alpha=.9)
    ax.plot(px[0], py[0], "o", color="#c8d3e6", ms=11, label="start")
    for (x, y) in ep["foods"]:
        ax.plot(x, y, "*", color="#ffd166", ms=17, mec="#7a5b00")
    for (x, y) in ep["eaten"]:
        ax.plot(x, y, "*", color="#63dcbe", ms=20, mec="#063")
    ax.set_title(title, fontsize=11)

enc = CreatureEncoder(256, seed=1)
mind = HolographicMind(256, GridWorld.ACTIONS, k=15, epsilon=0.35, novelty_bonus=0.1, memory_cap=5000, seed=7)
world = GridWorld(7, 7, n_poison=2, seed=3)
with contextlib.redirect_stdout(io.StringIO()):
    _train(world, enc, mind, episodes=200)

before = rollout(11, enc, mind=None)
after  = rollout(11, enc, mind=mind)
print("random  before: stars collected =", len(before["eaten"]))
print("trained after : stars collected =", len(after["eaten"]))
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
draw(ax[0], before, f"before training (random)\nstars collected: {len(before['eaten'])}")
draw(ax[1], after,  f"after training (greedy)\nstars collected: {len(after['eaten'])}")
fig.tight_layout(); fig.savefig("creature_viz.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("rendered creature_viz.png")
