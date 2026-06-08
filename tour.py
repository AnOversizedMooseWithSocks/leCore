"""
tour.py -- a guided tour of the whole holographic system in one run.

Each section exercises a different subsystem with a small, readable example,
so you can confirm end-to-end that numbers, text, mixed records, key->value
memory, reasoning, the learning creature, and the image archive are all wired
up and working together on the same vector substrate.

    python tour.py

(Takes ~15-20 seconds; the creature section trains a little RL agent.)
"""
import numpy as np

from holographic_ai import (Vocabulary, HolographicMemory,
                            bind, unbind, bundle, cosine, random_vector)
from holographic_encoders import ScalarEncoder, TextEncoder, RecordEncoder, _CORPUS
from holographic_reasoning import ResonatorNetwork
from holographic_creature import (GridWorld, CreatureEncoder, HolographicMind,
                                  _baseline, _train, _evaluate)


def title(s):
    print("\n" + "=" * 66 + "\n  " + s + "\n" + "=" * 66)


# 1. Core operations -------------------------------------------------------
title("1. Core VSA operations  (bind / unbind / bundle)")
rng = np.random.default_rng(0); dim = 1024
a, b = random_vector(dim, rng), random_vector(dim, rng)
print(f"  bind a,b then unbind by b recovers a : cosine = {cosine(unbind(bind(a, b), b), a):.2f}")
mix = bundle([a, b])
print(f"  a bundle stays similar to its parts  : cos(mix,a)={cosine(mix, a):.2f}  cos(mix,b)={cosine(mix, b):.2f}")


# 2. Key -> value memory ---------------------------------------------------
title("2. Key->value memory  (many facts in ONE vector)")
vocab = Vocabulary(dim, seed=1); mem = HolographicMemory(dim)
facts = {"france": "paris", "japan": "tokyo", "egypt": "cairo", "brazil": "brasilia"}
for k, v in facts.items():
    vocab.get(k); vocab.get(v); mem.learn(vocab.get(k), vocab.get(v))
vals = list(facts.values())
ok = sum(vocab.cleanup(mem.recall(vocab.get(k)), candidates=vals)[0] == v for k, v in facts.items())
print(f"  stored {len(facts)} country->capital pairs in one {dim}-d vector; recalled {ok}/{len(facts)} correctly")


# 3. Numbers ---------------------------------------------------------------
title("3. Numbers  (a real value as a vector, read back out)")
sc = ScalarEncoder(1024, lo=0, hi=10, seed=1)
print(f"  decode(encode(7.2))            = {sc.decode(sc.encode(7.2)):.2f}")
noisy = sc.encode(4.0) + 0.3 * random_vector(1024, np.random.default_rng(5))
print(f"  decode of a noisy 4.0 vector   = {sc.decode(noisy):.2f}   (survives noise)")
print(f"  similarity fades with distance : cos(e5,e6)={cosine(sc.encode(5), sc.encode(6)):.2f}  "
      f"cos(e5,e10)={cosine(sc.encode(5), sc.encode(10)):.2f}")


# 4. Text ------------------------------------------------------------------
title("4. Text  (meaning learned from co-occurrence, no gradients)")
te = TextEncoder(1024, window=2, seed=2)
for _ in range(5):
    for s in _CORPUS:
        te.learn(s.split())
print(f"  same-category close, cross-category far: cat~dog={cosine(te.wordvec('cat'), te.wordvec('dog')):.2f}  "
      f"cat~car={cosine(te.wordvec('cat'), te.wordvec('car')):.2f}")
print("  nearest to 'truck': " + ", ".join(f"{w} ({s:.2f})" for w, s in te.nearest('truck', 3)))


# 5. Mixed records ---------------------------------------------------------
title("5. Mixed records  (number + category + text in one vector)")
text = TextEncoder(2048, window=2, seed=2)
for _ in range(5):
    for s in _CORPUS:
        text.learn(s.split())
rec = RecordEncoder(2048, text, num_range=(0, 200), seed=7)
vec = rec.encode({"price": ("num", 142.5), "trend": ("cat", "up"),
                  "note": ("text", "the car raced past quickly")})
cat, sim = rec.read_category(vec, "trend", candidates=["up", "down", "flat"])
print(f"  one 2048-d vector holds all three fields, read individually:")
print(f"    price -> {rec.read_number(vec, 'price'):.1f}  (stored 142.5)   trend -> {cat} ({sim:.2f})  (stored 'up')")


# 6. Reasoning -------------------------------------------------------------
title("6. Reasoning  (resonator factors a bound fact back apart)")
v2 = Vocabulary(2048, seed=1)
subs = ["alice", "bob", "carol", "dave"]; rels = ["likes", "knows", "avoids", "trusts"]; objs = ["coffee", "jazz", "rain", "python"]
cb = lambda ws: np.array([v2.get(w) for w in ws])
res = ResonatorNetwork([cb(subs), cb(rels), cb(objs)])
r2 = np.random.default_rng(0); ok = 0
for _ in range(6):
    s, r, o = (int(r2.integers(4)) for _ in range(3))
    fact = bind(bind(v2.get(subs[s]), v2.get(rels[r])), v2.get(objs[o]))
    si, ri, oi = res.factor(fact)
    ok += (si == s and ri == r and oi == o)
print(f"  encoded subject(x)relation(x)object into ONE vector; recovered {ok}/6 facts exactly")


# 7. Creature --------------------------------------------------------------
title("7. Creature  (learning to forage from scratch -- no neural net)")
import io, contextlib
enc = CreatureEncoder(256, seed=1)
mind = HolographicMind(256, GridWorld.ACTIONS, k=15, epsilon=0.35, novelty_bonus=0.1, memory_cap=5000, seed=2)
world = GridWorld(7, 7, n_poison=0, seed=3)
br, bf = _baseline(world, enc)
with contextlib.redirect_stdout(io.StringIO()):          # hide the per-block training log
    _train(world, enc, mind, episodes=120)
er, ef = _evaluate(world, enc, mind)
print(f"  random baseline    : reward {br:+.2f}, food eaten {bf:.1f}")
print(f"  after 120 episodes : reward {er:+.2f}, food eaten {ef:.1f}   (it taught itself to find food)")
print("  (run `python holographic_creature.py` for the full demo incl. poison avoidance)")


# 8. Image archive ---------------------------------------------------------
title("8. Image archive  (recall a clean image from a wrecked query)")
from holographic_archive import HolographicArchive, _gallery
from holographic_image import _psnr
imgs = _gallery(128)
tags = [["quadrants", "colour"], ["bands", "horizontal"], ["gradient", "diagonal"],
        ["radial", "rings", "pink"], ["ripples", "waves"], ["checker", "squares"]]
arc = HolographicArchive((128, 128, 3), capacity=len(imgs), keep=2000, dim=32768, seed=0)
for im, t in zip(imgs, tags):
    arc.add(im, tags=t)
i = 3; q = np.clip(imgs[i] + 0.5 * np.random.default_rng(1).standard_normal(imgs[i].shape), 0, 1)
j, recon = arc.recall(q, mask=arc.damage_mask(0.4, 7))
print(f"  query = image #{i} under heavy noise AND 40% of the plate destroyed")
print(f"  recalled #{j} ({'correct' if j == i else 'wrong'}), reconstructed at {_psnr(imgs[i], recon):.0f} dB")

# cross-modal: no picture at all, just words
k, _r, conf = arc.recall_by_tags(words=["radial", "pink"])
print(f"  cross-modal: the words 'radial pink' alone recalled image #{k} (conf {conf:.2f})")

# quantize the plates and confirm content recall survives the 8x shrink
before = arc.stored_bytes(); arc.quantize(4); after = arc.stored_bytes()
still = sum(arc.recall(imgs[n])[0] == n for n in range(arc.n))
print(f"  4-bit plates: store {before//1000} KB -> {after//1000} KB, recall still {still}/{arc.n}")

print("\n" + "-" * 66)
print("  All eight subsystems ran on the same vector substrate. Wired up.")
print("-" * 66 + "\n")
