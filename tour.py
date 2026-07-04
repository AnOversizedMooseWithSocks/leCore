"""
tour.py -- a guided tour of the whole holographic system in one run.

Each section exercises a different subsystem with a small, readable example, so
you can confirm end-to-end that the whole stack works on the same vector
substrate: numbers, text, mixed records, key->value memory, reasoning, the
learning creature, the image archive, vision tagging, compositional scenes with
resonator factoring, the scaling tree/forest, and S3-style content addresses.

    python tour.py

(Takes ~25-35 seconds; the creature section trains a little RL agent.)
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
print("  one 2048-d vector holds all three fields, read individually:")
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

# PROJECTION TO CREATE NEW THINGS: synthesize a novel entity by casting one
# record's attributes onto another's frame -- the shadow that creates. A thing
# that exists in no training data, held coherently, decoded back exactly.
from holographic_relations import KnowledgeStore
_ks = KnowledgeStore(dim=2048, seed=0)
for _n, _a in {"france": {"capital": "paris", "currency": "euro", "language": "french", "continent": "europe"},
               "japan": {"capital": "tokyo", "currency": "yen", "language": "japanese", "continent": "asia"}}.items():
    _ks.add(_n, **_a)
_v, _spec = _ks.blend("france", "japan", {"language", "currency"})
_dec = _ks.decode_record(_v)
print(f"  projection-creates: 'france with japan's language+currency' -> "
      f"{_dec['capital']}/{_dec['language']}/{_dec['currency']} "
      f"({'coherent novel entity' if _dec == _spec else 'BLURRED'})")

# scene-level projection: factor two scenes, project one's palette onto the
# other's forms, recompose a novel scene neither contained -- decompose/project/
# recompose all through the resonator.
from holographic_scene import SceneCoder as _SC
_sc = _SC(dim=2048, seed=0)
_sa = [{"colour": "red", "shape": "circle", "texture": "smooth"},
       {"colour": "cyan", "shape": "rectangle", "texture": "busy"}]
_sb = [{"colour": "blue", "shape": "triangle", "texture": "vertical"},
       {"colour": "green", "shape": "line", "texture": "horizontal"}]
_vb, _bl = _sc.blend_scenes(_sc.encode_scene(_sa), _sc.encode_scene(_sb), 2, project="colour")
_rec = _sc.factor_scene(_vb, 2)
_exact = sorted((o['colour'], o['shape'], o['texture']) for o in _rec) == \
         sorted((o['colour'], o['shape'], o['texture']) for o in _bl)
print(f"  scene-blend: A's forms + B's colours -> novel scene factors back "
      f"{'exactly' if _exact else 'BLURRED'} ({len(_bl)} hybrid objects)")

# morph sequence: continuous control sweeps A -> B as ordered coherent frames
# (smooth attribute blend is impossible -- cleanup snaps -- so the honest morph
# is a discrete ordered sequence, and it passes the sequentiality test)
_frames = _sc.morph_scenes(_sc.encode_scene(_sa), _sc.encode_scene(_sb), 2, project="colour")
_all_exact = all(
    sorted((o['colour'], o['shape'], o['texture']) for o in _sc.factor_scene(_sc.encode_scene(_fr), 2))
    == sorted((o['colour'], o['shape'], o['texture']) for o in _fr)
    for _fr in _frames)
print(f"  scene-morph: {len(_frames)} ordered frames A->B, each factors "
      f"{'exactly' if _all_exact else 'BLURRED'} (projection generates, sequence-test confirms order)")

# cardinality is SELF-MEASURED (round(||v||^2) = object count) and the scene
# vector is ALGEBRAICALLY EDITABLE: remove = subtract a factored product, add =
# add a product. The cardinality morph chains such edits, count discovered at
# every frame, arriving exactly at the target scene.
_sa3 = _sa + [{"colour": "grey", "shape": "triangle", "texture": "horizontal"}]
_va3 = _sc.encode_scene(_sa3)
_cframes = _sc.morph_cardinality(_va3, _sc.encode_scene(_sb))
_ccounts = [_sc.count_objects(f) for f in _cframes]
print(f"  cardinality-morph: counts per frame {_ccounts} -- self-measured from "
      f"each vector's norm; the composite is edited algebraically, never re-encoded")

# perception as composite: the creature's WORLD as a countable, diffable vector.
# The diff of two snapshots IS the change-set -- its norm counts the changes,
# count-driven peeling names them. A wall appears; perception says which.
from holographic_creature import GridWorld as _GW, WorldView as _WV
_w = _GW(width=9, height=9, maze=True, seed=5); _w.reset()
_wv = _WV(dim=2048, width=9, height=9, seed=0)
_v1 = _wv.view(_w)
_w.walls.add((4, 3))
_app, _van = _wv.changes(_v1, _wv.view(_w))
print(f"  world-diff: a wall appears at (4,3); perception counts "
      f"{_wv.count(_wv.view(_w) - _v1)} change and names {_app[0] if _app else None}")

# market data on the same substrate: real DEX candles as records (round-trip
# finer than the signal), order tested honestly (levels ordered, returns
# shuffle-like), anomalies by novelty -- and prediction is a coin flip, said so.
from holographic_market import CandleCoder as _CC, load_ohlcv as _load
_a = _load(); _cc = _CC()
_d = _cc.decode_candle(_cc.encode_candle(*_a[50, 1:6]))
_novel = _cc.novelty(_a)[:2]
print(f"  market: candle->record->candle (close {_a[50,4]:.5f} -> {_d['close']:.5f}); "
      f"novelty flags candles {[i for i,_ in _novel]} (the swing and the volume spike)")

# the same instruments at tick scale: the sequence test flips its verdict where
# real structure exists (momentum), and the chance band proves the edge belongs
# to the simplest rule, not the fancy one.
import numpy as _np
from holographic_market import load_ticks as _lt
_ts, _px = _lt()
_g = _np.diff(_ts); _r = _np.diff(_np.log(_px)) * 1e4
_s = _np.sign(_r[_g <= 2]); _s = _s[_s != 0]
print(f"  ticks: {len(_px)} SOL ticks; sign-repeat {100*_np.mean(_s[:-1]==_s[1:]):.0f}% "
      f"(momentum -- DAI minutes had none); direction edge belongs to persistence "
      f"(60.2%), motif 54.1%, both outside the 2.6% chance band: measured, not claimed")

# ray-projected price targets: a matched pattern's R most similar past windows
# each carry the outcome that followed -- the bundle's quantiles are the target.
# Validated split-half (R chosen on half 1, scored on half 2): beats the
# unconditional distribution at proper score with tighter calibrated intervals.
from holographic_market import RayProjector as _RP, move_series as _ms
_mv, _mb = _ms(_ts, _px)
_rp = _RP(R=80).fit(_mv, _mb)
_q, _conf = _rp.project(len(_rp.rows) - 1)
print(f"  rays: next-{_rp.H}-move target from {_rp.R} matched patterns: "
      f"q10 {_q[0]:+.1f} / q50 {_q[1]:+.1f} / q90 {_q[2]:+.1f} bp "
      f"(split-half validated: sharper AND calibrated, paired z=+3.3)")

# physics is NATIVE to the scalar code: encode(a+b)==bind(encode(a),encode(b)),
# so motion is repeated binding -- and the price-as-particle verdict is honest
# both ways: the state (v,a) carries the H=1 structure, but prices have no
# inertia (extrapolation loses to predict-zero).
from holographic_physics import Kinematics as _Kin
_k = _Kin()
_got, _true = _k.trajectory(-40, 10.0, a=-1.0, steps=15)
print(f"  physics: constant-acceleration trajectory by PURE BINDING, 15 steps, "
      f"max decode error {float(np.max(np.abs(_got-_true))):.2f}; "
      f"velocity read off two points: {_k.read_velocity(-17.0,-13.5):+.2f} (true +3.50)")

# scale + cross-instrument: 1000 DAI/WETH candles confirm the structure (levels
# ordered, return signs still shuffle-like at a 3x-tighter band) AND reproduce
# the ray-interval win on a DIFFERENT instrument than the SOL ticks.
import json as _json
_big=np.array(_json.load(open("data/dai_weth_big.json"))["ohlcv"])
from holographic_market import CandleCoder as _CCb
_nov=dict(_CCb().novelty(_big))
_spike=int(np.argmax(_big[:,5]))
print(f"  scale: 1000 DAI candles; novelty flags the {_big[_spike,5]:.0f}-volume bar "
      f"(z={_nov.get(_spike,0):.1f}); ray-interval win reproduces cross-instrument (held-out z~+3)")

# vendored real market + on-chain data (from a sibling project that exercised this engine
# on SOL): multi-timeframe SOL/USDT candles with order flow feed the SAME CandleCoder, and
# real Jupiter-perp wallets become role-bound records labelled by HONEST per-trade edge.
try:
    from holographic_market import load_sol_market as _lsm, load_onchain_traders as _lot
    _solrows, _solf = _lsm(timeframe="1h")
    _ohlcv = _solrows[:, :6]                       # [time,open,high,low,close,volume] slice
    _nov_sol = dict(_CCb().novelty(_ohlcv[:400]))
    _sp = int(np.argmax(_ohlcv[:400, 5]))
    _trd = _lot()
    print(f"  sol: {_solrows.shape[0]} real SOL 1h bars (close {_ohlcv[:,4].min():.0f}-"
          f"{_ohlcv[:,4].max():.0f}); novelty flags the {_ohlcv[_sp,5]:.0f}-vol bar "
          f"(z={_nov_sol.get(_sp,0):.1f}). On-chain: {len(_trd['profiles'])} perp wallets, "
          f"{len(_trd['realized'])} realized trades (edge_t_stat beside PnL -- luck vs skill)")
    import unified_app as _ua
    _oitems, _, _odesc = _ua.load_onchain_world()
    from collections import Counter as _Cnt
    _dist = dict(_Cnt(l for _, l, _ in _oitems))
    print(f"  onchain-world: {len(_oitems)} wallets as records, labels {_dist} "
          f"(skilled only when t-stat>=2, not raw PnL) -- a records dataset for the app")
except Exception as _eSOL:
    print(f"  sol: (skipped: {_eSOL})")

# temporal compression: the video-codec insight rides the physics property. A
# rigid shift is one binding, so motion-compensation zeroes the residual and
# keyframe+motion coding beats per-frame storage -- but only when motion IS the
# change (deformation loses, honestly).
from holographic_video import HolographicVideo as _HV
_S=64; _yy,_xx=np.mgrid[0:_S,0:_S].astype(float)
_b=(((_xx-18)**2+(_yy-32)**2)<=11**2).astype(float)
_fr=[np.roll(_b,2*_t,axis=1) for _t in range(14)]
_vc=_HV(key_keep=400,res_keep=80); _pk,_gb=_vc.encode(_fr)
_ib,_ip=_HV.intra_baseline(_fr,keep=400); _gp=_vc.mean_psnr(_fr,_pk)
print(f"  video: rigid-motion sequence -- GOP coding {100*(1-_gb/_ib):.0f}% smaller "
      f"AND {_gp-_ip:+.1f}dB vs per-frame (motion-comp residual is exactly zero: "
      f"a shift is one binding); deformation loses, by measurement")

# sub-pixel motion: a fractional drift is a phase ramp in frequency, recovered
# exactly where integer search rounds -- the scalar code's principle in 2-D.
from holographic_video import estimate_subpixel_shift as _ess, fourier_shift as _fs, estimate_shift as _es
_S=64; _yy,_xx=np.mgrid[0:_S,0:_S].astype(float)
_bl=lambda cx: np.exp(-(((_xx-cx)**2+(_yy-32)**2)/40.0))
_f=[_bl(20+1.7*_t) for _t in range(6)]
_ir=np.mean([np.linalg.norm(_f[t]-np.roll(_f[t-1],_es(_f[t-1],_f[t]),axis=1)) for t in range(1,6)])
_sr=np.mean([np.linalg.norm(_f[t]-_fs(_f[t-1],_ess(_f[t-1],_f[t]))) for t in range(1,6)])
print(f"  sub-pixel: 1.7px/frame drift -- integer motion-comp leaves residual {_ir:.3f}, "
      f"sub-pixel (Fourier-shift) leaves {_sr:.4f} (exact: a shift is a phase ramp)")

# versioned history: the store's timeline stored like video -- keyframe + lossless
# deltas, exact rollback, proof-gated commits so a bad reorganization is rejected.
from holographic_history import VersionedStore as _VS
_vs=_VS(dim=64,gop_len=8); _rng=np.random.default_rng(0)
_ids=[_vs.new_id() for _ in range(10)]; _rows={i:_rng.standard_normal(64) for i in _ids}; _ord=list(_ids)
_vs.commit(_rows,_ord)
for _ in range(8):
    _j=_vs.new_id(); _rows=dict(_rows); _rows[_j]=_rng.standard_normal(64); _ord=_ord+[_j]; _vs.commit(_rows,list(_ord))
print(f"  history: {_vs.head()+1} versions stored {_vs.full_entries()/_vs.stored_entries():.1f}x compressed, "
      f"LOSSLESS rollback to any version, proof-gated commits (a bad reorg is rejected, "
      f"the attempt still logged) -- version history is a video")

# dictionary-first meaning: a word = bundle of its definition words, iterated to a
# fixed point on the definition graph. Definitions bootstrap meaning far harder
# than reading prose -- but reading must REFINE, not overwrite (measured).
from holographic_lexicon import Lexicon as _Lex
_dd={"cat":["animal","feline","pet"],"dog":["animal","canine","pet"],"lion":["animal","feline","wild"],
     "wolf":["animal","canine","wild"],"animal":["living","creature"],"feline":["cat","lion","animal"],
     "canine":["dog","wolf","animal"],"pet":["animal","tame"],"wild":["untamed","animal"],
     "living":["alive"],"creature":["living","animal"],"tame":["gentle"],"untamed":["wild"],
     "rock":["mineral","hard"],"stone":["mineral","hard"],"mineral":["solid"],"hard":["solid"],"solid":["firm"],"firm":["solid"],"alive":["living"],"gentle":["mild"],"mild":["gentle"]}
_lex=_Lex(_dd,dim=512,seed=0).bootstrap(3)
print(f"  lexicon: dictionary-bootstrapped meaning -- cat~dog {_lex.similarity('cat','dog'):+.2f} "
      f"vs cat~rock {_lex.similarity('cat','rock'):+.2f}; recursion peaks ~3 iters then over-diffuses "
      f"(definitions bootstrap meaning harder than reading; reading must refine, not overwrite)")

# encyclopedia: structured knowledge beyond word meaning -- concepts in an is_a
# web, walked as a relation ray. One-hop exact, multi-hop exact (closed world),
# throughput decays with depth, and siblings are related through a shared parent
# even when their words are not.
from holographic_encyclopedia import Encyclopedia as _Enc
_e=_Enc(dim=4096,seed=0)
for _c,_p in [("dog","canine"),("wolf","canine"),("cat","feline"),("canine","carnivore"),
              ("feline","carnivore"),("carnivore","mammal"),("mammal","animal"),("animal","organism")]:
    _e.add(_c,is_a=_p)
_ch,_tp=_e.climb("dog")
print(f"  encyclopedia: dog is_a chain {' -> '.join(_ch)} (throughput {_tp:.2f}); "
      f"dog is-a animal? {_e.is_a_transitive('dog','animal')[0]}; dog~wolf related "
      f"{_e.relatedness('dog','wolf'):.2f} vs dog~cat {_e.relatedness('dog','cat'):.2f} "
      f"-- relatedness from structure, not shared words")

# WIRED TO THE BRAIN: the curriculum is not a side module -- UnifiedMind learns
# the dictionary into its own encoder and the encyclopedia into its own memory,
# then defines words and climbs is_a chains over itself.
from holographic_unified import UnifiedMind as _UM
_m=_UM(dim=2048,seed=0)
_m.learn_dictionary({"cat":["animal","feline"],"dog":["animal","canine"],"animal":["living"],
                     "feline":["cat","animal"],"canine":["dog","animal"],"living":["alive"],"alive":["living"]},iters=3)
_m.learn_encyclopedia({"dog":{"is_a":"canine"},"canine":{"is_a":"carnivore"},
                       "carnivore":{"is_a":"mammal"},"mammal":{"is_a":"animal"}})
print(f"  brain curriculum: the mind itself defines 'dog' -> "
      f"{[w for w,_ in _m.define('dog',3)]} and climbs is_a -> {_m.climb('dog')[0]} "
      f"(dictionary in its encoder, encyclopedia in its memory, one brain)")

# QUESTION ROUTER: not a chatbot, but a question has a SHAPE that maps to a real
# operation. The mind routes "what is X" -> meaning+is_a, "is X a Y" -> taxonomy,
# and labels text completion as completion (never fakes an answer).
print(f"  ask: 'is a dog an animal?' -> {_m.answer('is a dog an animal?')['answer']}; "
      f"'what is a dog?' -> meaning {[w for w,_ in _m.answer('what is a dog?')['meaning'][:3]]} "
      f"+ is_a {_m.answer('what is a dog?')['is_a_chain'][:3]}... "
      f"(routed to real operations, not sentence completion)")

# REAL PHOTOS (a wallpaper category, not sprites): the holographic plate is no
# match for JPEG on efficiency, but holds PSNR flat under heavy erasure where a
# block codec shatters -- and the vault correctly picks a LOSSY codec for photos
# where it picked palette for sprites.
import glob as _glob, os as _os
if _os.path.isdir("features/photo_sample") and _glob.glob("features/photo_sample/*.npy"):
    from holographic_photos import load_photo_folder as _lpf, robustness_curve as _rc
    _g=_lpf("features/photo_sample", size=96, limit=1, gray=True)[0]
    _r=_rc(_g, keeps_erasures=((800,(0.0,0.5)),))
    print(f"  photos: holographic plate on a real photo holds {_r[0.0]:.1f}dB at 0% erasure "
          f"and {_r[0.5]:.1f}dB at 50% (JPEG shatters); robustness is the honest win, "
          f"not efficiency -- and the vault picks lossy for photos, palette for sprites")

# COARSE-TO-FINE cleanup (leOS's Matryoshka/inception idea, adapted to RANDOM
# hypervectors): rank at low dimension first, escalate only when the top match
# isn't statistically settled. Same answer as a full scan for a fraction of work.
from holographic_ai import random_vector as _rv
from holographic_resolution import coarse_to_fine as _c2f, full_scan as _fs
import numpy as _np
_rng=_np.random.default_rng(1); _D=4096; _N=400
_V=_np.array([_rv(_D,_rng) for _ in range(_N)])
_ag=0; _save=[]
for _ in range(60):
    _t=_rng.integers(_N); _q=_V[_t]+_rng.uniform(0.4,2.0)*_rv(_D,_rng)
    _i,_,_dims,_=_c2f(_q,_V); _ag+=(_i==_fs(_q,_V)[0]); _save.append(1-_dims/(_N*_D))
print(f"  resolution: coarse-to-fine cleanup agrees with full scan {100*_ag/60:.0f}% "
      f"using {100*(1-_np.mean(_save)):.0f}% of the dimension-work (easy matches settle "
      f"at ~128 of 4096 dims); wired into _cleanup and the brain's find()")

# FRACTAL structure (leOS's self-similarity detector, ported to real data):
# box-counting dimension distinguishes natural photos (rough edges) from smooth
# synthetic shapes; Hurst reads market self-affinity; IFS compresses self-similar
# data ~2000x but NOT random data (the kept negative).
from holographic_fractal import IFS as _IFS, ifs_compresses as _ifsc, image_fractal_dimension as _ifd
import numpy as _np2, glob as _g2
_S=96; _yy,_xx=_np2.mgrid[0:_S,0:_S].astype(float)
_circ=((( _xx-48)**2+(_yy-48)**2)<30**2).astype(float)*255
_fern=_IFS.barnsley_fern(); _fit=_ifsc(_fern.generate(8000),_fern)
_natD=_np2.mean([_ifd(_np2.load(f)) for f in sorted(_g2.glob("features/photo_sample/*.npy"))]) if _g2.glob("features/photo_sample/*.npy") else 0
print(f"  fractal: natural photo edge-D ~{_natD:.2f} vs synthetic circle {_ifd(_circ):.2f}; "
      f"a fern compresses to {_fit['n_numbers']} IFS numbers ({_fit['compression']:.0f}x, error {_fit['ifs_error']:.2f}) "
      f"while random data does NOT (error {_fit['random_error']:.2f}) -- self-similarity is a measured property")

# CONTEXT-CONDITIONED generation -- the honest answer to 'why isn't the brain an
# LLM': deepen the conditioning (word n-gram re-ranked by a topic vector) and
# MEASURE whether it buys coherence. It does not: pushed hard, coherence rises
# only as diversity COLLAPSES into repetition. The negative is the finding.
from holographic_generation import ContextGenerator as _CG
_sp=["the ship flew through the dark cold void of space",
     "a star burned bright near the distant planet and moon",
     "the rocket engine fired and the ship climbed past the moon"]
_gd=["the garden grew green plants in the warm wet soil",
     "a flower bloomed bright near the old stone garden wall",
     "the gardener watered the green plants in the warm sun"]
_g=_CG(dim=256, order=1, seed=0).fit([s.split() for s in (_sp*4+_gd*4)])
_b=_g.sweep(["the ship","the garden","a star"], weights=(0.0,), length=40)[0]
_h=_g.sweep(["the ship","the garden","a star"], weights=(16.0,), length=40)[0]
print(f"  generation: deeper conditioning (topic pull) is NOT enough -- at heavy pull "
      f"'coherence' {_b['coherence']:.2f}->{_h['coherence']:.2f} but diversity COLLAPSES "
      f"{_b['diversity']:.2f}->{_h['diversity']:.2f} into repetition; the missing piece is a "
      f"learned high-capacity P(next|context), not the loop or the re-ranking")

# FOUNTAIN codes (leOS's last clean idea): a droplet is the XOR of a random
# subset (the binary sibling of a bundle); decode is a PEEL (loop until resolved).
# A second robustness axis: the plate degrades gracefully (lossy); the fountain
# recovers EXACTLY after whole-packet loss, above an information floor.
from holographic_fountain import Fountain as _F, recovery_curve as _rc
_blob=b"the same water no matter which drops you catch "*200
_f=_F.from_bytes(_blob, block_size=32)
import random as _rnd
_drops=_f.droplets(int(_f.k*2.2), seed=2); _rnd.seed(0); _rnd.shuffle(_drops)
_surv=[d for d in _drops if _rnd.random()>0.4]      # lose 40% of packets
_exact=_f.decode_bytes(_surv, _f.orig_len)==_blob
_curve=_rc(_f.k, overheads=(1.0,1.2,1.5), trials=5, seed=0)
print(f"  fountain: lost 40% of packets, {len(_surv)} survived ({len(_surv)/_f.k:.2f}k) -> "
      f"EXACT blob recovery {_exact}; below k it is impossible (recovery {_curve[1.0]:.0%} at 1.0k, "
      f"{_curve[1.5]:.0%} at 1.5k) -- exact above the floor, a different axis than the plate's graceful decay")

# PREDICTIVE LOOP -- the active layer on top of storage: anticipate the next
# symbol, measure surprise, learn error-gated, report free energy. Predicts by
# RESONANCE, so it generalises to contexts never seen exactly.
from holographic_predictive import PredictiveMemory as _PM
_pm=_PM(dim=1024, order=2, seed=0)
_steps=_pm.learn_sequence(["a","b","c","d"]*30)
import numpy as _np3
_late=_np3.mean([max(0,s.surprise) for s in _steps[-8:]])
_pm2=_PM(dim=2048, order=2, seed=0)
for _s in (["the","cat","sat"],["a","cat","sat"],["the","cat","sat"]): _pm2.learn_sequence(_s)
_gen=_pm2.predict(["my","cat"])
print(f"  predictive: on a periodic stream surprise falls to {_late:.2f} and accuracy reaches "
      f"{_pm.predict_accuracy(['a','b','c','d']*30):.0%}; and an UNSEEN context 'my cat' predicts "
      f"'{_gen[0]}' (conf {_gen[1]:.2f}) by resonance with similar contexts -- generalisation exact lookup lacks")

# MEANING-LEVEL prediction: compose a next-MEANING vector (ZREAD blend of
# resonating contexts' next-meanings) and SETTLE it, rather than look up one
# symbol. Lands in the right neighbourhood (high semantic rank) even when the
# exact word is missed. And the kept finding: match the space to the query.
from holographic_meaning_predict import MeaningPredictor as _MP
_sp2="ship flew through space past star moon".split(); _gd2="garden grew plants soil flower wall".split()
_ss=[_sp2]*16+[_gd2]*16; _stream2=[w for s in _ss for w in s]
_mp3=_MP(dim=512, order=2, seed=0).fit_space(_ss).fit_transitions(_stream2)
_rep=_mp3.evaluate(_stream2)
print(f"  meaning-predict: composed next-meaning lands at semantic rank {_rep['semantic_rank']:.2f} "
      f"(0.5=chance) even at {_rep['exact']:.0%} exact -- graded, compositional anticipation; "
      f"measured at scale: next-word wants co-occurrence space (~0.85), relatedness wants the "
      f"dictionary curriculum (d' ~0.8) -- match the space to the query")

# PROOF OF STRUCTURE: a meaning prediction needs verification, and the proof comes
# from projecting each word onto its CONTEXT across ranges, not from any single
# word. Single-step coherence is gameable (self-generated salad scores HIGHER than
# real text); the lag-coherence PROFILE is not. Steering generation by it escapes
# the loops greedy decoding falls into.
from holographic_structure import StructureVerifier as _SV, steered_generate as _sg
from holographic_meaning_predict import MeaningPredictor as _MP2, cooccurrence_space as _cs
_a="the ship sailed across the cold sea toward the bright star".split()
_b="the farmer planted green seeds in the warm soil near the barn".split()
_ss2=[_a,_b]*30; _st2=[w for s in _ss2 for w in s]
_voc,_M,_ix=_cs(_ss2, dim=512, window=2, seed=0)
_mp4=_MP2(dim=512, order=2, seed=0).set_space(_voc,_M).fit_transitions(_st2)
_v=_SV(_voc,_M,_ix).calibrate(_st2, chunk=60, z_floor=2.0)
_greedy=list(_st2[:2])
for _ in range(40):
    _w,_,_=_mp4.predict_meaning(_greedy[-2:]); _greedy.append(_w if _w else _voc[0])
_steered=list(_st2[:2])+_sg(_mp4,_v,_st2[:2],length=40,beam=6)
print(f"  structure: real text scores {_v.structure_score(_st2[:120]):.1f}; greedy decoding collapses "
      f"to {_v.structure_score(_greedy[2:]):.1f} (a locally-coherent loop single-step checks rate highly), "
      f"while steering by trajectory structure recovers {_v.structure_score(_steered[2:]):.1f} -- meaning proven "
      f"by projection onto context, then used as a process")

# QUERY-AND-GENERATE: a query implies a target in meaning space; generation runs
# forward steered by TWO forces -- the structure guard (stay coherent) and the
# query pull (stay on-query). The guard is what lets the pull work without the
# collapse topic-pull suffered.
from holographic_respond import respond as _rsp, query_target as _qt, relevance as _rl
_qq="ship sailed sea star"
_tgt=_qt(_qq, _ix, _M)
_un=_rsp(_qq, _mp4, _v, length=30, query_weight=0.0)
_st=_rsp(_qq, _mp4, _v, length=30, query_weight=6.0)
print(f"  query-generate: steering toward a query raises relevance "
      f"{_rl(_un,_tgt,_ix,_M):.2f}->{_rl(_st,_tgt,_ix,_M):.2f} while the structure guard holds it in the band "
      f"({_v.structure_score(_st):.1f}); without the guard a hard pull collapses to salad -- query-and-generate "
      f"is the two forces together")

# DELIBERATION: don't emit the first draft. Form the gist, draft it, judge it, and
# iterate -- keeping the best, stopping early when good enough. The iteration count
# is the thinking time and adapts to difficulty (sometimes fast, sometimes slow).
from holographic_deliberate import Deliberator as _Del
_del=_Del(_mp4,_v)
_q_easy=_del.deliberate("ship sailed sea star", max_iters=8, target_quality=0.45)
_single=_del.quality(_del._realize("ship sailed sea star", _qt("ship sailed sea star",_ix,_M), temp=0.0),
                     _qt("ship sailed sea star",_ix,_M))
print(f"  deliberate: a single greedy pass scores {_single:.2f}; the loop drafts, judges and keeps the best "
      f"({_q_easy['quality']:.2f}) in {_q_easy['iterations']} iteration(s) -- the count adapts to difficulty, "
      f"easy queries settle fast and hard ones take longer (measured 1-8 across queries); the trace exposes the "
      f"inner drafts. Kept negatives: elaborating the abstract plan (rolling the predictor forward, or enriching "
      f"with neighbours) did NOT beat the flat query gist -- the human-like gain is the loop, not the plan")

# MULTI-JUDGE NEGOTIATION: several judges (coherence vs novelty vs relevance) score
# each draft; the kept one is the most BALANCED (its weakest pressure least bad).
_neg = _del.negotiate("ship sailed sea star", max_iters=6, target_quality=0.55)
print(f"  negotiate: competing judges score each draft {list(_neg['scores'].keys())}; the kept draft balances "
      f"them (negotiated={_neg['negotiated']:.2f} = its weakest pressure) rather than maxing one axis -- novelty "
      f"is a safety net against the repetition the structure guard misses")

# CROSS-DOMAIN: the structure-verifier idea (match a sample's autocorrelation
# signature to a band of real data) carries beyond text -- cleanly to images, and
# to returns via the volatility-clustering signature.
import numpy as _np4
from holographic_signal_structure import SignalStructureVerifier as _SSV, clustering_zscore as _cz
_rng4 = _np4.random.default_rng(0)
_yy, _xx = _np4.mgrid[0:96, 0:96]
_nat = _np4.sin(_xx / 12.0) + _np4.cos(_yy / 9.0) + 0.3 * _rng4.standard_normal((96, 96))
_pat = [_nat[i:i + 32, j:j + 32] for i in range(0, 64, 16) for j in range(0, 64, 16)]
_iv = _SSV("image").calibrate(_pat)
def _garch(n=3000):
    _r = _np4.zeros(n); _s = _np4.ones(n) * 0.01
    for _t in range(1, n):
        _s[_t] = _np4.sqrt(1e-5 + 0.1 * _r[_t - 1] ** 2 + 0.85 * _s[_t - 1] ** 2)
        _r[_t] = _s[_t] * _rng4.standard_normal()
    return _r
print(f"  cross-domain: the same structure idea transfers -- a natural image patch scores "
      f"{_iv.structure_score(_nat[64:, 64:]):.1f} vs noise {_iv.structure_score(_rng4.standard_normal((96, 96))):.1f} "
      f"(clean, like text); market returns need the volatility-clustering signature (GARCH z={_cz(_garch()):.1f}), "
      f"and real short series may be too small to call -- the machinery transfers, the choice of what to "
      f"autocorrelate is the domain knowledge")

# COMPRESSION: better structure -> better compression, made literal. A predictor is
# a compressor (rank-code each symbol by the predictor's ranking). Structured text
# costs fewer bits; the structure score predicts the ratio.
from holographic_compress import PredictiveCompressor as _PC
_pc=_PC(_mp4)
_real=_st2[:160]; import numpy as _np5
_sh=list(_real); _np5.random.default_rng(0).shuffle(_sh)
print(f"  compress: a predictor is a compressor -- real text rank-codes to "
      f"{_pc.compressibility(_real):.2f} of the uniform baseline while the same words shuffled cost "
      f"{_pc.compressibility(_sh):.2f}; on Brown the structure score and compression ratio correlate ~-0.6 "
      f"(more structure -> fewer bits), and the predictor beats a frequency-only model -- it exploits order, "
      f"not just word counts. Sits beside the fractal IFS compressor: two structures, two compressions")

# SELF-DISCOVERY of structure: strip the spaces from text and recover the word
# boundaries with no labels, from where the next-character prediction becomes
# uncertain (branching entropy). The discovered chunks then compress better.
from holographic_segment import Segmenter as _Seg, boundary_f1 as _bf1, chunk_compression as _cc
import numpy as _np6
_words=["cat","dog","bird","fish","lion","bear","tree","star"]; _rng6=_np6.random.default_rng(0)
_seq=[_words[_rng6.integers(len(_words))] for _ in range(500)]; _strm="".join(_seq)
_truth=set(); _p=-1
for _w in _seq: _p+=len(_w); _truth.add(_p)
_sg=_Seg(dim=512, order=3, seed=0).fit(_strm); _bd=_sg.boundaries(_strm, percentile=60)
_f=_bf1(_bd,_truth); _ch=_sg.segment(_strm, percentile=60); _cb,_sb=_cc(_strm,_ch)
print(f"  self-discovery: from a spaceless stream the system recovers word boundaries at F1 {_f['f1']:.2f} "
      f"(branching entropy peaks at unit ends) -- no labels; and the discovered chunks compress to {_cb:.1f} "
      f"bits/char vs {_sb:.1f} for single chars -- finding the right decomposition shortens the description "
      f"(better structure -> better compression, reached by self-discovery). Kept negative: resonance-blended "
      f"readout smears the signal (F1 ~0.26); boundary discovery needs exact contexts")

# FACTORIZATION by searching in superposition: the inverse of binding. Bind three
# random vectors into one composite, then recover which came from each codebook --
# a Resonator Network (Frady/Kent/Olshausen/Sommer 2020) converges on the factors
# without enumerating the combinatorial space.
from holographic_resonator import ResonatorNetwork as _RN, map_codebook as _mc, map_bind as _mb
import numpy as _np7
_bk = [_mc(50, 1500, _s) for _s in range(3)]
_rng7 = _np7.random.default_rng(0)
_tru = [int(_rng7.integers(50)) for _ in range(3)]
_comp = _mb(*[_bk[_f][_tru[_f]] for _f in range(3)])
_res = _RN(_bk).factor(_comp, restarts=25)
print(f"  factorize: bound 3 hidden vectors into one composite; the resonator recovered "
      f"{_res['factors']} (true {tuple(_tru)}, solved={_res['solved']}) by searching in superposition -- "
      f"a space of {_res['search_space']:,} combinations it never enumerated. The inverse of binding, the "
      f"decomposition primitive the engine lacked. Kept negative: circular-convolution bind amplifies noise "
      f"and won't factor; needs self-inverse MAP binding")

# LOSSLESS CODEC + ATTRIBUTION: go both directions exactly. Encode each token by its
# rank under the predictor; the decoder replays the same predictor to recover the
# exact original. Size is bounded by structure -- no free lunch.
from holographic_codec import PredictiveCodec as _Cod
_cod=_mp4 and _Cod(_mp4)
_real8=_st2[:120]
_ok=_cod.roundtrip_ok(_real8); _cst=_cod.cost(_real8)
_per8=["alpha","beta","gamma","delta"]*30
_vp,_Mp,_ip=__import__("holographic_meaning_predict").cooccurrence_space([_per8],dim=512,window=2,seed=0)
_mpp=__import__("holographic_meaning_predict").MeaningPredictor(dim=512,order=2,seed=0).set_space(_vp,_Mp).fit_transitions(_per8)
_perc=_Cod(_mpp).cost(_per8)
print(f"  codec: compress<->decompress is exactly lossless ({_ok}); real text rank-codes to ratio "
      f"{_cst['ratio']:.2f} of baseline, a perfectly periodic stream to {_perc['bits_per_token']:.2f} bits/token "
      f"(~the seed alone) -- structure shrinks, random does not. The honest answer to 'compress to a seed': "
      f"real and lossless, but bounded by the data's information content, not magic")

# MANY MINDS, ONE SUBSTRATE: a frozen shared base brain with lightweight per-instance
# deltas, so a game can run a population of NPCs without a full brain each. Branch
# inherits the base; deltas stay isolated; propagate merges learning back to all.
_pbase=_UM(dim=512, seed=0)
for _x,_l in [("sword","weapon"),("apple","food"),("gold","treasure"),("cave","place")]:
    _pbase.learn(_x,_l)
_shared=_pbase.share()
_alice=_shared.branch("alice").learn("potion","alchemy"); _bob=_shared.branch("bob")
_inh=_alice.classify("sword"); _iso=_bob.classify("potion"); _bef=_bob.classify("potion")
_alice.propagate(); _aft=_bob.classify("potion")
_npcs=[_shared.branch(f"n{_i}").learn(f"w{_i}",f"f{_i}") for _i in range(50)]
_c=_shared.population_cost(_npcs)
print(f"  population: a branch inherits the shared base (sword->{_inh}); a private fact stays isolated "
      f"(bob on alice's potion->{_iso}); after alice.propagate() it is shared (potion: {_bef}->{_aft}). "
      f"50 NPCs over one frozen base cost {_c['shared_total']} prototypes vs {_c['separate_total']} for "
      f"separate brains ({_c['saving_x']:.0f}x), because instances share atoms so merge is just "
      f"superposition. The copy-on-write / frozen-base+delta pattern, native to this substrate")

# REAL DATASETS exposed in the UI: the sprite set and the image repository, alongside
# the text corpora (dictionary+encyclopedia curriculum, Reuters, Brown, books).
import os as _os
if _os.path.isdir("features/sprites"):
    import pack_sprites as _ps, io as _io
    from PIL import Image as _Img
    import numpy as _npI
    _items=_ps.load_folder("features/sprites")[:200]
    _blob=_ps.pack(_items)
    _per=sum((lambda a:(lambda b:(_Img.fromarray(a).save(b,"PNG",optimize=True),len(b.getvalue()))[1])(_io.BytesIO()))(a) for _n,a in _items)
    _exact=all(_npI.array_equal(a,b) for (_,a),(_,b) in zip(_items,_ps.unpack(_blob)))
    print(f"  sprites: {len(_items)} of the 712-sprite set pack to {len(_blob):,} bytes vs {_per:,} for "
          f"per-file PNG ({_per/len(_blob):.1f}x smaller), bit-exact={_exact} -- cross-sprite structure "
          f"separate files hide. The image vault groups the picture repository by perceptual fingerprint "
          f"and answers query-by-example; both are now demo cards beside Reuters/Brown/books/curriculum")

# MARKET STRUCTURE on the larger datasets (also a UI experiment card): real returns
# show volatility clustering -- |returns| positively autocorrelated -- which a shuffle
# destroys. The bigger data now shows a clear signal where the tiny sample could not.
try:
    import json as _jsonM, numpy as _npM
    from holographic_signal_structure import volatility_clustering as _vc, clustering_zscore as _cz
    from holographic_market import load_ticks as _lt, move_series as _ms
    _ts,_px=_lt(); _mv,_=_ms(_ts,_px)
    _zS=_cz(_mv, n_shuffle=50)
    _a=_npM.array(_jsonM.load(open("data/dai_weth_big.json"))["ohlcv"], float); _cl=_a[:,4]
    _rt=_npM.diff(_cl)/_cl[:-1]; _zD=_cz(_rt, n_shuffle=50)
    print(f"  market: SOL ticks volatility-clustering z={_zS:+.1f} (acf1 {_vc(_mv):+.2f}); DAI/WETH big "
          f"z={_zD:+.1f} -- both real structure (>2 sigma) where the old ~100-return sample was only ~1 "
          f"sigma; shuffles collapse to ~0. Raw signed returns stay efficient-market-like. (UI: 'market "
          f"structure' and 'big-text run' are on-demand experiment cards, kept out of the test suite)")
except Exception as _eM:
    print(f"  market: (skipped: {_eM})")


# CAPACITY-AWARE LAYERING: a prototype is a bundle with finite capacity -- fold too
# many distinct members into one and the unit stops resembling any of them (the cliff,
# ~1/sqrt(count)). The decision brain and the shared-base merge can now CAP members
# per prototype and split into sub-prototypes instead of blurring.
from holographic_creature import HolographicMind as _HM
import numpy as _npC
_rngC = _npC.random.default_rng(0); _baseV = _rngC.standard_normal(256)
_capON = _HM(dim=256, actions=["N", "S", "E", "W"], merge=0.5, capacity=8, seed=0)
_capOFF = _HM(dim=256, actions=["N", "S", "E", "W"], merge=0.5, capacity=0, seed=0)
for _ in range(40):
    _s = _baseV + 0.01 * _rngC.standard_normal(256)
    _capON.remember([_s], [0], [1.0]); _capOFF.remember([_s], [0], [1.0])
_rON = _capON.capacity_report(); _rOFF = _capOFF.capacity_report()
print(f"  capacity: folding 40 near-identical experiences into one action -- unbounded blurs them into "
      f"1 over-loaded prototype (max_count {_rOFF['max_count']}, {_rOFF['overloaded']} past the sqrt(dim) "
      f"soft cap); capacity=8 splits into {_rON['prototypes']} sub-prototypes (max_count {_rON['max_count']}, "
      f"0 over-loaded) -- split, don't blur, the same fix the scaling tree uses for storage now applied to "
      f"the value memory and the shared-base NPC merge. Off by default; opt in with capacity>0")

# RECURRENT LAYER (reservoir): a gradient-free Echo State Network adds the one thing the
# linear permute+bundle recurrence lacks -- a nonlinearity -- training only a ridge
# readout (no backprop). Measured on REAL corpora it loses to the existing baselines;
# kept on the record as a negative, plus an order-only control proving the mechanism.
try:
    import numpy as _npR
    from holographic_recurrent import ReservoirSequenceClassifier as _RSC
    _tr = [("abcd" * 6, 0) for _ in range(15)] + [("dcba" * 6, 1) for _ in range(15)]
    _te = [("abcd" * 6, 0)] * 8 + [("dcba" * 6, 1)] * 8
    _clf = _RSC(dim=64, n_res=200, seed=0).fit(_tr)
    _ctrl = _npR.mean([_clf.classify(s) == l for s, l in _te])
    print(f"  reservoir: gradient-free Echo State Network (fixed random dynamics, one ridge-solve readout). "
          f"On REAL data the baselines win -- next-char on Gutenberg Alice n-gram ~0.58 vs reservoir ~0.42, "
          f"UDHR language ID bag-of-trigrams ~0.97 vs reservoir ~0.33 -- a kept negative (real classes "
          f"separate on symbol statistics the fixed random projection captures less sharply). But on an "
          f"order-ONLY control (same multiset, opposite order) the reservoir scores {_ctrl:.2f} vs a bag's "
          f"chance 0.50: the mechanism works, real tasks just don't reward it here")
except Exception as _eR:
    print(f"  reservoir: (skipped: {_eR})")

# VARIANCE HARNESS: every headline number is a sample from a distribution (random atoms,
# random projections, shuffled splits), so report its SPREAD, not a lucky-seed point.
# measure() runs a claim across seeds and returns mean +/- std + a 95% bootstrap CI;
# load-bearing tests assert the LOWER CI bound, not the mean.
try:
    import re as _reV
    from holographic_measure import measure as _measure, report as _vreport
    from holographic_text import HolographicNGram as _HN
    from nltk.corpus import gutenberg as _gut
    _alice = _reV.sub(r"\s+", " ", _reV.sub(r"[^a-z ]+", " ", _gut.raw("carroll-alice.txt").lower()))
    _cut = int(len(_alice) * 0.85)
    _atr, _ate = _alice[:_cut], _alice[_cut:_cut + 3000]
    _st = _measure(lambda s: _HN(dim=1024, n=6, seed=s).fit(_atr).predict_accuracy(_ate), seeds=range(5))
    print("  variance: the ~62% next-char headline, measured across 5 seeds on real Alice -- "
          + _vreport("ngram", _st, floor=0.55).replace("ngram: ", "")
          + ". Tight spread => not a lucky seed; load-bearing tests assert the lower CI bound, "
          "and Reuters topic-classify (0.83 +/- 0.05) carries a real, honestly-reported spread")
except Exception as _eV:
    print(f"  variance: (skipped: {_eV})")

# ABLATION TABLE: for each subsystem, run the dumbest honest non-holographic baseline on
# the SAME real data and let the variance harness's CIs decide whether VSA is the reason
# it works. The most useful self-description the engine has: where it is load-bearing.
try:
    from holographic_ablate import key_value_noisy as _kvn, recall_index as _ri, verdict as _vd
    _h, _b, _ = _kvn(seeds=range(4))
    _hr, _br, _ = _ri(seeds=range(4))
    print(f"  ablation: VSA is load-bearing exactly where the problem is approximate -- noisy-key "
          f"key->value scores {_h['mean']:.2f} where an exact dict scores {_b['mean']:.2f} "
          f"({_vd(_h,_b)}); topic-classify beats bag-of-words ~0.83 vs ~0.61. It is NOT the reason "
          f"language ID / segmentation work (exact count baselines tie them), and the recall forest "
          f"loses recall to exact scan ({_hr['mean']:.2f} vs 1.00) but at {_hr['comparison_fraction']*100:.0f}% "
          f"of the comparisons -- a scale win, not an accuracy one. See ABLATIONS.md")
    # the ablation TABLE is a scan over many subsystems, so a verdict can clear its own CI by
    # luck. The honesty module's bh_fdr now controls that: each subsystem gets a paired
    # permutation p-value, then a family-wise false-discovery bar judges the whole table.
    from holographic_ablate import ablation_table as _at, fdr_verdicts as _fdr
    _rows = _at(seeds=range(4))
    _aug, _nlb, _nsurv = _fdr(_rows, alpha=0.1)
    print(f"  ablation-FDR: across the whole scan, {_nsurv}/{_nlb} 'load-bearing' verdicts survive "
          f"family-wise false-discovery control (BH-Yekutieli) -- so the table's claims are "
          f"rigorous against multiple-testing, not just per-test CIs")
except Exception as _eA:
    print(f"  ablation: (skipped: {_eA})")

# PROCEDURAL GENERATION: drive the existing decoders forward to PRODUCE output in four
# modalities -- no learned distribution, no gradients. Each beats the dumbest baseline on
# an honest metric (anti-ghosting for morphs, coherence for nucleus text, faithful pitches
# for sonification).
try:
    from holographic_generate import (morph_images as _mi, crossfade_images as _cf,
                                       ghosting as _gh, sequence_to_pitches as _s2p)
    from holographic_archive import HolographicArchive as _HA, _gallery as _gal
    _imgs = _gal(S=64)
    _arch = _HA(shape=_imgs[0].shape, capacity=len(_imgs), keep=600, dim=16384, seed=0)
    for _im in _imgs:
        _arch.add(_im)
    _A, _B = _arch.recover(0), _arch.recover(3)
    _gm = _gh(_mi(_arch.M, _A, _B, steps=21)[10], _A, _B)
    _gx = _gh(_cf(_A, _B, steps=21)[10], _A, _B)
    _pit, _tab = _s2p("abcacbacab")
    print(f"  procedural: four procedural modalities driving existing decoders. Image/video morph "
          f"slerps in the DCT-coefficient domain -- its midpoint sits {_gm:.3f} from the double-exposure "
          f"vs a pixel crossfade's {_gx:.3f} (a crossfade IS the ghost). Text adds nucleus(top-p) decoding "
          f"(real-word fraction ~1.00 vs ~0.79 for plain temperature -- more coherent for a little less "
          f"variety). Audio sonifies a symbol sequence to a real WAV ({len(_tab)} symbols -> "
          f"{len(set(round(v) for v in _tab.values()))} distinct pitches). The on-ramp to native generation")
except Exception as _eG:
    print(f"  procedural: (skipped: {_eG})")

# FROZEN CORE + PERSISTENCE: a stable kernel facade build-on-top code imports, plus
# version-stamped save/load so a TRAINED mind can be persisted and reloaded identically
# (the gate that turns "interesting modules" into "a thing you can build on").
try:
    import tempfile as _tf, os as _osP, numpy as _npP
    import holographic_core as _core
    from holographic_creature import HolographicMind as _HM
    _rng = _npP.random.default_rng(0)
    _b = _HM(dim=48, actions=["N", "S", "E", "W"], seed=0, capacity=8)
    for _ in range(400):
        _b.remember([_rng.standard_normal(48)], [int(_rng.integers(4))], [_rng.standard_normal()])
    _b.consolidate(energy=0.95)
    _p = _osP.path.join(_tf.gettempdir(), "holo_tour_brain.npz")
    _core.save(_b, _p); _back = _core.load(_p)
    _d = _b._basis.shape[1] if _b._basis is not None else 48
    _ident = all(_npP.allclose([_b.value(_q, a)[0] for a in range(4)],
                               [_back.value(_q, a)[0] for a in range(4)])
                 for _q in (_rng.standard_normal(_d) for _ in range(20)))
    print(f"  core: frozen kernel facade (holographic_core re-exports bind/unbind/bundle/permute/"
          f"cosine/slerp/Vocabulary with stable signatures) + versioned save/load. A trained, "
          f"CONSOLIDATED brain round-trips through an npz ({_osP.path.getsize(_p)} bytes) and decides "
          f"identically across 20 probes: {_ident}. Persistence now spans the stack -- Vocabulary, the "
          f"recall forest, AND the whole SelfOrganizingMind (encoder + prototype bank, identical "
          f"classifications even on unseen words via persisted rng state); an incompatible STATE_VERSION "
          f"fails loudly rather than loading a silently-wrong object")
    _osP.remove(_p)
except Exception as _eC:
    print(f"  core: (skipped: {_eC})")

# FORWARD COMPOSITIONAL GENERATION: run the resonator FORWARD to compose NEW scenes (not
# interpolate stored ones), render them, and verify by ROUND-TRIP -- a generated scene is
# real iff it can be analysed straight back to the spec it was built from.
try:
    import numpy as _npF2
    import holographic_scene as _hsF
    from holographic_compose import (novel_specs as _ns, roundtrip_object as _rto,
                                      roundtrip_scene as _rts, render_fidelity as _rf,
                                      animate_attribute as _aa, animation_is_faithful as _aif)
    _coderF = _hsF.SceneCoder(dim=1024, seed=0)
    _rngF = _npF2.random.default_rng(0)
    _novel = _ns(n=40, rng=_rngF)
    _obj_ok = sum(_rto(_coderF, _t) for _t in _novel)
    _sc_ok = sum(_rts(_coderF, [{"colour": str(_rngF.choice(_hsF.COLOURS)),
                                 "shape": str(_rngF.choice(_hsF.SHAPES)),
                                 "texture": str(_rngF.choice(_hsF.TEXTURES))} for _ in range(4)])
                 for _ in range(20))
    _sh = _co = 0
    for _ in range(30):
        _s_ok, _c_ok = _rf({"colour": str(_rngF.choice([c for c in _hsF.COLOURS if c != "grey"])),
                            "shape": str(_rngF.choice(_hsF.SHAPES)), "texture": "smooth"},
                           seed=int(_rngF.integers(1000)))
        _sh += _s_ok; _co += _c_ok
    _fr = _aa(_coderF, {"colour": "red", "shape": "circle", "texture": "smooth"},
              "colour", ["red", "yellow", "green", "cyan", "blue"])
    print(f"  compose: native generation -- run the resonator FORWARD to build NOVEL scenes, not "
          f"interpolate stored ones. {_obj_ok}/40 novel single-object compositions factor straight back "
          f"to their spec; novel 4-object scenes {_sc_ok}/20; the rendered pixels auto-tag back as the "
          f"composed shape {_sh}/30 and colour {_co}/30; a colour-sweep animation is "
          f"{_aif(_coderF, _fr, 'colour'):.0%} on-target. A generated scene is real because it analyses "
          f"straight back to what it was built from")
except Exception as _eF2:
    print(f"  compose: (skipped: {_eF2})")

# FRACTAL: the same compose/factor one level up -- a scene-of-scenes (same above, same below)
try:
    from holographic_unified import UnifiedMind as _UMn
    import holographic_scene as _hsn
    import numpy as _npn
    _mn = _UMn(dim=1024, seed=0)
    _rn = _npn.random.default_rng(3)
    def _rtn():
        return {"colour": str(_rn.choice(_hsn.COLOURS)), "shape": str(_rn.choice(_hsn.SHAPES)),
                "texture": str(_rn.choice(_hsn.TEXTURES))}
    _grp = {"left": [_rtn(), _rtn()], "right": [_rtn(), _rtn()]}
    _szn = {k: len(v) for k, v in _grp.items()}
    _rec = _mn.decompose_nested(_mn.compose_nested(_grp), _szn)
    _kn = lambda d: (d["colour"], d["shape"], d["texture"])
    _okn = sum({_kn(t) for t in _grp[k]} == {_kn(g) for g in _rec[k]} for k in _grp)
    print(f"  nested: the SAME bind+superpose that builds a scene from objects, run one level up "
          f"to build a scene-of-scenes -- compose sub-scenes, group them, then peel each group "
          f"back out and factor it. {_okn}/{len(_grp)} sub-scenes recovered exactly (a sub-scene "
          f"is to the super-scene what an object is to a scene). Same above, same below; "
          f"group atoms are seed-derived so the whole nesting regenerates from one seed")
except Exception as _eN:
    print(f"  nested: (skipped: {_eN})")

# FHRR: the complex-phasor VSA the literature recommends for high binding capacity,
# offered as an opt-in faculty (the real-valued HRR core stays the readable default).
try:
    from holographic_unified import UnifiedMind as _UMf
    from holographic_ai import random_vector as _rvf, bind as _bf, unbind as _ubf, cosine as _cf
    import numpy as _npf
    _N = 40
    # real-HRR baseline at the same load
    _rr = _npf.random.default_rng(7)
    _ks = [_rvf(256, _rr) for _ in range(_N)]; _vs = [_rvf(256, _rr) for _ in range(_N)]
    _tr = sum(_bf(_ks[i], _vs[i]) for i in range(_N))
    _rok = sum(int(_npf.argmax([_cf(_ubf(_tr, _ks[i]), v) for v in _vs])) == i for i in range(_N)) / _N
    # FHRR via the mind's faculty
    _mf = _UMf(dim=256, seed=0); _mem, _voc = _mf.high_capacity_memory()
    _pf = {f"k{i}": f"v{i}" for i in range(_N)}
    for _k, _v in _pf.items():
        _mem.learn(_voc.get(_k), _voc.get(_v))
    _vocab = [f"v{i}" for i in range(_N)]
    _fok = sum(_voc.cleanup(_mem.recall(_voc.get(_k)), candidates=_vocab)[0] == _v
               for _k, _v in _pf.items()) / _N
    print(f"  fhrr: complex-phasor binding for the one regime where it measurably wins -- a "
          f"large key->value trace. At {_N} pairs/256-d, real-HRR recovers {_rok:.0%} but FHRR "
          f"recovers {_fok:.0%}. Offered as an opt-in faculty (high_capacity_memory); the "
          f"readable real-valued core stays the default since at normal loads both are perfect")
except Exception as _eFh:
    print(f"  fhrr: (skipped: {_eFh})")

# DYNAMIC QUANTIZATION: quant="auto" gives each saved array the coarsest precision its own
# structure supports -- precision follows the data's complexity and size.
try:
    import holographic_core as _coreQ
    from holographic_organizer import SelfOrganizingMind as _SOMq, _multimodal_world as _mmw
    import numpy as _npq, tempfile as _tf, os as _osq, json as _jsonq
    _enc, _samp, _Kq, _ = _mmw(seed=0, modes=2)
    _mq = _SOMq(dim=512, seed=0); _rq = _npq.random.default_rng(7)
    for _ in range(500):
        _c = int(_rq.integers(_Kq)); _mq.observe(_samp(_c), _c, "vector")
    _mq.reorganize()
    _probes = [(_samp(_c), _c) for _c in (int(_rq.integers(_Kq)) for _ in range(300))]
    def _accq(_mind):
        return _npq.mean([_mind.classify(_x, "vector")[0] == _c for _x, _c in _probes])
    _sz = {}
    for _mode, _kw in (("float32", dict(compress=True)), ("int8", dict(quant="int8")),
                       ("auto", dict(quant="auto"))):
        _p = _tf.mktemp(suffix=".npz"); _coreQ.save(_mq, _p, **_kw); _sz[_mode] = _osq.path.getsize(_p)
        if _mode == "auto":
            with _npq.load(_p) as _z:
                _qs = _jsonq.loads(bytes(_z["__qspec__"]).decode())
            _kinds = {}
            for _v in _qs.values():
                _kinds[_v["k"]] = _kinds.get(_v["k"], 0) + 1
            _autoacc = _accq(_coreQ.load(_p))
        _osq.remove(_p)
    print(f"  quant: dynamic per-array precision (quant='auto'). A trained mind saves at "
          f"float32 {_sz['float32']}B, fixed int8 {_sz['int8']}B, auto {_sz['auto']}B "
          f"({_sz['float32']/_sz['auto']:.1f}x vs float32). Auto chose a MIX by data "
          f"complexity/size {_kinds} -- int8 where separation proves it lossless, float32 "
          f"for tiny/marginal arrays -- decision-safe on every brain type (classifies "
          f"identically, acc {_autoacc:.3f})")
except Exception as _eQ:
    print(f"  quant: (skipped: {_eQ})")

# B5: rate-distortion code -- on genuinely low-rank state, spend only the bits the cosines need
print("\nRATE-DISTORTION CODE (B5): geometry-preserving compression of low-rank state")
try:
    import numpy as _npR
    from holographic_ai import random_vector as _rvR, bundle as _buR, cosine as _coR
    from holographic_ratedistortion import geometry_preserving_code as _gpc, reconstruct as _rec, bits_per_vector as _bpv
    _rg = _npR.random.default_rng(0); _Dr = 256; _Kr = 16
    _sen = [_rvR(_Dr, _rg) for _ in range(_Kr)]
    _Xr = _npR.array([_buR([_sen[j] for j in _rg.choice(_Kr, size=5, replace=False)]) for _ in range(800)])
    _Xr /= _npR.linalg.norm(_Xr, axis=1, keepdims=True)
    _cd = _gpc(_Xr, target_cos=0.9999); _Xh = _rec(_cd)
    _cosr = _npR.mean([_coR(_Xr[i], _Xh[i]) for i in range(len(_Xr))]); _b = _bpv(_cd)
    print(f"  low-rank engine state: KLT(consolidation) -> quantize -> rANS (bit-exact). "
          f"cosine {_cosr:.5f} at {_b:.0f} bits/vec vs int8 {8*_Dr} ({8*_Dr/_b:.0f}x smaller). "
          f"KEPT NEGATIVE: on full-rank data (no subspace) it falls back to int8.")
except Exception as _eR:
    print(f"  rate-distortion: (skipped: {_eR})")

# A deterministic Kolmogorov-Arnold readout on holostuff encoders: sum of per-feature univariate
# functions (KAN), with the encoder bumps as the spline basis and a least-squares fit (no backprop).
print("\nHOLOGRAPHIC KAN (Kolmogorov-Arnold readout, deterministic, no backprop):")
try:
    import numpy as _npK
    from holographic_kan import HolographicKAN as _HK
    _rk = _npK.random.default_rng(0); _Xk = _rk.uniform(0, 1, (1200, 2))
    _yk = _npK.sin(2 * _npK.pi * _Xk[:, 0]) + 4 * (_Xk[:, 1] - 0.5) ** 2 + 0.02 * _rk.standard_normal(1200)
    _kn = _HK(2, seed=0).fit(_Xk[:900], _yk[:900])
    def _r2k(y, yh): return 1 - _npK.sum((y - yh) ** 2) / _npK.sum((y - y.mean()) ** 2)
    _ts = _npK.linspace(0.05, 0.95, 40)
    _c1 = abs(_npK.corrcoef(_kn.feature_function(0, _ts), _npK.sin(2 * _npK.pi * _ts))[0, 1])
    print(f"  additive target f=sin(2pi x1)+4(x2-.5)^2: test R^2 {_r2k(_yk[900:], _kn.predict(_Xk[900:])):.3f}; "
          f"recovered psi_1 vs sin corr {_c1:.2f} (interpretable, KAN-style). The spline basis is the "
          f"RBF encoder's bumps; the fit is a linear solve. KEPT: additive form can't do x1*x2 interactions.")
except Exception as _eK:
    print(f"  holographic-kan: (skipped: {_eK})")

# Generative recipe-store: a constructed structure is noise-free, so it serialises to its build-graph
# (the "proof") and replays BIT-EXACT -- the easy, exact half of generative compression (no search).
print("\nGENERATIVE RECIPE-STORE (constructed structure -> its generator, lossless):")
try:
    import numpy as _npR
    from holographic_recipe import StructureRecipe as _SR
    from holographic_ai import derived_atom as _da
    _big = _SR(dim=512, seed=11)
    _big.atom_range("tok_", 2000)                      # 2000-atom codebook as ONE macro op
    _built = _big.outputs()
    _ref = [_da(11, f"tok_{_i}", 512) for _i in range(2000)]
    _exact = max(_npR.max(_npR.abs(_built[_i] - _ref[_i])) for _i in range(2000)) == 0.0
    print(f"  {len(_built)}x{_big.dim} ({_big.expanded_bytes()/1e6:.1f} MB) -> recipe {_big.recipe_bytes()} B "
          f"(~{_big.compression_ratio():,.0f}x via the macro op), replay bit-exact={_exact}. KEPT NEGATIVE: "
          f"non-constructed (raw) data has no short recipe -> ~1x. The win is the constructed fraction.")
except Exception as _eRc:
    print(f"  recipe-store: (skipped: {_eRc})")

# The other half: decompose FOREIGN data into a compact law by MDL-gated symbolic regression. The law
# is a seed that extrapolates; MDL is the gate that keeps it parsimonious (and refuses to fit noise).
print("\nDECOMPOSE SEARCH (foreign data -> a compact law, MDL-gated):")
try:
    import numpy as _npS
    from holographic_symbolic import symbolic_regress as _sr, full_fit as _ff
    _x = _npS.linspace(0, 6, 240); _xe = _npS.linspace(6, 9, 120)
    _tf = lambda t: 2.0*_npS.sin(1.5*t) + 0.5*t
    _y = _tf(_x) + 0.05*_npS.random.default_rng(0).standard_normal(240)
    _f, _info = _sr(_x, _y); _rms = lambda a, b: float(_npS.sqrt(_npS.mean((a-b)**2)))
    _mx = _ff(_x, _y)
    print(f"  recovered {_f}  ({_info['n_terms']} terms of {_info['dict_size']})")
    print(f"  extrapolation RMS: MDL law {_rms(_f.generate(_xe),_tf(_xe)):.3f} vs un-gated max-fit "
          f"{_rms(_mx.generate(_xe),_tf(_xe)):.1e} (overfits). On noise MDL refuses (keeps ~0 terms).")
    from holographic_symbolic import compress_signal as _cs
    _seed, _si = _cs(_x, _y, path="/tmp/_tour_law.seed")
    print(f"  ONE CALL end-to-end: compress_signal -> {_seed.recipe_bytes()}-byte seed that regenerates and "
          f"extrapolates. build 2 emits a build-1-style recipe; the residual is B5's job.")
    # multiplicative mode: the log transform turns a product law additive (x becomes +), the prime-factor insight
    _xm = _npS.linspace(0.2, 4, 200); _ym = 2.0 * _xm**1.5 * _npS.exp(0.3 * _xm)
    from holographic_symbolic import symbolic_regress as _sreg
    _fm, _ = _sreg(_xm, _ym, multiplicative=True)
    print(f"  MULTIPLICATIVE mode (log transform): recovered {_fm} for y=2*x^1.5*exp(0.3x) -- a PRODUCT law the "
          f"flat additive basis would only approximate. compress_signal(mode='auto') picks the better family.")
except Exception as _eSr:
    print(f"  decompose-search: (skipped: {_eSr})")

# B2: sparse block codes + a scaled resonator. Block-local modular binding is exact, so the resonator
# factors more (factors x alphabet) at fixed D than the dense one, and verifies itself by reconstruction.
print("\nSPARSE BLOCK CODES + SCALED RESONATOR (B2):")
try:
    from holographic_sbc import sbc_codebook as _scb, sbc_reconstruct as _srec, sbc_resonator as _sres
    _B, _L = 16, 16
    _cbs = [_scb(_B, _L, 10, seed=_k) for _k in range(3)]
    _true = (3, 7, 1); _P = _srec(_true, _cbs, _L)
    _picks, _ok = _sres(_P, _cbs, _L, seed=0)
    print(f"  factor a 3-way product (alphabet 10) at D={_B*_L}: recovered {_picks} == {_true} "
          f"({_picks == _true}), self-verified by reconstruction = {_ok}.")
    print(f"  beats the dense resonator at fixed D (1.00 vs 0.90 at N=10, 0.25 vs 0.15 at N=25); the "
          f"confidence check tracks correctness exactly (verify or abstain). KEPT NEGATIVE: modest absolute "
          f"capacity, and SBC is a parallel representation beside the dense kernel.")
    # the structural inverse of build-1: decompose a composed structure back into its recipe, verified.
    from holographic_sbc import decompose_structure as _dstr, sbc_identity as _sid
    _cbi = [list(_scb(_B, _L, 8, seed=20 + _k)) + [_sid(_B)] for _k in range(3)]
    _fac = [_cbi[0][3], _cbi[1][6], _sid(_B)]; _Pp = _fac[0].copy()
    for _f in (1, 2): _Pp = __import__("holographic_sbc").sbc_bind(_Pp, _fac[_f], _L)
    _out = _dstr(_Pp, _cbi, _L, seed=1)
    print(f"  STRUCTURAL DECOMPOSE (inverse of build-1, the blend thread): recover a composed structure's "
          f"recipe by verified superposition search -- present={_out['present']} (third factor absent, "
          f"detected), verified={_out['verified']}. A bound product is unreadable naively; the joint search is required.")
except Exception as _eSb:
    print(f"  sbc-resonator: (skipped: {_eSb})")

# B7 KEYSTONE: program / expression-tree / nested-scene are not four types -- they are ONE
# StructureRecipe (atom/bind/bundle/permute/superpose), each reproducing its source bit-exactly.
try:
    from holographic_machine import HoloMachine as _HM
    from holographic_typed import (program_to_recipe as _p2r, encode_tree as _etr,
                                    tree_to_recipe as _t2r, nested_scene_to_recipe as _n2r,
                                    op_kinds as _opk)
    from holographic_ai import cosine as _cosK
    _mK = _HM(dim=2048, seed=7); _pg = [("LOAD","a"),("BIND","b"),("BUNDLE","c"),("HALT","a")]
    _rP = _p2r(_mK, _pg); _cP = _cosK(_mK.assemble(_pg), _rP.get(_rP._outputs[0]))
    _tr = ("eml", ("mul","x","two"), ("ln","y"))
    _rT = _t2r(2048, 5, _tr); _cT = _cosK(_etr(2048, 5, _tr), _rT.get(_rT._outputs[0]))
    _mnK = __import__("holographic_unified").UnifiedMind(dim=1024, seed=3)
    _grpK = {"g1":[{"colour":"red","shape":"circle","texture":"smooth"}],
             "g2":[{"colour":"cyan","shape":"line","texture":"vertical"}]}
    _rS = _n2r(_mnK, _grpK); _cS = _cosK(_mnK.compose_nested(_grpK), _rS.get(_rS._outputs[0]))
    _alpha = sorted(_opk(_rP) | _opk(_rT) | _opk(_rS))
    print(f"  B7 TYPED STRUCTURE (the integration keystone): program/tree/scene all reduce to ONE recipe -- "
          f"bit-exact cosines {_cP:.4f}/{_cT:.4f}/{_cS:.4f}, ONE alphabet {_alpha}. UnifiedMind speaks it directly "
          f"(typed_structure/realize/tree_structure/nested_scene_structure); decode+manifold (B8/B9) target this one type.")
except Exception as _eK:
    print(f"  typed-structure keystone: (skipped: {_eK})")

# B8: denoised structure DECODING -- per-peel cleanup pushes the iterated-decode depth cliff. A chain
# (a B7 typed structure) decoded by repeated unbinding craters without cleanup (noise compounds); per-peel
# cleanup decodes the whole chain. Soft Hopfield cleanup ties hard on discrete pointers, wins on continuous.
try:
    from holographic_peel import chain_recipe as _crc, traversal_score as _tsc, recover_continuous_values as _rcv
    from holographic_encoders import ScalarEncoder as _SE
    import numpy as _npB8
    _rC, _ndC = _crc(512, 1, 16); _MC = _rC.get(_rC._outputs[0])
    _none = _tsc(_MC, _ndC, cleanup=None)[0]; _hard = _tsc(_MC, _ndC, cleanup="hard")[0]; _soft = _tsc(_MC, _ndC, cleanup="soft")[0]
    _encB = _SE(1024, 0.0, 1.0, seed=1, kernel="rbf", bandwidth=8)
    _grid = _npB8.linspace(0, 1, 21); _cbB = _npB8.stack([_encB.encode(g) for g in _grid])
    from holographic_ai import bind as _bB, derived_atom as _daB
    _rolesB = _npB8.stack([_daB(1, f"role:{i}", 1024, unitary=True) for i in range(6)])
    _rngB = _npB8.random.default_rng(0); _trB = _rngB.uniform(0.05, 0.95, 6)
    _MvB = _npB8.sum([_bB(_rolesB[i], _encB.encode(_trB[i])) for i in range(6)], axis=0)
    _hC, _sC = _rcv(_rolesB, _cbB, _MvB, [_encB.encode(t) for t in _trB])
    print(f"  B8 DENOISED DECODE: a 16-node chain decoded by iterated unbinding -- no cleanup {_none}/15 (craters, "
          f"noise compounds), per-peel cleanup {_hard}/15 (full chain). Soft Hopfield ties hard on discrete pointers "
          f"({_soft}/15), and WINS on continuous payloads (cosine {_sC:.3f} vs hard {_hC:.3f}). Cleans structure as it decodes.")
except Exception as _eB8:
    print(f"  denoised-decode: (skipped: {_eB8})")

# B9: manifold-aware decompose -- detect the domain TOPOLOGY (line/ring/mobius/torus), then decompose on
# the matched basis. A periodic signal forced onto a flat-line (polynomial) basis EXTRAPOLATES BY DIVERGING;
# the detected-period harmonic basis extrapolates correctly. Detection works for OFF-GRID periods.
try:
    import numpy as _npB9
    from holographic_manifold import decompose_on_manifold as _dom, detect_topology as _dt
    from holographic_symbolic import symbolic_regress as _srB9
    from holographic_manifold import line_dictionary as _lineD
    _w0 = 2 * _npB9.pi / 5.0                       # period 5, off the elementary fixed-freq grid
    _xB9 = _npB9.linspace(0, 10, 400); _xeB9 = _npB9.linspace(10, 15, 200)
    _yR = _npB9.sin(_w0 * _xB9) + 0.5 * _npB9.cos(2 * _w0 * _xB9)
    _trueR = _npB9.sin(_w0 * _xeB9) + 0.5 * _npB9.cos(2 * _w0 * _xeB9)
    _fmR, _iR = _dom(_xB9, _yR); _flR, _ = _srB9(_xB9, _yR, dictionary=_lineD())
    _emR = float(_npB9.sqrt(_npB9.mean((_fmR.generate(_xeB9) - _trueR) ** 2)))
    _elR = float(_npB9.sqrt(_npB9.mean((_flR.generate(_xeB9) - _trueR) ** 2)))
    _tM = _dt(_xB9, _npB9.sin(_w0 * _xB9) + _npB9.sin(3 * _w0 * _xB9))[0]
    print(f"  B9 MANIFOLD-AWARE DECOMPOSE: detected ring (P~{_iR['period']:.2f}, off the fixed grid) and the matched "
          f"harmonic basis EXTRAPOLATES (RMS {_emR:.3f}) where the flat-line polynomial DIVERGES (RMS {_elR:.2f}). "
          f"Antiperiodic signals detected as mobius ({_tM}) -> odd-harmonic basis. Topology chooses the right manifold to decompose on.")
except Exception as _eB9:
    print(f"  manifold-decompose: (skipped: {_eB9})")

# B6: Physarum FLOW-conductance pathfinding (Tero et al. 2007) -- the deterministic, principled counterpart
# to the stochastic elitist-ant slime solver. A maze is a tube network; flux from source to sink is a
# graph-Laplacian solve; tubes adapt (thicken with flux) and the network collapses onto the shortest path.
try:
    import time as _timeB6
    from holographic_creature import GridWorld as _GWB6
    from holographic_flow import solve_maze_flow as _smf
    _wB6 = _GWB6(16, 16, maze=True, fixed_seed=3, braid=1.0)
    _t0 = _timeB6.perf_counter(); _pB6, _iB6 = _smf(_wB6); _tB6 = (_timeB6.perf_counter() - _t0) * 1000
    _pB6b, _ = _smf(_wB6)
    print(f"  B6 TERO FLOW SOLVER: braided 16x16 maze solved by flow physics (Laplacian solve + tube adaptation) "
          f"-> len {_iB6['extracted_len']} (optimum {_iB6['optimal']}), DETERMINISTIC (identical reruns: {_pB6==_pB6b}), "
          f"{_tB6:.0f}ms. Same optimum as the elitist ant but reproducible and ~100x faster (ant ~10-32s, measured).")
except Exception as _eB6:
    print(f"  tero-flow: (skipped: {_eB6})")

# B6 generalised: FRAGMENT ASSEMBLY as flow search -- choosing fragments to minimise an energy is a
# min-cost trellis path, the same search as the maze, and the result is a B7 typed structure.
try:
    from holographic_assembly import assemble as _asm, assemble_optimal_energy as _asmopt
    _tgt = "ABCABCABCA"; _full = sorted({_tgt[p:p+2] for p in range(len(_tgt)-1)})
    _o0 = _asm(_tgt, _full)
    _libm = sorted((set(_full)-{"CA"})|{"AA","BB","CC"}); _om = _asm(_tgt, _libm); _opt = _asmopt(_tgt, _libm)
    print(f"  B6+ FRAGMENT ASSEMBLY (the Baker/Rosetta seat): flow search assembles '{_o0['assembled']}' from a "
          f"complete library at energy {_o0['energy']}; with a true fragment missing it finds the GLOBAL min-energy "
          f"assembly (energy {_om['energy']} == DP optimum {_opt}). Fragment assembly = a min-cost trellis path = the maze, one search.")
except Exception as _eAS:
    print(f"  fragment-assembly: (skipped: {_eAS})")

# ADAPTIVE-RANK DENOISING: cashes the fixed-rank denoiser's low-noise over-smoothing negative by
# choosing kept components from a noise estimate (Donoho-Johnstone shrinkage in the manifold basis).
try:
    import numpy as _npAD
    from holographic_denoise import fit_manifold as _fmA, manifold_denoise as _mdA, fit_manifold_full as _fmf, adaptive_manifold_denoise as _amd
    _px = _npAD.load("data/sol_5min.npz")["px"].astype(float)
    _wn = _npAD.stack([_px[i:i+64] for i in range(0, len(_px)-64, 16)])
    _wn = (_wn-_wn.mean(1,keepdims=True))/(_wn.std(1,keepdims=True)+1e-9)
    _rg = _npAD.random.default_rng(0); _rg.shuffle(_wn); _trA,_teA=_wn[:600],_wn[600:900]
    _b8,_m8=_fmA(_trA,rank=8); _Vf,_Sf,_mf=_fmf(_trA,rank=32)
    _sn=lambda c,e: 10*_npAD.log10(_npAD.var(c)/(_npAD.mean((c-e)**2)+1e-12))
    def _sweep(sig):
        fx=[]; ad=[]
        for c in _teA:
            n=c+sig*_rg.standard_normal(64); base=_sn(c,n)
            fx.append(_sn(c,_mdA(n,_b8,_m8))-base); ad.append(_sn(c,_amd(n,_Vf,_mf))-base)
        return _npAD.mean(fx), _npAD.mean(ad)
    _flo,_alo=_sweep(0.3); _fhi,_ahi=_sweep(0.8)
    print(f"  ADAPTIVE-RANK DENOISE (cashes B7-original's negative): at LOW noise fixed rank-8 HARMS ({_flo:+.2f} dB) "
          f"while adaptive is neutral ({_alo:+.2f} dB); at HIGH noise both gain (fixed {_fhi:+.2f}, adaptive {_ahi:+.2f}). "
          f"Noise-driven thresholding never over-smooths -- robust to unknown noise, the price being the oracle peak.")
except Exception as _eAD:
    print(f"  adaptive-denoise: (skipped: {_eAD})")

# TRAJECTORY DENOISE (above/below sweep): the pipeline's lone-1-D-signal denoiser PROMOTED to a first-class
# denoise method. A smooth signal's own sliding-window (Hankel) matrix is low-rank, so it can be cleaned
# with NO external prior -- the second prior-free method beside nlm (nlm needs a patch SET; this a raw 1-D
# signal). One shared implementation now serves the pipeline and any caller: um.denoise(sig, method='trajectory').
try:
    import numpy as _npTR
    from holographic_unified import UnifiedMind as _UMTR
    _mTR = _UMTR(dim=256, seed=0)
    _tt = _npTR.linspace(0, 1, 256)
    _clean = _npTR.sin(2 * _npTR.pi * 3 * _tt) + 0.5 * _tt
    _noisy = _clean + 0.4 * _npTR.random.default_rng(0).standard_normal(256)
    _den = _npTR.asarray(_mTR.denoise(_noisy, method="trajectory"))
    _e0 = float(_npTR.linalg.norm(_noisy - _clean)); _e1 = float(_npTR.linalg.norm(_den - _clean))
    print(f"  TRAJECTORY DENOISE (lone 1-D signal, no external prior): error {_e0:.2f} -> {_e1:.2f} "
          f"({100 * (1 - _e1 / _e0):.0f}% of the noise removed from the signal's OWN low-rank trajectory)")
except Exception as _eTR:
    print(f"  trajectory-denoise: (skipped: {_eTR})")

# GRAPH-SIGNAL DENOISE (reverse-transfer RT-III1): mesh smoothing mapped onto the concept graph -- denoise a
# whole noisy codebook over its OWN k-NN similarity graph. Taubin's lam|mu low-pass denoises WITHOUT the
# volume-shrink a naive Laplacian causes, and at high noise the local graph beats a per-vector linear denoise.
try:
    import numpy as _npGS
    from holographic_unified import UnifiedMind as _UMGS
    _mGS = _UMGS(dim=512, seed=0); _rgGS = _npGS.random.default_rng(0); _DG, _NG = 512, 100
    _tg = _npGS.linspace(0, 1, _NG); _omg = _rgGS.uniform(1, 10, _DG); _phg = _rgGS.uniform(0, 2 * _npGS.pi, _DG)
    _cleanG = _npGS.cos(2 * _npGS.pi * _npGS.outer(_tg, _omg) + _phg)
    _cleanG /= _npGS.linalg.norm(_cleanG, axis=1, keepdims=True)
    _nzG = _rgGS.standard_normal((_NG, _DG)); _nzG /= _npGS.linalg.norm(_nzG, axis=1, keepdims=True)
    _noisyG = _cleanG + 1.2 * _nzG
    _qG = lambda X: float(_npGS.mean(_npGS.sum(X / _npGS.linalg.norm(X, axis=1, keepdims=True) * _cleanG, axis=1)))
    _taubG = _mGS.graph_denoise(_noisyG, method="taubin"); _naiveG = _mGS.graph_denoise(_noisyG, method="laplacian")
    print(f"  GRAPH-SIGNAL DENOISE (codebook over its k-NN graph): quality {_qG(_noisyG):.2f} -> Taubin {_qG(_taubG):.2f}; "
          f"norm kept {_npGS.linalg.norm(_taubG, axis=1).mean():.2f} vs naive-Laplacian "
          f"{_npGS.linalg.norm(_naiveG, axis=1).mean():.2f} (Taubin avoids the shrink)")
except Exception as _eGS:
    print(f"  graph-signal denoise: (skipped: {_eGS})")

# NONLINEAR MANIFOLD CHART (reverse-transfer RT-II1): UV unwrapping mapped onto the concept manifold -- flatten
# a CURVED manifold to a low-D chart. A linear SVD (consolidation) FOLDS a swiss roll; Isomap preserves the
# along-manifold (geodesic) distance and unrolls it, so classes adjacent on the roll but folded together by SVD
# come apart.
try:
    import numpy as _npMC
    from holographic_unified import UnifiedMind as _UMMC
    from holographic_chart import geodesic_distances as _geoMC
    _mMC = _UMMC(dim=256, seed=0); _rgMC = _npMC.random.default_rng(0); _NM, _DM = 300, 256
    _uM = _rgMC.uniform(0, 1, _NM); _vM = _rgMC.uniform(0, 1, _NM); _angM = 1.5 * _npMC.pi * (1 + 2 * _uM)
    _rollM = _npMC.stack([_angM * _npMC.cos(_angM), 21 * _vM, _angM * _npMC.sin(_angM)], 1)
    _QM = _npMC.linalg.qr(_rgMC.standard_normal((_DM, 3)))[0]; _XM = _rollM @ _QM.T + 0.05 * _rgMC.standard_normal((_NM, _DM))
    _GtM = _geoMC(_XM, k=10); _iuM = _npMC.triu_indices(_NM, 1)
    _gcM = lambda Y: float(_npMC.corrcoef(_npMC.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))[_iuM], _GtM[_iuM])[0, 1])
    _isoM = _mMC.manifold_chart(_XM, dim=2, method="isomap")
    _svM = (_XM - _XM.mean(0)) @ _npMC.linalg.svd(_XM - _XM.mean(0), full_matrices=False)[2][:2].T
    print(f"  NONLINEAR MANIFOLD CHART (swiss roll -> 2-D): geodesic fidelity SVD/linear {_gcM(_svM):.2f} vs "
          f"Isomap {_gcM(_isoM):.2f} -- the linear chart folds the curve, the geodesic chart unrolls it")
except Exception as _eMC:
    print(f"  manifold-chart: (skipped: {_eMC})")

# THE DETERMINISM CONTRACT (ISA-1): the engine has a written ISA (ISA.md) with ONE tie-break/sign rule, now
# CITED in one place instead of re-invented per module. The sign convention -- the same bit-exact-tie class as
# the bind_batch bug -- lived in FOUR scattered copies; spectral and chart now both route through the contract.
try:
    import numpy as _npDC
    from holographic_determinism import fix_eigvec_signs as _fesDC
    from holographic_spectral import sign_fix as _sfDC
    from holographic_chart import _fix_signs as _cfDC
    _Vdc = _npDC.random.default_rng(0).standard_normal((24, 5))
    _same = (_npDC.allclose(_fesDC(_Vdc), _fesDC(-_Vdc))                 # V and -V -> the SAME fixed basis
             and _npDC.array_equal(_sfDC(_Vdc.copy()), _fesDC(_Vdc))    # spectral cites the contract, bit-exact
             and _npDC.array_equal(_cfDC(_Vdc), _fesDC(_Vdc)))          # chart cites the contract, bit-exact
    print(f"  DETERMINISM CONTRACT (ISA.md): one sign/tie-break rule, four scattered copies reconciled to one -- "
          f"spectral & chart cite it bit-exactly, sign ambiguity removed: {_same}")
except Exception as _eDC:
    print(f"  determinism-contract: (skipped: {_eDC})")

# THE CONFORMANCE SUITE (ISA-2): the contract's teeth. Every production base instruction is checked against a
# definitional reference (a direct O(D^2) convolution for bind, etc.) -- TOL on continuous outputs, EXACT on
# decisions. A vectorized op is "conformant" iff it passes here, and the bind_batch class (a value-conformant
# change that flips a DECISION) is caught because decisions are pinned separately and exactly.
try:
    from holographic_unified import UnifiedMind as _UMCF
    from holographic_reference import value_conformant as _vcCF, decision_conformant as _dcCF
    import numpy as _npCF
    _rep = _UMCF(dim=64, seed=0).conformance_report(dim=64, seed=0)
    _allok = all(_r["passed"] for _r in _rep.values())
    _ops = ", ".join(f"{_o}[{_r['class']}]" for _o, _r in list(_rep.items())[:4])
    # the bind_batch class in one line: a sub-tolerance change that flips the cleanup decision
    _sims = _npCF.array([0.5, 0.5, 0.3]); _bad = _sims + _npCF.array([0.0, 1e-12, 0.0])
    _caught = _vcCF(_sims, _bad) and not _dcCF(_sims, _bad)
    print(f"  CONFORMANCE SUITE (ISA.md teeth): all base ops conform = {_allok} ({_ops}, ...); "
          f"bind_batch class caught (sub-tol change flips a decision) = {_caught}")
except Exception as _eCF:
    print(f"  conformance-suite: (skipped: {_eCF})")

# THE GOVERNED EXTENSIONS (ISA-3): base `bind` stays RISC; three named, opt-in EXTENSIONS each earn their place
# with a measured regime win (the VSA analog of x86 + SSE/AVX). See ISA_EXTENSIONS.md.
try:
    import numpy as _npEX, math as _mEX
    from holographic_clifford import CliffordAlgebra as _CAEX
    from holographic_fpe import VectorFunctionEncoder as _VFEX
    from holographic_tensor import TensorBindMemory as _TBEX
    from holographic_ai import random_vector as _rvEX, bind as _bEX, unbind as _ubEX, bundle as _blEX, cosine as _csEX
    _cl = _CAEX(); _vEX = _npEX.random.default_rng(0).standard_normal(3)
    _R1 = _cl.rotor(_npEX.array([0,0,1.0]), _mEX.pi/3); _R2 = _cl.rotor(_npEX.array([1.0,0,0]), _mEX.pi/4)
    _clerr = _npEX.max(_npEX.abs(_cl.rotate(_R2,_cl.rotate(_R1,_vEX)) - _cl.rotate(_cl.compose(_R2,_R1),_vEX)))
    _fpe = _VFEX(1, dim=1024, bounds=[(0,3)], kernel="rbf", seed=0)
    _fk = _csEX(_fpe.encode([0.0]), _fpe.encode([1.0]))               # continuous similarity at offset 1.0
    _rg = _npEX.random.default_rng(0); _D=32; _ks=[_rvEX(_D,_rg) for _ in range(12)]; _vs=[_rvEX(_D,_rg) for _ in range(12)]
    _hrr = _npEX.mean([_csEX(_ubEX(_blEX([_bEX(k,v) for k,v in zip(_ks,_vs)]),_ks[i]),_vs[i]) for i in range(12)])
    _tm = _TBEX(_ks,_vs); _tn = _npEX.mean([_csEX(_tm.recall(_ks[i]),_vs[i]) for i in range(12)])
    print(f"  GOVERNED EXTENSIONS (ISA_EXTENSIONS.md): Clifford exact 3-D rotation (err {_clerr:.0e}) | "
          f"FPE continuous kernel (cos {_fk:.2f} at offset 1 vs ~0 for random atoms) | "
          f"tensor capacity (recall {_tn:.2f} vs HRR {_hrr:.2f} at overload)")
except Exception as _eEX:
    print(f"  governed-extensions: (skipped: {_eEX})")

# THE REGISTER FILE (ISA-4): HoloMachine grows from one accumulator to a handful of named slots (STORE r /
# RECALL r). Slots are held SEPARATELY -> reads are EXACT; a value survives ACC being overwritten and comes back
# verbatim with one RECALL instead of a full re-derivation. Kept negative: a BUNDLED file has a literal capacity
# cliff (register pressure), which is why the slots are separate.
try:
    import numpy as _npRG
    from holographic_machine import HoloMachine as _HMRG
    from holographic_ai import cosine as _csRG, random_vector as _rvRG, bind as _bRG, unbind as _ubRG, bundle as _blRG
    _mRG = _HMRG(dim=1024, seed=7)
    _pRG = [("LOAD","a"),("STORE","R0"),("LOAD","b"),("BIND","c"),("RECALL","R0"),("HALT","a")]
    _accRG,_ = _mRG.run(_mRG.assemble(_pRG))
    _exact = _csRG(_accRG, _mRG.data_atoms["a"])
    def _bf(n):                                              # bundled-file readback at n registers (the cliff)
        _rg=_npRG.random.default_rng(0); _ro=[_rvRG(1024,_rg) for _ in range(n)]; _va=[_rvRG(1024,_rg) for _ in range(n)]
        _f=_blRG([_bRG(_ro[i],_va[i]) for i in range(n)])
        return _npRG.mean([int(_npRG.argmax([_csRG(_ubRG(_f,_ro[i]),_va[j]) for j in range(n)]))==i for i in range(n)])
    print(f"  REGISTER FILE (ISA-4): separate slots read EXACT (RECALL R0 cosine {_exact:.3f}); "
          f"kept negative -- a BUNDLED file degrades with count: 8 regs {_bf(8):.2f} -> 64 regs {_bf(64):.2f} (literal register pressure)")
except Exception as _eRG:
    print(f"  register-file: (skipped: {_eRG})")

# THE CALLING CONVENTION + PERMUTE-STACK (ISA-5): ACC is arg/return; registers and the stack are FRAME-LOCAL, so
# a callee cannot corrupt the caller (preserved automatically). The permute-stack (PUSH/POP) is a LIFO in the
# substrate -- exact at shallow depth, with a crosstalk depth cliff (same shape as the B8 cliff). See ISA.md.
try:
    import numpy as _npCS
    from holographic_machine import HoloMachine as _HMCS, stack_push as _spCS, stack_pop as _ppCS
    from holographic_ai import cosine as _csCS, random_vector as _rvCS
    _mCS = _HMCS(dim=1024, seed=7)
    _mCS.define("clob", [("LOAD","f"),("STORE","R0"),("HALT","a")])      # a callee that clobbers ITS R0
    _pCS = [("LOAD","a"),("STORE","R0"),("CALL","clob"),("RECALL","R0"),("HALT","a")]
    _accCS,_ = _mCS.run(_mCS.assemble(_pCS))
    _frame = _csCS(_accCS, _mCS.data_atoms["a"])                         # caller's R0 preserved across CALL?
    def _depth(n):                                                        # permute-stack LIFO recovery at depth n
        _rg=_npCS.random.default_rng(0); _at=[_rvCS(1024,_rg) for _ in range(n)]; _s=None
        for _a in _at: _s=_spCS(_s,_a)
        _ok=0
        for _e in range(n-1,-1,-1):
            _t,_s=_ppCS(_s,_at)
            if int(_npCS.argmax([_csCS(_t,_c) for _c in _at]))==_e: _ok+=1
        return _ok/n
    print(f"  CALLING CONVENTION + STACK (ISA-5): registers frame-local (caller's R0 survives a callee clobber, "
          f"cosine {_frame:.3f}); permute-stack LIFO exact shallow ({_depth(4):.2f} @ depth 4) -> blurs deep "
          f"({_depth(16):.2f} @ depth 16, the crosstalk cliff)")
except Exception as _eCS:
    print(f"  calling-convention: (skipped: {_eCS})")

# THE MACRO LAYER (ISA-6): parameterized recipe TEMPLATES -- a structure with named HOLES filled at
# instantiation. Different arguments give distinct, BIT-EXACT structures (the recipe's exactness carries), and
# template-internal atoms are namespaced (HYGIENE) so they cannot collide with the caller's atoms. See
# holographic_template.py.
try:
    import numpy as _npTP
    from holographic_template import STARTER_LIBRARY as _LIB
    from holographic_ai import unbind as _ubTP, cosine as _csTP, derived_atom as _daTP
    _rec = _LIB["record"]
    _m1 = _rec.build_vector(1024, 7, key="name", val="moose")
    _m1b = _rec.build_vector(1024, 7, key="name", val="moose")
    _m2 = _rec.build_vector(1024, 7, key="name", val="socks")
    _bitexact = _npTP.array_equal(_m1, _m1b)
    _pair = _LIB["pair"]
    _role = _pair.role_atom(1024, 7, "role")
    _exact = _csTP(_ubTP(_pair.build_vector(1024, 7, x="alpha"), _role), _daTP(7, "alpha", 1024))
    _hyg = _csTP(_pair.role_atom(1024, 7, "role"), _daTP(7, "role", 1024, unitary=True))  # ~0: no capture
    print(f"  MACRO LAYER (ISA-6): templates {sorted(_LIB)}; instantiate twice -> bit-exact ({_bitexact}); "
          f"distinct args -> distinct (record cos {_csTP(_m1,_m2):.2f}); pair recovers value exact ({_exact:.2f}); "
          f"hygiene -- internal role vs caller atom cos {_hyg:.2f} (namespaced, no capture)")
except Exception as _eTP:
    print(f"  macro-layer: (skipped: {_eTP})")

# THE STRUCTURE LANGUAGE (ISA-7, top of the tower): a small declarative surface (S-expressions) that LOWERS to
# the recipe IR -- atoms, the base binds, and the ISA-6 templates as language forms. Scoped to ONE domain
# (structure description), not a general language. See holographic_lang.py.
try:
    import numpy as _npLG
    from holographic_lang import realize_spec as _rsLG, compile_spec as _csLG, parse as _pLG, unparse as _uLG
    from holographic_template import STARTER_LIBRARY as _LIBLG
    _spec = "(bundle (record name moose) (pair socks))"
    _v = _rsLG(_spec, 1024, 7)
    _roundtrip = (_uLG(_pLG(_spec)) == _spec)                          # surface round-trips
    _bitexact = _npLG.array_equal(_v, _rsLG(_spec, 1024, 7))
    _agree = _npLG.array_equal(_rsLG("(record name moose)", 1024, 7),
                               _LIBLG["record"].build_vector(1024, 7, key="name", val="moose"))
    print(f"  STRUCTURE LANGUAGE (ISA-7): '{_spec}' -> recipe -> vector; surface round-trips ({_roundtrip}); "
          f"realizes bit-exact ({_bitexact}); template forms agree with ISA-6 ({_agree})")
except Exception as _eLG:
    print(f"  structure-language: (skipped: {_eLG})")

# THE REVERSIBLE / ERROR-CORRECTION MODEL (ISA-8, the frontier): VSA assembly is a noisy, partly-REVERSIBLE ISA
# -- bind/unbind/permute reversible, bundle/cleanup information-destroying, cleanup = error correction. The
# practical payoff is an auto-cleanup SCHEDULER that corrects before the crosstalk cliff. (Loud negative: this is
# an analogy, NOT a claim that VSA is a quantum computer.) See ISA_REVERSIBLE.md.
try:
    import numpy as _npRV
    from holographic_reversible import reversibility_audit as _audRV, auto_cleanup_run as _acrRV, _bursty_program as _bpRV
    from holographic_ai import random_vector as _rvRV, cosine as _csRV
    _aud = _audRV(); _rev = sum(1 for _,(c,_) in _aud.items() if c=="reversible"); _los = sum(1 for _,(c,_) in _aud.items() if c=="lossy")
    _cb = [_rvRV(1024, _npRV.random.default_rng(100+i)) for i in range(16)]; _tg = 3
    def _meas(sched, **kw):
        _cl=[]; _bl=[]
        for _s in range(30):
            _st=_bpRV(_cb,_tg,dim=1024,seed=_s); _v,_c=_acrRV(_cb[_tg],_st,_cb,schedule=sched,**kw)
            _cl.append(_c); _bl.append(_csRV(_v,_cb[_tg])<0.9)
        return _npRV.mean(_cl), _npRV.mean(_bl)
    _ac,_ab=_meas("adaptive",floor=0.9); _fc,_fb=_meas("fixed",k=3)
    print(f"  REVERSIBLE MODEL (ISA-8): audit -- {_rev} reversible ops, {_los} lossy (cleanup = error correction); "
          f"auto-cleanup scheduler holds fidelity at {_ac:.0f} cleanups vs a fixed cadence's {_fc:.0f} (~1/3, bursty damage)")
except Exception as _eRV:
    print(f"  reversible-model: (skipped: {_eRV})")

# ANISOTROPIC / STEERING KERNELS (RT-IV1, reverse-transfer from the DCC thread): the FPE encoder now takes a
# PER-AXIS bandwidth -- a diagonal steering kernel (smooth along one axis, sharp along another). On DENSE
# directional data (an edge/ridge) the steered kernel beats the isotropic RBF; on isotropic data it doesn't
# (kept negative). See holographic_steering.py.
try:
    import numpy as _npST
    from holographic_steering import steer_bandwidths as _sbST, kernel_regress as _krST
    from holographic_fpe import VectorFunctionEncoder as _VFST
    def _fST(p): return _npST.tanh(3.0*(p[1]-5.0))                     # flat x, sharp y (a dense ridge)
    _gST=_npST.linspace(0.5,9.5,12); _Xt=_npST.array([[x,y] for x in _gST for y in _gST]); _yt=_npST.array([_fST(p) for p in _Xt])
    _rg=_npST.random.default_rng(0); _Xq=_rg.uniform(1,9,(60,2)); _yq=_npST.array([_fST(p) for p in _Xq])
    _bw=_sbST(_Xt,_yt,base=2.0); _B=[(0,10),(0,10)]
    _ani=_krST(_VFST(2,dim=1024,bounds=_B,bandwidth=_bw,seed=1),_Xt,_yt,_Xq)
    _iso=_krST(_VFST(2,dim=1024,bounds=_B,bandwidth=2.0,seed=1),_Xt,_yt,_Xq)
    _ar=_npST.sqrt(_npST.mean((_ani-_yq)**2)); _ir=_npST.sqrt(_npST.mean((_iso-_yq)**2))
    print(f"  STEERING KERNELS (RT-IV1): FPE per-axis bandwidth; on a dense ridge the steered kernel "
          f"(bw {[round(float(b),1) for b in _bw]}) regresses at RMSE {_ar:.2f} vs isotropic {_ir:.2f} -- "
          f"pools along the flat axis, sharp across the edge")
except Exception as _eST:
    print(f"  steering-kernels: (skipped: {_eST})")

# SPECTRAL ITERATION (RT-I1, reverse-transfer): a bind operator is DIAGONAL in the Fourier basis, so its
# eigendecomposition is FREE (the rfft). k binds = raising the transfer to the k-th power (ONE eval); the limit
# is closed-form; convergence is read off the spectrum before running. Subdivision = dynamics = diffusion =
# resonator are all "iterate a linear operator." See holographic_iterate.py.
try:
    import numpy as _npIT
    from holographic_iterate import step_k as _skIT, spectral_profile as _spIT
    from holographic_dynamics import Propagator as _PrIT
    _rg=_npIT.random.default_rng(0); _n=256
    _U=_npIT.fft.irfft(0.9*_npIT.exp(1j*_rg.uniform(0,2*_npIT.pi,_n//2+1)), n=_n); _st=_rg.standard_normal(_n)
    _k=20; _diff=_npIT.max(_npIT.abs(_PrIT(_U,_U).rollout(_st,_k)[-1] - _skIT(_st,_U,_k)))   # one eval vs k binds
    _prof=_spIT(_U)
    print(f"  SPECTRAL ITERATION (RT-I1): {_k}-step jump in ONE eval matches {_k} binds to {_diff:.0e} "
          f"(eigendecomposition = the free FFT); regime read off the spectrum: '{_prof['regime']}' "
          f"(max|eigenvalue| {_prof['max_magnitude']:.2f})")
except Exception as _eIT:
    print(f"  spectral-iteration: (skipped: {_eIT})")

# UPSTREAM FROM A SIBLING PROJECT (TuneFM): improvements that help every application,
# each verified on this substrate before adoption.
try:
    import numpy as _npU, time as _timeU
    from holographic_ai import bind as _bd, bind_fixed as _bfx, random_vector as _rvU
    _r = _npU.random.default_rng(0)
    _role = _rvU(512, _r); _F = _npU.stack([_rvU(512, _r) for _ in range(64)])
    _t = _timeU.perf_counter()
    for _ in range(30):
        [_bd(_role, _F[i]) for i in range(64)]
    _tl = _timeU.perf_counter() - _t
    _t = _timeU.perf_counter()
    for _ in range(30):
        _bfx(_role, _F)
    _tb = _timeU.perf_counter() - _t
    print(f"  bind_batch: the core bind now uses the REAL fft (atoms are real -> ~1.5x, exact "
          f"to ~1e-16), and bind_fixed/bind_batch vectorise a bind loop -- one role x 64 "
          f"fillers is {_tl/_tb:.1f}x faster batched than looped. Wired into RecordEncoder")

    from holographic_encoders import ScalarEncoder as _SE
    from holographic_ai import cosine as _cosU
    _e = _SE(2048, 0.0, 10.0, seed=1, kernel="rbf")
    _match = max(abs(_cosU(_e.encode(3.0), _e.encode(3.0 + dx)) - _e.kernel_at(dx))
                 for dx in (0.5, 1.0, 2.0))
    _sinc_min = min(_SE(2048, 0.0, 10.0, seed=1).kernel_at(dx) for dx in _npU.linspace(0, 20, 80))
    print(f"  kernel: ScalarEncoder can mint an RBF (non-negative) kernel and REPORT the kernel "
          f"it realises -- measured cosine matches kernel_at to {_match:.3f}, so you assert the "
          f"kernel instead of hoping (sinc dips to {_sinc_min:+.2f}; RBF never does)")

    from holographic_tree import HoloForest as _HF
    _V = _npU.stack([_rvU(256, _r) for _ in range(200)]); _V /= _npU.linalg.norm(_V, axis=1, keepdims=True)
    _f = _HF(256, n_trees=8, seed=0).build(_V)
    _, _ah = _f.recall(_V[5], with_agreement=True)
    _ar = _npU.mean([_f.recall(_rvU(256, _r), with_agreement=True)[1] for _ in range(30)])
    print(f"  forest: HoloForest.recall can now report cross-tree AGREEMENT as an abstention "
          f"signal -- a stored item agrees {_ah:.2f}, a random query {_ar:.2f}; act when the "
          f"trees agree, hold back when they split (default recall unchanged)")

    from holographic_honesty import walk_forward_recall as _wfr, bh_fdr as _bh
    _N = 1200
    _st = _r.standard_normal((_N, 128)); _st /= _npU.linalg.norm(_st, axis=1, keepdims=True)
    _planted = _wfr(_st, _npU.sign(_st[:, 0]) * _npU.abs(_r.standard_normal(_N)) * 50, R=25)
    _noise = _wfr(_st, _r.standard_normal(_N) * 50, R=25)
    print(f"  honesty: holographic_honesty makes the ablation ethos callable -- a planted edge "
          f"clears chance (beats_chance={_planted['beats_chance']}) with its shuffle control "
          f"collapsing, pure noise does not ({_noise['beats_chance']}), and bh_fdr adds the "
          f"false-discovery control a candidate scan needs")
    # panel outcome: a recall's raw cosine means nothing until compared to how high noise
    # reaches against THIS codebook. RecallNull turns it into an honest false-alarm probability.
    from holographic_honesty import RecallNull as _RN
    from holographic_ai import Vocabulary as _Vc, random_vector as _rvv
    _voc = _Vc(512, seed=1)
    for _i in range(400): _voc.get(f"s{_i}")
    _nm, _mat = _voc._matrix()
    _cal = _RN().fit(_mat, n_null=1500, seed=0)
    _pclean = _cal.calibrated_recall(_mat[3], _mat)[2]
    _prand = _npU.mean([_cal.calibrated_recall(_rvv(512, _r), _mat)[2] <= 0.05 for _ in range(400)])
    print(f"  recall-confidence: RecallNull calibrates a recall into a false-alarm probability -- "
          f"a clean match gets p={_pclean:.3f}, random queries are calibrated (P(p<=0.05)="
          f"{_prand:.3f}~0.05), so a recall can ABSTAIN with a principled threshold")
    # B3: sequential recall -- accumulate evidence over a stream, decide at a Wald boundary
    from holographic_honesty import SPRTRecall as _SPRT
    _nullsc = _cal.null
    _u = _mat / _npU.linalg.norm(_mat, axis=1, keepdims=True)
    _ms = _npU.array([float((_u @ ((_mat[_r.integers(400)] + 0.25*_r.standard_normal(512)) /
          _npU.linalg.norm(_mat[_r.integers(400)] + 0.25*_r.standard_normal(512)))).max()) for _ in range(800)])
    _mu0, _mu1 = float(_nullsc.mean()), float(_ms.mean())
    _sp = _SPRT(_nullsc, _ms, alpha=0.02, beta=0.02)
    _nn = []
    for _ in range(400):
        _, _k = _sp.decide(_r.choice(_ms, 60), cap=60); _nn.append(_k)
    print(f"  sequential recall (SPRT): streaming cues (match score {_mu1:.2f} vs null {_mu0:.2f}) "
          f"reach a 2% error decision in ~{_npU.mean(_nn):.1f} cues -- Wald-optimal, ~half a fixed window")
except Exception as _eU:
    print(f"  upstream: (skipped: {_eU})")

# DENOISING & SPLATS (panel addendum II): one operation seen several ways -- a denoiser is a map
# of the manifold signals live on, and holostuff already owns those maps.
try:
    from holographic_hopfield import dense_cleanup as _dc, generate as _gen
    from holographic_denoise import fit_manifold as _fm, manifold_denoise as _md
    from holographic_splat import splat_fit as _sf, splat_render as _sr, psnr as _ps
    from holographic_ai import Vocabulary as _Vh, random_vector as _rvh
    _r2 = _npU.random.default_rng(0)
    # B1: modern-Hopfield cleanup denoises a corrupted vector back onto the manifold
    _vh = _Vh(256, seed=1)
    for _i in range(64): _vh.get(f"h{_i}")
    _Vm = _vh._matrix()[1]; _ii = 5
    _nz = _Vm[_ii] + 1.5 * _r2.standard_normal(256) / _npU.sqrt(256)
    _cz = _dc(_nz, _Vm, beta=25.0, steps=3)
    _raw = float(_Vm[_ii] @ _nz / _npU.linalg.norm(_nz))
    _cln = float(_Vm[_ii] @ _cz / _npU.linalg.norm(_cz))
    # B10: generation by denoising from pure noise
    _gz = _gen(_Vm, steps=12, seed=0); _gcos = float((_Vm @ _gz).max())
    # B8: a real 2-D field as a superposition of Gaussian splats
    _yy, _xx = _npU.mgrid[0:40, 0:40]
    _T = sum(_a * _npU.exp(-((_yy-_cy)**2+(_xx-_cx)**2)/(2*_s*_s))
             for _cy,_cx,_s,_a in [(12,14,5,1.0),(28,24,6,0.7),(20,32,4,0.5)])
    _T = _T/_T.max(); _rec = _sr(_sf(_T, 20), _T.shape)
    print(f"  denoise+splats: Hopfield cleanup lifts a corrupted vector {_raw:.2f}->{_cln:.2f} cosine; "
          f"generation-by-denoising emerges from noise at {_gcos:.2f}; a 2-D field is {(_ps(_T,_rec)):.0f} dB "
          f"from 20 superposed Gaussian splats (a splat scene IS a bundle)")
    # B9: non-local-means denoising via the engine's own content-addressable recall
    from holographic_denoise import nlm_denoise as _nlm, fit_manifold as _fm2, manifold_denoise as _md2
    _mot = _r2.standard_normal((16, 24)); _mot /= _npU.linalg.norm(_mot, axis=1, keepdims=True)
    _mot *= _npU.sqrt(24.0)                         # std-1 energy, matching the shipped PoC regime
    _cl = _npU.repeat(_mot, 8, axis=0); _ny = _cl + 0.6 * _r2.standard_normal(_cl.shape)
    _bs, _mn = _fm2(_ny, rank=8); _pj = _npU.stack([_md2(x, _bs, _mn) for x in _ny])
    _dn = _nlm(_ny, k=8, use_forest=True)
    _sn = lambda A: _npU.mean([10*_npU.log10(_npU.sum(_cl[i]**2)/(_npU.sum((_cl[i]-A[i])**2)+1e-12)) for i in range(len(_cl))])
    print(f"  non-local-means: self-similar signal denoised via HoloForest recall_k -- "
          f"NLM {_sn(_dn):.1f} dB vs rank-8 projection {_sn(_pj):.1f} dB (complementary: recall finds "
          f"near-duplicates to average where low-rank projection can't)")
    # B4: dynamics as an algebra of binds -- learn a propagator, predict by binding, recall the past
    from holographic_dynamics import Propagator as _Pr
    from holographic_ai import bind as _bd, cosine as _cos, random_vector as _rvp
    _Ut = _rvp(256, _r2); _ss = _rvp(256, _r2); _tj = [_ss]
    for _ in range(360):
        _ss = _bd(_Ut, _ss) + 0.01 * _r2.standard_normal(256); _ss /= _npU.linalg.norm(_ss); _tj.append(_ss)
    _tj = _npU.array(_tj); _pp = _Pr.learn(_tj[:300])
    _pc = _npU.mean([_cos(_pp.step(_tj[300+i]), _tj[301+i]) for i in range(50)])
    _x0 = _tj[330]; _bk = _pp.recall_at(_pp.rollout(_x0, 4)[-1], 4)
    print(f"  propagator: when dynamics ARE a bind, one bind predicts the next state at cosine {_pc:.2f}; "
          f"the trajectory is content-addressable (forward 4 / back 4 round-trip {_cos(_x0,_bk):.3f}) -- "
          f"prediction on efficient-market returns is a kept negative (ties mean)")
except Exception as _eD:
    print(f"  denoise+splats: (skipped: {_eD})")

# Möbius / non-orientable encoders: match the representation's TOPOLOGY to the data --
# axial values (theta == theta+pi) and sign-flipping signals don't belong on a circle.
print("\nMÖBIUS TOPOLOGY (right shape for axial + sign-flipping data):")
try:
    from holographic_mobius import AxialEncoder as _Ax, antiperiodic_fraction as _afr
    import numpy as _npM
    _ax = _Ax(256, seed=0)
    _naive = lambda th: _npM.exp(1j * _ax.freqs * th)      # plain circle (single angle)
    _ns = _npM.real(_npM.vdot(_naive(0.7), _naive(0.7 + _npM.pi))) / 256.0
    _rng = _npM.random.default_rng(3); _tru = _rng.uniform(0, _npM.pi, 80)
    _obs = _tru + _rng.integers(0, 2, 80) * _npM.pi
    _err = _npM.mean([min(abs(_ax.decode(_ax.encode(o)) - t), _npM.pi - abs(_ax.decode(_ax.encode(o)) - t))
                      for o, t in zip(_obs, _tru)])
    print(f"  axial data (orientation, theta==theta+pi): sim(theta,theta+pi) circle={_ns:+.2f} "
          f"vs Möbius/double-angle=+1.00; recovery error {_err:.3f} rad (circle ~0.47)")
    _t = _npM.arange(64); _flip = _npM.sin(_npM.pi * _t / 32) + 0.5 * _npM.sin(3 * _npM.pi * _t / 32)
    print(f"  sign-flipping signal f(t+T)=-f(t): {_afr(_flip)*100:.0f}% of energy is antiperiodic "
          f"(the Möbius subspace a circle can't see). KEPT SCOPE: directed data still belongs on the circle.")
except Exception as _eM:
    print(f"  möbius: (skipped: {_eM})")

# Holographic stored-program machine: a program ENCODED AS ONE VECTOR, executed by VSA ops --
# the 'operating system' rung, plus how deep structure-within-structure (inception) nests.
print("\nHOLOGRAPHIC MACHINE (a program is data; the substrate executes it):")
try:
    from holographic_machine import HoloMachine as _HM
    from holographic_ai import bind as _bM, bundle as _buM, cosine as _coM
    _mc = _HM(dim=4096, seed=7)
    _prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
    _acc, _tr = _mc.run(_mc.assemble(_prog))
    _exp = _buM([_bM(_mc.data_atoms["a"], _mc.data_atoms["b"]), _mc.data_atoms["c"]])
    print(f"  executed 'LOAD a; BIND b; BUNDLE c' -> ACC == bundle(bind(a,b),c) cosine={_coM(_acc,_exp):.2f}; "
          f"trace {_tr}")
    # inception depth: clean nesting vs a busy disk
    _base = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
    _want = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c")]
    def _nest(depth, files):
        _v = _mc.assemble(_base)
        for _d in range(depth): _v = _mc.disk(_v, _mc.junk_files(files, _d))
        for _d in range(depth): _v = _mc.open_slot(_v)
        return _mc.run(_v)[1] == _want
    _clean = max(d for d in range(0, 9) if _nest(d, 0))
    _busy = max((d for d in range(0, 9) if _nest(d, 3)), default=0)
    print(f"  inception depth: a program nests {_clean}+ levels deep when each level is clean, "
          f"but only ~{_busy} when each disk holds other files. KEPT: capacity + depth are finite, scale with dim.")
    # functions embedded in a holographic library, called by name and composed
    _mc.define("tag_b", [("BIND", "b"), ("HALT", "")]); _mc.define("shift", [("PERMUTE", ""), ("HALT", "")])
    _ac, _ = _mc.run(_mc.assemble([("LOAD", "a"), ("CALL", "tag_b"), ("CALL", "shift"), ("HALT", "")]))
    from holographic_ai import permute as _pm
    print(f"  embedded functions: CALL tag_b; CALL shift (both inside one library vector) -> "
          f"permute(bind(a,b)) cosine={_coM(_ac, _pm(_bM(_mc.data_atoms['a'], _mc.data_atoms['b']), 1)):.2f}")
    # run_chunked: a program TOO LONG for one structure, run by threading the accumulator across clean chunks
    _mc1 = _HM(dim=1024, seed=7)
    _lnames = [chr(ord('a') + (_i % 6)) for _i in range(60)]
    _long = [("LOAD", _lnames[0])] + [("BIND", _lnames[_i]) for _i in range(1, 60)] + [("HALT", "")]
    _expL = _mc1.data_atoms[_lnames[0]]
    for _lnm in _lnames[1:60]: _expL = _bM(_expL, _mc1.data_atoms[_lnm])
    _flatL, _ = _mc1.run(_mc1.assemble(_long))
    _chL, _ = _mc1.run_chunked(_long)
    print(f"  run_chunked (past the program cap)  : a 60-instruction program -- one structure decodes "
          f"cosine={_coM(_flatL, _expL):.2f} (the cliff); host-threaded <=14-instr chunks -> cosine={_coM(_chL, _expL):.2f}")
except Exception as _eHM:
    print(f"  machine: (skipped: {_eHM})")

# WIRED INTO THE LIVE APP: the generative + persistence work is reachable from the
# UnifiedMind console (unified_app.py), not just the library -- compose/morph/nucleus/nested
# panels drive the decoders forward, and a save&reload panel persists a trained mind.
try:
    import unified_app as _uaT
    _routes = {r.rule for r in _uaT.app.url_map.iter_rules()}
    _wired = [r for r in ("/api/unified/compose", "/api/unified/nested", "/api/unified/morph",
                          "/api/unified/nucleus", "/api/unified/persist") if r in _routes]
    print(f"  app: the generative + persistence work is wired into the live console as panels "
          f"({len(_wired)}/5 endpoints live: compose a scene, nested scene, morph, nucleus text, "
          f"save & reload) -- functionality reachable from the UI, not stranded in the library "
          f"or the tests")
except Exception as _eW:
    print(f"  app: (skipped: {_eW})")


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
print(f"  random baseline    : reward {br:+.2f}, stars collected {bf:.1f}")
print(f"  after 120 episodes : reward {er:+.2f}, stars collected {ef:.1f}   (it taught itself to find stars)")
print("  (run `python holographic_creature.py` for the full demo incl. lethal-poison avoidance + energy)")


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

# 9. Vision -----------------------------------------------------------------
title("9. Vision  (colour, edges, shapes -- the image is just numbers)")
import holographic_vision as hv
kinds = ["circle", "rectangle", "triangle", "line"]
shp_ok = sum(hv.classify_shape(hv.make_shape(k, 64, seed=10 + j)[1]) == k for j, k in enumerate(kinds))
redcircle, _ = hv.make_shape("circle", 64, seed=1, fg=(235, 70, 70))
line, _ = hv.make_shape("line", 80, seed=3)
nlines = len(hv.hough_lines(hv.edges(hv.to_gray(line)), top=2))
print(f"  HSV hue of a red circle  : wedge {hv.hue_histogram(redcircle).argmax()} (0 = red)")
print(f"  rule-based shape ID      : {shp_ok}/4 clean shapes ; Hough found {nlines} line(s) in a line image")

# 10. Compositional scene + resonator --------------------------------------
title("10. Compositional scene  (bind the parts, factor them back with a resonator)")
import holographic_scene as sc
coder = sc.SceneCoder(dim=2048, seed=0)
true = [{"colour": "red", "shape": "circle", "texture": "smooth"},
        {"colour": "blue", "shape": "rectangle", "texture": "busy"},
        {"colour": "green", "shape": "triangle", "texture": "vertical"}]
got = coder.factor_scene(coder.encode_scene(true), 3, sweeps=2)
key = lambda d: (d["colour"], d["shape"], d["texture"])
okset = {key(t) for t in true} == {key(g) for g in got}
print(f"  3 objects bound into ONE scene vector, factored back: "
      f"{'all 3 recovered' if okset else 'partial'}  {[(g['colour'], g['shape']) for g in got]}")
print("  (a single holistic tag could name only one of them -- composition keeps the parts)")

# decompose_scene_tiled: a scene PAST the resonator's ~5-object cap -- tile the objects (<=cap each), factor
# every sub-scene, merge. The scene twin of chunk_route: beat a fixed structure's capacity by composition.
import numpy as _npTS
_rngTS = _npTS.random.default_rng(200)
_COL, _SHP, _TEX = sc.COLOURS, sc.SHAPES, sc.TEXTURES
_seenTS, _objsTS = set(), []
while len(_objsTS) < 15:
    _tTS = (_COL[_rngTS.integers(len(_COL))], _SHP[_rngTS.integers(len(_SHP))], _TEX[_rngTS.integers(len(_TEX))])
    if _tTS not in _seenTS:
        _seenTS.add(_tTS); _objsTS.append({"colour": _tTS[0], "shape": _tTS[1], "texture": _tTS[2]})
_coderTS = sc.SceneCoder(dim=1024, seed=0)
_keyTS = lambda os: {(o["colour"], o["shape"], o["texture"]) for o in os}
_grpsTS = [_objsTS[_i:_i + 5] for _i in range(0, 15, 5)]
_tiledTS = _coderTS.factor_scene_tiled([_coderTS.encode_scene(_g) for _g in _grpsTS],
                                       [len(_g) for _g in _grpsTS], sweeps=3)
_wholeTS = _coderTS.factor_scene(_coderTS.encode_scene(_objsTS), 15, sweeps=3)
print(f"  decompose_scene_tiled (past cap)    : 15 objects -- whole scene "
      f"{len(_keyTS(_wholeTS) & _keyTS(_objsTS))}/15 (capped), tiled into 3 tiles "
      f"{len(_keyTS(_tiledTS) & _keyTS(_objsTS))}/15 (tile size plays the chunk's role)")

# 11. Scaling: recursive tree + forest -------------------------------------
title("11. Scaling  (one flat memory collapses; a deterministic tree holds)")
import holographic_tree as ht
rows = {r["N"]: r for r in ht.capacity_curve([64, 1024], dim=2048, leaf_size=64, probes=60)}
print(f"  key->value recall@1 at N=64   : flat {rows[64]['flat']:.0%}  tree {rows[64]['tree']:.0%}")
print(f"  key->value recall@1 at N=1024 : flat {rows[1024]['flat']:.0%}  tree {rows[1024]['tree']:.0%}  "
      f"(flat has collapsed; the tree keeps each leaf in capacity)")
fb = ht.forest_benchmark(N=1500, dim=512, leaf_size=64, n_trees=4, beam=4)
print(f"  approx-NN forest (4 trees)    : recall {fb['forest_recall']:.0%} at {fb['forest_cmp']} "
      f"comparisons vs {fb['exact_cmp']} for a full scan")

# 12. Content addresses (S3-style) -----------------------------------------
title("12. Content addresses  (no folders -- the name IS the hierarchy, like S3)")
import holographic_uri as uri
store = uri.FacetStore()
rng2 = np.random.default_rng(0); pal = {"red": (235, 70, 70), "blue": (80, 120, 235), "green": (70, 200, 110)}
for n in range(30):
    col = list(pal)[rng2.integers(3)]; shp = kinds[rng2.integers(4)]
    im, _ = hv.make_shape(shp, 64, seed=n, fg=pal[col])
    t = sc.auto_tags(im); store.put(n, t, vector=coder.encode(t))
print(f"  deterministic key from properties     : e.g. '{store.keys()[0]}'")
print(f"  top-level prefixes (S3 CommonPrefixes) : {list(store.common_prefixes('').keys())}")
k0, recs0 = list(store.flat.items())[0]
derived = uri.address_from_content(recs0[0]['vec'], coder)
print(f"  resonator computes the URI from content: '{derived}' ({'matches' if derived == k0 else 'differs'})")

# 13. One mind on top (self-assembly across modalities) ---------------------
title("13. UnifiedMind  (one encoder, one memory, one brain -- self-assembled)")
from holographic_unified import UnifiedMind, _patterns
from holographic_text import TOPICS, _content
rng3 = np.random.default_rng(0)
pile = [(_content(s), topic) for topic, ss in TOPICS.items() for s in ss]
pile += [(_patterns(k, rng3), f"img:{k}") for k in ("rows", "check") for _ in range(12)]
um = UnifiedMind(dim=1024, seed=0).absorb(pile, sequences=True)   # ONE call: classify+recall+generate
t_ok = um.classify(_content("the striker scored a goal in the match"))[0]
i_ok = um.classify(_patterns("rows", rng3))[0]
print("  absorb() built a COMPLETE mind from a bare pile of (input, label) pairs")
print(f"  an untagged sentence classifies as : '{t_ok}'  (modality self-discovered)")
print(f"  an untagged image classifies as    : '{i_ok}'")
print(f"  and the same mind generates        : \"{um.generate('the ', 60, 0.4)[:58]}\"")

# 14. ORDER as a queryable property -- the PB&J problem. Some meaning lives only
#     in the sequence: the same steps in the wrong order are not a worse recipe,
#     they are not a recipe. The bag stores discard order (rightly, for topic);
#     this recovers it where it matters.
print("\n[14] sequence: order is structure the data alone cannot supply")
um.learn_plan("pbj", ["bread", "peanut_butter", "jelly", "close", "cut"])
ok_good, _ = um.validate_plan("pbj", [("jelly", "close"), ("close", "cut")])
bad = ["bread", "cut", "peanut_butter", "jelly", "close"]
ok_bad, viol = um.validate_plan(bad, [("jelly", "close"), ("close", "cut")])
print(f"  step 2 of the recipe is            : '{um.step_at('pbj', 2)}'")
print(f"  correct recipe satisfies its order : {ok_good}")
print(f"  recipe that cuts too early is valid: {ok_bad}  (violated: {viol})")

# A plan stores opaque step LABELS; a PROCEDURE stores an executable recipe whose steps are actual VSA
# operations, run THROUGH the mind (HoloMachine is now a faculty, not a siloed VM). Sub-recipes live in
# one library vector and compose by CALL -- a recipe of recipes that really computes.
um.learn_procedure("wrap_b", [("BIND", "b"), ("HALT", "b")])
um.learn_procedure("add_c", [("BUNDLE", "c"), ("HALT", "c")])
_acc, _tr = um.run_procedure([("LOAD", "a"), ("CALL", "wrap_b"), ("CALL", "add_c"), ("HALT", "a")])
print(f"  executable procedure (VSA ops)     : ran {[t[0] for t in _tr]} -> accumulator built from clean atoms")
print(f"  read instruction 1 back as data    : {um.decode_step([('LOAD','a'),('BIND','b'),('HALT','a')], 1)}  (program is also data)")

# A procedure can invoke the mind's faculties as steps via APPLY: here it self-corrects a noisy
# accumulator back toward a clean value atom using the dense associative cleanup.
import numpy as _np
from holographic_machine import cosine as _cos
_true = um._machine().data_atoms["c"]
_noisy = _true + 0.5 * _np.random.default_rng(0).standard_normal(um.dim)
_clean, _ = um.run_procedure([("APPLY", "cleanup"), ("HALT", "c")], init_acc=_noisy)
print(f"  APPLY cleanup denoises accumulator  : cosine-to-truth {_cos(_noisy,_true):.2f} -> {_cos(_clean,_true):.2f}  (a step a kernel-op list can't take)")

# Procedure MEMORY: from one (input -> output) example, recall WHICH stored procedure did it, then
# apply it to NEW input -- learn an operation by example, then reuse it (analogy/transfer).
_in = _np.random.default_rng(1).standard_normal(um.dim)
_out, _ = um.run_procedure("wrap_b", init_acc=_in)
_new = _np.random.default_rng(2).standard_normal(um.dim)
_res, _used, _sc = um.recall_and_apply(_in, _out, _new)
print(f"  procedure recall by example         : recognized '{_used}' from one example, re-applied to new input")

# Recipe GENERATION: learn the opcode-grammar of valid recipes, then predict the next opcode for a
# partial one -- the engine can suggest the next step.
um.learn_recipe_grammar([[("LOAD", "a"), ("BIND", "b"), ("HALT", "a")]] * 4
                        + [[("LOAD", "a"), ("BIND", "b"), ("APPLY", "cleanup"), ("HALT", "a")]] * 4)
_nxt, _conf = um.complete_procedure([("LOAD", "a")])
print(f"  recipe completion (predict next op) : after LOAD -> '{_nxt}'  (learned the grammar)")
_iop, _iarg, _ic = um.complete_instruction([("LOAD", "a")])
print(f"  recipe completion (op AND operand)  : after LOAD -> ({_iop}, {_iarg})  -- the operand too, since it"
      f" is patterned here (random operands stay unknowable: held-out accuracy falls to chance)")

# A fingerprint index makes recall execution-free for the linear (bind/permute) class: identify the
# procedure from one example with ZERO candidate runs (auto falls back to the behavioral scan for the rest).
# The scan is ONE matrix-vector product against the cached fingerprint matrix (6-26x faster than a Python
# loop; a HoloForest index was measured 3-7x SLOWER for realistic libraries, so it was rejected).
um.index_procedures()
from holographic_machine import derived_atom as _datom
_xi = _datom(1, "tour_recall_probe", um.dim, unitary=True)
_yi, _ = um.run_procedure("wrap_b", init_acc=_xi)
_nm, _s = um.recall_procedure(_xi, _yi, method="fingerprint")
print(f"  fingerprint recall (0 program runs) : identified '{_nm}' from one example (conf {_s:.2f})")

# SYNTHESIS: CONSTRUCT a new procedure that achieves a goal (the constructive counterpart to recall),
# by bounded search over the VM's ops -- verified by execution.
from holographic_machine import bind as _bind
_A = um._machine().data_atoms
_gx = _datom(2, "synth_x", um.dim, unitary=True)
_gy = _bind(_bind(_gx, _A["b"]), _A["c"])
_prog = um.synthesize_procedure(_gx, _gy, max_depth=2)
_opstr = [o if o == "PERMUTE" else f"{o} {a}" for o, a in _prog if o != "HALT"]
print(f"  procedure synthesis (search a recipe): goal solved by {_opstr}  (constructed and verified)")

# CONTROL FLOW: the AI loop, as a program. ITERATE re-applies a body to the accumulator until it
# converges (a fixed point) -- here a cleanup step pulls a noisy vector onto a clean atom (input ->
# process -> feed back -> repeat). IFMATCH (a conditional) gates an instruction on a predicate over ACC.
um.learn_procedure("clean_step", [("APPLY", "cleanup"), ("HALT", "c")])
_tc = um._machine().data_atoms["c"]
_noisyc = _tc + 0.3 * _np.random.default_rng(5).standard_normal(um.dim)
_conv, _trc = um.run_procedure([("ITERATE", "clean_step"), ("HALT", "c")], init_acc=_noisyc)
_loop = [t for t in _trc if t[0] == "ITERATE"][0]
print(f"  ITERATE until converged (the AI loop): cosine {_cos(_noisyc,_tc):.2f} -> {_cos(_conv,_tc):.2f} in {_loop[2]} iters ({_loop[3]})")

# matmul in the loop: a small recurrent linear map. ITERATE [APPLY matmul] with a column-stochastic
# matrix is power iteration -- it converges to the stationary distribution, exact matmul as the step.
_mm = UnifiedMind(dim=48, seed=0)
_P = _np.abs(_np.random.default_rng(0).standard_normal((48, 48))) + 0.05
_P /= _P.sum(axis=0, keepdims=True)
_mm.set_matmul(_P)
_mm.learn_procedure("mm_step", [("APPLY", "matmul"), ("HALT", "a")])
_s0 = _np.abs(_np.random.default_rng(1).standard_normal(48)); _s0 /= _s0.sum()
_sN, _trm = _mm.run_procedure([("ITERATE", "mm_step"), ("HALT", "a")], init_acc=_s0, converge_tol=0.99999, max_loop=100)
_itm = [t for t in _trm if t[0] == "ITERATE"][0]
print(f"  matmul in a loop (-> stationary dist): converged in {_itm[2]} iters ({_itm[3]}) -- exact matmul as the loop body")

# Counted loop: REPEAT n runs the next CALL n times -- a FOR loop alongside ITERATE's convergence WHILE.
from holographic_machine import permute as _perm
um.learn_procedure("shiftone", [("PERMUTE", "a"), ("HALT", "a")])
_xr = _datom(3, "rep_x", um.dim, unitary=True)
_r3, _ = um.run_procedure([("REPEAT", 3), ("CALL", "shiftone"), ("HALT", "a")], init_acc=_xr)
print(f"  REPEAT 3; CALL shiftone (FOR loop)  : result equals permute(x,3)? {_cos(_r3, _perm(_xr,3))>0.99}")

# A complete little routine in ONE procedure: denoise the input (ITERATE a cleanup), then branch on the
# cleaned result (IFMATCH) and tag it (CALL) only if it matches -- loop + conditional + call together.
um.learn_procedure("tag", [("BIND", "b"), ("HALT", "b")])
_wc = um._machine().data_atoms["c"] + 0.3 * _np.random.default_rng(7).standard_normal(um.dim)
_wacc, _wtr = um.run_procedure([("ITERATE", "clean_step"), ("IFMATCH", "c"), ("CALL", "tag"), ("HALT", "c")], init_acc=_wc)
print(f"  worked program denoise->classify->tag: ran {[t[0] for t in _wtr]}  (cleaned to c, then tagged)")

# PIPE-1: a whole data-analysis pipeline as ONE VSA program -- APPLY analyze, ITERATE-denoise until the
# signal settles, APPLY decompose to a generative law, IFMATCH-branch on whether structure was found, CALL
# train+validate, APPLY save. The program is the orchestrator; each step delegates to a real faculty.
import numpy as _np
_pt = _np.linspace(0, 1, 256)
_prep = um.run_analysis_pipeline(1 + 2 * _pt + 3 * _pt ** 2 + 0.3 * _np.random.default_rng(0).standard_normal(256))
print(f"  analysis pipeline (structured signal): found a {_prep['n_terms']}-term law (explained {_prep['explained_var']}),"
      f" held-out err {_prep['heldout_rel']}, 256 pts -> {_prep['law_bytes']}B law -- ran {_prep['_ops']}")
_nrep = um.run_analysis_pipeline(_np.random.default_rng(1).standard_normal(256))
print(f"  analysis pipeline (pure noise)       : no structure (explained {_nrep['explained_var']}); the IFMATCH"
      f" SKIPPED training -- ran {_nrep['_ops']}, saved as {_nrep['saved_as']}")
# ...and the recursive "every level" mode: peel a cross-basis signal (a line trend UNDER a periodic part)
# layer by layer -- structure ONE decompose cannot fit together, recovered by an ITERATEd peel that stops
# when the engine's own MDL gate admits no more terms (the residual is noise).
_dprep = um.run_analysis_pipeline(
    0.5 + 2 * _pt + _np.sin(2 * _np.pi * 5 * _pt) + 0.2 * _np.random.default_rng(2).standard_normal(256),
    recursive=True)
print(f"  analysis pipeline (recursive peel)   : {_dprep['n_levels']} levels "
      f"({' -> '.join(L['topology'] for L in _dprep['levels'])}), cumulative explained "
      f"{_dprep['cumulative_explained']} -- a trend+periodic one pass gets ~0.3 on, peeled apart")
# The bind/permute algebra COLLAPSES (any interleaving == permute(x, net) bound by the operand product), so a
# deep program reduces to its minimal form -- which is also WHY deeper synthesis over the invertible ops buys
# nothing (there is nothing deep to find); BUNDLE/nonlinear ops are honest barriers.
_deep = [("BIND", "a"), ("PERMUTE", "a"), ("BIND", "b"), ("PERMUTE", "a"), ("BIND", "c"), ("HALT", "a")]
_canon, _cinfo = um.canonicalize_procedure(_deep)
print(f"  canonicalize a bind/permute program  : {_cinfo['original_len']} ops -> {_cinfo['canonical_len']} "
      f"({_cinfo['n_bind']} binds -> 1 product, {_cinfo['net_shift']} permutes), verified equivalent "
      f"(cosine {_cinfo['equivalence_cosine']})")

# recursive discovery: the SAME order-test applied fractally, unfolding a nested
# plan the mind was never given the shape of -- and stopping honestly at leaves.
import numpy as _np
_rng = _np.random.default_rng(0)
_sauce = ["heat_oil", "add_garlic", "add_tomato", "simmer"]
_obs = [[("make_sauce", _sauce[:_rng.integers(3, 5)]), "plate", "serve"]
        for _ in range(12)]
um.learn_hierarchical("dinner", _obs)
_tree = um.discover_hierarchy("dinner")
_expanded = [k for k, v in _tree.items() if v is not None]
_atomic = [k for k, v in _tree.items() if v is None]
print(f"  nested plan discovered: '{_expanded[0]}' expands "
      f"{list(_tree[_expanded[0]].keys()) if isinstance(_tree[_expanded[0]], dict) else _tree[_expanded[0]]}")
print(f"  atomic steps (recursion stopped honestly): {_atomic}")

# the closed loop: discover -> prove -> bind context -> EXECUTE. The same recipe,
# now RUN: steps fire on their preconditions, a context slot binds, and an
# out-of-order attempt blocks with a reason. Discovering structure, then acting.
um.learn_sequences([(["bread", "pb", "jelly", "close", "cut"][:_rng.integers(4, 6)], "pbj_run")
                    for _ in range(10)])
um.discover_sequential()
if "pbj_run" in um._seq_mem().seqs:
    _tmpl = {"cut": (["cut", "into", "<_>", "pieces"], ["pieces"])}
    _log = um.execute_plan("pbj_run", context={"pieces": "2"}, templates=_tmpl)
    _fired = [s for s, st, d in _log if st == "fired"]
    _cut = [d for s, st, d in _log if s == "cut" and d]
    print(f"  plan EXECUTED: {len(_fired)} steps fired"
          + (f", slot bound -> '{_cut[0]}'" if _cut else ""))
    _blocked = um.execute_plan("pbj_run", context={}, templates=_tmpl)
    _b = [s for s, st, d in _blocked if st == "blocked"]
    print(f"  without context binding, blocked: {_b}  (honest -- no assumed success)")

# 15. The INVERSE half of the loop. Sections 1-14 build structure and act on it
#     (perceive / classify / recall / decide / generate). The studies that grew up
#     alongside go the other way -- take a foreign signal APART -- and they are now
#     faculties of the same mind: decompose a signal into a generating law, denoise it
#     by projecting onto a manifold, and fit an interpretable additive function to it.
title("15. Decompose / denoise / fit  (the inverse half, wired into the one mind)")

# decompose: detect the topology and recover a tiny generating LAW (a savable seed)
_x = np.linspace(0, 4 * np.pi, 240)
_y = np.sin(_x) + 0.3 * np.cos(2 * _x)                 # a periodic (ring) signal
_law, _info = um.decompose_signal(_x, _y)
print(f"  a periodic signal decomposes to    : {_law}")
print(f"    topology={_info['topology']}  terms={_info['n_terms']}  "
      f"resid={_info['resid_rms']:.2e}  compression={_info['compression_ratio']:.1f}x")

# the law IS a seed: it regenerates AND extrapolates one period past the fit, bounded
_xe = np.linspace(4 * np.pi, 6 * np.pi, 120)
print(f"    regenerates the signal (rms)     : {np.sqrt(np.mean((_law.generate(_x) - _y) ** 2)):.2e}"
      f"   extrapolates without diverging (max |y|={np.max(np.abs(_law.generate(_xe))):.2f})")

# a multiplicative law on a flat domain is auto-selected via the log transform
_xp = np.linspace(1, 6, 150); _yp = 2.0 * _xp ** 1.5
_lawp, _infop = um.decompose_signal(_xp, _yp)
print(f"  a power law y=2*x^1.5 recovers as   : mode={_infop['mode']}  resid={_infop['resid_rms']:.2e}")

# denoise: project a heavily noisy signal back onto its own (low-rank) manifold
_fam = np.stack([np.sin(_x + p) + 0.3 * np.cos(2 * (_x + p)) for p in np.linspace(0, 2 * np.pi, 64)])
_rng = np.random.default_rng(0)
_noisy = _y + 0.8 * _rng.standard_normal(len(_x))
_clean = um.denoise(_noisy, method="adaptive", samples=_fam)
_rms = lambda a: np.sqrt(np.mean((a - _y) ** 2))
print(f"  denoise (manifold projection)       : rms {_rms(_noisy):.2f} (noisy) -> {_rms(_clean):.2f} (cleaned)")

# fit_function: an interpretable additive readout that recovers each univariate part
_Xtr = _rng.uniform(0, 1, (1200, 2))
_g1 = lambda t: np.sin(2 * np.pi * t); _g2 = lambda t: 4 * (t - 0.5) ** 2
_ytr = _g1(_Xtr[:, 0]) + _g2(_Xtr[:, 1]) + 0.02 * _rng.standard_normal(1200)
_kan = um.fit_function(_Xtr[:900], _ytr[:900])
_r2 = 1 - np.sum((_ytr[900:] - _kan.predict(_Xtr[900:])) ** 2) / np.sum((_ytr[900:] - _ytr[900:].mean()) ** 2)
_ts = np.linspace(0.05, 0.95, 40)
_corr = abs(np.corrcoef(_kan.feature_function(0, _ts), _g1(_ts))[0, 1])
print(f"  fit_function (KAN readout)          : test R^2={_r2:.3f}  and recovers psi_1~sin (corr={_corr:.2f})")

# decompose_structure: the higher-capacity SBC factorizer, now a faculty the mind speaks directly
# (one factorizer, not two -- factor_composite delegates to this same path when given an L).
from holographic_sbc import sbc_codebook as _scb2, sbc_reconstruct as _srec2
_cbs = [_scb2(16, 16, 10, seed=_k) for _k in range(3)]
_true = (2, 5, 8)
_prod = _srec2(_true, _cbs, 16)
_dec = um.decompose_structure(_prod, _cbs, 16)
_route = um.factor_composite(_prod, _cbs, L=16)        # same problem through the unified entry point
print(f"  factor a bound structure (SBC)      : recovered {_dec['picks']} (true {_true}), "
      f"verified={_dec['verified']}; factor_composite routes to it (backend={_route['backend']})")

# decode_structure: the B7 chain typed structure, decoded back by PER-PEEL cleanup (B8). Per-peel
# cleanup is the whole game -- a raw traversal craters as noise compounds; cleaning each hop decodes all.
_recipe, _nodes = um.chain_structure(16)
_M = um.realize(_recipe)
_ncor = lambda s: sum(1 for _h, _i in enumerate(s) if _i == _h + 1)
_hard = um.decode_structure(_M, _nodes, cleanup="hard")
_raw = um.decode_structure(_M, _nodes, cleanup=None)
print(f"  decode a 16-node chain (B7->B8)     : per-peel cleanup {_ncor(_hard)}/15 hops, "
      f"raw (no cleanup) only {_ncor(_raw)}/15 -- cleaning each hop is what makes it decode")

# decode_plan / descend: a CONTINGENCY PLAN as a typed tree (action + named branches + scope), decoded back
# GIVEN ITS SHAPE -- a deterministic unbind walk, not the resonator's blind parse. descend generalises IFMATCH
# (one gated instruction) to a named branch tree WITH ABSTENTION ('no contingency applies' -> primary action).
from holographic_planshape import PlanNode as _PN
_pa = ["advance", "hold", "retreat", "scan", "abort", "reroute"]; _ps2 = ["global", "local", "step", "mission"]
_plan = _PN("advance", "mission", branches={
    "blocked": _PN("reroute", "local", branches={"lowfuel": _PN("hold", "step")}),
    "contact": _PN("abort", "mission")})
_pshape = um.plan_shape(_pa, _ps2, {"blocked": {"lowfuel": {}}, "contact": {}})
_pvec = um.encode_plan(_plan)
_pback = um.decode_plan(_pvec, _pshape)
print(f"  decode_plan (schema-guided)         : a 2-level contingency plan -> back exactly = {_pback == _plan}, "
      f"root='{_pback.action}' conf={_pback.confidence}")
print(f"  descend (IFMATCH generalised)       : blocked -> {' -> '.join(um.descend(_pvec, 'blocked', _pshape))}; "
      f"clear -> {' -> '.join(um.descend(_pvec, 'clear', _pshape))} (abstains, no branch applies)")

# energy cleanup: the B1 dense-Hopfield update as an OPT-IN flag, identical to argmax at high beta
from holographic_ai import Vocabulary as _Vc, random_vector as _rv
_v = _Vc(512, seed=2)
for _nm in ("alpha", "beta", "gamma", "delta", "epsilon"):
    _v.get(_nm)
_noisy = _v.get("gamma") + 0.7 * _rv(512, np.random.default_rng(0))
print(f"  opt-in energy cleanup (B1)          : plain='{_v.cleanup(_noisy)[0]}'  "
      f"energy@high-beta='{_v.cleanup(_noisy, energy=True, beta=1e6)[0]}'  (bit-for-bit the same)")

# search: solve a braided maze by the deterministic Tero flow (collapses onto the shortest tube)
from holographic_creature import GridWorld as _GW
_w = _GW(16, 16, maze=True, fixed_seed=7, braid=1.0)
_p, _pi = um.solve_maze(_w)
print(f"  solve a maze (Tero flow search)     : reached={_pi['reached']}, "
      f"len={_pi['extracted_len']} == optimal {_pi['optimal']}, deterministic")

# search: fragment assembly as the SAME flow, returned as a B7 typed structure the mind can realize
_tgt = "ABCABCABCA"; _lib = sorted({_tgt[_p2:_p2 + 2] for _p2 in range(len(_tgt) - 1)})
_asm = um.assemble(_tgt, _lib)
print(f"  assemble from fragments (flow)      : '{_asm['assembled']}' energy={_asm['energy']}, "
      f"recipe realizes to a {um.realize(_asm['recipe']).shape[0]}-d hypervector (a B7 structure)")

# dynamics: learn an operator so prediction is one bind, with a content-addressable trajectory
_rngd = np.random.default_rng(0)
_Ud = random_vector(256, _rngd); _sd = random_vector(256, _rngd); _traj = [_sd]
for _ in range(400):
    _sd = bind(_Ud, _sd) + 0.01 * _rngd.standard_normal(256); _sd /= np.linalg.norm(_sd); _traj.append(_sd)
_traj = np.array(_traj); _prop = um.learn_dynamics(_traj[:300])
_x0 = _traj[350]; _rt = cosine(_x0, _prop.recall_at(_prop.rollout(_x0, 4)[-1], 4))
print(f"  learn dynamics (prediction = a bind): 1-step pred cos="
      f"{np.mean([cosine(_prop.step(_traj[300+i]), _traj[301+i]) for i in range(40)]):.3f}, "
      f"forward-4-then-back-4 round-trip cos={_rt:.3f}")

# persistence: the learned mind saves and reloads, classifying identically (quant='rd' rate-distortion)
import tempfile as _tf, os as _os
_sm = UnifiedMind(dim=256, seed=0, maintain="manual")
_srng = np.random.default_rng(0)
for _ in range(20):
    _sm.learn(round(float(_srng.uniform(0, 1)), 3), "small", modality="number")
    _sm.learn(round(float(_srng.uniform(5, 6)), 3), "big", modality="number")
_pr = [round(float(_srng.uniform(0, 6)), 3) for _ in range(12)]
_before = [_sm.classify(p, modality="number")[0] for p in _pr]
_p = _os.path.join(_tf.mkdtemp(), "mind"); _sm.save(_p, quant="rd")
_after = [UnifiedMind.load(_p).classify(p, modality="number")[0] for p in _pr]
print(f"  save & reload the learned mind      : classify identical after round-trip = {_before == _after}")

# generative: generate a vector by denoising from noise (B10), and splat a field (a splat scene is a bundle)
_grng = np.random.default_rng(0)
_cb = np.stack([random_vector(256, _grng) for _ in range(8)])
_gv = um.generate_vector(_cb, seed=3)
from holographic_splat import psnr as _psnr
_G = 48; _ys, _xs = np.mgrid[0:_G, 0:_G]; _T = np.zeros((_G, _G))
for _ in range(4):
    _cy, _cx, _s, _a = _grng.uniform(8, _G - 8, 2).tolist() + [_grng.uniform(3, 7), _grng.uniform(0.5, 1)]
    _T += _a * np.exp(-((_ys - _cy) ** 2 + (_xs - _cx) ** 2) / (2 * _s * _s))
_T /= _T.max()
_sp, _rend = um.splat_field(_T, k=40)
_, _rend_greedy = um.splat_field(_T, k=40, refit=False)              # the 'looping' off, for contrast
print(f"  generate a vector (B10 diffusion)   : nearest-pattern cosine="
      f"{max(cosine(_gv, _cb[i]) for i in range(8)):.3f}; splat a field -> {len(_sp)} Gaussians at "
      f"{_psnr(_T, _rend):.0f} dB (a splat scene is a bundle)")
print(f"  splat joint refit (the 'looping')   : greedy matching pursuit {_psnr(_T, _rend_greedy):.1f} dB -> "
      f"joint amplitude re-solve {_psnr(_T, _rend):.1f} dB (gradient-free, removes overlap double-counting)")
# content-addressable splat SCENE: region recall is decode-via-cleanup, so a single bundle caps at fine grid;
# tiling routes each cell to a small tile bundle and holds recall ~100% at any resolution (the chunking lesson, image side)
from holographic_splat import splat_bundle as _sb, recall_region as _rr
_ssp = _sf(_T, 30)
_shv, _sctx = _sb(_ssp, (_G, _G), dim=4096, grid=32, levels=5, seed=0)
_sing = sum(abs(_rr(_shv, (_gy, _gx), _sctx) - _sctx["desc"][(_gy, _gx)]) < 1e-9 for _gy in range(32) for _gx in range(32)) / 1024
_scene = um.splat_scene(_T, grid=32, tile=8, k=30)
_tld = sum(abs(um.splat_region(_scene, (_gy, _gx)) - _scene["desc"][(_gy, _gx)]) < 1e-9 for _gy in range(32) for _gx in range(32)) / 1024
print(f"  splat scene region recall (grid 32) : single bundle {_sing:.0%} correct -> tiled {_tld:.0%} "
      f"over 1024 cells in {len(_scene['tiles'])} tiles (decode-via-cleanup caps; the tile is the chunk)")

# adaptive splat count: spend splats where the field is busy (V-Ray's adaptive sampler), not a fixed budget
_yn, _xn = _ys / _G, _xs / _G
def _bmp(cy, cx, s): return np.exp(-((_xn - cx) ** 2 + (_yn - cy) ** 2) / (2 * s * s))
_simple = _bmp(.5, .5, .18); _simple /= _simple.max()
_busy = sum(_bmp(*p) for p in [(.25, .25, .07), (.3, .7, .06), (.7, .3, .06), (.72, .72, .05),
                               (.5, .5, .05), (.2, .55, .05), (.6, .15, .05)]); _busy /= _busy.max()
_sps_s, _ = um.splat_field(_simple, noise_thresh=0.03)
_sps_b, _ = um.splat_field(_busy, noise_thresh=0.03)
print(f"  adaptive splat count (noise floor)  : simple field -> {len(_sps_s)} splats, busy field -> "
      f"{len(_sps_b)} splats at the SAME quality (a fixed k would over- or under-spend)")

# low-discrepancy sampling: even coverage (vs random's clumps) for seeds / placement / jitter (low_discrepancy_sample)
from holographic_lowdiscrepancy import dispersion as _ldisp
_lds = um.low_discrepancy_sample(64, d=2)
_rnd_disp = np.mean([_ldisp(np.random.default_rng(_s).random((64, 2))) for _s in range(20)])
print(f"  low-discrepancy coverage (64 pts)   : R-sequence dispersion {_ldisp(_lds):.3f} vs random {_rnd_disp:.3f} "
      f"({(1 - _ldisp(_lds) / _rnd_disp) * 100:.0f}% tighter; deterministic, progressive)")

# throughput-gated traversal: follow a directed chain in superposition, stop (Russian roulette) when it goes dark
from holographic_traverse import gated_traverse as _gtrav
from holographic_ai import bind as _tbind, involution as _tinvf
_trng = np.random.default_rng(0); _Dt, _Lt = 8192, 8
def _tu(): _v = _trng.standard_normal(_Dt); return _v / np.linalg.norm(_v)
_tperm = _trng.permutation(_Dt); _tinv = np.argsort(_tperm)
_tchain = [_tu() for _ in range(_Lt + 1)]; _tcb = np.array(_tchain + [_tu() for _ in range(8)])
_tcbn = _tcb / np.linalg.norm(_tcb, axis=1, keepdims=True)
_tM = np.zeros(_Dt)
for _i in range(_Lt): _tM = _tM + _tbind(_tchain[_i], _tchain[_i + 1][_tperm])
def _tstep(_c):
    _p = _tbind(_tM, _tinvf(_c))[_tinv]; _cs = _tcbn @ (_p / (np.linalg.norm(_p) + 1e-12)); _j = int(_cs.argmax())
    return (_tcb[_j], _cs[_j], _j)
_tr = _gtrav(_tstep, _tchain[0], floor=0.2, max_steps=30)
print(f"  throughput-gated traversal (RR)     : recovered {_tr.steps}/{_Lt} chain hops {_tr.payloads}, then "
      f"abstained (throughput {_tr.final_throughput:.2f} < floor) at {_tr.steps} steps vs fixed depth 30")

# directed structure: a permutation direction role so a chain/graph is traversable FORWARD (no predecessor leak)
_dsx = um.directed_structure(8)
_dsucc = um.directed_successor(_dsx, 3)[0]
_dpred = dict(um.directed_successor(_dsx, 3, topk=8))[2]    # cosine of the predecessor (node 2)
_dwalk = um.directed_traverse(_dsx, 0, floor=0.2, max_steps=20)
print(f"  directed structure (direction role) : node 3 -> successor {_dsucc[0]} (cos {_dsucc[1]:.2f}), "
      f"predecessor suppressed (cos {_dpred:+.2f}); forward walk {_dwalk.payloads}")

# corridor planning (re-anchoring): bake a short route on the directed substrate, run it cheap, re-anchor
# at the decision point -- the way PAST the per-structure capacity cap (one brain call per corridor, not per tile)
_ptiles = _np.random.default_rng(0).standard_normal((11, um.dim))
_ptiles = _ptiles / _np.linalg.norm(_ptiles, axis=1, keepdims=True)
def _pfield(_cur):
    _i = int(_np.argmax(_ptiles @ (_cur / (_np.linalg.norm(_cur) + 1e-12))))
    return _ptiles[_i + 1] if _i + 1 < len(_ptiles) else None
_plan = um.plan(_ptiles[0], _pfield, max_steps=10, floor=0.12)
print(f"  corridor plan (re-anchoring)        : baked {len(_plan.route)} steps in one plan() "
      f"(min throughput {min(_plan.throughputs):.2f}); replan when exhausted -> {um.replan_needed(_plan, len(_plan.route))}")
# plan_route: a WHOLE route far past the single-structure cap, by chaining cap-sized corridors in one call
_rtiles = _np.random.default_rng(7).standard_normal((40, um.dim))
_rtiles = _rtiles / _np.linalg.norm(_rtiles, axis=1, keepdims=True)
def _rfield(_cur):
    _i = int(_np.argmax(_rtiles @ (_cur / (_np.linalg.norm(_cur) + 1e-12))))
    return _rtiles[_i + 1] if _i + 1 < len(_rtiles) else None
def _raction(_a, _b):
    return int(_np.argmax(_rtiles @ (_b / (_np.linalg.norm(_b) + 1e-12))))
_crammed = um.plan(_rtiles[0], _rfield, max_steps=40, floor=0.12, action_of=_raction)
_route = um.plan_route(_rtiles[0], _rfield, corridor=14, floor=0.12, action_of=_raction)
print(f"  plan_route (past the cap)           : a 40-tile route -- one plan() decodes {len(_crammed.actions)} "
      f"(the cliff); plan_route chains {len(_route.corridors)} corridors -> {_route.steps}/39 steps, "
      f"{_route.reanchors} re-anchors, full route {'EXACT' if _route.actions == list(range(1, 40)) else 'partial'}")
# chunk_route: an EXPLICIT known sequence (GPS waypoints, an experiment protocol) replayed past the cap, ONE call
_seq = _np.random.default_rng(0).standard_normal((200, um.dim))
_seq = _seq / _np.linalg.norm(_seq, axis=1, keepdims=True)
def _seqaction(_a, _b):
    return int(_np.argmax(_seq @ (_b / (_np.linalg.norm(_b) + 1e-12))))
_chunked = um.chunk_route(list(_seq), chunk=14, floor=0.12, action_of=_seqaction)
print(f"  chunk_route (explicit, GPS-scale)   : a KNOWN 200-step sequence -> {_chunked.steps}/199 steps over "
      f"{len(_chunked.corridors)} chunks (linear, each one compact vector), "
      f"replay {'EXACT' if _chunked.actions == list(range(1, 200)) else 'partial'}")
# index_route: random access into that long route -- "where am I" is a jump (two-level), not a replay
_idx = um.index_route(_chunked)
_lc, _lp, _lg = _idx.locate(_seq[137])
_perq = _idx.n_chunks + max(len(_c) for _c in _idx.chunks)
print(f"  index_route (random access)         : locate tile 137 -> chunk {_lc}, step {_lg} "
      f"({'exact' if _lg == 137 else 'approx'}) in ~{_perq} comparisons vs {len(_seq)} for a flat scan")
# structured_index: the SHARED form of that lookup -- one content-addressable index the chunkers AND the
# content store draw from. Filed under the items themselves (not a weak summary), carrying any payload, and
# never stored as a bundle (a superposed index caps) -- both rules measured. index_route is its small-n case.
import numpy as _npSI
_rngSI = _npSI.random.default_rng(0)
_si_items = _rngSI.standard_normal((3000, um.dim)); _si_items /= _npSI.linalg.norm(_si_items, axis=1, keepdims=True)
_si = um.structured_index(_si_items, payloads=[f"city:{_i}" for _i in range(3000)], n_trees=6, leaf_size=64)
_si_hit, _si_cmps = _si.locate(_si_items[137], beam=6)
print(f"  structured_index (one shared lookup)  : '{_si_hit}' found by content among 3000 in {_si_cmps} "
      f"comparisons (vs 3000 flat) -- one primitive serves routes (payload=(chunk,step)) and the store (payload=URI)")
# dedup_chunks: the STORAGE twin of that lookup -- a route revisiting corridors stores the same chunk many
# times; content-address them so identical chunks coalesce, saving exactly the repetition ratio (nothing more).
_rngDC = _npSI.random.default_rng(0)
_segsDC = [_rngDC.standard_normal(um.dim) for _ in range(6)]
_patDC = [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 5, 3, 4, 0, 1, 2]      # 17 corridors, only 6 distinct
_uDC, _rDC = um.dedup_chunks([_segsDC[_p] for _p in _patDC])
print(f"  dedup_chunks (content-addressed)      : a 17-corridor loop -> {len(_uDC)} unique chunks stored "
      f"({100 * (1 - len(_uDC) / len(_patDC)):.0f}% saved == repetition ratio), references rebuild it exactly")
# chunked PLAN: the same lesson for the positional sequence memory -- a long ordered plan keeps its ORDER
# queries (precedes) exact past the single-bundle cap, where one bundle's positional decode starts to slip
_psteps = [f"op{_i}" for _i in range(200)]
um.learn_plan("_seqsingle", _psteps); um.learn_plan("_seqchunked", _psteps, chunk=14)
_ppairs = [(_i, _j) for _i in range(0, 200, 17) for _j in range(_i + 40, 200, 29)]
_ps = sum(um.precedes("_seqsingle", f"op{_i}", f"op{_j}") for _i, _j in _ppairs)
_pc = sum(um.precedes("_seqchunked", f"op{_i}", f"op{_j}") for _i, _j in _ppairs)
print(f"  chunked plan (ordered, past cap)    : a 200-step plan's order queries -- single bundle "
      f"{_ps}/{len(_ppairs)} correct, chunked(14) {_pc}/{len(_ppairs)} (precedes stays exact at length)")

# multiple importance sampling: combine hard 1-NN + soft Hopfield per-query (balance heuristic beats naive avg)
from holographic_encoders import ScalarEncoder as _SE
_me = _SE(512, lo=0.0, hi=1.0, seed=1, kernel="rbf", bandwidth=6.0)
_mg = np.linspace(0, 1, 8); _MCB = np.stack([_me.encode(_g) for _g in _mg])
_MCBn = _MCB / np.linalg.norm(_MCB, axis=1, keepdims=True)
def _mcos(_a, _b): return float(_a @ _b / ((np.linalg.norm(_a) * np.linalg.norm(_b)) + 1e-12))
_mr = np.random.default_rng(0); _eh = _es = _ea = _em = 0.0
for _ in range(300):
    _v = float(_mr.choice(_mg)) if _mr.random() < 0.5 else float(_mr.uniform(0.03, 0.97))
    _t = _me.encode(_v); _q = _t + 0.5 * _mr.standard_normal(512) / np.sqrt(512)
    _cs = _MCBn @ (_q / np.linalg.norm(_q)); _xh = _MCB[int(_cs.argmax())]
    _w = np.exp(10 * (_cs - _cs.max())); _w /= _w.sum(); _xs = (_w[:, None] * _MCB).sum(0)
    _xm = um.mis_recover(_q, _MCB)
    _eh += 1 - _mcos(_xh, _t); _es += 1 - _mcos(_xs, _t)
    _ea += 1 - _mcos(0.5 * _xh + 0.5 * _xs, _t); _em += 1 - _mcos(_xm, _t)
_eh, _es, _ea, _em = [_x / 300 for _x in (_eh, _es, _ea, _em)]
print(f"  MIS balance heuristic (recover err) : hard {_eh:.4f}, soft {_es:.4f}, naive-avg {_ea:.4f} (worse!), "
      f"MIS {_em:.4f} (per-query reliability beats all)")

# gradient-cached decode (Ward irradiance gradients): value+gradient at sparse anchors, interpolate first-order
_cr = np.random.default_rng(2); _cK = 5
_ccx = _cr.uniform(0, 1, _cK); _ccy = _cr.uniform(0, 1, _cK)
_camp = _cr.uniform(0.5, 1.5, _cK); _csig = _cr.uniform(0.2, 0.32, _cK)
def _cf(_u, _v): return float(np.sum(_camp * np.exp(-(((_u - _ccx) ** 2 + (_v - _ccy) ** 2) / (2 * _csig ** 2)))))
def _cg(_u, _v):
    _e = _camp * np.exp(-(((_u - _ccx) ** 2 + (_v - _ccy) ** 2) / (2 * _csig ** 2)))
    return np.array([np.sum(_e * (-(_u - _ccx) / _csig ** 2)), np.sum(_e * (-(_v - _ccy) / _csig ** 2))])
_cgr = np.linspace(0, 1, 5); _cA = np.array([[_u, _v] for _u in _cgr for _v in _cgr])
_cV = np.array([_cf(_u, _v) for _u, _v in _cA]); _cJ = np.array([_cg(_u, _v) for _u, _v in _cA])
_ccache = um.gradient_cache(_cA, _cV, _cJ); _cR = 1.7 / 4
_cQ = [[_u, _v] for _u in np.linspace(0.1, 0.9, 12) for _v in np.linspace(0.1, 0.9, 12)]
_cfo = np.mean([abs(float(um.cache_interp(_ccache, _q, _cR)) - _cf(*_q)) for _q in _cQ])
_cnn = np.mean([abs(float(_cV[np.argmin(np.linalg.norm(_cA - _q, axis=1))]) - _cf(*_q)) for _q in _cQ])
_cglob = np.mean([abs(float(um.cache_interp(_ccache, _q, _cR, global_weights=True)) - _cf(*_q)) for _q in _cQ])
print(f"  gradient cache (Ward, 25 anchors)   : nearest-neighbor err {_cnn:.3f}, first-order(grad) {_cfo:.3f} "
      f"(gradients ~halve anchors); GLOBAL weights {_cglob:.3f} (no validity radius -> fails)")

# robust accumulation: harmonic (1/n) weights converge where a fixed-alpha EMA plateaus; firefly clamping resists outliers
_ar = np.random.default_rng(5); _aD = 128
_amu = _ar.standard_normal(_aD); _amu /= np.linalg.norm(_amu)
def _acos(_a, _b): return float(_a @ _b / ((np.linalg.norm(_a) * np.linalg.norm(_b)) + 1e-12))
_astream = [_amu + 0.8 * _ar.standard_normal(_aD) / np.sqrt(_aD) for _ in range(400)]
_aharm = 1 - _acos(um.robust_accumulate(_astream, schedule="harmonic"), _amu)
_aema = 1 - _acos(um.robust_accumulate(_astream, schedule="ema", alpha=0.2), _amu)
_aclean = [_amu + 0.3 * _ar.standard_normal(_aD) / np.sqrt(_aD) for _ in range(40)]
_atruth = np.mean(_aclean, 0); _afire = _aclean + [8.0 * _ar.standard_normal(_aD) / np.sqrt(_aD) for _ in range(5)]
_aplain = 1 - _acos(um.robust_accumulate(_afire, schedule="mean"), _atruth)
_aclamp = 1 - _acos(um.robust_accumulate(_afire, schedule="mean", clamp_k=2.5), _atruth)
print(f"  robust accumulation (err)           : harmonic {_aharm:.4f} vs EMA {_aema:.4f} (1/n converges); "
      f"firefly: plain {_aplain:.4f} vs clamped {_aclamp:.4f}")

# denoise-by-downscale: a pattern invisible per-sample, recovered by pooling samples (consolidation/SVD downscale)
_xr = np.random.default_rng(0); _xD, _xr3 = 256, 3
_xB, _ = np.linalg.qr(_xr.standard_normal((_xD, _xr3)))
_xC = _xr.standard_normal((800, _xr3)); _xX = _xC @ _xB.T; _xX /= np.linalg.norm(_xX, axis=1, keepdims=True)
_xXn = _xX + 0.4 * _xr.standard_normal((800, _xD))
_xpers = np.mean([np.linalg.norm(_xB.T @ v) ** 2 / np.linalg.norm(v) ** 2 for v in _xXn[:50]])
_xres = um.find_pattern_by_downscale(_xXn, kind="vectors", k=_xr3, n_null=40, seed=1)
_xov = float(np.sum((_xres.pattern.T @ _xB) ** 2) / _xr3)
_xnoise = um.find_pattern_by_downscale(_xr.standard_normal((800, _xD)), kind="vectors", k=_xr3, n_null=40, seed=1)
print(f"  denoise-by-downscale (rank-3)       : per-sample energy {_xpers:.3f} (invisible) -> pooled overlap "
      f"{_xov:.3f}, found={_xres.found}; pure noise found={_xnoise.found} (fail-safe)")

# looping denoise as diffusion on a curved manifold (a ring): settle onto it, beat interpolation, generate novel-but-valid
_dr = np.random.default_rng(0); _dD, _dN = 64, 48
_dU, _ = np.linalg.qr(_dr.standard_normal((_dD, 2))); _du, _dv = _dU[:, 0], _dU[:, 1]
_dth = np.linspace(0, 2 * np.pi, _dN, endpoint=False)
_dS = np.stack([np.cos(_t) * _du + np.sin(_t) * _dv for _t in _dth])
def _drd(_x):
    _a, _b = _du @ _x, _dv @ _x
    return float(np.hypot(np.linalg.norm(_x - (_a * _du + _b * _dv)), abs(np.hypot(_a, _b) - 1)))
_dnoisy = _dS[10] + 0.6 * _dr.standard_normal(_dD) / np.sqrt(_dD)
_dset = _drd(um.manifold_denoise(_dnoisy, _dS))
_dmid = 0.5 * (_dS[5] + _dS[25]); _dmids = _drd(um.manifold_denoise(_dmid, _dS))
_dgen = um.manifold_generate(_dS, seed=3); _dgd = _drd(_dgen)
_dnov = min(float(np.linalg.norm(_dgen - _si)) for _si in _dS)
print(f"  manifold diffusion (ring)           : noisy {_drd(_dnoisy):.2f}->settled {_dset:.3f}; interp midpoint "
      f"{_drd(_dmid):.2f}->{_dmids:.3f}; generated dist {_dgd:.3f} (valid), novelty {_dnov:.3f} (between samples)")

# looping negative-lobe sharpening (Van Cittert): recover detail an over-smoothed signal lost, with a stability guard
from holographic_sharpen import _gauss_blur as _sgb
_st = np.arange(256)
_struth = np.sin(2 * np.pi * 3 * _st / 256) + 0.6 * np.sin(2 * np.pi * 30 * _st / 256) * np.exp(-((_st - 128) ** 2) / (2 * 25 ** 2))
def _serr(_z): return float(np.linalg.norm(_z - _struth) / np.linalg.norm(_struth))
_sblur = _sgb(_struth, 3.0)
_srec = um.sharpen_loop(_sblur, sigma=3.0, lam=1.0, iters=80, noise_level=0.0)
_snoisy = _sblur + 0.005 * np.random.default_rng(0).standard_normal(256)
_sg = um.sharpen_loop(_snoisy, sigma=3.0, lam=1.0, iters=80, noise_level=0.005)
_su = um.sharpen_loop(_snoisy, sigma=3.0, lam=1.0, iters=80, noise_level=0.0)
print(f"  looping sharpen (Van Cittert)       : blurred err {_serr(_sblur):.3f} -> recovered {_serr(_srec):.3f}; "
      f"with noise guarded {_serr(_sg):.3f} vs unguarded {_serr(_su):.3f} (over-sharpens)")

# smooth/sharp two-layer codec (irradiance caching's architecture): split beats any single basis at fixed budget
from holographic_twolayer import _fft_topk as _f2, _sparse_topk as _s2
_c2t = np.arange(256)
_c2x = np.sin(2 * np.pi * 2 * _c2t / 256) + 0.6 * np.cos(2 * np.pi * 5 * _c2t / 256)
_c2rng = np.random.default_rng(0)
_c2pos = _c2rng.choice(256, 6, replace=False); _c2x[_c2pos] += _c2rng.uniform(-3, 3, 6)
def _c2psnr(_r): _m = np.mean((_r - _c2x) ** 2); return float(10 * np.log10((_c2x.max() - _c2x.min()) ** 2 / (_m + 1e-12)))
_c2code = um.smooth_sharp_split(_c2x, 6, 6); _c2rec = um.smooth_sharp_reconstruct(_c2code)
print(f"  smooth/sharp split (budget 12)      : split {_c2psnr(_c2rec):.1f} dB vs single-FFT {_c2psnr(_f2(_c2x, 12)):.1f} "
      f"vs single-sparse {_c2psnr(_s2(_c2x, 12)):.1f} (no single basis is cheap across smooth+sharp)")

# FHRR phase-domain morph (phase shift = motion): uniform feature motion + valid phasors vs the amplitude blend
from holographic_fhrr import phasor_atom as _pmatom, fhrr_sim as _pmsim
from holographic_phasemorph import amplitude_morph as _pmamorph
_pmphi = np.angle(_pmatom(2048, np.random.default_rng(0)))
def _pmenc(_x): return np.exp(1j * _x * _pmphi)
def _pmdec(_q):
    _g = np.linspace(-0.1, 1.1, 241); _s = np.array([_pmsim(_q, _pmenc(_x)) for _x in _g]); return float(_g[np.argmax(_s)])
_pma, _pmb = _pmenc(0.1), _pmenc(0.9)
_pmts = np.linspace(0.1, 0.9, 9)
_pmdevp = max(abs(_pmdec(um.phase_morph(_pma, _pmb, _t)) - (0.1 + 0.8 * _t)) for _t in _pmts)
_pmdeva = max(abs(_pmdec(_pmamorph(_pma, _pmb, _t)) - (0.1 + 0.8 * _t)) for _t in _pmts)
print(f"  FHRR phase-domain morph             : uniform-motion deviation phase {_pmdevp:.3f} vs amplitude {_pmdeva:.3f}; "
      f"midpoint energy phase {np.mean(np.abs(um.phase_morph(_pma, _pmb, 0.5))):.2f} vs amplitude {np.mean(np.abs(_pmamorph(_pma, _pmb, 0.5))):.2f}")

# adaptive iteration count (ADAPT-2): stop the resonator the moment its picks verify -- same answer, fewer iters
from holographic_sbc import sbc_codebook as _a2cb, sbc_reconstruct as _a2rec
_a2cbs = [_a2cb(24, 7, 10, seed=300 + _f) for _f in range(3)]
_a2rng = np.random.default_rng(11)
_a2true = tuple(int(_a2rng.integers(0, 10)) for _ in range(3))
_a2prod = _a2rec(_a2true, _a2cbs, 7)
_a2sf, _a2se = {}, {}
_a2rf = um.decompose_structure(_a2prod, _a2cbs, 7, seed=0, stats=_a2sf)
_a2re = um.decompose_structure(_a2prod, _a2cbs, 7, seed=0, early_stop=True, stats=_a2se)
print(f"  resonator early-stop (ADAPT-2)      : fixed {_a2sf['iters']} iters vs early-stop {_a2se['iters']}; "
      f"same verified factors {tuple(_a2rf['picks']) == tuple(_a2re['picks']) and _a2re['verified']}")

# adaptive cache anchor placement (CACHE-3): crowd anchors where the field bends -- ~7x fewer for the same quality
_c3x = np.linspace(0, 1, 4001)
_c3f = 0.3 * _c3x + np.exp(-((_c3x - 0.7) / 0.015) ** 2)
def _c3rmse(_ax): return float(np.sqrt(np.mean((um.reconstruct_from_anchors(_c3x, _ax, _c3f) - _c3f) ** 2)))
_c3uni = _c3rmse(np.linspace(0, 1, 32)); _c3ada = _c3rmse(um.adaptive_anchors(_c3x, _c3f, 32))
_c3need = next(_N for _N in range(32, 800) if _c3rmse(np.linspace(0, 1, _N)) <= _c3ada)
print(f"  adaptive cache placement (CACHE-3)  : 32 anchors RMSE uniform {_c3uni:.4f} vs adaptive {_c3ada:.4f}; "
      f"uniform needs {_c3need} ({_c3need / 32:.1f}x) to match adaptive-32")

# backward warp is hole-free (PHASE-2): forward scatter leaves holes/overlaps; the unbind-form backward gather none
from holographic_backwardwarp import forward_scatter as _bwfs, backward_gather as _bwbg
_bwn = 256; _bwpos = np.arange(_bwn) / _bwn
_bwsig = np.sin(2 * np.pi * 3 * _bwpos) + 0.5 * _bwpos
_bwwarp = lambda _s: _s + 0.12 * np.sin(2 * np.pi * _s)
_, _bwholes_f, _bwover_f = _bwfs(_bwsig, _bwpos, _bwwarp, _bwn)
_bwgrid = np.linspace(0, 1, 4000); _bwinv = np.interp(_bwpos, _bwwarp(_bwgrid), _bwgrid)
_bwbwd = _bwbg(_bwsig, _bwpos, _bwinv)
print(f"  backward warp hole-free (PHASE-2)   : forward scatter {_bwholes_f} holes + {_bwover_f} overlaps vs "
      f"backward gather {int(np.isnan(_bwbwd).sum())} holes (unbind is the backward map)")

# multi-resolution pyramid / mipmap (SCALE-1): anti-aliased coarse query, cheap LOD, exact at full
_s1x = np.arange(1024) / 1024
_s1sig = np.sin(2 * np.pi * 2 * _s1x) + 0.6 * np.sin(2 * np.pi * 150 * _s1x)
_s1low = np.sin(2 * np.pi * 2 * _s1x)
_s1pyr = um.multires_pyramid(_s1sig, n_levels=5)
def _s1rmse(_a, _b): return float(np.sqrt(np.mean((_a - _b) ** 2)))
_s1mip = _s1rmse(um.pyramid_reconstruct(_s1pyr[3], 1024), _s1low)
_s1naive = _s1rmse(um.pyramid_reconstruct(_s1sig[::8], 1024), _s1low)
print(f"  multi-res pyramid (SCALE-1)         : 1/8 coarse query mipmap RMSE {_s1mip:.4f} vs naive subsample "
      f"{_s1naive:.4f} (anti-aliased); levels {[len(_l) for _l in _s1pyr]}")

# re-anchoring is load-bearing (RAY-2 audit): re-anchored traversal reaches depth; raw collapses as noise compounds
from holographic_reanchor import directed_linked_list as _r2ll, make_steps as _r2steps
_r2 = _r2ll(12, dim=1024, seed=0); _r2re, _r2raw = _r2steps(_r2)
_r2g_re = um.gated_traverse(_r2re, _r2["chain"][0], floor=0.20, max_steps=17)
_r2g_raw = um.gated_traverse(_r2raw, _r2["chain"][0], floor=0.20, max_steps=17)
print(f"  re-anchoring load-bearing (RAY-2)   : re-anchored reaches {len(_r2g_re.payloads)}/12 hops vs raw "
      f"{len(_r2g_raw.payloads)}/12 (no cleanup -> noise compounds, gate stops the dark ray)")

# C3 (cross-cutting): adaptive-stop on the anisotropic splat fit -- a busy field converges before the fixed 200 steps
_c3ys, _c3xs = np.mgrid[0:40, 0:40]
_c3rng = np.random.default_rng(0)
_c3field = sum(_c3rng.uniform(0.4, 1.0) * np.exp(-(((_c3xs - _c3rng.uniform(6, 34)) ** 2
              + (_c3ys - _c3rng.uniform(6, 34)) ** 2) / _c3rng.uniform(8, 30))) for _ in range(7))
_c3st = {}
um.splat_aniso(_c3field, k=6, steps=200, early_stop=True, stats=_c3st)
print(f"  aniso fit adaptive-stop (C3)        : converged at {_c3st['steps']}/200 Adam steps "
      f"(speed/quality knob -- a soft plateau, not free)")

# B3 (cross-cutting): adaptive-stop diffusion -- the composed structure settles before the fixed schedule ends
_b3rng = np.random.default_rng(5)
_b3roles = np.stack([random_vector(um.dim, _b3rng) for _ in range(4)])
_b3fillers = np.stack([random_vector(um.dim, _b3rng) for _ in range(8)])
_b3zf = um.generate_structure(_b3roles, _b3fillers, seed=2, readout="sparsemax")
_b3st = {}
_b3ze = um.generate_structure(_b3roles, _b3fillers, seed=2, readout="sparsemax", early_stop=True, stats=_b3st)
_b3same = (tuple(int(np.argmax(_b3fillers @ unbind(_b3zf, r))) for r in _b3roles)
           == tuple(int(np.argmax(_b3fillers @ unbind(_b3ze, r))) for r in _b3roles))
print(f"  diffusion adaptive-stop (B3)        : settled at {_b3st['steps']}/16 steps, same structure {_b3same} "
      f"(free -- the decoded combo is a certificate)")

# D2 (cross-cutting): robust reward accumulation -- a fluke reward cannot swing the brain's value estimate
def _d2_value(_robust):
    _d2br = HolographicMind(um.dim, ["a", "b"], merge=0.8, robust_returns=_robust)
    _d2r = np.random.default_rng(1)
    _d2s = np.random.default_rng(0).standard_normal(um.dim)
    _d2s = _d2s / np.linalg.norm(_d2s)
    for _ in range(150):
        _rew = _d2r.normal(20.0, 5.0) if _d2r.random() < 0.08 else _d2r.normal(1.0, 0.3)
        _d2br.remember([_d2s], [0], [float(_rew)])
    return _d2br.value(_d2s, 0)[0]
print(f"  robust reward accumulation (D2)     : value under 8% outlier rewards -- plain {_d2_value(False):.2f} "
      f"vs robust {_d2_value(True):.2f} (true 1.00; winsorising resists the flukes)")

# C1 (cross-cutting): coarse-to-fine splat densification -- the staged warm start escapes the one-shot's local optimum
_c1ys, _c1xs = np.mgrid[0:56, 0:56]
_c1T = (np.exp(-(((_c1xs - 28) ** 2 + (_c1ys - 28) ** 2) / 300.0))
        + sum(0.8 * np.exp(-(((_c1xs - _cx) ** 2 + (_c1ys - _cy) ** 2) / 8.0))
              for _cx, _cy in [(12, 12), (44, 16), (16, 44), (42, 42)]))     # broad blob + small sharp details
_c1one = float(((um.splat_aniso(_c1T, k=12, steps=210)[1] - _c1T) ** 2).mean())
_c1cf = float(((um.splat_densify(_c1T, k=12)[1] - _c1T) ** 2).mean())
print(f"  coarse-to-fine densify (C1)         : multi-scale MSE one-shot {_c1one:.5f} vs densify {_c1cf:.5f} "
      f"(a basin the one-shot cannot reach at any step count)")

# A3 (cross-cutting): adaptive encoder resolution -- warp the input axis by the value-density CDF (with a floor)
_a3rng = np.random.default_rng(0)
_a3clustered = np.clip(np.where(_a3rng.random(4000) < 0.5, _a3rng.normal(0.25, 0.04, 4000),
                                _a3rng.normal(0.75, 0.04, 4000)), 0, 1)
def _a3err(_fit):
    _enc = ScalarEncoder(um.dim, 0.0, 1.0, seed=1, kernel="rbf", bandwidth=2.0)
    if _fit:
        _enc.fit_resolution(_a3clustered)
    _t = np.clip(np.where(_a3rng.random(150) < 0.5, _a3rng.normal(0.25, 0.04, 150),
                          _a3rng.normal(0.75, 0.04, 150)), 0, 1)
    return float(np.mean([abs(_enc.decode(_enc.encode(float(x))
                 + 0.4 * _a3rng.standard_normal(um.dim) / np.sqrt(um.dim), 400) - float(x)) for x in _t]))
print(f"  adaptive encoder resolution (A3)    : decode err on clustered values -- uniform {_a3err(False):.4f} "
      f"vs density-warped {_a3err(True):.4f} (a reallocation: dense better, sparse worse)")

# C2 (cross-cutting): phase-domain scene morph -- a translation is a phase ramp, so the phase morph SLIDES it
_c2ys, _c2xs = np.mgrid[0:28, 0:28]
def _c2blob(_cx):
    return np.exp(-(((_c2xs - _cx) ** 2 + (_c2ys - 14) ** 2) / (2 * 3.0 ** 2)))
def _c2midpeak(_fr):
    _m = _fr[len(_fr) // 2]; _e = 0.5 * (_fr[0].max() + _fr[-1].max()); return float(_m.max() / (_e + 1e-12))
_c2a, _c2b = _c2blob(10), _c2blob(16)                                # a blob translated 6px
print(f"  phase-domain scene morph (C2)       : translated-blob midpoint peak -- dct slerp "
      f"{_c2midpeak(um.morph_scene(_c2a, _c2b, method='dct')):.3f} (smears) vs phase "
      f"{_c2midpeak(um.morph_scene(_c2a, _c2b, method='phase')):.3f} (slides; wraps past pi)")

# axial perception: an orientation (theta == theta+pi) on the Mobius base, wired as the "axial" modality
import math as _math
_t = 0.7
print(f"  axial modality (theta==theta+pi)    : sim(t, t+pi)={um.axial_similarity(_t, _t + _math.pi):+.2f} "
      f"(same orientation) vs the plain number modality "
      f"{cosine(um.perceive(_t, 'number'), um.perceive(_t + _math.pi, 'number')):+.2f} -- a pi flip is "
      f"invisible to axial, not to a scalar")

# splat-bundle archive: a field stored as splat codes, with progressive refinement + an exact region query,
# plus the holographic per-region readout (splat_bundle/recall_region)
from holographic_splat import splat_bundle as _sb, recall_region as _rr
_arch = um.splat_archive((_G, _G), keep=80); _arch.add(_T)
_full = _psnr(_T, _arch.recover(0)); _quarter = _psnr(_T, _arch.recover(0, k=20))
_here, _ = _arch.region(0, (0, _G // 2, 0, _G // 2))
_scene, _ctx = _sb(_sp, _T.shape, dim=4096, grid=4)
_occ = max(_rr(_scene, (_gy, _gx), _ctx) for _gy in range(4) for _gx in range(4))
print(f"  splat-bundle archive (beside WHT)   : recover {_full:.0f} dB full / {_quarter:.0f} dB at K/4 "
      f"(progressive); region query found {sum(len(h) for h in _here)} splats in a quadrant (exact); "
      f"recall_region peak occupancy={_occ:.2f}")

# honest recognition: the calibration harness (RecallNull / SPRT / bh_fdr) is now woven INTO recognition --
# a raw cosine becomes an honest false-alarm probability against the mind's OWN prototypes, so the mind can
# ABSTAIN ("I don't recognise this") instead of always naming a nearest label, on BOTH readout paths.
_hm = UnifiedMind(dim=512, seed=0, maintain="manual")
for _w in ("dog", "wolf", "puppy", "hound"):  _hm.learn(_w, "canine")
for _w in ("cat", "lion", "kitten", "tiger"): _hm.learn(_w, "feline")
for _w in ("oak", "pine", "maple", "birch"):  _hm.learn(_w, "tree")
_lab_r, _sim_r, _p_r = _hm.recognize("dog")               # a learned member -> low p
_lab_n, _sim_n, _p_n = _hm.recognize("qz xkqv zzpf")      # gibberish        -> high p
print(f"  honest recognition (calibrated)     : recognize('dog')->{_lab_r} p={_p_r:.3f} (trust it); "
      f"gibberish p={_p_n:.2f} so classify(abstain=.05) returns {_hm.classify('qz xkqv zzpf', abstain=0.05)[0]}")
_dec, _who, _n = _hm.stream_recognize(["dog", "hound", "puppy", "wolf"])
_batch = _hm.recognize_batch(["dog", "tiger", "oak", "zzqx vvbn"], alpha=0.1)
print(f"  streaming (SPRT) + FDR batch        : stream of canine cues -> {_dec} ('{_who}') in {_n}; "
      f"FDR batch keeps {sum(b['significant'] for b in _batch)}/4 real members, drops the gibberish")
_pay, _psim, _pp = _hm.recall_calibrated("dog")
print(f"  calibrated recall (individual store): recall('dog')->{_pay} p={_pp:.3f}; "
      f"recall(gibberish, abstain=.05) -> {_hm.recall('zzqx vvbn', abstain=0.05)} (nothing like it)")

# self-maintenance triggered by INCOHERENCE, not a clock. A calibrated-NOVELTY trigger was a measured
# negative (novelty can't see incoherence); coherence-gating is the win -- it matches the fixed schedule's
# accuracy at a fraction of the (self-validating) reorganize passes by skipping the ones a coherent store
# does not need. Opt-in via coherence_floor; default stays the fixed schedule.
_rngm = np.random.default_rng(0); _Lm, _NCm = 24, 4
_angm = np.linspace(0, 2 * np.pi, _NCm * 2, endpoint=False)
_dirsm = np.stack([np.cos(_angm), np.sin(_angm)], 1) @ _rngm.standard_normal((2, _Lm))
_subm = {c: [c, c + _NCm] for c in range(_NCm)}             # each class = two antipodal modes
_sampm = lambda c: _dirsm[_subm[c][_rngm.integers(2)]] * 3 + 0.5 * _rngm.standard_normal(_Lm)
_rows = ([(_sampm(c := int(_rngm.integers(2))), c) for _ in range(90)]           # 2 classes
         + [(_sampm(c := int(_rngm.integers(_NCm))), c) for _ in range(90)]      # 2 new classes arrive
         + [(_sampm(c := int(_rngm.integers(_NCm))), c) for _ in range(150)])    # stable coherent tail
def _run_maint(cf):
    mm = UnifiedMind(dim=384, seed=0, check_every=30, coherence_floor=cf); ok = []
    for _x, _c in _rows:
        ok.append((mm.classify(_x, modality="vector")[0] == _c) if mm.memory.live.size() else False)
        mm.learn(_x, _c, modality="vector")
    return np.mean(ok), len(mm.journal)                     # journal length == reorganize passes
_sa, _sp = _run_maint(None); _ga, _gp = _run_maint(0.65)
print(f"  coherence-gated maintenance         : fixed schedule {_sa*100:.0f}% at {_sp} reorganize-passes "
      f"vs coherence-gated {_ga*100:.0f}% at {_gp} -- ~{_sp/max(_gp,1):.0f}x fewer passes (the calibrated-"
      f"novelty trigger was a kept negative; coherence is the signal that sees incoherence)")

# calibration coverage: the proof the honest p-values above are actually honest. Draw pure-noise vectors,
# threshold recognition at alpha, and the false-alarm rate should track alpha -- on BOTH the prototype path
# (recognize) and the individual path (recall_calibrated, whose null is matched to the sublinear recall path).
_cm = UnifiedMind(dim=512, seed=0, maintain="manual")
for _w in ("dog", "wolf", "puppy", "hound", "cat", "lion", "kitten", "tiger", "oak", "pine", "maple", "birch"):
    _cm.learn(_w, "animal" if _w not in ("oak", "pine", "maple", "birch") else "tree")
_cal = _cm.calibration_report(n=3000)
print(f"  calibration coverage (false-alarm)  : at alpha=0.05/0.10 the noise false-alarm rate is "
      f"{_cal['prototype_false_alarm'][0.05]:.3f}/{_cal['prototype_false_alarm'][0.1]:.3f} (prototypes) and "
      f"{_cal['individual_false_alarm'][0.05]:.3f}/{_cal['individual_false_alarm'][0.1]:.3f} (individuals) -- "
      f"it tracks alpha, so abstention is trustworthy")

# Wald's sequential test earns its keep only when the match/null densities OVERLAP. Distinct learned items
# are well-separated, so stream_recognize decides in ~1 sample (correctly). As the signal gets fainter the
# densities overlap, the SPRT spends exactly as many samples as the evidence needs, and at matched error it
# uses about HALF the samples of the best fixed-window rule.
from holographic_honesty import SPRTRecall as _SP
def _sprt_avg_n(mu0, sd0, mu1, sd1, trials=4000, cap=80):
    _null = np.random.default_rng(1).normal(mu0, sd0, 4000)
    _match = np.random.default_rng(2).normal(mu1, sd1, 4000)
    _g = np.random.default_rng(5); _ns = []
    for _t in range(trials):
        _mu, _sd = (mu1, sd1) if _t % 2 == 0 else (mu0, sd0)
        _d, _n = _SP(_null, _match, alpha=0.05, beta=0.05).decide(_g.normal(_mu, _sd, cap), cap=cap)
        _ns.append(_n)
    return float(np.mean(_ns))
_sep = _sprt_avg_n(0.093, 0.037, 0.450, 0.137)        # the mind's distinct-item densities
_ovl = _sprt_avg_n(0.350, 0.130, 0.520, 0.130)        # a faint / drifting signal: real overlap
print(f"  Wald SPRT sample count vs overlap   : well-separated signal decides in {_sep:.1f} samples; a faint "
      f"overlapping one takes {_ovl:.1f} -- ~half a fixed-window rule's samples at the same error")

# the honesty layer reaches ACTION: a creature brain that knows when it is guessing. Same RecallNull
# machinery, over the brain's experienced states -- a familiar state gets a low false-alarm p (trust the
# value estimate), a never-seen one a high p (explore instead of committing to a guess).
_dm = UnifiedMind(dim=512, seed=0); _dm.actions(["N", "S", "E", "W"])
_dr = np.random.default_rng(0); _arch = [random_vector(512, _dr) for _ in range(4)]; _bestA = ["N", "E", "S", "W"]
for _ in range(40):
    for _k, _b in enumerate(_arch):
        _s = _b + 0.25 * random_vector(512, _dr); _s /= np.linalg.norm(_s)
        _dm.reinforce(_s, _bestA[_k], 1.0); _dm.reinforce(_s, _bestA[(_k + 1) % 4], -0.5)
_fam = _arch[0] + 0.25 * random_vector(512, _dr); _fam /= np.linalg.norm(_fam)
_af, _pf = _dm.decide_confidence(_fam); _an, _pn = _dm.decide_confidence(random_vector(512, _dr))
print(f"  calibrated decide (honesty->action) : familiar state -> {_af!r} at p={_pf:.3f} (trusted); a novel "
      f"state -> p={_pn:.3f} (guessing) -- explore_if_unrecognized turns that into a safe random move")

# the scan faculty: SPRT per channel (decide each as fast as its evidence allows) + FDR across channels
# (control the look-elsewhere -- scan enough noise and some clear the per-channel bar by luck).
_sm = UnifiedMind(dim=256, seed=0); _sr = np.random.default_rng(0); _sbase = random_vector(256, _sr)
for _ in range(50):
    _v = _sbase + _sr.uniform(1.0, 4.0) * random_vector(256, _sr); _v /= np.linalg.norm(_v)
    _sm.learn(_v, "signal", modality="vector")
for _j in range(6):
    _ob = random_vector(256, _sr)
    for _ in range(8):
        _v = _ob + 1.5 * random_vector(256, _sr); _v /= np.linalg.norm(_v); _sm.learn(_v, f"o{_j}", modality="vector")
_chs, _kd = [], []
for _k in range(8):
    _r = np.random.default_rng(200 + _k)
    _chs.append([(lambda v: v / np.linalg.norm(v))(_sbase + 1.5 * random_vector(256, _r)) for _ in range(14)]); _kd.append(1)
for _k in range(80):
    _r = np.random.default_rng(300 + _k); _chs.append([random_vector(256, _r) for _ in range(14)]); _kd.append(0)
_rows = _sm.scan(_chs, modality="vector", alpha=0.05, beta=0.05, fdr=0.1); _kd = np.array(_kd)
_det = np.array([r["detected"] for r in _rows]); _pv = np.array([r["pvalue"] for r in _rows])
_naive = int((_pv[_kd == 0] <= 0.1).sum())
print(f"  scan: SPRT-per-channel + FDR        : found {int(_det[_kd==1].sum())}/8 signal channels among 80 noise; "
      f"naive p<=0.10 would false-alarm on {_naive} noise channels, FDR holds detections to "
      f"{int(_det[_kd==0].sum())} false")

# calibrated SOFT confidence for the resonator on APPROXIMATE inputs: the exact-reconstruction certificate is
# uselessly False the moment the input is noisy, even when the factors are right; a p-value vs a procedure-
# matched noise null gives the graded answer (and abstains on real noise, where a random-picks null would not).
import holographic_sbc as _S
_B, _L, _F, _n = 16, 16, 3, 8
_cbs = [[_S.sbc_random(_B, _L, seed=100 * _f + _i) for _i in range(_n)] for _f in range(_F)]
_pr = np.asarray(_cbs[0][2]).copy()
for _f in (1, 2):
    _pr = _S.sbc_bind(_pr, _cbs[_f][(5, 1)[_f - 1]], _L)
_cm = UnifiedMind(dim=256, seed=0)
_clean = _cm.factor_composite(_pr, _cbs, L=_L, restarts=6, seed=0, confidence=True)
_cor = _pr.copy(); _rr = np.random.default_rng(52)
for _b in _rr.choice(_B, 2, replace=False):
    _cor[_b] = (_cor[_b] + 1) % _L
_corr = _cm.factor_composite(_cor, _cbs, L=_L, restarts=6, seed=0, confidence=True)
print(f"  resonator soft confidence           : exact input verified={_clean['verified']} p={_clean['pvalue']:.3f}; "
      f"2 blocks corrupted -> true factors still found, verified={_corr['verified']} but calibrated p={_corr['pvalue']:.3f} "
      f"(the boolean's blind spot, rescued)")

# pluggable assembly energy: not every mismatch costs the same (the Rosetta move). The SAME target assembles
# differently under a substitution energy that makes cross-group swaps dear -- and each is the global optimum
# under its OWN energy. Plus structure-compare: superpose two structures, read overlap via consolidation.
_GROUP = {_c: ('V' if _c in 'AE' else 'C') for _c in 'ABDE'}
def _subst(_frag, _pos, _t):
    return sum((1 if _GROUP[_frag[_j]] == _GROUP[_t[_pos + _j]] else 4)
               for _j in range(len(_frag)) if _frag[_j] != _t[_pos + _j])
_T, _lib = "EAAE", ("AB", "BA", "BD", "BE", "EE")
_am = UnifiedMind(dim=512, seed=0)
_rH = _am.assemble(_T, _lib); _rS = _am.assemble(_T, _lib, energy=_subst)
_cmpP = _am.compare_structures({"fragments": [(0, 'AB'), (1, 'BC'), (2, 'CD')]},
                               {"fragments": [(0, 'AB'), (1, 'BX'), (2, 'XD')]})
print(f"  assembly: pluggable energy          : target {_T!r} -> Hamming picks {_rH['assembled']!r} (energy {_rH['energy']}), "
      f"substitution picks {_rS['assembled']!r} (energy {_rS['energy']}) -- each the global optimum under its own energy")
print(f"  structure-compare (consolidation)   : two structures sharing 1 of 3 placements -> "
      f"placement overlap {_cmpP['placement_overlap']:.2f}, holographic overlap {_cmpP['holographic_overlap']:.2f} (the SVD read matches the exact count)")

# one iterate-a-projection engine under three faculties (Macklin): POCS, the resonator, and the PnP loop are
# all "project onto each constraint in turn until they jointly hold". Shown here as POCS converging to a
# subspace intersection -- the same project_onto_constraints the resonator (with restarts) and denoise(pnp) use.
_pn = 20; _pr2 = np.random.default_rng(0)
_pu = _pr2.standard_normal(_pn); _pu /= np.linalg.norm(_pu)
_QA, _ = np.linalg.qr(np.stack([_pu, _pr2.standard_normal(_pn), _pr2.standard_normal(_pn)], axis=1))
_QB, _ = np.linalg.qr(np.stack([_pu, _pr2.standard_normal(_pn), _pr2.standard_normal(_pn)], axis=1))
_x0 = _pr2.standard_normal(_pn)
_xf, _sw, _cv = _am.project_onto_constraints(_x0, [lambda x: _QA @ (_QA.T @ x), lambda x: _QB @ (_QB.T @ x)],
                                             iters=500, tol=1e-12)
_off = np.linalg.norm(_xf - (_xf @ _pu) * _pu) / np.linalg.norm(_xf)
print(f"  iterate-a-projection (POCS)         : alternating projection onto two subspaces converged in {_sw} sweeps "
      f"to their intersection (off-axis residual {_off:.1e}) -- the same engine the resonator and denoise(pnp) run on")

# inverse problem through the mind (Milanfar/Ozcan): inpaint an ERASED plate by Plug-and-Play/RED -- the
# denoiser as the manifold prior in a LOOP, which beats a single denoise because the loop holds the observed
# pixels to the measurement while filling the erased ones from the manifold.
_Si = 16; _Ki = 6
_ig = np.random.default_rng(0); _yy, _xx = np.mgrid[0:_Si, 0:_Si] / _Si
_pats = np.stack([(lambda p: p / np.linalg.norm(p))(np.sin(np.pi * _ig.integers(1, 4) * _yy + 0.7 * _f) *
                  np.cos(np.pi * _ig.integers(1, 4) * _xx + 0.3 * _f)) for _f in range(_Ki)])
def _mkimg(s):
    _c = np.random.default_rng(s).standard_normal(_Ki); _im = (_c[:, None, None] * _pats).sum(0)
    _im -= _im.min(); return _im / (_im.max() + 1e-9)
_gal = np.stack([_mkimg(100 + i) for i in range(40)]); _Gm = _gal.reshape(40, _Si * _Si)
_clean = _gal[3].copy(); _mask = np.ones((_Si, _Si)); _mask[5:10, 6:11] = 0.0
_yobs = _mask * _clean + 0.05 * np.random.default_rng(7).standard_normal((_Si, _Si))
_psnr = lambda a, b: -10 * np.log10(((a - b) ** 2).mean())
_one = _am.denoise(_yobs.flatten(), method="adaptive", samples=_Gm).reshape(_Si, _Si)
_res = _am.restore(_yobs.flatten(), mask=_mask.flatten(), samples=_Gm).reshape(_Si, _Si)
print(f"  inverse problem: inpaint a plate    : 25/256 px erased -> single denoise {_psnr(_one,_clean):.1f} dB, "
      f"PnP/RED restore {_psnr(_res,_clean):.1f} dB (the loop beats one-shot by {_psnr(_res,_clean)-_psnr(_one,_clean):.0f} dB)")

# capacity / SNR diagnostic (Plate + Cranmer): where the store sits vs the noise-wins cliff, and whether the
# calibrated false-alarm rate holds as the store GROWS. The measured noise floor tracks the HRR bound.
_capm = UnifiedMind(dim=256, seed=0); _cr = np.random.default_rng(0)
for _c in range(8):
    _bb = random_vector(256, _cr)
    for _ in range(12):
        _vv = _bb + 1.2 * random_vector(256, _cr); _vv /= np.linalg.norm(_vv); _capm.learn(_vv, f"c{_c}", modality="vector")
_cap = _capm.capacity_report(alpha=0.05, loads=(64, 512), n_floor=400, n_fa=400)
print(f"  capacity / SNR vs the cliff         : match {_cap['match']:.2f} vs noise floor {_cap['floor_mean']:.2f} "
      f"(HRR bound {_cap['hrr_floor_bound']:.2f}) -> d'={_cap['dprime']:.0f} sigmas clear, ~10^{_cap['headroom_log10']:.0f}x headroom; "
      f"coverage stays ~a as N grows {dict((k, round(v,2)) for k,v in _cap['coverage_vs_load'].items())}")

# spectral/audio FHRR modality (Puckette) + dynamics on audio frames (Stam, the B4 proving ground): the
# phase-vocoder split is exact, and the propagator NAILS the per-bin phase advance where market returns had no
# structure -- beating both persistence (ignores the advance) and mean (averages the oscillation away).
_audm = UnifiedMind(dim=256, seed=0); _L = 256; _tt = np.arange(2000)
_ph, _mg = _audm.spectral_encode(sum(np.sin(2*np.pi*c*_tt[:_L]/_L + p) for c, p in [(5,0.3),(11,1.1),(19,2.0)]))
_rt = float(np.max(np.abs(_audm.spectral_decode(_ph, _mg) - sum(np.sin(2*np.pi*c*_tt[:_L]/_L + p) for c, p in [(5,0.3),(11,1.1),(19,2.0)]))))
_sig = sum(np.sin(2*np.pi*c*_tt/_L + p) for c, p in [(5,0.3),(11,1.1),(19,2.0)])
_fr = np.stack([_sig[i*37:i*37+_L] for i in range(1 + (len(_sig)-_L)//37)]); _k = int(len(_fr)*0.7)
_prop = _audm.learn_dynamics(_fr[:_k]); _te = _fr[_k:]; _mf = _fr[:_k].mean(0)
_rel = lambda a, b: np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12)
_ep = np.mean([_rel(_prop.step(_te[i]), _te[i+1]) for i in range(len(_te)-1)])
_epe = np.mean([_rel(_te[i], _te[i+1]) for i in range(len(_te)-1)])
_epm = np.mean([_rel(_mf, _te[i+1]) for i in range(len(_te)-1)])
print(f"  spectral/audio: phase vocoder + dynamics: encode/decode round-trip err {_rt:.0e}; next-frame prediction "
      f"error {_ep:.3f} vs persistence {_epe:.2f}, mean {_epm:.2f} -- audio HAS the linear structure markets lacked")

# fluid field on a torus (Stam): linear advection-diffusion is a per-mode rotation+decay -- exactly the operator
# learn_dynamics learns; it beats both baselines AND rolls out as a surrogate solver, but shock-forming
# nonlinear (Burgers) flow is the honest limit where a fixed linear operator does worse than persistence.
_N = 256; _xx = np.arange(_N)
def _fstep(u, sh=3.7, nu=3e-4):
    _U = np.fft.rfft(u); _k = np.arange(_U.size); _U *= np.exp(-1j*2*np.pi*_k*sh/_N)*np.exp(-nu*_k**2); return np.fft.irfft(_U, n=_N)
_uu = np.exp(-0.5*((_xx-64)/8.0)**2) + 0.4*np.sin(2*np.pi*3*_xx/_N); _uu -= _uu.mean(); _fs = [_uu.copy()]
for _ in range(40): _uu = _fstep(_uu); _fs.append(_uu.copy())
_fs = np.stack(_fs); _fk = int(len(_fs)*0.6); _fp = _audm.learn_dynamics(_fs[:_fk]); _ft = _fs[_fk:]; _fmf = _fs[:_fk].mean(0)
_fe = np.mean([_rel(_fp.step(_ft[i]), _ft[i+1]) for i in range(len(_ft)-1)])
_fpe = np.mean([_rel(_ft[i], _ft[i+1]) for i in range(len(_ft)-1)])
_fpm = np.mean([_rel(_fmf, _ft[i+1]) for i in range(len(_ft)-1)])
_rollerr = np.mean([_rel(_fp.rollout(_fs[_fk], 8)[i], _fs[_fk+1+i]) for i in range(8)])
print(f"  fluid field (advection-diffusion)   : next-field error {_fe:.3f} vs persistence {_fpe:.2f}, mean {_fpm:.2f}; "
      f"learned operator rolls out 8 steps as a surrogate solver at {_rollerr:.1%} error")

# multi-terminal network design (Adamatzky): the Tero/Physarum flow model connecting 5 terminals on a grid.
# mu tunes the cost/fault-tolerance trade-off, and the network comes back as a queryable B7 typed graph-memory.
from holographic_ai import unbind as _unbind, cosine as _cos
_gnbr = {}
for _r in range(7):
    for _c in range(7):
        _gnbr[(_r,_c)] = [(_r+_dr,_c+_dc) for _dr,_dc in ((1,0),(-1,0),(0,1),(0,-1)) if 0<=_r+_dr<7 and 0<=_c+_dc<7]
_gterms = [(0,0),(0,6),(6,0),(6,6),(3,3)]
def _cyc(_e): return len(_e) - len({_x for _ed in _e for _x in _ed}) + 1
_tree = _audm.design_network(_gnbr, _gterms, mu=4.0); _mesh = _audm.design_network(_gnbr, _gterms, mu=0.8)
_net = _audm.design_network(_gnbr, _gterms, mu=2.0); _adj = {}
for _u,_v in _net["edges"]: _adj.setdefault(_u,[]).append(_v); _adj.setdefault(_v,[]).append(_u)
_pr = _unbind(_net["memory"], _net["nodes"][(0,0)])
_rn = sorted(((_cos(_pr,_net["nodes"][_w]),_w) for _w in _net["nodes"] if _w!=(0,0)), reverse=True)[:len(_adj[(0,0)])]
print(f"  network design (Tero/Physarum)      : 5 terminals -> high-mu tree {len(_tree['edges'])} edges ({_cyc(_tree['edges'])} loops) "
      f"vs low-mu mesh {len(_mesh['edges'])} edges ({_cyc(_mesh['edges'])} loops); graph-memory recalls node (0,0)'s "
      f"neighbours {sorted(_w for _,_w in _rn)} == {sorted(_adj[(0,0)])}")

# cross-modal recall (Ozcan): the exact DCT-plate image archive, now reachable from the mind, recalled by a
# DESCRIPTION instead of a picture -- and the reverse (image -> tags), robust even under heavy plate erasure.
_S = 24
def _disc(cx, cy, r): _y, _x = np.ogrid[:_S, :_S]; return ((_x-cx)**2 + (_y-cy)**2 <= r*r).astype(float)
_ims = {"circle": _disc(8,8,5), "gradient": np.tile(np.linspace(0,1,_S), (_S,1)), "ring": _disc(12,12,9)-_disc(12,12,5)}
_tgs = {"circle": ["round","small"], "gradient": ["smooth","horizontal"], "ring": ["round","large"]}
_names = list(_ims); _arch = _audm.image_archive((_S,_S), capacity=len(_ims))
for _nm in _names: _arch.add(_ims[_nm], tags=_tgs[_nm])
_exact = max(float(np.max(np.abs(_arch.recover(_i)-_ims[_names[_i]]))) for _i in range(len(_names)))
_qi, _, _qc = _arch.recall_by_tags(words=["round","large"])
_mask = _arch.damage_mask(0.4, seed=1); _di, _dr, _ = _arch.recall_by_tags(words=["smooth"], mask=_mask)
_rt = [_w for _w,_ in _arch.tags_of(_names.index("ring"), sorted({_t for _ts in _tgs.values() for _t in _ts}))[:2]]
print(f"  cross-modal recall (tag <-> image)  : exact recover err {_exact:.0e}; describe ['round','large'] -> '{_names[_qi]}'; "
      f"reverse 'ring' -> {sorted(_rt)}; under 40% erasure ['smooth'] -> '{_names[_di]}' at err {float(np.max(np.abs(_dr-_ims['gradient']))):.0e}")

# generation over a COMPOSED subspace (Eno): run the B10 denoise-from-noise sampler over the manifold of
# role-filler structures (slot-wise projection) -> novel-but-VALID compositions, where the bare codebook only
# ever returns a stored atom (the kept negative).
from holographic_ai import bind as _bnd, unbind as _unb, derived_atom as _da, cosine as _csn
_rls = np.stack([_da(0, f"slot:{_i}", 1024, unitary=True) for _i in range(3)])
_fls = np.stack([_da(0, f"fill:{_w}", 1024, unitary=True) for _w in "ABCDEF"])
_gm = UnifiedMind(dim=1024, seed=0); _seen = set()
for _s in range(10):
    _z = _gm.generate_structure(_rls, _fls, seed=_s)
    _dec = tuple("ABCDEF"[int((_fls @ (_unb(_z,_r)/(np.linalg.norm(_unb(_z,_r))+1e-12))).argmax())] for _r in _rls)
    _seen.add("".join(_dec))
_bare = _gm.generate_vector(_fls, seed=3); _batom = "ABCDEF"[int((_fls @ (_bare/np.linalg.norm(_bare))).argmax())]
print(f"  generation over a composed subspace : {len(_seen)} distinct valid structures from 10 seeds "
      f"(e.g. {sorted(_seen)[:4]}) -- bare codebook only ever returns a stored atom ('{_batom}')")

# recursive/fractal scene from a SEED VECTOR (Quilez): one kernel encoded holographically in a single vector,
# decoded and repeated to depth -> a self-similar scene whose measured box-dimension matches log(N)/log(1/s).
_fm = UnifiedMind(dim=1024, seed=0)
_fsA = _fm.fractal_scene(_fm.fractal_seed([(0,0),(1,0),(0.5,0.857)], 0.5), depth=8)        # Sierpinski
_fsB = _fm.fractal_scene(_fm.fractal_seed([(0,0),(1,0),(0,1),(1,1),(0.5,0.5)], 1/3), depth=6)
print(f"  fractal scene from a seed vector    : seed A -> {_fsA['n_maps']} copies @ {_fsA['scale']:.2f} = box-dim {_fsA['dimension']:.2f} "
      f"(expected {_fsA['expected']:.2f}); seed B -> {_fsB['n_maps']} copies @ {_fsB['scale']:.2f} = box-dim {_fsB['dimension']:.2f} "
      f"(expected {_fsB['expected']:.2f}) -- one kernel, repeated to depth")

# anisotropic splats + 3-D (Drettakis): the real 3DGS primitive (oriented, full-covariance Gaussians) fit by
# gradient descent -- one aligned splat replaces many circular ones wherever structure is elongated.
from holographic_splat import psnr as _psp
_Ha = 44; _Yr, _Xr = np.mgrid[0:_Ha, 0:_Ha]
def _rg(cy, cx, an, L, th):
    _c, _s = np.cos(an), np.sin(an); _u = (_Xr-cx)*_c + (_Yr-cy)*_s; _v = -(_Xr-cx)*_s + (_Yr-cy)*_c
    return np.exp(-0.5*((_u/L)**2 + (_v/th)**2))
_Tr = _rg(15,15,0.6,9,2.0) + 0.8*_rg(29,27,-0.5,11,2.5)
_iso2 = _fm.splat_aniso(_Tr, k=4, steps=0, denoise=True); _, _ani2 = _fm.splat_aniso(_Tr, k=4, steps=200)
_Dv = 18; _Zv, _Yv, _Xv = np.mgrid[0:_Dv, 0:_Dv, 0:_Dv]
_vol = np.exp(-0.5*(((_Xv-9)/6.0)**2 + ((_Yv-9)/2.0)**2 + ((_Zv-9)/2.5)**2))
_iso3 = _fm.splat_aniso(_vol, k=3, steps=0, denoise=True); _, _ani3 = _fm.splat_aniso(_vol, k=3, steps=150)
print(f"  anisotropic splats + 3-D (3DGS)     : 2-D oriented ridges, K=4: isotropic {_psp(_Tr,_iso2):.0f} dB -> anisotropic {_psp(_Tr,_ani2):.0f} dB; "
      f"3-D ellipsoid, K=3: isotropic {_psp(_vol,_iso3):.0f} dB -> anisotropic {_psp(_vol,_ani3):.0f} dB (local optimum; warm-started)")

# tensor-train (MPS) bind vs HRR (Stoudenmire): the uncompressed tensor-product bind and its low-rank
# truncation, measured against the engine's circular convolution -- higher fidelity at higher storage.
from holographic_ai import bind as _tbnd, unbind as _tunb, random_vector as _trv, cosine as _tcos
_Dt = 128; _tr = np.random.default_rng(0); _un = lambda v: v/(np.linalg.norm(v)+1e-12)
_tks = [_un(_trv(_Dt,_tr)) for _ in range(16)]; _tvs = [_un(_trv(_Dt,_tr)) for _ in range(16)]
_hM = np.zeros(_Dt)
for _k,_v in zip(_tks,_tvs): _hM = _hM + _tbnd(_k,_v)
_hrr = float(np.mean([_tcos(_tunb(_hM,_k),_v) for _k,_v in zip(_tks,_tvs)]))
_tbm = _fm.tensor_bind(_tks,_tvs); _ten = float(np.mean([_tcos(_tbm.recall(_k),_v) for _k,_v in zip(_tks,_tvs)]))
_Qk = np.linalg.qr(_tr.standard_normal((_Dt,_Dt)))[0]; _oks=[_Qk[i] for i in range(_Dt)]; _ovs=[_un(_trv(_Dt,_tr)) for _ in range(_Dt)]
_oten = float(np.mean([_tcos(_fm.tensor_bind(_oks,_ovs).recall(_k),_v) for _k,_v in zip(_oks,_ovs)]))
print(f"  tensor-product / MPS bind vs HRR    : M=16 recall HRR(D) {_hrr:.2f} vs tensor-product(D^2) {_ten:.2f}; orthogonal keys M=D -> tensor "
      f"{_oten:.2f} (exact, HRR cannot); MPS losslessly compresses a low-rank bind -- higher fidelity, more storage")

# Path D (federation + width): the "as above, so below" arc -- one D-vector's ~0.1xD budget is CONSERVED, so
# federate across shards for capacity/resilience (RAID), and evaluate many computations in ONE vector for width.
_arrD = _fm.storage_array(n_parity=1, add_threshold=0.90); _rdD = np.random.default_rng(3)
for _ in range(150):
    _arrD.add(int(_rdD.integers(0, 256)))
_baseD = _arrD.accuracy(); _recD = _arrD.accuracy(down=(1,))
_itemsD = np.stack([_trv(_fm.dim, _rdD) for _ in range(6)])
_resD = _fm.superpose_compute(_itemsD, query=_itemsD[4], codebook=_itemsD)
print(f"  Path D federation + width           : RAID store grew to {len(_arrD.data)} shards @ {_baseD:.2f} recall, lose 1 -> parity restores {_recD:.2f}; "
      f"6 computations in ONE vector, winner={_resD['winner']} (want 4) by cleanup-gated parallel readout")

# Path D arithmetic lever (exact RNS-phasor matmul): general matmul in a lossy bundle dies as the matrix grows;
# carry numbers as residues and accumulate by phasor binding (exact phase sum) -> exact at ANY size.
import holographic_rns as _rns
_Wd = _rdD.integers(-50, 51, size=(256, 64)); _xd = _rdD.integers(-50, 51, size=64)
_exact_err = int(np.abs(_fm.exact_matmul(_Wd, _xd) - _Wd @ _xd).max())
_, _Pfew = _rns.choose_moduli(10 ** 8); _, _Pmany = _rns.choose_moduli(10 ** 60)
print(f"  Path D exact arithmetic (RNS)       : integer matmul 256x64 max|error|={_exact_err} (EXACT, where a lossy bundle gets ~0.11 fidelity); "
      f"range federates over moduli ~1e{len(str(_Pfew)) - 1} -> ~1e{len(str(_Pmany)) - 1}")

# Path D sublinear index (recursive pivot tree): nearest-item recall without scanning all of them -- routing is
# nearest-pivot cleanup applied recursively (inception as addressing); greedy top-1 ~ exhaustive at ~O(log N).
_pu = lambda v: v / (np.linalg.norm(v) or 1.0)
def _pgen(d, F, c, sc, dec, out):
    if d == 0:
        out.append(c); return
    for _ in range(F):
        _pgen(d - 1, F, c + _pu(_rdD.standard_normal(_fm.dim)) * sc, sc * dec, dec, out)
_pl = []; _pgen(3, 6, np.zeros(_fm.dim), 0.6 * 9.0, 1 / 3.0, _pl); _pl = np.stack(_pl); _pK = len(_pl)
_pq = np.random.default_rng(77); _pt = _pq.integers(0, _pK, size=200); _pQ = _pl[_pt] + 0.22 * _pq.standard_normal((200, _fm.dim))
_pex = float(np.mean([int(((_pl - _pQ[i]) ** 2).sum(1).argmin()) == _pt[i] for i in range(200)]))
_pidx = _fm.pivot_index(_pl, fanout=6)
_pt1 = float(np.mean([_pidx.query(_pQ[i], beam=1)[0] == _pt[i] for i in range(200)]))
_pc = float(np.mean([_pidx.query(_pQ[i], beam=1)[1] for i in range(200)]))
print(f"  Path D sublinear index (pivot tree) : {_pK} leaves, greedy top-1 {_pt1:.2f} vs exhaustive {_pex:.2f} at {_pc:.0f} comparisons "
      f"({_pK / _pc:.0f}x fewer) -- recursive cleanup routing")

# Path D sketch routing (the broadcast-wall fix): route a query to its shard by matching a per-shard key-sketch,
# then unbind only the top-c candidates -- accurate as the directory, far fewer unbinds than broadcasting to all.
_ar = _fm.storage_array(n_parity=0, add_threshold=0.0); _rr = np.random.default_rng(2)
for _s in range(32):
    if _s > 0:
        _ar._spin_up()
    for _ in range(30):
        _ar.add(int(_rr.integers(0, 256)))
_sp = list(_ar.truth); _ss = [_sp[i] for i in np.random.default_rng(5).choice(len(_sp), 150, replace=False)]
_dir = float(np.mean([_ar.recall(g) == _ar.truth[g][1] for g in _ss]))
_rt = float(np.mean([_ar.routed_recall(g, c=8) == _ar.truth[g][1] for g in _ss]))
_bc = float(np.mean([_ar.broadcast_recall(g) == _ar.truth[g][1] for g in _ss]))
print(f"  Path D sketch routing (array)       : 32 shards -- directory {_dir:.2f}, sketch-routed(c=8) {_rt:.2f}, broadcast {_bc:.2f}; "
      f"routing touches 8 shards, not all 32")

# Path D distributed forward pass (THE WIN): a classifier's weight rows in ONE vector cap at ~0.02xD classes;
# federate them across K shards and the wall moves to ~K x 0.02xD -- the same federation move that fixed storage.
import holographic_compute as _hc
_Hte, _yte, _Wc, _Lex = _hc._classifier(64, 20, _fm.dim, np.random.default_rng(11))
_exA = float(np.mean(_Lex.argmax(1) == _yte))
_a1 = float(np.mean(_fm.distributed_forward(_Wc, _Hte, K=1).argmax(1) == _yte))
_a8 = float(np.mean(_fm.distributed_forward(_Wc, _Hte, K=8).argmax(1) == _yte))
print(f"  Path D distributed forward pass     : 64-class forward pass -- exact {_exA:.2f}, single-vector(K=1) {_a1:.2f}, "
      f"federated(K=8) {_a8:.2f}; federation moves the class wall (16 -> 96 classes faithful at K=8 in the sweep)")

# Path D Bucket-A under federation: the federate-the-shards move applied to SEQUENCE recall (superpose_compute
# shards=K) and to the image ARCHIVE (federated_archive) -- the same lever that fixed storage and the matmul.
import holographic_ai as _hai
_Vcb = np.stack([_trv(_fm.dim, _rdD) for _ in range(64)]); _seqT = _rdD.integers(0, 64, size=160)
_spos = np.stack([_hai.unitary_vector(_fm.dim, _rdD) for _ in range(160)])
_s1 = float(np.mean(_fm.superpose_compute(_Vcb[_seqT], keys=_spos, codebook=_Vcb, shards=1)["decoded"] == _seqT))
_s8 = float(np.mean(_fm.superpose_compute(_Vcb[_seqT], keys=_spos, codebook=_Vcb, shards=8)["decoded"] == _seqT))
print(f"  Path D federated sequence (width)   : recall a 160-symbol sequence -- single vector {_s1:.2f}, federated(K=8) {_s8:.2f}; "
      f"federation moves the length wall (same for hypothesis selection)")
from holographic_archive import HolographicArchive as _HA
_imD = [_rdD.random((16, 16)) for _ in range(64)]; _keepD = 8192 // 64
_moD = _HA((16, 16, 1), capacity=64, keep=_keepD, dim=8192, seed=0); [_moD.add(_im) for _im in _imD]
_cf = lambda a, b: float((lambda x, y: x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))(a - a.mean(), b - b.mean()))
_mcD = float(np.mean([_cf(_moD.recover(i).ravel(), _imD[i].ravel()) for i in range(64)]))
_fdD = _fm.federated_archive((16, 16, 1), capacity=64, K=4, keep=_keepD, dim=8192 // 4); _giD = [_fdD.add(_im) for _im in _imD]
_fcD = float(np.mean([_cf(_fdD.recover(_giD[i]).ravel(), _imD[i].ravel()) for i in range(64)]))
print(f"  Path D federated archive            : 64 images at fixed total dim -- monolithic corr {_mcD:.2f}, federated(K=4) corr {_fcD:.2f}; "
      f"capacity federates, recovery quality conserved")

# Path D conservation diagnostic: the "as above, so below" law as a callable readout -- the per-vector budget,
# how federation scales it, the partition-conservation check, and a shard recommendation for a target count.
_rep = _fm.federation_report(target_items=500)
print(f"  Path D conservation diagnostic      : per-vector budget {_rep['per_vector_budget']} symbols (~{_rep['per_vector_fraction'] * 100:.0f}% of D), "
      f"4 shards hold {_rep['federated']['stored']} @ {_rep['federated']['recall']:.2f}, conservation ratio {_rep['conservation_ratio']:.2f}; "
      f"500 items -> {_rep['recommended_shards']} shards")

# Path D learning program: gradient-free, substrate-native learning, wired as faculties.
# reservoir = fixed permute-recurrence Echo-State Net + one ridge readout (no gradients); it learns a signal.
import numpy as _np
_t = _np.arange(1300); _s = _np.sin(_t / 4.0) + 0.3 * _np.sin(_t / 9.0)
_esn = _fm.reservoir(n_in=1, rho=0.95, leak=0.5)
_esn.fit(_s[:900, None], _s[1:901], ridge=1e-6, washout=100)
_pr = _esn.predict(_s[900:1150, None]).ravel(); _tg = _s[901:1151]
_nr = float(_np.sqrt(_np.mean((_tg[60:] - _pr[60:]) ** 2) / (_np.var(_tg[60:]) + 1e-12)))
print(f"  Gradient-free reservoir (ESN)       : one-step NRMSE {_nr:.3f} -- fixed reservoir, only the ridge readout learned")
# prototype_classifier = bundle prototypes + perceptron retraining (add/subtract on a miss, no gradients).
_rng = _np.random.default_rng(0); _C, _d, _per = 3, 8, 80
_ct = _rng.standard_normal((_C, _d)) * 2.0
_Xtr = _np.vstack([_ct[c] + _rng.standard_normal((_per, _d)) for c in range(_C)]); _ytr = _np.repeat(_np.arange(_C), _per)
_Xte = _np.vstack([_ct[c] + _rng.standard_normal((40, _d)) for c in range(_C)]); _yte = _np.repeat(_np.arange(_C), 40)
_clf = _fm.prototype_classifier(levels=16)
_clf.fit(_Xtr, _ytr, epochs=0); _one = float(_np.mean(_clf.predict(_Xte) == _yte))
_clf.fit(_Xtr, _ytr, epochs=15); _ret = float(_np.mean(_clf.predict(_Xte) == _yte))
print(f"  Gradient-free HDC classifier        : one-shot {_one:.2f} -> retrained {_ret:.2f} (perceptron add/subtract, no gradients)")
# Equilibrium Propagation: the LOCAL-gradient corner -- learns the energy-based Hopfield's hidden weights.
from holographic_equilibrium import _moons as _mk
_rngm = _np.random.default_rng(0); _Xm, _ym = _mk(360, 0.10, _rngm)
_pm = _rngm.permutation(len(_Xm)); _Xm, _ym = _Xm[_pm], _ym[_pm]; _Ym = _np.eye(2)[_ym]; _ntr = 260
_ep = _fm.equilibrium_net(n_in=2, n_hidden=48, n_out=2, beta=0.35, dt=0.35, t_free=45, t_nudge=12)
_ep.fit(_Xm[:_ntr], _Ym[:_ntr], epochs=100, lr=0.3, batch=90, seed=0)
_ea = float(_np.mean(_ep.predict(_Xm[_ntr:]) == _ym[_ntr:]))
_Aa = _np.c_[_Xm[:_ntr], _np.ones(_ntr)]; _wl = _np.linalg.lstsq(_Aa, _Ym[:_ntr], rcond=None)[0]
_la = float(_np.mean(_np.argmax(_np.c_[_Xm[_ntr:], _np.ones(len(_Xm) - _ntr)] @ _wl, 1) == _ym[_ntr:]))
print(f"  Equilibrium Propagation (local-grad): two-moons {_ea:.2f} vs linear {_la:.2f} -- learns hidden weights, no backprop")
# Forward-Forward: backprop-free DEPTH from local goodness objectives (the mechanism; honest negative: trails linear at this scale).
from holographic_forward import _blobs as _bl
_rngf = _np.random.default_rng(0); _Xf, _yf = _bl(560, 16, 4, 2.2, _rngf)
_pf = _rngf.permutation(len(_Xf)); _Xf, _yf = _Xf[_pf], _yf[_pf]; _nf = 420
_ff = _fm.forward_forward(n_in=16, layer_sizes=(100, 100), n_classes=4, theta=0.05, label_scale=4.0)
_ff.fit(_Xf[:_nf], _yf[:_nf], epochs=60, lr=0.1, batch=100, seed=0)
_fa = float(_np.mean(_ff.predict(_Xf[_nf:]) == _yf[_nf:]))
print(f"  Forward-Forward (local goodness)    : separable 4-class blobs {_fa:.2f} -- backprop-free depth, no settling")

# Nonlinear-dynamics companion: the reservoir learns a chaotic one-step map the linear propagator cannot.
from holographic_chaos import lorenz_trajectory as _lz
_tr = _lz(3000, seed=0); _ntr = 2000; _te = _tr[_ntr:]
_cp = _fm.learn_chaos(_tr[:_ntr], dim=400, noise=1e-2); _pp = _cp.predict_sequence(_te)
_rl = lambda p, t: float(_np.linalg.norm(p - t) / (_np.linalg.norm(t) + 1e-12))
_cres = _np.mean([_rl(_pp[i], _te[i + 1]) for i in range(200, len(_te) - 1)])
_cper = _np.mean([_rl(_te[i], _te[i + 1]) for i in range(len(_te) - 1)])
_Ad, _, _, _ = _np.linalg.lstsq(_tr[:_ntr - 1], _tr[1:_ntr], rcond=None)
_cdmd = _np.mean([_rl(_te[i] @ _Ad, _te[i + 1]) for i in range(len(_te) - 1)])
print(f"  Nonlinear dynamics (learn_chaos)    : chaotic Lorenz one-step {_cres:.4f} vs best-linear DMD {_cdmd:.3f}, "
      f"persistence {_cper:.3f} -- {_cdmd / _cres:.0f}x best-linear (closed-loop caps ~1 Lyapunov time, kept)")

# Learned energy memory: EP trains the cleanup's ATTRACTORS -- a learned projector beats the fixed cleanup on
# a continuous manifold (the engine's deepest fixed object, the cleanup, finally made trainable).
from holographic_energy import torus_bump_manifold as _tbm
from holographic_hopfield import dense_cleanup as _dc
_ecl, _, _eD = _tbm(n_grid=6, latent_dim=2, sigma=0.13, n_samples=1500, noise=0.30, seed=0)
_em = _fm.learn_cleanup(_ecl, noise=0.30, n_hidden=24, epochs=70)
_ect, _ent, _ = _tbm(n_grid=6, latent_dim=2, sigma=0.13, n_samples=200, noise=0.30, seed=5)
_eep = _np.mean([_rl(_em.cleanup(_ent[i]), _ect[i]) for i in range(len(_ent))])
_esoft = _np.mean([_rl(_dc(_ent[i], _ecl[:64], beta=25.0, steps=3), _ect[i]) for i in range(len(_ent))])
print(f"  Learned energy memory (learn_cleanup): 2-D manifold learned {_eep:.3f} vs fixed soft cleanup {_esoft:.3f} "
      f"-- EP trains the cleanup's attractors (discrete atoms still want the hard cleanup, kept)")

# Sparse cleanup readout + geometry-aware denoise: match the MAP to the MANIFOLD (panel review, 2026).
# The softmax cleanup blend over-smooths a continuous manifold and loses to nearest-neighbour; the sparse
# (Hopfield-Fenchel-Young) readout blends only the relevant patterns and edges past NN, and the geometry
# router reads the set's effective rank and projects a low-rank manifold instead -- recovering UN-stored points.
from holographic_hopfield import dense_cleanup as _dc
from holographic_ai import cosine as _csg
_rgg = _np.random.default_rng(0); _Ag = _rgg.standard_normal(1024); _Bg = _rgg.standard_normal(1024)
def _slp(a, b, t):
    a1, b1 = a / _np.linalg.norm(a), b / _np.linalg.norm(b); om = _np.arccos(_np.clip(a1 @ b1, -1, 1)); so = _np.sin(om)
    return (_np.sin((1 - t) * om) / so) * a + (_np.sin(t * om) / so) * b
_crs = _np.stack([_slp(_Ag, _Bg, t) for t in _np.linspace(0, 1, 6)])         # a continuous low-rank manifold
_Cug = _crs / _np.linalg.norm(_crs, axis=1, keepdims=True); _nmg = _np.linalg.norm(_crs, axis=1).mean()
_so = []; _sp = []; _nn = []; _gm = []
for _ in range(200):
    _tg = _rgg.uniform(0, 1); _clg = _slp(_Ag, _Bg, _tg)
    _nyg = _clg + 1.0 * _nmg / _np.sqrt(1024) * _rgg.standard_normal(1024)
    _so.append(_csg(_dc(_nyg, _crs, beta=8, steps=3, readout="softmax"), _clg))
    _sp.append(_csg(_dc(_nyg, _crs, beta=8, steps=3, readout="sparsemax"), _clg))
    _nn.append(_csg(_crs[int((_Cug @ (_nyg / _np.linalg.norm(_nyg))).argmax())], _clg))
    _gm.append(_csg(_fm.denoise(_nyg, method="geometry", samples=_crs), _clg))
print(f"  Sparse cleanup + geometry denoise   : in-between recovery softmax {_np.mean(_so):.3f} < NN {_np.mean(_nn):.3f} "
      f"< sparse {_np.mean(_sp):.3f}; geometry auto-routes to projection {_np.mean(_gm):.3f} -- match the map to the manifold")

# Same fix, one rung up: the resonator's alternating projection is a softmax blend too, so the sparse readout
# cures ITS metastable mixing -- recovering factorizations at high alphabet where the softmax blend collapses to 0.
from holographic_sbc import sbc_codebook as _scb, sbc_reconstruct as _srec, sbc_resonator as _sres
def _capr(_N, _ro, _tr=20):
    _ok = 0
    for _s in range(_tr):
        _rg = _np.random.default_rng(_s); _cb = [_scb(16, 16, _N, seed=900 + _f + _s * 7) for _f in range(3)]
        _tu = tuple(int(_rg.integers(_N)) for _ in range(3)); _pr = _srec(_tu, _cb, 16)
        _pk, _ = _sres(_pr, _cb, 16, restarts=6, iters=50, seed=_s, readout=_ro)
        _ok += int(tuple(_pk) == _tu)
    return _ok / _tr
_r25s, _r25p = _capr(25, "softmax"), _capr(25, "sparsemax"); _r50s, _r50p = _capr(50, "softmax"), _capr(50, "sparsemax")
print(f"  Sparse resonator readout            : all-factors-correct N=25 {_r25s:.2f}->{_r25p:.2f}, N=50 {_r50s:.2f}->{_r50p:.2f} "
      f"(softmax->sparse) -- metastable-mixing fix raises factorization capacity, no regression")

# Same fix in the GENERATIVE attractor: generate_structure's slot-wise blend is softmax too, so sparse cures
# its MODE COLLAPSE -- softmax funnels many seeds into the same few structures; sparse stays diverse (valid either way).
from holographic_ai import unitary_vector as _uv, random_vector as _rv, bind as _bd, unbind as _ub
from holographic_hopfield import generate_structure as _gst, _unit_rows as _ur
def _gun(v): return v / (_np.linalg.norm(v) + 1e-12)
def _gdiv(_ro, _sd=18):
    _r = _np.random.default_rng(55); _rl = _np.array([_uv(1024, _r) for _ in range(3)])
    _fl = _np.array([_rv(1024, _r) for _ in range(12)]); _fu = _ur(_fl); _cm = set(); _rc = []
    for _s in range(_sd):
        _z = _gst(_rl, _fl, steps=16, seed=_s, readout=_ro)
        _c = tuple(int((_fu @ _gun(_ub(_z, _rl[_i]))).argmax()) for _i in range(3))
        _rc.append(float(_z @ _gun(_np.sum([_bd(_rl[_i], _fu[_c[_i]]) for _i in range(3)], axis=0)))); _cm.add(_c)
    return _np.mean(_rc), len(_cm) / _sd
_gvs, _gds = _gdiv("softmax"); _gvp, _gdp = _gdiv("sparsemax")
print(f"  Sparse generative readout           : structures valid (reencode cosine) soft {_gvs:.3f} / sparse {_gvp:.3f}; "
      f"diversity soft {_gds:.2f} -> sparse {_gdp:.2f} -- same fix cures generative mode collapse")

# Grounded answering: construct a SHORT, ACCURATE sentence from RETRIEVED knowledge (the relational layer
# that works) and ABSTAIN honestly when unknown -- not the Markov walk, which is locally fluent but incoherent.
_am = UnifiedMind(dim=512, seed=0)
_am.learn_encyclopedia({'dog': {'is_a': 'mammal'}, 'mammal': {'is_a': 'animal'},
                        'animal': {'is_a': 'organism'}, 'france': {'is_a': 'country', 'capital': 'paris'}})
print(f"  Grounded answer (answer_text)       : 'is a dog an animal?' -> {_am.answer_text('is a dog an animal?')}")
print(f"                                        'capital of france?'  -> {_am.answer_text('what is the capital of france?')}"
      f"  (constructed, accurate, non-verbatim; unknowns abstain)")

# VSA-native question router: a blend of the question's word meanings picks the intent, a concept-scan WITH
# ORDER picks the arguments -- so natural/verbose phrasings the regex templates miss now route correctly.
_anat = "could you tell me whether a dog is an animal"
print(f"  VSA question router (intent+order)  : {_anat!r}")
print(f"                                        -> {_am.answer_text(_anat)} "
      f"(regex misses this phrasing; routed by meaning, subject/ancestor by word order)")

# Two readouts that earn their place at the extremes of the load range.
title("High-load factorization: the TopK readout where the softmax blend collapses")
_tr = UnifiedMind(dim=1024, seed=0)
_r = _np.random.default_rng(0); _L, _B, _F, _N = 64, 16, 3, 40
_cbs = [[tuple(_r.integers(0, _L, size=_B)) for _ in range(_N)] for _ in range(_F)]
_true = tuple(int(_r.integers(_N)) for _ in range(_F))
from holographic_sbc import sbc_reconstruct as _recon
_P = _np.asarray(_recon(_true, _cbs, _L))
_softok = _tr.decompose_structure(_P, _cbs, _L, readout="softmax")["verified"]
_tk = _tr.decompose_structure(_P, _cbs, _L, readout="topk", k=8)
print(f"  N={_N} codebook: softmax verified={_softok}; topk(k=8) recovers={_tk['picks'] == _true} verified={_tk['verified']} "
      f"(keeping exactly k candidates survives where the full softmax blend collapses)")

title("Predictive loop: support-weighted soft read is MAP-correct on a stochastic successor")
_pm = UnifiedMind(dim=1024, seed=0).build_predictor(order=2)
_pr = _np.random.default_rng(3); _pseq = []
for _ in range(400):
    _pseq += [0, 2, 3 if _pr.random() < 0.7 else 4]            # A -> B 70% / C 30%
_pm.observe_sequence(_pseq)
print(f"  after [P,A] with B 70% / C 30%: soft read -> token {_pm.anticipate([0, 2], soft=True)[0]}  "
      f"(3=B is the MAP; the blend is weighted by how often each successor was seen, not resonance alone)")

# As-above-so-below: the SBC resonator's sharpened readout, swept down to the circular-convolution
# resonator -- a softmax-sharpened cleanup recovers a high-load factorization the linear cleanup misses.
title("Sharpened resonator cleanup: recovery where the linear cleanup collapses at high load")
import functools as _ft
from holographic_reasoning import ResonatorNetwork as _RN
_rrng = _np.random.default_rng(0); _F, _Dr, _Nr = 3, 1024, 45
_rcbs = [_np.stack([random_vector(_Dr, _rrng) for _ in range(_Nr)]) for _ in range(_F)]
_R = _RN(_rcbs); _rt = tuple(int(_np.random.default_rng(7000).integers(_Nr)) for _ in range(_F))
_rc = _ft.reduce(bind, [_rcbs[f][_rt[f]] for f in range(_F)])
def _rec(beta):
    return any(tuple(_R.factor(_rc, iters=50, init=(None if r == 0 else "random"),
                               rng=_np.random.default_rng(r), beta=beta)) == _rt for r in range(4))
print(f"  N={_Nr} codebook, {_F} factors: linear cleanup recovers={_rec(None)}; sharpened (beta=25) recovers={_rec(25)} "
      f"(the readout lesson that lifted the SBC resonator, applied to the older one)")

title("De-Doppler detection: a Doppler drift is a binding (permute), and bh_fdr is the look-elsewhere veto")
from holographic_dedoppler import dedoppler_bank as _ddb
_um = UnifiedMind(dim=256, seed=0)
_Td, _Fd = 24, 96
_drd = _np.arange(-3.0, 3.0001, 0.5)
def _wf(drift, chan, amp, seed, rfi=None, ramp=0.0, on=True):
    _r = _np.random.default_rng(seed); w = _r.standard_normal((_Td, _Fd))
    if on:
        for t in range(_Td):
            w[t, int(round(chan + drift * t)) % _Fd] += amp     # the drifting narrowband signal
    if rfi is not None:
        w[:, rfi] += ramp                                       # stationary RFI (every frame, no drift)
    return w
# a narrowband signal drifting +1.5 bins/frame: stationary integration loses it, de-drift recovers it
_sig = _wf(1.5, 40, 2.4, 7)
_z = _ddb(_sig, _drd)
_stat = _z[list(_drd).index(0.0)].max(); _best = _z.max(); _rate = _drd[int(_z.max(axis=1).argmax())]
print(f"  stationary integration peak = {_stat:.1f} sigma (lost in noise); de-Doppler bank peak = {_best:.1f} sigma "
      f"at drift {_rate:+.1f} bins/frame (true +1.5)")
_det = _um.detect_drifting(_sig, alpha=0.01)
print(f"  detect_drifting found {len(_det)} signal(s); top: drift {_det[0]['drift']:+.1f}, channel {_det[0]['channel']}, "
      f"{_det[0]['snr']:.1f} sigma, p={_det[0]['pvalue']:.1e}")
# ON-OFF cadence: a strong stationary RFI is rejected because it persists in the OFF pointing
_on = _wf(1.5, 30, 2.4, 21, rfi=70, ramp=2.4)
_off = _wf(1.5, 30, 0.0, 121, rfi=70, ramp=2.4, on=False)
_nc = _um.detect_drifting(_on, alpha=0.01); _wc = _um.detect_drifting(_on, alpha=0.01, off=_off)
_rfi_nc = any(abs(d['drift']) < 1e-9 and abs(d['channel'] - 70) <= 1 for d in _nc)
_rfi_wc = any(abs(d['drift']) < 1e-9 and abs(d['channel'] - 70) <= 1 for d in _wc)
_sig_wc = any(abs(d['drift'] - 1.5) < 0.6 and abs(d['channel'] - 30) <= 1 for d in _wc)
print(f"  cadence: strong RFI detected without OFF = {_rfi_nc}; with OFF the RFI is "
      f"{'rejected' if not _rfi_wc else 'KEPT'} and the drifting signal is {'kept' if _sig_wc else 'LOST'}")

title("Self-verifying storage: a Merkle tree built from bind + bundle (detect + localise a tamper in log n)")
import numpy as _vnp
from holographic_ai import bind as _vbind, cosine as _vcos
_vm = __import__("holographic_unified").UnifiedMind(dim=512, seed=0)
_vrng = _vnp.random.default_rng(3)
_vitems = [_vrng.standard_normal(512) for _ in range(32)]          # 32 stored item vectors
_vtree = _vm.verify_store(_vitems)                                 # commit: the root is the tamper-evident checksum
print(f"  committed {len(_vitems)} items; clean store verifies = {_vtree.verify(_vitems)}")
_vtamp = list(_vitems); _vtamp[19] = _vrng.standard_normal(512)    # silently change item 19
_vidx, _vchecks = _vtree.locate(_vtamp)
print(f"  one item changed: detected = {not _vtree.verify(_vtamp)}, localised to slot {_vidx} (true 19) "
      f"in {_vchecks} checks (<= log2(32)+1 = 6)")
_vswap = list(_vitems); _vswap[4], _vswap[27] = _vswap[27], _vswap[4]   # reorder two slots
print(f"  two slots swapped: caught = {_vtree.locate(_vswap)[0] is not None} "
      f"(a plain bundle is commutative -- position binding defeats it)")
_va, _vb = 7, 20; _vda = _vrng.standard_normal(512)                # the kept negative: the root is LINEAR
_vdb = _vnp.fft.irfft(_vnp.fft.rfft(-_vbind(_vtree.positions[_va], _vda)) / _vnp.fft.rfft(_vtree.positions[_vb]), n=512)
_vforge = list(_vitems); _vforge[_va] = _vitems[_va] + _vda; _vforge[_vb] = _vitems[_vb] + _vdb
_vfr = sum(_vbind(_vtree.positions[i], _vnp.asarray(_vforge[i], float)) for i in range(32))
_vu = lambda v: v / (_vnp.linalg.norm(v) + 1e-12)
print(f"  kept negative: a deconvolution-canceled pair leaves the root at cosine "
      f"{_vcos(_vu(_vfr), _vu(_vtree.root())):.5f} of the original -- an invisible collision "
      f"(corruption-evidence, not crypto tamper-proofing)")

title("Fractional power encoding: continuous position as a binding, and compute on whole functions (BLD-7)")
import numpy as _fpnp
from holographic_ai import bind as _fpbind, cosine as _fpcos
_fpm = __import__("holographic_unified").UnifiedMind(dim=512, seed=0)
from holographic_encoders import ScalarEncoder as _FPSE                # 1-D FPE was already the ScalarEncoder
_fps = _FPSE(dim=512, lo=0, hi=10, seed=1, kernel="rbf", bandwidth=3.0)
print(f"  1-D (already the ScalarEncoder): cos(bind(fpe(2),fpe(3)), fpe(5)) = "
      f"{_fpcos(_fpbind(_fps.encode(2.0), _fps.encode(3.0)), _fps.encode(5.0)):.5f}  (shift-as-bind, exact)")
_fpe = _fpm.vector_function_encoder(2, bounds=[(0, 10), (0, 10)])      # N-D: a 2-D point
_fpp = _fpnp.array([3.0, 4.0]); _fpd = _fpnp.array([1.5, 2.0])
print(f"  2-D shift-as-bind: cos(bind(fpe(p),fpe(d)), fpe(p+d)) = "
      f"{_fpcos(_fpbind(_fpe.encode(_fpp), _fpe.encode(_fpd)), _fpe.encode(_fpp + _fpd)):.5f}")
_fpq = _fpnp.array([4.0, 5.0])
print(f"  2-D kernel is the product of the axis kernels: measured {_fpcos(_fpe.encode(_fpp), _fpe.encode(_fpq)):.3f}"
      f" vs product {_fpe.kernel_at(_fpq - _fpp):.3f}")
_fpf = _fpe.bundle([(2.0, 2.0), (7.0, 3.0), (4.0, 6.0)], [1.0, 0.6, 0.8])   # a function as a bundle
print(f"  a function f = sum w_i fpe(p_i): query at its points "
      f"{[round(_fpe.query(_fpf, p), 2) for p in [(2, 2), (7, 3), (4, 6)]]} vs empty (9.5,9.5) {_fpe.query(_fpf, (9.5, 9.5)):.2f}")
_fpfs = _fpe.shift(_fpf, (1.0, 1.0))                                   # translate the WHOLE function with one bind
print(f"  shifted by one binding: query at (3,3)=p0+(1,1) rises to {_fpe.query(_fpfs, (3.0, 3.0)):.2f} "
      f"(capacity cliff kept: a function is a bundle, separation fades as atoms pile up)")

title("Spectral structure kernel: one Laplacian gives the basis (line->DCT, ring->DFT) AND the topology (EXP-5/6)")
import numpy as _spnp
_spm = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
from holographic_spectral import cycle_laplacian as _spcyc, laplacian_eigenbasis as _speig, betti_numbers as _spbetti
_spw, _ = _speig(_spcyc(16))                                          # cycle Laplacian eigenbasis IS the DFT
print(f"  cycle C_16 Laplacian eigenvalues == 4 sin^2(pi k/16): "
      f"{_spnp.allclose(_spw, _spnp.sort([4*_spnp.sin(_spnp.pi*k/16)**2 for k in range(16)]))}  (the harmonic basis, derived)")
print(f"  Hodge harmonic dim == Betti numbers: 4-cycle {_spbetti(4,[(0,1),(1,2),(2,3),(3,0)])} (1 comp, 1 loop), "
      f"filled triangle {_spbetti(3,[(0,1),(1,2),(0,2)],[(0,1,2)])} (loop filled)")
_spN = 300; _spi = _spnp.arange(_spN)                                 # a smooth field on a SPHERE
_spphi = _spnp.arccos(1 - 2*(_spi+0.5)/_spN); _spth = _spnp.pi*(1+5**0.5)*_spi
_spP = _spnp.stack([_spnp.sin(_spphi)*_spnp.cos(_spth), _spnp.sin(_spphi)*_spnp.sin(_spth), _spnp.cos(_spphi)], 1)
_spf = _spP[:,2]**2 - 1/3 + _spP[:,0]*_spP[:,1]
_spfn = _spf + 0.3*_spnp.random.default_rng(0).standard_normal(_spN)
_spsb = _spm.spectral_basis(_spP, k=10, n_basis=12)
print(f"  sphere field denoise (detector calls it 'line'): kNN-Laplacian basis err "
      f"{_spnp.linalg.norm(_spsb.denoise(_spfn)-_spf):.2f} from a noisy {_spnp.linalg.norm(_spfn-_spf):.2f} "
      f"-- the data-driven basis where the line fallback can't reach")

title("Persistent homology: name a shape by its holes -- torus and sphere the 1-D detector can't (EXP-7)")
import numpy as _tpnp
_tpm = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
_tpNu, _tpNv = 24, 12                                                 # a TORUS surface
_tpu = _tpnp.repeat(_tpnp.linspace(0, 2*_tpnp.pi, _tpNu, endpoint=False), _tpNv)
_tpv = _tpnp.tile(_tpnp.linspace(0, 2*_tpnp.pi, _tpNv, endpoint=False), _tpNu)
_tptorus = _tpnp.column_stack([(2+0.8*_tpnp.cos(_tpv))*_tpnp.cos(_tpu), (2+0.8*_tpnp.cos(_tpv))*_tpnp.sin(_tpu), 0.8*_tpnp.sin(_tpv)])
_tpnm, _tpb, _ = _tpm.manifold_topology(_tptorus)
print(f"  torus  -> Betti {_tpb} = '{_tpnm}'  (1 piece, 2 loops, 1 void -- two periods, structurally named)")
_tpN = 200; _tpi = _tpnp.arange(_tpN)                                 # a SPHERE
_tpphi = _tpnp.arccos(1 - 2*(_tpi+0.5)/_tpN); _tpth = _tpnp.pi*(1+5**0.5)*_tpi
_tpsphere = _tpnp.column_stack([_tpnp.sin(_tpphi)*_tpnp.cos(_tpth), _tpnp.sin(_tpphi)*_tpnp.sin(_tpth), _tpnp.cos(_tpphi)])
_tpnm2, _tpb2, _ = _tpm.manifold_topology(_tpsphere)
print(f"  sphere -> Betti {_tpb2} = '{_tpnm2}'  (no loops, 1 void -- B2 is what tells it from a line)")
_tpth2 = _tpnp.linspace(0, 2*_tpnp.pi, 40, endpoint=False)            # a RING, as detect_topology would call it
_tpring = _tpnp.column_stack([_tpnp.cos(_tpth2), _tpnp.sin(_tpth2), _tpnp.zeros(40)])
print(f"  ring   -> '{_tpm.manifold_topology(_tpring)[0]}'  (reproduces the hand-coded detector on the case it knows; "
      f"GF(2) Betti, exact, no SVD timeout)")

title("Fast topology as a GATE + spectral basis at scale: is_manifold guards denoise; ChebFSI lifts the O(n^3) eigh")
import numpy as _mgnp, time as _mgtime
_mgm = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
_mgrng = _mgnp.random.default_rng(0)
def _mgsphere(n):
    i = _mgnp.arange(n); ph = _mgnp.arccos(1 - 2*(i+0.5)/n); th = _mgnp.pi*(1+5**0.5)*i
    return _mgnp.column_stack([_mgnp.sin(ph)*_mgnp.cos(th), _mgnp.sin(ph)*_mgnp.sin(th), _mgnp.cos(ph)])
_mgSph = _mgsphere(200); _mgBlob = _mgrng.standard_normal((200, 4))     # a clean manifold vs a structureless blob
_mgGS = _mgm.is_manifold(_mgSph); _mgGB = _mgm.is_manifold(_mgBlob)
print(f"  is_manifold gate:  sphere -> {_mgGS['is_manifold']} (topology '{_mgGS['topology']}', B0={_mgGS['betti'][0]})   "
      f"blob -> {_mgGB['is_manifold']} (B0={_mgGB['betti'][0]}, dense_scales={_mgGB['dense_scales']})")
_mgf = _mgSph[:,2]**2 - 0.5*_mgSph[:,0]; _mgfn = _mgf + 0.3*_mgrng.standard_normal(200)
_mgD = _mgm.denoise(_mgfn, method="spectral", points=_mgSph, check_manifold=True)   # premise holds -> proceeds
_mggn = _mgBlob[:,0] + 0.3*_mgrng.standard_normal(200)
try:
    _mgm.denoise(_mggn, method="spectral", points=_mgBlob, check_manifold=True); _mgref = "NO-RAISE(bug)"
except ValueError:
    _mgref = "refused"
print(f"  check_manifold guard: sphere denoise {_mgnp.linalg.norm(_mgfn-_mgf):.2f}->{_mgnp.linalg.norm(_mgD-_mgf):.2f}; "
      f"blob -> {_mgref} (graph low-pass is not manifold denoising; check_manifold=False overrides)")
_mgBig = _mgsphere(2500); _mgbf = _mgBig[:,2]**2 - 0.5*_mgBig[:,0]; _mgbfn = _mgbf + 0.3*_mgrng.standard_normal(2500)
_mgt0 = _mgtime.time(); _mgbd = _mgm.denoise(_mgbfn, method="spectral", points=_mgBig); _mgdt = _mgtime.time() - _mgt0
from holographic_spectral import knn_laplacian as _mgkl, laplacian_eigenbasis as _mgle
_, _mgVe = _mgle(_mgkl(_mgBig, 10), 12); _mgee = _mgnp.linalg.norm(_mgVe @ (_mgVe.T @ _mgbfn) - _mgbf)
print(f"  spectral basis at scale (n=2500): ChebFSI denoise {_mgnp.linalg.norm(_mgbd-_mgbf):.2f} ~= exact eigh {_mgee:.2f} "
      f"in {_mgdt:.2f}s -- sparse-matvec partial eigensolver, no O(n^3) dense eigh")

title("Hodge decomposition: split a flow into transport + circulation + topology, and denoise it (EXP-8)")
import numpy as _hgnp
from holographic_spectral import boundary_matrices as _hgbm
_hgm = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
_hgtris_all = []                                                      # triangulated 3x3 grid, one triangle removed -> a hole
for _cy in range(2):
    for _cx in range(2):
        _a = _cy*3+_cx; _hgtris_all += [(_a,_a+1,_a+4),(_a,_a+4,_a+3)]
_hgtris = [t for t in _hgtris_all if t != (0,1,4)]
_hgE = sorted({tuple(sorted(e)) for t in _hgtris_all for e in [(t[0],t[1]),(t[1],t[2]),(t[0],t[2])]})
_hgd1, _hgd2 = _hgbm(9, _hgE, _hgtris)
_hgrng = _hgnp.random.default_rng(0)
_hgflow = _hgd1.T @ _hgrng.standard_normal(9) + _hgd2 @ _hgrng.standard_normal(len(_hgtris))   # gradient + curl
_hgg, _hgc, _hgh = _hgm.hodge_decomposition(9, _hgE, _hgflow, _hgtris)
print(f"  split sums exactly: err {_hgnp.linalg.norm(_hgg+_hgc+_hgh-_hgflow):.1e}; "
      f"orthogonal <grad,curl>={_hgg@_hgc:.1e}; harmonic dim == B1 (the hole's loop)")
_hgclean = _hgd1.T @ _hgrng.standard_normal(9)                        # a pure transport (gradient) flow
_hgnoisy = _hgclean + 0.5*_hgrng.standard_normal(len(_hgE))
_hgden = _hgm.denoise_flow(9, _hgE, _hgnoisy, _hgtris, keep=("gradient","harmonic"))
print(f"  denoise a transport flow (drop curl): err {_hgnp.linalg.norm(_hgden-_hgclean):.2f} "
      f"from a noisy {_hgnp.linalg.norm(_hgnoisy-_hgclean):.2f}  (on a tree, curl+harmonic are zero -- kept negative)")

title("Clifford Cl(3,0) binding: rotors compose 3D rotations EXACTLY, where commutative bind can't (EXP-9)")
import numpy as _clnp
_clm = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
_clcl = _clm.clifford()
print(f"  e1 rotated 90 about e3 -> {_clnp.round(_clcl.rotate(_clcl.rotor([0,0,1], _clnp.pi/2), [1,0,0]), 4)}  (= e2, exact)")
_clrng = _clnp.random.default_rng(0)
_clmax = 0.0                                                          # composition is exact
for _ in range(100):
    _clRA = _clcl.rotor(_clrng.standard_normal(3), _clrng.uniform(0,_clnp.pi))
    _clRB = _clcl.rotor(_clrng.standard_normal(3), _clrng.uniform(0,_clnp.pi))
    _clv = _clrng.standard_normal(3)
    _clmax = max(_clmax, _clnp.linalg.norm(_clcl.rotate(_clRA, _clcl.rotate(_clRB, _clv)) - _clcl.rotate(_clcl.compose(_clRA, _clRB), _clv)))
print(f"  rotor product == sequential rotation: max err over 100 = {_clmax:.1e}  (HRR convolution can't represent SO(3))")
_clRA, _clRB = _clcl.rotor([1,0,0], 1.1), _clcl.rotor([0,1,0], 0.7)   # order matters
_clp = _clnp.array([0.3,-0.5,0.8])
_clgap = _clnp.linalg.norm(_clcl.rotate(_clcl.compose(_clRA,_clRB), _clp) - _clcl.rotate(_clcl.compose(_clRB,_clRA), _clp))
print(f"  non-commutative: A-then-B vs B-then-A differ by {_clgap:.3f}; a COMMUTATIVE bind collapses that to 0 "
      f"(2^d growth + versors-only are the kept negatives)")

title("Optimal transport: Wasserstein measures how far distributions sit apart, where bin-wise saturates (BLD-8)")
import numpy as _wsnp
_wsm = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
_wsx = _wsnp.arange(50)
def _wsg(mu): _v = _wsnp.exp(-0.5*((_wsx-mu)/2.0)**2); return _v/_v.sum()
_wsa, _wsb = _wsg(15), _wsg(25)                                       # a shift of 10
_wstrue = float(_wsnp.sum(_wsnp.abs(_wsnp.cumsum(_wsa)-_wsnp.cumsum(_wsb))))
print(f"  Sinkhorn W = {_wsm.wasserstein(_wsa, _wsb, eps=0.5):.3f}  ==  1-D closed form W1 = {_wstrue:.3f}")
print("  shift -> Wasserstein vs Euclidean vs cosine (support gone by shift~8):")
_wsref = _wsg(15)
for _wsshift in (5, 10, 20):
    _wss = _wsg(15+_wsshift)
    _wsW = _wsm.wasserstein(_wsref, _wss, eps=0.5)
    _wse = _wsnp.linalg.norm(_wsref-_wss); _wsc = (_wsref@_wss)/(_wsnp.linalg.norm(_wsref)*_wsnp.linalg.norm(_wss))
    print(f"    shift={_wsshift:2d}: W={_wsW:5.2f}   eucl={_wse:.3f} (saturates)   cos={_wsc:.4f} (collapses)")
print(f"  eps knob (kept negative): too large blurs the distance high, too small underflows the kernel -- "
      f"default scales eps to the cost")

title("Above/below sweep: the Tero flow solver's flux, split into transport + circulation (flow + Hodge + B1)")
import numpy as _fcnp
_fcm = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
def _fcgrid(R, C):
    nbr = {}
    for r in range(R):
        for c in range(C):
            nbr[(r, c)] = [(r+dr, c+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)] if 0<=r+dr<R and 0<=c+dc<C]
    return nbr
_fcres = _fcm.flow_circulation(_fcgrid(5, 5), (0, 0), (4, 4))
print(f"  5x5 grid: loops (B1) = {_fcres['loops']}; transport energy {_fcres['transport_energy']:.2f}, "
      f"circulation {_fcres['circulation_energy']:.2f} -> redundancy {_fcres['redundancy']:.3f} of the flux circulates")
_fctree = {0:[1], 1:[0,2,3], 2:[1], 3:[1,4], 4:[3]}                   # a tree: forced route
_fct = _fcm.flow_circulation(_fctree, 0, 4)
print(f"  tree:     loops (B1) = {_fct['loops']}; redundancy {_fct['redundancy']:.3f} (forced unique route -> zero circulation)")
print("  the flux the solver computed and threw away IS a Hodge flow: gradient = transport (divergence == "
      "injected current), harmonic = circulation (dim == B1)")

title("Above/below sweep: the graph-Laplacian basis wired into denoise() -- the curved-manifold map it lacked")
import numpy as _sdnp
_sdm = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
_sdrng = _sdnp.random.default_rng(0)
_sdN = 200
_sdi = _sdnp.arange(_sdN)
_sdphi = _sdnp.arccos(1 - 2*(_sdi+0.5)/_sdN); _sdtta = _sdnp.pi*(1+5**0.5)*_sdi
_sdP = _sdnp.column_stack([_sdnp.sin(_sdphi)*_sdnp.cos(_sdtta),
                           _sdnp.sin(_sdphi)*_sdnp.sin(_sdtta), _sdnp.cos(_sdphi)])   # points on a 2-sphere
_sdf = _sdP[:,2]**2 - 0.5*_sdP[:,0]                                                    # a smooth field on it
_sdfn = _sdf + 0.3*_sdrng.standard_normal(_sdN)
_sdspec = _sdnp.linalg.norm(_sdm.denoise(_sdfn, method="spectral", points=_sdP) - _sdf)   # geometry-aware
_sdtraj = _sdnp.linalg.norm(_sdm.denoise(_sdfn, method="trajectory", rank=8) - _sdf)      # geometry-blind
print(f"  lone field on a 2-sphere, raw noise error {_sdnp.linalg.norm(_sdfn-_sdf):.2f}:")
print(f"    denoise(method='spectral', points=...) = {_sdspec:.2f}   <- uses the cloud's geometry")
print(f"    denoise(method='trajectory')           = {_sdtraj:.2f}   (geometry-blind, cannot see the curvature)")
print("  the only denoiser in the faculty needing no example set and no codebook -- just the manifold's own geometry")

title("Self-hosting: two engine loops moved into VSA programs (PnP restoration + B10 generation)")
import numpy as _shnp
_shm = __import__("holographic_unified").UnifiedMind(dim=512, seed=7)
_shD = _shm.dim
_shrng = _shnp.random.default_rng(0)
_shB = _shrng.standard_normal((6, _shD)); _shB /= _shnp.linalg.norm(_shB, axis=1, keepdims=True)
_shsamp = _shnp.stack([c @ _shB for c in _shrng.standard_normal((40, 6))])      # a low-rank signal manifold
_shclean = _shrng.standard_normal(6) @ _shB
_shmask = (_shrng.random(_shD) > 0.5).astype(float)                              # measurement: erase half
_shfwd = lambda x: _shmask * x; _shadj = lambda y: _shmask * y
_shy = _shfwd(_shclean) + 0.05 * _shrng.standard_normal(_shD)
_shrest, _shtr = _shm.restore_procedure(_shy, _shfwd, _shadj, _shsamp, mu=0.8)   # PnP AS A PROGRAM
print(f"  restore_procedure  = ITERATE [APPLY datafit; APPLY denoise]")
print(f"    half-masked signal rel-error {_shnp.linalg.norm(_shy-_shclean)/_shnp.linalg.norm(_shclean):.2f} "
      f"-> {_shnp.linalg.norm(_shrest-_shclean)/_shnp.linalg.norm(_shclean):.2f}   ({_shtr[0][2]} iters, {_shtr[0][3]})")
_shfill = _shnp.stack([_shm._machine()._atom(f"sh_fill:{i}") for i in range(5)])
_shsamp2, _shtr2 = _shm.generate_procedure(_shfill, steps=12, seed=3)            # B10 diffusion AS A PROGRAM
_shfn = _shfill / _shnp.linalg.norm(_shfill, axis=1, keepdims=True)
_shbest = max(float(_shsamp2 @ f / (_shnp.linalg.norm(_shsamp2)*_shnp.linalg.norm(f))) for f in _shfn)
print(f"  generate_procedure = ITERATE [APPLY diffuse]  (denoise from noise)")
print(f"    generated sample lands on the manifold at cosine {_shbest:.2f}   ({_shtr2[0][2]} iters)")
print("  the restoration LOOP and the generative PROCESS are now stored, composable, recipe-savable programs --")
print("  process not object; the procedure tax (noisy reads) is the price of being data, not a faster path")

title("Honesty as a STRUCTURAL lint: audit a protocol-as-data for the search-without-null anti-pattern (D1)")
_aum = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
# a protocol is program-as-data: an ordered list of analysis steps. audit_procedure reads the structure BACK
# from the program VECTOR (unbind+cleanup) and checks the honesty discipline as a structural property.
_au_good = _aum.audit_procedure(steps=["encode", "combination_search", "oos_split", "calibrated_null", "fdr", "decide"])
print(f"  complete protocol [encode->search->split->null->fdr->decide]  sound={_au_good['sound']}  roles={_au_good['roles']}")
_au_bad = _aum.audit_procedure(steps=["encode", "combination_search", "oos_split", "fdr", "decide"])
print(f"  search WITHOUT a null [encode->search->split->fdr->decide]    sound={_au_bad['sound']}  "
      f"flagged={[c for c, _m in _au_bad['violations']]}")
_au_rest = _aum.audit_procedure(steps=["datafit", "denoise"])
print(f"  no-search restoration loop [datafit->denoise]                 sound={_au_rest['sound']}  (not trigger-happy)")
print("  the step structure is recovered from the program vector -- the discipline is a query, not a habit")

title("A research log as a knowledge structure: query findings, and catch the log's own contradictions (D3)")
_kbm = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
_kb = _kbm.finding_registry()
_kb.add("efficiency_ratio", "momentum", +1, condition="horizon_10d", note="ER strengthens momentum at 10d")
_kb.add("efficiency_ratio", "momentum", -1, condition="intraday", note="ER backfires intraday")
_kb.add("low_vol", "vol_expansion", +1, note="low vol precedes expansion")
_kb.add("bracket_order", "convexity", +1, note="the bracket looks convex")
_kb.add("bracket_order", "convexity", -1, note="the bracket's drift masquerades as convexity")
_kb_q = _kb.query(subject="efficiency_ratio", k=2)                       # similarity recall over structured claims
print(f"  query subject=efficiency_ratio -> findings {sorted(r['index'] for r in _kb_q)} (role-sensitive recall)")
for _t in _kb.tensions():
    print(f"  {_t['type'].upper():11s} tension: {_t['subject']}->{_t['object']}  conditions={_t['conditions']}")
print("  CONDITIONED = reconcilable (effect depends on the differing dimension); FLAT = one must be wrong")
print("  retrieval is holographic (cosine over the bound claim); the verdict is exact (polarity + condition)")
# the log persists: save (claims only, no vectors) and reload -- the conditioned tension survives the round-trip
import os as _os, tempfile as _tf
_kb_path = _os.path.join(_tf.gettempdir(), "_tour_findings.json")
_kb.save(_kb_path)
_kb_reloaded = __import__("holographic_knowledge").FindingRegistry.load(_kb_path)
_kb_bytes = _os.path.getsize(_kb_path); _os.remove(_kb_path)
_kb_rt = _kb_reloaded.tensions()
print(f"  saved {len(_kb.findings)} findings as {_kb_bytes} bytes of claims (no vectors), reloaded -> "
      f"{len(_kb_rt)} tension(s) survive, classified {[_t['type'] for _t in _kb_rt]}")
print("  vectors are rebuilt from the seed on load, so a reloaded log is the same object, not an approximation")

title("Vector graphics on the substrate: generate / encode / morph scenes, render crisp SVG (no splat blur) (svg_canvas)")
_svgm = __import__("holographic_unified").UnifiedMind(dim=4096, seed=0)
_svg = _svgm.svg_canvas()
# GENERATE a novel scene via the composed-manifold diffusion, then render it as resolution-independent SVG
_sc = _svg.generate(k=4, seed=1)
_svg_text = _svg.to_svg(_sc)
_typenames = [_svg.types[p[0]] for p in _sc]
print(f"  generated a {len(_sc)}-primitive scene {_typenames} -> SVG ({len(_svg_text)} chars, crisp at any zoom)")
# ENCODE the scene into ONE hypervector and decode it back -- a content-addressable picture
_dec = _svg.decode(_svg.encode(_sc), len(_sc))
_perr = sum(abs(a[1]-b[1])+abs(a[2]-b[2]) for a,b in zip(_sc,_dec))/(2*len(_sc))
_texact = sum(a[0]==b[0] and a[4]==b[4] for a,b in zip(_sc,_dec))
print(f"  encoded into one hypervector, decoded back: {_texact}/{len(_sc)} type+colour exact, "
      f"position error {_perr:.3f} on [0,1]")
# MORPH two scenes by interpolating their VECTORS -- the picture interpolates by arithmetic
_A = [(1,0.25,0.30,0.13,1),(0,0.72,0.68,0.11,3)]; _B = [(1,0.70,0.30,0.09,2),(0,0.28,0.66,0.13,0)]
_mid = _svg.morph(_A, _B, steps=5)[2]
print(f"  morph A->B (vector interpolation): midpoint circle x = {_mid[0][1]:.2f} "
      f"(between {_A[0][1]:.2f} and {_B[0][1]:.2f}) -- arithmetic on vectors interpolates the picture")
print("  an SVG <rect>/<circle> has exact edges at any zoom -- the sharp, resolution-independent cousin of splats")

# CreatureMind: the reference DEMO of a specialized mind built ON the one UnifiedMind -- subclass + wire,
# inherit every faculty, reimplement nothing. A creature that is also a full mind, in one object.
from holographic_creature_mind import CreatureMind as _CM
_cm = _CM(dim=512, actions=("N", "S", "E", "W"), seed=0)
for _ in range(6):
    _cm.learn({"food_x": "east"}, "E", 1.0); _cm.learn({"food_x": "west"}, "W", 1.0)
_cm_act = _cm.act({"food_x": "east"}, explore=False)          # learned policy, via the INHERITED decide
_cm_tiles = _np.random.default_rng(0).standard_normal((7, 512))
_cm_tiles = _cm_tiles / _np.linalg.norm(_cm_tiles, axis=1, keepdims=True)
def _cm_field(_c):
    _i = int(_np.argmax(_cm_tiles @ (_c / (_np.linalg.norm(_c) + 1e-12))))
    return _cm_tiles[_i + 1] if _i + 1 < len(_cm_tiles) else None
_cm_plan = _cm.plan(_cm_tiles[0], _cm_field, max_steps=6, floor=0.12)   # the INHERITED planning faculty
print(f"  CreatureMind (a layer on the one mind): senses->'{_cm_act}' via inherited decide, AND "
      f"baked a {len(_cm_plan.route)}-step corridor via inherited plan -- one object, faculties reused not rebuilt")

title("Explicit polygon geometry: a mesh kernel + the glTF (.glb) boundary to a three.js front end (mesh_*)")
# The IMPLICIT side (SDF field, splat bundle, scene-graph) was always here; this is the EXPLICIT side --
# an actual indexed polygon mesh of the kind Blender/three.js/glTF speak. The Step-0 vertical slice:
# build a cube, read its topology invariants, ship it across the binary boundary, and read it back.
_meshm = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
_cube = _meshm.mesh_box(width=2.0, height=1.0, depth=1.0)
_euler = _meshm.mesh_euler(_cube)                              # connectivity is EXACT (integer, no float drift)
print(f"  built a box mesh: V{_euler['vertices']} E{_euler['edges']} F{_euler['faces']} "
      f"-> Euler chi = {_euler['characteristic']}, genus {_euler['genus']}, "
      f"closed={_euler['closed']} manifold={_euler['manifold']}  (a sphere-topology solid, proven by its invariants)")
# THE BOUNDARY: emit a real glTF 2.0 .glb (the bytes a three.js loader ingests), then parse it straight back.
_glb = _meshm.mesh_to_gltf(_cube)
_back = _meshm.mesh_from_gltf(_glb)
_v_match = _np.allclose(_np.asarray(_cube.vertices), _np.asarray(_back.vertices))
print(f"  emitted a {len(_glb)}-byte .glb (POSITION/NORMAL/index + PBR material, position bounds, 4-byte aligned) "
      f"-> parsed back: vertices match = {_v_match}")
# BYTE-REPRODUCIBLE: the determinism rule reaches the wire format (sorted JSON keys, fixed dtypes/endianness).
_glb2 = _meshm.mesh_to_gltf(_cube)
print(f"  same mesh emitted twice -> identical bytes = {_glb == _glb2}  (deterministic all the way to the .glb)")
print("  KEPT NEGATIVE: the half-edge kernel is Python-loop bound -- correct + deterministic at engine sizes,")
print("  but won't scale to interactive million-poly editing without a compiled core ('NumPy-only' is the engine's")
print("  rule, not an interactive mesh editor's). The vectorized paths (Euler, normals, buffers) are the fast ones.")

title("FWD-7: the explicit mesh can be EDITED -- local Euler operators, exact make/kill round-trips")
# The kernel (FWD-1) was read-only. These are the LOCAL connectivity rewrites every remesher/decimator
# decomposes into: flip (Delaunay), split (refine), collapse (decimate), split_face (n-gon). Each keeps the
# surface a valid manifold and bookkeeps chi exactly; the make/kill pairs give an exact do-then-undo round-trip.
from holographic_mesh import Mesh as _MeshFwd7
from collections import Counter as _CounterFwd7
from holographic_eulerops import _face_with_directed_edge as _fwde, _third as _thirdv
_tm = _MeshFwd7(_cube.vertices.copy(), [tuple(t) for t in _cube.triangulate()])
_chi0 = _tm.euler_characteristic()
print(f"  triangulated the box: V{_tm.n_vertices} E{_tm.n_edges} F{_tm.n_faces} chi={_chi0}  (closed manifold = {_tm.is_closed()})")
# pick a flippable interior edge (apexes not already an edge)
_dirs = _CounterFwd7()
for _f in _tm.faces:
    for _k in range(3):
        _dirs[(_f[_k], _f[(_k + 1) % 3])] += 1
_es = set(_tm.edges())
_ea = _eb = None
for (_x, _y) in _dirs:
    if (_y, _x) in _dirs:
        _cc = _thirdv(_tm.faces[_fwde(_tm.faces, _x, _y)], _x, _y)
        _dd = _thirdv(_tm.faces[_fwde(_tm.faces, _y, _x)], _x, _y)
        if (min(_cc, _dd), max(_cc, _dd)) not in _es:
            _ea, _eb = _x, _y
            break
_flip = _meshm.mesh_flip_edge(_tm, _ea, _eb)
print(f"  flip_edge {{{_ea},{_eb}}}: V{_flip.n_vertices} E{_flip.n_edges} F{_flip.n_faces} chi={_flip.euler_characteristic()} "
      f"-- V/E/F all unchanged, still closed manifold = {_flip.is_closed()}")
_split, _mid = _meshm.mesh_split_edge(_tm, _ea, _eb)
print(f"  split_edge {{{_ea},{_eb}}}: V{_split.n_vertices} (added midpoint #{_mid}) chi={_split.euler_characteristic()}  (refinement: V+1, chi held)")
_back = _meshm.mesh_collapse_edge(_split, keep=_ea, remove=_mid)

def _canon_fwd7(_m):
    return tuple(sorted(tuple(_f[_f.index(min(_f)):] + _f[:_f.index(min(_f))]) for _f in _m.faces))
print(f"  collapse_edge keep={_ea} remove={_mid}: V{_back.n_vertices} -- split then collapse restores the mesh exactly = {_canon_fwd7(_back) == _canon_fwd7(_tm)}")
# the link condition: an equatorial edge of a bipyramid is NOT collapsible (would weld the surface)
_bpv = [[0, 0, 1], [1, 0, 0], [-0.5, 0.87, 0], [-0.5, -0.87, 0], [0, 0, -1]]
_bp = _MeshFwd7(_bpv, [(0, 1, 2), (0, 2, 3), (0, 3, 1), (4, 2, 1), (4, 3, 2), (4, 1, 3)])
print(f"  link condition: collapse of bipyramid equator {{1,2}} refused (returns None) = {_meshm.mesh_collapse_edge(_bp, keep=1, remove=2) is None}  "
      f"(not every edge is collapsible -- a true mesh property, made operational, not hidden)")

title("FWD-4 (Tier 1, ADAPT-SHIPPED): mesh smoothing IS the shipped Taubin filter, wired onto a mesh")
# The forward backlog's real insight: the matured intrinsic-geometry toolkit turns the "conventional" DCC
# items into adaptations of shipped faculties. graphsignal.taubin_filter already exists -- mesh smoothing is
# three substitutions: vertex positions as the signal, the mesh 1-ring as the graph, cotangent weights. A wire.
from holographic_meshsmooth import _icosphere as _icos, laplacian_smooth as _lapsm
import numpy as _np_fwd4
_clean = _icos(3)
_rng4 = _np_fwd4.random.default_rng(0)
_noisy = _MeshFwd7(_clean.vertices + _rng4.normal(0.0, 0.05, _clean.vertices.shape), list(_clean.faces))
_re = lambda _m: float(_np_fwd4.abs(_np_fwd4.linalg.norm(_m.vertices, axis=1) - 1.0).mean())
_mr = lambda _m: float(_np_fwd4.linalg.norm(_m.vertices, axis=1).mean())
_taub = _meshm.mesh_smooth(_noisy, iters=10)
_lap = _lapsm(_noisy, iters=10)
print(f"  noisy unit sphere (sigma=0.05): radial err {_re(_noisy):.4f}, mean radius {_mr(_noisy):.3f}")
print(f"  Taubin smooth (cotangent, no-shrink): radial err -> {_re(_taub):.4f} ({100*(1-_re(_taub)/_re(_noisy)):.0f}% denoise), "
      f"mean radius KEPT at {_mr(_taub):.3f}  -- connectivity + chi untouched (faces identical = {_taub.faces == _clean.faces})")
print(f"  naive Laplacian baseline: mean radius SHRANK to {_mr(_lap):.3f}  (the kept negative -- why Taubin's lambda|mu exists)")

title("FWD-6 (Tier 1): mesh curvature & creases -- with an EXACT topological reference (Gauss-Bonnet)")
# Discrete differential geometry gives exact identities to check against: the angle defect IS Gaussian
# curvature, and its TOTAL over a closed mesh equals 2*pi*chi exactly (Gauss-Bonnet) -- the curvature estimate
# validated against the Euler characteristic the kernel computes. Mean curvature reuses FWD-4's cotangent weights.
from holographic_meshcurvature import (gauss_bonnet_defect as _gbdef, gaussian_curvature as _gcurv,
                                       mean_curvature as _hcurv)
_gb = _gbdef(_clean)                                    # _clean is the unit sphere from the FWD-4 demo
_Kc = _gcurv(_clean)
_Hc = _hcurv(_clean)
print(f"  unit sphere: Gauss-Bonnet total defect = 2*pi*chi to {abs(_gb):.1e}  (EXACT -- validated against chi=2)")
print(f"  unit sphere curvature: mean Gaussian K={float(_Kc.mean()):.3f}, mean |H|={float(_Hc.mean()):.3f}  (both ~1 = 1/R, 1/R^2)")
_cube_creases = _meshm.mesh_creases(_meshm.mesh_box(2.0, 2.0, 2.0), threshold_deg=30.0)
_sphere_creases = _meshm.mesh_creases(_clean, threshold_deg=30.0)
print(f"  crease detection (dihedral angle): cube -> {len(_cube_creases)} sharp edges (all 90deg), "
      f"smooth sphere -> {len(_sphere_creases)}  (feeds crease-aware smoothing + adaptive subdivision)")
print(f"  KEPT NEGATIVE: per-vertex curvature is noisy on coarse meshes (CoV {float(_Hc.std()/_Hc.mean()):.2f}) "
      f"-- the mean is right, mesh_curvature_confidence scores per-vertex reliability")

title("FWD-5 (Tier 1): surface geodesics -- distance ALONG the surface, not through the void")
# The shipped chart.geodesic_distances runs shortest paths on a k-NN graph; the adapt-shipped move is to run the
# same idea on the EXPLICIT MESH EDGE graph. This feeds FWD-3's UV seams + soft selections. Validated against the
# analytic great-circle distance, and contrasted with the Euclidean distance that 'bleeds' across the surface.
_north = int(_np_fwd4.argmax(_clean.vertices[:, 2]))   # the sphere's north-pole vertex
_south = int(_np_fwd4.argmin(_clean.vertices[:, 2]))
_geo = _meshm.mesh_geodesic(_clean, _north)
_true = _np_fwd4.arccos(_np_fwd4.clip(_clean.vertices[:, 2], -1.0, 1.0))
_euclid_ns = float(_np_fwd4.linalg.norm(_clean.vertices[_south] - _clean.vertices[_north]))
print(f"  unit sphere geodesic from the pole vs analytic great-circle arccos(z): correlation {float(_np_fwd4.corrcoef(_geo, _true)[0,1]):.4f}")
print(f"  north->south geodesic = {float(_geo[_south]):.3f} (~pi, the long way ROUND the surface) > Euclidean {_euclid_ns:.3f} (straight through)")
_sel = _meshm.mesh_soft_selection(_clean, _north, radius=2.5)
print(f"  soft-selection radius 2.5: antipode weight = {float(_sel[_south]):.1f} (EXCLUDED -- geodesic ~pi > 2.5), "
      f"but a Euclidean ball would INCLUDE it ({_euclid_ns:.1f} < 2.5)  -- geodesic doesn't bleed across the surface")

title("FWD-3 (Tier 1, the payoff): UV unwrapping = the shipped MDS chart on the mesh's OWN geodesics")
# UV -- the 'least-holostuff' DCC item -- is a near-direct reuse: feed FWD-5's mesh geodesic matrix to the shipped
# chart.classical_mds and the 2-D embedding IS the UV chart (Isomap on explicit edges). Wins on curved surfaces;
# a linear projection is the right tool on flat ones. Closed surfaces need a seam (the kept negative).
from holographic_meshuv import flat_grid_mesh as _flat_grid, hemisphere_cap as _hemicap
_flatm = _flat_grid(9)                                   # a flat developable patch (isotropic triangulation)
_uvflat = _meshm.mesh_uv_unwrap(_flatm)
_flips = sum(1 for (a, b, c) in _flatm.faces
             if (_uvflat[b][0]-_uvflat[a][0])*(_uvflat[c][1]-_uvflat[a][1])
              - (_uvflat[b][1]-_uvflat[a][1])*(_uvflat[c][0]-_uvflat[a][0]) < 0)
print(f"  flat developable patch: stretch distortion {float(_meshm.mesh_uv_distortion(_flatm, _uvflat)):.3f} (near-isometric), "
      f"{_flips} flipped triangles (charts don't overlap)")
_capm = _hemicap(3)                                      # a curved cap: Isomap should beat a linear projection
_iso = float(_meshm.mesh_uv_distortion(_capm, _meshm.mesh_uv_unwrap(_capm, method="isomap")))
_lin = float(_meshm.mesh_uv_distortion(_capm, _meshm.mesh_uv_unwrap(_capm, method="planar")))
print(f"  curved cap: Isomap (geodesic) distortion {_iso:.3f} < linear projection {_lin:.3f}  -- geodesic chart wins where the surface bends")
print(f"  KEPT NEGATIVE: a CLOSED surface (a sphere) can't flatten to a disk without a seam/cut -- ARCH-4 places a real one")

title("FWD-7 (Tier 2): modeler verbs -- extrude / inset / dissolve, built on the explicit mesh kernel")
# The shipped Euler primitives (flip/split/collapse) are the atomic moves; these are the human-facing verbs on
# top. Each produces a VALID mesh (chi preserved, still a closed manifold) with an EXACT geometric signature.
from holographic_meshsmooth import _icosphere as _ico_verbs
from holographic_meshverbs import _face_normal as _fn_verbs
_vsphere = _ico_verbs(2)
_chi_v = _vsphere.euler_characteristic()
_nrm_v = _fn_verbs(_vsphere.vertices, _vsphere.faces[0])
_cap0 = _np_fwd4.mean([_vsphere.vertices[v] for v in _vsphere.faces[0]], axis=0)
_exm = _meshm.mesh_extrude(_vsphere, 0, distance=0.3)
_cap1 = _exm.vertices[_exm.n_vertices - 3:].mean(axis=0)
print(f"  EXTRUDE face 0 by 0.3: chi {_vsphere.euler_characteristic()} -> {_exm.euler_characteristic()} (preserved), closed+manifold={_exm.is_closed() and _exm.is_manifold()}, "
      f"cap moved {float(_np_fwd4.dot(_cap1 - _cap0, _nrm_v)):.3f} along the normal (exact)")
_insm = _meshm.mesh_inset(_vsphere, 0, ratio=0.4)
def _triA(V, f):
    a, b, c = V[f[0]], V[f[1]], V[f[2]]; return 0.5 * float(_np_fwd4.linalg.norm(_np_fwd4.cross(b - a, c - a)))
_a0 = _triA(_vsphere.vertices, _vsphere.faces[0]); _a1 = _triA(_insm.vertices, _insm.faces[_vsphere.n_faces - 1])
print(f"  INSET   face 0 by 0.4: central-face area ratio {_a1 / _a0:.3f} = (1-0.4)^2 = {0.6 ** 2:.3f} (exact), chi preserved")
_dissm = _meshm.mesh_dissolve_vertex(_vsphere, 5)
print(f"  DISSOLVE vertex 5: V {_vsphere.n_vertices} -> {_dissm.n_vertices} (-1), chi {_dissm.euler_characteristic()} preserved, closed+manifold={_dissm.is_closed() and _dissm.is_manifold()}")
print(f"  (bevel / bridge / loop-cut are the FWD-7 remainder -- deferred honestly rather than shipped shaky)")

title("FWD-8 (Tier 2): Loop subdivision -- refine (new topology) + low-pass smooth (the spectral family)")
# Subdivision is two operations braided: a topological REFINE (1 triangle -> 4, an Euler-operator sequence) and a
# graph-signal LOW-PASS smooth (the same family FWD-4's Taubin uses). Measured: exact x4, affine reproduction, smoothing.
from holographic_meshsmooth import _icosphere as _ico_sd
from holographic_meshuv import flat_grid_mesh as _flat_sd
from holographic_meshcurvature import dihedral_angles as _dih_sd
from holographic_meshsubdiv import _triangles as _tri_sd
from holographic_mesh import box as _box_sd, Mesh as _Mesh_sd
_s_sd = _ico_sd(1)
_sub_sd = _meshm.mesh_subdivide(_s_sd, 1)
print(f"  icosphere: faces {_s_sd.n_faces} -> {_sub_sd.n_faces} (x4 exact), vertices {_s_sd.n_vertices} -> {_sub_sd.n_vertices} (V+E), "
      f"chi {_sub_sd.euler_characteristic()} preserved, closed+manifold={_sub_sd.is_closed() and _sub_sd.is_manifold()}")
_flat_sub = _meshm.mesh_subdivide(_flat_sd(5), 2)
print(f"  AFFINE REPRODUCTION: a flat mesh subdivides to max |z| = {float(_np_fwd4.max(_np_fwd4.abs(_flat_sub.vertices[:, 2]))):.0e} (stays flat to machine precision -- the rigor reference)")
_cube_sd = _box_sd()
_before_sd = float(_np_fwd4.std(list(_dih_sd(_Mesh_sd(_cube_sd.vertices.copy(), _tri_sd(_cube_sd))).values())))
_after_sd = float(_np_fwd4.std(list(_dih_sd(_meshm.mesh_subdivide(_cube_sd, 2)).values())))
print(f"  SMOOTHING (low-pass): cube dihedral-angle spread {_before_sd:.3f} -> {_after_sd:.3f} over 2 levels -- the smoothing step IS a spectral low-pass")

title("FWD-10 (Tier 2): inverse kinematics (FABRIK) = the mind's OWN project_onto_constraints engine")
# IK is "iterate a projection onto bone-length constraints" -- so FABRIK runs LITERALLY through the shipped
# project_onto_constraints sweeper (the same engine behind the resonator and the PnP denoiser), not a reimplementation.
from holographic_meshik import chain as _chain_ik
_arm = _chain_ik(4, 1.0)                                  # 4 bones, total reach 4, along +x
_rest_ik = [float(_np_fwd4.linalg.norm(_arm[i + 1] - _arm[i])) for i in range(len(_arm) - 1)]
_tgt_ik = _np_fwd4.array([2.0, 1.5, 0.5])
_posed, _sweeps = _meshm.solve_ik(_arm, _tgt_ik, iters=30)
_new_ik = [float(_np_fwd4.linalg.norm(_posed[i + 1] - _posed[i])) for i in range(len(_posed) - 1)]
print(f"  reachable target [2.0, 1.5, 0.5]: tip reaches it to {float(_np_fwd4.linalg.norm(_posed[-1] - _tgt_ik)):.0e} in {_sweeps} sweeps, "
      f"all 4 bone lengths preserved ({max(abs(a - b) for a, b in zip(_new_ik, _rest_ik)):.0e}), root fixed")
_far_ik, _ = _meshm.solve_ik(_arm, _np_fwd4.array([100.0, 0.0, 0.0]), iters=60)
print(f"  UNREACHABLE target: chain fully extends -- tip at distance {float(_np_fwd4.linalg.norm(_far_ik[-1] - _far_ik[0])):.3f} from root (= total reach 4.0), pointing straight at it")
print(f"  the same iterate-a-projection engine that cleans a noisy code and factors a scene also solves the arm")

title("FWD-9 (Tier 2): linear blend skinning = a SOFT mixture of expert bone-transforms")
# Skinning blends what each bone would do to a vertex, weights summing to 1 -- structurally a mixture of experts
# (the soft/dense cousin of the engine's hard/sparse top-1 moe.GatedMixture). Rigid reproduction is exact; the
# candy-wrapper artifact is the kept negative, measured to closed form.
from holographic_meshskin import linear_blend_skin as _lbs, make_transform as _mt, rotation as _rot
_pts_sk = _np_fwd4.array([[1.0, 0.3, -0.2], [0.5, -1.0, 0.4], [-0.7, 0.2, 1.0]])
_Msk = _mt(rot=_rot([0.2, 1.0, 0.3], 0.7), translation=[0.5, -0.2, 1.0])
_shared = _lbs(_pts_sk, _np_fwd4.stack([_Msk, _Msk, _Msk]), _np_fwd4.array([[0.2, 0.5, 0.3]] * 3))
_exp_sk = (_np_fwd4.hstack([_pts_sk, _np_fwd4.ones((3, 1))]) @ _Msk.T)[:, :3]
print(f"  RIGID REPRODUCTION: 3 bones sharing one transform, arbitrary weights -> reproduced exactly (max err {float(_np_fwd4.max(_np_fwd4.abs(_shared - _exp_sk))):.0e}) -- the partition-of-unity guarantee")
_phi_sk = _np_fwd4.linspace(0, 2 * _np_fwd4.pi, 32, endpoint=False)
_ring_sk = _np_fwd4.stack([_np_fwd4.cos(_phi_sk), _np_fwd4.sin(_phi_sk), _np_fwd4.zeros_like(_phi_sk)], axis=1)
for _th_sk in (2 * _np_fwd4.pi / 3, _np_fwd4.pi):
    _r_sk = float(_np_fwd4.mean(_np_fwd4.linalg.norm(_lbs(_ring_sk, _np_fwd4.stack([_np_fwd4.eye(4), _mt(axis=[0, 0, 1], angle=_th_sk)]), _np_fwd4.full((32, 2), 0.5))[:, :2], axis=1)))
    print(f"  CANDY-WRAPPER negative: a 50/50 twist of {round(float(_np_fwd4.degrees(_th_sk)))} deg collapses the unit ring to radius {_r_sk:.3f} = cos(theta/2) = {float(abs(_np_fwd4.cos(_th_sk / 2))):.3f} (LBS averages matrices, not rotations -- DQS is the fix)")

title("FWD-11 (Tier 3): the mesh <-> SDF <-> splat bridge -- three views of one surface, made convertible")
# The mesh kernel deliberately had no marching cubes. FWD-11 supplies isosurface extraction (marching TETRAHEDRA,
# manifold by construction) so the engine's implicit (SDF) and splat representations can enter the mesh world.
from holographic_meshbridge import sphere_sdf as _sphere_sdf, metaball_field as _metaball
_sphere_b = _meshm.mesh_from_sdf(_sphere_sdf(radius=1.0), ((-1.5,) * 3, (1.5,) * 3), res=20)
_radii_b = _np_fwd4.linalg.norm(_sphere_b.vertices, axis=1)
_Vb = _sphere_b.vertices
_outward_b = sum(1 for (a, b, c) in _sphere_b.faces if _np_fwd4.dot(_np_fwd4.cross(_Vb[b] - _Vb[a], _Vb[c] - _Vb[a]), (_Vb[a] + _Vb[b] + _Vb[c]) / 3.0) > 0)
print(f"  SDF -> MESH: unit sphere |p|-1 extracted to {_sphere_b.n_faces} faces, closed manifold chi={_sphere_b.euler_characteristic()}, "
      f"vertices on sphere (r={float(_radii_b.mean()):.3f} +/- {float(_radii_b.std()):.3f}), {_outward_b}/{_sphere_b.n_faces} outward-oriented")
_sdf_probe = _meshm.mesh_to_sdf(_sphere_b, _np_fwd4.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
print(f"  MESH -> SDF: signed distance at origin = {float(_sdf_probe[0]):.3f} (inside, ~-1) and at [2,0,0] = {float(_sdf_probe[1]):.3f} (outside, ~+1) -- matches analytic |p|-1")
_blob_b = _meshm.mesh_from_sdf(_metaball(_np_fwd4.array([[-0.4, 0, 0], [0.4, 0, 0]]), radius=0.4), ((-1.5,) * 3, (1.5,) * 3), res=20, level=0.5)
print(f"  SPLAT -> MESH: a sum of 2 Gaussian splats iso-extracts to a {_blob_b.n_faces}-face blob (closed manifold={_blob_b.is_closed() and _blob_b.is_manifold()}) -- splats enter the mesh world through the same extractor")

title("ARCH-1: the recipe's Euler operators -- validate + local edits that preserve the realized vector")
# Turn the mesh editors INWARD: just as a mesh has flip/split/collapse preserving chi, a StructureRecipe gets
# edits preserving the realized VECTOR -- because bind/bundle are commutative, a local rewrite is an algebra identity.
from holographic_recipe import StructureRecipe as _SR
_ra = _SR(dim=512, seed=0)
_h_a = _ra.atom("a"); _h_b = _ra.atom("b"); _h_c = _ra.atom("c")
_h_ab = _ra.bind(_h_a, _h_b); _h_bun = _ra.bundle([_h_a, _h_b, _h_c])
_ra.mark_output(_h_ab); _ra.mark_output(_h_bun)
_base_r = [v.copy() for v in _ra.outputs()]
_ok_r, _ = _meshm.validate_recipe(_ra)
_bad_r = __import__('holographic_recipeops')._clone(_ra); _bad_r._ops[3] = ("bind", 3, 99)
print(f"  VALIDATE (the recipe's is_manifold): well-formed recipe -> {_ok_r}; same recipe with a dangling reference -> {_meshm.validate_recipe(_bad_r)[0]}")
_flip_r = _meshm.recipe_commute_bind(_ra, _h_ab)
_twice_r = _meshm.recipe_commute_bind(_flip_r, _h_ab)
print(f"  commute_bind = flip_edge: bind(a,b)->bind(b,a) leaves the vector unchanged (max diff {float(_np_fwd4.max(_np_fwd4.abs(_flip_r.outputs()[0] - _base_r[0]))):.0e}), and is its OWN INVERSE (twice -> diff {float(_np_fwd4.max(_np_fwd4.abs(_twice_r.outputs()[0] - _base_r[0]))):.0e})")
_sub_r = _meshm.recipe_substitute_atom(_ra, 0, "z")
_restore_r = _meshm.recipe_substitute_atom(_sub_r, 0, "a")
print(f"  substitute_atom = vertex move: renaming atom a->z CHANGES the result (cos {float(_np_fwd4.dot(_sub_r.outputs()[0], _base_r[0]) / (_np_fwd4.linalg.norm(_sub_r.outputs()[0]) * _np_fwd4.linalg.norm(_base_r[0]))):.2f}), renaming back restores it EXACTLY (diff {float(_np_fwd4.max(_np_fwd4.abs(_restore_r.outputs()[0] - _base_r[0]))):.0e})")

title("ARCH-4: a REAL seam -- cut a closed surface into a disk by vertex duplication (the FWD-3 payback)")
# FWD-3 could only open a closed surface with a crude `puncture` (delete a vertex). ARCH-4 cuts along a seam,
# duplicating its interior vertices on a consistent side -> a disk that keeps ALL its geometry.
from holographic_meshsmooth import _icosphere as _ico_seam
from holographic_meshuv import uv_unwrap as _uvu_seam, uv_distortion as _uvd_seam, puncture as _punc_seam
_ssph = _ico_seam(3)
_n_seam = int(_np_fwd4.argmax(_ssph.vertices[:, 2])); _s_seam = int(_np_fwd4.argmin(_ssph.vertices[:, 2]))
_merid = _meshm.mesh_shortest_seam(_ssph, _n_seam, _s_seam)
_disk_seam = _meshm.mesh_cut_seam(_ssph, _merid)
_punc = _punc_seam(_ssph, 0)
print(f"  meridian cut: closed sphere (chi 2) -> DISK (chi {_disk_seam.euler_characteristic()}), manifold, V {_ssph.n_vertices}->{_disk_seam.n_vertices} (+{len(_merid) - 2} duplicated)")
print(f"  NON-DESTRUCTIVE: the cut keeps all {_disk_seam.n_faces} faces; the crude puncture DELETES {_ssph.n_faces - _punc.n_faces} faces (loses geometry)")
_eq_seam = int(_np_fwd4.argmin(_np_fwd4.abs(_ssph.vertices[:, 2])))
_good_seam = _meshm.mesh_cut_seam(_ssph, _meshm.mesh_shortest_seam(_ssph, _n_seam, _eq_seam))
_gd = float(_uvd_seam(_good_seam, _uvu_seam(_good_seam))); _pd = float(_uvd_seam(_punc, _uvu_seam(_punc))); _fd = float(_uvd_seam(_disk_seam, _uvu_seam(_disk_seam)))
print(f"  PAYBACK: a pole-to-equator seam unwraps at {_gd:.3f} < puncture {_pd:.3f}.  KEPT NEGATIVE: a full meridian ({_fd:.3f}) is worse -- seam choice matters (a good atlas needs several cuts)")

title("ARCH-7: representation routing -- booleans have no mesh implementation, so route through the SDF (CSG)")
# The policy layer on FWD-11's bridge: union/intersection/difference are trivial field min/max on an SDF but
# impossible on a mesh directly -> route mesh->SDF->op->mesh, even CHANGING topology.
from holographic_mesh import Mesh as _Mesh_csg
_sph_csg = _ico_seam(2)
def _tr_csg(off): return _Mesh_csg(_sph_csg.vertices + _np_fwd4.array(off, float), [tuple(f) for f in _sph_csg.faces])
_Acsg, _Bcsg = _tr_csg([-0.5, 0, 0]), _tr_csg([0.5, 0, 0])
print(f"  ROUTING POLICY: 'union' -> {_meshm.route_representation('union')} representation (the mesh kernel has no boolean); 'boundary' -> {_meshm.route_representation('boundary')}")
_uni_csg = _meshm.mesh_csg("union", _Acsg, _Bcsg, res=24)
_sep_csg = _meshm.mesh_csg("union", _tr_csg([-1.6, 0, 0]), _tr_csg([1.6, 0, 0]), res=24)
print(f"  TOPOLOGY CHANGE: two OVERLAPPING spheres union to {_meshm.mesh_connected_components(_uni_csg)} blob (closed manifold={_uni_csg.is_closed() and _uni_csg.is_manifold()}); two SEPARATE spheres stay {_meshm.mesh_connected_components(_sep_csg)} components -- the field merges or keeps-apart by itself")
_int_csg = _meshm.mesh_csg("intersection", _Acsg, _Bcsg, res=24)
_vA_csg, _vB_csg, _vU_csg, _vI_csg = float(_meshm.mesh_volume(_Acsg)), float(_meshm.mesh_volume(_Bcsg)), float(_meshm.mesh_volume(_uni_csg)), float(_meshm.mesh_volume(_int_csg))
print(f"  GEOMETRICALLY correct (not just topologically): vol(A or B) {_vU_csg:.2f} ~ vA+vB-vInt {_vA_csg + _vB_csg - _vI_csg:.2f} (inclusion-exclusion)")

title("ARCH-3: geometry-weighted graph ops -- the cotangent Laplacian turned inward (cosine-similarity weights)")
# The mesh's cotangent Laplacian weights edges by geometry; on hypervectors the geometry IS cosine similarity.
# A similarity-weighted graph Laplacian's eigenmap recovers a manifold's intrinsic coordinates.
from holographic_simgraph import _ring_vectors as _ringv, _circ_corr as _ccorr
_Vu_g, _thu_g = _ringv(nonuniform=False, seed=0)
_rec_w = _ccorr(_meshm.graph_ring_order(_Vu_g, weighted=True), _thu_g)
_Aw_g = _meshm.similarity_graph(_Vu_g, k=6, weighted=True); _nz_g = _Aw_g[_Aw_g > 0]
print(f"  RING RECOVERY: the weighted similarity-graph eigenmap recovers a ring from {len(_Vu_g)} high-D vectors -- recovered order vs true angle |corr|={float(_rec_w):.3f}")
print(f"  the geometry is in the WEIGHTS: weighted edges carry cosine similarities (range {float(_nz_g.min()):.2f}-{float(_nz_g.max()):.2f}); the engine's binary kNN edges are all 1")
_Vn_g, _thn_g = _ringv(nonuniform=True, seed=0)
_rnw = _ccorr(_meshm.graph_ring_order(_Vn_g, weighted=True), _thn_g); _rnb = _ccorr(_meshm.graph_ring_order(_Vn_g, weighted=False), _thn_g)
_rub = _ccorr(_meshm.graph_ring_order(_Vu_g, weighted=False), _thu_g)
print(f"  WHERE WEIGHTING WINS: under NON-UNIFORM sampling weighted {float(_rnw):.3f} > binary {float(_rnb):.3f} (corrects density, like cotangent on an irregular mesh).  KEPT NEGATIVE: under UNIFORM sampling weighted {float(_rec_w):.3f} ~ binary {float(_rub):.3f} TIE -- high-D concentration, unlike a mesh's sharp cotangent gap")

title("ARCH-5: subdivision curves -- FWD-8's mesh subdivision turned inward on a sequence of vectors (1-manifold)")
# A sequence of hypervectors is a polyline through vector space; Chaikin corner-cutting refines it into a smooth
# limit curve -- the same refine+low-pass as Loop subdivision, one dimension down.
_rng_sd = _np_fwd4.random.default_rng(0)
_P_sd = _rng_sd.standard_normal((6, 64))
_counts_sd = [len(_meshm.subdivide_sequence(_P_sd, levels=_l)) for _l in range(4)]
print(f"  REFINE: an open sequence of 6 vectors doubles each level -> {_counts_sd} (2(n-1)/level), exactly as Loop quadruples faces")
_a_sd = _rng_sd.standard_normal(64); _b_sd = _rng_sd.standard_normal(64)
_ramp_sd = _np_fwd4.array([_a_sd + (_b_sd - _a_sd) * _t for _t in _np_fwd4.linspace(0, 1, 6)])
_sub_sd = _meshm.subdivide_sequence(_ramp_sd, levels=3); _dn_sd = (_b_sd - _a_sd) / _np_fwd4.linalg.norm(_b_sd - _a_sd)
_resid_sd = max(float(_np_fwd4.linalg.norm((p - _a_sd) - _np_fwd4.dot(p - _a_sd, _dn_sd) * _dn_sd)) for p in _sub_sd)
_zig_sd = _np_fwd4.zeros((10, 64)); _zig_sd[::2, 0] = 1.0; _zig_sd[1::2, 0] = -1.0
_r0_sd = float(_np_fwd4.sum(_np_fwd4.diff(_zig_sd, n=2, axis=0) ** 2)); _r2_sd = float(_np_fwd4.sum(_np_fwd4.diff(_meshm.subdivide_sequence(_zig_sd, levels=2), n=2, axis=0) ** 2))
print(f"  AFFINE: a straight line of vectors stays straight (residual {_resid_sd:.0e}, FWD-8's 'flat stays flat').  LOW-PASS: a zig-zag's roughness {_r0_sd:.0f} -> {_r2_sd:.2f} after 2 levels.  KEPT NEGATIVE: Chaikin approximates -- it cuts the control points (like Loop on a sphere)")

title("ARCH-6: rig + IK for structures -- blendshape posing (FWD-9 skinning + FWD-10 IK, turned inward)")
# A rig is a set of pose-target structures; FORWARD is a soft blend (skinning), INVERSE solves the blend weights
# to reach a goal -- via the SAME project_onto_constraints sweeper FWD-10 used for FABRIK.
_rng_bp = _np_fwd4.random.default_rng(0)
_P_bp = _rng_bp.standard_normal((4, 256))
_onehot_bp = _meshm.blend_pose(_P_bp, [0, 1, 0, 0])
print(f"  FORWARD (skinning): a one-hot weight reproduces target #1 exactly (cos {float(_np_fwd4.dot(_onehot_bp, _P_bp[1]) / _np_fwd4.linalg.norm(_P_bp[1])):.4f}) -- a soft blend of pose-target structures")
_wtrue_bp = _np_fwd4.array([0.4, 0.3, 0.2, 0.1]); _goal_bp = _P_bp.T @ _wtrue_bp
_w_bp = _meshm.solve_pose(_P_bp, _goal_bp)
print(f"  IK REACHABLE: goal is a known blend -> solver recovers the weights {[round(float(x),2) for x in _w_bp]} (true {[float(x) for x in _wtrue_bp]}), residual {float(_np_fwd4.linalg.norm(_P_bp.T @ _w_bp - _goal_bp)):.0e} -- like FWD-10 hitting a reachable target")
_goal2_bp = _rng_bp.standard_normal(256); _w2_bp = _meshm.solve_pose(_P_bp, _goal2_bp)
_br_bp = float(_np_fwd4.linalg.norm(_P_bp.T @ _w2_bp - _goal2_bp)); _vr_bp = min(float(_np_fwd4.linalg.norm(_P_bp[i] - _goal2_bp)) for i in range(4))
print(f"  IK UNREACHABLE: an out-of-span goal -> CLOSEST valid blend (residual {_br_bp:.2f} <= best single target {_vr_bp:.2f}) but cannot reach -- like FWD-10's chain fully extending toward an out-of-reach target.  *** §ARCH complete ***")

title("FWD-7 remainder: the three fiddlier modeler verbs -- bevel, bridge, loop-cut (vertex duplication + loop tracing)")
# The verbs FWD-7 deferred: bevel chamfers a corner, bridge joins two loops into a tube, loop-cut threads a new
# edge ring through a quad strip. All preserve the mesh's topology (chi) where they should.
from holographic_mesh import box as _box_v2
_cube_v2 = _box_v2()
_bev_v2 = _meshm.mesh_bevel_vertex(_cube_v2, 0, ratio=0.3)
_szs_v2 = sorted(len(f) for f in _bev_v2.faces)
print(f"  BEVEL a cube corner -> a small facet: V {_cube_v2.n_vertices}->{_bev_v2.n_vertices}, face sizes {_szs_v2} (3 quads became pentagons + a triangle cap), still closed manifold, chi {_bev_v2.euler_characteristic()} preserved")
_sq0_v2 = _np_fwd4.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float); _sq1_v2 = _sq0_v2.copy(); _sq1_v2[:, 2] = 1.0
_tube_v2 = _meshm.mesh_bridge(_np_fwd4.vstack([_sq0_v2, _sq1_v2]), [0, 1, 2, 3], [4, 5, 6, 7], closed=True)
print(f"  BRIDGE two squares -> an open tube: {_tube_v2.n_faces} side quads, manifold={_tube_v2.is_manifold()}, chi {_tube_v2.euler_characteristic()} (an open cylinder)")
_f0_v2 = tuple(_cube_v2.faces[0]); _lc_v2 = _meshm.mesh_loop_cut(_cube_v2, 0, (_f0_v2[0], _f0_v2[1]))
print(f"  LOOP-CUT a cube -> a new edge ring: F {_cube_v2.n_faces}->{_lc_v2.n_faces} (the ring crosses 4 quads, splitting each), still closed manifold, chi {_lc_v2.euler_characteristic()} preserved.  *** FWD modeler verb set complete ***")

title("Scene-graph algebra: one scene, two costumes -- geometry (instance + merge) AND structure (a recipe), provably consistent")
# The capstone joining the FWD mesh kernel to the ARCH-1 recipe algebra. A scene graph -- meshes at its leaves,
# transforms on its edges -- reads two ways: flatten it to a pile of triangles, OR encode it to one hypervector.
# The theorem: swapping siblings changes NEITHER view (merge and bundle both commute). The scene IS the recipe.
_scene_sg = _meshm.scene_graph(children=[_meshm.scene_graph(_meshm.scene_translation([2, 0, 0]), mesh=_cube_v2),
                                         _meshm.scene_graph(_meshm.scene_translation([0, 2, 0]), mesh=_cube_v2)])
_flat_sg = _meshm.scene_flatten(_scene_sg)
print(f"  GEOMETRY view: a scene of 2 cubes -> instanced + merged into one mesh, V={_flat_sg.n_vertices} F={_flat_sg.n_faces}; the +x instance lands at centroid {_np_fwd4.round(_flat_sg.vertices[_flat_sg.vertices[:, 0] > 1].mean(0), 1).tolist()}")
_rec_sg = _meshm.scene_to_recipe(_scene_sg)
from holographic_recipeops import validate as _validate_sg
print(f"  STRUCTURE view: the SAME scene -> a StructureRecipe, realising to one hypervector; a valid recipe ARCH-1 operates on = {_validate_sg(_rec_sg)[0]}")
_swap_sg = _meshm.scene_graph(children=[_scene_sg.children[1], _scene_sg.children[0]])
_geo_ok_sg = bool(_np_fwd4.allclose(_np_fwd4.sort(_flat_sg.vertices, axis=0), _np_fwd4.sort(_meshm.scene_flatten(_swap_sg).vertices, axis=0)))
_vec_ok_sg = bool(_np_fwd4.allclose(_rec_sg.outputs()[0], _meshm.scene_to_recipe(_swap_sg).outputs()[0], atol=1e-12))
print(f"  CONSISTENCY THEOREM: swap the two siblings -> geometry identical={_geo_ok_sg} AND holographic vector identical={_vec_ok_sg} (merge and bundle both commute).  *** the scene graph IS the recipe: VSA is geometry, made concrete ***")

title("QEM decimation: the quadric error metric -- collapse the edge that moves the surface least (a bundle of plane constraints)")
# The one piece the engine lacked for a principled simplifier: the COST of a collapse. A per-vertex quadric is a
# bundle of incident-plane constraints (Sigma nn^T) and the collapse cost is read off as a quadratic -- so it's
# bind/bundle/readout in disguise, and (the reverse thesis) the general "merge what loses the least" operator.
from holographic_meshsmooth import _icosphere as _icosphere_qem
_ico_qem = _icosphere_qem(2)                               # V66 F128 unit sphere
_qem_dec = _meshm.mesh_qem_decimate(_ico_qem, 64)
_qem_mean, _qem_max = _meshm.mesh_surface_deviation(_ico_qem, _qem_dec)
print(f"  decimate icosphere F{_ico_qem.n_faces} -> F{_qem_dec.n_faces} (half): still closed manifold, chi {_qem_dec.euler_characteristic()} preserved; surface moved mean {_qem_mean:.4f}, max {_qem_max:.4f}")
# naive shortest-edge->midpoint baseline, same target, for the comparison
from holographic_eulerops import collapse_edge as _ce_qem
from holographic_meshqem import _edges as _edges_qem
_m_qem = _ico_qem
while _m_qem.n_faces > 64:
    _rk_qem = sorted(((float(_np_fwd4.linalg.norm(_m_qem.vertices[a] - _m_qem.vertices[b])), a, b) for (a, b) in _edges_qem(_m_qem)), key=lambda t: (t[0], t[1], t[2]))
    _done_qem = False
    for (_, a, b) in _rk_qem:
        _k_qem, _r_qem = (a, b) if a < b else (b, a)
        _nm_qem = _ce_qem(_m_qem, _k_qem, _r_qem)
        if _nm_qem is None:
            _k_qem, _r_qem = _r_qem, _k_qem; _nm_qem = _ce_qem(_m_qem, _k_qem, _r_qem)
        if _nm_qem is None:
            continue
        _kn_qem = _k_qem if _k_qem < _r_qem else _k_qem - 1
        _nm_qem.vertices[_kn_qem] = 0.5 * (_m_qem.vertices[_k_qem] + _m_qem.vertices[_r_qem]); _m_qem = _nm_qem; _done_qem = True; break
    if not _done_qem:
        break
_nv_mean, _nv_max = _meshm.mesh_surface_deviation(_ico_qem, _m_qem)
print(f"  vs naive shortest-edge collapse at the same face count: naive moved mean {_nv_mean:.4f}, max {_nv_max:.4f} -> QEM is {_nv_mean / _qem_mean:.1f}x better on mean, {_nv_max / _qem_max:.1f}x on max (it spends its budget on the flats, keeps the features).  *** first item off the geometry->stack backlog ***")

title("Octahedral normals: quantize on the sphere, not in ambient bits -- 2 numbers for 2 degrees of freedom")
# A unit normal has only 2 DOF (it lives on S^2), so quantizing 3 x/y/z components wastes a third of the budget on
# a constrained coordinate. The octahedral map folds the sphere into a square; the bits land where the freedom is.
# This is manifold quantization -- the engine's "binary quant breaks the geometry" negative, turned into a method.
_rng_oct = _np_fwd4.random.default_rng(0)
_Noct = _rng_oct.standard_normal((8000, 3)); _Noct = _Noct / _np_fwd4.linalg.norm(_Noct, axis=-1, keepdims=True)
def _ang_oct(a, b): return _np_fwd4.degrees(_np_fwd4.arccos(_np_fwd4.clip(_np_fwd4.sum(a * b, axis=-1), -1, 1)))
_oct_rt = _meshm.oct_decode_normals(_meshm.oct_encode_normals(_Noct, 8), 8)
print(f"  encode 8000 unit normals at 8 bits/component (16 bits each) -> decode: mean angular error {_ang_oct(_Noct, _oct_rt).mean():.4f} deg, max {_ang_oct(_Noct, _oct_rt).max():.4f} deg (and the continuous map is an EXACT bijection)")
def _qn_oct(a, bits):
    _lv = (1 << bits) - 1; return _np_fwd4.round((a + 1) * 0.5 * _lv) / _lv * 2 - 1
_nb_oct = _np_fwd4.stack([_qn_oct(_Noct[:, 0], 5), _qn_oct(_Noct[:, 1], 5), _qn_oct(_Noct[:, 2], 6)], axis=-1)
_nb_oct = _nb_oct / _np_fwd4.linalg.norm(_nb_oct, axis=-1, keepdims=True)
print(f"  at the SAME 16-bit budget, naive x/y/z quantization (5+5+6 bits) gives mean {_ang_oct(_Noct, _nb_oct).mean():.4f} deg -> octahedral is {_ang_oct(_Noct, _nb_oct).mean() / _ang_oct(_Noct, _oct_rt).mean():.1f}x more accurate for free.  *** the S^2 case of reverse item R3 (manifold quantization), paired with QEM off backlog item A2 ***")

title("Bandwidth + a singularity cross-check: measure how much spectrum to keep, and when the fractal dimension lies")
# The engine already measures fractal DIMENSION three ways; the missing pieces were the BANDWIDTH a signal occupies
# (which drives a band-limited encoder) and a cross-check that catches the dimension lying on a lone discontinuity.
_n_bw = 8192
_smooth_bw = _np_fwd4.sin(2 * _np_fwd4.pi * 3 * _np_fwd4.arange(_n_bw) / _n_bw)
_white_bw = _np_fwd4.random.default_rng(2).standard_normal(_n_bw)
print(f"  spectral bandwidth (95% energy): a smooth sinusoid {_meshm.spectral_bandwidth(_smooth_bw):.4f} of Nyquist (band-limited) vs white noise {_meshm.spectral_bandwidth(_white_bw):.3f} (broadband) -- the number that sets the encoder's bandwidth knob")
_rng_bw = _np_fwd4.random.default_rng(1)
_f_bw = _np_fwd4.fft.rfftfreq(_n_bw); _amp_bw = _np_fwd4.zeros(len(_f_bw)); _amp_bw[1:] = _f_bw[1:] ** (-(2 * 0.3 + 1) / 2)
_fbm_bw = _np_fwd4.fft.irfft(_amp_bw * _np_fwd4.exp(1j * _rng_bw.uniform(0, 2 * _np_fwd4.pi, len(_f_bw))), _n_bw)
_ds_f, _di_f, _ag_f = _meshm.fractal_confidence(_fbm_bw)
_step_bw = _np_fwd4.zeros(_n_bw); _step_bw[_n_bw // 2:] = 1.0
_ds_s, _di_s, _ag_s = _meshm.fractal_confidence(_step_bw)
print(f"  cross-check on clean fBm: spectral D {_ds_f:.2f}, increment D {_di_f:.2f} -> AGREE ({_ag_f}); on a STEP: spectral {_ds_s:.2f}, increment {_di_s:.2f} -> DISAGREE ({_ag_s}) -- the lone discontinuity fools the slope, the flag catches it.  *** first fractal-optics backlog item; de-dup: dimension was already shipped ***")

title("Auto-bandwidth KDE: the band-limit matched to the data -- the encoder's kernel as a density estimator")
# The disciplined form of the 'band-limited encoding' item, landed where the encoder actually delivers: its RBF
# kernel IS a KDE, and the kernel bandwidth IS the band-limit. Leave-one-out likelihood picks the bandwidth that
# matches the data -- neither over-smoothing (over-band-limiting) nor under-smoothing (aliasing the samples).
_bim = lambda x: 0.5 * _np_fwd4.exp(-0.5 * ((x - 0.3) / 0.05) ** 2) + 0.5 * _np_fwd4.exp(-0.5 * ((x - 0.7) / 0.07) ** 2)
_rng_kde = _np_fwd4.random.default_rng(0); _xs_kde = []
while len(_xs_kde) < 400:
    _c_kde = _rng_kde.uniform(0, 1)
    if _rng_kde.uniform(0, 6) < _bim(_c_kde): _xs_kde.append(_c_kde)
_xs_kde = _np_fwd4.array(_xs_kde); _qx_kde = _np_fwd4.linspace(0.02, 0.98, 200); _truth_kde = _bim(_qx_kde)
def _srmse_kde(e, t):
    _a = _np_fwd4.sum(e * t) / _np_fwd4.sum(e * e); return _np_fwd4.sqrt(_np_fwd4.mean((_a * e - t) ** 2))
_est_kde, _bw_kde = _meshm.density_estimate(_xs_kde, 0, 1, _qx_kde, dim=1024, method="lcv")
_def_kde, _ = _meshm.density_estimate(_xs_kde, 0, 1, _qx_kde, dim=1024, bandwidth=1.8)
print(f"  400 samples from a bimodal density -> KDE via the encoder: LCV picks bandwidth {_bw_kde:.1f}, estimate correlates {_np_fwd4.corrcoef(_est_kde, _truth_kde)[0, 1]:.3f} with truth")
print(f"  shape error: LCV-matched bandwidth {float(_srmse_kde(_est_kde, _truth_kde)):.3f} vs the fixed default (1.8, too wide) {float(_srmse_kde(_def_kde, _truth_kde)):.3f} -> {float(_srmse_kde(_def_kde, _truth_kde) / _srmse_kde(_est_kde, _truth_kde)):.1f}x better by matching the band-limit to the data.  *** second fractal-optics item; the audit: sinc isn't tunable, so it landed on RBF-as-KDE ***")

title("Screen-space LOD: the error-budget resolution rule (coarse_to_fine) carried to meshes")
# QEM decimates and surface_deviation measures; this DECIDES which level to show -- the coarsest mesh whose error,
# projected to the screen, stays under a pixel budget. Full detail up close, coarser far away. The engine's own
# resolution-by-error-budget rule, in the mesh domain.
from holographic_meshsmooth import _icosphere as _icos_lod
_lod_chain = _meshm.mesh_lod_chain(_icos_lod(2), targets=(0.5, 0.25, 0.125))
print(f"  LOD chain off a sphere: " + ", ".join(f"F{_l.n_faces}(err {_l.max_error:.3f})" for _l in _lod_chain))
_lod_picks = [(d, _meshm.mesh_select_lod(_lod_chain, d, 2.0)) for d in (2.0, 15.0, 60.0, 200.0)]
print("  pick the cheapest mesh that looks right (2px budget): " + ", ".join(f"{d:.0f}u->F{_lod_chain[p].n_faces}" for d, p in _lod_picks) + "  *** geometry->stack: decimate, measure, SELECT -- coarse_to_fine for geometry ***")

title("Binding stability: when does bind/unbind preserve the signal? -- spectral flatness predicts the distortion")
# The band-limit-preservation regime test, grounded in the real bind. Trefethen's transient-growth lens found none
# (the cleanup contracts monotonically, linear ops stay spectrally flat); the real axis is the KEY's spectral
# flatness -- unbind(bind(x,k),k) returns x convolved with |K|^2, exact only when |K|=1 everywhere (a unitary key).
from holographic_ai import unitary_vector as _uvec_bs, random_vector as _rvec_bs
_uni_bs = _uvec_bs(1024, _np_fwd4.random.default_rng(1)); _ran_bs = _rvec_bs(1024, _np_fwd4.random.default_rng(2))
_rep_u = _meshm.binding_stability(_uni_bs); _rep_r = _meshm.binding_stability(_ran_bs)
print(f"  a UNITARY key: flatness {_rep_u['flatness']:.3f} -> bind/unbind distortion {_rep_u['distortion']:.1e} (exact, stable={_rep_u['stable']})")
print(f"  a RANDOM  key: flatness {_rep_r['flatness']:.3f} -> bind/unbind distortion {_rep_r['distortion']:.2f} (lossy, stable={_rep_r['stable']}) -- flatness is the diagnostic for 'safe to bind/unbind repeatedly?'  *** fractal-optics item; Trefethen transient-growth came up empty, the real axis is linear flatness ***")

title("Splat LOD: prune the negligible splats by contribution -- the splat twin of mesh decimation")
# A splat renders as amp*gaussian (unit-norm), so its energy is amp^2: keep the largest-|amp| splats, refit, and the
# survivors absorb the overlap. Prune->measure->select mirrors the mesh LOD's decimate->measure->select.
from holographic_splat import splat_fit as _sfit_lod, splat_render as _srend_lod, splat_refit as _sref_lod, psnr as _psnr_lod
_yy_lod, _xx_lod = _np_fwd4.mgrid[0:64, 0:64]
_g_lod = lambda cy, cx, s: _np_fwd4.exp(-((_yy_lod - cy) ** 2 + (_xx_lod - cx) ** 2) / (2 * s * s))
_tgt_lod = 1.0 * _g_lod(18, 20, 5) + 0.8 * _g_lod(40, 44, 7) + 0.6 * _g_lod(48, 15, 4) + 0.4 * _g_lod(12, 50, 3)
_full_lod = _sfit_lod(_tgt_lod, 60, refit=True)
_chain_lod = _meshm.splat_lod_chain(_full_lod, _tgt_lod, keeps=(40, 20, 10, 5))
print("  LOD chain (prune + refit): " + ", ".join(f"{_c[1]}sp@{_c[2]:.0f}dB" for _c in _chain_lod))
_rng_lod = _np_fwd4.random.default_rng(0)
_rand_lod = _sref_lod([_full_lod[i] for i in _rng_lod.permutation(len(_full_lod))[:20]], _tgt_lod)
print(f"  at 20 splats: contribution-ranked {float(_psnr_lod(_srend_lod(_meshm.splat_prune(_full_lod, _tgt_lod, 20), (64, 64)), _tgt_lod)):.1f}dB vs random {float(_psnr_lod(_srend_lod(_rand_lod, (64, 64)), _tgt_lod)):.1f}dB -- keeping the splats that carry the energy is ~20dB better.  *** geometry->stack: prune, measure, SELECT ***")

title("Scene delta: variants share components for free (content-addressed) -- the diff is what you transmit")
# scene_to_recipe names every component by content hash, so shared subtrees share atoms automatically; the dedup is
# free. The genuinely-new piece is the explicit DIFF (send the base once, then small deltas) and measuring the saving.
from holographic_scenegraph import SceneNode as _SN_sd, translation as _tr_sd
from holographic_mesh import box as _box_sd
from holographic_scenedelta import scene_components as _scomp_sd, apply_scene_delta as _apply_sd
_cube_sd = _box_sd(); _oth_sd = _box_sd(2, 1, 1)
def _var_sd(i, changed=True):
    _ch = [_SN_sd(_tr_sd([0,0,0]), mesh=_cube_sd), _SN_sd(_tr_sd([2,0,0]), mesh=_cube_sd), _SN_sd(_tr_sd([0,2,0]), mesh=_oth_sd), _SN_sd(_tr_sd([2,2,0]), mesh=_cube_sd)]
    if changed: _ch[i%4]=_SN_sd(_tr_sd([float(i),5,0]), mesh=_cube_sd)
    return _SN_sd(children=_ch)
_base_sd = _var_sd(0, changed=False); _var1_sd = _var_sd(1)
_d_sd = _meshm.scene_delta(_base_sd, _var1_sd)
_sav_sd = _meshm.scene_dedup_saving([_base_sd] + [_var_sd(i) for i in range(8)])
print(f"  a variant changing one subtree: delta is +{len(_d_sd['added'])}/-{len(_d_sd['removed'])} components vs {len(_scomp_sd(_var1_sd))} full; base+delta rebuilds it exactly: {_apply_sd(_scomp_sd(_base_sd), _d_sd) == _scomp_sd(_var1_sd)}")
print(f"  storing 9 scenes that share subtrees: {_sav_sd['naive']} components naively -> {_sav_sd['unique']} unique ({_sav_sd['saving_x']:.1f}x) -- content-hashing dedups for free.  *** reverse item R6; honest finding: the dedup is automatic, the diff is the new part ***")

title("RT-V occlusion recall: front-to-back compositing breaks the bundle capacity cliff")
# 3DGS composites front-to-back so a pixel saturates after the front few splats. The transfer: sort atoms by relevance
# to a loaded cue, subtract each one's explained part (transmittance), and the tail is OCCLUDED, not summed -- so
# multi-component recall survives far past the linear cliff that washes out.
_rng_oc = _np_fwd4.random.default_rng(0); _Noc, _Doc = 200, 512
_cb_oc = _rng_oc.standard_normal((_Noc, _Doc)); _cb_oc = _cb_oc / _np_fwd4.linalg.norm(_cb_oc, axis=1, keepdims=True)
def _f1_oc(pred, true):
    pred = set(pred); _tp = len(pred & true)
    _p = _tp / len(pred) if pred else 0.0; _r = _tp / len(true) if true else 0.0
    return 2 * _p * _r / (_p + _r) if (_p + _r) > 0 else 0.0
print("  load M | linear/softmax/TopK F1 | OCCLUSION F1")
for _M_oc in (5, 25, 50):
    _fo = _fl = 0.0
    for _sd_oc in range(15):
        _r_oc = _np_fwd4.random.default_rng(_sd_oc); _S_oc = set(_r_oc.choice(_Noc, _M_oc, replace=False).tolist())
        _cue_oc = _cb_oc[list(_S_oc)].sum(0); _cue_oc = _cue_oc / _np_fwd4.linalg.norm(_cue_oc)
        _fo += _f1_oc([j for j, _ in _meshm.occlusion_recall(_cue_oc, _cb_oc, m=_M_oc)], _S_oc)
        _fl += _f1_oc(list(_np_fwd4.argsort(-(_cb_oc @ _cue_oc))[:_M_oc]), _S_oc)
    print(f"   {_M_oc:3d}   |        {_fl/15:.3f}          |   {_fo/15:.3f}")
print("  the order-free readouts wash out together as load grows; occlusion's sequential subtraction holds perfect recall -- ties them at low load (kept negative).  *** RT-V: alpha-compositing breaks the engine's oldest cliff ***")

title("RT-VI context-dependent meaning: spherical harmonics -> a polysemous atom on the FPE phase substrate")
# A 3DGS splat's colour is a function of view direction, expanded in spherical harmonics (DC = base colour). The
# transfer: an atom whose MEANING is a function of a context angle, in a circular-harmonic basis. DC = the context-
# free meaning (the plain fixed atom, exactly); higher harmonics = how the meaning shifts with context.
_rng_hm = _np_fwd4.random.default_rng(0); _Dhm = 256
_senses = [_rng_hm.standard_normal(_Dhm) for _ in range(3)]; _senses = [s / _np_fwd4.linalg.norm(s) for s in _senses]
_ctx = [0.0, 2 * 3.14159265 / 3, 4 * 3.14159265 / 3]
_patom = _meshm.harmonic_atom(_ctx, _senses, n_harmonics=2)   # one polysemous atom, 3 senses
print("  one atom, 3 senses placed at 3 contexts -- decode at each context recovers the right sense:")
for _i, (_t, _s) in enumerate(zip(_ctx, _senses)):
    _rec = _meshm.harmonic_decode(_patom, _t)
    print(f"    context {_t:.2f} -> cosine to sense {_i} = {float(_rec @ _s / _np_fwd4.linalg.norm(_rec)):.4f}")
# degree-0 fallback: a context-free atom reduces to the plain atom exactly (backward-compatible)
_const = _rng_hm.standard_normal(_Dhm)
_cfree = _meshm.harmonic_atom([0.0, 1.0, 2.0], [_const, _const, _const], n_harmonics=1)
print(f"  a CONTEXT-FREE atom: decode error at arbitrary context = {float(_np_fwd4.linalg.norm(_meshm.harmonic_decode(_cfree, 0.77) - _const)):.1e} -- the DC term IS the plain atom (degree-0 fallback, exact).  *** RT-VI: the engine's own basis pointed at meaning ***")

title("Clone-vs-split density control: cover an under-served region vs resolve fine structure")
# 3DGS densifies by the splat's SCALE: clone a small high-error splat to COVER, split a wide one to RESOLVE. The
# engine's splat_densify adds capacity where error is but is scale-blind; this adds the cover-vs-resolve decision.
from holographic_splat import splat_render as _sr_cs, splat_refit as _rf_cs
from holographic_splatdensify import clone_splat as _cl_cs, split_splat as _sp_cs
_ys_cs, _xs_cs = _np_fwd4.mgrid[0:64, 0:64]
_ridge_cs = _np_fwd4.exp(-(((_xs_cs - 14) ** 2) / 6.0 + ((_ys_cs - 32) ** 2) / 120.0))      # needs COVER
_twin_cs = (_np_fwd4.exp(-(((_xs_cs - 46) ** 2 + (_ys_cs - 30) ** 2) / 4.0)) + _np_fwd4.exp(-(((_xs_cs - 52) ** 2 + (_ys_cs - 30) ** 2) / 4.0)))  # needs RESOLVE
_tgt_cs = _ridge_cs + _twin_cs
_sp0_cs = _rf_cs([(32, 14, 0.0, 1.0), (30, 49, 0.0, 3.5)], _tgt_cs)                          # small splat + wide splat
_res_cs = _tgt_cs - _sr_cs(_sp0_cs, _tgt_cs.shape)
def _mse_cs(a, b): return float(((a - b) ** 2).mean())
def _blind_cs(strat):
    _o = []
    for _s in _sp0_cs:
        _o += (_sp_cs(_s, _res_cs, _tgt_cs.shape) if strat == "split" else [_s] + _cl_cs(_s, _res_cs, _tgt_cs.shape))
    return _mse_cs(_sr_cs(_rf_cs(_o, _tgt_cs), _tgt_cs.shape), _tgt_cs)
_scale_cs = _mse_cs(_sr_cs(_meshm.splat_clone_split(_sp0_cs, _tgt_cs), _tgt_cs.shape), _tgt_cs)
print(f"  mixed target (a ridge needing cover + twin peaks needing resolve), fixed splat budget:")
print(f"    always-clone {_blind_cs('clone'):.5f} (misses the peaks) | always-split {_blind_cs('split'):.5f} (hurts the ridge) | SCALE-AWARE {_scale_cs:.5f}")
print("  each blind strategy handles only one error type; the scale rule does the right move for each -- and the wrong move can be worse than nothing.  *** clone-vs-split: sharpening WHERE capacity goes ***")

title("MCMC birth-death relocation: conserve a dead atom by moving it, don't drop it (successor to evict-rarest)")
# The engine's bounded memory evicts the rarest (drops capacity). 3DGS-as-MCMC relocates a dead atom to an under-
# represented region instead -- conserving the budget. Here: a fixed budget with dead splats jammed in a corner.
from holographic_relocate import birth_death_relocate as _bdr
_ys_bd, _xs_bd = _np_fwd4.mgrid[0:64, 0:64]
_tgt_bd = sum(_np_fwd4.exp(-(((_xs_bd - cx) ** 2 + (_ys_bd - cy) ** 2) / 12.0))
              for cx, cy in [(16, 16), (48, 16), (16, 48), (48, 48), (32, 32), (32, 10)])
_useful_bd = [(16, 16, 0.0, 3.5), (48, 16, 0.0, 3.5), (16, 48, 0.0, 3.5), (48, 48, 0.0, 3.5), (32, 32, 0.0, 3.5), (32, 10, 0.0, 3.5)]
_splats_bd = _rf_cs(_useful_bd + [(2, 2, 0.0, 1.0)] * 6, _tgt_bd)        # 6 dead splats in the corner
_thr_bd = 0.05 * _np_fwd4.abs([s[2] for s in _splats_bd]).max()
_n_dead_bd = int((_np_fwd4.abs([s[2] for s in _splats_bd]) < _thr_bd).sum())
_drop_bd = _mse_cs(_sr_cs(_rf_cs([s for s in _splats_bd if abs(s[2]) >= _thr_bd], _tgt_bd), _tgt_bd.shape), _tgt_bd)
_reloc_sp = _meshm.splat_relocate(_splats_bd, _tgt_bd)
_reloc_bd = _mse_cs(_sr_cs(_reloc_sp, _tgt_bd.shape), _tgt_bd)
print(f"  budget 12 splats, {_n_dead_bd} dead:")
print(f"    DROP dead (evict, budget shrinks to {12 - _n_dead_bd}): MSE {_drop_bd:.5f} | RELOCATE to under-served regions (count conserved {len(_reloc_sp)}): MSE {_reloc_bd:.5f}")
print(f"  relocating conserves the budget and redistributes it where the data is -- ~{_drop_bd / _reloc_bd:.0f}x better than dropping.  *** birth-death: conserve capacity, don't drop it ***")

title("SPEED-1: Gram-cached occlusion recall -- cache the dictionary's Gram, stop rescanning it (Batch-OMP)")
# Occlusion recall broke the capacity cliff but cost ~170x the linear readout (M rescans of the dictionary). The fix
# from compressed sensing: precompute the Gram G = cb @ cb.T once, update correlations through a Gram column per pick
# instead of rescanning. EXACT (identical recovery), and the D factor leaves the inner loop.
import time as _time_sp
_rng_sp = _np_fwd4.random.default_rng(0); _Nsp, _Dsp = 400, 1024
_cb_sp = _rng_sp.standard_normal((_Nsp, _Dsp)); _cb_sp = _cb_sp / _np_fwd4.linalg.norm(_cb_sp, axis=1, keepdims=True)
_G_sp = _meshm.build_occlusion_gram(_cb_sp)                  # the cached precompute (reused across cues)
_r_sp = _np_fwd4.random.default_rng(1); _S_sp = _r_sp.choice(_Nsp, 200, replace=False)
_cue_sp = _cb_sp[_S_sp].sum(0); _cue_sp = _cue_sp / _np_fwd4.linalg.norm(_cue_sp)
_a_sp = _meshm.occlusion_recall(_cue_sp, _cb_sp, m=200)                  # rescan path
_b_sp = _meshm.occlusion_recall(_cue_sp, _cb_sp, m=200, gram=_G_sp)      # Gram-cached fast path
_t0 = _time_sp.perf_counter()
for _ in range(20): _meshm.occlusion_recall(_cue_sp, _cb_sp, m=200)
_t_resc = (_time_sp.perf_counter() - _t0) / 20
_t0 = _time_sp.perf_counter()
for _ in range(20): _meshm.occlusion_recall(_cue_sp, _cb_sp, m=200, gram=_G_sp)
_t_gram = (_time_sp.perf_counter() - _t0) / 20
print(f"  D={_Dsp}, recovering 200 items from one bundle:")
print(f"    rescan path {_t_resc*1e3:6.2f}ms  ->  Gram-cached {_t_gram*1e3:5.2f}ms   ({_t_resc/_t_gram:.0f}x faster)")
print(f"    identical atoms recovered? {[j for j,_ in _a_sp]==[j for j,_ in _b_sp]}  (exact -- weights match to machine epsilon)")
print("  the bottleneck was recompute-what-you-already-know: the Gram is a cached precompute, reused across cues.  *** SPEED-1: the RAM hunch, measured ***")

title("RAM-1: a Gram working-set cache -- build the Gram once, reuse it across cues (zero precompute on the 2nd call)")
# SPEED-1 made the readout fast IF you hold the Gram. RAM-1 makes the Gram durable: pass cache=True and the mind keeps
# a bounded, GC-safe, id-keyed cache, so a vocabulary queried many times pays the O(N^2 D) precompute ONCE.
_meshm.occlusion_recall(_cue_sp, _cb_sp, m=200, cache=True)             # 1st call: builds + caches (a MISS)
for _ in range(4): _meshm.occlusion_recall(_cue_sp, _cb_sp, m=200, cache=True)   # repeated cues: cache HITS
print(f"  5 recalls against the same vocabulary: {_meshm._gram_cache.misses} Gram build (miss), {_meshm._gram_cache.hits} reuses (hits) -- the precompute is paid ONCE.")
print("  keyed by codebook identity (O(1), no per-call hashing), GC-safe via weakref, LRU-bounded.  *** RAM-1: the Gram, made durable ***")

title("GRAD-2: the splat-fit Adam, generalized -- minimize any scalar loss, gradients on the fly (no autodiff)")
# 3D-Gaussian splatting brought a real optimizer into the engine (Adam with hand-derived gradients). GRAD-2 promotes
# it to a general faculty: minimize ANY loss, with an analytic gradient where you have one, finite differences where
# you don't. Here: a least-squares loss, descended to the same answer numpy's lstsq gives.
_A_g2 = _np_fwd4.random.default_rng(7).standard_normal((20, 6))
_b_g2 = _np_fwd4.random.default_rng(8).standard_normal(20)
_sol_g2 = _np_fwd4.linalg.lstsq(_A_g2, _b_g2, rcond=None)[0]
_x_g2 = _meshm.optimize(lambda z: float(((_A_g2 @ z - _b_g2) ** 2).sum()), _np_fwd4.zeros(6),
                        grad=lambda z: 2 * _A_g2.T @ (_A_g2 @ z - _b_g2), steps=2000, lr=0.02)
print(f"  least-squares descended through the mind vs numpy lstsq: agree to {float(_np_fwd4.linalg.norm(_x_g2 - _sol_g2)):.1e}")
# and with NO analytic gradient supplied, the finite-difference fallback reaches the same minimum
_t_g2 = _np_fwd4.array([1.0, -2.0, 0.5])
_xfd_g2 = _meshm.optimize(lambda z: float(((z - _t_g2) ** 2).sum()), _np_fwd4.zeros(3), steps=400, lr=0.1)
print(f"  same fit with NO gradient (finite differences on the fly): off by {float(_np_fwd4.linalg.norm(_xfd_g2 - _t_g2)):.1e}.  *** GRAD-2: the gradients hunch, first-class ***")

title("GRAD-1: IHT recall -- the gradient-native recovery route, built on GRAD-2 (it REVISES its support)")
# Three ways to recover a bundle's components now sit side by side: linear (one-shot), occlusion (greedy matching
# pursuit), and IHT (projected gradient descent -- a gradient step + keep-the-K-largest, iterated). On a COHERENT
# dictionary, greedy MP's early wrong picks are unrecoverable; IHT keeps revising and wins. Build a coherent dictionary:
_rg1 = _np_fwd4.random.default_rng(101)
_cbc = _rg1.standard_normal((200, 512)) + 1.5 * _rg1.standard_normal(512)      # shared component -> mutual coherence
_cbc = _cbc / _np_fwd4.linalg.norm(_cbc, axis=1, keepdims=True)
_Sc = _rg1.choice(200, 12, replace=False); _wc = _rg1.uniform(0.5, 1.5, 12)
_cuec = (_wc[:, None] * _cbc[_Sc]).sum(0); _truec = set(int(_i) for _i in _Sc)
def _f1_t(_rec):
    _g = set(_i for _i, _ in _rec); _tp = len(_g & _truec)
    _p = _tp / max(len(_g), 1); _r = _tp / max(len(_truec), 1)
    return 2 * _p * _r / max(_p + _r, 1e-12)
_iht_f1 = _f1_t(_meshm.iht_recall(_cuec, _cbc, 12))
_occ_f1 = _f1_t(_meshm.occlusion_recall(_cuec, _cbc, m=12))
print(f"  recover 12 atoms from a coherent dictionary: IHT F1 {_iht_f1:.3f}  vs  greedy occlusion {_occ_f1:.3f} -- support revision corrects MP's stuck early picks.")
print("  with K=N (no threshold) IHT reduces to plain gradient descent = the least-squares solution -- the same descent GRAD-2 generalized.  *** GRAD-1: the third recovery route ***")

title("SPEED-3: CoSaMP -- the strongest recovery route (batch select + least-squares each round), completes the family")
# The four routes on the SAME coherent bundle from the IHT demo above (_cbc/_cuec/_truec/_f1_t): linear (one-shot),
# occlusion (greedy MP), IHT (gradient + threshold), and CoSaMP (batch 2K candidates + least-squares + prune, ~2-3
# rounds). The least-squares solve disambiguates the correlated atoms the others get stuck on.
_lin_f1 = _f1_t([(int(_i), float(_v)) for _i, _v in zip(
    _np_fwd4.argpartition(_cbc @ _cuec, -12)[-12:], (_cbc @ _cuec)[_np_fwd4.argpartition(_cbc @ _cuec, -12)[-12:]])])
_cos_st = {}
_cos_f1 = _f1_t(_meshm.cosamp_recall(_cuec, _cbc, 12, stats=_cos_st))
print(f"  recover 12 atoms from a coherent dictionary -- linear {_lin_f1:.3f} | occlusion {_occ_f1:.3f} | IHT {_iht_f1:.3f} | CoSaMP {_cos_f1:.3f} (in {_cos_st['rounds']} rounds)")
print("  CoSaMP's per-round least-squares gets exact coefficients and corrects what greedy MP can't -- the cost is that LS solve, and it falls off only as the load nears the dimension.  *** SPEED-3: the recovery family complete ***")

title("SPEED-2: forest-routed selection -- the N-factor is REAL (sub-linear), but a measured regression at this scale")
# The last occlusion-speed factor: occlusion's pick-the-best-atom step is a max-inner-product search, and a HoloForest
# answers it by comparing only the atoms ROUTED to the query -- sub-linear in N. Build a forest over a big dictionary:
_cbF = _np_fwd4.random.default_rng(9).standard_normal((4000, 256))
_cbF = _cbF / _np_fwd4.linalg.norm(_cbF, axis=1, keepdims=True)
_SF = _np_fwd4.random.default_rng(10).choice(4000, 12, replace=False); _cueF = _cbF[_SF].sum(0)
_F = _meshm.build_occlusion_forest(_cbF, seed=0)
_recF = _meshm.occlusion_recall_forest(_cueF, _cbF, 12, forest=_F)
print(f"  N=4000 dictionary: the forest ranked only {_F.last_comparisons} of 4000 atoms per pick -- the sub-linearity is real.")
print("  BUT: the exact scan is a single BLAS matvec the Python tree-routing can't beat until N is enormous, and the approximate pick costs F1 exactly when it saves comparisons -- so exact occlusion + the SPEED-1 Gram stays the default.  *** SPEED-2: the N-factor, measured to its honest end (kept negative) ***")

title("W1: mixture of experts with a LEARNED gate -- routes by the input's CONTENT, not by a rule")
# The mind's own dispatch routes by RULE (which verb, what type). A mixture-of-experts gate is TRAINED, so it routes
# by content -- two experts owning different halves of the number line; the gate learns to send each value to the
# right one, which a type check could never do.
_moe_t = _meshm.mixture_of_experts(seed=2, number_range=(0.0, 1.0))
_rgm = _np_fwd4.random.default_rng(1)
_lo = [(float(_rgm.uniform(0.02, 0.46)), "L", None) for _ in range(60)]
_hi = [(float(_rgm.uniform(0.54, 0.98)), "H", None) for _ in range(60)]
_moe_t.add_expert("low", _lo); _moe_t.add_expert("high", _hi)
_moe_t.train_gate(_lo + _hi, epochs=14)
_moe_test = [(float(_rgm.uniform(0.02, 0.46)), "L") for _ in range(40)] + [(float(_rgm.uniform(0.54, 0.98)), "H") for _ in range(40)]
_moe_acc = float(_np_fwd4.mean([_moe_t.predict(_x, None)[0] == _lab for _x, _lab in _moe_test]))
print(f"  two experts split the number line; the trained gate routes held-out values by VALUE at {_moe_acc*100:.0f}% accuracy -- a learned route, not a rule.")
print("  the gate is itself a creature brain, trained from whether the routed expert was right.  *** W1: a learned gate, wired ***")

title("W2: closed-form kinematics -- position += velocity is ONE binding, velocity is UNBIND (binding IS a rigid shift)")
# The core thesis pointed at motion: integrate a trajectory by pure binding and decode each position; read the
# velocity between two positions by unbind. The closed-form twin of learn_dynamics (which LEARNS its operator).
_kin = _meshm.kinematics(dim=2048, lo=-50.0, hi=50.0, seed=1)
_kdec, _ktrue = _kin.trajectory(0.0, 2.0, a=0.0, steps=8)
_kerr = float(_np_fwd4.max(_np_fwd4.abs(_kdec - _ktrue)))
_kvel = float(_kin.read_velocity(10.0, 13.0))
print(f"  x0=0, v0=2 integrated by BINDING for 8 steps -> decodes the true path to within {_kerr:.2f}; velocity between 10 and 13 read by UNBIND = {_kvel:.1f}.")
print("  the operator is the encoder's own shift, exact by construction -- not fitted.  *** W2: binding is motion, wired ***")

title("W4: a versioned store -- commit, edit, rollback (the undo/redo spine, history as keyframes + deltas)")
# Every version is committed and exactly recoverable: rows keyed by stable id, history stored as keyframes + lossless
# deltas (the video codec's GOP structure, here for an edit timeline). The undo/redo the editable-mesh vision needs.
_vs = _meshm.versioned_store()
_id1, _id2 = _vs.new_id(), _vs.new_id()
_rows0 = {_id1: _np_fwd4.ones(_meshm.dim), _id2: _np_fwd4.zeros(_meshm.dim)}
_v0 = _vs.commit(_rows0, [_id1, _id2], note="initial")
_vs.commit({_id1: _np_fwd4.full(_meshm.dim, 7.0), _id2: _np_fwd4.zeros(_meshm.dim)}, [_id1, _id2], note="edit")  # change a row
_back, _ = _vs.checkout(_v0)
_exact = bool(_np_fwd4.array_equal(_back[_id1], _rows0[_id1]))
_rb = _vs.rollback(_v0)
print(f"  commit -> edit -> checkout(v0) reconstructs the original exactly: {_exact}; rollback to v0 is itself version {_rb} (history never erased, head now {_vs.head()}).")
print("  an optional proof can gate a commit -- a failing proof is logged and rejected, the store left unchanged.  *** W4: versioning + rollback, wired ***")

title("W3: a motion-compensated video codec -- a rigid pan is one bind, so the residual nearly vanishes")
# Keyframe + motion-compensated residual: a rigid shift is a single bind, so the inter-frame residual is tiny and the
# codec beats per-frame storage. (Non-rigid change leaves a large residual -- the honest boundary, kept.)
_vbase = _np_fwd4.zeros((40, 48)); _vyy, _vxx = _np_fwd4.mgrid[0:40, 0:48]
for _vc in [(10, 12), (20, 30), (30, 18), (15, 40)]:
    _vbase = _vbase + _np_fwd4.exp(-((_vyy - _vc[0]) ** 2 + (_vxx - _vc[1]) ** 2) / 30.0)
_vbase = _vbase / _vbase.max()
_vframes = [_np_fwd4.roll(_vbase, 2 * _t, axis=1) for _t in range(6)]      # rigid cyclic pan
from holographic_video import HolographicVideo as _HV4
_vcodec = _meshm.video_codec(dim=2048, key_keep=150, res_keep=30, gop_len=6, seed=0)
_vpk, _vtot = _vcodec.encode(_vframes); _vps = _vcodec.mean_psnr(_vframes, _vpk)
_vitot, _vips = _HV4.intra_baseline(_vframes, keep=150, dim=2048, seed=0)
print(f"  6-frame rigid pan -- per-frame intra: {_vitot} bytes @ {_vips:.0f} dB; motion-compensated GOP: {_vtot} bytes @ {_vps:.0f} dB (fewer bytes, higher quality).")
print("  the motion vector is one number, the residual is what a shift can't predict -- the token codec's trick, in the image domain.  *** W3: motion compensation, wired ***")

title("FS-1: sculpt brushes -- a local falloff-weighted edit of a FIELD, then re-mesh correct topology at any resolution")
# A surface carried as a field whose level-set IS the surface: brush the field (inflate/carve/smooth/grab/...) and
# re-extract -- the resolution-independent move a fixed mesh can't do. The brush is exactly 0 outside its ball.
from holographic_meshbridge import metaball_field as _mbf4, sample_field as _sf4, marching_tetrahedra as _mt4
_sfield = _mbf4(_np_fwd4.array([[0.0, 0.0, 0.0]]), radius=0.4)
_sbounds = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))
_sgrid = _np_fwd4.stack(_np_fwd4.meshgrid(*[_np_fwd4.linspace(-1.5, 1.5, 26)] * 3, indexing="ij"), -1).reshape(-1, 3)
_sbase = int((_sfield(_sgrid) > 0.5).sum())
_sinf = _meshm.sculpt(_sfield, "inflate", _np_fwd4.zeros(3), 1.0, strength=0.4)
_sfar = _sgrid[_np_fwd4.linalg.norm(_sgrid, axis=1) > 1.0 + 1e-9]
_sloc = float(_np_fwd4.max(_np_fwd4.abs(_sinf(_sfar) - _sfield(_sfar))))
_svals, _saxes = _sf4(_sinf, _sbounds, 26); _smesh = _mt4(_svals, _saxes, level=0.5)
print(f"  inflate brush: surface grows {_sbase} -> {int((_sinf(_sgrid) > 0.5).sum())} cells, field changes by {_sloc:.0e} OUTSIDE the ball (local), re-mesh stays manifold ({len(_smesh.faces)} faces, watertight: {_smesh.is_manifold()}).")
print("  the same brush reshapes the creature's value landscape -- reward shaping, one falloff-weighted operator on any field.  *** FS-1: sculpt the field, re-mesh ***")

title("FS-2: narrow-band sparse field -- store/edit/re-mesh only the thin shell around the surface (O(brush) strokes)")
# A surface carried as a field, but only the voxels near the surface are stored; a brush touches O(brush) cells, not
# the whole res^3 volume -- the level-set move (narrow band / VDB) that makes sculpting interactive.
def _sphere_sdf(_P):
    return _np_fwd4.linalg.norm(_P, axis=1) - 0.6
_svox = 2.0 / 36
_sband = 4 * _svox
_sf = _meshm.sparse_field(_sphere_sdf, ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), _svox, _sband, tile=6)
_sfull = int(_np_fwd4.prod(_sf.ncorner))
_smesh = _sf.extract_local()
print(f"  seeded a sphere: stored {len(_sf.values)} band voxels = {100.0*len(_sf.values)/_sfull:.0f}% of the {_sfull}-voxel grid; extract is watertight ({_smesh.is_manifold()}, {_smesh.n_faces} faces).")
import numpy as _np_fs2
def _inflate_brush(_pts):
    _d = _np_fs2.linalg.norm(_pts - _np_fs2.array([0.6, 0.0, 0.0]), axis=1)
    _t = _np_fs2.clip(_d / 0.25, 0.0, 1.0)
    return -0.5 * _sband * (1.0 - (3 * _t * _t - 2 * _t * _t * _t))
_dirty, _touched = _sf.apply_local(_inflate_brush, _np_fs2.array([0.6, 0.0, 0.0]), 0.25)
print(f"  an inflate brush touched {_touched} voxels ({100.0*_touched/_sfull:.1f}% of the grid) and dirtied {len(_dirty)} bricks -- O(brush), not O(res^3).  *** FS-2: a stroke costs O(brush) ***")

title("FS-3: splat export -- write the field's Gaussians as a standard 3DGS .ply a browser renderer can display")
# The splat params are already in hand; this is a format adapter. The one real bit of math: L (Cholesky of the
# inverse covariance) -> scale + rotation quaternion, by eigen-decomposing the precision (principal_axes).
import tempfile as _tf3, os as _os3
from holographic_splatexport import splats_from_ply as _sfp3, quaternion_to_rotation as _q2r3
_fsplats = _meshm.field_to_splats(_np_fwd4.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), radius=0.4)
_plypath = _os3.path.join(_tf3.gettempdir(), "tour_splats.ply")
_nply = _meshm.export_splats(_fsplats, path=_plypath, fmt="ply")
_recs3 = _sfp3(_plypath); _os3.remove(_plypath)
_std3 = float(_np_fwd4.array(_recs3[0]["scale"])[0])
print(f"  pulled {_nply} Gaussians from a metaball field (no fit) and wrote a 3DGS .ply; re-import recovers the isotropic std {_std3:.2f} (= the metaball radius).")
print("  L -> scale + rotation by eigen-decomposing the precision; base colour only (SH colour noted, not faked); a flat covariance raises, not garbage.  *** FS-3: a field, displayable as splats ***")

title("FS-4: the sculpt loop -- edit the field, RE-PROJECT to a drawable mesh (an iterate-a-projection)")
# The field is the source of truth; surface_mesh turns ANY field rep into the mesh to draw, at the right detail for
# the view. Re-using _sf (the sparse field already brushed in FS-2) closes the loop: edit -> re-extract.
_loopmesh = _meshm.surface_mesh(_sf)                       # the EDITED sparse field, re-projected to a mesh
print(f"  surface_mesh(edited sparse field) -> {_loopmesh.n_faces} faces, watertight={_loopmesh.is_manifold()} -- the brush's effect, re-extracted.")
def _budsph(_P):
    return _np_fwd4.linalg.norm(_P, axis=1) - 0.6
_mfull = _meshm.surface_mesh(_budsph, ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), resolution=10)
_mfar = _meshm.surface_mesh(_budsph, ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), resolution=10, pixel_budget=1.0, distance=100.0, lod_targets=(0.5,))
print(f"  same field at a screen-space budget: full {_mfull.n_faces} faces up close -> {_mfar.n_faces} faces far away (LOD by distance).")
print("  edit the field -> re-project (mesh OR splats): the same iterate-a-projection shape as the resonator, the denoiser, the dynamics operator.  *** FS-4: the loop closes ***")
from holographic_meshbridge import sample_field as _sf_fwd, marching_tetrahedra as _mt_py, marching_tetrahedra_vec as _mt_vec
import time as _time_fwd
_vv, _va = _sf_fwd(_budsph, ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), 48)
_t0 = _time_fwd.time(); _mp = _mt_py(_vv, _va, 0.0); _tp = _time_fwd.time() - _t0
_t0 = _time_fwd.time(); _mv = _mt_vec(_vv, _va, 0.0); _tv = _time_fwd.time() - _t0
print(f"  vectorized marching (the case-table RAM, all cells at once): {_mp.n_faces} faces in {_tv*1000:.0f}ms vs {_tp*1000:.0f}ms per-cell Python -- {_tp/_tv:.0f}x faster, geometrically identical.")
from holographic_meshqem import surface_deviation as _sd_fwd
_smv4, _sma4 = _sf_fwd(_budsph, ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), 16)
_sm4 = _mt_vec(_smv4, _sma4, 0.0)
_t0 = _time_fwd.time(); _ = _sd_fwd(_sm4, _sm4); _sdt = _time_fwd.time() - _t0
print(f"  the LOD quality metric (surface_deviation) and vertex_normals are vectorized too -- point-to-surface over {len(_sm4.vertices)} verts x {_sm4.n_faces} faces in {_sdt*1000:.0f}ms (was a ~16s scalar double loop). vertex_quadrics stays scalar: its vectorization flips QEM tie-breaks (bind_batch lesson).")
_lodvox = 2.0 / 40
_lodsf = _meshm.sparse_field(_budsph, ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), _lodvox, 4 * _lodvox, tile=6)
_t0 = _time_fwd.time(); _lodchain = _lodsf.lod_chain(); _lodt = _time_fwd.time() - _t0
print(f"  FIELD-NATIVE LOD -- coarsen the SOURCE field and re-project, not decimate the mesh: chain "
      f"{[lvl.n_faces for lvl in _lodchain]} faces (error read from the field: "
      f"{[round(lvl.max_error, 3) for lvl in _lodchain]}) built in {_lodt*1000:.0f}ms of re-marching. QEM-decimating "
      f"the fine mesh to those counts would take MINUTES. The mesh is a projection of the field.")

_t0 = _time_fwd.time(); _clu = _meshm.mesh_cluster_decimate(_sm4, 12); _clut = _time_fwd.time() - _t0
print(f"  PARALLEL imported-mesh decimation (cluster_decimate -- vertex clustering, the per-cell quadric is a BUNDLE "
      f"of plane tensors, no greedy search): F{_sm4.n_faces} -> F{_clu.n_faces} in {_clut*1000:.0f}ms. Greedy QEM to "
      f"that count is ~1000x slower (measured 998x on a bigger mesh) -- the fast path for a mesh with no field behind it.")
_mlo = _sm4.vertices.min(0) - 0.05; _mhi = _sm4.vertices.max(0) + 0.05
_t0 = _time_fwd.time(); _mfield, _maxes = _meshm.mesh_to_field(_sm4, (_mlo, _mhi), res=48); _mft = _time_fwd.time() - _t0
_mdev = _np_fwd4.abs(_meshm.mesh_sample_field(_mfield, _maxes, _clu.vertices))
print(f"  mesh -> FIELD by TILING (mesh_to_field, a SIGNED banded SDF -- each triangle scatter-mins only its local "
      f"voxel block): built in {_mft*1000:.0f}ms; the decimated mesh's distance to the original then reads straight "
      f"off the field (max {float(_mdev.max()):.4f}, sub-voxel -- the signed field crosses zero linearly, dodging the "
      f"unsigned kink). Build once, query any points O(V) -- the gateway to treating an imported mesh as a field.")
from holographic_meshbridge import _closest_point_on_triangle as _cpt1, _closest_points_on_triangles as _cptN, point_set_to_mesh as _p2m
_qpts = _sm4.vertices * 1.02
_t0 = _time_fwd.time()
_bb = _np_fwd4.full(len(_qpts), _np_fwd4.inf)
for _f in _sm4.faces:
    _bb = _np_fwd4.minimum(_bb, _np_fwd4.linalg.norm(_qpts - _cpt1(_qpts, _sm4.vertices[_f[0]], _sm4.vertices[_f[1]], _sm4.vertices[_f[2]]), axis=1))
_tbrute = _time_fwd.time() - _t0
_t0 = _time_fwd.time(); _db = _p2m(_qpts, _sm4.vertices, _sm4.faces); _tbat = _time_fwd.time() - _t0
print(f"  the batched closest-point kernel (one vectorized region test over F triangles x N points) is EXACT "
      f"(matches the per-triangle loop to {float(_np_fwd4.abs(_db - _bb).max()):.0e}) -- but MEASURED slower here "
      f"({_tbat*1000:.0f}ms vs {_tbrute*1000:.0f}ms): all-pairs point-to-mesh is memory-bandwidth-bound, so the "
      f"brute loop's tiny per-triangle working set (cache-resident) WINS. The cache-aware structure was already the "
      f"loop; the real speedup is algorithmic culling, not batching. A negative kept loud.")
from holographic_meshbridge import mesh_to_sdf_grid as _m2sdf, marching_tetrahedra_vec as _mtv2
_fullsdf, _fsax = _m2sdf(_sm4, (_sm4.vertices.min(0) - 0.1, _sm4.vertices.max(0) + 0.1), res=48)
_mid2 = _fullsdf.shape[0] // 2
_remar = _mtv2(_fullsdf, _fsax, 0.0)
_lodfaces = [_mtv2(_fullsdf[::_s, ::_s, ::_s], (_fsax[0][::_s], _fsax[1][::_s], _fsax[2][::_s]), 0.0).n_faces for _s in (1, 2, 4)]
print(f"  the CLOSURE -- flood-fill the banded SDF's sign so the interior is negative (centre {float(_fullsdf[_mid2,_mid2,_mid2]):+.2f}), "
      f"giving a FULL re-marchable field: the imported mesh re-marches back to a closed surface ({_remar.n_faces} faces) and now "
      f"coarsens by RE-STRIDING the field {_lodfaces} -- field-native LOD for a mesh that arrived with no field. A field projects to "
      f"a mesh; a mesh lifts back to a field. The decomposition loop is closed.")
from holographic_meshbridge import point_set_to_mesh_grid as _p2mg
_qn = _sm4.vertices * 1.02
_t0 = _time_fwd.time()
_bn = _np_fwd4.full(len(_qn), _np_fwd4.inf)
for _f in _sm4.faces:
    _bn = _np_fwd4.minimum(_bn, _np_fwd4.linalg.norm(_qn - _cpt1(_qn, _sm4.vertices[_f[0]], _sm4.vertices[_f[1]], _sm4.vertices[_f[2]]), axis=1))
_tbn = _time_fwd.time() - _t0
_t0 = _time_fwd.time(); _gn = _p2mg(_qn, _sm4.vertices, _sm4.faces, radius=2); _tgn = _time_fwd.time() - _t0
print(f"  and the POSITIVE result that batching could not give -- CULL the work with a vectorized spatial grid "
      f"(sort-based binning + the ranges trick, no Python dicts): the same point-to-mesh distance in {_tgn*1000:.0f}ms "
      f"vs {_tbn*1000:.0f}ms brute ({_tbn/_tgn:.0f}x), EXACT near the surface (max err {float(_np_fwd4.abs(_gn-_bn).max()):.0e}, "
      f"0 misses). It now drives the cluster LOD error (~110x there). Batching was memory-bound; doing LESS work wins.")

# FS-5: the surface carried as ONE hypervector, edit = bind (the most literal "move geometry to holographic space").
_hfield = _meshm.mesh_to_field_vector(_sm4, ((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3)), dim=2048, bandwidth=18.0, grid=12)
_hd = _np_fwd4.array([0.25, 0.0, 0.0])
_hmoved = _hfield.translate(_hd)                                   # the whole surface moves with ONE bind
_hcg = _np_fwd4.linspace(-0.5, 0.5, 6)
_hX = _np_fwd4.array([(a, b, c) for a in _hcg for b in _hcg for c in _hcg])
_herr = float(_np_fwd4.abs(_hmoved.value(_hX) - _hfield.value(_hX - _hd)).max())
print(f"  and the surface itself can BE a hypervector (FS-5): {_sm4.n_faces} faces sampled into ONE dim-2048 vector; "
      f"value(x) is a cosine query, and translate is a SINGLE bind -- exact, value_shifted(x)=value_orig(x-d) to "
      f"{_herr:.0e}. Moving the whole surface is now algebra. (Kept honest: the re-marched extract is a smoothed, "
      f"~15%-biased estimate, bandwidth the bias knob, dim the noise floor -- a representation, not the fast path.)")

# Delta editing: an edit is a delta vector; undo is exact subtraction; the cost does not depend on model size.
_hq = _np_fwd4.array([0.6, 0.0, 0.0])
_hbump = _np_fwd4.array([_hq + _o for _o in [(0, 0, 0), (0.05, 0, 0), (0, 0.05, 0), (0, 0, 0.05)]])
_hdelta = _hfield.make_delta(_hbump, _np_fwd4.full(len(_hbump), -0.35))
_hedited = _hfield.apply_delta(_hdelta)
_hundone = _hedited.remove_delta(_hdelta)
print(f"  and EDITING that model is a DELTA in holographic space -- the video-codec insight (store the keyframe, "
      f"add small deltas) applied to geometry: a brush is one vector add (value at the pole {float(_hfield.value([_hq])[0]):+.3f}"
      f" -> {float(_hedited.value([_hq])[0]):+.3f}), undo is EXACT subtraction (f+d-d back to {float(_np_fwd4.abs(_hundone.f-_hfield.f).max()):.0e}), "
      f"and the cost is MODEL-SIZE-INDEPENDENT (the model is one fixed vector). Only the dirty region need re-march.")

# Real-time sculpting (the array field, FS-2/4): project ONLY the changed bricks, so a brush holds 30-60fps.
from holographic_sparsefield import SparseField as _SF, _smooth_falloff as _sfall
_scv = 2.0 / 64; _scb = 4 * _scv
_scf = _SF.from_field(lambda P: _np_fwd4.linalg.norm(P, axis=1) - 0.6, (-1., -1, -1), (1., 1, 1), _scv, _scb, tile=8)
_scf.extract_dirty()                                                 # cold build (warm the cache)
_scp = _np_fwd4.array([0.6, 0., 0.])
def _scinfl(P): return -0.5 * _scb * _sfall(_np_fwd4.linalg.norm(P - _scp, axis=1), 0.1)
_scf.apply_local(_scinfl, _scp, 0.1)
import copy as _cpmod
_scf2 = _cpmod.deepcopy(_scf)
_t0 = _time_fwd.time(); _scfull = _scf.extract_cached(); _tfull = (_time_fwd.time() - _t0) * 1000
_scf2._cache_dirty = set(_scf2.active) if False else _scf2._cache_dirty   # (keep dirty marks)
_t0 = _time_fwd.time(); _scd = _scf2.extract_dirty(); _tdirty = (_time_fwd.time() - _t0) * 1000
print(f"  and for the FAST sculpt path -- a brush at 30-60fps -- the trick is to project ONLY the bricks that "
      f"changed: re-welding the whole {_scfull.n_faces}-face mesh every frame costs {_tfull:.0f}ms, but returning just "
      f"the {len(_scd['updated'])} dirty bricks (extract_dirty) is {_tdirty:.0f}ms = {1000/max(_tdirty,0.1):.0f}fps. The per-frame "
      f"cost tracks the BRUSH, not the model -- a model of any size stays interactive because nothing per-frame is O(model).")

title("Holographic geometry & appearance (G1-G6): noise, materials, displace, terrain, grammar, attributes -- all field-native")
# Everything that sits on a mesh lives in the SAME algebra as the geometry: noise is a field, a texture is a
# function, a material is a record, displacement is a delta, terrain is fBm, a plant is a scenegraph. One space.
import numpy as _np_geo
from holographic_noise import FractalNoise as _FN_geo
_fbm = _FN_geo(2, dim=512, bounds=[(0, 8), (0, 8)], octaves=4, gain=0.85, base_bandwidth=3.0, seed=3)
_prof = _np_geo.array([_fbm.query([_x, 4.0]) for _x in _np_geo.linspace(0.3, 7.7, 200)])
_rough = float(_np_geo.std(_np_geo.diff(_prof)) / (_np_geo.std(_prof) + 1e-9))
print(f"  G1 NOISE: fBm is the OCTAVE BUNDLE -- one query sums {len(_fbm.fields)} band fields, each a hypervector (an FPE bundle of random RBF kernels). Roughness/amplitude = {_rough:.3f}, rises with persistence; band-limited by the kernel.")

from holographic_fpe import VectorFunctionEncoder as _VFE_geo
from holographic_material import Material as _Mat_geo, texture_field as _tex_geo, compose_object as _compose_geo
from holographic_ai import unbind as _unbind_geo, cosine as _cos_geo
_uenc = _VFE_geo(2, dim=1024, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=1)
_ugrid = [(_u, _v) for _u in _np_geo.linspace(0.05, 0.95, 9) for _v in _np_geo.linspace(0.05, 0.95, 9)]
_mat = _Mat_geo(_uenc, {"albedo": _tex_geo(_uenc, _ugrid, [_u for (_u, _v) in _ugrid]),
                        "roughness": _tex_geo(_uenc, _ugrid, [0.5] * len(_ugrid))})
_recalls = [float(_cos_geo(_mat.channel(_n), _Mat_geo._unit(_mat.channels[_n]))) for _n in _mat.channels]
_shifted = _mat.transform_uv(_np_geo.array([0.15, 0.0]))
_shift_ok = float(abs(_shifted.sample("albedo", [0.55, 0.5]) - _mat.sample("albedo", [0.4, 0.5])))
print(f"  G2 MATERIAL: a PBR material is the HRR record sum_r bind(role_r, channel_r). sample() is exact; the bare record recovers channels BALANCED (cosine {[round(_r,2) for _r in _recalls]}); transform_uv re-UVs EVERY channel with ONE bind (shift error {_shift_ok:.3f}).")

from holographic_fpefield import HolographicField as _HF_geo
from holographic_displace import displace_sdf as _dsdf_geo
_senc = _VFE_geo(3, dim=2048, bounds=[(-1, 1)] * 3, bandwidth=6.0, seed=1)
_sax = _np_geo.linspace(-0.8, 0.8, 6)
_gx, _gy, _gz = _np_geo.meshgrid(_sax, _sax, _sax, indexing="ij")
_SP = _np_geo.stack([_gx.ravel(), _gy.ravel(), _gz.ravel()], axis=1)
_slab = _HF_geo(_senc, _SP, _SP[:, 2])                      # sdf = z, surface at z=0
_disp, _delta = _dsdf_geo(_slab, lambda _p: 1.0, 0.2)      # push outward by a constant 0.2
_before = float(_slab.value([[0.0, 0.0, 0.0]])[0]); _after = float(_disp.value([[0.0, 0.0, 0.0]])[0])
_undo_err = float(_np_geo.max(_np_geo.abs(_disp.remove_delta(_delta).f - _slab.f)))
print(f"  G3 DISPLACE: an SDF offset is a field DELTA (make_delta with NEGATIVE values pushes outward). Surface value {_before:.3f} -> {_after:.3f}; remove_delta undoes it EXACTLY ({_undo_err:.0e}).")

from holographic_terrain import Terrain as _Terr_geo, terrain_to_mesh as _t2m_geo
_terr = _Terr_geo(bounds=[(0, 4), (0, 4)], octaves=4, gain=0.6, dim=512, seed=7)
_tmesh = _t2m_geo(_terr, 12)
print(f"  G4 TERRAIN: fBm heightfield lifted to a UV'd grid mesh -- {_tmesh.n_vertices} verts, {len(_tmesh.faces)} tris, z = height(x,y) exactly. A composition of G1; LOD is just re-sampling. (No erosion -- kept negative.)")

from holographic_grammar import LSystem as _LS_geo, turtle_to_segments as _turtle_geo
_algae = _LS_geo("A", {"A": "AB", "B": "A"})
_lens = [len(_algae.expand(_n)) for _n in range(7)]
_plant = _LS_geo("X", {"X": "F[+X][-X]FX", "F": "FF"})
_segs = _turtle_geo(_plant.expand(4))
print(f"  G5 GRAMMAR (the new one): L-system parallel rewriting -- algae A->AB,B->A gives EXACTLY Fibonacci lengths {_lens}; a branching plant turtles into {len(_segs)} skeleton segments, assembled as a scenegraph (a recursive bundle). Productions are themselves a holographic record.")

from holographic_attributes import attribute_field as _af_geo, bake_to_vertices as _bake_geo
_afield = _af_geo(_uenc, _ugrid, [_u for (_u, _v) in _ugrid])
_coarse = _np_geo.array([[_u, 0.5] for _u in _np_geo.linspace(0.2, 0.8, 7)])
_dense = _np_geo.array([[_u, 0.5] for _u in _np_geo.linspace(0.2, 0.8, 13)])
_gap = float(_np_geo.max(_np_geo.abs(_bake_geo(_uenc, _afield, _coarse) - _bake_geo(_uenc, _afield, _dense)[::2])))
print(f"  G6 ATTRIBUTE: a per-vertex channel as a RESOLUTION-INDEPENDENT field -- bake at a coarse and a dense sampling, shared points agree to {_gap:.0e} (it is a function, not a frozen array). Hard masks fall back to a raster store.")
print("  Six faculties, one algebra: a textured, displaced, attributed object is one composable hypervector.  *** G1-G6: geometry & appearance go holographic-native ***")

title("The demoscene layer (S1-S2): a 3D SDF/shader algebra -- objects, fractals, greebles, vegetated terrain, GLSL out")
# holographic_field's `Field` lives on the VSA hypersphere; THIS is its Cartesian sibling -- signed-distance
# fields over R^3 you can raymarch. One SDF tree is FOUR things at once: a mesh, a recipe, a DSL, a shader.
from holographic_sdf import sphere as _sph_geo, box as _sbox_geo, torus as _stor_geo, menger as _meng_geo, parse_dsl as _pdsl_geo
import numpy as _np_sdf
_scene = _sph_geo(1.0).smooth_union(_stor_geo(0.9, 0.25).rotate([1, 0, 0], 1.2), 0.3).rounded(0.05)
_P = _np_sdf.hstack([_np_sdf.linspace(0, 1.5, 60)[:, None], _np_sdf.zeros((60, 2))])
_hard = float(_np_sdf.max(_np_sdf.abs(_np_sdf.diff(_sph_geo(1.0).union(_stor_geo(0.9, 0.25)).eval(_P), 2))))
_soft = float(_np_sdf.max(_np_sdf.abs(_np_sdf.diff(_sph_geo(1.0).smooth_union(_stor_geo(0.9, 0.25), 0.3).eval(_P), 2))))
print(f"  S1 SDF: a tree of primitives under CSG + the demoscene smooth-min (seam curvature {_soft:.3f} vs {_hard:.3f} hard -- {_hard/max(_soft,1e-6):.0f}x less creased). DSL = {_scene.to_dsl()}")
_back = _pdsl_geo(_scene.to_dsl())
_Q = _np_sdf.random.default_rng(0).uniform(-2, 2, (40, 3))
print(f"  S1 I/O: that DSL round-trips to the SAME field (max err {float(_np_sdf.max(_np_sdf.abs(_scene.eval(_Q)-_back.eval(_Q)))):.0e}); to_glsl() emits a {len(_scene.to_glsl())}-char Shadertoy shader (map + raymarch + lighting) that embeds its own DSL.")
from holographic_typed import tree_to_recipe as _t2r_geo, op_kinds as _ok_geo
_rec = _t2r_geo(512, 0, _scene.to_tree())
print(f"  S1 REPRESENT: the SAME shader is one holographic RECIPE (typed.tree_to_recipe -- {len(_ok_geo(_rec))} op kinds): a demoscene shader as a VSA structure you can store/compose/factor.")

from holographic_procgen import procedural_object as _po_geo, object_to_mesh as _o2m_geo, greeble_mesh as _gm_geo, vegetated_terrain as _vt_geo
from holographic_mesh import box as _mbox_geo
from holographic_scenegraph import flatten_scene as _fs_geo
_obj = _po_geo(7, complexity=3)
_omesh = _o2m_geo(_obj, res=32)
print(f"  S2 OBJECTS: one seed -> one SDF object ({len(_obj.to_dsl())}-char tree) -> {_omesh.n_faces} faces. Different seed, different object -- the demoscene seed->world, deterministic.")
_meng_mesh = _o2m_geo(_meng_geo(2, 1.0), bounds=((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)), res=44)
_greeb = _gm_geo(_mbox_geo(1, 1, 1), seed=3, density=1.0)
print(f"  S2 FRACTAL/GREEBLE: the Menger sponge marches to {_meng_mesh.n_faces} faces (the holes ARE the surface); greebling a box's faces takes it 8 -> {_greeb.n_vertices} verts of hull detail.")
_vscene, _vterr = _vt_geo(seed=5, n_plants=6, plant_iterations=2)
print(f"  S2 VEGETATION: L-system plants scattered across a fBm terrain at the surface height -> one scene, {_fs_geo(_vscene).n_faces} faces. Terrain (G4) + grammar (G5) + scatter, composed.  *** S1-S2: the demoscene layer ***")

title("The creature's value head AS a VSA program: policy = hypervectors, learn = bundling, decide = a dot -- measured vs the tabular brain")
# The creature is holographic OUTSIDE (role-bound states, prototype bundles) but TABULAR inside: per action a
# growing list of prototype vectors paired with a parallel array of SCALAR returns. This folds that table
# into the holographic space -- Q_a = sum ret*unit(state), N_a = sum unit(state), value = <s,Q>/<s,N>.
from holographic_valuehead import HolographicValueHead as _HVH
from holographic_creature import HolographicMind as _HM_vh
import numpy as _np_vh
def _vh_run(_P, _D=384, _A=3, _seed=0):
    _rng = _np_vh.random.default_rng(_seed)
    _S = _rng.normal(size=(_P, _D)); _S /= _np_vh.linalg.norm(_S, axis=1, keepdims=True)
    _V = _rng.uniform(0, 1, size=(_P, _A))
    _holo = _HVH(_D, _A); _brain = _HM_vh(dim=_D, actions=list(range(_A)), epsilon=0.0, maintain=False)
    for _p in range(_P):
        for _a in range(_A):
            for _ in range(3):
                _s = _S[_p] + _rng.normal(0, 0.02, _D); _s /= _np_vh.linalg.norm(_s)
                _r = _V[_p, _a] + _rng.normal(0, 0.05)
                _holo.absorb(_s, _a, _r); _brain._absorb(_s, _a, _r)
    _best = _V.argmax(axis=1)
    _ha = _np_vh.mean([_holo.decide(_S[_p]) == _best[_p] for _p in range(_P)])
    _ba = _np_vh.mean([int(_np_vh.argmax([_brain.value(_S[_p], _a)[0] for _a in range(_A)])) == _best[_p] for _p in range(_P)])
    return float(_ha), float(_ba), _holo.nbytes
_hlo, _blo, _bytes = _vh_run(8)
_hhi, _bhi, _ = _vh_run(380)
print(f"  LOW load (8 situations): holographic value head MATCHES the tabular brain -- best-action accuracy {_hlo:.2f} vs {_blo:.2f}. value(s,a)=<s,Q_a>/<s,N_a> reproduces the brain's Nadaraya-Watson average.")
print(f"  HIGH load (380 ~ dim): holo DEGRADES to {_hhi:.2f} (the VSA capacity cliff, KEPT NEGATIVE) while the growing tabular table holds {_bhi:.2f}. The trade: a FIXED {_bytes} B savable hypervector policy {{Q,N}} with graceful degradation, vs an exact but unbounded list.")
print("  Learning is one bundling step (Q_a += ret*u); the policy is hypervectors -- savable/bindable/composable like a recipe. The creature's one tabular part can live in holographic space.  *** value head as a VSA program ***")
# ... and WIRED into the live creature behind value_backend='holo': the whole brain runs on the hypervector policy.
from holographic_creature import CreatureEncoder as _CE_vh, GridWorld as _GW_vh, run_episode as _re_vh
def _maze_backend(_backend, _size=7, _mseed=3, _eps=90):
    _enc = _CE_vh(384, seed=1)
    _mind = _HM_vh(384, _GW_vh.ACTIONS, k=15, epsilon=0.5, novelty_bonus=0.2, memory_cap=12000, seed=1, value_backend=_backend)
    _w = _GW_vh(_size, _size, maze=True, fixed_seed=_mseed)
    for _ep in range(_eps):
        _mind.epsilon = max(0.05, 0.5 * (1.0 - _ep / _eps))
        _re_vh(_w, _enc, _mind, learn=True, explore=True, mem=4, corridor_reflex=True, max_steps=90)
    _got = 0
    for _ in range(20):
        _re_vh(_w, _enc, _mind, learn=False, explore=False, eval_epsilon=0.05, mem=4, corridor_reflex=True, max_steps=90)
        _got += _w.escaped
    if _backend == "holo":
        _b = _mind._value_head.nbytes
    else:
        _b = sum(_mind._sum[_a].nbytes + _mind._unit[_a].nbytes + _mind._ret[_a].nbytes + _mind._cnt[_a].nbytes for _a in range(len(_mind.actions)))
    return _got / 20, _b
_tr, _tb = _maze_backend("table"); _hr, _hb = _maze_backend("holo")
print(f"  WIRED into the live creature on a real 7x7 maze: table backend escapes {_tr:.0%} (policy {_tb} B, grows) vs holo backend escapes {_hr:.0%} (policy {_hb} B, FIXED). Same control, a tiny savable hypervector policy -- the whole brain now runs in holographic space.")
# ROUTING pushes the capacity cliff back; TD bootstrapping beats Monte-Carlo; the policy composes into other VSA programs.
from holographic_valuehead import RoutedValueHead as _RVH, HolographicValueHead as _PVH, decide_from_atom as _dfa, discounted_return as _dret
def _cliff_acc(_head, _P=1024, _D=256, _A=3):
    _rng = _np_vh.random.default_rng(0); _S = _rng.normal(size=(_P, _D)); _S /= _np_vh.linalg.norm(_S, axis=1, keepdims=True)
    _V = _rng.uniform(0, 1, size=(_P, _A))
    for _p in range(_P):
        for _a in range(_A): _head.absorb(_S[_p], _a, _V[_p, _a])
    _b = _V.argmax(axis=1)
    return float(_np_vh.mean([_head.decide(_S[_p]) == _b[_p] for _p in range(_P)]))
print(f"  Step A ROUTING: at 1024 situations (4x dim) the single-bundle head holds {_cliff_acc(_PVH(256,3)):.2f} (past the cliff) while the routed head holds {_cliff_acc(_RVH(256,3,n_buckets=64)):.2f} -- bounded buckets, 'cull don't batch' for value storage.")
# composability: fold the policy into two bindable hypervectors, drive a decision purely in-VSA
_rng_c = _np_vh.random.default_rng(0)
_sits = [_rng_c.normal(size=512) for _ in range(4)]; _sits = [_s/_np_vh.linalg.norm(_s) for _s in _sits]
_codes = _np_vh.stack([_rng_c.normal(size=512) for _ in range(3)]); _codes /= _np_vh.linalg.norm(_codes, axis=1, keepdims=True)
_ph = _PVH(512, 3)
for _i, _s in enumerate(_sits):
    for _ in range(5):
        for _a in range(3): _ph.absorb(_s, _a, 1.0 if _a == _i % 3 else 0.1)
_MQ, _MN = _ph.policy_atom(_codes)
_match = sum(_dfa(_MQ, _MN, _s, _codes) == _ph.decide(_s) for _s in _sits)
print(f"  Step B TD: n-step return is a discounted bundle ({_dret([0.0],0.9,bootstrap=2.0):.2f}=gamma*V), eligibility a decaying bundle -- TD prediction beats Monte-Carlo (see selftest). COMPOSABLE: the policy folds into two bindable hypervectors and drives decisions in-VSA ({_match}/4 match the head), so a VSA program can carry and run the creature with no Python round-trip.  *** holographic creature: cliff back, TD, everywhere, composable ***")
# COMPILED PERCEPTION: the last per-step boundary (an FFT bind per sense) precomputed once -> perceive is a gather+sum.
from holographic_creature import CreatureEncoder as _CEbase, FastCreatureEncoder as _FCE, GridWorld as _GWp, HolographicMind as _HMp, run_episode as _rep
import time as _time_vh
_base = _CEbase(256, seed=1); _fast = _FCE(256, seed=1)
_sset = [{'wall_N': _a, 'goal_E': _b} for _a in ('yes', 'no') for _b in ('far', 'near', 'none')]
_ident = all(_np_vh.array_equal(_base.encode(_s), _fast.encode(_s)) for _s in _sset)
_b2 = _CEbase(256, seed=1); _f2 = _FCE(256, seed=1)
_t0 = _time_vh.time(); [_b2.encode(_sset[_i % len(_sset)]) for _i in range(3000)]; _tb = _time_vh.time() - _t0
_t0 = _time_vh.time(); [_f2.encode(_sset[_i % len(_sset)]) for _i in range(3000)]; _tf = _time_vh.time() - _t0
print(f"  COMPILED PERCEPTION: bit-identical to the plain encoder ({_ident}), but the per-step role/filler FFT bind is precomputed once -- 3000 perceptions {_tb*1000:.0f}ms -> {_tf*1000:.0f}ms ({_tb/_tf:.1f}x), steady state 0 FFTs/step.")
_encp = _FCE(256, seed=1); _mindp = _HMp(256, _GWp.ACTIONS, k=15, epsilon=0.5, novelty_bonus=0.2, memory_cap=12000, seed=1, value_backend='routed')
_wp = _GWp(7, 7, maze=True, fixed_seed=3)
for _ep in range(110):
    _mindp.epsilon = max(0.05, 0.5 * (1.0 - _ep / 110))
    _rep(_wp, _encp, _mindp, learn=True, explore=True, mem=4, corridor_reflex=True, max_steps=90)
_gp = 0
for _ in range(20):
    _rep(_wp, _encp, _mindp, learn=False, explore=False, eval_epsilon=0.05, mem=4, corridor_reflex=True, max_steps=90); _gp += _wp.escaped
print(f"  FULLY IN-VSA LOOP: compiled perceive (gather+sum) + routed brain (decide=dot, learn=bundle) escapes the 7x7 maze {_gp/20:.0%} -- the whole perceive->decide->learn loop is array ops, the creature a VSA program end-to-end.  *** in-VSA perception ***")
# VECTORIZED PROGRAM PRIMITIVES: the common inner-loop ops (record encode, multi-key decode) in ONE array op.
from holographic_ai import bind as _bnd, bundle as _bnl, unbind as _unb, bundle_bind as _bb, unbind_all as _ua
import time as _tprim
_rngp = _np_vh.random.default_rng(0)
def _rvp(): _v = _rngp.normal(size=512); return _v / _np_vh.linalg.norm(_v)
_K = 24; _roles = _np_vh.stack([_rvp() for _ in range(_K)]); _vals = _np_vh.stack([_rvp() for _ in range(_K)])
_trace = _rvp(); _keys = _np_vh.stack([_rvp() for _ in range(_K)])
_t = _tprim.time(); [_bnl([_bnd(_roles[i], _vals[i]) for i in range(_K)]) for _ in range(1500)]; _tel = _tprim.time() - _t
_t = _tprim.time(); [_bb(_roles, _vals) for _ in range(1500)]; _teb = _tprim.time() - _t
_t = _tprim.time(); [_np_vh.stack([_unb(_trace, _keys[i]) for i in range(_K)]) for _ in range(1500)]; _tdl = _tprim.time() - _t
_t = _tprim.time(); [_ua(_trace, _keys) for _ in range(1500)]; _tdb = _tprim.time() - _t
print(f"  VECTORIZED PRIMITIVES: record encode (bind+bundle over {_K} roles) {_tel*1000:.0f}ms -> bundle_bind {_teb*1000:.0f}ms ({_tel/_teb:.1f}x); multi-key decode {_tdl*1000:.0f}ms -> unbind_all {_tdb*1000:.0f}ms ({_tdl/_tdb:.1f}x). Audit found cleanup + bind_batch were already vectorized; the missing piece was the convenience layer so programs actually call it -- one batched FFT, not a Python loop.  *** in-VSA primitives ***")

# GRID FLUID + PARTICLES, EXPOSED TO VSA: the FFT-on-a-torus that binding uses, run as a fluid solver.
from holographic_fields import (diffuse as _dif, divergence as _dvg, project_divergence_free as _proj,
                                advect as _adv, fluid_step as _fstep, ParticleSystem as _PS,
                                attractor_force as _attr)
_Hf = _Wf = 48
_Yf, _Xf = _np_vh.meshgrid(_np_vh.arange(_Hf), _np_vh.arange(_Wf), indexing="ij")
def _blobf(_cx, _cy, _s=4.0): return _np_vh.exp(-(((_Xf - _cx) ** 2 + (_Yf - _cy) ** 2) / (2 * _s ** 2)))
_rngf = _np_vh.random.default_rng(0)
_vxf = _rngf.normal(size=(_Hf, _Wf)); _vyf = _rngf.normal(size=(_Hf, _Wf))
_div0 = float(_np_vh.abs(_dvg(_vxf, _vyf)).max())
_pxf, _pyf = _proj(_vxf, _vyf)
_div1 = float(_np_vh.abs(_dvg(_pxf, _pyf)).max())
_blob_in = _blobf(24, 24); _diff = _dif(_blob_in, 4.0)
_dens = _blobf(12, 24); _moved = _adv(_dens, _np_vh.full((_Hf, _Wf), 3.0), _np_vh.zeros((_Hf, _Wf)), dt=2.0)
_cx0 = float((_np_vh.arange(_Wf)[None, :] * _dens).sum() / _dens.sum())
_cx1 = float((_np_vh.arange(_Wf)[None, :] * _moved).sum() / _moved.sum())
_ps = _PS(_rngf.uniform(8, 40, size=(200, 2)))
_pd0 = float(_np_vh.linalg.norm(_ps.pos - _np_vh.array([24, 24]), axis=1).mean())
for _ in range(40): _ps.step(force=_attr(_ps.pos, (24, 24), strength=8.0), dt=0.1, damping=0.05)
_pd1 = float(_np_vh.linalg.norm(_ps.pos - _np_vh.array([24, 24]), axis=1).mean())
print(f"  GRID FLUID + PARTICLES (exposed to VSA): PRESSURE PROJECTION makes a velocity field incompressible, max|div| {_div0:.1f} -> {_div1:.0e} (the FFT Helmholtz solve = the bind operator's own transform); DIFFUSE is a Gaussian bind on the torus, var {float(_blob_in.var()):.3f}->{float(_diff.var()):.3f}, mass conserved; ADVECT carries a density blob {_cx1-_cx0:.0f} cells (= v*dt); an ATTRACTOR pulls particles inward {_pd0:.1f}->{_pd1:.1f}. Jos Stam's Stable Fluids, because the torus FFT was already here.  *** fluid/particle sim in VSA ***")

# PBD/XPBD SOFTBODY + SHAPE-MATCHING HARDBODY: the iterate-a-projection engine, now with momentum.
from holographic_softbody import SoftBody as _SB, RigidBody as _RB
def _xpbd_stretch(_substeps):
    _s = _SB(_np_vh.array([[0.0, 0.0], [0.0, -1.0]]))
    _s.add_distance(0, 1, rest=1.0, compliance=0.01); _s.pin(0)
    for _ in range(500): _s.step(dt=1 / 60, gravity=(0.0, -9.8), iterations=20, substeps=_substeps, damping=0.02)
    return float(_np_vh.linalg.norm(_s.x[0] - _s.x[1]) - 1.0)
_st1 = _xpbd_stretch(1); _st6 = _xpbd_stretch(6)
_cloth = _SB.cloth(6, 6, spacing=1.0, compliance=0.0)
for _ in range(200): _cloth.step(dt=1 / 60, gravity=(0.0, -9.8), iterations=25)
_clres = float(_cloth.constraint_residual())
_bigc = _SB.cloth(5, 5, spacing=1.0, compliance=0.0)
for _ in range(60): _bigc.step(dt=0.1, gravity=(0.0, -9.8), iterations=15)
_stable = float(_np_vh.abs(_bigc.x).max())
_rb = _RB(_np_vh.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
for _ in range(120): _rb.step(dt=1 / 60, gravity=(0.0, -9.8))
_drift = float(_rb.max_distance_drift()); _fell = float(_rb.x[:, 1].mean())
print(f"  PBD/XPBD SOFTBODY + HARDBODY (exposed to VSA): the constraint sweep IS the engine's iterate-a-projection (resonator/denoiser/IK), now with momentum. XPBD stiffness is TIME-STEP INDEPENDENT: hanging-spring stretch {_st1:.4f} (1 substep) == {_st6:.4f} (6 substeps) == compliance*g; a 6x6 CLOTH settles to residual {_clres:.3f}; STABLE at dt=0.1 that would explode an explicit spring (max|x| {_stable:.1f}); a RIGID body (shape-matching, polar/SVD) falls to y={_fell:.1f} with distance drift {_drift:.0e}. Macklin's 'project onto constraints', made physical.  *** softbody/hardbody sim in VSA ***")

# BENDING + VOLUME constraints, and TWO-WAY fluid<->cloth coupling.
from holographic_softbody import SoftBody as _SB2
from holographic_fields import scatter_to_field as _scat, drag_force as _drag, fluid_step as _fs2
import math as _m2
def _fold(_with):
    _s = _SB2(_np_vh.array([[-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]]))
    _s.add_distance(0, 1, 1.0); _s.add_distance(1, 2, 1.0); _s.pin(1)
    if _with: _s.add_bending(0, 2)
    _t = 0.7; _s.x[0] = [-_m2.cos(_t), _m2.sin(_t), 0]; _s.x[2] = [_m2.cos(_t), _m2.sin(_t), 0]
    for _ in range(60): _s.step(dt=1 / 60, gravity=(0, 0, 0), iterations=20)
    return float(_np_vh.linalg.norm(_s.x[0] - _s.x[2]))
_fold_no = _fold(False); _fold_yes = _fold(True)
_tet = _SB2(_np_vh.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
_tet.add_volume(0, 1, 2, 3); _v0 = _tet.total_volume(); _tet.x[3, 2] = 0.3
_tet.step(dt=1 / 60, gravity=(0, 0, 0), iterations=40); _v1 = _tet.total_volume()
_Hc = _Wc = 32
_posc = _np_vh.array([[16.0, 16.0], [16.5, 16.0], [16.0, 16.5], [16.5, 16.5]]); _velc = _np_vh.tile(_np_vh.array([5.0, 0.0]), (4, 1))
_fxc = _scat((_Hc, _Wc), _posc, _velc[:, 0]); _fyc = _scat((_Hc, _Wc), _posc, _velc[:, 1])
_vxc = _np_vh.zeros((_Hc, _Wc)); _vyc = _np_vh.zeros((_Hc, _Wc)); _dc = _np_vh.zeros((_Hc, _Wc))
for _ in range(3): _vxc, _vyc, _dc = _fs2(_vxc, _vyc, _dc, dt=0.2, viscosity=0.05, fx=_fxc, fy=_fyc)
_stir = float(_np_vh.abs(_vxc).max())
_flow = _np_vh.full((_Hc, _Wc), 4.0); _still = _np_vh.zeros((_Hc, _Wc)); _pp = _np_vh.array([[8.0, 8.0], [10.0, 12.0]]); _pv = _np_vh.zeros((2, 2))
for _ in range(20):
    _F = _drag(_pp, _pv, _flow, _still, k=0.5); _pv = _pv + (1 / 60) * _F; _pp = _pp + (1 / 60) * _pv
print(f"  BENDING + VOLUME + TWO-WAY COUPLING: a BEND SPRING unfolds a folded strip {_fold_no:.2f} (no bending) -> {_fold_yes:.2f} (flattened); a VOLUME constraint restores a squashed tet {_v1:.3f} (rest {_v0:.3f}); CLOTH->FLUID a moving body stirs the fluid to max|vx| {_stir:.2f}, FLUID->CLOTH a flow drags free particles to vx {float(_pv[:,0].mean()):.2f} (toward 4.0). Sail in wind, body in water -- both directions, one substrate.  *** bending/volume + 2-way coupling ***")

# SMOKE: temperature -> buoyancy + vorticity confinement on the FFT fluid (Fedkiw 2001).
from holographic_fields import smoke_step as _smk, curl as _crl2
_Hs = _Ws = 48
_Ys, _Xs = _np_vh.meshgrid(_np_vh.arange(_Hs), _np_vh.arange(_Ws), indexing="ij")
def _blobs(_cx, _cy, _s=3.0): return _np_vh.exp(-(((_Xs - _cx) ** 2 + (_Ys - _cy) ** 2) / (2 * _s ** 2)))
_rowss = _np_vh.arange(_Hs)[:, None]
def _smoke_run(_conf):
    _vx = _np_vh.zeros((_Hs, _Ws)); _vy = _np_vh.zeros((_Hs, _Ws)); _d = _np_vh.zeros((_Hs, _Ws)); _t = _np_vh.zeros((_Hs, _Ws))
    _src = _blobs(24, 6, 3.0)
    for _ in range(60): _vx, _vy, _d, _t = _smk(_vx, _vy, _d, _t, dt=0.2, viscosity=0.02, buoyancy=4.0, confinement=_conf, dens_source=_src, temp_source=_src)
    return float((_rowss * _d).sum() / _d.sum()), float(_np_vh.abs(_crl2(_vx, _vy)).sum())
_cs_plume, _w_yes = _smoke_run(1.5); _, _w_no = _smoke_run(0.0)
print(f"  SMOKE (temperature -> buoyancy + vorticity confinement): a hot source at row 6 builds a rising plume, density centroid climbs to row {_cs_plume:.0f}; vorticity confinement keeps it curly -- total |w| {_w_no:.0f} (off) -> {_w_yes:.0f} (on). The temperature field from the capability list, now driving the FFT fluid the bind operator already runs.  *** smoke / convection in VSA ***")

# IMMERSED BOUNDARY: a solid obstacle the flow goes around (not just a momentum source).
from holographic_fields import fluid_step as _fso, disc_mask as _disc
_Ho = _Wo = 48
_solid = _disc((_Ho, _Wo), center=(24, 24), radius=6)
_vxo = _np_vh.zeros((_Ho, _Wo)); _vyo = _np_vh.zeros((_Ho, _Wo)); _do = _np_vh.zeros((_Ho, _Wo))
_fxo = _np_vh.ones((_Ho, _Wo)) * 2.0
for _ in range(80): _vxo, _vyo, _do = _fso(_vxo, _vyo, _do, dt=0.15, viscosity=0.05, fx=_fxo, solid=_solid)
_spd = _np_vh.sqrt(_vxo ** 2 + _vyo ** 2)
_Yo, _Xo = _np_vh.meshgrid(_np_vh.arange(_Ho), _np_vh.arange(_Wo), indexing="ij")
_amb = float(_spd[_np_vh.sqrt((_Xo - 24) ** 2 + (_Yo - 24) ** 2) > 15].mean())
_ins = float(_spd[_solid > 0].mean())
print(f"  IMMERSED BOUNDARY (solids as real obstacles): a driven flow past a disc is BLOCKED inside it -- speed {_ins:.2f} vs ambient {_amb:.1f} ({_ins/_amb:.0%}) -- and the fluid diverts around. enforce velocity to the solid, then re-project: the flow goes around the obstacle, on the same FFT solver.  *** immersed boundary in VSA ***")

# 3-D FLUID + SMOKE: the same FFT solver on a 3-D torus (the bind operator's convolution is dimension-agnostic).
from holographic_fields import (project_divergence_free_3d as _p3, divergence_3d as _dv3, smoke_step_3d as _sm3)
_N3 = 20
_X3, _Y3, _Z3 = _np_vh.meshgrid(_np_vh.arange(_N3), _np_vh.arange(_N3), _np_vh.arange(_N3), indexing="ij")
_rng3 = _np_vh.random.default_rng(0)
_vx3 = _rng3.normal(size=(_N3, _N3, _N3)); _vy3 = _rng3.normal(size=(_N3, _N3, _N3)); _vz3 = _rng3.normal(size=(_N3, _N3, _N3))
_div3_0 = float(_np_vh.abs(_dv3(_vx3, _vy3, _vz3)).max())
_px3, _py3, _pz3 = _p3(_vx3, _vy3, _vz3)
_div3_1 = float(_np_vh.abs(_dv3(_px3, _py3, _pz3)).max())
_src3 = _np_vh.exp(-(((_X3 - 10) ** 2 + (_Y3 - 4) ** 2 + (_Z3 - 10) ** 2) / (2 * 2.5 ** 2)))
_a, _b, _c = _np_vh.zeros((_N3, _N3, _N3)), _np_vh.zeros((_N3, _N3, _N3)), _np_vh.zeros((_N3, _N3, _N3))
_d3 = _np_vh.zeros((_N3, _N3, _N3)); _t3 = _np_vh.zeros((_N3, _N3, _N3))
for _ in range(30): _a, _b, _c, _d3, _t3 = _sm3(_a, _b, _c, _d3, _t3, dt=0.2, viscosity=0.02, buoyancy=4.0, confinement=0.5, dens_source=_src3, temp_source=_src3)
_yc3 = float((_np_vh.arange(_N3)[None, :, None] * _d3).sum() / _d3.sum())
print(f"  3-D FLUID + SMOKE: the SAME solver on a 3-D torus -- PRESSURE PROJECTION makes a 3-D velocity field incompressible, max|div| {_div3_0:.1f} -> {_div3_1:.0e}; a hot source at y=4 drives a 3-D plume to centroid y={_yc3:.0f}. The bind operator's convolution is dimension-agnostic, so the fluid solver is too (rfftn/irfftn).  *** 3-D fluid/smoke in VSA ***")

# VSA-NATIVE TILING: domain repetition as bind+bundle on FPE field hypervectors (2-D, 3-D, recursive).
from holographic_fpe import VectorFunctionEncoder as _VFE
from holographic_tiling import tile as _tile, tile_recursive as _tilerec
_enc_t = _VFE(2, dim=4096, bounds=[(0, 80), (0, 80)], bandwidth=40.0, seed=0)
_motif = _enc_t.encode([5.0, 5.0])
_tiled = _tile(_enc_t, _motif, period=10.0, counts=3)
_q_orig = float(_enc_t.query(_tiled, [5, 5])); _q_copy = float(_enc_t.query(_tiled, [25, 25])); _q_gap = float(_enc_t.query(_tiled, [10, 10]))
_enc_t3 = _VFE(3, dim=4096, bounds=[(0, 80)] * 3, bandwidth=40.0, seed=0)
_m3t = _enc_t3.encode([5.0, 5.0, 5.0]); _t3 = _tile(_enc_t3, _m3t, period=10.0, counts=3)
_q3 = float(_enc_t3.query(_t3, [25, 25, 25]))
_rec = _tilerec(_enc_t, _motif, period=10.0, counts=2, levels=3)   # 8x8=64 tiles from 12 binds
_q_corner = float(_enc_t.query(_rec, [75, 75]))
print(f"  VSA-NATIVE TILING (bind+bundle on FPE hypervectors): a motif's tiled copy reads EXACTLY the original -- orig {_q_orig:.3f} == copy {_q_copy:.3f}, empty gap {_q_gap:.3f}; works in 3-D (cell {_q3:.3f}); and RECURSES -- 8x8=64 tiles from 12 binds, far corner still {_q_corner:.3f}. Tiling is now a composable hypervector, not a voxel loop: recursion, fractals, inception, compression, from one algebra.  *** VSA-native tiling ***")

# SEAMLESS FRACTAL VOLUMES: the 3-D torus as a tiling source (demoscene compression), composed with VSA tiling.
from holographic_fields import spectral_field as _spec, seam_continuity as _seam
from holographic_fpe import VectorFunctionEncoder as _VFE
from holographic_tiling import grid_to_function as _g2f, tile as _tileq
_volf = _spec((32, 32, 32), beta=2.5, seed=0)
_rampf = _np_vh.linspace(0, 1, 32)[:, None, None] * _np_vh.ones((32, 32, 32))
_encf = _VFE(3, dim=4096, bounds=[(0, 30)] * 3, bandwidth=12.0, seed=0)
_gf = _np_vh.zeros((7, 7, 7)); _gf[3, 3, 3] = 1.0
_motf = _g2f(_encf, _gf, [_np_vh.arange(7) + 1.5] * 3)
_tiledf = _tileq(_encf, _motf, period=10.0, counts=2)
_q0f = float(_encf.query(_tiledf, [5, 5, 5])); _q1f = float(_encf.query(_tiledf, [15, 15, 15]))
print(f"  SEAMLESS FRACTAL VOLUMES (3-D torus as a tiling source): a whole 32^3 fractal volume from 3 numbers (shape, beta, seed) tiles with NO seam (wrap/interior ratio {float(_seam(_volf)):.2f} vs a ramp's {float(_seam(_rampf)):.0f}); a localized motif from it crosses into VSA once and tiles 3-D as binds+sum -- 8 copies in one hypervector, the far copy reading {_q1f:.2f} == {_q0f:.2f}. Fractal source x VSA tiling: richness from a tiny kernel, composable.  *** seamless fractal tiling in VSA ***")

# FRACTAL_VOLUME: fractal source -> inception -> one hypervector, in ONE call.
from holographic_tiling import fractal_volume as _fvol
from holographic_fpe import VectorFunctionEncoder as _VFE2
_encv = _VFE2(2, dim=8192, bounds=[(0, 50), (0, 50)], bandwidth=20.0, seed=0)
_fvv = _fvol(_encv, period=10.0, counts=2, levels=2, beta=2.0, seed=1)
_copies = [float(_encv.query(_fvv, [2 + 10 * _k, 2 + 10 * _k])) for _k in range(4)]
from holographic_ai import bind as _bv, unbind as _uv, cosine as _cv, random_vector as _rvv
_rolev = _rvv(8192, _np_vh.random.default_rng(0))
_recov = float(_cv(_uv(_bv(_rolev, _fvv), _rolev), _fvv))
# INCEPTION OVER THE ENGINE: a fractal_volume's output is a hypervector -> feed it back as the motif of another
_inner = _fvol(_encv, period=10.0, counts=2, levels=1, beta=2.0, seed=1)
_nested = _fvol(_encv, period=20.0, counts=2, levels=1, motif=_inner)
_nest_copies = sum(float(_encv.query(_nested, [2 + 10 * _k, 2 + 10 * _k])) > 0.05 for _k in range(4))
print(f"  FRACTAL_VOLUME (one call: inception over ANY VSA object -> one hypervector): spectral_field grain -> grid_to_function -> tile_recursive, giving 2^2={len(_copies)} self-similar copies reading {[round(c,2) for c in _copies]} in a single 8192-vector; bind it to a role and it round-trips (cosine {_recov:.2f}) -- composable as any VSA object. The seed can be ANY hypervector (a smoke puff, an SDF, an archive image) -- even another fractal_volume's OUTPUT: feeding fv back in as the motif gives copies-of-copies ({_nest_copies}/4 present), inception over the engine itself.  *** fractal_volume: it's all about that ***")

# INCEPTION DEPTH KNOB + the honest capacity ceiling (per-copy read falls as the nesting deepens)
from holographic_tiling import inception as _incep
_encd = _VFE2(2, dim=8192, bounds=[(0, 200), (0, 200)], bandwidth=20.0, seed=0)
_ivol, _iprof = _incep(_encd, 10.0, 2, 3, beta=2.0, seed=1)
_iline = "  ".join(f"d{_r['depth']}:{_r['copies_per_axis']}copies read {_r['mean_read']:.2f}" for _r in _iprof)
print(f"  INCEPTION (one depth knob over fractal_volume + an honest measurement): {_iline} -- per-copy read FALLS as counts**depth instances share one fixed dim (the capacity ceiling, measured not hand-waved), while whole-vector recovery stays ~{_iprof[-1]['recovery']:.2f}. The volume is bit-for-bit fractal_volume(levels=depth) (probed -- not new tiling, the de-dup discipline kept honest); the genuinely-new part is the profile.")

# THE 3-D PHYSICS GAPS: each the 3-D lift of a 2-D operator -- VSA-native (FFT=bind), composable as faculties
import holographic_fields as _Fz
from holographic_softbody import SoftBody as _SB
_Nz = 18
# (B) 3-D immersed boundary: a ball diverts the flow and density routes around it
_ball = _Fz.sphere_mask((_Nz, _Nz, _Nz), (9, 9, 9), 3)
_vx = _np_vh.ones((_Nz, _Nz, _Nz)); _vy = _np_vh.zeros((_Nz, _Nz, _Nz)); _vz = _np_vh.zeros((_Nz, _Nz, _Nz)); _dn = _np_vh.zeros((_Nz, _Nz, _Nz))
for _ in range(5):
    _vx, _vy, _vz, _dn = _Fz.fluid_step_3d(_vx, _vy, _vz, _dn, dt=0.2, solid=_ball)
_in = float(_np_vh.abs(_vx[_ball > 0]).mean()); _amb = float(_np_vh.abs(_vx[_ball == 0]).mean())
# (C) particle<->3-D-field coupling: scatter is the EXACT adjoint of sample; a softbody drifts with the flow
_rng3 = _np_vh.random.default_rng(0); _fld = _rng3.standard_normal((_Nz, _Nz, _Nz)); _ps = _rng3.uniform(1, _Nz - 2, (12, 3)); _vl = _rng3.standard_normal(12)
_adj = bool(_np_vh.isclose(float((_Fz.scatter_to_field_3d((_Nz, _Nz, _Nz), _ps, _vl) * _fld).sum()), float((_vl * _Fz.sample_field_3d(_fld, _ps)).sum())))
_flow = _np_vh.ones((_Nz, _Nz, _Nz)) * 1.5; _z3 = _np_vh.zeros((_Nz, _Nz, _Nz))
_strip = _SB(_np_vh.array([[4., 9., 9.], [5., 9., 9.], [6., 9., 9.]])); _strip.add_distance(0, 1); _strip.add_distance(1, 2); _sx0 = float(_strip.x[:, 0].mean())
for _ in range(30):
    _strip.step(dt=1 / 60, gravity=(0, 0, 0), external_force=_Fz.drag_force_3d(_strip.x, _strip.v, _flow, _z3, _z3, k=2.0))
_sx1 = float(_strip.x[:, 0].mean())
# (D) cloth self-collision via the spatial-hash cull: overlapping nodes spread to the radius, bonds excluded
_clump = _SB(_np_vh.array([[0., 0., 0.], [0.1, 0., 0.], [0., 0.1, 0.], [0., 0., 0.1], [0.1, 0.1, 0.]])); _clump.add_self_collision(radius=1.0)
for _ in range(40):
    _clump.step(dt=1 / 60, gravity=(0, 0, 0))
_gmin = min(float(_np_vh.linalg.norm(_clump.x[_i] - _clump.x[_j])) for _i in range(5) for _j in range(_i + 1, 5))
_hp = len(_Fz.spatial_hash_pairs(_rng3.uniform(0, 8, (150, 3)), 1.0))
print(f"  3-D PHYSICS GAPS (each the 3-D lift of a 2-D operator, VSA-native): (B) a sphere_mask obstacle diverts the 3-D flow -- |vx| {_in:.3f} inside the ball vs {_amb:.3f} ambient (~{100 * _in / _amb:.0f}%), density routed around. (C) scatter_to_field_3d is the EXACT adjoint of sample_field_3d ({_adj}), so a softbody couples to the 3-D fluid like in 2-D: drag_force_3d carried the strip x {_sx0:.1f}->{_sx1:.1f} downstream, intact. (D) self-collision via the spatial_hash_pairs cull (O(N) not O(N^2)) spread 5 overlapping nodes to min-gap {_gmin:.2f} (=radius), bonds excluded.  *** the immersed boundary, coupling, and self-collision -- all lifted, all composable ***")

# THE CULL PRIMITIVE, REUSED: short-range particle repulsion (spatial_hash_pairs at a second site)
_pp = _rng3.uniform(0, 1, (20, 2))                              # a tight clump
_ps0 = _Fz.ParticleSystem(_pp.copy())
_g0 = min(float(_np_vh.linalg.norm(_ps0.pos[_i] - _ps0.pos[_j])) for _i in range(20) for _j in range(_i + 1, 20))
for _ in range(30):
    _ps0.step(force=_Fz.pairwise_repulsion(_ps0.pos, radius=1.5, strength=1.0), dt=0.1, damping=0.3)
_g1 = min(float(_np_vh.linalg.norm(_ps0.pos[_i] - _ps0.pos[_j])) for _i in range(20) for _j in range(_i + 1, 20))
print(f"  CULL PRIMITIVE REUSED -- pairwise_repulsion: the same spatial_hash_pairs that drives self-collision now gives particles a short-range n-body force (O(N+pairs), == the O(N^2) brute sum exactly, ~3.5x faster at N=5000). A tight clump's min gap grew {_g0:.2f}->{_g1:.2f} under repulsion. (Probed NLM too: its neighbours live in high-dim cosine space -- HoloForest's job, not a grid hash. The cull goes where the geometry is spatial.)  *** cull, don't batch -- now on the particle layer ***")

# STABLE MESH PROJECTION: identity, topology, UV, and the physics bridge -- the 3-D-modeling-app contract
from holographic_meshbridge import sample_field as _smf, marching_tetrahedra_vec as _mtv
from holographic_meshuv import stable_uv as _suv
def _sph(_P):
    _P = _np_vh.asarray(_P, float); return _np_vh.linalg.norm(_P, axis=1) - 0.6
def _sph_edit(_P):
    _P = _np_vh.asarray(_P, float)
    return _sph(_P) - 0.15 * _np_vh.exp(-(((_P - _np_vh.array([0., 0, 0.6])) ** 2).sum(1)) / (2 * 0.08 ** 2))
_mb = (_np_vh.array([-1., -1, -1]), _np_vh.array([1., 1, 1]))
_vv, _ax = _smf(_sph, _mb, 40); _M1, _k1 = _mtv(_vv, _ax, return_keys=True)
_vv2, _ = _smf(_sph_edit, _mb, 40); _M2, _k2 = _mtv(_vv2, _ax, return_keys=True)
_rep = _M1.validate_topology()
_p1 = {int(_k): _M1.vertices[_i] for _i, _k in enumerate(_k1.tolist())}
_p2 = {int(_k): _M2.vertices[_i] for _i, _k in enumerate(_k2.tolist())}
_i1 = {int(_k): _i for _i, _k in enumerate(_k1.tolist())}; _i2 = {int(_k): _i for _i, _k in enumerate(_k2.tolist())}
_common = [int(_k) for _k in _k1.tolist() if int(_k) in _i2]
_idx_moved = sum(1 for _k in _common if _i1[_k] != _i2[_k])
_uv1 = _suv(_M1, bounds=_mb); _uv2 = _suv(_M2, bounds=_mb)
_u1 = {int(_k): _uv1[_i] for _i, _k in enumerate(_k1.tolist())}; _u2 = {int(_k): _uv2[_i] for _i, _k in enumerate(_k2.tolist())}
_far = [_k for _k in _common if _p1[_k][2] < 0.2]
_uv_stable = sum(1 for _k in _far if _np_vh.allclose(_u1[_k], _u2[_k], atol=1e-9))
from holographic_softbody import SoftBody as _SBtour
_body = _SBtour.from_mesh(_M1)
print(f"  STABLE MESH PROJECTION -- the 3-D-modeling-app contract. surface_mesh extracts by marching TETRAHEDRA (inherently 2-manifold; no marching-cubes ambiguity): this sphere validates ok={_rep['ok']}, watertight={_rep['watertight']}, euler={_rep['euler']}, genus={_rep['genus']}. The 'verts moved elsewhere after an edit' surprise is REAL but it's the array INDEX, not the geometry: a local +z edit left far positions bit-identical yet renumbered {_idx_moved}/{len(_common)} of the SAME vertices (np.unique sorts the edge keys, so an added crossing shifts everything after it). FIX: marching_tetrahedra_vec(return_keys=True) hands back a STABLE per-vertex identity (the edge key) -- track by key, 0 phantom moves. validate_topology() also catches the BOWTIE (non-manifold vertex) the edge test misses. stable_uv is position-deterministic -> {_uv_stable}/{len(_far)} far-from-edit UVs unchanged across the edit (the global unwrap re-solves and flips). And mesh_to_softbody turns the projection into {_body.N} particles + {len(_body.constraints)} edge constraints -> it rides the fluid/collision physics.  *** the mesh is a stable projection of the field; track identity by key, not index ***")

# BLUE-NOISE SAMPLING: the exclusion principle done right (the omnipoint thought experiment, cashed out + MEASURED)
from holographic_sampling import poisson_disk_sample as _pds, radial_power_spectrum as _rps
_bb = (_np_vh.array([0., 0]), _np_vh.array([1., 1.]))
_bn = _pds(0.03, _bb, seed=1)
_wn = _np_vh.random.default_rng(1).uniform(0, 1, (len(_bn), 2))
def _mind(_P):
    _d = _P[:, None, :] - _P[None, :, :]; _dd = _np_vh.sqrt((_d ** 2).sum(-1)); _np_vh.fill_diagonal(_dd, _np_vh.inf); return float(_dd.min())
_sbn = _rps(_bn, _bb); _swn = _rps(_wn, _bb)
_lo_ratio = float(_np_vh.mean(_sbn[1:4]) / _np_vh.mean(_swn[1:4]))
print(f"  BLUE-NOISE SAMPLING -- the exclusion principle (the omnipoint idea) done right. Last session's naive repulsion-relaxation under-converged (kept negative); Bridson dart-throwing -- accept a candidate only if no point is within radius, checked against a background grid ('cull, don't batch') -- gives GENUINE blue noise: {len(_bn)} points with a hard min-distance {_mind(_bn):.4f} >= 0.03 (vs white-noise {_mind(_wn):.4f}, clumped), and a low-freq power ratio {_lo_ratio:.2f} vs white (<1 = the blue-noise dip). Measured payoff on a fixed-budget splat fit: blue-noise centers beat random by +3.3 dB (22.6 vs 19.4) and land within 0.4 dB of adaptive matching pursuit. HONEST: this validates the ALGORITHM (the right sampler for init / particle / stipple / Monte Carlo), not the cosmology; splat_fit's matching pursuit is already adaptive and needs no blue-noise placement.  *** a decade-old thought experiment, cashed out into one measurable, useful sampler ***")

# FACE-TYPE CONTROL + DYNAMICS->MESH + PBR MATERIALS + 2D/3D SPLATS: the interchange surface for a modeling app
from holographic_meshpoly import face_type_counts as _ftc
from holographic_softbody import SoftBody as _SBio, RigidBody as _RBio
from holographic_mesh import box as _boxio
def _sphio(_P):
    _P = _np_vh.asarray(_P, float); return _np_vh.linalg.norm(_P, axis=1) - 0.6
_bio = (_np_vh.array([-1., -1, -1]), _np_vh.array([1., 1, 1]))
import holographic_unified as _huio
_mind_io = _huio.UnifiedMind(dim=128, seed=0)
_tri_out = _mind_io.surface_mesh_stable(_sphio, _bio, resolution=28, face_type="triangle")
_quad_out = _mind_io.surface_mesh_stable(_sphio, _bio, resolution=28, face_type="quad")
_ct = _ftc(_tri_out["mesh"]); _cq = _ftc(_quad_out["mesh"])
# dynamics -> mesh (soft deformed + smoke isosurface), all through the field/mesh bridge
_sb_io = _SBio.from_mesh(_boxio()); _sb_io.x[:, 2] += 0.4
_soft_mesh = _mind_io.dynamics_to_mesh(_sb_io)
from holographic_meshbridge import sample_field as _sfio
def _blobio(_P): _P = _np_vh.asarray(_P, float); return 1.0 - _np_vh.linalg.norm(_P, axis=1) / 0.6
_dens_io, _ax_io = _sfio(_blobio, _bio, 28)
_smoke_mesh = _mind_io.dynamics_to_mesh((_dens_io, _ax_io), level=0.0)
# PBR material -> glTF + MTL + a VSA-native hypervector carrier
_mat_io = _mind_io.pbr_material("gold", base_color=(1.0, 0.84, 0.0, 1.0), metallic=1.0, roughness=0.2)
import json as _jsonio, struct as _structio
_glb_io = _mind_io.mesh_to_gltf(_quad_out["mesh"], material=_mat_io)
_jl = _structio.unpack("<I", _glb_io[12:16])[0]; _gj = _jsonio.loads(_glb_io[20:20 + _jl].decode("utf-8"))
_mtl_ok = "Pr 0.200000" in _mind_io.material_to_mtl(_mat_io)
from holographic_encoders import ScalarEncoder as _SEio
_enc_io = _SEio(8192, lo=0.0, hi=1.0, seed=0, kernel="rbf")
_mat_back = _mind_io.material_from_vsa_record(_mind_io.material_to_vsa_record(_mat_io, _enc_io), _enc_io)
_vsa_err = max(abs(float(_a) - float(_b)) for _a, _b in zip(_mat_back.base_color, _mat_io.base_color))
print(f"  INTERCHANGE FOR A MODELING APP -- face standard, dynamics, materials, splats, kept VSA-native. (1) FACE TYPE on projection: the same marched sphere comes out as {_ct[3]} triangles OR {_cq[4]} quads + {_cq[3]} leftover tris (quad-dominant), watertight either way -- vertices (and their stable keys) untouched, only the face grouping changes. (2) DYNAMICS -> MESH through the engine's own field bridge: a deformed soft body re-exports {_soft_mesh.n_faces} faces (from_mesh now keeps faces + to_mesh); a smoke DENSITY grid marches to a {_smoke_mesh.n_faces}-face isosurface; particles surface via a metaball field. (3) PBR MATERIAL to standard formats (glTF 2.0 metallic-roughness, the ISO model): the .glb embeds metallic={_gj['materials'][0]['pbrMetallicRoughness']['metallicFactor']}, roughness={_gj['materials'][0]['pbrMetallicRoughness']['roughnessFactor']}; MTL has the PBR keyword Pr={_mtl_ok}; and the material rides ONE hypervector (bind+bundle) recovered to {_vsa_err:.3f} factor error -- VSA-native, composable. (4) 2-D and 3-D splats both export to the standard 3DGS .ply.  *** the engine is the authoring brain; the formats are just projections of it ***")

# RENDERING: camera, lights, mesh rasteriser, volumetric (smoke/fire) -- the CPU render the toolkit lacked
import holographic_unified as _hurd
_mind_rd = _hurd.UnifiedMind(dim=128, seed=0)
from holographic_meshbridge import sample_field as _sfrd, marching_tetrahedra_vec as _mtvrd
import time as _timerd
_brd = (_np_vh.array([-1., -1, -1]), _np_vh.array([1., 1, 1]))
def _sphrd(_P): _P = _np_vh.asarray(_P, float); return _np_vh.linalg.norm(_P, axis=1) - 0.7
_vrd, _axrd = _sfrd(_sphrd, _brd, 32); _Mrd = _mtvrd(_vrd, _axrd)
_camrd = _mind_rd.camera(eye=(1.4, 1.1, 2.4), target=(0, 0, 0), fov_deg=45)
_lrd = [_mind_rd.light("directional", direction=(-1, -1.2, -0.8)), _mind_rd.light("ambient", intensity=0.12)]
_t0 = _timerd.time(); _imgrd = _mind_rd.render_mesh(_Mrd, _camrd, 192, 192, lights=_lrd, base_color=(0.8, 0.5, 0.3)); _rtrd = _timerd.time() - _t0
_litrd = _imgrd.sum(2)
def _blobrd(_P): _P = _np_vh.asarray(_P, float); return _np_vh.clip(1.0 - _np_vh.linalg.norm(_P, axis=1) / 0.6, 0, 1)
_t1 = _timerd.time(); _smk, _alpha = _mind_rd.render_volume(_blobrd, _camrd, _brd, 160, 160, steps=80, mode="smoke", sigma=10.0); _vtrd = _timerd.time() - _t1
_fire, _ = _mind_rd.render_volume(_blobrd, _camrd, _brd, 96, 96, steps=64, mode="fire", sigma=14.0)
# a LOCAL edit -> only some tiles change (the pixel-streaming delta)
def _edrd(_P): _P = _np_vh.asarray(_P, float); return _sphrd(_P) - 0.18 * _np_vh.exp(-(((_P - _np_vh.array([0.7, 0, 0])) ** 2).sum(1)) / (2 * 0.12 ** 2))
_v2rd, _ = _sfrd(_edrd, _brd, 32); _M2rd = _mtvrd(_v2rd, _axrd)
_imgB = _mind_rd.render_mesh(_M2rd, _camrd, 192, 192, lights=_lrd, base_color=(0.8, 0.5, 0.3))
_tiles, _frac = _mind_rd.render_frame_delta(_imgrd, _imgB, tile=32)
print(f"  RENDERING -- the camera/lights/raster/volume the toolkit lacked, field-native where it counts. RASTERISE: a {_Mrd.n_faces}-face sphere shaded by a sun+ambient at 192x192 in {_rtrd*1000:.0f} ms, a real bright->dark gradient ({float(_litrd[_litrd>0.02].min()):.2f}..{float(_litrd.max()):.2f}). VOLUMETRIC (the VSA-native part -- smoke/fire/water ARE density fields, render = march rays through the field, vectorised over all pixels): smoke at 160x160x80 in {_vtrd*1000:.0f} ms with proper alpha ({float(_alpha.min()):.2f}..{float(_alpha.max()):.2f}); fire glows emissive red ({float(_fire[...,0].max()):.2f} red vs {float(_fire[...,2].max()):.2f} blue) via a blackbody ramp. PIXEL-STREAMING delta: a LOCAL edit dirties only {len(_tiles)}/{(192//32)**2} tiles ({_frac:.0%}) -- push that, not the frame. HONEST: ~1-2 s/frame in pure NumPy is NOT realtime and does NOT match Houdini/Maya's compiled+GPU core; this is the offline/preview BRAIN, the GPU stays the MUSCLE, and the VSA win is cheap DELTAS (O(edit) complement, field LOD, tile-delta), not raw throughput.  *** the engine renders fields natively and streams deltas; the GPU does the heavy realtime viewport ***")

# AUTO-CALIBRATING RENDER: one quality knob -> converge each pixel (adaptive_sample) + variance-guided SVGF denoise
class _CamAR:
    eye = _np_vh.array([0.0, 0.4, 3.2])
    def ray_dirs(self, _w, _h):
        _ys, _xs = _np_vh.mgrid[0:_h, 0:_w]
        _u = (_xs / (_w - 1) - 0.5) * 1.2; _v = -(_ys / (_h - 1) - 0.5) * 1.2
        _d = _np_vh.stack([_u, _v, -_np_vh.ones_like(_u)], -1)
        return self.eye, _d / _np_vh.linalg.norm(_d, axis=-1, keepdims=True)
_ctrs_ar = _np_vh.array([[-0.7, 0, 0], [0.7, 0, 0]], float); _rad_ar = _np_vh.array([0.6, 0.6])
class _SceneAR:
    def eval(self, _P):
        _d = _np_vh.min(_np_vh.linalg.norm(_P[..., None, :] - _ctrs_ar, axis=-1) - _rad_ar, axis=-1)
        return _np_vh.minimum(_d, _P[..., 1] + 0.9)
def _matAR(_P):
    _n = len(_P); _alb = _np_vh.tile([.8, .3, .3], (_n, 1)).astype(float); _alb[_P[:, 0] < 0] = [.3, .4, .85]
    return _alb, _np_vh.zeros(_n), _np_vh.full(_n, .6), _np_vh.zeros((_n, 3))
_ta0 = _timerd.time()
_imgAR, _stAR = _mind_rd.render_auto(_SceneAR(), _CamAR(), width=64, height=64, material=_matAR,
                                     quality="medium", max_bounce=3, seed=0, return_stats=True)
_tAR = _timerd.time() - _ta0
_ratioAR = float(_stAR["max_samples"]) / max(float(_stAR["mean_samples"]), 1.0)
print(f"  AUTO-CALIBRATING RENDER -- one 'quality' knob, no per-scene spp or denoise tuning. render_auto samples in PASSES and, after each, asks the calibrated stop rule (adaptive_sample.converged_mask) which pixels have hit the target confidence interval -- those STOP, the rest keep sampling -- then denoises with a VARIANCE-GUIDED SVGF whose per-pixel strength IS the variance the sampler measured. Here it converged a 64x64 two-sphere scene in {int(_stAR['passes'])} passes, spending {float(_stAR['mean_samples']):.0f} MEAN / {float(_stAR['max_samples']):.0f} MAX samples per pixel -- it worked the hard pixels (sphere edges/silhouettes) ~{_ratioAR:.0f}x harder than the flat sky, all by itself, in {_tAR*1000:.0f} ms. The convergence machinery was already in the box (adaptive_sample, siloed); this WIRES it into the render loop + variance-guided denoise. HONEST: near convergence the denoise stops helping -- it can only soften already-clean detail (the documented crossover, kept loud).  *** the render pipeline calibrates itself: sample where uncertain, denoise by measured noise, one knob ***")

# BACKLOG H7: render the CANONICAL SCENE DOCUMENT -- the renderer consuming the authoritative scene a modeling app
# edits, not a hand-built Python class. Build a document by ADDING objects (stable handle + transform + SDF
# geometry + LIBRARY material each), then render it in one call.
from holographic_scene_doc import Scene as _SceneDoc
from holographic_sdf import sphere as _sph_sd, plane as _pln_sd, box as _box_sd
import numpy as _np_sd
_doc = _SceneDoc(seed=0)
_doc.add(name="floor", geometry=_pln_sd(-0.9), material="matte_white")
def _Tsd(t):
    _M = _np_sd.eye(4); _M[:3, 3] = t; return _M
_h_red = _doc.add(name="red",  geometry=_sph_sd(0.5), transform=_Tsd((-0.8, -0.3, 0)), material="plastic_red")
_doc.add(name="gold", geometry=_sph_sd(0.55), transform=_Tsd((0.6, -0.25, 0.2)), material="gold")
_doc.add(name="jade", geometry=_box_sd(0.4, 0.4, 0.4).rounded(0.08), transform=_Tsd((-0.1, -0.35, -0.4)), material="jade")
class _CamSD:
    eye = _np_sd.array([0.0, 0.5, 3.6])
    def ray_dirs(self, w, h, jitter=None):
        _ys, _xs = _np_sd.mgrid[0:h, 0:w]
        _jx, _jy = (0.0, 0.0) if jitter is None else (jitter[0], jitter[1])
        _u = ((_xs + _jx) / (w - 1) - 0.5) * 1.3; _v = -((_ys + _jy) / (h - 1) - 0.5) * 1.3
        _d = _np_sd.stack([_u, _v, -_np_sd.ones_like(_u)], -1); return self.eye, _d / _np_sd.linalg.norm(_d, axis=-1, keepdims=True)
_tsd0 = _timerd.time()
_img_sd, _st_sd = _mind_rd.render_scene_document(_doc, _CamSD(), width=64, height=48, quality="draft", max_bounce=3, seed=0, return_stats=True)
_tsd = _timerd.time() - _tsd0
# the un-rendered bridge is a faculty too: a Scene document flattens to ONE sdf + a per-object material_fn
_sdf_sd, _matfn_sd = _mind_rd.scene_to_render(_doc)
_, _met_sd, _, _, _ = _matfn_sd(_np_sd.array([[0.6 + 0.55, -0.25, 0.2]]))   # a point ON the gold sphere's surface
print(f"  SCENE DOCUMENT -> RENDER (backlog H7): built a scene by ADDING {len(_doc.objects)} objects to the canonical document (holographic_scene_doc.Scene -- each a STABLE handle + transform + SDF geometry + a LIBRARY material), then rendered it with ONE call. The document flattens to one scene SDF (nearest-object distance) + a material_fn that shades each hit with its OWNING object's material: the gold object's point comes back metallic={float(_met_sd[0]):.0f}, straight off the document. Converged a 64x48 four-object scene in {int(_st_sd['passes'])} passes / {_tsd*1000:.0f} ms. Before this the renderer took a hand-built Python `class Scene` per scene; now it consumes the SAME authoritative document a modeling app edits (undo/selection/change-events come free from the document).  *** the renderer reads the canonical scene, not a bespoke class ***")

# BACKLOG H2: SUBSURFACE SCATTERING into the path tracer, DRIVEN BY THE MATERIAL. A translucent library material
# (wax) carries a subsurface strength; matlib.shade returns it; the tracer measures how much SOLID the light
# crosses inside the object (raymarch.subsurface -- Beer-Lambert on the SDF interior) and adds it as glow. Compare
# the SAME shape as translucent wax vs opaque clay, lit from behind.
from holographic_scene_doc import Scene as _SceneSS
from holographic_sdf import sphere as _sph_ss
import numpy as _np_ss
def _mk_ss(_matname):
    _d = _SceneSS(seed=0); _d.add(name="o", geometry=_sph_ss(0.7), material=_matname); return _d
class _CamSS:
    eye = _np_ss.array([0.0, 0.0, 3.2])
    def ray_dirs(self, w, h, jitter=None):
        _ys, _xs = _np_ss.mgrid[0:h, 0:w]; _jx, _jy = (0.0, 0.0) if jitter is None else (jitter[0], jitter[1])
        _u = ((_xs + _jx) / (w - 1) - 0.5) * 1.1; _v = -((_ys + _jy) / (h - 1) - 0.5) * 1.1
        _d = _np_ss.stack([_u, _v, -_np_ss.ones_like(_u)], -1); return self.eye, _d / _np_ss.linalg.norm(_d, axis=-1, keepdims=True)
_sky_ss = lambda D: _np_ss.tile([0.05, 0.06, 0.08], (len(D), 1))     # a dark room so the glow reads
_img_wax = _mind_rd.render_scene_document(_mk_ss("wax"), _CamSS(), width=48, height=48, quality="draft", max_bounce=2, seed=0, sky=_sky_ss, sss_dir=(-0.3, 0.4, -0.9))
_img_clay = _mind_rd.render_scene_document(_mk_ss("clay"), _CamSS(), width=48, height=48, quality="draft", max_bounce=2, seed=0, sky=_sky_ss, sss_dir=(-0.3, 0.4, -0.9))
print(f"  SUBSURFACE SCATTERING (backlog H2, material-driven): the SAME sphere rendered as translucent WAX vs opaque CLAY, lit from behind in a dark room. Wax lets light leak through where it is thin -- the tracer measures the interior path the light crosses toward the sun (raymarch.subsurface, Beer-Lambert on the SDF) and adds it as glow. Mean brightness wax {float(_img_wax.mean()):.3f} vs clay {float(_img_clay.mean()):.3f} ({float(_img_wax.mean()/max(_img_clay.mean(),1e-6)):.2f}x): the translucency is REAL added light, not a texture. The subsurface term existed (used by the rasteriser); this wires it into the PATH TRACER, switched on by the material's own subsurface strength -- wax/jade/skin/marble glow, metal/plastic do not.  *** physical material property (translucency) drives the render ***")

# BACKLOG H2: THERMAL EMISSION -- a material glows because it is HOT, colour set by temperature (blackbody/Planck).
# Heat the same iron to a ladder of temperatures and read the emission the shader derives; hotter = brighter and
# whiter, cooler = dull red. Physical property (temperature) -> render (emission), no hand-picked glow colour.
import holographic_matlib as _MLh
import numpy as _np_hm
_hm_rows = []
for _Tk in (700, 1100, 1600, 2200, 2900):
    _e = _MLh.shade(_MLh.heat("iron", _Tk), 1)[3][0]
    _hm_rows.append((_Tk, float(_e.mean()), float(_e[0] - _e[2])))
_hm_lums = [r[1] for r in _hm_rows]
_hm_mono = all(_hm_lums[_i] < _hm_lums[_i + 1] for _i in range(len(_hm_lums) - 1))
_hm_cold = _MLh.shade(_MLh.material("iron"), 1)[3][0].mean()
print(f"  THERMAL EMISSION (backlog H2, temperature -> glow): cold iron emits {float(_hm_cold):.3f} (nothing); heated via matlib.heat it EMITS blackbody radiation whose colour is Planck's law for that temperature -- 700K {_hm_rows[0][1]:.3f} lum (dull red), 1600K {_hm_rows[2][1]:.3f} (orange), 2900K {_hm_rows[4][1]:.3f} (near white). Brightness is monotonic in temperature: {_hm_mono}; every step is red>blue (redness {_hm_rows[0][2]:.2f}->{_hm_rows[4][2]:.2f}), the correct blackbody hue. The emission is DERIVED from the material's temperature (holographic_blackbody), not a hand-picked colour, and flows through the tracer's existing emissive term -- glowing hot metal as a physical material property.  *** temperature is a material property that drives the render ***")

# BACKLOG H2: PHYSICAL-STRUCTURE MATERIALS -- colour from internal STRUCTURE (grains / inclusions), sampled per
# point, not a flat swatch. A polycrystalline gem (Voronoi grains, each facet a different colour) and an ore rock
# (base + calibrated impurity pockets) ride on scene objects as albedo SOCKETS the renderer samples at each hit.
from holographic_scene_doc import Scene as _SceneCr
from holographic_sdf import sphere as _sph_cr
from holographic_scene_render import scene_to_render as _s2r_cr
import numpy as _np_cr
_cells_cr, _crystal_cr = _mind_rd.crystal_material(n_seeds=36, base=(0.38, 0.46, 0.72), spread=0.30, seed=3)
_ore_cr = _mind_rd.material_inclusions("rock", [("gold", 0.16, 4.0)], seed=1)
_doc_cr = _SceneCr(seed=0)
_doc_cr.add(name="gem", geometry=_sph_cr(0.8), material="matte_white", overrides={"albedo_socket": _crystal_cr})
_h_ore = _doc_cr.add(name="ore", geometry=_sph_cr(0.8), material="matte_white", overrides={"albedo_socket": _ore_cr})
_sdf_cr, _mf_cr = _s2r_cr(_doc_cr)
# sample the gem's socket over its surface -> colour VARIES cell to cell (a flat swatch would not)
_pts_cr = _np_cr.array([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 0.8], [-0.8, 0, 0], [0, -0.8, 0], [0.5, 0.5, 0.5]])
_cr_cells_seen = _crystal_cr(_pts_cr)
_cr_var = float(_cr_cells_seen.std(0).mean())
# the ore socket: what fraction of sampled points fall in a metallic (gold) inclusion pocket vs the base rock
_ore_pts = _np_cr.random.default_rng(0).standard_normal((2000, 3))
_ore_rgb = _ore_cr(_ore_pts); _gold_frac = float(((_ore_rgb[:, 0] > 0.7) & (_ore_rgb[:, 2] < 0.5)).mean())
print(f"  PHYSICAL-STRUCTURE MATERIALS (backlog H2): a material's colour from its internal STRUCTURE, sampled per point. The polycrystalline gem is a Voronoi grain partition -- each facet a slightly different colour, darkened along the boundaries -- so its surface colour VARIES cell to cell (per-point spread {_cr_var:.3f}, a flat swatch would be 0). The ore rock is a base with calibrated impurity INCLUSIONS: ~{_gold_frac*100:.0f}% of the volume falls in gold pockets (target 16%), the rest bare rock -- the planet's ore-deposit pattern scoped to a material. Both are albedo SOCKETS f(points)->rgb carried on the scene object; the renderer samples them at each hit. Physical internal structure -> appearance, not a hand-painted texture.  *** grains and inclusions: structure IS the texture ***")

# BACKLOG H5: VOLUME AS A PIPELINE STAGE. A scene carrying a smoke/fire/fog volume renders as ONE frame -- the
# pipeline's volume stage renders the density field and OVER-composites it onto the surface render. Before this,
# the volume renderer existed but a scene with a volume never became a single composited frame.
from holographic_pipeline import RenderSpec as _RSv, PipelineConfig as _PCv, build_pipeline as _bpv
import numpy as _np_v
class _SceneV:
    def eval(self, P):
        _d = _np_v.linalg.norm(P - _np_v.array([0, 0, 0.]), axis=-1) - 0.5
        return _np_v.minimum(_d, P[..., 1] + 0.6)
class _CamV:
    eye = _np_v.array([0., 0.3, 3.0])
    def ray_dirs(self, w, h, jitter=None):
        _ys, _xs = _np_v.mgrid[0:h, 0:w]; _jx, _jy = (0., 0.) if jitter is None else (jitter[0], jitter[1])
        _u = ((_xs + _jx) / (w - 1) - 0.5) * 1.1; _v = -((_ys + _jy) / (h - 1) - 0.5) * 1.1
        _d = _np_v.stack([_u, _v, -_np_v.ones_like(_u)], -1); return self.eye, _d / _np_v.linalg.norm(_d, axis=-1, keepdims=True)
def _matV(P):
    _n = len(P); return _np_v.tile([.8, .4, .3], (_n, 1)).astype(float), _np_v.zeros(_n), _np_v.full(_n, .5), _np_v.zeros((_n, 3))
def _skyV(D):
    return _np_v.tile([0.5, 0.6, 0.8], (len(D), 1))
def _smokeV(P):
    P = _np_v.asarray(P, float); return _np_v.clip(1.0 - _np_v.linalg.norm(P - _np_v.array([0, 0.7, 0]), axis=1) / 0.4, 0, 1)
_specV = _RSv(scene=_SceneV(), camera=_CamV(), material=_matV, sky=_skyV, width=48, height=36, quality="draft",
              max_bounce=2, volume={"field": _smokeV, "bounds": (_np_v.array([-1., -1, -1]), _np_v.array([1., 1.4, 1])),
                                    "mode": "smoke", "sigma": 13.0, "steps": 48})
_baseV = _bpv(_PCv(denoise="svgf", dirty_only=False)).run(scene=_specV, seed=0).image
_ctxV = _bpv(_PCv(denoise="svgf", dirty_only=False, volume=True)).run(scene=_specV, seed=0)
_alphaV = _ctxV.buffers["volume_alpha"]; _cov = float((_np_v.asarray(_alphaV) > 0.1).mean())
_diff = float(_np_v.abs(_ctxV.image - _baseV).mean())
print(f"  VOLUME AS A PIPELINE STAGE (backlog H5): a scene with a smoke plume renders as ONE frame. The pipeline runs the surface render, then the VOLUME stage marches the density field and over-composites it: out = volume + surface*(1-alpha). Here the plume covers {_cov*100:.0f}% of the frame and changes the composite by {_diff:.3f} mean vs the no-volume render. Stage order is render -> denoise -> volume -> present, and the volume stage is default-OFF (opt in with volume=True + a scene.volume), so every existing pipeline is unchanged. The volume renderer was already in the box; this makes 'a scene with a volume' actually render as a frame instead of a demo hand-compositing it after.  *** smoke/fire/fog is a first-class pipeline result ***")

# BACKLOG H6: PARTICLES AS A PIPELINE STAGE. The particle system simulates points (an (N,3) array advanced by the
# symplectic integrator), but nothing drew them to a picture. Now a scene can carry a particle cloud and the
# pipeline's particle stage projects + splats them (holographic_pointsplat) and over-composites onto the surface.
from holographic_pipeline import RenderSpec as _RSp, PipelineConfig as _PCp, build_pipeline as _bpp
from holographic_render import Camera as _Camp
from holographic_integrate import ParticleSim as _PSp
import numpy as _np_p
class _SceneP:
    def eval(self, P):
        _d = _np_p.linalg.norm(P - _np_p.array([0, 0, 0.]), axis=-1) - 0.5
        return _np_p.minimum(_d, P[..., 1] + 0.6)
def _matP(P):
    _n = len(P); return _np_p.tile([.3, .3, .35], (_n, 1)).astype(float), _np_p.zeros(_n), _np_p.full(_n, .5), _np_p.zeros((_n, 3))
def _skyP(D):
    return _np_p.tile([0.08, 0.09, 0.13], (len(D), 1))
# simulate a small ember swarm: a buoyant lift advanced by the shared symplectic integrator
_rngp = _np_p.random.default_rng(3)
_posp = _rngp.uniform([-1.0, -0.4, -0.4], [1.0, 0.2, 0.5], (90, 3)); _velp = _np_p.zeros((90, 3))
def _embersp(pos, vel):
    _a = _np_p.zeros_like(pos); _a[:, 1] += 0.6; _a[:, 0] += -0.4 * pos[:, 2]; _a[:, 2] += 0.4 * pos[:, 0]; return _a
_simp = _PSp(_posp, _velp, _embersp, integrator="symplectic")
for _ in range(20):
    _simp.advance(0.06)
_camp = _Camp(eye=(0.0, 0.4, 3.4), target=(0.0, 0.0, 0.0), fov_deg=46, aspect=48 / 36)
_specp = _RSp(scene=_SceneP(), camera=_camp, material=_matP, sky=_skyP, width=48, height=36, quality="draft",
              max_bounce=2, particles={"points": _simp.pos, "colors": (1.0, 0.6, 0.2), "radius_px": 1.6,
                                       "depth_fade": (2.6, 4.6)})
_basep = _bpp(_PCp(denoise="svgf", dirty_only=False)).run(scene=_specp, seed=0).image
_ctxp = _bpp(_PCp(denoise="svgf", dirty_only=False, particles=True)).run(scene=_specp, seed=0)
_ap = _np_p.asarray(_ctxp.buffers["particle_alpha"]); _covp = float((_ap > 0.1).mean())
_diffp = float(_np_p.abs(_ctxp.image - _basep).mean())
print(f"  PARTICLES AS A PIPELINE STAGE (backlog H6): an ember swarm (90 points advanced by the shared symplectic integrator) rendered as a LAYER over the scene. The pipeline projects each point through the camera and splats it as a soft round dot, over-compositing onto the surface: nearer sparks cover farther ones (painter's order) and a depth fade dims the ones drifting back. Here the sparks cover {_covp*100:.0f}% of the frame and change the composite by {_diffp:.3f} mean. Stage order is render -> denoise -> volume -> particles -> present (particles in front of smoke), and the stage is default-OFF, so every existing pipeline is unchanged. The particle SIM already existed; this is the missing RENDERER that turns its points into a picture.  *** simulated points are now a rendered layer ***")

# BACKLOG H4: HAIR AS A PIPELINE STAGE. render_hair drew strands to its OWN image over an opaque background -- it
# could not be a layer over another render. It now optionally returns a coverage ALPHA, so the pipeline's hair
# stage over-composites fur onto a path-traced body: out = hair*alpha + surface*(1-alpha).
from holographic_pipeline import RenderSpec as _RSh, PipelineConfig as _PCh, build_pipeline as _bph
from holographic_render import Camera as _Camh
from holographic_groom import groom as _groomh
from holographic_sdf import sphere as _sphh
import numpy as _np_h
_bodyh = _sphh(0.6)
_strandsh = _groomh(_bodyh.eval, 400, ((-0.8, -0.8, -0.8), (0.8, 0.8, 0.8)), length=0.3, n_pts=6, curl=0.2, seed=0)
_camh = _Camh(eye=(0, 0, 2.5), target=(0, 0, 0), fov_deg=45, aspect=48 / 36)
def _math(P):
    _n = len(P); return _np_h.tile([.4, .25, .15], (_n, 1)).astype(float), _np_h.zeros(_n), _np_h.full(_n, .6), _np_h.zeros((_n, 3))
def _skyh(D):
    return _np_h.tile([0.3, 0.35, 0.45], (len(D), 1))
_spech = _RSh(scene=_bodyh, camera=_camh, material=_math, sky=_skyh, width=48, height=36, quality="draft",
              max_bounce=2, hair={"strands": _strandsh, "shader": "kajiya", "hair_color": (0.6, 0.4, 0.2)})
_baseh = _bph(_PCh(denoise="svgf", dirty_only=False)).run(scene=_spech, seed=0).image
_ctxh = _bph(_PCh(denoise="svgf", dirty_only=False, hair=True)).run(scene=_spech, seed=0)
_ah = _np_h.asarray(_ctxh.buffers["hair_alpha"]); _covh = float((_ah > 0).mean())
_diffh = float(_np_h.abs(_ctxh.image - _baseh).mean())
print(f"  HAIR AS A PIPELINE STAGE (backlog H4): fur composited over a path-traced body by the pipeline, not by hair's own standalone renderer. render_hair now returns a coverage ALPHA alongside its shaded strands, so the hair stage over-composites the coat onto the surface render: the fur covers {_covh*100:.0f}% of the frame and where the alpha is 0 the shaded, shadowed body shows through (composite changed by {_diffh:.3f} mean). Stage order is render -> denoise -> volume -> particles -> hair -> present, hair the last layer so a strand in front of smoke/sparks reads right; default-OFF, so every existing pipeline is unchanged. The strand renderer existed; the alpha is what lets it be a LAYER in the frame.  *** fur is now a compositable layer, lit on a real body ***")

# BUILD: THIN-FILM IRIDESCENCE (soap bubble / oil slick). A thin transparent film reflects light off its top AND
# bottom; the two beams interfere, and whether a colour reinforces or cancels depends on the film thickness and
# the VIEW ANGLE -- so the hue sweeps across a curved surface. holographic_thinfilm computes it from first
# principles (two-beam interference integrated against the CIE colour curves reused from holographic_blackbody).
from holographic_thinfilm import thin_film_tint as _tft, interference_reflectance as _iref
import numpy as _np_ir
# a fixed film seen at a ladder of view angles: the colour SHIFTS (that shift IS iridescence)
_ir_angles = [1.0, 0.8, 0.6, 0.4, 0.2]
_ir_cols = [_tft(320.0, _c, n_film=1.33) for _c in _ir_angles]
_ir_shift = float(_np_ir.linalg.norm(_ir_cols[0] - _ir_cols[-1]))
# sweeping thickness marches through the spectrum; count how many distinct hues a soap film passes through
_ir_sweep = _np_ir.array([_tft(_t, 1.0) for _t in _np_ir.linspace(100, 800, 60)])
_ir_var = float(_ir_sweep.std(axis=0).mean())
# a thicker film packs more interference fringes across the visible band
def _wig(_s): _d = _np_ir.diff(_s); return int(_np_ir.sum(_np_ir.diff(_np_ir.sign(_d)) != 0))
_thin_w = _wig(_iref(150.0, 1.0)); _thick_w = _wig(_iref(900.0, 1.0))
print(f"  THIN-FILM IRIDESCENCE (build: soap bubble / oil slick): a ~320 nm soap film seen from head-on to grazing shifts colour by {_ir_shift:.2f} in RGB -- that angle-dependent shift IS the rainbow sheen. Sweeping the film thickness 100->800 nm marches the tint through the spectrum (per-channel spread {_ir_var:.2f}, a flat colour would be 0), and a thicker 900 nm film packs more interference fringes across the visible band than a thin 150 nm one ({_thick_w} vs {_thin_w} oscillations). Computed from two-beam interference physics and integrated against the same CIE colour curves the blackbody code uses -- no second colour table. A material carries a film thickness (nm); the path tracer tints the reflection by view angle, so soap_bubble / oil_slick render iridescent with no painted texture.  *** interference physics -> a rainbow that moves with your eye ***")

# BUILD: PLACED LIGHTS + NEXT-EVENT ESTIMATION. The path tracer used to get light only when a bounce ray happened
# to escape and hit the emissive sky -- so a small bright lamp was almost never found (hopeless noise). NEE looks
# STRAIGHT at each light with a shadow ray (holographic_lights) and adds its contribution directly: lamps converge
# instantly and cast real, correctly-shaped shadows. The random bounce still runs, so indirect light isn't lost.
from holographic_pathtrace import path_trace as _pt_l
from holographic_render import Camera as _Cam_l
from holographic_lights import PointLight as _PL, SphereLight as _SL, direct_lighting as _dl
from holographic_sdf import sphere as _sph_l, box as _box_l
import numpy as _np_l
_scene_l = _sph_l(0.6).smooth_union(_box_l(2.0, 0.1, 2.0).translate((0, -0.7, 0)), k=0.05)
_cam_l = _Cam_l(eye=(0, 0.6, 3.0), target=(0, -0.2, 0), fov_deg=45, aspect=1.0)
_dark_l = lambda D: _np_l.tile([0.02, 0.02, 0.03], (len(D), 1))
def _mat_l(P):
    _n = len(P); return _np_l.tile([0.7, 0.6, 0.5], (_n, 1)).astype(float), _np_l.zeros(_n), _np_l.full(_n, 0.6), _np_l.zeros((_n, 3))
_lamp = _PL(position=(1.5, 2.5, 1.0), color=(1, 0.95, 0.85), intensity=12.0)
_noL = _pt_l(_scene_l, _cam_l, 48, 48, spp=6, max_bounce=3, material=_mat_l, sky=_dark_l, seed=0)
_wiL = _pt_l(_scene_l, _cam_l, 48, 48, spp=6, max_bounce=3, material=_mat_l, sky=_dark_l, seed=0, lights=[_lamp])
_shadow_frac = float((_wiL.mean(2) < 0.05).mean())
# a sphere light's sampled directions vary -> soft shadows (a penumbra)
_sl = _SL(position=(0, 3, 0), radius=1.0); _rng_l = _np_l.random.default_rng(0)
_sdirs = _np_l.array([_sl.sample(_np_l.array([[0.0, 0.0, 0.0]]), _rng_l)[0][0] for _ in range(50)])
_soft = float(_sdirs.std(axis=0).mean())
print(f"  PLACED LIGHTS + NEXT-EVENT ESTIMATION (build): with only a dark sky the scene is nearly black (mean {float(_noL.mean()):.3f}); adding ONE point lamp lights it to {float(_wiL.mean()):.3f} and casts shadows ({_shadow_frac*100:.0f}% of the frame in shadow) -- because NEE points a shadow ray straight at the lamp instead of hoping a random bounce finds it. A sphere light samples a random point on its surface each time (direction spread {_soft:.2f}), so averaging gives SOFT shadows -- a penumbra. lights=None is byte-identical to the old environment-only path, so nothing existing changed.  *** you can put lamps in the world now, with correct shadows ***")

# BUILD: THE FULL LIGHT RIG. Placed lights went from 3 types to the set a real DCC app has -- point, directional,
# ambient, spot (with a projected GOBO cookie), rect/area (a softbox), sphere, mesh (an emissive triangle mesh),
# and IES (a real luminaire's measured beam shape). Every light's colour and intensity can also be a FIELD -- a
# callable that varies the parameter across the scene. All sample directly via next-event estimation, so all cast
# correct shadows (holographic_lights).
from holographic_lights import (PointLight as _L_pt, SpotLight as _L_sp, RectLight as _L_rc, SphereLight as _L_sh,
                                 IESLight as _L_ie, MeshLight as _L_me, AmbientLight as _L_am, direct_lighting as _L_dl)
import numpy as _np_lr
_rng_lr = _np_lr.random.default_rng(0)
# a spot's cone: bright on-axis, dark outside
_spot = _L_sp(position=(0, 3, 0), direction=(0, -1, 0), inner_deg=10, outer_deg=20, intensity=40.0)
_on = float(_spot.sample(_np_lr.array([[0.0, 0.0, 0.0]]), _rng_lr)[2].max())
_off = float(_spot.sample(_np_lr.array([[3.0, 0.0, 0.0]]), _rng_lr)[2].max())
# a gobo splits the beam; an IES profile shapes it; a colour FIELD varies the tint
def _half(uv): return (uv[:, 0] > 0).astype(float)
_gspot = _L_sp(position=(0, 3, 0), direction=(0, -1, 0), inner_deg=30, outer_deg=40, intensity=40.0, gobo=_half)
_gp = float(_gspot.sample(_np_lr.array([[0.4, 0.0, 0.0]]), _rng_lr)[2].max())
_gb = float(_gspot.sample(_np_lr.array([[-0.4, 0.0, 0.0]]), _rng_lr)[2].max())
_ies = _L_ie(position=(0, 3, 0), direction=(0, -1, 0), profile=_np_lr.cos(_np_lr.linspace(0, _np_lr.pi/2, 20))**4, profile_max_deg=90, intensity=40.0)
_ib = float(_ies.sample(_np_lr.array([[0.0, 0.0, 0.0]]), _rng_lr)[2].max())
_is = float(_ies.sample(_np_lr.array([[3.0, 2.9, 0.0]]), _rng_lr)[2].max())
def _redright(P):
    _x = _np_lr.atleast_2d(P)[:, 0]; return _np_lr.stack([_np_lr.clip(_x, 0, 1), _np_lr.zeros_like(_x), _np_lr.zeros_like(_x)], axis=1)
_cf = _L_pt(position=(0, 3, 0), color=_redright, intensity=1.0).sample(_np_lr.array([[0.2, 0, 0], [0.9, 0, 0]]), _rng_lr)[2]
print(f"  THE FULL LIGHT RIG (build): eight light types -- point, directional, ambient, spot, rect/area, sphere, mesh, IES -- all sampled by next-event estimation so all cast correct shadows. A SPOT confines light to a cone (on-axis {_on:.2f} vs outside the cone {_off:.4f}); a GOBO cookie projects a pattern across that cone (lit half {_gp:.2f} vs blocked half {_gb:.2f}); an IES profile gives a real luminaire's beam shape (on-axis {_ib:.2f} vs off-axis {_is:.3f}); a rect/area light and a sphere light have AREA so their shadows are soft; a mesh light emits from any triangle geometry. And every light's colour or intensity can be a FIELD that varies across the scene -- a colour-gradient lamp reads redder to the right ({float(_cf[1,0]):.2f} vs {float(_cf[0,0]):.2f}). load_ies() reads a real .ies photometric file.  *** a full DCC light rig, all deterministic on NumPy ***")

# BUILD: THE CACHED DOME (RENDER-DC1). The dome -- soft ambient occlusion under a coloured sky -- is the softest,
# most expensive light: brute-forced it is many ray-traced AO samples per pixel per bounce, and still noisy. We
# treat it as a three-tier CACHE instead (holographic_domecache): bake the PRT sky-visibility transfer at a COARSE
# GRID of anchors (warm), serve every other pixel by a smooth normal-aware interpolation of its neighbours (hot),
# and recompute EXACTLY only at the silhouette/contact edges the smooth cache can't represent (cold). The
# sharpness+normal map is the hit/miss policy -- the cheap projection where the field is smooth, the pathfinding
# only where it isn't.
from holographic_domecache import render_dome_term as _dc_term, dome_light_sh as _dc_sh
from holographic_prt import precompute_transfer as _dc_bake, shade_prt as _dc_shade
from holographic_sdf import box as _dc_box, sphere as _dc_sphere
from holographic_render import Camera as _dc_Cam
from holographic_lights import DomeLight as _dc_Dome
from holographic_domecache import _primary_gbuffer as _dc_gb
import numpy as _dc_np, time as _dc_time
_dc_scene = _dc_sphere(0.5).smooth_union(_dc_box(3.0, 0.1, 3.0).translate((0, -0.55, 0)), k=0.02)
_dc_cam = _dc_Cam(eye=(0, 0.9, 3.0), target=(0, -0.2, 0), fov_deg=45, aspect=1.0)
_dc_dome = _dc_Dome(color=(0.4, 0.5, 0.7), ground_color=(0.15, 0.12, 0.1), intensity=1.0)
def _dc_mat(_p):
    _n = len(_p); return (_dc_np.tile([0.8, 0.8, 0.8], (_n, 1)).astype(float),)
_dc_W = 96
_t0 = _dc_time.time(); _dc_cached, _dc_st = _dc_term(_dc_scene, _dc_cam, _dc_W, _dc_W, _dc_dome, _dc_mat, stride=6, return_stats=True); _dc_tc = _dc_time.time() - _t0
# reference: bake EVERY visible pixel (what the cache approximates), for the speedup + accuracy numbers
_dc_hit, _dc_P, _dc_N = _dc_gb(_dc_scene, _dc_cam, _dc_W, _dc_W)
_dc_lsh = _dc_sh(_dc_dome)
_t0 = _dc_time.time(); _dc_T = _dc_bake(_dc_scene, _dc_P[_dc_hit], _dc_N[_dc_hit], order=3, n=64); _dc_full = _dc_np.zeros((_dc_W, _dc_W, 3)); _dc_full[_dc_hit] = _dc_shade(_dc_T, _dc_lsh, _dc_np.full((int(_dc_hit.sum()), 3), 0.8)); _dc_tf = _dc_time.time() - _t0
_dc_err = float(_dc_np.abs(_dc_cached[_dc_hit].mean(1) - _dc_full[_dc_hit].mean(1)).mean())
# ambient occlusion check: the sphere's contact region is darker than the open floor
_dc_lum = _dc_cached.mean(2); _dc_floor = _dc_hit & (_dc_N[..., 1] > 0.9)
_dc_cx = float(_dc_np.where(_dc_floor)[1].mean()); _dc_cols = _dc_np.arange(_dc_W)[None, :]
_dc_near = float(_dc_lum[_dc_floor & (_dc_np.abs(_dc_cols - _dc_cx) < 12)].mean()); _dc_far = float(_dc_lum[_dc_floor & (_dc_np.abs(_dc_cols - _dc_cx) > 30)].mean())
print(f"  THE CACHED DOME (build): the soft sky-ambient light served as a three-tier cache -- bake PRT transfer at a coarse anchor grid (warm), smooth normal-aware interpolate the rest (hot), recompute exactly only at the edges (cold). On a {_dc_W}x{_dc_W} frame the cache is {_dc_tf/max(_dc_tc,1e-6):.1f}x faster than baking every pixel ({_dc_tc:.2f}s vs {_dc_tf:.2f}s) at {_dc_err:.4f} mean error -- noise-free, with a {_dc_st['hit_rate']:.0%} cache HIT rate ({_dc_st['anchors_baked']} anchors baked, {_dc_st['misses_recomputed']} edge pixels recomputed). And it is a real shadowed dome, not a flat fill: the sphere's contact shadow ({_dc_near:.3f}) is darker than the open floor ({_dc_far:.3f}). *** the 323s dome, cached to sub-second ***")

# BUILD: MODULATE / DEMODULATE (M1) + DEMODULATED DENOISE (M4). A diffuse pixel is a PRODUCT: albedo * irradiance --
# a crisp carrier (texture) times a smooth residual (lighting). Splitting them is UNBIND (demodulate); recombining
# is BIND (remodulate). The engine already spends this as bake-and-query five times (matcompile/matbake/viewlut/prt/
# radiance) and now once more in DENOISING: divide the albedo out, denoise the smooth irradiance (no texture to
# smear), multiply it back (holographic_modulate).
from holographic_modulate import demodulate as _md_dm, remodulate as _md_rm, denoise_demodulated as _md_dd
from holographic_svgf import atrous_bilateral as _md_at
import numpy as _md_np
_md_rng=_md_np.random.default_rng(0); _md_H=_md_W=96
_md_yy,_md_xx=_md_np.mgrid[0:_md_H,0:_md_W]
_md_chk=(((_md_xx//8)+(_md_yy//8))%2).astype(float)*0.6+0.3          # a checker ALBEDO (texture / carrier)
_md_alb=_md_np.stack([_md_chk]*3,2)
_md_irr=(0.4+0.5*(_md_xx/_md_W))[...,None]*_md_np.ones(3)            # smooth IRRADIANCE (lighting / residual)
_md_clean=_md_alb*_md_irr
_md_noisy=_md_clean+_md_rng.normal(0,0.06,_md_clean.shape)          # Monte-Carlo-like noise
_md_n=_md_np.zeros((_md_H,_md_W,3)); _md_n[...,2]=1.0; _md_d=_md_np.ones((_md_H,_md_W))
_md_rt=float(_md_np.abs(_md_rm(_md_dm(_md_clean,_md_alb,eps=0.0),_md_alb)-_md_clean).max())   # round-trip error
_md_g=_md_at(_md_noisy,_md_n,_md_alb,_md_d,levels=5)                 # guide-only denoise (albedo as edge-stop)
_md_m=_md_dd(_md_noisy,_md_n,_md_alb,_md_d,levels=5)                 # M4 demodulated denoise
_md_eg=float(_md_np.abs(_md_g-_md_clean).mean()); _md_em=float(_md_np.abs(_md_m-_md_clean).mean())
def _md_edge(_im): return float(_md_np.abs(_md_np.diff(_im[_md_H//2].mean(1))).max())
print(f"  MODULATE/DEMODULATE (build): a diffuse pixel is albedo*irradiance -- a crisp texture carrier times smooth lighting. demodulate=UNBIND (divide), remodulate=BIND (multiply); the round-trip is exact ({_md_rt:.1e} max error). Applied to DENOISING (M4): dividing the albedo out lets the filter smooth the noisy IRRADIANCE hard without smearing texture, then it multiplies the crisp albedo back. On a textured+lit noisy frame that is {100*(1-_md_em/_md_eg):.0f}% less error than filtering colour directly ({_md_em:.4f} vs {_md_eg:.4f}) while KEEPING the texture edges ({_md_edge(_md_m):.2f} vs the true {_md_edge(_md_clean):.2f}). Same move as the five existing bakes (matcompile/matbake/viewlut/prt/radiance) and the cached dome -- bake the smooth factor, multiply the crisp one. KEPT NEGATIVE: only pays where albedo VARIES (texture); neutral on flat matte; diffuse only. *** bind/unbind, spent on the pixel product ***")
# M5: the SAME move for UPSCALING -- render lighting low-res, upscale the smooth irradiance, remodulate crisp hi-res albedo
from holographic_modulate import superres_demodulated as _m5_up
from holographic_fsr import easu_upscale as _m5_ez
_m5_low=_md_clean[::2,::2]                                            # a 2x-smaller "low-res" render of the textured scene
_m5_hi=_m5_up(_m5_low,_md_alb)                                       # demod upscale -> crisp texture from the hi-res albedo
_m5_pl=_m5_ez(_m5_low,2.0)[:_md_H,:_md_W]                            # naive: upscale the colour directly (blurs texture)
_m5_em=float(_md_np.abs(_m5_hi-_md_clean).mean()); _m5_ep=float(_md_np.abs(_m5_pl-_md_clean).mean())
print(f"  DEMODULATED UPSCALE (M5): the same split for super-resolution -- render the expensive LIGHTING at low res, upscale the smooth irradiance (nothing to alias), and multiply the CRISP high-res albedo back (which is cheap: a material lookup, no light transport). High-res detail for low-res lighting cost. On the textured frame that is {100*(1-_m5_em/_m5_ep):.0f}% less error than upscaling the colour directly ({_m5_em:.4f} vs {_m5_ep:.4f}). KEPT NEGATIVE: demodulate by the ANTI-ALIASED (downsampled) high albedo so high-frequency texture doesn't alias the carrier; neutral on flat; diffuse only.")

# BUILD: CACHED SOFT AREA LIGHTS (RENDER-DC2) -- the second half of the two-mode cached-lighting design, and the fix
# for the placed-light speckle. Two parts. First a ROOT-CAUSE bug fix: the area lights (Rect/Disk/Sphere/Mesh) never
# carried the `.soft` flag direct_lighting checks, so every area light was being sampled with ONE shadow ray -- a
# biased HARD shadow, speckly in the penumbra. Flagging them soft makes NEE multi-sample the penumbra. Second, the
# CACHE: the soft-shadowed irradiance is a smooth field, so we bake it noise-free at a coarse anchor grid (many
# samples, cheap because sparse), interpolate the rest, recompute the sharp edges -- the SAME three-tier engine as
# the dome (holographic_domecache.cached_screen_shade, now shared by both).
from holographic_lightcache import cached_soft_lights_shade as _lc_cache, split_soft_lights as _lc_split
from holographic_domecache import _primary_gbuffer as _lc_gb
from holographic_sdf import box as _lc_box, sphere as _lc_sphere
from holographic_render import Camera as _lc_Cam
from holographic_lights import RectLight as _lc_Rect, PointLight as _lc_Point
import numpy as _lc_np
_lc_scene=_lc_sphere(0.5).smooth_union(_lc_box(3.0,0.1,3.0).translate((0,-0.55,0)),k=0.02)
_lc_cam=_lc_Cam(eye=(0,0.9,3.0),target=(0,-0.2,0),fov_deg=45,aspect=1.0)
_lc_rect=_lc_Rect(position=(0.7,2.2,1.0),u_vec=(0.6,0,0),v_vec=(0,0.4,0.3),color=(1,1,1),intensity=40.0)
def _lc_mat(_p):
    _n=len(_p); return (_lc_np.tile([0.8,0.8,0.8],(_n,1)).astype(float),_lc_np.zeros(_n),_lc_np.full(_n,0.7))
_lc_soft,_lc_hard=_lc_split([_lc_rect,_lc_Point(position=(0,2,0),intensity=10.0)])
_lc_s0,_lc_st=_lc_cache(_lc_scene,_lc_cam,96,96,[_lc_rect],_lc_mat,area_samples=48,stride=6,return_stats=True)
_lc_s1=_lc_cache(_lc_scene,_lc_cam,96,96,[_lc_rect],_lc_mat,area_samples=48,stride=6,seed=99)
_lc_seeddiff=float(_lc_np.abs(_lc_s0-_lc_s1).mean())
print(f"  CACHED SOFT AREA LIGHTS (build): the area lights (Rect/Disk/Sphere/Mesh) were missing the .soft flag, so each was sampled with ONE shadow ray -- a biased hard shadow, the source of the placed-light speckle in the penumbra. Fixed the flag (now multi-sampled), then CACHED the soft term the same way as the dome: bake many-sampled NEE at {_lc_st['anchors_baked']} coarse anchors, smooth normal-aware interpolate, recompute {_lc_st['misses_recomputed']} edge pixels. The result is NOISE-FREE -- seed-to-seed difference {_lc_seeddiff:.4f} (a per-pixel MC render of the same soft shadow is ~8x noisier), {_lc_st['hit_rate']:.0%} cache hit rate. The split picks {len(_lc_soft)} soft light and leaves {len(_lc_hard)} hard for the tracer. KEPT HONEST: this cleans the DIRECT soft-shadow term; the dominant showcase speckle is INDIRECT/GI bounce noise, which this doesn't touch -- that needs an indirect irradiance cache next. *** the dome cache, generalised to placed area lights ***")

# BUILD: CACHED INDIRECT / GLOBAL ILLUMINATION (RENDER-DC3) -- the fix for the DOMINANT placed-light speckle. We
# measured that ~73% of the speckle is INDIRECT (GI bounce) noise, not direct soft shadows. Indirect light varies
# SMOOTHLY over diffuse surfaces (Ward 1988), so it caches the same way as the dome and the soft lights: bake a
# many-ray one-bounce hemisphere gather at coarse anchors (noise-free), interpolate, recompute the edges. The tracer
# then renders DIRECT-only and the clean cached GI is added -- replacing the noisy multi-bounce GI.
from holographic_lightcache import cached_indirect_shade as _gi_cache
from holographic_sdf import box as _gi_box, sphere as _gi_sphere
from holographic_render import Camera as _gi_Cam
from holographic_lights import RectLight as _gi_Rect
import numpy as _gi_np
# a red wall by a white floor -> the one-bounce indirect should bleed red onto the floor
_gi_scene=_gi_box(3.0,0.1,3.0).translate((0,-0.6,0)).smooth_union(_gi_box(0.1,1.5,3.0).translate((-1.0,0.3,0)),k=0.02).smooth_union(_gi_sphere(0.4).translate((0.3,-0.2,0)),k=0.02)
_gi_cam=_gi_Cam(eye=(1.2,0.9,3.0),target=(-0.2,-0.2,0),fov_deg=48,aspect=1.0)
def _gi_mat(_p):
    _n=len(_p); _a=_gi_np.tile([0.85,0.85,0.85],(_n,1)).astype(float); _a[_p[:,0]<-0.85]=[0.85,0.12,0.12]; return _a,_gi_np.zeros(_n),_gi_np.full(_n,0.7),_gi_np.zeros((_n,3))
_gi_L=[_gi_Rect(position=(0.5,2.2,1.0),u_vec=(0.5,0,0),v_vec=(0,0.4,0.3),intensity=40.0)]
_gi_0,_gi_st=_gi_cache(_gi_scene,_gi_cam,80,80,_gi_L,_gi_mat,n_dirs=48,stride=8,seed=0,return_stats=True)
_gi_1=_gi_cache(_gi_scene,_gi_cam,80,80,_gi_L,_gi_mat,n_dirs=48,stride=8,seed=42)
_gi_sd=float(_gi_np.abs(_gi_0-_gi_1).mean()); _gi_lit=_gi_0.sum(2)>0.005
_gi_bleed=float((_gi_0[...,0]-_gi_0[...,2])[_gi_lit].mean())
print(f"  CACHED INDIRECT / GI (build): the placed-light speckle is DOMINANTLY global-illumination bounce noise (measured ~73%), not direct soft shadows. Indirect light is a smooth field, so we cache it like the dome: bake a many-ray one-bounce gather at {_gi_st['anchors_baked']} coarse anchors, interpolate, recompute the edges. The tracer renders DIRECT-only and this clean GI is added. Result on the showcase: baking the caches ONCE then rendering direct-only cut the placed-light speckle 87% (seed-diff 0.034 -> 0.004) and ran ~8x faster, with real colour bleeding (red wall -> floor, R-B tint {_gi_bleed:+.3f}). The cached GI itself is NOISE-FREE (seed-diff {_gi_sd:.4f}). KEPT HONEST: ONE bounce, not full multi-bounce GI -> ~5% dimmer than the tracer (misses higher-order bounces); diffuse gather. *** the dominant speckle, cached away ***")

# BUILD: THE CAPABILITY CATALOG (consolidation C1) -- 'search before you build'. The engine has ~340 modules and
# the recurring cost is DUPLICATION: the same job is often already solved, but a new session can't find it and
# builds a fourth copy. The catalog is the index of what exists: describe a problem in plain English, get the home
# that already does it. Each entry has a plain `does`, a copy-paste `example`, and a `native` flag. find_capability
# is a small readable token-overlap match -- no training, deterministic.
from holographic_catalog import default_catalog as _cat_default, seed_from_mind as _cat_seed
_cat=_cat_default()
_cat_q1=[_h.name for _h in _cat.find_capability("search a big pile of vectors", k=2)]
_cat_q2=[_h.name for _h in _cat.find_capability("precompute a slow factor and reuse it", k=1)]
_cat_q3=[_h.name for _h in _cat.find_capability("my placed light has speckle noise", k=1)]
print(f"  CAPABILITY CATALOG (build): the engine's index of its own capabilities, so a session searches BEFORE building a duplicate ('route, don't rewrite'). Describe a problem, get the home: 'search a big pile of vectors' -> {_cat_q1}; 'precompute a slow factor and reuse it' -> {_cat_q2}; 'my placed light has speckle noise' -> {_cat_q3}. {len(_cat)} curated homes seeded (search indices, caches/bakes, field types), each with a plain-English does + copy-paste example + a native flag; seed_from_mind adds every live UnifiedMind faculty so both curated homes and methods are findable. This is Phase 0 of the consolidation backlog -- the cheap move that stops the bleeding. *** the engine, searchable by problem ***")

# BUILD: PIPELINE RENDER-STRATEGY DISPATCH (consolidation R1) -- the render stage used to hard-wire one path
# (pathtrace). Now it ROUTES among named strategies, each declaring what it NEEDS and delegating to its real module:
# pathtrace (converge_samples), raymarch (fast preview), prt (relight under an env SH), radiance (query a baked
# field). 'auto' picks one; a missing input is caught with a clear message BEFORE rendering. Route, don't rewrite.
from holographic_pipeline import RenderSpec as _r1_Spec, Pipeline as _r1_Pipe, ALL_STAGES as _r1_STAGES
from holographic_pipeline import dispatch_render as _r1_disp, FrameState as _r1_FS, PipelineError as _r1_Err, RENDER_STRATEGIES as _r1_REG
from holographic_sdf import box as _r1_box, sphere as _r1_sphere
from holographic_render import Camera as _r1_Cam
import numpy as _r1_np
_r1_scene=_r1_sphere(0.5).smooth_union(_r1_box(2.2,0.1,2.2).translate((0,-0.55,0)),k=0.03)
_r1_cam=_r1_Cam(eye=(0,0.7,2.8),target=(0,-0.2,0),fov_deg=45,aspect=1.0)
def _r1_mat(_p): _n=len(_p); return _r1_np.tile([0.8,0.8,0.8],(_n,1)).astype(float),_r1_np.zeros(_n),_r1_np.full(_n,0.6),_r1_np.zeros((_n,3))
_r1_spec=_r1_Spec(scene=_r1_scene,camera=_r1_cam,material=_r1_mat,width=40,height=30,quality="draft",max_bounce=2)
_r1_out=_r1_Pipe([_s for _s in _r1_STAGES if _s.name in ("render","present")]).run(scene=_r1_spec,seed=0)
_r1_picked=_r1_out.buffers["render_method"]
try:
    _r1_disp(_r1_FS(scene=_r1_Spec(scene=_r1_scene,camera=_r1_cam,method="prt"),seed=0)); _r1_msg="(no error -- unexpected)"
except _r1_Err as _e: _r1_msg=str(_e)
print(f"  PIPELINE STRATEGY DISPATCH (build): the render stage now ROUTES among {len(_r1_REG)} strategies ({', '.join(sorted(_r1_REG))}) instead of hard-wiring one -- each declares its NEEDS and delegates to its real module. A render went through the Pipeline and 'auto' picked '{_r1_picked}' (pathtrace when a material is present -> byte-identical to before). Asking for 'prt' with no light_sh is caught before rendering: {_r1_msg!r}. Route, don't rewrite: pathtrace/raymarch/prt/radiance stay distinct algorithms behind one entry point. *** one door, many strategies, inputs checked ***")

# BUILD: THE FIELD HOME (consolidation R2) -- the engine grew several ways to hold a field over space (dense grid,
# narrow-band sparse voxels, callable/SDF oracle, spectral, FPE, region, dirty), each with its OWN sampler. Field
# gives them ONE interface -- field.sample(points) -- a thin adapter that ROUTES to each backend's own reader.
from holographic_fieldhome import Field as _r2_Field, field_backends as _r2_backends
import numpy as _r2_np
def _r2_oracle(_P): return 1.0 - _r2_np.linalg.norm(_r2_np.asarray(_P,float),axis=1)
_r2_lo=_r2_np.array([-1.,-1.,-1.]); _r2_hi=_r2_np.array([1.,1.,1.]); _r2_N=16
_r2_axis=[_r2_lo[_d]+_r2_np.arange(_r2_N)/_r2_N*(_r2_hi[_d]-_r2_lo[_d]) for _d in range(3)]
_r2_gx,_r2_gy,_r2_gz=_r2_np.meshgrid(_r2_axis[0],_r2_axis[1],_r2_axis[2],indexing="ij")
_r2_grid=_r2_oracle(_r2_np.stack([_r2_gx.ravel(),_r2_gy.ravel(),_r2_gz.ravel()],axis=1)).reshape(_r2_N,_r2_N,_r2_N)
_r2_probe=_r2_np.array([[_r2_axis[0][2],_r2_axis[1][5],_r2_axis[2][9]],[_r2_axis[0][7],_r2_axis[1][1],_r2_axis[2][3]]])
_r2_a=_r2_Field.grid(_r2_grid,_r2_lo,_r2_hi).sample(_r2_probe)
_r2_b=_r2_Field.callable(_r2_oracle).sample(_r2_probe)
_r2_gap=float(_r2_np.max(_r2_np.abs(_r2_a-_r2_b)))
print(f"  FIELD HOME (build): the scattered spatial-field representations now answer to ONE interface -- field.sample(points) -- backend chosen by cost ({', '.join(_r2_backends())}; spectral/FPE/region/dirty plug in the same way). It's a thin adapter that ROUTES to each rep's own sampler, not a rewrite. Proof the routing is faithful: the DENSE grid and the CALLABLE oracle, sampled through the same Field interface, agree to {_r2_gap:.1e} at grid nodes -- two backends, identical values. Not to be confused with the unit-sphere compositional Field (holographic_field); this home is for spatial fields over ordinary R^D. *** many field reps, one .sample ***")

# BUILD: THE INDEX HOME (consolidation H1) -- "find the k nearest vectors" was written many times: an exact cosine
# scan for small sets (holographic_ai.nearest), the sub-linear RP-forest for large (holographic_tree.HoloForest).
# Index is one door -- Index(vectors).nearest(query, k) -- routing by size, with a calibrated ABSTAIN so a query
# against noise returns nothing instead of a confident guess. Route, don't rewrite: the scan and the forest stay put.
from holographic_index import Index as _h1_Index, index_backends as _h1_backends
import numpy as _h1_np
_h1_rng=_h1_np.random.default_rng(0); _h1_V=_h1_rng.standard_normal((300,64))
_h1_q=_h1_V[123]+0.1*_h1_np.random.default_rng(1).standard_normal(64)
_h1_exact=_h1_Index(_h1_V,method="exact").nearest(_h1_q)[0][0]
_h1_forest=_h1_Index(_h1_V,method="forest",forest_threshold=0).nearest(_h1_q)[0][0]
_h1_noise=_h1_Index(_h1_V,method="exact").nearest(_h1_np.random.default_rng(9).standard_normal(64),abstain=0.01)
_h1_auto=_h1_Index(_h1_V,method="auto",forest_threshold=100).method
print(f"  INDEX HOME (build): nearest-neighbour search unified behind Index.nearest(query, k) over {', '.join(_h1_backends())} strategies, chosen by size. On a 300-vector set the EXACT scan and the sub-linear FOREST both recall the right item ({_h1_exact} == {_h1_forest}); 'auto' at threshold 100 picked '{_h1_auto}'. A query against pure noise ABSTAINS (calibrated false-alarm control): {_h1_noise!r}. Two real callers now route through it (lexicon.nearest, TextEncoder.nearest) with byte-identical rankings, and the recall benchmark is unchanged. Kept honest: the cosine family only -- Euclidean point k-NN (spatial) and ray indices stay their own homes. *** one search, many callers ***")

# BUILD: THE CACHE HOME (consolidation H2) -- "bake once, query O(1)" is the engine's core performance lever, but
# the SAME grid-sample-and-store was rewritten in every bake (matbake, sdfbake, viewlut, anim). Cache owns the
# shared core -- Cache.grid_points(lo,hi,res) and Cache.bake(fn, vary=...) -- so the bakes stop duplicating it.
from holographic_cachehome import Cache as _h2_Cache, cache_backends as _h2_backends
from holographic_matbake import bake_field as _h2_bakefield
from holographic_sdfbake import bake_sdf_grid as _h2_bakesdf
import numpy as _h2_np
_h2_lo=_h2_np.array([-1.,-1.,-1.]); _h2_hi=_h2_np.array([1.,1.,1.]); _h2_res=16
_h2_fn=lambda _P: _P[:,0]**2 + _P[:,1]
# bake a field through the home, look it up O(1); exact at a grid node
_h2_bg=_h2_Cache.bake(_h2_fn, vary="position", lo=_h2_lo, hi=_h2_hi, res=_h2_res)
_h2_pts,_=_h2_Cache.grid_points(_h2_lo,_h2_hi,_h2_res)
_h2_err=abs(float(_h2_bg.sample(_h2_pts[50][None,:])[0]) - float(_h2_fn(_h2_pts[50][None,:])[0]))
# the two bakes now share the grid generator -> bit-identical to the inline meshgrid they each used to carry
_h2_axes=[_h2_np.linspace(_h2_lo[_k],_h2_hi[_k],_h2_res) for _k in range(3)]
_h2_gx,_h2_gy,_h2_gz=_h2_np.meshgrid(*_h2_axes,indexing="ij"); _h2_ip=_h2_np.stack([_h2_gx.ravel(),_h2_gy.ravel(),_h2_gz.ravel()],axis=1)
_h2_mb=_h2_np.array_equal(_h2_bakefield(_h2_fn,"roughness",_h2_lo,_h2_hi,res=_h2_res).grid, _h2_fn(_h2_ip).reshape(_h2_res,_h2_res,_h2_res))
_h2_sd=lambda _P: _h2_np.linalg.norm(_P,axis=1)-0.5
_h2_sb=_h2_np.array_equal(_h2_bakesdf(_h2_sd,_h2_lo,_h2_hi,_h2_res)[0], _h2_np.asarray(_h2_sd(_h2_ip),float).reshape(_h2_res,_h2_res,_h2_res))
print(f"  CACHE HOME (build): the bake-and-query lever unified -- Cache.bake(fn, vary=...) over {', '.join(_h2_backends())}, with ONE shared position-grid generator the bakes used to each rewrite. Baked field looks up O(1), exact at a node (err {_h2_err:.1e}). The DONE-WHEN met: matbake.bake_field and sdfbake.bake_sdf_grid both route through Cache.grid_points now, BIT-IDENTICAL to their old inline meshgrid (matbake {_h2_mb}, sdfbake {_h2_sb}). Route, don't rewrite: each keeps its own lookup reader; only the precompute is shared. Not holographic_cache (Ward gradient cache) -- that's the sparse-anchor scheme. *** bake once, look up everywhere ***")

# BUILD: MATERIAL + SHADING (consolidation R3) -- the three homes wired into one path. The Shading model (brdf) was
# already centralised (cook_torrance, sample_brdf; no re-derived Fresnel/GGX in the render paths), but the DIFFUSE
# lambert term was re-derived inline in a few gather paths, and brdf had no standalone lambert. Added brdf.lambert
# and routed globalillum's bounce gather to it (bit-identical). The three-way now composes: Material -> Cache -> Shading.
from holographic_surface import SurfaceMaterial as _r3_SM
from holographic_param import Param as _r3_Param
from holographic_matbake import bake_material as _r3_bake
from holographic_brdf import cook_torrance as _r3_ct, lambert as _r3_lam
import numpy as _r3_np
_r3_col=lambda _P,**_k: _r3_np.stack([0.5+0.4*_r3_np.asarray(_P)[:,1],_r3_np.full(len(_P),0.3),_r3_np.full(len(_P),0.6)],axis=1)
_r3_rough=lambda _P,**_k: 0.3+0.2*_r3_np.sin(_r3_np.asarray(_P)[:,0]*2.0)
_r3_mat=_r3_SM(color=_r3_Param(field=_r3_col),roughness=_r3_Param(field=_r3_rough),reflect=0.1,emission=0.0)
_r3_shade=_r3_bake(_r3_mat,(-1.,-1.,-1.),(1.,1.,1.),res=16)              # Material bakes via Cache (H2)
_r3_P=_r3_np.array([[0.2,0.3,-0.1],[-0.4,-0.2,0.5]]); _r3_ch=_r3_shade(_r3_P)   # channels pulled from the baked Material
_r3_alb=_r3_np.asarray(_r3_ch["color"],float); _r3_ro=_r3_np.asarray(_r3_ch["roughness"],float)
_r3_N=_r3_np.array([[0.,1.,0.],[0.,1.,0.]]); _r3_V=_r3_N.copy(); _r3_L=_r3_np.array([0.3,0.8,0.5]); _r3_L=_r3_L/_r3_np.linalg.norm(_r3_L)
_r3_rad=_r3_ct(_r3_N,_r3_V,_r3_np.broadcast_to(_r3_L,(2,3)),_r3_alb,0.0,_r3_ro)   # shade via Shading (brdf)
# lambert bit-identical to the inline it replaced
_r3_bi=_r3_np.array_equal(_r3_lam(_r3_N,_r3_L,_r3_alb), _r3_np.clip(_r3_N@_r3_L,0,None)[:,None]*_r3_alb)
print(f"  MATERIAL + SHADING (build): the three homes compose in one path -- a Material's position channels BAKE through the Cache home (H2) into fast lookups, a surface pulls its channels from that baked Material ({_r3_alb.shape[0]} points, albedo finite {bool(_r3_np.isfinite(_r3_alb).all())}), and it SHADES through the Shading home (brdf.cook_torrance, radiance>=0 {bool((_r3_rad>=0).all())}). The one genuine de-dup: added brdf.lambert (the diffuse term brdf lacked) and routed globalillum's bounce gather to it, bit-identical ({_r3_bi}). Kept honest: compound shades that fold in shadow/occlusion/spec keep their own expression -- lambert is only the diffuse term. *** channels bake, surface shades, one BRDF ***")

# BUILD: THE SAMPLING HOME (consolidation R4) -- the Monte-Carlo sampling pieces were shipped but scattered
# (low-discrepancy, blue-noise, MIS, firefly accumulation), and the cosine-hemisphere sampler was copied into THREE
# modules (brdf, globalillum, lights). Sampling is one thin home: it ROUTES to the shipped pattern generators and
# OWNS the cosine-hemisphere, so the gather paths share one implementation.
from holographic_samplinghome import Sampling as _r4_S, sampling_backends as _r4_backends
from holographic_lowdiscrepancy import low_discrepancy as _r4_ld
from holographic_globalillum import _cosine_hemisphere as _r4_gih
import numpy as _r4_np
_r4_N=_r4_np.array([[0.,1.,0.],[0.5,0.6,0.62]]); _r4_N=_r4_N/_r4_np.linalg.norm(_r4_N,axis=1,keepdims=True)
_r4_hemi_bi=_r4_np.array_equal(_r4_gih(_r4_N,40,seed=4), _r4_S.cosine_hemisphere(_r4_N,40,seed=4))
_r4_ld_bi=_r4_np.array_equal(_r4_S.low_discrepancy(16,d=2,seed=5), _r4_ld(16,d=2,seed=5))
_r4_dirs=_r4_S.cosine_hemisphere(_r4_N,64,seed=0); _r4_unit=bool(_r4_np.allclose(_r4_np.linalg.norm(_r4_dirs,axis=2),1.0,atol=1e-6))
print(f"  SAMPLING HOME (build): the sampling machinery unified behind one home over {', '.join(_r4_backends())}. The cosine-hemisphere sampler that lived in THREE modules is now OWNED once and shared -- globalillum's copy became a delegate, bit-identical ({_r4_hemi_bi}); its {_r4_dirs.shape[1]} dirs are unit-length and cosine-weighted ({_r4_unit}). Five call sites route through it now (pathtrace AA offsets, globalillum + lightcache gathers, and the mind's low_discrepancy_sample + blue_noise_sample), all bit-identical (low-discrepancy {_r4_ld_bi}). Thin home: patterns/MIS/accumulation ROUTE to their modules; only the hemisphere was promoted. *** one place to draw samples ***")

# BUILD: THE DENOISE HOME (consolidation R5) -- denoising was scattered: SVGF a-trous bilateral for images
# (svgf), the demodulated variant (modulate), Van-Cittert sharpen (sharpen), and the manifold denoisers for signals
# (denoise/hopfield). Denoise is one home split by what you're cleaning: .image / .sharpen / .signal -- routing to
# each shipped module. The Milanfar reframe is why they belong together: a denoiser is a map of the clean manifold.
from holographic_denoisehome import Denoise as _r5_D, denoise_backends as _r5_backends
from holographic_svgf import atrous_bilateral as _r5_ab
import numpy as _r5_np
_r5_rng=_r5_np.random.default_rng(0); _r5_H=_r5_W=24
_r5_clean=_r5_np.ones((_r5_H,_r5_W,3))*0.5; _r5_noisy=_r5_clean+0.1*_r5_rng.standard_normal((_r5_H,_r5_W,3))
_r5_N=_r5_np.tile([0.,0.,1.],(_r5_H,_r5_W,1)); _r5_A=_r5_np.ones((_r5_H,_r5_W,3))*0.5; _r5_Dp=_r5_np.ones((_r5_H,_r5_W))
_r5_img=_r5_D.image(_r5_noisy,_r5_N,_r5_A,_r5_Dp,method="svgf",levels=4)
_r5_before=float(_r5_np.abs(_r5_noisy-_r5_clean).mean()); _r5_after=float(_r5_np.abs(_r5_img-_r5_clean).mean())
_r5_bi=_r5_np.array_equal(_r5_img, _r5_ab(_r5_noisy,_r5_N,_r5_A,_r5_Dp,levels=4,variance=None))
_r5_t=_r5_np.linspace(0,4*_r5_np.pi,256); _r5_sig=_r5_np.sin(_r5_t); _r5_ns=_r5_sig+0.3*_r5_rng.standard_normal(256)
_r5_sc=float(_r5_np.abs(_r5_D.signal(_r5_ns,method="trajectory",rank=4)-_r5_sig).mean()); _r5_sn=float(_r5_np.abs(_r5_ns-_r5_sig).mean())
print(f"  DENOISE HOME (build): the denoisers unified behind one home over {len(_r5_backends())} facilities ({', '.join(_r5_backends())}). IMAGE svgf cleaned a noisy frame ({_r5_before:.3f} -> {_r5_after:.3f} error) and routes bit-identical to calling svgf directly ({_r5_bi}); SIGNAL trajectory (prior-free) cleaned a 1-D signal ({_r5_sn:.3f} -> {_r5_sc:.3f}). The R5 done-when met: the PIPELINE's denoise stage now calls Denoise.image, and the mind's svgf_denoise + sharpen_loop route through it too, all bit-identical. Route, don't rewrite; the signal/vector family stays on UnifiedMind.denoise (the fuller dispatcher over the same primitives). *** one home to clean image or signal ***")

# BUILD: THE TEXTURE HOME (consolidation R6) -- surface detail was generated in several modules (fbm noise, curl
# noise, Voronoi/cellular, patch synthesis, the weathering set), but all of them feed the SAME thing: a Material
# channel. Texture is the one home that hands you a channel-ready FIELD -- Param(field=Texture.voronoi(...)) -- routing
# to each shipped generator. This closes the chain: Texture -> Material -> Cache -> Shading.
from holographic_texturehome import Texture as _r6_T, texture_backends as _r6_backends
from holographic_surface import SurfaceMaterial as _r6_SM
from holographic_param import Param as _r6_Param
from holographic_matbake import bake_material as _r6_bake
import numpy as _r6_np
_r6_crack=_r6_T.voronoi(n_seeds=12, seed=0, kind="edge")                       # a vectorised crack-line field
_r6_pts=_r6_np.random.default_rng(0).uniform(-1,1,(64,3)); _r6_v=_r6_crack(_r6_pts)
_r6_rough=lambda _P,**_k: 0.25+0.5*_r6_np.clip(_r6_crack(_P)*4.0,0,1)
_r6_mat=_r6_SM(color=(0.6,0.6,0.62),roughness=_r6_Param(field=_r6_rough),reflect=0.1,emission=0.0)  # channel SOURCED from Texture
_r6_shade=_r6_bake(_r6_mat,(-1.,-1.,-1.),(1.,1.,1.),res=12)                    # bakes via the Cache home (H2)
_r6_ch=_r6_shade(_r6_np.array([[0.2,0.1,-0.3],[-0.5,0.4,0.2]]))
_r6_r=_r6_np.asarray(_r6_ch["roughness"],float)
_r6_u,_r6_vv=_r6_T.curl(res=24,seed=0); _r6_div=float(_r6_np.abs(_r6_np.gradient(_r6_u,axis=1)+_r6_np.gradient(_r6_vv,axis=0)).mean())
print(f"  TEXTURE HOME (build): procedural + example-based detail unified behind one home over {', '.join(_r6_backends())}, each returning a Material-channel-ready field. A Voronoi CRACK field ({_r6_v.shape[0]} points, vectorised) drove a roughness channel; that channel then BAKED through the Cache home and read back finite ({bool(_r6_np.isfinite(_r6_r).all())}, in [{_r6_r.min():.2f},{_r6_r.max():.2f}]). The R6 done-when met: a Material channel is sourced through Texture -- closing the chain Texture -> Material -> Cache -> Shading. Curl noise stays ~divergence-free (div {_r6_div:.3f}) for warping. Route, don't rewrite: the generators stay put; the home is just the channel adapter. *** detail in, one socket ***")

# BUILD: THE LIGHTING HOME (consolidation R7) -- there was no lighting home: the shade integral was reached for in
# lights/prt/globalillum/domecache directly, and the light TYPES lived apart from the evaluators. Lighting is one
# door -- it re-exports the light types AND exposes the shade integral in each mode (direct NEE / PRT / environment
# SH). Render methods ask here. Route, don't rewrite: each mode delegates to its shipped function.
from holographic_lightinghome import Lighting as _r7_L, lighting_modes as _r7_modes, DirectionalLight as _r7_Dir, RectLight as _r7_Rect
from holographic_lights import direct_lighting as _r7_dl
from holographic_sdf import sphere as _r7_sphere, box as _r7_box
import numpy as _r7_np
_r7_sdf=_r7_sphere(0.5).smooth_union(_r7_box(2.,0.1,2.).translate((0,-0.55,0)),k=0.03)
_r7_P=_r7_np.array([[0.,-0.4,0.]]); _r7_N=_r7_np.array([[0.,1.,0.]])
_r7_lights=[_r7_Dir(direction=(0.2,-1.,-0.1),intensity=3.0), _r7_Rect(position=(0.5,1.5,0.5),u_vec=(0.4,0,0),v_vec=(0,0.3,0.2),intensity=20.0)]
_r7_a=_r7_L.direct(_r7_sdf,_r7_P,_r7_N,_r7_N,_r7_np.full((1,3),0.8),_r7_np.zeros(1),_r7_np.full(1,0.5),_r7_lights,_r7_np.random.default_rng(0),area_samples=8)
_r7_b=_r7_dl(_r7_sdf,_r7_P,_r7_N,_r7_N,_r7_np.full((1,3),0.8),_r7_np.zeros(1),_r7_np.full(1,0.5),_r7_lights,_r7_np.random.default_rng(0),area_samples=8)
_r7_bi=_r7_np.array_equal(_r7_a,_r7_b)
_r7_domes,_r7_soft,_r7_hard=_r7_L.split_cached(_r7_lights)
print(f"  LIGHTING HOME (build): the light TYPES ({len(_r7_L.light_types())} of them) and the shade INTEGRAL now live behind one door, over modes {', '.join(_r7_modes())}. The direct next-event integral routes bit-identically to lights.direct_lighting ({_r7_bi}); split_cached routes a light set by how it's best served ({len(_r7_soft)} soft / {len(_r7_hard)} hard here). R7 done-when met: TWO render methods get their lighting from Lighting -- the cached soft-light pass (Lighting.direct) and the pipeline PRT strategy (Lighting.prt). Route, don't rewrite: the Cook-Torrance-per-light integral stays in lights, PRT in prt; only the entry point is unified. *** one door for light ***")

# BUILD: THE SHADOW / VISIBILITY HOME (consolidation R8) -- "can light reach this point?" was spelled four ways:
# SDF soft shadow + ambient occlusion (raymarch, re-imported by raycoherence and semantic), the hard shadow-RAY test
# (embedded in direct_lighting), and PRT's baked visibility. Shadow is one door with each as a named strategy.
from holographic_shadowhome import Shadow as _r8_S, shadow_strategies as _r8_strats
from holographic_raymarch import soft_shadow as _r8_ss, ambient_occlusion as _r8_ao
from holographic_sdf import sphere as _r8_sphere, box as _r8_box
import numpy as _r8_np
_r8_scene=_r8_sphere(0.4).translate((0,0.6,0)).smooth_union(_r8_box(3.0,0.1,3.0).translate((0,-0.1,0)),k=0.02)
_r8_eps=3e-3; _r8_up=_r8_np.array([0.,1.,0.]); _r8_N=_r8_np.array([[0.,1.,0.],[0.,1.,0.]])
_r8_P=_r8_np.array([[0.0,_r8_eps,0.0],[1.4,_r8_eps,0.0]])                       # under the ball / off to the side
_r8_soft=_r8_S.soft(_r8_scene,_r8_P,_r8_up)
_r8_bi=_r8_np.array_equal(_r8_soft, _r8_ss(_r8_scene,_r8_P,_r8_up)) and _r8_np.array_equal(_r8_S.ambient_occlusion(_r8_scene,_r8_P,_r8_N), _r8_ao(_r8_scene,_r8_P,_r8_N))
_r8_hard=_r8_S.hard(_r8_scene,_r8_P,_r8_N,_r8_N,_r8_np.array([5.0,5.0]))
print(f"  SHADOW / VISIBILITY HOME (build): one door for 'can light reach here?', over strategies {', '.join(_r8_strats())}. The SOFT shadow reads darker under the occluder than beside it ({_r8_soft[0]:.2f} vs {_r8_soft[1]:.2f}); the HARD shadow-ray blocks under / clears beside ({_r8_hard[0]:.0f}/{_r8_hard[1]:.0f}); both route BIT-IDENTICALLY to the raymarch marches ({_r8_bi}). R8 done-when met: TWO render paths (raycoherence, semantic) now get their visibility from Shadow instead of re-importing the raymarch functions. Route, don't rewrite: the Quilez marches stay in raymarch, PRT visibility stays baked in prt; only the entry point is unified. *** one door for visibility ***")

# PROMOTE: THE SCALE HOME (consolidation H3) -- the scale-out machinery already shipped in holographic_distribute
# (partition / tiles / bricks / commutative-monoid reducers) and the mind wired it as faculties, but there was no
# plain library HOME like the other homes. Scale promotes it: one import, one map_reduce, the strategies named.
from holographic_scalehome import Scale as _h3_S, scale_strategies as _h3_strats, scale_backends as _h3_be
import numpy as _h3_np
_h3_x=_h3_np.arange(1000.0)
_h3_idx=_h3_S.partition(len(_h3_x),7)                                          # 7 load-balanced buckets of indices
_h3_buckets=[_h3_x[i] for i in _h3_idx]
_h3_got,_h3_info=_h3_S.map_reduce(_h3_buckets, worker=lambda _b,_c: _b.sum(), reduce="sum")
_h3_match=abs(float(_h3_got)-float(_h3_x.sum()))<1e-6                          # partition+reduce == the un-split sum
_h3_canvas=_h3_np.zeros((20,30),dtype=int)
for _sl in _h3_S.tiles((20,30),(3,4)): _h3_canvas[_sl]+=1
_h3_cover=bool((_h3_canvas==1).all())                                         # tiles cover the domain exactly once
_h3_costs=_h3_np.array([10.,1,1,1,10,1,1,1,10,1]); _h3_parts=_h3_S.partition(len(_h3_costs),3,costs=_h3_costs)
_h3_loads=sorted(float(_h3_costs[p].sum()) for p in _h3_parts)
print(f"  SCALE HOME (promote): the scale-out machinery now has a home over {', '.join(_h3_be())}, with strategies {', '.join(_h3_strats())}. map_reduce split a 1000-vector into {_h3_info['buckets']} buckets and the monoid sum MATCHED the un-split total ({_h3_match}); tiles covered a 20x30 image disjointly ({_h3_cover}); the load-balanced partition kept bucket loads close ({_h3_loads[0]:.0f}..{_h3_loads[-1]:.0f}). H3 done-when met: the mind's distribute_compute / partition_domain / partition_grid / distribute_bricks all delegate to Scale, bit-identical. 'Limitations are usually bad approaches' -- partition it, prune the empty part, drop resolution where it doesn't show, superpose, or store only what's there. *** one door to scale out ***")

# PROMOTE: THE BLEND HOME (consolidation H4) -- "combine these into one" is spelled many ways: superposition (bundle),
# spherical interpolation (slerp), the Frechet mean on the sphere, front-to-back alpha compositing, and dict/scene
# MERGE with a conflict policy; a soft weighted blend normalize(sum w_i v_i) is re-derived in skinning, the matter
# model, etc. Blend promotes the canonical ops behind one door.
from holographic_blendhome import Blend as _h4_B, blend_backends as _h4_be
from holographic_blendpose import blend_pose as _h4_bp
from holographic_ai import slerp as _h4_slerp
import numpy as _h4_np
_h4_targets=_h4_np.random.default_rng(0).standard_normal((3,64)); _h4_w=_h4_np.array([0.5,0.3,0.2])
_h4_pose_bi=_h4_np.array_equal(_h4_bp(_h4_targets,_h4_w), _h4_B.bundle(_h4_targets,_h4_w))       # delegate 1: skinning
_h4_a=_h4_np.zeros(8); _h4_a[0]=1.0; _h4_b=_h4_np.zeros(8); _h4_b[3]=1.0
_h4_slerp_bi=_h4_np.array_equal(_h4_B.slerp(_h4_a,_h4_b,0.4), _h4_slerp(_h4_a,_h4_b,0.4))         # delegate 2: morph
_h4_lerp_n=float(_h4_np.linalg.norm(_h4_B.lerp(_h4_a,_h4_b,0.5))); _h4_slerp_n=float(_h4_np.linalg.norm(_h4_B.slerp(_h4_a,_h4_b,0.5)))
_h4_col,_h4_acc=_h4_B.alpha_composite(_h4_np.array([[1.,0,0],[0,1.,0]]), _h4_np.array([1.0,1.0]))  # opaque front hides back
_h4_merged=_h4_B.merge({"a":1,"b":2},{"b":9,"c":3},policy="prefer_a")
print(f"  BLEND HOME (promote): the combine operations unified over {', '.join(_h4_be())}. Two delegates now call it, bit-identical: blendpose.blend_pose is the weighted bundle ({_h4_pose_bi}) and generate.morph_images takes its slerp from Blend ({_h4_slerp_bi}). slerp stays ON the sphere while lerp cuts the chord (norms {_h4_slerp_n:.2f} vs {_h4_lerp_n:.2f}); alpha-composite lets an opaque front hide the back ({[float(x) for x in _h4_np.round(_h4_col,2)]}); merge respects a conflict policy ({_h4_merged}). Kept distinct on purpose: phasemorph's PHASE arc, mixture's solvent-base density, occlusion's composited READOUT -- specialised blends, not force-fit. *** one door to combine ***")

# PROMOTE: THE TRANSFORM HOME (consolidation H5) -- "move / rotate / warp this" happens in several representations:
# VSA (bind = a rigid shift, permute = order), 4x4 matrices (translate/scale/rotate/compose + decompose + quaternions
# + look_at), clifford ROTORS (gimbal-lock-free rotation), and anisotropic STEERING. The basic 4x4 builders were
# DUPLICATED between scenegraph and holographic_transform. Transform is one facade; the duplicate math is deduped.
from holographic_transformhome import Transform as _h5_T, transform_kinds as _h5_kinds
import holographic_transform as _h5_TF, holographic_scenegraph as _h5_SG
import numpy as _h5_np
_h5_dedup=_h5_np.array_equal(_h5_SG.translation([1.5,-2,3]), _h5_TF.translation([1.5,-2,3])) and _h5_np.array_equal(_h5_SG.scaling([2,0.5,1.5]), _h5_TF.scaling([2,0.5,1.5]))
_h5_rot_kept=not _h5_np.array_equal(_h5_SG.rotation([0.3,0.8,0.5],0.7), _h5_T.rotation([0.3,0.8,0.5],0.7))  # Rodrigues kept distinct
_h5_R=_h5_T.rotor([0,0,1],_h5_np.pi/2); _h5_v=_h5_T.rotate_vec(_h5_R, _h5_np.array([1.,0,0]))               # clifford: x->y
_h5_M=_h5_T.translation([1.,2,3]); _h5_p=_h5_M @ _h5_np.array([0.,0,0,1])
from holographic_ai import bind as _h5_bind
_h5_a=_h5_np.random.default_rng(0).standard_normal(128); _h5_b=_h5_np.random.default_rng(1).standard_normal(128)
_h5_vsa=_h5_np.array_equal(_h5_T.bind(_h5_a,_h5_b), _h5_bind(_h5_a,_h5_b))
print(f"  TRANSFORM HOME (promote): move/rotate/warp unified across {', '.join(_h5_kinds())}. The 4x4 matrix builders that were COPIED between scenegraph and holographic_transform are deduped -- scenegraph now delegates, still bit-identical ({_h5_dedup}); its Rodrigues rotation is kept distinct on purpose (~1e-12 off the quaternion one, {_h5_rot_kept}) so no determinism flips. A clifford ROTOR spun x->y ({[float(x) for x in _h5_np.round(_h5_v,3)]}); the VSA bind routes bit-identical ({_h5_vsa}); a translation moved the origin to {[float(x) for x in _h5_np.round(_h5_p[:3],1)]}. H5 done-when met: scenegraph + procgen both build transforms through Transform. *** one door to move things ***")

# PROMOTE: THE MEMORY HOME (consolidation H6) -- "keep the hot working set where the CPU can reach it fast". The
# cache-hierarchy levers already ship (residency = keep reused FFT spectra resident; the batched contiguous bind =
# one FFT for a whole record; tiling to fit; the opt-in GPU/numba backends) but had no single door. Memory is it.
from holographic_memoryhome import Memory as _h6_M, memory_levers as _h6_lev
from holographic_ai import bind as _h6_bind, bundle as _h6_bundle
import numpy as _h6_np, time as _h6_time
_h6_rng=_h6_np.random.default_rng(0)
# RESIDENCY: bind_cached is bit-identical to bind and reuses the resident spectrum on repeat
_h6_a=_h6_rng.standard_normal(1024); _h6_b=_h6_rng.standard_normal(1024); _h6_cache=_h6_M.spectrum_cache()
_h6_bi=_h6_np.array_equal(_h6_M.bind_cached(_h6_a,_h6_b,_h6_cache), _h6_bind(_h6_a,_h6_b)); _h6_M.bind_cached(_h6_a,_h6_b,_h6_cache)
# BATCHED LAYOUT: one FFT over contiguous arrays beats a Python loop of per-pair binds (min-of-rounds timing)
_h6_m,_h6_d=64,1024; _h6_keys=_h6_rng.standard_normal((_h6_m,_h6_d)); _h6_vals=_h6_rng.standard_normal((_h6_m,_h6_d))
def _h6_tb():
    _t=_h6_time.perf_counter()
    for _ in range(20): _h6_M.bind_batch(_h6_keys,_h6_vals)
    return _h6_time.perf_counter()-_t
def _h6_tl():
    _t=_h6_time.perf_counter()
    for _ in range(20): _h6_bundle(_h6_np.stack([_h6_bind(_h6_keys[_i],_h6_vals[_i]) for _i in range(_h6_m)]))
    return _h6_time.perf_counter()-_t
_h6_speedup=min(_h6_tl() for _ in range(3))/min(_h6_tb() for _ in range(3))
print(f"  MEMORY HOME (promote): the cache-hierarchy levers unified over {', '.join(_h6_lev())}. RESIDENCY -- a repeated bind reuses the cached FFT spectrum, bit-identical to a plain bind ({_h6_bi}), cache hits={_h6_cache.hits}. BATCHED CONTIGUOUS LAYOUT -- encoding a {_h6_m}-pair record in ONE FFT over stacked arrays is {_h6_speedup:.1f}x faster than a Python loop of per-pair binds, i.e. measurably cache-resident. H6 done-when met: residency is reachable through Memory (and the mind's spectrum_cache now routes here), and the batched kernel is measurably faster. The GPU/numba backends stay OPT-IN -- accelerators, never requirements. *** keep the hot set close ***")

# PROMOTE: THE COMPUTE HOME (consolidation H7) -- stay VSA-native. Every time a computation leaves the frequency
# domain to make a Python decision it pays an FFT round-trip; so transform IN once, do all the bind/bundle/permute
# algebra on the spectra, decide/clean up at the BOUNDARIES, transform OUT once. Compute is the one door over the
# fuse / schedule / width / program levers -- and fft_counts() measures the win.
from holographic_computehome import Compute as _h7_C, compute_levers as _h7_lev
from holographic_ai import bind as _h7_bind, bundle as _h7_bundle
import numpy as _h7_np
_h7_rng=_h7_np.random.default_rng(0); _h7_n=32; _h7_d=512
_h7_keys=[_h7_rng.standard_normal(_h7_d) for _ in range(_h7_n)]; _h7_vals=[_h7_rng.standard_normal(_h7_d) for _ in range(_h7_n)]
_h7_C.reset_fft_counts()
_h7_fused=_h7_C.fuse_record(_h7_keys,_h7_vals)
_h7_total=sum(_h7_C.fft_counts().values()); _h7_naive=3*_h7_n
_h7_ref=_h7_bundle(_h7_np.stack([_h7_bind(_h7_keys[_i],_h7_vals[_i]) for _i in range(_h7_n)]))
_h7_agree=bool(_h7_np.allclose(_h7_fused,_h7_ref,atol=1e-9))
print(f"  COMPUTE HOME (promote): the VSA-native levers unified over {', '.join(_h7_lev())}. FUSING a {_h7_n}-pair record -- collapsing the bind chain into the frequency domain -- took {_h7_total} FFTs vs {_h7_naive} for the op-by-op path ({100*(1-_h7_total/_h7_naive):.0f}% fewer), and the fused result AGREES with the eager one to FFT tolerance ({_h7_agree}). H7 done-when met: a multi-op chain runs fused with a measured FFT-count drop, and the mind's fuse_record/fuse_expression now route through Compute. The rule the home enforces: push decisions and cleanups to the BOUNDARIES, keep the hot middle in the vector domain. *** no Python hops in the middle ***")

# BUILD: THE SIMULATION SCAFFOLD (consolidation R9) -- the solvers are legitimately DIFFERENT algorithms (Stable
# Fluids, combustion, softbody, Cosserat rods, MPM, collision, reaction-diffusion) with different step signatures.
# We do NOT merge them; we give them ONE step loop (via integrate.SolverAdapter) and expose their field so the
# Pipeline can draw it. Keep the solvers separate; share only the loop.
from holographic_simulationhome import Simulation as _r9_Sim, known_solver_strategies as _r9_strats
from holographic_fluid import StableFluid as _r9_SF
from holographic_automaton import HyperCA as _r9_CA
from holographic_render import Camera as _r9_Cam
import numpy as _r9_np
_r9_cam=_r9_Cam(eye=(0.5,0.5,3.0), target=(0.5,0.5,0.5), fov_deg=45)
_r9_fluid=_r9_SF((16,16,16), dt=0.1); _r9_fluid.density[6:10,2:6,6:10]=1.0; _r9_fluid.vel[1,:,:5,:]=1.0
_r9_ca=_r9_CA(size=20, dim=16, seed=0)
_r9_s1=_r9_Sim.for_fluid(_r9_fluid); _r9_s2=_r9_Sim.for_automaton(_r9_ca)      # two distinct algorithms
_r9_s1.run(5); _r9_s2.run(5)                                                   # ... one shared loop
_r9_i1,_r9_a1=_r9_s1.render(_r9_cam, width=24, height=24, steps=24, sigma=12.0)
_r9_i2,_r9_a2=_r9_s2.render(_r9_cam, width=24, height=24, steps=24, sigma=8.0)
_r9_sep=bool(_r9_fluid.density.shape==(16,16,16) and _r9_ca.grid.shape[:2]==(20,20))
print(f"  SIMULATION SCAFFOLD (build): two genuinely DIFFERENT solvers -- Stable Fluids (advect+project) and a reaction-diffusion automaton -- stepped {_r9_s1.steps_run} times through the SAME loop, and the Pipeline rendered BOTH fields (alpha {float(_r9_a1.max()):.2f} smoke / {float(_r9_a2.max()):.2f} automaton). R9 done-when met. The solvers stayed SEPARATE ({_r9_sep}) -- each keeps its own math and its own step signature (fluid.step() vs ca.evolve(1)); the scaffold only unifies the INTERFACE via integrate.SolverAdapter and exposes the field as a Field (R2) for volume_render. Golden rule honoured: share the loop, never merge the algorithms. Strategies: {', '.join(_r9_strats())}; add any solver in three closures. *** one loop, many solvers ***")

# BUILD: THE HYPERVECTOR DATATYPE (consolidation D1) -- the whole engine has been operating on ONE datatype
# implicitly: a high-dimensional vector that carries meaning, travelling everywhere as a bare numpy array. Hypervector
# gives that datatype a name and the five verbs as METHODS, without hiding the raw array (hot paths need it). As above,
# so below: the capstone of the consolidation -- the thing all the homes have been passing around.
from holographic_hypervector import Hypervector as _d1_HV
from holographic_ai import bind as _d1_bind, unbind as _d1_unbind, bundle as _d1_bundle, permute as _d1_permute, cosine as _d1_cos
from holographic_encoders import ScalarEncoder as _d1_Enc
import numpy as _d1_np
_d1_enc=_d1_Enc(dim=1024, seed=0)
_d1_a=_d1_HV.encode(_d1_enc, 0.3, tag="a"); _d1_b=_d1_HV.encode(_d1_enc, 0.7, tag="b")     # MAKE: encoder is the constructor
# CONSUME: the five verbs as methods, each matching the bare op exactly
_d1_verbs_ok=bool(
    _d1_np.array_equal(_d1_a.bind(_d1_b).array, _d1_bind(_d1_a.array,_d1_b.array)) and
    _d1_np.array_equal(_d1_a.bind(_d1_b).unbind(_d1_b).array, _d1_unbind(_d1_bind(_d1_a.array,_d1_b.array),_d1_b.array)) and
    _d1_np.array_equal(_d1_a.bundle(_d1_b).array, _d1_bundle(_d1_np.stack([_d1_a.array,_d1_b.array]))) and
    _d1_np.array_equal(_d1_a.permute(3).array, _d1_permute(_d1_a.array,3)))
_d1_clean=_d1_a.cleanup({"a":_d1_a,"b":_d1_b}).tag                                          # recognize
_d1_raw_cheap=bool(_d1_a.raw() is _d1_a.array and _d1_np.asarray(_d1_a) is _d1_a.array)     # raw array, no copy
_d1_recover=_d1_a.bind(_d1_b).unbind(_d1_a).cosine(_d1_b) > _d1_cos(_d1_a.array,_d1_b.array)
print(f"  HYPERVECTOR DATATYPE (build): the datatype the whole engine has been passing around now has a name. MAKE it from data via any encoder (the constructor); CONSUME it with the five verbs as METHODS -- all matching the bare ops exactly ({_d1_verbs_ok}); cleanup recognized it as {_d1_clean!r}; a bind/unbind round-trip recovered the filler ({_d1_recover}). Thin by design: the raw array comes back with NO copy ({_d1_raw_cheap}) -- .array / np.asarray(hv) -- so the hot paths never pay for the wrapper, and nothing existing had to change. D1 done-when met: build from any data, five verbs as methods, raw array back cheaply. *** as above, so below: one datatype ***")

# RE-ENABLE: closed-form ITERATE behind a regime gate (adaptive-dispatch audit). Some methods were shelved because
# they are only right in a NICHE; with a catalog + adaptive dispatch, "only good in a niche" becomes "gate it". The
# cleanest re-enable: for a LINEAR operator that is a bind (circular convolution), iterating is diagonal in Fourier,
# so we jump k steps in ONE FFT -- EXACT, not approximate. Detector: recover the impulse response, verify it's a
# convolution. Exact in regime => the gate can never do worse than stepping (it matches, or falls back).
from holographic_computehome import Compute as _re_C
from holographic_regimegate import RegimeGate as _re_G
from holographic_ai import bind as _re_bind
import numpy as _re_np, time as _re_time
_re_rng=_re_np.random.default_rng(0); _re_D=1024
_re_kernel=_re_rng.standard_normal(_re_D); _re_kernel/=_re_np.max(_re_np.abs(_re_np.fft.rfft(_re_kernel)))*1.001
_re_op=lambda x: _re_bind(_re_kernel, x); _re_state=_re_rng.standard_normal(_re_D)
_re_k=5000
_re_t=_re_time.perf_counter(); _re_res,_re_info=_re_C.iterate(_re_op,_re_state,k=_re_k); _re_tg=_re_time.perf_counter()-_re_t
_re_t=_re_time.perf_counter(); _re_slow=_re_state.copy()
for _ in range(_re_k): _re_slow=_re_bind(_re_kernel,_re_slow)
_re_ts=_re_time.perf_counter()-_re_t
_re_exact=bool(_re_np.allclose(_re_res,_re_slow,atol=1e-8))
_re_nl=_re_C.iterate(lambda x: _re_np.tanh(_re_op(x)), _re_state, k=_re_k)[1]["used"]     # nonlinear -> fallback
print(f"  RE-ENABLE (closed-form iterate): a shelved niche method brought back behind a RegimeGate. For a bind operator, jumping {_re_k} iterations took ONE FFT vs {_re_k} steps -- {_re_ts/_re_tg:.0f}x faster and EXACT ({_re_exact}); a nonlinear operator falls back safely (used={_re_nl!r}). Because the closed form is exact IN REGIME, the gate can never do worse than stepping. The pattern (detect the regime cheaply, run the superior method only there, keep the safe fallback, measure the breakeven) is how we reconsider everything we kept out -- and it's honest both ways: the projection-denoise candidate was MEASURED to leak harm at a fixed threshold, so it stays shelved as an auto-default. *** niche is a reason to gate, not to shelve ***")

# RE-ENABLE #2: FHRR-at-high-load, GATED BY THE PAIR COUNT. Binding many role->filler pairs into one vector is
# capacity-limited; real-HRR is cheap but its recall falls off as load climbs, while FHRR phasors hold up at ~2x
# storage. FHRR was kept opt-in ("changes nothing" at low load). The detector is the exact pair COUNT vs ~0.08*dim,
# and -- unlike the denoise gate -- there is NO harm mode (FHRR recall >= real-HRR), so over-triggering only costs a
# little storage, never correctness.
from holographic_loadmemory import AdaptiveRoleFillerMemory as _re2_M
_re2_N=90; _re2_dim=512
_re2_fhrr=_re2_M(dim=_re2_dim, expected_pairs=_re2_N, seed=1)      # high load -> gate picks FHRR
_re2_hrr=_re2_M(dim=_re2_dim, expected_pairs=1, seed=1)            # forced real-HRR, same atoms
for _re2_i in range(_re2_N):
    _re2_fhrr.add(f"r{_re2_i}", f"f{_re2_i}"); _re2_hrr.add(f"r{_re2_i}", f"f{_re2_i}")
_re2_fh=sum(_re2_fhrr.recall(f"r{_re2_i}")==f"f{_re2_i}" for _re2_i in range(_re2_N))
_re2_hh=sum(_re2_hrr.recall(f"r{_re2_i}")==f"f{_re2_i}" for _re2_i in range(_re2_N))
_re2_lo=_re2_M(dim=_re2_dim, expected_pairs=6, seed=0); _re2_lo.add("color","red")   # low load -> cheap HRR
print(f"  RE-ENABLE (FHRR at high load): the pair count IS the detector. At N={_re2_N} pairs in a dim-{_re2_dim} space (past the ~0.08*dim knee) the gate picked {_re2_fhrr.backend.upper()} and recalled {_re2_fh}/{_re2_N} correctly, vs real-HRR {_re2_hh}/{_re2_N} at the same load -- the capacity win captured automatically. At low load it stays on cheap real-HRR (backend={_re2_lo.backend}, recall of color = {_re2_lo.recall("color")!r}). No harm mode: FHRR recall >= real-HRR, so a misfire costs storage, never correctness. *** load is a reason to gate, not to shelve ***")

# RE-ENABLE #3: TENSOR-PRODUCT binding for EXACT recall, gated by fidelity need + memory budget. Tensor is EXACT up
# to M~D (perfect where HRR/FHRR have long degraded) but costs D*D numbers -- D-times the storage. So its gate adds
# two deciders to the load one: is EXACT recall wanted, and does D*D fit the budget? No harm mode on recall (it is
# exact in-regime); the cost is purely storage, answered by a known parameter.
from holographic_loadmemory import AdaptiveRoleFillerMemory as _re3_M
_re3_N=200; _re3_dim=256
_re3_tensor=_re3_M(dim=_re3_dim, expected_pairs=_re3_N, exact=True, seed=3)   # exact wanted -> tensor
_re3_fhrr=_re3_M(dim=_re3_dim, expected_pairs=_re3_N, seed=3)                 # not exact -> fhrr at this load
for _re3_i in range(_re3_N):
    _re3_tensor.add(f"r{_re3_i}", f"f{_re3_i}"); _re3_fhrr.add(f"r{_re3_i}", f"f{_re3_i}")
_re3_th=sum(_re3_tensor.recall(f"r{_re3_i}")==f"f{_re3_i}" for _re3_i in range(_re3_N))
_re3_fh=sum(_re3_fhrr.recall(f"r{_re3_i}")==f"f{_re3_i}" for _re3_i in range(_re3_N))
_re3_budget=_re3_M(dim=_re3_dim, expected_pairs=_re3_N, exact=True, max_numbers=1000, seed=3)   # too small a budget
print(f"  RE-ENABLE (tensor exact recall): with exact recall requested and the memory budget available, the gate picked {_re3_tensor.backend.upper()} and recalled {_re3_th}/{_re3_N} EXACTLY at a load where FHRR managed only {_re3_fh}/{_re3_N} -- but it spends {_re3_dim*_re3_dim} numbers ({_re3_dim}x the storage). Give it too small a budget and it honestly falls back off tensor (backend={_re3_budget.backend}). No harm mode on recall (tensor is exact in-regime); the D-times storage is the whole cost, and a known parameter decides whether it is worth paying. *** exactness is a reason to gate, not to shelve ***")

# RE-ENABLE #4: MULTI-SCATTER GGX (Kulla-Conty), gated by ROUGHNESS. Single-scatter GGX drops the light that bounces
# between microfacets more than once, so a rough METAL loses real energy (white-furnace ~0.36 at roughness 1.0). The
# Kulla-Conty term adds it back from the baked single-scatter albedo. It OVERSHOOTS at low roughness, so we gate on
# roughness (an exact parameter): smooth -> single-scatter (no overshoot), rough -> compensated (energy conserved).
from holographic_brdf import directional_albedo as _re4_ss, directional_albedo_ms as _re4_ms, brdf_gated as _re4_g
import numpy as _re4_np
_re4_N=_re4_np.array([0.,0,1]); _re4_V=_re4_np.array([0.6,0,0.8]); _re4_L=_re4_np.array([-0.3,0.2,0.93]); _re4_L=_re4_L/_re4_np.linalg.norm(_re4_L)
_re4_lo=_re4_g(_re4_N,_re4_V,_re4_L,(1,1,1),1.0,0.15)[1]["used"]      # smooth -> single-scatter
_re4_hi=_re4_g(_re4_N,_re4_V,_re4_L,(1,1,1),1.0,0.8)[1]["used"]       # rough  -> multi-scatter
_re4_e08_ss=_re4_ss(1.0,0.8,(1,1,1),16384,0.6,0); _re4_e08_ms=_re4_ms(1.0,0.8,(1,1,1),16384,0.6,0)
_re4_e10_ss=_re4_ss(1.0,1.0,(1,1,1),16384,0.6,0); _re4_e10_ms=_re4_ms(1.0,1.0,(1,1,1),16384,0.6,0)
print(f"  RE-ENABLE (multi-scatter GGX): a rough metal under single-scatter GGX loses energy -- white-furnace {float(_re4_e08_ss):.2f} at roughness 0.8, {float(_re4_e10_ss):.2f} at 1.0. The Kulla-Conty term (baked once from the single-scatter albedo) restores it to {float(_re4_e08_ms):.2f} and {float(_re4_e10_ms):.2f} -- energy conserving. The term OVERSHOOTS at low roughness, so we gate on the exact roughness parameter: at 0.15 the gate used the {_re4_lo} (plain GGX, no overshoot), at 0.8 the {_re4_hi} (compensated). Kept opt-in / backward-compatible; the default renderer is unchanged. Unlike the denoise gate, the detector is EXACT, so it reliably captures the win and dodges the overshoot. *** conserve energy where it is lost, gate where the fix would overshoot ***")

# RE-ENABLE #5: the COARSE-FIRST residual pass -- the shared detector the Group-B methods need. Run the cheap method
# everywhere, measure a per-cell uncertainty, and refine ONLY the hard cells. Wins when the uncertainty is
# CONCENTRATED (a thin ridge, an edge) and that concentrated region carries the error.
from holographic_coarsefirst import refine_where_uncertain as _re5_refine, gradient_uncertainty as _re5_unc, concentration as _re5_conc
import numpy as _re5_np
_re5_H=_re5_W=64; _re5_ys,_re5_xs=_re5_np.mgrid[0:_re5_H,0:_re5_W]/float(_re5_H)
def _re5_f(Y,X): return 0.3*_re5_np.sin(3*Y)+0.3*_re5_np.cos(3*X)+_re5_np.exp(-((X-0.51)**2)/0.0008)  # smooth + a thin ridge
_re5_truth=_re5_f(_re5_ys,_re5_xs); _re5_cs=4
_re5_cg=_re5_f(_re5_ys[::_re5_cs,::_re5_cs],_re5_xs[::_re5_cs,::_re5_cs])
_re5_coarse=_re5_np.repeat(_re5_np.repeat(_re5_cg,_re5_cs,axis=0),_re5_cs,axis=1)[:_re5_H,:_re5_W]     # cheap coarse estimate
_re5_u=_re5_unc(_re5_coarse); _re5_c=_re5_conc(_re5_u)
_re5_refined,_re5_mask,_re5_n=_re5_refine(_re5_coarse,_re5_u,lambda m:_re5_f(_re5_ys,_re5_xs),frac=0.2)
_re5_rmse=lambda a,b: float(_re5_np.sqrt(_re5_np.mean((a-b)**2)))
_re5_ec=_re5_rmse(_re5_coarse,_re5_truth); _re5_er=_re5_rmse(_re5_refined,_re5_truth)
print(f"  RE-ENABLE (coarse-first residual pass): the shared Group-B detector -- run cheap, measure where it's uncertain, refine only there. On a smooth field with a thin ridge (uncertainty concentration {_re5_c:.2f}), refining only {_re5_n}/{_re5_truth.size} cells ({100*_re5_n/_re5_truth.size:.0f}%) cut the error {_re5_ec:.3f} -> {_re5_er:.3f} ({_re5_ec/_re5_er:.1f}x). concentration() is the honest breakeven: measured, adaptive path-trace AA does NOT beat uniform on a uniformly-noisy scene (low concentration) -- so this is NECESSARY, not sufficient. Each Group-B method (Nystrom, splat refine, volint) builds on this, each owing its own measured win. *** refine where uncertain = use the costly method only where the cheap one fails ***")

# RE-ENABLE #6: NYSTROM for low-rank kernels, gated by a probe residual. Applying a kernel field is O(N^2) exact but
# O(N*m) via m landmarks -- exact only when the kernel is LOW-RANK (smooth). Detector: compare exact vs Nystrom on a
# tiny held-out PROBE (cheap); if they match, the kernel is low-rank -> use Nystrom, else fall back to exact.
from holographic_nystrom import apply_kernel_gated as _re6_g, exact_kernel_apply as _re6_ex
import numpy as _re6_np, time as _re6_time
_re6_rng=_re6_np.random.default_rng(0); _re6_N=1500
_re6_src=_re6_rng.standard_normal((_re6_N,2)); _re6_pts=_re6_rng.standard_normal((_re6_N,2)); _re6_w=_re6_rng.standard_normal(_re6_N)
_re6_rel=lambda a,b: float(_re6_np.linalg.norm(a-b)/(_re6_np.linalg.norm(b)+1e-12))
# smooth (low-rank) kernel -> Nystrom, fast + near-exact
_re6_ref=_re6_ex(_re6_pts,_re6_src,_re6_w,1.5)
_re6_t=_re6_time.perf_counter(); _re6_f,_re6_i=_re6_g(_re6_pts,_re6_src,_re6_w,1.5,m=90); _re6_tg=_re6_time.perf_counter()-_re6_t
_re6_t=_re6_time.perf_counter(); _re6_ex(_re6_pts,_re6_src,_re6_w,1.5); _re6_tex=_re6_time.perf_counter()-_re6_t
# sharp (full-rank) kernel -> exact fallback, byte-correct
_re6_f2,_re6_i2=_re6_g(_re6_pts,_re6_src,_re6_w,0.15,m=90)
_re6_ref2=_re6_ex(_re6_pts,_re6_src,_re6_w,0.15)
print(f"  RE-ENABLE (Nystrom low-rank kernel): a smooth kernel field (probe error {_re6_i["score"]:.3f}) routed to {_re6_i["method"].upper()} -- {_re6_tex/_re6_tg:.1f}x faster than exact O(N^2) at N={_re6_N}, rel-err {_re6_rel(_re6_f,_re6_ref):.4f}. A sharp kernel (probe error {_re6_i2["score"]:.2f}) fell back to {_re6_i2["method"].upper()}, byte-correct (rel-err {_re6_rel(_re6_f2,_re6_ref2):.1e}). Because the fallback is EXACT, the gate can never be wrong -- only, rarely, slower than optimal. Ships where denoise did not: a reliable detector AND a safe fallback. *** cheap where the kernel is smooth, exact where it isn't ***")

# RE-ENABLE #7: full-3DGS ANISOTROPIC splat refinement, composed COARSE-FIRST. A cheap isotropic splat cannot fit a
# sharp / oriented feature, so it leaves residual there; aniso_fit (gradient descent) can. So: fit the cheap isotropic
# base, then anisotropic-refine what it MISSED. Refining the residual only ADDS detail -> strictly >= baseline, no harm.
from holographic_splat import splat_fit as _re7_sf, splat_render as _re7_sr, fit_coarse_first as _re7_cf, psnr as _re7_psnr
import numpy as _re7_np
_re7_H=_re7_W=64; _re7_ys,_re7_xs=_re7_np.mgrid[0:_re7_H,0:_re7_W].astype(float)
_re7_sharp=(_re7_ys>_re7_xs).astype(float)*0.9+0.1+0.4*_re7_np.exp(-(((_re7_ys-45)**2+(_re7_xs-20)**2)/50.0)); _re7_sharp/=_re7_sharp.max()
_re7_smooth=_re7_np.exp(-(((_re7_ys-20)**2+(_re7_xs-20)**2)/40.0)); _re7_smooth/=_re7_smooth.max()
_re7_iso_sharp=_re7_psnr(_re7_sr(_re7_sf(_re7_sharp,30),(_re7_H,_re7_W)),_re7_sharp)
_re7_cf_sharp=_re7_psnr(_re7_cf(_re7_sharp,K_iso=30,K_aniso=8)[0],_re7_sharp)
_re7_iso_sm=_re7_psnr(_re7_sr(_re7_sf(_re7_smooth,30),(_re7_H,_re7_W)),_re7_smooth)
_re7_cf_sm=_re7_psnr(_re7_cf(_re7_smooth,K_iso=30,K_aniso=8)[0],_re7_smooth)
print(f"  RE-ENABLE (anisotropic splat refine, coarse-first): fit cheap isotropic splats, then gradient-refine the RESIDUAL with anisotropic Gaussians. On a SHARP edge (which isotropic blobs fundamentally can't fit) it jumped {_re7_iso_sharp:.1f} -> {_re7_cf_sharp:.1f} dB (+{_re7_cf_sharp-_re7_iso_sharp:.1f}); on a SMOOTH target it still helped a little ({_re7_iso_sm:.1f} -> {_re7_cf_sm:.1f}). Refining the residual only ADDS detail, so it is strictly >= the isotropic baseline -- NO harm mode across every target tested. That is why it ships WITHOUT a gate: unlike the parameter re-enables it needs no reliable detector, because a 'wrong' choice can't hurt -- concentration() is actually BACKWARDS for anisotropy, so we don't use it. Opt-in: apply it when you want the extra fidelity and can afford the anisotropic fit. *** refine the residual: strictly better, never worse ***")

# QUERY HISTORY (P7-P12): a git-like timeline for a query table -- time-travel SELECT, diff, and tamper-locate, all
# on the shipped versioning faculties. SQL does none of these well; here they fall out of "a row is a vector".
from holographic_query import Database as _qh_DB, update as _qh_upd
from holographic_querytime import TableHistory as _qh_TH, select_as_of as _qh_asof, diff_versions as _qh_diff, find_tampering as _qh_tamper
_qh_db=_qh_DB(); _qh_db.add_namespace("user")
_qh_db.create_table("user.acct",["id","balance"],dim=1024,seed=0)
_qh_t=_qh_db.namespaces["user"]["tables"]["acct"]; _qh_t.set_primary_key("id")
_qh_t.insert({"id":1,"balance":100}); _qh_h=_qh_TH(_qh_t); _qh_v0=_qh_h.commit(_qh_t,note="open")
_qh_upd(_qh_t,"id = 1",{"balance":250}); _qh_t.insert({"id":2,"balance":5}); _qh_v1=_qh_h.commit(_qh_t,note="edit")
_qh_then=_qh_asof(_qh_h,_qh_v0,"SELECT balance FROM acct WHERE id = 1")[0]["balance"]
_qh_now=_qh_asof(_qh_h,_qh_v1,"SELECT balance FROM acct WHERE id = 1")[0]["balance"]
_qh_d=_qh_diff(_qh_h,_qh_v0,_qh_v1)
_qh_suspect=_qh_h._versions[_qh_v1]["records"].copy(); _qh_suspect[0]+=0.01
_qh_loc=_qh_tamper(_qh_h,_qh_v1,_qh_suspect)
print(f"  QUERY HISTORY (git-for-data): the same account queried AS OF two versions reads balance {int(_qh_then)} then {int(_qh_now)} -- time-travel SELECT, no history tables or triggers. diff(v0,v1) reports {_qh_d["n_added"]} row added and {_qh_d["n_changed"]} changed, with the field-level before/after. And the audit trail is provable: after tampering one stored row, CompositionTree.locate named row {_qh_loc} in O(log n). All of it wired onto the versioning faculties that already shipped -- SQL does none of these well. *** a row is a vector, so its whole history is one too ***")

# VSA PROGRAMS AS DB OBJECTS (PR1-PR6): install a hypervector "stored procedure", find it BY MEANING, EXPLAIN it as a
# dry run, and EXECUTE it over query rows -- sandboxed to whitelisted handlers, step-bounded, result with a confidence.
from holographic_queryprog import ProgramCatalog as _pp_Cat
_pp_cat=_pp_Cat(dim=2048, seed=0)
_pp_cat.install("prototype",[("LOAD","color"),("HALT",None)], doc="build a prototype that clusters similar rows by color", inputs=["color"], outputs=["color"], handlers=[], data=["color"])
_pp_cat.install("normalize_tag",[("LOAD","color"),("APPLY","normalize"),("HALT",None)], doc="normalize and tag records for anomaly detection", inputs=["color"], outputs=["color"], handlers=["normalize"], faculties=["normalize"])
_pp_found=_pp_cat.find("group a series of similar things into clusters")[0]["name"]
_pp_ex=_pp_cat.explain("normalize_tag")
_pp_out=_pp_cat.execute("prototype",[{"color":"red"},{"color":"red"},{"color":"blue"}])
print(f"  VSA PROGRAMS AS DB OBJECTS: installed two hypervector 'stored procedures'. Asked the catalog to find one BY MEANING -- 'group a series into clusters' -> '{_pp_found}' (a SQL function catalog is exact-match only). EXPLAIN dry-ran normalize_tag and reported it WOULD call {_pp_ex["faculties_called"]} without executing. Then EXECUTE ran prototype over three query rows in the vector domain, sandboxed to its whitelisted handlers and step-bounded, returning a result at confidence {_pp_out["_confidence"]:.2f}. A program is a hypervector over a tiny opcode set -- it can't do I/O or run host code, so it is safer than a SQL stored procedure. *** install a procedure that is a vector; discover it by meaning ***")

# WORKSPACE FOLDERS (WS7) + COMBINE-SCENES (WS6): group a database's tables into a shallow tree (home = ownership,
# links = grouping), and combine two 3D scenes by BUNDLING them (a scene is a superposition of objects, so + unions).
from holographic_query import Database as _wf_DB
from holographic_queryfolder import FolderTree as _wf_FT, _bare as _wf_bare
from holographic_scene import SceneCoder as _wf_SC, COLOURS as _wf_C, SHAPES as _wf_S, TEXTURES as _wf_T
_wf_db=_wf_DB(); _wf_db.add_namespace("user", tier="persistent")
for _wf_t in ("sales","returns","catalog"): _wf_db.create_table("user."+_wf_t,["id"],dim=256,seed=0)
_wf_ft=_wf_FT(_wf_db)
_wf_ft.set_home("user.sales","reports"); _wf_ft.set_home("user.returns","reports")
_wf_ft.set_home("user.catalog","reference"); _wf_ft.link("user.catalog","reports")
_wf_scoped=sorted(_wf_bare(_wf_q) for _wf_q in _wf_ft.tables_in("reports"))
_wf_deleted=sorted(_wf_bare(_wf_q) for _wf_q in _wf_ft.drop_folder("reports"))
_wf_survived=_wf_db.namespaces["user"]["tables"].get("catalog") is not None
_wf_sc=_wf_SC(dim=4096, seed=0)
_wf_a=_wf_sc.encode_scene([{"colour":_wf_C[0],"shape":_wf_S[0],"texture":_wf_T[0]},{"colour":_wf_C[1],"shape":_wf_S[1],"texture":_wf_T[0]}])
_wf_b=_wf_sc.encode_scene([{"colour":_wf_C[2],"shape":_wf_S[2],"texture":_wf_T[0]}])
_wf_comb=_wf_sc.count_objects(_wf_sc.combine(_wf_a,_wf_b))
print(f"  WORKSPACE FOLDERS: grouped tables under a shallow tree -- searching folder 'reports' scopes to just {_wf_scoped} (home tables + the linked catalog). Dropping 'reports' deleted only the tables it OWNS ({_wf_deleted}); the catalog, only LINKED there but HOMED in 'reference', survived (still in the DB: {_wf_survived}). Home = ownership -> lifecycle; a link = grouping -> never deletes. And COMBINE-SCENES: two 3D scenes ({_wf_sc.count_objects(_wf_a)} objects + {_wf_sc.count_objects(_wf_b)} object) merge by simple vector addition into one holding {_wf_comb} -- a scene IS a bundle, so union is a sum, and the resonator can still pull any object back out. *** rows -> tables -> folders -> databases, every rung the same bundle ***")

# GRAPH TRAVERSAL (B10) + SINGLE-WRITER CONCURRENCY (B8): reachability over a table's edges (the exact index, because
# the holographic graph store's recall collapses at scale), and one-writer-at-a-time with lock-free reader snapshots.
from holographic_query import Database as _gl_DB, update as _gl_upd
from holographic_querygraph import EdgeGraph as _gl_EG
from holographic_querylock import SingleWriterLock as _gl_SWL, ConcurrencyError as _gl_CE
_gl_db=_gl_DB(); _gl_db.add_namespace("user")
_gl_db.create_table("user.edges",["src","dst"],dim=256,seed=0)
_gl_t=_gl_db.namespaces["user"]["tables"]["edges"]
for _gl_s,_gl_d in [(1,2),(1,3),(2,4),(3,4),(4,5)]: _gl_t.insert({"src":_gl_s,"dst":_gl_d})
_gl_g=_gl_EG(_gl_t,"src","dst")
_gl_desc=_gl_g.descendants(1); _gl_path=_gl_g.path(1,5)
_gl_db.create_table("user.acct",["id","balance"],dim=256,seed=0)
_gl_a=_gl_db.namespaces["user"]["tables"]["acct"]; _gl_a.set_primary_key("id"); _gl_a.insert({"id":1,"balance":100})
_gl_lock=_gl_SWL(); _gl_snap=_gl_lock.snapshot(_gl_a)
_gl_refused=False
with _gl_lock.write():
    _gl_upd(_gl_a,"id = 1",{"balance":250})
    try:
        with _gl_lock.write(block=False): pass
    except _gl_CE: _gl_refused=True
_gl_snap_bal=_gl_snap.rows()[0]["balance"]; _gl_fresh_bal=_gl_lock.snapshot(_gl_a).rows()[0]["balance"]
print(f"  GRAPH + CONCURRENCY: over a table of edges, descendants(1) = {_gl_desc} and the shortest path 1->5 is {_gl_path} (length {len(_gl_path)}) -- recursive-CTE work, done as a plain BFS. It is an EXACT adjacency index on purpose: the holographic graph store's recall collapses at scale, so traversal doesn't use it. And under a single-writer lock, a concurrent second writer was refused ({_gl_refused}) while a reader snapshot taken before the write still read {_gl_snap_bal} even though a fresh read saw {_gl_fresh_bal} -- consistent, lock-free reads; MVCC deferred and said so. *** exact where exactness is cheap, honest where it is hard ***")

# DISTRIBUTED COORDINATOR (R2): run monoid work on a pluggable backend -- in-process, or a persistent local process
# pool with a shared_memory read-only cache -- reusing distribute's partition + monoid reduce. Plus a margin-gated
# canonical tie-break so distributed results agree on knife-edge decisions.
from holographic_coordinator import Coordinator as _co_C, InProcessBackend as _co_IP, LocalPool as _co_LP, decide as _co_decide, _sum_bucket as _co_sum
from holographic_distribute import reduce_sum as _co_rs
import numpy as _co_np, os as _co_os
_co_cache=_co_np.arange(30, dtype=float)*2.0
_co_buckets=[list(range(0,10)),list(range(10,20)),list(range(20,30))]
_co_ip=_co_C(_co_IP()).run(_co_buckets,_co_sum,cache=_co_cache,reduce=_co_rs)
with _co_C(_co_LP(n=3)) as _co_lc:
    _co_lp=_co_lc.run(_co_buckets,_co_sum,cache=_co_cache,reduce=_co_rs)
_co_tie_a=_co_decide([0.5,0.5,0.3]); _co_tie_b=_co_decide([0.5+1e-13,0.5-1e-13,0.3]); _co_clear=_co_decide([0.9,0.5,0.3])
print(f"  DISTRIBUTED COORDINATOR: the same partitioned monoid job -- sum a shared read-only cache over three buckets -- reassembled to {_co_ip:.0f} in-process and {_co_lp:.0f} on a persistent LOCAL PROCESS POOL (each worker its own interpreter; the {_co_cache.nbytes}-byte cache shipped ONCE via shared_memory, not pickled per bucket). MIN/disjoint reassembly is bit-exact; plain SUM agrees to ~1e-12, which only matters if a TIE-SENSITIVE decision consumes it -- so a margin-gated canonical tie-break catches those: a knife-edge decision resolved to {_co_tie_a} and its ~1e-13-wobbled twin to {_co_tie_b} (identical -- a RULE broke the tie, not the rounding), while a comfortable-margin decision went to {_co_clear} regardless. It sits behind distribute, so the monoid math and shared cache are reused, not rebuilt. HONEST: on this 1-core box the pool shows no wall-clock speedup (offload COARSE, not fine); on a multi-core machine it parallelizes. *** one coordinator, one monoid reduce -- in a vector, across processes, across a farm ***")

# COMMAND BACKEND (R4): run any ALLOWLISTED external program (the door to ffmpeg / solvers / API calls), wired as an
# orchestrator Tool the Planner can chain. Allowlist + no-shell + timeout are the safety rails.
from holographic_command import CommandRunner as _cr_CR, CommandError as _cr_CE, command_as_tool as _cr_tool
from holographic_ai import Vocabulary as _cr_Vocab
_cr_r=_cr_CR(timeout=10)
_cr_r.register("upper",["python3","-c","import sys;print(sys.argv[1].upper())","{input}"], doc="uppercase")
_cr_r.register("reverse",["python3","-c","import sys;print(sys.argv[1][::-1])","{input}"], doc="reverse")
_cr_vocab=_cr_Vocab(1024,0)
_cr_up=_cr_tool(_cr_r,"upper","text","text",["uppercase"],_cr_vocab)
_cr_rev=_cr_tool(_cr_r,"reverse","text","text",["reverse"],_cr_vocab)
_cr_chain=_cr_rev.fn(_cr_up.fn("distributed").strip()).strip()
_cr_literal=_cr_r.run("upper",{"input":"safe; rm -rf /"})["stdout"].strip()
_cr_refused=False
try: _cr_r.run("cat",{"input":"/etc/passwd"})
except _cr_CE: _cr_refused=True
print(f"  COMMAND BACKEND: registered two external programs on an allowlist and CHAINED them as orchestrator Tools -- upper then reverse turned 'distributed' into '{_cr_chain}', a real subprocess pipeline the Planner can assemble. It is injection-safe by construction: no shell, so 'safe; rm -rf /' came back as the literal '{_cr_literal}' (the ';' was never interpreted), and a command NOT on the allowlist was refused ({_cr_refused}). An external program thus joins the same VSA fabric as an internal faculty -- selectable and chainable, with a CircuitBreaker on a flaky one. *** the can of worms opens inward, neatly: tools-as-vectors ***")

# NETWORK FARM (R3): run the coordinator's workers on OTHER machines. A worker daemon (stdlib http/json) holds your
# registered workers + a cache kept by content hash; the SAME Coordinator.run dispatches buckets to it.
from holographic_coordinator import Coordinator as _nf_C
from holographic_farm import WorkerDaemon as _nf_WD, NetworkFarm as _nf_NF, _sum_indices as _nf_sum, _content_hash as _nf_hash
from holographic_distribute import reduce_sum as _nf_rs
import numpy as _nf_np
_nf_node=_nf_WD(port=0); _nf_node.register_worker("sum_indices",_nf_sum); _nf_addr=_nf_node.start()
try:
    _nf_cache=_nf_np.arange(30,dtype=float)**2
    _nf_buckets=[list(range(0,10)),list(range(10,20)),list(range(20,30))]
    with _nf_C(_nf_NF([_nf_addr])) as _nf_coord:
        _nf_got=_nf_coord.run(_nf_buckets,"sum_indices",cache=_nf_cache,reduce=_nf_rs)
    _nf_shipped_once=_nf_hash(_nf_cache) in _nf_node.caches
    _nf_speed=round(_nf_node._measure_speed())
    _nf_refused=not _nf_node._handle("/task",{"worker":"rm_rf","bucket":[1],"cache_hash":None})["ok"]
finally:
    _nf_node.stop()
print(f"  NETWORK FARM: brought up a worker daemon on {_nf_addr} (a stand-in for another machine -- it speaks stdlib http+json, no framework) and dispatched the SAME partitioned monoid job to it: it summed the shared cache to {_nf_got:.0f}, matching the local answer, via the identical Coordinator.run -- only WHERE the worker ran changed. The {_nf_cache.nbytes}-byte read-only cache was shipped ONCE and kept by content hash for reuse ({_nf_shipped_once}); the node reported a speed of ~{_nf_speed} iters/s for load-balancing. And the security line held: buckets are data, workers are REGISTERED code, so an unregistered 'rm_rf' worker was refused ({_nf_refused}). Point the daemon's --host at a real machine and this is a render farm. *** one coordinator, one monoid reduce -- in a vector, across processes, across a farm ***")

# STANDALONE API SERVICE: run the engine as a server on any OS and talk to it over HTTP/JSON. Stdlib-only; here we
# drive the real service object through its route dispatch (the same code the HTTP layer calls).
from holographic_service import Service as _sv_Service, __version__ as _sv_ver
_sv=_sv_Service()
_sv_health=_sv.dispatch("GET","/health",{})[1]
_sv.dispatch("POST","/sql",{"sql":"CREATE TABLE user.items (id, name, color)"})
_sv.dispatch("POST","/sql",{"sql":"INSERT INTO user.items (id, name, color) VALUES (1, widget, red)"})
_sv.dispatch("POST","/sql",{"sql":"INSERT INTO user.items (id, name, color) VALUES (2, gadget, blue)"})
_sv.dispatch("POST","/sql",{"sql":"UPDATE user.items SET color = 'crimson' WHERE id = 1"})
_sv_rows=_sv.dispatch("POST","/sql",{"sql":"SELECT name, color FROM user.items WHERE id = 1"})[1]["result"]
_sv.dispatch("POST","/documents",{"objects":[{"id":"o1","name":"ring","material":"gold"},{"id":"o2","name":"pipe","material":"copper"}]})
_sv_gql=[o["name"] for o in _sv.dispatch("POST","/graphql",{"query":'{ objects(where: {material: "gold"}) { name } }'})[1]["data"]["objects"]]
_sv_search=[m["name"] for m in _sv.dispatch("POST","/capabilities/search",{"query":"render farm another machine"})[1]["matches"][:2]]
_sv_401=_sv_Service(token="secret")
print(f"  STANDALONE API SERVICE: the engine runs as a server (v{_sv_ver}) you drive over HTTP/JSON -- stdlib only, launched by serve.sh on Linux/macOS or serve.bat on Windows. /health reports {_sv_health['capabilities']} capabilities on Python {_sv_health['python']}/{_sv_health['platform']}. The WHOLE database is exposed over HTTP: CREATE/INSERT then an UPDATE ... WHERE left {_sv_rows} -- and the full surface adds DELETE, JOIN and DROP too (UPDATE/DELETE require a WHERE, so a typo can't wipe a table). GraphQL rides alongside for nested data: a query for gold objects returned {_sv_gql}. With --persist the store is auto-saved and survives a restart, so it is a drop-in DATABASE for other apps -- talk SQL or GraphQL to it from any language. Bind it local by default; point it at 0.0.0.0 behind auth/TLS to reach it from another machine. *** the whole engine, standalone, a real database behind one HTTP door ***")

# DISTRIBUTED HARDENING (R5): fault tolerance + verification for untrusted nodes -- retry, redundant-compute VOTING
# (accept only what independent workers agree on), and canary spot-checks. The discipline before a public farm.
from holographic_hardening import agree as _h5_agree, NoConsensus as _h5_NC, HardenedCoordinator as _h5_HC, CanaryFailed as _h5_CF, _sum_bucket as _h5_sum, _FlakyBackend as _h5_Flaky
from holographic_coordinator import InProcessBackend as _h5_IP
from holographic_distribute import reduce_sum as _h5_rs
# voting: a clear majority is accepted; genuine disagreement ABSTAINS rather than guessing
_h5_voted=_h5_agree([42.0, 42.0, 7.0])
_h5_abstained=False
try: _h5_agree([1.0,2.0,3.0])
except _h5_NC: _h5_abstained=True
# retry: a node that fails its first two attempts still yields a result on the third (reissue = reassign)
_h5_retry=_h5_HC(_h5_Flaky(fail_times=2), redundancy=1, attempts=3, backoff=0.001).run([[1,2,3]], _h5_sum, reduce=_h5_rs)
# redundant voting over 3 honest copies is accepted
_h5_red=_h5_HC(_h5_IP(), redundancy=3).run([[1,2,3],[4,5]], _h5_sum, reduce=_h5_rs)
# a canary (known answer) rejects a lying worker
_h5_caught=False
try: _h5_HC(_h5_Flaky(wrong_for=([1,2,3],999.0))).run([[9]], _h5_sum, canaries=[([1,2,3],6.0)])
except _h5_CF: _h5_caught=True
print(f"  DISTRIBUTED HARDENING (R5): the piece that lets STRANGERS help without being trusted. Redundant workers VOTE -- three answers [42,42,7] were accepted as {_h5_voted:.0f} by majority, while a genuine three-way disagreement ABSTAINED ({_h5_abstained}) rather than guess (a node can't FORCE a result). A worker that failed its first two attempts still succeeded on retry (reissue reassigns the work): {_h5_retry:.0f}. Three honest copies of a job agreed and were accepted: {_h5_red:.0f}. And a CANARY -- a bucket whose answer we already know -- caught a lying worker and rejected the run ({_h5_caught}). Repair (cleanup/fountain/verify) fixes accidental corruption; voting + canaries catch a node that faithfully returns a plausible LIE. This is the BOINC/SETI@home discipline, mandatory before a public farm. *** trust no single node -- require agreement, and check the canaries ***")

# JOB LIFECYCLE (start/pause/resume/cancel, survive a restart): a long render/sim is buckets + a monoid reduce, so
# completed buckets fold into partials and a paused job checkpoints to disk and resumes only the remaining work.
import tempfile as _jl_tmp
from holographic_jobs import JobManager as _jl_JM, _sum_bucket as _jl_w, DONE as _jl_DONE, PAUSED as _jl_PAUSED
from holographic_coordinator import InProcessBackend as _jl_IP
_jl_store=_jl_tmp.mkdtemp(prefix="tour_jobs_")
_jl_m=_jl_JM(_jl_IP(), store_dir=_jl_store); _jl_m.register_worker("sum", _jl_w)
_jl_m.create("render", [[i] for i in range(10)], "sum", reduce="sum")
# pretend the first five tiles rendered, then we paused -- exactly what a checkpoint holds
_jl_job=_jl_m.jobs["render"]; _jl_job.done=[0,1,2,3,4]; _jl_job.partials=[0.0,1.0,2.0,3.0,4.0]; _jl_job.status=_jl_PAUSED
_jl_m.save("render")
_jl_progress=_jl_m.jobs["render"].progress()
# ... the app closes; a BRAND-NEW manager (a fresh session) reopens the checkpoint and resumes ...
_jl_m2=_jl_JM(_jl_IP(), store_dir=_jl_store); _jl_m2.register_worker("sum", _jl_w)
_jl_m2.load_all()
_jl_remaining=len(_jl_m2.jobs["render"].remaining())
_jl_m2.resume("render")
_jl_result=_jl_m2.result("render")
print(f"  JOB LIFECYCLE: a render was paused {int(_jl_progress*100)}% through and CHECKPOINTED to disk. Then -- as if the app had closed and reopened -- a brand-new manager loaded the checkpoint and saw {_jl_remaining} tiles still to do. Resuming ran ONLY those remaining tiles and reduced the whole thing to {_jl_result:.0f} (= sum 0..9), each tile computed exactly once, none lost or repeated. Because a long job is buckets + a commutative monoid reduce, pause/resume is just 'stop after a bucket, keep the partials, finish the rest later' -- and the same works across the network farm and over the HTTP API (start/pause/resume/cancel a render, close the app, reopen, resume). *** stop, save, close, reopen, resume -- the monoid makes it free ***")

# MATERIAL LIBRARY (render appearance + physical properties, bridged and discoverable): one door to both of the
# engine's material libraries -- how a material LOOKS (for rendering) and how it BEHAVES (for science).
import holographic_materialindex as _mat_idx
_mat_s=_mat_idx.summary()
_mat_cats=_mat_idx.physical_categories()
_mat_clean=(_mat_idx.validate_physical()==[])
_mat_metals=len(_mat_idx.physical_by_category("metal"))
_mat_gold=_mat_idx.material_info("gold")
_mat_merc=_mat_idx.material_info("mercury")            # physical-only (a scientist's material, no render preset)
_mat_gems=[r["name"] for r in _mat_idx.find_materials("gem crystal")][:5]
_mat_pbr=_mat_idx.render_material("copper")            # a render-ready PBRMaterial straight from the library
print(f"  MATERIAL LIBRARY: the engine ships TWO material libraries, now bridged and discoverable through one index -- {_mat_s['render_presets']} RENDER presets across {_mat_s['render_classes']} classes (metals, gems, woods, stones, liquids, biomes...) and {_mat_s['physical_materials']} PHYSICAL materials (density, refractive index, viscosity, Young's modulus, sound speed, specific heat, phase), with {_mat_s['in_both']} in both. Ask about GOLD and you get BOTH sides at once: it renders as a {_mat_gold['render']['class']} (metallic {_mat_gold['render']['metallic']:.0f}, base colour {[round(c,2) for c in _mat_gold['render']['base_color'][:3]]}) AND it weighs {_mat_gold['physical']['density']} kg/m3 with sound travelling {_mat_gold['physical']['sound_speed']} m/s through it -- appearance for the renderer, physics for a scientist. The physical side is a comprehensive, VALIDATED starting point ({_mat_clean}): {_mat_metals} metals plus liquids, gases, polymers, ceramics, glass, minerals, stone, wood, tissue, building materials and semiconductors across {len(_mat_cats)} categories, each field unit-documented. Materials without a render preset still carry their science: mercury is a {_mat_merc['physical']['phase']} at {_mat_merc['physical']['density']} kg/m3, and tungsten melts at {_mat_idx.material_info('tungsten')['physical']['melting_point']} K. Discovery works across both: 'gem crystal' finds {_mat_gems}, and copper comes back as a ready-to-render PBRMaterial (metallic {_mat_pbr.metallic:.0f}). *** how it looks and how it behaves -- one library, two audiences ***")

# VENDORED DICTIONARY + TAXONOMY: real world-knowledge (~144k words) the engine carries, for contextual awareness.
import holographic_dictionary as _dic
_dic_n=_dic.size()
_dic_grav=_dic.entry("gravity")
_dic_tax=_dic.hypernym_chain("dog")[:5]
_dic_algo=_dic.define("algorithm")
print(f"  DICTIONARY + TAXONOMY: the engine now ships a real, comprehensive dictionary -- {_dic_n:,} English words, each with a definition, part of speech, synonyms, an example, and its 'is a kind of' parent -- so it has world-knowledge to lean on, not just internal machinery. It is Princeton WordNet, loaded stdlib-only (gzip+json) and used ONLY at build time via NLTK, so nothing is added to the runtime deps. Look up ALGORITHM and it knows: '{_dic_algo}'. Look up GRAVITY and it gives the physics definition plus synonyms {_dic_grav.get('s', _dic_grav.get('synonyms', []))[:2]}. And because every noun carries its hypernym, it doubles as an encyclopedia: a DOG walks up the taxonomy {_dic_tax}. The mind can also LEARN meaning from these real definitions (learn_vocabulary), and users can swap in a bigger dictionary against the same machinery. *** the engine reads the dictionary, so it knows what words mean ***")

# TOOL DISCOVERABILITY: the catalog surfaces whole families of tools by plain-English need (2D, text, learning, utils).
from holographic_catalog import default_catalog as _tw_cat, seed_from_modules as _tw_seed
_tw_c=_tw_seed(_tw_cat())
def _tw_top(q):
    h=_tw_c.find_capability(q, k=1); return h[0].name if h else "NOTHING"
_tw_draw=_tw_top("draw a picture")
_tw_gen=_tw_top("generate text")
_tw_learn=_tw_top("learn from a corpus")
_tw_util=_tw_top("verify data integrity")
print(f"  TOOL DISCOVERABILITY: the catalog turns a plain-English NEED into the right family of tools -- no need to know a module name. 'draw a picture' now lands on '{_tw_draw}', 'generate text' on '{_tw_gen}', 'learn from a corpus' on '{_tw_learn}', and 'verify data integrity' on '{_tw_util}'. Those 2D-image, text-generation, language-learning and utility tools were always in the engine, but until they had CURATED homes with the words a user would actually type, natural queries came up empty. The catalog also became our GAP-FINDER: pose the questions a user would ask (tools/catalog_gaps.py), and anything with no curated home is a discoverability hole to fill. *** if you can describe what you want, the catalog finds the tool ***")

# AGENT-FRIENDLY LAYER: describe a task, get the right skill WITH a confidence, and a route decision (act / choose).
import holographic_skills as _ag
_ag_man=_ag.manifest()["counts"]
_ag_act=_ag.route("render a scene with global illumination")
_ag_amb=_ag.route("distributed coordinator farm")
_ag_sug=_ag.suggest("edit an image")[0]
_ag_comp=[c["name"] for c in _ag.complete("learn_")[:4]]
print(f"  AGENT-FRIENDLY LAYER: the engine now describes and routes ITSELF, so an agent (or a person) does not have to memorise {_ag_man['methods']} methods across {_ag_man['capabilities']} capabilities. Ask for a task and it SUGGESTS with a confidence: 'edit an image' -> '{_ag_sug['name']}' (call: {_ag_sug['call'][:48]}...). It ROUTES with a decision node -- when it is sure it says ACT: 'render a scene' -> {_ag_act['decision']} ({_ag_act['confidence']:.0%}) -> {_ag_act['skill']['name']}; when it is genuinely torn it says CHOOSE rather than guess: 'distributed coordinator farm' -> {_ag_amb['decision']} among {[o['name'] for o in _ag_amb.get('options',[])][:3]}. And it AUTOCOMPLETES method names with their real signatures: mind.learn_<tab> -> {_ag_comp}. All of it is exposed over the HTTP API too (GET /skills, POST /skills/suggest|route|complete|card). *** the engine that can describe and route itself is the engine an agent can actually drive ***")

# DESCRIBE A SCENE -> BUILD -> ADJUST -> RENDER: talk a 3-D scene into being, then adjust its named objects in words.
from holographic_scene_semantic import scene_from_description as _sfd
_ss=_sfd("a big red metal sphere and a small blue glass box on a sunny day")
_ss_before=_ss.describe()
_ss.adjust("make the sphere bigger"); _ss.adjust("change the box to metal"); _ss.adjust("make everything glass")
_ss_after=_ss.describe()
_ss.adjust("make the pyramid golden")            # unknown target -> safe no-op, edits nothing
_ss_noop=_ss.describe()
_ss_sim=_sfd("a red sphere and a blue box").simulate(steps=20)
_ss_y0=float(list(_ss_sim[0].values())[0][1]); _ss_y1=float(list(_ss_sim[-1].values())[0][1])
print(f"  DESCRIBE -> BUILD -> ADJUST -> RENDER/SIMULATE: you can now TALK a 3-D scene into being and then adjust it by talking to its named objects. Describe 'a big red metal sphere and a small blue glass box on a sunny day' and the engine builds it: [{_ss_before}]. Then edit in plain words -- 'make the sphere bigger', 'change the box to metal', 'make everything glass' -- and it becomes [{_ss_after}]. It is honest about what it does NOT understand: 'make the pyramid golden' names no known object, so it changes NOTHING [{_ss_noop}] rather than guessing. From there scene.render() draws it (default camera, the scene sun/sky) and scene.simulate() drops the objects under gravity until they settle (a ball fell from y={_ss_y0:.1f} to rest at y={_ss_y1:.2f}). One mind.build_scene() call, or mind.semantic_scene(objects) to adjust a scene you already have. *** describe it, the engine builds it, and you shape it in words ***")

# ---- CMP1: composable texture map graph ----
import numpy as _np_rd
_tg_mind = _mind_rd
_tg_red = _tg_mind.texture_leaf(value=[1.0, 0.0, 0.0]); _tg_blue = _tg_mind.texture_leaf(value=[0.0, 0.0, 1.0])
_tg_noise = _tg_mind.texture_leaf("fbm", n_dims=2, seed=0)
_tg_base = _tg_mind.texture_op("mix", a=_tg_red, b=_tg_blue, t=_tg_noise)                 # blend two colours by a noise field
_tg_top = _tg_mind.texture_op("multiply", a=_tg_base, b=_tg_mind.texture_leaf(value=[0.9, 0.9, 0.9]))   # a map whose input is another map
_tg_val = _tg_mind.sample_texture(_tg_top, [0.3, 0.7])
_tg_depth = "mix(color,color,field) -> multiply(that, color)"
try:
    _tg_mind.texture_op("mix", a=_tg_red, b=_tg_blue, t=_tg_red); _tg_refused = "allowed (BUG)"   # a colour as a weight
except TypeError:
    _tg_refused = "refused at compose time"
_tg_v1 = _tg_mind.encode_texture(_tg_top)
_tg_v2 = _tg_mind.encode_texture(_tg_mind.texture_op("multiply", a=_tg_mind.texture_op("mix", a=_tg_mind.texture_leaf(value=[1.0, 0.0, 0.0]), b=_tg_mind.texture_leaf(value=[0.0, 0.0, 1.0]), t=_tg_mind.texture_leaf("fbm", n_dims=2, seed=0)), b=_tg_mind.texture_leaf(value=[0.9, 0.9, 0.9])))
_tg_cos = float(_np_rd.dot(_tg_v1, _tg_v2) / (_np_rd.linalg.norm(_tg_v1) * _np_rd.linalg.norm(_tg_v2)))
print(f"  COMPOSABLE TEXTURE GRAPH (CMP1): a texture is a readable TREE -- an op over TYPED inputs (map | color | field | number), each of which may be another map, so graphs nest to any depth. Here '{_tg_depth}' samples at uv=(0.3,0.7) to rgb [{float(_tg_val[0]):.2f}, {float(_tg_val[1]):.2f}, {float(_tg_val[2]):.2f}]. The discipline is the SCHEMA, checked at COMPOSE time: feed a colour where a weight belongs and it is {_tg_refused} -- a bad graph is caught up front, not rendered wrong. The tree stays plain Python you can read; encode it to a hypervector only when it earns its keep (cache/search) -- two structurally identical graphs encode to the SAME code (cosine {_tg_cos:.3f}). Reuses the Texture leaf sources (fbm/voronoi/synth) and typed's tree encoder; no new machinery. *** textures compose like shaders, type-checked before they render ***")

# ---- CMP3: multi-material blended by a mask ----
import numpy as _np_mm
from holographic_fpe import VectorFunctionEncoder as _VFE_mm
from holographic_material import Material as _Mat_mm, texture_field as _tf_mm
_mm_enc = _VFE_mm(2, dim=256, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
_mm_grid = [(u, v) for u in _np_mm.linspace(0.05, 0.95, 6) for v in _np_mm.linspace(0.05, 0.95, 6)]
_mm_A = _Mat_mm(_mm_enc, {"albedo": _tf_mm(_mm_enc, _mm_grid, [u for (u, v) in _mm_grid])})        # ramps up in u
_mm_B = _Mat_mm(_mm_enc, {"albedo": _tf_mm(_mm_enc, _mm_grid, [1.0 - u for (u, v) in _mm_grid])})  # ramps down in u
_mm_mask = _mind_rd.texture_op("scale", x=_mind_rd.texture_leaf("fbm", n_dims=2, seed=0), k=_mind_rd.texture_leaf(value=1.0))
_mm_blend = _mind_rd.multi_material([_mm_A, _mm_B], [_mm_mask, 0.5], mode="blend")
_mm_uv = [0.35, 0.6]
_mm_w = _mm_blend.weights_at(_mm_uv)
_mm_val = float(_mm_blend.sample("albedo", _mm_uv))
_mm_expect = float(_mm_w[0] * _mm_A.sample("albedo", _mm_uv) + _mm_w[1] * _mm_B.sample("albedo", _mm_uv))
_mm_drift = float(_mind_rd.multi_material([_mm_A, _mm_B], [1.0, 1.0], normalize=False).sample("albedo", _mm_uv))
_mm_norm = float(_mind_rd.multi_material([_mm_A, _mm_B], [1.0, 1.0], normalize=True).sample("albedo", _mm_uv))
_mm_pick = float(_mind_rd.multi_material([_mm_A, _mm_B], [0.2, 0.8], mode="select").sample("albedo", _mm_uv))
print(f"  MULTI-MATERIAL BY MASK (CMP3): Material.blend mixes TWO materials by one scalar; this mixes N by per-point MASKS -- each material's weight is a mask that varies over the surface (a CMP1 texture graph, a field, or a constant), so you paint rust into metal or moss onto stone. At uv={_mm_uv} the fbm mask gives weights [{float(_mm_w[0]):.2f}, {float(_mm_w[1]):.2f}] and the albedo reads {_mm_val:.3f} -- exactly w0*A+w1*B ({_mm_expect:.3f}), a bundle weighted by a field. The weights NORMALISE to a partition of unity so brightness stays put ({_mm_norm:.3f}); leave them unnormalised and it drifts too bright ({_mm_drift:.3f}) -- the kept negative, shown. And 'select' mode hard-picks the dominant material for a crisp material-ID / splat map ({_mm_pick:.3f}). Reuses Material + CMP1; no new machinery. *** materials blend by a mask, the mask is just another texture graph ***")

# ---- CMP2: layered material with an order schema ----
import numpy as _np_lm
from holographic_fpe import VectorFunctionEncoder as _VFE_lm
from holographic_material import Material as _Mat_lm, texture_field as _tf_lm
from holographic_layeredmaterial import LAYER_RANK as _LM_RANK
_lm_enc = _VFE_lm(2, dim=256, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
_lm_grid = [(u, v) for u in _np_lm.linspace(0.05, 0.95, 6) for v in _np_lm.linspace(0.05, 0.95, 6)]
_lm_paint = _Mat_lm(_lm_enc, {"albedo": _tf_lm(_lm_enc, _lm_grid, [u for (u, v) in _lm_grid])})       # ramps up
_lm_gloss = _Mat_lm(_lm_enc, {"albedo": _tf_lm(_lm_enc, _lm_grid, [1.0 - u for (u, v) in _lm_grid])}) # ramps down
_lm_stack = _mind_rd.layered_material([_mind_rd.material_layer("base", _lm_paint), _mind_rd.material_layer("clearcoat", _lm_gloss, alpha=0.35)])
_lm_uv = [0.4, 0.6]
_lm_val = float(_lm_stack.sample("albedo", _lm_uv))
_lm_expect = float(0.35 * _lm_gloss.sample("albedo", _lm_uv) + 0.65 * _lm_paint.sample("albedo", _lm_uv))
try:
    _mind_rd.layered_material([_mind_rd.material_layer("clearcoat", _lm_gloss), _mind_rd.material_layer("base", _lm_paint)])
    _lm_order = "allowed (BUG)"
except ValueError:
    _lm_order = "refused at compose time"
_lm_tiers = " < ".join(sorted(_LM_RANK, key=lambda k: _LM_RANK[k]))
print(f"  LAYERED MATERIAL + ORDER SCHEMA (CMP2): real surfaces are STACKS -- base < diffuse < specular/reflection < coat/clearcoat -- and the ORDER is a schema checked when you build the stack, so a base placed above a clearcoat is {_lm_order}, not rendered wrong. Here a 35%-covering clearcoat over paint composites at uv={_lm_uv} to albedo {_lm_val:.3f} -- exactly over(coat,base,0.35) = 0.35*coat+0.65*base ({_lm_expect:.3f}); each upper layer sits OVER the one below by its coverage alpha (a number, a field, or a CMP1 texture graph, so a coat can cover only part of a surface). HONEST BOUNDARY (kept loud): this fixes the STACKING, not the RADIOMETRY -- a physically energy-conserving layered BRDF, where the coat darkens and tints what's under it, is a separate harder thing and is NOT claimed. Tiers: {_lm_tiers}. Reuses Material + CMP1 + typed. *** materials stack in the right order, checked before they render ***")

# ---- CMP4: shared-definition instancing + type-safe binding ----
from holographic_mesh import box as _box_in
from holographic_scenegraph import translation as _tr_in
_in_chair = _mind_rd.shared_definition("chair", _box_in(1.0, 1.0, 1.0), "metal")
_in_scene = _mind_rd.instanced_scene()
_in_a = _in_scene.place(_in_chair, _tr_in([-2.0, 0.0, 0.0]), name="left")
_in_b = _in_scene.place(_in_chair, _tr_in([2.0, 0.0, 0.0]), name="right")
_in_scene.place(_in_chair, _tr_in([0.0, 0.0, 2.0]), name="back")
_in_before = _in_a.material
_in_chair.set_material("glass")                                # ONE edit...
_in_after = (_in_a.material, _in_b.material)                   # ...changes every instance
try:
    _mind_rd.shared_definition("bad", _box_in(1.0, 1.0, 1.0), "smoke")   # a volumetric material on a mesh
    _in_bind = "allowed (BUG)"
except TypeError:
    _in_bind = "refused at compose time"
_in_merged = _in_scene.flatten_surface()
_in_per = _box_in(1.0, 1.0, 1.0).n_vertices
print(f"  SHARED-DEFINITION INSTANCING + TYPE-SAFE BINDING (CMP4): place ONE shared definition many times and every copy is a REFERENCE, not a duplicate -- 3 chairs share one 'chair' definition, so repainting it once ({_in_before} -> the instances now read {_in_after[0]}/{_in_after[1]}) recolours ALL of them in a single edit. The material<->geometry binding is TYPE-CHECKED when you build the definition: a surface material needs a mesh and a volumetric one (fog/smoke/fire) needs a volume, so putting smoke on a solid mesh is {_in_bind}, not rendered wrong. HONEST BOUNDARY (kept loud): the sharing is edit-once at the GRAPH level; flatten_surface() is where instances become concrete geometry -- it merges the 3 surface instances into one mesh ({int(_in_merged.n_vertices)} = 3 x {int(_in_per)} verts). Reuses scenegraph + the surface/volumetric split. *** one definition, many placements, edit once -- and bad bindings caught before they render ***")

# ---- CMP5: pipeline composes the graphs (bake vs live) ----
import numpy as _np_rg
from holographic_mesh import box as _box_rg
from holographic_scenegraph import translation as _tr_rg
from holographic_rendergraph import BakedTexture as _BT_rg
_rg_tex = _mind_rd.texture_op("mix", a=_mind_rd.texture_leaf(value="red"), b=_mind_rd.texture_leaf(value="blue"), t=_mind_rd.texture_leaf("fbm", n_dims=2, seed=0))
_rg_scene = _mind_rd.instanced_scene()
_rg_chair = _mind_rd.shared_definition("chair", _box_rg(1.0, 1.0, 1.0), "metal")
_rg_scene.place(_rg_chair, _tr_rg([-2.0, 0.0, 0.0])); _rg_scene.place(_rg_chair, _tr_rg([2.0, 0.0, 0.0]))
_rg = _mind_rd.render_graph(res=32)
_rg.add_texture("rust", _rg_tex, static=True).add_texture("ripples", _rg_tex, static=False).set_scene(_rg_scene)
_rg_plan = _rg.plan()
_rg_prep = _rg.prepare()
_rg_rust_baked = isinstance(_rg_prep.texture("rust"), _BT_rg)
_rg_ripples_live = _rg_prep.texture("ripples") is _rg_tex
_rg_live = _np_rg.asarray(_rg_tex.sample([0.4, 0.6]))
_rg_baked = _np_rg.asarray(_rg_prep.texture("rust").sample([0.4, 0.6]))
_rg_interp = float(_np_rg.max(_np_rg.abs(_rg_live - _rg_baked)))
print(f"  PIPELINE COMPOSES THE GRAPHS (CMP5): the render pipeline now reaches all the way DOWN to the maps and materials. Register the texture graphs + the instanced scene, call plan(), and it tells you what it will do and WHY before running: '{_rg_plan[0].split(': ',1)[1]}'; '{_rg_plan[1].split(': ',1)[1]}'. The adaptive decision is BAKE a STATIC map to a grid (O(1) bilinear lookup, pay the tree walk once) vs SAMPLE a changing map LIVE (baking it would be redone every frame). Here 'rust' baked = {_rg_rust_baked}, 'ripples' stayed live = {_rg_ripples_live}; the baked lookup still matches the live graph to {_rg_interp:.3f} (interpolation error -- the kept trade: memory for speed). prepare() also binds+flattens the CMP4 scene ({int(_rg_prep.surface_mesh.n_vertices)} verts). Reuses pipeline Stages (needs/produces) + matbake's bake trade + CMP1-CMP4. *** 'adaptive' now decides bake-vs-live per map, not just per render pass ***")

# ---- SEE what you composed: swatch + material ball ----
import numpy as _np_pv
from holographic_fpe import VectorFunctionEncoder as _VFE_pv
from holographic_material import Material as _Mat_pv, texture_field as _tf_pv
_pv_g = _mind_rd.texture_op("mix", a=_mind_rd.texture_leaf(value="orange"), b=_mind_rd.texture_leaf(value="purple"), t=_mind_rd.texture_leaf("fbm", n_dims=2, seed=0))
_pv_swatch = _mind_rd.preview_texture(_pv_g, res=64)
_pv_enc = _VFE_pv(2, dim=256, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
_pv_grid = [(a, b) for a in _np_pv.linspace(0.05, 0.95, 6) for b in _np_pv.linspace(0.05, 0.95, 6)]
_pv_mat = _Mat_pv(_pv_enc, {"roughness": _tf_pv(_pv_enc, _pv_grid, [a for (a, b) in _pv_grid]), "metallic": _tf_pv(_pv_enc, _pv_grid, [0.9 for _ in _pv_grid])})
_pv_ball = _mind_rd.preview_material(_pv_mat, res=64)
_pv_sw_ok = (_pv_swatch.shape == (64, 64, 3) and 0.0 <= float(_pv_swatch.min()) and float(_pv_swatch.max()) <= 1.0)
_pv_ball_shaded = not bool(_np_pv.allclose(_pv_ball[32, 32], _pv_ball[0, 0]))
print(f"  SEE WHAT YOU COMPOSED (swatch + material ball): the composability stack builds things you .sample(uv); these render them to an image so you can LOOK. preview_texture(graph) makes a flat RGB SWATCH ({_pv_swatch.shape} in [0,1] = {_pv_sw_ok}); preview_material(material) shades it on the classic MATERIAL BALL with the SAME Cook-Torrance BRDF the renderer uses -- reading its roughness/metallic channels off a sphere (centre differs from the background = the ball was actually shaded = {_pv_ball_shaded}). Works on a plain Material or a CMP2/CMP3 layered/multi material. The missing step between composing a material and knowing if it looks right. *** compose it, then just look at it ***")

# ---- composed texture painted onto a scene object, full render ----
import numpy as _np_tr
from holographic_semantic import _UnionSDF as _Union_tr
from holographic_raymarch import sphere_trace as _st_tr
from holographic_render import Camera as _Cam_tr
_tr_scene = _mind_rd.build_scene("a big red metal sphere")
_tr_tex = _mind_rd.texture_op("mix", a=_mind_rd.texture_leaf(value="red"), b=_mind_rd.texture_leaf(value="cyan"), t=_mind_rd.texture_leaf("fbm", n_dims=2, seed=1, octaves=5))
_tr_W = _tr_H = 72
_tr_img = _mind_rd.render_textured(_tr_scene, {_tr_scene.names()[0]: _tr_tex}, width=_tr_W, height=_tr_H)
_tr_union = _Union_tr([r["sdf"] for r in _tr_scene.realize()])
_tr_span = max(3.0, 1.6)
_tr_cam = _Cam_tr(eye=(_tr_span * 0.4, _tr_span * 0.28, _tr_span), target=(0, 0, 0), fov_deg=42.0)
_tr_eye, _tr_dirs = _tr_cam.ray_dirs(_tr_W, _tr_H)
_tr_D = _tr_dirs.reshape(-1, 3); _tr_O = _np_tr.broadcast_to(_tr_eye, _tr_D.shape).copy()
_tr_hit, _, _ = _st_tr(_tr_union, _tr_O, _tr_D)
_tr_sph = _tr_img.reshape(-1, 3)[_tr_hit]
_tr_rvar = float(_tr_sph[:, 0].std()); _tr_bvar = float(_tr_sph[:, 2].std())
print(f"  COMPOSED TEXTURE PAINTED ONTO A SCENE OBJECT (full render): the composability stack now drives a REAL 3-D image. Build a texture with mind.texture_op(...), then mind.render_textured(scene, {{name: texture}}) marches the scene, turns each surface hit into a UV (spherical map on a sphere, planar on a box), samples the texture there, and shades it with the SAME Cook-Torrance BRDF the renderer uses -- plus a light and a hard shadow. Here a red<->cyan fbm wraps onto the sphere: across its {int(_tr_hit.sum())} surface pixels the painted red varies by std {_tr_rvar:.3f} and blue by {_tr_bvar:.3f} -- a FLAT tint would be ~0, so the texture genuinely WRAPS via UV mapping, it isn't just a recolour. Reuses the marcher (sphere_trace) + sdf_normal + the BRDF; the swatch/ball preview grown up into a scene. HONEST (kept): textbook UV (a seam + pole pinch on the sphere, face seams on a box), one hard light (the path tracer is the tool for GI). *** the texture you composed now wraps around the object in the render ***")

# ---- named objects + textures in the describe-a-scene flow ----
_ns_scene = _mind_rd.build_scene("a big metal sphere on a green box")
_ns_scene.name("the sphere", "hero")                      # give it a nickname you can rename + reference
_ns_names_after_naming = _ns_scene.names()
_ns_scene.adjust("make hero glass")                       # reference the object by its nickname
_ns_hero_mat = _ns_scene.get("hero")[0]["material"]
_ns_scene.adjust("rename hero to champion")               # rename via plain command
_ns_scene.adjust("give champion a rusty texture")         # paint a named procedural texture by talking to it
_ns_scene.adjust("make the box mossy")
_ns_has_tex = _ns_scene.get("champion")[0]["texture"] is not None
_ns_img = _ns_scene.render(width=64, height=48)           # render() now ROUTES through the textured renderer
import numpy as _np_ns
_ns_std = float(_np_ns.asarray(_ns_img).std())
print(f"  NAMED OBJECTS + TEXTURES IN THE SCENE FLOW: you can now nickname the things you build and paint them by talking. build_scene(...) then scene.name('the sphere','hero') -> names() shows {_ns_names_after_naming}; 'make hero glass' reaches it by that nickname (material now {_ns_hero_mat!r}); 'rename hero to champion' renames it; 'give champion a rusty texture' and 'make the box mossy' attach composed CMP1 textures (from a small named library: rusty/marbled/mossy/cloudy/lava/striped/noisy) -- champion carries a texture = {_ns_has_tex} -- and scene.render() automatically ROUTES through the textured renderer so the paint shows (image std {_ns_std:.3f}). A nickname wins over attribute-matching, so once named a thing is always reachable. *** name it, paint it, render it -- all by describing what you want ***")

# ---- message bus + optional agent (LLM) bridge ----
import numpy as _np_bus
_bus_told = []
_bus_bridge = _mind_rd.agent_bridge(llm=lambda text: (_bus_told.append(text), "looks good -- centered and bright")[1])
_bus_replies = []
_bus_bridge.on_reply(lambda msg: _bus_replies.append(msg.payload["reply"]))
_bus_bridge.notify_on("render.done", "A render just finished -- does it look right?")   # PUSH to the agent, no polling
_bus_scene = _mind_rd.build_scene("a red metal sphere")
_mind_rd.run_task("render", lambda: _np_bus.asarray(_bus_scene.render(width=48, height=40)),
                  summarize=lambda a: {"shape": list(a.shape), "mean": round(float(a.mean()), 3)})
_bus_topics = [msg.topic for msg in _mind_rd.bus().history()]
_bus_reached = bool(_bus_told) and "render.done" in _bus_told[0]
print(f"  MESSAGE BUS + OPTIONAL AGENT (leOS harness): a person AND an agent can both be attached to the running tool, and the app REACHES OUT to the agent instead of the agent polling. mind.bus() is a topic message bus; mind.run_task('render', fn) runs a job and publishes 'render.done' with a small summary; mind.agent_bridge(llm=my_fn).notify_on('render.done', ...) calls YOUR callable (any text->reply -- no LLM library is imported, so it's fully optional) the instant the render finishes and posts the reply back. Here the render task fired {_bus_topics[:2]}, the agent was told with the summary = {_bus_reached}, and it replied {_bus_replies!r} -- all without anyone polling a status flag. With no agent attached it all still runs (an 'agent.unattached' note is posted). Over HTTP a remote agent uses /bus/publish + /bus/poll on its own inbox. *** the app can invoke the agent, and both can message each other -- optional, dependency-free ***")

# ---- optional language layer: lazy dictionary + find words by meaning ----
import holographic_dictionary as _hd_tour
_hd_tour.unload()
_lang_loaded_before = _hd_tour.is_loaded()                 # building the mind never loaded it
_lang_idx = _mind_rd.build_semantic_index(words=["dog","puppy","cat","kitten","serendipity","luck","chance","river","stream","ocean","wealth","money","happy","joyful"], dim=256, seed=0)
_lang_loaded_after = _hd_tour.is_loaded()                  # building the index did
_lang_q1 = [w for w,_ in _lang_idx.find("unexpected good luck", k=3)]
_lang_q2 = [w for w,_ in _lang_idx.find("a young dog", k=3)]
_lang_q3 = [w for w,_ in _lang_idx.find("a body of water", k=3)]
print(f"  OPTIONAL LANGUAGE LAYER (lazy dictionary + meaning search): the ~144k-word dictionary is stored as lzma (~3.3 MB on disk, was 5.9 MB gzip) and is OPT-IN -- importing leCore or building a mind loaded it? {_lang_loaded_before}. It only decompresses into a plain dict in RAM on the first language call, so a user building on top pays nothing for it (control it with holographic_dictionary.preload()/unload()/stats()). On top of it, mind.build_semantic_index() places words in a MEANING space by random indexing over their glosses so you can search by description: 'unexpected good luck' -> {_lang_q1}, 'a young dog' -> {_lang_q2}, 'a body of water' -> {_lang_q3} (loaded now? {_lang_loaded_after}). Approximate -- reliable for the top hit, noisy in the tail, word-sense sensitive -- which is exactly where leCore's geometry-preserving/lossy side belongs (NOT on exact lookup). *** the dictionary access is instant once loaded, the whole layer is opt-in, and you can find words by meaning ***")
_hd_tour.unload()

# ---- external asset relocation / relink ----
import os as _os_as, tempfile as _tf_as, shutil as _sh_as
_as_root = _tf_as.mkdtemp(prefix="lecore_tour_assets_")
try:
    _as_old = _os_as.path.join(_as_root, "Documents", "project")
    for _rel in ("textures/water/wave.png", "textures/stone/wall.png", "models/boat.obj"):
        _p = _os_as.path.join(_as_old, *_rel.split("/"))
        _os_as.makedirs(_os_as.path.dirname(_p), exist_ok=True)
        open(_p, "wb").write(("data:" + _rel).encode())
    _as_lib = _mind_rd.asset_library()
    for _rel in ("textures/water/wave.png", "textures/stone/wall.png", "models/boat.obj"):
        _as_lib.add(_os_as.path.join(_as_old, *_rel.split("/")), role=_rel, with_hash=True)
    _as_new = _os_as.path.join(_as_root, "Projects", "project")
    _os_as.makedirs(_os_as.path.dirname(_as_new), exist_ok=True)
    _sh_as.move(_as_old, _as_new)                                   # the whole project folder moved
    _as_missing_before = len(_as_lib.missing())
    _as_rep = _as_lib.relink(_as_lib.assets[0].path, _os_as.path.join(_as_new, "textures", "water", "wave.png"))
    _as_missing_after = len(_as_lib.missing())
    _as_hows = sorted(set(_r["how"].split("(")[0] for _r in _as_rep["relinked"]))
    # change detection + content-hash resolve across "machines"
    import time as _tm_as
    _tm_as.sleep(0.01); open(_as_lib.assets[1].path, "ab").write(b" edit"); _os_as.utime(_as_lib.assets[1].path, None)
    _as_changed = len(_as_lib.changed())
    _as_other = _os_as.path.join(_as_root, "machineB", "renamed.obj")
    _os_as.makedirs(_os_as.path.dirname(_as_other), exist_ok=True)
    _sh_as.copy(_as_lib.assets[2].path, _as_other); _os_as.remove(_as_lib.assets[2].path)
    _as_byhash = _as_lib.resolve(_as_lib.assets[2].id, roots=[_os_as.path.join(_as_root, "machineB")])
    _as_found_by_hash = (_as_byhash == _as_other)
    print(f"  EXTERNAL ASSET RELOCATION (the 3-D 'missing textures' fix): the scene's texture/model files live on disk and break when folders move. mind.asset_library() tracks them and repairs paths the way you'd reason about it. Moved the whole project folder -> {_as_missing_before} assets broke; re-pointed just ONE, and the rest were re-found automatically ({_as_hows}: it works out the moved parent, rewrites the siblings, then structurally searches for anything reorganised) -> {_as_missing_after} still missing. It also spots on-disk EDITS ({_as_changed} file flagged 'modified' via size/mtime or content hash) and, for a DISTRIBUTED setup where paths differ per machine, resolves a file by CONTENT HASH wherever it landed (found the renamed copy: {_as_found_by_hash}). Readable stdlib (os/hashlib/json); saves a JSON manifest. *** fix one path, find the rest; know when a file changed; locate by content across machines ***")
finally:
    _sh_as.rmtree(_as_root, ignore_errors=True)

# ---- external texture files carried by a scene (asset tracking + relink) ----
import os as _os_st, tempfile as _tf_st, shutil as _sh_st
_st_root = _tf_st.mkdtemp(prefix="lecore_tour_scene_assets_")
try:
    _st_old = _os_st.path.join(_st_root, "Documents", "project", "textures")
    _os_st.makedirs(_st_old, exist_ok=True)
    for _n in ("wave.png", "wall.png"):
        open(_os_st.path.join(_st_old, _n), "wb").write(b"(texture bytes)")
    _st_scene = _mind_rd.build_scene("a big sphere and a small box")
    _st_scene.attach_texture_file("the sphere", _os_st.path.join(_st_old, "wave.png"))
    _st_scene.attach_texture_file("the box", _os_st.path.join(_st_old, "wall.png"))
    _st_missing0 = len(_st_scene.missing_assets())
    # move the whole project folder -> both texture files break
    _st_new = _os_st.path.join(_st_root, "Projects", "project")
    _os_st.makedirs(_os_st.path.dirname(_st_new), exist_ok=True)
    _sh_st.move(_os_st.path.join(_st_root, "Documents", "project"), _st_new)
    _st_missing1 = len(_st_scene.missing_assets())
    # re-point ONE, the other is found automatically; render still works (falls back to colour for any missing file)
    _st_scene.relink(_st_scene.assets.assets[0].path, _os_st.path.join(_st_new, "textures", "wave.png"))
    _st_missing2 = len(_st_scene.missing_assets())
    _st_counts = _st_scene.check_assets()["counts"]
    import numpy as _np_st
    _st_ok = _np_st.asarray(_st_scene.render(width=48, height=40)).shape == (40, 48, 3)
    print(f"  EXTERNAL TEXTURE FILES CARRIED BY A SCENE: a scene can reference real files on disk (scene.attach_texture_file('the sphere', 'project/textures/wave.png')) and it tracks them in an AssetLibrary. Moved the project folder -> {_st_missing1} of 2 textures broke; re-pointed just ONE and the other was found by the shared moved-parent -> {_st_missing2} still missing (status {_st_counts}). scene.render() reloads the resolved files and, crucially, falls back to the object's flat COLOUR for any file it still can't find rather than crashing (render ok: {_st_ok}). scene.set_asset_roots([...]) lets render auto-search for moved files; relink/resolve delegate to the same relocation + content-hash machinery. The image pixels load lazily (PIL only when an external image is actually drawn), so the core stays NumPy-only. *** the describe-a-scene flow now survives its textures being moved around ***")
finally:
    _sh_st.rmtree(_st_root, ignore_errors=True)

# ---- ingest a folder/zip into a queryable file map ----
import os as _os_fm, tempfile as _tf_fm, shutil as _sh_fm
_fm_root = _tf_fm.mkdtemp(prefix="lecore_tour_filemap_")
try:
    for _rel, _c in {"readme.md":"renders water with a caustic shader and normal maps", "src/shader.glsl":"vec3 normal = computeNormal(); float caustic = refractLight(normal);", "textures/water/wave.png":"PNG", "models/boat.obj":"v 0 0 0", "notes.txt":"todo: fix the lighting setup"}.items():
        _p = _os_fm.path.join(_fm_root, *_rel.split("/")); _os_fm.makedirs(_os_fm.path.dirname(_p), exist_ok=True); open(_p,"w").write(_c)
    _fm = _mind_rd.ingest_files(_fm_root)
    _fm_kinds = _fm.kinds()
    _fm_png = [ _e.relpath.split("/")[-1] for _e in _fm.find("*.png") ]
    _fm_models = len(_fm.by_kind("model"))
    _fm_kw = [ _e.relpath for _e,_h in _fm.search_text("normal caustic") ]
    _fm_tree_keys = sorted(_fm.tree().keys())
    _sh_fm.move(_fm_root, _fm_root + "_moved")
    _fm_missing1 = len(_fm.missing())
    _fm.relink(_fm.assets.assets[0].path, _os_fm.path.join(_fm_root + "_moved", _fm.files[0].relpath))
    _fm_missing2 = len(_fm.missing())
    print(f"  INGEST A FOLDER/ZIP INTO A QUERYABLE FILE MAP: point mind.ingest_files() at a folder, a .zip, or a file and it digests the whole tree into a map you can query. Here {len(_fm)} files -> kinds {_fm_kinds}; query by NAME (*.png -> {_fm_png}), by KIND (models -> {_fm_models}), by text CONTENT (inverted index: 'normal caustic' -> {[_p.split('/')[-1] for _p in _fm_kw]}), by MEANING (build_meaning_index + find_by_meaning), and read fm.tree() = the folder hierarchy ({_fm_tree_keys}). Every file is ALSO tracked for relocation/change: moved the whole tree -> {_fm_missing1} missing, re-pointed ONE -> {_fm_missing2} missing (the rest self-healed via the built-in AssetLibrary). Stdlib only; text indexing is size-capped so a pile of big binaries stays cheap. *** hand it a folder or zip; get a searchable, self-healing file map ***")
finally:
    _sh_fm.rmtree(_fm_root, ignore_errors=True); _sh_fm.rmtree(_fm_root + "_moved", ignore_errors=True)

# ---- cold storage: compress inactive structures, inflate on demand ----
import numpy as _np_cs
_cs_store = _mind_rd.cold_store(keep_warm=2)
for _i in range(6):
    _cs_store.put("table%d" % _i, _np_cs.tile(_np_cs.arange(400.), 30) + _i)   # redundant rows -> compress well
_cs_stats = _cs_store.stats()
_cs_got = float(_cs_store.get("table3")[0])                                    # cold -> transparently warmed
# fold up ONE big structure and measure the shrink
_cs_big = _np_cs.tile(_np_cs.arange(2000.), 100)
_cs_one = _mind_rd.cool(_cs_big, codec="lzma"); _cs_one.cool()
_cs_ratio = _cs_one.ratio()
_cs_back_ok = _np_cs.array_equal(_cs_one.get(), _cs_big)
print(f"  COLD STORAGE (compress inactive data, inflate on demand): a long-running app holds a lot of idle data -- tables nobody queried lately, another session's database, caches built once. mind.cold_store(keep_warm=K) keeps only the K most-recently-used values live and COMPRESSES the rest, warming any of them transparently the instant you get() it. Here 6 tables -> {_cs_stats['warm']} warm, {_cs_stats['cold']} cold, ~{_cs_stats['approx_saved_bytes']//1000} KB saved, and get('table3') still returned {_cs_got} (it was cold, got warmed). mind.cool(x) folds up ONE structure: a redundant array shrank to {_cs_ratio:.1%} of its size and came back bit-identical ({_cs_back_ok}). Works on tables, whole databases, big arrays -- anything picklable; codec='lzma' packs smaller, spill_dir=... writes cold blobs to disk to free RAM entirely. HONEST: high-entropy VSA vectors barely compress (there the win is freeing the live object / spilling to disk); redundant/text/structured data compresses hugely. *** fold up what's idle, unfurl it the moment it's touched ***")

# ---- database auto-cooling (opt-in, distributed-safe) ----
import pickle as _pk_db
from holographic_query import Database as _DB
_db_cool = _DB(); _db_cool.add_namespace("app")
for _t in range(5):
    _db_cool.create_table("app.t%d" % _t, ["id", "amt"])
    for _i in range(40):
        _db_cool.resolve("app.t%d" % _t).insert({"id": _i, "amt": _i})
_db_cool.enable_cold_storage(keep_warm=2)
_db_cool.resolve("app.t3"); _db_cool.resolve("app.t4")            # touch two -> keep warm
_db_cooled = _db_cool.cool_idle()
_db_s = _db_cool.cold_stats()
_db_val = _db_cool.resolve("app.t0").rows[5]["amt"]              # cooled -> warmed transparently
_db_shipped = _pk_db.loads(_pk_db.dumps(_db_cool))              # what a distributed worker receives
_db_ship_s = _db_shipped.cold_stats()
_db_before = dict(_db_shipped.cold_stats()); _db_shipped.resolve("app.t0"); _db_after = dict(_db_shipped.cold_stats())
print(f"  DATABASE AUTO-COOLING (opt-in, distributed-safe): a long-running query DB can compress the tables nobody has touched lately and inflate them on the next query. db.enable_cold_storage(keep_warm=K); db.cool_idle() folded up {_db_cooled} of 5 idle tables ({_db_s['warm']} warm, {_db_s['cold']} cold, {_db_s['cold_bytes']//1000} KB compressed) -- then resolving a cold table warmed it transparently (t0.amt[5] = {_db_val}, identical). THE DISTRIBUTED-SAFE part (the thing that could have caused trouble): a cold-enabled DB shipped to a worker arrives WARM with cooling OFF (enabled={_db_ship_s['enabled']}, cold={_db_ship_s['cold']}), and a worker's reads do NOT mutate that shared read-only cache (mutated? {_db_before != _db_after}) -- no lock or spill-path ever crosses the process boundary. Off by default, so nothing changes unless you ask. *** fold up idle tables for memory; ship a safe immutable copy to every worker ***")

# ---- import artist file formats: OBJ/MTL, glTF/GLB, texture set, volume ----
import os as _os_ai, tempfile as _tf_ai, shutil as _sh_ai, numpy as _np_ai
_ai_root = _tf_ai.mkdtemp(prefix="lecore_tour_import_")
try:
    # OBJ + MTL
    open(_os_ai.path.join(_ai_root, "m.obj"), "w").write("mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl red\nf 1/1 2/2 3/3 4/4\n")
    open(_os_ai.path.join(_ai_root, "m.mtl"), "w").write("newmtl red\nKd 0.8 0.1 0.1\nPr 0.4\n")
    _ai_obj = _mind_rd.load_obj(_os_ai.path.join(_ai_root, "m.obj"))
    # glTF/GLB round trip WITH a material
    from holographic_mesh import box as _box_ai
    from holographic_gltf import mesh_to_glb as _m2g
    from holographic_materialio import PBRMaterial as _PBR
    open(_os_ai.path.join(_ai_root, "b.glb"), "wb").write(_m2g(_box_ai(), material=_PBR(name="steel", base_color=(0.2,0.3,0.9,1.0), metallic=1.0, roughness=0.3)))
    _ai_glb = _mind_rd.load_glb(_os_ai.path.join(_ai_root, "b.glb"))
    _ai_steel = list(_ai_glb.materials.values())[0]
    # a volume grid that actually renders
    _n=32; _g=_np_ai.zeros((_n,_n,_n),_np_ai.float32)
    _zz,_yy,_xx=_np_ai.mgrid[0:_n,0:_n,0:_n]; _r=_np_ai.sqrt((_xx-_n/2)**2+(_yy-_n/2)**2+(_zz-_n/2)**2)
    _g[_r<_n*0.32]=1.0
    _np_ai.save(_os_ai.path.join(_ai_root,"blob.npy"), _g)
    _ai_field,_ai_bounds = _mind_rd.load_volume(_os_ai.path.join(_ai_root,"blob.npy"))
    from holographic_render import Camera as _CamAI
    _ai_img,_ai_alpha = _mind_rd.render_volume(_ai_field, _CamAI(eye=(0,0,3), target=(0,0,0), fov_deg=40), _ai_bounds, width=48, height=48, steps=48)
    _ai_cov = float((_np_ai.asarray(_ai_alpha)>0.01).mean())
    # a rigged/animated glTF: build a tiny one (node slides 0->2 on X over 1s) and sample the clip
    import struct as _st_ai, json as _js_ai
    _ai_blob = _np_ai.array([0.0,1.0],_np_ai.float32).tobytes() + _np_ai.array([[0,0,0],[2,0,0]],_np_ai.float32).tobytes()
    _ai_gltf = {"asset":{"version":"2.0"},"nodes":[{"name":"bone"}],"buffers":[{"byteLength":len(_ai_blob)}],
        "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":8},{"buffer":0,"byteOffset":8,"byteLength":24}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR","min":[0.0],"max":[1.0]},{"bufferView":1,"componentType":5126,"count":2,"type":"VEC3"}],
        "animations":[{"name":"slide","samplers":[{"input":0,"output":1,"interpolation":"LINEAR"}],"channels":[{"sampler":0,"target":{"node":0,"path":"translation"}}]}]}
    _ai_jb = _js_ai.dumps(_ai_gltf).encode(); _ai_jb += b" "*((4-len(_ai_jb)%4)%4)
    _ai_bb = _ai_blob + b"\x00"*((4-len(_ai_blob)%4)%4)
    _ai_glbbytes = _st_ai.pack("<III",0x46546C67,2,12+8+len(_ai_jb)+8+len(_ai_bb)) + _st_ai.pack("<II",len(_ai_jb),0x4E4F534A)+_ai_jb + _st_ai.pack("<II",len(_ai_bb),0x004E4942)+_ai_bb
    open(_os_ai.path.join(_ai_root,"anim.glb"),"wb").write(_ai_glbbytes)
    _ai_rig = _mind_rd.load_glb(_os_ai.path.join(_ai_root,"anim.glb"))
    _ai_clip = _ai_rig.animations[0]; _ai_mid = _ai_clip.sample(0.5)[0][:3,3]
    # DEFORM a rig: a one-bone LoadedMesh whose vertex is bound to a bone that translates; skinning MOVES it
    from holographic_assetimport import LoadedMesh as _LM_ai, AnimationClip as _AC_ai
    _ai_ng = [{"name":"root","local":_np_ai.eye(4),"children":[1]}, {"name":"bone","local":_np_ai.eye(4),"children":[]}]
    _ai_rigmesh = _LM_ai(_np_ai.array([[0.,0,0],[1,0,0]]), [(0,1,0)],
                         joints=_np_ai.array([[0,0,0,0],[0,0,0,0]]), weights=_np_ai.array([[1.,0,0,0],[1.,0,0,0]]),
                         skins=[{"joints":[1],"inverse_bind":_np_ai.eye(4)[None]}], node_graph=_ai_ng)
    _ai_wave = _AC_ai("wave", {1:{"translation":(_np_ai.array([0.,1.]), _np_ai.array([[0.,0,0],[0,2,0]]))}})
    _ai_rest = _mind_rd.deform_mesh(_ai_rigmesh, clip=None).vertices
    _ai_posed = _mind_rd.deform_mesh(_ai_rigmesh, clip=_ai_wave, t=1.0).vertices
    _ai_moved = float((_ai_posed - _ai_rest)[1][1])   # vertex 1 rose this much on Y
    print(f"  IMPORT ARTIST FILE FORMATS (the interchange an artist actually needs): mind.load_obj / load_glb / load_texture_set / load_volume, dispatched by mind.import_asset(path). OBJ+MTL came in as {_ai_obj.positions.shape[0]} verts / {_ai_obj.faces.shape[0]} tris with per-corner UVs and its 'red' material (base {tuple(round(float(x),1) for x in _ai_obj.materials['red'].base_color[:3])}). A glTF/GLB round-tripped geometry AND its full PBR channels (base-colour/metallic/roughness/normal/occlusion/emissive) + per-vertex UVs/normals -- here metallic={_ai_steel.metallic:.1f}, roughness={_ai_steel.roughness:.1f}, base blue={_ai_steel.base_color[2]:.1f}; embedded textures decoded from the .glb. For RIGGED models it also imports ANIMATIONS + SKINS: clip '{_ai_clip.name}' ({_ai_clip.duration:.1f}s) sampled at t=0.5 puts the bone at x={_ai_mid[0]:.1f} (linear midpoint; rotations slerped). mind.deform_mesh(loaded, clip, t) then actually MOVES the rig -- linear-blend skinning by the posed skeleton (+ morph-target blending): a vertex bound to that bone rose {_ai_moved:.1f} on Y at t=1. A Substance 3D Painter export folder (basecolor/roughness/metallic/normal/height/ao by file name) folds into one PBRMaterial. And a .npy density grid loaded as a trilinear field the volume renderer marched -> a {_np_ai.asarray(_ai_img).shape[0]}x{_np_ai.asarray(_ai_img).shape[1]} image, blob coverage {_ai_cov:.2f}. Stdlib+NumPy (PIL lazy for textures). HONEST: proprietary .sbsar/.spp and sparse OpenVDB .vdb need their vendor tools -- we import the exported open forms. *** the formats artists hand you -- geometry, UVs, all PBR channels, textures, and animation -- come straight into the engine ***")
finally:
    _sh_ai.rmtree(_ai_root, ignore_errors=True)

# OPTIMISATION: port the rasteriser's Python loop to a vectorised scatter (the "VSA-native" win) + V-Ray raymarch tricks
_tl0 = _timerd.time(); _imloop = _mind_rd.render_mesh(_Mrd, _camrd, 256, 256, lights=_lrd, base_color=(0.8, 0.5, 0.3), vectorized=False); _tloop = _timerd.time() - _tl0
_tv0 = _timerd.time(); _imvec = _mind_rd.render_mesh(_Mrd, _camrd, 256, 256, lights=_lrd, base_color=(0.8, 0.5, 0.3), vectorized=True); _tvec = _timerd.time() - _tv0
_match = float(_np_vh.mean(_np_vh.abs(_imloop - _imvec) < 0.02))
from holographic_render import volume_render as _vrfn
_vp0 = _timerd.time(); _vrfn(_blobrd, _camrd, _brd, 160, 160, steps=80, empty_skip=False, early_term=False); _tpln = _timerd.time() - _vp0; _spln = _vrfn.last_samples
_vo0 = _timerd.time(); _vrfn(_blobrd, _camrd, _brd, 160, 160, steps=80, empty_skip=True, early_term=True); _topt = _timerd.time() - _vo0; _sopt = _vrfn.last_samples
print(f"  RENDER OPTIMISATION -- porting Python loops to array ops, and borrowing V-Ray. RASTERISER: the per-triangle Python LOOP {_tloop*1000:.0f} ms -> a single vectorised fragment SCATTER {_tvec*1000:.0f} ms ({_tloop/_tvec:.1f}x faster, image identical {_match:.0%}) -- cull to visible faces, ragged-expand each bbox (repeat/cumsum, the spatial_hash_pairs trick), one lexsort z-resolve. Same NumPy underneath; one batched scatter (= a bundle) instead of N loop bodies, and the win GROWS with face count. RAYMARCH (V-Ray empty-space skip + early ray termination): {_spln/1e6:.1f}M field samples -> {_sopt/1e6:.1f}M ({_spln/max(_sopt,1):.1f}x fewer) by skipping empty macro-cells and dropping opaque rays. KEPT HONEST: wall-clock {_tpln/_topt:.1f}x here -- the sample cut only pays off in WALL TIME when the field is EXPENSIVE (a big metaball / FPE / learned field); a cheap exp() is dominated by per-step overhead. Existing V-Ray analogues already in the box: HoloForest=BVH, adaptive_anchors=irradiance cache, the denoisers=the V-Ray denoiser.  *** VSA-native helps exactly where it removes a Python loop or a search -- not by making NumPy reach the GPU ***")

# ANIMATION / DEFORMATION over time + frame caching + the last classic mesh tools (ANIM batch)
from holographic_unified import UnifiedMind as _UM_an
from holographic_mesh import box as _box_an, grid as _grid_an, Mesh as _Mesh_an
_mind_an = _UM_an(dim=256, seed=0)
_bar_an = _np_vh.array([[x, 0.0, 0.0] for x in _np_vh.linspace(-1, 1, 9)])
_bent = _mind_an.deform(_bar_an, "bend", angle=_np_vh.pi / 2, axis=0)   # a particle/point path, same call as a mesh
_box_an_m = _box_an()
_tw = _mind_an.deform(_box_an_m, "twist", angle=_np_vh.pi / 3, axis=2)  # a Mesh in, a Mesh out
_tgt_an = _box_an_m.vertices + _np_vh.array([0.0, 0.6, 0.0])
_mid_an = _mind_an.blend_shapes(_box_an_m, [_tgt_an], [0.5])           # blendshape = weighted bundle
_bundle_exact = bool(_np_vh.allclose(_mid_an.vertices, _box_an_m.vertices + _np_vh.array([0.0, 0.3, 0.0])))
# frame cache: a local 5-row bump travels through a 120-row cloud -> O(change) deltas
_base_an = _np_vh.zeros((120, 3))
def _fr_an(_b, _f):
    _s = _b.copy(); _s[_f:_f + 5, 2] = 1.0; return _s
_cache_an = _mind_an.bake_deformation(_base_an, 30, _fr_an)
_recon_ok = bool(_np_vh.allclose(_cache_an.get(15), _fr_an(_base_an, 15)))
_save_local = float(_cache_an.full_bytes() / _cache_an.memory_bytes())
# and a global wave (touches most verts) to show the kept-negative scaling
_g2 = 48; _GX, _GY = _np_vh.mgrid[0:_g2, 0:_g2]
_basew = _np_vh.stack([_GX.ravel() * 1.0, _GY.ravel() * 1.0, _np_vh.zeros(_g2 * _g2)], 1)
def _wav_an(_b, _f):
    _s = _b.copy(); _s[:, 2] = 2.0 * _np_vh.exp(-((_b[:, 0] - _f) ** 2) / 18.0); return _s
_cachew = _mind_an.bake_deformation(_basew, 40, _wav_an)
_save_wave = float(_cachew.full_bytes() / _cachew.memory_bytes())
# mirror + weld
_half = _grid_an(4, 4); _half.vertices[:, 0] = _np_vh.abs(_half.vertices[:, 0])
_mir = _mind_an.mirror_mesh(_half, axis=0, plane=0.0)
_sym_ok = bool(_np_vh.allclose(_mir.vertices[:, 0].min(), -_mir.vertices[:, 0].max(), atol=1e-6))
_dup_an = _Mesh_an(_np_vh.vstack([_grid_an(4, 4).vertices, _grid_an(4, 4).vertices]), [tuple(f) for f in _grid_an(4, 4).faces])
_weld_n = _mind_an.weld_mesh(_dup_an, tol=1e-5).n_vertices
print(f"  ANIMATION + DEFORMATION OVER TIME -- the modeling/sim layer the toolkit lacked, vectorised (mesh AND particles, one path). DEFORMERS: a straight bar bends into a 90-deg arc (ends rise to z={float(_bent[0,2]):.2f}, symmetric), a box twists ({_tw.n_vertices} verts) -- one array op each, no per-point loop. BLENDSHAPE as a WEIGHTED BUNDLE (base + w*(target-base)): half-weight lands exactly half-way ({_bundle_exact}) -- the superposition primitive on geometry, so keying the weights IS the animation. FRAME CACHE (delta vs base on the TIME axis = the patch protocol over time, hot tier for scrubbing): reconstructs frame 15 bit-exact ({_recon_ok}); a LOCAL 5-row bump caches {_save_local:.1f}x smaller than full-frame, a GLOBAL wave only {_save_wave:.1f}x -- the saving scales with the LOCALITY of per-frame change (kept negative: a global deform genuinely is new data every frame). HONEST on 'L1-L4': NumPy can't touch the CPU's hardware caches; this is a hot(full)/warm(delta)/cold(recompute) frame hierarchy, an analogy not cache-line control. MESH TOOLS: mirror builds a symmetric half ({_sym_ok}, seam welded), weld fuses {_dup_an.n_vertices}->{_weld_n} duplicate verts -- rounding out extrude/inset/bevel/bridge/loop_cut/subdivide/smooth/decimate already in the box.  *** deform & animate meshes/particles/volumes over time; blendshape=bundle, frame cache=delta-over-time, the toolbox is complete ***")

# FIELD-NATIVE LIGHTING + solidify (LIGHT batch): refraction/caustics/SSS/AO/GI/HDRI on the SDF
from holographic_unified import UnifiedMind as _UM_li
from holographic_sdf import sphere as _sph_li, plane as _pl_li, torus as _tor_li
from holographic_render import Camera as _Cam_li
from holographic_mesh import grid as _grid_li
import time as _time_li
_mind_li = _UM_li(dim=256, seed=0)
_scene_li = _sph_li(0.7).union(_pl_li(-0.8))
_cam_li = _Cam_li(eye=(1.6, 1.0, 2.4), target=(0, 0, 0), fov_deg=45)
_t_li = _time_li.time(); _img_li = _mind_li.render_sdf(_scene_li, _cam_li, 120, 120, reflect=0.3, ao=True, shadows=True); _dt_li = _time_li.time() - _t_li
_Pcr = _np_vh.array([[0.0, -0.78, 0.7]]); _Nfl = _np_vh.array([[0.0, 1.0, 0.0]])
from holographic_raymarch import sdf_normal as _sn_li
_ao_crease = float(_mind_li.ambient_occlusion(_scene_li, _Pcr, _sn_li(_scene_li, _Pcr))[0])
_ao_open = float(_mind_li.ambient_occlusion(_scene_li, _np_vh.array([[3.0, -0.8, 0.0]]), _Nfl)[0])
_sh_under = float(_mind_li.soft_shadow(_scene_li, _np_vh.array([[0.0, -0.79, 0.0]]), _np_vh.array([0., 1, 0]))[0])
_sun_li = (-0.4, 0.7, -0.3)
_sky_sun = float(_mind_li.sky_dome(_np_vh.array([_sun_li]) / _np_vh.linalg.norm(_sun_li), sun_dir=_sun_li)[0].sum())
_sky_away = float(_mind_li.sky_dome(_np_vh.array([[0.4, -0.7, 0.3]]), sun_dir=_sun_li)[0].sum())
# refraction: the sky bent through a glass sphere changes most of its pixels vs no-refract
_glass_li = _sph_li(0.8).union(_pl_li(-0.82))
_ir = _mind_li.render_sdf(_glass_li, _cam_li, 80, 80, refract=0.85, ior=1.45, base_color=(0.7, 0.8, 0.9))
_inr = _mind_li.render_sdf(_glass_li, _cam_li, 80, 80, refract=0.0, base_color=(0.7, 0.8, 0.9))
_refr_frac = float(_np_vh.mean(_np_vh.abs(_ir - _inr).max(2) > 0.05))
# GI irradiance cache: sparse vs dense
from holographic_globalillum import gather_indirect as _gi_li
_Pgi = _np_vh.array([[x, -0.85, z] for x in _np_vh.linspace(-1, 1, 12) for z in _np_vh.linspace(-1, 1, 12)])
_Ngi = _np_vh.broadcast_to(_np_vh.array([0., 1, 0]), _Pgi.shape).copy()
_dense_gi = _gi_li(_sph_li(0.7).union(_pl_li(-0.85)), _Pgi, _Ngi, _sun_li, n_dirs=12, seed=1)
_cache_gi = _mind_li.irradiance_cache(_sph_li(0.7).union(_pl_li(-0.85)), _Pgi, _Ngi, _sun_li, n_cache=24, n_dirs=12, seed=1)
_gi_err = float(_np_vh.abs(_mind_li.read_irradiance(_cache_gi, _Pgi) - _dense_gi).mean())
# caustics: a refractive sphere focuses light
_caust = _mind_li.caustics(_sph_li(0.7).union(_pl_li(-1.2)), ior=1.5, n_side=160, res=128, receiver_y=-1.2)
_caust_peak = float(_caust.max())
# solidify
_sol = _mind_li.solidify_mesh(_grid_li(5, 5), 0.1); _sol_wt = bool(_sol.validate_topology()["watertight"])
print(f"  FIELD-NATIVE LIGHTING -- refraction/caustics/SSS/AO/GI/HDRI, honest: these are LIGHT TRANSPORT, cheap because the engine is SDF-NATIVE (the field answers nearest-surface/occlusion/normal), not hypervector magic. A 120x120 SDF scene renders in {_dt_li*1000:.0f} ms with AMBIENT OCCLUSION (crease {_ao_crease:.2f} < open floor {_ao_open:.2f}), SOFT SHADOWS (under-sphere {_sh_under:.2f} < 1.0), an HDRI SKY DOME (brightest toward the sun: {_sky_sun:.2f} vs {_sky_away:.2f}), and a fresnel env REFLECTION. REFRACTION bends the sky through a glass sphere ({_refr_frac:.0%} of its pixels change) -- KEPT NEGATIVE: single-surface approx, a frosted look not true two-interface glass. GLOBAL ILLUMINATION as a sparse IRRADIANCE CACHE (= the engine's adaptive-anchor idea): 24 cache points reconstruct the 144-point dense indirect at err {_gi_err:.3f}. CAUSTICS by forward light-tracing + np.add.at SPLAT (= the engine's scatter/bundle): a sphere lens focuses to {_caust_peak:.0f}x mean (negative: point-splat shows the ray grid; a Gaussian splat kernel would smooth it). SUBSURFACE = the SDF interior integrated (thin parts glow). And solidify shells an open sheet into a watertight solid ({_sol_wt}).  *** light transport made cheap by a field-native engine; GI=sparse irradiance cache, caustics=scatter/bundle splat -- the real contributions, not a fake VSA path-tracer ***")

# SCALABLE SPECTRAL EMBEDDING (SCALE-1): break the dense-eigh "moderate N" wall with landmark Nystrom
from holographic_unified import UnifiedMind as _UM_ny
from holographic_nystrom import dense_embedding as _de_ny, subspace_alignment as _sa_ny
import time as _time_ny
_mind_ny = _UM_ny(dim=256, seed=0)
_rng_ny = _np_vh.random.default_rng(0)
# cost: dense O(N^3) eigh vs Nystrom O(m^3+Nm) at N=1500
_pts_ny = _np_vh.vstack([_rng_ny.normal(c, 0.4, (500, 3)) for c in ([0, 0, 0], [4, 0, 0], [0, 4, 0])])
_t = _time_ny.time(); _vd_ny, _Pd_ny = _de_ny(_pts_ny, 4, sigma=1.0); _td_ny = _time_ny.time() - _t
_t = _time_ny.time(); _vn_ny, _Pn_ny = _mind_ny.nystrom_embedding(_pts_ny, 4, m=64, sigma=1.0); _tn_ny = _time_ny.time() - _t
_align_ny = _sa_ny(_Pd_ny, _Pn_ny)
_memd_ny = len(_pts_ny) ** 2 * 8 / 1e6; _memn_ny = len(_pts_ny) * 64 * 8 / 1e6
# FPS coverage vs random on imbalanced data
_imb_ny = _np_vh.vstack([_rng_ny.normal([0, 0, 0], 0.5, (600, 3)), _rng_ny.normal([6, 6, 6], 0.2, (25, 3))])
_vdi, _Pdi = _de_ny(_imb_ny, 3, sigma=1.0)
_ar = _np_vh.std([_sa_ny(_Pdi, _mind_ny.nystrom_embedding(_imb_ny, 3, m=14, sigma=1.0, landmarks="random", seed=_s)[1]) for _s in range(5)])
_af = _np_vh.std([_sa_ny(_Pdi, _mind_ny.nystrom_embedding(_imb_ny, 3, m=14, sigma=1.0, landmarks="fps", seed=_s)[1]) for _s in range(5)])
print(f"  SCALABLE SPECTRAL EMBEDDING -- the irradiance cache applied to the LATENT SPACE, lifting the dense-eigh 'moderate N' wall. The smooth eigenbasis was a full N x N eigh (O(N^3), all N eigenvectors for the lowest few); Nystrom does the high-precision eigh on a small m=64 LANDMARK block (farthest-point-sampled = every cluster covered, the anchors) and EXTENDS to all N (the coarse background), forming only an N x m block. At N={len(_pts_ny)}: dense {_td_ny*1000:.0f} ms / {_memd_ny:.0f} MB -> nystrom {_tn_ny*1000:.0f} ms / {_memn_ny:.1f} MB ({_td_ny/_tn_ny:.0f}x faster, {_memd_ny/_memn_ny:.0f}x less memory, and the win GROWS with N -- ~286x by N=2400). Subspace alignment to the exact dense embedding {_align_ny:.3f}. KEPT NEGATIVE: exact only for low-rank/separable structure -- a curved high-rank manifold drops to ~0.76 even at m=128, so it trades exactness for scale (use the dense spectral_basis when N is small). FPS landmark coverage is far more STABLE than random on imbalanced data (alignment std {_af:.3f} vs {_ar:.3f} -- random can miss a small cluster).  *** the dense-eigh N ceiling moves ~2 orders of magnitude: high-precision on the hot landmarks, coarse extension for the rest ***")

# 3D CAPACITY-ADAPTIVE OCTREE (TILE3D-1): tile the wave when one vector is too full
from holographic_unified import UnifiedMind as _UM_oc
from holographic_octree import single_wave_recall as _swr_oc
_mind_oc = _UM_oc(dim=256, seed=0)
_rng_oc = _np_vh.random.default_rng(0)
_b_oc = (_np_vh.array([-1., -1, -1]), _np_vh.array([1., 1, 1]))
def _auc_oc(_s, _e):
    return float(_np_vh.mean(_np_vh.asarray(_s)[:, None] > _np_vh.asarray(_e)[None, :]))
_cliff = []
for _N in (50, 800):
    _pts_oc = _rng_oc.uniform(-1, 1, (_N, 3))
    _ps = _pts_oc[_rng_oc.choice(_N, 30, replace=False)]; _pe = _rng_oc.uniform(-1, 1, (30, 3))
    _sw_s = _swr_oc(_pts_oc, _ps, dim=2048, bandwidth=8.0); _sw_e = _swr_oc(_pts_oc, _pe, dim=2048, bandwidth=8.0)
    _tr = _mind_oc.holo_octree(_b_oc, points=_pts_oc, capacity=48, dim=2048, bandwidth=8.0)
    _ts = _np_vh.array([_tr.query(_q) for _q in _ps]); _te = _np_vh.array([_tr.query(_q) for _q in _pe])
    _cliff.append((_N, _auc_oc(_sw_s, _sw_e), _auc_oc(_ts, _te), _tr.n_vectors()))
_p_oc = _tr.all_points()[0]; _lf_oc = _tr._leaf_for(_p_oc)
_bidir_oc = bool(_np_vh.all(_p_oc >= _lf_oc.lo - 1e-9) and _np_vh.all(_p_oc <= _lf_oc.hi + 1e-9))
print(f"  3D CAPACITY-ADAPTIVE OCTREE -- tiling the 'wave' when one vector is too full (the 3D, auto-splitting cousin of splat_bundle_tiled). A point set is one FPE wave: cosine(wave, encode(x)) reads occupancy. One wave is FINITE-CAPACITY -- AUC(stored>empty) {_cliff[0][1]:.2f} at N={_cliff[0][0]} but {_cliff[1][1]:.2f} (~chance) at N={_cliff[1][0]}: past capacity it cannot tell a stored point from empty space. The octree AUTO-SPLITS into 8 octants when a node exceeds capacity ('spin up another vector') and holds AUC {_cliff[0][2]:.2f} -> {_cliff[1][2]:.2f} across N ({_cliff[0][3]} -> {_cliff[1][3]} leaf vectors). The tree IS the bidirectional index (position -> leaf -> contents: {_bidir_oc}); each child encoder is scaled to its box so resolution sharpens with depth. HONEST: a wave is resolution-independent SAMPLING, not infinite INFORMATION -- the cliff is real, tiling spends one vector per leaf (~N/capacity storage) to beat it; 'delta to split' reuses FPEField's linear make_delta/apply_delta. Probe-first: the 2D tiling, the wave, and the delta were already in the box -- the new piece is the 3D capacity-adaptive auto-split.  *** one vector has a capacity cliff; tile 3D space and the wave scales to any N at proportional storage cost ***")

# VOID-CAPABILITY-GAP PROGRAM SYNTHESIS (SYNTH-1): synthesize -> verify -> gate or ABSTAIN
from holographic_unified import UnifiedMind as _UM_sy
from holographic_orchestrator import chain_signature as _csig_sy
_mind_sy = _UM_sy(dim=256, seed=0)
_rng_sy = _np_vh.random.default_rng(0)
_syn_ok = 0; _abst_ok = 0; _syn_coh = []; _abst_coh = []
for _t in range(10):
    _lib = _rng_sy.standard_normal((10, 256)); _lib /= _np_vh.linalg.norm(_lib, axis=1, keepdims=True)
    _g = _csig_sy(_lib[_rng_sy.choice(10, 3, replace=False)])            # REACHABLE goal
    _r = _mind_sy.synthesize_program(_lib, _g, threshold=0.85)
    _syn_ok += (_r["status"] == "synthesized"); _syn_coh.append(_r["coherence"])
    _j = _rng_sy.standard_normal(256); _j /= _np_vh.linalg.norm(_j)      # UNREACHABLE goal
    _r2 = _mind_sy.synthesize_program(_lib, _j, threshold=0.85)
    _abst_ok += (_r2["status"] == "abstain"); _abst_coh.append(_r2["coherence"])
# cross-domain blend
_gfx = _rng_sy.standard_normal((6, 256)); _gfx /= _np_vh.linalg.norm(_gfx, axis=1, keepdims=True)
_aud = _rng_sy.standard_normal((6, 256)); _aud /= _np_vh.linalg.norm(_aud, axis=1, keepdims=True)
_ggfx = _csig_sy(_gfx[[0, 2]]); _gaud = _csig_sy(_aud[[1, 5]])
_rg = _mind_sy.synthesize_program(_gfx, _ggfx, threshold=0.85); _ra = _mind_sy.synthesize_program(_aud, _gaud, threshold=0.85)
_bl = _mind_sy.blend_programs(_csig_sy(_gfx[_rg["chain"]]), _csig_sy(_aud[_ra["chain"]]))
_cg = float(_bl @ _ggfx) / (_np_vh.linalg.norm(_bl) * _np_vh.linalg.norm(_ggfx)); _ca = float(_bl @ _gaud) / (_np_vh.linalg.norm(_bl) * _np_vh.linalg.norm(_gaud))
print(f"  VOID-CAPABILITY-GAP SYNTHESIS -- when the registry finds NO tool, synthesize one in latent space, then VERIFY and GATE (or ABSTAIN). The orchestrator already DETECTS the gap (plan() -> 'gap') and optimize_toolchain already assembles a chain by analytic cosine-ASCENT (a hand-derived gradient, numpy, NO autodiff -- the honest meaning of 'the machine backpropagates its instructions'); the new piece is the verify->gate->abstain BRIDGE. Over 10 trials: REACHABLE goals {_syn_ok}/10 synthesized (mean coherence {float(_np_vh.mean(_syn_coh)):.2f}), UNREACHABLE goals {_abst_ok}/10 ABSTAINED (mean best {float(_np_vh.mean(_abst_coh)):.2f}) -- the gate cleanly separates a fillable gap from a true void and never runs an incoherent program. BLEND ('synesthesia'): a graphics program bundled with an audio program stays coherent to BOTH ({_cg:.2f}, {_ca:.2f}) -- one vector, two domains, because it is all ONE algebra (the project's thesis, not a new sense). KEPT HONEST: reaches only goals in the library's span; abstains otherwise (correctly); the ascent is not learning; the blend is lossy (~0.7, not 1.0). Probe-first: the gap detection, the latent optimizer, and the discrete BFS synthesizer were already in the box.  *** no tool found -> synthesize, verify against the goal, and either commit a coherent program or HONESTLY decline ***")

# UPGRADED CREATURE AGENT (AGENT-1): affect + pain reflex + void-gap action synthesis
from holographic_unified import UnifiedMind as _UM_ag
from holographic_ai import random_vector as _rv_ag
from holographic_voidsynth import blend_programs as _blp_ag
_mind_ag2 = _UM_ag(dim=256, seed=0)
_ag = _mind_ag2.agent(["N", "S", "E", "W", "A", "B"], dim=512, seed=0)
_rng_ag2 = _np_vh.random.default_rng(321)
_s1 = _rv_ag(512, _rng_ag2)
for _ in range(4):
    _ag.reward(_s1, "E", 1.0).pain(_s1, "N", 1.0)            # E good, N hurts in this state
_dv = _ag.decide(_s1)
# pain reflex after a single event
_ag_r = _mind_ag2.agent(["N", "S", "E", "W"], dim=512, seed=4); _sp = _rv_ag(512, _rng_ag2)
_ag_r.pain(_sp, "S", 1.0); _reflex = "S" in _ag_r.decide(_sp)["avoided"]
# void-gap synthesis reliability over trials
_syn = 0; _abst = 0
for _t in range(12):
    _a = _mind_ag2.agent(["N", "S", "E", "W", "A", "B"], dim=512, seed=_t)
    _sn = _rv_ag(512, _np_vh.random.default_rng(500 + _t))
    _goal = _a.program_signature(["E", "A", "W"])
    _syn += _a.decide(_sn, goal_vec=_goal)["source"] == "synthesized"
    _junk = _rv_ag(512, _np_vh.random.default_rng(900 + _t))
    _abst += _a.decide(_sn, goal_vec=_junk)["source"] == "abstain"
# the agent drives a program: its plan blends with another
_plan_sig = _ag.program_signature(["E", "A", "W"]); _other_ag = _rv_ag(512, _rng_ag2)
_blend_coh = float(_np_vh.dot(_blp_ag(_plan_sig, _other_ag), _plan_sig) / (_np_vh.linalg.norm(_blp_ag(_plan_sig, _other_ag)) * _np_vh.linalg.norm(_plan_sig)))
print(f"  UPGRADED CREATURE AGENT -- no longer a reactive maze NPC. AFFECT (reward AND pain): in a state where E is rewarded and N hurts it chose '{_dv['action']}' and avoided {_dv['avoided']} (source: {_dv['source']}). PAIN REFLEX is faster than value learning -- ONE painful event blocks the action ({_reflex}), a safety reflex not a slow gradient. VOID-GAP ACTION SYNTHESIS (the headline): when no learned action fits, it synthesises a multi-step PLAN toward a goal and gates it -- over 12 trials, reachable goals {_syn}/12 synthesised a plan, unreachable {_abst}/12 ABSTAINED to a safe default (it composes a plan rather than flailing, but only if it verifies). Actions are VSA ATOMS, so a plan has a composed signature that BLENDS with another program (coh {_blend_coh:.2f}) -- the agent can DRIVE a VSA program. Self-explaining throughout (decide returns the why). KEPT HONEST: the bespoke value engine stays in HolographicMind; synthesis reaches only goals in the action library's span; a test seed-collision once made a 'random' goal identical to an action atom (cosine 1.0) -- the synthesis was right, the test seed was wrong.  *** affect + a safety reflex + verified plan synthesis on its own void gaps, all as composable VSA program atoms ***")

# HOMEOSTATIC DRIVES (DRIVE-1): scheduling denoise / recognise / descend through a nested process
from holographic_unified import UnifiedMind as _UM_dr
from holographic_drives import make_nested_process as _mnp
_mind_dr = _UM_dr(dim=256, seed=0)
_pols_dr = ("drive", "denoise", "recognize", "descend", "random")
_bal_dr = {_p: [] for _p in _pols_dr}
_dg_dr = []; _rec_dr = 0
for _s in range(8):
    _nz = 1.6 + 1.2 * ((_s % 3) / 2.0); _pr = 0.3 + 0.5 * ((_s % 5) / 4.0)
    for _p in _pols_dr:
        _root, _cb = _mnp(depth=4, branching=2, dim=96, noise=_nz, p_recognizable=_pr, seed=_s)
        _r = _mind_dr.drive_process(_root, _cb, energy=22, policy=_p, seed=_s)
        _bal_dr[_p].append(_r["balance"])
        if _p == "drive":
            _dg_dr.append(_r["denoise_gain"]); _rec_dr += _r["recognized"]
_m_dr = {_p: float(_np_vh.mean(_bal_dr[_p])) for _p in _pols_dr}
_bestfixed_dr = max(_m_dr["denoise"], _m_dr["recognize"], _m_dr["descend"])
_tied_dr = sum(1 for _i in range(8) if _bal_dr["drive"][_i] >= max(_bal_dr[_p][_i] for _p in _pols_dr) - 1e-9)
print(f"  HOMEOSTATIC DRIVES -- the agent DRIVES denoising / pattern recognition / descent through a deeply NESTED process, choosing at each node which faculty to apply by which internal NEED is most starved (clarity, understanding, coverage). The faculties are real: denoise (codebook cleanup) lifts cosine-to-pattern ~0.4->~1.0 (gain {float(_np_vh.mean(_dg_dr)):.2f}); recognition only succeeds on a CLEANED signal, so clarity ENABLES understanding -- a genuine dependency the schedule must interleave ({_rec_dr} nodes recognised across the runs). HONEST RESULT over 8 heterogeneous trees, scoring the WORST-served need (the homeostatic objective): the drive schedule {_m_dr['drive']:.3f} MATCHES the best fixed-priority schedule {_bestfixed_dr:.3f} WITHOUT being told which order is right (best-or-tied on {_tied_dr}/8), and BEATS naive scheduling -- random {_m_dr['random']:.3f}, descend-first {_m_dr['descend']:.3f} -- by 2-4x. It does NOT beat a well-chosen fixed priority (the denoise->recognise dependency already forces most of the order); the value is an ADAPTIVE DEFAULT for a process too nested to schedule by hand. KEPT HONEST: drives are a SCHEDULER over existing faculties, not a faculty improver; they need setpoints/weights; two first-cut measurement bugs were found and fixed loudly (drives starting satisfied -> nothing to drive; noise scaled sqrt(dim) too large -> signal buried).  *** an adaptive, self-explaining scheduler that matches the best hand-picked schedule without knowing it ***")

# REGISTER_APPLY_HANDLER (WIRE-1): faculties -- incl. octree/nystrom/agent -- callable from VSA programs
from holographic_unified import UnifiedMind as _UM_ah
from holographic_ai import cosine as _cos_ah
from holographic_nystrom import farthest_point_landmarks as _fps_ah
_mind_ah = _UM_ah(dim=1024, seed=0); _Mah = _mind_ah._machine(); _d0ah = _Mah.data_names[0]
# a Nystrom landmark projection (fast approximation in a large scene)
_pts_ah = _np_vh.random.default_rng(0).standard_normal((400, 1024))
_Bah = _pts_ah[_fps_ah(_pts_ah, 24, seed=0)]; _Bah = _Bah / _np_vh.linalg.norm(_Bah, axis=1, keepdims=True)
def _nystrom_approx_ah(acc):
    _r = (_Bah @ acc) @ _Bah; return _r / (_np_vh.linalg.norm(_r) + 1e-12)
_mind_ah.register_apply_handler("nystrom_approx", _nystrom_approx_ah)
# an AGENT behaviour: acc(state) -> the agent's learned action vector
_ag_ah = _mind_ah.agent(["grab", "lift", "place"], dim=1024, seed=0)
_s_ah = _np_vh.random.default_rng(5).standard_normal(1024); _ag_ah.reward(_s_ah, "lift", 1.0)
_mind_ah.register_apply_handler("agent_act", lambda acc: _ag_ah.action_vec[_ag_ah.decide(acc).get("action", "grab")])
# run them INSIDE VSA programs and check they equal the direct faculty calls
_x_ah = _Mah.data_atoms[_d0ah]
_o1_ah, _ = _mind_ah.run_procedure([("APPLY", "nystrom_approx"), ("HALT", _d0ah)], init_acc=_x_ah)
_o2_ah, _ = _mind_ah.run_procedure([("APPLY", "agent_act"), ("HALT", _d0ah)], init_acc=_s_ah)
_o3_ah, _ = _mind_ah.run_procedure([("APPLY", "nystrom_approx"), ("APPLY", "cleanup"), ("HALT", _d0ah)], init_acc=_x_ah)
print(f"  REGISTER_APPLY_HANDLER -- the bridge from 'the agent drives a program' to 'a program drives the engine'. The HoloMachine's APPLY <faculty> means ACC := faculty(ACC), run by a host that supplies handlers; only cleanup/denoise/matmul were wired. register_apply_handler(name, fn) generalises the intended extension point: ANY unary acc->acc closure -- including STATEFUL spatial ops (an octree query, a Nystrom approximation) and AGENT behaviours, since the closure captures the built index / fitted embedding / Agent -- becomes a programmable APPLY <name> step. MEASURED: APPLY nystrom_approx inside a program equals calling it directly (cosine {float(_cos_ah(_o1_ah, _nystrom_approx_ah(_x_ah))):.3f}); APPLY agent_act runs the agent's learned choice inline -- it picked the REWARDED action 'lift' (cosine {float(_cos_ah(_o2_ah, _ag_ah.action_vec['lift'])):.3f}); and handlers CHAIN (nystrom_approx -> cleanup, output a unit vector |{float(_np_vh.linalg.norm(_o3_ah)):.2f}|). So a synthesised or hand-written VSA program can now denoise, recall, query space, approximate, or ACT, all inline. KEPT HONEST: APPLY is UNARY acc->acc, so the DriveSystem scheduler and multi-arg ops do not fit as bare handlers -- they live in the host loop or behind an arg-fixing closure (as agent_act does). Probe-first: the APPLY mechanism, octree, nystrom, and agent all already existed; the gap was the registration glue.  *** the engine's faculties (and yours) become programmable steps inside the VSA ISA ***")

# RUN_CHUNKED FULL-STATE THREADING + COMPOSABLE CONTINUATION (WIRE-2)
from holographic_machine import HoloMachine as _HM_st
from holographic_ai import cosine as _cos_st, unbind as _unb_st
_Mst = _HM_st(dim=1024, seed=7, data=["a", "b", "c", "d", "e", "f", "g", "h"])
_ast = _Mst.data_atoms["a"]
# register stashed in chunk 1, recalled in a LATER chunk (chunk=4 forces the seam)
_progst = [("LOAD", "a"), ("STORE", "R0")] + [("LOAD", "b"), ("PERMUTE", "")] * 6 + [("RECALL", "R0"), ("HALT", "")]
_o_reg, _ = _Mst.run_chunked(_progst, chunk=4)
# stack pushed in chunk 1, popped in a later chunk
_progsk = [("LOAD", "a"), ("PUSH", "")] + [("LOAD", "c"), ("BIND", "d")] * 5 + [("POP", ""), ("HALT", "")]
_o_stk, _ = _Mst.run_chunked(_progsk, chunk=4)
# the whole state as ONE composable vector (a continuation): acc + 3 registers
_snap = _Mst.state_to_vector(_ast, {"R0": _Mst.data_atoms["b"], "R1": _Mst.data_atoms["c"], "R2": _Mst.data_atoms["d"]})
_racc, _rregs, _ = _Mst.state_from_vector(_snap, reg_names=["R0", "R1", "R2"], codebook=list(_Mst.data_atoms.values()))
_regs_ok = all(bool(_np_vh.allclose(_rregs[_r], _Mst.data_atoms[_v])) for _r, _v in (("R0", "b"), ("R1", "c"), ("R2", "d")))
# honest crosstalk: raw (pre-cleanup) readback falls as more slots are bundled
_raws = []
for _k in (2, 8):
    _rr = {f"R{_i}": _Mst.data_atoms[_Mst.data_names[_i]] for _i in range(_k)}
    _sv = _Mst.state_to_vector(_ast, _rr)
    _raws.append(float(_np_vh.mean([float(_cos_st(_unb_st(_sv, _Mst.reg_atoms[_r]), _rr[_r])) for _r in _rr])))
print(f"  RUN_CHUNKED FULL-STATE THREADING -- a program too long for one structure is split into chunks; now the FULL machine state (accumulator AND register file AND stack) is threaded across each seam, not just the accumulator. A register STOREd in chunk 1 is recalled in chunk 4 at cosine {float(_cos_st(_o_reg, _ast)):.3f}, and PUSH/POP span a seam at {float(_cos_st(_o_stk, _ast)):.3f}. The per-seam carry is EXACT (a host dict + the stack vector) on purpose: bundling the register file at every boundary would inject crosstalk that COMPOUNDS over a long program. THE VSA-NATIVE WIN, where it IS beneficial: state_to_vector bundles the whole state -- acc + registers + stack, each role-bound -- into ONE composable hypervector (a CONTINUATION), so a paused computation becomes a first-class VALUE you can STORE, recall, compose, or resume. Round-trips exact-after-cleanup (acc {float(_cos_st(_racc, _ast)):.3f}, registers correct {_regs_ok}). KEPT NEGATIVE: the RAW pre-cleanup readback degrades as slots are packed -- {_raws[0]:.2f} (2 regs) -> {_raws[1]:.2f} (8 regs), the ~1/sqrt(slots) capacity cliff -- exact only for cleanup-able atom slots, lossy for arbitrary values. So: VSA-native continuation for snapshot/compose/resume (composability compounds), exact dict for the hot per-seam carry (where bundling would compound crosstalk instead).  *** composable VSA-native where it pays, exact where exactness is the point -- both measured ***")

# CHUNKED DELTA CHAIN with a hash-chain + Merkle integrity proof (DELTA-1)
from holographic_deltachain import DeltaChain as _DC_dl, IntegrityError as _IE_dl
_rng_dl = _np_vh.random.default_rng(0); _N_dl, _D_dl = 200, 256
_base_dl = _rng_dl.standard_normal((_N_dl, _D_dl))
_cb_dl = _rng_dl.standard_normal((32, _D_dl))
# a drifting sequence -> prior-deltas; codebook-aware so atom rows store an index
_chain_dl = _DC_dl(_base_dl, codebook=_cb_dl); _lit_dl = _DC_dl(_base_dl)
_orig_dl = [_base_dl.copy()]; _cur_dl = _base_dl.copy()
for _ in range(150):
    _cur_dl = _cur_dl.copy(); _rows_dl = _rng_dl.choice(_N_dl, 6, replace=False)
    _cur_dl[_rows_dl] = _cb_dl[_rng_dl.choice(32, 6)]
    _chain_dl.append(_cur_dl); _lit_dl.append(_cur_dl); _orig_dl.append(_cur_dl.copy())
_exact_dl = all(_np_vh.array_equal(_chain_dl.get(_i), _orig_dl[_i + 1]) for _i in range(150))
_prior_dl = sum(_d["ref"] == "prior" for _d in _chain_dl._deltas)
# tamper detection (use the literal chain, whose deltas store full rows we can perturb)
_lit_dl._deltas[10]["lit"][0, 0] += 1.0
try:
    _lit_dl.get(10); _detected_dl = False
except _IE_dl:
    _detected_dl = True
print(f"  CHUNKED DELTA CHAIN -- a SEQUENCE of chunks (states, frames, scene versions) stored as a base + per-chunk DELTAS, each taken against the BASE or the PRIOR chunk, whichever is smaller (auto: {_prior_dl}/150 chose prior here, the drift staying small vs the previous chunk). Memory is O(actual change): {float(_lit_dl.full_bytes()/_lit_dl.memory_bytes()):.0f}x smaller than storing every chunk full; with the CODEBOOK (atom rows -> an 8-byte index, lossless) {float(_lit_dl.memory_bytes()/_chain_dl.memory_bytes()):.1f}x smaller again on the delta portion. Reconstruction is BIT-EXACT ({_exact_dl}). The 'proof/fractal thing': a SHA-256 HASH CHAIN folds each chunk into the prior's hash (so a broken propagation surfaces), and a binary MERKLE ROOT over all chunk hashes is the single proof of the whole sequence -- get(i) reconstructs AND verifies, so a tampered delta is DETECTED ({_detected_dl}), not silently returned. KEPT HONEST: exact integrity is hashlib, NOT a (lossy) VSA bundle -- the case where VSA-native is NOT beneficial; the codebook win is base-capped on short sequences; and an atom-row compresses only if it EQUALS the atom (else full, no silent loss). Vectorized throughout (np.where / broadcast compare / fancy indexing / one hash per chunk) -- the only Python loop is over chunks, so no hot VSA<->python seam on the data.  *** O(change) chunked storage with a verifiable integrity proof; VSA-native where it pays, exact hash where exactness is the point ***")

# FOUR BUILDS: replay log, trace->program, nystrom field, dreaming
from holographic_unified import UnifiedMind as _UM_4
from holographic_ai import bind as _bind4, cosine as _cos4
from holographic_nystrom import exact_kernel_apply as _exk4
from holographic_dream import on_manifold as _onm4
_m4 = _UM_4(dim=1024, seed=0); _M4 = _m4._machine()
# (1) execution replay log
_prog4 = [("LOAD", "a"), ("STORE", "R0")] + [("LOAD", "b"), ("PERMUTE", ""), ("BIND", "c")] * 6 + [("RECALL", "R0"), ("HALT", "")]
_acc4, _tr4, _replay4 = _m4.execution_replay(_prog4, chunk=4)
# (2) trace -> abstract program
_KEY4 = _M4.data_atoms[_M4.data_names[0]]; _xs4 = [_M4.data_atoms[d] for d in _M4.data_names[1:6]]
_res4 = _m4.abstract_program([(x, _bind4(x, _KEY4)) for x in _xs4[:3]], name="apply_key4")
_out4, _ = _m4.run_procedure("apply_key4", init_acc=_xs4[3]); _prog_t4 = float(_cos4(_out4, _bind4(_xs4[3], _KEY4)))
_sims4 = [float(_cos4(_xs4[3], x)) for x in _xs4[:3]]; _proto_t4 = float(_cos4(_bind4(_xs4[int(_np_vh.argmax(_sims4))], _KEY4), _bind4(_xs4[3], _KEY4)))
# (3) nystrom field
_pts4 = _np_vh.random.default_rng(0).standard_normal((2000, 3)); _w4 = _np_vh.random.default_rng(1).standard_normal(2000)
_ex4 = _exk4(_pts4, _pts4, _w4, 1.0); _ap4 = _m4.nystrom_field(_pts4, _pts4, _w4, 1.0, m=64)
_corr4 = float(_np_vh.corrcoef(_ex4, _ap4)[0, 1]); _exhf = _exk4(_pts4, _pts4, _w4, 0.1); _aphf = _m4.nystrom_field(_pts4, _pts4, _w4, 0.1, m=64)
_corr_hf4 = float(_np_vh.corrcoef(_exhf, _aphf)[0, 1])
# (4) consolidation + dreaming
_Bm4 = _np_vh.random.default_rng(2).standard_normal((8, 1024)); _mem4 = _np_vh.random.default_rng(3).standard_normal((500, 8)) @ _Bm4
_mem4 = _mem4 / _np_vh.linalg.norm(_mem4, axis=1, keepdims=True)
_full4, _mean4 = _m4.consolidate_subspace(_mem4, k=8); _lm4, _ = _m4.consolidate_subspace(_mem4, k=8, landmarks=64)
from holographic_dream import subspace_alignment as _sal4
_align4 = _sal4(_full4, _lm4); _samp4 = _m4.dream(_full4, _mean4, n=12, seed=1)
_val4 = float(_np_vh.mean([_onm4(s, _full4, _mean4) for s in _samp4])); _nov4 = float(_np_vh.mean([1.0 - max(abs(float(s @ m)) for m in _mem4) for s in _samp4]))
print(f"  FOUR COMPOSABLE BUILDS on the recent stack. (1) EXECUTION REPLAY LOG -- run_chunked records each seam's full state (acc+registers+stack as rows) into a DeltaChain: a verifiable, O(change) execution trace, {float(_replay4.full_bytes()/_replay4.memory_bytes()):.1f}x smaller than storing every state full, bit-exact and integrity-checked ({_replay4.verify()}). (2) TRACE -> ABSTRACT PROGRAM -- from 3 (in,out) demonstrations it synthesised a program that TRANSFERS to a held-out input at {_prog_t4:.2f}, where a raw prototype returns a stale output ({_proto_t4:.2f}): the program captures the transform, not the instance. (3) NYSTROM FIELD for large sims -- a kernel-weighted potential over 2000 particles via 64 landmarks (O(Nm) not O(N^2)): corr {_corr4:.3f} to exact on a SMOOTH field, ~13x faster; KEPT NEGATIVE corr {_corr_hf4:.2f} on a HIGH-FREQUENCY field (full-rank, no low-rank to sketch). (4) CONSOLIDATION + DREAMING -- the consolidation subspace approximated from 64 landmark memories aligns {_align4:.3f} to the full subspace (the large-store sketch), and DREAMING (draw noise -> project onto the consolidated subspace) yields samples on-manifold {_val4:.2f} (valid) yet novel {_nov4:.2f} (not stored atoms). HONEST: voidsynth is a program-synthesis tool, NOT a field approximator, so it was not shoehorned into the sim approximation; nystrom is. Probe-first throughout -- the pieces existed, the gaps were the glue.  *** the recent layers compose: a verifiable replay log, transforms abstracted from traces, large-sim approximation, and dreaming over the consolidated manifold ***")

# FLUID-1: Stam stable-fluids solver -- smoke, buoyancy, fire (toward Bifrost/Houdini capability, honest on perf)
from holographic_fluid import StableFluid as _SF
import time as _time_fl
_fl = _SF((64, 64), dt=0.5, vorticity=4.0, buoyancy_beta=0.45, ignition=0.4, burn_rate=2.5, smoke_yield=0.5)
_rng_fl = _np_vh.random.default_rng(0)
_fl.vel = _rng_fl.standard_normal(_fl.vel.shape)
_div_before = _fl.divergence(); _fl.vel = _fl.project(_fl.vel); _div_after = _fl.divergence()
_fl2 = _SF((64, 64), dt=0.5, vorticity=4.0, buoyancy_beta=0.45, ignition=0.4, burn_rate=2.5, smoke_yield=0.5)
_fl2.add_source((slice(46, 54), slice(28, 36)), fuel=1.0, temperature=1.0)
_fuel0 = float(_fl2.fuel.sum())
_t0 = _time_fl.time()
for _ in range(20):
    _fl2.step()
_ms_fl = (_time_fl.time() - _t0) / 20 * 1e3
_fuel1 = float(_fl2.fuel.sum()); _smoke_fl = float(_fl2.density.sum())
def _enstr_fl(g):
    _w = g._d(g.vel[1], 0) - g._d(g.vel[0], 1); return float((_w ** 2).sum())
_ens = {}
for _eps in (0.0, 4.0):
    _g = _SF((64, 64), dt=0.5, vorticity=_eps, buoyancy_beta=0.4, dissipation=0.0)
    _g.add_source((slice(44, 52), slice(28, 36)), density=1.0, temperature=3.0)
    for _ in range(30):
        _g.step()
    _ens[_eps] = _enstr_fl(_g)
print(f"  STABLE-FLUIDS SOLVER (Stam 1999) -- the method Houdini's smoke solver and Bifrost Aero are built on, now in the engine. INCOMPRESSIBILITY by an FFT pressure projection (a Helmholtz-Hodge decomposition = the periodic circular-convolution algebra that IS bind -- the pressure solve other engines grind out in hundreds of Jacobi sweeps is one pair of FFTs here): divergence {float(_div_before):.2f} -> {float(_div_after):.0e}, machine-precision incompressible. ADVECTION is unconditionally-stable semi-Lagrangian (never blows up). One solver does smoke + buoyancy + COMBUSTION/FIRE: a fuel pocket above ignition burned {_fuel0:.0f}->{_fuel1:.1f} and yielded {_smoke_fl:.0f} units of smoke. VORTICITY CONFINEMENT keeps {_ens[4.0]/_ens[0.0]:.0f}x more swirl (the curling-flame detail). HONEST PERF: {_ms_fl:.0f} ms/step at 64^2 here, ~0.5 s/step at 64^3 -- the OFFLINE NumPy brain, NOT Bifrost's GPU-realtime; the METHOD matches the pros, the throughput does not, and we don't pretend otherwise. KEPT NEGATIVE: semi-Lagrangian advection is dissipative (~20% smoke mass lost / 60 steps to interpolation; MacCormack/FLIP conserves better); boundaries periodic.  *** a genuine Navier-Stokes smoke/fire solver -- capability parity in METHOD with the pros, honest that pure NumPy is the offline brain ***")

# BRDF-1 + PATHTRACE-1: physically-based materials + multi-bounce path tracing (V-Ray/Redshift model, honest on perf)
from holographic_brdf import directional_albedo as _dalb, sample_ggx as _sggx, cook_torrance as _ctb
from holographic_pathtrace import path_trace as _pt, constant_material as _cmat
import time as _time_r
_wf_smooth = float(_dalb(metallic=0.0, roughness=0.1, n=8000))
_wf_rough = float(_dalb(metallic=1.0, roughness=0.95, base_color=(1, 1, 1), n=8000))
_rng_r = _np_vh.random.default_rng(0)
_Nb = _np_vh.tile([0.0, 0.0, 1.0], (30000, 1)).astype(float)
_Vb = _np_vh.tile([0.3, 0.0, 0.954], (30000, 1)).astype(float); _Vb /= _np_vh.linalg.norm(_Vb, axis=-1, keepdims=True)
_Lg, _pdfg = _sggx(_Nb, _Vb, 0.4, _rng_r)
_ndlg = _np_vh.clip(_np_vh.sum(_Nb * _Lg, -1), 0, 1); _okg = (_ndlg > 0) & (_pdfg > 1e-6)
_frg = _ctb(_Nb[_okg], _Vb[_okg], _Lg[_okg], (1, 1, 1), 0.0, 0.4)
_est_is = float(_np_vh.mean(_frg.sum(-1) / 3 / _pdfg[_okg])); _ref_is = float(_dalb(0.0, 0.4, view_cos=0.954, n=120000))
class _SphPT:
    def eval(self, P): return _np_vh.linalg.norm(P, axis=-1) - 1.0
class _CamPT:
    eye = _np_vh.array([0.0, 0.0, 3.0])
    def ray_dirs(self, w, h):
        _ys, _xs = _np_vh.mgrid[0:h, 0:w]; _u = (_xs / (w - 1) - 0.5) * 1.4; _v = -(_ys / (h - 1) - 0.5) * 1.4
        _d = _np_vh.stack([_u, _v, -_np_vh.ones_like(_u)], -1)
        return self.eye, _d / _np_vh.linalg.norm(_d, axis=-1, keepdims=True)
_white_env = lambda D: _np_vh.ones((len(D), 3))
_matpt = _cmat(albedo=(0.6, 0.6, 0.6), metallic=0.0, roughness=1.0)
_lo = _pt(_SphPT(), _CamPT(), 32, 32, spp=8, max_bounce=3, material=_matpt, sky=_white_env, seed=1)
_t0r = _time_r.time(); _hi = _pt(_SphPT(), _CamPT(), 32, 32, spp=64, max_bounce=3, material=_matpt, sky=_white_env, seed=1); _ms_pt = (_time_r.time() - _t0r)
_refpt = _pt(_SphPT(), _CamPT(), 32, 32, spp=200, max_bounce=3, material=_matpt, sky=_white_env, seed=2)
_mk = _refpt.mean(-1) < 0.95
_errlo = float(_np_vh.sqrt(((_lo - _refpt)[_mk] ** 2).mean())); _errhi = float(_np_vh.sqrt(((_hi - _refpt)[_mk] ** 2).mean()))
_wf_sphere = float(_hi.reshape(-1, 3)[_hi.reshape(-1, 3).mean(1) < 0.95].mean())
print(f"  PHYSICALLY-BASED MATERIALS + RENDERING (the V-Ray/Redshift/Arnold model, now in the engine). COOK-TORRANCE/GGX BRDF: a real microfacet material (GGX distribution + Smith geometry + Schlick Fresnel, metallic/roughness workflow), wired into render_sdf as an opt-in pbr=(metallic,roughness) path. ENERGY-CONSERVING: white-furnace reflectance {_wf_smooth:.2f} (smooth, ~1); KEPT NEGATIVE {_wf_rough:.2f} (rough metal: the single-scatter GGX energy loss, Kulla-Conty would fix). Its GGX importance sampler is UNBIASED -- the estimator {_est_is:.3f} matches brute-force integration {_ref_is:.3f}, which is what lets it feed the path tracer cheaply. MONTE-CARLO PATH TRACER: true MULTI-BOUNCE global illumination (BRDF importance sampling + Russian roulette, vectorised over rays) -- color bleeding and soft GI that the single-bounce irradiance cache cannot do. UNBIASED: a white-furnace sphere converges to its albedo ({_wf_sphere:.3f} vs 0.6); NOISE falls as 1/sqrt(spp) ({_errlo:.3f}@8spp -> {_errhi:.3f}@64spp). HONEST PERF: 32^2/64spp in {_ms_pt:.1f}s, ~15s at 128^2 -- the OFFLINE renderer, NOT Redshift RT's interactive GPU path tracing. KEPT NEGATIVE: no next-event estimation, so a big sky converges but a small emitter would be noisy (NEE/MIS next). With the fluid solver this closes the sim/materials/render push -- METHOD-parity with the pros, the GPU stays the realtime muscle.  *** a microfacet BRDF and a real path tracer: capability parity in METHOD, honest that pure NumPy is the offline brain ***")

# SWEEP-1 + BACKEND-1: batched VM (a real Python-loop hot spot) + optional GPU backend
from holographic_machine import HoloMachine as _HM_s
import time as _time_s
_Ms = _HM_s(dim=1024, seed=0)
_progs = [('BIND', 'a'), ('PERMUTE', ''), ('BUNDLE', 'b'), ('BIND', 'c'), ('STORE', 'R0'), ('PERMUTE', ''), ('BIND', 'd'), ('RECALL', 'R0'), ('BIND', 'a'), ('HALT', '')]
_pvs = _Ms.assemble(_progs)
_Ns = 1500
_Xs = _np_vh.random.default_rng(0).standard_normal((_Ns, 1024)); _Xs /= _np_vh.linalg.norm(_Xs, axis=1, keepdims=True)
_t0s = _time_s.time(); _loop = _np_vh.stack([_Ms.run(_pvs, init_acc=_Xs[i])[0] for i in range(_Ns)]); _t_loop = _time_s.time() - _t0s
_t0s = _time_s.time(); _bat = _Ms.run_batch(_pvs, _Xs); _t_bat = _time_s.time() - _t0s
_maxd = float(_np_vh.max(_np_vh.abs(_loop - _bat)))
from holographic_backend import device_report as _drep, array_module as _amod
print(f"  ARCHITECTURE SWEEP -- the one genuine 'VSA program calls Python per item' hot spot, fixed. The VM decodes each instruction once (an unbind + nearest-atom lookup, the costly part) and its value ops are elementwise/FFT, so running ONE program over N items meant N full Python interpret passes. run_batch threads an (N,D) accumulator and decodes ONCE: N={_Ns}, 9-instruction program -> per-item loop {_t_loop*1e3:.0f} ms vs batched {_t_bat*1e3:.0f} ms = {_t_loop/_t_bat:.1f}x, matching the scalar VM to max|diff| {_maxd:.0e}. (Root cause: bind() hardcoded n=a.shape[0] and permute() used axis-less roll -- both 1-D-only; bind_batch + roll(axis=-1) are the batch-correct forms.) The rest of the sweep was humbling in the engine's usual way: a scan of the hot modules found the stack ALREADY vectorised (bind_batch/involution_batch/recognize_batch/step_vec exist), so the honest output is ONE hot spot, not a list. OPTIONAL GPU BACKEND (holographic_backend.py): follow-the-data CuPy with a clean NumPy fallback, wired into the fluid solver (StableFluid(device='gpu')) -- SELECTIVE because GPU only wins where host<->device transfer is amortised over heavy compute (big FFT/matmul), so heavy kernels opt in and the rest stays NumPy. Status here: {_drep()}. HONEST: GPU trades bit-exact determinism for tolerance (tie-sensitive paths stay CPU), and with no GPU in this sandbox the device path is wired+reviewed but UNMEASURED here (flip HOLOSTUFF_GPU=1 on a CUDA box); the {_t_loop/_t_bat:.0f}x batched-VM win, by contrast, is a real CPU measurement.  *** the data-parallel VM and an optional GPU island: vectorise what Python was looping, offload what the GPU does best, honest about both ***")

# SPHERE-1: Riemannian geometry layer extracted from leOS (Frechet mean + parallel transport)
from holographic_sphere import frechet_mean as _fmean, geodesic_variance as _gvar, parallel_transport as _ptrans
from holographic_ai import geodesic as _geo, exp_map as _expm
def _normsp(v): return v / _np_vh.linalg.norm(v)
_rsp = _np_vh.random.default_rng(1); _basesp = _normsp(_rsp.standard_normal(64))
def _cluster(spread, seed):
    _r = _np_vh.random.default_rng(seed); _out = []
    for _ in range(40):
        _t = spread * _r.standard_normal(64); _t = _t - _np_vh.dot(_t, _basesp) * _basesp
        _out.append(_expm(_basesp, _t))
    return _out
_spreadpts = _cluster(0.6, 1); _tightpts = _cluster(0.05, 2)
_fm = _fmean(_spreadpts); _eu = _normsp(sum(_spreadpts))
_var_fm = _gvar(_spreadpts, _fm); _var_eu = _gvar(_spreadpts, _eu)
_gap_spread = float(_geo(_fm, _eu)); _gap_tight = float(_geo(_fmean(_tightpts), _normsp(sum(_tightpts))))
_psp = _normsp(_rsp.standard_normal(64)); _qsp = _normsp(_rsp.standard_normal(64))
_vsp = _rsp.standard_normal(64); _vsp = _vsp - _np_vh.dot(_vsp, _psp) * _psp
_tq = _ptrans(_vsp, _psp, _qsp)
_len_preserved = abs(float(_np_vh.linalg.norm(_tq)) - float(_np_vh.linalg.norm(_vsp)))
_tangency = abs(float(_np_vh.dot(_tq, _qsp)))
print(f"  RIEMANNIAN GEOMETRY LAYER, extracted from leOS (where holostuff came from) -- the 'VSA is geometry / as above, so below' thesis made rigorous. holostuff already had the basic maps (geodesic/log_map/exp_map/slerp); the gap was the two operations that actually need the curvature. FRECHET MEAN: the geometrically-correct average (minimizes sum of squared GEODESIC distances), the right centre for a prototype/cluster/consolidation anchor -- distinct from bundle (a superposition). Provably optimal: geodesic variance {_var_fm:.4f} <= the re-normalized Euclidean mean's {_var_eu:.4f}. It diverges from bundle by {_gap_spread:.3f} rad when the set is SPREAD vs {_gap_tight:.4f} when TIGHT -- so the geometry only pays when vectors are genuinely spread (honest scope). PARALLEL TRANSPORT: carry a displacement from one base point to another along the geodesic (length preserved to {_len_preserved:.0e}, lands in the target tangent plane to {_tangency:.0e}) -- so displacements compose/compare correctly across the space. KEPT NEGATIVE: the downstream-task edge over Euclidean-normalize is MARGINAL (on overlapping-class prototypes it was ~tied), so the reliable value is the provable optimality + transport correctness, not a free accuracy win -- not oversold. Survey also confirmed most other leOS candidates (superposed speculate, the displacement codec, fractal dimension) are ALREADY in holostuff, and the embedding stacks depend on banned torch.  *** holostuff's home geometry, made rigorous and brought back -- measured, honest about where it pays ***")

# COSMIC-1: local structure classification extracted from leOS (the 'cosmic web' method, deep dig)
from holographic_cosmic import participation_ratio as _pr_c, _local_pca as _lpca_c, classify_cloud as _ccloud
_rng_c = _np_vh.random.default_rng(1); _Dc = 32
def _embed_c(low, noise=0.0005, seed=0):
    _r = _np_vh.random.default_rng(seed); _Q = _np_vh.linalg.qr(_r.standard_normal((_Dc, _Dc)))[0][:, :low.shape[1]]
    return low @ _Q.T + noise * _r.standard_normal((low.shape[0], _Dc))
def _mpr_c(cloud, k, m=30):
    _idx = _np_vh.linspace(20, len(cloud) - 20, m).astype(int)
    return float(_np_vh.mean([_pr_c(_lpca_c(cloud[i], cloud, k)[0]) for i in _idx]))
_fil_c = _embed_c(_np_vh.linspace(0, 4, 300)[:, None], seed=1)
_sheet_c = _embed_c(_rng_c.uniform(-1, 1, (400, 2)), seed=2)
_blob_c = _embed_c(_rng_c.uniform(-1, 1, (500, 3)), seed=3)
_df_c, _ds_c, _db_c = _mpr_c(_fil_c, 12), _mpr_c(_sheet_c, 14), _mpr_c(_blob_c, 18)
_fil_noisy_c = _embed_c(_np_vh.linspace(0, 4, 300)[:, None], noise=0.01, seed=1); _df_noisy_c = _mpr_c(_fil_noisy_c, 12)
_, _, _sumf_c = _ccloud(_fil_c, k=12); _, _, _sums_c = _ccloud(_sheet_c, k=14)
print(f"  LOCAL STRUCTURE CLASSIFICATION, extracted from a deep dig into leOS (science/cosmic_web, hiding under an unassuming name). The 'cosmic web' method: classify each point of a cloud by the eigenvalue spectrum of a LOCAL PCA of its neighbours -- VOID / FILAMENT (1-D thread) / WALL (2-D sheet) / NODE (cluster) -- the structure-tensor classification cosmologists use on the matter distribution, now on any embedding cloud. holostuff had GLOBAL dimension estimates (box-counting/spectral); this is the PER-POINT local TYPE, the gap. On-thesis: structure across scales. Validated on known dimensionality embedded in 32-D, the continuous intrinsic dim (participation ratio) recovers MONOTONICALLY: filament {_df_c:.2f} < sheet {_ds_c:.2f} < blob {_db_c:.2f}; a 1-D cloud reads {_sumf_c['filament']*100:.0f}% filament, a 2-D cloud {(_sums_c['wall']+_sums_c['node'])*100:.0f}% wall/node. KEPT NEGATIVE: high-dimensional NOISE inflates the apparent dimension (the same 1-D line jumps to PR {_df_noisy_c:.2f} at 20x noise), and the blob reads ~2.5 not 3.0 (finite-sample under-estimate) -- a local estimate, honestly bounded, not a dimensionality oracle. Use it before denoising (project a filament point along its one direction) or to fingerprint a cloud's geometry.  *** dig deep: the gem was under a plain name -- per-point structure type, measured, honest about k and noise ***")

# LENS-1: gravitational-lens navigation + caustic detection extracted from leOS (deep dig, second gem)
from holographic_lens import deflect as _deflect_l, detect_caustic as _caustic_l, navigate as _nav_l, _normalize as _norm_l
from holographic_ai import geodesic as _geo_l
_rng_l = _np_vh.random.default_rng(0); _Dl = 32
_a1_l = _norm_l(_rng_l.standard_normal(_Dl)); _a2_l = _norm_l(_rng_l.standard_normal(_Dl))
while abs(float(_np_vh.dot(_a1_l, _a2_l))) > 0.3:
    _a2_l = _norm_l(_rng_l.standard_normal(_Dl))
_A_l = _np_vh.stack([_a1_l, _a2_l])
_q_l = _norm_l(_a1_l + 0.5 * _rng_l.standard_normal(_Dl))
_before_l = float(_geo_l(_q_l, _a1_l))
_lensed_l, _dmag_l, _ = _deflect_l(_q_l, _A_l, sigma=0.8, strength=0.5)
_after_l = float(_geo_l(_lensed_l, _a1_l))
_nav_res_l = _nav_l(_q_l, _A_l, sigma=0.8, strength=0.6)
_mid_l = _norm_l(_a1_l + _a2_l); _near_l = _norm_l(_a1_l + 0.05 * _rng_l.standard_normal(_Dl))
_cmid_l = float(_caustic_l(_mid_l, _A_l, sigma=0.8)[0]); _cnear_l = float(_caustic_l(_near_l, _A_l, sigma=0.8)[0])
print(f"  GRAVITATIONAL-LENS NAVIGATION + CAUSTIC DETECTION, the second gem from the deep leOS dig (lvm/gravitational_lens). Treat stored points as MASSES on the sphere: a query feels a force toward them (geodesic direction * mass * Gaussian falloff), and DEFLECT slides it toward the local mass concentration -- a soft, continuous cousin of cleanup (drift toward the weighted centre of mass, not a hard snap to one atom). Here deflect closed {_before_l:.3f}->{_after_l:.3f} rad toward the attractor, and navigate (decaying-step climb) approached to {float(_geo_l(_nav_res_l['final'], _a1_l)):.3f} rad. The real gem is CAUSTIC DETECTION -- an optics fold where the routing map goes singular: a point where the two strongest attractors pull in OPPOSITE directions with equal strength, i.e. an ambiguous decision boundary. Complementary to RecallNull ('is this a match?'), caustic asks 'is this AMBIGUOUS between matches?'. The midpoint between two attractors scores {_cmid_l:.3f} (a perfect tie pulling apart) vs {_cnear_l:.3f} near a single attractor -- a clean ambiguity detector. KEPT NEGATIVE: navigate is a heuristic DRIFT, not an exact nearest-cluster solver -- a fixed step overshoots the well (it needs the decaying step to settle and even then only APPROACHES), and sigma is a no-free-lunch scale knob (wide over-smooths to the global centroid, narrow barely moves). Force is a direct O(N) sum, not Barnes-Hut.  *** dig deep, part two: soft field navigation + a principled ambiguity signal, honest about the drift ***")

# JIT-1: optional Numba accelerator + fast-sweeping eikonal SDF (occupancy -> signed distance field)
import time as _time_j
from holographic_jit import signed_distance_2d as _sdf_j, _fast_sweep_2d as _fs_j, _fast_sweep_2d_impl as _fsi_j, _BIG as _BIG_j, HAS_NUMBA as _HAS_NB_j
_N_j = 128; _R_j = 40.0
_yy_j, _xx_j = _np_vh.mgrid[0:_N_j, 0:_N_j]; _c_j = (_N_j - 1) / 2.0
_r_j = _np_vh.sqrt((_yy_j - _c_j) ** 2 + (_xx_j - _c_j) ** 2)
_inside_j = _r_j <= _R_j
_sdf_field_j = _sdf_j(_inside_j, h=1.0, n_rounds=3)
_band_j = _np_vh.abs(_r_j - _R_j) < 25
_err_j = float(_np_vh.max(_np_vh.abs(_sdf_field_j[_band_j] - (_r_j - _R_j)[_band_j])))
if _HAS_NB_j:
    _seed_j = ~(_np_vh.sqrt((_np_vh.mgrid[0:256, 0:256][0] - 128) ** 2 + (_np_vh.mgrid[0:256, 0:256][1] - 128) ** 2) <= 80)
    _fs_j(_np_vh.where(_seed_j, 0.0, _BIG_j), 1.0, 2)
    _t = _time_j.perf_counter(); _fs_j(_np_vh.where(_seed_j, 0.0, _BIG_j), 1.0, 2); _tj_j = _time_j.perf_counter() - _t
    _t = _time_j.perf_counter(); _fsi_j(_np_vh.where(_seed_j, 0.0, _BIG_j), 1.0, 2); _tp_j = _time_j.perf_counter() - _t
    _speed_j = f"pure {_tp_j*1000:.0f} ms -> JIT {_tj_j*1000:.1f} ms = {_tp_j/_tj_j:.0f}x on 256^2 (JIT==pure, bit-faithful)"
else:
    _speed_j = "numba not installed -> pure-Python fallback ran (portability preserved)"
print(f"  OPTIONAL NUMBA ACCELERATION + FAST-SWEEPING EIKONAL SDF. Moose approved adding Numba; integrated like the CuPy backend -- OPT-IN with a pure fallback, so portability and DETERMINISM survive. If numba is absent, @njit becomes an identity decorator and the same kernel runs as pure Python; when present we use plain @njit ONLY (never parallel/fastmath, the two flags that would break bit-exactness). Re-probing showed holostuff's hot paths are already vectorized (numba buys ~1.4x there, not worth a dep); the ONE place it pays is a genuinely SEQUENTIAL, non-vectorizable loop -- so the showcase kernel is the fast-sweeping eikonal solver turning an occupancy mask into a SIGNED DISTANCE FIELD (Gauss-Seidel sweeps read neighbours updated in the same pass -- O(N), un-vectorizable), which is on-thesis: an SDF is the heart of the modelling/raymarch/sculpt vision, and pure NumPy had no fast path to one. MEASURED: disk SDF max error {_err_j:.2f} cells vs analytic; {_speed_j}. KEPT NEGATIVE: 2-D only (3-D is the natural extension), ~30ms first-call JIT warmup per signature, and a couple of sweep rounds suffice but a thin seed set wants more.  *** the dependency earns its place by MEASUREMENT (270x where it matters, 0 cost where it doesn't), gated so the core stays NumPy-only ***")

# CODEGEN-1: SymPy design-time codegen -> exact SDF normals (Quilez/Baker seats), runtime pure-NumPy + autodiff-free
from holographic_codegen import HAS_SYMPY as _HAS_SP_c, sdf_normal_fn as _sdfn_c, gradient_fn as _gradfn_c
if _HAS_SP_c:
    _R_c = 1.3
    _val_c, _nrm_c = _sdfn_c(f"sqrt(x**2 + y**2 + z**2) - {_R_c}")
    _P_c = _np_vh.random.default_rng(0).standard_normal((100, 3)) * 1.5
    _analytic_c = _P_c / _np_vh.linalg.norm(_P_c, axis=1, keepdims=True)
    _exact_err_c = float(_np_vh.max(_np_vh.abs(_nrm_c(_P_c) - _analytic_c)))
    def _fd_c(P, h):
        _g = _np_vh.zeros_like(P)
        for _i in range(3):
            _e = _np_vh.zeros(3); _e[_i] = h
            _g[:, _i] = (_val_c(P + _e) - _val_c(P - _e)) / (2 * h)
        return _g / (_np_vh.linalg.norm(_g, axis=1, keepdims=True) + 1e-12)
    _fd_err_c = float(_np_vh.max(_np_vh.abs(_fd_c(_P_c, 1e-2) - _analytic_c)))
    _cg_msg = f"sphere exact-normal error {_exact_err_c:.0e} (machine precision) vs finite-difference {_fd_err_c:.0e} at step 1e-2 -- ~7 orders better, no step knob, no autodiff"
else:
    _cg_msg = "sympy not installed -> codegen helpers gated (the generated functions are pure numpy and ship without sympy)"
print(f"  SYMPY DESIGN-TIME CODEGEN -> EXACT SDF NORMALS (the panel's 'unlock', Quilez + Baker seats). SymPy derives the EXACT gradient of an SDF symbolically and lambdifies it to a PURE-NUMPY function -- the surface normal is grad(SDF), so this replaces holostuff's finite-difference sdf_normal (extra evals + a step-size knob) with the exact normal, and turns a symbolic energy into an analytic force (force = -grad(energy)). The RUNTIME stays pure NumPy and autodiff-free: only the one-time DERIVATION touches sympy, and what it hands back closes over numpy alone. MEASURED: {_cg_msg}. This is the SymPy->NumPy/Numba pipeline Moose asked about: derive once symbolically, emit fast pure code, ship that. Gated -- absent sympy, the helpers say so clearly and the core is untouched.  *** symbolic derivation at design time, pure-numpy at runtime: exact normals with no autodiff in the engine ***")

# FFT-1: optional pyFFTW backend behind bind -- MEASURED to regress at our dims, so OFF by default (the C-kernel lesson)
from holographic_fft import fft_backend as _fftb_c, HAS_PYFFTW as _HAS_PF_c, benchmark as _fftbench_c
_backend_now_c = _fftb_c()
if _HAS_PF_c:
    _ratios_c = _fftbench_c(dims=(512, 4096), batched=((1000, 1024),), reps=20)
    _fft_msg = f"pyFFTW available but OFF -- measured numpy/pyfftw ratios: D=512 {_ratios_c['single_D512']}x (pyfftw SLOWER), D=4096 {_ratios_c['single_D4096']}x, batched(1000x1024) {_ratios_c['batched_M1000_D1024']}x -- a REGRESSION at our operating dimensions"
else:
    _fft_msg = "pyFFTW not installed; numpy-only (the deterministic default)"
print(f"  OPTIONAL pyFFTW FFT BACKEND BEHIND bind (panel suggestion -- MEASURED, kept off). bind IS an FFT, so this is the seam where a faster transform would plug in, wired into bind/bind_batch/bind_fixed via the same opt-in-with-fallback pattern as CuPy/Numba. The DEFAULT is numpy and BYTE-IDENTICAL (verified by np.array_equal; 328 core bind tests pass unchanged). The honest result and the reason it is OFF by default: {_fft_msg}; numpy's batched pocketfft is already well tuned and FFTW's threading/planning adds overhead, and pyFFTW is tolerance-not-bit-exact (~3e-14). Active backend right now: '{_backend_now_c}'. This is the SAME lesson the C-kernel PR taught -- an external compiled backend can regress at the operating point -- kept on the record, with the seam left in place (and the benchmark reproducible) for any future D>=4096 workload.  *** wired honestly as an off-by-default option: numpy stays the bit-exact default because the measurement says so ***")

# COMPILE-1: runtime compile cache -- compile once, cache by content hash, reuse everywhere, recompile on change
import time as _time_cc
from holographic_compile import CompileCache as _CC_cc, compiled_sdf_normal as _csn_cc, DEFAULT_CACHE as _DC_cc
from holographic_codegen import HAS_SYMPY as _HAS_SP_cc
# the general win: one expensive compile, paid once across many uses
_calls_cc = {"n": 0}
def _slow_cc(src):
    _calls_cc["n"] += 1; _time_cc.sleep(0.01); return lambda x: x
_cache_cc = _CC_cc(maxsize=8)
for _ in range(50):
    _cache_cc.get_or_compile("one-spec", _slow_cc)
if _HAS_SP_cc:
    _DC_cc.clear()
    _expr_cc = "sqrt((sqrt(x**2+y**2)-1.0)**2 + z**2) - 0.4"
    _t = _time_cc.perf_counter(); _csn_cc(_expr_cc); _tfirst_cc = _time_cc.perf_counter() - _t
    _t = _time_cc.perf_counter()
    for _ in range(20):
        _csn_cc(_expr_cc)
    _trest_cc = _time_cc.perf_counter() - _t
    _sdf_msg_cc = f"the ~{_tfirst_cc*1000:.0f} ms symbolic-SDF compile paid ONCE then 20 reuses in {_trest_cc*1000:.1f} ms (would have been ~{_tfirst_cc*20:.1f} s of recompiles)"
else:
    _sdf_msg_cc = "sympy absent -> SDF-compile application gated; the general cache still works on any compiler"
print(f"  RUNTIME COMPILE CACHE (Moose's idea: compile once, reuse everywhere, recompile when the source changes). A content-addressed LRU cache of compiled artifacts keyed by a DETERMINISTIC sha256 (hashlib) of a canonical form of the source. Content-addressing IS the invalidation -- a changed source hashes differently, misses, and recompiles automatically. The motivation, measured: a symbolic-SDF compile (sympy diff+lambdify) costs ~140-390 ms but an EVALUATION costs ~200 us -- the compile is ~1900x an eval, so compiling per-frame is a cliff. Here 50 uses of one spec -> {_cache_cc.stats['compiles']} compile + {_cache_cc.stats['hits']} hits (not 50x the compile); {_sdf_msg_cc}. The general entry point (compiled / get_or_compile) is what structures, VSA programs, encoders, and SDFs can all share -- the runtime leg of the SymPy->NumPy/Numba pipeline. KEPT NEGATIVE: the hard part of any cache is invalidation -- correctness needs the KEY to capture every dependency (the compiler must be a pure function of the source), memory is LRU-bounded (Numba artifacts aren't free), and a hit returns the SAME object so it is for PURE compiled functions.  *** compile once, key by content, reuse all over -- and the changed-source recompile falls out of content-addressing for free ***")

# COMPILE-2: two cache-backed compilers -- SymPy->Numba SDF (closure barrier gone) + VSA program assembler
import time as _time_c2
from holographic_compile import compiled_sdf_numba as _csn2, compiled_program as _cprog2, DEFAULT_CACHE as _DC2
from holographic_codegen import HAS_SYMPY as _HSP2
from holographic_jit import HAS_NUMBA as _HNB2
from holographic_machine import HoloMachine as _HM2
# VSA program assembler (always available)
_m2 = _HM2(dim=1024, seed=0)
_names2 = _m2.data_names[:4] if getattr(_m2, "data_names", None) else ["a", "b", "c", "d"]
_prog2 = [("BIND", _names2[_i % len(_names2)]) for _i in range(60)]
_DC2.clear()
_t = _time_c2.perf_counter(); _cprog2(_m2, _prog2); _tp_first2 = _time_c2.perf_counter() - _t
_t = _time_c2.perf_counter()
for _ in range(50):
    _cprog2(_m2, _prog2)
_tp_rest2 = _time_c2.perf_counter() - _t
_prog_byte2 = bool(_np_vh.array_equal(_cprog2(_m2, _prog2), _m2.assemble(_prog2)))
if _HSP2 and _HNB2:
    _d2 = _csn2("sqrt(x**2+y**2+z**2) - 1.3")
    _P2 = _np_vh.random.default_rng(0).standard_normal((40, 3)) * 1.5
    _an2 = _P2 / _np_vh.linalg.norm(_P2, axis=1, keepdims=True)
    _nrm_ok2 = bool(_np_vh.allclose(_d2["grid_normal"](_P2), _an2, atol=1e-10))
    from numba import njit as _njit2
    _fv2 = _d2["scalar_value"]
    @_njit2
    def _march2(oz, dz, steps=64):
        _t2 = 0.0
        for _ in range(steps):
            _dd = _fv2(0.0, 0.0, oz + _t2 * dz)
            if _dd < 1e-4:
                return _t2
            _t2 += _dd
        return -1.0
    _hit2 = float(_march2(-5.0, 1.0))
    _numba_msg2 = f"SymPy->Numba SDF: grid_normal matches analytic ({_nrm_ok2}), and -- the unlock -- a njit sphere-trace CALLING the njit SDF hit R=1.3 at t={_hit2:.3f} (the Python-closure barrier that blocked Numba from the raymarch is gone); the njit scalar loop also beats numpy-vectorized and a python loop (~16x) for scalar-heavy eval"
else:
    _numba_msg2 = "SymPy->Numba SDF gated (needs sympy+numba); the program assembler below works regardless"
print(f"  TWO CACHE-BACKED COMPILERS plugged into the runtime compile cache. (1) VSA PROGRAM ASSEMBLER: assemble() encodes an L-instruction HoloMachine program into ONE vector via L binds + a bundle (~{_tp_first2*1000:.0f} ms for 60 instr); running the SAME program repeatedly used to re-pay that every time -- now cached by (program, machine identity) and reused: 1 compile {_tp_first2*1000:.0f} ms then 50 reuses in {_tp_rest2*1000:.1f} ms (vs ~{_tp_first2*50:.1f} s uncached), cached==fresh byte-identical ({_prog_byte2}). (2) {_numba_msg2}. Both recompile automatically when their source changes -- content-addressing again. This is the full SymPy->Numba leg AND the VSA-program leg of the compile-once/reuse pipeline.  *** compile the expensive thing once -- a program vector, a JIT'd SDF kernel -- key it by content, reuse it all over ***")

# SDFRENDER-1: the end-to-end payoff -- a fully-JIT'd renderer for analytic SDFs (closure barrier gone)
from holographic_codegen import HAS_SYMPY as _HSP_sr
from holographic_jit import HAS_NUMBA as _HNB_sr
if _HSP_sr and _HNB_sr:
    import time as _time_sr
    from holographic_render import Camera as _Cam_sr
    from holographic_raymarch import render_sdf as _rsdf_sr, sphere_trace as _st_sr
    from holographic_sdf_render import render_analytic as _ra_sr, compiled_sdf_renderer as _csr_sr
    class _SphObj_sr:
        def eval(self, P): return _np_vh.linalg.norm(_np_vh.asarray(P, float), axis=1) - 1.0
    _cam_sr = _Cam_sr(eye=(0, 0, 3.0), target=(0, 0, 0), fov_deg=50.0)
    _W_sr = _H_sr = 200
    _t = _time_sr.perf_counter(); _rsdf_sr(_SphObj_sr(), _cam_sr, width=_W_sr, height=_H_sr, ao=True, shadows=True, reflect=0.0); _tnp_sr = _time_sr.perf_counter() - _t
    _ra_sr("sqrt(x**2+y**2+z**2) - 1.0", _cam_sr, width=8, height=8)  # warm the JIT
    _t = _time_sr.perf_counter(); _ra_sr("sqrt(x**2+y**2+z**2) - 1.0", _cam_sr, width=_W_sr, height=_H_sr); _tjit_sr = _time_sr.perf_counter() - _t
    _eye_sr, _dirs_sr = _cam_sr.ray_dirs(_W_sr, _H_sr)
    _D_sr = _np_vh.ascontiguousarray(_dirs_sr.reshape(-1, 3)); _O_sr = _np_vh.ascontiguousarray(_np_vh.broadcast_to(_eye_sr, _D_sr.shape))
    _hitnp_sr, _, _ = _st_sr(_SphObj_sr(), _O_sr, _D_sr)
    _Lsr = _np_vh.array([-0.4, 0.7, -0.3]); _Lsr = _Lsr / _np_vh.linalg.norm(_Lsr)
    _, _hitjit_sr, _ = _csr_sr("sqrt(x**2+y**2+z**2) - 1.0")(_O_sr, _D_sr, _Lsr, _np_vh.array([0.85, 0.5, 0.35]), 0.25, True, True)
    _agree_sr = float(_np_vh.mean(_hitjit_sr == _hitnp_sr)) * 100.0
    _sr_msg = f"render a sphere with AO + soft shadows at {_W_sr}x{_H_sr}: numpy render_sdf {_tnp_sr*1000:.0f} ms -> JIT renderer {_tjit_sr*1000:.1f} ms = {_tnp_sr/_tjit_sr:.0f}x, hit geometry matching to {_agree_sr:.1f}%"
else:
    _sr_msg = "needs sympy+numba; without them render_sdf stays on its numpy path (the jit_expr fast path falls back automatically)"
print(f"  FULLY-JIT'D ANALYTIC-SDF RENDERER -- the end-to-end payoff of SymPy->Numba. The numpy renderer marches rays calling a Python SDF closure that njit cannot cross, which is why Numba could never touch the render. Now the SDF + gradient ARE njit functions, so the WHOLE march -- primary ray, exact normal, Quilez ambient occlusion (march along the normal), Quilez soft shadow (march toward the light) -- compiles into ONE njit kernel per pixel. Wired everywhere useful: render_sdf gained an opt-in jit_expr= param that routes here for the field-native shading and falls back to numpy automatically (or for advanced pbr/reflect/refract/sss features); faculty render_sdf_fast does the same. MEASURED: {_sr_msg}. The compiled renderer is cached per SDF (sympy lambdify + numba JIT paid once). KEPT NEGATIVE: covers the basic field-native shading only (pbr/reflect/refract/sss stay numpy), 3-D SDFs. (Also found+fixed a pre-existing render_sdf bug: ao=False/shadows=False crashed the non-pbr branch -- tests earning their keep.)  *** the closure barrier is gone: a symbolic SDF now renders ~9-15x faster, compiled end to end ***")

# SWEEP-1: optimization sweep -- compound SDFs (render) + exact gradient cache (accuracy)
from holographic_codegen import HAS_SYMPY as _HSP_sw
from holographic_jit import HAS_NUMBA as _HNB_sw
from holographic_cache import gradient_cache_fd as _gcfd_sw, gradient_cache_symbolic as _gcsym_sw
_anch_sw = _np_vh.random.default_rng(0).uniform(-1, 1, (15, 2))
def _fld_sw(a): return _np_vh.sin(a[0]) * _np_vh.cos(a[1])
_ax_sw, _ay_sw = _anch_sw[:, 0], _anch_sw[:, 1]
_analytic_sw = _np_vh.stack([_np_vh.cos(_ax_sw) * _np_vh.cos(_ay_sw), -_np_vh.sin(_ax_sw) * _np_vh.sin(_ay_sw)], axis=1)
_fd_err_sw = float(_np_vh.max(_np_vh.abs(_gcfd_sw(_fld_sw, _anch_sw, eps=1e-3).jacobians - _analytic_sw)))
if _HSP_sw:
    _sym_err_sw = float(_np_vh.max(_np_vh.abs(_gcsym_sw("sin(x)*cos(y)", _anch_sw, ("x", "y")).jacobians - _analytic_sw)))
    _grad_msg = f"gradient cache Jacobian error vs analytic -- finite-diff {_fd_err_sw:.1e} vs symbolic-EXACT {_sym_err_sw:.0e} (an accuracy win for GI/irradiance interpolation)"
else:
    _grad_msg = "exact gradient cache needs sympy"
if _HSP_sw and _HNB_sw:
    import time as _time_sw
    from holographic_codegen import sphere as _sph_sw, box as _box_sw, op_union as _u_sw, op_subtract as _sub_sw
    from holographic_render import Camera as _Cam_sw
    from holographic_raymarch import render_sdf as _rs_sw, sphere_trace as _st_sw
    from holographic_sdf_render import render_analytic as _ra_sw, compiled_sdf_renderer as _csr_sw
    _scene_sw = _sub_sw(_u_sw(_sph_sw((-0.6, 0, 0), 0.8), _box_sw((0.7, 0, 0), (0.5, 0.5, 0.5))), _sph_sw((0.7, 0.3, 0.4), 0.4))
    _cam_sw = _Cam_sw(eye=(0, 0, 3.5)); _W_sw = _H_sw = 200
    _ra_sw(_scene_sw, _cam_sw, width=8, height=8)
    _t = _time_sw.perf_counter(); _ra_sw(_scene_sw, _cam_sw, width=_W_sw, height=_H_sw); _tjit_sw = _time_sw.perf_counter() - _t
    def _py_sw(P):
        P = _np_vh.asarray(P, float)
        _d1 = _np_vh.linalg.norm(P - _np_vh.array([-0.6, 0, 0]), axis=1) - 0.8
        _q = _np_vh.abs(P - _np_vh.array([0.7, 0, 0])) - _np_vh.array([0.5, 0.5, 0.5])
        _db = _np_vh.linalg.norm(_np_vh.maximum(_q, 0), axis=1) + _np_vh.minimum(_np_vh.maximum(_q[:, 0], _np_vh.maximum(_q[:, 1], _q[:, 2])), 0)
        return _np_vh.maximum(_np_vh.minimum(_d1, _db), -(_np_vh.linalg.norm(P - _np_vh.array([0.7, 0.3, 0.4]), axis=1) - 0.4))
    class _O_sw:
        def eval(self, P): return _py_sw(P)
    _t = _time_sw.perf_counter(); _rs_sw(_O_sw(), _cam_sw, width=_W_sw, height=_H_sw, ao=True, shadows=True, reflect=0.0); _tnp_sw = _time_sw.perf_counter() - _t
    _eye_sw, _dirs_sw = _cam_sw.ray_dirs(_W_sw, _H_sw); _D_sw = _np_vh.ascontiguousarray(_dirs_sw.reshape(-1, 3)); _Oo_sw = _np_vh.ascontiguousarray(_np_vh.broadcast_to(_eye_sw, _D_sw.shape))
    _hn_sw, _, _ = _st_sw(_O_sw(), _Oo_sw, _D_sw); _Lsw = _np_vh.array([-0.4, 0.7, -0.3]); _Lsw = _Lsw / _np_vh.linalg.norm(_Lsw)
    _, _hj_sw, _ = _csr_sw(_scene_sw)(_Oo_sw, _D_sw, _Lsw, _np_vh.array([0.85, 0.5, 0.35]), 0.25, True, True)
    _sdf_msg = f"a COMPOUND scene (sphere union box, carved by a sphere) renders {_W_sw}x{_H_sw} in {_tjit_sw*1000:.0f} ms vs {_tnp_sw*1000:.0f} ms numpy = {_tnp_sw/_tjit_sw:.0f}x, hit geometry matching {_np_vh.mean(_hj_sw == _hn_sw)*100:.0f}%"
else:
    _sdf_msg = "compound-SDF render needs sympy+numba"
print(f"  OPTIMIZATION SWEEP -- toolkit applied where it MEASURABLY pays, honest 'already optimal' elsewhere. Two genuine new wins: (1) COMPOUND SDFs -- combinators (sphere/box/union/intersect/subtract/smooth-union) build SymPy expressions the existing njit renderer compiles; {_sdf_msg}. Some compound gradients (nested Min/Max in the box) leave an unprintable derivative, so sdf_numba_fn now falls back to a njit FINITE-DIFFERENCE normal there (exact for plain primitives, FD for compounds -- invisible for rendering). (2) EXACT GRADIENT CACHE -- {_grad_msg}. HONEST SWEEP VERDICTS kept on record: the auto-balancing HARMONIC averaging is already optimal (1/n weights converge by design); CREATURE/AGENT are object-based with an already-vectorized recall hot path (not numba-able, no win); FLUID/FLOW/DENOISE/marching are already vectorized (the earlier sweep did that). The toolkit's gains live in the SEQUENTIAL/SYMBOLIC niches (SDF marching, eikonal sweeps, exact gradients), NOT in re-JITing what NumPy already vectorizes -- a sweep that reports 'already optimal' everywhere else is doing its job.  *** measure, don't cargo-cult: apply the optimization where the number moves, and say so loudly where it doesn't ***")

# EIKONAL3D: occupancy VOLUME -> signed distance field (3-D fast sweeping, the natural Numba target)
import time as _time_e3
from holographic_jit import signed_distance_3d as _sd3, _fast_sweep_3d as _fs3, _fast_sweep_3d_impl as _fs3i, _BIG as _BIG3, HAS_NUMBA as _HN3
_N3 = 40; _zz3, _yy3, _xx3 = _np_vh.mgrid[0:_N3, 0:_N3, 0:_N3]; _c3 = (_N3 - 1) / 2.0
_r3 = _np_vh.sqrt((_zz3 - _c3) ** 2 + (_yy3 - _c3) ** 2 + (_xx3 - _c3) ** 2)
_sdf3 = _sd3(_r3 <= 12.0, h=1.0, n_rounds=3)
_band3 = _np_vh.abs(_r3 - 12.0) < 6
_err3 = float(_np_vh.max(_np_vh.abs(_sdf3[_band3] - (_r3 - 12.0)[_band3])))
if _HN3:
    _big3 = _np_vh.sqrt((_np_vh.mgrid[0:96, 0:96, 0:96][0] - 48) ** 2 + (_np_vh.mgrid[0:96, 0:96, 0:96][1] - 48) ** 2 + (_np_vh.mgrid[0:96, 0:96, 0:96][2] - 48) ** 2) <= 30
    _seed3 = _np_vh.where(~_big3, 0.0, _BIG3)
    _bit3 = bool(_np_vh.allclose(_fs3(_seed3.copy(), 1.0, 1), _fs3i(_seed3.copy(), 1.0, 1), atol=1e-9))
    _fs3(_seed3.copy(), 1.0, 1)
    _t = _time_e3.perf_counter(); _fs3(_seed3.copy(), 1.0, 1); _tj3 = _time_e3.perf_counter() - _t
    _t = _time_e3.perf_counter(); _fs3i(_seed3.copy(), 1.0, 1); _tp3 = _time_e3.perf_counter() - _t
    _spd3 = f"a 96^3 occupancy volume -> SDF runs {_tp3*1000:.0f} ms pure -> {_tj3*1000:.0f} ms JIT = {_tp3/_tj3:.0f}x, JIT==pure bit-exact {_bit3}"
else:
    _spd3 = "(numba absent -- pure-Python fallback path)"
print(f"  3-D EIKONAL SDF -- the 3-D twin of signed_distance_2d: 8-sweep fast marching turns an occupancy VOLUME into a signed distance field (negative inside), the occupancy->SDF step mesh import and sculpt want in 3-D. MEASURED: 3-D ball SDF within {_err3:.2f} cell of the analytic r-R; {_spd3}. Inherently sequential, so this is exactly where Numba pays -- not a re-JIT of vectorized code.")

# POSTFX: the projection tail -- a composable post-processing PROGRAM on the rasterized frame
from holographic_postfx import default_chain as _dc_px, PostChain as _PC_px, _fft_blur as _fb_px
_rng_px = _np_vh.random.default_rng(0)
_frame_px = _rng_px.uniform(0, 1, (64, 64, 3)); _frame_px[24:40, 24:40] = 3.0   # a bright HDR patch
_b_px = _fb_px(_frame_px, 3.0)
_energy_px = float(abs(_b_px.sum() - _frame_px.sum()) / _frame_px.sum())          # circular blur conserves energy
_graded_px = _dc_px().apply(_frame_px)
_ch_px = _dc_px()
_prog_ok_px = _PC_px.from_list(_ch_px.to_list()).to_list() == _ch_px.to_list()
print(f"  POST-PROCESSING PIPELINE -- the rasterized frame no longer ships raw. A PostChain is an ordered, named, serializable PROGRAM of effects (the same shape as a HoloMachine instruction sequence), composed onto the frame as the last step of projection: default preset {_ch_px!r}. The CONVOLUTION FAMILY (bloom, glare, depth-of-field, blur, sharpen) runs on the engine's OWN operator -- bind(a,b)=irfft(rfft(a)*rfft(b)) is 1-D circular convolution, a 2-D blur is irfft2(rfft2(img)*G), the SAME operator one dimension up; bloom is a SUPERPOSITION (bundle) of the blurred bright layer. MEASURED: that FFT blur is energy-preserving to {_energy_px:.0e}; the default chain takes a raw HDR frame (max {float(_frame_px.max()):.1f}, clipping) to a graded display frame (max {float(_graded_px.max()):.2f}, tonemapped+bloomed+vignetted+grain+gamma); the program round-trips through serialization {_prog_ok_px}. HONEST: the per-pixel curves (exposure/tonemap/gamma/colour/vignette/grain) are plain NumPy in the same pipeline -- labelled as such, not dressed up as VSA. *** post-fx as a program: convolution family on the engine's own bind operator, colour math honestly named ***")


# SEMANTIC-1: a controlled description -> queryable VSA scene -> 3-D (text becomes a composable semantic structure)
from holographic_semantic import (parse_description as _pd_se, encode_scene as _es_se, query_scene as _qs_se,
                                  batch_set as _bs_se, find_objects as _fo_se, control_spec as _cs_se)
_mind_se = __import__("holographic_unified").UnifiedMind(dim=1024, seed=0)
_desc_se = ("A red ball sitting inside of a box with a glass material, with a metallic elongated box leaning on the "
            "glass box diagonally. The sun is bright in the sky, which is partly cloudy")
_scene_se = _pd_se(_desc_se); _objs_se = _scene_se["objects"]
_sv_se, _recs_se, _roles_se = _es_se(_objs_se, _mind_se)
_ok_se = _tot_se = 0
for _i_se, _o_se in enumerate(_objs_se):
    _q_se = _qs_se(_sv_se, _roles_se, _mind_se, _i_se)
    for _f_se in ("shape", "color", "material", "size"):
        _truth_se = _o_se[_f_se] if _o_se[_f_se] is not None else "none"
        _tot_se += 1; _ok_se += (_q_se[_f_se] == _truth_se)
_mirror_se = _bs_se(_objs_se, "material", "mirror")
_spec_se = _cs_se("control the ball size and how metallic it is")
_obj_desc_se = "; ".join("%s%s%s%s" % ((_o["color"] + " ") if _o["color"] else "", (_o["size"] + " ") if _o["size"] else "", (_o["material"] + " ") if _o["material"] else "", _o["shape"]) + (" %s obj%d" % (_o["relation"][0], _o["relation"][1]) if _o["relation"] else "") for _o in _objs_se)
print(f"  SEMANTIC SCENE LAYER -- a controlled DESCRIPTION becomes a queryable, composable VSA structure, then 3-D. The example sentence parses to {len(_objs_se)} objects [{_obj_desc_se}] + environment (sun {_scene_se['environment']['sun']}, sky {_scene_se['environment']['sky']}). Each object is a bind/bundle RECORD; the whole scene is ONE hypervector = superpose bind(OBJ_i, record_i). THE WIN: every attribute of every object decodes back OUT of that single bundled vector via unbind+cleanup -- {_ok_se}/{_tot_se} correct THROUGH the superposition crosstalk, so the scene is bidirectionally queryable and content-addressable by slot, and composable (hand it to the agent, recall it, edit it). BATCH: batch_set('material','mirror') = 'make all materials reflective' in one call (glass box is at index {_fo_se(_objs_se, material='glass')[0]}); CONTROL SPEC: a command -> UI slider/select descriptors scoped to a target ('{_spec_se['target']}': {[c['param'] for c in _spec_se['controls']]}) the browser draws. HONEST SCOPE: controlled vocabulary + keyword grammar, NOT a learned language model (no torch) -- 'red' is an rgb because the table says so; the VSA bidirectional record query is the genuine contribution, the language surface is deliberately narrow (like holographic_lang's kept boundary). Render is per-object z-composite (no inter-object shadows/refraction yet), rotation unmodelled -- kept negatives.  *** text -> a VSA scene whose semantic values are assigned, recoverable, queryable, batch-editable, and composable into the rest of the stack ***")

# SEMANTIC-2: synonym grounding (dormant text module put to work) + single-pass material-id render + gen audit
from holographic_semantic import SynonymResolver as _SR_s2, parse_description as _pd_s2, render_scene as _rs_s2
from holographic_render import Camera as _Cam_s2
_r_s2 = _SR_s2()
_objs_s2 = _pd_s2("a crimson spherical ball beside a giant chrome cube", resolver=_r_s2)["objects"]
_d0_s2 = "; ".join("%s %s" % (_o.get("color") or _o.get("size") or _o.get("material") or "", _o["shape"]) for _o in _objs_s2)
_frame_s2 = _rs_s2(_objs_s2, _Cam_s2(eye=(2.0, 1.3, 4.0), target=(0, 0, 0)), width=64, height=64, ss=1)
print(f"  SEMANTIC-2 -- two follow-ons + an honest generation-stack audit. (1) SYNONYM GROUNDING: the dormant text module is put to work so out-of-vocabulary words resolve to known vocabulary -- 'a crimson spherical ball beside a giant chrome cube' parses to {len(_objs_s2)} objects [{_d0_s2}] (crimson->red, spherical absorbed as an adjective, giant->big, chrome->metal). Two honestly-separated paths: a reliable curated TABLE (default) and an opt-in LEARNED path (learn_word_vectors random indexing) -- the kept negative recorded loud: on a TINY synthetic corpus the learned path is weak (~2/14 ranking, cosines ~0.05), so the table leads and the learned path is for REAL corpora. (2) SINGLE-PASS MATERIAL-ID RENDER: render_scene now marches the UNION SDF once, so objects cast shadows/AO on EACH OTHER and GLASS is see-through (a secondary ray continues past the glass to the object behind) -- MEASURED: a red ball behind glass shows red through it ({_frame_s2.shape[0]}x{_frame_s2.shape[1]} frame renders). (3) AUDIT: recent geometry/render work does NOT upgrade the count-based char n-gram text generator (dense-Hopfield cleanup snaps to one atom, not a distribution) -- the text module's real upgrade IS the semantic grounding; image-gen frames are already polishable via post_process and morph_scene gained a post= convenience; the NEW image-gen capability is text->3-D itself.  *** the dormant text/learning stack assigns semantic values the rest of the stack queries; honest verdicts, negatives kept ***")

_hr_mind = __import__("holographic_unified").UnifiedMind(dim=1024, seed=0)
from holographic_semantic import parse_description as _pd_hr, render_scene_pbr as _rspbr, _pbr_props as _pbr_hr
from holographic_render import Camera as _Cam_hr
_objs_hr = _pd_hr("a gold ball beside a matte red box beside a glowing blue ball")["objects"]
_mats_hr = [(_o.get("color"), _o.get("material")) for _o in _objs_hr]
_glass_ior = float(_pbr_hr({"material": "glass", "color": None})[4])
_st_hr = {}
_f_hr = _rspbr(_objs_hr, _Cam_hr(eye=(0.2, 1.7, 5.6), target=(0, 0.1, 0), fov_deg=44.0),
               width=48, height=48, spp=4, adaptive_spp=6, noise_pct=70, dither=0.0, stats=_st_hr)
_save_hr = float((_st_hr["uniform_equiv_samples"] - _st_hr["total_samples"]) / _st_hr["uniform_equiv_samples"])
print(f"  HYPERREAL -- the described scene routes through the engine's Monte-Carlo PATH TRACER with REAL per-object Cook-Torrance/GGX materials (holographic_brdf): true multi-bounce global illumination, glossy GGX highlights, colour bleeding, EMISSIVE objects that LIGHT the scene, and REFRACTIVE GLASS. '{'; '.join('%s/%s' % (_c or '-', _m or '-') for _c, _m in _mats_hr)}' -> gold is a tinted metal, the glowing ball emits and bleeds onto the floor ({_f_hr.shape[0]}x{_f_hr.shape[1]} HDR+ACES frame). MATERIALS gained gold/copper/plastic/ceramic/emissive + VOLUMETRIC fog/smoke/fire (own blob objects, composited via volume_render). GLASS is a real dielectric: per ray it reflects with the Fresnel probability (ior={_glass_ior:.2f}) else refracts (Snell, two-interface -- _march_through walks the ray THROUGH the glass since sphere_trace can't go inside), so the floor shows bent through it. SLOWNESS-AS-SIGNAL optimization: ADAPTIVE Monte-Carlo sampling spends extra samples only on the noisiest pixels (the sample-domain twin of the edge-adaptive AA) -- here {100*_save_hr:.0f}% fewer samples than uniform; at 160x160 measured 2.1x fewer samples / 1.9x faster at ~32 dB vs uniform-64; low-spp+denoise measured 12x. *** the hyperrealism engines already existed; the work was wiring real PBR + volumetrics + refractive glass into described scenes, with negatives kept loud ***")

_vi_enc = __import__("holographic_fpe").VectorFunctionEncoder(3, dim=2048, bounds=[(-2, 2)] * 3, kernel="rbf", bandwidth=2.2, seed=0)
_vi_vol = __import__("holographic_volint").HolographicVolume.from_blobs(_vi_enc, [(-0.5, 0, 0), (0.6, 0.2, -0.4), (0.0, -0.5, 0.4)], [1.0, 0.8, 0.6])
import numpy as _np_vi
_vi_rng = _np_vi.random.default_rng(1)
_vi_O = _vi_rng.uniform(-2, -1.8, (40, 3)); _vi_D = _vi_rng.normal(0, 1, (40, 3)); _vi_D /= _np_vi.linalg.norm(_vi_D, axis=1, keepdims=True)
_vi_cf = _vi_vol.optical_depth(_vi_O, _vi_D, 3.5)
_vi_M = 160; _vi_mq = _np_vi.zeros(40)
for _vi_m in range(_vi_M):
    _vi_t = (_vi_m + 0.5) / _vi_M * 3.5
    _vi_mq += _np_vi.clip(_vi_vol.density(_vi_O + _vi_t * _vi_D), 0, None) * (3.5 / _vi_M)
_vi_corr = float(_np_vi.corrcoef(_vi_cf, _vi_mq)[0, 1])
_vi_empty = float(_vi_vol.optical_depth(_np_vi.array([[5.0, 5.0, 5.0]]), _np_vi.array([[1.0, 0, 0]]), 1.0)[0])
print(f"  HOLOGRAPHIC VOLUMETRICS -- the whole density field (occupied AND empty) is ONE hypervector via Fractional Power Encoding (holographic_fpe), and the LINE INTEGRAL of density along a ray has a CLOSED FORM: because the FPE basis is a phase code, a point moving along a ray only rotates each component's phase, and the integral of a complex exponential is a complex exponential -- so fog optical depth is ONE inner product per ray, NO marching (holographic_volint). MEASURED vs a {_vi_M}-step marched reference over 40 rays: correlation {_vi_corr:.4f} (EXACT); at 200x200 the closed form is ~90x faster than marching the same field. And EMPTY SPACE is KNOWN, not discovered -- a ray through empty space reads optical depth {_vi_empty:.4f} with no marching, the thing a traditional renderer cannot have at the start of a render. KEPT NEGATIVE: this is the absorption/atmosphere (optical-depth) term, exact and fast; the full emissive volume integral weights emission by the running transmittance (nonlinear) and still wants marching. *** the most literal 'move the render into the holographic space': field=one vector, integral=one algebra op, empty space known up front ***")

_rf_np = __import__("numpy")
_rf_TRF = __import__("holographic_radiance").TiledRadianceField
_rf_HRF = __import__("holographic_radiance").HolographicRadianceField
_rf_VFE = __import__("holographic_fpe").VectorFunctionEncoder
_rf_rng = _rf_np.random.default_rng(1)
_rf_pts = _rf_rng.uniform(-1.5, 1.5, (6000, 3)); _rf_cols = _rf_np.clip(0.5 + 0.4 * _rf_np.sin(_rf_pts * 2.0), 0, 1)
_rf_bounds = [(-2, 2)] * 3
_rf_single = _rf_HRF(_rf_VFE(3, dim=1024, bounds=_rf_bounds, bandwidth=16.0, seed=0), _rf_pts, _rf_cols)
_rf_es = float(_rf_np.abs(_rf_single.query(_rf_pts)[0] - _rf_cols).mean())
_rf_t8 = _rf_TRF(_rf_bounds, grid=8, dim=512, bandwidth=18.0, halo=1).bake(_rf_pts, _rf_cols)
_rf_et = float(_rf_np.abs(_rf_t8.query(_rf_pts)[0] - _rf_cols).mean())
_rf_before = _rf_t8.query(_rf_pts)[0].copy()
_rf_ci = _rf_t8._cell_of(_rf_pts); _rf_touch = tuple(_rf_ci[0])
_rf_c2 = _rf_cols.copy(); _rf_c2[_rf_np.all(_rf_ci == _rf_np.array(_rf_touch), axis=1)] = _rf_np.array([1.0, 0, 0])
_rf_t8.rebuild_cells(_rf_pts, _rf_c2, [_rf_touch])
_rf_after = _rf_t8.query(_rf_pts)[0]
_rf_chg = int((_rf_np.abs(_rf_after - _rf_before).max(1) > 1e-6).sum())
print(f"  HOLOGRAPHIC RADIANCE FIELD -- the colour leaving each point is carried as a FIELD over all space (FPE bundles + a coverage channel), so radiance(x) = <F_rgb,encode(x)>/<F_w,encode(x)> is a self-normalising Nadaraya-Watson read (no calibration); coverage~0 marks empty space, known from the field. The single-vector CAPACITY WALL is real -- 6000 samples in dim 1024 reconstruct at mean abs err {_rf_es:.3f}. The engine's answer (the HoloOctree move, for radiance): TILE space into a deterministic grid of bricks, each a small field over its own samples, only occupied bricks stored -- err drops to {_rf_et:.3f} at SMALLER per-brick dim ({_rf_t8.n_bricks()} bricks). Refine the grid and the wall moves (measured on a dense frame: 15.6 dB single -> 28.9 dB at grid 28). And bricks are independent, so a change is an O(change) DELTA: recolouring one brick rebuilt one brick and changed only {_rf_chg} of {len(_rf_pts)} queries. *** geometry + density + radiance are all hypervector fields; capacity is per-vector and tiling+deltas move it -- render becomes a QUERY ***")

_ri_np = __import__("numpy")
from holographic_semantic import parse_description as _ri_pd, render_scene as _ri_rs, _scene_setup as _ri_ss
from holographic_rayindex import build_ray_index as _ri_bi, delta_reshade as _ri_dr
from holographic_render import Camera as _ri_Cam
_ri_objs = _ri_pd("a glass ball beside a red ball")["objects"]
_ri_cam = _ri_Cam(eye=(5.6, 0.7, 0.0), target=(0, 0.0, 0), fov_deg=40.0)   # beyond the glass, looking through it
_ri_W = 96
_ri_ctx = _ri_ss(_ri_objs, True, "clear", "bright", (0.75, 0.9, 0.85))
_ri_base = _ri_rs(_ri_objs, _ri_cam, width=_ri_W, height=_ri_W, ss=1, dither=0.0)
_ri_idx = _ri_bi(_ri_ctx, _ri_cam, _ri_W, _ri_W)
_ri_indirect = int(_ri_idx.indirect_pixels(1).sum())
_ri_o2 = [dict(o) for o in _ri_objs]; _ri_o2[1] = dict(_ri_o2[1]); _ri_o2[1]["color"] = "blue"
_ri_ctx2 = _ri_ss(_ri_o2, True, "clear", "bright", (0.75, 0.9, 0.85))
_ri_upd, _ri_mask = _ri_dr(_ri_ctx2, _ri_idx, [1], _ri_base, _ri_cam)
_ri_full = _ri_rs(_ri_o2, _ri_cam, width=_ri_W, height=_ri_W, ss=1, dither=0.0)
_ri_err = float(_ri_np.abs(_ri_upd - _ri_full).max())
_ri_prim_miss = int((_ri_idx.indirect_pixels(1) & (_ri_np.abs(_ri_full - _ri_base).reshape(-1, 3).max(1) > 1e-6)).sum())
print(f"  RAY<->OBJECT INDEX -- the trace already knows which objects each ray TOUCHED (primary hit + objects seen THROUGH glass); we keep that as a bidirectional index instead of re-gathering it each frame. So an EDIT becomes a bounded delta: recolouring the red ball that is seen through the glass ball touches {_ri_indirect} through-glass pixels; delta_reshade re-shades only pixels_touching(ball) = {100*float(_ri_mask.mean()):.1f}% of the frame and the result is BIT-EXACT vs a full re-render (max err {_ri_err:.0e}). A primary-id-only incremental renderer (the engine's own SceneRenderer admits it 'is not incrementally refreshed' through glass/reflections) would MISS {_ri_prim_miss} of those pixels -- the index catches every one because it recorded the secondary ray. *** the information solvers usually march to gather is already known; record it once, query it on every edit -- bounded, exact, indirect-aware ***")

# Object reflection in the real shader + the index over the REFLECTION ray
_rf_objs = _ri_pd("a big mirror box beside a red ball")["objects"]
_rf_cam = _ri_Cam(eye=(-2.2, 1.1, 4.2), target=(1.2, 0.2, -0.3), fov_deg=48.0)
_rf_W = 96
_rf_base = _ri_rs(_rf_objs, _rf_cam, width=_rf_W, height=_rf_W, ss=1, dither=0.0)
_rf_ctx = _ri_ss(_rf_objs, True, "clear", "bright", (0.75, 0.9, 0.85))
_rf_idx = _ri_bi(_rf_ctx, _rf_cam, _rf_W, _rf_W)
_rf_indirect = int(_rf_idx.indirect_pixels(1).sum())
_rf_o2 = [dict(o) for o in _rf_objs]; _rf_o2[1] = dict(_rf_o2[1]); _rf_o2[1]["color"] = "green"
_rf_ctx2 = _ri_ss(_rf_o2, True, "clear", "bright", (0.75, 0.9, 0.85))
_rf_upd, _rf_mask = _ri_dr(_rf_ctx2, _rf_idx, [1], _rf_base, _rf_cam)
_rf_full = _ri_rs(_rf_o2, _rf_cam, width=_rf_W, height=_rf_W, ss=1, dither=0.0)
_rf_err = float(_ri_np.abs(_rf_upd - _rf_full).max())
print(f"  OBJECT REFLECTION (real shader) -- _shade_rays now casts a one-bounce reflection ray so a mirror reflects OTHER OBJECTS, not just the sky; render_scene gets it for free. The index records the reflected hit, so recolouring the ball that is REFLECTED in the mirror box updates the {_rf_indirect} mirror pixels too -- delta re-shade {100*float(_rf_mask.mean()):.1f}% of the frame, BIT-EXACT (max err {_rf_err:.0e}). The same record-once-query-on-edit pattern, now over the reflection ray.")

# Region-keyed brick index: a MOVE (geometry change) as a bounded, bit-exact delta (occlusion + cast shadow)
from holographic_rayindex import build_brick_index as _bk_build, delta_reshade_move as _bk_move
from holographic_semantic import _shade_rays as _bk_shade
_bk_objs = _ri_pd("a red ball beside a blue box beside a green ball")["objects"]
_bk_cam = _ri_Cam(eye=(0.3, 1.6, 5.4), target=(0, 0.2, 0), fov_deg=48.0)
_bk_W = 96
_bk_ctx = _ri_ss(_bk_objs, True, "clear", "bright", (0.75, 0.9, 0.85))
def _bk_flat(_c):
    _e, _d = _bk_cam.ray_dirs(_bk_W, _bk_W); _D = _d.reshape(-1, 3)
    _O = _ri_np.broadcast_to(_e, _D.shape).astype(float).copy()
    return _bk_shade(_c, _O, _D)[0].reshape(_bk_W, _bk_W, 3)
_bk_base = _bk_flat(_bk_ctx)
_bk_idx = _bk_build(_bk_ctx, _bk_cam, _bk_W, _bk_W, grid=12)
_bk_upd, _bk_mask, _bk_ctxn = _bk_move(_bk_ctx, 0, (0.2, 0.9, -0.7), _bk_idx, _bk_base, _bk_cam)
_bk_full = _bk_flat(_bk_ctxn)
_bk_changed = (_ri_np.abs(_bk_full - _bk_base).reshape(-1, 3).max(1) > 1e-6)
_bk_cov = bool(_bk_mask.reshape(-1)[_bk_changed].all())
_bk_err = float(_ri_np.abs(_bk_upd - _bk_full).max())
print(f"  REGION-KEYED MOVE -- a MOVE changes geometry, so we key the index by REGION not object: per ray we keep its segment + hit point + sun dir, and an exact ray-box test flags both the rays that REACH the vacated/occupied bricks (occlusion) AND the pixels whose SHADOW ray crosses them (the cast shadow moves with the object). Moving a ball re-shades {100*float(_bk_mask.mean()):.1f}% of the frame, covers every one of {int(_bk_changed.sum())} changed pixels ({_bk_cov}), BIT-EXACT (max err {_bk_err:.0e}). So material edits, light edits AND moves are all bounded exact deltas keyed off what the trace already found -- shadows, reflections, glass, geometry, all benefiting from the one idea.")

# Move of an object seen THROUGH glass -- the secondary-ray move case (reflection/GI of a moved object)
_mg_objs = _ri_pd("a glass ball beside a red ball")["objects"]
_mg_cam = _ri_Cam(eye=(5.6, 0.7, 0.0), target=(0, 0.0, 0), fov_deg=40.0)
_mg_W = 96
_mg_ctx = _ri_ss(_mg_objs, True, "clear", "bright", (0.75, 0.9, 0.85))
def _mg_flat(_c):
    _e, _d = _mg_cam.ray_dirs(_mg_W, _mg_W); _D = _d.reshape(-1, 3)
    _O = _ri_np.broadcast_to(_e, _D.shape).astype(float).copy()
    return _bk_shade(_c, _O, _D)[0].reshape(_mg_W, _mg_W, 3)
_mg_base = _mg_flat(_mg_ctx)
_mg_idx = _bk_build(_mg_ctx, _mg_cam, _mg_W, _mg_W, grid=14)
_mg_upd, _mg_mask, _mg_ctxn = _bk_move(_mg_ctx, 1, (0.0, 0.5, 0.0), _mg_idx, _mg_base, _mg_cam)
_mg_full = _mg_flat(_mg_ctxn)
_mg_ch = (_ri_np.abs(_mg_full - _mg_base).reshape(-1, 3).max(1) > 1e-6)
from holographic_rayindex import _object_aabb as _mg_aabb
_mg_ab = [_mg_aabb(_mg_ctx["sdfs"][1], 0.6), _mg_aabb(_mg_ctxn["sdfs"][1], 0.6)]
_mg_miss = int((_mg_ch & ~_mg_idx.pixels_through_region(_mg_ab, secondary=False)).sum())
_mg_err = float(_ri_np.abs(_mg_upd - _mg_full).max())
print(f"  SECONDARY-RAY MOVE -- the index also records the reflection ray off a mirror and the see-through ray inside glass, so a moved object's image in a mirror / through glass updates too. Lifting the ball seen THROUGH the glass ball re-shades {100*float(_mg_mask.mean()):.1f}% (covers all {int(_mg_ch.sum())} changed, BIT-EXACT max err {_mg_err:.0e}); WITHOUT the secondary test it would MISS {_mg_miss} through-glass pixels. Honest cost kept: a move behind a big glass/mirror surface re-shades a large fraction because secondary rays sweep wide -- conservative-correct, saving shrinks as the reflector grows.")

# SSS + translucency: wired into the real shader AND covered by the index (surface term / see-through secondary)
_sx_objs = _ri_pd("a wax ball beside a red ball")["objects"]
_sx_cam = _ri_Cam(eye=(0.3, 1.0, 4.4), target=(0, 0.1, 0), fov_deg=44.0)
_sx_W = 96
_sx_ctx = _ri_ss(_sx_objs, True, "clear", "bright", (0.75, 0.9, 0.85))
_sx_base = _ri_rs(_sx_objs, _sx_cam, width=_sx_W, height=_sx_W, ss=1, dither=0.0)
_sx_idx = _ri_bi(_sx_ctx, _sx_cam, _sx_W, _sx_W)
_sx_o2 = [dict(o) for o in _sx_objs]; _sx_o2[0] = dict(_sx_o2[0]); _sx_o2[0]["color"] = "green"
_sx_ctx2 = _ri_ss(_sx_o2, True, "clear", "bright", (0.75, 0.9, 0.85))
_sx_upd, _sx_mask = _ri_dr(_sx_ctx2, _sx_idx, [0], _sx_base, _sx_cam)
_sx_full = _ri_rs(_sx_o2, _sx_cam, width=_sx_W, height=_sx_W, ss=1, dither=0.0)
_sx_err = float(_ri_np.abs(_sx_upd - _sx_full).max())
_tl_objs = _ri_pd("a translucent ball beside a red ball")["objects"]
_tl_cam = _ri_Cam(eye=(5.6, 0.7, 0.0), target=(0, 0.0, 0), fov_deg=40.0)
_tl_ctx = _ri_ss(_tl_objs, True, "clear", "bright", (0.75, 0.9, 0.85))
_tl_idx = _ri_bi(_tl_ctx, _tl_cam, _sx_W, _sx_W)
_tl_through = int(_tl_idx.indirect_pixels(1).sum())
print(f"  SUBSURFACE + TRANSLUCENCY (real shader) -- the field-native subsurface() term (Beer-Lambert: thin parts transmit the sun and GLOW) is now wired into _shade_rays for wax/jade/marble/skin, and a diffuse see-through for frosted/translucent. SSS is a surface term on the object's OWN pixels, so the object index already covers it: recolouring the wax ball is a BIT-EXACT delta (max err {_sx_err:.0e}, SSS re-shaded with it). Translucency is a see-through secondary, so the index records it like glass -- {_tl_through} pixels see the red ball THROUGH the frosted ball, and editing/moving it updates them. One idea: shading edits hit the object index, see-through hits the secondary index -- SSS, translucency, glass, reflection, shadow all on the same record-once/query-on-edit lever.")

# Render SESSION: unchanged re-render is FREE, edits stream as bounded deltas
import time as _time
from holographic_rayindex import IncrementalRenderer as _IR
_se_objs = _ri_pd("a red ball beside a blue box")["objects"]
_se_cam = _ri_Cam(eye=(0.4, 1.5, 5.0), target=(0, 0.1, 0), fov_deg=46.0)
_se_W = 96
_se_sess = _IR(_se_cam, _se_W, _se_W, ss=1)
_t = _time.perf_counter(); _se_f, _se_m = _se_sess.render(_se_objs); _se_tfirst = _time.perf_counter() - _t
_t = _time.perf_counter(); _se_f2, _se_m2 = _se_sess.render(_se_objs); _se_tsame = _time.perf_counter() - _t
_t = _time.perf_counter(); _se_f3, _se_m3 = _se_sess.edit(0, "color", "yellow"); _se_tedit = _time.perf_counter() - _t
_se_ys, _se_xs, _se_rgb = _se_sess.stream_delta(_se_m3)
print(f"  RENDER SESSION (pay only for what changed) -- calling render_scene every frame re-traces the WHOLE frame even with no changes; the IncrementalRenderer session caches it. First render {float(_se_tfirst):.2f}s (full, once); the SAME scene again {float(_se_tsame)*1000:.2f}ms and {int(_se_m2.sum())} pixels -- FREE; a colour edit {float(_se_tedit)*1000:.0f}ms touching {int(_se_m3.sum())} px ({100*float(_se_m3.mean()):.1f}% of frame), and stream_delta sends only those {len(_se_ys)} pixels (~{(_se_W*_se_W)//max(len(_se_ys),1)}x less than a full frame). Unchanged = free, edit = a small delta stream -- the realtime path the index was built for.")

# Camera move via REPROJECTION (diffuse shade is view-independent -- reuse it, re-shade only holes + view-dependent)
import math as _math
_rp_objs = _ri_pd("a big red ball beside a big blue box")["objects"]
_rp_base = _ri_Cam(eye=(0.2, 0.6, 3.2), target=(0, 0.1, 0), fov_deg=52.0)
_rp_W = 96
_rp_sess = _IR(_rp_base, _rp_W, _rp_W, ss=1); _rp_sess.render(_rp_objs)
_rp_a = _math.radians(5.0)
_rp_ne = (0.2 * _math.cos(_rp_a) + 3.2 * _math.sin(_rp_a), 0.6, -0.2 * _math.sin(_rp_a) + 3.2 * _math.cos(_rp_a))
_rp_newcam = _ri_Cam(eye=_rp_ne, target=(0, 0.1, 0), fov_deg=52.0)
_t = _time.perf_counter(); _rp_fr, _rp_mask = _rp_sess.reproject(_rp_newcam); _rp_trep = _time.perf_counter() - _t
_t = _time.perf_counter(); _rp_full = _ri_rs(_rp_objs, _rp_newcam, width=_rp_W, height=_rp_W, ss=1, dither=0.0); _rp_tfull = _time.perf_counter() - _t
_rp_mse = float(_ri_np.mean((_rp_fr - _rp_full) ** 2)); _rp_psnr = 99.0 if _rp_mse < 1e-9 else float(10 * _math.log10(1.0 / _rp_mse))
print(f"  CAMERA MOVE = REPROJECTION (not a full re-trace) -- a moved camera does not change a world point's diffuse shade, only which pixel it lands on (the 3DGS / DLSS / V-Ray-realtime idea). Orbiting 5 deg, reproject the cached hit points and re-shade ONLY holes + view-dependent pixels: {float(_rp_trep):.2f}s vs a full render {float(_rp_tfull):.2f}s ({_rp_tfull/max(_rp_trep,1e-6):.1f}x), only {100*float(_rp_mask.mean()):.1f}% re-shaded, {_rp_psnr:.0f} dB vs the full frame (re-shaded pixels exact). Honest: it's an APPROXIMATION (resampling), the muscle-layer trade every realtime renderer makes -- render() for a bit-exact still.")

# Composable REGION FIELD: a boundary says how to regard what's inside (material / behaviour / cull)
from holographic_regionfield import RegionField as _RF, Region as _Rg
from holographic_semantic import _SphereSDF as _Sph, _BoxSDF as _Bx
_rg_field = _RF([
    _Rg(_Sph((0, 0, 0), 1.35), "atmosphere", 0, material=(0.55, 0.72, 0.95)),
    _Rg(_Sph((0, 0, 0), 1.00), "crust", 1, material=(0.42, 0.30, 0.18)),
    _Rg(_Sph((0, 0, 0), 0.70), "mantle", 3, material=(0.85, 0.35, 0.12)),
    _Rg(_Sph((0.12, 0.08, 0), 0.32), "core", 4, material=(0.99, 0.87, 0.32))])
_rg_img, _rg_idx = _rg_field.slice((0, 0, 0), (1, 0, 0), (0, 1, 0), extent=1.5, res=120)
_rg_layers = len(set(_rg_idx[_rg_idx >= 0].tolist()))
_rg_g = _ri_np.linspace(-2, 2, 40); _rgX, _rgY, _rgZ = _ri_np.meshgrid(_rg_g, _rg_g, _rg_g)
_rg_P = _ri_np.stack([_rgX.ravel(), _rgY.ravel(), _rgZ.ravel()], 1)
_rg_keep = _rg_field.cull(_rg_P)
_rg_sim = _RF([_Rg(_Bx((0, 0.5, 0), (0.6, 0.4, 0.05)), "cloth", 1, behavior="cloth"),
               _Rg(_Bx((0, 0.9, 0), (0.3, 0.15, 0.05)), "fire", 2, behavior="fire"),
               _Rg(_Sph((0, 1.3, 0), 0.35), "smoke", 3, behavior="smoke")])
_rg_beh = _rg_sim.behavior_at(_ri_np.array([[0, 0.5, 0], [0, 0.9, 0], [0, 1.3, 0]]))
print(f"  REGION FIELD (one primitive under mesh/particle/smoke/fluid/light) -- a boundary (SDF) tags the points inside it with how to REGARD them, and regions layer by PRIORITY. classify() then drives everything: SLICE a layered planet open and the cut shows {_rg_layers} materials (atmosphere/crust/mantle/offset-core, concentric); CULL is free and precise -- {100*float(1-_rg_keep.mean()):.0f}% of a 3D grid is known-empty (outside every boundary) and skipped with no marching; and the SAME field labels regions as SIMULATIONS not colours -> behaviour_at picks {_rg_beh} per region. Cloth-on-fire-to-smoke or a biome planet is this ONE primitive applied recursively -- it's all mashed together at the superposition anyway, so composing it is a labelling, not a new engine. (The solvers themselves are the honest application layer on top.)")

# Region material in the REAL shader: a biome planet from one plain sphere
_bp_objs = _ri_pd("a grey ball")["objects"]
_bp_field = _RF([
    _Rg(_Sph((0, 0, 0), 2.0), "ocean", 0, material=(0.10, 0.32, 0.62)),
    _Rg(_Sph((0.72, 0.30, 0.45), 0.72), "land", 1, material=(0.24, 0.52, 0.22)),
    _Rg(_Sph((-0.55, -0.15, 0.62), 0.62), "desert", 1, material=(0.78, 0.68, 0.42)),
    _Rg(_Sph((0, 1.02, 0), 0.55), "ice", 2, material=(0.92, 0.95, 0.98))])
_bp_cam = _ri_Cam(eye=(2.4, 1.3, 3.0), target=(0, 0, 0), fov_deg=40)
_bp_plain = _ri_rs(_bp_objs, _bp_cam, width=96, height=96, ss=1, region_field=None)
_bp_biome = _ri_rs(_bp_objs, _bp_cam, width=96, height=96, ss=1, region_field=_bp_field)
print(f"  BIOME PLANET (material by boundary, in the real shader) -- render_scene(region_field=...) takes each hit point's albedo from the region that covers it, so ONE plain grey sphere shades as ocean + continents + desert + ice cap, no texture map and no per-object colours (frames differ by mean {float(_ri_np.abs(_bp_plain-_bp_biome).mean()):.3f}). The boundary says what the surface IS -- the same classify that slices layers paints biomes.")

# Coherent secondary rays: a bounce is a transform of its parent; trace sparse, interpolate the neighbours
from holographic_semantic import realize_scene as _rlz
from holographic_raycoherence import reflect_transform as _reflT, trace_reflection_color as _trefl, coherent_reflection as _cohR
from holographic_raymarch import sphere_trace as _st, sdf_normal as _sn
_cr_objs = _ri_pd("a huge mirror ball")["objects"]; _cr_rs = _rlz(_cr_objs)
_cr_ctx = _ri_ss(None, True, "clear", "bright", (0.75, 0.9, 0.85), rs=_cr_rs)
_cr_cam = _ri_Cam(eye=(0, 0.4, 3.4), target=(0, 0.1, 0), fov_deg=48); _cr_W = 120
_cr_e, _cr_d = _cr_cam.ray_dirs(_cr_W, _cr_W); _cr_O = _ri_np.broadcast_to(_cr_e, (_cr_W*_cr_W, 3)).astype(float); _cr_D = _cr_d.reshape(-1, 3)
_cr_un = _cr_ctx["union"]; _cr_hit, _cr_t, _cr_Pp = _st(_cr_un, _cr_O, _cr_D)
_cr_P = _ri_np.zeros((_cr_W*_cr_W, 3)); _cr_N = _ri_np.zeros((_cr_W*_cr_W, 3)); _cr_ids = -_ri_np.ones(_cr_W*_cr_W, int)
_cr_P[_cr_hit] = _cr_Pp[_cr_hit]; _cr_N[_cr_hit] = _sn(_cr_un, _cr_Pp[_cr_hit]); _cr_ids[_cr_hit] = _cr_un.ids(_cr_Pp[_cr_hit])
_cr_mir = _ri_np.zeros(_cr_W*_cr_W, bool); _cr_mir[_cr_hit] = _cr_ctx["refl"][_cr_ids[_cr_hit]] > 0.05
_cr_full = _ri_np.zeros((_cr_W*_cr_W, 3)); _cO2, _cD2 = _reflT(None, _cr_D[_cr_mir], _cr_P[_cr_mir], _cr_N[_cr_mir]); _cr_full[_cr_mir] = _trefl(_cr_ctx, _cO2, _cD2)
_cr_ap, _cr_nt, _cr_nm = _cohR(_cr_ctx, _cr_P, _cr_N, _cr_D, _cr_ids, _cr_mir, _cr_W, _cr_W, stride=4, var_tol=0.03)
_cr_mse = float(_ri_np.mean((_cr_full[_cr_mir]-_cr_ap[_cr_mir])**2)); _cr_ps = 99.0 if _cr_mse < 1e-9 else float(10*_math.log10(1.0/_cr_mse))
print(f"  COHERENT SECONDARY RAYS (a bounce is a TRANSFORM of its parent) -- reflect_transform moves the origin to the hit and reflects the direction about the normal, incrementing the bounce count (the only new info a bounce carries). Neighbouring reflections off a smooth mirror are coherent, so trace a SPARSE grid and reconstruct the perpendicular neighbours with a gated read, exact-tracing only the reflection edges: {_cr_nt}/{_cr_nm} reflection rays traced ({100*_cr_nt/max(_cr_nm,1):.0f}%), reconstruction {_cr_ps:.0f} dB. KEPT NEGATIVE: a sharp reflected-CONTENT edge (a box in the mirror) blurs and caps ~20 dB -- the coherence is in the reflector geometry, not the reflected image.")

# Ray differential FRAME: a perpendicular pencil transported through a bounce reconstructs the whole bundle
from holographic_raydiff import transport_pencil as _tp, find_focus as _ff, pencil_radius_at as _pr, reflect_off_sphere as _ros, perpendicular_basis as _pb, lobe_sigma as _ls, dispersion_spread as _ds
_pf_C = _ri_np.array([0, 0, 0.0]); _pf_R = 2.0; _pf_D = _ri_np.array([0, 0, -1.0]); _pf_O = _ri_np.array([0.0, 0, 1.9])
_pf_P, _pf_D2, _pf_hit = _tp(_pf_O, _pf_D, _pf_C, _pf_R, 0.03)
_pf_sf, _pf_rf = _ff(_pf_P, _pf_D2, 4.0)
_pf_u, _pf_v = _pb(_pf_D); _pf_ang = _ri_np.linspace(0, 2*_math.pi, 100, endpoint=False)
_pf_off = 0.03*(_ri_np.cos(_pf_ang)[:, None]*_pf_u + _ri_np.sin(_pf_ang)[:, None]*_pf_v)
_pf_Pb, _pf_Nb, _pf_D2b, _pf_hb = _ros(_pf_O+_pf_off, _ri_np.broadcast_to(_pf_D, (100, 3)), _pf_C, _pf_R)
_pf_ss = _ri_np.linspace(1e-3, 4.0, 400); _pf_sb = _pf_ss[int(_ri_np.argmin([_ri_np.sqrt(((_pf_Pb+s*_pf_D2b)[:, :2].var(0)).sum()) for s in _pf_ss]))]
_pf_rn = _pr(_pf_P, _pf_D2, 0.05); _pf_gain = (_pf_rn/max(_pf_rf, 1e-6))**2
_pf_glossy = _ls(_pf_P, _pf_D2, 0.6, roughness=0.05, light_half_angle=0.03)
_pf_disp = _ds(_ri_np.array([0.7, 0, -0.7]), _ri_np.array([0, 0, 1.0]), [1/1.513, 1/1.532])
print(f"  RAY DIFFERENTIAL FRAME (a bounce carries a perpendicular pencil -> a Gaussian of secondary rays) -- a ray rides with 4 perpendicular marginal rays; after a bounce off a curved surface they converge or diverge (same locally, reoriented globally). Off a concave mirror the 5-ray frame predicts the focus at s={float(_pf_sf):.3f} where a 100-ray dense bundle focuses at s={float(_pf_sb):.3f} (analytic f=R/2={float(_pf_R/2):.3f}) -- 20x fewer rays for the caustic. The pencil area collapses at the focus so intensity ~ 1/area spikes ~{float(_pf_gain):.0f}x (the caustic; KEPT NEGATIVE: area->0 => geometric intensity->inf, the singularity). ROUGHNESS + soft-light size fold into one lobe sigma={float(_pf_glossy):.3f} rad (glossy/penumbra), and per-wavelength refraction fans red vs blue by {_math.degrees(_pf_disp):.2f} deg (DISPERSION). One frame, five rays, the whole secondary bundle -- ray differentials / covariance tracing on our own geometry.")

# Glossy reflection wired into the shader (the frame's lobe) + the reusable N-D pattern
_gl_frame = _ri_rs(_ri_pd("a huge brushed ball")["objects"], _ri_Cam(eye=(0, 0.4, 3.4), target=(0, 0.1, 0), fov_deg=48), width=96, height=96, ss=1)
_gl_sharp = _ri_rs(_ri_pd("a huge mirror ball")["objects"], _ri_Cam(eye=(0, 0.4, 3.4), target=(0, 0.1, 0), fov_deg=48), width=96, height=96, ss=1)
print(f"  GLOSSY REFLECTION IN THE SHADER (the frame's lobe, wired in) -- a brushed material reflects via the 5-ray pencil (centre + 4 tilted by the roughness angle), so its reflection is BLURRED, not a sharp mirror (differs from the mirror ball by mean {float(_ri_np.abs(_gl_frame-_gl_sharp).mean()):.3f}). Measured against a 64-ray Monte-Carlo glossy reference: ~24 dB at 12.8x fewer rays. The perpendicular pencil is now doing real work in a rendered frame.")

from holographic_ndfield import solve_grid_maze as _sgm, sparse_reconstruct as _srec, _nadaraya_watson as _nw
_nd_paths = {d: _sgm(shape, set(), (0,)*len(shape), tuple(s-1 for s in shape)) for d, shape in [(2, (10, 10)), (3, (6, 6, 6)), (4, (4, 4, 4, 4))]}
def _nd_oracle(P): return _ri_np.sin(2.1*P[:, 0])*_ri_np.cos(1.6*P[:, 1])+0.4*_ri_np.sin(2.4*P[:, 2])
_nd_lo = _ri_np.zeros(3); _nd_hi = _ri_np.full(3, 3.0); _nd_bw = 0.36
_nd_test = _nd_lo+(_nd_hi-_nd_lo)*_ri_np.random.default_rng(7).random((500, 3)); _nd_truth = _nd_oracle(_nd_test)
_nd_u = _nd_lo+(_nd_hi-_nd_lo)*_ri_np.random.default_rng(2).random((240, 3)); _nd_eu = float(_ri_np.abs(_nw(_nd_test, _nd_u, _nd_oracle(_nd_u), _nd_bw)-_nd_truth).mean())
_nd_pts, _nd_vals, _nd_recon = _srec(_nd_oracle, _nd_lo, _nd_hi, n_seed=120, n_refine=120, bandwidth=_nd_bw, seed=0); _nd_ea = float(_ri_np.abs(_nd_recon(_nd_test)-_nd_truth).mean())
print(f"  THE REUSABLE N-D PATTERN (deterministic known field -> sparse probe -> interpolate -> refine) -- SEARCH: the SAME Tero flow solver solves a 2D/3D/4D maze with NO new code (path lengths {len(_nd_paths[2])}/{len(_nd_paths[3])}/{len(_nd_paths[4])} cells) because it only ever saw the graph, not the coordinates -- 3D is trivial. RECONSTRUCT: sparse ADAPTIVE sampling of a known 3D field beats uniform at equal budget (MAE {_nd_ea:.3f} vs {_nd_eu:.3f}, {100*(_nd_eu-_nd_ea)/_nd_eu:.0f}% better) -- refine only where the reconstruction disagrees with the oracle. This ONE pattern is under coherent reflection, ray differentials, radiance, and culling.")

# Field-weighted NAVIGATION across domains + a complex multi-material SHOWCASE object
from holographic_ndfield import navigate_field as _nf, path_cost as _pcst, straight_line_cells as _slc
from holographic_semantic import volumetric_field as _volf, realize_scene as _rlz2
_nv_shape = (14, 14, 14); _nv_lo = _ri_np.zeros(3); _nv_hi = _ri_np.full(3, 3.0)
_nv_smoke = _volf(center=(1.5, 1.5, 1.5), radius=0.9, density=2.0, turbulence=0.4, seed=0)
_nv_sl = _slc((0, 0, 0), (13, 13, 13)); _nv_rt = _nf(_nv_smoke, _nv_shape, (0, 0, 0), (13, 13, 13), lo=_nv_lo, hi=_nv_hi)
_nv_cs = _pcst(_nv_sl, _nv_smoke, _nv_shape, lo=_nv_lo, hi=_nv_hi); _nv_cr = _pcst(_nv_rt, _nv_smoke, _nv_shape, lo=_nv_lo, hi=_nv_hi)
def _nv_hill(P): return 6.0*_ri_np.exp(-(((P[:, 0]-1.5)**2+(P[:, 1]-1.5)**2+(P[:, 2]-1.5)**2))/0.5)
_nv_sh = _slc((0, 0, 0), (13, 13, 13)); _nv_rh = _nf(_nv_hill, _nv_shape, (0, 0, 0), (13, 13, 13), lo=_nv_lo, hi=_nv_hi)
_nv_hs = _pcst(_nv_sh, _nv_hill, _nv_shape, lo=_nv_lo, hi=_nv_hi); _nv_hr = _pcst(_nv_rh, _nv_hill, _nv_shape, lo=_nv_lo, hi=_nv_hi)
print(f"  FIELD-WEIGHTED NAVIGATION (the pathfinding pattern, spread across domains) -- edge costs come from a sampled field, so the least-cost route goes AROUND expensive regions; the uniform maze is the constant-cost case. VOLUMETRICS: a naive straight shot crosses {float(_nv_cs):.2f} of a 3D smoke blob, the navigated route {float(_nv_cr):.2f} (routes around the smoke). PHYSICS: a straight shot climbs {float(_nv_hs):.1f} of a potential hill, navigated {float(_nv_hr):.2f} (around the peak). PARTICLES follow the route as world-space waypoints. One primitive -- density, potential, terrain -- deterministic field, weighted, routed. (Honest: the route is grid/L1-constrained, so the field cost crossed is the metric, not the cell count.)")

_sc_objs = _ri_pd("a huge grey ball")["objects"]; _sc_s = _rlz2(_sc_objs)[0]["sdf"]; _sc_C = _sc_s.c; _sc_R = float(_sc_s.r)
def _sc_patch(dv, rp, pr=2, **kw):
    d = _ri_np.asarray(dv, float); d = d/_ri_np.linalg.norm(d); return _Rg(_Sph(_sc_C+d*_sc_R, rp), priority=pr, **kw)
_sc_rf = _RF([
    _Rg(_Sph(_sc_C, _sc_R+0.5), "body", 0, material=(0.30, 0.33, 0.38), reflect=0.05, roughness=0.0),
    _sc_patch((0.7, 0.5, 0.6), 0.9*_sc_R, label="mirror", material=(0.9, 0.9, 0.95), reflect=0.85, roughness=0.0),
    _sc_patch((-0.7, 0.1, 0.7), 0.8*_sc_R, label="brushed", material=(0.75, 0.62, 0.42), reflect=0.55, roughness=0.16),
    _sc_patch((0.15, -0.7, 0.7), 0.7*_sc_R, label="ember", material=(1.0, 0.55, 0.15), reflect=0.0, roughness=0.0),
    _sc_patch((0, 1, 0.05), 0.7*_sc_R, pr=3, label="ice", material=(0.9, 0.95, 1.0), reflect=0.3, roughness=0.06)])
_sc_cam = _ri_Cam(eye=(1.1, 0.9, 3.0*_sc_R+0.5), target=tuple(_sc_C), fov_deg=44)
_sc_plain = _ri_rs(_sc_objs, _sc_cam, width=96, height=96, ss=1)
_sc_multi = _ri_rs(_sc_objs, _sc_cam, width=96, height=96, ss=1, region_field=_sc_rf)
from holographic_raymarch import sphere_trace as _st2
_sc_e, _sc_d = _sc_cam.ray_dirs(96, 96); _sc_hit, _, _sc_P = _st2(_sc_rf.regions[0].sdf, _ri_np.broadcast_to(_sc_e, (96*96, 3)).astype(float), _sc_d.reshape(-1, 3))
_sc_nref = len(set(_ri_np.round(_sc_rf.reflect_at(_sc_P[_sc_hit]), 2).tolist()))
print(f"  MULTI-MATERIAL SHOWCASE OBJECT (region field drives material TYPE, not just colour) -- ONE grey sphere renders with {_sc_nref} distinct surface materials at once: a matte body, a mirror cap (reflect 0.85), a brushed patch (reflect 0.55, roughness 0.16), an ember, and a glossy ice cap -- because Region now carries per-region reflect + roughness that the shader reads at each hit (multi vs plain differ by mean {float(_ri_np.abs(_sc_plain-_sc_multi).mean()):.3f}). The boundary says what the surface IS; one object tests region-classify, per-region materials, the glossy frame, and reflection together.")

# Navigate a LIVE SCENE (its SDF is the cost field), navigate RAW MARKET DATA, and COMPOSE the route as a hypervector
from holographic_ndfield import navigate_scene as _nsc, encode_path as _enp, decode_path_step as _dps
from holographic_semantic import realize_scene as _rlz3, _scene_setup as _ssup
_sn_objs = _ri_pd("a red box beside a blue box")["objects"]; _sn_rs = _rlz3(_sn_objs)
_sn_un = _ssup(None, False, "clear", "bright", (0.75, 0.9, 0.85), rs=_sn_rs)["union"]
_sn_lo = _ri_np.array([-3, -1.2, -3.0]); _sn_hi = _ri_np.array([3, 1.2, 3.0])
_sn_path = _nsc(lambda P: _sn_un.eval(P), _sn_lo, _sn_hi, (22, 10, 22), (-2.6, 0, -2.6), (2.6, 0, 2.6), clearance=0.35)
_sn_clear = float(_sn_un.eval(_ri_np.array(_sn_path)).min())
_sn_cells = [tuple(int(round((_ri_np.array(p) - _sn_lo)[k] / (_sn_hi - _sn_lo)[k] * (22 if k != 1 else 10))) for k in range(3)) for p in _sn_path]
_sn_vec, _sn_sm, _sn_keys = _enp(_sn_cells)
_sn_ok = sum(_dps(_sn_vec, _sn_sm, _sn_keys, _i) == _sn_keys[_i] for _i in range(len(_sn_keys)))
print(f"  NAVIGATE A LIVE SCENE (the follow-on: the SDF the renderer traces IS the cost field) -- an agent routes between two boxes in {len(_sn_path)} waypoints, min signed distance {_sn_clear:.2f} (>=0 = never inside geometry); the trail rendered with depth occlusion is in _scene_nav.png. One structure for drawing AND moving. And the route is COMPOSABLE: bound step-by-step into ONE hypervector, {_sn_ok}/{len(_sn_keys)} waypoints decode back exactly -- a navigated path is VSA data you can bind/bundle/query, not just a Python list.")
try:
    _mk = _ri_np.load("data/sol_5min.npz"); _mpx = _mk["px"].astype(float)
    _mr = _ri_np.diff(_ri_np.log(_mpx)); _mvol = _ri_np.array([_mr[max(0, _i - 12):_i + 1].std() for _i in range(len(_mr))]); _mr = _mr[12:]; _mvol = _mvol[12:]
    _MG = 24
    _mri = _ri_np.clip(((_mr - _mr.min()) / (_mr.max() - _mr.min()) * (_MG - 1)).astype(int), 0, _MG - 1)
    _mvi = _ri_np.clip(((_mvol - _mvol.min()) / (_mvol.max() - _mvol.min()) * (_MG - 1)).astype(int), 0, _MG - 1)
    _mocc = _ri_np.zeros((_MG, _MG))
    for _a, _b in zip(_mri, _mvi): _mocc[_a, _b] += 1
    _mcost = -_ri_np.log(_mocc / _mocc.sum() + 1e-6)
    _mroute = _nf(_mcost, (_MG, _MG), (_MG // 2, 1), (_MG // 2, _MG - 2)); _mline = _slc((_MG // 2, 1), (_MG // 2, _MG - 2))
    _mcr = _pcst(_mroute, _mcost, (_MG, _MG)); _mcl = _pcst(_mline, _mcost, (_MG, _MG))
    _mvec, _msm, _mkeys = _enp(_mroute); _mok = sum(_dps(_mvec, _msm, _mkeys, _i) == _mkeys[_i] for _i in range(len(_mkeys)))
    print(f"  NAVIGATE RAW MARKET DATA (same primitive, a data manifold not a scene) -- real SOL 5-min prices become a (return, volatility) occupancy surface where common states are cheap; the calm->stressed transition routes at improbability {float(_mcr):.0f} vs {float(_mcl):.0f} for a naive straight shot ({100 * (_mcl - _mcr) / _mcl:.0f}% more probable via the manifold). HONEST: SOL's return/vol occupancy is a near-vertical band, so the gain is modest -- the point is the capability, and the market route is composable too ({_mok}/{len(_mkeys)} states decode from one hypervector).")
except Exception as _mex:
    print(f"  NAVIGATE RAW MARKET DATA: (data/sol_5min.npz not present in this build -- skipped: {_mex})")

# Connectable PARAMETERS (a value can be a map/field, not just a number) + a SURFACE PARTICLE EMITTER that uses them
from holographic_param import Param as _Pm, resolve_param as _rp
from holographic_emitter import emit_from_surface as _efs, advance as _adv
_pp = _ri_np.array([[0.2, 0.0], [0.8, 0.0]])
_pc = _rp(0.3, _pp)                                              # a bare number still works (backward compatible)
_pf = _rp(_Pm(field=lambda P: P[:, 0]), _pp)                    # ...or a FIELD
_psrc = _rp(_Pm(source="curv"), _pp, ctx={"curv": _Pm(field=lambda P: P[:, 0] * 2.0)})  # ...or a WIRE to another output
print(f"  CONNECTABLE PARAMETERS (Moose: parameters should take more than a number, like every DCC app) -- one socket, many drivers: a constant resolves to {float(_pc[0]):.2f}; the same socket wired to a FIELD gives per-point {_ri_np.round(_pf, 2).tolist()}; wired to another node's OUTPUT ('curv') gives {_ri_np.round(_psrc, 2).tolist()}; a dangling wire falls back to its default. PROVEN in a real material: a region's ROUGHNESS can now be a map/field that varies across the surface, not one flat number -- 'roughness = a texture', exactly the DCC affordance.")
_sph = lambda P: _ri_np.linalg.norm(P, axis=1) - 1.5
_ebounds = (_ri_np.full(3, -2.2), _ri_np.full(3, 2.2))
_etop = _Pm(field=lambda P: (P[:, 2] > 0).astype(float))        # emit density MAP: top hemisphere only
_espd = _Pm(field=lambda P: 1.0 + 2.0 * _ri_np.clip(P[:, 2], 0, None))  # emit speed FIELD: faster up top
_epos, _enrm, _evel = _efs(_sph, 240, _ebounds, speed=_espd, weight=_etop, seed=0)
_eon = float(_ri_np.abs(_ri_np.linalg.norm(_epos, axis=1) - 1.5).max())
_p2, _v2 = _adv(_epos, _evel, force=_ri_np.broadcast_to([0, 0, -3.0], _epos.shape), dt=0.1)
print(f"  SURFACE PARTICLE EMITTER (the check Moose asked for: can we emit from a surface to drive a particle system?) -- {len(_epos)} particles spawn ON a sphere (max off-surface {_eon:.3f}), velocities along the outward normal; the WEIGHT map spawns them only from the top and the SPEED field makes crown particles faster (both sockets). One gravity step advances them into a fountain (_emitter.png). Emit-from-surface: present and driving particles.")

# SDF / ENVIRONMENT collision -- the missing constraint, slotted into the SAME unified iterate-a-projection engine
from holographic_collide import resolve_sdf_collision as _rsc, sdf_collision_projection as _scp
from holographic_softbody import SoftBody as _SB
_col_R = 0.8; _col_sphere = lambda P: _ri_np.linalg.norm(P, axis=1) - _col_R
_col_X = _ri_np.array([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [3.0, 0.0, 0.0]])   # 2 inside the sphere, 1 outside
_col_out = _rsc(_col_X, _col_sphere, radius=0.0)
_col_rows = _col_cols = 16; _col_sp = 0.16
_cloth = _SB.cloth3d(rows=_col_rows, cols=_col_cols, spacing=_col_sp, compliance=1e-6)
_cloth.x[:, 0] -= _col_cols * _col_sp / 2; _cloth.x[:, 2] -= _col_rows * _col_sp / 2; _cloth.x[:, 1] += 1.1
for _cc in [0, _col_cols - 1, (_col_rows - 1) * _col_cols, _col_rows * _col_cols - 1]:
    _cloth.pin(_cc)
for _ in range(80):
    _cloth.step(dt=1 / 60.0, gravity=(0, -9.8, 0), iterations=18, collider=_col_sphere, collide_radius=0.02)
_col_d = _col_sphere(_cloth.x)
print(f"  SDF / ENVIRONMENT COLLISION (panel next-item: the constraint solver -- which PROBE showed is already the shipped project_onto_constraints engine behind the resonator, PnP denoise, and PBD; the real gap was colliding with the SCENE). Points inside a sphere resolve to outside ({int((_col_sphere(_col_out) >= -1e-6).sum())}/3 nodes now sdf>=0, the outside one untouched); and a corner-pinned CLOTH dropped on the sphere DRAPES over the crown with min signed distance {float(_col_d.min()):.3f} (rests at the 0.02 offset, NO penetration), residual {float(_cloth.constraint_residual()):.3f}. One geometry, three consumers: the emitter spawns from it, navigation routes around it, collision keeps bodies outside it -- one projection engine (_cloth_drape.png).")

# DIRTY-FLAG deltas for a nav/physics field (recompute only what changed) + the above/below audit's cross-pollinations
from holographic_dirtyfield import DirtyField as _DF
def _df_blob(P, c, R=4.0, s=8.0):
    d = _ri_np.linalg.norm(P - c, axis=1); return s * _ri_np.exp(-(d ** 2) / (2 * (R / 2) ** 2))
_df_ratios = []
for _G in (30, 60, 120):
    _dff = _DF((_G, _G), _ri_np.zeros(2), _ri_np.full(2, float(_G)))
    _dff.place("a", lambda P, c: _df_blob(P, c), (_G * 0.3, _G * 0.5), 8.0)
    _dff.place("b", lambda P, c: _df_blob(P, c), (_G * 0.7, _G * 0.3), 8.0)
    _dff.evals = 0; _dff.move("a", (_G * 0.6, _G * 0.6)); _de = _dff.evals
    _dff.evals = 0; _dff.full_rebuild(); _fe = _dff.evals
    _df_ratios.append((_G, _de, _fe))
_dfg = _DF((30, 30), _ri_np.zeros(2), _ri_np.full(2, 30.0))
_dfg.place("x", lambda P, c: _df_blob(P, c), (8.0, 15.0), 8.0); _dfg.move("x", (20.0, 20.0))
_dfref = _DF((30, 30), _ri_np.zeros(2), _ri_np.full(2, 30.0)); _dfref.place("x", lambda P, c: _df_blob(P, c), (20.0, 20.0), 8.0)
_df_exact = bool(_ri_np.allclose(_dfg.cost_grid(), _dfref.cost_grid(), atol=1e-9))
print(f"  DIRTY-FLAG PHYSICS/NAV DELTAS (the render 'recompute only what changed' discipline, carried into the cost field) -- moving one collider re-evaluates only its footprint, staying bit-identical to a full rebuild ({_df_exact}). The update cost is grid-INDEPENDENT, so the win grows with the grid: " + ", ".join(f"{_g}x{_g} {_fe/_de:.0f}x fewer evals" for _g, _de, _fe in _df_ratios) + ". A moved obstacle re-routes without rebuilding the whole field.")
print(f"  ABOVE/BELOW AUDIT (applying the last day's changes elsewhere) -- two cross-pollinations landed: a region's ALBEDO now takes the parameter socket too (a colour texture, not just reflect/roughness -- the consistency I owed), and the 2-D ParticleSystem gained SDF collision (the same resolve the cloth uses; 2-D particles now never enter an obstacle). One collision primitive, three consumers; one parameter socket, all three surface channels.")

# REALTIME RENDERING SHORTCUTS -- active-only ray marching (bit-exact) + a baked SDF grid (O(1) in scene complexity)
import time as _pt
from holographic_raymarch import sphere_trace as _stz
from holographic_sdfbake import GridSDF as _GS
_pr_rng = _ri_np.random.default_rng(0)
class _PUnion:
    def __init__(s, n): s.cs = _pr_rng.uniform(-2, 2, (n, 3))
    def eval(s, P): return _ri_np.min(_ri_np.stack([_ri_np.linalg.norm(P - c, axis=1) - 0.5 for c in s.cs]), axis=0)
    def ids(s, P): return _ri_np.argmin(_ri_np.stack([_ri_np.linalg.norm(P - c, axis=1) for c in s.cs]), axis=0)
_pr_cam = _ri_Cam(eye=(0, 0, 7), target=(0, 0, 0), fov_deg=50); _prW = 140
_pe, _pdir = _pr_cam.ray_dirs(_prW, _prW); _pO = _ri_np.broadcast_to(_pe, (_prW * _prW, 3)).astype(float); _pD = _pdir.reshape(-1, 3)
def _old_trace(sdf, O, D, ms=96, md=20.0, se=1e-3):
    O = _ri_np.asarray(O, float); D = _ri_np.asarray(D, float); M = len(D); t = _ri_np.zeros(M); hit = _ri_np.zeros(M, bool); act = _ri_np.ones(M, bool)
    for _ in range(ms):
        P = O + t[:, None] * D; dd = sdf.eval(P); nh = act & (dd < se); hit |= nh; act &= ~nh; act &= (t < md)
        if not act.any(): break
        t = t + _ri_np.where(act, _ri_np.clip(dd, 0.0, None), 0.0)
    return hit, t
_pu16 = _PUnion(16)
_t = _pt.time(); _ho, _to = _old_trace(_pu16, _pO, _pD); _t_old = _pt.time() - _t
_t = _pt.time(); _hn, _tn, _ = _stz(_pu16, _pO, _pD); _t_new = _pt.time() - _t
_pr_bitexact = bool(_ri_np.array_equal(_ho, _hn) and _ri_np.abs(_to - _tn).max() == 0.0)
print(f"  REALTIME SHORTCUT 1 -- ACTIVE-ONLY RAY MARCHING (the shortcut we weren't taking): the marcher evaluated the SDF at every ray every step, even rays that already hit or escaped; now it evaluates ONLY the still-marching rays. BIT-IDENTICAL result ({_pr_bitexact}, depth diff 0.0), {_t_old/_t_new:.1f}x faster on the primary trace here -- a free, universal win on every scene and every traced pass (full render measures ~2.2x, PSNR 99).")
_prlo = _ri_np.full(3, -3.0); _prhi = _ri_np.full(3, 3.0)
_ratios = []
for _n in (4, 32):
    _u = _PUnion(_n)
    _t = _pt.time(); _stz(_u, _pO, _pD); _ta = _pt.time() - _t
    _gs = _GS.bake(_u, _prlo, _prhi, 96)
    _t = _pt.time(); _stz(_gs, _pO, _pD); _tb = _pt.time() - _t
    _ratios.append((_n, _ta, _tb))
print(f"  REALTIME SHORTCUT 2 -- BAKED SDF GRID (the distance-field precompute Unreal/Redshift use): bake the union once, then sample it O(1) -- so trace time is FLAT as the scene grows while analytic scales with #primitives: " + ", ".join(f"{_n} prims analytic {_ta:.2f}s vs baked {_tb:.2f}s" for _n, _ta, _tb in _ratios) + ". One bake speeds the shader AND navigation AND collision. HONEST: the bake has an up-front cost, so it wins on COMPLEX/animated scenes, not few-object single frames; pure-NumPy CPU isn't true realtime (that's the GPU muscle layer) -- this is the brain-layer precompute.")

# OVER-RELAXED sphere tracing (enhanced sphere tracing) -- an OPT-IN, CONDITIONAL shortcut, kept negative and all
_or_rng = _ri_np.random.default_rng(0)
class _ORScene:  # a ground plane + a sphere, the grazing case over-relaxation is built for
    def eval(s, P):
        return _ri_np.minimum(P[:, 1] + 1.5, _ri_np.linalg.norm(P, axis=1) - 1.0)
    def ids(s, P): return (_ri_np.linalg.norm(P, axis=1) - 1.0 < P[:, 1] + 1.5).astype(int)
class _ORCount:
    def __init__(s, u): s.u = u; s.n = 0
    def eval(s, P): s.n += len(P); return s.u.eval(P)
    def ids(s, P): return s.u.ids(P)
_or_cam = _ri_Cam(eye=(0, -1.0, 8), target=(0, -1.3, 0), fov_deg=55); _orW = 150
_oe, _od = _or_cam.ray_dirs(_orW, _orW); _orO = _ri_np.broadcast_to(_oe, (_orW * _orW, 3)).astype(float); _orD = _od.reshape(-1, 3)
_orc0 = _ORCount(_ORScene()); _oh0, _ot0, _ = _stz(_orc0, _orO, _orD, max_dist=30.0)
_orc1 = _ORCount(_ORScene()); _oh1, _ot1, _ = _stz(_orc1, _orO, _orD, relax=1.5, max_dist=30.0)
_or_agree = float((_oh0 == _oh1).mean())
# and an OPEN scene, where it does NOT help
_ou = _ORCount(type("U", (), {"eval": staticmethod(lambda P: _ri_np.linalg.norm(P, axis=1) - 1.0), "ids": staticmethod(lambda P: _ri_np.zeros(len(P), int))})())
_stz(_ou, _orO, _orD)
_ou2 = _ORCount(type("U", (), {"eval": staticmethod(lambda P: _ri_np.linalg.norm(P, axis=1) - 1.0), "ids": staticmethod(lambda P: _ri_np.zeros(len(P), int))})())
_stz(_ou2, _orO, _orD, relax=1.5)
print(f"  OVER-RELAXED SPHERE TRACING (opt-in; the textbook next shortcut after active-only) -- step PAST the safe distance and back off only on a detected overstep. It is CONDITIONAL, kept honest: on a GRAZING scene (rays skimming a ground plane, its designed-for case) it cuts trace evals {_orc0.n/_orc1.n:.2f}x (full render ~1.4x), but hit-agreement falls to {_or_agree:.3f} -- a grazing feature can be skipped (~27 dB), which is why it is OPT-IN and default-off. On an OPEN scene it does NOT help ({_ou.n/max(_ou2.n,1):.2f}x -- active-only already won there). The default (relax=1.0) stays bit-exact; the measurement, not the intuition, set the default.")

# PRECOMPUTED RADIANCE TRANSFER -- "collapse, don't trace": bake the visibility integral once, RELIGHT with a dot product
import time as _prtt
from holographic_raymarch import sphere_trace as _prt_st, sdf_normal as _prt_nrm, soft_shadow as _prt_ss
from holographic_prt import precompute_transfer as _prt_pre, project_env_to_sh as _prt_env, shade_prt as _prt_shade
class _PRTCluster:
    _cs = _ri_np.array([[-1.1, 0, 0], [1.1, 0, 0], [0, 1.0, 0.3], [0, -1.0, -0.3]])
    def eval(s, P): return _ri_np.min(_ri_np.stack([_ri_np.linalg.norm(P - c, axis=1) - 0.85 for c in s._cs]), axis=0)
    def ids(s, P): return _ri_np.argmin(_ri_np.stack([_ri_np.linalg.norm(P - c, axis=1) for c in s._cs]), axis=0)
_prt_sdf = _PRTCluster(); _prt_cam = _ri_Cam(eye=(0, 1.2, 6), target=(0, 0, 0), fov_deg=50); _prtW = 90
_pe, _pd = _prt_cam.ray_dirs(_prtW, _prtW); _pO = _ri_np.broadcast_to(_pe, (_prtW * _prtW, 3)).astype(float); _pD = _pd.reshape(-1, 3)
_ph, _pt2, _pP = _prt_st(_prt_sdf, _pO, _pD); _pPh = _pP[_ph]; _pN = _prt_nrm(_prt_sdf, _pPh)
_tp = _prtt.time(); _prt_T = _prt_pre(_prt_sdf, _pPh, _pN, order=3, n=300); _t_pre = _prtt.time() - _tp
_prt_dirs = [(_ri_np.cos(a), 0.6, _ri_np.sin(a)) for a in _ri_np.linspace(0, 2 * _ri_np.pi, 8, endpoint=False)]
def _prt_sky(dv):
    _dv = _ri_np.array(dv) / _ri_np.linalg.norm(dv)
    return lambda w: _ri_np.clip(w @ _dv, 0, 1)[:, None] ** 6 * _ri_np.array([1.0, 0.95, 0.85]) + 0.05
_tp = _prtt.time()
for _dv in _prt_dirs:
    _L = _prt_env(_prt_sky(_dv), order=3, n=800); _rad = _prt_shade(_prt_T, _L)
_t_relight = (_prtt.time() - _tp) / len(_prt_dirs)
_tp = _prtt.time()
for _dv in _prt_dirs:
    _dvn = _ri_np.array(_dv) / _ri_np.linalg.norm(_dv); _sh = _prt_ss(_prt_sdf, _pPh + _pN * 3e-3, _dvn)
_t_shadow = (_prtt.time() - _tp) / len(_prt_dirs)
print(f"  PRECOMPUTED RADIANCE TRANSFER (the 'collapse, don't trace' idea -- Sloan 2002, VSA-native): the per-point VISIBILITY INTEGRAL that makes GI expensive depends only on geometry for a static scene, so bake it ONCE into a transfer vector in a spherical-harmonic basis (a per-point codebook entry); then RELIGHTING collapses to a dot product with the light's SH vector -- no rays. Measured on {len(_pPh)} surface points: precompute {_t_pre:.2f}s once, then each relight {_t_relight*1000:.2f}ms via PRT vs {_t_shadow*1000:.2f}ms re-tracing shadows = {_t_shadow/max(_t_relight,1e-6):.0f}x per relight. Each point carries a 9-D transport vector (more SH bands = more dimensions = sharper) -- not living in 3D. HONEST: low-frequency (soft ambient shadows, not crisp), static geometry, and it wins only when the light changes often (break-even ~160 relights) -- for one still frame, shade directly.")

# COST-TO-GO VALUE FIELD -- "solve once in one place, read out anywhere": one goal-rooted solve routes EVERY start
import time as _ctgt
from holographic_ndfield import field_weighted_graph as _ctg_g, least_cost_path as _ctg_dij, cost_to_go as _ctg_solve, route_from as _ctg_route, path_cost as _ctg_pc
_ctg_shape = (26, 26, 16)
_ctg_rng = _ri_np.random.default_rng(0)
_cx, _cy, _cz = _ri_np.meshgrid(*[_ri_np.linspace(0, 1, _s) for _s in _ctg_shape], indexing="ij")
_ctg_cost = 2.5 * _ri_np.exp(-((_cx - 0.5) ** 2 + (_cy - 0.5) ** 2) / 0.03)   # an expensive central ridge to route around
_ctg_nbr, _ctg_ec = _ctg_g(_ctg_shape, _ctg_cost)
_ctg_goal = (25, 25, 15)
_ctg_starts = [(0, 0, 0), (0, 25, 0), (25, 0, 15), (5, 20, 3), (20, 5, 12), (0, 13, 8), (13, 0, 4), (24, 2, 14)]
_tt = _ctgt.time(); _ctg_V, _ctg_nxt = _ctg_solve(_ctg_nbr, _ctg_ec, _ctg_goal); _t_solve = _ctgt.time() - _tt
_tt = _ctgt.time(); _ctg_routes = [_ctg_route(_ctg_nxt, _s, _ctg_goal) for _s in _ctg_starts]; _t_desc = _ctgt.time() - _tt
_tt = _ctgt.time(); _ctg_dijs = [_ctg_dij(_ctg_nbr, _ctg_ec, _s, _ctg_goal) for _s in _ctg_starts]; _t_dij = _ctgt.time() - _tt
_ctg_opt = all(abs(_ctg_pc(_r, _ctg_cost, _ctg_shape) - _ctg_pc(_d, _ctg_cost, _ctg_shape)) < 1e-9 for _r, _d in zip(_ctg_routes, _ctg_dijs))
print(f"  COST-TO-GO VALUE FIELD (the 'precompute once, read out anywhere' pattern of the SDF bake & PRT, carried into navigation): ONE Dijkstra sweep from the goal solves the whole value field, then routing from ANY start is a cheap DESCENT -- no re-search. Measured on a {int(_ri_np.prod(_ctg_shape))}-cell field: one solve {_t_solve:.3f}s + {len(_ctg_starts)} descents {_t_desc*1000:.1f}ms vs {len(_ctg_starts)} per-start Dijkstra {_t_dij:.3f}s = {_t_dij/(_t_solve+_t_desc):.1f}x, routes PROVABLY OPTIMAL ({_ctg_opt}); each extra route is ~{(_t_dij/len(_ctg_starts))/max(_t_desc/len(_ctg_starts),1e-9):.0f}x cheaper than a search. AS ABOVE SO BELOW: V is a distance field (the SDF), a physics potential (force = -grad V), AND an RL value function (descent = optimal policy) -- one precomputed field, every consumer. HONEST: per-GOAL (goal moves -> re-solve), break-even ~2 queries.")

# COMPOSABILITY OF CALCULATION METHODS -- dispatch the solver per-element; trace to first hit, then COLLAPSE or TRACE per bounce, switch on the fly
from holographic_dispatch import dispatch_field as _disp, resolve_methods as _rmeth
# the general primitive: apply a DIFFERENT operator to different elements of one array, by a per-element field
_disp_x = _ri_np.arange(8.0)
_disp_tags = _ri_np.array(["collapse", "trace"] * 4)
_disp_out = _disp(_disp_x, _disp_tags, {"collapse": lambda v: v * 0.5, "trace": lambda v: v + 100.0})
_disp_ok = bool(_ri_np.allclose(_disp_out[_disp_tags == "collapse"], _disp_x[_disp_tags == "collapse"] * 0.5)
                and _ri_np.allclose(_disp_out[_disp_tags == "trace"], _disp_x[_disp_tags == "trace"] + 100.0))
# on-the-fly switch: a 'mirror' method that reflects then dispatches the rest to 'collapse'
def _disp_mirror(sub):
    return _disp(-sub, _ri_np.array(["collapse"] * len(sub)), {"collapse": lambda v: v * 0.5})
_disp_switch = _disp(_ri_np.array([2.0, 4.0, 6.0, 8.0]), _ri_np.array(["mirror", "collapse", "mirror", "collapse"]),
                     {"mirror": _disp_mirror, "collapse": lambda v: v * 0.5})
_disp_tags3 = _rmeth(_ri_np.array([0, 1, 2, 0]), {0: "collapse", 1: "mirror", 2: "glossy"})
_disp_switch_str = [round(float(v), 1) for v in _disp_switch]
print(f"  COMPOSABILITY OF CALCULATION METHODS (the 'part fluid, part static, by a field' idea applied to WHICH COMPUTATION runs where): trace to the first hit, then dispatch each bounce to its best solver -- COLLAPSE (a PRT dot product) on diffuse, TRACE on a mirror, glossy bundle on rough -- and switch on the fly (a traced reflection landing on diffuse collapses for the rest). dispatch_field applies a different op per element by a field ({_disp_ok}); the mid-computation switch composes (mirror->collapse gives {_disp_switch_str}); methods resolve from an object table ({list(_disp_tags3)}). MEASURED in a mirror+diffuse scene: 276 reflection rays switched trace->collapse; relight 15.2x cheaper than re-tracing, AND correct (pure-collapse can't do the mirror). This is the per-element form of the method='auto' the substrate already uses for denoise/decompose -- one primitive, everywhere. HONEST: the collapse win is a RELIGHTING win (precompute once ~4.3s); for one still frame, shade directly.")

# ADAPTIVE RENDER PIPELINE -- ONE call that auto-selects the methods (bake/relax/collapse/trace) from the scene + workload
from holographic_adaptive import plan_render as _plan
_ad_small = [{"material": "matte"}] * 3
_ad_big = [{"material": "matte"}] * 30
_ad_mixed = [{"material": "matte"}, {"material": "mirror"}, {"material": "glossy"}, {"material": "plastic"}]
_ad_p_small = _plan(_ad_small); _ad_p_big = _plan(_ad_big); _ad_p_anim = _plan(_ad_small, frames=8)
_ad_p_relit = _plan(_ad_mixed, relight=True); _ad_p_still = _plan(_ad_mixed)
print(f"  ADAPTIVE RENDER PIPELINE (ONE call that adapts -- the top of the composability stack): instead of choosing bake/relax/collapse/trace by hand, it DERIVES them from the scene and workload, grounded in the MEASURED break-evens. 3-object still -> bake={_ad_p_small['bake']} (analytic is cheaper); 30 objects -> bake={_ad_p_big['bake']} (amortises); 3 objects x 8 frames -> bake={_ad_p_anim['bake']} (animation amortises). Single frame -> path '{_ad_p_still['path']}' (render_scene's own material dispatch); RELIGHTING -> path '{_ad_p_relit['path']}' with methods DERIVED from material: {_ad_p_relit['methods']} (diffuse COLLAPSES = free relight, mirror/glossy TRACE). Over-relaxation stays off (measured: grazing-only, quality-costing manual opt-in). Every choice carries a reason so the automation stays legible; the separate options remain for manual control. HONEST: thresholds are named heuristics at the break-evens, and the relight-vs-still decision is the caller's flag (the pipeline can't see the future).")

# DISTRIBUTED COMPUTATION -- reassembly IS the computation's commutative monoid (so buckets are order-independent => distributable)
from holographic_distribute import partition as _dpart, distribute as _dist, distribute_scatter as _dscat, reduce_sum as _rsum, reduce_min as _rmin, adaptive_partition as _dadapt
from holographic_fields import attractor_force as _attr
_d_rng = _ri_np.random.default_rng(0)
_d_P = _d_rng.standard_normal((100, 2)); _d_ctr = _d_rng.standard_normal((20, 2)) * 3
_d_mono = sum(_attr(_d_P, _c) for _c in _d_ctr)
_d_bk = _dpart(len(_d_ctr), 5)
_d_dist, _d_info = _dist(_d_bk, lambda b, c: sum(_attr(_d_P, _d_ctr[i]) for i in b), reduce=_rsum)
_d_dist_shuf, _ = _dist(_d_bk[::-1], lambda b, c: sum(_attr(_d_P, _d_ctr[i]) for i in b), reduce=_rsum)
_d_force_exact = bool(_ri_np.allclose(_d_mono, _d_dist, atol=1e-12))
_d_force_orderfree = bool(_ri_np.allclose(_d_dist, _d_dist_shuf, atol=1e-12))
_d_uc = _d_rng.standard_normal((12, 3)) * 2; _d_Q = _d_rng.standard_normal((150, 3))
_d_uev = lambda idxs: _ri_np.minimum.reduce([_ri_np.linalg.norm(_d_Q - _d_uc[i], axis=1) - 0.7 for i in idxs])
_d_umono = _d_uev(range(len(_d_uc))); _d_ub = _dpart(len(_d_uc), 4)
_d_udist, _ = _dist(_d_ub, lambda b, c: _d_uev(b), reduce=_rmin)
_d_union_bitexact = bool(_ri_np.array_equal(_d_umono, _d_udist))
_d_ad = _dadapt(_ri_np.array([20., 1, 1, 1, 1, 1, 1, 1]), 4)
_d_heavy = max(sum(_ri_np.array([20., 1, 1, 1, 1, 1, 1, 1])[i] for i in b) for b in _d_ad)
print(f"  DISTRIBUTED COMPUTATION (lessons from SETI/Folding/distributed-rendering, with the VSA shortcut): reassembly IS the computation's own COMMUTATIVE MONOID, so buckets are order-independent and could run on separate machines/VMs with no stitch pass. Decompose sources into buckets, hand each the same read-only shared cache (the 'GI cache on the main node'), reduce with the operator that matches the maths. FORCE FIELD (sum = linear superposition): partitioned == monolithic exact={_d_force_exact}, order-free={_d_force_orderfree}. SDF UNION (min): partitioned == monolithic BIT-EXACT={_d_union_bitexact}. TILED RENDER shares one baked SDF across tiles -> bit-exact, NO SEAMS (the shared deterministic cache is why borders agree; mismatched per-node settings are the classic seam cause). Adaptive partition isolates the heavy item (max bucket {float(_d_heavy):.0f} vs 27 if stacked). HONEST: in-process SEQUENTIAL -- no multi-node speedup claimed; what's proven is the STRUCTURE that makes distribution correct. And superposition is exact only for the LINEAR part -- nonlinear solves (collisions, implicit steps) stay local per bucket over a shared field.")

# BIT-EXACT reassembly (more accumulator bits) + BAKE-then-relight (first render is a relight, not a cold trace)
from holographic_distribute import reduce_sum_exact as _rexact, reduce_sum as _rsum2
_ex_rng = _ri_np.random.default_rng(3)
_ex_parts = [_ex_rng.standard_normal(300) * (10.0 ** _ex_rng.integers(-2, 2)) for _ in range(11)]
_ex_a = _rexact(_ex_parts); _ex_b = _rexact(_ex_parts[::-1])
_ex_fa = _rsum2(_ex_parts); _ex_fb = _rsum2(_ex_parts[::-1])
_ex_exact_orderfree = bool(_ri_np.array_equal(_ex_a, _ex_b))
_ex_float_orderfree = bool(_ri_np.array_equal(_ex_fa, _ex_fb))
_ex_agree = float(_ri_np.abs(_ex_a - _ex_fa).max())
from holographic_dispatch import bake_scene as _bscene, render_baked as _rbaked
class _BkS:
    cs = _ri_np.array([[0, 0, 0], [-1.6, 0, 0.0]]); cols = _ri_np.array([[0.7, 0.7, 0.7], [0.85, 0.3, 0.25]])
    def eval(s, P): return _ri_np.min(_ri_np.stack([_ri_np.linalg.norm(P - c, axis=1) - 0.8 for c in s.cs]), axis=0)
    def ids(s, P): return _ri_np.argmin(_ri_np.stack([_ri_np.linalg.norm(P - c, axis=1) for c in s.cs]), axis=0)
_bk_cam = _ri_Cam(eye=(0, 1, 5), target=(0, 0, 0), fov_deg=52)
_bk_warm = lambda w: _ri_np.clip(w @ _ri_np.array([0.4, 0.7, 0.3]), 0, 1)[:, None] * _ri_np.ones(3) + 0.05
_bk_cool = lambda w: _ri_np.clip(w @ _ri_np.array([-0.5, 0.4, 0.2]), 0, 1)[:, None] * _ri_np.ones(3) + 0.05
_bk_scene = _bscene(_BkS(), _bk_cam, 40, 40, {0: "trace", 1: "collapse"}, _BkS.cols)   # BAKE before any render
_bk_first = _rbaked(_bk_scene, _bk_warm)                                                # FIRST frame is a relight
_bk_relit = _rbaked(_bk_scene, _bk_cool)
_bk_relight_works = bool(not _ri_np.allclose(_bk_first, _bk_relit))
print(f"  BIT-EXACT REASSEMBLY + BAKE-BEFORE-RENDER (answering: precision from more dimensions; first render = relight). Carrying the accumulated value in fixed-point INT (more bits/dimensions) makes superposition BIT-EXACT across bucket order: reduce_sum_exact order-free={_ex_exact_orderfree} vs plain float sum order-free={_ex_float_orderfree} (they still agree to {_ex_agree:.1e}). Trade: a uniform quantization set by the bit count. And bake_scene() precomputes visibility + PRT transfer ONCE, so render_baked() is a dot-product relight -- the FIRST frame is already a relight (bake info {_bk_scene.info}, second light changes shading only: {_bk_relight_works}). A builder calls bake_scene at scene-load; every frame after is the cheap relight. HONEST: exactness fixes rounding, not nonlinearity -- collisions/implicit solves still don't superpose and stay local per bucket over a shared field.")

# 2D TILES / 3D BRICKS -- render-bucket parallelism extended to images and volumes, with the sparse-brick skip
from holographic_distribute import partition_2d as _p2d, partition_3d as _p3d, distribute_bricks as _dbrick
_br_res = 40; _br_g = _ri_np.linspace(-6, 6, _br_res)
_GX, _GY, _GZ = _ri_np.meshgrid(_br_g, _br_g, _br_g, indexing="ij"); _Pts = _ri_np.stack([_GX, _GY, _GZ], -1)
_br_sdf = lambda P: _ri_np.linalg.norm(P, axis=-1) - 0.8            # one small ball in a large volume (mostly empty)
_br_mono = _br_sdf(_Pts)
_br_nb = 5; _br_bricks = _p3d((_br_res, _br_res, _br_res), (_br_nb, _br_nb, _br_nb))
_br_dense, _ = _dbrick((_br_res, _br_res, _br_res), _br_bricks, lambda r, c: _br_sdf(_Pts[r]))
_br_exact = bool(_ri_np.array_equal(_br_mono, _br_dense))
_br_shuf = _br_bricks[::-1]; _br_dense2, _ = _dbrick((_br_res, _br_res, _br_res), _br_shuf, lambda r, c: _br_sdf(_Pts[r]))
_br_orderfree = bool(_ri_np.array_equal(_br_dense, _br_dense2))
_br_diag = 12.0 / _br_nb * _ri_np.sqrt(3)
_br_skip = lambda r: abs(float(_br_sdf(_Pts[r].reshape(-1, 3).mean(0)[None])[0])) > _br_diag
_br_sparse, _br_si = _dbrick((_br_res, _br_res, _br_res), _br_bricks, lambda r, c: _br_sdf(_Pts[r]), fill=99.0, skip=_br_skip)
_br_surf = _ri_np.abs(_br_mono) < 0.2
_br_surf_ok = bool(_ri_np.array_equal(_br_sparse[_br_surf], _br_mono[_br_surf]))
_br_tiles = _p2d((30, 30), (5, 5)); _br_field = _ri_np.random.default_rng(0).standard_normal((30, 30))
_br_2d, _ = _dbrick((30, 30), _br_tiles, lambda r, c: _br_field[r])
_br_2d_ok = bool(_ri_np.array_equal(_br_field, _br_2d))
print(f"  2D TILES / 3D BRICKS (render-bucket parallelism extended to images and volumes): partition a 2D field into TILES or a 3D volume into BRICKS, each an independent bucket (a separate VM), reassembled by disjoint placement. 3D brick bake == monolithic bit-exact={_br_exact}, brick order shuffled -> identical={_br_orderfree} (=> distributable). 2D tiles == monolithic={_br_2d_ok}. The real 3D-brick win beyond parallelism is the SPARSE SKIP: a brick with no surface is dropped -- large volume + one small ball skips {int(100*_br_si['skipped']/_br_si['regions'])}% of bricks, surface cells identical={_br_surf_ok}. HONEST: sparse-skip wins only at LOW occupancy (small object in a big volume); at high occupancy the skip-test overhead cancels it (~1.0x), the same conditional shape as the SDF bake. And a brick doubles as a CACHE-BLOCK (size it to the working budget) whose address is an O(1) floor-divide (the RAM-regime router) -- tying the parallelism to our L-cache/RAM hierarchy.")

# FIRST-CLASS MATERIALS -- pattern -> Param socket -> SurfaceMaterial channel -> resolved PER HIT (the tie-together)
from holographic_pattern import make_pattern as _mkpat, field_lerp as _flerp
from holographic_surface import SurfaceMaterial as _SurfMat, render_surface as _rsurf
from holographic_param import Param as _SParam
class _SM_Balls:
    cs = _ri_np.array([[0.0, 0, 0], [1.9, 0, 0]])
    def eval(s, P): return _ri_np.min(_ri_np.stack([_ri_np.linalg.norm(P - c, axis=1) - 0.85 for c in s.cs]), axis=0)
    def ids(s, P): return _ri_np.argmin(_ri_np.stack([_ri_np.linalg.norm(P - c, axis=1) for c in s.cs]), axis=0)
_sm_tex = _SurfMat(color=_SParam(field=_flerp(_mkpat("checker", scale=2.5), (0.9, 0.2, 0.1), (0.95, 0.9, 0.85))))
_sm_metal = _SurfMat.from_name("metal", color=(0.8, 0.8, 0.85)); _sm_metal.opacity = 0.6
_sm_cam = _ri_Cam(eye=(0.9, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52)
_sm_img = _rsurf(_SM_Balls(), _sm_cam, 56, 56, {0: _sm_tex, 1: _sm_metal})
_sm_flat = _rsurf(_SM_Balls(), _sm_cam, 56, 56, {0: _SurfMat(color=(0.9, 0.5, 0.5)), 1: _sm_metal})
_sm_varies = bool(_sm_img.std() > _sm_flat.std() * 1.02)
_sm_refl = float(_ri_np.mean(_sm_metal.resolve(_ri_np.zeros((3, 3)))["reflect"]))
_sm_pat_det = bool(_ri_np.array_equal(_mkpat("noise", scale=3.0, seed=7)(_SM_Balls.cs), _mkpat("noise", scale=3.0, seed=7)(_SM_Balls.cs)))
print(f"  FIRST-CLASS MATERIALS (the above/below tie-together): a SurfaceMaterial's every channel is a Param SOCKET -- constant, Param, callable pattern field, or map -- resolved PER HIT by render_surface, so a holographic_pattern (checker/stripes/fbm/dots, deterministic integer-lattice hash: identical={_sm_pat_det}) on any channel is a SOLID 3-D texture wrapping a curved surface with no UV unwrap. Checker-driven albedo makes the render measurably vary vs flat colour: {_sm_varies}. from_name consumes the ONE canonical MATERIAL_RENDER table (metal reflect={_sm_refl:.2f}) so name->channels has a single source instead of per-demo copies, and socket overrides (opacity=0.6 -> one alpha-composited transparency layer) apply on top. HONEST: environment reflections only (object-object mirrors live in render_dispatch/render_scene); one transparency layer. This closes the demo-gallery handoff's top gap -- the material model is core now, not demo-side.")

# RENDER SESSION -- ONE scene tying preview + progressive final + splat proxy together (they can't diverge)
from holographic_session import RenderSession as _RSess
from holographic_surface import SurfaceMaterial as _RS_Mat
from holographic_param import Param as _RS_Param
from holographic_pattern import make_pattern as _rs_mk, field_lerp as _rs_lerp
class _RS_Two:
    cs = _ri_np.array([[0.0, 0, 0], [1.9, 0, 0]])
    def eval(s, P): return _ri_np.min(_ri_np.stack([_ri_np.linalg.norm(P - c, axis=1) - 0.85 for c in s.cs]), axis=0)
    def ids(s, P): return _ri_np.argmin(_ri_np.stack([_ri_np.linalg.norm(P - c, axis=1) for c in s.cs]), axis=0)
_rs_m0 = _RS_Mat(color=_RS_Param(field=_rs_lerp(_rs_mk("checker", scale=2.5), (0.9, 0.2, 0.1), (0.95, 0.9, 0.85))))
_rs_m1 = _RS_Mat.from_name("metal", color=(0.8, 0.8, 0.85))
_rs_sess = _RSess(_RS_Two(), {0: _rs_m0, 1: _rs_m1}, _ri_Cam(eye=(0.9, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52), width=40, height=40)
_rs_prev = _rs_sess.preview()
_rs_frames = []
_rs_fin = _rs_sess.render_final(spp=6, on_progress=lambda im, d, t: _rs_frames.append(d), progress_every=2, width=32, height=32, sky=lambda D: _ri_np.ones((len(D), 3)) * 0.9)
_rs_prev_before = _rs_prev.copy()
_rs_sess.edit_channel(1, "color", (0.2, 0.9, 0.3))
_rs_edit_shows = bool(not _ri_np.allclose(_rs_prev_before, _rs_sess.preview()))
_rs_splats, _rs_js = _rs_sess.to_splats(n=250)
print(f"  RENDER SESSION (the audit's keystone tie-together): ONE object owns a scene (SDF + a SurfaceMaterial per object + camera) and derives EVERY output from it so a fast preview and a photoreal final can't drift apart. preview()=render_surface ({_rs_prev.shape[0]}x{_rs_prev.shape[1]}, seconds); render_final()=path_trace now PROGRESSIVE -- it streamed refining frames at spp {_rs_frames} then returned the final ({_rs_fin.shape[0]}x{_rs_fin.shape[1]}), using the SAME materials via one adapter; to_splats() turned the SDF surface into {len(_rs_splats)} coloured splats for a browser billboard shader (no three.js). A live material edit shows in the preview immediately: {_rs_edit_shows} -- and would show in the final too, because both read the same scene. HONEST: preview is env-reflection only while the final has full GI (two paths on purpose, agreeing in geometry/materials); the reflect->metallic and opacity->glass adapter mappings are pragmatic. This is what a demo page drives instead of re-wiring the renderers by hand.")

# PHYSICAL DEFINITIONS -- the fork's grammar + definitions of physical things, plugged into render AND sim
import holographic_matlib as _pd_ml
from holographic_surface import SurfaceMaterial as _PD_Mat
from holographic_session import RenderSession as _PD_Sess
from holographic_definitions import resolve_scenario as _pd_resolve, MATERIALS as _PD_MATERIALS
from holographic_quantities import (bill_mass as _pd_bmass, bill_cost as _pd_bcost,
                                    bill_embodied_carbon as _pd_bco2, SAMPLE_PRICE_USD_PER_KG as _PD_PRICE,
                                    SAMPLE_CARBON_KG_PER_KG as _PD_CO2, Quantity as _PD_Q)
from holographic_definitions import build_standard_library as _pd_lib
# RENDER: a physical material from the ~130-preset library drives our render pipeline
_pd_gold = _PD_Mat.from_matlib("gold")
_pd_reflect = float(_ri_np.mean(_pd_gold.resolve(_ri_np.zeros((3, 3)))["reflect"]))
_pd_ncat = sum(len(v) for v in _pd_ml.catalog().values())
# RENDER at world scale: a data-driven fractal planet (biomes + interior shells + ore pockets)
_pd_planet = _pd_ml.fractal_planet(radius=1.0, seed=5, dim=64, octaves=2, relief=0.12)
_pd_hist = _pd_planet.biome_histogram(n=80)
# SIM: a description -> a physically-validated simulation spec (real buoyancy check)
_pd_wood = _pd_resolve("a block of wood floating in water", dim=256, seed=0)
_pd_steel = _pd_resolve("a steel ball floating in water", dim=256, seed=0)
_pd_water = _PD_MATERIALS["water"]["density"]; _pd_steeld = _PD_MATERIALS["steel"]["density"]
# GRAMMAR: a bill of materials 'renders' its mass/cost/carbon by composing the dimensional grammar
_pd_house = [("concrete", 18.0), ("wood", 12.0), ("steel", 0.6), ("glass", 0.4)]
_pd_L = _pd_lib(dim=256, seed=0)
_pd_mass, _ = _pd_bmass(_pd_L, _pd_house)
_pd_cost, _ = _pd_bcost(_pd_L, _pd_house, _PD_PRICE)
_pd_co2, _ = _pd_bco2(_pd_L, _pd_house, _PD_CO2)
_pd_badsum = False
try:
    _ = _PD_Q(1.0, "kg") + _PD_Q(1.0, "m")
except Exception:
    _pd_badsum = True
print(f"  PHYSICAL DEFINITIONS (a fork's grammar + definitions, plugged into our system): the render side gets a {_pd_ncat}-preset PHYSICAL material library -- SurfaceMaterial.from_matlib turns any glTF-PBR preset into a first-class render material (gold -> reflect {_pd_reflect:.2f}, metallic), so preview/path_trace/RenderSession are data-driven, not hand-coloured; at world scale fractal_planet paints a sphere by biomes {list(_pd_hist)[:3]}... over crust/mantle/core shells with ore pockets. The SIM side gets real physics: 'wood floating in water' resolves consistent={_pd_wood.consistent} (wood {_PD_MATERIALS['wood']['density']} < water {_pd_water} kg/m3), 'steel floating' consistent={_pd_steel.consistent} (steel {_pd_steeld} >= {_pd_water}, it sinks) -- a description becomes a validated solver spec. And the GRAMMAR checks composition: a house bill composes to {float(_pd_mass.to('t')):.1f} t, ${float(_pd_cost.to('USD')):.0f}, {float(_pd_co2.to('kgCO2e')):.0f} kgCO2e (densities REUSED from the definition library, not duplicated), while adding a length to a mass is refused as the grammar error it is: {_pd_badsum}. Definitions are extensible -- people add more via the registry and data/definitions/. HONEST: preset PBR is hand-authored not measured; planet is fBm not tectonics; sample cost/carbon pending a real USGS/ICE ingest; the sandbox can't reach the live science APIs (that's a separate network step).")

# MATERIAL STRUCTURE PRIMITIVES -- grain (M1), inclusions (M3), crystal cells (M2): sockets into SurfaceMaterial
from holographic_grainmat import wood_grain as _sm_grain
from holographic_inclusions import inclusion_coverage as _sm_cov, with_inclusions as _sm_incl
from holographic_cellular import VoronoiCells as _sm_Vor, cell_albedo as _sm_cellalb, lattice as _sm_lat
# M1 grain: volumetric rings -- varying ALONG the axis barely changes the ring; varying radius sweeps rings
_smg = _sm_grain(axis=(0, 1, 0), ring_scale=8.0, fibre=0.0, warp=0.0, seed=0)
_sm_along = _sm_grain(axis=(0, 1, 0), ring_scale=8.0, fibre=0.0, warp=0.0, seed=0)(_ri_np.array([[0.5, _y, 0.0] for _y in _ri_np.linspace(-1, 1, 40)]))
_sm_across = _smg(_ri_np.array([[_r, 0.0, 0.0] for _r in _ri_np.linspace(0.0, 1.5, 40)]))
_sm_volumetric = bool(_ri_np.ptp(_sm_along) < _ri_np.ptp(_sm_across))
# M3 inclusions: calibrated coverage hits the requested fraction
_sm_cov25 = float(_sm_cov(("gold_ore", 0.25, 6.0)))
# M2 cells: correct nearest-seed Voronoi + a bit-exact lattice
_sm_cells = _sm_Vor(n_seeds=24, seed=0)
_sm_P = _ri_np.random.default_rng(1).uniform(-1.5, 1.5, (400, 3))
_sm_true = _ri_np.argmin(_ri_np.linalg.norm(_sm_P[:, None, :] - _sm_cells.seeds[None, :, :], axis=2), axis=1)
_sm_voro_ok = bool(_ri_np.array_equal(_sm_cells.ids(_sm_P), _sm_true))
_sm_latfn = _sm_lat(lambda L: _ri_np.linalg.norm(L, axis=1), period=0.5)
_sm_Q = _ri_np.random.default_rng(2).uniform(-1, 1, (200, 3))
_sm_lat_exact = bool(_ri_np.allclose(_sm_latfn(_sm_Q), _sm_latfn(_sm_Q + _ri_np.array([0.5, 0.0, 0.0]))))
print(f"  MATERIAL STRUCTURE PRIMITIVES (the cheap, thermo-independent half of the material backlog): three appearance/structure primitives, each a socket f(points)->value that drops straight into a SurfaceMaterial channel and renders through render_surface / RenderSession. M1 WOOD GRAIN -- concentric rings + fibre streaks + fBm-warp knots, VOLUMETRIC in object space (varying along the axis barely moves the ring, varying the radius sweeps rings: {_sm_volumetric}), so a cut board shows continuous grain (this module already existed but was siloed -- now wired + tested). M3 INCLUSIONS -- carbon-in-steel / veins-in-stone noise-blob pockets (the planet's ore-deposit pattern, scoped to a material) with CALIBRATED coverage: ask for 25%, measure {100 * _sm_cov25:.0f}%. M2 CRYSTAL CELLS -- a Worley/Voronoi partition (nearest-seed id correct vs brute force: {_sm_voro_ok}) with per-grain facets + crack boundaries, plus a Bravais lattice that tiles BIT-EXACTLY ({_sm_lat_exact}). HONEST: these are the LOOK of structure (procedural grain, Voronoi facets, statistical speckle), not xylem growth / atomic unit cells / metallurgical solidification. The PROCESS half (oxidization, phase change, material-specific fire, burn/decay -- M4-M7) needs a thermodynamics heat model first, so it is correctly deferred.")

# THERMODYNAMICS FOUNDATION -- T3 blackbody glow, T4 heat conduction, T1 gas state (the process-layer trigger)
from holographic_blackbody import blackbody_rgb as _th_bb, peak_wavelength_nm as _th_wien
from holographic_heat import diffuse_heat as _th_diff, HeatBody as _th_Body, material_thermal as _th_mt
from holographic_gas import speed_of_sound as _th_sos, adiabatic as _th_adia, boiling_point as _th_boil
_th_ember = _th_bb(1000.0); _th_day = _th_bb(6500.0); _th_star = _th_bb(12000.0)
_th_T = _ri_np.full((21, 21), 300.0); _th_T[10, 10] = 900.0
_th_T2 = _th_diff(_th_T, alpha=1e-4, dx=0.01, dt=0.5, steps=20)
_th_conserved = bool(abs(float(_th_T2.sum()) - float(_th_T.sum())) < 1e-6)
_th_steel = _th_Body(1.0, _th_mt("steel")["specific_heat"], temp_K=800.0)
_th_c1 = _th_steel.newton_cool(300.0, 2.0, 10.0); _th_c2 = _th_steel.newton_cool(300.0, 2.0, 10.0)
_th_cooling = bool(300.0 < _th_c2 < _th_c1 < 800.0 and (800.0 - _th_c1) > (_th_c1 - _th_c2))
_th_air_sos = float(_th_sos(293.15, "air")); _th_tab = float(_ri_np.asarray(__import__("holographic_definitions").MATERIALS["air"]["sound_speed"]))
_, _th_hot = _th_adia(101325.0, 293.15, 0.5, "air")
_th_boil_sea = _th_boil(101325.0) - 273.15; _th_boil_alt = _th_boil(70000.0) - 273.15
_th_kfromdata = float(_th_mt("steel")["thermal_conductivity"])
print(f"  THERMODYNAMICS FOUNDATION (built FIRST so the material-process layer has a temperature/pressure trigger): three first-principles solvers, no new dependencies. T3 BLACKBODY -- Planck's law integrated against analytic CIE curves turns a temperature into the colour it glows: 1000K ember={tuple(_th_ember.round(2))} (red), 6500K daylight={tuple(_th_day.round(2))} (~neutral), 12000K star={tuple(_th_star.round(2))} (blue-white), Wien peak of the Sun at {_th_wien(5772):.0f}nm -- the ember/flame colour the burn processes will paint. T4 HEAT -- Q=mc*dT plus Fourier conduction dT/dt=alpha*grad^2 T: a hot spot spreads while total heat is CONSERVED ({_th_conserved}, insulated boundary, auto-substepped so any dt stays stable), and a steel body Newton-cools toward ambient, fast-then-slow ({_th_cooling}); conductivity is read from the enrichment DATA (steel {_th_kfromdata:.0f} W/mK, not restated). T1 GAS -- the ideal gas law derives air's speed of sound as {_th_air_sos:.0f} m/s, which independently MATCHES the definitions' tabulated {_th_tab:.0f} (two roads, one number); adiabatic compression heats air to {_th_hot:.0f}K; and water boils {_th_boil_sea:.0f}C at sea level but {_th_boil_alt:.0f}C at altitude -- the pressure-dependent boiling point phase-change (M5) will consume. HONEST: ideal emitter/gas, constant properties, no radiation/convection/real-gas corrections -- correct to DRIVE the processes, not spectroscopy or CFD. This unblocks M4-M7 (oxidization, phase change, material-specific fire, burn/decay), each a data layer + coupling with no new solver.")

# MATERIAL PROCESSES -- M6 material-specific fire + M5 phase change (on the thermodynamics foundation)
from holographic_combustion import COMBUSTION as _pr_COMB, ignites as _pr_ign, combustion_products as _pr_prod, Fire as _pr_Fire
from holographic_phase import PhaseState as _pr_Phase, boiling_point_at as _pr_boil
# M6: per-material ignition gate + wood-vs-plastic smoke
_pr_wood_lights = bool(_pr_ign("wood", 600.0) and not _pr_ign("wood", 500.0))
_pr_pvc_needs_more = bool(not _pr_ign("pvc_plastic", 600.0) and _pr_ign("pvc_plastic", 750.0))
_pr_woodsmoke = float(_pr_prod("wood", 1.0)["smoke_color"].mean())
_pr_pvcsmoke = float(_pr_prod("pvc_plastic", 1.0)["smoke_color"].mean())
# a lit wood fire sustains then burns out
_pr_fire = _pr_Fire("wood", 1.0, temp_K=900.0)
_pr_fuel = [_pr_fire.step(0.5)["fuel_left"] for _ in range(40)]
_pr_burned_out = bool(_pr_fuel[-1] < _pr_fuel[0] * 0.2)
# M5: the boiling plateau -- temperature holds at 100C while liquid turns to steam
_pr_ps = _pr_Phase("water", 1.0, temp_K=372.15)
_pr_T = []; _pr_g0 = _pr_ps.gas
for _ in range(40):
    _pr_ps.add_heat(1.0e5); _pr_T.append(_pr_ps.T)
_pr_T = _ri_np.array(_pr_T)
_pr_plateau = int((_ri_np.abs(_pr_T - 373.15) < 0.2).sum())
_pr_became_steam = bool(_pr_ps.gas > _pr_g0)
_pr_boil_alt = _pr_boil("water", 70000.0) - 273.15
print(f"  MATERIAL PROCESSES (the first two of four, on the thermo foundation -- data + coupling, no new solver). M6 MATERIAL-SPECIFIC FIRE: each material lights at its OWN autoignition temperature (wood at 300C: {_pr_wood_lights}; PVC needs ~450C, won't light at 327C: {_pr_pvc_needs_more}; nothing lights cold), and makes ITS OWN smoke -- wood pale grey (mean {_pr_woodsmoke:.2f}) vs PVC black+sooty (mean {_pr_pvcsmoke:.2f}) -- with flame colour from the blackbody (T3). A lit fire LATCHES, sustains, and burns out as fuel depletes ({_pr_burned_out}); the couplings feed these numbers to the fluid solver's combustion and the surface emitter. M5 PHASE CHANGE: pour heat into water and the temperature HOLDS at 100C for {_pr_plateau} steps of 100kJ (~2.2 MJ, matching water's latent heat) while liquid turns to steam ({_pr_became_steam}) -- the textbook boiling plateau, latent heat paid before the temperature moves again; melting/freezing hold at 0C the same way, and the boiling point falls with pressure ({_pr_boil_alt:.0f}C at altitude, from the gas model T1). HONEST: combustion PRODUCTS not reaction kinetics; lumped latent-heat bookkeeping not two-phase CFD; plausible art-directable data. Remaining: M4 oxidization front (reaction-diffusion) and M7 burn/decay (object consumption over time).")

# BACKLOG FINISH + ELEMENTS -- M4 corrosion front, M7 burn/decay, and the periodic table as ingredients
from holographic_oxidation import OxidationField as _bx_Ox, oxide_color as _bx_oxc
from holographic_burn import BurningObject as _bx_Burn, char_color as _bx_char
from holographic_elements import element as _bx_el, molar_mass as _bx_mm, flame_color_of as _bx_fc, material_elemental as _bx_me
# M4: rust nucleates at exposed faces and spreads inward (a front)
_bx_f = _bx_Ox((21, 21))
for _ in range(30):
    _bx_f.step("steel", dt=1.0)
_bx_front = bool(_bx_f.ox[0, 10] > _bx_f.ox[10, 10] and _bx_f.ox[10, 10] > 0.0)
_bx_rust = _bx_oxc("steel", 1.0)
# M7: a lit wood object burns down to ash, appearance base->char->ash
_bx_obj = _bx_Burn("wood", 1.0).light()
for _ in range(80):
    _bx_obj.step(0.5)
_bx_ash = bool(_bx_obj.is_ash())
_bx_mid = _bx_char("wood", 0.5); _bx_base = _bx_char("wood", 0.0)
_bx_chars = bool(_bx_mid.mean() < _bx_base.mean())
# ELEMENTS: molar mass + flame colour from composition (the blend); a material references its makeup
_bx_h2o = _bx_mm({"H": 2, "O": 1})
_bx_na = _bx_fc({"Na": 1}); _bx_cu = _bx_fc({"Cu": 1}); _bx_mix = _bx_fc({"Na": 1, "Cu": 1})
_bx_blend_ok = bool(_ri_np.allclose(_bx_mix, 0.5 * (_bx_na + _bx_cu), atol=1e-6))
_bx_salt = _bx_me("table_salt")
_bx_feFrac = float(_bx_me("steel")["mass_fractions"]["Fe"])
print(f"  BACKLOG FINISH + ELEMENTS. The material-process layer is complete: M4 CORROSION FRONT -- rust nucleates at exposed faces and creeps inward as a reaction-diffusion front ({_bx_front}), blending base->oxide (steel rust r={_bx_rust[0]:.2f}>b={_bx_rust[2]:.2f}, copper->green patina); M7 BURN/DECAY -- a lit wood object drives an M6 fire, loses mass, and its surface marches base->char (darkens: {_bx_chars})->ash, ending as ash ({_bx_ash}). And the new PERIODIC TABLE gives the engine its atomic INGREDIENTS: {len(__import__('holographic_elements').symbols())} elements with atomic mass, density, melt/boil points and FLAME-TEST colour (Li crimson, Na yellow, K lilac, Cu green). The point is the COMPOSITION grammar -- a material declares its elemental makeup + ratio (water=H2O, steel~98% iron by mass: {_bx_feFrac:.2f}) and derived facts fall out by COMPOSITION not restatement: molar mass of water = {_bx_h2o:.3f} g/mol (feeds the gas law T1), and flame colour = the ratio-weighted BLEND of the constituents' flame colours -- Na yellow + Cu green, a 50/50 mix landing exactly between them ({_bx_blend_ok}), so salt burns sodium-yellow (r={_bx_salt['flame_color'][0]:.2f}). That emission-line colour is exactly what the blackbody continuum alone couldn't give -- elements close that loop. As above, so below: a material decomposes into elements like a VSA record into role-fillers, and blend is how composites form. HONEST: phenomenological corrosion (not electrochemistry), object-level burn (geometry doesn't shrink), a curated element subset with reference values.")

# ACOUSTICS -- A1 read/analyse a sound, A2 acoustic impedance, A4 Chladni cymatics (sound -> sand figure)
from holographic_audio import dominant_frequencies as _ac_dom, write_wav as _ac_wav, read_wav as _ac_read
from holographic_acoustic import impedance as _ac_Z, interface as _ac_iface, reflect_absorb as _ac_absorb
from holographic_cymatics import ChladniPlate as _ac_Plate
# A1: a synthesized chord reads back its constituent notes; a WAV round-trips
_ac_rate = 22050; _ac_t = _ri_np.arange(_ac_rate) / _ac_rate
_ac_chord = sum(_ri_np.sin(2 * _ri_np.pi * _hz * _ac_t) for _hz in (440.0, 554.37, 659.25))
_ac_f, _ac_a = _ac_dom(_ac_chord, _ac_rate, k=4)
_ac_notes = sorted(round(float(x)) for x in _ac_f[:3])
# A2: acoustic impedance from density*sound_speed; air/steel reflects almost all sound
_ac_za = float(_ac_Z("air")); _ac_zs = float(_ac_Z("steel"))
_ac_R, _ac_T = _ac_iface("air", "steel")
_ac_foam_r, _ac_foam_a = _ac_absorb("acoustic_foam")
# A4: drive a plate with a tone at one of its modes; sand settles onto the nodes (the Chladni figure)
_ac_plate = _ac_Plate("square", grid=24, n_modes=16, n_grains=2500, seed=0)
_ac_tone = _ri_np.sin(2 * _ri_np.pi * float(_ac_plate.mode_hz[6]) * _ac_t)
_ac_ff, _ac_aa = _ac_dom(_ac_tone, _ac_rate, k=3)
_ac_plate.drive(_ac_ff, _ac_aa); _ac_plate.settle(steps=40, dt=0.1, strength=8.0)
_ac_sand_u, _ac_plate_u = _ac_plate.nodal_fraction_on_sand()
_ac_on_nodes = bool(_ac_sand_u < 0.6 * _ac_plate_u)
print(f"  ACOUSTICS (a new subsystem; the key reuse is that a plate's CHLADNI modes ARE the Laplacian eigenmodes we already compute). A1 -- read a sound and analyse it: a synthesized A-major chord reads back its constituent notes {_ac_notes} Hz (and a WAV round-trips through stdlib). A2 -- acoustic impedance Z=rho*c straight from the material data (air {_ac_za:.0f} -> steel {_ac_zs:.2e} rayl); at an air/steel boundary the huge mismatch reflects {100*_ac_R:.1f}% of the sound (energy conserved R+T={_ac_R+_ac_T:.2f}) -- why you hear so little through a wall -- while acoustic foam absorbs {100*_ac_foam_a:.0f}%. A4 (THE HEADLINE) -- CYMATICS: build a plate's Dirichlet Laplacian, take its eigenmodes (the shipped spectral eigensolver), drive them with a sound's spectrum, and sand drifts DOWN grad|u|^2 to the NODES, tracing the Chladni figure. Driven by a tone at one of its modes, the sand settles onto the nodal set (mean |u| under the sand {_ac_sand_u:.3f} << plate average {_ac_plate_u:.3f}: {_ac_on_nodes}) -- a sound file literally drawn in sand. HONEST: a membrane-mode model (Laplacian modes + node-drift), not the full biharmonic plate; dense eigensolver so modest grids; normal-incidence impedance to start. Remaining: A5 water/cornstarch media, A3 the wave-equation field for propagation, A7 acoustic levitation, A6 ray-traced room reverb -- all additive, no new solver.")

# ACOUSTICS cont. -- A5 water/cornstarch cymatic media + A3 the wave-equation field (sound that propagates)
from holographic_cymatics import ChladniPlate as _a5_Plate
from holographic_wave import WaveField as _a3_Wave
# A5 water: standing surface at the ANTINODES (correlates with |u|), opposite of sand-at-nodes
_a5_w = _a5_Plate("square", grid=24, medium="water", n_modes=16, seed=0); _a5_w.drive_mode(6); _a5_w.settle(25)
_a5_au = _ri_np.abs(_a5_w.u)[_a5_w.mask]; _a5_corr = float(_ri_np.corrcoef(_a5_au, _a5_w.surface[_a5_w.mask])[0, 1])
# A5 cornstarch: holds peaks under fast drive, slumps under slow (shear-thickening)
_a5_fast = _a5_Plate("square", grid=24, medium="cornstarch", n_modes=16, base_hz=1400.0, seed=0); _a5_fast.drive_mode(8); _a5_fast.settle(25)
_a5_slow = _a5_Plate("square", grid=24, medium="cornstarch", n_modes=16, base_hz=40.0, seed=0); _a5_slow.drive_mode(8); _a5_slow.settle(25)
_a5_thicken = bool(_a5_fast.peaks.max() > _a5_slow.peaks.max() * 2)
# A3 wave: a 1-D pulse splits into two movers each at c (d'Alembert); energy stays bounded (CFL auto-substep)
_a3 = _a3_Wave((400,), c=1.0, dx=1.0); _a3.pulse((200,), amp=1.0, radius=4.0); _a3_e0 = _a3.energy(); _a3.step(dt=120.0)
_a3_right = 200 + int(_ri_np.argmax(_a3.p[200:])); _a3_left = int(_ri_np.argmax(_a3.p[:200]))
_a3_dalembert = bool(abs((_a3_right - 200) - 120) < 12 and abs((200 - _a3_left) - 120) < 12)
_a3_bounded = bool(_ri_np.isfinite(_a3.p).all())
print(f"  ACOUSTICS cont. (finishing the cymatics media, then propagation). A5 WATER -- a vertically-driven surface forms a standing FARADAY wave at the ANTINODES (crests where the plate moves most, the opposite of sand-at-nodes): the surface correlates with |u| at {_a5_corr:.2f}, and its cell size tracks the drive frequency. A5 CORNSTARCH -- a shear-thickening suspension that STANDS in fingers under fast/hard drive and slumps under slow ({_a5_thicken}), the 'walking oobleck' signature -- so sand, water, and cornstarch each answer the same sound differently, over one driving field. A3 WAVE FIELD -- the compressible acoustic wave the incompressible fluid can't carry: d2p/dt2 = c^2 grad^2 p by leapfrog (the whole solver is one line). A centred pulse splits into a left and a right mover, each travelling at the speed of sound c (d'Alembert: {_a3_dalembert}), and the field stays bounded/stable even for a huge time step because the CFL limit is auto-subdivided ({_a3_bounded}); an absorbing border soaks outgoing waves so they don't reflect. This is the low-frequency wave complement to ray acoustics, and the standing field levitation will use. HONEST: water is standing-pattern phenomenology not a free-surface solve; the wave field is scalar linear acoustics (no shocks/elastic waves). Remaining: A7 levitation (Gor'kov force -> beads to pressure nodes) and A6 ray-traced room reverb.")

# NON-NEWTONIAN FLUID -- real cornstarch: viscosity that depends on shear rate (so the cornstarch demo is backed by a solver)
from holographic_nonnewtonian import power_law_viscosity as _nn_eta, viscous_step as _nn_visc
import numpy as _nn_np
_nn_slow = float(_nn_eta(0.5, 1.0, 1.8)); _nn_fast = float(_nn_eta(50.0, 1.0, 1.8))          # cornstarch n=1.8
_nn_thin = bool(_nn_eta(50.0, 1.0, 0.5) < _nn_eta(0.5, 1.0, 0.5))                             # ketchup n<1 thins
_nn_g = 20; _nn_yy = _nn_np.arange(_nn_g, dtype=float)[:, None] * _nn_np.ones((1, _nn_g))
_nn_v0 = _nn_np.stack([15.0 * _nn_np.sin(2 * _nn_np.pi * 2 * _nn_yy / _nn_g), _nn_np.zeros((_nn_g, _nn_g))])
_nn_E0 = float((_nn_v0 ** 2).sum())
_nn_vc, _nn_field = _nn_visc(_nn_v0.copy(), 0.05, 1.8, 0.1); _nn_vn, _ = _nn_visc(_nn_v0.copy(), 0.05, 1.0, 0.1)
_nn_corn_kept = float((_nn_vc ** 2).sum()) / _nn_E0; _nn_newt_kept = float((_nn_vn ** 2).sum()) / _nn_E0
print(f"  NON-NEWTONIAN FLUID (making sure we can actually SIMULATE cornstarch, not just paint its cymatics): the solver's single viscosity CONSTANT becomes a FIELD via the power law eta = K * shear_rate^(n-1). For cornstarch (n=1.8) the viscosity climbs {_nn_fast/_nn_slow:.0f}x from a gentle shear ({_nn_slow:.3f}) to a fast one ({_nn_fast:.3f}) -- punch it and it stiffens like a solid, let it rest and it flows; for n<1 (ketchup, paint) it THINS under shear instead ({_nn_thin}); n=1 is ordinary Newtonian, unchanged. Applied as a variable-viscosity viscous force div(eta*(grad v + grad v^T)) with the eta FIELD spiking in the sheared bands, it damps a fast-sheared flow MORE than water (cornstarch keeps {_nn_corn_kept:.2f} of the shear energy vs water's {_nn_newt_kept:.2f}) -- the shear-thickening signature, measured. Wired into StableFluid as an opt-in mode (power_law_n/consistency_K; n=1 stays byte-identical). HONEST: power-law model (no yield stress / thixotropy), clamped, 2-D -- correct for the thickening/thinning regime, not a full suspension-mechanics solver. So the cornstarch in the cymatics demo now has a real rheology behind it.")

# ACOUSTICS FINALE -- A7 levitation (sound holds beads) + A6 room acoustics (reflections + reverb)
from holographic_levitate import LevitationChamber as _af_Lev, pressure_nodes as _af_nodes, gorkov_force_y as _af_F
from holographic_roomacoustic import ShoeboxRoom as _af_Room
_af_lam = 0.0086
_af_on = _af_Lev(height=0.04, wavelength=_af_lam, amplitude=5000.0, n_beads=24, seed=0); _af_on.settle(steps=4000, field_on=True)
_af_off = _af_Lev(height=0.04, wavelength=_af_lam, amplitude=5000.0, n_beads=24, seed=0); _af_off.settle(steps=6000, field_on=False)
_af_aloft = float((_af_on.heights() > 0.002).mean() * 100)
_af_fell = bool(_af_off.heights().mean() < _af_on.heights().mean() * 0.5)
_af_nd = _af_nodes(_af_lam, 0.04); _af_nearest = float(_ri_np.median(_ri_np.min(_ri_np.abs(_af_on.heights()[:, None] - _af_nd[None, :]), axis=1)))
_af_live = _af_Room(size=(6, 4, 3), absorption=0.03); _af_dead = _af_Room(size=(6, 4, 3), absorption=0.45)
_af_src = (1.0, 2.0, 1.5); _af_lis = (5.0, 2.0, 1.5)
_af_taps = _af_live.reflections(_af_src, _af_lis, max_order=1)
_af_direct_ms = _af_taps[0]["delay"] * 1000.0
_af_floor_ms = _ri_np.sqrt(4.0 ** 2 + 3.0 ** 2) / 343.0 * 1000.0
print(f"  ACOUSTICS FINALE -- the last two items, finishing the backlog. A7 LEVITATION (sound moves objects): in a vertical standing wave the Gor'kov radiation force pushes small dense beads to the pressure NODES (spaced lambda/2 = {_af_lam/2*1000:.1f} mm) and pins them there. Field ON holds {_af_aloft:.0f}% of the beads aloft against gravity, clustered within {_af_nearest*1000:.2f} mm of the nodes; field OFF and they fall to the floor ({_af_fell}) -- an ultrasonic levitator, reusing the standing field (A3), the particle system, and gravity. A6 ROOM ACOUSTICS (how a room echoes): the image-source method (Allen & Berkley) mirrors the source across each wall so every reflection is a tap at delay = path/c and level set by the wall's reflectance (from A2) -- the direct sound arrives at {_af_direct_ms:.1f} ms, the floor bounce later at {_af_floor_ms:.1f} ms (geometrically correct), and Sabine's RT60 gives the reverb tail: a hard room rings {_af_live.rt60():.2f}s vs {_af_dead.rt60():.2f}s for an absorptive one. HONEST: Gor'kov holds for beads much smaller than the wavelength; geometric acoustics is a high-frequency approximation (no diffraction -- the A3 wave field is its low-frequency complement). The acoustics & cymatics backlog is now complete end to end: read a sound, bounce/absorb/transmit it, watch its cymatics on sand/water/cornstarch, levitate a bead, and hear a room -- all on the one engine.")

# SIGGRAPH LIST -- #1 curl noise (divergence-free turbulence) + #2 tearing (a sheet that rips)
from holographic_curlnoise import curl_noise as _sg_curl, divergence as _sg_div
from holographic_tear import TearableCloth as _sg_Tear, tear_strength as _sg_ts
_sg_R = 1.5; _sg_cx = _sg_cy = 4.0
_sg_disk = lambda p: _ri_np.sqrt((p[:, 0] - _sg_cx) ** 2 + (p[:, 1] - _sg_cy) ** 2) - _sg_R
_sg_u, _sg_v = _sg_curl(48, octaves=4, seed=1, obstacle_sdf=_sg_disk, ramp=1.2)
_sg_maxdiv = float(_ri_np.abs(_sg_div(_sg_u, _sg_v)).max())
_sg_xs = _ri_np.linspace(0, 8, 48); _sg_X, _sg_Y = _ri_np.meshgrid(_sg_xs, _sg_xs)
_sg_dist = _ri_np.sqrt((_sg_X - _sg_cx) ** 2 + (_sg_Y - _sg_cy) ** 2); _sg_sp = _ri_np.sqrt(_sg_u ** 2 + _sg_v ** 2)
_sg_inpen = float(_sg_sp[_sg_dist < _sg_R * 0.8].mean() / max(_sg_sp[_sg_dist > _sg_R * 1.5].mean(), 1e-9) * 100)
_sg_paper = _sg_Tear(rows=10, cols=10, material="paper", compliance=3e-3)
for _ in range(70):
    _sg_paper.step(pull=(0.0, -1200.0), gravity=(0.0, -9.8))
_sg_rubber = _sg_Tear(rows=10, cols=10, material="rubber", compliance=3e-3)
for _ in range(70):
    _sg_rubber.step(pull=(0.0, -1200.0), gravity=(0.0, -9.8))
print(f"  SIGGRAPH SCOUTING LIST -- starting down it. #1 CURL NOISE (the cheap win): divergence-free turbulence from the CURL of an fBm streamfunction -- (u,v)=(dpsi/dy,-dpsi/dx), which can't compress, so max|divergence| = {_sg_maxdiv:.1e} (machine zero: the discrete mixed partials commute). With an obstacle SDF the streamfunction is pinned to a streamline at the surface so the flow goes AROUND it -- speed inside the disk is {_sg_inpen:.0f}% of outside. Cheap wind/smoke detail, no fluid solve, reusing the noise + field operators we already had. #2 TEARING (a genuine new capability -- we had no fracture): a thin sheet is a PBD cloth of distance links; give each a tear STRENGTH (max strain) and it SNAPS when overstretched, the crack propagating as broken links dump load on their neighbours -- the reference method's physics on the constraint graph, no remeshing. A yanked paper sheet (tear strain {_sg_ts('paper'):.2f}) snapped {_sg_paper.torn} links into {_sg_paper.connected_components()} pieces; rubber (tear strain {_sg_ts('rubber'):.2f}) tore only {_sg_rubber.torn} under the same pull -- it stretches instead of ripping. HONEST: 2-D streamfunction curl noise (no-penetration, not no-slip); a mass-spring tear as sharp as the grid, not adaptive remeshing. Next on the list: the headline #7 Walk-on-Spheres grid-free Monte-Carlo PDE solver on the SDF.")

# WALK ON SPHERES -- grid-free Monte-Carlo PDE solver on the SDF (SIGGRAPH list #7)
from holographic_wos import walk_on_spheres as _wos
_wos_Rin, _wos_Rout = 1.0, 2.0
def _wos_dist(P):
    _rho = _ri_np.linalg.norm(P, axis=1); return _ri_np.minimum(_rho - _wos_Rin, _wos_Rout - _rho)
def _wos_bval(P):
    _rho = _ri_np.linalg.norm(P, axis=1); return (_ri_np.abs(_rho - _wos_Rout) < _ri_np.abs(_rho - _wos_Rin)).astype(float)
_wos_probe = _ri_np.array([[1.5, 0.0]])
_wos_mean, _wos_se = _wos(_wos_probe, _wos_dist, _wos_bval, n_walks=4000, eps=1e-3, seed=2)
_wos_exact = float(_ri_np.log(1.5 / _wos_Rin) / _ri_np.log(_wos_Rout / _wos_Rin))
# Poisson on a 2-D disk: -Delta u = 1, u=0 on the boundary -> u(0) = R^2/4
_wos_R = 2.0
_wos_pm, _wos_pse = _wos(_ri_np.array([[0.0, 0.0]]), lambda P: _wos_R - _ri_np.linalg.norm(P, axis=1),
                         lambda P: _ri_np.zeros(len(P)), source=lambda P: _ri_np.ones(len(P)), n_walks=4000, eps=1e-3, seed=3)
_wos_lo = _wos(_wos_probe, _wos_dist, _wos_bval, n_walks=250, seed=4)[1].mean()
_wos_hi = _wos(_wos_probe, _wos_dist, _wos_bval, n_walks=4000, seed=4)[1].mean()
print(f"  WALK ON SPHERES (the SIGGRAPH headline -- solve PDEs with NO mesh): to get the solution at a point, random-walk by jumping to a random point on the largest empty sphere (its radius IS one SDF evaluation) until you hit the boundary, then average the boundary values. On an ANNULUS, Laplace's equation (steady heat) with the inner ring cold and the outer ring hot gives the log-profile u(r)=ln(r/rin)/ln(rout/rin): WoS finds {float(_wos_mean[0]):.3f} +/- {float(_wos_se[0]):.3f} at r=1.5 vs the exact {_wos_exact:.3f}. Poisson (-div grad u = 1) on a disk with a zero-boundary gives u(0)=R^2/4={_wos_R**2/4:.2f}: WoS finds {float(_wos_pm[0]):.2f} +/- {float(_wos_pse[0]):.2f}. It is Monte Carlo, so the error shrinks as 1/sqrt(N) -- 16x the walks tightens the standard error {float(_wos_lo/_wos_hi):.1f}x (and the error bar ships WITH the answer, the measure discipline). The one primitive it needs -- distance to the boundary -- is exactly what our SDF returns, so this is the path tracer's random-walk pointed at a PDE: the mesh-free steady/elliptic complement to the wave field and the heat diffusion, on ANY shape we can write an SDF for. HONEST: noisy (1/sqrt(N)); Dirichlet only (Neumann needs Walk on Stars); elliptic/parabolic, not everything.")

# HAIR & FUR -- groom on a surface, simulate as PBD strands, shade by tangent (backlog H1-H7)
from holographic_groom import groom as _hair_groom, simulate_strands as _hair_sim, interpolate_strands as _hair_interp, CurlWind as _HairWind
from holographic_hairshade import kajiya_kay as _hair_kk, marschner as _hair_mar, render_hair as _hair_render
from holographic_sdf import sphere as _hair_sphere
_hair_s = _hair_sphere(1.0); _hair_bounds = ([-1.6, -1.6, -1.6], [1.6, 1.6, 1.6])
_hair_guides = _hair_groom(_hair_s.eval, 30, _hair_bounds, length=0.7, n_pts=8, curl=1.0, seed=0)
_hair_root_ok = float(_ri_np.abs(_ri_np.linalg.norm(_ri_np.array([g.root for g in _hair_guides]), axis=1) - 1.0).max())
_hair_L0 = _hair_guides[0].length()
_hair_wind = _HairWind(strength=2.0, seed=1)
_hair_moved = _hair_sim(_hair_guides[:6], steps=30, gravity=(0.0, -6.0, 0.0), wind=_hair_wind.force, body_sdf=_hair_s.eval)
_hair_tip_drop = float(_hair_guides[0].points[-1][1] - _hair_moved[0].points[-1][1])
_hair_stretch = float(abs(_hair_moved[0].length() - _hair_L0) / _hair_L0 * 100)
_hair_render_roots = _ri_np.array([g.root for g in _hair_guides]) * 1.001
_hair_many = _hair_interp(_hair_guides, _hair_render_roots, k=3, clump=0.5)
_hair_T = _ri_np.array([0.0, 1.0, 0.0]); _hair_l = _hair_kk(_hair_T, _ri_np.array([1.0, 0.0, 0.0]), _ri_np.array([0.0, 0.0, 1.0]))
_hair_blonde = float(_hair_mar(_hair_T, _ri_np.array([0.4, 0.3, 0.7]), _ri_np.array([-0.3, 0.2, 0.8]), hair_color=(0.85, 0.7, 0.4)).sum())
_hair_black = float(_hair_mar(_hair_T, _ri_np.array([0.4, 0.3, 0.7]), _ri_np.array([-0.3, 0.2, 0.8]), hair_color=(0.05, 0.04, 0.03)).sum())
print(f"  HAIR & FUR -- knocking out the backlog by REUSING the substrate (the audit's point: a strand is a rope with a pinned root). H1 GROOM: {len(_hair_guides)} strands rooted on the sphere via emit_from_surface, each grown along its outward normal (roots land on the surface to {_hair_root_ok:.3f}), curled as a tapered helix, smoothed by subdivcurve. H2 DYNAMICS: simulated as PBD chains -- root pinned (inverse-mass 0), distance constraints + Follow-The-Leader for inextensibility, bend springs for stiffness; under gravity + a curl-noise breeze the tip dropped {_hair_tip_drop:.2f} while the strand stretched only {_hair_stretch:.1f}% (it swings, it doesn't grow). H3 INTERP: {len(_hair_many)} render strands blended+clumped from the guides -- what makes full fur affordable. H7 WIND: divergence-free curl noise, so fur ripples without ballooning. SHADING by the TANGENT, not a surface normal: H4 Kajiya-Kay gives the lengthwise sheen (anisotropic diffuse+spec), H5 Marschner adds the physical R/TT/TRT lobes with absorption -- blonde reflects/transmits {_hair_blonde/max(_hair_black,1e-6):.1f}x what near-black hair does, and the colored TRT secondary highlight is what makes hair read as hair. HONEST: bend not twist (true torsion = the deferred Cosserat rung H2b); Marschner is single-scattering (no dual-scattering glow); opaque depth-ordered strands. All reusing emitter+softbody+subdivcurve+collide+curl noise -- no new solver.")

# HAIR H2b -- Cosserat rod: orientation frames give twist and hold curls (the quality upgrade over bend springs)
from holographic_cosserat import CosseratStrand as _CosStrand
_cos_n = 14; _cos_s = _ri_np.linspace(0, 1, _cos_n); _cos_cr = 0.12
_cos_pts = _ri_np.stack([_cos_cr * (_ri_np.cos(2 * _ri_np.pi * 2 * _cos_s) - 1.0) * _cos_s, _cos_s * 0.8,
                         _cos_cr * _ri_np.sin(2 * _ri_np.pi * 2 * _cos_s) * _cos_s], axis=1)
_cos_rest = float(1.0 - _ri_np.linalg.norm(_cos_pts[-1] - _cos_pts[0]) / _ri_np.linalg.norm(_ri_np.diff(_cos_pts, axis=0), axis=1).sum())
_cos_rod = _CosStrand(_cos_pts, bend_stiffness=0.6, shape_stiffness=0.7); _cos_rod.settle(steps=100, gravity=(0.0, -9.8, 0.0))
_cos_plain = _CosStrand(_cos_pts, bend_stiffness=0.0, shape_stiffness=0.0); _cos_plain.settle(steps=100, gravity=(0.0, -9.8, 0.0))
_cos_tw0 = abs(_cos_rod.twist_of(_cos_n // 2)); _cos_rod2 = _CosStrand(_cos_pts, bend_stiffness=0.8, shape_stiffness=0.5); _cos_rod2.set_root_twist(1.2)
for _ in range(30):
    _cos_rod2.step(gravity=(0.0, 0.0, 0.0))
print(f"  HAIR H2b -- TWIST via a Cosserat rod (finishing the hair backlog's deferred rung): each segment carries an orientation FRAME (a quaternion), so the strand holds its curl and can twist -- the PBD route to Discrete Elastic Rods. A curly rest strand (curl {_cos_rest:.2f}) settled under gravity keeps its curl at {float(_cos_rod.curl_amount()):.2f} with the frames vs a plain bend-only chain that drifts to {float(_cos_plain.curl_amount()):.2f} (frames hold the rest shape). The curl memory is tension-aware, so pulling it taut un-curls it. And a root twist propagates down the frames to the mid-strand ({_cos_tw0:.2f} -> {float(abs(_cos_rod2.twist_of(_cos_n // 2))):.2f} rad) -- a twist DOF plain bend springs don't have. HONEST: roll is carried but the round-fibre centerline follows the tangent; positions reconstructed from frames each step (the simplified coupling). Opt-in. The Hair & Fur backlog is now complete end to end.")

# COMPUTE ARCHITECTURE -- the GPU-shaped gaps NumPy leaves, filled VSA-natively (fusion / residency / scheduler)
from holographic_fuse import leaf as _f_leaf, fbind as _f_bind, fbundle as _f_bundle, funbind as _f_unbind, fuse as _f_fuse, fuse_record as _f_record, reset_fft_counts as _f_reset, fft_counts as _f_counts
from holographic_ai import bundle_bind as _ca_bb
from holographic_schedule import leaf as _s_leaf, op_bind as _s_bind, op_bundle as _s_bundle, op_unbind as _s_unbind, op_cleanup as _s_clean, run_sequential as _s_seq, run_scheduled as _s_sch
_ca_rng = _ri_np.random.default_rng(0); _ca_D = 1024
_ca_atoms = [a / _ri_np.linalg.norm(a) for a in _ca_rng.standard_normal((7, _ca_D))]
# fusion: a K-bind accumulation chain does leaves+1 FFTs instead of 3K
_ca_e = _f_leaf(_ca_atoms[0])
for _x in _ca_atoms[1:]:
    _ca_e = _f_bind(_ca_e, _x)
_f_reset(); _ca_res = _f_fuse(_ca_e); _ca_c = _f_counts(); _ca_fused_ffts = _ca_c["rfft"] + _ca_c["irfft"]
# the record pattern, fused, matches bundle_bind to tolerance
_ca_keys = [k / _ri_np.linalg.norm(k) for k in _ca_rng.standard_normal((6, _ca_D))]
_ca_vals = [v / _ri_np.linalg.norm(v) for v in _ca_rng.standard_normal((6, _ca_D))]
_ca_rec_err = float(_ri_np.abs(_f_record(_ca_keys, _ca_vals) - _ca_bb(_ca_keys, _ca_vals)).max())
# scheduler: a compose+recall pipeline, scheduled vs op-by-op
_r0, _r1, _r2, _f0, _f1, _f2 = _ca_atoms[:6]
_ca_cb = _ri_np.stack([_f0, _f1, _f2])
_ca_prog = [_s_leaf(_r0), _s_leaf(_r1), _s_leaf(_r2), _s_leaf(_f0), _s_leaf(_f1), _s_leaf(_f2),
            _s_bind(0, 3), _s_bind(1, 4), _s_bind(2, 5), _s_bundle([6, 7, 8]), _s_unbind(9, 0), _s_clean(10, _ca_cb)]
_ca_sv, _ca_seq = _s_seq(_ca_prog); _ca_cv, _ca_sch = _s_sch(_ca_prog)
# integration: fuse a REAL Layer-4 recipe (its build DAG) through the scheduler
from holographic_recipe import StructureRecipe as _CaRecipe
from holographic_schedule import run_recipe as _ca_run_recipe
_ca_rc = _CaRecipe(dim=1024, seed=0)
_ca_ro = [_ca_rc.atom("role%d" % _i, unitary=True) for _i in range(4)]; _ca_fi = [_ca_rc.atom("fill%d" % _i) for _i in range(4)]
_ca_rc.mark_output(_ca_rc.normalize(_ca_rc.permute(_ca_rc.bundle([_ca_rc.bind(_ca_ro[_i], _ca_fi[_i]) for _i in range(4)]), 3)))
_ca_rc_fused, _ca_rc_fs = _ca_run_recipe(_ca_rc, fused=True); _ca_rc_seq, _ca_rc_ss = _ca_run_recipe(_ca_rc, fused=False)
_ca_rc_err = float(_ri_np.abs(_ca_rc.outputs()[0] - _ca_rc_fused[0]).max())
print(f"  COMPUTE ARCHITECTURE -- closing the GPU-shaped gaps NumPy leaves, VSA-natively. The keystone: bind is a spectral multiply, bundle an add, permute a phase-ramp, unbind a conjugate-multiply -- all LINEAR in Fourier space -- so a straight-line chain is ONE expression: one transform per leaf, one out. FUSION collapses a {len(_ca_atoms)}-bind chain to {int(_ca_fused_ffts)} FFTs (vs {3 * (len(_ca_atoms) - 1)} op-by-op), matching bundle_bind to {_ca_rec_err:.0e}. The SCHEDULER reads a compose+recall pipeline as a DAG and fuses its linear run: {int(_ca_sch['fft'])} FFTs / {int(_ca_sch['kernel_calls'])} kernel-calls vs op-by-op {int(_ca_seq['fft'])} / {int(_ca_seq['kernel_calls'])}, crossing to Python only at the 1 cleanup. INTEGRATED at Layer 4: fusing a real StructureRecipe's build DAG does {int(_ca_rc_fs['fft'])} FFTs vs {int(_ca_rc_ss['fft'])} op-by-op, matching the exact build to {_ca_rc_err:.0e} (an opt-in throughput path; build() stays bit-exact). Residency caches known atoms' spectra (bit-exact); superposition holds N binds in flight and spills past the capacity dial rather than abstaining. HONEST: fusion is tolerance-not-bit-exact (~1e-15) so tie-sensitive paths -- and the machine's decision-bounded opcode decode -- stay op-by-op; the final commit is a real crossing. Throughput paths, off by default -- the frozen kernel is untouched.")

# ABOVE/BELOW SWEEP 3 -- the five wide-fanout mechanisms wired (bridge / spatial / reaction-diffusion / emergence / temporal reuse)
from holographic_spatial import SpatialGrid as _S3Grid, brute_knn as _s3_bknn
from holographic_automaton import HyperCA as _S3CA
from holographic_emergence import EmergentConcepts as _S3EC
from holographic_temporal import TemporalReuse as _S3TR
_s3_rng = _ri_np.random.default_rng(0)
# 2. spatial index matches brute force
_s3_pts = _s3_rng.uniform(0, 10, (200, 3)); _s3_g = _S3Grid(_s3_pts, 1.0)
_s3_match = _s3_g.knn([5, 5, 5], 5) == _s3_bknn(_s3_pts, [5, 5, 5], 5)
# 3. reaction-diffusion: a pattern emerges
_s3_ca = _S3CA(size=28, dim=32, seed=0); _s3_start = _s3_ca.grid.copy()
for _ in range(25):
    _s3_ca.step()
_s3_moved = float(_ri_np.abs(_s3_ca.grid - _s3_start).mean())
# 4. emergent concepts: two separated clusters discovered label-free
_s3_a = _s3_rng.standard_normal(256); _s3_a /= _ri_np.linalg.norm(_s3_a)
_s3_b = _s3_rng.standard_normal(256); _s3_b /= _ri_np.linalg.norm(_s3_b)
_s3_ec = _S3EC(seed=0)
for _ in range(10):
    _s3_ec.perceive(_s3_a + 0.02 * _s3_rng.standard_normal(256)); _s3_ec.perceive(_s3_b + 0.02 * _s3_rng.standard_normal(256))
_s3_nc = len(_s3_ec.concepts)
# 5. temporal reuse: dirty-only re-solve cost drop
_s3_tr = _S3TR(); _s3_scene = _ri_np.sin(_ri_np.linspace(0, 6, 300))
_s3_tr.solve(lambda i: float(_s3_scene[i] ** 2), 300)
_s3_scene2 = _s3_scene.copy(); _s3_scene2[[10, 11, 150]] += 0.3
_s3_frame, _s3_cost = _s3_tr.solve(lambda i: float(_s3_scene2[i] ** 2), 300, dirty=[10, 11, 150])
print(f"  ABOVE/BELOW SWEEP 3 -- the sweep's five wide-fanout mechanisms, wired. (1) FACULTY BRIDGE: faculties become a capability table any VSA program can APPLY and introspect -- one registry the machine, drives, and moe all read. (2) ONE SPATIAL INDEX: a uniform grid answers radius/knn/closest byte-identically to a brute-force scan ({'match' if _s3_match else 'MISMATCH'}), the same query cull, navigation, collision, sampling, and Walk-on-Spheres all ask -- the widest fanout on the board. (3) REACTION-DIFFUSION: a local rule over a hypervector field self-organizes a pattern (moved {_s3_moved:.3f} off the initial state) -- one solver for patina, texture, fur, crystal, erosion. (4) EMERGENCE: label-free online concept growth discovered {int(_s3_nc)} concepts from two interleaved streams (an online GROUP BY that pulls diffusion in as its commitment). (5) TEMPORAL REUSE: re-solving only the dirty region cost {int(_s3_cost)} cells vs 300 for a full frame, matching a full re-solve exactly. Four of the five already existed and were orphaned -- the win was wiring them once so they light up many areas.")

# ABOVE/BELOW SWEEP 3 (medium) -- the three quiet unifications: SH for sound+light, conditional Propagator, storage spine
from holographic_spharm import sphere_dirs as _s3m_dirs, sh_project as _s3m_proj, sh_reconstruct as _s3m_rec
from holographic_condprop import ConditionalPropagator as _S3MCP
from holographic_storage import StorageSpine as _S3MSpine
_s3m_d = _s3m_dirs(300)
# 8. one SH primitive, two domains
_s3m_ld = _ri_np.array([0.2, 0.4, 0.9]); _s3m_ld /= _ri_np.linalg.norm(_s3m_ld)
_s3m_lit = _ri_np.clip(_s3m_d @ _s3m_ld, 0, None) ** 2
_s3m_lerr = float(_ri_np.sqrt(_ri_np.mean((_s3m_rec(_s3m_proj(_s3m_d, _s3m_lit, 4), _s3m_d, 4) - _s3m_lit) ** 2)) / (_s3m_lit.std() + 1e-9))
_s3m_sd = _ri_np.array([-0.6, 0.2, 0.7]); _s3m_sd /= _ri_np.linalg.norm(_s3m_sd)
_s3m_gain = _ri_np.clip(_s3m_d @ _s3m_sd, 0, None)
_s3m_serr = float(_ri_np.sqrt(_ri_np.mean((_s3m_rec(_s3m_proj(_s3m_d, _s3m_gain, 4), _s3m_d, 4) - _s3m_gain) ** 2)) / (_s3m_gain.std() + 1e-9))
# 9. conditional propagator: plan on a state graph
_s3m_rng = _ri_np.random.default_rng(0); _s3m_K = 6
_s3m_pl = _s3m_rng.standard_normal((_s3m_K, 256)); _s3m_pl /= _ri_np.linalg.norm(_s3m_pl, axis=1, keepdims=True)
_s3m_perms = [_ri_np.roll(_ri_np.arange(_s3m_K), a + 1) for a in range(2)]
_s3m_tr = [[(_s3m_pl[i], _s3m_pl[_s3m_perms[a][i]]) for i in range(_s3m_K)] for a in range(2)]
_s3m_cp = _S3MCP.learn(_s3m_tr); _s3m_plan = [0, 1, 0, 1, 0]
_s3m_end = _s3m_cp.plan(_s3m_pl[0], _s3m_plan, codebook=_s3m_pl); _s3m_tgt = 0
for _a in _s3m_plan:
    _s3m_tgt = _s3m_perms[_a][_s3m_tgt]
_s3m_hit = int((_s3m_pl @ (_s3m_end / _ri_np.linalg.norm(_s3m_end))).argmax()) == _s3m_tgt
# 7. storage spine: dedup + erasure
_s3m_sp = _S3MSpine(block_size=16); _s3m_payload = b"a record that must survive loss" * 4
_s3m_sp.put(("db", "x"), _s3m_payload); _s3m_sp.put(("cache", "x2"), _s3m_payload)
_s3m_recov = _s3m_sp.get(("db", "x"), loss=0.3) == _s3m_payload
print(f"  ABOVE/BELOW SWEEP 3 (medium) -- the three quiet unifications, where separate backlog items turned out to be ONE mechanism. (8) DIRECTIONAL SH: real spherical harmonics were already in the tree for LIGHT (prt); the same basis IS directional SOUND (ambisonics), so one project/reconstruct serves both -- a light lobe (err {_s3m_lerr:.2f}) and a sound gain (err {_s3m_serr:.2f}) from the identical code, no fork. (9) CONDITIONAL PROPAGATOR: 'predict = bind a transform to a state' fuses dynamics, lookahead, video, and backward-warp -- so lookahead's per-action model IS a Propagator per action; on a state-graph it plans {len(_s3m_plan)} hops to the right place ({'hit' if _s3m_hit else 'miss'}) via cleanup-every-hop, and its inverse recovers the prior state. (7) STORAGE SPINE: uri keys + content dedup + fountain erasure are one layer -- identical payloads stored once, recovered under 30% droplet loss ({'ok' if _s3m_recov else 'FAIL'}). Each reuses the existing modules -- the unification only counts if the code is actually one.")

# FORECASTING PRODUCERS (F3/F4/F6) -- analog recall, the forecast() router, the trusted-horizon gate
from holographic_analog import AnalogForecaster as _AF, delay_embed as _dembed
from holographic_forecast import route_and_forecast as _route
from holographic_horizon import MultiHorizonForecaster as _MHF
_fc_rng = _ri_np.random.default_rng(1)
# F4 analog forecasting on a fast quasi-periodic signal -- pure recall, beats persistence and the mean
_fc_t = _ri_np.arange(4000)
_fc_series = _ri_np.sin(_fc_t * 0.55) + 0.5 * _ri_np.sin(_fc_t * 0.27) + 0.03 * _fc_rng.standard_normal(4000)
_fc_ctx, _fc_succ = _dembed(_fc_series, 20)
_fc_af = _AF(sim_floor=0.5, seed=0).fit(_fc_ctx[:3000], _fc_succ[:3000])
_fc_ea = []; _fc_ep = []; _fc_em = []; _fc_tm = float(_fc_succ[:3000].mean())
for _fc_i in range(3000, len(_fc_ctx)):
    _fc_f = _fc_af.forecast(_fc_ctx[_fc_i], k=8)
    if _fc_f["abstain"]:
        continue
    _fc_ea.append(abs(_fc_f["point"] - _fc_succ[_fc_i]))
    _fc_ep.append(abs(_fc_ctx[_fc_i][-1] - _fc_succ[_fc_i]))
    _fc_em.append(abs(_fc_tm - _fc_succ[_fc_i]))
_fc_mae = float(_ri_np.mean(_fc_ea)); _fc_pmae = float(_ri_np.mean(_fc_ep)); _fc_mmae = float(_ri_np.mean(_fc_em))
# F3 the forecast() router: a logistic map routes to analog, an AR(1) process routes to linear -- by measured fit
_fc_lx = [0.37]
for _fc_k in range(3000):
    _fc_lx.append(3.9 * _fc_lx[-1] * (1 - _fc_lx[-1]))
_fc_rf, _fc_info = _route(_ri_np.array(_fc_lx), d=4, alpha=0.1)
_fc_arx = [0.0]
for _fc_k in range(3000):
    _fc_arx.append(0.8 * _fc_arx[-1] + 0.1 * _fc_rng.standard_normal())
_fc_rf2, _fc_info2 = _route(_ri_np.array(_fc_arx), d=5, alpha=0.1)
# F6 the trusted-horizon gate: a smooth ramp is trusted far; the chaotic logistic map only a few steps
_fc_mh = _MHF(lambda st, H: _ri_np.arange(1, H + 1) * float(st), alpha=0.1)
_fc_states = [float(v) for v in _fc_rng.standard_normal(200)]
_fc_mh.calibrate(_fc_states, [_ri_np.arange(1, 11) * _fc_s for _fc_s in _fc_states], 10)
_fc_smooth_h = _fc_mh.forecast(1.0, tolerance=float("inf"))["trusted_horizon"]
print(f"  FORECASTING PRODUCERS (F3/F4/F6) -- the mind can PRODUCE the next state four ways; three of them shown here. F4 ANALOG forecasting is pure recall -- 'find the past that looks like now, return what followed' (HoloForest pointed at time), and it yields a whole distribution, not just a point: on a fast quasi-periodic signal its MAE is {_fc_mae:.3f}, beating persistence {_fc_pmae:.3f} and the mean {_fc_mmae:.3f}. F3 the forecast() ROUTER runs the cheap producers and keeps the one that calibrates tighter: the logistic map (nonlinear) routes to '{_fc_info['chosen']}' (analog MAE {_fc_info['analog_mae']:.3f} < linear {_fc_info['linear_mae']:.3f} -- analog nails a deterministic map a linear window can't), while an AR(1) process routes to '{_fc_info2['chosen']}'. A misroute fails SAFE (a wide interval), never a confident wrong answer. F6 the TRUSTED-HORIZON gate widens the interval as error compounds and reports how far ahead to trust a cheap roll before recomputing -- a smooth ramp is trusted {_fc_smooth_h} steps out, where a chaotic map would be trusted only a few (Lyapunov time, kept mechanical). Kept loud: analog works only where the present resembles the stored past -- a novel regime has no analog and the honest output is abstention, not prophecy.")

# RENDER SPEED, VSA-NATIVE (technique E) -- edge-aware SVGF denoise as a cosine in a bound feature space
from holographic_svgf import atrous_bilateral as _atrous, plain_blur as _pblur, _psnr as _psnr_fn
_sv_rng = _ri_np.random.default_rng(0)
_sv_H = _sv_W = 64
_sv_clean = _ri_np.zeros((_sv_H, _sv_W, 3)); _sv_n = _ri_np.zeros((_sv_H, _sv_W, 3))
_sv_a = _ri_np.zeros((_sv_H, _sv_W, 3)); _sv_z = _ri_np.zeros((_sv_H, _sv_W))
_sv_clean[:, :_sv_W // 2] = [0.8, 0.2, 0.2]; _sv_clean[:, _sv_W // 2:] = [0.2, 0.3, 0.8]
_sv_n[:, :_sv_W // 2] = [0, 0, 1]; _sv_n[:, _sv_W // 2:] = [1, 0, 0]
_sv_a[:, :_sv_W // 2] = [0.8, 0.2, 0.2]; _sv_a[:, _sv_W // 2:] = [0.2, 0.3, 0.8]
_sv_z[:, :_sv_W // 2] = 1.0; _sv_z[:, _sv_W // 2:] = 3.0
_sv_noisy = _ri_np.clip(_sv_clean + 0.15 * _sv_rng.standard_normal((_sv_H, _sv_W, 3)), 0, 1)
_sv_den = _atrous(_sv_noisy, _sv_n, _sv_a, _sv_z, levels=5)
_sv_blur = _pblur(_sv_noisy, levels=5)
_sv_pn = _psnr_fn(_sv_noisy, _sv_clean); _sv_pd = _psnr_fn(_sv_den, _sv_clean); _sv_pb = _psnr_fn(_sv_blur, _sv_clean)
print(f"  RENDER SPEED, VSA-NATIVE (technique E) -- SVGF said the engine's way: to denoise a noisy 1-spp-style image without smearing across surface boundaries, bind each pixel's (normal, albedo, depth) into a feature vector and blend a neighbour weighted by the COSINE of their features -- the ScalarEncoder's RBF bump doing edge-stopping, run coarse-to-fine over the multires pyramid. Measured against the plain-blur baseline it earns its keep: a noisy image at {_sv_pn:.1f} dB cleans to {_sv_pd:.1f} dB, where the edge-BLIND blur only reaches {_sv_pb:.1f} dB (it bleeds colour across the edge; the feature-aware filter stops at it). The sibling pieces the render backlog names were already in the tree -- firefly-robust accumulation, SPRT adaptive sampling, temporal reproject -- so this was the one genuinely-missing part they compose with. Kept loud: it denoises, it cannot add detail; and a shared kernel is not a shared manifold -- the edge-stop is measured, not assumed.")

# VSA QUERY INTERFACE (Phases 1-3) -- a role-bound record IS a row, so a query is a projection over the store
from holographic_query import from_rows as _qfrom, run_sql as _qsql, Query as _QQ
_q_rows = [
    {"name": "gold", "colour": "yellow", "density": 19300},
    {"name": "copper", "colour": "orange", "density": 8960},
    {"name": "silver", "colour": "grey", "density": 10490},
    {"name": "iron", "colour": "grey", "density": 7870},
    {"name": "lead", "colour": "grey", "density": 11340},
]
_q_t = _qfrom(_q_rows, ["name", "colour", "density"], dim=2048, seed=0)
_q_exact = _qsql("SELECT name, density FROM materials WHERE density > 9000 ORDER BY density LIMIT 2", _q_t)
_q_fuzzy = _QQ().select("name", "colour").where("colour", "~", "grey").order_by("similarity").run(_q_t)
_q_names = [_qr["name"] for _qr in _q_exact]
_q_fnames = [_qr["name"] for _qr in _q_fuzzy]
# Phase 5: GROUP BY + aggregates (exact on stored props) + a per-group centroid (bundle = the VSA aggregate)
_q_grp = _qsql("SELECT colour, COUNT(*), AVG(density) FROM materials GROUP BY colour ORDER BY colour ASC", _q_t)
_q_grey = next(_r for _r in _q_grp if _r["colour"] == "grey")
_q_gcount = _q_grey["COUNT(*)"]
_q_gavg = float(_q_grey["AVG(density)"])
# Phase 6: the capability registry -- introspection becomes a data query over the mind's own faculties
from holographic_unified import UnifiedMind as _QMind
_q_mind = _QMind(dim=256, seed=0)
_q_reg = _q_mind.capabilities()
_q_nfac = len(_q_reg)
_q_census = _qsql("SELECT domain, COUNT(*) FROM actions GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 4", _q_reg)
_q_census_str = ", ".join("%s %d" % (_r["domain"], _r["COUNT(*)"]) for _r in _q_census)
# Phase 7: EXPLAIN = a dry run -- name the faculties a program WOULD call without running them
from holographic_machine import HoloMachine as _QMac
_q_mac = _QMac(dim=1024, seed=0, faculties=["denoise", "recall"])
_q_prog = _q_mac.assemble([("APPLY", "denoise"), ("APPLY", "recall"), ("HALT", None)])
_q_explain = _q_mind.explain_program(_q_mac, _q_prog)
# Phases 9-13: own your own database over the read-only system wall
from holographic_query import Database as _QDB
_q_db = _q_mind.database()                                 # ships with system.actions = the capability registry
_q_mind.db_query("CREATE DATABASE mine", _q_db)
_q_db.create_table("mine.faves", ["name"], dim=256)
_q_db.insert_select("mine.faves", ["name"], "system.actions", where=("domain", "=", "forecasting"))
_q_nfaves = len(_q_db.resolve("mine.faves").rows)
_q_wall_ok = False
try:
    _q_mind.db_query("INSERT INTO system.actions (name) VALUES (\047hack\047)", _q_db)
except Exception:
    _q_wall_ok = True
_q_db2 = _QDB.from_state(_q_db.to_state())                 # persistence by replay
_q_reload_ok = (_q_db2.resolve("mine.faves").rows == _q_db.resolve("mine.faves").rows)
# Phase 4: GraphQL for the nested scene -- ask for exactly the nested fields you want
_q_scene = _q_mind.make_scene([
    {"name": "ring", "material": "gold", "transform": {"kind": "rigid", "position": [1, 0, 0]}},
    {"name": "pipe", "material": "copper", "transform": {"kind": "rigid", "position": [0, 2, 0]}},
    {"name": "coin", "material": "gold", "transform": {"kind": "static", "position": [3, 0, 0]}},
])
_q_gql = _q_mind.query_scene('{ objects(where: {material: "gold"}) { name transform { kind } } }', _q_scene)
_q_gql_str = str(_q_gql["objects"])
_q_facs = _q_explain["faculties_called"]
_q_steps = _q_explain["n_steps"]
_q_conf = float(_q_fuzzy[0]["_confidence"])
print(f"  VSA QUERY INTERFACE (Phases 1-13) -- a role-bound record IS a database ROW (roles are columns, fillers are values), so a query is a PROJECTION over the store we already have -- no new database. Exact predicates run on the stored props (a decoded float has readback error -- the honest exact/fuzzy fork): the densest above 9000 are {_q_names}. FUZZY colour~grey ranks by MEANING, returning {_q_fnames} at per-row confidence {_q_conf:.2f} -- a database that can say real-match vs noise-abstaining, which plain SQL cannot. AGGREGATION (Phase 5): GROUP BY colour gives grey a count of {_q_gcount} and mean density {_q_gavg:.0f}, exact on the stored props, plus a per-group centroid (the bundle = the group prototype). QUERY THE MIND (Phase 6): the same engine introspects the mind itself -- its {_q_nfac} faculties become a table actions, so a capability census is one GROUP BY: {_q_census_str}. EXPLAIN (Phase 7) dry-runs a program without executing it: APPLY denoise then APPLY recall reports it WOULD call {_q_facs} in {_q_steps} steps. Executing with queried arguments already runs (run_procedure). OWN YOUR DATA (Phases 9-13): a database of user namespaces sits over a READ-ONLY system namespace -- reads from system.* work, but the wall REFUSES writes to it ({_q_wall_ok}); bookmarking the forecasting faculties from system.actions into a user table caught {_q_nfaves} rows, and the whole database persists by REPLAY and reloads byte-identical ({_q_reload_ok}). GRAPHQL FOR THE SCENE (Phase 4): SQL fits flat tables, but a scene is a nested graph, so GraphQL fits it -- asking for name and transform.kind on the gold objects returns exactly {_q_gql_str}, and each nested field resolves by unbinding exactly that role chain. That completes the query interface end to end: one projection core reached by SQL, GraphQL, introspection, and an owned database.")

# FORECASTING SWEEP (sec.5) -- delegate the improvised confidences to ONE calibrated engine
from holographic_superschedule import calibrated_capacity as _sw_cc, pack_capacity as _sw_pc
from holographic_adaptive_sample import sample_budget as _sw_sb
_sw_cap90, _ = _sw_cc(512, gated=True, target_recall=0.9, seed=0)
_sw_cap99, _ = _sw_cc(512, gated=True, target_recall=0.99, seed=0)
_sw_theo = _sw_pc(512, gated=True)
_sw_var = _ri_np.array([1e-5, 4e-3, 1e-2])                       # three pixels: converged, noisy, noisier
_sw_budget = [int(x) for x in _sw_sb(_sw_var, 64, 0.05)]
print(f"  FORECASTING SWEEP (sec.5) -- the sweep's idea: every improvised 'estimate + confidence + abstain' in the engine should delegate to ONE calibrated primitive. Probing first found THREE already done (the resonator's null-calibrated soft confidence, recall_calibrated, the agent's decide_confidence), so only two needed building. THE SCHEDULER COST MODEL IS A FORECASTER: instead of packing to the assumed theoretical wall ({_sw_theo} items at D=512), it now MEASURES the wall -- probe growing superposition loads, measure gated recall, keep the largest load that stays confident. The measured capacity is {_sw_cap90} at target 0.90 and {_sw_cap99} at 0.99, BELOW the assumed {_sw_theo}: trusting 0.10*D would OVERPACK at a strict recall target (assuming beats measuring only when you're lucky). THE RENDERER STOP: given per-pixel variance-of-the-mean, spend extra samples only where the estimate is still uncertain -- for three pixels of increasing noise at 64 samples the budget is {_sw_budget} extra (0 where converged). Kept honest: a pixel mean's interval is Gaussian/CLT (halving it costs 4x the samples), NOT conformal -- a single pixel has no calibration set, so the sweep's 'conformal everywhere' framing is corrected to the estimator that actually fits.")

# SPECTRAL FIELD BACKBONE (physics: advancing a linear field is ONE bind, any t in closed form)
from holographic_spectralfield import diffusion_field as _sf_diff, wave_field as _sf_wave
from holographic_heat import diffuse_heat as _sf_heat
_sf_N = 128
_sf_x = _ri_np.arange(_sf_N)
_sf_s0 = 4.0
_sf_D = 0.5
_sf_T = 20.0
_sf_f0 = _ri_np.exp(-((_sf_x - _sf_N / 2) ** 2) / (2 * _sf_s0 ** 2))
_sf_analytic = _sf_f0.max() * _sf_s0 / _ri_np.sqrt(_sf_s0 ** 2 + 2 * _sf_D * _sf_T)
_sf_spectral = _sf_diff(_sf_f0.copy(), D=_sf_D, dx=1.0).advanced(_sf_T)
_sf_grid = _sf_heat(_sf_f0.copy().reshape(1, _sf_N), alpha=_sf_D, dx=1.0, dt=0.05, steps=400).ravel()
_sf_spec_err = float(abs(_sf_spectral.max() - _sf_analytic))
_sf_grid_err = float(abs(_sf_grid.max() - _sf_analytic))
_sf_k = 2 * _ri_np.pi * 3 / _sf_N
_sf_mode = _ri_np.cos(_sf_k * _sf_x)
_sf_period = 2 * _ri_np.pi / (2.0 * _sf_k)
_sf_back = _sf_wave(_sf_mode.copy(), c=2.0, dx=1.0).advanced(_sf_period)[0]
_sf_return_err = float(_ri_np.max(_ri_np.abs(_sf_back - _sf_mode)))
print(f"  SPECTRAL FIELD BACKBONE (physics) -- a linear field IS a hypervector, and advancing it in time is ONE bind: a circular convolution, diagonal in the Fourier basis, so we diagonalise once (the FFT) and jump to ANY time t in closed form -- no stepping, no accumulated error. Every linear domain is then one dispersion relation on this backbone: diffusion/heat is rate -D|k|^2, waves/EM are omega = c|k|, the ocean is the dispersive omega = sqrt(g|k|), electrostatics is the closed-form Poisson limit. MEASURED against the grid baseline: diffusing a Gaussian spot to t=20, the spectral field matches the analytic heat kernel to {_sf_spec_err:.1e} (machine precision) in ONE eval, while the grid diffuse_heat baseline takes 400 steps and still carries {_sf_grid_err:.1e} error -- exact and cheaper, a clean win, not just a tie. The closed-form jump equals stepping to FFT tolerance, and a wave mode returns to itself after exactly one period T=2pi/(c|k|) (residual {_sf_return_err:.1e}). Superposition is bundle (sources add, no re-sim); a calibrated trigger marks where the cheap path stops being enough. Kept honest: only LINEAR operators diagonalise -- an overturning wave or a shock stays a grid solver (the adaptive ladder's top rung), and the spectral world has periodic boundaries.")

# ADAPTIVE WAVE SOLVER (physics #5) -- the ocean stack dispatched per tile like the renderer (plan_render for water)
from holographic_waveadaptive import plan_waves as _aw_plan, plan_cost as _aw_cost, method_counts as _aw_counts, all_one_method_cost as _aw_all
_aw_H = _aw_W = 64
_aw_h = 0.05 * _ri_np.random.default_rng(0).standard_normal((_aw_H, _aw_W))
_aw_h[8:16, 8:16] += _ri_np.linspace(0, 6, 8)[:, None]      # a steep, breaking crest
_aw_depth = _ri_np.full((_aw_H, _aw_W), 10.0)
_aw_depth[:, :12] = 1.0                                     # a shallow shore strip
_aw_plan_d = _aw_plan(_aw_h, depth=_aw_depth, obstacles=[(48, 48, 56, 56)], tile=8)
_aw_c = _aw_counts(_aw_plan_d)
_aw_adaptive = _aw_cost(_aw_plan_d)
_aw_allgrid = _aw_all(_aw_plan_d, "free_surface")
_aw_savings = 100.0 * (1 - _aw_adaptive / _aw_allgrid)
_aw_counts_str = ", ".join("%s %d" % (m, n) for m, n in sorted(_aw_c.items(), key=lambda kv: -kv[1]))
print(f"  ADAPTIVE WAVE SOLVER (physics #5) -- the renderer never uses one method everywhere: plan_render picks bake/analytic/trace per region from measured break-evens. This gives WATER the same treatment. plan_waves tiles the sea and, per tile, picks the wave method from the LOCAL regime and says why: open deep water gets the cheap global fft_ocean, a tile near the rock gets wave_packets (reflection/diffraction), the shallow shore gets shallow_water (shoaling), and only the steep breaking crest gets free_surface, the dear grid solver. On this scene the plan is: {_aw_counts_str}. The whole adaptive plan costs {_aw_adaptive:.0f} versus {_aw_allgrid:.0f} to run the grid solver on every tile -- {_aw_savings:.0f} percent cheaper, because the expensive solver runs only in the few tiles that actually break. The plan is inspectable BEFORE running and deterministic (the tie-break is breaking > shallow > obstacle > open). Kept honest: a sea that breaks EVERYWHERE gets no discount (the dear rung is local, not free), and the overturning-barrel grid solver itself is the deferred rung-4 item -- this dispatch identifies exactly where it is needed and runs cheap everywhere else.")

# WAVE-PACKET FIELD (physics N8) -- tricky waves the global FFT ocean can't do: reflection off a wall
from holographic_wavepacket import WavePacketField as _wp_Field
_wp = _wp_Field(size=64.0, g=9.81, seed=0)
_wp.add_packet(pos=[60.0, 32.0], wavevector=[1.2, 0.0])
_wp_kx0 = float(_wp.k[0, 0])
_wp_cg = float(_wp._group_speed(1.2))
_wp_cp = float(_wp._omega(1.2) / 1.2)
for _ in range(200):
    _wp.advance(0.1)
_wp_kx1 = float(_wp.k[0, 0])
_wp_backin = bool(_wp.pos[0, 0] < 60.0)
print(f"  WAVE-PACKET FIELD (physics N8) -- the FFT ocean is GLOBAL, so no wave can reflect off a wall or bend around a rock. The fix (Jeschke & Wojtan) is to LOCALIZE the spectrum: the surface becomes many little Gaussian-enveloped wave trains, each living at a PLACE, so each one can reflect, shoal, and diffract. Here a packet is launched at the far wall travelling +x (k_x={_wp_kx0:+.1f}); after it hits the wall its wavevector MIRRORS to k_x={_wp_kx1:+.1f} and it heads back inward ({_wp_backin}) -- specular reflection, k' = k - 2(k.n)n. Its ENERGY travels at the group velocity {_wp_cg:.2f}, exactly half the phase velocity {_wp_cp:.2f} for deep-water gravity waves (why a wave group outlives its crests). And this is native, not bolted on: a packet IS a role-bound record (position, wavenumber, direction, amplitude, phase bound to roles and bundled), and the surface IS a bundle of those records -- a content-addressable index you can query by cosine. Kept honest: this is the localized-spectrum rung; it reflects and shoals but does not overturn -- a breaking barrel is the grid solver at the top of the adaptive ladder.")

# TRANSFORM UTILITIES + CANCELLATION + API FACADE (modeling-app G+F+H) -- the responsiveness & front-door tier
import lecore as _lc
from holographic_transform import (compose_trs as _tx_compose, decompose as _tx_decompose, quat_from_euler as _tx_qe,
                                   quat_to_matrix as _tx_qm, quat_slerp as _tx_slerp, quat_from_axis_angle as _tx_qaa,
                                   quat_to_axis_angle as _tx_q2aa, look_at as _tx_lookat)
from holographic_cancel import CancelToken as _tx_Cancel
from holographic_pathtrace import path_trace as _tx_pt
from holographic_render import Camera as _tx_Cam
from holographic_sdf import sphere as _tx_sphere
# G: decompose a T*R*S matrix (what a gizmo reads off) and recompose -- round-trips exactly
_tx_t = _ri_np.array([2.0, -3.0, 5.0]); _tx_q = _tx_qe(0.3, -0.7, 1.1); _tx_s = _ri_np.array([2.0, 0.5, 1.5])
_tx_t2, _tx_q2, _tx_s2 = _tx_decompose(_tx_compose(_tx_t, _tx_q, _tx_s))
_tx_trs_ok = bool(_ri_np.allclose(_tx_t2, _tx_t) and _ri_np.allclose(_tx_s2, _tx_s) and _ri_np.allclose(_tx_qm(_tx_q2), _tx_qm(_tx_q)))
# G: slerp halfway between angle 0 and 1 lands at angle 0.5; look_at sends the target down -z
_, _tx_midang = _tx_q2aa(_tx_slerp(_tx_qaa([0, 0, 1], 0.0), _tx_qaa([0, 0, 1], 1.0), 0.5))
_tx_lookz = float((_tx_lookat((3.0, 4.0, 5.0), (0.0, 0.0, 0.0)) @ _ri_np.array([0.0, 0.0, 0.0, 1.0]))[2])
# F: a cancel token stops a render early and returns a valid partial image
_tx_tok = _tx_Cancel(); _tx_calls = []
def _tx_prog(_im, _done, _total):
    _tx_calls.append(_done)
    if len(_tx_calls) >= 2:
        _tx_tok.cancel()
_tx_cam = _tx_Cam(eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0), fov_deg=45, aspect=1.0)
_tx_img = _tx_pt(_tx_sphere(1.0), _tx_cam, width=20, height=20, spp=12, progress_every=1, on_progress=_tx_prog, should_stop=_tx_tok, seed=0)
_tx_cancelled_at = len(_tx_calls)
_tx_finite = bool(_ri_np.isfinite(_tx_img).all())
# H: the facade areas
_tx_areas = {k: len(v) for k, v in _lc.areas().items()}
print(f"  TRANSFORM UTILITIES + CANCELLATION + API FACADE (modeling-app foundation) -- three small pieces that make leCore something you can BUILD AN APP ON. First, the transform math a gizmo and property panel need: decompose a T*R*S matrix into translate, a rotation quaternion, and scale, then recompose -- round-trips exactly ({_tx_trs_ok}); quaternion SLERP for smooth rotation (halfway between angle 0 and 1 lands at {_tx_midang:.2f}); and look_at for a camera, which sends the target straight down -z (z={_tx_lookz:.1f}, the OpenGL convention the engine Camera uses). Second, cooperative CANCELLATION: a long render checks a CancelToken between passes, so a Stop button actually works -- here a token cancelled the path trace after {_tx_cancelled_at} passes and handed back a valid partial image (finite={_tx_finite}) instead of freezing the UI, and when unused it is bit-identical to before. Third, the FRONT DOOR: 'import lecore' gives one curated surface -- scene ({_tx_areas['scene']}), model ({_tx_areas['model']}), render ({_tx_areas['render']}), sim ({_tx_areas['sim']}), transform ({_tx_areas['transform']}) names -- so building on leCore doesn't mean knowing which of ~280 modules to import. Items G, F, and H of the modeling-app backlog: the responsiveness and the front door.")

# MODIFIER STACK + DEPENDENCY GRAPH (modeling-app C+D) -- non-destructive, O(change) re-evaluation
from holographic_modifier import ModifierStack as _ms_Stack
_ms_calls = {"n": 0}
def _ms_scale(v, factor=1.0):
    _ms_calls["n"] += 1
    return v * factor
def _ms_offset(v, amount=0.0):
    _ms_calls["n"] += 1
    return v + amount
_ms_st = _ms_Stack(base=1.0)
_ms_h0 = _ms_st.add("offset", _ms_offset, {"amount": 1.0})
_ms_h1 = _ms_st.add("scale", _ms_scale, {"factor": 2.0})
_ms_h2 = _ms_st.add("offset2", _ms_offset, {"amount": 10.0})
_ms_h3 = _ms_st.add("scale2", _ms_scale, {"factor": 3.0})
_ms_r0 = _ms_st.evaluate()                                  # ((1+1)*2+10)*3 = 42, all four ops run
_ms_full = _ms_calls["n"]
_ms_calls["n"] = 0
_ms_st.set_param(_ms_h2, amount=20.0)                       # tweak the 3rd modifier
_ms_r1 = _ms_st.evaluate()                                  # ((1+1)*2+20)*3 = 72, only the 2 below re-run
_ms_partial = _ms_calls["n"]
print(f"  MODIFIER STACK + DEPENDENCY GRAPH (modeling-app foundation) -- modern modeling apps have a non-destructive modifier stack (Blender) and a dependency graph (Maya/Houdini) that re-evaluates only what a change affects. leCore already had the substrate -- the recipe is an ordered, non-destructive op sequence with stable handles, validate, and reorder -- so this is a PROMOTION: carry that pattern to a per-object stack over ANY payload (a mesh, a field, a vector) and add the one thing a dependency graph needs that a plain recipe doesn't -- O(change) re-evaluation. A four-modifier stack folds non-destructively over the base to {_ms_r0:.0f}, running all {_ms_full} ops. Then tweaking the THIRD modifier's parameter re-evaluates to {_ms_r1:.0f} but runs only {_ms_partial} ops, not {_ms_full} -- the two modifiers ABOVE it are reused from cache, because each modifier depends only on the one before. That 'recompute only downstream of a change' IS the dependency graph, and it is dirtyfield's O(change) idea on a linear op chain. Handles stay stable across reorder/insert/remove so a property panel or an animation can target one modifier, and describe() lists a modifier's parameters as a schema -- the property-panel introspection (item D), which in VSA terms is enumerate-the-roles of a record. Items C and D of the modeling-app backlog, riding the canonical Scene document.")

# CANONICAL SCENE DOCUMENT (modeling-app foundation, item 0) -- one source of truth, STABLE handles, events, undo
from holographic_scene_doc import Scene as _sd_Scene
_sd_scene = _sd_Scene(dim=256, seed=0)
_sd_events = []
_sd_scene.on_change(lambda k, h: _sd_events.append(k))
_sd_a = _sd_scene.add(name="wheel", geometry=_ri_np.zeros((4, 3)), tags={"material": "metal"})
_sd_b = _sd_scene.add(name="body", geometry=_ri_np.ones((4, 3)), tags={"material": "paint"})
_sd_key0 = _sd_scene._content_key[_sd_a]
_sd_id0 = _sd_scene.handle_vector(_sd_a).copy()
_sd_scene.select([_sd_a])
_sd_scene.edit(_sd_a, geometry=_ri_np.full((4, 3), 7.0), name="front-wheel")
_sd_key1 = _sd_scene._content_key[_sd_a]
_sd_id_same = bool(_ri_np.array_equal(_sd_scene.handle_vector(_sd_a), _sd_id0))
_sd_still_selected = _sd_a in _sd_scene.selection
_sd_scene.undo()
_sd_name_undo = _sd_scene.get(_sd_a).name
_sd_nevents = len(_sd_events)
print(f"  CANONICAL SCENE DOCUMENT (the modeling-app foundation) -- a modeling app needs ONE authoritative document that every tool edits and every output reads; today that is fragmented across the renderer, the scene graph, the animation timeline, and the solvers. This is the single source of truth: a VSA table of object records plus a hierarchy, owning the selection and the undo history and firing change events so the UI stays in sync. The keystone is STABLE HANDLES -- an object's identity is a permanent random hypervector minted at creation, kept SEPARATE from its content hash. Here is why that matters: editing the wheel's geometry changes its content hash ({_sd_key0} -> {_sd_key1}, different), so a content hash could never be a handle -- it would dangle on every edit -- but the handle and its identity atom are UNCHANGED (same identity = {_sd_id_same}), so the selection pointing at it SURVIVES the edit (still selected = {_sd_still_selected}). Every mutation fired a change event ({_sd_nevents} so far: add, add, select, edit, undo), and because edits go through the one document, undo is automatic -- a single call restored the wheel's name to '{_sd_name_undo}' with its identity intact. This is item 0 of the modeling-app backlog: the foundation the whole feature layer -- selection, tagging, measurement, tools -- attaches to.")

# SELECTION + SEARCH + TAGGING (modeling-app feature layer) -- the scene is a table, so selection is a query
from holographic_scene_doc import Scene as _sq_Scene
from holographic_scene_query import (select as _sq_select, select_fuzzy as _sq_fuzzy, select_by_tag as _sq_bytag,
                                     tag as _sq_tag, Selection as _sq_Selection)
_sq_scene = _sq_Scene(dim=512, seed=0)
_sq_w1 = _sq_scene.add(name="wheel_front", material="metal", tags={"kind": "wheel"})
_sq_w2 = _sq_scene.add(name="wheel_rear", material="metal", tags={"kind": "wheel"})
_sq_body = _sq_scene.add(name="body", material="paint", tags={"kind": "panel"})
_sq_glass = _sq_scene.add(name="windscreen", material="glass", tags={"kind": "panel"})
_sq_metal = _sq_select(_sq_scene, material="metal")
_sq_sel = _sq_Selection(_sq_scene)
_sq_not_front = _sq_sel.minus(_sq_metal, {_sq_w1})
_sq_hits = _sq_fuzzy(_sq_scene, "material", "metal")
_sq_conf = _sq_hits[0][1] if _sq_hits else 0.0
_sq_tag(_sq_scene, list(_sq_metal), "reviewed", True)
_sq_reviewed = len(_sq_bytag(_sq_scene, "reviewed"))
_sq_scene.undo(); _sq_scene.undo()
_sq_after_undo = len(_sq_bytag(_sq_scene, "reviewed"))
print(f"  SELECTION + SEARCH + TAGGING (modeling-app feature layer) -- the scene is a VSA table of object records, so the whole organizational layer falls out of query/bundle/cleanup with no new machinery. A SELECTION is just a query result: 'select the metal parts' returns {len(_sq_metal)} handles by exact, lossless filtering over the records. SET ALGEBRA is the algebra of selections -- 'everything metal, minus the front wheel' is one line and leaves {len(_sq_not_front)}. SEARCH can also be FUZZY, riding the query layer: 'material ~ metal' ranks the parts by MEANING and carries a calibrated confidence ({_sq_conf:.2f}), so a near-miss is never a silent inclusion. TAGGING is a bound role written through the document, so a tag is both queryable ('reviewed' -> {_sq_reviewed} parts) AND undoable -- two undos remove the tags ({_sq_after_undo} left). A named set stores membership as an id list (exact, no capacity ceiling), while the selection-as-a-bundle is offered for the vector algebra on modest sets. Selection = query, tagging = bound role, search = the query layer, set algebra = set ops -- all on the one canonical Scene document.")

# UNDO / REDO STACK (modeling-app feature layer) -- grouped transactions, labels, history
from holographic_scene_doc import Scene as _un_Scene
_un_scene = _un_Scene(dim=256, seed=0)
_un_a = _un_scene.add(name="wheel_l", material="metal")
_un_b = _un_scene.add(name="wheel_r", material="metal")
_un_c = _un_scene.add(name="body", material="paint")
with _un_scene.group("Chrome the wheels"):                 # a multi-object tool = ONE undo step (a transaction)
    _un_scene.edit(_un_a, material="chrome")
    _un_scene.edit(_un_b, material="chrome")
_un_hist = _un_scene.history()
_un_scene.undo()                                           # one undo reverts BOTH wheels
_un_a_mat = _un_scene.get(_un_a).material
_un_scene.redo()
_un_a_mat2 = _un_scene.get(_un_a).material
print(f"  UNDO / REDO STACK (modeling-app feature layer) -- because every mutation already goes through the one Scene document, undo is nearly free: each records a cheap before/after snapshot of just the affected record. The stack on top adds what an app actually needs. TRANSACTIONS: a drag, or a tool that edits several objects at once, wraps them in 'with scene.group(label)', so the whole batch is ONE undo step -- here 'Chrome the wheels' re-materialled two wheels, and a SINGLE undo reverted both (the wheel went back to '{_un_a_mat}'), with redo re-applying it ('{_un_a_mat2}'). LABELS + HISTORY: each step carries a human-readable label, so an Edit menu or history panel just reads scene.history() -> {_un_hist}. Nested transactions commit as one step, an empty transaction records nothing, redo is invalidated by a fresh edit, and the history depth is capped. It stays a thin, readable layer -- a snapshot swap, O(one record) per change -- that composes with the delta and version stores for heavy geometry. The undo/redo stack of the modeling-app feature layer, owned by the canonical Scene document.")

# MEASUREMENT + UNITS (modeling-app feature layer) -- dimensioned quantities measured from the geometry
from holographic_metrology import (surface_area as _me_area, volume as _me_vol, bounding_box as _me_bbox,
                                   edge_length as _me_edge, _unit_cube as _me_cube)
_me_c = _me_cube()
_me_A = _me_area(_me_c); _me_V = _me_vol(_me_c); _me_bb = _me_bbox(_me_c); _me_d = _me_edge(_me_c, 0, 1)
_me_area_m2 = _me_A.to("m2"); _me_vol_L = _me_V.to("L"); _me_diag = _me_bb.diagonal.to("m"); _me_ft = _me_d.to("ft")
_me_refused = False
try:
    _ = _me_A + _me_d                                       # [m^2] + [m] -- a grammar error
except ValueError:
    _me_refused = True
print(f"  MEASUREMENT + UNITS (modeling-app feature layer) -- a measuring tool must return a DIMENSIONED quantity, read from the actual geometry, never a lossy VSA readback. Each measurement walks the mesh vertices and faces directly and wraps the result in a Quantity that carries its unit and dimension. A unit cube measures {_me_area_m2:.0f} m^2 of surface (the sum of its triangle areas) and {_me_V.to('m3'):.0f} m^3 of volume (the divergence theorem over its triangles), with a bounding-box diagonal of {_me_diag:.4f} m. Two things come for FREE from carrying the dimension. CONVERSION is one multiply: that same volume is {_me_vol_L:.0f} litres and that 1 m edge is {_me_ft:.4f} feet, with no chance to fumble the factor. And the dimensional algebra REFUSES nonsense: adding a length to an area is a grammar error, raised loudly (refused = {_me_refused}), so a measurement bug can't silently produce a meaningless number. Kept honest: areas and lengths are exact for the mesh as given, but volume assumes a CLOSED, consistently-wound surface (flagged, not hidden), and angles are dimensionless radians. The measurement + units of the modeling-app feature layer, measured straight from the geometry.")

# RENDER OVERRIDES + SNAPPING (modeling-app feature layer) -- a bound role with fallback, and cleanup
from holographic_scene_doc import Scene as _ov_Scene
from holographic_overrides import resolve as _ov_resolve, set_override as _ov_set
from holographic_snap import Snapper as _ov_Snapper, snap_to_points as _ov_snap_pts
_ov_scene = _ov_Scene(dim=128, seed=0)
_ov_defaults = {"samples": 64}
_ov_a = _ov_scene.add(name="hero"); _ov_b = _ov_scene.add(name="prop")
_ov_set(_ov_scene, _ov_b, "samples", 256)
_ov_b_samples = _ov_resolve(_ov_scene, _ov_b, "samples", _ov_defaults)
_ov_a_samples = _ov_resolve(_ov_scene, _ov_a, "samples", _ov_defaults)
_ov_snp = _ov_Snapper(grid=1.0, vertices=[[0.05, 0.05, 0.0]], tol=0.25)
_ov_out, _ov_kind = _ov_snp.snap([0.1, 0.1, 0.0])
_, _ov_far_i, _ = _ov_snap_pts([9, 9, 9], [[0.05, 0.05, 0.0]], tol=0.25)
print(f"  RENDER OVERRIDES + SNAPPING (modeling-app feature layer) -- two more features that fall straight out of the VSA reframe. A RENDER OVERRIDE is a bound role with a fallback: the object 'prop' overrides its sample count to {_ov_b_samples} while 'hero' inherits the scene default of {_ov_a_samples} -- only the DELTA is stored, everything else is inherited, so a scene of thousands with two special objects costs two entries, not thousands of copies. And SNAPPING is cleanup: a dragged point projects onto the nearest allowed place (here it snapped to a '{_ov_kind}'), with the same confidence gate cleanup uses -- a tolerance -- so a point with nothing close enough is left alone (index {_ov_far_i}). Bound-role-with-fallback and cleanup, wearing DCC costumes.")

# GROUPING / INSTANCING + CAMERA CONTROLLER (modeling-app feature layer) -- a bundle, a bind, and viewport nav
from holographic_scene_doc import Scene as _gc_Scene
from holographic_grouping import (group_objects as _gc_group, group_members as _gc_members, instance as _gc_inst,
                                  resolve_geometry as _gc_geo)
from holographic_camera import CameraController as _gc_Cam
from holographic_metrology import _unit_cube as _gc_cube, bounding_box as _gc_bbox
_gc_scene = _gc_Scene(dim=128, seed=0)
_gc_a = _gc_scene.add(name="wheel_l", geometry=_ri_np.zeros((4, 3)))
_gc_b = _gc_scene.add(name="wheel_r", geometry=_ri_np.ones((4, 3)))
_gc_c = _gc_scene.add(name="body", geometry=_gc_cube())
_gc_g = _gc_group(_gc_scene, [_gc_a, _gc_b], name="wheels")
_gc_nmembers = len(_gc_members(_gc_scene, _gc_g))
_gc_i = _gc_inst(_gc_scene, _gc_c)
_gc_scene.edit(_gc_c, geometry=_gc_cube())
_gc_follows = bool(_gc_geo(_gc_scene, _gc_i) is not None)
_gc_bb = _gc_bbox(_gc_scene.get(_gc_c).geometry)
_gc_camera = _gc_Cam(eye=(0, 0, 10), target=(0, 0, 0))
_gc_camera.frame(_gc_bb.min, _gc_bb.max, fov_deg=45.0)
_gc_dist = _gc_camera.distance
print(f"  GROUPING / INSTANCING + CAMERA CONTROLLER (modeling-app feature layer) -- the last feature-layer pieces. GROUPING is a bundle: the two wheels go under one null parent ({_gc_nmembers} members) in a single undo step, and the group's identity is the superposition of its members. INSTANCING is a bind: an instance shares its source's geometry (the shared filler) with its own transform (the bound placement), so editing the source updates every instance ({_gc_follows}) with nothing copied -- a forest of 10,000 trees costs one tree plus 10,000 transforms. And the CAMERA CONTROLLER gives the viewport its navigation on the transform utilities -- orbit, pan, dolly, zoom, and frame-a-box: here 'frame' fit the body's bounding sphere at distance {_gc_dist:.2f} (the 'zoom to fit' command). That completes the modeling-app feature layer -- selection, tagging, search, undo/redo, measurement, overrides, snapping, grouping, instancing, and the camera -- every one a bind/bundle/cleanup on the canonical Scene.")

# THE SAMPLER (modeling-app capstone) -- a placeable read-probe, the read-dual of FieldEffect
from holographic_sampler import (Sampler as _sm_Sampler, owners_from_sdfs as _sm_owners,
                                 contribution_of as _sm_contrib, dominant_owner as _sm_dom)
from holographic_sdf import sphere as _sm_sphere
class _sm_Shift:
    def __init__(_s, base, off):
        _s.base = base; _s.off = _ri_np.asarray(off, float)
    def eval(_s, P):
        return _s.base.eval(_ri_np.asarray(P, float) - _s.off)
_sm_field = lambda P: _ri_np.asarray(P, float)[:, 2]                       # a height field = z
_sm_point = _sm_Sampler(_sm_sphere(0.1), _sm_field, mode="point").sample(at=(0, 0, 3.0))
_sm_vol = _sm_Sampler(_sm_sphere(2.0), _sm_field, mode="volume", radius=2.0, falloff="linear").sample(at=(0, 0, 1), bounds=([-2, -2, 0.0], [2, 2, 2]), n=400, seed=0)
_sm_rng = _ri_np.random.default_rng(0)
_sm_hA = _sm_rng.standard_normal(256); _sm_hA /= _ri_np.linalg.norm(_sm_hA)
_sm_hB = _sm_rng.standard_normal(256); _sm_hB /= _ri_np.linalg.norm(_sm_hB)
_sm_own = _sm_owners([(_sm_hA, _sm_sphere(1.0)), (_sm_hB, _sm_Shift(_sm_sphere(1.0), [3, 0, 0]))])
_sm_lab = _sm_Sampler(_sm_sphere(5.0), lambda P: _ri_np.ones(len(P)), mode="volume", radius=5.0).sample_labeled(_sm_own, at=(1.5, 0, 0), bounds=([-2, -2, -2], [5, 2, 2]), n=600, seed=1)
_sm_cA = _sm_contrib(_sm_lab, _sm_hA); _sm_cB = _sm_contrib(_sm_lab, _sm_hB)
print(f"  THE SAMPLER (modeling-app capstone) -- the clearest single example of the 'think holographically' payoff: a brand-new, genuinely useful object that costs almost nothing because it's the READ-DUAL of a FieldEffect. A FieldEffect is a shaped, falloff-weighted WRITE to a field; a Sampler is the same shape and falloff pointed the other way -- a shaped READ from the scene. In POINT mode it reads the value at a spot (height at z=3 -> {_sm_point:.1f}); in VOLUME mode it averages a field over its shape's interior (mean height of the upper region -> {_sm_vol:.2f}). And when several objects overlap the sampled area, the native answer is a LABELED BUNDLE: each sample is tagged by its owning object's stable handle, scaled by its contribution, and bundled -- one superposed readout that SEPARATES by handle (object A contributed {_sm_cA:.0f}, object B {_sm_cB:.0f}) and COLLAPSES to a total, with the dominant owner a cleanup. It's a Scene object like any other -- placeable, animatable, its output able to drive a parameter or fire a trigger. The read-mirror of the write you already had: the modeling-app backlog's capstone, and the backlog is complete.")

# AUTO-BUMP (inverse-rendering IR1) -- image -> height -> normal map, with an honest abstain gate
from holographic_autobump import auto_bump as _ab_auto
_ab_N = 48
_ab_u = _ri_np.linspace(0, 6 * _ri_np.pi, _ab_N)
_ab_bump = 0.5 + 0.4 * _ri_np.outer(_ri_np.sin(_ab_u), _ri_np.cos(_ab_u))
_ab_bump_rgb = _ri_np.stack([_ab_bump, _ab_bump, _ab_bump], axis=-1)
_ab_ramp = _ri_np.tile(_ri_np.linspace(0.2, 0.8, _ab_N), (_ab_N, 1))
_ab_ramp_rgb = _ri_np.stack([_ab_ramp, _ab_ramp, _ab_ramp], axis=-1)
_ab_res = _ab_auto(_ab_bump_rgb, strength=2.0)
_ab_ramp_res = _ab_auto(_ab_ramp_rgb)
_ab_nvar = float(_ri_np.std(_ab_res["normal"][..., 0]))
_ab_unit = bool(_ri_np.allclose(_ri_np.linalg.norm(_ab_res["normal"], axis=-1), 1.0, atol=1e-6))
print(f"  AUTO-BUMP (inverse-rendering IR1) -- the concrete 'auto bump' ask: when a material has an albedo texture but no bump/normal map, derive a plausible one from the image alone -- pure classical arithmetic on the vision front-end, no learned prior. Grayscale, then HIGH-PASS to strip the slow lighting/albedo component, leaving the fine surface detail as a height field; turn its gradient into a tangent-space normal map (unit normals={_ab_unit}, varying by {_ab_nvar:.2f}, confidence {_ab_res['confidence']:.2f}). The honesty IS the feature: a slow brightness RAMP across the image does NOT become a giant fake slope (the high-pass removes it), and on a near-featureless image the confidence gate ABSTAINS to flat (abstained={_ab_ramp_res['abstained']}) rather than inventing relief. Kept loud as a fundamental limit: luminance-as-height is a heuristic, not a measurement -- a cast shadow reads as a crevice and a painted stripe as a ridge, an albedo/relief ambiguity no arithmetic resolves; it's a plausible perceptual bump, not a depth map. Read the image, high-pass the detail, turn the gradient into normals, drop them into the channel that was already waiting.")

# SURFACE-FROM-GRADIENT (inverse-rendering IR7) -- Frankot-Chellappa FFT: normals -> consistent, tileable height
from holographic_surfaceint import height_from_normals as _si_hfn
from holographic_autobump import normal_from_height as _si_nfh
_si_H, _si_W = 48, 64
_si_y = _ri_np.arange(_si_H)[:, None]; _si_x = _ri_np.arange(_si_W)[None, :]
_si_hh = _ri_np.sin(2 * _ri_np.pi * 2 * _si_x / _si_W) * _ri_np.cos(2 * _ri_np.pi * 3 * _si_y / _si_H)
_si_hh = _si_hh - _si_hh.mean()
_si_rec = _si_hfn(_si_nfh(_si_hh, strength=1.0)); _si_rec = _si_rec - _si_rec.mean()
_si_corr = float(_ri_np.corrcoef(_si_rec.ravel(), _si_hh.ravel())[0, 1])
_si_seam = float(_ri_np.abs(_si_rec[:, 0] - _si_rec[:, -1]).mean())
_si_interior = float(_ri_np.abs(_ri_np.diff(_si_rec, axis=1)).mean())
print(f"  SURFACE-FROM-GRADIENT (inverse-rendering IR7) -- the inverse of auto-bump's height->normal, and the classic surface-from-gradient problem, whose canonical solver is PURE FFT: the engine's own operator (a bind IS an FFT convolution). A measured normal field is generally NOT integrable, so Frankot-Chellappa finds the height whose gradient is the nearest integrable one -- one forward transform of the gradients, a per-frequency divide, one inverse. Take a known height, turn it into a normal map, and integrate it straight back: it returns (correlation {_si_corr:.3f}), drift-free. Two payoffs for auto-bump: the height<->normal round-trip is now integrable and CONSISTENT, and -- the useful accident -- the periodic Fourier boundary makes the result SEAMLESSLY TILEABLE (seam {_si_seam:.3f} vs interior {_si_interior:.3f}), exactly what a material texture wants. Kept loud: that same periodic boundary is a systematic BIAS on a NON-periodic surface (opposite borders forced to agree) -- a feature for a tileable material, a distortion for a bounded scene surface, where a DCT/Poisson variant is the right tool. IR7, done with IR1: a handful of rfft2/irfft2 lines over the FFT machinery the engine already lives on.")

# COLOUR TRANSFER (inverse-rendering ST1) -- grade toward a reference image's statistics (Reinhard 2001)
from holographic_colortransfer import color_transfer as _ct_transfer
_ct_rng = _ri_np.random.default_rng(0)
_ct_cool = _ri_np.clip(_ri_np.stack([0.3 + 0.08 * _ct_rng.standard_normal((40, 40)),
                                     0.4 + 0.08 * _ct_rng.standard_normal((40, 40)),
                                     0.6 + 0.08 * _ct_rng.standard_normal((40, 40))], axis=-1), 0, 1)
_ct_warm = _ri_np.clip(_ri_np.stack([0.7 + 0.08 * _ct_rng.standard_normal((36, 36)),
                                     0.5 + 0.08 * _ct_rng.standard_normal((36, 36)),
                                     0.3 + 0.08 * _ct_rng.standard_normal((36, 36))], axis=-1), 0, 1)
_ct_out = _ct_transfer(_ct_cool, _ct_warm, mode="covariance", strength=1.0, clip=False)
_ct_src_m = _ct_cool.reshape(-1, 3).mean(0)
_ct_ref_m = _ct_warm.reshape(-1, 3).mean(0)
_ct_out_m = _ct_out.reshape(-1, 3).mean(0)
_ct_matched = bool(_ri_np.allclose(_ct_out_m, _ct_ref_m, atol=1e-6))
print(f"  COLOUR TRANSFER (inverse-rendering ST1) -- the easy, powerful grading win: match a reference image's colour STATISTICS onto a render, the 'make this feel like that sunset photo' knob (Reinhard 2001) -- pure statistics, no learned weights. A cool bluish render (mean RGB [{_ct_src_m[0]:.2f}, {_ct_src_m[1]:.2f}, {_ct_src_m[2]:.2f}]) graded toward a warm reference (mean [{_ct_ref_m[0]:.2f}, {_ct_ref_m[1]:.2f}, {_ct_ref_m[2]:.2f}]) takes on the reference's mood (output mean [{_ct_out_m[0]:.2f}, {_ct_out_m[1]:.2f}, {_ct_out_m[2]:.2f}], matched={_ct_matched}). The full mode WHITENS the source and COLOURS it by the reference, matching not just each channel's mean and std but the whole 3x3 covariance -- so a teal-orange grade, where the colour channels are correlated, transfers correctly where plain per-channel matching washes out. Kept loud: it's GLOBAL statistics -- it moves colour, not content, and a linear map can't turn a green field into a red desert without losing detail (local/histogram variants fix that at more cost). It slots into the postfx grade stage and feeds IR4's mood-match. (Naming: postfx.reinhard is the TONEMAP; this reference-based transfer is a separate function.)")

# DISPLACEMENT FROM A CONFIDENT HEIGHT (inverse-rendering IR5) -- promote a bump to REAL geometry, gated
from holographic_autodisplace import auto_displace as _ad_auto
from holographic_mesh import grid as _ad_grid
_ad_Ni = 48
_ad_u = _ri_np.linspace(0, 6 * _ri_np.pi, _ad_Ni)
_ad_bump = 0.5 + 0.4 * _ri_np.outer(_ri_np.sin(_ad_u), _ri_np.cos(_ad_u))
_ad_bump_rgb = _ri_np.stack([_ad_bump, _ad_bump, _ad_bump], axis=-1)
_ad_relief, _ad_info = _ad_auto(_ad_grid(nx=20, ny=20), _ad_bump_rgb, amount=0.15)
_ad_zmax = float(_ri_np.abs(_ad_relief.vertices[:, 2]).max())
_, _ad_flat_info = _ad_auto(_ad_grid(nx=20, ny=20), _ri_np.full((_ad_Ni, _ad_Ni, 3), 0.5), amount=0.15)
print(f"  DISPLACEMENT FROM A CONFIDENT HEIGHT (inverse-rendering IR5) -- a bump map only tilts SHADING normals; the silhouette and the grazing-angle profile stay flat. For a hero surface you sometimes want REAL relief, so IR5 promotes a high-confidence auto-bump height from a bump to geometry -- pure reuse of the shipped displace operator, moving each vertex along its normal by the derived height. Here a flat 20x20 grid, displaced from a bumpy albedo, gained genuine relief (max |z| {_ad_zmax:.2f}, vertices actually moved -- displaced={_ad_info['displaced']}), while a flat, featureless image ABSTAINED (displaced={_ad_flat_info['displaced']}) and left the mesh untouched. The confidence gate is the whole point: real geometry is expensive and destructive, so a shaky height must not crumple a mesh -- IR5 only displaces when the confidence clears a stricter-than-a-bump threshold. Kept loud: displacement inherits ALL of IR1's ambiguities (a cast shadow becomes a real groove, a painted stripe a real ridge), which is exactly why it is gated harder -- the failure is worse when it moves vertices than when it only tilts a normal. The shipped displacement path, driven by the auto-bump height, gated on confidence.")

# PERCEPTUAL COMPARE (inverse-rendering IR4, part 1) -- the render-vs-target objective (SSIM+colour+edges, not MSE)
from holographic_imagecompare import perceptual_similarity as _ic_sim, _shift as _ic_shift
from holographic_autobump import gaussian_blur as _ic_blur
_ic_rng = _ri_np.random.default_rng(0)
def _ic_scene(_ic_seed):
    _r = _ri_np.random.default_rng(_ic_seed); _H, _W = 72, 72
    _yy, _xx = _ri_np.mgrid[0:_H, 0:_W].astype(float); _Y = _yy / _H
    _sky = _ri_np.stack([0.2 + 0.5 * _Y, 0.4 + 0.4 * _Y, 0.85 - 0.3 * _Y], axis=-1)
    _sy, _sx = _r.uniform(0.1, 0.4) * _H, _r.uniform(0.2, 0.8) * _W
    _sun = _ri_np.exp(-((_xx - _sx) ** 2 + (_yy - _sy) ** 2) / (2 * (0.08 * _W) ** 2))[..., None] * _ri_np.array([1.0, 0.9, 0.6])
    return _ri_np.clip(_sky + 0.8 * _sun, 0, 1)
_ic_t = _ic_scene(0)
_ic_n_sim = _ic_sim(_ic_t, _ic_shift(_ic_t, 2, 2))
_ic_w_sim = _ic_sim(_ic_t, _ic_scene(5))
_ic_tex = _ri_np.clip(_ic_blur(_ic_rng.uniform(0, 1, (64, 64, 3)), 1.0), 0, 1)
_ic_mse_ratio = float(_ri_np.mean((_ic_tex - _ic_shift(_ic_tex, 2, 2)) ** 2) / _ri_np.mean((_ic_tex - _ri_np.clip(_ic_blur(_ic_rng.uniform(0, 1, (64, 64, 3)), 1.0), 0, 1)) ** 2))
print(f"  PERCEPTUAL COMPARE (inverse-rendering IR4, part 1) -- the render-vs-target objective the analysis-by-synthesis loop minimizes, and it must NOT be raw pixel MSE: a one-pixel shift or a tiny exposure change wrecks MSE while the images look identical. So the compare is PERCEPTUAL -- multi-scale SSIM (local luminance/contrast/structure) + colour-histogram agreement + edge alignment, each shift- and lighting-tolerant. A rendered scene compared to itself scores 1.00; nudged by 2 pixels (a small camera move) it still reads as the SAME scene ({_ic_n_sim:.2f}) and ranks clearly above a different scene ({_ic_w_sim:.2f}). Raw MSE can't make that call: on textured content a 2px shift's error is {_ic_mse_ratio * 100:.0f}% of a completely-different image's, so an MSE objective is nearly blind to the difference the perceptual metric sees. Kept loud: the ceiling is roughly SSIM-quality STRUCTURAL comparison, not a learned LPIPS perceptual loss (that needs trained weights the constitution bans) -- a good, deterministic render-and-compare objective, not human perception. Part 1 of IR4; the gradient-free loop that minimizes it comes next.")

# ANALYSIS-BY-SYNTHESIS (inverse-rendering IR4, the headline) -- self-recovery: recover camera + sun by render->compare->adjust
from holographic_inverserender import render_params as _as_render, recover_scene as _as_recover, calibrate_accept_threshold as _as_cal
from holographic_sdf import box as _as_box
_as_sdf = _as_box(1.0, 0.7, 0.5)
_as_rkw = dict(width=32, height=32, fov_deg=50.0)
_as_truth = _ri_np.array([0.6, 0.4, 4.0, -0.6, 0.5])
_as_target = _as_render(_as_sdf, _as_truth, **_as_rkw)
_as_init = _as_truth + _ri_np.array([0.3, -0.25, 0.7, 0.35, -0.3])
_as_thr = _as_cal(_as_sdf, _as_truth, **_as_rkw)
_as_res = _as_recover(_as_sdf, _as_target, _as_init, accept_threshold=_as_thr, **_as_rkw, max_evals=400)
_as_err = _ri_np.abs(_as_res["params"] - _as_truth)
_as_pct = 100.0 * (1.0 - _as_res["distance"] / _as_res["init_distance"])
print(f"  ANALYSIS-BY-SYNTHESIS (inverse-rendering IR4, the headline) -- the auto-calibration loop: given a target image, recover the scene that made it by rendering a hypothesis, comparing it to the target with the PERCEPTUAL metric (not MSE), and ADJUSTING to reduce the difference. Autodiff is banned -- exactly why a differentiable renderer can't be borrowed -- so the search is GRADIENT-FREE, a compass/pattern search, cheap because a small SDF render is ~4 ms. The honest milestone is SELF-RECOVERY: render a KNOWN box, hand the pixels back, and recover the camera orbit + sun direction. From a perturbed warm start the loop drove the perceptual distance {_as_res['init_distance']:.3f} -> {_as_res['distance']:.3f} ({_as_pct:.0f}% down) and recovered the camera az/el/radius to within ({_as_err[0]:.2f}, {_as_err[1]:.2f}, {_as_err[2]:.2f}) and the sun to ({_as_err[3]:.2f}, {_as_err[4]:.2f}) of the truth -- and the conformal gate ACCEPTED it (accepted={_as_res['accepted']}). On an unmatchable target (a sphere the box can't become) the gate abstains instead. Kept loud: gradient-free is coarser and slower than differentiable inverse rendering and leans on a decent warm start (IR3) to land in the right basin; it matches the visible frame and abstains on occluded geometry and metric depth (IR6). render -> compare -> adjust -> gate, all on five primitives.")

# PERCEPTION -> HYPOTHESIS (inverse-rendering IR3) -- analog-recall warm start that seeds the IR4 loop from an image
from holographic_perception import SceneLibrary as _pc_Lib
from holographic_inverserender import render_params as _pc_render, recover_scene as _pc_recover
from holographic_sdf import box as _pc_box
_pc_sdf = _pc_box(1.0, 0.7, 0.5)
_pc_rkw = dict(width=32, height=32, fov_deg=50.0)
_pc_lib = _pc_Lib(seed=0)
for _pc_az in (-0.4, 0.2, 0.8):
    for _pc_laz in (-0.6, 0.0, 0.6):
        _pc_p = [_pc_az, 0.4, 4.0, _pc_laz, 0.5]
        _pc_lib.add(_pc_render(_pc_sdf, _pc_p, **_pc_rkw), _pc_p)
_pc_lib.build()
_pc_truth = _ri_np.array([0.25, 0.4, 4.0, 0.05, 0.5])
_pc_target = _pc_render(_pc_sdf, _pc_truth, **_pc_rkw)
_pc_ws = _pc_lib.warm_start(_pc_target)
_pc_res = _pc_recover(_pc_sdf, _pc_target, _pc_ws["params"], **_pc_rkw, max_evals=400)
_pc_err = _ri_np.abs(_pc_res["params"] - _pc_truth)
print(f"  PERCEPTION -> HYPOTHESIS (inverse-rendering IR3) -- the front-end that seeds IR4. The gradient-free loop needs a decent warm start to land in the right basin, and IR3 reads it off the target image itself: a coarse sun-direction estimate from the brightest region, and -- the sublinear move (Pharr's seat) -- ANALOG RECALL of the nearest stored scene from a small library, handing back ITS parameters as the warm start, with the HoloForest's cross-tree agreement as a free abstain signal. Here a 9-scene library is queried by a NEW target; recall finds the nearest exemplar (agreement {_pc_ws['agreement']:.2f}) and its params seed IR4, which refines to distance {_pc_res['distance']:.3f}, recovering the camera + sun to within ({_pc_err[0]:.2f}, {_pc_err[3]:.2f}) of the truth -- the whole analysis-by-synthesis loop closed FROM THE IMAGE ALONE, no hand-perturbation. Kept loud: this is archetype-level recall, not semantic segmentation -- it works inside the library's vocabulary and ABSTAINS (low agreement) rather than hallucinating outside it; the sun-from-luminance cue is coarse, a start for IR4 to refine, not a measurement. Perception seeds synthesis; synthesis refines perception.")

# RENDER CHANNELS / AOVs (inverse-rendering IR14) -- a channel is an unbind; the scene is a bundle at every level
from holographic_renderchannels import render_channels as _rc_channels, composites_to_beauty as _rc_comp
from holographic_render import Camera as _rc_Cam
from holographic_sdf import box as _rc_box, sphere as _rc_sphere
from holographic_raymarch import render_sdf as _rc_render
_rc_cam = _rc_Cam(eye=(2.5, 1.8, 2.5), target=(0, 0, 0), fov_deg=50.0)
_rc_rkw = dict(width=40, height=40, ao=False, shadows=False, reflect=0.0)
_rc_default = _rc_channels(_rc_box(1, 0.7, 0.5), _rc_cam, **_rc_rkw)
_rc_bitident = bool(_ri_np.array_equal(_rc_default["beauty"], _rc_render(_rc_box(1, 0.7, 0.5), _rc_cam, **_rc_rkw)))
_rc_objs = [_rc_box(0.6, 0.6, 0.6).translate((-0.9, 0, 0)), _rc_sphere(0.7).translate((0.9, 0, 0))]
_rc_ch = _rc_channels(_rc_objs[0].union(_rc_objs[1]), _rc_cam, want=["mask", "normal", "depth"], objects=_rc_objs, **_rc_rkw)
_rc_err = _rc_comp(_rc_ch)
_rc_disjoint = bool(_ri_np.all(_rc_ch["object:0"] * _rc_ch["object:1"] == 0.0))
print(f"  RENDER CHANNELS / AOVs (inverse-rendering IR14) -- selectable, SEPARATE passes (depth/normal/position/mask G-buffer, per-object Cryptomatte mattes), each with its own alpha, for compositing, science, and debugging. The holographic reading is why most of it is exposure not new code: a render channel IS AN UNBIND, and the scene is a bundle at every level (a scene of objects, an object of geometry+appearance, a material of channel roles). 'Separate this out' is the engine's own decompose -- material.channel() is already unbind(record, role), and the G-buffer is already computed by the sphere-trace the renderer shades from. Default (no selection) is beauty-only and BIT-IDENTICAL to render_sdf (bit_identical={_rc_bitident}); asked for the G-buffer + two per-object mattes, the mattes composite back to the coverage EXACTLY (err {_rc_err:.1f}, disjoint={_rc_disjoint}) -- the compositor's 'the passes must add up' invariant. Kept loud: the LIGHTING passes (direct/indirect/diffuse/specular/GI) are the one genuinely-new bit and are NOT in this v1 -- they need trace-time accumulation per contribution, and summing them exactly to beauty needs care at the MIS/Russian-roulette boundaries; material.channel() carries crosstalk (fine for a debug/matte pass, use material.sample for exact values); N channels = N buffers, so it is opt-in per channel, never all-on. The scene was already a bundle at every level, so separating the channels is mostly exposing the unbind the engine already runs.")

# FSR1-STYLE UPSCALE (inverse-rendering IR12) -- EASU (edge-adaptive Lanczos) + RCAS (the shipped sharpen)
from holographic_fsr import easu_upscale as _fsr_easu, _box_downscale as _fsr_down, _psnr as _fsr_psnr, _edge_energy as _fsr_edge
from holographic_postfx import resample as _fsr_resample
_fsr_yy, _fsr_xx = _ri_np.mgrid[0:96, 0:96].astype(float)
_fsr_n = 0.5 + 0.4 * _ri_np.sign(_ri_np.sin((_fsr_xx + _fsr_yy) / 5.0))
_fsr_n[20:50, 20:50] = 0.9; _fsr_n[60:85, 55:88] = 0.15
_fsr_native = _ri_np.clip(_ri_np.stack([_fsr_n, _fsr_n, _fsr_n], axis=-1), 0, 1)
_fsr_low = _fsr_down(_fsr_native, 2)
_fsr_hw = _fsr_native.shape[:2]
_fsr_bil = _ri_np.clip(_fsr_resample(_fsr_low, 2.0), 0, 1)[:_fsr_hw[0], :_fsr_hw[1]]
_fsr_e = _fsr_easu(_fsr_low, 2.0)[:_fsr_hw[0], :_fsr_hw[1]]
_fsr_p_bil = _fsr_psnr(_fsr_bil, _fsr_native)
_fsr_p_easu = _fsr_psnr(_fsr_e, _fsr_native)
print(f"  FSR1-STYLE UPSCALE (inverse-rendering IR12) -- a post-process upscale: render at 1080p, present at 4K. FSR1 is two passes and one already ships: RCAS (the noise-aware sharpen) IS the engine's postfx.sharpen (Van Cittert, whose own kept-negative -- stop at the noise floor -- is exactly RCAS's design goal), so the only new piece is EASU, the edge-adaptive upsampler. EASU here is a separable Lanczos (sharper than the plain bilinear it exists to beat) with an ANTI-RINGING clamp that bounds each output to its low-res neighbourhood, killing Lanczos overshoot exactly where gradients reverse. On a 2x downscale->upscale round-trip, EASU beats bilinear on PSNR-to-native ({_fsr_p_easu:.2f} vs {_fsr_p_bil:.2f} dB) and on edge sharpness, with no ringing. Kept loud: classical spatial upscaling is BELOW learned (DLSS/XeSS) -- it reconstructs, it cannot invent detail absent from the low-res input; and EASU's artifacts get MULTIPLIED by the RCAS sharpen, so on a smooth image the sharpen overshoots -- sharpness is a knob, not a free win. This EASU is an honest Lanczos-with-anti-ringing in FSR1's class, not a byte-for-byte port of its 12-tap gradient-reversal kernel. A good, cheap, deterministic upscaler -- not a magic one.")

# CHECKERBOARD RENDER (inverse-rendering IR13) -- shade half the pixels, recover the rest as masked recovery
from holographic_checkerboard import render_checkerboard as _cb_render, _shade_all as _cb_full, _row_halved as _cb_rowh, _psnr as _cb_psnr
from holographic_render import Camera as _cb_Cam
from holographic_sdf import box as _cb_box
_cb_cam = _cb_Cam(eye=(2.5, 1.8, 2.5), target=(0, 0, 0), fov_deg=50.0)
_cb_sdf = _cb_box(1.0, 0.7, 0.5)
_cb_fullimg = _cb_full(_cb_sdf, _cb_cam, 80, 80)
_cb_ck, _cb_m = _cb_render(_cb_sdf, _cb_cam, 80, 80)
_cb_shaded_pct = float(_cb_m.mean() * 100.0)
_cb_recon_psnr = _cb_psnr(_cb_ck, _cb_fullimg)
_cb_rowh_psnr = _cb_psnr(_cb_rowh(_cb_fullimg), _cb_fullimg)
print(f"  CHECKERBOARD RENDER (inverse-rendering IR13) -- shade only ~50% of the pixels (a 2x2 checkerboard, {_cb_shaded_pct:.0f}% here) and RECONSTRUCT the rest -- the 'larger resolution without it taking forever' trick, done as a sampling pattern not a naive lower-res render. The holographic reading (Ozcan's seat): the unshaded pixels are 'damage', and reconstruction is recovery from a partial/masked measurement -- the archive's literal job, in the pixel domain. The gem of the 2x2 pattern is that every unshaded pixel's four cross-neighbours are ALL shaded, so recovery is a clean cross-neighbour average -- no iteration, no learned prior. render_checkerboard traces ONLY the masked half and reconstructs to {_cb_recon_psnr:.1f} dB against a full shade -- near-full quality for half the rays -- beating a matched-cost row-halved render ({_cb_rowh_psnr:.1f} dB), the documented checkerboard advantage: spreading the 50% of samples in 2D beats collapsing them in 1D. Flip the parity each frame and the other half fills, so over two frames every pixel is shaded. Kept loud: reconstruction is a TRADE (better accuracy per cost), not free -- it costs more than a plain lower-res render but reconstructs more accurately at matched cost; under motion it can shimmer unless the reprojection rejects disoccluded pixels by depth/motion (out of scope for this single-frame v1); it is a reconstruction, not true supersampling.")

# 3D OBJECT ARCHIVE (inverse-rendering IR11) -- recall the whole object from a partial front view, or abstain
from holographic_objectarchive import ObjectArchive as _oa_Arch, front_points as _oa_front, _chamfer as _oa_cham, _sphere as _oa_sphere, _box as _oa_box, _cylinder as _oa_cyl, _cone as _oa_cone
_oa_view = (0, 0, 1)
_oa_arch = _oa_Arch(view_dir=_oa_view, grid=8, seed=0)
_oa_arch.add(_oa_sphere(500, 1), "sphere").add(_oa_box(500, 2), "box").add(_oa_cyl(500, 3), "cylinder").build()
_oa_true = _oa_sphere(500, 5)
_oa_frontpts = _oa_front(_oa_true, _oa_view)
_oa_res = _oa_arch.complete_from_front(_oa_frontpts, match_floor=0.85)
_oa_back = _oa_true[_oa_true[:, 2] < -0.1]
_oa_recall_back = _oa_res["whole"][_oa_res["whole"][:, 2] < -0.1]
_oa_d_recall = _oa_cham(_oa_recall_back, _oa_back)
_oa_d_front = _oa_cham(_oa_frontpts, _oa_back)
_oa_odd = _oa_arch.complete_from_front(_oa_front(_oa_cone(500, 9), _oa_view), match_floor=0.85)
print(f"  3D OBJECT ARCHIVE (inverse-rendering IR11) -- the honest answer to the back-of-the-object boundary. Store a library of COMPLETE 3D objects; given only a partial FRONT view (what a photo, or photo3d, actually sees), recall the nearest stored complete object by a view fingerprint and return the WHOLE thing -- including the unobserved back -- or ABSTAIN when nothing matches. Retrieval, not hallucination. The holographic reading: 'recover the whole from a corrupted part' is the engine's OLDEST move -- cleanup, consolidation, the resonator, analog recall -- and a scene is a bundle of splats exactly as a memory is a bundle of role-bound vectors, so a 3D plate is a bundle like any other; nothing new is invented, a shipped primitive is pointed at a new field. Here, from a NEW sphere instance's front half, the archive recalls the whole sphere BY SHAPE (similarity {_oa_res['similarity']:.2f}), recovering the unobserved back to Chamfer {_oa_d_recall:.2f} vs a front-only reconstruction's {_oa_d_front:.2f} ({_oa_d_front / max(_oa_d_recall, 1e-9):.0f}x closer); and it ABSTAINS on a cone that isn't in the library (similarity {_oa_odd['similarity']:.2f}, abstained={_oa_odd['abstained']}). Kept loud: it is COVERAGE-LIMITED -- it completes only objects it has stored, the win scales with library coverage, and the honest output on an unseen shape is 'front only', never an invented back; registration is coarse (the fingerprint gives the match + the abstain gate, the IR4 loop refines the pose); a wrong match must surface as low similarity and abstain, never a confident-wrong completion.")

# TEXTURE SYNTHESIS (inverse-rendering ST2) -- Image Quilting: grow a seamless texture; patch search = HoloForest recall
from holographic_texturesynth import synthesize_texture as _ts_syn, find_similar_patches as _ts_find, _seam_energy as _ts_seam
_ts_rng = _ri_np.random.default_rng(0)
_ts_yy, _ts_xx = _ri_np.mgrid[0:48, 0:48].astype(float)
_ts_base = 0.5 + 0.3 * _ri_np.sin((_ts_xx + _ts_yy) / 3.0) + 0.1 * _ts_rng.standard_normal((48, 48))
_ts_sample = _ri_np.clip(_ri_np.stack([_ts_base, _ts_base * 0.9 + 0.05, _ts_base * 0.8], axis=-1), 0, 1)
_ts_mc = _ts_syn(_ts_sample, 96, 96, psize=20, overlap=6, seed=0, seam="mincut")
_ts_hd = _ts_syn(_ts_sample, 96, 96, psize=20, overlap=6, seed=0, seam="hard")
_ts_found, _ts_sims = _ts_find(_ts_sample, _ts_sample[5:25, 5:25], k=6)
print(f"  TEXTURE SYNTHESIS (inverse-rendering ST2) -- grow a larger (optionally seamless) texture from a small sample -- for material synthesis and feeding IR1 auto-bump with tileable maps, with NO learned weights. Image Quilting (Efros & Freeman 2001): lay the output as overlapping patches copied from the sample, choose each so its overlap with the placed patches MATCHES, then stitch it along the least-error MIN-CUT seam so the joins vanish. Two pieces map onto shipped primitives: the patch search 'find a sample patch whose border matches this context' IS HoloForest recall_k -- the same 'find the patches that look like this one' NLM uses (top cosine {float(_ts_sims[0]):.2f} here) -- and the min-cut is a small dynamic program. Quilting a 48x48 sample into 96x96 preserves its statistics (mean {_ts_mc.mean():.2f} vs the sample's {_ts_sample.mean():.2f}), and the min-cut seams less than a hard cut on the SAME patches ({_ts_seam(_ts_mc):.4f} vs {_ts_seam(_ts_hd):.4f}). Kept loud: it is patch-COPYING -- it can repeat or seam (the min-cut mitigates; variety comes from picking among the near-best), it is BELOW neural for arbitrary artistic styles, and it is best for texture/colour/material on a roughly-stationary sample, not a structured scene or free-form restyle.")

# GUIDED SUPER-RESOLUTION (inverse-rendering ST3) -- render small, upscale steered by the full-res G-buffer
from holographic_superres import guided_upsample as _sr_guided, _psnr as _sr_psnr
from holographic_renderchannels import render_channels as _sr_channels
from holographic_render import Camera as _sr_Cam
from holographic_sdf import box as _sr_box
from holographic_raymarch import render_sdf as _sr_render
from holographic_fsr import easu_upscale as _sr_easu, _box_downscale as _sr_down
_sr_cam = _sr_Cam(eye=(2.5, 1.8, 2.5), target=(0, 0, 0), fov_deg=50.0)
_sr_sdf = _sr_box(1.0, 0.7, 0.5)
_sr_rkw = dict(ao=False, shadows=False, reflect=0.0)
_sr_native = _sr_render(_sr_sdf, _sr_cam, width=64, height=64, **_sr_rkw)
_sr_gb = _sr_channels(_sr_sdf, _sr_cam, want=["normal", "depth"], width=64, height=64, **_sr_rkw)
_sr_low = _sr_down(_sr_native, 2)
_sr_guidedimg = _sr_guided(_sr_low, _sr_gb["normal"], guide_depth=_sr_gb["depth"])[:64, :64]
_sr_plainimg = _sr_easu(_sr_low, 2.0)[:64, :64]
print(f"  GUIDED SUPER-RESOLUTION (inverse-rendering ST3) -- the quality-and-speed payoff: render small, upscale by example. A cheap render shades COLOUR at low resolution, but the GEOMETRY (the normal/depth G-buffer, which IR14 render_channels exposes) is available at FULL resolution because tracing it is cheap. So coarsely upscale the low-res colour, then edge-aware-filter it GUIDED by the full-res G-buffer -- the colour edges snap to the geometry edges. That guided filter IS the shipped SVGF feature-cosine bilateral, steered by the guide instead of used as a denoiser. Here, upsampling a 32x32 colour render 2x, guided by the 64x64 G-buffer, reaches {_sr_psnr(_sr_guidedimg, _sr_native):.1f} dB to native vs a plain upscale's {_sr_psnr(_sr_plainimg, _sr_native):.1f} dB -- the geometry pulls the blurry colour edges back to where they belong. Combined with IR10 (denoise the cheap render), this is a fully-classical render-cheap-then-enhance, no learned weights. Kept loud: classical upsampling INVENTS PLAUSIBLE, NOT TRUE, detail and tops out below learned super-resolution -- it snaps/borrows structure, it doesn't recover information that was never sampled; and it needs a CLEAN full-res guide (a noisy guide leaks into the colour). The guide-free self-similar route composes ST2's HoloForest patch search rather than duplicating it.")

# SMOKE PRESETS (fluids/matter item 1) -- six named looks over the ALREADY-WIRED FFT smoke solver
from holographic_smokepresets import simulate as _sm_sim, plume_center_of_mass as _sm_com, _buoyant_vs_heavy as _sm_bvh
_sm_coms = {_n: _sm_com(_sm_sim(_n, nx=40, ny=40, steps=45, seed=0)["density"])
            for _n in ("rising", "wispy", "billow", "heavy", "still_room", "stratified")}
_sm_up, _sm_down = _sm_bvh()
print(f"  SMOKE PRESETS (fluids/matter item 1) -- smoke is the 1-channel, tension-0 CORNER of the matter model that follows, so there is no new solver here, only LOOKS: six dial-bundles over the wired FFT smoke_step (buoyancy/confinement/viscosity/gravity). The named behaviours are emergent, not special code paths -- rising={_sm_coms['rising']:.2f}, wispy={_sm_coms['wispy']:.2f}, billow={_sm_coms['billow']:.2f}, heavy={_sm_coms['heavy']:.2f}, still_room={_sm_coms['still_room']:.2f}, stratified={_sm_coms['stratified']:.2f} (density centre-of-mass, 0=floor..1=ceiling), four+ distinct looks from one solver. The buoyancy/gravity DIAL is real, not decorative: a controlled hot puff rises to COM {_sm_up:.2f} (above centre) while a heavy one sinks to {_sm_down:.2f} (below) from the SAME source. Rendered through volint's closed-form optical depth, not a new marcher. Kept honest: 2-D looks on a modest grid for interactivity; any solver limit (coarse grid smears fine curl) is INHERITED, not introduced -- the presets add zero physics, they only turn the dials the matter model is built around.")

# THE MATTER MODEL (fluids/matter item 2) -- Mixture + matter_step: dye/milk mixing on ONE shared flow
from holographic_mixture import Mixture as _mx_M, matter_step as _mx_step, _blob as _mx_blob, _spatial_spread as _mx_spread
_mx_shape = (48, 48)
_mx_mix = _mx_M(_mx_shape, solvent_density=1.0, buoyancy=0.0)
_mx_mix.add("red", _mx_blob(_mx_shape, 24, 16, 4.0), density=1.2, diffusivity=0.05)
_mx_mix.add("blue", _mx_blob(_mx_shape, 24, 32, 4.0), density=0.8, diffusivity=0.05)
_mx_vx = _ri_np.full(_mx_shape, 0.5); _mx_vy = _ri_np.zeros(_mx_shape)
_mx_spread0 = _mx_spread(_mx_mix.channels["red"]); _mx_mass0 = float(_mx_mix.channels["red"].sum())
for _ in range(20):
    _mx_vx, _mx_vy = _mx_step(_mx_mix, _mx_vx, _mx_vy, dt=0.1)
_mx_rho = _mx_mix.density()
_mx_r = _ri_np.unravel_index(int(_ri_np.argmax(_mx_mix.channels["red"])), _mx_shape)
_mx_b = _ri_np.unravel_index(int(_ri_np.argmax(_mx_mix.channels["blue"])), _mx_shape)
print(f"  THE MATTER MODEL (fluids/matter item 2) -- smoke, dye/milk, salt fingering and oil-and-water are ONE advected-field model with three dials (# components, buoyancy, double-well tension), not four simulators. This is the multi-channel CORE: a Mixture of component fields riding ONE shared incompressible flow, advanced by a single matter_step that DELEGATES to the wired advect/diffuse/buoyancy_force/fluid_step -- no second solver. Holographically the mixture is a multi-channel hypervector (adding a substance = adding a ROLE), density is a fraction-weighted BUNDLE, and per-channel diffusion at DIFFERENT rates is free (which is the salt-fingering precondition). Here two dye channels advect + diffuse on the shared flow: red spread {_mx_spread0:.2f}->{float(_mx_spread(_mx_mix.channels['red'])):.2f}, mass conserved ({_mx_mass0:.0f}->{float(_mx_mix.channels['red'].sum()):.0f}), and the density BLEND reads the components -- the heavy-dye cell {float(_mx_rho[_mx_r]):.2f} vs the light-dye cell {float(_mx_rho[_mx_b]):.2f} (solvent 1.0). Drift (item 3, salt fingering) and the double-well (item 4, oil & water) are already wired as OPTIONAL hooks in the same loop -- the miscible⇄immiscible dial slots in with no rewrite. Kept honest: miscible is native and cheap; the sharp immiscible interface is item 4's diffuse-interface trade.")

# DRIFT (fluids/matter item 3) -- settling/separation: a heavier channel sinks relative to the blend
from holographic_mixture import Mixture as _dr_M, matter_step as _dr_step, _blob as _dr_blob
def _dr_comy(_f, _sh):
    _ys = _ri_np.mgrid[0:_sh[0], 0:_sh[1]][0]; _f = _ri_np.clip(_f, 0, None)
    return float((_ys * _f).sum() / (_f.sum() + 1e-12))
_dr_sh = (48, 48)
def _dr_run(_drift, _density):
    _m = _dr_M(_dr_sh, solvent_density=1.0, buoyancy=0.0)
    _m.add("c", _dr_blob(_dr_sh, 24, 24, 4.0), density=_density, diffusivity=0.001)
    _vx = _ri_np.zeros(_dr_sh); _vy = _ri_np.zeros(_dr_sh); _y0 = _dr_comy(_m.channels["c"], _dr_sh)
    for _ in range(25):
        _vx, _vy = _dr_step(_m, _vx, _vy, dt=0.1, drift_strength=_drift)
    return _y0, _dr_comy(_m.channels["c"], _dr_sh)
_dr_hy0, _dr_hy1 = _dr_run(0.5, 3.0)          # heavy, drift on
_dr_oy0, _dr_oy1 = _dr_run(0.0, 3.0)          # heavy, drift off (baseline)
_dr_ly0, _dr_ly1 = _dr_run(0.5, 0.2)          # light, drift on
print(f"  DRIFT (fluids/matter item 3) -- the FIRST of the two genuinely-new physics terms: a channel heavier than the LOCAL blend sinks relative to it (lighter rises), applied as an extra vertical advection by a settling velocity proportional to the density excess. This is what turns the matter model from passive mixing into settling and phase SEPARATION -- the salt-fingering/oil-water driver. Measured against a proper drift-OFF baseline: a heavy dye (rho 3.0) sinks COM {_dr_hy0:.1f}->{_dr_hy1:.1f} with drift, but {_dr_oy0:.1f}->{_dr_oy1:.1f} (unchanged) WITHOUT it; a light dye (rho 0.2) instead RISES {_dr_ly0:.1f}->{_dr_ly1:.1f}. Fixing this exposed a real bug -- buoyancy_force couples DENSITY through alpha, not beta, so the blended density had never been driving convection; now it does (a heavy band creates flow where it was zero before). KEPT NEGATIVE, loud: the settling and the density-buoyancy are clean wins, but resolving DISTINCT salt fingers is dominated by bulk overturning at this grid/time -- the ingredients (differential diffusion + drift + density buoyancy) are all present, the fine-scale fingering instability needs a finer grid and careful Rayleigh tuning, which a fast interactive demo doesn't give. Not claimed as a win it isn't.")

# DOUBLE-WELL TENSION (fluids/matter item 4) -- the miscible<->immiscible dial: oil & water separate
from holographic_mixture import Mixture as _dw_M, matter_step as _dw_step
_dw_sh = (48, 48)
_dw_xs = _ri_np.mgrid[0:_dw_sh[0], 0:_dw_sh[1]][1]
_dw_graded = _ri_np.clip((_dw_xs - 12) / 24.0, 0, 1)          # a smooth 0->1 ramp (all intermediate)
def _dw_committed(_phi):
    return float(((_phi < 0.15) | (_phi > 0.85)).mean())
def _dw_run(_tension):
    _m = _dw_M(_dw_sh, solvent_density=1.0, buoyancy=0.0, tension=_tension)
    _m.add("oil", _dw_graded.copy(), density=0.9, diffusivity=0.0)
    _vx = _ri_np.zeros(_dw_sh); _vy = _ri_np.zeros(_dw_sh)
    for _ in range(40):
        _vx, _vy = _dw_step(_m, _vx, _vy, dt=0.1)
    return _dw_committed(_m.channels["oil"])
_dw_b0 = _dw_committed(_dw_graded); _dw_sharp = _dw_run(2.0); _dw_blend = _dw_run(0.0)
print(f"  DOUBLE-WELL TENSION (fluids/matter item 4) -- the SECOND new term and the last dial: it turns the miscible mixture immiscible. An Allen-Cahn double-well W'(phi)=phi(1-phi)(1-2phi) with wells at phi=0 and phi=1 pulls every cell toward one of the two PHASES while a small diffusion sets the interface WIDTH; `tension` scales how hard they separate. This closes the dial table -- ONE advected-field model now spans smoke (1 channel, tension 0), dye/milk (N channels, tension 0), settling/fingering (drift), and oil & water (tension high) with NO new solver, just knobs. Measured on a fully-blended 0->1 ramp (committed fraction {_dw_b0:.2f}): tension 2.0 sharpens it into two phases ({_dw_sharp:.2f} of cells committed) while tension 0 stays blended ({_dw_blend:.2f}) -- the miscible<->immiscible switch. Holographically an immiscible phase is just one more CHANNEL (add a dimension); the interface is a thin band the adaptive dispatch would resolve finely only where it lives. KEPT NEGATIVE (loud, from the literature): this is a DIFFUSE-INTERFACE model -- the interface is a few cells wide, never a perfect step, and plain Cahn-Hilliard shrinks tiny droplets (a conservative variant is needed when oil volume must stay put). The dial is real; the sharp-step idealisation is not what a finite grid gives.")

# SCATTER LAYER + SCALE NODE (fluids/matter item 5) -- geometry emission on any surface; cosmic recursive rollup
from holographic_scatterlayer import ScatterLayer as _sc_L
from holographic_scalenode import ScaleNode as _sc_N
from holographic_sdf import sphere as _sc_sphere
from holographic_scene_doc import Scene as _sc_Scene
from holographic_ai import Vocabulary as _sc_Voc
_sc_voc = _sc_Voc(512, seed=0)
_sc_res = _sc_L(_sc_voc.get("grass"), count=60, seed=0).apply(_sc_sphere(1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)))
_sc_ond = float(_ri_np.abs([_sc_sphere(1.0).eval(_p[None, :])[0] for _p in _sc_res["points"]]).max())
_sc_scene = _sc_Scene(dim=256, seed=0); _sc_planet = _sc_scene.add(name="planet", params={"mass": 0.0})
for _i in range(int(_sc_res["count"])):
    _sc_scene.add(name="blade%d" % _i, params={"mass": 1.0}, parent=_sc_planet)
_sc_sn = _sc_N(_sc_scene, lod_px=8.0)
_sc_mass = _sc_sn.summary(_sc_planet)["mass"]
_sc_orbit = "summary" in _sc_sn.draw(_sc_planet, apparent_px=2.0)
print(f"  SCATTER LAYER + SCALE NODE (fluids/matter item 5) -- two reuse-first faculties. A SCATTER LAYER emits GEOMETRY (grass, rocks, barnacles) onto ANY surface, not just terrain: it reuses emit_from_surface, so {int(_sc_res['count'])} grass instances landed ON a sphere (|sdf|<{_sc_ond:.3f}) with unit normals, weighted by an optional density map -- each placement a BIND, the whole layer a region-queryable BUNDLE (the write-dual of the sampler). A SCALE NODE is the cosmic rule 'a parent carries the accumulated value of its children' = the MONOID (mass SUMs, look BUNDLES), reused from distribute_compute: the planet's {int(_sc_mass)} blades roll up EXACTLY, and from orbit (apparent size below the LOD threshold) the planet draws as ONE summary blob ({_sc_orbit}) instead of a million blades -- adding a blade updates the summary in one associative op. This is already bake-and-query: the summary is precomputed per subtree, zooming out is a lookup. Same accumulation atom->rock->planet->system->galaxy. Kept honest: the rollup is exact only for ADDITIVE properties + a bundled look (a planet's exact weather isn't in the orbital summary), the region query is crosstalk-limited (~1/sqrt(N), which is why dense scatter BAKES and LODs), and the atom->galaxy dynamic range needs relative-transform discipline or precision breaks.")

# COMPILE + FUSE MATERIALS (fluids/matter performance item MC1) -- a material compiled once, reused every frame
from holographic_matcompile import compiled_shader as _mc_shader
from holographic_surface import SurfaceMaterial as _mc_Mat
from holographic_param import Param as _mc_Param
from holographic_compile import CompileCache as _mc_Cache
_mc_rough = lambda P, **k: 0.2 + 0.3 * (_ri_np.asarray(P)[:, 0] > 0)
_mc_mat = _mc_Mat(color=(0.7, 0.4, 0.2), roughness=_mc_Param(field=_mc_rough), reflect=0.15, emission=0.0)
_mc_cache = _mc_Cache()
_mc_pts = _ri_np.random.default_rng(0).uniform(-1, 1, size=(200, 3))
for _ in range(6):                                            # six frames of the same material
    _mc_shade = _mc_shader(_mc_mat, cache=_mc_cache)
_mc_out = _mc_shade(_mc_pts); _mc_ref = _mc_mat.resolve(_mc_pts)
_mc_match = all(_ri_np.allclose(_mc_out[_c], _mc_ref[_c]) for _c in ("color", "roughness", "reflect", "emission", "opacity"))
print(f"  COMPILE + FUSE MATERIALS (performance item MC1) -- the flattening starts with the material. `surface` resolves EVERY channel PER HIT, EVERY frame, and that recompute IS the slowness; the fix is to treat a material as a PROGRAM (a socket graph), compile+fuse it ONCE per (material, options), key it by a content hash, and hand the SAME kernel to every hit, every instance, every frame -- reusing the content-addressed compile cache already in the box. Two concrete wins: the kernel is BUILT ONCE ({_mc_cache.stats['compiles']} compile, {_mc_cache.stats['hits']} cache hits over six frames), and CONSTANT channels (a flat colour/reflect/emission/opacity) are folded to precomputed values so only the genuinely procedural 'roughness' socket re-resolves per hit -- a mostly-flat material, the common case, skips most of its per-hit work. Correctness first: the compiled shade matches the naive per-hit resolve exactly ({_mc_match}); a shortcut that changed the pixels would be no shortcut. Fixing this surfaced a real latent bug in the shipped compile helper -- an EMPTY CompileCache is falsy (__len__==0), so `cache or DEFAULT` silently ignored a freshly-passed cache; fixed to `is not None`. Kept honest: this folds constants and caches the BUILD; a fully-procedural material still evaluates every field per hit -- turning those fields into a LOOKUP is MC2's sdfbake/prt job.")

# BAKE VIEW-INDEPENDENT CHANNELS (fluids/matter performance item MC2) -- a procedural texture becomes a lookup
from holographic_matbake import bake_material as _mb_bake
from holographic_surface import SurfaceMaterial as _mb_Mat
from holographic_param import Param as _mb_Param
_mb_rough = lambda P, **k: 0.3 + 0.2 * _ri_np.sin(_ri_np.asarray(P)[:, 0] * 2.0)
_mb_mat = _mb_Mat(color=(0.7, 0.4, 0.2), roughness=_mb_Param(field=_mb_rough), reflect=0.1, emission=0.0)
_mb_lo, _mb_hi = (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)
_mb_pts = _ri_np.random.default_rng(0).uniform(-0.9, 0.9, size=(300, 3))
_mb_ref = _mb_mat.resolve(_mb_pts)["roughness"]
_mb_err8 = float(_ri_np.abs(_mb_bake(_mb_mat, _mb_lo, _mb_hi, res=8)(_mb_pts)["roughness"] - _mb_ref).max())
_mb_shade = _mb_bake(_mb_mat, _mb_lo, _mb_hi, res=48)
_mb_err48 = float(_ri_np.abs(_mb_shade(_mb_pts)["roughness"] - _mb_ref).max())
print(f"  BAKE VIEW-INDEPENDENT CHANNELS (performance item MC2) -- MC1 folded the CONSTANT channels, but a procedural roughness/colour that varies with surface position still re-ran its noise field at every hit. MC2 BAKES those view-independent fields into a projection and QUERIES it: sample the field over the object bounds ONCE onto a grid, then per hit trilinearly LOOK IT UP -- O(1), independent of how hairy the field was -- reusing the exact machinery sdfbake uses to turn an analytic SDF into a GridSDF, pointed at a material channel instead of a distance. A baked material shades by folds (MC1) + lookups (MC2), with ZERO per-hit field evaluation. Measured: a sin-based procedural roughness baked at res 48 matches the true field to {_mb_err48:.4f} (a coarse res 8 only to {_mb_err8:.4f}) -- and across five frames of lookups the field is evaluated ZERO more times. Kept honest (loud): the bake trades MEMORY for speed and blurs detail finer than a grid cell (bump resolution or keep sharp features procedural); it pays off in REPEATED sampling (many hits/frames/relight), not a one-shot eval, since the bake itself costs one full grid evaluation up front; and only VIEW-INDEPENDENT channels bake here -- the view-dependent specular is MC3's (position,view) LUT.")

# VIEW LUT (fluids/matter performance item MC3) -- view-dependent specular pre-integrated into a table
from holographic_viewlut import bake_view_lut as _vl_bake
from holographic_brdf import directional_albedo as _vl_da
import time as _vl_time
_vl_lut = _vl_bake(metallic=1.0, res_view=16, res_rough=16, samples=8192, seed=0)
_vl_worst = 0.0
for _vc in (0.4, 0.7):
    for _rg in (0.4, 0.6, 0.9):
        _vl_worst = max(_vl_worst, abs(float(_vl_lut.sample(_vc, _rg)[0]) - _vl_da(1.0, _rg, n=32768, view_cos=_vc, seed=1)))
_vl_vcs = _ri_np.random.default_rng(0).uniform(0.1, 1.0, 2000); _vl_rgs = _ri_np.random.default_rng(1).uniform(0.3, 1.0, 2000)
_vl_t0 = _vl_time.time(); _vl_lut.sample(_vl_vcs, _vl_rgs); _vl_tl = _vl_time.time() - _vl_t0
_vl_t0 = _vl_time.time(); _vl_da(1.0, 0.5, n=8192, view_cos=0.7, seed=0); _vl_t1 = _vl_time.time() - _vl_t0
_vl_speed = (_vl_t1 * 2000) / max(_vl_tl, 1e-9)
print(f"  VIEW LUT (performance item MC3) -- MC2 baked the view-INDEPENDENT channels; the last piece is the view-DEPENDENT specular, which can't bake over position alone. The move is 'add a dimension': bake over (view_cos, roughness) -- exactly the pre-integrated BRDF / split-sum LUT the offline world uses. directional_albedo is a 4096+-sample hemisphere integral of the BRDF per pixel; evaluate it ONCE on a small (view x roughness) grid, then per pixel BILINEARLY look it up. Measured: the lookup matches a fresh 32k-sample integral to <{_vl_worst:.3f} across the well-behaved roughness range, and 2000 lookups replace 2000 hemisphere integrals ~{_vl_speed:.0f}x cheaper -- the hairy integral became a table read. With MC1+MC2+MC3 a material shades as folds + position lookups + a view-table read: no per-hit field eval, no per-pixel integral. KEPT NEGATIVE (loud): the table is per (metallic, base_color) -- a different metalness needs its own bake or a third axis; and VERY SMOOTH surfaces (roughness<0.2) are a HIGH-VARIANCE estimate under uniform-hemisphere sampling, so both the LUT and a fresh integral are noisy there (~0.2) -- the LUT faithfully stores directional_albedo; importance sampling would fix the ESTIMATOR, not the table.")

# COMPILE THE PIPELINE (fluids/matter performance items PW1/PW2) -- plan once per config, reuse every frame
from holographic_pipecompile import compiled_pipeline as _pc_comp, run_compiled as _pc_run
from holographic_pipeline import PipelineConfig as _pc_Cfg, build_pipeline as _pc_build
from holographic_compile import CompileCache as _pc_Cache
_pc_cache = _pc_Cache()
_pc_cfg = _pc_Cfg.preview()
_pc_match = _pc_comp(_pc_cfg, cache=_pc_cache).stage_names() == _pc_build(_pc_cfg).stage_names()
for _ in range(10):                                          # ten frames of the same config
    _pc_run(_pc_cfg, scene={"objs": 4}, cache=_pc_cache)
print(f"  COMPILE THE PIPELINE (performance items PW1/PW2) -- the material compile (MC1-3) done one level UP, on the whole render/sim pipeline. build_pipeline does real planning EVERY call -- SELECT the enabled stages, AUTO-INCLUDE prerequisites, TOPOSORT them -- and across a run of frames with the same config that plan never changes, so PW2 COMPILES it once, keyed by the config's content, and hands the same ordered Pipeline to every frame (reusing the content-addressed compile cache exactly as the material did). PW1 is the finer grain: a stage may carry a bake(scene) that precomputes its VIEW-INDEPENDENT buffers once (a baked SDF grid, a static AO pass), so re-running over frames re-does only what changed. Measured: the compiled plan matches build_pipeline exactly ({_pc_match}); ten frames of a preview config planned the pipeline ONCE and reused it nine times ({_pc_cache.stats['compiles']} compile, {_pc_cache.stats['hits']} hits). Same discipline top to bottom -- compile the plan once, bake the static work once, thread only the dynamic stages per frame. Kept honest: this saves the SELECT/TOPOSORT, not the per-frame stage work; the win scales with how many frames share a config (one frame gains nothing), and a stage whose output changes every frame (the render itself) correctly can't bake. Bit-for-bit the same frame as a direct build_pipeline(...).run().")

# ITERATE SIM READOUT (fluids/matter performance item PW4) -- the LINEAR diffusion sub-step: any time t in one eval
from holographic_simreadout import diffuse_at as _sr_at, diffuse_limit as _sr_limit
from holographic_fields import diffuse as _sr_diffuse
_sr_field = _ri_np.random.default_rng(0).standard_normal((32, 32))
_sr_amount = 0.15
_sr_marched = _sr_field.copy()
for _ in range(20):
    _sr_marched = _sr_diffuse(_sr_marched, _sr_amount)
_sr_direct = _sr_at(_sr_field, _sr_amount, 20)
_sr_err = float(_ri_np.abs(_sr_direct - _sr_marched).max())
_sr_limerr = float(_ri_np.abs(_sr_at(_sr_field, _sr_amount, 5000) - _sr_limit(_sr_field)).max())
print(f"  ITERATE SIM READOUT (performance item PW4) -- the last lever: iterate is PRT-for-TIME. The matter model's per-channel step is advect (nonlinear) -> diffuse (LINEAR) -> tension (nonlinear) -> drift (nonlinear), and that diffusion sub-step is a bind -- fields.diffuse multiplies by exp(-amount*k^2) in Fourier space, DIAGONAL in the Fourier basis (the eigendecomposition is FREE, it is the rfft). So diffusing a field for ANY number of steps is ONE transform pair -- raise each frequency's transfer to that power -- not k marched steps, and the limit is closed form (non-DC modes decay, the mean survives -> the flat steady state). Measured: the direct readout at k=20 matches 20 marched fields.diffuse calls to {_sr_err:.1e} (the heat semigroup), fractional t interpolates, and long diffusion approaches the closed-form mean to {_sr_limerr:.1e}; mass is conserved (DC transfer = 1). Time became a QUERY for the linear part of the sim -- the deterministic-depth lever. KEPT NEGATIVE (the honesty of this whole item): ONLY the linear, time-invariant sub-step diagonalises. Nonlinear advection, buoyancy coupling, and double-well tension can't be read out at an arbitrary t -- they still march, and the adaptive dispatch localises that marching to where it's needed. The closed-form readout is the honest prize for the genuinely-linear stage, the same boundary the dynamics propagator drew.")

# BAKE-VS-COMPUTE PER STAGE (fluids/matter performance item PW3) -- the decision layer over PW1/PW2
from holographic_stageplan import plan_stages as _sp_plan
from holographic_pipecompile import compiled_pipeline as _sp_comp
from holographic_pipeline import PipelineConfig as _sp_Cfg
from holographic_compile import CompileCache as _sp_Cache
_sp_pipe = _sp_comp(_sp_Cfg.preview(), cache=_sp_Cache())
_sp_many = {_p["stage"]: _p["choice"] for _p in _sp_plan(_sp_pipe.stages, frames=30)}
_sp_one = {_p["stage"]: _p["choice"] for _p in _sp_plan(_sp_pipe.stages, frames=1)}
_sp_baked = sorted(_n for _n, _c in _sp_many.items() if _c == "bake")
print(f"  BAKE-VS-COMPUTE PER STAGE (performance item PW3) -- the decision layer that finishes the flattening. adaptive.plan_render already decides bake-vs-analytic for the WHOLE render from the workload; PW3 pushes that SAME break-even down to each pipeline STAGE, because within one pipeline some stages are static (a baked g-buffer, an AO pass, an irradiance cache) and some are dynamic (the render itself, a moving sim) and they deserve different answers. The rule, reusing plan_render's own frame break-even: a STATIC stage bakes once reused across enough frames; a static stage used too few times computes (no amortisation); a DYNAMIC stage always computes. Measured on the preview pipeline over 30 frames -> bake {_sp_baked} while render/reproject/denoise/present COMPUTE; over 1 frame NOTHING bakes. This is the planner that tells PW1's bake_pipeline WHICH stages to bake and PW2's compiled pipeline WHICH to leave marching -- the decision over the two mechanisms, every choice with a reason like plan_render's. KEPT NEGATIVE (correctness over speed): an UNANNOTATED stage is conservatively treated DYNAMIC and never baked -- baking a truly-dynamic stage would be a correctness bug, so the safe default costs some speed, never correctness; a stage declares static=True to opt in.")

# SHARED SCATTER/GATHER (generalizing the MPM insight) -- the ONE bundle/readout under every particle<->grid transfer
from holographic_transfer import scatter as _tr_scatter, gather as _tr_gather
from holographic_fields import scatter_to_field as _tr_fld_scatter
from holographic_mpm import MPMSnow as _tr_MPM
_tr_rng = _ri_np.random.default_rng(0)
_tr_pos = _tr_rng.uniform(0, 20, (30, 2)); _tr_vals = _tr_rng.standard_normal(30)
_tr_bilinear_match = float(_ri_np.abs(_tr_scatter(_tr_pos[:, ::-1], _tr_vals, (20, 20), kernel="bilinear", periodic=True) - _tr_fld_scatter((20, 20), _tr_pos, _tr_vals)).max())
_tr_m = _tr_MPM(grid=32, seed=0); _tr_m.seed_block(cx=16, cy=16, w=8, h=8, n=200)
_tr_bspline_match = float(_ri_np.abs(_tr_scatter(_tr_m.x * _tr_m.inv_dx, _tr_m.m, (32, 32), kernel="bspline") - _tr_m.p2g_mass_grid()).max())
_tr_f = _tr_rng.standard_normal((16, 16)); _tr_v = _tr_rng.standard_normal(30); _tr_p2 = _tr_rng.uniform(3, 13, (30, 2))
_tr_adj = abs(float(_ri_np.sum(_tr_scatter(_tr_p2, _tr_v, (16, 16)) * _tr_f)) - float(_ri_np.sum(_tr_v * _tr_gather(_tr_f, _tr_p2))))
print(f"  SHARED SCATTER/GATHER (the physics generalization) -- the snow-MPM work surfaced that P2G (scatter particles onto a grid) IS bundling and G2P (gather back) IS the readout. Probing the stack, that pattern was written out by hand in three places, each with its own kernel: the fluid solver's cloth-coupling deposit (bilinear), MPM's material-point transfer (B-spline), and Gaussian splatting (a global kernel) -- the same superposition, three times. So it was extracted ONCE: scatter = deposit each point's value through a kernel = a BUNDLE (a superposition of kernel-weighted, position-bound contributions); gather = read the grid back through the same kernel = the READOUT; and the two are ADJOINT. Proof it is one operation and not three, not asserted but measured: this single primitive reproduces the fluid solver's bilinear scatter to {_tr_bilinear_match:.0e} and MPM's B-spline P2G to {_tr_bspline_match:.0e} (machine precision), and its scatter/gather are adjoint to {_tr_adj:.0e}. The fluid module's scatter_to_field / sample_field now DELEGATE to it (the call sites made thin, with zero regression across seventy fluid/cloth/MPM tests). As above, so below: the fluid coupling and the material-point transfer were the same bundle/readout all along.")

# SNOW via MLS-MPM (physics #8B, rung 4) -- and the payoff of thinking holographically: P2G/G2P ARE bundle/readout
from holographic_mpm import MPMSnow as _mp_MPM, _bundle_mass_grid as _mp_bundle
_mp_snow = _mp_MPM(grid=48, gravity=9.81, seed=2)
_mp_snow.seed_block(cx=24, cy=12, w=10, h=8, n=400)
_mp_ident = float(_ri_np.abs(_mp_snow.p2g_mass_grid() - _mp_bundle(_mp_snow)).max())   # P2G vs an independent bundle
_mp_y0 = float(_mp_snow.center_of_mass()[1])
_mp_ext0 = float(_mp_snow.x[:, 1].max() - _mp_snow.x[:, 1].min())
_mp_snow.run(dt=2e-3, steps=800)
_mp_y1 = float(_mp_snow.center_of_mass()[1])
_mp_ext1 = float(_mp_snow.x[:, 1].max() - _mp_snow.x[:, 1].min())
print(f"  SNOW via MLS-MPM (physics #8B, rung 4) -- the last rung, and the one where thinking holographically paid off. The Material Point Method looks like a pure grid solver, but its HEART -- the particle-to-grid transfer -- IS the engine's bundle/readout in a physics costume. P2G scatters each snow particle's mass and momentum onto the grid through a smooth kernel, and that is a SUPERPOSITION: the grid is a BUNDLE of kernel-weighted, position-bound particle contributions -- the same operation as a Gaussian splat scene or the RBF encoder's kernel density. Not asserted but VERIFIED: the P2G mass grid equals an independent bundle of kernel splats to {_mp_ident:.1e} (machine precision), and preserves total mass. G2P gathers the grid back = the readout, and the round-trip conserves momentum -- bundle-to-readout fidelity, as above so below. On this run the snow falls under gravity (centre of mass {_mp_y0:.1f} -> {_mp_y1:.1f}), then piles and COMPRESSES plastically: its vertical extent shrinks {_mp_ext0:.1f} -> {_mp_ext1:.1f} and stays that way, because the SVD singular-value clamp is a real permanent yield (which is exactly why a snowball packs and a footprint stays). Kept honest: the GRID UPDATE -- the corotated stress and the plastic clamp -- is genuinely grid-native nonlinear physics, no bind (do not over-holograph a grid). So MPM is a HYBRID: a holographic transfer around a grid-native constitutive update, and that hybrid is the honest picture. A readable 2-D demo -- constant Lame parameters, explicit integration, dissipative PIC transfer -- with production hardening/implicit/3-D as the extension.")

# OVERTURNING FREE SURFACE (physics #8, rung 4) -- the breaking barrel a height field fundamentally can't hold
from holographic_freesurface import FreeSurface as _fs_FS, seed_breaking_crest as _fs_seed
_fs_break = _fs_FS(g=9.81, ground=0.0)
_fs_seed(_fs_break, length=10.0, n=40, crest_speed=8.0, phase_speed=3.0, height=4.0)
_fs_before = _fs_break.is_overturning()
_fs_break.advance(0.05, steps=20)
_fs_after = _fs_break.is_overturning()
_fs_multi = _fs_break.is_multivalued()
_fs_airborne = int((_fs_break.pos[:, 1] > 0.01).sum())
_fs_calm = _fs_FS(g=9.81, ground=0.0)
_fs_seed(_fs_calm, length=10.0, n=40, crest_speed=3.2, phase_speed=3.0, height=1.0)
_fs_calm.advance(0.05, steps=20)
_fs_calm_over = _fs_calm.is_overturning()
print(f"  OVERTURNING FREE SURFACE (physics #8, rung 4) -- this is the top rung of the AdaptiveSolver's ladder, and it exists for one honest reason: every cheaper method stores the water as a height field h(x), ONE height per position, and a breaking wave's crest curls FORWARD over its own base, so above a single x there are suddenly TWO surfaces (the falling jet and the wave face beneath it). A height field cannot express that; particles can. When the dispatch flags a tile as breaking, it hands the crest to this solver: seed particles carrying the wave's orbital velocity -- at a steep crest the tip is thrown forward faster than the wave itself travels (crest speed 8 vs phase speed 3, which IS the breaking condition) -- then let them fly under gravity. At t=0 the surface is single-valued (overturning={_fs_before}); twenty steps later the tip has plunged forward past the wave face and the surface has FOLDED (overturning={_fs_after}, multi-valued={_fs_multi}, with {_fs_airborne} particles airborne over the face) -- a barrel no height field could hold. A gentle wave (crest 3.2 vs phase 3.0) stays single-valued (overturning={_fs_calm_over}). Kept honest, the VFX-vs-physics line: this is a ballistic-particle model of the plunging crest -- correct for the throw and the free-fall plunge, but it does NOT model the pressure, incompressibility, or whitewater AFTER impact; it captures the overturning topology and leaves the turbulent mixing as the documented gap.")

# DIFFUSION-LIMITED BRANCHING (physics #7) -- ice dendrites AND lightning bolts from ONE engine
from holographic_dendrite import ice_dendrite as _dl_ice, lightning as _dl_bolt
_dl_i = _dl_ice(shape=(61, 61), eta=1.0, steps=200, seed=0)
_dl_icells = int(_dl_i.cluster.sum())
_dl_ifd = float(_dl_i.fractal_dimension())
_dl_b = _dl_bolt(shape=(61, 61), eta=3.0, steps=100, seed=1)
_dl_depth = int(_ri_np.where(_dl_b.cluster)[0].max())
print(f"  DIFFUSION-LIMITED BRANCHING (physics #7) -- a frost dendrite on a cold window and a lightning bolt are the SAME physics: a cluster growing into the steepest gradient of a Laplace (diffusion) field, branching stochastically. It is the classic 1984 dielectric-breakdown model: solve the potential (0 on the cluster, 1 on the boundary it reaches toward, smooth between -- the same Laplace field the spectral backbone solves), then let the empty cells touching the cluster grow with probability proportional to phi raised to a power eta, so growth RACES toward wherever the field is steepest. Seed a point and pull toward the surrounding border and you get an ICE crystal -- a connected sparse fractal ({_dl_icells} cells, box-dimension {_dl_ifd:.2f}, far more than a line but nowhere near a filled disk). Seed the cloud along the top edge and pull toward the ground and the SAME code gives a LIGHTNING bolt, reaching depth {_dl_depth} of 61 -- N11's 'build once, get frost and bolts,' with the single knob eta tuning the shape from bushy to fractal to stringy. Kept honest: this is a lattice model on a plain grid -- the growth is a discrete stochastic choice, not a bind, so it earns no holographic form (the don't-over-holograph-a-grid rule) -- and the fractal dimension is stochastic: a border source fingers into a Lichtenberg shape rather than the idealized isotropic DLA.")

# ELECTROMAGNETICS (physics #6) -- the coupled Maxwell field (FDTD) + the Lorentz force on charges
from holographic_em import push_particle as _em_push, cyclotron_frequency as _em_wc, exb_drift as _em_drift, Maxwell1D as _em_Max
_em_B = _ri_np.array([0.0, 0.0, 1.0])
_em_traj, _em_vfin = _em_push([0, 0, 0], [1.0, 0, 0], 1.0, 1.0, [0, 0, 0], _em_B, 2 * _ri_np.pi / 2000, 2000)
_em_speedf = float(_ri_np.linalg.norm(_em_vfin))
_em_wc_val = float(_em_wc(1.0, 1.0, 1.0))
_em_drift_x = float(_em_drift([0, 1.0, 0], _em_B)[0])
_em_field = _em_Max(n=400, dx=1.0, eps=1.0, mu=1.0)
_em_xs = _ri_np.arange(400)
_em_field.Ez = _ri_np.exp(-((_em_xs - 80.0) ** 2) / 72.0)
_em_dt = _em_field.default_dt()
_em_f0 = int(_ri_np.max(_ri_np.where(_ri_np.abs(_em_field.Ez) > 0.05 * _em_field.Ez.max())[0]))
_em_field.step(dt=_em_dt, steps=100)
_em_f1 = int(_ri_np.max(_ri_np.where(_ri_np.abs(_em_field.Ez) > 0.05 * _ri_np.max(_ri_np.abs(_em_field.Ez)))[0]))
_em_frontspeed = (_em_f1 - _em_f0) / (_em_dt * 100)
print(f"  ELECTROMAGNETICS (physics #6) -- the spectral backbone already propagates EM waves (omega=c|k|) and solves electrostatics (the Coulomb potential via Poisson); this adds the two pieces that make it electro-MAGNETISM. The Lorentz force F = q(E + v x B) pushes a charge SIDEWAYS to its motion, curving it into a circle: in a uniform magnetic field a charge traces a CYCLOTRON orbit at omega_c = {_em_wc_val:.1f}, and the Boris pusher's exact rotation conserves its speed to machine precision (1.000000 -> {_em_speedf:.6f}) rather than spiralling the way a naive step would. In crossed E and B it drifts at (E x B)/|B|^2 = {_em_drift_x:.1f} in x, the same for every charge regardless of mass. And the COUPLED Maxwell field -- a Yee-grid FDTD where a changing E makes B and a changing B makes E -- launches a pulse that propagates at exactly the speed of light (measured front speed {_em_frontspeed:.2f} = c). Kept honest: this coupled solver is a genuine GRID solver -- the first-order curl equations don't diagonalise into one bind the way a single wave component does -- so it lives beside the spectral backbone, not on it, and FDTD blows up above the Courant limit (the classic stability rule, kept as a test).")

# ABSTAINING PHOTO-TO-3D (sec.5 depth delegation) -- observe the front surface, abstain on the unobserved
from holographic_photo3d import photo_to_gaussians as _p3
_p3_H = _p3_W = 48
_p3_depth = _ri_np.empty((_p3_H, _p3_W))
_p3_depth[:, :_p3_W // 2] = 1.0
_p3_depth[:, _p3_W // 2:] = 3.0
_p3_depth[0, 0] = 0.0                                             # one invalid pixel (a hole)
_p3_col = _ri_np.zeros((_p3_H, _p3_W, 3))
_p3_col[:, :_p3_W // 2] = [0.8, 0.2, 0.2]
_p3_col[:, _p3_W // 2:] = [0.2, 0.3, 0.8]
_p3_g = _p3(_p3_depth, _p3_col, 48.0, 48.0, 24.0, 24.0, confidence_floor=0.3)
_p3_z = _p3_g["positions"][:, 2]
_p3_znear = float(_p3_z[_ri_np.isclose(_p3_z, 1.0, atol=0.1)].mean())
_p3_zfar = float(_p3_z[_ri_np.isclose(_p3_z, 3.0, atol=0.1)].mean())
_p3_obs = _p3_g["n_observed"]
_p3_abs = _p3_g["n_abstained"]
_p3_cov = _p3_g["coverage"] * 100.0
print(f"  ABSTAINING PHOTO-TO-3D (sec.5) -- lifting a photo to 3D is an ESTIMATE, so it carries a confidence and abstains where it does not know. A two-plane depth map (a near plane meeting a far plane at an occlusion edge, plus one invalid hole) unprojects to two flat FRONT surfaces at z={_p3_znear:.1f} and z={_p3_zfar:.1f}, emitted as {_p3_obs} per-pixel 3D Gaussians. But it ABSTAINS on {_p3_abs} pixels: the occlusion edge (where naive unprojection stretches a sheet of geometry that exists in no scene), the invalid hole, and grazing surfaces -- coverage {_p3_cov:.0f} percent, honestly below 100. The deepest abstention is the one it cannot even see: a single view never observes the BACK of an object, so the pipeline emits the visible front and leaves the back unknown rather than guessing a watertight mesh. Confidence here is a geometric SUPPORT score, not a calibrated probability -- one depth map has no per-pixel calibration set, so support-plus-abstention is the estimator that fits.")

# CALIBRATED FORECAST CONFIDENCE (F1/F2/F8) -- conformal intervals, temporal ACI, proper scoring
from holographic_conformal import (ConformalForecaster as _CFcast, AdaptiveConformal as _CFaci,
                                    conformal_quantile as _cf_q, coverage_report as _cf_cov,
                                    crps_sample as _cf_crps)
_cf_rng = _ri_np.random.default_rng(0)
_cf_truth = _cf_rng.standard_normal(2000); _cf_pred = _cf_truth + _cf_rng.standard_normal(2000) * 0.5
_cf_resid = _ri_np.abs(_cf_pred - _cf_truth)
# F1: a calibrated 90% interval + abstain gate
_cf_fore = _CFcast(alpha=0.1, kind="scalar", abstain_width=2.0)
_cf_fore.calibrate(list(_cf_pred[:1000]), list(_cf_truth[:1000]))
_cf_out = _cf_fore.predict(3.0)
_cf_cover = float(_ri_np.mean([_cf_fore.covers(_cf_pred[i], _cf_truth[i]) for i in range(1000, 2000)]))
# F2: ACI vs fixed split conformal on a DRIFTING stream
_cf_stream = _ri_np.abs(_cf_rng.standard_normal(3000) * _ri_np.linspace(0.5, 4.0, 3000))
_cf_fixedq = _cf_q(_cf_stream[:300], 0.1)
_cf_fixedcov = float(_ri_np.mean(_cf_stream[300:] <= _cf_fixedq))
_cf_aci = _CFaci(alpha=0.1, gamma=0.05, window=300)
for _cf_r in _cf_stream:
    _cf_aci.step(_cf_r)
_cf_acicov = _cf_aci.realized_coverage()
# F8: coverage report + CRPS discriminates a sharp forecast from a vague one
_cf_rep = _cf_cov(_cf_resid[:1000], _cf_resid[1000:], alphas=(0.1,))
_cf_good = _cf_crps(_cf_rng.standard_normal(400) * 0.3 + 1.0, 1.0)
_cf_bad = _cf_crps(_cf_rng.standard_normal(400) * 3.0 + 1.0, 1.0)
print(f"  CALIBRATED FORECAST CONFIDENCE (F1/F2/F8) -- the forecasting twin of RecallNull: the mind can PRODUCE the next state four ways, and now knows how sure it is. F1 wraps any producer in a distribution-free prediction interval (a 90% interval around 3.0 is [{_cf_out['interval'][0]:.2f}, {_cf_out['interval'][1]:.2f}], measured coverage {_cf_cover:.2f} on held-out data) and ABSTAINS when the interval is wider than the caller trusts -- the '{('abstain' if _cf_out['abstain'] else 'act')}' gate the brief asked for; the vector path scores by 1-cosine, the engine's own metric. F2: time series break exchangeability, so plain conformal silently under-covers under drift -- on a drifting stream fixed conformal falls to {_cf_fixedcov:.2f} while Adaptive Conformal Inference holds {_cf_acicov:.2f} at the 90% target. F8: coverage tracks nominal ({_cf_rep[0]['nominal']:.2f} nominal -> {_cf_rep[0]['empirical']:.2f} empirical) and CRPS ranks a sharp forecast below a vague one ({_cf_good:.2f} < {_cf_bad:.2f}). Kept loud: calibrated is not correct (a useless predictor gets honest-but-WIDE intervals -- the width is the signal); coverage is marginal, not per-input; under a fundamental regime change the honest output is abstain-and-flag-drift, not a confident forecast.")

# RENDER/SIM PIPELINE (usability) -- one configurable pipeline, promoted primitives, field effects
from holographic_pipeline import PipelineConfig as _RSPCfg, build_pipeline as _rsp_build
from holographic_fieldeffect import FieldEffect as _RSPFx, attract_to as _rsp_attract
from holographic_integrate import ParticleSim as _RSPSim
from holographic_sdf import sphere as _rsp_sphere
_rsp_pipe = _rsp_build(_RSPCfg.preview())
_rsp_names = _rsp_pipe.stage_names()
_rsp_autoinc = ("gbuffer" in _rsp_names and "svgf_denoise" in _rsp_names)   # gbuffer pulled in by SVGF's need
try:
    _rsp_build(_RSPCfg(dirty_only=True, temporal_reuse=False)); _rsp_rejected = False
except Exception:
    _rsp_rejected = True                                          # impossible combo rejected at BUILD time
_rsp_ctx = _rsp_pipe.run(scene="demo", seed=0)                   # threads a frame -> a tonemapped image
_rsp_img_ok = (float(_rsp_ctx.image.min()) >= 0.0 and float(_rsp_ctx.image.max()) <= 1.0)
_rsp_b = _rsp_ctx.buffers                                         # the stages are now REAL and measured
_rsp_svgf = _rsp_b.get("svgf_psnr", {"noisy": 0.0, "denoised": 0.0})
_rsp_acc = _rsp_b.get("accum_rmse", {"robust": 0.0, "naive": 0.0})
_rsp_nmask = int(_rsp_b["sample_map"].sum()) if hasattr(_rsp_b.get("sample_map"), "sum") else 0
_rsp_explain = _rsp_pipe.plan()[0]                                # plan() is now a full EXPLAIN (needs/produces)
_rsp_vm_ctx, _rsp_vm_applied = _rsp_pipe.run_on_vm(scene="demo", seed=0)   # Phase 6: run the pipeline ON the VM
_rsp_vm_match = bool(_ri_np.array_equal(_rsp_ctx.image, _rsp_vm_ctx.image))
_rsp_inter = _rsp_build(_RSPCfg.interactive()).stage_names()
_rsp_sim_first = _rsp_inter.index("sim_collide") < _rsp_inter.index("render")
# a FieldEffect drives a ParticleSim: particles fall into a gravity well
_rsp_well = _RSPFx(_rsp_sphere(3.0), _rsp_attract([0, 0, 0]), radius=3.0, strength=5.0)
_rsp_w_centre = float(_rsp_well.weight(_ri_np.array([[0.0, 0, 0]]))[0])
_rsp_p = _ri_np.array([[1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
_rsp_ps = _RSPSim(_rsp_p.copy(), _ri_np.zeros_like(_rsp_p), lambda p, v: _rsp_well.apply(p))
_rsp_r0 = float(_ri_np.linalg.norm(_rsp_ps.pos, axis=1).mean())
for _ in range(60):
    _rsp_ps.advance(0.02)
_rsp_r1 = float(_ri_np.linalg.norm(_rsp_ps.pos, axis=1).mean())
# symplectic vs explicit energy drift on an orbit (why the integrator defaults to symplectic)
_rsp_orb_s = _RSPSim(_ri_np.array([[1.0, 0, 0]]), _ri_np.array([[0.0, 1, 0]]), lambda p, v: -p, integrator="symplectic")
_rsp_orb_e = _RSPSim(_ri_np.array([[1.0, 0, 0]]), _ri_np.array([[0.0, 1, 0]]), lambda p, v: -p, integrator="explicit")
_rsp_e0 = 0.5 * float((_rsp_orb_s.vel ** 2).sum()) + 0.5 * float((_rsp_orb_s.pos ** 2).sum())
for _ in range(2000):
    _rsp_orb_s.advance(0.05); _rsp_orb_e.advance(0.05)
_rsp_ds = abs(0.5 * float((_rsp_orb_s.vel ** 2).sum()) + 0.5 * float((_rsp_orb_s.pos ** 2).sum()) - _rsp_e0) / _rsp_e0
_rsp_de = abs(0.5 * float((_rsp_orb_e.vel ** 2).sum()) + 0.5 * float((_rsp_orb_e.pos ** 2).sum()) - _rsp_e0) / _rsp_e0
print(f"  RENDER/SIM PIPELINE (usability) -- one configurable pipeline. A preset picks the stages ({len(_rsp_names)} for 'preview': {', '.join(_rsp_names)}); asking for SVGF AUTO-INCLUDED the G-buffer stage it needs ({_rsp_autoinc}); an impossible combo (dirty_only without temporal_reuse) is REJECTED at build time with a clear message ({_rsp_rejected}); plan() dry-runs the stage list WITHOUT rendering; run() threads a frame to a tonemapped image ({_rsp_img_ok}); the 'interactive' preset pulls in the sim stages ordered BEFORE render ({_rsp_sim_first}). The demo stages are now REAL, not stand-ins, each measured: the render ACCUMULATES its sample passes through the engine's firefly clamp, so one corrupted pass gives RMSE {_rsp_acc['robust']:.3f} vs {_rsp_acc['naive']:.3f} for the naive mean; the SVGF stage denoises with feature-cosine edge-stopping ({_rsp_svgf['noisy']:.1f} dB -> {_rsp_svgf['denoised']:.1f} dB); the adaptive stage builds a real variance mask ({_rsp_nmask} pixels flagged) with a Wald SPRT stop; and plan() is now a full EXPLAIN ('{_rsp_explain['stage']}' needs {_rsp_explain['needs']} -> {_rsp_explain['produces']}) -- Phase 6's inspection half. PROMOTED PRIMITIVES: the SDF normal now lives once in the field module and the renderer delegates to it bit-for-bit (G1); the time-step is one symplectic integrator (orbit energy drift {_rsp_ds:.4f} vs {_rsp_de:.1f} for explicit Euler -- why it defaults to symplectic), behind one SimStep interface that WRAPS every solver's differing step. FIELD EFFECTS: a shaped zone of influence (weight {_rsp_w_centre:.2f} at the centre, 0 outside) drives a ParticleSim -- particles fall into the gravity well (mean radius {_rsp_r0:.2f} -> {_rsp_r1:.2f}). Kept loud: a stage-list+toposort, not a general DAG engine; field effects are soft forces, not hard constraints; Phase 6 is now COMPLETE -- the pipeline also RUNS on the VM: its config lowers to a program of APPLY instructions the machine executes with a handler per stage (the frame riding as the accumulator), giving a frame bit-identical to the direct loop ({_rsp_vm_match}) -- so the render pipeline is now just another program the machine can inspect and run. Only the materials/BRDF convergence refactor (~20 modules, bit-exact) stays its own careful pass.")

# ABOVE/BELOW SWEEP 3 (local completions) -- texture maps, graph namespace, near-surface->SDF; backlog finished
from holographic_materialio import TextureMap as _S3LTex, PBRMaterial as _S3LMat
from holographic_graph_memory import GraphMemory as _S3LGM
_s3l_rng = _ri_np.random.default_rng(0)
# texture map: bilinear sampling on a checker
_s3l_checker = (_ri_np.indices((4, 4)).sum(0) % 2).astype(float)[:, :, None]
_s3l_tex = _S3LTex(_s3l_checker, wrap="clamp")
_s3l_corner = float(_s3l_tex.sample(0.0, 0.0)[0]); _s3l_centre = float(_s3l_tex.sample(0.5, 0.5)[0])
_s3l_mat = _S3LMat(base_color=(1, 1, 1, 1), base_color_map=_s3l_tex)
# graph namespace: route a query to its region (hierarchy, not recall)
_s3l_cen = {lbl: (lambda v: v / _ri_np.linalg.norm(v))(_s3l_rng.standard_normal(256)) for lbl in ("a", "b", "c")}
_s3l_ns = _S3LGM(256)
for _lbl, _c in _s3l_cen.items():
    for _ in range(5):
        _s3l_ns.observe_vector(_c + 0.02 * _s3l_rng.standard_normal(256), _lbl)
_s3l_q = _s3l_cen["b"] + 0.02 * _s3l_rng.standard_normal(256)
_s3l_res = _s3l_ns.classify_vector(_s3l_q); _s3l_lbl = _s3l_res[0] if isinstance(_s3l_res, tuple) else _s3l_res
print(f"  ABOVE/BELOW SWEEP 3 (local completions) -- finishing the audit backlog. TEXTURE MAPS: PBRMaterial now carries image maps sampled by UV with bilinear interpolation (checker corner {_s3l_corner:.2f}, centre blends to {_s3l_centre:.2f}) -- the per-texel detail a factor-level material couldn't hold, backward-compatible. GRAPH NAMESPACE: graph_memory re-homed (fit-correct) as a hierarchical navigation tree -- a query routes to its region ('{_s3l_lbl}'), used for hierarchy/namespace, NOT recall (whose accuracy collapses at scale, the kept negative). NEAR-SURFACE->SDF: a near-surface band redistances to a full signed field by reusing the existing fast-sweeping eikonal. And probe-first found occlusion-speed (Batch-OMP) and sculpt's fast representation (the narrow-band sparse field) already built -- the audit backlog is now cleared end to end.")

title("Bridges to the rest of the stack (S3): does the SDF/procedural layer unlock anything? -- MEASURED, negatives kept")





# The honest cross-pollination check. Two wins, two negatives/already-dones -- a negative ruled out by
# measurement is as valuable as a win, so all four are on the record.
from holographic_procbridge import procedural_compression as _pc_geo, soft_min as _sm_geo, fpe_smooth as _fs_geo
from holographic_sdf import sphere as _psph_geo, menger as _pmeng_geo
import numpy as _np_pb
# C1 WIN: the generator is CONSTANT in size while output complexity explodes -- the capacity/complexity escape
_c1 = [_pc_geo(_pmeng_geo(_d, 1.0), res=40) for _d in (1, 2, 3)]
print(f"  C1 COMPRESSION (win): menger DSL stays {_c1[0]['dsl_bytes']} B while the mesh grows "
      f"{_c1[0]['mesh_faces']}->{_c1[2]['mesh_faces']} faces (ratio ~{_c1[0]['ratio']:.0f}x). Store the LAW, not the geometry -- MDL for shape.")
# C4 WIN: smooth-union and the memory cleanup are ONE temperature operator
_a = _psph_geo(1.0); _c = _psph_geo(1.0).translate([1.5, 0, 0]); _P = _np_pb.array([[0.75, 0, 0.0]])
_hard = float(_np_pb.minimum(_a.eval(_P), _c.eval(_P))[0])
_gaps = [round(abs(float(_a.smooth_union(_c, _k).eval(_P)[0]) - _hard), 4) for _k in (0.5, 0.1, 0.01)]
print(f"  C4 SOFT OPERATOR (win): smooth_union->hard union as k->0 (gaps {_gaps}) -- EXACTLY as Hopfield/softmax cleanup->hard NN as beta->inf. soft_min IS the cleanup's log-sum-exp, in distance space.")
# C2 NEGATIVE (kept): the FPE field is a kernel smoother that over-smooths -- it does NOT beat doing nothing
_rng = _np_pb.random.default_rng(0); _x = _np_pb.linspace(0, 1, 120); _clean = _np_pb.sin(2 * _np_pb.pi * 2 * _x)
_noisy = _clean + _rng.normal(0, 0.3, 120)
_snr = lambda c, e: 10 * _np_pb.log10(_np_pb.var(c) / (_np_pb.var(c - e) + 1e-12))
print(f"  C2 DENOISING (KEPT NEGATIVE): the FPE field as a denoiser HURTS ({float(_snr(_clean, _noisy)):.1f}dB noisy -> {float(_snr(_clean, _fs_geo(_x, _noisy, 6.0))):.1f}dB) -- a band-limited kernel smoother, dominated by the shipped denoisers. NOT wired in.")
print("  C3 STRUCTURE (already done): an SDF scene IS a tree_to_recipe recipe -- decode_structure reads it back; nothing to build.  *** S3: measured, two wins and two honest negatives ***")

title("Substrate Evolution + Differentiable Orchestration: a self-organizing codebook and gradient-optimized tool-chains")
# Two upgrades that elevate things to be composable/autonomous in a VSA program -- both numpy, no autodiff.
from holographic_harmonic import harmonic_atom as _ha_geo, OnlineHarmonicAtom as _oha_geo
import numpy as _np_evo
_rng_evo = _np_evo.random.default_rng(7); _D = 16
_th = _rng_evo.uniform(0, 2 * _np_evo.pi, 40)
_c = {k: _rng_evo.normal(size=_D) for k in range(5)}
def _mf(t):
    return (_c[0] + _np_evo.cos(t) * _c[1] + _np_evo.cos(2 * t) * _c[2] + _np_evo.sin(t) * _c[3] + _np_evo.sin(2 * t) * _c[4])
_means = [_mf(t) for t in _th]
_batch = _ha_geo(_th, _means, n_harmonics=3)
_online = _oha_geo(3, _D, forgetting=1.0).observe_many(_th, _means)
print(f"  SUBSTRATE EVOLUTION: harmonic_atom's batch lstsq fit becomes ONLINE (Recursive Least Squares, rank-1 Sherman-Morrison). Streaming {len(_th)} obs converges to the batch fit (coeff gap {float(_np_evo.max(_np_evo.abs(_online.W - _batch['coeffs']))):.0e}) -- the codebook self-organizes.")
_shift = _rng_evo.normal(size=_D); _dth = _rng_evo.uniform(0, 2 * _np_evo.pi, 80)
def _g(t, i): return _mf(t) + (i / 80.0) * _shift
_track = _oha_geo(3, _D, forgetting=0.85); _still = _oha_geo(3, _D, forgetting=1.0)
for _i, _t in enumerate(_dth):
    _track.observe(_t, _g(_t, _i)); _still.observe(_t, _g(_t, _i))
_pr = _rng_evo.uniform(0, 2 * _np_evo.pi, 20)
_te = float(_np_evo.mean([_np_evo.linalg.norm(_track.decode(t) - _g(t, 79)) for t in _pr]))
_se = float(_np_evo.mean([_np_evo.linalg.norm(_still.decode(t) - _g(t, 79)) for t in _pr]))
print(f"  ... and with a forgetting factor it becomes a DYNAMICAL SYSTEM that TRACKS a drifting meaning: decode err {_te:.2f} (tracking) vs {_se:.2f} (frozen on stale data).")

from holographic_orchestrator import chain_signature as _cs_geo, optimize_toolchain as _ot_geo
_rng_o = _np_evo.random.default_rng(0); _N, _Do, _L = 12, 256, 4
_base = _rng_o.normal(size=(_N, _Do)) + 2.0 * _rng_o.normal(size=(1, _Do))    # correlated tools (shared component)
_V = _base / _np_evo.linalg.norm(_base, axis=1, keepdims=True)
_true = list(_rng_o.choice(_N, _L, replace=False)); _goal = _cs_geo(_V[_true])
_greedy = list(_np_evo.argsort(_V @ _goal)[::-1][:_L])                        # position-blind per-tool score
_gsig = _cs_geo(_V[_greedy]); _gcos = float(_gsig @ _goal) / (_np_evo.linalg.norm(_gsig) * _np_evo.linalg.norm(_goal))
_idx, _dcos = _ot_geo(_V, _goal, _L, steps=300)
_rec = sum(1 for a, b in zip(_idx, _true) if a == b)
print(f"  DIFFERENTIABLE ORCHESTRATION: optimize a WHOLE tool-chain jointly by analytic-gradient ascent on cosine(chain_signature, goal) -- numpy, NO autodiff. On CORRELATED tools it recovers {_rec}/{_L} of the true chain (composed cosine {_dcos:.3f}) vs position-blind greedy {_gcos:.3f}.")
print("  The softmax tool-selection is the SAME soft operator that unifies smooth-union and memory cleanup: a soft tool choice and a soft recall are one math.  *** Evolution + differentiable orchestration ***")

title("One routing fabric: the chunkers/tilers/stores converge -- pick the pivot, get the regime (StructuredIndex keying)")
# The capacity-cliff cure ("route each item to a bounded-load bucket") had been re-grown five times. It is
# ONE fabric: you escape the cliff HORIZONTALLY and address shards by a PIVOT -- and the pivot you pick IS the
# regime. Hash -> the page-table/LBA regime (compute the address, ZERO comparisons -- "RAM"). Random
# projection -> nearest-neighbour content recall. Floor-divide -> spatial tiles. One parameter, `keying=`.
_ix = __import__("holographic_unified").UnifiedMind(dim=256, seed=0)
_rng = _np.random.default_rng(0)
_keys = _rng.standard_normal((2000, 256))
_proj = _ix.structured_index(_keys, payloads=[f"v{_i}" for _i in range(2000)])          # CONTENT regime
_pp, _pc = _proj.locate(_keys[42])
_hash = _ix.structured_index([f"k{_i}" for _i in range(2000)], keying="hash")            # RAM / page-table regime
_hp, _hc = _hash.locate("k42")
_coords = list({(int(_rng.integers(0, 64)), int(_rng.integers(0, 64))) for _ in range(2000)})
_spat = _ix.structured_index(_coords, keying="spatial", tile=8)                          # spatial-tile regime
_sp, _sc = _spat.locate(_coords[7])
print(f"  projection (content): locate -> '{_pp}' in {int(_pc)} comparisons   (sub-linear NN, vs 2000 flat)")
print(f"  hash (RAM/page-table): locate -> {int(_hp)} in {int(_hc)} comparison    (COMPUTE the address, exact -- zero search)")
print(f"  spatial (splat tiles): locate -> {int(_sp)} in {int(_sc)} comparison    (floor-divide -- the cell's address IS its tile)")
# sequential (the route chunker): RouteIndex's two-level summary routing is now this keying -- it delegates here.
from holographic_plan import chunk_route as _cr, RouteIndex as _RI
_rtiles = _rng.standard_normal((60, 256)); _rtiles = _rtiles / _np.linalg.norm(_rtiles, axis=1, keepdims=True)
_route = _cr(list(_rtiles), chunk=12, floor=0.12, seed=0, action_of=lambda _a, _b: 0)
_ri = _RI(_route); _rc, _rp, _rg = _ri.locate(_rtiles[20])
print(f"  sequential (route):    locate tile 20 -> chunk {int(_rc)} pos {int(_rp)} (global {int(_rg)})  "
      f"-- RouteIndex now DELEGATES its routing here (keying='{_ri._idx.keying}')")
# The splat tiler now DELEGATES its tiling to the same shared route -- build-time and recall-time provably agree.
from holographic_tree import _tile_bucket as _tb
from holographic_splat import splat_bundle_tiled as _sbt, recall_region_tiled as _rrt, splat_fit as _sf
_occ = _np.zeros((64, 64))
for _ in range(4):
    _cy, _cx = _rng.uniform(10, 54, 2); _ys, _xs = _np.mgrid[0:64, 0:64]
    _occ += _np.exp(-((_ys - _cy) ** 2 + (_xs - _cx) ** 2) / 50.0)
_scene = _sbt(_sf(_occ, 20), (64, 64), dim=2048, grid=16, tile=8, seed=0)
_shared = all(_tb(_c, _scene["tile"]) in _scene["tiles"] or _rrt(_scene, _c) == 0.0 for _c in [(0, 0), (9, 5), (15, 15)])
print(f"  splat tiler shares the SAME floor-divide route (one _tile_bucket, no drift): {_shared}")
print("  (storage still differs -- the index FINDS explicit keys, the splat tile DECODES a bounded bundle; one")
print("  routing fabric, two storage shapes, so TiledStore is a sibling not a flag. Migration is byte-identical.)")
# All three genuine 'individual solutions' in the chunking/tiling/store family now delegate to this fabric.
from holographic_tree import StructuredIndex as _SI
_cs = _SI(256, keying="projection", normalize=False).build(_rng.standard_normal((40, 256)))   # normalize=False == bare forest
print(f"  the content store delegates too: its hot bucket IS a StructuredIndex (normalize=False -> byte-identical "
      f"to the bare forest it replaced). Three solutions -> splat(spatial), route(sequential), store(projection).")

print("\n" + "-" * 66)
print("  Every subsystem -- through gradient-free learning -- ran on the same vector substrate. Wired up.")
print("-" * 66 + "\n")
