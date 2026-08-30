"""holographic_catalog_p03 -- part 3/6 of the capability registry (split from holographic_catalog).

MECHANICAL SPLIT, no edits. holographic_catalog.py hit 81% of the 1 MB agent-read cap, so the file
that makes capabilities discoverable was becoming the one file an agent could not open. The parts are
called IN ORDER by default_catalog() and the emitted catalog is byte-identical -- verified by hashing
every capability field before and after. Order matters: find_capability ranks by score and ties break
by registration order, so a reordering would silently move search results.

Add new capabilities to the LAST part, or to whichever part is topically right -- never to a new file
without registering it in default_catalog(), or it will simply not exist.
"""


def register_p03(c):
    """Register this part's capabilities on `c`. Called by default_catalog() in order."""
    c.register_capability("Do the two SDF emitters agree? (both executed, not asserted)", "holographic_sdf.to_glsl and sdfemit.sdf_dialect both emit a map() for one tree, and sdfemit's own header warns that TWO TABLES FOR ONE CONCEPT WILL DISAGREE -- but only one was ever executed, so agreement was narrative. mind.sdf_emitters_agree(tree) now RUNS both: the GLSL through a vec3 shim under g++ (no GL runtime needed), the C dialect under cc, each compared to the Python tree. Bars differ on purpose: C must be EXACT, GLSL gets 1e-5 because GLSL float is 32-bit and to_glsl writes 6-significant-digit literals (cos(0.7) -> 0.764842). MEASURED worst 4.3e-7; they agree.",
                          example="import numpy as np; import lecore; import holographic.mesh_and_geometry.holographic_sdf as S; m=lecore.UnifiedMind(dim=256,seed=0); r=m.sdf_emitters_agree(S.sphere(1.0)); (r['agree'], round(r['worst'],9))",
                          native=True, module="sdfemit",
                          aliases=("do the two shader emitters agree", "validate the glsl emitter",
                                   "is the shadertoy shader correct", "check emitted glsl against python",
                                   "run the glsl without a gpu", "compare shader to the sdf tree",
                                   "shader emitter regression"))

    c.register_capability("Run an SDF on the GPU (emitted map + per-pixel sphere trace)", "Bridges the shader EMITTER to the shader RUNNER, which two parallel merges left open: sdf_dialect emitted WGSL nothing dispatched; wgpurun dispatched WGSL nothing emitted. mind.sdf_depth_device(tree,w,h) sphere-traces an SDF ON ANY GPU -> (H,W) depth, -1 on miss; sdf_trace_shader returns the WGSL as inspectable TEXT (no device needed); sdf_depth_cpu is the NumPy reference on the SAME rays; sdf_depth_agrees differentially tests the two. Reuses run_wgsl_kernel bindings; raises without an adapter. sdf_trace_placement asks whether a device pays (144 flops/byte vs a 4.0 bar).",
                          example="import lecore; from holographic.mesh_and_geometry.holographic_sdf import sphere; m=lecore.UnifiedMind(dim=256,seed=0); d=m.sdf_depth_cpu(sphere(1.0), 17, 13); (d.shape, round(float(d[6,8]),3))",
                          native=True, module="wgpurun",
                          aliases=("run an sdf on the gpu", "raymarch on the gpu", "sdf compute shader",
                                   "render an sdf scene on the device", "gpu accelerated sdf render",
                                   "dispatch a shader from an sdf tree", "sphere trace on the device",
                                   "sdf depth buffer on the gpu",
                                   # the PLACEMENT half: same capability, the question a caller asks first
                                   "should i offload the render", "is it worth putting this on the gpu",
                                   "where should this trace run"))

    c.register_capability("Candles as a wave", "represent and operate on OHLC price candles as the SAMPLED WAVE "
                          "they actually are (holographic_candles): each bar is a sample of a continuous price "
                          "wave, and Open/High/Low/Close are four time-ordered facts about where it went. "
                          "candle_carrier gives the one-value-per-bar signal, candle_envelope the high/low band "
                          "(the intra-bar swing a close-line discards), candle_intrabar_path a 4x-resolution "
                          "reconstruction O->{H,L}->C. Once price IS a wave, spectrum / band-limit / phase-random "
                          "null / fit_deterministic / ladder_predict all apply",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "ohlc=np.array([[10,12,9,11],[11,13,10,12]]); print(list(m.candle_intrabar_path(ohlc)))",
                          native=True, aliases=("price candles as a wave", "ohlc as a signal",
                                                "represent a candlestick series", "candle high low envelope",
                                                "intrabar price path", "reconstruct a price wave from candles",
                                                "treat candles as a sampled signal", "price wave from ohlc"))
    c.register_capability("phase_randomized_null", "the honest NULL for a CONTINUOUS, autocorrelated signal "
                          "(holographic_surrogate) -- a phase-randomized surrogate has the SAME power spectrum "
                          "(same autocorrelation) as the signal but random phases, so deterministic/nonlinear "
                          "structure is destroyed while linear second-order stats are preserved (Theiler 1992). "
                          "Unlike a permutation, it does NOT destroy the autocorrelation a trivial forecaster "
                          "exploits. surrogate_zscore measures any structure statistic against this null -- a high "
                          "z means structure BEYOND autocorrelation",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "x=np.cumsum(np.random.default_rng(0).normal(size=512)); "
                          "print(round(float(np.abs(np.fft.rfft(x)).sum() - np.abs(np.fft.rfft(m.phase_randomize(x))).sum()),3))",
                          native=True, aliases=("phase randomized surrogate", "surrogate data null",
                                                "null preserving autocorrelation", "continuous signal null model",
                                                "is a time series more than autocorrelation",
                                                "structure beyond the spectrum", "spectrum-preserving shuffle",
                                                "honest baseline for a continuous signal"))
    c.register_capability("Route or abstain (find_capability judged against its own noise floor)",
                          "mind.route_or_abstain(query): null-referenced routing (J1) -- the top-1 "
                          "find_capability score judged against a null of scrambled queries drawn from the "
                          "CATALOG'S OWN vocabulary at matched token count (out-of-vocab gibberish scores 0 "
                          "and gates nothing). Below z_min the router says 'no capability matches' WITH the "
                          "z, instead of returning its argmax on noise. Logged misroutes abstain at "
                          "z=-0.9/-1.5; real queries route from z=+1.0; z_min=0.8 sits in the measured gap. "
                          "KEPT NEG: a genuine query in words the catalog never uses abstains CORRECTLY -- "
                          "the fix is aliases.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=mind.route_or_abstain('counter traders'); "
                                  "print(r['abstain'], round(r['z'],1))",
                          native=True, aliases=("no capability matches", "router that can abstain",
                                                "abstain instead of misroute",
                                                "routing confidence against a null",
                                                "is this query answerable by the catalog",
                                                "gate the capability search",
                                                # WAS "refuse to route nonsense" -- the bare token `nonsense` made
                                                # the garbage query "qwzx nonsense zzzq" match THIS entry at 0.333,
                                                # which is precisely the failure test_pure_nonsense_routes_to_unknown
                                                # exists to catch (and had already caught once, in a does-field).
                                                # The irony is instructive: the capability whose whole job is to
                                                # abstain on gibberish was the one gibberish routed to. Reworded, not
                                                # deleted -- the user intent is real, only the bare token was toxic,
                                                # and two additive phrasings replace the reach it lost.
                                                "refuse to route a query it cannot match",
                                                "abstain instead of guessing",
                                                "say no capability matches instead of guessing",
                                                "null referenced retrieval"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Wave-state encoder (carrier + envelope as one recallable state)",
                          "mind.wave_state_encoder(dim, window): one OHLC window -> one unit state vector "
                          "carrying carrier SHAPE (close-based, unit-RMS), both envelope excursion channels "
                          "in scale units (their amplitude is exactly what a close-only encoder cannot see; "
                          "identical closes with 4x swing separate at cos 0.77), and an energy term. Offset/"
                          "scale invariant (same shape at 10x level: cos 0.94 -- the invariance IS the "
                          "level-blindness kept negative). Feeds causal_index recall (5/5 right-regime "
                          "neighbours, fitless) and signal_program screening. D4 note travels with it: "
                          "calibration on these states is NOT exploitability.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); e=mind.wave_state_encoder(256, window=8); "
                                  "o=np.arange(8.0); w=np.stack([o,o+1,o-1,o+0.5],axis=1); "
                                  "v=e.encode(w); print(round(float(v@v),2))",
                          native=True, aliases=("wave state encoder carrier and envelope",
                                                "encode candle high low as one vector",
                                                "carrier plus envelope state",
                                                "within interval extremes encoding",
                                                "envelope excursion state vector",
                                                "resonance recall state for candles",
                                                "ohlc window to hypervector",
                                                "state vector with intra bar swing"),
                          semantic="create/emit", consumes=(), produces=())

    c.register_capability("Decomposition contract (do the pieces sum back, and may you use them at time t)",
                          "mind.decomposition_contract(decompose_fn, x): judge ANY decomposition on its three "
                          "implicit promises. COMPLETE: components sum back within atol, else it is a "
                          "projection wearing a decomposition's name. CAUSAL: lookahead_lint PER COMPONENT "
                          "-- which parts are usable at time t vs diagnosis-only. HONEST RESIDUAL: flags "
                          "when 'residual' carries the majority (a sliver removed, the rest renamed). Energy "
                          "shares NOT normalised: correlated components stay visibly double-counted. Dogfood "
                          "on record: smooth_sharp_split certifies COMPLETE + NON-CAUSAL.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); x=np.cumsum(np.random.default_rng(0).standard_normal(200)); "
                                  "f=lambda s:{'mean':np.full(s.size,0.0),'residual':s}; "
                                  "print(mind.decomposition_contract(f,x)['residual_dominates'])",
                          native=True, aliases=("decomposition contract components plus residual",
                                                "split a signal into parts that sum back",
                                                "trend seasonal residual split audit",
                                                "decompose then verify the pieces add up",
                                                "causal decomposition of a series",
                                                "audit a decomposition for leakage",
                                                "is my residual secretly the signal",
                                                "which decomposition parts are usable live"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Resting fills + paper book (passive adverse selection; forward test with gates)",
                          "mind.resting_fill_sim(path, events, delta): unconditional mark-out is +delta by "
                          "construction (the discount a naive backtest banks); FILLED mark-out on a random "
                          "walk is NEGATIVE -- being chosen claws back more than the discount. Extra "
                          "adverse: momentum -2.45 << rw -0.53 < reversion -0.21; depth shrinks the per-fill "
                          "extra while fills collapse. Price-path only: real queues are WORSE. "
                          "mind.paper_book(lag, cost): forward harness with gates attached -- actionable "
                          "entries (lag>=1), costs, gate masks, sleeves with the MEDIAN beside the mean. "
                          "Proves plumbing, not edge.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); p=list(np.cumsum(r.standard_normal(3000))); "
                                  "res=mind.resting_fill_sim(p, list(range(50,2900,40)), delta=1.0); "
                                  "print(round(res['selection_cost'],2), round(res['fill_rate'],2))",
                          native=True, aliases=("resting order adverse selection",
                                                "limit order fill simulator", "who fills against me",
                                                "passive fill toxicity", "queue position fill model",
                                                "paper trading harness", "forward test book with gates",
                                                "walk forward paper account",
                                                "simulated account with sleeves and medians",
                                                "cost of being filled passively"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Hostile-data guide (the honesty layer's field manual)",
                          "docs/HOSTILE_DATA_GUIDE.md: find real structure in noisy sequential data and "
                          "refuse to be fooled -- pipelines that manufacture (79.4% persistence on white "
                          "noise), evaluations that leak (self-matching kNN at MSE 0.0; 28% false-alarm "
                          "under overlap), batteries that select (p=4e-4 dies on a 64-look book), aggregates "
                          "hiding the loss shape. Names the tool per failure and THE ORDER TO RUN THEM (lint "
                          "-> pipeline_null -> effects -> battery+ledger -> events -> conditions -> costs -> "
                          "committee); a refusal is a result. Every snippet is executed by its test, so the "
                          "guide cannot rot without a failure.",
                          example="import pathlib; p=pathlib.Path('docs/HOSTILE_DATA_GUIDE.md'); "
                                  "t=p.read_text(); print(t.splitlines()[0], len(t) > 4000)",
                          native=True, aliases=("guide to analyzing hostile data",
                                                "how to find real structure in noisy data",
                                                "honest analysis workflow",
                                                "which honesty tool do I use when",
                                                "recipe for validating a signal",
                                                "hostile data checklist",
                                                "field manual for the honesty layer",
                                                "order to run the honesty tools"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Circular encoder (angles and clocks with an EXACT wrap)",
                          "mind.circular_encoder(dim, period): encode a CIRCULAR variable (angle, hour, "
                          "weekday, phase) so encode(x) == encode(x+period) to 1e-12 and similarity depends "
                          "ONLY on the circular gap: 23:59 and 00:01 read as 2-minute neighbours where the "
                          "LINE ScalarEncoder reads cos 0.21 (periodicity needs INTEGER harmonics -- a "
                          "construction, not a parameter). Poisson-minus-DC kernel: small antipodal dip "
                          "(<0.25, measured); concentration trades lobe width for dip. decode() = circular "
                          "cleanup. Audit carried: SignedEncoder REFUTED -- signed is native to "
                          "ScalarEncoder.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); e=mind.circular_encoder(512, period=24.0); "
                                  "a=e.encode(23.9); b=e.encode(0.1); c=e.encode(12.0); "
                                  "print(round(float(a@b),2), round(float(a@c),2), round(e.decode(a),1))",
                          native=True, aliases=("circular variable encoding", "encode an angle as a vector",
                                                "hour of day encoder", "day of week embedding",
                                                "encode a phase with wraparound",
                                                "periodic value to hypervector",
                                                "clock arithmetic similarity",
                                                "wraparound aware encoder", "encode headings or bearings"),
                          semantic="create/emit", consumes=(), produces=())

    c.register_capability("Loss space report (where the losses live, per axis, vs its own null)",
                          "mind.loss_space_report(values, conditions=None): the SHAPE of a loss record on "
                          "three axes, each vs the null erasing only the structure under test. TAIL: worst-5% "
                          "share of loss vs a matched Gaussian (heavier = the mean is a comfort blanket). "
                          "TIME: longest losing streak vs the permutation null (z>2 = losses arrive "
                          "together). CONDITION: per mask, loss share vs occupancy under the circular-shift "
                          "null -- 10% occupancy carrying 60% of loss is the gate candidate. Loss-side "
                          "sibling of the insurance profile. Too few losses -> a scarcity report, not a z.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); v=r.normal(0.05,1,500); "
                                  "storm=np.zeros(500,bool); storm[100:160]=True; v[storm]-=1.5; "
                                  "rep=mind.loss_space_report(v, conditions={'storm': storm}); "
                                  "print(rep['verdict'][:60])",
                          native=True, aliases=("where do the losses concentrate",
                                                "characterize my failures", "loss concentration report",
                                                "which states lose the money",
                                                "are losses clustered in time",
                                                "breakdown of losses by condition",
                                                "longest losing streak versus chance",
                                                "loss tail heavier than gaussian",
                                                "profile of the worst outcomes"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Calibration vs value (a good forecast is not yet a good decision)",
                          "mind.calibration_vs_value(probs, outcomes): Murphy-decomposed Brier (reliability /"
                          " resolution / uncertainty) beside realized net under act-if-p>=tau (tau sweep, "
                          "never/always baselines), verdicts SEPARATE. Pinned: a calibrated CONSTANT forecast "
                          "is worthless -- resolution is the number that failed, and the verdict names it -- "
                          "while the same forecast monotone-squashed to 38x worse reliability keeps 100% of "
                          "its achievable value: calibration is a REPAIR, resolution is the SOURCE. KEPT NEG: "
                          "value_best is an argmax over taus (a selection) -- pick tau elsewhere or ledger "
                          "the sweep.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); p=np.clip(r.beta(2,2,500),.01,.99); "
                                  "y=(r.random(500)<p).astype(float); "
                                  "res=mind.calibration_vs_value(p,y,cost=0.05); "
                                  "print(res['verdicts']['value'][:40], round(res['resolution'],3))",
                          native=True, aliases=("calibration is not profit",
                                                "does my good forecast make money",
                                                "forecast quality versus decision value",
                                                "well calibrated but worthless",
                                                "score a forecast by the decisions it drives",
                                                "value of a forecast under an action rule",
                                                "brier score versus realized payoff",
                                                "reliability diagram with payoffs",
                                                "resolution versus reliability"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Event study (what happens after the signal fires, vs a shift null)",
                          "mind.event_study(outcome, events, horizon): cumulative mean path around each "
                          "event vs the CIRCULAR-SHIFT null -- the whole pattern slid by a random offset, "
                          "preserving count and every spacing, so the null inherits clustering AND overlap; "
                          "only alignment is tested. Returns forward {z,p}, pre_trend {z,p} (large pre-trend "
                          "z = the event DEFINITION already contains the move), n_overlapping, "
                          "shared_fraction. KEPT NEGATIVE, measured: overlap makes the naive across-events t "
                          "false-alarm at 28% on noise where this null holds 2% -- never rebuild a CI from "
                          "mean_path. Edges dropped, never truncated.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); y=r.standard_normal(3000); "
                                  "ev=list(range(100,2900,140)); "
                                  "[y.__setitem__(slice(e+1,e+9), y[e+1:e+9]+0.4) for e in ev]; "
                                  "res=mind.event_study(y, ev, horizon=15); print(round(res['forward']['z'],1), res['n_events'])",
                          native=True, aliases=("event study forward paths",
                                                "average path after an event",
                                                "forward returns after a trigger",
                                                "what happens after the signal fires",
                                                "aligned windows around events",
                                                "post event drift measurement",
                                                "superposed epoch analysis",
                                                "does my trigger predict anything",
                                                "pre event trend check", "post event drift measurement",
                                                "what happens after the signal fires"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Rolling / streaming statistics (causal by construction, exact by default)",
                          "mind.rolling_stats(x, window, stats=(...)): trailing mean/std/min/max/range/"
                          "quantile/drawdown/ewma/ewm_std series -- window ENDING at each position, NaN in "
                          "warm-up (never a silently-shrunk window), every stat lint-causal at 0.0 drift and "
                          "BIT-identical to the conditioning gate's TRAILING_STATS. Exact per-window default; "
                          "the O(n) cumsum path is opt-in (1e8 offset: fast std off by 8.75 vs 2e-9 exact). "
                          "mind.streaming_stats(window) is the live sibling: Welford + monotonic deques; "
                          "warm_start replays history through the same push() so live == backtest tail.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); x=np.cumsum(np.random.default_rng(0).standard_normal(100)); "
                                  "r=mind.rolling_stats(x, 20, stats=('std','drawdown')); "
                                  "s=mind.streaming_stats(window=20).warm_start(x); "
                                  "print(round(r['std'][-1],6), round(s.std(),6))",
                          native=True, aliases=("causal rolling standard deviation",
                                                "trailing window statistics kit",
                                                "rolling statistics without look ahead",
                                                "streaming quantile", "running percentile of a stream",
                                                "trailing drawdown series", "online mean and variance",
                                                "ewma volatility series", "rolling quantile",
                                                "warm start live statistics from a backtest",
                                                # the everyday names a stranger reaches for FIRST: the
                                                # specialist aliases above all missed "moving average",
                                                # which lost to a reprojection-VELOCITY entry on lexical
                                                # gravity ("moving"/"motion") in the post-merge sweep.
                                                "moving average", "moving average over a window",
                                                "simple moving average", "rolling mean",
                                                "smooth a series with a sliding window"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Look-ahead linter (prove the signal only used the past)",
                          "mind.lookahead_lint(signal_fn, x): recomputes signal_fn on truncated prefixes and "
                          "demands the shared range be IDENTICAL -- a causal pipeline cannot know whether "
                          "data exists after t, so drift IS leakage: full-sample z-score, centred smoother, "
                          "global min-max and detrend all caught at machine precision with a first-bad index; "
                          "trailing EMA/z pass at exactly 0.0. Pair with mind.target_shift_probe (signal "
                          "AHEAD of its target or explaining it? catches the contemporaneous leak; a "
                          "symmetric centred-label leak belongs to the lint). Necessary, not sufficient.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); x=np.cumsum(np.random.default_rng(0).standard_normal(300)); "
                                  "bad=lambda s:(s-s.mean())/s.std(); good=lambda s: np.concatenate([[0.],np.diff(s)]); "
                                  "print(mind.lookahead_lint(bad,x)['causal'], mind.lookahead_lint(good,x)['causal'])",
                          native=True, aliases=("look ahead linter", "detect future leakage in an evaluation",
                                                "did the mask use future data",
                                                "find look ahead bias in my backtest",
                                                "check an evaluation for future information",
                                                "leak detector for a signal pipeline",
                                                "is my feature peeking at the future",
                                                "prefix consistency check", "prove a pipeline is causal",
                                                "is the signal ahead of its target"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Causal recall (an index that cannot see the future)",
                          "mind.causal_index() -> CausalIndex: append(vector, t) in time order -- backfilling "
                          "the past refuses by name -- and nearest(query, t, k, lag>=1) searches ONLY items "
                          "with time <= t - lag (lag=0 refused: simultaneous is not past). audit_causality "
                          "VERIFIES the mask by perturbing future items and checking results are bit-identical. "
                          "The demo it pins: naive full-history k=1 history-matching finds the query ITSELF -- "
                          "perfect fake skill, 100% inflation -- while this index cannot self-match at any k. "
                          "Exact scan only: a similarity forest cannot be time-masked (declared, not a TODO).",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); ci=mind.causal_index(); import numpy as np; r=np.random.default_rng(0); "
                                  "[ci.append(r.standard_normal(8), float(t)) for t in range(50)]; "
                                  "print(ci.nearest(r.standard_normal(8), 25.0, k=2), ci.nearest(r.standard_normal(8), 0.0))",
                          native=True, aliases=("nearest neighbour search restricted to the past",
                                                "recall only older items", "time filtered index",
                                                "append only memory before t",
                                                "history matching without look ahead",
                                                "what did similar past states lead to",
                                                "analog lookup that cannot see the future",
                                                "knn over trailing history only",
                                                "point in time similarity search"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Selection ledger (correct over everything you TRIED, not what survived)",
                          "mind.selection_ledger() -> SelectionLedger: record(name, p, family) every test AT "
                          "THE MOMENT IT IS RUN, correct(alpha) computes FDR q-values over the WHOLE book or a "
                          "named family; report() shows, per family, how many pass in-family but DIE on the "
                          "book -- the look-elsewhere effect made visible (a p=4e-4 family winner dies on a "
                          "64-look book). Append-only: withdraw() needs a reason and keeps the multiplicity "
                          "cost; re-runs are sequences. to_json/from_json persist behind a hashlib chain that "
                          "refuses a book with a deleted row. KEPT NEGATIVE: covers only what is written down.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); led=mind.selection_ledger(); led.record('effect_a', 0.0004, family='routing'); "
                                  "[led.record('sweep_%d'%i, 0.5, family='sweep') for i in range(60)]; "
                                  "r=led.correct(alpha=0.05); print(r['family_size'], r['n_passed'])",
                          native=True, aliases=("ledger of every test I ran",
                                                "record all hypotheses tried this session",
                                                "did I run it until it passed",
                                                "ledger record over http", "session ledger for an agent",
                                                "family wise correction across batteries",
                                                "look elsewhere effect bookkeeping",
                                                "how many things did I try before this worked",
                                                "selection debt tracker",
                                                "append a test result to a running ledger",
                                                "session wide false discovery correction",
                                                "multiple testing across the whole project"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Screen a battery of detectors (honesty gates inside the loop)",
                          "mind.signal_program() -> SignalProgram: add_check registers detectors, "
                          "screen(states, targets) evaluates ALL at once -- every effect returns WITH its "
                          "split-half replication and FDR verdict; no path yields the seductive number "
                          "alone. Passers are correlation-clustered (0.9-correlated checks = ONE finding); "
                          "an empty pass-list is a RESULT with a reason. build_committee seats a VETO "
                          "COMMITTEE (one rep per cluster, tie=abstain) that must pass ITS OWN gates on "
                          "fresh data; empty committee refuses. program_vector fingerprints the battery.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); s=r.standard_normal((600,4)); "
                                  "t=np.sign(s[:,0])*np.abs(r.standard_normal(600)); p=mind.signal_program(seed=0); "
                                  "p.add_check('real', lambda x: x[:,0]); p.add_check('noise', lambda x: x[:,1]); "
                                  "rep=p.screen(s,t); print(rep['passed'], rep['clusters'], rep['refused'])",
                          native=True, aliases=("screen many detectors in one pass",
                                                "battery of checks as one program",
                                                "evaluate all signal checks simultaneously",
                                                "test many hypotheses with fdr built in",
                                                "committee of detectors that refuses to overfit",
                                                "which of my signals actually survive",
                                                "multiple comparisons across a detector family",
                                                "screen candidates honestly", "detector battery",
                                                "veto committee", "build a committee of detectors",
                                                "combine signals with survival gates",
                                                "majority vote of gated signals",
                                                "empty committee as a result", "multiple comparisons across a detector family",
                                                "battery screening with honesty gates"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Re-clock a series (sample when it moves, not when time passes)",
                          "mind.reclock(series, step, axis) emits one event per `step` of axis movement -- "
                          "quiet stretches cheap, busy dense; per-event DURATION is the activity channel with "
                          "magnitude divided out. axis=None is the price clock (cumulative |diff| of the "
                          "series itself), the only configuration whose sharpening is MEASURED; foreign axes "
                          "added nothing (|z|<1.4). duration_stats + duration_resolution_check read the "
                          "channel honestly. KEPT NEGATIVES: events completing inside one sample are counted "
                          "(skipped_gap), never fabricated; a quantised duration grid makes stats artifacts.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); x=np.cumsum(np.random.default_rng(0).normal(size=800)); "
                                  "ev=mind.reclock(x, step=2.0); "
                                  "print(ev['n_events'], mind.duration_resolution_check(ev)['ok'])",
                          native=True, aliases=("reclock a series by movement", "renko bricks",
                                                "sample when it moves not when time passes",
                                                "event time sampling", "emit an event per unit of change",
                                                "price clock", "volume clock", "photon count clock",
                                                "duration per event", "activity channel of a series",
                                                "time per unit of progress"),
                          semantic="analyze/measure", consumes=(), produces=())
    c.register_capability("Reclock persistence vs its own null (the manufactured-momentum trap)",
                          "mind.rotation_persistence(events) is the NAIVE readout; mind.null_persistence("
                          "series, step) is the honest one -- the full reclock chain run on surrogates via "
                          "pipeline_null. The manufactured DIRECTION is a property of the mechanism: renko "
                          "made +72% fake momentum on pure noise, this total-variation clock makes ~25% fake "
                          "reversion on the SAME noise -- two clocks, two confident opposite stories, one "
                          "structureless input. null_mean far from 0.5 IS the manufacturing, on display. KEPT "
                          "NEGATIVE: price clock only -- an external axis has no defined reordering under a "
                          "surrogate.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); x=np.random.default_rng(0).normal(size=2000); "
                                  "r=mind.null_persistence(x, step=2.0, n=60); "
                                  "print(round(r['observed'],2), round(r['null_mean'],2), round(r['z'],1))",
                          native=True, aliases=("brick direction persistence", "renko momentum test",
                                                "is my reclocked momentum real",
                                                "persistence of reclocked events against null",
                                                "did the re-clocking invent the momentum",
                                                "honest brick persistence", "event clock direction bias"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Envelope forecast (predict the SIZE of the next move, not its direction)",
                          "mind.envelope_forecast(series): a calibrated band for |next move| from trailing "
                          "scale + conformal RATIO residuals -- one quantile serves every volatility state; "
                          "an additive margin under-covers storms, over-covers calm (pinned). Ships with "
                          "holdout coverage and a zero-directional-bits note (never launders scale skill "
                          "into direction). envelope_vs_constant is the mandatory baseline; verdict names "
                          "the case: BOTH-COVER (ratio is the score), CONSTANT-FAILED (drift broke the "
                          "constant band; ratio is not a ranking), CONDITIONAL-FAILED (do not quote).",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); "
                                  "s=np.where((np.arange(2000)//250)%2==0,0.5,2.5); x=np.cumsum(r.normal(size=2000)*s); "
                                  "e=mind.envelope_forecast(x); print(round(e['coverage_holdout'],2), round(e['upper'],2))",
                          native=True, aliases=("predict the size of the next move not its direction",
                                                "volatility forecast band", "how big will the next change be",
                                                "magnitude forecast band", "scale of the next move",
                                                "envelope forecast with intervals", "forecast a band not a point",
                                                "volatility clustering forecast", "range forecast calibrated"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Conditional coverage (is the interval's guarantee real in every state?)",
                          "mind.conditional_coverage(resid_calib, resid_test, condition): the conformal "
                          "coverage check split inside/outside a condition (regime, storm gate, load level). "
                          "Marginal coverage is an AVERAGE and can hold while both sides fail in opposite "
                          "directions -- canon: nominal 90%, ~97% calm / ~70% storm, calibrated on paper, "
                          "useless where needed. `degraded` flags a side missing nominal by >2 binomial SEs; "
                          "thin sides report reliable=False. KEPT NEGATIVE: the split-conformal guarantee IS "
                          "marginal; closing a gap needs per-condition calibration -- this says whether.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); "
                                  "storm=np.arange(400)%4==0; test=np.where(storm,r.normal(0,3,400),r.normal(0,1,400)); "
                                  "print(mind.conditional_coverage(r.normal(0,1,400), test, storm, alphas=(0.1,))[0]['degraded'])",
                          native=True, aliases=("conformal coverage by regime", "coverage report conditional",
                                                "does the interval hold in storms",
                                                "per regime interval coverage", "coverage inside a condition",
                                                "is my forecast interval calibrated in every state",
                                                "conditional conformal check"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Cost wall + actionable fills (was the edge real at the moment of ACTION?)",
                          "The action layer's two honesty gates. mind.net_of_costs(values, cost): net mean/t, "
                          "wall_ratio, survives, breakeven ('survives at 5 bp, dies at 9' travels; 'survives' "
                          "does not) -- per-event cost arrays supported, since a constant cost is a model. "
                          "mind.realizable_fills(events, path, horizon): entry at the first REACHABLE state "
                          "after the event is known vs the idealized emission price; latency_cost = the move "
                          "that completed during recognition (canon: z=+20 at emission, NEGATIVE actionable). "
                          "lag=0 refused by name; sweep the lag before believing an edge.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); "
                                  "v=r.normal(10,20,300); print(mind.net_of_costs(v,cost=17)['survives'], "
                                  "round(mind.net_of_costs(v,cost=17)['breakeven_cost'],1))",
                          native=True, aliases=("does the signal survive costs", "gross edge versus transaction costs",
                                                "net of costs per trade", "cost wall evaluator",
                                                "breakeven cost of a signal",
                                                "enter at the price when the signal is known",
                                                "emission versus actionable price", "signal known too late",
                                                "backtest fill at the actionable price", "latency cost of acting",
                                                "detection latency versus action latency",
                                                "is my signal late by construction", "latency artifact check",
                                                "can I actually trade this signal", "fees eat my profit"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("DPI guard (is this feature NEW information or a re-dressing?)",
                          "mind.dpi_guard(features, new_feature): fit the proposal from a linear/quadratic "
                          "expansion of the existing set on a train split, report R^2 on train AND HOLDOUT "
                          "(never train alone); novel_frac = the reproducibly-unexplained share, the MOST it "
                          "could add. DPI: a transform CONCENTRATES information, never creates it (canon: "
                          "kernel lifts / embeddings / foreign clocks, weeks spent, ~0 new bits). KEPT "
                          "NEGATIVES: novel may be noise (owes a target-side test in bits); exotic transforms "
                          "outside the basis can slip. mind.holdout_auc pairs separability the same way.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); F=r.normal(size=(500,3)); "
                                  "g=np.tanh(F[:,0]+0.3*F[:,1]*F[:,2]); "
                                  "print(mind.dpi_guard(F,g)['verdict'][:9], round(mind.dpi_guard(F,g)['r2_holdout'],2))",
                          native=True, aliases=("is this feature actually new information",
                                                "is it just a transform of existing features",
                                                "data processing inequality guard", "dpi guard",
                                                "does this embedding add anything",
                                                "new feature or re-representation", "feature redundancy check",
                                                "train and holdout auc", "overfit separability check",
                                                "holdout auc"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("Split-half replication (the gate that kills artifacts)", "mind.split_half(values) or "
                          "mind.split_half(events, values): cut the measurements in two, measure the effect in "
                          "each half, PASS only if both halves agree in SIGN and each is significant. "
                          "mode='contiguous' (default) does the killing; mode='interleave' shares the regime, "
                          "so passing interleaved while failing contiguous means REGIME-BOUND. Returns per-half "
                          "mean/t/p plus `passed`. Measured: killed four artifacts every other readout called "
                          "real, no false rejections. KEPT NEGATIVE: normal-approx p (small_sample flags halves "
                          "under 30); replication is not multiplicity control -- run bh_fdr too.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); "
                                  "v=np.concatenate([r.normal(0.6,1,200), r.normal(0.0,1,200)]); "
                                  "print(mind.split_half(v)['passed'], mind.split_half(v,mode='interleave')['passed'])",
                          native=True, aliases=("split half replication", "does it hold in both halves",
                                                "check the effect replicates", "first half second half agreement",
                                                "did this survive out of sample", "is this result an artifact",
                                                "replicate on two halves", "sanity check my effect",
                                                "did the edge decay", "test stability over time"),
                          semantic="analyze/measure", consumes=(), produces=())
    c.register_capability("Pipeline null (did my PROCESSING manufacture the structure?)",
                          "mind.pipeline_null(pipeline_fn, x, surrogate): run your WHOLE chain on surrogates "
                          "and score the statistic against the null the pipeline itself produces. Any smoothing, "
                          "quantising, re-clocking or clustering step imposes correlations on whatever it is fed, "
                          "INCLUDING pure noise, so a null on the raw input credits the pipeline's artifacts to "
                          "the data. Measured: a re-clock made 72% direction persistence on noise (referenced "
                          "truth: ANTI-persistence z=-7.3); a denoiser made 83.6%. Returns z/p/collapsed. KEPT "
                          "NEGATIVE: a bad surrogate gives a healthy-looking meaningless z.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); d=np.random.default_rng(0).normal(size=1500); "
                                  "pipe=lambda v:(lambda s: float(np.mean(s[1:]==s[:-1])))(np.sign(np.convolve(v,np.ones(9)/9,'valid'))); "
                                  "r=mind.pipeline_null(pipe,d,surrogate='iid_shuffle',n=50); "
                                  "print(round(r['observed'],3), round(r['z'],2))",
                          native=True, aliases=("run my whole pipeline on surrogates",
                                                "does my pipeline manufacture structure",
                                                "null for a processing chain", "is my smoothing creating the signal",
                                                "test the pipeline not just the statistic",
                                                "baseline for a multi step analysis",
                                                "did the preprocessing invent this", "surrogate through the same steps",
                                                "am I fooling myself with resampling", "baseline for a multi step analysis",
                                                "did my pipeline invent the structure"),
                          semantic="analyze/measure", consumes=(), produces=())
    c.register_capability("Detection floor (no effect above X)", "mind.min_detectable_effect(test_fn, x, "
                          "effect_grid, surrogate, power): turn 'we found nothing' into 'nothing here above X' "
                          "-- the only null result that can be argued with. Injects effects of known size into "
                          "surrogates of your OWN x (so the noise level is the one you face) and reports the "
                          "smallest size the test catches at the target power, plus the power curve. floor=None "
                          "means extend the grid upward, not that the floor is zero. KEPT NEGATIVE: a floor is "
                          "conditional on the injection SHAPE, and the surrogate must DESTROY the statistic "
                          "tested or the curve degenerates to 0/1.",
                          example="import numpy as np, math; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); x=np.random.default_rng(0).normal(size=400); "
                                  "t=lambda v: math.erfc(abs(v.mean()/(v.std(ddof=1)/math.sqrt(len(v))))/math.sqrt(2)); "
                                  "print(mind.min_detectable_effect(t,x,[0.05,0.1,0.15,0.2],surrogate='sign_flip',n_trials=40)['floor'])",
                          native=True, aliases=("smallest effect I could detect", "detection floor",
                                                "minimum detectable effect", "statistical power curve",
                                                "how big would an effect need to be", "how strong is my null result",
                                                "could my test even have seen it", "power analysis",
                                                "quantify what I ruled out"),
                          semantic="analyze/measure", consumes=(), produces=())
    c.register_capability("Arrow of time (is this series time-reversible?)", "mind.trev(x, lag) and "
                          "mind.time_arrow_test(x, kind): the normalised third moment of the lagged difference "
                          "is exactly zero for a time-reversal-invariant process and non-zero when rises and "
                          "falls have different SHAPES; time_arrow_test scores it against a surrogate ensemble "
                          "(value/null_mean/z/p). Large |z| says NONLINEAR -- a triage flag, not a detection. "
                          "Defaults to the IAAFT null because a merely SKEWED series scores big against a "
                          "phase-randomised one. KEPT NEGATIVE, measured: a global arrow can be entirely DIFFUSE "
                          "(z=+6.4, all three localisation attempts null) -- never a per-window signal.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); saw=(np.arange(1024)%50)/50.0; "
                                  "print(round(mind.trev(saw),2), round(mind.time_arrow_test(saw,n_surrogates=40)['z'],1))",
                          native=True, aliases=("time reversal asymmetry", "is this series time reversible",
                                                "arrow of time in a signal", "trev statistic",
                                                "does the series look different backwards",
                                                "detect nonlinearity in a time series",
                                                "irreversibility test", "asymmetry between rises and falls"),
                          semantic="analyze/measure", consumes=(), produces=())
    c.register_capability("Directional & scale surrogates (pick the null that destroys YOUR claim)",
                          "mind.sign_flip / iid_shuffle / block_shuffle / surrogate_ensemble: the null is a "
                          "CHOICE -- destroy exactly what you claim, preserve everything else. sign_flip "
                          "randomises DIRECTION keeping every magnitude exactly (a plain shuffle would "
                          "over-credit magnitude structure). block_shuffle keeps structure shorter than `block`, "
                          "destroys longer (the SCALE dial). iid_shuffle destroys all order. surrogate_ensemble "
                          "streams n of any kind, memory-light. KEPT NEGATIVES: sign_flip is degenerate for "
                          "magnitude-only statistics; block joins are fake jumps; block=1 IS iid_shuffle.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); x=np.cumsum(np.random.default_rng(0).normal(size=512)); "
                                  "s=mind.sign_flip(x); "
                                  "print(bool(np.array_equal(np.abs(s),np.abs(x))), "
                                  "len(list(mind.surrogate_ensemble(x,'block_shuffle',n=3,block=64))))",
                          native=True, aliases=("sign flipped surrogate", "flip the signs of my data randomly",
                                                "randomize direction keep magnitudes", "shuffle in blocks",
                                                "block bootstrap", "destroy short range structure keep long",
                                                "null that keeps volatility but randomizes direction",
                                                "which null should I use", "shuffle my data as a baseline",
                                                "generate many surrogates cheaply"),
                          semantic="analyze/measure", consumes=(), produces=())
    c.register_capability("Causal gates (act only on what you knew at the time)",
                          "mind.causal_gate(stat, window, threshold, compare): a condition that sees only "
                          "TRAILING data, so it can be ACTED on, not merely described. Causal by construction "
                          "and PROVABLY so -- audit_causality scrambles the future and checks the past does not "
                          "move, catching full-sample normalisations and global-quantile thresholds. Composable "
                          "with & | ~. Measured: a storm gate (trailing drawdown <=-15% OR vol top decile) left "
                          "entries untouched and moved a book +22% -> +58.4% CAGR, maxDD -85.9% -> -47.1%. KEPT "
                          "NEGATIVE: a hand-written mask claiming causal=True is a claim, not a proof.",
                          example="import numpy as np, lecore; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); "
                                  "path=np.cumsum(np.random.default_rng(0).normal(size=300))+100.0; "
                                  "g=mind.causal_gate('drawdown',window=60,threshold=-0.05,compare='le'); "
                                  "print(mind.causal_gate('std',window=60,threshold=1.0,compare='ge',"
                                  "context=path)['audit']['passed'], int(g.mask(path).sum()))",
                          native=True, aliases=("only act on information available at the time",
                                                "stand aside when conditions are bad",
                                                "causal filter no look ahead", "trailing window condition",
                                                "did I accidentally use future data to filter",
                                                "ex ante versus ex post", "risk off switch",
                                                "gate my signal on volatility", "drawdown gate",
                                                "prove my filter is not peeking"),
                          semantic="analyze/measure", consumes=(), produces=())
    c.register_capability("Conditional statistics (all / inside / outside / difference)",
                          "mind.conditional(values, condition): any measurement FOUR ways in one call -- "
                          "overall, inside the condition, outside, and the difference (Welch z + p) -- with "
                          "detection floors and a loud warning when the split is EX-POST. condition is a Gate "
                          "(causal), an ExPostMask, or a raw boolean array (deliberately ex-post: trusting the "
                          "caller is how look-ahead gets in). Measured reframe: an unconditional average hid two "
                          "OPPOSITE behaviours -- trending when calm, whipsawing in storms, flat on average. "
                          "Condition a weak effect before abandoning it, a strong one before believing it.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); v=r.normal(0,1,600); "
                                  "f=np.zeros(600,bool); f[::3]=True; v[f]+=1.0; "
                                  "c=mind.conditional(v,f); print(round(c['diff'],2), c['separates'], c['causal'])",
                          native=True, aliases=("compare the statistic inside and outside a condition",
                                                "break down a result by condition",
                                                "split my results by market state",
                                                "does the effect depend on conditions",
                                                "conditional average", "subgroup analysis",
                                                "is the effect different when x is true",
                                                "measure inside versus outside"),
                          semantic="analyze/measure", consumes=(), produces=())
    c.register_capability("Per-regime validation (one effect, or one regime's story?)",
                          "mind.across_regimes(values, series=...): evaluate an effect inside EVERY measured "
                          "regime -- pass segments, or pass the series and they are measured by the engine's "
                          "change-point segmenter. Per segment: n/mean/t/p plus a DETECTION FLOOR, so an empty "
                          "regime reports 'nothing above X', not 'nothing'. Across segments: sign consistency, a "
                          "sign test, and `concentration` (share carried by one regime). Measured: a real effect "
                          "was positive in 3 of 4 regimes; an artifact with a comparable headline had >0.9 in "
                          "one. KEPT NEGATIVE: the sign test is underpowered -- read concentration first.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); r=np.random.default_rng(0); v=r.normal(0,1,600); "
                                  "v[150:300]+=1.2; a=mind.across_regimes(v,segments=[(0,150),(150,300),"
                                  "(300,450),(450,600)]); print(round(a['concentration'],2), a['consistent'])",
                          native=True, aliases=("measure the effect separately in each regime",
                                                "does the effect hold in every period or just one",
                                                "per regime breakdown", "did this work in all market conditions",
                                                "is one period carrying my result",
                                                "validate across time periods", "regime by regime table",
                                                "check stability across segments"),
                          semantic="analyze/measure", consumes=(), produces=())
    c.register_capability("Insurance profile (does filtering delete the effect?)",
                          "mind.insurance_profile(values, condition): before excluding the ugly periods, ask "
                          "whether the payoff LIVES there. Reports share_inside, frac_events, lift and "
                          "`premium_inside` -- a minority of events carrying a majority of the value. Measured: "
                          "an effect paid +36bp per event inside storms, +4bp outside; it WAS storm insurance, "
                          "and filtering them removed ~90% of the edge while every other statistic improved. "
                          "Applies to code and caches: pruning a rarely-hit path deletes error-path insurance. "
                          "KEPT NEGATIVE: a premium in a rare state also signals too little data there.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); f=np.zeros(500,bool); f[:60]=True; "
                                  "pay=np.where(f,0.36,0.04); i=mind.insurance_profile(pay,f); "
                                  "print(i['premium_inside'], round(i['lift'],1))",
                          native=True, aliases=("is the payoff concentrated in the times I would exclude",
                                               "should I filter out the bad periods",
                                               "does removing the worst cases hurt me",
                                               "where does my profit actually come from",
                                               "is this effect insurance", "rare event pays for everything",
                                               "safe to prune this rarely used path",
                                               "value concentrated in few events"),
                          semantic="analyze/measure", consumes=(), produces=())

    c.register_capability("ladder_predict", "predict what comes NEXT after a history using the ladder's learned "
                          "HIERARCHICAL alphabet (holographic_ladder) -- the compression<->prediction duality (a "
                          "good compressor is a good predictor). Predicts the next CHUNK and decodes it, so one "
                          "step emits a whole learned pattern, not one flat symbol -- beats a flat n-gram on "
                          "structured data. ABSTAINS to the persistence baseline ('next = last') when it can't beat "
                          "persistence on held-out (a forecast that can't beat 'same as last' is a null result)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.ladder_predict([0,1,2,3]*40)['prediction'])",
                          native=True, aliases=("predict the next symbol", "forecast the next value",
                                                "what comes next in this sequence", "continue a sequence",
                                                "hierarchical prediction", "predict from a learned model",
                                                "anticipate the future from history", "next chunk prediction"))
    c.register_capability("extend_generator", "FORECAST by playing a fitted generator PAST its data "
                          "(holographic_fitgen) -- store the formula, play the future. Given a fit_deterministic "
                          "result, regenerate N samples beyond the end. Refuses beyond the validated window (a "
                          "generator fit on [0,1] evaluated at t=100 is confidently wrong) -- flags valid=False "
                          "when extrapolating too far. The demoscene economy applied to time",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "t=np.linspace(0,1,200); fit=m.fit_deterministic(np.sin(2*np.pi*5*t)); "
                          "print(m.extend_generator(fit,10,200)['valid'])",
                          native=True, aliases=("extrapolate a fitted generator", "play a formula forward",
                                                "forecast from a fitted formula", "extend a generator past its data",
                                                "regenerate future samples", "evaluate a generator at future time"))
    c.register_capability("adaptive_pipeline", "MEASUREMENT-DRIVEN adaptive dispatcher (holographic_ladder) -- "
                          "run identify_level, then route the data to the method its REGIME names instead of "
                          "hard-coding one: ABSTAIN on null-indistinguishable input (the SETI gate -- never 'clean' "
                          "noise into a fabricated signal), FOLD repetitive data (cheap, no climb), CLIMB nested "
                          "structure with the lens picked per-signal (the lens is the analysis window). A readable, "
                          "refusable dispatch on numbers already computed -- no black box",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.agents_and_reasoning.holographic_ladder import _make_planted_corpus; "
                          "print(m.adaptive_pipeline(_make_planted_corpus())['method'])",
                          native=True, aliases=("adaptive pipeline for data", "pick the right method for this data",
                                                "route data to the best method", "choose a strategy automatically",
                                                "abstain if no structure", "dispatch by data regime",
                                                "what should I do with this data", "structure gate"))
    c.register_capability("fit_deterministic", "recover the deterministic GENERATOR that made a noisy 1-D signal "
                          "(holographic_fitgen, the inverse of the ladder): SNAP the data against a baked bank of "
                          "generator families (sine/chirp/gauss/sawtooth/harmonic/am -- harmonic and am are "
                          "Puckette's playable audio tones) then REFINE the winner's params. Returns "
                          "family + params + correlation + residual, or REFUSES when no generator beats the noise "
                          "('no deterministic structure' is a result). Band-limited snap (Quilez Q8) so families "
                          "differing only above the coarse rate tie honestly. If it fits, store bytes not samples",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "t=np.linspace(0,1,400); sig=np.sin(2*np.pi*7*t)+0.1*np.random.default_rng(0).normal(size=400); "
                          "print(m.fit_deterministic(sig)['family'])",
                          native=True, aliases=("which formula made this data", "fit a generator to a signal",
                                                "reverse engineer a signal", "recover the program behind data",
                                                "identify a generator", "what function produced this",
                                                "compress a signal to a formula", "is this signal deterministic"))
    c.register_capability("assemble_pipeline", "find which candidate transform(s) connect an input signal to a "
                          "target output, VALIDATED against a shuffle null (holographic_assemble). Each candidate "
                          "is scored on a HELD-OUT segment and gated by MI-over-shuffle-null: does the REAL input "
                          "drive the output more than a shuffled one? Survivors are returned sorted by significance; "
                          "a candidate passes only if it clears the null (else it is chance alignment, not a "
                          "discovery). The gate that stops 'any random projection works' -- the synesthesia case, "
                          "made honest",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "x=np.random.default_rng(0).normal(size=2000); y=np.tanh(2*x); "
                          "print([s['name'] for s in m.assemble_pipeline(x,y,{'tanh':lambda z:np.tanh(2*z),'lin':lambda z:z})])",
                          native=True, aliases=("assemble a pipeline", "find a transform from x to y",
                                                "which transform connects these signals", "discover a mapping",
                                                "build a path from input to output", "does this input drive that output",
                                                "validate a discovered relationship", "find what drives a signal"))
    c.register_capability("guide_structure", "guide a state toward a goal by ITERATING A PROJECTION "
                          "(holographic_guide) -- the level-generic form of IK / PBD / denoise / resonator, which "
                          "are all the SAME move: repeatedly project a state onto a constraint set until it settles "
                          "(Macklin). Pass a list of projection callables (pin a root to a target, clamp a link "
                          "length, snap to a codebook); the constraints ARE the structure of the space. One solver, "
                          "many costumes -- move this thing legally toward a target",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "r=m.guide_structure(np.array([0.,5.,9.]), [m.guide_pin(0,3.0), m.guide_clamp_link(0,1,1.0)]); print(r['converged'])",
                          native=True, aliases=("iterate a projection", "move a thing toward a target legally",
                                                "constrained movement", "solve inverse kinematics generically",
                                                "project onto constraints", "settle a state under constraints",
                                                "reach a goal under constraints", "constraint satisfaction by projection"))
    c.register_capability("mutual_information", "MUTUAL INFORMATION between two signals (holographic_mutualinfo) "
                          "-- bits of shared information, zero iff independent (discrete or continuous, continuous "
                          "quantile-binned). Raw MI is biased upward by finite samples, so mutual_information_vs_null "
                          "reports MI ABOVE a SHUFFLE NULL -- read `excess` (BITS) as the HEADLINE, z as "
                          "support: z answers is-it-nonzero and inflates with sample size at fixed dependence "
                          "(same ~0.01-bit coupling: z=2.5 at n=3k, z=31.5 at n=48k; canon z=+92 that was "
                          "~0.02 bits -- present, useless). Raw MI without its null is a Rorschach test.",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "x=np.random.default_rng(0).normal(size=2000); print(round(m.mutual_information_vs_null(x, x)['z'],1))",
                          native=True, aliases=("mutual information between two signals", "how much does x tell me about y",
                                                "dependence between two variables", "information shared between signals",
                                                "are two signals related", "statistical dependence", "shared information",
                                                "does one signal predict another", "effect size in bits",
                                                "excess bits of dependence", "mutual information in bits",
                                                "is my z score sample inflated"))
    c.register_capability("Capability URI namespace", "address every public function by a URI "
                          "'family/module/name' (holographic_capuri) so the 42 colliding short names disambiguate "
                          "by PATH -- 'sphere' -> mesh_and_geometry/sdf/sphere vs misc/codegen/sphere. Browse the "
                          "namespace like a context menu (root -> families -> modules -> functions) via prefix "
                          "roll-up, the same S3-style machinery that addresses scene items. The name IS the "
                          "hierarchy, so the view never drifts from the code",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.resolve_capability_uri('sphere')); print(list(m.browse_capabilities('')))",
                          native=True, aliases=("disambiguate a capability name", "resolve a function by path",
                                                "browse capabilities like a menu", "capability namespace",
                                                "address a function by uri", "which module has this function",
                                                "path for a colliding name", "menu of capabilities",
                                                "list capabilities under a prefix"))
    c.register_capability("bank_or_formula", "decide whether to BANK computed values or keep the FORMULA and "
                          "regenerate on demand (holographic_ladder, Quilez Q1 'store the formula not the "
                          "samples'). The demoscene economy as a measured gate: banking pays iff hit_rate*eval - "
                          "lookup > 0 (a miss must build the entry, so only reused evals amortize; break-even = "
                          "lookup/eval). A bank of things a cheap formula gives for free is negative storage",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.bank_or_formula(eval_cost_us=5000, hit_rate=0.9, n_entries=100, bytes_per_entry=4096)['bank'])",
                          native=True, aliases=("should I cache or recompute", "is it worth precomputing this",
                                                "bank versus formula decision", "when to store versus recompute",
                                                "should I bake this or regenerate it", "amortize a precomputed bank",
                                                "is precomputing worth it", "cache or regenerate decision"))
    c.register_capability("chart_space", "chart a holographic ALPHABET as a measured atlas (holographic_ladder): "
                          "march rays between atoms and record where they enter cleanup BASINS (nearest atom "
                          "distinctively nearer than the runner-up). Reports basin coverage, dead zones, and the "
                          "honest verdict structure_over_null (coverage minus a band-limited random null, Quilez "
                          "Q8 -- high-D noise has basins too). For capacity forecasting and codebook placement",
                          example="import numpy as np; import lecore, numpy as np; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "A=np.random.default_rng(0).standard_normal((8,128)); "
                          "print(m.chart_space(A)['structure_over_null'])",
                          native=True, aliases=("map the basins of an alphabet", "atlas of a vector space",
                                                "chart the holographic space", "measure cleanup basins",
                                                "find dead zones in an alphabet", "map the structure of a space",
                                                "raytrace the holographic space", "how well separated are these atoms"))
    c.register_capability("reconstruct_tower", "expand a climbed ladder TOWER back to its ORIGINAL corpus of base "
                          "symbols -- the INVERSE of climb_ladder (holographic_ladder.reconstruct). For a "
                          "sequence-lens tower this is LOSSLESS (reconstruct(climb(corpus)) == corpus exactly); for "
                          "a structure-lens tower it recovers the SET of base part-types (order and counts dropped "
                          "by design). A tower you cannot decompress is useless -- this is the decompress half",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.agents_and_reasoning.holographic_ladder import _make_planted_corpus as mk; "
                          "c=mk(); print(m.reconstruct_tower(m.climb_ladder(c))==c)",
                          native=True, aliases=("decompress a tower", "reconstruct the original from a tower",
                                                "expand a tower to base symbols", "invert the abstraction ladder",
                                                "undo a climb", "get the original data back from a tower",
                                                "expand a promoted atom"))
    c.register_capability("identify_level", "'what am I looking at?' -- classify a CORPUS by which ladder "
                          "operations pay on it (holographic_ladder), returning MEASUREMENTS not a label: is there "
                          "a level above it, does compression survive a shuffle-null (high-D noise has basins too, "
                          "so only gain-over-null counts), which lens fits (sequence vs structure, picked not "
                          "guessed), and the regime (repetitive / nested-structured / irreducible). The step-0 "
                          "question of a climb",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.agents_and_reasoning.holographic_ladder import _make_planted_corpus; "
                          "print(m.identify_level(_make_planted_corpus())['regime'])",
                          native=True, aliases=("what level of abstraction is this", "classify a corpus",
                                                "is there structure in this data", "is there a level above this",
                                                "what am i looking at", "does this data have hierarchy",
                                                "which lens fits this data", "is this data compressible or noise"))
    c.register_capability("Abstraction ladder (climb)", "climb a CORPUS into a TOWER of abstraction levels "
                          "(holographic_ladder): consolidate -> find patterns -> promote to a new alphabet -> "
                          "repeat, STOPPING when the MDL compression gain drops below a floor. The generic form of "
                          "the seven-step loop run by hand for letters->words and verts->parts->scene. Returns a "
                          "tower with stable hashlib atom ids and a loud terminal refusal (a shallow ceiling is a "
                          "RESULT -- most data tops out fast). A zlib pre-gate prunes levels that cannot pay before "
                          "the expensive pass (Quilez 'don't march empty space')",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.agents_and_reasoning.holographic_ladder import _make_planted_corpus; "
                          "print(m.ladder_summary(m.climb_ladder(_make_planted_corpus())))",
                          native=True, aliases=("level up my representation", "find hierarchy in this data",
                                                "automatic abstraction", "recursive chunking",
                                                "build a tower of patterns", "keep compressing until it stops paying",
                                                "climb a corpus into levels", "discover nested structure",
                                                "hierarchical pattern discovery", "compress into an alphabet of patterns"))
    c.register_capability("Atmosphere (fog & light shafts)", "atmospheric post-effects over a rendered image "
                          "(holographic_atmosphere, W16): depth_fog fades pixels toward a fog colour by distance "
                          "(exponential Beer-Lambert -- the air of a scene in one pass), and light_shafts streaks "
                          "god rays outward from an on-screen light/sky by radial blur (Mitchell GPU Gems 3). "
                          "Cheap screen-space passes -- no volume marching. The atmosphere of iq's cathedral",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "img=np.full((32,32,3),0.3); d=np.full((32,32),5.0); print(m.depth_fog(img,d,density=0.2).shape)",
                          native=True, aliases=("volumetric fog", "depth fog", "atmospheric fog", "light shafts",
                                                "god rays", "sun rays", "crepuscular rays", "add fog to a render",
                                                "hazy atmosphere", "volumetric light", "foggy scene"))
    c.register_capability("scene_cost", "estimate the per-ray evaluation COST of an SDF scene (W2) -- an "
                          "ALU/machine-model annotation for deciding if a scene raymarches in real time. Returns "
                          "alu (approx ops per map() call), nodes, depth, iterative (has a fractal/tiling loop), "
                          "and a plain-language verdict (cheap / moderate / heavy). Know the price before you ship "
                          "the scene",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_sdf import menger; print(m.scene_cost(menger(5,1.0))['verdict'])",
                          native=True, aliases=("estimate shader cost", "cost of an sdf tree", "how expensive is this scene",
                                                "is this scene realtime", "shader complexity", "raymarch budget",
                                                "sdf performance estimate", "will this run at 60fps"))
    c.register_capability("SDF primitive pack", "the everyday SDF PRIMITIVE leaves for building scenes "
                          "(holographic_sdf): sphere, box, torus, cylinder, plane, menger -- and the W8 additions "
                          "CAPSULE (a pill/limb), CONE (a spike/funnel), OCTAHEDRON (a crystal/gem, exact), and "
                          "ELLIPSOID (iq's bounded approx). All are exact distances except ellipsoid; capsule/cone/"
                          "octahedron emit to a GLSL shader. Compose with union/smooth_union/domain warps into any "
                          "scene",
                          example="from holographic.mesh_and_geometry.holographic_sdf import capsule, octahedron; "
                          "s = octahedron(0.8).union(capsule(1.0, 0.2).translate([1.0,0,0])); print(s.eval([[0,0,0]]).round(3))",
                          native=True, aliases=("capsule sdf", "cone sdf", "ellipsoid sdf", "octahedron sdf",
                                                "pill shape", "crystal shape", "gem sdf", "sdf primitive",
                                                "basic sdf shapes", "cylinder sdf", "sphere sdf", "box sdf",
                                                "add a primitive to a scene", "sdf building blocks"))
    c.register_capability("NURBS", "Non-Uniform Rational B-Splines (holographic_nurbs) -- the CAD/industrial-design "
                          "surface primitive. nurbs_curve and nurbs_surface add per-control-point WEIGHTS to a "
                          "B-spline, which is what lets a NURBS represent CONICS EXACTLY (a circle, sphere, "
                          "cylinder) -- a polynomial B-spline only approximates them. nurbs_surface_mesh "
                          "tessellates a patch into a mesh for the render/voxelise pipeline; nurbs_circle proves "
                          "the exactness (radius to 1e-12). Built on the existing Cox-de Boor basis in homogeneous "
                          "coordinates",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "import numpy as np; c=m.nurbs_circle(radius=2.0,n=100); print(round(float(np.linalg.norm(c[0,:2])),6))",
                          native=True, aliases=("nurbs surface", "nurbs curve", "rational bspline",
                                                "rational b-spline", "nurbs to mesh", "weighted control points",
                                                "evaluate a nurbs patch", "cad surface", "exact circle spline",
                                                "tensor product spline surface", "nurbs patch"))
    c.register_capability("Voxelization", "turn a mesh or an SDF into a VOXEL occupancy grid (holographic_voxelize). "
                          "voxelize_mesh uses the generalised WINDING NUMBER (Jacobson 2013) -- robust to "
                          "non-watertight / self-intersecting meshes, unlike ray-parity; voxelize_sdf is the "
                          "O(voxels) fast path for an implicit. Get solid-voxel centres as a point cloud, or run "
                          "occupancy_to_mesh (surface_nets) to close the round trip mesh -> voxels -> mesh. Also "
                          "exposes mesh_winding_number as a robust inside/outside test",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_curves import torus_knot, sweep_tube; "
                          "V,F=sweep_tube(torus_knot(120,2,3),radius=0.18,closed=True); occ,o,s=m.voxelize_mesh(V,F,res=24); print(int(occ.sum()))",
                          native=True, aliases=("voxelize a mesh", "mesh to voxel grid", "occupancy grid from a mesh",
                                                "sample an sdf onto a voxel grid", "dense voxel volume",
                                                "point in mesh test", "inside outside mesh", "winding number",
                                                "voxel point cloud", "mesh to voxels to mesh", "rasterize a mesh"), consumes=('mesh', 'sdf'), produces=('field',))
    c.register_capability("Curves, splines & knots", "parametric CURVES and geometry (holographic_curves): BEZIER "
                          "(de Casteljau), CATMULL-ROM (interpolating, centripetal), B-SPLINE (Cox-de Boor); "
                          "tangent + rotation-minimizing / Frenet FRAMES; arc-length resampling; SWEEP a profile "
                          "along a curve into a watertight TUBE mesh; and parametric primitives -- TORUS KNOTS, "
                          "TREFOIL, HELIX, SUPERELLIPSOID, GYROID field, KLEIN BOTTLE. A curve drives a camera "
                          "path, a tube centreline, or a scatter path -- one abstraction, many uses",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "knot=m.torus_knot(n=200,p=2,q=3); V,F=m.sweep_tube(knot,radius=0.12,closed=True); print(V.shape,F.shape)",
                          native=True, aliases=("bezier curve", "catmull rom spline", "b-spline", "bspline",
                                                "evaluate a spline", "sample points along a curve", "curve tangent",
                                                "frenet frame", "rotation minimizing frame", "arc length of a curve",
                                                "sweep a profile along a curve", "tube along a path", "bezier tube",
                                                "torus knot", "trefoil knot", "superellipsoid", "gyroid",
                                                "klein bottle", "helix", "spline camera path", "parametric curve",
                                                "knot geometry", "make a tube from a curve"), consumes=(), produces=('curve',))
    c.register_capability("audio_param_bus", "drive scene PARAMETERS from audio (W5') -- build a per-frame bus of "
                          "band-energy envelopes (bass / low-mid / high-mid / treble, normalised 0..1) plus an "
                          "onset/beat signal, then subscribe a scene knob to a band. bus.subscribe(band, lo, hi, "
                          "frame) maps a band onto a parameter range (metaball viscosity from the bass, palette "
                          "phase from the treble); bus.onset gives beats. Reuses the existing STFT -- only the "
                          "band binning is new. The wire that makes a demo react to music",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "t=np.linspace(0,1,22050,endpoint=False); sig=np.sin(2*np.pi*60*t); "
                          "bus=m.audio_param_bus(sig, 22050); print(round(bus.subscribe(0,0.1,0.6,frame=5),2))",
                          native=True, aliases=("audio reactive parameters", "drive parameters from audio",
                                                "music reactive demo", "band energy envelope", "beat driven scene",
                                                "onset to parameter", "sync visuals to audio", "audio param bus",
                                                "frequency bands over time", "make a demo react to music"))
    c.register_capability("orbit_trap_render", "render an SDF scene coloured by ORBIT TRAP -- the signature Quilez "
                          "fractal look, in one call. Sphere-traces every pixel, tracks each ray's closest "
                          "approach to a trap set (point / origin / axis / plane), and maps that scalar through a "
                          "cosine palette, Lambert-lit. Composes with any domain-warped SDF (fold/repeat/twist). "
                          "This is orbit traps + cosine palettes, the two halves meeting",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                          "cam=m.camera(eye=(1.6,1.2,2.4)); img=m.orbit_trap_render(sphere(0.5).repeat((1.0,1.0,1.0)), cam, width=64, height=64); print(img.shape)",
                          native=True, aliases=("orbit trap coloring", "fractal orbit trap render",
                                                "color a raymarch by closest approach", "quilez orbit trap look",
                                                "trap set coloring", "render with orbit traps",
                                                "iq fractal colors", "closest-approach coloring"))
    c.register_capability("sphere_trace_trapped", "sphere-trace rays AND return each ray's ORBIT TRAP -- the "
                          "closest approach of its march to a trap set (the Quilez fractal-colouring scalar). "
                          "Returns (hit, t, pos, trap_val); hit/t/pos are identical to sphere_trace, trap_val is "
                          "the per-ray minimum distance to the trap (point/origin/axis/plane). Feed trap_val "
                          "through a cosine palette. Use orbit_trap_render for the whole render in one call",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                          "h,t,p,tv=m.sphere_trace_trapped(sphere(0.5), np.array([[0,0,3.]]), np.array([[0,0,-1.]]), trap_kind='origin'); print(round(float(tv[0]),2))",
                          native=True, aliases=("closest approach along a ray", "orbit trap value per ray",
                                                "raymarch with trap", "trap distance per pixel",
                                                "nearest approach to a trap set"))
    c.register_capability("ascii_animate", "render an ASCII ANIMATION to a list of text frames (holographic_ascii) "
                          "-- the demoscene 'tunnel in a terminal' as data. Pass frame(i,u) or frame(u) (u = "
                          "normalised time) returning an image, an SDF node / DSL text (raymarched), or a 2-D "
                          "field sampler each frame; get back n deterministic strings to diff, write as a reel, "
                          "or drive your own loop. For live in-terminal playback with timing use "
                          "holographic_ascii.ascii_play",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_sdf import torus; "
                          "frames=m.ascii_animate(lambda u: torus(0.5,0.15).twist(u), n=8, width=40, mode='braille'); print(len(frames))",
                          native=True, aliases=("animate ascii", "ascii animation", "text animation",
                                                "animate in the terminal", "render an animation as text frames",
                                                "ascii movie", "terminal animation", "play frames as ascii",
                                                "animated ascii art", "render a sequence of frames to text"))
    c.register_capability("Text generation", "GENERATE text on the VSA substrate: generate(seed, length, temperature) "
                          "and generate_structured (n-gram / beam), respond(query) for a query-conditioned reply, and "
                          "answer(question) / answer_text for factual answers. The engine's write-a-sentence faculties",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); mind.generate('once upon a', length=120); mind.respond('describe a sunset'); mind.answer('what is gravity')",
                          native=True, aliases=("generate text", "write", "write a sentence", "write a paragraph",
                                                "text generation", "compose text", "respond", "reply", "answer a question",
                                                "language model", "ngram", "sentence", "paragraph", "prose"))
    c.register_capability("Language learning", "TEACH the mind language natively: read (read a corpus), "
                          "learn_dictionary / learn_vocabulary (word meaning from definitions -- including the vendored "
                          "dictionary), learn_encyclopedia (relational facts + is_a taxonomy), and learn_sequence "
                          "(order/grammar). The language CURRICULUM -- definitions, then facts, then reading",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); mind.read(corpus); mind.learn_vocabulary(words); mind.learn_encyclopedia(facts)",
                          native=True, aliases=("learn from a corpus", "train on text", "teach the model", "teach language",
                                                "language curriculum", "learn word meanings", "learn a language",
                                                "read a corpus", "curriculum", "learn vocabulary", "learn facts"))
    c.register_capability("Utilities & helpers", "the engine's cross-cutting UTILITY tools: content addressing & "
                          "hashing (uri), tamper-evident verification (verify), erasure/rateless coding for reliability "
                          "(fountain), chunked delta chains with integrity proofs (deltachain), versioned rollback "
                          "history (history), lossless compression (compress/codec), and the determinism contract "
                          "(determinism). The plumbing every faculty leans on",
                          example="from holographic.io_and_interop.holographic_uri import address_from_content, make_key; from holographic.misc.holographic_verify import CompositionTree",
                          native=True, aliases=("utility", "helper", "tool", "hash", "checksum", "content address",
                                                "content id", "verify integrity", "verify data integrity", "check data integrity",
                                                "is my data corrupted",
                                                # ^ the FULL user phrasing, not just the two-word stem: this
                                                # entry sat at rank 3 of 3 on "verify data integrity" -- inside
                                                # the assertion by one slot -- until a GPU capability whose
                                                # does() honestly mentions "verify" and "data" landed at rank 2
                                                # and pushed it out. Additive fix: strengthen the target, never
                                                # weaken the honest neighbour. (Ported here when the catalog
                                                # was split into parts; the pin lives in test_routing_pins.)
                                                "tamper", "erasure code", "reliability",
                                                "delta chain", "version history", "rollback", "compress", "determinism",
                                                "plumbing", "reliability code"))
    # --- describe a scene in words, build it, adjust named objects, render or simulate ---
    # rev. 9 discoverability audit: the pinned route probe "describe a scene and build it" shipped RED. Mechanics,
    # measured: the tokenizer stopwords "build/make/create", so the probe reduces to {describe, scene}; this entry
    # then maxes at 2.5 (2 overlap + 0.5 name bonus for "scene") while the essay-length `does` of "The scene's own
    # SDF, emitted" soaks up 1.5 as runner-up -- dominance 0.625 x strength 0.833 = confidence 0.521 < 0.6, and
    # route() said "choose" for its own headline skill. "Describe" in the NAME is honest (it is what the skill
    # does) and restores the name bonus the stopword list took away: 3.0 vs 1.5 -> confidence 0.667 -> "act".
    c.register_capability("Describe a scene (scene from description, semantic)", "DESCRIBE a 3-D scene in plain words and the engine "
                          "builds it, then you ADJUST it by talking to named objects: mind.build_scene('a big red metal "
                          "sphere and a small blue glass box on a sunny day') returns a live SemanticScene; then "
                          "scene.adjust('make the sphere bigger'), scene.adjust('change the box to metal'), "
                          "scene.set('the red sphere', material='glass'), scene.render(), scene.simulate(). NAME objects "
                          "to reference them easily -- scene.name('the red sphere', 'hero') or scene.adjust('call the box "
                          "crate'), then scene.adjust('make hero glass'); scene.rename('hero','champion'). PAINT a "
                          "procedural TEXTURE by talking to it -- scene.adjust('give hero a rusty texture'), scene.paint("
                          "'crate', 'marbled') (rusty/marbled/mossy/cloudy/lava/striped/noisy) -- and scene.render() "
                          "paints it on. Set the MOOD with a time-of-day/lighting word in the description -- 'a white "
                          "sphere at sunset', '...at noon', '...on an overcast day', 'a dramatic ...' -- which sets the "
                          "sun direction, colour and ambient (noon/morning/afternoon/sunset/sunrise/golden/dusk/overcast/"
                          "night/moonlit/studio/dramatic). Or CHANGE the lighting on a LIVE scene by talking to it -- "
                          "scene.adjust('make it sunset'), scene.adjust('studio lighting'), scene.adjust('moody') -- which "
                          "sets environment['lighting'] and render() honours it (bare 'make it golden' stays a material "
                          "change; 'golden hour' is the preset). scene.options()['lighting'] lists the presets. Relative "
                          "BRIGHTNESS too -- scene.adjust('make it brighter'), scene.adjust('dimmer'), scene.adjust('much "
                          "darker') -- scales environment['sun_scale'] (compounds, clamped) which the fast renderer applies. "
                          "Place objects RELATIVE to each other -- scene.adjust('put the sphere on top of the box'), "
                          "scene.adjust('move the cone next to the sphere'), '...inside...', '...behind...', '...in front "
                          "of...' -- deterministic exact layout (sets the object's relation; the realizer re-positions it). "
                          "MOVE or SCALE by an amount -- scene.adjust('move the sphere left 2'), scene.adjust('nudge the "
                          "box up'), scene.adjust('scale the sphere up'), scene.adjust('make the box twice as big'), "
                          "'halve it' -- exact offsets/scale (+x right, +y up, +z toward camera). Attach an EXTERNAL image file as a texture -- scene.attach_texture_file('the "
                          "sphere', 'project/textures/wave.png') -- and the scene tracks it in an AssetLibrary: if the "
                          "files move, scene.set_asset_roots([...]) + scene.resolve_assets() (or scene.relink(one, new)) "
                          "re-find them and render() reloads them, falling back to the object's colour if one is missing. "
                          "When a command is unclear it SUGGESTS rather than fails -- scene.interpret(cmd) "
                          "previews what it understood + 'did you mean?' hints, scene.options() lists what you can say, "
                          "scene.feedback holds the last report. Or wrap an existing object list with "
                          "mind.semantic_scene(objects). Controlled vocabulary, deterministic",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); scene = mind.build_scene('a red metal sphere and a blue box'); scene.name('the sphere','hero'); scene.adjust('give hero a rusty texture'); scene.render()",
                          native=True, module="scene_semantic", aliases=("scene", "describe a scene", "build a scene", "make a scene", "create a scene",
                                                # the ROUTER probe, verbatim: "Describe to document" (J-3D) shares this
                                                # entry's whole vocabulary, and route() decays as the catalog grows, so
                                                # the two tied at 0.5 and the decision fell act -> choose. Additive fix:
                                                # give the incumbent the exact phrasing it owns; the newcomer keeps its
                                                # own ("scene document", "handles for an agent"). Pinned in
                                                # test_routing_pins.test_describe_a_scene_routes_to_act.
                                                "describe a scene and build it", "describe it and build it",
                                                "describe and build", "build what I describe", "build from a description",
                                                "scene from text", "3d scene", "adjust the scene", "semantic scene",
                                                "named objects", "make the sphere bigger", "change the material", "render a scene",
                                                "text to 3d", "text to scene", "scene editor", "reference objects by name",
                                                "name an object", "rename object", "give it a texture", "rusty texture", "paint the scene",
                                                "cylinder", "cone", "torus", "donut", "pyramid", "tube", "pillar", "ring shape",
                                                "teal", "navy", "silver", "brown", "lavender", "crimson", "colours", "shapes",
                                                "at sunset", "at noon", "golden hour", "time of day", "lighting", "overcast",
                                                "dramatic lighting", "moody lighting", "studio lighting", "night scene", "sunrise",
                                                "make it sunset", "make it night", "change the lighting", "adjust the lighting",
                                                "set the lighting", "set the mood", "make the scene dramatic", "make it moody",
                                                "control lighting semantically", "adjust scene lighting",
                                                "make it brighter", "make it dimmer", "brighten the scene", "dim the scene",
                                                "make it darker", "turn up the brightness", "brighter", "dimmer",
                                                "put the sphere on top of the box", "move it next to", "place one object on another",
                                                "put one inside another", "relative layout", "arrange objects", "position objects",
                                                "on top of", "next to", "stack objects", "attach one object to another",
                                                "move the sphere left", "nudge the object up", "shift it right", "translate an object",
                                                "scale the sphere up", "make it twice as big", "shrink an object", "resize an object",
                                                "move an object by an amount"))
    c.register_capability("Instancing (shared definition + type-safe binding)", "place ONE shared definition many "
                          "times so editing it once updates every copy (edit-once): mind.shared_definition('chair', "
                          "mesh, 'metal') then scene.place(defn, transform) in mind.instanced_scene(); repaint the "
                          "definition and all instances change. The material<->geometry binding is TYPE-CHECKED at "
                          "compose time -- a surface material only binds to a mesh, a volumetric one (fog/smoke/fire) "
                          "only to a volume -- so a bad binding is refused, not rendered wrong. flatten_surface() "
                          "materialises the surface instances into one mesh. CMP4",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); chair = mind.shared_definition('chair', box_mesh, 'metal'); s = mind.instanced_scene(); s.place(chair); chair.set_material('glass')",
                          native=True, aliases=("instance", "instancing", "shared definition", "edit once", "duplicate",
                                                "reuse geometry", "material binding", "surface volume", "place copies",
                                                "instanced scene", "clone", "prototype"))
    c.register_capability("Messaging across machines (distributed bus)", "the same publish/subscribe/send bus, spread "
                          "across nodes: mind.distributed_bus(peers, token, node_id) publishes locally AND fans out to "
                          "peer nodes (each running holographic_distbus.serve_bus), so agents on different machines "
                          "share topics -- a swarm coordinates across the farm the way it does in one process. Received "
                          "messages deliver local-only (no loops), dedup by a global id, and a dead peer never blocks "
                          "the publisher. Bound a mailbox (open_mailbox(maxlen=)) for backpressure at high fan-out.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); bus = mind.distributed_bus(['hostB:9100'], token, node_id='A'); from holographic.scene_and_pipeline.holographic_distbus import serve_bus  # serve_bus(bus, port=9100, token) in a thread",
                          native=True, aliases=("distributed bus", "messaging across machines", "cross-node messaging",
                                                "swarm messaging", "pub sub across nodes", "fan out", "gossip",
                                                "backpressure", "bounded mailbox", "flow control", "topic across nodes"))
    c.register_capability("Distributed compute across machines (farm)", "run the same partition-and-reduce work across "
                          "a FARM of machines. Each node runs holographic.scene_and_pipeline.holographic_coordinator.serve_worker(workers={name: fn}); "
                          "mind.farm(['host1:9000','host2:9000'], token).run(buckets, worker_name, cache, reduce) "
                          "round-robins the buckets across nodes and reassembles by the monoid reducer -- the same call "
                          "as the local pool, just cross-machine. SAFE by design: workers run BY NAME (a node only runs "
                          "workers it registered), so no code crosses the wire, only data. stdlib sockets/JSON.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); from holographic.scene_and_pipeline.holographic_coordinator import serve_worker; serve_worker(port=9000, workers={'sum': fn})  # then: mind.farm(['host:9000'], token).run(buckets, 'sum', None, reduce_sum)",
                          native=True, aliases=("farm", "distributed compute", "cluster", "network farm", "worker node",
                                                "serve_worker", "render farm", "compute across machines", "scale out",
                                                "map reduce", "parallel across nodes", "grid"))
    c.register_capability("Who's online (presence registry)", "mind.registry tracks live actors: announce(principal) is "
                          "a heartbeat, registry.list(kind=, workspace=) discovers who's here, is_online() checks one, "
                          "and an actor that stops heart-beating for `ttl` seconds drops out on its own. Rides the "
                          "mind's bus so presence is visible across nodes -- how a swarm or farm finds its peers.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); mind.registry.announce(agent); online = mind.registry.list(kind='agent'); mind.registry.is_online(agent)",
                          native=True, aliases=("registry", "presence", "who is online", "heartbeat", "discover peers",
                                                "list agents", "who's connected", "liveness", "roster", "online users",
                                                "node discovery"))
    c.register_capability("Invite guests and share selectively (access control)", "control who reads what. mind.invite("
                          "kind, grants) mints a token admitting a guest with specific initial read grants; mind.admit("
                          "code, id) redeems it into a scoped Principal that reads ONLY what it was granted (default: "
                          "nothing but its own namespace) and writes only its own. mind.grant / mind.revoke share and "
                          "un-share namespaces later; holographic_access.require_readable is the read chokepoint (the "
                          "symmetric twin of the DB's write-only-your-own rule).",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); code = mind.invite(kind='user', grants={'read':['lab/scene']}); g = mind.admit(code, 'visitor'); mind.grant(g, read='lab/notes')",
                          native=True, aliases=("access control", "invite", "grant", "revoke", "permissions", "share",
                                                "who can read", "admit a guest", "invite token", "selective sharing",
                                                "read grant", "guest access", "authorize", "private namespace"))
    c.register_capability("Fork and apply a shared world (workspace)", "mind.workspace.fork(name) hands out a "
                          "copy-on-write editing view of a named world (a set of vector SLOTS): reads fall through to "
                          "the shared base, writes accumulate in the fork's private .delta, so your edits don't touch "
                          "the shared world (or another fork) until you reconcile. Feed the deltas to mind.merge_forks, "
                          "then mind.apply(merged, world=name) writes the agreed edits back. Closes the "
                          "fork -> edit -> merge -> apply loop; a world is a seed + deltas, so only the sparse changes "
                          "travel.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); f = mind.workspace.fork('lab'); f.set('sky', v); mind.apply(mind.merge_forks([f.delta, other])['merged'], world='lab')",
                          native=True, aliases=("workspace", "fork a world", "apply changes", "copy on write", "world",
                                                "shared world", "checkout", "branch a world", "edit in isolation",
                                                "commit changes", "seed and deltas", "single-player fork"))
    c.register_capability("Merge forked worlds (fork/merge)", "mind.merge_forks(forks, policy, tol) reconciles several "
                          "forked copies of a world, each a {slot: vector} delta. Slots the forks AGREE on merge "
                          "conflict-free into the consensus (pairwise opponent divergence below tol, matching leOS's "
                          "pairwise convention); slots they DISAGREE on are handled by policy: 'select' surfaces the "
                          "conflict for a human, 'auto' keeps only agreements, 'left'/'right'/callable resolve it. "
                          "Because a world is a seed + deltas, forking to single-player and merging back is cheap. "
                          "Returns {merged, conflicts}.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); res = mind.merge_forks([mine, theirs], policy='select'); apply(res['merged']); resolve(res['conflicts'])",
                          native=True, aliases=("merge", "merge forks", "fork and merge", "reconcile", "combine worlds",
                                                "resolve conflicts", "multiplayer merge", "branch and merge", "diff merge",
                                                "three-way merge", "collaborative edit", "sync changes"))
    c.register_capability("Scoped identity for any actor (Principal)", "mind.principal(id, workspace, kind) gives an "
                          "agent, user, service, or peer leCore instance ONE scoped identity where isolation is the "
                          "default: a private database namespace (it writes only there), a directed inbox topic (it "
                          "reads only its own messages, sender-stamped), a provenance role that tags everything it "
                          "contributes (holographic_provenance.source_role / from_external), and an optional private "
                          "learning overlay. Signals and state can't cross between principals -- so multiplayer "
                          "workspaces, agent swarms, and guest peer nodes are the same isolation solved once.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); alice = mind.principal('alice', workspace='lab', kind='user'); alice.send(mind.bus(), to='bob', payload={...}); alice.poll(mind.bus())",
                          native=True, aliases=("principal", "identity", "scoped identity", "per-agent state",
                                                "per-user namespace", "multiplayer", "multi-user", "swarm", "agent isolation",
                                                "inbox", "directed message", "provenance", "source role", "who sent this",
                                                "guest", "peer node", "federation", "workspace member"))
    c.register_capability("Serve leCore as a tool (/tools + /invoke)", "run the HTTP service (holographic_service.serve) "
                          "and any harness, LLM, or another leCore drives this node over two endpoints: GET /tools "
                          "returns the manifest of every public faculty (name, description, params); POST /invoke with "
                          "{name, args} runs one faculty and returns its result as JSON. Token-gated; private methods "
                          "are refused. This is leCore AS a tool provider -- the same shape every node speaks.",
                          example="from holographic_service import serve; serve(host='127.0.0.1', port=8080, token='secret')  # GET /tools ; POST /invoke {name,args}",
                          native=True, aliases=("serve as a tool", "tool server", "/tools", "/invoke", "expose faculties",
                                                "http api", "call leCore remotely", "function calling", "tool manifest",
                                                "let an agent use leCore", "let an llm call leCore"))
    c.register_capability("Use external tools (remote nodes / LLMs / commands)", "leCore CALLS tools in the same shape it "
                          "serves them. holographic.io_and_interop.holographic_toolclient.remote_tools(base_url, token) fetches another node's "
                          "/tools and yields each as a callable RemoteTool (its run(args) POSTs to that node's /invoke). "
                          "mind.attach_llm(callable) wires an LLM (any text->text, no SDK). mind.orchestrator.register / "
                          "register_command / register_remote add remote tools, shell programs (allowlisted), and whole "
                          "remote nodes so a planner can chain local faculties, remote tools, LLMs, and commands "
                          "uniformly.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); for t in remote_tools('http://host:8080', token='x'): mind.orchestrator.register(t)  # + mind.attach_llm(llm); mind.orchestrator.register_command('ffmpeg', ['ffmpeg','-i','{}'])",
                          native=True, aliases=("call a tool", "remote tools", "use an llm", "attach llm", "orchestrator",
                                                "register a tool", "run a command", "shell command tool", "call another node",
                                                "chain tools", "planner", "toolclient", "peer node", "federation"))
    c.register_capability("Agreement across estimates (opponent)", "given TWO estimates of the SAME thing (two models, "
                          "two solvers, two forked worlds, two farm nodes), mind.opponent_channels(a, b) decomposes "
                          "their disagreement (opponent-processing, ported from leOS) into: agreement (what both see), "
                          "a_exclusive / b_exclusive (what only each sees), magnitude_dispute, PURPLE (a_exclusive + "
                          "b_exclusive -- the emergent signal in NEITHER alone), and divergence_score (the angular "
                          "disagreement). Act on the agreement when divergence is small; surface the conflict when "
                          "it's large. classify() names the disagreement type; blend() mixes them by the channels.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); ch = mind.opponent_channels(est_a, est_b); if ch['divergence_score'] < 0.2: use ch['agreement']  # else look at ch['purple']",
                          native=True, aliases=("opponent", "agreement", "disagreement", "purple channel", "consensus",
                                                "vote", "voting", "ensemble", "combine estimates", "reconcile",
                                                "who agrees", "divergence", "abstain when uncertain", "cross-check",
                                                "opponent channels", "emergent signal", "leos opponent"))
    c.register_capability("Refine loop (produce / critique / adjust)", "mind.refine(produce, critique, adjust, accept, "
                          "budget) makes a result, has a CRITIC score it (a metric, opponent agreement, a model, or a "
                          "human), adjusts, and retries until it's good enough or the budget runs out -- the pipeline "
                          "middle that sits leCore between a big compute and a checker. Returns {result, score, "
                          "accepted, tries}. The callable-critic sibling of project_onto_constraints.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); log = mind.refine(produce=lambda: gen(), critique=score, adjust=lambda r,s: tweak(r,s), accept=0.9)",
                          native=True, aliases=("refine", "iterate", "produce critique adjust", "retry until good",
                                                "optimization loop", "analysis by synthesis", "draft and revise",
                                                "improve until accepted", "critic loop", "feedback loop"))
    c.register_capability("Purity & effect analysis (the gate a cache needs)", "decide whether a Python function is "
                          "PURE -- side-effect free and deterministic -- so a shape-keyed cache can safely memoize it. "
                          "mind.function_purity(source, name) is the verdict; mind.purity_report(source) explains every "
                          "function; mind.purity_scan(root) runs the whole tree. Built from stdlib `ast` alone: no "
                          "linter dependency, no constitutional exception. CONSERVATIVE BY CONTRACT -- a wrong 'impure' "
                          "costs a cache miss; a wrong 'pure' silently corrupts a cache and everything downstream, so an "
                          "unresolved callee, an unrecognised method and any attribute write are impure. Escape analysis "
                          "is implemented: mutating a container the function itself allocated is invisible from outside, "
                          "so `out = []; out.append(x)` is pure. THE CORRECTION: the analysis is closed over the CALL "
                          "GRAPH, because a function that calls an impure function is impure however clean its own body "
                          "looks. Measured on this tree (2,154 module-level functions): a LOCAL rule that ignores calls "
                          "reports 54.3% pure; the sound fixpoint reports 32.1%. The backlog's '76.0% with escape "
                          "analysis' is a local-rule number, and a local purity rule is unsound for a cache -- so "
                          "purity_report carries BOTH figures and never lets the flattering one travel alone.",
                          example="src = 'def f(xs):\\n    out = []\\n    for x in xs: out.append(x*2)\\n    return out\\n'; "
                                  "print(mind.function_purity(src, 'f'), mind.purity_report(src)['fraction'])",
                          native=True, aliases=("purity", "pure function", "side effects", "effect analysis",
                                                "decide whether a python function is pure", "is this function pure",
                                                "can i cache this function", "memoization gate", "linter",
                                                "static analysis", "escape analysis", "call graph", "ast analysis",
                                                "safe to memoize", "deterministic function", "impure"))
    c.register_capability("Recursive factoring (past the resonator's cliff)", "factor a DEEP bound composite by "
                          "solving a SHALLOW problem over composed chunks, then expanding each chunk by LOOKUP instead "
                          "of by search. mind.recursive_factor(composite, codebook, vocab) tries each chunk level "
                          "deepest-first, VERIFIES every candidate by re-composition, and falls back one level on "
                          "failure -- so it is verified correct or reported unsolved, never a silent guess. The "
                          "codebook is R1's mind.learn_chunks output: one codebook family, second consumer. MEASURED "
                          "(D=4096, 32 symbols, MAP binding): the flat resonator is a CLIFF, not a slope -- 93.3% at "
                          "depth 2, 60.0% at depth 4, 0.0% at depth 5 and beyond. With promoted chunks (62 pairs -> 64 "
                          "quads) a depth-8 composite factors at 90.0% here vs 0.0% flat, and 3x FASTER (a 64-entry "
                          "codebook is a smaller search space than V^8). HONEST SCOPE: below the cliff recursion is a "
                          "modest gain at 5x the cost (depth 4: 93.3% vs 86.7% flat) -- use it past the cliff. The "
                          "condition is R1's: no structure, no dividend, and mind.structure_score measures it first. "
                          "Note MAP binding is self-inverse, so a leaf appearing twice CANCELS -- mind.reduce_involution "
                          "recovers the minimal multiset, and a non-minimal expansion can still be exactly correct.",
                          example="import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); vocab = mind.map_codebook(16, 2048, seed=0); cb = mind.learn_chunks(stream); "
                                  "res = mind.recursive_factor(mind.map_bind(*[vocab[i] for i in [0,1,2,3,4,5,6,7]]), cb, vocab)",
                          native=True, aliases=("recursive factoring", "factor using learned chunks", "chunk levels",
                                                "factor a deep composite that the resonator cannot handle",
                                                "my resonator fails past four factors", "resonator cliff",
                                                "break a bound product into eight parts", "deep factorization",
                                                "macro codebook factoring", "expand by lookup", "verify gate",
                                                "map bind", "involution", "self inverse binding", "multiset factors"), module="resonator", consumes=("hypervector",), produces=("hypervector",))
    c.register_capability("Fast preview render (a rough look, 12x, for the see-fix loop)", "a rough look in 3.81s where the full render takes 45.85s (12.0x, same 240x180 output, mean abs err 0.0159) -- for the see->fix loop, where eight looks beat one render. THE OBVIOUS PLAN WAS WRONG: 'render small and upscale' buys under 2x, because the tracer is DISPATCH-bound at preview sizes (16x the pixels cost 2.8x the time). The win is PASSES -- max_bounce=1 is 2.76x, quality='draft' another 1.72x. Upscaling is an OUTPUT-SIZE lever, not a speed one. Trade: one bounce means no indirect light, so a preview is flatter with darker shadows",
                          example="import lecore; m=lecore.UnifiedMind(); s=m.new_scene(); s.add(name='b', geometry=m.shape('sphere'), material='copper'); print(m.render_preview(s, m.camera(eye=(2,2,3), target=(0,0,0)), 64, 48).shape)",
                          native=True, module="scene_render",
                          aliases=("make a quick preview before the full render",
                                   "draft quality fast render", "render small and enlarge",
                                   "my render is too slow to iterate on", "rough look at my scene",
                                   "speed up my render", "preview the scene quickly",
                                   "low quality fast render", "iterate faster on a 3d scene",
                                   "cheap render to check framing", "render a thumbnail of my scene"))
    c.register_capability("Object handles over /invoke (name a live object across calls)", "POST /invoke new_scene used to return '<Scene object at 0x7fe17ba58fe0>' -- a memory address is not a handle, so the whole Scene family was listed in /tools and IMPOSSIBLE to call. Now every un-serialisable result also carries ref:Type:N, and any ref passed as an argument resolves back to the live object. With scene_add/scene_edit/scene_remove/scene_undo an HTTP-only agent can build, inspect, FIX and render a scene end to end. Handles are a counter (never id(): a reused address would silently alias). KEPT NEG: process-local, bounded, evicted oldest-first",
                          example="import lecore; m=lecore.UnifiedMind(); s=m.new_scene(); h=m.scene_add(s, name='ball', geometry=m.shape('sphere')); print(m.scene_info(s)['n_objects'])",
                          native=True, module="objectref",
                          aliases=("add an object to my scene", "put a sphere into the scene document",
                                   "change an object I already added", "delete an object from the scene",
                                   "undo my last scene edit", "insert an object and get its handle",
                                   "keep a python object between two api calls",
                                   "reference a returned object in the next call",
                                   "pass a scene to invoke over http", "server side object registry",
                                   "stateful tool calls", "handle for a non serializable result"))
    c.register_capability("What is in my scene (read the document before you edit it)", "the Scene document could be BUILT and RENDERED and not READ -- an agent that added four objects could not confirm it, recall the names, or spot a mistake before paying for a trace. Read this FIRST; never assume the scene is empty. JSON-safe: objects (handle/name/geometry/material/position/scale/rotated/parent), cameras, lights, selection, materials, problems. `problems` is a PRE-FLIGHT check catching in ms what costs minutes: an unknown material (raises at RENDER time), no geometry, or a ROTATION scene_to_render silently DROPS. KEPT NEG: no bbox, an SDF has no extent",
                          example="import lecore; m=lecore.UnifiedMind(); s=m.new_scene(); print(m.scene_info(s)['empty'])",
                          native=True, module="scene_doc",
                          aliases=("what is in my scene right now", "list the objects in the scene",
                                   "is my scene empty", "how many objects have I added",
                                   "what did I name that object", "inspect the scene before I edit it",
                                   "summarise the scene", "show me the scene contents",
                                   "what are the handles of my objects", "check my scene for mistakes",
                                   "which materials is my scene using", "did my object get added"))
    c.register_capability("View transform (linear render -> a display image)", "a path tracer emits LINEAR radiance with no upper bound; saving that straight to a PNG is a wrong answer, not a missing polish step. MEASURED on a dome + area-light still life: 15.5% of pixels left the tracer above 1.0 and clipped flat. view='display' meters the frame then ACES+gamma (0.0000 clipped, 0.0000 crushed); view='graded' adds bloom/vignette/grain but its FIXED stop crushes 1.97% to black. DEFAULT OFF: a caller measuring radiance or diffing renders needs the linear buffer. KEPT NEG: auto-exposure hides a brightness difference, so hold ev fixed to A/B two light rigs",
                          example="import lecore; m=lecore.UnifiedMind(); print(m.postfx_chain(('auto_exposure\', {}), ('aces', {}), ('gamma', {})))",
                          native=True, module="postfx",
                          aliases=("my render is blown out", "the image is too bright",
                                   "why does my render look washed out", "my highlights are clipping",
                                   "tonemap an hdr render", "aces filmic view transform",
                                   "convert a linear render to a display image", "exposure for a render",
                                   "the render looks flat and grey", "fix the exposure on my image",
                                   "make the render look cinematic", "auto exposure"))
    c.register_capability("Post-effect kernel fusion (N linear passes, one FFT pair)", "compose a RUN of linear, "
                          "shift-invariant post-effects (denoise, sharpen) into ONE transfer and evaluate it with a "
                          "single FFT pair instead of one per stage -- diagonal operators commute and multiply, so the "
                          "composed operator is the elementwise product of theirs. This is holographic_shader's "
                          "Pipeline, in image space. mind.postfx_fuse_transfers(shape, steps) composes; "
                          "mind.postfx_apply_transfer(img, T) evaluates; mind.postfx_fusable_runs(steps) shows which "
                          "runs qualify; PostChain.apply(img, fuse=True) is the wired door. MEASURED (256x256x3, three "
                          "linear stages): 14.76 ms sequential vs 5.03 ms fused -- 2.9x, max|diff| 4.44e-16. THREE KEPT "
                          "NEGATIVES: (1) the SHIPPED chains have no adjacent linear stages -- every blur is separated "
                          "by a nonlinear tone curve -- so fuse=True is correctly a bit-identical NO-OP on "
                          "default_chain and cinematic_chain; it is a capability for chains that HAVE such runs. "
                          "(2) sharpen clips internally, so fusing DEFERS the clamp -- which only matters when the "
                          "clamped stage is FOLLOWED by another in the run: denoise->sharpen is exact (1.33e-15), "
                          "sharpen->denoise differs by 2.81e-01. (3) batching the 3 channels into one FFT is 0.66x "
                          "SLOWER (non-contiguous strides), bit-identical output -- the per-channel loop stays. "
                          "motion_blur and glare clamp their edges, so they are not shift-equivariant and are REFUSED "
                          "rather than approximated. THE ALGEBRA IS NOT GRAPHICS (G1): mind.diffusion_operator(shape, "
                          "alpha, t) builds the heat equation's exact periodic propagator exp(-alpha|k|^2 t) as a "
                          "Pipeline -- bit-identical to diffuse_spectral, ~1.9x faster on reuse because the transfer "
                          "is composed once rather than re-exponentiated per call, and it COMPOSES (two half-steps "
                          "multiply into one full step, exact to 1.1e-15). Nothing in Pipeline knows what a pixel is. "
                          "Same gate: applying it to a Neumann problem is 4.76e-02 WRONG.",
                          example="import numpy as np; from holographic.rendering.holographic_postfx import PostChain; "
                                  "img = np.random.default_rng(0).uniform(0.2, 0.6, size=(64,64,3)); "
                                  "ch = PostChain().then('denoise', sigma=1.0).then('sharpen', amount=0.3, sigma=1.5); "
                                  "print(abs(ch.apply(img) - ch.apply(img, fuse=True)).max())",
                          native=True, aliases=("kernel fusion", "fuse post effects", "compose filter passes",
                                                "one fft instead of many", "post processing chain", "postfx",
                                                "fuse blur and sharpen", "compose transfers", "pipeline fusion",
                                                "apply the same filter a million times", "linear passes"))
    c.register_capability("Information-rate rendering (shade the news, reproject the rest)", "instead of shading "
                          "every pixel every frame, warp the previous frame forward and shade only a budget: the "
                          "disocclusion border (the strip the camera just revealed) plus the OLDEST k pixels, so "
                          "nothing goes stale. mind.refresh_renderer(frame0, budget=0.2).step(shade, known_shift=...) "
                          "runs the loop; mind.refresh_report(...) scores it. MEASURED on a parallax-free procedural "
                          "scene (12 frames, 20% budget): 57.5 dB mean / 55.9 dB worst with a KNOWN camera shift -- "
                          "FIVE TIMES FEWER SHADER EVALUATIONS at visually-indistinguishable quality, tail slope "
                          "+0.22 dB (stable). THREE KEPT NEGATIVES: (1) recovering the shift from pixels with est_dx "
                          "costs 10.5 dB and turns the tail slope to -9.52 (decay) -- the loop warps its own output, "
                          "so a 0.07 px error compounds; the renderer knows how the camera moved, so tell it. "
                          "(2) integer np.roll decays too: 40.7 dB against 57.5 for the same budget -- bilinear warp "
                          "is the mechanism, not a refinement. (3) THE FAKE-PERFECT BUG: a threshold selection "
                          "('refresh every pixel whose age >= the k-th largest') selects ALL 16,384 pixels when ages "
                          "are tied, which they are on frame 0 -- 100% shaded, PSNR 99 dB, a perfect score achieved by "
                          "doing all the work. mind.exact_k_oldest takes exactly k with a stated tie-break. HONEST "
                          "SCOPE: 57.5 dB belongs to a scene with no parallax and no view-dependent shading; on a real "
                          "3-D scene the reprojection ceiling is itself ~38-41 dB, and refresh cannot beat it.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); H = W = 64; "
                                  "world = lambda ox: np.sin((np.arange(W)+ox)*0.11)[None,:] * np.cos(np.arange(H)*0.09)[:,None]; "
                                  "r = mind.refresh_report(lambda i: world(i*1.7), n_frames=6, known_shift=(0.0, -1.7)); "
                                  "print(round(r['shaded_fraction'],3), round(r['psnr_mean'],1))",
                          native=True, aliases=("information rate rendering", "reproject and refresh",
                                                "shade fewer pixels", "temporal upsampling", "TAA render mode",
                                                "age budget", "oldest pixel refresh", "disocclusion border",
                                                "exact k selection", "amortized shading", "render fewer pixels"))
    c.register_capability("The scene's own SDF, emitted (brain/muscle, realised)", "the backlog's brain/muscle claim "
                          "is 'the compute shaders the demos hand-write become a PROJECTION of the authoritative "
                          "Python kernel -- one source of truth, two runtimes, no drift.' It was NOT realised: "
                          "sdf.to_glsl() emitted GLSL for a tree, emit_kernel emitted WGSL from a scalar function's "
                          "SOURCE TEXT, and THE TWO NEVER MET -- so RealtimeSession.payload('shader') carried "
                          "whatever kernel_src the caller passed: a shader written by hand, about a scene the engine "
                          "never saw. That is drift by construction. mind.sdf_dialect(tree, dialect) walks the SAME "
                          "tree that _eval walks and emits map(p) -> distance in wgsl | glsl | c_f64 | c_f32, and "
                          "payload('shader') now emits the SCENE's own map(). THE BAR IS EXECUTED: WGSL cannot run "
                          "here, so mind.sdf_validate_c COMPILES the C twin with cc and RUNS it against the Python "
                          "_eval. MEASURED on a scaled smooth-union of a translated sphere and a rotated box, 200 "
                          "points: c_f64 agrees to 6.7e-16 and is NOT bit-identical -- because np.linalg.norm "
                          "rescales to avoid overflow and sums in a different order than sqrt(x*x+y*y+z*z), so the "
                          "emitted C computes the same FUNCTION by a different summation (K8's scalar kernel WAS "
                          "bit-identical, because it emitted the same expression). c_f32 differs by 3.3e-07, which IS "
                          "the tolerance a WGSL port is judged against -- and the `f` literal suffix is LOAD-BEARING: "
                          "unsuffixed, a C literal is a DOUBLE and the whole expression evaluates in double before "
                          "truncating, so the first table published an optimistic 2.83e-07. An audit found it because "
                          "holographic_emit's dialect table used `f` and this one did not: TWO TABLES FOR ONE CONCEPT "
                          "WILL DISAGREE, AND THE DISAGREEMENT WILL BE A BUG IN ONE OF THEM. A test now pins the "
                          "shared dialects to agree, field by field. And mind.sdf_dialect takes an SDF tree OR ITS "
                          "DSL TEXT, because a live tree does not survive JSON and parse_dsl(to_dsl(t)) round-trips "
                          "to 0.0e+00 -- the kernel is text; so is the scene. THREE KEPT NEGATIVES: (1) `menger` and "
                          "`repeat` fold the domain ITERATIVELY -- unrolling makes the shader's size a parameter -- "
                          "and `twist`/`displace` are inexact distance warps; all four are REFUSED by name, and "
                          "mind.sdf_emit_coverage asserts emitted + refused == every one of the 18 node kinds, "
                          "because a gap there is a shader that silently omits geometry. (2) `scale` is not `p / s`, "
                          "it is `map(p / s) * s`; drop the outer factor and the shape renders correctly with WRONG "
                          "DISTANCES, and a raymarcher oversteps it. (3) WGSL IS NOT C: it infers a local's type with "
                          "`let`, and rejects `vec3<f32> name = ...`. The first emitter wrote the C form for every "
                          "dialect and the structural test -- which checked only the signature and the brace balance "
                          "-- passed the invalid WGSL. An emitted shader is not a rendered image: this validates the "
                          "DISTANCE FUNCTION, not WGSL's precision rules, its fast-math latitude, or whether it "
                          "compiles.",
                          example="from holographic.mesh_and_geometry import holographic_sdf as S; import numpy as np; "
                                  "tree = S.sphere(0.7).translate((0.4, 0, -0.2)).smooth_union(S.box(0.5, 0.3, 0.6), 0.25); "
                                  "print(mind.sdf_dialect(tree.to_dsl(), 'wgsl').splitlines()[0]); "
                                  "print(mind.sdf_validate_c(tree, np.random.default_rng(0).uniform(-2, 2, (50, 3)), 'c_f64'))",
                          native=True, aliases=("emit the scene's sdf", "sdf to wgsl", "sdf shader",
                                                "brain muscle contract", "one source of truth two runtimes",
                                                "compute shader from the scene", "sdf dialect", "map function",
                                                "no drift", "webgpu sdf"))
    c.register_capability("Realtime session (draft frames, refine pass, multi-format payload)", "a viewport wants a "
                          "frame NOW; a render wants it RIGHT. mind.realtime_session(render_session) gives both: "
                          "`frame(camera, known_shift=)` is a DRAFT that reprojects the previous frame and re-shades "
                          "only the news (an exact-k oldest-age budget plus the disocclusion border, which must be "
                          "shaded because the previous frame never saw it); `refine()` traces every pixel; "
                          "`payload(kinds)` pushes the same scene as PIXELS, MESH, SPLATS, SHADER (WGSL) and LOD "
                          "(progressive TT descriptor) -- every value plain data, strict-JSON safe. THE MISSING HALF, "
                          "NOW SHIPPED: RefreshRenderer computed a budget and called shade(mask), and its own "
                          "docstring admitted 'a real renderer WOULD shade only those pixels' -- nothing did, because "
                          "render_surface traced every pixel. The famous '5x fewer shader evaluations' was an "
                          "arithmetic statement about a mask, not a saving anyone had realised. render_surface now "
                          "takes pixel_mask= and base=: MEASURED 3.2x faster at a 20% mask and 6.2x at 5%, "
                          "BIT-IDENTICAL on the pixels it shades, base preserved elsewhere, and bit-identical to "
                          "before when no mask is given. KEPT NEGATIVE: PASS `known_shift` -- recovering the camera's "
                          "motion from the pixels costs 2,280 extra traces, 3.7 dB, and a -4.52 dB TAIL SLOPE (the "
                          "loop warps its own output and the error compounds); with a known shift the tail is "
                          "+0.16 dB. THE CONTRACT'S HONEST ASYMMETRY: a draft frame CONVERGES to the refined frame, "
                          "but a draft SIMULATION does not converge to its refinement -- mind.draft_vs_refine_simulation "
                          "measures it, and `fluid` at grid 32 against 48 has relative error 1.000 while grid 24 has "
                          "0.669, NON-MONOTONIC. The coarse run is a different trajectory of a chaotic system, not a "
                          "blurred one. Refining a render sharpens it; refining a chaotic solve replaces it. CACHES: "
                          "the previous frame and a per-pixel AGE buffer; `scene_version` keys the mesh/splat/lod "
                          "payloads so a camera move rebuilds no geometry; the RenderSession's fat-margin preview "
                          "cache is deliberately left alone, because serving a stale frame into a warp compounds.",
                          example="import numpy as np; import lecore; mind=lecore.UnifiedMind(dim=256, seed=0); from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial; "
                                  "from holographic.rendering.holographic_render import Camera; "
                                  "from holographic.scene_and_pipeline.holographic_session import RenderSession; "
                                  "class S:\n    def eval(self, P): return np.linalg.norm(P, axis=1) - 0.9\n    def ids(self, P): return np.zeros(len(P), int)\n"
                                  "sess = RenderSession(S(), {0: SurfaceMaterial.from_name('plastic')}, Camera(eye=(0,0,3.2), target=(0,0,0), fov_deg=50), width=32, height=32); "
                                  "rt = mind.realtime_session(sess, budget=0.2); "
                                  "print(rt.frame(known_shift=(0.0, -0.3))); print(rt.stats())",
                          native=True, aliases=("realtime", "realtime preview then refine", "viewport",
                                                "draft frame", "refine pass", "push updates to a front end",
                                                "pixel stream", "multi-format payload", "shade only the news",
                                                "frame budget", "progressive refinement", "stream a frame"))
    c.register_capability("Cross field (smoothest 4-RoSy) + the bar that was vacuous", "field-aligned retopology "
                          "begins with a cross field: a direction at every face, defined up to 90-degree rotation, as "
                          "smooth as the surface allows. mind.cross_field(mesh) solves for it as the eigenvector of "
                          "the smallest eigenvalue of the complex CONNECTION LAPLACIAN (Knoppel, Crane, Pinkall & "
                          "Schroder, SIGGRAPH 2013) -- a solve, not an iteration. mind.singularity_index gives a "
                          "per-vertex index that is EXACTLY a multiple of 1/4 (residual 0.0e+00); mind.field_report "
                          "carries every number. THE HEADLINE IS A RETRACTION: the previous session recorded "
                          "'sum of the singularity indices equals the Euler characteristic' as this item's bar -- an "
                          "integer, no tolerance to argue about. It is true, it is exact here, AND IT IS VACUOUS. "
                          "Measured on the same sphere: the smoothest field sums to +2.0 with 49 singularities; a "
                          "uniformly RANDOM field sums to +2.0 with 127; an all-zero field sums to +2.0 with 203; an "
                          "adversarial alternating field sums to +2.0. The matching integers are antisymmetric, so "
                          "their contribution cancels pairwise around every dual edge and what remains is a function "
                          "of the MESH alone. A BAR THAT PASSES FOR EVERY INPUT IS NOT A BAR. Judge a field by its "
                          "singularity COUNT and its Dirichlet ENERGY (54.7 smoothest against 1542.2 random). "
                          "Poincare-Hopf validates the transport and the dual rings, which is worth having and is "
                          "not what it was advertised as. TWO MORE KEPT NEGATIVES: antisymmetry must be ENFORCED, "
                          "not hoped for -- computing the transport from both directed edges lets atan2's branch cut "
                          "differ by 2pi, which shifts the matching by 4 and the index by 1 per edge (a sphere's "
                          "indices summed to -43 instead of +2), and `wrap` at exactly +-pi is a tie that broke "
                          "antisymmetry on a tetrahedron; and Jacobi smoothing does NOT converge -- a torus's energy "
                          "fell to 2788 by 50 sweeps and ROSE to 2866 by 400. HONEST SCOPE: eigh on a dense "
                          "(faces, faces) matrix is O(F^3), fine to a few thousand faces; the mesh must be closed and "
                          "consistently oriented (mind.mesh_is_oriented); quad EXTRACTION is a mixed-integer problem "
                          "and is not here. AGENT-FACING: use mind.field_singularities(mesh) -- a STATELESS one-shot "
                          "that takes buffers and returns plain data. mind.cross_field returns a `ctx` whose `rho` is "
                          "keyed by (face, face) TUPLES; serialised, those become the strings '(0, 1)', so the payload "
                          "LOOKS like a context and cannot be fed back (singularity_index dies with KeyError). An "
                          "object that serialises into something that looks right but cannot be used is worse than "
                          "one that raises -- so singularity_index now detects a JSON-round-tripped ctx and names the "
                          "twin. Every mesh faculty also accepts {vertices, faces} or (vertices, faces), because a "
                          "live Mesh handle does not survive JSON either.",
                          example="from holographic.mesh_and_geometry.holographic_mesh import tetrahedron; "
                                  "print(mind.field_singularities(tetrahedron()))",
                          native=True, aliases=("field singularities", "cross field", "cross field on a surface", "4-rosy",
                                                "smoothest direction field", "field aligned remesh",
                                                "singularities of a direction field", "instant meshes",
                                                "quad mesh from a field", "retopology", "connection laplacian",
                                                "poincare hopf", "direction field", "retopologize a mesh",
                                                "remesh to quads", "remesh a mesh", "clean up mesh topology"))
    c.register_capability("Quad remesh (field-guided tris-to-quads)", "FIELD-GUIDED tri-to-quad RETOPOLOGY: m.quad_remesh(mesh) pairs adjacent triangles into quads, preferring pairs whose edges align with the 4-RoSy cross field and form convex near-square quads. Returns a QUAD-DOMINANT mesh + report {quads, tris, quad_fraction, field_used}. Reuses cross_field, so input wants a CLOSED oriented manifold TRIANGLE mesh -- run mesh_repair(triangulate=True) first; falls back to squareness if the field cannot solve. HONEST: places quads on EXISTING vertices, does NOT move vertices or regularise valence, so NOT a full Instant-Meshes remesh (deferred).",
                          example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; qm,rep=m.quad_remesh(triangulate_ngons(box())); (rep['quads'], rep['quad_fraction'])",
                          native=True, aliases=("quad remesh", "tris to quads", "quadrangulate a mesh", "merge triangles into quads",
                                                "quad dominant mesh", "field aligned quad mesh", "retopologize to quads",
                                                "convert triangles to quads", "make a quad mesh"))
    c.register_capability("Guided cross field (deformation/curvature-aware field design)", "A GUIDED 4-RoSy field (field DESIGN): m.guided_cross_field(mesh, guide_dirs, guide_weight) solves the smoothest field that ALSO aligns to a prescribed per-face direction. guide_dirs is (n_faces,3): a non-zero row guides that face (length=confidence), zero row free. Soft-constrained solve (L + w)u = w c -- a linear SOLVE, not an eigenproblem; no guides == cross_field. Returns (phi, ctx) for quad_remesh(field=...). Makes retopo DEFORMATION-AWARE (feed strain_directions) or curvature-aware, following deliberate topology instead of only minimising distortion. Needs a CLOSED oriented manifold mesh.",
                          example="import numpy as np; import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; from holographic.mesh_and_geometry.holographic_crossfield import face_frames; tb=triangulate_ngons(box()); n,ex,ey=face_frames(np.asarray(tb.vertices,float),np.asarray(tb.faces,int)); phi,ctx=m.guided_cross_field(tb, np.cos(np.pi/8)*ex+np.sin(np.pi/8)*ey, guide_weight=12.0); round(float(np.mean(np.abs(np.cos(4*(phi-np.pi/8))))),2)",
                          native=True, aliases=("guided cross field", "field design", "constrain a cross field", "align a field to a direction",
                                                "deformation aware field", "curvature aware field", "steer a cross field"))
    c.register_capability("Deformation strain directions (retopo guide)", "Per-face PRINCIPAL STRETCH direction of a deformation (rest -> deformed vertices): m.strain_directions(mesh, deformed_vertices) -- the DEFORMATION guide that makes retopo place edge loops FOLLOWING how a surface bends/stretches, which an off-the-shelf remesher cannot (no strain signal). Per triangle: deformation gradient -> right Cauchy-Green C -> max-stretch eigenvector to 3-D, SCALED by anisotropy (isotropic face -> ~0 confidence, free). Returns (n_faces,3) as guide_dirs for guided_cross_field; guiding the field to the stretch puts quad LOOPS perpendicular to it -- encircling the bend.",
                          example="import numpy as np; import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; tb=triangulate_ngons(box()); V=np.asarray(tb.vertices,float); Vs=V.copy(); Vs[:,0]=V[:,0]+0.8*V[:,1]; m.strain_directions(tb, Vs).shape",
                          native=True, aliases=("deformation aware retopology", "strain directions", "principal stretch direction",
                                                "edge loops that follow deformation", "animation aware retopo", "deformation guide for retopo",
                                                "loops around a joint"))
    c.register_capability("Position field (IFAM 4-PoSy lattice remesh)", "IFAM POSITION FIELD (4-PoSy, Jakob et al. 2015): m.position_field(mesh, orient, edge_length) optimises a per-vertex LATTICE position aligned to the orientation field by local extrinsic smoothing -- per edge it forms q_ij, translates the neighbour by INTEGER rho-steps to line up, then averages, so neighbours differ by integer lattice steps. Regularises vertex spacing/valence (a field-aligned grid). Vertex-graph only. Returns P; position_field_regularity scores convergence (0=perfect grid). HONEST: the position FIELD only; extraction to the quad MESH (IFAM 4.4) is next, not built.",
                          example="import numpy as np; import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import grid; g=grid(8,8,width=7.0,height=7.0); V=np.asarray(g.vertices,float); o=np.tile([1.,0,0],(len(V),1)); rng=np.random.default_rng(0); g.vertices=V+np.column_stack([rng.normal(0,.24,len(V)),rng.normal(0,.24,len(V)),np.zeros(len(V))]); P=m.position_field(g,o,7.0/8,iterations=20); round(m.position_field_regularity(g,P,o,7.0/8),3)",
                          native=True, aliases=("position field", "posy field", "instant meshes position field", "field aligned lattice",
                                                "regularise vertex spacing", "position field remesh", "ifam position field", "snap vertices to a field grid"))
    c.register_capability("Trace streamlines (field -> curves)", "Trace STREAMLINES of a per-face direction field on a triangle mesh: m.trace_streamlines(mesh, field) walks the field edge to edge until a boundary / max_steps / a loop, returning polylines. The general FIELD -> CURVES primitive, source-agnostic -- the SAME tracer serves a cross_field (retopo guides, hatching), strain_directions (deformation flow lines), an SDF gradient, or a SIMULATION velocity field (streamlines / pathlines). field is per-face angles or 3-D vectors; four_rosy=True treats it as a 4-RoSy cross (nearest-travel branch, never reverses), False for a true vector field.",
                          example="import numpy as np; import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import grid; g=grid(10,10,width=5.0,height=5.0); uni=np.tile([1.,0,0],(len(g.faces),1)); lines=m.trace_streamlines(g, uni, four_rosy=False, seeds=[0,40,80]); (len(lines), max(len(L) for L in lines)>5)",
                          native=True, aliases=("trace streamlines", "integral curves of a field", "field lines", "flow lines",
                                                "streamlines of a velocity field", "pathlines", "hatching curves from a field",
                                                "trace a direction field", "flow visualization", "guide curves from a cross field"))
    c.register_capability("UV / attribute transfer (texture-preserving retopo)", "TRANSFER per-vertex UVs -- or ANY per-vertex attribute (colours, weights, normals) -- onto NEW vertices by closest-point + barycentric interpolation: m.transfer_uv(source_mesh, source_uv, target_vertices) -> (attr, residual). THE step that makes retopo TEXTURE-PRESERVING: the remeshed surface lies on the original, so each new vertex takes the interpolated UV of its closest source triangle. Spatial-hash accelerated; the residual is the honest error signal. MEASURED: exact on-surface; mantis 1490 verts in 1.6s, residual mean 4e-5. KEPT NEG: wrong across UV SEAMS; seam-split not built.",
                          example="import numpy as np; import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import grid; g=grid(6,6,width=6.0,height=6.0); V=np.asarray(g.vertices,float); uv=(V[:,:2]-V[:,:2].min(0))/6.0; got,res=m.transfer_uv(g, uv, np.array([[0.5,0.5,0.0]])); (np.round(got,3).tolist(), float(res[0]))",
                          native=True, aliases=("transfer uvs to a new mesh", "reproject uv coordinates", "texture preserving retopology",
                                                "keep the texture after remeshing", "attribute transfer between meshes",
                                                "closest point barycentric transfer", "bake uvs onto a retopo mesh"))


_PART = "holographic_catalog_p03"


def _selftest():
    """Delegates to holographic_catalog.check_catalog_part -- one home for the shared contract."""
    from holographic.caching_and_storage.holographic_catalog import check_catalog_part
    n = check_catalog_part(_PART, register_p03)
    print("%s selftest OK -- %d capabilities, no internal duplicates" % (_PART, n))


if __name__ == "__main__":
    _selftest()
