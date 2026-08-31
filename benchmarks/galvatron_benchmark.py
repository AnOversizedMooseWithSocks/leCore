"""GALVATRON BENCHMARK on a REAL trained model.

Every claim in this arc was measured on a random-weight subject, which is
degenerate: it emits one token forever and is uncertain about everything. That
made the MECHANISMS provable and the SEMANTICS unmeasurable. This suite reruns
the load-bearing claims against a model actually trained on leCore's own data
(WordNet definitions + leCore docs + leCore source), byte-level, so text can be
read directly and the three registers can be told apart.
"""
import sys, time, json
sys.path.insert(0, '/home/claude/work')
import numpy as np

from holographic.io_and_interop.holographic_gdnruntime import load_runtime
from holographic.agents_and_reasoning.holographic_leap import leap_generate, RouteMemory
from holographic.agents_and_reasoning.holographic_knowres import SalienceTrigger
from holographic.agents_and_reasoning.holographic_galvatron import DreamerResident, WardResident, Galvatron
from holographic.agents_and_reasoning.holographic_voidmanifold import manifold_voids
from holographic.agents_and_reasoning.holographic_carrier import StreamCarrier
import lecore

B = lambda s: [int(c) for c in s.encode('utf-8')]
S = lambda ids: bytes(bytearray(int(t) % 256 for t in ids)).decode('utf-8', 'replace')

def main():
    rt, cfg = load_runtime('/home/claude/bench/model')
    mind = lecore.UnifiedMind(dim=512, seed=0)
    print('=' * 78)
    print('SUBJECT: byte-level GDN-hybrid, hidden %d, %d layers, trained on '
          'WordNet + leCore docs + leCore source' % (cfg['hidden'], cfg['n_layers']))
    print('=' * 78)

    # --- 1. RUNTIME FIDELITY on a trained model + register perplexity ---
    print('\n[1] IN-ENGINE PERPLEXITY per register (leCore owns the forward pass)')
    texts = {
        'dictionary': "abandon: to give up completely; forsake. abbey: a church",
        'lecore docs': "The holographic engine binds and bundles hypervectors ",
        'lecore code': "def _selftest():\n    rng = np.random.default_rng(0)\n",
        'unseen (random bytes)': S(list(np.random.default_rng(0).integers(97, 122, 60))),
    }
    for name, t in texts.items():
        ids = B(t)[:64]
        print('    %-24s ppl %8.2f   (%d bytes)' % (name, rt.perplexity(ids), len(ids)))

    # --- 2. GENERATION: does it produce real text? ---
    print('\n[2] GENERATION (greedy, in-engine)')
    for prompt in ("the meaning of ", "def compress(", "holographic "):
        ids = B(prompt)
        out, _ = rt.generate_fast(ids, n_new=60)
        print('    %-16s -> %r' % (repr(prompt), S(out[len(ids):])))

    # --- 3. SALIENCE: does a TRAINED model's hesitation actually vary? ---
    print('\n[3] SALIENCE (the open question: entropy spread on a trained model)')
    probe_layer = max(0, cfg['n_layers'] - 2)
    cap = {}
    long_ids = B(texts['dictionary'] + texts['lecore docs'])[:200]
    rt.forward(long_ids, hooks={probe_layer: lambda h: cap.__setitem__('h', h.copy()) or None})
    sal = SalienceTrigger(rt); sal.calibrate(cap['h'], quantile=0.8)
    sc = np.array([sal.score(x) for x in cap['h']])
    final = rt.forward(long_ids)
    fl = final - final.max(-1, keepdims=True); pf = np.exp(fl); pf /= pf.sum(-1, keepdims=True)
    true_ent = -np.sum(pf * np.log(pf + 1e-30), axis=-1)
    print('    lens entropy: mean %.3f  spread %.3f  min %.3f  max %.3f  (max possible %.3f)'
          % (sc.mean(), sc.std(), sc.min(), sc.max(), np.log(256)))
    print('    RANDOM-MODEL BASELINE was spread 0.004 (uncertain about everything)')
    print('    correlation with true final entropy: %.3f' % np.corrcoef(sc, true_ent)[0, 1])
    hi = np.argsort(sc)[-6:]; lo = np.argsort(sc)[:6]
    print('    most uncertain bytes: %r' % S([long_ids[i] for i in sorted(hi)]))
    print('    most confident bytes: %r' % S([long_ids[i] for i in sorted(lo)]))

    # --- 4. LEAP: speculative decoding on real text ---
    print('\n[4] LEAP (speculative decoding, output must be token-identical)')
    for prompt, label in ((texts['dictionary'][:40], 'dictionary'), (texts['lecore code'][:40], 'code')):
        ids = B(prompt)
        t0 = time.time(); base, _ = rt.generate_fast(ids, n_new=48); t_plain = time.time() - t0
        t0 = time.time(); g1, mem, r1 = leap_generate(rt, ids, n_new=48, k=8); t_cold = time.time() - t0
        t0 = time.time(); g2, _m, r2 = leap_generate(rt, ids, n_new=48, memory=mem, k=8); t_warm = time.time() - t0
        print('    %-11s plain %.2fs | cold %.2fs (acc %.2f) | warm %.2fs (acc %.2f) -> %.2fx | identical %s'
              % (label, t_plain, t_cold, r1['acceptance_rate'], t_warm, r2['acceptance_rate'],
                 t_plain / max(t_warm, 1e-9), g1 == base and g2 == base))

    # --- 5. DREAMER headroom on a trained stream ---
    print('\n[5] DREAMER (headroom = how concentrated the trained stream is)')
    H = cap['h']
    dr = DreamerResident(mind, H, probe_layer, strength=1.0)
    d = H.shape[1]
    print('    healthy subspace rank %d of %d -> removable noise energy (d-r)/d = %.2f'
          % (dr.rank, d, (d - dr.rank) / d))
    print('    RANDOM-MODEL BASELINE was r=25/64 (headroom 0.39)')
    clean_top = np.argmax(rt.forward(long_ids), -1)
    for noise in (0.5, 1.0):
        r1 = np.random.default_rng(5)
        a_bad = float(np.mean(np.argmax(rt.forward(long_ids, hooks={probe_layer: lambda h: noise * r1.standard_normal(h.shape)}), -1) == clean_top))
        r1 = np.random.default_rng(5)
        def ctr(h, _n=noise):
            dd = _n * r1.standard_normal(h.shape); rep = dr.hook(h + dd)
            return dd + (rep if rep is not None else 0.0)
        a_rep = float(np.mean(np.argmax(rt.forward(long_ids, hooks={probe_layer: ctr}), -1) == clean_top))
        print('    noise %.1f: corrupted agreement %.3f -> repaired %.3f  (recovered %.0f%%)'
              % (noise, a_bad, a_rep, 100 * (a_rep - a_bad) / max(1 - a_bad, 1e-9)))

    # --- 6. VOIDS on a trained manifold, with the surrogate control ---
    print('\n[6] VOID MANIFOLD (structure vs matched-covariance surrogate)')
    Hc = H - H.mean(0)
    U, Sv, Vt = np.linalg.svd(Hc, full_matrices=False)
    for k in (3, 6):
        X = Hc @ Vt[:k].T
        r = manifold_voids(X, n_probes=400, surrogate_trials=3)
        print('    top-%d PCs (%.0f%% energy): void frac %.3f vs surrogate %.3f +- %.3f -> %s'
              % (k, 100 * (Sv[:k] ** 2).sum() / (Sv ** 2).sum(), r['void_fraction'],
                 r['surrogate_fraction'], r['surrogate_sd'], r['verdict'][:40]))

    # --- 7. CARRIER capacity on a trained stream ---
    print('\n[7] CARRIER (exact structured data riding a trained residual stream)')
    for reserve in (16, 32):
        car = StreamCarrier(H, reserve=reserve, amplitude=0.5)
        pairs = {'subject': 'moose', 'project': 'lecore', 'state': 'shipping'}
        got = {}
        base_lg = rt.forward(long_ids)
        out = rt.forward(long_ids, hooks={1: car.writer(pairs),
                                          cfg['n_layers'] - 1: lambda h: got.__setitem__('h', h.copy()) or None})
        cands = ['moose', 'lecore', 'shipping', 'otter', 'pytorch', 'idle']
        ok = sum(car.read(got['h'], r, cands)[0] == v for r, v in pairs.items())
        interf = float(np.max(np.abs(out - base_lg)) / np.max(np.abs(base_lg)))
        rep = car.report(len(pairs))
        print('    reserve %2d dims (%.1f%% of stream energy): %d/%d pairs recovered, '
              'logit interference %.3f' % (reserve, 100 * rep['borrowed_energy_fraction'],
                                           ok, len(pairs), interf))
        print('        RANDOM-MODEL BASELINE: 32 dims borrowed 15.6%% energy for 0.219 interference')

    # --- 8. WARD on real text ---
    print('\n[8] WARD (hard bans on a trained model)')
    ids = B("the meaning of ")
    bare, _ = rt.generate_fast(ids, n_new=40)
    vowels = [int(c) for c in b'aeiou']
    warded, _ = Galvatron(rt, guards=[WardResident(banned=vowels)]).generate(ids, n_new=40)
    print('    unguarded: %r' % S(bare[len(ids):]))
    print('    no vowels: %r' % S(warded[len(ids):]))
    print('    vowels emitted under ban: %d' % len(set(warded[len(ids):]) & set(vowels)))

if __name__ == '__main__':
    main()
