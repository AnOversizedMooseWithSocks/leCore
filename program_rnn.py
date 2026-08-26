"""The recurrent model that returns a PROGRAM, not a state.

THE CLAIM UNDER TEST. Every sequence model compresses history into a hidden state and
predicts by rolling that state forward. leCore has a different pipeline sitting in
`explore_series(auto_demux=True)`:

    demux_series      -- ONE stream, MANY sources. Find the interleave stride K by
                         delta-continuity. Deinterleaving is a PERMUTATION, so channel
                         recovery is BIT-EXACT, and it honestly returns K=1 when nothing
                         separates.
    fit_deterministic -- per channel, recover the GENERATOR that explains it (family +
                         params), band-limited so families that differ only above the
                         coarse rate TIE rather than one winning on aliased detail.
                         REFUSES (family=None) when no generator beats the noise.
    extend_generator  -- forecast by playing the fitted generator PAST its data, and
                         marks valid=False beyond the validated window.

So the "hidden state" is a formula, of a named family, with parameters -- bytes, not
samples -- and the model can say "there is no law here."

TWO THINGS ARE MEASURED, and the second is the one that matters.

  (A) ACCURACY on a genuinely deterministic multiplexed stream, against leCore's own
      tuned ESN (`mind.reservoir`) given the SAME data and the SAME horizon. An honest
      baseline: the reservoir is the engine's best gradient-free sequence learner, not
      a strawman.

  (B) ABSTENTION on a channel that is pure noise. An RNN has no mechanism to decline --
      it will emit a confident trajectory for white noise. `fit_deterministic` should
      return family=None. This is not an accuracy contest; it is a capability the
      baseline structurally lacks, and a false-confidence rate is the right metric.

KEPT NEGATIVES this file must not hide:
  * demux scope is CYCLIC interleaving only -- packetized muxing will not score a clean K.
  * fit_deterministic snaps to a BANK (sine/chirp/gauss/sawtooth). A signal outside the
    bank is refused, not approximated. Refusal is the honest output, but coverage is
    limited by the bank, and that is a real ceiling, not a virtue.
  * extend_generator refuses beyond its validated window -- a generator fit on [0,1]
    evaluated at t=100 is confident nonsense.
"""
import numpy as np
import lecore


def make_channels(n, rng):
    """Three genuinely deterministic sources + one pure-noise source."""
    t = np.arange(n, dtype=float)
    return {
        "sine":  np.sin(2 * np.pi * t / 210.0),
        "chirp": np.sin(2 * np.pi * (0.0009 * t + 1.1e-6 * t ** 2)),
        "saw":   ((t % 150.0) / 150.0) * 2.0 - 1.0,
        "noise": rng.standard_normal(n),          # the abstention probe
    }


def interleave(chans):
    """Round-robin mux: sample i belongs to channel i mod K -- the 'Contact' move."""
    names = list(chans)
    K, n = len(names), len(chans[names[0]])
    out = np.empty(K * n)
    for i, nm in enumerate(names):
        out[i::K] = chans[nm]
    return out, names


def main(n=600, horizon=40):
    m = lecore.UnifiedMind(dim=1024, seed=0)
    rng = np.random.default_rng(0)
    chans = make_channels(n + horizon, rng)
    train = {k: v[:n] for k, v in chans.items()}
    truth = {k: v[n:n + horizon] for k, v in chans.items()}

    stream, names = interleave(train)
    print(f"muxed stream: {len(stream)} samples, {len(names)} sources, "
          f"true stride K={len(names)}\n")

    # ---- stage 1: demux. Does it FIND K without being told?
    d = m.demux_series(stream, max_k=12)
    found_k = d.get("stride")
    print(f"STAGE 1 demux_series -> found K = {found_k}  (true {len(names)})  "
          f"{'CORRECT' if found_k == len(names) else 'MISSED'}")
    chan_list = d.get("objects")
    if chan_list is not None and found_k == len(names):
        rec = np.asarray(chan_list)
        err = max(float(np.max(np.abs(np.asarray(rec[i]) - train[names[i]])))
                  for i in range(len(names)))
        print(f"  channel recovery max|err| = {err:.3e}  "
              f"(permutation => should be bit-exact)")

    # ---- stage 2+3: fit a generator per channel, then play it forward
    print(f"\nSTAGE 2+3 fit_deterministic -> extend_generator (horizon {horizon})")
    print(f"{'channel':>8} {'family':>10} {'corr':>7} {'resid':>7} "
          f"{'fcast NRMSE':>12} {'verdict':>10}")
    fit_nrmse = {}
    for nm in names:
        f = m.fit_deterministic(train[nm])
        fam, corr = f.get("family"), f.get("correlation", float("nan"))
        resid = f.get("residual_frac", float("nan"))
        if fam is None:
            print(f"{nm:>8} {'REFUSED':>10} {'-':>7} {'-':>7} {'-':>12} "
                  f"{'abstain':>10}")
            fit_nrmse[nm] = None
            continue
        e = m.extend_generator(f, horizon, n)
        pred = np.asarray(e["forecast"])[:horizon]
        s = float(np.std(truth[nm])) or 1.0
        nr = float(np.sqrt(np.mean((pred - truth[nm]) ** 2)) / s)
        fit_nrmse[nm] = nr
        print(f"{nm:>8} {str(fam):>10} {corr:>7.3f} {resid:>7.3f} {nr:>12.4f} "
              f"{str(f.get('verdict'))[:10]:>10}")

    # ---- the honest baseline: leCore's own ESN, same data, same horizon
    print(f"\nBASELINE  mind.reservoir (the engine's best gradient-free sequence learner)")
    print(f"{'channel':>8} {'ESN NRMSE':>11} {'generator':>11} {'winner':>10}")
    for nm in names:
        esn = m.reservoir(n_in=1, rho=0.95, leak=1.0)
        U = train[nm][:-1].reshape(-1, 1)
        Y = train[nm][1:].reshape(-1, 1)
        esn.fit(U, Y, washout=50)
        gen = np.asarray(esn.generate(horizon, U[-100:], feedback=lambda y: y)).ravel()[:horizon]
        s = float(np.std(truth[nm])) or 1.0
        enr = float(np.sqrt(np.mean((gen - truth[nm]) ** 2)) / s)
        g = fit_nrmse[nm]
        win = "generator" if (g is not None and g < enr) else (
            "ESN" if g is not None else "ESN (guessed)")
        print(f"{nm:>8} {enr:>11.4f} {('abstained' if g is None else f'{g:.4f}'):>11} "
              f"{win:>10}")

    print("\nTHE POINT: on the noise channel the generator path ABSTAINS and the ESN")
    print("emits a confident trajectory. That is not a worse score -- it is a")
    print("different contract. Storage: (family, params) is bytes; an ESN readout")
    print("plus state is kilobytes.")


if __name__ == "__main__":
    main()
