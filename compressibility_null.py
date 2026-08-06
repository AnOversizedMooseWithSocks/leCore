"""OPEN PROBLEM (1): a compressibility estimator that fails SAFE.

WHY THIS IS THE BINDING GAP. The Structured Recall Bound says H(v) <= [q(d)+1]/(1-eps),
so an architecture can take all three corners of the Impossibility Triangle exactly when
the sequence has vanishing entropy rate. But H(v) is not observable. Everything therefore
rests on an ESTIMATOR of compressibility, and its failure modes are asymmetric:

    overestimate H  ->  waste state, fall back to the triangle. Cheap.
    underestimate H  ->  CONFIDENT FALSE RECALL. Catastrophic.

`fit_deterministic` refuses on white noise. That is the EASY null and proves little. The
number that matters -- and that nobody has published for any architecture on the
Impossibility Triangle's list of 52 -- is the false-fit rate against a HARD null: data
with real spectral structure but no deterministic generator.

THE FOUR NULLS, ordered by how hard they are to refuse:
  white       -- i.i.d. Gaussian. Trivial; a detector failing here is broken.
  ar1         -- autocorrelated noise. Has a trend a naive fitter can chase.
  walk        -- Brownian motion. Smooth, non-stationary, LOOKS deterministic.
  phase_rand  -- THE decisive one: a phase-randomised surrogate of a real sine. It has the
                 IDENTICAL power spectrum to genuine structure and no deterministic
                 generator whatsoever (Theiler et al.'s surrogate-data method, the standard
                 null in nonlinear time-series analysis). A fitter that cannot refuse this
                 is keying on spectrum, not on determinism.

AND THE POWER SIDE, because a detector that refuses everything is useless: the same
procedure must still FIT genuine generators. A false-fit rate of 0 with a true-fit rate of
0 is not calibration, it is silence.

The calibration itself uses `mind.permutation_null` -- the engine's own composable
procedure-matched null ("score it, then prove it isn't an artifact of your own pipeline").
Its documented kept negative applies directly and is the thing to be careful about: a
resample_fn that does NOT destroy the structure the score keys on gives a meaningless p.
Phase randomisation destroys phase coupling while PRESERVING the spectrum, which is exactly
the structure a deterministic-generator fit should be keying on and a spectral artifact
should not.
"""
import numpy as np
import lecore


def phase_randomize(x, rng):
    """Theiler surrogate: keep |FFT| exactly, randomise the phases. Same spectrum, no
    deterministic generator. The standard hard null for 'is this signal deterministic?'."""
    X = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0
    if len(x) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))


def make_null(kind, T, rng):
    t = np.arange(T, dtype=float)
    if kind == "white":
        return rng.standard_normal(T)
    if kind == "ar1":
        e = rng.standard_normal(T)
        y = np.zeros(T)
        for i in range(1, T):
            y[i] = 0.95 * y[i - 1] + e[i]
        return y
    if kind == "walk":
        return np.cumsum(rng.standard_normal(T))
    if kind == "phase_rand":
        return phase_randomize(np.sin(2 * np.pi * t / 210.0), rng)
    raise ValueError(kind)


def main(T=800, trials=40):
    m = lecore.UnifiedMind(dim=512, seed=0)

    print("PART A -- raw false-fit rate of fit_deterministic, by null hardness")
    print(f"(T={T}, {trials} trials each; 'fit' = a family returned, i.e. NOT refused)")
    print(f"{'null':>12} {'fit rate':>10} {'tie rate':>10} {'refused':>10} {'mean corr':>11}")
    for kind in ("white", "ar1", "walk", "phase_rand"):
        fits = ties = ref = 0
        corrs = []
        for s in range(trials):
            rng = np.random.default_rng(1000 + s)
            f = m.fit_deterministic(make_null(kind, T, rng))
            if f.get("family") is None:
                ref += 1
            else:
                fits += 1
                corrs.append(f.get("correlation", 0.0))
                if str(f.get("verdict")) == "tie":
                    ties += 1
        print(f"{kind:>12} {fits/trials:>10.3f} {ties/trials:>10.3f} {ref/trials:>10.3f} "
              f"{(np.mean(corrs) if corrs else float('nan')):>11.3f}")

    print("\nPART B -- POWER: does it still fit genuine generators? (a silent detector is useless)")
    t = np.arange(T, dtype=float)
    real = {
        "sine":     np.sin(2 * np.pi * t / 210.0),
        "chirp":    np.sin(2 * np.pi * (0.0009 * t + 1.1e-6 * t ** 2)),
        "sawtooth": ((t % 150.0) / 150.0) * 2 - 1,
    }
    print(f"{'signal':>12} {'family':>12} {'corr':>8} {'verdict':>10}")
    for nm, sig in real.items():
        f = m.fit_deterministic(sig)
        print(f"{nm:>12} {str(f.get('family')):>12} "
              f"{f.get('correlation', float('nan')):>8.3f} {str(f.get('verdict')):>10}")

    print("\nPART C -- CALIBRATION via mind.permutation_null (phase-randomised surrogates)")
    print("Score = the fit's correlation. Resample = phase randomisation of the SAME signal,")
    print("which preserves the spectrum and destroys determinism -- so the null is matched to")
    print("the claim ('this is a deterministic generator'), not to the spectrum.")

    def score_of(x):
        f = m.fit_deterministic(np.asarray(x))
        return float(f.get("correlation") or 0.0)

    print(f"\n{'signal':>12} {'observed':>10} {'null mean':>10} {'p':>8} {'collapsed':>10}")
    cases = [("sine (real)", real["sine"]), ("sawtooth (real)", real["sawtooth"])]
    for nm in ("walk", "phase_rand"):
        cases.append((f"{nm} (null)", make_null(nm, T, np.random.default_rng(7))))
    for nm, sig in cases:
        base = np.asarray(sig)
        try:
            r = m.permutation_null(base, score_of,
                                   lambda rng, _b=base: phase_randomize(_b, rng),
                                   n_null=60, seed=0, alpha=0.05, side="greater")
            print(f"{nm:>12} {r['observed']:>10.3f} {r['null_mean']:>10.3f} "
                  f"{r['p']:>8.3f} {str(r['collapsed']):>10}")
        except Exception as e:
            print(f"{nm:>12} -> {type(e).__name__}: {str(e)[:60]}")
    print("\n'collapsed=True' means the real score stood out from its own null at alpha=0.05.")
    print("A null row that collapses is a FALSE FIT surviving calibration -- report it loudly.")


if __name__ == "__main__":
    main()
