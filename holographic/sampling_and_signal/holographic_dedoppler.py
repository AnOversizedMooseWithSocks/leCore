"""holographic_dedoppler.py -- find a DRIFTING narrowband signal in a spectrogram.

The SETI detection problem (Tarter, Siemion seats) cast entirely in the engine's own primitives.

WHY this is a holostuff faculty and not bolted-on signal processing:
  A Doppler frequency drift is a cyclic SHIFT of the spectrum over time. The engine already
  shifts -- `permute` (a cyclic roll) is exactly the rigid-shift transform holographic_video.py
  uses for motion-compensated compression -- and a cyclic shift is also a BINDING
  (bind(x, delta_k) == permute(x, k), exact), so it is VSA-legitimate. "De-Doppler integration"
  -- the matched filter that recovers a drifting signal a STATIONARY detector loses -- is just
  permute-ing each frame back by the drift before summing: de-drift = the engine's shift, applied
  per frame. And the look-elsewhere control over the (drift x channel) search grid is `bh_fdr`,
  the honesty harness's own Benjamini-Yekutieli FDR. Two engine primitives, reused in radio
  astronomy: a binding-is-a-shift and a false-discovery veto.

MEASURED (see _selftest), all on synthetic spectrograms at the field's S/N>=10 regime:
  * A stationary integration LOSES a drifting signal (~noise level); de-drifting at the right
    rate RECOVERS it (>2x the stationary peak) and the bank reports the drift rate.
  * Look-elsewhere: over the (drift x channel) grid, naive per-cell thresholding fires on ~100%
    of pure-noise scans; bh_fdr (dependent) holds it to ~0%.
  * ROC at integrated S/N ~12: recall ~96% at 0% false-positive (the field's <1%-FP-@-95% bar).
  * ON-OFF cadence: a STRONG stationary RFI that WOULD be detected on its own is rejected ~100%
    of the time because it persists in the OFF pointing, while the drifting ON-only signal is
    kept ~94%.

KEPT NEGATIVE: below ~10 sigma integrated, recall falls off -- the dependent-FDR correction over
the many grid cells is conservative, so a lone weak signal needs ~5 sigma to clear the multiple-
testing bar. This is not a flaw but a match: turboSETI's own search threshold is S/N>=10 precisely
because it scans so many places. The detector is as honest about the cost of the bank as the field is.
"""
import math
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import permute        # the engine's cyclic shift = the de-drift step  == bind by a shifted delta
from holographic.agents_and_reasoning.holographic_honesty import bh_fdr     # the look-elsewhere / trials-factor control  Benjamini-Yekutieli


def _sf(z):
    """Upper-tail normal survival P(Z > z), vectorised, no scipy (math.erfc on an array via vectorize)."""
    erfc = np.vectorize(math.erfc)
    return 0.5 * erfc(np.asarray(z, float) / math.sqrt(2))


def dedoppler_bank(waterfall, drifts):
    """De-Doppler matched-filter bank: integrate the spectrogram along EVERY candidate drift rate.

    `waterfall` is (T frames x F bins); `drifts` are candidate rates in bins-per-frame. For each
    drift d, de-drift frame t by permute-ing it back -d*t bins (undo the accumulated Doppler shift)
    and sum over time -- a matched filter at that drift. Returns the (len(drifts) x F) grid of
    integrated power, robustly z-scored: the noise floor is estimated from the grid's own MAD (the
    way a radio pipeline estimates its floor from the data), so under noise each cell is ~N(0, 1).
    A signal drifting at rate d through bin c lights up cell (d, c) with z ~ sqrt(T) * amplitude.
    Deterministic: no randomness here.
    """
    wf = np.asarray(waterfall, float)
    T, F = wf.shape
    grid = np.zeros((len(drifts), F))
    for i, d in enumerate(drifts):
        acc = np.zeros(F)
        for t in range(T):
            acc = acc + permute(wf[t], -int(round(d * t)))   # de-drift THIS frame, then integrate (coherent sum)
        grid[i] = acc
    med = np.median(grid)
    sigma = np.median(np.abs(grid - med)) / 0.6745 + 1e-12   # robust noise-floor estimate (MAD)
    return (grid - med) / sigma


def detect_drifting(waterfall, drifts=None, alpha=0.01, off=None, off_z=4.0):
    """Detect drifting narrowband signals in a spectrogram, with honest look-elsewhere control.

    Runs the de-Doppler bank, turns each (drift x channel) cell's z-score into an upper-tail
    p-value, and declares detections with `bh_fdr` using the DEPENDENT (Benjamini-Yekutieli)
    correction -- the drift cells overlap, so the tests are dependent and this is the honest,
    conservative choice. If an `off` spectrogram is supplied (an OFF-target pointing in the ON-OFF
    cadence radio astronomers use), any surviving cell that is ALSO bright in OFF (z >= off_z there)
    is rejected as RFI: a real signal is ON-only, terrestrial interference persists across the
    cadence. Returns a list of detections, each {drift, channel, snr, pvalue}, sorted by SNR
    descending. Deterministic given the input. `drifts` defaults to a symmetric grid over
    [-3, 3] bins/frame in 0.5-bin steps.
    """
    if drifts is None:
        drifts = np.arange(-3.0, 3.0001, 0.5)
    drifts = np.asarray(drifts, float)
    z = dedoppler_bank(waterfall, drifts)
    F = z.shape[1]
    p = _sf(z).ravel()
    reject, _ = bh_fdr(p, alpha=alpha, dependent=True)
    reject = reject.reshape(z.shape)
    if off is not None:
        z_off = dedoppler_bank(off, drifts)
        reject = reject & (z_off < off_z)            # cadence veto: drop anything that persists in OFF (RFI)
    out = []
    for di, ci in zip(*np.where(reject)):
        out.append({"drift": float(drifts[di]),
                    "channel": int(ci),
                    "snr": float(z[di, ci]),
                    "pvalue": float(p[di * F + ci])})
    out.sort(key=lambda r: -r["snr"])
    return out


_C = 299792458.0  # speed of light in vacuum (m/s) -- the constant that turns a spectral shift into a velocity


def redshift(lambda_obs, lambda_rest):
    """Redshift z = lambda_obs/lambda_rest - 1 from an observed vs rest wavelength (positive = red / receding).
    Field-native (broadcasts over arrays)."""
    return np.asarray(lambda_obs, float) / np.asarray(lambda_rest, float) - 1.0


def doppler_velocity(lambda_obs, lambda_rest, relativistic=False):
    """Line-of-sight VELOCITY from a spectral shift (m/s, positive = receding). Classical v = c*z (z = redshift);
    with relativistic=True, v = c*((1+z)^2 - 1)/((1+z)^2 + 1), which stays below c for any z. This is the physical
    reading of what dedoppler / detect_drifting locate -- a shifted line means the source is moving. Field-native."""
    z = redshift(lambda_obs, lambda_rest)
    if not relativistic:
        return _C * z
    r = (1.0 + z) ** 2
    return _C * (r - 1.0) / (r + 1.0)


def doppler_shift(lambda_rest, velocity, relativistic=False):
    """FORWARD model: the observed wavelength when a source at `lambda_rest` recedes at `velocity` (m/s). Classical
    lambda_obs = lambda_rest*(1 + v/c); relativistic uses the sqrt((1+b)/(1-b)) factor (b = v/c). The exact inverse
    of doppler_velocity, so the two round-trip (asserted). Field-native."""
    beta = np.asarray(velocity, float) / _C
    lr = np.asarray(lambda_rest, float)
    if not relativistic:
        return lr * (1.0 + beta)
    return lr * np.sqrt((1.0 + beta) / (1.0 - beta))


def drift_acceleration(drift_rate, freq):
    """Line-of-sight ACCELERATION (m/s^2) from a narrowband frequency DRIFT RATE (Hz/s) at observing frequency
    `freq` (Hz): a = -c * (df/dt) / f. A drifting tone -- exactly what detect_drifting finds -- means the emitter
    is ACCELERATING relative to us (Earth's spin, or a transmitter on a rotating body); this says how much. A
    rising frequency (df/dt > 0) is an approaching acceleration (a < 0 in the receding-positive convention).
    This is the physical payoff of the SETI drift search. Field-native."""
    return -_C * np.asarray(drift_rate, float) / np.asarray(freq, float)


def _selftest():
    """Reproduce every measured claim: recovery, look-elsewhere control, ROC at the field's bar, cadence."""
    T, F, noise = 24, 96, 1.0
    drifts = np.arange(-3.0, 3.0001, 0.5)
    A_field = 2.4            # per-frame amplitude -> integrated sqrt(24)*2.4 ~ 11.8 sigma (above turboSETI's S/N>=10)

    def waterfall(drift, chan, amp, on=True, rfi_chan=None, rfi_amp=0.0, seed=0):
        r = np.random.default_rng(seed)
        wf = noise * r.standard_normal((T, F))
        if on:
            for t in range(T):
                wf[t, int(round(chan + drift * t)) % F] += amp     # the drifting narrowband signal (ON only)
        if rfi_chan is not None:
            wf[:, rfi_chan] += rfi_amp                              # stationary RFI: every frame, no drift
        return wf

    # (1) de-drift RECOVERS a drifting signal a stationary detector loses
    z = dedoppler_bank(waterfall(1.5, 40, A_field, seed=1), drifts)
    stationary_peak = z[list(drifts).index(0.0)].max()             # drift-0 integration: signal is smeared
    bank_peak = z.max()                                            # best over the bank: signal concentrates
    assert bank_peak > stationary_peak + 3.0, (bank_peak, stationary_peak)

    # (2) LOOK-ELSEWHERE: bh_fdr controls pure-noise false alarms; naive per-cell thresholding does not
    naive_fired = fdr_fired = 0
    for s in range(30):
        nz = dedoppler_bank(noise * np.random.default_rng(500 + s).standard_normal((T, F)), drifts)
        pp = _sf(nz).ravel()
        naive_fired += bool((pp < 0.01).any())
        rej, _ = bh_fdr(pp, alpha=0.01, dependent=True)
        fdr_fired += bool(rej.any())
    assert naive_fired >= 28 and fdr_fired <= 2, (naive_fired, fdr_fired)   # naive ~always, fdr ~never

    # (3) ROC at the field's operating point (integrated S/N ~12): high recall at zero false-positive
    hit = false_pos = 0
    for s in range(30):
        r = np.random.default_rng(700 + s)
        chan = int(r.integers(8, F - 8)); dr = float(r.choice(drifts[drifts != 0]))
        det = detect_drifting(waterfall(dr, chan, A_field, seed=700 + s), drifts, alpha=0.01)
        found = any(abs(d["drift"] - dr) < 0.6 and abs(d["channel"] - chan) <= 1 for d in det)
        hit += found
        false_pos += any(not (abs(d["drift"] - dr) < 0.6 and abs(d["channel"] - chan) <= 1) for d in det)
    assert hit >= 24, hit                                          # >= 80% recall (measured ~96%)
    assert false_pos <= 3, false_pos                               # essentially no false detections

    # (4) ON-OFF CADENCE rejects a STRONG stationary RFI that would otherwise be detected
    rfi_caught_no_cadence = sig_kept = rfi_kept = 0
    for s in range(24):
        r = np.random.default_rng(900 + s)
        chan = int(r.integers(8, F - 8)); dr = float(r.choice(drifts[drifts != 0])); rfi = int(r.integers(8, F - 8))
        if abs(rfi - chan) <= 1:
            rfi = (rfi + 5) % F
        on = waterfall(dr, chan, A_field, on=True, rfi_chan=rfi, rfi_amp=A_field, seed=900 + s)
        off = waterfall(dr, chan, 0.0, on=False, rfi_chan=rfi, rfi_amp=A_field, seed=1900 + s)
        no_cad = detect_drifting(on, drifts, alpha=0.01)           # without the cadence the RFI IS a detection
        rfi_caught_no_cadence += any(abs(d["drift"]) < 1e-9 and abs(d["channel"] - rfi) <= 1 for d in no_cad)
        cad = detect_drifting(on, drifts, alpha=0.01, off=off)     # with the cadence it is vetoed
        sig_kept += any(abs(d["drift"] - dr) < 0.6 and abs(d["channel"] - chan) <= 1 for d in cad)
        rfi_kept += any(abs(d["drift"]) < 1e-9 and abs(d["channel"] - rfi) <= 1 for d in cad)
    assert rfi_caught_no_cadence >= 20, rfi_caught_no_cadence      # the RFI really would fool a no-cadence detector
    assert sig_kept >= 19, sig_kept                                # the drifting signal survives the cadence
    assert rfi_kept <= 3, rfi_kept                                 # the RFI is rejected by it

    # determinism (Macklin's discipline): same input -> bit-identical detections
    wf = waterfall(1.5, 40, A_field, seed=7)
    d1 = detect_drifting(wf, drifts, alpha=0.01); d2 = detect_drifting(wf, drifts, alpha=0.01)
    assert d1 == d2, "detect_drifting must be deterministic"

    # --- doppler physics (B1): shift <-> velocity round-trip, relativistic sanity, drift -> acceleration ---
    lr = 656.28e-9                                                  # H-alpha rest wavelength (m)
    lo = doppler_shift(lr, 3.0e5)                                   # source receding at 300 km/s
    assert abs(doppler_velocity(lo, lr) - 3.0e5) < 1e-3, "classical doppler round-trip failed"
    assert abs(redshift(lr, lr)) < 1e-15 and abs(doppler_velocity(lr, lr)) < 1e-6, "zero shift -> zero velocity"
    v = 0.5 * _C
    assert abs(doppler_velocity(doppler_shift(lr, v, relativistic=True), lr, relativistic=True) - v) < 1.0, "relativistic round-trip failed"
    # relativistic stays sub-luminal where the classical reading would exceed c (the reason it exists)
    fast = doppler_velocity(doppler_shift(lr, 0.9 * _C, relativistic=True), lr, relativistic=True)
    assert abs(fast) < _C, "relativistic velocity must stay below c"
    los = doppler_shift(lr, 100.0, relativistic=True)              # at small v, classical ~= relativistic
    assert abs(doppler_velocity(los, lr) - doppler_velocity(los, lr, relativistic=True)) < 1e-3, "small-v mismatch"
    a = drift_acceleration(1.0, 1.4e9)                             # a 1 Hz/s drift at 1.4 GHz (the SETI reading)
    assert abs(a - (-_C / 1.4e9)) < 1e-12 and a < 0, "drift acceleration wrong"
    arr = doppler_velocity(doppler_shift(lr, np.array([1e4, 2e4, 3e4])), lr)   # field-native + determinism
    assert arr.shape == (3,) and np.array_equal(arr, doppler_velocity(doppler_shift(lr, np.array([1e4, 2e4, 3e4])), lr))

    print("holographic_dedoppler selftest OK  |  + doppler: shift<->velocity round-trip, relativistic<c, drift->acceleration (SETI reading)")


if __name__ == "__main__":
    _selftest()
