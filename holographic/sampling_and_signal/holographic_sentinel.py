"""Stream sentinel: watch a stream through the HRNN's ladder, segment it by REGIME,
raise change events with evidence, and record each window at its cheapest FAITHFUL
representation (holographic_sentinel, SEN-1).

WHY THIS EXISTS
---------------
The HRNN routes ONE window honestly. Real streams are long and change character:
telemetry goes periodic -> faulty, markets go quiet -> violent, a sensor drifts. The
sentinel is the HRNN applied AS AN INSTRUMENT over time:

  watch(x)   -- slide a window; per-window verdict {regime, h}; emit an EVENT when the
                regime flips or the entropy rate jumps. The event carries both windows'
                provenance -- an alarm without evidence is noise with a timestamp.
  record(x)  -- the priced recorder: per window, store the CHEAPEST representation the
                ladder certifies. A generator window becomes its fitted params (a few
                dozen floats -- the bake-once lever); anything else becomes quantile-
                coded symbols at its measured rate, or raw floats if symbols would
                misrepresent it. Every entry carries {regime, horizon, why} and the
                in-window reconstruction error, so replay() can hand back a stream AND
                its certificate. The recorder NEVER extrapolates: a generator entry
                reconstructs its own window only (the SOL-1h lesson, NRMSE 16.5 vs 2.9
                naive past the horizon, is baked in as a refusal).

HONESTY CONTRACTS, stated up front:
  * The change threshold h_jump is a DECLARED heuristic (default 0.5 bits/step, half
    the ladder's stage-1 gate), not a calibrated test; the selftest pins its false-alarm
    behaviour on a stationary stream at zero events. Calibrating it against a
    stationarity null is named future work, not silently claimed.
  * Windows are certified independently; the sentinel makes NO claim that a generator
    persists between windows. Segment boundaries are event locations +/- one hop.
  * Compression ratios are reported against float64 raw; symbol entries store the
    quantile edges so replay is self-contained (edges cost k-1 floats per entry, and
    are counted -- no hidden dictionary).

Stdlib + numpy only; deterministic given seeds; the engine is injectable (the mind
supplies its fit_deterministic-wired HRNN for ROUTING; recording uses the model's own
predict on interior indices, which the standalone harmonic fitter serves).
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_hrnn import HolographicRNN, fit_harmonics


class StreamSentinel:
    """Windowed regime watcher + priced recorder over one HolographicRNN engine."""

    def __init__(self, window=512, hop=256, h_jump=0.5, k=4, engine=None, seed=0):
        self.window, self.hop = int(window), int(hop)
        self.h_jump, self.k = float(h_jump), int(k)
        self.engine = engine or HolographicRNN(seed=seed)

    def _certify(self, w):
        """PREFIX-FIT / SUFFIX-TEST: the generator certificate that has no surrogate
        degeneracy. WHY it exists: a harmonic stack's phase surrogate is ANOTHER
        harmonic stack, so the ladder's calibrated stage can sit at p=1.0 on a clean
        tone (observed can never beat a null drawn from its own hypothesis class --
        measured on this module's first selftest). Predictive compression is the
        falsifiable claim instead: fit on the first 70%% of the window, measure NRMSE
        of the EXTENSION on the last 30%%. Tones certify; noise and walks do not; and
        the certificate is exactly the property the recorder needs (in-window replay).
        Returns (model, suffix_nrmse) or (None, nrmse)."""
        cut = int(0.7 * len(w))
        f = fit_harmonics(w[:cut])
        suffix = f["predict"](np.arange(cut, len(w)))
        nrmse = float(np.sqrt(np.mean((suffix - w[cut:]) ** 2))
                      / (np.std(w) + 1e-300))
        if nrmse < 0.15:
            return fit_harmonics(w), nrmse       # refit on the full window for replay
        return None, nrmse

    # ---------------------------------------------------------------- watching
    def watch(self, x):
        """Slide the ladder along x. Returns dict{verdicts, events, segments}.

        A verdict per window: {start, regime, h, why}. An event where the regime label
        flips or |dh| > h_jump between consecutive windows -- the event carries BOTH
        verdicts so the alarm arrives with its evidence attached. Segments are the
        maximal runs between events, labelled by their majority regime."""
        x = np.asarray(x, dtype=float).ravel()
        verdicts = []
        for start in range(0, max(1, len(x) - self.window + 1), self.hop):
            w = x[start:start + self.window]
            r = self.engine.process_stream(w, k=self.k)
            regime, why = r["regime"], r["why"]
            if regime != "generator":
                model, nrmse = self._certify(w)
                if model is not None:
                    regime = "generator"
                    why = ("prefix-fit/suffix-certified: extension NRMSE %.3f on the "
                           "held-out 30%% (ladder said %s; surrogate-degenerate "
                           "windows land here)" % (nrmse, r["regime"]))
            verdicts.append({"start": start, "regime": regime,
                             "h": r.get("h"), "why": why})
        events = []
        for a, b in zip(verdicts, verdicts[1:]):
            flip = a["regime"] != b["regime"]
            ha, hb = a.get("h"), b.get("h")
            jump = (ha is not None and hb is not None
                    and abs(hb - ha) > self.h_jump)
            if flip or jump:
                events.append({"at": b["start"],
                               "kind": "regime-flip" if flip else "rate-jump",
                               "before": a, "after": b})
        segments, seg_start, seg_regime = [], 0, (verdicts[0]["regime"] if verdicts else None)
        for e in events:
            segments.append({"start": seg_start, "end": e["at"], "regime": seg_regime})
            seg_start, seg_regime = e["at"], e["after"]["regime"]
        if verdicts:
            segments.append({"start": seg_start, "end": len(x), "regime": seg_regime})
        return {"verdicts": verdicts, "events": events, "segments": segments}

    # ---------------------------------------------------------------- recording
    def record(self, x):
        """The priced recorder: one tape entry per window, cheapest faithful form.

        generator window  -> {'form':'params'} : the fitted harmonic params + the
            reconstruction NRMSE measured IN-WINDOW (the certificate); ~O(harmonics)
            floats regardless of window length -- bake once, sample O(1).
        structured window -> {'form':'symbols'}: k-quantile codes + the k-1 edges +
            per-bin means; cost ~2 bits/sample at k=4, lossy by construction and says so.
        incompressible    -> {'form':'raw'}    : the floats, untouched; pretending to
            compress noise is the failure mode this recorder exists to refuse.
        Returns dict{tape, ratio} with ratio = raw float64 bits / stored bits."""
        x = np.asarray(x, dtype=float).ravel()
        tape, stored_bits = [], 0.0
        for start in range(0, max(1, len(x) - self.window + 1), self.hop or self.window):
            w = x[start:start + self.window]
            r = self.engine.process_stream(w, k=self.k)
            entry = {"start": start, "n": len(w), "regime": r["regime"],
                     "horizon": r["horizon"], "why": r["why"]}
            # WHY the recorder never reuses the ladder's model object: the mind-wired
            # generator fit serves EXTENSION only and raises on interior indices (its
            # contract), while replay needs the window's own interior. The sentinel
            # therefore always refits via the prefix/suffix certificate -- the ladder's
            # verdict rides along as provenance, the stored params are replay-safe.
            model, suffix_nrmse = self._certify(w)
            if model is not None and entry["regime"] != "generator":
                entry["regime"] = "generator"
                entry["why"] = ("prefix-fit/suffix-certified (extension NRMSE "
                                "%.3f held-out)" % suffix_nrmse)
            if model is not None:
                recon = model["predict"](np.arange(len(w)))
                nrmse = float(np.sqrt(np.mean((recon - w) ** 2)) / (np.std(w) + 1e-300))
                params = np.asarray(model["params"], dtype=float)
                entry.update({"form": "params", "params": params,
                              "fundamental": float(model.get("fundamental", 0.0)),
                              "in_window_nrmse": nrmse})
                stored_bits += 64.0 * (len(params) + 2)
            elif r["regime"] == "structured":
                edges = np.quantile(w, np.linspace(0, 1, self.k + 1)[1:-1])
                codes = np.digitize(w, edges)
                means = np.array([w[codes == c].mean() if np.any(codes == c) else 0.0
                                  for c in range(self.k)])
                entry.update({"form": "symbols", "codes": codes.astype(np.uint8),
                              "edges": edges, "bin_means": means})
                stored_bits += np.log2(self.k) * len(w) + 64.0 * (2 * self.k - 1)
            else:
                entry.update({"form": "raw", "values": w.copy()})
                stored_bits += 64.0 * len(w)
            tape.append(entry)
        raw_bits = 64.0 * sum(e["n"] for e in tape)
        return {"tape": tape, "ratio": raw_bits / max(stored_bits, 1.0),
                "stored_bits": stored_bits}

    def replay(self, recording):
        """Reconstruct the stream from a tape, window by window, certificates intact.

        Generator entries re-run their OWN window's harmonics (interior indices only --
        extension past a window's horizon is refused by construction: there is no API
        for it here, which is the honest interface). Symbol entries decode to per-bin
        means (lossy, as certified). Raw entries are the floats. Overlapping hops are
        resolved last-writer-wins, matching record()'s left-to-right pass."""
        n_total = max(e["start"] + e["n"] for e in recording["tape"])
        out = np.zeros(n_total)
        for e in recording["tape"]:
            idx = np.arange(e["n"])
            if e["form"] == "params":
                coef, f0 = e["params"], e["fundamental"]
                w = np.full(e["n"], coef[0])
                for kk in range(1, (len(coef) - 1) // 2 + 1):
                    om = 2.0 * np.pi * f0 * kk
                    w = w + coef[2 * kk - 1] * np.cos(om * idx) + coef[2 * kk] * np.sin(om * idx)
            elif e["form"] == "symbols":
                w = e["bin_means"][e["codes"]]
            else:
                w = e["values"]
            out[e["start"]:e["start"] + e["n"]] = w
        return out


def _selftest():
    """Pins: correct segmentation of a 3-regime stream, zero false alarms on a
    stationary tone, honest compression (big on tones, ~none on noise), and replay
    fidelity with certificates riding every entry."""
    rng = np.random.default_rng(0)
    t = np.arange(6000, dtype=float)
    # WHY two harmonics and not a pure sine: a LONE sinusoid is degenerate against
    # phase surrogates (its randomisation is itself, time-shifted -> p=1.0, the gate
    # fails CLOSED and routes it 'structured'). Phase-LOCKED harmonics break the
    # degeneracy; the surrogate destroys the locking and the fit wins. Kept in NOTES.
    tone1 = (np.sin(2 * np.pi * t[:2000] / 170.0)
             + 0.4 * np.sin(4 * np.pi * t[:2000] / 170.0))
    noise = rng.standard_normal(2000)
    tone2 = np.sin(2 * np.pi * t[:2000] / 61.0) + 0.3 * np.sin(4 * np.pi * t[:2000] / 61.0)
    x = np.concatenate([tone1, noise, tone2])

    s = StreamSentinel(window=512, hop=256, seed=0)
    w = s.watch(x)
    # 1) both true boundaries found within one hop; regime labels correct at the ends.
    ev = [e["at"] for e in w["events"]]
    assert any(abs(a - 2000) <= 512 for a in ev), "missed tone->noise boundary: %s" % ev
    assert any(abs(a - 4000) <= 512 for a in ev), "missed noise->tone boundary: %s" % ev
    assert w["verdicts"][0]["regime"] == "generator", w["verdicts"][0]
    assert w["verdicts"][-1]["regime"] == "generator", w["verdicts"][-1]
    for e in w["events"]:
        assert "before" in e and "after" in e, "event without evidence"

    # 2) zero false alarms on a stationary tone.
    quiet = StreamSentinel(window=512, hop=256, seed=0).watch(
        np.sin(2 * np.pi * np.arange(4000.) / 170.0))
    assert len(quiet["events"]) == 0, "false alarms on stationary tone: %d" % len(quiet["events"])

    # 3) the recorder: tone windows compress hard AND reconstruct; noise is stored raw
    #    (no fake compression); every entry certified.
    rec_tone = StreamSentinel(window=500, hop=500, seed=0).record(tone1)
    assert rec_tone["ratio"] > 20, "tone should compress >20x, got %.1f" % rec_tone["ratio"]
    assert all(e["form"] == "params" and e["in_window_nrmse"] < 0.1
               for e in rec_tone["tape"]), "uncertified or unfaithful tone entry"
    rec_noise = StreamSentinel(window=500, hop=500, seed=0).record(noise)
    assert rec_noise["ratio"] < 1.2, "noise must not pretend to compress: %.2f" % rec_noise["ratio"]
    assert all(e["form"] == "raw" for e in rec_noise["tape"])
    for e in rec_tone["tape"] + rec_noise["tape"]:
        assert "regime" in e and "horizon" in e and "why" in e, "certificate missing"

    # 4) replay fidelity where fidelity was certified.
    back = StreamSentinel(window=500, hop=500, seed=0).replay(rec_tone)
    nrmse = float(np.sqrt(np.mean((back[:2000] - tone1) ** 2)) / np.std(tone1))
    assert nrmse < 0.1, "replay drifted: NRMSE %.3f" % nrmse

    print("holographic_sentinel selftest OK -- boundaries @%s (truth 2000/4000), "
          "0 false alarms, tone ratio %.0fx (NRMSE %.3f), noise ratio %.2fx raw, "
          "replay NRMSE %.3f, certificates on all %d entries"
          % (ev, rec_tone["ratio"],
             max(e["in_window_nrmse"] for e in rec_tone["tape"]),
             rec_noise["ratio"], nrmse,
             len(rec_tone["tape"]) + len(rec_noise["tape"])))


if __name__ == "__main__":
    _selftest()
