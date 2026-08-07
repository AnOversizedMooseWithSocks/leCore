"""holographic_sciencereport.py -- SCI-5: one front door for the science instruments.

The scientist's entry point, mirroring market_residual_report's shape: ONE call, an explicit
`kind`, and a uniform report {kind, verdict, why, result} coming back -- where `result` is the
full instrument output for the audit trail. No kind is ever guessed: routing a light curve into
a spectrum instrument would produce a confident nonsense verdict, and a wrong confident answer
is the one failure a refusing instrument family must not commit at its own front door.

Kinds and the instruments they route to (each documented, with its literature ancestor, in
docs/SCIENCE_INSTRUMENTS.md):

  'light_curve'   transit_search        box-matched period hunt, block-shuffle null (Kovacs 2002)
  'pulsar_panel'  hd_search             Hellings-Downs pattern with the sky-scramble null
  'spectrum'      find/identify + z     lines, margin identification, one-shift-or-refuse
  'decay'         fit_decay             A exp(-lambda t)+C with the truncation flag
  'levels'        level_statistics      Poisson/GOE/GUE spacing ratios (Atas 2013), refusing
  'chsh'          chsh_verdict          Bell verdict with the Tsirelson alarm
  'series'        residual_ladder       the interrogation tower (level/scale/fold rungs)

Every route inherits its instrument's refusals verbatim -- 'underpowered', 'indeterminate',
'no-consistent-shift', 'suspect-instrument' are results, not errors.
"""

import numpy as np

KINDS = ("light_curve", "pulsar_panel", "spectrum", "decay", "levels", "chsh", "series")


def _get(data, *names):
    """Pull named fields from a dict, tolerating a tuple/list in declaration order -- the front
    door meets scientists where their data already is, but NEVER renames or reinterprets."""
    if isinstance(data, dict):
        missing = [n for n in names if n not in data]
        if missing:
            raise ValueError("kind requires fields %s; missing %s" % (list(names), missing))
        return [data[n] for n in names]
    seq = list(data) if isinstance(data, (tuple, list)) else [data]
    if len(seq) != len(names):
        raise ValueError("expected %d fields %s in order, got %d" % (
            len(names), list(names), len(seq)))
    return seq


def science_report(data, kind, seed=0, **kw):
    """THE FRONT DOOR: route `data` to the matching science instrument and return the uniform
    report {'kind', 'verdict', 'why', 'result'}. `kind` is explicit and mandatory -- see KINDS;
    an unknown kind raises with the full list rather than guessing. Extra keyword arguments pass
    through to the instrument (e.g. min_period/max_period for 'light_curve', catalog for
    'spectrum', positions travel inside `data` for 'pulsar_panel')."""
    if kind == "light_curve":
        from holographic.sampling_and_signal.holographic_transitbox import transit_search
        t, y = _get(data, "times", "values")
        t = np.asarray(t, float); y = np.asarray(y, float)
        span = float(t[-1] - t[0])
        kw.setdefault("min_period", span / 50.0)
        kw.setdefault("max_period", span / 3.0)
        r = transit_search(t, y, seed=seed, **kw)
    elif kind == "pulsar_panel":
        from holographic.sampling_and_signal.holographic_pulsarpanel import hd_search
        panel, pos = _get(data, "panel", "positions")
        r = hd_search(panel, pos, seed=seed, **kw)
    elif kind == "spectrum":
        from holographic.sampling_and_signal.holographic_spectralline import (
            find_lines, identify_lines, redshift_verdict)
        catalog = kw.pop("catalog", None)
        x, y = _get(data, "x", "y")
        r = find_lines(x, y, seed=seed, **kw)
        centers = [l["center"] for l in r["lines"]]
        if catalog is not None:
            r["identification"] = identify_lines(centers, catalog)
            r["redshift"] = redshift_verdict(centers, catalog, seed=seed) if centers else None
        n = len(r["lines"])
        if catalog is not None and r.get("redshift"):
            r["verdict"] = r["redshift"]["verdict"]
            r["why"] = "%d line(s) gated; %s" % (n, r["redshift"]["why"])
        else:
            r["verdict"] = "lines-found" if n else "no-lines"
            r["why"] = ("%d line(s) beat the noise-only max null" % n) if n else \
                "no candidate beats the largest excursion pure noise of this length produces"
    elif kind == "decay":
        from holographic.sampling_and_signal.holographic_spectralline import fit_decay
        t, y = _get(data, "t", "y")
        r = fit_decay(t, y, seed=seed, **kw)
    elif kind == "levels":
        from holographic.sampling_and_signal.holographic_quantumstats import level_statistics
        (levels,) = _get(data, "levels")
        r = level_statistics(levels, seed=seed, **kw)
    elif kind == "chsh":
        from holographic.sampling_and_signal.holographic_quantumstats import chsh_verdict
        a_s, b_s, A, B = _get(data, "a_setting", "b_setting", "a_out", "b_out")
        r = chsh_verdict(a_s, b_s, A, B, seed=seed, **kw)
    elif kind == "series":
        from holographic.sampling_and_signal.holographic_residualvoid import residual_ladder
        (y,) = _get(data, "y")
        r = residual_ladder(np.asarray(y, float), seed=seed, **kw)
        rungs = [lv.get("grammar", "?") for lv in r.get("tower", []) if lv.get("removed_frac", 0)]
        r["verdict"] = r.get("terminal", "?")
        r["why"] = ("the interrogation tower ran %d rung(s) [%s] and terminated '%s'"
                    % (len(r.get("tower", [])), ", ".join(rungs) or "none", r["verdict"]))
    else:
        raise ValueError("unknown kind %r -- one of %s" % (kind, list(KINDS)))
    return {"kind": kind, "verdict": r.get("verdict", "?"), "why": r.get("why", ""), "result": r}


def _selftest():
    # one modest plant per kind: the front door's job is ROUTING + the uniform shape, so each
    # plant is the instrument's own easy case -- the hard cases live in the instruments' tests.
    rng = np.random.default_rng(0)

    t = np.arange(1500, dtype=float)
    y = 0.002 * rng.standard_normal(1500); y[(t % 137.0) < 8] -= 0.012
    rep = science_report({"times": t, "values": y}, "light_curve", n_periods=400, n_null=24)
    assert rep["kind"] == "light_curve" and rep["verdict"] == "periodic", rep["why"]
    assert abs(rep["result"]["period"] - 137.0) < 5 or \
           any(abs(f["period"] - 137.0) < 5 for f in rep["result"]["family"])

    from holographic.sampling_and_signal.holographic_pulsarpanel import make_hd_panel
    panel, pos = make_hd_panel(k=10, n=1000, gw_amp=0.5, seed=3, mode="hd")
    rep = science_report({"panel": panel, "positions": pos}, "pulsar_panel", n_null=24)
    assert rep["verdict"] == "hd-consistent", rep["why"]

    BAL = {"H-alpha": 656.279, "H-beta": 486.135, "H-gamma": 434.047, "H-delta": 410.173}
    x = np.linspace(400, 700, 2400)
    ys = 10.0 + 0.15 * rng.standard_normal(len(x))
    for w in BAL.values():
        ys += 2.5 * np.exp(-0.5 * ((x - w * 1.0213) / 0.35) ** 2)
    rep = science_report({"x": x, "y": ys}, "spectrum", catalog=BAL)
    assert rep["verdict"] == "consistent-shift" and abs(rep["result"]["redshift"]["z"] - 0.0213) < 1e-3

    td = np.linspace(0, 150, 300)
    rd = np.random.default_rng(7)                      # plants own their seeds -- standing rule
    rep = science_report({"t": td, "y": 25 * np.exp(-0.05 * td) + 2 + 0.4 * rd.standard_normal(300)},
                         "decay")
    assert rep["verdict"] == "decay" and abs(rep["result"]["lam"] - 0.05) < 0.01

    M = rng.standard_normal((300, 300))
    rep = science_report({"levels": np.linalg.eigvalsh((M + M.T) / 2)}, "levels")
    assert rep["verdict"].startswith("goe"), rep["verdict"]

    from holographic.sampling_and_signal.holographic_quantumstats import make_chsh_trials
    a_s, b_s, A, B = make_chsh_trials(3000, "quantum", seed=5)
    rep = science_report({"a_setting": a_s, "b_setting": b_s, "a_out": A, "b_out": B}, "chsh")
    assert rep["verdict"].startswith("nonclassical"), rep["verdict"]

    e = np.random.default_rng(11).standard_normal(1200)
    ar = np.zeros(1200)
    for i in range(1, 1200):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    rep = science_report({"y": ar}, "series", n_surrogates=32)
    assert rep["kind"] == "series" and rep["verdict"] in ("irreducible", "rungs-exhausted")
    assert "rung" in rep["why"]

    try:
        science_report({}, "telescope")
        raise RuntimeError("unknown kind must raise")
    except ValueError as ex:
        assert "light_curve" in str(ex), "the error must LIST the kinds, not just refuse"

    print("holographic_sciencereport selftest OK -- 7 kinds routed with uniform verdicts "
          "(transit, HD panel, redshift, decay, GOE, CHSH violation, ladder terminal), unknown "
          "kind refused with the list")


if __name__ == "__main__":
    _selftest()
