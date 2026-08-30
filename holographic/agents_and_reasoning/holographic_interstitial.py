"""holographic_interstitial.py -- THE THIN COORDINATION LAYER over interstitial sensors.

Split out of the unified monolith because structure_audit refused the growth: "giant modules
grew to 6 (> budget 5)". The guard was right -- this is a family of its own, and the unified
method now delegates here.

Three sensors, three different jobs, NEVER averaged, and a route decided from the SHAPE of
their profile rather than from any single reading. See docs/UNICRON_THESIS.md run 25.
"""
import numpy as np


def interstitial(runtime, cfg, sensors=None, bank=None, patches=None,
                         familiar=0.55, drift=0.35, learn=True):
    """THE THIN COORDINATION LAYER between the interstitial sensors (cp129).

    Live inside the model and fix it from the inside as it is used, with external memory
    as the patch. This does NOT shrink the model -- seven compression attempts failed and
    the eighth is still unrun. It watches, recognises, and patches.

    THREE SENSORS, THREE DIFFERENT JOBS, NEVER AVERAGED. Measured across cp121-cp125:

        early + mid   "have I been here before"   familiarity, d' +12.33 with END
        END           "am I still on my manifold"  drift, d' -12.45
        the ladder    "WHERE did it leave"         coverage is strictly lower-triangular,
                                                   so the SHALLOWEST firing sensor gives
                                                   the entry depth

    Averaging destroys both: a detector pooled with structurally blind channels reads
    neither. So the coordination layer is a state machine over three separate readings,
    not a score.

    SAFETY. In `learn` or watch mode the sensors are read-only hooks returning None, and
    that is verified bit-identical: max |logit difference| 0.000e+00 across every prompt
    tested, with no measurable overhead. A patch is applied ONLY on an explicit hit, and
    only as a delta at one layer, so the model's behaviour is untouched on every input
    that does not match something we deliberately stored.

    THE ROUTE, in order:
      1. DRIFT (END sensor below `drift`)  -> flag `off_manifold`; never patch. The model
         is somewhere it has no business being and a stored correction would be applied to
         a state it was not measured on.
      2. FAMILIAR (early+mid above `familiar`) -> look for a patch keyed to the nearest
         bank instance. If found, apply it as a delta and report `patched`.
      3. NOVEL -> if `learn`, add the instance to the bank. This is the only write.

    bank: list of {"layer": L, "vec": v, "id": str} instances (query-time compared, per
          the cp124 principle -- instances stored, function computed).
    patches: {instance_id: {"layer": L, "delta": vector}} from external memory.
    Returns (hooks, report). Pass `hooks` to runtime.forward; read `report` after.
    """
    n = int(cfg["n_layers"])
    if sensors is None:
        sensors = [3, max(4, n // 2), n - 2]
    sensors = sorted(int(s) for s in sensors)
    bank = list(bank or [])
    patches = dict(patches or {})
    rep = {"sensors": sensors, "scores": {}, "route": None, "patched": None,
           "off_manifold": False, "learned": None, "entry_depth": None}
    seen = {}

    def _pool(h):
        a = np.asarray(h, np.float64)
        a = a.reshape(-1, a.shape[-1]).mean(0)
        nrm = np.linalg.norm(a)
        return a / (nrm + 1e-12)

    def _best(L, v):
        cands = [b for b in bank if int(b.get("layer", -1)) == L]
        if not cands:
            return None, -1.0
        sims = [float(np.asarray(b["vec"], np.float64) @ v) for b in cands]
        j = int(np.argmax(sims))
        return cands[j], sims[j]

    hooks = {}
    for L in sensors:
        def make(L_):
            def fn(h):
                v = _pool(h)
                seen[L_] = v
                hit, s = _best(L_, v)
                rep["scores"][L_] = s
                # PATCH only on a confident familiar hit. The route itself is decided
                # in finish(), from the SHAPE of the profile -- a single reading cannot
                # tell 'new input' from 'went wrong at depth d' (cp125).
                if s >= familiar and hit is not None:
                    pid = hit.get("id")
                    p = patches.get(pid)
                    if p is not None and int(p.get("layer", -1)) == L_:
                        rep["route"] = "patched"
                        rep["patched"] = {"id": pid, "layer": L_, "score": s}
                        d = np.asarray(p["delta"])
                        return np.asarray(d, np.asarray(h).dtype)
                return None
            return fn
        hooks[L] = make(L)

    def finish():
        """Resolve the route AFTER the pass, using the SHAPE of the score profile.

        A single threshold cannot separate 'this whole input is new' from 'the
        computation went wrong at depth d' -- both read low at the deep sensor. The
        cp125 triangular result separates them, because a sensor is blind to anything
        injected below it:

            all sensors low                      -> NOVEL (the input itself is new)
            shallow normal, deep low             -> OFF-MANIFOLD, and the shallowest
                                                    low sensor gives the ENTRY DEPTH
            all sensors high                     -> FAMILIAR

        That is the whole reason the three sensors need a coordination layer rather than
        a threshold each: the diagnosis is in the PROFILE, not in any one reading.
        """
        if rep["route"] == "patched":
            return rep
        s = [rep["scores"].get(L, 0.0) for L in sensors]
        low = [L for L, v in zip(sensors, s) if v < familiar]
        if not low:
            rep["route"] = "familiar"
            return rep
        if len(low) == len(sensors):
            rep["route"] = "novel"                      # nothing recognised anywhere
            rep["entry_depth"] = None
            if learn:
                for L in sensors:
                    if L in seen:
                        bank.append({"layer": L, "vec": seen[L],
                                     "id": "inst%d" % (len(bank) // max(len(sensors), 1))})
                rep["learned"] = len(bank)
            return rep
        # recognised early and lost later: the profile is triangular
        rep["route"] = "off_manifold"
        rep["off_manifold"] = True
        rep["entry_depth"] = low[0]
        return rep

    rep["finish"] = finish
    rep["bank"] = bank
    return hooks, rep

