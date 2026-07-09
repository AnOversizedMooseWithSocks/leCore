"""Decompose foreign data into a compact generating law -- MDL-gated symbolic regression.

WHY THIS EXISTS
---------------
This is the hard half of the generative compressor: the recipe-store (build 1) STORES a known
construction; this module SEARCHES for the construction behind data we were handed. The panel's verdict:
for measured data the residual is real and the search is the cost, so the search must be (a) tractable
and (b) gated against overfitting, or "extrapolation" becomes a liability (the kept negative from the
generative-compression debate: an over-complex fit nails the window and explodes outside it).

THE METHOD. Rather than enumerate EML/operator trees (combinatorially explosive, and EML is expensive to
evaluate -- the EML debate's kept negative), we search a *dictionary* of elementary basis functions
(SINDy-style sparse symbolic regression: Brunton, Proctor, Kutz 2016) by deterministic greedy forward
selection, and choose the model by **Minimum Description Length**:

    total bits = L(model) + L(residual)
               = (#terms x [index bits + coefficient bits])  +  (Gaussian coding cost of the residual)

MDL is the gate. A term is kept only if the bits it saves on the residual exceed the bits it costs to
name -- so the recovered law is the *shortest program that explains the data*, which is exactly the
parsimony that makes extrapolation valid. On data with no compact law (noise) MDL adds nothing: it
REFUSES to manufacture a formula. That is the honest "no free lunch", enforced rather than hoped for.

THE OUTPUT IS A SEED. The recovered `Formula` is a handful of numbers that regenerate and extrapolate
the signal -- the measured-regime analogue of build 1's `StructureRecipe`. Build 2 finds the recipe;
build 1 stores it; what the law does not capture is the residual a rate-distortion coder (B5) would take.

KEPT NEGATIVES (honest scope):
  * The dictionary bounds what is discoverable; a law outside it (or a nonlinear rate off the frequency
    grid) won't be found -- the search reaches only as far as its basis.
  * MDL's coefficient-cost constant trades sensitivity vs over-fitting; it is a knob, not a law.
  * This is the tractable proxy for the full EML-tree search; the uniform single-operator tree remains
    the theoretical search space (and a far larger one).

Pure NumPy + holostuff spirit, deterministic, no new dependencies.
"""

import numpy as np


# ---- the elementary-function dictionary: each atom is (kind, param) with an evaluator ----
def _eval_atom(kind, param, x):
    if kind == "const": return np.ones_like(x)
    if kind == "pow":   return x ** param
    if kind == "sin":   return np.sin(param * x)
    if kind == "cos":   return np.cos(param * x)
    if kind == "exp":   return np.exp(np.clip(param * x, -30, 30))
    if kind == "log":   return np.log(np.abs(x) + 1e-12)   # log|x| -- the log-space basis for power laws
    raise ValueError(kind)


def elementary_dictionary(powers=(1, 2, 3), freqs=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), rates=(-0.5, 0.5)):
    """A modest basis of elementary functions. Nonlinear rates (freq/decay) are handled by a small grid
    -- a standard way to avoid nonlinear optimisation; an off-grid rate is a kept negative."""
    atoms = [("pow", p) for p in powers]
    atoms += [("sin", w) for w in freqs] + [("cos", w) for w in freqs]
    atoms += [("exp", a) for a in rates]
    return atoms


def log_dictionary(powers=(1, 2, 3)):
    """Log-space basis for MULTIPLICATIVE laws. An additive fit here -- log y = c0 + c1*log|x| + c2*x + ...
    -- exponentiates to a PRODUCT of power and exponential factors in y: y = a * |x|^c1 * exp(c2*x + ...).
    log|x| catches power laws; powers of x catch exp-of-polynomial factors."""
    return [("log", 1)] + [("pow", p) for p in powers]


class Formula:
    """A recovered generating law: an intercept plus a few (atom, coefficient) terms. A tiny seed that
    regenerates and extrapolates the signal -- the measured-regime analogue of a StructureRecipe."""

    def __init__(self, intercept, terms, log_space=False):
        self.intercept = float(intercept)
        self.terms = terms                       # list of ((kind, param), coef)
        self.log_space = bool(log_space)         # if True the fit is in log y: generate exponentiates

    def generate(self, x):
        x = np.asarray(x, float)
        s = np.full_like(x, self.intercept)
        for (kind, param), c in self.terms:
            s = s + c * _eval_atom(kind, param, x)
        return np.exp(s) if self.log_space else s   # multiplicative law = exp of an additive log-space fit

    def model_bits(self, n_dict, coef_bits=20):
        """Description length of the formula itself: one index + one coefficient per term, plus intercept."""
        idx_bits = np.log2(max(n_dict, 2))
        return coef_bits + len(self.terms) * (idx_bits + coef_bits)

    # ---- the seed interface (mirrors StructureRecipe): a Formula IS the measured-regime recipe ----
    def to_recipe(self):
        """A portable generator record -- the scalar-signal analogue of StructureRecipe.to_dict()."""
        return {"kind": "formula", "intercept": self.intercept, "log_space": self.log_space,
                "terms": [[list(atom), coef] for atom, coef in self.terms]}

    @classmethod
    def from_recipe(cls, d):
        return cls(d["intercept"], [((t[0][0], t[0][1]), t[1]) for t in d["terms"]],
                   log_space=d.get("log_space", False))

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.to_recipe(), f)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as f:
            return cls.from_recipe(json.load(f))

    def recipe_bytes(self):
        import json
        return len(json.dumps(self.to_recipe()))

    def compression_ratio(self, n_samples, dtype_bytes=4):
        return (n_samples * dtype_bytes) / self.recipe_bytes()

    def __repr__(self):
        parts = [f"{self.intercept:.3g}"] + [f"{c:+.3g}*{k}({p})" for (k, p), c in self.terms]
        inner = " ".join(parts)
        return f"Formula[ exp({inner}) ]" if self.log_space else f"Formula[ {inner} ]"


def _resid_bits(resid_rms, n, data_std):
    """Gaussian coding cost of the residual relative to a data-precision floor (bits)."""
    floor = 1e-3 * data_std + 1e-12
    return n * max(0.0, np.log2((resid_rms + floor) / floor))


def symbolic_regress(x, y, dictionary=None, max_terms=6, coef_bits=20, multiplicative=False):
    """Greedy forward selection gated by MDL. Returns (best Formula, info dict). Deterministic.

    multiplicative=True fits log(y) over a log-space basis, so the recovered law is a PRODUCT of power and
    exponential factors -- the log-transform that turns multiplicative structure additive (the prime-factor
    insight: x becomes + in the right basis). Requires y > 0. Term selection runs in the fitted (log) space;
    the reported resid_rms is always measured in the ORIGINAL space, so additive and multiplicative fits are
    comparable.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(y)
    if multiplicative:
        if np.any(y <= 0):
            raise ValueError("multiplicative mode fits log(y) and needs y > 0")
        target = np.log(y)
        atoms = dictionary if dictionary is not None else log_dictionary()
    else:
        target = y
        atoms = dictionary if dictionary is not None else elementary_dictionary()
    data_std = target.std() + 1e-12
    cols = [_eval_atom(k, p, x) for (k, p) in atoms]
    D = len(atoms)

    def fit(indices):
        A = np.column_stack([cols[i] for i in indices] + [np.ones(n)])
        beta, *_ = np.linalg.lstsq(A, target, rcond=None)
        resid = target - A @ beta
        return beta[:-1], beta[-1], float(np.sqrt(np.mean(resid ** 2)))

    def make(intercept, idxs, coefs):
        return Formula(intercept, [(atoms[idxs[j]], float(coefs[j])) for j in range(len(idxs))],
                       log_space=multiplicative)

    chosen = []
    base_rms = float(np.sqrt(np.mean((target - target.mean()) ** 2)))
    base_f = make(target.mean(), [], [])
    best = (base_f, _resid_bits(base_rms, n, data_std) + base_f.model_bits(D, coef_bits))
    for _ in range(max_terms):
        cand = [i for i in range(D) if i not in chosen]
        if not cand:
            break                                # dictionary exhausted (e.g. small log basis)
        scored = sorted((fit(chosen + [i])[2], i) for i in cand)
        pick = scored[0][1]
        trial = chosen + [pick]
        coefs, intercept, rms = fit(trial)
        f = make(intercept, trial, coefs)
        mdl = _resid_bits(rms, n, data_std) + f.model_bits(D, coef_bits)
        if mdl < best[1]:                        # MDL gate: keep the term only if it shortens the code
            best = (f, mdl); chosen = trial
        else:
            break                                # no further term pays for itself -> stop (refusal on noise)
    f = best[0]
    return f, {"mdl_bits": best[1], "n_terms": len(f.terms), "dict_size": D,
               "resid_rms": float(np.sqrt(np.mean((y - f.generate(x)) ** 2))),   # ORIGINAL space, comparable
               "multiplicative": multiplicative}


def compress_signal(x, y, dictionary=None, max_terms=6, coef_bits=20, path=None, mode="additive"):
    """One call, end to end: decompose foreign data into a law and return it as a saved generative seed.

    mode in {"additive", "multiplicative", "auto"}. "auto" tries both (when y > 0) and keeps whichever fits
    the ORIGINAL signal better -- so a multiplicative law (a*x^p*exp(cx)) is caught by the log transform that
    the flat additive dictionary would miss. The returned Formula IS the recipe (call .generate to regenerate
    or extrapolate, .save/.load to persist); info['resid_rms'] is the residual a B5 coder would take.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)

    def run(mult):
        return symbolic_regress(x, y, dictionary=dictionary, max_terms=max_terms,
                                coef_bits=coef_bits, multiplicative=mult)

    if mode == "auto":
        # Model selection between the additive and multiplicative FAMILIES is genuinely hard when both
        # fit in-sample. Robust rule (honest heuristic): switch to multiplicative only if it is COMPETITIVE
        # in-sample (so we don't pick it when additive clearly fits better) AND generalizes better on a
        # held-out tail (so we do pick it for a true multiplicative law, where in-sample is a near-tie).
        fa, ia = run(False)
        if not np.all(y > 0):
            f, info = fa, ia; info["mode"] = "additive"     # y not positive -> multiplicative N/A
        else:
            fm, im = run(True)
            nfit = max(4, int(0.8 * len(y)))

            def heldout(mult):
                try:
                    fh, _ = symbolic_regress(x[:nfit], y[:nfit], dictionary=dictionary,
                                             max_terms=max_terms, coef_bits=coef_bits, multiplicative=mult)
                except ValueError:
                    return np.inf
                return float(np.sqrt(np.mean((y[nfit:] - fh.generate(x[nfit:])) ** 2)))

            competitive = im["resid_rms"] <= 1.1 * ia["resid_rms"]
            generalizes = heldout(True) < heldout(False)
            if competitive and generalizes:
                f, info = fm, im; info["mode"] = "multiplicative"
            else:
                f, info = fa, ia; info["mode"] = "additive"
    elif mode == "multiplicative":
        f, info = run(True); info["mode"] = "multiplicative"
    else:
        f, info = run(False); info["mode"] = "additive"
    info["compression_ratio"] = f.compression_ratio(len(y))
    if path is not None:
        f.save(path)
    return f, info


def full_fit(x, y, dictionary=None):
    """The un-gated max-fit baseline: use the WHOLE dictionary (no parsimony). Overfits by construction."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    atoms = dictionary if dictionary is not None else elementary_dictionary()
    A = np.column_stack([_eval_atom(k, p, x) for (k, p) in atoms] + [np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return Formula(beta[-1], [(atoms[i], float(beta[i])) for i in range(len(atoms))])
