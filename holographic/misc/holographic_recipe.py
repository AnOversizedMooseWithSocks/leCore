"""A generative recipe-store for constructed holostuff structures.

WHY THIS EXISTS
---------------
The panel's "proven structure has no noise" result: a structure BUILT by a deterministic proof (a
derivation over a seed) carries no noise, so it serialises to its *generator* losslessly -- you store the
recipe, not the expanded vectors, and replaying the recipe reproduces the structure BIT-FOR-BIT. This is
the easy, exact half of generative compression: when we are the builder we already hold the proof, so
there is nothing to search for and no residual to code.

A `StructureRecipe` is a tiny replayable build-graph. Each op produces one result vector from a seed and
earlier results: `atom` (a derived atom -- regenerated from the seed, never stored), `bind`, `bundle`,
`permute`, `normalize`. You build your structure THROUGH the recipe, so you get both the vectors and the
recipe that regenerates them. Serialising stores only the op list (and the seed) -- a few hundred bytes
that regenerate megabytes of structure, exactly.

THE ESCAPE HATCH AND THE KEPT NEGATIVE
The `raw` op stores a literal vector verbatim. It is the honest boundary: data that was NOT constructed
(a measured or random vector) has no short recipe, so it must be stored as-is and gets no compression.
The recipe's compression ratio is therefore exactly the *constructed fraction* of the structure -- all
recipe -> enormous ratio; all raw -> ~1x. That is the constructed-vs-measured partition made literal.

THE CAPACITY-CLIFF POINT
Reading structure back out of a single bounded encoded vector degrades past the capacity cliff (crosstalk).
A recipe does not: it names its leaves explicitly and replays the construction, so a deeply nested
structure is recovered EXACTLY at any depth. The recipe is the right store for deep constructed structure;
the expanded superposition is bounded.

Pure NumPy + holostuff kernel, deterministic, JSON serialisation (readable), no new dependencies.
"""

import base64
import json
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, bundle, permute, derived_atom


class StructureRecipe:
    """A replayable build-graph: store the deterministic ops that built a structure, not the vectors."""

    def __init__(self, dim, seed=0):
        self.dim = int(dim)
        self.seed = int(seed)
        self._ops = []         # each op produces 1+ results; a handle is an absolute result index
        self._raws = []        # literal payloads for 'raw' ops (the non-constructed escape hatch)
        self._outputs = []     # handles the caller cares about (for size accounting)
        self._n_results = 0    # running result count (handles are assigned from this, not op position)

    # ---- builder methods. A handle is an absolute RESULT index. Most ops produce one result;
    #      `repeat` produces many, so result indices are tracked separately from op positions. ----
    def _emit(self, op, n_results=1):
        start = self._n_results
        self._ops.append(op)
        self._n_results += n_results
        return start if n_results == 1 else list(range(start, start + n_results))

    def atom(self, name, unitary=False):
        return self._emit(("atom", str(name), bool(unitary)))

    def raw(self, vector):
        """Store a literal vector verbatim -- for data that was NOT constructed (no short recipe exists)."""
        idx = len(self._raws)
        self._raws.append(np.asarray(vector, float).copy())
        return self._emit(("raw", idx))

    def bind(self, a, b):
        return self._emit(("bind", int(a), int(b)))

    def bundle(self, members):
        return self._emit(("bundle", [int(m) for m in members]))

    def superpose(self, members):
        """Un-normalized superposition (plain sum). The counterpart to bundle, which renormalizes:
        some compositions (e.g. UnifiedMind.compose_nested) superpose with raw np.sum, so the unified
        structure type needs both to reproduce them bit-exactly."""
        return self._emit(("superpose", [int(m) for m in members]))

    def permute(self, a, shift):
        return self._emit(("permute", int(a), int(shift)))

    def normalize(self, a):
        return self._emit(("normalize", int(a)))

    # ---- the macro: a parameterised iteration captured as ONE op (regular structure -> tiny recipe) ----
    def repeat(self, count, template):
        """Run a parameterised sub-recipe `count` times; produce `count` results (one per iteration).

        `template` is a list of abstract ops with refs LOCAL to one iteration (indices 0..). Forms:
          ("atom", name_with_{i}, unitary)   -- {i} is replaced by the iteration index
          ("bind", a, b) / ("bundle", [refs]) / ("normalize", a)
          ("permute", a, shift)              -- shift is an int, or the string "i" (the loop index)
        The iteration's OUTPUT is its last template result. This is how a regular structure (a codebook,
        a positional sequence) collapses from N explicit ops to one -- reaching the large ratios.
        """
        return self._emit(("repeat", int(count), [list(t) for t in template]), n_results=int(count))

    def atom_range(self, prefix, count, unitary=False):
        """Convenience: `count` derived atoms named prefix+i, as ONE op. Returns a list of handles."""
        return self.repeat(count, [("atom", prefix + "{i}", unitary)])

    def mark_output(self, handle):
        self._outputs.append(int(handle))
        return handle

    # ---- execution: deterministic, so replay is bit-exact ----
    def _run_template(self, template, i, res):
        """Execute one iteration of a repeat template; return its output vector (last template result)."""
        loc = []
        for t in template:
            kind = t[0]
            if kind == "atom":
                loc.append(derived_atom(self.seed, t[1].format(i=i), self.dim, unitary=t[2]))
            elif kind == "bind":
                loc.append(bind(loc[t[1]], loc[t[2]]))
            elif kind == "bundle":
                loc.append(bundle([loc[m] for m in t[1]]))
            elif kind == "permute":
                loc.append(permute(loc[t[1]], i if t[2] == "i" else int(t[2])))
            elif kind == "normalize":
                v = loc[t[1]]; n = np.linalg.norm(v); loc.append(v / n if n > 0 else v)
            else:
                raise ValueError(f"unknown template op {kind}")
        return loc[-1]

    def build(self):
        """Materialise every result vector in order. Returns a list indexed by handle."""
        res = []
        for op in self._ops:
            kind = op[0]
            if kind == "atom":
                res.append(derived_atom(self.seed, op[1], self.dim, unitary=op[2]))
            elif kind == "raw":
                res.append(self._raws[op[1]].copy())
            elif kind == "bind":
                res.append(bind(res[op[1]], res[op[2]]))
            elif kind == "bundle":
                res.append(bundle([res[m] for m in op[1]]))
            elif kind == "superpose":
                res.append(np.sum([res[m] for m in op[1]], axis=0))   # plain sum, no renormalization
            elif kind == "permute":
                res.append(permute(res[op[1]], op[2]))
            elif kind == "normalize":
                v = res[op[1]]
                n = np.linalg.norm(v)
                res.append(v / n if n > 0 else v)
            elif kind == "repeat":
                count, template = op[1], op[2]
                for i in range(count):
                    res.append(self._run_template(template, i, res))
            else:
                raise ValueError(f"unknown op {kind}")
        return res

    def get(self, handle):
        return self.build()[int(handle)]

    def outputs(self):
        built = self.build()
        handles = self._outputs if self._outputs else list(range(self._n_results))
        return [built[h] for h in handles]

    # ---- serialisation: store only seed + ops + raw payloads ----
    # Constructed ops (atom/bind/bundle/permute/normalize) replay BIT-EXACT -- they are regenerated
    # from the seed at full precision. `raw` payloads (the non-constructed escape hatch) are stored as
    # binary float32: that is the measured/lossy regime where a residual coder (B5) belongs, and it keeps
    # the size accounting honest (a raw vector costs ~ what it would cost stored expanded, i.e. ~1x).
    def to_dict(self):
        raws_b64 = [base64.b64encode(r.astype(np.float32).tobytes()).decode("ascii") for r in self._raws]
        return {"dim": self.dim, "seed": self.seed, "ops": self._ops,
                "raws": raws_b64, "outputs": self._outputs}

    @classmethod
    def from_dict(cls, d):
        r = cls(d["dim"], d["seed"])
        r._ops = [tuple(op) if op[0] != "bundle" else ("bundle", list(op[1])) for op in d["ops"]]
        r._raws = [np.frombuffer(base64.b64decode(x), dtype=np.float32).astype(float) for x in d.get("raws", [])]
        r._outputs = list(d.get("outputs", []))
        r._n_results = sum(op[1] if op[0] == "repeat" else 1 for op in r._ops)   # repeat emits many
        return r

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ---- accounting: what the recipe saves you ----
    def recipe_bytes(self):
        """Efficient store size of the generator: the ops list plus raw payloads as binary float32."""
        ops_bytes = len(json.dumps({"dim": self.dim, "seed": self.seed,
                                    "ops": self._ops, "outputs": self._outputs}))
        raw_bytes = sum(r.size * 4 for r in self._raws)
        return ops_bytes + raw_bytes

    def expanded_bytes(self, dtype_bytes=4):
        """Bytes to store the OUTPUT vectors expanded (default float32)."""
        n = len(self._outputs) if self._outputs else self._n_results
        return n * self.dim * dtype_bytes

    def compression_ratio(self, dtype_bytes=4):
        rb = self.recipe_bytes()
        return self.expanded_bytes(dtype_bytes) / rb if rb else float("inf")
