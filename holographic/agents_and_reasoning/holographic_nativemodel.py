"""holographic_nativemodel.py -- F28 FIRST LANDING: the BAKED native micro-model.

The F27 conformance test proved the sentence "same program, same answer, different substrate" with
the installed chain still living in Python closures. This module takes the next step the plan
names: the model IS the VSA application. No pretrained host, no vminstall shims around someone
else's attention -- a from-scratch micro-model whose LAYERS are the certified parameterizations
(circulant = D floats, permutation = D ints, dense = D^2 -- exactly what the projector emits),
whose RECURRENT STATE is the register file, and whose forward pass IS the compiled program chain
stepped by a token loop.

BAKED, NOT TRAINED (the plan's first route; the trained-in-ecosystem route is the stacc-side
follow-up): every weight is a deterministic function of (seed, program). Which yields the
demoscene move at the model level -- "store the rule, not the bytes": the MODEL FILE is a few
hundred bytes of {dim, seed, program}, and load() REGENERATES the full weight set bit-identically
through the same bake (machine atoms -> projector -> parameterizations). A 64-parameter save for
a model whose dense export would be megabytes, because the weights were never information --
the program was.

to_dense() exports any layer as the literal matrix a host framework would install -- the bridge
to real weight surgery stays one call wide, and the manifest (F26) travels with the model as its
discoverability sidecar.

SUBSTRATE WALLS RESPECTED (the theorems survive the architecture): state SNR still degrades as
1/sqrt(n) under bundling, float32 write-accumulation cliffs still exist, and nothing here claims
otherwise -- this model executes EXACT-tagged linear programs, which is precisely the class the
projector certifies. The nonlinear shell stays runtime; the refusal is inherited, not re-derived.
"""
import json
import numpy as np


class NativeHoloModel:
    """A micro-model whose layers are certified installed ops and whose forward pass is the program."""

    def __init__(self, dim, seed, program, symbolic_functions=None, data=None, unitary=False):
        self.dim = int(dim)
        self.seed = int(seed)
        self.program = [tuple(p) for p in program]
        self.symbolic_functions = {k: [tuple(x) for x in v] for k, v in (symbolic_functions or {}).items()}
        self.data = list(data) if data is not None else None
        # unitary=True (AUDIT FIX for the depth wall, measured): data atoms baked with
        # |spectrum| = 1 per bin, so every bind is norm-preserving and deep chains stay
        # conditioned -- depth-256 error 7.8e82 (default atoms) -> 6.4e-15 (unitary), norm 1.0 to
        # the last bit, amplification bound ~exp(0), no manifest warning. Default OFF: unitary
        # atoms are a DIFFERENT codebook (existing baked models must not shift), and shallow
        # programs never hit the wall. Deep/REPEAT-heavy programs should turn it on -- the
        # manifest's conditioning warning names this exact switch.
        self.unitary = bool(unitary)
        self._bake()

    def _bake(self):
        """Deterministic weight bake: machine atoms from (seed) -> projector-certified layers.
        Same (dim, seed, program) => bit-identical weights, on any machine, any day -- the model
        file stores the RULE and this method is the rule's evaluator."""
        from holographic.agents_and_reasoning.holographic_machine import HoloMachine
        from holographic.agents_and_reasoning.holographic_compileinstall import compile_installed
        names = self.data
        if names is None:
            names = sorted({a for _, a in self.program if isinstance(a, str) and a not in
                            ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7")}
                           | {a for b in self.symbolic_functions.values() for _, a in b if isinstance(a, str)})
        self._machine = HoloMachine(dim=self.dim, seed=self.seed, data=names)
        if self.unitary:
            # one consistent override: VM, compiler and symbolic referee all read data_atoms,
            # so conformance is preserved BY CONSTRUCTION (same dict, three substrates).
            self._machine.data_atoms = {n: self._machine._atom("dat:%s" % n, unitary=True)
                                        for n in self._machine.data_atoms}
        for fn, body in self.symbolic_functions.items():
            self._machine.define(fn, list(body) + [("HALT", None)])
        self._machine.functions_symbolic = self.symbolic_functions
        self._run, self.manifest = compile_installed(self._machine, self.program)

    def forward(self):
        """The forward pass IS the program: the token loop steps the compiled chain, threading
        (state, register file). Returns the final state vector."""
        return self._run()

    def layers(self):
        """[(name, kind, n_params)] -- the model card. Circulant layers cost D floats, permutation
        layers D ints, dense layers D^2: the parameterizations ARE the projector's taxonomy."""
        out = []
        for name, pr in self.manifest["ops"].items():
            kind = pr["kind"]
            n = {"circulant": self.dim, "permutation": self.dim,
                 "dense": self.dim * self.dim}.get(kind, 0)
            out.append((name, kind, n))
        return out

    def to_dense(self, op_name):
        """Export one layer as the literal (dim, dim) matrix a host framework would install --
        the one-call bridge to real weight surgery. Circulants and permutations EXPAND here
        (that is the point of not storing them this way)."""
        pr = self.manifest["ops"][op_name]
        if pr["kind"] == "dense":
            return pr["matrix"].copy()
        if pr["kind"] == "permutation":
            M = np.zeros((self.dim, self.dim))
            M[pr["perm"], np.arange(self.dim)] = 1.0
            return M
        if pr["kind"] == "circulant":
            return np.stack([np.roll(pr["column"], i) for i in range(self.dim)], axis=1)
        raise ValueError("no dense form for kind %r" % pr["kind"])

    def save(self, path):
        """The model file: {dim, seed, program, functions, data} -- the RULE, a few hundred bytes.
        No weights are written; load() re-bakes them bit-identically. (Contrast: the dense export
        of even this toy's layers is megabytes.)"""
        with open(path, "w") as f:
            json.dump({"dim": self.dim, "seed": self.seed, "unitary": self.unitary,
                       "program": [list(p) for p in self.program],
                       "functions": {k: [list(x) for x in v] for k, v in self.symbolic_functions.items()},
                       "data": self.data}, f)
        return path

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["dim"], d["seed"], [tuple(p) for p in d["program"]],
                   {k: [tuple(x) for x in v] for k, v in d.get("functions", {}).items()},
                   data=d.get("data"), unitary=d.get("unitary", False))


class ModelLibrary:
    """G14 -- MANY PROGRAMS, ONE RULE FILE: a function library whose members share one machine
    (same dim/seed/atoms), so certified operators are SHARED BY CONSTRUCTION -- the same BIND:k
    payload (same sha256) serves every program that uses it. save() writes one small JSON of
    {dim, seed, programs, functions, data, unitary}; load() re-bakes EVERY member bit-identically.
    The 258-byte pattern, plural: a whole library still costs less than one dense row."""

    def __init__(self, dim, seed, programs, symbolic_functions=None, data=None, unitary=False):
        self.programs = {k: [tuple(x) for x in v] for k, v in programs.items()}
        self._members = {}
        for name, prog in self.programs.items():
            self._members[name] = NativeHoloModel(dim, seed, prog, symbolic_functions,
                                                  data=data, unitary=unitary)
        first = next(iter(self._members.values()))
        self.dim, self.seed, self.unitary = first.dim, first.seed, first.unitary
        self.symbolic_functions, self.data = first.symbolic_functions, first.data

    def forward(self, name, *a, **k):
        return self._members[name].forward(*a, **k)

    def manifest(self, name):
        return self._members[name].manifest

    def save(self, path):
        import json
        with open(path, "w") as fh:
            json.dump({"dim": self.dim, "seed": self.seed, "unitary": self.unitary,
                       "programs": {k: [list(p) for p in v] for k, v in self.programs.items()},
                       "functions": {k: [list(x) for x in v] for k, v in self.symbolic_functions.items()},
                       "data": self.data}, fh)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as fh:
            d = json.load(fh)
        return cls(d["dim"], d["seed"], {k: [tuple(p) for p in v] for k, v in d["programs"].items()},
                   {k: [tuple(x) for x in v] for k, v in d.get("functions", {}).items()},
                   data=d.get("data"), unitary=d.get("unitary", False))


def _selftest():
    import os, tempfile
    prog = [("LOAD", "a"), ("REPEAT", 3), ("CALL", "twist"),
            ("STORE", "R1"), ("LOAD", "b"), ("BIND", "k2"), ("RECALL", "R1"), ("HALT", None)]
    model = NativeHoloModel(dim=1024, seed=7, program=prog,
                            symbolic_functions={"twist": [("BIND", "k")]},
                            data=["a", "b", "k", "k2"])

    # planted truth A: the model's forward pass equals the VM running the same program (F27 carried up)
    y = model.forward()
    vm, _ = model._machine.run(model._machine.assemble(prog))
    assert np.allclose(y, vm, atol=1e-6), "forward != VM"

    # planted truth B: the model card shows the parameterizations, and REPEAT is one layer
    card = dict((n, (k, p)) for n, k, p in model.layers())
    assert card["BODY:twist"][0] == "circulant" and card["BODY:twist"][1] == 1024, card

    # planted truth C: rule-not-bytes -- save is tiny, load re-bakes, forward is BIT-identical
    fp = os.path.join(tempfile.gettempdir(), "native_model.json")
    model.save(fp)
    size = os.path.getsize(fp)
    dense_bytes = sum(p for _, k, p in model.layers() if k == "dense") * 8 + 3 * 1024 * 8
    model2 = NativeHoloModel.load(fp)
    y2 = model2.forward()
    assert np.array_equal(y, y2), "re-baked forward must be BIT-identical (same seed, same rule)"
    assert size < 1000, "the model file must stay rule-sized (got %d bytes)" % size

    # planted truth D: the export bridge -- to_dense of the powered body equals three live binds
    from holographic.agents_and_reasoning.holographic_ai import bind
    M = model.to_dense("BODY:twist^3")
    a = model._machine.data_atoms["a"]; k = model._machine.data_atoms["k"]
    truth = bind(bind(bind(a, k), k), k)
    assert np.allclose(M @ a, truth, atol=1e-8), "dense export of the power layer must match live math"

    # UNITARY-BAKE PINS (the depth wall killed at the source, measured): (a) a depth-96 unitary
    # model matches the VM AND keeps unit norm to 1e-12 (default atoms explode by depth 64);
    # (b) the flag survives save/load with a bit-identical re-bake; (c) default stays bit-stable
    # (unitary=False -> the original atoms; asserted above by pin C already).
    deep_prog = [("LOAD", "a")] + [("BIND", "k")] * 96 + [("HALT", None)]
    mu = NativeHoloModel(dim=512, seed=3, program=deep_prog, data=["a", "k"], unitary=True)
    yu = mu.forward()
    from holographic.agents_and_reasoning.holographic_compileinstall import symbolic_run
    assert np.allclose(yu, symbolic_run(mu._machine, deep_prog), atol=1e-9)
    assert abs(float(np.linalg.norm(yu)) - 1.0) < 1e-10, "unitary chain must preserve norm"
    assert not mu.manifest.get("warnings"), "unitary chain must carry no amplification warning"
    fpu = os.path.join(tempfile.gettempdir(), "native_unitary.json")
    mu.save(fpu)
    assert np.array_equal(NativeHoloModel.load(fpu).forward(), yu), "unitary flag must survive save/load"

    # G14 PINS: (a) two programs in ONE library share the certified op BY CONSTRUCTION -- the
    # BIND:k payload sha256 is IDENTICAL in both manifests; (b) one rule file re-bakes BOTH
    # members bit-identically; (c) the file stays rule-sized.
    lib = ModelLibrary(512, 7, {"twist": [("LOAD", "a"), ("BIND", "k"), ("HALT", None)],
                                "twist2": [("LOAD", "b"), ("BIND", "k"), ("BIND", "k"), ("HALT", None)]},
                       data=["a", "b", "k"])
    ya, yb = lib.forward("twist"), lib.forward("twist2")
    # in-memory manifests carry the raw payload (sha256 is save_manifest's job): compare the
    # payloads bit-for-bit -- stronger than hash equality
    ca = lib.manifest("twist")["ops"]["BIND:k"]["column"]
    cb = lib.manifest("twist2")["ops"]["BIND:k"]["column"]
    assert np.array_equal(ca, cb), "shared machine must mean shared certified payloads"
    fpl = os.path.join(tempfile.gettempdir(), "native_lib.json")
    lib.save(fpl)
    lib2 = ModelLibrary.load(fpl)
    assert np.array_equal(lib2.forward("twist"), ya) and np.array_equal(lib2.forward("twist2"), yb)
    assert os.path.getsize(fpl) < 600, os.path.getsize(fpl)

    print("OK: holographic_nativemodel self-test passed (forward == VM; REPEAT is one circulant "
          "layer of D params; %d-byte model file re-bakes bit-identical weights; dense export of "
          "the power layer matches three live binds)" % size)


if __name__ == "__main__":
    _selftest()
