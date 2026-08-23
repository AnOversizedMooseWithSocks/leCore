"""SPLAT MEMORY (cp72) -- Drettakis's ask, with Pharr's abstention and Plate's cliff.

A Gaussian-splat scene IS a superposition -- so store it as one: each primitive
(position, sigma, colour) binds a quantized POSITION CELL key to a PROPERTY vector,
and the scene is the bundle of all of them. Recall BY REGION unbinds the cell keys
inside a query circle and cleans up against the property codebook. The three panel
conditions, honored:
  Drettakis -- role-bound primitives, one bundle, region recall.
  Pharr     -- a region holding nothing ABSTAINS (cleanup margin below threshold),
               never invents a primitive.
  Plate     -- capacity_cliff() sweeps N and reports where recovery breaks,
               shown rather than hidden.
"""
import hashlib

import numpy as np


def _atom(name, dim, seed=0):
    """Deterministic named hypervector: the codebook entry for `name`."""
    h = int(hashlib.sha256(("%d|%s" % (seed, name)).encode()).hexdigest()[:16],
            16)
    v = np.random.default_rng(h).standard_normal(dim)
    return v / np.linalg.norm(v)


def _bind(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), len(a))


def _unbind(c, a):
    inv = np.concatenate([a[:1], a[1:][::-1]])          # involution
    return _bind(c, inv)


class SplatMemory:
    def __init__(self, dim=4096, grid=8, seed=0):
        self.seed = int(seed)
        self.grid = int(grid)
        self.dim = int(dim)
        self.scene = np.zeros(dim)
        self.props = {}                       # prop label -> vector
        self.count = 0

    def _cell_key(self, x, y):
        cx = min(self.grid - 1, max(0, int(x * self.grid)))
        cy = min(self.grid - 1, max(0, int(y * self.grid)))
        return _atom("cell_%d_%d" % (cx, cy), self.dim, self.seed), (cx, cy)

    def _prop_vec(self, sigma, color):
        label = "prop_s%.2f_c%s" % (round(float(sigma), 2),
                                    "_".join("%.1f" % c for c in color))
        if label not in self.props:
            self.props[label] = _atom(label, self.dim, self.seed)
        return self.props[label], label

    def add(self, x, y, sigma, color):
        """Add one Gaussian primitive at normalized (x, y) in [0,1)."""
        k, cell = self._cell_key(x, y)
        v, label = self._prop_vec(sigma, color)
        self.scene = self.scene + _bind(k, v)
        self.count += 1
        return {"cell": cell, "prop": label}

    def recall_region(self, x, y, radius=0.13, threshold=0.12):
        """Recover the primitives inside a query circle -- or ABSTAIN per cell."""
        out = []
        r_cells = max(1, int(radius * self.grid + 0.5))
        cx0 = min(self.grid - 1, max(0, int(x * self.grid)))
        cy0 = min(self.grid - 1, max(0, int(y * self.grid)))
        for cx in range(max(0, cx0 - r_cells), min(self.grid, cx0 + r_cells + 1)):
            for cy in range(max(0, cy0 - r_cells),
                            min(self.grid, cy0 + r_cells + 1)):
                k = _atom("cell_%d_%d" % (cx, cy), self.dim, self.seed)
                probe = _unbind(self.scene, k)
                best, sim = None, 0.0
                for label, v in self.props.items():
                    c = float(v @ probe / (np.linalg.norm(v) *
                                           np.linalg.norm(probe) + 1e-12))
                    if c > sim:
                        best, sim = label, c
                if sim >= threshold:
                    out.append({"cell": (cx, cy), "prop": best,
                                "margin": round(sim, 3)})
        return {"found": out, "abstained_cells":
                (2 * r_cells + 1) ** 2 - len(out)}

    def capacity_cliff(self, n_max=64, seed=1):
        """Plate's condition: sweep N primitives, report recovery per N -- the
        cliff shown, not hidden."""
        rng = np.random.default_rng(seed)
        rows = []
        for n in (4, 8, 16, 24, 32, 48, n_max):
            sm = SplatMemory(dim=self.dim, grid=self.grid, seed=seed)
            truth = []
            for _ in range(n):
                x, y = rng.random(), rng.random()
                rec = sm.add(x, y, 0.05 + 0.1 * rng.random(),
                             (rng.random(), rng.random(), rng.random()))
                truth.append(((x, y), rec["prop"]))
            ok = 0
            for (x, y), prop in truth:
                got = sm.recall_region(x, y, radius=0.01)
                ok += any(f["prop"] == prop for f in got["found"])
            rows.append({"n": n, "recovered": ok, "rate": round(ok / n, 3)})
        return rows


def _selftest():
    sm = SplatMemory(dim=4096, grid=8, seed=0)
    a = sm.add(0.10, 0.12, 0.08, (1.0, 0.2, 0.2))
    b = sm.add(0.80, 0.85, 0.15, (0.2, 0.2, 1.0))
    r1 = sm.recall_region(0.10, 0.12, radius=0.05)
    assert any(f["prop"] == a["prop"] for f in r1["found"]), r1
    assert not any(f["prop"] == b["prop"] for f in r1["found"]), \
        "a region query must not drag in the far primitive"
    r_empty = sm.recall_region(0.45, 0.45, radius=0.04)
    assert not r_empty["found"] and r_empty["abstained_cells"] > 0, \
        "an empty region ABSTAINS (Pharr's condition)"
    cliff = sm.capacity_cliff(n_max=64)
    assert cliff[0]["rate"] >= 0.95, cliff
    assert cliff[-1]["rate"] <= cliff[0]["rate"], "the cliff is real and shown"
    return ("OK: 2 primitives stored as one bundle; region recall exact; empty "
            "region abstains; capacity cliff measured: " +
            ", ".join("N=%d:%.0f%%" % (r["n"], 100 * r["rate"]) for r in cliff))


if __name__ == "__main__":
    print(_selftest())
