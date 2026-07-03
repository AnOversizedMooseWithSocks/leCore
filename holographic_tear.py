"""holographic_tear.py -- #2 from the SIGGRAPH list: TEARING a thin sheet. Cloth/paper that RIPS when overstretched.

WHY THIS EXISTS (SIGGRAPH scouting list #2 -- the "ripping" capability, a genuine gap)
-------------------------------------------------------------------------------------
The engine had no fracture. The canonical graphics way to tear a thin sheet (Pfaff/Narain/O'Brien) tracks the
stress in the sheet and splits it where the stress exceeds the material's strength. Our sheet is already a
position-based-dynamics cloth (`holographic_softbody`): a grid of particles linked by DISTANCE CONSTRAINTS
(springs). So the readable, on-substrate way to tear it is a BREAKABLE-CONSTRAINT model: give each link a tear
strength (a maximum strain), and when a link is stretched past it, it SNAPS -- remove it. Enough snaps and the
sheet separates into pieces. No half-edge surgery, deterministic, and it reuses the PBD solver we already have;
the material tear strength is a new column on the material data layer we built.

THE MODEL (readable)
--------------------
Each distance constraint links particles i,j at a rest length L. Its STRAIN is how far it is stretched beyond
rest: strain = |x_i - x_j| / L - 1. After each PBD step we check every link: if its strain exceeds the tear
strength, it has torn -- drop it from the constraint list. The crack advances because once a link goes, its
neighbours carry more load and reach their own tear strain next -- a propagating tear front, the same physics
as the reference method, expressed on the constraint graph instead of by remeshing. Which particles are still
linked (connected components of the surviving graph) tells you how many PIECES the sheet is now in.

HONEST SCOPE (kept negative): a mass-spring / breakable-constraint tear (the readable baseline), NOT the full
adaptive REMESHING of Pfaff et al. -- the crack follows the existing grid links, so it is as sharp as the mesh
resolution (a finer grid tears more smoothly), and torn edges are not re-triangulated into new geometry. Sits
on the PBD softbody (XPBD compliance lets the sheet stretch enough to reach a tear strain). Deterministic;
NumPy + stdlib. Tear strengths are plausible, art-directable data.
"""
import numpy as np

# Tear strength = the strain a material's links tolerate before snapping (a new material data column). Paper
# tears at a small stretch; a knit or rubber sheet stretches far first. Plausible/art-directable values.
TEAR_STRENGTH = {
    "paper": 0.06, "wet_paper": 0.03, "foil": 0.10, "cotton": 0.18, "denim": 0.25,
    "leather": 0.35, "knit": 0.45, "rubber": 0.80,
}


def tear_strength(material):
    """The tear strain of a named material (fraction of stretch before a link snaps), or a paper-ish default."""
    return float(TEAR_STRENGTH.get(material, 0.08))


class TearableCloth:
    """A rectangular cloth whose links SNAP when overstretched. Build it, pin some edge, pull or load it, and step
    -- links that exceed the tear strain break, and the sheet separates. Wraps a PBD SoftBody (reused, not
    reimplemented)."""

    def __init__(self, rows=12, cols=12, spacing=1.0, compliance=2e-3, material="paper",
                 tear_strain=None, pin="top"):
        from holographic_softbody import SoftBody
        self.rows, self.cols = int(rows), int(cols)
        self.body = SoftBody.cloth(self.rows, self.cols, spacing=spacing, compliance=compliance)
        self.tear_strain = float(tear_strain) if tear_strain is not None else tear_strength(material)
        self.material = material
        self.torn = 0
        # by default cloth() pins the top row; allow a couple of common grips for tearing tests
        if pin == "top_corners":
            self.body.w[:] = 1.0
            self.body.w[self._idx(0, 0)] = 0.0
            self.body.w[self._idx(0, self.cols - 1)] = 0.0
        elif pin == "none":
            self.body.w[:] = 1.0

    def _idx(self, r, c):
        return r * self.cols + c

    def n_constraints(self):
        return len(self.body.constraints)

    def _tear(self):
        """Drop every link stretched past the tear strain. Returns how many snapped this step."""
        x = self.body.x
        kept = []
        snapped = 0
        for (i, j, rest, comp) in self.body.constraints:
            strain = np.linalg.norm(x[i] - x[j]) / rest - 1.0
            if strain > self.tear_strain:
                snapped += 1                                       # this link tore
            else:
                kept.append((i, j, rest, comp))
        self.body.constraints = kept
        self.torn += snapped
        return snapped

    def step(self, pull=None, pull_rows="bottom", dt=1.0 / 60.0, gravity=None, iterations=20, damping=0.02):
        """Advance one frame and then tear. `pull` (an (D,) force vector) is applied to a grip of particles
        (`pull_rows='bottom'` by default) to yank the sheet; `gravity` also loads it. Links that overstretch snap."""
        ext = None
        if pull is not None:
            ext = np.zeros_like(self.body.x)
            if pull_rows == "bottom":
                grip = [self._idx(self.rows - 1, c) for c in range(self.cols)]
            else:
                grip = list(pull_rows)
            ext[grip] = np.asarray(pull, float)
        self.body.step(dt=dt, gravity=gravity, iterations=iterations, external_force=ext, damping=damping)
        return self._tear()

    def connected_components(self):
        """How many separate PIECES the sheet is in now = connected components of the surviving link graph
        (union-find over the particles). One intact sheet is 1; a full tear gives 2+."""
        n = self.rows * self.cols
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a

        for (i, j, _r, _c) in self.body.constraints:
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[ra] = rb
        roots = {find(a) for a in range(n)}
        return len(roots)

    def piece_sizes(self):
        """The particle count of each connected piece, largest first -- so you can see the sheet split in two."""
        n = self.rows * self.cols
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a

        for (i, j, _r, _c) in self.body.constraints:
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[ra] = rb
        sizes = {}
        for a in range(n):
            r = find(a); sizes[r] = sizes.get(r, 0) + 1
        return sorted(sizes.values(), reverse=True)


def _selftest():
    """A pulled sheet snaps its links and separates into pieces; a stronger material tears less under the same
    pull; a gentle load does not tear; deterministic."""
    # (1) yank a weak (paper) sheet hard: links snap and it separates into more than one piece
    cloth = TearableCloth(rows=12, cols=12, material="paper", compliance=3e-3)
    n0 = cloth.n_constraints()
    assert cloth.connected_components() == 1                       # starts as one intact sheet
    for _ in range(120):
        cloth.step(pull=(0.0, -1200.0), gravity=(0.0, -9.8))       # grab the bottom edge and pull down hard
    assert cloth.torn > 0 and cloth.n_constraints() < n0           # links tore
    assert cloth.connected_components() > 1                        # the sheet came apart

    # (2) a STRONGER material (rubber, high tear strain) tears less under the SAME pull
    tough = TearableCloth(rows=12, cols=12, material="rubber", compliance=3e-3)
    for _ in range(120):
        tough.step(pull=(0.0, -1200.0), gravity=(0.0, -9.8))
    assert tough.torn < cloth.torn                                 # rubber stretches instead of ripping
    assert tear_strength("rubber") > tear_strength("paper")

    # (3) a GENTLE load does not tear anything
    calm = TearableCloth(rows=10, cols=10, material="cotton", compliance=1e-3)
    for _ in range(60):
        calm.step(gravity=(0.0, -0.5))                             # just hanging under light gravity
    assert calm.torn == 0 and calm.connected_components() == 1

    # (4) piece sizes: after a full tear the sheet is in two sizable pieces (not just chipped corners)
    big = TearableCloth(rows=14, cols=14, material="wet_paper", compliance=4e-3)
    for _ in range(150):
        big.step(pull=(0.0, -1500.0), gravity=(0.0, -9.8))
    sizes = big.piece_sizes()
    assert len(sizes) >= 2 and sizes[1] >= 5                       # a real split, second piece is substantial

    # (5) deterministic
    a = TearableCloth(rows=8, cols=8, material="paper", compliance=3e-3)
    b = TearableCloth(rows=8, cols=8, material="paper", compliance=3e-3)
    for _ in range(50):
        a.step(pull=(0.0, -1000.0), gravity=(0.0, -9.8)); b.step(pull=(0.0, -1000.0), gravity=(0.0, -9.8))
    assert a.torn == b.torn and np.array_equal(a.body.x, b.body.x)
    print("holographic_tear selftest OK: a yanked paper sheet snapped %d links and split into %d pieces; rubber "
          "tore only %d under the same pull; a gentle load tore nothing; deterministic"
          % (cloth.torn, cloth.connected_components(), tough.torn))


if __name__ == "__main__":
    _selftest()
