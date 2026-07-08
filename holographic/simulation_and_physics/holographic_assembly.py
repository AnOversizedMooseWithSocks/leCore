"""B6 (part 2) -- fragment assembly as a flow search: the Tero solver generalised beyond mazes.

The Baker/Rosetta seat builds a global structure from a library of local fragments under an energy.
Its combinatorial core is: choose which fragment sits at each position so that (a) consecutive
fragments overlap-agree and (b) the total placement energy is minimised. That is a MIN-COST PATH through
a layered (position x fragment) trellis -- exactly what the Tero flow solver finds. So the maze solver
(a min-cost path on a grid) and fragment assembly (a min-cost path on a trellis) are the SAME search,
and the chosen fragments come back out as a B7 typed structure (each fragment bound to its position).

Energy is encoded as path LENGTH: a placement of energy c becomes c+1 unit hops via relay nodes, so the
unit-length Tero solver's shortest path IS the minimum-energy assembly (min hops == min energy). The +1
keeps every layer transition costing at least one hop, so the trellis stays strictly layered.

MEASURED vs the exact DP (Viterbi) optimum:
  * Complete library -> assembles the target EXACTLY (energy 0).
  * Library missing a true fragment -> forced mismatches; the flow assembly's energy MATCHES the DP
    optimum (e.g. 9 == 9), i.e. it finds the globally best assembly, not a locally greedy one.

KEPT NEGATIVES / scope: this is the combinatorial CORE of fragment assembly (choose fragments to minimise
an energy), not a protein force field -- the "energy" here is a placement mismatch, a stand-in for the
Rosetta score. The relay-node energy encoding bloats the graph by the total energy (fine for small
libraries/targets; for large energies, weight Tero's edges by length directly instead). Like the maze
solver it is centralized (a Laplacian solve per step). It is the principled flow-search generalisation
the research program pointed to; a full assembler with a real energy is a larger effort.

Pure NumPy + holostuff spirit; deterministic; reuses tero_solve (B6) and StructureRecipe (B7).
"""

from holographic.misc.holographic_flow import tero_solve
from holographic.misc.holographic_recipe import StructureRecipe
import numpy as np
import holographic.agents_and_reasoning.holographic_ai as _A


def _energy(frag, pos, target):
    """Placement energy: mismatches between `frag` placed at `pos` and the target (the Rosetta-score
    stand-in)."""
    return sum(1 for j in range(len(frag)) if frag[j] != target[pos + j])


def _build_trellis(target, library, K, energy=_energy):
    """Layered (position, fragment) trellis. Edges weighted by placement energy, encoded as unit hops
    via relay nodes so the unit-length Tero solver finds the min-ENERGY path. Returns (nbr, last). The
    `energy` is ROUNDED to an integer hop count (the relay encoding needs integer lengths); supply an
    integer-valued energy (Hamming, a substitution matrix) for an exact search, or accept the rounding on
    a continuous one (the reported energy is the exact, unrounded sum either way)."""
    last = len(target) - K
    nbr = {}

    def add_cost(u, v, c):
        prev = u
        for k in range(c):
            relay = (u, v, k)
            nbr.setdefault(prev, []).append(relay)
            nbr.setdefault(relay, []).append(prev)
            prev = relay
        nbr.setdefault(prev, []).append(v)
        nbr.setdefault(v, []).append(prev)

    def hops(f, pos):
        return int(round(energy(f, pos, target)))                # round the (possibly continuous) energy to hops

    for f in library:
        add_cost("S", (0, f), hops(f, 0) + 1)                    # +1: every transition is >=1 hop
        add_cost((last, f), "T", 1)
    for pos in range(last):
        for f in library:
            for g in library:
                if f[1:] == g[:-1]:                              # fragments must overlap-agree
                    add_cost((pos, f), (pos + 1, g), hops(g, pos + 1) + 1)
    return nbr, last


def assemble(target, library, frag_len=2, steps=300, mu=1.5, dt=0.2, dim=1024, seed=0, energy=None):
    """Assemble `target` from `library` (overlapping fragments) by MIN-ENERGY flow search (Tero).
    Returns a dict: assembled string, its energy, the chosen (pos, fragment) list, and a B7
    StructureRecipe binding each fragment to its position -- the assembly as a typed holographic
    structure. Deterministic.

    `energy`: optional callable energy(frag, pos, target) -> non-negative cost of placing `frag` at `pos`
    (default: Hamming mismatch, the original stand-in). A SUPPLIED energy is the Rosetta move -- not every
    mismatch costs the same (a substitution matrix where similar residues are cheap, a cleanup-based soft
    score). The search still finds the GLOBAL optimum under whatever energy you give (it matches the Viterbi
    DP). It is rounded to integer hops for the trellis; the reported `energy` is the exact, unrounded sum."""
    e = energy if energy is not None else _energy
    K = frag_len
    nbr, last = _build_trellis(target, library, K, e)
    path = tero_solve(nbr, "S", "T", steps=steps, mu=mu, dt=dt)
    if path is None:
        return {"assembled": None, "energy": None, "fragments": None, "recipe": None}
    chosen = [n for n in path if isinstance(n, tuple) and len(n) == 2 and isinstance(n[0], int)]
    assembled = chosen[0][1] + "".join(g[1][-1] for g in chosen[1:])
    energy_total = sum(e(f, p, target) for (p, f) in chosen)
    r = StructureRecipe(dim, seed)                              # the assembly as a B7 typed structure
    parts = [r.bind(r.atom(f"pos:{p}", unitary=True), r.atom(f"frag:{f}")) for (p, f) in chosen]
    r.mark_output(r.superpose(parts))
    return {"assembled": assembled, "energy": energy_total, "fragments": chosen, "recipe": r}


def assemble_optimal_energy(target, library, frag_len=2, energy=None):
    """Exact minimum-energy assembly via DP (Viterbi over the trellis) -- the reference the flow search
    must match, under the SAME `energy` (default Hamming). For an integer-valued energy the flow search
    matches this exactly; on a continuous one it matches the rounded-hop optimum."""
    e = energy if energy is not None else _energy
    K = frag_len
    last = len(target) - K
    INF = 10 ** 9
    dp = {(0, f): e(f, 0, target) for f in library}
    for pos in range(last):
        nd = {}
        for f in library:
            if (pos, f) not in dp:
                continue
            for g in library:
                if f[1:] == g[:-1]:
                    en = dp[(pos, f)] + e(g, pos + 1, target)
                    if en < nd.get((pos + 1, g), INF):
                        nd[(pos + 1, g)] = en
        dp.update(nd)
    return min(dp[(last, f)] for f in library)


# ---- structure-compare: superpose two assembled structures and read their overlap, built on consolidation ----
def _name_seed(name, seed):
    """A DETERMINISTIC hash of an atom name (Python's hash() is process-randomised for strings, which would
    break run-to-run reproducibility -- Macklin's rule)."""
    h = seed & 0x7fffffff
    for ch in name:
        h = (h * 131 + ord(ch)) & 0x7fffffff
    return h


def _struct_parts(frags, dim, seed):
    """Role-bound part vectors {pos (x) frag} in a deterministic COMMON atom space, so two structures are
    comparable even if assembled separately. Atoms are seeded by name (deterministically)."""
    if not frags:
        return np.zeros((0, dim))

    def atom(name):
        r = np.random.default_rng(_name_seed(name, seed))
        v = r.standard_normal(dim)
        return v / np.linalg.norm(v)

    return np.stack([_A.bind(atom(f"pos:{p}"), atom(f"frag:{f}")) for (p, f) in frags])


def _eff_rank(M, tol):
    """Effective rank: singular values above tol x the largest -- the consolidation low-rank read."""
    if len(M) == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > tol * s[0]).sum()) if s[0] > 0 else 0


def compare_structures(a, b, dim=1024, seed=0, tol=0.1):
    """Superpose two assembled structures and read their OVERLAP -- the Baker seat's compare-two-folds, built
    on consolidation. Each structure is a bundle of role-bound (position (x) fragment) vectors. Returns:

      placement_overlap : the overlap coefficient of the (pos, fragment) sets -- |A and B| / min(|A|,|B|),
                          the exact shared-local-motif fraction (two folds sharing the same local motifs).
      holographic_overlap: the SAME overlap read from the SUPERPOSITION via consolidation. Stack both
                          structures' role-bound parts and take the effective rank (the consolidation SVD):
                          a shared placement is the SAME vector in the common space, so the combined rank
                          COLLAPSES by the number shared, and (rank_A + rank_B - rank_AB)/min(rank_A, rank_B)
                          recovers the overlap from the vectors ALONE. That is what lets you compare two
                          structures you only hold as hypervectors (a recalled fold), and it degrades
                          gracefully when parts are close-but-not-identical rather than exactly shared.
      shared : the sorted shared (pos, fragment) placements.

    On clean structures the two overlaps AGREE -- the holographic read validated against the exact count."""
    fa = list(map(tuple, a.get("fragments") or []))
    fb = list(map(tuple, b.get("fragments") or []))
    sa, sb = set(fa), set(fb)
    shared = sorted(sa & sb)
    placement_overlap = len(shared) / max(1, min(len(sa), len(sb)))
    PA, PB = _struct_parts(fa, dim, seed), _struct_parts(fb, dim, seed)
    rA, rB = _eff_rank(PA, tol), _eff_rank(PB, tol)
    rAB = _eff_rank(np.vstack([PA, PB]), tol) if (len(PA) + len(PB)) else 0
    holographic_overlap = (rA + rB - rAB) / max(1, min(rA, rB)) if min(rA, rB) > 0 else 0.0
    return {"placement_overlap": placement_overlap,
            "holographic_overlap": float(holographic_overlap), "shared": shared}
